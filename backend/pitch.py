"""Pitch mapping: turn on-screen pixel positions into metres on the pitch.

How it works
------------
A pretrained keypoint model (Roboflow "football-pitch-detection", YOLOv8-pose)
finds up to 32 known pitch landmarks in a frame: corners, box corners, penalty
spots, the centre circle... We know where each of those landmarks is on a real
pitch (in metres), so with enough of them visible we can compute a homography:
a 3x3 matrix that maps any point on the grass in the image to pitch metres.

Bad detections are filtered out (too few points, inconsistent points,
implausible camera view, jumps between frames) and the camera's movement is
smoothed over time.

Coordinates: origin at one corner flag, x along the length (0..length),
y across the width (0..width), in metres.

Model download: run `bash download_models.sh` (or see README).
"""

from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "football-pitch-detection.pt"

# Standard (FIFA) markings in metres. Only the overall length/width vary between pitches.
PENALTY_BOX_WIDTH = 40.32
PENALTY_BOX_LENGTH = 16.5
GOAL_BOX_WIDTH = 18.32
GOAL_BOX_LENGTH = 5.5
CENTRE_CIRCLE_RADIUS = 9.15
PENALTY_SPOT_DISTANCE = 11.0


def pitch_vertices(length=105.0, width=68.0):
    """The 32 landmarks, in the same order the keypoint model outputs them.

    Order follows roboflow/sports SoccerPitchConfiguration.vertices (MIT licence).
    """
    L, W = float(length), float(width)
    pb, pl = PENALTY_BOX_WIDTH, PENALTY_BOX_LENGTH
    gb, gl = GOAL_BOX_WIDTH, GOAL_BOX_LENGTH
    r, spot = CENTRE_CIRCLE_RADIUS, PENALTY_SPOT_DISTANCE
    return np.array([
        (0, 0),                      # 1  corner
        (0, (W - pb) / 2),           # 2  penalty box meets goal line
        (0, (W - gb) / 2),           # 3  goal box meets goal line
        (0, (W + gb) / 2),           # 4
        (0, (W + pb) / 2),           # 5
        (0, W),                      # 6  corner
        (gl, (W - gb) / 2),          # 7  goal box corner
        (gl, (W + gb) / 2),          # 8
        (spot, W / 2),               # 9  penalty spot
        (pl, (W - pb) / 2),          # 10 penalty box corner
        (pl, (W - gb) / 2),          # 11 box edge in line with goal box
        (pl, (W + gb) / 2),          # 12
        (pl, (W + pb) / 2),          # 13
        (L / 2, 0),                  # 14 halfway line meets touchline
        (L / 2, W / 2 - r),          # 15 centre circle on halfway line
        (L / 2, W / 2 + r),          # 16
        (L / 2, W),                  # 17
        (L - pl, (W - pb) / 2),      # 18
        (L - pl, (W - gb) / 2),      # 19
        (L - pl, (W + gb) / 2),      # 20
        (L - pl, (W + pb) / 2),      # 21
        (L - spot, W / 2),           # 22 penalty spot
        (L - gl, (W - gb) / 2),      # 23
        (L - gl, (W + gb) / 2),      # 24
        (L, 0),                      # 25 corner
        (L, (W - pb) / 2),           # 26
        (L, (W - gb) / 2),           # 27
        (L, (W + gb) / 2),           # 28
        (L, (W + pb) / 2),           # 29
        (L, W),                      # 30 corner
        (L / 2 - r, W / 2),          # 31 centre circle left
        (L / 2 + r, W / 2),          # 32 centre circle right
    ], dtype=np.float32)


class PitchDetector:
    """Wraps the keypoint model. detect(frame) -> (xy[32,2], conf[32]) or None."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        from ultralytics import YOLO
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pitch model not found at {model_path}. Run: bash download_models.sh")
        self.model = YOLO(str(model_path))

    def detect(self, frame_bgr):
        result = self.model(frame_bgr, verbose=False)[0]
        kp = result.keypoints
        if kp is None or kp.xy is None or len(kp.xy) == 0:
            return None
        # If it finds several "pitches", keep the most confident one.
        best = int(result.boxes.conf.argmax()) if result.boxes is not None and len(result.boxes) else 0
        xy = kp.xy[best].cpu().numpy().astype(np.float32)
        conf = (kp.conf[best].cpu().numpy() if kp.conf is not None
                else np.ones(len(xy))).astype(np.float32)
        return xy, conf


def homography_from_keypoints(xy, conf, length=105.0, width=68.0, image_size=None,
                              min_conf=0.5, min_points=5, ransac_m=2.0, max_rms_m=1.5,
                              min_spread_m=4.0, why=None):
    """Image -> pitch homography from one frame's keypoints, or None if unreliable.

    Checks, in order (the reason for a rejection is added to `why` if given):
      few_points   fewer than `min_points` confident landmarks
      collinear    landmarks bunched on (nearly) one line, e.g. only the halfway line
      no_fit       RANSAC couldn't find a consistent fit
      bad_fit      the landmarks it kept don't line up within `max_rms_m`
      implausible  the result isn't a camera looking at a pitch (flipped, absurd size)
    """
    def reject(reason):
        if why is not None:
            why[reason] = why.get(reason, 0) + 1
        return None

    xy, conf = np.asarray(xy, np.float32), np.asarray(conf, np.float32)
    mask = (conf >= min_conf) & (xy[:, 0] > 1) & (xy[:, 1] > 1)
    if mask.sum() < min_points:
        return reject("few_points")
    src = xy[mask]
    dst = pitch_vertices(length, width)[mask]

    # Spread: the weaker direction of the point cloud must still be several metres wide.
    centred = dst - dst.mean(axis=0)
    spread = np.linalg.svd(centred, compute_uv=False) / np.sqrt(len(dst))
    if spread.min() < min_spread_m:
        return reject("collinear")

    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, ransac_m)
    if H is None or inliers is None or inliers.sum() < min_points:
        return reject("no_fit")
    inl = inliers.ravel().astype(bool)
    err = np.linalg.norm(apply_homography(H, src[inl]) - dst[inl], axis=1)
    if np.sqrt(np.mean(err ** 2)) > max_rms_m:
        return reject("bad_fit")

    if image_size is not None and not _plausible(H, image_size, length, width):
        return reject("implausible")
    return H


def _plausible(H, image_size, length, width):
    """Does H look like a real camera view of the pitch?

    Map the lower two-thirds of the image (where the grass is) and check:
    no point maps "behind" the camera, the patch of grass it covers is a
    sensible size, and the centre of the view lands on or near the pitch.
    """
    w, h = image_size
    grid = np.array([(x, y) for x in np.linspace(0, w, 5) for y in np.linspace(h / 3, h, 4)],
                    np.float32)
    denom = (np.c_[grid, np.ones(len(grid))] @ H.T)[:, 2]
    if not (np.all(denom > 0) or np.all(denom < 0)):
        return False  # the view folds over itself
    pts = apply_homography(H, grid)
    corners = apply_homography(H, np.array([(0, h / 3), (w, h / 3), (w, h), (0, h)], np.float32))
    area = 0.5 * abs(np.dot(corners[:, 0], np.roll(corners[:, 1], 1))
                     - np.dot(corners[:, 1], np.roll(corners[:, 0], 1)))
    if not (30 <= area <= 4 * length * width):
        return False
    cx, cy = apply_homography(H, np.array([(w / 2, 2 * h / 3)], np.float32))[0]
    margin = 25
    if not (-margin <= cx <= length + margin and -margin <= cy <= width + margin):
        return False
    return np.isfinite(pts).all()


def apply_homography(H, points):
    pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def _consistent(Hs, image_size, window=7, max_jump_m=6.0):
    """Drop homographies that disagree with their neighbours in time.

    The camera moves smoothly, so where the centre of the view lands on the
    pitch should change smoothly too. A homography whose view-centre is far from
    the running median of its neighbours is a bad detection that slipped through.
    """
    if len(Hs) < 3:
        return Hs, 0
    w, h = image_size
    probe = np.array([(w / 2, 2 * h / 3), (w / 4, h), (3 * w / 4, h)], np.float32)
    frames = sorted(Hs)
    centres = np.array([apply_homography(Hs[f], probe).ravel() for f in frames])
    keep, dropped = {}, 0
    half = window // 2
    for i, f in enumerate(frames):
        lo, hi = max(0, i - half), min(len(frames), i + half + 1)
        med = np.median(centres[lo:hi], axis=0)
        dev = np.max(np.linalg.norm((centres[i] - med).reshape(-1, 2), axis=1))
        if dev <= max_jump_m:
            keep[f] = Hs[f]
        else:
            dropped += 1
    return keep, dropped


def _smooth(Hs, image_size, window=9):
    """Smooth the camera's movement over time.

    Each homography says where 4 fixed spots of the image land on the pitch.
    A real camera pans smoothly, so we smooth those landing spots over
    neighbouring frames (median, then mean) and rebuild each homography from
    the smoothed spots. This removes frame-to-frame wobble from noisy landmarks.
    """
    if len(Hs) < 5:
        return Hs
    w, h = image_size
    quad = np.array([(0.1 * w, 0.45 * h), (0.9 * w, 0.45 * h),
                     (0.9 * w, 0.95 * h), (0.1 * w, 0.95 * h)], np.float32)
    frames = sorted(Hs)
    spots = np.array([apply_homography(Hs[f], quad).ravel() for f in frames])
    import pandas as pd
    df = pd.DataFrame(spots)
    df = (df.rolling(window, center=True, min_periods=1).median()
            .rolling(window, center=True, min_periods=1).mean())
    out = {}
    for f, row in zip(frames, df.to_numpy(np.float32)):
        H = cv2.getPerspectiveTransform(quad, row.reshape(4, 2))
        out[f] = H if np.isfinite(H).all() else Hs[f]
    return out


def build_homographies(keypoints, length=105.0, width=68.0, image_size=None):
    """All usable homographies for a clip, plus a quality report."""
    why, Hs = {}, {}
    for f, (xy, conf) in keypoints.items():
        H = homography_from_keypoints(xy, conf, length, width, image_size=image_size, why=why)
        if H is not None:
            Hs[int(f)] = H
    jumped = 0
    if image_size is not None:
        Hs, jumped = _consistent(Hs, image_size, max_jump_m=8.0)
        if jumped:
            why["jumped"] = jumped
        Hs = _smooth(Hs, image_size)
    n_points = [int(((c >= 0.5) & (xy[:, 0] > 1)).sum()) for xy, c in keypoints.values()]
    report = {
        "checked": len(keypoints),
        "usable": len(Hs),
        "usable_share": len(Hs) / len(keypoints) if keypoints else 0.0,
        "median_landmarks": float(np.median(n_points)) if n_points else 0.0,
        "rejected": why,
    }
    return Hs, report


def map_tracks_to_pitch(tracks, keypoints, length=105.0, width=68.0, image_size=None,
                        max_gap_frames=30, margin_m=5.0):
    """Add pitch_x / pitch_y (metres) to every tracked box.

    keypoints: {frame_index: (xy, conf)} from the pitch detector (every few frames).
    For each frame we use the nearest good homographies before and after it
    (within max_gap_frames) and blend the mapped positions by time.
    Positions more than margin_m outside the pitch are dropped as bad mappings.

    Returns (tracks_with_pitch_xy, quality_report).
    """
    out = tracks.copy()
    out["pitch_x"] = np.nan
    out["pitch_y"] = np.nan
    Hs, report = build_homographies(keypoints or {}, length, width, image_size)
    report["mapped_share"] = 0.0
    if out.empty or not Hs:
        return out, report

    good = np.array(sorted(Hs))
    feet = out[["foot_x", "foot_y"]].to_numpy(np.float32)
    px = np.full(len(out), np.nan)
    py = np.full(len(out), np.nan)

    for frame, idx in out.groupby("frame").indices.items():
        pos = np.searchsorted(good, frame)
        before = good[pos - 1] if pos > 0 else None
        after = good[pos] if pos < len(good) else None
        if after is not None and after == frame:
            before = after
        if before is not None and frame - before > max_gap_frames:
            before = None
        if after is not None and after - frame > max_gap_frames:
            after = None
        if before is None and after is None:
            continue
        if before is None or after is None or before == after:
            mapped = apply_homography(Hs[before if before is not None else after], feet[idx])
        else:
            w = (frame - before) / (after - before)
            mapped = ((1 - w) * apply_homography(Hs[before], feet[idx])
                      + w * apply_homography(Hs[after], feet[idx]))
        px[idx], py[idx] = mapped[:, 0], mapped[:, 1]

    outside = (px < -margin_m) | (px > length + margin_m) | (py < -margin_m) | (py > width + margin_m)
    px[outside], py[outside] = np.nan, np.nan
    out["pitch_x"], out["pitch_y"] = px, py
    report["mapped_share"] = float(np.isfinite(px).mean())
    return out, report


def nearest_homography(Hs, frame, max_gap_frames=30):
    if not Hs:
        return None
    f = min(Hs, key=lambda k: abs(k - frame))
    return Hs[f] if abs(f - frame) <= max_gap_frames else None


def pitch_lines(length=105.0, width=68.0, step=1.0):
    """Pitch markings as lists of points in metres (for drawing)."""
    L, W = length, width
    def seg(a, b):
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / step))
        return np.linspace(a, b, n)
    def rect(x0, y0, x1, y1):
        return [seg((x0, y0), (x1, y0)), seg((x1, y0), (x1, y1)),
                seg((x1, y1), (x0, y1)), seg((x0, y1), (x0, y0))]
    lines = rect(0, 0, L, W) + [seg((L / 2, 0), (L / 2, W))]
    for x0, sgn in ((0, 1), (L, -1)):
        lines += rect(x0, (W - PENALTY_BOX_WIDTH) / 2, x0 + sgn * PENALTY_BOX_LENGTH,
                      (W + PENALTY_BOX_WIDTH) / 2)
        lines += rect(x0, (W - GOAL_BOX_WIDTH) / 2, x0 + sgn * GOAL_BOX_LENGTH,
                      (W + GOAL_BOX_WIDTH) / 2)
    a = np.linspace(0, 2 * np.pi, 80)
    lines.append(np.c_[L / 2 + CENTRE_CIRCLE_RADIUS * np.cos(a), W / 2 + CENTRE_CIRCLE_RADIUS * np.sin(a)])
    return lines


def draw_projected_pitch(frame_bgr, H, length=105.0, width=68.0, color=(255, 255, 0)):
    """Draw the pitch lines where homography H says they are. If H is right,
    they sit on top of the real white lines."""
    img = frame_bgr.copy()
    Hinv = np.linalg.inv(H)
    h, w = img.shape[:2]
    for line in pitch_lines(length, width):
        p = np.c_[line, np.ones(len(line))] @ Hinv.T
        ok = p[:, 2] > 1e-6
        if (H[2] @ np.array([w / 2, h, 1])) < 0:  # sign convention depends on H scale
            ok = p[:, 2] < -1e-6
        pts = p[:, :2] / p[:, 2:]
        pts[~ok] = np.nan
        run = []
        for q, good in zip(pts, ok & np.all(np.abs(pts) < 1e5, axis=1)):
            if good:
                run.append(q)
            elif len(run) > 1:
                cv2.polylines(img, [np.int32(run)], False, color, 2, cv2.LINE_AA)
                run = []
            else:
                run = []
        if len(run) > 1:
            cv2.polylines(img, [np.int32(run)], False, color, 2, cv2.LINE_AA)
    return img


def keypoint_coverage(keypoints, length=105.0, width=68.0, image_size=None):
    """Share of sampled frames where a usable homography was found."""
    return build_homographies(keypoints, length, width, image_size)[1]["usable_share"]
