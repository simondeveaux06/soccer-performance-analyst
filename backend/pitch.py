"""Pitch mapping: turn on-screen pixel positions into metres on the pitch.

How it works
------------
A pretrained keypoint model (Roboflow "football-pitch-detection", YOLOv8-pose)
finds up to 32 known pitch landmarks in a frame: corners, box corners, penalty
spots, the centre circle... We know where each of those landmarks is on a real
pitch (in metres), so with 4+ of them visible we can compute a homography: a
3x3 matrix that maps any point on the grass in the image to pitch metres.

The camera pans, so we redo this every few frames and blend between them.

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


def homography_from_keypoints(xy, conf, length=105.0, width=68.0,
                              min_conf=0.5, min_points=4, ransac_m=1.0):
    """Image -> pitch homography from one frame's keypoints, or None if unreliable."""
    xy, conf = np.asarray(xy, np.float32), np.asarray(conf, np.float32)
    mask = (conf >= min_conf) & (xy[:, 0] > 1) & (xy[:, 1] > 1)
    if mask.sum() < min_points:
        return None
    src = xy[mask]
    dst = pitch_vertices(length, width)[mask]
    # All points on one line (e.g. only the halfway line) can't fix a homography.
    if np.linalg.matrix_rank(np.c_[dst, np.ones(len(dst))]) < 3:
        return None
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, ransac_m)
    if H is None or inliers is None or inliers.sum() < min_points:
        return None
    return H


def apply_homography(H, points):
    pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def map_tracks_to_pitch(tracks, keypoints, length=105.0, width=68.0,
                        max_gap_frames=30, margin_m=5.0):
    """Add pitch_x / pitch_y (metres) to every tracked box.

    keypoints: {frame_index: (xy, conf)} from the pitch detector (every few frames).
    For each frame we use the nearest good homographies before and after it
    (within max_gap_frames) and blend the mapped positions by time.
    Positions more than margin_m outside the pitch are dropped as bad mappings.
    """
    out = tracks.copy()
    out["pitch_x"] = np.nan
    out["pitch_y"] = np.nan
    if out.empty or not keypoints:
        return out

    Hs = {}
    for f, (xy, conf) in keypoints.items():
        H = homography_from_keypoints(xy, conf, length, width)
        if H is not None:
            Hs[int(f)] = H
    if not Hs:
        return out

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
    return out


def keypoint_coverage(keypoints, length=105.0, width=68.0):
    """Share of sampled frames where a usable homography was found."""
    if not keypoints:
        return 0.0
    ok = sum(homography_from_keypoints(xy, c, length, width) is not None
             for xy, c in keypoints.values())
    return ok / len(keypoints)
