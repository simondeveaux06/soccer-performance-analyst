"""Core analysis functions used by the Streamlit app (app.py).

Flow:
    1. track_players()  -> runs YOLO + ByteTrack over the whole video, returns every box
    2. pick_track()     -> turns a click on a frame into a track ID
    3. auto_stitch()    -> guesses which later IDs are the same player after he was lost
    4. player_stats() / heatmap_figure() / render_highlight() -> results
"""

from math import hypot
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MODEL_NAME = "yolo11n.pt"

COLUMNS = ["frame", "track_id", "x1", "y1", "x2", "y2", "confidence"]


# ---------------------------------------------------------------- video helpers

def video_info(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    info = {
        "fps": fps,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    info["duration"] = info["frames"] / fps if fps else 0
    return info


def read_frame(path, index):
    """Return one frame (BGR) by its 0-based index, or None."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ---------------------------------------------------------------- tracking

def track_players(path, conf=0.30, progress=None, model_name=MODEL_NAME):
    """Track every person in the video. Returns a DataFrame with one row per box.

    foot_x / foot_y is the bottom-centre of the box (where the player touches
    the ground). That's the point we'll map onto the pitch later.
    """
    from ultralytics import YOLO  # imported here so the app starts fast

    model = YOLO(model_name)
    total = video_info(path)["frames"]
    rows = []

    results = model.track(
        source=str(path),
        tracker="bytetrack.yaml",
        persist=True,
        conf=conf,
        classes=[0],  # 0 = person
        stream=True,
        verbose=False,
    )

    for frame_index, result in enumerate(results):
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for (x1, y1, x2, y2), tid, c in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.id.int().cpu().tolist(),
                boxes.conf.cpu().tolist(),
            ):
                rows.append((frame_index, tid, float(x1), float(y1), float(x2), float(y2), float(c)))
        if progress and total:
            progress(min((frame_index + 1) / total, 1.0))

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["foot_x"] = (df.x1 + df.x2) / 2
    df["foot_y"] = df.y2
    return df


# ---------------------------------------------------------------- picking the player

def pick_track(tracks, frame, x, y):
    """Return the track ID of the box at (x, y) on this frame, or None.

    If the click lands inside several boxes (players overlapping), the smallest
    box wins. If it misses every box, the nearest player within 60 px wins.
    """
    here = tracks[tracks.frame == frame]
    if here.empty:
        return None

    inside = here[(here.x1 <= x) & (x <= here.x2) & (here.y1 <= y) & (y <= here.y2)]
    if not inside.empty:
        area = (inside.x2 - inside.x1) * (inside.y2 - inside.y1)
        return int(inside.loc[area.idxmin(), "track_id"])

    cx = (here.x1 + here.x2) / 2
    cy = (here.y1 + here.y2) / 2
    dist = np.hypot(cx - x, cy - y)
    if dist.min() <= 60:
        return int(here.loc[dist.idxmin(), "track_id"])
    return None


def auto_stitch(tracks, selected_ids, max_gap_frames=60, max_distance_px=150):
    """Guess which new track IDs belong to the selected player.

    When the tracker loses a player and finds him again, he gets a new ID. For
    each selected track that ends, look for a track that starts shortly after,
    close to where he was last seen, and never overlaps in time with him.
    This is a guess: always check the result on the frames.
    """
    ids = set(int(i) for i in selected_ids)
    if not ids:
        return ids

    spans = tracks.groupby("track_id")["frame"].agg(["min", "max"])
    first_rows = tracks.loc[tracks.groupby("track_id")["frame"].idxmin()].set_index("track_id")
    last_rows = tracks.loc[tracks.groupby("track_id")["frame"].idxmax()].set_index("track_id")

    changed = True
    while changed:
        changed = False
        selected_frames = set(tracks.loc[tracks.track_id.isin(ids), "frame"])

        for tid in sorted(ids, key=lambda t: spans.loc[t, "max"]):
            end = spans.loc[tid, "max"]
            # Someone selected already covers the frames right after this track ends.
            if any(f in selected_frames for f in range(end + 1, end + 1 + max_gap_frames)):
                continue

            last = last_rows.loc[tid]
            candidates = spans[(spans["min"] > end) & (spans["min"] <= end + max_gap_frames)].index
            best = None
            for cand in candidates:
                if cand in ids:
                    continue
                cand_frames = set(tracks.loc[tracks.track_id == cand, "frame"])
                if cand_frames & selected_frames:
                    continue  # he can't be two people at once
                first = first_rows.loc[cand]
                d = hypot(first.foot_x - last.foot_x, first.foot_y - last.foot_y)
                if d <= max_distance_px and (best is None or d < best[0]):
                    best = (d, int(cand))
            if best:
                ids.add(best[1])
                changed = True
                break
    return ids


def player_rows(tracks, selected_ids):
    """One row per frame for the chosen player (if two IDs overlap, keep the more confident)."""
    rows = tracks[tracks.track_id.isin(selected_ids)]
    rows = rows.sort_values("confidence", ascending=False).drop_duplicates("frame")
    return rows.sort_values("frame").reset_index(drop=True)


def missing_stretches(rows, total_frames, min_length=15):
    """Stretches of frames (start, end) where the player isn't tracked."""
    seen = np.zeros(total_frames, dtype=bool)
    seen[rows.frame[rows.frame < total_frames].to_numpy()] = True
    stretches, start = [], None
    for i, s in enumerate(seen):
        if not s and start is None:
            start = i
        elif s and start is not None:
            if i - start >= min_length:
                stretches.append((start, i - 1))
            start = None
    if start is not None and total_frames - start >= min_length:
        stretches.append((start, total_frames - 1))
    return stretches


# ---------------------------------------------------------------- results

def player_stats(rows, info):
    fps = info["fps"]
    visible = len(rows)
    return {
        "Time tracked": f"{visible / fps:.0f} s",
        "Share of clip": f"{100 * visible / max(info['frames'], 1):.0f}%",
        "Track IDs used": rows.track_id.nunique(),
        "Average confidence": f"{rows.confidence.mean():.2f}" if visible else "-",
    }


def heatmap_figure(rows, background_bgr, bins=48):
    """Heatmap of where his feet were, drawn over a video frame.

    In pixels for now: with a moving camera this shows where he was on screen,
    not on the pitch. Pitch mapping fixes that.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = background_bgr.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 10 * h / w))
    ax.imshow(cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB))
    if len(rows):
        grid, _, _ = np.histogram2d(
            rows.foot_y.clip(0, h - 1), rows.foot_x.clip(0, w - 1),
            bins=[int(bins * h / w), bins], range=[[0, h], [0, w]],
        )
        grid = cv2.GaussianBlur(grid.astype(np.float32), (0, 0), 2.5)
        grid = np.ma.masked_where(grid < grid.max() * 0.05, grid)
        ax.imshow(grid, extent=[0, w, h, 0], cmap="inferno", alpha=0.6, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def draw_boxes(frame_bgr, frame_tracks, selected_ids):
    """All boxes with their IDs; the selected player is thick and green."""
    img = frame_bgr.copy()
    for r in frame_tracks.itertuples():
        chosen = r.track_id in selected_ids
        color = (0, 220, 0) if chosen else (0, 165, 255)
        thick = 4 if chosen else 2
        p1, p2 = (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2))
        cv2.rectangle(img, p1, p2, color, thick)
        cv2.putText(img, str(r.track_id), (p1[0], max(p1[1] - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if chosen else 1)
    return img


def render_highlight(path, rows, out_path, progress=None):
    """Write a copy of the video with only the chosen player boxed."""
    info = video_info(path)
    by_frame = rows.set_index("frame")
    cap = cv2.VideoCapture(str(path))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             info["fps"], (info["width"], info["height"]))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in by_frame.index:
            r = by_frame.loc[i]
            cv2.rectangle(frame, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 220, 0), 3)
            cv2.ellipse(frame, (int(r.foot_x), int(r.foot_y)), (25, 8), 0, 0, 360, (0, 220, 0), 2)
        writer.write(frame)
        i += 1
        if progress and info["frames"]:
            progress(min(i / info["frames"], 1.0))
    cap.release()
    writer.release()
    return Path(out_path)
