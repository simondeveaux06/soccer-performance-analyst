"""Saving and loading tracking runs, so a match only has to be tracked once.

A run is a folder (default: runs/<video name>/) containing:
    meta.json            video path, fps, size, settings, status ("partial" / "done"),
                         next_frame (where to resume)
    tracks_000.parquet   tracked boxes, saved in chunks as tracking goes
    keypoints_000.npz    pitch landmarks for the same stretch
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def run_dir_for(video_path, runs_dir=RUNS_DIR):
    return Path(runs_dir) / Path(video_path).stem


def read_meta(run_dir):
    p = Path(run_dir) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def write_meta(run_dir, meta):
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    meta["updated"] = time.strftime("%Y-%m-%d %H:%M")
    tmp = Path(run_dir) / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2))
    tmp.replace(Path(run_dir) / "meta.json")  # atomic: never a half-written file


def save_chunk(run_dir, tracks, keypoints):
    """Append one chunk of tracks + keypoints to the run folder."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(run_dir.glob("tracks_*.parquet")))
    tracks.to_parquet(run_dir / f"tracks_{n:03d}.parquet", index=False)
    frames = np.array(sorted(keypoints), dtype=np.int64)
    if len(frames):
        xy = np.stack([np.asarray(keypoints[f][0], np.float32) for f in frames])
        conf = np.stack([np.asarray(keypoints[f][1], np.float32) for f in frames])
    else:
        xy, conf = np.zeros((0, 32, 2), np.float32), np.zeros((0, 32), np.float32)
    np.savez_compressed(run_dir / f"keypoints_{n:03d}.npz", frames=frames, xy=xy, conf=conf)


def load_run(run_dir):
    """Returns (tracks DataFrame, keypoints dict, meta dict)."""
    run_dir = Path(run_dir)
    meta = read_meta(run_dir)
    parts = [pd.read_parquet(p) for p in sorted(run_dir.glob("tracks_*.parquet"))]
    tracks = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(tracks):
        tracks = tracks.drop_duplicates(["frame", "track_id"]).sort_values("frame",
                                                                          ignore_index=True)
    keypoints = {}
    for p in sorted(run_dir.glob("keypoints_*.npz")):
        z = np.load(p)
        for f, xy, c in zip(z["frames"], z["xy"], z["conf"]):
            keypoints[int(f)] = (xy, c)
    return tracks, keypoints, meta


def clear_run(run_dir):
    for p in list(Path(run_dir).glob("tracks_*.parquet")) + list(Path(run_dir).glob("keypoints_*.npz")):
        p.unlink()


def list_runs(runs_dir=RUNS_DIR):
    """All saved runs, newest first: [(run_dir, meta), ...]."""
    out = []
    for m in Path(runs_dir).glob("*/meta.json"):
        meta = read_meta(m.parent)
        if meta:
            out.append((m.parent, meta))
    return sorted(out, key=lambda rm: rm[1].get("updated", ""), reverse=True)
