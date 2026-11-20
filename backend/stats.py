"""Physical stats from pitch positions, and comparison with pro benchmarks.

Speed bands follow SkillCorner's physical definitions so the comparison is
like-for-like:
    running       15–20 km/h
    HSR           20–25 km/h  (high-speed running)
    sprinting     > 25 km/h
    HI            > 20 km/h   (HSR + sprinting)
An HSR / sprint "effort" counts when he stays in that band for at least 1 s.
"""

from pathlib import Path

import numpy as np
import pandas as pd

BENCHMARK_FILE = (Path(__file__).resolve().parent.parent / "data" / "benchmarks"
                  / "skillcorner_aleague_2024_25_physical_p90.csv")

# Our position names -> SkillCorner position groups
POSITION_GROUPS = {
    "Centre-back": "Central Defender",
    "Full-back / Wing-back": "Full Back",
    "Defensive midfielder": "Midfield",
    "Central midfielder": "Midfield",
    "Attacking midfielder": "Midfield",
    "Winger": "Wide Attacker",
    "Striker": "Center Forward",
}

# metric key -> (label, unit)
METRICS = {
    "total_distance_p90": ("Total distance", "m / 90"),
    "running_distance_p90": ("Running 15–20 km/h", "m / 90"),
    "hsr_distance_p90": ("High-speed running 20–25 km/h", "m / 90"),
    "sprint_distance_p90": ("Sprint distance >25 km/h", "m / 90"),
    "hi_distance_p90": ("High intensity >20 km/h", "m / 90"),
    "hsr_count_p90": ("HSR efforts", "per 90"),
    "sprint_count_p90": ("Sprints", "per 90"),
    "top_speed": ("Top speed (99th pct)", "km/h"),
}

MAX_HUMAN_SPEED = 11.5  # m/s (~41 km/h). Anything faster is a tracking glitch.


# ---------------------------------------------------------------- cleaning

def clean_trajectory(rows, fps, gap_s=0.5, smooth_s=0.5):
    """One player's pitch positions -> smoothed positions + speed.

    Splits into segments wherever he wasn't mapped for > gap_s (no distance is
    counted across a gap), removes spikes, resamples to 10 Hz, smooths, then
    computes speed. Returns a DataFrame with frame, t, x, y, speed (m/s), segment.
    """
    df = rows[["frame", "pitch_x", "pitch_y"]].dropna().sort_values("frame")
    if len(df) < 3:
        return pd.DataFrame(columns=["frame", "t", "x", "y", "speed", "segment"])
    df = df.assign(t=df.frame / fps)

    new_seg = df.t.diff().fillna(np.inf) > gap_s
    df["segment"] = new_seg.cumsum()

    from scipy.signal import savgol_filter

    hz = 10.0                                         # same rate as SkillCorner tracking
    win = max(5, int(round(smooth_s * hz * 2)) | 1)   # odd, ~1 s window
    parts = []
    for seg_id, seg in df.groupby("segment"):
        if seg.t.iloc[-1] - seg.t.iloc[0] < 1.0:
            continue  # too short to say anything about speed
        seg = seg.copy()
        # Median over ~0.3 s removes spikes from box wobble / bad mappings.
        k = max(3, int(round(0.3 * fps)) | 1)
        for c in ("pitch_x", "pitch_y"):
            seg[c] = seg[c].rolling(k, center=True, min_periods=1).median()
        # Resample to a steady 10
