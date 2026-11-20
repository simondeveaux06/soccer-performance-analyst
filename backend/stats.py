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
        # Resample to a steady 10 Hz.
        grid = np.arange(seg.t.iloc[0], seg.t.iloc[-1], 1 / hz)
        x = np.interp(grid, seg.t, seg.pitch_x)
        y = np.interp(grid, seg.t, seg.pitch_y)
        w = min(win, len(grid) - (1 - len(grid) % 2))
        if w < 5:
            continue
        # Savitzky–Golay: smooth positions and get velocity from the fitted curve.
        xs, ys = savgol_filter(x, w, 2), savgol_filter(y, w, 2)
        vx = savgol_filter(x, w, 2, deriv=1, delta=1 / hz)
        vy = savgol_filter(y, w, 2, deriv=1, delta=1 / hz)
        speed = np.hypot(vx, vy)
        speed = np.where(speed > MAX_HUMAN_SPEED, np.nan, speed)
        speed = pd.Series(speed).interpolate(limit_direction="both").fillna(0).to_numpy()
        parts.append(pd.DataFrame({"frame": np.round(grid * fps).astype(int), "t": grid,
                                   "x": xs, "y": ys, "speed": speed, "segment": seg_id}))

    if not parts:
        return pd.DataFrame(columns=["frame", "t", "x", "y", "speed", "segment"])
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- stats

def _efforts(speed_kmh, dt, lo, hi, min_s=1.0):
    """Count runs where speed stays within [lo, hi) for at least min_s seconds."""
    inside = (speed_kmh >= lo) & (speed_kmh < hi)
    count, run = 0, 0.0
    for flag, d in zip(inside, dt):
        if flag:
            run += d
        else:
            if run >= min_s:
                count += 1
            run = 0.0
    if run >= min_s:
        count += 1
    return count


def physical_stats(traj):
    """Raw totals over the time he was tracked on the pitch, plus per-90 rates."""
    if traj.empty:
        return None
    stats = {}
    seg_stats = []
    for _, seg in traj.groupby("segment"):
        dt = seg.t.diff().fillna(0).to_numpy()
        v = seg.speed.to_numpy()
        kmh = v * 3.6
        dist = v * dt
        seg_stats.append({
            "time_s": dt.sum(),
            "total": dist.sum(),
            "running": dist[(kmh >= 15) & (kmh < 20)].sum(),
            "hsr": dist[(kmh >= 20) & (kmh < 25)].sum(),
            "sprint": dist[kmh >= 25].sum(),
            "hsr_count": _efforts(kmh, dt, 20, 25),
            "sprint_count": _efforts(kmh, dt, 25, 99),
        })
    s = pd.DataFrame(seg_stats).sum()
    minutes = s.time_s / 60
    stats["minutes_tracked"] = minutes
    stats["total_distance"] = s.total
    stats["running_distance"] = s.running
    stats["hsr_distance"] = s.hsr
    stats["sprint_distance"] = s.sprint
    stats["hi_distance"] = s.hsr + s.sprint
    stats["hsr_count"] = s.hsr_count
    stats["sprint_count"] = s.sprint_count
    stats["top_speed"] = float(np.percentile(traj.speed * 3.6, 99)) if len(traj) else 0.0
    stats["max_speed"] = float(traj.speed.max() * 3.6)

    per90 = 90 / minutes if minutes > 0 else np.nan
    for k in ("total_distance", "running_distance", "hsr_distance", "sprint_distance",
              "hi_distance", "hsr_count", "sprint_count"):
        stats[f"{k}_p90"] = stats[k] * per90
    return stats


# ---------------------------------------------------------------- pro comparison

def load_benchmarks(path=BENCHMARK_FILE):
    return pd.read_csv(path) if Path(path).exists() else None


def compare_to_pros(stats, position, benchmarks):
    """Percentile of each metric vs pros in that position group (0–100)."""
    group = POSITION_GROUPS.get(position)
    if stats is None or benchmarks is None or group is None:
        return None, group
    pros = benchmarks[benchmarks.position_group == group]
    rows = []
    for key, (label, unit) in METRICS.items():
        if key not in pros or key not in stats:
            continue
        vals = pros[key].dropna().to_numpy()
        mine = stats[key]
        pct = 100 * (vals < mine).mean() + 50 * (vals == mine).mean()
        rows.append({"metric": label, "unit": unit, "him": mine,
                     "pro_median": float(np.median(vals)),
                     "pro_p25": float(np.percentile(vals, 25)),
                     "pro_p75": float(np.percentile(vals, 75)),
                     "percentile": pct})
    return pd.DataFrame(rows), group


# ---------------------------------------------------------------- drawing

def draw_pitch(ax, length=105.0, width=68.0, line="white", grass="#3a7d3a"):
    """Draw a pitch in metres (origin bottom-left corner)."""
    from matplotlib.patches import Arc, Circle, Rectangle

    from pitch import (CENTRE_CIRCLE_RADIUS, GOAL_BOX_LENGTH, GOAL_BOX_WIDTH,
                       PENALTY_BOX_LENGTH, PENALTY_BOX_WIDTH, PENALTY_SPOT_DISTANCE)
    L, W = length, width
    ax.set_facecolor(grass)
    kw = dict(fill=False, ec=line, lw=1.5, zorder=3)
    ax.add_patch(Rectangle((0, 0), L, W, **kw))
    ax.plot([L / 2, L / 2], [0, W], color=line, lw=1.5, zorder=3)
    ax.add_patch(Circle((L / 2, W / 2), CENTRE_CIRCLE_RADIUS, **kw))
    for x0, sign in ((0, 1), (L, -1)):
        ax.add_patch(Rectangle((x0 if sign > 0 else x0 - PENALTY_BOX_LENGTH, (W - PENALTY_BOX_WIDTH) / 2),
                               PENALTY_BOX_LENGTH, PENALTY_BOX_WIDTH, **kw))
        ax.add_patch(Rectangle((x0 if sign > 0 else x0 - GOAL_BOX_LENGTH, (W - GOAL_BOX_WIDTH) / 2),
                               GOAL_BOX_LENGTH, GOAL_BOX_WIDTH, **kw))
        spot = x0 + sign * PENALTY_SPOT_DISTANCE
        ax.scatter([spot], [W / 2], color=line, s=8, zorder=3)
        ax.add_patch(Arc((spot, W / 2), 2 * CENTRE_CIRCLE_RADIUS, 2 * CENTRE_CIRCLE_RADIUS,
                         theta1=-53 if sign > 0 else 127, theta2=53 if sign > 0 else 233,
                         ec=line, lw=1.5, zorder=3))
    ax.scatter([L / 2], [W / 2], color=line, s=8, zorder=3)
    ax.set_xlim(-3, L + 3)
    ax.set_ylim(-3, W + 3)
    ax.set_aspect("equal")
    ax.axis("off")


def pitch_heatmap(traj, length=105.0, width=68.0, flip=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    fig, ax = plt.subplots(figsize=(10, 10 * (width + 6) / (length + 6)))
    fig.patch.set_facecolor("#3a7d3a")
    draw_pitch(ax, length, width)
    if len(traj):
        x, y = traj.x.to_numpy(), traj.y.to_numpy()
        if flip:
            x, y = length - x, width - y
        grid, xe, ye = np.histogram2d(x, y, bins=[int(length), int(width)],
                                      range=[[0, length], [0, width]])
        grid = gaussian_filter(grid, 3)
        grid = grid / grid.max() if grid.max() > 0 else grid
        ax.imshow(grid.T, origin="lower", extent=[0, length, 0, width], cmap="magma",
                  alpha=np.clip(grid.T * 1.6, 0, 0.85), zorder=2, interpolation="bilinear")
        ax.scatter([x.mean()], [y.mean()], s=160, color="white", ec="black", zorder=4,
                   label="Average position")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=False,
                  labelcolor="white", fontsize=10)
    fig.tight_layout(pad=0.3)
    return fig


def _ordinal(p):
    n = int(round(p))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def percentile_chart(comparison, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = comparison.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(c) + 1))
    colors = ["#d9534f" if p < 33 else "#f0ad4e" if p < 66 else "#5cb85c" for p in c.percentile]
    ax.barh(c.metric, c.percentile, color=colors)
    ax.axvline(50, color="grey", ls="--", lw=1)
    for i, (p, him, med, unit) in enumerate(zip(c.percentile, c.him, c.pro_median, c.unit)):
        fmt = "{:.1f}" if unit == "km/h" or him < 50 else "{:.0f}"
        ax.text(p + 1.5, i, f"{_ordinal(p)}  ({fmt.format(him)} vs pro {fmt.format(med)})",
                va="center", fontsize=9)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Percentile vs pros (50 = typical pro)")
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
