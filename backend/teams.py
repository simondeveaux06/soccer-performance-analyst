"""Team detection from shirt colour.

1. shirt_colour(): during tracking, every few frames, take the torso part of
   each player's box, throw away grass-green pixels, and keep the median
   colour (in Lab colour space, which matches how humans see colour).
2. assign_teams(): combine each track's samples into one colour, cluster all
   tracks into groups, and call the two biggest groups the teams. Smaller
   groups and odd colours become "Other" (referees, goalkeepers).
3. People who spend most of their time off the pitch (subs, coaches, fans)
   become "Off pitch", if pitch mapping is available.
"""

import cv2
import numpy as np
import pandas as pd

TEAM_A, TEAM_B, OTHER, OFF_PITCH, UNKNOWN = "Team A", "Team B", "Other", "Off pitch", "Unknown"


def shirt_colour(frame_bgr, x1, y1, x2, y2, min_pixels=30):
    """Median Lab colour of the shirt area of a player box, or None.

    Torso = 15-50% down the box, middle 60% across. Grass and very dark shadow
    pixels are ignored. (Bright pixels are kept: a white kit in sun is that bright.)
    """
    h, w = frame_bgr.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if bw < 6 or bh < 15:
        return None
    tx1, tx2 = int(max(0, x1 + 0.2 * bw)), int(min(w, x2 - 0.2 * bw))
    ty1, ty2 = int(max(0, y1 + 0.15 * bh)), int(min(h, y1 + 0.5 * bh))
    crop = frame_bgr[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    grass = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 90) & (hsv[:, 1] >= 50) & (hsv[:, 2] >= 40)
    keep = ~grass & (hsv[:, 2] >= 25)
    if keep.sum() < min_pixels:
        return None
    return np.median(lab[keep], axis=0)


def _kmeans(X, k, weights=None, restarts=8, iters=50, seed=0):
    """Small weighted k-means (k-means++ init, best of several restarts)."""
    rng = np.random.default_rng(seed)
    n = len(X)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    best = None
    for _ in range(restarts):
        centres = [X[rng.choice(n, p=w / w.sum())]]
        for _ in range(1, k):
            d = np.min([((X - c) ** 2).sum(1) for c in centres], axis=0) * w
            centres.append(X[rng.choice(n, p=d / d.sum())] if d.sum() > 0 else X[rng.integers(n)])
        C = np.array(centres)
        for _ in range(iters):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(2), 1)
            newC = np.array([np.average(X[lab == j], axis=0, weights=w[lab == j])
                             if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        cost = (w * ((X - C[lab]) ** 2).sum(1)).sum()
        if best is None or cost < best[0]:
            best = (cost, C, lab)
    return best[1], best[2]


def _feature(lab):
    """Lab -> clustering feature. Lightness counts a bit less than colour
    (shadows change lightness), but enough to split black from white kits."""
    lab = np.atleast_2d(lab).astype(np.float32)
    return np.c_[lab[:, 0] * 0.6, lab[:, 1], lab[:, 2]]


def lab_to_bgr(lab):
    px = np.uint8(np.clip(np.round(lab), 0, 255)).reshape(1, 1, 3)
    return tuple(int(v) for v in cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0])


def assign_teams(tracks, min_samples=3, pitch_xy=None, length=105.0, width=68.0):
    """Label every track. Returns (labels, info).

    labels: {track_id: "Team A" | "Team B" | "Other" | "Off pitch" | "Unknown"}
    info:   {"colours": {label: (b, g, r)}, "counts": {label: n_tracks}}

    pitch_xy: optional DataFrame with track_id, pitch_x, pitch_y (from pitch
    mapping). Tracks that are outside the lines most of the time -> Off pitch.
    """
    labels, info = {}, {"colours": {}, "counts": {}}
    if tracks.empty or "shirt_l" not in tracks:
        return labels, info

    samples = tracks.dropna(subset=["shirt_l"])
    per = samples.groupby("track_id")[["shirt_l", "shirt_a", "shirt_b"]].median()
    n_samples = samples.groupby("track_id").size()
    length_by_track = tracks.groupby("track_id").size()

    # Off pitch first: they shouldn't pull the team colours around.
    off = set()
    if pitch_xy is not None and len(pitch_xy):
        p = pitch_xy.dropna(subset=["pitch_x"])
        inside = ((p.pitch_x >= -1) & (p.pitch_x <= length + 1) &
                  (p.pitch_y >= -1) & (p.pitch_y <= width + 1))
        share_inside = inside.groupby(p.track_id).mean()
        mapped = p.groupby("track_id").size()
        off = set(share_inside[(share_inside < 0.3) & (mapped >= 10)].index)

    good = [t for t in per.index if n_samples[t] >= min_samples and t not in off]
    for t in tracks.track_id.unique():
        labels[int(t)] = OFF_PITCH if t in off else UNKNOWN
    if len(good) < 4:
        return labels, info

    X = _feature(per.loc[good].to_numpy())
    wts = length_by_track.loc[good].to_numpy(float)
    k = 3 if len(good) >= 8 else 2
    C, lab = _kmeans(X, k, weights=wts)

    # The two clusters with the most player-time are the teams.
    mass = np.array([wts[lab == j].sum() for j in range(k)])
    order = np.argsort(-mass)
    team_of = {order[0]: TEAM_A, order[1]: TEAM_B}

    # How spread each team's colours are, to spot odd ones out (keepers, refs).
    d_all = np.sqrt(((X[:, None] - C[None]) ** 2).sum(2))
    spread = {j: np.median(d_all[lab == j, j]) if (lab == j).any() else 1.0 for j in order[:2]}

    for t, j, d in zip(good, lab, d_all):
        if j in team_of and d[j] <= max(3.0 * spread[j], 12.0):
            labels[int(t)] = team_of[j]
        else:
            labels[int(t)] = OTHER

    # Short tracks with a few samples: give them the nearest team if clearly close.
    for t in per.index:
        if labels.get(int(t)) != UNKNOWN:
            continue
        x = _feature(per.loc[t].to_numpy())[0]
        d = np.sqrt(((C[order[:2]] - x) ** 2).sum(1))
        j = int(np.argmin(d))
        if d[j] <= max(2.0 * spread[order[j]], 10.0):
            labels[int(t)] = team_of[order[j]]

    for j, name in team_of.items():
        members = per.loc[[t for t in good if labels[int(t)] == name]]
        if len(members):
            info["colours"][name] = lab_to_bgr(members.median().to_numpy())
    others = per.loc[[t for t in per.index if labels.get(int(t)) == OTHER]]
    if len(others):
        info["colours"][OTHER] = lab_to_bgr(others.median().to_numpy())
    info["counts"] = pd.Series(labels).value_counts().to_dict()
    return labels, info
