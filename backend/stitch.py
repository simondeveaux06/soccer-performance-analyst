"""Linking a player's track IDs together, on the pitch (in metres).

The tracker gives a player a new ID whenever it loses him (blocked, off screen).
To rebuild his whole match we chain those IDs:

    his ID ends here --(gap of a few seconds)--> which ID starts nearby?

"Nearby" is measured on the pitch, not on screen, so a panning camera doesn't
matter, and the allowed distance grows with the gap (a player can't run faster
than ~8 m/s). A candidate must be on his team, and must never be on screen at the
same time as him. If two candidates are about equally likely, we skip rather than
guess: one wrong ID would mix another player's running into his stats.
"""

import numpy as np
import pandas as pd

MAX_SPEED = 8.0   # m/s a player can plausibly average across a gap
SLACK_M = 3.0     # metres of leeway for mapping error


def _intervals(spans, ids):
    """Merged [start, end] frame intervals covered by these IDs."""
    iv = sorted((int(spans.loc[i, "min"]), int(spans.loc[i, "max"])) for i in ids if i in spans.index)
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def _overlaps(a, b, intervals):
    return any(a <= y and x <= b for x, y in intervals)


def suggest_at_frame(tracks, pitch_xy, selected_ids, frame, fps, allowed_ids=None, top=3):
    """Likely candidates for him at a frame where he isn't selected.

    Scores each teammate on screen by (a) how far he could have run since he was
    last seen (before or after this frame, whichever is closer in time) and
    (b) how close they are to where he usually plays (his average spot so far,
    mostly how wide he plays). Returns [(track_id, score)], best first.
    """
    if pitch_xy is None or pitch_xy.empty or not selected_ids:
        return []
    spans = tracks.groupby("track_id")["frame"].agg(["min", "max"])
    p = pitch_xy.dropna(subset=["pitch_x"])
    mine = p[p.track_id.isin(selected_ids)]
    if mine.empty:
        return []
    home = mine[["pitch_x", "pitch_y"]].median().to_numpy()
    spread = np.maximum(mine[["pitch_x", "pitch_y"]].std().fillna(10).to_numpy(), [8.0, 5.0])
    before = mine[mine.frame <= frame].tail(1)
    after = mine[mine.frame > frame].head(1)
    covered = _intervals(spans, set(selected_ids))
    here = tracks[tracks.frame == frame].track_id.unique()
    out = []
    for t in here:
        t = int(t)
        if t in selected_ids or (allowed_ids is not None and t not in allowed_ids):
            continue
        if _overlaps(spans.loc[t, "min"], spans.loc[t, "max"], covered):
            continue
        pt = p[p.track_id == t]
        if pt.empty:
            continue
        near = pt.iloc[(pt.frame - frame).abs().argsort()[:1]].iloc[0]
        refs = [r.iloc[0] for r in (before, after) if not r.empty]
        r = min(refs, key=lambda q: abs(q.frame - near.frame))
        gap_s = abs(near.frame - r.frame) / fps
        d = np.hypot(near.pitch_x - r.pitch_x, near.pitch_y - r.pitch_y)
        reach = d / (SLACK_M + MAX_SPEED * gap_s)
        if reach > 1.0:
            continue  # couldn't have got there in time
        motion = d / (SLACK_M + 3.0 * gap_s)          # how surprising the move is
        area = np.hypot(*((np.array([near.pitch_x, near.pitch_y]) - home) / spread))
        w = min(gap_s / 10.0, 1.0)                    # long gap: rely more on his usual area
        out.append((t, (1 - w) * min(motion, 3) + w * area + 0.3 * area))
    return sorted(out, key=lambda s: s[1])[:top]


def _track_summaries(pitch_xy):
    """Per track: first/last frame + position, and velocity at each end (m/frame)."""
    p = pitch_xy.dropna(subset=["pitch_x"]).sort_values(["track_id", "frame"])
    out = {}
    for t, g in p.groupby("track_id"):
        f = g.frame.to_numpy(float)
        xy = g[["pitch_x", "pitch_y"]].to_numpy(float)
        k = min(3, len(g))
        a, b = xy[:k].mean(0), xy[-k:].mean(0)
        v_end = v_start = np.zeros(2)
        if len(g) >= 2:
            j = min(len(g) - 1, 4)
            v_end = (xy[-1] - xy[-1 - j]) / max(f[-1] - f[-1 - j], 1)
            v_start = (xy[j] - xy[0]) / max(f[j] - f[0], 1)
        out[int(t)] = (f[0], a, v_start, f[-1], b, v_end)
    return out


def link_all(pitch_xy, fps, labels=None, max_gap_s=10.0, ambiguity=0.7):
    """Link every track to the track most likely to be the same person next.

    Every player's broken-off IDs compete at once: a new ID goes to whichever
    ended track explains it best (predicted position from speed and direction,
    same team, and an allowed error that grows with the length of the gap).
    Links are made best-first, each ID gets at most one successor and one
    predecessor, and a link is skipped when another candidate fits almost as well.
    Returns {track_id: next_track_id}.
    """
    S = _track_summaries(pitch_xy)
    if not S:
        return {}
    ids = np.array(list(S))
    f0 = np.array([S[t][0] for t in ids]); p0 = np.array([S[t][1] for t in ids]); v0 = np.array([S[t][2] for t in ids])
    f1 = np.array([S[t][3] for t in ids]); p1 = np.array([S[t][4] for t in ids]); v1 = np.array([S[t][5] for t in ids])
    team = np.array([(labels or {}).get(int(t), "Unknown") for t in ids])
    max_gap = max_gap_s * fps
    order = np.argsort(f0)
    f0_sorted = f0[order]
    pairs = []
    for i in range(len(ids)):
        lo = np.searchsorted(f0_sorted, f1[i], side="right")
        hi = np.searchsorted(f0_sorted, f1[i] + max_gap, side="right")
        js = order[lo:hi]
        if not len(js):
            continue
        ok = (team[js] == team[i]) | (team[js] == "Unknown") | (team[i] == "Unknown")
        js = js[ok & (team[js] != "Off pitch")]
        if not len(js):
            continue
        gap = f0[js] - f1[i]
        gap_s = gap / fps
        h = np.minimum(gap, fps * 1.0)[:, None]  # trust his direction for about a second
        pred_i = p1[i] + v1[i] * h                # where i was heading
        pred_j = p0[js] - v0[js] * h              # where j came from
        meet = (pred_i + pred_j) / 2
        d = np.linalg.norm(p0[js] - meet, axis=1) + np.linalg.norm(p1[i] - meet, axis=1)
        reach = np.linalg.norm(p0[js] - p1[i], axis=1)
        tol = 2.0 + 2.5 * gap_s
        cost = d / tol + 0.02 * gap_s
        fine = (d <= tol) & (reach <= SLACK_M + MAX_SPEED * gap_s)
        for j, c in zip(js[fine], cost[fine]):
            pairs.append((c, i, j))
    pairs.sort()
    # Candidates per end / per start, to check for real ambiguity. Only IDs that are
    # on screen at the SAME time compete (two different people). His own later IDs
    # are not competitors: they never overlap with each other.
    cands_i, cands_j = {}, {}
    for c, i, j in pairs:
        cands_i.setdefault(i, []).append((c, j))
        cands_j.setdefault(j, []).append((c, i))

    def overlap(a, b):
        return f0[a] <= f1[b] and f0[b] <= f1[a]

    nxt, has_prev = {}, set()          # both keyed by position in `ids`
    for c, i, j in pairs:
        if i in nxt or j in has_prev:
            continue
        rival_j = [cc for cc, jj in cands_i[i] if jj != j and jj not in has_prev and overlap(jj, j)]
        rival_i = [cc for cc, ii in cands_j[j] if ii != i and ii not in nxt and overlap(ii, i)]
        if (rival_j and c > ambiguity * min(rival_j)) or (rival_i and c > ambiguity * min(rival_i)):
            continue
        nxt[i] = j
        has_prev.add(j)
    return {int(ids[i]): int(ids[j]) for i, j in nxt.items()}


def chain_of(links, seeds):
    """All IDs chained (forwards and backwards) to any of the seed IDs."""
    prev = {b: a for a, b in links.items()}
    out = set()
    for s in seeds:
        t = int(s)
        while t in prev and prev[t] not in out:
            t = prev[t]
        while t is not None and t not in out:
            out.add(t)
            t = links.get(t)
    return out | set(int(s) for s in seeds)


def selected_minutes(tracks, selected_ids, fps, stride=1):
    """Minutes of video in which he's selected."""
    frames = tracks.loc[tracks.track_id.isin(selected_ids), "frame"].nunique()
    return frames * stride / fps / 60
