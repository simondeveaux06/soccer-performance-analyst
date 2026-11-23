"""Grouping track IDs into real players with a "lost players" list.

1. Short gaps first: IDs that clearly continue each other by motion (stitch.link_all)
   are chained into pieces.
2. Then the match is replayed in time order. Players who disappear go onto the
   lost list. Each new piece is compared with the lost players of its team, in
   this order of priority:
       a) shirt number   - same number = same player; different numbers = never
       b) last position  - could he have run from where he was lost to here in time?
       c) appearance     - fingerprint similarity (build, hair, boots, how he looks)
   If the best two candidates are too close to call, it doesn't guess: the piece
   starts a new "player" and the site's yellow "him?" suggestions cover it.

Result: {track_id: player_number} and {track_id: reason}.
"""

import numpy as np
import pandas as pd

from stitch import MAX_SPEED, SLACK_M, _track_summaries, chain_of, link_all

MIN_SIMILARITY = 0.80   # appearance below this is never treated as a match
MARGIN = 0.02           # best candidate must beat the runner-up by this much
NUMBER_VOTES = 2        # readings needed before a shirt number is trusted


def _number_of(numbers, ids):
    """Most-read shirt number for these IDs, if read often and consistently enough."""
    if numbers is None or numbers.empty:
        return None
    r = numbers[numbers.track_id.isin(ids)]
    if r.empty:
        return None
    votes = r.groupby("number").conf.sum()
    counts = r.number.value_counts()
    best = int(votes.idxmax())
    if counts.get(best, 0) >= NUMBER_VOTES and votes[best] >= 0.6 * votes.sum():
        return best
    return None


def _pieces(pitch_xy, fps, labels, tracks):
    """IDs chained by motion (short gaps) -> list of pieces (sets of IDs)."""
    links = link_all(pitch_xy, fps, labels) if pitch_xy is not None and len(pitch_xy) else {}
    seen, pieces = set(), []
    for t in sorted(tracks.track_id.unique()):
        t = int(t)
        if t in seen:
            continue
        p = chain_of(links, {t}) if links else {t}
        seen |= p
        pieces.append(p)
    return pieces


def group_players(tracks, pitch_xy, labels, extras, fps, max_lost_s=600):
    """Group every track ID into a player. Returns (player_of, reason_of)."""
    spans = tracks.groupby("track_id")["frame"].agg(["min", "max"])
    S = _track_summaries(pitch_xy) if pitch_xy is not None and len(pitch_xy) else {}
    looks_ids, _, looks_vecs = extras["looks"] if extras else (np.zeros(0), None, np.zeros((0, 0)))
    look_of = {}
    if len(looks_ids):
        v = looks_vecs.astype(np.float32)
        df = pd.DataFrame(v)
        df["tid"] = looks_ids
        for t, g in df.groupby("tid"):
            m = g.drop(columns="tid").to_numpy().mean(0)
            look_of[int(t)] = m / (np.linalg.norm(m) + 1e-9)
    numbers = extras.get("numbers") if extras else None

    def piece_info(ids):
        ids = sorted(ids, key=lambda t: spans.loc[t, "min"])
        first, last = ids[0], ids[-1]
        teams = [labels.get(t, "Unknown") for t in ids]
        team = max(set(teams), key=teams.count)
        vecs = [look_of[t] * (spans.loc[t, "max"] - spans.loc[t, "min"] + 1) for t in ids if t in look_of]
        look = None
        if vecs:
            m = np.sum(vecs, axis=0)
            look = m / (np.linalg.norm(m) + 1e-9)
        return {
            "ids": set(ids), "team": team,
            "start": spans.loc[first, "min"], "end": spans.loc[last, "max"],
            "start_pos": S[first][1] if first in S else None,
            "end_pos": S[last][4] if last in S else None,
            "look": look, "number": _number_of(numbers, ids),
            "size": sum(spans.loc[t, "max"] - spans.loc[t, "min"] + 1 for t in ids),
        }

    pieces = [piece_info(p) for p in _pieces(pitch_xy, fps, labels, tracks)]
    pieces.sort(key=lambda p: p["start"])

    players = []          # each: dict(ids, team, end, end_pos, look_sum, numbers)
    player_of, reason_of = {}, {}

    def same_team(a, b):
        return a == b or "Unknown" in (a, b)

    for p in pieces:
        reason = "new player"
        best = None
        if p["team"] not in ("Other", "Off pitch"):
            lost = [q for q in players
                    if q["end"] < p["start"] and same_team(q["team"], p["team"])
                    and (p["start"] - q["end"]) / fps <= max_lost_s]
            # a) shirt number: decisive both ways
            if p["number"] is not None:
                same = [q for q in lost if q["number"] == p["number"]]
                if len(same) == 1:
                    best, reason = same[0], f"shirt number {p['number']}"
                lost = [q for q in lost if q["number"] in (None, p["number"])]
            # b) position: drop anyone who couldn't have got here in time
            if best is None:
                fits = []
                for q in lost:
                    gap_s = (p["start"] - q["end"]) / fps
                    if p["start_pos"] is not None and q["end_pos"] is not None:
                        d = float(np.linalg.norm(np.asarray(p["start_pos"]) - np.asarray(q["end_pos"])))
                        if d > SLACK_M + MAX_SPEED * gap_s:
                            continue
                        pos_score = 1 - d / (SLACK_M + MAX_SPEED * gap_s)
                    else:
                        pos_score = 0.0
                    # c) appearance
                    sim = None
                    if p["look"] is not None and q["look"] is not None:
                        sim = float(p["look"] @ q["look"])
                        if sim < MIN_SIMILARITY:
                            continue
                    # Short gaps lean on position, long gaps on appearance.
                    w_pos = float(np.exp(-gap_s / 5.0))
                    score = (1 - w_pos) * (sim if sim is not None else 0.5) + w_pos * (0.8 + 0.2 * pos_score)
                    fits.append((score, sim, q))
                fits.sort(key=lambda f: -f[0])
                if fits and (len(fits) == 1 or fits[0][0] - fits[1][0] >= MARGIN):
                    # With no appearance at all, only accept a lone candidate on a short gap.
                    if fits[0][1] is not None or (len(fits) == 1 and (p["start"] - fits[0][2]["end"]) / fps <= 3):
                        best = fits[0][2]
                        reason = "position + appearance" if fits[0][1] is not None else "position"
                elif fits:
                    reason = "too close to call"
        if best is None:
            best = {"ids": set(), "team": p["team"], "end": -1, "end_pos": None,
                    "look_sum": None, "look": None, "number": None, "n": len(players) + 1}
            players.append(best)
        best["ids"] |= p["ids"]
        best["end"] = max(best["end"], p["end"])
        best["end_pos"] = p["end_pos"] if p["end_pos"] is not None else best["end_pos"]
        if p["look"] is not None:
            w = p["size"]
            best["look_sum"] = p["look"] * w if best["look_sum"] is None else best["look_sum"] + p["look"] * w
            best["look"] = best["look_sum"] / (np.linalg.norm(best["look_sum"]) + 1e-9)
        if best["number"] is None and p["number"] is not None:
            best["number"] = p["number"]
        if best["team"] == "Unknown":
            best["team"] = p["team"]
        for t in p["ids"]:
            player_of[int(t)] = best["n"]
            reason_of[int(t)] = reason if len(p["ids"]) == 1 or t == min(p["ids"], key=lambda i: spans.loc[i, "min"]) else "motion"
    return player_of, reason_of


def ids_of_player(player_of, tid):
    n = player_of.get(int(tid))
    return {t for t, m in player_of.items() if m == n} if n is not None else {int(tid)}
