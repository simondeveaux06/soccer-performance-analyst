"""Build per-90 pro benchmarks from SkillCorner's open A-League 2024/25 physical data.

Source: https://github.com/SkillCorner/opendata (data/aggregates/aus1league_physicalaggregates_20242025.csv)
Data (c) SkillCorner, shared for the analytics community; please credit SkillCorner.

SkillCorner's values are per-match averages; minutes_full_all is average minutes
per match. Per 90 = value / minutes * 90. Only performances over 60 minutes are
included (SkillCorner's filter).

Usage:
    python scripts/build_benchmarks.py path/to/aus1league_physicalaggregates_20242025.csv
"""
import sys
from pathlib import Path

import pandas as pd

src = sys.argv[1] if len(sys.argv) > 1 else (
    "https://raw.githubusercontent.com/SkillCorner/opendata/master/"
    "data/aggregates/aus1league_physicalaggregates_20242025.csv")
d = pd.read_csv(src)
m = d.minutes_full_all
out = pd.DataFrame({
    "player_name": d.player_short_name,
    "team_name": d.team_name,
    "position_group": d.position_group,
    "matches": d.count_match,
    "minutes_per_match": m.round(1),
    "total_distance_p90": (d.total_metersperminute_full_all * 90).round(0),
    "running_distance_p90": (d.running_distance_full_all / m * 90).round(0),
    "hsr_distance_p90": (d.hsr_distance_full_all / m * 90).round(0),
    "sprint_distance_p90": (d.sprint_distance_full_all / m * 90).round(0),
    "hi_distance_p90": (d.hi_distance_full_all / m * 90).round(0),
    "hsr_count_p90": (d.hsr_count_full_all / m * 90).round(1),
    "sprint_count_p90": (d.sprint_count_full_all / m * 90).round(1),
    "top_speed": d.psv99.round(2),
})
dest = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "skillcorner_aleague_2024_25_physical_p90.csv"
dest.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(dest, index=False)
print(f"Wrote {len(out)} players to {dest}")
print(out.groupby("position_group").median(numeric_only=True).round(1).to_string())
