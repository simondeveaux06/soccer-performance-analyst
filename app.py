"""Soccer Performance Analyst — web app.

Run from the repo root:
    streamlit run app.py
then open http://localhost:8501
"""

import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

sys.path.insert(0, str(Path(__file__).parent / "backend"))
from analysis import (  # noqa: E402
    auto_stitch, draw_boxes, heatmap_figure, missing_stretches, pick_track,
    player_rows, player_stats, read_frame, render_highlight, track_players, video_info,
)
from pitch import DEFAULT_MODEL_PATH, PitchDetector, keypoint_coverage, map_tracks_to_pitch  # noqa: E402
from stats import (  # noqa: E402
    clean_trajectory, compare_to_pros, load_benchmarks, percentile_chart,
    physical_stats, pitch_heatmap,
)

DISPLAY_WIDTH = 960
POSITIONS = [
    "Goalkeeper", "Centre-back", "Full-back / Wing-back", "Defensive midfielder",
    "Central midfielder", "Attacking midfielder", "Winger", "Striker",
]

st.set_page_config(page_title="Soccer Performance Analyst", page_icon="⚽", layout="wide")
st.title("⚽ Soccer Performance Analyst")
ss = st.session_state

# ------------------------------------------------------------------ 1. upload
st.header("1. Upload a clip")
uploaded = st.file_uploader("Drag your match clip here", type=["mp4", "mov", "avi", "mkv"])
if uploaded is None:
    st.info("Drop in a clip and tracking starts automatically. "
            "Start with a short one (1–3 minutes) so it's quick.")
    st.stop()

key = (uploaded.name, uploaded.size)
if ss.get("video_key") != key:
    folder = Path(tempfile.mkdtemp(prefix="spa_"))
    path = folder / uploaded.name
    path.write_bytes(uploaded.getbuffer())
    ss.video_key, ss.video_path, ss.workdir = key, str(path), str(folder)
    ss.info = video_info(path)
    ss.tracks, ss.keypoints, ss.selected, ss.last_click = None, {}, set(), None
    ss.pop("highlight", None)

info = ss.info
st.caption(f"{info['width']}×{info['height']} · {info['fps']:.0f} fps · "
           f"{info['frames']} frames · {info['duration']:.0f} s")

# ------------------------------------------------------------------ 2. track
st.header("2. Track the players")
with st.expander("Settings"):
    conf = st.slider("Detection confidence", 0.10, 0.80, 0.30, 0.05,
                     help="Lower finds more players (and more false boxes).")
    use_pitch = st.checkbox("Map positions onto the pitch", value=DEFAULT_MODEL_PATH.exists(),
                            disabled=not DEFAULT_MODEL_PATH.exists(),
                            help="Needs the pitch model: run `bash download_models.sh` once.")
    rerun = st.button("Re-run tracking with these settings")
if not DEFAULT_MODEL_PATH.exists():
    st.warning("Pitch model not found, so you'll only get on-screen stats. "
               "To get distance, speed and the pro comparison, stop the app and run "
               "`bash download_models.sh`, then start it again.")
if ss.tracks is None or rerun:  # starts automatically when a clip is dropped in
    bar = st.progress(0.0, text="Tracking… the first run also downloads the model")
    detector = PitchDetector() if use_pitch else None
    ss.tracks, ss.keypoints = track_players(
        ss.video_path, conf=conf, pitch_detector=detector,
        progress=lambda p: bar.progress(p, text=f"Tracking… {p:.0%}"))
    ss.selected, ss.last_click = set(), None
    ss.pop("highlight", None)
    bar.empty()

tracks = ss.tracks
if tracks is None:
    st.stop()
if tracks.empty:
    st.error("No players were detected. Try a lower confidence or a different clip.")
    st.stop()
st.success(f"Found {tracks.track_id.nunique()} track IDs.")

# ------------------------------------------------------------------ 3. pick player
st.header("3. Pick your player")
st.write("Click him on the frame. If he gets a new ID later (number next to his box changes), "
         "move the slider there and click him again. Click a green box to unselect it.")

frame_idx = st.slider("Frame", 0, max(info["frames"] - 1, 0), ss.get("frame_idx", 0), key="frame_idx")
frame = read_frame(ss.video_path, frame_idx)
left, right = st.columns([3, 1])

with left:
    if frame is None:
        st.warning("Couldn't read that frame.")
    else:
        drawn = draw_boxes(frame, tracks[tracks.frame == frame_idx], ss.selected)
        if ss.keypoints and ss.get("show_kp"):
            near = min(ss.keypoints, key=lambda f: abs(f - frame_idx))
            xy, kconf = ss.keypoints[near]
            for i, ((kx, ky), kc) in enumerate(zip(xy, kconf)):
                if kc >= 0.5:
                    cv2.circle(drawn, (int(kx), int(ky)), 7, (255, 0, 255), -1)
                    cv2.putText(drawn, str(i + 1), (int(kx) + 8, int(ky) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        scale = DISPLAY_WIDTH / drawn.shape[1]
        shown = cv2.resize(drawn, (DISPLAY_WIDTH, int(drawn.shape[0] * scale)))
        click = streamlit_image_coordinates(
            Image.fromarray(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB)), key=f"img_{frame_idx}")
        if click and (frame_idx, click["x"], click["y"]) != ss.last_click:
            ss.last_click = (frame_idx, click["x"], click["y"])
            tid = pick_track(tracks, frame_idx, click["x"] / scale, click["y"] / scale)
            if tid is None:
                st.toast("No player there.")
            else:
                ss.selected ^= {tid}  # toggle
                st.rerun()

with right:
    if ss.keypoints:
        st.checkbox("Show pitch landmarks", key="show_kp",
                    help="Pink dots = pitch points the model found. They should sit on line "
                         "corners, the penalty spot or the centre circle.")
    st.subheader("Selected IDs")
    st.write(", ".join(map(str, sorted(ss.selected))) or "None yet")
    if st.button("Auto-find his other IDs", disabled=not ss.selected,
                 help="Guesses which new IDs are him after the tracker lost him. Check the result!"):
        before = set(ss.selected)
        ss.selected = auto_stitch(tracks, ss.selected, max_gap_frames=int(info["fps"] * 2))
        added = ss.selected - before
        st.toast(f"Added {len(added)} ID(s): {sorted(added)}" if added else "Nothing new found.")
        st.rerun()
    if st.button("Clear selection", disabled=not ss.selected):
        ss.selected = set()
        st.rerun()

    if ss.selected:
        rows = player_rows(tracks, ss.selected)
        gaps = missing_stretches(rows, info["frames"], min_length=int(info["fps"]))
        st.subheader("Where he's missing")
        if not gaps:
            st.write("Tracked the whole clip.")
        for start, end in gaps[:8]:
            if st.button(f"Frames {start}–{end} ({(end - start + 1) / info['fps']:.0f} s)",
                         key=f"gap_{start}"):
                ss.frame_idx = start
                st.rerun()
        if len(gaps) > 8:
            st.caption(f"…and {len(gaps) - 8} more")
        st.caption("Gaps are normal when he's off screen.")

if not ss.selected:
    st.stop()

# ------------------------------------------------------------------ 4. results
st.header("4. Results")
c1, c2, c3 = st.columns(3)
position = c1.selectbox("His position", POSITIONS, index=POSITIONS.index("Winger"))
length = c2.number_input("Pitch length (m)", 80.0, 120.0, 105.0, 1.0,
                         help="Measure or look it up; distances scale with this.")
width = c3.number_input("Pitch width (m)", 45.0, 90.0, 68.0, 1.0)
rows = player_rows(tracks, ss.selected)

cols = st.columns(4)
for col, (label, value) in zip(cols, player_stats(rows, info).items()):
    col.metric(label, value)

traj, phys = None, None
if ss.keypoints:
    mapped = map_tracks_to_pitch(rows, ss.keypoints, length, width,
                                 max_gap_frames=int(info["fps"]))
    traj = clean_trajectory(mapped, info["fps"])
    phys = physical_stats(traj)
    cover = keypoint_coverage(ss.keypoints, length, width)
    st.caption(f"Pitch found in {cover:.0%} of checked frames · "
               f"his position mapped in {mapped.pitch_x.notna().mean():.0%} of his frames.")

if phys is None:
    st.subheader("Heatmap (on screen)")
    st.caption("Without pitch mapping this is where he was on screen, not on the pitch.")
    background = read_frame(ss.video_path, int(rows.frame.iloc[len(rows) // 2]))
    st.pyplot(heatmap_figure(rows, background))
    if ss.keypoints:
        st.warning("Couldn't map him onto the pitch: the pitch lines weren't found often enough "
                   "while he was visible.")
else:
    st.subheader("Physical stats")
    st.caption(f"Over the {phys['minutes_tracked']:.1f} minutes he was tracked on the pitch. "
               "Per-90 numbers scale that up to a full match.")
    m = st.columns(4)
    m[0].metric("Distance", f"{phys['total_distance']:.0f} m",
                f"{phys['total_distance_p90'] / 1000:.1f} km per 90", delta_color="off")
    m[1].metric("Top speed", f"{phys['top_speed']:.1f} km/h",
                f"max {phys['max_speed']:.1f}", delta_color="off")
    m[2].metric("Sprints (>25 km/h)", f"{phys['sprint_count']:.0f}",
                f"{phys['sprint_count_p90']:.1f} per 90", delta_color="off")
    m[3].metric("High intensity (>20 km/h)", f"{phys['hi_distance']:.0f} m",
                f"{phys['hi_distance_p90']:.0f} m per 90", delta_color="off")
    if phys["minutes_tracked"] < 10:
        st.info("Less than 10 minutes tracked, so per-90 numbers are rough. "
                "Longer clips give a fairer comparison.")

    st.subheader("Heatmap")
    flip = st.checkbox("Flip pitch (if his team attacks the other way)")
    st.pyplot(pitch_heatmap(traj, length, width, flip=flip))

    st.subheader(f"Compared with pro {position.lower()}s")
    benchmarks = load_benchmarks()
    comparison, group = compare_to_pros(phys, position, benchmarks)
    if benchmarks is None:
        st.info("Pro data not found. Run `python scripts/build_benchmarks.py` once.")
    elif comparison is None or comparison.empty:
        st.info("No pro benchmarks for this position.")
    else:
        n = int((benchmarks.position_group == group).sum())
        st.pyplot(percentile_chart(
            comparison, f"vs {n} A-League {group.lower()}s (SkillCorner, 2024/25)"))
        table = comparison.assign(
            him=comparison.him.round(1), pro_median=comparison.pro_median.round(1),
            percentile=comparison.percentile.round(0).astype(int))
        st.dataframe(table[["metric", "unit", "him", "pro_median", "percentile"]]
                     .rename(columns={"metric": "Metric", "unit": "Unit", "him": "Him",
                                      "pro_median": "Pro median", "percentile": "Percentile"}),
                     hide_index=True, width="stretch")
        st.caption("Pro data: SkillCorner open data, Australian A-League 2024/25, performances "
                   "over 60 min. Pro numbers come from full-match broadcast tracking; his come from "
                   "only the time he was visible, so treat this as a guide, not an exact ranking.")

st.subheader("Downloads")
d1, d2, d3 = st.columns(3)
d1.download_button("His tracking data (CSV)", rows.to_csv(index=False),
                   file_name="player_tracking.csv", mime="text/csv")
if traj is not None and len(traj):
    d2.download_button("His pitch positions + speed (CSV)", traj.to_csv(index=False),
                       file_name="player_pitch.csv", mime="text/csv")
if d3.button("Make highlight video (only him boxed)"):
    bar = st.progress(0.0, text="Rendering…")
    out = Path(ss.workdir) / "highlight.mp4"
    render_highlight(ss.video_path, rows, out, progress=lambda p: bar.progress(p))
    ss.highlight = str(out)
    bar.empty()
if ss.get("highlight"):
    d3.download_button("Download highlight video", Path(ss.highlight).read_bytes(),
                       file_name="highlight.mp4", mime="video/mp4")
