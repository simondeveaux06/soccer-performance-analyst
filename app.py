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
uploaded = st.file_uploader("Match video", type=["mp4", "mov", "avi", "mkv"])
if uploaded is None:
    st.info("Start with a short clip (1–3 minutes) so tracking is quick.")
    st.stop()

key = (uploaded.name, uploaded.size)
if ss.get("video_key") != key:
    folder = Path(tempfile.mkdtemp(prefix="spa_"))
    path = folder / uploaded.name
    path.write_bytes(uploaded.getbuffer())
    ss.video_key, ss.video_path, ss.workdir = key, str(path), str(folder)
    ss.info = video_info(path)
    ss.tracks, ss.selected, ss.last_click = None, set(), None
    ss.pop("highlight", None)

info = ss.info
st.caption(f"{info['width']}×{info['height']} · {info['fps']:.0f} fps · "
           f"{info['frames']} frames · {info['duration']:.0f} s")

# ------------------------------------------------------------------ 2. track
st.header("2. Track the players")
conf = st.slider("Detection confidence", 0.10, 0.80, 0.30, 0.05,
                 help="Lower finds more players (and more false boxes).")
if st.button("Run tracking", type="primary"):
    bar = st.progress(0.0, text="Tracking… the first run also downloads the model")
    ss.tracks = track_players(ss.video_path, conf=conf,
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
position = st.selectbox("His position", POSITIONS, index=POSITIONS.index("Central midfielder"))
rows = player_rows(tracks, ss.selected)

cols = st.columns(4)
for col, (label, value) in zip(cols, player_stats(rows, info).items()):
    col.metric(label, value)

st.subheader("Heatmap")
st.caption("On-screen position for now. With a moving camera this isn't pitch position yet; "
           "that comes with pitch mapping.")
background = read_frame(ss.video_path, int(rows.frame.iloc[len(rows) // 2]))
st.pyplot(heatmap_figure(rows, background))

st.subheader("Downloads")
d1, d2 = st.columns(2)
d1.download_button("His tracking data (CSV)", rows.to_csv(index=False),
                   file_name="player_tracking.csv", mime="text/csv")
if d2.button("Make highlight video (only him boxed)"):
    bar = st.progress(0.0, text="Rendering…")
    out = Path(ss.workdir) / "highlight.mp4"
    render_highlight(ss.video_path, rows, out, progress=lambda p: bar.progress(p))
    ss.highlight = str(out)
    bar.empty()
if ss.get("highlight"):
    d2.download_button("Download highlight video", Path(ss.highlight).read_bytes(),
                       file_name="highlight.mp4", mime="video/mp4")

st.subheader(f"Compared with pro {position.lower()}s")
st.info("Coming next: once positions are mapped onto the pitch (metres instead of pixels), "
        "his distance, speed and sprints get compared with pro tracking data for this position.")
