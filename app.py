"""Soccer Performance Analyst - web app.

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
from pitch import (  # noqa: E402
    DEFAULT_MODEL_PATH, PitchDetector, build_homographies, draw_projected_pitch, map_tracks_to_pitch,
    nearest_homography,
)
from teams import OFF_PITCH, OTHER, TEAM_A, TEAM_B, UNKNOWN, assign_teams  # noqa: E402
from stats import (  # noqa: E402
    clean_trajectory, compare_to_pros, load_benchmarks, percentile_chart, physical_stats,
    pitch_heatmap, quality_problems,
)

DISPLAY_WIDTH = 960
POSITIONS = [
    "Goalkeeper", "Centre-back", "Full-back / Wing-back", "Defensive midfielder",
    "Central midfielder", "Attacking midfielder", "Winger", "Striker",
]

st.set_page_config(page_title="Soccer Performance Analyst", page_icon=":soccer:", layout="wide")
st.title(":soccer: Soccer Performance Analyst")
ss = st.session_state

# ------------------------------------------------------------------ 1. upload
st.header("1. Upload a clip")
uploaded = st.file_uploader("Drag your match clip here", type=["mp4", "mov", "avi", "mkv"])
if uploaded is None:
    st.info("Drop in a clip and tracking starts automatically. "
            "Start with a short one (1-3 minutes) so it's quick.")
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
st.caption(f"{info['width']}x{info['height']} | {info['fps']:.0f} fps | "
           f"{info['frames']} frames | {info['duration']:.0f} s")

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
    bar = st.progress(0.0, text="Tracking... the first run also downloads the model")
    detector = PitchDetector() if use_pitch else None
    ss.tracks, ss.keypoints = track_players(
        ss.video_path, conf=conf, pitch_detector=detector,
        progress=lambda p: bar.progress(p, text=f"Tracking... {p:.0%}"))
    ss.selected, ss.last_click = set(), None
    ss.pop("highlight", None)
    ss.pop("teams", None)
    bar.empty()

tracks = ss.tracks
if tracks is None:
    st.stop()
if tracks.empty:
    st.error("No players were detected. Try a lower confidence or a different clip.")
    st.stop()
st.success(f"Found {tracks.track_id.nunique()} track IDs.")

IMAGE_SIZE = (info["width"], info["height"])


def homographies(length, width):
    """Pitch homographies for this clip, cached per pitch size."""
    key = ("H", length, width, len(ss.keypoints))
    if ss.get("h_key") != key:
        ss.h_key = key
        ss.h_cache = build_homographies(ss.keypoints, length, width, IMAGE_SIZE)
    return ss.h_cache

def team_labels():
    """Team for every track (shirt colour + who's off the pitch), cached."""
    if "teams" not in ss:
        pitch_xy = None
        if ss.keypoints:
            # Every 5th frame is plenty to tell who stays off the pitch.
            sample = tracks[tracks.frame % 5 == 0]
            pitch_xy, _ = map_tracks_to_pitch(sample, ss.keypoints, image_size=IMAGE_SIZE,
                                              max_gap_frames=int(info["fps"]), margin_m=40.0)
        ss.teams = assign_teams(tracks, pitch_xy=pitch_xy)
    return ss.teams


labels, team_info = team_labels()
BOX_COLOURS = {OTHER: (160, 160, 160), OFF_PITCH: (70, 70, 70), UNKNOWN: (0, 165, 255)}
BOX_COLOURS.update(team_info.get("colours", {}))
BOX_COLOURS[OTHER], BOX_COLOURS[OFF_PITCH] = (160, 160, 160), (70, 70, 70)
colour_of = {t: BOX_COLOURS.get(lab, (0, 165, 255)) for t, lab in labels.items()}


def his_team():
    """Most common team label among the selected IDs (or None)."""
    picked = [labels.get(t) for t in ss.selected if labels.get(t) in (TEAM_A, TEAM_B)]
    return max(set(picked), key=picked.count) if picked else None


# ------------------------------------------------------------------ 3. pick player
st.header("3. Pick your player")
st.write("Click him on the frame. If he gets a new ID later (number next to his box changes), "
         "move the slider there and click him again. Click a green box to unselect it.")

frame_idx = st.slider("Frame", 0, max(info["frames"] - 1, 0), ss.get("frame_idx", 0), key="frame_idx")
f1, f2 = st.columns([2, 3])
show = f1.radio("Show boxes for", ["Everyone", TEAM_A, TEAM_B], horizontal=True, key="show_team")
hide_others = f2.checkbox("Hide referees, keepers and people off the pitch", value=True,
                          key="hide_others")
hidden = {t for t, lab in labels.items()
          if t not in ss.selected and (
              (hide_others and lab in (OTHER, OFF_PITCH)) or
              (show != "Everyone" and lab != show))}
frame = read_frame(ss.video_path, frame_idx)
left, right = st.columns([3, 1])

with left:
    if frame is None:
        st.warning("Couldn't read that frame.")
    else:
        drawn = draw_boxes(frame, tracks[tracks.frame == frame_idx], ss.selected,
                           colour_of=colour_of, hidden=hidden)
        if ss.keypoints and ss.get("show_lines"):
            Hs, _ = homographies(ss.get("pitch_len", 105.0), ss.get("pitch_wid", 68.0))
            H = nearest_homography(Hs, frame_idx, max_gap_frames=int(info["fps"]))
            if H is None:
                cv2.putText(drawn, "Pitch not located near this frame", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            else:
                drawn = draw_projected_pitch(drawn, H, ss.get("pitch_len", 105.0),
                                             ss.get("pitch_wid", 68.0))
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
            visible = tracks[~tracks.track_id.isin(hidden)]
            tid = pick_track(visible, frame_idx, click["x"] / scale, click["y"] / scale)
            if tid is None:
                st.toast("No player there.")
            else:
                ss.selected ^= {tid}  # toggle
                st.rerun()

with right:
    if ss.keypoints:
        st.checkbox("Check pitch mapping", key="show_lines",
                    help="Draws the pitch lines where the app thinks they are (cyan). If mapping "
                         "is working they sit right on top of the real white lines.")
        st.checkbox("Show pitch landmarks", key="show_kp",
                    help="Pink dots = pitch points the model found. They should sit on line "
                         "corners, the penalty spot or the centre circle.")
    if team_info.get("counts"):
        st.subheader("Teams")
        def swatch(lab):
            b, g, r = BOX_COLOURS.get(lab, (0, 165, 255))
            return (f"<span style='display:inline-block;width:14px;height:14px;border:1px solid #888;"
                    f"background:rgb({r},{g},{b});margin-right:6px;vertical-align:middle'></span>")
        names = {TEAM_A: TEAM_A, TEAM_B: TEAM_B, OTHER: "Refs / keepers",
                 OFF_PITCH: "Off pitch", UNKNOWN: "Not sure"}
        st.markdown("<br>".join(f"{swatch(k)}{names[k]}: {team_info['counts'][k]} IDs"
                                for k in names if team_info["counts"].get(k)),
                    unsafe_allow_html=True)
        st.caption("Box colour = shirt colour the app detected.")
    st.subheader("Selected IDs")
    st.write(", ".join(map(str, sorted(ss.selected))) or "None yet")
    team = his_team()
    if team:
        st.write(f"His team: **{team}**")
    if st.button("Auto-find his other IDs", disabled=not ss.selected,
                 help="Guesses which new IDs are him after the tracker lost him. Only looks at "
                      "his own team. Check the result!"):
        before = set(ss.selected)
        allowed = ({t for t, lab in labels.items() if lab in (team, UNKNOWN)} if team else None)
        ss.selected = auto_stitch(tracks, ss.selected, max_gap_frames=int(info["fps"] * 2),
                                  allowed_ids=allowed)
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
            if st.button(f"Frames {start}-{end} ({(end - start + 1) / info['fps']:.0f} s)",
                         key=f"gap_{start}"):
                ss.frame_idx = start
                st.rerun()
        if len(gaps) > 8:
            st.caption(f"...and {len(gaps) - 8} more")
        st.caption("Gaps are normal when he's off screen.")

if not ss.selected:
    st.stop()

# ------------------------------------------------------------------ 4. results
st.header("4. Results")
c1, c2, c3 = st.columns(3)
position = c1.selectbox("His position", POSITIONS, index=POSITIONS.index("Winger"))
length = c2.number_input("Pitch length (m)", 80.0, 120.0, 105.0, 1.0, key="pitch_len",
                         help="Measure or look it up; distances scale with this.")
width = c3.number_input("Pitch width (m)", 45.0, 90.0, 68.0, 1.0, key="pitch_wid")
rows = player_rows(tracks, ss.selected)

cols = st.columns(4)
for col, (label, value) in zip(cols, player_stats(rows, info).items()):
    col.metric(label, value)

traj, phys, report = None, None, None
if ss.keypoints:
    mapped, report = map_tracks_to_pitch(rows, ss.keypoints, length, width, image_size=IMAGE_SIZE,
                                         max_gap_frames=int(info["fps"]))
    traj = clean_trajectory(mapped, info["fps"])
    phys = physical_stats(traj)
    with st.expander(f"Pitch mapping quality: pitch located in {report['usable_share']:.0%} of "
                     f"checked frames, him placed in {report['mapped_share']:.0%} of his frames"):
        st.write(f"Frames checked:
