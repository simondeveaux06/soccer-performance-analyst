"""Soccer Performance Analyst - web app.

Run from the repo root:
    streamlit run app.py
then open http://localhost:8501
"""

import sys
import tempfile
from pathlib import Path

import cv2
import pandas as pd
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
from heatmap import pitch_heatmap as pitch_heatmap_v2  # noqa: E402
from stitch import chain_of, link_all, selected_minutes, suggest_at_frame  # noqa: E402
from reid import group_players, ids_of_player  # noqa: E402
from storage import RUNS_DIR, list_runs, load_extras, load_run, run_dir_for, save_chunk, write_meta  # noqa: E402
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

# ------------------------------------------------------------------ 1. choose a match
st.header("1. Choose a match")
runs = list_runs()
source = "Upload a clip"
if runs:
    source = st.radio("Source", ["Open a tracked match", "Upload a clip"], horizontal=True,
                      help="Long videos: track them once with `python track_match.py video.mp4`, "
                           "then open them here instantly.")

if source == "Open a tracked match":
    def run_label(rm):
        d, m = rm
        done = "" if m["status"] == "done" else f" (unfinished: {100 * m['next_frame'] / max(m['frames'], 1):.0f}%)"
        mode = " | fast mode" if m.get("stride", 1) > 1 else ""
        return f"{m['name']} | {m['frames'] / m['fps'] / 60:.0f} min{mode} | {m['updated']}{done}"
    picked = st.selectbox("Tracked matches", runs, format_func=run_label)
    run_path, meta = picked
    key = ("run", str(run_path), meta.get("updated"))
    if ss.get("video_key") != key:
        if not Path(meta["video"]).exists():
            st.error(f"The video for this match isn't at {meta['video']} anymore. "
                     "Move it back there, or track it again.")
            st.stop()
        with st.spinner("Loading saved tracking..."):
            tracks_loaded, keypoints_loaded, _ = load_run(run_path)
            ss.extras = load_extras(run_path)
        ss.video_key, ss.video_path, ss.workdir = key, meta["video"], str(run_path)
        ss.info = dict(video_info(meta["video"]), stride=meta.get("stride", 1))
        ss.tracks, ss.keypoints, ss.selected, ss.last_click = tracks_loaded, keypoints_loaded, set(), None
        for k in ("highlight", "teams", "h_key", "frame_idx", "links", "pitch_sample", "track_frames", "players"):
            ss.pop(k, None)
    if meta["status"] != "done":
        st.warning("This match isn't fully tracked yet. Run the same `track_match.py` command "
                   "again to finish it; for now you'll see what's done so far.")
else:
    uploaded = st.file_uploader("Drag your match clip here", type=["mp4", "mov", "avi", "mkv"])
    if uploaded is None:
        st.info("Drop in a clip and tracking starts automatically. Start with a short one "
                "(1-3 minutes). For a whole half, use `python track_match.py video.mp4` "
                "in the terminal instead: it can't be interrupted and saves its progress.")
        st.stop()
    key = ("upload", uploaded.name, uploaded.size)
    if ss.get("video_key") != key:
        folder = RUNS_DIR / Path(uploaded.name).stem
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / uploaded.name
        path.write_bytes(uploaded.getbuffer())
        ss.video_key, ss.video_path, ss.workdir = key, str(path), str(folder)
        ss.info = dict(video_info(path), stride=1)
        ss.tracks, ss.keypoints, ss.selected, ss.last_click = None, {}, set(), None
        ss.extras = None
        for k in ("highlight", "teams", "h_key", "frame_idx", "links", "pitch_sample", "track_frames", "players"):
            ss.pop(k, None)

info = ss.info
st.caption(f"{info['width']}x{info['height']} | {info['fps']:.0f} fps | "
           f"{info['frames']} frames | {info['duration']:.0f} s")

# ------------------------------------------------------------------ 2. track
st.header("2. Track the players")
with st.expander("Settings"):
    conf = st.slider("Detection confidence", 0.10, 0.80, 0.30, 0.05,
                     help="Lower finds more players (and more false boxes).")
    fast = st.checkbox("Fast mode (check every 2nd frame, about 2x quicker)", value=False)
    use_pitch = st.checkbox("Map positions onto the pitch", value=DEFAULT_MODEL_PATH.exists(),
                            disabled=not DEFAULT_MODEL_PATH.exists(),
                            help="Needs the pitch model: run `bash download_models.sh` once.")
    rerun = st.button("Re-run tracking with these settings")
if not DEFAULT_MODEL_PATH.exists():
    st.warning("Pitch model not found, so you'll only get on-screen stats. "
               "To get distance, speed and the pro comparison, stop the app and run "
               "`bash download_models.sh`, then start it again.")
if ss.tracks is None or rerun:  # starts automatically when a clip is dropped in
    bar = st.progress(0.0, text="Tracking... the first run also downloads the model. "
                                "Don't click anything until it's done.")
    detector = PitchDetector() if use_pitch else None
    stride = 2 if fast else 1
    ss.tracks, ss.keypoints, ss.extras = track_players(
        ss.video_path, conf=conf, pitch_detector=detector, stride=stride,
        progress=lambda p: bar.progress(p, text=f"Tracking... {p:.0%}"))
    ss.info["stride"] = stride
    # Save it, so this clip shows up under "Open a tracked match" next time.
    run_path = Path(ss.workdir)
    for pattern in ("tracks_*.parquet", "keypoints_*.npz", "looks_*.npz", "numbers_*.parquet"):
        for old in run_path.glob(pattern):
            old.unlink()
    save_chunk(run_path, ss.tracks, ss.keypoints, ss.extras)
    write_meta(run_path, {"video": ss.video_path, "name": Path(ss.video_path).stem,
                          "fps": info["fps"], "frames": info["frames"], "width": info["width"],
                          "height": info["height"], "stride": stride, "status": "done",
                          "next_frame": info["frames"],
                          "settings": {"stride": stride, "conf": conf, "pitch": use_pitch}})
    ss.selected, ss.last_click = set(), None
    ss.pop("highlight", None)
    ss.pop("teams", None)
    ss.pop("links", None)
    ss.pop("pitch_sample", None)
    ss.pop("track_frames", None)
    ss.pop("players", None)
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
        ss.pitch_sample = pitch_xy
        labels_, info_ = assign_teams(tracks, pitch_xy=pitch_xy)
        if "role" in tracks:  # the football detector knows who the referees are
            role = tracks.groupby("track_id")["role"].agg(lambda r: r.value_counts().idxmax())
            for t in role[role == "referee"].index:
                labels_[int(t)] = OTHER
            info_["counts"] = pd.Series(labels_).value_counts().to_dict() if labels_ else {}
        ss.teams = (labels_, info_)
        ss.pop("links", None)
        ss.pop("players", None)
    return ss.teams


labels, team_info = team_labels()

BOX_COLOURS = {OTHER: (160, 160, 160), OFF_PITCH: (70, 70, 70), UNKNOWN: (0, 165, 255)}
BOX_COLOURS.update(team_info.get("colours", {}))
BOX_COLOURS[OTHER], BOX_COLOURS[OFF_PITCH] = (160, 160, 160), (70, 70, 70)
colour_of = {t: BOX_COLOURS.get(lab, (0, 165, 255)) for t, lab in labels.items()}


def id_links():
    """Which ID most likely continues which (all players at once), cached."""
    if "links" not in ss:
        ps = ss.get("pitch_sample")
        ss.links = link_all(ps, info["fps"], labels) if ps is not None and len(ps) else {}
    return ss.links


def player_groups():
    """Every ID grouped into a real player (number > position > appearance), cached.
    Returns (player_of, reason_of)."""
    if "players" not in ss:
        ss.players = group_players(tracks, ss.get("pitch_sample"), labels,
                                   ss.get("extras"), info["fps"])
    return ss.players


def track_frames():
    """{track_id: set of frames it's on screen}, cached."""
    if "track_frames" not in ss:
        ss.track_frames = {int(t): set(f) for t, f in tracks.groupby("track_id")["frame"]}
    return ss.track_frames


def selected_frames():
    tf = track_frames()
    out = set()
    for t in ss.selected:
        out |= tf.get(t, set())
    return out


def add_ids_safely(candidates):
    """Add IDs one by one (longest first), skipping any that would be on screen at the
    same time as what's already selected: that would be a second person. Returns count."""
    tf = track_frames()
    taken = selected_frames()
    added = 0
    for t in sorted(set(candidates) - ss.selected, key=lambda t: -len(tf.get(t, ()))):
        f = tf.get(t, set())
        if f & taken:
            continue
        ss.selected.add(t)
        taken |= f
        added += 1
    return added


def add_player(tid):
    """Add this ID and every ID chained to it, never a second person.
    Returns (ids_added, clash_frame); clash_frame is set if tid itself is someone else."""
    both = track_frames().get(tid, set()) & selected_frames()
    if both:
        return 0, min(both)
    ss.selected.add(tid)
    group = ids_of_player(player_groups()[0], tid)
    if id_links():
        group |= chain_of(id_links(), {tid})
    return 1 + add_ids_safely(group - {tid}), None


def mixed_frames():
    """Frames where two selected IDs are on screen at once (= two people selected)."""
    sel = tracks[tracks.track_id.isin(ss.selected)]
    counts = sel.groupby("frame").size()
    return int((counts > 1).sum())


def suggestions(frame_no):
    """Yellow 'him?' candidates at this frame (only when he isn't selected here)."""
    if not ss.selected or ss.get("pitch_sample") is None:
        return []
    if tracks[(tracks.frame == frame_no) & tracks.track_id.isin(ss.selected)].shape[0]:
        return []
    team = his_team()
    allowed = {t for t, lab in labels.items() if lab in (team, UNKNOWN)} if team else None
    return suggest_at_frame(tracks, ss.pitch_sample, ss.selected, frame_no, info["fps"],
                            allowed_ids=allowed)


def his_team():
    """Most common team label among the selected IDs (or None)."""
    picked = [labels.get(t) for t in ss.selected if labels.get(t) in (TEAM_A, TEAM_B)]
    return max(set(picked), key=picked.count) if picked else None


# ------------------------------------------------------------------ 3. pick player
st.header("3. Pick your player")
st.write("Click him on the frame. If he gets a new ID later (number next to his box changes), "
         "move the slider there and click him again. Click a green box to unselect it.")

STRIDE = info.get("stride", 1)
last_frame = max(info["frames"] - 1, 0)
last_frame -= last_frame % STRIDE
if "jump_to" in ss:  # buttons ask for a jump; apply it before the slider is drawn
    ss.frame_idx = ss.pop("jump_to")
ss.frame_idx = min(ss.get("frame_idx", 0), last_frame)
frame_idx = st.slider("Frame", 0, last_frame, step=STRIDE, key="frame_idx")
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
        sugg = suggestions(frame_idx)
        hidden -= {t for t, _ in sugg}  # always show (and allow clicking) the yellow guesses
        drawn = draw_boxes(frame, tracks[tracks.frame == frame_idx], ss.selected,
                           colour_of=colour_of, hidden=hidden)
        here = tracks[tracks.frame == frame_idx].set_index("track_id")
        for rank, (tid, _) in enumerate(sugg):
            if tid in here.index:
                r = here.loc[tid]
                r = r.iloc[0] if getattr(r, "ndim", 1) > 1 else r
                p1, p2 = (int(r.x1) - 4, int(r.y1) - 4), (int(r.x2) + 4, int(r.y2) + 4)
                cv2.rectangle(drawn, p1, p2, (0, 215, 255), 4 if rank == 0 else 2)
                cv2.putText(drawn, "him?" if rank == 0 else "?", (p1[0], p1[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
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
            elif tid in ss.selected:
                ss.selected.discard(tid)  # click a green box to remove that ID
                st.rerun()
            else:
                n_added, clash = add_player(tid)
                if clash is not None:
                    st.toast(f"That's a different player: ID {tid} is on screen at the same "
                             f"time as yours (frame {clash}). To switch players, press "
                             "Clear selection first.")
                else:
                    st.toast(f"Added {n_added} ID(s) linked to {tid}.")
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
    n_mixed = mixed_frames() if ss.selected else 0
    if n_mixed:
        st.error(f"Your selection has **two people** on screen at once in {n_mixed} frames "
                 f"({n_mixed * STRIDE / info['fps']:.0f} s). That mixes another player into "
                 "his stats and heatmap.")
        if st.button("Fix it (keep one player)", type="primary",
                     help="Keeps the longest IDs and drops any that overlap with them."):
            old = set(ss.selected)
            ss.selected = set()
            add_ids_safely(old)
            st.toast(f"Kept {len(ss.selected)} of {len(old)} IDs. Check they're all him.")
            st.rerun()
    if ss.selected:
        reasons = pd.Series([player_groups()[1].get(t, "clicked") for t in ss.selected]).value_counts()
        st.caption("Linked by: " + ", ".join(f"{r} ({n})" for r, n in reasons.items()))
        mins = selected_minutes(tracks, ss.selected, info["fps"], STRIDE)
        st.progress(min(1.0, mins / max(info["duration"] / 60, 1e-6)),
                    text=f"Selected: {mins:.1f} of {info['duration'] / 60:.1f} min")
    if st.button("Auto-find his other IDs", disabled=not ss.selected,
                 help="Links IDs on the pitch: same team, close enough to where he was going, "
                      "never on screen at the same time as him. Skips anything ambiguous."):
        n = 0
        player_of = player_groups()[0]
        for t in list(ss.selected):
            n += add_ids_safely(ids_of_player(player_of, t))
        if id_links():
            n += add_ids_safely(chain_of(id_links(), ss.selected))
        allowed = ({t for t, lab in labels.items() if lab in (team, UNKNOWN)} if team else None)
        n += add_ids_safely(auto_stitch(tracks, ss.selected, max_gap_frames=int(info["fps"] * 2),
                                        allowed_ids=allowed))
        st.toast(f"Added {n} ID(s)." if n else "Nothing new found.")
        st.rerun()
    g1, g2 = st.columns(2)
    if g1.button("Next gap", disabled=not ss.selected,
                 help="Jump to the next stretch where he isn't selected, at a moment where "
                      "the app has a guess (yellow box)."):
        rows_sel = player_rows(tracks, ss.selected)
        gaps = [g for g in missing_stretches(rows_sel, info["frames"], min_length=int(info["fps"]))
                if g[1] > frame_idx]
        target = None
        for start, end in gaps:
            step = max(STRIDE, int(info["fps"] / 2))
            for f in range(max(start, frame_idx + 1), end + 1, step):
                f -= f % STRIDE
                if suggestions(f):
                    target = f
                    break
            if target is not None:
                break
        if target is None and gaps:
            target = max(gaps[0][0], frame_idx + STRIDE)
        if target is None:
            st.toast("No more gaps after this frame.")
        else:
            ss.jump_to = target - target % STRIDE
            st.rerun()
    sugg_now = suggestions(frame_idx)
    if g2.button("Yes, that's him", disabled=not sugg_now, type="primary",
                 help="Adds the top yellow box (and every ID linked to it)."):
        n_added, clash = add_player(sugg_now[0][0])
        st.toast(f"Added {n_added} ID(s). Press Next gap to continue." if clash is None else
                 "That one turned out to overlap with him, so it wasn't added.")
        st.rerun()
    if sugg_now:
        st.caption("Yellow box = the app's best guess for him here. If it's wrong, click the "
                   "right player instead (or Next gap if he's off screen).")
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
                ss.jump_to = start - start % STRIDE
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
        st.write(f"Frames checked: {report['checked']} | usable: {report['usable']} | "
                 f"typical landmarks per frame: {report['median_landmarks']:.0f}")
        reasons = {"few_points": "fewer than 5 landmarks found",
                   "collinear": "landmarks all on one line",
                   "no_fit": "landmarks didn't agree with each other",
                   "bad_fit": "landmarks only roughly agreed",
                   "implausible": "result wasn't a sensible camera view",
                   "jumped": "jumped away from neighbouring frames"}
        for k, v in sorted(report["rejected"].items(), key=lambda kv: -kv[1]):
            st.write(f"- {v} frames rejected: {reasons.get(k, k)}")
        st.caption("Tick 'Check pitch mapping' above to see the lines the app is using.")

problems = quality_problems(phys, report) if ss.keypoints else []

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
    if problems:
        st.warning("**Don't trust these numbers yet:** " + "; ".join(problems) + ".")

    st.subheader("Heatmap")
    flip = st.checkbox("Flip pitch (if his team attacks the other way)")
    st.pyplot(pitch_heatmap_v2(traj, length, width, flip=flip, minutes=phys["minutes_tracked"]))
    if phys["minutes_tracked"] < 0.6 * info["duration"] / 60:
        st.caption(f"Only {phys['minutes_tracked']:.1f} of {info['duration'] / 60:.0f} min of him so "
                   "far. Use Auto-find and Next gap above to add the rest.")

    st.subheader(f"Compared with pro {position.lower()}s")
    benchmarks = load_benchmarks()
    comparison, group = compare_to_pros(phys, position, benchmarks)
    if problems:
        st.info("Comparison hidden until the data passes the checks above. A misleading "
                "comparison is worse than none.")
    elif benchmarks is None:
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
    bar = st.progress(0.0, text="Rendering...")
    out = Path(ss.workdir) / "highlight.mp4"
    render_highlight(ss.video_path, rows, out, progress=lambda p: bar.progress(p), stride=STRIDE)
    ss.highlight = str(out)
    bar.empty()
if ss.get("highlight"):
    d3.download_button("Download highlight video", Path(ss.highlight).read_bytes(),
                       file_name="highlight.mp4", mime="video/mp4")
