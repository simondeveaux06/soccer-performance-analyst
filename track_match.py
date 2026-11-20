"""Track a whole match (or half) from the terminal and save the results.

    python track_match.py first_half.mp4            # normal
    python track_match.py first_half.mp4 --fast     # about 2x quicker (every 2nd frame)

Results go to runs/<video name>/. Open them in the website under
"Open a tracked match", with no waiting.

Safe to stop: press Ctrl+C (or if the computer sleeps / it crashes), then run the
same command again and it carries on from where it stopped.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))
from analysis import track_players, video_info  # noqa: E402
from pitch import DEFAULT_MODEL_PATH, PitchDetector  # noqa: E402
from storage import clear_run, load_run, read_meta, run_dir_for, save_chunk, write_meta  # noqa: E402

CHECKPOINT_EVERY = int(os.environ.get("SPA_CHECKPOINT_EVERY", 1500))  # frames between saves


def main():
    ap = argparse.ArgumentParser(description="Track a match video and save the results.")
    ap.add_argument("video", help="path to the video file")
    ap.add_argument("--fast", action="store_true", help="only check every 2nd frame (~2x faster)")
    ap.add_argument("--stride", type=int, default=None, help="check every Nth frame (overrides --fast)")
    ap.add_argument("--conf", type=float, default=0.30, help="detection confidence (default 0.30)")
    ap.add_argument("--no-pitch", action="store_true", help="skip pitch mapping")
    ap.add_argument("--restart", action="store_true", help="ignore any saved progress, start over")
    args = ap.parse_args()

    video = Path(args.video).resolve()
    if not video.exists():
        sys.exit(f"Can't find {video}")
    info = video_info(video)
    if not info["frames"]:
        sys.exit(f"Can't read {video} as a video")
    stride = args.stride or (2 if args.fast else 1)
    use_pitch = not args.no_pitch and DEFAULT_MODEL_PATH.exists()
    if not args.no_pitch and not use_pitch:
        print("Note: pitch model not found (run: bash download_models.sh), so no pitch mapping.")

    run_dir = run_dir_for(video)
    settings = {"stride": stride, "conf": args.conf, "pitch": use_pitch}
    meta = read_meta(run_dir)
    start, id_offset = 0, 0
    if meta and not args.restart and meta.get("video") == str(video) and meta.get("settings") == settings:
        if meta["status"] == "done":
            print(f"Already tracked: {run_dir}\nOpen it in the website, or add --restart to redo it.")
            return
        tracks, _, _ = load_run(run_dir)
        start = int(meta["next_frame"])
        id_offset = int(tracks.track_id.max()) + 1 if len(tracks) else 0
        print(f"Resuming from frame {start} of {info['frames']} "
              f"({100 * start / info['frames']:.0f}% done earlier).")
    else:
        clear_run(run_dir)

    meta = {"video": str(video), "name": video.stem, "fps": info["fps"], "frames": info["frames"],
            "width": info["width"], "height": info["height"], "settings": settings,
            "stride": stride, "status": "partial", "next_frame": start}
    write_meta(run_dir, meta)

    def checkpoint(new_tracks, new_keypoints, next_frame):
        save_chunk(run_dir, new_tracks, new_keypoints)
        meta["next_frame"] = int(next_frame)
        write_meta(run_dir, meta)

    t0, last = time.time(), [0.0]

    def progress(p):
        now = time.time()
        if now - last[0] < 2 and p < 1:
            return
        last[0] = now
        done_now = p - start / info["frames"]
        eta = (now - t0) / done_now * (1 - p) if done_now > 0 else 0
        print(f"\r  {p:6.1%}  |  about {eta / 60:4.0f} min left   ", end="", flush=True)

    mode = "fast mode, every 2nd frame" if stride == 2 else f"every {stride} frames" if stride > 1 else "every frame"
    print(f"Tracking {video.name}: {info['duration'] / 60:.0f} min of video, {mode}.")
    print("Ctrl+C stops safely; run the same command again to continue.")
    try:
        track_players(video, conf=args.conf, progress=progress,
                      pitch_detector=PitchDetector() if use_pitch else None,
                      stride=stride, start_frame=start, id_offset=id_offset,
                      checkpoint=checkpoint, checkpoint_every=CHECKPOINT_EVERY)
    except KeyboardInterrupt:
        print(f"\nStopped at frame {meta['next_frame']}. Progress saved; "
              "run the same command again to continue.")
        return

    meta["status"] = "done"
    meta["next_frame"] = info["frames"]
    write_meta(run_dir, meta)
    print(f"\nDone in {(time.time() - t0) / 60:.0f} min. Saved to {run_dir}")
    print("Open the website and choose 'Open a tracked match'.")


if __name__ == "__main__":
    main()
