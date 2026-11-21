"""Find a drone's route on a satellite map from its camera frames: the run folder, the trajectory image and video.

    python scripts/run.py --map data/hackatons-data/global_map.png --frames data/hackatons-data/crops --out outputs/run1
    python scripts/run.py --map data/map.png --video data/video/take.mp4 --every 10 --out outputs/phone

Writes into --out: run.json and run.jsonl (one telemetry record per frame), trajectory.png and
trajectory_preview.jpg (the route on the map), trajectory.mp4 (a camera following the drone); from a video, also
frames/ (the sampled frames the records point to). The map's features are cached in --cache, so later runs on the
same map start faster. See dronenav.pipeline for the steps.
"""
from __future__ import annotations

import argparse
import sys

import cv2

from dronenav.config import add_shared_args, init_cli
from dronenav.pipeline import MATCHERS, TRACKERS, VIDEO_RATE, NothingLocalized, run


def _count(text: str) -> int:
    """--every: a whole number >= 1 (argparse turns the ValueError into a usage error, exit 2)."""
    n = int(text)
    if n < 1:
        raise ValueError(text)
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--map", required=True, help="the satellite map image")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", help="folder of frames, numbered in flight order (crop_0.png, ...)")
    src.add_argument("--video", help="a video (mp4, mov, ...) instead of a folder of frames")
    ap.add_argument("--out", required=True, help="run folder to write")
    ap.add_argument("--matcher", default="sift", help=f"feature matcher (available: {', '.join(sorted(MATCHERS))})")
    ap.add_argument("--fps", type=float, default=None,
                    help="the rate of the numbers in the frame names: t = number / fps (keyframes frame_00030 of a "
                         "30 fps video: 30). Default: meta.json next to the frames, else 1")
    ap.add_argument("--every", type=_count, default=None, metavar="N",
                    help=f"video: process every N-th frame (default: from the video's fps, about {VIDEO_RATE:g} "
                         f"frames/s)")
    ap.add_argument("--speed", type=float, default=1.0, help="video: run seconds per video second (default 1)")
    ap.add_argument("--tracker", default="kalman",
                    help=f"kalman: the Kalman tracker's states and filtered positions (M5); none: M1's interim "
                         f"states, raw fixes (available: {', '.join(TRACKERS)})")
    ap.add_argument("--cache", default="outputs/cache", help="where map features are cached (default outputs/cache)")
    add_shared_args(ap)
    a = ap.parse_args(argv)
    # Picked and logged like in every matching script; the SIFT matcher runs on the CPU, a torch matcher takes this device.
    init_cli(a)
    if a.every is not None and a.video is None:
        ap.error("--every samples a --video; a folder of frames is taken whole")
    if a.fps is not None and a.video is not None:
        ap.error("--fps is the rate of a folder's frame numbers; a --video has its own (sample it with --every)")
    try:
        s = run(a.map, a.frames, a.out, matcher=a.matcher, fps=a.fps, speed=a.speed, cache_dir=a.cache,
                tracker=a.tracker, video=a.video, every=a.every)
    except NothingLocalized as e:
        print(f"run: error: {e}", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, cv2.error) as e:   # missing inputs, bad options, encoder trouble
        print(f"run: error: {e}", file=sys.stderr)
        return 2
    print(f"run: {s.localized} of {s.frames} frames localized in {s.seconds:.1f} s -> {s.out_dir} "
          f"({s.image.name}, {s.video.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
