"""Find a drone's route on a satellite map from its camera frames: the run folder, the trajectory image and video.

    python scripts/run.py --map data/hackatons-data/global_map.png --frames data/hackatons-data/crops --out outputs/run1

Writes into --out: run.json and run.jsonl (one telemetry record per frame), trajectory.png and
trajectory_preview.jpg (the route on the map), trajectory.mp4 (a camera following the drone). The map's features
are cached in --cache, so later runs on the same map start faster. See dronenav.pipeline for the steps.
"""
from __future__ import annotations

import argparse
import sys

import cv2

from dronenav.pipeline import MATCHERS, NothingLocalized, run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--map", required=True, help="the satellite map image")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", help="folder of frames, numbered in flight order (crop_0.png, ...)")
    src.add_argument("--video", help="a video instead of frames (needs the video source, M2-02: not yet)")
    ap.add_argument("--out", required=True, help="run folder to write")
    ap.add_argument("--matcher", default="sift", help=f"feature matcher (available: {', '.join(sorted(MATCHERS))})")
    ap.add_argument("--fps", type=float, default=None,
                    help="frames per second, for the frames' times (default: meta.json next to them, else 1)")
    ap.add_argument("--speed", type=float, default=1.0, help="video: run seconds per video second (default 1)")
    ap.add_argument("--cache", default="outputs/cache", help="where map features are cached (default outputs/cache)")
    a = ap.parse_args(argv)
    if a.video is not None:
        print("run: error: --video needs the video frame source (M2-02), which is not built yet; "
              "extract frames to a folder and use --frames", file=sys.stderr)
        return 2
    try:
        s = run(a.map, a.frames, a.out, matcher=a.matcher, fps=a.fps, speed=a.speed, cache_dir=a.cache)
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
