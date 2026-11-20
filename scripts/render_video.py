"""Render a run as a video: a camera that follows the drone over the map, with the trail, the footprint, the
frame it saw and a HUD (M1-08).

    python scripts/render_video.py --run outputs/fake

Writes trajectory.mp4 (H.264, plays in QuickTime and browsers) into the run folder, or --out.
See dronenav.viz.trajectory_video for what each frame shows.
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2

from dronenav.config import add_shared_args, init_cli
from dronenav.viz.trajectory_video import write_video


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"size must be WxH, e.g. 1280x720, got {text!r}") from None
    return w, h


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run folder (run.json, run.jsonl, frames/)")
    ap.add_argument("--out", default=None, help="folder for trajectory.mp4 (default: the run folder)")
    ap.add_argument("--speed", type=float, default=1.0, help="run seconds per video second (default 1: real time)")
    ap.add_argument("--fps", type=float, default=30.0, help="video frames per second (default 30)")
    ap.add_argument("--size", type=_size, default=(1280, 720), help="video size WxH, even (default 1280x720)")
    add_shared_args(ap, device=False)   # only draws and writes files: logging yes, no device to pick
    a = ap.parse_args(argv)
    init_cli(a)
    t0 = time.perf_counter()
    try:
        path = write_video(a.run, a.out, fps=a.fps, speed=a.speed, size=a.size)
    except (OSError, ValueError, RuntimeError, cv2.error) as e:   # missing run/map, bad options, no encoder
        print(f"render_video: error: {e}", file=sys.stderr)
        return 2
    print(f"render_video: {path} in {time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
