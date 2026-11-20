"""Write a fake run folder over a map, so renderers and the UI can be built with no engine (M0-04).

    python scripts/fake_run.py --map data/hackatons-data/global_map.png --out outputs/fake

The folder gets run.json (which map, frame size, fps, seed), run.jsonl (one telemetry record per frame) and
frames/ (what the camera sees). A jam in the middle takes the tracker through LOST and RELOCALIZING; see
dronenav.fake_run for the script.
"""
from __future__ import annotations

import argparse
import sys

from dronenav.fake_run import FakeRunConfig, make_fake_run


def _frame_size(text: str) -> tuple[int, int]:
    """'640' or '640x480'."""
    parts = text.lower().split("x")
    try:
        w, h = (int(parts[0]), int(parts[0])) if len(parts) == 1 else (int(parts[0]), int(parts[1]))
    except ValueError:
        raise argparse.ArgumentTypeError(f"frame size must be W or WxH, got {text!r}") from None
    return w, h


def main(argv: list[str] | None = None) -> int:
    d = FakeRunConfig()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--map", required=True, help="map image the route is flown over")
    ap.add_argument("--out", required=True, help="run folder to write (created; its old frames are replaced)")
    ap.add_argument("--frames", type=int, default=d.frames, help=f"number of frames (default {d.frames})")
    ap.add_argument("--fps", type=float, default=d.fps, help=f"frames per second (default {d.fps:g})")
    ap.add_argument("--seed", type=int, default=d.seed, help=f"random seed (default {d.seed})")
    ap.add_argument("--frame-size", type=_frame_size, default=d.frame_size,
                    help=f"W or WxH in px (default {d.frame_size[0]}x{d.frame_size[1]})")
    ap.add_argument("--scale", type=float, default=d.scale, help=f"map px per frame px (default {d.scale:g})")
    ap.add_argument("--jam-start", type=int, default=None, help="first jammed frame (default: 40%% into the run)")
    ap.add_argument("--jam-len", type=int, default=d.jam_len, help=f"jammed frames (default {d.jam_len})")
    a = ap.parse_args(argv)
    cfg = FakeRunConfig(frames=a.frames, fps=a.fps, seed=a.seed, frame_size=a.frame_size, scale=a.scale,
                        jam_start=a.jam_start, jam_len=a.jam_len)
    try:
        info = make_fake_run(a.map, a.out, cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"fake_run: error: {e}", file=sys.stderr)
        return 2
    print(f"fake_run: {cfg.frames} frames over {info.map} ({info.map_size[0]}x{info.map_size[1]}) -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
