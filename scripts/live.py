"""Live mode: play a simulated flight in real time, find every frame as it arrives, stream the telemetry.

    python scripts/live.py --flight data/irpin/flights/mild/2022 --map data/irpin/maps/irpin_2022.png --out outputs/live
    python scripts/live.py --flight data/irpin/flights/mild/2022 --map data/irpin/maps/irpin_2022.png --out outputs/live \
        --start 60 --seconds 120 --jam 30:10

The simulator is the pre-rendered flight (``dronenav.sim.FlightSource``), played at its own pace: --rate frames
per second of flight (default 2), one second of flight per second (--speed). The engine localizes each frame when it
arrives and appends its record to --out/run.jsonl, so the console (``cd ui && npm run dev``, then /?run=<name of
--out>) follows the flight while it happens: it reads the new records twice a second and glides the drone between
fixes. A frame the engine is too late for is skipped, as a real camera would not wait.

Chaos is switched on while it runs: the console's "Jam camera" button, or by hand
``echo '{"jam": true}' > OUT/control.json`` (false to stop). --jam START:LENGTH schedules one, in seconds counted from
the first frame played. Ctrl-C stops the flight; the run so far stays a readable run folder. See dronenav.live.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2

from dronenav.config import add_shared_args, init_cli
from dronenav.live import live_run
from dronenav.pipeline import MATCHERS
from dronenav.sim import FlightSource


def _jam(text: str) -> tuple[float, float]:
    """--jam START:LENGTH, both in seconds of flight: START >= 0, LENGTH > 0 (as in run.py)."""
    start, length = (float(v) for v in text.split(":"))      # not two numbers: ValueError, a usage error
    if not (math.isfinite(start) and math.isfinite(length) and start >= 0 and length > 0):
        raise ValueError(text)
    return start, length


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flight", required=True, help="a rendered flight folder (flight.mp4, truth.jsonl, meta.json)")
    ap.add_argument("--map", required=True, help="the satellite map image the engine searches")
    ap.add_argument("--out", required=True, help="run folder to write; the console shows it as /?run=<its name>")
    ap.add_argument("--rate", type=float, default=2.0, help="frames per second of flight sent to the engine (default 2)")
    ap.add_argument("--speed", type=float, default=1.0, help="flight seconds played per second (default 1: real time)")
    ap.add_argument("--start", type=float, default=0.0, metavar="S", help="start at this second of the flight")
    ap.add_argument("--seconds", type=float, default=None, metavar="S", help="play this many seconds (default: to the end)")
    ap.add_argument("--keyframes", action="store_true", help="read the flight's frames/ folder instead of flight.mp4")
    ap.add_argument("--matcher", default="sift", help=f"feature matcher (available: {', '.join(sorted(MATCHERS))})")
    ap.add_argument("--jam", type=_jam, default=None, metavar="START:LENGTH",
                    help="jam the camera for LENGTH seconds, START seconds after the first frame played")
    ap.add_argument("--cache", default="outputs/cache", help="where map features are cached (default outputs/cache)")
    add_shared_args(ap)
    a = ap.parse_args(argv)
    device = init_cli(a)
    out = Path(a.out)
    try:
        if a.seconds is not None and not (math.isfinite(a.seconds) and a.seconds > 0):
            raise ValueError(f"--seconds must be a positive number, got {a.seconds!r}")
        source = FlightSource(a.flight, fps=a.rate, keyframes=a.keyframes, start_s=a.start,
                              end_s=None if a.seconds is None else a.start + a.seconds)
        # the jam is counted from the first frame played, the source's clock from the flight's start
        jam = None if a.jam is None else (a.start + a.jam[0], a.jam[1])
        print(f"live: playing {a.flight} at {source.rate:g} frames/s -> {out} (console: /?run={out.name}; "
              f"Ctrl-C stops)", flush=True)
        s = live_run(a.map, source, out, matcher=a.matcher, cache_dir=a.cache, device=device, speed=a.speed, jam=jam)
    except KeyboardInterrupt:
        # live_run has marked the run "stopped"; what was flown is a whole run folder
        print(f"live: stopped -> {out}")
        return 0
    except (OSError, ValueError, RuntimeError, cv2.error) as e:   # missing inputs, bad options
        print(f"live: error: {e}", file=sys.stderr)
        return 2
    print(f"live: {s.fixes} fixes on {s.frames} frames in {s.seconds:.1f} s ({s.fixes_per_s:.2f} fixes/s, "
          f"{s.skipped} frames skipped) -> {s.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
