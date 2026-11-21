"""Live mode: play a simulated flight in real time, find every frame as it arrives, stream the telemetry.

    python scripts/live.py --flight data/irpin/flights/mild/2022 --map data/irpin/maps/irpin_2022.png --out outputs/live
    python scripts/live.py --flight data/irpin/flights/mild/2022 --map data/irpin/maps/irpin_2022.png --out outputs/live \
        --start 60 --seconds 120 --jam 30:10

The simulator is the pre-rendered flight (``dronenav.sim.FlightSource``), played at its own pace: --rate frames
per second of flight (default 2), one second of flight per second (--speed). The engine localizes each frame when it
arrives and appends its record to --out/run.jsonl, so the console (``cd ui && npm run dev``, then /?run=<name of
--out>) follows the flight while it happens: it reads the new records twice a second and glides the drone between
fixes. A frame the engine is too late for is skipped, as a real camera would not wait.

The judge controls (M7-05) reach the flight while it runs, through OUT/control.json (the console's buttons write it,
or by hand ``echo '{"jam": true}' > OUT/control.json``):

    jam camera     {"jam": true}       every frame is noise until false; --jam START:LENGTH schedules one, in seconds
                                       counted from the first frame played
    spoof GPS      {"spoof": true}     the simulated GPS receiver walks away from the truth at --spoof-speed px/s; the
                                       spoof detector's verdict is in every record (extra.gps) and in the console
    map year       {"map": "2017"}     search another map of the same place: --maps, by default the maps that lie next
                                       to --map and differ in the year only (irpin_2017.png, irpin_2022.png, ...)
    new flight     {"flight": {"name": "wild/2022", "speed": 2, "id": 1}}
                                       land and start another pre-rendered flight: --flights, by default the flights
                                       that lie next to --flight (<root>/<preset>/<year>)

Ctrl-C stops the flight; the run so far stays a readable run folder. See dronenav.live.
"""
from __future__ import annotations

import argparse
import math
import re
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


_YEAR = re.compile(r"^(.*)_(\d{4})$")


def _map_name(path: Path) -> str:
    """A map's name on its button: the year its file name ends in (irpin_2022.png: "2022"), else the file name."""
    m = _YEAR.match(path.stem)
    return m.group(2) if m else path.stem


def _maps(map_path: Path, given: list[str] | None) -> dict[str, Path] | None:
    """The maps the operator can switch between, by name; None when there is only --map.

    --maps items are NAME=PATH or a path (named by :func:`_map_name`), and --map is always one of them. Without
    --maps: the files next to --map whose names differ from it in the year only, so a pack of the same town in
    several years needs no option. Two maps under one name are a usage error."""
    if given is None:
        m = _YEAR.match(map_path.stem)
        if not m:
            return None
        found = [p for p in sorted(map_path.parent.glob(f"{glob_escape(m.group(1))}_[0-9][0-9][0-9][0-9]{map_path.suffix}"))
                 if p.is_file()]
        named = [(_map_name(p), p) for p in found if p.resolve() != map_path.resolve()]
    else:
        named = []
        for item in given:
            name, sep, rest = item.partition("=")
            named.append((name, Path(rest)) if sep and name and rest else (_map_name(Path(item)), Path(item)))
    maps = {_map_name(map_path): map_path}
    for name, path in named:
        if path.resolve() == map_path.resolve():
            continue
        if name in maps:
            raise ValueError(f"two maps are named {name!r}: {maps[name]} and {path} (name them: --maps NAME=PATH)")
        maps[name] = path
    return maps if len(maps) > 1 else None


def glob_escape(text: str) -> str:
    return re.sub(r"([*?\[\]])", r"[\1]", text)


def _flight_name(path: Path) -> str:
    """A flight's name: its last two folders, "<preset>/<year>" in a flight pack."""
    path = path.resolve()
    return f"{path.parent.name}/{path.name}"


def _flights(flight: Path, root: Path | None) -> dict[str, Path]:
    """The flights the operator can start, by name: every <root>/<preset>/<year> that is a rendered flight, and
    --flight itself. The default root is the pack --flight lies in."""
    root = flight.resolve().parent.parent if root is None else root
    found = {_flight_name(d): d for d in sorted(root.glob("*/*"))
             if (d / "meta.json").is_file() and (d / "truth.jsonl").is_file()}
    return found | {_flight_name(flight): flight}


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
    ap.add_argument("--maps", nargs="+", default=None, metavar="[NAME=]PATH",
                    help="the maps the operator can switch between, all of one size "
                         "(default: the files next to --map that differ in the year only)")
    ap.add_argument("--flights", default=None, metavar="ROOT",
                    help="a folder of flights (<preset>/<year>) the operator can start (default: the one --flight is in)")
    ap.add_argument("--spoof-speed", type=float, default=8.0, metavar="PX_S",
                    help="how fast a spoofed GPS reading walks away from the truth, map px per second (default 8)")
    ap.add_argument("--cache", default="outputs/cache", help="where map features are cached (default outputs/cache)")
    add_shared_args(ap)
    a = ap.parse_args(argv)
    device = init_cli(a)
    out = Path(a.out)
    try:
        if a.seconds is not None and not (math.isfinite(a.seconds) and a.seconds > 0):
            raise ValueError(f"--seconds must be a positive number, got {a.seconds!r}")

        def opener(folder: Path):
            # every flight is played the same way: the same rate and the same stretch of it
            return lambda: FlightSource(folder, fps=a.rate, keyframes=a.keyframes, start_s=a.start,
                                        end_s=None if a.seconds is None else a.start + a.seconds)

        flight = Path(a.flight)
        source = opener(flight)()
        flights = {name: opener(folder) for name, folder in
                   _flights(flight, None if a.flights is None else Path(a.flights)).items()}
        maps = _maps(Path(a.map), a.maps)
        # the jam is counted from the first frame played, the source's clock from the flight's start
        jam = None if a.jam is None else (a.start + a.jam[0], a.jam[1])
        print(f"live: playing {a.flight} at {source.rate:g} frames/s -> {out} (console: /?run={out.name}; "
              f"Ctrl-C stops)", flush=True)
        s = live_run(a.map, source, out, matcher=a.matcher, cache_dir=a.cache, device=device, speed=a.speed, jam=jam,
                     maps=maps, flights=flights, flight=_flight_name(flight), spoof_speed=a.spoof_speed)
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
