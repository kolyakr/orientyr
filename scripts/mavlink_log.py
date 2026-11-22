"""Turn a run's fixes into MAVLink ``GPS_INPUT`` messages, the GPS replacement an autopilot reads (M9-01, a stub).

    python scripts/mavlink_log.py --run outputs/irpin --utc-start 2025-06-01T10:00:00Z
    python scripts/mavlink_log.py --run outputs/irpin --udp 127.0.0.1:14550   # also feed a SITL simulator

Writes ``<run>/gps_input.tlog`` (replay it with MAVExplorer or Mission Planner). The run's map needs a
georeference sidecar (the Irpin maps have one; the task map has none, so it is refused). A stub: the messages
follow the MAVLink spec and read back cleanly, but no autopilot has flown on them. See dronenav.mavlink.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from dronenav.config import add_shared_args, init_cli
from dronenav.georef import load_georef
from dronenav.mavlink import STUB_TEXT, run_to_tlog
from dronenav.telemetry import iter_run, read_run_info


def _utc(text: str) -> dt.datetime:
    """An ISO 8601 time; without a zone it is taken as UTC."""
    try:
        when = dt.datetime.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an ISO 8601 time: {text!r}") from None
    return when.replace(tzinfo=dt.UTC) if when.tzinfo is None else when.astimezone(dt.UTC)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, type=Path, help="run folder (run.json + run.jsonl)")
    ap.add_argument("--out", type=Path, default=None, help="the .tlog to write (default <run>/gps_input.tlog)")
    ap.add_argument("--utc-start", type=_utc, default=None,
                    help="UTC time of the run's t = 0, ISO 8601 (default: now)")
    ap.add_argument("--udp", default=None, metavar="HOST:PORT", help="also send the messages there (a SITL)")
    add_shared_args(ap, device=False)
    a = ap.parse_args(argv)
    init_cli(a)
    out = a.out or a.run / "gps_input.tlog"
    start = a.utc_start or dt.datetime.now(dt.UTC)
    try:
        info = read_run_info(a.run)
        sidecar = info.map_path.with_suffix(".json")
        if not sidecar.exists():
            raise ValueError(f"the map {info.map_path} has no georeference ({sidecar.name} next to it), "
                             "so its pixels have no lat/lon")
        georef = load_georef(sidecar)
        # the sidecar describes two resolutions of the area: the run's pixels are the one the size matches
        blocks = [b for b in (georef.map, georef.hires) if tuple(b.size_px) == tuple(info.map_size)]
        if not blocks:
            raise ValueError(f"no block of the georeference {sidecar.name} has the map's size "
                             f"{info.map_size[0]}x{info.map_size[1]}")
        n = run_to_tlog(iter_run(a.run / "run.jsonl"), blocks[0], out, start, udp=a.udp)
    except (OSError, ValueError) as e:     # missing run, malformed telemetry or sidecar, no georeference
        print(f"mavlink_log: error: {e}", file=sys.stderr)
        return 2
    print(f"mavlink_log: {n} GPS_INPUT messages -> {out}" + (f" and udp {a.udp}" if a.udp else ""))
    print(f"mavlink_log: {STUB_TEXT}: valid MAVLink, never flown with a real autopilot")
    return 0


if __name__ == "__main__":
    sys.exit(main())
