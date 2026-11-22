"""Live mode (M7-04): the simulator plays a flight in real time, the engine finds each frame as it arrives, and the
telemetry is written record by record, so the console follows the flight while it happens.

    source (FlightSource: a pre-rendered     ─► pacing ─► chaos, switched ─► pipeline.Engine ─► run.jsonl, appended
    Irpin flight; any FrameSource in tests)    (clock)    on live (jam)      (search, tracker)   + frames/ + live.json

**Pacing.** A frame with flight time ``t`` is handed to the engine ``(t - t_first) / speed`` seconds after the
start, never earlier: the engine sees the flight at the rate a real camera would send it. A real camera does not
wait for a slow engine either. When the engine is more than one frame period behind, the frames it has missed are
dropped (counted in ``skipped``) and it goes on with the newest one, so the position on screen is never older than
one frame period plus one search. The clock and the sleep are parameters: tests run a whole flight in no time.

**The run folder** is the ordinary one (``dronenav.telemetry``), written in the order a reader needs it:
``run.json`` first, then for every frame its image (``frames/frame_<idx>.jpg``: what the engine saw, a jam's noise
too), then its line in ``run.jsonl`` (flushed), then ``live.json``. Each record carries ``extra["wall_s"]`` (seconds
since the start when it was made), ``extra["truth"]`` when the source knows it, and the fields of ``run.py``'s
records. A frame before the first fix has no position: live, there is no later fix to borrow one from.

**live.json** (``STATUS``) is the heartbeat the console reads: ``state`` ("starting" while the map's features load,
"running", then "done" or "stopped"), ``frames``, ``fixes``, ``skipped``, ``fixes_per_s``, ``t`` and what the
operator has switched (below). It is replaced atomically after every record, so its modification time says the engine is alive.

**control.json** (``CONTROL``) holds the operator's orders while the flight runs (M7-05: the judge controls). It is
read before every frame, so an order takes effect on the next one. A key that is missing, or whose value is not an
order, keeps what was ordered before; so does a file that is missing or half-written. The console's buttons write
it; so does ``echo '{"jam": true}' > RUN/control.json``.

- ``"jam": true | false``: every frame arrives as noise (``dronenav.sim.chaos.Jam``) until it is switched off.
  ``jam=(start_s, length_s)`` schedules a jam as in ``run.py``; the two add up.
- ``"spoof": true | false``: the GPS receiver is taken over. A simulated flight knows where the drone really is, so
  it has a simulated receiver (``dronenav.sim.gps``'s model: the truth plus seeded Gaussian noise). While the spoof is
  on, the reading walks away from the truth at ``spoof_speed`` map px per second of flight, starting from zero: no
  jump to notice. The spoof detector (``dronenav.tracking.spoof``, M5-07) compares every reading with the tracker
  and puts its verdict into ``extra["gps"]``; switched off, the receiver is honest again at once and the alarm
  drops after the detector's good frames in a row. The engine never uses the receiver for the position.
- ``"map": "<name>"``: search another map of the same place (``maps``: name to path, e.g. the Irpin pack's years).
  All of them are loaded at the start (they must have one size: the track is kept in map px), so the switch costs
  nothing; ``run.json`` is rewritten to name the new map, and each record says which one it was found on
  (``extra["map"]``). A name that is not offered changes nothing.
- ``"flight": {"name": "<name>", "speed": <number>, "id": <integer>}``: land and start another flight (``flights``:
  name to a function that opens its frame source) at ``speed``. The run starts again in the same folder:
  ``run.jsonl`` from its first line, a new track, the counts at zero, ``flight_no`` one higher. An order is flown
  once: the next one needs another ``id``.

``live.json`` reports ``jam``, ``spoof``, ``map``, ``maps``, ``flight``, ``flights``, ``speed`` and ``flight_no``.

``fixes_per_s`` is the spec's number: accepted fixes per second of wall time, counted between the first and the
last fix (``(fixes - 1) / (wall of the last - wall of the first)``), so a short run is not flattered by its ends.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dronenav.io import _is_rate
from dronenav.pipeline import FRAMES_DIR, Engine, _check_out, _jam, tracked_record
from dronenav.sim.chaos import Jam, degrade
from dronenav.sim.gps import GpsSample, WalkOff
from dronenav.telemetry import Record, RunInfo, RunWriter, write_run_info
from dronenav.tracking.spoof import SpoofConfig, SpoofDetector, gps_extra
from dronenav.types import FrameSource

__all__ = ["CONTROL", "STATUS", "FlightOrder", "LiveSummary", "Orders", "live_run", "read_control", "read_orders"]

STATUS = "live.json"         # the engine's heartbeat, in the run folder
CONTROL = "control.json"     # the operator's orders, in the run folder
GENERATOR = "live.py"


@dataclass(frozen=True)
class LiveSummary:
    out_dir: Path
    frames: int               # frames the engine was given (one record each)
    fixes: int                # of them, with an accepted fix
    skipped: int              # frames dropped because the engine was behind
    seconds: float            # wall time from the first frame to the last record
    fixes_per_s: float        # see the module docstring; 0 with fewer than two fixes


@dataclass(frozen=True)
class FlightOrder:
    """Start flight ``name`` at ``speed``; ``id`` tells one order from the next."""

    name: str
    speed: float
    id: int


@dataclass(frozen=True)
class Orders:
    """What the operator has ordered so far (``control.json``). map: a name, None = the map the run started on.
    flight: the last flight order, None = none yet."""

    jam: bool = False
    spoof: bool = False
    map: str | None = None
    flight: FlightOrder | None = None


def _flight_order(v) -> FlightOrder | None:
    if not isinstance(v, dict):
        return None
    name, speed, oid = v.get("name"), v.get("speed"), v.get("id")
    if not isinstance(name, str) or not _is_rate(speed) or isinstance(oid, bool) or not isinstance(oid, int):
        return None
    return FlightOrder(name, float(speed), oid)


def read_orders(run_dir: str | Path, last: Orders = Orders()) -> Orders:
    """``control.json`` on top of ``last``: each key that holds an order replaces its part, everything else (no
    file, half a file, another shape, a value that is not an order) keeps what was ordered before, so a write caught
    halfway never flips a switch."""
    try:
        data = json.loads((Path(run_dir) / CONTROL).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return last
    if not isinstance(data, dict):
        return last
    jam, spoof, name = data.get("jam"), data.get("spoof"), data.get("map")
    return Orders(jam=jam if isinstance(jam, bool) else last.jam,
                  spoof=spoof if isinstance(spoof, bool) else last.spoof,
                  map=name if isinstance(name, str) else last.map,
                  flight=_flight_order(data.get("flight")) or last.flight)


def read_control(run_dir: str | Path, last: bool = False) -> bool:
    """Is the jam ordered on? The jam switch alone of :func:`read_orders` (M7-04's reader, kept for its callers)."""
    return read_orders(run_dir, Orders(jam=last)).jam


class _Receiver:
    """The simulated GPS receiver of a live flight and the detector that checks it against the tracker.

    ``dronenav.sim.gps``'s model, one frame at a time: the true position plus Gaussian noise (seeded), and, while
    the spoof is on, a displacement that grows from zero at ``speed`` px per second of flight in one direction."""

    def __init__(self, speed: float, cfg: SpoofConfig | None = None, seed: int = 0, direction_deg: float = 60.0):
        self.cfg = cfg if cfg is not None else SpoofConfig()
        self.walk = WalkOff(0, speed, direction_deg)          # validates the speed; start_idx is not used live
        self._seed = seed
        self.restart()

    def restart(self) -> None:
        self._rng = np.random.default_rng(self._seed)
        self.detector = SpoofDetector(self.cfg)
        self._since: float | None = None                      # flight time the spoof was switched on at

    def read(self, f, center, spoof: bool, rec: Record) -> dict:
        """The reading for frame ``f`` (true position ``center``), checked against ``rec``'s track: the block for
        ``extra["gps"]``."""
        nx, ny = self._rng.normal(0.0, self.cfg.gps_sigma_px, 2)
        self._since = (f.t if self._since is None else self._since) if spoof else None
        offset = 0.0 if self._since is None else self.walk.speed_px_s * (f.t - self._since)
        ux, uy = self.walk.unit
        sample = GpsSample(idx=f.idx, t=f.t, x=center[0] + float(nx) + offset * ux, y=center[1] + float(ny) + offset * uy,
                           spoofed=spoof, offset_px=offset)
        tr = rec.track
        at = None if tr.x is None or tr.cov is None else (tr.x, tr.y)
        return gps_extra(sample, self.detector.update((sample.x, sample.y), at, tr.cov))


def _write_status(out_dir: Path, **status) -> None:
    """Replace live.json in one step: a reader sees the old status or the new one, never half of one."""
    part = out_dir / f".{STATUS}.partial"
    part.write_text(json.dumps(status), encoding="utf-8")
    os.replace(part, out_dir / STATUS)


def _rate(source) -> float:
    """Frames per second of flight the source gives: FlightSource's ``rate``, a folder's or a cut-out's ``fps``."""
    for name in ("rate", "fps"):
        v = getattr(source, name, None)
        if _is_rate(v):
            return float(v)
    raise ValueError("the frame source does not say its frame rate (rate or fps)")


def _fix_rate(first: float | None, last: float | None, fixes: int) -> float:
    return (fixes - 1) / (last - first) if fixes >= 2 and last > first else 0.0


def live_run(map_path: str | Path, source: FrameSource, out_dir: str | Path, *, matcher: str = "sift",
             cache_dir: str | Path | None = "outputs/cache", device: str | None = None, speed: float = 1.0,
             jam: tuple[float, float] | None = None, maps: Mapping[str, str | Path] | None = None,
             flights: Mapping[str, Callable[[], FrameSource]] | None = None, flight: str | None = None,
             spoof_speed: float = 8.0, clock: Callable[[], float] = time.monotonic,
             sleep: Callable[[float], None] = time.sleep,
             on_record: Callable[[Record], None] | None = None) -> LiveSummary:
    """Play ``source`` in real time (``speed`` flight seconds per second) through the engine and write the run
    into ``out_dir`` as it is made (see the module docstring). ``on_record`` is called after each record is on disk.

    The operator's choices (M7-05): ``maps`` are the maps that can be switched to, by name (``map_path`` is one of
    them, or joins them under its file name); ``flights`` open the flights that can be started, by name, and
    ``flight`` is the name of ``source`` among them; ``spoof_speed`` is how fast a spoofed GPS reading walks away,
    map px per second of flight. The summary is of the last flight flown.
    Bad options, a missing map and a map of another size are refused before anything is written."""
    out_dir = Path(out_dir)
    if not _is_rate(speed):
        raise ValueError(f"speed must be a positive number, got {speed!r}")
    if not _is_rate(spoof_speed):
        raise ValueError(f"spoof_speed must be a positive number (map px per second), got {spoof_speed!r}")
    _check_out(out_dir)
    scheduled = _jam(jam)
    flights = dict(flights or {})
    if flight is not None and flight not in flights:
        raise ValueError(f"the flight {flight!r} is not among the flights {sorted(flights)}")
    rate = _rate(source)
    existed = out_dir.exists()
    engine = Engine(map_path, matcher, None if cache_dir is None else Path(cache_dir), "kalman", device)
    # every map the operator may switch to is made ready now: a switch in the air must not cost a frame
    packs, map_name = {}, None
    for name, path in (maps or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"every map needs a name, got {name!r}")
        if map_name is None and _same_file(path, map_path):
            packs[name], map_name = engine.pack(), name
        else:
            packs[name] = engine.load_map(path)
    if map_name is None:
        map_name = Path(map_path).stem
        packs[map_name] = engine.pack()
    first_map = map_name
    # switched on by the operator: covers every frame while it is applied; seeded, so the noise is repeatable
    operator = Jam(1.0, start_s=0.0, max_s=1e12)
    receiver = _Receiver(spoof_speed)

    out_dir.mkdir(parents=True, exist_ok=True)
    # an old run in this folder: its orders are not for this flight, and its status must not read as ours
    for name in (CONTROL, STATUS):
        (out_dir / name).unlink(missing_ok=True)
    counts = {}
    chosen = {"maps": sorted(packs), "flights": sorted(flights), "flight": flight, "flight_no": 0}
    h, w = engine.map.shape[:2]
    state = "stopped"                             # what the status says if anything below raises
    start = t_first = first_fix = last_fix = None
    orders = Orders()
    flown = None                                  # the id of the last flight order that was flown

    def status(state: str, **more) -> None:
        _write_status(out_dir, state=state, **(counts | chosen | {"map": map_name, "speed": speed} | more))

    def info() -> None:
        frame_size = getattr(source, "size", None) or getattr(source, "frame_size", None)
        write_run_info(out_dir, RunInfo(map=str(engine.map_path), map_size=(w, h), generator=GENERATOR,
                                        frame_size=None if frame_size is None else tuple(frame_size), fps=rate))

    try:
        while True:                               # one turn per flight: the first, then each one ordered
            counts = {"frames": 0, "fixes": 0, "skipped": 0, "fixes_per_s": 0.0, "jam": False, "spoof": False,
                      "t": None}
            start = t_first = first_fix = last_fix = None
            period = 1.0 / (rate * speed)         # wall seconds between two frames
            ordered = None
            with RunWriter(out_dir / "run.jsonl") as writer:
                info()
                status("starting")
                for f in source:
                    if start is None:
                        start, t_first = clock(), f.t
                    due = start + (f.t - t_first) / speed
                    now = clock()
                    if now < due:
                        sleep(due - now)
                    elif now - due >= period:     # the next frame is due already: this one is history
                        counts["skipped"] += 1
                        continue
                    orders = read_orders(out_dir, orders)
                    new = orders.flight
                    if new is not None and new.id != flown:
                        flown = new.id            # an order is looked at once, whether it can be flown or not
                        if new.name in flights:
                            ordered = new
                            break
                    want = first_map if orders.map is None else orders.map
                    if want != map_name and want in packs:
                        engine.use_map(packs[want])
                        map_name = want
                        info()                    # the console draws the map run.json names
                    jammed = orders.jam or (scheduled is not None and scheduled.covers(f.t))
                    seen = degrade(f, [operator]) if jammed else f
                    found = engine.locate(seen, out_dir / FRAMES_DIR, jammed, save=True)
                    rec = tracked_record(found, out_dir, engine.backend)
                    wall = clock() - start
                    extra = rec.extra | {"wall_s": round(wall, 3), "map": map_name}
                    truth = getattr(source, "truth", None)
                    if truth is not None:
                        tr = truth(f.idx)
                        extra["truth"] = tr.extra()
                        extra["gps"] = receiver.read(f, tr.center, orders.spoof, rec)
                    rec = dataclasses.replace(rec, extra=extra)
                    writer.write(rec)
                    counts["frames"] += 1
                    if found.accepted:
                        counts["fixes"] += 1
                        first_fix, last_fix = (wall if first_fix is None else first_fix), wall
                    counts.update(fixes_per_s=round(_fix_rate(first_fix, last_fix, counts["fixes"]), 3), jam=jammed,
                                  spoof=orders.spoof, t=f.t)
                    status("running")
                    if on_record is not None:
                        on_record(rec)
            if ordered is None:
                break
            # the new flight: its source, its pace, and an engine that has never seen the old one
            source, speed = flights[ordered.name](), ordered.speed
            rate = _rate(source)
            engine.restart()
            receiver.restart()
            chosen.update(flight=ordered.name, flight_no=chosen["flight_no"] + 1)
        state = "done"
    finally:
        if not counts.get("frames") and not existed and state != "done" and chosen["flight_no"] == 0:
            # nothing was flown (the source failed on its first frame): leave no run folder behind
            for name in ("run.json", "run.jsonl", STATUS, f".{STATUS}.partial"):
                (out_dir / name).unlink(missing_ok=True)
            try:
                out_dir.rmdir()
            except OSError:
                pass
        else:
            status(state, jam=False, spoof=False)
    return LiveSummary(out_dir=out_dir, frames=counts["frames"], fixes=counts["fixes"], skipped=counts["skipped"],
                       seconds=0.0 if start is None else clock() - start,
                       fixes_per_s=_fix_rate(first_fix, last_fix, counts["fixes"]))


def _same_file(a, b) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False
