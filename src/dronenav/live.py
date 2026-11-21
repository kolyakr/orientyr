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
"running", then "done" or "stopped"), ``frames``, ``fixes``, ``skipped``, ``fixes_per_s``, ``jam`` and ``t``. It is
replaced atomically after every record, so its modification time says the engine is alive.

**control.json** (``CONTROL``) is how chaos is switched on while the flight runs: ``{"jam": true}`` makes every
frame from the next one on arrive as noise (``dronenav.sim.chaos.Jam``) until ``{"jam": false}``. The console's
"Jam camera" button writes it; so does ``echo '{"jam": true}' > RUN/control.json``. It is read before every frame;
a file that is missing, half-written or not a jam switch changes nothing. ``jam=(start_s, length_s)`` schedules a
jam as in ``run.py``; the two add up.

``fixes_per_s`` is the spec's number: accepted fixes per second of wall time, counted between the first and the
last fix (``(fixes - 1) / (wall of the last - wall of the first)``), so a short run is not flattered by its ends.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dronenav.io import _is_rate
from dronenav.pipeline import FRAMES_DIR, Engine, _check_out, _jam, tracked_record
from dronenav.sim.chaos import Jam, degrade
from dronenav.telemetry import Record, RunInfo, RunWriter, write_run_info
from dronenav.types import FrameSource

__all__ = ["CONTROL", "STATUS", "LiveSummary", "live_run", "read_control"]

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


def read_control(run_dir: str | Path, last: bool = False) -> bool:
    """Is the jam ordered on? ``control.json``'s ``{"jam": true | false}``; anything else (no file, half a file,
    another shape) keeps ``last``, so a write caught halfway never flips the switch."""
    try:
        data = json.loads((Path(run_dir) / CONTROL).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return last
    jam = data.get("jam") if isinstance(data, dict) else None
    return jam if isinstance(jam, bool) else last


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
             jam: tuple[float, float] | None = None, clock: Callable[[], float] = time.monotonic,
             sleep: Callable[[float], None] = time.sleep,
             on_record: Callable[[Record], None] | None = None) -> LiveSummary:
    """Play ``source`` in real time (``speed`` flight seconds per second) through the engine and write the run
    into ``out_dir`` as it is made (see the module docstring). ``on_record`` is called after each record is on disk.
    Bad options and a missing map are refused before anything is written."""
    out_dir = Path(out_dir)
    if not _is_rate(speed):
        raise ValueError(f"speed must be a positive number, got {speed!r}")
    _check_out(out_dir)
    scheduled = _jam(jam)
    rate = _rate(source)
    period = 1.0 / (rate * speed)                 # wall seconds between two frames
    existed = out_dir.exists()
    engine = Engine(map_path, matcher, None if cache_dir is None else Path(cache_dir), "kalman", device)
    # switched on by the operator: covers every frame while it is applied; seeded, so the noise is repeatable
    operator = Jam(1.0, start_s=0.0, max_s=1e12)

    out_dir.mkdir(parents=True, exist_ok=True)
    # an old run in this folder: its orders are not for this flight, and its status must not read as ours
    for name in (CONTROL, STATUS):
        (out_dir / name).unlink(missing_ok=True)
    counts = {"frames": 0, "fixes": 0, "skipped": 0, "fixes_per_s": 0.0, "jam": False, "t": None}
    h, w = engine.map.shape[:2]
    state = "stopped"                             # what the status says if anything below raises
    start = t_first = first_fix = last_fix = None
    try:
        with RunWriter(out_dir / "run.jsonl") as writer:
            frame_size = getattr(source, "size", None) or getattr(source, "frame_size", None)
            write_run_info(out_dir, RunInfo(map=str(map_path), map_size=(w, h), generator=GENERATOR,
                                            frame_size=None if frame_size is None else tuple(frame_size),
                                            fps=rate))
            _write_status(out_dir, state="starting", **counts)
            jam_on = False
            for f in source:
                if start is None:
                    start, t_first = clock(), f.t
                due = start + (f.t - t_first) / speed
                now = clock()
                if now < due:
                    sleep(due - now)
                elif now - due >= period:         # the next frame is due already: this one is history
                    counts["skipped"] += 1
                    continue
                jam_on = read_control(out_dir, jam_on)
                jammed = jam_on or (scheduled is not None and scheduled.covers(f.t))
                seen = degrade(f, [operator]) if jammed else f
                found = engine.locate(seen, out_dir / FRAMES_DIR, jammed, save=True)
                rec = tracked_record(found, out_dir, engine.backend)
                wall = clock() - start
                extra = rec.extra | {"wall_s": round(wall, 3)}
                truth = getattr(source, "truth", None)
                if truth is not None:
                    extra["truth"] = truth(f.idx).extra()
                rec = dataclasses.replace(rec, extra=extra)
                writer.write(rec)
                counts["frames"] += 1
                if found.accepted:
                    counts["fixes"] += 1
                    first_fix, last_fix = (wall if first_fix is None else first_fix), wall
                counts.update(fixes_per_s=round(_fix_rate(first_fix, last_fix, counts["fixes"]), 3), jam=jammed,
                              t=f.t)
                _write_status(out_dir, state="running", **counts)
                if on_record is not None:
                    on_record(rec)
        state = "done"
    finally:
        if counts["frames"] == 0 and not existed and state != "done":
            # nothing was flown (the source failed on its first frame): leave no run folder behind
            for name in ("run.json", "run.jsonl", STATUS, f".{STATUS}.partial"):
                (out_dir / name).unlink(missing_ok=True)
            try:
                out_dir.rmdir()
            except OSError:
                pass
        else:
            _write_status(out_dir, state=state, **(counts | {"jam": False}))
    return LiveSummary(out_dir=out_dir, frames=counts["frames"], fixes=counts["fixes"], skipped=counts["skipped"],
                       seconds=0.0 if start is None else clock() - start,
                       fixes_per_s=_fix_rate(first_fix, last_fix, counts["fixes"]))
