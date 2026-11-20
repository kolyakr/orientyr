"""Telemetry: one JSON record per frame, written to ``run.jsonl``; the link between the engine and every consumer.

The engine (``scripts/run.py``, live mode) writes a run; the renderers, the replay viewer, the benchmark and the
UI read it back. One line per frame, in frame order, so a run can be streamed while it is being written and
each line parses on its own (also with a browser's ``JSON.parse``).

A line is the ``TrackState`` flattened, plus the frame's identity and a few optional fields::

    {"idx": 0, "t": 0.0, "source": "crops/crop_0.png",
     "state": "TRACKING", "x": 1877.6, "y": 1324.1, "vx": 0.0, "vy": 0.0, "cov": [[4.0, 0.0], [0.0, 4.0]],
     "fix": {"H": [[...], [...], [...]], "x": 1877.6, "y": 1324.1, "heading_deg": 41.3, "scale": 0.73,
             "footprint": [[...], [...], [...], [...]], "inliers": 191, "inlier_ratio": 0.62, "reproj_px": 1.4,
             "accepted": true, "reject_reason": null},
     "backend": "sift", "ms": 42.5, "interpolated": false, "extra": {}}

- ``fix`` is ``null`` when no homography was produced; its ``idx`` is the record's and is not repeated.
- ``backend``, ``ms``, ``interpolated`` and ``extra`` are optional when reading (defaults: null, null, false, {});
  the writer always writes them, so readers in other languages see one fixed shape.
- ``extra`` holds fields that later milestones add (simulator truth, GPS, search window, events). Unknown
  top-level keys are rejected, so a typo fails loudly instead of being silently dropped.
- Numbers must be finite: ``NaN`` and ``Infinity`` are not JSON and break the web viewer.
- Records are in frame order: ``idx`` strictly increasing (gaps are fine, e.g. every N-th video frame) and ``t``
  never decreasing. Both the writer and the reader check it.

A run folder holds ``run.jsonl``, the frames each record's ``source`` names (relative to the folder), and
``run.json``: the run-level facts (:class:`RunInfo`), above all which map the run was flown over, so a renderer
needs only ``--run <dir>``. The map path is stored relative to the run folder.

Every problem raises :class:`TelemetryError` with the field (and, when reading a file, the line) it is about.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np

from dronenav.types import Fix, TrackMode, TrackState

__all__ = ["RUN_INFO", "Record", "RunInfo", "RunWriter", "TelemetryError", "iter_run", "read_run",
           "read_run_info", "record_from_dict", "record_to_dict", "write_run", "write_run_info"]

RUN_INFO = "run.json"  # the run-level facts, next to run.jsonl

# Top-level keys of a line, in the order they are written. The first group is required when reading.
_REQUIRED = ("idx", "t", "source", "state", "x", "y", "vx", "vy", "cov", "fix")
_OPTIONAL = ("backend", "ms", "interpolated", "extra")
# Keys of the nested fix object (the Fix fields without idx).
_FIX_KEYS = ("H", "x", "y", "heading_deg", "scale", "footprint", "inliers", "inlier_ratio", "reproj_px",
             "accepted", "reject_reason")


class TelemetryError(ValueError):
    """A telemetry record or ``run.jsonl`` line that does not follow the format. A ValueError: bad input."""


# --- the record ------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)  # holds numpy arrays through track: identity equality, like the types it wraps
class Record:
    """One frame of a run: which frame it was, and what the tracker believed after it.

    idx: frame index, the same as ``track.idx`` (and ``track.fix.idx``). t: seconds from the start of the flight.
    source: the frame's file path or video name. track: the tracker's state after this frame.
    backend: matcher name, if one ran. ms: processing time of the frame, milliseconds.
    interpolated: the position was filled in, not measured (M1-06). extra: JSON-ready fields added later.
    """

    idx: int
    t: float
    source: str
    track: TrackState
    backend: str | None = None
    ms: float | None = None
    interpolated: bool = False
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "idx", _int("idx", self.idx))
        object.__setattr__(self, "t", _number("t", self.t))
        if not isinstance(self.source, str):
            raise TelemetryError(f"source must be a str, got {type(self.source).__name__}")
        if not isinstance(self.track, TrackState):
            raise TelemetryError(f"track must be a TrackState, got {type(self.track).__name__}")
        # A record mixing two frames would draw a fix at the wrong time: catch it where it is made.
        if self.track.idx != self.idx:
            raise TelemetryError(f"idx {self.idx} and track.idx {self.track.idx} disagree")
        if self.track.fix is not None and self.track.fix.idx != self.idx:
            raise TelemetryError(f"idx {self.idx} and track.fix.idx {self.track.fix.idx} disagree")
        if self.backend is not None and not isinstance(self.backend, str):
            raise TelemetryError(f"backend must be a str or null, got {type(self.backend).__name__}")
        if self.ms is not None:
            object.__setattr__(self, "ms", _number("ms", self.ms, minimum=0.0))
        if not isinstance(self.interpolated, (bool, np.bool_)):
            raise TelemetryError(f"interpolated must be a bool, got {type(self.interpolated).__name__}")
        object.__setattr__(self, "interpolated", bool(self.interpolated))
        if not isinstance(self.extra, dict):
            raise TelemetryError(f"extra must be an object, got {type(self.extra).__name__}")


# --- record <-> dict -------------------------------------------------------------------------------------------

def record_to_dict(rec: Record) -> dict:
    """The JSON-ready dict of one line: plain Python numbers, arrays as nested lists, finite values only."""
    tr = rec.track
    d = {
        "idx": rec.idx, "t": rec.t, "source": rec.source,
        "state": tr.state.value,
        "x": _number("x", tr.x), "y": _number("y", tr.y),
        "vx": _number("vx", tr.vx), "vy": _number("vy", tr.vy),
        "cov": _array_out("cov", tr.cov),
        "fix": None if tr.fix is None else _fix_to_dict(tr.fix),
        "backend": rec.backend, "ms": rec.ms, "interpolated": rec.interpolated,
        "extra": _json_copy("extra", rec.extra),
    }
    return d


def _json_copy(name: str, value: dict) -> dict:
    """A deep copy of ``value`` made through JSON, so the line owns its data and the round trip is exact."""
    try:
        copy = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as e:
        raise TelemetryError(f"{name} must hold only JSON values with finite numbers: {e}") from None
    if copy != value:  # a tuple would come back as a list, an int key as a string
        raise TelemetryError(f"{name} must survive JSON unchanged: use lists, not tuples, and string keys")
    return copy


def record_from_dict(d: dict) -> Record:
    """Validate one parsed line and build its Record; raises TelemetryError naming the bad field."""
    if not isinstance(d, dict):
        raise TelemetryError(f"a record must be a JSON object, got {type(d).__name__}")
    _check_keys(d, _REQUIRED, _OPTIONAL, prefix="")
    idx = _int("idx", d["idx"])
    fix = d["fix"]
    if fix is not None and not isinstance(fix, dict):
        raise TelemetryError(f"fix must be an object or null, got {type(fix).__name__}")
    try:
        state = TrackMode(d["state"])
    except ValueError:
        raise TelemetryError(f"state must be one of {[m.value for m in TrackMode]}, got {d['state']!r}") from None
    track = _built("track", TrackState, idx=idx, state=state,
                   x=_number("x", d["x"]), y=_number("y", d["y"]),
                   vx=_number("vx", d["vx"]), vy=_number("vy", d["vy"]),
                   cov=_array_in("cov", d["cov"], (2, 2)),
                   fix=None if fix is None else _fix_from_dict(fix, idx))
    ms = d.get("ms")
    return Record(idx=idx, t=_number("t", d["t"]), source=d["source"], track=track,
                  backend=d.get("backend"), ms=None if ms is None else _number("ms", ms),
                  interpolated=d.get("interpolated", False), extra=d.get("extra", {}))


def _fix_to_dict(f: Fix) -> dict:
    return {
        "H": _array_out("fix.H", f.H), "x": _number("fix.x", f.x), "y": _number("fix.y", f.y),
        "heading_deg": _number("fix.heading_deg", f.heading_deg), "scale": _number("fix.scale", f.scale),
        "footprint": _array_out("fix.footprint", f.footprint),
        "inliers": int(f.inliers), "inlier_ratio": _number("fix.inlier_ratio", f.inlier_ratio),
        "reproj_px": _number("fix.reproj_px", f.reproj_px),
        "accepted": bool(f.accepted), "reject_reason": f.reject_reason,
    }


def _fix_from_dict(d: dict, idx: int) -> Fix:
    _check_keys(d, _FIX_KEYS, (), prefix="fix.")
    if not isinstance(d["accepted"], bool):
        raise TelemetryError(f"fix.accepted must be a bool, got {type(d['accepted']).__name__}")
    reason = d["reject_reason"]
    if reason is not None and not isinstance(reason, str):
        raise TelemetryError(f"fix.reject_reason must be a str or null, got {type(reason).__name__}")
    return _built("fix", Fix, idx=idx, H=_array_in("fix.H", d["H"], (3, 3)),
                  x=_number("fix.x", d["x"]), y=_number("fix.y", d["y"]),
                  heading_deg=_number("fix.heading_deg", d["heading_deg"]), scale=_number("fix.scale", d["scale"]),
                  footprint=_array_in("fix.footprint", d["footprint"], (4, 2)),
                  inliers=_int("fix.inliers", d["inliers"]),
                  inlier_ratio=_number("fix.inlier_ratio", d["inlier_ratio"]),
                  reproj_px=_number("fix.reproj_px", d["reproj_px"]),
                  accepted=d["accepted"], reject_reason=reason)


# --- files -----------------------------------------------------------------------------------------------------

class RunWriter:
    """Writes a run line by line, flushing each one, so a reader can follow the file while the run goes on.

    Use as a context manager: ``with RunWriter(out / "run.jsonl") as w: w.write(record)``.
    The file is replaced, not appended to: one file is one run.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self._last: Record | None = None

    def write(self, rec: Record) -> None:
        # Check and serialise before writing, so an invalid record never leaves half a line in the file.
        _check_order(self._last, rec)
        line = json.dumps(record_to_dict(rec), allow_nan=False)
        self._f.write(line + "\n")
        self._f.flush()
        self._last = rec

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_run(path: str | Path, records: Iterable[Record]) -> int:
    """Write a whole run to ``path``; returns the number of records written."""
    n = 0
    with RunWriter(path) as w:
        for rec in records:
            w.write(rec)
            n += 1
    return n


def iter_run(path: str | Path) -> Iterator[Record]:
    """Read a run lazily, record by record; blank lines are skipped. Errors name the file and the line."""
    path = Path(path)
    last: Record | None = None
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            where = f"{path.name} line {lineno}"
            try:
                d = json.loads(line, parse_constant=_reject_constant)
            except _NonFinite as e:
                raise TelemetryError(f"{where}: {e} is not a finite number (NaN and Infinity are not JSON)") from None
            except json.JSONDecodeError as e:
                if not line.endswith("\n"):  # the writer ends every line with one
                    raise TelemetryError(f"{where}: incomplete last line (the file is truncated or still "
                                         f"being written)") from None
                raise TelemetryError(f"{where}: invalid JSON: {e.msg} at column {e.colno}") from None
            try:
                rec = record_from_dict(d)
                _check_order(last, rec)
            except TelemetryError as e:
                raise TelemetryError(f"{where}: {e}") from None
            last = rec
            yield rec


def read_run(path: str | Path) -> list[Record]:
    """Read a whole run into a list, in file order."""
    return list(iter_run(path))


# --- run.json: run-level facts --------------------------------------------------------------------------------

@dataclass(frozen=True)
class RunInfo:
    """What a whole run shares, stored as ``<run dir>/run.json``.

    map: the map the run was flown over. Given to :func:`write_run_info` as the caller knows it (absolute, or
    relative to the working directory); stored relative to the run folder; returned by :func:`read_run_info`
    as an absolute path. map_size: (w, h) in px. generator: what made the run ("fake_run", "run.py").
    frame_size: (w, h) of the frames, if fixed. fps: frame rate, if fixed. seed: the random seed, if any.
    """

    map: str
    map_size: tuple[int, int]
    generator: str
    frame_size: tuple[int, int] | None = None
    fps: float | None = None
    seed: int | None = None

    def __post_init__(self):
        if not isinstance(self.map, (str, os.PathLike)) or not str(self.map):
            raise TelemetryError(f"map must be a path, got {self.map!r}")
        object.__setattr__(self, "map", os.fspath(self.map))
        object.__setattr__(self, "map_size", _size("map_size", self.map_size))
        if not isinstance(self.generator, str):
            raise TelemetryError(f"generator must be a str, got {type(self.generator).__name__}")
        if self.frame_size is not None:
            object.__setattr__(self, "frame_size", _size("frame_size", self.frame_size))
        if self.fps is not None:
            fps = _number("fps", self.fps)
            if fps <= 0:
                raise TelemetryError(f"fps must be > 0, got {fps:g}")
            object.__setattr__(self, "fps", fps)
        if self.seed is not None:
            object.__setattr__(self, "seed", _int("seed", self.seed))

    @property
    def map_path(self) -> Path:
        return Path(self.map)


_INFO_REQUIRED = ("map", "map_size", "generator")
_INFO_OPTIONAL = ("frame_size", "fps", "seed")


def write_run_info(run_dir: str | Path, info: RunInfo) -> Path:
    """Write ``run.json`` into ``run_dir`` (created if needed); returns its path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    # relative to the run folder, so the folder can be read from any working directory (or moved with its map)
    rel = Path(os.path.relpath(os.path.abspath(info.map), os.path.abspath(run_dir))).as_posix()
    d = {"map": rel, "map_size": list(info.map_size), "generator": info.generator,
         "frame_size": None if info.frame_size is None else list(info.frame_size), "fps": info.fps, "seed": info.seed}
    path = run_dir / RUN_INFO
    path.write_text(json.dumps(d, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    return path


def read_run_info(run_dir: str | Path) -> RunInfo:
    """Read ``<run_dir>/run.json``; the returned ``map`` is absolute. Raises TelemetryError naming the field."""
    run_dir = Path(run_dir)
    path = run_dir / RUN_INFO
    text = path.read_text(encoding="utf-8")   # FileNotFoundError names the path
    try:
        d = json.loads(text, parse_constant=_reject_constant)
        if not isinstance(d, dict):
            raise TelemetryError(f"must be a JSON object, got {type(d).__name__}")
        _check_keys(d, _INFO_REQUIRED, _INFO_OPTIONAL, prefix="")
        if not isinstance(d["map"], str):
            raise TelemetryError(f"map must be a str, got {type(d['map']).__name__}")
        return RunInfo(map=os.path.normpath(os.path.join(os.path.abspath(run_dir), d["map"])),
                       map_size=d["map_size"], generator=d["generator"], frame_size=d.get("frame_size"),
                       fps=d.get("fps"), seed=d.get("seed"))
    except _NonFinite as e:
        raise TelemetryError(f"{RUN_INFO}: {e} is not a finite number") from None
    except json.JSONDecodeError as e:
        raise TelemetryError(f"{RUN_INFO}: invalid JSON: {e.msg} at line {e.lineno}") from None
    except TelemetryError as e:
        raise TelemetryError(f"{RUN_INFO}: {e}") from None


# --- validation helpers ----------------------------------------------------------------------------------------

def _check_order(prev: Record | None, rec: Record) -> None:
    """Records follow the flight: idx strictly increasing, t never going back."""
    if prev is None:
        return
    if rec.idx <= prev.idx:
        raise TelemetryError(f"idx {rec.idx} after idx {prev.idx}: records must be in frame order "
                             f"(idx strictly increasing)")
    if rec.t < prev.t:
        raise TelemetryError(f"t {rec.t:g} after t {prev.t:g}: records must be in frame order (t never decreasing)")


class _NonFinite(Exception):
    pass


def _reject_constant(token: str):
    raise _NonFinite(token)


def _check_keys(d: dict, required: tuple, optional: tuple, prefix: str) -> None:
    for k in required:
        if k not in d:
            raise TelemetryError(f"missing field '{prefix}{k}'")
    for k in d:
        if k not in required and k not in optional:
            hint = " (put extensions under 'extra')" if not prefix else ""
            raise TelemetryError(f"unknown field '{prefix}{k}'{hint}")


def _is_number(v) -> bool:
    # bool is an int in Python, but true/false as a coordinate is a bug, not a number
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _number(name: str, v, minimum: float | None = None) -> float:
    if not _is_number(v):
        raise TelemetryError(f"{name} must be a number, got {type(v).__name__}")
    try:
        v = float(v)
    except OverflowError:  # an int with hundreds of digits: valid JSON, not a float64
        raise TelemetryError(f"{name} must be finite, got an integer too large for a float") from None
    if not math.isfinite(v):
        raise TelemetryError(f"{name} must be finite, got {v}")
    if minimum is not None and v < minimum:
        raise TelemetryError(f"{name} must be >= {minimum:g}, got {v:g}")
    return v


def _int(name: str, v) -> int:
    if not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)):
        raise TelemetryError(f"{name} must be an int, got {type(v).__name__}")
    if v < 0:
        raise TelemetryError(f"{name} must be >= 0, got {v}")
    return int(v)


def _size(name: str, v) -> tuple[int, int]:
    """(w, h): two positive ints, from a tuple or a JSON list."""
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise TelemetryError(f"{name} must be [w, h], got {v!r}")
    w, h = (_int(name, x) for x in v)
    if w == 0 or h == 0:
        raise TelemetryError(f"{name} must be positive, got {list(v)}")
    return w, h


def _array_in(name: str, v, shape: tuple) -> np.ndarray:
    """A nested list of numbers with exactly this shape, as float64; no strings or bools sneaking in."""
    want = "x".join(map(str, shape))

    def rows(x, depth):
        if depth == len(shape):
            if not _is_number(x):
                raise TelemetryError(f"{name} must be {want} numbers, found {type(x).__name__}")
            return
        if not isinstance(x, list):
            raise TelemetryError(f"{name} must be {want} nested lists, found {type(x).__name__}")
        for item in x:
            rows(item, depth + 1)

    rows(v, 0)
    try:
        a = np.array(v, dtype=np.float64)
    except ValueError:  # ragged rows
        raise TelemetryError(f"{name} must be {want}, got ragged rows") from None
    except OverflowError:
        raise TelemetryError(f"{name} must be finite, got an integer too large for a float") from None
    if a.shape != shape:
        raise TelemetryError(f"{name} must be {want}, got shape {a.shape}")
    if not np.isfinite(a).all():
        raise TelemetryError(f"{name} must be finite, got {a.tolist()}")
    return a


def _array_out(name: str, a: np.ndarray) -> list:
    if not np.isfinite(a).all():
        raise TelemetryError(f"{name} must be finite, got {a.tolist()}")
    return a.tolist()


def _built(name: str, cls, **kw):
    """Construct a core type, turning its own validation error into a TelemetryError that names the field."""
    try:
        return cls(**kw)
    except ValueError as e:
        raise TelemetryError(f"{name}: {e}") from None
