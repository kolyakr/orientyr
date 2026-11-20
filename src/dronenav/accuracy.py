"""Accuracy against truth (M1-10): how far each frame's position is from where the drone really was.

Two kinds of truth, both read by :func:`load_truth` into ``{frame idx: (x, y)}`` in map px:
- ``truth.jsonl`` of an Irpin flight (``center_map_px``): exact, from the renderer. This is the accuracy reference;
- ``probe.json`` of the task crops (``cx``, ``cy`` of its ``ok`` rows): a full-resolution fit, not ground truth.
  It catches regressions on the task frames but can't show errors below its own (~0.2 px).

:func:`score` compares a run's records with the truth. An accepted fix is *localized*: *correct* within
``within_px`` of the truth, otherwise *wrong*. Wrong fixes are counted apart because they are the dangerous kind:
a miss leaves the route uncertain, a wrong fix moves it with full confidence. The track position's error is also
kept for every frame, so the coasting of LOST frames can be judged too.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dronenav.telemetry import Record

__all__ = ["Accuracy", "FrameError", "load_truth", "score"]

WITHIN_PX = 10.0   # "localized correctly": the README's tables use the same radius (M1-03, M1-05)


def load_truth(path: str | Path) -> dict[int, tuple[float, float]]:
    """Frame centres in map px by frame idx, from ``truth.jsonl``, a folder holding one, or ``probe.json``."""
    path = Path(path)
    if path.is_dir():
        if not (path / "truth.jsonl").is_file():
            raise FileNotFoundError(f"{path} has no truth.jsonl")
        path = path / "truth.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no truth file at {path}")
    text = path.read_text()
    try:
        if path.suffix == ".jsonl":
            rows = (json.loads(line) for line in text.splitlines() if line.strip())   # blank lines are fine
            truth = {int(r["i"]): _xy(r["center_map_px"]) for r in rows}
        else:
            truth = {int(r["i"]): _xy((r["cx"], r["cy"])) for r in json.loads(text)["rows"] if r["ok"]}
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{path} is neither truth.jsonl (i, center_map_px) nor probe.json (rows: i, ok, cx, cy): "
                         f"{type(e).__name__} {e}") from None
    if not truth:
        raise ValueError(f"{path} holds no truth")
    return truth


def _xy(v) -> tuple[float, float]:
    x, y = (float(c) for c in v)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"not a finite position: {v!r}")
    return x, y


@dataclass(frozen=True)
class FrameError:
    """One frame. fix_px: the accepted fix's distance from the truth, None when the frame wasn't localized.
    track_px: the recorded position's distance from the truth, for every frame (LOST ones coast or hold)."""

    idx: int
    state: str
    fix_px: float | None
    track_px: float


@dataclass(frozen=True)
class Accuracy:
    """A run against its truth. n: the run's frames that have a truth; missing_truth: the run's frames without
    one; not_in_run: truth frames the run doesn't have (a partial run, or the wrong truth). The error statistics
    cover the localized frames and are NaN when there is none."""

    frames: tuple[FrameError, ...]
    within_px: float
    missing_truth: int
    not_in_run: int = 0

    @property
    def n(self) -> int:
        return len(self.frames)

    @property
    def errors(self) -> np.ndarray:
        return np.array([f.fix_px for f in self.frames if f.fix_px is not None], float)

    @property
    def localized(self) -> int:
        return len(self.errors)

    @property
    def correct(self) -> int:
        return int((self.errors <= self.within_px).sum())

    @property
    def wrong(self) -> int:
        return self.localized - self.correct

    @property
    def median_px(self) -> float:
        return float(np.median(self.errors)) if self.localized else math.nan

    @property
    def p95_px(self) -> float:
        return float(np.percentile(self.errors, 95)) if self.localized else math.nan

    @property
    def max_px(self) -> float:
        return float(self.errors.max()) if self.localized else math.nan

    def summary(self) -> dict:
        """The numbers, JSON-safe (NaN becomes None)."""
        def num(v: float) -> float | None:
            return None if math.isnan(v) else round(v, 3)
        return {"frames": self.n, "localized": self.localized, "correct": self.correct, "wrong": self.wrong,
                "within_px": self.within_px, "median_px": num(self.median_px), "p95_px": num(self.p95_px),
                "max_px": num(self.max_px), "missing_truth": self.missing_truth,
                "not_in_run": self.not_in_run}


def score(records: Iterable[Record], truth: dict[int, tuple[float, float]],
          within_px: float = WITHIN_PX) -> Accuracy:
    """Each record's error against ``truth``. Raises ValueError if no record has a truth (the wrong file) or
    ``within_px`` is not a number >= 0."""
    if not (isinstance(within_px, (int, float)) and math.isfinite(within_px) and within_px >= 0):
        raise ValueError(f"within_px must be a finite number >= 0, got {within_px!r}")
    frames, missing = [], 0
    for rec in records:
        if rec.idx not in truth:
            missing += 1
            continue
        tx, ty = truth[rec.idx]
        tr, fix = rec.track, rec.track.fix
        fix_px = math.hypot(fix.x - tx, fix.y - ty) if fix is not None and fix.accepted else None
        frames.append(FrameError(rec.idx, tr.state.value, fix_px, math.hypot(tr.x - tx, tr.y - ty)))
    if not frames:
        raise ValueError(f"none of the run's {missing} frames has a truth: is it the truth of this run?")
    return Accuracy(tuple(frames), float(within_px), missing, len(truth.keys() - {f.idx for f in frames}))
