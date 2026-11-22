"""The benchmark's numbers (M6-03), one definition for every benchmark task so their tables are comparable.

A run's records are scored against the true frame centres, in metres (map px x the map's measured ``m_per_px``):

- **localised %**: frames with an accepted fix within ``WITHIN_M`` (10 m) of the truth;
- **wrong-fix %**: frames with an accepted fix more than ``WRONG_M`` (30 m) off. Reported apart because it is the
  dangerous number: a missed frame leaves the route uncertain, a wrong fix moves it with full confidence.
  Both are shares of *all* frames, so they add up with the misses; a rejected fix is neither;
- **coverage %**: frames where the tracker reports a position at all (before the first fix it has none, M0-12);
- **median, RMSE, p95**: the error of the reported (tracked) position, over the frames that have one;
- **uncertainty coverage %**: frames whose truth lies inside the tracker's 95 % region, the ellipse
  ``d' cov^-1 d <= 5.991`` (chi-square, 2 d.o.f.). About 95 % means the covariance is honest; far less, overconfident.

Every number comes with a bootstrap 95 % interval over frames: a preset is one route, so the frames are the only
sample there is. The resampling is seeded, so the same seed gives the same intervals. Time per frame is the
machine's, not the method's: :func:`timing` keeps it apart from the numbers that must repeat.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from dronenav.telemetry import Record

__all__ = ["CHI2_95", "METRICS", "WITHIN_M", "WRONG_M", "FrameEval", "frame_metrics", "missed", "summarize", "timing"]

WITHIN_M = 10.0     # "localised": the spec's radius
WRONG_M = 30.0      # "wrong fix": far enough that it is not just an imprecise right answer
CHI2_95 = 5.991     # chi-square quantile, 2 d.o.f., p = 0.95: the 95 % ellipse of a 2D Gaussian


@dataclass(frozen=True)
class FrameEval:
    """One frame against its truth. fix_m: the accepted fix's error, None when there is none. track_m: the reported
    position's error, None when there is no position. inside95: truth inside the 95 % region, None without a
    covariance. ms: the frame's processing time, if recorded."""

    idx: int
    fix_m: float | None
    track_m: float | None
    inside95: bool | None
    ms: float | None


def frame_metrics(records: Iterable[Record], truth_px: dict[int, tuple[float, float]],
                  m_per_px: float) -> list[FrameEval]:
    """Each record that has a truth, scored in metres. ``truth_px``: frame idx -> true centre in this map's px."""
    out = []
    for rec in records:
        if rec.idx not in truth_px:
            continue
        tx, ty = truth_px[rec.idx]
        tr, fix = rec.track, rec.track.fix
        fix_m = math.hypot(fix.x - tx, fix.y - ty) * m_per_px if fix is not None and fix.accepted else None
        track_m = inside = None
        if tr.has_position:
            d = np.array([tr.x - tx, tr.y - ty])
            track_m = float(np.hypot(*d)) * m_per_px
            if tr.cov is not None:
                inside = bool(d @ np.linalg.solve(tr.cov, d) <= CHI2_95)   # in px: the covariance's own units
        out.append(FrameEval(rec.idx, fix_m, track_m, inside, rec.ms))
    return out


def missed(idxs: Iterable[int], truth_px: dict[int, tuple[float, float]]) -> list[FrameEval]:
    """The frames of a run that localized none of them (the pipeline's NothingLocalized): each a frame with no fix
    and no position, so the run scores 0 % rather than dropping out of the table."""
    return [FrameEval(i, None, None, None, None) for i in idxs if i in truth_px]


def _pct(mask: np.ndarray) -> float:
    return 100.0 * float(mask.mean()) if mask.size else math.nan


def _stats(e: np.ndarray, valid: np.ndarray) -> Callable:
    """A statistic of the valid entries of ``e`` (NaN when none)."""
    def of(fn):
        def f(idx):
            v = e[idx][valid[idx]]
            return float(fn(v)) if v.size else math.nan
        return f
    return of


def _metric_fns(frames: list[FrameEval]) -> dict[str, Callable[[np.ndarray], float]]:
    """metric name -> function of a sample of frame indices (the identity sample gives the value itself)."""
    nan = math.nan
    fix = np.array([nan if f.fix_m is None else f.fix_m for f in frames], float)
    track = np.array([nan if f.track_m is None else f.track_m for f in frames], float)
    has_fix, has_pos = ~np.isnan(fix), ~np.isnan(track)
    has_cov = np.array([f.inside95 is not None for f in frames])
    inside = np.array([bool(f.inside95) for f in frames])
    err = _stats(track, has_pos)
    return {
        "localised_pct": lambda i: _pct(has_fix[i] & (np.nan_to_num(fix[i], nan=math.inf) <= WITHIN_M)),
        "wrong_fix_pct": lambda i: _pct(has_fix[i] & (np.nan_to_num(fix[i], nan=-math.inf) > WRONG_M)),
        "coverage_pct": lambda i: _pct(has_pos[i]),
        "median_m": err(np.median),
        "rmse_m": err(lambda v: np.sqrt(np.mean(v ** 2))),
        "p95_m": err(lambda v: np.percentile(v, 95)),
        "unc_coverage_pct": lambda i: _pct(inside[i][has_cov[i]]),
    }


METRICS = ("localised_pct", "wrong_fix_pct", "coverage_pct", "median_m", "rmse_m", "p95_m", "unc_coverage_pct")


def _num(v: float) -> float | None:
    return None if math.isnan(v) else float(v)   # full precision: the same frames give the same floats


def summarize(frames: list[FrameEval], seed: int = 7, n_boot: int = 1000) -> dict:
    """Every metric as ``{"value", "lo", "hi"}`` (None where undefined, e.g. no position at all), plus ``frames``.
    lo, hi: the 2.5 and 97.5 percentiles over ``n_boot`` resamples of the frames, drawn with ``seed``."""
    n = len(frames)
    fns = _metric_fns(frames)
    out: dict = {"frames": n}
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, n, (n_boot, n)) if n else np.zeros((0, 0), int)
    for name in METRICS:
        value = fns[name](np.arange(n)) if n else math.nan
        boot = np.array([fns[name](s) for s in samples], float)
        boot = boot[~np.isnan(boot)]
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot.size and not math.isnan(value) else (math.nan, math.nan))
        out[name] = {"value": _num(value), "lo": _num(float(lo)), "hi": _num(float(hi))}
    return out


def timing(frames: list[FrameEval]) -> dict:
    """Mean processing time per frame, ms (None if the run did not record it). Not repeatable: kept apart."""
    ms = [f.ms for f in frames if f.ms is not None]
    return {"ms_per_frame": round(float(np.mean(ms)), 1) if ms else None}
