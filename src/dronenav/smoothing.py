"""Gap interpolation and light smoothing: from per-frame fixes to a complete track (M1-06).

The engine localises each frame on its own. Some frames give no accepted fix (blur, glare, a jammed camera, a
frame with nothing on it), but a route with holes can't be drawn or followed. A drone moves smoothly, so a missing
position is filled in between the accepted fixes around it, linearly in time, and marked ``interpolated`` so the
renderers draw it as "not measured" (dashed, no footprint). This is the placeholder for the Kalman filter (M5).

- Frames before the first fix or after the last one have only one neighbour: they hold that fix's position.
  Holding, rather than extrapolating, never sends the route off into the unknown.
- Headings are interpolated along the shorter arc (350° → 10° turns through north) and recorded for every frame
  in ``extra["heading_deg"]``: ``TrackState`` has no heading, and an unmeasured frame has no fix to carry one.
- Smoothing (off by default) is a centred moving average over ``radius`` frames on each side; the window shrinks
  near the ends, so the start and finish stay where they were measured and a straight constant-speed flight is
  unchanged. Headings are averaged on the circle. Fixes themselves are never changed: they are the measurements.
- Uncertainty is a placeholder until M5: ``fix_sigma_px`` on a measured frame, growing by ``gap_sigma_px`` per
  frame of distance to the nearest fix inside a gap.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from dronenav.telemetry import Record
from dronenav.types import Fix, TrackMode, TrackState

__all__ = ["Observation", "SmoothConfig", "fill_track"]


@dataclass(frozen=True)
class SmoothConfig:
    """radius: frames on each side of the moving average; 0 turns smoothing off.
    fix_sigma_px: position σ of a measured frame. gap_sigma_px: added to σ per frame away from the nearest fix."""

    radius: int = 0
    fix_sigma_px: float = 1.0
    gap_sigma_px: float = 10.0

    def __post_init__(self):
        if self.radius < 0:
            raise ValueError(f"radius must be >= 0, got {self.radius}")
        if self.fix_sigma_px <= 0 or self.gap_sigma_px < 0:
            raise ValueError(f"fix_sigma_px must be > 0 and gap_sigma_px >= 0, got "
                             f"{self.fix_sigma_px}, {self.gap_sigma_px}")


_CFG = SmoothConfig()


class Observation(NamedTuple):
    """What the engine knows about one frame before the track is built. fix: the frame's best fix (accepted,
    rejected, or None: no homography). backend, ms, extra: passed through to the record."""

    idx: int
    t: float
    source: str
    fix: Fix | None
    backend: str | None = None
    ms: float | None = None
    extra: dict | None = None


def _measured(o: Observation) -> bool:
    return o.fix is not None and o.fix.accepted


def _arc(a: float, b: float) -> float:
    """The signed shorter turn from heading a to heading b, degrees in [-180, 180)."""
    return (b - a + 180.0) % 360.0 - 180.0


def _wrap(h: float) -> float:
    h %= 360.0
    return 0.0 if h >= 360.0 else h          # -1e-17 % 360 == 360.0 in float arithmetic


def _check_order(obs: Sequence[Observation]) -> None:
    for a, b in itertools.pairwise(obs):
        if b.idx <= a.idx:
            raise ValueError(f"frames must come in order: idx {b.idx} after {a.idx}")
        if b.t < a.t:
            raise ValueError(f"time must not go back: frame {b.idx} at t={b.t} after frame {a.idx} at t={a.t}")


def _fill(obs: Sequence[Observation], fixed: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Positions (n x 2), headings (n) and frames to the nearest fix (n), gaps filled in."""
    n = len(obs)
    xy, head, gap = np.zeros((n, 2)), np.zeros(n), np.zeros(n, int)
    for i, o in enumerate(obs):
        if _measured(o):
            xy[i], head[i] = (o.fix.x, o.fix.y), o.fix.heading_deg
            continue
        before = [j for j in fixed if j < i]
        after = [j for j in fixed if j > i]
        a, b = (before[-1] if before else None), (after[0] if after else None)
        if a is None or b is None:                       # an end of the flight: hold the one fix there is
            j = b if a is None else a
            xy[i], head[i], gap[i] = (obs[j].fix.x, obs[j].fix.y), obs[j].fix.heading_deg, abs(i - j)
            continue
        fa, fb = obs[a].fix, obs[b].fix
        dt = obs[b].t - obs[a].t
        # along the route in time; frames sharing a timestamp fall back to their place in the sequence
        u = (o.t - obs[a].t) / dt if dt > 0 else (i - a) / (b - a)
        xy[i] = (fa.x + (fb.x - fa.x) * u, fa.y + (fb.y - fa.y) * u)
        head[i] = _wrap(fa.heading_deg + _arc(fa.heading_deg, fb.heading_deg) * u)
        gap[i] = min(i - a, b - i)
    return xy, head, gap


def _smooth(xy: np.ndarray, head: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Centred moving average; the window shrinks symmetrically at the ends, so the ends stay put."""
    n = len(xy)
    rad = np.radians(head)
    unit = np.stack([np.sin(rad), np.cos(rad)], axis=1)
    out_xy, out_h = xy.copy(), head.copy()
    for i in range(n):
        k = min(radius, i, n - 1 - i)
        if k == 0:
            continue
        out_xy[i] = xy[i - k:i + k + 1].mean(axis=0)
        s, c = unit[i - k:i + k + 1].mean(axis=0)
        out_h[i] = _wrap(math.degrees(math.atan2(s, c)))
    return out_xy, out_h


def _velocity(obs: Sequence[Observation], xy: np.ndarray) -> np.ndarray:
    """Map px per second from the neighbouring frames (one-sided at the ends); 0 where no time passes."""
    n = len(xy)
    v = np.zeros((n, 2))
    for i in range(n):
        a, b = max(0, i - 1), min(n - 1, i + 1)
        dt = obs[b].t - obs[a].t
        if dt > 0:
            v[i] = (xy[b] - xy[a]) / dt
    return v


def fill_track(observations: Sequence[Observation], cfg: SmoothConfig = _CFG) -> list[Record]:
    """One record per observation, in order: measured frames at their fix, the rest filled in and marked
    ``interpolated`` (state LOST); optionally smoothed. Raises ValueError if the frames are out of order or no
    frame has an accepted fix (there is then no position to fill in from)."""
    obs = list(observations)
    if not obs:
        return []
    _check_order(obs)
    fixed = [i for i, o in enumerate(obs) if _measured(o)]
    if not fixed:
        raise ValueError(f"no accepted fix in {len(obs)} frames: nothing to build a route from")
    xy, head, gap = _fill(obs, fixed)
    if cfg.radius > 0:
        xy, head = _smooth(xy, head, cfg.radius)
    vel = _velocity(obs, xy)
    records = []
    for i, o in enumerate(obs):
        measured = _measured(o)
        sigma = cfg.fix_sigma_px + cfg.gap_sigma_px * gap[i]
        track = TrackState(idx=o.idx, state=TrackMode.TRACKING if measured else TrackMode.LOST,
                           x=float(xy[i, 0]), y=float(xy[i, 1]), vx=float(vel[i, 0]), vy=float(vel[i, 1]),
                           cov=sigma ** 2 * np.eye(2), fix=o.fix)
        extra = {**(o.extra or {}), "heading_deg": float(head[i])}
        records.append(Record(idx=o.idx, t=o.t, source=o.source, track=track, backend=o.backend, ms=o.ms,
                              interpolated=not measured, extra=extra))
    return records
