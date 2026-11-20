"""Offline smoother and gap interpolation (M5-05): Rauch–Tung–Striebel over a finished run.

The live filter (M5-01) only knows the past: in a gap it flies on the last velocity, and after a turn it lags.
Once the run is over the future is known too. The RTS smoother runs backwards over the filter's steps and corrects
each state with what came after it:

    C_k  = P_k Fᵀ (P_pred,k+1)⁻¹
    xs_k = x_k + C_k (xs_k+1 − x_pred,k+1)
    Ps_k = P_k + C_k (Ps_k+1 − P_pred,k+1) C_kᵀ

so a gap between two fixes becomes an interpolation between them (with the motion model's curvature), not an
extrapolation from the first. The last state of a segment has no future: it stays the filtered one.

The prediction ``F``, ``x_pred``, ``P_pred`` is recomputed here from consecutive filtered states and their times,
not taken from the steps: the tracker (M5-02) may change the state between steps (it stops the velocity after a
long loss), and the backward pass must use the prediction that links the stored states.

A restart (INIT: the first fix, or a fix after LOST) begins a new segment: the filter forgot everything before it,
and the smoother must not pull one track towards another across it. Each output is marked by where it comes from:

- ``fix``: the step used a fix (INIT or HIT);
- ``interpolated``: no usable fix (COAST, or a GATED outlier), but there is a fix later in the same segment;
- ``extrapolated``: no fix and none later in the segment: nothing to interpolate towards, still a prediction.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from dronenav.tracking.kalman import KalmanConfig, Step, StepKind, _transition

__all__ = ["Smoothed", "smooth"]

_FIX = (StepKind.INIT, StepKind.HIT)


class Smoothed(NamedTuple):
    """One frame's smoothed state: t (s), x = (x, y, vx, vy) in map px and px/s, its 4 x 4 covariance P, and
    source: "fix", "interpolated" or "extrapolated" (see the module doc)."""

    t: float
    x: np.ndarray
    P: np.ndarray
    source: str


def smooth(steps: list[Step], cfg: KalmanConfig | None = None) -> list[Smoothed | None]:
    """Smooth a run's filter steps (in time order, as ``KalmanFilter.step`` returned them) with the filter's
    setting ``cfg``. One output per step; None where the filter had not started (NONE)."""
    cfg = cfg or KalmanConfig()
    out: list[Smoothed | None] = [None] * len(steps)
    start = None
    for k, s in enumerate(steps):
        if s.kind is StepKind.NONE:
            if start is not None:
                _segment(steps, start, k, cfg, out)
            start = None
        elif s.kind is StepKind.INIT or start is None:
            if start is not None:
                _segment(steps, start, k, cfg, out)
            start = k
    if start is not None:
        _segment(steps, start, len(steps), cfg, out)
    return out


def _segment(steps: list[Step], a: int, b: int, cfg: KalmanConfig, out: list) -> None:
    """RTS over steps[a:b], one filter run without a restart; writes into out[a:b]."""
    seg = steps[a:b]
    last_fix = max((i for i, s in enumerate(seg) if s.kind in _FIX), default=-1)
    xs, Ps = seg[-1].x.copy(), seg[-1].P.copy()
    smoothed = [(xs, Ps)]
    for k in range(len(seg) - 2, -1, -1):
        x, P = seg[k].x, seg[k].P
        F, Q = _transition(seg[k + 1].t - seg[k].t, cfg.sigma_acc)
        x_pred, P_pred = F @ x, F @ P @ F.T + Q
        C = np.linalg.solve(P_pred, F @ P).T              # P Fᵀ P_pred⁻¹ (P_pred symmetric)
        xs = x + C @ (xs - x_pred)
        Ps = P + C @ (Ps - P_pred) @ C.T
        Ps = (Ps + Ps.T) / 2
        smoothed.append((xs, Ps))
    smoothed.reverse()
    for i, (s, (x, P)) in enumerate(zip(seg, smoothed, strict=True)):
        source = "fix" if s.kind in _FIX else ("interpolated" if i < last_fix else "extrapolated")
        out[a + i] = Smoothed(s.t, x, P, source)
