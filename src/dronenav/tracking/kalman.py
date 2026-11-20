"""Constant-velocity Kalman filter on the map (M5-01).

Each frame's fix is a noisy, occasionally wrong measurement of where the drone is. The filter keeps a belief, the
state ``x = (x, y, vx, vy)`` in map px and px/s with a 4 x 4 covariance ``P``, and moves it frame by frame:

- **predict** to the frame's timestamp: the drone keeps its velocity, and the uncertainty grows by the process
  noise of a random acceleration (``sigma_acc``). The time step comes from the timestamps, so dropped or
  unevenly spaced frames are handled exactly.
- **gate**: a fix is compared with the prediction by its Mahalanobis distance d² = yᵀ S⁻¹ y, where y is the
  innovation (fix − predicted position) and S = P_pos + R combines both uncertainties. Above ``gate_chi2`` (χ² with
  2 degrees of freedom; 13.8 = 99.9 %) the fix is too unlikely to be right: a wrong match far away. It is GATED and
  the state stays the prediction.
- **update** with an accepted fix and its own covariance R (M4-08 ``fix_covariance``): a weak fix moves the state
  less than a strong one. Joseph form, so P stays symmetric positive-definite.
- **coast** when there is no fix (or a rejected one): the prediction is the state, and σ keeps growing.

The first accepted fix starts the filter at that position with zero velocity and a wide speed prior. Each step
returns the prediction, the transition and the result, which is what a backward (RTS) smoother needs (M5-05).
Started from the prototype in ``experiments/kyiv_cluster/walkthrough.py``; unlike it, the time step comes from the
timestamps, the noise of each fix is its own, and the process noise model is a damped one (see ``_transition``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np

from dronenav.confidence import fix_covariance
from dronenav.types import Fix, TrackMode, TrackState

__all__ = ["KalmanConfig", "KalmanFilter", "Step", "StepKind"]

_H = np.eye(2, 4)                      # we measure position, not velocity


@dataclass(frozen=True)
class KalmanConfig:
    """sigma_acc: intensity of the random acceleration, map px/s^1.5: after dt seconds of coasting the position σ
    grows by sigma_acc·√(dt³/3). The default is measured on the Irpin flights at one fix per second, where turns
    and camera sway move the image centre unevenly (a 1 s prediction is off by 34 px median on mild, 70 px on
    harsh): at 80 the predicted 95 % circle holds the truth for 97 % (harsh) to 100 % (mild) of keyframes, and no
    good fix is gated. Much less locks the filter out: it gates good fixes and never recovers.
    init_speed_sigma: σ of the speed before any motion is seen, map px/s.
    gate_chi2: Mahalanobis d² above which a fix is rejected as an outlier (χ², 2 dof: 9.21 = 99 %, 13.8 = 99.9 %)."""

    sigma_acc: float = 80.0
    init_speed_sigma: float = 80.0
    gate_chi2: float = 13.8

    def __post_init__(self):
        if self.sigma_acc < 0:
            raise ValueError(f"sigma_acc must be >= 0, got {self.sigma_acc}")
        if not self.init_speed_sigma > 0 or not self.gate_chi2 > 0:
            raise ValueError(f"init_speed_sigma and gate_chi2 must be > 0, got {self.init_speed_sigma}, "
                             f"{self.gate_chi2}")


_CFG = KalmanConfig()


class StepKind(str, Enum):
    """What one step did with the frame's fix."""

    NONE = "NONE"      # not started yet and no fix to start from
    INIT = "INIT"      # the first accepted fix started (or restarted) the filter
    HIT = "HIT"        # the fix passed the gate and was used
    GATED = "GATED"    # the fix failed the gate: an outlier, ignored
    COAST = "COAST"    # no usable fix: the prediction is the state


class Step(NamedTuple):
    """One frame's step. x, P: the state after it. x_pred, P_pred: the prediction before the fix; F: the
    transition from the previous step (identity on INIT). d2: the fix's Mahalanobis distance (None if no fix
    was gated). x and P are None only for NONE."""

    kind: StepKind
    t: float
    x: np.ndarray | None
    P: np.ndarray | None
    x_pred: np.ndarray | None
    P_pred: np.ndarray | None
    F: np.ndarray | None
    d2: float | None = None


def _transition(dt: float, sigma_acc: float) -> tuple[np.ndarray, np.ndarray]:
    """F and Q over dt for constant velocity driven by continuous white-noise acceleration (intensity sigma_acc²).

    Not the piecewise-constant ("discrete white noise") Q = G Gᵀ σ² of the prototype: with precise fixes and a large
    σ that model drives the velocity gain to the stability limit (β → 2) and the velocity rings, flipping sign every
    frame. This one stays damped (measured on the Irpin flights)."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q = sigma_acc ** 2 * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
    Q = np.zeros((4, 4))
    for i in (0, 1):                                 # the same for x and y, independent
        Q[np.ix_([i, i + 2], [i, i + 2])] = q
    return F, Q


def _measurement(z, R) -> tuple[np.ndarray, np.ndarray] | None:
    """(position, covariance) from a Fix (accepted only, R from its evidence) or a raw position with R."""
    if z is None:
        return None
    if isinstance(z, Fix):
        if not z.accepted:
            return None
        return np.array([z.x, z.y], float), (fix_covariance(z) if R is None else np.asarray(R, float))
    if R is None:
        raise ValueError("a raw position needs its covariance R")
    return np.asarray(z, float).reshape(2), np.asarray(R, float).reshape(2, 2)


class KalmanFilter:
    """Constant-velocity filter on (x, y, vx, vy). Use ``step(t, fix)`` once per frame, in time order."""

    def __init__(self, cfg: KalmanConfig = _CFG):
        self.cfg = cfg
        self.t: float | None = None
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None

    @property
    def initialized(self) -> bool:
        return self.x is not None

    @property
    def sigma(self) -> float:
        """1-σ position uncertainty along the worst direction, map px."""
        self._need_state()
        return math.sqrt(float(np.linalg.eigvalsh(self.P[:2, :2]).max()))

    def _need_state(self) -> None:
        if not self.initialized:
            raise RuntimeError("the filter has no state yet: it starts on the first accepted fix")

    def reset(self, t: float, xy, cov) -> None:
        """(Re)start at a position with its covariance, speed unknown: the first fix, or after relocalising."""
        self.t = float(t)
        self.x = np.array([*np.asarray(xy, float).reshape(2), 0.0, 0.0])
        self.P = np.zeros((4, 4))
        self.P[:2, :2] = np.asarray(cov, float).reshape(2, 2)
        self.P[2, 2] = self.P[3, 3] = self.cfg.init_speed_sigma ** 2

    def predict_to(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """The state and covariance predicted at time t, without changing the filter."""
        self._need_state()
        dt = float(t) - self.t
        if dt < 0:
            raise ValueError(f"time must not go back: t={t} after t={self.t}")
        F, Q = _transition(dt, self.cfg.sigma_acc)
        return F @ self.x, F @ self.P @ F.T + Q

    def gate_d2(self, z, R) -> float:
        """Mahalanobis d² of position z (covariance R) against the current state, at the current time."""
        self._need_state()
        y = np.asarray(z, float).reshape(2) - _H @ self.x
        S = _H @ self.P @ _H.T + np.asarray(R, float).reshape(2, 2)
        return float(y @ np.linalg.solve(S, y))

    def step(self, t: float, z=None, R=None) -> Step:
        """Advance to time t and use the frame's measurement z: a Fix (R from its evidence unless given), a raw
        (x, y) with its covariance R, or None. Returns what was done (see ``StepKind``)."""
        m = _measurement(z, R)
        if not self.initialized:
            if m is None:
                return Step(StepKind.NONE, float(t), None, None, None, None, None)
            self.reset(t, *m)
            return Step(StepKind.INIT, self.t, self.x.copy(), self.P.copy(), self.x.copy(), self.P.copy(), np.eye(4))
        dt = float(t) - self.t
        if dt < 0:
            raise ValueError(f"time must not go back: t={t} after t={self.t}")
        F, Q = _transition(dt, self.cfg.sigma_acc)
        x_pred, P_pred = F @ self.x, F @ self.P @ F.T + Q
        self.t, self.x, self.P = float(t), x_pred, P_pred
        if m is None:
            return Step(StepKind.COAST, self.t, x_pred.copy(), P_pred.copy(), x_pred, P_pred, F)
        zpos, Rm = m
        d2 = self.gate_d2(zpos, Rm)
        if d2 > self.cfg.gate_chi2:
            return Step(StepKind.GATED, self.t, x_pred.copy(), P_pred.copy(), x_pred, P_pred, F, d2)
        S = _H @ P_pred @ _H.T + Rm
        K = np.linalg.solve(S, _H @ P_pred).T            # P Hᵀ S⁻¹ (S symmetric)
        self.x = x_pred + K @ (zpos - _H @ x_pred)
        IKH = np.eye(4) - K @ _H
        self.P = IKH @ P_pred @ IKH.T + K @ Rm @ K.T      # Joseph form: stays symmetric positive-definite
        self.P = (self.P + self.P.T) / 2
        return Step(StepKind.HIT, self.t, self.x.copy(), self.P.copy(), x_pred, P_pred, F, d2)

    def track_state(self, idx: int, mode: TrackMode, fix: Fix | None = None) -> TrackState:
        """The current belief as the telemetry's TrackState (the mode comes from the state machine, M5-02)."""
        self._need_state()
        return TrackState(idx=idx, state=mode, x=float(self.x[0]), y=float(self.x[1]), vx=float(self.x[2]),
                          vy=float(self.x[3]), cov=self.P[:2, :2].copy(), fix=fix)
