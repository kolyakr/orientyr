"""The tracker's state machine (M5-02): TRACKING / LOST / RELOCALIZING on top of the Kalman filter (M5-01).

The filter knows where the drone is and how unsure it is; the state machine decides how much to trust that, and
therefore how the next frame is searched (docs/proposal.md "The state machine", docs/SPEC.md 4.3):

- **LOST** (the start, and after ``lost_after`` misses in a row): the prediction is not trusted, the frame is
  searched on the whole map (GLOBAL). An accepted fix is a *candidate*: the filter restarts there -> RELOCALIZING.
- **RELOCALIZING**: one fix may be a coincidence (a look-alike block elsewhere), so wait for a second one that
  agrees with it (passes the filter's gate) -> TRACKING. A fix that disagrees becomes the new candidate. Its speed
  is unknown, but at least as uncertain as the speed the two candidates imply (M5-06): otherwise a drone moving
  faster than the filter's initial speed σ puts every next fix outside the gate and is never confirmed, and a
  look-alike before a true fix still lets the next true fix confirm at once. A miss
  -> LOST.
- **TRACKING**: fixes that pass the gate update the filter and reset the miss count. A miss (no fix, a rejected
  one) or a gated outlier counts one; the filter coasts on its prediction and stays TRACKING until the
  ``lost_after``-th miss in a row -> LOST. Counting gated fixes is also the safety net against lock-out: a filter
  that has drifted and gates every true fix is reset by the next global search.

Every state but LOST searches a window around the prediction (WINDOW): :meth:`Tracker.predict` gives where the
drone is expected at the next frame and how sure that is, and the last fix's footprint gives the window's size
(M5-03, ``dronenav.search.prediction_window``).

A long loss must not fly on a stale velocity: ``hold_after_s`` seconds after the last fix the velocity is set to
zero and the position holds there, while the uncertainty keeps growing (as M1-09's interim rule did).

Visual odometry (M9-02, ``dronenav.vo``): ``update(..., vo=)`` also takes the velocity measured between the frame and
the one before, which the filter uses after its step (``KalmanFilter.update_velocity``). While VO measures, the
position does not hold (the velocity is measured, not stale), and a frame without a fix is not a miss: the track
keeps TRACKING on VO for up to ``vo_max_s`` seconds after the last fix, because chained frame-to-frame motion
drifts and only a map match bounds it. A gated fix is still a miss: it is evidence that something disagrees.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from dronenav.confidence import fix_covariance
from dronenav.search import SearchMode
from dronenav.tracking.kalman import KalmanConfig, KalmanFilter, Step, StepKind, _transition
from dronenav.types import Fix, TrackMode, TrackState
from dronenav.vo import VoVelocity

__all__ = ["Prediction", "Tracker", "TrackerConfig", "TrackerUpdate"]


@dataclass(frozen=True)
class TrackerConfig:
    """lost_after: misses in a row (no fix, a rejected or a gated one) that make a track LOST (3: the proposal).
    hold_after_s: seconds without a fix (or a VO velocity) after which the position stops coasting and holds.
    vo_max_s: seconds after the last fix that VO alone keeps a track TRACKING (misses are not counted meanwhile).
    kalman: the filter's setting (M5-01)."""

    lost_after: int = 3
    hold_after_s: float = 5.0
    vo_max_s: float = 30.0
    kalman: KalmanConfig = field(default_factory=KalmanConfig)

    def __post_init__(self):
        if self.lost_after < 1:
            raise ValueError(f"lost_after must be >= 1, got {self.lost_after}")
        if not self.hold_after_s >= 0:
            raise ValueError(f"hold_after_s must be >= 0, got {self.hold_after_s}")
        if not self.vo_max_s >= 0:
            raise ValueError(f"vo_max_s must be >= 0, got {self.vo_max_s}")


class TrackerUpdate(NamedTuple):
    """One frame's result. mode: the state after the frame. step: what the filter did with the fix.
    track: the belief for the telemetry, None while there is no position at all (LOST before the first fix).
    misses: misses in a row so far."""

    mode: TrackMode
    step: Step
    track: TrackState | None
    misses: int


class Prediction(NamedTuple):
    """Where the tracker expects the drone at a time: the centre (x, y) in map px and its 2 x 2 covariance (px²)."""

    x: float
    y: float
    cov: np.ndarray


class Tracker:
    """The Kalman filter plus the state machine. Per frame, in time order: read :attr:`search` to choose how to
    search the frame, then pass the result to :meth:`update`."""

    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.kf = KalmanFilter(self.cfg.kalman)
        self.mode = TrackMode.LOST
        self.misses = 0
        self._t_fix: float | None = None          # time of the last fix the filter took
        self._t_vo: float | None = None           # time of the last VO velocity the filter took (M9-02)
        self.last_fix: Fix | None = None          # that fix: its footprint sizes the search window (M5-03)
        self._cand: tuple[float, Fix] | None = None   # the last restart (time, fix): the candidate's speed bound

    @property
    def search(self) -> SearchMode:
        """How the next frame should be searched: the whole map when LOST, else a window around the prediction."""
        return SearchMode.GLOBAL if self.mode is TrackMode.LOST else SearchMode.WINDOW

    def predict(self, t: float) -> Prediction | None:
        """Where the drone should be at time ``t`` if no fix comes before: exactly what :meth:`update` would give
        without a fix (the same hold after ``hold_after_s``), so the frame is searched where the telemetry would
        put it. None when LOST or before the first fix: there is no prediction to search around. The tracker is
        not changed."""
        if self.mode is TrackMode.LOST or not self.kf.initialized:
            return None
        t, sigma_acc = float(t), self.cfg.kalman.sigma_acc
        x, P, t0 = self.kf.x, self.kf.P, self.kf.t
        if t < t0:
            raise ValueError(f"time must not go back: t={t} after t={t0}")
        hold = self._hold_time()
        if t >= hold:                             # as _step: coast up to the hold time, then stop there
            if t0 < hold:
                F, Q = _transition(hold - t0, sigma_acc)
                x, P, t0 = F @ x, F @ P @ F.T + Q, hold
            x = x.copy()
            x[2:] = 0.0
        F, Q = _transition(t - t0, sigma_acc)
        x, P = F @ x, F @ P @ F.T + Q
        return Prediction(float(x[0]), float(x[1]), P[:2, :2].copy())

    def update(self, idx: int, t: float, fix: Fix | None, vo: VoVelocity | None = None) -> TrackerUpdate:
        """Use frame ``idx`` (time ``t`` s), its fix (None, rejected or accepted) and, optionally, the VO velocity
        since the previous frame (M9-02); returns the new state."""
        t = float(t)
        usable = fix is not None and fix.accepted
        if self.mode is TrackMode.LOST and usable:
            step = self._restart(t, fix)          # a candidate: the stale prediction is not blended in
            self.mode = TrackMode.RELOCALIZING
        elif not self.kf.initialized:
            step = self.kf.step(t, None)          # NONE: nothing to start from yet
        else:
            step = self._step(t, fix if usable else None, vo)
            on_vo = (step.kind is StepKind.COAST and self._t_vo == t
                     and t - self._t_fix <= self.cfg.vo_max_s)
            if step.kind is StepKind.HIT:
                self.mode, self.misses = TrackMode.TRACKING, 0
            elif self.mode is TrackMode.RELOCALIZING:
                if step.kind is StepKind.GATED:   # disagrees with the candidate: the newer fix is the candidate
                    step = self._restart(t, fix, prev=self._cand)
                else:                             # no confirmation: back to searching everywhere
                    self.mode, self.misses = TrackMode.LOST, 1
            elif not on_vo:                       # VO carries a track without fixes for vo_max_s, not longer
                self.misses += 1
                if self.misses >= self.cfg.lost_after:
                    self.mode = TrackMode.LOST
        track = self.kf.track_state(idx, self.mode, fix) if self.kf.initialized else None
        return TrackerUpdate(self.mode, step, track, self.misses)

    def _restart(self, t: float, fix: Fix, prev: tuple[float, Fix] | None = None) -> Step:
        """Restart the filter at ``fix``, speed unknown (zero, σ = the filter's initial speed σ). With ``prev``, the
        previous candidate (time, fix) this one disagreed with, the speed σ is at least the speed between the two:
        one of them may be a look-alike, so the speed is not taken from them, only how large it may be."""
        self.kf.reset(t, (fix.x, fix.y), fix_covariance(fix))
        if prev is not None and t > prev[0]:
            t0, f0 = prev
            speed = math.hypot(fix.x - f0.x, fix.y - f0.y) / (t - t0)
            self.kf.P[2, 2] = self.kf.P[3, 3] = max(self.cfg.kalman.init_speed_sigma, speed) ** 2
        self._t_fix, self._t_vo, self.misses, self.last_fix, self._cand = t, None, 0, fix, (t, fix)
        return Step(StepKind.INIT, t, self.kf.x.copy(), self.kf.P.copy(), self.kf.x.copy(), self.kf.P.copy(),
                    None)

    def _hold_time(self) -> float:
        """When the position starts to hold: ``hold_after_s`` after the last fix or VO velocity."""
        return max(self._t_fix, self._t_vo or self._t_fix) + self.cfg.hold_after_s

    def _step(self, t: float, fix: Fix | None, vo: VoVelocity | None = None) -> Step:
        """A filter step, holding the position once ``hold_after_s`` has passed without a fix or VO; then the VO
        velocity, if any (the step's state is replaced by the one after it, so the smoother sees what was used)."""
        hold = self._hold_time()
        if fix is None and vo is None and t >= hold:
            if self.kf.t < hold:
                self.kf.step(hold)                # coast up to the hold time, then stop there
            self.kf.x[2:] = 0.0
        step = self.kf.step(t, fix)
        if step.kind is StepKind.HIT:
            self._t_fix, self.last_fix = t, fix
        if vo is not None and self.kf.update_velocity(vo.v, vo.cov):
            self._t_vo = t
            step = step._replace(x=self.kf.x.copy(), P=self.kf.P.copy())
        return step
