"""The tracker's state machine (M5-02): TRACKING / LOST / RELOCALIZING on top of the Kalman filter (M5-01).

The filter knows where the drone is and how unsure it is; the state machine decides how much to trust that, and
therefore how the next frame is searched (docs/proposal.md "The state machine", docs/SPEC.md 4.3):

- **LOST** (the start, and after ``lost_after`` misses in a row): the prediction is not trusted, the frame is
  searched on the whole map (GLOBAL). An accepted fix is a *candidate*: the filter restarts there -> RELOCALIZING.
- **RELOCALIZING**: one fix may be a coincidence (a look-alike block elsewhere), so wait for a second one that
  agrees with it (passes the filter's gate) -> TRACKING. A fix that disagrees becomes the new candidate; a miss
  -> LOST.
- **TRACKING**: fixes that pass the gate update the filter and reset the miss count. A miss (no fix, a rejected
  one) or a gated outlier counts one; the filter coasts on its prediction and stays TRACKING until the
  ``lost_after``-th miss in a row -> LOST. Counting gated fixes is also the safety net against lock-out: a filter
  that has drifted and gates every true fix is reset by the next global search.

Every state but LOST searches a window around the prediction (WINDOW).

A long loss must not fly on a stale velocity: ``hold_after_s`` seconds after the last fix the velocity is set to
zero and the position holds there, while the uncertainty keeps growing (as M1-09's interim rule did).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from dronenav.confidence import fix_covariance
from dronenav.search import SearchMode
from dronenav.tracking.kalman import KalmanConfig, KalmanFilter, Step, StepKind
from dronenav.types import Fix, TrackMode, TrackState

__all__ = ["Tracker", "TrackerConfig", "TrackerUpdate"]


@dataclass(frozen=True)
class TrackerConfig:
    """lost_after: misses in a row (no fix, a rejected or a gated one) that make a track LOST (3: the proposal).
    hold_after_s: seconds without a fix after which the position stops coasting and holds.
    kalman: the filter's setting (M5-01)."""

    lost_after: int = 3
    hold_after_s: float = 5.0
    kalman: KalmanConfig = field(default_factory=KalmanConfig)

    def __post_init__(self):
        if self.lost_after < 1:
            raise ValueError(f"lost_after must be >= 1, got {self.lost_after}")
        if not self.hold_after_s >= 0:
            raise ValueError(f"hold_after_s must be >= 0, got {self.hold_after_s}")


class TrackerUpdate(NamedTuple):
    """One frame's result. mode: the state after the frame. step: what the filter did with the fix.
    track: the belief for the telemetry, None while there is no position at all (LOST before the first fix).
    misses: misses in a row so far."""

    mode: TrackMode
    step: Step
    track: TrackState | None
    misses: int


class Tracker:
    """The Kalman filter plus the state machine. Per frame, in time order: read :attr:`search` to choose how to
    search the frame, then pass the result to :meth:`update`."""

    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.kf = KalmanFilter(self.cfg.kalman)
        self.mode = TrackMode.LOST
        self.misses = 0
        self._t_fix: float | None = None          # time of the last fix the filter took

    @property
    def search(self) -> SearchMode:
        """How the next frame should be searched: the whole map when LOST, else a window around the prediction."""
        return SearchMode.GLOBAL if self.mode is TrackMode.LOST else SearchMode.WINDOW

    def update(self, idx: int, t: float, fix: Fix | None) -> TrackerUpdate:
        """Use frame ``idx`` (time ``t`` s) and its fix (None, rejected or accepted); returns the new state."""
        t = float(t)
        usable = fix is not None and fix.accepted
        if self.mode is TrackMode.LOST and usable:
            step = self._restart(t, fix)          # a candidate: the stale prediction is not blended in
            self.mode = TrackMode.RELOCALIZING
        elif not self.kf.initialized:
            step = self.kf.step(t, None)          # NONE: nothing to start from yet
        else:
            step = self._step(t, fix if usable else None)
            if step.kind is StepKind.HIT:
                self.mode, self.misses = TrackMode.TRACKING, 0
            elif self.mode is TrackMode.RELOCALIZING:
                if step.kind is StepKind.GATED:   # disagrees with the candidate: the newer fix is the candidate
                    step = self._restart(t, fix)
                else:                             # no confirmation: back to searching everywhere
                    self.mode, self.misses = TrackMode.LOST, 1
            else:
                self.misses += 1
                if self.misses >= self.cfg.lost_after:
                    self.mode = TrackMode.LOST
        track = self.kf.track_state(idx, self.mode, fix) if self.kf.initialized else None
        return TrackerUpdate(self.mode, step, track, self.misses)

    def _restart(self, t: float, fix: Fix) -> Step:
        self.kf.reset(t, (fix.x, fix.y), fix_covariance(fix))
        self._t_fix, self.misses = t, 0
        return Step(StepKind.INIT, t, self.kf.x.copy(), self.kf.P.copy(), self.kf.x.copy(), self.kf.P.copy(),
                    None)

    def _step(self, t: float, fix: Fix | None) -> Step:
        """A filter step, holding the position once ``hold_after_s`` has passed without a fix."""
        hold = self._t_fix + self.cfg.hold_after_s
        if fix is None and t >= hold:
            if self.kf.t < hold:
                self.kf.step(hold)                # coast up to the hold time, then stop there
            self.kf.x[2:] = 0.0
        step = self.kf.step(t, fix)
        if step.kind is StepKind.HIT:
            self._t_fix = t
        return step
