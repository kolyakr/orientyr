"""GPS spoof detector (M5-07): does the GPS still agree with what the camera sees?

The filter (M5-01) knows where the drone is from the ground under it, with a covariance. A GPS receiver says where
it is too. Jamming makes the receiver go silent, which is easy to notice; **spoofing** makes it lie, and a careful
spoofer drags the reported position away slowly, so no single reading looks wrong. The two sources are
independent, so their difference tells:

- **normalised gap**: the Mahalanobis distance d² = gᵀ S⁻¹ g of the gap g = GPS − filter position, where
  S = P_pos + σ_gps² I adds both uncertainties. For an honest receiver d² follows χ² with 2 degrees of freedom
  (mean 2); it is above 13.8 once in a thousand frames. Normalising matters: 30 px is an alarm when the filter is
  sure to 2 px, and nothing at all while it coasts with σ = 50 px;
- **N frames in a row**: one frame above the threshold is SUSPECT. Only ``n_frames`` of them in a row raise the
  alarm (SPOOFED): a glitch of the receiver or one bad fix of ours passes, a walk-off does not come back. The
  alarm clears the same way, after ``n_frames`` good frames in a row;
- a frame without GPS, or before the filter has a position, is NO_DATA: no evidence either way, so the count and
  the alarm hold.

The result goes into the telemetry as ``Record.extra["gps"]`` (:func:`annotate_run`), the extension point that
``run.jsonl`` keeps for later milestones, so the file format does not change::

    "extra": {"gps": {"x": 1890.2, "y": 1321.7, "lon": 30.2217, "lat": 50.5264,     # the reading, map px and WGS84
                      "gap_px": 12.9, "d2": 9.6, "state": "SUSPECT", "alarm": false, "streak": 2}}

``x``, ``y``, ``gap_px`` and ``d2`` are null on a NO_DATA frame; ``lon``/``lat`` are null without a georef.
:func:`score_trial` reads that block back to measure one flight: false alarms and the detection delay
(``scripts/spoof_check.py`` runs it over many seeded flights).
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from dronenav.sim.gps import GpsSample
from dronenav.telemetry import Record

__all__ = ["SpoofConfig", "SpoofDetector", "SpoofReading", "SpoofState", "TrialScore", "annotate_run", "gps_extra",
           "score_trial"]


@dataclass(frozen=True)
class SpoofConfig:
    """gps_sigma_px: the receiver's 1-σ noise per axis, map px (4 px is about 3 m on the zoom-17 Irpin maps).
    threshold_chi2: d² above which a frame is suspect (χ², 2 dof: 9.21 = 99 %, 13.8 = 99.9 %, as the filter's gate).
    n_frames: frames in a row above the threshold that raise the alarm, and below it that clear it."""

    gps_sigma_px: float = 4.0
    threshold_chi2: float = 13.8
    n_frames: int = 5

    def __post_init__(self):
        for name in ("gps_sigma_px", "threshold_chi2"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be a finite number > 0, got {v!r}")
        if isinstance(self.n_frames, bool) or not isinstance(self.n_frames, int) or self.n_frames < 1:
            raise ValueError(f"n_frames must be an integer >= 1, got {self.n_frames!r}")


class SpoofState(str, Enum):
    """The detector's verdict on one frame. A str enum, so it serialises to its name in JSON."""

    NO_DATA = "NO_DATA"    # no GPS reading or no filter position this frame: nothing to compare
    OK = "OK"              # GPS and the filter agree within their uncertainties
    SUSPECT = "SUSPECT"    # the gap is above the threshold, but not for long enough to call it
    SPOOFED = "SPOOFED"    # the alarm: above the threshold n_frames in a row, and not cleared yet


@dataclass(frozen=True)
class SpoofReading:
    """One frame's result. alarm: the alarm is on after this frame (it also holds through NO_DATA frames).
    d2: the normalised gap; gap_px: the plain distance GPS to filter, map px; both None on NO_DATA.
    streak: frames in a row above the threshold so far (0 after a good frame)."""

    state: SpoofState
    alarm: bool
    d2: float | None
    gap_px: float | None
    streak: int


def _xy(name: str, v) -> np.ndarray:
    try:
        a = np.asarray(v, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be (x, y) in map px, got {v!r}") from None
    if a.shape != (2,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be two finite numbers (x, y) in map px, got {v!r}")
    return a


def _cov(v) -> np.ndarray:
    if v is None:
        raise ValueError("cov is missing: the filter's position needs its 2 x 2 covariance")
    try:
        a = np.asarray(v, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"cov must be a 2 x 2 covariance in map px², got {v!r}") from None
    if a.shape != (2, 2) or not np.isfinite(a).all():
        raise ValueError(f"cov must be a finite 2 x 2 covariance in map px², got shape {a.shape}")
    a = (a + a.T) / 2
    if np.linalg.eigvalsh(a).min() < 0:            # zero is fine: the GPS noise keeps S invertible
        raise ValueError(f"cov must be a covariance (no negative variance), got {a.tolist()}")
    return a


class SpoofDetector:
    """Compares GPS with the filter frame by frame. Use ``update`` once per frame, in time order."""

    def __init__(self, cfg: SpoofConfig | None = None):
        self.cfg = cfg if cfg is not None else SpoofConfig()
        self._above = 0          # frames in a row above the threshold
        self._below = 0          # frames in a row at or below it
        self._alarm = False

    @property
    def alarm(self) -> bool:
        return self._alarm

    def reset(self) -> None:
        """Forget the count and the alarm (a new flight, or the operator has dealt with it)."""
        self._above = self._below = 0
        self._alarm = False

    def update(self, gps_xy, track_xy, track_cov) -> SpoofReading:
        """One frame: the GPS position and the filter's position with its covariance, all in map px.

        ``gps_xy`` None (the receiver is silent) or ``track_xy`` None (the filter has no state yet) gives NO_DATA.
        Bad values raise ValueError and leave the detector as it was.
        """
        if gps_xy is None or track_xy is None:
            return SpoofReading(SpoofState.NO_DATA, self._alarm, None, None, self._above)
        gap = _xy("gps", gps_xy) - _xy("position", track_xy)
        S = _cov(track_cov) + self.cfg.gps_sigma_px ** 2 * np.eye(2)
        d2 = float(gap @ np.linalg.solve(S, gap))
        if d2 > self.cfg.threshold_chi2:
            self._above, self._below = self._above + 1, 0
            if self._above >= self.cfg.n_frames:
                self._alarm = True
        else:
            self._above, self._below = 0, self._below + 1
            if self._below >= self.cfg.n_frames:
                self._alarm = False
        if self._alarm:
            state = SpoofState.SPOOFED
        else:
            state = SpoofState.SUSPECT if self._above else SpoofState.OK
        return SpoofReading(state, self._alarm, d2, float(np.hypot(*gap)), self._above)


def gps_extra(sample: GpsSample | None, reading: SpoofReading) -> dict:
    """The JSON-ready block for ``Record.extra["gps"]`` (see the module docstring for its shape)."""
    return {"x": None if sample is None else float(sample.x), "y": None if sample is None else float(sample.y),
            "lon": None if sample is None or sample.lon is None else float(sample.lon),
            "lat": None if sample is None or sample.lat is None else float(sample.lat),
            "gap_px": reading.gap_px, "d2": reading.d2, "state": reading.state.value, "alarm": reading.alarm,
            "streak": reading.streak}


def annotate_run(records: Iterable[Record], gps: Iterable[GpsSample], cfg: SpoofConfig | None = None) -> list[Record]:
    """Run the detector over a run and return new records with ``extra["gps"]`` filled in.

    Each record's filtered position and covariance (``record.track``) is compared with the GPS sample of the same
    frame ``idx``; a record without a sample is a NO_DATA frame. The input records are not changed, and whatever
    else is in their ``extra`` is kept. Raises ValueError when no record has a sample: the track of another flight.
    """
    records = list(records)
    by_idx = {s.idx: s for s in gps}
    if records and not any(r.idx in by_idx for r in records):
        raise ValueError(f"no GPS sample matches any of the run's {len(records)} frames: is it the GPS track of "
                         "this flight?")
    det, out = SpoofDetector(cfg), []
    for rec in records:
        s = by_idx.get(rec.idx)
        tr = rec.track
        reading = det.update(None if s is None else (s.x, s.y), (tr.x, tr.y), tr.cov)
        out.append(dataclasses.replace(rec, extra={**rec.extra, "gps": gps_extra(s, reading)}))
    return out


@dataclass(frozen=True)
class TrialScore:
    """One flight, scored. frames: records looked at. false_alarm_frames: frames with the alarm on while the GPS was
    honest (before the walk-off; the whole flight when there is none); false_alarm: any of them.
    detected: the alarm came on after the walk-off began (None for an honest flight; False also when the alarm was
    already on as it began, which is a false alarm, not a detection). alarm_idx: the frame it came on.
    delay_frames, delay_s: from the walk-off's first frame to that frame, in records and in seconds."""

    frames: int
    false_alarm: bool
    false_alarm_frames: int
    detected: bool | None = None
    alarm_idx: int | None = None
    delay_frames: int | None = None
    delay_s: float | None = None


def score_trial(records: Sequence[Record], start_idx: int | None = None) -> TrialScore:
    """False alarms and detection delay of one annotated run (see :func:`annotate_run`).

    start_idx: the walk-off's ``start_idx``; None for an honest flight, where every alarm is false.
    Raises ValueError if a record has no ``extra["gps"]`` block or the walk-off starts after the last record.
    """
    alarms = []
    for rec in records:
        g = rec.extra.get("gps")
        alarm = g.get("alarm") if isinstance(g, dict) else None
        if alarm is not True and alarm is not False:      # bad data in a run, so a ValueError like the rest
            raise ValueError(f"frame {rec.idx} has no extra['gps'] block with an alarm: annotate the run first")
        alarms.append(alarm)
    if start_idx is None:
        return TrialScore(frames=len(records), false_alarm=any(alarms), false_alarm_frames=sum(alarms))
    first = next((k for k, rec in enumerate(records) if rec.idx >= start_idx), None)
    if first is None:
        raise ValueError(f"start_idx {start_idx} is after the run's last frame: the walk-off never started")
    false_frames = sum(alarms[:first])
    score = {"frames": len(records), "false_alarm": false_frames > 0, "false_alarm_frames": false_frames}
    if first > 0 and alarms[first - 1]:           # already on: whatever follows is that false alarm continuing
        return TrialScore(**score, detected=False)
    hit = next((k for k in range(first, len(records)) if alarms[k]), None)
    if hit is None:
        return TrialScore(**score, detected=False)
    return TrialScore(**score, detected=True, alarm_idx=records[hit].idx, delay_frames=hit - first,
                      delay_s=records[hit].t - records[first].t)
