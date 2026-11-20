"""M5-02 · Tracker state machine: TRACKING / LOST / RELOCALIZING, the search each frame runs, the telemetry state.

Done when: a scripted sequence of hits and misses produces the expected state sequence (test).

Rules (docs/proposal.md "The state machine", docs/SPEC.md 4.3):
- the run starts LOST: there is no prediction, so the frame is searched on the whole map (GLOBAL);
- LOST + an accepted fix -> RELOCALIZING: a candidate, the filter restarts there (the old prediction is not trusted);
- RELOCALIZING + a fix that agrees (passes the gate) -> TRACKING; one that disagrees becomes the new candidate;
  a miss -> LOST;
- TRACKING: a fix that passes the gate resets the miss count; a miss or a gated outlier counts; after
  ``lost_after`` (3) in a row -> LOST. While coasting the state stays TRACKING and the filter predicts.
- every state but LOST searches a window around the prediction (WINDOW); LOST searches the whole map (GLOBAL).
"""
import itertools
import math

import numpy as np
import pytest

from dronenav.search import SearchMode
from dronenav.tracking.kalman import KalmanConfig, StepKind
from dronenav.tracking.states import Tracker, TrackerConfig
from dronenav.types import Fix, TrackMode

T, L, R = TrackMode.TRACKING, TrackMode.LOST, TrackMode.RELOCALIZING
G, W = SearchMode.GLOBAL, SearchMode.WINDOW
VEL = (30.0, -12.0)                       # map px/s
START = (1000.0, 800.0)


def _fix(idx, xy, accepted=True):
    x, y = xy
    fp = np.array([[x - 160, y - 120], [x + 160, y - 120], [x + 160, y + 120], [x - 160, y + 120]], float)
    return Fix(idx=idx, H=np.eye(3), x=float(x), y=float(y), heading_deg=0.0, scale=1.0, footprint=fp,
               inliers=300, inlier_ratio=0.8, reproj_px=0.8, accepted=accepted,
               reject_reason=None if accepted else "too few inliers")


def _truth(t):
    return START[0] + VEL[0] * t, START[1] + VEL[1] * t


def _event(ev, i, t):
    if ev == "-":
        return None
    if ev == "o":
        return _fix(i, np.add(_truth(t), (500.0, 0.0)))
    return _fix(i, _truth(t), accepted=ev == "h")


def _run(script, cfg=None):
    """Feed a script of events at 1 s steps: 'h' a true fix, '-' no fix, 'r' a rejected fix, 'o' an outlier
    500 px off. Returns (search asked before each frame, state after each frame, updates)."""
    tr = Tracker(cfg or TrackerConfig())
    searches, states, ups = [], [], []
    for i, ev in enumerate(script):
        searches.append(tr.search)
        t = float(i)
        up = tr.update(i, t, _event(ev, i, t))
        states.append(up.mode)
        ups.append(up)
    return searches, states, ups


# --- Done when: a scripted sequence of hits and misses -> the expected state sequence ---------------------------

def test_a_scripted_flight_gives_the_expected_states_and_searches():
    #        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
    script = "-hhh--h---" "-hhoh"
    searches, states, _ = _run(script)
    assert states == [L, R, T, T, T, T, T, T, T, L, L, R, T, T, T]
    # the search for a frame is decided before it, by the state after the previous one
    assert searches == [G, G, W, W, W, W, W, W, W, W, G, G, W, W, W]


def test_the_step_kinds_follow_the_script():
    _, _, ups = _run("-hh-ho")
    assert [u.step.kind for u in ups] == [StepKind.NONE, StepKind.INIT, StepKind.HIT, StepKind.COAST, StepKind.HIT,
                                         StepKind.GATED]
    assert [u.misses for u in ups] == [0, 0, 0, 1, 0, 1]


# --- each rule on its own ---------------------------------------------------------------------------------------

def test_the_run_starts_lost_and_searches_the_whole_map():
    tr = Tracker()
    assert tr.mode is L and tr.search is G
    up = tr.update(0, 0.0, None)
    assert up.mode is L and up.track is None                          # no position yet: nothing to report


def test_a_candidate_that_a_second_fix_contradicts_is_replaced():
    tr = Tracker()
    tr.update(0, 0.0, _fix(0, (1000.0, 800.0)))
    up = tr.update(1, 1.0, _fix(1, (3000.0, 200.0)))                  # far from the first: one of them is wrong
    assert up.mode is R and up.step.kind is StepKind.INIT             # the newer one is the candidate now
    assert (up.track.x, up.track.y) == (3000.0, 200.0)
    up = tr.update(2, 2.0, _fix(2, (3010.0, 195.0)))                  # and this one agrees with it
    assert up.mode is T and up.step.kind is StepKind.HIT


def test_a_miss_while_relocalizing_is_lost_again():
    _, states, _ = _run("h-")
    assert states == [R, L]


def test_a_rejected_fix_is_a_miss():
    _, states, _ = _run("hhrrr")
    assert states == [R, T, T, T, L]


def test_repeated_outliers_lose_the_track_and_the_next_fix_relocalizes_there():
    # the safety net against lock-out: a filter that gates every fix would otherwise never come back
    tr = Tracker()
    for i in range(3):
        tr.update(i, float(i), _fix(i, _truth(i)))
    # 3000 px: clearly outside the gate even after 3 s of coasting (σ grows to ~300 px at the default sigma_acc)
    far = [(np.add(_truth(i), (3000.0, 0.0))) for i in range(3, 7)]
    modes = [tr.update(i, float(i), _fix(i, xy)).mode for i, xy in zip(range(3, 6), far, strict=False)]
    assert modes == [T, T, L]
    up = tr.update(6, 6.0, _fix(6, far[3]))
    assert up.mode is R and (up.track.x, up.track.y) == (pytest.approx(far[3][0]), pytest.approx(far[3][1]))


def test_lost_after_is_configurable():
    _, states, _ = _run("hh-", TrackerConfig(lost_after=1))
    assert states == [R, T, L]
    with pytest.raises(ValueError):
        TrackerConfig(lost_after=0)
    with pytest.raises(ValueError):
        TrackerConfig(hold_after_s=-1.0)


def test_a_fix_after_lost_restarts_the_filter_at_the_fix():
    # the stale prediction is not blended in: after LOST the fix is the position, and the speed is unknown again
    tr = Tracker()
    for i, ev in enumerate("hhh---"):
        tr.update(i, float(i), _fix(i, _truth(i)) if ev == "h" else None)
    assert tr.mode is L
    up = tr.update(6, 6.0, _fix(6, (200.0, 300.0)))
    assert up.mode is R and up.step.kind is StepKind.INIT
    assert (up.track.x, up.track.y, up.track.vx, up.track.vy) == (200.0, 300.0, 0.0, 0.0)


# --- telemetry ------------------------------------------------------------------------------------------------

def test_the_update_carries_the_telemetry_state():
    tr = Tracker()
    tr.update(0, 0.0, _fix(0, _truth(0)))
    fix = _fix(1, _truth(1))
    up = tr.update(1, 1.0, fix)
    tk = up.track
    assert tk.idx == 1 and tk.state is T and tk.fix is fix
    assert math.hypot(tk.x - _truth(1)[0], tk.y - _truth(1)[1]) < 1.0
    assert tk.cov.shape == (2, 2) and np.allclose(tk.cov, tr.kf.P[:2, :2])
    coast = tr.update(2, 2.0, None).track
    assert coast.fix is None and coast.cov[0, 0] > tk.cov[0, 0]       # coasting: less sure


def test_coasting_holds_after_a_few_seconds():
    # a long loss must not fly on a stale velocity off the map (as M1-09's rule): after hold_after_s without a fix
    # the position holds, while the uncertainty keeps growing
    cfg = TrackerConfig(hold_after_s=3.0)
    tr = Tracker(cfg)
    for i in range(4):
        tr.update(i, float(i), _fix(i, _truth(i)))
    vx = tr.kf.x[2]
    assert vx == pytest.approx(VEL[0], abs=2.0)
    tracks = [tr.update(i, float(i), None).track for i in range(4, 12)]
    held = tracks[3]                                                   # t = 7: 4 s after the last fix (t = 3)
    assert held.x == pytest.approx(tracks[0].x + 2 * vx, abs=1.0)     # coasted until t = 6, then held
    for tk in tracks[3:]:
        assert (tk.x, tk.y, tk.vx, tk.vy) == (pytest.approx(held.x), pytest.approx(held.y), 0.0, 0.0)
    sig = [tk.cov[0, 0] for tk in tracks]
    assert all(b > a for a, b in itertools.pairwise(sig))


def test_the_kalman_setting_is_passed_through():
    tr = Tracker(TrackerConfig(kalman=KalmanConfig(gate_chi2=1e9)))   # a gate that takes anything
    _, _, ups = _run("hho", TrackerConfig(kalman=KalmanConfig(gate_chi2=1e9)))
    assert ups[2].step.kind is StepKind.HIT and tr.kf.cfg.gate_chi2 == 1e9
