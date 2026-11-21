"""M5-03 · Search window from the prediction.

Done when: the window shrinks while tracking and grows while coasting (visible in the telemetry).

The rule, with the tracker on (it replaces M1's "last footprint + step + 25 %" window):
- the window is the last fix's footprint moved to the filter's predicted centre, grown on each axis by
  ``n_sigma`` (3) times that axis's predicted σ (never less than ``min_margin_px``), whole pixels, clipped to the
  map; None when it is (almost) off the map, as M1's window;
- the prediction is what the tracker's next update would give without a fix (the same hold after a long loss),
  so the window and the telemetry's position agree; LOST has no prediction (the whole map is searched);
- ``Searcher.locate_in`` searches a given window (None: the whole map), falling back to the whole map in the same
  frame when the window fails, as ``locate`` does.
The telemetry side (the window in run.jsonl shrinking and growing) is tested end to end in test_run.py.
"""
import itertools
import math

import numpy as np
import pytest
from tests.synth import make_h, make_map, render_frame

from dronenav.matching import RootSiftMatcher
from dronenav.search import SearchConfig, Searcher, SearchMode, prediction_window
from dronenav.tracking.kalman import KalmanConfig
from dronenav.tracking.states import Tracker, TrackerConfig
from dronenav.types import Fix, TrackMode

SQUARE = [(-50, -50), (50, -50), (50, 50), (-50, 50)]      # a 100 x 100 footprint around its centre
NO_FLOOR = SearchConfig(min_margin_px=0.0)
FRAME = (480, 360)


def _fix(x, y, footprint=SQUARE, idx=0, accepted=True) -> Fix:
    fp = np.asarray(footprint, float) + (x, y)
    return Fix(idx=idx, H=np.eye(3), x=x, y=y, heading_deg=0.0, scale=1.0, footprint=fp, inliers=100,
               inlier_ratio=0.5, reproj_px=1.0, accepted=accepted,
               reject_reason=None if accepted else "too few inliers")


# --- the window ------------------------------------------------------------------------------------------------

def test_the_window_is_the_footprint_at_the_prediction_grown_by_3_sigma():
    # footprint 100 x 100 at (500, 500); predicted at (600, 520) with σx 20, σy 10: grown by 60 and 30 px
    w = prediction_window(_fix(500, 500), (600.0, 520.0), np.diag([20.0 ** 2, 10.0 ** 2]), (2000, 2000), NO_FLOOR)
    assert w == (490, 440, 710, 600)


def test_more_uncertainty_is_a_bigger_window():
    areas = []
    for s in (5.0, 20.0, 80.0, 300.0):
        x0, y0, x1, y1 = prediction_window(_fix(1000, 1000), (1000.0, 1000.0), np.eye(2) * s ** 2, (2000, 2000),
                                           NO_FLOOR)
        areas.append((x1 - x0) * (y1 - y0))
    assert all(b > a for a, b in itertools.pairwise(areas))


def test_the_margin_has_the_m1_floor():
    # a near-certain prediction still leaves room for a turn of the footprint: min_margin_px (96 by default)
    w = prediction_window(_fix(500, 500), (500.0, 500.0), np.eye(2) * 1e-6, (2000, 2000))
    assert w == (354, 354, 646, 646)


def test_the_window_is_clipped_to_the_map_and_none_off_it():
    w = prediction_window(_fix(40, 30), (40.0, 30.0), np.eye(2) * 100.0, (1000, 800), NO_FLOOR)
    assert w == (0, 0, 120, 110)                                       # 40 + 50 + 30 = 120, 30 + 50 + 30 = 110
    assert all(isinstance(v, int) for v in w)
    assert prediction_window(_fix(40, 30), (5000.0, 5000.0), np.eye(2) * 100.0, (1000, 800), NO_FLOOR) is None


def test_n_sigma_is_a_setting():
    w = prediction_window(_fix(500, 500), (500.0, 500.0), np.eye(2) * 100.0, (2000, 2000),
                          SearchConfig(min_margin_px=0.0, n_sigma=1.0))
    assert w == (440, 440, 560, 560)
    with pytest.raises(ValueError, match="n_sigma"):
        SearchConfig(n_sigma=0.0)


# --- the prediction: what the tracker expects at the next frame -------------------------------------------------

def _tracker():
    return Tracker(TrackerConfig(kalman=KalmanConfig(sigma_acc=20.0)))


def _flight(track, n, vel=(30.0, -12.0), start=(1000.0, 800.0), t0=0):
    for k in range(t0, t0 + n):
        track.update(k, float(k), _fix(start[0] + vel[0] * k, start[1] + vel[1] * k, idx=k))


def test_no_prediction_before_a_fix_or_when_lost():
    track = _tracker()
    assert track.predict(0.0) is None and track.last_fix is None
    _flight(track, 3)
    assert track.predict(3.0) is not None
    for k in range(3, 6):                                              # lost_after 3 misses
        track.update(k, float(k), None)
    assert track.mode is TrackMode.LOST and track.predict(6.0) is None


def test_the_prediction_is_what_the_next_update_gives_without_a_fix():
    # the window must be searched where the telemetry will put the drone if the frame is not found:
    # this holds through the coast and the hold after hold_after_s (velocity stopped)
    track = Tracker(TrackerConfig(lost_after=20, hold_after_s=3.0))
    _flight(track, 5)
    for k in range(5, 14):
        p = track.predict(float(k))
        up = track.update(k, float(k), None)
        assert (p.x, p.y) == pytest.approx((up.track.x, up.track.y), abs=1e-6), k
        np.testing.assert_allclose(p.cov, up.track.cov, atol=1e-6)


def test_the_prediction_does_not_change_the_tracker():
    track = _tracker()
    _flight(track, 4)
    before = (track.kf.t, track.kf.x.copy(), track.kf.P.copy(), track.mode, track.misses)
    track.predict(10.0)
    assert track.kf.t == before[0] and track.mode is before[3] and track.misses == before[4]
    np.testing.assert_array_equal(track.kf.x, before[1])
    np.testing.assert_array_equal(track.kf.P, before[2])


def test_the_last_fix_gives_the_footprint():
    track = _tracker()
    _flight(track, 3)
    assert track.last_fix.idx == 2
    track.update(3, 3.0, None)
    track.update(4, 4.0, _fix(0, 0, idx=4, accepted=False))           # a rejected fix is not a footprint
    assert track.last_fix.idx == 2


def test_sigma_shrinks_while_tracking_and_grows_while_coasting():
    track = _tracker()
    sig = []
    for k in range(12):
        p = track.predict(float(k))
        if p is not None:
            sig.append(math.sqrt(max(p.cov[0, 0], p.cov[1, 1])))
        z = None if 6 <= k < 9 else _fix(1000 + 30 * k, 800 - 12 * k, idx=k)
        track.update(k, float(k), z)
    # sig[i] is the prediction for frame i + 1: frames 1-6 tracked, 6-8 coasted, 9 a fix again
    tracking, coasting = sig[0:6], sig[5:9]
    assert all(b < a for a, b in itertools.pairwise(tracking)), tracking
    assert all(b > a for a, b in itertools.pairwise(coasting)), coasting
    assert sig[9] < sig[8]                                             # a fix again: the window closes in


# --- searching a given window -----------------------------------------------------------------------------------

def test_locate_in_searches_the_given_window_and_falls_back(tmp_path):
    # M1-05's synthetic flight: a rendered map, frames at known centres (scale 0.8)
    the_map = make_map()
    path = [(330.0 + 40 * k, 300.0 + 25 * k, 10.0 + 4 * k) for k in range(3)]
    frames = [render_frame(the_map, make_h(h, 0.8, cx=x, cy=y, frame_size=FRAME), FRAME) for x, y, h in path]
    s = Searcher(RootSiftMatcher(the_map, cache_dir=tmp_path), the_map)
    near = lambda fix, p: fix is not None and fix.accepted and math.hypot(fix.x - p[0], fix.y - p[1]) < 1.0

    x, y, _ = path[0]
    win = (int(x) - 320, int(y) - 260, int(x) + 320, int(y) + 260)
    loc = s.locate_in(0, frames[0], win)
    assert loc.mode is SearchMode.WINDOW and loc.window == win and near(loc.fix, path[0])
    # a window far from the frame: not found there, found on the whole map in the same frame
    h, w = the_map.shape[:2]
    loc = s.locate_in(1, frames[1], (w - 300, h - 300, w, h))
    assert loc.mode is SearchMode.FALLBACK and loc.window is None and near(loc.fix, path[1])
    loc = s.locate_in(2, frames[2], None)
    assert loc.mode is SearchMode.GLOBAL and loc.window is None and near(loc.fix, path[2])
