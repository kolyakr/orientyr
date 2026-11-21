"""M5-05 · Offline smoother and gap interpolation (Rauch–Tung–Striebel over a finished run).

Done when: the smoothed track has lower error than the filtered one on simulated flights.

Simulated flights: a seeded smooth route (random acceleration), noisy fixes (σ 3 px) with random dropouts and one
longer gap, run through the M5-01 filter; the smoother runs over its steps. Ground truth is exact, so the errors
are exact. The contract beyond that: one output per step, each marked fix / interpolated / extrapolated (None
before the filter starts), a restart (INIT) is a boundary the smoothing doesn't cross, the last point of a
segment is the filtered one, and the smoothed uncertainty is never larger than the filtered one.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from dronenav.tracking.kalman import KalmanConfig, KalmanFilter, StepKind
from dronenav.tracking.smoother import Smoothed, smooth

ROOT = Path(__file__).resolve().parents[1]
data = pytest.mark.data
NOISE = 3.0                                  # px, 1σ of each simulated fix
CFG = KalmanConfig(sigma_acc=8.0)            # a smooth simulated drone: px/s^1.5, see _route


def _route(seed, n=120, dt=1.0):
    """A smooth route: velocity driven by a random acceleration (σ 4 px/s², a gentle turning drone)."""
    rng = np.random.default_rng(seed)
    pos, vel = np.array([2000.0, 1500.0]), rng.normal(0, 20, 2)
    out = []
    for _ in range(n):
        out.append(pos.copy())
        vel = vel + rng.normal(0, 4.0, 2) * dt
        pos = pos + vel * dt
    return np.array(out)


def _flight(seed, n=120, drop=0.2, gap=(60, 68)):
    """Truth, the filter's steps and which frames had a fix."""
    rng = np.random.default_rng(seed + 1000)
    truth = _route(seed, n)
    kf = KalmanFilter(CFG)
    steps, has = [], []
    for i, p in enumerate(truth):
        miss = (i > 0 and rng.random() < drop) or gap[0] <= i < gap[1]
        z = None if miss else p + rng.normal(0, NOISE, 2)
        steps.append(kf.step(float(i), z, None if z is None else np.eye(2) * NOISE ** 2))
        has.append(not miss)
    return truth, steps, np.array(has)


def _err(points, truth):
    return np.hypot(*(np.array(points) - truth).T)


# --- Done when: the smoothed track has lower error than the filtered one on simulated flights -------------------

@pytest.mark.parametrize("seed", range(5))
def test_the_smoothed_track_is_more_accurate_than_the_filtered_one(seed):
    truth, steps, has = _flight(seed)
    filt = _err([s.x[:2] for s in steps], truth)
    smo = _err([s.x[:2] for s in smooth(steps, CFG)], truth)
    rms = lambda e: float(np.sqrt(np.mean(e ** 2)))
    # overall: using the future as well as the past is worth well over 10 %
    assert rms(smo) < 0.85 * rms(filt), (rms(smo), rms(filt))
    # in the gaps: interpolating between fixes beats flying on the last velocity by far
    assert rms(smo[~has]) < 0.6 * rms(filt[~has]), (rms(smo[~has]), rms(filt[~has]))


# --- the contract ----------------------------------------------------------------------------------------------

def test_each_point_is_marked_fix_interpolated_or_extrapolated():
    kf = KalmanFilter(CFG)
    R = np.eye(2)
    script = [None, (0, 0), (10, 0), None, None, (40, 0), None, None]
    steps = [kf.step(float(i), z, None if z is None else R) for i, z in enumerate(script)]
    out = smooth(steps, CFG)
    assert len(out) == len(steps) and out[0] is None                       # before the filter starts: nothing
    assert [o.source for o in out[1:]] == ["fix", "fix", "interpolated", "interpolated", "fix",
                                          "extrapolated", "extrapolated"]
    assert all(isinstance(o, Smoothed) for o in out[1:])
    assert [o.t for o in out[1:]] == [1, 2, 3, 4, 5, 6, 7]


def test_a_gated_fix_is_a_gap_too():
    kf = KalmanFilter(CFG)
    R = np.eye(2)
    steps = [kf.step(float(i), (10.0 * i, 0.0), R) for i in range(4)]
    steps.append(kf.step(4.0, (5000.0, 0.0), R))                           # far off: gated
    steps.append(kf.step(5.0, (50.0, 0.0), R))
    assert steps[4].kind is StepKind.GATED
    out = smooth(steps, CFG)
    assert out[4].source == "interpolated" and abs(out[4].x[0] - 40.0) < 2.0


def test_the_gap_is_bridged_between_the_fixes_on_both_sides():
    # a straight flight at 10 px/s with a 4-frame gap: the smoothed gap is on the line, not on a stale guess
    kf = KalmanFilter(CFG)
    R = np.eye(2) * 0.01
    steps = [kf.step(float(i), None if 5 <= i < 9 else (10.0 * i, 3.0 * i), R) for i in range(14)]
    out = smooth(steps, CFG)
    for i in range(5, 9):
        assert out[i].x[0] == pytest.approx(10.0 * i, abs=0.5) and out[i].x[1] == pytest.approx(3.0 * i, abs=0.5)


def test_the_last_point_is_the_filtered_one_and_uncertainty_only_shrinks():
    _, steps, _ = _flight(7, n=40)
    out = smooth(steps, CFG)
    np.testing.assert_allclose(out[-1].x, steps[-1].x)
    np.testing.assert_allclose(out[-1].P, steps[-1].P)
    for o, s in zip(out, steps, strict=True):
        assert np.trace(o.P[:2, :2]) <= np.trace(s.P[:2, :2]) + 1e-9
        assert np.allclose(o.P, o.P.T) and np.all(np.linalg.eigvalsh(o.P) > 0)


def test_a_restart_is_a_boundary():
    # the filter restarted far away (INIT after LOST): the first segment must not be pulled towards the second
    kf = KalmanFilter(CFG)
    R = np.eye(2)
    first = [kf.step(float(i), (10.0 * i, 0.0), R) for i in range(5)]
    kf.reset(5.0, (9000.0, 9000.0), R)
    restart = kf.step(5.0, None)._replace(kind=StepKind.INIT)
    rest = [kf.step(float(i), (9000.0 + 10 * i, 9000.0), R) for i in range(6, 10)]
    out = smooth([*first, restart, *rest], CFG)
    alone = smooth(first, CFG)
    for a, b in zip(out[:5], alone, strict=True):
        np.testing.assert_allclose(a.x, b.x)
    assert out[5].source == "fix"


def test_an_empty_run_and_a_run_that_never_starts():
    assert smooth([], CFG) == []
    kf = KalmanFilter(CFG)
    assert smooth([kf.step(0.0), kf.step(1.0)], CFG) == [None, None]


# --- the Irpin flights: real motion, simulated fix noise --------------------------------------------------------

@data
@pytest.mark.parametrize("flight", ["mild", "harsh"])
def test_smoothing_helps_on_the_irpin_motion(flight):
    # the true keyframe centres of a rendered flight (real turns and camera sway), with 3 px fix noise and drops
    path = ROOT / "data" / "irpin" / "flights" / flight / "2025" / "truth.jsonl"
    if not path.is_file():
        pytest.skip("Irpin data not pulled")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["i"] % 30 == 0]
    truth = np.array([r["center_map_px"] for r in rows], float)
    t = np.array([r["t"] for r in rows], float)
    rng = np.random.default_rng(11)
    kf, steps, has = KalmanFilter(), [], []        # the default (Irpin-tuned) setting
    for k, p in enumerate(truth):
        miss = k > 0 and rng.random() < 0.2
        z = None if miss else p + rng.normal(0, NOISE, 2)
        steps.append(kf.step(float(t[k]), z, None if z is None else np.eye(2) * NOISE ** 2))
        has.append(not miss)
    has = np.array(has)
    filt = _err([s.x[:2] for s in steps], truth)
    smo = _err([s.x[:2] for s in smooth(steps, KalmanConfig())], truth)
    assert np.sqrt(np.mean(smo ** 2)) < np.sqrt(np.mean(filt ** 2))
    assert np.sqrt(np.mean(smo[~has] ** 2)) < np.sqrt(np.mean(filt[~has] ** 2))
