"""M5-01 · Kalman filter.

"Done when: unit tests: a straight line is tracked, an outlier is gated, σ grows while coasting and shrinks on a
fix." Plus what the spec's "What" asks for: the time step comes from the frame timestamps (so frames need not be
evenly spaced), each update uses the fix's own covariance (M4-08), and the Mahalanobis gate has a set χ²
threshold. The filter is constant velocity on (x, y, vx, vy) in map px and seconds.

All data is synthetic with a fixed seed: a drone flying a known straight line, fixes = truth + Gaussian noise.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

from dronenav.confidence import fix_covariance
from dronenav.tracking.kalman import KalmanConfig, KalmanFilter, StepKind
from dronenav.types import Fix, TrackMode

ROOT = Path(__file__).resolve().parents[1]
SMOOTH = KalmanConfig(sigma_acc=0.25)    # for synthetic flights that are exactly straight
START, VEL = np.array([500.0, 800.0]), np.array([30.0, -12.0])   # px, px/s: a plausible slow flight


def truth(t: float) -> np.ndarray:
    return START + VEL * t


def _fix(xy, inliers=500, ratio=0.8, reproj=1.0, accepted=True, idx=0) -> Fix:
    x, y = float(xy[0]), float(xy[1])
    return Fix(idx=idx, H=np.eye(3), x=x, y=y, heading_deg=0.0, scale=0.5,
               footprint=np.array([[x - 50, y - 50], [x + 50, y - 50], [x + 50, y + 50], [x - 50, y + 50]]),
               inliers=inliers, inlier_ratio=ratio, reproj_px=reproj, accepted=accepted,
               reject_reason=None if accepted else "too_few_inliers")


def _noisy_line(n=60, dt=0.5, sigma=2.0, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.arange(n) * dt
    return ts, [truth(t) + rng.normal(0, sigma, 2) for t in ts]


# --- Done when: a straight line is tracked -------------------------------------------------------------------------

def test_a_straight_line_is_tracked():
    ts, zs = _noisy_line()
    # an exactly straight synthetic flight, so a near-zero process noise. The default (80) is set for the real
    # flights, whose image centre sways by tens of px per second; with it the filter trusts these fixes almost fully
    kf, R = KalmanFilter(SMOOTH), 2.0 ** 2 * np.eye(2)
    est = []
    for t, z in zip(ts, zs, strict=True):
        r = kf.step(t, z, R)
        assert r.kind in (StepKind.INIT, StepKind.HIT)          # no good fix is gated on a clean line
        est.append(r.x.copy())
    est = np.array(est)
    tail = slice(20, None)                                       # after the velocity has settled
    err_f = np.linalg.norm(est[tail, :2] - [truth(t) for t in ts[tail]], axis=1)
    err_z = np.linalg.norm(np.array(zs)[tail] - [truth(t) for t in ts[tail]], axis=1)
    # the filter averages the jitter: clearly below the raw fixes' error (2 px σ per axis)
    assert np.sqrt(np.mean(err_f ** 2)) < 0.6 * np.sqrt(np.mean(err_z ** 2))
    # and it learns the speed, from position alone (1.5 px/s: a 5 % of |v| tolerance)
    assert np.linalg.norm(est[-1, 2:] - VEL) < 1.5


def test_the_time_step_comes_from_the_timestamps():
    # uneven frame spacing, exact fixes: the prediction lands on the line at any time
    ts = [0.0, 0.3, 1.0, 1.1, 2.5, 2.6, 4.0]
    kf = KalmanFilter()
    for t in ts:
        kf.step(t, truth(t), 0.01 * np.eye(2))
    x, _ = kf.predict_to(7.3)
    assert np.allclose(x[:2], truth(7.3), atol=0.5)               # 0.5 px after 3.3 s of coasting at ~32 px/s
    assert np.allclose(x[2:], VEL, atol=0.2)


def test_time_must_not_go_back():
    kf = KalmanFilter()
    kf.step(1.0, truth(1.0), np.eye(2))
    with pytest.raises(ValueError):
        kf.step(0.5, truth(0.5), np.eye(2))


def test_a_zero_time_step_changes_nothing_but_the_update():
    kf = KalmanFilter()
    kf.step(0.0, truth(0.0), np.eye(2))
    p0 = kf.P.copy()
    r = kf.step(0.0, None)
    assert r.kind is StepKind.COAST and np.allclose(kf.P, p0)


# --- Done when: an outlier is gated --------------------------------------------------------------------------------

def test_an_outlier_is_gated_and_the_track_goes_on():
    ts, zs = _noisy_line(n=30)
    kf, R = KalmanFilter(), 2.0 ** 2 * np.eye(2)
    for t, z in zip(ts, zs, strict=True):
        kf.step(t, z, R)
    t = ts[-1] + 0.5
    pred, _ = kf.predict_to(t)
    r = kf.step(t, truth(t) + [120.0, -90.0], R)                # a wrong match: 150 px away
    assert r.kind is StepKind.GATED
    assert r.d2 > kf.cfg.gate_chi2
    assert np.allclose(r.x, pred)                                 # the state is the prediction, untouched
    t2 = t + 0.5
    assert kf.step(t2, truth(t2), R).kind is StepKind.HIT        # the next good fix is taken again


def test_the_gate_is_a_chi2_threshold_on_the_mahalanobis_distance():
    kf = KalmanFilter(KalmanConfig(gate_chi2=9.21))              # 99 % for 2 degrees of freedom
    kf.step(0.0, [0.0, 0.0], np.eye(2))
    S = kf.P[:2, :2] + np.eye(2)                                   # innovation covariance for R = I at dt = 0
    s = math.sqrt(S[0, 0])
    assert kf.gate_d2([2.9 * s, 0.0], np.eye(2)) == pytest.approx(2.9 ** 2, rel=1e-6)
    inside = KalmanFilter(KalmanConfig(gate_chi2=9.21))
    inside.step(0.0, [0.0, 0.0], np.eye(2))
    assert inside.step(0.0, [2.9 * s, 0.0], np.eye(2)).kind is StepKind.HIT     # d² = 8.41 < 9.21
    outside = KalmanFilter(KalmanConfig(gate_chi2=9.21))
    outside.step(0.0, [0.0, 0.0], np.eye(2))
    assert outside.step(0.0, [3.1 * s, 0.0], np.eye(2)).kind is StepKind.GATED  # d² = 9.61 > 9.21


# --- Done when: σ grows while coasting and shrinks on a fix -------------------------------------------------------

def test_sigma_grows_while_coasting_and_shrinks_on_a_fix():
    ts, zs = _noisy_line(n=20)
    kf, R = KalmanFilter(), 2.0 ** 2 * np.eye(2)
    for t, z in zip(ts, zs, strict=True):
        kf.step(t, z, R)
    sig = [kf.sigma]
    t = ts[-1]
    for _ in range(8):                                            # 8 frames with no fix
        t += 0.5
        r = kf.step(t, None)
        assert r.kind is StepKind.COAST
        sig.append(kf.sigma)
    assert all(b > a for a, b in itertools.pairwise(sig))  # strictly growing
    before = kf.sigma
    t += 0.5
    kf.step(t, truth(t), R)
    assert kf.sigma < 0.5 * before                                # one fix pulls it back down


def test_a_rejected_fix_is_coasted_through():
    kf = KalmanFilter()
    kf.step(0.0, _fix(truth(0.0)))
    r = kf.step(0.5, _fix(truth(0.5) + 300, accepted=False))
    assert r.kind is StepKind.COAST


def test_a_weak_fix_moves_the_state_less_than_a_strong_one():
    # the update uses the fix's own covariance (M4-08): the same 0.5 px offset pulls less when the fix is weak.
    # With a confident prediction (smooth setting); with the wide default both fixes are trusted almost fully
    pulls = []
    for f in ({"inliers": 2000, "ratio": 0.9, "reproj": 0.5}, {"inliers": 20, "ratio": 0.15, "reproj": 2.8}):
        kf = KalmanFilter(SMOOTH)
        for t in (0.0, 0.5, 1.0, 1.5):
            kf.step(t, _fix(truth(t)))
        pred, _ = kf.predict_to(2.0)
        z = _fix(truth(2.0) + [0.5, 0.0], **f)       # small enough to pass the gate for both
        r = kf.step(2.0, z)
        assert r.kind is StepKind.HIT
        pulls.append(r.x[0] - pred[0])
    assert pulls[0] > 2 * pulls[1] > 0
    assert np.allclose(fix_covariance(_fix([0, 0])), fix_covariance(_fix([0, 0])))  # helper is deterministic


# --- the rest of the contract --------------------------------------------------------------------------------------

def test_the_first_fix_starts_the_filter():
    kf = KalmanFilter()
    assert not kf.initialized
    assert kf.step(0.0, None).kind is StepKind.NONE               # nothing to coast on yet
    r = kf.step(0.5, _fix([100.0, 200.0]))
    assert r.kind is StepKind.INIT and kf.initialized
    assert np.allclose(r.x, [100, 200, 0, 0])
    assert kf.sigma == pytest.approx(math.sqrt(fix_covariance(_fix([0, 0]))[0, 0]))
    with pytest.raises(RuntimeError):
        KalmanFilter().predict_to(1.0)


def test_the_covariance_stays_a_valid_covariance():
    ts, zs = _noisy_line(n=200, dt=0.1, sigma=1.0, seed=3)
    kf = KalmanFilter()
    for k, (t, z) in enumerate(zip(ts, zs, strict=True)):
        kf.step(t, None if k % 7 == 3 else z, np.eye(2))
        assert np.allclose(kf.P, kf.P.T) and np.all(np.linalg.eigvalsh(kf.P) > 0)


def test_each_step_keeps_what_a_smoother_needs():
    kf = KalmanFilter()
    kf.step(0.0, truth(0.0), np.eye(2))
    r = kf.step(0.5, truth(0.5), np.eye(2))
    assert r.x_pred.shape == (4,) and r.P_pred.shape == (4, 4) and r.F.shape == (4, 4)
    assert np.allclose(r.F @ np.array([1, 2, 3, 4.0]), [1 + 1.5, 2 + 2, 3, 4])   # constant velocity over dt=0.5
    assert r.P.shape == (4, 4) and r.t == 0.5


def test_track_state_for_the_telemetry():
    kf = KalmanFilter()
    kf.step(0.0, truth(0.0), np.eye(2))
    s = kf.track_state(idx=7, mode=TrackMode.TRACKING)
    assert s.idx == 7 and s.state is TrackMode.TRACKING
    assert (s.x, s.y) == pytest.approx(tuple(truth(0.0)))
    assert s.cov.shape == (2, 2)


def test_reset_restarts_from_a_new_fix():
    kf = KalmanFilter()
    kf.step(0.0, truth(0.0), np.eye(2))
    kf.step(1.0, truth(1.0), np.eye(2))
    kf.reset(2.0, [3000.0, 100.0], 4.0 * np.eye(2))              # relocalised far away
    assert np.allclose(kf.x, [3000, 100, 0, 0]) and kf.sigma == pytest.approx(2.0)


@pytest.mark.parametrize("bad", [{"sigma_acc": -1}, {"init_speed_sigma": 0}, {"gate_chi2": 0}])
def test_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        KalmanConfig(**bad)


# --- found on the Irpin flights: the default must neither ring nor lock out -------------------------------------

def test_precise_fixes_with_the_default_setting_do_not_make_the_velocity_ring():
    # once per second, 0.5 px fixes, the wide default: the velocity is then close to the difference of the last two
    # fixes, whose error is √2·0.5/1 s per axis = 1.0 px/s in 2D. The prototype's piecewise-constant noise model
    # amplifies it instead (β -> 2: 3.9 px/s here, the sign flipping every frame); the continuous one gives 0.96
    rng = np.random.default_rng(0)
    kf = KalmanFilter()
    vs = np.array([kf.step(t, truth(t) + rng.normal(0, 0.5, 2), 0.25 * np.eye(2)).x[2:] for t in np.arange(60.0)])
    assert np.sqrt(np.mean(np.sum((vs[10:] - VEL) ** 2, axis=1))) < 2.0


IRPIN_FLIGHTS = ROOT / "data" / "irpin" / "flights"


@pytest.mark.data
@pytest.mark.skipif(not IRPIN_FLIGHTS.exists(), reason="data/ not downloaded")
@pytest.mark.parametrize("flight", ["mild", "harsh", "mild_30mps", "harsh_30mps"])
def test_the_default_setting_takes_every_true_fix_on_the_irpin_flights(flight):
    # fixes at the true image centre once per keyframe (1 s), with a typical accepted-fix σ of 0.5 px: the image
    # centre sways by tens of px per second, and a too-confident filter gates good fixes and never recovers
    f = IRPIN_FLIGHTS / flight / "2025"
    rows = [json.loads(line) for line in (f / "truth.jsonl").read_text().splitlines() if line]
    keys = [k["i"] for k in json.loads((f / "meta.json").read_text())["keyframes"]]
    kf, inside = KalmanFilter(), []
    for i in keys:
        z = np.array(rows[i]["center_map_px"])
        r = kf.step(rows[i]["t"], z, 0.25 * np.eye(2))
        assert r.kind in (StepKind.INIT, StepKind.HIT), f"keyframe {i}: {r.kind.value}, d²={r.d2:.1f}"
        if r.kind is StepKind.HIT:
            d = r.x_pred[:2] - z
            inside.append(d @ np.linalg.solve(r.P_pred[:2, :2], d) < 5.991)     # χ²(2) 95 %
    # the prediction's 95 % circle is honest or cautious (measured: 97 % on harsh, 100 % on mild)
    assert np.mean(inside) >= 0.93
