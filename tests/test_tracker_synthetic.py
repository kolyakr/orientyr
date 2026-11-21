"""M5-06 · Tracker tests on synthetic fixes: dropouts, noise, a wrong fix far away, a slow wrong drift, and an
honest uncertainty.

Done when: the tests run in CI in seconds. Spec update (20 Nov): on generated fixes with known noise, the true
position lies inside the tracker's 95 % circle in 90–98 % of frames.

The world is generated from the filter's own motion model (constant velocity, white random acceleration of
intensity sigma_acc: each step adds a sample of the model's Q), and every fix is the truth plus Gaussian noise of
exactly the σ that the fix's evidence gives (``fix_sigma``). So the filter's model is right by construction, and
the tests check the tracker (states, gate, hold, restart) and the arithmetic, not a tuning. Each scenario is seeded;
the whole file runs in about a second on a CPU.

"The 95 % circle": the region the tracker's 2 x 2 position covariance P holds with 95 % probability, Mahalanobis
d² ≤ 5.991 (χ², 2 dof); for an isotropic P that is the circle of radius 2.45 σ drawn in the viz (M4-08).
"""
import math

import numpy as np
import pytest

from dronenav.confidence import ConfidenceConfig, fix_sigma
from dronenav.tracking.kalman import KalmanConfig, StepKind, _transition
from dronenav.tracking.states import Tracker, TrackerConfig
from dronenav.types import Fix, TrackMode

T, L, R = TrackMode.TRACKING, TrackMode.LOST, TrackMode.RELOCALIZING
CHI2_95 = 5.991                            # χ², 2 dof, 95 %
GENTLE = KalmanConfig(sigma_acc=5.0, init_speed_sigma=20.0)    # a smooth drone: the filter can average fixes
_CC = ConfidenceConfig()


def _fix(idx: int, xy, sigma: float, inliers: int = 100) -> Fix:
    """An accepted fix at ``xy`` whose evidence says σ = ``sigma`` px (inlier ratio 1, so σ² = floor² + k² e²/n:
    the reprojection error e is solved for)."""
    e = math.sqrt(max(sigma ** 2 - _CC.sigma_floor_px ** 2, 0.0) * inliers) / _CC.k_reproj
    x, y = float(xy[0]), float(xy[1])
    fp = np.array([(x - 160, y - 120), (x + 160, y - 120), (x + 160, y + 120), (x - 160, y + 120)])
    return Fix(idx=idx, H=np.eye(3), x=x, y=y, heading_deg=0.0, scale=1.0, footprint=fp, inliers=inliers,
               inlier_ratio=1.0, reproj_px=e, accepted=True)


def _truth(seed: int, n: int, cfg: KalmanConfig, dt: float = 1.0) -> np.ndarray:
    """n true states (x, y, vx, vy), one per dt, drawn from the filter's own motion model."""
    rng = np.random.default_rng(seed)
    F, Q = _transition(dt, cfg.sigma_acc)
    s = np.array([5000.0, 5000.0, *rng.normal(0.0, cfg.init_speed_sigma, 2)])
    out = []
    for _ in range(n):
        out.append(s.copy())
        s = F @ s + rng.multivariate_normal(np.zeros(4), Q)
    return np.array(out)


def _fly(truth, fixes, cfg: KalmanConfig, lost_after: int = 3):
    """Run the tracker over ``fixes`` (one per frame: an (x, y) with its σ, a ready Fix, or None);
    returns the updates."""
    track = Tracker(TrackerConfig(lost_after=lost_after, kalman=cfg))
    ups = []
    for k, z in enumerate(fixes):
        if isinstance(z, tuple):
            z = _fix(k, z[0], z[1])
        ups.append(track.update(k, float(k), z))
    return ups


def _err(up, truth_k) -> float:
    return math.hypot(up.track.x - truth_k[0], up.track.y - truth_k[1])


def _d2(up, truth_k) -> float:
    d = np.array([truth_k[0] - up.track.x, truth_k[1] - up.track.y])
    return float(d @ np.linalg.solve(up.track.cov, d))


def test_the_fix_helper_gives_the_sigma_asked_for():
    for s in (0.5, 3.0, 5.0, 20.0):
        assert fix_sigma(_fix(0, (0, 0), s)) == pytest.approx(s, rel=1e-9)


# --- noise: the filter averages it out ------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(3))
def test_noisy_fixes_give_a_track_closer_to_the_truth_than_the_fixes(seed):
    n, sigma = 300, 5.0
    truth = _truth(seed, n, GENTLE)
    rng = np.random.default_rng(seed + 100)
    zs = truth[:, :2] + rng.normal(0.0, sigma, (n, 2))
    ups = _fly(truth, [(z, sigma) for z in zs], GENTLE)
    # the gate is 99.9 %: a true fix is refused only on a rare burst of acceleration (0-3 of 300 here)
    assert sum(u.step.kind is StepKind.GATED for u in ups) <= 0.01 * n
    assert sum(u.mode is T for u in ups) >= 0.97 * n
    # averaging fixes beats any single one: at this setting the filter's steady σ is ~0.88 of the fix σ. The
    # median, so that one honest burst (a few coasted frames) doesn't decide it
    fix_err = np.hypot(*(zs - truth[:, :2]).T)
    trk_err = np.array([_err(u, t) for u, t in zip(ups, truth, strict=True)])
    assert np.median(trk_err) < 0.95 * np.median(fix_err), (np.median(trk_err), np.median(fix_err))


# --- dropouts: coasted through, never lost while shorter than lost_after -------------------------------------

def test_dropouts_shorter_than_lost_after_are_coasted_through():
    n, sigma = 400, 5.0
    truth = _truth(7, n, GENTLE)
    rng = np.random.default_rng(107)
    drop = rng.random(n) < 0.3
    drop[0] = False
    for k in range(1, n):                     # at most 2 misses in a row: below lost_after (3)
        if drop[k] and drop[k - 1] and k >= 2 and drop[k - 2]:
            drop[k] = False
    zs = truth[:, :2] + rng.normal(0.0, sigma, (n, 2))
    ups = _fly(truth, [None if d else (z, sigma) for d, z in zip(drop, zs, strict=True)], GENTLE)
    assert drop.sum() > 80
    assert all(u.mode is not L for u in ups), "a short dropout must not lose the track"
    assert all(u.step.kind is StepKind.COAST for u, d in zip(ups, drop, strict=True) if d)
    coasted = [_d2(u, t) for u, t, d in zip(ups, truth, drop, strict=True) if d]
    # the coasted positions are as honest as the rest: their 95 % circles hold the truth ~95 % of the time
    assert 0.88 <= np.mean(np.array(coasted) <= CHI2_95) <= 0.99


def test_a_long_dropout_is_lost_then_relocalized_on_the_truth():
    n, sigma = 40, 3.0
    truth = _truth(3, n, GENTLE)
    rng = np.random.default_rng(103)
    zs = truth[:, :2] + rng.normal(0.0, sigma, (n, 2))
    fixes = [None if 15 <= k < 25 else (z, sigma) for k, z in enumerate(zs)]
    ups = _fly(truth, fixes, GENTLE)
    modes = [u.mode for u in ups]
    assert modes[15:17] == [T, T] and modes[17:25] == [L] * 8                 # LOST at the third miss
    assert modes[25] is R and ups[25].step.kind is StepKind.INIT               # restarted at the fix, not blended
    # confirmed within two more fixes: the drone flies ~100 px/s, faster than the 20 px/s the restart assumes, so
    # the next fix may disagree; that restart then allows the speed the two fixes imply, and the third agrees
    assert modes[27:] == [T] * (n - 27)
    assert all(_err(u, t) < 4 * sigma for u, t in zip(ups[25:], truth[25:], strict=True))


# --- a wrong fix far away: gated, not followed -----------------------------------------------------------------

def test_far_wrong_fixes_are_gated_and_not_followed():
    n, sigma = 300, 3.0
    truth = _truth(11, n, GENTLE)
    rng = np.random.default_rng(111)
    zs = truth[:, :2] + rng.normal(0.0, sigma, (n, 2))
    wrong = np.zeros(n, bool)
    wrong[rng.choice(np.arange(5, n), 25, replace=False)] = True
    for k in range(1, n):                     # isolated look-alikes: never two in a row
        if wrong[k] and wrong[k - 1]:
            wrong[k] = False
    far = rng.uniform(400, 3000, (n, 1)) * np.stack([np.cos(a := rng.uniform(0, 2 * np.pi, n)), np.sin(a)], 1)
    fixes = [(z + o, sigma) if w else (z, sigma) for z, o, w in zip(zs, far, wrong, strict=True)]
    ups = _fly(truth, fixes, GENTLE)
    assert wrong.sum() >= 15
    for k in np.flatnonzero(wrong):
        u = ups[k]
        assert u.step.kind is StepKind.GATED and u.mode is T, k
        assert _err(u, truth[k]) < 6 * sigma, k                               # the position did not jump
    assert sum(u.mode is T for u in ups) >= 0.98 * n
    true_fixes = [ups[k].step.kind for k in range(1, n) if not wrong[k]]
    assert true_fixes.count(StepKind.HIT) >= 0.99 * len(true_fixes)          # the gate is 99.9 %, see above


def test_a_run_of_wrong_fixes_loses_the_track_and_the_truth_wins_it_back():
    # three look-alike fixes in a row at the same wrong place: gated, gated, gated -> LOST; then the true fixes
    # come back and the tracker relocalizes on them
    n, sigma = 40, 3.0
    truth = _truth(5, n, GENTLE)
    rng = np.random.default_rng(105)
    zs = truth[:, :2] + rng.normal(0.0, sigma, (n, 2))
    fixes = [(z, sigma) for z in zs]
    for k in (20, 21, 22):
        fixes[k] = (truth[k, :2] + (1500.0, -900.0), sigma)
    ups = _fly(truth, fixes, GENTLE)
    assert [u.step.kind for u in ups[20:23]] == [StepKind.GATED] * 3
    assert [u.mode for u in ups[20:25]] == [T, T, L, R, T]
    assert all(_err(u, t) < 4 * sigma for u, t in zip(ups[24:], truth[24:], strict=True))


# --- a slow wrong drift: followed (nothing to tell it from motion), then let go when the truth returns ----------

def test_a_slow_drift_is_followed_and_the_snap_back_relocalizes():
    # the matcher's error creeps up by 1 px per frame for 120 frames (a slowly sliding match), then the fixes are
    # right again. A slow drift looks like motion, so the tracker follows it, but it must not run away from the
    # fixes it is given (no lock-out). When the fixes snap back by 120 px, the gate refuses them until the track is
    # LOST, and the next fix restarts it on the truth.
    n, sigma, ramp = 200, 3.0, (30, 150)
    truth = _truth(9, n, GENTLE)
    rng = np.random.default_rng(109)
    bias = np.clip(np.arange(n) - ramp[0], 0, ramp[1] - ramp[0]).astype(float)
    bias[ramp[1]:] = 0.0
    zs = truth[:, :2] + rng.normal(0.0, sigma, (n, 2)) + np.stack([bias, 0 * bias], 1)
    ups = _fly(truth, [(z, sigma) for z in zs], GENTLE)
    drift = range(ramp[0], ramp[1])
    assert all(ups[k].mode is T and ups[k].step.kind is StepKind.HIT for k in drift)
    for k in drift:                                         # follows the fixes it is given: stays near them
        assert math.hypot(ups[k].track.x - zs[k, 0], ups[k].track.y - zs[k, 1]) < 4 * sigma, k
    after = [u.mode for u in ups[ramp[1]:]]
    assert after[:4] == [T, T, L, R]                    # refused three times (coasting), LOST, restarted at the truth
    assert after[5:] == [T] * (len(after) - 5)          # confirmed within two fixes (as the long dropout)
    back = ramp[1] + 3
    assert all(_err(u, t) < 4 * sigma for u, t in zip(ups[back:], truth[back:], strict=True))


# --- honest uncertainty: the 95 % circle holds the truth in 90–98 % of frames ----------------------------------

@pytest.mark.parametrize(("cfg", "sigma"), [(GENTLE, 5.0), (KalmanConfig(), 3.0)], ids=["gentle", "default"])
def test_the_95_percent_circle_holds_the_truth_in_90_to_98_percent_of_frames(cfg, sigma):
    # 2000 frames with 20 % dropouts (at most two in a row): with the correlation between consecutive frames,
    # ±3 points around 95 % is far wider than the seeded spread. A rare refused true fix (the 99.9 % gate) next to
    # two dropouts makes three misses, so the track may be lost a handful of times; it must come back each time
    n = 2000
    truth = _truth(21, n, cfg)
    rng = np.random.default_rng(121)
    drop = rng.random(n) < 0.2
    drop[0] = False
    for k in range(2, n):
        if drop[k] and drop[k - 1] and drop[k - 2]:
            drop[k] = False
    zs = truth[:, :2] + rng.normal(0.0, sigma, (n, 2))
    ups = _fly(truth, [None if d else (z, sigma) for d, z in zip(drop, zs, strict=True)], cfg)
    assert sum(u.mode is T for u in ups) >= 0.99 * n and ups[-1].mode is T
    inside = np.array([_d2(u, t) <= CHI2_95 for u, t in zip(ups, truth, strict=True)])
    assert 0.90 <= inside.mean() <= 0.98, inside.mean()
    # and the coasted frames alone, where the uncertainty is the filter's prediction, not the fix's
    assert 0.90 <= inside[drop].mean() <= 0.98, inside[drop].mean()
