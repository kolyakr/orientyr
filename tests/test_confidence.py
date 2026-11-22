"""M4-08 · Per-fix confidence.

"Done when: on simulated flights, low-confidence fixes have clearly larger errors (a reliability plot)." So:
- the confidence is one number in [0, 1] per fix, from inliers, inlier ratio and reprojection error: more inliers
  and a higher ratio raise it, a larger reprojection error lowers it; a rejected fix (or none) has confidence 0;
- the position covariance (the filter's measurement noise) is a valid 2 x 2 covariance whose σ shrinks with more
  inliers and a higher ratio, grows with the reprojection error, and never goes below the floor;
- on the Irpin flights with exact truth (``@pytest.mark.data``): the least confident third of the fixes has a
  clearly larger error than the most confident third, and the covariance is honest (the truth falls inside
  the 95 % circle for about 95 % of the fixes).
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.confidence import ConfidenceConfig, fix_confidence, fix_covariance, fix_sigma
from dronenav.types import Fix

ROOT = Path(__file__).resolve().parents[1]


def _fix(inliers=500, ratio=0.8, reproj=1.5, accepted=True) -> Fix:
    return Fix(idx=0, H=np.eye(3), x=100.0, y=100.0, heading_deg=0.0, scale=0.5,
               footprint=np.array([[0, 0], [200, 0], [200, 200], [0, 200]], float), inliers=inliers,
               inlier_ratio=ratio, reproj_px=reproj, accepted=accepted,
               reject_reason=None if accepted else "too_few_inliers")


# --- confidence ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [{}, {"inliers": 15, "ratio": 0.05, "reproj": 3.0}, {"inliers": 5000, "ratio": 1.0,
                                                                                     "reproj": 0.0}])
def test_confidence_is_in_0_1(kw):
    assert 0.0 <= fix_confidence(_fix(**kw)) <= 1.0


@pytest.mark.parametrize(("field", "low", "high"), [("inliers", 30, 1500), ("ratio", 0.2, 0.9), ("reproj", 2.8, 0.6)])
def test_confidence_rises_with_evidence(field, low, high):
    assert fix_confidence(_fix(**{field: low})) < fix_confidence(_fix(**{field: high}))


def test_a_strong_fix_is_confident_and_a_weak_one_is_not():
    assert fix_confidence(_fix(inliers=2000, ratio=0.9, reproj=1.0)) > 0.8
    assert fix_confidence(_fix(inliers=20, ratio=0.25, reproj=2.5)) < 0.15


def test_no_fix_or_a_rejected_fix_has_zero_confidence():
    assert fix_confidence(None) == 0.0
    assert fix_confidence(_fix(accepted=False)) == 0.0


# --- covariance ---------------------------------------------------------------------------------------------------

def test_covariance_is_a_valid_2x2_covariance():
    c = fix_covariance(_fix())
    assert c.shape == (2, 2) and np.allclose(c, c.T) and np.all(np.linalg.eigvalsh(c) > 0)
    assert math.sqrt(np.linalg.eigvalsh(c).max()) == pytest.approx(fix_sigma(_fix()))


@pytest.mark.parametrize(("field", "worse", "better"), [("inliers", 30, 1500), ("ratio", 0.2, 0.9),
                                                        ("reproj", 2.8, 0.6)])
def test_sigma_is_larger_for_weaker_evidence(field, worse, better):
    assert fix_sigma(_fix(**{field: worse})) > fix_sigma(_fix(**{field: better}))


def test_sigma_never_goes_below_the_floor():
    cfg = ConfidenceConfig()
    assert fix_sigma(_fix(inliers=10 ** 6, ratio=1.0, reproj=0.0)) == pytest.approx(cfg.sigma_floor_px)
    assert fix_sigma(_fix(inliers=10 ** 6, ratio=1.0, reproj=0.0)) >= cfg.sigma_floor_px


def test_sigma_of_a_zero_ratio_is_finite():
    # an accepted fix always has inliers, but the formula must not divide by zero on a bad input
    assert math.isfinite(fix_sigma(_fix(ratio=0.0)))


def test_covariance_needs_a_fix():
    with pytest.raises(ValueError):
        fix_covariance(None)


@pytest.mark.parametrize("bad", [{"inliers_half": 0}, {"ratio_half": -0.1}, {"reproj_half_px": 0},
                                 {"sigma_floor_px": 0}, {"k_reproj": -1}, {"k_ratio": -1}])
def test_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        ConfidenceConfig(**bad)


# --- Done when: on the Irpin flights, low confidence = larger error ----------------------------------------------

IRPIN = ROOT / "data" / "irpin"


def _irpin_fixes(tmp_path, flight: str, year: str, n: int):
    """(fix, true centre on that map) for the first n keyframes of a 2025 flight matched against `year`'s map."""
    from dronenav.matching import RootSiftMatcher
    from dronenav.search import Searcher

    side = json.loads((IRPIN / "maps" / f"irpin_{year}.json").read_text())["map"]["alignment_to_2025"]["shift_px"]
    mp = cv2.imread(str(IRPIN / "maps" / f"irpin_{year}.png"))
    f = IRPIN / "flights" / flight / "2025"
    meta = json.loads((f / "meta.json").read_text())
    truth = {r["i"]: r for r in map(json.loads, (f / "truth.jsonl").read_text().splitlines()) if r}
    s = Searcher(RootSiftMatcher(mp, cache_dir=tmp_path), mp)
    out = []
    for k, key in enumerate(meta["keyframes"][:n]):
        fix = s.locate(k, cv2.imread(str(f / key["file"]))).fix
        cx, cy = truth[key["i"]]["center_map_px"]
        # the older maps are aligned to 2025 to within ~1 px; on them the 2025 truth sits at (cx, cy) - shift
        out.append((fix, (cx - side[0], cy - side[1])))
    return out


@pytest.mark.data
@pytest.mark.skipif(not (IRPIN / "maps" / "irpin_2022.json").exists(), reason="data/ not downloaded")
def test_low_confidence_fixes_have_clearly_larger_errors_on_irpin(tmp_path):
    # one flight on its own map (good fixes) and on the 2022 map (cross-season: weak fixes). mild_30mps gives the
    # most accepted fixes on 2022 (49 of 80 keyframes; the others 17-26), so the thirds are not tiny
    pairs = _irpin_fixes(tmp_path, "mild_30mps", "2025", 40) + _irpin_fixes(tmp_path, "mild_30mps", "2022", 80)
    pairs = [(f, t) for f, t in pairs if f is not None and f.accepted]
    conf = np.array([fix_confidence(f) for f, _ in pairs])
    err = np.array([math.hypot(f.x - t[0], f.y - t[1]) for f, t in pairs])
    sigma = np.array([fix_sigma(f) for f, _ in pairs])
    assert len(pairs) >= 60
    order = np.argsort(conf)
    third = len(order) // 3
    low, high = err[order[:third]], err[order[-third:]]
    # "clearly": the low third's median error at least twice the high third's (measured on all four flights
    # x 2025/2022: 1.5 px vs 0.3 px by fifths), and its bad tail too
    assert np.median(low) > 2 * np.median(high)
    assert np.percentile(low, 90) > 2 * np.percentile(high, 90)
    # an honest covariance: the 95 % circle (2.45 σ in 2D) holds the truth for 85-100 % of fixes (95.1 % on the
    # full 430-fix set; the band is wide for this smaller sample)
    assert 0.85 <= np.mean(err < 2.448 * sigma) <= 1.0
