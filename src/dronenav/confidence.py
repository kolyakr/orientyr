"""Per-fix confidence and position covariance (M4-08).

Not every accepted fix is equally good. A fix from 2000 inliers at 0.9 inlier ratio and 1 px reprojection error is
almost always within half a pixel of the truth; one from 25 inliers at 0.3 ratio may be off by several pixels. The
tracker needs to know which is which, in two forms:

- **confidence**, one number in [0, 1] for people and plots (trail colour, HUD, reliability plot). It multiplies
  three saturating terms, one per kind of evidence, each 1 at its best and 0 at its worst:

      inliers:   n / (n + inliers_half)                    more agreeing points, more support
      ratio:     r (1 + ratio_half) / (r + ratio_half)     few outliers among the matches: the match is not luck
      reproj:    h² / (h² + e²), h = reproj_half_px        the points agree tightly with one homography

- **position covariance**, the filter's measurement noise R, in map px². It is isotropic, σ² I, with

      σ² = sigma_floor² + k_reproj² · e² / n + k_ratio² · ((1 − r) / r)²

  The middle term is the standard error of a mean of n residuals of size e; the last one grows as the inliers drown
  in outliers, the case where a homography can be wrong as a whole while fitting its inliers well.

Calibration (``scripts/confidence_check.py``): 430 accepted fixes of the four Irpin flights against the 2025 and
2022 maps, with exact truth. Within each map, inliers and ratio correlate with the error more than the reprojection
error does, and none is enough alone. The σ constants are a maximum-likelihood fit (2D Rayleigh) of the error,
rounded: the 95 % circle (2.45 σ) then holds the truth for 95 % of the fixes on both maps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dronenav.types import Fix

__all__ = ["ConfidenceConfig", "fix_confidence", "fix_covariance", "fix_sigma"]

_MIN_RATIO = 1e-3           # a ratio of 0 can't happen for an accepted fix; keep the formula finite anyway


@dataclass(frozen=True)
class ConfidenceConfig:
    """Half-saturation points of the confidence terms, and the σ model of the covariance (map px).

    inliers_half: inliers giving the inlier term 0.5. ratio_half: sets how fast the ratio term rises.
    reproj_half_px: reprojection RMSE giving the reprojection term 0.5 (the geometry gate is 3 px).
    sigma_floor_px, k_reproj, k_ratio: σ² = floor² + k_reproj² e²/n + k_ratio² ((1-r)/r)²."""

    inliers_half: float = 100.0
    ratio_half: float = 0.5
    reproj_half_px: float = 3.0
    sigma_floor_px: float = 0.2
    k_reproj: float = 3.0
    k_ratio: float = 0.75

    def __post_init__(self):
        for name in ("inliers_half", "ratio_half", "reproj_half_px", "sigma_floor_px"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        for name in ("k_reproj", "k_ratio"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")


_CFG = ConfidenceConfig()


def _ratio(fix: Fix) -> float:
    return min(1.0, max(_MIN_RATIO, float(fix.inlier_ratio)))


def fix_confidence(fix: Fix | None, cfg: ConfidenceConfig = _CFG) -> float:
    """How much to trust this fix, in [0, 1]. No fix or a rejected one: 0."""
    if fix is None or not fix.accepted:
        return 0.0
    n, r, e = max(0.0, float(fix.inliers)), _ratio(fix), max(0.0, float(fix.reproj_px))
    h2 = cfg.reproj_half_px ** 2
    c = (n / (n + cfg.inliers_half)) * (r * (1 + cfg.ratio_half) / (r + cfg.ratio_half)) * (h2 / (h2 + e * e))
    return min(1.0, max(0.0, c))


def fix_sigma(fix: Fix, cfg: ConfidenceConfig = _CFG) -> float:
    """1-σ position error of the fix along each axis, map px (see the module docstring for the model)."""
    if fix is None:
        raise ValueError("a covariance needs a fix")
    n, r, e = max(1.0, float(fix.inliers)), _ratio(fix), max(0.0, float(fix.reproj_px))
    var = cfg.sigma_floor_px ** 2 + cfg.k_reproj ** 2 * e * e / n + cfg.k_ratio ** 2 * ((1 - r) / r) ** 2
    return math.sqrt(var)


def fix_covariance(fix: Fix, cfg: ConfidenceConfig = _CFG) -> np.ndarray:
    """The fix's position covariance in map px² (2 x 2, isotropic): the measurement noise R for the filter."""
    return fix_sigma(fix, cfg) ** 2 * np.eye(2)
