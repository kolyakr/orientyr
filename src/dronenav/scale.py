"""Frame-to-map scale: shrink a frame to about the map's ground resolution before matching (M1-03).

A drone frame usually shows the ground in finer detail than the satellite map: a task crop has 0.25 map px
per frame px, an Irpin frame about 0.5. Matching works on features both images share, and the map can't
show detail finer than its own pixels, so the extra frame pixels cost time without adding matches.

The rule: shrink by ``factor = scale / target_scale``, clipped to ``[min_factor, 1]``. Then one shrunk frame
pixel covers ``target_scale`` map pixels. ``target_scale = 0.65`` keeps the frame a little sharper than the map;
the README table shows why. On the task crops the factor is about 0.39 and matching is ~4x faster with the
same inliers. At a factor of 1/scale or coarser (frame px as big as map px or bigger), inliers start to drop.
Frames are never enlarged: that adds no detail.

``scale`` comes from the last accepted fix (:class:`ScaleTracker`). With none yet (the first frame, or after
the track was lost), a few probe factors are tried and the fix with the most inliers wins.

Matches from a shrunk frame are mapped back to the original frame's pixels, so everything downstream (the
homography, the pose, the run log, the next scale estimate) sees the frame as it came in.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np

from dronenav.geometry import GeometryConfig, localize
from dronenav.types import Fix, Matcher, Matches

_GEOM = GeometryConfig()   # the default checks, shared (it is frozen)


@dataclass(frozen=True)
class ScaleConfig:
    """target_scale: map px per shrunk frame px to aim for. min_factor: never shrink below this (keeps a
    frame from collapsing to a few pixels on a wild scale estimate). probe_factors: tried in this order when
    there is no scale yet; 0.5 first, since it suits both datasets."""

    target_scale: float = 0.65
    min_factor: float = 0.1
    probe_factors: tuple[float, ...] = (0.5, 0.25, 1.0)

    def __post_init__(self):
        if not self.target_scale > 0:
            raise ValueError(f"target_scale must be > 0, got {self.target_scale}")
        if not 0 < self.min_factor <= 1:
            raise ValueError(f"min_factor must be in (0, 1], got {self.min_factor}")
        if not self.probe_factors or not all(0 < f <= 1 for f in self.probe_factors):
            raise ValueError(f"probe_factors must be non-empty, each in (0, 1], got {self.probe_factors}")


def downscale_factor(scale: float, cfg: ScaleConfig = ScaleConfig()) -> float:
    """The shrink factor for a frame with ``scale`` map px per frame px, in ``[cfg.min_factor, 1]``."""
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError(f"scale must be finite and > 0, got {scale}")
    return float(min(1.0, max(cfg.min_factor, scale / cfg.target_scale)))


def resize_frame(img: np.ndarray, factor: float) -> np.ndarray:
    """``img`` shrunk by ``factor`` with area averaging (no aliasing); factor 1 returns ``img`` itself."""
    if not 0 < factor <= 1:
        raise ValueError(f"factor must be in (0, 1], got {factor}")
    if factor == 1.0:
        return img
    h, w = img.shape[:2]
    size = (max(1, round(w * factor)), max(1, round(h * factor)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def match_scaled(matcher: Matcher, frame_img: np.ndarray, map_img: np.ndarray, factor: float,
                 map_offset: tuple = (0, 0), mask: np.ndarray | None = None) -> Matches:
    """Match the frame shrunk by ``factor``; ``pts_frame`` come back in the ORIGINAL frame's pixels.

    ``ms`` covers the resize and the match.
    """
    t0 = time.perf_counter()
    small = resize_frame(frame_img, factor)
    m = matcher.match(small, map_img, map_offset=map_offset, mask=mask)
    if factor != 1.0:
        sy = frame_img.shape[0] / small.shape[0]      # the exact ratios of the rounded size, per axis
        sx = frame_img.shape[1] / small.shape[1]
        # a pixel centre at p in the small image sits at (p + 0.5) * s - 0.5 in the original (cv2's convention)
        pts = (m.pts_frame + 0.5) * np.array([sx, sy]) - 0.5
        m = Matches(pts, m.pts_map, m.scores, m.backend, m.ms)
    return Matches(m.pts_frame, m.pts_map, m.scores, m.backend, (time.perf_counter() - t0) * 1e3)


class ScaledLocalization(NamedTuple):
    """The result of :func:`localize_scaled`: the best fix (or None), the factor that gave it, its matches."""

    fix: Fix | None
    factor: float
    matches: Matches


def _rank(fix: Fix | None) -> tuple[int, int]:
    """Accepted before rejected before none; then more inliers first."""
    if fix is None:
        return (0, 0)
    return (2 if fix.accepted else 1, fix.inliers)


def localize_scaled(idx: int, matcher: Matcher, frame_img: np.ndarray, map_img: np.ndarray,
                    factors: tuple[float, ...], map_offset: tuple = (0, 0), mask: np.ndarray | None = None,
                    geom: GeometryConfig = _GEOM) -> ScaledLocalization:
    """Localize frame ``idx`` at each factor in ``factors`` and keep the best fix.

    One factor (from :class:`ScaleTracker` once a fix exists) costs one match; several (probing) cost one
    match each, so probing is meant for the first frame and after a loss.
    """
    if not factors:
        raise ValueError("factors must not be empty")
    size = (frame_img.shape[1], frame_img.shape[0])
    best: ScaledLocalization | None = None
    for f in factors:
        m = match_scaled(matcher, frame_img, map_img, f, map_offset=map_offset, mask=mask)
        res = ScaledLocalization(localize(idx, m, size, geom), f, m)
        if best is None or _rank(res.fix) > _rank(best.fix):
            best = res
    return best


class ScaleTracker:
    """The frame -> map scale from the last accepted fix, and the factors to try for the next frame."""

    def __init__(self, cfg: ScaleConfig = ScaleConfig()):
        self.cfg = cfg
        self.scale: float | None = None

    def factors(self) -> tuple[float, ...]:
        """One factor from the known scale, or the probe factors when there is none."""
        if self.scale is None:
            return self.cfg.probe_factors
        return (downscale_factor(self.scale, self.cfg),)

    def update(self, fix: Fix | None) -> None:
        """Take the scale of an accepted fix; a rejected or missing one says nothing about the scale."""
        if fix is not None and fix.accepted and math.isfinite(fix.scale) and fix.scale > 0:
            self.scale = float(fix.scale)

    def reset(self) -> None:
        """Forget the scale (the track was lost; the altitude may have changed meanwhile)."""
        self.scale = None
