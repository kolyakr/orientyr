"""Geometry: from frame <-> map correspondences to a checked pose (a :class:`~dronenav.types.Fix`).

Three steps, each usable on its own:

1. :func:`localize` fits a frame -> map homography to the matches with MAGSAC++ (robust to wrong matches);
2. :func:`check_homography` decides whether that homography can be a camera looking down at the map,
   and names the first check that fails;
3. :func:`pose_from_homography` reads the pose off it: where the frame centre lands, which way the frame's
   "up" points, how many map pixels one frame pixel covers, and the frame's four corners on the map.

Conventions (map/frame coordinates, heading, scale) are those of :mod:`dronenav.types`.

The checks run in a fixed order and the first failure is reported, so each reason means one thing:

=====================  ==========================================================================================
``degenerate``         H is not finite, is (nearly) zero, sends the frame centre to infinity, or squashes the frame
                       to a line or a point (no area on the map).
``too_few_inliers``    fewer than ``min_inliers`` matches agree with H: too little evidence.
``not_convex``         the footprint is not a convex quadrilateral: the plane's horizon crosses the frame or the
                       corners are twisted. A real camera over a flat map can't produce either.
``mirrored``           the footprint is convex but its corners run the other way round: a reflection.
``scale_out_of_range`` map px per frame px at the centre is outside [``min_scale``, ``max_scale``].
``aspect``             at the centre one direction is stretched more than ``max_aspect`` times the other.
``reproj_error``       the inliers' RMS reprojection error is above ``max_reproj_rmse_px``: a sloppy fit.
=====================  ==========================================================================================
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np

from dronenav.types import Fix, Matches, _wrap_deg

__all__ = [
    "ASPECT",
    "DEGENERATE",
    "MIRRORED",
    "NOT_CONVEX",
    "REPROJ_ERROR",
    "SCALE_OUT_OF_RANGE",
    "TOO_FEW_INLIERS",
    "GeometryConfig",
    "Pose",
    "check_homography",
    "localize",
    "pose_from_homography",
]

# Reject reasons, as written to Fix.reject_reason and run.jsonl.
DEGENERATE = "degenerate"
TOO_FEW_INLIERS = "too_few_inliers"
NOT_CONVEX = "not_convex"
MIRRORED = "mirrored"
SCALE_OUT_OF_RANGE = "scale_out_of_range"
ASPECT = "aspect"
REPROJ_ERROR = "reproj_error"

# Below this turn at every footprint corner (cross product of adjacent edges, map px^2) the frame has collapsed
# to a line or a point. Any real footprint has turns of thousands of map px^2, so the exact value is not critical.
_MIN_AREA_PX2 = 1e-6


@dataclass(frozen=True)
class GeometryConfig:
    """Thresholds of the homography fit and its checks. Distances are in map px.

    ransac_px: MAGSAC++ inlier threshold. 5 px is generous for the task crops (median residual 0.16 px) and
    leaves room for blur and perspective in phone video. confidence / max_iters: MAGSAC++ stopping rule.
    min_inliers: fewer agreeing matches than this is not evidence (a random 4-point fit always has 4).
    min_scale / max_scale: plausible map px per frame px; the defaults only exclude absurd fits, the engine
    narrows them from the altitude or the previous fix. max_aspect: largest stretch of one direction over
    the other at the centre; a camera tilted by t gives about 1 / cos(t), so 2.0 allows tilts up to 60 deg.
    max_reproj_rmse_px: RMS residual of the inliers above which the fit is too loose to trust.
    """

    ransac_px: float = 5.0
    confidence: float = 0.999
    max_iters: int = 10_000
    min_inliers: int = 15
    min_scale: float = 0.01
    max_scale: float = 100.0
    max_aspect: float = 2.0
    max_reproj_rmse_px: float = 3.0

    def __post_init__(self):
        if not (self.ransac_px > 0 and 0 < self.confidence < 1 and self.max_iters > 0):
            raise ValueError("ransac_px and max_iters must be positive and confidence in (0, 1)")
        if self.min_inliers < 4:
            raise ValueError(f"min_inliers must be >= 4 (a homography needs 4 points), got {self.min_inliers}")
        if not 0 < self.min_scale < self.max_scale:
            raise ValueError(f"need 0 < min_scale < max_scale, got {self.min_scale}, {self.max_scale}")
        if self.max_aspect < 1:
            raise ValueError(f"max_aspect must be >= 1, got {self.max_aspect}")
        if self.max_reproj_rmse_px <= 0:
            raise ValueError(f"max_reproj_rmse_px must be positive, got {self.max_reproj_rmse_px}")


class Pose(NamedTuple):
    """The pose read off a homography (see :func:`pose_from_homography`)."""

    x: float              # frame centre, map px
    y: float
    heading_deg: float    # [0, 360), clockwise from map north
    scale: float          # map px per frame px at the centre
    aspect: float         # >= 1: stretch of the most over the least stretched direction at the centre
    footprint: np.ndarray  # 4 x 2 map px: frame corners TL, TR, BR, BL


def _corners(frame_size: tuple[int, int]) -> np.ndarray:
    """Frame corners TL, TR, BR, BL in the pixel-corner convention (the frame spans [0, w] x [0, h])."""
    w, h = frame_size
    return np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)


def _project(H: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map points through H: (N x 2 map px, N homogeneous w). w <= 0 means "beyond the plane's horizon"."""
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:], p[:, 2]


def _jacobian(H: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """2 x 2 derivative of the map point w.r.t. the frame point at ``pt``: how H moves and stretches there.

    For m = (H p)[:2] / w with w = (H p)[2]: dm/dp = (H[:2, :2] - m outer H[2, :2]) / w. It is the exact local
    version of "map a second point a little way off": under perspective a finite offset would mix in the
    distortion from elsewhere in the frame, the derivative does not.
    """
    u = H @ np.array([pt[0], pt[1], 1.0])
    m = u[:2] / u[2]
    return (H[:2, :2] - np.outer(m, H[2, :2])) / u[2]


def pose_from_homography(H: np.ndarray, frame_size: tuple[int, int]) -> Pose:
    """Pose of the frame on the map from a frame -> map homography.

    frame_size: (width, height) of the frame in px, as given to the matcher. Heading is the direction of the
    frame's up (0, -1) at the centre, clockwise from map north; scale is sqrt(|det J|) at the centre, the
    geometric mean of the two local stretches (for a similarity it is just the zoom). No checks: a broken H
    gives a broken pose, :func:`check_homography` is what decides.
    """
    H = np.asarray(H, dtype=np.float64)
    w, h = frame_size
    centre = np.array([w / 2.0, h / 2.0])
    c, _ = _project(H, centre[None])
    cx, cy = c[0]
    J = _jacobian(H, centre)
    up = J @ np.array([0.0, -1.0])                 # where the frame's up goes on the map (y down)
    heading = _wrap_deg(math.degrees(math.atan2(up[0], -up[1])))
    s = np.linalg.svd(J, compute_uv=False)         # local stretches, largest first
    scale = math.sqrt(abs(float(np.linalg.det(J))))
    aspect = float(s[0] / s[1]) if s[1] > 0 else math.inf
    footprint, _ = _project(H, _corners(frame_size))
    return Pose(float(cx), float(cy), heading, scale, aspect, footprint)


def check_homography(H: np.ndarray, frame_size: tuple[int, int], inliers: int, reproj_rmse: float,
                     cfg: GeometryConfig = GeometryConfig()) -> str | None:
    """The first failed check's reason code (see the module docstring), or None if H is plausible.

    inliers: how many matches agree with H. reproj_rmse: their RMS reprojection error, map px.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all() or not np.abs(H).max() > 0:
        return DEGENERATE
    w, h = frame_size
    _, wc = _project(H, np.array([[w / 2.0, h / 2.0]]))
    if not abs(wc[0]) > 1e-12 * np.abs(H).max():      # the centre itself goes to infinity
        return DEGENERATE
    if wc[0] < 0:
        H = -H                                         # same homography, centre in front: w > 0
    if inliers < cfg.min_inliers:
        return TOO_FEW_INLIERS

    footprint, wk = _project(H, _corners(frame_size))
    if (wk <= 0).any():                                # the horizon runs through the frame
        return NOT_CONVEX
    # Signed turn at each corner (z of edge_i x edge_i+1). Frame corners TL, TR, BR, BL turn the same way at
    # every corner (positive in y-down coordinates); a convex, unreflected footprint keeps all four positive.
    edges = np.roll(footprint, -1, axis=0) - footprint
    turns = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    # All turns ~0: the corners lie on a line or in a point. (Not the signed area: a bow-tie has zero signed
    # area but is twisted, not collapsed.)
    if not np.isfinite(turns).all() or np.abs(turns).max() < _MIN_AREA_PX2:
        return DEGENERATE
    if not ((turns > 0).all() or (turns < 0).all()):
        return NOT_CONVEX
    if (turns < 0).all():
        return MIRRORED

    pose = pose_from_homography(H, frame_size)
    if not cfg.min_scale <= pose.scale <= cfg.max_scale:
        return SCALE_OUT_OF_RANGE
    if pose.aspect > cfg.max_aspect:
        return ASPECT
    if reproj_rmse > cfg.max_reproj_rmse_px:
        return REPROJ_ERROR
    return None


def localize(idx: int, matches: Matches, frame_size: tuple[int, int],
             cfg: GeometryConfig = GeometryConfig()) -> Fix | None:
    """Fit a homography to ``matches`` and turn it into a checked :class:`Fix` for frame ``idx``.

    Returns None when there is no homography to report: fewer than 4 matches, MAGSAC++ found none, or the
    pose isn't finite (a frame corner exactly on the horizon). That is run.jsonl's ``fix: null``. Otherwise
    the Fix is accepted, or rejected with the reason of the first failed check; either way all its numbers
    are finite, so it can be written to run.jsonl.
    """
    n = len(matches)
    if n < 4:
        return None
    H, mask = cv2.findHomography(matches.pts_frame, matches.pts_map, method=cv2.USAC_MAGSAC,
                                 ransacReprojThreshold=cfg.ransac_px, maxIters=cfg.max_iters,
                                 confidence=cfg.confidence)
    if H is None or mask is None:
        return None
    H = np.asarray(H, dtype=np.float64)
    inl = mask.ravel().astype(bool)
    proj, _ = _project(H, matches.pts_frame[inl])
    resid = np.linalg.norm(proj - matches.pts_map[inl], axis=1)
    resid = resid[np.isfinite(resid)]
    reproj_median = float(np.median(resid)) if len(resid) else 0.0
    reproj_rmse = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else 0.0

    reason = check_homography(H, frame_size, int(inl.sum()), reproj_rmse, cfg)
    pose = pose_from_homography(H, frame_size)
    if not (np.isfinite([pose.x, pose.y, pose.heading_deg, pose.scale]).all() and np.isfinite(pose.footprint).all()
            and np.isfinite(H).all()):
        return None
    return Fix(idx=idx, H=H, x=pose.x, y=pose.y, heading_deg=pose.heading_deg, scale=pose.scale,
               footprint=pose.footprint, inliers=int(inl.sum()), inlier_ratio=float(inl.sum()) / n,
               reproj_px=reproj_median, accepted=reason is None, reject_reason=reason)
