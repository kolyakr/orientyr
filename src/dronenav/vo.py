"""Visual odometry (M9-02): the drone's motion from consecutive frames, as the filter's motion input.

When the map match fails (a field with no features, clouds, a matcher jammed or too slow) the camera usually still
sees the ground, and consecutive frames overlap. Matching frame k to frame k-1 gives the homography between them
(SIFT on the two frames, Lowe's ratio, RANSAC), with no map at all. Chained onto the last frame -> map homography
(from the last accepted fix) it says how far the frame centre moved *on the map*: a velocity in map px/s, which the
Kalman filter takes as a measurement (``KalmanFilter.update_velocity``). The tracker then keeps a moving estimate
through map-match misses instead of coasting on a stale velocity and holding still.

Why a velocity and not the chained position: the chain drifts (every link adds a little scale and rotation error),
and a position from it would look precise while running away. The step between two frames is short, so its error
is small, and the filter integrates it with an honestly growing uncertainty. Every accepted fix restarts the chain
from the fix's homography, so the drift never outlives a map match.

Sparse optical flow (LK) was the other candidate the task names. At this project's frame rates consecutive frames
are a third of a frame apart (a cut-out flight steps ~110 px in 320) and turn by up to several degrees, which is
outside what pyramidal LK tracks without a motion guess; descriptors match it directly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import cv2
import numpy as np

from dronenav.types import Fix

__all__ = ["Motion", "VisualOdometry", "VoConfig", "VoVelocity", "frame_motion"]


@dataclass(frozen=True)
class VoConfig:
    """max_side: frames are shrunk to this longer side before matching (speed; the motion is scaled back).
    ratio: Lowe's ratio. ransac_px: RANSAC's reprojection threshold, px at the matching size.
    min_inliers: fewer RANSAC inliers than this is no motion (noise matches a few by chance, never this many).
    min_inlier_ratio: the same, as a share of the ratio-test survivors.
    floor_px: the least error credited to one frame-to-frame step, frame px (resampling, feature localisation).
    rel_sigma: the chain's relative error per step (scale and rotation of the links), as a share of the step."""

    max_side: int = 640
    ratio: float = 0.75
    ransac_px: float = 2.0
    min_inliers: int = 30
    min_inlier_ratio: float = 0.3
    floor_px: float = 0.5
    rel_sigma: float = 0.03


class Motion(NamedTuple):
    """H: 3 x 3 homography cur frame px -> prev frame px. inliers: RANSAC inliers. rms_px: their reprojection RMS."""

    H: np.ndarray
    inliers: int
    rms_px: float


class VoVelocity(NamedTuple):
    """The drone's velocity from VO: v (vx, vy) in map px/s, cov its 2 x 2 covariance, inliers behind it."""

    v: np.ndarray
    cov: np.ndarray
    inliers: int


_SIFT = None


def _sift():
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create()
    return _SIFT


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _shrink(gray: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    f = min(1.0, max_side / max(gray.shape[:2]))
    if f < 1.0:
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return gray, f


def frame_motion(prev: np.ndarray, cur: np.ndarray, cfg: VoConfig | None = None) -> Motion | None:
    """The homography taking ``cur``'s pixels to ``prev``'s, or None when the frames share too little to trust.

    Images are BGR or grayscale, at the frames' own resolution; H is in those pixels."""
    cfg = cfg or VoConfig()
    (g0, f0), (g1, f1) = _shrink(_gray(prev), cfg.max_side), _shrink(_gray(cur), cfg.max_side)
    k0, d0 = _sift().detectAndCompute(g0, None)
    k1, d1 = _sift().detectAndCompute(g1, None)
    if d0 is None or d1 is None or len(k0) < cfg.min_inliers or len(k1) < cfg.min_inliers:
        return None                                       # a blank frame: nothing to match
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d0, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < cfg.ratio * p[1].distance]
    if len(good) < cfg.min_inliers:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k0[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, cfg.ransac_px, maxIters=5000, confidence=0.999)
    if H is None or mask is None:
        return None
    inl = mask.ravel().astype(bool)
    n = int(inl.sum())
    if n < cfg.min_inliers or n < cfg.min_inlier_ratio * len(good):
        return None
    J = H[:2, :2] / H[2, 2]
    det = float(np.linalg.det(J))
    if not 0.5 < det < 2.0:                               # mirrored, collapsed or a 2x zoom between frames: wrong
        return None
    p = cv2.perspectiveTransform(src[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((p - dst[inl]) ** 2, axis=1))))
    # back to full resolution: H_full = S0^-1 H S1, with S the shrink of each frame
    S0, S1 = np.diag([f0, f0, 1.0]), np.diag([f1, f1, 1.0])
    H_full = np.linalg.inv(S0) @ H @ S1
    return Motion(H_full / H_full[2, 2], n, rms / f0)


def _apply(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    p = H @ np.array([xy[0], xy[1], 1.0])
    return p[:2] / p[2]


def _local_scale(H: np.ndarray, xy: np.ndarray) -> float:
    """Map px per frame px of H near the point xy (the root of the local area ratio)."""
    e = 1.0
    o, a, b = _apply(H, xy), _apply(H, xy + (e, 0.0)), _apply(H, xy + (0.0, e))
    (ux, uy), (wx, wy) = a - o, b - o
    return math.sqrt(abs(float(ux * wy - uy * wx))) / e


class VisualOdometry:
    """Frame-to-frame motion on the map. Per frame, in time order: ``step(t, image, fix)`` with the frame's map fix
    (None, rejected or accepted); returns the velocity since the previous frame, or None."""

    def __init__(self, cfg: VoConfig | None = None):
        self.cfg = cfg or VoConfig()
        self._prev: tuple[float, np.ndarray] | None = None   # the previous frame: (t, gray image)
        self._H_map: np.ndarray | None = None                 # its frame -> map homography (fix or chained)

    def step(self, t: float, image: np.ndarray, fix: Fix | None = None) -> VoVelocity | None:
        t, gray = float(t), _gray(image)
        out = None
        if self._prev is not None and self._H_map is not None and t > self._prev[0]:
            t0, g0 = self._prev
            mo = frame_motion(g0, gray, self.cfg)
            if mo is None:
                self._H_map = None                            # the chain is broken until the next fix
            else:
                c = np.array([gray.shape[1] / 2, gray.shape[0] / 2])
                H_cur = self._H_map @ mo.H
                d = _apply(H_cur, c) - _apply(self._H_map, c)
                dt = t - t0
                s = _local_scale(self._H_map, c)
                sigma = s * (self.cfg.floor_px + mo.rms_px) + self.cfg.rel_sigma * float(np.hypot(*d))
                out = VoVelocity(d / dt, np.eye(2) * (sigma / dt) ** 2, mo.inliers)
                self._H_map = H_cur / H_cur[2, 2]
        if fix is not None and fix.accepted:
            self._H_map = np.asarray(fix.H, float)            # the map match is the truth: restart the chain on it
        self._prev = (t, gray)
        return out
