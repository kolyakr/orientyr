"""Metric 6DoF pose from a calibrated camera and planar frame<->map correspondences (M9-03).

``geometry.pose_from_homography`` already reads a *combined* tilt off a homography (M2-04):
it needs no camera intrinsics, but can't tell roll from pitch apart, and has no real-world
altitude. Here, with a calibrated :class:`~dronenav.camera.Camera` and the map's real-world
scale (``m_per_px``), the same inlier correspondences give a full planar pose
(``cv2.solvePnP``, ``SOLVEPNP_IPPE``) split into altitude (metres), roll and pitch (degrees).

Yaw is not new: it is exactly ``Fix.heading_deg`` (M2-04), so it is not written back to
telemetry from here (two slightly-disagreeing yaws would be confusing, not useful) -- but it
is still recovered and returned, so a caller (and this module's own tests) can cross-check it.

Convention (matches tests/test_perspective.py's own M2-04 test helper ``camera_h``, so both
are provably consistent): world coordinates are the map's px convention carried into metres --
x right, y "down" (south), z into the ground, so a camera ``alt`` metres above the map plane
(z = 0) sits at world z = -alt. The camera's rotation from nadir is
``R_cw = Rz(yaw) @ Rx(pitch) @ Ry(roll)`` (camera -> world); ``cv2.solvePnP`` returns the
world -> camera rotation ``R = R_cw.T`` and translation ``t`` such that ``Xc = R @ Xw + t``.

IPPE's planar-pose ambiguity (two candidate poses with similar reprojection error at low tilt,
one the true pose and one its mirror through the image plane) is resolved by keeping only
candidates with the camera above the plane (``altitude_m > 0``) and taking the one with the
smaller reprojection error among those.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import cv2
import numpy as np

from dronenav.camera import Camera
from dronenav.types import _wrap_deg

__all__ = ["Pose6DoF", "solve_planar_pose"]

_MIN_POINTS = 4  # a planar pose needs at least 4 non-collinear points (IPPE's own minimum)


class Pose6DoF(NamedTuple):
    """The pose read off a calibrated camera's planar PnP solve (see :func:`solve_planar_pose`)."""

    altitude_m: float      # camera height above the map plane, metres
    roll_deg: float        # camera roll from nadir, degrees
    pitch_deg: float        # camera pitch from nadir, degrees
    yaw_deg: float          # camera yaw; equals Fix.heading_deg up to fit noise, not written to telemetry
    reproj_rmse: float      # RMS reprojection error of the chosen solution, frame px


def _euler_from_R_cw(R_cw: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) degrees from R_cw = Rz(yaw) @ Rx(pitch) @ Ry(roll), the closed-form
    inverse of that product (see the module docstring for the convention)."""
    pitch = math.asin(float(np.clip(-R_cw[2, 1], -1.0, 1.0)))
    roll = math.atan2(-R_cw[2, 0], R_cw[2, 2])
    yaw = math.atan2(-R_cw[0, 1], R_cw[1, 1])
    return _wrap_deg(math.degrees(yaw)), math.degrees(pitch), math.degrees(roll)


def solve_planar_pose(pts_frame, pts_map, camera: Camera, m_per_px: float) -> Pose6DoF | None:
    """Solve the planar pose from frame<->map inlier correspondences and a calibrated camera.

    pts_frame: N x 2 frame px (as given to ``camera.K``/``camera.dist``, not pre-undistorted --
    ``solvePnP`` takes the distortion itself). pts_map: N x 2 map px of the same points.
    m_per_px: the map's real-world scale, metres per map px (from a georef sidecar, or given
    explicitly), so the object points are in real metres and ``altitude_m`` is metric.

    Returns None when there is no pose to report: fewer than 4 points, a degenerate (near-
    collinear) planar configuration, or neither IPPE candidate has the camera above the plane.
    """
    pts_frame = np.asarray(pts_frame, dtype=np.float64).reshape(-1, 2)
    pts_map = np.asarray(pts_map, dtype=np.float64).reshape(-1, 2)
    n = len(pts_frame)
    if n < _MIN_POINTS or len(pts_map) != n:
        return None
    object_points = np.c_[pts_map * m_per_px, np.zeros(n)].reshape(-1, 1, 3)
    image_points = pts_frame.reshape(-1, 1, 2)
    try:
        ok, rvecs, tvecs, errors = cv2.solvePnPGeneric(object_points, image_points, camera.K, camera.dist,
                                                        flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    if not ok or not rvecs:
        return None
    errors = np.ravel(errors) if errors is not None else [math.nan] * len(rvecs)

    best = None
    for rvec, tvec, err in zip(rvecs, tvecs, errors):
        R, _ = cv2.Rodrigues(rvec)
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        if not (np.isfinite(R).all() and np.isfinite(t).all()):
            continue
        C = -R.T @ t                            # camera centre, world metres
        altitude = -float(C[2])
        if not (math.isfinite(altitude) and altitude > 0):
            continue                             # the mirror solution: camera "below" the plane
        err = float(err) if math.isfinite(err) else _reproj_rmse(object_points, image_points, R, t, camera.K)
        if best is None or err < best[0]:
            best = (err, R, altitude)
    if best is None:
        return None

    err, R, altitude = best
    yaw, pitch, roll = _euler_from_R_cw(R.T)
    return Pose6DoF(altitude_m=altitude, roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw, reproj_rmse=err)


def _reproj_rmse(object_points: np.ndarray, image_points: np.ndarray, R: np.ndarray, t: np.ndarray,
                 K: np.ndarray) -> float:
    """RMS reprojection error, frame px, when ``solvePnPGeneric`` didn't give one of its own."""
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(object_points, rvec, t, K, None)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - image_points.reshape(-1, 2)) ** 2, axis=1))))
