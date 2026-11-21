"""M9-03 · 6DoF pose for the phone video (issue #74).

"Done when: tilting the phone shows the matching tilt in the telemetry within a few degrees."
M2-04 (test_perspective.py) already reads a *combined* tilt off the homography, with no camera
intrinsics and no metres. This solves the planar pose with real intrinsics (``cv2.solvePnP``,
``SOLVEPNP_IPPE``) from the frame<->map inlier correspondences, splitting that combined tilt
into roll and pitch, and giving a metric altitude. Yaw is already solved elsewhere (it is
exactly ``Fix.heading_deg``); this module still recovers it internally (and exposes it) so the
synthetic truth below can check the whole rotation, not just two of its three angles.

Convention: reuses test_perspective.py's own pinhole model exactly --
``R_cw = Rz(yaw) @ Rx(pitch) @ Ry(roll)`` (camera -> world), world -> camera ``R = R_cw.T``,
world x right, y "down" (south), z into the ground (so a camera ``alt`` metres above the plane
sits at world z = -alt). Object points here are already in metres (``m_per_px`` folded in by
the caller), so this test builds them directly in metres and does not need a map at all.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from dronenav.camera import Camera
from dronenav.pose6dof import solve_planar_pose

FRAME = (1024, 768)
GRID = [(dx, dy) for dx in (-40.0, -20.0, 0.0, 20.0, 40.0) for dy in (-30.0, 0.0, 30.0)]  # 15 pts, metres


def focal_for_hfov(hfov_deg: float, width: int) -> float:
    return (width / 2) / math.tan(math.radians(hfov_deg) / 2)


def camera_pose(yaw=0.0, pitch=0.0, roll=0.0, cx=200.0, cy=150.0, alt=100.0, hfov=70.0, frame=FRAME):
    """K, R (world -> camera), t for a pinhole camera at (cx, cy, metres) alt metres above the z=0
    plane, same rotation convention as test_perspective.py's camera_h."""
    w, h = frame
    f = focal_for_hfov(hfov, w)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(r), 0, math.sin(r)], [0, 1, 0], [-math.sin(r), 0, math.cos(r)]])
    R_cw = Rz @ Rx @ Ry
    C = np.array([cx, cy, -alt])
    R = R_cw.T
    t = -R @ C
    return K, R, t


def project(K, R, t, pts_world_xy):
    """World (x, y, 0) metres -> frame px, for a pinhole K, R (world -> camera), t."""
    Xw = np.c_[pts_world_xy, np.zeros(len(pts_world_xy))]
    Xc = Xw @ R.T + t
    p = Xc @ K.T
    return p[:, :2] / p[:, 2:]


def correspondences(yaw=0.0, pitch=0.0, roll=0.0, cx=200.0, cy=150.0, alt=100.0, hfov=70.0,
                    frame=FRAME, offsets=GRID):
    """(pts_frame, pts_map, camera, m_per_px) for solve_planar_pose, with m_per_px = 1 so map px
    == world metres: pts_map is directly the synthetic truth's world (x, y)."""
    K, R, t = camera_pose(yaw, pitch, roll, cx, cy, alt, hfov, frame)
    pts_map = np.array([(cx + dx, cy + dy) for dx, dy in offsets])
    pts_frame = project(K, R, t, pts_map)
    camera = Camera(K, np.zeros(5), frame)
    return pts_frame, pts_map, camera, 1.0


@pytest.mark.parametrize("yaw", [0.0, 60.0, 190.0])
@pytest.mark.parametrize("pitch,roll", [(0, 0), (8, 0), (0, 6), (10, -5), (20, 8), (30, -12)])
def test_pose_is_recovered_from_noise_free_correspondences(yaw, pitch, roll):
    pts_frame, pts_map, camera, m_per_px = correspondences(yaw, pitch, roll, alt=100.0)
    pose = solve_planar_pose(pts_frame, pts_map, camera, m_per_px)
    assert pose is not None
    # noise-free synthetic data, a closed-form (IPPE) solve: tight tolerances, not a loose fit
    assert pose.altitude_m == pytest.approx(100.0, abs=1e-2)
    assert pose.roll_deg == pytest.approx(roll, abs=5e-2)
    assert pose.pitch_deg == pytest.approx(pitch, abs=5e-2)
    assert (pose.yaw_deg - yaw) % 360 == pytest.approx(0.0, abs=5e-2) or \
        (pose.yaw_deg - yaw) % 360 == pytest.approx(360.0, abs=5e-2)


@pytest.mark.parametrize("alt", [20.0, 100.0, 500.0])
def test_altitude_scales_with_the_true_height(alt):
    pts_frame, pts_map, camera, m_per_px = correspondences(pitch=12.0, roll=-4.0, alt=alt)
    pose = solve_planar_pose(pts_frame, pts_map, camera, m_per_px)
    assert pose is not None
    assert pose.altitude_m == pytest.approx(alt, rel=1e-3)


def test_a_nadir_view_has_no_roll_or_pitch():
    pts_frame, pts_map, camera, m_per_px = correspondences(yaw=40.0, pitch=0.0, roll=0.0)
    pose = solve_planar_pose(pts_frame, pts_map, camera, m_per_px)
    assert pose is not None
    assert pose.roll_deg == pytest.approx(0.0, abs=5e-2)
    assert pose.pitch_deg == pytest.approx(0.0, abs=5e-2)


def test_fewer_than_four_points_is_not_a_pose():
    pts_frame, pts_map, camera, m_per_px = correspondences(offsets=GRID[:3])
    assert solve_planar_pose(pts_frame, pts_map, camera, m_per_px) is None


def test_collinear_points_are_degenerate_not_a_pose():
    # every object point on the line y = cy: a planar PnP solve has no unique pose to find
    offsets = [(dx, 0.0) for dx in (-40.0, -20.0, -10.0, 10.0, 20.0, 40.0)]
    pts_frame, pts_map, camera, m_per_px = correspondences(pitch=10.0, offsets=offsets)
    assert solve_planar_pose(pts_frame, pts_map, camera, m_per_px) is None


def test_no_matches_is_not_a_pose():
    camera = Camera(np.eye(3) * 500 + np.diag([0, 0, 1]), np.zeros(5), FRAME)
    assert solve_planar_pose(np.zeros((0, 2)), np.zeros((0, 2)), camera, 1.0) is None


def test_near_fronto_parallel_resolves_the_true_solution_not_its_mirror():
    # IPPE's known ambiguity zone: at low tilt the two planar-pose candidates have similar
    # reprojection error. The true one has the camera above the plane and (here) the smaller tilt.
    pts_frame, pts_map, camera, m_per_px = correspondences(pitch=3.0, roll=0.0, alt=100.0)
    pose = solve_planar_pose(pts_frame, pts_map, camera, m_per_px)
    assert pose is not None
    assert pose.altitude_m > 0
    assert pose.pitch_deg == pytest.approx(3.0, abs=5e-2)
