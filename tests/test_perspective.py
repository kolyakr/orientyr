"""M2-04 · Perspective-aware checks and pose (issue #24).

"Done when: tilted frames that are visibly correct are accepted; degenerate ones are still rejected (tests with
synthetic tilted homographies)." Every homography here is built from a pinhole camera whose pose is known:

    H_map->frame = K [r1 r2 t]        (map plane z = 0; map x right, y down, z into the ground)

so the tilt, the focal length and the point straight under the camera are known exactly, and a perspective that
no centred camera can produce is known to be wrong.

- the tilt read off H (the stretch at the image centre is 1 / cos(tilt)) matches the camera to 1e-6°;
- the focal length read off H (the projective row against the tilt) matches K, without being told K;
- the ground point under the camera is recovered, with K or with the focal read off H;
- tilted cameras up to 45° with 40–100° fields of view are accepted, at any heading;
- a perspective without tilt, or along a direction the tilt does not explain, is rejected ("perspective");
- mirrored, horizon-in-frame and over-stretched tilted homographies keep their M1 reasons;
- lens distortion: undistorting with the camera's K and dist puts points back where the pinhole puts them.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest
from dronenav.camera import Camera

from dronenav.geometry import (
    PERSPECTIVE,
    GeometryConfig,
    check_homography,
    focal_from_homography,
    nadir_point,
    pose_from_homography,
)
from dronenav.sim import CutoutPose, cutout_homography

W, H_ = 1024, 768
FRAME = (W, H_)
GOOD = {"inliers": 200, "reproj_rmse": 0.5}


def focal_for_hfov(hfov_deg: float, width: int = W) -> float:
    return (width / 2) / math.tan(math.radians(hfov_deg) / 2)


def camera_h(yaw=0.0, pitch=0.0, roll=0.0, cx=2000.0, cy=1500.0, alt=1000.0, hfov=70.0, frame=FRAME):
    """Frame px -> map px of a pinhole camera at (cx, cy) map px, ``alt`` map px above the map, centred principal
    point. Nadir: camera axes = map axes (x right, y down, z into the ground). yaw turns it about the vertical,
    pitch tips the view forward (towards the frame's up), roll sideways. Returns (H, K)."""
    w, h = frame
    f = focal_for_hfov(hfov, w)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(r), 0, math.sin(r)], [0, 1, 0], [-math.sin(r), 0, math.cos(r)]])
    R_cw = Rz @ Rx @ Ry                       # camera -> world
    C = np.array([cx, cy, -alt])
    R = R_cw.T                                # world -> camera
    t = -R @ C
    H_m2f = K @ np.c_[R[:, 0], R[:, 1], t]
    H = np.linalg.inv(H_m2f)
    return H / H[2, 2], K


def true_tilt(pitch, roll):
    """Angle between the optical axis and the vertical for pitch then roll."""
    return math.degrees(math.acos(math.cos(math.radians(pitch)) * math.cos(math.radians(roll))))


# --- tilt, focal and the point under the camera, read off H -------------------------------------------------

@pytest.mark.parametrize("yaw", [0.0, 73.0, 200.0])
@pytest.mark.parametrize("pitch,roll", [(0, 0), (10, 0), (0, 8), (12, -6), (25, 5), (40, 10)])
def test_tilt_is_read_off_the_homography(yaw, pitch, roll):
    H, _ = camera_h(yaw, pitch, roll)
    assert pose_from_homography(H, FRAME).tilt_deg == pytest.approx(true_tilt(pitch, roll), abs=1e-6)


def test_tilt_of_a_nadir_view_is_zero():
    H, _ = camera_h(30.0)
    assert pose_from_homography(H, FRAME).tilt_deg == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("hfov", [40.0, 70.0, 100.0])
@pytest.mark.parametrize("pitch,roll", [(10, 0), (5, 5), (30, -10)])
def test_focal_is_read_off_the_homography(hfov, pitch, roll):
    H, K = camera_h(140.0, pitch, roll, hfov=hfov)
    assert focal_from_homography(H, FRAME) == pytest.approx(K[0, 0], rel=1e-6)


def test_focal_of_a_nadir_view_is_unknown():
    # no tilt: the perspective vanishes, so nothing in H says how long the lens is
    H, _ = camera_h(30.0)
    assert focal_from_homography(H, FRAME) is None


@pytest.mark.parametrize("pitch,roll", [(10, 0), (12, -6), (30, 10)])
@pytest.mark.parametrize("use_k", [True, False])
def test_nadir_point_is_the_ground_under_the_camera(pitch, roll, use_k):
    H, K = camera_h(65.0, pitch, roll, cx=2100.0, cy=1300.0, alt=800.0)
    nx, ny = nadir_point(H, FRAME, focal=K[0, 0] if use_k else None)
    # exact up to float round-off: the camera is a pure pinhole
    assert math.hypot(nx - 2100.0, ny - 1300.0) < 1e-6 * 800
    # and it is not where the image centre lands: that is ~alt * tan(tilt) away
    p = pose_from_homography(H, FRAME)
    assert math.hypot(p.x - nx, p.y - ny) == pytest.approx(800 * math.tan(math.radians(true_tilt(pitch, roll))),
                                                           rel=1e-6)


def test_nadir_point_of_a_nadir_view_is_the_centre():
    H, _ = camera_h(30.0, cx=1234.0, cy=987.0)
    assert nadir_point(H, FRAME) == pytest.approx((1234.0, 987.0), abs=1e-6)


def test_nadir_point_without_any_focal_is_unknown_for_a_tilted_view_without_perspective():
    # an affine "tilt" (stretch, no projective row) has no camera behind it to find the nadir of
    H = np.array([[0.5, 0, 1000], [0, 1.0, 800], [0, 0, 1.0]])
    assert nadir_point(H, FRAME) is None


# --- accepted: tilted cameras ------------------------------------------------------------------------------

@pytest.mark.parametrize("yaw", [0.0, 95.0, 260.0])
@pytest.mark.parametrize("pitch,roll", [(10, 0), (15, -8), (25, 8), (40, 15)])
@pytest.mark.parametrize("hfov", [40.0, 70.0, 100.0])
def test_tilted_cameras_are_accepted(yaw, pitch, roll, hfov):
    H, _ = camera_h(yaw, pitch, roll, hfov=hfov)
    assert check_homography(H, FRAME, **GOOD) is None


@pytest.mark.parametrize("tilt,tilt_dir", [(10.0, 0.0), (20.0, 45.0), (35.0, 200.0)])
@pytest.mark.parametrize("frame", [(640, 480), (320, 240)])
def test_simulator_tilted_cutouts_are_accepted(tilt, tilt_dir, frame):
    # the M6 simulator's tilted cut-outs (focal = the frame's longer side) must pass the checks they are scored by
    H = cutout_homography(CutoutPose(1500.0, 1000.0, 30.0, 0.8, tilt, tilt_dir), frame)
    assert check_homography(H, frame, **GOOD) is None
    assert pose_from_homography(H, frame).tilt_deg == pytest.approx(tilt, abs=1e-6)


# --- rejected: perspective no camera makes ------------------------------------------------------------------

def _with_projective_row(H, frame, g):
    """H followed by a pure projective change in the frame: (x, y) -> (x, y) / (1 + g . (p - centre)). Keeps the
    centre and its scale-free shape but adds a perspective of ``g`` (1 / frame px) along the frame direction g."""
    w, h = frame
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]])
    P = np.array([[1, 0, 0], [0, 1, 0], [g[0], g[1], 1.0]])
    return H @ np.linalg.inv(T) @ P @ T


def test_perspective_without_tilt_is_rejected():
    # a strong keystone on a view that is not stretched at all: the depth changes 40 % across the frame but the
    # ground is not foreshortened. A camera would have to have a fish-eye's focal length to do that.
    H, _ = camera_h(20.0)
    Hb = _with_projective_row(H, FRAME, (0.0, 0.4 / (H_ / 2)))
    assert check_homography(Hb, FRAME, **GOOD) == PERSPECTIVE


def test_perspective_across_the_tilt_is_rejected():
    # tilted forward (stretch along the frame's y), but the depth changes along x: no centred camera does that
    H, _ = camera_h(0.0, 20.0)
    Hb = _with_projective_row(H, FRAME, (0.3 / (W / 2), 0.0))
    assert check_homography(Hb, FRAME, **GOOD) == PERSPECTIVE


def test_small_perspective_noise_is_tolerated():
    # a nadir fit whose projective row is off by 1 % of depth across the frame (RANSAC noise) stays accepted
    H, _ = camera_h(20.0)
    Hb = _with_projective_row(H, FRAME, (0.01 / (W / 2), -0.01 / (H_ / 2)))
    assert check_homography(Hb, FRAME, **GOOD) is None


def test_task_crops_style_similarity_is_accepted():
    # the organisers' crops: rotation + scale, no perspective at all
    H, _ = camera_h(123.0)
    H[2, :2] = 0.0
    assert check_homography(H / H[2, 2], FRAME, **GOOD) is None


# --- rejected: the M1 reasons still hold on tilted views -----------------------------------------------------

def test_tilted_mirror_is_still_mirrored():
    H, _ = camera_h(40.0, 15.0, 5.0)
    flip = np.array([[-1, 0, W], [0, 1, 0], [0, 0, 1.0]])
    assert check_homography(H @ flip, FRAME, **GOOD) == "mirrored"


def test_horizon_in_frame_is_still_not_convex():
    # tilted 70° with a 100° field of view: the top of the frame looks above the horizon
    H, _ = camera_h(0.0, 70.0, hfov=100.0)
    assert check_homography(H, FRAME, **GOOD) == "not_convex"


def test_tilt_beyond_the_aspect_limit_is_still_aspect():
    H, _ = camera_h(10.0, 55.0, hfov=30.0)       # 1 / cos 55° = 1.74 > 1.5
    assert check_homography(H, FRAME, **GOOD, cfg=GeometryConfig(max_aspect=1.5)) == "aspect"


def test_perspective_check_can_be_turned_off():
    H, _ = camera_h(20.0)
    Hb = _with_projective_row(H, FRAME, (0.0, 0.4 / (H_ / 2)))
    assert check_homography(Hb, FRAME, **GOOD, cfg=GeometryConfig(max_hfov_deg=None)) is None


@pytest.mark.parametrize("kw", [{"max_hfov_deg": 0.0}, {"max_hfov_deg": 180.0}, {"perspective_tol": -0.1}])
def test_config_rejects_bad_perspective_limits(kw):
    with pytest.raises(ValueError):
        GeometryConfig(**kw)


# --- lens distortion ------------------------------------------------------------------------------------------

K_IRPIN = np.array([[731.2117794519627, 0.0, 512.0], [0.0, 731.2117794519627, 384.0], [0.0, 0.0, 1.0]])
DIST_IRPIN = [-0.08, 0.012, 0, 0, 0]


def _distort(pts, K, dist):
    """Pinhole pixels -> where the distorted lens puts them (OpenCV's model)."""
    n = cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, None)   # to normalised coords
    obj = np.c_[n.reshape(-1, 2), np.ones(len(pts))]
    out, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, np.asarray(dist, float))
    return out.reshape(-1, 2)


def test_camera_from_meta_reads_k_and_dist():
    meta = {"camera": {"size": [1024, 768], "K": K_IRPIN.tolist(), "dist": DIST_IRPIN}}
    cam = Camera.from_meta(meta)
    assert cam.size == (1024, 768)
    assert np.allclose(cam.K, K_IRPIN) and np.allclose(cam.dist, DIST_IRPIN)


def test_camera_from_meta_without_a_camera_is_none():
    assert Camera.from_meta({"fps": 30}) is None


def test_undistort_points_inverts_the_lens():
    cam = Camera(K_IRPIN, np.array(DIST_IRPIN), (1024, 768))
    pinhole = np.array([[512, 384], [20, 20], [1000, 700], [100, 600], [900, 50]], float)
    assert np.allclose(cam.undistort_points(_distort(pinhole, cam.K, cam.dist)), pinhole, atol=0.01)


def test_undistort_image_straightens_a_line():
    # a straight line drawn in pinhole pixels, bent by the lens, is straight again after undistortion
    cam = Camera(K_IRPIN, np.array(DIST_IRPIN), (1024, 768))
    xs = np.linspace(40, 984, 200)
    line = np.c_[xs, np.full_like(xs, 80.0)]
    img = np.zeros((768, 1024), np.uint8)
    bent = _distort(line, cam.K, cam.dist)
    cv2.polylines(img, [np.round(bent * 16).astype(np.int32)], False, 255, 3, cv2.LINE_AA, shift=4)
    ys, _ = np.nonzero(cam.undistort_image(img)[:, 100:924] > 128)
    assert np.ptp(ys) <= 4 and abs(np.median(ys) - 80) <= 1.0     # flat to the line's own width


def test_undistort_image_rejects_another_size():
    cam = Camera(K_IRPIN, np.array(DIST_IRPIN), (1024, 768))
    with pytest.raises(ValueError, match="1024 x 768"):
        cam.undistort_image(np.zeros((480, 640), np.uint8))
