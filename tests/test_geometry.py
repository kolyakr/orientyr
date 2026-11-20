"""M1-04 · Geometry: homography, checks, pose.

"Done when: unit tests with synthetic homographies (known rotation, scale, translation, mirrored, degenerate)
pass." Every test builds the frame -> map homography itself, so the true pose is known exactly:

- pose from an exact H: centre, heading, scale and footprint match the construction;
- localize() recovers a known pose from noisy matches with outliers (MAGSAC++);
- each sanity check rejects its own failure with its own reason code: degenerate, too_few_inliers,
  not_convex, mirrored, scale_out_of_range, aspect, reproj_error;
- no homography at all (fewer than 4 matches, empty input) gives None, as run.jsonl's ``fix: null``;
- every Fix produced is finite, so it can be written to run.jsonl (which forbids NaN / Infinity).
"""
import math

import numpy as np
import pytest

from dronenav.geometry import GeometryConfig, check_homography, localize, pose_from_homography
from dronenav.telemetry import Record, record_to_dict
from dronenav.types import Fix, Matches, TrackState

W, H_ = 640, 480            # frame size (width, height) in frame px
FRAME = (W, H_)
CORNERS = np.array([[0, 0], [W, 0], [W, H_], [0, H_]], float)   # TL, TR, BR, BL (pixel-corner convention)


def make_h(heading_deg=0.0, scale=1.0, cx=1000.0, cy=800.0, mirror=False, sx=1.0, sy=1.0):
    """Frame px -> map px: frame centre to (cx, cy), the frame's up turned ``heading_deg`` clockwise from north.

    Built as T(cx, cy) @ R @ S @ T(-W/2, -H/2). With y down, R = [[cos, -sin], [sin, cos]] sends the frame's up
    (0, -1) to (sin, -cos): heading 90 points east (+x), the module's convention. ``sx``, ``sy`` stretch the axes
    (for the aspect check); ``mirror`` flips x (a camera can never produce that).
    """
    a = math.radians(heading_deg)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    S = np.diag([scale * sx * (-1 if mirror else 1), scale * sy, 1.0])
    to_centre = np.array([[1, 0, -W / 2], [0, 1, -H_ / 2], [0, 0, 1]], float)
    to_map = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], float)
    return to_map @ R @ S @ to_centre


def apply(H, pts):
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:]


def synth_matches(H, n=300, outlier_frac=0.3, noise_px=0.3, seed=0, map_size=(4000, 3000)):
    """n correspondences: frame points uniform in the frame, true map points through H plus Gaussian noise
    (map px), and a fraction replaced by random map points (wrong matches). Seeded: same input, same result."""
    rng = np.random.default_rng(seed)
    pf = rng.uniform([0, 0], [W, H_], size=(n, 2))
    pm = apply(H, pf) + rng.normal(0, noise_px, size=(n, 2))
    bad = rng.random(n) < outlier_frac
    pm[bad] = rng.uniform([0, 0], map_size, size=(bad.sum(), 2))
    return Matches(pf, pm, np.ones(n), "synthetic", 0.0)


def angdiff(a, b):
    """Smallest absolute difference of two angles in degrees (359.9 vs 0.1 is 0.2, not 359.8)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


GOOD = dict(inliers=200, reproj_rmse=0.2)   # check_homography inputs that pass, to isolate the geometric checks


# --- pose from an exact homography --------------------------------------------------------------------------

@pytest.mark.parametrize("heading", [0.0, 37.0, 90.0, 180.0, 271.5, 359.9])
@pytest.mark.parametrize("scale", [0.2535, 1.0, 3.0])   # 0.2535 = the task crops (4x finer than the map)
def test_pose_exact_rotation_scale_translation(heading, scale):
    H = make_h(heading, scale, cx=1234.5, cy=678.9)
    p = pose_from_homography(H, FRAME)
    # an exact H has no noise: only float round-off is left
    assert p.x == pytest.approx(1234.5, abs=1e-6) and p.y == pytest.approx(678.9, abs=1e-6)
    assert angdiff(p.heading_deg, heading) < 1e-6
    assert 0.0 <= p.heading_deg < 360.0
    assert p.scale == pytest.approx(scale, rel=1e-9)
    np.testing.assert_allclose(p.footprint, apply(H, CORNERS), atol=1e-6)


def test_heading_convention_east_is_90():
    # the frame's up pointing to map +x (east) is heading 90; to map +y (south) is 180
    assert pose_from_homography(make_h(90.0), FRAME).heading_deg == pytest.approx(90.0, abs=1e-9)
    assert pose_from_homography(make_h(180.0), FRAME).heading_deg == pytest.approx(180.0, abs=1e-9)


def test_footprint_order_tl_tr_br_bl():
    p = pose_from_homography(make_h(0.0, 1.0, cx=1000, cy=800), FRAME)
    np.testing.assert_allclose(p.footprint, [[680, 560], [1320, 560], [1320, 1040], [680, 1040]], atol=1e-9)


@pytest.mark.parametrize("eps", [1e-17, 1e-15, 1e-13])
def test_heading_just_left_of_north_stays_in_range(eps):
    # a north-up frame with a hair of noise: atan2 gives about -eps degrees, which must wrap below 360 (#86)
    p = pose_from_homography(make_h(-eps), FRAME)
    assert 0.0 <= p.heading_deg < 360.0 and angdiff(p.heading_deg, 0.0) < 1e-9


def test_pose_translation_only():
    p = pose_from_homography(np.array([[1, 0, 50.0], [0, 1, -20.0], [0, 0, 1]]), FRAME)
    assert (p.x, p.y) == pytest.approx((W / 2 + 50, H_ / 2 - 20))
    assert p.heading_deg == pytest.approx(0.0, abs=1e-9) and p.scale == pytest.approx(1.0)


# --- localize: matches -> Fix ---------------------------------------------------------------------------------

@pytest.mark.parametrize("heading,scale", [(0.0, 0.2535), (63.0, 0.2535), (200.0, 1.0), (315.0, 2.0)])
def test_localize_recovers_known_pose_with_outliers(heading, scale):
    H = make_h(heading, scale, cx=2100.0, cy=1300.0)
    m = synth_matches(H, n=300, outlier_frac=0.3, noise_px=0.3, seed=1)
    fix = localize(7, m, FRAME)
    assert isinstance(fix, Fix) and fix.accepted and fix.reject_reason is None
    assert fix.idx == 7
    # ~200 inliers with 0.3 px noise put the fitted centre within ~0.05 px; 0.5 px / 0.2 deg / 0.5 % leave
    # room for MAGSAC's sampling while still catching a wrong convention (those are off by many px / degrees)
    assert math.hypot(fix.x - 2100.0, fix.y - 1300.0) < 0.5
    assert angdiff(fix.heading_deg, heading) < 0.2
    assert fix.scale == pytest.approx(scale, rel=5e-3)
    # 70 % of the 300 are true matches; MAGSAC may keep a few lucky outliers
    assert 190 <= fix.inliers <= 230
    assert fix.inlier_ratio == pytest.approx(fix.inliers / 300)
    assert 0.0 <= fix.reproj_px < 1.0           # median residual of inliers, ~0.3 px noise
    np.testing.assert_allclose(fix.footprint, apply(H, CORNERS), atol=2.0)


def test_localize_fix_has_the_fitted_h():
    H = make_h(30.0, 0.5)
    fix = localize(0, synth_matches(H, outlier_frac=0.0, noise_px=0.0), FRAME)
    Hn = fix.H / fix.H[2, 2]
    np.testing.assert_allclose(Hn, H / H[2, 2], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("n", [0, 1, 3])
def test_localize_without_enough_matches_is_none(n):
    # fewer than 4 correspondences: no homography exists, so there is no Fix (run.jsonl writes fix: null)
    m = synth_matches(make_h(), n=n, outlier_frac=0.0) if n else Matches.empty("synthetic")
    assert localize(0, m, FRAME) is None


def test_localize_too_few_inliers():
    # 10 perfect matches give an exact H, but 10 < min_inliers: too little evidence to trust it
    m = synth_matches(make_h(10.0, 0.5), n=10, outlier_frac=0.0, noise_px=0.0)
    fix = localize(0, m, FRAME, GeometryConfig(min_inliers=15))
    assert not fix.accepted and fix.reject_reason == "too_few_inliers"


def test_localize_mirrored():
    m = synth_matches(make_h(25.0, 0.5, mirror=True), outlier_frac=0.2, seed=2)
    fix = localize(0, m, FRAME)
    assert not fix.accepted and fix.reject_reason == "mirrored"


def test_localize_reprojection_error():
    # 4 px noise is inside a loose 20 px RANSAC threshold, so everything is an inlier, but the RMSE (~5.7 px)
    # is far above 2 px: the fit is too sloppy to trust
    m = synth_matches(make_h(0.0, 1.0), n=200, outlier_frac=0.0, noise_px=4.0, seed=3)
    fix = localize(0, m, FRAME, GeometryConfig(ransac_px=20.0, max_reproj_rmse_px=2.0))
    assert not fix.accepted and fix.reject_reason == "reproj_error"


def test_every_fix_is_finite_and_serialisable():
    # accepted and rejected fixes alike go to run.jsonl, which refuses NaN and Infinity
    cases = [synth_matches(make_h(40.0, 0.3), seed=4),
             synth_matches(make_h(25.0, 0.5, mirror=True), seed=5),
             synth_matches(make_h(0.0, 1.0, sx=6.0), seed=6),
             synth_matches(make_h(), n=10, outlier_frac=0.0)]
    for i, m in enumerate(cases):
        fix = localize(i, m, FRAME)
        assert fix is not None
        rec = Record(idx=i, t=float(i), source=f"f{i}.png",
                     track=TrackState(i, "TRACKING", fix.x, fix.y, 0.0, 0.0, np.eye(2), fix=fix))
        record_to_dict(rec)   # raises on any non-finite number


# --- the checks, one failure each -----------------------------------------------------------------------------

def test_check_accepts_a_good_homography():
    assert check_homography(make_h(45.0, 0.25), FRAME, **GOOD) is None


@pytest.mark.parametrize("H", [
    np.zeros((3, 3)),                                                     # nothing at all
    np.array([[1, 0, 0], [2, 0, 0], [0, 0, 1.0]]),                        # rank 2: the frame collapses to a line
    np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]]),                     # broken numbers
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, np.inf]]),
    make_h(0.0, 1e-9),                                                     # the frame shrinks to a point
], ids=["zero", "rank2", "nan", "inf", "point"])
def test_check_degenerate(H):
    assert check_homography(H, FRAME, **GOOD) == "degenerate"


def test_check_too_few_inliers():
    assert check_homography(make_h(), FRAME, inliers=5, reproj_rmse=0.1,
                            cfg=GeometryConfig(min_inliers=15)) == "too_few_inliers"


def test_check_not_convex_when_the_horizon_crosses_the_frame():
    # w = 0.01 x - 1 is positive at the centre (x = 320) but negative at the left edge: the plane's horizon runs
    # through the frame, so its footprint is not a convex quadrilateral (part of it maps "behind the camera")
    H = np.array([[1, 0, 0], [0, 1, 0], [0.01, 0, -1.0]])
    assert check_homography(H, FRAME, **GOOD) == "not_convex"


def test_check_not_convex_for_a_twisted_footprint():
    # corners sent to a bow-tie (TR and BR swapped): no camera can see a plane like that
    import cv2
    dst = np.float32([[0, 0], [W, H_], [W, 0], [0, H_]]) + 500
    H = cv2.getPerspectiveTransform(CORNERS.astype(np.float32), dst).astype(float)
    assert check_homography(H, FRAME, **GOOD) == "not_convex"


@pytest.mark.parametrize("heading", [0.0, 120.0, 250.0])
def test_check_mirrored(heading):
    assert check_homography(make_h(heading, 0.5, mirror=True), FRAME, **GOOD) == "mirrored"


def test_check_negated_h_is_the_same_homography():
    # H and -H are the same projective map; the sign must not turn a good fix into a rejected one
    assert check_homography(-make_h(30.0, 0.5), FRAME, **GOOD) is None


@pytest.mark.parametrize("scale", [0.001, 500.0])
def test_check_scale_out_of_range(scale):
    cfg = GeometryConfig(min_scale=0.01, max_scale=100.0)
    assert check_homography(make_h(0.0, scale), FRAME, **GOOD, cfg=cfg) == "scale_out_of_range"


def test_check_aspect():
    # one axis stretched 4x relative to the other: not a camera looking down at a flat map
    assert check_homography(make_h(30.0, 0.5, sy=4.0), FRAME, **GOOD, cfg=GeometryConfig(max_aspect=2.0)) == "aspect"
    assert check_homography(make_h(30.0, 0.5, sy=1.5), FRAME, **GOOD, cfg=GeometryConfig(max_aspect=2.0)) is None


def test_check_reproj_error():
    assert check_homography(make_h(), FRAME, inliers=200, reproj_rmse=5.0,
                            cfg=GeometryConfig(max_reproj_rmse_px=2.0)) == "reproj_error"


def test_mild_perspective_is_accepted():
    # a phone tilted a little: corners shift by a few percent, still convex, not mirrored, mild aspect
    src = CORNERS.astype(np.float32)
    dst = np.float32([[10, 0], [W - 10, 0], [W + 20, H_], [-20, H_]]) * 0.5 + 300
    import cv2
    H = cv2.getPerspectiveTransform(src, dst).astype(float)
    assert check_homography(H, FRAME, **GOOD) is None
    p = pose_from_homography(H, FRAME)
    assert angdiff(p.heading_deg, 0.0) < 1.0 and 0.4 < p.scale < 0.6
