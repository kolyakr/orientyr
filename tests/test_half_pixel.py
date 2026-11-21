"""M1-12 · Half-pixel convention: matcher points in corner-origin pixels (issue #106).

dronenav.types puts the origin at the top-left corner of the top-left pixel, so a pixel's centre is at +0.5.
OpenCV keypoints put pixel centres at integers. A matcher that hands OpenCV's coordinates on unshifted gives
fixes off by up to ~0.5 px at 180° and more at larger scales: the frame side's missing 0.5 is turned and
scaled by the homography, the map side's is not.

"Done when: the fix centre of exact cut-outs is within 0.1 px of the truth at headings 0, 90, 180° and scales
0.5 to 2." Truth comes from dronenav.sim.cutout, which renders in the corner convention (checked to 0.04 px in
M6-01). The 0.1 px tolerance: the measured bias is 0.24 to 0.5 px at these poses, and an unbiased RootSIFT fix
on this textured map is within ~0.02 px, so 0.1 px separates the two with room for detector jitter.
"""
import cv2
import numpy as np
import pytest

from dronenav.matching import RootSiftMatcher
from dronenav.scale import localize_scaled
from dronenav.sim import CutoutPose, cutout_homography, render_cutout

FRAME_SIZE = (320, 240)
CENTRE = (1000.0, 750.0)


def _map() -> np.ndarray:
    """A 2000 x 1500 map of random rotated boxes: corners everywhere, no repeats (the issue's reproduction)."""
    rng = np.random.default_rng(3)
    img = np.full((1500, 2000), 90, np.uint8)
    for _ in range(3300):
        box = cv2.boxPoints(((float(rng.uniform(0, 2000)), float(rng.uniform(0, 1500))),
                             (float(rng.uniform(6, 40)), float(rng.uniform(6, 40))), float(rng.uniform(0, 180))))
        cv2.fillConvexPoly(img, box.astype(np.int32), int(rng.integers(0, 256)))
    return cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 0.8), cv2.COLOR_GRAY2BGR)


MAP = _map()


@pytest.fixture(scope="module")
def matcher():
    return RootSiftMatcher(MAP, cache_dir=None)


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("heading", [0.0, 90.0, 180.0])
def test_fix_centre_of_an_exact_cutout_is_unbiased(matcher, heading, scale):
    H = cutout_homography(CutoutPose(*CENTRE, heading, scale), FRAME_SIZE)
    fix = localize_scaled(0, matcher, render_cutout(MAP, H, FRAME_SIZE), MAP, (1.0,)).fix
    assert fix is not None and fix.accepted
    assert np.hypot(fix.x - CENTRE[0], fix.y - CENTRE[1]) < 0.1


def test_matched_points_follow_the_corner_convention(matcher):
    # a frame that is the map patch turned 180° exactly (np.rot90, no resampling): in corner-origin pixels the
    # frame point (x, y) is the map point (x0 + w - x, y0 + h - y). In OpenCV's centre convention it would be
    # (x0 + w - 1 - x, ...): every match 1 px off on each axis, so a 0.1 px median catches it
    x0, y0, w, h = 840, 630, *FRAME_SIZE
    frame = np.ascontiguousarray(np.rot90(MAP[y0:y0 + h, x0:x0 + w], 2))
    m = matcher.match(frame, MAP)
    assert len(m) >= 50
    expected = np.column_stack([x0 + w - m.pts_frame[:, 0], y0 + h - m.pts_frame[:, 1]])
    err = np.linalg.norm(m.pts_map - expected, axis=1)
    assert np.median(err) < 0.1

