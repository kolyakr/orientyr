"""The shared ``Matcher`` contract: what every backend (RootSIFT, XFeat, LightGlue) must do.

A later backend joins by adding a factory to ``MATCHERS``. A learned backend needs its weights, which are neither
committed nor downloaded by tests: it gets a mark in ``MARKS`` and skips where the weights are not cached (CI). The contract (dronenav.types.Matcher):
- it is a ``Matcher`` and returns ``Matches`` with its backend name and a wall time;
- ``pts_map`` are in full-map pixels, also when ``map_img`` is a window at ``map_offset``;
- matches from a window lie inside that window, and ``mask`` keeps them out of masked-off map pixels;
- a frame with nothing to match (blank) gives zero matches, not an error.

Ground truth: a synthetic map and a frame rendered from it through a known homography (tests/synth.py).
"""
import numpy as np
import pytest
from tests.synth import apply, make_h, make_map, render_frame
from tests.weights import needs_xfeat_weights

from dronenav.geometry import localize
from dronenav.matching import RootSiftMatcher, XFeatMatcher
from dronenav.matching.lightglue import LightGlueMatcher, weights_available
from dronenav.types import Matcher, Matches

FRAME_SIZE = (480, 360)
MAP = make_map()
# 35 deg turn and a zoom (scale 0.8: the frame a little more detailed than the map). Stronger zooms, like the
# task crops (0.25), need the frame downscaled first: that is M1-03, not the matcher
H_TRUE = make_h(35.0, 0.8, cx=640.0, cy=420.0, frame_size=FRAME_SIZE)
FRAME = render_frame(MAP, H_TRUE, FRAME_SIZE)
WINDOW = (380, 160, 900, 680)   # x0, y0, x1, y1 of a map window that contains the whole footprint

MATCHERS = {
    "rootsift": lambda cache_dir: RootSiftMatcher(cache_dir=cache_dir),
    # CPU and no download: the contract is the same on every device, and a test never reaches for the network
    "xfeat": lambda cache_dir: XFeatMatcher(cache_dir=cache_dir, device="cpu", download=False),
    "lightglue": lambda cache_dir: LightGlueMatcher(cache_dir=cache_dir, device="cpu"),
}
# a backend whose weights are not on disk (CI is offline) skips its run of the contract, visibly
MARKS = {"xfeat": needs_xfeat_weights,
         "lightglue": pytest.mark.skipif(not weights_available("disk"), reason="DISK/LightGlue weights missing")}


@pytest.fixture(params=[pytest.param(name, marks=MARKS.get(name, ())) for name in MATCHERS])
def matcher(request, tmp_path):
    return MATCHERS[request.param](tmp_path / "cache")


def _good_fraction(m: Matches, tol_px: float = 3.0) -> float:
    """Share of matches whose map point is within tol_px of where the true homography sends the frame point."""
    err = np.linalg.norm(apply(H_TRUE, m.pts_frame) - m.pts_map, axis=1)
    return float(np.mean(err < tol_px))


def test_is_a_matcher_and_returns_matches(matcher):
    assert isinstance(matcher, Matcher)
    m = matcher.match(FRAME, MAP)
    assert isinstance(m, Matches)
    assert isinstance(m.backend, str) and m.backend
    assert m.ms >= 0.0


def test_full_map_matches_are_correct(matcher):
    m = matcher.match(FRAME, MAP)
    # a textured synthetic scene gives hundreds of matches; most agree with the truth to a few px (the rest
    # are left for MAGSAC to throw out), and the fitted pose lands on the true centre
    assert len(m) >= 50
    assert _good_fraction(m) > 0.5
    fix = localize(0, m, FRAME_SIZE)
    assert fix is not None and fix.accepted
    assert np.hypot(fix.x - 640.0, fix.y - 420.0) < 1.0   # sub-pixel truth; 1 px allows detector jitter


def test_window_matches_are_in_full_map_coordinates(matcher):
    x0, y0, x1, y1 = WINDOW
    m = matcher.match(FRAME, MAP[y0:y1, x0:x1], map_offset=(x0, y0))
    assert len(m) >= 50
    # offsets forgotten would put every point (420, 200) px off: the truth check catches it
    assert _good_fraction(m) > 0.5
    assert (m.pts_map[:, 0] >= x0).all() and (m.pts_map[:, 0] < x1).all()
    assert (m.pts_map[:, 1] >= y0).all() and (m.pts_map[:, 1] < y1).all()
    fix = localize(0, m, FRAME_SIZE)
    assert fix is not None and fix.accepted and np.hypot(fix.x - 640.0, fix.y - 420.0) < 1.0


def test_mask_excludes_map_pixels(matcher):
    x0, y0, x1, y1 = WINDOW
    win = MAP[y0:y1, x0:x1]
    mask = np.zeros(win.shape[:2], np.uint8)
    mask[:, : win.shape[1] // 2] = 255          # only the left half of the window may be used
    m = matcher.match(FRAME, win, map_offset=(x0, y0), mask=mask)
    assert len(m) > 0
    assert (m.pts_map[:, 0] < x0 + win.shape[1] // 2 + 1).all()


def test_blank_frame_gives_no_matches(matcher):
    blank = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), 128, np.uint8)
    m = matcher.match(blank, MAP)
    assert isinstance(m, Matches) and len(m) == 0


def test_blank_map_window_gives_no_matches(matcher):
    blank = np.full((300, 300, 3), 90, np.uint8)
    m = matcher.match(FRAME, blank, map_offset=(5000, 5000))
    assert isinstance(m, Matches) and len(m) == 0


def test_grayscale_input_is_accepted(matcher):
    import cv2
    m = matcher.match(cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY), cv2.cvtColor(MAP, cv2.COLOR_BGR2GRAY))
    assert _good_fraction(m) > 0.5
