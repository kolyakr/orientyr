"""M4-02 · LightGlue backend.

"What: a Matcher with SuperPoint or DISK features and LightGlue matching, on MPS with CPU fallback; keypoint budget
as a parameter; map features read from the pack when present. Done when: it passes the shared Matcher contract test
and localizes the 50 task frames and the mild 2025 Irpin keyframes."

So:
- the shared contract: tests/test_matcher_contract.py runs the "lightglue" entry (when the weights are on disk);
- DISK and LightGlue are not rotation-invariant (a frame turned 180° from the map gets ~15 matches instead of
  ~400), so the matcher tries the frame turned in 45° steps and maps the winner's points back: frames localize
  at any heading. Quarter turns alone leave a frame up to 45° off, where task crops fail (measured: ~300 inliers
  aligned, often under 10 at 45°). The turn arithmetic is checked exactly, without weights;
- the device: picked like every torch model here (dronenav.device), or forced;
- the keypoint budget bounds the frame's keypoints (and so the matches);
- map features: extracted once, cached under a content key, and read from a pack folder when one holds them
  (M4-06 writes the pack; this backend reads it). Neither a cache hit nor a pack read extracts anything;
- the real data (``@pytest.mark.data``): the 50 task crops and every keyframe of the mild 2025 Irpin flight,
  through the same localize_run the deliverable uses, all localized and none wrong.

CI is offline and has no weights: tests that run the network skip cleanly there (``needs_weights``).
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.synth import make_h, make_map, render_frame

from dronenav.geometry import localize
from dronenav.matching.lightglue import (
    LightGlueConfig,
    LightGlueMatcher,
    auto_upsample,
    padded_size,
    quarter_turn_back,
    turn_back,
    turn_image,
    weights_available,
)
from dronenav.search import Searcher

ROOT = Path(__file__).resolve().parents[1]
needs_weights = pytest.mark.skipif(not weights_available("disk"), reason="DISK/LightGlue weights not downloaded")

FRAME_SIZE = (480, 360)
MAP = make_map()


@pytest.fixture(autouse=True)
def _unit_tests_on_cpu(request, monkeypatch):
    """Unit tests feed DISK many image sizes; on MPS each new size compiles a graph (~1 s, measured), which made
    this file take an hour. CPU costs the same for any size. Data tests keep the real device, as the run would."""
    if request.node.get_closest_marker("data") is None:
        import dronenav.matching.lightglue as lg
        auto = lg.pick_device
        monkeypatch.setattr(lg, "pick_device", lambda d=None: "cpu" if d in (None, "auto") else auto(d))


# --- quarter turns: exact, no network -------------------------------------------------------------------------

@pytest.mark.parametrize("q", [0, 1, 2, 3, 4, -1])
def test_quarter_turn_back_undoes_np_rot90(q):
    """A pixel of np.rot90(img, q) at (x', y') shows the pixel of img at quarter_turn_back((x', y')). Checked on
    an image whose pixels encode their own coordinates, so every pixel is a test point."""
    h, w = 5, 7
    ys, xs = np.mgrid[0:h, 0:w]
    img = np.stack([xs, ys], axis=-1)                       # img[y, x] = (x, y)
    turned = np.rot90(img, q)
    th, tw = turned.shape[:2]
    pts = np.array([(x, y) for y in range(th) for x in range(tw)], float)
    back = quarter_turn_back(pts, q, (w, h))
    expect = np.array([turned[int(y), int(x)] for x, y in pts], float)
    np.testing.assert_array_equal(back, expect)


@pytest.mark.parametrize("deg", [45.0, 135.0, -45.0, 30.0])
def test_turn_back_undoes_any_turn(deg):
    """turn_image turns a picture by any angle onto a canvas that holds all of it; turn_back sends points of the
    turned picture to the original. A point and its turned image must show the same pixel."""
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 255, (60, 90), dtype=np.uint8), (0, 0), 3)
    turned, A = turn_image(img, deg)
    pts = np.array([[45.0, 30.0], [20.0, 10.0], [70.0, 50.0]])
    there = (A[:, :2] @ pts.T).T + A[:, 2]                         # original -> turned
    assert (there >= 0).all() and (there[:, 0] < turned.shape[1]).all() and (there[:, 1] < turned.shape[0]).all()
    np.testing.assert_allclose(turn_back(there, A), pts, atol=1e-9)
    # the same content: bilinear values at the turned points match the original pixels to a few gray levels
    vals = [cv2.getRectSubPix(turned, (1, 1), tuple(p))[0, 0] for p in there.astype(np.float32)]
    np.testing.assert_allclose(vals, [img[int(y), int(x)] for x, y in pts], atol=6)


def test_quarter_turns_are_exact():
    """Multiples of 90° are np.rot90: no resampling, and turn_back agrees with quarter_turn_back."""
    img = np.arange(35, dtype=np.uint8).reshape(5, 7)
    for q in range(4):
        turned, A = turn_image(img, 90.0 * q)
        np.testing.assert_array_equal(turned, np.rot90(img, q))
        pts = np.array([[0.0, 0.0], [3.0, 1.0], [turned.shape[1] - 1.0, turned.shape[0] - 1.0]])
        np.testing.assert_allclose(turn_back(pts, A), quarter_turn_back(pts, q, (7, 5)), atol=1e-12)


def test_quarter_turn_back_works_between_pixels():
    """Sub-pixel points move rigidly too: the centre of a 7 x 5 image stays its centre under every turn."""
    for q in range(4):
        turned_size = (7, 5) if q % 2 == 0 else (5, 7)
        centre = np.array([[(turned_size[0] - 1) / 2, (turned_size[1] - 1) / 2]])
        np.testing.assert_allclose(quarter_turn_back(centre, q, (7, 5)), [[3.0, 2.0]])


# --- configuration --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {"features": "orb"}, {"max_keypoints": 0}, {"map_keypoints": 0}, {"tile_px": 16},
    {"turns": 0}, {"turns": 3}, {"turns": 5}, {"turns": 16}, {"window_keypoints": 0},
    {"upsample": 0.5}, {"upsample": 0.0}, {"upsample": 5.0}, {"upsample_budget_mp": 0.0},
])
def test_config_is_checked(bad):
    with pytest.raises(ValueError):
        LightGlueConfig(**bad)


# the two real maps: the task map (3645 x 1845, 6.7 MP) and an Irpin map (3936 x 3936, 15.5 MP)
@pytest.mark.parametrize("shape, expected", [((1845, 3645, 3), 2.0), ((3936, 3936, 3), 1.0), ((10, 10), 2.0)])
def test_upsampling_is_automatic_within_the_pixel_budget(shape, expected):
    # 2x only while the enlarged map stays within the budget (32 MP): the low-resolution task map needs it (frames
    # 21 and 35 go from ~12 inliers to 150+), the Irpin maps already match at 1x and would cost 4x to extract
    assert auto_upsample(shape, 32.0) == expected


@pytest.mark.parametrize("hw, expected", [((360, 480), (384, 512)), ((512, 512), (512, 512)), ((1, 1), (128, 128)),
                                          ((513, 700), (640, 768))])
def test_images_are_padded_to_a_few_sizes(hw, expected):
    # torch's MPS backend compiles a graph per input size: DISK takes 0.24 s on a repeated size, 1.12 s on a new
    # one (M4-02 notes). Padding to multiples of 128 px leaves few sizes, so the compiled graphs are reused.
    assert padded_size(*hw) == expected


@needs_weights
def test_no_keypoint_lands_in_the_padding():
    # 300 x 200 pads to 384 x 256: everything past the picture is black, and no keypoint may come from it
    patch = np.ascontiguousarray(MAP[100:300, 100:400])
    f = LightGlueMatcher(cache_dir=None, device="cpu", cfg=LightGlueConfig(upsample=1.0))._extract(patch, 2048)
    assert len(f) > 50
    assert f.pts[:, 0].max() < 300 and f.pts[:, 1].max() < 200 and f.pts.min() >= 0


def test_an_explicit_upsample_wins_over_the_budget():
    assert LightGlueConfig(upsample=1.0).upsample == 1.0 and LightGlueConfig().upsample is None


def test_unknown_device_is_an_error():
    with pytest.raises(ValueError):
        LightGlueMatcher(device="tpu", cache_dir=None)


# --- with the weights -----------------------------------------------------------------------------------------

@needs_weights
def test_the_device_is_picked_and_can_be_forced():
    assert LightGlueMatcher(cache_dir=None).device in ("mps", "cuda", "cpu")
    assert LightGlueMatcher(cache_dir=None, device="cpu").device == "cpu"


@pytest.fixture(scope="module")
def map_cache(tmp_path_factory):
    """One cache folder for the tests that only match: MAP's features (2x: a 2400 x 1800 image) are extracted
    once for the file instead of once per test. Tests that count extractions keep their own tmp_path."""
    return tmp_path_factory.mktemp("lightglue_map_cache")


@needs_weights
# 45°, 135° and 315° lie halfway between quarter turns: the hardest headings for quarter turns alone
@pytest.mark.parametrize("heading", [0.0, 45.0, 90.0, 135.0, 150.0, 200.0, 290.0, 315.0])
def test_a_frame_localizes_at_any_heading(heading, map_cache):
    """Without the quarter turns, a frame turned 150° or 200° from the map finds next to nothing."""
    cx, cy = 600.0, 430.0
    frame = render_frame(MAP, make_h(heading, 0.8, cx=cx, cy=cy, frame_size=FRAME_SIZE), FRAME_SIZE)
    m = LightGlueMatcher(MAP, cache_dir=map_cache).match(frame, MAP)
    fix = localize(0, m, FRAME_SIZE)
    assert fix is not None and fix.accepted, heading
    # an exact synthetic homography: 1.5 px allows DISK's coarser keypoints (it localizes on a 4 px grid
    # refined by a soft-argmax, not SIFT's sub-pixel fit)
    assert math.hypot(fix.x - cx, fix.y - cy) < 1.5, (heading, fix.x, fix.y)
    assert abs((fix.heading_deg - heading + 180) % 360 - 180) < 1.0


@needs_weights
def test_the_keypoint_budget_bounds_the_frame(map_cache):
    frame = render_frame(MAP, make_h(20.0, 0.8, cx=600, cy=430, frame_size=FRAME_SIZE), FRAME_SIZE)
    small = LightGlueMatcher(MAP, cache_dir=map_cache, cfg=LightGlueConfig(max_keypoints=128)).match(frame, MAP)
    big = LightGlueMatcher(MAP, cache_dir=map_cache, cfg=LightGlueConfig(max_keypoints=2048)).match(frame, MAP)
    assert 0 < len(small) <= 128
    assert len(big) > len(small)


@needs_weights
def test_map_features_are_cached_and_reused(tmp_path):
    first = LightGlueMatcher(MAP, cache_dir=tmp_path)
    assert first.map_extractions == 1 and first.map_features_source == "extracted"
    assert list(tmp_path.glob("lightglue_*.npz"))
    again = LightGlueMatcher(MAP, cache_dir=tmp_path)
    assert again.map_extractions == 0 and again.map_features_source == "cache"
    # a different budget is a different extraction: never a stale hit
    other = LightGlueMatcher(MAP, cache_dir=tmp_path, cfg=LightGlueConfig(map_keypoints=512))
    assert other.map_extractions == 1
    # so is another upsampling: its keypoints come from a finer image
    finer = LightGlueMatcher(MAP, cache_dir=tmp_path, cfg=LightGlueConfig(upsample=1.5))
    assert finer.map_extractions == 1


@needs_weights
def test_map_features_are_read_from_the_pack_when_present(tmp_path):
    pack = tmp_path / "pack"
    LightGlueMatcher(MAP, cache_dir=pack)                     # the pack builder (M4-06) writes the same file
    m = LightGlueMatcher(MAP, cache_dir=None, pack_dir=pack)
    assert m.map_extractions == 0 and m.map_features_source == "pack"
    # a pack without this backend's features: computed on the fly, the pack itself is left untouched
    empty = tmp_path / "empty_pack"
    empty.mkdir()
    m = LightGlueMatcher(MAP, cache_dir=None, pack_dir=empty)
    assert m.map_extractions == 1 and not list(empty.iterdir())


@needs_weights
def test_the_matcher_reports_the_upsampling_it_chose(map_cache):
    assert LightGlueMatcher(MAP, cache_dir=map_cache).upsample == 2.0                # a small map: 2x
    assert LightGlueMatcher(MAP, cache_dir=map_cache, cfg=LightGlueConfig(upsample=1.0)).upsample == 1.0


@needs_weights
def test_windows_reuse_the_map_features(tmp_path):
    m = LightGlueMatcher(MAP, cache_dir=tmp_path)
    frame = render_frame(MAP, make_h(35.0, 0.8, cx=640, cy=420, frame_size=FRAME_SIZE), FRAME_SIZE)
    m.match(frame, MAP[160:680, 380:900], map_offset=(380, 160))
    assert m.map_extractions == 1                             # the registration only


@needs_weights
def test_the_irpin_fixture_keyframes_localize(irpin_mild, tmp_path):
    """Real imagery in CI-sized form: each fixture keyframe found on its 512 px crop, through the Searcher."""
    crop = irpin_mild.crop
    for k, frame in enumerate(irpin_mild.frames):
        fix = Searcher(LightGlueMatcher(crop, cache_dir=tmp_path), crop).locate(0, frame).fix
        truth = irpin_mild.to_crop([irpin_mild.truth[k]["center_map_px"]])[0]
        assert fix is not None and fix.accepted, (k, fix and fix.reject_reason)
        # the truth acts on undistorted frames; the fix uses the raw one: a few px of lens distortion at most
        assert math.hypot(fix.x - truth[0], fix.y - truth[1]) < 10.0, k


# --- the real data, locally -----------------------------------------------------------------------------------

DATA = ROOT / "data"
PROBE = ROOT / "experiments" / "data_probe" / "results" / "probe.json"


@pytest.mark.data
@needs_weights
def test_the_50_task_frames_localize(tmp_path):
    from dronenav.accuracy import load_truth, score
    from dronenav.pipeline import localize_run
    from dronenav.telemetry import read_run

    task = DATA / "hackatons-data"
    if not (task / "global_map.png").is_file() or not PROBE.is_file():
        pytest.skip("needs data/hackatons-data and probe.json")
    localize_run(task / "global_map.png", task / "crops", tmp_path, matcher="lightglue", fps=1,
                 cache_dir=ROOT / "outputs" / "cache")
    acc = score(read_run(tmp_path / "run.jsonl"), load_truth(PROBE))
    assert (acc.n, acc.localized, acc.wrong) == (50, 50, 0)


@pytest.mark.data
@needs_weights
def test_the_mild_2025_irpin_keyframes_localize(tmp_path):
    from dronenav.accuracy import load_truth, score
    from dronenav.pipeline import localize_run
    from dronenav.telemetry import read_run

    flight = DATA / "irpin" / "flights" / "mild" / "2025"
    the_map = DATA / "irpin" / "maps" / "irpin_2025.png"
    if not (flight / "truth.jsonl").is_file() or not the_map.is_file():
        pytest.skip("needs data/irpin (mild 2025 flight and map)")
    localize_run(the_map, flight / "frames", tmp_path, matcher="lightglue", fps=30,
                 cache_dir=ROOT / "outputs" / "cache")
    acc = score(read_run(tmp_path / "run.jsonl"), load_truth(flight))
    # every keyframe: the flight turns through all headings and crosses the whole map
    assert acc.n == len(list((flight / "frames").glob("frame_*.png")))
    assert acc.localized == acc.n and acc.wrong == 0


@needs_weights
def test_upsampling_keeps_points_in_input_pixels(map_cache):
    # the enlarged image is an internal detail: a fix from an upsampling matcher lands where the plain one does
    cx, cy = 600.0, 430.0
    frame = render_frame(MAP, make_h(30.0, 0.8, cx=cx, cy=cy, frame_size=FRAME_SIZE), FRAME_SIZE)
    m = LightGlueMatcher(MAP, cache_dir=map_cache, cfg=LightGlueConfig(upsample=2.0)).match(frame, MAP)
    assert m.pts_frame.max(axis=0)[0] <= FRAME_SIZE[0] and m.pts_frame.max(axis=0)[1] <= FRAME_SIZE[1]
    fix = localize(0, m, FRAME_SIZE)
    assert fix is not None and fix.accepted
    assert math.hypot(fix.x - cx, fix.y - cy) < 1.5                # the same 1.5 px as at any heading above


def test_the_pipeline_offers_it():
    """`scripts/run.py --matcher lightglue` picks this backend."""
    from dronenav.pipeline import MATCHERS

    assert MATCHERS["lightglue"][0] == LightGlueMatcher.backend == "lightglue"


def test_the_pipeline_hands_it_the_device():
    """`run.py --device cpu` reaches the matcher (constructing it loads no weights: no network needed)."""
    from dronenav.pipeline import MATCHERS

    assert MATCHERS["lightglue"][1](MAP, None, "cpu").device == "cpu"


@needs_weights
@pytest.mark.parametrize("up", [1.0, 2.0])
def test_matched_points_follow_the_corner_convention(up):
    # M1-12 (#106): matchers return corner-origin pixels (pixel centres at +0.5), like RootSIFT. A frame that is a
    # map patch turned 180° exactly (np.rot90, no resampling) has its point (x, y) at map (x0 + w - x, y0 + h - y);
    # OpenCV's centre convention would put every match 1 px off on each axis. DISK keypoints sit on whole pixels,
    # so on this exact copy the right convention gives ~0 px and the wrong one ~1.4 px: 0.25 px separates them.
    x0, y0, w, h = 300, 200, 320, 240
    frame = np.ascontiguousarray(np.rot90(MAP[y0:y0 + h, x0:x0 + w], 2))
    # Upsampled (up = 2), keypoints sit on whole pixels of the enlarged image, i.e. on half pixels of the original;
    # mapping them back without the half-pixel shift would put both sides 0.25 px off, ~0.7 px after the 180° turn.
    cfg = LightGlueConfig(upsample=up)
    m = LightGlueMatcher(MAP, cache_dir=None, device="cpu", cfg=cfg).match(frame, MAP)
    assert len(m) >= 50
    expected = np.column_stack([x0 + w - m.pts_frame[:, 0], y0 + h - m.pts_frame[:, 1]])
    assert np.median(np.linalg.norm(m.pts_map - expected, axis=1)) < 0.25
