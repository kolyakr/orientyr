"""M4-05 · Frame preprocessing: the contract of dronenav.preprocess.

"CLAHE on luminance (both map and frame, same settings); scale normalisation so a frame pixel covers about one
map pixel (from the scale prior or the last fix); optional grey-world colour balance. One function used by every
backend. Done when: an ablation on the go/no-go set shows the effect of each step." So:

- one function, ``preprocess``, takes the map or a frame and a config that switches each step on or off; with every
  step off it changes nothing; it never changes its input; the same input gives the same output;
- scale normalisation shrinks a frame by the M1-03 rule (dronenav.scale) from a given scale, never enlarges it,
  and points found on the shrunk frame map back to the original frame's pixels exactly;
- the scale prior is the frame's ground resolution over the map's (``nadir_gsd_m`` / ``m_per_px``);
- CLAHE works on luminance: it raises the contrast of a flat image, brings two exposures of one scene closer, and
  gives a grey image and its BGR copy the same result;
- grey-world removes a colour cast and leaves a neutral image alone;
- empty, wrongly shaped or non-uint8 images and bad scales are refused with a ValueError; a blank frame is an
  ordinary frame (it comes out blank and gives no matches);
- a dim, tinted frame still localises on the map after both are preprocessed (exact synthetic truth);
- the ablation script runs on a tiny Irpin-shaped dataset and prints one row per step.

Ground truth: the synthetic ground and map of tests/synth.py, as in test_scale.py: the frame is 4x more detailed
than the map (scale 0.25).
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.synth import make_h, make_map, render_frame

from dronenav.geometry import localize
from dronenav.matching import RootSiftMatcher
from dronenav.preprocess import Prepared, PreprocessConfig, preprocess, scale_prior
from dronenav.scale import ScaleConfig, downscale_factor
from dronenav.types import Matches

ROOT = Path(__file__).resolve().parents[1]
FRAME_SIZE = (640, 480)
GROUND = make_map(1600, 1200, seed=5, shapes=1600)                     # the density of the default synthetic map
MAP = cv2.resize(GROUND, (800, 600), interpolation=cv2.INTER_AREA)     # 1 map px = 2 ground px
TRUE_SCALE = 0.25                                                      # map px per frame px
CX, CY = 430.0, 290.0                                                  # the frame centre on the map
FRAME = render_frame(GROUND, make_h(30.0, 0.5, cx=2 * CX, cy=2 * CY, frame_size=FRAME_SIZE), FRAME_SIZE)

OFF = PreprocessConfig(clahe=False, scale_norm=False, grey_world=False)
CLAHE = PreprocessConfig(clahe=True, scale_norm=False, grey_world=False)
SCALE = PreprocessConfig(clahe=False, scale_norm=True, grey_world=False)
GREY_WORLD = PreprocessConfig(clahe=False, scale_norm=False, grey_world=True)
ALL = PreprocessConfig(clahe=True, scale_norm=True, grey_world=True)


def _grey(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)


def _exposure(img: np.ndarray, gain: float, offset: float) -> np.ndarray:
    return np.clip(img.astype(np.float64) * gain + offset, 0, 255).round().astype(np.uint8)


def _tint(img: np.ndarray, bgr_gains: tuple[float, float, float]) -> np.ndarray:
    return np.clip(img.astype(np.float64) * np.array(bgr_gains), 0, 255).round().astype(np.uint8)


# --- one function, steps that switch off ------------------------------------------------------------------------

def test_with_every_step_off_the_image_is_unchanged():
    out = preprocess(FRAME, OFF, scale=TRUE_SCALE)
    assert isinstance(out, Prepared)
    assert np.array_equal(out.image, FRAME)
    assert out.factor == 1.0 and out.size == FRAME_SIZE


@pytest.mark.parametrize("cfg", [CLAHE, SCALE, GREY_WORLD, ALL])
def test_the_input_is_never_changed_and_the_output_is_a_uint8_image_of_the_same_kind(cfg):
    before = FRAME.copy()
    out = preprocess(FRAME, cfg, scale=TRUE_SCALE)
    assert np.array_equal(FRAME, before)
    assert out.image.dtype == np.uint8 and out.image.ndim == 3 and out.image.shape[2] == 3
    grey = preprocess(FRAME[:, :, 0].copy(), cfg, scale=TRUE_SCALE)
    assert grey.image.dtype == np.uint8 and grey.image.ndim == 2


def test_the_same_input_gives_the_same_output():
    a, b = preprocess(FRAME, ALL, scale=TRUE_SCALE), preprocess(FRAME, ALL, scale=TRUE_SCALE)
    assert np.array_equal(a.image, b.image) and a.factor == b.factor


def test_the_map_goes_through_the_same_function_and_keeps_its_size():
    # the map defines the coordinates: no scale is given for it, so it is never resized
    out = preprocess(MAP, ALL)
    assert out.image.shape == MAP.shape and out.factor == 1.0
    assert not np.array_equal(out.image, MAP)             # CLAHE and grey-world did act on it


# --- scale normalisation ----------------------------------------------------------------------------------------

def test_a_frame_is_shrunk_by_the_scale_rule():
    # the M1-03 rule (dronenav.scale): factor = scale / target_scale; 0.325 / 0.65 is exactly one half
    out = preprocess(FRAME, SCALE, scale=0.325)
    assert out.factor == pytest.approx(0.5)
    assert out.image.shape[:2] == (240, 320)
    assert out.size == FRAME_SIZE                         # the original (w, h), kept for mapping points back


def test_the_target_scale_comes_from_the_config():
    # "a frame pixel covers about one map pixel": target_scale = 1 is that, literally
    cfg = PreprocessConfig(clahe=False, scale_norm=True, grey_world=False, scale=ScaleConfig(target_scale=1.0))
    out = preprocess(FRAME, cfg, scale=TRUE_SCALE)
    assert out.factor == pytest.approx(0.25) and out.image.shape[:2] == (120, 160)
    assert preprocess(FRAME, SCALE, scale=TRUE_SCALE).factor == pytest.approx(downscale_factor(TRUE_SCALE))


def test_a_coarse_frame_is_never_enlarged():
    out = preprocess(FRAME, SCALE, scale=3.0)
    assert out.factor == 1.0 and np.array_equal(out.image, FRAME)


def test_without_a_scale_the_frame_keeps_its_size():
    # no prior and no fix yet: nothing to normalise by (the caller probes, see dronenav.scale)
    out = preprocess(FRAME, SCALE)
    assert out.factor == 1.0 and np.array_equal(out.image, FRAME)


def test_a_given_scale_is_ignored_when_the_step_is_off():
    assert preprocess(FRAME, CLAHE, scale=TRUE_SCALE).image.shape == FRAME.shape


def test_points_on_the_shrunk_frame_map_back_to_the_original_frame():
    img = np.zeros((300, 400, 3), np.uint8)
    img[120:128, 200:208] = 255                           # a white square: its centre is at (203.5, 123.5)
    out = preprocess(img, SCALE, scale=0.325)             # factor 0.5
    ys, xs = np.nonzero(out.image[:, :, 0])
    back = out.to_original(np.array([[xs.mean(), ys.mean()]]))
    # an exact 2x area average: the square is 4 x 4 px there and its centre maps back exactly. 0.05 px leaves room
    # for float error only; forgetting the half-pixel term of the pixel-centre convention would be 0.5 px off
    assert np.allclose(back, [[203.5, 123.5]], atol=0.05)


def test_mapping_back_uses_the_real_size_ratio_per_axis():
    # 101 x 77 at factor 0.5 rounds to a size that is not exactly half: the ratio of the two sizes is what counts
    out = preprocess(np.zeros((77, 101, 3), np.uint8), SCALE, scale=0.325)
    h, w = out.image.shape[:2]
    pts = np.array([[0.0, 0.0], [w - 1.0, h - 1.0], [10.0, 20.0]])
    want = (pts + 0.5) * np.array([101 / w, 77 / h]) - 0.5      # pixel centres, OpenCV's resize convention
    assert np.allclose(out.to_original(pts), want, atol=1e-9)
    assert out.to_original(np.zeros((0, 2))).shape == (0, 2)


def test_unshrunk_points_stay_where_they_are():
    pts = np.array([[3.0, 4.0], [100.5, 7.25]])
    assert np.array_equal(preprocess(FRAME, CLAHE).to_original(pts), pts)


def test_restore_puts_matches_into_original_frame_pixels():
    out = preprocess(FRAME, SCALE, scale=0.325)
    m = Matches(np.array([[10.0, 20.0]]), np.array([[500.0, 300.0]]), np.array([0.7]), "x", 3.0)
    r = out.restore(m)
    assert np.allclose(r.pts_frame, [[20.5, 40.5]])       # (p + 0.5) * 2 - 0.5
    assert np.array_equal(r.pts_map, m.pts_map) and np.array_equal(r.scores, m.scores)
    assert (r.backend, r.ms) == ("x", 3.0)
    assert len(out.restore(Matches.empty("x"))) == 0


@pytest.mark.parametrize("bad", [0.0, -0.5, float("nan"), float("inf")])
def test_a_bad_scale_is_refused(bad):
    with pytest.raises(ValueError, match="scale"):
        preprocess(FRAME, SCALE, scale=bad)


# --- the scale prior --------------------------------------------------------------------------------------------

def test_the_scale_prior_is_frame_resolution_over_map_resolution():
    assert scale_prior(0.3, 0.6) == pytest.approx(0.5)
    # the Irpin flights: meta.json nadir_gsd_m 0.38745 against the map's m_per_px 0.75933: half a map px per frame px
    assert scale_prior(0.3874516013785503, 0.7593316812880911) == pytest.approx(0.5103, abs=1e-4)


@pytest.mark.parametrize("gsd, m_per_px", [(0.0, 0.5), (-1.0, 0.5), (0.3, 0.0), (0.3, -2.0), (float("nan"), 0.5),
                                           (0.3, float("inf"))])
def test_the_scale_prior_refuses_impossible_resolutions(gsd, m_per_px):
    with pytest.raises(ValueError):
        scale_prior(gsd, m_per_px)


# --- CLAHE on luminance -----------------------------------------------------------------------------------------

def test_clahe_raises_the_contrast_of_a_flat_image():
    flat = _exposure(FRAME, 0.25, 100)                    # the scene squeezed into a quarter of the range
    out = preprocess(flat, CLAHE).image
    # the scene fills a quarter of the range; equalising each tile spreads it out again. 1.5x is a low floor for
    # "clearly more contrast" (the clip limit keeps it from reaching the full 4x)
    assert _grey(out).std() > 1.5 * _grey(flat).std()


def test_clahe_brings_two_exposures_of_one_scene_closer():
    # the point of "both map and frame, same settings": the map and the frame see the ground in different light
    dark = _exposure(FRAME, 0.4, 30)
    out, out_dark = _grey(preprocess(FRAME, CLAHE).image), _grey(preprocess(dark, CLAHE).image)
    before, after = _grey(dark).std() / _grey(FRAME).std(), out_dark.std() / out.std()
    # The dark image has 0.4 of the contrast. CLAHE is contrast-LIMITED: it may stretch a tile by at most the clip
    # limit, so it narrows the gap rather than closing it (measured: 0.40 -> 0.51 at clip 2). 1.15x is a floor well
    # under that; without equalisation the ratio stays at 0.40.
    assert before == pytest.approx(0.4, abs=0.01)
    assert before * 1.15 < after <= 1.0
    assert np.abs(out - out_dark).mean() < np.abs(_grey(FRAME) - _grey(dark)).mean()     # and no further apart


def test_clahe_gives_a_grey_image_and_its_bgr_copy_the_same_result():
    g = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    out_g, out_bgr = preprocess(g, CLAHE).image, preprocess(bgr, CLAHE).image
    # luminance of a neutral image is its grey value; 1 level for the colour conversion's rounding
    assert np.abs(out_bgr.astype(int) - out_g[:, :, None].astype(int)).max() <= 1


def test_clahe_keeps_a_neutral_image_neutral():
    # it works on luminance only: no colour appears where there was none (1 level: conversion rounding)
    bgr = cv2.cvtColor(cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    out = preprocess(bgr, CLAHE).image.astype(int)
    assert np.abs(out - out[:, :, :1]).max() <= 1


@pytest.mark.parametrize("bad", [{"clahe_clip": 0.0}, {"clahe_clip": -1.0}, {"clahe_tile_px": 0}])
def test_the_config_rejects_bad_clahe_settings(bad):
    with pytest.raises(ValueError):
        PreprocessConfig(**bad)


# --- grey-world -------------------------------------------------------------------------------------------------

def _mid_tones() -> np.ndarray:
    """A colourful image that stays inside 40..160, so removing the casts below never saturates a channel."""
    return (40 + FRAME.astype(np.float64) * (120 / 255)).round().astype(np.uint8)


def test_grey_world_removes_a_colour_cast():
    cast = _tint(_mid_tones(), (1.0, 0.85, 0.7))          # blue kept, green and red dimmed: a cold tint
    means = preprocess(cast, GREY_WORLD).image.reshape(-1, 3).mean(axis=0)
    # equal channel means is the definition; rounding to uint8 moves a mean by less than half a level
    assert means.max() - means.min() < 1.0
    assert abs(means.mean() - cast.mean()) < 1.0          # the overall brightness is kept


def test_grey_world_makes_two_tints_of_one_scene_alike():
    a = preprocess(_tint(_mid_tones(), (1.0, 0.85, 0.7)), GREY_WORLD).image.astype(int)
    b = preprocess(_tint(_mid_tones(), (0.75, 0.9, 1.0)), GREY_WORLD).image.astype(int)
    # both come back to the same balance, at slightly different brightness (the tints dim the image differently):
    # compare the colour, i.e. each pixel's channels relative to its own mean. 4 levels: two roundings per image
    # (the tint, then the balance), each stretched by a gain of up to 1.25
    colour_a, colour_b = a - a.mean(axis=2, keepdims=True), b - b.mean(axis=2, keepdims=True)
    assert np.abs(colour_a - colour_b).max() <= 4.0


def test_grey_world_leaves_a_neutral_image_alone():
    bgr = cv2.cvtColor(cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    assert np.array_equal(preprocess(bgr, GREY_WORLD).image, bgr)


def test_grey_world_has_nothing_to_do_on_a_grey_or_black_image():
    g = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)
    assert np.array_equal(preprocess(g, GREY_WORLD).image, g)             # one channel: no colour to balance
    black = np.zeros((60, 80, 3), np.uint8)
    assert np.array_equal(preprocess(black, GREY_WORLD).image, black)     # zero means: no division by zero
    blue = np.zeros((60, 80, 3), np.uint8)
    blue[:, :, 0] = 200                                   # an empty channel can't be scaled up from nothing
    out = preprocess(blue, GREY_WORLD).image
    assert out.dtype == np.uint8 and out.shape == blue.shape


def test_grey_world_saturates_instead_of_wrapping_around():
    img = np.full((40, 40, 3), (250, 250, 60), np.uint8)
    img[:20] = (60, 60, 10)                               # the red channel is weak, so it is scaled up a lot
    out = preprocess(img, GREY_WORLD).image
    assert out[30, 30, 2] > img[30, 30, 2]                # red went up, and nothing wrapped past 255 to a low value
    assert out[30, 30, 2] >= out[5, 5, 2]


# --- failure cases ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    np.zeros((0, 0, 3), np.uint8),                        # empty
    np.zeros((0, 10), np.uint8),
    np.zeros((10, 10, 3), np.float32),                    # wrong dtype
    np.zeros((10, 10), np.uint16),
    np.zeros((10, 10, 4), np.uint8),                      # BGRA
    np.zeros((10,), np.uint8),                            # not an image
    [[1, 2], [3, 4]],                                     # not an array
    None,
])
@pytest.mark.parametrize("cfg", [OFF, ALL])
def test_a_wrong_image_is_refused(bad, cfg):
    with pytest.raises(ValueError, match="image"):
        preprocess(bad, cfg)


def test_a_blank_frame_comes_out_blank_and_gives_no_matches(tmp_path):
    blank = np.full((480, 640, 3), 128, np.uint8)
    out = preprocess(blank, ALL, scale=TRUE_SCALE)
    assert out.image.std() == 0                           # no texture invented from nothing
    m = RootSiftMatcher(preprocess(MAP, ALL).image, cache_dir=tmp_path).match(out.image, preprocess(MAP, ALL).image)
    assert len(out.restore(m)) == 0


# --- a degraded frame still localises -----------------------------------------------------------------------------

def test_a_dim_tinted_frame_localises_after_preprocessing(tmp_path):
    frame = _tint(_exposure(FRAME, 0.45, 40), (1.0, 0.9, 0.75))
    prepared_map = preprocess(MAP, ALL).image
    out = preprocess(frame, ALL, scale=TRUE_SCALE)
    assert out.image.shape[0] < frame.shape[0]
    m = out.restore(RootSiftMatcher(prepared_map, cache_dir=tmp_path).match(out.image, prepared_map))
    fix = localize(0, m, FRAME_SIZE)
    assert fix is not None and fix.accepted
    # the clean frame localises within 1 px (test_scale.py); 2 px allows for the dimmed frame's coarser grey levels
    assert math.hypot(fix.x - CX, fix.y - CY) < 2.0
    assert fix.scale == pytest.approx(TRUE_SCALE, rel=0.02)    # points came back in ORIGINAL frame pixels


# --- Done when: the ablation shows the effect of each step ---------------------------------------------------------

def _tiny_dataset(root: Path) -> tuple[Path, Path]:
    """An Irpin-shaped dataset: a map with its sidecar, and a flight of three frames with meta.json and truth."""
    (root / "maps").mkdir(parents=True)
    flight = root / "flights" / "synthetic" / "a"
    (flight / "frames").mkdir(parents=True)
    cv2.imwrite(str(root / "maps" / "map_a.png"), MAP)
    h, w = MAP.shape[:2]
    block = {"file": "maps/map_a.png", "m_per_px": 0.8, "size_px": [w, h], "alignment_to_2025": {"shift_px": [0.0, 0.0]}}
    (root / "maps" / "map_a.json").write_text(json.dumps({"map": block}))
    keyframes, truth = [], []
    for i, (cx, cy, heading) in enumerate([(430.0, 290.0, 30.0), (300.0, 250.0, 100.0), (520.0, 380.0, 200.0)]):
        H = make_h(heading, 0.5, cx=2 * cx, cy=2 * cy, frame_size=FRAME_SIZE)
        name = f"frames/frame_{i:05d}.png"
        cv2.imwrite(str(flight / name), render_frame(GROUND, H, FRAME_SIZE))
        keyframes.append({"i": i, "file": name})
        truth.append(json.dumps({"i": i, "center_map_px": [cx, cy]}))
    # 0.2 m per frame px over 0.8 m per map px: the prior is the true scale, 0.25
    (flight / "meta.json").write_text(json.dumps({"map": "maps/map_a.png", "nadir_gsd_m": 0.2, "keyframes": keyframes}))
    (flight / "truth.jsonl").write_text("\n".join(truth) + "\n")
    return flight, root / "maps" / "map_a.png"


def test_the_ablation_script_prints_one_row_per_step(tmp_path):
    flight, map_path = _tiny_dataset(tmp_path / "set")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "preprocess_ablation.py"), "--flight", str(flight),
         "--maps", str(map_path), "--out", str(tmp_path / "out"), "--cache", str(tmp_path / "cache"),
         "--device", "cpu", "--log-level", "warning"],
        capture_output=True, text=True, cwd=tmp_path, env=env, check=False)
    assert res.returncode == 0, res.stderr
    rows = {}
    for line in res.stdout.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) >= 2 and "/" in cells[1]:
            rows[cells[0]] = cells
    for name in ("none", "CLAHE", "scale", "grey-world", "all"):
        assert name in rows, res.stdout
        assert rows[name][1].endswith("/3"), rows[name]   # every row is measured on the same three frames
    assert rows["scale"][1] == "3/3"                      # clean synthetic frames at the true scale all localise
    assert "0.250" in res.stdout                          # the prior it used: nadir_gsd_m / m_per_px
    saved = json.loads((tmp_path / "out" / "ablation.json").read_text())
    assert {r["variant"] for r in saved["rows"]} >= {"none", "CLAHE", "scale", "grey-world", "all"}


def test_the_ablation_script_stops_cleanly_on_a_missing_flight(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "preprocess_ablation.py"), "--flight", str(tmp_path / "nope"),
         "--maps", str(tmp_path / "nope.png"), "--device", "cpu"],
        capture_output=True, text=True, cwd=tmp_path, env=env, check=False)
    assert res.returncode == 2 and "nope" in res.stderr and "Traceback" not in res.stderr
