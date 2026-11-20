"""M1-03 · Frame-to-map scale: downscaling.

"Done when: the chosen scale rule is in the code and a short table (scale vs inliers vs ms) goes into the README
notes." So:
- the rule: a frame whose pixels are finer than the map's is shrunk by ``scale / target_scale`` (never enlarged,
  never below ``min_factor``); bad scales are refused;
- the scale comes from the last accepted fix; with none, a few probe factors are tried and the best fix wins;
- matching a shrunk frame still returns points in the ORIGINAL frame's pixels, so geometry, the run log and
  the next scale estimate are unaffected by the shrink;
- on the task data, every crop localises at the rule's factor (``@pytest.mark.data``);
- the README carries the scale table.

Ground truth: a synthetic "ground" texture and a map at half its resolution; frames are rendered from the
ground, so they are 4x more detailed than the map (scale 0.25, like the task crops) and the truth is exact.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.synth import apply, make_h, make_map, render_frame

from dronenav.geometry import localize
from dronenav.matching import RootSiftMatcher
from dronenav.scale import (
    ScaleConfig,
    ScaleTracker,
    downscale_factor,
    localize_scaled,
    match_scaled,
    resize_frame,
)
from dronenav.types import Fix, Matches

ROOT = Path(__file__).resolve().parents[1]
FRAME_SIZE = (640, 480)
GROUND = make_map(2400, 1800, seed=3, shapes=3600)   # 4x the default area, so 4x the shapes: same density
MAP = cv2.resize(GROUND, (1200, 900), interpolation=cv2.INTER_AREA)        # 1 map px = 2 ground px
TO_MAP = np.diag([0.5, 0.5, 1.0])                                          # ground px -> map px
# frame -> ground at scale 0.5, so frame -> map at 0.25: one frame px covers a quarter of a map px
H_GROUND = make_h(30.0, 0.5, cx=1300.0, cy=860.0, frame_size=FRAME_SIZE)
H_TRUE = TO_MAP @ H_GROUND
CX, CY = 650.0, 430.0                                                      # the frame centre on the map
TRUE_SCALE = 0.25
FRAME = render_frame(GROUND, H_GROUND, FRAME_SIZE)


@pytest.fixture
def matcher(tmp_path):
    return RootSiftMatcher(MAP, cache_dir=tmp_path)


def _near(fix: Fix | None, tol_px: float = 1.0) -> bool:
    return fix is not None and fix.accepted and math.hypot(fix.x - CX, fix.y - CY) < tol_px


# --- the rule -------------------------------------------------------------------------------------------------

def test_detailed_frames_are_shrunk_towards_the_target():
    cfg = ScaleConfig()
    # the task crops: 0.2535 map px per frame px -> shrink so one frame px covers target_scale map px
    assert downscale_factor(0.2535, cfg) == pytest.approx(0.2535 / cfg.target_scale)
    assert downscale_factor(0.2535, ScaleConfig(target_scale=0.5)) == pytest.approx(0.507)


def test_frames_as_coarse_as_the_target_are_never_enlarged():
    # enlarging adds no detail, only cost: the factor stops at 1
    assert downscale_factor(0.65) == 1.0
    assert downscale_factor(3.0) == 1.0


def test_tiny_scales_stop_at_the_minimum_factor():
    cfg = ScaleConfig(min_factor=0.1)
    assert downscale_factor(0.001, cfg) == 0.1


@pytest.mark.parametrize("bad", [0.0, -0.3, float("nan"), float("inf")])
def test_bad_scales_are_refused(bad):
    with pytest.raises(ValueError):
        downscale_factor(bad)


@pytest.mark.parametrize("bad", [{"target_scale": 0.0}, {"min_factor": 0.0}, {"min_factor": 1.5},
                                 {"probe_factors": ()}, {"probe_factors": (0.5, 1.2)}])
def test_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        ScaleConfig(**bad)


# --- resizing and matching a shrunk frame ---------------------------------------------------------------------

def test_resize_frame_halves_the_size_and_keeps_factor_one_as_is():
    small = resize_frame(FRAME, 0.5)
    assert small.shape == (240, 320, 3) and small.dtype == np.uint8
    assert resize_frame(FRAME, 1.0) is FRAME          # no copy, no resampling blur


def test_resize_frame_rejects_bad_factors():
    for f in (0.0, -1.0, 1.5):
        with pytest.raises(ValueError):
            resize_frame(FRAME, f)


def test_shrunk_matches_are_in_original_frame_pixels(matcher):
    m = match_scaled(matcher, FRAME, MAP, 0.4)
    assert isinstance(m, Matches) and len(m) >= 50
    # points fall inside the ORIGINAL frame, and agree with the truth there (a forgotten 1/factor would put
    # every frame point 2.5x too close to the corner: the truth check catches it)
    assert (m.pts_frame[:, 0] < FRAME_SIZE[0]).all() and (m.pts_frame[:, 1] < FRAME_SIZE[1]).all()
    err = np.linalg.norm(apply(H_TRUE, m.pts_frame) - m.pts_map, axis=1)
    assert np.mean(err < 3.0) > 0.5
    fix = localize(0, m, FRAME_SIZE)
    # the pose is about the original frame: its scale is the true 0.25, not 0.25 / 0.4
    assert _near(fix) and fix.scale == pytest.approx(TRUE_SCALE, rel=0.02)


def test_shrunk_matching_respects_window_offset(matcher):
    x0, y0 = 450, 250
    m = match_scaled(matcher, FRAME, MAP[y0:y0 + 400, x0:x0 + 450], 0.4, map_offset=(x0, y0))
    assert len(m) > 0 and _near(localize(0, m, FRAME_SIZE))


def test_shrinking_to_the_rule_keeps_the_fix_and_cuts_frame_keypoints(matcher):
    full = match_scaled(matcher, FRAME, MAP, 1.0)
    f = downscale_factor(TRUE_SCALE)
    shrunk = match_scaled(matcher, FRAME, MAP, f)
    assert _near(localize(0, shrunk, FRAME_SIZE))
    # wall time is too noisy to assert in CI; what the shrink saves is frame work, visible as fewer matches
    # to verify, never as a worse fix (the timing goes into the README table)
    assert len(shrunk) < len(full)


# --- choosing the factor: from the last fix, or by probing ----------------------------------------------------

def test_tracker_without_a_fix_probes_several_factors():
    t = ScaleTracker()
    assert t.scale is None
    assert t.factors() == ScaleConfig().probe_factors and len(t.factors()) >= 2


def _fix(scale: float, accepted: bool = True) -> Fix:
    return Fix(idx=0, H=np.eye(3), x=0.0, y=0.0, heading_deg=0.0, scale=scale, footprint=np.zeros((4, 2)),
               inliers=100 if accepted else 3, inlier_ratio=0.5, reproj_px=1.0, accepted=accepted,
               reject_reason=None if accepted else "too_few_inliers")


def test_tracker_follows_the_last_accepted_fix():
    t = ScaleTracker()
    t.update(_fix(0.2535))
    assert t.scale == pytest.approx(0.2535)
    assert t.factors() == (pytest.approx(downscale_factor(0.2535)),)
    t.update(_fix(0.5))                               # altitude changed: the next factor follows
    assert t.factors() == (pytest.approx(downscale_factor(0.5)),)


def test_tracker_ignores_rejected_and_missing_fixes():
    t = ScaleTracker()
    t.update(_fix(0.2535))
    t.update(_fix(5.0, accepted=False))               # a bad homography says nothing about the scale
    t.update(None)
    assert t.scale == pytest.approx(0.2535)


def test_tracker_reset_goes_back_to_probing():
    t = ScaleTracker()
    t.update(_fix(0.2535))
    t.reset()                                         # lost track: the scale may have changed meanwhile
    assert t.scale is None and t.factors() == ScaleConfig().probe_factors


def test_localize_scaled_probes_and_keeps_the_best(matcher):
    res = localize_scaled(0, matcher, FRAME, MAP, ScaleConfig().probe_factors)
    assert _near(res.fix)
    assert res.factor in ScaleConfig().probe_factors
    assert res.fix.scale == pytest.approx(TRUE_SCALE, rel=0.02)
    # the probe that won has at least as many inliers as each one tried
    for f in ScaleConfig().probe_factors:
        other = localize(0, match_scaled(matcher, FRAME, MAP, f), FRAME_SIZE)
        if other is not None and other.accepted:
            assert res.fix.inliers >= other.inliers


def test_localize_scaled_feeds_the_tracker(matcher):
    t = ScaleTracker()
    first = localize_scaled(0, matcher, FRAME, MAP, t.factors())
    t.update(first.fix)
    second = localize_scaled(1, matcher, FRAME, MAP, t.factors())
    assert second.factor == pytest.approx(downscale_factor(TRUE_SCALE), rel=0.03)
    assert _near(second.fix) and second.fix.idx == 1


def test_localize_scaled_on_a_blank_frame_gives_no_accepted_fix(matcher):
    blank = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), 128, np.uint8)
    res = localize_scaled(0, matcher, blank, MAP, (1.0, 0.5))
    assert res.fix is None or not res.fix.accepted
    assert len(res.matches) == 0


# --- the README notes -----------------------------------------------------------------------------------------

def test_readme_has_the_scale_table():
    text = (ROOT / "README.md").read_text()
    assert "Frame-to-map scale" in text
    # a markdown table with the factor, inliers and time columns
    header = next((ln for ln in text.splitlines() if ln.startswith("|") and "factor" in ln.lower()), "")
    assert "inliers" in header.lower() and "ms" in header.lower()


# --- on the task data, when data/ is present ------------------------------------------------------------------

TASK = ROOT / "data" / "hackatons-data"
PROBE = ROOT / "experiments" / "data_probe" / "results" / "probe.json"


@pytest.mark.data
@pytest.mark.skipif(not (TASK / "global_map.png").exists() or not PROBE.exists(), reason="data/ not downloaded")
def test_every_task_crop_localizes_at_the_rule_factor(tmp_path):
    truth = {r["i"]: r for r in json.loads(PROBE.read_text())["rows"]}
    mp = cv2.imread(str(TASK / "global_map.png"))
    m = RootSiftMatcher(mp, cache_dir=tmp_path)
    t = ScaleTracker()
    for i in range(50):
        crop = cv2.imread(str(TASK / "crops" / f"crop_{i}.png"))
        res = localize_scaled(i, m, crop, mp, t.factors())
        t.update(res.fix)
        # probe.json is the prototype's RootSIFT + MAGSAC fit, rounded to 0.1 px; 2 px is the M1 accuracy goal
        assert res.fix is not None and res.fix.accepted, i
        assert math.hypot(res.fix.x - truth[i]["cx"], res.fix.y - truth[i]["cy"]) < 2.0, i
        if i > 0:   # after the first fix the tracker uses one factor: the rule's, about 0.39 for these crops
            assert res.factor == pytest.approx(downscale_factor(0.2535), rel=0.05)
