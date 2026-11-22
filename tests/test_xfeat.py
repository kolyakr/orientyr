"""M4-03 · XFeat backend: a learned ``Matcher`` for the fast tracking path.

"Done when: it passes the contract test; time per frame is measured." So:
- the shared contract (tests/test_matcher_contract.py) runs on ``XFeatMatcher`` too;
- time per frame is measured by scripts/xfeat_timing.py, XFeat against RootSIFT on the same frames; its table is
  in the README notes, and here the script is run on a few synthetic cut-outs.

And what the contract leaves to each backend:
- weights: never committed, loaded from a local file, downloaded only when allowed; a missing file is a clear
  error, not a traceback from inside torch;
- the package still imports without torch (the classic pipeline needs none);
- matched points are in corner-origin pixels (M1-12), like RootSIFT's;
- XFeat's descriptors are not rotation-invariant, so the matcher tries the frame at several turns: a frame at any
  heading localizes, and the turn found for one frame is tried first for the next (the tracking path: one
  extraction per frame instead of one per turn);
- the map's features are extracted in tiles once and cached on disk, as RootSIFT's are.

Trained weights are needed for anything that has to find real matches: an untrained network detects nothing.
Those tests carry ``needs_xfeat_weights`` and skip where the weights are not in the local cache (CI); the rest
run everywhere.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.synth import apply, make_h, make_map, render_frame
from tests.weights import needs_xfeat_weights

from dronenav import pipeline
from dronenav.geometry import localize
from dronenav.matching import XFeatConfig, XFeatMatcher, XFeatWeightsMissing, xfeat_weights_path
from dronenav.sim import CutoutSource, cutout_route
from dronenav.telemetry import read_run
from dronenav.types import Matcher, Matches

ROOT = Path(__file__).resolve().parents[1]
FRAME_SIZE = (480, 360)
MAP = make_map()
H_TRUE = make_h(35.0, 0.8, cx=640.0, cy=420.0, frame_size=FRAME_SIZE)
FRAME = render_frame(MAP, H_TRUE, FRAME_SIZE)


def _matcher(cache_dir=None, **kw) -> XFeatMatcher:
    """On the CPU and never downloading: tests are the same on every machine and stay off the network."""
    return XFeatMatcher(cache_dir=cache_dir, device="cpu", download=False, **kw)


def _script(name: str, *args) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, args)], check=False,
                          capture_output=True, text=True, env=env, cwd=ROOT)


# --- no weights needed ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [{"top_k": 0}, {"map_top_k": 0}, {"detection_threshold": 0.0}, {"min_cossim": 1.5},
                                 {"ratio": 0.0}, {"ratio": 1.5}, {"turns": 0}, {"win_ratio": 0.5},
                                 {"keep_fraction": 1.5}, {"keep_min": 0}, {"tile": 100}, {"tile": 1000},
                                 {"tile_margin": 48}, {"tile_margin": -32}])
def test_config_rejects_bad_values(bad):
    # tile and tile_margin: multiples of 32, the network's stride, so a tile is never resized on its way in
    with pytest.raises(ValueError):
        XFeatConfig(**bad)


def test_the_package_imports_without_torch():
    # the classic pipeline must run where torch is not installed: torch is imported when a matcher is built
    code = "import sys, dronenav.matching, dronenav.pipeline; sys.exit('torch' in sys.modules or 'kornia' in sys.modules)"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    assert subprocess.run([sys.executable, "-c", code], check=False, env=env).returncode == 0


def test_the_default_weights_file_is_in_the_torch_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("DRONENAV_XFEAT_WEIGHTS", raising=False)
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    assert xfeat_weights_path() == tmp_path / "torch" / "hub" / "checkpoints" / "xfeat.pt"
    monkeypatch.setenv("DRONENAV_XFEAT_WEIGHTS", str(tmp_path / "mine.pt"))     # an explicit file wins
    assert xfeat_weights_path() == tmp_path / "mine.pt"


def test_missing_weights_are_a_clear_error_when_downloading_is_off(tmp_path):
    with pytest.raises(XFeatWeightsMissing, match="nope.pt") as e:
        _matcher(weights=tmp_path / "nope.pt")
    assert isinstance(e.value, FileNotFoundError) and "download" in str(e.value)


def test_an_unknown_device_is_a_value_error(tmp_path):
    # checked before the weights: the device name is wrong whether or not the weights are there
    with pytest.raises(ValueError, match="gpu0"):
        XFeatMatcher(device="gpu0", weights=tmp_path / "nope.pt", download=False)


def test_the_pipeline_knows_the_matcher():
    assert pipeline.MATCHERS["xfeat"][0] == XFeatMatcher.backend == "xfeat"


@pytest.fixture
def untrained(tmp_path):
    """A weights file of the right layout with untrained values: enough to load the network and run it."""
    torch = pytest.importorskip("torch")
    from kornia.feature import XFeatModel
    torch.manual_seed(0)
    path = tmp_path / "untrained.pt"
    torch.save(XFeatModel().state_dict(), path)
    return path


def test_a_given_weights_file_is_loaded_and_runs_on_the_cpu(untrained):
    matcher = _matcher(weights=untrained)
    assert isinstance(matcher, Matcher) and matcher.device == "cpu"
    for frame, win, off in ((FRAME, MAP, (0, 0)),
                            (np.full((360, 480, 3), 128, np.uint8), MAP, (0, 0)),              # blank frame
                            (FRAME, np.full((300, 300, 3), 90, np.uint8), (5000, 5000))):      # blank map window
        m = matcher.match(frame, win, map_offset=off)
        assert isinstance(m, Matches) and m.backend == "xfeat" and m.ms >= 0.0
        assert m.pts_frame.shape == m.pts_map.shape == (len(m), 2)


def test_bad_arguments_are_value_errors(untrained):
    matcher = _matcher(weights=untrained)
    with pytest.raises(ValueError, match="mask"):
        matcher.match(FRAME, MAP, mask=np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError, match="whole pixels"):
        matcher.match(FRAME, MAP[:300, :300], map_offset=(0.5, 0))
    with pytest.raises(ValueError, match="shape"):
        matcher.match(FRAME[:, :, :2], MAP)


def test_a_file_that_is_not_xfeat_weights_is_a_clear_error(tmp_path):
    pytest.importorskip("torch")
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="bad.pt"):
        _matcher(weights=bad)


# --- with the trained weights -----------------------------------------------------------------------------------

@needs_xfeat_weights
def test_matched_points_follow_the_corner_convention():
    # as in test_half_pixel.py: a frame that is the map patch turned 180° exactly (no resampling). In corner-origin
    # pixels the frame point (x, y) is the map point (x0 + w - x, y0 + h - y); in the pixel-centre convention the
    # network's keypoints come in, every match would be 1 px off on each axis
    x0, y0, (w, h) = 400, 300, FRAME_SIZE
    frame = np.ascontiguousarray(np.rot90(MAP[y0:y0 + h, x0:x0 + w], 2))
    m = _matcher().match(frame, MAP)
    assert len(m) >= 50
    resid = m.pts_frame + m.pts_map - (x0 + w, y0 + h)
    # the same pixels on both sides give the same keypoints, so right matches agree exactly; 0.1 px as in M1-12
    assert np.abs(np.median(resid, axis=0)).max() < 0.1


@needs_xfeat_weights
@pytest.mark.parametrize("heading", [0.0, 35.0, 90.0, 135.0, 180.0, 247.0, 300.0])
def test_a_frame_at_any_heading_localizes(heading):
    H = make_h(heading, 0.8, cx=640.0, cy=420.0, frame_size=FRAME_SIZE)
    m = _matcher().match(render_frame(MAP, H, FRAME_SIZE), MAP)
    err = np.linalg.norm(apply(H, m.pts_frame) - m.pts_map, axis=1)
    assert len(m) >= 50 and np.mean(err < 3.0) > 0.5          # the contract's numbers, at every heading
    fix = localize(0, m, FRAME_SIZE)
    assert fix is not None and fix.accepted
    assert np.hypot(fix.x - 640.0, fix.y - 420.0) < 1.0       # 1 px, as in the contract: detector jitter


@needs_xfeat_weights
def test_the_turn_found_for_one_frame_is_tried_first_for_the_next():
    matcher = _matcher()
    assert matcher.turn_deg is None
    first = matcher.match(FRAME, MAP)
    assert matcher.frame_extractions == matcher.cfg.turns      # no turn known yet: all of them are tried
    assert matcher.turn_deg == 45.0                            # the nearest of the 8 turns to the 35° heading
    again = matcher.match(FRAME, MAP)
    assert matcher.frame_extractions == matcher.cfg.turns + 1  # the tracking path: one extraction
    np.testing.assert_array_equal(first.pts_frame, again.pts_frame)
    np.testing.assert_array_equal(first.pts_map, again.pts_map)


@needs_xfeat_weights
def test_a_frame_that_turned_away_gets_all_turns_again():
    matcher = _matcher()
    matcher.match(FRAME, MAP)
    H = make_h(200.0, 0.8, cx=640.0, cy=420.0, frame_size=FRAME_SIZE)       # the next frame faces the other way
    before = matcher.frame_extractions
    m = matcher.match(render_frame(MAP, H, FRAME_SIZE), MAP)
    assert matcher.frame_extractions == before + matcher.cfg.turns          # the kept turn first, then the rest
    assert matcher.turn_deg in (180.0, 225.0)
    fix = localize(0, m, FRAME_SIZE)
    assert fix is not None and fix.accepted and np.hypot(fix.x - 640.0, fix.y - 420.0) < 1.0


@needs_xfeat_weights
def test_chance_matches_do_not_leave_a_turn_to_trust():
    # the scale probe (dronenav.scale) hands the matcher the same frame shrunk several times: at the wrong size
    # it matches by chance at every turn, and the "winner" of that must not be kept for the next call
    matcher = _matcher()
    H = make_h(200.0, 0.8, cx=640.0, cy=420.0, frame_size=FRAME_SIZE)
    small = cv2.resize(render_frame(MAP, H, FRAME_SIZE), None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    matcher.match(small, MAP)
    m = matcher.match(FRAME, MAP)
    assert matcher.turn_deg == 45.0
    fix = localize(0, m, FRAME_SIZE)
    assert fix is not None and fix.accepted and np.hypot(fix.x - 640.0, fix.y - 420.0) < 1.0


@needs_xfeat_weights
def test_one_turn_is_enough_for_a_north_up_frame():
    # turns=1: the caller turns frames north-up itself (a compass), and each frame costs one extraction
    matcher = _matcher(cfg=XFeatConfig(turns=1))
    H = make_h(5.0, 0.8, cx=640.0, cy=420.0, frame_size=FRAME_SIZE)
    m = matcher.match(render_frame(MAP, H, FRAME_SIZE), MAP)
    assert matcher.frame_extractions == 1 and matcher.turn_deg == 0.0
    fix = localize(0, m, FRAME_SIZE)
    assert fix is not None and fix.accepted and np.hypot(fix.x - 640.0, fix.y - 420.0) < 1.0


@needs_xfeat_weights
def test_second_run_loads_map_features_from_cache(tmp_path):
    first = _matcher(tmp_path)
    m1 = first.match(FRAME, MAP)
    assert first.map_extractions == 1 and first.map_features_source == "extracted"
    assert len([p for p in tmp_path.iterdir() if p.is_file()]) == 1

    second = _matcher(tmp_path)                    # a new run: nothing in memory
    m2 = second.match(FRAME, MAP)
    assert second.map_extractions == 0 and second.map_features_source == "cache"
    np.testing.assert_array_equal(m1.pts_frame, m2.pts_frame)
    np.testing.assert_array_equal(m1.pts_map, m2.pts_map)

    other = _matcher(tmp_path, cfg=XFeatConfig(map_top_k=2048))     # other extraction parameters: another file
    other.match(FRAME, MAP)
    assert other.map_extractions == 1 and len([p for p in tmp_path.iterdir() if p.is_file()]) == 2


@needs_xfeat_weights
def test_windows_of_the_registered_map_reuse_its_features(tmp_path):
    matcher = _matcher(tmp_path, map_img=MAP)
    assert matcher.map_extractions == 1
    for x0, y0, x1, y1 in ((380, 160, 900, 680), (300, 100, 1000, 800)):
        m = matcher.match(FRAME, MAP[y0:y1, x0:x1], map_offset=(x0, y0))
        fix = localize(0, m, FRAME_SIZE)
        assert fix is not None and fix.accepted and np.hypot(fix.x - 640.0, fix.y - 420.0) < 1.0
    assert matcher.map_extractions == 1


@needs_xfeat_weights
def test_a_map_larger_than_a_tile_is_matched_across_the_seams(tmp_path):
    # tiles of 256 px cut the frame's 384 x 288 px footprint several times: features next to a seam must still
    # be there and in the right place, or the fix would lean away from the seams
    whole = _matcher(cfg=XFeatConfig(tile=2048)).match(FRAME, MAP)            # one tile holds the map
    tiled = _matcher(cfg=XFeatConfig(tile=256)).match(FRAME, MAP)
    assert len(tiled) > 0.8 * len(whole)
    err = np.linalg.norm(apply(H_TRUE, tiled.pts_frame) - tiled.pts_map, axis=1)
    assert np.mean(err < 3.0) > 0.5
    fix = localize(0, tiled, FRAME_SIZE)
    assert fix is not None and fix.accepted and np.hypot(fix.x - 640.0, fix.y - 420.0) < 1.0


@needs_xfeat_weights
def test_run_py_localizes_with_the_xfeat_matcher(tmp_path):
    world = make_map(2000, 1500, shapes=2500)      # room for a cut-out route to turn around in, as in test_run.py
    cv2.imwrite(str(tmp_path / "map.png"), world)
    poses = cutout_route((world.shape[1], world.shape[0]), 5, seed=3, frame_size=(320, 240), scale=(0.9, 1.1),
                         tilt_max_deg=3.0)
    src = CutoutSource(world, poses, (320, 240))
    (tmp_path / "frames").mkdir()
    for f in src:
        cv2.imwrite(str(tmp_path / "frames" / f"frame_{f.idx:03d}.png"), f.image)
    env = {"DRONENAV_XFEAT_WEIGHTS": str(xfeat_weights_path())}
    res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run.py"), "--map", str(tmp_path / "map.png"),
                          "--frames", str(tmp_path / "frames"), "--out", str(tmp_path / "out"), "--cache",
                          str(tmp_path / "cache"), "--matcher", "xfeat", "--device", "cpu", "--tracker", "none"],
                         check=False, capture_output=True, text=True, env={**os.environ, **env})
    assert res.returncode == 0, res.stderr
    recs = read_run(tmp_path / "out" / "run.jsonl")
    assert len(recs) == 5
    for r in recs:
        tr = src.truth(r.idx)
        assert r.backend == "xfeat" and r.track.fix.accepted, r.idx
        # 2 px, as for RootSIFT in test_run.py: an undistorted cut-out with a mild tilt
        assert np.hypot(r.track.x - tr.center[0], r.track.y - tr.center[1]) <= 2.0, r.idx


# --- time per frame ---------------------------------------------------------------------------------------------

@needs_xfeat_weights
def test_the_timing_script_measures_both_matchers(tmp_path):
    res = _script("xfeat_timing.py", "--n", 4, "--device", "cpu", "--no-download", "--out", tmp_path / "t",
                  "--cache", tmp_path / "cache")
    assert res.returncode == 0, res.stderr
    rows = {r["matcher"]: r for r in json.loads((tmp_path / "t" / "timing.json").read_text())["rows"]}
    assert set(rows) >= {"RootSIFT", "XFeat"}
    for r in rows.values():
        assert r["n"] == 4 and r["ms"] > 0 and 0 <= r["localised"] <= 4 and r["device"] == "cpu"
    table = (tmp_path / "t" / "timing.md").read_text()
    assert "| RootSIFT |" in table and "| XFeat |" in table and table in res.stdout


def test_the_timing_script_says_when_the_weights_are_missing(tmp_path):
    res = _script("xfeat_timing.py", "--n", 2, "--device", "cpu", "--no-download", "--weights", tmp_path / "nope.pt",
                  "--out", tmp_path / "t", "--cache", tmp_path / "cache")
    assert res.returncode == 2 and "nope.pt" in res.stderr and "Traceback" not in res.stderr, res.stderr
