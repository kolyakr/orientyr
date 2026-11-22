"""M1-05 · Search strategy.

"Done when: the 50 task frames and the mild 2025 Irpin keyframes run with windows after frame 0, and a forced
miss recovers through the global fallback." So:
- the window: the last footprint moved by the observed step, plus a margin; whole pixels, clipped to the map;
- the first frame, and the frame after a miss, search the whole map ("global");
- after a fix, frames search the window ("window") and are found there, in full-map coordinates;
- a window that fails falls back to the whole map in the same frame ("fallback"); a forced miss (a blank frame)
  is recovered on the next frame by the global search, then windows resume;
- on the task data and the Irpin mild 2025 keyframes: windows after frame 0 (``@pytest.mark.data``).

Ground truth: a synthetic flight over the shared synthetic map (tests/synth.py), exact homographies.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.synth import make_h, make_map, render_frame

from dronenav.matching import RootSiftMatcher
from dronenav.search import SearchConfig, Searcher, SearchMode, search_window
from dronenav.types import Fix

ROOT = Path(__file__).resolve().parents[1]
FRAME_SIZE = (480, 360)
MAP = make_map()
MAP_SIZE = (MAP.shape[1], MAP.shape[0])
# a flight: 40 px steps to the south-east, turning slowly; scale 0.8 (no shrink needed at this zoom)
PATH = [(330.0 + 40 * k, 300.0 + 25 * k, 10.0 + 4 * k) for k in range(8)]
FRAMES = [render_frame(MAP, make_h(h, 0.8, cx=x, cy=y, frame_size=FRAME_SIZE), FRAME_SIZE) for x, y, h in PATH]
BLANK = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), 128, np.uint8)


@pytest.fixture
def searcher(tmp_path):
    return Searcher(RootSiftMatcher(MAP, cache_dir=tmp_path), MAP)


def _near(fix: Fix | None, x: float, y: float, tol_px: float = 1.0) -> bool:
    return fix is not None and fix.accepted and math.hypot(fix.x - x, fix.y - y) < tol_px


def _fix(x, y, footprint, scale=0.8) -> Fix:
    return Fix(idx=0, H=np.eye(3), x=x, y=y, heading_deg=0.0, scale=scale, footprint=np.asarray(footprint, float),
               inliers=100, inlier_ratio=0.5, reproj_px=1.0, accepted=True)


# --- the window -----------------------------------------------------------------------------------------------

SQUARE = [[400, 300], [600, 300], [600, 500], [400, 500]]      # a 200 px footprint centred on (500, 400)


def test_window_is_the_footprint_plus_margin():
    cfg = SearchConfig(margin_frac=0.25, min_margin_px=10)
    x0, y0, x1, y1 = search_window(_fix(500, 400, SQUARE), (0.0, 0.0), (2000, 2000), cfg)
    assert (x0, y0, x1, y1) == (350, 250, 650, 550)               # margin = 0.25 * 200 = 50 px each side


def test_window_margin_has_a_floor():
    cfg = SearchConfig(margin_frac=0.25, min_margin_px=96)
    x0, y0, x1, y1 = search_window(_fix(500, 400, SQUARE), (0.0, 0.0), (2000, 2000), cfg)
    assert (x0, y0, x1, y1) == (304, 204, 696, 596)


def test_window_moves_with_the_step():
    cfg = SearchConfig(margin_frac=0.25, min_margin_px=10)
    still = search_window(_fix(500, 400, SQUARE), (0.0, 0.0), (2000, 2000), cfg)
    moved = search_window(_fix(500, 400, SQUARE), (60.0, -30.0), (2000, 2000), cfg)
    assert moved == (still[0] + 60, still[1] - 30, still[2] + 60, still[3] - 30)
    # two frames since the fix: the prediction goes two steps
    twice = search_window(_fix(500, 400, SQUARE), (60.0, -30.0), (2000, 2000), cfg, frames_ahead=2)
    assert twice[:2] == (still[0] + 120, still[1] - 60)


def test_window_is_clipped_to_the_map_and_whole_pixels():
    cfg = SearchConfig(margin_frac=0.25, min_margin_px=10)
    fp = np.array(SQUARE, float) - 380.5                           # near the top-left corner, fractional
    w = search_window(_fix(119.5, 19.5, fp), (0.0, 0.0), (1000, 800), cfg)
    assert all(isinstance(v, int) for v in w)
    assert w[0] == 0 and w[1] == 0 and w[2] <= 1000 and w[3] <= 800


def test_window_off_the_map_is_none():
    cfg = SearchConfig()
    fp = np.array(SQUARE, float) + 5000
    assert search_window(_fix(5500, 5400, fp), (0.0, 0.0), (1000, 800), cfg) is None


@pytest.mark.parametrize("bad", [{"margin_frac": -0.1}, {"min_margin_px": -1}, {"min_window_px": 0}])
def test_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        SearchConfig(**bad)


# --- the flight: global first, then windows --------------------------------------------------------------------

def test_first_frame_is_global_then_windows(searcher):
    modes = []
    for i, (frame, (x, y, _)) in enumerate(zip(FRAMES, PATH, strict=True)):
        r = searcher.locate(i, frame)
        modes.append(r.mode)
        assert _near(r.fix, x, y), i                               # full-map coordinates in every mode
        assert r.fix.idx == i
        if r.mode == SearchMode.WINDOW:
            x0, y0, x1, y1 = r.window
            assert x0 <= x < x1 and y0 <= y < y1                   # the truth was inside the searched window
            assert (x1 - x0) * (y1 - y0) < MAP_SIZE[0] * MAP_SIZE[1]   # and it is smaller than the map
    assert modes == [SearchMode.GLOBAL] + [SearchMode.WINDOW] * (len(FRAMES) - 1)


def test_global_search_has_no_window(searcher):
    r = searcher.locate(0, FRAMES[0])
    assert r.mode == SearchMode.GLOBAL and r.window is None


def test_windows_follow_the_step(searcher):
    # after two fixes the step is known: the window's centre is ahead of the last fix by about one step
    for i in range(3):
        r = searcher.locate(i, FRAMES[i])
    x0, y0, x1, y1 = r.window
    _, (xb, yb, _), (xc, yc, _) = PATH[:3]
    centre = ((x0 + x1) / 2, (y0 + y1) / 2)
    # the window centre follows the predicted frame centre, up to the footprint's asymmetry under rotation and
    # the map clip (the flight is well inside the map here)
    assert math.hypot(centre[0] - xc, centre[1] - yc) < 25
    assert math.hypot(centre[0] - xb, centre[1] - yb) > 25         # not left at the previous position


def test_windows_use_the_cached_map_features(searcher):
    for i, frame in enumerate(FRAMES):
        searcher.locate(i, frame)
    assert searcher.matcher.map_extractions == 1                   # windows never re-extract the map


# --- failures: fallback within the frame, global after a miss --------------------------------------------------

def test_a_jump_out_of_the_window_falls_back_to_global_in_the_same_frame(searcher):
    for i in range(3):
        searcher.locate(i, FRAMES[i])
    # the drone "teleports" (a skipped stretch of video): the window around the prediction can't hold it
    far = (900.0, 650.0, 200.0)
    frame = render_frame(MAP, make_h(far[2], 0.8, cx=far[0], cy=far[1], frame_size=FRAME_SIZE), FRAME_SIZE)
    r = searcher.locate(3, frame)
    assert r.mode == SearchMode.FALLBACK
    assert _near(r.fix, far[0], far[1])


def test_a_forced_miss_recovers_through_global_search(searcher):
    for i in range(3):
        searcher.locate(i, FRAMES[i])
    miss = searcher.locate(3, BLANK)                               # forced miss: nothing to match
    assert miss.mode == SearchMode.FALLBACK                        # the window was tried, then the whole map
    assert miss.fix is None or not miss.fix.accepted
    back = searcher.locate(4, FRAMES[4])
    assert back.mode == SearchMode.GLOBAL                          # after a miss: the whole map
    assert _near(back.fix, *PATH[4][:2])
    again = searcher.locate(5, FRAMES[5])
    assert again.mode == SearchMode.WINDOW and _near(again.fix, *PATH[5][:2])   # and windows resume


def test_reset_forgets_the_track(searcher):
    searcher.locate(0, FRAMES[0])
    searcher.reset()
    assert searcher.locate(1, FRAMES[1]).mode == SearchMode.GLOBAL


# --- on the data, when data/ is present -----------------------------------------------------------------------

TASK = ROOT / "data" / "hackatons-data"
PROBE = ROOT / "experiments" / "data_probe" / "results" / "probe.json"
IRPIN = ROOT / "data" / "irpin" / "flights" / "mild" / "2025"
IRPIN_MAP = ROOT / "data" / "irpin" / "maps" / "irpin_2025.png"


@pytest.mark.data
@pytest.mark.skipif(not (TASK / "global_map.png").exists() or not PROBE.exists(), reason="data/ not downloaded")
def test_task_frames_run_with_windows_after_frame_0(tmp_path):
    truth = {r["i"]: r for r in json.loads(PROBE.read_text())["rows"]}
    mp = cv2.imread(str(TASK / "global_map.png"))
    s = Searcher(RootSiftMatcher(mp, cache_dir=tmp_path), mp)
    modes = []
    for i in range(50):
        r = s.locate(i, cv2.imread(str(TASK / "crops" / f"crop_{i}.png")))
        modes.append(r.mode)
        # probe.json: the prototype's RootSIFT + MAGSAC fit, rounded to 0.1 px; 2 px is the M1 accuracy goal
        assert _near(r.fix, truth[i]["cx"], truth[i]["cy"], 2.0), i
    assert modes == [SearchMode.GLOBAL] + [SearchMode.WINDOW] * 49


def _irpin():
    meta = json.loads((IRPIN / "meta.json").read_text())
    truth = {r["i"]: r for r in map(json.loads, (IRPIN / "truth.jsonl").read_text().splitlines()) if r}
    return meta["keyframes"], truth


@pytest.mark.data
@pytest.mark.skipif(not (IRPIN / "meta.json").exists() or not IRPIN_MAP.exists(), reason="data/ not downloaded")
def test_irpin_keyframes_run_with_windows_after_frame_0_and_recover_from_a_miss(tmp_path):
    keys, truth = _irpin()
    mp = cv2.imread(str(IRPIN_MAP))
    s = Searcher(RootSiftMatcher(mp, cache_dir=tmp_path), mp)
    keys = keys[:40]                                   # the first 40 s of flight; scripts/search_run.py runs all
    modes = []
    for n, k in enumerate(keys):
        frame = cv2.imread(str(IRPIN / k["file"]))
        if n == 20:                                    # forced miss mid-flight
            miss = s.locate(n, np.zeros_like(frame))
            assert miss.mode == SearchMode.FALLBACK and (miss.fix is None or not miss.fix.accepted)
        r = s.locate(n, frame)
        modes.append(r.mode)
        cx, cy = truth[k["i"]]["center_map_px"]
        # truth is for undistorted pixels and the frames carry the lens distortion; the matcher sees raw frames,
        # which moves the centre by well under a pixel (measured median 0.4 px); 3 px leaves room for the worst
        assert _near(r.fix, cx, cy, 3.0), n
    # window after frame 0; the frame after the forced miss is found by the global search
    expected = [SearchMode.GLOBAL] + [SearchMode.WINDOW] * (len(keys) - 1)
    expected[20] = SearchMode.GLOBAL
    assert modes == expected
