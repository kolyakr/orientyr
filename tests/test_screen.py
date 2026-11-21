"""M2-08: a simulated phone filming the map on a screen, with exact truth.

Contract (issue #117, spec update of 21 Nov):
- the truth is exact: with only the geometric effects (tilt, shake), every frame registers to the map warped by its
  truth within 1 px; with the rolling shutter on, the centre rows do (the truth is the centre row's time);
- each effect, at strength 1, changes the frame the way it should; all at 0 is the clean render;
- the same seed gives the same truth and frames;
- the output is an Irpin-style flight that FlightSource, VideoSource and FolderSource read unchanged;
- presets calm / hard / jam; the jam's covered frames are dark and flagged;
- clear errors for bad settings; the CLI writes the flight or the check sheet; the README says it is simulated.

The tests use a small screen and camera so they run in seconds; the physics is the same at full size.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.io import FolderSource
from dronenav.sim import FlightSource
from dronenav.sim.cutout import render_cutout
from dronenav.sim.screen import EFFECTS, PRESETS, Camera, Screen, ScreenSim, effects_for, write_screen_video
from dronenav.video import VideoSource

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "screen_video.py"

SCREEN = Screen(size=(2400, 1600))           # the whole test map at 100 %, at the default 0.233 mm pitch
CAM = Camera(size=(320, 180), fps=10.0)
DIST = 40.0                                  # mm: the view is ~240 screen px wide, ~1.3 camera px per screen px
KW = {"screen": SCREEN, "camera": CAM, "distance_mm": DIST}


def _sim(map_img, seconds=2.0, seed=0, **effects) -> ScreenSim:
    return ScreenSim(map_img, seconds=seconds, seed=seed, effects=effects, **KW)


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Sub-pixel translation between two equal-size grey images, and the peak response (1 = identical)."""
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return math.hypot(dx, dy), resp


@pytest.fixture(scope="module")
def city(city_map):
    return city_map(2400, 1600)       # room for a 25-degree tilt to stay on the screen (view_contained)


@pytest.fixture(scope="module")
def grey():
    return np.full((1600, 2400, 3), 128, np.uint8)       # flat: anything that appears is the effect itself


# --- the truth is exact --------------------------------------------------------------------------------------

def test_the_truth_registers_every_frame_within_1_px(city):
    sim = _sim(city, tilt=1.0, shake=1.0)
    assert len(sim) == 20 and sim.fps == 10.0 and sim.view_contained
    for i in range(0, len(sim), 3):
        t = sim.truth(i)
        ref = render_cutout(city, t.H, CAM.size)                     # the map as the truth says the frame sees it
        d, resp = _shift(_gray(ref), _gray(sim.frame(i)))
        # measured ~0.05 px; 1 px is the spec. resp: the same content, not a lucky peak on a different view
        assert d <= 1.0 and resp > 0.3, f"frame {i}: shift {d:.2f} px, response {resp:.2f}"


def test_the_truth_heading_is_the_engines_heading_and_points_along_the_route(city):
    """The heading in truth.jsonl must mean what the engine's pose means (types.py: clockwise from map up), or a
    heading error measured against it is wrong; and the phone looks ahead, as a drone camera does."""
    from dronenav.geometry import pose_from_homography
    sim = _sim(city, seconds=4.0, tilt=1.0, shake=1.0)
    ahead = []
    for i in range(0, len(sim) - 1, 3):
        t = sim.truth(i)
        d = (pose_from_homography(t.H, CAM.size).heading_deg - t.heading_deg + 180) % 360 - 180
        assert abs(d) < 0.5, f"frame {i}: heading off by {d:.2f} degrees"        # measured 0.0
        v = np.subtract(sim.truth(i + 1).center, t.center)
        ahead.append(abs((math.degrees(math.atan2(v[0], -v[1])) - t.heading_deg + 180) % 360 - 180))
    assert np.median(ahead) < 20                                                 # measured ~6 with shake


def test_the_centre_is_where_the_optical_axis_meets_the_map(city):
    """center_map_px is H at the principal point (the flight format's meaning, sim/truth.py), which this camera
    puts at the frame's centre (w/2, h/2 in the corner-origin convention)."""
    sim = _sim(city, tilt=1.0)
    assert np.allclose(sim.K[:2, 2], (CAM.size[0] / 2, CAM.size[1] / 2))
    for i in (0, 9, 19):
        t = sim.truth(i)
        q = t.H @ [sim.K[0, 2], sim.K[1, 2], 1.0]
        assert np.allclose(t.center, q[:2] / q[2], atol=1e-9)


def test_one_effects_strength_changes_that_effect_only(city):
    """The route depends on the seed (and edge) only: tilting the phone keeps it looking at the same points, so a
    sweep over one strength (the check sheet) shows the same place."""
    flat, tilted = _sim(city), _sim(city, tilt=1.0)
    assert all(np.allclose(flat.truth(i).center, tilted.truth(i).center, atol=1e-6) for i in range(len(flat)))


def test_tilt_makes_a_perspective_view_and_shake_moves_the_truth(city):
    flat, tilted, shaken = _sim(city), _sim(city, tilt=1.0), _sim(city, shake=1.0)
    pitch = [abs(tilted.truth(i).pitch_deg) + abs(tilted.truth(i).roll_deg) for i in range(len(tilted))]
    assert max(pitch) > 5 and all(abs(flat.truth(i).pitch_deg) < 1e-9 for i in range(len(flat)))
    assert max(abs(tilted.truth(i).H[2, :2]).max() for i in range(len(tilted))) > 1e-5   # not an affine view
    moved = [math.dist(flat.truth(i).center, shaken.truth(i).center) for i in range(len(flat))]
    assert 0 < max(moved) < 60                                          # a hand trembles; it does not fly away


def test_the_rolling_shutter_shears_the_frame_but_the_centre_rows_hold(city):
    sim = ScreenSim(city, seconds=2.0, seed=0, effects={"rolling_shutter": 1.0}, speed_mm_s=120.0, **KW)
    h = CAM.size[1]
    i = 10
    ref, img = _gray(render_cutout(city, sim.truth(i).H, CAM.size)), _gray(sim.frame(i))
    # the middle 10 rows: across a wider band the shutter itself stretches rows by up to ~6 px at this speed
    mid, top = slice(h // 2 - 5, h // 2 + 5), slice(0, int(h * 0.2))
    assert _shift(ref[mid], img[mid])[0] <= 1.0          # the truth is the centre row's moment
    assert _shift(ref[top], img[top])[0] > 3.0           # the top row was read ~45 ms earlier, at 120 mm/s


# --- each effect does what it should -------------------------------------------------------------------------

def test_all_effects_off_is_a_clean_view(grey):
    img = _sim(grey).frame(5)
    assert img.dtype == np.uint8 and img.shape == (180, 320, 3)
    assert img.std() < 1.0 and abs(img.mean() - 128) < 3


def test_the_pixel_grid_makes_moire(grey):
    assert _sim(grey, grid=1.0).frame(5).std() > 2.0


def test_glare_is_a_bright_spot_that_moves_with_the_camera(grey):
    sim = _sim(grey, glare=1.0)
    base = _gray(_sim(grey).frame(0))
    spots = []
    for i in (0, len(sim) - 1):
        diff = _gray(sim.frame(i)) - base
        assert diff.max() > 40
        spots.append(np.unravel_index(np.argmax(cv2.GaussianBlur(diff, (0, 0), 3)), diff.shape))
    assert math.dist(*spots) > 3                         # a reflection: it moves when the phone moves


def test_banding_rolls_across_rows(grey):
    img = _gray(_sim(grey, banding=1.0).frame(5))
    assert img.mean(axis=1).std() > 2.0 and img.mean(axis=0).std() < 1.0


def test_noise_is_noise(grey):
    a, b = _gray(_sim(grey, noise=1.0).frame(5)), _gray(_sim(grey).frame(5))
    assert (a - b).std() > 1.5 and abs((a - b).mean()) < 1.0


def test_auto_exposure_drifts_the_brightness(grey):
    on, off = _sim(grey, exposure=1.0), _sim(grey)
    assert np.std([on.frame(i).mean() for i in range(len(on))]) > 1.0
    assert np.std([off.frame(i).mean() for i in range(len(off))]) < 0.2


def test_motion_blur_softens_and_keeps_the_truth(city):
    sharp, blurred = _sim(city), _sim(city, motion_blur=1.0)
    lap = [cv2.Laplacian(_gray(s.frame(10)), cv2.CV_32F).var() for s in (sharp, blurred)]
    assert lap[1] < 0.8 * lap[0]
    assert np.allclose(sharp.truth(10).H, blurred.truth(10).H)


def test_edge_lets_the_view_run_past_the_screen(city):
    def past(sim) -> int:
        w, h = sim.window_size                                   # the map px shown on the screen
        x0, y0 = sim.window_origin
        return sum(bool(((sim.truth(i).footprint < (x0, y0)) | (sim.truth(i).footprint > (x0 + w, y0 + h))).any())
                   for i in range(len(sim)))
    # a 1200 x 800 screen: the route reaches its edges within 6 s whatever the seed (on the big test screen a
    # route can stay central that long)
    kw = {**KW, "screen": Screen(size=(1200, 800))}
    for seed in (0, 1, 2):
        assert past(ScreenSim(city, seconds=6.0, seed=seed, **kw)) == 0
        assert past(ScreenSim(city, seconds=6.0, seed=seed, effects={"edge": 1.0}, **kw)) > 0


def test_a_covered_lens_is_dark_and_flagged(city):
    sim = _sim(city, seconds=6.0, cover=1.0)
    covered = [i for i in range(len(sim)) if sim.covered(i)]
    assert 10 <= len(covered) <= 40                              # ~3 s at strength 1, inside the take
    assert covered == list(range(covered[0], covered[-1] + 1))   # one stretch
    assert all(sim.frame(i).mean() < 40 for i in covered)
    assert not any(_sim(city, seconds=6.0).covered(i) for i in range(60))


def test_the_same_seed_gives_the_same_take(city):
    a, b, c = _sim(city, seed=4, shake=0.5, noise=0.5), _sim(city, seed=4, shake=0.5, noise=0.5), \
        _sim(city, seed=5, shake=0.5, noise=0.5)
    assert np.array_equal(a.frame(7), b.frame(7)) and np.allclose(a.truth(7).H, b.truth(7).H)
    assert a.frame(3) is not None and not np.allclose(a.truth(7).H, c.truth(7).H)
    assert np.array_equal(a.frame(7), _sim(city, seed=4, shake=0.5, noise=0.5).frame(7))   # random access


# --- presets and settings ------------------------------------------------------------------------------------

def test_presets_mirror_the_real_takes():
    assert set(PRESETS) == {"calm", "hard", "jam"}
    for p in PRESETS.values():
        assert set(p) == set(EFFECTS) and all(0 <= v <= 1 for v in p.values())
    assert PRESETS["calm"]["cover"] == 0 and PRESETS["jam"]["cover"] > 0
    assert PRESETS["hard"]["tilt"] > PRESETS["calm"]["tilt"] and PRESETS["hard"]["glare"] > PRESETS["calm"]["glare"]
    e = effects_for("calm", {"glare": 0.9})
    assert e["glare"] == 0.9 and e["tilt"] == PRESETS["calm"]["tilt"]


@pytest.mark.parametrize("call", [
    lambda m: effects_for("windy"),                                         # unknown preset
    lambda m: effects_for("calm", {"sparkle": 0.5}),                        # unknown effect
    lambda m: effects_for("calm", {"glare": 1.5}),                          # strength out of [0, 1]
    lambda m: effects_for("calm", {"glare": math.nan}),
    lambda m: ScreenSim(m, seconds=0, **KW),                                # no duration
    lambda m: ScreenSim(m[:200, :200], seconds=2, **KW),                    # no room for a route
], ids=["preset", "effect", "strength", "nan", "seconds", "small-map"])
def test_bad_settings_are_clear_errors(city, call):
    with pytest.raises(ValueError) as e:
        call(city)
    assert str(e.value)


# --- the output: an Irpin-style flight -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def take(tmp_path_factory, city):
    d = tmp_path_factory.mktemp("take")
    cv2.imwrite(str(d / "map.png"), city)
    out = write_screen_video(d / "map.png", d / "take", preset="jam", seconds=3.0, seed=1, **KW)
    return out


def test_the_take_is_an_irpin_style_flight(take):
    for name in ("flight.mp4", "truth.jsonl", "meta.json"):
        assert (take / name).is_file(), name
    meta = json.loads((take / "meta.json").read_text())
    assert meta["simulated"] is True and meta["fps"] == 10 and meta["frames"] == 30 and meta["preset"] == "jam"
    assert set(meta["effects"]) == set(EFFECTS) and meta["seed"] == 1
    rows = [json.loads(line) for line in (take / "truth.jsonl").read_text().splitlines()]
    assert [r["i"] for r in rows] == list(range(30)) and all("lens_covered" in r for r in rows)
    assert any(r["lens_covered"] for r in rows)


def test_every_reader_reads_the_take(take):
    flight = FlightSource(take, fps=10)
    frames = list(flight)
    assert len(frames) == 30 and frames[0].image.shape == (180, 320, 3)
    assert np.allclose(flight.truth(12).center, json.loads((take / "truth.jsonl").read_text().splitlines()[12])
                       ["center_map_px"])
    keys = list(FlightSource(take, fps=1, keyframes=True))
    assert [f.idx for f in keys] == [0, 10, 20]
    video = VideoSource(take / "flight.mp4")
    assert len(video) == 30 and video.fps == pytest.approx(10)
    folder = FolderSource(take / "frames")
    assert folder.idxs == [0, 10, 20] and folder.fps == pytest.approx(10)


def test_compression_is_the_videos_crf(tmp_path, city):
    cv2.imwrite(str(tmp_path / "map.png"), city)
    sizes = {}
    for s in (0.0, 1.0):
        out = write_screen_video(tmp_path / "map.png", tmp_path / f"c{s}", preset="calm", seconds=1.0,
                                 effects={"compression": s}, **KW)
        sizes[s] = (out / "flight.mp4").stat().st_size
        assert json.loads((out / "meta.json").read_text())["crf"] == (16 if s == 0 else 36)
    assert sizes[1.0] < sizes[0.0]


def test_a_missing_map_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_screen_video(tmp_path / "nope.png", tmp_path / "out", **KW)
    assert not (tmp_path / "out").exists()


# --- the script ----------------------------------------------------------------------------------------------

def _cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], check=False, capture_output=True,
                          text=True, cwd=ROOT)


def test_the_script_writes_a_take(tmp_path, city):
    cv2.imwrite(str(tmp_path / "map.png"), city)
    r = _cli("--map", tmp_path / "map.png", "--out", tmp_path / "take", "--preset", "hard", "--seconds", 1,
             "--size", "320x180", "--fps", 10, "--screen", "2400x1600", "--distance", DIST, "--effect", "glare=0.2")
    assert r.returncode == 0, r.stderr
    meta = json.loads((tmp_path / "take" / "meta.json").read_text())
    assert meta["preset"] == "hard" and meta["effects"]["glare"] == 0.2 and meta["frames"] == 10


def test_the_script_draws_the_check_sheet(tmp_path, city):
    cv2.imwrite(str(tmp_path / "map.png"), city)
    r = _cli("--map", tmp_path / "map.png", "--sheet", tmp_path / "sheet.png")
    assert r.returncode == 0, r.stderr
    sheet = cv2.imread(str(tmp_path / "sheet.png"))
    assert sheet is not None and sheet.shape[0] > sheet.shape[1] / 2      # one row per effect: tall


@pytest.mark.parametrize("args", [
    ["--preset", "windy"], ["--effect", "glare"], ["--effect", "sparkle=1"], ["--size", "big"], ["--seconds", "-1"],
], ids=["preset", "effect-form", "effect-name", "size", "seconds"])
def test_the_script_fails_clearly(tmp_path, city, args):
    cv2.imwrite(str(tmp_path / "map.png"), city)
    r = _cli("--map", tmp_path / "map.png", "--out", tmp_path / "take", *args)
    assert r.returncode == 2 and r.stderr.strip() and "Traceback" not in r.stderr
    assert not (tmp_path / "take").exists()


def test_the_readme_says_it_is_a_simulation():
    text = (ROOT / "README.md").read_text()
    i = text.find("(M2-08)")
    assert i >= 0, "no M2-08 section in the README"
    section = text[i:i + 3000].lower()
    assert "simulat" in section and "m2-01" in section
