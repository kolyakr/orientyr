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

def _corner_error(ref: np.ndarray, img: np.ndarray) -> float:
    """Register img to ref with a full homography (ECC) and return how far it moves the frame corners, px: a
    translation-only check would miss a truth off in scale or perspective (review of M2-08)."""
    warp = np.eye(3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    _, warp = cv2.findTransformECC(ref, img, warp, cv2.MOTION_HOMOGRAPHY, crit, None, 5)
    h, w = ref.shape
    c = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
    return float(np.hypot(*(cv2.perspectiveTransform(c, warp.astype(np.float64)) - c)[0].T).max())


def test_the_truth_registers_every_frame_within_1_px(city):
    sim = _sim(city, tilt=1.0, shake=1.0)
    assert len(sim) == 20 and sim.fps == 10.0 and sim.view_contained
    for i in range(0, len(sim), 3):
        t = sim.truth(i)
        ref = render_cutout(city, t.H, CAM.size)                     # the map as the truth says the frame sees it
        img = sim.frame(i)
        d, resp = _shift(_gray(ref), _gray(img))
        # measured ~0.05 px; 1 px is the spec. resp: the same content, not a lucky peak on a different view
        assert d <= 1.0 and resp > 0.3, f"frame {i}: shift {d:.2f} px, response {resp:.2f}"
        e = _corner_error(_gray(ref), _gray(img))
        assert e <= 1.0, f"frame {i}: the corners are {e:.2f} px off"            # measured <= 0.06 px


def test_the_corner_check_catches_a_truth_off_in_scale(city):
    """The check above can fail: a truth scaled 2 % about the frame centre (~3.7 px at the corners) is caught."""
    sim = _sim(city)
    t = sim.truth(5)
    w, h = CAM.size
    S = np.array([[1.02, 0, -0.02 * w / 2], [0, 1.02, -0.02 * h / 2], [0, 0, 1.0]])
    assert _corner_error(_gray(render_cutout(city, t.H @ S, CAM.size)), _gray(sim.frame(5))) > 2.0


def test_pitch_and_roll_mean_what_they_mean_in_the_irpin_flights(city):
    """In the Irpin truth, pitch > 0 makes the top of the frame the far side (more map px per frame px there) and
    roll > 0 the right side: all 67 sampled records agree. The simulator must mean the same (review of M2-08)."""
    from dronenav.sim.truth import centre_scale
    sim = _sim(city, seconds=4.0, tilt=1.0)
    w, h = CAM.size
    checked = 0
    for i in range(len(sim)):
        t = sim.truth(i)
        if abs(t.pitch_deg) > 2:
            assert (t.pitch_deg > 0) == (centre_scale(t.H, w / 2, 0) > centre_scale(t.H, w / 2, h)), i
            checked += 1
        if abs(t.roll_deg) > 2:
            assert (t.roll_deg > 0) == (centre_scale(t.H, w, h / 2) > centre_scale(t.H, 0, h / 2)), i
            checked += 1
    assert checked > 20


def test_altitude_is_the_perpendicular_distance_to_the_screen(city):
    sim = _sim(city, tilt=1.0)
    for i in (0, 10):
        _, C, _ = sim._camera(i / sim.fps)
        assert sim.truth(i).altitude_m == pytest.approx(-C[2] / 1000, rel=1e-9)   # not the slant range


def test_frame_0_moves_like_any_other(city):
    """Time before 0 exists: frame 0's speed is the route's, and its rolling shutter shears it (it used to be
    clamped at t = 0, review of M2-08)."""
    sim = ScreenSim(city, seconds=2.0, seed=0, effects={"rolling_shutter": 1.0}, speed_mm_s=120.0, **KW)
    assert sim.truth(0).speed_mps == pytest.approx(sim.truth(1).speed_mps, rel=0.15)
    h = CAM.size[1]
    ref, img = _gray(render_cutout(city, sim.truth(0).H, CAM.size)), _gray(sim.frame(0))
    assert _shift(ref[: int(h * 0.2)], img[: int(h * 0.2)])[0] > 3.0


def test_truth_outside_the_take_is_an_error(city):
    sim = _sim(city)
    with pytest.raises(IndexError):
        sim.truth(len(sim))
    with pytest.raises(IndexError):
        sim.truth(-1)


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
    # the blur is centred on the frame's moment: the blurred frame still sits where its truth says
    ref = _gray(render_cutout(city, blurred.truth(10).H, CAM.size))
    assert _shift(ref, _gray(blurred.frame(10)))[0] <= 1.0


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
    def make(seed=4):
        return _sim(city, seed=seed, shake=0.5, noise=0.5, glare=0.5, exposure=0.5)
    a, b = make(), make()
    first = a.frame(7)                                    # a: frame 7 first
    for i in (3, 12, 0):
        b.frame(i)                                        # b: other frames first, then 7
    assert np.array_equal(first, b.frame(7)) and np.allclose(a.truth(7).H, b.truth(7).H)
    assert not np.allclose(a.truth(7).H, make(5).truth(7).H)


@pytest.mark.parametrize("name", ["shake", "tilt", "grid", "glare", "motion_blur", "rolling_shutter", "exposure",
                                  "banding", "noise"])
def test_strength_0_is_the_effect_off(city, name):
    """Strength 0 is the clean frame, and a tiny strength is next to it (no jump at 0). Not for edge (it widens
    the route's box), cover (a window in time) or compression (CRF 16 at 0: near-lossless, not lossless)."""
    clean, tiny = _sim(city).frame(9).astype(int), _sim(city, **{name: 1e-4}).frame(9).astype(int)
    assert np.array_equal(clean, _sim(city, **{name: 0.0}).frame(9))
    assert np.abs(clean - tiny).mean() < 0.5


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


def test_keyframes_follow_the_readers_rounding(tmp_path, city):
    """At 12.5 fps a keyframe is every 13th frame (round half up, as FlightSource and VideoSource do)."""
    cv2.imwrite(str(tmp_path / "map.png"), city)
    cam = Camera(size=(320, 180), fps=12.5)
    out = write_screen_video(tmp_path / "map.png", tmp_path / "take", seconds=3.0, camera=cam,
                             screen=SCREEN, distance_mm=DIST)
    assert FolderSource(out / "frames").idxs == [0, 13, 26]
    assert [f.idx for f in FlightSource(out, fps=1, keyframes=True)] == [0, 13, 26]


def test_an_existing_folder_is_replaced_only_if_it_is_a_simulated_take(tmp_path, city):
    cv2.imwrite(str(tmp_path / "map.png"), city)
    real = tmp_path / "flight"                            # e.g. a real flight's folder: never overwritten
    real.mkdir()
    (real / "meta.json").write_text('{"fps": 30}')
    (real / "flight.mp4").write_bytes(b"real")
    with pytest.raises(ValueError, match="simulated"):
        write_screen_video(tmp_path / "map.png", real, seconds=1.0, **KW)
    assert (real / "flight.mp4").read_bytes() == b"real" and not (real / "truth.jsonl").exists()
    take = write_screen_video(tmp_path / "map.png", tmp_path / "take", seconds=2.0, **KW)
    (take / "frames" / "frame_99999.png").write_bytes(b"old")       # left from a longer take
    write_screen_video(tmp_path / "map.png", take, seconds=1.0, **KW)
    assert sorted(p.name for p in take.iterdir()) == ["flight.mp4", "frames", "meta.json", "truth.jsonl"]
    assert FolderSource(take / "frames").idxs == [0]                # the old take is gone, whole
    assert not any(p.name.startswith(".") for p in tmp_path.iterdir())


def test_the_crf_is_recorded_only_when_x264_writes_the_video(tmp_path, city, monkeypatch):
    from dronenav.sim import screen
    cv2.imwrite(str(tmp_path / "map.png"), city)
    monkeypatch.setattr(screen, "_has_x264", lambda: False)
    out = write_screen_video(tmp_path / "map.png", tmp_path / "take", seconds=1.0, **KW)
    assert json.loads((out / "meta.json").read_text())["crf"] is None


def test_bad_seeds_and_sizes_are_clear_errors(city):
    with pytest.raises(ValueError, match="seed"):
        ScreenSim(city, seconds=1.0, seed=-1, **KW)
    for cam in (Camera(size=(321, 180), fps=10.0), Camera(size=(320, 180), hfov_deg=0.0, fps=10.0),
                Camera(size=(320, 180), hfov_deg=180.0, fps=10.0)):
        with pytest.raises(ValueError):
            ScreenSim(city, seconds=1.0, screen=SCREEN, camera=cam, distance_mm=DIST)
    with pytest.raises(ValueError):
        ScreenSim(city, seconds=1.0, screen=Screen(size=(0, 800)), camera=CAM, distance_mm=DIST)


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


@pytest.mark.parametrize(("args", "says"), [
    (["--preset", "windy"], "windy"), (["--effect", "glare"], "NAME=STRENGTH"), (["--effect", "sparkle=1"], "sparkle"),
    (["--size", "big"], "WxH"), (["--seconds", "-1"], "seconds"),
    (["--size", "321x181"], "even"),           # H.264 needs even sizes: said at once, not by x264 at the end
    (["--seed", "-1"], "seed"),
], ids=["preset", "effect-form", "effect-name", "size", "seconds", "odd-size", "negative-seed"])
def test_the_script_fails_clearly(tmp_path, city, args, says):
    cv2.imwrite(str(tmp_path / "map.png"), city)
    r = _cli("--map", tmp_path / "map.png", "--out", tmp_path / "take", *args)
    assert r.returncode == 2 and says in r.stderr and "Traceback" not in r.stderr, r.stderr
    assert not (tmp_path / "take").exists()


def test_the_notes_say_it_is_a_simulation():
    text = (ROOT / "NOTES.md").read_text()
    i = text.find("(M2-08)")
    assert i >= 0, "no M2-08 section in NOTES.md"
    section = text[i:i + 3000].lower()
    assert "simulat" in section and "m2-01" in section
