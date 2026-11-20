"""M6-01 · Flight simulator: FlightSource (pre-rendered Irpin flights) and the cut-out generator.

Done when (issue spec update):
1. FlightSource frames, undistorted, match the map warped by their truth at NCC ≥ 0.9 on sampled frames, for the
   video and the keyframes, on the CI fixture, a synthetic flight and the real flights; without undistortion they
   match clearly worse;
2. t = i / meta.fps and equals the truth's t; the rate and time window select the expected frames;
3. each frame's truth is exposed and its extra() passes telemetry's JSON checks;
4. cut-out generator: the truth H maps frame pixels back onto the map within 0.5 px (tilt included); same seed →
   byte-identical frames and routes, other seed → other route; routes stay inside the map.
"""
import itertools
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.sim import (
    CutoutPose,
    CutoutSource,
    FlightSource,
    FlightTruth,
    cutout_homography,
    cutout_route,
    render_cutout,
)
from dronenav.telemetry import Record, record_to_dict
from dronenav.types import FrameSource, TrackState

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "irpin_mild_2025"
FLIGHTS = ROOT / "data" / "irpin" / "flights"
MAPS = ROOT / "data" / "irpin" / "maps"


# --- the measure: NCC of a frame warped by its truth against the map (as tests/test_fixtures_irpin.py) ---------

def _ncc(a, b, mask):
    a, b = a[mask].astype(float), b[mask].astype(float)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def registration_ncc(frame_bgr, H_frame_to_map, map_bgr) -> float:
    """Warp the frame onto the map by H, blur both (σ 1.5: structure, not sensor grain), NCC over the eroded area
    the frame covers. Works on a window around the footprint, so a 3936 px map stays cheap."""
    h, w = frame_bgr.shape[:2]
    corners = cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), H_frame_to_map)[0]
    x0, y0 = np.floor(corners.min(axis=0)).astype(int) - 4
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int) + 4
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, map_bgr.shape[1]), min(y1, map_bgr.shape[0])
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    size = (x1 - x0, y1 - y0)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    warped = cv2.warpPerspective(gray, shift @ H_frame_to_map, size, flags=cv2.INTER_AREA)
    cover = cv2.warpPerspective(np.full(gray.shape, 255, np.uint8), shift @ H_frame_to_map, size,
                                flags=cv2.INTER_NEAREST)
    crop = cv2.cvtColor(map_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    mask = cv2.erode(cover, np.ones((9, 9), np.uint8)) > 0
    return _ncc(cv2.GaussianBlur(warped, (0, 0), 1.5), cv2.GaussianBlur(crop, (0, 0), 1.5), mask)


# --- a flight made from the CI fixture: real keyframes and truth, keyframes only --------------------------------

@pytest.fixture(scope="module")
def fixture_flight(tmp_path_factory):
    """The CI fixture laid out as a flight folder (keyframes only), with its map crop as a small map: truth moved
    from full-map px to crop px."""
    fx = json.loads((FIX / "fixture.json").read_text())
    d = tmp_path_factory.mktemp("fixture_flight")
    (d / "frames").mkdir()
    x0, y0 = fx["crop"]["origin_px"]
    with open(d / "truth.jsonl", "w") as out:
        for ln in (FIX / "truth.jsonl").read_text().splitlines():
            r = json.loads(ln)
            r["H_frame_to_map"] = (np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]]) @ np.array(r["H_frame_to_map"])).tolist()
            r["center_map_px"] = [r["center_map_px"][0] - x0, r["center_map_px"][1] - y0]
            r["footprint_map_px"] = [[x - x0, y - y0] for x, y in r["footprint_map_px"]]
            out.write(json.dumps(r) + "\n")
            shutil.copy(FIX / f"frame_{r['i']:06d}.jpg", d / "frames" / f"frame_{r['i']:06d}.jpg")
    (d / "meta.json").write_text(json.dumps({"fps": fx["source"]["fps"], "frames": 13260, "camera": fx["camera"]}))
    return d, cv2.imread(str(FIX / fx["crop"]["file"]))


def test_fixture_keyframes_match_the_map_through_their_truth(fixture_flight):
    d, crop = fixture_flight
    src = FlightSource(d, keyframes=True)
    frames = list(src)
    assert [f.idx for f in frames] == [6150, 6300, 6450]
    for f in frames:
        assert registration_ncc(f.image, src.truth(f.idx).H, crop) >= 0.9, f.idx


def test_without_undistortion_the_match_is_clearly_worse(fixture_flight):
    d, crop = fixture_flight
    good = [registration_ncc(f.image, FlightSource(d, keyframes=True).truth(f.idx).H, crop)
            for f in FlightSource(d, keyframes=True)]
    raw_src = FlightSource(d, keyframes=True, undistort=False)
    raw = [registration_ncc(f.image, raw_src.truth(f.idx).H, crop) for f in raw_src]
    assert all(r < g - 0.03 for r, g in zip(raw, good, strict=True)), (raw, good)


def test_times_come_from_meta_and_equal_the_truth(fixture_flight):
    d, _ = fixture_flight
    stored = {r["i"]: r["t"] for r in map(json.loads, (d / "truth.jsonl").read_text().splitlines())}
    src = FlightSource(d, keyframes=True)
    for f in src:
        # against the record as stored: a truth that recomputed t from i would agree with anything
        assert f.t == pytest.approx(f.idx / 30) and f.t == pytest.approx(stored[f.idx])
        assert src.truth(f.idx).t == pytest.approx(stored[f.idx])


def test_the_truth_feeds_the_telemetry(fixture_flight):
    d, _ = fixture_flight
    src = FlightSource(d, keyframes=True)
    f = next(iter(src))
    tr = src.truth(f.idx)
    assert isinstance(tr, FlightTruth)
    assert tr.H.shape == (3, 3) and tr.footprint.shape == (4, 2) and len(tr.center) == 2
    assert {"x", "y", "heading_deg", "altitude_m"} <= tr.extra().keys()
    rec = Record(idx=f.idx, t=f.t, source=f.source, extra={"truth": tr.extra()},
                 track=TrackState(idx=f.idx, state="TRACKING", x=tr.center[0], y=tr.center[1], vx=0, vy=0,
                                  cov=np.eye(2)))
    assert record_to_dict(rec)["extra"]["truth"]["x"] == pytest.approx(tr.center[0])


def test_it_is_a_frame_source(fixture_flight):
    assert isinstance(FlightSource(fixture_flight[0], keyframes=True), FrameSource)


# --- a synthetic flight: the video path, with real lens distortion ---------------------------------------------

def _texture(w, h, seed=3):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.uniform(0, 255, (h, w, 3)).astype(np.float32), (0, 0), 3)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _nadir_H(x, y, heading, scale, w, h):
    """Frame px -> map px, independent of the module under test (conventions of dronenav.types)."""
    a = math.radians(heading)
    sR = scale * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    t = np.array([x, y]) - sR @ np.array([w / 2, h / 2])
    return np.array([[*sR[0], t[0]], [*sR[1], t[1]], [0, 0, 1.0]])


def _distort(undistorted, K, dist):
    """The lens: each distorted pixel shows the undistorted point it maps to (as the renderer does)."""
    h, w = undistorted.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1)[:, None, :]
    und = cv2.undistortPoints(pts, K, dist, P=K).reshape(h, w, 2)
    return cv2.remap(undistorted, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


@pytest.fixture(scope="module")
def synthetic_flight(tmp_path_factory):
    """20 frames at 10 fps over a textured map, lens-distorted, as flight.mp4 + truth.jsonl + meta.json."""
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    d = tmp_path_factory.mktemp("synthetic_flight")
    w, h, fps, n = 320, 240, 10, 20
    K = np.array([[230.0, 0, w / 2], [0, 230.0, h / 2], [0, 0, 1]])
    dist = np.array([-0.18, 0.04, 0, 0, 0])
    world = _texture(1600, 1200)
    shift = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])        # pixel-centre conventions, as in the renderers
    unshift = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])
    writer = imageio_ffmpeg.write_frames(str(d / "flight.mp4"), (w, h), fps=fps, codec="libx264", quality=None,
                                         output_params=["-crf", "12", "-pix_fmt", "yuv420p"], pix_fmt_in="bgr24",
                                         macro_block_size=1)
    writer.send(None)
    with open(d / "truth.jsonl", "w") as out:
        for i in range(n):
            x, y, head = 500 + 25 * i, 600 + 8 * i, 30 + 3 * i
            H = _nadir_H(x, y, head, 1.2, w, h)
            und = cv2.warpPerspective(world, unshift @ H @ shift, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            writer.send(np.ascontiguousarray(_distort(und, K, dist)))
            fp = cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), H)[0]
            out.write(json.dumps({"i": i, "t": i / fps, "H_frame_to_map": H.tolist(), "center_map_px": [x, y],
                                  "footprint_map_px": fp.tolist(), "heading_deg": head, "altitude_m": 100.0,
                                  "pitch_deg": 0.0, "roll_deg": 0.0, "speed_mps": 5.0}) + "\n")
    writer.close()
    (d / "meta.json").write_text(json.dumps({"fps": fps, "frames": n,
                                             "camera": {"size": [w, h], "K": K.tolist(), "dist": dist.tolist()}}))
    return d, world


def test_video_frames_match_the_map_through_their_truth(synthetic_flight):
    d, world = synthetic_flight
    src = FlightSource(d, fps=10)
    frames = list(src)
    assert [f.idx for f in frames] == list(range(20))
    for f in frames[::4]:
        assert f.image.shape == (240, 320, 3)
        assert registration_ncc(f.image, src.truth(f.idx).H, world) >= 0.9, f.idx


def test_video_without_undistortion_matches_worse(synthetic_flight):
    d, world = synthetic_flight
    good = FlightSource(d, fps=10)
    raw = FlightSource(d, fps=10, undistort=False)
    for fg, fr in list(zip(good, raw, strict=True))[::5]:
        assert registration_ncc(fr.image, raw.truth(fr.idx).H, world) < \
            registration_ncc(fg.image, good.truth(fg.idx).H, world) - 0.03, fg.idx


@pytest.mark.parametrize("fps, idx", [(10, list(range(20))), (5, list(range(0, 20, 2))), (2.5, [0, 4, 8, 12, 16]),
                                      (3, [0, 3, 6, 9, 12, 15, 18])])
def test_the_rate_keeps_every_nth_frame(synthetic_flight, fps, idx):
    d, _ = synthetic_flight
    frames = list(FlightSource(d, fps=fps))
    assert [f.idx for f in frames] == idx
    assert [f.t for f in frames] == pytest.approx([i / 10 for i in idx])


def test_a_time_window_limits_the_frames(synthetic_flight):
    d, _ = synthetic_flight
    assert [f.idx for f in FlightSource(d, fps=10, start_s=0.5, end_s=1.2)] == list(range(5, 13))


@pytest.mark.parametrize("kw, word", [({"fps": 0}, "fps"), ({"fps": -1}, "fps"), ({"fps": 20}, "fps"),
                                      ({"start_s": 2, "end_s": 1}, "start_s")])
def test_bad_rates_and_windows_are_refused(synthetic_flight, kw, word):
    with pytest.raises(ValueError, match=word):
        FlightSource(synthetic_flight[0], **kw)


@pytest.mark.parametrize("missing", ["meta.json", "truth.jsonl", "flight.mp4"])
def test_a_missing_flight_file_is_named(synthetic_flight, tmp_path, missing):
    d = tmp_path / "flight"
    shutil.copytree(synthetic_flight[0], d)
    (d / missing).unlink()
    with pytest.raises(FileNotFoundError, match=missing.replace(".", r"\.")):
        FlightSource(d)


def test_keyframes_need_a_frames_folder(synthetic_flight):
    with pytest.raises(FileNotFoundError, match="frames"):
        FlightSource(synthetic_flight[0], keyframes=True)


def test_a_frame_without_truth_is_a_clear_error(synthetic_flight):
    with pytest.raises(KeyError, match="999"):
        FlightSource(synthetic_flight[0]).truth(999)


# --- the real flights, when data/ is present -------------------------------------------------------------------

def _flight(name):
    d = FLIGHTS / name
    return d if (d / "flight.mp4").exists() else None


@pytest.mark.data
@pytest.mark.parametrize("name, year", [("harsh/2017", 2017), ("mild/2025", 2025)])
def test_real_flight_video_matches_its_map(name, year):
    d = _flight(name)
    if d is None:
        pytest.skip(f"{name} not in data/")
    world = cv2.imread(str(MAPS / f"irpin_{year}.png"))
    src = FlightSource(d, fps=1, end_s=12)                         # sequential decode: keep it short
    frames = list(src)
    assert [f.idx for f in frames] == list(range(0, 361, 30))
    for f in frames[::3]:
        assert registration_ncc(f.image, src.truth(f.idx).H, world) >= 0.9, (name, f.idx)
        assert f.t == pytest.approx(src.truth(f.idx).t)


@pytest.mark.data
def test_real_keyframes_match_their_map():
    d = _flight("harsh/2017")
    if d is None or not (d / "frames").is_dir():
        pytest.skip("harsh/2017 keyframes not in data/")
    world = cv2.imread(str(MAPS / "irpin_2017.png"))
    src = FlightSource(d, keyframes=True, end_s=40)
    for f in list(src)[::10]:
        assert registration_ncc(f.image, src.truth(f.idx).H, world) >= 0.9, f.idx


# --- the cut-out generator -------------------------------------------------------------------------------------

def _dots(w, h, points, sigma=1.6):
    """A dark map with small bright blobs at known sub-pixel map positions (pixel centres at +0.5)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    img = np.zeros((h, w), np.float32)
    for x, y in points:
        img += np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
    return cv2.cvtColor(np.clip(img * 255, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _centroids(img):
    """Intensity centroids of the blobs, skipping blobs the frame edge cuts (their centroid is pulled inwards)."""
    g = img[..., 0].astype(np.float32)
    n, labels = cv2.connectedComponents((g > 20).astype(np.uint8))
    ys, xs = np.mgrid[0:g.shape[0], 0:g.shape[1]].astype(np.float32) + 0.5
    out = []
    for k in range(1, n):
        m = labels == k
        r, c = np.nonzero(m)
        if r.min() == 0 or c.min() == 0 or r.max() == g.shape[0] - 1 or c.max() == g.shape[1] - 1:
            continue
        out.append(((xs * g)[m].sum() / g[m].sum(), (ys * g)[m].sum() / g[m].sum()))
    return out


@pytest.mark.parametrize("tilt, heading, scale", [(0.0, 37.0, 0.9), (6.0, 37.0, 0.9), (12.0, 37.0, 0.9),
                                                  (0.0, 180.0, 1.6), (8.0, 180.0, 1.6)])
def test_the_truth_maps_frame_pixels_back_onto_the_map_within_half_a_pixel(tilt, heading, scale):
    # heading 180° and scale 1.6: where dropping the half-pixel conventions would move every point by ~1 px
    # dots on a grid 24 px apart (sub-pixel offsets): random dots merged into one blob and moved its centroid
    dots = [(380.3 + 24 * i, 380.7 + 24 * j) for i in range(11) for j in range(11)]
    world = _dots(1000, 1000, dots)
    pose = CutoutPose(x=500.3, y=499.7, heading_deg=heading, scale=scale, tilt_deg=tilt, tilt_dir_deg=60.0)
    H = cutout_homography(pose, (320, 240))
    frame = render_cutout(world, H, (320, 240))
    found = _centroids(frame)
    assert len(found) >= 4
    for cx, cy in found:
        mx, my = cv2.perspectiveTransform(np.float32([[[cx, cy]]]), H)[0, 0]
        err = min(math.hypot(mx - x, my - y) for x, y in dots)
        assert err <= 0.5, (tilt, err)                               # the done-when tolerance


def test_the_pose_is_the_frame_centre_with_its_heading_and_scale():
    pose = CutoutPose(x=400, y=300, heading_deg=90.0, scale=2.0)
    H = cutout_homography(pose, (320, 240))
    c = cv2.perspectiveTransform(np.float32([[[160, 120]]]), H)[0, 0]
    up = cv2.perspectiveTransform(np.float32([[[160, 119]]]), H)[0, 0] - c
    assert c == pytest.approx([400, 300], abs=1e-4)
    assert up == pytest.approx([2.0, 0.0], abs=1e-4)                 # frame up -> east (90°), 2 map px per frame px
    assert np.allclose(H[2, :2], 0)                                  # no tilt: no perspective


def test_tilt_adds_a_mild_perspective():
    H = cutout_homography(CutoutPose(x=400, y=300, heading_deg=0, scale=1, tilt_deg=10, tilt_dir_deg=0), (320, 240))
    assert not np.allclose(H[2, :2], 0)
    fp = cv2.perspectiveTransform(np.float32([[[0, 0], [320, 0], [320, 240], [0, 240]]]), H)[0]
    top, bottom = np.hypot(*(fp[1] - fp[0])), np.hypot(*(fp[2] - fp[3]))
    assert 1.02 < max(top, bottom) / min(top, bottom) < 1.5            # a trapezoid, not a wild warp


def test_routes_are_reproducible_from_a_seed_and_differ_between_seeds():
    a = cutout_route((2000, 1500), 30, seed=1)
    b = cutout_route((2000, 1500), 30, seed=1)
    c = cutout_route((2000, 1500), 30, seed=2)
    assert a == b and a != c


def test_frames_are_byte_identical_for_the_same_seed():
    world = _texture(1200, 900)
    first = [f.image.tobytes() for f in CutoutSource(world, cutout_route((1200, 900), 8, seed=4), (160, 120))]
    second = [f.image.tobytes() for f in CutoutSource(world, cutout_route((1200, 900), 8, seed=4), (160, 120))]
    assert first == second


@pytest.mark.parametrize("seed", range(10))
def test_routes_stay_inside_the_map_and_are_smooth(seed):
    size, frame = (2000, 1500), (320, 240)
    poses = cutout_route(size, 40, seed=seed, frame_size=frame)
    for p in poses:
        fp = cv2.perspectiveTransform(np.float32([[[0, 0], [320, 0], [320, 240], [0, 240]]]),
                                      cutout_homography(p, frame))[0]
        assert (fp >= 0).all() and (fp[:, 0] <= size[0]).all() and (fp[:, 1] <= size[1]).all(), (seed, p)
    steps = [math.hypot(b.x - a.x, b.y - a.y) for a, b in itertools.pairwise(poses)]
    assert min(steps) > 0 and max(steps) / min(steps) < 4
    turns = [abs((b.heading_deg - a.heading_deg + 180) % 360 - 180) for a, b in itertools.pairwise(poses)]
    assert max(turns) < 45


def test_the_cutout_source_carries_each_frames_truth():
    world = _texture(1200, 900)
    poses = cutout_route((1200, 900), 5, seed=0)
    src = CutoutSource(world, poses, (160, 120), fps=2.0)
    frames = list(src)
    assert isinstance(src, FrameSource)
    assert [f.idx for f in frames] == list(range(5)) and [f.t for f in frames] == [0, 0.5, 1.0, 1.5, 2.0]
    for f in frames:
        tr = src.truth(f.idx)
        assert np.allclose(tr.H, cutout_homography(poses[f.idx], (160, 120)))
        assert tr.center == pytest.approx((poses[f.idx].x, poses[f.idx].y)) and tr.footprint.shape == (4, 2)
        assert tr.extra()["x"] == pytest.approx(poses[f.idx].x)
        # a sanity check that the frame shows the right place (a misplaced one scores near 0): at scale s > 1 the
        # frame holds s² fewer pixels than its footprint, so warped back it is softer than the map: 0.97–1.0 here
        assert registration_ncc(f.image, tr.H, world) >= 0.95



def test_shrinking_the_map_does_not_alias():
    # scale 2: each frame pixel covers 2 x 2 map pixels; a 1 px checkerboard must come out as even grey, not moiré
    board = (np.indices((400, 400)).sum(axis=0) % 2 * 255).astype(np.uint8)
    world = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    for heading in (0.0, 17.0, 45.0):
        frame = render_cutout(world, cutout_homography(CutoutPose(200, 200, heading, 2.0), (64, 48)), (64, 48))
        assert frame.std() < 12 and abs(frame.mean() - 127.5) < 10, (heading, frame.std())


# --- review ----------------------------------------------------------------------------------------------------

def test_a_window_of_one_instant_is_allowed(synthetic_flight):
    # the window is inclusive (start_s <= t <= end_s): start_s == end_s asks for one frame, not an error
    assert [f.idx for f in FlightSource(synthetic_flight[0], fps=10, start_s=0, end_s=0)] == [0]
    assert [f.idx for f in FlightSource(synthetic_flight[0], fps=10, start_s=0.7, end_s=0.7)] == [7]



def test_a_tilted_pose_keeps_its_centre_and_scale():
    # with tilt the homography is recentred so the frame centre still lands on (x, y), and rescaled so the scale
    # at the centre is still pose.scale (the types.py convention: map px per frame px at the frame centre)
    for tilt, d in ((8.0, 0.0), (12.0, 75.0)):
        pose = CutoutPose(x=812.0, y=455.0, heading_deg=30.0, scale=1.3, tilt_deg=tilt, tilt_dir_deg=d)
        H = cutout_homography(pose, (320, 240))
        c = cv2.perspectiveTransform(np.float32([[[160, 120]]]), H)[0, 0]
        assert c == pytest.approx([812.0, 455.0], abs=1e-3)
        e = cv2.perspectiveTransform(np.float32([[[161, 120]], [[160, 121]]]), H)[:, 0] - c   # 1 px steps
        assert math.sqrt(abs(np.cross(e[0], e[1]))) == pytest.approx(1.3, rel=1e-3)


@pytest.mark.parametrize("where", [(400.0, 400.0), (2000.0, 1500.0), (6000.0, 5000.0)])
def test_shrinking_a_tilted_view_does_not_alias_anywhere_on_the_map(where):
    # the anti-aliasing scale came from H's top-left block, which with tilt depends on the map position
    board = (np.indices((200, 200)).sum(axis=0) % 2 * 255).astype(np.uint8)
    x0, y0 = int(where[0]) - 100, int(where[1]) - 100
    world = np.zeros((int(where[1]) + 200, int(where[0]) + 200, 3), np.uint8)
    world[y0:y0 + 200, x0:x0 + 200] = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    pose = CutoutPose(where[0], where[1], 17.0, 2.0, tilt_deg=8.0, tilt_dir_deg=40.0)
    frame = render_cutout(world, cutout_homography(pose, (48, 36)), (48, 36))
    assert frame.std() < 12, frame.std()


# (1200 x 900 with 320 px frames leaves a 350 px box: too tight to turn around in, refused, see the test below)
@pytest.mark.parametrize("size, frame", [((2000, 1500), (320, 240)), ((1200, 900), (160, 120))])
def test_routes_turn_before_the_edge_and_never_slide_along_it(size, frame):
    # every step full length and along the heading: a clipped step slid along the wall, sideways to the camera
    for seed in range(15):
        poses = cutout_route(size, 120, seed=seed, frame_size=frame)
        steps = [math.hypot(b.x - a.x, b.y - a.y) for a, b in itertools.pairwise(poses)]
        assert max(steps) - min(steps) < 1e-6, (seed, min(steps), max(steps))
        for a, b in itertools.pairwise(poses):
            moved = math.degrees(math.atan2(b.x - a.x, -(b.y - a.y))) % 360
            assert abs((moved - b.heading_deg + 180) % 360 - 180) < 1e-6, seed


def test_a_map_too_tight_to_turn_in_is_refused():
    # just wider than a footprint, but too narrow to turn around at 25° a frame
    with pytest.raises(ValueError, match="turn"):
        cutout_route((800, 700), 20, seed=0, frame_size=(320, 240))


@pytest.mark.parametrize("kw, word", [({"scale": (1.25, 0.8)}, "scale"), ({"scale": (0, 1)}, "scale"),
                                      ({"overlap": 1.0}, "overlap"), ({"overlap": -0.1}, "overlap"),
                                      ({"tilt_max_deg": -1}, "tilt"), ({"seed": None}, "seed"), ({"n": 0}, "n")])
def test_route_settings_that_cannot_work_are_refused(kw, word):
    args = {"map_size": (2000, 1500), "n": 10, "seed": 1, **kw}
    with pytest.raises(ValueError, match=word):
        cutout_route(args.pop("map_size"), args.pop("n"), **args)


def test_footprints_stay_inside_with_strong_tilt_on_a_small_map():
    # the margin must account for tilt: a margin from untilted frames let tilted corners leave the map
    for seed in range(20):
        for p in cutout_route((1500, 1150), 60, seed=seed, frame_size=(320, 240), tilt_max_deg=15.0):
            fp = cv2.perspectiveTransform(np.float32([[[0, 0], [320, 0], [320, 240], [0, 240]]]),
                                          cutout_homography(p, (320, 240)))[0]
            assert (fp >= 0).all() and (fp[:, 0] <= 1500).all() and (fp[:, 1] <= 1150).all(), seed


def test_a_cutout_source_needs_a_real_frame_rate():
    with pytest.raises(ValueError, match="fps"):
        CutoutSource(_texture(400, 300), [CutoutPose(200, 150, 0, 1)], (64, 48), fps=0)


@pytest.mark.parametrize("fps, idx", [(6, list(range(0, 20, 2))), (4, list(range(0, 20, 3)))])
def test_rates_between_steps_round_half_up_and_report_the_effective_rate(synthetic_flight, fps, idx):
    # 10 fps video: fps 6 -> every 2nd frame (5 fps), fps 4 -> 2.5 rounds up to every 3rd (3.33 fps)
    src = FlightSource(synthetic_flight[0], fps=fps)
    assert [f.idx for f in src] == idx and src.rate == pytest.approx(10 / (idx[1] - idx[0]))


@pytest.mark.parametrize("kw", [{"start_s": -1}, {"start_s": True}, {"end_s": True}, {"end_s": "5"},
                                {"end_s": float("nan")}, {"fps": float("inf")}])
def test_window_and_rate_values_that_are_not_times_are_refused(synthetic_flight, kw):
    with pytest.raises(ValueError):
        FlightSource(synthetic_flight[0], **kw)


def test_a_truncated_video_is_an_error_not_a_short_flight(synthetic_flight, tmp_path):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    d = tmp_path / "short"
    shutil.copytree(synthetic_flight[0], d)
    w = imageio_ffmpeg.write_frames(str(d / "flight.mp4"), (320, 240), fps=10, macro_block_size=1)
    w.send(None)
    for _ in range(5):
        w.send(np.zeros((240, 320, 3), np.uint8))
    w.close()                                                     # 5 frames, while meta and truth say 20
    with pytest.raises(ValueError, match=r"flight\.mp4.*5.*20"):
        list(FlightSource(d, fps=10))
    assert [f.idx for f in FlightSource(d, fps=10, end_s=0.3)] == [0, 1, 2, 3]   # a window inside the video is fine


def _edit_flight(src_dir, tmp_path, meta=None, truth=None):
    d = tmp_path / "edited"
    shutil.copytree(src_dir, d)
    if meta:
        m = json.loads((d / "meta.json").read_text())
        meta(m)
        (d / "meta.json").write_text(json.dumps(m))
    if truth:
        recs = [json.loads(ln) for ln in (d / "truth.jsonl").read_text().splitlines()]
        recs = truth(recs)
        (d / "truth.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return d


@pytest.mark.parametrize("what, edit, word", [
    ("meta without fps", {"meta": lambda m: m.pop("fps")}, "fps"),
    ("meta without camera", {"meta": lambda m: m.pop("camera")}, "camera"),
    ("truth without H", {"truth": lambda rs: [{k: v for k, v in r.items() if k != "H_frame_to_map"} for r in rs]},
     "H_frame_to_map"),
    ("truth without altitude", {"truth": lambda rs: [{k: v for k, v in r.items() if k != "altitude_m"} for r in rs]},
     "altitude_m"),
    ("duplicate record", {"truth": lambda rs: rs + [rs[3]]}, "twice"),
    ("NaN in the truth", {"truth": lambda rs: [dict(r, heading_deg=float("nan")) if r["i"] == 2 else r for r in rs]},
     "heading_deg"),
])
def test_malformed_flight_files_are_named(synthetic_flight, tmp_path, what, edit, word):
    d = _edit_flight(synthetic_flight[0], tmp_path, **edit)
    with pytest.raises(ValueError, match=word):
        src = FlightSource(d, fps=10)
        [src.truth(f.idx) for f in src]


def test_a_video_of_another_size_than_the_camera_is_an_error(synthetic_flight, tmp_path):
    d = _edit_flight(synthetic_flight[0], tmp_path, meta=lambda m: m["camera"].update(size=[640, 480]))
    with pytest.raises(ValueError, match="320 x 240.*640 x 480"):
        list(FlightSource(d, fps=10))


def test_keyframes_outside_the_window_are_not_read(fixture_flight, monkeypatch):
    reads = []
    real = cv2.imread
    monkeypatch.setattr(cv2, "imread", lambda p, *a: reads.append(p) or real(p, *a))
    frames = list(FlightSource(fixture_flight[0], keyframes=True, end_s=205))   # 6150 / 30 = 205 s
    assert [f.idx for f in frames] == [6150] and len(reads) == 1


def test_both_sources_have_a_length(synthetic_flight):
    assert len(FlightSource(synthetic_flight[0], fps=5)) == 10
    assert len(FlightSource(synthetic_flight[0], fps=10, start_s=0.5, end_s=1.2)) == 8
    assert len(CutoutSource(_texture(400, 300), [CutoutPose(200, 150, 0, 1)] * 3, (64, 48))) == 3


def test_priors_are_opt_in_and_come_from_the_truth(synthetic_flight):
    # M6-02 degrades a compass and an altimeter: the clean values come from the truth
    plain = next(iter(FlightSource(synthetic_flight[0], fps=10)))
    assert plain.heading_prior_deg is None and plain.scale_prior is None
    src = FlightSource(synthetic_flight[0], fps=10, priors=True)
    for f in list(src)[:3]:
        assert f.heading_prior_deg == pytest.approx(src.truth(f.idx).heading_deg)
        assert f.scale_prior == pytest.approx(1.2, rel=1e-3)        # the synthetic flight's scale
    poses = [CutoutPose(200, 150, 33.0, 1.1)]
    f = next(iter(CutoutSource(_texture(400, 300), poses, (64, 48), priors=True)))
    assert f.heading_prior_deg == pytest.approx(33.0) and f.scale_prior == pytest.approx(1.1)
