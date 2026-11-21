"""M1-09 · scripts/run.py end to end: map + folder of frames -> run.json, run.jsonl, trajectory image, video.

Done when (issue spec update):
1. the acceptance command on the task data, from another working directory with an empty cache, exits 0 within
   2 minutes and writes all five outputs; run.jsonl is valid, 50 records, >= 48 localized; the video decodes;
2. CI end to end: cut-out frames of a textured map -> run.py -> every frame localized within 2 px of its truth;
3. misses: jammed frames become LOST with coasted positions, the next fix is RELOCALIZING, the run completes;
   jammed frames before the first fix take its position; all frames jammed -> exit 1, clear message;
   (these are M1's interim states, kept as ``--tracker none``; the default since M5-02 is the Kalman tracker, tested
   at the end of this file)
4. errors (missing map/frames, unknown matcher) are clear and leave no output folder (--video: test_run_video.py).
"""
import itertools
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.sim import CutoutSource, cutout_route
from dronenav.telemetry import read_run, read_run_info
from dronenav.types import TrackMode

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "scripts" / "run.py"
OUTPUTS = ("run.json", "run.jsonl", "trajectory.png", "trajectory_preview.jpg", "trajectory.mp4")


def _cli(*args, cwd=None):
    return subprocess.run([sys.executable, str(RUN), *map(str, args)], check=False, capture_output=True, text=True,
                          cwd=cwd)


def _city(w, h, seed=3):
    """A stand-in for satellite imagery that SIFT can work with: buildings (rotated boxes of varied brightness)
    and roads on a noisy ground. Blurred noise alone gave SIFT 6 keypoints on 2000 x 1500 px; this gives ~17 000."""
    rng = np.random.default_rng(seed)
    img = cv2.add(np.full((h, w), 90, np.uint8), rng.integers(0, 25, (h, w), dtype=np.uint8))
    for _ in range(w * h // 900):
        c = (float(rng.uniform(0, w)), float(rng.uniform(0, h)))
        size = (float(rng.uniform(6, 40)), float(rng.uniform(6, 40)))
        box = cv2.boxPoints((c, size, float(rng.uniform(0, 180)))).astype(np.int32)
        cv2.fillConvexPoly(img, box, int(rng.integers(0, 256)))
    for _ in range(w * h // 60000):
        p1 = tuple(int(v) for v in rng.uniform(0, (w, h)))
        p2 = tuple(int(v) for v in rng.uniform(0, (w, h)))
        cv2.line(img, p1, p2, int(rng.integers(150, 230)), int(rng.integers(3, 9)))
    return cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 0.8), cv2.COLOR_GRAY2BGR)


def _jam(shape, seed):
    """What a jammed camera sends: noise, nothing of the map."""
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("world")
    img = _city(2000, 1500)                        # room for a cut-out route to turn around in (M6-01)
    cv2.imwrite(str(d / "map.png"), img)
    return d / "map.png", img


def _frames(tmp_path, world, n=12, jam=(), seed=3):
    """n cut-out frames along a seeded route, written as frame_000.png ...; the indices in ``jam`` are noise."""
    _, img = world
    poses = cutout_route((img.shape[1], img.shape[0]), n, seed=seed, frame_size=(320, 240), scale=(0.9, 1.1),
                         tilt_max_deg=3.0)
    src = CutoutSource(img, poses, (320, 240))
    d = tmp_path / "frames"
    d.mkdir()
    for f in src:
        image = _jam(f.image.shape, f.idx) if f.idx in jam else f.image
        cv2.imwrite(str(d / f"frame_{f.idx:03d}.png"), image)
    return d, src


# --- 2. CI end to end ------------------------------------------------------------------------------------------

def test_every_cutout_frame_is_localized_within_2_px(tmp_path, world):
    frames, src = _frames(tmp_path, world)
    res = _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out", "--cache", tmp_path / "cache", "--tracker", "none")
    assert res.returncode == 0, res.stderr
    for name in OUTPUTS:
        assert (tmp_path / "out" / name).is_file(), name
    recs = read_run(tmp_path / "out" / "run.jsonl")
    assert [r.idx for r in recs] == list(range(12))
    for r in recs:
        tr = src.truth(r.idx)
        assert r.track.state is TrackMode.TRACKING and r.track.fix.accepted, r.idx
        # 2 px: the fix of an undistorted cut-out (no noise, mild tilt) is sub-pixel; 2 px leaves room for MAGSAC
        assert math.hypot(r.track.x - tr.center[0], r.track.y - tr.center[1]) <= 2.0, r.idx


def test_the_records_carry_the_search(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=6)
    assert _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out",
                "--cache", tmp_path / "cache").returncode == 0
    recs = read_run(tmp_path / "out" / "run.jsonl")
    assert recs[0].extra["search"] == "global" and {r.extra["search"] for r in recs[1:]} <= {"window", "fallback"}
    for r in recs:
        assert r.backend == "rootsift" and r.ms > 0 and r.track.fix.inliers > 0
        assert 0 < r.extra["factor"] <= 1
    info = read_run_info(tmp_path / "out")
    assert info.generator == "run.py" and info.frame_size == (320, 240) and info.map_path.samefile(world[0])
    assert (tmp_path / "out" / recs[0].source).is_file()                # sources resolve from the run folder


def test_accepted_fixes_report_their_tilt(tmp_path, world):
    # M2-04: the tilt read off each accepted fix's homography goes into the record; the cut-outs tilt by at most
    # 3° (``_frames``), so a reported tilt above 5° would be a misread; a jammed frame has no fix, so no tilt
    frames, _ = _frames(tmp_path, world, n=6, jam=(3,))
    assert _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out",
                "--cache", tmp_path / "cache").returncode == 0
    for r in read_run(tmp_path / "out" / "run.jsonl"):
        if r.track.fix is not None and r.track.fix.accepted:
            assert 0 <= r.extra["tilt_deg"] < 5
        else:
            assert "tilt_deg" not in r.extra


def test_frames_carry_their_time(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=4)
    assert _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out", "--fps", 5,
                "--cache", tmp_path / "cache").returncode == 0
    assert [r.t for r in read_run(tmp_path / "out" / "run.jsonl")] == pytest.approx([0, 0.2, 0.4, 0.6])
    assert read_run_info(tmp_path / "out").fps == 5


# --- 3. misses -------------------------------------------------------------------------------------------------

def test_jammed_frames_are_lost_and_coast_then_relocalize(tmp_path, world):
    frames, _ = _frames(tmp_path, world, jam={5, 6})
    res = _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out", "--cache", tmp_path / "cache", "--tracker", "none")
    assert res.returncode == 0, res.stderr
    recs = {r.idx: r for r in read_run(tmp_path / "out" / "run.jsonl")}
    assert recs[5].track.state is TrackMode.LOST and recs[6].track.state is TrackMode.LOST
    assert recs[7].track.state is TrackMode.RELOCALIZING and recs[7].track.fix.accepted
    assert recs[4].track.state is TrackMode.TRACKING
    # coasting: the last fix plus the last measured step, once per missed frame
    step = np.subtract((recs[4].track.x, recs[4].track.y), (recs[3].track.x, recs[3].track.y))
    for k, i in enumerate((5, 6), start=1):
        expected = np.add((recs[4].track.x, recs[4].track.y), k * step)
        assert (recs[i].track.x, recs[i].track.y) == (pytest.approx(expected[0]), pytest.approx(expected[1])), i


def test_jammed_frames_before_the_first_fix_take_its_position(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=6, jam={0, 1})
    assert _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out",
                "--cache", tmp_path / "cache", "--tracker", "none").returncode == 0
    recs = read_run(tmp_path / "out" / "run.jsonl")
    assert [r.track.state for r in recs[:3]] == [TrackMode.LOST, TrackMode.LOST, TrackMode.TRACKING]
    for r in recs[:2]:
        assert (r.track.x, r.track.y) == (recs[2].track.x, recs[2].track.y)


def test_a_run_where_nothing_is_found_is_an_error(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=3, jam={0, 1, 2})
    res = _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out", "--cache", tmp_path / "cache")
    assert res.returncode == 1 and "no frame" in res.stderr.lower()


# --- 4. errors -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["no map", "no frames", "unknown matcher"])
def test_errors_are_clear_and_leave_nothing_behind(tmp_path, world, case):
    frames, _ = _frames(tmp_path, world, n=2)
    args = {"no map": ["--map", tmp_path / "nope.png", "--frames", frames],
            "no frames": ["--map", world[0], "--frames", tmp_path / "nothing"],
            "unknown matcher": ["--map", world[0], "--frames", frames, "--matcher", "orb"]}[case]
    word = {"no map": "nope.png", "no frames": "nothing", "unknown matcher": "orb"}[case]
    res = _cli(*args, "--out", tmp_path / "out", "--cache", tmp_path / "cache")
    assert res.returncode != 0 and word in res.stderr, res.stderr
    assert "Traceback" not in res.stderr
    assert not (tmp_path / "out").exists()


# --- 1. the acceptance command on the task data ----------------------------------------------------------------

TASK = ROOT / "data" / "hackatons-data"


@pytest.mark.data
@pytest.mark.skipif(not (TASK / "crops").is_dir(), reason="data/ not downloaded")
def test_the_acceptance_command_on_the_task_data(tmp_path):
    out = tmp_path / "run1"
    t0 = time.perf_counter()
    # another working directory, so the cache (outputs/cache, relative to it) starts empty: a fresh clone
    res = _cli("--map", TASK / "global_map.png", "--frames", TASK / "crops", "--out", out, cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert time.perf_counter() - t0 < 120
    for name in OUTPUTS:
        assert (out / name).is_file(), name
    recs = read_run(out / "run.jsonl")
    assert [r.idx for r in recs] == list(range(50))
    assert sum(r.track.fix is not None and r.track.fix.accepted for r in recs) >= 48
    cap, n = cv2.VideoCapture(str(out / "trajectory.mp4")), 0
    while cap.read()[0]:
        n += 1
    assert n == pytest.approx((49 + 1) * 30, abs=1)                      # 49 s of run at 1 fps, plus 1 s hold
    assert json.loads((out / "run.json").read_text())["generator"] == "run.py"


# --- review ----------------------------------------------------------------------------------------------------

def test_coasting_stops_after_a_few_seconds(tmp_path, world):
    # a 213-frame loss on a stale map coasted 8.8 km off the map on a stale velocity: after COAST_MAX_S the
    # position holds (the uncertainty keeps growing)
    from dronenav.pipeline import COAST_MAX_S
    frames, _ = _frames(tmp_path, world, n=14, jam=set(range(4, 13)))      # 9 s lost at 1 fps
    assert _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out",
                "--cache", tmp_path / "cache", "--tracker", "none").returncode == 0
    recs = {r.idx: r for r in read_run(tmp_path / "out" / "run.jsonl")}
    held = (recs[3 + COAST_MAX_S].track.x, recs[3 + COAST_MAX_S].track.y)   # the last fix was frame 3
    for i in range(4 + COAST_MAX_S, 13):
        assert (recs[i].track.x, recs[i].track.y) == (pytest.approx(held[0]), pytest.approx(held[1])), i
        assert recs[i].track.vx == 0 and recs[i].track.vy == 0
    sig = [recs[i].track.cov[0, 0] for i in range(4, 13)]
    assert all(b > a for a, b in itertools.pairwise(sig))                   # less and less sure


def _gapped_frames(tmp_path, world, idxs, jam=()):
    """Cut-out frames named by video frame number (frame_000, frame_030, ...), as Irpin keyframes are."""
    _, img = world
    poses = cutout_route((img.shape[1], img.shape[0]), len(idxs), seed=5, frame_size=(320, 240), scale=(0.9, 1.1),
                         tilt_max_deg=3.0)
    src = CutoutSource(img, poses, (320, 240))
    d = tmp_path / "gapped"
    d.mkdir()
    for f, i in zip(src, idxs, strict=True):
        cv2.imwrite(str(d / f"frame_{i:03d}.png"), _jam(f.image.shape, i) if i in jam else f.image)
    return d


def test_time_not_frame_numbers_drives_coasting_and_uncertainty(tmp_path, world):
    # keyframes numbered 0, 30, 60 ... at 30 fps are 1 s apart: sigma grew 600 px per missed keyframe when it
    # counted frame numbers instead of seconds
    from dronenav.pipeline import LOST_SIGMA_GROWTH_PX_PER_S
    frames = _gapped_frames(tmp_path, world, [0, 30, 60, 90, 120, 150], jam={90, 120})
    assert _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out", "--fps", 30,
                "--cache", tmp_path / "cache", "--tracker", "none").returncode == 0
    r = {x.idx: x for x in read_run(tmp_path / "out" / "run.jsonl")}
    assert [r[i].t for i in sorted(r)] == pytest.approx([0, 1, 2, 3, 4, 5])
    # velocity in map px per second between the last two fixes, and the coast uses seconds
    vx = (r[60].track.x - r[30].track.x) / 1.0
    assert r[60].track.vx == pytest.approx(vx) and r[90].track.vx == pytest.approx(vx)
    assert r[120].track.x == pytest.approx(r[60].track.x + 2 * vx)
    s60, s90, s120 = (math.sqrt(r[i].track.cov[0, 0]) for i in (60, 90, 120))
    assert s90 == pytest.approx(s60 + LOST_SIGMA_GROWTH_PX_PER_S) and s120 == pytest.approx(s60 + 2 * LOST_SIGMA_GROWTH_PX_PER_S)


def test_a_fix_carries_its_velocity_and_uncertainty(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=4)
    assert _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out",
                "--cache", tmp_path / "cache", "--tracker", "none").returncode == 0
    recs = read_run(tmp_path / "out" / "run.jsonl")
    assert (recs[0].track.vx, recs[0].track.vy) == (0, 0)                   # one fix: no motion yet
    for a, b in itertools.pairwise(recs):
        assert b.track.vx == pytest.approx(b.track.x - a.track.x) and b.track.vy == pytest.approx(b.track.y - a.track.y)
    for r in recs:
        assert r.track.cov[0, 0] == pytest.approx(max(1.0, r.track.fix.reproj_px) ** 2)
        assert r.track.cov[0, 1] == 0


def test_window_searches_record_their_window(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=4)
    assert _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out",
                "--cache", tmp_path / "cache").returncode == 0
    recs = read_run(tmp_path / "out" / "run.jsonl")
    assert recs[0].extra["window"] is None
    for r in recs[1:]:
        if r.extra["search"] == "window":
            x0, y0, x1, y1 = r.extra["window"]
            assert x0 <= r.track.fix.x <= x1 and y0 <= r.track.fix.y <= y1


def test_sources_are_relative_to_the_run_folder(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=2)
    assert _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out",
                "--cache", tmp_path / "cache").returncode == 0
    for r in read_run(tmp_path / "out" / "run.jsonl"):
        assert not Path(r.source).is_absolute() and (tmp_path / "out" / r.source).is_file()


def test_nothing_localized_leaves_no_folder(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=2, jam={0, 1})
    assert _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out",
                "--cache", tmp_path / "cache").returncode == 1
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("case", ["speed", "out is a file"])
def test_bad_settings_fail_before_any_work(tmp_path, world, case):
    frames, _ = _frames(tmp_path, world, n=2)
    out = tmp_path / "out"
    if case == "out is a file":
        out.write_text("not a folder")
    args = ["--speed", "0"] if case == "speed" else []
    res = _cli("--map", world[0], "--frames", frames, "--out", out, "--cache", tmp_path / "cache", *args)
    assert res.returncode != 0 and ("speed" in res.stderr or "not a folder" in res.stderr), res.stderr
    assert not (tmp_path / "cache").exists()                                   # no map features were computed
    assert case == "out is a file" or not out.exists()


def test_a_failed_render_leaves_the_previous_run_whole(tmp_path, world, monkeypatch):
    # run.jsonl of a new run next to trajectory.mp4 of an old one would be a silently inconsistent folder
    from dronenav import pipeline
    frames, _ = _frames(tmp_path, world, n=2)
    out = tmp_path / "out"
    out.mkdir()
    (out / "run.jsonl").write_text("old run\n")
    (out / "trajectory.mp4").write_bytes(b"old video")
    (out / "notes.txt").write_text("mine")

    def broken(*a, **k):
        raise RuntimeError("encoder exploded")

    monkeypatch.setattr(pipeline, "write_video", broken)
    with pytest.raises(RuntimeError, match="exploded"):
        pipeline.run(world[0], frames, out, cache_dir=tmp_path / "cache")
    assert (out / "run.jsonl").read_text() == "old run\n" and (out / "trajectory.mp4").read_bytes() == b"old video"
    assert sorted(p.name for p in tmp_path.iterdir() if "partial" in p.name) == []


def test_a_new_run_replaces_the_old_outputs_and_keeps_other_files(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=2)
    out = tmp_path / "out"
    out.mkdir()
    (out / "run.jsonl").write_text("old run\n")
    (out / "notes.txt").write_text("mine")
    assert _cli("--map", world[0], "--frames", frames, "--out", out, "--cache", tmp_path / "cache").returncode == 0
    assert len(read_run(out / "run.jsonl")) == 2 and (out / "notes.txt").read_text() == "mine"


def test_frames_of_mixed_sizes_have_no_single_frame_size(tmp_path, world):
    from dronenav.sim import CutoutPose, cutout_homography, render_cutout
    frames, _ = _frames(tmp_path, world, n=3)
    pose = CutoutPose(1000, 750, 0, 1.0)
    cv2.imwrite(str(frames / "frame_003.png"), render_cutout(world[1], cutout_homography(pose, (400, 300)), (400, 300)))
    assert _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out",
                "--cache", tmp_path / "cache").returncode == 0
    assert read_run_info(tmp_path / "out").frame_size is None


def test_a_map_without_features_is_a_clear_error(tmp_path, world):
    flat = tmp_path / "flat.png"
    cv2.imwrite(str(flat), np.full((600, 800, 3), 120, np.uint8))
    frames, _ = _frames(tmp_path, world, n=2)
    res = _cli("--map", flat, "--frames", frames, "--out", tmp_path / "out", "--cache", tmp_path / "cache")
    assert res.returncode != 0 and "feature" in res.stderr and "Traceback" not in res.stderr, res.stderr
    assert not (tmp_path / "out").exists()


def test_a_looser_fit_is_a_less_certain_position(tmp_path):
    # cut-outs fit to under 1 px, where the 1 px floor hides the rule; a real 2.5 px fit must widen sigma to 2.5
    from dronenav.pipeline import _Found, _records
    from dronenav.types import Fix
    H = np.array([[1.0, 0, 100], [0, 1.0, 100], [0, 0, 1]])
    fp = np.array([[100, 100], [420, 100], [420, 340], [100, 340]], float)
    found = [_Found(0, 0.0, tmp_path / "f.png", (320, 240),
                    Fix(idx=0, H=H, x=260, y=220, heading_deg=0, scale=1, footprint=fp, inliers=80,
                        inlier_ratio=0.5, reproj_px=2.5, accepted=True), "global", None, 1.0, 5.0)]
    (rec,) = _records(found, tmp_path, "rootsift")
    assert math.sqrt(rec.track.cov[0, 0]) == pytest.approx(2.5)


# --- M5-02: the tracker writes the states (the default) -------------------------------------------------------

def test_the_tracker_states_reach_the_telemetry(tmp_path, world):
    # three jammed frames: two coasted while TRACKING, LOST at the third miss, then RELOCALIZING and TRACKING
    frames, src = _frames(tmp_path, world, n=12, jam={5, 6, 7})
    res = _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out", "--cache", tmp_path / "cache")
    assert res.returncode == 0, res.stderr
    recs = read_run(tmp_path / "out" / "run.jsonl")
    T, L, R = TrackMode.TRACKING, TrackMode.LOST, TrackMode.RELOCALIZING
    assert [r.track.state for r in recs] == [R, T, T, T, T, T, T, L, R, T, T, T]
    assert [r.extra["step"] for r in recs] == ["INIT", "HIT", "HIT", "HIT", "HIT", "COAST", "COAST", "COAST",
                                               "INIT", "HIT", "HIT", "HIT"]
    for r in recs:
        if r.track.fix is not None and r.track.fix.accepted:
            tr = src.truth(r.idx)
            # 2 px as M1-09: the filter follows sub-pixel fixes closely at sigma_acc 80
            assert math.hypot(r.track.x - tr.center[0], r.track.y - tr.center[1]) <= 2.0, r.idx
    sig = [r.track.cov[0, 0] for r in recs[4:8]]
    assert all(b > a for a, b in itertools.pairwise(sig))                   # coasting: less and less sure


def test_frames_before_the_first_fix_are_lost_at_its_position(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=5, jam={0, 1})
    assert _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out",
                "--cache", tmp_path / "cache").returncode == 0
    recs = read_run(tmp_path / "out" / "run.jsonl")
    assert [r.track.state for r in recs] == [TrackMode.LOST, TrackMode.LOST, TrackMode.RELOCALIZING,
                                             TrackMode.TRACKING, TrackMode.TRACKING]
    assert [r.extra["step"] for r in recs[:2]] == ["NONE", "NONE"]
    for r in recs[:2]:
        assert (r.track.x, r.track.y) == (recs[2].track.x, recs[2].track.y)


def test_an_unknown_tracker_is_a_clear_error(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=2)
    res = _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out", "--tracker", "magic")
    assert res.returncode != 0 and "tracker" in res.stderr and "Traceback" not in res.stderr
    assert not (tmp_path / "out").exists()


# --- M5-03: the search window from the prediction (the tracker on) --------------------------------------------

def _area(w):
    return (w[2] - w[0]) * (w[3] - w[1])


def test_the_window_shrinks_while_tracking_and_grows_while_coasting(tmp_path, world):
    # two jammed frames (5, 6): coasted while TRACKING, found again at 7 in the grown window
    frames, src = _frames(tmp_path, world, n=12, jam={5, 6})
    res = _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out", "--cache", tmp_path / "cache")
    assert res.returncode == 0, res.stderr
    recs = read_run(tmp_path / "out" / "run.jsonl")
    sig = [r.extra["window_sigma_px"] for r in recs]
    win = [r.extra["predicted_window"] for r in recs]
    assert sig[0] is None and win[0] is None                    # LOST at the start: no prediction, whole map
    # tracking 1..5: the velocity is learned, the prediction firms up, then settles at the filter's steady state
    # (equal to the 0.01 px written, so: it shrinks, and never grows while the fixes keep coming)
    assert sig[1] > sig[2] > sig[3], sig
    assert all(b <= a for a, b in itertools.pairwise(sig[1:6])), sig
    # coasting 5..7: no fix, less and less sure where the drone is
    assert all(b > a for a, b in itertools.pairwise(sig[5:8])), sig
    assert sig[8] < sig[7]                                      # a fix at 7: the window closes in again
    assert _area(win[7]) > _area(win[4])
    for r, w in zip(recs, win, strict=True):
        if w is not None:                                       # the drone is inside the window it was searched in
            cx, cy = src.truth(r.idx).center
            assert w[0] <= cx < w[2] and w[1] <= cy < w[3], r.idx
    # M1 would search frame 7 on the whole map (the track broke at 5); the prediction keeps it in a window
    assert recs[7].extra["search"] == "window" and recs[7].extra["window"] == win[7]
    assert recs[7].track.fix.accepted


def test_without_the_tracker_the_m1_window_stays(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=8, jam={5})
    assert _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out", "--cache", tmp_path / "cache",
                "--tracker", "none").returncode == 0
    recs = read_run(tmp_path / "out" / "run.jsonl")
    assert all("predicted_window" not in r.extra and "window_sigma_px" not in r.extra for r in recs)
    assert recs[6].extra["search"] == "global"                  # M1: a miss restarts the search on the whole map
