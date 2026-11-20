"""M1-09 · scripts/run.py end to end: map + folder of frames -> run.json, run.jsonl, trajectory image, video.

Done when (issue spec update):
1. the acceptance command on the task data, from another working directory with an empty cache, exits 0 within
   2 minutes and writes all five outputs; run.jsonl is valid, 50 records, >= 48 localized; the video decodes;
2. CI end to end: cut-out frames of a textured map -> run.py -> every frame localized within 2 px of its truth;
3. misses: jammed frames become LOST with coasted positions, the next fix is RELOCALIZING, the run completes;
   jammed frames before the first fix take its position; all frames jammed -> exit 1, clear message;
4. errors (missing map/frames, unknown matcher, --video) are clear and leave no output folder.
"""
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


def _texture(w, h, seed=3):
    """Structure at several scales, so SIFT finds plenty of distinctive features (a stand-in for satellite)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), np.float32)
    for sigma, weight in ((1.5, 0.5), (4, 0.8), (12, 1.0)):
        img += weight * cv2.GaussianBlur(rng.uniform(0, 1, (h, w)).astype(np.float32), (0, 0), sigma)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.merge([img, cv2.GaussianBlur(img, (0, 0), 2), 255 - img])


def _jam(shape, seed):
    """What a jammed camera sends: noise, nothing of the map."""
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("world")
    img = _texture(1600, 1200)
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
    res = _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out", "--cache", tmp_path / "cache")
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


def test_frames_carry_their_time(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=4)
    assert _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out", "--fps", 5,
                "--cache", tmp_path / "cache").returncode == 0
    assert [r.t for r in read_run(tmp_path / "out" / "run.jsonl")] == pytest.approx([0, 0.2, 0.4, 0.6])
    assert read_run_info(tmp_path / "out").fps == 5


# --- 3. misses -------------------------------------------------------------------------------------------------

def test_jammed_frames_are_lost_and_coast_then_relocalize(tmp_path, world):
    frames, _ = _frames(tmp_path, world, jam={5, 6})
    res = _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out", "--cache", tmp_path / "cache")
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
                "--cache", tmp_path / "cache").returncode == 0
    recs = read_run(tmp_path / "out" / "run.jsonl")
    assert [r.track.state for r in recs[:3]] == [TrackMode.LOST, TrackMode.LOST, TrackMode.TRACKING]
    for r in recs[:2]:
        assert (r.track.x, r.track.y) == (recs[2].track.x, recs[2].track.y)


def test_a_run_where_nothing_is_found_is_an_error(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=3, jam={0, 1, 2})
    res = _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out", "--cache", tmp_path / "cache")
    assert res.returncode == 1 and "no frame" in res.stderr.lower()


# --- 4. errors -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["no map", "no frames", "unknown matcher", "video"])
def test_errors_are_clear_and_leave_nothing_behind(tmp_path, world, case):
    frames, _ = _frames(tmp_path, world, n=2)
    args = {"no map": ["--map", tmp_path / "nope.png", "--frames", frames],
            "no frames": ["--map", world[0], "--frames", tmp_path / "nothing"],
            "unknown matcher": ["--map", world[0], "--frames", frames, "--matcher", "orb"],
            "video": ["--map", world[0], "--video", tmp_path / "x.mp4"]}[case]
    word = {"no map": "nope.png", "no frames": "nothing", "unknown matcher": "orb", "video": "M2-02"}[case]
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
