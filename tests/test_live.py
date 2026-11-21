"""M7-04 · live mode: the engine and the simulator run together, and the telemetry is written as it is made.

Done when: "a simulated flight runs live at at least one fix per second, and the viewer stays smooth (interpolating
between fixes)". The engine's half of that is tested here; the viewer's half (tailing run.jsonl, the glide between
fixes) is tested in ui/tests/live.test.ts and ui/tests/live-console.test.tsx.

1. Pace: a frame is handed to the engine at its flight time (``t / speed`` seconds after the start), not before.
   Proven with an injected clock, so the test does not depend on the machine: the records carry the wall time they
   were made at (``extra["wall_s"]``).
2. Rate: at least one fix per second. With the injected clock for an engine slower than the camera (it drops stale
   frames instead of falling behind), and once measured with the real clock on a small synthetic flight.
3. Streaming: run.json is there before the first record; after every record run.jsonl holds exactly the records so
   far, each a whole line, and live.json says where the flight is. A reader can follow the run while it goes on.
4. Chaos, live: control.json in the run folder switches a jam on and off while the flight runs; a control file that
   cannot be read changes nothing. The tracker loses the track and finds it again.
5. Frames before the first fix have no position (there is no later fix to borrow one from while the flight runs).
6. ``scripts/live.py``: one command, a flight folder (FlightSource) and a map in, a live run folder out.
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
from tests.test_run import _city

from dronenav.live import CONTROL, STATUS, live_run
from dronenav.sim import CutoutSource, cutout_route
from dronenav.telemetry import read_run, read_run_info
from dronenav.types import TrackMode

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "scripts" / "live.py"
FPS = 2.0            # the simulated camera: two frames a second, so one fix a second leaves room to lose half
SIZE = (320, 240)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """A textured map on disk and a warm feature cache, so each test pays only for its frames."""
    d = tmp_path_factory.mktemp("live_world")
    img = _city(2000, 1500)
    cv2.imwrite(str(d / "map.png"), img)
    return d / "map.png", img, d / "cache"


def _source(world, n=8, fps=FPS, seed=3):
    _, img, _ = world
    poses = cutout_route((img.shape[1], img.shape[0]), n, seed=seed, frame_size=SIZE, scale=(0.9, 1.1),
                         tilt_max_deg=3.0)
    return CutoutSource(img, poses, SIZE, fps=fps)


class FakeClock:
    """A clock that moves only when the run sleeps (or when a test says the engine took time)."""

    def __init__(self):
        self.now = 100.0          # not zero: the run must measure from its own start
        self.slept = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        assert seconds > 0, "sleep is called only when there is time to wait"
        self.slept.append(seconds)
        self.now += seconds


def _live(world, out, source, **kw):
    return live_run(world[0], source, out, cache_dir=world[2], **kw)


# --- 1. pace ---------------------------------------------------------------------------------------------------

def test_frames_reach_the_engine_at_flight_pace(tmp_path, world):
    clock = FakeClock()
    s = _live(world, tmp_path / "out", _source(world), clock=clock, sleep=clock.sleep)
    recs = read_run(tmp_path / "out" / "run.jsonl")
    assert [r.idx for r in recs] == list(range(8)) and s.frames == 8 and s.skipped == 0
    # the engine costs no time on this clock, so each record is made exactly when its frame is due: t seconds in
    assert [r.extra["wall_s"] for r in recs] == pytest.approx([r.t for r in recs], abs=1e-6)
    assert sum(clock.slept) == pytest.approx(3.5, abs=1e-6)          # 8 frames at 2 fps: the last is due at 3.5 s
    assert s.fixes == 8 and s.fixes_per_s == pytest.approx(2.0)     # 7 intervals between 8 fixes in 3.5 s


def test_speed_plays_the_flight_faster(tmp_path, world):
    clock = FakeClock()
    _live(world, tmp_path / "out", _source(world, n=5), speed=4.0, clock=clock, sleep=clock.sleep)
    recs = read_run(tmp_path / "out" / "run.jsonl")
    assert [r.extra["wall_s"] for r in recs] == pytest.approx([r.t / 4.0 for r in recs], abs=1e-6)
    assert [r.t for r in recs] == [0.0, 0.5, 1.0, 1.5, 2.0]           # flight time itself is untouched


@pytest.mark.parametrize("speed", [0, -1, math.nan, math.inf, "2", True])
def test_a_speed_that_is_not_a_positive_number_is_refused(tmp_path, world, speed):
    with pytest.raises(ValueError, match="speed"):
        _live(world, tmp_path / "out", _source(world, n=2), speed=speed)
    assert not (tmp_path / "out").exists()


# --- 2. rate ---------------------------------------------------------------------------------------------------

def test_a_slow_engine_drops_stale_frames_and_still_gives_a_fix_a_second(tmp_path, world):
    clock = FakeClock()

    def engine_took(_rec):           # every frame costs the engine 0.7 s: longer than the 0.5 s between frames
        clock.now += 0.7

    s = _live(world, tmp_path / "out", _source(world), clock=clock, sleep=clock.sleep, on_record=engine_took)
    recs = read_run(tmp_path / "out" / "run.jsonl")
    assert s.skipped > 0 and s.frames + s.skipped == 8 and len(recs) == s.frames
    # never further behind the flight than one frame period: a late frame is dropped, not queued
    for r in recs:
        assert r.extra["wall_s"] - r.t < 1 / FPS + 1e-6, r.idx
    assert recs[-1].idx == 7, "the newest frame is the one that is processed"
    # the spec's bar. A 2 fps camera and an engine at 0.7 s a frame give 5 intervals in 3.5 s: 1.43 fixes/s
    assert s.fixes_per_s >= 1.0


def test_a_small_flight_runs_live_at_one_fix_a_second_on_the_real_clock(tmp_path, world):
    t0 = time.monotonic()
    s = _live(world, tmp_path / "out", _source(world, n=7))
    wall = time.monotonic() - t0
    recs = read_run(tmp_path / "out" / "run.jsonl")
    # Real time: 7 frames at 2 fps last 3 s, and the run may not finish before the last frame is due.
    assert wall >= 3.0 - 0.05
    assert recs[-1].extra["wall_s"] >= 3.0 - 0.05
    # The bar is 1 fix/s and the camera gives 2: the engine may lose every second frame to a loaded CI machine
    # (each frame of this size takes it well under 0.5 s) and the test still holds.
    assert s.fixes_per_s >= 1.0, s
    for r in recs:
        if r.track.fix is not None and r.track.fix.accepted:
            tr = r.extra["truth"]
            # 2 px: an undistorted cut-out with a mild tilt, as in test_run.py
            assert math.hypot(r.track.fix.x - tr["x"], r.track.fix.y - tr["y"]) <= 2.0, r.idx


# --- 3. streaming ----------------------------------------------------------------------------------------------

def test_the_run_can_be_read_while_it_is_being_written(tmp_path, world):
    out = tmp_path / "out"
    clock = FakeClock()
    seen = []

    def reader(rec):
        info = read_run_info(out)                                    # run.json is there from the first record on
        on_disk = read_run(out / "run.jsonl")                        # whole lines only: it parses at any moment
        status = json.loads((out / STATUS).read_text())
        seen.append((rec.idx, info.generator, [r.idx for r in on_disk], status["state"], status["frames"]))
        assert (out / rec.source).is_file(), "the frame of a record is in the run folder before the record"

    src = _source(world, n=5)
    s = _live(world, out, src, clock=clock, sleep=clock.sleep, on_record=reader)
    assert seen == [(k, "live.py", list(range(k + 1)), "running", k + 1) for k in range(5)]
    info = read_run_info(out)
    assert info.fps == FPS and info.frame_size == SIZE and info.map_path.samefile(world[0])
    status = json.loads((out / STATUS).read_text())
    assert status["state"] == "done" and status["frames"] == 5 and status["fixes"] == s.fixes == 5
    assert status["fixes_per_s"] == pytest.approx(s.fixes_per_s) and status["jam"] is False
    for r in read_run(out / "run.jsonl"):                            # every frame is kept inside the run folder
        assert r.source == f"frames/frame_{r.idx:06d}.jpg"
        tr = src.truth(r.idx)
        assert (r.extra["truth"]["x"], r.extra["truth"]["y"]) == pytest.approx(tr.center)


def test_a_failed_start_leaves_no_run_folder(tmp_path, world):
    with pytest.raises((OSError, ValueError)):
        live_run(tmp_path / "no_map.png", _source(world, n=2), tmp_path / "out", cache_dir=world[2])
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="matcher"):
        _live(world, tmp_path / "out", _source(world, n=2), matcher="nope")
    assert not (tmp_path / "out").exists()


def test_a_stopped_run_says_so(tmp_path, world):
    out = tmp_path / "out"
    clock = FakeClock()

    def stop(rec):
        if rec.idx == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _live(world, out, _source(world, n=6), clock=clock, sleep=clock.sleep, on_record=stop)
    assert json.loads((out / STATUS).read_text())["state"] == "stopped"
    assert [r.idx for r in read_run(out / "run.jsonl")] == [0, 1, 2]   # what was made stays readable


# --- 4. chaos, switched on live --------------------------------------------------------------------------------

def test_a_jam_is_switched_on_and_off_while_the_flight_runs(tmp_path, world):
    out = tmp_path / "out"
    clock = FakeClock()
    jam_in_status = {}

    def operator(rec):
        jam_in_status[rec.idx] = json.loads((out / STATUS).read_text())["jam"]
        if rec.idx == 3:
            (out / CONTROL).write_text(json.dumps({"jam": True}))
        if rec.idx == 5:
            (out / CONTROL).write_text('{"jam": fal')                # half-written: the last order stands
        if rec.idx == 6:
            (out / CONTROL).write_text(json.dumps({"jam": False}))

    src = _source(world, n=12)
    _live(world, out, src, clock=clock, sleep=clock.sleep, on_record=operator)
    recs = read_run(out / "run.jsonl")
    assert [r.idx for r in recs if r.extra.get("jammed")] == [4, 5, 6]
    assert [i for i, on in jam_in_status.items() if on] == [4, 5, 6]
    for r in recs[4:7]:
        assert r.track.fix is None or not r.track.fix.accepted, r.idx
        noise = cv2.imread(str(out / r.source), cv2.IMREAD_GRAYSCALE)   # what the engine saw is what is saved
        assert noise is not None and noise.std() > 40, r.idx
    for r in recs[:4]:
        assert r.track.state is TrackMode.TRACKING and r.track.fix.accepted, r.idx
    end = recs[-1]
    tr = src.truth(end.idx)
    assert end.track.state is TrackMode.TRACKING and end.track.fix.accepted
    # 2 px on the fix, as above: after the jam the engine finds the same frame-to-map answer again
    assert math.hypot(end.track.fix.x - tr.center[0], end.track.fix.y - tr.center[1]) <= 2.0


def test_a_scheduled_jam_needs_no_operator(tmp_path, world):
    out = tmp_path / "out"
    clock = FakeClock()
    _live(world, out, _source(world, n=6), jam=(1.0, 1.0), clock=clock, sleep=clock.sleep)
    assert [r.idx for r in read_run(out / "run.jsonl") if r.extra.get("jammed")] == [2, 3]   # t = 1.0 and 1.5


@pytest.mark.parametrize("jam", [(1.0,), (0.0, 0.0), (-1.0, 2.0), "soon"])
def test_a_bad_jam_window_is_refused(tmp_path, world, jam):
    with pytest.raises(ValueError, match="jam"):
        _live(world, tmp_path / "out", _source(world, n=2), jam=jam)
    assert not (tmp_path / "out").exists()


# --- 5. before the first fix -----------------------------------------------------------------------------------

def test_frames_before_the_first_fix_have_no_position(tmp_path, world):
    out = tmp_path / "out"
    clock = FakeClock()
    _live(world, out, _source(world, n=5), jam=(0.0, 1.0), clock=clock, sleep=clock.sleep)
    recs = read_run(out / "run.jsonl")
    for r in recs[:2]:                                               # jammed from the start: nothing to stand on
        assert r.track.state is not TrackMode.TRACKING and r.track.x is None and r.track.cov is None, r.idx
    assert recs[2].track.x is not None and recs[2].track.fix.accepted


# --- 6. the command --------------------------------------------------------------------------------------------

def _flight(tmp_path, world, n=6):
    """A flight folder as FlightSource reads it (keyframes, truth, meta), cut out of the test map."""
    src = _source(world, n=n)
    d = tmp_path / "flight"
    (d / "frames").mkdir(parents=True)
    w, h = SIZE
    with open(d / "truth.jsonl", "w") as out:
        for f in src:
            cv2.imwrite(str(d / "frames" / f"frame_{f.idx:06d}.png"), f.image)
            tr = src.truth(f.idx)
            out.write(json.dumps({"i": f.idx, "t": f.t, "H_frame_to_map": np.asarray(tr.H).tolist(),
                                  "center_map_px": list(tr.center),
                                  "footprint_map_px": np.asarray(tr.footprint).tolist(),
                                  "heading_deg": tr.heading_deg, "altitude_m": 100.0, "pitch_deg": 0.0,
                                  "roll_deg": 0.0, "speed_mps": 10.0}) + "\n")
    K = [[300.0, 0, w / 2], [0, 300.0, h / 2], [0, 0, 1]]
    (d / "meta.json").write_text(json.dumps({"fps": FPS, "frames": n,
                                             "camera": {"size": [w, h], "K": K, "dist": [0, 0, 0, 0, 0]}}))
    return d


def _cli(*args):
    return subprocess.run([sys.executable, str(LIVE), *map(str, args)], check=False, capture_output=True, text=True)


def test_one_command_plays_a_flight_live(tmp_path, world):
    flight, out = _flight(tmp_path, world), tmp_path / "runs" / "live"
    res = _cli("--flight", flight, "--map", world[0], "--out", out, "--cache", world[2], "--keyframes",
               "--rate", 2, "--speed", 2)
    assert res.returncode == 0, res.stderr
    assert "fixes/s" in res.stdout
    status = json.loads((out / STATUS).read_text())
    recs = read_run(out / "run.jsonl")
    assert status["state"] == "done" and status["frames"] == len(recs) and len(recs) + status["skipped"] == 6
    assert recs[0].idx == 0 and recs[-1].idx == 5
    assert read_run_info(out).generator == "live.py"
    for r in recs:
        assert (out / r.source).is_file() and "truth" in r.extra


def test_the_command_refuses_bad_input_and_writes_nothing(tmp_path, world):
    flight = _flight(tmp_path, world, n=2)
    for args in (["--flight", tmp_path / "nope", "--map", world[0]],
                 ["--flight", flight, "--map", tmp_path / "nope.png"],
                 ["--flight", flight, "--map", world[0], "--matcher", "nope"],
                 ["--flight", flight, "--map", world[0], "--speed", "0"],
                 ["--flight", flight, "--map", world[0], "--jam", "soon"]):
        res = _cli(*args, "--out", tmp_path / "out", "--cache", world[2], "--keyframes")
        assert res.returncode == 2, (args, res.stdout, res.stderr)
        assert not (tmp_path / "out").exists(), args
