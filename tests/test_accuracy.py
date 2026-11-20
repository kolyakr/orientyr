"""M1-10: accuracy check against truth, and the regression test that fails if the error grows.

Contract (issue #19, spec update of 21 Nov):
- ``load_truth`` reads ``truth.jsonl`` (``center_map_px``), a flight folder holding one, or ``probe.json`` (its
  ``ok`` rows): ``{frame idx: (x, y)}`` in map px; anything else is a clear error;
- ``score`` gives each frame's error (an accepted fix's distance from the truth; the track position's for every
  frame) and the summary: frames with truth, localized, correct (<= within_px), wrong (accepted but farther),
  median / p95 / max of the localized frames' errors; a run sharing no frame with the truth is an error;
- ``scripts/accuracy.py --run OUT --truth PATH`` prints the error per frame and the summary and writes
  ``accuracy.json`` into OUT; with ``--map`` and ``--frames`` it localizes into OUT first;
- regression: the CI fixture's 3 keyframes stay within 1.5 px; locally, the task crops and an Irpin stretch stay
  within their measured limits.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.accuracy import Accuracy, load_truth, score
from dronenav.pipeline import localize_run
from dronenav.telemetry import Record, read_run, write_run
from dronenav.types import Fix, TrackMode, TrackState

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "accuracy.py"
FIX = ROOT / "tests" / "fixtures" / "irpin_mild_2025"
DATA = ROOT / "data"
PROBE = ROOT / "experiments" / "data_probe" / "results" / "probe.json"
TASK_MEDIAN_PX, TASK_MAX_PX = 0.5, 1.5       # see the data tests for where these come from
IRPIN_MEDIAN_PX, IRPIN_MAX_PX = 0.5, 1.5


def _cli(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], check=False, capture_output=True,
                          text=True, cwd=ROOT)


def _fix(idx: int, x: float, y: float, accepted: bool = True) -> Fix:
    fp = np.array([[x - 10, y - 10], [x + 10, y - 10], [x + 10, y + 10], [x - 10, y + 10]], float)
    return Fix(idx=idx, H=np.eye(3), x=x, y=y, heading_deg=0.0, scale=1.0, footprint=fp, inliers=100,
               inlier_ratio=0.8, reproj_px=0.5, accepted=accepted,
               reject_reason=None if accepted else "too_few_inliers")


def _rec(idx: int, x: float, y: float, state=TrackMode.TRACKING, fix: Fix | None = None) -> Record:
    track = TrackState(idx=idx, state=state, x=x, y=y, vx=0.0, vy=0.0, cov=np.eye(2), fix=fix)
    return Record(idx=idx, t=float(idx), source=f"frames/{idx}.png", track=track, backend="test")


def _records():
    """Truth is (100 i, 50) for i in 0..4; frame 9 has no truth. Expected errors are exact by construction."""
    return [
        _rec(0, 0, 50, fix=_fix(0, 3, 54)),           # the fix is 5 px off (correct); the track sits on the truth
        _rec(1, 100, 50, fix=_fix(1, 100, 50)),                                  # 0 px, correct
        _rec(2, 206, 58, state=TrackMode.LOST),                                  # no fix; track 10 px off
        _rec(3, 330, 50, state=TrackMode.RELOCALIZING, fix=_fix(3, 330, 50)),    # 30 px: accepted but wrong
        _rec(4, 400, 53, state=TrackMode.LOST, fix=_fix(4, 900, 900, accepted=False)),  # rejected; track 3 px
        _rec(9, 0, 0, state=TrackMode.LOST),                                     # no truth for this frame
    ]


TRUTH = {i: (100.0 * i, 50.0) for i in range(5)}


# --- load_truth ---------------------------------------------------------------------------------------------

def test_truth_jsonl_gives_the_frame_centres():
    truth = load_truth(FIX / "truth.jsonl")
    want = {r["i"]: tuple(r["center_map_px"]) for r in map(json.loads, (FIX / "truth.jsonl").read_text().splitlines())
            if r}
    assert set(truth) == set(want) == {6150, 6300, 6450}
    for i, c in want.items():
        assert truth[i] == pytest.approx(c)


def test_a_flight_folder_reads_its_truth_jsonl(tmp_path):
    shutil.copy(FIX / "truth.jsonl", tmp_path / "truth.jsonl")
    assert load_truth(tmp_path) == load_truth(FIX / "truth.jsonl")


def test_probe_json_keeps_only_the_frames_it_could_fit(tmp_path):
    p = tmp_path / "probe.json"
    p.write_text(json.dumps({"map": "m.png", "rows": [{"i": 0, "ok": True, "cx": 1.5, "cy": 2.5},
                                                      {"i": 1, "ok": False, "cx": 9.0, "cy": 9.0},
                                                      {"i": 2, "ok": True, "cx": 3.0, "cy": 4.0}]}))
    assert load_truth(p) == {0: (1.5, 2.5), 2: (3.0, 4.0)}


@pytest.mark.parametrize("make", [
    lambda d: d / "missing.jsonl",                                                 # no such file
    lambda d: (d / "x.json").write_text('{"nothing": 1}') and d / "x.json",       # unknown JSON
    lambda d: (d / "t.jsonl").write_text("\n") and d / "t.jsonl",                 # no truth in it
    lambda d: (d / "t.jsonl").write_text('{"i": 0}\n') and d / "t.jsonl",        # a row without a centre
    lambda d: d,                                                                   # a folder without truth.jsonl
], ids=["missing", "unknown-json", "empty", "no-centre", "folder-without-truth"])
def test_bad_truth_is_a_clear_error(tmp_path, make):
    with pytest.raises((ValueError, FileNotFoundError)) as e:
        load_truth(make(tmp_path))
    assert str(e.value)


# --- score --------------------------------------------------------------------------------------------------

def test_each_frame_gets_its_error_and_the_summary_counts_right():
    acc = score(_records(), TRUTH)
    assert isinstance(acc, Accuracy)
    by = {f.idx: f for f in acc.frames}
    assert set(by) == {0, 1, 2, 3, 4}                         # frame 9 has no truth: not scored ...
    assert acc.missing_truth == 1                             # ... but counted
    assert by[0].fix_px == pytest.approx(5) and by[1].fix_px == pytest.approx(0)
    assert by[3].fix_px == pytest.approx(30)
    assert by[2].fix_px is None and by[4].fix_px is None      # no fix, and a rejected one, are not localized
    assert by[0].track_px == pytest.approx(0)                # a fix is scored by the fix, not the track
    assert by[2].track_px == pytest.approx(10) and by[4].track_px == pytest.approx(3)
    assert [by[i].state for i in range(5)] == ["TRACKING", "TRACKING", "LOST", "RELOCALIZING", "LOST"]
    assert (acc.n, acc.localized, acc.correct, acc.wrong) == (5, 3, 2, 1)
    assert acc.median_px == pytest.approx(5) and acc.max_px == pytest.approx(30)
    assert acc.p95_px == pytest.approx(np.percentile([5, 0, 30], 95))


def test_truth_frames_the_run_lacks_are_counted():
    truth = {**TRUTH, 5: (500.0, 50.0), 6: (600.0, 50.0)}
    acc = score(_records(), truth)
    assert acc.not_in_run == 2 and acc.n == 5 and acc.summary()["not_in_run"] == 2


@pytest.mark.parametrize("bad", [-1, math.nan, math.inf])
def test_the_correct_radius_must_be_a_distance(bad):
    with pytest.raises(ValueError, match="within_px"):
        score(_records(), TRUTH, within_px=bad)


def test_blank_lines_in_truth_jsonl_are_skipped(tmp_path):
    p = tmp_path / "truth.jsonl"
    p.write_text('{"i": 0, "center_map_px": [1, 2]}\n\n{"i": 1, "center_map_px": [3, 4]}\n\n')
    assert load_truth(p) == {0: (1.0, 2.0), 1: (3.0, 4.0)}


def test_the_correct_radius_is_a_setting():
    acc = score(_records(), TRUTH, within_px=4)
    assert (acc.correct, acc.wrong) == (1, 2) and acc.within_px == 4


def test_a_run_with_no_fix_has_no_error_numbers():
    acc = score([_rec(0, 0, 0, state=TrackMode.LOST)], TRUTH)
    assert (acc.n, acc.localized, acc.correct, acc.wrong) == (1, 0, 0, 0)
    assert all(math.isnan(v) for v in (acc.median_px, acc.p95_px, acc.max_px))


def test_a_run_that_shares_no_frame_with_the_truth_is_an_error():
    with pytest.raises(ValueError, match="truth"):
        score([_rec(9, 0, 0)], TRUTH)


# --- scripts/accuracy.py ------------------------------------------------------------------------------------

def test_the_script_prints_every_frame_and_the_summary_and_writes_accuracy_json(tmp_path):
    write_run(tmp_path / "run.jsonl", _records())
    truth = tmp_path / "truth.jsonl"
    truth.write_text("".join(json.dumps({"i": i, "center_map_px": list(c)}) + "\n" for i, c in TRUTH.items()))
    r = _cli("--run", tmp_path, "--truth", truth)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    for i in range(5):
        assert any(line.split()[:1] == [str(i)] for line in lines), f"no line for frame {i}:\n{r.stdout}"
    assert "median" in r.stdout and "3/5" in r.stdout and "wrong 1" in r.stdout
    out = json.loads((tmp_path / "accuracy.json").read_text())
    assert out["summary"]["localized"] == 3 and out["summary"]["median_px"] == pytest.approx(5)
    assert [f["idx"] for f in out["frames"]] == [0, 1, 2, 3, 4]
    assert out["frames"][2]["fix_px"] is None                          # JSON null, not NaN


def test_localizing_refuses_a_folder_with_a_rendered_run(tmp_path):
    """run.py's image and video would be left from another run: the folder would contradict itself."""
    write_run(tmp_path / "run.jsonl", _records())
    (tmp_path / "trajectory.mp4").write_bytes(b"old video")
    r = _cli("--map", FIX / "map_crop.jpg", "--frames", FIX, "--run", tmp_path, "--truth", FIX / "truth.jsonl")
    assert r.returncode == 2 and "stale" in r.stderr
    assert (tmp_path / "trajectory.mp4").read_bytes() == b"old video"
    assert len(read_run(tmp_path / "run.jsonl")) == len(_records())     # the old run is untouched


@pytest.mark.parametrize("args", [
    lambda d: ["--run", d / "nope", "--truth", FIX / "truth.jsonl"],              # no run there
    lambda d: ["--run", d, "--truth", d / "nope.jsonl"],                          # no truth there
    lambda d: ["--run", d, "--truth", FIX / "truth.jsonl", "--frames", d],        # --frames without --map
], ids=["missing-run", "missing-truth", "frames-without-map"])
def test_the_script_fails_clearly(tmp_path, args):
    write_run(tmp_path / "run.jsonl", _records())
    r = _cli(*args(tmp_path))
    assert r.returncode == 2 and r.stderr.strip() and "Traceback" not in r.stderr


def _fixture_map(path: Path) -> Path:
    """The fixture's 512 px crop pasted at its origin on a black canvas: map px == full-map px, so the truth
    needs no shift. The black area has no features, so the search sees only the crop."""
    meta = json.loads((FIX / "fixture.json").read_text())
    (x0, y0), (w, h) = meta["crop"]["origin_px"], meta["crop"]["size_px"]
    crop = cv2.imread(str(FIX / meta["crop"]["file"]))
    canvas = np.zeros((y0 + h, x0 + w, 3), np.uint8)
    canvas[y0:, x0:] = crop
    cv2.imwrite(str(path), canvas)
    return path


def test_the_fixture_keyframes_stay_within_1_5_px(tmp_path):
    """The CI regression test: the real engine on three real keyframes against exact truth."""
    frames = tmp_path / "frames"
    frames.mkdir()
    for f in FIX.glob("frame_*.jpg"):
        shutil.copy(f, frames / f.name)
    t0 = time.perf_counter()
    r = _cli("--map", _fixture_map(tmp_path / "map.png"), "--frames", frames, "--fps", 30,
             "--run", tmp_path / "run", "--truth", FIX / "truth.jsonl", "--cache", "none")
    assert r.returncode == 0, r.stderr
    assert time.perf_counter() - t0 < 60                                # "CPU, under a minute"
    out = json.loads((tmp_path / "run" / "accuracy.json").read_text())
    assert out["summary"]["localized"] == 3 and out["summary"]["wrong"] == 0
    # measured 0.24, 0.53, 0.48 px; 1.5 px leaves room for #106's half-pixel offset (up to ~0.7 px) and
    # platform noise in SIFT, and still fails on any real drift
    assert max(f["fix_px"] for f in out["frames"]) <= 1.5
    assert not (tmp_path / "run" / "trajectory.mp4").exists()          # localizing only: nothing rendered


# --- full data, locally --------------------------------------------------------------------------------------

@pytest.mark.data
def test_the_task_crops_stay_within_their_measured_error(tmp_path):
    task = DATA / "hackatons-data"
    if not (task / "global_map.png").is_file() or not PROBE.is_file():
        pytest.skip("needs data/hackatons-data and probe.json")
    t0 = time.perf_counter()
    localize_run(task / "global_map.png", task / "crops", tmp_path, fps=1, cache_dir=None)
    acc = score(read_run(tmp_path / "run.jsonl"), load_truth(PROBE))
    assert time.perf_counter() - t0 < 60
    # measured 50/50, median 0.21 px, max 0.51 px (21 Nov); the limits are ~2-3x that. probe.json is itself a fit
    # (~0.2 px), so finer limits would test the probe. A real regression (a lost half-pixel conversion, a window
    # offset, a wrong scale) moves the median by a pixel or more.
    assert (acc.n, acc.localized, acc.wrong) == (50, 50, 0)
    assert acc.median_px <= TASK_MEDIAN_PX and acc.max_px <= TASK_MAX_PX


@pytest.mark.data
def test_an_irpin_stretch_stays_within_its_measured_error(tmp_path):
    flight = DATA / "irpin" / "flights" / "mild" / "2025"
    the_map = DATA / "irpin" / "maps" / "irpin_2025.png"
    if not (flight / "truth.jsonl").is_file() or not the_map.is_file():
        pytest.skip("needs data/irpin (mild 2025 flight and map)")
    frames = tmp_path / "frames"
    frames.mkdir()
    for f in sorted((flight / "frames").glob("frame_*.png"))[:40]:      # 40 consecutive keyframes: 39 s of flight
        (frames / f.name).symlink_to(f)
    t0 = time.perf_counter()
    localize_run(the_map, frames, tmp_path / "run", fps=30, cache_dir=None)
    acc = score(read_run(tmp_path / "run" / "run.jsonl"), load_truth(flight))
    assert time.perf_counter() - t0 < 60
    # measured 40/40, median 0.15 px, max 0.65 px against exact truth (21 Nov); limits ~2-3x that, as above
    assert (acc.n, acc.localized, acc.wrong) == (40, 40, 0)
    assert acc.median_px <= IRPIN_MEDIAN_PX and acc.max_px <= IRPIN_MAX_PX
