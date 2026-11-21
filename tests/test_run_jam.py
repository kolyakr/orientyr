"""M7-02 · what the HUD video needs from the engine: real match points in the telemetry, and a run with a jam.

1. Match points: every record with an accepted fix carries a small sample of the pairs that agree with its
   homography, ``extra["matches"] = {"frame": [[x, y], ...], "map": [[x, y], ...]}`` (frame px, map px); a record
   without an accepted fix carries none. The sample is capped, deterministic and spread over the frame.
2. ``run.py --jam START:LENGTH``: the frames of that stretch of the flight reach the engine as noise
   (dronenav.sim.chaos.Jam). The records say so (``extra["jammed"]``) and point to the noise the camera sent, saved
   in the run folder, so a renderer shows what the engine saw. The tracker loses the track and finds it again.
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.test_run import _city, _frames

from dronenav.pipeline import MATCH_SAMPLE, match_sample
from dronenav.telemetry import read_run
from dronenav.types import Fix, Matches, TrackMode

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "scripts" / "run.py"
RANSAC_PX = 5.0     # GeometryConfig.ransac_px: a pair further than this from the homography is not an inlier


def _cli(*args):
    return subprocess.run([sys.executable, str(RUN), *map(str, args)], check=False, capture_output=True, text=True)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("world")
    img = _city(2000, 1500)
    cv2.imwrite(str(d / "map.png"), img)
    return d / "map.png", img


def _project(H, pts):
    p = np.hstack([np.asarray(pts, float), np.ones((len(pts), 1))]) @ np.asarray(H).T
    return p[:, :2] / p[:, 2:]


# --- 1. match points -------------------------------------------------------------------------------------------

H = np.array([[2.0, 0.0, 100.0], [0.0, 2.0, 50.0], [0.0, 0.0, 1.0]])


def _fix(accepted=True):
    fp = _project(H, [[0, 0], [320, 0], [320, 240], [0, 240]])
    return Fix(idx=0, H=H, x=420.0, y=290.0, heading_deg=0.0, scale=2.0, footprint=fp, inliers=40,
               inlier_ratio=0.5, reproj_px=0.5, accepted=accepted, reject_reason=None if accepted else "aspect")


def _matches(pts_frame, pts_map):
    return Matches(np.asarray(pts_frame, float), np.asarray(pts_map, float), np.ones(len(pts_frame)), "test", 1.0)


def test_the_sample_holds_only_pairs_that_agree_with_the_fix():
    rng = np.random.default_rng(0)
    good = rng.uniform((0, 0), (320, 240), (10, 2))
    bad = rng.uniform((0, 0), (320, 240), (10, 2))
    pts_map = np.vstack([_project(H, good), _project(H, bad) + 60.0])      # the second ten are 85 px off
    s = match_sample(_matches(np.vstack([good, bad]), pts_map), _fix())
    assert len(s["frame"]) == len(s["map"]) == 10
    # the engine writes 0.1 px: half of that in x and y is the most the rounding moves a point
    want = {(round(x, 1), round(y, 1)) for x, y in good}
    assert {tuple(p) for p in s["frame"]} == want
    assert np.abs(_project(H, s["frame"]) - np.array(s["map"])).max() <= 0.2


def test_the_sample_is_capped_spread_over_the_frame_and_deterministic():
    rng = np.random.default_rng(1)
    corners = np.array([[20, 20], [300, 20], [300, 220], [20, 220]], float)
    pts = np.vstack([c + rng.uniform(-5, 5, (50, 2)) for c in corners])      # four clumps of 50
    m = _matches(pts, _project(H, pts))
    s = match_sample(m, _fix(), cap=4)
    assert len(s["frame"]) == 4
    nearest = {int(np.argmin(np.linalg.norm(corners - p, axis=1))) for p in s["frame"]}
    assert nearest == {0, 1, 2, 3}                                           # one from each clump, not four of one
    assert match_sample(m, _fix(), cap=4) == s
    assert len(match_sample(m, _fix())["frame"]) == MATCH_SAMPLE == 24
    assert all(isinstance(v, float) for p in s["frame"] + s["map"] for v in p)   # JSON-ready, not numpy


def test_no_sample_without_an_accepted_fix_or_agreeing_pairs():
    pts = np.array([[10.0, 10.0], [200.0, 30.0], [100.0, 200.0], [300.0, 220.0]])
    m = _matches(pts, _project(H, pts))
    assert match_sample(m, None) is None
    assert match_sample(m, _fix(accepted=False)) is None
    assert match_sample(_matches(pts, _project(H, pts) + 60.0), _fix()) is None
    assert match_sample(_matches(np.zeros((0, 2)), np.zeros((0, 2))), _fix()) is None


def test_the_records_carry_the_match_points(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=6, jam={3})
    assert _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out",
                "--cache", tmp_path / "cache").returncode == 0
    recs = read_run(tmp_path / "out" / "run.jsonl")
    assert "matches" not in recs[3].extra                     # noise: no accepted fix, nothing to show
    for r in recs[:3] + recs[4:]:
        m = r.extra["matches"]
        pf, pm = np.array(m["frame"]), np.array(m["map"])
        assert pf.shape == pm.shape and pf.shape[1] == 2 and 4 <= len(pf) <= MATCH_SAMPLE, r.idx
        assert (pf >= 0).all() and (pf[:, 0] <= 320).all() and (pf[:, 1] <= 240).all()
        # each pair is an inlier of the record's own homography (plus the 0.1 px rounding)
        assert np.linalg.norm(_project(r.track.fix.H, pf) - pm, axis=1).max() <= RANSAC_PX + 0.2, r.idx


# --- 2. --jam --------------------------------------------------------------------------------------------------

def test_a_jam_is_lost_and_found_and_the_run_folder_shows_what_the_camera_sent(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=12)                # clean frames, 1 per second
    out = tmp_path / "out"
    res = _cli("--map", world[0], "--frames", frames, "--out", out, "--cache", tmp_path / "cache", "--jam", "5:3")
    assert res.returncode == 0, res.stderr
    recs = read_run(out / "run.jsonl")
    T, L, R = TrackMode.TRACKING, TrackMode.LOST, TrackMode.RELOCALIZING
    # t = 5, 6, 7 are jammed: two misses coasted while TRACKING, LOST at the third, then found again
    assert [r.track.state for r in recs] == [R, T, T, T, T, T, T, L, R, T, T, T]
    assert [r.idx for r in recs if r.extra.get("jammed")] == [5, 6, 7]
    assert all("jammed" not in r.extra for r in recs if r.idx not in (5, 6, 7))
    for r in recs:
        shown = cv2.imread(str(out / r.source))
        assert shown is not None and shown.shape == (240, 320, 3), r.idx
        clean = cv2.imread(str(frames / f"frame_{r.idx:03d}.png"))
        if r.idx in (5, 6, 7):
            assert (out / r.source).parent == out / "frames"            # the noise lives in the run folder
            assert float(np.abs(shown.astype(int) - clean).mean()) > 20  # and it is not the clean frame
        else:
            assert (out / r.source).samefile(frames / f"frame_{r.idx:03d}.png")
    assert all((frames / f"frame_{i:03d}.png").is_file() for i in range(12))    # the input is untouched


def test_the_same_jam_gives_the_same_run(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=10)
    for name in ("a", "b"):
        assert _cli("--map", world[0], "--frames", frames, "--out", tmp_path / name, "--cache", tmp_path / "cache",
                    "--jam", "4:2").returncode == 0
    a, b = (cv2.imread(str(tmp_path / n / "frames" / "frame_000004.jpg")) for n in ("a", "b"))
    assert a is not None and np.array_equal(a, b)


@pytest.mark.parametrize("value", ["5", "5:", "a:b", "-1:3", "5:0", "5:-2", "1:2:3", "5:inf"])
def test_a_bad_jam_is_a_usage_error(tmp_path, world, value):
    frames, _ = _frames(tmp_path, world, n=2)
    res = _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out", f"--jam={value}")
    assert res.returncode == 2 and "--jam" in res.stderr and "Traceback" not in res.stderr
    assert not (tmp_path / "out").exists()


def test_a_jam_over_the_whole_flight_localizes_nothing(tmp_path, world):
    frames, _ = _frames(tmp_path, world, n=3)
    res = _cli("--map", world[0], "--frames", frames, "--out", tmp_path / "out", "--cache", tmp_path / "cache",
               "--jam", "0:100")
    assert res.returncode == 1 and "localized" in res.stderr and not (tmp_path / "out").exists()
