"""M9-02 · Visual odometry: frame-to-frame motion as the filter's motion input.

Done when: during a simulated jam of the map match (not the camera), the error grows more slowly with VO than
without.

The contract:
- ``frame_motion(prev, cur)``: the homography cur frame px -> prev frame px from the two images alone (features
  matched frame to frame, RANSAC), or None when they share too little to trust (noise, a blank frame, no overlap);
- ``VisualOdometry.step(t, image, fix)``: chains that motion onto the last frame -> map homography (reset by every
  accepted fix) and returns the drone's velocity in map px/s with its covariance, or None (the first frame, no fix
  yet to put it on the map, or no motion found);
- ``KalmanFilter.update_velocity(v, R)``: a velocity measurement, gated like a fix; ``Tracker.update(..., vo=)``
  uses it. A frame with no fix but with VO is not a miss (the track stays TRACKING and keeps its window) for up to
  ``vo_max_s`` seconds of VO alone; a gated fix is still a miss; the position does not hold while VO measures;
- the pipeline: ``vo=True`` feeds VO to the tracker; ``jam_match`` frames skip the map match (search "jammed"),
  the camera still works. ``scripts/run.py --vo --jam-match 8-19``.
"""
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.pipeline import localize_run, parse_frame_set
from dronenav.sim import CutoutSource, cutout_route
from dronenav.sim.cutout import render_cutout
from dronenav.telemetry import read_run
from dronenav.tracking.kalman import KalmanConfig, KalmanFilter
from dronenav.tracking.states import Tracker, TrackerConfig
from dronenav.types import Fix, TrackMode
from dronenav.vo import VisualOdometry, VoVelocity, frame_motion
from tests.synth import apply, make_h, make_map

T, L = TrackMode.TRACKING, TrackMode.LOST
ROOT = Path(__file__).resolve().parents[1]
FS = (320, 240)
CORNERS = np.array([(0, 0), (320, 0), (320, 240), (0, 240)], float)


def _big_map(seed=4):
    return make_map(2000, 1500, seed=seed, shapes=900 * 2000 * 1500 // (1200 * 900))


def _truth_fix(idx, H) -> Fix:
    """A perfect accepted fix with homography H (frame -> map)."""
    fp = apply(H, CORNERS)
    c = apply(H, np.array([[FS[0] / 2, FS[1] / 2]]))[0]
    return Fix(idx=idx, H=H, x=float(c[0]), y=float(c[1]), heading_deg=0.0, scale=1.0, footprint=fp, inliers=200,
               inlier_ratio=0.9, reproj_px=0.5, accepted=True)


# --- frame to frame -----------------------------------------------------------------------------------------------

def test_frame_motion_recovers_the_known_motion():
    m = make_map(seed=1)
    H1, H2 = make_h(30.0, 1.0, 500, 400, FS), make_h(36.0, 1.05, 570, 430, FS)    # 76 px on, 6° turn, 5 % higher
    f1, f2 = render_cutout(m, H1, FS), render_cutout(m, H2, FS)     # the renderer the cut-out flights use
    mo = frame_motion(f1, f2)
    assert mo is not None and mo.inliers >= 30
    want = apply(np.linalg.inv(H1) @ H2, CORNERS)
    # 1 px: noise-free renders, hundreds of matches; RANSAC on them is sub-pixel, the rest is the resampling
    assert np.abs(apply(mo.H, CORNERS) - want).max() <= 1.0


def test_frame_motion_refuses_frames_with_nothing_in_common():
    m = make_map(seed=1)
    f = render_cutout(m, make_h(0.0, 1.0, 500, 400, FS), FS)
    far = render_cutout(m, make_h(0.0, 1.0, 150, 150, FS), FS)          # no overlap with f
    noise = np.random.default_rng(0).integers(0, 256, f.shape, dtype=np.uint8)
    blank = np.full_like(f, 128)
    for other in (noise, blank, far):
        assert frame_motion(f, other) is None
        assert frame_motion(other, f) is None


# --- velocity on the map ------------------------------------------------------------------------------------------

def _flight(n=15, fps=2.0, seed=3, overlap=0.65):
    img = _big_map()
    poses = cutout_route((img.shape[1], img.shape[0]), n, seed=seed, frame_size=FS, scale=(0.9, 1.1),
                         tilt_max_deg=2.0, overlap=overlap)
    return img, CutoutSource(img, poses, FS, fps=fps)


def test_vo_velocity_follows_the_flight_after_one_fix():
    _, src = _flight()
    vo = VisualOdometry()
    out = [vo.step(f.t, f.image, _truth_fix(0, src.truth(0).H) if f.idx == 0 else None) for f in src]
    assert out[0] is None                                    # nothing to compare the first frame with
    for k in range(1, len(src)):
        v = out[k]
        assert isinstance(v, VoVelocity), k
        c0, c1 = np.array(src.truth(k - 1).center), np.array(src.truth(k).center)
        true_v = (c1 - c0) * src.fps
        # 5 % + 2 px a frame: chained over 14 frames with no fix, the frame-to-frame homographies' small scale and
        # rotation errors add up; a ~110 px step is still measured to a few px
        assert np.linalg.norm(v.v - true_v) <= 0.05 * np.linalg.norm(true_v) + 2.0 * src.fps, (k, v.v, true_v)
        assert np.all(np.linalg.eigvalsh(v.cov) > 0)


def test_vo_without_any_fix_has_no_map_to_measure_on():
    _, src = _flight(n=4)
    vo = VisualOdometry()
    assert [vo.step(f.t, f.image, None) for f in src] == [None] * 4


# --- the filter and the tracker -----------------------------------------------------------------------------------

def test_a_velocity_measurement_sets_the_speed_and_a_wild_one_is_gated():
    kf = KalmanFilter(KalmanConfig())
    kf.reset(0.0, (100.0, 100.0), np.eye(2) * 4.0)
    kf.step(1.0)
    assert kf.update_velocity((30.0, -10.0), np.eye(2) * 1.0) is True
    assert kf.x[2:] == pytest.approx([30.0, -10.0], abs=0.5)
    assert kf.P[2, 2] < 2.0 and kf.P[3, 3] < 2.0
    before = kf.x.copy(), kf.P.copy()
    assert kf.update_velocity((3000.0, 0.0), np.eye(2) * 1.0) is False         # far outside the gate: refused
    assert np.array_equal(kf.x, before[0]) and np.array_equal(kf.P, before[1])


def _circle_truth(n, speed=60.0, radius=800.0):
    """A drone circling at ``speed`` px/s, one state per second."""
    w = speed / radius
    return np.array([(2000 + radius * math.cos(w * k), 2000 + radius * math.sin(w * k)) for k in range(n)])


def _fix_at(idx, xy, sigma=2.0):
    x, y = float(xy[0]), float(xy[1])
    fp = np.array([(x - 160, y - 120), (x + 160, y - 120), (x + 160, y + 120), (x - 160, y + 120)])
    return Fix(idx=idx, H=np.eye(3), x=x, y=y, heading_deg=0.0, scale=1.0, footprint=fp, inliers=200,
               inlier_ratio=1.0, reproj_px=sigma, accepted=True)


def _jam_flight(use_vo, jam=range(20, 45), n=50, cfg=None, wrong=()):
    """Fixes (truth + 2 px noise) except in ``jam``; VO (the true mean velocity + 2 px/s noise) on every frame."""
    truth = _circle_truth(n)
    rng = np.random.default_rng(7)
    track = Tracker(cfg or TrackerConfig())
    errs, ups = [], []
    for k in range(n):
        fix = None if k in jam else _fix_at(k, truth[k] + rng.normal(0, 2.0, 2))
        if k in wrong:
            fix = _fix_at(k, truth[k] + (1500.0, -900.0))
        vo = None
        if use_vo and k > 0:
            vo = VoVelocity(v=truth[k] - truth[k - 1] + rng.normal(0, 2.0, 2), cov=np.eye(2) * 4.0, inliers=100)
        ups.append(track.update(k, float(k), fix, vo=vo))
        errs.append(float(np.hypot(*(track.kf.x[:2] - truth[k]))))
    return np.array(errs), ups


def test_with_vo_a_map_match_jam_keeps_tracking_and_the_error_grows_slower():
    jam = range(20, 45)
    e_vo, u_vo = _jam_flight(True, jam)
    e_no, u_no = _jam_flight(False, jam)
    assert [u.mode for u in u_no[22:45]] == [L] * 23                              # without VO: lost at the 3rd miss
    assert all(u.mode is T and u.misses == 0 for u in u_vo[jam.start:jam.stop])   # with VO: still tracking
    grow_vo, grow_no = e_vo[jam.stop - 1] - e_vo[jam.start - 1], e_no[jam.stop - 1] - e_no[jam.start - 1]
    # 25 s without the map: with no VO the track coasts 5 s then holds while the drone flies on at 60 px/s (~1000
    # px off); with VO only the velocity noise adds up (2 px/s, a random walk: ~10 px). A tenth is a wide margin
    assert grow_vo < 0.1 * grow_no, (grow_vo, grow_no)
    assert all(u.mode is T for u in u_vo[jam.stop:])


def test_a_gated_fix_is_still_a_miss_with_vo():
    _, ups = _jam_flight(True, jam=range(20, 30), n=35, wrong=(24, 25, 26))
    assert [u.mode for u in ups[24:27]] == [T, T, L]          # VO doesn't excuse a fix that disagrees


def test_vo_alone_is_trusted_for_vo_max_s_then_misses_count_again():
    cfg = TrackerConfig(vo_max_s=10.0)
    _, ups = _jam_flight(True, jam=range(20, 45), cfg=cfg)
    modes = [u.mode for u in ups]
    assert modes[20:30] == [T] * 10                           # 10 s after the last fix (t = 19): VO alone
    assert L in modes[30:34]                                  # then the misses count: LOST within lost_after
    with pytest.raises(ValueError):
        TrackerConfig(vo_max_s=-1.0)


# --- the pipeline: a jam of the map match, the camera still working (Done when) ------------------------------------

def _frames_dir(tmp_path, n=24):
    img, src = _flight(n=n, fps=1.0, seed=5)
    cv2.imwrite(str(tmp_path / "map.png"), img)
    d = tmp_path / "frames"
    d.mkdir()
    for f in src:
        cv2.imwrite(str(d / f"frame_{f.idx:03d}.png"), f.image)
    return tmp_path / "map.png", d, src


def _errors(run_dir, src):
    out = []
    for r in read_run(run_dir / "run.jsonl"):
        c = src.truth(r.idx).center
        out.append(math.inf if r.track.x is None else math.hypot(r.track.x - c[0], r.track.y - c[1]))
    return np.array(out)


def test_in_a_map_match_jam_the_error_grows_slower_with_vo(tmp_path):
    map_path, frames, src = _frames_dir(tmp_path)
    jam = range(8, 20)
    kw = {"cache_dir": tmp_path / "cache", "jam_match": set(jam)}
    localize_run(map_path, frames, tmp_path / "no_vo", vo=False, **kw)
    localize_run(map_path, frames, tmp_path / "vo", vo=True, **kw)
    recs = read_run(tmp_path / "vo" / "run.jsonl")
    assert [r.extra["search"] == "jammed" for r in recs] == [k in jam for k in range(len(src))]
    assert all(r.track.fix is None for r in recs if r.idx in jam)          # no map match at all in the jam
    assert all(r.extra["vo"] is not None for r in recs[1:])                 # the camera kept working
    e_vo, e_no = _errors(tmp_path / "vo", src), _errors(tmp_path / "no_vo", src)
    j = list(jam)
    # the drone flies ~110 px a frame: coasting then holding falls behind by hundreds of px within the jam, while
    # chained frame-to-frame motion stays within tens
    assert e_vo[j].mean() < 0.5 * e_no[j].mean(), (e_vo[j], e_no[j])
    assert e_vo[j[-1]] < e_no[j[-1]]
    assert all(r.track.state is T for r in recs if r.idx in jam)          # never lost: the window keeps following
    assert all(r.track.state is T for r in recs[jam.stop + 1:])            # and the map match takes over again


def test_vo_off_by_default_records_say_none(tmp_path):
    map_path, frames, _ = _frames_dir(tmp_path, n=4)
    localize_run(map_path, frames, tmp_path / "out", cache_dir=tmp_path / "cache")
    assert all(r.extra.get("vo") is None and r.extra["search"] != "jammed" for r in read_run(tmp_path / "out" / "run.jsonl"))


# --- the options --------------------------------------------------------------------------------------------------

def test_parse_frame_set():
    assert parse_frame_set("3-5,9") == {3, 4, 5, 9}
    assert parse_frame_set("7") == {7}
    assert parse_frame_set("") == set()
    for bad in ("5-3", "a", "1,,2", "-2"):
        with pytest.raises(ValueError):
            parse_frame_set(bad)


def _cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "run.py"), *map(str, args)], check=False,
                          capture_output=True, text=True, cwd=ROOT)


def test_run_cli_takes_vo_and_jam_match(tmp_path):
    map_path, frames, _ = _frames_dir(tmp_path, n=6)
    res = _cli("--map", map_path, "--frames", frames, "--out", tmp_path / "out", "--cache", tmp_path / "cache",
               "--vo", "--jam-match", "2-3")
    assert res.returncode == 0, res.stderr
    recs = read_run(tmp_path / "out" / "run.jsonl")
    assert [r.extra["search"] == "jammed" for r in recs] == [k in (2, 3) for k in range(6)]
    assert _cli("--map", map_path, "--frames", frames, "--out", tmp_path / "o2", "--vo", "--tracker", "none"
                ).returncode == 2                                   # VO feeds the tracker: needs one
    assert _cli("--map", map_path, "--frames", frames, "--out", tmp_path / "o3", "--jam-match", "x").returncode == 2

