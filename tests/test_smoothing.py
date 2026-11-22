"""M1-06 · Simple smoothing and gap interpolation.

"Done when: a run with 3 frames removed still draws a complete, ordered route with those points marked." So:
- every frame becomes a record, in frame order, measured or not;
- a frame without an accepted fix (no homography, or a rejected one) gets a position interpolated along the route
  between the accepted fixes around it (linearly in time), and is marked ``interpolated``; its fix stays in the
  record (a rejected fix is still telemetry);
- before the first fix / after the last one there is nothing to interpolate between: the nearest fix is held;
- optional light smoothing (off by default) of positions and headings: a centred moving average that keeps the
  ends and leaves a straight constant-speed flight unchanged; headings are averaged on the circle;
- the result goes through run.jsonl and the trajectory renderer: the route is complete, the 3 points are drawn as
  missed (no footprint) and counted in the legend.

Ground truth: a synthetic flight with exact poses; the data test blanks 3 task crops (``@pytest.mark.data``).
"""
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.synth import make_h, make_map

from dronenav.smoothing import Observation, SmoothConfig, fill_track
from dronenav.telemetry import RunInfo, read_run, write_run, write_run_info
from dronenav.types import Fix, TrackMode
from dronenav.viz.trajectory_image import render_trajectory

ROOT = Path(__file__).resolve().parents[1]
FRAME_SIZE = (480, 360)
FPS = 2.0
# a curved flight: x steps 40 px, y bends, heading turns 6° per frame through north (330° -> 48°)
N = 14
POSES = [(200.0 + 40 * k, 300.0 + 60 * math.sin(k / 3), (330.0 + 6 * k) % 360) for k in range(N)]
DROPPED = {4, 8, 9}                     # 3 frames "removed": one alone, two in a row


def _fix(i: int, x: float, y: float, heading: float, accepted: bool = True) -> Fix:
    H = make_h(heading, 0.5, cx=x, cy=y, frame_size=FRAME_SIZE)
    w, h = FRAME_SIZE
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ H.T
    return Fix(idx=i, H=H, x=x, y=y, heading_deg=heading, scale=0.5, footprint=corners[:, :2] / corners[:, 2:],
               inliers=120 if accepted else 6, inlier_ratio=0.4 if accepted else 0.03, reproj_px=1.0,
               accepted=accepted, reject_reason=None if accepted else "too_few_inliers")


def _obs(i: int, fix: Fix | None) -> Observation:
    return Observation(idx=i, t=i / FPS, source=f"frames/frame_{i:05d}.png", fix=fix, backend="rootsift", ms=50.0)


def _flight(dropped=DROPPED, rejected=frozenset()) -> list[Observation]:
    """The flight with ``dropped`` frames unmatched (no fix) and ``rejected`` ones with a rejected fix."""
    out = []
    for i, (x, y, h) in enumerate(POSES):
        if i in dropped:
            out.append(_obs(i, None))
        elif i in rejected:
            out.append(_obs(i, _fix(i, x + 900, y - 200, h, accepted=False)))     # a wrong place, rejected
        else:
            out.append(_obs(i, _fix(i, x, y, h)))
    return out


def _lerp(a, b, u):
    return a + (b - a) * u


# --- every frame, in order, the gaps marked ---------------------------------------------------------------------

def test_every_frame_gives_one_record_in_order():
    recs = fill_track(_flight())
    assert [r.idx for r in recs] == list(range(N))
    assert all(b.t > a.t for a, b in itertools.pairwise(recs))
    assert {r.idx for r in recs if r.interpolated} == DROPPED


def test_gaps_are_interpolated_along_the_route_in_time():
    recs = fill_track(_flight())
    # frame 4 sits halfway between 3 and 5; frames 8 and 9 at a third and two thirds between 7 and 10
    for i, a, b in ((4, 3, 5), (8, 7, 10), (9, 7, 10)):
        u = (i - a) / (b - a)
        x, y = _lerp(POSES[a][0], POSES[b][0], u), _lerp(POSES[a][1], POSES[b][1], u)
        assert recs[i].track.x == pytest.approx(x, abs=1e-9) and recs[i].track.y == pytest.approx(y, abs=1e-9)


def test_interpolation_follows_time_not_frame_count():
    # frame 5 comes late (a slow frame grab): it is closer in time to 6 than to 4, so closer in space too
    obs = _flight(dropped={5})
    obs[5] = obs[5]._replace(t=obs[4].t + 0.9 * (obs[6].t - obs[4].t))
    rec = fill_track(obs)[5]
    assert rec.track.x == pytest.approx(_lerp(POSES[4][0], POSES[6][0], 0.9), abs=1e-9)


def test_measured_frames_keep_their_fix_position_without_smoothing():
    recs = fill_track(_flight())
    for r in recs:
        if not r.interpolated:
            assert (r.track.x, r.track.y) == (POSES[r.idx][0], POSES[r.idx][1])
            assert r.track.fix.accepted and not r.interpolated


def test_a_rejected_fix_is_interpolated_too_and_stays_in_the_record():
    recs = fill_track(_flight(dropped=set(), rejected={6}))
    r = recs[6]
    assert r.interpolated and r.track.fix is not None and not r.track.fix.accepted   # telemetry keeps the reject
    assert r.track.x == pytest.approx(_lerp(POSES[5][0], POSES[7][0], 0.5), abs=1e-9)   # not the wrong place


def test_states_velocity_and_uncertainty():
    recs = fill_track(_flight())
    for r in recs:
        assert r.track.state == (TrackMode.LOST if r.interpolated else TrackMode.TRACKING)
    # velocity in map px per second: x moves 40 px per frame at 2 fps
    assert recs[2].track.vx == pytest.approx(80.0, abs=1e-6)
    # a filled-in point is less certain than a measured one, and more so deeper into a gap
    s = [math.sqrt(np.linalg.eigvalsh(r.track.cov).max()) for r in recs]
    assert s[4] > s[3] and s[8] > s[7]
    assert s[8] == pytest.approx(s[9])              # 8 and 9 are each one frame from a fix
    assert all(np.isfinite(r.track.cov).all() for r in recs)


def test_heading_is_recorded_for_every_frame_and_gaps_turn_the_short_way():
    recs = fill_track(_flight(dropped={4, 5, 6}))     # a gap across north: frame 3 is 348°, frame 7 is 12°
    for r in recs:
        assert 0.0 <= r.extra["heading_deg"] < 360.0
    # 6° per frame through north (354, 0, 6), not the long way back through south
    assert [recs[i].extra["heading_deg"] for i in (4, 6)] == pytest.approx([354.0, 6.0], abs=1e-9)
    assert min(recs[5].extra["heading_deg"], 360 - recs[5].extra["heading_deg"]) < 1e-9


def test_the_input_extra_and_timing_are_kept():
    obs = _flight()
    obs[2] = obs[2]._replace(extra={"search": "window"})
    r = fill_track(obs)[2]
    assert r.extra["search"] == "window" and r.backend == "rootsift" and r.ms == 50.0
    assert r.source == "frames/frame_00002.png"


# --- the ends: nothing to interpolate between -------------------------------------------------------------------

def test_leading_and_trailing_gaps_hold_the_nearest_fix():
    recs = fill_track(_flight(dropped={0, 1, N - 1}))
    assert [r.interpolated for r in recs[:3]] == [True, True, False]
    assert (recs[0].track.x, recs[0].track.y) == (recs[2].track.x, recs[2].track.y)
    assert (recs[-1].track.x, recs[-1].track.y) == (POSES[N - 2][0], POSES[N - 2][1])
    assert recs[-1].interpolated
    assert recs[0].extra["heading_deg"] == pytest.approx(POSES[2][2])


def test_no_accepted_fix_at_all_is_an_error():
    with pytest.raises(ValueError, match="no accepted fix"):
        fill_track(_flight(dropped=set(range(N))))


def test_empty_input_gives_no_records():
    assert fill_track([]) == []


@pytest.mark.parametrize("bad", ["idx", "t"])
def test_out_of_order_frames_are_rejected(bad):
    obs = _flight()
    obs[3], obs[4] = (obs[4], obs[3]) if bad == "idx" else (obs[3], obs[4]._replace(t=obs[2].t - 0.1))
    with pytest.raises(ValueError):
        fill_track(obs)


# --- optional smoothing -----------------------------------------------------------------------------------------

def test_smoothing_is_off_by_default():
    assert SmoothConfig().radius == 0


def test_smoothing_reduces_jitter_and_keeps_the_ends():
    rng = np.random.default_rng(0)
    obs = []
    for i in range(30):                                     # straight line, 1.5 px jitter on every fix
        x, y = 100.0 + 20 * i, 400.0
        obs.append(_obs(i, _fix(i, x + rng.normal(0, 1.5), y + rng.normal(0, 1.5), 90.0)))
    raw = fill_track(obs)
    smooth = fill_track(obs, SmoothConfig(radius=2))
    def err(recs):
        return np.mean([abs(r.track.y - 400.0) for r in recs[2:-2]])
    assert err(smooth) < 0.7 * err(raw)
    assert (smooth[0].track.x, smooth[0].track.y) == (raw[0].track.x, raw[0].track.y)
    assert (smooth[-1].track.x, smooth[-1].track.y) == (raw[-1].track.x, raw[-1].track.y)
    assert smooth[5].track.fix is raw[5].track.fix          # the measurement itself is never changed


def test_smoothing_leaves_a_straight_constant_speed_flight_unchanged():
    obs = [_obs(i, _fix(i, 100.0 + 20 * i, 50.0 + 7 * i, 120.0)) for i in range(10)]
    for r in fill_track(obs, SmoothConfig(radius=3)):
        assert r.track.x == pytest.approx(100.0 + 20 * r.idx) and r.track.y == pytest.approx(50.0 + 7 * r.idx)
        assert r.extra["heading_deg"] == pytest.approx(120.0)


def test_heading_smoothing_averages_on_the_circle():
    obs = [_obs(i, _fix(i, 100.0 + 20 * i, 50.0, h)) for i, h in enumerate([350.0, 10.0, 350.0, 10.0, 350.0])]
    recs = fill_track(obs, SmoothConfig(radius=1))
    mid = recs[2].extra["heading_deg"]
    assert min(mid, 360 - mid) < 5.0                     # around north, never ~180°


def test_config_rejects_a_negative_radius():
    with pytest.raises(ValueError):
        SmoothConfig(radius=-1)


# --- Done when: the run draws a complete, ordered route with those points marked -------------------------------

def _write(run_dir: Path, recs, mp) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(run_dir / "map.png"), mp)
    write_run_info(run_dir, RunInfo(map=str(run_dir / "map.png"), map_size=(mp.shape[1], mp.shape[0]),
                                    generator="test", frame_size=FRAME_SIZE, fps=FPS))
    write_run(run_dir / "run.jsonl", recs)


def test_a_run_with_3_frames_removed_draws_a_complete_marked_route(tmp_path):
    recs = fill_track(_flight())
    _write(tmp_path, recs, make_map())
    back = read_run(tmp_path / "run.jsonl")
    assert [r.idx for r in back] == list(range(N))
    assert {r.idx for r in back if r.interpolated} == DROPPED          # the marks survive run.jsonl
    _, layout = render_trajectory(tmp_path, label_every=1)
    assert layout.summary.startswith(f"{N} frames, {N - 3} fixed, 3 missed")
    assert not DROPPED & set(layout.footprints)                        # filled-in points have no footprint
    assert {lb.idx for lb in layout.labels if lb.idx is not None} >= DROPPED   # but they are numbered


TASK = ROOT / "data" / "hackatons-data"
PROBE = ROOT / "experiments" / "data_probe" / "results" / "probe.json"


@pytest.mark.data
@pytest.mark.skipif(not (TASK / "global_map.png").exists() or not PROBE.exists(), reason="data/ not downloaded")
def test_task_frames_with_3_removed_give_a_complete_route(tmp_path):
    from dronenav.matching import RootSiftMatcher
    from dronenav.search import Searcher

    truth = {r["i"]: r for r in json.loads(PROBE.read_text())["rows"]}
    mp = cv2.imread(str(TASK / "global_map.png"))
    s = Searcher(RootSiftMatcher(mp, cache_dir=tmp_path / "cache"), mp)
    removed = {12, 30, 31}
    obs = []
    for i in range(50):
        frame = cv2.imread(str(TASK / "crops" / f"crop_{i}.png"))
        loc = s.locate(i, np.zeros_like(frame) if i in removed else frame)
        obs.append(Observation(idx=i, t=float(i), source=f"crops/crop_{i}.png", fix=loc.fix, backend="rootsift",
                               ms=loc.ms, extra={"search": loc.mode.value}))
    recs = fill_track(obs)
    assert [r.idx for r in recs] == list(range(50))
    assert {r.idx for r in recs if r.interpolated} == removed
    for i in removed:                                   # filled in between its neighbours' true positions
        a, b = i - 1, i + 1 + (i == 30)
        lo = np.minimum([truth[a]["cx"], truth[a]["cy"]], [truth[b]["cx"], truth[b]["cy"]]) - 2
        hi = np.maximum([truth[a]["cx"], truth[a]["cy"]], [truth[b]["cx"], truth[b]["cy"]]) + 2
        assert np.all(lo <= (recs[i].track.x, recs[i].track.y)) and np.all((recs[i].track.x, recs[i].track.y) <= hi)
    _write(tmp_path / "run", recs, mp)
    _, layout = render_trajectory(tmp_path / "run")
    assert layout.summary.startswith("50 frames, 47 fixed, 3 missed")
