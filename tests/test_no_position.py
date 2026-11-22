"""M0-12 · TrackState: no position before the first fix (issue #87, option 2: an optional position).

"Done when: the choice is written in types.py (docstring + validation) and in the M0 spec, with a test for a
record before the first fix." The decision (recorded in the issue):

- ``x``, ``y``, ``vx``, ``vy`` and ``cov`` are all set or all None (``has_position`` says which); only LOST and
  RELOCALIZING may have no position, TRACKING without one is a contradiction;
- run.jsonl writes null for the five and reads None back, checking the same rule;
- consumers follow: the trajectory image and video draw only records with a position (a run with none is a clear
  error), and accuracy counts such a record as not localized, with no track error.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.accuracy import score
from dronenav.telemetry import (
    Record,
    RunInfo,
    RunWriter,
    TelemetryError,
    read_run,
    record_from_dict,
    record_to_dict,
    write_run_info,
)
from dronenav.types import Fix, TrackMode, TrackState
from dronenav.viz.trajectory_image import render_trajectory
from dronenav.viz.trajectory_video import plan_video, write_video

NONE5 = {"x": None, "y": None, "vx": None, "vy": None, "cov": None}


def _unknown(idx: int, state=TrackMode.LOST) -> TrackState:
    return TrackState(idx=idx, state=state, **NONE5)


def _known(idx: int, x: float, y: float) -> TrackState:
    H = np.array([[1.0, 0, x - 32], [0, 1.0, y - 32], [0, 0, 1]])
    fp = np.array([[x - 32, y - 32], [x + 32, y - 32], [x + 32, y + 32], [x - 32, y + 32]])
    fix = Fix(idx=idx, H=H, x=x, y=y, heading_deg=90.0, scale=1.0, footprint=fp, inliers=200, inlier_ratio=0.6,
              reproj_px=0.8, accepted=True)
    return TrackState(idx=idx, state=TrackMode.TRACKING, x=x, y=y, vx=0.0, vy=0.0, cov=np.eye(2), fix=fix)


def _rec(track: TrackState) -> Record:
    return Record(idx=track.idx, t=float(track.idx), source=f"frames/frame_{track.idx:05d}.png", track=track)


# --- the type --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("state", [TrackMode.LOST, TrackMode.RELOCALIZING])
def test_a_record_before_the_first_fix_has_no_position(state):
    tr = _unknown(0, state)
    assert not tr.has_position
    assert tr.x is None and tr.y is None and tr.vx is None and tr.vy is None and tr.cov is None


def test_a_record_with_a_position_says_so():
    assert _known(3, 100.0, 50.0).has_position


def test_tracking_needs_a_position():
    with pytest.raises(ValueError, match="TRACKING"):
        _unknown(0, TrackMode.TRACKING)


@pytest.mark.parametrize("missing", ["x", "y", "vx", "vy", "cov"])
def test_a_position_is_all_or_nothing(missing):
    full = {"x": 1.0, "y": 2.0, "vx": 0.0, "vy": 0.0, "cov": np.eye(2)}
    with pytest.raises(ValueError, match="all"):
        TrackState(idx=0, state=TrackMode.LOST, **{**full, missing: None})


# --- run.jsonl -------------------------------------------------------------------------------------------------

def test_run_jsonl_writes_null_and_reads_none_back(tmp_path):
    path = tmp_path / "run.jsonl"
    with RunWriter(path) as wr:
        wr.write(_rec(_unknown(0)))
        wr.write(_rec(_known(1, 100.0, 50.0)))
    first = json.loads(path.read_text().splitlines()[0])
    assert all(first[k] is None for k in NONE5)
    recs = read_run(path)
    assert not recs[0].track.has_position and recs[0].track.state is TrackMode.LOST
    assert recs[1].track.has_position and recs[1].track.x == 100.0


def test_a_line_with_half_a_position_is_rejected():
    d = record_to_dict(_rec(_known(0, 100.0, 50.0)))
    d["x"] = None
    with pytest.raises(TelemetryError):
        record_from_dict(d)


# --- consumers -------------------------------------------------------------------------------------------------

def test_accuracy_counts_no_position_as_not_localized():
    recs = [_rec(_unknown(0)), _rec(_unknown(1)), _rec(_known(2, 100.0, 50.0)), _rec(_known(3, 110.0, 50.0))]
    truth = {i: (100.0 + 10 * (i - 2), 50.0) for i in range(4)}
    acc = score(recs, truth)
    assert [f.fix_px for f in acc.frames[:2]] == [None, None]
    assert [f.track_px for f in acc.frames[:2]] == [None, None]
    assert acc.localized == 2 and acc.n == 4
    assert math.isclose(acc.frames[2].track_px, 0.0, abs_tol=1e-9)
    json.dumps(acc.summary())                       # the summary still serialises


def _run(tmp_path, tracks) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    cv2.imwrite(str(d / "map.png"), np.full((400, 600, 3), 128, np.uint8))
    write_run_info(d, RunInfo(map=str(d / "map.png"), map_size=(600, 400), generator="test", frame_size=(64, 64),
                              fps=1.0))
    with RunWriter(d / "run.jsonl") as wr:
        for tr in tracks:
            wr.write(_rec(tr))
    return d


def _flight(n_unknown=3, n_known=6):
    return [_unknown(i) for i in range(n_unknown)] + \
           [_known(n_unknown + k, 100.0 + 60 * k, 200.0) for k in range(n_known)]


def test_the_trajectory_image_starts_at_the_first_fix(tmp_path):
    d = _run(tmp_path, _flight())
    img, _ = render_trajectory(d)
    assert img.shape == (400, 600, 3)                 # drawn over the whole map, from the first fix on


def test_the_video_plan_follows_only_records_with_a_position(tmp_path):
    d = _run(tmp_path, _flight())
    recs = read_run(d / "run.jsonl")
    info = RunInfo(map=str(d / "map.png"), map_size=(600, 400), generator="test", frame_size=(64, 64), fps=1.0)
    plan = plan_video(recs, info, fps=5.0, size=(320, 180))
    shown = {recs[f.rec].idx for f in plan.frames}
    assert shown and min(shown) == 3                  # the first fix; frames 0-2 have nothing to show
    assert all(math.isfinite(f.x) and math.isfinite(f.y) for f in plan.frames)


def test_the_video_renders_a_run_that_starts_without_a_position(tmp_path):
    d = _run(tmp_path, _flight(n_known=3))
    out = write_video(d, fps=5.0, size=(320, 180))
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.parametrize("draw", ["image", "video"])
def test_a_run_with_no_position_at_all_is_a_clear_error(tmp_path, draw):
    d = _run(tmp_path, [_unknown(i) for i in range(3)])
    with pytest.raises(ValueError, match="position"):
        render_trajectory(d) if draw == "image" else write_video(d, fps=5.0, size=(320, 180))
