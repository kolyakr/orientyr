"""M7-02 · HUD video v2.

"Done when: a video of a run with a jam reads clearly without narration." Only a person can judge "reads clearly";
the checkable contract behind it:
1. the pure helpers: confidence -> trail colour, covariance -> 3-sigma ellipse, the layout (stage, side panel,
   dock), the camera (eased: no jump across a relocalisation; clamped: the stage never shows anything but map), the
   projection from the camera picture to its footprint (two lines that cannot cross), caption timing and wording;
2. each frame shows the right thing: the state lamp's colour, the alarm while LOST, trail colours, the ellipse, a
   rejected fix and its reason, the camera picture and its projection, match lines only from the run's own match
   points, the caption, no text too small for a demo screen;
3. the CLI writes the HUD video by default (H.264, the right size and frame count) and still writes the M1 one;
   square and wide maps, a 442-frame run.

Hand-made runs on a plain grey map make every drawn pixel and position known.
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

from dronenav.confidence import fix_confidence
from dronenav.fake_run import FakeRunConfig, make_fake_run
from dronenav.telemetry import Record, RunInfo, RunWriter, read_run, read_run_info, write_run_info
from dronenav.types import Fix, TrackMode, TrackState
from dronenav.viz import hud_video as hv

ROOT = Path(__file__).resolve().parents[1]
GREY = 128
BG = round(GREY * hv.MAP_DIM)   # the HUD video darkens the map: this is what bare map looks like in it
FRAME = (96, 64)            # frame px
SCALE = 2.0                 # map px per frame px: a 192 x 128 map px footprint
STEP = 30.0                 # map px between records
MAP = (4000, 2400)
SURE = {"inliers": 2000, "inlier_ratio": 0.9, "reproj_px": 0.5}       # confidence 0.89
UNSURE = {"inliers": 30, "inlier_ratio": 0.3, "reproj_px": 2.5}       # confidence 0.08
JUMP = 400.0                # map px the position jumps sideways when the tracker relocalises


# --- hand-made runs --------------------------------------------------------------------------------------------

def _H(x, y, heading_deg, scale=SCALE, frame=FRAME):
    """Frame px -> map px, conventions of dronenav.types (heading clockwise from up, y down)."""
    w, h = frame
    a = math.radians(heading_deg)
    sR = scale * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    t = np.array([x, y]) - sR @ np.array([w / 2, h / 2])
    return np.array([[*sR[0], t[0]], [*sR[1], t[1]], [0, 0, 1.0]])


def _fix(i, x, y, heading=0.0, accepted=True, reason=None, **quality):
    H = _H(x, y, heading)
    w, h = FRAME
    p = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ H.T
    q = {**SURE, **quality}
    return Fix(idx=i, H=H, x=x, y=y, heading_deg=heading, scale=SCALE, footprint=p[:, :2] / p[:, 2:],
               accepted=accepted, reject_reason=reason, **q)


def _frame_image(i):
    w, h = FRAME
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = ((37 * i) % 200 + 30, (91 * i) % 200 + 30, (53 * i) % 200 + 30)
    ramp = (np.add.outer(np.arange(h), np.arange(w)) % 64).astype(np.uint8)
    img[..., 0] = np.clip(img[..., 0].astype(int) + ramp, 0, 255)
    return img


def _write(tmp_path, rows, map_size=MAP, write_frames=True, name="run"):
    """rows: dicts with x, y, state and optionally fix, cov, extra. A run folder over a plain grey map, 1 fps."""
    w, h = map_size
    d = tmp_path / name
    (d / "frames").mkdir(parents=True, exist_ok=True)
    m = d / "map.png"
    cv2.imwrite(str(m), np.full((h, w, 3), GREY, np.uint8))
    write_run_info(d, RunInfo(map=str(m), map_size=(w, h), generator="test", frame_size=FRAME, fps=1.0))
    with RunWriter(d / "run.jsonl") as wr:
        for i, r in enumerate(rows):
            src = f"frames/frame_{i:05d}.png"
            if write_frames:
                cv2.imwrite(str(d / src), _frame_image(i))
            track = TrackState(idx=i, state=r["state"], x=r["x"], y=r["y"], vx=STEP, vy=0.0,
                               cov=r.get("cov", np.eye(2)), fix=r.get("fix"))
            wr.write(Record(idx=i, t=float(i), source=src, track=track, extra=r.get("extra", {})))
    return d


def _jam_rows(x0=1500.0, y0=1200.0, matches=False):
    """17 records flying east. 0-3 sure, 4-5 unsure, 6 sure, 7-9 LOST (the ellipse grows), 10 RELOCALIZING with a
    rejected fix above the route, 11 RELOCALIZING with a good fix, 12-16 TRACKING again."""
    rows = []
    for i in range(17):
        x, y = x0 + STEP * i, y0
        r = {"x": x, "y": y, "state": TrackMode.TRACKING}
        if i in (7, 8, 9):
            s = 20.0 * (i - 6)
            r.update(state=TrackMode.LOST, cov=np.diag([s * s, (s / 2) ** 2]))
        elif i == 10:
            r.update(state=TrackMode.RELOCALIZING, cov=np.diag([80.0 ** 2, 40.0 ** 2]),
                     fix=_fix(i, x, y - 110, accepted=False, reason="too_few_inliers", inliers=5, inlier_ratio=0.1,
                              reproj_px=6.0))
        else:
            r["fix"] = _fix(i, x, y, heading=90.0, **(UNSURE if i in (4, 5) else SURE))
            if i == 11:
                r["state"] = TrackMode.RELOCALIZING
        rows.append(r)
    rows[7]["extra"] = {"event": "camera jammed"}
    rows[12]["extra"] = {"event": "relocalized"}                  # 10 changes state with no event text of its own
    if matches:
        pf = np.array([[10.0, 10.0], [80.0, 12.0], [50.0, 50.0]])
        pm = (np.c_[pf, np.ones(3)] @ rows[3]["fix"].H.T)[:, :2]
        rows[3]["extra"] = {"matches": {"frame": pf.tolist(), "map": pm.tolist()}}
    return rows


def _load(d):
    return read_run(d / "run.jsonl"), read_run_info(d)


def _close(px, colour, tol=45):
    return np.abs(np.asarray(px).astype(int) - np.array(colour)).max() <= tol


@pytest.fixture(scope="module")
def jam(tmp_path_factory):
    """The jam run planned at 10 video frames per record, and its renderer: video frame 10 * i is record i."""
    d = _write(tmp_path_factory.mktemp("jam"), _jam_rows())
    records, info = _load(d)
    plan = hv.plan_hud_video(records, info, fps=10, speed=1.0)
    return {"dir": d, "records": records, "info": info, "plan": plan,
            "renderer": hv.HudRenderer(d, records, info, plan)}


def _px(plan, k, x, y):
    """Map px -> video px (floats) at plan frame k."""
    return hv.to_video(plan, k, x, y)


def _inside(inner, outer):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _apart(a, b):
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def _row(img, plan, k, x0, x1, y):
    """The video pixels along the map segment (x0, y) - (x1, y)."""
    (a, v), (b, _) = _px(plan, k, x0, y), _px(plan, k, x1, y)
    return img[round(v), round(a):round(b)]


# --- 1. pure helpers -------------------------------------------------------------------------------------------

def test_the_trail_colour_goes_from_red_through_yellow_to_green():
    assert hv.confidence_colour(None) == hv.RED                    # no fix: dead reckoning
    assert hv.confidence_colour(1.0) == hv.GREEN
    assert hv.confidence_colour(0.0) == hv.YELLOW                  # a fix, but a weak one
    assert hv.confidence_colour(hv.CONF_HIGH) == hv.GREEN and hv.confidence_colour(hv.CONF_LOW) == hv.YELLOW
    mid = hv.confidence_colour((hv.CONF_LOW + hv.CONF_HIGH) / 2)
    for c, lo, hi in zip(mid, hv.YELLOW, hv.GREEN, strict=True):
        assert min(lo, hi) <= c <= max(lo, hi)
    assert mid not in (hv.GREEN, hv.YELLOW)
    # the three anchors are told apart by eye: far apart in BGR
    for a, b in ((hv.GREEN, hv.YELLOW), (hv.YELLOW, hv.RED), (hv.GREEN, hv.RED)):
        assert np.abs(np.array(a) - np.array(b)).max() >= 100


def test_the_state_colours_are_green_red_and_amber():
    assert set(hv.STATE_COLOURS) == {"TRACKING", "LOST", "RELOCALIZING"}
    b, g, r = hv.STATE_COLOURS["TRACKING"]
    assert g > r + 60 and g > b + 60
    b, g, r = hv.STATE_COLOURS["LOST"]
    assert r > g + 100 and r > b + 100
    b, g, r = hv.STATE_COLOURS["RELOCALIZING"]
    assert r > b + 100 and g > b + 60 and r >= g


def test_the_layout_is_a_stage_a_side_panel_and_a_dock():
    lay = hv.layout((1920, 1080))
    assert lay.u == 1.0
    assert lay.stage == (0, 0, 1400, 948) and lay.side == (1400, 0, 1920, 1080) and lay.dock == (0, 948, 1400, 1080)
    half = hv.layout((960, 540))                                     # everything scales with the video
    assert half.u == 0.5 and half.stage == (0, 0, 700, 474) and half.side == (700, 0, 960, 540)
    for size in ((1920, 1080), (1280, 720), (640, 360), (1280, 1024), (320, 180)):
        lay = hv.layout(size)
        w, h = size
        # the three tile the video: no gap, no overlap
        assert lay.stage[:2] == (0, 0) and lay.stage[2] == lay.side[0] == lay.dock[2] and lay.stage[3] == lay.dock[1]
        assert lay.side[1:] == (0, w, h) and lay.dock[0] == 0 and lay.dock[3] == h
        assert lay.stage[2] >= 0.6 * w and lay.stage[3] >= 0.8 * h   # the map keeps most of the screen


def test_the_camera_is_clamped_so_the_stage_shows_only_map():
    cam = np.array([[500.0, 300.0], [50.0, 20.0], [990.0, 595.0], [-40.0, 700.0]])
    got = hv.clamp_camera(cam, (1000, 600), (200.0, 100.0))          # the stage shows 400 x 200 map px
    assert got.tolist() == [[500.0, 300.0], [200.0, 100.0], [800.0, 500.0], [200.0, 500.0]]
    # a map smaller than the stage in one direction cannot fill it: it is centred there, and still clamped in the other
    assert hv.clamp_camera(np.array([[10.0, 10.0]]), (1000, 150), (200.0, 100.0)).tolist() == [[200.0, 75.0]]
    assert cam[1].tolist() == [50.0, 20.0]                           # the input is not changed


def test_the_projection_is_the_two_outer_lines_from_the_footprint_to_the_picture():
    foot = np.array([[0.0, 2.0], [10.0, 2.0], [10.0, 8.0], [0.0, 8.0]])
    links = hv.projection_links(foot, (20, 0, 30, 10))
    # the hull of the two shapes: its two edges that join them. (10, 2) and (10, 8) lie inside it.
    assert sorted(links) == [((0.0, 2.0), (20.0, 0.0)), ((0.0, 8.0), (20.0, 10.0))]
    assert hv.projection_links(foot[[2, 3, 0, 1]], (20, 0, 30, 10)) and \
        sorted(hv.projection_links(foot[[2, 3, 0, 1]], (20, 0, 30, 10))) == sorted(links)   # corner order is no matter


def _cross(p, q, r, s):
    """Do the open segments p-q and r-s cross?"""
    def side(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return side(p, q, r) * side(p, q, s) < 0 and side(r, s, p) * side(r, s, q) < 0


@pytest.mark.parametrize("heading", range(0, 360, 15))
def test_the_projection_lines_never_cross_whatever_the_heading(heading):
    a = math.radians(heading)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    foot = np.array([[-60.0, -40.0], [60.0, -40.0], [60.0, 40.0], [-60.0, 40.0]]) @ rot.T + (300.0, 400.0)
    box = (900, 60, 1200, 290)
    links = hv.projection_links(foot, box)
    assert len(links) == 2
    (p, q), (r, s) = links
    assert not _cross(p, q, r, s)
    for a_end, b_end in links:                                       # from a footprint corner to a picture corner
        assert min(np.hypot(*(foot - a_end).T)) < 1e-6
        assert b_end[0] in box[::2] and b_end[1] in box[1::2]
    # neither line cuts through the footprint: every corner is on one side of it (or on it)
    for p, q in links:
        d = [(q[0] - p[0]) * (c[1] - p[1]) - (q[1] - p[1]) * (c[0] - p[0]) for c in foot]
        assert min(d) >= -1e-6 or max(d) <= 1e-6


def test_no_projection_when_the_footprint_covers_the_picture():
    foot = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    assert hv.projection_links(foot, (40, 40, 60, 60)) == []


@pytest.mark.parametrize("sigma, word", [(0.0, "very precise"), (3.0, "very precise"), (3.1, "precise"),
                                         (15.0, "precise"), (15.1, "drifting"), (40.0, "drifting"), (40.1, "wide"),
                                         (7000.0, "wide")])
def test_the_uncertainty_has_a_word(sigma, word):
    assert hv.uncertainty_word(sigma) == word


@pytest.mark.parametrize("cov, want", [
    (np.diag([4.0, 1.0]), (6.0, 3.0, 0.0)),                        # 3 sigma of sigma 2 and 1; the long axis along x
    (np.diag([1.0, 9.0]), (9.0, 3.0, 90.0)),                       # ... along y
    (np.array([[5.0, 4.0], [4.0, 5.0]]), (9.0, 3.0, 45.0)),        # eigenvalues 9 and 1, along (1, 1)
    (np.array([[5.0, -4.0], [-4.0, 5.0]]), (9.0, 3.0, 135.0)),     # ... along (1, -1)
])
def test_the_ellipse_axes_come_from_the_covariance(cov, want):
    a, b, angle = hv.sigma_ellipse(cov)
    assert (a, b) == pytest.approx(want[:2], abs=1e-9)
    assert angle % 180 == pytest.approx(want[2], abs=1e-6)


def test_the_ellipse_of_a_round_or_empty_or_broken_covariance():
    a, b, _ = hv.sigma_ellipse(np.eye(2) * 16.0, k=2.0)
    assert (a, b) == pytest.approx((8.0, 8.0))
    assert hv.sigma_ellipse(np.zeros((2, 2)))[:2] == (0.0, 0.0)
    # rounding can leave a hair of a negative eigenvalue: no NaN, no error
    a, b, _ = hv.sigma_ellipse(np.array([[1.0, 1.0], [1.0, 1.0 - 1e-12]]))
    assert math.isfinite(a) and b == pytest.approx(0.0, abs=1e-5)


def _reloc_rows(n=40, jump_at=24):
    """A steady flight east whose position jumps JUMP px south between two records: a relocalisation."""
    rows = []
    for i in range(n):
        x, y = 1200.0 + STEP * i, 900.0 + (JUMP if i >= jump_at else 0.0)
        lost = jump_at - 6 <= i < jump_at
        rows.append({"x": x, "y": y, "state": TrackMode.LOST if lost else TrackMode.TRACKING,
                     "fix": None if lost else _fix(i, x, y, heading=90.0)})
    return rows


def test_the_camera_glides_across_a_relocalisation(tmp_path):
    records, info = _load(_write(tmp_path, _reloc_rows()))
    plan = hv.plan_hud_video(records, info, fps=30, speed=8.0)       # the jump takes 1/8 s: under 4 video frames
    s = plan.size[0] / plan.view_w
    drone = np.array([[f.x, f.y] for f in plan.frames]) * s
    cam = np.array([[f.cx, f.cy] for f in plan.frames]) * s
    drone_step = np.hypot(*np.diff(drone, axis=0).T)
    cam_step = np.hypot(*np.diff(cam, axis=0).T)
    assert drone_step.max() > 100                                    # the drone itself does jump: ~630 px in 4 frames
    # the camera spreads it over about a second. 30 px a frame at 1280 px wide is a fast pan, not a cut; a camera
    # locked to the drone would move by drone_step.max()
    assert cam_step.max() <= 30, cam_step.max()
    assert np.abs(np.diff(cam_step)).max() <= 4                      # and it speeds up and slows down gradually
    assert np.hypot(*(cam[-1] - drone[-1])) < 1                      # ... and ends on the drone
    k = int(np.argmax(drone_step))
    _, y0, _, y1 = plan.stage
    assert np.hypot(*(cam - drone).T)[k - 45:k + 45].max() < (y1 - y0) / 2      # the drone never leaves the stage


def test_the_camera_stays_on_the_drone_in_steady_flight(tmp_path):
    rows = [{"x": 1200.0 + STEP * i, "y": 900.0, "state": TrackMode.TRACKING,
             "fix": _fix(i, 1200.0 + STEP * i, 900.0, heading=90.0)} for i in range(30)]
    records, info = _load(_write(tmp_path, rows))
    plan = hv.plan_hud_video(records, info, fps=30, speed=4.0)
    s = plan.size[0] / plan.view_w
    off = [math.hypot(f.cx - f.x, f.cy - f.y) * s for f in plan.frames]
    # constant velocity: the eased camera has no lag, except while it starts and stops (under a 20th of the width)
    assert max(off[60:-90]) < 2 and max(off) < plan.size[0] / 20


def test_the_zoom_shows_the_footprint_large(tmp_path):
    records, info = _load(_write(tmp_path, _reloc_rows()))
    plan = hv.plan_hud_video(records, info, size=(1920, 1080))
    diag = math.hypot(FRAME[0] * SCALE, FRAME[1] * SCALE) * plan.size[0] / plan.view_w
    assert 0.2 * 1920 <= diag <= 0.4 * 1920                          # the footprint reads, and so does its surrounding


def test_the_plan_keeps_the_m1_timing(tmp_path):
    records, info = _load(_write(tmp_path, _jam_rows()))
    plan = hv.plan_hud_video(records, info, fps=10, speed=1.0)
    assert len(plan.frames) == 16 * 10 + 1 + 10                      # the run, then a 1 s hold
    assert plan.frames[50].rec == 5 and (plan.frames[50].x, plan.frames[50].y) == pytest.approx((1650.0, 1200.0))


def test_captions_appear_at_the_event_and_stay_a_couple_of_seconds(jam):
    plan = jam["plan"]                                               # 10 video frames per second and per record
    assert 2.0 <= hv.CAPTION_S <= 4.0
    assert hv.caption_at(plan, 69) is None
    text, alpha = hv.caption_at(plan, 70)
    assert "frame 7" in text and "Signal lost" in text and "camera jammed" in text and "dead reckoning" in text
    assert alpha > 0
    text2, alpha2 = hv.caption_at(plan, 70 + 20)                     # two seconds later: still fully there
    assert text2 == text and alpha2 == pytest.approx(1.0)
    # the next event (record 10) replaces it at once
    text3, _ = hv.caption_at(plan, 100)
    assert "frame 10" in text3 and "Relocalizing" in text3 and "too few inliers" in text3
    # record 11 is still RELOCALIZING: no new caption; record 12 has one, and it goes away after CAPTION_S
    assert hv.caption_at(plan, 110)[0] == text3
    last, _ = hv.caption_at(plan, 120)
    assert "frame 12" in last and "Position recovered" in last and "relocalized" in last
    gone = 120 + math.ceil(hv.CAPTION_S * 10) + 1
    assert gone < len(plan.frames) and hv.caption_at(plan, gone) is None
    fading = hv.caption_at(plan, gone - 2)
    assert fading is not None and 0 < fading[1] < 1                  # it fades out, it does not blink off


def test_a_caption_is_a_title_and_a_short_note(jam, tmp_path):
    lost, reloc, back = jam["plan"].captions
    assert (lost.idx, lost.state, lost.title) == (7, "LOST", "Signal lost")
    assert lost.note == "frame 7 · camera jammed · dead reckoning"
    assert (reloc.title, reloc.note) == ("Relocalizing", "frame 10 · match rejected: too few inliers")
    assert (back.title, back.note) == ("Position recovered", "frame 12 · relocalized")
    # without an event text of its own the note comes from the telemetry: a jammed frame says so, and the frame
    # that finds the track again says how many points it matched
    rows = _jam_rows()
    rows[7]["extra"] = {"jammed": True}
    rows[12]["extra"] = {}
    rows[10] = {"x": rows[10]["x"], "y": rows[10]["y"], "state": TrackMode.RELOCALIZING}      # no fix at all
    records, info = _load(_write(tmp_path, rows))
    lost, reloc, back = hv.plan_hud_video(records, info, fps=10).captions
    assert lost.note == "frame 7 · camera jammed · dead reckoning"
    assert reloc.note == "frame 10 · searching the whole map"
    assert back.note == "frame 12 · 2 000 points matched"
    rows[7]["extra"] = {}
    records, info = _load(_write(tmp_path, rows, name="plain"))
    assert hv.plan_hud_video(records, info, fps=10).captions[0].note == "frame 7 · no match · dead reckoning"


def test_the_first_fix_of_a_run_is_announced_as_a_lock_not_a_recovery(tmp_path):
    rows = [{"x": 1200.0 + STEP * i, "y": 900.0, "state": TrackMode.TRACKING,
             "fix": _fix(i, 1200.0 + STEP * i, 900.0)} for i in range(5)]
    rows[0]["state"] = TrackMode.RELOCALIZING                        # the tracker's first frame, as run.py writes it
    records, info = _load(_write(tmp_path, rows))
    caps = hv.plan_hud_video(records, info, fps=10).captions
    assert [(c.idx, c.title) for c in caps] == [(1, "Position locked")]


def test_a_caption_is_not_replaced_before_it_can_be_read(tmp_path):
    records, info = _load(_write(tmp_path, _jam_rows()))
    plan = hv.plan_hud_video(records, info, fps=30, speed=8.0)       # records 10, 11, 12: 1/8 s of video apart
    starts = [c.frame for c in plan.captions]
    assert len(starts) == 3                                          # LOST (7), RELOCALIZING (10), TRACKING (12)
    least = math.ceil(hv.CAPTION_MIN_S * 30)
    assert all(b - a >= least for a, b in itertools.pairwise(starts))
    assert hv.CAPTION_MIN_S >= 1.0                                   # a line of text needs about a second
    k = starts[1]                                                    # the RELOCALIZING caption, while it is up
    assert "Relocalizing" in hv.caption_at(plan, k)[0] and "Relocalizing" in hv.caption_at(plan, k + least - 1)[0]


def test_a_run_without_events_has_no_captions(tmp_path):
    rows = [{"x": 1200.0 + STEP * i, "y": 900.0, "state": TrackMode.TRACKING,
             "fix": _fix(i, 1200.0 + STEP * i, 900.0)} for i in range(5)]
    records, info = _load(_write(tmp_path, rows))
    plan = hv.plan_hud_video(records, info, fps=10)
    assert all(hv.caption_at(plan, k) is None for k in range(len(plan.frames)))


# --- 2. what a frame shows -------------------------------------------------------------------------------------

def test_the_state_is_a_lamp_green_when_tracking_red_when_lost_amber_when_relocalizing(jam):
    r = jam["renderer"]
    boxes = []
    for k, state in ((30, "TRACKING"), (80, "LOST"), (100, "RELOCALIZING")):
        img, fi = r.draw(k)
        assert fi.state == state
        x0, y0, x1, y1 = fi.lamp_box
        boxes.append(fi.lamp_box)
        assert _inside(fi.lamp_box, fi.side_box)
        assert _close(img[(y0 + y1) // 2, (x0 + x1) // 2], hv.STATE_COLOURS[state], tol=30)      # the lamp itself
        # a lamp and a word on dark glass, not a filled plate: most of the row around the lamp stays dark
        row = img[y0:y1, x0:fi.side_box[2]]
        assert (row.max(axis=2) < 110).mean() > 0.6
    assert boxes[0] == boxes[1] == boxes[2]                          # the lamp does not move about


def test_while_lost_the_whole_stage_says_so(jam):
    img_t, fi_t = jam["renderer"].draw(30)
    img_l, fi_l = jam["renderer"].draw(80)
    x0, y0, x1, y1 = fi_l.stage_box
    n = (y1 - y0) // 8

    def redness(img):                                                # the top-left corner of the stage: bare map
        b, g, r = img[y0:y0 + n, x0:x0 + n].reshape(-1, 3).mean(0)
        return r - (g + b) / 2

    assert redness(img_t) < 5 and redness(img_l) > 40                # a red alarm around the map, seen at a glance
    assert "dead reckoning" in fi_l.note and "dead reckoning" not in fi_t.note
    # ... while the middle of the stage, where the drone and the ellipse are, stays clear
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2 - (y1 - y0) // 5
    assert _close(img_l[cy, cx], img_t[cy, cx], tol=25)


def test_no_text_is_too_small_for_a_demo_screen(tmp_path):
    assert hv.MIN_TEXT >= 18                                         # px on a 1920 x 1080 video
    d = _write(tmp_path, _jam_rows(matches=True))
    records, info = _load(d)
    plan = hv.plan_hud_video(records, info, fps=10, size=(1920, 1080))
    r = hv.HudRenderer(d, records, info, plan)
    for k in (30, 75, 100, 125):                                     # tracking, lost + caption, rejected fix, recovered
        assert r.draw(k)[1].min_text_px >= hv.MIN_TEXT


def test_the_frame_reports_the_numbers_it_shows(jam):
    _, fi = jam["renderer"].draw(30)
    fix = jam["records"][3].track.fix
    assert (fi.idx, fi.state, fi.inliers) == (3, "TRACKING", 2000)
    assert fi.confidence == pytest.approx(fix_confidence(fix)) and fi.heading_deg == pytest.approx(90.0)
    assert fi.xy == pytest.approx((1590.0, 1200.0))
    assert fi.footprint and fi.rejected is None
    _, fi = jam["renderer"].draw(80)
    assert (fi.idx, fi.state, fi.inliers, fi.confidence) == (8, "LOST", None, None)
    assert not fi.footprint and fi.rejected is None
    assert fi.sigma_px == pytest.approx(40.0) and fi.word == "drifting"      # record 8: sigma 40 x 20 map px


def test_the_trail_is_green_where_sure_yellow_where_unsure_red_dashed_without_a_fix(jam):
    plan, r = jam["plan"], jam["renderer"]
    img, _ = r.draw(60)                                              # the drone at record 6
    y, x = 1200.0, 1500.0
    sure = _row(img, plan, 60, x + STEP * 1.3, x + STEP * 1.7, y)   # between records 1 and 2
    unsure = _row(img, plan, 60, x + STEP * 4.3, x + STEP * 4.7, y)
    assert len(sure) > 5 and all(_close(p, hv.GREEN) for p in sure)
    assert all(_close(p, hv.YELLOW) for p in unsure)
    ahead = _px(plan, 60, x + STEP * 8.5, y - 4 * STEP)             # nothing is drawn where the drone has not been
    assert _close(img[round(ahead[1]), round(ahead[0])], (BG,) * 3, tol=30)

    img, _ = r.draw(95)                                              # inside the jam, halfway from record 9 to 10
    jammed = _row(img, plan, 95, x + STEP * 7.0, x + STEP * 9.0, y)
    red = np.array([_close(p, hv.RED) for p in jammed])
    assert red.any() and not red.all()                               # dashes and gaps
    assert not any(_close(p, hv.GREEN) for p in jammed)
    assert np.count_nonzero(np.diff(red.astype(int)) == 1) >= 2      # several dashes, not one piece


def test_the_three_sigma_ellipse_is_drawn_at_its_size(jam):
    plan = jam["plan"]
    img, fi = jam["renderer"].draw(90)                               # record 9: sigma 60 x 30 map px
    s = plan.size[0] / plan.view_w
    a, b, angle = fi.ellipse_px
    assert (a, b) == pytest.approx((180 * s, 90 * s), rel=1e-6) and angle % 180 == pytest.approx(0.0, abs=1e-6)
    cx, cy = fi.drone_px
    top = img[round(cy - b) - 3:round(cy - b) + 4, round(cx)]
    assert max(np.abs(p.astype(int) - BG).max() for p in top) > 60          # the outline, 3 sigma above the drone
    outside = img[round(cy - 1.35 * b), round(cx)]
    assert _close(outside, (BG,) * 3, tol=30)
    # while tracking the ellipse is a few px: reported, not hidden
    _, fi = jam["renderer"].draw(30)
    assert fi.ellipse_px[0] == pytest.approx(3 * s)


def test_the_ellipse_grows_smoothly_between_records(jam):
    sizes = [jam["renderer"].draw(k)[1].ellipse_px[0] for k in range(70, 91)]
    assert all(b >= a for a, b in itertools.pairwise(sizes)) and sizes[10] > sizes[0] and sizes[-1] > sizes[10]


def test_a_rejected_fix_is_drawn_apart_with_its_reason(jam):
    plan = jam["plan"]
    img, fi = jam["renderer"].draw(100)                              # record 10: a rejected fix 110 px above the route
    assert fi.state == "RELOCALIZING" and fi.rejected == "too_few_inliers"
    assert not fi.footprint and fi.inliers == 5 and fi.confidence is None
    x, y = 1500.0 + STEP * 10, 1200.0 - 110
    top = _row(img, plan, 100, x - 90, x + 90, y - 64)               # its top edge, clear of everything else
    red = [_close(p, hv.BAD, tol=60) for p in top]
    assert any(red) and not all(red)                                  # red and dashed: not the solid accepted outline
    # an accepted footprint (record 3) is a solid light outline, not red
    img, fi = jam["renderer"].draw(30)
    f = plan.frames[30]
    assert f.footprint is not None
    (ax, ay), (bx, by) = _px(plan, 30, *f.footprint[0]), _px(plan, 30, *f.footprint[1])
    edge = [img[round(ay + (by - ay) * u), round(ax + (bx - ax) * u)] for u in np.linspace(0.3, 0.7, 9)]
    assert all(np.abs(p.astype(int) - BG).max() > 40 for p in edge)
    assert not any(_close(p, hv.BAD, tol=60) for p in edge)


def test_the_camera_picture_is_the_current_frame_projected_onto_its_footprint(jam):
    img, fi = jam["renderer"].draw(30)
    x0, y0, x1, y1 = fi.inset_box
    assert (x1 - x0) >= 0.2 * jam["plan"].size[0] and _inside(fi.inset_box, fi.side_box)
    assert (x1 - x0) / (y1 - y0) == pytest.approx(FRAME[0] / FRAME[1], rel=0.02)     # the frame's own shape
    src = cv2.imread(str(jam["dir"] / jam["records"][3].source))
    expected = cv2.resize(src, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(int)
    assert np.abs(img[y0:y1, x0:x1].astype(int) - expected).mean() < 4
    assert not fi.no_signal and not fi.frame_missing
    # the picture is tied to the place it shows: two lines from the footprint to the picture's edge that faces the
    # map (to its far corners they would run across the panel's text)
    assert fi.projection and fi.match_lines == 0                     # this run folder carries no match points
    (p, q), _ = hv.projection_links(np.array([_px(jam["plan"], 30, *c) for c in jam["plan"].frames[30].footprint]),
                                    (x0, y0, x0, y1))
    assert q[0] == x0 and q[1] in (y0, y1)
    mid = (0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]))
    bare, _ = jam["renderer"].draw(80)                               # LOST: the same place with no projection
    assert not jam["renderer"].draw(80)[1].projection                # nothing on the map to project onto
    assert not jam["renderer"].draw(100)[1].projection               # a rejected fix is not projected either
    y, x = round(mid[1]), round(mid[0])
    assert np.abs(img[y - 2:y + 3, x - 2:x + 3].astype(int).max(axis=(0, 1)) - bare[y, x]).max() > 25


def test_match_lines_are_drawn_only_from_the_runs_own_match_points(tmp_path):
    rows = _jam_rows(matches=True)
    rows[2]["extra"] = {"matches": {"frame": [[1.0, 2.0]], "map": "nonsense"}}       # malformed: not drawn
    rows[8]["extra"] = {"matches": rows[3]["extra"]["matches"]}                      # LOST: no fix, so no lines
    d = _write(tmp_path, rows)
    records, info = _load(d)
    plan = hv.plan_hud_video(records, info, fps=10)
    r = hv.HudRenderer(d, records, info, plan)
    img, fi = r.draw(30)                                             # record 3 has extra.matches: three pairs
    assert fi.match_lines == 3 and fi.projection
    for (fx, fy), (mx, my) in zip(*(rows[3]["extra"]["matches"][k] for k in ("frame", "map")), strict=True):
        x0, y0, x1, y1 = fi.inset_box                                # a dot at each end of each line
        a = (x0 + fx * (x1 - x0) / FRAME[0], y0 + fy * (y1 - y0) / FRAME[1])
        b = _px(plan, 30, mx, my)
        assert _close(img[round(a[1]), round(a[0])], hv.MATCH, tol=40)
        assert _close(img[round(b[1]), round(b[0])], hv.MATCH, tol=40)
    assert r.draw(10)[1].match_lines == 0 and r.draw(10)[1].projection       # record 1 has none: projection only
    assert r.draw(20)[1].match_lines == 0                            # malformed points are not guessed at
    assert r.draw(80)[1].match_lines == 0


def test_lost_with_a_picture_is_marked_and_lost_without_one_says_no_signal(jam, tmp_path):
    img, fi = jam["renderer"].draw(80)                               # LOST, the (useless) picture exists
    x0, y0, x1, y1 = fi.inset_box
    src = cv2.imread(str(jam["dir"] / jam["records"][8].source))
    shown = cv2.resize(src, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(int)
    assert not fi.no_signal
    assert np.abs(img[y0:y1, x0:x1].astype(int) - shown).mean() > 15       # dimmed and stamped: not a normal view
    assert not fi.jammed and "no match" in fi.note

    d = _write(tmp_path, _jam_rows(), write_frames=False)
    records, info = _load(d)
    plan = hv.plan_hud_video(records, info, fps=10)
    r = hv.HudRenderer(d, records, info, plan)
    img, fi = r.draw(80)
    x0, y0, x1, y1 = fi.inset_box
    assert fi.no_signal and fi.frame_missing
    panel = img[y0:y1, x0:x1]
    assert np.median(panel) < 50 and panel.max() > 150               # dark, with a bright message
    _, fi = r.draw(30)                                               # TRACKING without a picture: only a placeholder
    assert fi.frame_missing and not fi.no_signal


def test_the_caption_is_drawn_while_it_lasts(jam):
    img0, fi0 = jam["renderer"].draw(60)
    img1, fi1 = jam["renderer"].draw(75)
    assert fi0.caption is None and fi0.caption_box is None
    assert "frame 7" in fi1.caption and "Signal lost" in fi1.caption
    x0, y0, x1, y1 = fi1.caption_box
    assert y0 > jam["plan"].size[1] / 2                              # in the lower half, off the drone
    assert (y1 - y0) >= 0.06 * jam["plan"].size[1]                   # large enough for a projector
    assert np.abs(img1[y0:y1, x0:x1].astype(int) - img0[y0:y1, x0:x1].astype(int)).max() > 100


def test_the_panels_do_not_overlap(jam):
    _, fi = jam["renderer"].draw(75)
    lay = hv.layout(jam["plan"].size)
    assert (fi.stage_box, fi.side_box, fi.dock_box) == (lay.stage, lay.side, lay.dock)
    assert _apart(fi.stage_box, fi.side_box) and _apart(fi.stage_box, fi.dock_box) and _apart(fi.side_box, fi.dock_box)
    assert _inside(fi.inset_box, fi.side_box) and _inside(fi.lamp_box, fi.side_box)
    assert _inside(fi.caption_box, fi.stage_box) and _apart(fi.inset_box, fi.lamp_box)
    x0, y0, x1, y1 = fi.stage_box
    assert x0 <= fi.drone_px[0] <= x1 and y0 <= fi.drone_px[1] <= y1
    # the caption sits low in the stage: the drone and its footprint stay clear of it
    assert fi.caption_box[1] > fi.drone_px[1] + 0.1 * (y1 - y0)


def test_drawing_is_deterministic_and_order_independent(jam):
    a = jam["renderer"].draw(95)[0]
    jam["renderer"].draw(20)
    b = jam["renderer"].draw(95)[0]
    records, info = _load(jam["dir"])
    c = hv.HudRenderer(jam["dir"], records, info, hv.plan_hud_video(records, info, fps=10)).draw(95)[0]
    assert np.array_equal(a, b) and np.array_equal(a, c)


# --- 3. the video ----------------------------------------------------------------------------------------------

def _cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "render_video.py"), *map(str, args)],
                          check=False, capture_output=True, text=True)


def _mp4(path: Path) -> dict:
    b = path.read_bytes()
    cap = cv2.VideoCapture(str(path))
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    return {"avc1": b"avc1" in b, "moov_first": b.find(b"moov") < b.find(b"mdat"), "frames": decoded,
            "size": (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": cap.get(cv2.CAP_PROP_FPS)}


def test_the_cli_writes_the_hud_video_by_default(tmp_path):
    d = _write(tmp_path, _jam_rows())                                # t = 0..16 s
    res = _cli("--run", d, "--speed", "4", "--size", "640x360")
    assert res.returncode == 0, res.stderr
    assert "hud" in res.stdout
    info = _mp4(d / "trajectory.mp4")
    assert info["avc1"] and info["moov_first"]
    assert info["size"] == (640, 360) and info["fps"] == pytest.approx(30)
    assert abs(info["frames"] - (math.floor(16 / 4 * 30) + 1 + 30)) <= 1     # the run at 4x, plus a 1 s hold

    cap = cv2.VideoCapture(str(d / "trajectory.mp4"))                # the file holds the HUD frames, lamp and all
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)                             # t = 8 s: LOST
    ok, frame = cap.read()
    assert ok
    records, run_info = _load(d)
    plan = hv.plan_hud_video(records, run_info, fps=30, speed=4.0, size=(640, 360))
    x0, y0, x1, y1 = hv.HudRenderer(d, records, run_info, plan).draw(60)[1].lamp_box
    b, g, r = frame[y0:y1, x0:x1].reshape(-1, 3).mean(0)
    assert r > g + 40 and r > b + 40


def test_the_m1_video_is_still_there_as_a_style(tmp_path):
    d = _write(tmp_path, _jam_rows())
    res = _cli("--run", d, "--speed", "4", "--size", "640x360", "--style", "classic")
    assert res.returncode == 0, res.stderr
    assert "classic" in res.stdout
    assert _mp4(d / "trajectory.mp4")["size"] == (640, 360)
    res = _cli("--run", d, "--style", "fancy")
    assert res.returncode == 2 and "--style" in res.stderr and "Traceback" not in res.stderr


@pytest.mark.parametrize("map_size", [(1100, 1100), (2600, 700)], ids=["square", "wide"])
def test_a_fake_run_with_a_jam_renders_on_square_and_wide_maps(tmp_path, map_size):
    rng = np.random.default_rng(0)
    m = tmp_path / "map.png"
    w, h = map_size
    cv2.imwrite(str(m), cv2.GaussianBlur(rng.uniform(0, 255, (h, w, 3)).astype(np.uint8), (0, 0), 2))
    make_fake_run(m, tmp_path / "run", FakeRunConfig(frames=20))
    path = hv.write_hud_video(tmp_path / "run", speed=10, size=(640, 360))
    assert path.name == "trajectory.mp4"
    info = _mp4(path)
    assert info["avc1"] and info["size"] == (640, 360)
    assert abs(info["frames"] - (19 * 3 + 1 + 30)) <= 1
    records, run_info = _load(tmp_path / "run")
    plan = hv.plan_hud_video(records, run_info, speed=10, size=(640, 360))
    s = 640 / plan.view_w
    x0, y0, x1, y1 = plan.stage
    for k, f in enumerate(plan.frames):                              # the drone is on the stage all the way
        dx, dy = hv.to_video(plan, k, f.x, f.y)
        assert x0 <= dx <= x1 and y0 <= dy <= y1
        if (x1 - x0) / s <= w:                                       # ... and the stage shows nothing but map
            assert (x1 - x0) / 2 / s - 1e-6 <= f.cx <= w - (x1 - x0) / 2 / s + 1e-6
        if (y1 - y0) / s <= h:
            assert (y1 - y0) / 2 / s - 1e-6 <= f.cy <= h - (y1 - y0) / 2 / s + 1e-6
    states = {hv.HudRenderer(tmp_path / "run", records, run_info, plan).draw(k)[1].state
              for k in range(0, len(plan.frames), 3)}
    assert states == {"TRACKING", "LOST", "RELOCALIZING"}


def test_along_a_map_edge_the_stage_never_shows_anything_but_map(tmp_path):
    # The flight hugs the bottom edge and ends in the bottom-right corner of the map. The camera stops where the
    # stage would leave the map (the drone then moves off the stage's centre, towards the edge it is near): the
    # stage shows map and only map, in every frame. The earlier rule kept the drone centred and showed what is
    # past the map instead: a black strip a quarter of the screen wide.
    d = _write(tmp_path, _jam_rows(x0=MAP[0] - 16 * STEP - 60, y0=MAP[1] - 80))
    records, info = _load(d)
    plan = hv.plan_hud_video(records, info, fps=10, speed=1.0)
    s = plan.size[0] / plan.view_w
    x0, y0, x1, y1 = plan.stage
    for k, f in enumerate(plan.frames):
        left, top = f.cx - (x1 - x0) / 2 / s, f.cy - (y1 - y0) / 2 / s
        right, bottom = f.cx + (x1 - x0) / 2 / s, f.cy + (y1 - y0) / 2 / s
        assert left >= -1e-6 and top >= -1e-6 and right <= MAP[0] + 1e-6 and bottom <= MAP[1] + 1e-6, k
        dx, dy = hv.to_video(plan, k, f.x, f.y)
        assert x0 <= dx <= x1 and y0 <= dy <= y1, k                  # and the drone is still on it
    assert plan.frames[-1].cy == pytest.approx(MAP[1] - (y1 - y0) / 2 / s)   # pressed against the bottom edge
    img, fi = hv.HudRenderer(d, records, info, plan).draw(30)        # TRACKING: no alarm tint on the map
    for cx, cy in ((x0 + 3, y1 - 12), (x0 + 3, y0 + 3), (x1 - 12, y0 + 3)):   # stage corners clear of the drawing
        assert _close(img[cy:cy + 8, cx:cx + 8].reshape(-1, 3).mean(0), (BG,) * 3, tol=12)
    # under the side panel there is glass over more map (mirrored past the map's edge), never a flat black strip
    sx0, _, sx1, sy1 = fi.side_box
    assert img[sy1 - 40:sy1 - 4, sx0 + 4:sx1 - 4].mean() > 20


def test_records_without_a_position_are_left_out(tmp_path):
    rows = [{"x": 1200.0 + STEP * i, "y": 900.0, "state": TrackMode.TRACKING,
             "fix": _fix(i, 1200.0 + STEP * i, 900.0)} for i in range(6)]
    d = _write(tmp_path, rows)
    lines = (d / "run.jsonl").read_text().splitlines()
    first = json.loads(lines[0])
    first.update(state="LOST", x=None, y=None, vx=None, vy=None, cov=None, fix=None)
    (d / "run.jsonl").write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    records, info = _load(d)
    assert not records[0].track.has_position
    plan = hv.plan_hud_video(records, info, fps=10)
    img, fi = hv.HudRenderer(d, records, info, plan).draw(0)         # the video starts at the first position
    assert fi.idx == 1 and img.shape == (720, 1280, 3)
    assert hv.write_hud_video(d, speed=5, size=(640, 360)).is_file()


def test_a_long_run_plans_fast_and_draws(tmp_path):
    rows = []
    for i in range(442):                                             # the length of the Irpin flights
        x, y = 600.0 + 6.0 * i, 1500.0 + 500.0 * math.sin(i / 40)
        lost = 133 <= i < 147
        rows.append({"x": x, "y": y, "state": TrackMode.LOST if lost else TrackMode.TRACKING,
                     "fix": None if lost else _fix(i, x, y, heading=90.0)})
    d = _write(tmp_path, rows, write_frames=False)
    records, info = _load(d)
    t0 = time.perf_counter()
    plan = hv.plan_hud_video(records, info, fps=30, speed=8.0, size=(1920, 1080))
    assert time.perf_counter() - t0 < 5
    assert len(plan.frames) == math.floor(441 / 8 * 30) + 1 + 30
    r = hv.HudRenderer(d, records, info, plan)
    t0 = time.perf_counter()
    for k in (0, len(plan.frames) // 2, len(plan.frames) - 1):       # the last frame draws the whole 442-point trail
        img, _ = r.draw(k)
        assert img.shape == (1080, 1920, 3) and img.dtype == np.uint8
    assert (time.perf_counter() - t0) / 3 < 0.5                      # far from it in practice; a minute of video
                                                                     # must not take an hour


def test_a_missing_run_is_a_clear_error(tmp_path):
    res = _cli("--run", tmp_path / "nothing")
    assert res.returncode != 0 and "run.json" in res.stderr and "Traceback" not in res.stderr
    records, info = _load(_write(tmp_path, _jam_rows()))
    with pytest.raises(ValueError, match="at least"):
        hv.plan_hud_video(records, info, size=(64, 48))
