"""M7-02 · HUD video v2.

"Done when: a video of a run with a jam reads clearly without narration." Only a person can judge "reads clearly";
the checkable contract behind it:
1. the pure helpers: confidence -> trail colour, covariance -> 3-sigma ellipse, the eased camera (no jump across a
   relocalisation), caption timing;
2. each frame shows the right thing: state badge colours, trail colours, the ellipse, a rejected fix and its reason,
   the camera inset (picture, NO SIGNAL, link to the map), the caption;
3. the CLI writes the HUD video by default (H.264, the right size and frame count) and still writes the M1 one;
   square and wide maps, a 442-frame run.

Hand-made runs on a plain grey map make every drawn pixel and position known.
"""
import itertools
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
    f = plan.frames[k]
    s = plan.size[0] / plan.view_w
    return plan.size[0] / 2 + (x - f.cx) * s, plan.size[1] / 2 + (y - f.cy) * s


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
    assert hv.STATE_COLOURS["TRACKING"] == hv.GREEN and hv.STATE_COLOURS["LOST"] == hv.RED
    assert hv.STATE_COLOURS["RELOCALIZING"] == hv.AMBER
    b, g, r = hv.GREEN
    assert g > r + 60 and g > b + 60
    b, g, r = hv.RED
    assert r > g + 100 and r > b + 100
    b, g, r = hv.AMBER
    assert r > b + 100 and g > b + 60 and r >= g


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
    assert np.hypot(*(cam - drone).T)[k - 45:k + 45].max() < plan.size[1] / 2   # the drone never leaves the view


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
    assert "frame 7" in text.lower() and "lost" in text.lower() and "camera jammed" in text
    assert alpha > 0
    text2, alpha2 = hv.caption_at(plan, 70 + 20)                     # two seconds later: still fully there
    assert text2 == text and alpha2 == pytest.approx(1.0)
    # the next event (record 10) replaces it at once
    text3, _ = hv.caption_at(plan, 100)
    assert "frame 10" in text3.lower() and "relocalizing" in text3.lower() and "too few inliers" in text3.lower()
    # record 11 is still RELOCALIZING: no new caption; record 12 has one, and it goes away after CAPTION_S
    assert hv.caption_at(plan, 110)[0] == text3
    last, _ = hv.caption_at(plan, 120)
    assert "frame 12" in last.lower() and "tracking" in last.lower() and "relocalized" in last
    gone = 120 + math.ceil(hv.CAPTION_S * 10) + 1
    assert gone < len(plan.frames) and hv.caption_at(plan, gone) is None
    fading = hv.caption_at(plan, gone - 2)
    assert fading is not None and 0 < fading[1] < 1                  # it fades out, it does not blink off


def test_a_caption_is_not_replaced_before_it_can_be_read(tmp_path):
    records, info = _load(_write(tmp_path, _jam_rows()))
    plan = hv.plan_hud_video(records, info, fps=30, speed=8.0)       # records 10, 11, 12: 1/8 s of video apart
    starts = [c.frame for c in plan.captions]
    assert len(starts) == 3                                          # LOST (7), RELOCALIZING (10), TRACKING (12)
    least = math.ceil(hv.CAPTION_MIN_S * 30)
    assert all(b - a >= least for a, b in itertools.pairwise(starts))
    assert hv.CAPTION_MIN_S >= 1.0                                   # a line of text needs about a second
    k = starts[1]                                                    # the RELOCALIZING caption, while it is up
    assert "RELOCALIZING" in hv.caption_at(plan, k)[0] and "RELOCALIZING" in hv.caption_at(plan, k + least - 1)[0]


def test_a_run_without_events_has_no_captions(tmp_path):
    rows = [{"x": 1200.0 + STEP * i, "y": 900.0, "state": TrackMode.TRACKING,
             "fix": _fix(i, 1200.0 + STEP * i, 900.0)} for i in range(5)]
    records, info = _load(_write(tmp_path, rows))
    plan = hv.plan_hud_video(records, info, fps=10)
    assert all(hv.caption_at(plan, k) is None for k in range(len(plan.frames)))


# --- 2. what a frame shows -------------------------------------------------------------------------------------

def test_the_badge_is_green_when_tracking_red_when_lost_amber_when_relocalizing(jam):
    r = jam["renderer"]
    means = {}
    for k, state in ((30, "TRACKING"), (80, "LOST"), (100, "RELOCALIZING")):
        img, fi = r.draw(k)
        assert fi.state == state
        x0, y0, x1, y1 = fi.badge_box
        assert (x1 - x0) >= 0.12 * jam["plan"].size[0] and (y1 - y0) >= 0.05 * jam["plan"].size[1]   # big enough
        means[state] = img[y0:y1, x0:x1].reshape(-1, 3).mean(0)
    b, g, red = means["TRACKING"]
    assert g > red + 40 and g > b + 40
    b, g, red = means["LOST"]
    assert red > g + 60 and red > b + 60
    b, g, red = means["RELOCALIZING"]
    assert red > b + 60 and g > b + 40
    assert np.abs(means["TRACKING"] - means["LOST"]).max() > 80


def test_a_lost_frame_differs_from_a_tracking_frame_in_the_badge_region(jam):
    img_t, fi_t = jam["renderer"].draw(30)
    img_l, fi_l = jam["renderer"].draw(80)
    x0, y0, x1, y1 = fi_t.badge_box
    assert fi_l.badge_box[:2] == fi_t.badge_box[:2]                  # the badge does not move about
    diff = np.abs(img_t[y0:y1, x0:x1].astype(int) - img_l[y0:y1, x0:x1].astype(int)).max(axis=2)
    assert (diff > 60).mean() > 0.5


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
    x0, y0, x1, y1 = fi.hud_box
    assert x1 > x0 and y1 > y0


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
    red = [_close(p, hv.RED, tol=60) for p in top]
    assert any(red) and not all(red)                                  # red and dashed: not the solid accepted outline
    # an accepted footprint (record 3) is a solid light outline, not red
    img, fi = jam["renderer"].draw(30)
    f = plan.frames[30]
    assert f.footprint is not None
    (ax, ay), (bx, by) = _px(plan, 30, *f.footprint[0]), _px(plan, 30, *f.footprint[1])
    edge = [img[round(ay + (by - ay) * u), round(ax + (bx - ax) * u)] for u in np.linspace(0.3, 0.7, 9)]
    assert all(np.abs(p.astype(int) - BG).max() > 40 for p in edge)
    assert not any(_close(p, hv.RED, tol=60) for p in edge)


def test_the_inset_is_the_current_frame_linked_to_its_footprint(jam):
    img, fi = jam["renderer"].draw(30)
    x0, y0, x1, y1 = fi.inset_box
    assert (x1 - x0) >= 0.2 * jam["plan"].size[0]
    src = cv2.imread(str(jam["dir"] / jam["records"][3].source))
    expected = cv2.resize(src, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(int)
    assert np.abs(img[y0:y1, x0:x1].astype(int) - expected).mean() < 4
    assert not fi.no_signal and not fi.frame_missing
    # the run folder carries no match points, so the link is the honest one: inset corners to footprint corners
    assert fi.link == "footprint"
    assert jam["renderer"].draw(80)[1].link is None                  # LOST: nothing on the map to link to
    assert jam["renderer"].draw(100)[1].link is None                 # a rejected fix is not linked either


def test_match_lines_are_drawn_only_when_the_run_carries_match_points(tmp_path):
    d = _write(tmp_path, _jam_rows(matches=True))
    records, info = _load(d)
    plan = hv.plan_hud_video(records, info, fps=10)
    r = hv.HudRenderer(d, records, info, plan)
    assert r.draw(30)[1].link == "matches"                           # record 3 has extra.matches
    assert r.draw(20)[1].link == "footprint"                         # record 2 has none


def test_lost_with_a_picture_is_marked_and_lost_without_one_says_no_signal(jam, tmp_path):
    img, fi = jam["renderer"].draw(80)                               # LOST, the (useless) picture exists
    x0, y0, x1, y1 = fi.inset_box
    src = cv2.imread(str(jam["dir"] / jam["records"][8].source))
    shown = cv2.resize(src, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(int)
    assert not fi.no_signal
    assert np.abs(img[y0:y1, x0:x1].astype(int) - shown).mean() > 15       # dimmed and stamped: not a normal view

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
    assert "frame 7" in fi1.caption.lower()
    x0, y0, x1, y1 = fi1.caption_box
    assert y0 > jam["plan"].size[1] / 2                              # in the lower half, off the drone
    assert (y1 - y0) >= 0.04 * jam["plan"].size[1]                   # large enough for a projector
    assert np.abs(img1[y0:y1, x0:x1].astype(int) - img0[y0:y1, x0:x1].astype(int)).max() > 100


def test_the_panels_do_not_overlap(jam):
    _, fi = jam["renderer"].draw(75)
    boxes = [fi.hud_box, fi.inset_box, fi.caption_box]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1], (a, b)
    bx0, by0, bx1, by1 = fi.badge_box
    hx0, hy0, hx1, hy1 = fi.hud_box
    assert hx0 <= bx0 and bx1 <= hx1 and hy0 <= by0 and by1 <= hy1   # the badge heads the HUD panel


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

    cap = cv2.VideoCapture(str(d / "trajectory.mp4"))                # the file holds the HUD frames, badge and all
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)                             # t = 8 s: LOST
    ok, frame = cap.read()
    assert ok
    records, run_info = _load(d)
    plan = hv.plan_hud_video(records, run_info, fps=30, speed=4.0, size=(640, 360))
    x0, y0, x1, y1 = hv.HudRenderer(d, records, run_info, plan).draw(60)[1].badge_box
    b, g, r = frame[y0:y1, x0:x1].reshape(-1, 3).mean(0)
    assert r > g + 50 and r > b + 50


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
    half_h = plan.view_w * 360 / 640 / 2
    assert plan.view_w <= w + 1e-6 and 2 * half_h <= h + 1e-6        # the view never shows more than the map
    for f in plan.frames:                                            # the drone is on screen all the way
        assert abs(f.x - f.cx) <= 0.45 * plan.view_w and abs(f.y - f.cy) <= 0.9 * half_h
    states = {hv.HudRenderer(tmp_path / "run", records, run_info, plan).draw(k)[1].state
              for k in range(0, len(plan.frames), 3)}
    assert states == {"TRACKING", "LOST", "RELOCALIZING"}


def test_along_a_map_edge_the_drone_stays_between_the_panels(tmp_path):
    # Near a map edge the camera does not stop at the edge (M1 does): that would slide the drone under the side
    # panels. It stays in the free stage between them: the HUD ends at 26 % of the width, the camera inset
    # starts at 69 %, so 0.19 of the view to either side of the centre is clear.
    records, info = _load(_write(tmp_path, _jam_rows(x0=MAP[0] - 16 * STEP - 60, y0=MAP[1] - 80)))
    plan = hv.plan_hud_video(records, info, fps=10, speed=1.0)
    view_h = plan.view_w * plan.size[1] / plan.size[0]
    for f in plan.frames:
        assert abs(f.x - f.cx) <= 0.19 * plan.view_w and abs(f.y - f.cy) <= 0.3 * view_h


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
