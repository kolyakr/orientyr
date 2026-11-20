"""M1-08 · Trajectory video renderer.

"Done when: a 50-frame run renders a smooth video (interpolated between fixes) in under a minute." The testable
form from the issue's spec update:
1. the CLI writes trajectory.mp4: H.264 4:2:0 8-bit, index first, the right size and frame count, readable back;
   clear errors for a missing run or map and bad options;
2. a 50-frame run (the fake run on the task map) renders in under a minute;
3. smooth: the path passes through every record at its time, no jumps between video frames, gradual heading;
4. each frame shows the right content: drone, trail, footprint, inset, HUD values;
5. (human) a still and a clip in the PR, plays in QuickTime and Chrome.

Hand-made runs on a plain grey map make every drawn pixel and position known.
"""
import math
import struct
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.fake_run import make_fake_run
from dronenav.telemetry import Record, RunInfo, RunWriter, read_run, read_run_info, write_run_info
from dronenav.types import Fix, TrackMode, TrackState
from dronenav.viz import draw
from dronenav.viz import trajectory_image as ti
from dronenav.viz import trajectory_video as tv

ROOT = Path(__file__).resolve().parents[1]
GREY = 128


# --- hand-made runs --------------------------------------------------------------------------------------------

def _H(x, y, heading_deg, scale, frame):
    """Frame px -> map px, conventions of dronenav.types (heading clockwise from up, y down)."""
    w, h = frame
    a = math.radians(heading_deg)
    sR = scale * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    t = np.array([x, y]) - sR @ np.array([w / 2, h / 2])
    return np.array([[*sR[0], t[0]], [*sR[1], t[1]], [0, 0, 1.0]])


def _corners(H, frame):
    w, h = frame
    p = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float) @ H.T
    return p[:, :2] / p[:, 2:]


def _frame_image(i, frame):
    """A distinct picture per frame: a colour from the index plus a diagonal ramp, so insets can be compared."""
    w, h = frame
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = ((37 * i) % 200 + 30, (91 * i) % 200 + 30, (53 * i) % 200 + 30)
    ramp = (np.add.outer(np.arange(h), np.arange(w)) % 64).astype(np.uint8)
    img[..., 0] = np.clip(img[..., 0].astype(int) + ramp, 0, 255)
    return img


def _make_run(tmp_path, map_size, pts, missed=(), frame=(96, 64), scale=1.0, fps=1.0, write_frames=True,
              name="run"):
    """A run folder over a plain grey map, records at t = i / fps; points in ``missed`` are LOST (no fix)."""
    w, h = map_size
    d = tmp_path / name
    (d / "frames").mkdir(parents=True, exist_ok=True)
    m = d / "map.png"
    cv2.imwrite(str(m), np.full((h, w, 3), GREY, np.uint8))
    write_run_info(d, RunInfo(map=str(m), map_size=(w, h), generator="test", frame_size=frame, fps=fps))
    with RunWriter(d / "run.jsonl") as wr:
        for i, (x, y) in enumerate(pts):
            nxt = pts[min(i + 1, len(pts) - 1)]
            prv = pts[max(i - 1, 0)]
            head = math.degrees(math.atan2(nxt[0] - prv[0], -(nxt[1] - prv[1]))) % 360
            fix, state = None, TrackMode.TRACKING
            if i in missed:
                state = TrackMode.LOST
            else:
                H = _H(x, y, head, scale, frame)
                fix = Fix(idx=i, H=H, x=x, y=y, heading_deg=head, scale=scale, footprint=_corners(H, frame),
                          inliers=100 + i, inlier_ratio=0.6, reproj_px=0.8, accepted=True)
            src = f"frames/frame_{i:05d}.png"
            if write_frames:
                cv2.imwrite(str(d / src), _frame_image(i, frame))
            track = TrackState(idx=i, state=state, x=x, y=y, vx=0.0, vy=0.0, cov=np.eye(2), fix=fix)
            wr.write(Record(idx=i, t=i / fps, source=src, track=track))
    return d


def _line(map_size, n=10, y_frac=0.5):
    w, h = map_size
    return [(w * (0.2 + 0.6 * i / (n - 1)), h * y_frac) for i in range(n)]


def _sine(map_size, n=12):
    w, h = map_size
    return [(w * (0.2 + 0.6 * i / (n - 1)), h * (0.5 + 0.2 * math.sin(i / 2))) for i in range(n)]


def _close(px, colour, tol=45):
    return np.abs(px.astype(int) - np.array(colour)).max() <= tol


def _changed(px, tol=30):
    return np.abs(px.astype(int) - GREY).max() > tol


def _load(d):
    return read_run(d / "run.jsonl"), read_run_info(d)


# --- 1. the file -----------------------------------------------------------------------------------------------

def _mp4(path: Path) -> dict:
    """What a player needs from the file: codec box, chroma format, bit depth, index position."""
    b = path.read_bytes()
    i = b.find(b"avcC") + 4
    size = struct.unpack(">I", b[i - 8:i - 4])[0]
    box = b[i:i - 8 + size]
    prof = box[1]
    j = 6
    for _ in range(box[5] & 0x1F):
        j += 2 + struct.unpack(">H", box[j:j + 2])[0]
    n_pps = box[j]
    j += 1
    for _ in range(n_pps):
        j += 2 + struct.unpack(">H", box[j:j + 2])[0]
    high = prof in (100, 110, 122, 244) and j < len(box)
    cap = cv2.VideoCapture(str(path))
    return {"avc1": b"avc1" in b, "chroma": (box[j] & 3) if high else 1, "depth": 8 + (box[j + 1] & 7) if high else 8,
            "moov_first": b.find(b"moov") < b.find(b"mdat"), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "size": (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": cap.get(cv2.CAP_PROP_FPS)}


def _cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "render_video.py"), *map(str, args)],
                          check=False, capture_output=True, text=True)


def test_cli_writes_a_browser_playable_h264(tmp_path):
    d = _make_run(tmp_path, (1600, 1000), _line((1600, 1000), n=10))   # t = 0..9 s
    res = _cli("--run", d, "--speed", "3")
    assert res.returncode == 0, res.stderr
    info = _mp4(d / "trajectory.mp4")
    assert info["avc1"] and info["chroma"] == 1 and info["depth"] == 8      # H.264 4:2:0 8-bit
    assert info["moov_first"]                                                 # streams on the web
    assert info["size"] == (1280, 720) and info["fps"] == pytest.approx(30)
    assert abs(info["frames"] - round((9 / 3 + 1) * 30)) <= 1                 # 3 s of run at 3x, plus 1 s hold


def test_size_fps_and_out_options(tmp_path):
    d = _make_run(tmp_path, (1600, 1000), _line((1600, 1000), n=4))
    res = _cli("--run", d, "--out", tmp_path / "elsewhere", "--size", "640x360", "--fps", "10")
    assert res.returncode == 0, res.stderr
    info = _mp4(tmp_path / "elsewhere" / "trajectory.mp4")
    assert info["size"] == (640, 360) and info["fps"] == pytest.approx(10)
    assert abs(info["frames"] - (3 + 1) * 10) <= 1
    assert not (d / "trajectory.mp4").exists()


@pytest.mark.parametrize("args, word", [(["--speed", "0"], "speed"), (["--fps", "0"], "fps"),
                                        (["--size", "641x360"], "even"), (["--size", "big"], "size")])
def test_bad_options_are_refused(tmp_path, args, word):
    d = _make_run(tmp_path, (1600, 1000), _line((1600, 1000), n=4))
    res = _cli("--run", d, *args)
    assert res.returncode != 0 and word in res.stderr.lower(), res.stderr


def test_a_missing_run_or_map_is_a_clear_error(tmp_path):
    res = _cli("--run", tmp_path / "nothing")
    assert res.returncode != 0 and "run.json" in res.stderr
    assert not (tmp_path / "nothing").exists()
    d = _make_run(tmp_path, (1600, 1000), _line((1600, 1000), n=4))
    (d / "map.png").unlink()
    res = _cli("--run", d)
    assert res.returncode != 0 and "map.png" in res.stderr


def test_both_renderers_share_one_style():
    assert ti.FIXED == draw.FIXED and ti.MISSED == draw.MISSED and ti.START == draw.START


# --- 2. under a minute -----------------------------------------------------------------------------------------

TASK_MAP = ROOT / "data" / "hackatons-data" / "global_map.png"


@pytest.mark.data
@pytest.mark.skipif(not TASK_MAP.exists(), reason="data/ not downloaded")
def test_a_50_frame_run_renders_in_under_a_minute(tmp_path):
    make_fake_run(TASK_MAP, tmp_path / "run")                   # 50 frames at 1 fps: a 50 s video
    t0 = time.perf_counter()
    path = tv.write_video(tmp_path / "run")
    assert time.perf_counter() - t0 < 60
    assert _mp4(path)["frames"] == pytest.approx((49 + 1) * 30, abs=1)


# --- 3. smooth -------------------------------------------------------------------------------------------------

def test_the_path_passes_through_every_record_at_its_time(tmp_path):
    pts = _sine((1600, 1000))
    records, info = _load(_make_run(tmp_path, (1600, 1000), pts))
    plan = tv.plan_video(records, info, fps=30, speed=1.0)
    for i, (x, y) in enumerate(pts):
        f = plan.frames[30 * i]                                  # record i is at t = i s = video frame 30 i
        assert f.t == pytest.approx(i) and (f.x, f.y) == (pytest.approx(x), pytest.approx(y)), i
        assert f.rec == i


def test_no_jumps_and_a_gradual_heading(tmp_path):
    records, info = _load(_make_run(tmp_path, (1600, 1000), _sine((1600, 1000))))
    plan = tv.plan_video(records, info, fps=30, speed=1.0)
    moving = [f for f in plan.frames if f.t < records[-1].t]
    xy = np.array([(f.x, f.y) for f in moving])
    steps = np.hypot(*np.diff(xy, axis=0).T)
    assert steps.max() < 2.5 * np.median(steps)                 # a curve, not straight lines with kinks
    heads = np.array([f.heading_deg for f in moving])
    assert np.abs((np.diff(heads) + 180) % 360 - 180).max() < 5  # degrees per video frame
    cam = np.array([(f.cx, f.cy) for f in plan.frames])
    assert np.hypot(*np.diff(cam, axis=0).T).max() < 2.5 * np.median(steps) + 1e-9


def test_the_hold_repeats_the_last_moment(tmp_path):
    records, info = _load(_make_run(tmp_path, (1600, 1000), _line((1600, 1000), n=4)))
    plan = tv.plan_video(records, info, fps=30, speed=1.0)
    last = [f for f in plan.frames if f.t >= records[-1].t]
    assert len(last) >= 30 and len({(f.x, f.y, f.rec) for f in last}) == 1


def test_heading_comes_from_the_motion_when_there_is_no_fix(tmp_path):
    pts = [(300 + 100 * i, 500) for i in range(8)]               # due east
    records, info = _load(_make_run(tmp_path, (1600, 1000), pts, missed={3, 4, 5}))
    plan = tv.plan_video(records, info, fps=30, speed=1.0)
    for f in plan.frames[30 * 3:30 * 5]:
        assert abs((f.heading_deg - 90 + 180) % 360 - 180) < 1


def test_the_footprint_is_interpolated_between_fixes_and_absent_without_one(tmp_path):
    pts = [(300 + 100 * i, 500 + 20 * i) for i in range(8)]
    records, info = _load(_make_run(tmp_path, (1600, 1000), pts, missed={5}))
    plan = tv.plan_video(records, info, fps=30, speed=1.0)
    a, b = records[1].track.fix.footprint, records[2].track.fix.footprint
    assert np.allclose(plan.frames[45].footprint, (a + b) / 2)     # halfway between records 1 and 2
    assert plan.frames[30 * 5].footprint is None                   # LOST: no fix to show
    assert plan.frames[30 * 4 + 15].footprint is None              # between a fix and no fix: none either


def test_the_camera_keeps_the_view_on_the_map(tmp_path):
    pts = [(60 + 120 * i, 60) for i in range(12)]                 # hugging the top edge
    records, info = _load(_make_run(tmp_path, (1600, 1000), pts))
    plan = tv.plan_video(records, info, fps=30, speed=1.0)
    half_w = plan.view_w / 2
    half_h = half_w * plan.size[1] / plan.size[0]
    for f in plan.frames:
        assert half_w - 1e-6 <= f.cx <= 1600 - half_w + 1e-6 and half_h - 1e-6 <= f.cy <= 1000 - half_h + 1e-6


def test_the_zoom_follows_the_footprint_size(tmp_path):
    small = tv.plan_video(*_load(_make_run(tmp_path, (3200, 2000), _line((3200, 2000)), scale=1.0, name="a")))
    large = tv.plan_video(*_load(_make_run(tmp_path, (3200, 2000), _line((3200, 2000)), scale=2.0, name="b")))
    assert large.view_w == pytest.approx(2 * small.view_w, rel=0.01)


# --- 4. what each frame shows ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def drawn(tmp_path_factory):
    """Record 5 of a line with 7 and 8 lost, drawn at its own time, and a lost frame."""
    d = _make_run(tmp_path_factory.mktemp("drawn"), (1600, 1000), _line((1600, 1000), n=10), missed={7, 8})
    records, info = _load(d)
    plan = tv.plan_video(records, info, fps=30, speed=1.0)
    renderer = tv.FrameRenderer(d, records, info, plan)
    img5, info5 = renderer.draw(30 * 5)
    img7, info7 = renderer.draw(30 * 7)
    return {"dir": d, "records": records, "plan": plan, "img5": img5, "info5": info5, "img7": img7, "info7": info7}


def _to_video(plan, k, x, y):
    f = plan.frames[k]
    s = plan.size[0] / plan.view_w
    return plan.size[0] / 2 + (x - f.cx) * s, plan.size[1] / 2 + (y - f.cy) * s


def test_the_drone_is_drawn_where_the_plan_says(drawn):
    x, y = drawn["info5"].drone_px
    assert (x, y) == pytest.approx(_to_video(drawn["plan"], 150, drawn["plan"].frames[150].x,
                                             drawn["plan"].frames[150].y), abs=0.5)
    assert _close(drawn["img5"][round(y), round(x)], draw.STATE_COLOURS["TRACKING"])
    x, y = drawn["info7"].drone_px
    assert _close(drawn["img7"][round(y), round(x)], draw.STATE_COLOURS["LOST"])


def test_the_trail_so_far_is_drawn_and_the_future_is_not(drawn):
    plan, recs, img = drawn["plan"], drawn["records"], drawn["img5"]
    behind = _to_video(plan, 150, (recs[3].track.x + recs[4].track.x) / 2, recs[3].track.y)
    ahead = _to_video(plan, 150, (recs[5].track.x + recs[6].track.x) / 2, recs[5].track.y)
    assert _close(img[round(behind[1]), round(behind[0])], draw.FIXED)
    assert not _changed(img[round(ahead[1]), round(ahead[0])])


def test_the_footprint_is_drawn_when_there_is_a_fix(drawn):
    plan, img = drawn["plan"], drawn["img5"]
    fp = plan.frames[150].footprint
    for k in range(4):
        mx, my = (fp[k] + fp[(k + 1) % 4]) / 2
        vx, vy = _to_video(plan, 150, mx, my)
        if abs(vy - drawn["info5"].drone_px[1]) > 8:                  # off the trail line
            assert _changed(img[round(vy), round(vx)]), k
    assert drawn["info5"].footprint and not drawn["info7"].footprint


def test_the_inset_is_the_current_frame(drawn):
    x0, y0, x1, y1 = drawn["info5"].inset_box
    inset = drawn["img5"][y0:y1, x0:x1].astype(int)
    src = cv2.imread(str(drawn["dir"] / drawn["records"][5].source))
    expected = cv2.resize(src, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(int)
    assert np.abs(inset - expected).mean() < 4


def test_the_hud_shows_the_record(drawn):
    assert (drawn["info5"].idx, drawn["info5"].state, drawn["info5"].inliers) == (5, "TRACKING", 105)
    assert (drawn["info7"].idx, drawn["info7"].state, drawn["info7"].inliers) == (7, "LOST", None)
    x0, y0, x1, y1 = drawn["info5"].hud_box
    assert np.mean([_changed(px) for px in drawn["img5"][y0:y1, x0:x1].reshape(-1, 3)]) > 0.3


def test_a_missing_frame_file_gets_a_placeholder(tmp_path):
    d = _make_run(tmp_path, (1600, 1000), _line((1600, 1000), n=4), write_frames=False)
    records, info = _load(d)
    plan = tv.plan_video(records, info)
    img, fi = tv.FrameRenderer(d, records, info, plan).draw(0)
    x0, y0, x1, y1 = fi.inset_box
    assert fi.frame_missing and img[y0:y1, x0:x1].std() < 40


def test_the_video_renders_the_fake_run(tmp_path):
    rng = np.random.default_rng(0)
    m = tmp_path / "map.png"
    cv2.imwrite(str(m), cv2.GaussianBlur(rng.uniform(0, 255, (900, 1400, 3)).astype(np.uint8), (0, 0), 2))
    make_fake_run(m, tmp_path / "run", )
    path = tv.write_video(tmp_path / "run", speed=10, size=(640, 360))
    assert _mp4(path)["size"] == (640, 360)


def test_the_file_is_small_enough_to_share(tmp_path):
    # OpenCV's encoder ignores quality settings and wrote 14 Mbit/s (87 MB for the 50 s fake run): too big for a
    # PR attachment or a README; a textured map is the hard case for the encoder
    rng = np.random.default_rng(0)
    m = tmp_path / "map.png"
    cv2.imwrite(str(m), cv2.GaussianBlur(rng.uniform(0, 255, (900, 1400, 3)).astype(np.uint8), (0, 0), 2))
    make_fake_run(m, tmp_path / "run", )
    path = tv.write_video(tmp_path / "run", speed=5)             # 49 s of run at 5x: ≈ 11 s of video
    seconds = _mp4(path)["frames"] / 30
    assert path.stat().st_size * 8 / seconds < 4e6, f"{path.stat().st_size * 8 / seconds / 1e6:.1f} Mbit/s"


def test_without_imageio_ffmpeg_opencv_still_writes_a_playable_file(tmp_path, monkeypatch):
    # the fallback: OpenCV's H.264 (big, but playable) with the index moved to the front by dronenav.viz.mp4
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)       # makes "import imageio_ffmpeg" fail
    d = _make_run(tmp_path, (1600, 1000), _line((1600, 1000), n=4))
    try:
        path = tv.write_video(d, size=(320, 180))
    except RuntimeError as e:                                        # an OpenCV build without H.264
        pytest.skip(str(e))
    info = _mp4(path)
    assert info["avc1"] and info["moov_first"] and info["size"] == (320, 180)
    assert abs(info["frames"] - (3 + 1) * 30) <= 1


def test_an_encoder_failure_says_what_ffmpeg_said(tmp_path, monkeypatch):
    pytest.importorskip("imageio_ffmpeg")
    monkeypatch.setattr(tv, "CRF", "not-a-number")                   # ffmpeg refuses the option and exits
    d = _make_run(tmp_path, (1600, 1000), _line((1600, 1000), n=4))
    with pytest.raises(RuntimeError, match="ffmpeg"):
        tv.write_video(d, size=(320, 180))
