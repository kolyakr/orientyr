"""M2-09 · scripts/run.py --video: the same run as --frames, from a video.

Done when (issue #127):
1. ``run.py --map M --video V --out O`` writes the run folder, the trajectory image and video, like --frames;
2. on a short synthetic video cut from a map with known truth (M6-01 cut-outs written as an mp4), every sampled
   frame is localized within a stated tolerance;
3. ``--every N`` gives the expected number of records (and the default comes from the video's fps: ~3 frames/s);
   a missing or unreadable video is a clear error with exit != 0;
4. NOTES.md shows the real --video command.
Records keep the video's frame index and time; the sampled frames are saved in the run folder for the inset.
"""
import math
from pathlib import Path

import cv2
import pytest
from tests.test_run import OUTPUTS, _city, _cli

from dronenav.sim import CutoutSource, cutout_route
from dronenav.telemetry import read_run, read_run_info

ROOT = Path(__file__).resolve().parents[1]
FPS = 6.0          # a slow take: at the default ~3 frames/s it is sampled every 2nd frame
N = 24             # frames in the take: 4 s


@pytest.fixture(scope="module")
def take(tmp_path_factory):
    """A map, and an mp4 of N cut-out frames along a seeded route over it, with each frame's truth."""
    d = tmp_path_factory.mktemp("take")
    img = _city(2000, 1500)
    cv2.imwrite(str(d / "map.png"), img)
    poses = cutout_route((img.shape[1], img.shape[0]), N, seed=5, frame_size=(320, 240), scale=(0.9, 1.1),
                         tilt_max_deg=3.0)
    src = CutoutSource(img, poses, (320, 240))
    out = cv2.VideoWriter(str(d / "take.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 240))
    assert out.isOpened()
    for f in src:
        out.write(f.image)
    out.release()
    return d / "map.png", d / "take.mp4", src


def _run(tmp_path, take, *extra):
    m, v, _ = take
    return _cli("--map", m, "--video", v, "--out", tmp_path / "out", "--cache", tmp_path / "cache", *extra)


# --- 1 + 2. the deliverable, and every sampled frame localized ------------------------------------------------

def test_every_frame_of_the_video_is_localized(tmp_path, take):
    res = _run(tmp_path, take, "--every", "1")
    assert res.returncode == 0, res.stderr
    for name in OUTPUTS:
        assert (tmp_path / "out" / name).is_file(), name
    recs = read_run(tmp_path / "out" / "run.jsonl")
    assert [r.idx for r in recs] == list(range(N))
    for r in recs:
        tr = take[2].truth(r.idx)
        assert r.track.fix is not None and r.track.fix.accepted, r.idx
        # 3 px: an undistorted cut-out is found to sub-pixel (2 px in test_run); MPEG-4 compression blurs edges
        assert math.hypot(r.track.fix.x - tr.center[0], r.track.fix.y - tr.center[1]) <= 3.0, r.idx
        assert r.t == pytest.approx(r.idx / FPS, abs=1e-3)       # the video's own clock


def test_the_run_info_has_the_video_rate_and_frame_size(tmp_path, take):
    assert _run(tmp_path, take, "--every", "4").returncode == 0
    info = read_run_info(tmp_path / "out")
    assert info.fps == pytest.approx(FPS)
    assert tuple(info.frame_size) == (320, 240)


def test_sampled_frames_are_saved_in_the_run_folder_for_the_inset(tmp_path, take):
    assert _run(tmp_path, take, "--every", "4").returncode == 0
    out = (tmp_path / "out").resolve()
    for r in read_run(out / "run.jsonl"):
        p = (out / r.source).resolve()
        assert out in p.parents, r.source                          # moves with the run folder
        img = cv2.imread(str(p))
        assert img is not None and img.shape[:2] == (240, 320), r.source


# --- 3. sampling and errors --------------------------------------------------------------------------------------

@pytest.mark.parametrize("every, idxs", [(5, [0, 5, 10, 15, 20]), (7, [0, 7, 14, 21])])
def test_every_n_gives_every_nth_frame(tmp_path, take, every, idxs):
    res = _run(tmp_path, take, "--every", str(every))
    assert res.returncode == 0, res.stderr
    assert [r.idx for r in read_run(tmp_path / "out" / "run.jsonl")] == idxs


def test_the_default_sampling_is_about_3_frames_per_second(tmp_path, take):
    res = _run(tmp_path, take)
    assert res.returncode == 0, res.stderr
    # 6 fps -> every 2nd frame: 3 frames/s, inside the spec's 2-5
    assert [r.idx for r in read_run(tmp_path / "out" / "run.jsonl")] == list(range(0, N, 2))


def test_a_missing_video_is_a_clear_error(tmp_path, take):
    m = take[0]
    res = _cli("--map", m, "--video", tmp_path / "nope.mp4", "--out", tmp_path / "out")
    assert res.returncode == 2
    assert "nope.mp4" in res.stderr
    assert not (tmp_path / "out").exists()


def test_an_unreadable_video_is_a_clear_error(tmp_path, take):
    bad = tmp_path / "bad.mp4"
    bad.write_text("not a video")
    res = _cli("--map", take[0], "--video", bad, "--out", tmp_path / "out")
    assert res.returncode == 2
    assert "bad.mp4" in res.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("args", [("--every", "0"), ("--every", "x"), ("--every", "-2")])
def test_a_bad_every_is_a_clear_error(tmp_path, take, args):
    res = _run(tmp_path, take, *args)
    assert res.returncode == 2
    assert "every" in res.stderr
    assert not (tmp_path / "out").exists()


def test_every_needs_a_video(tmp_path, take):
    frames = tmp_path / "frames"
    frames.mkdir()
    res = _cli("--map", take[0], "--frames", frames, "--every", "2", "--out", tmp_path / "out")
    assert res.returncode == 2
    assert "--every" in res.stderr and "--video" in res.stderr


# --- 4. the command is documented --------------------------------------------------------------------------------

def test_notes_show_the_real_video_command():
    lines = [ln for ln in (ROOT / "NOTES.md").read_text().splitlines()
             if "scripts/run.py" in ln and "--video" in ln]
    assert lines, "no run.py --video command in NOTES.md"
    assert any("--every" in ln for ln in lines), lines
    assert not any("data/phone.mp4" in ln for ln in lines), "still the placeholder command"
