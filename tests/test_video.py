"""M2-02 · Video frame source.

"Done when: a 60 s take at N = 10 yields the expected frame count with correct times." The spec: read MP4/MOV, keep
every N-th frame (or a target fps), carry real timestamps, optional resize, the same ``Frame`` type as the folder
source.

Every test films its own tiny take into a tmp dir. Each video frame carries its own number as a 4 x 3 grid of black
or white blocks (12 bits), which survives the codec and a resize, so a test can prove WHICH source frame was kept.
"""
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.types import Frame, FrameSource
from dronenav.video import VideoSource

W, H, FPS = 64, 48, 30
# Timestamps in an MP4/MOV are whole ticks of the container's time base (1/15360 s for these takes), and OpenCV hands
# them over in milliseconds as a float: far below 1 ms of error. 1 ms is still 33 times finer than one frame at
# 30 fps, so a time that belongs to a neighbouring frame can never pass.
T_TOL = 1e-3


def _pattern(i: int, w: int = W, h: int = H) -> np.ndarray:
    """Frame number i as 12 blocks: bit b is white at row b // 4, column b % 4."""
    img = np.zeros((h, w, 3), np.uint8)
    for b in range(12):
        if i >> b & 1:
            r, c = divmod(b, 4)
            img[r * h // 3:(r + 1) * h // 3, c * w // 4:(c + 1) * w // 4] = 255
    return img


def _number(img: np.ndarray) -> int:
    """The frame number a (possibly resized, lossy) frame carries: the middle of each block, black or white."""
    h, w = img.shape[:2]
    n = 0
    for b in range(12):
        r, c = divmod(b, 4)
        y, x = int((r + 0.5) * h / 3), int((c + 0.5) * w / 4)
        if img[y - 1:y + 2, x - 1:x + 2].mean() > 127:
            n |= 1 << b
    return n


def _take(path: Path, frames: int, fps: float = FPS, size: tuple[int, int] = (W, H)) -> Path:
    """A constant-frame-rate take of ``frames`` numbered frames (MPEG-4: in every OpenCV build, no ffmpeg needed)."""
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert out.isOpened()
    for i in range(frames):
        out.write(_pattern(i, *size))
    out.release()
    return path


def _ffmpeg() -> str:
    """An ffmpeg for the takes OpenCV cannot write (variable frame rate, rotation flag): the one the repo ships
    with (imageio-ffmpeg, in requirements.txt), else one on PATH; without either those tests are skipped."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if exe is None:
            pytest.skip("needs ffmpeg (pip install imageio-ffmpeg)")
        return exe


# every duration is a whole number of 1/25 s, the grid ffmpeg's concat reader puts still images on, so the
# timestamps in the file are exactly the running sum
VFR_DURATIONS = [0.04, 0.08, 0.20, 0.04, 0.12, 0.40, 0.04, 0.08, 0.16, 0.04]
VFR_TIMES = [round(sum(VFR_DURATIONS[:i]), 2) for i in range(len(VFR_DURATIONS))]


def _vfr_take(tmp_path: Path) -> Path:
    """A variable-frame-rate take, as phones film: frame i is shown for VFR_DURATIONS[i] seconds."""
    lines = []
    for i, d in enumerate(VFR_DURATIONS):
        assert cv2.imwrite(str(tmp_path / f"{i}.png"), _pattern(i))
        lines += [f"file '{i}.png'", f"duration {d}"]
    (tmp_path / "list.txt").write_text("\n".join(lines) + "\n")
    out = tmp_path / "vfr.mp4"
    done = subprocess.run([_ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(tmp_path / "list.txt"),
                           "-fps_mode", "vfr", "-c:v", "mpeg4", "-q:v", "2", str(out)], capture_output=True)
    assert done.returncode == 0, done.stderr[-500:]
    return out


class _Backend:
    """A real cv2.VideoCapture whose chosen properties lie, as some containers and backends do."""
    real = cv2.VideoCapture
    lies: dict = {}

    def __init__(self, *args, **kwargs):
        self._cap = self.real(*args, **kwargs)

    def get(self, prop):
        value = self._cap.get(prop)
        lie = self.lies.get(prop)
        return value if lie is None else lie(value)

    def __getattr__(self, name):
        return getattr(self._cap, name)


def _lying_backend(monkeypatch, lies: dict) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", type("Backend", (_Backend,), {"lies": lies}))


# --- the spec's "Done when" --------------------------------------------------------------------------------------

def test_a_60_s_take_at_n_10_yields_180_frames_with_their_real_times(tmp_path):
    src = VideoSource(_take(tmp_path / "take.mp4", frames=60 * FPS), every=10)
    frames = list(src)
    assert len(frames) == len(src) == 180                               # 1800 / 10
    assert [f.idx for f in frames] == list(range(0, 1800, 10))          # idx is the frame number in the video
    assert [_number(f.image) for f in frames] == [f.idx for f in frames]   # and it IS that frame, not a neighbour
    for f in frames:
        assert f.t == pytest.approx(f.idx / FPS, abs=T_TOL)
    assert frames[-1].t == pytest.approx(59.0 + 20 / 30, abs=T_TOL)     # the last kept frame, 1790


# --- which frames -------------------------------------------------------------------------------------------------

def test_without_a_step_every_frame_comes_out(tmp_path):
    src = VideoSource(_take(tmp_path / "take.mp4", frames=25))
    frames = list(src)
    assert src.step == 1 and len(src) == 25
    assert [f.idx for f in frames] == [_number(f.image) for f in frames] == list(range(25))


def test_a_last_partial_step_still_gives_its_first_frame(tmp_path):
    src = VideoSource(_take(tmp_path / "take.mp4", frames=95), every=10)
    assert [f.idx for f in src] == list(range(0, 95, 10))               # 0 ... 90: frames 91-94 fill no step
    assert len(src) == 10


def test_a_step_longer_than_the_take_gives_the_first_frame(tmp_path):
    src = VideoSource(_take(tmp_path / "take.mp4", frames=7), every=100)
    assert [f.idx for f in src] == [0] and len(src) == 1


@pytest.mark.parametrize(("fps", "step", "rate"), [(3, 10, 3.0), (30, 1, 30.0), (10.0, 3, 10.0),
                                                   (12, 3, 10.0),      # 30 / 12 = 2.5 rounds half up, as FlightSource
                                                   (0.5, 60, 0.5)])
def test_a_target_fps_becomes_a_whole_step(tmp_path, fps, step, rate):
    src = VideoSource(_take(tmp_path / "take.mp4", frames=65), fps=fps)
    assert src.step == step
    assert src.rate == pytest.approx(rate)                              # the rate it really gives
    assert src.fps == pytest.approx(FPS)                                # the take's own rate
    frames = list(src)
    assert [f.idx for f in frames] == [_number(f.image) for f in frames] == list(range(0, 65, step))
    for f in frames:
        assert f.t == pytest.approx(f.idx / FPS, abs=T_TOL)


def test_a_target_fps_above_the_takes_own_is_refused(tmp_path):
    with pytest.raises(ValueError, match="fps"):
        VideoSource(_take(tmp_path / "take.mp4", frames=5), fps=FPS + 1)


def test_a_step_and_a_target_fps_together_are_refused(tmp_path):
    with pytest.raises(ValueError, match="every.*fps|fps.*every"):
        VideoSource(_take(tmp_path / "take.mp4", frames=5), every=2, fps=15)


@pytest.mark.parametrize("every", [0, -1, 2.5, 2.0, True, "3", float("nan")])
def test_a_step_that_is_not_a_positive_whole_number_is_refused(tmp_path, every):
    with pytest.raises(ValueError, match="every"):
        VideoSource(_take(tmp_path / "take.mp4", frames=5), every=every)


@pytest.mark.parametrize("fps", [0, -5, float("nan"), float("inf"), True, "10"])
def test_a_target_fps_that_is_not_a_frame_rate_is_refused(tmp_path, fps):
    with pytest.raises(ValueError, match="fps"):
        VideoSource(_take(tmp_path / "take.mp4", frames=5), fps=fps)


def test_numpy_numbers_are_fine(tmp_path):
    take = _take(tmp_path / "take.mp4", frames=12)
    assert VideoSource(take, every=np.int64(4)).step == 4
    assert VideoSource(take, fps=np.float32(10)).step == 3
    assert VideoSource(take, max_side=np.int32(32)).size == (32, 24)


# --- the Frame contract, as FolderSource and FlightSource give it -------------------------------------------------

@pytest.mark.parametrize("name", ["take.mp4", "take.mov", "TAKE.MOV"])
def test_mp4_and_mov_give_frames_like_the_other_sources(tmp_path, name):
    take = _take(tmp_path / name, frames=12)
    src = VideoSource(take, every=5)
    assert isinstance(src, FrameSource)
    frames = list(src)
    assert [f.idx for f in frames] == [0, 5, 10]
    for f in frames:
        assert isinstance(f, Frame)
        assert f.image.dtype == np.uint8 and f.image.shape == (H, W, 3)        # BGR, as OpenCV reads it
        assert f.source == f"{take}#{f.idx}"                                    # FlightSource's name for a frame
        assert f.heading_prior_deg is None and f.scale_prior is None
    assert src.size == (W, H)


def test_a_path_may_be_a_string(tmp_path):
    take = _take(tmp_path / "take.mp4", frames=3)
    assert [f.source for f in VideoSource(str(take))] == [f"{take}#{i}" for i in range(3)]


def test_iterating_again_starts_again_and_gives_the_same_frames(tmp_path):
    src = VideoSource(_take(tmp_path / "take.mp4", frames=40), every=7, max_side=32)
    first, second = list(src), list(src)
    assert [f.idx for f in first] == [f.idx for f in second] == list(range(0, 40, 7))
    assert [f.t for f in first] == [f.t for f in second]
    assert all(np.array_equal(a.image, b.image) for a, b in zip(first, second, strict=True))
    assert first[0].image is not first[1].image                         # each frame owns its pixels


def test_an_iteration_left_half_way_does_not_disturb_the_next(tmp_path):
    src = VideoSource(_take(tmp_path / "take.mp4", frames=30), every=3)
    it = iter(src)
    assert [next(it).idx for _ in range(4)] == [0, 3, 6, 9]
    assert [f.idx for f in src] == list(range(0, 30, 3))
    assert next(it).idx == 12                                           # the first one carries on where it was


# --- timestamps ---------------------------------------------------------------------------------------------------

def test_a_variable_frame_rate_take_keeps_the_times_of_the_container(tmp_path):
    src = VideoSource(_vfr_take(tmp_path))
    frames = list(src)
    assert [_number(f.image) for f in frames] == [f.idx for f in frames] == list(range(10))
    assert [f.t for f in frames] == pytest.approx(VFR_TIMES, abs=T_TOL)   # not idx / fps: the camera's own clock
    # and idx / fps would be wrong by far more than the tolerance, so this test can tell the two clocks apart
    assert max(abs(f.idx / src.fps - t) for f, t in zip(frames, VFR_TIMES, strict=True)) > 0.1


def test_every_nth_frame_of_a_variable_rate_take_has_its_own_time(tmp_path):
    frames = list(VideoSource(_vfr_take(tmp_path), every=3))
    assert [f.idx for f in frames] == [_number(f.image) for f in frames] == [0, 3, 6, 9]
    assert [f.t for f in frames] == pytest.approx([VFR_TIMES[i] for i in (0, 3, 6, 9)], abs=T_TOL)


@pytest.mark.parametrize("stamp", [0.0, -1.0, float("nan"), float("inf")])
def test_without_usable_container_times_the_time_is_index_over_fps(tmp_path, monkeypatch, stamp):
    take = _take(tmp_path / "take.mp4", frames=31)
    _lying_backend(monkeypatch, {cv2.CAP_PROP_POS_MSEC: lambda _: stamp})
    frames = list(VideoSource(take, every=10))
    assert [f.idx for f in frames] == [0, 10, 20, 30]
    # exact arithmetic on both sides; the tolerance only covers float rounding
    assert [f.t for f in frames] == pytest.approx([0.0, 10 / FPS, 20 / FPS, 1.0], abs=1e-9)


def test_times_that_stop_half_way_carry_on_at_the_frame_rate(tmp_path, monkeypatch):
    take = _take(tmp_path / "take.mp4", frames=40)
    _lying_backend(monkeypatch, {cv2.CAP_PROP_POS_MSEC: lambda ms: ms if ms < 500 else 0.0})   # dies at frame 15
    frames = list(VideoSource(take, every=5))
    ts = [f.t for f in frames]
    assert all(b > a for a, b in zip(ts, ts[1:], strict=False))         # time never runs back or stands still
    assert ts == pytest.approx([f.idx / FPS for f in frames], abs=T_TOL)   # the same clock, carried on


def test_time_starts_at_zero_when_the_container_starts_later(tmp_path, monkeypatch):
    take = _take(tmp_path / "take.mp4", frames=10)
    _lying_backend(monkeypatch, {cv2.CAP_PROP_POS_MSEC: lambda ms: ms + 2500.0})    # a stream that starts at 2.5 s
    frames = list(VideoSource(take, every=3))
    assert [f.t for f in frames] == pytest.approx([0.0, 0.1, 0.2, 0.3], abs=T_TOL)   # seconds from the first frame


# --- the length ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("claim", [0.0, 7.0, 5000.0, -1.0, float("nan")])
def test_the_length_is_counted_not_taken_from_the_header(tmp_path, monkeypatch, claim):
    take = _take(tmp_path / "take.mp4", frames=23)
    _lying_backend(monkeypatch, {cv2.CAP_PROP_FRAME_COUNT: lambda _: claim})
    src = VideoSource(take, every=4)
    assert len(src) == 6                                                # 0, 4, ..., 20: before any iteration
    assert len(list(src)) == 6
    assert len(src) == 6


# --- resize -------------------------------------------------------------------------------------------------------

def test_max_side_shrinks_the_frame_and_keeps_its_shape(tmp_path):
    src = VideoSource(_take(tmp_path / "take.mp4", frames=9), every=4, max_side=32)
    assert src.size == (32, 24)                                         # (w, h) of the frames it yields
    assert src.source_size == (W, H)
    frames = list(src)
    assert all(f.image.shape == (24, 32, 3) and f.image.dtype == np.uint8 for f in frames)
    assert [_number(f.image) for f in frames] == [0, 4, 8]              # the same picture, smaller


def test_max_side_is_the_longer_side_of_a_portrait_take(tmp_path):
    src = VideoSource(_take(tmp_path / "take.mp4", frames=3, size=(48, 64)), max_side=32)
    assert src.size == (24, 32)
    assert next(iter(src)).image.shape == (32, 24, 3)


def test_a_size_that_does_not_divide_evenly_keeps_the_aspect_to_the_pixel(tmp_path):
    src = VideoSource(_take(tmp_path / "take.mp4", frames=3), max_side=50)
    w, h = src.size
    assert w == 50                                                      # the longer side is exactly max_side
    assert abs(h - 50 * H / W) <= 0.5                                   # 37.5: the nearest whole pixel, 37 or 38
    assert next(iter(src)).image.shape == (h, w, 3)


@pytest.mark.parametrize("max_side", [64, 65, 4000])
def test_max_side_never_enlarges(tmp_path, max_side):
    src = VideoSource(_take(tmp_path / "take.mp4", frames=3), max_side=max_side)
    assert src.size == (W, H)
    assert next(iter(src)).image.shape == (H, W, 3)


@pytest.mark.parametrize("max_side", [0, -32, 31.5, 32.0, True, "32", float("nan")])
def test_a_max_side_that_is_not_a_positive_whole_number_is_refused(tmp_path, max_side):
    with pytest.raises(ValueError, match="max_side"):
        VideoSource(_take(tmp_path / "take.mp4", frames=3), max_side=max_side)


# --- phone files --------------------------------------------------------------------------------------------------

def test_a_take_filmed_with_the_phone_upright_comes_out_upright(tmp_path):
    """Phones store portrait video as landscape pixels plus a rotation flag; players (and we) turn it."""
    stored = _take(tmp_path / "stored.mp4", frames=6)                   # 64 x 48 pixels in the file
    flagged = tmp_path / "portrait.mp4"
    done = subprocess.run([_ffmpeg(), "-y", "-display_rotation", "90", "-i", str(stored), "-c", "copy",
                           str(flagged)], capture_output=True)
    assert done.returncode == 0, done.stderr[-500:]
    src = VideoSource(flagged, every=1)
    assert src.size == (H, W)                                           # 48 x 64: turned, and size says so
    frames = list(src)
    assert len(frames) == 6 and all(f.image.shape == (W, H, 3) for f in frames)
    # frame 1 has one white block, in the stored top-left corner: a quarter turn moves it to the top-right or the
    # bottom-left corner (which one is the flag's sign convention), never leaves it or mirrors it elsewhere
    white = frames[1].image[..., 0] > 127
    ys, xs = np.nonzero(white)
    corner = (ys.mean() < W / 2, xs.mean() < H / 2)                     # (in the top half, in the left half)
    assert corner in [(True, False), (False, True)]
    # with max_side the turned frame is what gets measured
    assert VideoSource(flagged, max_side=32).size == (24, 32)


# --- failures -----------------------------------------------------------------------------------------------------

def test_a_missing_video_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.mp4"):
        VideoSource(tmp_path / "nope.mp4")


def test_a_folder_is_not_a_video(tmp_path):
    with pytest.raises(FileNotFoundError, match="no video"):
        VideoSource(tmp_path)


@pytest.mark.parametrize("content", [b"", b"this is not a video" * 200])
def test_an_empty_or_unreadable_file_is_named(tmp_path, content):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(content)
    with pytest.raises(ValueError, match="bad.mp4"):
        VideoSource(bad)


def test_a_take_cut_off_while_copying_is_named(tmp_path):
    whole = _take(tmp_path / "whole.mp4", frames=60)
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(whole.read_bytes()[:whole.stat().st_size // 2])     # the index at the end of the file is gone
    with pytest.raises(ValueError, match="cut.mp4"):
        VideoSource(cut)


def test_a_video_without_frames_is_named(tmp_path):
    out = cv2.VideoWriter(str(tmp_path / "none.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    out.release()                                                       # a header (or nothing), no picture
    with pytest.raises((ValueError, FileNotFoundError), match="none.mp4"):
        VideoSource(tmp_path / "none.mp4")


@pytest.mark.parametrize("claim", [0.0, -1.0, float("nan"), float("inf")])
def test_a_take_that_does_not_say_its_frame_rate_is_refused(tmp_path, monkeypatch, claim):
    take = _take(tmp_path / "take.mp4", frames=5)
    _lying_backend(monkeypatch, {cv2.CAP_PROP_FPS: lambda _: claim})
    with pytest.raises(ValueError, match="frame rate"):
        VideoSource(take)


def test_the_test_pattern_reads_back_its_own_numbers():
    """The proof of "which frame" rests on this."""
    for i in (0, 1, 2, 1023, 1790, 4095):
        assert _number(_pattern(i)) == i
        assert _number(cv2.resize(_pattern(i), (32, 24), interpolation=cv2.INTER_AREA)) == i
