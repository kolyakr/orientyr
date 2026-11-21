"""The pipeline (M1-09): a map and a folder of frames (or a video, M2-09) in, a run folder with the trajectory
image and video out.

    frames (io.FolderSource,   ─► Searcher.locate per frame ─► tracker ─► records ─► run.json + run.jsonl ─► viz
     or video.VideoSource)      (matcher, scale, geometry,     (states,              (telemetry)
                                 window or whole map)           Kalman)

A video (M2-09) is sampled every N-th frame (``every``; by default N is chosen from the take's fps so that about
VIDEO_RATE frames per second are processed: a phone's 30 fps would otherwise be ten times the work for frames a
third of a metre apart). Records keep the video's frame index and time. Each sampled frame is saved as
``frames/frame_<index>.jpg`` in the run folder and is the record's source, so the renderers' inset (and anything
else that reads ``source``) sees an ordinary image, and the run folder stays whole when it is moved.

The engine (matcher, scale rule, geometry checks, search strategy) is M1-02 to M1-05; this module wires it to the
run folder the renderers read (M0-03, M0-04 convention) and gives every frame a state.

``tracker="kalman"`` (the default, M5-02): the tracker (``dronenav.tracking.states``) takes each frame's fix; its
state (TRACKING / LOST / RELOCALIZING), filtered position, velocity and covariance are the frame's TrackState, and
the filter step (INIT, HIT, GATED, COAST, NONE) is written to ``extra["step"]``. While it is LOST the next frame is
searched on the whole map. Otherwise it is searched in the window from the tracker's prediction (M5-03): the last
footprint at the predicted centre, grown by 3σ; ``extra["predicted_window"]`` and ``extra["window_sigma_px"]`` (the
larger axis σ) record it, so the window shrinking while tracking and growing while coasting is in the telemetry.
Frames before the first fix are LOST at that fix's position, as below.

``tracker="none"``: M1's interim rule, kept for comparison:

- TRACKING: an accepted fix; RELOCALIZING: the first accepted fix after the track was lost (misses before the
  very first fix are not a lost track: that first fix is TRACKING);
- LOST: no accepted fix. The position coasts on the velocity between the last two fixes for at most
  COAST_MAX_S seconds, then holds: a stale velocity flown for minutes ends kilometres off the map. The
  uncertainty grows by LOST_SIGMA_GROWTH_PX_PER_S every second since the last fix. Misses before the first fix
  take that fix's position (there is no "position unknown" yet, see #87). Interpolation and smoothing: M1-06.

Everything is checked and localized before anything is written; the run is then written into a temporary
sibling folder (``.OUT.partial``, so relative paths are the same) and its five outputs move into OUT only once
all of them exist. A failure leaves OUT as it was, and other files in OUT are never touched. A run in which no
frame is localized raises :class:`NothingLocalized` instead of drawing an empty trajectory.
"""
from __future__ import annotations

import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dronenav.io import FolderSource, _is_rate, load_map
from dronenav.matching import RootSiftMatcher
from dronenav.search import Located, Searcher, SearchMode, prediction_window
from dronenav.telemetry import Record, RunInfo, RunWriter, write_run_info
from dronenav.tracking.states import Tracker, TrackerUpdate
from dronenav.types import Fix, TrackMode, TrackState
from dronenav.video import VideoSource
from dronenav.viz.trajectory_image import write_trajectory
from dronenav.viz.trajectory_video import write_video

__all__ = [
    "COAST_MAX_S",
    "LOST_SIGMA_GROWTH_PX_PER_S",
    "MATCHERS",
    "OUTPUTS",
    "TRACKERS",
    "VIDEO_RATE",
    "NothingLocalized",
    "RunSummary",
    "localize_run",
    "run",
]

# --matcher name -> (backend, factory); the learned matchers (M4) join here
MATCHERS = {"sift": (RootSiftMatcher.backend, lambda map_img, cache_dir: RootSiftMatcher(map_img, cache_dir=cache_dir))}
TRACKERS = ("kalman", "none")    # --tracker: the M5 tracker, or M1's interim states
VIDEO_RATE = 3.0                 # frames/s taken from a video by default (the spec asks for 2-5)
FRAMES_DIR = "frames"            # a video's sampled frames, in the run folder
OUTPUTS = ("run.json", "run.jsonl", "trajectory.png", "trajectory_preview.jpg", "trajectory.mp4")

COAST_MAX_S = 5                  # seconds a lost track coasts on its last velocity before it holds
LOST_SIGMA_GROWTH_PX_PER_S = 20.0  # how much less sure the position gets every second without a fix


class NothingLocalized(ValueError):
    """No frame of the run could be found on the map."""


@dataclass(frozen=True)
class RunSummary:
    out_dir: Path
    frames: int
    localized: int
    seconds: float
    image: Path | None = None
    video: Path | None = None


@dataclass(frozen=True)
class _Found:
    """What one frame's search left, without the match arrays (a long run must not keep them all)."""

    idx: int
    t: float
    path: Path
    size: tuple[int, int]
    fix: Fix | None
    mode: str
    window: tuple[int, int, int, int] | None
    factor: float
    ms: float
    tracked: TrackerUpdate | None = None      # the tracker's result for this frame (tracker="kalman")
    predicted: tuple[int, int, int, int] | None = None   # the window from the prediction (tracker="kalman")
    sigma: float | None = None                # the prediction's larger axis σ, px (tracker="kalman")

    @property
    def accepted(self) -> bool:
        return self.fix is not None and self.fix.accepted


def _check_out(out_dir: Path) -> None:
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"--out {out_dir} exists and is not a folder")


def _check_options(matcher: str, tracker: str) -> None:
    if matcher not in MATCHERS:
        raise ValueError(f"unknown matcher {matcher!r} (available: {', '.join(sorted(MATCHERS))})")
    if tracker not in TRACKERS:
        raise ValueError(f"unknown tracker {tracker!r} (available: {', '.join(TRACKERS)})")


def _frame_source(frames_dir, video, fps, every):
    """The frames: a folder (FolderSource) or a video (VideoSource, every N-th frame; see the module docstring)."""
    if (frames_dir is None) == (video is None):
        raise ValueError("give a folder of frames or a video, not both and not neither")
    if video is None:
        if every is not None:
            raise ValueError("every (each N-th frame) is for a video, not a folder of frames")
        return FolderSource(frames_dir, fps=fps)
    if fps is not None:
        raise ValueError("fps (the rate of the numbers in frame names) is for a folder of frames; a video has its "
                         "own: sample it with every")
    if every is not None:
        return VideoSource(video, every=every)
    take = VideoSource(video)
    return VideoSource(video, fps=min(VIDEO_RATE, take.fps))


def _keep(f, keep_dir: Path | None) -> Path:
    """The frame's path for the record: its file, or (a video frame) a JPEG saved into keep_dir."""
    if keep_dir is None or Path(f.source).is_file():
        return Path(f.source)
    keep_dir.mkdir(parents=True, exist_ok=True)
    path = keep_dir / f"frame_{f.idx:06d}.jpg"
    if not cv2.imwrite(str(path), f.image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"cannot write {path}")
    return path


def _localize(map_path, frames_dir, matcher: str, fps, cache_dir, tracker: str = "kalman", *, video=None,
              every=None, keep_dir: Path | None = None):
    """Check the inputs and find every frame; nothing is written but a video's sampled frames (into keep_dir).
    Returns (map image, frames source, found)."""
    _check_options(matcher, tracker)
    frames = _frame_source(frames_dir, video, fps, every)
    the_map = load_map(map_path)
    track = Tracker() if tracker == "kalman" else None
    try:
        searcher = Searcher(MATCHERS[matcher][1](the_map.image, cache_dir), the_map.image)
        found = []
        for f in frames:
            window = sigma = tracked = None
            if track is None:
                loc: Located = searcher.locate(f.idx, f.image)            # M1's window rule
            else:
                if track.search is SearchMode.GLOBAL:
                    searcher.reset()              # LOST: the old track is not trusted, search the whole map
                pred = track.predict(f.t)
                if pred is not None:              # M5-03: the window from the prediction
                    sigma = math.sqrt(max(pred.cov[0, 0], pred.cov[1, 1]))
                    window = prediction_window(track.last_fix, (pred.x, pred.y), pred.cov, searcher.map_size,
                                               searcher.cfg)
                loc = searcher.locate_in(f.idx, f.image, window)
                tracked = track.update(f.idx, f.t, loc.fix)
            found.append(_Found(f.idx, f.t, _keep(f, keep_dir), (f.image.shape[1], f.image.shape[0]), loc.fix,
                                loc.mode.value, loc.window, float(loc.factor), float(loc.ms), tracked, window,
                                sigma))
    except cv2.error as e:                # FLANN refuses an empty index: a blank or flat map
        raise ValueError(f"matching on {map_path} failed: the map has no usable features (blank or flat image?) "
                         f"[{e.err}]") from None
    if not any(x.accepted for x in found):
        raise NothingLocalized(f"no frame of {frames_dir if video is None else video} could be localized on {map_path} "
                               f"({len(found)} frames tried): is it the right map?")
    return the_map.image, frames, found


def _write_run(out_dir: Path, map_path, map_img, frames: FolderSource | VideoSource, found, backend: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = map_img.shape[:2]
    sizes = {x.size for x in found}
    write_run_info(out_dir, RunInfo(map=str(map_path), map_size=(w, h), generator="run.py",
                                    frame_size=sizes.pop() if len(sizes) == 1 else None, fps=frames.fps))
    with RunWriter(out_dir / "run.jsonl") as writer:
        for rec in _records(found, out_dir, backend):
            writer.write(rec)


def localize_run(map_path: str | Path, frames_dir: str | Path | None, out_dir: str | Path, *, matcher: str = "sift",
                 fps: float | None = None, cache_dir: str | Path | None = "outputs/cache",
                 tracker: str = "kalman", video: str | Path | None = None, every: int | None = None) -> RunSummary:
    """Find every frame on the map and write ``run.json`` and ``run.jsonl`` into ``out_dir`` (no trajectory
    image or video; a video's sampled frames go to ``out_dir/frames``)."""
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    _check_out(out_dir)
    map_img, frames, found = _localize(map_path, frames_dir, matcher, fps, cache_dir, tracker, video=video,
                                       every=every, keep_dir=None if video is None else out_dir / FRAMES_DIR)
    _write_run(out_dir, map_path, map_img, frames, found, MATCHERS[matcher][0])
    return RunSummary(out_dir=out_dir, frames=len(found), localized=sum(x.accepted for x in found),
                      seconds=time.perf_counter() - t0)


def run(map_path: str | Path, frames_dir: str | Path | None, out_dir: str | Path, *, matcher: str = "sift",
        fps: float | None = None, speed: float = 1.0,
        cache_dir: str | Path | None = "outputs/cache", tracker: str = "kalman",
        video: str | Path | None = None, every: int | None = None) -> RunSummary:
    """The whole deliverable: localize, then the trajectory image (M1-07) and the video (M1-08), all into
    ``out_dir`` at once (see the module docstring). Frames come from ``frames_dir`` or, instead, ``video``
    (every ``every``-th frame; by default about VIDEO_RATE per second)."""
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    if not _is_rate(speed):
        raise ValueError(f"speed must be a positive number, got {speed!r}")
    _check_out(out_dir)
    _check_options(matcher, tracker)
    # a sibling folder: the same depth, so every relative path written into it holds in out_dir too
    part = out_dir.parent / f".{out_dir.name}.partial"
    shutil.rmtree(part, ignore_errors=True)
    try:
        map_img, frames, found = _localize(map_path, frames_dir, matcher, fps, cache_dir, tracker, video=video,
                                           every=every, keep_dir=None if video is None else part / FRAMES_DIR)
        _write_run(part, map_path, map_img, frames, found, MATCHERS[matcher][0])
        write_trajectory(part, map_img=map_img)
        write_video(part, map_img=map_img, speed=speed)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in OUTPUTS:                               # all five exist: now they replace the old ones
            os.replace(part / name, out_dir / name)
        if (part / FRAMES_DIR).is_dir():                   # a video's frames: the old run's go with its records
            shutil.rmtree(out_dir / FRAMES_DIR, ignore_errors=True)
            os.replace(part / FRAMES_DIR, out_dir / FRAMES_DIR)
    finally:
        shutil.rmtree(part, ignore_errors=True)
    return RunSummary(out_dir=out_dir, frames=len(found), localized=sum(x.accepted for x in found),
                      seconds=time.perf_counter() - t0, image=out_dir / "trajectory.png",
                      video=out_dir / "trajectory.mp4")


def _records(found: list[_Found], out_dir: Path, backend: str):
    """One Record per frame, with the states, coasting and uncertainty described in the module docstring."""
    if found and found[0].tracked is not None:
        yield from _tracked_records(found, out_dir, backend)
        return
    anchor = next(x.fix for x in found if x.accepted)    # where misses before the first fix are put
    last = prev = None                                   # the last two accepted fixes: (t, x, y, sigma)
    lost = False
    for x in found:
        source = _source(x, out_dir)
        if x.accepted:
            fix = x.fix
            sigma = max(1.0, float(fix.reproj_px))
            vx = vy = 0.0
            if last is not None and x.t > last[0]:
                vx, vy = (fix.x - last[1]) / (x.t - last[0]), (fix.y - last[2]) / (x.t - last[0])
            state = TrackMode.RELOCALIZING if lost else TrackMode.TRACKING
            prev, last = last, (x.t, fix.x, fix.y, sigma)
            lost = False
            px, py = fix.x, fix.y
        else:
            state = TrackMode.LOST
            if last is None:                             # before the first fix: its position, not a lost track
                px, py, vx, vy, sigma = anchor.x, anchor.y, 0.0, 0.0, 1e3
            else:
                lost = True
                dt = x.t - last[0]
                vx, vy = _velocity(last, prev)
                coast = min(dt, COAST_MAX_S)
                px, py = last[1] + vx * coast, last[2] + vy * coast
                if dt >= COAST_MAX_S:                    # holding: no longer moving in the estimate
                    vx = vy = 0.0
                sigma = last[3] + LOST_SIGMA_GROWTH_PX_PER_S * dt
        extra = _extra(x)
        track = TrackState(idx=x.idx, state=state, x=px, y=py, vx=vx, vy=vy, cov=np.eye(2) * sigma ** 2, fix=x.fix)
        yield Record(idx=x.idx, t=x.t, source=source, track=track, backend=backend,
                     ms=round(x.ms, 2) if math.isfinite(x.ms) else None, extra=extra)


def _tracked_records(found: list[_Found], out_dir: Path, backend: str):
    """Records from the tracker's results (tracker="kalman")."""
    anchor = next(x.fix for x in found if x.accepted)    # frames before the first fix: LOST at its position
    for x in found:
        up = x.tracked
        track = up.track
        if track is None:
            track = TrackState(idx=x.idx, state=up.mode, x=anchor.x, y=anchor.y, vx=0.0, vy=0.0,
                               cov=np.eye(2) * 1e6, fix=x.fix)
        yield Record(idx=x.idx, t=x.t, source=_source(x, out_dir), track=track, backend=backend,
                     ms=round(x.ms, 2) if math.isfinite(x.ms) else None,
                     extra=_extra(x) | {"step": up.step.kind.value,
                                        "predicted_window": None if x.predicted is None else list(x.predicted),
                                        "window_sigma_px": None if x.sigma is None else round(x.sigma, 2)})


def _source(x: _Found, out_dir: Path) -> str:
    return Path(os.path.relpath(x.path.resolve(), out_dir.resolve())).as_posix()


def _extra(x: _Found) -> dict:
    return {"search": x.mode, "factor": x.factor, "window": None if x.window is None else [int(v) for v in x.window]}


def _velocity(last, prev) -> tuple[float, float]:
    """Map px per second between the last two accepted fixes (zero with only one)."""
    if prev is None or last[0] <= prev[0]:
        return 0.0, 0.0
    dt = last[0] - prev[0]
    return (last[1] - prev[1]) / dt, (last[2] - prev[2]) / dt
