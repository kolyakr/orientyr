"""The pipeline (M1-09): a map and a folder of frames in, a run folder with the trajectory image and video out.

    frames (io.FolderSource) ─► Searcher.locate per frame ─► records ─► run.json + run.jsonl ─► image, video
                                (matcher, scale, geometry,      (states,      (telemetry)       (viz)
                                 window or whole map)            coasting)

The engine (matcher, scale rule, geometry checks, search strategy) is M1-02 to M1-05; this module wires it to the
run folder the renderers read (M0-03, M0-04 convention) and gives every frame a state until the Kalman filter
(M5) takes over:

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
from dronenav.search import Located, Searcher
from dronenav.telemetry import Record, RunInfo, RunWriter, write_run_info
from dronenav.types import Fix, TrackMode, TrackState
from dronenav.viz.trajectory_image import write_trajectory
from dronenav.viz.trajectory_video import write_video

__all__ = ["COAST_MAX_S", "LOST_SIGMA_GROWTH_PX_PER_S", "MATCHERS", "OUTPUTS", "NothingLocalized", "RunSummary",
           "localize_run", "run"]

# --matcher name -> (backend, factory); the learned matchers (M4) join here
MATCHERS = {"sift": (RootSiftMatcher.backend, lambda map_img, cache_dir: RootSiftMatcher(map_img, cache_dir=cache_dir))}
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

    @property
    def accepted(self) -> bool:
        return self.fix is not None and self.fix.accepted


def _check_out(out_dir: Path) -> None:
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"--out {out_dir} exists and is not a folder")


def _localize(map_path, frames_dir, matcher: str, fps, cache_dir):
    """Check the inputs and find every frame; nothing is written. Returns (map image, frames source, found)."""
    if matcher not in MATCHERS:
        raise ValueError(f"unknown matcher {matcher!r} (available: {', '.join(sorted(MATCHERS))})")
    the_map = load_map(map_path)
    frames = FolderSource(frames_dir, fps=fps)
    try:
        searcher = Searcher(MATCHERS[matcher][1](the_map.image, cache_dir), the_map.image)
        found = []
        for f in frames:
            loc: Located = searcher.locate(f.idx, f.image)
            found.append(_Found(f.idx, f.t, Path(f.source), (f.image.shape[1], f.image.shape[0]), loc.fix,
                                loc.mode.value, loc.window, float(loc.factor), float(loc.ms)))
    except cv2.error as e:                # FLANN refuses an empty index: a blank or flat map
        raise ValueError(f"matching on {map_path} failed: the map has no usable features (blank or flat image?) "
                         f"[{e.err}]") from None
    if not any(x.accepted for x in found):
        raise NothingLocalized(f"no frame of {frames_dir} could be localized on {map_path} "
                               f"({len(found)} frames tried): is it the right map?")
    return the_map.image, frames, found


def _write_run(out_dir: Path, map_path, map_img, frames: FolderSource, found, backend: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = map_img.shape[:2]
    sizes = {x.size for x in found}
    write_run_info(out_dir, RunInfo(map=str(map_path), map_size=(w, h), generator="run.py",
                                    frame_size=sizes.pop() if len(sizes) == 1 else None, fps=frames.fps))
    with RunWriter(out_dir / "run.jsonl") as writer:
        for rec in _records(found, out_dir, backend):
            writer.write(rec)


def localize_run(map_path: str | Path, frames_dir: str | Path, out_dir: str | Path, *, matcher: str = "sift",
                 fps: float | None = None, cache_dir: str | Path | None = "outputs/cache") -> RunSummary:
    """Find every frame on the map and write ``run.json`` and ``run.jsonl`` into ``out_dir`` (no images)."""
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    _check_out(out_dir)
    map_img, frames, found = _localize(map_path, frames_dir, matcher, fps, cache_dir)
    _write_run(out_dir, map_path, map_img, frames, found, MATCHERS[matcher][0])
    return RunSummary(out_dir=out_dir, frames=len(found), localized=sum(x.accepted for x in found),
                      seconds=time.perf_counter() - t0)


def run(map_path: str | Path, frames_dir: str | Path, out_dir: str | Path, *, matcher: str = "sift",
        fps: float | None = None, speed: float = 1.0,
        cache_dir: str | Path | None = "outputs/cache") -> RunSummary:
    """The whole deliverable: localize, then the trajectory image (M1-07) and the video (M1-08), all into
    ``out_dir`` at once (see the module docstring)."""
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    if not _is_rate(speed):
        raise ValueError(f"speed must be a positive number, got {speed!r}")
    _check_out(out_dir)
    map_img, frames, found = _localize(map_path, frames_dir, matcher, fps, cache_dir)
    # a sibling folder: the same depth, so every relative path written into it holds in out_dir too
    part = out_dir.parent / f".{out_dir.name}.partial"
    shutil.rmtree(part, ignore_errors=True)
    try:
        _write_run(part, map_path, map_img, frames, found, MATCHERS[matcher][0])
        write_trajectory(part, map_img=map_img)
        write_video(part, map_img=map_img, speed=speed)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in OUTPUTS:                               # all five exist: now they replace the old ones
            os.replace(part / name, out_dir / name)
    finally:
        shutil.rmtree(part, ignore_errors=True)
    return RunSummary(out_dir=out_dir, frames=len(found), localized=sum(x.accepted for x in found),
                      seconds=time.perf_counter() - t0, image=out_dir / "trajectory.png",
                      video=out_dir / "trajectory.mp4")


def _records(found: list[_Found], out_dir: Path, backend: str):
    """One Record per frame, with the states, coasting and uncertainty described in the module docstring."""
    anchor = next(x.fix for x in found if x.accepted)    # where misses before the first fix are put
    last = prev = None                                   # the last two accepted fixes: (t, x, y, sigma)
    lost = False
    for x in found:
        source = Path(os.path.relpath(x.path.resolve(), out_dir.resolve())).as_posix()
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
        extra = {"search": x.mode, "factor": x.factor,
                 "window": None if x.window is None else [int(v) for v in x.window]}
        track = TrackState(idx=x.idx, state=state, x=px, y=py, vx=vx, vy=vy, cov=np.eye(2) * sigma ** 2, fix=x.fix)
        yield Record(idx=x.idx, t=x.t, source=source, track=track, backend=backend,
                     ms=round(x.ms, 2) if math.isfinite(x.ms) else None, extra=extra)


def _velocity(last, prev) -> tuple[float, float]:
    """Map px per second between the last two accepted fixes (zero with only one)."""
    if prev is None or last[0] <= prev[0]:
        return 0.0, 0.0
    dt = last[0] - prev[0]
    return (last[1] - prev[1]) / dt, (last[2] - prev[2]) / dt
