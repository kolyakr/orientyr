"""The pipeline (M1-09): a map and a folder of frames in, a run folder with the trajectory image and video out.

    frames (io.FolderSource) ─► Searcher.locate per frame ─► records ─► run.json + run.jsonl ─► image, video
                                (matcher, scale, geometry,      (states,      (telemetry)       (viz)
                                 window or whole map)            coasting)

The engine (matcher, scale rule, geometry checks, search strategy) is M1-02 to M1-05; this module wires it to the
run folder the renderers read (M0-03, M0-04 convention) and gives every frame a state until the Kalman filter
(M5) takes over:

- TRACKING: an accepted fix; RELOCALIZING: the first accepted fix after a miss (the track was lost and found);
- LOST: no accepted fix. The position coasts on the last measured step from the last fix; misses before the
  first fix take that fix's position (there is no "position unknown" yet, see #87). Proper interpolation and
  smoothing are M1-06's.

A run in which no frame is localized raises :class:`NothingLocalized` instead of drawing an empty trajectory.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dronenav.io import FolderSource, load_map
from dronenav.matching import RootSiftMatcher
from dronenav.search import Located, Searcher
from dronenav.telemetry import Record, RunInfo, RunWriter, write_run_info
from dronenav.types import TrackMode, TrackState
from dronenav.viz.trajectory_image import write_trajectory
from dronenav.viz.trajectory_video import write_video

__all__ = ["MATCHERS", "NothingLocalized", "RunSummary", "localize_run", "run"]

# --matcher name -> backend; the learned matchers (M4) join here
MATCHERS = {"sift": lambda map_img, cache_dir: RootSiftMatcher(map_img, cache_dir=cache_dir)}

LOST_SIGMA_GROWTH_PX = 20.0     # how much less sure the coasted position gets with every missed frame


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


def localize_run(map_path: str | Path, frames_dir: str | Path, out_dir: str | Path, *, matcher: str = "sift",
                 fps: float | None = None, cache_dir: str | Path | None = "outputs/cache") -> RunSummary:
    """Find every frame on the map and write ``run.json`` and ``run.jsonl`` into ``out_dir``.

    Everything is checked (matcher name, map, frames) and localized before ``out_dir`` is created, so a bad
    input or a run with nothing found leaves no folder behind.
    """
    t0 = time.perf_counter()
    if matcher not in MATCHERS:
        raise ValueError(f"unknown matcher {matcher!r} (available: {', '.join(sorted(MATCHERS))})")
    out_dir = Path(out_dir)
    the_map = load_map(map_path)
    frames = FolderSource(frames_dir, fps=fps)
    searcher = Searcher(MATCHERS[matcher](the_map.image, cache_dir), the_map.image)
    found: list[tuple[int, float, Path, tuple[int, int], Located]] = []
    for f in frames:
        found.append((f.idx, f.t, Path(f.source), (f.image.shape[1], f.image.shape[0]), searcher.locate(f.idx, f.image)))
    localized = sum(1 for *_, loc in found if _accepted(loc))
    if localized == 0:
        raise NothingLocalized(f"no frame of {frames_dir} could be localized on {map_path} "
                               f"({len(found)} frames tried): is it the right map?")

    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = the_map.image.shape[:2]
    write_run_info(out_dir, RunInfo(map=str(map_path), map_size=(w, h), generator="run.py",
                                    frame_size=found[0][3], fps=frames.fps))
    with RunWriter(out_dir / "run.jsonl") as writer:
        for rec in _records(found, out_dir, MATCHERS_BACKEND.get(matcher, matcher)):
            writer.write(rec)
    return RunSummary(out_dir=out_dir, frames=len(found), localized=localized, seconds=time.perf_counter() - t0)


MATCHERS_BACKEND = {"sift": RootSiftMatcher.backend}


def run(map_path: str | Path, frames_dir: str | Path, out_dir: str | Path, *, matcher: str = "sift",
        fps: float | None = None, speed: float = 1.0,
        cache_dir: str | Path | None = "outputs/cache") -> RunSummary:
    """The whole deliverable: localize, then draw the trajectory image (M1-07) and the video (M1-08)."""
    t0 = time.perf_counter()
    s = localize_run(map_path, frames_dir, out_dir, matcher=matcher, fps=fps, cache_dir=cache_dir)
    map_img = load_map(map_path).image
    image = write_trajectory(out_dir, map_img=map_img).full
    video = write_video(out_dir, map_img=map_img, speed=speed)
    return RunSummary(out_dir=s.out_dir, frames=s.frames, localized=s.localized,
                      seconds=time.perf_counter() - t0, image=image, video=video)


def _accepted(loc: Located) -> bool:
    return loc.fix is not None and loc.fix.accepted


def _records(found, out_dir: Path, backend: str):
    """One Record per frame, with the states and coasting described in the module docstring."""
    first = next(k for k, (*_, loc) in enumerate(found) if _accepted(loc))
    anchor = found[first][4].fix                      # where misses before the first fix are put
    last = prev = None                                # the last two accepted fixes: (idx, t, x, y, sigma)
    lost_since_fix = False
    for k, (idx, t, path, _, loc) in enumerate(found):
        source = Path(os.path.relpath(path.resolve(), out_dir.resolve())).as_posix()
        if _accepted(loc):
            fix = loc.fix
            sigma = max(1.0, float(fix.reproj_px))
            vx = vy = 0.0
            if last is not None and t > last[1]:
                vx, vy = (fix.x - last[2]) / (t - last[1]), (fix.y - last[3]) / (t - last[1])
            state = TrackMode.RELOCALIZING if lost_since_fix and last is not None else TrackMode.TRACKING
            prev, last = last, (idx, t, fix.x, fix.y, sigma)
            lost_since_fix = False
            x, y = fix.x, fix.y
        else:
            state = TrackMode.LOST
            lost_since_fix = True
            if last is None:                          # before the first fix: its position, not yet moving
                x, y, vx, vy, sigma = anchor.x, anchor.y, 0.0, 0.0, 1e3
            else:
                vx, vy = _velocity(last, prev)
                x, y = last[2] + vx * (t - last[1]), last[3] + vy * (t - last[1])
                sigma = last[4] + LOST_SIGMA_GROWTH_PX * (idx - last[0])
        extra = {"search": loc.mode.value, "factor": float(loc.factor),
                 "window": None if loc.window is None else [int(v) for v in loc.window]}
        track = TrackState(idx=idx, state=state, x=x, y=y, vx=vx, vy=vy, cov=np.eye(2) * sigma ** 2, fix=loc.fix)
        yield Record(idx=idx, t=t, source=source, track=track, backend=backend, ms=round(float(loc.ms), 2),
                     extra=extra)


def _velocity(last, prev) -> tuple[float, float]:
    """Map px per second between the last two accepted fixes (zero with only one)."""
    if prev is None or last[1] <= prev[1]:
        return 0.0, 0.0
    dt = last[1] - prev[1]
    return (last[2] - prev[2]) / dt, (last[3] - prev[3]) / dt
