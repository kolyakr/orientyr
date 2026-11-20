"""FlightSource: a pre-rendered Irpin flight played as frames with ground truth (M6-01).

A flight folder (``data/irpin/flights/<preset>[_30mps]/<year>/``) holds ``flight.mp4`` (the camera at
``meta.fps``, with lens distortion), ``frames/`` (one keyframe per second), ``truth.jsonl`` (one record per video
frame ``i``: ``H_frame_to_map``, ``center_map_px``, ``footprint_map_px``, ``heading_deg``, ``altitude_m``,
``pitch_deg``, ``roll_deg``, ``speed_mps``) and ``meta.json`` (``fps``, ``frames``, ``camera.K/dist/size``, ...).

What the truth means, measured on the data (see the issue's spec update):

- ``H_frame_to_map`` acts on **undistorted** pixels (``cv2.undistort`` with K and dist, P = K), so frames are
  undistorted before they leave this source;
- it is in **zoom-17 map px**, and the maps of all years are aligned to within about 1 px, so a flight's truth
  applies to any year's ``maps/irpin_<year>.png``: the map year is chosen independently of the flight.

The video is decoded **sequentially** (H.264 seeking is not frame-exact; skipped frames are only grabbed), and a
video that ends before its truth does is an error, not a shorter flight.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from dronenav.io import FolderSource, _is_rate
from dronenav.sim.truth import FlightTruth, centre_scale
from dronenav.types import Frame

__all__ = ["FlightSource", "FlightTruth"]

_TRUTH_KEYS = ("i", "t", "H_frame_to_map", "center_map_px", "footprint_map_px", "heading_deg", "altitude_m",
               "pitch_deg", "roll_deg", "speed_mps")


class FlightSource:
    """The frames of one flight, undistorted, with their truth.

    fps: frames per second wanted: every ``step = round_half_up(meta.fps / fps)``-th video frame, counted from
    frame 0 (``self.rate`` is the rate that gives, e.g. 15 fps for 12 on a 30 fps video).
    keyframes: read ``frames/`` instead of decoding the video (same selection; only the keyframes present).
    undistort: remove the lens distortion, so ``truth(i).H`` applies to the frame (False: the raw camera image).
    start_s, end_s: only frames with ``start_s <= t <= end_s`` (decoding stops at ``end_s``).
    priors: fill ``Frame.heading_prior_deg`` and ``Frame.scale_prior`` from the truth (a perfect compass and
    altimeter, for the chaos layer to degrade).
    """

    def __init__(self, flight_dir: str | Path, *, fps: float = 1.0, keyframes: bool = False, undistort: bool = True,
                 start_s: float = 0.0, end_s: float | None = None, priors: bool = False):
        self.dir = Path(flight_dir)
        meta_path, truth_path = self.dir / "meta.json", self.dir / "truth.jsonl"
        for p in (meta_path, truth_path):
            if not p.is_file():
                raise FileNotFoundError(f"no {p.name} in the flight {self.dir}")
        self.video = self.dir / "flight.mp4"
        if keyframes and not (self.dir / "frames").is_dir():
            raise FileNotFoundError(f"no frames/ folder in the flight {self.dir}")
        if not keyframes and not self.video.is_file():
            raise FileNotFoundError(f"no flight.mp4 in the flight {self.dir}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.video_fps, self.K, self.dist, self.size = _read_meta(self.meta, meta_path)

        if not _is_rate(fps) or fps > self.video_fps:
            raise ValueError(f"fps must be a positive number up to the flight's {self.video_fps:g}, got {fps!r}")
        if not _is_time(start_s):
            raise ValueError(f"start_s must be a time >= 0 in seconds, got {start_s!r}")
        if end_s is not None and not (_is_time(end_s) and end_s >= start_s):
            raise ValueError(f"end_s must be a time >= start_s ({start_s!r}), got {end_s!r}")
        self.step = max(1, math.floor(self.video_fps / fps + 0.5))   # round half up: 2.5 -> 3, never to even
        self.rate = self.video_fps / self.step
        self.keyframes, self.undistort, self.priors = keyframes, undistort, priors
        self.start_s, self.end_s = float(start_s), (None if end_s is None else float(end_s))

        self._truth = _read_truth(truth_path)
        n = self.meta.get("frames")
        # the video needs a record for every frame; a keyframes-only folder (the CI fixture) may carry fewer
        if not keyframes and n is not None and n != len(self._truth):
            raise ValueError(f"{meta_path} says {n} frames but {truth_path} has {len(self._truth)} records")
        self.n_frames = len(self._truth)
        w, h = self.size
        # P = K: the undistorted image keeps the camera matrix the truth homographies were made with
        self._maps = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, (w, h), cv2.CV_32FC1)

    # -- selection
    def _wanted(self, i: int) -> bool:
        t = i / self.video_fps
        return i % self.step == 0 and t >= self.start_s - 1e-9 and (self.end_s is None or t <= self.end_s + 1e-9)

    def _keyframes(self) -> list[tuple[int, Path]]:
        src = FolderSource(self.dir / "frames", fps=self.video_fps)       # order and numbers; nothing read yet
        return [(i, p) for i, p in zip(src.idxs, src.paths, strict=True) if self._wanted(i)]

    def __len__(self) -> int:
        if self.keyframes:
            return len(self._keyframes())
        return sum(1 for i in self._truth if self._wanted(i))

    # -- frames
    def _frame(self, i: int, image: np.ndarray, source: str) -> Frame:
        if (image.shape[1], image.shape[0]) != tuple(self.size):
            raise ValueError(f"{source}: frames are {image.shape[1]} x {image.shape[0]} px but meta.json's "
                             f"camera.size says {self.size[0]} x {self.size[1]}")
        if self.undistort:
            image = cv2.remap(image, *self._maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        heading = scale = None
        if self.priors:
            tr = self.truth(i)
            heading, scale = tr.heading_deg, centre_scale(tr.H, self.K[0, 2], self.K[1, 2])
        return Frame(idx=i, t=i / self.video_fps, image=image, source=source, heading_prior_deg=heading,
                     scale_prior=scale)

    def __iter__(self) -> Iterator[Frame]:
        if self.keyframes:
            for i, path in self._keyframes():
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)            # only the frames inside the window
                if image is None:
                    raise ValueError(f"cannot read the keyframe {path}")
                yield self._frame(i, image, str(path))
            return
        cap = cv2.VideoCapture(str(self.video))
        if not cap.isOpened():
            raise ValueError(f"cannot decode {self.video}")
        last = self.n_frames - 1 if self.end_s is None else min(self.n_frames - 1,
                                                                math.floor(self.end_s * self.video_fps + 1e-6))
        i = 0
        try:
            while i <= last:
                if not self._wanted(i):
                    if not cap.grab():                                      # decode, but skip the conversion
                        break
                else:
                    ok, image = cap.read()
                    if not ok:
                        break
                    yield self._frame(i, image, f"{self.video}#{i}")
                i += 1
        finally:
            cap.release()
        if i <= last:
            raise ValueError(f"{self.video} ended after {i} frames, but meta.json and truth.jsonl describe "
                             f"{self.n_frames}")

    def truth(self, i: int) -> FlightTruth:
        """The true pose of video frame ``i``; KeyError if the flight has no record for it."""
        try:
            return self._truth[int(i)]
        except KeyError:
            raise KeyError(f"frame {i} has no truth record in {self.dir / 'truth.jsonl'}") from None


def _is_time(v) -> bool:
    return _is_rate(v) or (not isinstance(v, bool) and isinstance(v, (int, float)) and v == 0)


def _read_meta(meta: dict, path: Path):
    try:
        fps, cam = meta["fps"], meta["camera"]
        K, dist, size = cam["K"], cam["dist"], cam["size"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"{path} lacks {e} (needs fps and camera.K, camera.dist, camera.size)") from None
    if not _is_rate(fps):
        raise ValueError(f"{path}: fps must be a positive number, got {fps!r}")
    K, dist = np.asarray(K, float), np.asarray(dist, float)
    if K.shape != (3, 3) or dist.ndim != 1 or len(size) != 2:
        raise ValueError(f"{path}: camera.K must be 3x3, camera.dist a list, camera.size [w, h]")
    return float(fps), K, dist, (int(size[0]), int(size[1]))


def _read_truth(path: Path) -> dict[int, FlightTruth]:
    """Every record, checked: all keys present, finite, each frame once."""
    out: dict[int, FlightTruth] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path} line {n}"
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{where}: invalid JSON: {e.msg}") from None
        missing = [k for k in _TRUTH_KEYS if k not in r]
        if missing:
            raise ValueError(f"{where}: missing {', '.join(missing)}")
        i = int(r["i"])
        if i in out:
            raise ValueError(f"{where}: frame {i} appears twice")
        try:
            out[i] = FlightTruth(i=i, t=float(r["t"]), H=r["H_frame_to_map"], center=r["center_map_px"],
                                 footprint=r["footprint_map_px"], heading_deg=float(r["heading_deg"]),
                                 altitude_m=float(r["altitude_m"]), pitch_deg=float(r["pitch_deg"]),
                                 roll_deg=float(r["roll_deg"]), speed_mps=float(r["speed_mps"]))
        except (ValueError, TypeError) as e:
            raise ValueError(f"{where}: {e}") from None
    return out
