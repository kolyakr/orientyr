"""FlightSource: a pre-rendered Irpin flight played as frames with ground truth (M6-01).

A flight folder (``data/irpin/flights/<preset>[_30mps]/<year>/``) holds ``flight.mp4`` (the camera at
``meta.fps``, with lens distortion), ``frames/`` (one keyframe per second), ``truth.jsonl`` (one record per video
frame ``i``: ``H_frame_to_map``, ``center_map_px``, ``footprint_map_px``, ``heading_deg``, ``altitude_m``,
``pitch_deg``, ``roll_deg``, ``speed_mps``) and ``meta.json`` (``fps``, ``camera.K``, ``camera.dist``, ...).

What the truth means, measured on the data (see the issue's spec update):

- ``H_frame_to_map`` acts on **undistorted** pixels (``cv2.undistort`` with K and dist, P = K), so frames are
  undistorted before they leave this source;
- it is in **zoom-17 map px**, and the maps of all years are aligned to within about 1 px, so a flight's truth
  applies to any year's ``maps/irpin_<year>.png``: the map year is chosen independently of the flight.

The video is decoded **sequentially**: H.264 seeking is not frame-exact, the truth is.
"""
from __future__ import annotations

import json
import math
import numbers
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dronenav.io import FolderSource
from dronenav.types import Frame

__all__ = ["FlightSource", "FlightTruth"]


@dataclass(frozen=True, eq=False)
class FlightTruth:
    """The true pose of video frame ``i``. Map px (zoom 17), degrees, metres, m/s."""

    i: int
    t: float
    H: np.ndarray                 # 3 x 3, undistorted frame px -> map px
    center: tuple[float, float]   # H applied to the principal point
    footprint: np.ndarray         # 4 x 2, TL TR BR BL
    heading_deg: float
    altitude_m: float
    pitch_deg: float
    roll_deg: float
    speed_mps: float

    def extra(self) -> dict:
        """JSON-safe values for ``Record.extra["truth"]`` (the renderers and the benchmark read x, y)."""
        return {"x": float(self.center[0]), "y": float(self.center[1]), "heading_deg": float(self.heading_deg),
                "altitude_m": float(self.altitude_m), "pitch_deg": float(self.pitch_deg),
                "roll_deg": float(self.roll_deg), "speed_mps": float(self.speed_mps)}


class FlightSource:
    """The frames of one flight, undistorted, with their truth.

    fps: frames per second to keep: every ``round(meta.fps / fps)``-th video frame (1 = the keyframe rate).
    keyframes: read ``frames/`` instead of decoding the video (same selection; only the keyframes present).
    undistort: remove the lens distortion, so ``truth(i).H`` applies to the frame (False: the raw camera image).
    start_s, end_s: only frames with ``start_s <= t <= end_s`` (decoding stops at ``end_s``).
    """

    def __init__(self, flight_dir: str | Path, *, fps: float = 1.0, keyframes: bool = False, undistort: bool = True,
                 start_s: float = 0.0, end_s: float | None = None):
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
        self.video_fps = float(self.meta["fps"])
        if not _is_rate(fps) or fps > self.video_fps:
            raise ValueError(f"fps must be a positive number up to the flight's {self.video_fps:g}, got {fps!r}")
        if not (_is_rate(start_s) or start_s == 0) or (end_s is not None and not end_s > start_s):
            raise ValueError(f"start_s must be >= 0 and below end_s, got start_s={start_s!r}, end_s={end_s!r}")
        self.step = max(1, round(self.video_fps / fps))
        self.keyframes, self.undistort = keyframes, undistort
        self.start_s, self.end_s = float(start_s), (None if end_s is None else float(end_s))
        self._truth = {}
        for line in truth_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                self._truth[int(rec["i"])] = rec
        cam = self.meta["camera"]
        self.K = np.asarray(cam["K"], float)
        self.dist = np.asarray(cam["dist"], float)
        w, h = cam["size"]
        # P = K: the undistorted image keeps the camera matrix the truth homographies were made with
        self._maps = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, (w, h), cv2.CV_32FC1)

    def _wanted(self, i: int) -> bool:
        t = i / self.video_fps
        return i % self.step == 0 and t >= self.start_s - 1e-9 and (self.end_s is None or t <= self.end_s + 1e-9)

    def _prepare(self, image: np.ndarray) -> np.ndarray:
        return cv2.remap(image, *self._maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) \
            if self.undistort else image

    def __iter__(self) -> Iterator[Frame]:
        if self.keyframes:
            for f in FolderSource(self.dir / "frames", fps=self.video_fps):
                if self._wanted(f.idx):
                    yield Frame(idx=f.idx, t=f.t, image=self._prepare(f.image), source=f.source)
            return
        cap = cv2.VideoCapture(str(self.video))
        if not cap.isOpened():
            raise ValueError(f"cannot decode {self.video}")
        last = math.inf if self.end_s is None else self.end_s * self.video_fps + 1e-6
        i = 0
        try:
            while i <= last:
                ok, image = cap.read()                     # every frame in turn: seeking is not frame-exact
                if not ok:
                    break
                if self._wanted(i):
                    yield Frame(idx=i, t=i / self.video_fps, image=self._prepare(image), source=f"{self.video}#{i}")
                i += 1
        finally:
            cap.release()

    def truth(self, i: int) -> FlightTruth:
        """The true pose of video frame ``i``; KeyError if the flight has no record for it."""
        try:
            r = self._truth[int(i)]
        except KeyError:
            raise KeyError(f"frame {i} has no truth record in {self.dir / 'truth.jsonl'}") from None
        return FlightTruth(i=int(r["i"]), t=float(r["t"]), H=np.asarray(r["H_frame_to_map"], float),
                           center=tuple(map(float, r["center_map_px"])),
                           footprint=np.asarray(r["footprint_map_px"], float), heading_deg=float(r["heading_deg"]),
                           altitude_m=float(r["altitude_m"]), pitch_deg=float(r.get("pitch_deg", 0.0)),
                           roll_deg=float(r.get("roll_deg", 0.0)), speed_mps=float(r.get("speed_mps", 0.0)))


def _is_rate(v) -> bool:
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        return False
    return math.isfinite(v) and v > 0
