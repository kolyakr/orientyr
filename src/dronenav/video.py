"""Reading a video take as frames (M2-02): the phone or webcam film of the advanced level.

:class:`VideoSource` yields every N-th frame of an MP4/MOV as the same :class:`~dronenav.types.Frame` the folder
source gives, so the pipeline does not care where its frames come from. What a real phone file needs:

- **Frame numbers are the video's own**: ``Frame.idx`` is the frame's index in the file (0, 10, 20 ... at N = 10),
  as an Irpin keyframe ``frame_00030`` is video frame 30, and ``Frame.source`` is ``<path>#<idx>``.
- **Real time**: phones film at a variable frame rate, so ``Frame.t`` is the container's own timestamp of the frame
  (seconds from the first frame), not ``idx / fps``. A file or backend without usable timestamps (missing, not
  finite, not increasing) falls back to ``idx / fps``; if they stop half way, time carries on from the last good
  one at the nominal frame rate, so it never runs back.
- **Upright frames**: a portrait take is stored as landscape pixels plus a rotation flag; OpenCV is asked to apply
  it, and ``size`` is measured on a decoded frame, never taken from the header.
- **The length is counted**: the header's frame count is an estimate in many files (variable rate, cut takes), so
  ``len()`` decodes the take once, without converting any picture, and remembers the answer.
- The take is decoded **sequentially**: seeking in H.264/HEVC is not frame-exact, skipped frames are only grabbed.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from dronenav.io import _is_rate
from dronenav.types import Frame

__all__ = ["VideoSource"]


def _is_count(v) -> bool:
    """A whole number >= 1; not a bool, not a float that happens to be whole (2.0 is a typo for a rate)."""
    return isinstance(v, numbers.Integral) and not isinstance(v, bool) and v >= 1


class VideoSource:
    """The frames of the video ``path``, every ``step``-th one, in time order (see the module docstring).

    every: keep every N-th frame, counted from frame 0 (N = 1: all of them, the default).
    fps: or the frames per second wanted: ``step = round_half_up(video fps / fps)``, the rule of ``FlightSource``
    (12 fps from a 30 fps take is every 3rd frame, 10 fps: see ``rate``). Not both; not above the take's own rate.
    max_side: shrink frames so the longer side is this many px, keeping the aspect; never enlarges.

    After construction: ``fps`` (the take's nominal rate), ``step``, ``rate`` (= fps / step, the rate really
    given), ``source_size`` and ``size`` ((w, h) of the decoded and of the yielded frames). Frames are decoded
    while iterating, one at a time; iterating again starts again from the first frame.
    """

    def __init__(self, path: str | Path, *, every: int | None = None, fps: float | None = None,
                 max_side: int | None = None):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"no video {self.path}")
        if every is not None and fps is not None:
            raise ValueError(f"give every (each N-th frame) or fps (a target rate), not both: got every={every!r}, "
                             f"fps={fps!r}")
        if every is not None and not _is_count(every):
            raise ValueError(f"every must be a whole number >= 1, got {every!r}")
        if fps is not None and not _is_rate(fps):
            raise ValueError(f"fps must be a positive, finite number, got {fps!r}")
        if max_side is not None and not _is_count(max_side):
            raise ValueError(f"max_side must be a whole number of px >= 1, got {max_side!r}")

        cap = self._open()
        try:
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            if not _is_rate(self.fps):
                raise ValueError(f"the video {self.path} does not say its frame rate (got {self.fps!r})")
            self.fps = float(self.fps)
            ok, first = cap.read()                    # the header's size ignores the rotation flag: measure a frame
            if not ok:
                raise ValueError(f"no frames in the video {self.path}")
        finally:
            cap.release()

        if fps is not None and fps > self.fps * (1 + 1e-6):   # a hair of slack: 29.97 asked of a 29.97003 take
            raise ValueError(f"fps must be at most the video's {self.fps:g}, got {fps!r}")
        # round half up: 2.5 -> 3, never to even (the same step FlightSource takes for the same numbers)
        self.step = int(every) if every is not None else (
            1 if fps is None else max(1, math.floor(self.fps / fps + 0.5)))
        self.rate = self.fps / self.step

        h, w = first.shape[:2]
        self.source_size = (w, h)
        self.size = _fit(w, h, None if max_side is None else int(max_side))
        self._n_video: int | None = None              # frames in the file, once counted

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise ValueError(f"cannot decode the video {self.path}")
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)     # the default with FFmpeg, but not something to leave to luck
        return cap

    def __len__(self) -> int:
        """The number of frames an iteration yields. Exact: the first call decodes the take once to count it."""
        if self._n_video is None:
            cap = self._open()
            try:
                n = 0
                while cap.grab():                     # decode only: no colour conversion, no copy
                    n += 1
            finally:
                cap.release()
            self._n_video = n
        return math.ceil(self._n_video / self.step)   # a last partial step still starts with a kept frame

    def __iter__(self) -> Iterator[Frame]:
        cap = self._open()
        clock = True                                  # the container's timestamps are usable so far
        ms0 = last_t = 0.0
        last_i = i = 0
        try:
            while True:
                if i % self.step:
                    if not cap.grab():                # decode, but skip the conversion
                        break
                    i += 1
                    continue
                ok, image = cap.read()
                if not ok:
                    break
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)   # the timestamp of the frame just decoded
                if i == 0:
                    # time counts from the first frame: a stream may start later than 0 in its container
                    clock, ms0, t = bool(math.isfinite(ms) and ms >= 0), ms, 0.0
                else:
                    t = (ms - ms0) / 1000.0 if clock else math.nan
                    if not (math.isfinite(t) and t > last_t):
                        # no timestamp, or one that does not move forward: from here on, the nominal rate
                        clock, t = False, last_t + (i - last_i) / self.fps
                last_i, last_t = i, t
                yield Frame(idx=i, t=t, image=self._sized(image), source=f"{self.path}#{i}")
                i += 1
            self._n_video = i                         # ran to the end: the count comes for free
        finally:
            cap.release()

    def _sized(self, image: np.ndarray) -> np.ndarray:
        if (image.shape[1], image.shape[0]) == self.size:
            return image
        # INTER_AREA averages the pixels it drops: no aliasing of fine map detail (or screen moiré) when shrinking
        return cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)


def _fit(w: int, h: int, max_side: int | None) -> tuple[int, int]:
    """(w, h) shrunk so the longer side is ``max_side``, the other rounded to the nearest pixel; never enlarged."""
    longer = max(w, h)
    if max_side is None or max_side >= longer:
        return w, h
    shorter = max(1, math.floor(min(w, h) * max_side / longer + 0.5))
    return (max_side, shorter) if w >= h else (shorter, max_side)
