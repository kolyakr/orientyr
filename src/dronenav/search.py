"""Search strategy: where on the map to look for each frame (M1-05).

A drone can't teleport: once a frame is found, the next one is near the last position plus the last step.
So after a fix, the frame is matched only against a window of the map. The window is the last footprint moved
by the observed step, plus a margin for turns and speed changes. That is faster than the whole map, and it
leaves far-away look-alikes out of reach.

Modes, per frame:
- ``GLOBAL``: no usable track (the first frame, after a miss, after ``reset``): search the whole map;
- ``WINDOW``: the window around the prediction gave an accepted fix;
- ``FALLBACK``: the window failed (a jump, a bad prediction, a frame with nothing on it), so the whole map
  was searched in the same frame. If that fails too, the frame is a miss and the next one starts global.

The window is cut from the registered map, so matchers that cache map features (RootSIFT) reuse them and never
re-extract. Every fix is in full-map pixels. The frame is shrunk by the scale rule (dronenav.scale) in every mode.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np

from dronenav.geometry import GeometryConfig
from dronenav.scale import ScaleConfig, ScaledLocalization, ScaleTracker, localize_scaled
from dronenav.types import Fix, Matcher, Matches

_GEOM = GeometryConfig()   # the default checks, shared (it is frozen)
_SCALE = ScaleConfig()


@dataclass(frozen=True)
class SearchConfig:
    """margin_frac: margin around the predicted footprint, as a share of its larger side. It absorbs turns
    and speed changes: on the Irpin keyframes the step changes by up to ~100 px against a ~650 px footprint.
    min_margin_px: the margin's floor, for small footprints. min_window_px: a window clipped by the map to less
    than this on a side is useless (the prediction left the map), so the search goes global."""

    margin_frac: float = 0.25
    min_margin_px: float = 96.0
    min_window_px: int = 32

    def __post_init__(self):
        if self.margin_frac < 0:
            raise ValueError(f"margin_frac must be >= 0, got {self.margin_frac}")
        if self.min_margin_px < 0:
            raise ValueError(f"min_margin_px must be >= 0, got {self.min_margin_px}")
        if self.min_window_px < 1:
            raise ValueError(f"min_window_px must be >= 1, got {self.min_window_px}")


def search_window(fix: Fix, step: tuple[float, float], map_size: tuple[int, int],
                  cfg: SearchConfig = SearchConfig(), frames_ahead: int = 1) -> tuple[int, int, int, int] | None:
    """The map window ``(x0, y0, x1, y1)`` (whole px, x1/y1 exclusive) where the frame ``frames_ahead`` after
    ``fix`` should be: its footprint moved by ``frames_ahead`` steps, plus the margin, clipped to the map.
    None when that window is (almost) entirely off the map."""
    fp = np.asarray(fix.footprint, float) + np.asarray(step, float) * frames_ahead
    (bx0, by0), (bx1, by1) = fp.min(axis=0), fp.max(axis=0)
    margin = max(cfg.min_margin_px, cfg.margin_frac * max(bx1 - bx0, by1 - by0))
    w, h = map_size
    x0, y0 = max(0, math.floor(bx0 - margin)), max(0, math.floor(by0 - margin))
    x1, y1 = min(w, math.ceil(bx1 + margin)), min(h, math.ceil(by1 + margin))
    if x1 - x0 < cfg.min_window_px or y1 - y0 < cfg.min_window_px:
        return None
    return int(x0), int(y0), int(x1), int(y1)


class SearchMode(str, Enum):
    GLOBAL = "global"
    WINDOW = "window"
    FALLBACK = "fallback"


class Located(NamedTuple):
    """One frame's search result. fix: the best fix (accepted, rejected, or None: no homography).
    mode: how it was found. window: the map window the fix came from, None for a whole-map search.
    factor: the frame's shrink factor. matches: the matches behind the fix. ms: wall time of all tries."""

    fix: Fix | None
    mode: SearchMode
    window: tuple[int, int, int, int] | None
    factor: float
    matches: Matches
    ms: float


def _accepted(res: ScaledLocalization) -> bool:
    return res.fix is not None and res.fix.accepted


class Searcher:
    """Finds each frame on the map: the whole map first, then windows around the prediction (see module doc).

    Feed frames in flight order through :meth:`locate`; it keeps the last fix, the step and the scale.
    """

    def __init__(self, matcher: Matcher, map_img: np.ndarray, cfg: SearchConfig = SearchConfig(),
                 scale: ScaleConfig = _SCALE, geom: GeometryConfig = _GEOM):
        self.matcher = matcher
        self.map_img = map_img
        self.map_size = (map_img.shape[1], map_img.shape[0])
        self.cfg = cfg
        self.geom = geom
        self.tracker = ScaleTracker(scale)
        self._last: Fix | None = None
        self._step = (0.0, 0.0)

    def reset(self) -> None:
        """Forget the track: the next frame searches the whole map and probes the scale again."""
        self._last, self._step = None, (0.0, 0.0)
        self.tracker.reset()

    def _localize(self, idx: int, frame: np.ndarray, window: tuple[int, int, int, int] | None):
        if window is None:
            return localize_scaled(idx, self.matcher, frame, self.map_img, self.tracker.factors(), geom=self.geom)
        x0, y0, x1, y1 = window
        return localize_scaled(idx, self.matcher, frame, self.map_img[y0:y1, x0:x1], self.tracker.factors(),
                               map_offset=(x0, y0), geom=self.geom)

    def locate(self, idx: int, frame_img: np.ndarray) -> Located:
        """Find frame ``idx``: in the window around the prediction if there is a track, else (or if the window
        fails) on the whole map."""
        t0 = time.perf_counter()
        mode, window, res = SearchMode.GLOBAL, None, None
        if self._last is not None:
            ahead = max(1, idx - self._last.idx)
            window = search_window(self._last, self._step, self.map_size, self.cfg, frames_ahead=ahead)
            if window is not None:
                res = self._localize(idx, frame_img, window)
            if res is not None and _accepted(res):
                mode = SearchMode.WINDOW
            else:
                mode, window = SearchMode.FALLBACK, None
        if mode != SearchMode.WINDOW:
            res = self._localize(idx, frame_img, None)
        self._update(res.fix)
        return Located(res.fix, mode, window, res.factor, res.matches, (time.perf_counter() - t0) * 1e3)

    def _update(self, fix: Fix | None) -> None:
        """An accepted fix moves the track on (and measures the step); anything else is a miss."""
        if fix is None or not fix.accepted:
            self._last, self._step = None, (0.0, 0.0)      # a miss: the next frame starts global
            return
        if self._last is not None and fix.idx > self._last.idx:
            n = fix.idx - self._last.idx
            self._step = ((fix.x - self._last.x) / n, (fix.y - self._last.y) / n)
        else:
            self._step = (0.0, 0.0)                         # first fix of a track: no step yet
        self._last = fix
        self.tracker.update(fix)
