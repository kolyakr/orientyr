"""A simulated GPS receiver: honest, or walked away by a spoofer (M5-07).

The spoof detector (``dronenav.tracking.spoof``) needs a GPS track to compare the filter with. A simulated flight
knows where the drone really is, so the receiver is made from the truth:

- **honest**: the true position plus independent Gaussian noise on each axis (``sigma_px``), seeded;
- **spoofed**: the honest track, and from frame ``WalkOff.start_idx`` on a displacement that grows at a constant
  speed in one direction. It starts at zero: a spoofer that wants to stay unnoticed takes over the receiver at the
  true position and then drags it away slowly, there is no jump to see.

Positions are map px (the package's convention). With the map's georef block (M0-05) each sample also carries the
lon/lat a real receiver would report; the detector converts them back to map px at its boundary.

The noise is white: each frame's error is independent. A real receiver's error also drifts slowly (atmosphere,
multipath), which this model leaves out; at one frame per second the detector's N-frames rule is what guards
against a short correlated excursion.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from dronenav.georef import GeorefBlock

__all__ = ["GpsSample", "TruthPoint", "WalkOff", "read_truth_track", "simulate_gps"]


class TruthPoint(NamedTuple):
    """Where the drone really is at one frame. idx: frame index. t: seconds. x, y: map px."""

    idx: int
    t: float
    x: float
    y: float


@dataclass(frozen=True)
class WalkOff:
    """A slow spoofing walk-off. start_idx: the first frame at or after this index is where it starts (offset 0).
    speed_px_s: how fast the reported position leaves the true one, map px per second.
    direction_deg: where to, in the heading convention of ``dronenav.types``: clockwise from map up, 90 = east."""

    start_idx: int
    speed_px_s: float
    direction_deg: float

    def __post_init__(self):
        if isinstance(self.start_idx, bool) or not isinstance(self.start_idx, (int, np.integer)) or self.start_idx < 0:
            raise ValueError(f"start_idx must be a frame index >= 0, got {self.start_idx!r}")
        if not (_finite(self.speed_px_s) and self.speed_px_s > 0):
            raise ValueError(f"speed_px_s must be a finite number > 0, got {self.speed_px_s!r}")
        if not _finite(self.direction_deg):
            raise ValueError(f"direction_deg must be a finite number, got {self.direction_deg!r}")

    @property
    def unit(self) -> tuple[float, float]:
        """The direction as a unit vector in map px (y grows down, so north is -y)."""
        a = math.radians(self.direction_deg)
        return math.sin(a), -math.cos(a)


@dataclass(frozen=True)
class GpsSample:
    """One GPS reading. idx, t: the frame it belongs to. x, y: the reported position, map px.
    lon, lat: the same position in WGS84, None when the map has no georef.
    spoofed: the walk-off is active at this frame (simulator truth, not known to the detector).
    offset_px: how far the walk-off has moved the reading by now, map px (0 for an honest reading)."""

    idx: int
    t: float
    x: float
    y: float
    lon: float | None = None
    lat: float | None = None
    spoofed: bool = False
    offset_px: float = 0.0


def _finite(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and math.isfinite(v)


def _check_track(truth: Sequence[TruthPoint]) -> None:
    if len(truth) == 0:
        raise ValueError("the truth track is empty")
    prev = None
    for p in truth:
        if not all(_finite(v) for v in (p.t, p.x, p.y)):
            raise ValueError(f"frame {p.idx}: t, x and y must be finite, got {(p.t, p.x, p.y)!r}")
        if prev is not None:
            if p.idx <= prev.idx:
                raise ValueError(f"idx {p.idx} after idx {prev.idx}: the truth must be in frame order")
            if p.t < prev.t:
                raise ValueError(f"t {p.t:g} after t {prev.t:g} (frame {p.idx}): time must not go back")
        prev = p


def simulate_gps(truth: Sequence[TruthPoint], *, sigma_px: float, seed: int, walk_off: WalkOff | None = None,
                 block: GeorefBlock | None = None) -> list[GpsSample]:
    """The GPS track of a flight: one sample per truth point.

    sigma_px: 1-σ noise per axis, map px (0 = a perfect receiver). seed: of the noise; the same seed gives the same
    noise with and without a walk-off, so a spoofed track is exactly its honest twin plus the displacement.
    walk_off: None for an honest track. block: the map's georef block, to fill lon/lat.
    Raises ValueError for a bad track or setting, or a walk-off that would start after the flight has ended.
    """
    if not (_finite(sigma_px) and sigma_px >= 0):
        raise ValueError(f"sigma_px must be a finite number >= 0, got {sigma_px!r}")
    _check_track(truth)
    t0 = None
    if walk_off is not None:
        t0 = next((p.t for p in truth if p.idx >= walk_off.start_idx), None)
        if t0 is None:
            raise ValueError(f"the walk-off's start_idx {walk_off.start_idx} is after the last frame "
                             f"({truth[-1].idx}): it would never start")
    # all the noise in one draw, before anything depends on the walk-off: the honest twin gets the same numbers
    noise = np.random.default_rng(seed).normal(0.0, 1.0, (len(truth), 2)) * sigma_px
    out = []
    for p, (nx, ny) in zip(truth, noise, strict=True):
        x, y, spoofed, offset = p.x + float(nx), p.y + float(ny), False, 0.0
        if walk_off is not None and p.idx >= walk_off.start_idx:
            spoofed, offset = True, walk_off.speed_px_s * (p.t - t0)
            ux, uy = walk_off.unit
            x, y = x + offset * ux, y + offset * uy
        lon, lat = block.px_to_lonlat(x, y) if block is not None else (None, None)
        out.append(GpsSample(idx=int(p.idx), t=float(p.t), x=x, y=y, lon=lon, lat=lat, spoofed=spoofed,
                             offset_px=offset))
    return out


def read_truth_track(path: str | Path, *, every: int = 1) -> list[TruthPoint]:
    """A flight's true track from its ``truth.jsonl`` (or the flight folder): ``i``, ``t`` and ``center_map_px``
    of every ``every``-th record (30 on a 30 fps flight: one per second, the keyframes the engine localizes).

    Only these three fields are needed, so a file made for a test may leave the others out.
    Raises FileNotFoundError for a missing file and ValueError naming the line for a malformed one.
    """
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError(f"every must be an integer >= 1, got {every!r}")
    path = Path(path)
    if path.is_dir():
        path = path / "truth.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"no truth.jsonl at {path}")
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            idx = int(r["i"])
            if idx % every:
                continue
            x, y = r["center_map_px"]
            out.append(TruthPoint(idx, float(r["t"]), float(x), float(y)))
        except KeyError as e:
            raise ValueError(f"{path} line {n}: missing {e.args[0]}") from None
        except (TypeError, ValueError) as e:
            raise ValueError(f"{path} line {n}: not a truth record (i, t, center_map_px): {e}") from None
    _check_track(out)        # also: an empty file is an error, not a flight with no frames
    return out
