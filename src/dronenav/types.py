"""Shared types and interfaces: the contract between frame sources, matchers, the tracker and the renderers.

Frozen by the M0 spec (docs/plan/M0-foundation/SPEC.md, "Interfaces"). Fields may be added later;
existing ones are never renamed, because renderers, the benchmark and recorded run.jsonl files depend on them.

Conventions used everywhere:

- **Map coordinates** are pixels of the full-resolution map, (x, y): x to the right (east), y down (south),
  origin at the top-left corner of the top-left pixel. Metres and lon/lat exist only at the edges,
  through the map's sidecar.
- **Frame coordinates** are pixels of the frame as given to the matcher, (x, y), same orientation.
- **Homographies** map frame pixels to map pixels: ``p_map ~ H @ [x, y, 1]``.
- **Heading** is in degrees in [0, 360), clockwise from map up: 0 = the frame's "up" points to map
  north (up), 90 = east (right). Inputs are normalised into that range.
- **Scale** is map pixels per frame pixel at the frame centre (> 1: the frame is more detailed than the map).
- **Time** is in seconds from the start of the flight; durations are in milliseconds.

Values are validated on construction so a malformed object fails where it is made, not three modules later.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["Fix", "Frame", "FrameSource", "Matcher", "Matches", "TrackMode", "TrackState"]


def _wrap_deg(a: float | None) -> float | None:
    """Normalise an angle to [0, 360); None stays None."""
    return None if a is None else float(a) % 360.0


def _array(value, shape: tuple, name: str) -> np.ndarray:
    """float64 copy of ``value`` with the given shape; -1 in ``shape`` means any length."""
    a = np.array(value, dtype=np.float64)
    if a.ndim != len(shape) or any(s != -1 and s != d for s, d in zip(shape, a.shape)):
        want = "x".join("N" if s == -1 else str(s) for s in shape)
        raise ValueError(f"{name} must be {want}, got shape {a.shape}")
    return a


@dataclass(frozen=True, eq=False)  # numpy fields: identity equality, no ambiguous array ==
class Frame:
    """One camera frame entering the pipeline.

    idx: frame number within its source, from 0. t: seconds from the start of the flight.
    image: BGR uint8, H x W x 3 (as OpenCV reads it). source: the file path or video name it came from.
    heading_prior_deg: optional heading hint (compass, simulator), degrees, see module conventions.
    scale_prior: optional hint of map px per frame px (altitude, previous fix).
    """

    idx: int
    t: float
    image: np.ndarray
    source: str
    heading_prior_deg: float | None = None
    scale_prior: float | None = None

    def __post_init__(self):
        img = self.image
        if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Frame.image must be an H x W x 3 uint8 BGR array, got "
                             f"{getattr(img, 'shape', type(img))} {getattr(img, 'dtype', '')}")
        object.__setattr__(self, "heading_prior_deg", _wrap_deg(self.heading_prior_deg))


@runtime_checkable
class FrameSource(Protocol):
    """Anything that yields Frames in time order: folder, video, camera, simulator."""

    def __iter__(self) -> Iterator[Frame]: ...


@dataclass(frozen=True, eq=False)  # numpy fields: identity equality, no ambiguous array ==
class Matches:
    """Point correspondences between one frame and the map.

    pts_frame: N x 2 frame pixels. pts_map: N x 2 map pixels at full map resolution (the matcher adds the
    window offset back). scores: N match confidences, higher is better, scale depends on the backend.
    backend: matcher name (e.g. "sift", "lightglue"). ms: wall time of the match call, milliseconds.
    N = 0 is valid: no matches is an ordinary outcome, the geometry step rejects it.
    """

    pts_frame: np.ndarray
    pts_map: np.ndarray
    scores: np.ndarray
    backend: str
    ms: float

    def __post_init__(self):
        pf = _array(self.pts_frame, (-1, 2), "pts_frame")
        pm = _array(self.pts_map, (-1, 2), "pts_map")
        sc = _array(self.scores, (-1,), "scores")
        if not len(pf) == len(pm) == len(sc):
            raise ValueError(f"pts_frame, pts_map and scores must have the same length, got "
                             f"{len(pf)}, {len(pm)}, {len(sc)}")
        for k, v in (("pts_frame", pf), ("pts_map", pm), ("scores", sc)):
            object.__setattr__(self, k, v)

    def __len__(self) -> int:
        return len(self.scores)

    @classmethod
    def empty(cls, backend: str, ms: float = 0.0) -> Matches:
        """No correspondences (blank frame, nothing in the window)."""
        return cls(np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0), backend, ms)


@runtime_checkable
class Matcher(Protocol):
    """A matching backend (RootSIFT, LightGlue, XFeat, ...).

    ``match`` finds correspondences between ``frame_img`` and ``map_img``. ``map_img`` may be a window cut
    from the full map whose top-left corner is at ``map_offset`` = (x, y) in full-map pixels; the returned
    ``pts_map`` are in full-map pixels (offset added). ``mask`` is optional, uint8, the size of ``map_img``,
    nonzero where map pixels may be used. Backends may cache map features between calls.
    """

    def match(self, frame_img: np.ndarray, map_img: np.ndarray, map_offset: tuple = (0, 0),
              mask: np.ndarray | None = None) -> Matches: ...


@dataclass(frozen=True, eq=False)  # numpy fields: identity equality, no ambiguous array ==
class Fix:
    """One frame's position estimate from a homography, accepted or rejected.

    idx: frame index. H: 3 x 3 homography frame px -> map px. x, y: frame centre in map px.
    heading_deg: heading, see module conventions. scale: map px per frame px at the centre.
    footprint: 4 x 2 map px of the frame corners, in order top-left, top-right, bottom-right, bottom-left.
    inliers: RANSAC inlier count. inlier_ratio: inliers / matches, in [0, 1].
    reproj_px: median reprojection error of the inliers, map px.
    accepted: passed the sanity checks. reject_reason: short code of the failed check ("mirrored",
    "too_few_inliers", ...) when rejected, None when accepted.
    """

    idx: int
    H: np.ndarray
    x: float
    y: float
    heading_deg: float
    scale: float
    footprint: np.ndarray
    inliers: int
    inlier_ratio: float
    reproj_px: float
    accepted: bool
    reject_reason: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "H", _array(self.H, (3, 3), "H"))
        object.__setattr__(self, "footprint", _array(self.footprint, (4, 2), "footprint"))
        object.__setattr__(self, "heading_deg", _wrap_deg(self.heading_deg))
        if self.inliers < 0:
            raise ValueError(f"inliers must be >= 0, got {self.inliers}")
        if not 0.0 <= self.inlier_ratio <= 1.0:
            raise ValueError(f"inlier_ratio must be in [0, 1], got {self.inlier_ratio}")
        if self.accepted and self.reject_reason is not None:
            raise ValueError("an accepted Fix has no reject_reason")
        if not self.accepted and not self.reject_reason:
            raise ValueError("a rejected Fix must give a reject_reason")


class TrackMode(str, Enum):
    """Tracker state. A str enum, so it serialises to its name in JSON."""

    TRACKING = "TRACKING"          # fixes arrive and agree with the motion model
    LOST = "LOST"                  # no accepted fix for a while: position is dead-reckoned
    RELOCALIZING = "RELOCALIZING"  # global search over the whole map to recover


@dataclass(frozen=True, eq=False)  # numpy fields: identity equality, no ambiguous array ==
class TrackState:
    """The tracker's belief after one frame.

    idx: frame index. state: TrackMode (a name such as "LOST" is accepted). x, y: filtered position, map px.
    vx, vy: velocity, map px per second. cov: 2 x 2 position covariance, map px squared.
    fix: the frame's Fix if one was produced (accepted or not), else None.
    """

    idx: int
    state: TrackMode
    x: float
    y: float
    vx: float
    vy: float
    cov: np.ndarray
    fix: Fix | None = None

    def __post_init__(self):
        try:
            object.__setattr__(self, "state", TrackMode(self.state))
        except ValueError:
            raise ValueError(f"state must be one of {[m.value for m in TrackMode]}, got {self.state!r}") from None
        object.__setattr__(self, "cov", _array(self.cov, (2, 2), "cov"))
