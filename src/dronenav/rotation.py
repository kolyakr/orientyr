"""Localization at any heading (M2-05): heading arithmetic and a success-by-heading report.

A drone can face any way, so the frame can be turned any amount against the north-up map. RootSIFT is
rotation-invariant (each keypoint carries its own orientation), so the matcher needs no heading at all; this module
measures that claim instead of trusting it. Frames with a known true heading go through the same Searcher the
pipeline uses, and the result is binned by heading: a matcher that fails only when the drone flies south shows up
as one bad bin, which an overall success rate would hide.

The learned matchers (M4) are not rotation-invariant; M4-04 reuses this report to show what north-up by compass
buys them.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dronenav.matching import RootSiftMatcher
from dronenav.search import Searcher
from dronenav.types import Fix, Matcher

__all__ = ["FrameResult", "HeadingBin", "by_heading", "check_flight", "heading_bin", "heading_error_deg"]

LOCALIZED_TOL_PX = 10.0   # an accepted fix this close to the truth is a success (see check_flight)


def heading_error_deg(estimate: float, truth: float) -> float:
    """Signed ``estimate - truth`` in (-180, 180]: 1° against 359° is +2°, not -358°."""
    d = (float(estimate) - float(truth)) % 360.0
    return d - 360.0 if d > 180.0 else d


def heading_bin(heading_deg: float, n_bins: int = 8) -> int:
    """Index of the equal bin of [0, 360) the heading falls in; bin 0 starts at north."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    return min(int((float(heading_deg) % 360.0) / (360.0 / n_bins)), n_bins - 1)   # min: 359.99… rounding


@dataclass(frozen=True)
class FrameResult:
    """One frame of a check: its true heading and centre (map px), and the fix the search gave (None: none)."""

    heading_deg: float
    fix: Fix | None
    truth_xy: tuple[float, float]


@dataclass(frozen=True)
class HeadingBin:
    """One heading bin [lo_deg, hi_deg). localized: accepted fixes within the tolerance of the truth. wrong:
    accepted fixes beyond it (worse than a miss: the tracker believes them). success: localized / frames, None
    for an empty bin. median_err_px and max_heading_err_deg are over the localized frames, None without any."""

    lo_deg: float
    hi_deg: float
    frames: int
    localized: int
    wrong: int
    success: float | None
    median_err_px: float | None
    max_heading_err_deg: float | None


def by_heading(rows: Iterable[FrameResult], n_bins: int = 8, tol_px: float = LOCALIZED_TOL_PX) -> list[HeadingBin]:
    """Group frames into ``n_bins`` heading bins by their true heading and count what localized in each."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if tol_px <= 0:
        raise ValueError(f"tol_px must be > 0, got {tol_px}")
    frames = [0] * n_bins
    wrong = [0] * n_bins
    errs: list[list[float]] = [[] for _ in range(n_bins)]
    dh: list[list[float]] = [[] for _ in range(n_bins)]
    for r in rows:
        b = heading_bin(r.heading_deg, n_bins)
        frames[b] += 1
        if r.fix is None or not r.fix.accepted:
            continue
        e = math.hypot(r.fix.x - r.truth_xy[0], r.fix.y - r.truth_xy[1])
        if e > tol_px:
            wrong[b] += 1
            continue
        errs[b].append(e)
        dh[b].append(abs(heading_error_deg(r.fix.heading_deg, r.heading_deg)))
    width = 360.0 / n_bins
    return [HeadingBin(lo_deg=b * width, hi_deg=(b + 1) * width, frames=frames[b], localized=len(errs[b]),
                       wrong=wrong[b], success=len(errs[b]) / frames[b] if frames[b] else None,
                       median_err_px=float(np.median(errs[b])) if errs[b] else None,
                       max_heading_err_deg=max(dh[b]) if dh[b] else None)
            for b in range(n_bins)]


def check_flight(flight: str | Path, map_path: str | Path, matcher: Matcher | None = None,
                 cache_dir: str | Path | None = None, n_bins: int = 8,
                 tol_px: float = LOCALIZED_TOL_PX) -> list[HeadingBin]:
    """Run a simulated flight's keyframes (meta.json, truth.jsonl) through the Searcher, in flight order, and
    report success by true heading. ``tol_px`` (10 px) is under 2 % of an Irpin footprint (~650 px): a fix that
    close is on the right spot, one further off is a wrong fix. The default matcher is RootSIFT."""
    flight = Path(flight)
    mp = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
    if mp is None:
        raise FileNotFoundError(f"cannot read the map {map_path}")
    if matcher is None:
        matcher = RootSiftMatcher(mp, cache_dir=cache_dir)
    meta = json.loads((flight / "meta.json").read_text())
    truth = {r["i"]: r for r in map(json.loads, (flight / "truth.jsonl").read_text().splitlines()) if r}
    searcher, rows = Searcher(matcher, mp), []
    for n, k in enumerate(meta["keyframes"]):
        frame = cv2.imread(str(flight / k["file"]), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"cannot read the keyframe {flight / k['file']}")
        t = truth[k["i"]]
        rows.append(FrameResult(t["heading_deg"], searcher.locate(n, frame).fix, tuple(t["center_map_px"])))
    return by_heading(rows, n_bins, tol_px)
