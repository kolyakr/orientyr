"""Shared pytest fixtures.

``irpin_mild``: the tiny Irpin bundle in tests/fixtures/irpin_mild_2025 (M0-08), so matcher, geometry and
tracking tests can run on real imagery in CI, without data/. See fixture.json there for the conventions:
truth is in full-map px (``to_crop`` converts), and its homographies act on undistorted frames.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

IRPIN_MILD = Path(__file__).resolve().parent / "fixtures" / "irpin_mild_2025"


@dataclass(frozen=True)
class IrpinFixture:
    crop: np.ndarray                 # BGR, 512 × 512: map px = crop px + origin
    origin: tuple[int, int]          # (x0, y0) of the crop in the full irpin_2025 map
    frames: list[np.ndarray]         # BGR keyframes as rendered (with lens distortion), in truth order
    truth: list[dict]                # the flight's truth.jsonl records, full-map coordinates
    K: np.ndarray                    # 3 × 3 camera matrix
    dist: np.ndarray                 # OpenCV distortion coefficients
    meta: dict                       # fixture.json as is

    def to_crop(self, pts_map: np.ndarray) -> np.ndarray:
        """Full-map px (N × 2) → crop px."""
        return np.asarray(pts_map, float) - np.asarray(self.origin, float)


@pytest.fixture(scope="session")
def irpin_mild() -> IrpinFixture:
    meta = json.loads((IRPIN_MILD / "fixture.json").read_text())
    truth = [json.loads(ln) for ln in (IRPIN_MILD / meta["truth"]).read_text().splitlines() if ln.strip()]
    return IrpinFixture(
        crop=cv2.imread(str(IRPIN_MILD / meta["crop"]["file"]), cv2.IMREAD_COLOR),
        origin=tuple(meta["crop"]["origin_px"]),
        frames=[cv2.imread(str(IRPIN_MILD / f["file"]), cv2.IMREAD_COLOR) for f in meta["frames"]],
        truth=truth,
        K=np.asarray(meta["camera"]["K"], float),
        dist=np.asarray(meta["camera"]["dist"], float),
        meta=meta,
    )
