"""M2-05 · Arbitrary in-plane rotation.

"Done when: frames localize at every heading" (spec update, 20 Nov: the Irpin flights turn through every heading,
so they check this before the 360° phone take is filmed). So:
- heading arithmetic that survives the 0/360 seam: signed error, bins over [0, 360);
- a per-heading-bin report (frames, localized, success rate, error) that the check script and M4-04 share;
  "localized" means an accepted fix near the truth: a confident wrong fix is not a success;
- a synthetic flight that turns through a full circle (the 360° take in miniature): every frame localized,
  through the same Searcher the pipeline uses, with the right heading;
- real imagery (the CI Irpin fixture): a keyframe turned by 90°, 180° and 270° gives the same position and its
  heading turned by the same angle;
- on the Irpin flights (``@pytest.mark.data``): every 45° heading bin has frames and is localized.

Ground truth: tests/synth.py (exact homographies) and the fixture's truth.jsonl.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from tests.synth import make_h, make_map, render_frame

from dronenav.matching import RootSiftMatcher
from dronenav.rotation import FrameResult, HeadingBin, by_heading, check_flight, heading_bin, heading_error_deg
from dronenav.search import Searcher
from dronenav.types import Fix

ROOT = Path(__file__).resolve().parents[1]


def _fix(x: float, y: float, heading: float = 0.0, accepted: bool = True) -> Fix:
    fp = np.array([[x - 10, y - 10], [x + 10, y - 10], [x + 10, y + 10], [x - 10, y + 10]], float)
    return Fix(idx=0, H=np.eye(3), x=x, y=y, heading_deg=heading, scale=1.0, footprint=fp, inliers=50,
               inlier_ratio=0.5, reproj_px=1.0, accepted=accepted, reject_reason=None if accepted else "test")


# --- heading arithmetic ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("est, truth, err", [
    (10, 0, 10), (0, 10, -10),
    (1, 359, 2), (359, 1, -2),            # across the seam: 2°, not 358°
    (180, 0, 180), (0, 180, 180),         # the half turn has one sign: (-180, 180]
    (725, 0, 5), (-5, 0, -5),             # inputs outside [0, 360) are fine
])
def test_heading_error_is_signed_and_wraps(est, truth, err):
    assert heading_error_deg(est, truth) == pytest.approx(err)


@pytest.mark.parametrize("h, b", [(0, 0), (44.9, 0), (45, 1), (180, 4), (359.9, 7), (360, 0), (-10, 7), (725, 0)])
def test_heading_bins_cover_the_circle(h, b):
    assert heading_bin(h, 8) == b


def test_heading_bin_count_is_checked():
    with pytest.raises(ValueError):
        heading_bin(10.0, 0)


# --- the per-bin report ---------------------------------------------------------------------------------------

def test_by_heading_counts_success_per_bin():
    rows = [
        FrameResult(heading_deg=10, fix=_fix(100, 100), truth_xy=(101, 100)),       # bin 0, localized
        FrameResult(heading_deg=20, fix=None, truth_xy=(0, 0)),                     # bin 0, no fix
        FrameResult(heading_deg=100, fix=_fix(50, 50, accepted=False), truth_xy=(50, 50)),  # bin 2, rejected
        FrameResult(heading_deg=100, fix=_fix(500, 50), truth_xy=(50, 50)),         # bin 2, accepted but wrong
        FrameResult(heading_deg=350, fix=_fix(0, 3), truth_xy=(0, 0)),              # bin 7, localized
    ]
    bins = by_heading(rows, n_bins=8, tol_px=20.0)
    assert [b.lo_deg for b in bins] == [0, 45, 90, 135, 180, 225, 270, 315]
    assert all(isinstance(b, HeadingBin) and b.hi_deg == b.lo_deg + 45 for b in bins)
    assert [b.frames for b in bins] == [2, 0, 2, 0, 0, 0, 0, 1]
    assert [b.localized for b in bins] == [1, 0, 0, 0, 0, 0, 0, 1]
    assert bins[0].success == pytest.approx(0.5) and bins[2].success == 0.0 and bins[7].success == 1.0
    assert bins[1].success is None                       # no frames: unknown, not 0 % and not 100 %
    assert bins[0].median_err_px == pytest.approx(1.0) and bins[7].median_err_px == pytest.approx(3.0)
    assert bins[2].median_err_px is None                 # nothing localized: no error to report
    assert bins[2].wrong == 1 and bins[0].wrong == 0     # accepted but far: counted apart, it misleads the tracker
    # heading error over the localized frames, across the seam: fix 0° vs truth 350° is 10°, fix 0° vs 10° is 10°
    assert bins[0].max_heading_err_deg == pytest.approx(10.0) and bins[7].max_heading_err_deg == pytest.approx(10.0)
    assert bins[2].max_heading_err_deg is None


def test_by_heading_validates_its_inputs():
    with pytest.raises(ValueError):
        by_heading([], n_bins=8, tol_px=0.0)
    with pytest.raises(ValueError):
        by_heading([], n_bins=0, tol_px=5.0)


# --- a synthetic flight that turns through a full circle -------------------------------------------------------

FRAME_SIZE = (480, 360)
MAP = make_map()
# 24 frames, 15° apart: a slow circle in the middle of the map, each frame turned 15° more than the last,
# like the phone turned through 360° over the screen; scale 0.8 (no shrink at this zoom)
CIRCLE = [(600 + 120 * math.sin(math.radians(h)), 450 - 120 * math.cos(math.radians(h)), float(h))
          for h in range(0, 360, 15)]


def test_a_full_turn_localizes_at_every_heading(tmp_path):
    searcher = Searcher(RootSiftMatcher(MAP, cache_dir=tmp_path), MAP)
    rows = []
    for n, (x, y, h) in enumerate(CIRCLE):
        frame = render_frame(MAP, make_h(h, 0.8, cx=x, cy=y, frame_size=FRAME_SIZE), FRAME_SIZE)
        r = searcher.locate(n, frame)
        rows.append(FrameResult(heading_deg=h, fix=r.fix, truth_xy=(x, y)))
        assert r.fix is not None and r.fix.accepted, f"heading {h}°: {r.fix and r.fix.reject_reason}"
        # exact synthetic homographies: SIFT recovers them to well under a pixel and a degree
        assert math.hypot(r.fix.x - x, r.fix.y - y) < 1.0, h
        assert abs(heading_error_deg(r.fix.heading_deg, h)) < 0.5, (h, r.fix.heading_deg)
    assert all(b.success == 1.0 for b in by_heading(rows, n_bins=8, tol_px=1.0))


# --- real imagery: the same keyframe, turned ----------------------------------------------------------------------

@pytest.mark.parametrize("quarter", [1, 2, 3])
def test_a_turned_irpin_keyframe_lands_in_the_same_place(irpin_mild, tmp_path, quarter):
    """np.rot90 turns the picture counter-clockwise about its centre by quarter × 90°; the new top edge is the old
    right edge, so the frame's up now points where its right pointed: heading + quarter × 90°. The centre pixel
    stays the centre pixel, so the position must not move."""
    crop = irpin_mild.crop
    for k, frame in enumerate(irpin_mild.frames):
        upright = Searcher(RootSiftMatcher(crop, cache_dir=tmp_path), crop).locate(0, frame).fix
        turned = Searcher(RootSiftMatcher(crop, cache_dir=tmp_path), crop).locate(0, np.rot90(frame, quarter).copy()).fix
        assert upright is not None and upright.accepted, k
        assert turned is not None and turned.accepted, (k, turned and turned.reject_reason)
        # same pixels, same matches up to resampling: 2 px is a few SIFT keypoint jitters at this scale
        assert math.hypot(turned.x - upright.x, turned.y - upright.y) < 2.0, k
        assert abs(heading_error_deg(turned.heading_deg, upright.heading_deg + 90 * quarter)) < 1.0, k


# --- on the Irpin flights, when data/ is present ------------------------------------------------------------------

IRPIN = ROOT / "data" / "irpin"


@pytest.mark.data
@pytest.mark.skipif(not (IRPIN / "flights" / "mild" / "2025" / "meta.json").exists(), reason="data/ not downloaded")
def test_the_irpin_mild_2025_flight_localizes_at_every_heading():
    bins = check_flight(IRPIN / "flights" / "mild" / "2025", IRPIN / "maps" / "irpin_2025.png",
                        cache_dir=ROOT / "outputs" / "cache")
    # the flight turns through every 45° bin (the spec's premise); a bin with no frames would prove nothing
    assert all(b.frames > 0 for b in bins), [b.frames for b in bins]
    # "at every heading", taken literally: every frame of every bin localized, none confidently wrong, and the
    # heading itself right. Measured (RootSIFT, 20 Nov): 442/442, error <= 1.2 px, heading <= 0.3°; a 1° bound
    # leaves room for other OpenCV builds yet catches a matcher that confuses a turned frame with a neighbour.
    for b in bins:
        assert b.success == 1.0, (b.lo_deg, b.success)
        assert b.wrong == 0, (b.lo_deg, b.wrong)
        assert b.max_heading_err_deg < 1.0, (b.lo_deg, b.max_heading_err_deg)


@pytest.mark.data
@pytest.mark.skipif(not (IRPIN / "flights" / "mild" / "2025" / "meta.json").exists(), reason="data/ not downloaded")
def test_rotation_check_writes_its_table(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "rot"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "rotation_check.py"), "--irpin", str(IRPIN),
                    "--flights", "mild/2025", "--out", str(out), "--cache", str(ROOT / "outputs" / "cache")],
                   check=True, cwd=ROOT)
    table = json.loads((out / "rotation.json").read_text())
    assert set(table) == {"mild/2025"} and len(table["mild/2025"]) == 8
    assert (out / "rotation.md").read_text().count("\n|") >= 9        # header + 8 bins
    assert (out / "rotation.png").stat().st_size > 0
