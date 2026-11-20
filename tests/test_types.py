"""M0-02 · Core types and interfaces: the contract every other milestone builds on.

"Done when: imported by a test; the spec table and the code agree." So:
- each type has exactly the fields of the M0 spec table, in that order (and, when the plan docs are
  present locally, the table itself is parsed and compared, so the two can't drift apart);
- the conventions in the docstrings are enforced where they can be checked: shapes, dtypes,
  heading range, the accepted / reject_reason pairing, the three track states.
"""
import dataclasses
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from dronenav import types as T

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "docs/plan/M0-foundation/SPEC.md"   # docs/ is not in git: compared only where present

FIELDS = {
    "Frame": ["idx", "t", "image", "source", "heading_prior_deg", "scale_prior"],
    "Matches": ["pts_frame", "pts_map", "scores", "backend", "ms"],
    "Fix": ["idx", "H", "x", "y", "heading_deg", "scale", "footprint", "inliers", "inlier_ratio",
            "reproj_px", "accepted", "reject_reason"],
    "TrackState": ["idx", "state", "x", "y", "vx", "vy", "cov", "fix"],
}


# --- the spec table and the code agree --------------------------------------------------------

@pytest.mark.parametrize("name", FIELDS)
def test_fields_match_spec(name):
    assert [f.name for f in dataclasses.fields(getattr(T, name))] == FIELDS[name]


def _spec_rows():
    rows = {}
    for line in SPEC.read_text().splitlines():
        m = re.match(r"\|\s*`(\w+)`\s*\|(.*)\|\s*$", line)
        if m:
            rows[m.group(1)] = m.group(2)
    return rows


@pytest.mark.skipif(not SPEC.exists(), reason="plan docs are kept outside git")
@pytest.mark.parametrize("name", FIELDS)
def test_spec_table_lists_the_same_fields(name):
    # every backticked identifier in the table cell, minus words that are values, not fields
    cell = _spec_rows()[name]
    names = [w for w in re.findall(r"`(\w+)`", cell) if w not in {"none", "None"}]
    assert names == FIELDS[name]


def test_matcher_signature_matches_spec():
    sig = inspect.signature(T.Matcher.match)
    assert list(sig.parameters) == ["self", "frame_img", "map_img", "map_offset", "mask"]
    assert sig.parameters["map_offset"].default == (0, 0)
    assert sig.parameters["mask"].default is None


def test_track_states_match_spec():
    assert [s.value for s in T.TrackMode] == ["TRACKING", "LOST", "RELOCALIZING"]


# --- Frame -----------------------------------------------------------------------------------

def _img(h=48, w=64):
    return np.zeros((h, w, 3), np.uint8)


def test_frame_minimal_and_optional_priors():
    f = T.Frame(idx=0, t=0.0, image=_img(), source="frames/0001.png")
    assert f.heading_prior_deg is None and f.scale_prior is None


@pytest.mark.parametrize("bad", [np.zeros((48, 64), np.uint8),          # grey: spec says BGR
                                 np.zeros((48, 64, 4), np.uint8),       # BGRA
                                 np.zeros((48, 64, 3), np.float32)])    # not uint8
def test_frame_rejects_non_bgr_uint8(bad):
    with pytest.raises(ValueError):
        T.Frame(idx=0, t=0.0, image=bad, source="x")


def test_frame_heading_prior_is_normalised_to_0_360():
    assert T.Frame(idx=0, t=0.0, image=_img(), source="x", heading_prior_deg=-90).heading_prior_deg == 270


@pytest.mark.parametrize("a", [-1e-17, -1e-15, -1e-14, -2e-14, -5e-324])
def test_tiny_negative_heading_wraps_below_360(a):
    # float % 360 rounds a hair below zero up to exactly 360.0, outside the documented [0, 360) (#86)
    h = T._wrap_deg(a)
    assert 0.0 <= h < 360.0
    assert h == 0.0 or h > 359.9


def test_types_are_immutable():
    f = T.Frame(idx=0, t=0.0, image=_img(), source="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        f.idx = 1


# --- Matches ---------------------------------------------------------------------------------

def test_matches_coerce_to_float_arrays():
    m = T.Matches(pts_frame=[[1, 2], [3, 4]], pts_map=[[10, 20], [30, 40]], scores=[0.9, 0.5],
                  backend="sift", ms=1.5)
    assert m.pts_frame.shape == (2, 2) and m.pts_frame.dtype == np.float64
    assert m.scores.shape == (2,) and len(m) == 2


def test_matches_empty_is_valid():
    # "no matches" is a normal outcome (blank frame, wrong window), not an error
    m = T.Matches.empty("sift")
    assert len(m) == 0 and m.pts_map.shape == (0, 2) and m.backend == "sift"


@pytest.mark.parametrize("kw", [{"pts_map": np.zeros((3, 2))},          # N differs
                                {"scores": np.zeros(3)},
                                {"pts_frame": np.zeros((2, 3))}])        # not N x 2
def test_matches_reject_inconsistent_shapes(kw):
    base = {"pts_frame": np.zeros((2, 2)), "pts_map": np.zeros((2, 2)), "scores": np.zeros(2),
            "backend": "b", "ms": 0.0}
    with pytest.raises(ValueError):
        T.Matches(**{**base, **kw})


# --- Fix -------------------------------------------------------------------------------------

def _fix(**kw):
    base = {"idx": 3, "H": np.eye(3), "x": 100.0, "y": 200.0, "heading_deg": 45.0, "scale": 1.0,
                "footprint": np.array([[0, 0], [64, 0], [64, 48], [0, 48]]), "inliers": 120,
                "inlier_ratio": 0.6, "reproj_px": 1.2, "accepted": True, "reject_reason": None}
    return T.Fix(**{**base, **kw})


def test_fix_valid_and_heading_normalised():
    f = _fix(heading_deg=370.0)
    assert f.heading_deg == pytest.approx(10.0)
    assert f.H.shape == (3, 3) and f.footprint.shape == (4, 2) and f.footprint.dtype == np.float64


@pytest.mark.parametrize("kw", [{"H": np.eye(2)}, {"footprint": np.zeros((3, 2))},
                                {"inlier_ratio": 1.5}, {"inliers": -1}])
def test_fix_rejects_bad_geometry_and_counts(kw):
    with pytest.raises(ValueError):
        _fix(**kw)


def test_fix_reject_reason_pairs_with_accepted():
    # a rejected fix must say why; an accepted fix carries no reason
    assert _fix(accepted=False, reject_reason="mirrored").reject_reason == "mirrored"
    with pytest.raises(ValueError):
        _fix(accepted=False, reject_reason=None)
    with pytest.raises(ValueError):
        _fix(accepted=True, reject_reason="mirrored")


# --- TrackState ------------------------------------------------------------------------------

def test_track_state_accepts_state_name_and_optional_fix():
    s = T.TrackState(idx=0, state="LOST", x=1.0, y=2.0, vx=0.0, vy=0.0, cov=np.eye(2), fix=None)
    assert s.state is T.TrackMode.LOST and s.cov.shape == (2, 2)
    s2 = T.TrackState(idx=1, state=T.TrackMode.TRACKING, x=1.0, y=2.0, vx=0.5, vy=0.0, cov=np.eye(2), fix=_fix())
    assert s2.fix.inliers == 120


@pytest.mark.parametrize("kw", [{"state": "FLYING"}, {"cov": np.eye(3)}])
def test_track_state_rejects_unknown_state_and_bad_cov(kw):
    with pytest.raises(ValueError):
        T.TrackState(**{"idx": 0, "state": "LOST", "x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0,
                        "cov": np.eye(2), "fix": None, **kw})


# --- protocols -------------------------------------------------------------------------------

def test_protocols_are_structural():
    class Toy:
        def match(self, frame_img, map_img, map_offset=(0, 0), mask=None):
            return T.Matches.empty("toy")

    class NotAMatcher:
        pass

    assert isinstance(Toy(), T.Matcher) and not isinstance(NotAMatcher(), T.Matcher)

    def source():
        yield T.Frame(idx=0, t=0.0, image=_img(), source="gen")

    assert isinstance(source(), T.FrameSource)
    assert not isinstance(42, T.FrameSource)
