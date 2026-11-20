"""M0-08 · the tiny Irpin fixture that lets CI test matching and geometry without data/.

Contract (from the spec: "a 512 px crop of irpin_2025.png and three keyframes of the mild 2025 flight with their
truth.jsonl records, JPEG, a few hundred KB in total"), plus what makes it usable:
- the files are there and small enough to live in git;
- the crop is 512 × 512 and its origin in the full map is recorded, so map px = crop px + origin;
- the frames are the camera's size and their names match the truth records;
- the truth records are the originals (map coordinates, self-consistent);
- every keyframe centre lies inside the crop, and the truth homography actually registers the frame pixels onto
  the crop pixels (the strongest check that nothing got shifted, flipped or mixed up on the way).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

FIX = Path(__file__).resolve().parent / "fixtures" / "irpin_mild_2025"


def _meta() -> dict:
    return json.loads((FIX / "fixture.json").read_text())


def _truth() -> list[dict]:
    return [json.loads(ln) for ln in (FIX / "truth.jsonl").read_text().splitlines() if ln.strip()]


def _crop_from_frame(rec: dict, meta: dict) -> np.ndarray:
    """Truth homography re-expressed as frame px → crop px."""
    x0, y0 = meta["crop"]["origin_px"]
    shift = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], float)
    return shift @ np.asarray(rec["H_frame_to_map"], float)


def _ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    a, b = a[mask].astype(float), b[mask].astype(float)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def _registration_score(rec: dict, meta: dict, extra: np.ndarray | None = None) -> float:
    """NCC between the crop and the keyframe warped onto it, over the part of the crop the frame covers."""
    crop = cv2.imread(str(FIX / meta["crop"]["file"]), cv2.IMREAD_GRAYSCALE)
    frame = cv2.imread(str(FIX / f"frame_{rec['i']:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
    H = _crop_from_frame(rec, meta) if extra is None else extra @ _crop_from_frame(rec, meta)
    size = crop.shape[::-1]
    warped = cv2.warpPerspective(frame, H, size, flags=cv2.INTER_AREA)
    cover = cv2.warpPerspective(np.full(frame.shape, 255, np.uint8), H, size, flags=cv2.INTER_NEAREST)
    # blur both a little: the frame is rendered from the z18 map with noise and JPEG; compare structure, not grain
    warped, crop = cv2.GaussianBlur(warped, (0, 0), 1.5), cv2.GaussianBlur(crop, (0, 0), 1.5)
    return _ncc(warped, crop, cv2.erode(cover, np.ones((9, 9), np.uint8)) > 0)


def test_the_bundle_is_complete_and_small():
    meta = _meta()
    files = [meta["crop"]["file"], meta["truth"], *[f["file"] for f in meta["frames"]], "fixture.json"]
    for f in files:
        assert (FIX / f).is_file(), f
    total = sum((FIX / f).stat().st_size for f in files)
    assert total <= 600_000, f"{total / 1e3:.0f} KB"         # "a few hundred KB": git must stay light


def test_there_are_three_jpeg_keyframes():
    meta = _meta()
    assert len(meta["frames"]) == 3
    assert all(f["file"].endswith(".jpg") for f in meta["frames"]) and meta["crop"]["file"].endswith(".jpg")


def test_the_crop_is_512_square_inside_the_map():
    meta = _meta()
    crop = cv2.imread(str(FIX / meta["crop"]["file"]))
    assert crop is not None and crop.shape == (512, 512, 3)
    assert meta["crop"]["size_px"] == [512, 512]
    (x0, y0), (w, h) = meta["crop"]["origin_px"], meta["source"]["map_size_px"]
    assert 0 <= x0 <= w - 512 and 0 <= y0 <= h - 512


def test_frames_have_the_camera_size_and_match_the_truth():
    meta, truth = _meta(), _truth()
    assert [f["i"] for f in meta["frames"]] == [r["i"] for r in truth]
    for f in meta["frames"]:
        assert f["file"] == f"frame_{f['i']:06d}.jpg"
        img = cv2.imread(str(FIX / f["file"]))
        assert img is not None and [img.shape[1], img.shape[0]] == meta["camera"]["size"]


def test_truth_records_are_the_originals():
    for r in _truth():
        assert {"i", "t", "H_frame_to_map", "center_map_px", "footprint_map_px", "heading_deg", "altitude_m"} <= r.keys()
        # the record's own centre is H applied to the principal point; 0.5 px covers float rounding in the generator
        K = np.asarray(_meta()["camera"]["K"])
        p = np.asarray(r["H_frame_to_map"]) @ [K[0, 2], K[1, 2], 1.0]
        assert np.allclose(p[:2] / p[2], r["center_map_px"], atol=0.5)


def test_keyframe_centres_lie_well_inside_the_crop():
    meta = _meta()
    x0, y0 = meta["crop"]["origin_px"]
    for r in _truth():
        cx, cy = r["center_map_px"]
        # 96 px margin: a local search window around the centre still sees mostly crop, not border
        assert 96 <= cx - x0 <= 512 - 96 and 96 <= cy - y0 <= 512 - 96


def test_keyframes_show_motion_and_turn():
    truth = _truth()
    c = np.array([r["center_map_px"] for r in truth])
    assert np.linalg.norm(np.diff(c, axis=0), axis=1).min() >= 20      # distinct positions, not one frame thrice
    assert abs(truth[-1]["heading_deg"] - truth[0]["heading_deg"]) >= 10   # rotation between frames: a real test


@pytest.mark.parametrize("k", [0, 1, 2])
def test_truth_registers_the_frame_onto_the_crop(k):
    meta, rec = _meta(), _truth()[k]
    good = _registration_score(rec, meta)
    off = _registration_score(rec, meta, np.array([[1, 0, 12], [0, 1, 12], [0, 0, 1]], float))
    # aligned imagery correlates strongly; a 12 px slip (~9 m) must cost a lot, or the check proves nothing
    assert good >= 0.6, good
    assert good - off >= 0.2, (good, off)


def test_conftest_fixture_loads_the_bundle(irpin_mild):
    assert irpin_mild.crop.shape == (512, 512, 3)
    assert len(irpin_mild.frames) == len(irpin_mild.truth) == 3
    x0, y0 = irpin_mild.origin
    assert irpin_mild.to_crop(np.array([[x0 + 5.0, y0 + 7.0]])).tolist() == [[5.0, 7.0]]
