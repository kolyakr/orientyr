"""M0-06 · Data fetch and layout.

"Done when: a fresh clone plus the Drive folder unpacked into data/ passes scripts/check_data.py; a missing or
truncated file is reported by name." Tested here on tiny synthetic data/ trees (so CI needs no download); the
real data/ is checked too, when present.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.check_data import Problem, check_data

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

YEARS = ("2017", "2022", "2025")
PRESETS = ("mild", "mild_30mps", "harsh", "harsh_30mps")
N_CROPS = 50


def _png(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((size[1], size[0], 3), np.uint8))


def _mp4(path: Path, frames: int = 3, size: tuple[int, int] = (16, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, size)
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.zeros((size[1], size[0], 3), np.uint8))
    writer.release()


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _build_dataset(root: Path, *, crops: int = N_CROPS) -> None:
    """A full, valid data/ tree (tiny images/videos) matching the layout check_data expects."""
    _png(root / "hackatons-data" / "global_map.png")
    for i in range(crops):
        _png(root / "hackatons-data" / "crops" / f"crop_{i}.png")

    irpin = root / "irpin"
    for year in YEARS:
        with_hires = year != "2017"
        sidecar = {"map": {"file": f"maps/irpin_{year}.png", "size_px": [8, 8]}}
        if with_hires:
            sidecar["hires"] = {"file": f"hires/irpin_{year}_z18.png", "size_px": [8, 8]}
        _write_json(irpin / "maps" / f"irpin_{year}.json", sidecar)
        _png(irpin / "maps" / f"irpin_{year}.png")
        if with_hires:
            _png(irpin / "hires" / f"irpin_{year}_z18.png")

    _write_json(irpin / "unosat.geojson", {"type": "FeatureCollection", "features": []})
    _write_json(irpin / "source" / "unosat_ukraine_zenodo15088349.geojson",
                {"type": "FeatureCollection", "features": []})
    _png(irpin / "preview" / "sample.jpg")

    for preset in PRESETS:
        for year in YEARS:
            flight = irpin / "flights" / preset / year
            keyframes = [{"i": 0, "file": "frames/frame_00000.png"}, {"i": 1, "file": "frames/frame_00001.png"}]
            truth_lines = 7
            _write_json(flight / "meta.json", {"year": year, "fps": 30, "frames": truth_lines,
                                                "keyframes": keyframes})
            (flight / "truth.jsonl").write_text(
                "\n".join(json.dumps({"i": i}) for i in range(truth_lines)) + "\n", encoding="utf-8")
            _mp4(flight / "flight.mp4")
            for kf in keyframes:
                _png(flight / kf["file"])

    for year in YEARS:
        drone = irpin / "drone" / year
        frames = [{"frame": "frame_000.png"}, {"frame": "frame_001.png"}]
        _write_json(drone / "truth.json", {"year": year, "frames": frames})
        for f in frames:
            _png(drone / f["frame"])


def _find(report, kind: str, suffix: str) -> list[Problem]:
    return [p for p in report.problems if p.kind == kind and p.path.replace("\\", "/").endswith(suffix)]


# --- a complete tree --------------------------------------------------------------------------------------------

def test_valid_tree_has_no_problems(tmp_path):
    _build_dataset(tmp_path)
    report = check_data(tmp_path)
    assert report.ok
    assert report.problems == []
    assert report.checked > 0


# --- missing files, reported by name ----------------------------------------------------------------------------

def test_missing_crop_is_reported_by_name(tmp_path):
    _build_dataset(tmp_path)
    (tmp_path / "hackatons-data" / "crops" / "crop_17.png").unlink()
    report = check_data(tmp_path)
    assert not report.ok
    assert _find(report, "missing", "crops/crop_17.png")


def test_drone_frame_missing_is_reported_by_name(tmp_path):
    _build_dataset(tmp_path)
    (tmp_path / "irpin" / "drone" / "2025" / "frame_000.png").unlink()
    report = check_data(tmp_path)
    assert _find(report, "missing", "drone/2025/frame_000.png")


def test_missing_subtree_is_reported_once_not_per_file(tmp_path):
    _png(tmp_path / "hackatons-data" / "global_map.png")
    for i in range(N_CROPS):
        _png(tmp_path / "hackatons-data" / "crops" / f"crop_{i}.png")
    report = check_data(tmp_path)   # no irpin/ at all
    assert not report.ok
    missing_irpin = [p for p in report.problems if p.kind == "missing" and p.path.split("/", 1)[0] == "irpin"]
    assert 1 <= len(missing_irpin) <= 3   # one notice, not thousands of per-expected-file entries


def test_missing_data_dir_itself_does_not_crash(tmp_path):
    report = check_data(tmp_path / "does-not-exist")
    assert not report.ok
    assert len(report.problems) <= 2


# --- truncated / corrupt files, reported by name ------------------------------------------------------------------

def test_empty_image_file_is_reported_as_truncated(tmp_path):
    _build_dataset(tmp_path)
    (tmp_path / "hackatons-data" / "crops" / "crop_3.png").write_bytes(b"")
    report = check_data(tmp_path)
    assert _find(report, "truncated", "crops/crop_3.png")


def test_corrupt_image_bytes_are_reported_as_truncated(tmp_path):
    _build_dataset(tmp_path)
    (tmp_path / "irpin" / "maps" / "irpin_2022.png").write_bytes(b"not a png, just garbage bytes")
    report = check_data(tmp_path)
    assert _find(report, "truncated", "irpin/maps/irpin_2022.png")


def test_malformed_sidecar_json_is_reported_as_truncated(tmp_path):
    _build_dataset(tmp_path)
    (tmp_path / "irpin" / "maps" / "irpin_2017.json").write_text("{not valid json", encoding="utf-8")
    report = check_data(tmp_path)
    assert _find(report, "truncated", "irpin/maps/irpin_2017.json")


def test_truth_jsonl_line_count_mismatch_is_reported_as_truncated(tmp_path):
    _build_dataset(tmp_path)
    target = tmp_path / "irpin" / "flights" / "mild" / "2017" / "truth.jsonl"
    lines = target.read_text(encoding="utf-8").splitlines()
    target.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    report = check_data(tmp_path)
    assert _find(report, "truncated", "flights/mild/2017/truth.jsonl")


def test_bad_json_line_in_truth_jsonl_is_reported_as_truncated(tmp_path):
    _build_dataset(tmp_path)
    target = tmp_path / "irpin" / "flights" / "harsh" / "2022" / "truth.jsonl"
    target.write_text("{broken\n" + target.read_text(encoding="utf-8"), encoding="utf-8")
    report = check_data(tmp_path)
    assert _find(report, "truncated", "flights/harsh/2022/truth.jsonl")


def test_garbage_video_is_reported_as_truncated(tmp_path):
    _build_dataset(tmp_path)
    target = tmp_path / "irpin" / "flights" / "mild_30mps" / "2025" / "flight.mp4"
    target.write_bytes(b"definitely not an mp4 container" * 10)
    report = check_data(tmp_path)
    assert _find(report, "truncated", "flights/mild_30mps/2025/flight.mp4")


# --- the CLI (scripts/check_data.py) -----------------------------------------------------------------------------

def _cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "check_data.py"), *map(str, args)],
                          check=False, capture_output=True, text=True)


def test_cli_exits_zero_on_a_valid_tree(tmp_path):
    _build_dataset(tmp_path)
    result = _cli("--data", tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_exits_nonzero_and_names_the_file_on_a_problem(tmp_path):
    _build_dataset(tmp_path)
    (tmp_path / "hackatons-data" / "crops" / "crop_9.png").unlink()
    result = _cli("--data", tmp_path)
    assert result.returncode == 1
    assert "crop_9.png" in result.stdout


# --- on the real data/, when present ------------------------------------------------------------------------------

@pytest.mark.data
@pytest.mark.skipif(not DATA.exists(), reason="data/ not downloaded")
def test_real_data_passes_check():
    report = check_data(DATA)
    assert report.ok, [f"{p.kind}: {p.path} ({p.detail})" for p in report.problems]
