"""Verify a downloaded data/ folder against the layout the team's Google Drive mirrors (M0-06).

"Done when: a fresh clone plus the Drive folder unpacked into data/ passes scripts/check_data.py; a missing or
truncated file is reported by name." See data/README.md for the layout this checks.

A missing directory that should hold many expected files (an undownloaded irpin/, a whole flight folder, ...)
is reported once, not once per file that would have been inside it - the point is to find the one bad file in
an otherwise-complete download, not to flood the output on a bare clone.

The Irpin map sidecars are checked with ``dronenav.georef.load_georef`` and ``dronenav.io.load_map`` - the same
code the rest of the pipeline uses to read them - rather than a second, looser parser: a sidecar missing a
field those modules require, or a map image whose pixel size no longer matches its sidecar, is exactly the
kind of "truncated" problem this checker exists to catch before a real run hits it instead.
"""
from __future__ import annotations

import json
import math
import numbers
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from dronenav import io as _io
from dronenav.georef import GeorefError, load_georef

__all__ = ["CheckReport", "Problem", "check_data"]

HACKATHON_CROPS = 50
IRPIN_YEARS = ("2017", "2022", "2025")
FLIGHT_PRESETS = ("mild", "mild_30mps", "harsh", "harsh_30mps")


@dataclass(frozen=True)
class Problem:
    path: str       # forward-slash path, relative to the data/ folder
    kind: str       # "missing" or "truncated"
    detail: str     # human-readable reason, for the printed report


@dataclass
class CheckReport:
    data_dir: Path
    checked: int = 0
    problems: list[Problem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def check_data(data_dir: str | Path) -> CheckReport:
    """Check ``data_dir`` against the layout in data/README.md: file presence, counts tied to each flight's own
    meta.json / truth.json (not a hard-coded frame count), and that every sidecar, meta.json and truth.jsonl
    parses. Never raises: problems go into the returned report."""
    data_dir = Path(data_dir)
    report = CheckReport(data_dir=data_dir)
    if not _require_dir(report, data_dir, data_dir):
        return report
    if _require_dir(report, data_dir / "hackatons-data", data_dir):
        _check_hackathon(report, data_dir)
    if _require_dir(report, data_dir / "irpin", data_dir):
        _check_irpin(report, data_dir)
    return report


# --- the organisers' task data -----------------------------------------------------------------------------------

def _check_hackathon(report: CheckReport, data_dir: Path) -> None:
    root = data_dir / "hackatons-data"
    _check_image(report, root / "global_map.png", data_dir)
    if _require_dir(report, root / "crops", data_dir):
        for i in range(HACKATHON_CROPS):
            _check_image(report, root / "crops" / f"crop_{i}.png", data_dir)


# --- our Irpin showcase data -------------------------------------------------------------------------------------

def _check_irpin(report: CheckReport, data_dir: Path) -> None:
    irpin = data_dir / "irpin"
    if _require_dir(report, irpin / "maps", data_dir):
        for year in IRPIN_YEARS:
            _check_map_year(report, irpin, year, data_dir)
    _check_json(report, irpin / "unosat.geojson", data_dir)
    _check_json(report, irpin / "source" / "unosat_ukraine_zenodo15088349.geojson", data_dir)
    _require_dir(report, irpin / "preview", data_dir)
    if _require_dir(report, irpin / "flights", data_dir):
        for preset in FLIGHT_PRESETS:
            for year in IRPIN_YEARS:
                _check_flight(report, irpin / "flights" / preset / year, data_dir)
    if _require_dir(report, irpin / "drone", data_dir):
        for year in IRPIN_YEARS:
            _check_drone(report, irpin / "drone" / year, data_dir)


def _check_map_year(report: CheckReport, irpin: Path, year: str, data_dir: Path) -> None:
    """The sidecar via georef.load_georef (full schema: required fields, no NaN/Infinity, both blocks present),
    then each layer's image via io.load_map (decodes it and cross-checks its pixel size against the sidecar)."""
    sidecar_path = irpin / "maps" / f"irpin_{year}.json"
    map_path = irpin / "maps" / f"irpin_{year}.png"
    georef = None
    if _require_file(report, sidecar_path, data_dir):
        report.checked += 1
        try:
            georef = load_georef(sidecar_path)
        except (OSError, GeorefError) as e:
            report.problems.append(Problem(_rel(sidecar_path, data_dir), "truncated", str(e)))
    _check_map_image(report, map_path, data_dir)
    if georef is not None and georef.hires.file != georef.map.file:   # 2017 has no separate hires layer
        _check_map_image(report, irpin / georef.hires.file, data_dir)


def _check_map_image(report: CheckReport, path: Path, data_dir: Path) -> None:
    if not _require_file(report, path, data_dir):
        return
    report.checked += 1
    try:
        _io.load_map(path)
    except (FileNotFoundError, ValueError) as e:
        report.problems.append(Problem(_rel(path, data_dir), "truncated", str(e)))


def _check_flight(report: CheckReport, flight: Path, data_dir: Path) -> None:
    if not _require_dir(report, flight, data_dir):
        return
    meta_path = flight / "meta.json"
    meta = _check_json(report, meta_path, data_dir)
    if isinstance(meta, dict) and "fps" in meta and not _is_rate(meta["fps"]):
        report.problems.append(Problem(_rel(meta_path, data_dir), "truncated",
                                       f"fps must be a positive, finite number, got {meta['fps']!r}"))
    expect_lines = meta.get("frames") if isinstance(meta, dict) else None
    _check_jsonl(report, flight / "truth.jsonl", data_dir, expect_lines=expect_lines)
    _check_video(report, flight / "flight.mp4", data_dir)
    for kf in (meta or {}).get("keyframes", []):
        name = kf.get("file") if isinstance(kf, dict) else None
        if not name:
            report.problems.append(Problem(_rel(meta_path, data_dir), "truncated",
                                           f"keyframes entry missing \"file\": {kf!r}"))
            continue
        _check_image(report, flight / name, data_dir)


def _check_drone(report: CheckReport, drone: Path, data_dir: Path) -> None:
    if not _require_dir(report, drone, data_dir):
        return
    truth_path = drone / "truth.json"
    truth = _check_json(report, truth_path, data_dir)
    for f in (truth or {}).get("frames", []):
        name = f.get("frame") if isinstance(f, dict) else None
        if not name:
            report.problems.append(Problem(_rel(truth_path, data_dir), "truncated",
                                           f"frames entry missing \"frame\": {f!r}"))
            continue
        _check_image(report, drone / name, data_dir)


def _is_rate(v) -> bool:
    """A positive, finite number; not a bool (True would mean 1 fps). Mirrors dronenav.io's own fps check."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        return False
    try:
        return math.isfinite(v) and v > 0
    except OverflowError:
        return False


# --- file-level checks: each appends at most one problem, by the file's own name -------------------------------

def _rel(path: Path, data_dir: Path) -> str:
    try:
        rel = path.resolve().relative_to(data_dir.resolve())
    except ValueError:
        rel = path
    return rel.as_posix()


def _require_dir(report: CheckReport, path: Path, data_dir: Path) -> bool:
    if not path.is_dir():
        report.problems.append(Problem(_rel(path, data_dir), "missing", "directory not found"))
        return False
    return True


def _require_file(report: CheckReport, path: Path, data_dir: Path) -> bool:
    if not path.is_file():
        report.problems.append(Problem(_rel(path, data_dir), "missing", "file not found"))
        return False
    if path.stat().st_size == 0:
        report.problems.append(Problem(_rel(path, data_dir), "truncated", "empty file"))
        return False
    return True


def _check_image(report: CheckReport, path: Path, data_dir: Path) -> None:
    if not _require_file(report, path, data_dir):
        return
    report.checked += 1
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or 0 in image.shape[:2]:
        report.problems.append(Problem(_rel(path, data_dir), "truncated", "cannot decode image"))


def _check_video(report: CheckReport, path: Path, data_dir: Path) -> None:
    if not _require_file(report, path, data_dir):
        return
    report.checked += 1
    cap = cv2.VideoCapture(str(path))
    # a decoded frame is a far more reliable truncation signal than CAP_PROP_FRAME_COUNT, which some
    # codecs/containers misreport as 0 or -1 for perfectly good files
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        report.problems.append(Problem(_rel(path, data_dir), "truncated", "cannot decode a frame from the video"))


def _check_json(report: CheckReport, path: Path, data_dir: Path) -> dict | None:
    if not _require_file(report, path, data_dir):
        return None
    report.checked += 1
    try:
        data = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        report.problems.append(Problem(_rel(path, data_dir), "truncated", f"invalid JSON: {e}"))
        return None
    if not isinstance(data, dict):
        report.problems.append(Problem(_rel(path, data_dir), "truncated", "JSON is not an object"))
        return None
    return data


def _check_jsonl(report: CheckReport, path: Path, data_dir: Path, *, expect_lines: int | None) -> None:
    if not _require_file(report, path, data_dir):
        return
    report.checked += 1
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    for i, ln in enumerate(lines, start=1):
        try:
            json.loads(ln)
        except json.JSONDecodeError as e:
            report.problems.append(Problem(_rel(path, data_dir), "truncated", f"line {i}: invalid JSON: {e}"))
            return
    if expect_lines is not None and len(lines) != expect_lines:
        report.problems.append(Problem(_rel(path, data_dir), "truncated",
                                       f"{len(lines)} lines, meta.json says {expect_lines}"))
