"""Verify a backups/ folder has every recording the pitch needs as a stand-in for a live step (M8-05).

"Done when: every live step in the pitch has a matching recording, on two machines." This checks the first half
(a recording exists and actually decodes) on one machine; run scripts/check_backups.py on the second machine too
- see README.md, "Backup recordings".

The 21 Nov re-plan (#128) cut the live judge-controlled UI demo; the three live steps it replaced are the HUD
videos of the 50-crop task map, an Irpin flight, and the phone take.
"""
from __future__ import annotations

from pathlib import Path

import cv2

from dronenav.check_data import CheckReport, Problem

__all__ = ["REQUIRED", "check_backups"]

REQUIRED = ("crops.mp4", "irpin.mp4", "phone.mp4")


def check_backups(backups_dir: str | Path) -> CheckReport:
    """Check that each of REQUIRED exists in ``backups_dir`` and decodes at least one frame. Never raises:
    problems go into the returned report."""
    backups_dir = Path(backups_dir)
    report = CheckReport(data_dir=backups_dir)
    if not backups_dir.is_dir():
        report.problems.append(Problem(backups_dir.name, "missing", "directory not found"))
        return report
    for name in REQUIRED:
        _check_video(report, backups_dir / name)
    return report


def _check_video(report: CheckReport, path: Path) -> None:
    if not path.is_file():
        report.problems.append(Problem(path.name, "missing", "file not found"))
        return
    if path.stat().st_size == 0:
        report.problems.append(Problem(path.name, "truncated", "empty file"))
        return
    report.checked += 1
    cap = cv2.VideoCapture(str(path))
    # a decoded frame is a far more reliable truncation signal than CAP_PROP_FRAME_COUNT, which some
    # codecs/containers misreport as 0 or -1 for perfectly good files (dronenav.check_data._check_video)
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        report.problems.append(Problem(path.name, "truncated", "cannot decode a frame from the video"))
