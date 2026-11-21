"""M8-05 · Backup recordings.

"Done when: every live step in the pitch has a matching recording, on two machines." The checkable half of
that: a backups/ folder has the three required recordings (crops, Irpin flight, phone take), each a video that
actually decodes - the "on two machines" half is a human running scripts/check_backups.py on each one before
the pitch (see README.md, "Backup recordings").

Mirrors tests/test_check_data.py: tiny synthetic mp4s, so CI needs no real recordings.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from dronenav.check_backups import REQUIRED, check_backups

ROOT = Path(__file__).resolve().parents[1]


def _mp4(path: Path, frames: int = 3, size: tuple[int, int] = (16, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, size)
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.zeros((size[1], size[0], 3), np.uint8))
    writer.release()


def _build_backups(root: Path) -> None:
    for name in REQUIRED:
        _mp4(root / name)


def test_a_complete_backups_folder_passes(tmp_path):
    _build_backups(tmp_path)
    report = check_backups(tmp_path)
    assert report.ok and report.checked == len(REQUIRED) and report.problems == []


def test_a_missing_recording_is_reported_by_name(tmp_path):
    _build_backups(tmp_path)
    (tmp_path / REQUIRED[0]).unlink()
    report = check_backups(tmp_path)
    assert not report.ok
    assert [p.path for p in report.problems] == [REQUIRED[0]]
    assert report.problems[0].kind == "missing"


def test_an_empty_file_is_reported_truncated_not_missing(tmp_path):
    _build_backups(tmp_path)
    (tmp_path / REQUIRED[1]).write_bytes(b"")
    report = check_backups(tmp_path)
    assert [p.path for p in report.problems] == [REQUIRED[1]]
    assert report.problems[0].kind == "truncated"


def test_a_file_that_cannot_decode_a_frame_is_reported_truncated(tmp_path):
    _build_backups(tmp_path)
    (tmp_path / REQUIRED[2]).write_bytes(b"not actually a video")
    report = check_backups(tmp_path)
    assert [p.path for p in report.problems] == [REQUIRED[2]]
    assert report.problems[0].kind == "truncated"


def test_a_missing_backups_folder_is_its_own_problem(tmp_path):
    report = check_backups(tmp_path / "nowhere")
    assert not report.ok and report.checked == 0
    assert len(report.problems) == 1 and report.problems[0].kind == "missing"


def _cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "check_backups.py"), *map(str, args)],
                          check=False, capture_output=True, text=True)


def test_the_cli_exits_ok_on_a_complete_folder(tmp_path):
    _build_backups(tmp_path)
    res = _cli("--backups", tmp_path)
    assert res.returncode == 0, res.stderr
    assert "ok" in res.stdout


def test_the_cli_names_every_missing_file_and_exits_1(tmp_path):
    _build_backups(tmp_path)
    (tmp_path / REQUIRED[0]).unlink()
    res = _cli("--backups", tmp_path)
    assert res.returncode == 1
    assert REQUIRED[0] in res.stdout and "missing" in res.stdout


def test_the_cli_exits_2_on_a_missing_backups_folder(tmp_path):
    res = _cli("--backups", tmp_path / "nowhere")
    assert res.returncode == 2 and "Traceback" not in res.stderr
