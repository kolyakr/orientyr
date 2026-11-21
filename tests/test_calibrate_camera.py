"""M9-03 · scripts/calibrate_camera.py: once-off camera calibration from checkerboard photos.

A phone's K and lens distortion are found the usual way: several photos of a checkerboard (here,
shown full-screen -- the spec's "checkerboard on the screen") at different angles, corner
detection, ``cv2.calibrateCamera``. Ground truth we control: render the checkerboard through a
few known pinhole poses (same camera model as test_perspective.py's own M2-04 test helper), so
the true K is known exactly and the recovered one can be checked against it.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from calibrate_camera import calibrate, find_corners, main  # noqa: E402

from dronenav.camera import Camera  # noqa: E402

COLS, ROWS = 9, 6           # inner corners
SQ_PX = 80                  # one board square, board-image px (== the "square" unit passed to calibrate)
FRAME = (1280, 960)
HFOV = 60.0
TRUE_FOCAL = (FRAME[0] / 2) / math.tan(math.radians(HFOV) / 2)


def _board() -> np.ndarray:
    """A flat (cols+1) x (rows+1)-square checkerboard, white-bordered, as a grayscale image."""
    sx, sy = COLS + 1, ROWS + 1
    board = np.zeros((sy * SQ_PX, sx * SQ_PX), np.uint8)
    for r in range(sy):
        for c in range(sx):
            if (r + c) % 2 == 0:
                board[r * SQ_PX:(r + 1) * SQ_PX, c * SQ_PX:(c + 1) * SQ_PX] = 255
    pad = SQ_PX
    return cv2.copyMakeBorder(board, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def _camera_h(yaw, pitch, roll, cx, cy, alt, hfov, frame):
    """Frame px -> board px of a pinhole camera (same convention as test_perspective.py's camera_h)."""
    w, h = frame
    f = (w / 2) / math.tan(math.radians(hfov) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(r), 0, math.sin(r)], [0, 1, 0], [-math.sin(r), 0, math.cos(r)]])
    R_cw = Rz @ Rx @ Ry
    C = np.array([cx, cy, -alt])
    R = R_cw.T
    t = -R @ C
    H_m2f = K @ np.c_[R[:, 0], R[:, 1], t]
    H = np.linalg.inv(H_m2f)
    return H / H[2, 2]


def _render(board: np.ndarray, H: np.ndarray, frame: tuple[int, int]) -> np.ndarray:
    """The frame a camera with board -> frame... (frame px -> board px) homography H would see."""
    w, h = frame
    half = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    H_cv = np.linalg.inv(half) @ H @ half
    return cv2.warpPerspective(board, H_cv, frame, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def synthetic_views(poses=((0, 0, 0), (18, 0, 0), (-15, 10, 0), (0, -12, 14), (10, -8, -10))):
    """A handful of checkerboard photos from known (yaw, pitch, roll), centred on the board."""
    board = _board()
    bh, bw = board.shape
    cx, cy = bw / 2, bh / 2
    alt = (bw / 2) / math.tan(math.radians(HFOV) / 2) * 1.15     # the whole board stays in frame
    views = []
    for yaw, pitch, roll in poses:
        H = _camera_h(yaw, pitch, roll, cx, cy, alt, HFOV, FRAME)
        views.append(_render(board, H, FRAME))
    return views


# --- calibrate() on known synthetic views --------------------------------------------------

def test_calibration_recovers_the_true_focal_length():
    K, dist, size = calibrate(synthetic_views(), COLS, ROWS, square=SQ_PX)
    assert size == FRAME
    # a handful of noise-free synthetic views, no real lens distortion to fight: a tight relative
    # tolerance is appropriate (checkerboard corner detection itself is sub-pixel accurate)
    assert K[0, 0] == pytest.approx(TRUE_FOCAL, rel=0.03)
    assert K[1, 1] == pytest.approx(TRUE_FOCAL, rel=0.03)
    assert K[0, 2] == pytest.approx(FRAME[0] / 2, rel=0.03)
    assert K[1, 2] == pytest.approx(FRAME[1] / 2, rel=0.03)


def test_calibration_output_loads_as_a_camera_block():
    K, dist, size = calibrate(synthetic_views(), COLS, ROWS, square=SQ_PX)
    camera = Camera(K, dist, size)
    assert camera.size == FRAME


def test_find_corners_locates_the_checkerboard():
    view = synthetic_views(((0, 0, 0),))[0]
    corners = find_corners(view, COLS, ROWS)
    assert corners is not None
    assert corners.shape == (COLS * ROWS, 2)


def test_find_corners_is_none_without_a_checkerboard():
    blank = np.full(FRAME[::-1], 128, np.uint8)
    assert find_corners(blank, COLS, ROWS) is None


def test_calibrate_rejects_a_view_with_no_checkerboard():
    views = synthetic_views(((0, 0, 0),)) + [np.full(FRAME[::-1], 128, np.uint8)]
    with pytest.raises(ValueError, match="checkerboard"):
        calibrate(views, COLS, ROWS, square=SQ_PX)


def test_calibrate_needs_at_least_two_views():
    with pytest.raises(ValueError, match="at least"):
        calibrate(synthetic_views(((0, 0, 0),)), COLS, ROWS, square=SQ_PX)


# --- the CLI: writes a camera.json Camera.from_meta can read straight away --------------------

def test_cli_writes_a_camera_json(tmp_path):
    out = tmp_path / "camera.json"
    paths = []
    for i, view in enumerate(synthetic_views()):
        p = tmp_path / f"view_{i}.png"
        cv2.imwrite(str(p), view)
        paths.append(p)
    main(["--images", str(tmp_path / "view_*.png"), "--cols", str(COLS), "--rows", str(ROWS),
         "--square", str(SQ_PX), "--out", str(out)])
    meta = json.loads(out.read_text())
    camera = Camera.from_meta(meta)
    assert camera is not None
    assert camera.size == FRAME


def test_cli_reports_no_images_found(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["--images", str(tmp_path / "nothing_*.png"), "--cols", str(COLS), "--rows", str(ROWS),
             "--out", str(tmp_path / "camera.json")])
    assert "no images" in capsys.readouterr().err.lower()
