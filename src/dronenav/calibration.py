"""Camera calibration from checkerboard photos (M9-03): corner detection + ``cv2.calibrateCamera``.

M9-03's "calibrate the phone camera once" step: several photos of a checkerboard (the spec's
"checkerboard shown on the screen") at different angles give ``K`` and lens distortion, in the
shape :class:`~dronenav.camera.Camera` already reads (``Camera.from_meta``'s ``camera`` block).
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["calibrate", "find_corners"]

_MIN_VIEWS = 2
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
_SUBPIX_WINDOW = (11, 11)


def find_corners(image: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    """The checkerboard's ``cols`` x ``rows`` inner corners in ``image`` (BGR or grayscale),
    sub-pixel refined, (cols * rows) x 2 image px. None if no such checkerboard is found."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, (cols, rows),
                                               cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    corners = cv2.cornerSubPix(gray, corners, _SUBPIX_WINDOW, (-1, -1), _SUBPIX_CRITERIA)
    return corners.reshape(-1, 2)


def calibrate(images: list[np.ndarray], cols: int, rows: int,
              square: float = 1.0) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """``K``, ``dist`` (OpenCV's 5 coefficients) and ``(w, h)`` from checkerboard photos: the same
    camera and resolution throughout, at least ``_MIN_VIEWS`` different angles.

    ``square`` is the checkerboard square's real-world size, in whatever unit the caller wants --
    it only rescales each view's own camera position (not returned here), never ``K`` or ``dist``,
    so the arbitrary default of 1.0 is fine when only the intrinsics are wanted.

    Raises ValueError (not enough views, a view without that checkerboard, mismatched sizes, or
    ``cv2.calibrateCamera`` failing to converge) naming what's wrong, so a CLI can report it plainly.
    """
    if len(images) < _MIN_VIEWS:
        raise ValueError(f"camera calibration needs at least {_MIN_VIEWS} checkerboard views, got {len(images)}")
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square

    size: tuple[int, int] | None = None
    object_points, image_points = [], []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        if size is None:
            size = (w, h)
        elif (w, h) != size:
            raise ValueError(f"view {i} is {w} x {h}, view 0 is {size[0]} x {size[1]}: "
                             "every view must be the same camera and resolution")
        corners = find_corners(img, cols, rows)
        if corners is None:
            raise ValueError(f"view {i}: no {cols} x {rows} checkerboard found")
        object_points.append(objp)
        image_points.append(corners.reshape(-1, 1, 2).astype(np.float32))

    ok, K, dist, _, _ = cv2.calibrateCamera(object_points, image_points, size, None, None)
    if not ok:
        raise ValueError("cv2.calibrateCamera failed to converge on the given views")
    return K, dist.reshape(-1), size
