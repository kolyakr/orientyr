"""Camera model: intrinsics K and lens distortion, and undistortion of frames and points (M2-04).

A real lens bends straight lines (barrel distortion), and a homography only holds between straight-line
(pinhole) images. So a frame is undistorted with its camera's K and ``dist`` before matching: the Irpin flights'
truth ``H_frame_to_map`` is for undistorted pixels ("cv2.undistortPoints with K, dist, P=K", their meta.json).
Undistorting keeps K (P = K), so the image size, centre and focal stay those of the original frame.

The model is OpenCV's (k1, k2, p1, p2[, k3...]), in OpenCV's pixel convention (centres at integers): K and
dist come from tools that use it, so the conversion stays inside cv2.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

__all__ = ["Camera"]


class Camera:
    """K (3 x 3, frame px), dist (OpenCV's coefficients) and the frame size (width, height) they belong to."""

    def __init__(self, K, dist, size: tuple[int, int]):
        self.K = np.asarray(K, dtype=np.float64)
        self.dist = np.asarray(dist, dtype=np.float64).ravel()
        self.size = (int(size[0]), int(size[1]))
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all():
            raise ValueError(f"K must be a finite 3 x 3 matrix, got shape {self.K.shape}")
        if self.dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(self.dist).all():
            raise ValueError(f"dist must hold 4, 5, 8, 12 or 14 finite coefficients, got {self.dist.size}")
        if min(self.size) <= 0:
            raise ValueError(f"size must be positive, got {self.size}")
        self._maps = None                    # remap tables, built on the first frame

    @classmethod
    def from_meta(cls, meta: dict | str | Path) -> Camera | None:
        """The ``camera`` block of a flight's meta.json (a dict or the file's path): ``K``, ``dist``, ``size``.
        None when there is no camera block (the task crops, a phone video without calibration)."""
        if not isinstance(meta, dict):
            meta = json.loads(Path(meta).read_text())
        cam = meta.get("camera")
        if not cam:
            return None
        return cls(cam["K"], cam.get("dist", [0, 0, 0, 0, 0]), tuple(cam["size"]))

    @property
    def focal(self) -> float:
        """Focal length in frame px (the mean of fx and fy)."""
        return float(self.K[0, 0] + self.K[1, 1]) / 2

    def undistort_points(self, pts) -> np.ndarray:
        """N x 2 distorted frame px -> where a pinhole camera with the same K puts them."""
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.undistortPoints(p, self.K, self.dist, P=self.K).reshape(-1, 2)

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        """The frame as the pinhole camera K would have seen it: same size, straight lines straight."""
        h, w = img.shape[:2]
        if (w, h) != self.size:
            raise ValueError(f"the camera is calibrated for {self.size[0]} x {self.size[1]} frames, "
                             f"got {w} x {h}")
        if self._maps is None:
            self._maps = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, self.size, cv2.CV_32FC1)
        return cv2.remap(img, *self._maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
