"""Synthetic map and frames with exact ground truth, shared by the matcher tests (not a test module itself).

The map is seeded texture (blobs, lines and rectangles at many sizes, plus a little noise), rich in corners at
several scales, so any feature detector finds plenty of distinctive points. A frame is rendered from the map
through a known frame -> map homography, so every match can be scored against the truth exactly.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def make_map(w: int = 1200, h: int = 900, seed: int = 0, shapes: int = 900) -> np.ndarray:
    """A BGR uint8 map of seeded random shapes; same seed, same pixels. Scale ``shapes`` with the area to keep
    the density of the default 1200 x 900 map."""
    rng = np.random.default_rng(seed)
    # background: noise smoothed at several scales, like fields and roofs seen from above; corners at every
    # scale, so a frame rendered at another zoom still shares features with the map
    base = np.zeros((h, w), np.float32)
    for k, amp in ((3, 25), (9, 40), (27, 50)):
        base += amp * cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), k) * k ** 0.5
    img = np.repeat(np.clip(120 + base, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    for _ in range(shapes):
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        kind = rng.integers(3)
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        size = int(rng.integers(4, 40))
        if kind == 0:
            cv2.circle(img, (x, y), size, color, -1)
        elif kind == 1:
            cv2.rectangle(img, (x, y), (x + size, y + int(rng.integers(4, 40))), color, -1)
        else:
            cv2.line(img, (x, y), (x + int(rng.integers(-60, 60)), y + int(rng.integers(-60, 60))), color,
                     int(rng.integers(1, 5)))
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def make_h(heading_deg: float, scale: float, cx: float, cy: float, frame_size: tuple[int, int]) -> np.ndarray:
    """Frame px -> map px: the frame centre lands on (cx, cy), the frame's up is turned ``heading_deg`` clockwise
    from north, one frame px covers ``scale`` map px (the conventions of dronenav.types)."""
    fw, fh = frame_size
    a = math.radians(heading_deg)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    S = np.diag([scale, scale, 1.0])
    return np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]]) @ R @ S @ np.array([[1, 0, -fw / 2], [0, 1, -fh / 2],
                                                                             [0, 0, 1]])


def render_frame(map_img: np.ndarray, H: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """The frame a camera with frame -> map homography H would see: each frame pixel samples the map at H p.

    H is in dronenav.types' corner-origin pixels (centres at +0.5); OpenCV samples at integer centres, so it gets
    H conjugated by the half-pixel shift. Without it every frame is rendered ~0.5 px off its own truth (#106)."""
    half = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    H_cv = np.linalg.inv(half) @ H @ half
    return cv2.warpPerspective(map_img, H_cv, frame_size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REFLECT)


def apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:]
