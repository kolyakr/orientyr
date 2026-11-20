"""Cut-out generator (M6-01): frames warped out of a map with a known homography, for tests and grid sweeps.

No camera model and no video (the realistic flights are data, see :mod:`dronenav.sim.flight`): a frame is the map
seen through a homography built from a pose. Conventions as in :mod:`dronenav.types`: frame px -> map px, heading
clockwise from map up with y down, scale in map px per frame px.

- :class:`CutoutPose`: where the frame centre lands, the heading, the scale, and an optional mild tilt (the camera
  turned ``tilt_deg`` away from straight down, towards ``tilt_dir_deg`` measured like a heading in the frame).
- :func:`cutout_homography` and :func:`render_cutout`: the pose's homography, and the frame it shows.
- :func:`cutout_route`: a smooth, seeded route of poses that keeps every footprint inside the map.
- :class:`CutoutSource`: a :class:`~dronenav.types.FrameSource` over a route, with each frame's true homography.
"""
from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from dronenav.types import Frame

__all__ = ["CutoutPose", "CutoutSource", "cutout_homography", "cutout_route", "render_cutout"]


@dataclass(frozen=True)
class CutoutPose:
    x: float
    y: float
    heading_deg: float
    scale: float
    tilt_deg: float = 0.0
    tilt_dir_deg: float = 0.0


def cutout_homography(pose: CutoutPose, frame_size: tuple[int, int]) -> np.ndarray:
    """Frame px -> map px. Without tilt: a rotation and scale about the frame centre, which lands on (x, y). With
    tilt: the nadir view seen by a camera turned about its centre (K R K^-1, focal = the frame's longer side),
    shifted on the map so the frame centre still lands on (x, y)."""
    w, h = frame_size
    a = math.radians(pose.heading_deg)
    sR = pose.scale * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    t = np.array([pose.x, pose.y]) - sR @ np.array([w / 2, h / 2])
    H = np.array([[sR[0, 0], sR[0, 1], t[0]], [sR[1, 0], sR[1, 1], t[1]], [0.0, 0.0, 1.0]])
    if pose.tilt_deg:
        f = float(max(w, h))
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        d = math.radians(pose.tilt_dir_deg)
        axis = np.array([math.cos(d), math.sin(d), 0.0])          # in the frame plane, across the tilt direction
        R, _ = cv2.Rodrigues(axis * math.radians(pose.tilt_deg))
        H = H @ (K @ R @ np.linalg.inv(K))
        c = H @ [w / 2, h / 2, 1.0]
        c = c[:2] / c[2]
        H = np.array([[1, 0, pose.x - c[0]], [0, 1, pose.y - c[1]], [0, 0, 1.0]]) @ H
    return H / H[2, 2]


def render_cutout(map_img: np.ndarray, H: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """The frame the homography H (frame px -> map px) sees. OpenCV puts pixel centres at integers and our
    convention at +0.5, hence the half-pixel shifts on both sides.

    When a frame pixel covers more than one map pixel (scale > 1), bilinear sampling would skip map detail and
    alias; the map is first blurred just enough (σ = 0.45·√(s² − 1), the usual anti-aliasing choice), and only
    around the footprint, so a big map stays cheap.
    """
    w, h = frame_size
    shift = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    unshift = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])
    s = math.sqrt(abs(np.linalg.det(H[:2, :2] / H[2, 2])))          # map px per frame px, near the centre
    src, Hs = map_img, H
    if s > 1.05:
        corners = cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), H)[0]
        pad = 8 + 3 * s
        x0, y0 = np.maximum(np.floor(corners.min(axis=0) - pad), 0).astype(int)
        x1 = int(min(math.ceil(corners[:, 0].max() + pad), map_img.shape[1]))
        y1 = int(min(math.ceil(corners[:, 1].max() + pad), map_img.shape[0]))
        if x1 > x0 and y1 > y0:
            src = cv2.GaussianBlur(map_img[y0:y1, x0:x1], (0, 0), 0.45 * math.sqrt(s * s - 1))
            Hs = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1.0]]) @ H
    return cv2.warpPerspective(src, unshift @ Hs @ shift, (w, h),
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)


def _reach(frame_size, scale_max: float, tilt_max: float) -> float:
    """How far a footprint can reach from its centre, map px, over every heading, tilt and tilt direction."""
    w, h = frame_size
    corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
    far = 0.0
    for tilt in (0.0, tilt_max):
        for d in range(0, 360, 15):
            H = cutout_homography(CutoutPose(0, 0, 0, scale_max, tilt, d), frame_size)
            far = max(far, float(np.hypot(*cv2.perspectiveTransform(corners, H)[0].T).max()))
    return far


def cutout_route(map_size: tuple[int, int], n: int, seed: int, *, frame_size: tuple[int, int] = (320, 240),
                 scale: tuple[float, float] = (0.8, 1.25), tilt_max_deg: float = 8.0,
                 overlap: float = 0.65) -> list[CutoutPose]:
    """n poses along a smooth, seeded route that keeps every footprint inside the map.

    The heading wanders (a few degrees per frame) and, when the next three steps would leave the allowed box,
    turns towards the map centre by up to 25° per frame, so it bends away from the edge before reaching it. Steps
    are ``1 - overlap`` of a frame width; scale drifts slowly within ``scale``; tilt is random up to
    ``tilt_max_deg``.
    """
    rng = np.random.default_rng(seed)
    W, H = map_size
    m = _reach(frame_size, scale[1], tilt_max_deg) + 2.0
    if W <= 2 * m or H <= 2 * m:
        raise ValueError(f"the map ({W} x {H} px) is too small for {frame_size[0]} x {frame_size[1]} frames at scale "
                         f"up to {scale[1]:g}: a footprint needs {2 * m:.0f} px")
    lo, hi = np.array([m, m]), np.array([W - m, H - m])
    pos = rng.uniform(lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo))
    heading = rng.uniform(0, 360)
    mid = (scale[0] + scale[1]) / 2
    phase = rng.uniform(0, 2 * math.pi)
    poses = []
    for k in range(n):
        s = mid + (scale[1] - mid) * math.sin(phase + 0.15 * k)
        poses.append(CutoutPose(float(pos[0]), float(pos[1]), float(heading % 360), float(s),
                                float(rng.uniform(0, tilt_max_deg)), float(rng.uniform(0, 360))))
        step = (1 - overlap) * frame_size[0] * mid
        turn = rng.normal(0, 6)
        a = math.radians(heading + turn)
        ahead = pos + 3 * step * np.array([math.sin(a), -math.cos(a)])
        if not ((ahead >= lo).all() and (ahead <= hi).all()):     # the edge is near: bend towards the centre
            centre = (lo + hi) / 2 - pos
            to_centre = math.degrees(math.atan2(centre[0], -centre[1])) % 360
            turn = max(-25.0, min(25.0, (to_centre - heading + 180) % 360 - 180))
        heading += turn
        a = math.radians(heading)
        pos = np.clip(pos + step * np.array([math.sin(a), -math.cos(a)]), lo, hi)   # clip: a last resort only
    return poses


class CutoutSource:
    """Frames cut out of ``map_img`` along ``poses``, at ``fps``; ``truth(k)`` is frame k's homography."""

    def __init__(self, map_img: np.ndarray, poses: list[CutoutPose], frame_size: tuple[int, int], fps: float = 1.0):
        self.map, self.poses, self.frame_size, self.fps = map_img, list(poses), tuple(frame_size), float(fps)

    def __len__(self) -> int:
        return len(self.poses)

    def truth(self, k: int) -> np.ndarray:
        return cutout_homography(self.poses[k], self.frame_size)

    def __iter__(self) -> Iterator[Frame]:
        for k in range(len(self.poses)):
            yield Frame(idx=k, t=k / self.fps, image=render_cutout(self.map, self.truth(k), self.frame_size),
                        source=f"cutout#{k}")
