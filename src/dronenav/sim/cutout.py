"""Cut-out generator (M6-01): frames warped out of a map with a known homography, for tests and grid sweeps.

No camera model and no video (the realistic flights are data, see :mod:`dronenav.sim.flight`): a frame is the map
seen through a homography built from a pose. Conventions as in :mod:`dronenav.types`: frame px -> map px, heading
clockwise from map up with y down, scale in map px per frame px at the frame centre.

- :class:`CutoutPose`: where the frame centre lands, the heading, the scale, and an optional mild tilt (the camera
  turned ``tilt_deg`` away from straight down, towards ``tilt_dir_deg`` measured like a heading in the frame).
- :func:`cutout_homography` and :func:`render_cutout`: the pose's homography, and the frame it shows.
- :func:`cutout_route`: a smooth, seeded route of poses that keeps every footprint inside the map.
- :class:`CutoutSource`: a :class:`~dronenav.types.FrameSource` over a route, with each frame's truth.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from dronenav.io import _is_rate
from dronenav.sim.truth import FlightTruth, centre_scale
from dronenav.types import Frame

__all__ = ["CutoutPose", "CutoutSource", "cutout_homography", "cutout_route", "render_cutout"]

MAX_TURN_DEG = 25.0         # route heading change per frame, at most


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
    tilt: the nadir view seen by a camera turned about its centre (K R K^-1, focal = the frame's longer side), then
    moved and rescaled on the map so the frame centre still lands on (x, y) at ``pose.scale``."""
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
        H = H / H[2, 2]
        c = H @ [w / 2, h / 2, 1.0]
        c = c[:2] / c[2]
        k = pose.scale / centre_scale(H, w / 2, h / 2)               # the tilt changed the centre's scale: undo
        H = np.array([[k, 0, pose.x - k * c[0]], [0, k, pose.y - k * c[1]], [0, 0, 1.0]]) @ H
    return H / H[2, 2]


def render_cutout(map_img: np.ndarray, H: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    """The frame the homography H (frame px -> map px) sees. OpenCV puts pixel centres at integers and our
    convention at +0.5, hence the half-pixel shifts on both sides. Outside the map is black (honest, not invented).

    When a frame pixel covers more than one map pixel (scale > 1 at the frame centre), bilinear sampling would skip
    map detail and alias; the map is first blurred just enough (σ = 0.45·√(s² − 1), the usual anti-aliasing
    choice), and only around the footprint, so a big map stays cheap.
    """
    w, h = frame_size
    shift = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    unshift = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])
    s = centre_scale(H, w / 2, h / 2)
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
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)


def _reach(frame_size, scale_max: float, tilt_max: float) -> float:
    """How far a footprint can reach from its centre, map px, over every heading, tilt and tilt direction."""
    w, h = frame_size
    corners = np.float32([[[0, 0], [w, 0], [w, h], [0, h]]])
    far = 0.0
    for tilt in (0.0, tilt_max / 2, tilt_max):
        for d in range(0, 360, 10):
            H = cutout_homography(CutoutPose(0, 0, 0, scale_max, tilt, d), frame_size)
            far = max(far, float(np.hypot(*cv2.perspectiveTransform(corners, H)[0].T).max()))
    return far


def _check_route(n, seed, frame_size, scale, tilt_max_deg, overlap) -> None:
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1:
        raise ValueError(f"n must be a positive whole number, got {n!r}")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ValueError(  # noqa: TRY004 - a bad setting, reported like the others
            f"seed must be a whole number (a route must be reproducible), got {seed!r}")
    if not (len(scale) == 2 and all(_is_rate(v) for v in scale) and scale[0] <= scale[1]):
        raise ValueError(f"scale must be (smallest, largest), both positive, got {scale!r}")
    if not (isinstance(overlap, (int, float)) and not isinstance(overlap, bool) and 0 <= overlap < 1):
        raise ValueError(f"overlap must be in [0, 1): consecutive frames must move forward, got {overlap!r}")
    if not (isinstance(tilt_max_deg, (int, float)) and not isinstance(tilt_max_deg, bool)
            and 0 <= tilt_max_deg < 45):
        raise ValueError(f"tilt_max_deg must be in [0, 45) degrees, got {tilt_max_deg!r}")
    if not (len(frame_size) == 2 and all(isinstance(v, numbers.Integral) and v > 0 for v in frame_size)):
        raise ValueError(f"frame_size must be (w, h) in px, got {frame_size!r}")


def _heading_vec(h: float) -> np.ndarray:
    a = math.radians(h)
    return np.array([math.sin(a), -math.cos(a)])                     # heading clockwise from map up, y down


def _circle(pos, heading, step, side) -> tuple[np.ndarray, float]:
    """Centre and radius of the circle traced by turning ``side`` * MAX_TURN_DEG every frame: constant turn and
    step put every later position on it (the circumcentre of the next three positions)."""
    p1 = pos + step * _heading_vec(heading + side * MAX_TURN_DEG)
    p2 = p1 + step * _heading_vec(heading + 2 * side * MAX_TURN_DEG)
    p3 = p2 + step * _heading_vec(heading + 3 * side * MAX_TURN_DEG)
    a, b, c = p1, p2, p3
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
    uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
    centre = np.array([ux, uy])
    return centre, float(np.hypot(*(p1 - centre)))


def _safe(pos, heading, step, lo, hi) -> bool:
    """Can the route circle (at MAX_TURN_DEG a frame, left or right) from here forever without leaving the box?
    Judged on the whole circle, not on samples of it: states along one circle then all get the same answer, which
    is what makes "keep circling" a safe move from every safe state."""
    for side in (1.0, -1.0):
        c, r = _circle(pos, heading, step, side)
        if (c - r >= lo).all() and (c + r <= hi).all():
            return True
    return False


def cutout_route(map_size: tuple[int, int], n: int, seed: int, *, frame_size: tuple[int, int] = (320, 240),
                 scale: tuple[float, float] = (0.8, 1.25), tilt_max_deg: float = 8.0,
                 overlap: float = 0.65) -> list[CutoutPose]:
    """n poses along a smooth, seeded route that keeps every footprint inside the map.

    Steps are ``1 - overlap`` of a frame width, always along the heading; the heading turns at most MAX_TURN_DEG a
    frame. It wanders randomly, but each frame takes the turn closest to the random one that leaves a *safe*
    state: one from which circling at the maximum turn (left or right) stays inside the allowed box. From a safe
    state, continuing that circle keeps it safe, so the route provably never leaves the box and is never clipped
    (it bends away from edges by itself). A box narrower than that circle is refused. Scale drifts slowly within
    ``scale``; tilt is random up to ``tilt_max_deg``.
    """
    _check_route(n, seed, frame_size, scale, tilt_max_deg, overlap)
    rng = np.random.default_rng(seed)
    W, H = map_size
    m = _reach(frame_size, scale[1], tilt_max_deg) + 2.0
    lo, hi = np.array([m, m]), np.array([W - m, H - m])
    mid = (scale[0] + scale[1]) / 2
    step = (1 - overlap) * frame_size[0] * mid
    turn_d = step / math.sin(math.radians(MAX_TURN_DEG) / 2)       # turning circle's diameter at this step
    if (hi - lo).min() < turn_d + 2 * step:
        raise ValueError(f"the map ({W} x {H} px) leaves a {hi[0] - lo[0]:.0f} x {hi[1] - lo[1]:.0f} px box for "
                         f"the route: too small to turn around in ({turn_d + 2 * step:.0f} px needed)")
    # start safe by construction: on a maximum-turn circle that fits the box (the proof needs a safe start)
    r = turn_d / 2
    centre = (lo + hi) / 2 + rng.uniform(-0.5, 0.5, 2) * (hi - lo - turn_d - 2 * step).clip(0)
    side = 1.0 if rng.random() < 0.5 else -1.0
    theta = rng.uniform(0, 2 * math.pi)
    pos = centre + r * np.array([math.cos(theta), math.sin(theta)])
    tangent = side * np.array([-math.sin(theta), math.cos(theta)])        # along the circle, turning by `side`
    heading = math.degrees(math.atan2(tangent[0], -tangent[1])) % 360
    phase = rng.uniform(0, 2 * math.pi)
    candidates = np.linspace(-MAX_TURN_DEG, MAX_TURN_DEG, 21)
    poses = []
    for k in range(n):
        s = mid + (scale[1] - mid) * math.sin(phase + 0.15 * k)
        poses.append(CutoutPose(float(pos[0]), float(pos[1]), float(heading % 360), float(s),
                                float(rng.uniform(0, tilt_max_deg)), float(rng.uniform(0, 360))))
        wish = float(np.clip(rng.normal(0, 6), -MAX_TURN_DEG, MAX_TURN_DEG))
        for turn in sorted(candidates, key=lambda c: abs(c - wish)):
            nxt = pos + step * _heading_vec(heading + turn)
            if _safe(nxt, heading + turn, step, lo, hi):
                break
        else:                                    # unreachable from a safe start (see the docstring)
            raise RuntimeError(f"route seed {seed} found no safe turn at frame {k}")
        heading += turn
        pos = nxt
    return poses


class CutoutSource:
    """Frames cut out of ``map_img`` along ``poses``, at ``fps``; ``truth(k)`` is frame k's pose as a FlightTruth.
    priors: fill ``Frame.heading_prior_deg`` / ``scale_prior`` with the pose's heading and scale."""

    def __init__(self, map_img: np.ndarray, poses: list[CutoutPose], frame_size: tuple[int, int], fps: float = 1.0,
                 *, priors: bool = False):
        if not _is_rate(fps):
            raise ValueError(f"fps must be a positive number, got {fps!r}")
        self.map, self.poses, self.frame_size, self.fps = map_img, list(poses), tuple(frame_size), float(fps)
        self.priors = priors

    def __len__(self) -> int:
        return len(self.poses)

    def truth(self, k: int) -> FlightTruth:
        p, (w, h) = self.poses[k], self.frame_size
        H = cutout_homography(p, self.frame_size)
        fp = cv2.perspectiveTransform(np.float64([[[0, 0], [w, 0], [w, h], [0, h]]]), H)[0]
        return FlightTruth(i=k, t=k / self.fps, H=H, center=(p.x, p.y), footprint=fp, heading_deg=p.heading_deg)

    def __iter__(self) -> Iterator[Frame]:
        for k, p in enumerate(self.poses):
            yield Frame(idx=k, t=k / self.fps, image=render_cutout(self.map, self.truth(k).H, self.frame_size),
                        source=f"cutout#{k}", heading_prior_deg=p.heading_deg if self.priors else None,
                        scale_prior=p.scale if self.priors else None)
