"""Image preprocessing before matching: one function for the map and for every frame, whatever the backend (M4-05).

The map and the frame show the same ground, but not the same picture: another sensor, season, sun and exposure,
and the frame's pixels are usually finer than the map's. Three steps bring the two closer; each one can be
switched off (:class:`PreprocessConfig`), so its effect can be measured (``scripts/preprocess_ablation.py``):

- **grey-world** colour balance (optional): scale each channel so the three channel means are equal. It removes a
  global colour cast (white balance, haze). Backends that read intensity only (SIFT) barely notice it.
- **scale normalisation** (frames only): shrink the frame so one of its pixels covers about as much ground as a
  map pixel. This is the M1-03 rule from :mod:`dronenav.scale`, reused as it is (``factor = scale /
  target_scale``, never enlarged). The scale comes from the last accepted fix, or from a prior:
  :func:`scale_prior` turns the flight's ``nadir_gsd_m`` and the map's ``m_per_px`` into one.
- **CLAHE on luminance**: contrast-limited adaptive histogram equalisation of the Y channel (YCrCb), the same
  settings for map and frame. It evens out exposure and local contrast, so a detector finds the same corners in
  both. Tiles are given in pixels, not as a grid: after scale normalisation a frame pixel and a map pixel cover
  about the same ground, so the same tile size means the same patch of ground in both images. Y is exactly the
  grey that ``cv2.cvtColor(BGR2GRAY)`` gives, so a grey-based backend sees the CLAHE of its own input.

The order is grey-world, shrink, CLAHE: the balance needs the whole image's statistics, and CLAHE must run at the
final resolution for its tiles to mean the same ground as on the map.

Usage, the same for every backend::

    map_ready = preprocess(map_img, cfg).image                  # once per run; register it with the matcher
    p = preprocess(frame.image, cfg, scale=tracker.scale or prior)
    matches = p.restore(matcher.match(p.image, map_ready))      # pts_frame back in the frame's own pixels

The map is never resized: its pixels are the coordinates everything is reported in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from dronenav.scale import ScaleConfig, downscale_factor, resize_frame
from dronenav.types import Matches

__all__ = ["Prepared", "PreprocessConfig", "preprocess", "scale_prior"]


@dataclass(frozen=True)
class PreprocessConfig:
    """Which steps run, and their settings (the same object for the map and the frames).

    clahe: equalise luminance. clahe_clip: CLAHE's clip limit; 2.0 is OpenCV's usual value, higher amplifies noise.
    clahe_tile_px: side of a CLAHE tile in pixels (64 px is about 50 m of ground on the Irpin map).
    scale_norm: shrink a frame when a scale is given. scale: the shrink rule (target scale, limits), see
    :class:`dronenav.scale.ScaleConfig`. grey_world: balance the colour channels.
    """

    clahe: bool = True
    scale_norm: bool = True
    grey_world: bool = False
    clahe_clip: float = 2.0
    clahe_tile_px: int = 64
    scale: ScaleConfig = field(default_factory=ScaleConfig)

    def __post_init__(self):
        if not (math.isfinite(self.clahe_clip) and self.clahe_clip > 0):
            raise ValueError(f"clahe_clip must be finite and > 0, got {self.clahe_clip}")
        if self.clahe_tile_px < 1:
            raise ValueError(f"clahe_tile_px must be >= 1, got {self.clahe_tile_px}")


@dataclass(frozen=True, eq=False)  # numpy field: identity equality, no ambiguous array ==
class Prepared:
    """A preprocessed image and how to get back to the original's pixels.

    image: uint8, BGR or grey like the input. factor: the shrink that was asked for (1.0: not resized).
    size: (w, h) of the ORIGINAL image.
    """

    image: np.ndarray
    factor: float
    size: tuple[int, int]

    def to_original(self, pts: np.ndarray) -> np.ndarray:
        """N x 2 points in ``image`` pixels -> the original image's pixels."""
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        h, w = self.image.shape[:2]
        if (w, h) == self.size:
            return pts
        # the exact ratio of the rounded size, per axis; a pixel centre at p in the small image sits at
        # (p + 0.5) * s - 0.5 in the original (OpenCV's resize convention)
        return (pts + 0.5) * np.array([self.size[0] / w, self.size[1] / h]) - 0.5

    def restore(self, matches: Matches) -> Matches:
        """``matches`` found on ``image``, with ``pts_frame`` in the original frame's pixels, so the homography,
        the pose and the run log never see the shrink."""
        return Matches(self.to_original(matches.pts_frame), matches.pts_map, matches.scores, matches.backend,
                       matches.ms)


def scale_prior(frame_gsd_m: float, map_m_per_px: float) -> float:
    """Map px per frame px from the two ground resolutions: metres per frame pixel (a flight's ``nadir_gsd_m``,
    or altitude and field of view) over metres per map pixel (the map sidecar's ``m_per_px``)."""
    for name, v in (("frame_gsd_m", frame_gsd_m), ("map_m_per_px", map_m_per_px)):
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            raise ValueError(f"{name} must be finite and > 0, got {v!r}")
    return float(frame_gsd_m) / float(map_m_per_px)


def _checked(img) -> np.ndarray:
    ok = (isinstance(img, np.ndarray) and img.dtype == np.uint8 and img.size > 0
          and (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 3)))
    if not ok:
        raise ValueError(f"the image must be a non-empty uint8 array, H x W x 3 (BGR) or H x W (grey), got "
                         f"{getattr(img, 'shape', type(img).__name__)} {getattr(img, 'dtype', '')}".rstrip())
    return img


def _grey_world(img: np.ndarray) -> np.ndarray:
    """Each channel scaled so all three means equal their average (the overall brightness stays). A channel that
    is empty can't be scaled up from nothing and is left alone; values saturate at 255."""
    if img.ndim == 2:
        return img                                        # one channel: there is no colour to balance
    means = img.reshape(-1, 3).mean(axis=0)
    gains = np.divide(means.mean(), means, out=np.ones(3), where=means > 0)
    if np.allclose(gains, 1.0, rtol=0, atol=1e-9):
        return img
    return np.clip(img * gains, 0, 255).round().astype(np.uint8)


def _clahe(img: np.ndarray, clip: float, tile_px: int) -> np.ndarray:
    """CLAHE of the luminance, with tiles of about ``tile_px`` pixels a side."""
    h, w = img.shape[:2]
    grid = (max(1, round(w / tile_px)), max(1, round(h / tile_px)))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    if img.ndim == 2:
        return clahe.apply(img)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = clahe.apply(np.ascontiguousarray(ycc[:, :, 0]))
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def preprocess(img: np.ndarray, cfg: PreprocessConfig = PreprocessConfig(), scale: float | None = None) -> Prepared:
    """Prepare the map or a frame for matching.

    img: uint8, BGR (H x W x 3) or grey (H x W); it is not modified. scale: map px per frame px of a frame (from
    the last accepted fix or :func:`scale_prior`); None for the map, and for a frame whose scale is not known yet
    (then it keeps its size and the caller probes, see :mod:`dronenav.scale`).
    Raises ValueError for anything that is not such an image, and for a scale that is not finite and positive.
    """
    img = _checked(img)
    h, w = img.shape[:2]
    factor = 1.0
    if scale is not None:
        factor = downscale_factor(scale, cfg.scale)       # validates the scale even when the step is off
        if not cfg.scale_norm:
            factor = 1.0
    out = img
    if cfg.grey_world:
        out = _grey_world(out)
    out = resize_frame(out, factor)
    if cfg.clahe:
        out = _clahe(out, cfg.clahe_clip, cfg.clahe_tile_px)
    return Prepared(image=out, factor=factor, size=(w, h))
