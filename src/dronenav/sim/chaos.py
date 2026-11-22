"""Chaos (M6-02): composable degradations with a strength and a seed, applied on top of any frame source.

The flights are data (their "mild" and "harsh" camera effects are baked into the video), so the stress tests
are made at run time: :class:`ChaosSource` wraps a :class:`~dronenav.types.FrameSource` (FlightSource, the
cut-out generator, a folder) and passes each frame through a chain of degradations, in order. The engine sees
ordinary Frames and cannot tell the difference.

Every degradation takes ``strength`` in [0, 1] and a ``seed``:

- strength 0 is the identity, exactly (the frame object itself comes back); the effect grows with the strength,
  and at a fixed seed a stronger setting contains the weaker one (the same box, larger; the same dropped frames,
  plus more), so sweeping the strength changes one thing only;
- all randomness is a function of (degradation, seed, frame idx), never of the order of iteration: a second
  pass, a skipped frame or another frame rate gives the same pixels for the same frame.

Image degradations (pixels only): :class:`Blur`, :class:`Noise`, :class:`Jpeg`, :class:`Gamma` vary per frame;
:class:`Glare` and :class:`Occlusion` sit at one place for the whole stream (a lamp on the screen, dirt on the
lens). Stream degradations: :class:`Drop` (the frame is missing), :class:`Jam` (a window of frames replaced by
noise), :class:`CompassError` (a bias and noise on ``Frame.heading_prior_deg``).

Sizes are relative to the frame's shorter side, so a strength looks the same on a 320 px cut-out and a 1024 px
flight frame. ``scripts/chaos_sheet.py`` draws a check sheet for each degradation.
"""
from __future__ import annotations

import dataclasses
import math
import numbers
import zlib
from collections.abc import Iterable, Iterator
from typing import ClassVar

import cv2
import numpy as np

from dronenav.fake_run import _jammed_frame  # one look for a jammed camera, in the fake run and here
from dronenav.io import FolderSource
from dronenav.types import Frame, FrameSource

__all__ = ["DEGRADATIONS", "Blur", "ChaosSource", "CompassError", "Degradation", "Drop", "Gamma", "Glare",
           "ImageDegradation", "Jam", "Jpeg", "Noise", "Occlusion", "degrade"]

# full-strength settings; lengths are fractions of the frame's shorter side
BLUR_SIGMA = 0.015          # defocus: Gaussian sigma (3.6 px on a 240 px frame)
BLUR_MOTION = 0.05          # motion: length of the smear (12 px on a 240 px frame)
NOISE_SIGMA = 40.0          # grey levels
JPEG_QUALITY = (95, 5)      # strength 0+ -> 1
GAMMA_MAX = 3.0             # exponent at full strength (or its inverse, when brightening)
GAMMA_FLICKER = 0.2         # per-frame variation of the exposure error, as a share of the strength
GLARE_RADIUS = 0.45         # the spot's radius; its inner half is burnt out
OCCLUSION_AREA = 0.5        # share of the frame the box covers


class Degradation:
    """One degradation. ``apply`` returns the degraded Frame, the same Frame (strength 0, or nothing to do), or
    None when the frame is dropped from the stream."""

    name: ClassVar[str] = ""

    def __init__(self, strength: float, seed: int = 0):
        if (isinstance(strength, bool) or not isinstance(strength, numbers.Real) or not math.isfinite(strength)
                or not 0.0 <= strength <= 1.0):
            raise ValueError(f"{self.name}: strength must be a number in [0, 1], got {strength!r}")
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
            raise ValueError(f"{self.name}: seed must be a whole number >= 0 (chaos must be reproducible), "
                             f"got {seed!r}")
        self.strength, self.seed = float(strength), int(seed)
        self._salt = zlib.crc32(self.name.encode())       # two degradations with one seed stay independent

    def _rng(self, idx: int | None = None) -> np.random.Generator:
        """The generator of frame ``idx``, or of the whole stream (None): seeded by what it is for, so it does
        not matter which frames were asked for before."""
        key = (self._salt, self.seed) if idx is None else (self._salt, self.seed, int(idx))
        return np.random.default_rng(key)

    def apply(self, frame: Frame) -> Frame | None:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(strength={self.strength:g}, seed={self.seed})"


class ImageDegradation(Degradation):
    """A degradation of the pixels only: idx, time, source and priors pass through."""

    def image(self, image: np.ndarray, idx: int = 0) -> np.ndarray:
        """The degraded copy of a BGR uint8 image (the image itself at strength 0); the input is not modified."""
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{self.name}: the image must be H x W x 3 uint8 BGR, got "
                             f"{getattr(image, 'shape', type(image))} {getattr(image, 'dtype', '')}")
        if self.strength == 0.0:
            return image
        return self._image(image, idx)

    def _image(self, image: np.ndarray, idx: int) -> np.ndarray:
        raise NotImplementedError

    def apply(self, frame: Frame) -> Frame:
        if self.strength == 0.0:
            return frame
        return dataclasses.replace(frame, image=self.image(frame.image, frame.idx))


def _short(image: np.ndarray) -> int:
    return min(image.shape[:2])


class Blur(ImageDegradation):
    """Defocus plus motion smear: a Gaussian and a line kernel, both growing with the strength. The direction of
    the smear is random per frame (the drone shakes, it does not slide one way)."""

    name = "blur"

    def _image(self, image, idx):
        side = _short(image)
        length = 1 + 2 * round(self.strength * BLUR_MOTION * side / 2)          # odd: the kernel has a centre
        if length > 1:
            a = self._rng(idx).uniform(0, math.pi)
            r = (length - 1) / 2
            kernel = np.zeros((length, length), np.float32)
            p = (round(r - r * math.cos(a)), round(r - r * math.sin(a)))
            q = (round(r + r * math.cos(a)), round(r + r * math.sin(a)))
            cv2.line(kernel, p, q, 1.0, 1)
            image = cv2.filter2D(image, -1, kernel / kernel.sum(), borderType=cv2.BORDER_REFLECT)
        return cv2.GaussianBlur(image, (0, 0), self.strength * BLUR_SIGMA * side, borderType=cv2.BORDER_REFLECT)


class Noise(ImageDegradation):
    """Sensor noise: Gaussian, independent per pixel and channel, sigma up to NOISE_SIGMA grey levels."""

    name = "noise"

    def _image(self, image, idx):
        noise = self._rng(idx).normal(0.0, self.strength * NOISE_SIGMA, image.shape)
        return np.clip(image + noise, 0, 255).astype(np.uint8)


class Jpeg(ImageDegradation):
    """Compression: a JPEG round trip, quality falling from 95 to 5. The 8 px block grid starts at a random
    offset per frame, as it does when a video link crops or scales before it compresses. Chroma is kept at full
    resolution (4:4:4): the usual 4:2:0 costs a fixed error at any quality, a jump at strength 0+ that the
    strength could not control."""

    name = "jpeg"

    def _image(self, image, idx):
        hi, lo = JPEG_QUALITY
        quality = round(hi - self.strength * (hi - lo))
        dy, dx = (int(v) for v in self._rng(idx).integers(0, 8, 2))
        padded = cv2.copyMakeBorder(image, dy, 0, dx, 0, cv2.BORDER_REPLICATE)
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality),
                  cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
        ok, buf = cv2.imencode(".jpg", padded, params)
        if not ok:                                                  # pragma: no cover - OpenCV built without JPEG
            raise RuntimeError("jpeg: OpenCV could not encode the frame")
        return np.ascontiguousarray(cv2.imdecode(buf, cv2.IMREAD_COLOR)[dy:, dx:])


class Gamma(ImageDegradation):
    """Exposure error: a gamma curve, up to GAMMA_MAX (darker) or 1 / GAMMA_MAX (brighter). ``direction`` is
    "dark", "bright", or None: one of the two for the whole stream, chosen by the seed. The amount flickers a
    little from frame to frame (auto-exposure hunting)."""

    name = "gamma"

    def __init__(self, strength: float, seed: int = 0, *, direction: str | None = None):
        super().__init__(strength, seed)
        if direction not in (None, "dark", "bright"):
            raise ValueError(f'gamma: direction must be "dark", "bright" or None, got {direction!r}')
        self.direction = direction or ("dark" if self._rng().random() < 0.5 else "bright")

    def exponent(self, idx: int) -> float:
        """The gamma applied to frame ``idx`` (> 1 darkens)."""
        amount = self.strength * (1.0 - GAMMA_FLICKER * self._rng(idx).random())
        return GAMMA_MAX ** (amount if self.direction == "dark" else -amount)

    def _image(self, image, idx):
        lut = np.round(255.0 * (np.arange(256) / 255.0) ** self.exponent(idx)).astype(np.uint8)
        return cv2.LUT(image, lut)


class Glare(ImageDegradation):
    """A glare spot: white light added around one point, burnt out in the inner half of its radius and fading to
    nothing at the radius. The point is fixed for the stream (a lamp reflected in the screen), in the middle half
    of the frame; the radius grows with the strength."""

    name = "glare"

    def centre(self, shape: tuple) -> tuple[float, float]:
        """(x, y) of the spot in a frame of ``shape`` (h, w, ...)."""
        u, v = self._rng().uniform(0.25, 0.75, 2)
        return float(u * shape[1]), float(v * shape[0])

    def _image(self, image, idx):
        h, w = image.shape[:2]
        cx, cy = self.centre(image.shape)
        radius = self.strength * GLARE_RADIUS * _short(image)
        ys, xs = np.mgrid[0:h, 0:w]
        d = np.hypot(xs + 0.5 - cx, ys + 0.5 - cy)
        light = 255.0 * np.clip(2.0 * (1.0 - d / radius), 0.0, 1.0)
        return np.clip(image + light[:, :, None], 0, 255).astype(np.uint8)


class Occlusion(ImageDegradation):
    """An occlusion box: a black rectangle of the frame's proportions over ``strength`` x OCCLUSION_AREA of the
    frame. It is centred on one seeded point for the stream and pushed inside the frame, so it grows in place."""

    name = "occlusion"

    def box(self, shape: tuple) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) of the box in a frame of ``shape`` (h, w, ...); x1 and y1 are exclusive."""
        h, w = shape[:2]
        if self.strength == 0.0:
            return 0, 0, 0, 0
        side = math.sqrt(self.strength * OCCLUSION_AREA)
        bw, bh = max(1, round(w * side)), max(1, round(h * side))
        u, v = self._rng().uniform(0.0, 1.0, 2)
        x0 = min(max(round(u * w - bw / 2), 0), w - bw)
        y0 = min(max(round(v * h - bh / 2), 0), h - bh)
        return x0, y0, x0 + bw, y0 + bh

    def _image(self, image, idx):
        x0, y0, x1, y1 = self.box(image.shape)
        out = image.copy()
        out[y0:y1, x0:x1] = 0
        return out


class Drop(Degradation):
    """Dropped frames: each frame is lost with probability ``strength`` x ``max_rate``, on its own. The frames
    that stay keep their idx and t, so the stream has gaps and the truth still lines up (nothing is renumbered)."""

    name = "drop"

    def __init__(self, strength: float, seed: int = 0, *, max_rate: float = 0.5):
        super().__init__(strength, seed)
        if isinstance(max_rate, bool) or not isinstance(max_rate, numbers.Real) or not 0.0 < max_rate <= 1.0:
            raise ValueError(f"drop: max_rate must be in (0, 1], got {max_rate!r}")
        self.rate = self.strength * float(max_rate)

    def drops(self, idx: int) -> bool:
        """Is frame ``idx`` lost? One uniform draw per frame against the rate: a higher strength loses the same
        frames and more."""
        return bool(self._rng(idx).random() < self.rate)

    def apply(self, frame: Frame) -> Frame | None:
        return None if self.drops(frame.idx) else frame


class Jam(Degradation):
    """A jam window: every frame with ``start_s <= t < start_s + strength * max_s`` arrives as noise (the camera
    link is jammed; idx, t and the priors still arrive). The window is set in seconds of flight, so it covers the
    same stretch at any frame rate. ``start_s=None`` draws the start from the seed, in [0, ``latest_start_s``]."""

    name = "jam"

    def __init__(self, strength: float, seed: int = 0, *, start_s: float | None = None, max_s: float = 10.0,
                 latest_start_s: float = 20.0):
        super().__init__(strength, seed)
        for key, v, ok in (("start_s", start_s, start_s is None or (_is_number(start_s) and start_s >= 0)),
                           ("max_s", max_s, _is_number(max_s) and max_s > 0),
                           ("latest_start_s", latest_start_s, _is_number(latest_start_s) and latest_start_s >= 0)):
            if not ok:
                raise ValueError(f"jam: {key} must be a time in seconds ({'> 0' if key == 'max_s' else '>= 0'}), "
                                 f"got {v!r}")
        self.start_s = float(self._rng().uniform(0.0, latest_start_s) if start_s is None else start_s)
        self.length_s = self.strength * float(max_s)

    def covers(self, t: float) -> bool:
        """Is the frame at time ``t`` inside the window?"""
        return self.start_s <= t < self.start_s + self.length_s

    def apply(self, frame: Frame) -> Frame:
        if not self.covers(frame.t):
            return frame
        h, w = frame.image.shape[:2]
        return dataclasses.replace(frame, image=_jammed_frame(w, h, self._rng(frame.idx)))


class CompassError(Degradation):
    """A wrong compass: ``Frame.heading_prior_deg`` gets a constant bias of ``strength`` x ``max_bias_deg`` (its
    sign comes from the seed) plus Gaussian noise of sigma ``strength`` x ``max_sigma_deg`` per frame. A frame
    without a heading prior passes through: there is no compass to be wrong."""

    name = "compass"

    def __init__(self, strength: float, seed: int = 0, *, max_bias_deg: float = 20.0, max_sigma_deg: float = 5.0):
        super().__init__(strength, seed)
        for key, v in (("max_bias_deg", max_bias_deg), ("max_sigma_deg", max_sigma_deg)):
            if not _is_number(v) or v < 0:
                raise ValueError(f"compass: {key} must be an angle >= 0 in degrees, got {v!r}")
        sign = 1.0 if self._rng().random() < 0.5 else -1.0
        self.bias_deg = self.strength * sign * float(max_bias_deg)
        self.sigma_deg = self.strength * float(max_sigma_deg)

    def error_deg(self, idx: int) -> float:
        """The error added to frame ``idx``'s heading, degrees (positive = clockwise). Linear in the strength."""
        return self.bias_deg + self.sigma_deg * float(self._rng(idx).normal())

    def apply(self, frame: Frame) -> Frame:
        if self.strength == 0.0 or frame.heading_prior_deg is None:
            return frame
        return dataclasses.replace(frame, heading_prior_deg=frame.heading_prior_deg + self.error_deg(frame.idx))


def _is_number(v) -> bool:
    return not isinstance(v, bool) and isinstance(v, numbers.Real) and math.isfinite(v)


DEGRADATIONS: dict[str, type[Degradation]] = {c.name: c for c in (Blur, Noise, Jpeg, Gamma, Glare, Occlusion, Drop,
                                                                  Jam, CompassError)}


def _chain(chain: Iterable[Degradation]) -> tuple[Degradation, ...]:
    steps = tuple(chain)
    for d in steps:
        if not isinstance(d, Degradation):
            raise TypeError(f"a chaos chain holds Degradation objects, got {d!r}")
    return steps


def degrade(frame: Frame, chain: Iterable[Degradation]) -> Frame | None:
    """``frame`` after every degradation of ``chain``, in order; None as soon as one of them drops it."""
    for d in _chain(chain):
        frame = d.apply(frame)
        if frame is None:
            return None
    return frame


class ChaosSource:
    """``source`` seen through ``chain``: a FrameSource of the same frames (idx, t, source name), degraded.

    Dropped frames are simply not yielded, and the others keep their idx, so ``truth(idx)`` and every other
    attribute of the wrapped source (``fps``, ``rate``, ``meta``, ...) are passed through unchanged.
    ``len`` is the number of frames that come out. With a Drop in the chain that needs every frame's idx: read
    from a FolderSource's file names, otherwise found by one pass over the source (cached).
    """

    def __init__(self, source: FrameSource, chain: Iterable[Degradation]):
        self.source, self.chain = source, _chain(chain)
        self._len: int | None = None

    def __iter__(self) -> Iterator[Frame]:
        for frame in self.source:
            out = degrade(frame, self.chain)
            if out is not None:
                yield out

    def __len__(self) -> int:
        drops = [d for d in self.chain if isinstance(d, Drop) and d.rate > 0]
        if not drops:
            return len(self.source)
        if self._len is None:
            idxs = self.source.idxs if isinstance(self.source, FolderSource) else [f.idx for f in self.source]
            self._len = sum(1 for i in idxs if not any(d.drops(i) for d in drops))
        return self._len

    def __getattr__(self, name: str):
        # only reached for names this class lacks; private names are never forwarded (and copy/pickle probe them
        # before ``source`` exists)
        if name.startswith("_") or name in ("source", "chain"):
            raise AttributeError(name)
        return getattr(self.source, name)
