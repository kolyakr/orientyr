"""Quality gate (M2-03): decide, before any matching, which frames are worth matching and which pixels of them.

A phone filming a screen (and a shaking drone camera) gives frames the matcher should never see: smeared ones,
whose few matches fit a wrong homography, and ones with a lamp reflected in the screen. :class:`QualityGate` wraps
any :class:`~dronenav.types.FrameSource` and passes on the good frames only, untouched; every rejected frame is
logged with a stable reason (:data:`REASONS`). Four checks, in this order:

- ``too_small``: the frame's shorter side is under ``min_side`` px, there is nothing to judge;
- ``blank``: no contrast at all (black-out, lens covered, burnt out): the grey levels' std is under ``blank_std``;
- ``glare``: more than ``max_glare`` of the frame is glare. :func:`glare_mask` finds the saturated blobs without
  texture; a frame with less glare is kept, and its mask says which pixels the matcher may use;
- ``blur``: :func:`blur_score` (variance of the Laplacian over the variance of the image, on the pixels outside
  the glare) is under ``blur_ratio`` x the median score of the last ``window`` frames (:class:`RunningMedian`).

The blur rule is relative on purpose. A sharp frame of a field scores lower than a smeared frame of a town, so
no absolute threshold works on a whole flight; a smear is a sudden fall against the frames just before it. The
price: a flight smeared from start to end passes (nothing to compare with), and a smear longer than half the
window is taken for a new scene. Dividing by the image variance makes the score blind to exposure: half the
light quarters the Laplacian's variance and the image's alike.

:func:`damp_moire` is the third tool: a light Gaussian against the screen's pixel grid before the frame is
shrunk for matching, and an optional notch on the peaks that grid leaves in the spectrum. Off by default in the
gate (``moire_sigma = 0``): it is for filmed screens, not for the drone flights.

    gate = QualityGate(VideoSource("take.mp4", fps=12), QualityConfig(moire_sigma=1.0))
    for frame in gate:                 # kept frames only; idx and t keep their gaps, like a Drop in the chaos
        usable = gate.last.mask        # None, or uint8: nonzero where the frame's pixels are not glare
    gate.rejected                      # the verdicts of the frames that did not pass
"""
from __future__ import annotations

import dataclasses
import logging
import math
import numbers
import statistics
from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import numpy as np

from dronenav.types import Frame, FrameSource

__all__ = ["REASONS", "FrameQuality", "QualityConfig", "QualityGate", "RunningMedian", "Verdict", "blur_score",
           "damp_moire", "glare_mask"]

log = logging.getLogger(__name__)

# Why a frame was rejected, in the order the checks run. Logs, sheets and tests rely on these names.
REASONS = ("too_small", "blank", "glare", "blur")

BLUR_SIDE = 256            # the shorter side the blur score is measured at, px
PRE_SIGMA = 0.7            # px at that size: the smoothing before the Laplacian, see blur_score
# The notch looks for the screen grid above this frequency only (periods under 25 px): below it the scene's own
# structure (streets, field borders) owns the spectrum and must not be cut.
NOTCH_MIN_FREQ = 0.04      # cycles per px
NOTCH_RATIO = 8.0          # a peak is this many times the mean magnitude around it
NOTCH_NEIGHBOURHOOD = 31   # bins: the box that mean is taken over
NOTCH_RADIUS = 2           # bins cut around each peak (the main lobe of a grid between two bins)
NOTCH_MAX_SHARE = 0.01     # more of the spectrum than this "peaking" is texture, not a grid: cut nothing


def _is_number(v) -> bool:
    return not isinstance(v, bool) and isinstance(v, numbers.Real) and math.isfinite(v)


def _is_count(v) -> bool:
    return not isinstance(v, bool) and isinstance(v, numbers.Integral) and v >= 1


def _checked(image, what: str) -> np.ndarray:
    """``image`` if it is a uint8 image, H x W (grey) or H x W x 3 (BGR), with at least one pixel."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0
            or not (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3))):
        raise ValueError(f"{what}: the image must be uint8, H x W or H x W x 3 (BGR), got "
                         f"{getattr(image, 'shape', type(image).__name__)} {getattr(image, 'dtype', '')}")
    return image


def _grey(image: np.ndarray) -> np.ndarray:
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _working(grey: np.ndarray, side: int) -> np.ndarray:
    """``grey`` shrunk so its shorter side is ``side`` px (never enlarged), as float32."""
    short = min(grey.shape)
    if short > side:
        s = side / short
        grey = cv2.resize(grey, (max(1, round(grey.shape[1] * s)), max(1, round(grey.shape[0] * s))),
                          interpolation=cv2.INTER_AREA)
    return grey.astype(np.float32)


# --- blur ----------------------------------------------------------------------------------------------------------

def blur_score(image: np.ndarray, mask: np.ndarray | None = None, *, side: int = BLUR_SIDE) -> float:
    """Sharpness of ``image``: the variance of its (lightly smoothed) Laplacian over the variance of its grey levels. Higher is
    sharper; 0 for a frame with no contrast or too small to have a Laplacian.

    The frame is first shrunk so its shorter side is ``side`` px. The matcher shrinks frames too, so a smear that
    disappears at that size costs it nothing; and the score of a take does not depend on the camera's resolution.
    ``mask`` (uint8, the image's size, nonzero = use) limits the score to those pixels: the frame without its
    glare. The value means something only against other frames of the same scene (see :class:`RunningMedian`).
    """
    grey = _grey(_checked(image, "blur_score"))
    if mask is not None and np.shape(mask) != grey.shape:
        raise ValueError(f"blur_score: mask must be {grey.shape} like the image, got {np.shape(mask)}")
    small = _working(grey, side)
    if min(small.shape) < 3:                        # smaller than the Laplacian's kernel
        return 0.0
    # A 0.7 px Gaussian first. A heavily smeared uint8 frame still has its rounding noise (+-0.5 grey level per
    # pixel), which the bare Laplacian reads as detail: without this the score of a smear stops falling.
    lap = cv2.Laplacian(cv2.GaussianBlur(small, (0, 0), PRE_SIGMA, borderType=cv2.BORDER_REFLECT), cv2.CV_32F,
                        borderType=cv2.BORDER_REFLECT)
    if mask is None:
        values, second = small, lap
    else:
        use = cv2.resize(np.asarray(mask, np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        # one px in from the mask's edge: there the Laplacian still sees the masked-off pixels
        use = cv2.erode(use, np.ones((3, 3), np.uint8)) > 0
        if use.sum() < 9:
            return 0.0
        values, second = small[use], lap[use]
    contrast = float(values.var())
    if contrast < 1e-6:                             # a flat frame: no contrast to be sharp or smeared
        return 0.0
    return float(second.var()) / contrast


class Verdict(NamedTuple):
    """What :meth:`RunningMedian.judge` says about one score. ``median`` and ``threshold`` are None while there
    is no history to compare with (the score is kept)."""

    kept: bool
    median: float | None
    threshold: float | None


class RunningMedian:
    """The blur rule: a score under ``ratio`` x the median of the last ``window`` scores is a smear.

    Every judged score enters the history, the rejected ones too. That is what lets the median follow the flight
    from a town to a field: with the kept scores only, the first low score would lock the threshold for good.
    Until ``min_history`` scores are in, every score is kept: there is no median yet.
    """

    def __init__(self, window: int = 15, ratio: float = 0.4, min_history: int = 3):
        if not _is_count(window):
            raise ValueError(f"window must be a whole number of frames >= 1, got {window!r}")
        if not _is_number(ratio) or not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1): a share of the median score, got {ratio!r}")
        if not _is_count(min_history) or min_history > window:
            raise ValueError(f"min_history must be a whole number in 1..window ({window}), got {min_history!r}")
        self.window, self.ratio, self.min_history = int(window), float(ratio), int(min_history)
        self._scores: deque[float] = deque(maxlen=self.window)

    def reset(self) -> None:
        """Forget the history: a new stream starts."""
        self._scores.clear()

    def judge(self, score: float) -> Verdict:
        """Compare ``score`` with the scores before it, then add it to the history."""
        if not _is_number(score) or score < 0:
            raise ValueError(f"a blur score is a finite number >= 0, got {score!r}")
        verdict = Verdict(True, None, None)
        if len(self._scores) >= self.min_history:
            median = float(statistics.median(self._scores))
            threshold = self.ratio * median
            verdict = Verdict(score >= threshold, median, threshold)
        self._scores.append(float(score))
        return verdict


# --- glare ---------------------------------------------------------------------------------------------------------

def glare_mask(image: np.ndarray, *, level: int = 240, texture: float = 3.0, min_area: float = 0.002,
               margin: float = 0.01) -> np.ndarray:
    """The glare in ``image``: uint8, H x W, 255 on glare and 0 elsewhere.

    Glare is a blob that is saturated (every channel >= ``level``: white light, not a red roof) and has no
    texture (local std of the grey levels <= ``texture``). A bright roof or a sunlit car park keeps its detail and
    stays; a burnt-out reflection has none. Blobs under ``min_area`` of the frame are ignored (hot pixels, small
    highlights cost the matcher nothing). Each blob is grown by ``margin`` x the shorter side: the features on its
    rim are damaged too. Invert it (``cv2.bitwise_not``) for a matcher-style mask of the usable pixels.
    """
    image = _checked(image, "glare_mask")
    for name, v, ok in (("level", level, _is_number(level) and 0 <= level <= 255),
                        ("texture", texture, _is_number(texture) and texture >= 0),
                        ("min_area", min_area, _is_number(min_area) and 0 <= min_area <= 1),
                        ("margin", margin, _is_number(margin) and 0 <= margin <= 1)):
        if not ok:
            raise ValueError(f"glare_mask: bad {name} {v!r}")
    h, w = image.shape[:2]
    bright = (image if image.ndim == 2 else image.min(axis=2)) >= level
    if not bright.any():
        return np.zeros((h, w), np.uint8)
    grey = _grey(image).astype(np.float32)
    k = max(3, round(0.02 * min(h, w)) | 1)                     # odd window, about 2 % of the shorter side
    mean = cv2.blur(grey, (k, k), borderType=cv2.BORDER_REFLECT)
    var = cv2.blur(grey * grey, (k, k), borderType=cv2.BORDER_REFLECT) - mean * mean
    flat = var <= texture * texture
    blobs = cv2.morphologyEx((bright & flat).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blobs, connectivity=8)
    big = np.zeros(n, bool)
    big[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(1.0, min_area * h * w)
    out = big[labels].astype(np.uint8) * 255
    if not out.any():
        return out
    # the flatness test fails within half a window of the blob's edge, where the window sees the scene: grow
    # the blob back by that, plus the margin
    grow = k // 2 + round(margin * min(h, w))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    return cv2.dilate(out, kernel)


# --- moiré -----------------------------------------------------------------------------------------------------------

def _grid_notch(grey: np.ndarray) -> np.ndarray | None:
    """The filter that cuts the screen-grid peaks of ``grey``'s spectrum: float32, H x W in ``np.fft.fft2``
    layout, 0 on a peak and 1 elsewhere; None when no peak stands out."""
    h, w = grey.shape
    if min(h, w) < 2 * NOTCH_NEIGHBOURHOOD:                      # too few bins to tell a peak from its surroundings
        return None
    g = grey.astype(np.float32)
    # a Hann window, for the search only: without it the frame's borders leak along both axes and hide the peaks
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    mag = np.fft.fftshift(np.abs(np.fft.fft2((g - g.mean()) * window))).astype(np.float32)
    around = cv2.blur(mag, (NOTCH_NEIGHBOURHOOD, NOTCH_NEIGHBOURHOOD), borderType=cv2.BORDER_REFLECT)
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    peaks = (mag > NOTCH_RATIO * around) & (np.hypot(fx, fy) >= NOTCH_MIN_FREQ)
    if not peaks.any() or peaks.mean() > NOTCH_MAX_SHARE:
        return None
    r = NOTCH_RADIUS
    cut = cv2.dilate(peaks.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    cut = cut | cut[::-1, ::-1] if h % 2 and w % 2 else cut      # odd sizes: the shifted spectrum mirrors exactly
    soft = cv2.GaussianBlur(cut.astype(np.float32), (0, 0), 0.7)  # a soft edge rings less than a hard one
    return np.fft.ifftshift(1.0 - np.clip(soft * 1.5, 0.0, 1.0))


def damp_moire(image: np.ndarray, *, sigma: float = 1.0, notch: bool = False) -> np.ndarray:
    """``image`` with the screen's pixel grid damped, same size and type; the input is not modified.

    ``sigma``: a light Gaussian, px. The grid is near the camera's own pixel pitch (periods of 2-4 px), and when the
    frame is shrunk for matching it would alias into coarse stripes: sigma 1 leaves 11 % of a 3 px grid and 98 %
    of anything coarser than 32 px. 0 skips it.
    ``notch``: also cut the peaks the grid leaves in the spectrum (found per frame, above ``NOTCH_MIN_FREQ``).
    That reaches the coarser moiré beat (7-10 px), which a Gaussian could only remove together with the scene.
    Slower (an FFT per channel), and a frame with no peak comes back as with ``notch=False``.
    With ``sigma=0`` and no notch the image itself is returned.
    """
    image = _checked(image, "damp_moire")
    if not _is_number(sigma) or sigma < 0:
        raise ValueError(f"damp_moire: sigma must be >= 0 px, got {sigma!r}")
    out = image
    if notch:
        keep = _grid_notch(_grey(image))
        if keep is not None:
            channels = image[:, :, None] if image.ndim == 2 else image
            cleaned = [np.fft.ifft2(np.fft.fft2(channels[:, :, c].astype(np.float32)) * keep).real
                       for c in range(channels.shape[2])]
            out = np.clip(np.stack(cleaned, axis=2), 0, 255).round().astype(np.uint8).reshape(image.shape)
    if sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), float(sigma), borderType=cv2.BORDER_REFLECT)
    return out


# --- the gate ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class QualityConfig:
    """The gate's settings.

    window, blur_ratio, min_history: the blur rule, see :class:`RunningMedian`. blur_side: see :func:`blur_score`.
    blank_std: a frame whose grey levels vary by less than this (std, grey levels) is blank.
    min_side: frames with a shorter side under this many px are not judged.
    glare_level, glare_texture, glare_min_area, glare_margin: see :func:`glare_mask`.
    max_glare: a frame with more than this share of glare is rejected; with less, kept and masked.
    moire_sigma, moire_notch: :func:`damp_moire` applied to the frames that pass (0 and False: frames pass as
    they came). The verdict is always made on the frame as it came.
    """

    window: int = 15
    blur_ratio: float = 0.4
    min_history: int = 3
    blur_side: int = BLUR_SIDE
    blank_std: float = 2.0
    min_side: int = 32
    glare_level: int = 240
    glare_texture: float = 3.0
    glare_min_area: float = 0.002
    glare_margin: float = 0.01
    max_glare: float = 0.5
    moire_sigma: float = 0.0
    moire_notch: bool = False

    def __post_init__(self):
        RunningMedian(self.window, self.blur_ratio, self.min_history)          # checks the three together
        for name, ok, want in (
                ("blur_side", _is_count(self.blur_side) and self.blur_side >= 3, "a whole number of px >= 3"),
                ("blank_std", _is_number(self.blank_std) and self.blank_std >= 0, ">= 0 grey levels"),
                ("min_side", _is_count(self.min_side), "a whole number of px >= 1"),
                ("glare_level", _is_number(self.glare_level) and 0 <= self.glare_level <= 255, "in 0..255"),
                ("glare_texture", _is_number(self.glare_texture) and self.glare_texture >= 0, ">= 0 grey levels"),
                ("glare_min_area", _is_number(self.glare_min_area) and 0 <= self.glare_min_area <= 1, "in [0, 1]"),
                ("glare_margin", _is_number(self.glare_margin) and 0 <= self.glare_margin <= 1, "in [0, 1]"),
                ("max_glare", _is_number(self.max_glare) and 0 < self.max_glare <= 1, "in (0, 1]"),
                ("moire_sigma", _is_number(self.moire_sigma) and self.moire_sigma >= 0, ">= 0 px")):
            if not ok:
                raise ValueError(f"{name} must be {want}, got {getattr(self, name)!r}")


@dataclass(frozen=True)
class FrameQuality:
    """The gate's verdict on one frame.

    idx, t, source: the frame's own. kept: it passed. reason: one of :data:`REASONS` when rejected, else None.
    score: its blur score (0 when the frame was not scored: too small or blank).
    median, threshold: what the score was compared with; None when the blur rule did not run (the first frames,
    or a frame rejected earlier). glare: the share of the frame under glare, 0..1.
    mask: for a kept frame with glare, uint8 H x W in the Matcher's convention (nonzero where the frame's pixels
    may be used); else None. Only :attr:`QualityGate.last` carries it; the history keeps numbers, not pixels.
    """

    idx: int
    t: float
    source: str
    kept: bool
    reason: str | None
    score: float
    median: float | None = None
    threshold: float | None = None
    glare: float = 0.0
    mask: np.ndarray | None = field(default=None, compare=False, repr=False)


class QualityGate:
    """``source`` without its bad frames: a FrameSource of the frames that pass the checks of the module docstring.

    Kept frames come out as they went in (the same Frame, unless the moiré damping is on: then only the pixels
    change); they keep their idx and t, so the stream has gaps and a wrapped flight's ``truth(idx)`` still lines
    up. Every other attribute of the wrapped source (``fps``, ``rate``, ``truth``, ...) is passed through.

    After, or during, a pass: ``verdicts`` holds a :class:`FrameQuality` for every frame seen so far, ``rejected``
    the ones that did not pass, ``last`` the verdict of the frame just yielded, with its glare mask.
    :meth:`judged` yields every frame, kept or not, next to its verdict (for a contact sheet). Each pass starts
    with an empty history, so two passes give the same frames. ``len`` is the number of frames that come out,
    found by one pass over the source when none has finished yet (cached).
    """

    def __init__(self, source: FrameSource, cfg: QualityConfig | None = None):
        cfg = QualityConfig() if cfg is None else cfg
        if not isinstance(cfg, QualityConfig):
            raise TypeError(f"cfg must be a QualityConfig, got {type(cfg).__name__}")
        self.source, self.cfg = source, cfg
        self.verdicts: list[FrameQuality] = []
        self.last: FrameQuality | None = None
        self._len: int | None = None

    @property
    def rejected(self) -> list[FrameQuality]:
        return [q for q in self.verdicts if not q.kept]

    # -- one frame
    def _assess(self, frame: Frame, rule: RunningMedian) -> tuple[FrameQuality, tuple[float, float]]:
        """The verdict on ``frame``, and (measured, limit): the two numbers its reason was decided on."""
        cfg, image = self.cfg, frame.image

        def verdict(reason: str | None, score: float = 0.0, **kw) -> FrameQuality:
            return FrameQuality(idx=frame.idx, t=frame.t, source=frame.source, kept=reason is None, reason=reason,
                                score=score, **kw)

        short = min(image.shape[:2])
        if short < cfg.min_side:
            return verdict("too_small"), (float(short), float(cfg.min_side))
        std = float(_working(_grey(image), cfg.blur_side).std())
        if std < cfg.blank_std:
            return verdict("blank"), (std, cfg.blank_std)
        glare = glare_mask(image, level=cfg.glare_level, texture=cfg.glare_texture, min_area=cfg.glare_min_area,
                           margin=cfg.glare_margin)
        share = float(np.count_nonzero(glare)) / glare.size
        usable = cv2.bitwise_not(glare) if share > 0 else None
        score = blur_score(image, usable, side=cfg.blur_side)
        if share > cfg.max_glare:
            # mostly burnt: what is left says little about the scene, so it stays out of the blur history
            return verdict("glare", score, glare=share), (share, cfg.max_glare)
        kept, median, threshold = rule.judge(score)
        q = verdict(None if kept else "blur", score, median=median, threshold=threshold, glare=share,
                    mask=usable if kept else None)
        return q, (score, threshold if threshold is not None else 0.0)

    # -- the stream
    def _pass(self, quiet: bool = False) -> Iterator[tuple[Frame, FrameQuality]]:
        rule = RunningMedian(self.cfg.window, self.cfg.blur_ratio, self.cfg.min_history)
        verdicts: list[FrameQuality] = []
        if not quiet:
            self.verdicts, self.last = verdicts, None
        for frame in self.source:
            q, (measured, limit) = self._assess(frame, rule)
            verdicts.append(dataclasses.replace(q, mask=None))
            if not q.kept and not quiet:
                log.info("rejected idx=%d reason=%s score=%.6g threshold=%.6g source=%s", q.idx, q.reason, measured,
                         limit, q.source)
            if q.kept and (self.cfg.moire_sigma > 0 or self.cfg.moire_notch):
                frame = dataclasses.replace(frame, image=damp_moire(frame.image, sigma=self.cfg.moire_sigma,
                                                                    notch=self.cfg.moire_notch))
            if not quiet:
                self.last = q
            yield frame, q
        self._len = sum(q.kept for q in verdicts)                 # only a pass that reached the end knows it
        if not quiet:
            reasons = Counter(q.reason for q in verdicts if not q.kept)
            log.info("quality gate: kept %d of %d frames; rejected %s", self._len, len(verdicts),
                     " ".join(f"{r}={reasons[r]}" for r in REASONS if reasons[r]) or "none")

    def judged(self) -> Iterator[tuple[Frame, FrameQuality]]:
        """Every frame of the source, in order, with its verdict: the rejected ones too (as they came)."""
        return self._pass()

    def __iter__(self) -> Iterator[Frame]:
        for frame, q in self._pass():
            if q.kept:
                yield frame

    def __len__(self) -> int:
        if self._len is None:
            for _ in self._pass(quiet=True):                      # count only: no log lines, no change of state
                pass
        return self._len

    def __getattr__(self, name: str):
        # only reached for names this class lacks; private names are never forwarded (and copy/pickle probe them
        # before ``source`` exists)
        if name.startswith("_") or name in ("source", "cfg", "verdicts", "last"):
            raise AttributeError(name)
        return getattr(self.source, name)
