"""RootSIFT matcher: SIFT keypoints, RootSIFT descriptors, FLANN nearest neighbours, Lowe's ratio test.

The map is large and fixed for a whole run, so its features are computed once and cached on disk
(``cache_dir``, by default ``outputs/cache/``). The cache file is named after the map's pixels and the
extraction parameters, so a different map or different parameters never reuse stale features, while a new run
on the same map loads them in a fraction of the extraction time.

Window queries (``map_img`` cut from the registered map at ``map_offset``) don't re-extract: they take the
cached map keypoints that lie inside the window. Keypoints near the window's edge keep the descriptors computed
from the full map, which is what a later run with a different window would see too. Every returned map point is
in full-map pixels.

RootSIFT (Arandjelović & Zisserman, 2012): L1-normalise each SIFT vector and take the element-wise square
root. Euclidean distance between RootSIFT vectors then compares histograms with the Hellinger kernel, which
matches noticeably better than plain SIFT's Euclidean distance, at no extra cost.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from dronenav.types import Matches

__all__ = ["RootSiftConfig", "RootSiftMatcher", "rootsift"]

BACKEND = "rootsift"
# Bump when the extraction or the file layout changes, so old cache files are ignored rather than misread.
# 2: keypoints in corner-origin pixels and SIFT's precise upscale (M1-12); version-1 files are 0.25 to 0.5 px off.
_CACHE_VERSION = 2


@dataclass(frozen=True)
class RootSiftConfig:
    """Parameters of the RootSIFT matcher.

    Extraction (part of the cache key): n_features (0 = keep all, the map needs them all for windowed queries),
    contrast_threshold and edge_threshold (OpenCV's SIFT defaults: lower contrast finds more, weaker points).
    Matching (not part of the key): ratio, Lowe's test, keep a match when best < ratio * second best (0.8 as in
    Lowe 2004 and our prototype: all 50 task crops register); flann_trees / flann_checks, the KD-tree forest
    size and how many leaves a query visits (more: slower, closer to exact).
    """

    ratio: float = 0.8
    n_features: int = 0
    contrast_threshold: float = 0.04
    edge_threshold: float = 10.0
    flann_trees: int = 8
    flann_checks: int = 128

    def __post_init__(self):
        if not 0.0 < self.ratio <= 1.0:
            raise ValueError(f"ratio must be in (0, 1], got {self.ratio}")
        if self.n_features < 0:
            raise ValueError(f"n_features must be >= 0, got {self.n_features}")
        if self.contrast_threshold <= 0 or self.edge_threshold <= 0:
            raise ValueError("contrast_threshold and edge_threshold must be positive")
        if self.flann_trees < 1 or self.flann_checks < 1:
            raise ValueError("flann_trees and flann_checks must be >= 1")

    def extraction_key(self) -> dict:
        """The parameters that change the extracted features: what the cache file name depends on."""
        return {k: v for k, v in asdict(self).items() if k in ("n_features", "contrast_threshold", "edge_threshold")}


def rootsift(desc: np.ndarray) -> np.ndarray:
    """RootSIFT of N x 128 SIFT descriptors: L1-normalise each row, then the square root. Rows come out with unit
    L2 norm (the squares sum to the L1 norm, 1); an all-zero row stays zero."""
    d = np.asarray(desc, dtype=np.float32)
    l1 = np.abs(d).sum(axis=1, keepdims=True)
    return np.sqrt(np.divide(d, l1, out=np.zeros_like(d), where=l1 > 0))


def _gray(img: np.ndarray) -> np.ndarray:
    """uint8 single-channel view of a BGR or grayscale image (SIFT works on intensity)."""
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 2:
        return img
    raise ValueError(f"expected an H x W x 3 BGR or H x W grayscale image, got shape {img.shape}")


class _Features:
    """Keypoint positions (N x 2 float32, image px) and their RootSIFT descriptors (N x 128 float32)."""

    __slots__ = ("desc", "pts")

    def __init__(self, pts: np.ndarray, desc: np.ndarray):
        self.pts, self.desc = pts, desc

    def __len__(self) -> int:
        return len(self.pts)


class RootSiftMatcher:
    """A :class:`~dronenav.types.Matcher` backend with RootSIFT features and a disk cache for the map's.

    map_img: optional full map to register now (its features are loaded or extracted at once); otherwise the
    first full-map ``match`` call (``map_offset == (0, 0)``) registers it. cache_dir: where map features are
    cached; None keeps them in memory only. After registration, ``map_extractions`` counts how many times this
    instance extracted map features (0: everything came from the cache) and ``map_features_source`` says where
    the current map's came from ("extracted" or "cache").
    """

    backend = BACKEND

    def __init__(self, map_img: np.ndarray | None = None, cache_dir: str | Path | None = "outputs/cache",
                 cfg: RootSiftConfig = RootSiftConfig()):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.map_extractions = 0
        self.map_features_source: str | None = None
        # precise upscale: SIFT's first octave doubles the image, and without it the keypoints come back a quarter
        # pixel off (+0.25 px on each axis, measured on a 180° turn; #106). With it they are at OpenCV's pixel
        # centres, which _extract shifts to ours.
        self._sift = cv2.SIFT_create(nfeatures=cfg.n_features, contrastThreshold=cfg.contrast_threshold,
                                     edgeThreshold=cfg.edge_threshold, enable_precise_upscale=True)
        self._map: np.ndarray | None = None          # registered map, grayscale
        self._map_feats: _Features | None = None
        self._map_flann: cv2.FlannBasedMatcher | None = None   # index over all map features, built on first use
        if map_img is not None:
            self.set_map(map_img)

    # --- the map and its cache -----------------------------------------------------------------------------

    def set_map(self, map_img: np.ndarray) -> None:
        """Register the full map: load its features from the cache, or extract and cache them."""
        gray = _gray(map_img)
        path = self._cache_path(gray)
        feats = self._load(path) if path is not None else None
        if feats is None:
            feats = self._extract(gray)
            self.map_extractions += 1
            self.map_features_source = "extracted"
            if path is not None:
                self._save(path, feats)
        else:
            self.map_features_source = "cache"
        self._map, self._map_feats, self._map_flann = gray, feats, None

    def _cache_path(self, gray: np.ndarray) -> Path | None:
        if self.cache_dir is None:
            return None
        # content hash: the same pixels hit the cache whatever array holds them; one changed pixel does not
        h = hashlib.sha1(f"{gray.shape}".encode())
        h.update(np.ascontiguousarray(gray).data)
        params = json.dumps({"v": _CACHE_VERSION, **self.cfg.extraction_key()}, sort_keys=True)
        key = hashlib.sha1(params.encode()).hexdigest()[:8]
        return self.cache_dir / f"{BACKEND}_{h.hexdigest()[:16]}_{key}.npz"

    @staticmethod
    def _load(path: Path) -> _Features | None:
        """The cached features, or None if the file is missing or unreadable (then the caller re-extracts)."""
        if not path.exists():
            return None
        try:
            with np.load(path) as z:
                pts, desc = z["pts"].astype(np.float32), z["desc"].astype(np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2 or desc.shape != (len(pts), 128):
                return None
            return _Features(pts, desc)
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            return None   # a truncated or foreign file is a cache miss, never a crash

    @staticmethod
    def _save(path: Path, feats: _Features) -> None:
        """Write atomically (temp file, then rename), so a crash or a parallel run never leaves half a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, pts=feats.pts, desc=feats.desc)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _extract(self, gray: np.ndarray) -> _Features:
        kps, desc = self._sift.detectAndCompute(gray, None)
        if desc is None or not kps:
            return _Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
        # OpenCV puts pixel centres at integers, dronenav.types at +0.5 (origin at the top-left pixel's corner).
        # Shifted here, once, both the frame's and the map's points are in our convention: the homography, the
        # fix and the footprint then are too. Left out, the frame side's 0.5 is turned and scaled by the
        # homography and the map side's is not: up to ~0.5 px of bias at 180°, more at larger scales (#106).
        pts = np.array([k.pt for k in kps], np.float32) + np.float32(0.5)
        return _Features(pts, rootsift(desc))

    # --- matching ------------------------------------------------------------------------------------------

    def match(self, frame_img: np.ndarray, map_img: np.ndarray, map_offset: tuple = (0, 0),
              mask: np.ndarray | None = None) -> Matches:
        """Correspondences between the frame and ``map_img`` (the full map or a window of it at ``map_offset``),
        with ``pts_map`` in full-map pixels. See :class:`dronenav.types.Matcher`."""
        t0 = time.perf_counter()
        ox, oy = _whole_px(map_offset)
        win = _gray(map_img)
        if mask is not None and np.asarray(mask).shape != win.shape:
            raise ValueError(f"mask must be {win.shape} like map_img, got {np.asarray(mask).shape}")

        feats, flann = self._map_features(win, ox, oy, mask)
        frame = self._extract(_gray(frame_img))
        if len(frame) == 0 or len(feats) < 2:
            return Matches.empty(BACKEND, _ms(t0))

        knn = flann.knnMatch(frame.desc, k=2) if flann is not None else self._flann(feats).knnMatch(frame.desc, k=2)
        qi, ti, score = [], [], []
        for pair in knn:
            if len(pair) < 2 or pair[1].distance <= 0:
                continue
            best, second = pair
            if best.distance < self.cfg.ratio * second.distance:
                qi.append(best.queryIdx)
                ti.append(best.trainIdx)
                score.append(1.0 - best.distance / second.distance)   # margin over the runner-up: higher is surer
        if not qi:
            return Matches.empty(BACKEND, _ms(t0))
        return Matches(frame.pts[qi], feats.pts[ti], np.array(score), BACKEND, _ms(t0))

    def _map_features(self, win: np.ndarray, ox: int, oy: int, mask: np.ndarray | None):
        """Map features for this query, in full-map px, and a ready FLANN index over them if one is cached.

        A window of the registered map: the cached keypoints inside it (no extraction). A new full map (offset
        0, 0): register it (cache). A window of a map never registered: extract it for this call only.
        """
        if self._is_window_of_map(win, ox, oy):
            f = self._map_feats
            h, w = win.shape
            keep = (f.pts[:, 0] >= ox) & (f.pts[:, 0] < ox + w) & (f.pts[:, 1] >= oy) & (f.pts[:, 1] < oy + h)
            if mask is not None:
                idx = np.flatnonzero(keep)
                c = (f.pts[idx] - (ox, oy)).astype(int)        # inside the window, so valid indices
                keep[idx] = np.asarray(mask)[c[:, 1], c[:, 0]] != 0
            if keep.all():                                    # the whole map: reuse one index for every frame
                if self._map_flann is None:
                    self._map_flann = self._flann(f)
                return f, self._map_flann
            return _Features(f.pts[keep], f.desc[keep]), None
        if (ox, oy) == (0, 0):
            self.set_map(win)
            return self._map_features(win, 0, 0, mask)
        f = self._extract(win)
        keep = np.ones(len(f), bool)
        if mask is not None and len(f):
            c = f.pts.astype(int)
            keep = np.asarray(mask)[c[:, 1], c[:, 0]] != 0
        return _Features(f.pts[keep] + np.float32([ox, oy]), f.desc[keep]), None

    def _is_window_of_map(self, win: np.ndarray, ox: int, oy: int) -> bool:
        """True when ``win`` is exactly the registered map's pixels at (ox, oy): only then are cached keypoints
        valid for it. Comparing the pixels costs far less than matching and guards against a stale map."""
        if self._map is None or ox < 0 or oy < 0:
            return False
        h, w = win.shape
        if oy + h > self._map.shape[0] or ox + w > self._map.shape[1]:
            return False
        return np.array_equal(self._map[oy:oy + h, ox:ox + w], win)

    def _flann(self, feats: _Features) -> cv2.FlannBasedMatcher:
        flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": self.cfg.flann_trees},   # 1 = randomised KD-trees
                                      {"checks": self.cfg.flann_checks})
        flann.add([feats.desc])
        # The KD-trees split on randomly chosen dimensions drawn from OpenCV's RNG; seeding it right before the
        # build makes the same input give the same matches on every run (unseeded, a run differs by a few).
        # This resets OpenCV's RNG for the calling thread; nothing in the pipeline relies on its state.
        cv2.setRNGSeed(0)
        flann.train()
        return flann


def _whole_px(offset) -> tuple[int, int]:
    """(x, y) offset as ints; a fractional offset can't index a window, so it is an error, not a rounding."""
    try:
        ox, oy = (float(v) for v in offset)
    except (TypeError, ValueError):
        raise ValueError(f"map_offset must be (x, y), got {offset!r}") from None
    if ox != int(ox) or oy != int(oy):
        raise ValueError(f"map_offset must be whole pixels, got {offset!r}")
    return int(ox), int(oy)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0
