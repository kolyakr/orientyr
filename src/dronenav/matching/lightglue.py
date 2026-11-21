"""LightGlue matcher: DISK (or SuperPoint) keypoints, matched by LightGlue on MPS / CUDA / CPU.

Learned features hold up better than SIFT where the frame and the map differ in season, light or sensor (M4-01),
but they bring three constraints this module deals with:

- **Not rotation-invariant.** DISK and SuperPoint describe a patch as it lies in the image: a frame turned 180°
  from the map gets ~15 matches instead of ~400. The frame is therefore tried turned in 45° steps
  (:func:`turn_image`: np.rot90 for quarter turns, exact; a resampled turn between them) and the turn with the
  most matches wins; its points are turned back into the frame's own pixels (:func:`turn_back`). Quarter turns
  alone are not enough: 45° off, a task crop drops from ~300 inliers to a handful. With 45° steps every heading
  is within 22.5° of a try. The last winning turn is tried first, and a strong answer from it skips the others:
  along a flight the heading changes slowly, so this is usually one try per frame. (M4-04 adds a heading prior:
  north-up by compass.)
- **A fixed budget of keypoints per image.** A 3936 px map in one go would get a few thousand keypoints for 15
  million pixels. So the map is cut into tiles (``tile_px``, plus a margin of context for the network), each
  extracted with its own budget, and only the keypoints in a tile's core are kept: dense, seamless coverage.
- **Attention cost grows with the keypoint count.** A query matches against at most ``window_keypoints`` map
  keypoints (the strongest in the window). A larger area, such as the whole map for the first frame, is
  searched in overlapping chunks, and the chunk with the most matches wins.

Map features are extracted once and cached on disk under a key made of the map's pixels and the extraction
parameters (like RootSIFT). A Map Pack (M4-06) holds the same file: when ``pack_dir`` has it, it is read from
there and nothing is extracted. Windows of the registered map reuse its keypoints. Every returned map point is
in full-map pixels.

Weights: DISK ("depth") and LightGlue's DISK weights (both Apache-2.0 / BSD-style) are downloaded by torch.hub on
first use into ``~/.cache/torch/hub/checkpoints``. SuperPoint's weights are for non-commercial use only, so DISK is
the default. :func:`weights_available` tells offline code (CI) whether they are there.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from dronenav.device import pick_device
from dronenav.types import Matches

__all__ = ["LightGlueConfig", "LightGlueMatcher", "quarter_turn_back", "turn_back", "turn_image", "weights_available"]

BACKEND = "lightglue"
_CACHE_VERSION = 1
# the files torch.hub stores for each feature type: the extractor's weights, then LightGlue's for those features
_WEIGHTS = {
    "disk": ("depth-save.pth", "disk_lightglue_v0-1_arxiv.pth"),
    "superpoint": ("superpoint_v1.pth", "superpoint_lightglue_v0-1_arxiv.pth"),
}


def weights_available(features: str = "disk") -> bool:
    """True when the extractor's and LightGlue's weights for ``features`` are already on disk (no download)."""
    try:
        import torch
    except ImportError:
        return False
    ckpt = Path(torch.hub.get_dir()) / "checkpoints"
    return features in _WEIGHTS and all((ckpt / f).is_file() for f in _WEIGHTS[features])


@dataclass(frozen=True)
class LightGlueConfig:
    """Parameters of the LightGlue matcher.

    features: "disk" (default, open weights) or "superpoint". max_keypoints: the frame's keypoint budget per
    try, the main speed / accuracy knob (M4-09 profiles it).
    Map extraction (part of the cache key): map_keypoints, the budget per map tile; tile_px, the tile core
    (a quarter of it is added around as context and cut off again).
    Matching: window_keypoints, the most map keypoints one LightGlue call sees (the strongest are kept);
    chunk_px, the side of the chunks a larger area is searched in (half a chunk apart); turns, how many evenly
    spaced turns of the frame are tried (1: upright only, 4: quarter turns, 8: 45° steps); strong_matches, a try with at least this
    many matches is taken without trying the other turns; filter_threshold, LightGlue's minimum match score;
    min_texture, the gray-level standard deviation around a keypoint below which it is dropped (a flat patch
    has nothing to describe: this keeps blank frames and blank map areas at zero matches).
    """

    features: str = "disk"
    max_keypoints: int = 2048
    map_keypoints: int = 2048
    tile_px: int = 512
    window_keypoints: int = 4096
    chunk_px: int = 1024
    turns: int = 8
    strong_matches: int = 150
    filter_threshold: float = 0.1
    min_texture: float = 3.0

    def __post_init__(self):
        if self.features not in _WEIGHTS:
            raise ValueError(f"features must be one of {', '.join(_WEIGHTS)}, got {self.features!r}")
        for name in ("max_keypoints", "map_keypoints", "window_keypoints", "strong_matches"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.tile_px < 64 or self.chunk_px < 64:
            raise ValueError("tile_px and chunk_px must be >= 64")
        if self.turns not in (1, 2, 4, 8):
            raise ValueError(f"turns must be 1, 2, 4 or 8, got {self.turns}")
        if not 0.0 <= self.filter_threshold < 1.0 or self.min_texture < 0:
            raise ValueError("filter_threshold must be in [0, 1) and min_texture >= 0")

    def extraction_key(self) -> dict:
        """The parameters that change the map's features: what the cache file name depends on."""
        return {k: v for k, v in asdict(self).items() if k in ("features", "map_keypoints", "tile_px", "min_texture")}


def quarter_turn_back(pts: np.ndarray, q: int, size: tuple[int, int]) -> np.ndarray:
    """Points of ``np.rot90(img, q)`` (x, y, pixel centres on integers) → the same points in ``img``, whose size
    is ``size = (w, h)``. np.rot90 turns counter-clockwise: one quarter turn sends img's (x, y) to (y, w-1-x)."""
    w, h = size
    p = np.asarray(pts, float).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    q %= 4
    if q == 0:
        out = (x, y)
    elif q == 1:
        out = (w - 1 - y, x)
    elif q == 2:
        out = (w - 1 - x, h - 1 - y)
    else:
        out = (y, h - 1 - x)
    return np.stack(out, axis=1)


def turn_image(img: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray]:
    """``img`` turned by ``deg`` counter-clockwise (as np.rot90 turns it) onto a canvas that holds all of it, and
    the 2 x 3 affine A sending img's (x, y) to the turned picture's. Multiples of 90° are np.rot90 itself: no
    resampling. Other angles are resampled (bilinear), with black outside the picture."""
    h, w = img.shape[:2]
    q = deg / 90.0
    if abs(q - round(q)) < 1e-9:
        c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][round(q) % 4]
    else:
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    R = np.array([[c, s], [-s, c]], float)               # y points down: this is a counter-clockwise turn on screen
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float) @ R.T
    A = np.hstack([R, -corners.min(axis=0)[:, None]])
    if abs(q - round(q)) < 1e-9:
        return np.ascontiguousarray(np.rot90(img, round(q) % 4)), A
    W, H = (math.ceil(v - 1e-9) + 1 for v in corners.max(axis=0) - corners.min(axis=0))
    return cv2.warpAffine(img, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT), A


def turn_back(pts: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Points of a turned picture → the original's, for the affine ``A`` from :func:`turn_image`."""
    p = np.asarray(pts, float).reshape(-1, 2)
    return (p - A[:, 2]) @ np.linalg.inv(A[:, :2]).T


class _Features:
    """Keypoints (N x 2 float32, px), descriptors (N x D float32) and detector scores (N float32)."""

    __slots__ = ("desc", "pts", "scores")

    def __init__(self, pts: np.ndarray, desc: np.ndarray, scores: np.ndarray):
        self.pts, self.desc, self.scores = pts, desc, scores

    def __len__(self) -> int:
        return len(self.pts)

    def take(self, keep) -> _Features:
        return _Features(self.pts[keep], self.desc[keep], self.scores[keep])

    def strongest(self, n: int) -> _Features:
        if len(self) <= n:
            return self
        return self.take(np.sort(np.argsort(-self.scores, kind="stable")[:n]))   # sorted: a stable order

    @staticmethod
    def empty(dim: int) -> _Features:
        return _Features(np.zeros((0, 2), np.float32), np.zeros((0, dim), np.float32), np.zeros(0, np.float32))


def _rgb(img: np.ndarray) -> np.ndarray:
    """uint8 RGB from a BGR or grayscale image (the extractors take RGB; SuperPoint turns it gray itself)."""
    img = np.asarray(img)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    raise ValueError(f"expected an H x W x 3 BGR or H x W grayscale image, got shape {img.shape}")


class LightGlueMatcher:
    """A :class:`~dronenav.types.Matcher` backend: DISK / SuperPoint features matched by LightGlue.

    map_img: optional full map to register now; otherwise the first full-map ``match`` call (offset (0, 0))
    registers it. cache_dir: where extracted map features are written (None: memory only). pack_dir: a Map Pack
    folder read first, never written. device: "mps", "cuda", "cpu", or None / "auto" for the best available
    (dronenav.device.pick_device). After registration, ``map_extractions`` counts map extractions by this
    instance and ``map_features_source`` says where the map's features came from ("extracted", "cache", "pack").
    """

    backend = BACKEND

    def __init__(self, map_img: np.ndarray | None = None, cache_dir: str | Path | None = "outputs/cache",
                 cfg: LightGlueConfig = LightGlueConfig(), *, pack_dir: str | Path | None = None,
                 device: str | None = None):
        self.cfg = cfg
        self.device = pick_device(device)      # before loading anything: a bad name fails fast
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.pack_dir = Path(pack_dir) if pack_dir is not None else None
        self.map_extractions = 0
        self.map_features_source: str | None = None
        self._extractor = self._matcher = None   # loaded on first use: constructing is cheap
        self._map: np.ndarray | None = None
        self._map_feats: _Features | None = None
        self._turn = 0                          # the last winning quarter turn, tried first next time
        if map_img is not None:
            self.set_map(map_img)

    # --- the networks ------------------------------------------------------------------------------------------

    def _nets(self):
        if self._extractor is None:
            import torch
            from lightglue import DISK, LightGlue, SuperPoint

            torch.set_grad_enabled(False)
            make = DISK if self.cfg.features == "disk" else SuperPoint
            self._extractor = make(max_num_keypoints=self.cfg.max_keypoints).eval().to(self.device)
            self._matcher = LightGlue(features=self.cfg.features,
                                      filter_threshold=self.cfg.filter_threshold).eval().to(self.device)
        return self._extractor, self._matcher

    @property
    def _dim(self) -> int:
        return 128 if self.cfg.features == "disk" else 256

    def _extract(self, img: np.ndarray, budget: int) -> _Features:
        """Keypoints of one image (any size), at most ``budget``, flat ones dropped. Pixels of ``img``."""
        import torch

        rgb = _rgb(img)
        extractor, _ = self._nets()
        extractor.conf.max_num_keypoints = budget
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
        with torch.inference_mode():
            f = extractor.extract(x, resize=None)        # no resize: the scale is the Searcher's business
        pts = f["keypoints"][0].cpu().numpy().astype(np.float32)
        feats = _Features(pts, f["descriptors"][0].cpu().numpy().astype(np.float32),
                          f["keypoint_scores"][0].cpu().numpy().astype(np.float32))
        return feats.take(self._textured(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), feats.pts))

    def _textured(self, gray: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Which keypoints sit on texture: the gray-level std over a 9 x 9 patch around them."""
        if not len(pts):
            return np.zeros(0, bool)
        g = gray.astype(np.float32)
        mean = cv2.blur(g, (9, 9))
        std = np.sqrt(np.maximum(cv2.blur(g * g, (9, 9)) - mean * mean, 0.0))
        c = np.clip(np.rint(pts).astype(int), 0, [gray.shape[1] - 1, gray.shape[0] - 1])
        return std[c[:, 1], c[:, 0]] >= self.cfg.min_texture

    # --- the map and its cache ---------------------------------------------------------------------------------

    def set_map(self, map_img: np.ndarray) -> None:
        """Register the full map: read its features from the pack or the cache, or extract and cache them."""
        img = np.ascontiguousarray(map_img)
        name = self._cache_name(img)
        for source, folder in (("pack", self.pack_dir), ("cache", self.cache_dir)):
            feats = self._load(folder / name) if folder is not None else None
            if feats is not None:
                self.map_features_source = source
                break
        else:
            feats = self._extract_tiled(img)
            self.map_extractions += 1
            self.map_features_source = "extracted"
            if self.cache_dir is not None:
                self._save(self.cache_dir / name, feats)
        self._map, self._map_feats = img, feats

    def _cache_name(self, img: np.ndarray) -> str:
        h = hashlib.sha1(f"{img.shape}".encode())
        h.update(img.data)
        params = json.dumps({"v": _CACHE_VERSION, **self.cfg.extraction_key()}, sort_keys=True)
        return f"{BACKEND}_{h.hexdigest()[:16]}_{hashlib.sha1(params.encode()).hexdigest()[:8]}.npz"

    def _extract_tiled(self, img: np.ndarray) -> _Features:
        """Each tile's core keeps the keypoints found with a margin of context around it: dense, seamless."""
        t, pad = self.cfg.tile_px, self.cfg.tile_px // 4
        h, w = img.shape[:2]
        parts = []
        for y in range(0, h, t):
            for x in range(0, w, t):
                x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + t + pad), min(h, y + t + pad)
                f = self._extract(img[y0:y1, x0:x1], self.cfg.map_keypoints)
                f.pts += np.float32([x0, y0])
                core = (f.pts[:, 0] >= x) & (f.pts[:, 0] < x + t) & (f.pts[:, 1] >= y) & (f.pts[:, 1] < y + t)
                parts.append(f.take(core))
        if not parts:
            return _Features.empty(self._dim)
        return _Features(*(np.concatenate([getattr(p, k) for p in parts]) for k in ("pts", "desc", "scores")))

    def _load(self, path: Path) -> _Features | None:
        """The stored features, or None if missing or unreadable (a cache miss, never a crash)."""
        if not path.is_file():
            return None
        try:
            with np.load(path) as z:
                f = _Features(z["pts"].astype(np.float32), z["desc"].astype(np.float32),
                              z["scores"].astype(np.float32))
            if f.pts.shape != (len(f), 2) or f.desc.shape != (len(f), self._dim) or f.scores.shape != (len(f),):
                return None
            return f
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            return None

    @staticmethod
    def _save(path: Path, feats: _Features) -> None:
        """Write atomically (temp file, then rename): a crash or a parallel run never leaves half a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, pts=feats.pts, desc=feats.desc, scores=feats.scores)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # --- matching ----------------------------------------------------------------------------------------------

    def match(self, frame_img: np.ndarray, map_img: np.ndarray, map_offset: tuple = (0, 0),
              mask: np.ndarray | None = None) -> Matches:
        """Correspondences between the frame and ``map_img`` (the full map or a window of it at ``map_offset``),
        with ``pts_map`` in full-map pixels. See :class:`dronenav.types.Matcher`."""
        t0 = time.perf_counter()
        ox, oy = _whole_px(map_offset)
        win = np.asarray(map_img)
        if mask is not None and np.asarray(mask).shape != win.shape[:2]:
            raise ValueError(f"mask must be {win.shape[:2]} like map_img, got {np.asarray(mask).shape}")
        feats = self._map_features(win, ox, oy, mask)
        if len(feats) < 4:
            return Matches.empty(BACKEND, _ms(t0))
        chunks = self._chunks(feats, ox, oy, win.shape[1], win.shape[0])

        frame = np.asarray(frame_img)
        n = self.cfg.turns
        tried: set[int] = set()
        best = None                               # (n matches, turn, frame points, map points, scores)

        def attempt(k: int) -> None:
            nonlocal best
            tried.add(k)
            turned, A = turn_image(frame, 360.0 * k / n)
            fk = self._extract(turned, self.cfg.max_keypoints)
            if 360 * k % (n * 90):                 # a resampled turn: drop keypoints on the black margin's edge
                fk = fk.take(_inside_picture(frame.shape, A, turned.shape, fk.pts))
            if len(fk) < 4:
                return
            for c in chunks:
                i0, i1, scores = self._lightglue(fk, (turned.shape[1], turned.shape[0]), c)
                if len(i0) and (best is None or len(i0) > best[0]):
                    best = (len(i0), k, turn_back(fk.pts[i0], A), c.pts[i1], scores)

        # start from the last winning turn: the heading changes slowly along a flight
        for k in ((self._turn + j) % n for j in range(n)):
            attempt(k)
            if best is not None and best[0] >= self.cfg.strong_matches:
                break
        # a strong answer can still come from a turn up to 45° off; its matches say which turn the frame really
        # is at, and the turn closest to that one matches better (more and more precise points), so try it too
        if best is not None:
            k_seen = _closest_turn(best[2], best[3], n)
            if k_seen is not None and k_seen not in tried:
                attempt(k_seen)
        if best is None:
            return Matches.empty(BACKEND, _ms(t0))
        _, k, pts_frame, pts_map, scores = best
        self._turn = k
        return Matches(pts_frame.astype(np.float32), pts_map.astype(np.float32), scores, BACKEND, _ms(t0))

    def _lightglue(self, frame: _Features, frame_size: tuple[int, int], chunk: _Chunk):
        """LightGlue between the frame's and the chunk's keypoints: matched indices on each side, and scores."""
        import torch

        _, lg = self._nets()

        def side(f: _Features, pts: np.ndarray, size) -> dict:
            return {"keypoints": torch.from_numpy(pts).float()[None].to(self.device),
                    "descriptors": torch.from_numpy(f.desc).float()[None].to(self.device),
                    "image_size": torch.tensor([size], dtype=torch.float32, device=self.device)}

        with torch.inference_mode():
            out = lg({"image0": side(frame, frame.pts, frame_size),
                      "image1": side(chunk.feats, chunk.pts - chunk.origin, chunk.size)})
        m = out["matches"][0].cpu().numpy().reshape(-1, 2)
        return m[:, 0], m[:, 1], out["scores"][0].cpu().numpy().astype(float)

    def _map_features(self, win: np.ndarray, ox: int, oy: int, mask: np.ndarray | None) -> _Features:
        """Map features for this query, in full-map px: the registered map's inside a window of it, a new full
        map registered first, or a window of an unregistered map extracted for this call only."""
        h, w = win.shape[:2]
        if self._is_window_of_map(win, ox, oy):
            f = self._map_feats
            keep = (f.pts[:, 0] >= ox) & (f.pts[:, 0] < ox + w) & (f.pts[:, 1] >= oy) & (f.pts[:, 1] < oy + h)
        elif (ox, oy) == (0, 0):
            self.set_map(win)
            return self._map_features(win, 0, 0, mask)
        else:
            f = self._extract_tiled(win)
            f.pts += np.float32([ox, oy])
            keep = np.ones(len(f), bool)
        if mask is not None:
            idx = np.flatnonzero(keep)
            c = np.clip(np.rint(f.pts[idx] - (ox, oy)).astype(int), 0, [w - 1, h - 1])
            keep[idx] = np.asarray(mask)[c[:, 1], c[:, 0]] != 0
        return f.take(keep)

    def _chunks(self, feats: _Features, ox: int, oy: int, w: int, h: int) -> list[_Chunk]:
        """The query area as one chunk if it fits ``chunk_px``, else overlapping chunks half a chunk apart; each
        keeps at most ``window_keypoints`` of its strongest keypoints."""
        c = self.cfg.chunk_px
        if w <= c and h <= c:
            return [_Chunk(feats.strongest(self.cfg.window_keypoints), (ox, oy), (w, h))]
        out = []
        step = c // 2
        for y in _starts(h, c, step):
            for x in _starts(w, c, step):
                cw, ch = min(c, w - x), min(c, h - y)
                p = feats.pts - (ox + x, oy + y)
                inside = (p[:, 0] >= 0) & (p[:, 0] < cw) & (p[:, 1] >= 0) & (p[:, 1] < ch)
                if inside.sum() >= 4:
                    out.append(_Chunk(feats.take(inside).strongest(self.cfg.window_keypoints), (ox + x, oy + y),
                                      (cw, ch)))
        return out

    def _is_window_of_map(self, win: np.ndarray, ox: int, oy: int) -> bool:
        """True when ``win`` is exactly the registered map's pixels at (ox, oy): only then are its keypoints valid."""
        if self._map is None or ox < 0 or oy < 0 or win.ndim != self._map.ndim:
            return False
        h, w = win.shape[:2]
        if oy + h > self._map.shape[0] or ox + w > self._map.shape[1]:
            return False
        return np.array_equal(self._map[oy:oy + h, ox:ox + w], win)


class _Chunk:
    """Map keypoints of one query area (full-map px), with the area's origin and size for LightGlue."""

    __slots__ = ("feats", "origin", "size")

    def __init__(self, feats: _Features, origin: tuple[int, int], size: tuple[int, int]):
        self.feats, self.origin, self.size = feats, np.float32(origin), size

    @property
    def pts(self) -> np.ndarray:
        return self.feats.pts


def _closest_turn(pts_frame: np.ndarray, pts_map: np.ndarray, turns: int) -> int | None:
    """The turn (of ``turns`` evenly spaced, as :func:`turn_image` turns) that best undoes the in-plane rotation
    from the frame to the map seen in these matches (a RANSAC similarity), or None when they fit none."""
    if len(pts_frame) < 4:
        return None
    M, _ = cv2.estimateAffinePartial2D(pts_frame.astype(np.float32), pts_map.astype(np.float32),
                                       method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None
    # turn_image(deg) applies the rotation of angle -deg (y down); undoing the map rotation phi needs deg = -phi
    phi = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    return round(-phi / (360.0 / turns)) % turns


def _inside_picture(frame_shape, A: np.ndarray, turned_shape, pts: np.ndarray, margin: int = 6) -> np.ndarray:
    """Which keypoints of a turned frame lie at least ``margin`` px inside the picture (not on the black corners
    the turn adds, whose straight edge would otherwise look like a feature)."""
    valid = cv2.warpAffine(np.full(frame_shape[:2], 255, np.uint8), A, (turned_shape[1], turned_shape[0]),
                           flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    c = np.clip(np.rint(pts).astype(int), 0, [turned_shape[1] - 1, turned_shape[0] - 1])
    return valid[c[:, 1], c[:, 0]] > 0


def _starts(length: int, size: int, step: int) -> list[int]:
    """Chunk starts covering [0, length): every ``step``, the last one flush with the end."""
    if length <= size:
        return [0]
    s = list(range(0, length - size, step))
    return s + [length - size]


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
