"""XFeat matcher: a small learned detector and descriptor for the fast tracking path (M4-03).

XFeat (Potje et al., "XFeat: Accelerated Features for Lightweight Image Matching", CVPR 2024) is a compact
network that gives keypoints and 64-D descriptors in one pass. Here it runs through ``kornia.feature.XFeat``
with the authors' weights; matching is mutual nearest neighbours on the cosine similarity, plus a ratio test.

What this backend adds around the network:

- **The frame is tried at several turns.** XFeat's descriptors are not rotation-invariant: they hold for some
  ±25°, and a frame at heading 60° finds nothing on a north-up map. So the frame is turned by k * 360° / turns,
  described and matched at each turn, and the turn with the most matches wins (a wrong turn gives a tenth of
  the right one's). A clear winner is remembered and tried first for the next frame: while the heading holds,
  a frame costs one extraction. ``turns=1`` is for a caller that turns the frames north-up itself (a compass).
- **The map is described in tiles, once, and cached on disk** (as RootSIFT's features are): a window query takes
  the cached keypoints inside the window, with no extraction. The map is described more densely than a frame,
  because every frame keypoint needs its partner to be there.
- **Exact geometry.** Images are padded (never resized) to the network's stride, and keypoints come back in
  corner-origin pixels (pixel centres at +0.5, see ``dronenav.types``).

Weights are never in the repository. They are read from a local file: the ``weights`` argument, else the
``DRONENAV_XFEAT_WEIGHTS`` environment variable, else torch's hub cache, where ``download=True`` puts them once
(about 6 MB, from the authors' repository). torch and kornia are imported only when a matcher is built.
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
from dronenav.matching.rootsift import _gray, _ms, _whole_px
from dronenav.types import Matches

__all__ = ["WEIGHTS_URL", "XFeatConfig", "XFeatMatcher", "XFeatWeightsMissing", "xfeat_weights_path"]

BACKEND = "xfeat"
WEIGHTS_URL = "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt"
WEIGHTS_ENV = "DRONENAV_XFEAT_WEIGHTS"
_STRIDE = 32          # the network halves the image five times: sizes must be multiples of 32
_DESC_DIM = 64
# Bump when the extraction or the file layout changes, so old cache files are ignored rather than misread.
_CACHE_VERSION = 1
# Map descriptors are compared in blocks of this many, so matching against a whole large map (hundreds of
# thousands of keypoints) needs a few hundred MB at a time instead of several GB.
_MATCH_BLOCK = 32768


class XFeatWeightsMissing(FileNotFoundError):
    """The XFeat weights file is not on this machine and downloading it was not allowed."""


def xfeat_weights_path() -> Path:
    """Where the weights are looked for by default: ``$DRONENAV_XFEAT_WEIGHTS`` if set, else torch's hub cache
    (``$TORCH_HOME/hub/checkpoints/xfeat.pt``, the file kornia's own loader uses, so one download serves both).
    Worked out without importing torch."""
    explicit = os.environ.get(WEIGHTS_ENV)
    if explicit:
        return Path(explicit).expanduser()
    home = os.environ.get("TORCH_HOME")
    if home:
        root = Path(home).expanduser()
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME") or "~/.cache").expanduser() / "torch"
    return root / "hub" / "checkpoints" / "xfeat.pt"


@dataclass(frozen=True)
class XFeatConfig:
    """Parameters of the XFeat matcher.

    Extraction of the map (part of the cache key): map_top_k, keypoints kept per megapixel of map (dense on
    purpose: a frame keypoint without its partner on the map cannot match); detection_threshold, the keypoint
    heatmap value to exceed (kornia's default); tile and tile_margin, the map is described in tiles of ``tile``
    px, each seen with ``tile_margin`` px of its surroundings so that keypoints next to a seam are described as
    in the whole image. Both are multiples of 32, the network's stride: a tile then sits on the same grid as
    the whole map would. The margin is wide because the network looks far: on the synthetic test map cut into
    256 px tiles, a frame finds 27% / 61% / 94% / 101% of the whole-image matches with a margin of 0 / 64 / 128 /
    192 px.

    Frame and matching (not part of the key): top_k, keypoints kept per frame; min_cossim, the least cosine
    similarity of a match; ratio, keep a match when its descriptor distance is under ratio * the runner-up's
    (Lowe's test; 1.0 turns it off); turns, how many turns of the frame are tried (8: every 45°; 1: frames are
    already north-up); win_ratio, a turn is remembered only when it wins clearly, with at least win_ratio times
    the matches of every turn that is not its neighbour (a frame at the wrong scale, or off the map, gives a
    few chance matches at every turn: no winner worth remembering); keep_fraction and keep_min, the remembered
    turn is kept while it gives at least keep_fraction of the matches it won with and at least keep_min,
    otherwise all turns are tried again.
    """

    top_k: int = 4096
    map_top_k: int = 16384
    detection_threshold: float = 0.05
    min_cossim: float = 0.82
    ratio: float = 0.9
    turns: int = 8
    win_ratio: float = 2.0
    keep_fraction: float = 0.5
    keep_min: int = 30
    tile: int = 1024
    tile_margin: int = 192

    def __post_init__(self):
        if self.top_k < 1 or self.map_top_k < 1:
            raise ValueError(f"top_k and map_top_k must be >= 1, got {self.top_k} and {self.map_top_k}")
        if not 0.0 < self.detection_threshold < 1.0:
            raise ValueError(f"detection_threshold must be in (0, 1), got {self.detection_threshold}")
        if not -1.0 <= self.min_cossim <= 1.0:
            raise ValueError(f"min_cossim must be in [-1, 1], got {self.min_cossim}")
        if not 0.0 < self.ratio <= 1.0:
            raise ValueError(f"ratio must be in (0, 1], got {self.ratio}")
        if self.turns < 1:
            raise ValueError(f"turns must be >= 1, got {self.turns}")
        if self.win_ratio < 1.0:
            raise ValueError(f"win_ratio must be >= 1, got {self.win_ratio}")
        if not 0.0 <= self.keep_fraction <= 1.0 or self.keep_min < 1:
            raise ValueError("keep_fraction must be in [0, 1] and keep_min >= 1")
        if self.tile < _STRIDE or self.tile % _STRIDE:
            raise ValueError(f"tile must be a positive multiple of {_STRIDE}, got {self.tile}")
        if self.tile_margin < 0 or self.tile_margin % _STRIDE:
            raise ValueError(f"tile_margin must be a multiple of {_STRIDE} and >= 0, got {self.tile_margin}")

    def extraction_key(self) -> dict:
        """The parameters that change the map's features: what the cache file name depends on."""
        keys = ("map_top_k", "detection_threshold", "tile", "tile_margin")
        return {k: v for k, v in asdict(self).items() if k in keys}


class _MapFeatures:
    """Map keypoints (N x 2 float32, full-map px) and their descriptors (N x 64 torch tensor on the device)."""

    __slots__ = ("desc", "pts")

    def __init__(self, pts: np.ndarray, desc):
        self.pts, self.desc = pts, desc

    def __len__(self) -> int:
        return len(self.pts)


class XFeatMatcher:
    """A :class:`~dronenav.types.Matcher` backend with XFeat features and a disk cache for the map's.

    map_img: optional full map to register now; otherwise the first full-map ``match`` call (``map_offset ==
    (0, 0)``) registers it. cache_dir: where map features are cached; None keeps them in memory only.
    device: "auto" / None, "mps", "cuda" or "cpu" (see :func:`dronenav.device.pick_device`). weights: the
    weights file; None looks where :func:`xfeat_weights_path` says. download: fetch the weights when the file is
    not there (False: raise :class:`XFeatWeightsMissing` instead, for tests and machines without a network).

    ``map_extractions`` and ``map_features_source`` are as in RootSiftMatcher. ``frame_extractions`` counts
    network passes over frames (one per turn tried) and ``turn_deg`` is the turn the last matches were found
    at, None before the first match and when there were none: together they show whether a frame took the short
    path.
    """

    backend = BACKEND

    def __init__(self, map_img: np.ndarray | None = None, cache_dir: str | Path | None = "outputs/cache",
                 cfg: XFeatConfig = XFeatConfig(), device: str | None = None, weights: str | Path | None = None,
                 download: bool = True):
        self.cfg = cfg
        self.device = pick_device(device)         # first: a wrong device name is wrong with or without weights
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.weights = Path(weights).expanduser() if weights is not None else xfeat_weights_path()
        self.map_extractions = 0
        self.map_features_source: str | None = None
        self.frame_extractions = 0
        self.turn_deg: float | None = None
        self._kept: tuple[float, int] | None = None   # the remembered turn and how many matches it won with
        self._torch, self._net, self._weights_key = _load_network(self.weights, self.device, cfg, download)
        self._map: np.ndarray | None = None       # registered map, grayscale
        self._map_feats: _MapFeatures | None = None
        if map_img is not None:
            self.set_map(map_img)

    # --- the map and its cache -----------------------------------------------------------------------------

    def set_map(self, map_img: np.ndarray) -> None:
        """Register the full map: load its features from the cache, or extract and cache them."""
        gray = _gray(map_img)
        path = self._cache_path(gray)
        loaded = self._load(path) if path is not None else None
        if loaded is None:
            pts, desc = self._extract_map(gray)
            self.map_extractions += 1
            self.map_features_source = "extracted"
            if path is not None:
                self._save(path, pts, desc)
        else:
            pts, desc = loaded
            self.map_features_source = "cache"
        # float16 on disk; the same rounding in memory, so a run from the cache matches exactly like the first
        desc = self._torch.from_numpy(desc.astype(np.float32)).to(self.device)
        self._map, self._map_feats = gray, _MapFeatures(pts, desc)

    def _cache_path(self, gray: np.ndarray) -> Path | None:
        if self.cache_dir is None:
            return None
        h = hashlib.sha1(f"{gray.shape}".encode())
        h.update(np.ascontiguousarray(gray).data)
        # the weights are part of the key: other weights give other features for the same pixels
        params = json.dumps({"v": _CACHE_VERSION, "weights": self._weights_key, **self.cfg.extraction_key()},
                            sort_keys=True)
        key = hashlib.sha1(params.encode()).hexdigest()[:8]
        return self.cache_dir / f"{BACKEND}_{h.hexdigest()[:16]}_{key}.npz"

    @staticmethod
    def _load(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        """The cached (pts, float16 descriptors), or None if the file is missing or unreadable."""
        if not path.exists():
            return None
        try:
            with np.load(path) as z:
                pts, desc = z["pts"].astype(np.float32), z["desc"].astype(np.float16)
            if pts.ndim != 2 or pts.shape[1] != 2 or desc.shape != (len(pts), _DESC_DIM):
                return None
            return pts, desc
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
            return None   # a truncated or foreign file is a cache miss, never a crash

    @staticmethod
    def _save(path: Path, pts: np.ndarray, desc: np.ndarray) -> None:
        """Write atomically (temp file, then rename), so a crash or a parallel run never leaves half a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, pts=pts, desc=desc)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _extract_map(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Keypoints (N x 2 float32, px of ``gray``) and descriptors (N x 64 float16) of a map image, in tiles.

        A tile is described together with ``tile_margin`` px around it and only the keypoints inside the tile
        itself are kept: every map pixel belongs to one tile, and its keypoint has the context it would have in
        the whole image. Descriptors are stored as float16: half the memory and disk, and the rounding (1e-4
        on unit vectors) is far below the gap between a match and a non-match.
        """
        cfg, (h, w) = self.cfg, gray.shape
        all_pts, all_desc = [], []
        for y0 in range(0, h, cfg.tile):
            for x0 in range(0, w, cfg.tile):
                x1, y1 = min(x0 + cfg.tile, w), min(y0 + cfg.tile, h)
                ax, ay = max(x0 - cfg.tile_margin, 0), max(y0 - cfg.tile_margin, 0)
                bx, by = min(x1 + cfg.tile_margin, w), min(y1 + cfg.tile_margin, h)
                area_mpx = (bx - ax) * (by - ay) / 1e6
                pts, desc = self._describe(gray[ay:by, ax:bx], max(1, math.ceil(cfg.map_top_k * area_mpx)))
                pts = pts + np.float32([ax, ay])
                keep = (pts[:, 0] >= x0) & (pts[:, 0] < x1) & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
                all_pts.append(pts[keep])
                all_desc.append(desc.cpu().numpy()[keep].astype(np.float16))
        return np.concatenate(all_pts), np.concatenate(all_desc)

    def _describe(self, gray: np.ndarray, top_k: int):
        """One network pass: keypoints (N x 2 float32, corner-origin px of ``gray``) and unit descriptors (N x 64
        tensor on the device), best first, at most ``top_k``."""
        torch = self._torch
        h, w = gray.shape
        # Padded to the stride, not resized (kornia would resize): a keypoint is then a pixel of the input. The
        # padding repeats the edge, which has no corners to detect; anything found in it is dropped below.
        ph, pw = -h % _STRIDE, -w % _STRIDE
        if ph or pw:
            gray = cv2.copyMakeBorder(gray, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
        x = torch.from_numpy(np.ascontiguousarray(gray)).to(self.device)[None, None].float() / 255.0
        with torch.inference_mode():
            out = self._net.detectAndCompute(x, top_k=top_k)[0]
        # The network's keypoints are pixel indices (pixel centres at integers); ours have centres at +0.5.
        pts = out["keypoints"].cpu().numpy().astype(np.float32) + np.float32(0.5)
        inside = (pts[:, 0] < w) & (pts[:, 1] < h)
        desc = out["descriptors"].float()
        if not inside.all():
            pts, desc = pts[inside], desc[torch.from_numpy(inside).to(self.device)]
        return pts, desc

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
        frame = _gray(frame_img)

        feats = self._map_features(win, ox, oy, mask)
        if len(feats) < 2:
            return Matches.empty(BACKEND, _ms(t0))

        found = {}                                    # turn -> (pts_frame, map indices, scores)
        if self._kept is not None:
            turn, won_with = self._kept
            found[turn] = self._match_at(frame, turn, feats)
            if len(found[turn][0]) >= max(self.cfg.keep_min, self.cfg.keep_fraction * won_with):
                self.turn_deg = turn
                return self._matches(found[turn], feats, t0)
        step = 360.0 / self.cfg.turns
        for k in range(self.cfg.turns):
            if k * step not in found:
                found[k * step] = self._match_at(frame, k * step, feats)
        count = {turn: len(f[0]) for turn, f in found.items()}
        best = max(sorted(count), key=count.get)      # a tie goes to the smaller turn
        # A clear win: over every turn that is not a neighbour (a heading between two turns matches at both).
        far = [n for turn, n in count.items() if min(abs(turn - best), 360.0 - abs(turn - best)) > step + 1e-9]
        if count[best] >= self.cfg.keep_min and count[best] >= self.cfg.win_ratio * max(far, default=0):
            self._kept = (best, count[best])
        self.turn_deg = best if count[best] else None
        return self._matches(found[best], feats, t0)

    @staticmethod
    def _matches(found, feats: _MapFeatures, t0: float) -> Matches:
        pts_frame, ti, score = found
        if not len(pts_frame):
            return Matches.empty(BACKEND, _ms(t0))
        return Matches(pts_frame, feats.pts[ti], score, BACKEND, _ms(t0))

    def _match_at(self, frame: np.ndarray, turn_deg: float, feats: _MapFeatures):
        """Describe the frame turned by ``turn_deg`` and match it: frame points (in the frame as given), the
        indices of their map features, and the scores."""
        turned, A = _turn(frame, turn_deg)
        self.frame_extractions += 1
        pts, desc = self._describe(turned, self.cfg.top_k)
        # back to the frame as given: p = R^T (p' - t), in row form (p' - t) @ R
        pts = ((pts - A[:, 2]) @ A[:, :2]).astype(np.float32)
        h, w = frame.shape
        inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)   # not in the border fill
        if not inside.all():
            pts, desc = pts[inside], desc[self._torch.from_numpy(inside).to(self.device)]
        if not len(pts):
            return pts, np.zeros(0, int), np.zeros(0)
        qi, ti, score = self._mutual(desc, feats.desc)
        return pts[qi], ti, score

    def _mutual(self, a, b):
        """Mutual nearest neighbours between unit descriptors ``a`` (frame) and ``b`` (map) that pass the cosine
        floor and the ratio test: indices into a, indices into b, scores (the margin over the runner-up)."""
        torch = self._torch
        n = len(a)
        top = torch.full((n, 2), -2.0, device=a.device)             # best and second-best cosine per frame point
        arg = torch.zeros(n, dtype=torch.long, device=a.device)      # which map point is the best
        back = []                                                    # per map point: its best cosine to a frame point
        with torch.inference_mode():
            for s in range(0, len(b), _MATCH_BLOCK):
                sim = a @ b[s:s + _MATCH_BLOCK].T
                back.append(sim.amax(0))      # the value, not the index: several times faster down a column
                k = min(2, sim.shape[1])
                v, i = sim.topk(k, dim=1)
                both = torch.cat([top, v], dim=1)
                top, pick = both.topk(2, dim=1)
                arg = torch.where(pick[:, 0] >= 2, i[:, 0] + s, arg)   # columns 0-1 are the earlier blocks' best
            back = torch.cat(back)
            rows = torch.arange(n, device=a.device)
            # on unit vectors, distance^2 = 2 - 2 cos: Lowe's test on the distances
            d1 = (2.0 - 2.0 * top[:, 0]).clamp_min(0.0).sqrt()
            d2 = (2.0 - 2.0 * top[:, 1]).clamp_min(1e-12).sqrt()
            # mutual: nothing on the frame is closer to this map point (the same number from the same product)
            keep = (top[:, 0] >= back[arg]) & (top[:, 0] > self.cfg.min_cossim) & (d1 <= self.cfg.ratio * d2)
            qi, ti = rows[keep].cpu().numpy(), arg[keep].cpu().numpy()
            score = (1.0 - d1[keep] / d2[keep]).cpu().numpy().astype(np.float64)
        return qi, ti, score

    def _map_features(self, win: np.ndarray, ox: int, oy: int, mask: np.ndarray | None) -> _MapFeatures:
        """Map features for this query, in full-map px.

        A window of the registered map: the cached keypoints inside it (no extraction). A new full map (offset
        0, 0): register it (cache). A window of a map never registered: extract it for this call only.
        """
        torch = self._torch
        if self._is_window_of_map(win, ox, oy):
            f = self._map_feats
            h, w = win.shape
            keep = (f.pts[:, 0] >= ox) & (f.pts[:, 0] < ox + w) & (f.pts[:, 1] >= oy) & (f.pts[:, 1] < oy + h)
            if mask is not None:
                idx = np.flatnonzero(keep)
                c = (f.pts[idx] - (ox, oy)).astype(int)        # inside the window, so valid indices
                keep[idx] = np.asarray(mask)[c[:, 1], c[:, 0]] != 0
            if keep.all():
                return f
            return _MapFeatures(f.pts[keep], f.desc[torch.from_numpy(np.flatnonzero(keep)).to(self.device)])
        if (ox, oy) == (0, 0):
            self.set_map(win)
            return self._map_features(win, 0, 0, mask)
        pts, desc = self._extract_map(win)
        if mask is not None and len(pts):
            c = pts.astype(int)
            keep = np.asarray(mask)[c[:, 1], c[:, 0]] != 0
            pts, desc = pts[keep], desc[keep]
        return _MapFeatures(pts + np.float32([ox, oy]), torch.from_numpy(desc.astype(np.float32)).to(self.device))

    def _is_window_of_map(self, win: np.ndarray, ox: int, oy: int) -> bool:
        """True when ``win`` is exactly the registered map's pixels at (ox, oy): only then are cached keypoints
        valid for it."""
        if self._map is None or ox < 0 or oy < 0:
            return False
        h, w = win.shape
        if oy + h > self._map.shape[0] or ox + w > self._map.shape[1]:
            return False
        return np.array_equal(self._map[oy:oy + h, ox:ox + w], win)


def _turn(gray: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray]:
    """``gray`` turned clockwise by ``deg`` about its centre, on a canvas that holds all of it, and the 2 x 3
    matrix A that takes a corner-origin point of ``gray`` to the turned image: p' = A[:, :2] @ p + A[:, 2].

    Quarter turns move pixels without resampling, so they are exact. Other turns are resampled; the canvas
    outside the frame is filled by reflection (a hard black edge would grow keypoints of its own), and the
    caller drops what is found there.
    """
    h, w = gray.shape
    quarter = deg / 90.0
    if quarter == int(quarter):
        k = int(quarter) % 4
        c, s = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[k]
        out = np.ascontiguousarray(np.rot90(gray, -k))
        W, H = out.shape[1], out.shape[0]
    else:
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        W = math.ceil(w * abs(c) + h * abs(s))
        H = math.ceil(w * abs(s) + h * abs(c))
        out = None
    R = np.array([[c, -s], [s, c]])
    A = np.c_[R, np.array([W / 2.0, H / 2.0]) - R @ np.array([w / 2.0, h / 2.0])]
    if out is None:
        # cv2.warpAffine works in pixel-centre coordinates: conjugate our corner-origin matrix by the half pixel
        half = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
        A_cv = (np.linalg.inv(half) @ np.vstack([A, [0.0, 0.0, 1.0]]) @ half)[:2]
        out = cv2.warpAffine(gray, A_cv, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out, A


def _load_network(path: Path, device: str, cfg: XFeatConfig, download: bool):
    """(torch module, kornia XFeat in eval mode on the device, a short hash of the weights file)."""
    try:
        import torch
        from kornia.feature import XFeat
    except ImportError as e:
        raise ImportError("the XFeat matcher needs torch and a kornia with kornia.feature.XFeat "
                          f"(pip install -r requirements.txt): {e}") from e
    if not path.is_file():
        if not download:
            raise XFeatWeightsMissing(
                f"XFeat weights are not at {path}. Allow the download (about 6 MB, once, from {WEIGHTS_URL}) "
                f"or put the file there, or point {WEIGHTS_ENV} or the weights argument at it.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS_URL, str(path), progress=False)   # writes a temp file, then renames
    net = XFeat(top_k=cfg.top_k, detection_threshold=cfg.detection_threshold)
    try:
        # weights_only: a checkpoint is data; never let a downloaded file run code when it is unpickled
        net.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    except Exception as e:   # torch raises several unrelated types for a bad file; all mean the same here
        raise ValueError(f"{path} is not an XFeat weights file: {e}") from e
    return torch, net.eval().to(device), hashlib.sha1(path.read_bytes()).hexdigest()[:12]
