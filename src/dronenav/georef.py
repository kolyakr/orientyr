"""Georef: a map's pixels placed on Earth — lon/lat and metres, and the sidecar JSON that ties them together.

A map image may come with a sidecar JSON next to it (same path, ``.json`` instead of the image's extension)
describing where it sits in Web-Mercator space: ``{"capture": {...}, "source": "...", "map": {...},
"hires": {...}}``, each of ``map``/``hires`` one resolution of the same area, e.g.::

    {"file": "maps/irpin_2025.png", "zoom": 17, "m_per_px": 0.759, "size_px": [3936, 3936],
     "merc_origin_px": [19592192, 11301728], "bounds_lonlat": [30.2014, 50.5086, 30.2436, 50.5355],
     "capture_coverage": 1.0, "alignment_to_2025": {"shift_px": [0.0, 0.0], "reference": true}}

- ``bounds_lonlat`` is ``[min_lon, min_lat, max_lon, max_lat]`` (west, south, east, north); pixel
  ``(0, 0)`` is the north-west corner (slippy-map tile orientation: x east, y south).
- ``merc_origin_px`` is the block's pixel ``(0, 0)`` in *world* Web-Mercator pixels at its own zoom
  (tile size 256); a block's own pixel is the world pixel minus this origin.
- ``m_per_px`` is the sidecar's own measured ground resolution; metre conversions use it directly rather
  than recomputing from latitude, since it is already a per-map calibrated value.
- ``alignment_to_2025`` varies in shape between captures (the reference year carries no residual
  fields) and is not interpreted here; a caller that needs it reads the dict itself.

A map with no sidecar (the task map) has no georeference at all: :func:`load_map` returns ``None`` for
it, not an error — everything outside this module only ever works in map pixels (see ``dronenav/__init__.py``).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

__all__ = ["Georef", "GeorefBlock", "GeorefError", "load_georef", "load_map"]

_TILE_PX = 256  # Web-Mercator tile size, as used throughout experiments/

_BLOCK_REQUIRED = ("file", "zoom", "m_per_px", "size_px", "merc_origin_px", "bounds_lonlat",
                   "capture_coverage", "alignment_to_2025")
_TOP_REQUIRED = ("capture", "source", "map", "hires")


class GeorefError(ValueError):
    """A sidecar that does not follow the georef format. A ValueError: bad input."""


def _world_px(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """lon/lat (WGS84) -> Web-Mercator pixel of the whole world at ``zoom``, tile size 256."""
    n = 2 ** zoom * _TILE_PX
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _world_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    """Inverse of :func:`_world_px`."""
    n = 2 ** zoom * _TILE_PX
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def _is_number(v) -> bool:
    # bool is an int in Python, but true/false as a coordinate is a bug, not a number
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _number(name: str, v) -> float:
    if not _is_number(v):
        raise GeorefError(f"{name} must be a number, got {type(v).__name__}")
    v = float(v)
    if not math.isfinite(v):
        raise GeorefError(f"{name} must be finite, got {v}")
    return v


def _point(name: str, v, n: int) -> tuple[float, ...]:
    """A sequence of exactly n numbers, as a tuple of floats."""
    if not isinstance(v, (list, tuple)) or len(v) != n:
        raise GeorefError(f"{name} must be {n} numbers, got {v!r}")
    return tuple(_number(f"{name}[{i}]", x) for i, x in enumerate(v))


def _size_px(name: str, v) -> tuple[int, int]:
    """(w, h): two positive ints, from a tuple or a JSON list."""
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise GeorefError(f"{name} must be [w, h], got {v!r}")
    w, h = v
    if not all(isinstance(e, int) and not isinstance(e, bool) and e > 0 for e in (w, h)):
        raise GeorefError(f"{name} must be two positive integers, got {v!r}")
    return int(w), int(h)


@dataclass(frozen=True)
class GeorefBlock:
    """One resolution of a georeferenced map (the sidecar's ``map`` or ``hires`` block).

    file: image path, relative to the sidecar's own folder. zoom: Web-Mercator zoom level, tile size 256.
    m_per_px: ground resolution, measured, metres per pixel. size_px: (w, h) of the image.
    merc_origin_px: (x, y) of this block's pixel (0, 0) in *world* Web-Mercator pixels at ``zoom``.
    bounds_lonlat: (min_lon, min_lat, max_lon, max_lat) = (west, south, east, north) covered by the image.
    capture_coverage: fraction of the area actually imaged, in [0, 1]. alignment_to_2025: this capture's
    registration to the 2025 reference; shape varies by capture (see module docstring) — read as-is.
    """

    file: str
    zoom: int
    m_per_px: float
    size_px: tuple[int, int]
    merc_origin_px: tuple[float, float]
    bounds_lonlat: tuple[float, float, float, float]
    capture_coverage: float
    alignment_to_2025: dict

    def __post_init__(self):
        if not isinstance(self.zoom, int) or isinstance(self.zoom, bool) or self.zoom < 0:
            raise GeorefError(f"zoom must be a non-negative integer, got {self.zoom!r}")
        m_per_px = _number("m_per_px", self.m_per_px)
        if not m_per_px > 0:
            raise GeorefError(f"m_per_px must be > 0, got {m_per_px!r}")
        object.__setattr__(self, "m_per_px", m_per_px)
        capture_coverage = _number("capture_coverage", self.capture_coverage)
        if not 0.0 <= capture_coverage <= 1.0:
            raise GeorefError(f"capture_coverage must be in [0, 1], got {capture_coverage!r}")
        object.__setattr__(self, "capture_coverage", capture_coverage)
        object.__setattr__(self, "size_px", _size_px("size_px", self.size_px))
        object.__setattr__(self, "merc_origin_px", _point("merc_origin_px", self.merc_origin_px, 2))
        object.__setattr__(self, "bounds_lonlat", _point("bounds_lonlat", self.bounds_lonlat, 4))
        if not isinstance(self.alignment_to_2025, dict):
            raise GeorefError(f"alignment_to_2025 must be an object, got {type(self.alignment_to_2025).__name__}")

    def lonlat_to_px(self, lon: float, lat: float) -> tuple[float, float]:
        """WGS84 lon/lat -> this block's pixel (x, y)."""
        x, y = _world_px(lon, lat, self.zoom)
        ox, oy = self.merc_origin_px
        return x - ox, y - oy

    def px_to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        """Inverse of :meth:`lonlat_to_px`."""
        ox, oy = self.merc_origin_px
        return _world_lonlat(x + ox, y + oy, self.zoom)

    def px_to_m(self, dx: float, dy: float) -> tuple[float, float]:
        """A pixel offset (dx, dy) -> metres, using the block's own measured resolution."""
        return dx * self.m_per_px, dy * self.m_per_px

    def m_to_px(self, dx: float, dy: float) -> tuple[float, float]:
        """Inverse of :meth:`px_to_m`."""
        return dx / self.m_per_px, dy / self.m_per_px


@dataclass(frozen=True)
class Georef:
    """A map's full georeference sidecar: one capture, at up to two resolutions.

    capture: {"date", "sensor"} of the imagery. source: free-text provenance string.
    map: the block most of the pipeline matches against. hires: a finer-resolution crop of the same area
    (sometimes identical to ``map``, when no finer capture exists).
    """

    capture: dict
    source: str
    map: GeorefBlock
    hires: GeorefBlock

    def __post_init__(self):
        if not isinstance(self.capture, dict):
            raise GeorefError(f"capture must be an object, got {type(self.capture).__name__}")
        if not isinstance(self.source, str):
            raise GeorefError(f"source must be a string, got {type(self.source).__name__}")
        for name in ("map", "hires"):
            if not isinstance(getattr(self, name), GeorefBlock):
                raise GeorefError(f"{name} must be a GeorefBlock, got {type(getattr(self, name)).__name__}")

    def reproject(self, x: float, y: float, *, src: str, dst: str) -> tuple[float, float]:
        """A pixel in the ``src`` block ("map"/"hires") -> the same location in the ``dst`` block."""
        for name in (src, dst):
            if name not in ("map", "hires"):
                raise GeorefError(f"src/dst must be 'map' or 'hires', got {name!r}")
        lon, lat = getattr(self, src).px_to_lonlat(x, y)
        return getattr(self, dst).lonlat_to_px(lon, lat)


class _NonFinite(Exception):
    pass


def _reject_constant(token: str):
    raise _NonFinite(token)


def _check_keys(d: dict, required: tuple, prefix: str) -> None:
    for k in required:
        if k not in d:
            raise GeorefError(f"missing field '{prefix}{k}'")
    for k in d:
        if k not in required:
            raise GeorefError(f"unknown field '{prefix}{k}'")


def _block_from_dict(d, name: str) -> GeorefBlock:
    if not isinstance(d, dict):
        raise GeorefError(f"'{name}' must be an object, got {type(d).__name__}")
    _check_keys(d, _BLOCK_REQUIRED, prefix=f"{name}.")
    return GeorefBlock(file=d["file"], zoom=d["zoom"], m_per_px=d["m_per_px"], size_px=d["size_px"],
                        merc_origin_px=d["merc_origin_px"], bounds_lonlat=d["bounds_lonlat"],
                        capture_coverage=d["capture_coverage"], alignment_to_2025=d["alignment_to_2025"])


def load_georef(path: str | Path) -> Georef:
    """Read and validate a georef sidecar JSON. Raises GeorefError naming the path and field."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")  # FileNotFoundError names the path
    try:
        d = json.loads(text, parse_constant=_reject_constant)
        if not isinstance(d, dict):
            raise GeorefError(f"must be a JSON object, got {type(d).__name__}")
        _check_keys(d, _TOP_REQUIRED, prefix="")
        return Georef(capture=d["capture"], source=d["source"],
                      map=_block_from_dict(d["map"], "map"), hires=_block_from_dict(d["hires"], "hires"))
    except _NonFinite as e:
        raise GeorefError(f"{path}: {e} is not a finite number (NaN and Infinity are not JSON)") from None
    except json.JSONDecodeError as e:
        raise GeorefError(f"{path}: invalid JSON: {e.msg} at line {e.lineno}") from None
    except GeorefError as e:
        raise GeorefError(f"{path}: {e}") from None


def load_map(map_path: str | Path) -> tuple[np.ndarray, Georef | None]:
    """Read a map image and its sidecar, if one sits next to it (same path, '.json'). No sidecar -> georef=None.

    Raises FileNotFoundError for an unreadable map and GeorefError for a malformed sidecar.
    """
    map_path = Path(map_path)
    image = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read the map {map_path}")
    sidecar = map_path.with_suffix(".json")
    return image, (load_georef(sidecar) if sidecar.exists() else None)
