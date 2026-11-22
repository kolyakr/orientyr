"""M3-03: build the three West Irpin maps (2017/2022/2025) + two zoom-18 hires maps, one command.

Consolidates the exploratory ``experiments/ukraine_scan/west_irpin*.py`` scripts (and the single-date
mosaic approach of ``experiments/kyiv_cluster/single_date_map.py``) into the one committed entry point
the issue names. Per epoch:

1. **Single-capture mosaic** -- query every Esri Wayback release's metadata for the (date, sensor)
   capture's footprint, then greedily cover the target square from whichever releases show it (a
   release is a mosaic of many captures; the capture itself is one unchanging photograph -- see
   ``single_date_map.py``'s own docstring for why this works).
2. **Align to 2025** -- the reference epoch is 2025 (shift = 0); every other epoch is aligned to it by
   the *median* of per-tile phase correlation on Sobel-gradient images (gradients cancel the
   brightness/contrast differences between captures so phase correlation locks onto edges, not
   exposure; the median is robust to the few tiles with new construction or seasonal foliage).
3. **Cut** the fixed 3 km square (2017/2022/2025, zoom 17, 3936 px) and the 2022/2025 zoom-18 hires
   square (7872 px; no zoom-18 pre-war capture covers the area, so 2017 has no independent hires).

Usage: python experiments/irpin/build_maps.py [--out data/irpin] [--cache outputs/cache/irpin_build]
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

SOURCE = "Esri World Imagery Wayback (Maxar): one capture only, no mosaic"
WAYBACK_CONFIG = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
WAYBACK_TILE = ("https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/"
                "WMTS/1.0.0/default028mm/MapServer/tile/{rel}/{z}/{y}/{x}")
ERODE_PX = 4  # kept away from footprint-polygon edges, which are approximate (see single_date_map.py)

SEED = (30.2216, 50.501)  # centre of the West Irpin square (DATA_CARD.md, "Why West Irpin specifically")
EPOCHS = [("2017-05-05", "WV02"), ("2022-03-31", "WV03"), ("2025-04-11", "WV02")]
YEAR_OF = {date: date[:4] for date, _ in EPOCHS}
REF_DATE = "2025-04-11"
ZOOM, SIZE_PX = 17, 3936
HIRES_ZOOM, HIRES_SIZE_PX = 18, 7872
HIRES_DATES = {"2022-03-31", "2025-04-11"}  # pre-war 2017 has no zoom-18 capture of the area


# ---------------------------------------------------------------------------
# Geo math (Web Mercator), generalised to an arbitrary zoom -- scan.py's merc()/unmerc() are the same
# formula fixed at zoom 16; the shipped maps live at zoom 17 (map) and 18 (hires).
# ---------------------------------------------------------------------------

def merc(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    n = 2 ** zoom * 256
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def unmerc(x: float, y: float, zoom: int) -> tuple[float, float]:
    n = 2 ** zoom * 256
    lon = x / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def m_per_px(lat: float, zoom: int) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / 2 ** zoom


def build_grid(seed_lonlat: tuple[float, float], size_px: int, zoom: int, margin_tiles: int = 1) -> dict:
    """A 256-px-tile-aligned grid big enough to fetch `size_px` square around the seed, plus
    `margin_tiles` of room on every side for the sub-pixel alignment shift. `target_origin_px` and
    `fetch_origin_px` are both global Web-Mercator pixel coordinates at `zoom`; the mosaic array
    itself is indexed locally from `fetch_origin_px` (see `local_origin()`)."""
    lon, lat = seed_lonlat
    cx, cy = merc(lon, lat, zoom)
    target_tile_x0 = round((cx - size_px / 2) / 256)
    target_tile_y0 = round((cy - size_px / 2) / 256)
    target_origin_px = (target_tile_x0 * 256, target_tile_y0 * 256)
    tile_x0, tile_y0 = target_tile_x0 - margin_tiles, target_tile_y0 - margin_tiles
    fetch_origin_px = (tile_x0 * 256, tile_y0 * 256)
    fetch_tiles = math.ceil(size_px / 256) + 2 * margin_tiles
    return {"zoom": zoom, "tile_x0": tile_x0, "tile_y0": tile_y0, "fetch_tiles": fetch_tiles,
            "fetch_origin_px": fetch_origin_px, "target_origin_px": target_origin_px, "size_px": size_px}


def local_origin(grid: dict) -> tuple[int, int]:
    """`target_origin_px` as an offset into the mosaic array (which is indexed from 0 at
    `fetch_origin_px`), for `crop_with_shift`."""
    fx0, fy0 = grid["fetch_origin_px"]
    tx0, ty0 = grid["target_origin_px"]
    return tx0 - fx0, ty0 - fy0


# ---------------------------------------------------------------------------
# Single-date mosaic: footprint rasterization + greedy cover (single_date_map.py's algorithm)
# ---------------------------------------------------------------------------

def _wayback_date(src_date2_ms: float) -> str:
    return dt.datetime.fromtimestamp(src_date2_ms / 1000, tz=dt.UTC).strftime("%Y-%m-%d")


def rasterize_footprint(features: list[dict], grid: dict, date: str, sensor: str) -> np.ndarray:
    """Boolean mask (`fetch_tiles`*256 square) of pixels where `features` show exactly (date, sensor).
    Later features are painted over earlier ones, matching how Wayback composites a release."""
    zoom = grid["zoom"]
    ox, oy = grid["fetch_origin_px"]
    size = grid["fetch_tiles"] * 256
    lab = np.zeros((size, size), np.uint8)
    for f in features:
        a = f["attributes"]
        ours = _wayback_date(a["SRC_DATE2"]) == date and a["SRC_DESC"] == sensor
        polys = [np.int32(np.round([[merc(lon, lat, zoom)[0] - ox, merc(lon, lat, zoom)[1] - oy]
                                     for lon, lat in ring])) for ring in f["geometry"]["rings"]]
        cv2.fillPoly(lab, polys, 1 if ours else 2)
    return lab == 1


def select_cover(masks: dict, min_new_px: int = 256) -> list[tuple]:
    """Greedy set cover: repeatedly take the release whose mask adds the most still-uncovered
    pixels, until the best addition falls below `min_new_px` (rounding/edge noise, not a real
    contribution). Returns [(key, new_pixels_mask), ...] in the order chosen; each `new_pixels_mask`
    is disjoint from every one before it."""
    remaining = dict(masks)
    filled = None
    order = []
    while remaining:
        key = max(remaining, key=lambda k: int((remaining[k] & ~filled).sum() if filled is not None
                                                else remaining[k].sum()))
        m = remaining.pop(key)
        new = (m & ~filled) if filled is not None else m.copy()
        if new.sum() < min_new_px:
            break
        order.append((key, new))
        filled = new if filled is None else (filled | new)
    return order


def fetch_release_config(session: requests.Session) -> dict:
    return session.get(WAYBACK_CONFIG, timeout=30).json()


def _release_footprint(session, release, meta_url_base, grid, date, sensor, erode_kernel):
    z = grid["zoom"]
    ox, oy = grid["fetch_origin_px"]
    size = grid["fetch_tiles"] * 256
    lon0, lat1 = unmerc(ox, oy, z)
    lon1, lat0 = unmerc(ox + size, oy + size, z)
    url = meta_url_base + f"/{max(0, 23 - z)}/query"
    params = {"geometry": f"{lon0},{lat0},{lon1},{lat1}", "geometryType": "esriGeometryEnvelope",
              "inSR": 4326, "outSR": 4326, "spatialRel": "esriSpatialRelIntersects",
              "outFields": "SRC_DATE2,SRC_DESC", "returnGeometry": "true", "f": "json"}
    r = session.get(url, params=params, timeout=120).json()
    mask = rasterize_footprint(r.get("features", []), grid, date, sensor)
    return cv2.erode(mask.astype(np.uint8), erode_kernel).astype(bool)


def _fetch_tiles(session, grid, release, tile_ij):
    z, tx0, ty0 = grid["zoom"], grid["tile_x0"], grid["tile_y0"]

    def get(ij):
        i, j = ij
        r = session.get(WAYBACK_TILE.format(rel=release, z=z, y=ty0 + i, x=tx0 + j), timeout=60)
        r.raise_for_status()
        return ij, np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))[:, :, ::-1]  # RGB -> BGR

    with ThreadPoolExecutor(8) as ex:
        return dict(ex.map(get, tile_ij))


def build_single_date_mosaic(session: requests.Session, cfg: dict, grid: dict, date: str, sensor: str):
    """Network step (not unit tested; swapped for a fake in the end-to-end test). Returns (img_bgr,
    valid_mask) for the whole `grid` fetch square, built from whichever Wayback releases show the
    (date, sensor) capture, as in kyiv_cluster/single_date_map.py."""
    size = grid["fetch_tiles"] * 256
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE_PX + 1, 2 * ERODE_PX + 1))
    rels = [r for r, v in cfg.items() if v["itemTitle"][-11:-1] >= date]
    masks = {}
    with ThreadPoolExecutor(6) as ex:
        futs = {r: ex.submit(_release_footprint, session, r, cfg[r]["metadataLayerUrl"], grid, date, sensor, kernel)
                for r in rels}
        for r, fut in futs.items():
            m = fut.result()
            if m.any():
                masks[r] = m

    img = np.zeros((size, size, 3), np.uint8)
    valid = np.zeros((size, size), bool)
    nt = grid["fetch_tiles"]
    for release, new in select_cover(masks):
        tiles = [(i, j) for i in range(nt) for j in range(nt)
                 if new[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256].any()]
        got = _fetch_tiles(session, grid, release, tiles)
        for (i, j), t in got.items():
            sl = np.s_[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256]
            img[sl][new[sl]] = t[new[sl]]
        valid |= new
    return img, valid


# ---------------------------------------------------------------------------
# Alignment: median of tiled phase correlation on Sobel-gradient images
# ---------------------------------------------------------------------------

def gradient_magnitude(bgr_or_gray: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY) if bgr_or_gray.ndim == 3 else bgr_or_gray
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _tile_shifts(ref_g: np.ndarray, img_g: np.ndarray, valid: np.ndarray, tile: int, min_response: float):
    h, w = ref_g.shape
    win = cv2.createHanningWindow((tile, tile), cv2.CV_32F)
    shifts = []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            if not valid[y:y + tile, x:x + tile].all():
                continue
            (dx, dy), resp = cv2.phaseCorrelate(ref_g[y:y + tile, x:x + tile], img_g[y:y + tile, x:x + tile], win)
            if resp >= min_response:
                shifts.append((dx, dy))
    return shifts


def tiled_gradient_shift(ref_bgr: np.ndarray, img_bgr: np.ndarray, valid_mask: np.ndarray,
                          tile: int = 128, min_response: float = 0.05) -> dict:
    """Median sub-pixel shift that aligns `img_bgr` onto `ref_bgr`, from phase correlation on
    non-overlapping `tile`x`tile` Sobel-gradient blocks fully inside `valid_mask`. The returned
    `shift_px` = (dx, dy) follows the same convention already used by west_irpin_preview.py:
    ``cv2.warpAffine(img, [[1,0,-dx],[0,1,-dy]])`` aligns `img` onto `ref`.

    Raises ValueError if no tile qualifies -- too small/empty a valid mask, or no texture at all
    (phase correlation on a blank image is meaningless, not just imprecise).
    """
    if valid_mask.sum() == 0:
        raise ValueError("no valid pixels to align on")
    ref_g, img_g = gradient_magnitude(ref_bgr), gradient_magnitude(img_bgr)
    shifts = _tile_shifts(ref_g, img_g, valid_mask, tile, min_response)
    if not shifts:
        raise ValueError("no tile had enough texture/response for phase correlation")
    arr = np.array(shifts)
    shift_px = np.median(arr, axis=0)
    tile_spread_px = float(np.median(np.hypot(*(arr - shift_px).T)))

    h, w = img_bgr.shape[:2]
    corrected = cv2.warpAffine(img_bgr, np.float32([[1, 0, -shift_px[0]], [0, 1, -shift_px[1]]]), (w, h),
                                flags=cv2.INTER_LINEAR)
    residual_shifts = _tile_shifts(ref_g, gradient_magnitude(corrected), valid_mask, tile, min_response)
    residual_px = np.median(residual_shifts, axis=0) if residual_shifts else np.zeros(2)

    return {"shift_px": [float(shift_px[0]), float(shift_px[1])], "tiles": len(shifts),
            "tile_spread_px": tile_spread_px,
            "residual_after_px": [float(residual_px[0]), float(residual_px[1])]}


def crop_with_shift(mosaic: np.ndarray, local_origin_px: tuple[float, float], size_px: int,
                     shift_px: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Cut a `size_px`x`size_px` square out of `mosaic` (top-left `local_origin_px`), applying the
    sub-pixel alignment `shift_px` in the same resample (crop + shift in one interpolation pass, as
    west_irpin_cut.py already does)."""
    ox, oy = local_origin_px
    dx, dy = shift_px
    m = np.float32([[1, 0, -(ox + dx)], [0, 1, -(oy + dy)]])
    return cv2.warpAffine(mosaic, m, (size_px, size_px), flags=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Sidecar schema (matches data/irpin/maps/*.json): capture / source / map / hires
# ---------------------------------------------------------------------------

def coverage_fraction(valid_mask: np.ndarray) -> float:
    if valid_mask.size == 0:
        raise ValueError("empty valid mask")
    return float(valid_mask.mean())


def alignment_block(shift_px: tuple[float, float] = (0.0, 0.0), reference: bool = False,
                     tiles: int | None = None, tile_spread_px: float | None = None,
                     residual_after_px: tuple[float, float] | None = None) -> dict:
    if reference:
        return {"shift_px": [0.0, 0.0], "reference": True}
    if tiles is None or tile_spread_px is None or residual_after_px is None:
        raise ValueError("tiles/tile_spread_px/residual_after_px are required for a non-reference epoch")
    return {"shift_px": [float(shift_px[0]), float(shift_px[1])], "tiles": int(tiles),
            "tile_spread_px": float(tile_spread_px),
            "residual_after_px": [float(residual_after_px[0]), float(residual_after_px[1])]}


def map_block(file: str, zoom: int, origin_px: tuple[int, int], size_px: int, valid_mask: np.ndarray,
              alignment: dict) -> dict:
    ox, oy = origin_px
    lon0, lat1 = unmerc(ox, oy, zoom)
    lon1, lat0 = unmerc(ox + size_px, oy + size_px, zoom)
    return {"file": file, "zoom": zoom, "m_per_px": m_per_px((lat0 + lat1) / 2, zoom),
            "size_px": [size_px, size_px], "merc_origin_px": [ox, oy],
            "bounds_lonlat": [lon0, lat1, lon1, lat0],
            "capture_coverage": coverage_fraction(valid_mask), "alignment_to_2025": alignment}


def build_sidecar(capture: dict, map_block_: dict, hires_block: dict | None = None) -> dict:
    """`hires_block` defaults to `map_block_` when there is no independent hires capture (2017)."""
    return {"capture": dict(capture), "source": SOURCE, "map": map_block_, "hires": hires_block or map_block_}


def validate_build(sidecar: dict, max_residual_px: float = 1.0) -> None:
    """The issue's own "Done when" bar: 100% single-capture coverage (the valid mask is the whole
    square) and an alignment residual under `max_residual_px`. Raises ValueError, loudly, rather than
    writing out-of-spec data."""
    for key in ("map", "hires"):
        block = sidecar[key]
        if block["capture_coverage"] < 1.0:
            raise ValueError(f"{key}: capture_coverage {block['capture_coverage']:.4f} < 1.0")
        align = block["alignment_to_2025"]
        if align.get("reference"):
            continue
        residual = math.hypot(*align["residual_after_px"])
        if residual >= max_residual_px:
            raise ValueError(f"{key}: alignment residual {residual:.3f}px >= {max_residual_px}px")


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/irpin", help="output dir (gets maps/ and hires/ subfolders)")
    p.add_argument("--cache", default="outputs/cache/irpin_build", help="cache dir for the Wayback fetch")
    p.add_argument("--max-residual-px", type=float, default=1.0,
                    help="validate_build()'s alignment-residual bar")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    os.makedirs(args.cache, exist_ok=True)
    session = requests.Session()
    cfg = fetch_release_config(session)

    grid = build_grid(SEED, SIZE_PX, ZOOM)
    hgrid = build_grid(SEED, HIRES_SIZE_PX, HIRES_ZOOM)

    mosaics = {date: build_single_date_mosaic(session, cfg, grid, date, sensor) for date, sensor in EPOCHS}
    hires_mosaics = {date: build_single_date_mosaic(session, cfg, hgrid, date, sensor)
                      for date, sensor in EPOCHS if date in HIRES_DATES}
    ref_img, ref_valid = mosaics[REF_DATE]
    hires_ref_img, hires_ref_valid = hires_mosaics[REF_DATE]

    out = Path(args.out)
    (out / "maps").mkdir(parents=True, exist_ok=True)
    (out / "hires").mkdir(parents=True, exist_ok=True)

    for date, sensor in EPOCHS:
        year = YEAR_OF[date]
        img, valid = mosaics[date]
        reference = date == REF_DATE
        if reference:
            align = alignment_block(reference=True)
        else:
            shift = tiled_gradient_shift(ref_img, img, valid & ref_valid)
            align = alignment_block(shift["shift_px"], tiles=shift["tiles"],
                                     tile_spread_px=shift["tile_spread_px"],
                                     residual_after_px=shift["residual_after_px"])
        shift_px = tuple(align["shift_px"])
        crop = crop_with_shift(img, local_origin(grid), SIZE_PX, shift_px)
        crop_valid = crop_with_shift(valid.astype(np.uint8) * 255, local_origin(grid),
                                      SIZE_PX, shift_px) > 0
        cv2.imwrite(str(out / "maps" / f"irpin_{year}.png"), crop)
        mblock = map_block(f"maps/irpin_{year}.png", ZOOM, grid["target_origin_px"], SIZE_PX, crop_valid, align)

        hblock = None
        if date in HIRES_DATES:
            himg, hvalid = hires_mosaics[date]
            if reference:
                halign = alignment_block(reference=True)
            else:
                hshift = tiled_gradient_shift(hires_ref_img, himg, hvalid & hires_ref_valid)
                halign = alignment_block(hshift["shift_px"], tiles=hshift["tiles"],
                                          tile_spread_px=hshift["tile_spread_px"],
                                          residual_after_px=hshift["residual_after_px"])
            hshift_px = tuple(halign["shift_px"])
            hcrop = crop_with_shift(himg, local_origin(hgrid), HIRES_SIZE_PX, hshift_px)
            hcrop_valid = crop_with_shift(hvalid.astype(np.uint8) * 255, local_origin(hgrid),
                                           HIRES_SIZE_PX, hshift_px) > 0
            cv2.imwrite(str(out / "hires" / f"irpin_{year}_z18.png"), hcrop)
            hblock = map_block(f"hires/irpin_{year}_z18.png", HIRES_ZOOM, hgrid["target_origin_px"],
                                HIRES_SIZE_PX, hcrop_valid, halign)

        sidecar = build_sidecar({"date": date, "sensor": sensor}, mblock, hblock)
        validate_build(sidecar, args.max_residual_px)
        with open(out / "maps" / f"irpin_{year}.json", "w") as f:
            json.dump(sidecar, f, indent=1)
        residual = math.hypot(*align["residual_after_px"]) if not align.get("reference") else 0.0
        print(f"{year}: coverage {mblock['capture_coverage']:.1%}, residual {residual:.3f}px")


if __name__ == "__main__":
    main()
