"""Build a cluster map from ONE satellite capture (one date, one sensor), with no seams.

Esri Wayback releases are mosaics: each one shows a given capture only where nothing newer was
laid over it. But the capture itself stays the same image across releases. So we query every
release's metadata, rasterise that capture's footprint at full resolution, and take its pixels
from whichever releases show them. Pixels outside the capture's footprint are never used.

Where two releases both show the capture, the pixels must agree. We measure that as a check that
we are really stitching one photograph.

Output: <out>.png (RGB, black where the capture is missing), <out>_valid_mask.png (255 = capture),
<out>.json (grid sidecar + per-release provenance + overlap check).

Usage: python single_date_map.py <grid_json> <date> <sensor> <out_prefix>
  e.g. python single_date_map.py ../../data/kyiv_cluster/maps/cluster_2025_rebuilt.json \
         2017-05-05 WV02 ../../data/kyiv_cluster/maps/cluster_2017-05-05
"""
import datetime as dt
import io
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image

from coverage import CONFIG, SESSION, tile_to_lonlat

TILE = ("https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/"
        "WMTS/1.0.0/default028mm/MapServer/tile/{rel}/{z}/{y}/{x}")
ERODE = 4  # px kept away from footprint edges: polygon edges are approximate, pixels beyond them may be another capture


def footprint(release: str, meta: dict, cfg: dict, date: str, sensor: str) -> np.ndarray:
    """Full-resolution mask (uint8 0/1) of where `release` shows the capture (date, sensor)."""
    z, t0x, t0y, nt = meta["zoom"], meta["tile_x0"], meta["tile_y0"], meta["tiles"]
    lon0, lat1 = tile_to_lonlat(t0x, t0y, z)
    lon1, lat0 = tile_to_lonlat(t0x + nt, t0y + nt, z)
    url = cfg[release]["metadataLayerUrl"] + f"/{max(0, 23 - z)}/query"
    r = SESSION.get(url, params=dict(geometry=f"{lon0},{lat0},{lon1},{lat1}", geometryType="esriGeometryEnvelope",
                                     inSR=4326, outSR=4326, spatialRel="esriSpatialRelIntersects",
                                     outFields="SRC_DATE2,SRC_DESC", returnGeometry="true", f="json"),
                    timeout=120).json()
    S, n = nt * 256, 2 ** z * 256
    # Later features are painted over earlier ones (as in coverage.py), so paint all, then keep our label.
    lab = np.zeros((S, S), np.uint8)
    for f in r.get("features", []):
        a = f["attributes"]
        ours = (dt.datetime.utcfromtimestamp(a["SRC_DATE2"] / 1000).strftime("%Y-%m-%d") == date
                and a["SRC_DESC"] == sensor)
        polys = [np.int32(np.round([((lon + 180) / 360 * n - t0x * 256,
                                     (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n - t0y * 256)
                                    for lon, lat in ring])) for ring in f["geometry"]["rings"]]
        cv2.fillPoly(lab, polys, 1 if ours else 2)
    return (lab == 1).astype(np.uint8)


def fetch_tiles(release: str, meta: dict, tiles: list[tuple[int, int]]) -> dict:
    """Download the given (row, col) tiles of one release."""
    z, t0x, t0y = meta["zoom"], meta["tile_x0"], meta["tile_y0"]

    def get(ij):
        i, j = ij
        r = SESSION.get(TILE.format(rel=release, z=z, y=t0y + i, x=t0x + j), timeout=60)
        r.raise_for_status()
        return ij, np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))

    with ThreadPoolExecutor(8) as ex:
        return dict(ex.map(get, tiles))


def main() -> None:
    grid_p, date, sensor, out = sys.argv[1:5]
    meta = json.load(open(grid_p))
    cfg = SESSION.get(CONFIG, timeout=30).json()
    S, nt = meta["tiles"] * 256, meta["tiles"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE + 1, 2 * ERODE + 1))

    # 1. Footprint of the capture in every release (a release dated before the capture cannot show it).
    rels = [r for r, v in cfg.items() if v["itemTitle"][-11:-1] >= date]
    masks = {}
    with ThreadPoolExecutor(6) as ex:
        for r, m in zip(rels, ex.map(lambda r: footprint(r, meta, cfg, date, sensor), rels)):
            if m.any():
                masks[r] = np.packbits(cv2.erode(m, kernel))  # 1 bit/px: ~13 MB per release
    print(f"{len(masks)} releases show {date} {sensor}", flush=True)

    # 2. Greedy cover: take the release that adds most new pixels, until nothing is added.
    img = np.zeros((S, S, 3), np.uint8)
    filled = np.zeros((S, S), bool)
    prov, diffs = [], []
    unpack = lambda b: np.unpackbits(b)[:S * S].reshape(S, S).astype(bool)
    while masks:
        r = max(masks, key=lambda k: (unpack(masks[k]) & ~filled).sum())
        m = unpack(masks.pop(r))
        new = m & ~filled
        if new.sum() < 256:  # less than one tile-row of pixels: rounding noise
            break
        tiles = [(i, j) for i in range(nt) for j in range(nt) if m[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256].any()]
        got = fetch_tiles(r, meta, tiles)
        for (i, j), t in got.items():
            sl = np.s_[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256]
            both = m[sl] & filled[sl]
            if both.sum() > 2000:  # overlap with an earlier release: the same photo must give the same pixels
                diffs.append(float(np.abs(img[sl][both].astype(int) - t[both].astype(int)).mean()))
            img[sl][new[sl]] = t[new[sl]]
        filled |= new
        prov.append({"release": int(r), "title": cfg[r]["itemTitle"], "new_frac": float(new.mean())})
        print(f"{cfg[r]['itemTitle']}: +{new.mean():.1%} -> {filled.mean():.1%}", flush=True)

    Image.fromarray(img).save(out + ".png")
    cv2.imwrite(out + "_valid_mask.png", filled.astype(np.uint8) * 255)
    check = {"tiles_compared": len(diffs),
             "mean_abs_diff": float(np.mean(diffs)) if diffs else None,
             "p95_abs_diff": float(np.percentile(diffs, 95)) if diffs else None}
    json.dump({**{k: meta[k] for k in ("zoom", "tile_x0", "tile_y0", "tiles", "m_per_px")},
               "capture": {"date": date, "sensor": sensor}, "valid_frac": float(filled.mean()),
               "source": "Esri World Imagery Wayback (Maxar), one capture only",
               "releases": prov, "overlap_check": check}, open(out + ".json", "w"), indent=1)
    print("valid", f"{filled.mean():.1%}", "overlap check", check)


if __name__ == "__main__":
    main()
