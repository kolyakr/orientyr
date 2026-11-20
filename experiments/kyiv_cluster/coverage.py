"""Which source captures make up a Wayback release inside our map extent?

Queries the release's metadata layer with the map envelope and rasterises the capture
polygons onto the map grid (downscaled), so seams between captures can be located exactly.

Usage: python coverage.py <map_json> <out_json> <release> [<release> ...]
"""
import datetime as dt
import json
import math
import sys

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(total=6, backoff_factor=2,
                                                        status_forcelist=(429, 500, 502, 503, 504))))
CONFIG = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
DOWN = 8  # raster coverage at 1/8 of map resolution


def tile_to_lonlat(x, y, z):
    n = 2 ** z
    return x / n * 360 - 180, math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))


def coverage(release, meta, cfg):
    """Return [(date, desc, frac)], label raster (1/DOWN), list of dates by label index."""
    z, t0x, t0y, nt = meta["zoom"], meta["tile_x0"], meta["tile_y0"], meta["tiles"]
    lon0, lat1 = tile_to_lonlat(t0x, t0y, z); lon1, lat0 = tile_to_lonlat(t0x + nt, t0y + nt, z)
    url = cfg[str(release)]["metadataLayerUrl"] + f"/{max(0, 23 - z)}/query"
    r = SESSION.get(url, params=dict(geometry=f"{lon0},{lat0},{lon1},{lat1}", geometryType="esriGeometryEnvelope",
                                     inSR=4326, outSR=4326, spatialRel="esriSpatialRelIntersects",
                                     outFields="SRC_DATE2,SRC_DESC,SRC_RES", returnGeometry="true", f="json"),
                    timeout=120).json()
    S = nt * 256 // DOWN
    n = 2 ** z * 256
    lab = np.zeros((S, S), np.int32)  # 0 = no metadata
    keys = []
    for f in r.get("features", []):
        a = f["attributes"]
        key = (dt.datetime.utcfromtimestamp(a["SRC_DATE2"] / 1000).strftime("%Y-%m-%d"), a["SRC_DESC"])
        if key not in keys:
            keys.append(key)
        polys = []
        for ring in f["geometry"]["rings"]:
            pts = [((lon + 180) / 360 * n - t0x * 256,
                    (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n - t0y * 256) for lon, lat in ring]
            polys.append(np.int32(np.round(np.array(pts) / DOWN)))
        cv2.fillPoly(lab, polys, keys.index(key) + 1)  # rings with holes are rare here; later fills win
    fr = [(k[0], k[1], float((lab == i + 1).mean())) for i, k in enumerate(keys)]
    return sorted(fr, key=lambda t: -t[2]), lab, keys


def main():
    meta = json.load(open(sys.argv[1])); out = sys.argv[2]
    cfg = SESSION.get(CONFIG, timeout=30).json()
    res = {}
    for rel in sys.argv[3:]:
        fr, lab, keys = coverage(int(rel), meta, cfg)
        res[rel] = {"title": cfg[rel]["itemTitle"], "patches": fr}
        np.save(out.replace(".json", f"_{rel}.npy"), lab)
        print(rel, cfg[rel]["itemTitle"], [(d, s, f"{f:.0%}") for d, s, f in fr if f > 0.005], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
