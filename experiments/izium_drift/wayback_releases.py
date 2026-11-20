"""List the Esri World Imagery Wayback releases that actually changed imagery at a point.

Wayback publishes ~200 releases, but most leave any given tile untouched. The
`tilemap` endpoint answers "which release does this tile really come from?",
so walking all releases and collecting the distinct answers gives the true
imagery history at that spot. The capture date comes from the release's
metadata layer (SRC_DATE).

Usage: python wayback_releases.py <lat> <lon> <out_json> [zoom]
"""
import json
import math
import sys

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Esri's metadata service times out under load; retry with backoff instead of dying mid-walk.
SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(
    total=6, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=None)))

CFG_URL = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
TILEMAP = ("https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/"
           "WMTS/1.0.0/default028mm/MapServer/tilemap/{rel}/{z}/{y}/{x}")


def latlon_to_tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    """Web-Mercator (slippy map) tile indices for a WGS84 point."""
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def capture_info(meta_url: str, lat: float, lon: float, z: int) -> dict:
    """Query the release's metadata layer for source, resolution and capture date."""
    # Metadata service has one layer per zoom band; layer id = 23 - zoom (Esri convention).
    layer = max(0, 23 - z)
    r = SESSION.get(f"{meta_url}/{layer}/query", params={
        "geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint", "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects", "outFields": "SRC_DATE2,SRC_DESC,SRC_RES,NICE_DESC",
        "returnGeometry": "false", "f": "json"}, timeout=60)
    feats = r.json().get("features", [])
    return feats[0]["attributes"] if feats else {}


def main() -> None:
    lat, lon, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    z = int(sys.argv[4]) if len(sys.argv) > 4 else 17
    x, y = latlon_to_tile(lat, lon, z)
    cfg = SESSION.get(CFG_URL, timeout=30).json()
    rels = sorted(cfg.items(), key=lambda kv: kv[1]["itemTitle"])  # chronological by title date
    seen: dict[int, dict] = {}
    for rel, info in rels:
        tm = SESSION.get(TILEMAP.format(rel=rel, z=z, y=y, x=x), timeout=30).json()
        if not tm.get("data", [0])[0]:
            continue
        src = int(tm.get("select", [rel])[0])
        if src not in seen:
            src_info = cfg[str(src)]
            seen[src] = {"release": src, "title": src_info["itemTitle"],
                         **capture_info(src_info["metadataLayerUrl"], lat, lon, z)}
            print(seen[src], flush=True)
    json.dump({"lat": lat, "lon": lon, "zoom": z, "tile": [x, y], "releases": list(seen.values())},
              open(out, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
