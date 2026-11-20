"""Download a Wayback release as one georeferenced mosaic (PNG + JSON sidecar).

Usage: python fetch_mosaic.py <release> <lat> <lon> <half_tiles> <zoom> <out_png>
The mosaic is (2*half_tiles) x (2*half_tiles) Web-Mercator tiles centred on lat/lon.
The sidecar stores the tile origin so map pixels <-> lat/lon stay convertible.
"""
import io
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from PIL import Image

TILE = ("https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/"
        "WMTS/1.0.0/default028mm/MapServer/tile/{rel}/{z}/{y}/{x}")


def latlon_to_tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    return (int((lon + 180) / 360 * n),
            int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n))


def main() -> None:
    rel, lat, lon = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    half, z, out = int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
    cx, cy = latlon_to_tile(lat, lon, z)
    x0, y0, n = cx - half, cy - half, 2 * half
    sess = requests.Session()

    def get(ij):
        i, j = ij
        r = sess.get(TILE.format(rel=rel, z=z, y=y0 + i, x=x0 + j), timeout=60)
        r.raise_for_status()
        return i, j, np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))

    mosaic = np.zeros((n * 256, n * 256, 3), np.uint8)
    with ThreadPoolExecutor(8) as ex:
        for i, j, t in ex.map(get, [(i, j) for i in range(n) for j in range(n)]):
            mosaic[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256] = t
    Image.fromarray(mosaic).save(out)
    m_per_px = 156543.03392 * math.cos(math.radians(lat)) / 2 ** z
    json.dump({"release": int(rel), "zoom": z, "tile_x0": x0, "tile_y0": y0, "tiles": n,
               "m_per_px": m_per_px, "source": "Esri World Imagery Wayback (Maxar)"},
              open(out.rsplit(".", 1)[0] + ".json", "w"), indent=2)
    print(out, mosaic.shape, f"{m_per_px:.2f} m/px")


if __name__ == "__main__":
    main()
