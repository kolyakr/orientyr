"""Build the West Irpin maps: 2017, 2020, 2022, 2025, one Maxar capture each, on one z16 grid.

1. Intersect the four captures' footprints (Wayback metadata, as in kyiv_years.py) and keep the
   connected piece around West Irpin.
2. Build each year with kyiv_cluster/single_date_map.py on a z16 grid covering that piece.
3. AND the four valid masks: the common area, where every year is a real single-date photo.

Usage: python west_irpin.py <cache_dir> <out_dir>
"""
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
import requests

import scan
from kyiv_years import BBOX

CAPS = [("2017-05-05", "WV02"), ("2020-03-03", "WV03"), ("2022-03-31", "WV03"), ("2025-04-11", "WV02")]
SEED = (30.2216, 50.501)  # centre of the largest common square
SDM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kyiv_cluster", "single_date_map.py")


def main():
    cache, out = sys.argv[1:3]
    out = os.path.abspath(out); os.makedirs(out, exist_ok=True)
    cfg = requests.get(scan.CONFIG, timeout=60).json()
    caps, (x0, y0, W, H) = scan.footprints(cfg, BBOX, cache)
    m = np.logical_and.reduce([caps[k]["mask"] for k in CAPS]).astype(np.uint8)
    n, lab = cv2.connectedComponents(m)
    sx, sy = scan.merc(*SEED)
    piece = lab == lab[int((sy - y0) // scan.CELL), int((sx - x0) // scan.CELL)]
    ys, xs = np.nonzero(piece)
    # bbox in z16 px -> square tile grid (single_date_map.py takes a square grid)
    bx0, bx1 = x0 + xs.min() * scan.CELL, x0 + (xs.max() + 1) * scan.CELL
    by0, by1 = y0 + ys.min() * scan.CELL, y0 + (ys.max() + 1) * scan.CELL
    tx0, ty0 = int(bx0 // 256), int(by0 // 256)
    nt = int(math.ceil(max(bx1 - tx0 * 256, by1 - ty0 * 256) / 256))
    lat = scan.unmerc(0, (by0 + by1) / 2)[1]
    grid = {"zoom": 16, "tile_x0": tx0, "tile_y0": ty0, "tiles": nt, "m_per_px": scan.m_per_px(lat)}
    print(f"common piece {piece.sum() * (scan.CELL * grid['m_per_px'] / 1000) ** 2:.1f} km2, "
          f"grid {nt}x{nt} tiles = {nt * 256 * grid['m_per_px'] / 1000:.1f} km side")
    gp = f"{out}/grid.json"; json.dump(grid, open(gp, "w"))
    valid = None
    for date, sensor in CAPS:
        pref = f"{out}/west_irpin_{date[:4]}"
        if not os.path.exists(pref + ".png"):
            subprocess.run([sys.executable, SDM, gp, date, sensor, pref], check=True, cwd=os.path.dirname(SDM))
        v = cv2.imread(pref + "_valid_mask.png", 0) > 0
        valid = v if valid is None else valid & v
    cv2.imwrite(f"{out}/common_valid_mask.png", valid.astype(np.uint8) * 255)
    print(f"common valid area {valid.sum() * grid['m_per_px'] ** 2 / 1e6:.1f} km2 "
          f"({valid.mean():.0%} of the grid)")


if __name__ == "__main__":
    main()
