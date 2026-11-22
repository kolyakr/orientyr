"""Kyiv region: one Maxar capture per chosen year -> how big a seamless square do all of them share?

Reuses the Wayback metadata scan from scan.py. For each year we list every capture (date + sensor)
over the region; for every combination of one capture per year we intersect the footprints and
find the largest axis-aligned square inside the intersection (chessboard distance transform).

Usage: python kyiv_years.py <unosat_labels.geojson> <cache_dir> <out_json> [years...]
"""
import itertools
import json
import sys

import cv2
import numpy as np
import requests

import scan

BBOX = (29.95, 50.25, 30.95, 50.70)   # lon0, lat0, lon1, lat1: Kyiv city + Irpin/Bucha/Hostomel/Brovary


def main():
    labels_path, cache, out = sys.argv[1:4]
    years = sys.argv[4:] or ["2017", "2020", "2022", "2025"]
    cfg = requests.get(scan.CONFIG, timeout=60).json()
    caps, (x0, y0, W, H) = scan.footprints(cfg, BBOX, cache)
    by_year = {y: [(k, c) for k, c in caps.items() if k[0].startswith(y) and c["mask"].mean() > 0.02] for y in years}
    m_cell = scan.CELL * scan.m_per_px((BBOX[1] + BBOX[3]) / 2) / 1000  # km per cell
    for y in years:
        print(y, [(k[0], k[1], f"{c['mask'].mean():.0%}") for k, c in sorted(by_year[y])])
    pts = []
    for f in json.load(open(labels_path))["features"]:
        lon, lat = f["geometry"]["coordinates"][:2]
        if BBOX[0] < lon < BBOX[2] and BBOX[1] < lat < BBOX[3]:
            x, y = scan.merc(lon, lat)
            pts.append(((x - x0) / scan.CELL, (y - y0) / scan.CELL))
    pts = np.array(pts)
    res = []
    for combo in itertools.product(*by_year.values()):
        m = np.logical_and.reduce([c["mask"] for _, c in combo]).astype(np.uint8)
        if not m.any():
            continue
        d = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_C, 3)[1:-1, 1:-1]
        r = int(d.max())
        cy, cx = np.unravel_index(d.argmax(), d.shape)
        side = (2 * r - 1) * m_cell
        inside = int(((abs(pts[:, 0] - cx) < r) & (abs(pts[:, 1] - cy) < r)).sum()) if len(pts) else 0
        lon, lat = scan.unmerc(x0 + (cx + .5) * scan.CELL, y0 + (cy + .5) * scan.CELL)
        res.append(dict(side_km=round(side, 2), center=[round(lon, 4), round(lat, 4)], unosat_inside=inside,
                        common_area_km2=round(m.sum() * m_cell ** 2, 1),
                        captures=[dict(date=k[0], sensor=k[1], res=c["res"]) for k, c in combo]))
    res.sort(key=lambda r: -r["side_km"])
    json.dump(res, open(out, "w"), indent=1)
    print(f"{len(res)} combinations with any overlap")
    for r in res[:15]:
        print(f"{r['side_km']:5.1f} km  area {r['common_area_km2']:6.1f} km2  unosat {r['unosat_inside']:4d}  "
              f"@{r['center']}  " + " | ".join(f"{c['date']} {c['sensor']} {c['res']}" for c in r["captures"]))


if __name__ == "__main__":
    main()
