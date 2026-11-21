"""Cut the 3x3 km West Irpin square (fully covered by all four years, most UNOSAT destroyed/severe).

Each year is shifted onto 2025 by its global phase-correlation shift (preview/shifts_to_2025.json),
then cropped. Writes <out>/irpin_<year>.png + .json sidecar, unosat.geojson in crop pixels, and a preview.

Usage: python west_irpin_cut.py <map_dir> <unosat_labels.geojson> <out_dir>
"""
import json
import os
import sys

import cv2
import numpy as np

import scan

BOX = (3072, 1200, 1968, 1968)  # x, y, w, h in grid px (z16), from the coverage x labels search
YEARS = {"2017": ("2017-05-05", "WV02"), "2020": ("2020-03-03", "WV03"),
         "2022": ("2022-03-31", "WV03"), "2025": ("2025-04-11", "WV02")}


def main():
    d, labels_p, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    g = json.load(open(f"{d}/grid.json")); shifts = json.load(open(f"{d}/preview/shifts_to_2025.json"))
    x, y, w, h = BOX
    crops = {}
    for yr, (date, sensor) in YEARS.items():
        im = cv2.imread(f"{d}/west_irpin_{yr}.png")
        dx, dy, _ = shifts[yr]
        # crop + sub-pixel shift in one warp: output(u,v) = input(u + x + dx, v + y + dy)
        crop = cv2.warpAffine(im, np.float32([[1, 0, -(x + dx)], [0, 1, -(y + dy)]]), (w, h),
                              flags=cv2.INTER_LINEAR)
        crops[yr] = crop
        cv2.imwrite(f"{out}/irpin_{yr}.png", crop)
        ox, oy = g["tile_x0"] * 256 + x, g["tile_y0"] * 256 + y
        lon0, lat0 = scan.unmerc(ox, oy); lon1, lat1 = scan.unmerc(ox + w, oy + h)
        json.dump({"capture": {"date": date, "sensor": sensor}, "zoom": 16, "merc_z16_origin_px": [ox, oy],
                   "size_px": [w, h], "m_per_px": g["m_per_px"], "bounds_lonlat": [lon0, lat1, lon1, lat0],
                   "shift_to_2025_px": [dx, dy],
                   "source": "Esri World Imagery Wayback (Maxar), one capture only"},
                  open(f"{out}/irpin_{yr}.json", "w"), indent=1)
    ox, oy = g["tile_x0"] * 256 + x, g["tile_y0"] * 256 + y
    feats = []
    for f in json.load(open(labels_p))["features"]:
        px, py = scan.merc(*f["geometry"]["coordinates"][:2]); px, py = px - ox, py - oy
        if 0 <= px < w and 0 <= py < h:
            feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(px, 1), round(py, 1)]},
                          "properties": f["properties"]})
    json.dump({"type": "FeatureCollection", "crs_note": "pixel coords in irpin_<year>.png", "features": feats},
              open(f"{out}/unosat.geojson", "w"))
    dmg = [f["properties"]["damage"] for f in feats]
    print(f"{len(feats)} UNOSAT points: " + ", ".join(f"class {k}: {dmg.count(k)}" for k in sorted(set(dmg))))
    # preview 2x2 at 0.5 scale; dots on 2022
    tiles = []
    for yr, c in crops.items():
        c = c.copy()
        if yr == "2022":
            for f in feats:
                px, py = map(int, f["geometry"]["coordinates"])
                cv2.circle(c, (px, py), 7, (0, 0, 255) if f["properties"]["damage"] <= 2 else (0, 200, 255), 2)
        c = cv2.resize(c, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.putText(c, yr, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 7)
        cv2.putText(c, yr, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        tiles.append(c)
    cv2.imwrite(f"{out}/preview.jpg", np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
