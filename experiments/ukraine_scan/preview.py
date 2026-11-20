"""Preview the top candidates from scan.py: three single-capture maps side by side with UNOSAT points.

Each epoch is built by kyiv_cluster/single_date_map.py (one capture, pixels taken only inside its
footprint) on a low-zoom grid covering the candidate square, so a preview costs a few hundred tiles.
Black = outside the capture's footprint; a correct candidate shows none inside the yellow square.

Usage: python preview.py <candidates.json> <unosat_labels.geojson> <work_dir> <out_dir> <rank> [<rank> ...] [--zoom 15]
"""
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SDM = os.path.join(HERE, "..", "kyiv_cluster", "single_date_map.py")


def main():
    args = sys.argv[1:]
    zoom = 15
    if "--zoom" in args:
        i = args.index("--zoom"); zoom = int(args[i + 1]); del args[i:i + 2]
    cand_p, labels_p, work, out = args[:4]
    work = os.path.abspath(work)  # single_date_map.py runs from its own folder (it imports coverage.py)
    ranks = [int(r) for r in args[4:]]
    os.makedirs(work, exist_ok=True); os.makedirs(out, exist_ok=True)
    cands = json.load(open(cand_p))["candidates"]
    labels = json.load(open(labels_p))["features"]
    f = 2 ** (16 - zoom)
    for rank in ranks:
        c = cands[rank]
        x, y, side = c["square_merc_z16"]
        # grid: the square plus a 10 % border, in whole tiles at the preview zoom
        pad = side * 0.1
        tx0, ty0 = int((x - pad) / f // 256), int((y - pad) / f // 256)
        nt = int(math.ceil((side + 2 * pad) / f / 256)) + 1
        lat = c["center"][0]
        grid = {"zoom": zoom, "tile_x0": tx0, "tile_y0": ty0, "tiles": nt,
                "m_per_px": 156543.03392 * math.cos(math.radians(lat)) / 2 ** zoom}
        tag = f"r{rank}_{c['aoi']}_{c['size_km']:g}km"
        gp = f"{work}/{tag}_grid.json"
        json.dump(grid, open(gp, "w"))
        panels = []
        for ep in ("pre", "war", "post"):
            cap = c[ep]; pref = f"{work}/{tag}_{ep}_{cap['date']}"
            if not os.path.exists(pref + ".png"):
                subprocess.run([sys.executable, SDM, gp, cap["date"], cap["sensor"], pref], check=True, cwd=os.path.dirname(SDM))
            im = cv2.imread(pref + ".png")
            # square outline and UNOSAT points (grid px = z16 px / f - tile origin)
            ox, oy = tx0 * 256, ty0 * 256
            p0 = (int(x / f - ox), int(y / f - oy)); p1 = (int((x + side) / f - ox), int((y + side) / f - oy))
            cv2.rectangle(im, p0, p1, (0, 255, 255), 3)
            if ep == "war":
                n = 2 ** zoom * 256
                for lf in labels:
                    if lf["properties"]["damage"] not in (1, 2):
                        continue
                    lon, la = lf["geometry"]["coordinates"][:2]
                    px = (lon + 180) / 360 * n - ox; py = (1 - math.asinh(math.tan(math.radians(la))) / math.pi) / 2 * n - oy
                    if 0 <= px < im.shape[1] and 0 <= py < im.shape[0]:
                        cv2.circle(im, (int(px), int(py)), 3, (0, 0, 255), -1)
            cv2.putText(im, f"{ep} {cap['date']} {cap['sensor']} {cap['res']}m", (12, 40), 0, 1.2, (255, 255, 255), 3)
            panels.append(im)
        h = min(p.shape[0] for p in panels)
        strip = np.hstack([p[:h] for p in panels])
        sc = min(1.0, 3600 / strip.shape[1])
        strip = cv2.resize(strip, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{out}/{tag}.jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(tag, c["cities"], c["destroyed_severe"], "labels", flush=True)


if __name__ == "__main__":
    main()
