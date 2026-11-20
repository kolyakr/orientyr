"""Split a multi-capture Wayback mosaic into seam-free single-capture maps.

Colour harmonisation cannot fix patches with uneven haze (see README), so instead each
epoch is cut into the largest axis-aligned rectangles that lie entirely inside one source
capture, kept away from the seams. Every sub-map keeps its offset in the parent grid, so
coordinates stay in one frame across epochs (map px = sub-map px + offset).

Usage: python split_maps.py <map_png> <labels_npy> <coverage_json> <release> <out_dir> <prefix>
"""
import json
import os
import sys

import cv2
import numpy as np

DOWN = 8          # label raster scale
SEAM_GAP = 4      # label px (= 32 map px) kept clear of every seam
MIN_SIDE = 1536   # map px (~2.3 km): smaller pieces are useless for a flight
MIN_AREA = 3000 ** 2


def largest_rectangle(mask):
    """Largest all-True axis-aligned rectangle (histogram-stack method) -> (x, y, w, h)."""
    h = np.zeros(mask.shape[1] + 1, int)
    best = (0, 0, 0, 0, 0)
    for y, row in enumerate(mask):
        h[:-1] = np.where(row, h[:-1] + 1, 0)
        stack = []
        for x, v in enumerate(h):
            start = x
            while stack and stack[-1][1] >= v:
                sx, sv = stack.pop()
                if sv * (x - sx) > best[0]:
                    best = (sv * (x - sx), sx, y - sv + 1, x - sx, sv)
                start = sx
            stack.append((start, v))
    return best[1:]


def main():
    map_p, lab_p, cov_p, rel, out, prefix = sys.argv[1:7]
    os.makedirs(out, exist_ok=True)
    lab = np.load(lab_p)
    meta = json.load(open(map_p.rsplit(".", 1)[0] + ".json"))
    # label k corresponds to the k-th distinct (date, sensor) in metadata order; recover names by area
    names = {}
    for d, s, f in json.load(open(cov_p))[rel]["patches"]:
        k = min((k for k in np.unique(lab) if k > 0 and k not in names),
                key=lambda k: abs((lab == k).mean() - f), default=None)
        if k is not None:
            names[k] = (d, s)
    img = cv2.imread(map_p)
    pieces = []
    for k, (date, sensor) in names.items():
        free = cv2.erode(((lab == k) * 1).astype(np.uint8), np.ones((2 * SEAM_GAP + 1,) * 2, np.uint8)) > 0
        while True:
            x, y, w, h = largest_rectangle(free)
            X, Y, Wd, Ht = (int(v) * DOWN for v in (x, y, w, h))
            if min(Wd, Ht) < MIN_SIDE or Wd * Ht < MIN_AREA:
                break
            free[y:y + h, x:x + w] = False
            name = f"{prefix}_{date}_{sensor}_{len([p for p in pieces if p['date'] == date])}"
            cv2.imwrite(f"{out}/{name}.png", img[Y:Y + Ht, X:X + Wd])
            sub = dict(meta, offset_px=[X, Y], size_px=[Wd, Ht], capture_date=date, sensor=sensor,
                       parent=os.path.basename(map_p))
            json.dump(sub, open(f"{out}/{name}.json", "w"), indent=1)
            pieces.append({"name": name, "date": date, "sensor": sensor, "x": X, "y": Y, "w": Wd, "h": Ht,
                           "km": [round(Wd * meta["m_per_px"] / 1000, 1), round(Ht * meta["m_per_px"] / 1000, 1)]})
            print(f"  {name}: offset ({X},{Y}) {Wd}x{Ht} px = {pieces[-1]['km'][0]}x{pieces[-1]['km'][1]} km")
    json.dump(pieces, open(f"{out}/{prefix}_pieces.json", "w"), indent=1)


if __name__ == "__main__":
    main()
