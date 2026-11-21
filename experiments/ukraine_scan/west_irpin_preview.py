"""Previews of the West Irpin maps: a 2x2 overview and close-ups at UNOSAT-damaged buildings.

Each year is shifted onto 2025 by one global sub-pixel shift (phase correlation inside the common
area), so close-ups compare the same ground. Shifts are printed and saved.

Usage: python west_irpin_preview.py <map_dir> <unosat_labels.geojson> <out_dir>
"""
import json
import os
import sys

import cv2
import numpy as np

import scan

YEARS = ["2017", "2020", "2022", "2025"]


def label(im, text, scale=1.0):
    cv2.putText(im, text, (12, int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (0, 0, 0), int(6 * scale))
    cv2.putText(im, text, (12, int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, (255, 255, 255), int(2 * scale))
    return im


def main():
    d, labels_p, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    grid = json.load(open(f"{d}/grid.json"))
    valid = cv2.imread(f"{d}/common_valid_mask.png", 0) > 0
    maps = {y: cv2.imread(f"{d}/west_irpin_{y}.png") for y in YEARS}
    # global alignment onto 2025, measured on a downscaled grey image inside the common area
    ref = cv2.cvtColor(maps["2025"], cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F) * valid
    shifts = {}
    for y in YEARS:
        g = cv2.cvtColor(maps[y], cv2.COLOR_BGR2GRAY).astype(np.float32)
        (dx, dy), resp = cv2.phaseCorrelate(ref * valid, g * valid, win)
        shifts[y] = (dx, dy, resp)
        maps[y] = cv2.warpAffine(maps[y], np.float32([[1, 0, -dx], [0, 1, -dy]]), ref.shape[::-1])
        print(f"{y}: shift to 2025 = ({dx:+.2f}, {dy:+.2f}) px, response {resp:.2f}")
    json.dump(shifts, open(f"{out}/shifts_to_2025.json", "w"), indent=1)

    # UNOSAT points in map px
    ox, oy = grid["tile_x0"] * 256, grid["tile_y0"] * 256
    pts = []
    for f in json.load(open(labels_p))["features"]:
        x, y = scan.merc(*f["geometry"]["coordinates"][:2])
        x, y = int(x - ox), int(y - oy)
        if 0 <= x < valid.shape[1] and 0 <= y < valid.shape[0] and valid[y, x]:
            pts.append((x, y, f["properties"]["damage"]))
    print(f"{len(pts)} UNOSAT points in the common area "
          f"({sum(p[2] in (1, 2) for p in pts)} destroyed/severe)")

    # overview: each year masked to the common area, downscaled, 2x2
    contours, _ = cv2.findContours(valid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    s = 1400 / valid.shape[1]
    tiles = []
    for y in YEARS:
        im = maps[y].copy(); im[~valid] //= 4
        cv2.drawContours(im, contours, -1, (0, 255, 255), 6)
        if y == "2022":
            for x, yy, dmg in pts:
                cv2.circle(im, (x, yy), 10, (0, 0, 255) if dmg in (1, 2) else (0, 165, 255), -1)
        tiles.append(label(cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), y))
    cv2.imwrite(f"{out}/overview.jpg", np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])

    # close-ups: 8 destroyed/severe sites, spread out (greedy, >= 600 px apart), 300 px crops -> 2x
    rng = np.random.default_rng(0)
    cand = [p for p in pts if p[2] in (1, 2)]
    rng.shuffle(cand)
    picked = []
    for x, y, _ in cand:
        if all((x - a) ** 2 + (y - b) ** 2 > 600 ** 2 for a, b in picked) and \
                150 <= x < valid.shape[1] - 150 and 150 <= y < valid.shape[0] - 150 and valid[y - 150:y + 150, x - 150:x + 150].all():
            picked.append((x, y))
        if len(picked) == 8:
            break
    rows = []
    for i, (x, y) in enumerate(picked):
        row = []
        for yr in YEARS:
            c = cv2.resize(maps[yr][y - 150:y + 150, x - 150:x + 150], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            cv2.circle(c, (300, 300), 40, (0, 255, 255), 2)
            row.append(label(c, f"{yr}" if i else f"{yr}", 0.8))
        rows.append(np.hstack(row))
    for k in range(0, len(rows), 4):
        cv2.imwrite(f"{out}/closeups_{k // 4 + 1}.jpg", np.vstack(rows[k:k + 4]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(picked)} close-up sites")


if __name__ == "__main__":
    main()
