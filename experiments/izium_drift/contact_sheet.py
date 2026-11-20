"""Before|after thumbnails for a list of scored buildings (for visual verification).

Usage: python contact_sheet.py <before_png> <after_png> <scores_json> <out_jpg> <mode> [n] [seed]
mode: top | bottom | random | band:<lo>:<hi>
"""
import json
import sys

import cv2
import numpy as np

before_p, after_p, scores_p, out, mode = sys.argv[1:6]
n = int(sys.argv[6]) if len(sys.argv) > 6 else 40
rng = np.random.default_rng(int(sys.argv[7]) if len(sys.argv) > 7 else 0)
b, a = cv2.imread(before_p), cv2.imread(after_p)
d = json.load(open(scores_p))
sx, sy = d["shift_2020_to_2022"]
b = cv2.warpAffine(b, np.float32([[1, 0, -sx], [0, 1, -sy]]), b.shape[1::-1])
rs = sorted([r for r in d["buildings"] if r["score"] is not None], key=lambda r: -r["score"])
if mode == "top":
    sel = rs[:n]
elif mode == "bottom":
    sel = rs[-n:]
elif mode.startswith("band"):
    lo, hi = map(float, mode.split(":")[1:])
    pool = [r for r in rs if lo <= r["score"] < hi]
    sel = [pool[i] for i in rng.choice(len(pool), min(n, len(pool)), replace=False)]
else:
    sel = [rs[i] for i in rng.choice(len(rs), n, replace=False)]
T = 150
tiles = []
for k, r in enumerate(sel):
    p = np.float32(r["poly"]); x0, y0 = p.min(0); x1, y1 = p.max(0)
    s = max(x1 - x0, y1 - y0) / 2 + 14; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0, Y0, X1, Y1 = int(cx - s), int(cy - s), int(cx + s), int(cy + s)
    pair = []
    for im in (b, a):
        c = im[Y0:Y1, X0:X1].copy()
        cv2.polylines(c, [np.int32(p - [X0, Y0])], True, (0, 255, 255), 1)
        pair.append(cv2.resize(c, (T, T), interpolation=cv2.INTER_CUBIC))
    t = np.hstack([pair[0], np.full((T, 3, 3), 255, np.uint8), pair[1]])
    cv2.putText(t, f"{k} {r['score']:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    tiles.append(cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
cols = 5
while len(tiles) % cols:
    tiles.append(np.full_like(tiles[0], 255))
sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(out, sheet.shape)
