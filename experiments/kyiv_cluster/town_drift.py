"""Cross-epoch registration on the seam-free town tiles (same protocol as drift_matrix.py).

Frames are cut from one epoch's tile and registered against every epoch's tile of the same town
(whole tile as search window, since a tile is about one tracker window wide).

Usage: python town_drift.py <towns_seamfree_dir> <out_json> [n_frames_per_town]
"""
import json
import sys

import cv2
import numpy as np

import drift_matrix as dm

EPOCHS = ["prewar", "war", "rebuilt"]


def main():
    T, out = sys.argv[1:3]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    clahe = cv2.createCLAHE(2.0, (8, 8))
    sift = cv2.SIFT_create(4000)
    rows = []
    for t in json.load(open(f"{T}/towns.json")):
        tn = t["town"].lower()
        imgs = {e: clahe.apply(cv2.imread(f"{T}/{tn}_{e}.png", cv2.IMREAD_GRAYSCALE)) for e in EPOCHS}
        S = imgs["war"].shape[0]
        dm.WINDOW = S  # whole tile is the search region
        rng = np.random.default_rng(0)
        m = 0.3 * min(S, 1400)  # keep the frame centre away from the tile border
        for i in range(n):
            cx, cy = rng.uniform(m, S - m, 2)
            for fe in EPOCHS:
                frame = dm.make_frame(imgs[fe], cx, cy, np.random.default_rng(100 * i + EPOCHS.index(fe)))
                kd = sift.detectAndCompute(frame, None)
                for me in EPOCHS:
                    pred, inl = dm.register(sift, kd, imgs[me], (S / 2, S / 2))
                    err = float(np.hypot(pred[0] - cx, pred[1] - cy)) if pred else None
                    rows.append({"town": t["town"], "flight": fe, "map": me, "inliers": inl, "err_px": err,
                                 "ok": err is not None and err < dm.SUCCESS_PX and inl >= dm.MIN_INLIERS})
        print(t["town"], "done", flush=True)
    summ = {}
    for fe in EPOCHS:
        for me in EPOCHS:
            rr = [r for r in rows if r["flight"] == fe and r["map"] == me]
            summ[f"{fe}->{me}"] = {"success": sum(r["ok"] for r in rr) / len(rr),
                                   "median_inliers": float(np.median([r["inliers"] for r in rr])),
                                   "per_town": {tw: sum(r["ok"] for r in rr if r["town"] == tw) / n
                                                for tw in dict.fromkeys(r["town"] for r in rr)}}
    json.dump({"summary": summ, "rows": rows}, open(out, "w"), indent=1)
    for k, v in summ.items():
        print(f"{k:16s} {v['success']:5.0%} / {v['median_inliers']:4.0f}   " +
              " ".join(f"{tw[:4]} {s:4.0%}" for tw, s in v["per_town"].items()))


if __name__ == "__main__":
    main()
