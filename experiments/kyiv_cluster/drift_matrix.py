"""Drift matrix for the Irpin/Bucha/Hostomel/Moschun cluster: fly over one epoch, navigate on another.

Three Esri Wayback maps of the same 15.6 km square (z16, 1.52 m/px):
  2020 pre-war (release 15423), 2022 war (45134, captures 18-31 Mar 2022), 2025 rebuilt (20512).
Synthetic 640x480 drone frames are cut from the FLIGHT epoch (random rotation/scale/blur) and
registered against the MAP epoch inside a local window around a noisy prior. Diagonal = control.

Frames come from two strata so we can attribute drift to destruction:
  damaged - centred within 150 px of a UNOSAT label (destroyed / severely damaged),
  random  - anywhere in the map interior.

Usage: python drift_matrix.py <maps_dir> <unosat_geojson> <out_dir> [n_per_stratum] [epochs]
  epochs: e.g. "2017=cluster_2017-05-05,2022=cluster_2022-03-31,2025=cluster_2025_rebuilt" (single-date maps,
  already on one pixel grid). Each map's <name>_valid_mask.png is honoured: frames are cut only where
  every epoch is a real capture, so no seam or cloud fill enters the test.
"""
import json
import math
import os
import sys

import cv2
import numpy as np

FRAME_W, FRAME_H = 640, 480
WINDOW = 1400          # local search window (px) ~ 2.1 km: the tracker's prior region
MARGIN = 1000          # keep frames away from the mosaic border
SUCCESS_PX = 15        # ~23 m at 1.52 m/px
MIN_INLIERS = 12
EPOCHS = {"2020": "cluster_2020_prewar", "2022": "cluster_2022_war", "2025": "cluster_2025_rebuilt"}


def lonlat_to_px(lon, lat, meta):
    """WGS84 -> pixel of the Web-Mercator mosaic described by its JSON sidecar."""
    n = 2 ** meta["zoom"] * 256
    x = (lon + 180) / 360 * n - meta["tile_x0"] * 256
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n - meta["tile_y0"] * 256
    return x, y


def make_frame(img, cx, cy, rng):
    """Rotated/scaled/blurred 640x480 view centred on (cx, cy) of img."""
    M = cv2.getRotationMatrix2D((cx, cy), rng.uniform(0, 360), rng.uniform(0.8, 1.25))
    M[:, 2] += [FRAME_W / 2 - cx, FRAME_H / 2 - cy]
    frame = cv2.warpAffine(img, M, (FRAME_W, FRAME_H), flags=cv2.INTER_AREA)
    return cv2.GaussianBlur(frame, (0, 0), rng.uniform(0.3, 1.2))


def register(sift, frame_kd, map_img, prior):
    """SIFT + ratio test + MAGSAC homography in a window around prior -> (centre, inliers)."""
    k1, d1 = frame_kd
    h, w = map_img.shape
    x0 = int(np.clip(prior[0] - WINDOW / 2, 0, w - WINDOW)); y0 = int(np.clip(prior[1] - WINDOW / 2, 0, h - WINDOW))
    k2, d2 = sift.detectAndCompute(map_img[y0:y0 + WINDOW, x0:x0 + WINDOW], None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None, 0
    good = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2) if m.distance < 0.8 * n.distance]
    if len(good) < 8:
        return None, len(good)
    src = np.float32([k1[m.queryIdx].pt for m in good]); dst = np.float32([k2[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 5.0)
    if H is None:
        return None, 0
    c = cv2.perspectiveTransform(np.float32([[[FRAME_W / 2, FRAME_H / 2]]]), H)[0, 0]
    return (float(c[0] + x0), float(c[1] + y0)), int(mask.sum())


def main() -> None:
    maps_dir, labels_p, out = sys.argv[1:4]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    global EPOCHS
    aligned = len(sys.argv) > 5
    if aligned:
        EPOCHS = dict(kv.split("=") for kv in sys.argv[5].split(","))
    ref = next(iter(EPOCHS))
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(f"{maps_dir}/{EPOCHS[ref]}.json"))
    clahe = cv2.createCLAHE(2.0, (8, 8))
    imgs = {e: clahe.apply(cv2.imread(f"{maps_dir}/{f}.png", cv2.IMREAD_GRAYSCALE)) for e, f in EPOCHS.items()}
    H, W = imgs[ref].shape
    # Where every epoch is a real capture, with room for a whole frame (radius 450 px), at 1/8 resolution.
    valid = np.ones((H // 8, W // 8), np.uint8)
    for f in EPOCHS.values():
        mp = f"{maps_dir}/{f}_valid_mask.png"
        if os.path.exists(mp):
            valid &= (cv2.imread(mp, cv2.IMREAD_REDUCED_GRAYSCALE_8) > 0).astype(np.uint8)
    valid = cv2.erode(valid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (113, 113)))
    inside = lambda x, y: MARGIN < x < W - MARGIN and MARGIN < y < H - MARGIN and valid[int(y) // 8, int(x) // 8] > 0
    print(f"frames allowed on {valid.mean():.0%} of the map")

    # Separate orthorectifications differ by a few px: measure each epoch's offset to 2022
    # on a central crop, so ground truth can be carried from flight epoch to map epoch.
    c = slice(H // 2 - 2048, H // 2 + 2048)
    shift = {e: np.zeros(2) for e in EPOCHS}
    for e in ([] if aligned else ("2020", "2025")):
        (sx, sy), resp = cv2.phaseCorrelate(np.float32(imgs[ref][c, c]), np.float32(imgs[e][c, c]))
        shift[e] = np.array([sx, sy]); print(f"offset {e} vs 2022: ({sx:.1f}, {sy:.1f}) px, response {resp:.2f}")

    # strata: UNOSAT destroyed/severe labels vs uniform random
    labs = [lonlat_to_px(*f["geometry"]["coordinates"][:2], meta) for f in json.load(open(labels_p))["features"]
            if f["properties"]["damage"] in (1, 2)]
    labs = np.array([p for p in labs if inside(*p)])
    print(f"{len(labs)} destroyed/severe labels inside the map interior")
    rng = np.random.default_rng(0)
    def draw(gen):
        while True:
            p = gen()
            if inside(*p):
                return p
    centres = [("damaged", *draw(lambda: labs[rng.integers(len(labs))] + rng.normal(0, 150, 2))) for _ in range(n)]
    centres += [("random", *draw(lambda: (rng.uniform(MARGIN, W - MARGIN), rng.uniform(MARGIN, H - MARGIN)))) for _ in range(n)]

    sift = cv2.SIFT_create(4000)
    rows = []
    for i, (stratum, cx, cy) in enumerate(centres):   # (cx, cy) in 2022 pixel coords
        for fe in EPOCHS:
            f_rng = np.random.default_rng(10_000 * i + int(fe))
            fc = np.array([cx, cy]) + shift[fe]
            frame = make_frame(imgs[fe], *fc, f_rng)
            kd = sift.detectAndCompute(frame, None)
            for me in EPOCHS:
                gt = np.array([cx, cy]) + shift[me]
                prior = gt + np.random.default_rng(i).normal(0, 150, 2)
                pred, inl = register(sift, kd, imgs[me], prior)
                err = float(np.hypot(*(np.array(pred) - gt))) if pred else None
                rows.append({"i": i, "stratum": stratum, "flight": fe, "map": me, "inliers": inl,
                             "err_px": err, "ok": err is not None and err < SUCCESS_PX and inl >= MIN_INLIERS})
        print(i, stratum, " ".join(f"{r['flight']}>{r['map']}:{r['inliers']}{'+' if r['ok'] else '-'}"
                                   for r in rows[-len(EPOCHS) ** 2:]), flush=True)

    summary = {}
    for fe in EPOCHS:
        for me in EPOCHS:
            for st in ("damaged", "random", "all"):
                rr = [r for r in rows if r["flight"] == fe and r["map"] == me and st in ("all", r["stratum"])]
                summary[f"{fe}->{me}|{st}"] = {
                    "success": sum(r["ok"] for r in rr) / len(rr),
                    "median_inliers": float(np.median([r["inliers"] for r in rr]))}
    json.dump({"epochs": EPOCHS, "shift_to_ref": {e: s.tolist() for e, s in shift.items()}, "n_per_stratum": n,
               "summary": summary, "rows": rows}, open(f"{out}/drift_matrix.json", "w"), indent=1)
    print("\nflight->map   damaged          random           (success / median inliers)")
    for fe in EPOCHS:
        for me in EPOCHS:
            d, r = summary[f"{fe}->{me}|damaged"], summary[f"{fe}->{me}|random"]
            print(f"{fe}->{me}   {d['success']:5.0%} / {d['median_inliers']:4.0f}    {r['success']:5.0%} / {r['median_inliers']:4.0f}")


if __name__ == "__main__":
    main()
