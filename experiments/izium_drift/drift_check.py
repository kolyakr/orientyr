"""Measure how much the pre-war -> wartime map change hurts frame-to-map registration.

Synthetic drone frames are cut from the AFTER (2022) mosaic with random rotation,
scale and blur, i.e. what a UAV over Izium would actually see. Each frame is
registered (SIFT + ratio test + RANSAC homography) against a search window of
  - the AFTER map  (control: same capture, only synthetic warp), and
  - the BEFORE map (drift: 2020 capture, other season / sun / view angle, and destroyed buildings).
The gap between the two is the "drift" our navigator must survive.

Usage: python drift_check.py <before_png> <after_png> <out_dir> [n_frames]
"""
import json
import os
import sys

import cv2
import numpy as np

FRAME_W, FRAME_H = 640, 480
WINDOW = 1400          # local search window (px) = tracking prior, ~1.1 km
SUCCESS_PX = 15        # ~12 m at 0.78 m/px


def make_frame(img, rng):
    """Cut a rotated/scaled/blurred frame; return frame and its frame->map homography."""
    h, w = img.shape[:2]
    cx, cy = rng.uniform(1500, w - 1500), rng.uniform(1500, h - 1500)
    ang, s = rng.uniform(0, 360), rng.uniform(0.8, 1.25)
    # map->frame: rotate/scale about (cx,cy) and move it to the frame centre
    M = cv2.getRotationMatrix2D((cx, cy), ang, s)
    M[:, 2] += [FRAME_W / 2 - cx, FRAME_H / 2 - cy]
    frame = cv2.warpAffine(img, M, (FRAME_W, FRAME_H), flags=cv2.INTER_AREA)
    frame = cv2.GaussianBlur(frame, (0, 0), rng.uniform(0.3, 1.2))
    H_m2f = np.vstack([M, [0, 0, 1]])
    return frame, np.linalg.inv(H_m2f), (cx, cy)


def register(sift, frame, map_img, centre, rng):
    """Register frame inside a window around a noisy prior; return predicted centre + inliers."""
    h, w = map_img.shape[:2]
    px, py = centre[0] + rng.normal(0, 150), centre[1] + rng.normal(0, 150)  # imperfect prior
    x0 = int(np.clip(px - WINDOW / 2, 0, w - WINDOW)); y0 = int(np.clip(py - WINDOW / 2, 0, h - WINDOW))
    win = map_img[y0:y0 + WINDOW, x0:x0 + WINDOW]
    k1, d1 = sift.detectAndCompute(frame, None)
    k2, d2 = sift.detectAndCompute(win, None)
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
    return (c[0] + x0, c[1] + y0), int(mask.sum())


def main() -> None:
    before_p, after_p, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n = int(sys.argv[4]) if len(sys.argv) > 4 else 60
    os.makedirs(out, exist_ok=True)
    gray = lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
    before, after = gray(before_p), gray(after_p)
    # The two captures are orthorectified separately; measure their residual offset
    # so ground truth on the BEFORE map is correct (after = before + shift).
    (sx, sy), _ = cv2.phaseCorrelate(np.float32(before[2048:6144, 2048:6144]),
                                     np.float32(after[2048:6144, 2048:6144]))
    clahe = cv2.createCLAHE(2.0, (8, 8))  # equalise season/illumination contrast a bit
    before, after = clahe.apply(before), clahe.apply(after)
    sift = cv2.SIFT_create(4000)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        frame, _, gt = make_frame(after, rng)
        rec = {"i": i, "gt": [float(gt[0]), float(gt[1])]}
        for name, m, off in [("after", after, (0, 0)), ("before", before, (-sx, -sy))]:
            gt_m = (gt[0] + off[0], gt[1] + off[1])
            pred, inl = register(sift, frame, m, gt_m, np.random.default_rng(1000 + i))
            err = float(np.hypot(pred[0] - gt_m[0], pred[1] - gt_m[1])) if pred is not None else None
            rec[name] = {"inliers": inl, "err_px": err,
                         "ok": err is not None and err < SUCCESS_PX and inl >= 12}
        rows.append(rec)
        print(i, rec["after"], rec["before"], flush=True)

    def summ(k):
        ok = [r[k] for r in rows if r[k]["ok"]]
        return {"success_rate": len(ok) / n,
                "median_inliers": float(np.median([r[k]["inliers"] for r in rows])),
                "median_err_px_success": float(np.median([o["err_px"] for o in ok])) if ok else None}
    summary = {"n": n, "capture_shift_px": [sx, sy], "after_map(control)": summ("after"),
               "before_map(drift)": summ("before")}
    json.dump({"summary": summary, "frames": rows}, open(f"{out}/drift_check.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
