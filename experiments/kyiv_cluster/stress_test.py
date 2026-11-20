"""Stress test: can learned matchers localise a drone frame on a map from another epoch?

On the seam-free town tiles (Irpin, Hostomel, Moschun, Bucha), frames are cut from the FLIGHT
epoch and matched against the whole tile of the (older) MAP epoch. Compared methods:
  sift     - SIFT + ratio test (rotation invariant baseline)
  sp_lg    - SuperPoint + LightGlue
  disk_lg  - DISK + LightGlue
Heading modes:
  known    - frame de-rotated by compass/IMU heading, residual rotation +-10 deg (realistic: the
             magnetometer is not affected by GNSS jamming)
  unknown  - any rotation; learned matchers try 0/90/180/270 deg and keep the most inliers
Success = frame centre within 15 px (23 m) of truth with >= 12 MAGSAC inliers.

Usage: python stress_test.py <towns_seamfree_dir> <out_json> [n_frames_per_town]
"""
import json
import sys
import time

import cv2
import numpy as np
import torch
from lightglue import DISK, LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd

FRAME_W, FRAME_H = 640, 480
SUCCESS_PX, MIN_INLIERS = 15, 12
SCENARIOS = [("war", "war"), ("prewar", "war"), ("war", "rebuilt"), ("prewar", "rebuilt")]  # (map, flight)
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_frame(img, cx, cy, ang, scale, blur):
    M = cv2.getRotationMatrix2D((cx, cy), ang, scale)
    M[:, 2] += [FRAME_W / 2 - cx, FRAME_H / 2 - cy]
    return cv2.GaussianBlur(cv2.warpAffine(img, M, (FRAME_W, FRAME_H), flags=cv2.INTER_AREA), (0, 0), blur)


def centre_from_matches(p0, p1, frame_shape):
    """MAGSAC homography frame->map; return mapped frame centre and inlier count."""
    if len(p0) < 8:
        return None, len(p0)
    H, mask = cv2.findHomography(np.float32(p0), np.float32(p1), cv2.USAC_MAGSAC, 5.0)
    if H is None:
        return None, 0
    h, w = frame_shape
    c = cv2.perspectiveTransform(np.float32([[[w / 2, h / 2]]]), H)[0, 0]
    return (float(c[0]), float(c[1])), int(mask.sum())


class Sift:
    name = "sift"

    def __init__(self):
        self.sift = cv2.SIFT_create(4000)

    def map_feats(self, img):
        return self.sift.detectAndCompute(img, None)

    def locate(self, frame, mf, try_rotations):
        k1, d1 = self.sift.detectAndCompute(frame, None)
        k2, d2 = mf
        if d1 is None or len(k1) < 8:
            return None, 0
        good = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2) if m.distance < 0.8 * n.distance]
        return centre_from_matches([k1[m.queryIdx].pt for m in good], [k2[m.trainIdx].pt for m in good], frame.shape)


class LG:
    def __init__(self, kind):
        self.name = f"{kind}_lg"
        ext = {"sp": lambda n: SuperPoint(max_num_keypoints=n), "disk": lambda n: DISK(max_num_keypoints=n)}[kind]
        self.ext_frame, self.ext_map = ext(2048).eval().to(DEV), ext(8192).eval().to(DEV)
        self.lg = LightGlue(features={"sp": "superpoint", "disk": "disk"}[kind]).eval().to(DEV)

    @torch.no_grad()
    def _extract(self, ext, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return ext.extract(numpy_image_to_torch(rgb).to(DEV), resize=None)

    def map_feats(self, img):
        return self._extract(self.ext_map, img)

    @torch.no_grad()
    def locate(self, frame, mf, try_rotations):
        best = (None, 0)
        # rotating by k*90 deg about the centre keeps the frame centre fixed, so the mapped centre
        # of the rotated frame is directly the answer
        for k in (range(4) if try_rotations else [0]):
            f = np.ascontiguousarray(np.rot90(frame, k))
            ff = self._extract(self.ext_frame, f)
            out = self.lg({"image0": ff, "image1": mf})
            a, b, m = rbd(ff), rbd(mf), rbd(out)["matches"]
            p0 = a["keypoints"][m[:, 0]].cpu().numpy(); p1 = b["keypoints"][m[:, 1]].cpu().numpy()
            res = centre_from_matches(p0, p1, f.shape)
            if res[1] > best[1]:
                best = res
        return best


def main():
    T, out = sys.argv[1:3]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    clahe = cv2.createCLAHE(2.0, (8, 8))
    methods = [Sift(), LG("sp"), LG("disk")]
    towns = json.load(open(f"{T}/towns.json"))
    rows = []
    for t in towns:
        tn = t["town"].lower()
        imgs = {e: clahe.apply(cv2.imread(f"{T}/{tn}_{e}.png", cv2.IMREAD_GRAYSCALE)) for e in ("prewar", "war", "rebuilt")}
        S = imgs["war"].shape[0]
        mfeats = {(m.name, e): m.map_feats(imgs[e]) for m in methods for e in ("prewar", "war")}
        rng = np.random.default_rng(0)
        mg = 0.3 * min(S, 1400)
        for i in range(n):
            cx, cy = rng.uniform(mg, S - mg, 2)
            heading_true = rng.uniform(0, 360)
            scale, blur = rng.uniform(0.8, 1.25), rng.uniform(0.3, 1.2)
            resid = rng.uniform(-10, 10)
            for map_e, fl_e in SCENARIOS:
                for mode, ang in (("known", resid), ("unknown", heading_true)):
                    frame = make_frame(imgs[fl_e], cx, cy, ang, scale, blur)
                    for m in methods:
                        t0 = time.perf_counter()
                        pred, inl = m.locate(frame, mfeats[(m.name, map_e)], mode == "unknown")
                        dt = time.perf_counter() - t0
                        err = float(np.hypot(pred[0] - cx, pred[1] - cy)) if pred else None
                        rows.append({"town": t["town"], "map": map_e, "flight": fl_e, "heading": mode,
                                     "method": m.name, "inliers": inl, "err_px": err, "sec": dt,
                                     "ok": err is not None and err < SUCCESS_PX and inl >= MIN_INLIERS})
        print(t["town"], "done", flush=True)

    summ = {}
    for map_e, fl_e in SCENARIOS:
        for mode in ("known", "unknown"):
            for m in methods:
                rr = [r for r in rows if r["map"] == map_e and r["flight"] == fl_e and r["heading"] == mode and r["method"] == m.name]
                ok = [r for r in rr if r["ok"]]
                summ[f"map {map_e} / flight {fl_e} | heading {mode} | {m.name}"] = {
                    "success": len(ok) / len(rr), "median_inliers": float(np.median([r["inliers"] for r in rr])),
                    "median_err_m_success": float(np.median([r["err_px"] for r in ok]) * 1.52) if ok else None,
                    "median_sec": float(np.median([r["sec"] for r in rr])),
                    "per_town": {tw: sum(r["ok"] for r in rr if r["town"] == tw) / n for tw in dict.fromkeys(r["town"] for r in rr)}}
    json.dump({"device": str(DEV), "n_per_town": n, "summary": summ, "rows": rows}, open(out, "w"), indent=1)
    for k, v in summ.items():
        print(f"{k:58s} {v['success']:5.0%}  inl {v['median_inliers']:5.0f}  err "
              f"{(v['median_err_m_success'] or float('nan')):4.1f} m  {v['median_sec']:.2f}s  " +
              " ".join(f"{tw[:4]} {s:4.0%}" for tw, s in v["per_town"].items()))


if __name__ == "__main__":
    main()
