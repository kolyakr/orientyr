"""Micro-benchmark of local feature matchers on Apple Silicon (MPS vs CPU).

Usage: python -I bench_matchers.py <xfeat_repo_dir> <map_png> <out_json> [devices]
Pair = a 640x640 map window vs a 640x480 synthetic "drone frame" warped from
the same area (rotation 30 deg, zoom 1.3, light blur) -> realistic tracking pair.
"""
import json
import statistics
import sys
import time

import cv2
import numpy as np
import torch

xfeat_repo, map_path, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
devices = sys.argv[4].split(",") if len(sys.argv) > 4 else ["mps", "cpu"]
sys.path.insert(0, xfeat_repo)

from lightglue import ALIKED, DISK, LightGlue, SuperPoint  # noqa: E402
from modules.lighterglue import LighterGlue  # noqa: E402
from modules.xfeat import XFeat  # noqa: E402

results = []


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()


def timeit(name, dev, fn, n=25, warm=5, extra=None):
    try:
        for _ in range(warm):
            out = fn()
            sync(dev)
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            out = fn()
            sync(dev)
            ts.append((time.perf_counter() - t) * 1000)
        rec = {"name": name, "dev": dev.type, "median_ms": round(statistics.median(ts), 1),
               "p90_ms": round(float(np.percentile(ts, 90)), 1)}
        if extra:
            rec.update(extra(out))
    except Exception as e:  # keep going: MPS op gaps are a finding too
        rec = {"name": name, "dev": dev.type, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    print(json.dumps(rec), flush=True)
    results.append(rec)
    return rec


# ---------------------------------------------------------------- test pair
M = cv2.imread(map_path, cv2.IMREAD_COLOR)
H, W = M.shape[:2]
cx, cy = W // 2, H // 2
win = np.ascontiguousarray(M[cy - 320:cy + 320, cx - 320:cx + 320])
A = cv2.getRotationMatrix2D((cx + 40.0, cy + 20.0), 30, 1.3)
A[0, 2] += 320 - (cx + 40.0)
A[1, 2] += 240 - (cy + 20.0)
frame = cv2.warpAffine(M, A, (640, 480), flags=cv2.INTER_LINEAR)
frame = cv2.GaussianBlur(frame, (3, 3), 0.8)
rgb_f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
rgb_w = cv2.cvtColor(win, cv2.COLOR_BGR2RGB)
gray_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_w = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY)


def run_lg(lg, f0, f1):
    with torch.inference_mode():
        return lg({"image0": f0, "image1": f1})


def to_t(rgb, dev):
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(dev)


# ------------------------------------------------------------- OpenCV baselines
cpu = torch.device("cpu")
sift = cv2.SIFT_create(nfeatures=2048)
timeit("SIFT detect+describe 640x480 (2048)", cpu, lambda: sift.detectAndCompute(gray_f, None))
kf, df = sift.detectAndCompute(gray_f, None)
kw, dw = sift.detectAndCompute(gray_w, None)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))


def sift_match():
    m = flann.knnMatch(df, dw, k=2)
    return [a for a, b in m if a.distance < 0.8 * b.distance]


timeit("SIFT FLANN+ratio match", cpu, sift_match, extra=lambda o: {"matches": len(o)})
good = sift_match()
p0 = np.float32([kf[m.queryIdx].pt for m in good])
p1 = np.float32([kw[m.trainIdx].pt for m in good])
timeit("MAGSAC++ homography (%d matches)" % len(good), cpu,
       lambda: cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 3.0, maxIters=2000, confidence=0.999),
       extra=lambda o: {"inliers": int(o[1].sum()) if o[1] is not None else 0})
orb = cv2.ORB_create(2000)
timeit("ORB detect+describe 640x480 (2000)", cpu, lambda: orb.detectAndCompute(gray_f, None))
lk_prev = gray_f
lk_next = cv2.warpAffine(gray_f, np.float32([[1, 0, 3], [0, 1, 2]]), (640, 480))
pts = cv2.goodFeaturesToTrack(lk_prev, 400, 0.01, 8)
timeit("KLT optical flow 400 pts (VO)", cpu,
       lambda: cv2.calcOpticalFlowPyrLK(lk_prev, lk_next, pts, None, winSize=(21, 21), maxLevel=3))

# ------------------------------------------------------------- deep matchers
for dname in devices:
    dev = torch.device(dname)
    torch.manual_seed(0)

    # XFeat sparse (+ MNN) and XFeat + LighterGlue
    xf = XFeat(top_k=4096)
    xf.dev = dev
    xf.net.to(dev)
    for k in (1024, 2048, 4096):
        timeit(f"XFeat detect 640x480 top{k}", dev, lambda k=k: xf.detectAndCompute(rgb_f, top_k=k)[0],
               extra=lambda o: {"kpts": int(o["keypoints"].shape[0])})
        timeit(f"XFeat detect+MNN match pair top{k} (frame+window)", dev,
               lambda k=k: xf.match_xfeat(rgb_f, rgb_w, top_k=k), extra=lambda o: {"matches": int(len(o[0]))})
    try:
        xf.lighterglue = LighterGlue()
        xf.lighterglue.net.to(dev)
        for k in (1024, 2048):
            d0 = xf.detectAndCompute(rgb_f, top_k=k)[0]
            d1 = xf.detectAndCompute(rgb_w, top_k=k)[0]
            d0.update({"image_size": (640, 480)})
            d1.update({"image_size": (640, 640)})
            timeit(f"LighterGlue match only top{k}", dev, lambda d0=d0, d1=d1: xf.match_lighterglue(d0, d1),
                   extra=lambda o: {"matches": int(len(o[0]))})
    except Exception as e:
        print(json.dumps({"name": "LighterGlue", "dev": dname, "error": str(e)[:200]}))

    # SuperPoint / ALIKED / DISK + LightGlue (cvg implementation)
    for feat, Ext, kw_ in (("superpoint", SuperPoint, {}), ("aliked", ALIKED, {}), ("disk", DISK, {})):
        for k in (1024, 2048):
            try:
                ext = Ext(max_num_keypoints=k, **kw_).eval().to(dev)
                lg = LightGlue(features=feat).eval().to(dev)
            except Exception as e:
                print(json.dumps({"name": f"{feat} init", "dev": dname, "error": str(e)[:200]}))
                continue
            tf, tw = to_t(rgb_f, dev), to_t(rgb_w, dev)
            r = timeit(f"{feat} extract 640x480 max{k}", dev,
                       lambda ext=ext, tf=tf: ext.extract(tf, resize=None),
                       extra=lambda o: {"kpts": int(o["keypoints"].shape[1])})
            if "error" in r:
                continue
            with torch.inference_mode():
                f0 = ext.extract(tf, resize=None)
                f1 = ext.extract(tw, resize=None)
            timeit(f"{feat}+LightGlue match only max{k}", dev,
                   lambda lg=lg, f0=f0, f1=f1: run_lg(lg, f0, f1),
                   extra=lambda o: {"matches": int(o["matches"][0].shape[0]), "stop_layer": int(o["stop"])})

json.dump(results, open(out_json, "w"), indent=1)
