"""Timing of two rotation-proof global relocalization strategies (whole map, no retrieval).

Usage: python -I bench_reloc_rot.py <xfeat_repo_dir> <map_png>
A) XFeat on 8 rotated copies of the query (45 deg steps) vs cached whole-map XFeat features.
B) SIFT on the query vs cached whole-map SIFT features (FLANN + ratio).
Query: phone-like frame rotated 110 deg (neither method sees an upright copy).
"""
import statistics
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, sys.argv[1])
from modules.xfeat import XFeat  # noqa: E402

MPS = torch.device("mps")
M = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
H, W = M.shape[:2]
S_G = 0.5
LVL = cv2.resize(M, None, fx=S_G, fy=S_G, interpolation=cv2.INTER_AREA)
c = np.array([0.4 * W, 0.55 * H])
th = np.deg2rad(110)
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
loc = np.array([[-605, -412.5], [605, -412.5], [522.5, 412.5], [-522.5, 412.5]])
mc = (loc @ R.T + c).astype(np.float32)
fc = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
Hmf = cv2.getPerspectiveTransform(fc, mc)
frame = cv2.warpPerspective(cv2.GaussianBlur(M, (0, 0), 1.0), np.linalg.inv(Hmf), (640, 480))
gt = Hmf @ np.array([320, 240, 1.0])
gt = gt[:2] / gt[2]


def err(Hm):
    if Hm is None:
        return None
    p = Hm @ np.array([320, 240, 1.0])
    return round(float(np.linalg.norm(p[:2] / p[2] - gt)), 1)


def bench(fn, n=10, warm=2):
    for _ in range(warm):
        out = fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        torch.mps.synchronize()
        ts.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(ts), 1), out


xf = XFeat(top_k=4096)
xf.dev = MPS
xf.net.to(MPS)
g = xf.detectAndCompute(cv2.cvtColor(LVL, cv2.COLOR_BGR2RGB), top_k=8192)[0]
g_full = g["keypoints"].cpu().numpy() / S_G
rgbf = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def xfeat_8way():
    best = (0, None)
    for k in range(8):
        A = cv2.getRotationMatrix2D((320, 240), 45 * k, 1.0)
        A[:, 2] += [80, 80]                       # 800x640 canvas keeps rotated corners
        rot = cv2.warpAffine(rgbf, A, (800, 640))
        ff = xf.detectAndCompute(rot, top_k=2048)[0]
        i0, i1 = xf.match(ff["descriptors"], g["descriptors"], min_cossim=-1)
        if len(i0) < 8:
            continue
        p_rot = ff["keypoints"][i0].cpu().numpy()
        Ainv = cv2.invertAffineTransform(A)
        p0 = p_rot @ Ainv[:, :2].T + Ainv[:, 2]  # back to original frame coords
        Hm, inl = cv2.findHomography(p0, g_full[i1.cpu().numpy()], cv2.USAC_MAGSAC, 6.0,
                                     maxIters=3000, confidence=0.999)
        n = int(inl.sum()) if inl is not None else 0
        if n > best[0]:
            best = (n, Hm)
    return best


ms, (ninl, Hm) = bench(xfeat_8way)
print(f"A) XFeat 8-way rotation vs whole map (8192 feats): {ms} ms, inliers {ninl}, centre err {err(Hm)} px")

sift = cv2.SIFT_create(nfeatures=16000)
gk, gd = sift.detectAndCompute(cv2.cvtColor(LVL, cv2.COLOR_BGR2GRAY), None)
gk_full = np.float32([k.pt for k in gk]) / S_G
sift_q = cv2.SIFT_create(nfeatures=2048)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
flann.add([gd])
flann.train()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def sift_global():
    kf, df = sift_q.detectAndCompute(gray, None)
    mt = [m for m, n in flann.knnMatch(df, k=2) if m.distance < 0.8 * n.distance]
    p0 = np.float32([kf[m.queryIdx].pt for m in mt])
    p1 = gk_full[[m.trainIdx for m in mt]]
    Hm, inl = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 6.0, maxIters=3000, confidence=0.999)
    return int(inl.sum()) if inl is not None else 0, Hm


ms, (ninl, Hm) = bench(sift_global)
print(f"B) SIFT 2048 vs whole map ({len(gk)} feats, FLANN): {ms} ms, inliers {ninl}, centre err {err(Hm)} px")
