"""Rotation sweep: which matchers survive an arbitrarily rotated phone frame?

Usage: python -I bench_rot.py <xfeat_repo_dir> <map_png> <out_json>
For each in-plane rotation, a phone-like 640x480 frame (mild perspective, blur, gamma,
noise, faint moire) is matched (a) against an 800x800 map window around the true position
at the matching scale (tracking case) and (b) against the whole map (relocalization case).
Reports MAGSAC++ inliers and the error of the frame centre in full-res map pixels.
"""
import json
import sys

import cv2
import numpy as np
import torch

xfeat_repo, map_path, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, xfeat_repo)
from lightglue import LightGlue, SuperPoint, ALIKED  # noqa: E402
from modules.lighterglue import LighterGlue  # noqa: E402
from modules.xfeat import XFeat  # noqa: E402

MPS = torch.device("mps")
M = cv2.imread(map_path, cv2.IMREAD_COLOR)
H, W = M.shape[:2]
S = 0.58                                     # map level whose scale matches the frame
LVL = cv2.resize(M, None, fx=S, fy=S, interpolation=cv2.INTER_AREA)
S_G = 0.5
LVL_G = cv2.resize(M, None, fx=S_G, fy=S_G, interpolation=cv2.INTER_AREA)
rng = np.random.default_rng(0)


def make_frame(c, theta_deg):
    w_m, h_m = 1100.0, 825.0
    th = np.deg2rad(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    loc = np.array([[-w_m / 2 * 1.1, -h_m / 2], [w_m / 2 * 1.1, -h_m / 2],
                    [w_m / 2 * 0.95, h_m / 2], [-w_m / 2 * 0.95, h_m / 2]])
    mc = (loc @ R.T + c).astype(np.float32)
    fc = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
    Hmf = cv2.getPerspectiveTransform(fc, mc)
    fr = cv2.warpPerspective(cv2.GaussianBlur(M, (0, 0), 1.0), np.linalg.inv(Hmf), (640, 480))
    yy, xx = np.mgrid[0:480, 0:640]
    moire = 6 * np.sin(2 * np.pi * (xx * np.cos(0.17) + yy * np.sin(0.17)) / 7.0)
    f = np.power(fr.astype(np.float32) / 255.0, 0.8) * 255 + moire[..., None] + rng.normal(0, 4, fr.shape)
    return np.clip(f, 0, 255).astype(np.uint8), Hmf


def err(Hest, Hmf):
    if Hest is None:
        return None
    p = Hest @ np.array([320, 240, 1.0])
    g = Hmf @ np.array([320, 240, 1.0])
    return round(float(np.linalg.norm(p[:2] / p[2] - g[:2] / g[2])), 1)


def fit(p0, p1_full, thr):
    if len(p0) < 8:
        return None, 0
    Hm, inl = cv2.findHomography(np.float32(p0), np.float32(p1_full), cv2.USAC_MAGSAC, thr,
                                 maxIters=5000, confidence=0.999)
    return Hm, int(inl.sum()) if inl is not None else 0


xf = XFeat(top_k=4096)
xf.dev = MPS
xf.net.to(MPS)
xf.lighterglue = LighterGlue()
xf.lighterglue.net.to(MPS)
sp = SuperPoint(max_num_keypoints=2048).eval().to(MPS)
sp_lg = LightGlue(features="superpoint").eval().to(MPS)
al = ALIKED(max_num_keypoints=2048).eval().to(MPS)
al_lg = LightGlue(features="aliked").eval().to(MPS)
sift = cv2.SIFT_create(nfeatures=4000)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)  # noqa: E731
tt = lambda im: torch.from_numpy(rgb(im)).permute(2, 0, 1).float().div(255).to(MPS)  # noqa: E731

g_feat = xf.detectAndCompute(rgb(LVL_G), top_k=8192)[0]
g_kp_full = g_feat["keypoints"].cpu().numpy() / S_G
g_sift = sift.detectAndCompute(cv2.cvtColor(LVL_G, cv2.COLOR_BGR2GRAY), None)

results = []
centres = [np.array([0.31 * W, 0.6 * H]), np.array([0.68 * W, 0.4 * H]), np.array([0.5 * W, 0.5 * H])]
for theta in (0, 30, 45, 60, 90, 135, 180):
    row = {"rot_deg": theta}
    agg = {}
    for c in centres:
        frame, Hmf = make_frame(c, theta)
        cx, cy = (c * S).astype(int)
        x0, y0 = cx - 400, cy - 400
        win = np.ascontiguousarray(LVL[y0:y0 + 800, x0:x0 + 800])
        to_full = lambda p: (np.asarray(p, np.float64) + [x0, y0]) / S  # noqa: E731
        out = {}
        with torch.inference_mode():
            # XFeat MNN, local window
            i0, i1 = None, None
            m0, m1 = xf.match_xfeat(rgb(frame), rgb(win), top_k=4096)
            out["xfeat_mnn_local"] = fit(m0, to_full(m1), 3.0 / S)
            d0 = dict(xf.detectAndCompute(rgb(frame), top_k=2048)[0], image_size=(640, 480))
            d1 = dict(xf.detectAndCompute(rgb(win), top_k=2048)[0], image_size=(800, 800))
            a, b, _ = xf.match_lighterglue(d0, d1)
            out["xfeat_lighterglue_local"] = fit(a, to_full(b), 3.0 / S)
            for nm, ext, lg in (("sp_lg_local", sp, sp_lg), ("aliked_lg_local", al, al_lg)):
                f0, f1 = ext.extract(tt(frame), resize=None), ext.extract(tt(win), resize=None)
                o = lg({"image0": f0, "image1": f1})
                mm = o["matches"][0].cpu().numpy()
                out[nm] = fit(f0["keypoints"][0].cpu().numpy()[mm[:, 0]],
                              to_full(f1["keypoints"][0].cpu().numpy()[mm[:, 1]]), 3.0 / S)
            # XFeat MNN vs whole map (relocalization)
            ff = xf.detectAndCompute(rgb(frame), top_k=4096)[0]
            j0, j1 = xf.match(ff["descriptors"], g_feat["descriptors"], min_cossim=-1)
            out["xfeat_mnn_global"] = fit(ff["keypoints"][j0].cpu().numpy(), g_kp_full[j1.cpu().numpy()], 3.0 / S_G)
        # SIFT (rotation invariant) local and global
        kf, df = sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
        kw, dw = sift.detectAndCompute(cv2.cvtColor(win, cv2.COLOR_BGR2GRAY), None)
        for nm, (kk, dd, tf) in (("sift_local", (kw, dw, to_full)),
                                 ("sift_global", (g_sift[0], g_sift[1], lambda p: np.asarray(p) / S_G))):
            mt = [m for m, n in flann.knnMatch(df, dd, k=2) if m.distance < 0.8 * n.distance]
            out[nm] = fit([kf[m.queryIdx].pt for m in mt], tf([kk[m.trainIdx].pt for m in mt]),
                          3.0 / (S if nm == "sift_local" else S_G))
        for k, (Hest, ninl) in out.items():
            e = err(Hest, Hmf)
            ok = e is not None and e < 30 and ninl >= 15
            agg.setdefault(k, []).append((ninl, e, ok))
    for k, v in agg.items():
        row[k] = {"ok": f"{sum(x[2] for x in v)}/{len(v)}", "inliers": [x[0] for x in v], "err_px": [x[1] for x in v]}
    print(json.dumps(row), flush=True)
    results.append(row)
json.dump(results, open(out_json, "w"), indent=1)
