"""Relocalization budget on Apple Silicon: XFeat whole-map matching, EfficientLoFTR, DINOv2.

Usage: python -I bench_reloc.py <xfeat_repo_dir> <map_png> <out_json>
Query = synthetic 640x480 phone-like frame (perspective trapezoid, 70 deg rotation,
blur, gamma, noise, faint moire) cut from the map with a known homography.
"""
import json
import statistics
import sys
import time

import cv2
import numpy as np
import torch

xfeat_repo, map_path, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, xfeat_repo)
from modules.lighterglue import LighterGlue  # noqa: E402
from modules.xfeat import XFeat  # noqa: E402

results = []
MPS, CPU = torch.device("mps"), torch.device("cpu")


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()


def timeit(name, dev, fn, n=10, warm=3, extra=None):
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
    except Exception as e:
        rec = {"name": name, "dev": dev.type, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    print(json.dumps(rec), flush=True)
    results.append(rec)
    return rec


# ------------------------------------------------------------------ query frame with GT
M = cv2.imread(map_path, cv2.IMREAD_COLOR)
H, W = M.shape[:2]
c = np.array([0.31 * W, 0.62 * H])
w_m, h_m = 1100.0, 825.0                      # frame footprint on the map (full-res px)
th = np.deg2rad(70)
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
# trapezoid: far (top) edge wider than near edge -> camera pitched ~15-20 deg
loc = np.array([[-w_m / 2 * 1.18, -h_m / 2], [w_m / 2 * 1.18, -h_m / 2],
                [w_m / 2 * 0.9, h_m / 2], [-w_m / 2 * 0.9, h_m / 2]])
map_corners = (loc @ R.T + c).astype(np.float32)
frame_corners = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])
H_map_from_frame = cv2.getPerspectiveTransform(frame_corners, map_corners)
blurred = cv2.GaussianBlur(M, (0, 0), 1.0)
frame = cv2.warpPerspective(blurred, np.linalg.inv(H_map_from_frame), (640, 480), flags=cv2.INTER_LINEAR)
yy, xx = np.mgrid[0:480, 0:640]
moire = 6 * np.sin(2 * np.pi * (xx * np.cos(0.17) + yy * np.sin(0.17)) / 7.0)
f = frame.astype(np.float32) / 255.0
f = np.power(f, 0.8) * 255 + moire[..., None] + np.random.default_rng(0).normal(0, 4, f.shape)
frame = np.clip(f, 0, 255).astype(np.uint8)
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
gt_center = (H_map_from_frame @ np.array([320, 240, 1.0]))
gt_center = gt_center[:2] / gt_center[2]


def center_err(Hm):
    if Hm is None:
        return None
    p = Hm @ np.array([320, 240, 1.0])
    return round(float(np.linalg.norm(p[:2] / p[2] - gt_center)), 1)


# ------------------------------------------------------------------ XFeat whole-map
xf = XFeat(top_k=4096)
xf.dev = MPS
xf.net.to(MPS)
xf.lighterglue = LighterGlue()
xf.lighterglue.net.to(MPS)

for s, map_k in ((0.5, 16384), (0.5, 8192), (0.35, 8192)):
    lvl = cv2.resize(M, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    rgb_lvl = cv2.cvtColor(lvl, cv2.COLOR_BGR2RGB)
    timeit(f"[MapPack] XFeat whole map L(s={s}) {lvl.shape[1]}x{lvl.shape[0]} top{map_k}", MPS,
           lambda: xf.detectAndCompute(rgb_lvl, top_k=map_k)[0], n=3, warm=1)
    mfeat = xf.detectAndCompute(rgb_lvl, top_k=map_k)[0]
    mk_full = mfeat["keypoints"].cpu().numpy() / s
    for fk in (2048, 4096):
        def reloc_mnn(fk=fk, mfeat=mfeat, mk_full=mk_full):
            ff = xf.detectAndCompute(rgb_frame, top_k=fk)[0]
            i0, i1 = xf.match(ff["descriptors"], mfeat["descriptors"], min_cossim=-1)
            p0 = ff["keypoints"][i0].cpu().numpy()
            p1 = mk_full[i1.cpu().numpy()]
            Hm, inl = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 3.0 / s, maxIters=5000, confidence=0.999)
            return Hm, inl, len(p0)
        timeit(f"RELOC XFeat MNN frame top{fk} vs whole map L(s={s}) top{map_k} (+MAGSAC)", MPS, reloc_mnn,
               extra=lambda o: {"tentative": o[2], "inliers": int(o[1].sum()) if o[1] is not None else 0,
                                "center_err_px_fullres": center_err(o[0])})

    if map_k == 8192:
        def reloc_lg(mfeat=mfeat, mk_full=mk_full, shape=lvl.shape):
            ff = xf.detectAndCompute(rgb_frame, top_k=2048)[0]
            d0 = dict(ff, image_size=(640, 480))
            d1 = dict(mfeat, image_size=(shape[1], shape[0]))
            m0, m1, idx = xf.match_lighterglue(d0, d1)
            p1 = mk_full[idx[:, 1]] if len(idx) else np.zeros((0, 2), np.float32)
            p0 = m0
            if len(p0) < 4:
                return None, None, len(p0)
            Hm, inl = cv2.findHomography(p0, p1, cv2.USAC_MAGSAC, 3.0 / s, maxIters=5000, confidence=0.999)
            return Hm, inl, len(p0)
        timeit(f"RELOC XFeat+LighterGlue frame top2048 vs whole map L(s={s}) top{map_k}", MPS, reloc_lg, n=5,
               extra=lambda o: {"tentative": o[2], "inliers": int(o[1].sum()) if o[1] is not None else 0,
                                "center_err_px_fullres": center_err(o[0])})

# ------------------------------------------------------------------ EfficientLoFTR (HF transformers)
try:
    from transformers import AutoImageProcessor, EfficientLoFTRForKeypointMatching
    proc = AutoImageProcessor.from_pretrained("zju-community/efficientloftr")
    s = 0.58  # map level that matches the frame scale; window centred on GT
    lvl = cv2.resize(M, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cx, cy = (gt_center * s).astype(int)
    win = cv2.cvtColor(lvl[cy - 240:cy + 240, cx - 320:cx + 320], cv2.COLOR_BGR2RGB)
    inputs = proc([rgb_frame, win], return_tensors="pt")
    for dev in (MPS, CPU):
        model = EfficientLoFTRForKeypointMatching.from_pretrained("zju-community/efficientloftr").eval().to(dev)
        pv = inputs["pixel_values"].to(dev)

        def eloftr(model=model, pv=pv):
            with torch.inference_mode():
                return model(pixel_values=pv)
        timeit("EfficientLoFTR pair 640x480 (HF, fp32)", dev, eloftr, n=5, warm=2,
               extra=lambda o: {"matches_conf>0.2": int((o.matching_scores[0, 0] > 0.2).sum())})
except Exception as e:
    print(json.dumps({"name": "EfficientLoFTR", "error": f"{type(e).__name__}: {str(e)[:200]}"}))

# ------------------------------------------------------------------ DINOv2-S/14 (retrieval embedding)
try:
    from transformers import AutoModel
    dino = AutoModel.from_pretrained("facebook/dinov2-small").eval().to(MPS)
    for side, b in ((224, 1), (448, 1), (224, 32)):
        x = torch.randn(b, 3, side, side, device=MPS)

        def fwd(x=x):
            with torch.inference_mode():
                return dino(pixel_values=x).last_hidden_state
        timeit(f"DINOv2-S/14 forward {side}px batch{b}", MPS, fwd, n=10, warm=3)
except Exception as e:
    print(json.dumps({"name": "DINOv2", "error": f"{type(e).__name__}: {str(e)[:200]}"}))

json.dump(results, open(out_json, "w"), indent=1)
