"""Stress test of the classic baseline (RootSIFT + FLANN + MAGSAC++, global search, no prior):
degrade real crops in controlled ways and measure success and wrong-fix rates against the clean registration."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

root = Path(sys.argv[1])
ref = json.loads(Path(sys.argv[2]).read_text())
rng = np.random.default_rng(0)

mp = cv2.imread(str(root / "global_map.png"), cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create(nfeatures=0)


def rootsift(d):
    d = d / (np.abs(d).sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(d).astype(np.float32)


kpm, dm = sift.detectAndCompute(mp, None)
ptsm = np.float32([k.pt for k in kpm])
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=8), dict(checks=128))
flann.add([rootsift(dm)])
flann.train()

W = 640
SQ = np.float32([(0, 0), (W, 0), (W, W), (0, W)])


def localize(img, mask):
    best = None
    for f in (1.0, 0.5, 0.25):
        im = img if f == 1.0 else cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        mk = None if mask is None else (mask if f == 1.0 else cv2.resize(mask, im.shape[::-1], interpolation=cv2.INTER_NEAREST))
        kp, d = sift.detectAndCompute(im, mk)
        if d is None or len(kp) < 10:
            continue
        knn = flann.knnMatch(rootsift(d), k=2)
        good = [p[0] for p in knn if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if len(good) < 8:
            continue
        src = np.float32([kp[m.queryIdx].pt for m in good]) / f
        dst = ptsm[[m.trainIdx for m in good]]
        H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 6.0, maxIters=20000, confidence=0.9999)
        if H is not None and (best is None or inl.sum() > best[1]):
            best = (H, int(inl.sum()))
    return best


def tilt_h(theta_deg, phi):
    f = W / (2 * np.tan(np.radians(30)))
    K = np.array([[f, 0, W / 2], [0, f, W / 2], [0, 0, 1]])
    ax = np.array([np.cos(phi), np.sin(phi), 0.0])
    R, _ = cv2.Rodrigues(ax * np.radians(theta_deg))
    H = K @ R @ np.linalg.inv(K)
    c = cv2.perspectiveTransform(SQ.reshape(-1, 1, 2), H).reshape(-1, 2)
    lo, hi = c.min(0), c.max(0)
    s = W / (hi - lo).max()
    S = np.array([[s, 0, -lo[0] * s], [0, s, -lo[1] * s], [0, 0, 1]])
    return S @ H


def grating(shape, period, angle, phase=0.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    u = xx * np.cos(angle) + yy * np.sin(angle)
    return np.cos(2 * np.pi * u / period + phase)


def degrade(img, kind, lvl):
    """Return (degraded image, mask of valid pixels or None, homography original→degraded)."""
    g = img.astype(np.float32)
    Hd = np.eye(3)
    mask = None
    if kind == "tilt":
        Hd = tilt_h([25, 45, 60][lvl], rng.uniform(0, np.pi))
    elif kind == "blur":
        g = cv2.GaussianBlur(g, (0, 0), [4, 8, 12][lvl])
    elif kind == "motion":
        L = [15, 30, 50][lvl]
        k = np.zeros((L, L), np.float32); k[L // 2, :] = 1.0 / L
        k = cv2.warpAffine(k, cv2.getRotationMatrix2D((L / 2, L / 2), rng.uniform(0, 180), 1.0), (L, L))
        g = cv2.filter2D(g, -1, k / k.sum())
    elif kind == "noise_jpeg":
        g = g + rng.normal(0, [10, 20, 35][lvl], g.shape)
        ok, buf = cv2.imencode(".jpg", np.clip(g, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, [50, 30, 15][lvl]])
        g = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    elif kind == "moire":
        a = [0.15, 0.3, 0.5][lvl]
        m = 0.6 * grating(g.shape, rng.uniform(2.5, 5), rng.uniform(0, np.pi)) + \
            0.4 * grating(g.shape, rng.uniform(2.5, 5), rng.uniform(0, np.pi))
        low = cv2.GaussianBlur(grating(g.shape, rng.uniform(40, 90), rng.uniform(0, np.pi)), (0, 0), 3)
        g = g * (1 + a * (0.6 * m + 0.4 * low))
    elif kind == "photo_glare":
        g = 255 * (np.clip(g, 0, 255) / 255) ** [0.7, 0.5, 0.35][lvl]
        cy, cx = rng.uniform(150, 490, 2)
        yy, xx = np.mgrid[0:W, 0:W]
        g = g + [80, 150, 220][lvl] * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 130 ** 2))
    elif kind == "phone_combo":
        Hd = tilt_h([20, 35, 50][lvl], rng.uniform(0, np.pi))
        g = cv2.GaussianBlur(g, (0, 0), [3, 5, 8][lvl])
        a = [0.1, 0.2, 0.35][lvl]
        g = g * (1 + a * grating(g.shape, rng.uniform(2.5, 5), rng.uniform(0, np.pi)))
        g = 255 * (np.clip(g, 0, 255) / 255) ** [0.8, 0.6, 0.45][lvl]
        cy, cx = rng.uniform(150, 490, 2)
        yy, xx = np.mgrid[0:W, 0:W]
        g = g + [60, 110, 170][lvl] * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 130 ** 2))
        g = g + rng.normal(0, [6, 12, 20][lvl], g.shape)
        ok, buf = cv2.imencode(".jpg", np.clip(g, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, [70, 50, 35][lvl]])
        g = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    out = np.clip(g, 0, 255).astype(np.uint8)
    if not np.allclose(Hd, np.eye(3)):
        out = cv2.warpPerspective(out, Hd, (W, W), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpPerspective(np.full((W, W), 255, np.uint8), Hd, (W, W), flags=cv2.INTER_NEAREST)
        mask = cv2.erode(mask, np.ones((7, 7), np.uint8))
    return out, mask, Hd


rows = {r["i"]: r for r in ref["rows"] if r["ok"]}
ids = list(range(0, 50, 3))
kinds = ["tilt", "blur", "motion", "noise_jpeg", "moire", "photo_glare", "phone_combo"]
report = {}
t0 = time.time()
for kind in kinds:
    for lvl in range(3):
        ok = wrong = 0
        errs = []
        for i in ids:
            img = cv2.imread(str(root / "crops" / f"crop_{i}.png"), cv2.IMREAD_GRAYSCALE)
            Hmap = cv2.getPerspectiveTransform(SQ, np.float32(rows[i]["corners"]))
            dimg, mask, Hd = degrade(img, kind, lvl)
            c_true = cv2.perspectiveTransform(np.float32([[[W / 2, W / 2]]]), Hmap)[0, 0]
            c_deg = cv2.perspectiveTransform(np.float32([[[W / 2, W / 2]]]), Hd)[0, 0]
            res = localize(dimg, mask)
            if res is None or res[1] < 25:
                continue  # no fix declared
            c_est = cv2.perspectiveTransform(c_deg.reshape(1, 1, 2), res[0])[0, 0]
            e = float(np.linalg.norm(c_est - c_true))
            if e < 5:
                ok += 1; errs.append(e)
            elif e > 20:
                wrong += 1
        n = len(ids)
        report[f"{kind}[{lvl}]"] = dict(success=round(ok / n, 2), wrong_fix=round(wrong / n, 2),
                                        median_err_px=round(float(np.median(errs)), 2) if errs else None)
        print(f"{kind:12s} L{lvl}: success {ok:2d}/{n}  wrong fixes {wrong:2d}  "
              f"median err {np.median(errs) if errs else float('nan'):.2f} px   ({time.time() - t0:.0f}s)", flush=True)
        if lvl == 2 and i == ids[-1]:
            cv2.imwrite(str(Path(sys.argv[2]).parent / f"stress_{kind}.jpg"), dimg)
Path(sys.argv[2]).with_name("stress.json").write_text(json.dumps(report, indent=1))
