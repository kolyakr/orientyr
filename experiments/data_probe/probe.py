"""Data probe: register every crop to the map with RootSIFT + MAGSAC++ and report
rotation, scale, perspective, motion between frames, and imagery-source differences."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

mp_bgr = cv2.imread(str(root / "global_map.png"), cv2.IMREAD_COLOR)
mp = cv2.cvtColor(mp_bgr, cv2.COLOR_BGR2GRAY)
H_map, W_map = mp.shape
sift = cv2.SIFT_create(nfeatures=0)


def rootsift(d: np.ndarray) -> np.ndarray:
    d = d / (np.abs(d).sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(d).astype(np.float32)


t = time.time()
kpm, dm = sift.detectAndCompute(mp, None)
t_map = time.time() - t
ptsm = np.float32([k.pt for k in kpm])
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=8), dict(checks=128))
flann.add([rootsift(dm)])
flann.train()
print(f"map {W_map}x{H_map}  keypoints {len(kpm)}  sift {t_map:.1f}s")


def warp_pt(Hm, x, y):
    p = Hm @ np.array([x, y, 1.0])
    return p[:2] / p[2]


rows = []
for i in range(50):
    c_bgr = cv2.imread(str(root / "crops" / f"crop_{i}.png"), cv2.IMREAD_COLOR)
    c = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2GRAY)
    h, w = c.shape
    best = None
    t0 = time.time()
    for f in (1.0, 0.5, 0.25):
        cs = c if f == 1.0 else cv2.resize(c, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        kpc, dc = sift.detectAndCompute(cs, None)
        if dc is None or len(kpc) < 10:
            continue
        knn = flann.knnMatch(rootsift(dc), k=2)
        good = [p[0] for p in knn if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
        if len(good) < 8:
            continue
        src = np.float32([kpc[m.queryIdx].pt for m in good]) / f
        dst = ptsm[[m.trainIdx for m in good]]
        Hm, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 6.0, maxIters=20000, confidence=0.9999)
        if Hm is None:
            continue
        inl = int(mask.sum())
        if best is None or inl > best["inliers"]:
            best = dict(inliers=inl, f=f, H=Hm, good=len(good), kp=len(kpc), mask=mask.ravel().astype(bool),
                        src=src, dst=dst)
    dt = time.time() - t0
    if best is None:
        rows.append(dict(i=i, ok=False, t=dt))
        print(f"crop {i:2d}: FAIL")
        continue
    Hm = best["H"]
    cx, cy = warp_pt(Hm, w / 2, h / 2)
    ex = warp_pt(Hm, w / 2 + 1, h / 2) - np.array([cx, cy])
    ey = warp_pt(Hm, w / 2, h / 2 + 1) - np.array([cx, cy])
    angle = float(np.degrees(np.arctan2(ex[1], ex[0])))
    scale = float(np.sqrt(abs(ex[0] * ey[1] - ex[1] * ey[0])))  # map px per crop px (area-based)
    aniso = float(np.linalg.norm(ex) / max(np.linalg.norm(ey), 1e-9))
    corners = np.array([warp_pt(Hm, x, y) for x, y in [(0, 0), (w, 0), (w, h), (0, h)]], np.float32)
    # residuals of inliers
    proj = cv2.perspectiveTransform(best["src"][best["mask"]].reshape(-1, 1, 2), Hm).reshape(-1, 2)
    reproj = float(np.median(np.linalg.norm(proj - best["dst"][best["mask"]], axis=1)))
    # perspective strength: how far the 4 corners deviate from the best affine fit
    A, _ = cv2.estimateAffine2D(np.float32([(0, 0), (w, 0), (w, h), (0, h)]), corners)
    aff = (np.hstack([np.float32([(0, 0), (w, 0), (w, h), (0, h)]), np.ones((4, 1), np.float32)]) @ A.T)
    persp_px = float(np.abs(aff - corners).max())
    # photometric / resolution comparison: warp the map patch into crop frame
    Hinv = np.linalg.inv(Hm)
    patch = cv2.warpPerspective(mp_bgr, Hinv, (w, h), flags=cv2.INTER_CUBIC)
    valid = cv2.warpPerspective(np.ones_like(mp), Hinv, (w, h), flags=cv2.INTER_NEAREST) > 0
    g_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g_crop = c.astype(np.float32)
    # high-frequency energy (Laplacian variance) of crop vs upsampled map patch
    lap = lambda g: float(cv2.Laplacian(g, cv2.CV_32F)[valid].var())
    ncc = float(np.corrcoef(g_patch[valid], g_crop[valid])[0, 1])
    # NCC after blurring the crop to map resolution (tests "same source, just finer" hypothesis)
    sigma = max(0.5, 0.5 / max(scale, 1e-3))
    g_crop_blur = cv2.GaussianBlur(g_crop, (0, 0), sigma)
    ncc_blur = float(np.corrcoef(g_patch[valid], g_crop_blur[valid])[0, 1])
    mean_c = c_bgr.reshape(-1, 3)[valid.ravel()].mean(0)[::-1].round(1).tolist()
    mean_m = patch.reshape(-1, 3)[valid.ravel()].mean(0)[::-1].round(1).tolist()
    rows.append(dict(i=i, ok=True, inliers=best["inliers"], good=best["good"], kp=best["kp"], f=best["f"],
                     inlier_ratio=round(best["inliers"] / best["good"], 3), cx=round(float(cx), 1),
                     cy=round(float(cy), 1), angle=round(angle, 1), scale=round(scale, 4), aniso=round(aniso, 3),
                     persp_px=round(persp_px, 2), reproj_px=round(reproj, 2), lap_crop=round(lap(g_crop), 1),
                     lap_mappatch=round(lap(g_patch), 1), ncc=round(ncc, 3), ncc_blur=round(ncc_blur, 3),
                     rgb_crop=mean_c, rgb_map=mean_m, corners=corners.round(1).tolist(), t=round(dt, 2)))
    print(f"crop {i:2d}: inl {best['inliers']:4d}/{best['good']:4d} f={best['f']:.2f} c=({cx:7.1f},{cy:7.1f}) "
          f"ang {angle:7.1f} s {scale:.3f} persp {persp_px:5.2f}px rep {reproj:.2f} ncc {ncc:.2f}/{ncc_blur:.2f} "
          f"lap {lap(g_crop):.0f}/{lap(g_patch):.0f} {dt:.1f}s")
    # save a side-by-side for a few frames
    if i in (0, 25, 49):
        side = np.hstack([c_bgr, patch])
        cv2.imwrite(str(out / f"side_{i}.jpg"), side, [cv2.IMWRITE_JPEG_QUALITY, 85])

# consecutive motion and overlap
ok = [r for r in rows if r["ok"]]
for a, b in zip(rows[:-1], rows[1:]):
    if not (a["ok"] and b["ok"]):
        continue
    pa = np.float32(a["corners"])
    pb = np.float32(b["corners"])
    inter, _ = cv2.intersectConvexConvex(pa, pb)
    area_a = cv2.contourArea(pa)
    a["step_px"] = round(float(np.hypot(b["cx"] - a["cx"], b["cy"] - a["cy"])), 1)
    a["overlap_next"] = round(float(inter / area_a), 3) if area_a > 0 else 0.0
    d = (b["angle"] - a["angle"] + 180) % 360 - 180
    a["dangle_next"] = round(d, 1)

(out / "probe.json").write_text(json.dumps(dict(map=[W_map, H_map], map_kp=len(kpm), rows=rows), indent=1))


def stats(key):
    v = np.array([r[key] for r in ok if key in r], float)
    return f"{key}: min {v.min():.3f}  median {np.median(v):.3f}  max {v.max():.3f}"


print(f"\nlocalized {len(ok)}/50")
for k in ("inliers", "inlier_ratio", "scale", "aniso", "persp_px", "reproj_px", "ncc", "ncc_blur", "lap_crop",
          "lap_mappatch", "step_px", "overlap_next", "dangle_next", "t"):
    print(stats(k))
angles = np.array([r["angle"] for r in ok])
print("angles (deg):", np.round(np.sort(angles), 0).tolist())
print("footprint side in map px (median):", round(640 * float(np.median([r['scale'] for r in ok])), 1))

# overview image: route on downscaled map
s = 0.5
ov = cv2.resize(mp_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
for r in ok:
    pts = (np.float32(r["corners"]) * s).astype(np.int32)
    cv2.polylines(ov, [pts], True, (0, 200, 255), 1, cv2.LINE_AA)
for a, b in zip(ok[:-1], ok[1:]):
    cv2.line(ov, (int(a["cx"] * s), int(a["cy"] * s)), (int(b["cx"] * s), int(b["cy"] * s)), (0, 0, 255), 2,
             cv2.LINE_AA)
for r in ok:
    cv2.putText(ov, str(r["i"]), (int(r["cx"] * s) + 4, int(r["cy"] * s) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
cv2.imwrite(str(out / "route_probe.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 88])
