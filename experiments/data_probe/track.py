"""Tracking-mode probe. A tracker always has a pose prior (the previous fix plus a motion model). Given such a
prior, how well do local methods localize the same degraded frames that defeat global search in stress.py?

Methods (each returns a frame->map homography; error = distance between the estimated and the true map position
of the frame centre, in map px):
  global    RootSIFT + FLANN against the whole map, no prior (stress.py; the degraded images are identical)
  prior     the prediction itself, uncorrected (how wrong the tracker's guess is)
  win_sift  predict-warp-match: warp the frame into the map frame with the prior (removes rotation, scale and
            tilt), detect SIFT there, match only against map keypoints inside the predicted window, MAGSAC++
  ncc       predict-warp-correlate (LVS-style): normalized cross-correlation of the warped frame against the
            map window, translation-only correction with a sub-pixel peak
  ecc       ECC homography refinement (findTransformECC, 2-level pyramid) started directly from the prior
  ncc_ecc   ncc, then ECC from the corrected prior

Priors: the true frame->map homography perturbed in map coordinates by translation (sigma_t per axis), rotation
(sigma_theta) and scale (sigma_s) about the true centre, plus a centre-preserving tilt error (sigma_phi) in frame
coordinates. Three levels: tight (smooth tracking), loose (a few missed frames), coast (long dropout).

Usage: python track.py <data root> <probe.json>   -> writes track.json next to probe.json
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

root = Path(sys.argv[1])
ref = json.loads(Path(sys.argv[2]).read_text())
out_dir = Path(sys.argv[2]).parent
rng = np.random.default_rng(0)        # degradations: same seed and call order as stress.py -> identical images
rng_prior = np.random.default_rng(1)  # priors: a separate stream, so it cannot disturb the degradations

mp = cv2.imread(str(root / "global_map.png"), cv2.IMREAD_GRAYSCALE)
MH, MW = mp.shape
sift = cv2.SIFT_create(nfeatures=0)


def rootsift(d):
    d = d / (np.abs(d).sum(axis=1, keepdims=True) + 1e-7)
    return np.sqrt(d).astype(np.float32)


kpm, dm = sift.detectAndCompute(mp, None)
ptsm = np.float32([k.pt for k in kpm])
dmr = rootsift(dm)
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=8), dict(checks=128))
flann.add([dmr])
flann.train()
bf = cv2.BFMatcher(cv2.NORM_L2)

W = 640
SQ = np.float32([(0, 0), (W, 0), (W, W), (0, W)])
C0 = np.array([W / 2, W / 2])
PRIORS = {"tight": (8, 5, 0.05, 3), "loose": (20, 15, 0.15, 8), "coast": (40, 30, 0.25, 12)}
ACCEPT = {"global": 25, "win_sift": 25, "ncc": 0.4, "ecc": 0.5, "ncc_ecc": 0.5}  # declared before the run
CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)


# ---------------------------------------------------------------- geometry helpers
def tr(t):
    return np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], float)


def apply(H, p):
    q = H @ np.array([p[0], p[1], 1.0])
    return q[:2] / q[2]


def resize_h(sx, sy):
    """Homography of cv2.resize under OpenCV's pixel-centre convention."""
    return np.array([[sx, 0, 0.5 * (sx - 1)], [0, sy, 0.5 * (sy - 1)], [0, 0, 1]])


def local_scale(H, p):
    """Map px per frame px at p (square root of the Jacobian determinant)."""
    a = apply(H, p)
    J = np.column_stack([apply(H, p + [1, 0]) - a, apply(H, p + [0, 1]) - a])
    return float(np.sqrt(abs(np.linalg.det(J))))


def in_map(o, w, h):
    return o[0] >= 0 and o[1] >= 0 and o[0] + w <= MW and o[1] + h <= MH


# ---------------------------------------------------------------- image helpers
def hp(img, m=None, sigma=6.0):
    """High-pass (removes glare, gamma and exposure gradients). With a mask it uses normalized convolution,
    so the border of the valid region does not create artificial edges."""
    x = img.astype(np.float32)
    if m is None:
        return cv2.GaussianBlur(x - cv2.GaussianBlur(x, (0, 0), sigma), (0, 0), 0.8)
    mf = (m > 0).astype(np.float32)
    low = cv2.GaussianBlur(x * mf, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(mf, (0, 0), sigma), 1e-3)
    return cv2.GaussianBlur((x - low) * mf, (0, 0), 0.8) * mf


mph = hp(mp)


def shrink(img, mask, s):
    """Shrink a frame to about map GSD with INTER_AREA (anti-aliasing); returns image, mask, resize homography."""
    w0 = max(16, int(round(img.shape[1] * s)))
    h0 = max(16, int(round(img.shape[0] * s)))
    small = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_AREA)
    vm = np.full(img.shape, 255, np.uint8) if mask is None else mask
    vms = cv2.resize(vm, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return small, vms, resize_h(w0 / img.shape[1], h0 / img.shape[0])


def warp_to_map(img, mask, H, p, o, size):
    """Warp a frame into the map-aligned window with top-left o (map px) and size (w, h) using frame->map H."""
    small, vms, Sr = shrink(img, mask, min(1.0, local_scale(H, p)))
    Hres = tr(-np.asarray(o, float)) @ H @ np.linalg.inv(Sr)
    wimg = cv2.warpPerspective(small, Hres, size, flags=cv2.INTER_LINEAR, borderValue=0)
    wm = cv2.warpPerspective(vms, Hres, size, flags=cv2.INTER_NEAREST, borderValue=0)
    return wimg, cv2.erode(wm, np.ones((3, 3), np.uint8))


# ---------------------------------------------------------------- degradations (verbatim from stress.py)
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


# ---------------------------------------------------------------- the prior
def tilt_err_h(phi_deg, axis_angle, p):
    """Perspective error in frame coordinates (camera rotated about an in-plane axis), shifted to keep p fixed."""
    f = W / (2 * np.tan(np.radians(30)))
    K = np.array([[f, 0, W / 2], [0, f, W / 2], [0, 0, 1]])
    R, _ = cv2.Rodrigues(np.array([np.cos(axis_angle), np.sin(axis_angle), 0.0]) * np.radians(phi_deg))
    H = K @ R @ np.linalg.inv(K)
    return tr(p - apply(H, p)) @ H


def make_prior(Hgt, c_true, c_deg, sig):
    st, sth, ss, sphi = sig
    dt = rng_prior.normal(0, st, 2)
    th = np.radians(rng_prior.normal(0, sth))
    s = max(0.5, 1 + rng_prior.normal(0, ss))
    phi, ax = rng_prior.normal(0, sphi), rng_prior.uniform(0, np.pi)
    Rs = np.array([[s * np.cos(th), -s * np.sin(th), 0], [s * np.sin(th), s * np.cos(th), 0], [0, 0, 1]])
    return tr(c_true + dt) @ Rs @ tr(-c_true) @ Hgt @ tilt_err_h(phi, ax, c_deg)


# ---------------------------------------------------------------- methods
def m_global(img, mask):
    """stress.py's localize(): RootSIFT at three scales against the whole map."""
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


def m_win_sift(img, mask, Hp, c_deg, st):
    half = int(120 + 3 * st + 10)
    o = np.floor(apply(Hp, c_deg) - half).astype(int)
    o = np.clip(o, 0, [MW - 2 * half, MH - 2 * half])
    wimg, wm = warp_to_map(img, mask, Hp, c_deg, o, (2 * half, 2 * half))
    kp, d = sift.detectAndCompute(wimg, cv2.erode(wm, np.ones((5, 5), np.uint8)))
    if d is None or len(kp) < 10:
        return None
    x, y = ptsm[:, 0], ptsm[:, 1]
    sel = np.where((x >= o[0]) & (x < o[0] + 2 * half) & (y >= o[1]) & (y < o[1] + 2 * half))[0]
    if len(sel) < 10:
        return None
    knn = bf.knnMatch(rootsift(d), dmr[sel], k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < 8:
        return None
    src = np.float32([kp[m.queryIdx].pt for m in good]) + o.astype(np.float32)  # predicted map coords
    dst = ptsm[sel[[m.trainIdx for m in good]]]
    Hc, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 6.0, maxIters=20000, confidence=0.9999)
    if Hc is None:
        return None
    return Hc @ Hp, int(inl.sum())


def m_ncc(img, mask, Hp, c_deg, st, T=96):
    R = int(np.ceil(3 * st)) + 4
    o_t = np.round(apply(Hp, c_deg) - T / 2).astype(int)
    o_s = o_t - R
    if not in_map(o_s, T + 2 * R, T + 2 * R):
        return None
    tw, tm = warp_to_map(img, mask, Hp, c_deg, o_t, (T, T))
    if (tm > 0).mean() < 0.3:
        return None
    res = cv2.matchTemplate(mph[o_s[1]:o_s[1] + T + 2 * R, o_s[0]:o_s[0] + T + 2 * R], hp(tw, tm),
                            cv2.TM_CCOEFF_NORMED, mask=tm)
    res = np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, peak, _, (u, v) = cv2.minMaxLoc(res)

    def sub(a, b, c):
        den = a - 2 * b + c
        return 0.0 if abs(den) < 1e-9 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))

    du = sub(res[v, u - 1], res[v, u], res[v, u + 1]) if 0 < u < res.shape[1] - 1 else 0.0
    dv = sub(res[v - 1, u], res[v, u], res[v + 1, u]) if 0 < v < res.shape[0] - 1 else 0.0
    r2 = res.copy()
    r2[max(0, v - 5):v + 6, max(0, u - 5):u + 6] = -1
    d = np.array([u + du - R, v + dv - R])
    return tr(d) @ Hp, float(peak), float(r2.max())


def m_ecc(img, mask, Hp, c_deg, T=128, levels=2):
    o = np.round(apply(Hp, c_deg) - T / 2).astype(int)
    if not in_map(o, T, T):
        return None
    small, vms, Sr = shrink(img, mask, min(1.0, local_scale(Hp, c_deg)))
    vms = cv2.erode(vms, np.ones((3, 3), np.uint8))
    inp = hp(small, vms)
    tmpl = mph[o[1]:o[1] + T, o[0]:o[0] + T]
    Wt = np.linalg.inv(Hp @ np.linalg.inv(Sr)) @ tr(o)  # template (map window) -> input (shrunk frame)
    Wt /= Wt[2, 2]
    cc = None
    try:
        for lv in reversed(range(levels)):
            k = 2 ** lv
            D = np.diag([1 / k, 1 / k, 1.0])
            tl = tmpl if k == 1 else cv2.resize(tmpl, (T // k, T // k), interpolation=cv2.INTER_AREA)
            sz = (inp.shape[1] // k, inp.shape[0] // k)
            il = inp if k == 1 else cv2.resize(inp, sz, interpolation=cv2.INTER_AREA)
            ml = vms if k == 1 else cv2.resize(vms, sz, interpolation=cv2.INTER_NEAREST)
            cc, Wl = cv2.findTransformECC(tl, il, (D @ Wt @ np.linalg.inv(D)).astype(np.float32),
                                          cv2.MOTION_HOMOGRAPHY, CRIT, ml, 5)
            Wt = np.linalg.inv(D) @ Wl.astype(float) @ D
    except cv2.error:
        return None
    return tr(o) @ np.linalg.inv(Wt) @ Sr, float(cc)


# ---------------------------------------------------------------- run
def err(H, c_deg, c_true):
    return float(np.linalg.norm(apply(H, c_deg) - c_true))


rows = {r["i"]: r for r in ref["rows"] if r["ok"]}
ids = list(range(0, 50, 3))
kinds = ["tilt", "blur", "motion", "noise_jpeg", "moire", "photo_glare", "phone_combo"]
QUICK = len(sys.argv) > 3 and sys.argv[3] == "--quick"  # smoke test; the degraded images then differ from stress.py
if QUICK:
    ids, kinds = [0, 24], ["tilt", "phone_combo"]
trials = []
t0 = time.time()
example = None
for kind in kinds:
    for lvl in range(3):
        for i in ids:
            img = cv2.imread(str(root / "crops" / f"crop_{i}.png"), cv2.IMREAD_GRAYSCALE)
            Hmap = cv2.getPerspectiveTransform(SQ, np.float32(rows[i]["corners"])).astype(float)
            dimg, mask, Hd = degrade(img, kind, lvl)
            Hgt = Hmap @ np.linalg.inv(Hd)  # degraded frame -> map
            c_true = apply(Hmap, C0)
            c_deg = apply(Hd, C0)
            tr_row = dict(kind=kind, lvl=lvl, i=i)
            ts = time.time(); g = m_global(dimg, mask)
            tr_row["global"] = dict(e=None if g is None else err(g[0], c_deg, c_true),
                                    conf=None if g is None else g[1], ms=round(1000 * (time.time() - ts), 1))
            for pname, sig in PRIORS.items():
                Hp = make_prior(Hgt, c_true, c_deg, sig)
                res = {"prior": dict(e=err(Hp, c_deg, c_true), conf=None, ms=0.0)}
                ts = time.time(); r = m_win_sift(dimg, mask, Hp, c_deg, sig[0])
                res["win_sift"] = dict(e=None if r is None else err(r[0], c_deg, c_true),
                                       conf=None if r is None else r[1], ms=round(1000 * (time.time() - ts), 1))
                ts = time.time(); r = m_ncc(dimg, mask, Hp, c_deg, sig[0])
                res["ncc"] = dict(e=None if r is None else err(r[0], c_deg, c_true),
                                  conf=None if r is None else r[1], second=None if r is None else r[2],
                                  ms=round(1000 * (time.time() - ts), 1))
                Hn = None if r is None else r[0]
                ts = time.time(); r = m_ecc(dimg, mask, Hp, c_deg)
                res["ecc"] = dict(e=None if r is None else err(r[0], c_deg, c_true),
                                  conf=None if r is None else r[1], ms=round(1000 * (time.time() - ts), 1))
                ts = time.time(); r = None if Hn is None else m_ecc(dimg, mask, Hn, c_deg)
                res["ncc_ecc"] = dict(e=None if r is None else err(r[0], c_deg, c_true),
                                      conf=None if r is None else r[1],
                                      ms=round(1000 * (time.time() - ts) + res["ncc"]["ms"], 1))
                tr_row[pname] = res
                if kind == "phone_combo" and lvl == 2 and pname == "tight" and i == 24:
                    example = (dimg, mask, Hp, r[0] if r is not None else None, c_deg)
            trials.append(tr_row)
        # progress line for this condition
        sel = [t for t in trials if t["kind"] == kind and t["lvl"] == lvl]
        n = len(sel)

        def score(get, thr):
            ok = sum(1 for t in sel if get(t)["e"] is not None and (thr is None or (get(t)["conf"] or 0) >= thr)
                     and get(t)["e"] < 5)
            wr = sum(1 for t in sel if get(t)["e"] is not None and (thr is None or (get(t)["conf"] or 0) >= thr)
                     and get(t)["e"] > 20)
            return f"{ok:2d}/{wr:<2d}"

        parts = [f"global {score(lambda t: t['global'], ACCEPT['global'])}"]
        for pname in PRIORS:
            parts.append(pname + ": " + " ".join(
                f"{m} {score(lambda t, m=m, p=pname: t[p][m], ACCEPT.get(m))}" for m in
                ("prior", "win_sift", "ncc", "ecc", "ncc_ecc")))
        print(f"{kind:12s} L{lvl}  " + " | ".join(parts) + f"   ({time.time() - t0:.0f}s)", flush=True)

(out_dir / ("track_quick.json" if QUICK else "track.json")).write_text(
    json.dumps(dict(priors=PRIORS, accept=ACCEPT, trials=trials), indent=0))

# one illustration: the hardest phone condition, warped into the map frame by the tight prior vs. the refined fix
if example is not None:
    dimg, mask, Hp, Hf, c_deg = example
    pc = apply(Hp, c_deg)
    o = np.round(pc - 110).astype(int)
    pw, _ = warp_to_map(dimg, mask, Hp, c_deg, o, (220, 220))
    panels = [cv2.resize(dimg, (220, 220), interpolation=cv2.INTER_AREA), pw]
    if Hf is not None:
        fw, _ = warp_to_map(dimg, mask, Hf, c_deg, o, (220, 220))
        panels.append(fw)
    panels.append(mp[o[1]:o[1] + 220, o[0]:o[0] + 220])
    cv2.imwrite(str(out_dir / "track_example.jpg"), cv2.resize(np.hstack(panels), None, fx=2, fy=2,
                                                               interpolation=cv2.INTER_NEAREST))
print("done", round(time.time() - t0), "s")
