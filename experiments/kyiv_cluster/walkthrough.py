"""Walk one simulated flight through every block of the Onboard Navigator and draw each step.

The flight is 22 frames over the seam-free Irpin tile (2025 imagery, 1.52 m/px). The map the drone
carries is the same 2025 tile, so the classic matcher (SIFT) is enough; the point here is to show the
*mechanics* of the pipeline, not the cross-year difficulty. Three kinds of trouble are injected:

  * frame 0       : no prior at all            -> global search over the whole Map Pack (LOST -> RELOCALIZING)
  * frames 11-13  : video link jammed (blurred) -> the blur check rejects them, Kalman filter coasts, LOST after 3
  * frame 18      : this frame shows 2022 ground (war imagery) while the map is 2025 -> inliers collapse (~40 vs ~1300) but the fix survives

Usage: python walkthrough.py <towns_seamfree_dir> <out_dir>
Writes w1..w6 pictures and walkthrough.json (one row per frame).
"""
import json
import math
import os
import sys

import cv2
import numpy as np

FW, FH = 640, 480                 # camera frame size (px)
M_PER_PX = 1.5179                 # Web-Mercator z16 at Kyiv latitude
N_FRAMES = 22
JAM = {11, 12, 13}                # frames whose video is jammed (heavy blur)
OTHER_YEAR = {18}                 # frames cut from 2022 imagery instead of 2025
SHOW = 6                          # the frame drawn step by step in w2..w5
BLUR_MIN = 120.0                  # variance of Laplacian below this = unusable frame
MIN_INLIERS = 12
SIGMA_ACC = 8.0                   # Kalman: expected change of velocity per frame (px/frame^2)
SIGMA_FIX = 4.0                   # Kalman: expected error of one visual fix (px)


# ---------------------------------------------------------------- simulation (the "world")
def route():
    """Ground-truth centres (x, y) and compass headings (deg, clockwise from north) of the flight."""
    t = np.linspace(0, 1, 400)
    x = 450 + 1100 * t
    y = 1500 - 1050 * t + 120 * np.sin(2 * np.pi * t)
    s = np.r_[0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]          # resample at equal spacing
    u = np.linspace(0, s[-1], N_FRAMES)
    xs, ys = np.interp(u, s, x), np.interp(u, s, y)
    dx, dy = np.gradient(xs), np.gradient(ys)
    return xs, ys, np.degrees(np.arctan2(dx, -dy)) % 360


def camera(img, cx, cy, heading, rng, blur):
    """What the drone camera sees: map rotated so that the drone's forward points up, slightly zoomed and blurred."""
    s = rng.uniform(0.95, 1.05)
    M = cv2.getRotationMatrix2D((cx, cy), heading, s)
    M[:, 2] += [FW / 2 - cx, FH / 2 - cy]
    f = cv2.warpAffine(img, M, (FW, FH), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT)
    f = cv2.GaussianBlur(f, (0, 0), blur)
    return np.clip(f.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.normal(0, 3, f.shape), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- step 1: prepare
def prepare(frame, compass_deg, clahe):
    """Grey + CLAHE, then rotate north-up using the (noisy) compass. Returns image, valid mask, 3x3 transform."""
    g = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    D = int(math.ceil(math.hypot(FW, FH)))
    R = cv2.getRotationMatrix2D((FW / 2, FH / 2), -compass_deg, 1.0)
    R[:, 2] += [D / 2 - FW / 2, D / 2 - FH / 2]
    up = cv2.warpAffine(g, R, (D, D))
    mask = cv2.erode(cv2.warpAffine(np.full_like(g, 255), R, (D, D)), np.ones((9, 9), np.uint8))
    return up, mask, np.vstack([R, [0, 0, 1]])


# ---------------------------------------------------------------- step 3: match
def match(sift, up, mask, map_kp, map_desc, rect):
    """SIFT on the frame, ratio test against the Map Pack keypoints inside rect, MAGSAC homography."""
    kp, d = sift.detectAndCompute(up, mask)
    x0, y0, x1, y1 = rect
    sel = [i for i, k in enumerate(map_kp) if x0 <= k.pt[0] < x1 and y0 <= k.pt[1] < y1]
    if d is None or len(kp) < 8 or len(sel) < 8:
        return None, kp, [], None, len(sel)
    pts_m = np.float32([map_kp[i].pt for i in sel])
    good = [m for m, n in cv2.BFMatcher().knnMatch(d, map_desc[sel], k=2) if m.distance < 0.8 * n.distance]
    if len(good) < 8:
        return None, kp, good, None, len(sel)
    src = np.float32([kp[m.queryIdx].pt for m in good])
    dst = pts_m[[m.trainIdx for m in good]]
    H, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 5.0)
    return H, kp, [(tuple(src[i]), tuple(dst[i]), bool(inl[i])) for i in range(len(good))] if H is not None else [], inl, len(sel)


# ---------------------------------------------------------------- step 5: pose
def pose(T):
    """Frame -> map transform to centre (map px), heading (deg) and scale (map px per frame px)."""
    c = cv2.perspectiveTransform(np.float32([[[FW / 2, FH / 2], [FW / 2, FH / 2 - 100]]]), T)[0]
    heading = math.degrees(math.atan2(c[1, 0] - c[0, 0], -(c[1, 1] - c[0, 1]))) % 360
    return c[0], heading, float(np.hypot(*(c[1] - c[0])) / 100)


def footprint(T):
    return cv2.perspectiveTransform(np.float32([[[0, 0], [FW, 0], [FW, FH], [0, FH]]]), T)[0]


# ---------------------------------------------------------------- step 6: Kalman filter
class Kalman:
    """Constant-velocity filter on (x, y, vx, vy) in map pixels, one step per frame."""

    def __init__(self, xy):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0])
        self.P = np.diag([SIGMA_FIX ** 2, SIGMA_FIX ** 2, 80.0 ** 2, 80.0 ** 2])   # speed unknown at first
        self.F = np.eye(4); self.F[0, 2] = self.F[1, 3] = 1
        G = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]])
        self.Q = G @ G.T * SIGMA_ACC ** 2
        self.Hm = np.eye(2, 4)
        self.R = np.eye(2) * SIGMA_FIX ** 2

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, z):
        S = self.Hm @ self.P @ self.Hm.T + self.R
        K = self.P @ self.Hm.T @ np.linalg.inv(S)          # how much to trust the fix vs the prediction
        self.x = self.x + K @ (np.asarray(z) - self.Hm @ self.x)
        self.P = (np.eye(4) - K @ self.Hm) @ self.P
        return float(K[0, 0])

    def sigma(self):
        return float(math.sqrt(max(self.P[0, 0], self.P[1, 1])))


# ---------------------------------------------------------------- drawing helpers
def ellipse(img, c, P, color, k=2.0, th=2):
    w, v = np.linalg.eigh(P[:2, :2])
    ang = math.degrees(math.atan2(v[1, 1], v[0, 1]))
    ax = tuple(int(k * math.sqrt(max(e, 1e-6))) for e in w[::-1])
    cv2.ellipse(img, (int(c[0]), int(c[1])), ax, ang, 0, 360, color, th, cv2.LINE_AA)


def label(img, text, org, scale=0.9, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def banner(img, text):
    out = cv2.copyMakeBorder(img, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(out, text, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def crop(img, c, half):
    x0 = int(np.clip(c[0] - half, 0, img.shape[1] - 2 * half)); y0 = int(np.clip(c[1] - half, 0, img.shape[0] - 2 * half))
    return img[y0:y0 + 2 * half, x0:x0 + 2 * half].copy(), (x0, y0)


def draw_matches(up, matches, rect, map_g, path, title, n_show=70):
    """Prepared frame (left) vs search window (right); a sample of inliers (green) and every outlier (red)."""
    r = np.random.default_rng(0)                              # own generator: drawing must not change the flight
    x0, y0 = int(max(rect[0], 0)), int(max(rect[1], 0))
    win = cv2.cvtColor(map_g[y0:int(rect[3]), x0:int(rect[2])], cv2.COLOR_GRAY2BGR)
    sc = FH * 2 / win.shape[0]; win = cv2.resize(win, None, fx=sc, fy=sc)
    left = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR); sl = FH * 2 / left.shape[0]; left = cv2.resize(left, None, fx=sl, fy=sl)
    canvas = np.hstack([left, win]); off = left.shape[1]
    good = [m for m in matches if m[2]]; bad = [m for m in matches if not m[2]]
    pick = [good[k] for k in r.choice(len(good), min(n_show, len(good)), replace=False)] if good else []
    for a, b, ok in bad + pick:
        pa = (int(a[0] * sl), int(a[1] * sl)); pb = (int((b[0] - x0) * sc + off), int((b[1] - y0) * sc))
        col = (60, 220, 60) if ok else (60, 60, 240)
        cv2.line(canvas, pa, pb, col, 2, cv2.LINE_AA)
        cv2.circle(canvas, pa, 5, col, -1); cv2.circle(canvas, pb, 5, col, -1)
    shown = f", {len(pick)} drawn" if len(pick) < len(good) else ""
    cv2.imwrite(path, banner(canvas, f"{title}: {len(matches)} pairs, {len(good)} agree on one homography (green{shown}), {len(bad)} outliers (red)"),
                [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---------------------------------------------------------------- main loop
def main() -> None:
    tiles, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    world = cv2.imread(f"{tiles}/irpin_rebuilt.png")             # what the drone flies over
    other = cv2.imread(f"{tiles}/irpin_war.png")                 # the same place in March 2022
    map_c = world.copy()                                         # the map it carries (same year here)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    map_g = clahe.apply(cv2.cvtColor(map_c, cv2.COLOR_BGR2GRAY))

    # ---- Map Pack (offline): keypoints + descriptors of the whole tile, computed once and cached
    sift_map = cv2.SIFT_create(40000)
    map_kp, map_desc = sift_map.detectAndCompute(map_g, None)
    sift = cv2.SIFT_create(3000)
    w1 = map_c.copy()
    for x in range(0, w1.shape[1], 496):
        cv2.line(w1, (x, 0), (x, w1.shape[0]), (255, 255, 255), 1)
        cv2.line(w1, (0, x), (w1.shape[1], x), (255, 255, 255), 1)
    for k in map_kp[::6]:
        cv2.circle(w1, (int(k.pt[0]), int(k.pt[1])), 2, (0, 255, 255), -1)
    cv2.imwrite(f"{out}/w1_mappack.jpg", banner(cv2.resize(w1, (1100, 1100), interpolation=cv2.INTER_AREA),
                f"Map Pack, Irpin 3 km: {len(map_kp)} keypoints (1 in 6 drawn), 4x4 search tiles"), [cv2.IMWRITE_JPEG_QUALITY, 85])

    xs, ys, hd = route()
    rng = np.random.default_rng(7)
    kf, state, misses, last_fix = None, "LOST", 0, None
    rows, track = [], []
    for i in range(N_FRAMES):
        truth = np.array([xs[i], ys[i]])
        src = other if i in OTHER_YEAR else world
        frame = camera(src, xs[i], ys[i], hd[i], rng, 9.0 if i in JAM else rng.uniform(0.5, 1.0))
        compass = hd[i] + rng.normal(0, 4)                      # magnetometer error, a few degrees
        row = {"i": i, "truth": truth.round(1).tolist(), "state_in": state, "trouble":
               "jammed" if i in JAM else "2022 ground" if i in OTHER_YEAR else ""}
        pred = kf.predict() if kf is not None else None
        P_pred = kf.P.copy() if kf is not None else None

        # step 1: prepare + blur check
        sharp = float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        row["sharpness"] = round(sharp, 1)
        up, mask, Rp = prepare(frame, compass, clahe)

        # step 2: where to look
        if state == "LOST" or kf is None:
            rect = (0, 0, map_g.shape[1], map_g.shape[0]); row["search"] = "global"
        else:
            half = int(min(1000, 380 + 3 * kf.sigma()))
            rect = (pred[0] - half, pred[1] - half, pred[0] + half, pred[1] + half); row["search"] = f"window {2 * half}px"

        fix, why, H, matches = None, "", None, []
        if sharp < BLUR_MIN:
            why = "blur check"
        else:
            # step 3: match
            H, kp, matches, inl, nmap = match(sift, up, mask, map_kp, map_desc, rect)
            row["map_keypoints_searched"] = nmap
            n_in = int(inl.sum()) if H is not None else 0
            row["matches"], row["inliers"] = len(matches), n_in
            # step 4: sanity checks
            if H is None or n_in < MIN_INLIERS:
                why = "too few inliers"
            else:
                T = H @ Rp
                c, head, scale = pose(T)
                sv = np.linalg.svd(T[:2, :2] / T[2, 2], compute_uv=False)
                if not (0.6 < scale < 1.6) or sv[0] / sv[1] > 1.5:
                    why = "bad shape"
                elif pred is not None and state != "LOST" and np.hypot(*(c - pred)) > max(60, 4 * kf.sigma()):
                    why = "jump vs prediction"
                else:
                    fix = (c, head, scale, T)
        row["rejected_by"] = why

        # step 6: Kalman filter + state machine
        if fix is not None:
            c = fix[0]
            if state == "LOST":
                kf, state = Kalman(c), "RELOCALIZING"; gain = 1.0
            else:
                gain = kf.update(c)
                state = "TRACKING"
            misses = 0
            row.update(fix=np.round(c, 1).tolist(), heading=round(fix[1], 1), heading_true=round(hd[i] % 360, 1),
                       altitude_scale=round(fix[2], 3), fix_err_m=round(float(np.hypot(*(c - truth))) * M_PER_PX, 1),
                       kalman_gain=round(gain, 2))
        else:
            misses += 1
            if state == "RELOCALIZING" or misses >= 3:
                state = "LOST"
        est = kf.x[:2].copy() if kf is not None else None
        row["state_out"] = state
        row["sigma_m"] = round(kf.sigma() * M_PER_PX, 1) if kf is not None else None
        row["err_m"] = round(float(np.hypot(*(est - truth))) * M_PER_PX, 1) if est is not None else None
        rows.append(row); track.append((est, kf.P.copy() if kf is not None else None, fix is not None))
        print(i, row["state_in"], "->", state, row.get("search"), row.get("inliers"), why, row["err_m"], flush=True)

        if i in OTHER_YEAR and matches:
            draw_matches(up, matches, rect, map_g, f"{out}/w4b_matches_2022.jpg", f"3  match (frame {i}, 2022 ground, 2025 map)")

        # ---- the step-by-step pictures for one frame
        if i == SHOW:
            raw = frame.copy(); label(raw, f"heading {hd[i]:.0f} deg, compass says {compass:.0f}", (12, 465), 0.7)
            prep = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR); prep[mask == 0] = 40
            prep = cv2.resize(prep, (FH * prep.shape[1] // prep.shape[0], FH))
            cv2.imwrite(f"{out}/w2_prepare.jpg", np.hstack([banner(raw, "1a  what the camera sees"),
                        banner(prep, "1b  CLAHE + north-up")]), [cv2.IMWRITE_JPEG_QUALITY, 85])

            v, (ox, oy) = crop(map_c, truth, 560)
            o = np.array([ox, oy])
            ellipse(v, pred - o, P_pred, (255, 200, 0), 2.0, 3)
            cv2.rectangle(v, tuple(int(a) for a in (np.array(rect[:2]) - o)), tuple(int(a) for a in (np.array(rect[2:]) - o)), (0, 255, 255), 3)
            cv2.drawMarker(v, tuple(int(a) for a in pred - o), (255, 200, 0), cv2.MARKER_CROSS, 30, 3)
            for e, _, _ in track[:-1]:
                if e is not None:
                    cv2.circle(v, tuple(int(a) for a in e - o), 6, (255, 120, 0), -1)
            label(v, "previous fixes", (12, 1100), 0.8, (255, 160, 60))
            label(v, "prediction + 2-sigma", tuple(int(a) for a in pred - o + [25, -20]), 0.8, (255, 220, 120))
            label(v, "search window", (int(rect[0] - ox) + 8, int(rect[1] - oy) + 30), 0.8, (0, 255, 255))
            cv2.imwrite(f"{out}/w3_window.jpg", banner(v, "2  where to look: Kalman prediction -> search window"), [cv2.IMWRITE_JPEG_QUALITY, 85])

            draw_matches(up, matches, rect, map_g, f"{out}/w4_matches.jpg", "3  match (frame 6, 2025 ground)")

            v, (ox, oy) = crop(map_c, truth, 560); o = np.array([ox, oy])
            fp = (footprint(fix[3]) - o).astype(np.int32)
            ov = v.copy(); cv2.fillPoly(ov, [fp], (0, 255, 255)); v = cv2.addWeighted(ov, 0.22, v, 0.78, 0)
            cv2.polylines(v, [fp], True, (0, 255, 255), 3, cv2.LINE_AA)
            c = fix[0] - o; a = math.radians(fix[1])
            cv2.arrowedLine(v, tuple(int(q) for q in c), (int(c[0] + 170 * math.sin(a)), int(c[1] - 170 * math.cos(a))), (0, 0, 255), 4, cv2.LINE_AA, tipLength=0.2)
            cv2.circle(v, tuple(int(q) for q in c), 9, (0, 0, 255), -1)
            cv2.drawMarker(v, tuple(int(q) for q in truth - o), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 26, 3)
            label(v, f"fix {M_PER_PX * np.hypot(*(fix[0] - truth)):.1f} m from truth (white x)", (12, 1060), 0.8)
            label(v, f"heading {fix[1]:.1f} deg (true {hd[i] % 360:.1f}), scale {fix[2]:.2f}", (12, 1100), 0.8)
            cv2.imwrite(f"{out}/w5_pose.jpg", banner(v, "5  pose: footprint, centre, heading"), [cv2.IMWRITE_JPEG_QUALITY, 85])

    json.dump({"rows": rows}, open(f"{out}/walkthrough.json", "w"), indent=1)
    flight_figure(map_c, xs, ys, rows, track, out)


def flight_figure(map_c, xs, ys, rows, track, out):
    """w6: the whole flight on the map, and error / uncertainty / state over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    col = {"LOST": "#9a9a9a", "RELOCALIZING": "#d6453d", "TRACKING": "#3b6fd6"}
    fig = plt.figure(figsize=(12, 6.2), dpi=150)
    ax = fig.add_axes([0.0, 0.02, 0.52, 0.92]); ax.imshow(cv2.cvtColor(map_c, cv2.COLOR_BGR2RGB)); ax.axis("off")
    ax.plot(xs, ys, "--", color="white", lw=1.6, label="true route")
    est = np.array([t[0] if t[0] is not None else [np.nan, np.nan] for t in track])
    ax.plot(est[:, 0], est[:, 1], "-", color="#ffcc00", lw=2.2, label="Kalman track")
    for r, (e, P, ok) in zip(rows, track):
        if e is None:
            continue
        if ok:
            ax.plot(*r["fix"], "o", ms=5, color="#3be36b", mec="k", mew=0.5)
        else:
            w, v = np.linalg.eigh(P[:2, :2])
            ax.add_patch(Ellipse(e, 4 * math.sqrt(w[1]), 4 * math.sqrt(w[0]), angle=math.degrees(math.atan2(v[1, 1], v[0, 1])),
                                 fill=False, ec="#ff5a4e", lw=1.4))
        ax.annotate(str(r["i"]), e, xytext=(6, -10), textcoords="offset points", color="white", fontsize=7)
    ax.plot([], [], "o", color="#3be36b", mec="k", label="accepted visual fix")
    ax.plot([], [], "o", mfc="none", mec="#ff5a4e", label="coasting: 2-sigma uncertainty")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax.set_title("the flight over Irpin (2025 map)", fontsize=10)

    n = len(rows); f = np.arange(n)
    a1 = fig.add_axes([0.58, 0.56, 0.4, 0.36])
    err = [r["err_m"] if r["err_m"] is not None else np.nan for r in rows]
    sig = [2 * r["sigma_m"] if r["sigma_m"] is not None else np.nan for r in rows]
    a1.fill_between(f, 0, sig, color="#3b6fd6", alpha=0.15, label="filter's own 2-sigma")
    a1.plot(f, err, "o-", color="#222", ms=3.5, lw=1.2, label="true error of the track")
    for r in rows:
        if r["trouble"]:
            a1.axvspan(r["i"] - 0.5, r["i"] + 0.5, color="#ff5a4e", alpha=0.12, lw=0)
    a1.set_ylabel("metres"); a1.set_xlim(-0.5, n - 0.5); a1.legend(fontsize=8, loc="upper left"); a1.grid(alpha=0.3)
    a1.set_title("error vs. what the filter believes (red = injected trouble)", fontsize=10)
    a2 = fig.add_axes([0.58, 0.30, 0.4, 0.17], sharex=a1)
    a2.bar(f, [r.get("inliers", 0) or 0 for r in rows], color="#3be36b", ec="k", lw=0.4)
    a2.axhline(MIN_INLIERS, color="#d6453d", lw=1, ls="--"); a2.set_ylabel("inliers"); a2.set_yscale("symlog"); a2.grid(alpha=0.3)
    a3 = fig.add_axes([0.58, 0.16, 0.4, 0.07], sharex=a1)
    for r in rows:
        a3.axvspan(r["i"] - 0.5, r["i"] + 0.5, color=col[r["state_out"]])
    a3.set_yticks([]); a3.set_xlabel("frame")
    for k, c in col.items():
        a3.plot([], [], "s", color=c, label=k)
    a3.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.9), frameon=False)
    fig.savefig(f"{out}/w6_flight.png"); plt.close(fig)


if __name__ == "__main__":
    main()
