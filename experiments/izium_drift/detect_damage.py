"""Unsupervised building-damage detection for Izium: 2020 (pre-war) vs 2022-08-29 (occupation).

There are no UNOSAT labels for Izium, so each OSM building footprint is scored by
how much its roof changed between the two captures. The cues are chosen to be blind
to the nuisance changes we know exist between these captures (season, sun, view angle):

  - struct:  1 - best normalised cross-correlation of the roof, before vs after,
             searched over +-SEARCH px. The search absorbs the off-nadir lean of the 2020 image;
             a roof that is gone/caved in has no good match anywhere.
  - color:   Lab distance of the roof's median colour after per-image histogram matching
             (removes the global season/radiometry shift, keeps "blue roof -> grey rubble").
  - texture: rise in high-frequency energy inside the footprint (roofless shells expose the
             grid of inner walls; rubble is speckled).
  - dark:    rise in the share of very dark pixels (char / open interiors).

Roofs that cannot be compared are flagged, not scored: hidden by summer canopy in 2022
(`occluded`) or lying in a neighbour's long April shadow in 2020 (`shadowed`).
The cues are combined by rank-averaging (no labels -> no learned weights).

Usage: python detect_damage.py <before_png> <after_png> <osm_json> <out_dir>
"""
import json
import math
import os
import sys

import cv2
import numpy as np

SEARCH = 28            # px (~22 m): the 2020 image is off-nadir, 5-storey roofs appear ~15-20 px north of their footprint
PAD = 6                # context around the footprint used for matching
MIN_AREA_PX = 60       # ~37 m2: below this a building is a handful of pixels
OCCLUDED_GREEN = 0.45  # share of footprint covered by vegetation in 2022
SHADOW_2020 = 0.35     # share of footprint in cast shadow in 2020 (low April sun) -> not comparable


def lonlat_to_px(lon, lat, meta):
    """WGS84 -> pixel of the Web-Mercator mosaic described by its JSON sidecar."""
    n = 2 ** meta["zoom"] * 256
    x = (lon + 180) / 360 * n - meta["tile_x0"] * 256
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n - meta["tile_y0"] * 256
    return x, y


def match_histograms(src, ref):
    """Per-channel CDF matching of src to ref (removes global season/radiometric shift)."""
    out = np.empty_like(src)
    for c in range(3):
        s_vals, s_idx, s_cnt = np.unique(src[..., c].ravel(), return_inverse=True, return_counts=True)
        r_vals, r_cnt = np.unique(ref[..., c].ravel(), return_counts=True)
        s_cdf = np.cumsum(s_cnt) / s_cnt.sum(); r_cdf = np.cumsum(r_cnt) / r_cnt.sum()
        out[..., c] = np.interp(s_cdf, r_cdf, r_vals)[s_idx].reshape(src.shape[:2])
    return out


def edge_alignment(gray, polys, rng=15):
    """Offset (dx,dy) that puts OSM outlines on the strongest image edges."""
    gm = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    outline = np.zeros(gray.shape, np.uint8)
    cv2.polylines(outline, [p.astype(np.int32) for p in polys], True, 1)
    ys, xs = np.nonzero(outline)
    best, arg = -1, (0, 0)
    for dy in range(-rng, rng + 1):
        for dx in range(-rng, rng + 1):
            yy, xx = np.clip(ys + dy, 0, gray.shape[0] - 1), np.clip(xs + dx, 0, gray.shape[1] - 1)
            v = gm[yy, xx].mean()
            if v > best:
                best, arg = v, (dx, dy)
    return arg


def rank01(v):
    """Rank-normalise to [0,1]; NaN stays NaN."""
    v = np.asarray(v, float); out = np.full_like(v, np.nan); ok = ~np.isnan(v)
    out[ok] = (np.argsort(np.argsort(v[ok])) + 0.5) / ok.sum()
    return out


def main() -> None:
    before_p, after_p, osm_p, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(after_p.rsplit(".", 1)[0] + ".json"))
    before, after = cv2.imread(before_p), cv2.imread(after_p)
    gb, ga = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # 1) bring 2020 onto 2022 (separate orthorectifications differ by a few px)
    c = slice(2048, 6144)
    (sx, sy), _ = cv2.phaseCorrelate(np.float32(ga[c, c]), np.float32(gb[c, c]))
    T = np.float32([[1, 0, -sx], [0, 1, -sy]])
    before = cv2.warpAffine(before, T, before.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
    gb_raw = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    before = match_histograms(before, after)
    gb = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)

    # 2) OSM footprints -> pixels, then snap them to the 2022 image edges
    polys, ids, tags = [], [], []
    for e in json.load(open(osm_p))["elements"]:
        if e["type"] != "way" or "geometry" not in e:
            continue
        polys.append(np.float32([lonlat_to_px(g["lon"], g["lat"], meta) for g in e["geometry"]]))
        ids.append(e["id"]); tags.append(e.get("tags", {}).get("building", "yes"))
    odx, ody = edge_alignment(ga, polys)
    polys = [p + [odx, ody] for p in polys]
    print(f"2020->2022 shift ({sx:.1f},{sy:.1f}) px, OSM->2022 shift ({odx},{ody}) px", flush=True)

    lab_b = cv2.cvtColor(before, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_a = cv2.cvtColor(after, cv2.COLOR_BGR2LAB).astype(np.float32)
    lap = lambda g: np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 0.8), cv2.CV_32F))
    hf_b, hf_a = lap(gb), lap(ga)
    # vegetation index on the 2022 image (excess green) to catch canopy-hidden roofs
    bgr = after.astype(np.float32)
    exg = 2 * bgr[..., 1] - bgr[..., 2] - bgr[..., 0]
    H, W = ga.shape

    recs = []
    for poly, oid, tag in zip(polys, ids, tags):
        x0, y0 = np.floor(poly.min(0)).astype(int) - PAD
        x1, y1 = np.ceil(poly.max(0)).astype(int) + PAD
        if x0 - SEARCH < 0 or y0 - SEARCH < 0 or x1 + SEARCH >= W or y1 + SEARCH >= H:
            continue
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(mask, [np.int32(np.round(poly - [x0, y0]))], 1)
        area = int(mask.sum())
        if area < MIN_AREA_PX:
            continue
        inner = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
        if inner.sum() < 20:
            inner = mask > 0
        sa, sb = (slice(y0, y1), slice(x0, x1)), (slice(y0 - SEARCH, y1 + SEARCH), slice(x0 - SEARCH, x1 + SEARCH))

        # struct: best NCC of the 2022 patch inside the 2020 search region (both gray)
        ncc = cv2.matchTemplate(gb[sb], ga[sa], cv2.TM_CCOEFF_NORMED)
        _, best, _, (bx, by) = cv2.minMaxLoc(ncc)
        dx, dy = bx - SEARCH, by - SEARCH
        sb_best = (slice(y0 + dy, y1 + dy), slice(x0 + dx, x1 + dx))  # 2020 roof at its best offset

        la, lb = lab_a[sa][inner], lab_b[sb_best][inner]
        color = float(np.linalg.norm(np.median(la, 0) - np.median(lb, 0)))
        texture = float(np.log((hf_a[sa][inner].mean() + 1) / (hf_b[sb_best][inner].mean() + 1)))
        dark = float((ga[sa][inner] < 50).mean() - (gb[sb_best][inner] < 50).mean())
        green = float((exg[sa][inner] > 25).mean())
        # shadow test on the raw 2020 brightness relative to its surroundings (cast shadows are
        # much darker than the local median, dark roofs are not that far below it)
        loc = gb_raw[max(0, y0 - 40):y1 + 40, max(0, x0 - 40):x1 + 40]
        shadow = float((gb_raw[sb_best][inner] < 0.55 * np.median(loc)).mean())
        cx, cy = poly.mean(0)
        recs.append({"osm_id": oid, "building": tag, "area_m2": area * meta["m_per_px"] ** 2,
                     "cx": float(cx), "cy": float(cy), "struct": 1 - float(best), "color": color,
                     "texture": texture, "dark": dark, "green_2022": green,
                     "shadow_2020": shadow, "occluded": green > OCCLUDED_GREEN,
                     "shadowed": shadow > SHADOW_2020, "poly": np.round(poly, 1).tolist()})

    vis = [r for r in recs if not (r["occluded"] or r["shadowed"])]
    # Weighted rank average: struct is the most specific cue; the rest add evidence.
    R = {k: rank01([r[k] for r in vis]) for k in ("struct", "color", "texture", "dark")}
    for i, r in enumerate(vis):
        r["score"] = float(0.4 * R["struct"][i] + 0.25 * R["color"][i] + 0.2 * R["texture"][i] + 0.15 * R["dark"][i])
    for r in recs:
        r.setdefault("score", None)
    json.dump({"shift_2020_to_2022": [sx, sy], "osm_shift": [odx, ody], "buildings": recs},
              open(f"{out}/damage_scores.json", "w"))
    print(f"{len(recs)} buildings, {sum(r['occluded'] for r in recs)} canopy-occluded, "
          f"{sum(r['shadowed'] and not r['occluded'] for r in recs)} shadowed in 2020, {len(vis)} scored", flush=True)


if __name__ == "__main__":
    main()
