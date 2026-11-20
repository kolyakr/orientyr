"""Remove colour/haze seams between source captures of a Wayback mosaic.

A Wayback map is stitched from several satellite captures (different years, seasons, haze).
Their geometry agrees to ~1-2 px, but each patch has its own tone curve, which makes
visible seams. We fit one monotone tone curve per patch and channel (quantile mapping)
onto a single-capture reference of the same ground, then blend the patch curves with
Gaussian weights so the transition across a seam is smooth.

Only the global colour response of each patch changes - no pixel content is taken from
the reference, so the map keeps its own epoch (buildings, roads, destruction).

Usage: python harmonize.py <map_png> <labels_npy> <ref_png> <ref_labels_npy> <out_png>
  labels_npy:   capture-patch raster at 1/8 resolution (from coverage.py)
  ref_labels:   reference patch raster; only its majority capture is used for fitting
"""
import json
import shutil
import sys

import cv2
import numpy as np
from scipy import ndimage

DOWN = 8               # label raster scale
MIN_FRAC = 0.005       # smaller patches inherit the curve of the nearest big patch
BLEND_SIGMA = 8        # in label px (= 64 map px, ~100 m): width of the seam transition
STRIP = 1024


def fill_small(lab):
    """Relabel tiny / unknown patches with the nearest large patch."""
    keep = [k for k in np.unique(lab) if k > 0 and (lab == k).mean() >= MIN_FRAC]
    bad = ~np.isin(lab, keep)
    _, (iy, ix) = ndimage.distance_transform_edt(bad, return_indices=True)
    return lab[iy, ix], keep


def quantile_lut(src, ref):
    """256-entry monotone LUT mapping the src distribution onto the ref distribution."""
    q = np.linspace(0, 100, 257)
    sq, rq = np.percentile(src, q), np.percentile(ref, q)
    sq = np.maximum.accumulate(sq + np.arange(257) * 1e-6)  # strictly increasing for interp
    return np.clip(np.interp(np.arange(256), sq, rq), 0, 255).astype(np.float32)


def main():
    map_p, lab_p, ref_p, rlab_p, out_p = sys.argv[1:6]
    img, ref = cv2.imread(map_p), cv2.imread(ref_p)
    lab, keep = fill_small(np.load(lab_p))
    rlab = np.load(rlab_p)
    ref_ok = rlab == np.bincount(rlab.ravel()).argmax()        # reference's own epoch only
    # cloud shadows in the reference: much darker than their surroundings at ~300 m scale
    rs = cv2.resize(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), lab.shape[::-1], interpolation=cv2.INTER_AREA).astype(np.float32)
    ref_ok &= cv2.GaussianBlur(rs, (0, 0), 3) > 0.75 * cv2.GaussianBlur(rs, (0, 0), 40)

    small_img = cv2.resize(img, lab.shape[::-1], interpolation=cv2.INTER_AREA)
    small_ref = cv2.resize(ref, lab.shape[::-1], interpolation=cv2.INTER_AREA)
    # fit on full-res pixels sampled inside each patch (and inside the valid reference)
    rng = np.random.default_rng(0)
    luts, report = {}, {}
    for k in keep:
        ys, xs = np.nonzero((lab == k) & ref_ok)
        sel = rng.choice(len(ys), min(len(ys), 200_000), replace=False)
        fy = ys[sel] * DOWN + rng.integers(0, DOWN, len(sel)); fx = xs[sel] * DOWN + rng.integers(0, DOWN, len(sel))
        luts[k] = np.stack([quantile_lut(img[fy, fx, c], ref[fy, fx, c]) for c in range(3)])
        report[int(k)] = {"frac": float((lab == k).mean()),
                          "mean_before": small_img[lab == k].mean(0).round(1).tolist(),
                          "mean_ref": small_ref[lab == k].mean(0).round(1).tolist()}

    # smooth per-patch weights -> each pixel gets a weighted mix of the patch curves
    W = {k: cv2.GaussianBlur((lab == k).astype(np.float32), (0, 0), BLEND_SIGMA) for k in keep}
    tot = sum(W.values())
    out = np.empty_like(img)
    H = img.shape[0]
    for y0 in range(0, H, STRIP):
        y1 = min(H, y0 + STRIP)
        blk = img[y0:y1]; acc = np.zeros(blk.shape, np.float32)
        for k in keep:
            w = cv2.resize(W[k] / tot, (img.shape[1], lab.shape[0] * DOWN), interpolation=cv2.INTER_LINEAR)[y0:y1, :, None]
            mapped = np.stack([luts[k][c][blk[..., c]] for c in range(3)], -1)
            acc += w * mapped
        out[y0:y1] = np.clip(acc + 0.5, 0, 255).astype(np.uint8)
    cv2.imwrite(out_p, out)
    side = map_p.rsplit(".", 1)[0] + ".json"
    meta = json.load(open(side)); meta["harmonized"] = {"reference": ref_p, "patches": report}
    json.dump(meta, open(out_p.rsplit(".", 1)[0] + ".json", "w"), indent=1)
    print("patches:", {k: f"{v['frac']:.0%}" for k, v in report.items()})


if __name__ == "__main__":
    main()
