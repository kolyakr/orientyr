"""Town-centred, seam-free tiles that exist in all three epochs, plus the full 2025 map's patch mask.

For each town we search for a square (3 km down to 1.56 km) that lies inside ONE source
capture in the pre-war map and ONE in the war map (32 px clear of seams), maximising the
number of that town's UNOSAT destroyed/severe labels inside it (score = labels x sqrt(size)). The 2025 map is 96 % a single
capture; its cloud-fill patches (2022 imagery) are not avoided but recorded as a mask, so
evaluation can ignore or flag them.

Usage: python town_tiles.py <data_dir>
  reads  maps/cluster_*.png, coverage/current_*.npy, unosat_labels_cluster.geojson
  writes towns_seamfree/<town>_{prewar,war,rebuilt}.png/.json, rebuilt_valid_mask.png
"""
import json
import math
import os
import sys

import cv2
import numpy as np

DOWN = 8
SEAM_GAP = 4                         # label px = 32 map px
SIZES = [1984, 1792, 1600, 1408, 1280, 1024]  # map px: 3.0 .. 1.56 km at 1.52 m/px (frame = 640 px)
TOWNS = ["Irpin", "Hostomel", "Moschun", "Bucha"]
EPOCHS = {"prewar": ("cluster_2020_prewar", "15423"), "war": ("cluster_2022_war", "45134"),
          "rebuilt": ("cluster_2025_rebuilt", "20512")}


def lonlat_to_px(lon, lat, meta):
    n = 2 ** meta["zoom"] * 256
    return ((lon + 180) / 360 * n - meta["tile_x0"] * 256,
            (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n - meta["tile_y0"] * 256)


def single_capture_ok(lab, s):
    """ok[y, x] = square of side s (label px) at top-left (x, y) lies inside one eroded capture patch."""
    ok = np.zeros((lab.shape[0] - s + 1, lab.shape[1] - s + 1), bool)
    for k in np.unique(lab):
        if k == 0:
            continue
        m = cv2.erode((lab == k).astype(np.uint8), np.ones((2 * SEAM_GAP + 1,) * 2, np.uint8))
        ii = cv2.integral(m)
        area = ii[s:, s:] - ii[:-s, s:] - ii[s:, :-s] + ii[:-s, :-s]
        ok |= area == s * s
    return ok


def main():
    D = sys.argv[1]
    out = f"{D}/towns_seamfree"; os.makedirs(out, exist_ok=True)
    meta = json.load(open(f"{D}/maps/cluster_2022_war.json"))
    labs = {e: np.load(f"{D}/coverage/current_{rel}.npy") for e, (_, rel) in EPOCHS.items()}
    feats = json.load(open(f"{D}/unosat_labels_cluster.geojson"))["features"]
    pts = {t: np.array([lonlat_to_px(*f["geometry"]["coordinates"][:2], meta) for f in feats
                        if f["properties"]["city"] == t and f["properties"]["damage"] in (1, 2)]) for t in TOWNS}

    # 2025: majority capture is valid, cloud-fill patches (other captures) are masked
    rl = labs["rebuilt"]; valid = rl == np.bincount(rl.ravel()).argmax()
    big = cv2.resize(valid.astype(np.uint8) * 255, (rl.shape[1] * DOWN, rl.shape[0] * DOWN), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"{D}/maps/cluster_2025_rebuilt_valid_mask.png", big)
    print(f"2025 map: {valid.mean():.1%} valid 2025 imagery, mask -> maps/cluster_2025_rebuilt_valid_mask.png")

    imgs = {e: cv2.imread(f"{D}/maps/{f}.png") for e, (f, _) in EPOCHS.items()}
    summary = []
    for town in TOWNS:
        P = pts[town] / DOWN
        best = None
        for S in SIZES:
            s = S // DOWN
            ok = single_capture_ok(labs["prewar"], s) & single_capture_ok(labs["war"], s)
            # label count per top-left via a 2D histogram + integral image
            H = np.zeros(labs["war"].shape, np.float32)
            for x, y in P.astype(int):
                if 0 <= y < H.shape[0] and 0 <= x < H.shape[1]:
                    H[y, x] += 1
            ii = cv2.integral(H)
            cnt = ii[s:, s:] - ii[:-s, s:] - ii[s:, :-s] + ii[:-s, :-s]
            cnt = np.where(ok, cnt, -1)
            y, x = np.unravel_index(cnt.argmax(), cnt.shape)
            # trade size against damage content: a smaller tile wins only with clearly more labels
            score = cnt[y, x] * math.sqrt(S / SIZES[0])
            if best is None or score > best[4]:
                best = (int(x) * DOWN, int(y) * DOWN, S, int(cnt[y, x]), score)
        X, Y, S, n, _ = best
        rec = {"town": town, "offset_px": [X, Y], "size_px": S, "km": round(S * meta["m_per_px"] / 1000, 2),
               "labels_inside": n, "labels_town": len(P),
               "rebuilt_valid_frac": float(valid[Y // DOWN:(Y + S) // DOWN, X // DOWN:(X + S) // DOWN].mean())}
        for e in EPOCHS:
            l = labs[e][Y // DOWN:(Y + S) // DOWN, X // DOWN:(X + S) // DOWN]
            rec[f"{e}_patch_label"] = int(np.bincount(l.ravel()).argmax())
            cv2.imwrite(f"{out}/{town.lower()}_{e}.png", imgs[e][Y:Y + S, X:X + S])
            json.dump(dict(json.load(open(f"{D}/maps/{EPOCHS[e][0]}.json")), offset_px=[X, Y], size_px=[S, S], town=town),
                      open(f"{out}/{town.lower()}_{e}.json", "w"), indent=1)
        summary.append(rec)
        print(f"{town:9s} {rec['km']} km square at ({X},{Y}): {n}/{len(P)} labels, 2025 valid {rec['rebuilt_valid_frac']:.0%}")
    json.dump(summary, open(f"{out}/towns.json", "w"), indent=1)


if __name__ == "__main__":
    main()
