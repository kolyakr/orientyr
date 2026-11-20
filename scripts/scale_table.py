"""M1-03: how much to shrink frames before matching. Prints a markdown table of factor vs inliers vs ms.

For each downscale factor (and for the rule in dronenav.scale), every frame is matched with RootSIFT, a
homography is fitted, and the fix is scored against the truth. Two datasets:

- the task crops (map + 50 crops; truth = experiments/data_probe/results/probe.json), matched to the whole map;
- one Irpin flight's keyframes (truth.jsonl), each matched to a window of the map around the true position.
  The window keeps the search out of the question (that is M1-05): only the scale changes between rows.

Usage:
    python scripts/scale_table.py --task data/hackatons-data --probe experiments/data_probe/results/probe.json \
        --irpin data/irpin/flights/mild/2025 --irpin-map data/irpin/maps/irpin_2025.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from dronenav.geometry import localize
from dronenav.matching import RootSiftMatcher
from dronenav.scale import ScaleConfig, downscale_factor, match_scaled

FACTORS = (1.0, 0.75, 0.5, 0.35, 0.25)
GOOD_PX = 10.0          # a fix this close to the truth counts as localised (well above the ~0.5 px typical error)


def task_items(task_dir: Path, probe: Path):
    """(frame, true centre, map window, window offset) for the 50 task crops; the window is the whole map."""
    mp = cv2.imread(str(task_dir / "global_map.png"))
    truth = {r["i"]: r for r in json.loads(probe.read_text())["rows"]}
    for i in sorted(truth):
        crop = cv2.imread(str(task_dir / "crops" / f"crop_{i}.png"))
        yield crop, (truth[i]["cx"], truth[i]["cy"]), mp, (0, 0)


def irpin_items(flight: Path, map_path: Path, n: int, half: int = 800):
    """n evenly spaced keyframes of an Irpin flight, each with a (2*half)^2 map window around its truth."""
    mp = cv2.imread(str(map_path))
    meta = json.loads((flight / "meta.json").read_text())
    truth = {r["i"]: r for r in map(json.loads, (flight / "truth.jsonl").read_text().splitlines()) if r}
    keys = meta["keyframes"]
    for k in keys[:: max(1, len(keys) // n)][:n]:
        cx, cy = truth[k["i"]]["center_map_px"]
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        yield cv2.imread(str(flight / k["file"])), (cx, cy), mp[y0:y0 + 2 * half, x0:x0 + 2 * half], (x0, y0)


def measure(items: list, factors, matcher_for) -> list[dict]:
    """One row per factor ("rule" = each frame at downscale_factor of its own true-ish scale)."""
    rows = []
    for f in factors:
        ok, inl, ms, err, scales = 0, [], [], [], []
        for frame, (cx, cy), win, off in items:
            m = matcher_for(win)
            fac = f
            if f == "rule":
                # the tracker's factor: from the scale of a fix, here the full-resolution fix of this frame
                ref = localize(0, match_scaled(m, frame, win, 1.0, map_offset=off), frame.shape[1::-1])
                fac = downscale_factor(ref.scale) if ref is not None and ref.accepted else 1.0
            res = match_scaled(m, frame, win, fac, map_offset=off)
            fix = localize(0, res, (frame.shape[1], frame.shape[0]))
            ms.append(res.ms)
            scales.append(fac)
            if fix is not None and fix.accepted and math.hypot(fix.x - cx, fix.y - cy) < GOOD_PX:
                ok += 1
                inl.append(fix.inliers)
                err.append(math.hypot(fix.x - cx, fix.y - cy))
        rows.append({"factor": f, "mean_factor": float(np.mean(scales)), "ok": ok, "n": len(items),
                     "inliers": float(np.median(inl)) if inl else 0.0,
                     "err": float(np.median(err)) if err else float("nan"), "ms": float(np.median(ms))})
    return rows


def table(name: str, rows: list[dict]) -> str:
    out = [f"**{name}**", "", "| factor | localised | inliers (median) | error px (median) | ms (median) |",
           "|---|---|---|---|---|"]
    for r in rows:
        f = f"rule ({r['mean_factor']:.2f})" if r["factor"] == "rule" else f"{r['factor']:.2f}"
        out.append(f"| {f} | {r['ok']}/{r['n']} | {r['inliers']:.0f} | {r['err']:.2f} | {r['ms']:.0f} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", type=Path, help="folder with global_map.png and crops/")
    ap.add_argument("--probe", type=Path, help="probe.json with the task crops' truth")
    ap.add_argument("--irpin", type=Path, help="an Irpin flight folder (meta.json, truth.jsonl, frames/)")
    ap.add_argument("--irpin-map", type=Path, help="the map that flight's truth refers to")
    ap.add_argument("--n", type=int, default=50, help="Irpin keyframes to use")
    ap.add_argument("--cache", type=Path, default=Path("outputs/cache"))
    args = ap.parse_args()
    factors = (*FACTORS, "rule")
    print(f"Rule: factor = scale / {ScaleConfig().target_scale}, clipped to [{ScaleConfig().min_factor}, 1].\n")
    if args.task:
        items = list(task_items(args.task, args.probe))
        m = RootSiftMatcher(items[0][2], cache_dir=args.cache)
        print(table("Task crops (50, whole map)", measure(items, factors, lambda win: m)), "\n")
    if args.irpin:
        items = list(irpin_items(args.irpin, args.irpin_map, args.n))
        full = cv2.imread(str(args.irpin_map))
        m = RootSiftMatcher(full, cache_dir=args.cache)      # windows reuse the full map's cached features
        print(table(f"Irpin {args.irpin.parent.name}/{args.irpin.name} keyframes ({len(items)}, 1600 px window)",
                    measure(items, factors, lambda win: m)))


if __name__ == "__main__":
    main()
