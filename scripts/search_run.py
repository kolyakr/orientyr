"""M1-05: run the search strategy over a frame sequence and report modes, accuracy and time.

Each frame goes through dronenav.search.Searcher (whole map first, then windows around the prediction, whole
map again when a window fails). Truth comes from probe.json (task crops) or truth.jsonl (an Irpin flight).
``--miss N`` replaces frame N with a black one, to show the recovery.

Usage:
    python scripts/search_run.py --task data/hackatons-data --probe experiments/data_probe/results/probe.json
    python scripts/search_run.py --irpin data/irpin/flights/mild/2025 --irpin-map data/irpin/maps/irpin_2025.png \
        --miss 200
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from dronenav.config import add_shared_args, init_cli
from dronenav.matching import RootSiftMatcher
from dronenav.search import Searcher, SearchMode


def task_frames(task_dir: Path, probe: Path):
    truth = {r["i"]: r for r in json.loads(probe.read_text())["rows"]}
    for i in sorted(truth):
        yield cv2.imread(str(task_dir / "crops" / f"crop_{i}.png")), (truth[i]["cx"], truth[i]["cy"])


def irpin_frames(flight: Path):
    meta = json.loads((flight / "meta.json").read_text())
    truth = {r["i"]: r for r in map(json.loads, (flight / "truth.jsonl").read_text().splitlines()) if r}
    for k in meta["keyframes"]:
        yield cv2.imread(str(flight / k["file"])), tuple(truth[k["i"]]["center_map_px"])


def run(searcher: Searcher, frames, miss: int | None, map_area: int) -> None:
    modes, err, ms, area = Counter(), [], {m: [] for m in SearchMode}, []
    lost, sequence = 0, []
    for n, (frame, (cx, cy)) in enumerate(frames):
        if n == miss:
            frame = np.zeros_like(frame)
        r = searcher.locate(n, frame)
        modes[r.mode] += 1
        ms[r.mode].append(r.ms)
        sequence.append(r.mode.value[0].upper())
        if r.window is not None:
            x0, y0, x1, y1 = r.window
            area.append((x1 - x0) * (y1 - y0) / map_area)
        if r.fix is None or not r.fix.accepted:
            lost += 1
            continue
        if n != miss:
            err.append(math.hypot(r.fix.x - cx, r.fix.y - cy))
    e = np.array(err)
    print(f"frames {n + 1}, accepted {n + 1 - lost}, misses {lost}")
    print("modes:", ", ".join(f"{m.value} {modes[m]}" for m in SearchMode))
    print("sequence (G global, W window, F fallback):", "".join(sequence))
    print(f"error px vs truth: median {np.median(e):.2f}, p95 {np.percentile(e, 95):.2f}, max {e.max():.2f}")
    print("ms median:", ", ".join(f"{m.value} {np.median(v):.0f}" for m, v in ms.items() if v))
    if area:
        print(f"window area: median {100 * np.median(area):.1f}% of the map")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", type=Path, help="folder with global_map.png and crops/")
    ap.add_argument("--probe", type=Path, help="probe.json with the task crops' truth")
    ap.add_argument("--irpin", type=Path, help="an Irpin flight folder (meta.json, truth.jsonl, frames/)")
    ap.add_argument("--irpin-map", type=Path, help="the map that flight's truth refers to")
    ap.add_argument("--miss", type=int, help="replace this frame with a black one (forced miss)")
    ap.add_argument("--cache", type=Path, default=Path("outputs/cache"))
    add_shared_args(ap)
    args = ap.parse_args()
    # Picked and logged like in every matching script; RootSIFT itself runs on the CPU, a torch matcher
    # plugged in here later takes this device.
    init_cli(args)
    if args.task:
        mp, frames = cv2.imread(str(args.task / "global_map.png")), task_frames(args.task, args.probe)
    elif args.irpin:
        mp, frames = cv2.imread(str(args.irpin_map)), irpin_frames(args.irpin)
    else:
        ap.error("give --task or --irpin")
    run(Searcher(RootSiftMatcher(mp, cache_dir=args.cache), mp), frames, args.miss, mp.shape[0] * mp.shape[1])


if __name__ == "__main__":
    main()
