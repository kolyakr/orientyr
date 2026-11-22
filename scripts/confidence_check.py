"""M4-08: check the per-fix confidence and covariance against exact truth, and draw the reliability plot.

Every Irpin flight (rendered from the 2025 imagery) is matched against the 2025 map and the older, cross-season
maps, frame by frame through dronenav.search.Searcher. Each accepted fix is compared with the true frame centre on
that map (the 2025 truth minus the map's alignment shift). The fixes are cached, so replotting is instant.

Prints the error by confidence fifth and the coverage of the σ circles; writes ``reliability.png``:
left, the error of every fix against its confidence, with the median per fifth; right, the share of fixes whose
error is within k σ, against what an honest 2D Gaussian gives (1 - exp(-k²/2)).

Usage:
    python scripts/confidence_check.py --irpin data/irpin --out outputs/confidence
    python scripts/confidence_check.py --irpin data/irpin --out outputs/confidence --keyframes 40 --maps 2025 2022
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from dronenav.confidence import fix_confidence, fix_sigma
from dronenav.config import add_shared_args, init_cli
from dronenav.matching import RootSiftMatcher
from dronenav.search import Searcher
from dronenav.types import Fix

FLIGHTS = ("mild", "harsh", "mild_30mps", "harsh_30mps")


def collect(irpin: Path, maps: list[str], keyframes: int, cache_dir: Path) -> list[dict]:
    """One row per keyframe x map: the fix's evidence and its true centre on that map."""
    rows = []
    for year in maps:
        mp = cv2.imread(str(irpin / "maps" / f"irpin_{year}.png"))
        if mp is None:
            raise SystemExit(f"cannot read map {irpin / 'maps' / f'irpin_{year}.png'}")
        sidecar = json.loads((irpin / "maps" / f"irpin_{year}.json").read_text())
        dx, dy = sidecar["map"]["alignment_to_2025"]["shift_px"]
        matcher = RootSiftMatcher(mp, cache_dir=cache_dir)
        for flight in FLIGHTS:
            f = irpin / "flights" / flight / "2025"
            meta = json.loads((f / "meta.json").read_text())
            truth = {r["i"]: r for r in map(json.loads, (f / "truth.jsonl").read_text().splitlines()) if r}
            searcher, t0 = Searcher(matcher, mp), time.time()
            for n, key in enumerate(meta["keyframes"][:keyframes]):
                fix = searcher.locate(n, cv2.imread(str(f / key["file"]))).fix
                cx, cy = truth[key["i"]]["center_map_px"]
                row = {"map": year, "flight": flight, "n": n, "tx": cx - dx, "ty": cy - dy, "accepted": False}
                if fix is not None:
                    row.update(x=fix.x, y=fix.y, inliers=fix.inliers, ratio=fix.inlier_ratio, reproj=fix.reproj_px,
                               accepted=fix.accepted)
                rows.append(row)
            print(f"  {year} {flight}: {time.time() - t0:.0f} s", flush=True)
    return rows


def as_fix(r: dict) -> Fix:
    """A stand-in Fix carrying only what confidence and σ read."""
    return Fix(idx=r["n"], H=np.eye(3), x=r["x"], y=r["y"], heading_deg=0.0, scale=1.0,
               footprint=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), inliers=r["inliers"],
               inlier_ratio=r["ratio"], reproj_px=r["reproj"], accepted=True)


def report(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    acc = [r for r in rows if r["accepted"]]
    conf = np.array([fix_confidence(as_fix(r)) for r in acc])
    sigma = np.array([fix_sigma(as_fix(r)) for r in acc])
    err = np.array([math.hypot(r["x"] - r["tx"], r["y"] - r["ty"]) for r in acc])
    maps = [r["map"] for r in acc]
    for year in sorted({r["map"] for r in rows}, reverse=True):
        n = sum(r["map"] == year for r in rows)
        print(f"map {year}: {sum(m == year for m in maps)} of {n} keyframes accepted")
    if len(acc) < 5:
        raise SystemExit("too few accepted fixes to report on")
    print(f"\n{len(acc)} accepted fixes. Error by confidence fifth (low -> high):")
    print(f"{'fifth':>5} {'confidence':>14} {'median px':>10} {'p90 px':>8}")
    order = np.argsort(conf)
    for q, idx in enumerate(np.array_split(order, 5)):
        print(f"{q + 1:>5} {conf[idx].min():6.3f}-{conf[idx].max():5.3f} {np.median(err[idx]):10.2f} "
              f"{np.percentile(err[idx], 90):8.2f}")
    for k, want in ((1.0, 0.393), (2.0, 0.865), (2.448, 0.95)):
        print(f"within {k:.3g} σ: {np.mean(err < k * sigma):.1%} (honest Gaussian: {want:.1%})")
    return conf, err, sigma, maps


def plot(conf: np.ndarray, err: np.ndarray, sigma: np.ndarray, maps: list[str], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colours = {"2025": "#2a6fdb", "2022": "#e07b39", "2017": "#7a7a7a"}
    for year in sorted(set(maps), reverse=True):
        m = np.array([y == year for y in maps])
        a.scatter(conf[m], err[m], s=10, alpha=0.55, color=colours.get(year, "#555"), label=f"{year} map")
    order = np.argsort(conf)
    fifths = np.array_split(order, 5)
    a.plot([np.median(conf[i]) for i in fifths], [np.median(err[i]) for i in fifths], "k-o", lw=2,
           label="median per fifth")
    a.set(yscale="log", xlabel="confidence", ylabel="position error, map px",
          title="Low confidence, larger error")
    a.legend(frameon=False, fontsize=8)
    k = np.linspace(0, 4, 200)
    b.plot(k, 1 - np.exp(-k ** 2 / 2), "k--", lw=1, label="honest 2D Gaussian")
    b.plot(k, [(err / sigma < x).mean() for x in k], color="#2a6fdb", lw=2, label="Irpin fixes")
    b.axvline(2.448, color="#999", lw=0.8)
    b.set(xlabel="error / σ", ylabel="share of fixes within", title="Covariance coverage", ylim=(0, 1.01))
    b.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--irpin", type=Path, required=True, help="data/irpin: maps/ and flights/")
    p.add_argument("--out", type=Path, default=Path("outputs/confidence"))
    p.add_argument("--maps", nargs="+", default=["2025", "2022", "2017"])
    p.add_argument("--keyframes", type=int, default=80, help="first N keyframes of each flight")
    p.add_argument("--refresh", action="store_true", help="re-run the matching even if fixes.json exists")
    add_shared_args(p)
    a = p.parse_args()
    # Picked and logged like in every matching script; RootSIFT runs on the CPU, a torch matcher takes this device.
    init_cli(a)
    a.out.mkdir(parents=True, exist_ok=True)
    cache = a.out / "fixes.json"
    key = {"maps": a.maps, "keyframes": a.keyframes}
    if cache.exists() and not a.refresh and json.loads(cache.read_text()).get("key") == key:
        rows = json.loads(cache.read_text())["rows"]
    else:
        print("matching (cached in fixes.json afterwards)...")
        rows = collect(a.irpin, a.maps, a.keyframes, Path("outputs/cache"))
        cache.write_text(json.dumps({"key": key, "rows": rows}))
    plot(*report(rows), a.out / "reliability.png")


if __name__ == "__main__":
    main()
