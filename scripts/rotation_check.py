"""M2-05: success by heading on the Irpin flights, which turn through every heading.

Each flight's keyframes go through dronenav.search.Searcher (RootSIFT) on the map they were rendered from, and the
result is binned by the frame's true heading (dronenav.rotation). Writes to --out:
- ``rotation.json``: per flight, one row per heading bin (frames, localized, wrong, success, errors);
- ``rotation.md``: the same as a table, for the README notes;
- ``rotation.png``: success per heading bin, one bar group per flight.

Usage:
    python scripts/rotation_check.py --irpin data/irpin --out outputs/rotation
    python scripts/rotation_check.py --irpin data/irpin --flights mild/2025 harsh/2025 --out outputs/rotation
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dronenav.config import add_shared_args, init_cli
from dronenav.rotation import HeadingBin, check_flight


def table_md(results: dict[str, list[HeadingBin]]) -> str:
    """One row per flight x heading bin."""
    lines = ["| flight | heading | frames | localized | wrong | success | median err px | max heading err ° |",
             "|---|---|---|---|---|---|---|---|"]

    def f(v, fmt):
        return "–" if v is None else format(v, fmt)

    for name, bins in results.items():
        for b in bins:
            lines.append(f"| {name} | {b.lo_deg:.0f}–{b.hi_deg:.0f}° | {b.frames} | {b.localized} | {b.wrong} | "
                         f"{f(b.success and 100 * b.success, '.0f')}% | {f(b.median_err_px, '.2f')} | "
                         f"{f(b.max_heading_err_deg, '.2f')} |")
    return "\n".join(lines) + "\n"


def plot(results: dict[str, list[HeadingBin]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.5))
    width = 0.8 / max(1, len(results))
    for k, (name, bins) in enumerate(results.items()):
        xs = [i + k * width for i in range(len(bins))]
        bars = ax.bar(xs, [100 * (b.success or 0) for b in bins], width, label=name)
        ax.bar_label(bars, [f"n={b.frames}" for b in bins], fontsize=7)   # the bins are uneven: show the evidence
    first = next(iter(results.values()))
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(first))],
                  [f"{b.lo_deg:.0f}–{b.hi_deg:.0f}°" for b in first])
    ax.set_ylim(0, 112)
    ax.set_ylabel("localized, %")
    ax.set_xlabel("true heading (clockwise from north)")
    ax.legend(fontsize=8, ncol=len(results), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--irpin", type=Path, required=True, help="the Irpin data folder (maps/, flights/)")
    ap.add_argument("--flights", nargs="+", default=["mild/2025", "harsh/2025"],
                    help="flights as <preset>/<year>; each is matched on the map of its year")
    ap.add_argument("--bins", type=int, default=8, help="heading bins over 360°")
    ap.add_argument("--out", type=Path, default=Path("outputs/rotation"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/cache"))
    add_shared_args(ap)
    args = ap.parse_args()
    init_cli(args)   # RootSIFT runs on the CPU; the device is picked and logged like in every matching script
    results = {}
    for name in args.flights:
        year = name.split("/")[-1]
        results[name] = check_flight(args.irpin / "flights" / name, args.irpin / "maps" / f"irpin_{year}.png",
                                     cache_dir=args.cache, n_bins=args.bins)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rotation.json").write_text(json.dumps({k: [asdict(b) for b in v] for k, v in results.items()},
                                                       indent=1))
    md = table_md(results)
    (args.out / "rotation.md").write_text(md)
    plot(results, args.out / "rotation.png")
    print(md, end="")


if __name__ == "__main__":
    main()
