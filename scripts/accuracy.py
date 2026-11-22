"""How far a run's positions are from the truth: the error per frame, then the summary.

    python scripts/accuracy.py --run outputs/run1 --truth experiments/data_probe/results/probe.json
    python scripts/accuracy.py --map data/irpin/maps/irpin_2025.png --frames data/irpin/flights/mild/2025/frames \
        --run outputs/mild2025 --truth data/irpin/flights/mild/2025

--truth is an Irpin flight's truth.jsonl (or its folder): exact; or the task's probe.json: a full-resolution fit,
good for catching regressions. With --map and --frames the frames are localized into --run first (run.json and
run.jsonl only: no image, no video). The numbers are also written to accuracy.json in --run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2

from dronenav.accuracy import WITHIN_PX, load_truth, score
from dronenav.config import add_shared_args, init_cli
from dronenav.pipeline import MATCHERS, OUTPUTS, NothingLocalized, localize_run
from dronenav.telemetry import TelemetryError, read_run


def _px(v: float | None) -> str:
    return "-" if v is None or math.isnan(v) else f"{v:.2f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path, help="run folder (run.jsonl); written first with --map/--frames")
    ap.add_argument("--truth", required=True, type=Path, help="truth.jsonl, a flight folder holding one, or probe.json")
    ap.add_argument("--map", help="localize first: the satellite map")
    ap.add_argument("--frames", help="localize first: the folder of frames")
    ap.add_argument("--fps", type=float, default=None, help="as in run.py: t = number in the frame name / fps")
    ap.add_argument("--matcher", default="sift", help=f"as in run.py (available: {', '.join(sorted(MATCHERS))})")
    ap.add_argument("--cache", default="outputs/cache", help="map feature cache; 'none' for no cache")
    ap.add_argument("--within", type=float, default=WITHIN_PX,
                    help=f"a fix within this many map px of the truth is correct (default {WITHIN_PX:g})")
    add_shared_args(ap)
    a = ap.parse_args(argv)
    if (a.map is None) != (a.frames is None):
        ap.error("--map and --frames go together (to localize first), or neither (to score an existing run)")
    device = init_cli(a)      # logging and the device, as in every matching script (M0-07); the device is for --map
    try:
        truth = load_truth(a.truth)        # before localizing: a wrong truth path must not cost a whole run
        if a.map is not None:
            # run.json and run.jsonl would be replaced, run.py's image and video left behind: a folder that lies
            stale = [n for n in OUTPUTS if n not in ("run.json", "run.jsonl") and (a.run / n).exists()]
            if stale:
                raise ValueError(f"{a.run} holds a rendered run ({', '.join(stale)}) that localizing would leave "
                                 "stale: score it without --map/--frames, or localize into another folder")
            localize_run(a.map, a.frames, a.run, matcher=a.matcher, fps=a.fps,
                         cache_dir=None if a.cache.lower() == "none" else a.cache, device=device)
        acc = score(read_run(a.run / "run.jsonl"), truth, within_px=a.within)
    except NothingLocalized as e:
        print(f"accuracy: error: {e}", file=sys.stderr)
        return 1
    except (OSError, ValueError, TelemetryError, cv2.error) as e:
        print(f"accuracy: error: {e}", file=sys.stderr)
        return 2

    print(f"{'frame':>8}  {'state':<12} {'fix px':>8} {'track px':>9}")
    for f in acc.frames:
        print(f"{f.idx:>8}  {f.state:<12} {_px(f.fix_px):>8} {_px(f.track_px):>9}")
    s = acc.summary()
    (a.run / "accuracy.json").write_text(json.dumps(
        {"truth": str(a.truth), "summary": s, "frames": [
            {"idx": f.idx, "state": f.state, "fix_px": None if f.fix_px is None else round(f.fix_px, 3),
             "track_px": round(f.track_px, 3)} for f in acc.frames]}, indent=1) + "\n")
    print(f"accuracy: localized {acc.localized}/{acc.n}, correct {acc.correct}, wrong {acc.wrong} "
          f"(> {acc.within_px:g} px); error px median {_px(acc.median_px)}, p95 {_px(acc.p95_px)}, "
          f"max {_px(acc.max_px)}" + (f"; {acc.missing_truth} run frames without truth" if acc.missing_truth else "")
          + (f"; {acc.not_in_run} truth frames not in the run" if acc.not_in_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
