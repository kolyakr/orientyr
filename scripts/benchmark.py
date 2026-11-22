"""The benchmark (M6-03): every Irpin scenario x preset x speed x matcher x tracker, scored against exact truth.

    python scripts/benchmark.py --data data --out outputs/bench                       # everything (long: ~30 runs)
    python scripts/benchmark.py --scenarios C E --speeds 15 --frames 100 --out outputs/bench_quick
    python scripts/benchmark.py --out outputs/bench --reuse                           # the table again, no matching

Each combination is one plain localisation run (dronenav.pipeline.localize_run, no rendering) into
``<out>/runs/<name>``; its records are scored by dronenav.eval.metrics against the flight's truth on that map
(dronenav.eval.irpin). Writes ``<out>/bench.json`` (config and one row per combination) and ``<out>/bench.md``
(the table, every cell "value (95 % interval)"). The numbers repeat exactly for the same seed (``--seed``: the
bootstrap's); time per frame is the machine's and kept apart. A combination whose flight or map is not downloaded
is reported as "missing data", not a failure. Exit code 0, or 1 if a run failed, 2 on bad arguments.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from dronenav.config import add_shared_args, init_cli
from dronenav.eval import irpin
from dronenav.eval.metrics import frame_metrics, missed, summarize, timing
from dronenav.io import FolderSource
from dronenav.pipeline import MATCHERS, TRACKERS, NothingLocalized, localize_run
from dronenav.telemetry import read_run

log = logging.getLogger("benchmark")

COLUMNS = {   # metric -> table header; percentages and metres, one decimal each
    "localised_pct": "Localised %", "wrong_fix_pct": "Wrong fix %", "coverage_pct": "Coverage %",
    "median_m": "Median m", "rmse_m": "RMSE m", "p95_m": "p95 m", "unc_coverage_pct": "Truth in 95 % region %",
}


def _frames_in(flight: Path, limit: int | None, run_dir: Path) -> Path:
    """The frames folder to localise: the flight's own, or a folder of links to its first ``limit`` keyframes
    (with meta.json beside them, so the camera and fps are the same as in a full run)."""
    if not limit:
        return flight / "frames"
    keys = json.loads((flight / "meta.json").read_text())["keyframes"][:limit]
    sub = run_dir / "frames_in"
    if sub.exists():
        shutil.rmtree(sub)
    sub.mkdir(parents=True)
    shutil.copy(flight / "meta.json", sub / "meta.json")
    for k in keys:
        (sub / Path(k["file"]).name).symlink_to((flight / k["file"]).resolve())
    return sub


def _one(data: Path, out: Path, cache: Path, scenario: str, preset: str, speed: int, matcher: str, tracker: str,
         a, device: str | None) -> dict:
    s = irpin.SCENARIOS[scenario]
    row = {"scenario": scenario, "preset": preset, "speed": speed, "matcher": matcher, "tracker": tracker}
    flight, map_png = irpin.flight_dir(data, preset, speed, s.flight), irpin.map_path(data, s.map)
    missing = [str(p) for p in (flight / "frames", flight / "truth.jsonl", map_png) if not p.exists()]
    if missing:
        return {**row, "status": "missing data", "missing": missing}
    run_dir = out / "runs" / f"{scenario}_{preset}_{speed}_{matcher}_{tracker}"
    key = {"frames": a.frames}
    key_file, nothing = run_dir / "bench_key.json", run_dir / "nothing_localized.json"
    done = (run_dir / "run.jsonl").is_file() or nothing.is_file()
    if not (a.reuse and done and key_file.is_file() and json.loads(key_file.read_text()) == key):
        log.info("run %s", run_dir.name)
        nothing.unlink(missing_ok=True)
        frames_in = _frames_in(flight, a.frames, run_dir)
        try:
            localize_run(map_png, frames_in, run_dir, matcher=matcher, cache_dir=cache, tracker=tracker,
                         device=device)
        except NothingLocalized:    # a result, not a failure: e.g. SIFT on the pre-war map (scenarios E, F)
            run_dir.mkdir(parents=True, exist_ok=True)
            nothing.write_text(json.dumps(FolderSource(frames_in, undistort=False).idxs))
        except Exception as e:      # one broken combination must not lose the others
            log.exception("run %s failed", run_dir.name)
            return {**row, "status": "error", "error": f"{type(e).__name__}: {e}"}
        key_file.write_text(json.dumps(key))
    truth = irpin.truth_on_map(flight, map_png)
    if nothing.is_file():
        frames, row["note"] = missed(json.loads(nothing.read_text()), truth), "nothing localized"
    else:
        frames = frame_metrics(read_run(run_dir / "run.jsonl"), truth, irpin.m_per_px(map_png))
    return {**row, "status": "ok", "run": str(run_dir.relative_to(out)),
            "metrics": summarize(frames, seed=a.seed, n_boot=a.boot), "timing": timing(frames)}


def _cell(m: dict) -> str:
    if m["value"] is None:
        return "n/a"
    return f"{m['value']:.1f} ({m['lo']:.1f}–{m['hi']:.1f})"


def table(rows: list[dict], config: dict) -> str:
    """The markdown table of ``rows``: one line per combination, every metric with its interval."""
    head = ["Scenario", "Preset", "m/s", "Matcher", "Tracker", "Frames", *COLUMNS.values(), "ms/frame"]
    lines = [(f"Benchmark (M6-03): {config['frames'] or 'all'} keyframes per flight, bootstrap 95 % intervals "
              f"over frames ({config['boot']} resamples, seed {config['seed']}). Localised: within 10 m; "
              "wrong fix: an accepted fix more than 30 m off."), "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lead = [r["scenario"], r["preset"], str(r["speed"]), r["matcher"], r["tracker"]]
        if r["status"] != "ok":
            what = r["status"] + (": " + ", ".join(r["missing"]) if "missing" in r else f": {r.get('error', '')}")
            lines.append("| " + " | ".join(lead) + f" | {what} |")
            continue
        m, ms = r["metrics"], r["timing"]["ms_per_frame"]
        cells = [str(m["frames"]), *(_cell(m[k]) for k in COLUMNS), "n/a" if ms is None else f"{ms:.0f}"]
        if "note" in r:
            cells[-1] += f" ({r['note']})"
        lines.append("| " + " | ".join(lead + cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data"), help="the data folder (default: data)")
    ap.add_argument("--out", type=Path, default=Path("outputs/bench"))
    ap.add_argument("--cache", type=Path, default=Path("outputs/cache"), help="map feature cache")
    ap.add_argument("--scenarios", nargs="+", choices=list(irpin.SCENARIOS), default=list(irpin.SCENARIOS))
    ap.add_argument("--presets", nargs="+", choices=irpin.PRESETS, default=list(irpin.PRESETS))
    ap.add_argument("--speeds", nargs="+", type=int, choices=irpin.SPEEDS, default=list(irpin.SPEEDS))
    ap.add_argument("--matchers", nargs="+", choices=sorted(MATCHERS), default=["sift"],
                    help="default sift; a learned matcher joins as soon as it is in the pipeline")
    ap.add_argument("--trackers", nargs="+", choices=TRACKERS, default=list(TRACKERS))
    ap.add_argument("--frames", type=int, default=None, help="only the first N keyframes of each flight")
    ap.add_argument("--seed", type=int, default=7, help="seed of the bootstrap intervals")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--reuse", action="store_true", help="score existing run folders instead of matching again")
    add_shared_args(ap)
    a = ap.parse_args(argv)
    if a.frames is not None and a.frames < 1 or a.boot < 1:
        ap.error("--frames and --boot must be >= 1")
    device = init_cli(a)

    config = {k: getattr(a, k) for k in ("scenarios", "presets", "speeds", "matchers", "trackers", "frames",
                                         "seed", "boot")}
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in a.scenarios:
        for preset in a.presets:
            for speed in a.speeds:
                for matcher in a.matchers:
                    for tracker in a.trackers:
                        rows.append(_one(a.data, a.out, a.cache, scenario, preset, speed, matcher, tracker, a, device))
                        # written after each run: a long benchmark that stops keeps what it has
                        (a.out / "bench.json").write_text(json.dumps({"config": config, "rows": rows}, indent=2))
    text = table(rows, config)
    (a.out / "bench.md").write_text(text)
    print(text)
    return 1 if any(r["status"] == "error" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
