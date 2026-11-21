"""One command per demo scenario (M8-01): list them, run one, several or all, and report each.

    python scripts/scenarios.py list                 # every scenario, its one command and the run.py it stands for
    python scripts/scenarios.py list --markdown      # the same as the README table
    python scripts/scenarios.py run all              # run each; PASS / FAIL / MISSING DATA, then a summary
    python scripts/scenarios.py run C E --data data --out outputs/scenarios

The scenarios are the plan's (docs/plan M6, "Scenarios"): what the drone sees x the map it carries. G (controlled
drift maps) and the live mode were cut in the last-day re-plan (#128), so they are not here. Each scenario is a
plain scripts/run.py call: the runner adds nothing to the pipeline, so what passes here is exactly what a judge
runs by hand. A scenario whose data is not downloaded (or not made yet: the phone takes) is MISSING DATA, not a
failure, unless --strict. Exit code 0: nothing failed. 1: a scenario failed (or, with --strict, had no data).
2: bad arguments. The report is also written to <out>/scenarios.json.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dronenav.config import add_shared_args, init_cli

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("scripts") / "run.py"

TASK_MAP = "hackatons-data/global_map.png"
MAPS = "irpin/maps/irpin_{}.png"
FLIGHT = "irpin/flights/mild/{}/frames"            # the "mild" preset: the showcase flights of the README


@dataclass(frozen=True)
class Scenario:
    key: str
    sees: str                  # what the drone's camera sees
    map: str                   # the map it carries
    shows: str                 # why the scenario is in the demo
    args: tuple[str, ...]      # run.py arguments; paths relative to the data folder are marked with {data}

    def argv(self, data: Path) -> list[str]:
        return [a.format(data=data) for a in self.args]

    def needs(self, data: Path) -> list[Path]:
        """The data paths the command reads: missing ones make the scenario MISSING DATA."""
        return [Path(a.format(data=data)) for a in self.args if "{data}" in a]


def _paths(source: str, map_rel: str) -> tuple[str, ...]:
    flag = "--video" if source.endswith(".mp4") else "--frames"
    return ("--map", f"{{data}}/{map_rel}", flag, f"{{data}}/{source}")


SCENARIOS = (
    Scenario("A", "the 50 task frames", "the task map", "the deliverable",
             _paths("hackatons-data/crops", TASK_MAP)),
    # the phone take: simulated over the task map by M2-08, published by M2-01 in its flight format
    Scenario("B", "the phone take (calm)", "the task map", "the advanced level",
             (*_paths("video/sim/task/calm/flight.mp4", TASK_MAP), "--every", "10")),
    # 30 s of a jammed camera at 150 s: the tracker loses the fix, dead-reckons, relocalizes (README, M7-02)
    Scenario("C", "post-war ground", "post-war map (2025)", "the mechanics: jam and recovery",
             (*_paths(FLIGHT.format(2025), MAPS.format(2025)), "--jam", "150:30")),
    Scenario("D", "post-war ground", "war map (2022)", "a war-time map, used after the rebuild",
             _paths(FLIGHT.format(2025), MAPS.format(2022))),
    Scenario("E", "post-war ground", "pre-war map (2017)", "the hardest case: an old map over a changed town",
             _paths(FLIGHT.format(2025), MAPS.format(2017))),
    Scenario("F", "war ground", "pre-war map (2017)", "flying into a town during the war",
             _paths(FLIGHT.format(2022), MAPS.format(2017))),
)
BY_KEY = {s.key: s for s in SCENARIOS}


def _shown(s: Scenario, data: str = "data", out: str = "outputs/scenarios") -> str:
    """The run.py command a scenario stands for, as a person would type it from the repo root."""
    return shlex.join(["python", str(RUN), *s.argv(Path(data)), "--out", f"{out}/{s.key}"])


def _list(markdown: bool) -> None:
    if markdown:
        print("| Scenario | Drone sees | Map it carries | Shows | Command |")
        print("|---|---|---|---|---|")
        for s in SCENARIOS:
            print(f"| {s.key} | {s.sees} | {s.map} | {s.shows} | `python scripts/scenarios.py run {s.key}` |")
        return
    for s in SCENARIOS:
        print(f"{s.key}  {s.sees} on {s.map}: {s.shows}")
        print(f"   python scripts/scenarios.py run {s.key}")
        print(f"   = {_shown(s)}")


def _run_one(s: Scenario, data: Path, out: Path, cache: Path) -> dict:
    missing = [str(p) for p in s.needs(data) if not p.exists()]
    if missing:
        return {"status": "MISSING DATA", "missing": missing, "command": _shown(s, str(data), str(out))}
    cmd = [sys.executable, str(ROOT / RUN), *s.argv(data), "--out", str(out / s.key), "--cache", str(cache)]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    rec = {"status": "PASS" if res.returncode == 0 else "FAIL", "exit": res.returncode,
           "seconds": round(time.perf_counter() - t0, 2), "command": _shown(s, str(data), str(out))}
    if res.returncode != 0:      # the last lines say why; the whole stderr would bury the summary
        rec["error"] = "\n".join(res.stderr.strip().splitlines()[-3:])
    return rec


def _keys(names: list[str]) -> list[str]:
    if names == ["all"]:
        return list(BY_KEY)
    bad = [n for n in names if n.upper() not in BY_KEY]
    if bad:
        raise ValueError(f"unknown scenario {', '.join(bad)} (known: {', '.join(BY_KEY)}, or all)")
    return [n.upper() for n in names]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_shared_args(ap, device=False)   # only starts run.py, which picks its own device
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list", help="every scenario and its one command")
    ls.add_argument("--markdown", action="store_true", help="as the README table")
    rn = sub.add_parser("run", help="run scenarios and report each")
    rn.add_argument("keys", nargs="+", metavar="SCENARIO", help=f"{', '.join(BY_KEY)}, or all")
    rn.add_argument("--data", default="data", help="the data folder (default: data)")
    rn.add_argument("--out", default="outputs/scenarios", help="where the run folders go (default: outputs/scenarios)")
    rn.add_argument("--cache", default="outputs/cache", help="map feature cache, shared by the runs")
    rn.add_argument("--strict", action="store_true", help="a scenario without its data counts as a failure")
    a = ap.parse_args(argv)
    init_cli(a)

    if a.cmd == "list":
        _list(a.markdown)
        return 0
    try:
        keys = _keys(a.keys)
    except ValueError as e:
        ap.error(str(e))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for key in keys:
        print(f"running {key} ...", file=sys.stderr, flush=True)
        report[key] = _run_one(BY_KEY[key], Path(a.data), out, Path(a.cache))
        (out / "scenarios.json").write_text(json.dumps(report, indent=2))   # after each: a crash keeps the rest
    for key, r in report.items():
        detail = (f"{r['seconds']:.1f} s  {out / key}" if r["status"] == "PASS"
                  else ", ".join(r["missing"]) if r["status"] == "MISSING DATA"
                  else f"exit {r['exit']}: {r['error'].splitlines()[-1] if r['error'] else '(no message)'}")
        print(f"{key}  {r['status']:<12}  {detail}")
    failed = [k for k, r in report.items() if r["status"] == "FAIL" or (a.strict and r["status"] != "PASS")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
