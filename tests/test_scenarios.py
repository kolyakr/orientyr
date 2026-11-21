"""M8-01 · one command per scenario: scripts/scenarios.py.

Done when (issue #65): "a list of commands, each run once successfully after the freeze". As a contract:
1. ``scenarios.py list`` names every scenario of the plan that we ship (A-F; G and the live mode were cut in #128),
   each with exactly one command, and the README shows that same list (generated, so it cannot drift);
2. ``scenarios.py run <keys|all> --data D --out O`` runs each scenario's command and reports PASS / FAIL /
   MISSING DATA per scenario; a PASS leaves the deliverable outputs in O/<key>; exit 0 only when nothing failed;
3. a scenario whose data is not there (the phone take before M2-01 has filmed it) is MISSING DATA, not a failure,
   unless --strict; a scenario that fails does not stop the others;
4. on the real data (``@pytest.mark.data``), scenario A passes.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import pytest
from tests.test_run import OUTPUTS, _city

from dronenav.sim import CutoutSource, cutout_route

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "scenarios.py"
SHIPPED = ["A", "B", "C", "D", "E", "F"]
N = 8              # frames per fake flight: enough for a track, few enough to keep five runs fast


def _cli(*args):
    return subprocess.run([sys.executable, str(CLI), *map(str, args)], check=False, capture_output=True, text=True)


def _flight(img, d: Path, prefix: str, seed: int) -> None:
    d.mkdir(parents=True)
    poses = cutout_route((img.shape[1], img.shape[0]), N, seed=seed, frame_size=(320, 240), scale=(0.9, 1.1),
                         tilt_max_deg=3.0)
    for f in CutoutSource(img, poses, (320, 240)):
        cv2.imwrite(str(d / f"{prefix}{f.idx}.png"), f.image)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """A data/ tree with the real layout and stand-in content: one textured city as every map, cut-out flights.
    No phone take (data/video/), as on a checkout before M2-01."""
    d = tmp_path_factory.mktemp("data")
    img = _city(2000, 1500)                        # the size test_run uses: room for a route to turn
    (d / "hackatons-data").mkdir()
    cv2.imwrite(str(d / "hackatons-data" / "global_map.png"), img)
    _flight(img, d / "hackatons-data" / "crops", "crop_", seed=1)
    (d / "irpin" / "maps").mkdir(parents=True)
    for year in (2017, 2022, 2025):
        cv2.imwrite(str(d / "irpin" / "maps" / f"irpin_{year}.png"), img)
    for year in (2022, 2025):
        _flight(img, d / "irpin" / "flights" / "mild" / str(year) / "frames", "frame_", seed=year)
    return d


def _run(tmp_path, data, *keys, extra=()):
    return _cli("run", *keys, "--data", data, "--out", tmp_path / "out", "--cache", tmp_path / "cache", *extra)


def _report(tmp_path):
    return json.loads((tmp_path / "out" / "scenarios.json").read_text())


# --- 1. the list ---------------------------------------------------------------------------------------------------

def test_the_list_has_every_shipped_scenario_with_one_command():
    res = _cli("list")
    assert res.returncode == 0, res.stderr
    rows = [ln.split(None, 1) for ln in res.stdout.splitlines() if ln[:1] in SHIPPED and ln[1:2] == " "]
    assert [k for k, _ in rows] == SHIPPED
    for key in SHIPPED:
        cmds = [ln for ln in res.stdout.splitlines() if ln.strip().startswith(f"python scripts/scenarios.py run {key}")]
        assert len(cmds) == 1, (key, cmds)


def test_the_readme_shows_the_same_list():
    res = _cli("list", "--markdown")
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() in (ROOT / "README.md").read_text(), "README out of date: paste `scenarios.py list --markdown`"


def test_agents_md_shows_the_scenario_command():
    assert "python scripts/scenarios.py run all" in (ROOT / "AGENTS.md").read_text()


# --- 2 + 3. running them -------------------------------------------------------------------------------------------

def test_all_scenarios_run_and_report(tmp_path, data):
    res = _run(tmp_path, data, "all")
    assert res.returncode == 0, res.stdout + res.stderr
    rep = _report(tmp_path)
    assert sorted(rep) == SHIPPED
    for key in "ACDEF":
        assert rep[key]["status"] == "PASS", (key, rep[key])
        for name in OUTPUTS:
            assert (tmp_path / "out" / key / name).is_file(), (key, name)
        assert rep[key]["seconds"] > 0
    assert rep["B"]["status"] == "MISSING DATA"
    assert "video" in rep["B"]["missing"][0]
    for key in SHIPPED:                            # the summary on screen names every scenario and its status
        assert any(ln.startswith(f"{key} ") and rep[key]["status"] in ln for ln in res.stdout.splitlines()), key


def test_strict_turns_missing_data_into_a_failure(tmp_path, data):
    res = _run(tmp_path, data, "B", extra=("--strict",))
    assert res.returncode == 1
    assert _report(tmp_path)["B"]["status"] == "MISSING DATA"


def test_a_failing_scenario_does_not_stop_the_others(tmp_path, data, monkeypatch):
    bad = tmp_path / "bad"
    (bad / "irpin" / "maps").mkdir(parents=True)
    for p in data.rglob("*"):                      # the same tree, but the pre-war map is not an image
        q = bad / p.relative_to(data)
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        elif p.name != "irpin_2017.png":
            q.symlink_to(p)
    (bad / "irpin" / "maps" / "irpin_2017.png").write_text("not an image")
    res = _run(tmp_path, bad, "E", "C")
    assert res.returncode == 1
    rep = _report(tmp_path)
    assert rep["E"]["status"] == "FAIL" and rep["E"]["exit"] != 0
    assert rep["C"]["status"] == "PASS"


@pytest.mark.parametrize("args", [("run", "Z"), ("run",), ("nope",)])
def test_bad_usage_is_exit_2(tmp_path, args):
    res = _cli(*args, "--data", tmp_path) if args[0] == "run" and len(args) > 1 else _cli(*args)
    assert res.returncode == 2


# --- 4. the real data ----------------------------------------------------------------------------------------------

@pytest.mark.data
@pytest.mark.skipif(not (ROOT / "data" / "hackatons-data" / "crops").is_dir(), reason="data/ not downloaded")
def test_scenario_a_passes_on_the_task_data(tmp_path):
    res = _cli("run", "A", "--data", ROOT / "data", "--out", tmp_path / "out", "--cache", tmp_path / "cache")
    assert res.returncode == 0, res.stdout + res.stderr
    assert _report(tmp_path)["A"]["status"] == "PASS"
