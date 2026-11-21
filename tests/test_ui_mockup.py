"""M7-01 · UI mock-up on fake telemetry: the part of the contract a machine can check.

"Done when" is a human decision (the team agrees on layout and look), proven by screenshots in the PR. What
is tested here is what must hold before anyone looks:

1. `ui/` is a Next.js project with the dev / build / test commands, and its dependencies and build output
   stay out of git;
2. the console runs on a fake run written by the Python generator (M0-04): the UI's own reader accepts
   every line of run.jsonl, finds every frame, and the page renders every part the issue names (map, trail,
   footprint, camera inset, HUD, state badge, inactive controls) in each tracker state.

Item 2 runs the UI's test suite (vitest) with ORIENTYR_RUN pointing at a fresh fake run, so the two languages
are checked against the same file. It needs Node and `npm install` in ui/; without them it skips cleanly.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.fake_run import FakeRunConfig, make_fake_run

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui"


def _package() -> dict:
    return json.loads((UI / "package.json").read_text(encoding="utf-8"))


def test_ui_is_a_next_project_with_the_agreed_commands():
    pkg = _package()
    deps = pkg.get("dependencies", {})
    assert {"next", "react", "react-dom"} <= deps.keys()
    assert {"dev", "build", "test"} <= pkg.get("scripts", {}).keys()
    assert (UI / "app" / "page.tsx").is_file(), "the console is the app's home page"


@pytest.mark.parametrize("path", ["ui/node_modules/x.js", "ui/.next/x.js", "ui/next-env.d.ts"])
def test_ui_dependencies_and_build_output_stay_out_of_git(path):
    r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT, check=False)
    assert r.returncode == 0, f"{path} is not ignored"


def test_ui_source_is_not_ignored():
    # the data rules ignore *.png and cache/ everywhere: the UI's code must not be caught by them
    for path in ("ui/package.json", "ui/app/page.tsx", "ui/lib/run.ts", "ui/tests/fixtures/run/run.jsonl"):
        r = subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT, check=False)
        assert r.returncode == 1, f"{path} is ignored"


def test_console_runs_on_a_fake_run_from_the_python_generator(tmp_path):
    assert (UI / "package.json").is_file(), "ui/package.json is missing"
    npm = shutil.which("npm")
    if npm is None:
        pytest.skip("Node is not installed: the UI suite cannot run")
    if not (UI / "node_modules").is_dir():
        pytest.skip("ui/node_modules is missing: run `npm install` in ui/")

    # a textured synthetic map: fast, and independent of data/
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, (500, 800, 3), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "map.png"), cv2.GaussianBlur(noise, (0, 0), 3))
    run_dir = tmp_path / "run"
    make_fake_run(tmp_path / "map.png", run_dir, FakeRunConfig(frames=24, frame_size=(128, 128), seed=3))

    r = subprocess.run([npm, "test"], cwd=UI, check=False, capture_output=True, text=True, timeout=300,
                       env={**os.environ, "ORIENTYR_RUN": str(run_dir), "ORIENTYR_RUN_FRAMES": "24", "CI": "1"})
    assert r.returncode == 0, f"the UI suite failed on the fake run:\n{r.stdout[-4000:]}\n{r.stderr[-2000:]}"
