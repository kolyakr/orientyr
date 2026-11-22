"""M0-08 · CI: the GitHub Actions workflow is part of the contract, so its shape is tested like code.

What the spec asks for, and how each item is pinned here:
- ``ruff check`` and ``pytest`` on Python 3.11            → test_lints_with_ruff / test_runs_pytest_like_task_sh / test_python_311
- CPU only                                                → test_torch_comes_from_the_cpu_index
- a PR shows a check (green or red)                       → test_runs_on_prs_into_develop_and_main
- lint is the same locally and in CI                      → test_ruff_is_pinned / test_ruff_skips_experiments

"A deliberately broken test turns it red" can only be shown on GitHub: the PR links that run.
"""
from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def _workflow() -> dict:
    yaml = pytest.importorskip("yaml")      # pyyaml is a dev requirement; CI always has it
    return yaml.safe_load(WORKFLOW.read_text())


def _triggers(wf: dict) -> dict:
    return wf.get("on", wf.get(True))       # YAML 1.1 reads a bare `on:` key as the boolean True


def _steps(wf: dict) -> list[dict]:
    return [s for job in wf["jobs"].values() for s in job["steps"]]


def _runs(wf: dict) -> str:
    return "\n".join(s.get("run", "") for s in _steps(wf))


def _cancels(concurrency: dict | str | None, event: str) -> bool:
    """Would a newer run in this concurrency group cancel the running one, for this triggering event?"""
    if not isinstance(concurrency, dict):            # a bare group string (or none) never cancels
        return False
    v = concurrency.get("cancel-in-progress", False)
    if isinstance(v, bool):
        return v
    m = re.fullmatch(r"\$\{\{\s*github\.event_name\s*(==|!=)\s*'(\w+)'\s*\}\}", str(v).strip())
    assert m, f"cancel-in-progress expression this test can't evaluate: {v!r}"
    return (event == m[2]) == (m[1] == "==")


def test_workflow_file_exists():
    assert WORKFLOW.is_file()


def test_stale_pr_runs_are_cancelled():
    # on a PR only the newest push matters: a stale run would just hold a runner. The runs of one PR must also
    # share a group (keyed by the PR number), or there is nothing for the newer run to cancel.
    concurrency = _workflow().get("concurrency")
    assert _cancels(concurrency, "pull_request")
    assert "github.event.pull_request.number" in concurrency["group"]


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_runs_on_develop_and_main_are_never_cancelled(event):
    # M0-10: every merge commit gets its own verdict; a cancelled run on develop hides which merge broke it
    wf = _workflow()
    assert not _cancels(wf.get("concurrency"), event)
    for name, job in wf["jobs"].items():
        assert not _cancels(job.get("concurrency"), event), name


def test_pushes_do_not_share_a_concurrency_group():
    # even with cancel-in-progress false, a group holds one running + one pending run and a third arrival
    # replaces (cancels) the pending one: outside PRs, every run needs a group of its own (its run id)
    wf = _workflow()
    groups = [c["group"] if isinstance(c, dict) else c
              for c in [wf.get("concurrency"), *(j.get("concurrency") for j in wf["jobs"].values())] if c]
    for g in groups:
        assert "github.run_id" in g, g


def test_runs_on_prs_into_develop_and_main():
    on = _triggers(_workflow())
    assert {"develop", "main"} <= set(on["pull_request"]["branches"])
    assert {"develop", "main"} <= set(on["push"]["branches"])   # the integration branches stay checked after merge


def test_python_311():
    versions = [str(s["with"]["python-version"]) for s in _steps(_workflow())
                if s.get("uses", "").startswith("actions/setup-python")]
    assert versions and all(v == "3.11" for v in versions)      # the floor in pyproject.toml, so 3.11 must work


def test_runners_are_cpu_only():
    for job in _workflow()["jobs"].values():
        assert job["runs-on"].startswith("ubuntu-")             # GitHub-hosted Linux: no GPU


def test_torch_comes_from_the_cpu_index():
    # the default Linux torch wheel drags ~2 GB of CUDA libraries into a GPU-less runner
    assert "download.pytorch.org/whl/cpu" in WORKFLOW.read_text()


def test_lints_with_ruff():
    assert re.search(r"^\s*ruff check\b", _runs(_workflow()), re.MULTILINE)


def test_runs_pytest_like_task_sh():
    # tools/task.sh runs the suite with src/ and the repo root importable; CI must see the same imports
    runs = _runs(_workflow())
    assert re.search(r"PYTHONPATH=\S*src\S*\s+pytest\b", runs) or "tools/task.sh test" in runs


def test_every_run_step_is_valid_bash():
    # check what GitHub will actually execute: the YAML-parsed script, not the text as it looks in the file.
    # (A plain-scalar `run:` containing " #" is cut there by YAML as a comment, leaving an unclosed quote.)
    for step in _steps(_workflow()):
        if "run" in step:
            r = subprocess.run(["bash", "-n"], input=step["run"], capture_output=True, text=True, check=False)
            assert r.returncode == 0, f"{step.get('name', step['run'])!r}: {r.stderr.strip()}"


def test_ruff_is_pinned():
    # ruff's default rule set changes between minor versions: an unpinned ruff can turn CI red with no code change
    line = next(ln for ln in (ROOT / "requirements.txt").read_text().splitlines() if ln.startswith("ruff"))
    assert "<" in line.split("#")[0]


def test_ruff_skips_experiments():
    # experiments/ is throwaway research code (see its READMEs), linted by nobody; the package, tests, scripts are
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["ruff"]
    assert "experiments" in cfg.get("extend-exclude", []) + cfg.get("exclude", [])


def test_pyyaml_is_a_dev_requirement():
    names = {re.split(r"[<>=!~\[; ]", ln.strip(), maxsplit=1)[0].lower()
             for ln in (ROOT / "requirements.txt").read_text().splitlines() if ln.strip() and ln[0] not in "#-"}
    assert "pyyaml" in names


@pytest.mark.parametrize("path", ["tests/fixtures/irpin_mild_2025/map_crop.jpg",
                                  "tests/fixtures/irpin_mild_2025/frame_006300.jpg",
                                  "tests/fixtures/irpin_mild_2025/truth.jsonl"])
def test_gitignore_keeps_the_fixtures(path):
    # exit code 0 = ignored, 1 = not ignored (so check=False: 1 is the answer we want, not an error);
    # anything else (128: not a git repo) means git never answered, which must not pass as "kept"
    code = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], check=False, cwd=ROOT).returncode
    assert code == 1, f"git check-ignore exit {code}"
