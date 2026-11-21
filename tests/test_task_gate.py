"""M0-11 · Faster test loop: the test gate of ``tools/task.sh`` (``test``, ``pr``, ``finish``).

The real script runs in a throwaway git repo with a bare ``origin``, and with ``gh`` and ``pytest`` replaced by
stubs on PATH that log how they were called: nothing here talks to GitHub or runs a real suite. "Done when":

- the default gate is fast: parallel, no ``data`` tests      → test_the_default_gate_is_parallel_and_skips_data_tests
  (its wall time on a laptop is measured in the PR, a unit test cannot hold a stopwatch)
- ``finish`` right after a green ``pr`` skips the run        → test_finish_right_after_a_green_pr_does_not_run_the_tests_again
  and runs it when the tree changed                          → test_finish_runs_the_tests_when_the_tree_changed
- ``--full`` runs the ``data`` tests too                     → test_full_adds_the_data_tests, test_the_pr_body_says_*
- green under ``-n auto``                                    → the suite itself, run that way by the gate and by CI
- CI uses ``-n auto``; the docs describe the gate            → test_ci_runs_the_suite_in_parallel, test_agents_md_*
- ``test [pytest args]`` passes the args through unchanged   → test_pytest_args_pass_through_unchanged
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(os.name != "posix" or not shutil.which("bash") or not shutil.which("git"),
                                reason="needs bash and git")

# What the stubbed GitHub knows: one task issue, #7, and (once `pr` has run) one open PR, #42, into develop.
GH_STUB = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >>"$STUB_DIR/gh.log"
case "$1 $2" in
  "issue list") echo 7 ;;
  "api user") echo tester ;;
  "api graphql")
    case "$*" in
      *"| .title"*) echo "M0-11 · Faster test loop" ;;
      *"| .projectItems"*) echo ITEM_1 ;;
    esac ;;
  "pr list")
    [[ -f "$STUB_DIR/pr.body" ]] || exit 0
    case "$*" in *baseRefName*) echo "42 develop" ;; *) echo "https://example.test/pr/42" ;; esac ;;
  "pr create")
    while [[ $# -gt 0 ]]; do [[ "$1" == --body ]] && printf '%s\n' "$2" >"$STUB_DIR/pr.body"; shift; done ;;
  "pr view") cat "$STUB_DIR/pr.body" ;;
  "pr edit")
    while [[ $# -gt 0 ]]; do [[ "$1" == --body ]] && printf '%s\n' "$2" >"$STUB_DIR/pr.body"; shift; done ;;
  "pr checks") echo 0 ;;
esac
exit 0
"""

# Logs one line per real run (its arguments), and fails when the test says the suite is red. `-h` is how the gate
# asks whether pytest-xdist is installed: it is not a run, so it is not logged.
PYTEST_STUB = r"""#!/usr/bin/env bash
for a in "$@"; do
  if [[ "$a" == -h || "$a" == --help ]]; then
    [[ -n "${STUB_NO_XDIST:-}" ]] || echo "  -n numprocesses, --numprocesses=numprocesses"
    exit 0
  fi
done
printf '%s\n' "$*" >>"$STUB_DIR/pytest.log"
exit "${STUB_PYTEST_EXIT:-0}"
"""


class Sandbox:
    """A task branch in a throwaway clone, with the real ``tools/task.sh`` and stubbed ``gh`` / ``pytest``."""

    def __init__(self, tmp: Path):
        self.repo, self.stubs = tmp / "repo", tmp / "stubs"
        self.stubs.mkdir()
        for name, text in (("gh", GH_STUB), ("pytest", PYTEST_STUB)):
            (self.stubs / name).write_text(text)
            (self.stubs / name).chmod(stat.S_IRWXU)
        keep = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "TASK_", "STUB_", "PYTEST_"))}
        self.env = {**keep, "PATH": f"{self.stubs}{os.pathsep}{os.environ['PATH']}", "STUB_DIR": str(self.stubs),
                    "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull,
                    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.test",
                    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.test"}
        self.git("init", "-q", "--bare", "-b", "develop", str(tmp / "origin.git"), cwd=tmp)
        self.git("init", "-q", "-b", "develop", str(self.repo), cwd=tmp)
        (self.repo / "tools").mkdir()
        shutil.copy(ROOT / "tools" / "task.sh", self.repo / "tools" / "task.sh")
        shutil.copy(ROOT / "tools" / "task.env", self.repo / "tools" / "task.env")
        (self.repo / "tests").mkdir()
        (self.repo / "tests" / "test_nothing.py").write_text("def test_nothing():\n    pass\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "base")
        self.git("remote", "add", "origin", str(tmp / "origin.git"))
        self.git("push", "-q", "-u", "origin", "develop")
        self.git("switch", "-q", "-c", "M0-11-faster-test-loop")
        self.commit("M0-11: work")

    def git(self, *args: str, cwd: Path | None = None) -> str:
        return subprocess.run(["git", *args], cwd=cwd or self.repo, env=self.env, check=True,
                              capture_output=True, text=True).stdout.strip()

    def commit(self, message: str) -> None:
        with (self.repo / "work.txt").open("a") as f:
            f.write(message + "\n")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", message)

    def task(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "tools/task.sh", *args], cwd=self.repo, env={**self.env, **env},
                              capture_output=True, text=True, check=False, timeout=60)

    def pytest_runs(self) -> list[str]:
        log = self.stubs / "pytest.log"
        return log.read_text().splitlines() if log.exists() else []

    def gh_calls(self, prefix: str) -> list[str]:
        log = self.stubs / "gh.log"
        return [ln for ln in (log.read_text().splitlines() if log.exists() else []) if ln.startswith(prefix)]

    def pr_body(self) -> str:
        return (self.stubs / "pr.body").read_text()


@pytest.fixture
def box(tmp_path) -> Sandbox:
    return Sandbox(tmp_path)


def _selects_data_tests(run: str) -> bool:
    return "not data" not in run


# --- the gate itself: `tools/task.sh test` ------------------------------------------------------------------------

def test_the_default_gate_is_parallel_and_skips_data_tests(box):
    r = box.task("test")
    assert r.returncode == 0, r.stderr
    (run,) = box.pytest_runs()
    assert re.search(r"-n auto\b", run), run
    assert "-m not data" in run, run


@pytest.mark.parametrize("how", ["flag", "env"])
def test_full_adds_the_data_tests(box, how):
    r = box.task("test", "--full") if how == "flag" else box.task("test", TASK_FULL="1")
    assert r.returncode == 0, r.stderr
    (run,) = box.pytest_runs()
    assert _selects_data_tests(run), run
    assert re.search(r"-n auto\b", run), run                    # full, and still parallel
    assert "--full" not in run                                  # ours, not a pytest option


@pytest.mark.parametrize("args", [["-k", "geometry and not slow", "tests/test_nothing.py"], ["-x"], ["-m", "data"]])
def test_pytest_args_pass_through_unchanged(box, args):
    r = box.task("test", *args)
    assert r.returncode == 0, r.stderr
    assert box.pytest_runs() == [" ".join(args)]                # nothing added, nothing dropped


def test_pass_through_keeps_pytests_exit_code(box):
    assert box.task("test", "-x", STUB_PYTEST_EXIT="5").returncode == 5


def test_a_red_gate_fails_and_says_so(box):
    r = box.task("test", STUB_PYTEST_EXIT="1")
    assert r.returncode != 0
    assert "tests are red" in r.stderr


def test_without_xdist_the_gate_runs_serially_and_says_so(box):
    # a laptop that has not re-installed requirements.txt yet must still be able to open and merge PRs
    r = box.task("test", STUB_NO_XDIST="1")
    assert r.returncode == 0, r.stderr
    (run,) = box.pytest_runs()
    assert "-n" not in run.split() and "-m not data" in run, run
    assert "pytest-xdist" in r.stderr


# --- `pr` ---------------------------------------------------------------------------------------------------------

def test_pr_runs_the_fast_gate_and_the_body_says_data_tests_were_not_run(box):
    r = box.task("pr")
    assert r.returncode == 0, r.stderr
    (run,) = box.pytest_runs()
    assert not _selects_data_tests(run) and re.search(r"-n auto\b", run), run
    body = box.pr_body()
    assert "Closes #7" in body
    assert re.search(r"`data` tests[^\n]*not run", body), body


def test_the_pr_body_says_when_the_data_tests_were_run(box):
    r = box.task("pr", TASK_FULL="1")
    assert r.returncode == 0, r.stderr
    (run,) = box.pytest_runs()
    assert _selects_data_tests(run), run
    body = box.pr_body()
    assert re.search(r"`data` tests", body) and "not run" not in body, body


def test_rerunning_pr_updates_what_the_body_says_about_the_data_tests(box):
    assert box.task("pr").returncode == 0
    (box.stubs / "pr.body").write_text(box.pr_body() + "\nA note the author added by hand.\n")
    r = box.task("pr", TASK_FULL="1")
    assert r.returncode == 0, r.stderr
    body = box.pr_body()
    assert "not run" not in body and len(re.findall(r"`data` tests", body)) == 1, body
    assert "A note the author added by hand." in body and "Closes #7" in body      # the rest is left alone


def test_pr_refuses_on_red_tests(box):
    r = box.task("pr", STUB_PYTEST_EXIT="1")
    assert r.returncode != 0 and "tests are red" in r.stderr
    assert box.gh_calls("pr create") == []
    assert box.git("ls-remote", "--heads", "origin", "M0-11-faster-test-loop") == ""    # nothing pushed either


# --- `finish` -----------------------------------------------------------------------------------------------------

def test_finish_right_after_a_green_pr_does_not_run_the_tests_again(box):
    assert box.task("pr").returncode == 0
    r = box.task("finish")
    assert r.returncode == 0, r.stderr
    assert len(box.pytest_runs()) == 1                          # the one `pr` made
    assert re.search(r"already (green|passed)", r.stdout + r.stderr)
    assert len(box.gh_calls("pr merge 42")) == 1


def test_finish_runs_the_tests_when_the_tree_changed(box):
    assert box.task("pr").returncode == 0
    box.commit("M0-11: one more change")
    box.git("push", "-q")
    r = box.task("finish")
    assert r.returncode == 0, r.stderr
    assert len(box.pytest_runs()) == 2
    assert len(box.gh_calls("pr merge 42")) == 1


def test_finish_refuses_when_the_changed_tree_is_red(box):
    assert box.task("pr").returncode == 0
    box.commit("M0-11: a change that breaks a test")
    box.git("push", "-q")
    r = box.task("finish", STUB_PYTEST_EXIT="1")
    assert r.returncode != 0 and "tests are red" in r.stderr
    assert box.gh_calls("pr merge") == []


def test_a_red_run_takes_back_the_earlier_green(box):
    # green, then red on the very same tree (a flaky or machine-dependent test): the last word counts
    assert box.task("pr").returncode == 0
    assert box.task("test", STUB_PYTEST_EXIT="1").returncode != 0
    assert box.task("finish").returncode == 0
    assert len(box.pytest_runs()) == 3


def test_a_green_run_with_uncommitted_changes_proves_nothing_about_the_commit(box):
    assert box.task("pr").returncode == 0
    box.commit("M0-11: committed, not yet tested")
    box.git("push", "-q")
    (box.repo / "work.txt").write_text("a fix that is not committed\n")
    assert box.task("test").returncode == 0                     # green, but on the dirty tree
    box.git("checkout", "--", "work.txt")
    assert box.task("finish").returncode == 0
    assert len(box.pytest_runs()) == 3                          # pr, test, and finish again


def test_a_fast_green_is_not_enough_for_a_full_finish(box):
    assert box.task("pr").returncode == 0
    r = box.task("finish", TASK_FULL="1")
    assert r.returncode == 0, r.stderr
    runs = box.pytest_runs()
    assert len(runs) == 2 and _selects_data_tests(runs[1]), runs


def test_a_full_green_covers_a_fast_finish(box):
    assert box.task("pr", TASK_FULL="1").returncode == 0
    assert box.task("finish").returncode == 0
    assert len(box.pytest_runs()) == 1


def test_pass_through_runs_do_not_count_as_the_gate(box):
    # `test -k one_thing` is green on a tree where the whole suite may not be
    assert box.task("pr").returncode == 0
    box.commit("M0-11: one more change")
    box.git("push", "-q")
    assert box.task("test", "-k", "one_thing").returncode == 0
    assert box.task("finish").returncode == 0
    assert len(box.pytest_runs()) == 3


# --- CI and the docs ----------------------------------------------------------------------------------------------

def test_ci_runs_the_suite_in_parallel():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    runs = [s["run"] for job in wf["jobs"].values() for s in job["steps"] if re.search(r"\bpytest\b", s.get("run", ""))]
    assert runs and all(re.search(r"\bpytest\b.*\s-n auto\b", r) for r in runs), runs


def test_xdist_is_a_dev_requirement():
    names = {re.split(r"[<>=!~\[; ]", ln.strip(), maxsplit=1)[0].lower()
             for ln in (ROOT / "requirements.txt").read_text().splitlines() if ln.strip() and ln[0] not in "#-"}
    assert "pytest-xdist" in names


def test_agents_md_describes_the_gate():
    text = (ROOT / "AGENTS.md").read_text()
    commands = text[text.index("## Commands"):]
    assert "tools/task.sh test --full" in commands
    lifecycle = text[text.index("### Task lifecycle"):text.index("### Blocked tasks")]
    rows = {m[1]: m[0] for m in re.finditer(r"^\| \d\. (\w+).*$", lifecycle, re.MULTILINE)}
    assert "data" in rows["PR"] and "--full" in rows["PR"]                  # what the gate leaves out, and how to add it
    assert re.search(r"already (green|passed)", rows["Merge"])              # finish does not repeat a green run
