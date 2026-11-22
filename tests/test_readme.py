"""M1-11 · README v1 and requirements: the part of "a teammate follows the README on a clean machine without
asking anything" that a machine can check.

- the README has the sections a newcomer needs, one diagram of the approach, and the result image;
- every command it shows names a script that exists and only flags that script accepts;
- every relative link and image points to a file that is in the repo (not a gitignored one), and images stay small;
- every line of ``requirements.txt`` is bounded from above (or pinned to a commit), and the bounds admit the
  versions the tests are running on.

Whether the prose is clear is for the reviewer: see the PR.
"""
from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
import sys
import tomllib
from functools import cache
from pathlib import Path

import pytest
from packaging.requirements import Requirement  # comes with pytest

from dronenav.pipeline import OUTPUTS

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")

MAX_IMAGE_BYTES = 500_000


# --- sections ---------------------------------------------------------------------------------------------------

def _headings() -> list[str]:
    in_code, out = False, []
    for ln in README.splitlines():
        if ln.startswith("```"):
            in_code = not in_code
        elif not in_code and re.match(r"#{1,3} ", ln):
            out.append(ln.lstrip("# ").strip().lower())
    return out


@pytest.mark.parametrize("word", ["approach", "install", "data", "run", "output", "result", "numbers", "tests",
                                  "layout"])
def test_readme_has_the_section(word):
    assert any(word in h for h in _headings()), f"no heading with {word!r} in {_headings()}"


def test_readme_sections_come_in_the_order_a_newcomer_needs():
    # install before data before run: following the page top to bottom must work
    heads = _headings()
    first = [next((i for i, h in enumerate(heads) if w in h), -1) for w in ("install", "data", "run")]
    assert -1 not in first and first == sorted(first), dict(zip(("install", "data", "run"), first, strict=True))


def test_readme_has_exactly_one_diagram():
    assert len(re.findall(r"^```mermaid\s*$", README, flags=re.MULTILINE)) == 1


def test_readme_states_the_python_version_of_pyproject():
    floor = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"].lstrip(">=")
    assert f"Python {floor}" in README


def test_readme_gives_the_install_command():
    assert "pip install -r requirements.txt" in README and "pip install -e ." in README
    assert "venv" in README


def test_readme_names_every_output_of_a_run():
    missing = [name for name in OUTPUTS if name not in README]
    assert not missing, missing


def test_readme_has_no_email_address():
    assert not re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", README)


# --- commands ---------------------------------------------------------------------------------------------------

def _commands() -> list[str]:
    """The lines of the README's shell blocks, with backslash continuations joined and comments cut."""
    out = []
    for block in re.findall(r"^```(?:bash|sh|shell)\n(.*?)^```", README, flags=re.MULTILINE | re.DOTALL):
        for ln in block.replace("\\\n", " ").splitlines():
            ln = re.sub(r"\s+#.*$", "", ln).strip()
            if ln and not ln.startswith("#"):
                out.append(ln)
    return out


@cache
def _help(script: str) -> str:
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{ROOT}"}
    r = subprocess.run([sys.executable, str(ROOT / script), "--help"], capture_output=True, text=True, env=env,
                       cwd=ROOT, check=False)
    assert r.returncode == 0, f"{script} --help: {r.stderr.strip()}"
    return r.stdout


def test_readme_shows_commands():
    cmds = _commands()
    assert any("scripts/run.py" in c for c in cmds)
    assert any(c.startswith("tools/task.sh test") for c in cmds)


def test_every_script_and_tool_the_readme_names_exists():
    paths = set(re.findall(r"\b((?:scripts|tools|src|tests)/[\w./-]*\w\.(?:py|sh))\b", README))
    assert paths, "the README names no script at all"
    missing = sorted(p for p in paths if not (ROOT / p).is_file())
    assert not missing, missing


def test_every_flag_in_a_readme_command_is_accepted_by_its_script():
    checked = 0
    for cmd in _commands():
        m = re.search(r"python3? (scripts/\w+\.py)(.*)", cmd)
        if not m:
            continue
        script, rest = m.group(1), m.group(2)
        assert (ROOT / script).is_file(), script
        known = set(re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", _help(script)))
        for flag in re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", rest):
            assert flag in known, f"{script} does not accept {flag}: {cmd}"
            checked += 1
    assert checked >= 3, "the README shows no script command with flags"


def test_the_run_command_is_the_deliverable_one():
    # the task: the map and the folder of frames come from CLI paths, the outputs go to a named folder
    runs = [c for c in _commands() if "scripts/run.py" in c]
    assert any(all(f in c for f in ("--map", "--frames", "--out")) for c in runs), runs


def test_readme_documents_local_inputs():
    links, _ = _targets()
    assert "data/README.md" in links
    assert "`--video`" in README
    assert all("data/" not in command for command in _commands() if "scripts/run.py" in command)


# --- links and images -------------------------------------------------------------------------------------------

def _targets() -> tuple[set[str], set[str]]:
    """(links, images): the relative targets of markdown links / images and of html src / href attributes."""
    text = re.sub(r"^```.*?^```", "", README, flags=re.MULTILINE | re.DOTALL)   # code blocks hold no links
    images = set(re.findall(r"!\[[^\]]*\]\(([^)\s]+)", text)) | set(re.findall(r'<img[^>]*\ssrc="([^"]+)"', text))
    links = set(re.findall(r"(?<!!)\[[^\]]*\]\(([^)\s]+)", text)) | set(re.findall(r'href="([^"]+)"', text))

    def local(ts):
        return {t.split("#")[0] for t in ts if not re.match(r"(https?:|mailto:|#)", t)} - {""}

    return local(links), local(images)


def _ignored(path: str) -> bool:
    """True when git would not carry ``path`` to a fresh clone: it is untracked and an ignore rule covers it."""
    if subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=ROOT, capture_output=True,
                      check=False).returncode == 0:
        return False   # tracked: a rule that matches it (a laptop's own .git/info/exclude, say) changes nothing
    # -v prints the pattern that decided: check-ignore also "matches" a negated one (!/data/README.md), which
    # un-ignores the file. Exit code 1: no pattern matched at all.
    r = subprocess.run(["git", "check-ignore", "-v", "--no-index", path], cwd=ROOT, capture_output=True, text=True,
                       check=False)
    assert r.returncode in (0, 1), r
    return r.returncode == 0 and not r.stdout.split("\t")[0].split(":", 2)[2].startswith("!")


def test_every_relative_link_points_into_the_repo():
    links, images = _targets()
    assert links, "the README links to nothing"
    for t in sorted(links | images):
        assert (ROOT / t).exists(), f"broken link: {t}"
        # a gitignored file is there on the author's laptop and missing for everyone else
        assert not _ignored(t), f"{t} is gitignored: the link is broken on a fresh clone"


def test_readme_shows_a_demo():
    _, images = _targets()
    hosted_video = re.search(r"^https://github\.com/user-attachments/assets/[a-f0-9-]+$", README, re.MULTILINE)
    assert hosted_video or any(t.startswith("results/") for t in images)
    for t in sorted(images):
        assert (ROOT / t).is_file(), t
        assert (ROOT / t).stat().st_size <= MAX_IMAGE_BYTES, f"{t}: {(ROOT / t).stat().st_size} bytes"


# --- requirements.txt -------------------------------------------------------------------------------------------

def _requirement_lines() -> list[str]:
    lines = [ln.split(" #")[0].strip() for ln in (ROOT / "requirements.txt").read_text().splitlines()]
    return [ln for ln in lines if ln and not ln.startswith(("#", "-"))]


def test_requirements_are_there():
    assert len(_requirement_lines()) >= 10


@pytest.mark.parametrize("line", _requirement_lines())
def test_requirement_is_pinned(line):
    req = Requirement(line)
    if req.url:   # a git dependency: only a full commit hash is a pin, a branch moves
        assert re.search(r"@[0-9a-f]{40}$", req.url), f"{req.name}: pin the commit, got {req.url}"
        return
    ops = {s.operator for s in req.specifier}
    # an upper bound, so a new major release can't break a clean install the day before the demo
    assert ops & {"==", "~=", "<", "<="}, f"{line}: no upper bound"
    assert ops & {"==", "~=", ">="}, f"{line}: no lower bound"


@pytest.mark.parametrize("line", _requirement_lines())
def test_requirement_admits_the_installed_version(line):
    # the bounds must describe an environment in which this very test suite passes
    req = Requirement(line)
    try:
        installed = importlib.metadata.version(req.name)
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(f"{req.name} is not installed here")
    if not req.url:
        assert req.specifier.contains(installed, prereleases=True), f"{req.name} {installed} is outside {line}"
