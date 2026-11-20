"""M0-01 · Repo skeleton: the contract from the issue's "Done when".

- `pip install -e .` works and `import dronenav` then works from anywhere (not only via PYTHONPATH);
- `requirements.txt` lists the agreed stack;
- `.gitignore` keeps data, outputs, videos and weights out of git, but not code or curated results.
"""
import re
import subprocess
import sys
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_from_src_layout():
    import dronenav

    assert Path(dronenav.__file__).resolve().parent == ROOT / "src" / "dronenav"
    assert re.fullmatch(r"\d+\.\d+\.\d+", dronenav.__version__)


def test_editable_install_makes_package_importable_anywhere(tmp_path):
    # A throwaway venv that sees the current site-packages (for setuptools), so the install
    # needs no network: --no-deps skips the heavy stack, --no-build-isolation reuses setuptools.
    env_dir = tmp_path / "venv"
    venv.EnvBuilder(system_site_packages=True, with_pip=True).create(env_dir)
    py = env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([py, "-m", "pip", "install", "-q", "--no-deps", "--no-build-isolation", "-e", ROOT],
                   check=True, cwd=tmp_path)
    # run from tmp_path with a clean PYTHONPATH: only the install can make this import work
    out = subprocess.run([py, "-c", "import dronenav; print(dronenav.__file__)"], check=True,
                         cwd=tmp_path, capture_output=True, text=True, env={"PYTHONPATH": ""})
    assert Path(out.stdout.strip()).resolve().parent == ROOT / "src" / "dronenav"


def test_requirements_list_the_stack():
    names = {re.split(r"[<>=!~\[; ]", line.strip(), maxsplit=1)[0].lower()
             for line in (ROOT / "requirements.txt").read_text().splitlines()
             if line.strip() and not line.lstrip().startswith(("#", "-"))}
    assert {"opencv-contrib-python", "numpy", "scipy", "torch", "kornia", "lightglue", "pytest", "ruff"} <= names


def test_layout_dirs_exist():
    for d in ("src/dronenav", "scripts", "tests", "results"):
        assert (ROOT / d).is_dir(), d


def _ignored(path: str) -> bool:
    # git check-ignore exits 0 when the path is ignored, 1 when not; --no-index also checks tracked paths
    r = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT)
    assert r.returncode in (0, 1), r
    return r.returncode == 0


@pytest.mark.parametrize("path", [
    "data/irpin/maps/irpin_2025.png", "outputs/run1/run.jsonl",                      # data and outputs
    "results/demo.mp4", "x.mov", "x.avi", "x.mkv", "x.webm",                          # videos
    "weights/superpoint.pt", "x.pth", "x.ckpt", "x.safetensors", "x.onnx", "x.bin",  # model weights
])
def test_gitignore_blocks_data_outputs_videos_weights(path):
    assert _ignored(path)


@pytest.mark.parametrize("path", [
    "src/dronenav/__init__.py", "scripts/run.py", "tests/test_skeleton.py",
    "requirements.txt", "pyproject.toml", "results/trajectory.png",
])
def test_gitignore_keeps_code_and_curated_results(path):
    assert not _ignored(path)
