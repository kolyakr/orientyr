"""Pack a run for the hosted console: one small, self-contained folder (scripts/pack_run.py).

A run folder points at a map outside itself (in data/) and holds PNG frames: tens of megabytes, and nothing a
static host can serve. The packed copy is what gets deployed. What is tested here:
1. the packed folder is self-contained: run.json names a map inside it, and every record's frame is inside it;
2. the telemetry is untouched: same records, same numbers, only `source` changes its extension;
3. the map keeps its pixel size (the console draws in map pixel coordinates), frames keep their aspect and
   are no wider than --frame-width, and the folder is much smaller than the run it came from;
4. it refuses what would lose data or mislead: a missing run, packing a run into itself, a run whose frame
   is missing, an output folder that holds something other than an older packed run.

A synthetic textured map and a short fake run keep the tests fast and independent of data/.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.fake_run import FakeRunConfig, make_fake_run
from dronenav.telemetry import read_run, read_run_info, record_to_dict

ROOT = Path(__file__).resolve().parents[1]
MAP_W, MAP_H = 1400, 900
FRAMES = 12
FRAME_W = 160   # the width the frames are packed at in these tests (they are 320 px wide in the run)


def _cli(*args) -> subprocess.CompletedProcess:
    """Run scripts/pack_run.py as a user would; the caller checks the exit code."""
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "pack_run.py"), *map(str, args)],
                          check=False, capture_output=True, text=True)


def _size_of(folder: Path) -> int:
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> Path:
    """A fake run whose map lives outside the run folder, as in the repo (outputs/<run> → data/…)."""
    base = tmp_path_factory.mktemp("pack")
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.uniform(0, 255, (MAP_H, MAP_W, 3)).astype(np.float32), (0, 0), 2.0)
    (base / "data").mkdir()
    cv2.imwrite(str(base / "data" / "map.png"), img.astype(np.uint8))
    out = base / "outputs" / "run"
    make_fake_run(base / "data" / "map.png", out, FakeRunConfig(frames=FRAMES, frame_size=(320, 240), jam_len=2))
    return out


@pytest.fixture(scope="module")
def packed(run_dir, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("packed") / "run"
    res = _cli("--run", run_dir, "--out", out, "--frame-width", FRAME_W)
    assert res.returncode == 0, res.stderr
    return out


def test_the_packed_folder_is_self_contained(packed):
    raw = json.loads((packed / "run.json").read_text())
    assert not Path(raw["map"]).is_absolute() and ".." not in Path(raw["map"]).parts   # the map is inside
    info = read_run_info(packed)
    assert Path(info.map).is_file()
    records = read_run(packed / "run.jsonl")
    assert len(records) == FRAMES
    for rec in records:
        assert ".." not in Path(rec.source).parts
        assert (packed / rec.source).is_file(), f"no frame for record {rec.idx}"


def test_the_telemetry_is_untouched_except_the_frame_extension(run_dir, packed):
    before, after = read_run(run_dir / "run.jsonl"), read_run(packed / "run.jsonl")
    assert len(before) == len(after)
    for a, b in zip(before, after):
        da, db = record_to_dict(a), record_to_dict(b)
        assert Path(da.pop("source")).with_suffix("") == Path(db.pop("source")).with_suffix("")
        assert da == db
    old, new = read_run_info(run_dir), read_run_info(packed)
    # frame_size stays the size the homographies were estimated at, even though the packed frames are smaller
    assert (new.map_size, new.frame_size, new.fps, new.seed) == (old.map_size, old.frame_size, old.fps, old.seed)


def test_the_map_keeps_its_size_and_the_frames_shrink(run_dir, packed):
    info = read_run_info(packed)
    map_img = cv2.imread(info.map)
    assert map_img is not None and (map_img.shape[1], map_img.shape[0]) == (MAP_W, MAP_H)
    for rec in read_run(packed / "run.jsonl"):
        frame = cv2.imread(str(packed / rec.source))
        h, w = frame.shape[:2]
        assert w == FRAME_W
        assert h == 120   # 320x240 at width 160: the aspect is kept
    # lossy map and half-size lossy frames against PNGs of noise-like texture: far below a half in practice;
    # a half is the bar so that the test does not depend on the encoder's version
    assert _size_of(packed) < 0.5 * (_size_of(run_dir) + Path(read_run_info(run_dir).map).stat().st_size)


def test_frames_smaller_than_the_width_are_not_enlarged(run_dir, tmp_path):
    out = tmp_path / "run"
    assert _cli("--run", run_dir, "--out", out, "--frame-width", 4000).returncode == 0
    first = read_run(out / "run.jsonl")[0]
    assert cv2.imread(str(out / first.source)).shape[:2] == (240, 320)


def test_packing_again_replaces_the_old_copy(run_dir, tmp_path):
    out = tmp_path / "run"
    (out / "frames").mkdir(parents=True)
    (out / "frames" / "stale.jpg").write_bytes(b"old")
    (out / "run.json").write_text("{}")                  # what marks the folder as an older packed run
    assert _cli("--run", run_dir, "--out", out, "--frame-width", FRAME_W).returncode == 0
    assert not (out / "frames" / "stale.jpg").exists()   # a frame of an older run must not be deployed


def test_it_never_replaces_a_folder_that_is_not_a_packed_run(run_dir, tmp_path):
    out = tmp_path / "work"
    out.mkdir()
    (out / "notes.txt").write_text("mine")               # a mistyped --out must not cost someone their files
    res = _cli("--run", run_dir, "--out", out)
    assert res.returncode != 0 and "not a packed run" in res.stderr
    assert (out / "notes.txt").read_text() == "mine"


def test_it_refuses_a_missing_run_and_packing_a_run_into_itself(run_dir, tmp_path):
    res = _cli("--run", tmp_path / "nope", "--out", tmp_path / "out")
    assert res.returncode != 0 and "nope" in res.stderr
    assert not (tmp_path / "out").exists()
    res = _cli("--run", run_dir, "--out", run_dir)
    assert res.returncode != 0 and "itself" in res.stderr
    assert (run_dir / "run.jsonl").is_file() and len(list((run_dir / "frames").iterdir())) == FRAMES


def test_it_refuses_a_run_with_a_missing_frame(run_dir, tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    info = json.loads((run_dir / "run.json").read_text())
    info["map"] = str(Path(read_run_info(run_dir).map))   # absolute: this copy sits elsewhere
    (broken / "run.json").write_text(json.dumps(info))
    (broken / "run.jsonl").write_text((run_dir / "run.jsonl").read_text())   # frames/ is not copied
    res = _cli("--run", broken, "--out", tmp_path / "out")
    assert res.returncode != 0 and "frame" in res.stderr
