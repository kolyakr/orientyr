"""M0-04 · run.json: the run-level facts next to run.jsonl, above all which map the run was flown over.

Renderers take only ``--run <dir>``, so the run folder must name its map. The path is stored relative to the
run folder (a run folder moved together with its map, or read from another working directory, still works).
"""
import json

import pytest

from dronenav.telemetry import RunInfo, TelemetryError, read_run_info, write_run_info


def _map(tmp_path):
    p = tmp_path / "maps" / "map.png"
    p.parent.mkdir()
    p.write_bytes(b"not really a png; run.json only stores the path")
    return p


def test_round_trip(tmp_path):
    m = _map(tmp_path)
    info = RunInfo(map=str(m), map_size=(3645, 1845), frame_size=(640, 640), fps=1.0, generator="fake_run", seed=7)
    write_run_info(tmp_path / "run", info)
    back = read_run_info(tmp_path / "run")
    assert back.map_path.samefile(m)
    assert (back.map_size, back.frame_size, back.fps, back.generator, back.seed) == \
           ((3645, 1845), (640, 640), 1.0, "fake_run", 7)


def test_map_is_stored_relative_to_the_run_folder(tmp_path, monkeypatch):
    m = _map(tmp_path)
    write_run_info(tmp_path / "run", RunInfo(map=str(m), map_size=(10, 10), generator="test"))
    stored = json.loads((tmp_path / "run" / "run.json").read_text())["map"]
    assert stored == "../maps/map.png"
    # read from a different working directory: the path resolves against the run folder, not the cwd
    monkeypatch.chdir(tmp_path / "maps")
    assert read_run_info(tmp_path / "run").map_path.samefile(m)


def test_relative_map_given_by_caller_is_taken_from_the_cwd(tmp_path, monkeypatch):
    m = _map(tmp_path)
    monkeypatch.chdir(tmp_path)
    write_run_info("out/run1", RunInfo(map="maps/map.png", map_size=(10, 10), generator="test"))
    assert read_run_info(tmp_path / "out" / "run1").map_path.samefile(m)


def test_optional_fields_default_to_null(tmp_path):
    write_run_info(tmp_path, RunInfo(map=str(_map(tmp_path)), map_size=(10, 10), generator="run.py"))
    back = read_run_info(tmp_path)
    assert (back.frame_size, back.fps, back.seed) == (None, None, None)


BAD = {
    "missing map": (lambda d: d.pop("map"), "'map'"),
    "map_size not two ints": (lambda d: d.update(map_size=[10]), "map_size"),
    "map_size negative": (lambda d: d.update(map_size=[10, -1]), "map_size"),
    "fps zero": (lambda d: d.update(fps=0), "fps"),
    "seed float": (lambda d: d.update(seed=1.5), "seed"),
    "unknown key": (lambda d: d.update(colour="red"), "'colour'"),
}


@pytest.mark.parametrize("case", BAD)
def test_malformed_run_json_names_the_field(tmp_path, case):
    write_run_info(tmp_path, RunInfo(map=str(_map(tmp_path)), map_size=(10, 10), fps=1.0, generator="t", seed=1))
    p = tmp_path / "run.json"
    d = json.loads(p.read_text())
    mutate, word = BAD[case]
    mutate(d)
    p.write_text(json.dumps(d))
    with pytest.raises(TelemetryError, match=rf"run\.json.*{word}"):
        read_run_info(tmp_path)


def test_missing_run_json_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_run_info(tmp_path)
