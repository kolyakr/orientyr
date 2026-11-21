"""M6-03 · benchmark runner: dronenav.eval (metrics, Irpin truth) and scripts/benchmark.py.

Done when (issue #55): "one command regenerates the table; two runs with the same seed give identical numbers",
with the 20 Nov metrics spec. As a contract:
1. metrics (dronenav.eval.metrics), each on frames with an exact, hand-made error:
   - error in metres = map px x the map's m_per_px (the M0-05 georeference);
   - localised % = frames with an accepted fix within 10 m; wrong-fix % = frames with an accepted fix more than
     30 m off (both over all frames; a rejected fix counts as neither);
   - coverage % = frames where the tracker gives a position at all;
   - median, RMSE and p95 of the reported position's error;
   - uncertainty coverage = frames whose truth lies inside the tracker's 95 % region (chi-square, 2 d.o.f.);
   - each number has a bootstrap 95 % interval over frames: lo <= value <= hi, identical for the same seed;
2. the truth of a flight on another year's map is shifted by both maps' ``alignment_to_2025`` (dronenav.eval.irpin);
3. ``scripts/benchmark.py`` runs scenarios x presets x speeds x matchers x trackers on the Irpin layout and writes
   ``bench.json`` and ``bench.md`` (a table whose cells show the intervals); a combination whose data is missing is
   reported as such, not a crash; two runs with the same seed give identical numbers (timing is the machine's,
   kept apart and not compared).
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.test_run import _city

from dronenav.eval import irpin
from dronenav.eval.metrics import WITHIN_M, WRONG_M, FrameEval, frame_metrics, missed, summarize
from dronenav.sim import CutoutSource, cutout_route
from dronenav.telemetry import Record
from dronenav.types import Fix, TrackMode, TrackState

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "benchmark.py"


# --- helpers: records with known positions ---------------------------------------------------------------------------

def _fix(idx, x, y, accepted=True):
    return Fix(idx=idx, H=np.eye(3), x=x, y=y, heading_deg=0.0, scale=1.0,
               footprint=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float), inliers=50, inlier_ratio=0.5,
               reproj_px=1.0, accepted=accepted, reject_reason=None if accepted else "too_few_inliers")


def _rec(idx, fix=None, track=None, cov=None, ms=10.0):
    """A record: ``fix`` = (x, y, accepted) or None; ``track`` = (x, y) or None (no position yet)."""
    f = _fix(idx, *fix) if fix is not None else None
    if track is None:
        st = TrackState(idx, TrackMode.LOST, None, None, None, None, None, fix=f)
    else:
        c = np.eye(2) * 4.0 if cov is None else np.asarray(cov, float)
        st = TrackState(idx, TrackMode.TRACKING, track[0], track[1], 0.0, 0.0, c, fix=f)
    return Record(idx=idx, t=float(idx), source=f"frame_{idx:05d}.png", track=st, ms=ms)


TRUTH = {i: (100.0, 100.0) for i in range(10)}


# --- 1. metrics ------------------------------------------------------------------------------------------------------

def test_error_is_in_metres_through_the_map_scale():
    # a fix 3, 4 px off is 5 px; at 2 m per px that is 10 m
    [f] = frame_metrics([_rec(0, fix=(103, 104, True), track=(103, 104))], TRUTH, m_per_px=2.0)
    assert isinstance(f, FrameEval)
    assert f.fix_m == pytest.approx(10.0)
    assert f.track_m == pytest.approx(10.0)


def test_localised_and_wrong_fix_thresholds():
    assert (WITHIN_M, WRONG_M) == (10.0, 30.0)     # the spec's numbers
    recs = [_rec(0, fix=(109.9, 100, True), track=(109.9, 100)),     # 9.9 m: localised
            _rec(1, fix=(110.1, 100, True), track=(110.1, 100)),     # 10.1 m: neither localised nor wrong
            _rec(2, fix=(129.9, 100, True), track=(129.9, 100)),     # 29.9 m: not wrong yet
            _rec(3, fix=(130.1, 100, True), track=(130.1, 100)),     # 30.1 m: wrong
            _rec(4, fix=(500, 500, False), track=(100, 100)),        # rejected: counts as neither
            ]
    s = summarize(frame_metrics(recs, TRUTH, m_per_px=1.0), seed=7, n_boot=200)
    assert s["frames"] == 5
    assert s["localised_pct"]["value"] == pytest.approx(20.0)
    assert s["wrong_fix_pct"]["value"] == pytest.approx(20.0)


def test_coverage_counts_frames_with_a_position():
    recs = [_rec(0, track=(100, 100)), _rec(1), _rec(2), _rec(3, track=(101, 100))]
    s = summarize(frame_metrics(recs, TRUTH, m_per_px=1.0), seed=7, n_boot=200)
    assert s["coverage_pct"]["value"] == pytest.approx(50.0)


def test_median_rmse_p95_of_the_reported_position():
    errs = [1.0, 2.0, 3.0, 4.0, 10.0]
    recs = [_rec(i, track=(100 + e, 100)) for i, e in enumerate(errs)] + [_rec(5)]   # frame 5: no position
    s = summarize(frame_metrics(recs, TRUTH, m_per_px=1.0), seed=7, n_boot=200)
    assert s["median_m"]["value"] == pytest.approx(3.0)
    assert s["rmse_m"]["value"] == pytest.approx(math.sqrt(np.mean(np.square(errs))))
    assert s["p95_m"]["value"] == pytest.approx(np.percentile(errs, 95))


def test_uncertainty_coverage_uses_the_95_percent_region():
    # sigma 2 px both ways: the 95 % region reaches sqrt(5.991) sigma = 4.9 px; 4 px is inside, 5.2 px outside
    cov = np.eye(2) * 4.0
    recs = [_rec(0, track=(104, 100), cov=cov), _rec(1, track=(105.2, 100), cov=cov),
            _rec(2, track=(100, 103), cov=cov), _rec(3)]                            # frame 3: no region at all
    s = summarize(frame_metrics(recs, TRUTH, m_per_px=1.0), seed=7, n_boot=200)
    assert s["unc_coverage_pct"]["value"] == pytest.approx(100 * 2 / 3)


def test_no_position_at_all_gives_none_not_a_crash():
    s = summarize(frame_metrics([_rec(0), _rec(1)], TRUTH, m_per_px=1.0), seed=7, n_boot=200)
    assert s["coverage_pct"]["value"] == 0.0
    assert s["median_m"] == {"value": None, "lo": None, "hi": None}


def test_bootstrap_interval_brackets_the_value_and_repeats_with_the_seed():
    rng = np.random.default_rng(0)
    recs = [_rec(i, fix=(100 + e, 100, True), track=(100 + e, 100)) for i, e in enumerate(rng.uniform(0, 20, 60))]
    frames = frame_metrics(recs, {i: (100.0, 100.0) for i in range(60)}, m_per_px=1.0)
    a, b = summarize(frames, seed=7, n_boot=300), summarize(frames, seed=7, n_boot=300)
    assert a == b
    for key in ("localised_pct", "median_m", "rmse_m", "p95_m"):
        v = a[key]
        assert v["lo"] <= v["value"] <= v["hi"], (key, v)
        assert v["lo"] < v["hi"], key                    # 60 spread-out errors: a real interval, not a point
    assert summarize(frames, seed=8, n_boot=300)["median_m"] != a["median_m"]


def test_a_flight_found_nowhere_scores_zero_not_nothing():
    # a run that localized no frame at all (the pipeline's NothingLocalized) is a result: 0 %, not a missing row
    s = summarize(missed([0, 1, 2, 99], TRUTH), seed=7, n_boot=50)          # 99 has no truth: left out
    assert s["frames"] == 3
    assert s["localised_pct"]["value"] == 0.0 and s["coverage_pct"]["value"] == 0.0
    assert s["wrong_fix_pct"]["value"] == 0.0
    assert s["median_m"]["value"] is None


def test_frames_without_truth_are_left_out():
    recs = [_rec(0, track=(100, 100)), _rec(99, track=(0, 0))]
    assert [f.idx for f in frame_metrics(recs, TRUTH, m_per_px=1.0)] == [0]


# --- 2. truth on another year's map -----------------------------------------------------------------------------------

def _sidecar(path: Path, shift, m_per_px=0.5, size=(2000, 1500)):
    block = {"file": path.with_suffix(".png").name, "zoom": 17, "m_per_px": m_per_px, "size_px": list(size),
             "merc_origin_px": [0, 0], "bounds_lonlat": [30.0, 50.0, 30.1, 50.1], "capture_coverage": 1.0,
             "alignment_to_2025": {"shift_px": list(shift)}}
    path.write_text(json.dumps({"capture": {"date": "2025-01-01", "sensor": "test"}, "source": "test",
                                "map": block, "hires": block}))


def test_truth_is_shifted_between_the_flight_map_and_the_scenario_map(tmp_path):
    maps = tmp_path / "irpin" / "maps"
    maps.mkdir(parents=True)
    _sidecar(maps / "irpin_2022.json", (-1.0, 1.0))
    _sidecar(maps / "irpin_2017.json", (3.0, 2.0))
    f = tmp_path / "irpin" / "flights" / "mild" / "2022"
    f.mkdir(parents=True)
    (f / "meta.json").write_text(json.dumps({"map": "maps/irpin_2022.png", "fps": 30, "keyframes": []}))
    (f / "truth.jsonl").write_text(json.dumps({"i": 0, "t": 0, "center_map_px": [500.0, 400.0]}) + "\n")
    # 2022 px + 2022 shift = 2025 px; 2025 px - 2017 shift = 2017 px
    assert irpin.truth_on_map(f, maps / "irpin_2017.png") == {0: pytest.approx((500 - 1 - 3, 400 + 1 - 2))}
    assert irpin.truth_on_map(f, maps / "irpin_2022.png") == {0: pytest.approx((500.0, 400.0))}
    assert irpin.m_per_px(maps / "irpin_2017.png") == 0.5


def test_scenarios_and_flight_folders():
    assert {k: (s.flight, s.map) for k, s in irpin.SCENARIOS.items()} == {
        "C": ("2025", "2025"), "D": ("2025", "2022"), "E": ("2025", "2017"), "F": ("2022", "2017")}
    d = Path("data")
    assert irpin.flight_dir(d, "mild", 15, "2025") == d / "irpin" / "flights" / "mild" / "2025"
    assert irpin.flight_dir(d, "harsh", 30, "2022") == d / "irpin" / "flights" / "harsh_30mps" / "2022"
    with pytest.raises(ValueError):
        irpin.flight_dir(d, "mild", 20, "2025")


# --- 3. the script ---------------------------------------------------------------------------------------------------

N = 6   # keyframes per fake flight: a track, and quick for 2 trackers x 2 runs


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """data/irpin with the real layout: one textured city as every year's map (2022 and 2017 slightly shifted by
    their sidecars, as the real ones are), cut-out flights mild/2025 and mild_30mps/2025, every 30th video frame a
    keyframe, truth at the keyframes. No harsh flights: those combinations are missing data."""
    d = tmp_path_factory.mktemp("data")
    img = _city(2000, 1500)
    maps = d / "irpin" / "maps"
    maps.mkdir(parents=True)
    for year, shift in (("2025", (0, 0)), ("2022", (-1.0, 1.0)), ("2017", (2.0, 1.5))):
        cv2.imwrite(str(maps / f"irpin_{year}.png"), img)
        _sidecar(maps / f"irpin_{year}.json", shift)
    for folder, seed in (("mild", 1), ("mild_30mps", 2)):
        f = d / "irpin" / "flights" / folder / "2025"
        (f / "frames").mkdir(parents=True)
        poses = cutout_route((2000, 1500), N, seed=seed, frame_size=(320, 240), scale=(0.9, 1.1), tilt_max_deg=3.0)
        src = CutoutSource(img, poses, (320, 240))
        keys, truth = [], []
        for k, fr in enumerate(src):
            i = 30 * k
            name = f"frames/frame_{i:05d}.png"
            cv2.imwrite(str(f / name), fr.image)
            keys.append({"i": i, "file": name})
            truth.append(json.dumps({"i": i, "t": i / 30, "center_map_px": list(src.truth(k).center)}))
        (f / "meta.json").write_text(json.dumps({"map": "maps/irpin_2025.png", "fps": 30, "keyframes": keys}))
        (f / "truth.jsonl").write_text("\n".join(truth) + "\n")
    return d


def _bench(data, out, *extra):
    return subprocess.run([sys.executable, str(CLI), "--data", str(data), "--out", str(out),
                           "--cache", str(out.parent / "cache"), *map(str, extra)],
                          check=False, capture_output=True, text=True)


ARGS = ("--scenarios", "C", "E", "--presets", "mild", "--speeds", "15", "30", "--trackers", "kalman", "none",
        "--boot", "200")


@pytest.fixture(scope="module")
def two_runs(data, tmp_path_factory):
    t = tmp_path_factory.mktemp("bench")
    res = [_bench(data, t / f"out{k}", *ARGS) for k in (1, 2)]
    return t, res


def test_one_command_writes_bench_json_and_the_table(two_runs):
    t, (res, _) = two_runs
    assert res.returncode == 0, res.stdout + res.stderr
    bench = json.loads((t / "out1" / "bench.json").read_text())
    rows = bench["rows"]
    assert len(rows) == 2 * 1 * 2 * 1 * 2                          # scenarios x presets x speeds x matchers x trackers
    assert {(r["scenario"], r["speed"], r["tracker"]) for r in rows} == {
        (s, v, tr) for s in "CE" for v in (15, 30) for tr in ("kalman", "none")}
    for r in rows:
        assert r["status"] == "ok" and r["matcher"] == "sift" and r["preset"] == "mild"
        m = r["metrics"]
        assert m["frames"] == N
        for key in ("localised_pct", "wrong_fix_pct", "coverage_pct", "median_m", "rmse_m", "p95_m",
                    "unc_coverage_pct"):
            assert set(m[key]) == {"value", "lo", "hi"}, key
        assert m["localised_pct"]["value"] >= 50, r     # exact cut-outs of the map: most frames are found
        assert r["timing"]["ms_per_frame"] > 0
    table = (t / "out1" / "bench.md").read_text()
    assert table.count("\n| C ") + table.count("\n| E ") == len(rows)
    assert "–" in table and "(" in table                         # cells carry their interval: "v (lo–hi)"


def test_same_seed_gives_identical_numbers(two_runs):
    t, res = two_runs
    assert all(r.returncode == 0 for r in res)
    a, b = (json.loads((t / f"out{k}" / "bench.json").read_text()) for k in (1, 2))
    strip = [[{k: v for k, v in r.items() if k != "timing"} for r in x["rows"]] for x in (a, b)]
    assert strip[0] == strip[1]


def test_missing_data_is_reported_not_a_crash(data, tmp_path):
    res = _bench(data, tmp_path / "out", "--scenarios", "C", "--presets", "harsh", "--speeds", "15",
                 "--trackers", "none", "--boot", "50")
    assert res.returncode == 0, res.stdout + res.stderr
    [row] = json.loads((tmp_path / "out" / "bench.json").read_text())["rows"]
    assert row["status"] == "missing data" and "harsh" in row["missing"][0]
    assert "missing data" in (tmp_path / "out" / "bench.md").read_text()


def test_a_map_where_nothing_is_found_is_a_zero_row(data, tmp_path):
    # E flies the 2025 route over the 2017 map; here that map is unrelated texture: full of features (a blank map
    # is refused earlier, as unusable), none of them the frames', so no frame can be found on it
    bad = tmp_path / "data"
    for p in data.rglob("*"):
        q = bad / p.relative_to(data)
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
        elif p.name != "irpin_2017.png":
            q.symlink_to(p)
    cv2.imwrite(str(bad / "irpin" / "maps" / "irpin_2017.png"), 
                cv2.resize(np.random.default_rng(3).integers(0, 256, (150, 200), np.uint8), (2000, 1500)))
    res = _bench(bad, tmp_path / "out", "--scenarios", "E", "--presets", "mild", "--speeds", "15",
                 "--trackers", "none", "--boot", "50")
    assert res.returncode == 0, res.stdout + res.stderr
    [row] = json.loads((tmp_path / "out" / "bench.json").read_text())["rows"]
    assert row["status"] == "ok" and row["note"] == "nothing localized"
    assert row["metrics"]["frames"] == N and row["metrics"]["localised_pct"]["value"] == 0.0
    assert "nothing localized" in (tmp_path / "out" / "bench.md").read_text()


@pytest.mark.parametrize("args", [("--scenarios", "Z"), ("--speeds", "20"), ("--matchers", "nope"),
                                  ("--trackers", "nope")])
def test_bad_arguments_are_exit_2(data, tmp_path, args):
    assert _bench(data, tmp_path / "out", *args).returncode == 2
