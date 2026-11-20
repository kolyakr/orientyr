"""M0-03 · Telemetry record and `run.jsonl`.

"Done when: round-trip test (write → read → equal) passes; a malformed record fails validation with a clear
message." The record schema is the one settled in the issue (spec update of 20 Nov): the TrackState flattened,
the Fix nested without its idx, optional backend / ms / interpolated / extra, finite values only, unknown keys
rejected. Equality is exact: Python writes floats with repr, which round-trips every float64 bit for bit.
"""
import json
import math

import numpy as np
import pytest

from dronenav.telemetry import (
    Record,
    RunWriter,
    TelemetryError,
    iter_run,
    read_run,
    record_from_dict,
    record_to_dict,
    write_run,
)
from dronenav.types import Fix, TrackMode, TrackState

# --- fixtures: records shaped like the ones the pipeline will write ------------------------------------------


def _fix(idx, accepted=True, rng=None):
    rng = rng or np.random.default_rng(idx)
    H = np.array([[0.57, 0.04, 1804.65], [0.48, 0.74, 960.19], [7.2e-05, 2.3e-04, 1.0]]) + rng.normal(0, 1e-3, (3, 3))
    footprint = np.array([[1799.3, 933.7], [2242.9, 1352.2], [1939.3, 1632.4], [1545.3, 1298.6]]) + rng.normal(0, 1, (4, 2))
    return Fix(idx=idx, H=H, x=1877.57 + idx, y=1324.13 - idx / 3, heading_deg=41.28 + idx, scale=0.731,
               footprint=footprint, inliers=191 if accepted else 4, inlier_ratio=0.62 if accepted else 0.05,
               reproj_px=1.37, accepted=accepted, reject_reason=None if accepted else "too_few_inliers")


def _record(idx, state=TrackMode.TRACKING, fix="accepted", **kw):
    rng = np.random.default_rng(100 + idx)
    f = None if fix is None else _fix(idx, accepted=(fix == "accepted"), rng=rng)
    track = TrackState(idx=idx, state=state, x=1877.0 + idx * 1.1, y=1324.0 - idx * 0.3, vx=1.1 / 3, vy=-0.1,
                       cov=np.array([[4.0, 0.3], [0.3, 2.5]]) * (1 + idx / 10), fix=f)
    return Record(idx=idx, t=idx / 30, source=f"frames/frame_{idx:05d}.png", track=track, **kw)


def _assert_records_equal(a: Record, b: Record):
    """Field-by-field, arrays with exact equality: the round trip must not lose a single bit."""
    assert (a.idx, a.t, a.source, a.backend, a.ms, a.interpolated, a.extra) == \
           (b.idx, b.t, b.source, b.backend, b.ms, b.interpolated, b.extra)
    ta, tb = a.track, b.track
    assert (ta.idx, ta.state, ta.x, ta.y, ta.vx, ta.vy) == (tb.idx, tb.state, tb.x, tb.y, tb.vx, tb.vy)
    assert np.array_equal(ta.cov, tb.cov)
    assert (ta.fix is None) == (tb.fix is None)
    if ta.fix is not None:
        fa, fb = ta.fix, tb.fix
        assert (fa.idx, fa.x, fa.y, fa.heading_deg, fa.scale, fa.inliers, fa.inlier_ratio, fa.reproj_px,
                fa.accepted, fa.reject_reason) == \
               (fb.idx, fb.x, fb.y, fb.heading_deg, fb.scale, fb.inliers, fb.inlier_ratio, fb.reproj_px,
                fb.accepted, fb.reject_reason)
        assert np.array_equal(fa.H, fb.H) and np.array_equal(fa.footprint, fb.footprint)


# --- Done when 1: write → read → equal -----------------------------------------------------------------------

ROUND_TRIP_CASES = {
    "tracking, accepted fix": lambda: _record(0),
    "lost, no fix": lambda: _record(7, state=TrackMode.LOST, fix=None),
    "relocalizing, rejected fix": lambda: _record(9, state=TrackMode.RELOCALIZING, fix="rejected"),
    "all optional fields": lambda: _record(3, backend="sift", ms=42.5, interpolated=True,
                                           extra={"truth": {"center_map_px": [1877.5, 1324.1]}, "event": "blur"}),
}


@pytest.mark.parametrize("case", ROUND_TRIP_CASES)
def test_round_trip_one_record(tmp_path, case):
    rec = ROUND_TRIP_CASES[case]()
    path = tmp_path / "run.jsonl"
    write_run(path, [rec])
    (back,) = read_run(path)
    _assert_records_equal(rec, back)


def test_round_trip_dict_level():
    rec = _record(5, backend="lightglue", ms=340.0)
    _assert_records_equal(rec, record_from_dict(record_to_dict(rec)))


def test_round_trip_a_whole_run_keeps_order(tmp_path):
    # 50 frames like the task run, with a jam episode in the middle
    states = [TrackMode.TRACKING] * 20 + [TrackMode.LOST] * 5 + [TrackMode.RELOCALIZING] * 2 + [TrackMode.TRACKING] * 23
    recs = [_record(i, state=s, fix=None if s is TrackMode.LOST else "accepted") for i, s in enumerate(states)]
    path = tmp_path / "run.jsonl"
    write_run(path, recs)
    back = read_run(path)
    assert [r.idx for r in back] == list(range(50))
    for a, b in zip(recs, back):
        _assert_records_equal(a, b)
    # iter_run streams the same records, lazily
    assert [r.idx for r in iter_run(path)] == list(range(50))


def test_one_line_per_record_and_arrays_as_lists(tmp_path):
    path = tmp_path / "run.jsonl"
    write_run(path, [_record(0), _record(1, fix=None, state=TrackMode.LOST)])
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    d = json.loads(lines[0])
    assert isinstance(d["cov"], list) and len(d["cov"]) == 2 and len(d["cov"][0]) == 2
    assert isinstance(d["fix"]["H"], list) and len(d["fix"]["H"]) == 3
    assert isinstance(d["fix"]["footprint"], list) and len(d["fix"]["footprint"]) == 4
    assert d["state"] == "TRACKING" and "idx" not in d["fix"]   # the fix's idx is the record's idx
    assert json.loads(lines[1])["fix"] is None


def test_numpy_scalars_are_written_as_plain_json(tmp_path):
    # values computed with numpy are numpy scalars; json.dumps rejects them unless converted
    track = TrackState(idx=np.int64(2), state="TRACKING", x=np.float32(10.5), y=np.float64(20.25), vx=0.0, vy=0.0,
                       cov=np.eye(2))
    rec = Record(idx=np.int64(2), t=np.float64(0.5), source="a.png", track=track, ms=np.float32(3.5))
    path = tmp_path / "run.jsonl"
    write_run(path, [rec])
    d = json.loads(path.read_text())
    assert (d["idx"], d["x"], d["y"], d["ms"]) == (2, 10.5, 20.25, 3.5)
    assert type(d["idx"]) is int


def test_writer_flushes_each_record_for_live_mode(tmp_path):
    # live mode (M7-04) reads the file while the engine is still writing it
    path = tmp_path / "run.jsonl"
    with RunWriter(path) as w:
        for i in range(3):
            w.write(_record(i))
            assert len(path.read_text().splitlines()) == i + 1
    assert [r.idx for r in read_run(path)] == [0, 1, 2]


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "run.jsonl"
    write_run(path, [_record(0)])
    path.write_text(path.read_text() + "\n\n")
    assert len(read_run(path)) == 1


# --- Done when 2: a malformed record fails validation with a clear message -------------------------------------

def _valid():
    return record_to_dict(_record(4, backend="sift", ms=12.0))


def _set(path, value):
    def mutate(d):
        *head, last = path
        for k in head:
            d = d[k]
        d[last] = value
    return mutate


def _drop(key):
    return lambda d: d.pop(key)


# (mutation, words the message must contain); each message names the offending field
MALFORMED = {
    "missing x": (_drop("x"), ["missing", "'x'"]),
    "missing state": (_drop("state"), ["missing", "'state'"]),
    "missing fix key": (_drop("fix"), ["missing", "'fix'"]),
    "unknown state": (_set(["state"], "FLYING"), ["state", "FLYING"]),
    "x is a string": (_set(["x"], "12.5"), ["x", "number"]),
    "x is a bool": (_set(["x"], True), ["x", "number"]),
    "idx is a float": (_set(["idx"], 1.5), ["idx", "int"]),
    "negative idx": (_set(["idx"], -1), ["idx"]),
    "cov wrong shape": (_set(["cov"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]), ["cov", "2x2"]),
    "H wrong shape": (_set(["fix", "H"], [[1, 0, 0], [0, 1, 0]]), ["fix.H", "3x3"]),
    "footprint wrong shape": (_set(["fix", "footprint"], [[0, 0], [1, 1]]), ["fix.footprint", "4x2"]),
    "fix missing inliers": (lambda d: d["fix"].pop("inliers"), ["missing", "fix.inliers"]),
    "fix unknown key": (_set(["fix", "colour"], "red"), ["unknown", "fix.colour"]),
    "accepted with reason": (_set(["fix", "reject_reason"], "mirrored"), ["fix", "reject_reason"]),
    "inlier_ratio above 1": (_set(["fix", "inlier_ratio"], 1.5), ["fix", "inlier_ratio"]),
    "fix not an object": (_set(["fix"], [1, 2]), ["fix", "object"]),
    "unknown top-level key": (_set(["sigma"], 3.0), ["unknown", "'sigma'", "extra"]),
    "extra not an object": (_set(["extra"], [1]), ["extra", "object"]),
    "interpolated not bool": (_set(["interpolated"], "yes"), ["interpolated", "bool"]),
    "negative ms": (_set(["ms"], -3.0), ["ms"]),
    "source not a string": (_set(["source"], 7), ["source", "str"]),
    "nan x": (_set(["x"], math.nan), ["x", "finite"]),
    "inf in cov": (_set(["cov"], [[math.inf, 0], [0, 1]]), ["cov", "finite"]),
    # an integer too large for a float64 (JSON allows any number of digits) must not escape as OverflowError
    "huge int x": (_set(["x"], 10**400), ["x", "finite"]),
    "huge int in H": (_set(["fix", "H"], [[10**400, 0, 0], [0, 1, 0], [0, 0, 1]]), ["fix.H", "finite"]),
}


@pytest.mark.parametrize("case", MALFORMED)
def test_malformed_dict_is_rejected_with_a_clear_message(case):
    mutate, words = MALFORMED[case]
    d = _valid()
    mutate(d)
    with pytest.raises(TelemetryError) as e:
        record_from_dict(d)
    msg = str(e.value)
    for w in words:
        assert w in msg, f"{w!r} not in {msg!r}"


def test_a_record_that_is_not_an_object_is_rejected():
    with pytest.raises(TelemetryError, match="object"):
        record_from_dict([1, 2, 3])


def test_telemetry_error_is_a_value_error():
    # callers that already catch ValueError (bad input) keep working
    assert issubclass(TelemetryError, ValueError)


def test_read_names_the_line_of_a_bad_record(tmp_path):
    path = tmp_path / "run.jsonl"
    good = json.dumps(_valid())
    bad = _valid()
    bad["fix"]["H"] = [[1, 0, 0], [0, 1, 0]]
    path.write_text(good + "\n" + good + "\n" + json.dumps(bad) + "\n")
    with pytest.raises(TelemetryError, match=r"line 3.*fix\.H"):
        read_run(path)


def test_read_names_the_line_of_invalid_json(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps(_valid()) + "\n{\"idx\": 1, \n")
    with pytest.raises(TelemetryError, match=r"line 2.*JSON"):
        read_run(path)


def test_read_rejects_nan_tokens(tmp_path):
    # Python's json accepts NaN by default, but a browser's JSON.parse does not: the replay viewer would break
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps({**_valid(), "x": "@X@"}).replace('"@X@"', "NaN") + "\n")
    assert '"x": NaN' in path.read_text()
    with pytest.raises(TelemetryError, match=r"line 1.*(finite|NaN)"):
        read_run(path)


def test_writing_a_non_finite_value_fails_and_names_the_field(tmp_path):
    track = TrackState(idx=0, state="LOST", x=math.nan, y=0.0, vx=0.0, vy=0.0, cov=np.eye(2))
    path = tmp_path / "run.jsonl"
    with pytest.raises(TelemetryError, match=r"\bx\b.*finite"):
        write_run(path, [Record(idx=0, t=0.0, source="a.png", track=track)])


@pytest.mark.parametrize("extra", [{"pt": (1, 2)}, {5: "a"}, {"a": np.float64(1.0)}, {"a": math.inf}])
def test_extra_that_json_would_change_or_reject_fails_on_write(tmp_path, extra):
    # a tuple comes back as a list and an int key as a string: the round trip would silently not be equal
    with pytest.raises(TelemetryError, match="extra"):
        write_run(tmp_path / "run.jsonl", [_record(0, extra=extra)])


def test_record_to_dict_does_not_share_extra_with_the_record():
    rec = _record(0, extra={"truth": {"x": 1.0}})
    record_to_dict(rec)["extra"]["truth"]["x"] = 99.0
    assert rec.extra == {"truth": {"x": 1.0}}


@pytest.mark.parametrize("order", [[3, 1], [1, 1]], ids=["backwards", "duplicate"])
def test_writer_refuses_records_out_of_frame_order(tmp_path, order):
    with pytest.raises(TelemetryError, match="order"):
        write_run(tmp_path / "run.jsonl", [_record(i) for i in order])


def test_writer_allows_gaps_in_idx(tmp_path):
    # a video source keeping every N-th frame skips indices; order is what matters
    write_run(tmp_path / "run.jsonl", [_record(0), _record(10), _record(20)])
    assert [r.idx for r in read_run(tmp_path / "run.jsonl")] == [0, 10, 20]


@pytest.mark.parametrize("order", [[3, 1], [1, 1]], ids=["backwards", "duplicate"])
def test_reader_refuses_records_out_of_frame_order(tmp_path, order):
    path = tmp_path / "run.jsonl"
    path.write_text("".join(json.dumps(record_to_dict(_record(i))) + "\n" for i in order))
    with pytest.raises(TelemetryError, match=r"line 2.*order"):
        read_run(path)


def test_time_must_not_go_backwards(tmp_path):
    a, b = _record(0), _record(1)
    b = Record(idx=1, t=a.t - 1.0, source=b.source, track=b.track)
    with pytest.raises(TelemetryError, match=r"\bt\b.*order"):
        write_run(tmp_path / "run.jsonl", [a, b])


def test_an_incomplete_last_line_says_so(tmp_path):
    # live mode can catch the writer mid-line; a crashed run ends the same way
    path = tmp_path / "run.jsonl"
    line = json.dumps(record_to_dict(_record(0)))
    path.write_text(line + "\n" + line[:40])
    with pytest.raises(TelemetryError, match=r"line 2.*incomplete"):
        read_run(path)


def test_record_checks_that_its_parts_belong_to_the_same_frame():
    track = TrackState(idx=4, state="TRACKING", x=0.0, y=0.0, vx=0.0, vy=0.0, cov=np.eye(2))
    with pytest.raises(TelemetryError, match="idx"):
        Record(idx=5, t=0.0, source="a.png", track=track)
    with pytest.raises(TelemetryError, match="idx"):
        Record(idx=4, t=0.0, source="a.png", track=TrackState(**{**track.__dict__, "fix": _fix(3)}))


def test_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_run(tmp_path / "nope.jsonl")
