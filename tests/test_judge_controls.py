"""M7-05 · judge controls: orders that reach the simulator and the engine while a live flight runs.

Done when: "each button visibly changes the tracker's behaviour within a few seconds". The engine's half is tested
here, with an injected clock (a flight runs in no time, and "seconds" are seconds of flight); the buttons' half (the
orders the console writes, and what it draws) is in ui/tests/judge-controls.test.ts(x). The orders are
``control.json`` in the run folder (see ``dronenav.live``), read before every frame:

1. **Spoof GPS** ``{"spoof": true}``: the simulated receiver walks away from the true position; the spoof detector
   (M5-07) raises its alarm within a few seconds and drops it after the spoof is switched off. The tracker flies on
   the camera, so its position does not move with the receiver.
2. **Map year** ``{"map": "<name>"}``: from the next frame on the engine searches another map of the same place.
   A map of another place loses the track within three frames (the tracker's ``lost_after``), and the first map
   brings it back. An unknown name changes nothing; a map of another size is refused at the start.
3. **New flight** ``{"flight": {"name", "speed", "id"}}``: the run starts again in the same folder with the other
   flight at the ordered speed. One order (one ``id``) restarts once.
4. **Jam camera** is M7-04's (tests/test_live.py); here: the orders live side by side in one file, an order that
   names one switch leaves the others as they were, and half a file changes nothing.
5. ``scripts/live.py`` finds the other years of the map and the other flights next to the ones it was given.
"""
import json
import math
import shutil

import cv2
import numpy as np
import pytest
from tests.test_live import FPS, FakeClock, _cli, _flight, _source
from tests.test_run import _city

from dronenav.live import CONTROL, STATUS, live_run
from dronenav.telemetry import read_run, read_run_info
from dronenav.types import TrackMode


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Three maps of one size: a town in 2022, the same town in 2017 (darker, hazier, a few buildings fewer), and
    another place altogether. And a bigger image that cannot stand in for them."""
    d = tmp_path_factory.mktemp("judge_world")
    img = _city(2000, 1500)
    old = cv2.GaussianBlur(cv2.convertScaleAbs(img, alpha=0.8, beta=12), (0, 0), 0.7)
    rng = np.random.default_rng(7)
    for _ in range(25):                                  # buildings that were not there yet: plain ground
        x, y = int(rng.uniform(0, 1960)), int(rng.uniform(0, 1460))
        old[y:y + 30, x:x + 30] = 95
    cv2.imwrite(str(d / "town_2022.png"), img)
    cv2.imwrite(str(d / "town_2017.png"), old)
    cv2.imwrite(str(d / "elsewhere.png"), _city(2000, 1500, seed=11))
    cv2.imwrite(str(d / "bigger.png"), _city(2200, 1500, seed=3))
    return d / "town_2022.png", img, d / "cache"


def _maps(world, *names):
    d = world[0].parent
    files = {"2022": "town_2022.png", "2017": "town_2017.png", "elsewhere": "elsewhere.png", "bigger": "bigger.png"}
    return {n: d / files[n] for n in names}


def _live(world, out, source, **kw):
    clock = FakeClock()
    return live_run(world[0], source, out, cache_dir=world[2], clock=clock, sleep=clock.sleep, **kw)


def _order(out, **order):
    (out / CONTROL).write_text(json.dumps(order))


def _status(out):
    return json.loads((out / STATUS).read_text())


def _err(rec, src):
    """Distance from the record's position to the true one, px."""
    tr = src.truth(rec.idx)
    return math.hypot(rec.track.x - tr.center[0], rec.track.y - tr.center[1])


# --- 1. spoof GPS ----------------------------------------------------------------------------------------------

def test_an_honest_receiver_agrees_with_the_camera(tmp_path, world):
    out, src = tmp_path / "out", _source(world, n=10)
    _live(world, out, src)
    recs = read_run(out / "run.jsonl")
    assert all("gps" in r.extra for r in recs)           # a simulated flight always has a receiver to check
    for r in recs[1:]:
        g = r.extra["gps"]
        tr = src.truth(r.idx)
        # the receiver's noise is 4 px per axis: 20 px is 5 sigma, never reached in ten frames
        assert math.hypot(g["x"] - tr.center[0], g["y"] - tr.center[1]) < 20, r.idx
        assert g["alarm"] is False and g["state"] in ("OK", "SUSPECT"), r.idx
    assert _status(out)["spoof"] is False


def test_spoofing_raises_the_alarm_within_seconds_and_stopping_it_clears_it(tmp_path, world):
    out, src = tmp_path / "out", _source(world, n=30)
    in_status = {}

    def operator(rec):
        in_status[rec.idx] = _status(out)["spoof"]
        if rec.idx == 3:
            _order(out, spoof=True)
        if rec.idx == 17:
            _order(out, spoof=False)

    _live(world, out, src, on_record=operator)
    recs = read_run(out / "run.jsonl")
    assert [i for i, on in in_status.items() if on] == list(range(4, 18))    # the status follows the order
    gps = {r.idx: r.extra["gps"] for r in recs}
    truth = {r.idx: src.truth(r.idx).center for r in recs}
    off = {i: math.hypot(g["x"] - truth[i][0], g["y"] - truth[i][1]) for i, g in gps.items()}
    # the walk-off starts at the true position and only grows: no jump to notice (8 px per second of flight by
    # default, so 4 px a frame here, on top of the receiver's 4 px noise)
    assert off[4] < 20
    assert off[17] > 40 and all(off[i + 4] > off[i] for i in range(4, 14))
    # "within a few seconds": the alarm is on at most 6 s of flight after the order (12 frames at 2 fps). The gap
    # must first leave the noise (about 16 px: 2 s), then stay out for the detector's 5 frames in a row (2 s more).
    first = next(i for i in sorted(gps) if gps[i]["alarm"])
    assert 4 < first <= 4 + 12, first
    assert all(not gps[i]["alarm"] for i in range(4))
    assert gps[17]["alarm"] and gps[17]["state"] == "SPOOFED" and gps[17]["gap_px"] > 40
    # switched off, the receiver is honest again from the next frame, and the alarm drops after the detector's 5
    # good frames in a row: 2.5 s, well within the same 6 s
    assert off[18] < 20
    cleared = next(i for i in range(18, 30) if not gps[i]["alarm"])
    assert cleared <= 18 + 12, cleared
    assert not gps[29]["alarm"] and gps[29]["state"] == "OK"
    # the tracker flies on the camera: the receiver's lie does not move it (2 px, as in tests/test_live.py)
    for r in recs[2:]:
        assert r.track.state is TrackMode.TRACKING and _err(r, src) <= 2.0, r.idx


def test_the_walk_off_speed_is_a_setting(tmp_path, world):
    out, src = tmp_path / "out", _source(world, n=12)
    _live(world, out, src, spoof_speed=40.0, on_record=lambda rec: rec.idx == 1 and _order(out, spoof=True))
    last = read_run(out / "run.jsonl")[-1]
    tr = src.truth(last.idx)
    g = last.extra["gps"]
    # ten frames = 4.5 s of spoofing at 40 px/s: 180 px away, give or take the noise
    assert math.hypot(g["x"] - tr.center[0], g["y"] - tr.center[1]) == pytest.approx(180, abs=20)
    with pytest.raises(ValueError, match="spoof"):
        _live(world, tmp_path / "bad", _source(world, n=2), spoof_speed=0)
    assert not (tmp_path / "bad").exists()


# --- 2. map year -----------------------------------------------------------------------------------------------

def test_the_map_is_switched_while_the_flight_runs(tmp_path, world):
    out, src = tmp_path / "out", _source(world, n=20)
    maps = _maps(world, "2017", "2022", "elsewhere")
    seen = {}

    def operator(rec):
        seen[rec.idx] = (_status(out)["map"], read_run_info(out).map_path.name)
        if rec.idx == 3:
            _order(out, map="2017")
        if rec.idx == 7:
            _order(out, map="elsewhere")
        if rec.idx == 13:
            _order(out, map="2022")

    _live(world, out, src, maps=maps, on_record=operator)
    recs = read_run(out / "run.jsonl")
    by = {r.idx: r for r in recs}
    assert [r.extra["map"] for r in recs] == ["2022"] * 4 + ["2017"] * 4 + ["elsewhere"] * 6 + ["2022"] * 6
    # the status and run.json (the map the console draws) say the same as the records, frame by frame
    assert seen[3] == ("2022", "town_2022.png") and seen[4] == ("2017", "town_2017.png")
    assert seen[8] == ("elsewhere", "elsewhere.png") and seen[14] == ("2022", "town_2022.png")
    st = _status(out)
    assert st["maps"] == ["2017", "2022", "elsewhere"] and st["map"] == "2022"
    # another year of the same place: the track goes on (3 px: the older picture is darker and hazier)
    for i in range(4, 8):
        assert by[i].track.state is TrackMode.TRACKING and by[i].track.fix.accepted and _err(by[i], src) <= 3.0, i
    # a map of another place: no fix on it, and the tracker says LOST after its three misses in a row
    for i in range(8, 14):
        assert by[i].track.fix is None or not by[i].track.fix.accepted, i
    assert by[10].track.state is TrackMode.LOST and by[13].track.state is TrackMode.LOST
    # back on the right map: found again within a few frames (the first fix is a candidate, the second confirms)
    assert by[14].track.fix.accepted
    assert by[16].track.state is TrackMode.TRACKING and _err(by[19], src) <= 2.0


def test_a_map_that_is_not_offered_changes_nothing(tmp_path, world):
    out, src = tmp_path / "out", _source(world, n=6)
    _live(world, out, src, maps=_maps(world, "2017", "2022"),
          on_record=lambda rec: rec.idx == 1 and _order(out, map="1999"))
    recs = read_run(out / "run.jsonl")
    assert [r.extra["map"] for r in recs] == ["2022"] * 6 and _status(out)["map"] == "2022"
    assert all(r.track.fix.accepted for r in recs)


def test_without_other_maps_the_one_map_is_the_only_choice(tmp_path, world):
    out = tmp_path / "out"
    _live(world, out, _source(world, n=2))
    st = _status(out)
    assert st["maps"] == [st["map"]] == ["town_2022"]    # named after its file


def test_a_map_of_another_size_is_refused_before_anything_is_written(tmp_path, world):
    with pytest.raises(ValueError, match="size"):
        _live(world, tmp_path / "out", _source(world, n=2), maps=_maps(world, "2022", "bigger"))
    assert not (tmp_path / "out").exists()


# --- 3. new flight ---------------------------------------------------------------------------------------------

def test_a_new_flight_starts_the_run_again_at_the_ordered_speed(tmp_path, world):
    out = tmp_path / "out"
    a, b = _source(world, n=8, seed=3), _source(world, n=5, seed=5)
    flights = {"mild/2022": lambda: a, "wild/2022": lambda: b}
    before = []

    def operator(rec):
        if not before and rec.idx == 2:
            before.append(_status(out))
            _order(out, flight={"name": "wild/2022", "speed": 2.0, "id": 1})

    s = _live(world, out, a, flights=flights, flight="mild/2022", on_record=operator)
    assert before[0]["flight"] == "mild/2022" and before[0]["flight_no"] == 0 and before[0]["speed"] == 1.0
    assert before[0]["flights"] == ["mild/2022", "wild/2022"]
    recs = read_run(out / "run.jsonl")
    # the file is the new flight only, from its first frame; the order stays in control.json and restarts once
    assert [r.idx for r in recs] == [0, 1, 2, 3, 4] and s.frames == 5
    assert [r.extra["wall_s"] for r in recs] == pytest.approx([r.t / 2.0 for r in recs], abs=1e-6)
    st = _status(out)
    assert (st["flight"], st["flight_no"], st["speed"], st["frames"], st["state"]) == ("wild/2022", 1, 2.0, 5, "done")
    # a new track, on the new flight's route (2 px, as above); the first fix is a candidate again
    assert recs[0].track.state is TrackMode.RELOCALIZING and recs[1].track.state is TrackMode.TRACKING
    for r in recs:
        assert r.track.fix.accepted and _err(r, b) <= 2.0, r.idx
        assert r.extra["gps"]["alarm"] is False


def test_the_same_flight_can_be_started_again(tmp_path, world):
    out = tmp_path / "out"
    made = []

    def again():
        made.append(_source(world, n=4))
        return made[-1]

    done = []

    def operator(rec):
        if not done and rec.idx == 2:
            done.append(1)
            _order(out, flight={"name": "mild/2022", "speed": 4, "id": 7})

    _live(world, out, again(), flights={"mild/2022": again}, flight="mild/2022", on_record=operator)
    assert len(made) == 2
    assert [r.idx for r in read_run(out / "run.jsonl")] == [0, 1, 2, 3] and _status(out)["flight_no"] == 1


@pytest.mark.parametrize("order", [{"name": "nope", "speed": 2, "id": 1}, {"name": "mild/2022", "speed": 0, "id": 1},
                                   {"name": "mild/2022", "speed": 2}, {"name": "mild/2022", "speed": 2, "id": "x"},
                                   "mild/2022"])
def test_a_flight_order_that_cannot_be_flown_changes_nothing(tmp_path, world, order):
    out, src = tmp_path / "out", _source(world, n=5)
    _live(world, out, src, flights={"mild/2022": lambda: _source(world, n=5)}, flight="mild/2022",
          on_record=lambda rec: rec.idx == 1 and _order(out, flight=order))
    assert [r.idx for r in read_run(out / "run.jsonl")] == [0, 1, 2, 3, 4]
    assert _status(out)["flight_no"] == 0


def test_a_flight_must_be_named_among_the_flights(tmp_path, world):
    with pytest.raises(ValueError, match="flight"):
        _live(world, tmp_path / "out", _source(world, n=2), flights={"a": lambda: None}, flight="b")
    assert not (tmp_path / "out").exists()


# --- 4. the orders live side by side ---------------------------------------------------------------------------

def test_an_order_for_one_switch_leaves_the_others_and_half_a_file_changes_nothing(tmp_path, world):
    out, src = tmp_path / "out", _source(world, n=10)

    def operator(rec):
        if rec.idx == 1:
            _order(out, jam=True, spoof=True)
        if rec.idx == 3:
            _order(out, jam=False)                       # says nothing of the spoof: it stays on
        if rec.idx == 5:
            (out / CONTROL).write_text('{"spoof": fal')  # caught halfway: the last orders stand
        if rec.idx == 7:
            _order(out, spoof=False, jam=False, map=3, flight=None)   # nonsense beside good orders is skipped

    seen = {}
    _live(world, out, src, on_record=lambda rec: (seen.update({rec.idx: _status(out)}), operator(rec)))
    recs = read_run(out / "run.jsonl")
    assert [r.idx for r in recs if r.extra.get("jammed")] == [2, 3]
    assert [i for i, st in seen.items() if st["spoof"]] == [2, 3, 4, 5, 6, 7]
    assert len(recs) == 10 and _status(out)["flight_no"] == 0


# --- 5. the command --------------------------------------------------------------------------------------------

def _pack(tmp_path, world):
    """The layout of the Irpin pack: maps/<place>_<year>.png, flights/<preset>/<year>/."""
    d = tmp_path / "pack"
    (d / "maps").mkdir(parents=True)
    for name in ("town_2017.png", "town_2022.png"):
        shutil.copy(world[0].parent / name, d / "maps" / name)
    (d / "maps" / "town_2022.json").write_text("{}")    # a sidecar is not a map
    for preset in ("mild", "wild"):
        (d / "flights" / preset).mkdir(parents=True)
        shutil.move(_flight(tmp_path / preset, world, n=4), d / "flights" / preset / "2022")
    (d / "flights" / "notes").mkdir()                    # a folder that is not a flight
    return d


def test_the_command_offers_the_other_years_and_the_other_flights(tmp_path, world):
    pack, out = _pack(tmp_path, world), tmp_path / "runs" / "live"
    res = _cli("--flight", pack / "flights" / "mild" / "2022", "--map", pack / "maps" / "town_2022.png", "--out", out,
               "--cache", world[2], "--keyframes", "--rate", FPS, "--speed", 4)
    assert res.returncode == 0, res.stderr
    st = _status(out)
    assert st["maps"] == ["2017", "2022"] and st["map"] == "2022"
    assert st["flights"] == ["mild/2022", "wild/2022"] and st["flight"] == "mild/2022" and st["speed"] == 4.0
    assert all(r.extra["map"] == "2022" and "gps" in r.extra for r in read_run(out / "run.jsonl"))


def test_the_command_takes_the_maps_and_flights_it_is_told(tmp_path, world):
    pack, out = _pack(tmp_path, world), tmp_path / "runs" / "live"
    d = world[0].parent
    res = _cli("--flight", pack / "flights" / "mild" / "2022", "--map", d / "town_2022.png", "--out", out,
               "--cache", world[2], "--keyframes", "--rate", FPS, "--speed", 4,
               "--maps", f"other={d / 'elsewhere.png'}", d / "town_2017.png", "--flights", tmp_path / "none")
    assert res.returncode == 0, res.stderr
    st = _status(out)
    assert st["maps"] == ["2017", "2022", "other"] and st["map"] == "2022"
    assert st["flights"] == ["mild/2022"]                # an empty --flights root: only the flight given
    bad = _cli("--flight", pack / "flights" / "mild" / "2022", "--map", d / "town_2022.png", "--out", tmp_path / "bad",
               "--cache", world[2], "--keyframes", "--maps", d / "bigger.png")
    assert bad.returncode == 2 and "size" in bad.stderr and not (tmp_path / "bad").exists()
    bad = _cli("--flight", pack / "flights" / "mild" / "2022", "--map", d / "town_2022.png", "--out", tmp_path / "bad",
               "--cache", world[2], "--keyframes", "--spoof-speed", "0")
    assert bad.returncode == 2 and not (tmp_path / "bad").exists()
