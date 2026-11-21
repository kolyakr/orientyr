"""M9-01 · MAVLink ``GPS_INPUT`` stub.

Done when: a run produces a valid message log; README and UI call it a stub, not flight-tested.

The contract:
- one ``GPS_INPUT`` per *accepted, measured* fix (a rejected fix, a frame with no fix and an interpolated frame send
  nothing: the autopilot's own filter coasts, as it would on a GPS dropout). The fix, not the smoothed track, is
  sent: the autopilot fuses it itself, and a pre-filtered position would be trusted twice;
- lat/lon from the map's georeference at the fix's centre, ``horiz_accuracy`` = the fix's σ (M4-08) in metres;
  no altitude or velocity (their ignore flags set), 2D fix;
- time: ``time_usec`` is UNIX µs of the frame (start + t), ``time_week``/``time_week_ms`` the GPS time (UTC + 18 s
  leap seconds since 1980-01-06);
- the log is a ``.tlog`` (8-byte big-endian µs timestamp + MAVLink 2 packet) that pymavlink reads back with no bad
  data, starting with a STATUSTEXT that calls it a stub, not flight-tested; the same messages can go out over UDP
  to a SITL simulator;
- ``scripts/mavlink_log.py --run DIR`` writes it; a map with no georeference is refused with a clear message.
"""
import datetime as dt
import json
import math
import socket
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from pymavlink import mavutil
from pymavlink.dialects.v20 import common as mav

from dronenav.confidence import fix_sigma
from dronenav.georef import GeorefBlock
from dronenav.mavlink import STUB_TEXT, gps_input, gps_time, run_to_tlog
from dronenav.telemetry import Record, read_run
from dronenav.types import Fix, TrackState

ROOT = Path(__file__).resolve().parents[1]
START = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
# Irpin 2025's own block (data/irpin/maps/irpin_2025.json), so the numbers are the real map's
BLOCK = {"file": "maps/irpin_2025.png", "zoom": 17, "m_per_px": 0.759, "size_px": (3936, 3936),
         "merc_origin_px": (19592192, 11301728), "bounds_lonlat": (30.2014, 50.5086, 30.2436, 50.5355),
         "capture_coverage": 1.0, "alignment_to_2025": {"reference": True}}


def _block(**kw) -> GeorefBlock:
    return GeorefBlock(**{**BLOCK, **kw})


def _fix(idx, x, y, accepted=True) -> Fix:
    fp = np.array([(-50, -40), (50, -40), (50, 40), (-50, 40)], float) + (x, y)
    return Fix(idx=idx, H=np.eye(3), x=x, y=y, heading_deg=10.0, scale=1.0, footprint=fp, inliers=120,
               inlier_ratio=0.6, reproj_px=1.5, accepted=accepted,
               reject_reason=None if accepted else "too few inliers")


def _rec(idx, t, fix=None, interpolated=False) -> Record:
    x, y = (fix.x, fix.y) if fix is not None else (1000.0, 1000.0)
    # the track deliberately elsewhere: what is sent must be the fix
    track = TrackState(idx=idx, state="TRACKING", x=x + 7.0, y=y - 5.0, vx=0.0, vy=0.0, cov=np.eye(2) * 25.0,
                       fix=fix)
    return Record(idx=idx, t=t, source=f"f{idx}.png", track=track, interpolated=interpolated)


# --- one message ------------------------------------------------------------------------------------------------

def test_only_accepted_measured_fixes_are_sent():
    b = _block()
    assert gps_input(_rec(0, 0.0, _fix(0, 1200.0, 900.0)), b, START) is not None
    assert gps_input(_rec(1, 1.0, None), b, START) is None                                  # no fix
    assert gps_input(_rec(2, 2.0, _fix(2, 1200.0, 900.0, accepted=False)), b, START) is None  # rejected
    assert gps_input(_rec(3, 3.0, _fix(3, 1200.0, 900.0), interpolated=True), b, START) is None


def test_position_and_accuracy_come_from_the_fix():
    b = _block()
    f = _fix(5, 1234.5, 2345.6)
    m = gps_input(_rec(5, 5.0, f), b, START)
    lon, lat = b.px_to_lonlat(1234.5, 2345.6)
    # degE7 integers: 1e-7 deg is ~1 cm, far below a pixel (0.76 m)
    assert abs(m.lat - lat * 1e7) <= 1 and abs(m.lon - lon * 1e7) <= 1
    assert 50.5086 < m.lat / 1e7 < 50.5355 and 30.2014 < m.lon / 1e7 < 30.2436   # inside Irpin
    assert m.horiz_accuracy == pytest.approx(fix_sigma(f) * 0.759, rel=1e-6)     # float32 on the wire
    assert m.fix_type == 2                                                          # 2D: no altitude
    for flag in ("ALT", "VEL_HORIZ", "VEL_VERT", "SPEED_ACCURACY", "VERTICAL_ACCURACY", "HDOP", "VDOP"):
        assert m.ignore_flags & getattr(mav, f"GPS_INPUT_IGNORE_FLAG_{flag}"), flag
    assert not m.ignore_flags & mav.GPS_INPUT_IGNORE_FLAG_HORIZONTAL_ACCURACY


def test_a_less_certain_fix_claims_a_worse_accuracy():
    b = _block()
    good = _fix(0, 1000.0, 1000.0)
    bad = Fix(**{**good.__dict__, "inliers": 15, "inlier_ratio": 0.15, "reproj_px": 4.0})
    acc = [gps_input(_rec(0, 0.0, f), b, START).horiz_accuracy for f in (good, bad)]
    assert acc[1] > acc[0]


def test_gps_time():
    # 2025-01-01 00:00:00 UTC is a Wednesday of GPS week 2347; 3 days + 18 leap seconds into the week
    assert gps_time(START) == (2347, (3 * 86400 + 18) * 1000)
    m = gps_input(_rec(7, 12.5, _fix(7, 1000.0, 1000.0)), _block(), START)
    assert m.time_usec == int((START.timestamp() + 12.5) * 1e6)
    assert (m.time_week, m.time_week_ms) == (2347, (3 * 86400 + 18) * 1000 + 12500)


# --- the log ----------------------------------------------------------------------------------------------------

def _records(n=10):
    recs = []
    for k in range(n):
        if k == 3:
            fix = None
        elif k == 6:
            fix = _fix(k, 900.0 + 20 * k, 1500.0 - 9 * k, accepted=False)
        else:
            fix = _fix(k, 900.0 + 20 * k, 1500.0 - 9 * k)
        recs.append(_rec(k, 0.5 * k, fix, interpolated=k == 8))
    return recs                                                     # 7 accepted, measured fixes: 0-2, 4, 5, 7, 9


def _read_tlog(path):
    log = mavutil.mavlink_connection(str(path), dialect="common")
    msgs = []
    while (m := log.recv_match()) is not None:
        msgs.append(m)
    return msgs


def test_the_log_reads_back_with_no_bad_data_and_says_stub(tmp_path):
    out = tmp_path / "gps_input.tlog"
    n = run_to_tlog(_records(), _block(), out, START)
    assert n == 7
    msgs = _read_tlog(out)
    assert [m.get_type() for m in msgs] == ["STATUSTEXT"] + ["GPS_INPUT"] * 7     # BAD_DATA would show here
    assert msgs[0].text == STUB_TEXT and "stub" in STUB_TEXT and "not flight-tested" in STUB_TEXT
    sent = [m.time_usec for m in msgs[1:]]
    want = [int((START.timestamp() + 0.5 * k) * 1e6) for k in (0, 1, 2, 4, 5, 7, 9)]
    assert sent == want
    # the tlog's own timestamps (what a ground station replays by) are the frames' times
    assert [m._timestamp for m in msgs[1:]] == pytest.approx([w / 1e6 for w in want], abs=1e-5)


def test_the_messages_also_go_out_over_udp(tmp_path):
    # a SITL simulator listens on UDP: here a local socket stands in for it
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(5.0)
    port = rx.getsockname()[1]
    try:
        run_to_tlog(_records(), _block(), tmp_path / "g.tlog", START, udp=f"127.0.0.1:{port}")
        parser = mav.MAVLink(None)
        got = []
        while len(got) < 8:
            for m in parser.parse_buffer(rx.recv(4096)) or []:
                got.append(m.get_type())
    finally:
        rx.close()
    assert got == ["STATUSTEXT"] + ["GPS_INPUT"] * 7


# --- the command ------------------------------------------------------------------------------------------------

def _map_with_sidecar(folder: Path, georef=True) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "irpin_test.png"
    rng = np.random.default_rng(0)
    cv2.imwrite(str(path), rng.integers(0, 255, (600, 800, 3), dtype=np.uint8))
    if georef:
        block = {**BLOCK, "size_px": [800, 600]}
        block["bounds_lonlat"] = list(block["bounds_lonlat"])
        block["merc_origin_px"] = list(block["merc_origin_px"])
        (folder / "irpin_test.json").write_text(json.dumps(
            {"capture": {"date": "2025-01-01", "sensor": "test"}, "source": "test", "map": block, "hires": block}))
    return path


def _script(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "mavlink_log.py"), *map(str, args)],
                          check=False, capture_output=True, text=True, cwd=ROOT, timeout=120)


def _fake_run(map_path, out):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "fake_run.py"), "--map", str(map_path),
                        "--out", str(out), "--frames", "30", "--frame-size", "160x120"],
                       check=False, capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr


def test_a_run_produces_a_valid_message_log(tmp_path):
    run = tmp_path / "run"
    _fake_run(_map_with_sidecar(tmp_path / "maps"), run)
    r = _script("--run", run, "--utc-start", "2025-01-01T00:00:00Z")
    assert r.returncode == 0, r.stderr
    assert "stub" in r.stdout and "not flight-tested" in r.stdout
    accepted = [rec for rec in read_run(run / "run.jsonl")
                if rec.track.fix is not None and rec.track.fix.accepted and not rec.interpolated]
    assert accepted, "the fake run has accepted fixes"
    msgs = _read_tlog(run / "gps_input.tlog")
    assert [m.get_type() for m in msgs] == ["STATUSTEXT"] + ["GPS_INPUT"] * len(accepted)
    for m in msgs[1:]:
        assert 50.5 < m.lat / 1e7 < 50.6 and 30.2 < m.lon / 1e7 < 30.3
        assert math.isfinite(m.horiz_accuracy) and m.horiz_accuracy > 0


def test_a_map_with_no_georeference_is_refused(tmp_path):
    run = tmp_path / "run"
    _fake_run(_map_with_sidecar(tmp_path / "maps", georef=False), run)
    r = _script("--run", run)
    assert r.returncode == 2
    assert "georeference" in r.stderr
    assert not (run / "gps_input.tlog").exists()


def test_the_readme_calls_it_a_stub():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("MAVLink"):]
    section = section[:section.index("\n## ") if "\n## " in section else len(section)]
    assert "stub" in section and "not flight-tested" in section
    assert "scripts/mavlink_log.py" in section
