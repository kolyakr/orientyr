"""M3-01 · Ukraine-wide scan: ranking, area status, retries and AOI bounding boxes.

``experiments/ukraine_scan/scan.py`` ranks pre/war/post capture triples per UNOSAT area. "Done when" needs
every penalty factor and the area-status lookup to be correct and testable, the HTTP retry to actually
retry (now stdlib ``urllib``, no ``requests``), and the per-area AOI box to come from the labels' own
extent (there is no separate AOI-polygon file in our data). None of this touches the network: ``urlopen``
is monkeypatched, and the bbox/penalty/summary functions are pure.
"""
from __future__ import annotations

import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCAN_PATH = ROOT / "experiments" / "ukraine_scan" / "scan.py"
_spec = importlib.util.spec_from_file_location("ukraine_scan_scan", SCAN_PATH)
scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan)


def make_candidate(**over):
    base = {
        "aoi": "UKR2",
        "destroyed_severe": 10,
        "pre": {"date": "2020-05-01"},
        "war": {"date": "2022-03-16", "is_unosat_date": True},
        "max_season_gap_days": 10,
        "same_sensor": True,
        "worst_res": 0.3,
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# penalties()
# ---------------------------------------------------------------------------

def test_penalty_pre_2019_capture():
    assert scan.penalties(make_candidate(pre={"date": "2018-12-31"}))["pre_2019"] == 0.7
    assert scan.penalties(make_candidate(pre={"date": "2019-01-01"}))["pre_2019"] == 1.0


def test_penalty_war_capture_off_unosat_date():
    assert scan.penalties(make_candidate(war={"date": "2022-03-16", "is_unosat_date": False}))["off_date"] == 0.7
    assert scan.penalties(make_candidate(war={"date": "2022-03-16", "is_unosat_date": True}))["off_date"] == 1.0


def test_penalty_season_gap_bounds():
    # 0 days apart: no penalty. 120+ days apart: the penalty is capped at its worst, x0.5 (spec: "x0.5 to x1").
    assert scan.penalties(make_candidate(max_season_gap_days=0))["season_gap"] == pytest.approx(1.0)
    assert scan.penalties(make_candidate(max_season_gap_days=120))["season_gap"] == pytest.approx(0.5)
    assert scan.penalties(make_candidate(max_season_gap_days=500))["season_gap"] == pytest.approx(0.5)


def test_penalty_mixed_sensor():
    assert scan.penalties(make_candidate(same_sensor=True))["mixed_sensor"] == 1.0
    assert scan.penalties(make_candidate(same_sensor=False))["mixed_sensor"] == 0.85


def test_penalty_resolution():
    assert scan.penalties(make_candidate(worst_res=0.6))["resolution"] == 1.0
    assert scan.penalties(make_candidate(worst_res=0.61))["resolution"] == 0.8


def test_penalty_status(monkeypatch):
    monkeypatch.setitem(scan.STATUS, "UKRTEST", "occupied")
    assert scan.penalties(make_candidate(aoi="UKRTEST"))["status"] == 0.2
    monkeypatch.setitem(scan.STATUS, "UKRTEST", "front_line")
    assert scan.penalties(make_candidate(aoi="UKRTEST"))["status"] == 0.5
    monkeypatch.setitem(scan.STATUS, "UKRTEST", "none")
    assert scan.penalties(make_candidate(aoi="UKRTEST"))["status"] == 1.0


def test_score_multiplies_every_named_penalty():
    c = make_candidate(destroyed_severe=10, same_sensor=False, worst_res=0.9)
    expected = 10
    for factor in scan.penalties(c).values():
        expected *= factor
    assert scan.score(c) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# STATUS / status_for()
# ---------------------------------------------------------------------------

def test_status_covers_every_unosat_area():
    assert set(scan.STATUS) == {f"UKR{i}" for i in range(1, 19)}
    assert set(scan.STATUS.values()) <= {"occupied", "front_line", "none"}


def test_status_for_unknown_area_is_an_error_not_safe():
    with pytest.raises(ValueError):
        scan.status_for("UKR99")


def test_status_for_known_area_matches_status_table():
    aoi = next(iter(scan.STATUS))
    assert scan.status_for(aoi) == scan.STATUS[aoi]


# ---------------------------------------------------------------------------
# AOI bounding box from the labels' own extent (no AOI-polygon file in our data)
# ---------------------------------------------------------------------------

def test_bbox_from_points_adds_the_margin_in_degrees():
    pts = [(30.0, 50.0), (30.2, 50.2)]
    lon0, lat0, lon1, lat1 = scan.bbox_from_points(pts, margin_km=4.0)
    # Ground truth: flat-earth deg/km approximation at this latitude (same one scan.py itself uses),
    # not a geodesic library -- so the tolerance is generous (1e-3 deg, ~100 m) rather than exact.
    assert lon0 == pytest.approx(30.0 - 4.0 / (111.0 * 0.6408), abs=1e-3)
    assert lat0 == pytest.approx(50.0 - 4.0 / 111.0, abs=1e-3)
    assert lon1 == pytest.approx(30.2 + 4.0 / (111.0 * 0.6408), abs=1e-3)
    assert lat1 == pytest.approx(50.2 + 4.0 / 111.0, abs=1e-3)


def test_bbox_from_points_rejects_empty_input():
    with pytest.raises(ValueError):
        scan.bbox_from_points([], margin_km=4.0)


# ---------------------------------------------------------------------------
# HTTP retry (stdlib urllib, no requests): retries on 429/5xx, not on other errors
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload


def _http_error(code):
    return urllib.error.HTTPError("https://example.test", code, "err", {}, None)


def test_fetch_json_retries_on_429_then_500_then_succeeds(monkeypatch):
    calls = iter([_http_error(429), _http_error(500), _FakeResponse({"ok": True})])

    def fake_urlopen(req, timeout=None):
        result = next(calls)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(scan, "urlopen", fake_urlopen)
    monkeypatch.setattr(scan.time, "sleep", lambda s: None)
    assert scan._fetch_json("https://example.test", {}, timeout=5) == {"ok": True}


def test_fetch_json_does_not_retry_non_retryable_error(monkeypatch):
    def fake_urlopen(req, timeout=None):
        raise _http_error(404)

    monkeypatch.setattr(scan, "urlopen", fake_urlopen)
    monkeypatch.setattr(scan.time, "sleep", lambda s: (_ for _ in ()).throw(AssertionError("must not back off")))
    with pytest.raises(urllib.error.HTTPError):
        scan._fetch_json("https://example.test", {}, timeout=5)


# ---------------------------------------------------------------------------
# candidates_summary.json: best candidate per (aoi, size_km), nothing else
# ---------------------------------------------------------------------------

def test_summarize_keeps_only_the_best_score_per_aoi_and_size():
    cands = [
        {"aoi": "UKR1", "size_km": 3.0, "score": 5.0},
        {"aoi": "UKR1", "size_km": 3.0, "score": 9.0},
        {"aoi": "UKR1", "size_km": 5.0, "score": 2.0},
        {"aoi": "UKR2", "size_km": 3.0, "score": 1.0},
    ]
    assert scan.summarize(cands) == [
        {"aoi": "UKR1", "size_km": 3.0, "score": 9.0},
        {"aoi": "UKR1", "size_km": 5.0, "score": 2.0},
        {"aoi": "UKR2", "size_km": 3.0, "score": 1.0},
    ]


def test_summarize_of_no_candidates_is_empty():
    assert scan.summarize([]) == []
