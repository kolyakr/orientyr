"""M0-05 · Map georeference sidecar: px <-> lon/lat <-> metres, and a map with no georeference.

The "Done when" items from the issue, each tested here:
1. known points of the Irpin maps (zoom 17 and zoom 18) convert both ways within 0.1 px;
2. a zoom-17 pixel and its zoom-18 pixel (the same physical point) agree;
3. the task map loads with no georeference.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from dronenav.georef import GeorefBlock, GeorefError, load_georef, load_map

ROOT = Path(__file__).resolve().parents[1]
IRPIN_2025 = ROOT / "data" / "irpin" / "maps" / "irpin_2025.json"
TASK_MAP = ROOT / "data" / "hackatons-data" / "global_map.png"


# --- the Web-Mercator math itself: synthetic, no data/ needed ----------------------------------

def _block(**over) -> GeorefBlock:
    base = {"file": "irpin_2025.png", "zoom": 17, "m_per_px": 0.7593316812880911, "size_px": (3936, 3936),
            "merc_origin_px": (19592192.0, 11301728.0),
            "bounds_lonlat": (30.201416015625, 50.508623285115206, 30.24364471435547, 50.53547149911047),
            "capture_coverage": 1.0, "alignment_to_2025": {"shift_px": [0.0, 0.0], "reference": True}}
    base.update(over)
    return GeorefBlock(**base)


@pytest.mark.parametrize("lon,lat", [(30.21, 50.52), (30.2014, 50.5355), (30.2436, 50.5086), (30.22, 50.53)])
def test_px_lonlat_round_trip_within_0_1_px(lon, lat):
    block = _block()
    x, y = block.lonlat_to_px(lon, lat)
    lon2, lat2 = block.px_to_lonlat(x, y)
    x2, y2 = block.lonlat_to_px(lon2, lat2)
    assert math.hypot(x2 - x, y2 - y) <= 0.1


def test_bounds_corners_match_the_pixel_grid():
    # (0, 0) is north-west, (w, h) is south-east: the orientation bounds_lonlat implies
    block = _block()
    west, south, east, north = block.bounds_lonlat
    x0, y0 = block.lonlat_to_px(west, north)
    assert math.hypot(x0, y0) <= 0.1
    w, h = block.size_px
    x1, y1 = block.lonlat_to_px(east, south)
    assert math.hypot(x1 - w, y1 - h) <= 0.1


def test_px_m_round_trip():
    block = _block()
    mx, my = block.px_to_m(100.0, -40.0)
    assert block.m_to_px(mx, my) == pytest.approx((100.0, -40.0))


def test_doubling_zoom_doubles_world_pixel_position():
    # the hires block of the same area is the map block's grid at one zoom finer: a given lon/lat lands
    # at exactly double the pixel offset once origins are accounted for (the tile-px convention itself)
    base = _block()
    hires = _block(zoom=18, m_per_px=base.m_per_px / 2, size_px=(7872, 7872),
                    merc_origin_px=(39184384.0, 22603456.0))
    lon, lat = 30.22, 50.53
    x, y = base.lonlat_to_px(lon, lat)
    hx, hy = hires.lonlat_to_px(lon, lat)
    assert (hx, hy) == pytest.approx((2 * x, 2 * y), abs=1e-6)


# --- sidecar parsing: synthetic malformed inputs, no data/ needed -------------------------------

def _valid_sidecar() -> dict:
    block = {"file": "irpin_2025.png", "zoom": 17, "m_per_px": 0.75, "size_px": [3936, 3936],
             "merc_origin_px": [19592192, 11301728],
             "bounds_lonlat": [30.2014, 50.5086, 30.2436, 50.5355],
             "capture_coverage": 1.0, "alignment_to_2025": {"shift_px": [0.0, 0.0], "reference": True}}
    return {"capture": {"date": "2025-04-11", "sensor": "WV02"}, "source": "test",
            "map": block, "hires": dict(block, zoom=18)}


def test_load_georef_reads_a_valid_sidecar(tmp_path):
    path = tmp_path / "sidecar.json"
    path.write_text(json.dumps(_valid_sidecar()))
    g = load_georef(path)
    assert g.map.zoom == 17 and g.hires.zoom == 18 and g.source == "test"


@pytest.mark.parametrize("mutate,words", [
    (lambda d: d.pop("map"), ["missing", "map"]),
    (lambda d: d["map"].pop("zoom"), ["missing", "zoom"]),
    (lambda d: d.update(extra_field=1), ["unknown", "extra_field"]),
])
def test_load_georef_rejects_missing_or_unknown_fields(tmp_path, mutate, words):
    d = _valid_sidecar()
    mutate(d)
    path = tmp_path / "sidecar.json"
    path.write_text(json.dumps(d))
    with pytest.raises(GeorefError) as e:
        load_georef(path)
    msg = str(e.value)
    for w in words:
        assert w in msg, f"{w!r} not in {msg!r}"


def test_load_georef_rejects_invalid_json(tmp_path):
    path = tmp_path / "sidecar.json"
    path.write_text("{not json")
    with pytest.raises(GeorefError):
        load_georef(path)


def test_load_georef_rejects_non_finite_numbers(tmp_path):
    path = tmp_path / "sidecar.json"
    path.write_text(json.dumps(_valid_sidecar()).replace("0.75", "NaN"))
    with pytest.raises(GeorefError):
        load_georef(path)


# --- loading a map image: synthetic "no sidecar" case, no data/ needed --------------------------

def test_load_map_with_no_sidecar_has_no_georeference(tmp_path):
    img_path = tmp_path / "plain_map.png"
    cv2.imwrite(str(img_path), np.zeros((10, 10, 3), np.uint8))
    image, georef = load_map(img_path)
    assert image is not None and georef is None


# --- on the real Irpin sidecar, when data/ is present --------------------------------------------

@pytest.mark.data
@pytest.mark.skipif(not IRPIN_2025.exists(), reason="data/ not downloaded")
def test_irpin_2025_corners_round_trip_both_zooms():
    g = load_georef(IRPIN_2025)
    for block in (g.map, g.hires):
        west, south, east, north = block.bounds_lonlat
        w, h = block.size_px
        x0, y0 = block.lonlat_to_px(west, north)
        assert math.hypot(x0, y0) <= 0.1
        x1, y1 = block.lonlat_to_px(east, south)
        assert math.hypot(x1 - w, y1 - h) <= 0.1
        lon, lat = block.px_to_lonlat(w / 2, h / 2)
        x2, y2 = block.lonlat_to_px(lon, lat)
        assert math.hypot(x2 - w / 2, y2 - h / 2) <= 0.1


@pytest.mark.data
@pytest.mark.skipif(not IRPIN_2025.exists(), reason="data/ not downloaded")
def test_irpin_2025_map_and_hires_pixel_agree():
    g = load_georef(IRPIN_2025)
    x, y = g.map.size_px[0] / 2, g.map.size_px[1] / 2  # the map's centre pixel
    hx, hy = g.reproject(x, y, src="map", dst="hires")
    x2, y2 = g.reproject(hx, hy, src="hires", dst="map")
    assert math.hypot(x2 - x, y2 - y) <= 0.1


@pytest.mark.data
@pytest.mark.skipif(not TASK_MAP.exists(), reason="data/ not downloaded")
def test_task_map_loads_with_no_georeference():
    image, georef = load_map(TASK_MAP)
    assert image is not None and image.shape[:2] == (1845, 3645) and georef is None
