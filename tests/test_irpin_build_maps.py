"""M3-03 · Build the three maps: one single-capture map per epoch, cut and aligned to 2025.

``experiments/irpin/build_maps.py`` consolidates the exploratory ``experiments/ukraine_scan/west_irpin*``
scripts into the one committed entry point the issue names. "Done when" needs: (a) each sidecar shows
100% single-capture coverage and an alignment residual under 1 px, and (b) the sidecar schema matches
what is already shipped in ``data/irpin/maps/*.json`` (``capture``, ``source``, ``map``/``hires`` blocks
with ``capture_coverage`` and ``alignment_to_2025``). None of this touches the network: the mosaic fetch
(ArcGIS metadata + tile download) is exercised only through a monkeypatched fake in the end-to-end test;
everything else -- geo math, the greedy single-date cover, the tiled-gradient-phase-correlation alignment,
cropping, the sidecar schema and its validation -- is pure and tested on synthetic data with known answers.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BM_PATH = ROOT / "experiments" / "irpin" / "build_maps.py"
_spec = importlib.util.spec_from_file_location("irpin_build_maps", BM_PATH)
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

SCAN_PATH = ROOT / "experiments" / "ukraine_scan" / "scan.py"
_scan_spec = importlib.util.spec_from_file_location("irpin_build_maps_scan_ref", SCAN_PATH)
scan = importlib.util.module_from_spec(_scan_spec)
_scan_spec.loader.exec_module(scan)


# ---------------------------------------------------------------------------
# Geo math: merc / unmerc / m_per_px, generalised to any zoom (the shipped maps
# live at zoom 17/18, not scan.py's fixed zoom 16)
# ---------------------------------------------------------------------------

def test_merc_matches_scan_py_at_its_own_zoom_16():
    # scan.py's merc()/unmerc() are already relied on to build the shipped data (DATA_CARD.md); cross
    # -checking our generalised version against them at z16 is reuse-verification, not reinvention.
    for lon, lat in [(30.2216, 50.501), (30.0, 50.0), (30.5, 50.9)]:
        assert bm.merc(lon, lat, 16) == pytest.approx(scan.merc(lon, lat), abs=1e-6)
    x, y = bm.merc(30.2216, 50.501, 16)
    assert bm.unmerc(x, y, 16) == pytest.approx(scan.unmerc(x, y), abs=1e-9)


def test_merc_unmerc_round_trip_at_zoom_17_and_18():
    for zoom in (17, 18):
        lon, lat = 30.21, 50.52
        x, y = bm.merc(lon, lat, zoom)
        lon2, lat2 = bm.unmerc(x, y, zoom)
        assert (lon2, lat2) == pytest.approx((lon, lat), abs=1e-9)


def test_m_per_px_halves_from_zoom_17_to_18():
    # one more zoom level halves the ground size of a pixel, like the shipped 0.759 -> 0.380 m/px.
    lat = 50.52
    assert bm.m_per_px(lat, 18) == pytest.approx(bm.m_per_px(lat, 17) / 2, rel=1e-9)


# ---------------------------------------------------------------------------
# build_grid: tile-aligned fetch grid around the seed, with room for the alignment shift
# ---------------------------------------------------------------------------

def test_build_grid_is_tile_aligned_and_centered():
    grid = bm.build_grid((30.2216, 50.501), size_px=3936, zoom=17, margin_tiles=1)
    assert grid["fetch_origin_px"][0] % 256 == 0 and grid["fetch_origin_px"][1] % 256 == 0
    assert grid["target_origin_px"][0] % 256 == 0 and grid["target_origin_px"][1] % 256 == 0
    # the target square must fit entirely inside the fetched mosaic
    fx0, fy0 = grid["fetch_origin_px"]
    tx0, ty0 = grid["target_origin_px"]
    fetch_size = grid["fetch_tiles"] * 256
    assert fx0 <= tx0 and tx0 + 3936 <= fx0 + fetch_size
    assert fy0 <= ty0 and ty0 + 3936 <= fy0 + fetch_size
    # centered on the seed within half a tile
    cx, cy = bm.merc(30.2216, 50.501, 17)
    assert abs((tx0 + 3936 / 2) - cx) <= 128
    assert abs((ty0 + 3936 / 2) - cy) <= 128


# ---------------------------------------------------------------------------
# rasterize_footprint: ArcGIS metadata features -> boolean mask, date/sensor filtered
# ---------------------------------------------------------------------------

def _ring_for_px_box(grid, x0, y0, x1, y1):
    corners_px = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    fx0, fy0 = grid["fetch_origin_px"]
    return [list(bm.unmerc(fx0 + px, fy0 + py, grid["zoom"])) for px, py in corners_px]


def _wayback_date_ms(date: str) -> int:
    import datetime as dt
    return int(dt.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


def test_rasterize_footprint_keeps_only_the_matching_date_and_sensor():
    grid = {"zoom": 14, "fetch_origin_px": (0, 0), "fetch_tiles": 1}  # 256x256
    ours = {"type": "Feature", "attributes": {"SRC_DATE2": _wayback_date_ms("2022-03-31"), "SRC_DESC": "WV03"},
            "geometry": {"rings": [_ring_for_px_box(grid, 0, 0, 128, 256)]}}
    other = {"type": "Feature", "attributes": {"SRC_DATE2": _wayback_date_ms("2020-01-01"), "SRC_DESC": "WV02"},
              "geometry": {"rings": [_ring_for_px_box(grid, 128, 0, 256, 256)]}}
    mask = bm.rasterize_footprint([ours, other], grid, "2022-03-31", "WV03")
    assert mask.shape == (256, 256)
    assert mask[128, 32]       # inside the "ours" polygon
    assert not mask[128, 224]  # inside the "other" (wrong date) polygon


def test_rasterize_footprint_with_no_matching_feature_is_all_false():
    grid = {"zoom": 14, "fetch_origin_px": (0, 0), "fetch_tiles": 1}
    other = {"type": "Feature", "attributes": {"SRC_DATE2": _wayback_date_ms("2020-01-01"), "SRC_DESC": "WV02"},
              "geometry": {"rings": [_ring_for_px_box(grid, 0, 0, 256, 256)]}}
    mask = bm.rasterize_footprint([other], grid, "2022-03-31", "WV03")
    assert not mask.any()


# ---------------------------------------------------------------------------
# select_cover: greedy set cover over release footprints (single_date_map.py's algorithm, factored out)
# ---------------------------------------------------------------------------

def test_select_cover_picks_biggest_marginal_addition_first_and_covers_the_union():
    a = np.zeros((10, 10), bool); a[:, :6] = True           # 60 px, biggest
    b = np.zeros((10, 10), bool); b[:, 4:10] = True          # overlaps a, adds 40 new
    order = bm.select_cover({"a": a, "b": b}, min_new_px=1)
    assert [k for k, _ in order] == ["a", "b"]
    union = np.zeros((10, 10), bool)
    for _, new in order:
        assert not (union & new).any()  # each piece disjoint from what came before
        union |= new
    assert (union == (a | b)).all()


def test_select_cover_drops_a_release_below_the_new_pixel_threshold():
    a = np.zeros((10, 10), bool); a[:, :10] = True   # covers everything
    b = np.zeros((10, 10), bool); b[0, 0] = True      # 1 px, all already covered by `a`
    order = bm.select_cover({"a": a, "b": b}, min_new_px=5)
    assert [k for k, _ in order] == ["a"]


def test_select_cover_of_no_masks_is_empty():
    assert bm.select_cover({}, min_new_px=1) == []


# ---------------------------------------------------------------------------
# tiled_gradient_shift: the alignment math at the heart of the "Done when" bar
# (median of tiled phase correlation on gradient images)
# ---------------------------------------------------------------------------

def _textured_image(rng, size):
    """Blurred noise: has real spatial structure for phase correlation (a flat image doesn't)."""
    base = rng.integers(0, 255, (size, size), dtype=np.uint8)
    g = cv2.GaussianBlur(base, (5, 5), 0)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def test_tiled_gradient_shift_recovers_a_known_subpixel_shift():
    rng = np.random.default_rng(0)
    size = 512
    ref = _textured_image(rng, size)
    true_dx, true_dy = 2.4, -1.6
    # img(x, y) = ref(x + true_dx, y + true_dy): matches the shift_px convention already used by
    # west_irpin_preview.py (warpAffine(img, [[1,0,-dx],[0,1,-dy]]) aligns img onto ref).
    img = cv2.warpAffine(ref, np.float32([[1, 0, true_dx], [0, 1, true_dy]]), (size, size),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    valid = np.zeros((size, size), bool)
    valid[32:size - 32, 32:size - 32] = True  # keep away from the warp's border artifacts

    result = bm.tiled_gradient_shift(ref, img, valid, tile=64)

    assert result["shift_px"] == pytest.approx([true_dx, true_dy], abs=0.3)
    assert result["tiles"] > 0
    assert result["tile_spread_px"] < 0.5
    assert math.hypot(*result["residual_after_px"]) < 0.5


def test_tiled_gradient_shift_rejects_an_empty_valid_mask():
    rng = np.random.default_rng(1)
    ref = _textured_image(rng, 128)
    img = ref.copy()
    with pytest.raises(ValueError):
        bm.tiled_gradient_shift(ref, img, np.zeros((128, 128), bool), tile=64)


def test_tiled_gradient_shift_rejects_a_blank_textureless_image():
    blank = np.full((256, 256, 3), 127, np.uint8)
    valid = np.ones((256, 256), bool)
    with pytest.raises(ValueError):
        bm.tiled_gradient_shift(blank, blank, valid, tile=64)


# ---------------------------------------------------------------------------
# crop_with_shift: crop + sub-pixel alignment shift combined in one resample
# ---------------------------------------------------------------------------

def test_crop_with_shift_zero_shift_is_an_exact_crop():
    mosaic = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    mosaic = cv2.cvtColor(mosaic, cv2.COLOR_GRAY2BGR)
    crop = bm.crop_with_shift(mosaic, local_origin_px=(10, 5), size_px=20)
    assert (crop == mosaic[5:25, 10:30]).all()


def test_crop_with_shift_integer_shift_matches_a_translated_crop():
    mosaic = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    mosaic = cv2.cvtColor(mosaic, cv2.COLOR_GRAY2BGR)
    crop = bm.crop_with_shift(mosaic, local_origin_px=(10, 5), size_px=20, shift_px=(3.0, -2.0))
    # shift_px=(dx,dy) moves the *sampled* window by (dx,dy): origin becomes (10+3, 5-2)
    assert (crop == mosaic[3:23, 13:33]).all()


# ---------------------------------------------------------------------------
# Sidecar schema: matches data/irpin/maps/*.json exactly (capture/source/map/hires,
# capture_coverage, alignment_to_2025 with or without "reference")
# ---------------------------------------------------------------------------

def test_coverage_fraction_of_a_fully_valid_mask_is_one():
    assert bm.coverage_fraction(np.ones((10, 10), bool)) == 1.0


def test_coverage_fraction_of_a_partially_valid_mask():
    mask = np.zeros((10, 10), bool); mask[:5] = True
    assert bm.coverage_fraction(mask) == pytest.approx(0.5)


def test_coverage_fraction_rejects_an_empty_mask():
    with pytest.raises(ValueError):
        bm.coverage_fraction(np.zeros((0, 0), bool))


def test_alignment_block_for_the_2025_reference_has_no_residual_fields():
    assert bm.alignment_block(reference=True) == {"shift_px": [0.0, 0.0], "reference": True}


def test_alignment_block_for_a_non_reference_epoch_matches_the_shipped_schema():
    block = bm.alignment_block((2.93, 2.70), tiles=42, tile_spread_px=1.25, residual_after_px=(0.16, -0.04))
    assert block == {"shift_px": [2.93, 2.70], "tiles": 42, "tile_spread_px": 1.25,
                      "residual_after_px": [0.16, -0.04]}
    assert "reference" not in block


def test_alignment_block_requires_tiles_and_residual_for_a_non_reference_epoch():
    with pytest.raises(ValueError):
        bm.alignment_block((1.0, 1.0))


def test_map_block_schema_and_geo_fields():
    valid = np.ones((100, 100), bool)
    align = bm.alignment_block(reference=True)
    block = bm.map_block("maps/irpin_2025.png", 17, origin_px=(19592192, 11301728), size_px=100,
                         valid_mask=valid, alignment=align)
    assert block["file"] == "maps/irpin_2025.png"
    assert block["zoom"] == 17
    assert block["size_px"] == [100, 100]
    assert block["merc_origin_px"] == [19592192, 11301728]
    assert block["capture_coverage"] == 1.0
    assert block["alignment_to_2025"] == align
    lon0, lat1, lon1, lat0 = block["bounds_lonlat"]
    assert lon0 < lon1 and lat0 < lat1  # (lon0,lat1) top-left, (lon1,lat0) bottom-right, like the shipped sidecars


def test_build_sidecar_duplicates_map_block_into_hires_when_no_independent_hires_capture():
    # the shipped irpin_2017.json: no zoom-18 pre-war capture exists, so "hires" == "map".
    mblock = bm.map_block("maps/irpin_2017.png", 17, (0, 0), 10, np.ones((10, 10), bool),
                          bm.alignment_block((1.0, 1.0), tiles=5, tile_spread_px=0.1, residual_after_px=(0.1, 0.1)))
    sidecar = bm.build_sidecar({"date": "2017-05-05", "sensor": "WV02"}, mblock, hires_block=None)
    assert sidecar["hires"] == sidecar["map"] == mblock
    assert sidecar["capture"] == {"date": "2017-05-05", "sensor": "WV02"}
    assert sidecar["source"] == bm.SOURCE


def test_build_sidecar_keeps_an_independent_hires_block_when_given_one():
    mblock = bm.map_block("maps/irpin_2025.png", 17, (0, 0), 10, np.ones((10, 10), bool), bm.alignment_block(reference=True))
    hblock = bm.map_block("hires/irpin_2025_z18.png", 18, (0, 0), 20, np.ones((20, 20), bool), bm.alignment_block(reference=True))
    sidecar = bm.build_sidecar({"date": "2025-04-11", "sensor": "WV02"}, mblock, hblock)
    assert sidecar["map"] == mblock
    assert sidecar["hires"] == hblock
    assert sidecar["map"] != sidecar["hires"]


# ---------------------------------------------------------------------------
# validate_build: the rejection path for the "Done when" bar itself
# ---------------------------------------------------------------------------

def _sidecar_with(coverage=1.0, residual=(0.1, 0.1), reference=False):
    valid = np.ones((10, 10), bool) if coverage == 1.0 else np.zeros((10, 10), bool)
    if coverage not in (0.0, 1.0):
        valid = np.zeros((10, 10), bool); valid[: int(coverage * 100)] = True
    align = bm.alignment_block(reference=True) if reference else \
        bm.alignment_block((1.0, 1.0), tiles=10, tile_spread_px=0.2, residual_after_px=residual)
    mblock = bm.map_block("maps/irpin_x.png", 17, (0, 0), 10, valid, align)
    return bm.build_sidecar({"date": "2022-03-31", "sensor": "WV03"}, mblock)


def test_validate_build_accepts_full_coverage_and_small_residual():
    bm.validate_build(_sidecar_with(coverage=1.0, residual=(0.1, 0.1)), max_residual_px=1.0)


def test_validate_build_rejects_incomplete_coverage():
    with pytest.raises(ValueError):
        bm.validate_build(_sidecar_with(coverage=0.9, residual=(0.1, 0.1)), max_residual_px=1.0)


def test_validate_build_rejects_a_residual_at_or_above_the_limit():
    with pytest.raises(ValueError):
        bm.validate_build(_sidecar_with(coverage=1.0, residual=(0.8, 0.8)), max_residual_px=1.0)


def test_validate_build_skips_the_residual_check_for_the_reference_epoch():
    bm.validate_build(_sidecar_with(coverage=1.0, reference=True), max_residual_px=1.0)


# ---------------------------------------------------------------------------
# parse_args: CLI defaults and the one-command usage the issue asks for
# ---------------------------------------------------------------------------

def test_parse_args_defaults():
    args = bm.parse_args([])
    assert args.out == "data/irpin"
    assert args.max_residual_px == 1.0


def test_parse_args_overrides():
    args = bm.parse_args(["--out", "/tmp/x", "--cache", "/tmp/c", "--max-residual-px", "0.5"])
    assert args.out == "/tmp/x"
    assert args.cache == "/tmp/c"
    assert args.max_residual_px == 0.5


# ---------------------------------------------------------------------------
# End-to-end: main() wiring, with the network mosaic fetch faked (deterministic, no network)
# ---------------------------------------------------------------------------

def _fake_mosaic_factory(rng, shifts_by_date):
    """A stand-in for build_single_date_mosaic(): every date is the same textured base image,
    globally shifted by a known sub-pixel amount, fully valid -- so the fake run must come out at
    100% coverage and under the residual bar, exercising main()'s own validate_build() call."""
    base = {}

    def fake(session, cfg, grid, date, sensor):
        size = grid["fetch_tiles"] * 256
        if "img" not in base:
            base["img"] = _textured_image(rng, size)
        dx, dy = shifts_by_date.get(date, (0.0, 0.0))
        img = cv2.warpAffine(base["img"], np.float32([[1, 0, dx], [0, 1, dy]]), (size, size),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return img, np.ones((size, size), bool)

    return fake


def test_main_builds_all_five_outputs_with_a_faked_mosaic_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "SIZE_PX", 256)
    monkeypatch.setattr(bm, "HIRES_SIZE_PX", 256)
    rng = np.random.default_rng(42)
    shifts = {"2017-05-05": (1.1, -0.6), "2022-03-31": (-0.8, 0.5), "2025-04-11": (0.0, 0.0)}
    monkeypatch.setattr(bm, "build_single_date_mosaic", _fake_mosaic_factory(rng, shifts))
    monkeypatch.setattr(bm, "fetch_release_config", lambda session: {})

    out = tmp_path / "irpin"
    bm.main(["--out", str(out), "--cache", str(tmp_path / "cache")])

    for year in ("2017", "2022", "2025"):
        png, sidecar_p = out / "maps" / f"irpin_{year}.png", out / "maps" / f"irpin_{year}.json"
        assert png.exists() and sidecar_p.exists()
        sidecar = json.loads(sidecar_p.read_text())
        assert sidecar["map"]["capture_coverage"] == 1.0
        assert sidecar["hires"]["capture_coverage"] == 1.0
        align = sidecar["map"]["alignment_to_2025"]
        if year == "2025":
            assert align == {"shift_px": [0.0, 0.0], "reference": True}
        else:
            assert math.hypot(*align["residual_after_px"]) < 1.0

    for year in ("2022", "2025"):
        assert (out / "hires" / f"irpin_{year}_z18.png").exists()
    assert not (out / "hires" / "irpin_2017_z18.png").exists()
