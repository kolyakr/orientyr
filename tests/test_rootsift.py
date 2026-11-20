"""M1-02 · RootSIFT matcher with cached map features.

"Done when: the second run on the same map skips map feature extraction; matches are returned in full-map
coordinates." So:
- a second matcher (a new run) on the same map loads the map features from the cache instead of extracting;
- the cache key covers the map's pixels and the extraction parameters: another map or other parameters
  extract again, a copy of the same map does not; a broken cache file is rebuilt, not fatal;
- window queries reuse the cached map features (no extraction) and return full-map coordinates
  (full-map coordinates for every backend are in test_matcher_contract.py);
- the RootSIFT transform itself, and the Lowe ratio.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from tests.synth import apply, make_h, make_map, render_frame

from dronenav.geometry import localize
from dronenav.matching import RootSiftConfig, RootSiftMatcher, rootsift

ROOT = Path(__file__).resolve().parents[1]
FRAME_SIZE = (480, 360)
MAP = make_map()
H_TRUE = make_h(-20.0, 0.8, cx=500.0, cy=450.0, frame_size=FRAME_SIZE)
FRAME = render_frame(MAP, H_TRUE, FRAME_SIZE)


def _cache_files(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*") if p.is_file()) if d.exists() else []


# --- the cache: the second run skips map feature extraction --------------------------------------------------

def test_second_run_loads_map_features_from_cache(tmp_path):
    cache = tmp_path / "cache"
    first = RootSiftMatcher(cache_dir=cache)
    m1 = first.match(FRAME, MAP)
    assert first.map_extractions == 1 and first.map_features_source == "extracted"
    assert len(_cache_files(cache)) == 1

    second = RootSiftMatcher(cache_dir=cache)      # a new run: nothing in memory
    m2 = second.match(FRAME, MAP)
    assert second.map_extractions == 0 and second.map_features_source == "cache"
    # the cached features are the extracted ones: the same matches come back
    np.testing.assert_array_equal(m1.pts_map, m2.pts_map)
    np.testing.assert_array_equal(m1.pts_frame, m2.pts_frame)


def test_map_given_at_construction_is_cached(tmp_path):
    RootSiftMatcher(MAP, cache_dir=tmp_path)
    again = RootSiftMatcher(MAP, cache_dir=tmp_path)
    assert again.map_extractions == 0 and again.map_features_source == "cache"


def test_repeated_calls_extract_the_map_once(tmp_path):
    m = RootSiftMatcher(cache_dir=tmp_path)
    for _ in range(3):
        m.match(FRAME, MAP)
    assert m.map_extractions == 1


def test_a_copy_of_the_same_map_hits_the_cache(tmp_path):
    RootSiftMatcher(MAP, cache_dir=tmp_path)
    again = RootSiftMatcher(MAP.copy(), cache_dir=tmp_path)   # same pixels, another array: keyed by content
    assert again.map_extractions == 0


def test_another_map_is_extracted(tmp_path):
    RootSiftMatcher(MAP, cache_dir=tmp_path)
    other = make_map(seed=1)
    m = RootSiftMatcher(other, cache_dir=tmp_path)
    assert m.map_extractions == 1
    assert len(_cache_files(tmp_path)) == 2


def test_a_one_pixel_change_is_another_map(tmp_path):
    RootSiftMatcher(MAP, cache_dir=tmp_path)
    edited = MAP.copy()
    # a change SIFT can see (it works on intensity): one pixel turned from its value to its inverse in every
    # channel. A colour change that leaves the grey value equal gives identical features and may share the cache.
    edited[10, 10] = 255 - edited[10, 10]
    assert RootSiftMatcher(edited, cache_dir=tmp_path).map_extractions == 1


def test_other_extraction_parameters_are_extracted(tmp_path):
    RootSiftMatcher(MAP, cache_dir=tmp_path)
    m = RootSiftMatcher(MAP, cache_dir=tmp_path, cfg=RootSiftConfig(contrast_threshold=0.03))
    assert m.map_extractions == 1
    assert len(_cache_files(tmp_path)) == 2


def test_matching_only_parameters_share_the_cache(tmp_path):
    # the ratio is applied at match time, it doesn't change the map's features: no new extraction
    RootSiftMatcher(MAP, cache_dir=tmp_path)
    m = RootSiftMatcher(MAP, cache_dir=tmp_path, cfg=RootSiftConfig(ratio=0.7))
    assert m.map_extractions == 0


def test_broken_cache_file_is_rebuilt(tmp_path):
    RootSiftMatcher(MAP, cache_dir=tmp_path)
    (f,) = _cache_files(tmp_path)
    f.write_bytes(b"not an npz")
    m = RootSiftMatcher(MAP, cache_dir=tmp_path)
    assert m.map_extractions == 1
    assert _localizes(m.match(FRAME, MAP))
    assert RootSiftMatcher(MAP, cache_dir=tmp_path).map_extractions == 0   # and the rebuilt file is good


def test_no_cache_dir_means_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = RootSiftMatcher(MAP, cache_dir=None)
    assert m.map_extractions == 1 and not any(tmp_path.rglob("*"))


# --- windows reuse the cached map features --------------------------------------------------------------------

def _localizes(m) -> bool:
    fix = localize(0, m, FRAME_SIZE)
    return fix is not None and fix.accepted and np.hypot(fix.x - 500.0, fix.y - 450.0) < 1.0


def test_window_queries_use_cached_features(tmp_path):
    RootSiftMatcher(MAP, cache_dir=tmp_path)
    m = RootSiftMatcher(MAP, cache_dir=tmp_path)        # second run: features from the cache
    x0, y0 = 250, 230
    win = MAP[y0:y0 + 450, x0:x0 + 520]
    res = m.match(FRAME, win, map_offset=(x0, y0))
    assert m.map_extractions == 0                        # the window was not re-extracted
    assert _localizes(res)                               # and the answer is in full-map coordinates
    assert (res.pts_map[:, 0] >= x0).all() and (res.pts_map[:, 1] >= y0).all()


def test_window_of_an_unknown_map_is_extracted_on_the_fly(tmp_path):
    # a window cut from a map the matcher has never seen still works: its features are computed for that call
    m = RootSiftMatcher(cache_dir=tmp_path)
    x0, y0 = 250, 230
    res = m.match(FRAME, MAP[y0:y0 + 450, x0:x0 + 520], map_offset=(x0, y0))
    assert _localizes(res)


def test_window_offset_must_be_whole_pixels(tmp_path):
    m = RootSiftMatcher(MAP, cache_dir=tmp_path)
    with pytest.raises(ValueError):
        m.match(FRAME, MAP[100:500, 100:600], map_offset=(100.5, 100))


# --- RootSIFT and the ratio test ------------------------------------------------------------------------------

def test_rootsift_is_unit_length_and_keeps_zero_rows():
    rng = np.random.default_rng(0)
    d = rng.integers(0, 256, size=(20, 128)).astype(np.float32)
    d[3] = 0
    r = rootsift(d)
    assert r.shape == d.shape and r.dtype == np.float32
    norms = np.linalg.norm(r, axis=1)
    np.testing.assert_allclose(np.delete(norms, 3), 1.0, atol=1e-5)
    assert (r[3] == 0).all()
    # Hellinger kernel: the dot product of two RootSIFT vectors is sum sqrt(p_i q_i) of the L1-normalised inputs
    p, q = d[0] / d[0].sum(), d[1] / d[1].sum()
    assert float(r[0] @ r[1]) == pytest.approx(float(np.sum(np.sqrt(p * q))), rel=1e-5)


def test_scores_are_ratio_margins(tmp_path):
    m = RootSiftMatcher(cache_dir=tmp_path).match(FRAME, MAP)
    # score = 1 - d1/d2: in (0, 1], higher is a more distinctive match; ratio 0.8 keeps scores above 0.2
    assert (m.scores > 0.2 - 1e-6).all() and (m.scores <= 1.0).all()


def test_stricter_ratio_keeps_fewer_matches(tmp_path):
    loose = RootSiftMatcher(cache_dir=tmp_path, cfg=RootSiftConfig(ratio=0.9)).match(FRAME, MAP)
    strict = RootSiftMatcher(cache_dir=tmp_path, cfg=RootSiftConfig(ratio=0.6)).match(FRAME, MAP)
    assert 0 < len(strict) < len(loose)
    good = np.linalg.norm(apply(H_TRUE, strict.pts_frame) - strict.pts_map, axis=1) < 3
    assert good.mean() > 0.9


@pytest.mark.parametrize("bad", [{"ratio": 0.0}, {"ratio": 1.5}, {"flann_trees": 0}, {"flann_checks": 0}])
def test_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        RootSiftConfig(**bad)


# --- on the task data, when data/ is present ------------------------------------------------------------------

TASK = ROOT / "data" / "hackatons-data"
PROBE = ROOT / "experiments" / "data_probe" / "results" / "probe.json"


@pytest.mark.data
@pytest.mark.skipif(not (TASK / "global_map.png").exists() or not PROBE.exists(), reason="data/ not downloaded")
def test_task_crops_localize_and_second_run_is_cached(tmp_path):
    truth = {r["i"]: r for r in json.loads(PROBE.read_text())["rows"]}
    mp = cv2.imread(str(TASK / "global_map.png"))
    first = RootSiftMatcher(mp, cache_dir=tmp_path)
    second = RootSiftMatcher(mp, cache_dir=tmp_path)
    assert first.map_extractions == 1 and second.map_extractions == 0
    # crops the prototype matched best at full resolution (probe.json "f": 1.0); the others need the frame
    # downscaled first, which is M1-03's job, not the matcher's
    for i in (0, 12, 28, 48):
        crop = cv2.imread(str(TASK / "crops" / f"crop_{i}.png"))
        fix = localize(i, second.match(crop, mp), (crop.shape[1], crop.shape[0]))
        # probe.json is the prototype's own RootSIFT + MAGSAC fit, rounded to 0.1 px; 2 px is the M1 accuracy goal
        assert fix is not None and fix.accepted
        assert np.hypot(fix.x - truth[i]["cx"], fix.y - truth[i]["cy"]) < 2.0
