"""M5-07 · GPS spoof detector.

"What: input: a GPS track (from the simulator: honest, then walked away slowly); compare with the filter's position
and covariance; alarm after the normalised gap stays above a threshold for N frames. Output into the telemetry."
"Done when: detection delay and false-alarm rate are measured on simulated flights."

Three parts, each with its contract here:

- the GPS simulator (``dronenav.sim.gps``): honest = truth + seeded Gaussian noise; spoofed = the same track, then
  from a given frame a walk-off at a set speed and direction; lon/lat through the map's georef (M0-05);
- the detector (``dronenav.tracking.spoof``): Mahalanobis d² of GPS against the filter position, with the filter's
  covariance plus the GPS noise; SUSPECT while above the threshold, SPOOFED (the alarm) after N frames in a row;
- the measurement: per-flight score (false alarm, detection delay) and ``scripts/spoof_check.py``.

All data is synthetic with fixed seeds: a straight flight at one frame per second, positions in map px.
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from dronenav.sim.gps import GpsSample, TruthPoint, WalkOff, read_truth_track, simulate_gps
from dronenav.tracking.spoof import (
    SpoofConfig,
    SpoofDetector,
    SpoofState,
    annotate_run,
    score_trial,
)

from dronenav.georef import GeorefBlock
from dronenav.telemetry import Record, read_run, write_run
from dronenav.tracking.kalman import KalmanFilter
from dronenav.types import TrackMode, TrackState

ROOT = Path(__file__).resolve().parents[1]
FLIGHT = ROOT / "data" / "irpin" / "flights" / "mild" / "2022"
MAP = ROOT / "data" / "irpin" / "maps" / "irpin_2022.png"

START, VEL = np.array([500.0, 800.0]), np.array([12.0, -5.0])     # px, px/s: about 10 m/s on the Irpin maps
GPS_SIGMA = 4.0          # px per axis: about 3 m on a zoom-17 map (0.76 m/px), an ordinary receiver
TRACK_SIGMA = 1.0        # px per axis: the filter's own error in these tests
CFG = SpoofConfig(gps_sigma_px=GPS_SIGMA, threshold_chi2=13.8, n_frames=5)
SIGMA_TOTAL = math.hypot(GPS_SIGMA, TRACK_SIGMA)
# the Irpin maps' zoom-17 block (data/irpin/maps/irpin_2022.json), so lon/lat tests use the real numbers
BLOCK = GeorefBlock(file="maps/irpin_2022.png", zoom=17, m_per_px=0.7593316812880911, size_px=(3936, 3936),
                    merc_origin_px=(19592192, 11301728),
                    bounds_lonlat=(30.201416015625, 50.508623285115206, 30.24364471435547, 50.53547149911047),
                    capture_coverage=1.0, alignment_to_2025={})


def _truth(n: int = 300, dt: float = 1.0, first_idx: int = 0, idx_step: int = 1) -> list[TruthPoint]:
    return [TruthPoint(first_idx + k * idx_step, k * dt, *(START + VEL * k * dt)) for k in range(n)]


def _records(truth: list[TruthPoint], seed: int = 0, sigma: float = TRACK_SIGMA) -> list[Record]:
    """A run whose filtered position is the truth plus noise of a known size, and says so in its covariance."""
    rng = np.random.default_rng(1000 + seed)
    out = []
    for p in truth:
        x, y = np.array([p.x, p.y]) + rng.normal(0, sigma, 2)
        track = TrackState(idx=p.idx, state=TrackMode.TRACKING, x=x, y=y, vx=VEL[0], vy=VEL[1],
                           cov=sigma ** 2 * np.eye(2))
        out.append(Record(idx=p.idx, t=p.t, source=f"frames/frame_{p.idx:06d}.jpg", track=track,
                          extra={"search": "window"}))
    return out


def _alarms(records: list[Record]) -> list[int]:
    return [r.idx for r in records if r.extra["gps"]["alarm"]]


# --- the simulator: honest track = truth + seeded Gaussian noise ---------------------------------------------------

def test_the_honest_track_is_the_truth_plus_gaussian_noise_of_the_given_size():
    truth = _truth(2000)
    gps = simulate_gps(truth, sigma_px=GPS_SIGMA, seed=3)
    assert all(isinstance(s, GpsSample) for s in gps)
    assert [(s.idx, s.t) for s in gps] == [(p.idx, p.t) for p in truth]
    err = np.array([[s.x - p.x, s.y - p.y] for s, p in zip(gps, truth, strict=True)])
    # 4000 draws: the mean's standard error is 4/sqrt(2000) = 0.09 px per axis, the std's is 4/sqrt(8000) = 0.045.
    # 0.3 px is more than 3 standard errors for both, and far below a wrong sigma or a bias of one pixel
    assert np.abs(err.mean(axis=0)).max() < 0.3
    assert abs(err.std() - GPS_SIGMA) < 0.3
    assert abs(np.corrcoef(err.T)[0, 1]) < 0.1          # x and y noise are independent (std error 0.022)
    assert not any(s.spoofed for s in gps) and all(s.offset_px == 0.0 for s in gps)


def test_the_simulator_is_seeded():
    truth = _truth(50)
    a, b = simulate_gps(truth, sigma_px=GPS_SIGMA, seed=7), simulate_gps(truth, sigma_px=GPS_SIGMA, seed=7)
    c = simulate_gps(truth, sigma_px=GPS_SIGMA, seed=8)
    assert [(s.x, s.y) for s in a] == [(s.x, s.y) for s in b]
    assert [(s.x, s.y) for s in a] != [(s.x, s.y) for s in c]


def test_zero_noise_gives_the_truth_exactly():
    truth = _truth(20)
    gps = simulate_gps(truth, sigma_px=0.0, seed=0)
    assert [(s.x, s.y) for s in gps] == [(p.x, p.y) for p in truth]


# --- the simulator: the walk-off is added on top of the honest track ----------------------------------------------

@pytest.mark.parametrize("direction, unit", [
    (0.0, (0.0, -1.0)),       # the heading convention of dronenav.types: 0 = map up (north) = -y
    (90.0, (1.0, 0.0)),       # 90 = east = +x
    (180.0, (0.0, 1.0)),
    (225.0, (-math.sqrt(0.5), math.sqrt(0.5))),
])
def test_the_walk_off_moves_the_honest_track_at_the_given_speed_and_direction(direction, unit):
    truth = _truth(100, dt=0.5)
    honest = simulate_gps(truth, sigma_px=GPS_SIGMA, seed=5)
    spoofed = simulate_gps(truth, sigma_px=GPS_SIGMA, seed=5,
                           walk_off=WalkOff(start_idx=40, speed_px_s=3.0, direction_deg=direction))
    t0 = truth[40].t
    for h, s in zip(honest, spoofed, strict=True):
        want = 3.0 * (s.t - t0) if s.idx >= 40 else 0.0
        # the same seed gives the same noise, so the difference is the walk-off alone: exact up to float rounding
        assert s.x - h.x == pytest.approx(want * unit[0], abs=1e-9)
        assert s.y - h.y == pytest.approx(want * unit[1], abs=1e-9)
        assert s.offset_px == pytest.approx(want, abs=1e-9)
        assert s.spoofed == (s.idx >= 40)


def test_the_walk_off_starts_at_the_first_frame_at_or_after_its_start_idx():
    truth = _truth(20, first_idx=0, idx_step=30)          # keyframes: idx 0, 30, 60, ... one per second
    gps = simulate_gps(truth, sigma_px=0.0, seed=0, walk_off=WalkOff(start_idx=100, speed_px_s=2.0, direction_deg=90))
    by_idx = {s.idx: s for s in gps}
    assert not by_idx[90].spoofed and by_idx[120].spoofed
    assert by_idx[120].offset_px == 0.0                    # the walk starts from the honest position: no jump
    assert by_idx[150].offset_px == pytest.approx(2.0)


# --- the simulator: lon/lat through the georef (M0-05) -------------------------------------------------------------

def test_samples_carry_lon_lat_when_the_map_is_georeferenced():
    truth = [TruthPoint(k, float(k), 1877.5 + 10 * k, 1324.1 + 4 * k) for k in range(30)]
    gps = simulate_gps(truth, sigma_px=GPS_SIGMA, seed=1, block=BLOCK,
                       walk_off=WalkOff(start_idx=10, speed_px_s=1.0, direction_deg=45))
    west, south, east, north = BLOCK.bounds_lonlat
    for s in gps:
        assert west < s.lon < east and south < s.lat < north
        x, y = BLOCK.lonlat_to_px(s.lon, s.lat)
        # the same Web-Mercator formulas both ways: the round trip is exact to float rounding at world px ~2e7
        assert (x, y) == pytest.approx((s.x, s.y), abs=1e-6)


def test_without_a_georef_there_is_no_lon_lat():
    gps = simulate_gps(_truth(5), sigma_px=1.0, seed=0)
    assert all(s.lon is None and s.lat is None for s in gps)


# --- the simulator: bad input --------------------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, word", [
    ({"sigma_px": -1.0}, "sigma_px"),
    ({"sigma_px": math.nan}, "sigma_px"),
    ({"sigma_px": 1.0, "walk_off": WalkOff(start_idx=10_000, speed_px_s=1.0, direction_deg=0)}, "start_idx"),
])
def test_simulate_gps_rejects_bad_settings(kwargs, word):
    with pytest.raises(ValueError, match=word):
        simulate_gps(_truth(10), seed=0, **kwargs)


@pytest.mark.parametrize("truth, word", [
    ([], "empty"),
    ([TruthPoint(0, 0.0, 1.0, math.nan)], "finite"),
    ([TruthPoint(0, 0.0, 1.0, 1.0), TruthPoint(0, 1.0, 2.0, 2.0)], "idx"),      # the same frame twice
    ([TruthPoint(0, 1.0, 1.0, 1.0), TruthPoint(1, 0.5, 2.0, 2.0)], "t"),        # time going back
])
def test_simulate_gps_rejects_a_bad_truth_track(truth, word):
    with pytest.raises(ValueError, match=word):
        simulate_gps(truth, sigma_px=1.0, seed=0)


@pytest.mark.parametrize("kwargs, word", [
    ({"start_idx": -1, "speed_px_s": 1.0, "direction_deg": 0.0}, "start_idx"),
    ({"start_idx": 5, "speed_px_s": 0.0, "direction_deg": 0.0}, "speed_px_s"),
    ({"start_idx": 5, "speed_px_s": -2.0, "direction_deg": 0.0}, "speed_px_s"),
    ({"start_idx": 5, "speed_px_s": 1.0, "direction_deg": math.inf}, "direction_deg"),
])
def test_a_walk_off_checks_its_settings(kwargs, word):
    with pytest.raises(ValueError, match=word):
        WalkOff(**kwargs)


def _write_truth(path: Path, n: int = 240, fps: float = 30.0) -> None:
    rows = [{"i": i, "t": round(i / fps, 4), "center_map_px": list(START + VEL * i / fps)} for i in range(n)]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_read_truth_track_reads_a_flight_and_can_thin_it(tmp_path):
    _write_truth(tmp_path / "truth.jsonl", n=240)
    full = read_truth_track(tmp_path / "truth.jsonl")
    assert len(full) == 240 and full[0] == TruthPoint(0, 0.0, 500.0, 800.0)
    keys = read_truth_track(tmp_path, every=30)            # a flight folder works too; every 30th: one per second
    assert [p.idx for p in keys] == [0, 30, 60, 90, 120, 150, 180, 210]
    assert keys[1].t == pytest.approx(1.0) and (keys[1].x, keys[1].y) == pytest.approx((512.0, 795.0))


def test_read_truth_track_names_what_is_wrong(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"truth\.jsonl"):
        read_truth_track(tmp_path)
    bad = tmp_path / "truth.jsonl"
    bad.write_text('{"i": 0, "t": 0.0}\n')
    with pytest.raises(ValueError, match="center_map_px"):
        read_truth_track(bad)
    with pytest.raises(ValueError, match="every"):
        read_truth_track(bad, every=0)


# --- the detector: the gap is normalised by both uncertainties ------------------------------------------------------

def test_the_normalised_gap_of_an_honest_track_follows_chi_square():
    truth = _truth(300)
    d2, suspect = [], 0
    for seed in range(20):
        det = SpoofDetector(CFG)
        gps = simulate_gps(truth, sigma_px=GPS_SIGMA, seed=seed)
        for rec, s in zip(_records(truth, seed), gps, strict=True):
            r = det.update((s.x, s.y), (rec.track.x, rec.track.y), rec.track.cov)
            d2.append(r.d2)
            suspect += r.state is not SpoofState.OK
    # GPS error and filter error are independent Gaussians, so d² = |gap|² / (σ_gps² + σ_track²) is χ² with 2 degrees
    # of freedom: mean 2, variance 4. Over 6000 frames the mean's standard error is 0.026; 0.1 is almost 4 of them.
    # A detector that forgot either uncertainty would give 2.13 (no track σ) or 34 (no GPS σ)
    assert np.mean(d2) == pytest.approx(2.0, abs=0.1)
    # P(d² > 13.8) = 0.1 % per frame: 6 of 6000 expected; 0.5 % leaves room for the seed, not for a wrong model
    assert suspect / len(d2) < 0.005


def test_the_same_gap_is_normal_when_the_filter_is_unsure_and_suspect_when_it_is_sure():
    gps, pos = (530.0, 800.0), (500.0, 800.0)              # 30 px apart
    sure = SpoofDetector(CFG).update(gps, pos, 2.0 ** 2 * np.eye(2))
    coasting = SpoofDetector(CFG).update(gps, pos, 50.0 ** 2 * np.eye(2))
    assert sure.gap_px == pytest.approx(30.0) and coasting.gap_px == pytest.approx(30.0)
    assert sure.d2 == pytest.approx(900 / (16 + 4)) and sure.state is SpoofState.SUSPECT          # 45 > 13.8
    assert coasting.d2 == pytest.approx(900 / (16 + 2500)) and coasting.state is SpoofState.OK    # 0.36


def test_the_gap_is_judged_along_the_direction_of_the_uncertainty():
    cov = np.diag([40.0 ** 2, 1.0])                        # the filter is unsure along x only (a coasting flight)
    along = SpoofDetector(CFG).update((540.0, 800.0), (500.0, 800.0), cov)
    across = SpoofDetector(CFG).update((500.0, 840.0), (500.0, 800.0), cov)
    assert along.d2 == pytest.approx(1600 / (1600 + 16)) and along.state is SpoofState.OK
    assert across.d2 == pytest.approx(1600 / (1 + 16)) and across.state is SpoofState.SUSPECT


# --- the detector: alarm after N frames in a row -------------------------------------------------------------------

POS, COV = (500.0, 800.0), np.eye(2)
NEAR, FAR = (502.0, 800.0), (600.0, 800.0)                 # d² = 4/17 = 0.24 and 10000/17 = 588


def _feed(det: SpoofDetector, pattern: str) -> list:
    """One update per character: 'n' a GPS reading near the filter, 'F' one far away, '-' no GPS this frame."""
    return [det.update({"n": NEAR, "F": FAR, "-": None}[c], POS, COV) for c in pattern]


def test_the_alarm_comes_on_the_nth_frame_in_a_row_above_the_threshold():
    out = _feed(SpoofDetector(CFG), "nnFFFFFF")
    assert [r.state.value for r in out] == ["OK", "OK", "SUSPECT", "SUSPECT", "SUSPECT", "SUSPECT", "SPOOFED",
                                            "SPOOFED"]
    assert [r.alarm for r in out] == [False] * 6 + [True] * 2
    assert [r.streak for r in out] == [0, 0, 1, 2, 3, 4, 5, 6]


def test_a_glitch_shorter_than_n_frames_raises_no_alarm():
    det = SpoofDetector(CFG)
    out = _feed(det, "nnFFFFnnFFFFnnn")                    # twice 4 frames far away (N = 5), back in between
    assert not any(r.alarm for r in out)
    assert SpoofState.SPOOFED not in {r.state for r in out}
    assert [r.streak for r in out][2:8] == [1, 2, 3, 4, 0, 0]       # the count starts again after a good frame


def test_the_alarm_clears_only_after_n_good_frames_in_a_row():
    out = _feed(SpoofDetector(CFG), "FFFFF" + "nnnnF" + "nnnnn" + "n")
    assert [r.alarm for r in out] == [False] * 4 + [True] + [True] * 5 + [True] * 4 + [False] + [False]
    assert out[-1].state is SpoofState.OK


def test_a_frame_without_gps_or_without_a_filter_state_holds_the_count():
    det = SpoofDetector(CFG)
    out = _feed(det, "FFFF-F")                             # the missing frame is no evidence either way
    assert out[4].state is SpoofState.NO_DATA and out[4].d2 is None and out[4].gap_px is None
    assert not out[4].alarm and out[5].alarm
    r = det.update(FAR, None, None)                        # the filter has not started (or was reset)
    assert r.state is SpoofState.NO_DATA and r.alarm       # an alarm that is on stays on through the gap


def test_n_frames_and_the_threshold_are_settings():
    quick = SpoofDetector(SpoofConfig(gps_sigma_px=GPS_SIGMA, threshold_chi2=13.8, n_frames=1))
    assert _feed(quick, "F")[0].alarm
    lax = SpoofDetector(SpoofConfig(gps_sigma_px=GPS_SIGMA, threshold_chi2=1000.0, n_frames=1))
    assert not _feed(lax, "F")[0].alarm                    # d² = 588 < 1000


def test_reset_forgets_the_count_and_the_alarm():
    det = SpoofDetector(CFG)
    assert _feed(det, "FFFFF")[-1].alarm
    det.reset()
    out = _feed(det, "FFFF")
    assert not any(r.alarm for r in out) and out[0].streak == 1


# --- the detector: bad input ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, word", [
    ({"gps_sigma_px": 0.0}, "gps_sigma_px"),
    ({"gps_sigma_px": -3.0}, "gps_sigma_px"),
    ({"gps_sigma_px": math.nan}, "gps_sigma_px"),
    ({"gps_sigma_px": 4.0, "threshold_chi2": 0.0}, "threshold_chi2"),
    ({"gps_sigma_px": 4.0, "n_frames": 0}, "n_frames"),
    ({"gps_sigma_px": 4.0, "n_frames": 2.5}, "n_frames"),
])
def test_the_config_checks_its_settings(kwargs, word):
    with pytest.raises(ValueError, match=word):
        SpoofConfig(**kwargs)


@pytest.mark.parametrize("gps, pos, cov, word", [
    ((math.nan, 1.0), POS, COV, "gps"),
    ((1.0, 2.0, 3.0), POS, COV, "gps"),
    (NEAR, (math.inf, 0.0), COV, "position"),
    (NEAR, POS, np.eye(3), "cov"),
    (NEAR, POS, [[1.0, math.nan], [math.nan, 1.0]], "cov"),
    (NEAR, POS, -np.eye(2), "cov"),                        # not a covariance: negative variance
    (NEAR, POS, None, "cov"),                              # a position without its covariance
])
def test_update_rejects_bad_input_and_keeps_its_count(gps, pos, cov, word):
    det = SpoofDetector(CFG)
    _feed(det, "FFFF")
    with pytest.raises(ValueError, match=word):
        det.update(gps, pos, cov)
    assert _feed(det, "F")[0].alarm                        # the bad call changed nothing: this is the 5th in a row


# --- output into the telemetry -------------------------------------------------------------------------------------

def test_annotate_run_writes_the_gps_block_into_each_records_extra(tmp_path):
    truth = _truth(60)
    records = _records(truth)
    gps = simulate_gps(truth, sigma_px=GPS_SIGMA, seed=2, block=BLOCK,
                       walk_off=WalkOff(start_idx=20, speed_px_s=4.0, direction_deg=90))
    out = annotate_run(records, gps, CFG)
    assert [r.idx for r in out] == [r.idx for r in records]
    for rec, s in zip(out, gps, strict=True):
        g = rec.extra["gps"]
        assert set(g) == {"x", "y", "lon", "lat", "gap_px", "d2", "state", "alarm", "streak"}
        assert (g["x"], g["y"], g["lon"], g["lat"]) == (s.x, s.y, s.lon, s.lat)
        assert g["gap_px"] == pytest.approx(math.hypot(s.x - rec.track.x, s.y - rec.track.y))
        assert g["state"] in {m.value for m in SpoofState} and isinstance(g["alarm"], bool)
        assert rec.extra["search"] == "window"             # what other modules put there stays
    assert all("gps" not in r.extra for r in records)      # the input records are not changed
    assert out[0].extra["gps"]["state"] == "OK" and out[-1].extra["gps"]["state"] == "SPOOFED"

    # the run.jsonl contract is untouched: the annotated run is written and read back by the ordinary functions
    write_run(tmp_path / "run.jsonl", out)
    back = read_run(tmp_path / "run.jsonl")
    assert [r.extra for r in back] == [r.extra for r in out]


def test_annotate_run_without_a_georef_writes_null_lon_lat_and_marks_frames_without_gps():
    truth = _truth(12)
    records = _records(truth)
    gps = simulate_gps(truth, sigma_px=GPS_SIGMA, seed=2)[:10]        # the receiver went silent for 2 frames
    out = annotate_run(records, gps, CFG)
    assert out[0].extra["gps"]["lon"] is None and out[0].extra["gps"]["lat"] is None
    assert out[-1].extra["gps"] == {"x": None, "y": None, "lon": None, "lat": None, "gap_px": None, "d2": None,
                                    "state": "NO_DATA", "alarm": False, "streak": 0}


def test_annotate_run_rejects_a_gps_track_of_another_flight():
    records = _records(_truth(10))
    other = simulate_gps(_truth(10, first_idx=500), sigma_px=1.0, seed=0)
    with pytest.raises(ValueError, match="no GPS sample"):
        annotate_run(records, other, CFG)


# --- Done when: detection delay and false-alarm rate are measured --------------------------------------------------

def test_an_honest_track_raises_no_alarm():
    truth = _truth(300)
    for seed in range(20):
        out = annotate_run(_records(truth, seed), simulate_gps(truth, sigma_px=GPS_SIGMA, seed=seed), CFG)
        score = score_trial(out)
        # one frame is above the threshold with probability 0.001, five in a row with 1e-15: any alarm is a bug
        assert _alarms(out) == [] and not score.false_alarm and score.false_alarm_frames == 0
        assert score.frames == 300 and score.detected is None and score.delay_s is None


@pytest.mark.parametrize("speed", [1.0, 2.0, 5.0])
def test_a_walk_off_raises_the_alarm_within_a_bounded_delay(speed):
    truth = _truth(300)
    # The gap crosses the threshold when the offset reaches sqrt(13.8) = 3.7 σ of the combined noise. At 3 σ more
    # the noise brings a frame back under the threshold with probability ~0.1 %, so 5 frames in a row follow at
    # once: the alarm cannot be later than that point plus N frames (1 frame = 1 s here).
    bound_s = (math.sqrt(CFG.threshold_chi2) + 3.0) * SIGMA_TOTAL / speed + CFG.n_frames
    delays = []
    for seed in range(10):
        gps = simulate_gps(truth, sigma_px=GPS_SIGMA, seed=seed,
                           walk_off=WalkOff(start_idx=100, speed_px_s=speed, direction_deg=36.0 * seed))
        out = annotate_run(_records(truth, seed), gps, CFG)
        score = score_trial(out, start_idx=100)
        assert not score.false_alarm and score.false_alarm_frames == 0       # nothing before the walk-off starts
        assert score.detected and score.alarm_idx == _alarms(out)[0] >= 100
        assert score.delay_frames == score.alarm_idx - 100 and score.delay_s == pytest.approx(score.delay_frames)
        assert score.delay_s <= bound_s, (seed, score.delay_s, bound_s)
        assert out[-1].extra["gps"]["alarm"]                # once walked away, the alarm stays on
        delays.append(score.delay_s)
    # and it is a slow walk-off, not a jump: nothing to see in the first frames, so the delay is not just N
    assert min(delays) >= CFG.n_frames - 1


def test_the_detector_works_on_the_kalman_filters_own_position_and_covariance():
    truth = _truth(200)
    rng = np.random.default_rng(11)
    kf, records = KalmanFilter(), []
    for p in truth:                                          # fixes: truth + 1 px noise, as a good matcher gives
        kf.step(p.t, np.array([p.x, p.y]) + rng.normal(0, TRACK_SIGMA, 2), TRACK_SIGMA ** 2 * np.eye(2))
        records.append(Record(idx=p.idx, t=p.t, source="", track=kf.track_state(p.idx, TrackMode.TRACKING)))
    honest = annotate_run(records, simulate_gps(truth, sigma_px=GPS_SIGMA, seed=4), CFG)
    assert _alarms(honest) == []
    spoofed = annotate_run(records, simulate_gps(truth, sigma_px=GPS_SIGMA, seed=4, walk_off=WalkOff(
        start_idx=80, speed_px_s=2.0, direction_deg=90.0)), CFG)
    score = score_trial(spoofed, start_idx=80)
    # the same bound as above: (3.7 + 3) σ / speed + N frames, with the filter's σ about that of one fix
    assert score.detected and score.delay_s <= (math.sqrt(13.8) + 3.0) * SIGMA_TOTAL / 2.0 + CFG.n_frames


def _with_alarms(pattern: str, dt: float = 0.5) -> list[Record]:
    """Records whose gps block has the alarm on where the pattern says 'A'; idx = 10 * position."""
    out = []
    for k, c in enumerate(pattern):
        track = TrackState(idx=10 * k, state=TrackMode.TRACKING, x=0.0, y=0.0, vx=0.0, vy=0.0, cov=np.eye(2))
        out.append(Record(idx=10 * k, t=k * dt, source="", track=track, extra={"gps": {"alarm": c == "A"}}))
    return out


def test_score_trial_measures_the_delay_from_the_start_of_the_walk_off():
    s = score_trial(_with_alarms("......AAAA"), start_idx=30)        # starts at position 3, alarm at position 6
    assert (s.frames, s.false_alarm, s.false_alarm_frames) == (10, False, 0)
    assert (s.detected, s.alarm_idx, s.delay_frames) == (True, 60, 3) and s.delay_s == pytest.approx(1.5)


def test_score_trial_counts_alarms_before_the_walk_off_as_false():
    s = score_trial(_with_alarms(".AA......A"), start_idx=50)
    assert (s.false_alarm, s.false_alarm_frames, s.detected, s.alarm_idx) == (True, 2, True, 90)
    honest = score_trial(_with_alarms(".AA......A"))                 # no walk-off at all: every alarm is false
    assert (honest.false_alarm, honest.false_alarm_frames, honest.detected) == (True, 3, None)


def test_score_trial_gives_no_credit_for_an_alarm_that_was_already_on():
    s = score_trial(_with_alarms("..AAAAAA.."), start_idx=40)         # on before the walk-off: not a detection
    assert s.false_alarm and s.detected is False and s.delay_s is None
    missed = score_trial(_with_alarms(".........."), start_idx=40)
    assert not missed.false_alarm and missed.detected is False and missed.alarm_idx is None


def test_score_trial_rejects_records_without_a_gps_block_and_an_unknown_start():
    plain = _records(_truth(5))
    with pytest.raises(ValueError, match="gps"):
        score_trial(plain)
    with pytest.raises(ValueError, match="start_idx"):
        score_trial(_with_alarms("....."), start_idx=1000)


# --- the script ----------------------------------------------------------------------------------------------------

def _spoof_check(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "spoof_check.py"), *args],
                          check=False, capture_output=True, text=True, env=env, cwd=ROOT)


def test_spoof_check_prints_the_table_and_writes_the_numbers(tmp_path):
    _write_truth(tmp_path / "truth.jsonl", n=30 * 200)      # 200 s at 30 fps; the script takes one frame per second
    out = tmp_path / "out"
    res = _spoof_check("--flight", str(tmp_path), "--seeds", "4", "--speeds", "1,4", "--m-per-px", "0.76",
                       "--out", str(out))
    assert res.returncode == 0, res.stderr
    assert "false alarm" in res.stdout and "delay" in res.stdout
    d = json.loads((out / "spoof_check.json").read_text())
    assert d["settings"]["seeds"] == 4 and d["settings"]["frames"] == 200
    assert d["honest"]["flights"] == 4 and d["honest"]["false_alarm_flights"] == 0
    rows = {r["speed_mps"]: r for r in d["walk_off"]}
    assert set(rows) == {1.0, 4.0} and all(r["detected"] == 4 for r in rows.values())
    assert rows[4.0]["delay_s_median"] < rows[1.0]["delay_s_median"]        # a faster walk-off shows sooner
    # the demo run: ordinary telemetry with the gps block, alarm on by the end
    demo = read_run(out / "spoof_demo.jsonl")
    assert len(demo) == 200 and demo[-1].extra["gps"]["alarm"] and not demo[0].extra["gps"]["alarm"]

    again = _spoof_check("--flight", str(tmp_path), "--seeds", "4", "--speeds", "1,4", "--m-per-px", "0.76")
    assert again.stdout == res.stdout                        # seeded: the same table every time


def test_spoof_check_fails_cleanly_on_a_missing_flight(tmp_path):
    res = _spoof_check("--flight", str(tmp_path / "nope"))
    assert res.returncode == 2 and "nope" in res.stderr and "Traceback" not in res.stderr


@pytest.mark.data
@pytest.mark.skipif(not (FLIGHT / "truth.jsonl").exists() or not MAP.with_suffix(".json").exists(),
                    reason="data/ not downloaded")
def test_spoof_check_on_the_irpin_flight(tmp_path):
    res = _spoof_check("--flight", str(FLIGHT), "--map", str(MAP), "--seeds", "5", "--out", str(tmp_path))
    assert res.returncode == 0, res.stderr
    d = json.loads((tmp_path / "spoof_check.json").read_text())
    assert d["honest"]["false_alarm_flights"] == 0
    assert all(r["detected"] == 5 for r in d["walk_off"])
    assert isinstance(read_run(tmp_path / "spoof_demo.jsonl")[0].extra["gps"]["lon"], float)
