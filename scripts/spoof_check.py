"""Measure the GPS spoof detector on a simulated flight: false-alarm rate and detection delay (M5-07).

    python scripts/spoof_check.py --flight data/irpin/flights/mild/2022 --map data/irpin/maps/irpin_2022.png
    python scripts/spoof_check.py --flight data/irpin/flights/mild/2022 --map data/irpin/maps/irpin_2022.png \
        --run outputs/mild2022 --out outputs/spoof

The flight's truth.jsonl is flown --seeds times. Each time a GPS receiver is simulated from the truth (Gaussian
noise of --gps-sigma-m) and compared with the Kalman filter's position, frame by frame:

- once honest: any alarm is a false alarm;
- once per --speeds value with a slow walk-off on top, starting at a seeded time in the middle of the flight and
  going in a seeded direction: the delay is the time from its start to the alarm.

What the filter is fed with: by default simulated fixes (truth + --fix-sigma-px noise, one per second), so only the
truth is needed. With --run the accepted fixes of a real engine run of that flight (its run.jsonl, e.g. from
scripts/accuracy.py --map ... --frames ... --run ...) are used instead: real matching errors and real misses.
With --out the numbers go to spoof_check.json and one spoofed flight to spoof_demo.jsonl (ordinary telemetry with
the detector's block in extra["gps"]).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from dronenav.config import add_shared_args, init_cli
from dronenav.georef import GeorefBlock, load_georef
from dronenav.sim.gps import TruthPoint, WalkOff, read_truth_track, simulate_gps
from dronenav.telemetry import Record, TelemetryError, read_run, write_run
from dronenav.tracking.kalman import KalmanFilter, StepKind
from dronenav.tracking.spoof import SpoofConfig, TrialScore, annotate_run, score_trial
from dronenav.types import Fix, TrackMode

log = logging.getLogger("dronenav.spoof_check")

START_WINDOW = (0.2, 0.6)     # the walk-off starts within this part of the flight: the filter has settled before
                              # it, and enough flight is left after it for the slowest walk-off to show


def _floats(text: str) -> list[float]:
    values = [float(v) for v in text.split(",") if v.strip()]
    if not values or any(not np.isfinite(v) or v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"expected positive numbers separated by commas, got {text!r}")
    return values


def _keyframes(flight: Path, rate: float) -> list[TruthPoint]:
    """The truth at about ``rate`` frames per second (the flight's own spacing comes from its timestamps)."""
    full = read_truth_track(flight)
    if len(full) < 2 or full[-1].t <= full[0].t:
        return full
    idx_per_s = (full[-1].idx - full[0].idx) / (full[-1].t - full[0].t)
    every = max(1, round(idx_per_s / rate))
    return full if every == 1 else read_truth_track(flight, every=every)


def _filtered(points: list[TruthPoint], fixes: dict[int, Fix | np.ndarray | None], R: np.ndarray | None,
              sources: dict[int, str]) -> list[Record]:
    """The Kalman filter flown over the frames: one record per frame from the first fix on.

    ``fixes[idx]`` is the frame's measurement: a Fix (its own covariance), a raw position (covariance ``R``) or
    None (the filter coasts and its covariance grows, which the detector then takes into account)."""
    kf, out = KalmanFilter(), []
    for p in points:
        z = fixes.get(p.idx)
        step = kf.step(p.t, z, None if isinstance(z, Fix) else R)
        if step.kind is StepKind.NONE:
            continue                                   # no position yet: nothing to compare the GPS with
        mode = TrackMode.TRACKING if step.kind in (StepKind.INIT, StepKind.HIT) else TrackMode.LOST
        track = kf.track_state(p.idx, mode, z if isinstance(z, Fix) else None)
        out.append(Record(idx=p.idx, t=p.t, source=sources.get(p.idx, ""), track=track))
    if not out:
        raise ValueError("the filter never started: no accepted fix in the run")
    return out


def _stats(values: list[float]) -> tuple[float | None, float | None, float | None]:
    """Median, 95th percentile and maximum, None when nothing was measured."""
    if not values:
        return None, None, None
    return float(np.median(values)), float(np.percentile(values, 95)), float(max(values))


def _num(v: float | None, digits: int = 1) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 3)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flight", required=True, type=Path, help="flight folder holding truth.jsonl, or the file")
    ap.add_argument("--map", type=Path, help="the map image: its sidecar JSON gives lon/lat and metres per px")
    ap.add_argument("--m-per-px", type=float, default=1.0,
                    help="metres per map px when there is no --map (default 1: metres below are then map px)")
    ap.add_argument("--run", type=Path, help="feed the filter with the fixes of this run folder (run.jsonl) "
                                             "instead of simulated ones")
    ap.add_argument("--rate", type=float, default=1.0, help="frames per second taken from the truth (default 1, "
                                                            "the engine's keyframes); ignored with --run")
    ap.add_argument("--fix-sigma-px", type=float, default=1.0,
                    help="noise of the simulated fixes per axis, map px (default 1); ignored with --run")
    ap.add_argument("--gps-sigma-m", type=float, default=3.0, help="GPS noise per axis, metres (default 3)")
    ap.add_argument("--speeds", type=_floats, default=[0.5, 1.0, 2.0, 5.0], metavar="M/S[,M/S...]",
                    help="walk-off speeds to try, metres per second (default 0.5,1,2,5)")
    ap.add_argument("--threshold", type=float, default=SpoofConfig.threshold_chi2,
                    help=f"normalised gap d² above which a frame is suspect (default {SpoofConfig.threshold_chi2:g})")
    ap.add_argument("--n-frames", type=int, default=SpoofConfig.n_frames,
                    help=f"suspect frames in a row that raise the alarm (default {SpoofConfig.n_frames})")
    ap.add_argument("--seeds", type=int, default=20, help="simulated flights per row (default 20): seeds 0..N-1")
    ap.add_argument("--out", type=Path, help="folder for spoof_check.json and spoof_demo.jsonl")
    add_shared_args(ap, device=False)       # numpy only: nothing here runs on a device
    a = ap.parse_args(argv)
    init_cli(a)
    try:
        if a.seeds < 1:
            raise ValueError(f"--seeds must be >= 1, got {a.seeds}")
        if not a.rate > 0 or not a.fix_sigma_px > 0 or not a.gps_sigma_m > 0 or not a.m_per_px > 0:
            raise ValueError("--rate, --fix-sigma-px, --gps-sigma-m and --m-per-px must be > 0")
        block: GeorefBlock | None = None
        m_per_px = a.m_per_px
        if a.map is not None:
            sidecar = a.map.with_suffix(".json")
            if not sidecar.is_file():
                raise FileNotFoundError(f"{a.map} has no sidecar {sidecar.name}: give --m-per-px instead of --map")
            block = load_georef(sidecar).map         # the truth is in the px of the sidecar's "map" block
            m_per_px = block.m_per_px
        gps_sigma_px = a.gps_sigma_m / m_per_px
        cfg = SpoofConfig(gps_sigma_px=gps_sigma_px, threshold_chi2=a.threshold, n_frames=a.n_frames)

        run_fixes = sources = None
        if a.run is not None:
            run = {r.idx: r for r in read_run(a.run / "run.jsonl")}
            by_idx = {p.idx: p for p in read_truth_track(a.flight)}
            points = [by_idx[i] for i in sorted(run) if i in by_idx]     # the truth's clock: t in seconds
            if not points:
                raise ValueError(f"none of the run's {len(run)} frames is in the flight's truth: is it a run of "
                                 "this flight?")
            run_fixes = {i: run[i].track.fix for i in run}
            sources = {i: run[i].source for i in run}
        else:
            points = _keyframes(a.flight, a.rate)
        if len(points) < 4 * cfg.n_frames:
            raise ValueError(f"only {len(points)} frames: too short a flight to measure anything")
        R = a.fix_sigma_px ** 2 * np.eye(2)

        def records_for(seed: int) -> list[Record]:
            if run_fixes is not None:
                return _filtered(points, run_fixes, None, sources)
            rng = np.random.default_rng([seed, 1])          # its own stream: not the GPS noise of the same seed
            fixes = {p.idx: np.array([p.x, p.y]) + rng.normal(0.0, a.fix_sigma_px, 2) for p in points}
            return _filtered(points, fixes, R, {})

        honest: list[TrialScore] = []
        rows: dict[float, list[TrialScore]] = {v: [] for v in a.speeds}
        demo: list[Record] | None = None
        for seed in range(a.seeds):
            records = records_for(seed)
            honest.append(score_trial(annotate_run(
                records, simulate_gps(points, sigma_px=gps_sigma_px, seed=seed, block=block), cfg)))
            rng = np.random.default_rng([seed, 2])          # where and when the spoofer starts, per seed
            lo, hi = (int(f * (len(records) - 1)) for f in START_WINDOW)
            start_idx = records[int(rng.integers(lo, hi + 1))].idx
            direction = float(rng.uniform(0.0, 360.0))
            for speed in a.speeds:
                walk = WalkOff(start_idx=start_idx, speed_px_s=speed / m_per_px, direction_deg=direction)
                out = annotate_run(records, simulate_gps(points, sigma_px=gps_sigma_px, seed=seed, block=block,
                                                         walk_off=walk), cfg)
                rows[speed].append(score_trial(out, start_idx=start_idx))
                if demo is None:
                    demo = out
    except (OSError, ValueError, TelemetryError) as e:
        print(f"spoof_check: error: {e}", file=sys.stderr)
        return 2

    n_frames = len(records)
    seconds = points[-1].t - points[0].t
    unit = "m" if (a.map is not None or a.m_per_px != 1.0) else "px"
    fa_flights, fa_frames = sum(s.false_alarm for s in honest), sum(s.false_alarm_frames for s in honest)
    total_frames = sum(s.frames for s in honest)
    print(f"flight: {n_frames} frames over {seconds:.0f} s, {m_per_px:.3f} m/px; filter fed by "
          + (f"the run's fixes ({sum(1 for r in records if r.track.state is TrackMode.TRACKING)} used)"
             if run_fixes is not None else f"simulated fixes (sigma {a.fix_sigma_px:g} px)"))
    print(f"GPS: sigma {a.gps_sigma_m:g} {unit} = {gps_sigma_px:.2f} px; alarm: d2 > {cfg.threshold_chi2:g} for "
          f"{cfg.n_frames} frames in a row; {a.seeds} seeded flights per row")
    print()
    print(f"honest GPS: false alarm in {fa_flights}/{len(honest)} flights "
          f"({100 * fa_flights / len(honest):.1f} %), {fa_frames}/{total_frames} frames "
          f"({100 * fa_frames / total_frames:.3f} %)")
    print()
    print(f"{'walk-off ' + unit + '/s':>14} {'detected':>9} {'false alarm':>12} {'delay s: median':>16} {'p95':>6} "
          f"{'max':>6} {'drift at alarm ' + unit:>18}")
    table = []
    for speed, scores in rows.items():
        delays = [s.delay_s for s in scores if s.detected]
        med, p95, worst = _stats(delays)
        early = sum(s.false_alarm for s in scores)
        detected = sum(bool(s.detected) for s in scores)
        drift = None if med is None else med * speed
        print(f"{speed:>14g} {f'{detected}/{len(scores)}':>9} {f'{early}/{len(scores)}':>12} {_num(med):>16} "
              f"{_num(p95):>6} {_num(worst):>6} {_num(drift):>18}")
        table.append({"speed_mps": speed, "speed_px_s": round(speed / m_per_px, 4), "flights": len(scores),
                      "detected": detected, "false_alarm_flights": early, "delay_s_median": _round(med),
                      "delay_s_p95": _round(p95), "delay_s_max": _round(worst),
                      "drift_at_alarm_m_median": _round(drift)})

    if a.out is not None:
        a.out.mkdir(parents=True, exist_ok=True)
        result = {
            "settings": {"flight": str(a.flight), "map": None if a.map is None else str(a.map),
                         "run": None if a.run is None else str(a.run), "m_per_px": m_per_px, "frames": n_frames,
                         "seconds": round(seconds, 3), "seeds": a.seeds,
                         "fixes": "run" if run_fixes is not None else "simulated",
                         "fix_sigma_px": None if run_fixes is not None else a.fix_sigma_px,
                         "gps_sigma_m": a.gps_sigma_m, "gps_sigma_px": round(gps_sigma_px, 4),
                         "threshold_chi2": cfg.threshold_chi2, "n_frames": cfg.n_frames},
            "honest": {"flights": len(honest), "frames": total_frames, "false_alarm_flights": fa_flights,
                       "false_alarm_frames": fa_frames, "false_alarm_rate": round(fa_flights / len(honest), 4),
                       "false_alarm_frame_rate": round(fa_frames / total_frames, 6)},
            "walk_off": table,
        }
        (a.out / "spoof_check.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        write_run(a.out / "spoof_demo.jsonl", demo)
        log.info("wrote spoof_check.json and spoof_demo.jsonl to %s", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
