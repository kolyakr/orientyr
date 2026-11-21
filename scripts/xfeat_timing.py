"""M4-03: time per frame of the XFeat matcher against RootSIFT, on the same frames. Prints a markdown table.

Consecutive keyframes of a flight are matched to a window of the map around their true position (as in
scripts/preprocess_ablation.py), so the search (M1-05) stays out of the question and the rows differ only in the
matcher. Each frame is shrunk first so that one of its pixels covers a set number of map pixels (the M1-03 rule,
``--sift-target`` and ``--xfeat-target``; the scale is the flight's ``nadir_gsd_m`` over the map's ``m_per_px``),
then matched, fitted (dronenav.geometry) and scored against the truth. The time is the whole match call: the
resize, the frame's features, the matching. The map's features are prepared before, once, and reported apart.

Rows: RootSIFT; XFeat as it runs while tracking (the turn found for one frame is tried first for the next);
XFeat with nothing remembered (every turn for every frame: what a frame costs after the track is lost).

Without ``--flight`` the frames are cut-outs along a seeded route: of ``--map`` if given, else of a synthetic
map, so the script runs without data/. The first run downloads the XFeat weights (about 6 MB) into the torch
cache unless ``--no-download`` is given.

Usage:
    python scripts/xfeat_timing.py --flight data/irpin/flights/mild/2022 --map data/irpin/maps/irpin_2022.png
    python scripts/xfeat_timing.py                       # synthetic cut-outs
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from dronenav.config import add_shared_args, init_cli
from dronenav.geometry import localize
from dronenav.io import load_map
from dronenav.matching import RootSiftMatcher, XFeatConfig, XFeatMatcher, XFeatWeightsMissing
from dronenav.preprocess import scale_prior
from dronenav.scale import ScaleConfig, downscale_factor, match_scaled, resize_frame
from dronenav.sim import CutoutSource, cutout_route

ROOT = Path(__file__).resolve().parents[1]
CUTOUT_SIZE = (320, 240)          # synthetic frames: small, so the route has room to turn on a small map
CUTOUT_SCALE = (0.9, 1.1)         # map px per frame px along the synthetic route


def flight_items(flight: Path, n: int, start: int) -> tuple[dict, list]:
    """meta.json and ``n`` consecutive keyframes from the ``start``-th: (frame number, image path, true centre)."""
    meta = json.loads((flight / "meta.json").read_text())
    truth = {r["i"]: r for r in map(json.loads, (flight / "truth.jsonl").read_text().splitlines()) if r}
    keys = meta["keyframes"][start:start + n]
    return meta, [(k["i"], flight / k["file"], tuple(truth[k["i"]]["center_map_px"])) for k in keys]


def cutout_items(world: np.ndarray, n: int, seed: int) -> list:
    """``n`` consecutive cut-outs of ``world`` along a seeded route: (index, image, true centre)."""
    poses = cutout_route((world.shape[1], world.shape[0]), n, seed, frame_size=CUTOUT_SIZE, scale=CUTOUT_SCALE,
                         tilt_max_deg=3.0)
    src = CutoutSource(world, poses, CUTOUT_SIZE)
    return [(f.idx, f.image, src.truth(f.idx).center) for f in src]


def _image(item) -> np.ndarray:
    if isinstance(item, np.ndarray):
        return item
    img = cv2.imread(str(item))
    if img is None:
        raise SystemExit(f"cannot read the frame {item}")
    return img


def measure(name: str, matcher, device: str, world: np.ndarray, items: list, factor: float, half: int,
            good_px: float, map_s: float) -> dict:
    """One table row: every frame shrunk by ``factor``, matched to its window, fitted and scored."""
    h, w = world.shape[:2]
    ok, wrong, inl, err, ms, found, size = 0, 0, [], [], [], [], None
    passes0 = getattr(matcher, "frame_extractions", 0)
    for idx, item, (cx, cy) in items:
        frame = _image(item)
        win, off = world, (0, 0)
        if half > 0:
            x0 = int(min(max(0, cx - half), max(0, w - 2 * half)))
            y0 = int(min(max(0, cy - half), max(0, h - 2 * half)))
            win, off = world[y0:y0 + 2 * half, x0:x0 + 2 * half], (x0, y0)
        m = match_scaled(matcher, frame, win, factor, map_offset=off)       # m.ms: the resize and the match
        ms.append(m.ms)
        found.append(len(m))
        small = resize_frame(frame, factor)
        size = (small.shape[1], small.shape[0])
        fix = localize(idx, m, (frame.shape[1], frame.shape[0]))
        if fix is None or not fix.accepted:
            continue
        e = math.hypot(fix.x - cx, fix.y - cy)
        if e < good_px:
            ok += 1
            inl.append(fix.inliers)
            err.append(e)
        else:
            wrong += 1
    passes = (getattr(matcher, "frame_extractions", 0) - passes0) / len(items) if hasattr(
        matcher, "frame_extractions") else None
    return {"matcher": name, "device": device, "n": len(items), "frame": f"{size[0]} x {size[1]}",
            "factor": factor, "localised": ok, "wrong": wrong, "matches": float(np.median(found)),
            "inliers": float(np.median(inl)) if inl else 0.0, "err_px": float(np.median(err)) if err else None,
            "ms": float(np.median(ms)), "ms_mean": float(np.mean(ms)), "passes": passes, "map_s": map_s}


def table(title: str, rows: list[dict]) -> str:
    out = [f"**{title}**", "",
           "| matcher | device | frame px | localised | wrong fixes | matches (median) | inliers (median) | "
           "error px (median) | ms/frame (median) | ms/frame (mean) | network passes/frame | map features, s |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        err = "-" if r["err_px"] is None else f"{r['err_px']:.2f}"
        passes = "-" if r["passes"] is None else f"{r['passes']:.2f}"
        out.append(f"| {r['matcher']} | {r['device']} | {r['frame']} | {r['localised']}/{r['n']} | {r['wrong']} | "
                   f"{r['matches']:.0f} | {r['inliers']:.0f} | {err} | {r['ms']:.0f} | {r['ms_mean']:.0f} | "
                   f"{passes} | {r['map_s']:.1f} |")
    return "\n".join(out)


def _timed(build):
    t0 = time.perf_counter()
    matcher = build()
    return matcher, time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--flight", type=Path, help="a flight folder: meta.json, truth.jsonl, frames/ "
                                                "(default: cut-outs of the map along a seeded route)")
    ap.add_argument("--map", type=Path, help="the map; with --flight, the one its truth is written in "
                                             "(default without --flight: a synthetic map)")
    ap.add_argument("--n", type=int, default=60, help="consecutive keyframes (or cut-outs) to time")
    ap.add_argument("--start", type=int, default=0, help="the first keyframe's position in the flight")
    ap.add_argument("--half", type=int, default=800, help="half the side of the map window around the truth, px; "
                                                          "0 = match against the whole map")
    ap.add_argument("--good-px", type=float, default=10.0, help="a fix this close to the truth is localised")
    ap.add_argument("--scale", type=float, help="map px per frame px, instead of nadir_gsd_m / m_per_px")
    ap.add_argument("--sift-target", type=float, default=ScaleConfig().target_scale,
                    help="RootSIFT: map px per shrunk frame px (the M1-03 rule)")
    ap.add_argument("--xfeat-target", type=float, default=1.0,
                    help="XFeat: map px per shrunk frame px (it is not scale-invariant: the frame goes to the "
                         "map's own resolution)")
    ap.add_argument("--seed", type=int, default=3, help="the synthetic route's seed")
    ap.add_argument("--weights", type=Path, help="the XFeat weights file (default: the torch cache)")
    ap.add_argument("--no-download", action="store_true", help="fail if the XFeat weights are not there yet")
    ap.add_argument("--out", type=Path, default=Path("outputs/xfeat_timing"), help="where timing.md and .json go")
    ap.add_argument("--cache", default="outputs/cache", help="map feature cache; 'none' extracts every time")
    add_shared_args(ap)
    a = ap.parse_args(argv)
    device = init_cli(a)
    if a.n < 1 or a.start < 0 or a.half < 0 or a.good_px <= 0 or a.sift_target <= 0 or a.xfeat_target <= 0:
        ap.error("--n must be >= 1, --start and --half >= 0, --good-px and the targets > 0")
    if a.flight is not None and a.map is None:
        ap.error("--flight needs --map: the map the flight's truth is written in")
    cache = None if str(a.cache).lower() == "none" else a.cache

    def xfeat(world, **cfg):
        return XFeatMatcher(world, cache_dir=cache, cfg=XFeatConfig(**cfg), device=device, weights=a.weights,
                            download=not a.no_download)

    try:
        # before anything slow: missing weights must not cost a map load and a RootSIFT extraction
        XFeatMatcher(cache_dir=None, device=device, weights=a.weights, download=not a.no_download)
        level = None
        if a.map is not None:
            mp = load_map(a.map)
            world, level, what = mp.image, mp.level, a.map.name
        else:
            sys.path.insert(0, str(ROOT))           # the tests' seeded texture: a map when there is no data/
            from tests.synth import make_map
            world, what = make_map(2000, 1500, seed=0, shapes=2500), "a synthetic map"
        if a.flight is not None:
            if not (a.flight / "meta.json").is_file() or not (a.flight / "truth.jsonl").is_file():
                ap.error(f"no flight in {a.flight} (meta.json and truth.jsonl are expected there)")
            meta, items = flight_items(a.flight, a.n, a.start)
            if not items:
                ap.error(f"{a.flight} has no keyframes from position {a.start}")
            title = f"{a.flight.parent.name}/{a.flight.name} on {what}"
            if a.scale is not None:
                scale = a.scale
            elif level is not None and "m_per_px" in level and "nadir_gsd_m" in meta:
                scale = scale_prior(meta["nadir_gsd_m"], level["m_per_px"])
            else:
                ap.error(f"no scale for {a.map}: it needs nadir_gsd_m in meta.json and m_per_px in the map's "
                         "sidecar, or give --scale")
        else:
            items = cutout_items(world, a.n, a.seed)
            title, scale = f"cut-outs of {what}", a.scale or sum(CUTOUT_SCALE) / 2
        f_sift = downscale_factor(scale, ScaleConfig(target_scale=a.sift_target))
        f_xfeat = downscale_factor(scale, ScaleConfig(target_scale=a.xfeat_target))

        rows = []
        sift, s = _timed(lambda: RootSiftMatcher(world, cache_dir=cache))
        rows.append(measure("RootSIFT", sift, "cpu", world, items, f_sift, a.half, a.good_px, s))
        tracking, s = _timed(lambda: xfeat(world))
        rows.append(measure("XFeat", tracking, device, world, items, f_xfeat, a.half, a.good_px, s))
        # keep_min out of reach: no turn is ever kept, so every frame tries them all
        every, s = _timed(lambda: xfeat(world, keep_min=10**9))
        rows.append(measure("XFeat, every turn each frame", every, device, world, items, f_xfeat, a.half,
                            a.good_px, s))
    except XFeatWeightsMissing as e:
        print(f"xfeat_timing: error: {e}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as e:
        print(f"xfeat_timing: error: {e}", file=sys.stderr)
        return 2

    where = f"{2 * a.half} px window" if a.half else "whole map"
    text = table(f"{title} ({len(items)} consecutive frames, {where}, {scale:.2f} map px per frame px)", rows)
    print(text)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "timing.md").write_text(text)
    (a.out / "timing.json").write_text(json.dumps(
        {"flight": None if a.flight is None else str(a.flight), "map": None if a.map is None else str(a.map),
         "n": len(items), "half": a.half, "good_px": a.good_px, "scale": scale, "device": device, "rows": rows},
        indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
