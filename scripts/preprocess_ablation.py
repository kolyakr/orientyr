"""M4-05: what each preprocessing step buys. Prints a markdown table per map: step vs localised vs inliers vs error.

Keyframes of one Irpin flight are matched with RootSIFT against one or more maps, once per variant of
dronenav.preprocess: no preprocessing, each step alone (CLAHE, scale normalisation, grey-world), and together.
The photometric steps are applied to the map and to the frame with the same settings. A map of another year than
the flight's imagery is the cross-epoch case (another sensor, season and light): that is where preprocessing has
to earn its place; the flight's own map is the control.

Each frame is matched to a window of the map around its true position, as in scripts/scale_table.py, so the
search (M1-05) stays out of the question and only the preprocessing changes between rows (``--half 0`` matches
against the whole map instead). The scale for the normalisation is the prior the spec names: the flight's
``nadir_gsd_m`` (meta.json) over the map's ``m_per_px`` (its sidecar). The truth is the frame centre from
truth.jsonl, moved onto each map by the sidecars' ``alignment_to_2025`` shifts.

Usage:
    python scripts/preprocess_ablation.py --flight data/irpin/flights/mild/2022 \
        --maps data/irpin/maps/irpin_2025.png data/irpin/maps/irpin_2017.png data/irpin/maps/irpin_2022.png
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from dronenav.config import add_shared_args, init_cli
from dronenav.geometry import localize
from dronenav.io import MapImage, load_map
from dronenav.matching import RootSiftMatcher
from dronenav.preprocess import PreprocessConfig, preprocess, scale_prior
from dronenav.scale import ScaleConfig

# name -> config; the order of the table rows. "none" is the baseline every other row is read against.
VARIANTS = {
    "none": PreprocessConfig(clahe=False, scale_norm=False, grey_world=False),
    "CLAHE": PreprocessConfig(clahe=True, scale_norm=False, grey_world=False),
    "scale": PreprocessConfig(clahe=False, scale_norm=True, grey_world=False),
    # the spec's literal "a frame pixel covers about one map pixel", against the M1-03 rule's 0.65 in the row above
    "scale to 1.0": PreprocessConfig(clahe=False, scale_norm=True, grey_world=False,
                                     scale=ScaleConfig(target_scale=1.0)),
    "grey-world": PreprocessConfig(clahe=False, scale_norm=False, grey_world=True),
    "CLAHE + scale": PreprocessConfig(clahe=True, scale_norm=True, grey_world=False),
    "all": PreprocessConfig(clahe=True, scale_norm=True, grey_world=True),
}


def _shift(level: dict | None) -> tuple[float, float]:
    """A map's ``alignment_to_2025`` shift (px): a point at p on this map is at p + shift on the 2025 map.
    (0, 0) without a sidecar: then the maps are taken as aligned."""
    try:
        dx, dy = level["alignment_to_2025"]["shift_px"]
    except (KeyError, TypeError, ValueError):
        return 0.0, 0.0
    return float(dx), float(dy)


def _flight_map_level(flight: Path, meta: dict, flight_map: Path | None) -> dict | None:
    """The sidecar block of the map the flight's truth is written in (read without loading the image)."""
    path = flight_map or flight.parents[2] / meta.get("map", "")
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return None
    return json.loads(sidecar.read_text()).get("map")


def load_items(flight: Path, n: int) -> tuple[dict, list[tuple[int, Path, tuple[float, float]]]]:
    """meta.json and ``n`` evenly spaced keyframes: (frame number, image path, true centre on the flight's map)."""
    meta = json.loads((flight / "meta.json").read_text())
    truth = {r["i"]: r for r in map(json.loads, (flight / "truth.jsonl").read_text().splitlines()) if r}
    keys = meta["keyframes"]
    picked = sorted({round(float(v)) for v in np.linspace(0, len(keys) - 1, min(n, len(keys)))})
    return meta, [(keys[k]["i"], flight / keys[k]["file"], tuple(truth[keys[k]["i"]]["center_map_px"]))
                  for k in picked]


def measure(name: str, cfg: PreprocessConfig, matcher: RootSiftMatcher, map_ready: np.ndarray, items: list,
            move: tuple[float, float], scale: float, half: int, good_px: float) -> dict:
    """One table row: every frame preprocessed with ``cfg``, matched, fitted and scored against the truth."""
    h, w = map_ready.shape[:2]
    ok, wrong, accepted, inl, err, ms = 0, 0, 0, [], [], []
    for idx, path, (tx, ty) in items:
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"cannot read the frame {path}")
        cx, cy = tx + move[0], ty + move[1]
        win, off = map_ready, (0, 0)
        if half > 0:
            x0, y0 = int(min(max(0, cx - half), max(0, w - 2 * half))), int(min(max(0, cy - half), max(0, h - 2 * half)))
            win, off = map_ready[y0:y0 + 2 * half, x0:x0 + 2 * half], (x0, y0)
        t0 = time.perf_counter()
        p = preprocess(frame, cfg, scale=scale)
        m = p.restore(matcher.match(p.image, win, map_offset=off))
        ms.append((time.perf_counter() - t0) * 1e3)
        fix = localize(idx, m, p.size)
        if fix is None or not fix.accepted:
            continue
        accepted += 1
        e = math.hypot(fix.x - cx, fix.y - cy)
        if e < good_px:
            ok += 1
            inl.append(fix.inliers)
            err.append(e)
        else:
            wrong += 1
    return {"variant": name, "n": len(items), "accepted": accepted, "localised": ok, "wrong": wrong,
            "inliers": float(np.median(inl)) if inl else 0.0, "err_px": float(np.median(err)) if err else None,
            "ms": float(np.median(ms))}


def table(title: str, rows: list[dict]) -> str:
    out = [f"**{title}**", "", "| step | localised | wrong fixes | inliers (median) | error px (median) | ms (median) |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        err = "-" if r["err_px"] is None else f"{r['err_px']:.2f}"
        out.append(f"| {r['variant']} | {r['localised']}/{r['n']} | {r['wrong']} | {r['inliers']:.0f} | {err} | "
                   f"{r['ms']:.0f} |")
    return "\n".join(out)


def ablate(mp: MapImage, items: list, move: tuple[float, float], scale: float, half: int, good_px: float,
           cache: Path) -> list[dict]:
    """All variants on one map. The map is prepared once per photometric setting (CLAHE, grey-world) and its
    features cached; variants that differ only in the frame's scale share it."""
    rows = {}
    for photometric in sorted({(c.clahe, c.grey_world) for c in VARIANTS.values()}):
        clahe, grey = photometric
        map_ready = preprocess(mp.image, PreprocessConfig(clahe=clahe, scale_norm=False, grey_world=grey)).image
        matcher = RootSiftMatcher(map_ready, cache_dir=cache)
        for name, cfg in VARIANTS.items():
            if (cfg.clahe, cfg.grey_world) == photometric:
                rows[name] = measure(name, cfg, matcher, map_ready, items, move, scale, half, good_px)
    return [rows[name] for name in VARIANTS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--flight", type=Path, required=True, help="a flight folder: meta.json, truth.jsonl, frames/")
    ap.add_argument("--maps", type=Path, nargs="+", required=True,
                    help="maps to match against (with their sidecar JSON next to them); another year than the "
                         "flight's is the cross-epoch case")
    ap.add_argument("--flight-map", type=Path, help="the map the flight's truth is written in "
                                                    "(default: meta.json's 'map', in the flight's dataset folder)")
    ap.add_argument("--n", type=int, default=60, help="keyframes to use, evenly spaced over the flight")
    ap.add_argument("--half", type=int, default=800, help="half the side of the map window around the truth, px; "
                                                          "0 = match against the whole map")
    ap.add_argument("--good-px", type=float, default=10.0, help="a fix this close to the truth is localised")
    ap.add_argument("--scale", type=float, help="map px per frame px, instead of nadir_gsd_m / m_per_px")
    ap.add_argument("--out", type=Path, default=Path("outputs/preprocess_ablation"), help="where ablation.json goes")
    ap.add_argument("--cache", type=Path, default=Path("outputs/cache"), help="map feature cache")
    add_shared_args(ap)
    args = ap.parse_args()
    # Picked and logged like in every matching script; RootSIFT runs on the CPU, a torch matcher takes this device.
    init_cli(args)
    if not (args.flight / "meta.json").is_file() or not (args.flight / "truth.jsonl").is_file():
        ap.error(f"no flight in {args.flight} (meta.json and truth.jsonl are expected there)")
    if args.n < 1 or args.half < 0 or args.good_px <= 0:
        ap.error("--n must be >= 1, --half >= 0 and --good-px > 0")
    meta, items = load_items(args.flight, args.n)
    from_flight = _shift(_flight_map_level(args.flight, meta, args.flight_map))
    where = f"{2 * args.half} px window" if args.half else "whole map"
    saved = []
    for path in args.maps:
        try:
            mp = load_map(path)
        except (FileNotFoundError, ValueError) as e:
            ap.error(str(e))
        to_map = _shift(mp.level)
        move = (from_flight[0] - to_map[0], from_flight[1] - to_map[1])      # flight map px -> this map's px
        if args.scale is not None:
            scale, how = args.scale, "--scale"
        elif mp.level is not None and "m_per_px" in mp.level and "nadir_gsd_m" in meta:
            scale = scale_prior(meta["nadir_gsd_m"], mp.level["m_per_px"])
            how = f"nadir_gsd_m {meta['nadir_gsd_m']:.4g} / m_per_px {mp.level['m_per_px']:.4g}"
        else:
            ap.error(f"no scale prior for {path}: it needs nadir_gsd_m in meta.json and m_per_px in the map's "
                     "sidecar, or give --scale")
        print(f"{path.name}: scale prior {scale:.3f} map px per frame px ({how})", flush=True)
        rows = ablate(mp, items, move, scale, args.half, args.good_px, args.cache)
        print("\n" + table(f"{args.flight.parent.name}/{args.flight.name} on {path.name} "
                           f"({len(items)} keyframes, {where})", rows) + "\n", flush=True)
        saved += [{"map": path.name, "scale": scale, **r} for r in rows]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ablation.json").write_text(json.dumps(
        {"flight": str(args.flight), "n": len(items), "half": args.half, "good_px": args.good_px, "rows": saved},
        indent=1))


if __name__ == "__main__":
    main()
