"""Film a simulated phone take of the map on a screen (M2-08), or draw the check sheet of its effects.

    python scripts/screen_video.py --map data/hackatons-data/global_map.png --out data/sim_phone/task_calm --preset calm
    python scripts/screen_video.py --map data/irpin/maps/irpin_2025.png --out data/sim_phone/irpin_hard --preset hard \
        --seconds 30 --seed 2 --effect glare=1
    python scripts/screen_video.py --map data/hackatons-data/global_map.png --sheet outputs/screen_sheet.png

The take is an Irpin-style flight folder (flight.mp4, truth.jsonl, meta.json, frames/) with exact truth, marked
"simulated": it stands in for the real takes (M2-01) while building and measuring, it does not replace them.
See dronenav.sim.screen for the model and the effects.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from dronenav.config import add_shared_args, init_cli
from dronenav.io import load_map
from dronenav.sim.screen import EFFECTS, PRESETS, Camera, Screen, ScreenSim, write_screen_video


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WxH, e.g. 1280x720, got {text!r}") from None
    if w < 16 or h < 16:
        raise argparse.ArgumentTypeError(f"a size of at least 16x16 px, got {text!r}")
    return w, h


def _effect(text: str) -> tuple[str, float]:
    name, eq, value = text.partition("=")
    if not eq:
        raise argparse.ArgumentTypeError(f"expected NAME=STRENGTH, e.g. glare=0.5, got {text!r}")
    if name not in EFFECTS:
        raise argparse.ArgumentTypeError(f"unknown effect {name!r} (available: {', '.join(EFFECTS)})")
    try:
        return name, float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"the strength of {name} must be a number, got {value!r}") from None


def _x264(img: np.ndarray, crf: int, fps: float) -> np.ndarray:
    """One frame through the same H.264 encoder as the take, decoded back: what compression does to it."""
    from dronenav.viz.trajectory_video import _encoder
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "one.mp4"
        with _encoder(path, fps, (img.shape[1], img.shape[0]), crf=crf) as write:
            for _ in range(3):                                   # a few frames: x264 needs a little to settle
                write(img)
        cap = cv2.VideoCapture(str(path))
        ok, out = cap.read()
        cap.release()
    return out if ok else img


def sheet(map_img: np.ndarray, path: Path, seed: int = 0) -> Path:
    """One row per effect at strengths 0, 0.5 and 1, on the same frame (the middle of the cover for cover)."""
    cam = Camera(size=(320, 180), fps=10.0)
    mh, mw = map_img.shape[:2]
    screen = Screen(size=(min(2560, mw), min(1440, mh)))
    # come close enough that the view is about a quarter of the screen's width: any map gets a route
    dist = 0.25 * screen.size[0] * screen.pitch_mm / (2 * np.tan(np.radians(cam.hfov_deg) / 2))
    label_w = 150
    rows = []
    for name in EFFECTS:
        tiles = []
        for s in (0.0, 0.5, 1.0):
            sim = ScreenSim(map_img, seconds=4.0, seed=seed, effects={"grid": 0.0, name: s}, screen=screen,
                            camera=cam, distance_mm=float(dist))
            start, dur, _ = sim._cover
            i = round((start + dur / 2) * sim.fps) if name == "cover" and dur > 0 else len(sim) // 2
            img = sim.frame(min(i, len(sim) - 1))
            if name == "compression":
                img = _x264(img, 16 + round(20 * s), sim.fps)
            cv2.putText(img, f"{s:g}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(img)
        label = np.full((cam.size[1], label_w, 3), 24, np.uint8)
        cv2.putText(label, name, (8, cam.size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1,
                    cv2.LINE_AA)
        rows.append(np.hstack([label, *tiles]))
    out = np.vstack(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), out):
        raise OSError(f"cannot write {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--map", required=True, help="the map shown on the screen")
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--out", type=Path, help="the take's folder (an Irpin-style flight)")
    what.add_argument("--sheet", type=Path, help="draw the effects' check sheet to this PNG instead")
    ap.add_argument("--preset", default="calm", choices=sorted(PRESETS), help="M2-01's takes (default calm)")
    ap.add_argument("--effect", action="append", type=_effect, default=[], metavar="NAME=STRENGTH",
                    help=f"override one effect, 0 to 1; repeatable ({', '.join(EFFECTS)})")
    ap.add_argument("--seconds", type=float, default=20.0, help="length of the take (default 20)")
    ap.add_argument("--seed", type=int, default=0, help="the route, the hand and the noise (default 0)")
    ap.add_argument("--size", type=_size, default=(1280, 720), help="camera frame WxH (default 1280x720)")
    ap.add_argument("--fps", type=float, default=30.0, help="camera frame rate (default 30)")
    ap.add_argument("--screen", type=_size, default=(2560, 1440), help="screen WxH px (default 2560x1440, 27 in)")
    ap.add_argument("--pitch", type=float, default=0.233, help="screen pixel pitch, mm (default 0.233)")
    ap.add_argument("--zoom", type=float, default=1.0, help="screen px per map px in the viewer (default 1)")
    ap.add_argument("--distance", type=float, default=100.0, help="phone to screen, mm (default 100: phones focus from ~10 cm)")
    ap.add_argument("--speed", type=float, default=None, help="look-at speed on the screen, mm/s (default: preset)")
    add_shared_args(ap, device=False)            # renders with numpy and OpenCV: no device to pick
    a = ap.parse_args(argv)
    init_cli(a)
    try:
        if a.sheet is not None:
            path = sheet(load_map(a.map).image, a.sheet, a.seed)
            print(f"screen_video: sheet -> {path}")
            return 0
        out = write_screen_video(a.map, a.out, preset=a.preset, seconds=a.seconds, seed=a.seed,
                                 effects=dict(a.effect), screen=Screen(size=a.screen, pitch_mm=a.pitch, zoom=a.zoom),
                                 camera=Camera(size=a.size, fps=a.fps), distance_mm=a.distance, speed_mm_s=a.speed)
    except (OSError, ValueError, RuntimeError, cv2.error) as e:
        print(f"screen_video: error: {e}", file=sys.stderr)
        return 2
    print(f"screen_video: simulated take ({a.preset}, {a.seconds:g} s) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
