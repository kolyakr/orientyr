"""Draw the chaos check sheets (M6-02): one image per degradation, to judge each strength by eye.

    python scripts/chaos_sheet.py                                          # cut-outs of a synthetic map, no data/
    python scripts/chaos_sheet.py --map data/irpin/maps/irpin_2022.png     # cut-outs of a real map
    python scripts/chaos_sheet.py --flight data/irpin/flights/mild/2022    # frames of a pre-rendered flight

Writes blur.png, noise.png, jpeg.png, gamma.png, glare.png, occlusion.png (one frame at every strength),
drop.png and jam.png (a strip of consecutive frames per strength: what is missing, what is jammed) and
compass.png (the heading error per frame, plotted) into --out. Every sheet names its seed; the same seed draws
the same sheets.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

from dronenav.sim import (
    DEGRADATIONS,
    ChaosSource,
    CompassError,
    CutoutSource,
    Drop,
    FlightSource,
    ImageDegradation,
    Jam,
    cutout_route,
)
from dronenav.types import Frame

ROOT = Path(__file__).resolve().parents[1]
TILE_W = 320                    # one frame of an image sheet
STRIP_W = 1560                  # all the thumbnails of a strip row together (each 128..320 px)
LABEL_W = 110                   # the strength column of a strip
BAR = 26                        # height of a caption line
PAGE, INK, DIM = (30, 30, 30), (235, 235, 235), (150, 150, 150)
COLOURS = [(160, 160, 160), (70, 170, 60), (200, 140, 30), (30, 140, 230), (50, 50, 220)]      # weak -> strong, BGR


class _Listed:
    """The frames already read, as a source (the chaos is applied to these few frames, not the whole flight)."""

    def __init__(self, frames: list[Frame]):
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)


def load_frames(a: argparse.Namespace) -> tuple[list[Frame], str]:
    """The strip of consecutive frames the sheets are drawn from, with priors, and a name for the headers."""
    if a.flight:
        src = FlightSource(a.flight, fps=a.fps, keyframes=a.keyframes, start_s=a.start_s, priors=True)
        return list(itertools.islice(src, a.frames)), f"flight {Path(a.flight).parent.name}/{Path(a.flight).name}"
    if a.map:
        world = cv2.imread(str(a.map), cv2.IMREAD_COLOR)
        if world is None:
            raise FileNotFoundError(f"cannot read the map {a.map}")
        name, scale = f"cut-outs of {Path(a.map).name}", (0.8, 1.25)
    else:
        sys.path.insert(0, str(ROOT))               # the tests' seeded texture: a map when there is no data/
        from tests.synth import make_map
        # zoomed in (0.5..0.8 map px per frame px): a 1200 x 900 map then has room for the route to turn in
        world, name, scale = make_map(1200, 900, seed=0), "cut-outs of the synthetic map", (0.5, 0.8)
    size = (world.shape[1], world.shape[0])
    poses = cutout_route(size, a.frames, a.seed, frame_size=(320, 240), scale=scale)
    return list(CutoutSource(world, poses, (320, 240), fps=a.fps, priors=True)), name


# --- drawing ---------------------------------------------------------------------------------------------------

def _text(img: np.ndarray, text: str, org: tuple[int, int], colour=INK, scale: float = 0.5) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def _fit(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(image, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def _captioned(tile: np.ndarray, caption: str, colour=INK) -> np.ndarray:
    bar = np.full((BAR, tile.shape[1], 3), PAGE, np.uint8)
    _text(bar, caption, (6, 18), colour, 0.45)
    return np.vstack([bar, tile])


def _page(title: str, body: np.ndarray) -> np.ndarray:
    """The sheet: a title line over the body, on a dark page with a margin."""
    head = np.full((BAR + 8, body.shape[1], 3), PAGE, np.uint8)
    _text(head, title, (6, 22), INK, 0.6)
    return cv2.copyMakeBorder(np.vstack([head, body]), 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=PAGE)


def image_sheet(cls: type[ImageDegradation], frame: Frame, strengths: list[float], seed: int, src: str) -> np.ndarray:
    """The same frame at every strength, side by side; each tile says how far its pixels moved."""
    tiles = []
    for s in strengths:
        out = cls(s, seed=seed).image(frame.image, frame.idx)
        diff = float(np.abs(out.astype(np.int16) - frame.image).mean())
        tiles.append(_captioned(_fit(out, TILE_W), f"strength {s:g}   mean |diff| {diff:.1f}"))
    return _page(f"{cls.name}   seed {seed}   frame {frame.idx} of the {src}", np.hstack(tiles))


def strip_sheet(name: str, chains: dict[float, list], frames: list[Frame], seed: int, src: str,
                note: str) -> np.ndarray:
    """One row of the consecutive frames per strength. A dropped frame is an empty, crossed slot; frames that
    arrive are drawn as they arrive (a jammed one is noise)."""
    w = int(np.clip(STRIP_W // len(frames), 128, TILE_W))
    h = _fit(frames[0].image, w).shape[0]
    rows = []
    for s, chain in chains.items():
        out = {f.idx: f for f in ChaosSource(_Listed(frames), chain)}
        label = np.full((BAR + h, LABEL_W, 3), PAGE, np.uint8)
        _text(label, f"strength {s:g}", (4, BAR + 20), INK, 0.45)
        _text(label, f"{len(frames) - len(out)} lost" if name == "drop" else
              f"{sum(not np.array_equal(out[f.idx].image, f.image) for f in frames)} jammed", (4, BAR + 42), DIM, 0.45)
        cells = [label]
        for f in frames:
            if f.idx in out:
                cells.append(_captioned(_fit(out[f.idx].image, w), f"{f.idx}  t {f.t:g}s"))
            else:
                gap = np.full((h, w, 3), (18, 18, 18), np.uint8)
                cv2.line(gap, (0, 0), (w - 1, h - 1), (50, 50, 200), 2, cv2.LINE_AA)
                cv2.line(gap, (0, h - 1), (w - 1, 0), (50, 50, 200), 2, cv2.LINE_AA)
                cells.append(_captioned(gap, f"{f.idx}  dropped", (90, 90, 230)))
        rows.append(np.hstack(cells))
    return _page(f"{name}   seed {seed}   {len(frames)} consecutive frames of the {src}   {note}", np.vstack(rows))


def compass_sheet(frames: list[Frame], strengths: list[float], seed: int, src: str) -> np.ndarray:
    """The heading error of each frame's prior (degraded minus original, degrees), one line per strength, and a
    table of the bias and the mean |error|."""
    W, H, left, top, bottom = 1100, 470, 60, 20, 40
    plot = np.full((H, W, 3), 245, np.uint8)
    errors = {}
    for s in strengths:
        out = list(ChaosSource(_Listed(frames), [CompassError(s, seed=seed)]))
        errors[s] = [(o.heading_prior_deg - f.heading_prior_deg + 180.0) % 360.0 - 180.0
                     for o, f in zip(out, frames, strict=True)]
    lim = max(10.0, 5.0 * np.ceil(max(abs(e) for es in errors.values() for e in es) / 5.0 + 0.2))
    x0, x1, y0, y1 = left, W - 330, top, H - bottom

    def at(k: int, e: float) -> tuple[int, int]:
        x = x0 + (x1 - x0) * (k / max(1, len(frames) - 1))
        return round(x), round((y0 + y1) / 2 - e / lim * (y1 - y0) / 2)

    for e in np.arange(-lim, lim + 1e-9, 5.0):                       # grid: every 5 degrees
        cv2.line(plot, at(0, e), at(len(frames) - 1, e), (150, 150, 150) if e == 0 else (220, 220, 220), 1)
        _text(plot, f"{e:+.0f}", (8, at(0, e)[1] + 4), (90, 90, 90), 0.4)
    for k in range(0, len(frames), max(1, len(frames) // 12)):
        _text(plot, str(frames[k].idx), (at(k, 0)[0] - 8, y1 + 22), (90, 90, 90), 0.4)
    _text(plot, "frame idx", (x1 - 60, H - 6), (90, 90, 90), 0.4)
    _text(plot, "error, deg", (8, 14), (90, 90, 90), 0.4)
    for n, (s, es) in enumerate(errors.items()):
        colour = COLOURS[n % len(COLOURS)]
        pts = np.array([at(k, e) for k, e in enumerate(es)], np.int32)
        cv2.polylines(plot, [pts], False, colour, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(plot, tuple(int(v) for v in p), 3, colour, -1, cv2.LINE_AA)
        comp = CompassError(s, seed=seed)
        y = top + 30 + 44 * n
        cv2.line(plot, (x1 + 20, y - 4), (x1 + 50, y - 4), colour, 3)
        _text(plot, f"strength {s:g}", (x1 + 60, y), (40, 40, 40), 0.5)
        _text(plot, f"bias {comp.bias_deg + 0.0:+.1f}  sigma {comp.sigma_deg:.1f}  mean |err| {np.abs(es).mean():.1f}",
              (x1 + 60, y + 18), (90, 90, 90), 0.4)
    return _page(f"compass   seed {seed}   heading prior error on {len(frames)} frames of the {src}", plot)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flight", default=None, help="flight folder (meta.json, truth.jsonl, flight.mp4 or frames/)")
    ap.add_argument("--keyframes", action="store_true", help="with --flight: read frames/ instead of the video")
    ap.add_argument("--start-s", type=float, default=0.0, help="with --flight: where the strip starts, s")
    ap.add_argument("--map", default=None, help="map image to cut frames from (default: a synthetic map)")
    ap.add_argument("--fps", type=float, default=1.0, help="frames per second of the strip (default 1)")
    ap.add_argument("--frames", type=int, default=12, help="consecutive frames in the strips (default 12)")
    ap.add_argument("--seed", type=int, default=0, help="seed of every degradation and of the cut-out route")
    ap.add_argument("--strengths", default="0,0.25,0.5,0.75,1", help="strengths to show, comma-separated")
    ap.add_argument("--out", default="outputs/chaos_sheets", help="folder for the sheets")
    a = ap.parse_args(argv)
    try:
        strengths = [float(s) for s in a.strengths.split(",")]
    except ValueError:
        ap.error(f"--strengths must be numbers separated by commas, got {a.strengths!r}")
    if a.frames < 2 or a.seed < 0 or not strengths or not all(0 <= s <= 1 for s in strengths):
        ap.error("--frames must be at least 2, --seed >= 0, --strengths within [0, 1]")
    if a.flight and a.map:
        ap.error("give --flight or --map, not both")
    try:
        frames, src = load_frames(a)
        if len(frames) < 2:
            raise ValueError(f"the {src} gave {len(frames)} frame(s) at {a.fps:g} fps: a strip needs at least 2")
        sheets = {}
        for name, cls in DEGRADATIONS.items():
            if issubclass(cls, ImageDegradation):
                sheets[name] = image_sheet(cls, frames[len(frames) // 2], strengths, a.seed, src)
        sheets["drop"] = strip_sheet("drop", {s: [Drop(s, seed=a.seed)] for s in strengths}, frames, a.seed, src,
                                     "each frame lost with p = strength x 0.5; idx keeps its gaps")
        # the window opens after a few clean frames and, at full strength, closes a few before the end
        first = max(1, len(frames) // 6)
        last = max(first, len(frames) - 1 - len(frames) // 6)
        start, span = frames[first].t, frames[last].t - frames[first].t + (frames[1].t - frames[0].t)
        sheets["jam"] = strip_sheet("jam", {s: [Jam(s, seed=a.seed, start_s=start, max_s=span)] for s in strengths},
                                    frames, a.seed, src, f"window from t = {start:g} s, strength x {span:g} s long")
        sheets["compass"] = compass_sheet(frames, strengths, a.seed, src)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for name in DEGRADATIONS:                                        # every degradation, or it is a bug
            if not cv2.imwrite(str(out / f"{name}.png"), sheets[name]):
                raise OSError(f"cannot write {out / f'{name}.png'}")
    except (OSError, ValueError) as e:      # missing flight or map, a frame rate the flight cannot give, bad output
        print(f"chaos_sheet: error: {e}", file=sys.stderr)
        return 2
    print(f"chaos_sheet: {len(sheets)} sheets, seed {a.seed}, {src}, in {out}")
    for name in DEGRADATIONS:
        print(f"  {out / f'{name}.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
