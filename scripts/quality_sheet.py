"""Run the quality gate (M2-03) over a take and draw a contact sheet to check its verdicts by eye.

    python scripts/quality_sheet.py --video data/video/take.mp4 --fps 12 --moire-sigma 1    # a phone take
    python scripts/quality_sheet.py --frames data/hackatons-data/crops                      # a folder of frames
    python scripts/quality_sheet.py --flight data/irpin/flights/mild/2022 --fps 2 --end-s 60
    python scripts/quality_sheet.py --flight data/irpin/flights/mild/2022 --keyframes \\
        --map data/irpin/maps/irpin_2022.png --smear 0.5 --smear-share 0.2                  # the gate against the truth

Writes into --out: ``sheet_01.png``, ``sheet_02.png``, ... (every frame as a thumbnail with its blur score, the
score's share of the running median, and the verdict: a green frame is kept, a red one rejected with its reason;
glare the gate masks is tinted) and ``summary.txt`` (one row per frame, then the counts).

With ``--flight`` and ``--map`` every frame, kept or dropped, is also searched for on the map on its own (a
whole-map search, no track), and its fix is compared with the flight's exact truth: a good gate drops the frames
whose fix would be wrong or missing and keeps the frames that match. ``--smear`` smears a seeded share of the
frames first (the chaos blur), to give the gate known bad frames on a clean flight.
"""
from __future__ import annotations

import argparse
import itertools
import math
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from dronenav.config import add_shared_args, init_cli
from dronenav.io import FolderSource
from dronenav.quality import REASONS, FrameQuality, QualityConfig, QualityGate, glare_mask
from dronenav.sim import Blur, FlightSource
from dronenav.types import Frame, FrameSource
from dronenav.video import VideoSource

TILE_W = 300                    # one thumbnail
COLUMNS = 5
BAR = 22                        # height of a caption line
EDGE = 4                        # the verdict frame around a thumbnail
PAGE, INK, DIM = (30, 30, 30), (235, 235, 235), (150, 150, 150)
KEPT, REJECTED, GLARE = (80, 180, 70), (60, 60, 225), (200, 60, 200)        # BGR


class _Smeared:
    """``source`` with a seeded share of its frames smeared by the chaos blur: known bad frames for the gate.
    ``smeared`` holds the idx of every frame it has smeared."""

    def __init__(self, source: FrameSource, strength: float, share: float, seed: int):
        self.source, self.share, self.seed = source, share, seed
        self.blur, self.smeared = Blur(strength, seed=seed), set()

    def __len__(self) -> int:
        return len(self.source)

    def __iter__(self) -> Iterator[Frame]:
        for n, frame in enumerate(self.source):
            # the first frames stay clean (the gate needs a few scores before it can judge); then one decision
            # per frame from its own idx, so the same frames are smeared whatever the sampling
            if n >= 3 and np.random.default_rng([self.seed, frame.idx]).random() < self.share:
                self.smeared.add(frame.idx)
                frame = Frame(frame.idx, frame.t, self.blur.image(frame.image, frame.idx), frame.source,
                              frame.heading_prior_deg, frame.scale_prior)
            yield frame

    def __getattr__(self, name: str):
        if name.startswith("_") or name == "source":
            raise AttributeError(name)
        return getattr(self.source, name)


def open_source(a: argparse.Namespace) -> tuple[FrameSource, str]:
    """The frames to judge, and a name for the sheet's header."""
    if a.flight:
        src = FlightSource(a.flight, fps=a.fps or 1.0, keyframes=a.keyframes, start_s=a.start_s, end_s=a.end_s)
        return src, f"flight {Path(a.flight).parent.name}/{Path(a.flight).name}"
    if a.video:
        return VideoSource(a.video, every=a.every, fps=a.fps), f"video {Path(a.video).name}"
    return FolderSource(a.frames), f"frames {Path(a.frames).name}"


# --- drawing ---------------------------------------------------------------------------------------------------

def _text(img: np.ndarray, text: str, org: tuple[int, int], colour=INK, scale: float = 0.42) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def tile(frame: Frame, q: FrameQuality, cfg: QualityConfig, note: str = "") -> np.ndarray:
    """The frame as a thumbnail in a green (kept) or red (rejected) frame, under two caption lines."""
    h, w = frame.image.shape[:2]
    size = (TILE_W, max(1, round(h * TILE_W / w)))
    thumb = cv2.resize(frame.image if frame.image.ndim == 3 else cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR),
                       size, interpolation=cv2.INTER_AREA)
    if q.glare > 0 and min(h, w) >= 3:
        # the glare the gate found (masked for the matcher on a kept frame), tinted so its extent can be judged
        glare = cv2.resize(glare_mask(frame.image, level=cfg.glare_level, texture=cfg.glare_texture,
                                      min_area=cfg.glare_min_area, margin=cfg.glare_margin), size,
                           interpolation=cv2.INTER_NEAREST) > 0
        thumb[glare] = (0.5 * thumb[glare] + 0.5 * np.array(GLARE)).astype(np.uint8)
    colour = KEPT if q.kept else REJECTED
    cv2.rectangle(thumb, (EDGE // 2, EDGE // 2), (size[0] - 1 - EDGE // 2, size[1] - 1 - EDGE // 2), colour, EDGE)
    bar = np.full((2 * BAR, TILE_W, 3), PAGE, np.uint8)
    share = f"  {q.score / q.median:.2f} x median" if q.median else ""
    _text(bar, f"#{q.idx}  t {q.t:.2f}s  {'KEPT' if q.kept else q.reason.upper()}  {note}", (4, 15), colour)
    _text(bar, f"score {q.score:.3g}{share}" + (f"  glare {q.glare:.0%}" if q.glare > 0 else ""), (4, 15 + BAR), DIM)
    return np.vstack([bar, thumb])


def page(tiles: list[np.ndarray], title: str) -> np.ndarray:
    """Tiles in rows of COLUMNS under a title line, on a dark page."""
    th = max(t.shape[0] for t in tiles)
    cells = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=PAGE) for t in tiles]
    cells += [np.full((th, TILE_W, 3), PAGE, np.uint8)] * (-len(cells) % COLUMNS)
    gap = 6
    rows = [np.hstack([cv2.copyMakeBorder(c, gap, 0, gap, 0, cv2.BORDER_CONSTANT, value=PAGE) for c in row])
            for row in itertools.batched(cells, COLUMNS)]
    body = np.vstack(rows)
    head = np.full((BAR + 10, body.shape[1], 3), PAGE, np.uint8)
    _text(head, title, (gap, 22), INK, 0.55)
    return cv2.copyMakeBorder(np.vstack([head, body]), gap, 2 * gap, gap, 2 * gap, cv2.BORDER_CONSTANT, value=PAGE)


# --- the gate against the truth --------------------------------------------------------------------------------

class TruthCheck:
    """Finds each frame on the map on its own and compares the fix with the flight's truth."""

    def __init__(self, map_path: str, cache_dir: str, source: FrameSource, tol_px: float):
        from dronenav.matching import RootSiftMatcher  # heavy imports, only when the truth is asked for
        from dronenav.search import Searcher

        map_img = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
        if map_img is None:
            raise FileNotFoundError(f"cannot read the map {map_path}")
        self.searcher = Searcher(RootSiftMatcher(map_img, cache_dir=cache_dir), map_img)
        self.source, self.tol_px = source, tol_px

    def fix_of(self, frame: Frame) -> tuple[str, float | None]:
        """("right" | "wrong" | "missing", error in map px): an accepted fix within the tolerance of the true
        frame centre, an accepted fix further away, or no accepted fix."""
        self.searcher.reset()           # no track: each frame's fix says what that frame alone is worth
        fix = self.searcher.locate(frame.idx, frame.image).fix
        if fix is None or not fix.accepted:
            return "missing", None
        cx, cy = self.source.truth(frame.idx).center
        err = math.hypot(fix.x - cx, fix.y - cy)
        return ("right" if err <= self.tol_px else "wrong"), err


def truth_table(rows: list[dict], tol_px: float, smeared: set[int] | None) -> list[str]:
    """The gate's two piles against what the matcher makes of each frame."""
    out = ["", (f"gate against the truth (a fix is right when accepted and within {tol_px:g} map px of the true "
           "frame centre; every frame searched on the whole map, on its own)"),
           f"{'':8}{'frames':>8}{'fix right':>11}{'wrong or missing':>18}{'wrong':>8}{'missing':>9}"]
    for name, kept in (("kept", True), ("dropped", False)):
        pile = Counter(r["fix"] for r in rows if r["q"].kept == kept)
        out.append(f"{name:8}{sum(pile.values()):>8}{pile['right']:>11}{pile['wrong'] + pile['missing']:>18}"
                   f"{pile['wrong']:>8}{pile['missing']:>9}")
    if smeared is not None:
        hit = sum(not r["q"].kept for r in rows if r["q"].idx in smeared)
        clean = [r for r in rows if r["q"].idx not in smeared]
        out.append(f"smeared on purpose: {len(smeared)} frames, {hit} of them dropped; "
                   f"untouched: {len(clean)} frames, {sum(not r['q'].kept for r in clean)} of them dropped")
    return out


# --- main ------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", default=None, help="folder of frames")
    ap.add_argument("--video", default=None, help="video file (a phone take)")
    ap.add_argument("--flight", default=None, help="flight folder (meta.json, truth.jsonl, flight.mp4 or frames/)")
    ap.add_argument("--every", type=int, default=None, help="with --video: take every N-th frame")
    ap.add_argument("--fps", type=float, default=None, help="with --video or --flight: frames per second to take "
                    "(flight default 1)")
    ap.add_argument("--keyframes", action="store_true", help="with --flight: read frames/ instead of the video")
    ap.add_argument("--start-s", type=float, default=0.0, help="with --flight: where to start, s")
    ap.add_argument("--end-s", type=float, default=None, help="with --flight: where to stop, s")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    ap.add_argument("--map", default=None, help="with --flight: the flight's map; compares the gate with the truth")
    ap.add_argument("--tol", type=float, default=25.0, help="with --map: a fix this close to the truth is right, "
                    "map px (default 25)")
    ap.add_argument("--cache", default="outputs/cache", help="with --map: folder for the map features")
    ap.add_argument("--smear", type=float, default=None, help="smear some frames with the chaos blur of this "
                    "strength, 0..1 (to test the gate on a clean take)")
    ap.add_argument("--smear-share", type=float, default=0.2, help="with --smear: the share of frames smeared")
    ap.add_argument("--seed", type=int, default=0, help="with --smear: which frames, and how")
    ap.add_argument("--window", type=int, default=QualityConfig.window, help="blur rule: frames in the median")
    ap.add_argument("--ratio", type=float, default=QualityConfig.blur_ratio, help="blur rule: reject under this "
                    "share of the median score")
    ap.add_argument("--max-glare", type=float, default=QualityConfig.max_glare, help="reject above this share of "
                    "glare")
    ap.add_argument("--moire-sigma", type=float, default=0.0, help="damp the screen grid on kept frames, px")
    ap.add_argument("--notch", action="store_true", help="also notch the grid's peaks in the spectrum")
    ap.add_argument("--per-page", type=int, default=40, help="thumbnails per sheet (default 40)")
    ap.add_argument("--out", default="outputs/quality_sheets", help="folder for the sheets and summary.txt")
    add_shared_args(ap)
    a = ap.parse_args(argv)
    init_cli(a)
    if sum(v is not None for v in (a.frames, a.video, a.flight)) != 1:
        ap.error("give one of --frames, --video, --flight")
    if a.map and not a.flight:
        ap.error("--map needs --flight: only a flight has the truth to compare with")
    if a.per_page < 1:
        ap.error("--per-page must be at least 1")
    if a.max_frames is not None and a.max_frames < 1:
        ap.error("--max-frames must be at least 1")
    if a.smear is not None and not (0 <= a.smear <= 1 and 0 <= a.smear_share <= 1 and a.seed >= 0):
        ap.error("--smear and --smear-share must be within [0, 1], --seed >= 0")
    try:
        cfg = QualityConfig(window=a.window, blur_ratio=a.ratio, min_history=min(QualityConfig.min_history, a.window),
                            max_glare=a.max_glare, moire_sigma=a.moire_sigma, moire_notch=a.notch)
        source, name = open_source(a)
        smeared = None
        if a.smear is not None:
            source = _Smeared(source, a.smear, a.smear_share, a.seed)
            smeared, name = source.smeared, f"{name}, {a.smear_share:.0%} smeared at {a.smear:g} (seed {a.seed})"
        check = TruthCheck(a.map, a.cache, source, a.tol) if a.map else None
        gate = QualityGate(source, cfg)
        rows, tiles = [], []
        for frame, q in itertools.islice(gate.judged(), a.max_frames):
            fix, err = check.fix_of(frame) if check else (None, None)
            note = "" if fix is None else f"fix {fix}" + (f" {err:.0f}px" if err is not None else "")
            if smeared is not None and q.idx in smeared:
                note = f"SMEARED  {note}"
            rows.append({"q": q, "fix": fix, "err": err})
            tiles.append(tile(frame, q, cfg, note))
        if not rows:
            raise ValueError(f"the {name} gave no frames")
        verdicts = [r["q"] for r in rows]
        kept = sum(q.kept for q in verdicts)
        reasons = Counter(q.reason for q in verdicts if not q.kept)
        counts = [f"kept {kept} of {len(verdicts)}", *(f"{r} {reasons[r]}" for r in REASONS if reasons[r])]

        lines = [f"quality gate over the {name}",
                 (f"window {cfg.window}, ratio {cfg.blur_ratio:g}, max glare {cfg.max_glare:g}, "
                 f"moire sigma {cfg.moire_sigma:g}{', notch' if cfg.moire_notch else ''}"), "",
                 f"{'idx':>7}{'t, s':>9}{'score':>11}{'median':>11}{'x median':>10}{'glare':>7}"
                 + (f"{'fix':>9}{'err px':>8}" if check else "") + ("  smeared" if smeared is not None else "")
                 + "  verdict"]
        for r in rows:
            q = r["q"]
            median = f"{q.median:.4g}" if q.median is not None else "-"
            share = f"{q.score / q.median:.2f}" if q.median else "-"
            fix = f"{r['fix']:>9}{'-' if r['err'] is None else format(r['err'], '.1f'):>8}" if check else ""
            mark = f"{'yes' if q.idx in smeared else 'no':>9}" if smeared is not None else ""
            lines.append(f"{q.idx:>7}{q.t:>9.2f}{q.score:>11.4g}{median:>11}{share:>10}{q.glare:>7.0%}{fix}{mark}"
                         f"  {'kept' if q.kept else q.reason}")
        lines += ["", *counts]
        table = truth_table(rows, a.tol, smeared) if check else []
        lines += table

        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        pages = list(itertools.batched(tiles, a.per_page))
        paths = []
        for n, batch in enumerate(pages, 1):
            paths.append(out / f"sheet_{n:02d}.png")
            title = f"{name}   {', '.join(counts)}   page {n} of {len(pages)}"
            if not cv2.imwrite(str(paths[-1]), page(list(batch), title)):
                raise OSError(f"cannot write {paths[-1]}")
        (out / "summary.txt").write_text("\n".join(lines) + "\n")
    except (OSError, ValueError) as e:      # a missing folder, video, flight or map; settings out of range
        print(f"quality_sheet: error: {e}", file=sys.stderr)
        return 2
    print(f"quality_sheet: {name}: {', '.join(counts)}")
    for ln in table:
        print(ln)
    for p in (*paths, out / "summary.txt"):
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
