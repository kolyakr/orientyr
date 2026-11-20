"""Trajectory image (M1-07): the map with the route, numbered points, footprints, start/finish and a legend.

Input is a run folder: ``run.json`` names the map, ``run.jsonl`` holds one record per frame (frames are not
needed). Output is the full-size image and a preview whose longest side is ``preview_px`` (1600 by default).

What is drawn, bottom to top:

- **footprints** at true size and rotation (thin, translucent), for numbered points with an accepted, measured
  fix: on a sparse run that is every such frame, on a dense run one per number (all would merge into a band);
- **the route** through each record's filtered position: solid between fixed points, dashed where a point is
  **missed** (no accepted fix, or interpolated);
- **points**, filled in the fixed or the missed colour: a full marker for numbered points, a small dot for the
  others, at most one per DOT_SPACING (on a dense run more would bury the route); START and FINISH on top;
- **labels**: points numbered by their frame index (the task's ``crop_N`` is point N), placed where they cover no
  route, no marker and no other label. A point is numbered when it is at least a label's width (the run's widest
  number, MIN_LABEL_SPACING at least) from every numbered point, so hovers, loops, zigzags and long runs stay
  readable. Points off the map are not numbered; the legend counts them. START and FINISH always show, pinned to
  the map's edge if they are off the map;
- **a legend** with a summary line, in a corner (else the middle of an edge) where it hides no point and covers
  the least route; left out when the map is too small for it or every such spot would hide a point.

Sizes (lines, markers, fonts) are fixed in *preview* pixels and multiplied by ``unit`` = map size / preview size,
so the preview looks the same for any map and the full image is the same picture, just sharper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

from dronenav.telemetry import Record, RunInfo, read_run, read_run_info
from dronenav.viz.draw import FINISH, FIXED, FONT, FOOTPRINT, MISSED, OUTLINE, START, TEXT
from dronenav.viz.draw import SHIFT as _SHIFT
from dronenav.viz.draw import dashed as _dashed
from dronenav.viz.draw import is_missed as _is_missed
from dronenav.viz.draw import px as _px

__all__ = ["FINISH", "FIXED", "FOOTPRINT", "MISSED", "START", "Label", "Layout", "Written", "render_records",
           "render_trajectory", "write_trajectory"]

# Colours (FIXED, MISSED, FOOTPRINT, START, FINISH, TEXT, OUTLINE) come from dronenav.viz.draw, shared with the
# video renderer.

# Sizes in preview pixels (multiplied by unit at full size).
LINE = 3.0
DASH, GAP = 6.0, 8.0         # each dash also gets round caps (LINE / 2 per end): visible gap ≈ 5 px
POINT_R = 4.5
DOT_R = 1.6                  # unnumbered points: a dot within the route line's width
DOT_SPACING = 8.0            # preview px between drawn points: denser dots would bury the dashes
END_R = 7.5
FONT_SCALE = 0.42            # ≈ 10 px digits in the preview
MIN_LABEL_SPACING = 18.0     # preview px between numbered points, at least (a 2-digit label is ≈ 18 px wide)
MARGIN = 12.0                # legend distance from the image edge
FOOTPRINT_ALPHA = 0.55
LEGEND_ALPHA = 0.78


@dataclass(frozen=True)
class Label:
    """A text placed on the image. idx: the frame it numbers, or None for START/FINISH. box: x0, y0, x1, y1 px."""

    idx: int | None
    text: str
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Layout:
    """Where things went, in full-image pixels; returned so placement can be checked and reused."""

    labels: list[Label]
    legend: tuple[float, float, float, float] | None   # None: the map is too small for one
    unit: float
    label_every: int | None                             # None: numbered by spacing
    unlabelled: list[int]                               # frames meant to be numbered that found no free spot
    off_map: list[int]                                  # frames whose position is outside the map (not numbered)
    footprints: list[int]                               # frames whose footprint was drawn
    summary: str                                        # the legend's first line, e.g. "50 frames, 44 fixed, 6 missed"


class Written(NamedTuple):
    """What write_trajectory wrote, and the layout of the picture."""

    full: Path
    preview: Path
    layout: Layout


def render_records(records: list[Record], info: RunInfo, map_img: np.ndarray | None = None, *,
                   label_every: int | None = None, preview_px: int = 1600) -> tuple[np.ndarray, Layout]:
    """Draw records over their map at full size; returns the image (BGR uint8) and the layout.

    The core of the renderer, for callers that already hold the run in memory (scripts/run.py).
    map_img: the map, if already loaded; otherwise it is read from ``info.map_path``.
    label_every: number every k-th frame (frame numbers divisible by k, plus the last point); None numbers points
    spaced at least a label apart.
    """
    if label_every is not None and label_every < 1:
        raise ValueError(f"label_every must be at least 1, got {label_every}")
    if preview_px < 1:
        raise ValueError(f"preview_px must be at least 1, got {preview_px}")
    if not records:
        raise ValueError("the run has no records")
    if map_img is None:
        map_img = cv2.imread(str(info.map_path), cv2.IMREAD_COLOR)
        if map_img is None:
            raise FileNotFoundError(f"cannot read the map {info.map_path}")
    map_img = _bgr(map_img)
    h, w = map_img.shape[:2]
    if (w, h) != info.map_size:
        # positions are map px of the map the run was flown over: on another size they land in the wrong place
        raise ValueError(f"the map is {w} x {h} px but run.json says the run was flown over "
                         f"{info.map_size[0]} x {info.map_size[1]} px ({info.map_path})")
    unit = max(1.0, max(w, h) / preview_px)
    img = map_img.copy()

    pts = np.array([[r.track.x, r.track.y] for r in records])
    idxs = np.array([r.idx for r in records])
    missed = [_is_missed(r) for r in records]
    on_map = (pts[:, 0] >= 0) & (pts[:, 0] <= w) & (pts[:, 1] >= 0) & (pts[:, 1] <= h)
    # numbered points at least one label apart: the widest number of the run sets the spacing (5 digits ≈ 45 px)
    widest = _text_size(str(int(idxs.max())), unit)[0]
    numbered = _numbered(pts, idxs, on_map, label_every, max(MIN_LABEL_SPACING * unit, widest))
    ends = [_clamp_end(*pts[0], w, h, unit), _clamp_end(*pts[-1], w, h, unit)]

    footprints = _draw_footprints(img, records, set(numbered), unit)
    obstacles = _Obstacles(w, h, unit)
    _draw_route(img, obstacles, pts, missed, unit)
    _draw_points(img, obstacles, pts, missed, set(numbered), ends, unit)

    n_missed, n_off = sum(missed), int((~on_map).sum())
    summary = f"{len(records)} frames, {len(records) - n_missed} fixed, {n_missed} missed"
    if n_off:
        summary += f" ({n_off} off the map)"      # off the map = dead-reckoned, so already counted as missed
    markers = [tuple(p) for p in pts[on_map]] + ends
    legend_box = _place_legend(img, obstacles, summary, markers, unit)
    labels, unlabelled = _place_labels(img, obstacles, records, pts, numbered, ends, unit)
    return img, Layout(labels=labels, legend=legend_box, unit=unit, label_every=label_every, unlabelled=unlabelled,
                       off_map=[int(i) for i in idxs[~on_map]], footprints=footprints, summary=summary)


def render_trajectory(run_dir: str | Path, *, label_every: int | None = None, preview_px: int = 1600,
                      map_img: np.ndarray | None = None) -> tuple[np.ndarray, Layout]:
    """Draw the run in ``run_dir`` (run.json + run.jsonl) over its map; see :func:`render_records`."""
    run_dir = Path(run_dir)
    info = read_run_info(run_dir)
    records = read_run(run_dir / "run.jsonl")
    if not records:
        raise ValueError(f"{run_dir / 'run.jsonl'} has no records")
    return render_records(records, info, map_img, label_every=label_every, preview_px=preview_px)


def write_trajectory(run_dir: str | Path, out_dir: str | Path | None = None, *, map_img: np.ndarray | None = None,
                     preview_px: int = 1600, label_every: int | None = None) -> Written:
    """Render and write ``trajectory.png`` (full size) and ``trajectory_preview.jpg`` (longest side preview_px,
    never upscaled) into ``out_dir`` (default: the run folder). Returns both paths and the layout."""
    run_dir = Path(run_dir)
    img, layout = render_trajectory(run_dir, label_every=label_every, preview_px=preview_px, map_img=map_img)
    out_dir = Path(out_dir) if out_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)          # only after rendering: a typo must not leave a folder
    full, preview = out_dir / "trajectory.png", out_dir / "trajectory_preview.jpg"
    h, w = img.shape[:2]
    k = preview_px / max(w, h)
    size = (preview_px, max(1, round(h * k))) if w >= h else (max(1, round(w * k)), preview_px)
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA) if k < 1 else img
    for path, image, params in ((full, img, []), (preview, small, [cv2.IMWRITE_JPEG_QUALITY, 92])):
        if not cv2.imwrite(str(path), image, params):   # OpenCV only logs a warning on failure
            raise OSError(f"cannot write {path}")
    return Written(full, preview, layout)


def _bgr(img: np.ndarray) -> np.ndarray:
    """The map as 3-channel BGR uint8: a grey or BGRA map would lose the colours of everything drawn on it."""
    if img.dtype == np.uint16:                          # a 16-bit PNG loaded unchanged
        img = (img // 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        raise ValueError(f"the map must be uint8 or uint16, got {img.dtype}")
    if img.ndim == 3 and img.shape[2] in (1, 2):         # grey, or grey + alpha
        img = img[..., 0]
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


# --- drawing ---------------------------------------------------------------------------------------------------

def _t(v: float, unit: float) -> int:
    """A size in preview px to a full-size thickness or radius (at least 1)."""
    return max(1, round(v * unit))


def _draw_footprints(img: np.ndarray, records: list[Record], numbered: set[int], unit: float) -> list[int]:
    """Footprints of numbered points with an accepted, measured fix. Returns the frames drawn."""
    layer = img.copy()
    drawn = []
    for i, r in enumerate(records):
        f = r.track.fix
        if f is None or not f.accepted or r.interpolated:       # interpolated: not measured, no footprint
            continue
        if i not in numbered:
            continue
        poly = np.array([_px(x, y) for x, y in f.footprint], np.int32)
        cv2.polylines(layer, [poly], True, FOOTPRINT, _t(1.2, unit), cv2.LINE_AA, _SHIFT)
        drawn.append(r.idx)
    cv2.addWeighted(layer, FOOTPRINT_ALPHA, img, 1 - FOOTPRINT_ALPHA, 0, dst=img)
    return drawn


def _draw_route(img, obstacles, pts, missed, unit) -> None:
    thick = _t(LINE, unit)
    phase = 0.0          # the dash pattern runs on across consecutive dashed segments: short steps stay dashed
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        obstacles.segment(a, b)
        if missed[i] or missed[i + 1]:
            phase = _dashed(img, a, b, MISSED, thick, DASH * unit, GAP * unit, phase)
        else:
            cv2.line(img, _px(*a), _px(*b), FIXED, thick, cv2.LINE_AA, _SHIFT)
            phase = 0.0


def _draw_points(img, obstacles, pts, missed, numbered: set[int], ends, unit) -> None:
    edge = _t(1.0, unit)
    last = pts[0]                                       # the last point drawn, to keep dots apart
    for i, (x, y) in enumerate(pts):
        if i == 0 or i == len(pts) - 1:
            continue
        colour = MISSED if missed[i] else FIXED
        if i not in numbered and np.hypot(x - last[0], y - last[1]) < DOT_SPACING * unit:
            continue                                    # too close to the last point drawn: the route shows it
        last = (x, y)
        if i in numbered:
            r = _t(POINT_R, unit)
            cv2.circle(img, _px(x, y), r << _SHIFT, colour, -1, cv2.LINE_AA, _SHIFT)
            cv2.circle(img, _px(x, y), r << _SHIFT, OUTLINE, edge, cv2.LINE_AA, _SHIFT)
            obstacles.disc(x, y, POINT_R + 1.5)
        else:
            cv2.circle(img, _px(x, y), _t(DOT_R, unit) << _SHIFT, colour, -1, cv2.LINE_AA, _SHIFT)
    for (x, y), colour in zip(ends, (START, FINISH), strict=True):   # on top of everything else on the route
        r = _t(END_R, unit)
        cv2.circle(img, _px(x, y), r << _SHIFT, colour, -1, cv2.LINE_AA, _SHIFT)
        cv2.circle(img, _px(x, y), r << _SHIFT, TEXT, _t(1.6, unit), cv2.LINE_AA, _SHIFT)
        obstacles.disc(x, y, END_R + 2)


# --- labels ----------------------------------------------------------------------------------------------------

def _clamp_end(x: float, y: float, w: int, h: int, unit: float) -> tuple[float, float]:
    """An end point off the map, pinned just inside its edge so START/FINISH still show; one on the map stays."""
    if 0 <= x <= w and 0 <= y <= h:
        return x, y
    r = (END_R + 2) * unit
    return min(max(x, r), w - r), min(max(y, r), h - r)


def _numbered(pts: np.ndarray, idxs: np.ndarray, on_map: np.ndarray, every: int | None, gap: float) -> list[int]:
    """Which points get their frame number (only points on the map can be numbered).

    With ``every``: frames whose number is divisible by k (point N is frame N, also when frames are skipped),
    plus the last point. Otherwise by spacing: a point is numbered when it is at least
    ``gap`` full-size px from every numbered point, so hovers, loops and zigzags do not pile up numbers.
    The first and last points on the map are tried first.
    """
    n = len(pts)
    idx = np.flatnonzero(on_map)
    if every is not None:
        return [int(i) for i in idx if idxs[i] % every == 0 or i == n - 1]
    if len(idx) == 0:
        return []
    order = [idx[0], idx[-1]] + list(idx[1:-1]) if len(idx) > 1 else [idx[0]]
    chosen: list[int] = []
    for i in order:
        if not chosen or np.hypot(*(pts[chosen] - pts[i]).T).min() >= gap:
            chosen.append(int(i))
    return sorted(chosen)


def _place_labels(img, obstacles, records, pts, numbered, ends, unit) -> tuple[list[Label], list[int]]:
    n = len(pts)
    end_at = {0: ends[0], n - 1: ends[1]}
    wanted = [(None, "START", 0), (None, "FINISH", n - 1)]          # always, and first: they matter most
    wanted += [(records[i].idx, str(records[i].idx), i) for i in numbered]
    labels, unlabelled = [], []
    for idx, text, i in wanted:
        x, y = end_at.get(i, pts[i])
        radius = END_R if i in end_at else POINT_R
        box = _find_label_spot(img, obstacles, text, x, y, radius, unit, force=idx is None)
        if box is None:
            unlabelled.append(idx)
            continue
        _draw_text(img, text, box, unit, tag=START if text == "START" else FINISH if text == "FINISH" else None)
        obstacles.box(box)
        labels.append(Label(idx=idx, text=text, box=box))
    return labels, unlabelled


def _text_size(text: str, unit: float) -> tuple[float, float, int, float]:
    """(box width, box height, baseline, padding) of a label at full size, outline included."""
    scale, thick = FONT_SCALE * unit, _t(1.0, unit)
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick + 2 * _t(1.0, unit))
    pad = 1.5 * unit
    return tw + 2 * pad, th + base + 2 * pad, base, pad


def _find_label_spot(img, obstacles, text, x, y, radius, unit, force) -> tuple[float, float, float, float] | None:
    """The first free box around (x, y): right, left, above, below, then the diagonals, then further out.

    ``force`` (START/FINISH) takes the least-covered spot when none is free, moved inside the image if needed.
    """
    bw, bh, _, _ = _text_size(text, unit)
    h, w = img.shape[:2]
    best, best_cover = None, math.inf
    for reach in (1.0, 2.2, 3.6):
        d = (radius + 3.0) * unit * reach
        for dx, dy in ((1, 0), (-1, 0), (0, -1), (0, 1), (1, -1), (-1, -1), (1, 1), (-1, 1)):
            cx = x + dx * (d + bw / 2)
            cy = y + dy * (d + bh / 2)
            box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            if box[0] < 0 or box[1] < 0 or box[2] > w or box[3] > h:
                if not force:
                    continue
                dx0 = min(max(box[0], 0), max(w - bw, 0)) - box[0]
                dy0 = min(max(box[1], 0), max(h - bh, 0)) - box[1]
                box = (box[0] + dx0, box[1] + dy0, box[2] + dx0, box[3] + dy0)
            cover = obstacles.cover(box)
            if cover == 0:
                return box
            if cover < best_cover:
                best, best_cover = box, cover
    return best if force else None


def _draw_text(img, text, box, unit, tag: tuple | None = None) -> None:
    """White text with a dark outline; with ``tag``, on a filled box of that colour (START, FINISH): coloured
    text on dark imagery is hard to read, a coloured tag is not."""
    scale, thick = FONT_SCALE * unit, _t(1.0, unit)
    _, _, base, pad = _text_size(text, unit)
    org = (round(box[0] + pad), round(box[3] - pad - base))
    if tag is not None:
        p0, p1 = (round(box[0]), round(box[1])), (round(box[2]), round(box[3]))
        cv2.rectangle(img, p0, p1, tag, -1, cv2.LINE_AA)
        cv2.rectangle(img, p0, p1, TEXT, _t(1.0, unit), cv2.LINE_AA)
    else:
        cv2.putText(img, text, org, FONT, scale, OUTLINE, thick + 2 * _t(1.0, unit), cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, TEXT, thick, cv2.LINE_AA)


# --- legend ----------------------------------------------------------------------------------------------------

def _place_legend(img, obstacles, title, markers, unit) -> tuple[float, float, float, float] | None:
    """In a corner, else the middle of an edge, where it hides no point marker and covers the least route; None if
    the map is too small for it or every such spot would hide a point."""
    rows = [("line", FIXED, "route, fixed point"),
            ("dash", MISSED, "missed or interpolated"),
            ("box", FOOTPRINT, "camera footprint"),
            ("dot", START, "start"),
            ("dot", FINISH, "finish")]
    scale, thick = FONT_SCALE * unit, _t(1.0, unit)
    row_h = 17 * unit
    swatch = 26 * unit
    pad = 8 * unit
    text_w = max(cv2.getTextSize(t, FONT, scale, thick)[0][0] for t in [title] + [r[2] for r in rows])
    bw = pad * 3 + swatch + text_w
    bh = pad * 2 + row_h * (len(rows) + 1)
    h, w = img.shape[:2]
    m = MARGIN * unit
    if bw + 2 * m > 0.5 * w or bh + 2 * m > 0.5 * h:      # it would hide too much of a small map: leave it out
        return None
    reach = END_R * unit                                   # no point marker under the legend, even partly

    def hides_a_marker(b):
        return any(b[0] - reach <= x <= b[2] + reach and b[1] - reach <= y <= b[3] + reach for x, y in markers)

    corners = [(m, m), (w - m - bw, m), (m, h - m - bh), (w - m - bw, h - m - bh)]
    edges = [((w - bw) / 2, m), ((w - bw) / 2, h - m - bh), (m, (h - bh) / 2), (w - m - bw, (h - bh) / 2)]
    for spots in (corners, edges):                       # a corner if any is free, else the middle of an edge
        free = [b for b in ((x, y, x + bw, y + bh) for x, y in spots) if not hides_a_marker(b)]
        if free:
            break
    else:
        return None
    box = min(free, key=obstacles.cover)
    x0, y0, x1, y1 = box

    layer = img.copy()
    cv2.rectangle(layer, (round(x0), round(y0)), (round(x1), round(y1)), (20, 20, 20), -1)
    cv2.addWeighted(layer, LEGEND_ALPHA, img, 1 - LEGEND_ALPHA, 0, dst=img)
    cv2.rectangle(img, (round(x0), round(y0)), (round(x1), round(y1)), (200, 200, 200), _t(1.0, unit))

    tx = x0 + pad * 2 + swatch
    cy = y0 + pad + row_h * 0.5
    cv2.putText(img, title, (round(x0 + pad), round(cy + 4 * unit)), FONT, scale, TEXT, thick, cv2.LINE_AA)
    for kind, colour, text in rows:
        cy += row_h
        sx0, sx1 = x0 + pad, x0 + pad + swatch
        if kind == "line":
            cv2.line(img, _px(sx0, cy), _px(sx1, cy), colour, _t(LINE, unit), cv2.LINE_AA, _SHIFT)
            cv2.circle(img, _px((sx0 + sx1) / 2, cy), _t(POINT_R, unit) << _SHIFT, colour, -1, cv2.LINE_AA, _SHIFT)
        elif kind == "dash":
            _dashed(img, np.array([sx0, cy]), np.array([sx1, cy]), colour, _t(LINE, unit), DASH * unit, GAP * unit)
            cv2.circle(img, _px((sx0 + sx1) / 2, cy), _t(POINT_R, unit) << _SHIFT, colour, -1, cv2.LINE_AA, _SHIFT)
        elif kind == "box":
            half = 6 * unit
            poly = np.array([_px(sx0 + 4 * unit, cy - half), _px(sx1 - 4 * unit, cy - half),
                             _px(sx1 - 4 * unit, cy + half), _px(sx0 + 4 * unit, cy + half)], np.int32)
            cv2.polylines(img, [poly], True, colour, _t(1.2, unit), cv2.LINE_AA, _SHIFT)
        else:
            cv2.circle(img, _px((sx0 + sx1) / 2, cy), _t(END_R, unit) << _SHIFT, colour, -1, cv2.LINE_AA, _SHIFT)
        cv2.putText(img, text, (round(tx), round(cy + 4 * unit)), FONT, scale, TEXT, thick, cv2.LINE_AA)
    obstacles.box(box)
    return box


# --- what labels and the legend must not cover -----------------------------------------------------------------

class _Obstacles:
    """A mask, at preview resolution, of what is already drawn and must stay visible."""

    def __init__(self, w: int, h: int, unit: float):
        self.unit = unit
        self.mask = np.zeros((math.ceil(h / unit) + 1, math.ceil(w / unit) + 1), np.uint8)

    def _p(self, x, y):
        return round(x / self.unit), round(y / self.unit)

    def segment(self, a, b) -> None:
        cv2.line(self.mask, self._p(*a), self._p(*b), 1, round(LINE + 4))

    def disc(self, x, y, r) -> None:
        cv2.circle(self.mask, self._p(x, y), math.ceil(r), 1, -1)

    def box(self, b) -> None:
        x0, y0 = self._p(b[0], b[1])
        x1, y1 = self._p(b[2], b[3])
        self.mask[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = 1

    def cover(self, b) -> int:
        """How many masked preview pixels the box b (full-size px) would cover."""
        x0, y0 = self._p(b[0], b[1])
        x1, y1 = self._p(b[2], b[3])
        return int(self.mask[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1].sum())
