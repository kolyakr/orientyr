"""Trajectory image (M1-07): the map with the route, numbered points, footprints, start/finish and a legend.

Input is a run folder: ``run.json`` names the map, ``run.jsonl`` holds one record per frame (frames are not
needed). Output is the full-size image and a preview whose longest side is ``preview_px`` (1600 by default).

What is drawn, bottom to top:

- **footprints** of accepted fixes, at true size and rotation (thin, translucent);
- **the route** through each record's filtered position: solid between fixed points, dashed where a point is
  **missed** (no accepted fix, or interpolated);
- **points**, one per frame, filled in the fixed or the missed colour; START and FINISH on top;
- **labels**: each point numbered by its frame index (the task's ``crop_N`` is point N), placed where they cover
  no route, no marker and no other label; on a dense run every k-th point is labelled;
- **a legend** in the corner the route uses least.

Sizes (lines, markers, fonts) are fixed in *preview* pixels and multiplied by ``unit`` = map size / preview size,
so the preview looks the same for any map and the full image is the same picture, just sharper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dronenav.telemetry import Record, read_run, read_run_info

__all__ = ["FINISH", "FIXED", "FOOTPRINT", "MISSED", "START", "Label", "Layout", "render_trajectory",
           "write_trajectory"]

# Colours, BGR. Chosen to stand out on satellite imagery (greens, greys, browns) and from each other.
FIXED = (255, 170, 0)        # azure: route and points with an accepted fix
MISSED = (0, 140, 255)       # orange: no accepted fix, or interpolated
FOOTPRINT = (235, 235, 235)  # near white, translucent
START = (60, 210, 60)        # green
FINISH = (60, 40, 230)       # red
TEXT = (255, 255, 255)
OUTLINE = (0, 0, 0)

# Sizes in preview pixels (multiplied by unit at full size).
LINE = 3.0
DASH, GAP = 6.0, 8.0         # each dash also gets round caps (LINE / 2 per end): visible gap ≈ 5 px
POINT_R = 4.5
END_R = 7.5
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.42            # ≈ 10 px digits in the preview
MIN_LABEL_SPACING = 22.0     # preview px between labelled points, below which labels are thinned
MARGIN = 12.0                # legend distance from the image edge
FOOTPRINT_ALPHA = 0.55
LEGEND_ALPHA = 0.78

_SHIFT = 4                   # sub-pixel drawing: OpenCV coordinates in 1/16 px


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
    legend: tuple[float, float, float, float]
    unit: float
    label_every: int


def render_trajectory(run_dir: str | Path, *, label_every: int | None = None, preview_px: int = 1600,
                      map_img: np.ndarray | None = None) -> tuple[np.ndarray, Layout]:
    """Draw the run over its map at full size; returns the image (BGR uint8) and the layout.

    label_every: label every k-th point (plus the last); None picks k so labels do not crowd.
    map_img: the map, if already loaded; otherwise it is read from the path in run.json.
    """
    run_dir = Path(run_dir)
    info = read_run_info(run_dir)
    records = read_run(run_dir / "run.jsonl")
    if not records:
        raise ValueError(f"{run_dir / 'run.jsonl'} has no records")
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
    missed = [_is_missed(r) for r in records]

    _draw_footprints(img, records, unit)
    obstacles = _Obstacles(w, h, unit)
    _draw_route(img, obstacles, pts, missed, unit)
    _draw_points(img, obstacles, pts, missed, unit)

    if label_every is None:
        label_every = _auto_label_every(pts, unit)
    legend_box = _place_legend(img, obstacles, records, missed, unit)
    labels = _place_labels(img, obstacles, records, pts, label_every, unit)
    return img, Layout(labels=labels, legend=legend_box, unit=unit, label_every=label_every)


def write_trajectory(run_dir: str | Path, out_dir: str | Path | None = None, *, preview_px: int = 1600,
                     label_every: int | None = None) -> tuple[Path, Path]:
    """Render and write ``trajectory.png`` (full size) and ``trajectory_preview.jpg`` (longest side preview_px,
    never upscaled) into ``out_dir`` (default: the run folder). Returns both paths."""
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    img, _ = render_trajectory(run_dir, label_every=label_every, preview_px=preview_px)
    full, preview = out_dir / "trajectory.png", out_dir / "trajectory_preview.jpg"
    k = preview_px / max(img.shape[:2])
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
    for path, image, params in ((full, img, []), (preview, small, [cv2.IMWRITE_JPEG_QUALITY, 92])):
        if not cv2.imwrite(str(path), image, params):   # OpenCV only logs a warning on failure
            raise OSError(f"cannot write {path}")
    return full, preview


def _bgr(img: np.ndarray) -> np.ndarray:
    """The map as 3-channel BGR uint8: a grey or BGRA map would lose the colours of everything drawn on it."""
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


# --- drawing ---------------------------------------------------------------------------------------------------

def _is_missed(r: Record) -> bool:
    """No accepted fix on this frame, or its position was filled in rather than measured."""
    return r.interpolated or r.track.fix is None or not r.track.fix.accepted


def _px(x: float, y: float) -> tuple[int, int]:
    """Map px (origin at the top-left corner of the top-left pixel) to OpenCV's sub-pixel integer coordinates
    (pixel centres at integers)."""
    return round((x - 0.5) * (1 << _SHIFT)), round((y - 0.5) * (1 << _SHIFT))


def _t(v: float, unit: float) -> int:
    """A size in preview px to a full-size thickness or radius (at least 1)."""
    return max(1, round(v * unit))


def _draw_footprints(img: np.ndarray, records: list[Record], unit: float) -> None:
    layer = img.copy()
    for r in records:
        f = r.track.fix
        if f is not None and f.accepted:
            poly = np.array([_px(x, y) for x, y in f.footprint], np.int32)
            cv2.polylines(layer, [poly], True, FOOTPRINT, _t(1.2, unit), cv2.LINE_AA, _SHIFT)
    cv2.addWeighted(layer, FOOTPRINT_ALPHA, img, 1 - FOOTPRINT_ALPHA, 0, dst=img)


def _draw_route(img, obstacles, pts, missed, unit) -> None:
    thick = _t(LINE, unit)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        obstacles.segment(a, b)
        if missed[i] or missed[i + 1]:
            _dashed(img, a, b, MISSED, thick, DASH * unit, GAP * unit)
        else:
            cv2.line(img, _px(*a), _px(*b), FIXED, thick, cv2.LINE_AA, _SHIFT)


def _dashed(img, a, b, colour, thick, dash, gap) -> None:
    length = float(np.hypot(*(b - a)))
    if length == 0:
        return
    d = (b - a) / length
    s = 0.0
    while s < length:
        e = min(s + dash, length)
        cv2.line(img, _px(*(a + d * s)), _px(*(a + d * e)), colour, thick, cv2.LINE_AA, _SHIFT)
        s = e + gap


def _draw_points(img, obstacles, pts, missed, unit) -> None:
    edge = _t(1.0, unit)
    for i, (x, y) in enumerate(pts):
        end = i == 0 or i == len(pts) - 1
        if end:
            continue
        r = _t(POINT_R, unit)
        cv2.circle(img, _px(x, y), r << _SHIFT, MISSED if missed[i] else FIXED, -1, cv2.LINE_AA, _SHIFT)
        cv2.circle(img, _px(x, y), r << _SHIFT, OUTLINE, edge, cv2.LINE_AA, _SHIFT)
        obstacles.disc(x, y, POINT_R + 1.5)
    for (x, y), colour in ((pts[0], START), (pts[-1], FINISH)):   # on top of everything else on the route
        r = _t(END_R, unit)
        cv2.circle(img, _px(x, y), r << _SHIFT, colour, -1, cv2.LINE_AA, _SHIFT)
        cv2.circle(img, _px(x, y), r << _SHIFT, TEXT, _t(1.6, unit), cv2.LINE_AA, _SHIFT)
        obstacles.disc(x, y, END_R + 2)


# --- labels ----------------------------------------------------------------------------------------------------

def _auto_label_every(pts: np.ndarray, unit: float) -> int:
    """Label every k-th point so labelled points are at least MIN_LABEL_SPACING preview px apart (median)."""
    if len(pts) < 2:
        return 1
    steps = np.hypot(*np.diff(pts, axis=0).T) / unit
    step = float(np.median(steps))
    return 1 if step <= 0 else max(1, math.ceil(MIN_LABEL_SPACING / step))


def _place_labels(img, obstacles, records, pts, every, unit) -> list[Label]:
    n = len(pts)
    wanted = [(None, "START", 0), (None, "FINISH", n - 1)]          # always, and first: they matter most
    wanted += [(r.idx, str(r.idx), i) for i, r in enumerate(records) if i % every == 0 or i == n - 1]
    labels = []
    for idx, text, i in wanted:
        x, y = pts[i]
        radius = END_R if i in (0, n - 1) else POINT_R
        box = _find_label_spot(img, obstacles, text, x, y, radius, unit, force=idx is None)
        if box is None:
            continue
        _draw_text(img, text, box, unit, tag=START if text == "START" else FINISH if text == "FINISH" else None)
        obstacles.box(box)
        labels.append(Label(idx=idx, text=text, box=box))
    return labels


def _text_size(text: str, unit: float) -> tuple[float, float, int, float]:
    """(box width, box height, baseline, padding) of a label at full size, outline included."""
    scale, thick = FONT_SCALE * unit, _t(1.0, unit)
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick + 2 * _t(1.0, unit))
    pad = 1.5 * unit
    return tw + 2 * pad, th + base + 2 * pad, base, pad


def _find_label_spot(img, obstacles, text, x, y, radius, unit, force) -> tuple[float, float, float, float] | None:
    """The first free box around (x, y): right, left, above, below, then the diagonals, then further out.

    ``force`` (START/FINISH) takes the least-covered spot when none is free.
    """
    bw, bh, _, _ = _text_size(text, unit)
    best, best_cover = None, math.inf
    for reach in (1.0, 2.2, 3.6):
        d = (radius + 3.0) * unit * reach
        for dx, dy in ((1, 0), (-1, 0), (0, -1), (0, 1), (1, -1), (-1, -1), (1, 1), (-1, 1)):
            cx = x + dx * (d + bw / 2)
            cy = y + dy * (d + bh / 2)
            box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            if box[0] < 0 or box[1] < 0 or box[2] > img.shape[1] or box[3] > img.shape[0]:
                continue
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

def _place_legend(img, obstacles, records, missed, unit) -> tuple[float, float, float, float]:
    n_missed = sum(missed)
    rows = [("line", FIXED, "route, fixed point"),
            ("dash", MISSED, "missed: no fix / interpolated"),
            ("box", FOOTPRINT, "camera footprint"),
            ("dot", START, "start"),
            ("dot", FINISH, "finish")]
    title = f"{len(records)} frames, {len(records) - n_missed} fixed, {n_missed} missed"
    scale, thick = FONT_SCALE * unit, _t(1.0, unit)
    row_h = 17 * unit
    swatch = 26 * unit
    pad = 8 * unit
    text_w = max(cv2.getTextSize(t, FONT, scale, thick)[0][0] for t in [title] + [r[2] for r in rows])
    bw = pad * 3 + swatch + text_w
    bh = pad * 2 + row_h * (len(rows) + 1)
    h, w = img.shape[:2]
    m = MARGIN * unit
    corners = [(m, m), (w - m - bw, m), (m, h - m - bh), (w - m - bw, h - m - bh)]
    box = min((( x, y, x + bw, y + bh) for x, y in corners), key=lambda b: obstacles.cover(b))
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
