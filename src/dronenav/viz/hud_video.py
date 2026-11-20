"""HUD video (M7-02): the trajectory video as a "mission control" screen that reads without narration.

It is the M1 video (:mod:`dronenav.viz.trajectory_video`: the same timing, the same smooth path through every
record) with a closer, eased camera and a layer that says what the tracker knows at each moment:

- **trail coloured by confidence**: green where the fix was sure, yellow where it was weak
  (:func:`dronenav.confidence.fix_confidence`), red and dashed where there was no accepted fix and the position is
  dead-reckoned;
- **footprint** of the accepted fix (light outline); a **rejected fix** is drawn apart, red and dashed, with its
  reason; the **3-sigma ellipse** of the position covariance, which visibly grows through a jam;
- **camera inset**: the frame the drone saw, linked to the map. A run folder holds no match points today (run.jsonl
  keeps the homography and its footprint, not the correspondences), so the link is the honest one: inset corners to
  footprint corners. Real match lines are drawn only for a record that carries them, as
  ``extra["matches"] = {"frame": [[x, y], ...], "map": [[x, y], ...]}`` (frame px, map px). A LOST frame's picture
  is dimmed and stamped NO FIX; with no picture at all the inset says NO SIGNAL;
- **state badge** (TRACKING green, LOST red, RELOCALIZING amber), the numbers (frame, time, position, heading,
  inliers, 3-sigma, confidence), a border in the state colour while not tracking, and a **timeline** of the states;
- **event captions** ("frame 133: LOST, camera jammed"), from each state change and each ``extra["event"]``. A
  caption stays ``CAPTION_S`` video seconds and is never replaced sooner than ``CAPTION_MIN_S``, so a quick
  LOST -> RELOCALIZING -> TRACKING at 8x speed can still be read;
- **eased camera**: the camera path is the drone's path smoothed with a ``CAMERA_EASE_S`` wide bell-shaped window.
  It is symmetric, so there is no lag in steady flight, and a jump of the position (a relocalisation) becomes a
  glide of about that long, never a cut.

Everything scales with the video size (unit: 1/1080 of the height), so text stays readable at any size. Drawing is
deterministic: the same run gives the same frames.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dronenav.confidence import fix_confidence
from dronenav.telemetry import Record, RunInfo, read_run, read_run_info
from dronenav.viz import trajectory_video as tv
from dronenav.viz.draw import SHIFT, dashed, is_missed, px
from dronenav.viz.trajectory_image import _bgr

__all__ = ["AMBER", "CAPTION_S", "CONF_HIGH", "CONF_LOW", "GREEN", "RED", "STATE_COLOURS", "YELLOW", "Caption",
           "HudInfo", "HudPlan", "HudRenderer", "caption_at", "confidence_colour", "ease_camera", "plan_hud_video",
           "sigma_ellipse", "write_hud_video"]

# Colours, BGR: the traffic-light reading everyone already knows.
GREEN = (90, 220, 70)
YELLOW = (30, 225, 255)
AMBER = (0, 170, 255)
RED = (60, 50, 240)
WHITE = (245, 245, 245)
LABEL = (175, 170, 165)     # field names: quieter than the values
INK = (18, 16, 14)          # panels, outlines, text on a coloured badge
STATE_COLOURS = {"TRACKING": GREEN, "LOST": RED, "RELOCALIZING": AMBER}

CONF_LOW, CONF_HIGH = 0.35, 0.65    # confidence at and below which the trail is yellow / at and above which green
SIGMAS = 3.0                # the ellipse holds the position with ~99 % probability (2D Gaussian: 98.9 %)
VIEW_FOOTPRINTS = 3.5       # the view spans this many median footprint diagonals (M1 video: 6)
CAMERA_EASE_S = 1.6         # width of the camera's smoothing window, video seconds
STAGE = (0.03, 0.12)        # the drone stays within this share of the view (x, y) from its centre: the panels
                            # cover the sides, so near a map edge the camera shows what is past the map instead
CAPTION_S = 2.5             # a caption stays this long, video seconds
CAPTION_MIN_S = 1.2         # ... and at least this long before the next one replaces it
CAPTION_FADE_S = 0.4        # the end of CAPTION_S is a fade-out
MAP_DIM = 0.72              # the map is darkened so the overlay stands out
BACKGROUND = (12, 12, 12)   # outside the map
MAX_MATCH_LINES = 40

FONT = cv2.FONT_HERSHEY_SIMPLEX
BOLD = cv2.FONT_HERSHEY_DUPLEX
_STATE_WORDS = {"TRACKING": "TRACKING", "LOST": "LOST", "RELOCALIZING": "RELOCALIZING"}


# --- pure helpers ----------------------------------------------------------------------------------------------

def confidence_colour(conf: float | None) -> tuple[int, int, int]:
    """Trail colour (BGR) of a position: red without an accepted fix (``None``), else from yellow at ``CONF_LOW``
    and below to green at ``CONF_HIGH`` and above."""
    if conf is None:
        return RED
    u = min(1.0, max(0.0, (float(conf) - CONF_LOW) / (CONF_HIGH - CONF_LOW)))
    return tuple(round(a + u * (b - a)) for a, b in zip(YELLOW, GREEN, strict=True))


def sigma_ellipse(cov, k: float = SIGMAS) -> tuple[float, float, float]:
    """The k-sigma ellipse of a 2 x 2 covariance: (semi-major, semi-minor, angle). The axes are in the covariance's
    length unit (map px); the angle is the major axis's, in degrees in [0, 180), from +x towards +y: clockwise on
    the screen, as cv2.ellipse takes it."""
    c = np.asarray(cov, float)
    vals, vecs = np.linalg.eigh((c + c.T) / 2)          # ascending; symmetrised against rounding
    vals = np.clip(vals, 0.0, None)                      # a hair below zero from rounding is zero
    angle = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1])) % 180.0
    if angle > 180.0 - 1e-9:
        angle = 0.0
    return k * math.sqrt(vals[1]), k * math.sqrt(vals[0]), angle


def ease_camera(path: np.ndarray, fps: float, ease_s: float = CAMERA_EASE_S) -> np.ndarray:
    """Smooth an N x 2 camera path (one row per video frame) with a Hann window ``ease_s`` wide.

    The window is symmetric, so a path at constant velocity is unchanged (no lag), and a step of D px is spread
    over ``ease_s``: at most about 2 D / (ease_s fps) px per frame. Past its ends the path is continued as its
    own mirror image through the end point, so the camera starts and stops exactly on the path, with no lead."""
    path = np.asarray(path, float)
    n = max(1, round(ease_s * fps)) | 1                   # odd: centred on the frame
    if n < 3 or len(path) < 2:
        return path.copy()
    w = np.hanning(n + 2)[1:-1]
    w /= w.sum()
    half = min(n // 2, len(path) - 1)                     # a mirror image needs that many frames to mirror
    padded = np.pad(path, ((half, half), (0, 0)), mode="reflect", reflect_type="odd")
    padded = np.pad(padded, ((n // 2 - half, n // 2 - half), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, c], w, mode="valid") for c in range(path.shape[1])], axis=1)


@dataclass(frozen=True)
class Caption:
    """One event line. frame: the video frame it appears at. idx: the run's frame number of the event."""

    frame: int
    idx: int
    state: str
    text: str


@dataclass(frozen=True, eq=False)
class HudPlan(tv.VideoPlan):
    captions: tuple[Caption, ...] = ()


def plan_hud_video(records: list[Record], info: RunInfo, *, fps: float = 30.0, speed: float = 1.0,
                   size: tuple[int, int] = (1280, 720), hold_s: float = tv.HOLD_S) -> HudPlan:
    """The M1 plan (same frame times, same path through the records) with the closer view, the eased camera and
    the captions' schedule."""
    plan = tv.plan_video(records, info, fps=fps, speed=speed, size=size, hold_s=hold_s,
                         view_footprints=VIEW_FOOTPRINTS)
    # M1 keeps the view inside the map, which near an edge pushes the drone to the side, under a panel
    drone = np.array([[f.x, f.y] for f in plan.frames])
    reach = np.array([STAGE[0] * plan.view_w, STAGE[1] * plan.view_w * size[1] / size[0]])
    cam = np.clip(np.array([[f.cx, f.cy] for f in plan.frames]), drone - reach, drone + reach)
    cam = ease_camera(cam, fps)
    frames = [dataclasses.replace(f, cx=float(cx), cy=float(cy)) for f, (cx, cy) in zip(plan.frames, cam, strict=True)]
    return HudPlan(frames=frames, fps=plan.fps, size=plan.size, view_w=plan.view_w,
                   captions=_schedule_captions(records, frames, plan.fps))


def _event_text(rec: Record, prev: Record | None) -> str | None:
    """"frame 133: LOST, camera jammed" for a state change or a record with extra["event"]; else None."""
    state = rec.track.state.value
    event = rec.extra.get("event")
    event = event.strip() if isinstance(event, str) and event.strip() else None
    if event is None and (prev is None or prev.track.state == rec.track.state):
        return None
    fix = rec.track.fix
    detail = event or (fix.reject_reason.replace("_", " ") if fix is not None and not fix.accepted else None)
    return f"frame {rec.idx}: {_STATE_WORDS[state]}" + (f", {detail}" if detail else "")


def _schedule_captions(records, frames, fps) -> tuple[Caption, ...]:
    first = {}                                           # record position -> the first video frame showing it
    for k, f in enumerate(frames):
        first.setdefault(f.rec, k)
    starts = sorted(first)
    out: list[Caption] = []
    gap = math.ceil(CAPTION_MIN_S * fps)
    for i, rec in enumerate(records):
        text = _event_text(rec, records[i - 1] if i else None)
        if text is None:
            continue
        j = np.searchsorted(starts, i)                   # a record skipped at a very high speed: the next shown
        if j == len(starts):
            break
        k = first[starts[j]]
        if out:
            k = max(k, out[-1].frame + gap)
        out.append(Caption(frame=int(k), idx=rec.idx, state=rec.track.state.value, text=text))
    return tuple(out)


def _caption(plan: HudPlan, k: int) -> tuple[Caption, float] | None:
    current = None
    for c in plan.captions:
        if c.frame > k:
            break
        current = c
    if current is None:
        return None
    left = CAPTION_S - (k - current.frame) / plan.fps
    if left <= 1e-9:
        return None
    return current, min(1.0, left / CAPTION_FADE_S)


def caption_at(plan: HudPlan, k: int) -> tuple[str, float] | None:
    """The caption on video frame k as (text, opacity in (0, 1]), or None."""
    c = _caption(plan, k)
    return None if c is None else (c[0].text, c[1])


# --- drawing ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class HudInfo:
    """What one drawn frame shows. Boxes are (x0, y0, x1, y1) in video px."""

    idx: int                         # frame number of the current record
    state: str
    inliers: int | None              # of the record's fix, accepted or not
    confidence: float | None         # of the accepted fix; None without one
    heading_deg: float
    xy: tuple[float, float]          # the drone, map px
    drone_px: tuple[float, float]    # ... and video px
    footprint: bool                  # an accepted fix's footprint is drawn
    rejected: str | None             # reject reason of a rejected fix drawn on this frame
    ellipse_px: tuple[float, float, float]   # 3-sigma semi-axes in video px, angle in degrees
    link: str | None                 # how the inset is tied to the map: "matches", "footprint" or None
    no_signal: bool                  # LOST and no picture
    frame_missing: bool
    caption: str | None
    badge_box: tuple[int, int, int, int]
    hud_box: tuple[int, int, int, int]
    inset_box: tuple[int, int, int, int]     # the picture itself
    caption_box: tuple[int, int, int, int] | None


class HudRenderer:
    """Draws the frames of a :class:`HudPlan`, in any order."""

    def __init__(self, run_dir: str | Path, records: list[Record], info: RunInfo, plan: HudPlan,
                 map_img: np.ndarray | None = None):
        self.run_dir, self.records, self.info, self.plan = Path(run_dir), records, info, plan
        if map_img is None:
            map_img = cv2.imread(str(info.map_path), cv2.IMREAD_COLOR)
            if map_img is None:
                raise FileNotFoundError(f"cannot read the map {info.map_path}")
        map_img = _bgr(map_img)
        h, w = map_img.shape[:2]
        if (w, h) != info.map_size:
            raise ValueError(f"the map is {w} x {h} px but run.json says the run was flown over "
                             f"{info.map_size[0]} x {info.map_size[1]} px ({info.map_path})")
        vw, vh = plan.size
        self.u = min(vh / 1080, vw / 1920)               # the layout unit: 1 px of a 1920 x 1080 video
        self.s = vw / plan.view_w                        # video px per map px
        self.k = min(self.s, 1.0)                        # layer px per map px (shrunk once when zoomed out)
        layer = cv2.resize(map_img, None, fx=self.k, fy=self.k, interpolation=cv2.INTER_AREA) if self.k < 1 \
            else map_img
        self.layer = cv2.convertScaleAbs(layer, alpha=MAP_DIM)
        self.t = np.array([r.t for r in records])
        self.pts = np.array([[r.track.x, r.track.y] for r in records])
        self.missed = [is_missed(r) for r in records]
        self.conf = [None if m else fix_confidence(r.track.fix) for r, m in zip(records, self.missed, strict=True)]
        self._cut_trail()
        self._timeline = self._timeline_strip()
        self._frame_cache: tuple[int, np.ndarray | None] = (-1, None)

    # -- sizes
    def _n(self, v: float, least: int = 1) -> int:
        """A length of the 1080p layout in this video's px."""
        return max(least, round(v * self.u))

    def _fs(self, v: float) -> float:
        return max(0.3, v * self.u)

    # -- geometry
    def _to_video(self, f: tv.VideoFrame, x: float, y: float) -> tuple[float, float]:
        w, h = self.plan.size
        return w / 2 + (x - f.cx) * self.s, h / 2 + (y - f.cy) * self.s

    def _video_px(self, f: tv.VideoFrame, pts: np.ndarray) -> np.ndarray:
        """Map px -> OpenCV's sub-pixel video coordinates (int32, SHIFT bits), for many points at once."""
        w, h = self.plan.size
        v = (np.asarray(pts, float) - (f.cx, f.cy)) * self.s + (w / 2, h / 2)
        return np.round((v - 0.5) * (1 << SHIFT)).astype(np.int32)

    # -- trail
    def _cut_trail(self) -> None:
        """Per segment i -> i + 1: its colour (the weaker end decides; red if either end has no fix), and for the
        red ones the dash pieces in map px, one pattern running on across a whole jam."""
        n = len(self.pts)
        self.dash, self.gap = 14.0 * self.u, 12.0 * self.u          # video px
        dash, gap = self.dash / self.s, self.gap / self.s            # map px
        self.seg_colour: list[tuple[int, int, int]] = []
        self.seg_phase = [0.0] * max(n - 1, 0)
        pieces: list[np.ndarray] = []
        self.pieces_before = [0] * n
        phase = 0.0
        for i in range(n - 1):
            self.pieces_before[i] = len(pieces)
            if self.missed[i] or self.missed[i + 1]:
                self.seg_colour.append(RED)
                self.seg_phase[i] = phase
                phase = tv._dash_pieces(self.pts[i], self.pts[i + 1], dash, gap, phase, pieces)
            else:
                self.seg_colour.append(confidence_colour(min(self.conf[i], self.conf[i + 1])))
                phase = 0.0
        if n:
            self.pieces_before[n - 1] = len(pieces)
        self.pieces = np.array(pieces).reshape(-1, 2, 2)

    def _draw_trail(self, img: np.ndarray, f: tv.VideoFrame) -> None:
        """A dark bed under the whole path so far, then each segment in its colour, then the live piece."""
        thick, bed = self._n(6, 2), self._n(6, 2) + 2 * self._n(2)
        v = self._video_px(f, self.pts[:f.rec + 1])
        d = self._to_video(f, f.x, f.y)
        live = f.rec < len(self.seg_colour)
        bed_pts = np.vstack([v, [px(*d)]]) if live else v
        cv2.polylines(img, [bed_pts], False, INK, bed, cv2.LINE_AA, SHIFT)
        for i in range(f.rec):
            if self.seg_colour[i] is not RED:
                cv2.line(img, v[i], v[i + 1], self.seg_colour[i], thick, cv2.LINE_AA, SHIFT)
        done = self.pieces[:self.pieces_before[f.rec]]
        if len(done):
            q = self._video_px(f, done.reshape(-1, 2)).reshape(-1, 2, 2)
            cv2.polylines(img, list(q), False, RED, thick, cv2.LINE_AA, SHIFT)
        if live:
            a = self._to_video(f, *self.pts[f.rec])
            if self.seg_colour[f.rec] is RED:
                dashed(img, a, d, RED, thick, self.dash, self.gap, self.seg_phase[f.rec] * self.s)
            else:
                cv2.line(img, px(*a), px(*d), self.seg_colour[f.rec], thick, cv2.LINE_AA, SHIFT)

    # -- pieces on the map
    def _fill(self, img: np.ndarray, poly: np.ndarray, colour, alpha: float) -> None:
        """Blend a filled polygon (sub-pixel int coordinates) into img, touching only its bounding box."""
        h, w = img.shape[:2]
        p = poly.astype(np.int64) >> SHIFT
        x0, y0 = max(int(p[:, 0].min()) - 2, 0), max(int(p[:, 1].min()) - 2, 0)
        x1, y1 = min(int(p[:, 0].max()) + 3, w), min(int(p[:, 1].max()) + 3, h)
        if x1 <= x0 or y1 <= y0:
            return
        roi = img[y0:y1, x0:x1]
        over = roi.copy()
        cv2.fillPoly(over, [poly - np.array([x0, y0], np.int32) * (1 << SHIFT)], colour, cv2.LINE_AA, SHIFT)
        cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, dst=roi)

    def _cov(self, f: tv.VideoFrame) -> np.ndarray:
        """The covariance at the frame's time: between two records, element by element (stays a covariance)."""
        i, j = f.rec, min(f.rec + 1, len(self.records) - 1)
        a = self.records[i].track.cov
        if j == i or self.t[j] <= self.t[i]:
            return a
        u = min(1.0, max(0.0, (f.t - self.t[i]) / (self.t[j] - self.t[i])))
        return (1 - u) * a + u * self.records[j].track.cov

    def _ellipse_poly(self, d, a: float, b: float, angle: float) -> np.ndarray:
        th = np.linspace(0, 2 * math.pi, 72, endpoint=False)
        ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        x, y = a * np.cos(th), b * np.sin(th)
        v = np.stack([d[0] + x * ca - y * sa, d[1] + x * sa + y * ca], axis=1)
        return np.round((v - 0.5) * (1 << SHIFT)).astype(np.int32)

    def _dashed_poly(self, img, pts, colour, thick) -> None:
        phase = 0.0
        for a, b in zip(pts, np.roll(pts, -1, axis=0), strict=True):
            phase = dashed(img, a, b, colour, thick, self.dash, self.gap, phase)

    def _label(self, img, text, x, y, scale, colour, thick=None, font=FONT) -> None:
        """Text with a dark edge, readable on any map. (x, y): left end of the baseline."""
        thick = thick or self._n(2)
        org = (round(x), round(y))
        cv2.putText(img, text, org, font, scale, INK, thick + 2 * self._n(2), cv2.LINE_AA)
        cv2.putText(img, text, org, font, scale, colour, thick, cv2.LINE_AA)

    def _chip(self, img, text, cx, cy, colour) -> None:
        """A solid label centred on (cx, cy): dark text on the colour, for the one thing that must not be missed."""
        scale, thick, pad = self._fs(0.85), self._n(2), self._n(12)
        (tw, th), _ = cv2.getTextSize(text, BOLD, scale, thick)
        x0, y0 = round(cx - tw / 2) - pad, round(cy - th / 2) - pad
        cv2.rectangle(img, (x0, y0), (x0 + tw + 2 * pad, y0 + th + 2 * pad), colour, -1)
        cv2.putText(img, text, (x0 + pad, y0 + pad + th), BOLD, scale, INK, thick, cv2.LINE_AA)

    def _draw_drone(self, img, d, heading_deg: float, colour) -> None:
        """An arrowhead pointing along the heading, in the state colour."""
        a = math.radians(heading_deg)
        fwd, right = np.array([math.sin(a), -math.cos(a)]), np.array([math.cos(a), math.sin(a)])
        r = 22 * self.u
        c = np.array(d)
        shape = np.array([c + 1.5 * r * fwd, c - r * fwd + 0.95 * r * right, c - 0.45 * r * fwd,
                          c - r * fwd - 0.95 * r * right])
        poly = np.round((shape - 0.5) * (1 << SHIFT)).astype(np.int32)
        cv2.polylines(img, [poly], True, INK, self._n(8, 3), cv2.LINE_AA, SHIFT)
        cv2.fillPoly(img, [poly], colour, cv2.LINE_AA, SHIFT)
        cv2.polylines(img, [poly], True, WHITE, self._n(2.5), cv2.LINE_AA, SHIFT)

    # -- the frame
    def draw(self, k: int) -> tuple[np.ndarray, HudInfo]:
        """Video frame k of the plan (BGR uint8) and what it shows."""
        f = self.plan.frames[k]
        rec = self.records[f.rec]
        fix = rec.track.fix
        state = rec.track.state.value
        colour = STATE_COLOURS[state]
        w, h = self.plan.size
        z = self.s / self.k                              # layer px -> video px
        m = np.array([[z, 0, w / 2 - f.cx * self.s], [0, z, h / 2 - f.cy * self.s]])
        m[:, 2] += 0.5 * (z - 1)                         # OpenCV's pixel centres sit at integers on both sides
        img = cv2.warpAffine(self.layer, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=BACKGROUND)
        d = self._to_video(f, f.x, f.y)

        # fills first, so the trail stays pure colour on top of them
        fp = None
        if f.footprint is not None:
            fp = self._video_px(f, f.footprint)
            self._fill(img, fp, WHITE, 0.10)
        a_map, b_map, angle = sigma_ellipse(self._cov(f))
        ell = (a_map * self.s, b_map * self.s, angle)
        ell_poly = self._ellipse_poly(d, *ell) if ell[0] >= 2 else None
        if ell_poly is not None:
            self._fill(img, ell_poly, colour, 0.16)

        self._draw_trail(img, f)

        if fp is not None:
            cv2.polylines(img, [fp], True, INK, self._n(3, 2) + 2, cv2.LINE_AA, SHIFT)
            cv2.polylines(img, [fp], True, WHITE, self._n(3, 2), cv2.LINE_AA, SHIFT)
        rejected = None
        if fix is not None and not fix.accepted and not rec.interpolated:
            rejected = fix.reject_reason
            corners = np.array([self._to_video(f, *c) for c in fix.footprint])
            self._dashed_poly(img, corners, RED, self._n(3, 2))
            cx, cy = corners.mean(axis=0)
            if 0.28 * w <= cx <= 0.66 * w and 0.15 * h <= cy <= 0.85 * h:      # in the free stage: label it there
                rejected_at = (cx, corners[:, 1].min() - self._n(30))
            else:                                         # far away or under a panel: say it next to the drone
                rejected_at = (d[0], d[1] - self._n(80))
        if ell_poly is not None:
            cv2.polylines(img, [ell_poly], True, INK, self._n(2.5) + 2, cv2.LINE_AA, SHIFT)
            cv2.polylines(img, [ell_poly], True, colour, self._n(2.5), cv2.LINE_AA, SHIFT)
            if ell[0] >= 60 * self.u:                    # big enough to wonder what it is
                reach = abs(ell[0] * math.sin(math.radians(angle))) + abs(ell[1] * math.cos(math.radians(angle)))
                self._label(img, "3-sigma", d[0] - 52 * self.u, d[1] + reach + self._n(34), self._fs(0.8), colour)

        # the corner links go under the camera panel (they come out from behind it), match lines over the picture
        inset_box = self._inset_box(f.rec)
        pairs = self._matches(rec) if self._frame(f.rec) is not None else None
        link = None
        if not is_missed(rec) and fix is not None and f.footprint is not None:
            link = "footprint" if pairs is None else "matches"
        if link == "footprint":
            self._draw_corner_link(img, f, inset_box)
        missing, no_signal, panel_box = self._draw_inset(img, f.rec, state, colour, inset_box)
        if link == "matches":
            self._draw_match_lines(img, f, pairs, inset_box)
        self._draw_drone(img, d, f.heading_deg, colour)
        if rejected is not None:
            self._chip(img, "FIX REJECTED: " + rejected.replace("_", " "), *rejected_at, RED)

        conf = self.conf[f.rec]
        inliers = None if fix is None else int(fix.inliers)
        hud_box, badge_box = self._draw_hud(img, rec, f, state, colour, inliers, rejected, conf, a_map)
        self._draw_legend(img, panel_box)
        self._draw_timeline(img, f)
        caption, caption_box = self._draw_caption(img, k)
        if state != "TRACKING":                           # the whole screen says it, also from the back row
            b = self._n(10, 2)
            cv2.rectangle(img, (b // 2, b // 2), (w - 1 - b // 2, h - 1 - b // 2), colour, b)
        return img, HudInfo(idx=rec.idx, state=state, inliers=inliers, confidence=conf, heading_deg=f.heading_deg,
                            xy=(f.x, f.y), drone_px=d, footprint=fp is not None, rejected=rejected, ellipse_px=ell,
                            link=link, no_signal=no_signal, frame_missing=missing, caption=caption,
                            badge_box=badge_box, hud_box=hud_box, inset_box=inset_box, caption_box=caption_box)

    # -- panels
    def _panel(self, img, box, alpha: float = 0.80) -> None:
        x0, y0, x1, y1 = box
        roi = img[max(y0, 0):y1, max(x0, 0):x1]
        if roi.size:
            cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, INK), alpha, 0, dst=roi)

    def _text_w(self, text, scale, thick, font=FONT) -> int:
        return cv2.getTextSize(text, font, scale, thick)[0][0]

    def _fit(self, text, scale, thick, room, font=FONT) -> float:
        """The scale at which ``text`` is no wider than ``room`` px (never larger than ``scale``)."""
        width = self._text_w(text, scale, thick, font)
        return scale if width <= room else scale * room / width

    def _draw_hud(self, img, rec, f, state, colour, inliers, rejected, conf, sigma_map):
        """Top left: the state badge, then the numbers."""
        m, pad = self._n(24), self._n(22)
        x0, y0 = m, m
        x1 = x0 + self._n(480)
        badge_h, row_h = self._n(86), self._n(52)
        rows = [("FRAME", f"{rec.idx} / {self.records[-1].idx}", WHITE),
                ("TIME", f"{f.t:.1f} s", WHITE),
                ("POSITION", f"{f.x:.0f}, {f.y:.0f}", WHITE),
                ("HEADING", f"{f.heading_deg:03.0f} deg", WHITE),
                ("INLIERS", "-" if inliers is None else f"{inliers} rejected" if rejected else str(inliers),
                 RED if rejected else WHITE),
                ("3-SIGMA", f"{sigma_map:.0f} px", WHITE),
                ("CONFIDENCE", "no fix" if conf is None else f"{100 * conf:.0f} %",
                 RED if conf is None else confidence_colour(conf))]
        bar_h = self._n(12, 2)
        y1 = y0 + badge_h + self._n(10) + row_h * len(rows) + bar_h + self._n(22)
        self._panel(img, (x0, y0, x1, y1))
        badge = (x0, y0, x1, y0 + badge_h)
        cv2.rectangle(img, badge[:2], (badge[2] - 1, badge[3] - 1), colour, -1)
        thick = self._n(3, 1)
        scale = self._fit(state, self._fs(1.55), thick, x1 - x0 - 2 * pad, BOLD)
        (tw, th), _ = cv2.getTextSize(state, BOLD, scale, thick)
        cv2.putText(img, state, (x0 + (x1 - x0 - tw) // 2, y0 + (badge_h + th) // 2), BOLD, scale, INK, thick,
                    cv2.LINE_AA)
        y = y0 + badge_h + self._n(10)
        for label, value, vc in rows:
            y += row_h
            base = y - self._n(15)
            cv2.putText(img, label, (x0 + pad, base), FONT, self._fs(0.68), LABEL, self._n(1.5), cv2.LINE_AA)
            vt = self._n(2)
            vs = self._fs(1.0)
            cv2.putText(img, value, (x1 - pad - self._text_w(value, vs, vt, BOLD), base), BOLD, vs, vc, vt,
                        cv2.LINE_AA)
        by = y + self._n(6)                               # the confidence as a bar under its number
        cv2.rectangle(img, (x0 + pad, by), (x1 - pad, by + bar_h), (70, 66, 62), -1)
        if conf is not None:
            cv2.rectangle(img, (x0 + pad, by), (x0 + pad + round((x1 - x0 - 2 * pad) * conf), by + bar_h),
                          confidence_colour(conf), -1)
        return (x0, y0, x1, y1), badge

    def _frame(self, i: int) -> np.ndarray | None:
        if self._frame_cache[0] != i:
            self._frame_cache = (i, cv2.imread(str(self.run_dir / self.records[i].source), cv2.IMREAD_COLOR))
        return self._frame_cache[1]

    def _inset_box(self, i: int) -> tuple[int, int, int, int]:
        """Where the camera picture of record i goes: top right, 30 % of the width, the frame's own shape."""
        w, h = self.plan.size
        m, head = self._n(24), self._n(46)
        frame = self._frame(i)
        fw, fh = self.info.frame_size if self.info.frame_size else (4, 3)
        if frame is not None:
            fh, fw = frame.shape[:2]
        iw = round(w * 0.30)
        ih = min(round(iw * fh / fw), round(0.42 * h))
        x1, y0 = w - m, m + head
        return x1 - iw, y0, x1, y0 + ih

    def _draw_inset(self, img, i, state, colour, box):
        """Top right: a title strip, then the frame the drone saw. Returns whether the file is missing, whether
        that reads NO SIGNAL, and the box of the whole panel."""
        x0, y0, x1, y1 = box
        iw, ih = x1 - x0, y1 - y0
        m = self._n(24)
        frame = self._frame(i)
        self._panel(img, (x0, m, x1, y0), 0.88)
        cv2.putText(img, "CAMERA", (x0 + self._n(14), y0 - self._n(14)), FONT, self._fs(0.75), LABEL, self._n(2),
                    cv2.LINE_AA)
        lost = state == "LOST"
        status = ("NO SIGNAL" if frame is None else "NO FIX") if lost else f"frame {self.records[i].idx}"
        st, ss = self._n(2), self._fs(0.75)
        cv2.putText(img, status, (x1 - self._n(14) - self._text_w(status, ss, st), y0 - self._n(14)), FONT, ss,
                    RED if lost else WHITE, st, cv2.LINE_AA)
        roi = img[y0:y1, x0:x1]
        if frame is None:
            roi[:] = (26, 24, 22)
            text, tc = ("NO SIGNAL", RED) if lost else ("no frame", (120, 116, 112))
        else:
            roi[:] = cv2.resize(frame, (iw, ih), interpolation=cv2.INTER_AREA)
            text, tc = ("NO FIX", RED) if lost else (None, None)
            if lost:                                      # the picture is there but tells the tracker nothing
                cv2.convertScaleAbs(roi, dst=roi, alpha=0.38)
        if text:
            thick = self._n(4, 1) if lost else self._n(2)
            scale = self._fit(text, self._fs(2.0 if lost else 0.9), thick, iw * 0.8, BOLD)
            (tw, th), _ = cv2.getTextSize(text, BOLD, scale, thick)
            org = (x0 + (iw - tw) // 2, y0 + (ih + th) // 2)
            if frame is not None:
                cv2.putText(img, text, org, BOLD, scale, INK, thick + 2 * self._n(3), cv2.LINE_AA)
            cv2.putText(img, text, org, BOLD, scale, tc, thick, cv2.LINE_AA)
        b = self._n(4, 2)                                 # a frame in the state colour, outside the picture
        for bx0, by0, bx1, by1 in ((x0 - b, m - b, x1 + b, m), (x0 - b, y1, x1 + b, y1 + b),
                                   (x0 - b, m, x0, y1), (x1, m, x1 + b, y1)):
            cv2.rectangle(img, (bx0, by0), (bx1 - 1, by1 - 1), colour, -1)
        return frame is None, lost and frame is None, (x0, m, x1, y1)

    def _matches(self, rec: Record) -> tuple[np.ndarray, np.ndarray] | None:
        """The record's match points (frame px, map px) if the run carries them, well-formed; else None."""
        m = rec.extra.get("matches")
        if not isinstance(m, dict):
            return None
        try:
            pf, pm = np.array(m["frame"], float), np.array(m["map"], float)
        except (KeyError, TypeError, ValueError):
            return None
        if pf.ndim != 2 or pf.shape[1:] != (2,) or pf.shape != pm.shape or not len(pf) \
                or not (np.isfinite(pf).all() and np.isfinite(pm).all()):
            return None
        keep = np.unique(np.linspace(0, len(pf) - 1, min(len(pf), MAX_MATCH_LINES)).round().astype(int))
        return pf[keep], pm[keep]

    def _draw_match_lines(self, img, f, pairs, inset_box) -> None:
        """The run's own match points: each from its place in the camera picture to its place on the map."""
        x0, y0, x1, y1 = inset_box
        fh, fw = self._frame(f.rec).shape[:2]
        r = self._n(3, 2) << SHIFT
        for (fx, fy), (mx, my) in zip(*pairs, strict=True):
            a = (x0 + fx * (x1 - x0) / fw, y0 + fy * (y1 - y0) / fh)
            b = self._to_video(f, mx, my)
            cv2.line(img, px(*a), px(*b), GREEN, self._n(1.5), cv2.LINE_AA, SHIFT)
            cv2.circle(img, px(*a), r, GREEN, -1, cv2.LINE_AA, SHIFT)
            cv2.circle(img, px(*b), r, GREEN, -1, cv2.LINE_AA, SHIFT)

    def _draw_corner_link(self, img, f, inset_box) -> None:
        """Without match points: the picture's corners to the footprint's corners (both TL, TR, BR, BL)."""
        x0, y0, x1, y1 = inset_box
        for a, c in zip([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], f.footprint, strict=True):
            b = self._to_video(f, *c)
            cv2.line(img, px(*a), px(*b), INK, self._n(1.5) + 2, cv2.LINE_AA, SHIFT)
            cv2.line(img, px(*a), px(*b), (215, 215, 215), self._n(1.5), cv2.LINE_AA, SHIFT)

    def _draw_legend(self, img, inset_panel) -> None:
        """Under the camera: what the trail colours and the ellipse mean."""
        x0, _, x1, y = inset_panel
        rows = [(GREEN, False, "fix: sure"), (YELLOW, False, "fix: weak"), (RED, True, "no fix: dead reckoning")]
        row_h, pad = self._n(38), self._n(14)
        top = y + self._n(14)
        bottom = top + row_h * len(rows) + self._n(14)
        if bottom > self.plan.size[1] - self._n(140):     # no room on a squat video: the caption needs the space
            return
        self._panel(img, (x0, top, x1, bottom))
        scale, thick = self._fs(0.72), self._n(2)
        for i, (colour, dash, text) in enumerate(rows):
            cy = top + self._n(7) + row_h * i + row_h // 2
            a, b = (x0 + pad, cy), (x0 + pad + self._n(70), cy)
            if dash:
                dashed(img, a, b, colour, self._n(6, 2), self.dash, self.gap)
            else:
                cv2.line(img, px(*a), px(*b), colour, self._n(6, 2), cv2.LINE_AA, SHIFT)
            cv2.putText(img, text, (b[0] + pad, cy + self._n(9)), FONT, scale, WHITE, thick, cv2.LINE_AA)

    def _timeline_strip(self) -> np.ndarray:
        """One row of the run's states over time, as wide as the timeline."""
        w = self.plan.size[0] - 2 * self._n(24)
        t0, t1 = self.t[0], self.t[-1]
        ts = t0 + (np.arange(w) + 0.5) / w * (t1 - t0)
        rec = np.clip(np.searchsorted(self.t, ts, side="right") - 1, 0, len(self.t) - 1)
        colours = np.array([STATE_COLOURS[r.track.state.value] for r in self.records], np.uint8)
        return colours[rec]

    def _draw_timeline(self, img, f) -> None:
        """Along the bottom: the states of the whole run, bright up to now, and a marker at now."""
        w, h = self.plan.size
        m, bar = self._n(24), self._n(12, 2)
        y1 = h - m
        y0 = y1 - bar
        span = self.t[-1] - self.t[0]
        now = round((f.t - self.t[0]) / span * (len(self._timeline) - 1)) if span > 0 else len(self._timeline) - 1
        strip = self._timeline.astype(np.float32)
        strip[now + 1:] = strip[now + 1:] * 0.35 + np.array(INK) * 0.65
        self._panel(img, (m - self._n(4), y0 - self._n(4), w - m + self._n(4), y1 + self._n(4)), 0.7)
        img[y0:y1, m:m + len(strip)] = strip.astype(np.uint8)[None]
        x = m + now
        cv2.rectangle(img, (x - self._n(2), y0 - self._n(7)), (x + self._n(2), y1 + self._n(7)), WHITE, -1)

    def _draw_caption(self, img, k):
        """Bottom centre, above the timeline: the latest event, with a tab in its state's colour."""
        c = _caption(self.plan, k)
        if c is None:
            return None, None
        cap, alpha = c
        w, h = self.plan.size
        thick, pad, tab = self._n(2.5), self._n(26), self._n(14)
        scale = self._fit(cap.text, self._fs(1.15), thick, w - 2 * self._n(60) - 2 * pad - tab, BOLD)
        (tw, th), _ = cv2.getTextSize(cap.text, BOLD, scale, thick)
        bh = self._n(78)
        bw = tw + 2 * pad + tab
        x0 = (w - bw) // 2
        y1 = h - self._n(62)
        y0 = y1 - bh
        box = (x0, y0, x0 + bw, y1)
        before = img[y0:y1, x0:x0 + bw].copy() if alpha < 1 else None
        self._panel(img, box, 0.88)
        cv2.rectangle(img, (x0, y0), (x0 + tab - 1, y1 - 1), STATE_COLOURS[cap.state], -1)
        cv2.putText(img, cap.text, (x0 + tab + pad, y0 + (bh + th) // 2), BOLD, scale, WHITE, thick, cv2.LINE_AA)
        if before is not None:
            roi = img[y0:y1, x0:x0 + bw]
            cv2.addWeighted(roi, alpha, before, 1 - alpha, 0, dst=roi)
        return cap.text, box


# --- writing ---------------------------------------------------------------------------------------------------

def write_hud_video(run_dir: str | Path, out_dir: str | Path | None = None, *, fps: float = 30.0, speed: float = 1.0,
                    size: tuple[int, int] = (1280, 720), map_img: np.ndarray | None = None) -> Path:
    """Render the run in ``run_dir`` as the HUD video to ``<out_dir>/trajectory.mp4`` (default: the run folder);
    returns the path. Same file format as :func:`dronenav.viz.trajectory_video.write_video`."""
    run_dir = Path(run_dir)
    tv._check_options(fps, speed, size)
    info = read_run_info(run_dir)
    records = read_run(run_dir / "run.jsonl")
    if not records:
        raise ValueError(f"{run_dir / 'run.jsonl'} has no records")
    plan = plan_hud_video(records, info, fps=fps, speed=speed, size=size)
    renderer = HudRenderer(run_dir, records, info, plan, map_img)
    return tv.encode_video(out_dir if out_dir is not None else run_dir, fps, size,
                           (renderer.draw(k)[0] for k in range(len(plan.frames))))
