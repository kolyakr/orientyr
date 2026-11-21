"""HUD video (M7-02): the trajectory video as a "mission control" screen that reads without narration.

It is the M1 video (:mod:`dronenav.viz.trajectory_video`: the same timing, the same smooth path through every
record) laid out like the web console (``ui/``): the map on a **stage**, a glass **side panel** with the camera and
the numbers, and a **dock** with the timeline (:func:`layout`).

On the stage:

- **trail coloured by confidence**: green where the fix was sure, yellow where it was weak
  (:func:`dronenav.confidence.fix_confidence`), red and dashed where there was no accepted fix and the position is
  dead-reckoned;
- **footprint** of the accepted fix (yellow outline with corner ticks); a **rejected fix** is drawn apart, red and
  dashed, with its reason; the **3-sigma ellipse** of the position covariance, which visibly grows through a jam;
- the **projection**: the camera picture in the side panel is tied to its footprint by the two outer lines of
  their common hull (:func:`projection_links`) over a faint cone. Two hull edges cannot cross and never cut through
  the footprint, whatever the heading. Inside it, **match lines** join points of the picture to their places on
  the map. They are drawn only from the run's own ``extra["matches"] = {"frame": [[x, y], ...], "map": [...]}``
  (frame px, map px; ``scripts/run.py`` writes a sample of the inliers); a record without them gets none;
- while **LOST** the stage gets a red vignette (amber while RELOCALIZING), the drone is a hollow outline, and the
  camera picture is dimmed and stamped (CAMERA JAMMED when the run says so, NO SIGNAL without a picture);
- **event captions**, a title and a short note ("Signal lost" / "frame 4560 · camera jammed · dead reckoning"),
  from each state change and each ``extra["event"]``. A caption stays ``CAPTION_S`` video seconds and is never
  replaced sooner than ``CAPTION_MIN_S``, so a quick LOST -> RELOCALIZING -> TRACKING at 8x speed can be read;
- **eased camera**: the drone's path smoothed with a ``CAMERA_EASE_S`` wide bell-shaped window (symmetric: no lag
  in steady flight; a relocalisation becomes a glide, never a cut) and **clamped to the map**
  (:func:`clamp_camera`): the stage shows map and nothing else, so near an edge the drone moves off its centre.
  Under the glass panels the map goes on; past its edge it is mirrored there, blurred and darkened by the glass.

In the side panel: the camera picture, the **state lamp** (TRACKING green, LOST red, RELOCALIZING amber) with a
line saying what the tracker is doing, confidence with a bar, matched points, the uncertainty in px with a word
(:func:`uncertainty_word`), the heading with a compass, and a locator plate (where on the whole map). In the dock:
the clock and a timeline of the whole run (matched points per frame coloured by confidence, the states under it),
bright up to now.

Everything scales with the video size (unit: 1 px of a 1920 x 1080 video); no text is smaller than ``MIN_TEXT``
of these. Text is set in DejaVu Sans (shipped with matplotlib, so the same on every machine) through Pillow.
Drawing is deterministic: the same run gives the same frames.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dronenav.confidence import fix_confidence
from dronenav.telemetry import Record, RunInfo, read_run, read_run_info
from dronenav.viz import trajectory_video as tv
from dronenav.viz.draw import SHIFT, dashed, is_missed, px
from dronenav.viz.trajectory_image import _bgr

__all__ = ["BAD", "CAPTION_S", "CONF_HIGH", "CONF_LOW", "GREEN", "MATCH", "MIN_TEXT", "OK", "RED", "STATE_COLOURS",
           "WARN", "YELLOW", "Caption", "HudInfo", "HudPlan", "HudRenderer", "Layout", "caption_at", "clamp_camera",
           "confidence_colour", "ease_camera", "layout", "plan_hud_video", "projection_links", "sigma_ellipse",
           "to_video", "uncertainty_word", "write_hud_video"]

Box = tuple[int, int, int, int]     # x0, y0, x1, y1 in video px

# Colours, BGR; the same tokens as the web console (ui/app/globals.css).
GREEN = (62, 196, 62)       # trail: a sure fix
YELLOW = (31, 210, 245)     # trail: a weak fix; also the footprint and the projection
RED = (28, 38, 227)         # trail: no fix
OK = (122, 224, 84)         # state lamps
WARN = (64, 201, 255)
BAD = (84, 95, 255)
SIGMA = (245, 201, 76)      # the uncertainty ellipse
MATCH = (240, 240, 236)     # match lines and their end points
INK = (251, 247, 245)       # text
MUTED = (188, 178, 170)     # labels: quieter than the values
FAINT = (120, 112, 106)     # hairlines, empty bars
DARK = (14, 12, 11)         # casings and shadows
GLASS = (28, 23, 20)        # the panels' tint
SLOT = (26, 24, 22)         # where a picture would be
STATE_COLOURS = {"TRACKING": OK, "LOST": BAD, "RELOCALIZING": WARN}

CONF_LOW, CONF_HIGH = 0.35, 0.65    # confidence at and below which the trail is yellow / at and above which green
SIGMAS = 3.0                # the ellipse holds the position with ~99 % probability (2D Gaussian: 98.9 %)
VIEW_FOOTPRINTS = 4.0       # the video's width spans this many median footprint diagonals (M1 video: 6)
CAMERA_EASE_S = 1.6         # width of the camera's smoothing window, video seconds
DRONE_REACH = 0.8           # the drone stays within this share of the stage's half-size from the camera centre,
                            # however far a relocalisation throws it (the map's edges come first)
CAPTION_S = 2.5             # a caption stays this long, video seconds
CAPTION_MIN_S = 1.2         # ... and at least this long before the next one replaces it
CAPTION_FADE_S = 0.4        # the end of CAPTION_S is a fade-out
MAP_DIM = 0.72              # the map is darkened so the overlay stands out
GLASS_ALPHA = 0.76          # how much of a panel is tint; the rest is the blurred map behind it
ALARM = {"LOST": ((36, 40, 214), 0.46), "RELOCALIZING": (WARN, 0.22)}   # the stage vignette: colour, strength
MAX_MATCH_LINES = 40
MIN_TEXT = 18               # the smallest text, px of a 1920 x 1080 video: readable on a demo screen
SIDE_W, DOCK_H = 520, 132   # the side panel's width and the dock's height, same unit
PAD = 28
_CAP = 0.727                # DejaVu Sans: height of a capital, in font sizes
_TITLES = {"LOST": "Signal lost", "RELOCALIZING": "Relocalizing"}


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


def uncertainty_word(sigma_px: float) -> str:
    """The 1-sigma position uncertainty (map px) as a word; the web console uses the same steps."""
    if sigma_px <= 3:
        return "very precise"
    if sigma_px <= 15:
        return "precise"
    return "drifting" if sigma_px <= 40 else "wide"


@dataclass(frozen=True)
class Layout:
    """Where things go on a video of some size. u: video px per px of the 1920 x 1080 design."""

    u: float
    stage: Box      # the map
    side: Box       # camera, state, numbers, locator: the full height on the right
    dock: Box       # clock and timeline: under the stage


def layout(size: tuple[int, int]) -> Layout:
    """The three areas of a ``size`` = (w, h) video. They tile it: no gap, no overlap."""
    w, h = size
    u = min(h / 1080, w / 1920)
    sx, dy = w - round(SIDE_W * u), h - round(DOCK_H * u)
    return Layout(u=u, stage=(0, 0, sx, dy), side=(sx, 0, w, h), dock=(0, dy, sx, h))


def clamp_camera(cam: np.ndarray, map_size: tuple[int, int], half: tuple[float, float]) -> np.ndarray:
    """Camera centres (N x 2, map px) moved so a view of half-size ``half`` (map px) stays inside the map. In a
    direction where the view is larger than the map it cannot: there the camera sits on the map's middle."""
    out = np.array(cam, float)
    for c in (0, 1):
        lo, hi = half[c], map_size[c] - half[c]
        out[:, c] = map_size[c] / 2 if lo >= hi else np.clip(out[:, c], lo, hi)
    return out


def _hull(foot: np.ndarray, box) -> tuple[np.ndarray, list[int]]:
    """The 8 points (footprint corners, then the box's corners) and the indices of their convex hull, in order."""
    x0, y0, x1, y1 = box
    pts = np.vstack([np.asarray(foot, float), [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]]).astype(float)
    idx = cv2.convexHull(pts.astype(np.float32), returnPoints=False).ravel().tolist()
    return pts, idx


def projection_links(foot: np.ndarray, box) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """The two lines that tie a footprint (4 x 2, video px) to the camera picture ``box``: the edges of the convex
    hull of both that run from one to the other, each as (footprint corner, box corner). Hull edges do not cross
    each other and do not cut through either shape, whatever the footprint's rotation. Empty when one shape hides
    the other's corners (the footprint covers the picture)."""
    pts, idx = _hull(foot, box)
    links = []
    for a, b in zip(idx, idx[1:] + idx[:1], strict=True):
        if (a < 4) != (b < 4):
            f, p = (a, b) if a < 4 else (b, a)
            links.append(((float(pts[f][0]), float(pts[f][1])), (float(pts[p][0]), float(pts[p][1]))))
    return links if len(links) == 2 else []


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
    """One event. frame: the video frame it appears at. idx: the run's frame number of the event."""

    frame: int
    idx: int
    state: str
    title: str
    note: str

    @property
    def text(self) -> str:
        return f"{self.title} · {self.note}"


@dataclass(frozen=True, eq=False)
class HudPlan(tv.VideoPlan):
    """The M1 plan; a frame's (cx, cy) is the map point at the middle of ``stage``."""

    captions: tuple[Caption, ...] = ()
    stage: Box = (0, 0, 0, 0)
    speed: float = 1.0


def plan_hud_video(records: list[Record], info: RunInfo, *, fps: float = 30.0, speed: float = 1.0,
                   size: tuple[int, int] = (1280, 720), hold_s: float = tv.HOLD_S) -> HudPlan:
    """The M1 plan (same frame times, same path through the records) with the closer view, the eased camera kept
    on the map and the captions' schedule."""
    plan = tv.plan_video(records, info, fps=fps, speed=speed, size=size, hold_s=hold_s,
                         view_footprints=VIEW_FOOTPRINTS)
    x0, y0, x1, y1 = stage = layout(plan.size).stage
    s = plan.size[0] / plan.view_w
    half = ((x1 - x0) / 2 / s, (y1 - y0) / 2 / s)
    drone = np.array([[f.x, f.y] for f in plan.frames])
    cam = ease_camera(clamp_camera(drone, info.map_size, half), fps)
    reach = DRONE_REACH * np.array(half)
    cam = clamp_camera(np.clip(cam, drone - reach, drone + reach), info.map_size, half)
    frames = [dataclasses.replace(f, cx=float(cx), cy=float(cy)) for f, (cx, cy) in zip(plan.frames, cam, strict=True)]
    return HudPlan(frames=frames, fps=plan.fps, size=plan.size, view_w=plan.view_w, stage=stage, speed=float(speed),
                   captions=_schedule_captions(records, frames, plan.fps))


def to_video(plan: HudPlan, k: int, x: float, y: float) -> tuple[float, float]:
    """Map px -> video px at plan frame k."""
    f = plan.frames[k]
    x0, y0, x1, y1 = plan.stage
    s = plan.size[0] / plan.view_w
    return (x0 + x1) / 2 + (x - f.cx) * s, (y0 + y1) / 2 + (y - f.cy) * s


def _count(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _reason(fix) -> str:
    return "match rejected" + (f": {fix.reject_reason.replace('_', ' ')}" if fix.reject_reason else "")


def _live_note(rec: Record) -> str:
    """What the tracker is doing on this record, in a few words."""
    state, fix = rec.track.state.value, rec.track.fix
    why = "camera jammed" if rec.extra.get("jammed") else "no match"
    if state == "LOST":
        return f"{why} · dead reckoning"
    if state == "RELOCALIZING":
        if fix is None:
            return "searching the whole map"
        return "match found · confirming" if fix.accepted else _reason(fix)
    return f"{why} · coasting" if is_missed(rec) else "locked to the map"


def _event(rec: Record, prev: Record | None, was_lost: bool) -> tuple[str, str] | None:
    """(title, note) for a state change or a record with extra["event"]; else None."""
    state, fix = rec.track.state.value, rec.track.fix
    event = rec.extra.get("event")
    event = event.strip() if isinstance(event, str) and event.strip() else None
    changed = prev is not None and prev.track.state != rec.track.state
    if event is None and not changed:
        return None
    if state == "LOST":
        detail = f"{event or ('camera jammed' if rec.extra.get('jammed') else 'no match')} · dead reckoning"
    elif state == "RELOCALIZING":
        detail = event or _live_note(rec)
    else:
        detail = event or (f"{_count(int(fix.inliers))} points matched" if fix is not None and fix.accepted else None)
    if state in _TITLES:
        title = _TITLES[state]
    elif changed or prev is None:
        title = "Position recovered" if was_lost else "Position locked"
    else:                                                 # an event of its own while tracking: it is the title
        title, detail = event[:1].upper() + event[1:], None
    return title, f"frame {rec.idx}" + (f" · {detail}" if detail else "")


def _schedule_captions(records, frames, fps) -> tuple[Caption, ...]:
    first = {}                                           # record position -> the first video frame showing it
    for k, f in enumerate(frames):
        first.setdefault(f.rec, k)
    starts = sorted(first)
    out: list[Caption] = []
    gap = math.ceil(CAPTION_MIN_S * fps)
    prev, was_lost = None, False
    for i, rec in enumerate(records):
        if not rec.track.has_position:                   # before the first position: nothing is on screen yet
            continue
        event = _event(rec, prev, was_lost)
        prev = rec
        was_lost = rec.track.state.value == "LOST" or (was_lost and rec.track.state.value != "TRACKING")
        if event is None:
            continue
        j = np.searchsorted(starts, i)                   # a record skipped at a very high speed: the next shown
        if j == len(starts):
            break
        k = first[starts[j]]
        if out:
            k = max(k, out[-1].frame + gap)
        out.append(Caption(frame=int(k), idx=rec.idx, state=rec.track.state.value, title=event[0], note=event[1]))
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


# --- text ------------------------------------------------------------------------------------------------------

class _Type:
    """Text as alpha masks, set with Pillow and cached: the HUD repeats most of its strings frame after frame."""

    def __init__(self) -> None:
        spec = importlib.util.find_spec("matplotlib")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("the HUD video needs matplotlib's fonts (pip install -r requirements.txt)")
        self._dir = Path(spec.submodule_search_locations[0]) / "mpl-data" / "fonts" / "ttf"
        self._fonts: dict = {}
        self._masks: dict = {}

    def _font(self, size: int, bold: bool):
        key = (size, bold)
        if key not in self._fonts:
            name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
            self._fonts[key] = ImageFont.truetype(str(self._dir / name), size)
        return self._fonts[key]

    def width(self, text: str, size: int, bold: bool = False, tracking: float = 0.0) -> float:
        font = self._font(size, bold)
        if not tracking:
            return font.getlength(text)
        return sum(font.getlength(c) for c in text) + tracking * size * max(len(text) - 1, 0)

    def mask(self, text: str, size: int, bold: bool = False, tracking: float = 0.0) -> tuple[np.ndarray, int]:
        """(alpha mask, row of the capitals' top in it)."""
        key = (text, size, bold, tracking)
        got = self._masks.get(key)
        if got is None:
            font = self._font(size, bold)
            ascent, descent = font.getmetrics()
            im = Image.new("L", (math.ceil(self.width(text, size, bold, tracking)) + 2, ascent + descent + 2), 0)
            d = ImageDraw.Draw(im)
            if tracking:
                x = 0.0
                for c in text:
                    d.text((x, 0), c, font=font, fill=255)
                    x += font.getlength(c) + tracking * size
            else:
                d.text((0, 0), text, font=font, fill=255)
            if len(self._masks) > 4000:                   # numbers change every frame: do not keep them all
                self._masks.clear()
            got = self._masks[key] = (np.asarray(im).copy(), ascent - round(_CAP * size))
        return got


def _stamp(img: np.ndarray, mask: np.ndarray, x: int, y: int, colour, alpha: float = 1.0) -> None:
    """Blend ``colour`` into img through an alpha mask whose top-left corner goes to (x, y)."""
    h, w = img.shape[:2]
    mh, mw = mask.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + mw, w), min(y + mh, h)
    if x1 <= x0 or y1 <= y0:
        return
    a = mask[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32)[..., None] * (alpha / 255.0)
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi + (np.array(colour, np.float32) - roi) * a + 0.5).astype(np.uint8)


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
    sigma_px: float                  # 1-sigma uncertainty along the worst direction, map px
    word: str                        # ... and the word shown for it
    note: str                        # the line under the state lamp
    projection: bool                 # the camera picture is tied to its footprint
    match_lines: int                 # how many of the run's match pairs are drawn
    jammed: bool                     # the run marks this frame as jammed
    no_signal: bool                  # LOST and no picture
    frame_missing: bool
    caption: str | None
    min_text_px: float               # the smallest text on the frame, px of a 1920 x 1080 video
    stage_box: Box
    side_box: Box
    dock_box: Box
    lamp_box: Box
    inset_box: Box                   # the camera picture itself
    caption_box: Box | None


class HudRenderer:
    """Draws the frames of a :class:`HudPlan`, in any order."""

    def __init__(self, run_dir: str | Path, records: list[Record], info: RunInfo, plan: HudPlan,
                 map_img: np.ndarray | None = None):
        self.run_dir, self.info, self.plan = Path(run_dir), info, plan
        # a plan frame's ``rec`` indexes the records as given; here only those with a position are kept
        self._at = {i: n for n, i in enumerate(i for i, r in enumerate(records) if r.track.has_position)}
        self.records = records = [r for r in records if r.track.has_position]
        if map_img is None:
            map_img = cv2.imread(str(info.map_path), cv2.IMREAD_COLOR)
            if map_img is None:
                raise FileNotFoundError(f"cannot read the map {info.map_path}")
        map_img = _bgr(map_img)
        h, w = map_img.shape[:2]
        if (w, h) != info.map_size:
            raise ValueError(f"the map is {w} x {h} px but run.json says the run was flown over "
                             f"{info.map_size[0]} x {info.map_size[1]} px ({info.map_path})")
        self.lay = layout(plan.size)
        self.u = self.lay.u                              # the layout unit: 1 px of a 1920 x 1080 video
        self.s = plan.size[0] / plan.view_w              # video px per map px
        self.k = min(self.s, 1.0)                        # layer px per map px (shrunk once when zoomed out)
        layer = cv2.resize(map_img, None, fx=self.k, fy=self.k, interpolation=cv2.INTER_AREA) if self.k < 1 \
            else map_img
        self.layer = cv2.convertScaleAbs(layer, alpha=MAP_DIM)
        x0, y0, x1, y1 = self.lay.stage
        self.centre = ((x0 + x1) / 2, (y0 + y1) / 2)     # where the camera's map point is on the video
        self.t = np.array([r.t for r in records])
        self.pts = np.array([[r.track.x, r.track.y] for r in records])
        self.missed = [is_missed(r) for r in records]
        self.conf = [None if m else fix_confidence(r.track.fix) for r, m in zip(records, self.missed, strict=True)]
        self.type = _Type()
        self._min_text = math.inf
        self._cut_trail()
        self._build_alarm()
        self._build_locator(layer)
        self._build_timeline()
        self._frame_cache: tuple[int, np.ndarray | None] = (-1, None)

    # -- sizes
    def _n(self, v: float, least: int = 1) -> int:
        """A length of the 1080p layout in this video's px."""
        return max(least, round(v * self.u))

    # -- geometry
    def _to_video(self, f: tv.VideoFrame, x: float, y: float) -> tuple[float, float]:
        return self.centre[0] + (x - f.cx) * self.s, self.centre[1] + (y - f.cy) * self.s

    def _video_px(self, f: tv.VideoFrame, pts: np.ndarray) -> np.ndarray:
        """Map px -> OpenCV's sub-pixel video coordinates (int32, SHIFT bits), for many points at once."""
        v = (np.asarray(pts, float) - (f.cx, f.cy)) * self.s + self.centre
        return self._sub(v)

    @staticmethod
    def _sub(v) -> np.ndarray:
        return np.round((np.asarray(v, float) - 0.5) * (1 << SHIFT)).astype(np.int32)

    # -- text
    def _text(self, img, text: str, x: float, y: float, size: float, colour, *, bold: bool = False,
              align: str = "l", tracking: float = 0.0, alpha: float = 1.0, halo: bool = False,
              fit: float | None = None) -> int:
        """Text whose capitals have their top at y; x is its left end, right end ("r") or middle ("c"). ``size``
        is in 1080p px; ``fit``: the room in video px, the text is set smaller to fit it. Returns its width."""
        px_size = max(6, round(size * self.u))
        if fit is not None:
            width = self.type.width(text, px_size, bold, tracking)
            if width > fit:
                px_size = max(6, math.floor(px_size * fit / width))
                size = min(size, px_size / self.u)
        self._min_text = min(self._min_text, size)
        mask, cap = self.type.mask(text, px_size, bold, tracking)
        w = mask.shape[1]
        left = round(x - (w if align == "r" else w / 2 if align == "c" else 0))
        top = round(y) - cap
        if halo:                                          # a soft dark edge: readable on any map
            r = max(1, round(px_size / 9))
            wide = cv2.copyMakeBorder(mask, 3 * r, 3 * r, 3 * r, 3 * r, cv2.BORDER_CONSTANT, value=0)
            wide = cv2.dilate(wide, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            _stamp(img, cv2.GaussianBlur(wide, (0, 0), r), left - 3 * r, top - 3 * r, DARK, 0.9 * alpha)
        _stamp(img, mask, left, top, colour, alpha)
        return w

    def _label(self, img, text: str, x: float, y: float, align: str = "l") -> int:
        """A field name: small capitals, spaced out, quiet."""
        return self._text(img, text, x, y, MIN_TEXT, MUTED, tracking=0.07, align=align)

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

    def _draw_trail(self, img: np.ndarray, f: tv.VideoFrame, i: int) -> None:
        """A dark bed under the whole path so far, then each segment in its colour, then the live piece."""
        thick, bed = self._n(5, 2), self._n(5, 2) + 2 * self._n(2)
        v = self._video_px(f, self.pts[:i + 1])
        d = self._to_video(f, f.x, f.y)
        live = i < len(self.seg_colour)
        bed_pts = np.vstack([v, [px(*d)]]) if live else v
        cv2.polylines(img, [bed_pts], False, DARK, bed, cv2.LINE_AA, SHIFT)
        for j in range(i):
            if self.seg_colour[j] is not RED:
                cv2.line(img, v[j], v[j + 1], self.seg_colour[j], thick, cv2.LINE_AA, SHIFT)
        done = self.pieces[:self.pieces_before[i]]
        if len(done):
            q = self._video_px(f, done.reshape(-1, 2)).reshape(-1, 2, 2)
            cv2.polylines(img, list(q), False, RED, thick, cv2.LINE_AA, SHIFT)
        if live:
            a = self._to_video(f, *self.pts[i])
            if self.seg_colour[i] is RED:
                dashed(img, a, d, RED, thick, self.dash, self.gap, self.seg_phase[i] * self.s)
            else:
                cv2.line(img, px(*a), px(*d), self.seg_colour[i], thick, cv2.LINE_AA, SHIFT)

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

    def _line(self, img, a, b, colour, thick: int, alpha: float = 1.0) -> None:
        """An anti-aliased line, optionally see-through."""
        if alpha >= 1:
            cv2.line(img, px(*a), px(*b), colour, thick, cv2.LINE_AA, SHIFT)
            return
        self._fill_shape(img, lambda over, off: cv2.line(
            over, px(a[0] - off[0], a[1] - off[1]), px(b[0] - off[0], b[1] - off[1]), colour, thick, cv2.LINE_AA,
            SHIFT), (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])), thick, alpha)

    @staticmethod
    def _fill_shape(img, draw, bounds, margin: int, alpha: float) -> None:
        """Run ``draw(copy, offset)`` on a copy of the part of img inside ``bounds`` and blend it back."""
        h, w = img.shape[:2]
        x0, y0 = max(math.floor(bounds[0]) - margin - 2, 0), max(math.floor(bounds[1]) - margin - 2, 0)
        x1, y1 = min(math.ceil(bounds[2]) + margin + 3, w), min(math.ceil(bounds[3]) + margin + 3, h)
        if x1 <= x0 or y1 <= y0:
            return
        roi = img[y0:y1, x0:x1]
        over = roi.copy()
        draw(over, (x0, y0))
        cv2.addWeighted(over, alpha, roi, 1 - alpha, 0, dst=roi)

    def _on_stage(self, corners: np.ndarray) -> bool:
        """Is this quadrilateral (video px) a footprint one can look at: finite, with its middle on the stage and
        no larger than the stage?"""
        x0, y0, x1, y1 = self.lay.stage
        if not np.isfinite(corners).all():
            return False
        cx, cy = corners.mean(axis=0)
        span = np.ptp(corners, axis=0)
        return bool(x0 <= cx <= x1 and y0 <= cy <= y1 and span[0] <= x1 - x0 and span[1] <= y1 - y0)

    def _cov(self, f: tv.VideoFrame, i: int) -> np.ndarray:
        """The covariance at the frame's time: between two records, element by element (stays a covariance)."""
        j = min(i + 1, len(self.records) - 1)
        a = self.records[i].track.cov
        if j == i or self.t[j] <= self.t[i]:
            return a
        u = min(1.0, max(0.0, (f.t - self.t[i]) / (self.t[j] - self.t[i])))
        return (1 - u) * a + u * self.records[j].track.cov

    def _ellipse_poly(self, d, a: float, b: float, angle: float) -> np.ndarray:
        th = np.linspace(0, 2 * math.pi, 96, endpoint=False)
        ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        x, y = a * np.cos(th), b * np.sin(th)
        return self._sub(np.stack([d[0] + x * ca - y * sa, d[1] + x * sa + y * ca], axis=1))

    def _dashed_poly(self, img, pts, colour, thick) -> None:
        phase = 0.0
        for a, b in zip(pts, np.roll(pts, -1, axis=0), strict=True):
            phase = dashed(img, a, b, colour, thick, self.dash, self.gap, phase)

    def _draw_footprint(self, img, fp: np.ndarray) -> None:
        """A thin outline with heavier corner ticks, as the console draws it."""
        cv2.polylines(img, [fp], True, DARK, self._n(2) + 2, cv2.LINE_AA, SHIFT)
        cv2.polylines(img, [fp], True, YELLOW, self._n(2), cv2.LINE_AA, SHIFT)
        for j in range(4):
            c = fp[j].astype(float)
            for o in (fp[(j + 1) % 4], fp[(j - 1) % 4]):
                e = np.round(c + 0.16 * (o - c)).astype(np.int32)
                cv2.line(img, tuple(fp[j].tolist()), tuple(e.tolist()), YELLOW, self._n(5, 2), cv2.LINE_AA, SHIFT)

    def _draw_drone(self, img, d, heading_deg: float, state: str) -> None:
        """An arrowhead pointing along the heading: light while tracking, a hollow outline in the state colour
        while the position is not measured."""
        a = math.radians(heading_deg)
        fwd, right = np.array([math.sin(a), -math.cos(a)]), np.array([math.cos(a), math.sin(a)])
        r = 20 * self.u
        c = np.array(d)
        shape = np.array([c + 1.5 * r * fwd, c - r * fwd + 0.95 * r * right, c - 0.45 * r * fwd,
                          c - r * fwd - 0.95 * r * right])
        poly = self._sub(shape)
        cv2.polylines(img, [poly], True, DARK, self._n(8, 3), cv2.LINE_AA, SHIFT)
        if state == "TRACKING":
            cv2.fillPoly(img, [poly], (220, 231, 236), cv2.LINE_AA, SHIFT)
            cv2.polylines(img, [poly], True, DARK, self._n(1.5), cv2.LINE_AA, SHIFT)
        else:
            self._fill(img, poly, DARK, 0.55)
            cv2.polylines(img, [poly], True, STATE_COLOURS[state], self._n(3.5, 2), cv2.LINE_AA, SHIFT)

    # -- the alarm
    def _build_alarm(self) -> None:
        """How much of the alarm colour each stage pixel gets: none in the middle, all of it in the corners."""
        x0, y0, x1, y1 = self.lay.stage
        yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
        r = np.hypot((xx - (x1 - x0 - 1) / 2) / ((x1 - x0) / 2), (yy - (y1 - y0 - 1) / 2) / ((y1 - y0) / 2))
        v = np.clip((r - 0.62) / 0.6, 0.0, 1.0)
        self._alarm = (v * v * (3 - 2 * v))[..., None]

    def _draw_alarm(self, img, state: str) -> None:
        if state not in ALARM:
            return
        x0, y0, x1, y1 = self.lay.stage
        roi = img[y0:y1, x0:x1]
        colour, strength = ALARM[state]
        roi[:] = (roi + (np.array(colour, np.float32) - roi) * (strength * self._alarm) + 0.5).astype(np.uint8)

    # -- the frame
    def draw(self, k: int) -> tuple[np.ndarray, HudInfo]:
        """Video frame k of the plan (BGR uint8) and what it shows."""
        f = self.plan.frames[k]
        i = self._at[f.rec]
        rec = self.records[i]
        fix = rec.track.fix
        state = rec.track.state.value
        w, h = self.plan.size
        self._min_text = math.inf
        z = self.s / self.k                              # layer px -> video px
        m = np.array([[z, 0, self.centre[0] - f.cx * self.s], [0, z, self.centre[1] - f.cy * self.s]])
        m[:, 2] += 0.5 * (z - 1)                         # OpenCV's pixel centres sit at integers on both sides
        # the camera keeps the stage on the map; under the panels the map is mirrored past its edge
        img = cv2.warpAffine(self.layer, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        d = self._to_video(f, f.x, f.y)
        self._draw_alarm(img, state)

        frame = self._frame(i)
        inset_box = self._inset_box(frame)
        foot_px = None if f.footprint is None else np.array([self._to_video(f, *c) for c in f.footprint])
        # the projection ends on the picture's edge that faces the map, so its lines stay off the panel's text
        near = (inset_box[0], inset_box[1], inset_box[0], inset_box[3])
        links = []
        if not self.missed[i] and foot_px is not None and frame is not None:
            links = projection_links(foot_px, near)
        pairs = self._matches(rec) if links else None

        # fills first, so the trail stays pure colour on top of them
        fp = None
        if foot_px is not None:
            fp = self._sub(foot_px)
            if links:
                pts, idx = _hull(foot_px, near)
                self._fill(img, self._sub(pts[idx]), YELLOW, 0.05)
            self._fill(img, fp, YELLOW, 0.06)
        cov = self._cov(f, i)
        a_map, b_map, angle = sigma_ellipse(cov)
        ell = (a_map * self.s, b_map * self.s, angle)
        ell_poly = self._ellipse_poly(d, *ell) if ell[0] >= 2 else None
        if ell_poly is not None:                         # an ellipse larger than the stage must not flood it
            self._fill(img, ell_poly, SIGMA, 0.14 * min(1.0, max(0.25, 0.3 * (self.lay.stage[3]) / ell[0])))

        self._draw_trail(img, f, i)

        if fp is not None:
            self._draw_footprint(img, fp)
        rejected = None
        if fix is not None and not fix.accepted and not rec.interpolated:
            rejected = fix.reject_reason
            corners = np.array([self._to_video(f, *c) for c in fix.footprint])
            if self._on_stage(corners):                  # a wild one (4 chance matches) would only be noise
                self._dashed_poly(img, corners, BAD, self._n(3, 2))
                top = corners[:, 1].min() - self._n(40)
                if top < self.lay.stage[1] + self._n(16):
                    top = corners[:, 1].max() + self._n(18)
                self._text(img, _reason(fix), corners[:, 0].mean(), top, 22, BAD, bold=True, align="c", halo=True)
        if ell_poly is not None:
            cv2.polylines(img, [ell_poly], True, DARK, self._n(2.5) + 2, cv2.LINE_AA, SHIFT)
            cv2.polylines(img, [ell_poly], True, SIGMA, self._n(2.5), cv2.LINE_AA, SHIFT)
            reach = math.hypot(ell[0] * math.cos(math.radians(angle)), ell[1] * math.sin(math.radians(angle)))
            if ell[0] >= 60 * self.u and d[0] + reach + self._n(70) < self.lay.stage[2]:   # big enough to wonder
                self._text(img, "3σ", d[0] + reach + self._n(12), d[1] - self._n(8), 22, SIGMA, bold=True, halo=True)
        self._draw_drone(img, d, f.heading_deg, state)
        caption, caption_box = self._draw_caption(img, k, d)

        # the panels: glass over whatever the map shows there
        self._glass(img, self.lay.side)
        self._glass(img, self.lay.dock)
        self._hairline(img, (self.lay.side[0], 0, self.lay.side[0] + 1, h))
        self._hairline(img, (0, self.lay.dock[1], self.lay.dock[2], self.lay.dock[1] + 1))
        for a, b in links:                               # over the glass: they end on the picture's corners
            self._line(img, a, b, DARK, self._n(1.5) + 2, 0.35)
            self._line(img, a, b, YELLOW, self._n(1.5), 0.8)
        jammed = bool(rec.extra.get("jammed"))
        missing, no_signal = self._draw_inset(img, rec, frame, state, jammed, inset_box, bool(links))
        n_lines = 0
        if pairs is not None:
            n_lines = self._draw_match_lines(img, f, pairs, inset_box, frame)

        conf = self.conf[i]
        inliers = None if fix is None else int(fix.inliers)
        sigma = a_map / SIGMAS
        note = _live_note(rec)
        lamp_box = self._draw_side(img, f, rec, state, note, conf, inliers, rejected, sigma, inset_box)
        self._draw_locator(img, f, i, state, cov)
        self._draw_dock(img, f, i)
        return img, HudInfo(idx=rec.idx, state=state, inliers=inliers, confidence=conf, heading_deg=f.heading_deg,
                            xy=(f.x, f.y), drone_px=d, footprint=fp is not None, rejected=rejected, ellipse_px=ell,
                            sigma_px=sigma, word=uncertainty_word(sigma), note=note, projection=bool(links),
                            match_lines=n_lines, jammed=jammed, no_signal=no_signal, frame_missing=missing,
                            caption=caption, min_text_px=self._min_text, stage_box=self.lay.stage,
                            side_box=self.lay.side, dock_box=self.lay.dock, lamp_box=lamp_box, inset_box=inset_box,
                            caption_box=caption_box)

    # -- panels
    def _glass(self, img, box: Box, alpha: float = GLASS_ALPHA) -> None:
        """Frosted dark glass: what is behind, blurred, under a dark tint."""
        x0, y0, x1, y1 = box
        roi = img[max(y0, 0):y1, max(x0, 0):x1]
        if not roi.size:
            return
        h, w = roi.shape[:2]
        small = cv2.resize(roi, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
        blur = cv2.resize(cv2.GaussianBlur(small, (0, 0), 2.5), (w, h), interpolation=cv2.INTER_LINEAR)
        cv2.convertScaleAbs(blur, dst=roi, alpha=1 - alpha)
        cv2.add(roi, tuple(alpha * c for c in GLASS) + (0,), dst=roi)

    @staticmethod
    def _hairline(img, box: Box, alpha: float = 0.14) -> None:
        x0, y0, x1, y1 = box
        roi = img[y0:y1, x0:x1]
        if roi.size:
            cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, INK), alpha, 0, dst=roi)

    def _frame(self, i: int) -> np.ndarray | None:
        if self._frame_cache[0] != i:
            self._frame_cache = (i, cv2.imread(str(self.run_dir / self.records[i].source), cv2.IMREAD_COLOR))
        return self._frame_cache[1]

    def _slot(self) -> Box:
        """Room for the camera picture: the side panel's width, 4:3."""
        x0 = self.lay.side[0] + self._n(PAD)
        x1 = self.lay.side[2] - self._n(PAD)
        y0 = self._n(58)
        return x0, y0, x1, y0 + round((x1 - x0) * 3 / 4)

    def _inset_box(self, frame: np.ndarray | None) -> Box:
        """Where the camera picture goes: the slot's width, the frame's own shape (a tall one is fitted in)."""
        x0, y0, x1, y1 = self._slot()
        fw, fh = self.info.frame_size if self.info.frame_size else (4, 3)
        if frame is not None:
            fh, fw = frame.shape[:2]
        iw = x1 - x0
        ih = round(iw * fh / fw)
        if ih > y1 - y0:
            ih = y1 - y0
            iw = round(ih * fw / fh)
            x0 += (x1 - x0 - iw) // 2
        return x0, y0, x0 + iw, y0 + ih

    def _draw_inset(self, img, rec, frame, state, jammed: bool, box: Box, linked: bool) -> tuple[bool, bool]:
        """The frame the drone saw, under its header. Returns whether the file is missing and whether that reads
        NO SIGNAL."""
        x0, y0, x1, y1 = box
        sx0, _, sx1, _ = self._slot()
        self._label(img, "CAMERA", sx0, self._n(26))
        self._text(img, f"frame {rec.idx}", sx1, self._n(24), 20, INK, align="r")
        roi = img[y0:y1, x0:x1]
        lost = state == "LOST"
        stamp, colour = None, BAD
        if frame is None:
            roi[:] = SLOT
            stamp, colour = ("NO SIGNAL", BAD) if lost else ("no picture", MUTED)
        else:
            roi[:] = cv2.resize(frame, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
            if jammed or lost:                           # the picture is there but tells the tracker nothing
                cv2.convertScaleAbs(roi, dst=roi, alpha=0.38)
                stamp = "CAMERA JAMMED" if jammed else "NO MATCH"
        if stamp:
            big = colour is BAD
            self._text(img, stamp, (x0 + x1) / 2, (y0 + y1) / 2 - self._n(13 if big else 8), 34 if big else 22,
                       colour, bold=big, align="c", tracking=0.04 if big else 0.0, halo=frame is not None,
                       fit=0.86 * (x1 - x0))
        edge = YELLOW if linked else FAINT               # the picture wears its footprint's colour when tied to it
        b = self._n(2)
        cv2.rectangle(img, (x0 - b, y0 - b), (x1 + b - 1, y1 + b - 1), edge, b)
        return frame is None, lost and frame is None

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

    def _draw_match_lines(self, img, f, pairs, inset_box: Box, frame: np.ndarray) -> int:
        """The run's own match points: each from its place in the camera picture to its place on the map."""
        x0, y0, x1, y1 = inset_box
        fh, fw = frame.shape[:2]
        ends = [((x0 + fx * (x1 - x0) / fw, y0 + fy * (y1 - y0) / fh), self._to_video(f, mx, my))
                for (fx, fy), (mx, my) in zip(*pairs, strict=True)]

        def lines(over, off):
            for a, b in ends:
                cv2.line(over, px(a[0] - off[0], a[1] - off[1]), px(b[0] - off[0], b[1] - off[1]), MATCH,
                         self._n(1.2), cv2.LINE_AA, SHIFT)

        xy = np.array(ends).reshape(-1, 2)                    # one blend for all of them: crossings do not add up
        self._fill_shape(img, lines, (*xy.min(axis=0), *xy.max(axis=0)), 2, 0.36)
        r = self._n(4.5, 3) << (SHIFT - 1)
        for a, b in ends:
            for p in (a, b):
                cv2.circle(img, px(*p), r + (1 << SHIFT), DARK, -1, cv2.LINE_AA, SHIFT)
                cv2.circle(img, px(*p), r, MATCH, -1, cv2.LINE_AA, SHIFT)
        return len(ends)

    def _draw_side(self, img, f, rec, state, note, conf, inliers, rejected, sigma, inset_box) -> Box:
        """State lamp and its line, then the numbers. Returns the lamp's box."""
        n = self._n
        x0 = self.lay.side[0] + n(PAD)
        x1 = self.lay.side[2] - n(PAD)
        colour = STATE_COLOURS[state]

        # the lamp: a glow, a dot, the state's name
        r = n(11)
        cx, cy = x0 + r, n(442)
        glow = np.zeros((6 * r + 1, 6 * r + 1), np.uint8)
        cv2.circle(glow, (3 * r, 3 * r), round(1.5 * r), 255, -1, cv2.LINE_AA)
        _stamp(img, cv2.GaussianBlur(glow, (0, 0), 0.8 * r), cx - 3 * r, cy - 3 * r, colour, 0.5)
        cv2.circle(img, px(cx + 0.5, cy + 0.5), r << SHIFT, colour, -1, cv2.LINE_AA, SHIFT)
        self._text(img, state, x0 + 2 * r + n(16), cy - n(12), 32, colour, bold=True, tracking=0.03)
        self._text(img, note, x0, n(474), 20, INK if state == "TRACKING" else colour, fit=x1 - x0)
        self._hairline(img, (x0, n(512), x1, n(512) + 1))

        # two columns of numbers
        col = x0 + (x1 - x0 + n(24)) // 2
        room = col - n(24) - x0
        self._label(img, "CONFIDENCE", x0, n(530))
        self._text(img, "—" if conf is None else f"{100 * conf:.0f} %", x0, n(556), 44,
                   MUTED if conf is None else INK, bold=True)
        bar = (x0, n(604), x0 + room, n(604) + n(6, 2))
        cv2.rectangle(img, bar[:2], (bar[2] - 1, bar[3] - 1), FAINT, -1)
        if conf is not None:
            cv2.rectangle(img, bar[:2], (bar[0] + max(1, round(room * conf)) - 1, bar[3] - 1),
                          confidence_colour(conf), -1)

        self._label(img, "MATCHED POINTS", col, n(530))
        if inliers is None:
            self._text(img, "—", col, n(556), 44, MUTED, bold=True)
        else:
            self._text(img, _count(inliers), col, n(556), 44, BAD if rejected else INK, bold=True)
            if rejected:
                self._text(img, "rejected", col, n(600), 20, BAD)

        self._label(img, "UNCERTAINTY", x0, n(638))
        # The filter reports a fraction of a pixel right after a fix; "± 0 px" would read as a claim of no error.
        amount = "< 1 px" if sigma < 1 else f"± {_count(round(sigma))} px"
        self._text(img, amount, x0, n(664), 44, INK, bold=True, fit=room)
        word = uncertainty_word(sigma)
        self._text(img, word, x0, n(710), 20, OK if "precise" in word else WARN if word == "drifting" else BAD)

        self._label(img, "HEADING", col, n(638))
        self._text(img, f"{round(f.heading_deg) % 360:03d}°", col, n(664), 44, INK, bold=True)
        self._compass(img, x1 - n(44), n(680), n(42), f.heading_deg)
        self._hairline(img, (x0, n(748), x1, n(748) + 1))
        return cx - r, cy - r, cx + r, cy + r

    def _compass(self, img, cx: int, cy: int, r: int, heading_deg: float) -> None:
        """A dial with north up and a two-tone needle along the heading."""
        cv2.circle(img, px(cx, cy), r << SHIFT, FAINT, self._n(2), cv2.LINE_AA, SHIFT)
        for a in (90, 180, 270):
            v = np.array([math.sin(math.radians(a)), -math.cos(math.radians(a))])
            cv2.line(img, px(*(np.array([cx, cy]) + v * (r - self._n(8)))), px(*(np.array([cx, cy]) + v * r)), FAINT,
                     self._n(2), cv2.LINE_AA, SHIFT)
        self._text(img, "N", cx, cy - r - self._n(22), MIN_TEXT, MUTED, bold=True, align="c")
        a = math.radians(heading_deg)
        fwd, right = np.array([math.sin(a), -math.cos(a)]), np.array([math.cos(a), math.sin(a)])
        c = np.array([cx, cy], float)
        tip, tail, half = c + fwd * 0.8 * r, c - fwd * 0.55 * r, right * 0.2 * r
        cv2.fillPoly(img, [self._sub([tail, c + half, c - half])], FAINT, cv2.LINE_AA, SHIFT)
        cv2.fillPoly(img, [self._sub([tip, c + half, c - half])], INK, cv2.LINE_AA, SHIFT)

    # -- locator
    def _build_locator(self, layer: np.ndarray) -> None:
        """The whole map, small: where the flight is on it."""
        n = self._n
        mw, mh = self.info.map_size
        room = n(262)
        self.kt = min(room / mw, room / mh)              # plate px per map px
        tw, th = max(2, round(mw * self.kt)), max(2, round(mh * self.kt))
        self._thumb = cv2.convertScaleAbs(cv2.resize(layer, (tw, th), interpolation=cv2.INTER_AREA), alpha=0.8)
        self._plate = (self.lay.side[0] + n(PAD), n(796))

    def _draw_locator(self, img, f, i: int, state: str, cov) -> None:
        n = self._n
        x0, y0 = self._plate
        th, tw = self._thumb.shape[:2]
        self._label(img, "POSITION ON THE MAP", x0, n(766))
        if y0 + th > img.shape[0] or x0 + tw > img.shape[1]:
            return
        plate = img[y0:y0 + th, x0:x0 + tw]              # a view: OpenCV clips what is drawn to the plate
        plate[:] = self._thumb
        sub = lambda p: self._sub(np.asarray(p, float) * self.kt)
        path = np.vstack([self.pts[:i + 1], [[f.x, f.y]]])
        cv2.polylines(plate, [sub(path)], False, DARK, n(2) + 2, cv2.LINE_AA, SHIFT)
        cv2.polylines(plate, [sub(path)], False, INK, n(2), cv2.LINE_AA, SHIFT)
        sx0, sy0, sx1, sy1 = self.lay.stage              # what the stage shows now
        hw, hh = (sx1 - sx0) / 2 / self.s, (sy1 - sy0) / 2 / self.s
        view = sub([(f.cx - hw, f.cy - hh), (f.cx + hw, f.cy - hh), (f.cx + hw, f.cy + hh), (f.cx - hw, f.cy + hh)])
        cv2.polylines(plate, [view], True, YELLOW, n(1.5), cv2.LINE_AA, SHIFT)
        a, b, angle = sigma_ellipse(cov)
        if a * self.kt >= 4:                             # the uncertainty, once it shows at this scale
            c = (f.x * self.kt, f.y * self.kt)
            cv2.polylines(plate, [self._ellipse_poly(c, a * self.kt, b * self.kt, angle)], True, SIGMA, n(2),
                          cv2.LINE_AA, SHIFT)
        c = px(f.x * self.kt, f.y * self.kt)
        cv2.circle(plate, c, n(7, 3) << SHIFT, DARK, -1, cv2.LINE_AA, SHIFT)
        cv2.circle(plate, c, n(5, 2) << SHIFT, STATE_COLOURS[state], -1, cv2.LINE_AA, SHIFT)

        # the key to the colours, beside the plate
        lx = x0 + self._n(262) + n(22)
        if lx + n(150) > self.lay.side[2]:
            return
        rows = [("sure fix", GREEN, "line"), ("weak fix", YELLOW, "line"), ("no fix", RED, "dash"),
                ("3σ area", SIGMA, "ring"), ("camera view", YELLOW, "box")]
        for j, (text, colour, kind) in enumerate(rows):
            cy = y0 + n(12) + j * n(42)
            a, b = (lx, cy), (lx + n(30), cy)
            if kind == "line":
                cv2.line(img, px(*a), px(*b), colour, n(5, 2), cv2.LINE_AA, SHIFT)
            elif kind == "dash":
                dashed(img, a, b, colour, n(5, 2), n(9), n(7))
            elif kind == "ring":
                cv2.ellipse(img, px(lx + n(15), cy), (n(15) << SHIFT, n(9) << SHIFT), 0, 0, 360, colour, n(2),
                            cv2.LINE_AA, SHIFT)
            else:
                cv2.rectangle(img, (lx + n(2), cy - n(9)), (lx + n(28), cy + n(9)), colour, n(2), cv2.LINE_AA)
            self._text(img, text, lx + n(42), cy - n(7), MIN_TEXT, INK)

    # -- dock
    def _build_timeline(self) -> None:
        """The whole run as one strip: a bar per moment (height: matched points, colour: confidence) over a band
        of the states. Built once, bright and dimmed; a frame shows the bright part up to now."""
        n = self._n
        x0 = self.lay.dock[0] + n(292)
        x1 = self.lay.dock[2] - n(PAD)
        w = max(2, x1 - x0)
        chart, band, gap = n(58), n(8, 2), n(8)
        ts = self.t[0] + (np.arange(w) + 0.5) / w * (self.t[-1] - self.t[0])
        rec = np.clip(np.searchsorted(self.t, ts, side="right") - 1, 0, len(self.t) - 1)
        inl = np.array([0 if m else int(r.track.fix.inliers) for r, m in zip(self.records, self.missed, strict=True)])
        top = float(np.percentile(inl[inl > 0], 95)) if (inl > 0).any() else 1.0
        height = np.where(inl > 0, np.maximum(np.round(np.clip(inl / top, 0, 1) * chart), n(4)), n(4)).astype(int)
        colours = np.array([confidence_colour(c) for c in self.conf], np.float32)
        states = np.array([STATE_COLOURS[r.track.state.value] for r in self.records], np.float32)
        strip = np.zeros((chart + gap + band, w, 3), np.float32)
        mask = np.zeros((chart + gap + band, w), bool)
        mask[:chart] = np.arange(chart)[:, None] >= chart - height[rec][None]
        strip[:chart] = colours[rec][None]
        mask[chart + gap:] = True
        strip[chart + gap:] = states[rec][None]
        self._tl_box = (x0, self.lay.dock[1] + n(26), x1)
        self._tl_mask = mask[..., None]
        self._tl_bright = strip.astype(np.uint8)
        self._tl_dim = (strip * 0.3 + np.array(GLASS, np.float32) * 0.7).astype(np.uint8)

    def _draw_dock(self, img, f, i: int) -> None:
        n = self._n
        x, y0 = self.lay.dock[0] + n(PAD), self.lay.dock[1]
        clock = lambda t: f"{int(t) // 60:02d}:{int(t) % 60:02d}"
        w = self._text(img, clock(f.t - self.t[0]), x, y0 + n(30), 44, INK, bold=True)
        self._text(img, f"/ {clock(self.t[-1] - self.t[0])}", x + w + n(10), y0 + n(46), 22, MUTED)
        speed = f"{self.plan.speed:g}"
        self._text(img, "real time" if speed == "1" else f"flight time · {speed}× speed", x, y0 + n(88), MIN_TEXT,
                   MUTED)

        tx0, ty0, _ = self._tl_box
        h, w = self._tl_mask.shape[:2]
        if ty0 + h > img.shape[0]:
            return
        span = self.t[-1] - self.t[0]
        now = round((f.t - self.t[0]) / span * (w - 1)) if span > 0 else w - 1
        roi = img[ty0:ty0 + h, tx0:tx0 + w]
        np.copyto(roi[:, :now + 1], self._tl_bright[:, :now + 1], where=self._tl_mask[:, :now + 1])
        np.copyto(roi[:, now + 1:], self._tl_dim[:, now + 1:], where=self._tl_mask[:, now + 1:])
        cv2.rectangle(img, (tx0 + now - n(1), ty0 - n(8)), (tx0 + now + n(1), ty0 + h + n(8) - 1), INK, -1)

    # -- caption
    def _draw_caption(self, img, k: int, d) -> tuple[str | None, Box | None]:
        """The latest event: a plate low in the stage (high, if the drone is low), a lamp in the event's state
        colour, the title and the note."""
        c = _caption(self.plan, k)
        if c is None:
            return None, None
        cap, alpha = c
        n = self._n
        sx0, sy0, sx1, sy1 = self.lay.stage
        pad, dot = n(26), n(9)
        px_title, px_note = max(6, round(34 * self.u)), max(6, round(22 * self.u))
        text_w = max(self.type.width(cap.title, px_title, True), self.type.width(cap.note, px_note))
        room = (sx1 - sx0) - 2 * n(40) - 2 * pad - 2 * dot - n(18)
        text_w = min(text_w, room)
        bw, bh = math.ceil(text_w) + 2 * pad + 2 * dot + n(18), n(108)
        x0 = sx0 + (sx1 - sx0 - bw) // 2
        low = self._to_video(self.plan.frames[cap.frame], self.plan.frames[cap.frame].x,
                             self.plan.frames[cap.frame].y)[1] <= sy0 + 0.6 * (sy1 - sy0)
        y0 = sy1 - n(PAD) - bh if low else sy0 + n(PAD)
        box = (x0, y0, x0 + bw, y0 + bh)
        before = img[y0:y0 + bh, x0:x0 + bw].copy() if alpha < 1 else None
        self._glass(img, box, 0.86)
        colour = STATE_COLOURS[cap.state]
        cv2.circle(img, px(x0 + pad + dot + 0.5, y0 + n(38) + 0.5), dot << SHIFT, colour, -1, cv2.LINE_AA, SHIFT)
        tx = x0 + pad + 2 * dot + n(18)
        self._text(img, cap.title, tx, y0 + n(26), 34, INK, bold=True, fit=text_w + 1)
        self._text(img, cap.note, tx, y0 + n(70), 22, MUTED, fit=text_w + 1)
        if before is not None:
            roi = img[y0:y0 + bh, x0:x0 + bw]
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
