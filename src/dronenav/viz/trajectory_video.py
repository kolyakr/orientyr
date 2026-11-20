"""Trajectory video (M1-08): a camera window that follows the drone over the map.

Each video frame shows the map around the drone at a fixed zoom, the trail so far (solid between fixed points,
dashed around missed ones, as in the trajectory image), the current footprint, the drone with a heading arrow,
an inset of the frame the camera saw, and a HUD with the frame number, time, inliers and tracker state.

Two steps, so the motion can be checked without drawing and drawn without re-planning:

- :func:`plan_video` turns the records into one pose per video frame. Video time is run time / ``speed``, then
  a 1 s hold on the last frame. Between records the position follows a cubic Hermite curve through each
  record's filtered position (tangents from the neighbours, like Catmull-Rom), so the motion is smooth and
  passes exactly through every record at its time. The heading comes from the fixes (turning along the shortest
  arc) and, where there is no fix, from the direction of motion. The zoom is fixed per run: the view spans six
  median footprint diagonals, and the camera keeps the view on the map where it can.
- :class:`FrameRenderer` draws plan frame k and reports what it showed (:class:`FrameInfo`).
  :func:`write_video` does it for every frame and writes an H.264 MP4 (8-bit 4:2:0, index first) that plays in
  QuickTime and browsers.

Encoding goes through the ffmpeg bundled with ``imageio-ffmpeg`` (x264 at CRF 23: a few MB per minute, the same on
every OS). Without that package it falls back to OpenCV's encoder, which ignores quality settings (about 14 Mbit/s
on macOS) and is missing from some Linux builds.
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dronenav.telemetry import Record, RunInfo, read_run, read_run_info
from dronenav.viz.draw import FIXED, FONT, FOOTPRINT, MISSED, OUTLINE, SHIFT, STATE_COLOURS, TEXT, dashed, is_missed, px
from dronenav.viz.mp4 import faststart
from dronenav.viz.trajectory_image import _bgr

__all__ = ["FrameInfo", "FrameRenderer", "VideoFrame", "VideoPlan", "plan_video", "write_video"]

VIEW_FOOTPRINTS = 6.0       # the view spans this many median footprint diagonals
HOLD_S = 1.0                # seconds the last frame is held
CRF = 23                    # x264 quality: lower is better and bigger; 23 is its default
INSET_W = 0.28              # inset width, as a fraction of the video width
MARGIN = 16                 # px around the inset and the HUD
BACKGROUND = (30, 30, 30)   # outside the map
TRAIL = 3                   # trail width, video px
DASH, GAP = 6.0, 8.0


@dataclass(frozen=True, eq=False)
class VideoFrame:
    """The drone and the camera at one video frame. Map px. rec: index of the last record at or before t."""

    t: float
    x: float
    y: float
    heading_deg: float
    rec: int
    footprint: np.ndarray | None     # 4 x 2, TL TR BR BL, or None when there is no fix to show
    cx: float                        # camera centre
    cy: float


@dataclass(frozen=True, eq=False)
class VideoPlan:
    frames: list[VideoFrame]
    fps: float
    size: tuple[int, int]            # video (w, h) px
    view_w: float                    # map px across the video's width


@dataclass(frozen=True)
class FrameInfo:
    """What one drawn frame shows, in video px."""

    idx: int                         # frame number of the current record
    state: str
    inliers: int | None
    drone_px: tuple[float, float]
    footprint: bool
    inset_box: tuple[int, int, int, int]
    hud_box: tuple[int, int, int, int]
    frame_missing: bool


# --- planning --------------------------------------------------------------------------------------------------

def plan_video(records: list[Record], info: RunInfo, *, fps: float = 30.0, speed: float = 1.0,
               size: tuple[int, int] = (1280, 720), hold_s: float = HOLD_S) -> VideoPlan:
    """One pose per video frame; see the module docstring."""
    _check_options(fps, speed, size)
    if not records:
        raise ValueError("the run has no records")
    t = np.array([r.t for r in records])
    p = np.array([[r.track.x, r.track.y] for r in records])
    tang = _tangents(t, p)
    map_w, map_h = info.map_size
    view_w = _view_width(records, map_w, map_h, size)
    half_w, half_h = view_w / 2, view_w * size[1] / size[0] / 2

    n = math.floor((t[-1] - t[0]) * fps / speed + 1e-9) + 1
    times = [t[0] + k * speed / fps for k in range(n)] + [t[-1]] * round(hold_s * fps)
    frames, last_heading = [], None
    for tk in times:
        i = min(int(np.searchsorted(t, tk, side="right")) - 1, len(t) - 1)
        x, y, vx, vy = _hermite(t, p, tang, i, tk)
        heading = _heading(records, t, i, tk, vx, vy, last_heading)
        last_heading = heading
        cx = map_w / 2 if 2 * half_w >= map_w else min(max(x, half_w), map_w - half_w)
        cy = map_h / 2 if 2 * half_h >= map_h else min(max(y, half_h), map_h - half_h)
        frames.append(VideoFrame(t=float(tk), x=x, y=y, heading_deg=heading, rec=i,
                                 footprint=_footprint(records, t, i, tk), cx=cx, cy=cy))
    return VideoPlan(frames=frames, fps=float(fps), size=tuple(size), view_w=view_w)


def _check_options(fps, speed, size) -> None:
    if not fps > 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if not speed > 0:
        raise ValueError(f"speed must be positive, got {speed}")
    w, h = size
    if w < 16 or h < 16 or w % 2 or h % 2:   # H.264 4:2:0 needs even sides
        raise ValueError(f"the video size must be even and at least 16 x 16 px, got {w} x {h}")


def _tangents(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Velocity at each record (map px / s) from its neighbours; one-sided at the ends, zero for a lone record."""
    m = np.zeros_like(p)
    for i in range(len(t)):
        a, b = max(i - 1, 0), min(i + 1, len(t) - 1)
        if t[b] > t[a]:
            m[i] = (p[b] - p[a]) / (t[b] - t[a])
    return m


def _hermite(t, p, m, i, tk) -> tuple[float, float, float, float]:
    """Position and velocity at time tk on the cubic Hermite segment from record i to i + 1."""
    if i >= len(t) - 1 or t[i + 1] <= t[i]:
        return float(p[i][0]), float(p[i][1]), float(m[i][0]), float(m[i][1])
    dt = t[i + 1] - t[i]
    u = (tk - t[i]) / dt
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    pos = h00 * p[i] + h10 * dt * m[i] + h01 * p[i + 1] + h11 * dt * m[i + 1]
    d00, d10, d01, d11 = 6 * u**2 - 6 * u, 3 * u**2 - 4 * u + 1, -6 * u**2 + 6 * u, 3 * u**2 - 2 * u
    vel = (d00 * p[i] + d01 * p[i + 1]) / dt + d10 * m[i] + d11 * m[i + 1]
    return float(pos[0]), float(pos[1]), float(vel[0]), float(vel[1])


def _measured(r: Record) -> bool:
    return not is_missed(r)


def _heading(records, t, i, tk, vx, vy, last) -> float:
    """From the fixes on both sides (shortest arc), else from the motion, else the last heading."""
    j = min(i + 1, len(records) - 1)
    if _measured(records[i]) and (_measured(records[j]) or j == i):
        h0 = records[i].track.fix.heading_deg
        if j == i or t[j] <= t[i]:
            return h0
        u = (tk - t[i]) / (t[j] - t[i])
        d = (records[j].track.fix.heading_deg - h0 + 180) % 360 - 180
        return (h0 + u * d) % 360
    if math.hypot(vx, vy) > 1e-9:
        return math.degrees(math.atan2(vx, -vy)) % 360   # y points down: east is 90
    return last if last is not None else 0.0


def _footprint(records, t, i, tk) -> np.ndarray | None:
    """The footprint between two measured fixes, corner by corner; a record's own at its time; else None."""
    if not _measured(records[i]):
        return None
    a = records[i].track.fix.footprint
    if tk <= t[i] or i == len(records) - 1:
        return a.copy()
    j = i + 1
    if not _measured(records[j]) or t[j] <= t[i]:
        return None
    u = (tk - t[i]) / (t[j] - t[i])
    return (1 - u) * a + u * records[j].track.fix.footprint


def _view_width(records, map_w, map_h, size) -> float:
    """Map px across the video: VIEW_FOOTPRINTS median footprint diagonals (a third of the map without fixes),
    no more than the map."""
    diags = [math.hypot(*(r.track.fix.footprint[2] - r.track.fix.footprint[0])) for r in records if _measured(r)]
    w = VIEW_FOOTPRINTS * float(np.median(diags)) if diags else map_w / 3
    return min(w, map_w, map_h * size[0] / size[1])


# --- drawing ---------------------------------------------------------------------------------------------------

class FrameRenderer:
    """Draws the frames of a plan. The trail is drawn once, onto a copy of the map scaled to video px, as the
    flight passes each record; a frame is then a crop of that layer plus what moves."""

    def __init__(self, run_dir: str | Path, records: list[Record], info: RunInfo, plan: VideoPlan,
                 map_img: np.ndarray | None = None):
        self.run_dir, self.records, self.plan = Path(run_dir), records, plan
        if map_img is None:
            map_img = cv2.imread(str(info.map_path), cv2.IMREAD_COLOR)
            if map_img is None:
                raise FileNotFoundError(f"cannot read the map {info.map_path}")
        map_img = _bgr(map_img)
        h, w = map_img.shape[:2]
        if (w, h) != info.map_size:
            raise ValueError(f"the map is {w} x {h} px but run.json says the run was flown over "
                             f"{info.map_size[0]} x {info.map_size[1]} px ({info.map_path})")
        self.s = plan.size[0] / plan.view_w             # video px per map px
        # zoomed out: shrink the map once (no aliasing, small layer); zoomed in: keep it and enlarge per frame
        self.k = min(self.s, 1.0)                        # layer px per map px
        self.layer = cv2.resize(map_img, None, fx=self.k, fy=self.k, interpolation=cv2.INTER_AREA) \
            if self.k < 1 else map_img.copy()
        self.drawn_upto = 0                              # records whose trail is on the layer
        self.phase = 0.0
        self.pts = np.array([[r.track.x, r.track.y] for r in records])
        self.missed = [is_missed(r) for r in records]
        self._frame_cache: tuple[int, np.ndarray | None] = (-1, None)

    # -- the trail layer
    def _grow_trail(self, upto: int) -> None:
        """Draw the trail up to record ``upto`` onto the layer (the trail only grows, so each piece once)."""
        thick = max(1, round(TRAIL * self.k / self.s))
        while self.drawn_upto < upto:
            i = self.drawn_upto
            a, b = self.pts[i] * self.k, self.pts[i + 1] * self.k
            if self.missed[i] or self.missed[i + 1]:
                self.phase = dashed(self.layer, a, b, MISSED, thick, DASH * self.k / self.s, GAP * self.k / self.s,
                                    self.phase)
            else:
                cv2.line(self.layer, px(*a), px(*b), FIXED, thick, cv2.LINE_AA, SHIFT)
                self.phase = 0.0
            self.drawn_upto += 1

    def _to_video(self, f: VideoFrame, x: float, y: float) -> tuple[float, float]:
        w, h = self.plan.size
        return w / 2 + (x - f.cx) * self.s, h / 2 + (y - f.cy) * self.s

    def draw(self, k: int) -> tuple[np.ndarray, FrameInfo]:
        """Video frame k of the plan (BGR uint8) and what it shows."""
        f = self.plan.frames[k]
        rec = self.records[f.rec]
        if f.rec < self.drawn_upto:                      # drawing out of order: start the layer's trail again
            raise ValueError("frames must be drawn in order")
        self._grow_trail(f.rec)
        w, h = self.plan.size
        z = self.s / self.k                              # layer px -> video px
        # video px (corner origin) -> layer px; OpenCV's pixel centres sit at integers on both sides
        m = np.array([[z, 0, w / 2 - f.cx * self.s], [0, z, h / 2 - f.cy * self.s]])
        m[:, 2] += 0.5 * (z - 1)
        img = cv2.warpAffine(self.layer, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=BACKGROUND)

        # the piece of trail from the last record to the drone
        a = self._to_video(f, *self.pts[f.rec])
        d = self._to_video(f, f.x, f.y)
        nxt = min(f.rec + 1, len(self.records) - 1)
        if self.missed[f.rec] or self.missed[nxt]:
            dashed(img, a, d, MISSED, TRAIL, DASH, GAP, self.phase)
        else:
            cv2.line(img, px(*a), px(*d), FIXED, TRAIL, cv2.LINE_AA, SHIFT)

        if f.footprint is not None:
            poly = np.array([px(*self._to_video(f, *c)) for c in f.footprint], np.int32)
            cv2.polylines(img, [poly], True, FOOTPRINT, 2, cv2.LINE_AA, SHIFT)

        state = rec.track.state.value
        colour = STATE_COLOURS[state]
        a = math.radians(f.heading_deg)
        tip = (d[0] + 30 * math.sin(a), d[1] - 30 * math.cos(a))
        cv2.arrowedLine(img, px(*d), px(*tip), OUTLINE, 5, cv2.LINE_AA, SHIFT, 0.35)
        cv2.arrowedLine(img, px(*d), px(*tip), TEXT, 2, cv2.LINE_AA, SHIFT, 0.35)
        cv2.circle(img, px(*d), 8 << SHIFT, colour, -1, cv2.LINE_AA, SHIFT)
        cv2.circle(img, px(*d), 8 << SHIFT, TEXT, 2, cv2.LINE_AA, SHIFT)

        inset_box, missing = self._draw_inset(img, f.rec, colour)
        fix = rec.track.fix
        inliers = None if fix is None else int(fix.inliers)
        hud_box = self._draw_hud(img, rec, f.t, inliers, fix is not None and not fix.accepted, state, colour)
        return img, FrameInfo(idx=rec.idx, state=state, inliers=inliers, drone_px=d, footprint=f.footprint is not None,
                              inset_box=inset_box, hud_box=hud_box, frame_missing=missing)

    # -- inset and HUD
    def _frame(self, i: int) -> np.ndarray | None:
        if self._frame_cache[0] != i:
            img = cv2.imread(str(self.run_dir / self.records[i].source), cv2.IMREAD_COLOR)
            self._frame_cache = (i, img)
        return self._frame_cache[1]

    def _draw_inset(self, img, i, colour) -> tuple[tuple[int, int, int, int], bool]:
        w, h = self.plan.size
        frame = self._frame(i)
        iw = round(w * INSET_W)
        aspect = frame.shape[0] / frame.shape[1] if frame is not None else 0.75
        ih = min(round(iw * aspect), h - 2 * MARGIN)
        x1, y0 = w - MARGIN, MARGIN
        x0, y1 = x1 - iw, y0 + ih
        if frame is None:
            img[y0:y1, x0:x1] = (40, 40, 40)
            cv2.putText(img, "no frame", (x0 + 12, y0 + ih // 2), FONT, 0.6, (110, 110, 110), 1, cv2.LINE_AA)
        else:
            img[y0:y1, x0:x1] = cv2.resize(frame, (iw, ih), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (x0 - 2, y0 - 2), (x1 + 1, y1 + 1), colour, 3)
        return (x0, y0, x1, y1), frame is None

    def _draw_hud(self, img, rec, t, inliers, rejected, state, colour) -> tuple[int, int, int, int]:
        lines = [f"frame {rec.idx}", f"t {t:6.1f} s",
                 "inliers -" if inliers is None else f"inliers {inliers}" + (" (rejected)" if rejected else "")]
        x0, y0 = MARGIN, MARGIN
        widths = [cv2.getTextSize(s, FONT, 0.6, 1)[0][0] for s in lines + [state]]
        x1, y1 = x0 + max(widths) + 24, y0 + 26 * (len(lines) + 1) + 16
        roi = img[y0:y1, x0:x1]
        roi[:] = (roi * 0.25 + np.array([20, 20, 20]) * 0.75).astype(np.uint8)
        for k, s in enumerate(lines):
            cv2.putText(img, s, (x0 + 12, y0 + 28 + 26 * k), FONT, 0.6, TEXT, 1, cv2.LINE_AA)
        by = y0 + 28 + 26 * len(lines) - 18
        (tw, th), _ = cv2.getTextSize(state, FONT, 0.6, 2)
        cv2.rectangle(img, (x0 + 10, by), (x0 + 22 + tw, by + th + 12), colour, -1)
        cv2.putText(img, state, (x0 + 16, by + th + 6), FONT, 0.6, OUTLINE, 2, cv2.LINE_AA)
        return x0, y0, x1, y1


# --- writing ---------------------------------------------------------------------------------------------------

def write_video(run_dir: str | Path, out_dir: str | Path | None = None, *, fps: float = 30.0, speed: float = 1.0,
                size: tuple[int, int] = (1280, 720), map_img: np.ndarray | None = None) -> Path:
    """Render the run in ``run_dir`` to ``<out_dir>/trajectory.mp4`` (default: the run folder); returns the path."""
    run_dir = Path(run_dir)
    _check_options(fps, speed, size)
    info = read_run_info(run_dir)
    records = read_run(run_dir / "run.jsonl")
    if not records:
        raise ValueError(f"{run_dir / 'run.jsonl'} has no records")
    plan = plan_video(records, info, fps=fps, speed=speed, size=size)
    renderer = FrameRenderer(run_dir, records, info, plan, map_img)
    out_dir = Path(out_dir) if out_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)          # only once the run and map are known to be good
    path = out_dir / "trajectory.mp4"
    with _encoder(path, fps, size) as encode:
        for k in range(len(plan.frames)):
            encode(renderer.draw(k)[0])
    return path


class _encoder:
    """Context manager taking BGR frames: x264 through the bundled ffmpeg, else OpenCV (see the module docstring).
    The file is complete, with its index first, when the block exits without an error."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        self.path, self.fps, self.size = path, fps, size
        self.proc = self.writer = None

    def __enter__(self):
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            exe = None
        w, h = self.size
        if exe:
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                   "-r", f"{self.fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(CRF),
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            return self._pipe
        self.writer = cv2.VideoWriter(str(self.path), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"avc1"), self.fps,
                                      self.size)
        if not self.writer.isOpened():
            raise RuntimeError("cannot write H.264: install imageio-ffmpeg (pip install imageio-ffmpeg)")
        return self.writer.write

    def _pipe(self, frame: np.ndarray) -> None:
        try:
            self.proc.stdin.write(frame.tobytes())
        except BrokenPipeError:                          # ffmpeg has exited: report what it said, not the pipe
            self._fail()

    def _fail(self):
        err = self.proc.stderr.read().decode(errors="replace").strip()
        self.proc.wait()
        raise RuntimeError(f"ffmpeg failed to encode {self.path}: {err[-500:] or 'no message'}")

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.proc is not None:
            if exc_type is not None:                     # our error (or _fail's): stop ffmpeg, keep the error
                self.proc.kill()
                self.proc.wait()
                return
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
            if self.proc.wait() != 0:
                self._fail()
        else:
            self.writer.release()
            if exc_type is None:
                faststart(self.path)
