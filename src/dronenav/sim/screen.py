"""A simulated phone filming the map on a monitor (M2-08): the advanced level's set-up, with exact truth.

The real takes (M2-01) are a phone held over the map shown full-screen. They have blur, tilt, glare and moiré,
but no truth: nobody knows exactly where the phone looked. This module films the same scene in simulation, so the
phone-video work (M2-03 to M2-07) can be built and measured against the exact answer. It is labelled simulated
everywhere (``meta.json``: ``simulated: true``); it does not replace the real takes.

The scene, in millimetres (world x right and y down along the screen, z into it; the screen is the plane z = 0):

- :class:`Screen`: a monitor showing a window of the map (``zoom`` screen px per map px, centred on the map by
  default; outside the map, the viewer's dark background). Around it, a bezel, and beyond that, the room.
- :class:`Camera`: a pinhole phone camera (no lens distortion: a phone's ISP corrects it), at ``distance_mm``
  in front of the screen, looking at a point that follows a smooth route (:func:`~dronenav.sim.cutout.cutout_route`
  on the visible map, at ``speed_mm_s``), turned with the route's heading.
- The truth of a frame is exact: camera px -> screen mm -> screen px -> map px, composed as homographies, with
  pixel centres at +0.5 like the rest of the code (:mod:`dronenav.types`). Records are in the Irpin flight format
  (``truth.jsonl``, :class:`~dronenav.sim.truth.FlightTruth`), so every reader takes a take as a flight.

Rendering is physical where that is cheap: each camera pixel is supersampled over the screen, every sample reads
the screen pixel it hits (a screen pixel is flat) and, with ``grid``, the RGB subpixel stripe inside it. Averaging
the samples is what a sensor does, so the moiré is real aliasing between the two grids, not a painted pattern.

Effects (:data:`EFFECTS`) each take a strength in [0, 1]; 0 is off. Randomness comes from (seed, name) for the
take and (seed, frame) for per-frame noise, so :meth:`ScreenSim.frame` gives the same pixels in any order:

- ``shake``: hand tremor (4 to 11 Hz) and sway (under 1 Hz) in position, distance and all three angles;
- ``tilt``: the phone not parallel to the screen, slowly changing, up to 25 degrees;
- ``grid``: the subpixel stripes (hence moiré); ``glare``: a lamp's mirror reflection plus a veil, which moves
  as the phone moves, as a reflection does; ``edge``: the route may run past the screen, onto the bezel and room;
- ``motion_blur``: sub-exposure samples; ``rolling_shutter``: each row read at its own time (the truth is the
  centre row's moment); ``exposure``: auto-exposure hunting and reacting late to the glare; ``banding``: refresh
  bands rolling across the rows; ``noise``: sensor noise; ``cover``: a finger over the lens (``lens_covered``);
- ``compression``: the H.264 CRF of ``flight.mp4`` (16 to 36); it acts in :func:`write_screen_video` only.
"""
from __future__ import annotations

import json
import math
import numbers
import os
import shutil
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from dronenav.io import _is_rate, load_map
from dronenav.sim.cutout import MAX_TURN_DEG, cutout_route
from dronenav.sim.truth import FlightTruth

__all__ = ["EFFECTS", "PRESETS", "PRESET_SPEED_MM_S", "Camera", "Screen", "ScreenSim", "effects_for",
           "write_screen_video"]

EFFECTS = ("shake", "tilt", "grid", "glare", "edge", "motion_blur", "rolling_shutter", "exposure", "banding",
           "noise", "cover", "compression")

# M2-01's three takes: calm; hard (fast, tilted, glare); jam (calm, then the lens covered for 2 s)
_CALM = {"shake": 0.25, "tilt": 0.15, "grid": 1.0, "glare": 0.2, "edge": 0.0, "motion_blur": 0.2,
         "rolling_shutter": 0.3, "exposure": 0.2, "banding": 0.15, "noise": 0.25, "cover": 0.0, "compression": 0.35}
PRESETS = {
    "calm": dict(_CALM),
    "hard": {"shake": 0.8, "tilt": 0.9, "grid": 1.0, "glare": 0.8, "edge": 0.5, "motion_blur": 0.8,
             "rolling_shutter": 0.8, "exposure": 0.6, "banding": 0.5, "noise": 0.6, "cover": 0.0, "compression": 0.6},
    "jam": {**_CALM, "cover": 2 / 3},
}
PRESET_SPEED_MM_S = {"calm": 40.0, "hard": 90.0, "jam": 40.0}   # how fast the look-at point moves on the screen

TILT_MAX_DEG = 25.0
COVER_MAX_S = 3.0           # a finger over the lens for this long at strength 1
KEY_HZ = 8.0                # route key poses per second (at most 25 degrees of turn between two: 200 deg/s)
_FILL, _ROW = 0.8, 0.88     # subpixel stripe width (of a third of the pixel) and lit height (of the pixel)


@dataclass(frozen=True)
class Screen:
    """size: px; pitch_mm: one pixel's width (0.233 mm: a 27-inch 2560 x 1440 monitor); zoom: screen px per map
    px; center: the map point at the screen's centre, map px (None: the map's centre); bezel_mm: the frame."""

    size: tuple[int, int] = (2560, 1440)
    pitch_mm: float = 0.233
    zoom: float = 1.0
    center: tuple[float, float] | None = None
    bezel_mm: float = 12.0


@dataclass(frozen=True)
class Camera:
    """size: frame px; hfov_deg: horizontal field of view (69 degrees: a phone's main camera); fps."""

    size: tuple[int, int] = (1280, 720)
    hfov_deg: float = 69.0
    fps: float = 30.0

    @property
    def K(self) -> np.ndarray:
        w, h = self.size
        f = (w / 2) / math.tan(math.radians(self.hfov_deg) / 2)
        return np.array([[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]])


def effects_for(preset: str = "calm", overrides: dict[str, float] | None = None) -> dict[str, float]:
    """The preset's strengths with ``overrides`` on top, checked: known names, numbers in [0, 1]."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r} (available: {', '.join(PRESETS)})")
    return _check_effects({**PRESETS[preset], **(overrides or {})})


def _check_effects(effects: dict) -> dict[str, float]:
    unknown = sorted(set(effects) - set(EFFECTS))
    if unknown:
        raise ValueError(f"unknown effect {', '.join(map(repr, unknown))} (available: {', '.join(EFFECTS)})")
    out = {}
    for name in EFFECTS:
        v = effects.get(name, 0.0)
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not 0 <= v <= 1:   # NaN fails too
            raise ValueError(f"effect {name} must be a strength in [0, 1], got {v!r}")
        out[name] = float(v)
    return out


def _rng(seed: int, name: str) -> np.random.Generator:
    """A generator for one part of the take: the same (seed, name) gives the same numbers whatever else is on."""
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


class _Wave:
    """A sum of sinusoids with seeded frequencies and phases, amplitude ~1: tremor, sway, slow drifts."""

    def __init__(self, rng: np.random.Generator, f_lo: float, f_hi: float, n: int = 3):
        self.f = rng.uniform(f_lo, f_hi, n)
        self.p = rng.uniform(0, 2 * math.pi, n)
        self.a = rng.uniform(0.5, 1.0, n)
        self.a /= self.a.sum()

    def __call__(self, t):
        t = np.asarray(t, float)[..., None]
        return (self.a * np.sin(2 * math.pi * self.f * t + self.p)).sum(-1)


def _rot(pitch: float, roll: float, yaw: float) -> np.ndarray:
    """World -> camera. yaw: the heading (clockwise from map up), so frame up points along it on the map; then
    the camera pitches about its x axis and rolls about its y axis."""
    a, p, r = (math.radians(v) for v in (yaw, pitch, roll))
    Rz = np.array([[math.cos(a), math.sin(a), 0], [-math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(r), 0, math.sin(r)], [0, 1, 0], [-math.sin(r), 0, math.cos(r)]])
    return Rx @ Ry @ Rz


class ScreenSim:
    """One simulated take: ``len(sim)`` frames at ``camera.fps``; :meth:`truth`, :meth:`covered`, :meth:`frame`
    for any frame, in any order. ``effects``: strengths by name (missing ones are 0); see the module docstring.

    After construction: ``window_origin`` and ``window_size`` (the map px the screen shows), ``K``, ``fps``.
    Raises ValueError when the settings are bad or the visible map has no room for a route.
    """

    def __init__(self, map_img: np.ndarray, *, seconds: float, seed: int = 0, effects: dict | None = None,
                 screen: Screen = Screen(), camera: Camera = Camera(), distance_mm: float = 100.0,
                 speed_mm_s: float = 40.0, supersample: int = 2):
        if not _is_rate(seconds):
            raise ValueError(f"seconds must be a positive number, got {seconds!r}")
        for name, v in (("distance_mm", distance_mm), ("speed_mm_s", speed_mm_s), ("fps", camera.fps),
                        ("pitch_mm", screen.pitch_mm), ("zoom", screen.zoom)):
            if not _is_rate(v):
                raise ValueError(f"{name} must be a positive number, got {v!r}")
        if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
            raise ValueError(  # noqa: TRY004 - a bad setting, reported like the others (as in cutout_route)
                f"seed must be a whole number, got {seed!r}")
        if not (isinstance(supersample, numbers.Integral) and 1 <= supersample <= 4):
            raise ValueError(f"supersample must be 1 to 4, got {supersample!r}")
        if map_img is None or map_img.ndim != 3 or map_img.shape[2] != 3:
            raise ValueError("the map must be a BGR image")
        self.fx = _check_effects(effects or {})
        self.seed, self.screen, self.camera, self.k = int(seed), screen, camera, int(supersample)
        self.fps, self.K = float(camera.fps), camera.K
        self.n = max(1, round(seconds * self.fps))
        self.distance_mm, self.speed_mm_s = float(distance_mm), float(speed_mm_s)
        self._framebuffer(map_img)
        self._route(map_img.shape[1], map_img.shape[0])
        self._motion()

    # -- the scene
    def _framebuffer(self, map_img: np.ndarray) -> None:
        """What the screen shows: the map window at ``zoom``, the viewer's dark grey outside the map."""
        sw, sh = self.screen.size
        cw, ch = max(1, round(sw / self.screen.zoom)), max(1, round(sh / self.screen.zoom))
        mh, mw = map_img.shape[:2]
        cx, cy = self.screen.center if self.screen.center is not None else (mw / 2, mh / 2)
        x0, y0 = round(cx - cw / 2), round(cy - ch / 2)
        crop = np.full((ch, cw, 3), 30, np.uint8)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(mw, x0 + cw), min(mh, y0 + ch)
        if xb > xa and yb > ya:
            crop[ya - y0:yb - y0, xa - x0:xb - x0] = map_img[ya:yb, xa:xb]
        interp = cv2.INTER_AREA if cw > sw else cv2.INTER_LINEAR
        self.fb = (crop if (cw, ch) == (sw, sh) else cv2.resize(crop, (sw, sh), interpolation=interp))
        self.fb = self.fb.astype(np.float32)
        self.window_origin, self.window_size = (x0, y0), (cw, ch)
        self._visible = (xa, ya, xb, yb)                                    # map px both shown and on the map
        # screen px -> map px, and screen mm -> screen px
        self.A = np.array([[cw / sw, 0, x0], [0, ch / sh, y0], [0, 0, 1.0]])
        self.S = np.diag([1 / self.screen.pitch_mm, 1 / self.screen.pitch_mm, 1.0])

    def _reach_mm(self, tilt_deg: float, distance_mm: float) -> float:
        """How far the view's corners reach from the look-at point, mm, at this tilt in its worst direction."""
        w, h = self.camera.size
        corners = np.float64([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]])
        far = 0.0
        for a in np.radians(np.arange(0, 360, 15)):
            R = _rot(tilt_deg * math.cos(a), tilt_deg * math.sin(a), 0.0)
            C = -distance_mm * (R.T @ [0, 0, 1.0])
            M = self.K @ R @ np.array([[1, 0, -C[0]], [0, 1, -C[1]], [0, 0, -C[2]]])
            q = corners @ np.linalg.inv(M).T
            far = max(far, float(np.hypot(*(q[:, :2] / q[:, 2:]).T).max()))
        return far

    def _route(self, mw: int, mh: int) -> None:
        """Key poses on the visible map. cutout_route keeps the straight-down view inside the box it gets; the box
        is shrunk by what tilt and shake add, as far as the screen has room for that (``view_contained`` says
        whether it had), and ``edge`` widens it on purpose, so the view runs past the screen."""
        w, h = self.camera.size
        px_mm = self.A[0, 0] / self.screen.pitch_mm                         # map px per screen mm
        view_w = 2 * self.distance_mm * math.tan(math.radians(self.camera.hfov_deg) / 2) * px_mm
        self.scale = view_w / w                                             # map px per frame px, straight down
        d_max = self.distance_mm * 1.1                                      # the route's distance varies +-10 %

        def extra_for(tilt_s: float, shake_s: float) -> float:              # what tilt and shake add to the reach
            shake_mm = 4.4 * shake_s                                        # sway + tremor, at most
            tilt = TILT_MAX_DEG * tilt_s + 1.75 * shake_s
            return ((self._reach_mm(tilt, d_max + shake_mm) - self._reach_mm(0.0, d_max)) + shake_mm) * px_mm + 2.0

        step = self.speed_mm_s / KEY_HZ * px_mm
        # what cutout_route needs inside the box (its own rule): the nadir reach on each side, then room to turn
        nadir = self._reach_mm(0.0, d_max) * px_mm + 2.0
        need = 2 * nadir + step / math.sin(math.radians(MAX_TURN_DEG) / 2) + 2 * step + 4.0
        xa, ya, xb, yb = self._visible
        room = (min(xb - xa, yb - ya) - need) / 2
        if room < 0:
            raise ValueError(f"the visible map ({xb - xa} x {yb - ya} map px) is too small for a "
                             f"{view_w:.0f} map px wide view at {self.distance_mm:g} mm moving at "
                             f"{self.speed_mm_s:g} mm/s: zoom in, come closer, slow down, or use a bigger map")
        # the margin is the same whatever tilt and shake are (as much as the full ones need, if there is room), so
        # the route depends on the seed and edge only: changing one effect's strength changes that effect only
        keep = min(extra_for(1.0, 1.0), room)
        self.view_contained = bool(self.fx["edge"] == 0 and extra_for(self.fx["tilt"], self.fx["shake"]) <= keep)
        margin = keep - self.fx["edge"] * 0.8 * view_w                      # edge: let the view out on purpose
        bx0, by0, bx1, by1 = xa + margin, ya + margin, xb - margin, yb - margin
        self.n_key = math.ceil(self.n / self.fps * KEY_HZ) + 4
        try:
            poses = cutout_route((round(bx1 - bx0), round(by1 - by0)), self.n_key, self.seed, frame_size=(w, h),
                                 scale=(self.scale * 0.9, self.scale * 1.1), tilt_max_deg=0.0,
                                 overlap=max(0.0, 1 - step / (w * self.scale)))
        except ValueError as e:
            raise ValueError(f"the visible map has no room for this route: {e}") from None
        self._kx = np.array([p.x + bx0 for p in poses])
        self._ky = np.array([p.y + by0 for p in poses])
        self._kh = np.unwrap(np.radians([p.heading_deg for p in poses]))
        self._kd = np.array([self.distance_mm * p.scale / self.scale for p in poses])

    def _motion(self) -> None:
        r = _rng(self.seed, "shake")
        self._tremor = [_Wave(r, 4, 11) for _ in range(6)]                  # x, y, d, yaw, pitch, roll
        self._sway = [_Wave(r, 0.15, 0.9) for _ in range(6)]
        r = _rng(self.seed, "tilt")
        self._tilt_base = r.uniform(0.3, 0.6, 2) * r.choice([-1, 1], 2)
        self._tilt_wave = [_Wave(r, 0.05, 0.2) for _ in range(2)]
        r = _rng(self.seed, "glare")
        sw, sh = self.screen.size
        p = self.screen.pitch_mm
        self._lamp = np.array([sw * p * r.uniform(0.25, 0.75), -0.1 * sh * p, -800.0])   # above, in front
        r = _rng(self.seed, "exposure")
        self._ae = _Wave(r, 0.25, 0.7)
        r = _rng(self.seed, "banding")
        self._band = (r.uniform(2.0, 3.0), r.uniform(0.8, 1.6), r.uniform(0, 2 * math.pi))  # periods, Hz, phase
        r = _rng(self.seed, "cover")
        dur = min(COVER_MAX_S * self.fx["cover"], self.n / self.fps)
        self._cover = (max(0.0, (self.n / self.fps - dur) * r.uniform(0.35, 0.65)), dur, r.uniform(0, 2 * math.pi))
        self._room = _rng(self.seed, "room").uniform(0, 2 * math.pi, 4)

    # -- poses
    def _pose(self, t: float):
        """(look-at x, y in screen mm, distance mm, heading, pitch, roll in degrees) at time t (s)."""
        u = min(max(t, 0.0), self.n / self.fps) * KEY_HZ + 1.0              # key 0 is only a handle for the spline
        k = min(int(u), self.n_key - 3)
        s = u - k

        def spline(v):                                                      # Catmull-Rom through the key poses
            p0, p1, p2, p3 = v[k - 1], v[k], v[k + 1], v[k + 2]
            return 0.5 * (2 * p1 + (p2 - p0) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s * s
                          + (3 * p1 - p0 - 3 * p2 + p3) * s ** 3)

        x, y = spline(self._kx), spline(self._ky)                           # map px
        d, yaw = spline(self._kd), math.degrees(spline(self._kh))
        sh, ti = self.fx["shake"], self.fx["tilt"]
        jit = [sh * (0.4 * tr(t) + 4.0 * sw(t)) for tr, sw in zip(self._tremor[:3], self._sway[:3], strict=True)]
        rot = [sh * (0.25 * tr(t) + 1.5 * sw(t)) for tr, sw in zip(self._tremor[3:], self._sway[3:], strict=True)]
        pitch = TILT_MAX_DEG * ti * float(np.clip(self._tilt_base[0] + 0.4 * self._tilt_wave[0](t), -1, 1))
        roll = 0.6 * TILT_MAX_DEG * ti * float(np.clip(self._tilt_base[1] + 0.4 * self._tilt_wave[1](t), -1, 1))
        # map px -> screen mm through the window
        sx = (x - self.window_origin[0]) / self.A[0, 0] * self.screen.pitch_mm + float(jit[0])
        sy = (y - self.window_origin[1]) / self.A[1, 1] * self.screen.pitch_mm + float(jit[1])
        return sx, sy, d + float(jit[2]), yaw + float(rot[0]), pitch + float(rot[1]), roll + float(rot[2])

    def _camera(self, t: float):
        """(M: screen mm -> image px, C: the camera centre in mm, pose) at time t."""
        sx, sy, d, yaw, pitch, roll = pose = self._pose(t)
        R = _rot(pitch, roll, yaw)
        C = np.array([sx, sy, 0.0]) - d * (R.T @ [0, 0, 1.0])              # d back along the optical axis
        M = self.K @ R @ np.array([[1, 0, -C[0]], [0, 1, -C[1]], [0, 0, -C[2]]])
        return M, C, pose

    def _H_screen(self, t: float) -> np.ndarray:
        """Frame px -> screen px at time t."""
        H = self.S @ np.linalg.inv(self._camera(t)[0])
        return H / H[2, 2]

    def __len__(self) -> int:
        return self.n

    def truth(self, i: int) -> FlightTruth:
        """Frame i's true pose, at the centre row's moment (the rolling shutter reads other rows earlier/later)."""
        t = i / self.fps
        M, _, (_, _, d, yaw, pitch, roll) = self._camera(t)
        H = self.A @ self.S @ np.linalg.inv(M)
        H = H / H[2, 2]
        w, h = self.camera.size
        corners = np.float64([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]) @ H.T
        c = H @ [w / 2, h / 2, 1.0]
        dt = 0.5 / self.fps
        a, b = self._pose(t - dt), self._pose(t + dt)
        speed = math.hypot(b[0] - a[0], b[1] - a[1]) / (2 * dt) / 1000      # m/s on the screen
        return FlightTruth(i=i, t=t, H=H, center=(c[0] / c[2], c[1] / c[2]), footprint=corners[:, :2] / corners[:, 2:],
                           heading_deg=yaw % 360, altitude_m=d / 1000, pitch_deg=pitch, roll_deg=roll,
                           speed_mps=speed)

    def _cover_amount(self, t: float) -> float:
        start, dur, _ = self._cover
        if dur <= 0:
            return 0.0
        ramp = 0.25                                                         # a finger slides on in a quarter second
        return float(np.clip((t - start) / ramp, 0, 1) * np.clip((start + dur - t) / ramp, 0, 1))

    def covered(self, i: int) -> bool:
        """Is the lens (nearly) covered in frame i? Such a frame shows nothing of the map."""
        return self._cover_amount(i / self.fps) >= 0.9

    # -- pixels
    def _sample(self, t: float, readout: float) -> np.ndarray:
        """One exposure sample at time t: the supersampled image of the screen, bezel and room."""
        w, h = self.camera.size
        k = self.k
        xs = ((np.arange(w * k) + 0.5) / k).astype(np.float32)[None, :]    # supersample centres, frame px
        ys = (np.arange(h * k) + 0.5) / k
        if readout > 0:                                                     # rows at their own times: H per row,
            knots = np.linspace(0, h, 17)                                   # interpolated between 17 exact ones
            Hk = np.stack([self._H_screen(t + (y / h - 0.5) * readout) for y in knots])
            Hr = np.stack([np.interp(ys, knots, Hk[:, a, b]) for a in range(3) for b in range(3)], -1)
            Hr = Hr.reshape(-1, 3, 3)
        else:
            Hr = np.broadcast_to(self._H_screen(t), (len(ys), 3, 3))
        # per row, u = (a x + b) / (c x + d): the row terms in float64, the full grid in float32 (2e-4 px at 4096)
        def row(r: int, col: int) -> np.ndarray:
            return Hr[:, r, col].astype(np.float32)[:, None]

        def off_(r: int) -> np.ndarray:
            return (Hr[:, r, 1] * ys + Hr[:, r, 2]).astype(np.float32)[:, None]

        den = row(2, 0) * xs + off_(2)
        u = (row(0, 0) * xs + off_(0)) / den
        v = (row(1, 0) * xs + off_(1)) / den
        sw, sh = self.screen.size
        iu, iv = np.floor(u), np.floor(v)
        # nearest on whole-pixel coordinates is exact: a sample takes the screen pixel it falls in
        img = cv2.remap(self.fb, iu, iv, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        g = self.fx["grid"]
        if g > 0:                                    # R, G, B stripes left to right; a dark gap between rows
            fu = (u - iu) * 3
            stripe = np.minimum(np.floor(fu), 2)
            lit = ((fu - stripe) < _FILL) & ((v - iv) < _ROW)
            on_, off_gain = np.float32((1 - g) + g * 3 / (_FILL * _ROW)), np.float32(1 - g)
            for ch, s in ((2, 0), (1, 1), (0, 2)):                          # BGR channel <- stripe index
                img[..., ch] *= np.where(lit & (stripe == s), on_, off_gain)
        if iu.min() < 0 or iv.min() < 0 or iu.max() >= sw or iv.max() >= sh:
            off = (iu < 0) | (iu >= sw) | (iv < 0) | (iv >= sh)
            img[off] = self._surround(u[off], v[off])
        return img

    def _surround(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """The bezel (dark plastic) and the room behind it (dim, soft shapes), at screen px outside the screen."""
        sw, sh = self.screen.size
        b = self.screen.bezel_mm / self.screen.pitch_mm
        bezel = (u > -b) & (u < sw + b) & (v > -b) & (v < sh + b)
        p = self._room
        mm_u, mm_v = u * self.screen.pitch_mm, v * self.screen.pitch_mm
        shade = 0.5 + 0.25 * np.sin(mm_u / 90 + p[0]) * np.cos(mm_v / 130 + p[1]) + 0.25 * np.sin((mm_u + mm_v) / 210 + p[2])
        room = np.stack([45 + 20 * shade, 55 + 25 * shade, 70 + 30 * shade], -1)
        return np.where(bezel[:, None], np.float32([20, 20, 22]), room).astype(np.float32)

    def _glare_at(self, t: float) -> tuple[float, float, float] | None:
        """Where the lamp's reflection is in the image (x, y px) and its size (σ px); None if behind the camera."""
        M, C, pose = self._camera(t)
        mirror = self._lamp * [1, 1, -1]                                    # the lamp seen in the screen-mirror
        lam = -C[2] / (mirror[2] - C[2])
        G = C + lam * (mirror - C)                                          # where the line of sight meets the screen
        q = M @ [G[0], G[1], 1.0]
        if q[2] <= 0:
            return None
        sigma_mm = 8 + 12 * self.fx["glare"]
        return q[0] / q[2], q[1] / q[2], sigma_mm * self.K[0, 0] / pose[2]

    def frame(self, i: int) -> np.ndarray:
        """Frame i as the phone records it (before H.264): BGR uint8, ``camera.size``."""
        if not 0 <= i < self.n:
            raise IndexError(f"frame {i} is outside the take (0 to {self.n - 1})")
        fx, t = self.fx, i / self.fps
        w, h = self.camera.size
        n_sub = 1 + math.ceil(4 * fx["motion_blur"])
        exposure = fx["motion_blur"] * 0.8 / self.fps
        readout = fx["rolling_shutter"] * 0.9 / self.fps
        acc = None
        for o in np.linspace(-0.5, 0.5, n_sub) if n_sub > 1 else [0.0]:
            s = self._sample(t + o * exposure, readout)
            acc = s if acc is None else acc + s
        img = acc / n_sub
        img = cv2.GaussianBlur(img, (0, 0), 0.5 * self.k)                   # the lens: about half a camera pixel
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if self.k > 1 else img
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
        if fx["glare"] > 0:
            spot = self._glare_at(t)
            veil = 12.0 * fx["glare"]
            if spot is not None:
                gx, gy, sg = spot
                blob = np.exp(-((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * sg * sg))
                img = img + (230.0 * fx["glare"] * blob + veil)[..., None] * np.float32([0.95, 1.0, 1.0])
            else:
                img = img + veil
        if fx["exposure"] > 0:
            lag = 0.4                                                       # auto-exposure answers late
            seen = self._glare_at(t - lag) if fx["glare"] > 0 else None
            near = 0.0 if seen is None else math.exp(-((seen[0] - w / 2) ** 2 + (seen[1] - h / 2) ** 2)
                                                     / (2 * (0.6 * w) ** 2))
            gain = 1 + 0.25 * fx["exposure"] * float(self._ae(t)) - 0.35 * fx["exposure"] * fx["glare"] * near
            img = img * gain
        if fx["banding"] > 0:
            periods, hz, ph = self._band
            row = 1 - 0.3 * fx["banding"] * (0.5 + 0.5 * np.sin(2 * math.pi * (yy[:, :1] / h * periods + hz * t) + ph))
            img = img * row[..., None]
        img = np.clip(img, 0, 255)
        if fx["noise"] > 0:
            rng = np.random.default_rng([self.seed, zlib.crc32(b"noise"), i])
            sd = fx["noise"] * (1.5 + 0.35 * np.sqrt(img))
            img = np.clip(img + rng.standard_normal(img.shape, dtype=np.float32) * sd, 0, 255)
        c = self._cover_amount(t)
        if c > 0:                                    # a finger slides across: dark red, out of focus
            ang = self._cover[2]
            along = ((xx - w / 2) * math.cos(ang) + (yy - h / 2) * math.sin(ang)) / math.hypot(w, h) + 0.5
            mask = np.clip((c * 1.3 - along) / 0.15, 0, 1)[..., None]
            finger = np.float32([10, 14, 38]) * (0.8 + 0.4 * along[..., None])
            img = img * (1 - mask) + finger * mask
        return np.clip(img + 0.5, 0, 255).astype(np.uint8)


def write_screen_video(map_path: str | Path, out_dir: str | Path, *, preset: str = "calm", seconds: float = 20.0,
                       seed: int = 0, effects: dict[str, float] | None = None, screen: Screen = Screen(),
                       camera: Camera = Camera(), distance_mm: float = 100.0, speed_mm_s: float | None = None,
                       supersample: int = 2) -> Path:
    """Film one take into ``out_dir`` as an Irpin-style flight: flight.mp4, truth.jsonl, meta.json and frames/
    (one keyframe a second, decoded from the video, so they carry its compression). The take is written into a
    sibling folder first and moved in whole; a failure leaves ``out_dir`` as it was."""
    from dronenav.viz.trajectory_video import _encoder  # the M1-08 x264 writer

    fx = effects_for(preset, effects)
    the_map = load_map(map_path)                                            # FileNotFoundError before any writing
    speed = PRESET_SPEED_MM_S[preset] if speed_mm_s is None else speed_mm_s
    sim = ScreenSim(the_map.image, seconds=seconds, seed=seed, effects=fx, screen=screen, camera=camera,
                    distance_mm=distance_mm, speed_mm_s=speed, supersample=supersample)
    out_dir = Path(out_dir)
    if out_dir.exists() and not out_dir.is_dir():
        raise ValueError(f"{out_dir} exists and is not a folder")
    part = out_dir.parent / f".{out_dir.name}.partial"
    shutil.rmtree(part, ignore_errors=True)
    crf = 16 + round(20 * fx["compression"])
    try:
        (part / "frames").mkdir(parents=True)
        with open(part / "truth.jsonl", "w", encoding="utf-8") as f:
            for i in range(len(sim)):
                tr = sim.truth(i)
                f.write(json.dumps({"i": i, "t": tr.t, "H_frame_to_map": tr.H.tolist(),
                                    "center_map_px": list(tr.center), "footprint_map_px": tr.footprint.tolist(),
                                    "heading_deg": tr.heading_deg, "altitude_m": tr.altitude_m,
                                    "pitch_deg": tr.pitch_deg, "roll_deg": tr.roll_deg, "speed_mps": tr.speed_mps,
                                    "lens_covered": sim.covered(i)}) + "\n")
        with _encoder(part / "flight.mp4", sim.fps, camera.size, crf=crf) as write:
            for i in range(len(sim)):
                write(sim.frame(i))
        every = max(1, round(sim.fps))
        cap = cv2.VideoCapture(str(part / "flight.mp4"))
        keys = 0
        try:
            for i in range(len(sim)):
                ok, img = cap.read()
                if not ok:
                    raise RuntimeError(f"the encoded video ended after {i} of {len(sim)} frames")
                if i % every == 0:
                    cv2.imwrite(str(part / "frames" / f"frame_{i:05d}.png"), img)
                    keys += 1
        finally:
            cap.release()
        meta = {"simulated": True, "kind": "phone filming the map on a screen (M2-08 simulation, not a real take)",
                "map": str(map_path), "fps": sim.fps, "frames": len(sim), "duration_s": len(sim) / sim.fps,
                "camera": {"size": list(camera.size), "hfov_deg": camera.hfov_deg, "K": sim.K.tolist(),
                           "dist": [0, 0, 0, 0, 0]},
                "screen": {**asdict(screen), "window_origin_map_px": list(sim.window_origin),
                           "window_size_map_px": list(sim.window_size)},
                "distance_mm": distance_mm, "speed_mm_s": speed, "preset": preset, "effects": fx, "seed": seed,
                "crf": crf, "supersample": supersample, "keyframes": keys, "view_contained": sim.view_contained,
                "truth_note": "H_frame_to_map is exact for the pinhole camera (no distortion), at the centre row's "
                              "moment; lens_covered frames show no map"}
        (part / "meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(out_dir / "frames", ignore_errors=True)
        for name in ("frames", "truth.jsonl", "meta.json", "flight.mp4"):
            os.replace(part / name, out_dir / name)
    finally:
        shutil.rmtree(part, ignore_errors=True)
    return out_dir
