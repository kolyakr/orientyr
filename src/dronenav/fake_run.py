"""Fake run: a scripted flight over a real map, written as a run folder, so renderers and the UI can be built
before the engine exists (M0-04).

The drone flies a smooth sine route across the map with the camera looking straight down. The frames are the
map seen through the true homography, so a real matcher could localise them. A jam is scripted in the middle:

    TRACKING ... → LOST × jam_len → RELOCALIZING × 2 → TRACKING ...
    (window       (camera jammed:    (global search:     (window
     search,       noise frames, no    one rejected fix,   search)
     accepted      fix; coasting on a  then one accepted)
     fixes)        slightly wrong
                   velocity, so the
                   position drifts
                   off; σ grows)

Fixes are the true pose plus a little noise. ``extra`` carries ``truth`` (x, y, heading_deg, scale),
``search`` (window / global / none) and an ``event`` caption on each state change. Defaults copy the task
data (experiments/data_probe): 50 frames of 640 × 640, 0.2535 map px per frame px, 120–384 inliers, 1 fps.

The route runs along the map's long side. Everything comes from one seeded generator, so the same settings
give byte-identical output (on the same machine and library versions: PNG encoding may differ elsewhere).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dronenav.telemetry import Record, RunInfo, RunWriter, write_run_info
from dronenav.types import Fix, TrackMode, TrackState

__all__ = ["FakeRunConfig", "make_fake_run"]

RELOC_FRAMES = 2            # global search: one miss, then the hit
INLIERS = (120, 384)        # the range of the 50 task crops (experiments/data_probe/README.md)
SCALE_WOBBLE = 0.04         # true scale varies ±4 % along the route (altitude changes)
COAST_BIAS_DEG = 6.0        # the filter's velocity is this far off when the jam starts: visible drift on any route


@dataclass(frozen=True)
class FakeRunConfig:
    """Settings of a fake run. jam_start=None puts the jam 40 % into the run."""

    frames: int = 50
    fps: float = 1.0
    seed: int = 0
    frame_size: tuple[int, int] = (640, 640)
    scale: float = 0.2535   # map px per frame px
    jam_start: int | None = None
    jam_len: int = 5


@dataclass(frozen=True)
class _Pose:
    x: float
    y: float
    heading_deg: float
    scale: float


def make_fake_run(map_path: str | Path, out_dir: str | Path, cfg: FakeRunConfig = FakeRunConfig()) -> RunInfo:
    """Write ``run.json``, ``run.jsonl`` and ``frames/`` into ``out_dir``; returns the RunInfo written.

    Raises FileNotFoundError for an unreadable map and ValueError for settings that cannot make a run.
    """
    map_path, out_dir = Path(map_path), Path(out_dir)
    map_img = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
    if map_img is None:
        raise FileNotFoundError(f"cannot read the map {map_path}")
    map_h, map_w = map_img.shape[:2]
    jam_start = _check(cfg, map_w, map_h)
    jam_end = jam_start + cfg.jam_len                   # first frame after the jam (relocalizing)
    rng = np.random.default_rng(cfg.seed)
    fw, fh = cfg.frame_size

    route = _route(cfg, map_w, map_h, rng)
    inlier_curve = _inlier_curve(cfg.frames, rng)
    info = RunInfo(map=str(map_path), map_size=(map_w, map_h), generator="fake_run",
                   frame_size=(fw, fh), fps=float(cfg.fps), seed=cfg.seed)
    write_run_info(out_dir, info)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):          # a previous run in the same folder: drop its frames only
        if re.fullmatch(r"frame_\d{5}\.png", old.name):
            old.unlink()

    last_x = last_y = vx = vy = 0.0
    coast = np.zeros(2)                                 # the velocity the filter coasts on during the jam
    prev_state: TrackMode | None = None
    with RunWriter(out_dir / "run.jsonl") as writer:
        for i in range(cfg.frames):
            truth = route[i]
            t = i / cfg.fps
            source = f"frames/frame_{i:05d}.png"
            extra = {"truth": {"x": truth.x, "y": truth.y, "heading_deg": truth.heading_deg, "scale": truth.scale}}

            if jam_start <= i < jam_end:
                # camera jammed: noise frames, no homography; coast on the last velocity, σ grows every frame
                state, fix, search, ms = TrackMode.LOST, None, "none", rng.uniform(3, 6)
                if i == jam_start:
                    coast = _coast_velocity(math.hypot(vx, vy), (last_x, last_y), route, jam_start, cfg)
                    vx, vy = coast
                k = i - jam_start + 1
                x, y = last_x + coast[0] * k / cfg.fps, last_y + coast[1] * k / cfg.fps
                cov = _cov(6.0 + 10.0 * k)
                image = _jammed_frame(fw, fh, rng)
            elif i == jam_end:
                # camera back, global search finds a false match somewhere else: rejected, still coasting
                state, search, ms = TrackMode.RELOCALIZING, "global", rng.uniform(230, 260)
                fix = _rejected_fix(i, cfg, map_w, map_h, rng)
                k = i - jam_start + 1
                x, y = last_x + coast[0] * k / cfg.fps, last_y + coast[1] * k / cfg.fps
                cov = _cov(6.0 + 10.0 * k)
                image = _render(map_img, truth, cfg.frame_size)
            elif i == jam_end + 1:
                # the global search hits: the filter restarts at the fix, velocity unknown, σ still wide
                state, search, ms = TrackMode.RELOCALIZING, "global", rng.uniform(230, 260)
                fix = _accepted_fix(i, truth, inlier_curve[i], cfg, rng)
                x, y, vx, vy = fix.x, fix.y, 0.0, 0.0
                cov = _cov(5.0)
                image = _render(map_img, truth, cfg.frame_size)
            else:
                state, search, ms = TrackMode.TRACKING, "window", rng.uniform(18, 40)
                fix = _accepted_fix(i, truth, inlier_curve[i], cfg, rng)
                x, y = truth.x + rng.normal(0, 0.8), truth.y + rng.normal(0, 0.8)   # the filter beats one fix
                v_true = _velocity(route, i, cfg.fps)
                vx, vy = v_true + rng.normal(0, 0.05 * (np.hypot(*v_true) + 1), 2)
                cov = _cov(300.0 / fix.inliers)         # ≈ 0.8–2.5 px: more inliers, tighter
                image = _render(map_img, truth, cfg.frame_size)
                last_x, last_y = x, y

            extra["search"] = search
            event = _event(state, prev_state)
            if event:
                extra["event"] = event
            if not cv2.imwrite(str(out_dir / source), image):   # OpenCV only logs a warning on failure
                raise OSError(f"cannot write the frame {out_dir / source}")
            track = TrackState(idx=i, state=state, x=x, y=y, vx=float(vx), vy=float(vy), cov=cov, fix=fix)
            writer.write(Record(idx=i, t=t, source=source, track=track, backend="fake", ms=round(float(ms), 2),
                                extra=extra))
            prev_state = state
    return info


# --- settings --------------------------------------------------------------------------------------------------

def _footprint_radius(cfg: FakeRunConfig) -> float:
    """Half the frame diagonal in map px at the largest true scale: the footprint never reaches further."""
    return cfg.scale * (1 + SCALE_WOBBLE) * math.hypot(*cfg.frame_size) / 2


def _margin(cfg: FakeRunConfig) -> float:
    # + room for fix noise (centre ~1 px, scale 0.5 %), so noisy footprints stay inside the map too
    return _footprint_radius(cfg) * 1.02 + 8.0


def _check(cfg: FakeRunConfig, map_w: int, map_h: int) -> int:
    """Validate the settings; returns the jam start frame."""
    if cfg.frames < 1 or cfg.fps <= 0 or cfg.scale <= 0 or min(cfg.frame_size) < 1:
        raise ValueError(f"frames, fps, scale and frame size must be positive, got {cfg}")
    if cfg.jam_len < 1:
        raise ValueError(f"the jam needs at least one frame, got jam_len={cfg.jam_len}")
    jam_start = round(0.4 * cfg.frames) if cfg.jam_start is None else cfg.jam_start
    if jam_start < 1:
        raise ValueError(f"the jam must start after at least one tracked frame, got jam_start={jam_start}")
    if jam_start + cfg.jam_len + RELOC_FRAMES >= cfg.frames:
        raise ValueError(f"the jam (frames {jam_start}..{jam_start + cfg.jam_len - 1}) plus {RELOC_FRAMES} "
                         f"relocalizing frames leaves no frame to track again in a run of {cfg.frames}")
    m = _margin(cfg)
    if map_w <= 2 * m + 1 or map_h <= 2 * m + 1:
        raise ValueError(f"the map ({map_w} x {map_h} px) is too small for {cfg.frame_size[0]} x "
                         f"{cfg.frame_size[1]} frames at scale {cfg.scale:g}: each footprint needs "
                         f"{2 * m:.0f} px of map")
    return jam_start


# --- route and poses -------------------------------------------------------------------------------------------

def _heading(dx: float, dy: float) -> float:
    """Heading of a map-px direction: clockwise from map up (y points down). Kept strictly below 360."""
    h = math.degrees(math.atan2(dx, -dy)) % 360.0
    return 0.0 if h >= 360.0 else h     # a tiny negative angle would otherwise give exactly 360.0 (issue #86)


def _route(cfg: FakeRunConfig, map_w: int, map_h: int, rng: np.random.Generator) -> list[_Pose]:
    """A sine along the map's long side, like the task route, with a random phase, size and direction.

    Built along a "long" axis a and a "short" axis b, then mapped to x, y (swapped on a portrait map).
    """
    portrait = map_h > map_w
    long_, short = (map_h, map_w) if portrait else (map_w, map_h)
    m = _margin(cfg)
    span = long_ - 2 * m
    amp = (short / 2 - m) * rng.uniform(0.6, 0.9)
    cycles = rng.uniform(0.7, 1.2)
    phase = rng.uniform(0, 2 * math.pi)
    wobble_phase = rng.uniform(0, 2 * math.pi)
    sign = 1.0 if rng.random() < 0.5 else -1.0          # left to right, or right to left
    poses = []
    for i in range(cfg.frames):
        u = i / (cfg.frames - 1) if cfg.frames > 1 else 0.5
        a = long_ / 2 + sign * span * (u - 0.5)
        b = short / 2 + amp * math.sin(2 * math.pi * cycles * u + phase)
        da = sign * span                                 # d/du
        db = amp * 2 * math.pi * cycles * math.cos(2 * math.pi * cycles * u + phase)
        x, y, dx, dy = (b, a, db, da) if portrait else (a, b, da, db)
        scale = cfg.scale * (1 + SCALE_WOBBLE * math.sin(math.pi * u + wobble_phase))
        poses.append(_Pose(x, y, _heading(dx, dy), scale))
    return poses


def _velocity(route: list[_Pose], i: int, fps: float) -> np.ndarray:
    """Map px per second at frame i, by central difference (one-sided at the ends)."""
    a, b = max(i - 1, 0), min(i + 1, len(route) - 1)
    if a == b:
        return np.zeros(2)
    return np.array([route[b].x - route[a].x, route[b].y - route[a].y]) * fps / (b - a)


def _coast_velocity(speed: float, start: tuple[float, float], route: list[_Pose], jam_start: int,
                    cfg: FakeRunConfig) -> np.ndarray:
    """The velocity the filter coasts on during the jam: the estimated speed, COAST_BIAS_DEG off the true heading.

    Of the two directions (bias left or right), the one that leaves the route faster is used, so the drift
    shows on any stretch: coasting exactly straight would hide it where the route is straight, and a fixed
    bias can be cancelled by the route turning the other way.
    """
    h0 = route[jam_start - 1].heading_deg
    best, best_drift = np.zeros(2), -math.inf
    for sign in (1.0, -1.0):
        a = math.radians(h0 + sign * COAST_BIAS_DEG)
        v = speed * np.array([math.sin(a), -math.cos(a)])          # heading → map px direction (y down)
        err = [math.hypot(start[0] + v[0] * k / cfg.fps - route[jam_start - 1 + k].x,
                          start[1] + v[1] * k / cfg.fps - route[jam_start - 1 + k].y)
               for k in range(1, cfg.jam_len + 1)]
        if err[-1] - err[0] > best_drift:
            best, best_drift = v, err[-1] - err[0]
    return best


def _homography(pose: _Pose, frame_size: tuple[int, int]) -> np.ndarray:
    """Frame px → map px for a nadir camera (conventions in dronenav.types).

    Heading h is clockwise from map up and y points down, so the frame's up vector (0, -1) must land on
    (sin h, -cos h): R = [[cos h, -sin h], [sin h, cos h]]. The frame centre (w/2, h/2) lands on (x, y).
    """
    w, h = frame_size
    a = math.radians(pose.heading_deg)
    sR = pose.scale * np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    t = np.array([pose.x, pose.y]) - sR @ np.array([w / 2, h / 2])
    return np.array([[sR[0, 0], sR[0, 1], t[0]], [sR[1, 0], sR[1, 1], t[1]], [0.0, 0.0, 1.0]])


def _footprint(H: np.ndarray, frame_size: tuple[int, int]) -> np.ndarray:
    w, h = frame_size
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64)   # TL, TR, BR, BL
    p = corners @ H.T
    return p[:, :2] / p[:, 2:]


# --- fixes, σ, frames ------------------------------------------------------------------------------------------

def _inlier_curve(n: int, rng: np.random.Generator) -> np.ndarray:
    """Inliers along the route: a slow swell (texture changes) plus noise, inside the task's range."""
    u = np.linspace(0, 1, n)
    base = 250 + 110 * np.sin(2 * math.pi * 0.9 * u + rng.uniform(0, 2 * math.pi))
    return np.clip(np.round(base + rng.normal(0, 12, n)), *INLIERS).astype(int)


def _accepted_fix(i: int, truth: _Pose, inliers: int, cfg: FakeRunConfig, rng: np.random.Generator) -> Fix:
    noisy = _Pose(truth.x + rng.normal(0, 1.2), truth.y + rng.normal(0, 1.2),
                  _heading_add(truth.heading_deg, rng.normal(0, 0.4)), truth.scale * (1 + rng.normal(0, 0.005)))
    return _fix(i, noisy, cfg, inliers=int(inliers),
                inlier_ratio=float(np.clip(0.45 + inliers / 800 + rng.normal(0, 0.03), 0.3, 0.95)),
                reproj_px=0.3 + 60 / inliers + abs(rng.normal(0, 0.05)), accepted=True)


def _rejected_fix(i: int, cfg: FakeRunConfig, map_w: int, map_h: int, rng: np.random.Generator) -> Fix:
    """A false match from the global search: somewhere else on the map, few inliers, rejected."""
    m = _margin(cfg)
    wrong = _Pose(rng.uniform(m, map_w - m), rng.uniform(m, map_h - m), _heading_add(rng.uniform(0, 360), 0),
                  cfg.scale * rng.uniform(0.97, 1.0))
    return _fix(i, wrong, cfg, inliers=int(rng.integers(5, 9)), inlier_ratio=float(rng.uniform(0.03, 0.06)),
                reproj_px=float(rng.uniform(5, 9)), accepted=False, reject_reason="too_few_inliers")


def _fix(i: int, pose: _Pose, cfg: FakeRunConfig, **kw) -> Fix:
    H = _homography(pose, cfg.frame_size)
    return Fix(idx=i, H=H, x=pose.x, y=pose.y, heading_deg=pose.heading_deg, scale=pose.scale,
               footprint=_footprint(H, cfg.frame_size), **kw)


def _heading_add(h: float, d: float) -> float:
    return _heading(math.sin(math.radians(h + d)), -math.cos(math.radians(h + d)))


def _cov(sigma: float) -> np.ndarray:
    """A slightly elongated, slightly tilted position covariance with largest σ ≈ ``sigma`` px."""
    return sigma ** 2 * np.array([[1.0, 0.1], [0.1, 0.8]]) / 1.05


def _render(map_img: np.ndarray, pose: _Pose, frame_size: tuple[int, int]) -> np.ndarray:
    """The map as the camera sees it. OpenCV puts pixel centres at integers, our convention at +0.5: shift."""
    H = _homography(pose, frame_size)
    shift = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    unshift = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])
    return cv2.warpPerspective(map_img, unshift @ H @ shift, frame_size,
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)


def _jammed_frame(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """What a jammed camera sends: grey noise with horizontal streaks."""
    noise = rng.normal(110, 35, (h, w, 1)) + rng.normal(0, 25, (h, 1, 1))
    return np.clip(np.repeat(noise, 3, axis=2), 0, 255).astype(np.uint8)


def _event(state: TrackMode, prev: TrackMode | None) -> str | None:
    """A caption for the HUD when the state changes (M7-02)."""
    if prev is None or state is prev:
        return None
    return {TrackMode.LOST: "camera jammed: coasting on the last velocity",
            TrackMode.RELOCALIZING: "camera back: global search",
            TrackMode.TRACKING: "relocalized: tracking again"}[state]
