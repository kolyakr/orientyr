/** What the console draws from the telemetry: pure functions, shared by the components and their tests. */
import type { RunRecord, TrackMode } from "./run";

/** Inliers at which a fix counts as fully trusted: the task crops give 120–384 (experiments/data_probe). */
const FULL_CONFIDENCE_INLIERS = 300;

/** 0 (no accepted fix) … 1 (as many inliers as a clean frame gives). Colours the trail and the HUD bar. */
export function confidence(rec: RunRecord): number {
  if (!rec.fix?.accepted) return 0;
  return Math.min(1, Math.max(rec.fix.inliers, 1) / FULL_CONFIDENCE_INLIERS);
}

/** Position uncertainty in map px: the standard deviation along the covariance's long axis. */
export function sigmaPx(cov: number[][]): number {
  const [[a, b], [, d]] = cov;
  const mean = (a + d) / 2;
  return Math.sqrt(mean + Math.hypot((a - d) / 2, b)); // the larger eigenvalue of [[a, b], [b, d]]
}

/** Heading in degrees, clockwise from map up: the fix's, else the direction of travel, else unknown. */
export function headingOf(rec: RunRecord): number | null {
  if (rec.fix?.accepted) return rec.fix.heading_deg;
  if (!rec.located) return null;
  if (rec.vx === 0 && rec.vy === 0) return null;
  const deg = (Math.atan2(rec.vx, -rec.vy) * 180) / Math.PI; // map y points down, so up is -y
  return ((deg % 360) + 360) % 360;
}

/**
 * The course the drone really flew into `frame`, in degrees clockwise from map up: the direction of the last step
 * that moved it. Unlike the camera's heading it turns around when the drone flies back. Null before the first move.
 */
export function courseOf(records: RunRecord[], frame: number): number | null {
  for (let i = Math.min(frame, records.length - 1); i > 0; i--) {
    if (!records[i].located || !records[i - 1].located) continue; // a borrowed place is not a move
    const dx = records[i].x - records[i - 1].x;
    const dy = records[i].y - records[i - 1].y;
    if (dx !== 0 || dy !== 0) return ((Math.atan2(dx, -dy) * 180) / Math.PI + 360) % 360; // map y points down
  }
  return null;
}

export interface TrailSegment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number; // of the frame the segment arrives at
  coasting: boolean; // arrived without an accepted fix: dead-reckoned, drawn dashed
}

/**
 * The route flown so far: one segment per step, from the first frame up to `frame`. Frames without a position
 * are not on the route: it starts at the first located frame and passes over the others.
 */
export function trail(records: RunRecord[], frame: number): TrailSegment[] {
  const segs: TrailSegment[] = [];
  let a: RunRecord | null = null;
  for (let i = 0; i <= Math.min(frame, records.length - 1); i++) {
    const b = records[i];
    if (!b.located) continue;
    if (a) segs.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, confidence: confidence(b), coasting: !b.fix?.accepted });
    a = b;
  }
  return segs;
}

export interface TrailRun {
  points: number[][]; // map px, in flight order; the first point is the last point of the run before
  color: string; // confidenceColor of its steps
  coasting: boolean;
}

/**
 * The flown route as a few polylines: neighbouring steps of one colour (and one kind, measured or dead-reckoned)
 * are joined. The minimap draws these: at its size one line per step would be hundreds of invisible elements.
 */
export function trailRuns(records: RunRecord[], frame: number): TrailRun[] {
  const runs: TrailRun[] = [];
  for (const s of trail(records, frame)) {
    const color = confidenceColor(s.confidence);
    const last = runs.at(-1);
    if (last && last.color === color && last.coasting === s.coasting) last.points.push([s.x2, s.y2]);
    else runs.push({ points: [[s.x1, s.y1], [s.x2, s.y2]], color, coasting: s.coasting });
  }
  return runs;
}

export interface StateRun {
  state: TrackMode;
  from: number; // first frame (position in the run, not idx)
  to: number; // last frame, inclusive
}

/** Consecutive frames in one state: the coloured bands of the timeline. */
export function stateRuns(records: RunRecord[]): StateRun[] {
  const runs: StateRun[] = [];
  records.forEach((r, i) => {
    const last = runs.at(-1);
    if (last && last.state === r.state) last.to = i;
    else runs.push({ state: r.state, from: i, to: i });
  });
  return runs;
}

export interface RunEvent {
  frame: number; // position in the run
  t: number;
  state: TrackMode;
  text: string;
}

/**
 * The mission log. The engine may attach captions to state changes (`extra.event`); a run that has them is shown
 * as written. The real tracker and the live mode do not write captions, so for those runs the log is read from
 * the records themselves: every change of tracking state, and every change of the GPS check's verdict.
 */
export function events(records: RunRecord[]): RunEvent[] {
  const written = records.flatMap((r, frame) =>
    typeof r.extra.event === "string" ? [{ frame, t: r.t, state: r.state, text: r.extra.event }] : [],
  );
  if (written.length) return written;
  const log: RunEvent[] = [];
  let tracked = false; // the first TRACKING is "found", later ones are "found again"
  records.forEach((r, frame) => {
    const prev = records[frame - 1];
    const add = (text: string) => log.push({ frame, t: r.t, state: r.state, text });
    if (prev && r.state !== prev.state) {
      if (r.state === "LOST") add(r.extra.jammed === true ? "camera jammed: holding the last position" : "signal lost: holding the last position");
      else if (r.state === "RELOCALIZING") add("camera back: searching the whole map");
      else add(tracked ? "relocalized: tracking again" : "position found: tracking");
    }
    if (r.state === "TRACKING") tracked = true;
    const gps = gpsOf(r)?.state;
    const before = prev ? gpsOf(prev)?.state : undefined;
    if (gps && before && gps !== before) {
      if (gps === "SUSPECT") add("GPS drifts away from the camera position");
      else if (gps === "SPOOFED") add("GPS spoofed: alarm raised");
      else if (gps === "OK" && before !== "NO_DATA") add("GPS agrees with the camera again");
    }
  });
  return log;
}

/** The trail's three colours, as the legend says: green = sure, yellow = less sure, red = guessing (no fix). */
export function confidenceColor(c: number): string {
  if (c <= 0) return "#e3261c";
  return c < 0.6 ? "#f5d21f" : "#3ec43e";
}

/**
 * The 3σ ellipse of a 2x2 covariance: radii along its axes (map px) and the long axis's angle in degrees.
 * The drone is inside it with ~99 % probability.
 */
export function sigmaEllipse(cov: number[][]): { rx: number; ry: number; angle: number } {
  const [[a, b], [, d]] = cov;
  const mean = (a + d) / 2;
  const half = Math.hypot((a - d) / 2, b);
  return {
    rx: 3 * Math.sqrt(mean + half),
    ry: 3 * Math.sqrt(Math.max(mean - half, 0)),
    angle: (0.5 * Math.atan2(2 * b, a - d) * 180) / Math.PI,
  };
}

export const STATE_LABEL: Record<TrackMode, string> = {
  TRACKING: "Tracking",
  LOST: "Signal lost",
  RELOCALIZING: "Relocalizing",
};

const COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];

/** The compass point nearest to a heading in degrees (0 = map up = N). */
export function compass(deg: number): string {
  return COMPASS[Math.round((((deg % 360) + 360) % 360) / 45) % 8];
}

/** mm:ss.s of flight time. */
export function clock(t: number): string {
  const m = Math.floor(t / 60);
  return `${String(m).padStart(2, "0")}:${(t - m * 60).toFixed(1).padStart(4, "0")}`;
}

/** A round step (1, 2, 5, 10, 20, 50, …) that puts about `target` marks on `n` frames: point numbers, axis ticks. */
export function niceStep(n: number, target: number): number {
  const raw = Math.max(1, n / target);
  const pow = 10 ** Math.floor(Math.log10(raw));
  return pow * ([1, 2, 5, 10].find((m) => m * pow >= raw) ?? 10);
}

/**
 * An angle for an SVG or CSS transform, rounded to 0.1°. `Math.atan2` may differ in the last digit between Node and
 * the browser: unrounded, the server's HTML and the client's first render disagree (a React hydration warning).
 */
export function drawAngle(deg: number): number {
  return Math.round(deg * 10) / 10;
}

/** A box in px: on the map, or on the screen. */
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Where the map is drawn on the screen: screen px = map px * scale + (x, y). */
export interface Camera {
  scale: number;
  x: number;
  y: number;
}

/** How big one camera footprint is on the map, in map px: the median larger side over the accepted fixes; 0 without any. */
export function footprintSize(records: RunRecord[]): number {
  const sizes = records.flatMap((r) => {
    if (!r.fix?.accepted) return [];
    const xs = r.fix.footprint.map((p) => p[0]);
    const ys = r.fix.footprint.map((p) => p[1]);
    return [Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys))];
  }).sort((a, b) => a - b);
  return sizes.length ? sizes[Math.floor(sizes.length / 2)] : 0;
}

/**
 * The box around the whole route, in map px, with a margin for what is drawn around the points (the footprint,
 * the labels). Frames without a position do not count. The margin never reaches outside the map; a point that drifted off the map still stays inside.
 */
export function routeBox(records: RunRecord[], [w, h]: [number, number], margin: number): Rect {
  const located = records.filter((r) => r.located);
  if (located.length === 0) return { x: 0, y: 0, width: w, height: h }; // never localized: the whole map
  const xs = located.map((r) => r.x);
  const ys = located.map((r) => r.y);
  const side = (lo: number, hi: number, size: number) => {
    const from = Math.min(lo, Math.max(0, lo - margin));
    const to = Math.max(hi, Math.min(size, hi + margin), from + 1); // never empty: the scale must stay finite
    return [from, to - from];
  };
  const [x, width] = side(Math.min(...xs), Math.max(...xs), w);
  const [y, height] = side(Math.min(...ys), Math.max(...ys), h);
  return { x, y, width, height };
}

/**
 * The camera of the whole-route view. `free` is the part of the screen no panel covers, `view` the whole window;
 * both in the same screen px. Three rules, in this order of weight:
 *   1. the route box lies inside `free`, as large as it can be;
 *   2. the map covers `view` where rule 1 allows: it slides under the panels, so no map edge shows;
 *   3. otherwise the route box sits in the middle of `free`.
 */
export function fitRoute(box: Rect, [w, h]: [number, number], free: Rect, view: Rect): Camera {
  const scale = Math.min(free.width / box.width, free.height / box.height);
  const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);
  const along = (b0: number, bLen: number, mapLen: number, f0: number, fLen: number, v0: number, vLen: number) => {
    let t = f0 + (fLen - bLen * scale) / 2 - b0 * scale;                              // rule 3
    if (mapLen * scale >= vLen) t = clamp(t, v0 + vLen - mapLen * scale, v0);         // rule 2
    return clamp(t, f0 - b0 * scale, f0 + fLen - (b0 + bLen) * scale);                // rule 1
  };
  return {
    scale,
    x: along(box.x, box.width, w, free.x, free.width, view.x, view.width),
    y: along(box.y, box.height, h, free.y, free.height, view.y, view.height),
  };
}

/**
 * The same camera for a group inside an svg whose viewBox is the map and which the browser fits into its cell
 * (preserveAspectRatio "xMidYMid meet"): the transform of that group, in map units. The camera's screen px are
 * counted from the cell's top-left corner.
 */
export function inViewBox(camera: Camera, [w, h]: [number, number], cell: { width: number; height: number }): Camera {
  const fit = Math.min(cell.width / w, cell.height / h);
  return {
    scale: camera.scale / fit,
    x: (camera.x - (cell.width - w * fit) / 2) / fit,
    y: (camera.y - (cell.height - h * fit) / 2) / fit,
  };
}

// ---- what a record may carry in `extra` ---------------------------------------------------------------------------
// `extra` is free-form (telemetry.py): each engine module adds its own keys, and an older run has none of them.
// These readers check what they read: a value that is missing or of the wrong kind is null, never a crash.

const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const obj = (v: unknown): Record<string, unknown> | null =>
  typeof v === "object" && v !== null && !Array.isArray(v) ? (v as Record<string, unknown>) : null;

/** How the matcher searched for this frame (`extra.search`: "global", "window", …). */
export function searchOf(rec: RunRecord): string | null {
  return typeof rec.extra.search === "string" && rec.extra.search !== "" ? rec.extra.search : null;
}

/** Where the drone really was (`extra.truth`, written by synthetic flights), in map px. */
export function truthOf(rec: RunRecord): { x: number; y: number } | null {
  const t = obj(rec.extra.truth);
  return t && finite(t.x) && finite(t.y) ? { x: t.x, y: t.y } : null;
}

export const GPS_STATES = ["NO_DATA", "OK", "SUSPECT", "SPOOFED"] as const;
export type GpsState = (typeof GPS_STATES)[number];

export interface GpsReading {
  x: number | null; // where the GPS receiver says the drone is, in map px; null when it says nothing
  y: number | null;
  state: GpsState; // the spoof detector's verdict (M5-07)
  alarm: boolean;
  gapPx: number | null; // how far the reading is from the camera's position
}

/** The GPS reading checked against the camera (`extra.gps`, written by the spoof detector). */
export function gpsOf(rec: RunRecord): GpsReading | null {
  const g = obj(rec.extra.gps);
  if (!g || !GPS_STATES.includes(g.state as GpsState)) return null;
  const at = finite(g.x) && finite(g.y);
  return {
    x: at ? (g.x as number) : null,
    y: at ? (g.y as number) : null,
    state: g.state as GpsState,
    alarm: g.alarm === true,
    gapPx: finite(g.gap_px) ? g.gap_px : null,
  };
}

/** The part of the map the matcher searched for this frame (`extra.predicted_window`: [x0, y0, x1, y1]); null = all of it. */
export function windowOf(rec: RunRecord): Rect | null {
  const w = rec.extra.predicted_window;
  if (!Array.isArray(w) || w.length !== 4 || !w.every(finite)) return null;
  const [x0, y0, x1, y1] = w as number[];
  return x1 > x0 && y1 > y0 ? { x: x0, y: y0, width: x1 - x0, height: y1 - y0 } : null;
}

/** The camera panel's own size (M7-05): what the viewer asked for, and the room it has. All in CSS px. */
export interface CameraRoom {
  /** the smallest panel: the size it has when nobody resized it */
  min: number;
  /** the camera frame's width / height: the picture keeps it at any size */
  aspect: number;
  /** what the panel adds around the picture: its padding left and right, and padding plus caption above and below */
  chrome: { x: number; y: number };
  /** the room on the stage: the widest and the tallest the panel may be (under the top row, inside the map) */
  max: { width: number; height: number };
}

/**
 * The camera panel's width for a wished one: never under the default size, never out of the stage, and the picture
 * keeps its aspect ratio, so a low stage limits the width too. A stage with no room for more gives the default.
 */
export function cameraWidth(want: number | null | undefined, room: CameraRoom): number {
  const { min, aspect, chrome, max } = room;
  if (want == null || !Number.isFinite(want) || !(aspect > 0)) return min;
  const byHeight = (max.height - chrome.y) * aspect + chrome.x; // the width at which the panel is as tall as the room
  const most = Math.max(min, Math.min(max.width, byHeight));
  return Math.round(Math.min(Math.max(want, min), most));
}

/** The width a drag of the corner handle asks for: the handle follows the pointer along the picture's diagonal. */
export function cameraDrag(startWidth: number, dx: number, dyUp: number, aspect: number): number {
  const byHeight = dyUp * aspect; // moving up by dy makes the picture dy higher, so dy * aspect wider
  return startWidth + (Math.abs(dx) >= Math.abs(byHeight) ? dx : byHeight);
}
