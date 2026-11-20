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
  if (rec.vx === 0 && rec.vy === 0) return null;
  const deg = (Math.atan2(rec.vx, -rec.vy) * 180) / Math.PI; // map y points down, so up is -y
  return ((deg % 360) + 360) % 360;
}

export interface TrailSegment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number; // of the frame the segment arrives at
  coasting: boolean; // arrived without an accepted fix: dead-reckoned, drawn dashed
}

/** The route flown so far: one segment per step, from frame 0 up to `frame`. */
export function trail(records: RunRecord[], frame: number): TrailSegment[] {
  const segs: TrailSegment[] = [];
  for (let i = 1; i <= Math.min(frame, records.length - 1); i++) {
    const a = records[i - 1];
    const b = records[i];
    segs.push({ x1: a.x, y1: a.y, x2: b.x, y2: b.y, confidence: confidence(b), coasting: !b.fix?.accepted });
  }
  return segs;
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

/** The captions the engine attaches to state changes (`extra.event`): the mission log. */
export function events(records: RunRecord[]): RunEvent[] {
  return records.flatMap((r, frame) =>
    typeof r.extra.event === "string" ? [{ frame, t: r.t, state: r.state, text: r.extra.event }] : [],
  );
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
