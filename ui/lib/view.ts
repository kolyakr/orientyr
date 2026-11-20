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

/** green (trusted) → amber → red (no fix), as an HSL colour. */
export function confidenceColor(c: number): string {
  return `hsl(${Math.round(4 + 148 * Math.min(1, Math.max(0, c)))} 82% 58%)`;
}

export const STATE_LABEL: Record<TrackMode, string> = {
  TRACKING: "Tracking",
  LOST: "Signal lost",
  RELOCALIZING: "Relocalizing",
};

/** mm:ss.s of flight time. */
export function clock(t: number): string {
  const m = Math.floor(t / 60);
  return `${String(m).padStart(2, "0")}:${(t - m * 60).toFixed(1).padStart(4, "0")}`;
}
