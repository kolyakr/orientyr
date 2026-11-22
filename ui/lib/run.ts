/**
 * A run, as the engine writes it (src/dronenav/telemetry.py is the source of truth for the format):
 * run.json holds the run-level facts, run.jsonl one record per frame. This file is the browser-safe half:
 * types and parsing, no file access. Coordinates are map pixels at full map resolution, y pointing down.
 */

export const TRACK_MODES = ["TRACKING", "LOST", "RELOCALIZING"] as const;
export type TrackMode = (typeof TRACK_MODES)[number];

export type Point = [number, number];

/** One homography fix: where the matcher put the frame on the map. */
export interface Fix {
  H: number[][];
  x: number;
  y: number;
  heading_deg: number; // clockwise from map up
  scale: number; // map px per frame px: an altitude proxy
  footprint: Point[]; // the frame's four corners on the map
  inliers: number;
  inlier_ratio: number;
  reproj_px: number;
  accepted: boolean;
  reject_reason: string | null;
}

/** One line of run.jsonl: what the tracker believed after this frame. */
export interface RunRecord {
  idx: number;
  t: number; // seconds from the start of the flight
  source: string; // the frame's path, relative to the run folder
  state: TrackMode;
  /**
   * False for a frame the tracker had no position for yet (the file holds null for x, y, vx, vy and cov, together:
   * the frames before the first fix). Such a frame is drawn nowhere; so that no reader has to deal with null,
   * its numbers are the place of the nearest located frame, standing still, with a zero covariance.
   */
  located: boolean;
  x: number;
  y: number;
  vx: number;
  vy: number;
  cov: number[][]; // 2x2 position covariance, px²
  fix: Fix | null;
  backend: string | null;
  ms: number | null;
  interpolated: boolean;
  extra: Record<string, unknown>;
}

/** run.json. `map` is relative to the run folder. */
export interface RunInfo {
  map: string;
  map_size: [number, number];
  generator: string;
  frame_size: [number, number] | null;
  fps: number | null;
  seed: number | null;
}

export interface Run {
  info: RunInfo;
  records: RunRecord[];
}

/** A run folder that does not follow the format. The message names the file, the line and the field. */
export class RunFormatError extends Error {
  name = "RunFormatError";
}

const REQUIRED = ["idx", "t", "source", "state", "x", "y", "vx", "vy", "cov", "fix"] as const;
const POSITION = ["x", "y", "vx", "vy"] as const;
const FIX_NUMBERS = ["x", "y", "heading_deg", "inliers", "inlier_ratio"] as const; // the ones the console draws or prints

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

const isNumber = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isPoint = (v: unknown) => Array.isArray(v) && v.length === 2 && v.every(isNumber);

function fixFrom(d: unknown): Fix | null {
  if (d === null) return null;
  if (!isObject(d)) throw new RunFormatError("'fix' must be an object or null");
  for (const k of FIX_NUMBERS) if (!isNumber(d[k])) throw new RunFormatError(`'fix.${k}' must be a finite number`);
  if (typeof d.accepted !== "boolean") throw new RunFormatError("'fix.accepted' must be true or false");
  if (!Array.isArray(d.footprint) || d.footprint.length < 3 || !d.footprint.every(isPoint)) {
    throw new RunFormatError("'fix.footprint' must be the frame's corners on the map: [[x, y], …]");
  }
  return { ...(d as unknown as Fix), reject_reason: typeof d.reject_reason === "string" ? d.reject_reason : null };
}

function recordFrom(d: unknown): RunRecord {
  if (!isObject(d)) throw new RunFormatError("a record must be a JSON object");
  for (const k of REQUIRED) if (d[k] === undefined) throw new RunFormatError(`missing field '${k}'`);
  if (!Number.isInteger(d.idx)) throw new RunFormatError("'idx' must be a whole number");
  if (!isNumber(d.t)) throw new RunFormatError("'t' must be a finite number");
  if (typeof d.source !== "string") throw new RunFormatError("'source' must be a string");
  if (!TRACK_MODES.includes(d.state as TrackMode)) {
    throw new RunFormatError(`state must be one of ${TRACK_MODES.join(", ")}, got ${JSON.stringify(d.state)}`);
  }
  // No position yet: the writer puts null into all five fields. Half a position is a broken file.
  const located = !([...POSITION, "cov"] as const).every((k) => d[k] === null);
  if (located) {
    for (const k of POSITION) {
      if (!isNumber(d[k])) throw new RunFormatError(`'${k}' must be a finite number (x, y, vx, vy and cov are null only together)`);
    }
    const cov = d.cov;
    if (!Array.isArray(cov) || cov.length !== 2 || !cov.every(isPoint)) {
      throw new RunFormatError("'cov' must be a 2 x 2 matrix of finite numbers");
    }
  }
  // The writer always writes the optional fields; the defaults are the reader's side of the same contract.
  return {
    ...(d as unknown as RunRecord),
    located,
    ...(located ? {} : { x: 0, y: 0, vx: 0, vy: 0, cov: [[0, 0], [0, 0]] }),
    fix: fixFrom(d.fix),
    backend: typeof d.backend === "string" ? d.backend : null,
    ms: isNumber(d.ms) ? d.ms : null,
    interpolated: d.interpolated === true,
    extra: isObject(d.extra) ? d.extra : {},
  };
}

/** Parse the text of run.jsonl. Blank lines are skipped; records must be in frame order. */
export function parseRunJsonl(text: string): RunRecord[] {
  return appendRunJsonl([], text);
}

/**
 * The records of a run that is still being written (live mode): `records` so far, plus the lines of `text` that
 * were appended to run.jsonl since. The result is what parseRunJsonl gives for the whole file. Nothing given is
 * changed: a record whose numbers change (see below) is replaced by a copy, every other one is the same object,
 * and with no new line the same list comes back, so React renders only what is new.
 */
export function appendRunJsonl(records: readonly RunRecord[], text: string): RunRecord[] {
  const all = records.slice();
  text.split("\n").forEach((line, i) => {
    if (!line.trim()) return;
    const where = `run.jsonl line ${records.length + i + 1}`;
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      throw new RunFormatError(`${where}: invalid JSON (is the file truncated or still being written?)`);
    }
    let rec: RunRecord;
    try {
      rec = recordFrom(parsed);
    } catch (e) {
      throw new RunFormatError(`${where}: ${(e as Error).message}`);
    }
    const prev = all.at(-1);
    if (prev && (rec.idx <= prev.idx || rec.t < prev.t)) {
      throw new RunFormatError(`${where}: records must be in frame order (idx ${rec.idx} after idx ${prev.idx})`);
    }
    all.push(rec);
  });
  if (all.length === records.length) return records as RunRecord[];
  // A frame without a position borrows the place of the located frame before it; the frames before the first
  // located one borrow its place (the first fix). Live, that fix may arrive later than they did.
  let near = all.find((r) => r.located);
  all.forEach((r, i) => {
    if (r.located) near = r;
    else if (near && (r.x !== near.x || r.y !== near.y)) all[i] = { ...r, x: near.x, y: near.y };
  });
  return all;
}

/** Parse the text of run.json. */
export function parseRunInfo(text: string): RunInfo {
  let d: unknown;
  try {
    d = JSON.parse(text);
  } catch {
    throw new RunFormatError("run.json: invalid JSON");
  }
  if (!isObject(d)) throw new RunFormatError("run.json: must be a JSON object");
  if (typeof d.map !== "string" || d.map === "") throw new RunFormatError("run.json: 'map' must be a path to the map");
  const size = d.map_size;
  if (!Array.isArray(size) || size.length !== 2 || !size.every((n) => Number.isInteger(n) && n > 0)) {
    throw new RunFormatError("run.json: 'map_size' must be [w, h]");
  }
  if (typeof d.generator !== "string") throw new RunFormatError("run.json: 'generator' must be a string");
  return {
    map: d.map,
    map_size: size as [number, number],
    generator: d.generator,
    frame_size: Array.isArray(d.frame_size) && d.frame_size.length === 2 && d.frame_size.every((n) => isNumber(n) && n > 0)
      ? (d.frame_size as [number, number]) : null,
    fps: isNumber(d.fps) && d.fps > 0 ? d.fps : null,
    seed: isNumber(d.seed) ? d.seed : null,
  };
}
