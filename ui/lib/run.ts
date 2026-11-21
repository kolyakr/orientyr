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
const NUMBERS = ["idx", "t", "x", "y", "vx", "vy"] as const;

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function recordFrom(d: unknown): RunRecord {
  if (!isObject(d)) throw new RunFormatError("a record must be a JSON object");
  for (const k of REQUIRED) if (d[k] === undefined) throw new RunFormatError(`missing field '${k}'`);
  for (const k of NUMBERS) {
    if (typeof d[k] !== "number" || !Number.isFinite(d[k])) throw new RunFormatError(`'${k}' must be a finite number`);
  }
  if (typeof d.source !== "string") throw new RunFormatError("'source' must be a string");
  if (!TRACK_MODES.includes(d.state as TrackMode)) {
    throw new RunFormatError(`state must be one of ${TRACK_MODES.join(", ")}, got ${JSON.stringify(d.state)}`);
  }
  if (d.fix !== null && (!isObject(d.fix) || !Array.isArray(d.fix.footprint))) {
    throw new RunFormatError("'fix' must be an object with a footprint, or null");
  }
  // The writer always writes the optional fields; the defaults are the reader's side of the same contract.
  return {
    ...(d as unknown as RunRecord),
    backend: (d.backend as string | null) ?? null,
    ms: (d.ms as number | null) ?? null,
    interpolated: d.interpolated === true,
    extra: isObject(d.extra) ? d.extra : {},
  };
}

/** Parse the text of run.jsonl. Blank lines are skipped; records must be in frame order. */
export function parseRunJsonl(text: string): RunRecord[] {
  const records: RunRecord[] = [];
  text.split("\n").forEach((line, i) => {
    if (!line.trim()) return;
    const where = `run.jsonl line ${i + 1}`;
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
    const prev = records.at(-1);
    if (prev && (rec.idx <= prev.idx || rec.t < prev.t)) {
      throw new RunFormatError(`${where}: records must be in frame order (idx ${rec.idx} after idx ${prev.idx})`);
    }
    records.push(rec);
  });
  return records;
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
    frame_size: (d.frame_size as [number, number] | null) ?? null,
    fps: (d.fps as number | null) ?? null,
    seed: (d.seed as number | null) ?? null,
  };
}
