/**
 * Live mode (M7-04), the browser-safe half: the engine's heartbeat (live.json, see src/dronenav/live.py) and how the
 * viewer moves the drone between two fixes. No file access here; lib/run-server.ts reads the files.
 *
 * The viewer learns of a new record by asking: every POLL_MS it reads what was appended to run.jsonl. A fix is a
 * point; between two fixes the drone marker glides in a straight line (a CSS transition on its transform). The glide
 * lasts a little longer than the fixes are apart, so the next fix arrives while the marker is still moving and it
 * turns towards the new point without having stopped: at one fix a second the picture still moves on every frame of
 * the screen.
 */

export const LIVE_STATES = ["starting", "running", "done", "stopped"] as const;
export type LiveState = (typeof LIVE_STATES)[number];

/** live.json, as the viewer uses it. */
export interface LiveStatus {
  state: LiveState;
  frames: number;           // records written
  fixes: number;            // of them, with an accepted fix
  skipped: number;          // frames the engine was too late for
  fixesPerS: number | null; // accepted fixes per second of wall time; null before the second fix
  jam: boolean;             // the last frame reached the engine jammed
  // the judge controls (M7-05); an engine before them reports none of this: nothing switched, nothing to choose
  spoof: boolean;           // the simulated GPS receiver is being spoofed
  map: string | null;       // the name of the map the engine searches
  maps: string[];           // the maps it can switch to, by name
  flight: string | null;    // the name of the flight in the air
  flights: string[];        // the flights it can start
  speed: number | null;     // flight seconds played per second
  flightNo: number;         // how many flights were started after the first: a change is a new run in the same folder
}

/** The fastest flight the console orders: at 2 frames a second of flight, more would only skip frames. */
export const MAX_FLIGHT_SPEED = 16;
/** The speeds the "New flight" control offers. */
export const FLIGHT_SPEEDS = [0.5, 1, 2, 4, 8] as const;

/** How often the viewer asks for new records. Well under the second between two fixes that the spec allows. */
export const POLL_MS = 400;
/** The longest glide: after a silence (a jam, a stalled engine) the drone must not crawl to the next fix. */
export const GLIDE_MAX_MS = 2500;
const PACE_MIN_MS = 50;      // two records in one answer are not "no time apart"
const PACE_FLOOR_MS = 1000;  // nothing measured yet: the spec's floor, one fix a second
const LEARN = 0.3;           // the weight of the newest gap in the pace

const isNumber = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** Parse live.json; null when it is not a status (half-written, another file, an unknown state). */
export function parseLiveStatus(text: string): LiveStatus | null {
  let d: unknown;
  try {
    d = JSON.parse(text);
  } catch {
    return null;
  }
  if (typeof d !== "object" || d === null || Array.isArray(d)) return null;
  const s = d as Record<string, unknown>;
  if (!LIVE_STATES.includes(s.state as LiveState)) return null;
  const count = (v: unknown) => (isNumber(v) && v >= 0 ? v : 0);
  const name = (v: unknown) => (typeof v === "string" && v !== "" ? v : null);
  const names = (v: unknown) => (Array.isArray(v) ? v.filter((x): x is string => name(x) !== null) : []);
  return {
    state: s.state as LiveState,
    frames: count(s.frames),
    fixes: count(s.fixes),
    skipped: count(s.skipped),
    fixesPerS: isNumber(s.fixes_per_s) ? s.fixes_per_s : null,
    jam: s.jam === true,
    spoof: s.spoof === true,
    map: name(s.map),
    maps: names(s.maps),
    flight: name(s.flight),
    flights: names(s.flights),
    speed: isNumber(s.speed) && s.speed > 0 ? s.speed : null,
    flightNo: count(s.flight_no),
  };
}

/**
 * The time between two records, learned from when they arrive: a running mean, so one late record (a slow search)
 * stretches the glide a little and a jam's silence does not become the new pace.
 */
export function nextPace(pace: number | null, gapMs: number): number | null {
  if (!isNumber(gapMs) || gapMs < 0) return pace;
  const gap = Math.min(Math.max(gapMs, PACE_MIN_MS), GLIDE_MAX_MS);
  return pace === null ? gap : pace * (1 - LEARN) + gap * LEARN;
}

/**
 * How long the drone glides to a new fix, ms. The next record is made one pace later and seen at most one poll after
 * that, so the glide covers both, plus a fifth of the pace for a record that is late: the marker is still moving
 * when its next target arrives.
 */
export function glideMs(pace: number | null): number {
  const p = pace ?? PACE_FLOOR_MS;
  return Math.min(GLIDE_MAX_MS, Math.round(p * 1.2 + POLL_MS));
}
