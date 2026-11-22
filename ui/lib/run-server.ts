/**
 * The server half of the run reader: where the run folder is and how to read files out of it.
 * Only server code (the page, the API routes, tests) imports this; the browser gets the parsed Run as props.
 */
import { closeSync, existsSync, openSync, readFileSync, readSync, readdirSync, realpathSync, renameSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { type LiveStatus, parseLiveStatus } from "./live";
import { type Run, type RunInfo, RunFormatError, parseRunInfo, parseRunJsonl } from "./run";

type Env = Record<string, string | undefined>;

/** Where packed runs (scripts/pack_run.py) lie in the app folder: under public/, so their images are static files. */
const PACKED = path.join("public", "runs");

/**
 * The folder whose sub-folders are the runs the console can open: ORIENTYR_RUNS, else the folder the one run of
 * ORIENTYR_RUN lies in, else the packed runs of the app (public/runs) when there are any: that is the hosted
 * console, which has no settings and no repo around it. Else outputs/ in the repo root (`npm run dev` starts in
 * ui/, hence the ".."). `app` is the app folder, the cwd of `next dev` and `next start`.
 */
export function runsDir(env: Env = process.env, app: string = process.cwd()): string {
  if (env.ORIENTYR_RUNS) return path.resolve(env.ORIENTYR_RUNS);
  if (env.ORIENTYR_RUN) return path.dirname(path.resolve(env.ORIENTYR_RUN));
  const packed = path.resolve(app, PACKED);
  if (listRuns(packed).length > 0) return packed;
  return path.resolve(app, "..", "outputs");
}

/** Where the browser gets a run's images: the map's URL and the URL of the folder the frames' `source` is relative to. */
export interface Assets {
  map: string;
  frames: string;
}

/**
 * The static URLs of a run's images, or null when they must go through /api/map and /api/frame. Static is only
 * right for a self-contained run under the app's public/ folder (a packed run): the host then serves the images
 * itself, with no server code and no size limit on a response. A run whose map lies outside its folder is not
 * reachable by a static URL, so it keeps the API.
 */
export function assetUrls(dir: string, info: RunInfo, app: string = process.cwd()): Assets | null {
  const pub = path.resolve(app, "public");
  const run = path.resolve(dir);
  const map = path.resolve(run, info.map);
  if (!run.startsWith(pub + path.sep) || !map.startsWith(run + path.sep)) return null;
  const url = (file: string) => "/" + path.relative(pub, file).split(path.sep).map(encodeURIComponent).join("/");
  return { map: url(map), frames: url(run) };
}

const isRun = (dir: string) => existsSync(path.join(dir, "run.json"));

/** The sub-folders of `dir`, links to folders too; none when it cannot be read. */
function folders(dir: string): string[] {
  let names: string[];
  try {
    names = readdirSync(dir);
  } catch {
    return [];
  }
  return names.filter((name) => {
    if (name.startsWith(".") || name === "node_modules") return false;
    try {
      return statSync(path.join(dir, name)).isDirectory();
    } catch {
      return false; // a dangling link
    }
  });
}

/**
 * The names of the runs in a runs folder, sorted: every sub-folder that holds a run.json, and one level further
 * down ("benchmark/mild_2022") for runs kept in groups. The name is the path from the runs folder, with "/".
 */
export function listRuns(dir: string): string[] {
  const names: string[] = [];
  for (const name of folders(dir)) {
    if (isRun(path.join(dir, name))) names.push(name);
    else for (const sub of folders(path.join(dir, name))) if (isRun(path.join(dir, name, sub))) names.push(`${name}/${sub}`);
  }
  return names.sort();
}

/**
 * From a run's name (the `?run=` of a URL) to its folder. The name is never used as a path: it must be one of the
 * names listRuns found, so nothing outside the runs folder can be reached, however the name is spelled.
 * Without a name: the run of ORIENTYR_RUN, else the run called "fake", else the first one.
 */
export function resolveRun(name: string | null | undefined, env: Env = process.env, app: string = process.cwd()): { name: string | null; dir: string } {
  const root = runsDir(env, app);
  const runs = listRuns(root);
  const have = runs.length ? `the runs there: ${runs.join(", ")}` : "there are no runs there (a run is a folder with a run.json)";
  if (name) {
    if (!runs.includes(name)) throw new RunFormatError(`${JSON.stringify(name)} is not a run in ${root}; ${have}`);
    return { name, dir: path.join(root, ...name.split("/")) };
  }
  if (env.ORIENTYR_RUN) {
    const dir = path.resolve(env.ORIENTYR_RUN);
    const rel = path.relative(root, dir).split(path.sep).join("/");
    return { name: runs.includes(rel) ? rel : null, dir };
  }
  const first = runs.includes("fake") ? "fake" : runs[0];
  if (first === undefined) throw new RunFormatError(`no run to open in ${root}: ${have}`);
  return { name: first, dir: path.join(root, ...first.split("/")) };
}

/** Read run.json alone: all the map route needs, without parsing every record. */
export function loadRunInfo(dir: string): RunInfo {
  const file = path.join(dir, "run.json");
  if (!existsSync(file)) throw new RunFormatError(`no run.json in the run folder ${dir}`);
  return parseRunInfo(readFileSync(file, "utf-8"));
}

export interface OpenRun {
  name: string | null; // the run's name in the runs folder; null for a run outside it (ORIENTYR_RUN)
  dir: string | null;
  runs: string[];      // every run that can be opened, for the picker
  run: Run | null;
  error: string | null; // why there is no run
  mapMissing: boolean;  // the run opens, but its map image is not where run.json says
  assets: Assets | null; // static URLs of the run's images; null = the API routes
  live: LiveStart | null; // the run is being written right now: where the viewer reads on (see readTail)
  waiting: boolean;       // a live run with no record yet: not an error, the page asks again
}

/** What the page hands the console about a live run: the byte of run.jsonl to read on from, and the judge controls. */
export interface LiveStart extends Switches {
  next: number;
}

/** The judge controls of a live run, as the console shows them: what is switched on and what can be chosen. */
export interface Switches {
  jam: boolean;
  spoof: boolean;
  map: string | null;      // the map that is ordered: the pressed button
  mapShown: string | null; // the map the engine searches by now (run.json names it): the picture
  maps: string[];
  flight: string | null;
  flights: string[];
  speed: number | null;
  flightNo: number;
}

/** Everything the page needs to show a run, or to say why it cannot: this never throws. */
export function openRun(name: string | null | undefined, env: Env = process.env, app: string = process.cwd()): OpenRun {
  const runs = listRuns(runsDir(env, app));
  let at: { name: string | null; dir: string | null } = { name: name || null, dir: null };
  const none = { run: null, mapMissing: false, assets: null, live: null, waiting: false };
  try {
    at = resolveRun(name, env, app);
    const dir = at.dir!;
    const status = liveStatus(dir);
    if (status?.live) {
      // The engine may be in the middle of a line: take the whole lines, the console reads on from there.
      const tail = readTail(dir, 0);
      const run = { info: loadRunInfo(dir), records: parseRunJsonl(tail.text) };
      if (run.records.length === 0) {
        return { ...at, runs, ...none, waiting: true, error: `the live run in ${dir} is waiting for its first frame` };
      }
      const assets = assetUrls(dir, run.info, app);
      return { ...at, runs, run, error: null, mapMissing: assets === null && !isFile(mapPath(dir, run.info)), assets,
               live: { next: tail.next, ...switches(dir, status) }, waiting: false };
    }
    const run = loadRun(dir);
    if (run.records.length === 0) throw new RunFormatError(`run.jsonl in ${dir} has no records`);
    const assets = assetUrls(dir, run.info, app);
    // static images are the host's files, not this server's: on a static host they are not on its disk to check
    const mapMissing = assets === null && !isFile(mapPath(dir, run.info));
    return { ...at, runs, run, error: null, mapMissing, assets, live: null, waiting: false };
  } catch (e) {
    return { ...at, runs, ...none, error: (e as Error).message };
  }
}

// ---- live mode (M7-04): a run that is still being written; the format is src/dronenav/live.py's ------------------

const STATUS_FILE = "live.json";
const CONTROL_FILE = "control.json";
/** live.json is rewritten after every record. Silent for this long, the engine is taken for dead (killed, crashed). */
const SILENT_MS = 30_000;
/** Before the first record the engine may be reading a long video's first frame: more patience. */
const STARTING_MS = 5 * 60_000;

/** The engine's heartbeat, and whether the run is being written right now. Null: the run was never live. */
export function liveStatus(dir: string, now: number = Date.now()): (LiveStatus & { live: boolean }) | null {
  const file = path.join(dir, STATUS_FILE);
  let status: LiveStatus | null, age: number;
  try {
    status = parseLiveStatus(readFileSync(file, "utf-8"));
    age = now - statSync(file).mtimeMs;
  } catch {
    return null;
  }
  if (!status) return null;
  const live = status.state === "running" ? age < SILENT_MS : status.state === "starting" ? age < STARTING_MS : false;
  return { ...status, live };
}

/** The operator's orders (control.json) as they stand; {} when there are none, or the file is being written. */
function orders(dir: string): Record<string, unknown> {
  try {
    const d: unknown = JSON.parse(readFileSync(path.join(dir, CONTROL_FILE), "utf-8"));
    if (typeof d === "object" && d !== null && !Array.isArray(d)) return d as Record<string, unknown>;
  } catch { /* no order yet, or one being written */ }
  return {};
}

/**
 * The judge controls as the console shows them: the operator's order (control.json) where there is one, else what
 * the engine reports. A switch shows a press at once, before the engine's next frame confirms it. `mapShown` alone
 * is always the engine's word: the map picture changes when run.json does.
 */
export function switches(dir: string, status: LiveStatus | null): Switches {
  const o = orders(dir);
  const flight = typeof o.flight === "object" && o.flight !== null ? (o.flight as { name?: unknown }).name : undefined;
  return {
    jam: typeof o.jam === "boolean" ? o.jam : status?.jam ?? false,
    spoof: typeof o.spoof === "boolean" ? o.spoof : status?.spoof ?? false,
    map: typeof o.map === "string" && status?.maps.includes(o.map) ? o.map : status?.map ?? null,
    mapShown: status?.map ?? null,
    maps: status?.maps ?? [],
    flight: typeof flight === "string" && status?.flights.includes(flight) ? flight : status?.flight ?? null,
    flights: status?.flights ?? [],
    speed: status?.speed ?? null,
    flightNo: status?.flightNo ?? 0,
  };
}

/** Is the jam switched on? (M7-04's reader: the jam switch alone of `switches`.) */
export function jamOrder(dir: string, status: LiveStatus | null): boolean {
  return switches(dir, status).jam;
}

/** One order of the operator: a switch, a map, or a flight to start. */
export type Order = { jam: boolean } | { spoof: boolean } | { map: string } | { flight: string; speed: number };

/**
 * Add an order to control.json and return what was ordered. The other orders stay as they are: the engine keeps
 * what a key says until it changes. A flight gets the next id (the engine flies each id once) and switches the jam
 * and the spoof off: they were ordered for the flight that ends. Written in one step (write, then rename): the
 * engine never reads half an order.
 */
export function writeOrder(dir: string, order: Order): Record<string, unknown> {
  const old = orders(dir);
  let said: Record<string, unknown> = order;
  let next: Record<string, unknown> = { ...old, ...order };
  if ("flight" in order) {
    const last = typeof old.flight === "object" && old.flight !== null ? (old.flight as { id?: unknown }).id : 0;
    const id = (typeof last === "number" && Number.isInteger(last) ? last : 0) + 1;
    said = { flight: order.flight, speed: order.speed, id };
    next = { ...old, jam: false, spoof: false, flight: { name: order.flight, speed: order.speed, id } };
  }
  const part = path.join(dir, `.${CONTROL_FILE}.partial`);
  writeFileSync(part, JSON.stringify(next));
  renameSync(part, path.join(dir, CONTROL_FILE));
  return said;
}

/** Write the jam order (M7-04's writer). */
export function writeJamOrder(dir: string, jam: boolean): void {
  writeOrder(dir, { jam });
}

/**
 * The lines appended to run.jsonl from byte `from` on: whole lines only (a line the engine is still writing waits
 * for the next read), and `next`, the byte to ask from next time. A file shorter than `from` is a run that was
 * started again in the same folder: `reset`, and the read starts from the top.
 */
export function readTail(dir: string, from: number): { text: string; next: number; reset: boolean } {
  let start = Number.isInteger(from) && from > 0 ? from : 0;
  let fd: number;
  try {
    fd = openSync(path.join(dir, "run.jsonl"), "r");
  } catch {
    return { text: "", next: 0, reset: false };
  }
  try {
    const size = statSync(path.join(dir, "run.jsonl")).size;
    const reset = start > size;
    if (reset) start = 0;
    const buf = Buffer.alloc(size - start);
    const got = readSync(fd, buf, 0, buf.length, start);
    const end = buf.subarray(0, got).lastIndexOf(0x0a) + 1; // bytes, so a cut never lands inside a character
    return { text: buf.toString("utf-8", 0, end), next: start + end, reset };
  } finally {
    closeSync(fd);
  }
}

function isFile(file: string): boolean {
  try {
    return statSync(file).isFile();
  } catch {
    return false;
  }
}

/** The default run's folder: ORIENTYR_RUN, else see resolveRun. Kept for callers that open one run only. */
export function runDir(): string {
  return resolveRun(null).dir;
}

/** Read run.json and run.jsonl from a run folder. */
export function loadRun(dir: string): Run {
  const read = (name: string) => {
    const file = path.join(dir, name);
    if (!existsSync(file)) throw new RunFormatError(`no ${name} in the run folder ${dir}`);
    return readFileSync(file, "utf-8");
  };
  return { info: parseRunInfo(read("run.json")), records: parseRunJsonl(read("run.jsonl")) };
}

/**
 * The map's absolute path: run.json stores it relative to the run folder. The real folder: a run reached through a
 * link (a folder of links to runs kept elsewhere) still finds its map.
 */
export function mapPath(dir: string, info: RunInfo): string {
  let real = dir;
  try {
    real = realpathSync(dir);
  } catch { /* a folder that is not there: the caller's read fails with the plain path */ }
  return path.resolve(real, info.map);
}

/**
 * A frame's absolute path. `source` comes from the URL, so it must stay inside the run folder:
 * anything that climbs out of it (../, an absolute path) is refused.
 */
export function framePath(dir: string, source: string): string {
  const root = path.resolve(dir);
  const file = path.resolve(root, source);
  if (path.isAbsolute(source) || !file.startsWith(root + path.sep)) {
    throw new RunFormatError(`the frame ${JSON.stringify(source)} is outside the run folder`);
  }
  return file;
}
