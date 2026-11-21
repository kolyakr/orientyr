/**
 * The server half of the run reader: where the run folder is and how to read files out of it.
 * Only server code (the page, the API routes, tests) imports this; the browser gets the parsed Run as props.
 */
import { existsSync, readFileSync, readdirSync, realpathSync, statSync } from "node:fs";
import path from "node:path";
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
export function resolveRun(name: string | null | undefined, env: Env = process.env): { name: string | null; dir: string } {
  const root = runsDir(env);
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
}

/** Everything the page needs to show a run, or to say why it cannot: this never throws. */
export function openRun(name: string | null | undefined, env: Env = process.env): OpenRun {
  const runs = listRuns(runsDir(env));
  let at: { name: string | null; dir: string | null } = { name: name || null, dir: null };
  try {
    at = resolveRun(name, env);
    const run = loadRun(at.dir!);
    if (run.records.length === 0) throw new RunFormatError(`run.jsonl in ${at.dir} has no records`);
    const assets = assetUrls(at.dir!, run.info);
    // static images are the host's files, not this server's: on a static host they are not on its disk to check
    const mapMissing = assets === null && !isFile(mapPath(at.dir!, run.info));
    return { ...at, runs, run, error: null, mapMissing, assets };
  } catch (e) {
    return { ...at, runs, run: null, error: (e as Error).message, mapMissing: false, assets: null };
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
