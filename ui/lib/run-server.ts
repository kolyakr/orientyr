/**
 * The server half of the run reader: where the run folder is and how to read files out of it.
 * Only server code (the page, the API routes, tests) imports this; the browser gets the parsed Run as props.
 */
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { type Run, type RunInfo, RunFormatError, parseRunInfo, parseRunJsonl } from "./run";

/**
 * The run folder the console shows: ORIENTYR_RUN, or the fake run at outputs/fake in the repo root
 * (`python scripts/fake_run.py --map … --out outputs/fake`). `npm run dev` starts in ui/, hence the "..".
 */
export function runDir(): string {
  return path.resolve(process.env.ORIENTYR_RUN ?? path.join(process.cwd(), "..", "outputs", "fake"));
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

/** The map's absolute path: run.json stores it relative to the run folder. */
export function mapPath(dir: string, info: RunInfo): string {
  return path.resolve(dir, info.map);
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
