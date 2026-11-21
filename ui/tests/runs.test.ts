// Any run opens, not just one: the runs folder, the names in it, and the safe way from a name in the URL to a
// folder on the disk. The API routes are called as plain functions, on a runs folder made for the test.
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { GET as getFrame } from "@/app/api/frame/route";
import { GET as getMap } from "@/app/api/map/route";
import { RunFormatError } from "@/lib/run";
import { listRuns, openRun, resolveRun, runsDir } from "@/lib/run-server";

const RUNS = path.join(__dirname, "fixtures", "runs");
const FIXTURE = path.join(__dirname, "fixtures", "run");
const env = (more: Record<string, string> = {}) => ({ ORIENTYR_RUNS: RUNS, ...more });

describe("runsDir", () => {
  it("is ORIENTYR_RUNS, else the folder ORIENTYR_RUN lies in, else outputs/ in the repo root", () => {
    expect(runsDir({ ORIENTYR_RUNS: RUNS, ORIENTYR_RUN: "/somewhere/else/run" })).toBe(RUNS);
    expect(runsDir({ ORIENTYR_RUN: FIXTURE })).toBe(path.dirname(FIXTURE));
    expect(runsDir({})).toBe(path.resolve(process.cwd(), "..", "outputs"));
  });
});

describe("listRuns", () => {
  it("lists the folders that hold a run.json, also one folder down, sorted, and nothing else", () => {
    expect(listRuns(RUNS)).toEqual(["broken", "empty", "hard", "nested/deep", "nowhere", "one"]);
  });

  it("is empty for a folder that does not exist", () => {
    expect(listRuns(path.join(RUNS, "nope"))).toEqual([]);
  });
});

describe("resolveRun", () => {
  it("finds a run by its name, also a nested one", () => {
    expect(resolveRun("hard", env())).toEqual({ name: "hard", dir: path.join(RUNS, "hard") });
    expect(resolveRun("nested/deep", env()).dir).toBe(path.join(RUNS, "nested", "deep"));
  });

  it.each(["../run", "..", ".", "/etc", "hard/../../run", "hard/..", "stray", "nope", "hard\\..\\..", "%2e%2e", "nested", "hard/"])(
    "refuses %j: only a listed run can be opened by name", (name) => {
      expect(() => resolveRun(name, env())).toThrow(RunFormatError);
    });

  it("names the runs folder and the runs it does have when the name is unknown", () => {
    expect(() => resolveRun("nope", env())).toThrow(/nope.*hard/s);
  });

  it("without a name opens ORIENTYR_RUN, as before", () => {
    const r = resolveRun(undefined, { ORIENTYR_RUN: FIXTURE });
    expect(r.dir).toBe(FIXTURE);
    expect(r.name).toBe("run"); // it lies in the runs folder, so it has a name there
    // ORIENTYR_RUN outside the runs folder: still the default, but it has no name to put in a URL
    expect(resolveRun(undefined, env({ ORIENTYR_RUN: FIXTURE }))).toEqual({ name: null, dir: FIXTURE });
    expect(resolveRun("", env({ ORIENTYR_RUN: FIXTURE })).dir).toBe(FIXTURE);
  });

  it("without a name and without ORIENTYR_RUN opens 'fake' when there is one, else the first run", () => {
    expect(resolveRun(undefined, env()).name).toBe("broken");
    expect(resolveRun(undefined, { ORIENTYR_RUNS: path.join(__dirname, "fixtures") }).name).toBe("run");
  });

  it("says where it looked when there is no run at all", () => {
    expect(() => resolveRun(undefined, { ORIENTYR_RUNS: path.join(RUNS, "stray") })).toThrow(/stray/);
  });
});

describe("openRun never throws: a run, or the reason there is none", () => {
  it("opens a good run and lists the others for the picker", () => {
    const o = openRun("hard", env());
    expect(o.error).toBeNull();
    expect(o.name).toBe("hard");
    expect(o.run!.records.length).toBeGreaterThan(5);
    expect(o.runs).toContain("one");
    expect(o.mapMissing).toBe(true); // the fixture's map image is not in git
  });

  it.each([
    ["empty", /no records/],
    ["broken", /line 2/],
    ["nope", /nope/],
    ["../run", /run/],
  ])("gives the reason for %j and still lists the runs", (name, reason) => {
    const o = openRun(name, env());
    expect(o.run).toBeNull();
    expect(o.error).toMatch(reason);
    expect(o.runs).toContain("hard");
  });
});

describe("the API serves the chosen run's map and frames", () => {
  const PNG = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==", "base64");
  let root: string;
  const before = { ...process.env };
  const line = (idx: number) => JSON.stringify({ idx, t: idx, source: `frames/f${idx}.png`, state: "LOST", x: 1, y: 1, vx: 0, vy: 0, cov: [[1, 0], [0, 1]], fix: null });

  beforeAll(() => {
    root = mkdtempSync(path.join(os.tmpdir(), "orientyr-runs-"));
    for (const name of ["a", "b"]) {
      mkdirSync(path.join(root, "runs", name, "frames"), { recursive: true });
      writeFileSync(path.join(root, "runs", name, "run.json"), JSON.stringify({ map: `../../map-${name}.png`, map_size: [1, 1], generator: name }));
      writeFileSync(path.join(root, "runs", name, "run.jsonl"), `${line(0)}\n`);
      writeFileSync(path.join(root, "runs", name, "frames", "f0.png"), Buffer.concat([PNG, Buffer.from(name)]));
      writeFileSync(path.join(root, `map-${name}.png`), Buffer.concat([PNG, Buffer.from(`map ${name}`)]));
    }
    writeFileSync(path.join(root, "secret.png"), PNG);
    // a run folder reached through a link (a scratch folder of links to runs elsewhere): its map path, relative to
    // the real folder, must still resolve
    symlinkSync(path.join(root, "runs", "a"), path.join(root, "runs", "link"));
    process.env.ORIENTYR_RUNS = path.join(root, "runs");
    delete process.env.ORIENTYR_RUN;
  });
  afterEach(() => { process.env.ORIENTYR_RUNS = path.join(root, "runs"); delete process.env.ORIENTYR_RUN; });
  afterAll(() => {
    rmSync(root, { recursive: true, force: true });
    for (const k of ["ORIENTYR_RUNS", "ORIENTYR_RUN"]) {
      if (before[k] === undefined) delete process.env[k];
      else process.env[k] = before[k];
    }
  });

  const bytes = async (r: Response) => Buffer.from(await r.arrayBuffer());
  const frame = (query: string) => getFrame(new Request(`http://localhost/api/frame?${query}`));
  const map = (query: string) => getMap(new Request(`http://localhost/api/map?${query}`));

  it("serves each run's own frame", async () => {
    for (const name of ["a", "b"]) {
      const r = await frame(`run=${name}&src=frames/f0.png`);
      expect(r.status).toBe(200);
      expect(r.headers.get("content-type")).toBe("image/png");
      expect((await bytes(r)).subarray(PNG.length).toString()).toBe(name);
    }
  });

  it("serves each run's own map, also for a linked run folder", async () => {
    for (const [name, text] of [["a", "map a"], ["b", "map b"], ["link", "map a"]]) {
      const r = await map(`run=${name}`);
      expect(r.status, name).toBe(200);
      expect((await bytes(r)).subarray(PNG.length).toString()).toBe(text);
    }
  });

  it("serves the default run without ?run=, and ORIENTYR_RUN when it is set", async () => {
    expect((await bytes(await frame("src=frames/f0.png"))).subarray(PNG.length).toString()).toBe("a");
    process.env.ORIENTYR_RUN = path.join(root, "runs", "b");
    expect((await bytes(await frame("src=frames/f0.png"))).subarray(PNG.length).toString()).toBe("b");
    expect((await bytes(await map(""))).subarray(PNG.length).toString()).toBe("map b");
  });

  it.each([
    "run=a&src=../../secret.png",
    "run=a&src=frames/nope.png",
    "run=a&src=frames",
    "run=a&src=",
    "run=a",
    `run=a&src=${encodeURIComponent("/etc/hosts")}`,
    "run=..&src=secret.png",
    "run=../runs/a&src=frames/f0.png",
    "run=nope&src=frames/f0.png",
  ])("answers 404 to /api/frame?%s", async (query) => {
    expect((await frame(query)).status).toBe(404);
  });

  it.each(["run=..", "run=nope", "run=a/../b"])("answers 404 to /api/map?%s", async (query) => {
    expect((await map(query)).status).toBe(404);
  });

  it("answers 404 to a run given as an absolute path", async () => {
    // the temp folder exists only while the tests run, so this case cannot sit in the table above
    expect((await map(`run=${encodeURIComponent(path.join(root, "runs", "a"))}`)).status).toBe(404);
    expect((await frame(`run=${encodeURIComponent(path.join(root, "runs", "a"))}&src=frames/f0.png`)).status).toBe(404);
  });
});
