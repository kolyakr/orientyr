// The UI's reader of a run folder (run.json + run.jsonl), checked against a run the Python generator wrote.
// With ORIENTYR_RUN set (the pytest contract test does it) the run is a fresh one, frames included;
// otherwise it is the tiny fixture in tests/fixtures/run (no frames: only the telemetry is kept in git).
import { existsSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { RunFormatError, parseRunInfo, parseRunJsonl } from "@/lib/run";
import { framePath, loadRun, mapPath } from "@/lib/run-server";

const FIXTURE = path.join(__dirname, "fixtures", "run");
const RUN_DIR = process.env.ORIENTYR_RUN ?? FIXTURE;
const FRESH = process.env.ORIENTYR_RUN !== undefined;

const LINE = JSON.stringify({
  idx: 0, t: 0, source: "frames/frame_00000.png", state: "TRACKING", x: 10, y: 20, vx: 1, vy: 0,
  cov: [[4, 0], [0, 4]], fix: null,
});

describe("parseRunJsonl", () => {
  it("reads one record per line, skips blank lines, fills the optional fields", () => {
    const recs = parseRunJsonl(`${LINE}\n\n${LINE.replace('"idx":0', '"idx":1')}\n`);
    expect(recs.map((r) => r.idx)).toEqual([0, 1]);
    expect(recs[0]).toMatchObject({ backend: null, ms: null, interpolated: false, extra: {}, fix: null });
  });

  it("names the line of invalid JSON", () => {
    expect(() => parseRunJsonl(`${LINE}\n{"idx": 1,\n`)).toThrow(RunFormatError);
    expect(() => parseRunJsonl(`${LINE}\n{"idx": 1,\n`)).toThrow(/line 2/);
  });

  it("names a missing field and an unknown state", () => {
    const noX = JSON.stringify({ ...JSON.parse(LINE), x: undefined });
    expect(() => parseRunJsonl(noX)).toThrow(/line 1.*'x'/);
    expect(() => parseRunJsonl(LINE.replace("TRACKING", "FLYING"))).toThrow(/state/);
  });

  it("rejects records out of frame order", () => {
    expect(() => parseRunJsonl(`${LINE}\n${LINE}\n`)).toThrow(/order/);
  });
});

describe("parseRunInfo", () => {
  it("reads run.json and rejects one without a map", () => {
    const info = parseRunInfo('{"map": "../map.png", "map_size": [600, 400], "generator": "fake_run"}');
    expect(info).toMatchObject({ map: "../map.png", map_size: [600, 400], frame_size: null, fps: null });
    expect(() => parseRunInfo('{"map_size": [600, 400], "generator": "x"}')).toThrow(/map/);
  });
});

describe("loadRun on a run written by the Python generator", () => {
  const run = loadRun(RUN_DIR);

  it("reads every record, in order, with the scripted jam", () => {
    const expected = Number(process.env.ORIENTYR_RUN_FRAMES ?? (FRESH ? run.records.length : 10));
    expect(run.records).toHaveLength(expected);
    expect(run.info.generator).toBe("fake_run");
    // TRACKING → LOST → RELOCALIZING → TRACKING, exactly once
    const changes = run.records.map((r) => r.state).filter((s, i, a) => i === 0 || s !== a[i - 1]);
    expect(changes).toEqual(["TRACKING", "LOST", "RELOCALIZING", "TRACKING"]);
    for (const r of run.records) {
      expect(r.state === "LOST" ? r.fix : r.fix?.footprint.length).toBe(r.state === "LOST" ? null : 4);
    }
  });

  it("resolves the map relative to the run folder", () => {
    expect(path.isAbsolute(mapPath(RUN_DIR, run.info))).toBe(true);
    expect(mapPath(RUN_DIR, run.info)).toBe(path.resolve(RUN_DIR, run.info.map));
  });

  it("finds every frame inside the run folder and refuses paths that leave it", () => {
    for (const r of run.records) {
      const p = framePath(RUN_DIR, r.source);
      expect(p.startsWith(path.resolve(RUN_DIR) + path.sep)).toBe(true);
      if (FRESH) expect(existsSync(p), `${r.source} is missing`).toBe(true);
    }
    expect(() => framePath(RUN_DIR, "../run.json")).toThrow(RunFormatError);
    expect(() => framePath(RUN_DIR, "/etc/passwd")).toThrow(RunFormatError);
  });

  it("says which folder is wrong when there is no run", () => {
    expect(() => loadRun(path.join(RUN_DIR, "nope"))).toThrow(/nope/);
  });
});
