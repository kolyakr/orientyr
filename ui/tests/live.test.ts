// Live mode (M7-04), the viewer's half: a run that is still being written is followed record by record. The pieces
// are plain functions and route handlers, tested on a runs folder made for the test; the engine's half is
// tests/test_live.py in the repo root.
import { appendFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { GET as getLive, POST as postLive } from "@/app/api/live/route";
import { GLIDE_MAX_MS, POLL_MS, glideMs, nextPace, parseLiveStatus } from "@/lib/live";
import { RunFormatError, appendRunJsonl, parseRunJsonl } from "@/lib/run";
import { liveStatus, openRun, readTail } from "@/lib/run-server";

const BASE = { idx: 0, t: 0, source: "frames/f0.jpg", state: "TRACKING", x: 10, y: 20, vx: 1, vy: 0, cov: [[4, 0], [0, 4]], fix: null };
const NOWHERE = { ...BASE, state: "LOST", x: null, y: null, vx: null, vy: null, cov: null };
const rec = (idx: number, more: object = {}) => JSON.stringify({ ...BASE, idx, t: idx / 2, x: 10 + idx, ...more });
const INFO = JSON.stringify({ map: "../map.png", map_size: [600, 400], generator: "live.py", frame_size: [96, 96], fps: 2 });
const STATUS = { state: "running", frames: 2, fixes: 2, skipped: 0, fixes_per_s: 2, jam: false };

let root: string;
const made: string[] = [];
/** A run folder in the runs folder of this test: run.json, run.jsonl with `text`, and live.json when given. */
function runFolder(name: string, text: string, status: object | null = STATUS): string {
  const dir = path.join(root, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(path.join(dir, "run.json"), INFO);
  writeFileSync(path.join(dir, "run.jsonl"), text);
  if (status) writeFileSync(path.join(dir, "live.json"), JSON.stringify(status));
  made.push(dir);
  return dir;
}
const longAgo = (dir: string) => {
  const then = new Date(Date.now() - 10 * 60 * 1000);
  utimesSync(path.join(dir, "live.json"), then, then);
};

const before = { ...process.env };
beforeAll(() => {
  root = mkdtempSync(path.join(os.tmpdir(), "orientyr-live-"));
  process.env.ORIENTYR_RUNS = root;
  delete process.env.ORIENTYR_RUN;
});
afterEach(() => {
  for (const dir of made.splice(0)) rmSync(dir, { recursive: true, force: true });
});
afterAll(() => {
  rmSync(root, { recursive: true, force: true });
  process.env = before;
});

describe("appendRunJsonl", () => {
  const text = [JSON.stringify(NOWHERE), JSON.stringify({ ...NOWHERE, idx: 1, t: 0.5 }), rec(2), rec(3), JSON.stringify({ ...NOWHERE, idx: 4, t: 2 })];

  it("gives the same records in pieces as parseRunJsonl gives at once, wherever the file is cut", () => {
    const whole = parseRunJsonl(text.join("\n"));
    for (let cut = 0; cut <= text.length; cut++) {
      const first = appendRunJsonl([], text.slice(0, cut).join("\n"));
      expect(appendRunJsonl(first, text.slice(cut).join("\n") + "\n")).toEqual(whole);
    }
  });

  it("puts the frames before the first fix at that fix once it arrives", () => {
    const waiting = appendRunJsonl([], text.slice(0, 2).join("\n"));
    expect(waiting.map((r) => r.located)).toEqual([false, false]);
    const found = appendRunJsonl(waiting, text[2]);
    expect(found.map((r) => [r.located, r.x, r.y])).toEqual([[false, 12, 20], [false, 12, 20], [true, 12, 20]]);
  });

  it("returns a new list and leaves the records it was given as they were: React compares by identity", () => {
    const waiting = appendRunJsonl([], text.slice(0, 2).join("\n"));
    const copy = structuredClone(waiting);
    const found = appendRunJsonl(waiting, text.slice(2).join("\n"));
    expect(waiting).toEqual(copy);
    expect(found).not.toBe(waiting);
    const more = appendRunJsonl(found, rec(5));
    expect(more.slice(0, 5).every((r, i) => r === found[i])).toBe(true); // nothing to change: the same objects
    expect(appendRunJsonl(found, "\n \n")).toBe(found); // nothing new: the same list, so nothing renders again
  });

  it("keeps the frame order across the pieces, and counts lines from the records it already has", () => {
    const first = appendRunJsonl([], [rec(0), rec(1), rec(2)].join("\n"));
    expect(() => appendRunJsonl(first, rec(2))).toThrow(/line 4: records must be in frame order/);
    expect(() => appendRunJsonl(first, `${rec(3)}\n{"idx": 4`)).toThrow(RunFormatError);
  });
});

describe("readTail", () => {
  it("returns whole lines only: a line the engine is still writing waits for the next read", () => {
    const one = rec(0, { source: "frames/кадр.jpg" }); // not ASCII: the offset is in bytes, not in characters
    const dir = runFolder("tail", `${one}\n${rec(1)}\n{"idx": 2, "t"`);
    const first = readTail(dir, 0);
    expect(first.text).toBe(`${one}\n${rec(1)}\n`);
    expect(first.next).toBe(Buffer.byteLength(first.text));
    expect(first.reset).toBe(false);
    expect(readTail(dir, first.next)).toEqual({ text: "", next: first.next, reset: false });
    writeFileSync(path.join(dir, "run.jsonl"), `${one}\n${rec(1)}\n${rec(2)}\n`);
    expect(readTail(dir, first.next).text).toBe(`${rec(2)}\n`);
    appendFileSync(path.join(dir, "run.jsonl"), `${rec(3)}\n${rec(4)}\n`);
    const rest = readTail(dir, first.next);
    expect(parseRunJsonl(rest.text).map((r) => r.idx)).toEqual([2, 3, 4]);
    expect(rest.next).toBe(readFileSync(path.join(dir, "run.jsonl")).length);
  });

  it("starts over when the file is shorter than what was read: the run was started again", () => {
    const dir = runFolder("again", `${rec(0)}\n`);
    expect(readTail(dir, 100000)).toEqual({ text: `${rec(0)}\n`, next: Buffer.byteLength(`${rec(0)}\n`), reset: true });
  });

  it("reads nothing from a run that has no run.jsonl yet, or from a bad offset", () => {
    const dir = runFolder("early", "");
    rmSync(path.join(dir, "run.jsonl"));
    expect(readTail(dir, 0)).toEqual({ text: "", next: 0, reset: false });
    const some = runFolder("some", `${rec(0)}\n`);
    for (const from of [-5, Number.NaN, 1.5]) expect(readTail(some, from).text).toBe(`${rec(0)}\n`);
  });
});

describe("liveStatus", () => {
  it("parses live.json, and nothing else", () => {
    // these fields, whatever else the status carries (M7-05 adds the judge controls: tests/judge-controls.test.tsx)
    expect(parseLiveStatus(JSON.stringify(STATUS))).toMatchObject({ state: "running", frames: 2, fixes: 2, skipped: 0, fixesPerS: 2, jam: false });
    expect(parseLiveStatus(JSON.stringify({ state: "done" }))).toMatchObject({ state: "done", frames: 0, fixes: 0, skipped: 0, fixesPerS: null, jam: false });
    for (const bad of ["", "{", "[]", "null", JSON.stringify({ state: "flying" })]) expect(parseLiveStatus(bad)).toBeNull();
  });

  it("is live while the engine says running and keeps saying it", () => {
    expect(liveStatus(runFolder("on", ""))).toMatchObject({ live: true, state: "running", jam: false });
    expect(liveStatus(runFolder("jammed", "", { ...STATUS, jam: true }))).toMatchObject({ live: true, jam: true });
    expect(liveStatus(runFolder("starting", "", { state: "starting" }))).toMatchObject({ live: true });
  });

  it("is not live when the flight is over, was stopped, or the engine went silent", () => {
    expect(liveStatus(runFolder("done", "", { ...STATUS, state: "done" }))).toMatchObject({ live: false, state: "done" });
    expect(liveStatus(runFolder("stopped", "", { ...STATUS, state: "stopped" }))).toMatchObject({ live: false });
    const dead = runFolder("dead", "");
    longAgo(dead); // "running", but live.json was last written ten minutes ago: the engine was killed
    expect(liveStatus(dead)).toMatchObject({ live: false, state: "running" });
  });

  it("is null for a run that was never live, or whose live.json cannot be read", () => {
    expect(liveStatus(runFolder("replay", "", null))).toBeNull();
    const dir = runFolder("garbage", "");
    writeFileSync(path.join(dir, "live.json"), "{");
    expect(liveStatus(dir)).toBeNull();
  });
});

describe("openRun on a live run", () => {
  it("opens what is written so far, also in the middle of a line, and says where to read on", () => {
    const text = `${rec(0)}\n${rec(1)}\n`;
    runFolder("flying", `${text}{"idx": 2,`);
    const opened = openRun("flying");
    expect(opened.error).toBeNull();
    expect(opened.run!.records.map((r) => r.idx)).toEqual([0, 1]);
    expect(opened.live).toMatchObject({ next: Buffer.byteLength(text), jam: false }); // and the judge controls (M7-05)
    expect(opened.waiting).toBe(false);
  });

  it("waits for the first frame of a live run instead of calling the run empty", () => {
    runFolder("warming", "", { state: "starting" });
    const opened = openRun("warming");
    expect(opened.run).toBeNull();
    expect(opened.waiting).toBe(true);
    expect(opened.error).toMatch(/waiting for (its|the) first frame/);
  });

  it("opens a finished run exactly as before: no live state, and a broken file is still an error", () => {
    runFolder("landed", `${rec(0)}\n${rec(1)}\n`, { ...STATUS, state: "done" });
    expect(openRun("landed")).toMatchObject({ live: null, waiting: false, error: null });
    runFolder("cut", `${rec(0)}\n{"idx": 1,`, null);
    expect(openRun("cut")).toMatchObject({ run: null, live: null, waiting: false });
    expect(openRun("cut").error).toMatch(/line 2: invalid JSON/);
    runFolder("none", "", null);
    expect(openRun("none").error).toMatch(/has no records/);
  });
});

describe("/api/live", () => {
  const get = (query: string) => getLive(new Request(`http://localhost/api/live?${query}`));
  const post = (query: string, body: unknown) => postLive(new Request(`http://localhost/api/live?${query}`, {
    method: "POST", body: typeof body === "string" ? body : JSON.stringify(body),
  }));

  it("hands the viewer the records made since its last question", async () => {
    const dir = runFolder("air", `${rec(0)}\n${rec(1)}\n`);
    const first = await (await get("run=air&from=0")).json();
    expect(first).toMatchObject({ live: true, state: "running", jam: false, reset: false, text: `${rec(0)}\n${rec(1)}\n`, fixesPerS: 2 });
    appendFileSync(path.join(dir, "run.jsonl"), `${rec(2)}\n`);
    const second = await (await get(`run=air&from=${first.next}`)).json();
    expect(second.text).toBe(`${rec(2)}\n`);
    expect((await (await get("run=air&from=nonsense")).json()).text).toContain('"idx":0');
  });

  it("says when the flight is over, with the last records", async () => {
    runFolder("over", `${rec(0)}\n`, { ...STATUS, state: "done" });
    expect(await (await get("run=over")).json()).toMatchObject({ live: false, state: "done", text: `${rec(0)}\n` });
    runFolder("old", `${rec(0)}\n`, null);
    expect(await (await get("run=old")).json()).toMatchObject({ live: false, state: null });
  });

  it("answers 404 for a run that is not in the runs folder", async () => {
    runFolder("air", "");
    for (const q of ["run=nope", "run=..", "run=air/../air"]) {
      expect((await get(q)).status).toBe(404);
      expect((await post(q, { jam: true })).status).toBe(404);
    }
  });

  it("switches the jam on and off through control.json, which the engine reads before every frame", async () => {
    const dir = runFolder("air", `${rec(0)}\n`);
    const on = await post("run=air", { jam: true });
    expect(on.status).toBe(200);
    expect(await on.json()).toEqual({ jam: true });
    expect(JSON.parse(readFileSync(path.join(dir, "control.json"), "utf-8"))).toEqual({ jam: true });
    // the viewer shows the order at once, before the engine's next frame confirms it in live.json
    expect((await (await get("run=air")).json()).jam).toBe(true);
    await post("run=air", { jam: false });
    expect(JSON.parse(readFileSync(path.join(dir, "control.json"), "utf-8"))).toEqual({ jam: false });
    expect((await (await get("run=air")).json()).jam).toBe(false);
  });

  it("refuses an order that is not a jam switch, and any order to a flight that is over", async () => {
    const dir = runFolder("air", "");
    for (const body of ["{", "[]", { jam: "yes" }, {}, { jam: 1 }]) expect((await post("run=air", body)).status).toBe(400);
    expect(existsSync(path.join(dir, "control.json"))).toBe(false);
    const done = runFolder("over", "", { ...STATUS, state: "done" });
    expect((await post("run=over", { jam: true })).status).toBe(409);
    expect(existsSync(path.join(done, "control.json"))).toBe(false);
  });
});

describe("the glide between fixes", () => {
  it("learns the pace of the fixes from when they arrive, without jumping at one late record", () => {
    expect(nextPace(null, 500)).toBe(500);
    const slower = nextPace(500, 900);
    expect(slower).toBeGreaterThan(500);
    expect(slower).toBeLessThan(900);
    let pace: number | null = null;
    for (let i = 0; i < 20; i++) pace = nextPace(pace, 1000);
    expect(pace).toBeCloseTo(1000, 0);
    expect(nextPace(500, 60_000)).toBeLessThanOrEqual(GLIDE_MAX_MS); // a jam's silence is not the new pace
    expect(nextPace(null, 0)).toBeGreaterThan(0);
    expect(nextPace(500, Number.NaN)).toBe(500);
  });

  it("is longer than the wait for the next fix, so the drone is still moving when it arrives", () => {
    // from 4 fixes a second down to the spec's floor of one a second: the marker never stops between two fixes.
    // The next record is seen at most one poll after it is made, hence the pace plus one poll.
    for (const pace of [250, 500, 700, 1000]) {
      expect(glideMs(pace)).toBeGreaterThanOrEqual(pace + POLL_MS);
      expect(glideMs(pace)).toBeLessThanOrEqual(pace * 2 + POLL_MS); // and not so slow that it trails far behind
    }
    expect(glideMs(60_000)).toBe(GLIDE_MAX_MS);
    expect(glideMs(null)).toBe(glideMs(1000)); // nothing known yet: the spec's floor, one fix a second
    expect(POLL_MS).toBeLessThanOrEqual(500);  // well under the one second between fixes the spec allows
  });
});
