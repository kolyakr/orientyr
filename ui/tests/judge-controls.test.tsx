// Judge controls (M7-05): the four orders the console sends to a live flight (jam camera, spoof GPS, map year, new
// flight) and what it draws for them. The orders go into the run's control.json, which the engine reads before every
// frame (src/dronenav/live.py; its half is tests/test_judge_controls.py). What a press does to the picture in a
// browser is for a person to check, see the PR.
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { renderToStaticMarkup } from "react-dom/server";
import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { GET as getLive, POST as postLive } from "@/app/api/live/route";
import { Console } from "@/components/Console";
import { parseLiveStatus } from "@/lib/live";
import { loadRun, openRun } from "@/lib/run-server";

const BASE = { idx: 0, t: 0, source: "frames/f0.jpg", state: "TRACKING", x: 10, y: 20, vx: 1, vy: 0, cov: [[4, 0], [0, 4]], fix: null };
const rec = (idx: number) => JSON.stringify({ ...BASE, idx, t: idx / 2, x: 10 + idx });
const INFO = JSON.stringify({ map: "../map.png", map_size: [600, 400], generator: "live.py", frame_size: [96, 96], fps: 2 });
const STATUS = {
  state: "running", frames: 2, fixes: 2, skipped: 0, fixes_per_s: 2, jam: false, spoof: false,
  map: "2022", maps: ["2017", "2022", "2025"], flight: "mild/2022", flights: ["mild/2022", "wild/2025"], speed: 1, flight_no: 0,
};

let root: string;
const made: string[] = [];
function runFolder(name: string, status: object | null = STATUS): string {
  const dir = path.join(root, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(path.join(dir, "run.json"), INFO);
  writeFileSync(path.join(dir, "run.jsonl"), `${rec(0)}\n${rec(1)}\n`);
  if (status) writeFileSync(path.join(dir, "live.json"), JSON.stringify(status));
  made.push(dir);
  return dir;
}
const control = (dir: string) => JSON.parse(readFileSync(path.join(dir, "control.json"), "utf-8"));

const before = { ...process.env };
beforeAll(() => {
  root = mkdtempSync(path.join(os.tmpdir(), "orientyr-judge-"));
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

const get = async (query: string) => (await getLive(new Request(`http://localhost/api/live?${query}`))).json();
const post = (query: string, body: unknown) => postLive(new Request(`http://localhost/api/live?${query}`, {
  method: "POST", body: typeof body === "string" ? body : JSON.stringify(body),
}));

describe("live.json with the judge controls", () => {
  it("carries what is switched on and what can be chosen", () => {
    expect(parseLiveStatus(JSON.stringify(STATUS))).toMatchObject({
      spoof: false, map: "2022", maps: ["2017", "2022", "2025"], flight: "mild/2022", flights: ["mild/2022", "wild/2025"], speed: 1, flightNo: 0,
    });
  });

  it("reads a status of the engine before M7-05 as one with nothing to choose", () => {
    const old = parseLiveStatus(JSON.stringify({ state: "running", frames: 1, fixes: 1, skipped: 0, fixes_per_s: null, jam: true }));
    expect(old).toMatchObject({ jam: true, spoof: false, map: null, maps: [], flight: null, flights: [], speed: null, flightNo: 0 });
    // a list that is not a list of names is no choice at all
    expect(parseLiveStatus(JSON.stringify({ ...STATUS, maps: "2022", flights: [1, "a"] }))).toMatchObject({ maps: [], flights: ["a"] });
  });
});

describe("/api/live: the orders", () => {
  it("tells the viewer what is on and what it can choose", async () => {
    runFolder("air");
    expect(await get("run=air")).toMatchObject({
      live: true, jam: false, spoof: false, map: "2022", maps: ["2017", "2022", "2025"],
      flight: "mild/2022", flights: ["mild/2022", "wild/2025"], speed: 1, flightNo: 0,
    });
  });

  it("starts and stops the GPS spoof, and leaves the jam as it was ordered", async () => {
    const dir = runFolder("air");
    await post("run=air", { jam: true });
    const on = await post("run=air", { spoof: true });
    expect(on.status).toBe(200);
    expect(await on.json()).toEqual({ spoof: true });
    expect(control(dir)).toEqual({ jam: true, spoof: true });
    // shown at once, before the engine's next frame confirms it in live.json
    expect(await get("run=air")).toMatchObject({ jam: true, spoof: true });
    await post("run=air", { spoof: false });
    expect(control(dir)).toEqual({ jam: true, spoof: false });
    expect(await get("run=air")).toMatchObject({ jam: true, spoof: false });
  });

  it("switches the map to another year the engine offers, and to no other", async () => {
    const dir = runFolder("air");
    const res = await post("run=air", { map: "2017" });
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ map: "2017" });
    expect(control(dir)).toEqual({ map: "2017" });
    expect((await get("run=air")).map).toBe("2017");
    for (const map of ["1999", "", 2017, null, "../2017"]) expect((await post("run=air", { map })).status).toBe(400);
    expect(control(dir)).toEqual({ map: "2017" });
  });

  it("orders a new flight: each order has a new id, and the chaos of the old flight is switched off", async () => {
    const dir = runFolder("air");
    await post("run=air", { jam: true });
    await post("run=air", { spoof: true });
    const res = await post("run=air", { flight: "wild/2025", speed: 2 });
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ flight: "wild/2025", speed: 2, id: 1 });
    expect(control(dir)).toEqual({ jam: false, spoof: false, flight: { name: "wild/2025", speed: 2, id: 1 } });
    // the same flight again is a new order; without a speed it flies at the speed of the flight in the air
    expect(await (await post("run=air", { flight: "wild/2025" })).json()).toEqual({ flight: "wild/2025", speed: 1, id: 2 });
    expect(control(dir).flight).toEqual({ name: "wild/2025", speed: 1, id: 2 });
    expect(await get("run=air")).toMatchObject({ flight: "wild/2025", jam: false, spoof: false });
  });

  it("refuses a flight the engine does not offer, a speed that is not one, and a body with no order or several", async () => {
    const dir = runFolder("air");
    const bad = [{ flight: "nope" }, { flight: "wild/2025", speed: 0 }, { flight: "wild/2025", speed: -1 }, { flight: "wild/2025", speed: "2" },
                 { flight: "wild/2025", speed: 100 }, { flight: 3 }, { speed: 2 }, { spoof: "yes" }, { jam: true, spoof: true }, {}, "[]", "{"];
    for (const body of bad) expect((await post("run=air", body)).status, JSON.stringify(body)).toBe(400);
    expect(existsSync(path.join(dir, "control.json"))).toBe(false);
  });

  it("refuses every order to a flight that is over, or to an engine that offers no such choice", async () => {
    const done = runFolder("over", { ...STATUS, state: "done" });
    for (const body of [{ spoof: true }, { map: "2017" }, { flight: "wild/2025" }]) expect((await post("run=over", body)).status).toBe(409);
    expect(existsSync(path.join(done, "control.json"))).toBe(false);
    runFolder("old", { state: "running", frames: 1, fixes: 1, skipped: 0, fixes_per_s: null, jam: false });
    expect((await post("run=old", { map: "2017" })).status).toBe(400);
    expect((await post("run=old", { flight: "mild/2022" })).status).toBe(400);
  });

  it("opens a live run with its switches and choices", () => {
    runFolder("air", { ...STATUS, spoof: true });
    expect(openRun("air").live).toMatchObject({ jam: false, spoof: true, map: "2022", maps: ["2017", "2022", "2025"], flight: "mild/2022", flights: ["mild/2022", "wild/2025"], speed: 1, flightNo: 0 });
  });
});

// ---- the page ---------------------------------------------------------------------------------------------------

const run = loadRun(path.join(__dirname, "fixtures", "run"));
const LIVE = { next: 0, jam: false, spoof: false, map: "2022", maps: ["2017", "2022", "2025"], flight: "mild/2022", flights: ["mild/2022", "wild/2025"], speed: 2, flightNo: 0 };
const gps = (state: string, gap: number | null, alarm = false) => ({ x: 100, y: 100, lon: null, lat: null, gap_px: gap, d2: 1, state, alarm, streak: 0 });
/** The fixture run with a GPS reading on its last record (the one a live view shows). */
const withGps = (g: object | null) => ({ ...run, records: run.records.map((r, i) => (g && i === run.records.length - 1 ? { ...r, extra: { ...r.extra, gps: g } } : r)) });
const page = (live: object | null, g: object | null = null) =>
  renderToStaticMarkup(<Console run={withGps(g)} initialFrame={run.records.length - 1} runName="air" live={live as never} />);
const section = (markup: string, name: string) => {
  const from = markup.slice(markup.indexOf(`data-part="${name}"`));
  return from.slice(0, from.indexOf("</section>"));
};
const tag = (markup: string, part: string) => new RegExp(`<[a-z]+[^>]*data-part="${part}"[^>]*>`).exec(markup)?.[0] ?? "";

describe("the mission controls on a live run", () => {
  it("offers the four controls, all of them working", () => {
    const controls = section(page(LIVE), "controls");
    for (const part of ["jam", "spoof", "new-flight", "flight", "flight-speed"]) {
      expect(tag(controls, part), part).not.toBe("");
      expect(tag(controls, part), part).not.toContain("disabled");
    }
    expect(controls).toMatch(/data-part="jam"[^>]*aria-pressed="false"[^>]*>Jam camera/);
    expect(controls).toMatch(/data-part="spoof"[^>]*aria-pressed="false"[^>]*>Spoof GPS/);
    expect(controls).toMatch(/data-part="new-flight"[^>]*>New flight/);
    expect(controls).not.toContain("Start mission"); // the placeholders of M7-04 are gone
    expect(controls).not.toContain("Change matcher");
  });

  it("shows what is switched on", () => {
    const controls = section(page({ ...LIVE, jam: true, spoof: true }), "controls");
    expect(controls).toMatch(/data-part="jam"[^>]*aria-pressed="true"[^>]*>Stop jamming/);
    expect(controls).toMatch(/data-part="spoof"[^>]*aria-pressed="true"[^>]*>Stop spoofing/);
  });

  it("offers every year of the map and marks the one in use", () => {
    const years = section(page(LIVE), "controls").match(/<button[^>]*data-map="[^"]*"[^>]*>[^<]*/g) ?? [];
    expect(years.map((b) => b.replace(/.*>/, ""))).toEqual(["2017", "2022", "2025"]);
    expect(years.map((b) => /aria-pressed="(\w+)"/.exec(b)?.[1])).toEqual(["false", "true", "false"]);
    for (const b of years) expect(b).not.toContain("disabled");
    // the picture of the map is asked for again when the year changes: the URL names it
    expect(page(LIVE)).toMatch(/api\/map\?run=air&amp;v=2022/);
    expect(page({ ...LIVE, map: "2017" })).toMatch(/api\/map\?run=air&amp;v=2017/);
  });

  it("offers every flight and a speed, with the one in the air chosen", () => {
    const controls = section(page(LIVE), "controls");
    const flights = /<select[^>]*data-part="flight"[^>]*>(.*?)<\/select>/.exec(controls)?.[1] ?? "";
    expect(flights.match(/<option[^>]*>[^<]*/g)?.map((o) => o.replace(/.*>/, ""))).toEqual(["mild/2022", "wild/2025"]);
    expect(flights).toMatch(/<option[^>]*selected=""[^>]*>mild\/2022/);
    const speeds = /<select[^>]*data-part="flight-speed"[^>]*>(.*?)<\/select>/.exec(controls)?.[1] ?? "";
    expect(speeds).toMatch(/<option[^>]*selected=""[^>]*>2×/);
  });

  it("with an engine that offers one map and one flight, the choice is that one", () => {
    const controls = section(page({ next: 0, jam: false }), "controls");
    expect(tag(controls, "jam")).not.toContain("disabled");
    expect(tag(controls, "spoof")).not.toContain("disabled");
    expect(tag(controls, "new-flight")).toContain("disabled"); // nothing to start: the engine named no flight
  });

  it("leaves a replay with every control inactive", () => {
    const controls = section(renderToStaticMarkup(<Console run={run} runName="air" />), "controls");
    const active = controls.match(/<(button|select)[^>]*>/g) ?? [];
    expect(active.length).toBeGreaterThanOrEqual(4);
    for (const el of active) expect(el).toContain("disabled");
    for (const part of ["jam", "spoof", "new-flight"]) expect(tag(controls, part)).not.toBe("");
  });
});

describe("the GPS check panel (the lie detector)", () => {
  it("says the receiver agrees with the camera, and by how much", () => {
    const panel = section(page(LIVE, gps("OK", 3.6)), "gps-check");
    expect(tag(page(LIVE, gps("OK", 3.6)), "gps-check")).toContain('data-gps="OK"');
    expect(panel).toContain("GPS agrees with the camera");
    expect(panel).toContain("4 px");
  });

  it("raises the alarm when the detector does", () => {
    const markup = page({ ...LIVE, spoof: true }, gps("SPOOFED", 61.2, true));
    expect(tag(markup, "gps-check")).toContain('data-gps="SPOOFED"');
    expect(tag(markup, "gps-check")).toContain('role="alert"');
    const panel = section(markup, "gps-check");
    expect(panel).toContain("GPS spoofed");
    expect(panel).toContain("61 px");
  });

  it("says suspect before the alarm, and unchecked when there is nothing to compare", () => {
    expect(section(page(LIVE, gps("SUSPECT", 18)), "gps-check")).toContain("GPS suspect");
    expect(tag(page(LIVE, gps("SUSPECT", 18)), "gps-check")).not.toContain('role="alert"');
    expect(section(page(LIVE, gps("NO_DATA", null)), "gps-check")).toContain("GPS not checked");
  });

  it("is not there on a run without a receiver", () => {
    expect(page(LIVE)).not.toContain('data-part="gps-check"');
    expect(renderToStaticMarkup(<Console run={run} runName="air" />)).not.toContain('data-part="gps-check"');
  });

  it("is there on a replay of a run that has one", () => {
    const markup = renderToStaticMarkup(<Console run={withGps(gps("SPOOFED", 50, true))} initialFrame={run.records.length - 1} runName="air" />);
    expect(tag(markup, "gps-check")).toContain('data-gps="SPOOFED"');
  });
});
