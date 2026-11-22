// Real runs differ: frames before the first fix have no position, `extra` may be missing or full of surprises,
// a frame image or the map may be gone, a run may hold one record or none. The viewer opens what it can and says
// plainly what it cannot; it never draws NaN and never throws. The runs are hand-made: tests/fixtures/runs.
import path from "node:path";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import Page from "@/app/page";
import { Console } from "@/components/Console";
import { RunFormatError, parseRunJsonl } from "@/lib/run";
import { loadRun } from "@/lib/run-server";
import { gpsOf, routeBox, searchOf, sigmaPx, trail, truthOf, windowOf } from "@/lib/view";

const RUNS = path.join(__dirname, "fixtures", "runs");
const load = (name: string) => loadRun(path.join(RUNS, name));
const part = (markup: string, name: string) => markup.includes(`data-part="${name}"`);

const BASE = { idx: 0, t: 0, source: "f.png", state: "TRACKING", x: 10, y: 20, vx: 1, vy: 0, cov: [[4, 0], [0, 4]], fix: null };
const NOWHERE = { ...BASE, state: "LOST", x: null, y: null, vx: null, vy: null, cov: null };
const FIX = {
  H: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], x: 10, y: 20, heading_deg: 0, scale: 1, footprint: [[0, 0], [20, 0], [20, 40], [0, 40]],
  inliers: 100, inlier_ratio: 0.5, reproj_px: 1, accepted: true, reject_reason: null,
};
const lines = (...recs: object[]) => recs.map((r) => JSON.stringify(r)).join("\n");

describe("parseRunJsonl on records the engine may write", () => {
  it("reads a frame with no position yet (x, y, vx, vy, cov null together) as not located", () => {
    const recs = parseRunJsonl(lines(NOWHERE, { ...NOWHERE, idx: 1, t: 1 }, { ...BASE, idx: 2, t: 2, x: 50, y: 60 }, { ...NOWHERE, idx: 3, t: 3 }));
    expect(recs.map((r) => r.located)).toEqual([false, false, true, false]);
    // it is drawn nowhere, but its numbers are still numbers: the place of the nearest located frame, standing still
    expect(recs[0]).toMatchObject({ x: 50, y: 60, vx: 0, vy: 0 });
    expect(recs[3]).toMatchObject({ x: 50, y: 60, vx: 0, vy: 0 });
    for (const r of recs) expect(Number.isFinite(sigmaPx(r.cov))).toBe(true);
  });

  it("reads a run that was never localized", () => {
    const recs = parseRunJsonl(lines(NOWHERE, { ...NOWHERE, idx: 1, t: 1 }));
    expect(recs.map((r) => r.located)).toEqual([false, false]);
    for (const r of recs) expect([r.x, r.y, r.vx, r.vy].every(Number.isFinite)).toBe(true);
  });

  it("refuses half a position, naming the line and the field", () => {
    expect(() => parseRunJsonl(lines(BASE, { ...BASE, idx: 1, y: null }))).toThrow(/line 2.*'y'/);
    expect(() => parseRunJsonl(lines({ ...NOWHERE, x: 5 }))).toThrow(RunFormatError);
  });

  it.each([
    ["a covariance that is not 2 x 2", { cov: [[1, 0]] }, /cov/],
    ["a covariance with a string in it", { cov: [[1, "a"], [0, 1]] }, /cov/],
    ["a fix whose footprint is not a list of points", { fix: { ...FIX, footprint: [[0, 0], "b", [1, 1]] } }, /footprint/],
    ["a fix with too few corners", { fix: { ...FIX, footprint: [[0, 0], [1, 1]] } }, /footprint/],
    ["a fix without its heading", { fix: { ...FIX, heading_deg: undefined } }, /heading_deg/],
    ["a fix whose inliers are not a number", { fix: { ...FIX, inliers: "many" } }, /inliers/],
    ["a fix that does not say if it was accepted", { fix: { ...FIX, accepted: undefined } }, /accepted/],
    ["an idx that is not a whole number", { idx: 1.5 }, /idx/],
  ])("refuses %s", (_what, change, reason) => {
    expect(() => parseRunJsonl(lines({ ...BASE, ...change }))).toThrow(RunFormatError);
    expect(() => parseRunJsonl(lines({ ...BASE, ...change }))).toThrow(reason);
  });

  it("takes `extra` of any shape and Windows line ends", () => {
    const recs = parseRunJsonl(`${JSON.stringify({ ...BASE, extra: [1, 2] })}\r\n${JSON.stringify({ ...BASE, idx: 1, extra: null })}\r\n`);
    expect(recs.map((r) => r.extra)).toEqual([{}, {}]);
  });
});

describe("what is read out of `extra` is checked, not trusted", () => {
  const hard = load("hard").records;
  const at = (idx: number) => hard.find((r) => r.idx === idx)!;

  it("reads the truth, the GPS reading and the search window when they are there", () => {
    expect(truthOf(at(4))).toEqual({ x: 125, y: 107 });
    expect(gpsOf(at(4))).toEqual({ x: 127, y: 110, state: "OK", alarm: false, gapPx: 3.6 });
    expect(gpsOf(at(6))).toMatchObject({ state: "SPOOFED", alarm: true });
    expect(gpsOf(at(8))).toEqual({ x: null, y: null, state: "NO_DATA", alarm: false, gapPx: null });
    expect(windowOf(at(4))).toEqual({ x: 70, y: 54, width: 110, height: 110 });
    expect(searchOf(at(4))).toBe("window");
  });

  it("gives null for what is missing, null or of the wrong kind", () => {
    for (const idx of [0, 9, 10]) {
      expect(truthOf(at(idx))).toBeNull();
      expect(gpsOf(at(idx))).toBeNull();
      expect(windowOf(at(idx))).toBeNull();
      expect(searchOf(at(idx))).toBeNull();
    }
    expect(windowOf(at(2))).toBeNull(); // predicted_window: null, the whole map was searched
    expect(windowOf({ ...at(4), extra: { predicted_window: [50, 50, 50, 10] } })).toBeNull(); // empty
  });
});

describe("the numbers skip frames without a position", () => {
  it("draws no route to or from them, and frames the route without them", () => {
    const hard = load("hard").records;
    const segs = trail(hard, hard.length - 1);
    expect(segs).toHaveLength(hard.filter((r) => r.located).length - 1);
    expect(segs[0]).toMatchObject({ x1: 100, y1: 100, x2: 124, y2: 108 });
    const never = load("nowhere");
    expect(trail(never.records, 2)).toEqual([]);
    expect(routeBox(never.records, never.info.map_size, 10)).toEqual({ x: 0, y: 0, width: 600, height: 400 });
  });
});

describe("the console on hard runs", () => {
  const html = (name: string, frame: number) => {
    const run = load(name);
    return renderToStaticMarkup(<Console run={run} initialFrame={frame} runName={name} runs={["hard", "one"]} mapMissing />);
  };

  it.each(["hard", "one", "nowhere", "nested/deep"])("draws every frame of '%s' with no NaN, undefined or [object …] on screen", (name) => {
    const n = load(name).records.length;
    for (let f = 0; f < n; f++) {
      // attribute values and text only: the markup has no scripts
      const markup = html(name, f);
      for (const bad of ["NaN", "undefined", "[object", "Infinity", ">null<", '"null"']) {
        expect(markup.includes(bad), `frame ${f} of ${name} shows ${bad}: …${markup.slice(Math.max(0, markup.indexOf(bad) - 120), markup.indexOf(bad) + 40)}…`).toBe(false);
      }
      for (const name of ["map", "hud", "timeline", "state-badge"]) expect(part(markup, name), `frame ${f}: no ${name}`).toBe(true);
    }
  });

  it("draws no drone, footprint or error area for a frame without a position, and says so", () => {
    const markup = html("hard", 0);
    for (const name of ["drone", "footprint", "sigma"]) expect(part(markup, name), name).toBe(false);
    expect(markup).toContain("No position yet");
    expect(part(html("hard", 2), "drone")).toBe(true);
    expect(html("hard", 2)).not.toContain("No position yet");
  });

  it("plays a run of one record: one frame, a timeline, nothing to step to", () => {
    const markup = html("one", 0);
    expect(markup).toMatch(/type="range"[^>]*max="0"/);
    expect(part(markup, "drone")).toBe(true);
    expect(markup.split('data-part="trail-segment"').length - 1).toBe(0);
  });

  it("says that the map image is missing instead of showing an empty stage without a word", () => {
    expect(part(html("hard", 2), "map-missing")).toBe(true);
    const run = load("hard");
    expect(part(renderToStaticMarkup(<Console run={run} runName="hard" />), "map-missing")).toBe(false);
  });

  it("has a place for a frame image that cannot be loaded", () => {
    // the <img> reports the failure in the browser; the markup must carry the part that then shows
    expect(html("hard", 4)).toMatch(/data-part="camera"/);
    expect(html("hard", 4)).toContain(`/api/frame?run=hard&amp;src=${encodeURIComponent("../outside.png")}`);
  });
});

describe("the page", () => {
  const before = { ...process.env };
  beforeEach(() => { process.env.ORIENTYR_RUNS = RUNS; delete process.env.ORIENTYR_RUN; });
  afterEach(() => {
    for (const k of ["ORIENTYR_RUNS", "ORIENTYR_RUN"]) {
      if (before[k] === undefined) delete process.env[k];
      else process.env[k] = before[k];
    }
  });
  const page = async (query: Record<string, string>) => renderToStaticMarkup(await Page({ searchParams: Promise.resolve(query) }));

  it("opens the run named in the URL, on the frame named in the URL", async () => {
    const markup = await page({ run: "hard", frame: "3" });
    expect(part(markup, "map")).toBe(true);
    expect(markup).toContain(`/api/frame?run=hard&amp;src=${encodeURIComponent("frames/frame_00004.png")}`);
    expect(markup).toContain("/api/map?run=hard");
  });

  it.each([
    ["empty", /no records/],
    ["broken", /line 2/],
    ["nope", /nope/],
    ["../../run", /not a run/],
  ])("shows why '%s' cannot be opened, and the runs that can", async (run, reason) => {
    const markup = await page({ run });
    expect(part(markup, "run-error")).toBe(true);
    expect(part(markup, "map")).toBe(false);
    expect(markup).toMatch(reason);
    expect(markup).toContain('href="/?run=hard"');
    expect(markup).toContain(`href="/?run=${encodeURIComponent("nested/deep")}"`);
  });

  it("ignores a frame that is not a number", async () => {
    expect(part(await page({ run: "one", frame: "abc" }), "drone")).toBe(true);
  });
});
