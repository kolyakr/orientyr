// The numbers the console draws: confidence, uncertainty, heading, the trail and the timeline.
import path from "node:path";
import { describe, expect, it } from "vitest";
import { loadRun } from "@/lib/run-server";
import type { RunRecord } from "@/lib/run";
import { type Camera, type Rect, confidence, events, fitRoute, footprintSize, headingOf, inViewBox, routeBox, sigmaPx, stateRuns, trail } from "@/lib/view";

const run = loadRun(process.env.ORIENTYR_RUN ?? path.join(__dirname, "fixtures", "run"));
const recs = run.records;
const firstOf = (state: string) => recs.findIndex((r) => r.state === state);

describe("confidence", () => {
  it("is 0 without an accepted fix and within (0, 1] with one", () => {
    for (const r of recs) {
      const c = confidence(r);
      if (r.fix?.accepted) expect(c).toBeGreaterThan(0);
      else expect(c).toBe(0);
      expect(c).toBeLessThanOrEqual(1);
    }
  });

  it("grows with the inliers", () => {
    const r = recs.find((x) => x.fix?.accepted)!;
    const withInliers = (n: number) => confidence({ ...r, fix: { ...r.fix!, inliers: n } });
    expect(withInliers(40)).toBeLessThan(withInliers(150));
    expect(withInliers(150)).toBeLessThan(withInliers(300));
    expect(withInliers(5000)).toBe(1);
  });
});

describe("sigmaPx", () => {
  it("is the larger standard deviation of the covariance", () => {
    expect(sigmaPx([[4, 0], [0, 9]])).toBeCloseTo(3, 9);
    expect(sigmaPx([[2, 1], [1, 2]])).toBeCloseTo(Math.sqrt(3), 9); // eigenvalues 1 and 3
  });

  it("grows through the jam", () => {
    const lost = recs.filter((r) => r.state === "LOST").map((r) => sigmaPx(r.cov));
    expect(lost.length).toBeGreaterThan(1);
    for (let i = 1; i < lost.length; i++) expect(lost[i]).toBeGreaterThan(lost[i - 1]);
  });
});

describe("headingOf", () => {
  it("is the fix heading when there is one, else the direction of the velocity (clockwise from map up)", () => {
    const tracked = recs[0];
    expect(headingOf(tracked)).toBe(tracked.fix!.heading_deg);
    const coasting = { ...tracked, fix: null };
    expect(headingOf({ ...coasting, vx: 0, vy: -5 })).toBeCloseTo(0, 9); // map y points down: up is -y
    expect(headingOf({ ...coasting, vx: 5, vy: 0 })).toBeCloseTo(90, 9);
    expect(headingOf({ ...coasting, vx: 0, vy: 0 })).toBeNull();
  });
});

describe("trail", () => {
  it("has one segment per step up to the current frame, dead-reckoned ones marked", () => {
    expect(trail(recs, 0)).toEqual([]);
    const upTo = recs.length - 1;
    const segs = trail(recs, upTo);
    expect(segs).toHaveLength(upTo);
    segs.forEach((s, i) => {
      expect([s.x1, s.y1, s.x2, s.y2]).toEqual([recs[i].x, recs[i].y, recs[i + 1].x, recs[i + 1].y]);
      expect(s.coasting).toBe(!recs[i + 1].fix?.accepted);
      expect(s.confidence).toBe(confidence(recs[i + 1]));
    });
    expect(trail(recs, firstOf("LOST"))).toHaveLength(firstOf("LOST"));
  });
});

describe("timeline", () => {
  it("groups the frames into runs of one state that cover the whole flight", () => {
    const runs = stateRuns(recs);
    expect(runs.map((r) => r.state)).toEqual(["TRACKING", "LOST", "RELOCALIZING", "TRACKING"]);
    expect(runs[0].from).toBe(0);
    expect(runs.at(-1)!.to).toBe(recs.length - 1);
    for (let i = 1; i < runs.length; i++) expect(runs[i].from).toBe(runs[i - 1].to + 1);
  });

  it("lists the captions of the state changes", () => {
    const ev = events(recs);
    expect(ev.map((e) => e.frame)).toEqual([firstOf("LOST"), firstOf("RELOCALIZING"), recs.findLastIndex((r, i) => i > 0 && r.state === "TRACKING" && recs[i - 1].state !== "TRACKING")]);
    expect(ev[0].text).toMatch(/jammed/);
  });
});

/**
 * The whole-route camera. Screen px = map px * scale + (x, y). The numbers below are the real layouts:
 * a 1512 px wide window, panels on the left and on the right, the dock under the map.
 */
describe("camera", () => {
  const at = (x: number, y: number) => ({ x, y, fix: null }) as unknown as RunRecord;
  const view: Rect = { x: -12, y: -12, width: 1512, height: 945 };   // the window, seen from the stage's corner
  const free: Rect = { x: 248, y: 50, width: 880, height: 780 };     // the stage without the panels over it
  const EPS = 1e-6; // floating point only: the camera is exact arithmetic

  const onScreen = (box: Rect, cam: Camera): Rect =>
    ({ x: box.x * cam.scale + cam.x, y: box.y * cam.scale + cam.y, width: box.width * cam.scale, height: box.height * cam.scale });
  const expectInside = (inner: Rect, outer: Rect) => {
    expect(inner.x).toBeGreaterThanOrEqual(outer.x - EPS);
    expect(inner.y).toBeGreaterThanOrEqual(outer.y - EPS);
    expect(inner.x + inner.width).toBeLessThanOrEqual(outer.x + outer.width + EPS);
    expect(inner.y + inner.height).toBeLessThanOrEqual(outer.y + outer.height + EPS);
  };
  /** The route box touches the free area on the axis that limits the zoom: a bigger scale would not fit. */
  const expectTight = (box: Rect, cam: Camera) => {
    const s = onScreen(box, cam);
    expect(Math.min(free.width - s.width, free.height - s.height)).toBeCloseTo(0, 6);
  };

  it("boxes the whole route with a margin, and never leaves the map for the margin", () => {
    const route = [at(1000, 800), at(3000, 1200), at(2000, 2600)];
    expect(routeBox(route, [4000, 4000], 300)).toEqual({ x: 700, y: 500, width: 2600, height: 2400 });
    // near the top-left corner the margin is cut at the map edge
    expect(routeBox([at(100, 50), at(900, 700)], [4000, 4000], 300)).toEqual({ x: 0, y: 0, width: 1200, height: 1000 });
    // a point that drifted off the map (dead reckoning) is still inside the box
    const off = routeBox([at(-40, 500), at(900, 700)], [4000, 4000], 300);
    expect(off.x).toBe(-40);
    // one point, no margin: the box is not empty, so the scale stays finite
    const dot = routeBox([at(500, 500)], [4000, 4000], 0);
    expect(dot.width).toBeGreaterThan(0);
    expect(dot.height).toBeGreaterThan(0);
  });

  it("takes the footprint size from the accepted fixes of the run", () => {
    const sizes = recs.filter((r) => r.fix?.accepted).map((r) => {
      const xs = r.fix!.footprint.map((p) => p[0]), ys = r.fix!.footprint.map((p) => p[1]);
      return Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    });
    const size = footprintSize(recs);
    expect(size).toBeGreaterThanOrEqual(Math.min(...sizes));
    expect(size).toBeLessThanOrEqual(Math.max(...sizes));
    expect(footprintSize([at(1, 1)])).toBe(0); // no fix at all: the caller picks a margin
  });

  it("square map, route across most of it: the route fits the free area, the map covers the height", () => {
    const map: [number, number] = [3936, 3936];
    const box = routeBox([at(594, 571), at(3713, 3116)], map, 340);
    const cam = fitRoute(box, map, free, view);
    expectInside(onScreen(box, cam), free);
    expectTight(box, cam);
    // 3936 px of map at this scale are narrower than the window: the map cannot cover the width, its edges show
    expect(map[0] * cam.scale).toBeLessThan(view.width);
    // but it is taller than the window, and then no edge shows above or below
    expect(map[1] * cam.scale).toBeGreaterThan(view.height);
    expect(cam.y).toBeLessThanOrEqual(view.y + EPS);
    expect(map[1] * cam.scale + cam.y).toBeGreaterThanOrEqual(view.y + view.height - EPS);
  });

  it("wide map, route along it: the route fits, the map stays a band inside the window", () => {
    const map: [number, number] = [3645, 1845];
    const box = routeBox([at(3400, 300), at(130, 483), at(1800, 1548)], map, 110);
    const cam = fitRoute(box, map, free, view);
    expectInside(onScreen(box, cam), free);
    expectTight(box, cam);
    expect(map[1] * cam.scale).toBeLessThan(view.height);
    // the band is centred on the route, in the free area: not pushed against the top or the bottom
    const s = onScreen(box, cam);
    expect(s.y - free.y).toBeCloseTo(free.y + free.height - (s.y + s.height), 6);
  });

  it("small route in the middle: zoomed in, and the map covers the whole window", () => {
    const map: [number, number] = [3936, 3936];
    const box = routeBox([at(1800, 1900), at(2300, 2200)], map, 200);
    const cam = fitRoute(box, map, free, view);
    expectInside(onScreen(box, cam), free);
    expectTight(box, cam);
    expectInside(view, onScreen({ x: 0, y: 0, width: map[0], height: map[1] }, cam));
  });

  it("route near an edge: the route stays in the free area, even if the map edge then shows", () => {
    const map: [number, number] = [3936, 3936];
    // a short route in the top-left corner of the map: to cover the window the map would have to move left,
    // under the left panels, and the route with it. The route wins.
    const box = routeBox([at(150, 120), at(700, 500)], map, 200);
    const cam = fitRoute(box, map, free, view);
    expectInside(onScreen(box, cam), free);
    expectTight(box, cam);
    expect(cam.x).toBeGreaterThan(view.x); // the left map edge is inside the window
    // and the map is pulled as far towards covering as the route allows: the box touches the free area's left side
    expect(onScreen(box, cam).x).toBeCloseTo(free.x, 6);
    // the far sides are covered: the map reaches past the right and the bottom of the window
    expect(map[0] * cam.scale + cam.x).toBeGreaterThanOrEqual(view.x + view.width);
    expect(map[1] * cam.scale + cam.y).toBeGreaterThanOrEqual(view.y + view.height);
  });

  it("is written for an svg whose viewBox is the map, fitted into its cell and centred", () => {
    const map: [number, number] = [3645, 1845];
    const cell = { width: 1136, height: 833 };
    const cam: Camera = { scale: 0.4, x: -120.5, y: 33 };
    const t = inViewBox(cam, map, cell);
    // what the browser does: the group transform in map units, then the viewBox fit (meet, centred)
    const fit = Math.min(cell.width / map[0], cell.height / map[1]);
    const off = [(cell.width - map[0] * fit) / 2, (cell.height - map[1] * fit) / 2];
    for (const [px, py] of [[0, 0], [3645, 1845], [1000, 400]]) {
      expect((px * t.scale + t.x) * fit + off[0]).toBeCloseTo(px * cam.scale + cam.x, 6);
      expect((py * t.scale + t.y) * fit + off[1]).toBeCloseTo(py * cam.scale + cam.y, 6);
    }
  });
});
