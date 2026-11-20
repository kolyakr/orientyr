// The numbers the console draws: confidence, uncertainty, heading, the trail and the timeline.
import path from "node:path";
import { describe, expect, it } from "vitest";
import { loadRun } from "@/lib/run-server";
import { confidence, events, headingOf, sigmaPx, stateRuns, trail } from "@/lib/view";

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
