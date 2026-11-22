// The console page has every part the issue names, in each tracker state. Rendered on the server, as Next
// does for the first paint, so no browser is needed. How it looks is for people to judge (PR screenshots).
import path from "node:path";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Console } from "@/components/Console";
import { loadRun } from "@/lib/run-server";

const run = loadRun(process.env.ORIENTYR_RUN ?? path.join(__dirname, "fixtures", "run"));
const firstOf = (state: string) => run.records.findIndex((r) => r.state === state);
const html = (frame: number) => renderToStaticMarkup(<Console run={run} initialFrame={frame} />);
const part = (markup: string, name: string) => markup.includes(`data-part="${name}"`);
const count = (markup: string, needle: string) => markup.split(needle).length - 1;

describe("Console", () => {
  it.each(["map", "trail", "footprint", "drone", "camera", "hud", "state-badge", "timeline", "controls", "events"])(
    "shows the %s while tracking", (name) => {
      expect(part(html(firstOf("LOST") - 1), name), `no data-part="${name}"`).toBe(true);
    });

  it("has a corner handle to resize the camera picture, at its default size on the first paint", () => {
    const markup = html(firstOf("LOST") - 1);
    expect(markup).toMatch(/<button[^>]*data-part="camera-resize"[^>]*aria-label="Resize the camera picture"/);
    expect(markup).not.toContain("--cam-w"); // the server knows no saved size: the default one
    expect(markup).not.toMatch(/data-part="camera-resize"[^>]*disabled/); // a view preference: it works on a replay too
  });

  it("draws the map at full map resolution and the trail up to the current frame", () => {
    const frame = firstOf("LOST") - 1;
    const markup = html(frame);
    const [w, h] = run.info.map_size;
    expect(markup).toContain(`viewBox="0 0 ${w} ${h}"`);
    expect(count(markup, 'data-part="trail-segment"')).toBe(frame);
  });

  it("shows the state, the frame and its camera image", () => {
    for (const state of ["TRACKING", "LOST", "RELOCALIZING"]) {
      const frame = firstOf(state);
      const markup = html(frame);
      expect(markup).toMatch(new RegExp(`data-part="state-badge"[^>]*data-state="${state}"`));
      expect(markup).toContain(encodeURIComponent(run.records[frame].source));
    }
  });

  it("has no footprint while lost, and marks a rejected fix instead of trusting it", () => {
    expect(part(html(firstOf("LOST")), "footprint")).toBe(false);
    expect(part(html(firstOf("LOST")), "drone")).toBe(true); // still drawn: dead-reckoned
    const rejected = run.records.findIndex((r) => r.fix && !r.fix.accepted);
    expect(part(html(rejected), "footprint")).toBe(false);
    expect(part(html(rejected), "rejected-fix")).toBe(true);
  });

  it("keeps the mission controls inactive: this is a mock-up", () => {
    const markup = html(0);
    const controls = markup.slice(markup.indexOf('data-part="controls"'));
    const buttons = controls.slice(0, controls.indexOf("</section>")).match(/<button[^>]*>/g) ?? [];
    expect(buttons.length).toBeGreaterThanOrEqual(3);
    for (const b of buttons) expect(b).toContain("disabled");
  });

  it("clamps a frame outside the run", () => {
    expect(html(9999)).toContain(encodeURIComponent(run.records.at(-1)!.source));
  });
});
