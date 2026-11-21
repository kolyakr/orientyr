// Live mode (M7-04) on the page: a run that is being written opens at its newest record, the drone glides for as
// long as the next fix takes to arrive, and the jam switch works. Rendered on the server, as Next does for the
// first paint; what happens after that in a browser (the polling) is for a person to check, see the PR.
import path from "node:path";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Console } from "@/components/Console";
import { glideMs } from "@/lib/live";
import { loadRun } from "@/lib/run-server";

const run = loadRun(path.join(__dirname, "fixtures", "run"));
const last = run.records.at(-1)!;
const live = (jam = false, frame = 0) => renderToStaticMarkup(<Console run={run} initialFrame={frame} runName="air" live={{ next: 0, jam }} />);
const replay = (frame = 0) => renderToStaticMarkup(<Console run={run} initialFrame={frame} runName="air" />);
const section = (markup: string, name: string) => {
  const from = markup.slice(markup.indexOf(`data-part="${name}"`));
  return from.slice(0, from.indexOf("</section>"));
};
const droneGlide = (markup: string) => Number(/data-part="drone"[^>]*transition:\s*transform (\d+)ms linear/.exec(markup)?.[1]);

describe("Console on a live run", () => {
  it("opens at the newest record, whatever frame the address names", () => {
    const markup = live(false, 3);
    expect(markup).toContain(encodeURIComponent(last.source));
    expect(markup).toMatch(new RegExp(`data-part="state-badge"[^>]*data-state="${last.state}"`));
    expect(markup).not.toContain("Flight complete"); // the newest record of a live run is not the end of the flight
  });

  it("glides the drone to each fix for as long as the next one takes: the marker interpolates between fixes", () => {
    // before any record has arrived the pace is the run's own: 1 / fps seconds between frames
    expect(droneGlide(live())).toBe(Math.round(glideMs(1000 / run.info.fps!)));
    expect(droneGlide(live())).toBeGreaterThan(1000 / run.info.fps!);
    expect(droneGlide(replay(3))).toBe(250); // a replay at rest moves as before
  });

  it("marks the page as live and offers the way back to the newest record", () => {
    expect(live()).toMatch(/<main[^>]*data-live/);
    expect(live()).toMatch(/data-part="live"[^>]*aria-pressed="true"/);
    expect(replay()).not.toMatch(/<main[^>]*data-live/);
    expect(replay()).not.toContain('data-part="live"');
  });

  it("makes the jam switch work (the other controls: tests/judge-controls.test.tsx)", () => {
    const jam = /<button[^>]*data-part="jam"[^>]*>[^<]*/.exec(section(live(), "controls"))![0];
    expect(jam).not.toContain("disabled");
    expect(jam).toContain('aria-pressed="false"');
    expect(jam).toContain("Jam camera");
    const jammed = section(live(true), "controls");
    expect(jammed).toMatch(/aria-pressed="true"[^>]*>Stop jamming/);
  });

  it("leaves a replay as it was: every mission control inactive", () => {
    const buttons = section(replay(), "controls").match(/<button[^>]*>/g) ?? [];
    expect(buttons.length).toBeGreaterThanOrEqual(3);
    for (const b of buttons) expect(b).toContain("disabled");
  });
});
