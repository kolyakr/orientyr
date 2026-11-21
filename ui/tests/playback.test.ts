// Playing and scrubbing, as numbers: which frame lies under the pointer, what a key does, how fast a frame passes.
import { describe, expect, it } from "vitest";
import { SPEEDS, frameAt, keyAction, nextSpeed, stepMs } from "@/lib/playback";

describe("frameAt: the frame under the pointer on the timeline", () => {
  // the timeline is 400 px wide from x = 100 and holds 10 frames: each bar is 40 px
  it("is the bar the pointer is on", () => {
    expect(frameAt(100, 100, 400, 10)).toBe(0);
    expect(frameAt(139, 100, 400, 10)).toBe(0);
    expect(frameAt(140, 100, 400, 10)).toBe(1);
    expect(frameAt(320, 100, 400, 10)).toBe(5);
    expect(frameAt(499, 100, 400, 10)).toBe(9);
  });

  it("stays in the run when the drag leaves the timeline", () => {
    expect(frameAt(-50, 100, 400, 10)).toBe(0);
    expect(frameAt(500, 100, 400, 10)).toBe(9);
    expect(frameAt(9000, 100, 400, 10)).toBe(9);
  });

  it("is frame 0 for a run of one frame and for a timeline with no width yet", () => {
    expect(frameAt(300, 100, 400, 1)).toBe(0);
    expect(frameAt(300, 100, 0, 10)).toBe(0);
  });
});

describe("keyAction", () => {
  const key = (code: string, more: object = {}) => keyAction({ code, ...more });

  it("keeps the keys of the mock-up", () => {
    expect(key("Space")).toEqual({ kind: "toggle" });
    expect(key("ArrowRight")).toEqual({ kind: "step", by: 1 });
    expect(key("ArrowLeft")).toEqual({ kind: "step", by: -1 });
    expect(key("KeyF")).toEqual({ kind: "follow" });
  });

  it("jumps to the ends with Home and End, and steps ten frames with Shift", () => {
    expect(key("Home")).toEqual({ kind: "jump", to: "start" });
    expect(key("End")).toEqual({ kind: "jump", to: "end" });
    expect(key("ArrowRight", { shiftKey: true })).toEqual({ kind: "step", by: 10 });
    expect(key("ArrowLeft", { shiftKey: true })).toEqual({ kind: "step", by: -10 });
  });

  it("changes the speed with , and . and opens the layers with L", () => {
    expect(key("Comma")).toEqual({ kind: "speed", by: -1 });
    expect(key("Period")).toEqual({ kind: "speed", by: 1 });
    expect(key("KeyL")).toEqual({ kind: "layers" });
    expect(key("Escape")).toEqual({ kind: "close" });
  });

  it("switches a layer with its digit", () => {
    expect(key("Digit1")).toEqual({ kind: "layer", index: 0 });
    expect(key("Digit8")).toEqual({ kind: "layer", index: 7 });
    expect(key("Digit0")).toBeNull();
  });

  it("leaves the browser's own shortcuts and the form controls alone", () => {
    expect(key("KeyF", { metaKey: true })).toBeNull(); // Cmd+F is the browser's search
    expect(key("KeyL", { ctrlKey: true })).toBeNull();
    expect(key("ArrowRight", { altKey: true })).toBeNull();
    for (const tag of ["SELECT", "TEXTAREA", "INPUT"]) {
      expect(key("ArrowRight", { tag, inputType: "text" })).toBeNull();
      expect(key("Space", { tag, inputType: "text" })).toBeNull();
      expect(key("Digit1", { tag, inputType: "text" })).toBeNull();
    }
    // the timeline's own slider is an input too: the keys keep working while it has the focus
    expect(key("Space", { tag: "INPUT", inputType: "range" })).toEqual({ kind: "toggle" });
    expect(key("KeyZ")).toBeNull();
  });
});

describe("speed", () => {
  it("offers half speed and the usual doublings", () => {
    for (const s of [0.5, 1, 2, 4]) expect(SPEEDS).toContain(s);
    expect([...SPEEDS]).toEqual([...SPEEDS].sort((a, b) => a - b));
  });

  it("moves one step up or down and stops at the ends", () => {
    expect(nextSpeed(1, 1)).toBe(2);
    expect(nextSpeed(1, -1)).toBe(0.5);
    expect(nextSpeed(SPEEDS[0], -1)).toBe(SPEEDS[0]);
    expect(nextSpeed(SPEEDS.at(-1)!, 1)).toBe(SPEEDS.at(-1));
    expect(nextSpeed(3, 1)).toBe(1); // not one of the speeds: back to real time
  });

  it("turns the run's fps and the speed into the time one frame stays on screen", () => {
    expect(stepMs(1, 1)).toBe(1000);
    expect(stepMs(1, 4)).toBe(250);
    expect(stepMs(25, 0.5)).toBe(80);
    // a run without a usable fps plays at one frame a second, and a step is never shorter than a screen refresh
    for (const fps of [null, 0, -3, Number.NaN]) expect(stepMs(fps, 2)).toBe(500);
    expect(stepMs(240, 8)).toBe(16);
  });
});
