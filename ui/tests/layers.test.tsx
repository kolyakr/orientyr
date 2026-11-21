// Layer toggles: which layers a run has, how the choice travels in the URL, and that a layer switched off is
// really not drawn. Also the run picker, which shares the URL with them.
import path from "node:path";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Console } from "@/components/Console";
import { DEFAULT_LAYERS, LAYERS, type LayerId, availableLayers, consoleUrl, parseLayers } from "@/lib/layers";
import { loadRun } from "@/lib/run-server";

const hard = loadRun(path.join(__dirname, "fixtures", "runs", "hard"));
const fake = loadRun(path.join(__dirname, "fixtures", "run")); // a fake run: its `extra` carries the truth, nothing else
const one = loadRun(path.join(__dirname, "fixtures", "runs", "one"));
const nowhere = loadRun(path.join(__dirname, "fixtures", "runs", "nowhere")); // no `extra` at all
const part = (markup: string, name: string) => markup.includes(`data-part="${name}"`);
const FRAME = 3; // idx 4 of the hard run: an accepted fix, the truth, a search window and a GPS reading

// the part of the page each layer draws
const PART: Record<LayerId, string> = {
  trail: "trail", points: "points", footprint: "footprint", sigma: "sigma", camera: "camera",
  window: "search-window", gps: "gps", truth: "truth",
};

describe("the layers", () => {
  it("are the ones the spec lists, plus what `extra` carries", () => {
    expect(LAYERS.map((l) => l.id)).toEqual(Object.keys(PART));
    for (const l of LAYERS) expect(l.label.length).toBeGreaterThan(2);
  });

  it("offer the truth, the GPS and the search window only for a run that has them", () => {
    expect(availableLayers(nowhere.records)).toEqual(["trail", "points", "footprint", "sigma", "camera"]);
    expect(availableLayers(fake.records)).toEqual(["trail", "points", "footprint", "sigma", "camera", "truth"]);
    expect(availableLayers(hard.records)).toEqual(LAYERS.map((l) => l.id));
  });

  it("are all on at first, except the truth: a check, not a part of the picture", () => {
    expect(DEFAULT_LAYERS).toEqual({ trail: true, points: true, footprint: true, sigma: true, camera: true, window: true, gps: true, truth: false });
  });
});

describe("the layers in the URL", () => {
  it("reads ?off= and ?on= against the defaults, and ignores names it does not know", () => {
    expect(parseLayers(undefined, undefined)).toEqual(DEFAULT_LAYERS);
    expect(parseLayers("points,sigma", "truth")).toEqual({ ...DEFAULT_LAYERS, points: false, sigma: false, truth: true });
    expect(parseLayers("nonsense,,trail ", "__proto__")).toEqual({ ...DEFAULT_LAYERS, trail: false });
  });

  it("writes only what differs from the defaults, and reads its own writing", () => {
    expect(consoleUrl({ run: null, frame: 0, layers: DEFAULT_LAYERS })).toBe("/");
    expect(consoleUrl({ run: "irpin", frame: 22, layers: DEFAULT_LAYERS })).toBe("/?run=irpin&frame=22");
    expect(consoleUrl({ run: "nested/deep", frame: 0, layers: { ...DEFAULT_LAYERS, points: false, sigma: false, truth: true } }))
      .toBe("/?run=nested%2Fdeep&off=points%2Csigma&on=truth");
    for (const id of LAYERS.map((l) => l.id)) {
      const layers = { ...DEFAULT_LAYERS, [id]: !DEFAULT_LAYERS[id] };
      const q = new URL(consoleUrl({ run: "x", frame: 1, layers }), "http://localhost").searchParams;
      expect(parseLayers(q.get("off"), q.get("on"))).toEqual(layers);
    }
  });
});

describe("the console draws the layers that are on", () => {
  const html = (layers = DEFAULT_LAYERS, run = hard, frame = FRAME) =>
    renderToStaticMarkup(<Console run={run} initialFrame={frame} runName="hard" runs={["fake", "hard", "nested/deep"]} initialLayers={layers} />);

  it("draws every default layer, and the truth once it is switched on", () => {
    const markup = html();
    for (const l of LAYERS) expect(part(markup, PART[l.id]), l.id).toBe(DEFAULT_LAYERS[l.id]);
    expect(part(html({ ...DEFAULT_LAYERS, truth: true }), "truth")).toBe(true);
  });

  it.each(LAYERS.map((l) => l.id))("does not draw '%s' when it is off, and still draws the rest", (id) => {
    const all = Object.fromEntries(LAYERS.map((l) => [l.id, true])) as Record<LayerId, boolean>;
    const markup = html({ ...all, [id]: false });
    for (const l of LAYERS) expect(part(markup, PART[l.id]), l.id).toBe(l.id !== id);
    for (const name of ["map", "drone", "hud", "timeline", "events"]) expect(part(markup, name), name).toBe(true);
  });

  it("keeps the numbered points, START and the rejected fix in their layers", () => {
    const all = Object.fromEntries(LAYERS.map((l) => [l.id, true])) as Record<LayerId, boolean>;
    expect(html(all)).toContain("START");
    expect(html({ ...all, points: false })).not.toContain("START");
    const rejected = hard.records.findIndex((r) => r.fix && !r.fix.accepted);
    expect(part(html(all, hard, rejected), "rejected-fix")).toBe(true);
    expect(part(html({ ...all, footprint: false }, hard, rejected), "rejected-fix")).toBe(false);
  });

  it("marks a spoofed GPS reading as an alarm", () => {
    const spoofed = hard.records.findIndex((r) => r.idx === 6);
    expect(html(DEFAULT_LAYERS, hard, spoofed)).toMatch(/data-part="gps"[^>]*data-gps="SPOOFED"/);
    expect(html(DEFAULT_LAYERS, hard, FRAME)).toMatch(/data-part="gps"[^>]*data-gps="OK"/);
  });

  it("has one switch per layer the run has, pressed when the layer is on", () => {
    const markup = html({ ...DEFAULT_LAYERS, sigma: false });
    expect(part(markup, "layers")).toBe(true);
    for (const l of LAYERS) {
      expect(markup).toMatch(new RegExp(`<button[^>]*data-layer="${l.id}"[^>]*aria-pressed="${l.id === "sigma" || l.id === "truth" ? "false" : "true"}"`));
    }
    const plain = html(DEFAULT_LAYERS, fake, 2);
    for (const id of ["window", "gps"]) expect(plain).not.toContain(`data-layer="${id}"`);
    for (const id of ["trail", "truth"]) expect(plain).toContain(`data-layer="${id}"`);
  });

  it("has a run picker with the runs, the open one chosen", () => {
    const markup = html();
    expect(part(markup, "run-picker")).toBe(true);
    for (const name of ["fake", "hard", "nested/deep"]) expect(markup).toMatch(new RegExp(`<option[^>]*value="${name}"`));
    expect(markup).toMatch(/<option[^>]*value="hard"[^>]*selected/);
    // a run outside the runs folder (ORIENTYR_RUN) has no name: the picker still shows that it is the open one
    const unnamed = renderToStaticMarkup(<Console run={hard} runs={["fake"]} />);
    expect(unnamed).toMatch(/<option[^>]*value=""[^>]*selected/);
    expect(unnamed).toContain("/api/map\"");
  });

  it("offers half speed, and no text smaller than 12 px in the map's own labels", () => {
    const markup = html();
    expect(markup).toMatch(/aria-label="Playback speed"/);
    expect(markup).toContain(">0.5×<");
  });
});
