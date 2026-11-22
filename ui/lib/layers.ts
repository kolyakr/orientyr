/**
 * The layers of the map view: what can be switched on and off, which of them a run has, and how the choice
 * travels in the URL (so a link or a reload shows the same picture). The drone, the HUD and the timeline are not
 * layers: without them there is no console.
 */
import type { RunRecord } from "./run";
import { gpsOf, truthOf, windowOf } from "./view";

export const LAYERS = [
  { id: "trail", label: "Route", note: "coloured by confidence" },
  { id: "points", label: "Numbered points", note: "with the start and the finish" },
  { id: "footprint", label: "Camera footprint", note: "and rejected fixes" },
  { id: "sigma", label: "Error area", note: "3σ of the position" },
  { id: "camera", label: "Camera inset", note: "the frame the drone sees" },
  { id: "window", label: "Search window", note: "where the matcher looked" },
  { id: "gps", label: "GPS reading", note: "checked against the camera" },
  { id: "truth", label: "Ground truth", note: "the true route, for checking" },
] as const;

export type LayerId = (typeof LAYERS)[number]["id"];
export type Layers = Record<LayerId, boolean>;

/** All on, except the ground truth: it is a check of the tracker, not something the drone knows. */
export const DEFAULT_LAYERS: Layers = {
  trail: true, points: true, footprint: true, sigma: true, camera: true, window: true, gps: true, truth: false,
};

const isLayer = (id: string): id is LayerId => LAYERS.some((l) => l.id === id);

/**
 * The layers this run can show, in the order of LAYERS. The first five need nothing but the telemetry every run
 * has; the search window, the GPS reading and the ground truth are offered only when some record carries them.
 */
export function availableLayers(records: RunRecord[]): LayerId[] {
  const has: Partial<Record<LayerId, boolean>> = {
    window: records.some((r) => windowOf(r) !== null),
    gps: records.some((r) => gpsOf(r) !== null),
    truth: records.some((r) => truthOf(r) !== null),
  };
  return LAYERS.map((l) => l.id).filter((id) => has[id] ?? true);
}

/** Read `?off=a,b` and `?on=c`: the layers that differ from the defaults. Unknown names are ignored. */
export function parseLayers(off: string | null | undefined, on: string | null | undefined): Layers {
  const layers = { ...DEFAULT_LAYERS };
  const each = (list: string | null | undefined, value: boolean) => {
    for (const id of (list ?? "").split(",").map((s) => s.trim())) if (isLayer(id)) layers[id] = value;
  };
  each(off, false);
  each(on, true);
  return layers;
}

/** The console's address for a run, a frame and a choice of layers: only what differs from the defaults is written. */
export function consoleUrl({ run, frame, layers }: { run: string | null; frame: number; layers: Layers }): string {
  const q = new URLSearchParams();
  if (run) q.set("run", run);
  if (frame > 0) q.set("frame", String(frame));
  const ids = LAYERS.map((l) => l.id);
  const off = ids.filter((id) => DEFAULT_LAYERS[id] && !layers[id]);
  const on = ids.filter((id) => !DEFAULT_LAYERS[id] && layers[id]);
  if (off.length) q.set("off", off.join(","));
  if (on.length) q.set("on", on.join(","));
  const text = q.toString();
  return text ? `/?${text}` : "/";
}
