/** Playing and scrubbing, as pure functions: Console.tsx wires them to the pointer, the keyboard and the clock. */

/** The playback speeds, slowest first. 1 is real time: one frame of the run per 1 / fps seconds. */
export const SPEEDS = [0.5, 1, 2, 4, 8] as const;

/** One step up (`by` > 0) or down the list of speeds, stopping at its ends. A speed that is not in the list: real time. */
export function nextSpeed(speed: number, by: number): number {
  const i = SPEEDS.indexOf(speed as (typeof SPEEDS)[number]);
  if (i < 0) return 1;
  return SPEEDS[Math.min(Math.max(i + Math.sign(by), 0), SPEEDS.length - 1)];
}

/**
 * How long one frame stays on screen, in ms. A run that does not say its fps (or says nonsense) plays at one frame
 * a second. Never shorter than one screen refresh: faster than that the browser would only skip frames.
 */
export function stepMs(fps: number | null | undefined, speed: number): number {
  const rate = typeof fps === "number" && Number.isFinite(fps) && fps > 0 ? fps : 1;
  return Math.max(16, 1000 / (rate * speed));
}

/**
 * The frame under the pointer on the timeline: `left` and `width` are the timeline's box on the screen, `n` the
 * number of frames (each owns an equal slice). A drag that leaves the timeline stays on the first or the last frame.
 */
export function frameAt(clientX: number, left: number, width: number, n: number): number {
  if (!(width > 0) || n <= 1) return 0;
  return Math.min(Math.max(Math.floor(((clientX - left) / width) * n), 0), n - 1);
}

export type KeyAction =
  | { kind: "toggle" }                      // play / pause
  | { kind: "step"; by: number }            // frames forward (+) or back (-)
  | { kind: "jump"; to: "start" | "end" }
  | { kind: "speed"; by: number }           // one step up or down the list of speeds
  | { kind: "follow" }
  | { kind: "layers" }                      // open or close the list of layers
  | { kind: "layer"; index: number }        // switch the layer with this place in the list
  | { kind: "close" };

export interface KeyInput {
  code: string;
  shiftKey?: boolean;
  metaKey?: boolean;
  ctrlKey?: boolean;
  altKey?: boolean;
  tag?: string;        // the focused element's tag name
  inputType?: string;  // and, for an <input>, its type
}

/**
 * What a key press means for the console, or null when it is not ours: a browser shortcut (with Cmd, Ctrl or Alt),
 * or typing in a form control. The timeline's slider is an <input> too, but it takes no text: the keys work on it.
 */
export function keyAction(e: KeyInput): KeyAction | null {
  if (e.metaKey || e.ctrlKey || e.altKey) return null;
  if (e.tag === "SELECT" || e.tag === "TEXTAREA" || (e.tag === "INPUT" && e.inputType !== "range")) return null;
  const many = e.shiftKey ? 10 : 1;
  switch (e.code) {
    case "Space": return { kind: "toggle" };
    case "ArrowRight": return { kind: "step", by: many };
    case "ArrowLeft": return { kind: "step", by: -many };
    case "Home": return { kind: "jump", to: "start" };
    case "End": return { kind: "jump", to: "end" };
    case "Comma": return { kind: "speed", by: -1 };
    case "Period": return { kind: "speed", by: 1 };
    case "KeyF": return { kind: "follow" };
    case "KeyL": return { kind: "layers" };
    case "Escape": return { kind: "close" };
  }
  const digit = /^Digit([1-8])$/.exec(e.code);
  return digit ? { kind: "layer", index: Number(digit[1]) - 1 } : null;
}
