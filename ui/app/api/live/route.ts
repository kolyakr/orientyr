/**
 * Live mode (M7-04): how the console follows a run while the engine writes it (python scripts/live.py), and the
 * judge controls (M7-05): how it gives the flight its orders.
 *
 *   GET  /api/live?run=<name>&from=<byte>   the lines appended to run.jsonl since byte `from`, the byte to ask from
 *                                           next (`next`), the engine's state, and the controls: what is switched on
 *                                           (jam, spoof, map, flight) and what can be chosen (maps, flights).
 *   POST /api/live?run=<name>               one order per request, written to the run's control.json, which the
 *                                           engine reads before every frame:
 *        {"jam": true|false}                jam the camera
 *        {"spoof": true|false}              spoof the GPS receiver
 *        {"map": "<name>"}                  search another map; one of `maps`
 *        {"flight": "<name>", "speed": 2}   land and start another flight; one of `flights`, speed optional
 */
import { MAX_FLIGHT_SPEED } from "@/lib/live";
import { type Order, liveStatus, readTail, resolveRun, switches, writeOrder } from "@/lib/run-server";

export const dynamic = "force-dynamic";

function runFolder(request: Request): string | null {
  try {
    return resolveRun(new URL(request.url).searchParams.get("run")).dir; // only a listed run: see resolveRun
  } catch {
    return null;
  }
}

export async function GET(request: Request): Promise<Response> {
  const dir = runFolder(request);
  if (!dir) return Response.json({ error: "no such run" }, { status: 404 });
  const status = liveStatus(dir);
  const tail = readTail(dir, Number(new URL(request.url).searchParams.get("from") ?? 0));
  return Response.json({
    live: status?.live ?? false,
    state: status?.state ?? null, // null: this run was never live
    ...switches(dir, status?.live ? status : null),
    ...(status?.live ? {} : { jam: false, spoof: false }), // a flight that is over has nothing switched on
    fixesPerS: status?.fixesPerS ?? null,
    ...tail,
  }, { headers: { "cache-control": "no-store" } });
}

const USAGE = 'the body must be one order: {"jam": bool}, {"spoof": bool}, {"map": name} or {"flight": name, "speed": number}';

/** The one order a body holds; null when it holds none, several, or something that is not an order. */
function orderOf(body: unknown): Order | null {
  if (typeof body !== "object" || body === null || Array.isArray(body)) return null;
  const b = body as Record<string, unknown>;
  const keys = Object.keys(b).sort().join(",");
  if (keys === "jam" && typeof b.jam === "boolean") return { jam: b.jam };
  if (keys === "spoof" && typeof b.spoof === "boolean") return { spoof: b.spoof };
  if (keys === "map" && typeof b.map === "string" && b.map !== "") return { map: b.map };
  if ((keys === "flight" || keys === "flight,speed") && typeof b.flight === "string" && b.flight !== "") {
    // NaN: "the speed of the flight in the air", filled in once the status is read
    const speed = b.speed === undefined ? Number.NaN : b.speed;
    if (typeof speed !== "number" || (b.speed !== undefined && !(speed > 0 && speed <= MAX_FLIGHT_SPEED))) return null;
    return { flight: b.flight, speed };
  }
  return null;
}

export async function POST(request: Request): Promise<Response> {
  const dir = runFolder(request);
  if (!dir) return Response.json({ error: "no such run" }, { status: 404 });
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    body = null;
  }
  let order = orderOf(body);
  if (!order) return Response.json({ error: USAGE }, { status: 400 });
  // an order nobody will read would leave the switch pressed over a flight that has landed
  const status = liveStatus(dir);
  if (!status?.live) return Response.json({ error: "this run is not live" }, { status: 409 });
  // only what the engine offers: a name it does not know would be ignored, and the button would lie
  if ("map" in order && !status.maps.includes(order.map)) {
    return Response.json({ error: `no such map; the engine offers: ${status.maps.join(", ") || "none"}` }, { status: 400 });
  }
  if ("flight" in order) {
    if (!status.flights.includes(order.flight)) {
      return Response.json({ error: `no such flight; the engine offers: ${status.flights.join(", ") || "none"}` }, { status: 400 });
    }
    if (Number.isNaN(order.speed)) order = { flight: order.flight, speed: status.speed ?? 1 };
  }
  return Response.json(writeOrder(dir, order));
}
