/**
 * Live mode (M7-04): how the console follows a run while the engine writes it (python scripts/live.py).
 *
 *   GET  /api/live?run=<name>&from=<byte>   the lines appended to run.jsonl since byte `from`, the byte to ask from
 *                                           next (`next`), and the engine's state. The console asks every POLL_MS.
 *   POST /api/live?run=<name>  {"jam": true|false}   the jam switch: written to the run's control.json, which the
 *                                           engine reads before every frame.
 */
import { jamOrder, liveStatus, readTail, resolveRun, writeJamOrder } from "@/lib/run-server";

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
    jam: status?.live ? jamOrder(dir, status) : false,
    fixesPerS: status?.fixesPerS ?? null,
    ...tail,
  }, { headers: { "cache-control": "no-store" } });
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
  const jam = typeof body === "object" && body !== null ? (body as { jam?: unknown }).jam : undefined;
  if (typeof jam !== "boolean") return Response.json({ error: 'the body must be {"jam": true} or {"jam": false}' }, { status: 400 });
  // an order nobody will read would leave the switch pressed over a flight that has landed
  if (!liveStatus(dir)?.live) return Response.json({ error: "this run is not live" }, { status: 409 });
  writeJamOrder(dir, jam);
  return Response.json({ jam });
}
