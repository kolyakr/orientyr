/** The map image of a run, streamed from where its run.json says it is: /api/map?run=<name> (no name: the default run). */
import { createReadStream, statSync } from "node:fs";
import { Readable } from "node:stream";
import { loadRunInfo, mapPath, resolveRun } from "@/lib/run-server";
import { contentType } from "@/lib/files";

export const dynamic = "force-dynamic"; // the run is chosen per request, not at build time

export function GET(request: Request): Response {
  try {
    const { dir } = resolveRun(new URL(request.url).searchParams.get("run")); // only a listed run: see resolveRun
    const file = mapPath(dir, loadRunInfo(dir));
    const stat = statSync(file);
    if (!stat.isFile()) throw new Error(`${file} is not a file`);
    return new Response(Readable.toWeb(createReadStream(file)) as ReadableStream, {
      headers: { "content-type": contentType(file), "content-length": String(stat.size), "cache-control": "no-cache" },
    });
  } catch (e) {
    return new Response(`map not available: ${(e as Error).message}`, { status: 404 });
  }
}
