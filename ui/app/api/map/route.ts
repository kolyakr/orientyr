/** The map image of the current run, streamed from where run.json says it is. */
import { createReadStream, statSync } from "node:fs";
import { Readable } from "node:stream";
import { loadRun, mapPath, runDir } from "@/lib/run-server";
import { contentType } from "@/lib/files";

export const dynamic = "force-dynamic"; // the run folder is chosen at start-up, not at build time

export function GET(): Response {
  try {
    const dir = runDir();
    const file = mapPath(dir, loadRun(dir).info);
    const { size } = statSync(file);
    return new Response(Readable.toWeb(createReadStream(file)) as ReadableStream, {
      headers: { "content-type": contentType(file), "content-length": String(size), "cache-control": "no-cache" },
    });
  } catch (e) {
    return new Response(`map not available: ${(e as Error).message}`, { status: 404 });
  }
}
