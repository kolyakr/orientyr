/** One camera frame of a run: /api/frame?run=<name>&src=frames/frame_00007.png (a record's `source`; no name: the default run). */
import { readFile } from "node:fs/promises";
import { framePath, resolveRun } from "@/lib/run-server";
import { contentType } from "@/lib/files";

export const dynamic = "force-dynamic";

export async function GET(request: Request): Promise<Response> {
  const query = new URL(request.url).searchParams;
  const src = query.get("src") ?? "";
  try {
    const { dir } = resolveRun(query.get("run")); // only a listed run: see resolveRun
    const file = framePath(dir, src); // refuses paths outside the run folder
    return new Response(new Uint8Array(await readFile(file)), {
      headers: { "content-type": contentType(file), "cache-control": "no-cache" },
    });
  } catch {
    return new Response(`no such frame: ${src}`, { status: 404 });
  }
}
