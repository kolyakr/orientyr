/** One camera frame of the current run: /api/frame?src=frames/frame_00007.png (a record's `source`). */
import { readFile } from "node:fs/promises";
import { framePath, runDir } from "@/lib/run-server";
import { contentType } from "@/lib/files";

export const dynamic = "force-dynamic";

export async function GET(request: Request): Promise<Response> {
  const src = new URL(request.url).searchParams.get("src") ?? "";
  try {
    const file = framePath(runDir(), src); // refuses paths outside the run folder
    return new Response(new Uint8Array(await readFile(file)), {
      headers: { "content-type": contentType(file), "cache-control": "no-cache" },
    });
  } catch {
    return new Response(`no such frame: ${src}`, { status: 404 });
  }
}
