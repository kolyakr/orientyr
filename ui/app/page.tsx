import { Console } from "@/components/Console";
import type { Run } from "@/lib/run";
import { loadRun, runDir } from "@/lib/run-server";

export const dynamic = "force-dynamic"; // the run folder is read on every request: regenerate it and reload

/** `/?frame=22` opens the console on that frame: handy for screenshots and for pointing a teammate at a moment. */
export default async function Page({ searchParams }: { searchParams: Promise<{ frame?: string }> }) {
  const frame = Number((await searchParams).frame ?? 0);
  const dir = runDir();
  let run: Run;
  try {
    run = loadRun(dir);
    if (run.records.length === 0) throw new Error(`run.jsonl in ${dir} has no records`);
  } catch (e) {
    return (
      <main className="empty">
        <h1>No run to show</h1>
        <p>{(e as Error).message}</p>
        <p>Make a fake run from the repo root, then reload:</p>
        <pre>python scripts/fake_run.py --map data/hackatons-data/global_map.png --out outputs/fake</pre>
        <p>Or point the console at another run folder: <code>ORIENTYR_RUN=/path/to/run npm run dev</code></p>
      </main>
    );
  }
  return <Console run={run} initialFrame={Number.isInteger(frame) ? frame : 0} />;
}
