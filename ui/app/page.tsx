import { Console } from "@/components/Console";
import { DEFAULT_LAYERS, consoleUrl, parseLayers } from "@/lib/layers";
import { openRun, runsDir } from "@/lib/run-server";

export const dynamic = "force-dynamic"; // the run folder is read on every request: regenerate it and reload

type Query = { [key: string]: string | string[] | undefined };
const one = (v: string | string[] | undefined) => (Array.isArray(v) ? v[0] : v);

/**
 * The replay viewer. The address says what is shown, so a link points a teammate at a moment:
 * `/?run=irpin&frame=22&off=points&on=truth`: the run (a folder name in the runs folder; without it the default
 * run), the frame, and the layers that differ from the defaults.
 */
export default async function Page({ searchParams }: { searchParams: Promise<Query> }) {
  const query = await searchParams;
  const frame = Number(one(query.frame) ?? 0);
  const opened = openRun(one(query.run));
  if (!opened.run) {
    return (
      <main className="empty" data-part="run-error">
        <h1>No run to show</h1>
        <p className="reason">{opened.error}</p>
        {opened.runs.length > 0 && (
          <>
            <p>Runs in <code>{runsDir()}</code>:</p>
            <ul>
              {opened.runs.map((name) => (
                <li key={name}><a href={consoleUrl({ run: name, frame: 0, layers: DEFAULT_LAYERS })}>{name}</a></li>
              ))}
            </ul>
          </>
        )}
        <p>Make a fake run from the repo root, then reload:</p>
        <pre>python scripts/fake_run.py --map data/hackatons-data/global_map.png --out outputs/fake</pre>
        <p>
          Or point the console at other runs: <code>ORIENTYR_RUNS=/path/to/runs npm run dev</code> (a folder of run
          folders), <code>ORIENTYR_RUN=/path/to/run npm run dev</code> (one run).
        </p>
      </main>
    );
  }
  return (
    <Console key={opened.dir} run={opened.run} initialFrame={Number.isInteger(frame) ? frame : 0}
             runName={opened.name} runs={opened.runs} mapMissing={opened.mapMissing} assets={opened.assets}
             initialLayers={parseLayers(one(query.off), one(query.on))} />
  );
}
