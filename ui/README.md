# Orientyr console (mock-up)

Mission control for a flight: the map, the route coloured by confidence, the footprint of the current frame,
the camera inset, the telemetry HUD, the tracker state and a mission log. It replays a **run folder**
(`run.json` + `run.jsonl` + `frames/`, the format in `src/dronenav/telemetry.py`), so it works today on the
fake run (M0-04) and later on real runs without changes. The mission controls are drawn but inactive.

```bash
# from the repo root: make a run to show
python scripts/fake_run.py --map data/hackatons-data/global_map.png --out outputs/fake

cd ui
npm install
npm run dev                               # http://localhost:3000, shows ../outputs/fake
ORIENTYR_RUN=/path/to/run npm run dev     # any other run folder
npm test                                  # vitest: the reader, the numbers, the page's parts
```

Keys: `Space` play / pause, `←` `→` one frame, `F` follow the drone. `/?frame=22` opens on that frame.

| Path | What it is |
|---|---|
| `lib/run.ts` | types and parsing of `run.json` / `run.jsonl` (browser-safe) |
| `lib/run-server.ts` | which run folder, and safe paths into it (server only) |
| `lib/view.ts` | what is drawn from the telemetry: confidence, σ, heading, trail, timeline |
| `components/MapView.tsx` | the map as an SVG in map pixel coordinates |
| `components/Console.tsx` | the page: panels, playback, keyboard |
| `app/api/map`, `app/api/frame` | serve the map and the frames out of the run folder |

Technology (decided in M7-01): a Next.js web app. It runs in any browser, is easy to record for the demo
video, and the same page can later follow a live `run.jsonl` while the engine writes it.
