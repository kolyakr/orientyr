# Orientyr console: replay viewer and live view

Mission control for a flight: the map, the route coloured by confidence, the footprint of the current frame,
the camera inset, the telemetry HUD, the tracker state and a mission log. The right column shows the state as a
lamp, the last event, confidence, heading on a compass, the drone's place on a small map plate, position, matched
pairs and uncertainty. The mission log is a timeline that follows the current event; in follow mode a minimap shows
the whole route. It replays a **run folder**
(`run.json` + `run.jsonl` + `frames/`, the format in `src/dronenav/telemetry.py`): the fake run (M0-04), a real
run of `scripts/run.py`, a benchmark run. Play, pause, scrub, speed and layers work. A run that `scripts/live.py`
is writing right now is followed as it grows (see "Live mode"); there the mission controls work (jam camera, spoof GPS,
map year, new flight); on a replay they are drawn but disabled.

```bash
# from the repo root: make a run to show
python scripts/fake_run.py --map data/hackatons-data/global_map.png --out outputs/fake

cd ui
npm install
npm run dev                               # http://localhost:3000, the runs of ../outputs, opens `fake`
ORIENTYR_RUNS=/path/to/runs npm run dev   # another folder of run folders
ORIENTYR_RUN=/path/to/run npm run dev     # one run folder: it opens first, its neighbours are in the picker
npm test                                  # vitest: the reader, the numbers, the page's parts
```

## Which run

The **runs folder** is `ORIENTYR_RUNS`, else the folder `ORIENTYR_RUN` lies in, else `public/runs` when a run is
packed there (see "Hosting"), else `outputs/` in the repo root.
Every folder in it with a `run.json` is a run, named by its path from the runs folder (`irpin`, or `bench/sift` one
level down). The **Run** list at the top of the map opens another run without a restart; `/?run=irpin` does the
same from the address. A name that is not in the list is refused, so the address cannot reach outside the runs
folder. Without `?run` the console opens `ORIENTYR_RUN`, else `fake`, else the first run.

The address keeps the view: `/?run=irpin&frame=22&off=points&on=truth` is that run, on that frame, without the
numbered points and with the ground truth. It follows what you do while the flight rests, so a reload or a pasted
link shows the same thing.

A run that cannot be read (no `run.jsonl`, an empty one, a broken line) gives a page that says what is wrong, with
the line number, and lists the other runs. A run with gaps still plays: frames before the first fix have no drone
on the map, a missing frame image shows "No image for this frame", a missing map leaves the route on a dark
ground with a note.

## Playback

| Key | What it does |
|---|---|
| `Space` | play / pause (at the end: replay) |
| `←` `→` | one frame; with `Shift` ten frames |
| `Home` `End` | the first / the last frame |
| `,` `.` | slower / faster: 0.5×, 1×, 2×, 4×, 8× of the run's own rate (`fps` in `run.json`) |
| `F` | follow the drone / show the whole map |
| `L` | open the list of layers; `Esc` closes it |
| `1` … `8` | switch one layer |

Scrub with the pointer: press anywhere on the timeline and drag, also outside it. A flight that was playing goes on
from where you let go.

## Live mode

```bash
# terminal 1, from the repo root: the simulator and the engine, in real time
python scripts/live.py --flight data/irpin/flights/mild/2022 --map data/irpin/maps/irpin_2022.png --out outputs/live
# terminal 2: the console; open http://localhost:3000/?run=live
cd ui && npm run dev
```

`scripts/live.py` plays a rendered flight at its own pace (2 frames per second of flight by default), localizes each
frame when it arrives and appends its record to `run.jsonl`. The console opens such a run at its newest record and
stays there: every 0.4 s it asks `/api/live` for the lines appended since, and the drone glides to each new fix for a
little longer than the fixes are apart, so it is still moving when the next one arrives (`lib/live.ts`). Before the
first record the page says it is waiting and asks again by itself.

- **Live** (next to the time): lit while the view follows the flight. Stepping, scrubbing or `Space` holds a frame
  to look back while the flight goes on; press Live to return. A replay that catches up with the flight goes on
  with it. When the flight ends the page becomes an ordinary replay.
- **Camera size**: drag the corner of the camera panel to make it bigger (double-click: the default size). Kept per browser; works on a replay too.
- **Jam camera** (Mission controls): the frames reach the engine as noise until "Stop jamming". The press is written
  to the run's `control.json`, which the engine reads before every frame.
- **Spoof GPS**: the simulated GPS receiver starts to walk away from the truth. The **GPS check** panel (under the
  HUD, shown when the run has GPS verdicts) goes from "GPS agrees with the camera" to "suspect" and then to a red
  "GPS spoofed" with the gap in px, a few seconds after the press. "Stop spoofing" clears it.
- **Map** 2017 / 2022 / 2025: the engine matches against the map of that year from the next frame on, and the map
  on screen changes with it. The lit button is the map the engine uses now.
- **New flight**: pick a flight and a speed (0.5× to 8×), press the button; the current flight stops, the new one
  starts from its first frame in the same run folder and the page reloads itself.
- Each press is one `POST /api/live` with one order (`{jam}`, `{spoof}`, `{map}` or `{flight, speed}`). A wrong shape
  is a 400, a run that is not live a 409, a map or flight the engine does not offer a 400. `GET /api/live` returns
  the switches and the choices with the new records.
- The engine says it is alive by rewriting `live.json` after every record; silent for 30 s, the run is shown as a
  replay of what was written.

## Layers

| Key | Layer | What is drawn |
|---|---|---|
| `1` | Route | the trail, coloured by confidence, dashed where the tracker coasted |
| `2` | Numbered points | numbered points, START and FINISH |
| `3` | Camera footprint | what the camera sees at this frame, and a rejected fix with its reason |
| `4` | Error area | the 3σ ellipse of the position |
| `5` | Camera inset | the frame the drone sees, with the matched points |
| `6` | Search window | `extra.predicted_window`: where the next match is looked for |
| `7` | GPS reading | `extra.gps`: the receiver's position, in the colour of the spoof detector's verdict |
| `8` | Ground truth | `extra.truth`: the true route and the tracker's distance from it (off at first) |

The last three are offered only for a run whose records carry them. The drone, the HUD, the state, the timeline
and the mission log are always shown.

| Path | What it is |
|---|---|
| `lib/run.ts` | types and parsing of `run.json` / `run.jsonl` (browser-safe) |
| `lib/run-server.ts` | the runs folder, which run is open, and safe paths into it (server only) |
| `lib/view.ts` | what is drawn from the telemetry: confidence, σ, heading, trail, timeline, and what `extra` carries |
| `lib/playback.ts` | speeds, the frame under the pointer, what each key does |
| `lib/layers.ts` | the layers, which of them a run has, and the address of a view |
| `components/MapView.tsx` | the map as an SVG in map pixel coordinates |
| `components/Minimap.tsx` | follow mode: the whole route's area, small, with the route flown so far |
| `components/Console.tsx` | the page: panels, run picker, layers, playback, scrubbing, keyboard |
| `app/page.tsx` | opens the run the address names, or says why it cannot |
| `app/api/map`, `app/api/frame` | serve the map and the frames of a run (`?run=<name>`) |
| `app/api/live` | live mode: the records appended since byte `from` and the switches (GET), one order of the judge controls (POST) |
| `lib/live.ts` | live mode: the engine's status, the pace of the fixes, how long the drone glides |
| `tests/fixtures/runs/` | hand-made hard runs: gaps, missing files, an empty and a broken `run.jsonl` |
| `public/runs/` | packed runs for hosting (gitignored), see "Hosting" |

## Hosting

A replay of the Irpin flight is online: <https://orientyr-nine.vercel.app> (`/?run=irpin&frame=120` opens on that
frame). It is the console only, replaying a recorded run; the engine does not run there.
Imagery: Esri World Imagery Wayback (Maxar).

A host has no `data/` and no `outputs/`, so the runs to show are packed into the app first: `scripts/pack_run.py`
copies the telemetry and writes the map as one WebP and the frames as small JPEGs (about 27 MB for 442 frames).
When `public/runs` holds a run and no `ORIENTYR_RUNS` / `ORIENTYR_RUN` is set, it is the runs folder: every run
packed there is in the Run list, and its map and frames are loaded as static files instead of through the API
routes. `public/runs` is gitignored (imagery is never committed), so the site is built on a laptop that has the
data and uploaded ready-made. It does not follow the repo: after a change to the console, deploy again.

```bash
# from the repo root: pack the run to show (never a run over the organisers' map: that map stays private)
python scripts/pack_run.py --run outputs/irpin --out ui/public/runs/irpin

cd ui
vercel pull --yes --environment=production   # once per clone, after `vercel link` to the project
vercel build --prod                          # builds here, with the packed runs inside
vercel deploy --prebuilt --prod              # uploads the build; the link above then shows it
```

Remove `ui/public/runs` to see the runs of `outputs/` again with `npm run dev`. `engines.node` in `package.json`
pins the Node version the host runs.

Technology (decided in M7-01): a Next.js web app. It runs in any browser, is easy to record for the demo
video, and the same page follows a live `run.jsonl` while the engine writes it (M7-04).
