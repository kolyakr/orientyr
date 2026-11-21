# Orientyr console: replay viewer

Mission control for a flight: the map, the route coloured by confidence, the footprint of the current frame,
the camera inset, the telemetry HUD, the tracker state and a mission log. The right column shows the state as a
lamp, the last event, confidence, heading on a compass, the drone's place on a small map plate, position, matched
pairs and uncertainty. The mission log is a timeline that follows the current event; in follow mode a minimap shows
the whole route. It replays a **run folder**
(`run.json` + `run.jsonl` + `frames/`, the format in `src/dronenav/telemetry.py`): the fake run (M0-04), a real
run of `scripts/run.py`, a benchmark run. Play, pause, scrub, speed and layers work; the mission controls are drawn
but inactive until the live engine (M7-04, M7-05).

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

The **runs folder** is `ORIENTYR_RUNS`, else the folder `ORIENTYR_RUN` lies in, else `outputs/` in the repo root.
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
| `tests/fixtures/runs/` | hand-made hard runs: gaps, missing files, an empty and a broken `run.jsonl` |

Technology (decided in M7-01): a Next.js web app. It runs in any browser, is easy to record for the demo
video, and the same page can later follow a live `run.jsonl` while the engine writes it.
