"use client";

/**
 * Mission control, replaying a recorded run (run.json + run.jsonl + frames/) on the map.
 * The map is the scene; everything else lies over it as dark glass panels: the brand and the camera
 * on the left, the run picker and the layers at the top right, the timeline dock under the map, and on the
 * right what the tracker knows at this frame. Play, pause, scrub, speed and layers work; the mission controls
 * are drawn but inactive until the live engine exists (M7-04, M7-05).
 */
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent } from "react";
import { DEFAULT_LAYERS, LAYERS, type LayerId, type Layers, availableLayers, consoleUrl } from "@/lib/layers";
import { SPEEDS, frameAt, stepMs as frameMs, keyAction, nextSpeed } from "@/lib/playback";
import type { Run, RunRecord } from "@/lib/run";
import { STATE_LABEL, clock, compass, confidence, confidenceColor, drawAngle, events, headingOf, niceStep, searchOf, sigmaPx, stateRuns } from "@/lib/view";
import { MapView, type StageLayout } from "./MapView";
import { Minimap } from "./Minimap";

const MAX_DOTS = 40;
const SPARK = 48; // frames in the matched-pairs strip chart of the HUD
const NO_RUNS: string[] = [];
const sentence = (s: string) => s.charAt(0).toUpperCase() + s.slice(1);

/**
 * Where to draw the matched points on the camera frame, as fractions of its size.
 * PLACEHOLDER: the telemetry does not carry the matches yet, so the points are scattered by a generator
 * seeded with the frame index (stable per frame, more points for more inliers). When the engine writes the
 * real inlier coordinates into `extra`, this function reads them instead.
 */
function matchedPoints(rec: RunRecord): [number, number][] {
  if (!rec.fix?.accepted) return [];
  let seed = (rec.idx + 1) * 2654435761;
  const rnd = () => {   // mulberry32: small, fast, good enough to scatter dots
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const n = Math.min(MAX_DOTS, Math.round(rec.fix.inliers / 8));
  return Array.from({ length: n }, () => [0.06 + 0.88 * rnd(), 0.06 + 0.88 * rnd()]);
}

/**
 * A value whose characters tick into place when they change: each character is keyed by its position and its
 * glyph, so only the ones that changed are mounted again and play the CSS entrance. Screen readers get the
 * plain value once.
 */
function Ticker({ value }: { value: string }) {
  return (
    <span className="ticker">
      <span className="sr">{value}</span>
      <span aria-hidden="true">
        {Array.from(value).map((ch, i) => <span key={`${i}${ch}`}>{ch}</span>)}
      </span>
    </span>
  );
}

export function Console({ run, initialFrame = 0, runName = null, runs = NO_RUNS, initialLayers = DEFAULT_LAYERS, mapMissing = false }: {
  run: Run;
  initialFrame?: number;
  runName?: string | null;  // the run's name in the runs folder: it goes into the map's and the frames' URLs; null = the default run
  runs?: string[];          // the runs the picker offers
  initialLayers?: Layers;   // from the URL
  mapMissing?: boolean;     // the server did not find the map image
}) {
  const n = run.records.length;
  const last = n - 1;
  const [frame, setFrame] = useState(Math.min(Math.max(initialFrame, 0), last));
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(2);
  const [follow, setFollow] = useState(false);
  const [layers, setLayers] = useState(initialLayers);
  const [layersOpen, setLayersOpen] = useState(false);
  const [scrubbing, setScrubbing] = useState(false);

  const rec = run.records[frame];
  const fix = rec.fix;
  const stepMs = frameMs(run.info.fps, speed);
  // the files of this run: the map and the camera frames
  const runQuery = runName ? `run=${encodeURIComponent(runName)}` : "";
  const mapUrl = runQuery ? `/api/map?${runQuery}` : "/api/map";
  const frameUrl = useCallback((r: RunRecord) => `/api/frame?${runQuery && `${runQuery}&`}src=${encodeURIComponent(r.source)}`, [runQuery]);
  // a layer is drawn when the run has it and it is switched on
  const offered = useMemo(() => availableLayers(run.records), [run]);
  const shown = useMemo(
    () => Object.fromEntries(LAYERS.map((l) => [l.id, offered.includes(l.id) && layers[l.id]])) as Layers,
    [offered, layers],
  );
  const toggleLayer = useCallback((id: LayerId) => setLayers((old) => ({ ...old, [id]: !old[id] })), []);
  const log = useMemo(() => events(run.records), [run]);
  const maxPairs = useMemo(() => Math.max(1, ...run.records.map((r) => r.fix?.inliers ?? 0)), [run]);
  const lastEvent = log.findLast((e) => e.frame <= frame);
  // the log keeps the current event in sight: it scrolls inside its own list only
  const logList = useRef<HTMLOListElement>(null);
  const lastEventFrame = lastEvent?.frame;
  useEffect(() => {
    const list = logList.current;
    const li = list?.querySelector<HTMLElement>("[aria-current]");
    if (!list || !li || typeof list.scrollTo !== "function") return;
    list.scrollTo({ top: li.offsetTop - (list.clientHeight - li.offsetHeight) / 2, behavior: "smooth" });
  }, [lastEventFrame]);

  const jump = useCallback((f: number) => {
    setPlaying(false);
    setFrame(Math.min(Math.max(f, 0), last));
  }, [last]);
  const step = useCallback((d: number) => {
    setPlaying(false);
    setFrame((f) => Math.min(Math.max(f + d, 0), last));
  }, [last]);
  const toggle = useCallback(() => {
    if (frame === last) setFrame(0); // play again from the start
    setPlaying((p) => !p);
  }, [frame, last]);

  useEffect(() => {
    if (!playing) return;
    const id = setTimeout(() => {
      if (frame >= last) setPlaying(false);
      else setFrame(frame + 1);
    }, frame >= last ? 0 : stepMs);
    return () => clearTimeout(id);
  }, [playing, frame, last, stepMs]);

  useEffect(() => {
    // fetch the next camera frame ahead, so the inset never flashes empty during playback
    if (frame < last && shown.camera) new Image().src = frameUrl(run.records[frame + 1]);
  }, [frame, last, run, frameUrl, shown.camera]);

  // A camera frame that cannot be loaded (the file is gone, or `source` is not an image): a placard, not a broken picture.
  const src = frameUrl(rec);
  const camera = useRef<HTMLImageElement>(null);
  const [failed, setFailed] = useState<string | null>(null);
  useEffect(() => {
    // an image that failed before React attached its handler (the server's HTML) reports nothing: ask it
    const img = camera.current;
    if (img && img.complete && img.naturalWidth === 0) setFailed(src);
  }, [src, shown.camera]);

  // Scrubbing: press on the timeline and drag. The frame under the pointer shows at once; a flight that was
  // playing goes on from where the pointer lets go.
  const chart = useRef<HTMLDivElement>(null);
  const resume = useRef(false);
  const scrubTo = (clientX: number) => {
    const box = chart.current?.getBoundingClientRect();
    if (box) setFrame(frameAt(clientX, box.left, box.width, n));
  };
  const scrubStart = (e: PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault(); // no text selection while dragging
    try {
      e.currentTarget.setPointerCapture(e.pointerId); // the drag goes on outside the timeline
    } catch { /* a pointer that is already gone */ }
    resume.current = playing;
    setPlaying(false);
    setScrubbing(true);
    scrubTo(e.clientX);
  };
  const scrubEnd = () => {
    if (!scrubbing) return;
    setScrubbing(false);
    if (resume.current && frame < last) setPlaying(true);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target instanceof HTMLElement ? e.target : null;
      const action = keyAction({
        code: e.code, shiftKey: e.shiftKey, metaKey: e.metaKey, ctrlKey: e.ctrlKey, altKey: e.altKey,
        tag: el?.tagName, inputType: el instanceof HTMLInputElement ? el.type : undefined,
      });
      if (!action || (action.kind === "close" && !layersOpen)) return;
      e.preventDefault(); // ours: the page does not scroll, the focused slider does not move a second time
      switch (action.kind) {
        case "toggle": toggle(); break;
        case "step": step(action.by); break;
        case "jump": jump(action.to === "start" ? 0 : last); break;
        case "speed": setSpeed((s) => nextSpeed(s, action.by)); break;
        case "follow": setFollow((v) => !v); break;
        case "layers": setLayersOpen((v) => !v); break;
        case "close": setLayersOpen(false); break;
        case "layer": {
          const layer = LAYERS[action.index];
          if (layer && offered.includes(layer.id)) toggleLayer(layer.id);
          break;
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, step, jump, last, layersOpen, offered, toggleLayer]);

  // the list of layers closes on a press anywhere outside it
  const layersBox = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!layersOpen) return;
    const onDown = (e: globalThis.PointerEvent) => {
      if (e.target instanceof Node && !layersBox.current?.contains(e.target)) setLayersOpen(false);
    };
    document.addEventListener("pointerdown", onDown);
    return () => document.removeEventListener("pointerdown", onDown);
  }, [layersOpen]);

  // The address follows the view: a reload or a shared link opens the same run, frame and layers. Written when the
  // flight rests, not on every frame of the playback (browsers limit how often the address may change).
  useEffect(() => {
    if (playing || scrubbing) return;
    const id = setTimeout(() => {
      try {
        window.history.replaceState(window.history.state, "", consoleUrl({ run: runName, frame, layers }));
      } catch { /* the address is a convenience: without it the console works as before */ }
    }, 300);
    return () => clearTimeout(id);
  }, [playing, scrubbing, frame, layers, runName]);

  /** Open another run, with the same layers. A full load: the page reads the run on the server. */
  const openRun = (name: string) => {
    window.location.assign(consoleUrl({ run: name || null, frame: 0, layers }));
  };
  // the picker lists the runs of the runs folder; the open run is in it even if it is not one of them
  const runOptions = runName === null || !runs.includes(runName) ? [runName ?? "", ...runs] : runs;

  const heading = headingOf(rec);
  const trust = Math.round(confidence(rec) * 100);
  const [mw, mh] = run.info.map_size;
  const tick = niceStep(n, 6);
  const [fw, fh] = run.info.frame_size ?? [1, 1];
  // The compass needle turns along the shortest arc: its drawn angle follows the heading without wrapping at 360°.
  const [needle, setNeedle] = useState(drawAngle(heading ?? 0));
  if (heading != null) {
    const next = drawAngle(needle + ((((heading - needle) % 360) + 540) % 360) - 180);
    if (next !== needle) setNeedle(next);
  }
  // The pairs matched over the last frames, as a strip chart that runs with the flight: x in 0..120, y in 4..30.
  const strip = run.records.slice(Math.max(0, frame - SPARK + 1), frame + 1).map((r, i, all) => [
    Math.round((SPARK - all.length + i) * (120 / (SPARK - 1)) * 10) / 10,
    Math.round((30 - ((r.fix?.inliers ?? 0) / maxPairs) * 26) * 10) / 10,
  ]);
  const sigma = sigmaPx(rec.cov);
  // the uncertainty ring grows with the logarithm of sigma: 1 px and 100 px both stay readable in one small dial
  const ringScale = Math.round(Math.min(1, 0.3 + 0.36 * Math.log10(1 + sigma)) * 100) / 100;
  // the drone on the locator plate of the HUD: the map is 100 units wide there
  const locH = Math.round((mh / mw) * 1000) / 10;
  const locX = Math.round(Math.min(Math.max(rec.x / mw, 0), 1) * 1000) / 10;
  const locY = Math.round(Math.min(Math.max(rec.y / mh, 0), 1) * locH * 10) / 10;
  const search = searchOf(rec);
  const sigmaWord = sigma <= 3 ? "very precise" : sigma <= 15 ? "precise" : sigma <= 40 ? "drifting" : "wide";
  // before the first step and after the last one nothing is being tracked: the lamp rests and says so
  const idle = playing ? null : frame === 0 ? "Ready" : frame === last ? "Flight complete" : null;
  // an event reads "what happened: what the tracker does now"; the HUD sets the two parts as a title and a note
  const [eventTitle, eventNote = ""] = sentence(lastEvent ? lastEvent.text : "mission start: tracking").split(/:\s*/, 2);

  // The timeline is drawn once: the playhead moves by transform, the frames ahead are paled by one mask over all bars.
  const bars = useMemo(() => run.records.map((r) => (
    <i key={r.idx} style={{ height: `${r.fix ? Math.max(10, (r.fix.inliers / maxPairs) * 100) : 6}%`, background: confidenceColor(confidence(r)) }} />
  )), [run, maxPairs]);
  const bands = useMemo(() => stateRuns(run.records), [run]);
  const progress = Math.round(((frame + 0.5) / n) * 100000) / 1000; // % of the timeline, rounded for a stable transform

  // The stage, measured: the map's camera needs to know which part of it the panels leave free.
  // Layout offsets, not client rects: the panels' entrance animation must not move the numbers.
  const root = useRef<HTMLElement>(null);
  const overlay = useRef<HTMLDivElement>(null);
  const [layout, setLayout] = useState<StageLayout | null>(null);
  useLayoutEffect(() => {
    const main = root.current, cell = overlay.current;
    if (!main || !cell) return;
    const PAD = 12;
    const measure = () => {
      const width = cell.offsetWidth, height = cell.offsetHeight;
      let left = 0, top = 0;   // the camera inset on the left, and the top row: the brand and the tools
      for (const el of Array.from(cell.children) as HTMLElement[]) {
        if (el.classList.contains("minimap")) continue; // it sits in the bottom right corner, the route may pass under it
        if (el.classList.contains("camera")) left = el.offsetLeft + el.offsetWidth;
        else top = Math.max(top, el.offsetTop + el.offsetHeight);
      }
      if (left > width * 0.4) left = 0; // a very narrow stage: the column would take too much, the route may pass under it
      const x = left + PAD, y = top + PAD;
      const next: StageLayout = {
        cell: { width, height },
        free: { x, y, width: Math.max(width - x - PAD / 2, 40), height: Math.max(height - y - PAD / 2, 40) },
        view: { x: -cell.offsetLeft, y: -cell.offsetTop, width: main.clientWidth, height: main.clientHeight },
      };
      setLayout((old) => (JSON.stringify(old) === JSON.stringify(next) ? old : next));
    };
    measure();
    const watch = new ResizeObserver(measure);
    watch.observe(main);
    watch.observe(cell);
    for (const el of Array.from(cell.children)) watch.observe(el);
    return () => watch.disconnect();
  }, [shown.camera]); // the camera inset comes and goes with its layer: measure again
  /** The light under the pointer: each glass panel gets the pointer's place in its own box, CSS draws the glow. */
  const light = (e: PointerEvent<HTMLElement>) => {
    const glass = (e.target as Element).closest<HTMLElement>(".glass");
    if (!glass) return;
    const box = glass.getBoundingClientRect();
    glass.style.setProperty("--mx", `${Math.round(e.clientX - box.left)}px`);
    glass.style.setProperty("--my", `${Math.round(e.clientY - box.top)}px`);
  };

  return (
    <main ref={root} className="console" data-state={rec.state} data-follow={follow || undefined} data-playing={playing || undefined}
          data-scrubbing={scrubbing || undefined} onPointerMove={light}>
      <div className="stage">
        <MapView run={run} frame={frame} follow={follow} playing={playing} stepMs={playing ? stepMs : scrubbing ? 90 : 250} layout={layout}
                 layers={shown} mapUrl={mapUrl} />
        {mapMissing && (
          <p data-part="map-missing" className="stage-note">
            <strong>Map image not found</strong>
            <span>run.json points at <code>{run.info.map}</code>. The route is drawn without it.</span>
          </p>
        )}
      </div>
      {/* the red breath of a lost signal, around the map and under the panels */}
      <div className="alarm" aria-hidden="true" />

      <div ref={overlay} className="overlay">
        <header className="brand">
          {/* the Orientyr mark: a drone between the four ticks of a sight. The same drawing as app/icon.svg, the favicon. */}
          <svg viewBox="0 0 32 32" aria-hidden="true">
            <rect width="32" height="32" rx="8" fill="#111318"/>
            <path d="M16 3.2v5 M16 23.8v5 M3.2 16h5 M23.8 16h5" stroke="#ff4438" strokeWidth="2.6" strokeLinecap="round"/>
            <path d="M16 16 8.5 8.5 M16 16 23.5 8.5 M16 16 8.5 23.5 M16 16 23.5 23.5" stroke="#f4f5f7" strokeWidth="2.6"/>
            <g fill="#111318" stroke="#f4f5f7" strokeWidth="2.2">
            <circle cx="8.5" cy="8.5" r="3.5"/><circle cx="23.5" cy="8.5" r="3.5"/>
            <circle cx="8.5" cy="23.5" r="3.5"/><circle cx="23.5" cy="23.5" r="3.5"/>
            </g>
            <circle cx="16" cy="16" r="4.8" fill="#f4f5f7"/>
            <circle cx="16" cy="16" r="2.3" fill="#ff4438"/>
          </svg>
          <h1>Orientyr</h1>
        </header>

        {/* the tools of the viewer, in one row at the top right: which run, which layers, which view */}
        <div className="tools">
          {runs.length > 0 && (
            <label data-part="run-picker" className="runs glass pill">
              <span>Run</span>
              <select value={runName ?? ""} onChange={(e) => openRun(e.target.value)} aria-label="Run to replay">
                {runOptions.map((name) => <option key={name} value={name}>{name || run.info.generator || "this run"}</option>)}
              </select>
              <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4" /></svg>
            </label>
          )}

          <div ref={layersBox} data-part="layers" className="layers">
            <button type="button" className="glass pill press" aria-expanded={layersOpen} aria-controls="layers-list"
                    onClick={() => setLayersOpen((v) => !v)}>
              <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2 2 5.2 8 8.4l6-3.2Z" /><path d="m2 8.2 6 3.2 6-3.2M2 11.2l6 3.2 6-3.2" /></svg>
              Layers
            </button>
            {/* always in the page, shown when open: each row is a switch, and its mark is the legend of that layer */}
            <ul id="layers-list" className="glass" hidden={!layersOpen}>
              {LAYERS.map((l, i) => offered.includes(l.id) && (
                <li key={l.id}>
                  <button type="button" data-layer={l.id} aria-pressed={layers[l.id]} onClick={() => toggleLayer(l.id)}>
                    <i className={`mark mark-${l.id}`} aria-hidden="true" />
                    <span>{l.label}<small>{l.note}</small></span>
                    <kbd>{i + 1}</kbd>
                  </button>
                </li>
              ))}
            </ul>
          </div>

          <button type="button" className="follow glass pill press" aria-pressed={follow} onClick={() => setFollow((v) => !v)}>
            <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="2.2" /><path d="M8 1v3M8 12v3M1 8h3M12 8h3" /></svg>
            {follow ? "Show whole map" : "Follow drone"}
          </button>
        </div>

        {/* in follow mode the big map shows only the ground near the drone: the minimap keeps the whole route in sight */}
        {follow && <Minimap run={run} frame={frame} mapUrl={mapUrl} />}

        {shown.camera && (
          <section data-part="camera" className="camera glass" data-state={rec.state}>
            <div className="camera-view" style={{ aspectRatio: `${fw} / ${fh}` }}>
              {/* eslint-disable-next-line @next/next/no-img-element -- a local run file, not a site asset */}
              <img ref={camera} src={src} alt={`Drone camera, frame ${rec.idx}`} hidden={failed === src}
                   onError={() => setFailed(src)} />
              {failed !== src && matchedPoints(rec).map(([x, y], i) => <i key={i} style={{ left: `${x * 100}%`, top: `${y * 100}%` }} />)}
              <span className="noise" aria-hidden="true" />
              <span className="corners" aria-hidden="true" />
              {failed === src ? <b className="no-image">No image for this frame</b> : !fix && <b className="no-signal">No signal</b>}
            </div>
            <h2>Camera</h2>
            <p>{!fix ? "No picture" : fix.accepted ? `${fix.inliers} points matched` : "Match rejected"}</p>
          </section>
        )}
      </div>

      <section data-part="timeline" className="dock glass">
        <div className="transport">
          <button type="button" className="icon press" onClick={() => step(-1)} aria-label="Previous frame">
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M10 3 5 8l5 5" /></svg>
          </button>
          <button type="button" className="play press" onClick={toggle}>
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path className="glyph-play" d="M5 2.8v10.4L13.5 8Z" />
              <path className="glyph-pause" d="M4.5 3v10M11.5 3v10" />
            </svg>
            {playing ? "Pause" : frame === last ? "Replay" : "Play"}
          </button>
          <button type="button" className="icon press" onClick={() => step(1)} aria-label="Next frame">
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="m6 3 5 5-5 5" /></svg>
          </button>
          <p className="time"><b><Ticker value={clock(rec.t)} /></b> / {clock(run.records[last].t)}</p>
        </div>

        {/* one bar per frame: its height is how many point pairs matched, its colour is the trail's */}
        <div className="scrub">
          <div ref={chart} className="chart" onPointerDown={scrubStart} onPointerMove={(e) => scrubbing && scrubTo(e.clientX)}
               onPointerUp={scrubEnd} onPointerCancel={scrubEnd} onLostPointerCapture={scrubEnd}>
            {/* the part not flown yet is the same bars, paler: no dark plate over the chart */}
            <div className="bars" data-dense={n > 120 || undefined} style={{ "--played": `${progress}%` } as CSSProperties} aria-hidden="true">{bars}</div>
            <div className="bands" aria-hidden="true">
              {bands.map((b) => (
                <i key={b.from} data-state={b.state} style={{ left: `${(b.from / n) * 100}%`, width: `${((b.to - b.from + 1) / n) * 100}%` }} />
              ))}
            </div>
            <span className="now" style={{ transform: `translateX(${progress}%)` }} aria-hidden="true"><i /></span>
            {/* the same timeline for the keyboard and for screen readers; the pointer scrubs on the chart itself */}
            <input type="range" min={0} max={last} value={frame} aria-label="Frame" aria-valuetext={`Frame ${rec.idx}, ${STATE_LABEL[rec.state]}`}
                   onChange={(e) => jump(Number(e.target.value))} />
          </div>
          <div className="axis" aria-hidden="true">
            <span>Matched pairs per frame</span>
            {run.records.map((r, i) => i % tick === 0 && (i + 0.5) / n < 0.72 && <b key={r.idx} style={{ left: `${((i + 0.5) / n) * 100}%` }}>{r.idx}</b>)}
          </div>
        </div>

        <div className="speeds" role="group" aria-label="Playback speed" style={{ ["--i" as string]: SPEEDS.indexOf(speed as (typeof SPEEDS)[number]) }}>
          <i className="thumb" aria-hidden="true" />
          {SPEEDS.map((s) => (
            <button key={s} type="button" aria-pressed={s === speed} onClick={() => setSpeed(s)}>{`${s}×`}</button>
          ))}
        </div>
      </section>

      <aside className="side">
        <section data-part="hud" className="hud glass">
          <div className="status">
            <span data-part="state-badge" data-state={rec.state} data-idle={idle ? "" : undefined} className="badge" role="status">{idle ?? STATE_LABEL[rec.state]}</span>
            <span className="frame">Frame <Ticker value={String(rec.idx)} /> of {run.records[last].idx}</span>
          </div>
          <p className="headline" key={lastEvent?.frame ?? -1}>
            <strong>{eventTitle}</strong>
            <span>{[eventNote, `since ${clock(lastEvent?.t ?? 0)}`].filter(Boolean).join(" · ")}</span>
          </p>
          <div className="trust">
            <span>Confidence</span>
            <div><i style={{ transform: `scaleX(${trust / 100})` }} /></div>
            <b><Ticker value={`${trust} %`} /></b>
          </div>
          {/* the instruments: each value has its own small live drawing, not four equal boxes */}
          <dl className="gauges">
            <div className="g-nav">
              {/* a compass: the needle shows the heading, north is map up */}
              <svg className="compass" viewBox="-40 -40 80 80" aria-hidden="true">
                <circle r="38.5" className="bezel" />
                <circle r="34" className="dial" />
                {Array.from({ length: 36 }, (_, i) => (
                  <line key={i} y1={-34} y2={i % 9 === 0 ? -27 : i % 3 === 0 ? -29.5 : -31.5} className={i % 9 === 0 ? "major" : i % 3 === 0 ? "mid" : undefined} transform={`rotate(${i * 10})`} />
                ))}
                <text y="-15" textAnchor="middle" className="north">N</text>
                {heading != null && (
                  <g className="needle" style={{ transform: `rotate(${needle}deg)` }}>
                    {/* two halves, one lit and one in shade: the needle reads as a folded blade, sharp at any size */}
                    <path d="M0 -27 0 0 -5.4 3.6Z" className="needle-head" />
                    <path d="M0 -27 5.4 3.6 0 0Z" className="needle-head shade" />
                    <path d="M0 22 -3.8 4.4 0 2Z" className="needle-tail" />
                    <path d="M0 22 0 2 3.8 4.4Z" className="needle-tail shade" />
                  </g>
                )}
                <circle r="2.6" className="hub" />
              </svg>
              <div>
                <dt>Heading</dt>
                <dd><Ticker value={heading == null ? "–" : `${Math.round(heading)}°`} /></dd>
                <dd className="sub">{heading == null ? "not known yet" : `${compass(heading)}${fix?.accepted ? "" : ", from motion"}`}</dd>
              </div>
              {/* the whole map as a small plate: the dot is where the drone is on it */}
              <svg className="locator" viewBox={`0 0 100 ${locH}`} aria-hidden="true">
                <rect width="100" height={locH} className="loc-plate" />
                <path className="loc-grid" d={`M33.3 0V${locH}M66.7 0V${locH}M0 ${locH / 3}H100M0 ${(locH * 2) / 3}H100`} />
                {rec.located && (
                  <g className="loc-at" style={{ transform: `translate(${locX}px, ${locY}px)` }}>
                    <path className="loc-cross" d="M-15 0h-200M15 0h200M0 -15v-200M0 15v200" />
                    <circle r="15" className="loc-ping" />
                    <circle r="8" className="loc-dot" />
                  </g>
                )}
              </svg>
              <div className="g-pos">
                <dt>Position</dt>
                <dd><i>x</i><Ticker value={rec.located ? String(Math.round(rec.x)) : "–"} /></dd>
                <dd><i>y</i><Ticker value={rec.located ? String(Math.round(rec.y)) : "–"} /></dd>
                <dd className="sub">map px</dd>
              </div>
            </div>
            <div className="g-row">
              <div>
                <dt>Matched pairs</dt>
                <dd><Ticker value={fix ? String(fix.inliers) : "0"} /></dd>
                <dd className="sub">{!fix ? "no picture to match" : fix.accepted ? `${Math.round(fix.inlier_ratio * 100)} % of all matches` : "fix rejected"}</dd>
              </div>
              {/* the last frames as a strip chart: the newest value is the dot on the right */}
              <span className="strip" aria-hidden="true">
              <svg viewBox="0 0 120 34" preserveAspectRatio="none">
                <defs>
                  <linearGradient id="strip-fill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0" stopColor="currentColor" stopOpacity=".34" />
                    <stop offset="1" stopColor="currentColor" stopOpacity="0" />
                  </linearGradient>
                </defs>
                <path className="strip-grid" d="M0 4h120M0 17h120M0 30h120" />
                {strip.length > 1 && <polygon className="strip-area" fill="url(#strip-fill)" points={`${strip[0][0]},32 ${strip.map((p) => p.join(",")).join(" ")} 120,32`} />}
                {strip.length > 1 && <polyline className="strip-line" points={strip.map((p) => p.join(",")).join(" ")} />}
              </svg>
              <i className="strip-dot" style={{ top: `${Math.round((strip.at(-1)![1] / 34) * 1000) / 10}%` }} />
              </span>
            </div>
            <div className="g-row">
              <div>
                <dt>Uncertainty</dt>
                <dd><Ticker value={rec.located ? `± ${Math.round(sigma)} px` : "–"} /></dd>
                <dd className="sub"><b>{rec.located ? sigmaWord : "No position yet"}</b> · search: {search ?? "none"}</dd>
              </div>
              {/* the same blue area as on the map, in a sight: it swells while the camera is jammed */}
              <svg className="sight" viewBox="-28 -28 56 56" aria-hidden="true">
                <defs>
                  <radialGradient id="sight-fill">
                    <stop offset="0" stopColor="#4cc9f5" stopOpacity=".04" />
                    <stop offset="1" stopColor="#4cc9f5" stopOpacity=".42" />
                  </radialGradient>
                </defs>
                {/* the fixed scale: range rings and four ticks */}
                <circle r="9" className="sight-ring" /><circle r="18" className="sight-ring" /><circle r="26.5" className="sight-ring" />
                <path className="sight-cross" d="M0 -27.5v6M0 21.5v6M-27.5 0h6M21.5 0h6" />
                {/* the area itself: its edge turns slowly, and a ring runs from the drone out to the edge */}
                <g className="sight-zone" style={{ transform: `scale(${rec.located ? ringScale : 1})` }}>
                  <circle r="26" className="sight-area" fill="url(#sight-fill)" />
                  <circle r="26" className="sight-orbit" />
                  <circle r="26" className="sight-probe" />
                </g>
                <circle r="2" className="sight-dot" />
              </svg>
            </div>
          </dl>
        </section>

        <section data-part="events" className="log glass">
          <h2>Mission log <span>{log.filter((e) => e.frame <= frame).length} of {log.length}</span></h2>
          {/* a timeline: one node per event on a rail, the event itself as a title and a note */}
          <ol ref={logList}>
            {log.map((e) => {
              const [title, note] = sentence(e.text).split(/:\s*/, 2);
              return (
                <li key={e.frame} data-state={e.state} data-ahead={e.frame > frame || undefined}
                    aria-current={e === lastEvent || undefined}>
                  <button type="button" onClick={() => jump(e.frame)}>
                    <time>{clock(e.t)}</time>
                    <i className="node" />
                    <span><strong>{title}</strong>{note && <small>{note}</small>}</span>
                  </button>
                </li>
              );
            })}
          </ol>
        </section>

        <section data-part="controls" className="controls glass">
          <h2>Mission controls <span>come with the live engine</span></h2>
          <div>
            <button type="button" disabled>Start mission</button>
            <button type="button" disabled>Jam camera</button>
            <button type="button" disabled>Change matcher</button>
          </div>
        </section>

        <footer className="meta">
          <p className="run" title={`${runName ?? "run"} · ${run.info.generator || "run"} · ${n} frames · map ${mw} × ${mh} px`}>
            <span>{run.info.generator || "run"}</span><span>{n} frames</span>
          </p>
          <p className="keys" title="Shift + arrows: 10 frames · Home / End: first / last frame · , and . : slower / faster · 1 to 8: one layer">
            <span><kbd>Space</kbd>play</span><span><kbd>←</kbd><kbd>→</kbd>step</span><span><kbd>F</kbd>follow</span><span><kbd>L</kbd>layers</span>
          </p>
          {/* the legend: what the colours and shapes on the map mean. Here, off the map, it does not compete with the route. */}
          <ul className="legend" aria-label="Map legend">
            <li><i style={{ background: confidenceColor(1) }} />sure</li>
            <li><i style={{ background: confidenceColor(0.3) }} />less sure</li>
            <li><i style={{ background: confidenceColor(0) }} />guessing</li>
            <li><i className="box" />camera</li>
            <li><i className="ring" />error area</li>
          </ul>
        </footer>
      </aside>
    </main>
  );
}
