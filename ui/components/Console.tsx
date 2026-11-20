"use client";

/**
 * Mission control, as a mock-up: a recorded run (run.json + run.jsonl + frames/) replayed on the map.
 * The map is the scene; everything else floats over it as glass: the brand, the legend and the camera on the
 * left, the timeline dock under the map, and on the right what the tracker knows at this frame.
 * Playback works; the mission controls are drawn but inactive until the live engine exists (M7).
 */
import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import type { Run, RunRecord } from "@/lib/run";
import { STATE_LABEL, clock, compass, confidence, confidenceColor, drawAngle, events, headingOf, niceStep, sigmaPx, stateRuns } from "@/lib/view";
import { MapView } from "./MapView";

const SPEEDS = [1, 2, 4, 8];
const MAX_DOTS = 40;
const frameUrl = (rec: RunRecord) => `/api/frame?src=${encodeURIComponent(rec.source)}`;
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

/**
 * The lens filter of the small glass pills: the backdrop is pushed inwards along the edges, as a thick glass
 * edge bends what is under it. The displacement map is a tiny SVG stretched over the element: red moves x,
 * green moves y, mid-grey moves nothing. Chromium only (CSS turns it on under [data-lens]); elsewhere the
 * pills keep a plain blur.
 */
const LENS_MAP = `data:image/svg+xml,${encodeURIComponent(
  `<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" preserveAspectRatio="none">` +
  `<defs><linearGradient id="x" x1="0" x2="1" y1="0" y2="0">` +
  `<stop offset="0" stop-color="#f00"/><stop offset=".04" stop-color="#c00"/><stop offset=".12" stop-color="#800000"/>` +
  `<stop offset=".88" stop-color="#800000"/><stop offset=".96" stop-color="#400000"/><stop offset="1" stop-color="#000"/></linearGradient>` +
  `<linearGradient id="y" x1="0" x2="0" y1="0" y2="1">` +
  `<stop offset="0" stop-color="#0f0"/><stop offset=".1" stop-color="#00c000"/><stop offset=".3" stop-color="#008000"/>` +
  `<stop offset=".7" stop-color="#008000"/><stop offset=".9" stop-color="#004000"/><stop offset="1" stop-color="#000"/></linearGradient></defs>` +
  `<rect width="100" height="100" fill="url(#x)"/><rect width="100" height="100" fill="url(#y)" style="mix-blend-mode:screen"/></svg>`,
)}`;

export function Console({ run, initialFrame = 0 }: { run: Run; initialFrame?: number }) {
  const n = run.records.length;
  const last = n - 1;
  const [frame, setFrame] = useState(Math.min(Math.max(initialFrame, 0), last));
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [follow, setFollow] = useState(false);

  const rec = run.records[frame];
  const fix = rec.fix;
  const stepMs = 1000 / ((run.info.fps ?? 1) * speed);
  const log = useMemo(() => events(run.records), [run]);
  const maxPairs = useMemo(() => Math.max(1, ...run.records.map((r) => r.fix?.inliers ?? 0)), [run]);
  const lastEvent = log.findLast((e) => e.frame <= frame);

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
    if (frame < last) new Image().src = frameUrl(run.records[frame + 1]);
  }, [frame, last, run]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === "Space") { e.preventDefault(); toggle(); }
      else if (e.code === "ArrowRight") step(1);
      else if (e.code === "ArrowLeft") step(-1);
      else if (e.code === "KeyF") setFollow((v) => !v);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, step]);

  const heading = headingOf(rec);
  const trust = Math.round(confidence(rec) * 100);
  const [mw, mh] = run.info.map_size;
  const tick = niceStep(n, 6);
  const [fw, fh] = run.info.frame_size ?? [1, 1];
  const tiles: { label: string; value: string; sub: string; turn?: number }[] = [
    { label: "Position", value: `${Math.round(rec.x)}, ${Math.round(rec.y)}`, sub: "map px, x and y" },
    heading == null
      ? { label: "Heading", value: "–", sub: "not known yet" }
      : { label: "Heading", value: `${Math.round(heading)}°`, sub: `${compass(heading)}${fix?.accepted ? "" : ", from motion"}`, turn: heading },
    { label: "Matched pairs", value: fix ? String(fix.inliers) : "0",
      sub: !fix ? "no picture to match" : fix.accepted ? `${Math.round(fix.inlier_ratio * 100)} % of all matches` : "fix rejected" },
    { label: "Uncertainty", value: `± ${Math.round(sigmaPx(rec.cov))} px`, sub: `search: ${String(rec.extra.search ?? "none")}` },
  ];

  // The timeline is drawn once: the playhead and the curtain over the frames ahead move by transform alone.
  const bars = useMemo(() => run.records.map((r) => (
    <i key={r.idx} style={{ height: `${r.fix ? Math.max(10, (r.fix.inliers / maxPairs) * 100) : 6}%`, background: confidenceColor(confidence(r)) }} />
  )), [run, maxPairs]);
  const bands = useMemo(() => stateRuns(run.records), [run]);
  const progress = Math.round(((frame + 0.5) / n) * 100000) / 1000; // % of the timeline, rounded for a stable transform

  const root = useRef<HTMLElement>(null);
  useEffect(() => {
    // Lens refraction needs an SVG filter as a backdrop-filter: only Chromium draws that.
    if (root.current && "chrome" in window) root.current.dataset.lens = "";
  }, []);
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
          onPointerMove={light}>
      <svg className="defs" width="0" height="0" aria-hidden="true">
        <filter id="glass-lens" x="0" y="0" width="1" height="1" primitiveUnits="objectBoundingBox" colorInterpolationFilters="sRGB">
          <feImage href={LENS_MAP} x="0" y="0" width="1" height="1" preserveAspectRatio="none" result="map" />
          <feDisplacementMap in="SourceGraphic" in2="map" scale="0.16" xChannelSelector="R" yChannelSelector="G" />
        </filter>
      </svg>

      {/* a blurred copy of the map behind everything: around the map the scene has soft edges, not black bars */}
      <div className="backdrop" aria-hidden="true" />
      <div className="stage">
        <MapView run={run} frame={frame} follow={follow} stepMs={playing ? stepMs : 250} />
      </div>
      {/* the red breath of a lost signal: under the panels, so it shows through the glass */}
      <div className="alarm" aria-hidden="true" />

      <div className="overlay">
        <header className="brand glass lens">
          <svg viewBox="0 0 32 32" aria-hidden="true">
            <circle cx="16" cy="16" r="11.5" />
            <path d="M16 2v6M16 24v6M2 16h6M24 16h6" />
            <path className="needle" d="m16 9 4 9-4-2.2L12 18Z" />
          </svg>
          <div>
            <p>Orientyr / replay</p>
            <h1>Live tracking</h1>
          </div>
        </header>

        <ul className="legend glass lens" aria-label="Map legend">
          <li><i style={{ background: confidenceColor(1) }} />sure</li>
          <li><i style={{ background: confidenceColor(0.3) }} />less sure</li>
          <li><i style={{ background: confidenceColor(0) }} />guessing</li>
          <li><i className="box" />camera view</li>
          <li><i className="ring" />where it can be</li>
        </ul>

        <button type="button" className="follow glass lens press" aria-pressed={follow} onClick={() => setFollow((v) => !v)}>
          <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="2.2" /><path d="M8 1v3M8 12v3M1 8h3M12 8h3" /></svg>
          {follow ? "Show whole map" : "Follow drone"}
        </button>

        <section data-part="camera" className="camera glass" data-state={rec.state}>
          <div className="camera-view" style={{ aspectRatio: `${fw} / ${fh}` }}>
            {/* eslint-disable-next-line @next/next/no-img-element -- a local run file, not a site asset */}
            <img src={frameUrl(rec)} alt={`Drone camera, frame ${rec.idx}`} />
            {matchedPoints(rec).map(([x, y], i) => <i key={i} style={{ left: `${x * 100}%`, top: `${y * 100}%` }} />)}
            <span className="noise" aria-hidden="true" />
            <span className="corners" aria-hidden="true" />
            {!fix && <b className="no-signal">No signal</b>}
          </div>
          <h2>Camera</h2>
          <p>{!fix ? "No picture" : fix.accepted ? `${fix.inliers} points matched` : "Match rejected"}</p>
        </section>
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
          <div className="chart">
            <div className="bars" data-dense={n > 120 || undefined} aria-hidden="true">{bars}</div>
            <span className="ahead" style={{ transform: `scaleX(${Math.round((100 - progress) * 1000) / 100000})` }} aria-hidden="true" />
            <div className="bands" aria-hidden="true">
              {bands.map((b) => (
                <i key={b.from} data-state={b.state} style={{ left: `${(b.from / n) * 100}%`, width: `${((b.to - b.from + 1) / n) * 100}%` }} />
              ))}
            </div>
            <span className="now" style={{ transform: `translateX(${progress}%)` }} aria-hidden="true"><i /></span>
            <input type="range" min={0} max={last} value={frame} aria-label="Frame"
                   onChange={(e) => jump(Number(e.target.value))} />
          </div>
          <div className="axis" aria-hidden="true">
            <span>Matched pairs per frame</span>
            {run.records.map((r, i) => i % tick === 0 && (i + 0.5) / n < 0.72 && <b key={r.idx} style={{ left: `${((i + 0.5) / n) * 100}%` }}>{r.idx}</b>)}
          </div>
        </div>

        <div className="speeds" role="group" aria-label="Playback speed" style={{ ["--i" as string]: SPEEDS.indexOf(speed) }}>
          <i className="thumb" aria-hidden="true" />
          {SPEEDS.map((s) => (
            <button key={s} type="button" aria-pressed={s === speed} onClick={() => setSpeed(s)}>{s}×</button>
          ))}
        </div>
      </section>

      <aside className="side">
        <section data-part="hud" className="hud glass">
          <div className="status">
            <span data-part="state-badge" data-state={rec.state} className="badge" role="status">{STATE_LABEL[rec.state]}</span>
            <span className="frame">Frame <Ticker value={String(rec.idx)} /> of {run.records[last].idx}</span>
          </div>
          <p className="headline" key={lastEvent?.frame ?? -1}>{sentence(lastEvent ? lastEvent.text : "mission start: tracking")}</p>
          <div className="trust">
            <span>Confidence</span>
            <div><i style={{ transform: `scaleX(${trust / 100})` }} /></div>
            <b><Ticker value={`${trust} %`} /></b>
          </div>
          <dl className="tiles">
            {tiles.map((t) => (
              <div key={t.label}>
                <dt>{t.label}</dt>
                <dd>
                  <Ticker value={t.value} />
                  {t.turn != null && (
                    <svg viewBox="0 0 16 16" style={{ transform: `rotate(${drawAngle(t.turn)}deg)` }} aria-hidden="true"><path d="M8 13V3M4 7l4-4 4 4" /></svg>
                  )}
                </dd>
                <dd className="sub">{t.sub}</dd>
              </div>
            ))}
          </dl>
        </section>

        <section data-part="events" className="log glass">
          <h2>Mission log</h2>
          <ol>
            {log.map((e) => (
              <li key={e.frame} data-state={e.state} data-ahead={e.frame > frame || undefined}
                  aria-current={e === lastEvent || undefined}>
                <button type="button" onClick={() => jump(e.frame)}>
                  <time>{clock(e.t)}</time>
                  <span>{sentence(e.text)}</span>
                </button>
              </li>
            ))}
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
          <p>{run.info.generator ?? "run"} · {n} frames · map {mw} × {mh} px</p>
          <p className="keys"><kbd>Space</kbd> play <kbd>←</kbd><kbd>→</kbd> step <kbd>F</kbd> follow</p>
        </footer>
      </aside>
    </main>
  );
}
