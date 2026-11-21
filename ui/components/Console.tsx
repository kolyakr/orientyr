"use client";

/**
 * Mission control, as a mock-up: a recorded run (run.json + run.jsonl + frames/) replayed on the map.
 * The map is the scene; everything else lies over it as dark glass panels: the brand, the legend and the camera
 * on the left, the timeline dock under the map, and on the right what the tracker knows at this frame.
 * Playback works; the mission controls are drawn but inactive until the live engine exists (M7).
 */
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import type { Run, RunRecord } from "@/lib/run";
import { STATE_LABEL, clock, compass, confidence, confidenceColor, drawAngle, events, headingOf, niceStep, sigmaPx, stateRuns } from "@/lib/view";
import { MapView, type StageLayout } from "./MapView";

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
      let left = 0, top = 0;   // the left column (brand, legend, camera) and the row of the follow button
      for (const el of Array.from(cell.children) as HTMLElement[]) {
        if (el.classList.contains("follow")) top = el.offsetTop + el.offsetHeight;
        else left = Math.max(left, el.offsetLeft + el.offsetWidth);
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
      <div className="stage">
        <MapView run={run} frame={frame} follow={follow} stepMs={playing ? stepMs : 250} layout={layout} />
      </div>
      {/* the red breath of a lost signal, around the map and under the panels */}
      <div className="alarm" aria-hidden="true" />

      <div ref={overlay} className="overlay">
        <header className="brand glass pill">
          {/* the Orientyr mark: a drone in a sight. The same drawing as app/icon.svg, the favicon. */}
          <svg viewBox="0 0 32 32" aria-hidden="true">
            <circle cx="16" cy="16" r="15" fill="#111318"/>
            <circle cx="16" cy="16" r="12" fill="none" stroke="#ff4438" strokeWidth="1.8"/>
            <path d="M16 2v5 M16 25v5 M2 16h5 M25 16h5" stroke="#ff4438" strokeWidth="2" strokeLinecap="round"/>
            <g fill="none" stroke="#F0F1F3" strokeWidth="2.1" strokeLinecap="round">
            <path d="M14 14L9 9 M18 14L23 9 M14 18L9 23 M18 18L23 23"/>
            <circle cx="8" cy="8" r="2.3"/><circle cx="24" cy="8" r="2.3"/>
            <circle cx="8" cy="24" r="2.3"/><circle cx="24" cy="24" r="2.3"/>
            <path d="M13 14L16 12L19 14V18L16 20L13 18Z"/>
            </g>
            <circle cx="16" cy="16" r="1.5" fill="#ff4438"/>
          </svg>
          <div>
            <p>Orientyr / replay</p>
            <h1>Live tracking</h1>
          </div>
        </header>

        <ul className="legend glass pill" aria-label="Map legend">
          <li><i style={{ background: confidenceColor(1) }} />sure</li>
          <li><i style={{ background: confidenceColor(0.3) }} />less sure</li>
          <li><i style={{ background: confidenceColor(0) }} />guessing</li>
          <li><i className="box" />camera view</li>
          <li><i className="ring" />where it can be</li>
        </ul>

        <button type="button" className="follow glass pill press" aria-pressed={follow} onClick={() => setFollow((v) => !v)}>
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
