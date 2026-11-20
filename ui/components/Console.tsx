"use client";

/**
 * Mission control, as a mock-up: a recorded run (run.json + run.jsonl + frames/) replayed on the map.
 * Playback works; the mission controls are drawn but inactive until the live engine exists (M7).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import type { Run, RunRecord } from "@/lib/run";
import { STATE_LABEL, clock, confidence, confidenceColor, events, headingOf, sigmaPx, stateRuns } from "@/lib/view";
import { MapView } from "./MapView";

const SPEEDS = [1, 2, 4];
const frameUrl = (rec: RunRecord) => `/api/frame?src=${encodeURIComponent(rec.source)}`;
const num = (v: number | null | undefined, digits = 1) => (v == null ? "—" : v.toFixed(digits));

/** Distance to the simulator's true position, when the run has one (fake and synthetic runs do). */
function errorPx(rec: RunRecord): number | null {
  const t = rec.extra.truth as { x?: number; y?: number } | undefined;
  return typeof t?.x === "number" && typeof t?.y === "number" ? Math.hypot(rec.x - t.x, rec.y - t.y) : null;
}

function Readout({ label, value, unit, tone }: { label: string; value: string; unit?: string; tone?: string }) {
  return (
    <div className="readout" data-tone={tone}>
      <dt>{label}</dt>
      <dd>{value}{unit && <span className="unit">{unit}</span>}</dd>
    </div>
  );
}

export function Console({ run, initialFrame = 0 }: { run: Run; initialFrame?: number }) {
  const last = run.records.length - 1;
  const [frame, setFrame] = useState(Math.min(Math.max(initialFrame, 0), last));
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [follow, setFollow] = useState(false);
  const [showTruth, setShowTruth] = useState(true);

  const rec = run.records[frame];
  const fix = rec.fix;
  const stepMs = 1000 / ((run.info.fps ?? 1) * speed);
  const bands = useMemo(() => stateRuns(run.records), [run]);
  const log = useMemo(() => events(run.records), [run]);
  const maxInliers = useMemo(() => Math.max(1, ...run.records.map((r) => r.fix?.inliers ?? 0)), [run]);

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
    if (frame === last) return setPlaying(false);
    const id = setTimeout(() => setFrame((f) => Math.min(f + 1, last)), stepMs);
    return () => clearTimeout(id);
  }, [playing, frame, last, stepMs]);

  useEffect(() => {
    // fetch the next camera frame ahead, so the inset never flashes empty during playback
    if (frame < last) new Image().src = frameUrl(run.records[frame + 1]);
  }, [frame, last, run]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return;
      if (e.code === "Space") { e.preventDefault(); toggle(); }
      else if (e.code === "ArrowRight") step(1);
      else if (e.code === "ArrowLeft") step(-1);
      else if (e.code === "KeyF") setFollow((v) => !v);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, step]);

  const conf = confidence(rec);
  const heading = headingOf(rec);
  const err = errorPx(rec);
  const [mapW, mapH] = run.info.map_size;

  return (
    <main className="console" data-state={rec.state}>
      <header className="topbar">
        <div className="brand">
          <svg viewBox="0 0 24 24" className="brand-mark" aria-hidden="true">
            <circle cx="12" cy="12" r="9.5" /><circle cx="12" cy="12" r="4.5" />
            <path d="M12 0v6M12 18v6M0 12h6M18 12h6" /><circle cx="12" cy="12" r="1.4" className="dot" />
          </svg>
          <span className="brand-name">ORIENTYR</span>
          <span className="brand-sub">Mission control</span>
        </div>
        <div className="chips">
          <span className="chip chip-accent">GNSS denied</span>
          <span className="chip">Replay · {run.info.generator}{run.info.seed != null && ` · seed ${run.info.seed}`}</span>
          <span className="chip">Map {mapW} × {mapH} px</span>
          <span className="chip">Matcher · {rec.backend ?? "—"}</span>
        </div>
        <div data-part="state-badge" data-state={rec.state} className="state-badge" role="status">
          <span className="state-led" />{STATE_LABEL[rec.state]}
        </div>
      </header>

      <section className="stage">
        <MapView run={run} frame={frame} follow={follow} showTruth={showTruth} stepMs={playing ? stepMs : 250} />
        <div className="stage-tools">
          <button type="button" aria-pressed={follow} onClick={() => setFollow((v) => !v)}>Follow drone</button>
          <button type="button" aria-pressed={showTruth} onClick={() => setShowTruth((v) => !v)}>Ground truth</button>
        </div>
        <ul className="legend">
          <li><i style={{ background: confidenceColor(1) }} />high confidence</li>
          <li><i style={{ background: confidenceColor(0.45) }} />low</li>
          <li><i className="dashed" />dead reckoning</li>
          <li><i className="truth-key" />ground truth</li>
        </ul>
        {typeof rec.extra.event === "string" && <div className="toast" key={frame}>{rec.extra.event}</div>}
      </section>

      <aside className="side">
        <section data-part="camera" className="panel camera">
          <h2>Camera <span>{rec.source.split("/").at(-1)}</span></h2>
          <div className="camera-view">
            {/* eslint-disable-next-line @next/next/no-img-element -- a local run file, not a site asset */}
            <img src={frameUrl(rec)} alt={`Drone camera, frame ${rec.idx}`} />
            <div className="reticle" aria-hidden="true" />
            {!fix && <div className="camera-flag">No fix · dead reckoning</div>}
            {fix && !fix.accepted && <div className="camera-flag">Fix rejected · {fix.reject_reason}</div>}
          </div>
          <div className="confidence">
            <span>Confidence</span>
            <div className="bar"><div style={{ width: `${conf * 100}%`, background: confidenceColor(conf) }} /></div>
            <b>{Math.round(conf * 100)}%</b>
          </div>
        </section>

        <section data-part="hud" className="panel hud">
          <h2>Telemetry <span>frame {rec.idx} / {run.records[last].idx}</span></h2>
          <dl>
            <Readout label="Flight time" value={clock(rec.t)} />
            <Readout label="Heading" value={heading == null ? "—" : heading.toFixed(1)} unit="°" />
            <Readout label="X" value={num(rec.x)} unit="px" />
            <Readout label="Y" value={num(rec.y)} unit="px" />
            <Readout label="Speed" value={num(Math.hypot(rec.vx, rec.vy))} unit="px/s" />
            <Readout label="Scale" value={num(fix?.scale, 3)} unit="×" />
            <Readout label="Inliers" value={fix ? String(fix.inliers) : "—"} tone={fix && !fix.accepted ? "bad" : undefined} />
            <Readout label="Reproj. error" value={num(fix?.reproj_px, 2)} unit="px" />
            <Readout label="Uncertainty σ" value={num(sigmaPx(rec.cov))} unit="px" tone={rec.state === "TRACKING" ? undefined : "warn"} />
            <Readout label="Error vs truth" value={num(err)} unit="px" tone={err != null && err > 15 ? "warn" : undefined} />
            <Readout label="Search" value={String(rec.extra.search ?? "—")} />
            <Readout label="Compute" value={num(rec.ms, 0)} unit="ms" />
          </dl>
        </section>

        <section data-part="events" className="panel events">
          <h2>Mission log</h2>
          <ol>
            <li data-past="true"><time>{clock(run.records[0].t)}</time>mission start: tracking</li>
            {log.map((e) => (
              <li key={e.frame} data-state={e.state} data-past={e.frame <= frame}>
                <button type="button" onClick={() => { setPlaying(false); setFrame(e.frame); }}>
                  <time>{clock(e.t)}</time>{e.text}
                </button>
              </li>
            ))}
          </ol>
        </section>
      </aside>

      <footer className="dock">
        <section data-part="timeline" className="timeline">
          <div className="transport">
            <button type="button" onClick={() => step(-1)} aria-label="Previous frame">‹</button>
            <button type="button" className="play" onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
              {playing ? "❚❚" : "▶"}
            </button>
            <button type="button" onClick={() => step(1)} aria-label="Next frame">›</button>
            <button type="button" className="speed" aria-label="Playback speed"
                    onClick={() => setSpeed((s) => SPEEDS[(SPEEDS.indexOf(s) + 1) % SPEEDS.length])}>{speed}×</button>
          </div>
          <div className="scrubber">
            {/* one bar per frame: height = inliers, colour = confidence; the band below is the tracker state */}
            <div className="bars" aria-hidden="true">
              {run.records.map((r, i) => (
                <i key={r.idx} data-now={i === frame} data-past={i <= frame}
                   style={{ height: `${8 + 92 * ((r.fix?.accepted ? r.fix.inliers : 0) / maxInliers)}%`,
                            background: confidenceColor(confidence(r)) }} />
              ))}
            </div>
            <div className="bands" aria-hidden="true">
              {bands.map((b) => <i key={b.from} data-state={b.state} style={{ flexGrow: b.to - b.from + 1 }} />)}
            </div>
            <input type="range" min={0} max={last} value={frame} aria-label="Frame"
                   onChange={(e) => { setPlaying(false); setFrame(Number(e.target.value)); }} />
          </div>
          <div className="timecode">{clock(rec.t)}<span> / {clock(run.records[last].t)}</span></div>
        </section>

        <section data-part="controls" className="controls" title="Inactive in the mock-up: they arrive with the live engine">
          <span className="controls-note">Live controls · soon</span>
          <button type="button" disabled>Start mission</button>
          <button type="button" disabled>Simulate jamming</button>
          <button type="button" disabled>Switch matcher</button>
          <button type="button" disabled>Export report</button>
        </section>
      </footer>
    </main>
  );
}
