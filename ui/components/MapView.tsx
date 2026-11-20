"use client";

/**
 * The map with everything the tracker believes drawn over it, in map pixel coordinates: the SVG's viewBox
 * is the map itself, so telemetry is plotted as it is written, with no conversion. The SVG is fitted into the
 * free part of the screen (the "stage") and may overflow it: in follow mode the zoomed map slides under the
 * glass panels, while the drone stays in the middle of the stage.
 */
import { useEffect, useMemo, useState } from "react";
import type { Run } from "@/lib/run";
import { confidenceColor, drawAngle, headingOf, niceStep, sigmaEllipse, stateRuns, trail } from "@/lib/view";

const FOLLOW_ZOOM = 2.4;
const STAGE_RATIO = 1.36; // the stage is about this wide for its height; a taller map is fitted by height
const INTRO_MS = 1500; // on load the flown route draws itself once, start to drone, in about this long

const pts = (points: number[][]) => points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
const tenth = (v: number) => Math.round(v * 10) / 10;

export function MapView({ run, frame, follow, stepMs }: {
  run: Run;
  frame: number;
  follow: boolean;
  stepMs: number; // how long one frame stays on screen: the camera and the drone glide for this long
}) {
  const [w, h] = run.info.map_size;
  const rec = run.records[frame];
  const zoom = follow ? FOLLOW_ZOOM : 1;
  // one "screen unit" in map px: marker and text sizes stay the same when zoomed, and on a tall map
  const u0 = Math.max(w, h * STAGE_RATIO) / 1000;
  const u = u0 / zoom;
  const labelEvery = niceStep(run.records.length, 10); // number about ten points: all of them would cover the route
  const segs = trail(run.records, frame);
  const flown = run.records.slice(0, frame + 1);
  const heading = headingOf(rec);
  const fix = rec.fix;
  const ellipse = sigmaEllipse(rec.cov);
  const start = run.records[0];
  const startOnRight = start.x > w * 0.9;
  // The frame at which the current state began: a new state replays its one-shot animation (the lock-on ring).
  const runs = useMemo(() => stateRuns(run.records), [run]);
  const since = runs.find((r) => frame >= r.from && frame <= r.to)?.from ?? 0;
  // Follow mode keeps the drone in the middle: scale about the drone, then move it to the map centre.
  const camera = follow
    ? `translate(${w / 2}px, ${h / 2}px) scale(${zoom}) translate(${tenth(-rec.x)}px, ${tenth(-rec.y)}px)`
    : "none";

  // The intro: the route already flown at load draws itself in order. Afterwards a new segment draws at once.
  const [introFrames] = useState(frame);
  const [intro, setIntro] = useState(true);
  useEffect(() => {
    const id = setTimeout(() => setIntro(false), INTRO_MS + 900);
    return () => clearTimeout(id);
  }, []);

  return (
    <svg data-part="map" className="map" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet"
         role="img" aria-label="Map with the drone's route" style={{ ["--step" as string]: `${Math.round(stepMs)}ms` }}>
      <defs>
        <clipPath id="map-clip"><rect width={w} height={h} rx={14 * u0} /></clipPath>
      </defs>
      <g className="map-camera" style={{ transform: camera, transition: `transform ${Math.round(Math.max(stepMs, 350))}ms cubic-bezier(.22, 1, .36, 1)` }}>
        <image href="/api/map" width={w} height={h} preserveAspectRatio="none" clipPath="url(#map-clip)" />
        <rect width={w} height={h} rx={14 * u0} className="map-dim" />
        <rect width={w} height={h} rx={14 * u0} className="map-rim" strokeWidth={1.2 * u0} />

        <g data-part="trail">
          {/* a dark casing under the coloured route, as one element: the three colours read on any terrain */}
          {flown.length > 1 && <polyline className="trail-casing" points={pts(flown.map((r) => [r.x, r.y]))} strokeWidth={6.4 * u} />}
          {segs.map((s, i) => (
            <line key={i} data-part="trail-segment" className="seg" pathLength={1} x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2}
                  stroke={confidenceColor(s.confidence)} strokeWidth={3.4 * u}
                  style={intro && i < introFrames ? { animationDelay: `${Math.round(500 + (i / introFrames) * INTRO_MS)}ms`, animationDuration: "160ms" } : undefined} />
          ))}
          {flown.map((r, i) => i % labelEvery === 0 && (
            <g key={r.idx} transform={`translate(${r.x} ${r.y})`} className="waypoint">
              <circle r={8.2 * u} className="point" strokeWidth={1.2 * u} />
              <text fontSize={8.4 * u} className="point-num" textAnchor="middle" dominantBaseline="central">{r.idx}</text>
            </g>
          ))}
          {/* near the right edge the label goes to the left of the point, so the screen never clips it */}
          <text x={start.x + (startOnRight ? -14 : 14) * u} y={start.y - 12 * u} fontSize={12.5 * u}
                className="label" textAnchor={startOnRight ? "end" : "start"}>START</text>
          {frame === run.records.length - 1 && (
            <text x={rec.x + 16 * u} y={rec.y + 26 * u} fontSize={12.5 * u} className="label finish">FINISH</text>
          )}
        </g>

        {fix?.accepted && (
          <polygon data-part="footprint" className="footprint" points={pts(fix.footprint)} strokeWidth={2 * u} />
        )}
        {fix && !fix.accepted && (
          <g data-part="rejected-fix" className="rejected">
            <polygon points={pts(fix.footprint)} strokeWidth={2 * u} strokeDasharray={`${7 * u} ${5 * u}`} />
            <text x={fix.x} y={fix.y} fontSize={12 * u} className="label" textAnchor="middle" dominantBaseline="central">
              rejected: {(fix.reject_reason ?? "bad fix").replaceAll("_", " ")}
            </text>
          </g>
        )}

        <g data-part="drone" data-state={rec.state} className="drone"
           style={{ transform: `translate(${tenth(rec.x)}px, ${tenth(rec.y)}px)`, transition: `transform ${Math.round(stepMs)}ms linear` }}>
          {/* 3σ ellipse: where the drone is with ~99 % probability; it swells while the camera is jammed */}
          <ellipse rx={tenth(Math.max(ellipse.rx, 22 * u))} ry={tenth(Math.max(ellipse.ry, 17 * u))}
                   transform={`rotate(${drawAngle(ellipse.angle)})`} className="sigma" strokeWidth={2 * u} />
          {/* decoration only: a sonar ring while tracking, a turning search ring while relocalizing */}
          <circle r={44 * u} className="sonar" strokeWidth={1.6 * u} />
          <circle r={44 * u} className="sonar late" strokeWidth={1.6 * u} />
          <circle r={34 * u} className="search" strokeWidth={2.2 * u} strokeDasharray={`${12 * u} ${9 * u}`} />
          {rec.state === "TRACKING" && since > 0 && (
            <circle key={since} r={30 * u} className="lock" strokeWidth={2.4 * u} />
          )}
          {heading !== null && (
            <g transform={`rotate(${drawAngle(heading)})`}>
              <line y1={-14 * u} y2={-46 * u} className="beam" strokeWidth={2.4 * u} />
              <path d={`M 0 ${-15 * u} L ${9.5 * u} ${9 * u} L 0 ${4 * u} L ${-9.5 * u} ${9 * u} Z`} className="puck" strokeWidth={1.4 * u} />
            </g>
          )}
          {heading === null && <circle r={6 * u} className="puck" strokeWidth={1.4 * u} />}
        </g>
      </g>
    </svg>
  );
}
