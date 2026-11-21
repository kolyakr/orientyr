"use client";

/**
 * The map with everything the tracker believes drawn over it, in map pixel coordinates: the SVG's viewBox
 * is the map itself, so telemetry is plotted as it is written, with no conversion. The SVG lies in the stage
 * cell and may overflow it: a camera (lib/view.ts) moves and scales the map group, so the map slides under the
 * panels. In the whole-route view the camera fits the route into the free part of the stage; in follow mode it
 * keeps the drone in the middle of that free part.
 */
import { useEffect, useMemo, useState } from "react";
import type { Run } from "@/lib/run";
import { type Camera, type Rect, confidenceColor, courseOf, drawAngle, fitRoute, footprintSize, headingOf, inViewBox, niceStep, routeBox, sigmaEllipse, stateRuns, trail } from "@/lib/view";

/** The stage as Console.tsx measures it, in CSS px counted from the stage cell's top-left corner. */
export interface StageLayout {
  cell: { width: number; height: number }; // the stage cell: the svg's own box
  free: Rect; // the part of the cell that no panel covers
  view: Rect; // the whole console: the map may cover all of it, under the panels
}

const FOLLOW_ZOOM = 2.4; // follow mode, against the map fitted into the stage cell
const STAGE_RATIO = 1.36; // before the stage is measured: it is about this wide for its height
const INTRO_MS = 1500; // on load the flown route draws itself once, start to drone, in about this long
const TAIL = 18; // the newest steps of the route are drawn wide and bright, like a comet's tail: the present dominates
const ROTORS = [[-1, -1], [1, -1], [1, 1], [-1, 1]]; // a quadcopter from above: the four arms of an X

const pts = (points: number[][]) => points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
const tenth = (v: number) => Math.round(v * 10) / 10;
/** The angle equal to `next` (mod 360) that is nearest to `prev`: a turn from 359° to 1° is 2°, not a full spin back. */
const nearestTurn = (prev: number, next: number) => drawAngle(prev + ((((next - prev) % 360) + 540) % 360) - 180);

/** Short ticks at the corners of a polygon, along both of its edges: the corners of a viewfinder. */
function cornerTicks(poly: number[][], size: number): string {
  return poly.map((c, i) => {
    const arm = (to: number[]) => {
      const dx = to[0] - c[0], dy = to[1] - c[1];
      const k = Math.min(size, Math.hypot(dx, dy) / 3) / (Math.hypot(dx, dy) || 1);
      return `${(c[0] + dx * k).toFixed(1)} ${(c[1] + dy * k).toFixed(1)}`;
    };
    return `M ${arm(poly[(i + poly.length - 1) % poly.length])} L ${c[0].toFixed(1)} ${c[1].toFixed(1)} L ${arm(poly[(i + 1) % poly.length])}`;
  }).join(" ");
}

export function MapView({ run, frame, follow, stepMs, layout }: {
  run: Run;
  frame: number;
  follow: boolean;
  stepMs: number; // how long one frame stays on screen: the camera and the drone glide for this long
  layout: StageLayout | null; // null until the stage is measured (the server's HTML, the first render)
}) {
  const [w, h] = run.info.map_size;
  const rec = run.records[frame];
  // The route with half a footprint around it: the footprint drawn at the ends of the route stays in sight.
  const box = useMemo(() => routeBox(run.records, [w, h], (footprintSize(run.records) || Math.max(w, h) * 0.08) / 2), [run, w, h]);
  let camera: Camera | null = null;
  if (layout) {
    const { cell, free, view } = layout;
    const whole = fitRoute(box, [w, h], free, view);
    if (follow) {
      // closer than the whole-route view in any case, with the drone in the middle of the free stage
      const scale = Math.max(Math.min(cell.width / w, cell.height / h) * FOLLOW_ZOOM, whole.scale * 1.6);
      camera = { scale, x: free.x + free.width / 2 - rec.x * scale, y: free.y + free.height / 2 - rec.y * scale };
    } else camera = whole;
  }
  const group = camera && layout ? inViewBox(camera, [w, h], layout.cell) : null;
  // follow mode looks at the map from behind and above: the map leans back around the drone's place on the screen
  const pivot = layout ? `${tenth(layout.free.x + layout.free.width / 2)}px ${tenth(layout.free.y + layout.free.height / 2)}px` : undefined;
  // one "screen unit" in map px: markers and text keep their size on screen at any zoom, a little larger on a big stage
  const u = camera && layout
    ? Math.min(1.3, Math.max(0.85, Math.min(layout.cell.width, layout.cell.height) / 720)) / camera.scale
    : Math.max(w, h * STAGE_RATIO) / 1000;
  const labelEvery = niceStep(run.records.length, 10); // number about ten points: all of them would cover the route
  const segs = trail(run.records, frame);
  const flown = run.records.slice(0, frame + 1);
  // the marker points the way the drone moved since the last point, so it turns around when the route doubles back;
  // before the first move it shows the camera's heading
  const heading = courseOf(run.records, frame) ?? headingOf(rec);
  const fix = rec.fix;
  const ellipse = sigmaEllipse(rec.cov);
  const start = run.records[0];
  const startOnRight = start.x > w * 0.9;
  const atEnd = frame === run.records.length - 1;
  const lost = rec.state === "LOST";
  // The marker turns along the shortest arc: the drawn angle follows the heading without ever wrapping at 360°.
  const [turn, setTurn] = useState(drawAngle(heading ?? 0));
  if (heading !== null && nearestTurn(turn, heading) !== turn) setTurn(nearestTurn(turn, heading));
  // The camera's footprint, counted from the drone: it is drawn inside the drone's group and glides with it.
  const view = fix?.accepted ? fix.footprint.map(([x, y]) => [x - rec.x, y - rec.y]) : null;
  // Dead-reckoned steps are dashed. Each step is its own line, so the dash pattern is carried on from one step to
  // the next by an offset: the dashes stay even along the whole coasted stretch.
  const dash = [5.5 * u, 4.5 * u];
  let coasted = 0;
  const phase = segs.map((s) => {
    const at = coasted;
    coasted = s.coasting ? coasted + Math.hypot(s.x2 - s.x1, s.y2 - s.y1) : 0;
    return at % (dash[0] + dash[1]);
  });
  // START and FINISH: when the flight ends near its start, one label goes above its point and the other below
  const startAbove = !atEnd || start.y <= rec.y;
  // The frame at which the current state began: a new state replays its one-shot animation (the lock-on ring).
  const runs = useMemo(() => stateRuns(run.records), [run]);
  const since = runs.find((r) => frame >= r.from && frame <= r.to)?.from ?? 0;
  const transform = group ? `translate(${tenth(group.x)}px, ${tenth(group.y)}px) scale(${Math.round(group.scale * 1e4) / 1e4})` : "none";
  // The camera glides between frames and views, but not into its first measured place.
  const measured = layout !== null;
  const [glide, setGlide] = useState(false);
  useEffect(() => {
    if (!measured) return;
    const id = setTimeout(() => setGlide(true), 120);
    return () => clearTimeout(id);
  }, [measured]);

  // The intro: the route already flown at load draws itself in order. Afterwards a new segment draws at once.
  const [introFrames] = useState(frame);
  const [intro, setIntro] = useState(true);
  useEffect(() => {
    const id = setTimeout(() => setIntro(false), INTRO_MS + 900);
    return () => clearTimeout(id);
  }, []);

  return (
    <svg data-part="map" className="map" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet"
         role="img" aria-label="Map with the drone's route" data-ready={measured || undefined} data-tilt={(follow && measured) || undefined}
         style={{ ["--step" as string]: `${Math.round(stepMs)}ms`, transformOrigin: pivot }}>
      <defs>
        <clipPath id="map-clip"><rect width={w} height={h} rx={12 * u} /></clipPath>
      </defs>
      <g className="map-camera" style={{ transform, transition: glide ? `transform ${Math.round(Math.max(stepMs, 350))}ms cubic-bezier(.22, 1, .36, 1)` : "none" }}>
        <image href="/api/map" width={w} height={h} preserveAspectRatio="none" clipPath="url(#map-clip)" />
        <rect width={w} height={h} rx={12 * u} className="map-dim" />
        <rect width={w} height={h} rx={12 * u} className="map-rim" strokeWidth={1 * u} />

        <g data-part="trail">
          {/* a dark casing under the coloured route, as one element: the three colours read on any terrain */}
          {flown.length > 1 && <polyline className="trail-casing" points={pts(flown.map((r) => [r.x, r.y]))} strokeWidth={5 * u} />}
          {flown.length > 1 && <polyline className="trail-casing" points={pts(flown.slice(-TAIL - 1).map((r) => [r.x, r.y]))} strokeWidth={7 * u} />}
          {segs.map((s, i) => {
            const recent = i >= segs.length - TAIL;
            return (
              <line key={i} data-part="trail-segment" className={`seg${s.coasting ? " coast" : ""}${recent ? "" : " old"}`}
                    pathLength={s.coasting ? undefined : 1} x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2}
                    stroke={confidenceColor(s.confidence)} strokeWidth={tenth((recent ? 4 : 2.6) * u)}
                    strokeDasharray={s.coasting ? `${tenth(dash[0])} ${tenth(dash[1])}` : undefined}
                    strokeDashoffset={s.coasting ? tenth(phase[i]) : undefined}
                    style={intro && i < introFrames ? { animationDelay: `${Math.round(500 + (i / introFrames) * INTRO_MS)}ms`, animationDuration: "160ms" } : undefined} />
            );
          })}
          {/* numbered points: white chips with dark numbers, readable on any terrain. The start and the finish have their own markers, and a chip never
              lies under the drone. */}
          {flown.map((r, i) => i % labelEvery === 0 && i > 0 && !(atEnd && i === frame) && Math.hypot(r.x - rec.x, r.y - rec.y) > 26 * u && (
            <g key={r.idx} transform={`translate(${r.x} ${r.y})`} className="waypoint">
              <circle r={9.5 * u} className="point" strokeWidth={1.6 * u} />
              <text fontSize={8.6 * u} className="point-num" textAnchor="middle" dominantBaseline="central">{r.idx}</text>
            </g>
          ))}
          {/* the start: a ring. Near the right edge its label goes to the left, so the screen never clips it. */}
          <g transform={`translate(${start.x} ${start.y})`} className="marker">
            <circle r={5.4 * u} className="marker-case" strokeWidth={4.6 * u} />
            <circle r={5.4 * u} className="marker-ring" strokeWidth={2.2 * u} />
            <text x={(startOnRight ? -13 : 13) * u} y={(startAbove ? -9 : 17) * u} fontSize={10.5 * u}
                  className="label" textAnchor={startOnRight ? "end" : "start"}>START</text>
          </g>
          {/* the finish: a double ring, with its label on the far side of the start's */}
          {atEnd && (
            <g transform={`translate(${rec.x} ${rec.y})`} className="marker finish">
              <circle r={9 * u} className="marker-case" strokeWidth={3.6 * u} />
              <circle r={9 * u} className="marker-ring" strokeWidth={1.5 * u} />
              <circle r={4 * u} className="marker-case" strokeWidth={3.6 * u} />
              <circle r={4 * u} className="marker-ring" strokeWidth={1.5 * u} />
              <text x={(startOnRight ? -24 : 24) * u} y={(startAbove ? 30 : -24) * u} fontSize={10.5 * u}
                    className="label" textAnchor={startOnRight ? "end" : "start"}>FINISH</text>
            </g>
          )}
        </g>

        {fix && !fix.accepted && (
          <g data-part="rejected-fix" className="rejected">
            <polygon points={pts(fix.footprint)} strokeWidth={1.4 * u} strokeDasharray={`${7 * u} ${5 * u}`} />
            <text x={fix.x} y={fix.y} fontSize={10.5 * u} className="label" textAnchor="middle" dominantBaseline="central">
              rejected: {(fix.reject_reason ?? "bad fix").replaceAll("_", " ")}
            </text>
          </g>
        )}

        <g data-part="drone" data-state={rec.state} className="drone"
           style={{ transform: `translate(${tenth(rec.x)}px, ${tenth(rec.y)}px)`, transition: `transform ${Math.round(stepMs)}ms linear` }}>
          {/* what the camera sees: a light patch with viewfinder corners, and faint lines from the drone to them */}
          {view && (
            <g className="view">
              <path className="view-cone" d={view.map(([x, y]) => `M 0 0 L ${x.toFixed(1)} ${y.toFixed(1)}`).join(" ")} strokeWidth={0.9 * u} />
              <polygon data-part="footprint" className="footprint" points={pts(view)} strokeWidth={1.1 * u} />
              <path className="view-ticks" d={cornerTicks(view, 13 * u)} strokeWidth={2.6 * u} />
            </g>
          )}
          {/* 3σ ellipse: where the drone is with ~99 % probability; it swells while the camera is jammed */}
          <ellipse rx={tenth(Math.max(ellipse.rx, 21 * u))} ry={tenth(Math.max(ellipse.ry, 21 * u))}
                   transform={`rotate(${drawAngle(ellipse.angle)})`} className="sigma" strokeWidth={1.2 * u} />
          {/* decoration only: a sonar ring while tracking, a turning search ring while relocalizing */}
          <circle r={44 * u} className="sonar" strokeWidth={1.4 * u} />
          <circle r={44 * u} className="sonar late" strokeWidth={1.4 * u} />
          <circle r={34 * u} className="search" strokeWidth={2 * u} strokeDasharray={`${12 * u} ${9 * u}`} />
          {rec.state === "TRACKING" && since > 0 && (
            <circle key={since} r={30 * u} className="lock" strokeWidth={2.2 * u} />
          )}
          {/* the drone from above: four arms with rotor discs, a body, and a nose mark that shows the heading.
              Every part is drawn twice, a dark casing under a light shape, to read on bright and on dark ground.
              While the signal is lost it is a ghost: outlines only, dashed. */}
          <g className="craft" style={{ transform: `rotate(${turn}deg)`, transition: `transform ${Math.round(Math.max(stepMs, 200))}ms ease-out` }}>
            {heading !== null && (
              <line y1={-19 * u} y2={-44 * u} className="beam" strokeWidth={1.8 * u} strokeDasharray={lost ? `${4 * u} ${4 * u}` : undefined} />
            )}
            {[5.2, 1.8].map((width, pass) => (
              <g key={pass} className={pass ? "craft-shape" : "craft-case"} strokeWidth={width * u}
                 strokeDasharray={lost && pass ? `${2.6 * u} ${2.2 * u}` : undefined}>
                {ROTORS.map(([x, y], i) => <line key={i} x2={x * 9.5 * u} y2={y * 9.5 * u} />)}
                {ROTORS.map(([x, y], i) => <circle key={i} cx={x * 9.5 * u} cy={y * 9.5 * u} r={5 * u} className="rotor-disc" />)}
                <rect x={-3.6 * u} y={-6 * u} width={7.2 * u} height={11.5 * u} rx={3 * u} className="craft-body" />
                {heading !== null && <path d={`M 0 ${-16.5 * u} L ${4.6 * u} ${-8.5 * u} L ${-4.6 * u} ${-8.5 * u} Z`} className="craft-nose" />}
              </g>
            ))}
            {/* a blade in each disc: it turns only while tracking, as a faint shimmer */}
            {ROTORS.map(([x, y], i) => (
              <line key={i} x1={(x * 9.5 - 3.6) * u} y1={y * 9.5 * u} x2={(x * 9.5 + 3.6) * u} y2={y * 9.5 * u} className="blade" strokeWidth={1.1 * u} />
            ))}
          </g>
        </g>
      </g>
    </svg>
  );
}
