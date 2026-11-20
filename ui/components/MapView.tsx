"use client";

/**
 * The map with everything the tracker believes drawn over it, in map pixel coordinates: the SVG's viewBox
 * is the map itself, so telemetry is plotted as it is written, with no conversion. The map fills the screen.
 */
import type { Run } from "@/lib/run";
import { confidenceColor, drawAngle, headingOf, niceStep, sigmaEllipse, trail } from "@/lib/view";

const FOLLOW_ZOOM = 2.4;
const CARD_RATIO = 1.9; // the map card is about this wide for its height; a taller map is fitted by height

const pts = (points: number[][]) => points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

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
  const u = Math.max(w, h * CARD_RATIO) / 1000 / zoom;
  const labelEvery = niceStep(run.records.length, 10); // number about ten points: all of them would cover the route
  const segs = trail(run.records, frame);
  const flown = run.records.slice(0, frame + 1);
  const heading = headingOf(rec);
  const fix = rec.fix;
  const ellipse = sigmaEllipse(rec.cov);
  const start = run.records[0];
  const startOnRight = start.x > w * 0.9;
  // Follow mode keeps the drone in the middle: scale about the drone, then move it to the map centre.
  const camera = follow
    ? `translate(${w / 2}px, ${h / 2}px) scale(${zoom}) translate(${-rec.x}px, ${-rec.y}px)`
    : "none";

  return (
    <svg data-part="map" className="map" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio={follow ? "xMidYMid slice" : "xMidYMid meet"}
         role="img" aria-label="Map with the drone's route">
      <g style={{ transform: camera, transition: `transform ${Math.max(stepMs, 350)}ms ease-out` }}>
        <image href="/api/map" width={w} height={h} preserveAspectRatio="none" />
        <rect width={w} height={h} className="map-dim" />

        <g data-part="trail">
          {segs.map((s, i) => (
            <line key={i} data-part="trail-segment" x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2}
                  stroke={confidenceColor(s.confidence)} strokeWidth={4 * u} strokeLinecap="round" />
          ))}
          {flown.map((r, i) => i % labelEvery === 0 && (
            <g key={r.idx} transform={`translate(${r.x} ${r.y})`}>
              <circle r={8 * u} className="point" />
              <text fontSize={8.5 * u} className="point-num" textAnchor="middle" dominantBaseline="central">{r.idx}</text>
            </g>
          ))}
          {/* near the right edge the label goes to the left of the point, so the screen never clips it */}
          <text x={start.x + (startOnRight ? -14 : 14) * u} y={start.y - 10 * u} fontSize={14 * u}
                className="label" textAnchor={startOnRight ? "end" : "start"}>START</text>
          {frame === run.records.length - 1 && (
            <text x={rec.x + 14 * u} y={rec.y + 24 * u} fontSize={14 * u} className="label">FINISH</text>
          )}
        </g>

        {fix?.accepted && (
          <polygon data-part="footprint" className="footprint" points={pts(fix.footprint)} strokeWidth={2.4 * u} />
        )}
        {fix && !fix.accepted && (
          <g data-part="rejected-fix" className="rejected">
            <polygon points={pts(fix.footprint)} strokeWidth={2.4 * u} strokeDasharray={`${7 * u} ${5 * u}`} />
            <text x={fix.x} y={fix.y} fontSize={13 * u} className="label" textAnchor="middle" dominantBaseline="central">
              rejected: {(fix.reject_reason ?? "bad fix").replaceAll("_", " ")}
            </text>
          </g>
        )}

        <g data-part="drone" style={{ transform: `translate(${rec.x}px, ${rec.y}px)`, transition: `transform ${stepMs}ms linear` }}>
          {/* 3σ ellipse: where the drone is with ~99 % probability; it swells while the camera is jammed */}
          <ellipse rx={Math.max(ellipse.rx, 22 * u)} ry={Math.max(ellipse.ry, 17 * u)}
                   transform={`rotate(${drawAngle(ellipse.angle)})`} className="sigma" strokeWidth={2.4 * u} />
          {heading !== null && (
            <g transform={`rotate(${drawAngle(heading)})`}>
              <line y2={-52 * u} className="arrow" strokeWidth={3.2 * u} />
              <path d={`M ${-9 * u} ${-44 * u} L 0 ${-56 * u} L ${9 * u} ${-44 * u}`} className="arrow" strokeWidth={3.2 * u} />
            </g>
          )}
          <circle r={5 * u} className="body" />
        </g>
      </g>
    </svg>
  );
}
