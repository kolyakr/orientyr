"use client";

/**
 * The map with everything the tracker believes drawn over it, in map pixel coordinates: the SVG's viewBox
 * is the map itself, so telemetry is plotted as it is written, with no conversion.
 */
import type { Run, RunRecord } from "@/lib/run";
import { confidenceColor, headingOf, sigmaPx, trail } from "@/lib/view";

const FOLLOW_ZOOM = 3;
const LABEL_EVERY = 5; // number every 5th point: all 50 would cover the route

interface Truth {
  x: number;
  y: number;
}

function truthOf(rec: RunRecord): Truth | null {
  const t = rec.extra.truth as Partial<Truth> | undefined;
  return typeof t?.x === "number" && typeof t?.y === "number" ? { x: t.x, y: t.y } : null;
}

const pts = (points: number[][]) => points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

export function MapView({ run, frame, follow, showTruth, stepMs }: {
  run: Run;
  frame: number;
  follow: boolean;
  showTruth: boolean;
  stepMs: number; // how long one frame stays on screen: the camera and the drone glide for this long
}) {
  const [w, h] = run.info.map_size;
  const rec = run.records[frame];
  const zoom = follow ? FOLLOW_ZOOM : 1;
  const u = w / 1000 / zoom; // one "screen unit" in map px: marker and text sizes stay the same when zoomed
  const segs = trail(run.records, frame);
  const flown = run.records.slice(0, frame + 1);
  const heading = headingOf(rec);
  const fix = rec.fix;
  const sigma = sigmaPx(rec.cov);
  const truth = flown.map(truthOf).filter((t): t is Truth => t !== null);
  const startOnRight = run.records[0].x > w * 0.92;
  const glide = { transition: `transform ${stepMs}ms linear` };
  // Follow mode keeps the drone in the middle: scale about the drone, then move it to the map centre.
  const camera = follow
    ? `translate(${w / 2}px, ${h / 2}px) scale(${zoom}) translate(${-rec.x}px, ${-rec.y}px)`
    : "none";

  return (
    <svg data-part="map" className="map" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet"
         role="img" aria-label="Map with the drone's route">
      <defs>
        <pattern id="grid" width={w / 12} height={w / 12} patternUnits="userSpaceOnUse">
          <path d={`M ${w / 12} 0 H 0 V ${w / 12}`} className="map-grid" />
        </pattern>
        <clipPath id="map-clip"><rect width={w} height={h} rx={6 * u * zoom} /></clipPath>
      </defs>
      <g clipPath="url(#map-clip)">
        <g style={{ transform: camera, transition: `transform ${Math.max(stepMs, 350)}ms ease-out` }}>
          <image href="/api/map" width={w} height={h} preserveAspectRatio="none" />
          <rect width={w} height={h} className="map-dim" />
          <rect width={w} height={h} fill="url(#grid)" />

          {rec.extra.search === "global" && (
            <g data-part="global-search" className="scan">
              <rect width={w} height={h} className="scan-frame" strokeWidth={3 * u} />
              <rect width={w * 0.08} height={h} className="scan-beam" />
            </g>
          )}

          {showTruth && truth.length > 1 && (
            <polyline data-part="truth" className="truth" points={pts(truth.map((t) => [t.x, t.y]))} />
          )}

          <g data-part="trail">
            <polyline className="trail-halo" points={pts(flown.map((r) => [r.x, r.y]))} strokeWidth={7 * u} />
            {segs.map((s, i) => (
              <line key={i} data-part="trail-segment" x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2}
                    stroke={confidenceColor(s.confidence)} strokeWidth={3.2 * u} strokeLinecap="round"
                    strokeDasharray={s.coasting ? `${1.5 * u} ${7 * u}` : undefined} />
            ))}
            {flown.map((r, i) => (
              <g key={r.idx}>
                <circle cx={r.x} cy={r.y} r={(i % LABEL_EVERY === 0 ? 3.4 : 2) * u} className="trail-dot"
                        strokeWidth={1.2 * u} />
                {i % LABEL_EVERY === 0 && i > 0 && (
                  <text x={r.x + 7 * u} y={r.y - 7 * u} fontSize={11 * u} className="trail-num">{r.idx}</text>
                )}
              </g>
            ))}
            <g transform={`translate(${run.records[0].x} ${run.records[0].y})`}>
              <circle r={7 * u} className="start" strokeWidth={2 * u} />
              {/* near the right edge the label goes to the left of the point, so the map never clips it */}
              <text x={(startOnRight ? -11 : 11) * u} y={-9 * u} fontSize={11 * u} className="marker-label"
                    textAnchor={startOnRight ? "end" : "start"}>START</text>
            </g>
            {frame === run.records.length - 1 && (
              <g transform={`translate(${rec.x} ${rec.y})`}>
                <rect x={-7 * u} y={-7 * u} width={14 * u} height={14 * u} className="finish" strokeWidth={2 * u} />
                <text x={11 * u} y={18 * u} fontSize={11 * u} className="marker-label">FINISH</text>
              </g>
            )}
          </g>

          {fix?.accepted && (
            <polygon data-part="footprint" className="footprint" points={pts(fix.footprint)} strokeWidth={1.6 * u} />
          )}
          {fix && !fix.accepted && (
            <g data-part="rejected-fix" className="rejected">
              <polygon points={pts(fix.footprint)} strokeWidth={1.6 * u} strokeDasharray={`${5 * u} ${4 * u}`} />
              <line x1={rec.x} y1={rec.y} x2={fix.x} y2={fix.y} strokeWidth={1.2 * u}
                    strokeDasharray={`${2 * u} ${5 * u}`} />
              <text x={fix.x} y={fix.y} fontSize={11 * u} textAnchor="middle" dominantBaseline="middle">
                REJECTED · {fix.reject_reason ?? "bad fix"}
              </text>
            </g>
          )}

          <g data-part="drone" data-state={rec.state} className="drone"
             style={{ transform: `translate(${rec.x}px, ${rec.y}px)`, ...glide }}>
            {/* 3σ: where the drone is with ~99 % probability; it swells while the camera is jammed */}
            <circle r={Math.max(3 * sigma, 9 * u)} className="sigma" strokeWidth={1.2 * u}
                    strokeDasharray={`${3 * u} ${3 * u}`} />
            <circle r={16 * u} className="pulse" strokeWidth={1.5 * u} />
            {heading !== null && (
              <g transform={`rotate(${heading})`}>
                <line y2={-30 * u} className="heading" strokeWidth={2 * u} />
                <path d={`M 0 ${-38 * u} L ${5.5 * u} ${-27 * u} L ${-5.5 * u} ${-27 * u} Z`} className="heading-tip" />
              </g>
            )}
            <circle r={6 * u} className="body" strokeWidth={2 * u} />
          </g>
        </g>
      </g>
    </svg>
  );
}
