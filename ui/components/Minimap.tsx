"use client";

/**
 * The minimap of follow mode: the whole route's area, small, with the route flown so far. In follow mode the big
 * map shows only the ground near the drone; this panel keeps the answer to "where on the flight are we".
 * Drawn in map pixel coordinates like the big map; the strokes keep their width on screen (non-scaling).
 */
import { useMemo } from "react";
import type { Run } from "@/lib/run";
import { routeBox, trailRuns } from "@/lib/view";

const pts = (points: number[][]) => points.map(([x, y]) => `${x.toFixed(0)},${y.toFixed(0)}`).join(" ");
/** A dot of a fixed screen size at a map point: a zero-length round-capped stroke. */
const dot = (x: number, y: number) => `M ${x.toFixed(0)} ${y.toFixed(0)} h .01`;

export function Minimap({ run, frame, mapUrl }: { run: Run; frame: number; mapUrl: string }) {
  const [w, h] = run.info.map_size;
  const rec = run.records[frame];
  // the route with a wide margin of ground around it: enough to recognise the place
  const box = useMemo(() => routeBox(run.records, [w, h], Math.max(w, h) * 0.1), [run, w, h]);
  const runs = trailRuns(run.records, frame);
  const start = run.records.find((r) => r.located); // none: the run was never localized
  const fix = rec.fix;
  return (
    <section data-part="minimap" className="minimap glass" data-state={rec.state}>
      <svg viewBox={`${box.x.toFixed(0)} ${box.y.toFixed(0)} ${box.width.toFixed(0)} ${box.height.toFixed(0)}`}
           preserveAspectRatio="xMidYMid slice" role="img" aria-label="The route flown so far, on the whole map"
           style={{ aspectRatio: `${Math.min(1.5, Math.max(0.8, box.width / box.height))}` }}>
        <image href={mapUrl} width={w} height={h} preserveAspectRatio="none" />
        <rect width={w} height={h} className="minimap-dim" />
        {runs.length > 0 && <polyline className="minimap-case" points={pts(runs.flatMap((r) => r.points))} />}
        {runs.map((r, i) => (
          <polyline key={i} className={r.coasting ? "minimap-seg coast" : "minimap-seg"} points={pts(r.points)} stroke={r.color} />
        ))}
        {/* what the camera sees now, then the start and the drone */}
        {fix?.accepted && <polygon className="minimap-view" points={pts(fix.footprint)} />}
        {start && <path className="minimap-start" d={dot(start.x, start.y)} />}
        {rec.located && <path className="minimap-halo" d={dot(rec.x, rec.y)} />}
        {rec.located && <path className="minimap-drone" d={dot(rec.x, rec.y)} />}
      </svg>
      <h2>Route so far</h2>
      <p>{Math.round((frame / Math.max(run.records.length - 1, 1)) * 100)} %</p>
    </section>
  );
}
