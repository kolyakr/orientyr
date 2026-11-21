// The hosted console: runs packed into public/runs (scripts/pack_run.py) are the runs folder when no setting says
// otherwise, and their map and frames are plain static files, so a static host serves them with no server code
// and no size limit. A run anywhere else keeps going through /api/map and /api/frame, as on a laptop.
import { cpSync, mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Console } from "@/components/Console";
import { assetUrls, listRuns, loadRun, openRun, runsDir } from "@/lib/run-server";

const fixture = path.join(__dirname, "fixtures", "run");

/** An app folder (the cwd of `next start`) with a packed run in public/runs/<name>: the fixture run plus a map inside it. */
function appWithPackedRun(name = "irpin"): { app: string; run: string } {
  const app = mkdtempSync(path.join(os.tmpdir(), "orientyr-app-"));
  const run = path.join(app, "public", "runs", name);
  mkdirSync(run, { recursive: true });
  cpSync(path.join(fixture, "run.jsonl"), path.join(run, "run.jsonl"));
  const info = { ...loadRun(fixture).info, map: "map.webp" };
  writeFileSync(path.join(run, "run.json"), JSON.stringify(info));
  return { app, run };
}

describe("runsDir with packed runs", () => {
  it("is public/runs of the app when a run is packed there", () => {
    const { app } = appWithPackedRun();
    expect(runsDir({}, app)).toBe(path.join(app, "public", "runs"));
    expect(listRuns(runsDir({}, app))).toEqual(["irpin"]);
  });

  it("is outputs/ of the repo when public/runs is empty or not there", () => {
    const app = mkdtempSync(path.join(os.tmpdir(), "orientyr-app-"));
    expect(runsDir({}, app)).toBe(path.resolve(app, "..", "outputs"));
    mkdirSync(path.join(app, "public", "runs", "not-a-run"), { recursive: true });
    expect(runsDir({}, app)).toBe(path.resolve(app, "..", "outputs"));
  });

  it("is ORIENTYR_RUNS or ORIENTYR_RUN first, packed runs or not", () => {
    const { app } = appWithPackedRun();
    expect(runsDir({ ORIENTYR_RUNS: "/some/runs" }, app)).toBe("/some/runs");
    expect(runsDir({ ORIENTYR_RUN: fixture }, app)).toBe(path.dirname(fixture));
  });
});

describe("assetUrls", () => {
  it("points a packed run at its static files", () => {
    const { app, run } = appWithPackedRun();
    expect(assetUrls(run, loadRun(run).info, app)).toEqual({ map: "/runs/irpin/map.webp", frames: "/runs/irpin" });
  });

  it("escapes a run name for the URL", () => {
    const { app, run } = appWithPackedRun("irpin 2025");
    expect(assetUrls(run, loadRun(run).info, app)).toEqual({ map: "/runs/irpin%202025/map.webp", frames: "/runs/irpin%202025" });
  });

  it("keeps a run outside public/ on the API routes", () => {
    const { app } = appWithPackedRun();
    expect(assetUrls(fixture, loadRun(fixture).info, app)).toBeNull();
  });

  it("keeps the API for a run in public/ whose map is outside it: a static URL could not reach that map", () => {
    const { app, run } = appWithPackedRun();
    const info = { ...loadRun(run).info, map: "../../../../data/map.png" };
    expect(assetUrls(run, info, app)).toBeNull();
  });
});

describe("openRun with packed runs", () => {
  it("lists a packed run, opens it without a name and gives the page its static URLs", () => {
    const { app, run } = appWithPackedRun();
    const opened = openRun(null, {}, app);
    expect(opened.error).toBeNull();
    expect(opened.runs).toEqual(["irpin"]);
    expect(opened.name).toBe("irpin");
    expect(opened.dir).toBe(run);
    expect(opened.assets).toEqual({ map: "/runs/irpin/map.webp", frames: "/runs/irpin" });
    // the map is the host's static file: the server does not look for it on its own disk
    expect(opened.mapMissing).toBe(false);
    expect(openRun("irpin", {}, app).assets).toEqual(opened.assets);
  });

  it("keeps the API routes for a run a setting points at, packed runs or not", () => {
    const { app } = appWithPackedRun();
    const opened = openRun(null, { ORIENTYR_RUN: fixture }, app);
    expect(opened.error).toBeNull();
    expect(opened.dir).toBe(fixture);
    expect(opened.assets).toBeNull();
    expect(opened.runs).not.toContain("irpin");
  });

  it("without a packed run it opens the runs of outputs/ as before: no static URLs", () => {
    const repo = mkdtempSync(path.join(os.tmpdir(), "orientyr-repo-"));
    const app = path.join(repo, "ui");
    mkdirSync(app);
    cpSync(fixture, path.join(repo, "outputs", "fake"), { recursive: true });
    const opened = openRun(null, {}, app);
    expect(opened.error).toBeNull();
    expect(opened.runs).toEqual(["fake"]);
    expect(opened.assets).toBeNull();
  });
});

describe("Console with static assets", () => {
  const run = loadRun(fixture);
  const rec = run.records[3];

  it("draws the map and the camera frame from the given URLs, also with a run name", () => {
    const assets = { map: "/runs/irpin/map.webp", frames: "/runs/irpin" };
    const markup = renderToStaticMarkup(<Console run={run} initialFrame={3} runName="irpin" runs={["irpin"]} assets={assets} />);
    expect(markup).toContain('href="/runs/irpin/map.webp"');
    expect(markup).toContain(`src="/runs/irpin/${rec.source}"`);
    expect(markup).not.toContain("/api/");
  });

  it("uses the API routes when no assets are given", () => {
    const markup = renderToStaticMarkup(<Console run={run} initialFrame={3} runName="irpin" runs={["irpin"]} />);
    expect(markup).toContain('href="/api/map?run=irpin"');
    expect(markup).toContain(`/api/frame?run=irpin&amp;src=${encodeURIComponent(rec.source)}`);
  });
});
