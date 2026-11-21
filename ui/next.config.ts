import type { NextConfig } from "next";

const config: NextConfig = {
  // The repo root holds the Python project; without this Next guesses the workspace root from lockfiles.
  turbopack: { root: __dirname },
  // A hosted console reads the packed runs (public/runs, see README "Hosting") from the disk on every request:
  // without this the server bundle would not carry their telemetry files. The images stay static files.
  outputFileTracingIncludes: { "/*": ["./public/runs/**/run.json", "./public/runs/**/run.jsonl"] },
  devIndicators: false, // the console is filmed for the demo video: no dev badge in the corner
};

export default config;
