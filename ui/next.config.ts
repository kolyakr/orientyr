import type { NextConfig } from "next";

const config: NextConfig = {
  // The repo root holds the Python project; without this Next guesses the workspace root from lockfiles.
  turbopack: { root: __dirname },
  devIndicators: false, // the console is filmed for the demo video: no dev badge in the corner
};

export default config;
