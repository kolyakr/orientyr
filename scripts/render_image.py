"""Draw a run's trajectory over its map: the full-size image and a 1600 px preview (M1-07).

    python scripts/render_image.py --run outputs/fake

Writes trajectory.png and trajectory_preview.jpg into the run folder (or --out). The run folder's run.json
names the map; see dronenav.viz.trajectory_image for what is drawn.
"""
from __future__ import annotations

import argparse
import sys

from dronenav.viz.trajectory_image import write_trajectory


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run folder (run.json + run.jsonl)")
    ap.add_argument("--out", default=None, help="folder for the images (default: the run folder)")
    ap.add_argument("--preview", type=int, default=1600, help="longest side of the preview, px (default 1600)")
    ap.add_argument("--label-every", type=int, default=None,
                    help="number every k-th point (default: as dense as stays readable)")
    a = ap.parse_args(argv)
    if a.preview < 1 or (a.label_every is not None and a.label_every < 1):
        ap.error("--preview and --label-every must be positive")
    try:
        full, preview = write_trajectory(a.run, a.out, preview_px=a.preview, label_every=a.label_every)
    except (OSError, ValueError) as e:   # missing run or map, unwritable output, malformed run (TelemetryError)
        print(f"render_image: error: {e}", file=sys.stderr)
        return 2
    print(f"render_image: {full} and {preview}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
