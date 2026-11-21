"""Calibrate a camera from checkerboard photos (M9-03): corner detection + cv2.calibrateCamera.

    python scripts/calibrate_camera.py --images "photos/*.jpg" --cols 9 --rows 6 --out outputs/camera.json

The spec's "calibrate the phone camera once": several photos of a checkerboard shown full-screen,
at different angles. Writes {"camera": {"K": ..., "dist": ..., "size": [w, h]}} to --out -- the
same shape a flight's meta.json camera block has, so the file loads straight away with
dronenav.camera.Camera.from_meta (standalone, or dropped into a meta.json). --cols/--rows count
inner corners (an N x M board of squares has (N-1) x (M-1) inner corners).
Exit code 0: written. 2: no images matched --images, a photo couldn't be read, or calibration
failed (printed).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2

from dronenav.calibration import calibrate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--images", required=True, help="glob of checkerboard photos, e.g. 'photos/*.jpg'")
    ap.add_argument("--cols", type=int, required=True, help="inner checkerboard corners per row")
    ap.add_argument("--rows", type=int, required=True, help="inner checkerboard corners per column")
    ap.add_argument("--square", type=float, default=1.0,
                    help="checkerboard square's real size; does not change K or dist (default: 1.0)")
    ap.add_argument("--out", required=True, help="camera.json to write")
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(a.images))
    if not paths:
        print(f"calibrate_camera: error: no images matched {a.images!r}", file=sys.stderr)
        return 2
    images = [cv2.imread(p) for p in paths]
    bad = next((p for p, img in zip(paths, images) if img is None), None)
    if bad is not None:
        print(f"calibrate_camera: error: could not read {bad}", file=sys.stderr)
        return 2

    try:
        K, dist, size = calibrate(images, a.cols, a.rows, square=a.square)
    except (ValueError, cv2.error) as e:
        print(f"calibrate_camera: error: {e}", file=sys.stderr)
        return 2

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"camera": {"K": K.tolist(), "dist": dist.tolist(), "size": list(size)}}, indent=1))
    print(f"calibrate_camera: wrote {out} from {len(paths)} views")
    return 0


if __name__ == "__main__":
    sys.exit(main())
