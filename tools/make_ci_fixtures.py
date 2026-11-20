#!/usr/bin/env python3
"""Cut the tiny Irpin fixture that CI tests run on (M0-08), from a local copy of the team data.

    python tools/make_ci_fixtures.py --data data --out tests/fixtures/irpin_mild_2025

What it writes (a few hundred KB, committed to git):
    map_crop.jpg            512 × 512 crop of maps/irpin_2025.png, centred on the keyframes
    frame_XXXXXX.jpg        three keyframes of flights/mild/2025/flight.mp4 at the camera's full size
    truth.jsonl             their records from the flight's truth.jsonl, verbatim (full-map coordinates)
    fixture.json            where it all came from, the crop origin and the camera

The truth stays in full-map pixels on purpose: tests convert with ``crop = map - origin``, so the same records
work for the fixture and for the full dataset. Note the flight's truth homographies act on *undistorted* frames
(the renderer adds lens distortion ``camera.dist`` after projecting). Deterministic: same data in, same bytes out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Five seconds apart in the 20 s turn of the mild flight (heading ~220° → ~290°): the three frames differ in
# position and rotation, yet all their centres fit in one 512 px crop of the 0.76 m/px map.
KEYFRAMES = (6150, 6300, 6450)
CROP = 512
# The frames carry the renderer's sensor noise, which JPEG pays dearly for: q70 keeps three full-size frames near
# 150 KB each (the bundle stays "a few hundred KB") and still registers at NCC ≈ 0.98; the map crop is clean, so q85.
FRAME_QUALITY, MAP_QUALITY = 70, 85


def read_frames(video: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Decode sequentially up to the last wanted index: H.264 seeking is not frame-exact, the truth is."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    out, i = {}, 0
    while i <= max(wanted):
        if i in wanted:
            ok, img = cap.read()
            if not ok:
                raise SystemExit(f"{video} ends before frame {i}")
            out[i] = img
        elif not cap.grab():
            raise SystemExit(f"{video} ends before frame {i}")
        i += 1
    cap.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=Path("data"), help="local copy of the team Drive (has irpin/)")
    ap.add_argument("--out", type=Path, default=Path("tests/fixtures/irpin_mild_2025"))
    args = ap.parse_args()

    irpin = args.data / "irpin"
    flight = irpin / "flights" / "mild" / "2025"
    meta = json.loads((flight / "meta.json").read_text())
    map_path = irpin / meta["map"]
    truth = {r["i"]: r for r in map(json.loads, (flight / "truth.jsonl").read_text().splitlines())
             if r["i"] in KEYFRAMES}

    full = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
    if full is None:
        raise SystemExit(f"cannot read {map_path}")
    h, w = full.shape[:2]
    centre = np.mean([truth[i]["center_map_px"] for i in KEYFRAMES], axis=0)
    x0 = int(np.clip(round(centre[0] - CROP / 2), 0, w - CROP))
    y0 = int(np.clip(round(centre[1] - CROP / 2), 0, h - CROP))

    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out / "map_crop.jpg"), full[y0:y0 + CROP, x0:x0 + CROP],
                [cv2.IMWRITE_JPEG_QUALITY, MAP_QUALITY])
    frames = read_frames(flight / "flight.mp4", set(KEYFRAMES))
    for i in KEYFRAMES:
        cv2.imwrite(str(args.out / f"frame_{i:06d}.jpg"), frames[i], [cv2.IMWRITE_JPEG_QUALITY, FRAME_QUALITY])
    (args.out / "truth.jsonl").write_text("".join(json.dumps(truth[i]) + "\n" for i in KEYFRAMES))

    fixture = {
        "about": "CI fixture (M0-08): map px = crop px + crop.origin_px; truth is in full-map px. "
                 "H_frame_to_map maps UNDISTORTED frame px (cv2.undistort with camera.K, camera.dist): on the raw "
                 "frame it is off by the lens distortion. Regenerate with tools/make_ci_fixtures.py.",
        "source": {"map": f"irpin/{meta['map']}", "map_size_px": [w, h],
                   "flight": "irpin/flights/mild/2025", "video": "flight.mp4", "fps": meta["fps"]},
        "crop": {"file": "map_crop.jpg", "origin_px": [x0, y0], "size_px": [CROP, CROP]},
        "camera": {k: meta["camera"][k] for k in ("size", "hfov_deg", "K", "dist")},
        "frames": [{"i": i, "t": truth[i]["t"], "file": f"frame_{i:06d}.jpg"} for i in KEYFRAMES],
        "truth": "truth.jsonl",
    }
    (args.out / "fixture.json").write_text(json.dumps(fixture, indent=1) + "\n")
    size = sum(p.stat().st_size for p in args.out.iterdir())
    print(f"wrote {args.out}: crop at ({x0}, {y0}), frames {KEYFRAMES}, {size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
