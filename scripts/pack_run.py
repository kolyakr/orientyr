"""Pack a run folder into one small, self-contained folder for the hosted console.

    python scripts/pack_run.py --run outputs/irpin --out ui/public/runs/irpin

A run folder is made for a laptop: run.json points at a map somewhere in data/, and the frames are PNG. A
static host has neither data/ nor room for it. The packed copy holds everything the console reads: run.json,
run.jsonl, the map as one WebP and the frames as small JPEGs. The telemetry is copied as it is; only each
record's `source` gets the new extension. The console shows the runs packed into ui/public/runs without any
setting (see ui/README.md, "Hosting").
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2

from dronenav.config import add_shared_args, init_cli
from dronenav.telemetry import RunInfo, read_run, read_run_info, write_run_info

MAP_NAME = "map.webp"


def _inside(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def _read(path: Path, what: str):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read the {what} {path}")
    return img


def _write(path: Path, img, params: list[int]) -> None:
    if not cv2.imwrite(str(path), img, params):
        raise OSError(f"cannot write {path}")


def pack_run(run_dir: Path, out_dir: Path, *, frame_width: int = 640, frame_quality: int = 80,
             map_quality: int = 80) -> tuple[int, int]:
    """Write the packed copy of ``run_dir`` into ``out_dir``; returns (records, bytes written).

    Raises FileNotFoundError for a missing run, map or frame and ValueError for a malformed run or an output
    folder that must not be replaced. Nothing is left in ``out_dir`` when it fails: the copy is built next to
    it and moved into place at the end.
    """
    run_dir, out_dir = run_dir.resolve(), out_dir.resolve()
    if _inside(out_dir, run_dir) or _inside(run_dir, out_dir):
        raise ValueError(f"cannot pack the run {run_dir} into itself: choose another --out")
    # the old copy is deleted, so --out must be one: a folder with other things in it is never removed
    if out_dir.exists() and any(out_dir.iterdir()) and not (out_dir / "run.json").is_file():
        raise ValueError(f"{out_dir} is not a packed run (no run.json in it): refusing to replace it")

    info = read_run_info(run_dir)                     # FileNotFoundError / TelemetryError name the problem
    records = read_run(run_dir / "run.jsonl")         # validates every line before anything is written
    lines = [json.loads(s) for s in (run_dir / "run.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    map_img = _read(Path(info.map), "map")

    tmp = out_dir.with_name(out_dir.name + ".packing")
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        tmp.mkdir(parents=True)
        # the map keeps its pixel size: the console draws in map pixel coordinates at full resolution
        _write(tmp / MAP_NAME, map_img, [cv2.IMWRITE_WEBP_QUALITY, map_quality])
        for rec, line in zip(records, lines, strict=True):
            # checked by spelling, not by resolving: frames/ may be a link to where the frames really are
            src = Path(os.path.normpath(run_dir / rec.source))
            if not _inside(src, run_dir):
                raise ValueError(f"record {rec.idx}: the frame {rec.source!r} is outside the run folder")
            frame = _read(src, f"frame of record {rec.idx}:")
            h, w = frame.shape[:2]
            if w > frame_width:   # never enlarged: that would only add bytes
                frame = cv2.resize(frame, (frame_width, round(h * frame_width / w)), interpolation=cv2.INTER_AREA)
            packed_source = Path(rec.source).with_suffix(".jpg")
            (tmp / packed_source).parent.mkdir(parents=True, exist_ok=True)
            _write(tmp / packed_source, frame, [cv2.IMWRITE_JPEG_QUALITY, frame_quality])
            line["source"] = packed_source.as_posix()
        with (tmp / "run.jsonl").open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(json.dumps(line, allow_nan=False) + "\n")
        # frame_size stays the size the homographies were estimated at; the packed frames only look smaller
        write_run_info(tmp, RunInfo(map=str(tmp / MAP_NAME), map_size=info.map_size, generator=info.generator,
                                    frame_size=info.frame_size, fps=info.fps, seed=info.seed))
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(out_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return len(records), sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run folder to pack (run.json + run.jsonl + frames/)")
    ap.add_argument("--out", required=True, help="folder for the packed copy (an older packed run there is replaced)")
    ap.add_argument("--frame-width", type=int, default=640,
                    help="widest packed frame, px; wider frames are scaled down (default 640)")
    ap.add_argument("--frame-quality", type=int, default=80, help="JPEG quality of the frames, 1-100 (default 80)")
    ap.add_argument("--map-quality", type=int, default=80, help="WebP quality of the map, 1-100 (default 80)")
    add_shared_args(ap, device=False)   # only reads and writes files: logging yes, no device to pick
    a = ap.parse_args(argv)
    init_cli(a)
    if a.frame_width < 1 or not (1 <= a.frame_quality <= 100 and 1 <= a.map_quality <= 100):
        ap.error("--frame-width must be positive and the qualities within 1-100")
    try:
        n, size = pack_run(Path(a.run), Path(a.out), frame_width=a.frame_width, frame_quality=a.frame_quality,
                           map_quality=a.map_quality)
    except (OSError, ValueError) as e:   # missing run, map or frame; malformed run (TelemetryError); bad --out
        print(f"pack_run: error: {e}", file=sys.stderr)
        return 2
    print(f"pack_run: {n} frames and the map in {a.out} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
