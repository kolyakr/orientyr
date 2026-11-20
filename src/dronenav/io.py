"""Reading the inputs (M1-01): frames from a folder, and a map with its sidecar.

- :class:`FolderSource` yields the images of a folder as :class:`~dronenav.types.Frame` s, in natural order
  (``crop_2`` before ``crop_10``). A frame's ``idx`` is the number in its file name, so the task's ``crop_N`` is
  frame N and an Irpin keyframe ``frame_00030`` is video frame 30; ``t = idx / fps``, with fps from the argument,
  else from the flight's ``meta.json``, else 1.
- :func:`load_map` reads a map image and finds its sidecar: the JSON that says where the map is on Earth. It hands
  over the raw sidecar and the block that describes this image (``map`` or ``hires``); turning that into pixel
  <-> lon/lat <-> metre conversions is the georeference module's job (M0-05), so the format is parsed once.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dronenav.types import Frame

__all__ = ["IMAGE_EXTENSIONS", "FolderSource", "MapImage", "load_map"]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_DIGITS = re.compile(r"(\d+)")


# --- frames ----------------------------------------------------------------------------------------------------

def _natural_key(name: str) -> list:
    """'crop_10.png' -> ['crop_', 10, '.png']: digit runs compare as numbers, the rest case-insensitively."""
    return [int(part) if part.isdigit() else part.lower() for part in _DIGITS.split(name)]


class FolderSource:
    """The images in ``folder`` (not its subfolders), as Frames in natural order. Images are read one at a time,
    while iterating, so a long flight does not sit in memory. Iterating again starts again from the first frame.

    fps: frames per second for ``t = idx / fps``; None takes ``fps`` from ``meta.json`` in the folder or its parent
    (an Irpin flight keeps it next to ``frames/``), else 1.
    """

    def __init__(self, folder: str | Path, fps: float | None = None):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"no frames folder {self.folder}")
        self.paths = sorted((p for p in self.folder.iterdir()
                             if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS),
                            key=lambda p: _natural_key(p.name))
        if not self.paths:
            raise ValueError(f"no images in {self.folder} (looked for {', '.join(sorted(IMAGE_EXTENSIONS))})")
        if fps is not None and not (isinstance(fps, (int, float)) and math.isfinite(fps) and fps > 0):
            raise ValueError(f"fps must be a positive number, got {fps!r}")
        self.fps = float(fps) if fps is not None else _meta_fps(self.folder)
        self.idxs = _frame_numbers(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[Frame]:
        for idx, path in zip(self.idxs, self.paths, strict=True):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)     # grey and alpha come back as BGR
            if image is None:
                raise ValueError(f"cannot read the frame {path}")
            yield Frame(idx=idx, t=idx / self.fps, image=image, source=str(path))


def _frame_numbers(paths: list[Path]) -> list[int]:
    """The last number in each name, if every name has one and they are all different; else 0, 1, 2, ..."""
    numbers = []
    for p in paths:
        found = _DIGITS.findall(p.stem)
        if not found:
            return list(range(len(paths)))
        numbers.append(int(found[-1]))
    return numbers if len(set(numbers)) == len(numbers) else list(range(len(paths)))


def _meta_fps(folder: Path) -> float:
    """fps from meta.json in the folder, else in its parent; 1 when neither has one. A meta.json without an "fps"
    key is about something else and is skipped; one whose fps is broken is an error, not a silent 1 fps."""
    for meta in (folder / "meta.json", folder.parent / "meta.json"):
        if not meta.is_file():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raise ValueError(f"{meta} is not valid JSON") from None
        if not isinstance(data, dict):
            raise ValueError(f"{meta} must be a JSON object")   # noqa: TRY004 - bad file, not a bad argument
        if "fps" not in data:
            continue
        fps = data["fps"]
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"{meta} has no usable fps (a positive number), got {fps!r}")
        return float(fps)
    return 1.0


# --- maps ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class MapImage:
    """A map image and what its sidecar says about it.

    image: BGR uint8. sidecar: the whole sidecar JSON, or None (the task's ``global_map.png`` has none).
    level: the sidecar block whose ``size_px`` is this image's size (``map`` or ``hires``), or None.
    """

    image: np.ndarray
    path: Path
    sidecar: dict | None
    level: dict | None


def load_map(path: str | Path) -> MapImage:
    """Read a map and find its sidecar: ``<stem>.json`` next to it, or for a layer in another folder (Irpin's
    ``hires/``) a JSON in the dataset's ``maps/`` whose block ``file`` names it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no map {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read the map {path}")
    h, w = image.shape[:2]
    found = _find_sidecar(path)
    if found is None:
        return MapImage(image=image, path=path, sidecar=None, level=None)
    sidecar_path, sidecar = found
    blocks = [b for b in sidecar.values() if isinstance(b, dict) and "size_px" in b]
    level = next((b for b in blocks if list(b["size_px"]) == [w, h]), None)
    if level is None:
        sizes = ", ".join(f"{b['size_px'][0]} x {b['size_px'][1]}" for b in blocks) or "none"
        raise ValueError(f"the map {path.name} is {w} x {h} px but its sidecar {sidecar_path} describes {sizes}")
    return MapImage(image=image, path=path, sidecar=sidecar, level=level)


def _find_sidecar(path: Path) -> tuple[Path, dict] | None:
    beside = path.with_suffix(".json")
    if beside.is_file():
        return beside, _read_sidecar(beside)
    maps = path.parent.parent / "maps"
    wanted = f"{path.parent.name}/{path.name}"            # e.g. hires/irpin_2025_z18.png, as the sidecar writes it
    for candidate in sorted(maps.glob("*.json")) if maps.is_dir() else []:
        sidecar = _read_sidecar(candidate)
        if any(isinstance(b, dict) and b.get("file") == wanted for b in sidecar.values()):
            return candidate, sidecar
    return None


def _read_sidecar(path: Path) -> dict:
    try:
        sidecar = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"the sidecar {path} is not valid JSON: {e.msg} at line {e.lineno}") from None
    if not isinstance(sidecar, dict):
        raise ValueError(f"the sidecar {path} must be a JSON object")   # noqa: TRY004 - bad file, not a bad argument
    return sidecar
