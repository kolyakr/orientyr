"""Reading the inputs (M1-01): frames from a folder, and a map with its sidecar.

- :class:`FolderSource` yields the images of a folder as :class:`~dronenav.types.Frame` s. A frame's ``idx`` is
  the last number in its file name, so the task's ``crop_N`` is frame N and an Irpin keyframe ``frame_00030`` is
  video frame 30, and frames come in that order (``crop_2`` before ``crop_10``); ``t = idx / fps``, with fps from
  the argument, else from the flight's ``meta.json``, else 1. Names without numbers, or whose numbers repeat, are
  numbered 0, 1, 2 ... in natural name order; a mix of numbered and unnumbered names is an error, because one
  stray file must not renumber a whole flight. When that ``meta.json`` has a ``camera`` block (K, dist, size; the
  Irpin flights), every frame is undistorted with it as it is read (M2-04): a homography only holds between
  pinhole images, and the flights' truth is for undistorted pixels.
- :func:`load_map` reads a map image and finds its sidecar: the JSON that says where the map is on Earth. It hands
  over the raw sidecar and the block that describes this image (``map`` or ``hires``); turning that into pixel
  <-> lon/lat <-> metre conversions is the georeference module's job (M0-05), so the format is parsed once.
"""
from __future__ import annotations

import json
import math
import numbers
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from dronenav.camera import Camera
from dronenav.types import Frame

__all__ = ["IMAGE_EXTENSIONS", "FolderSource", "MapImage", "load_map"]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_DIGITS = re.compile(r"(\d+)")


# --- frames ----------------------------------------------------------------------------------------------------

def _natural_key(name: str) -> tuple:
    """'crop_10.png' -> ['crop_', 10, '.png']: digit runs compare as numbers, the rest case-insensitively; the
    name itself breaks ties (f_1, f_01), so the order never depends on the filesystem."""
    parts = [(1, int(part)) if _DIGITS.fullmatch(part) else (0, part.lower()) for part in _DIGITS.split(name)]
    return parts, name


class FolderSource:
    """The images in ``folder`` (not its subfolders), as Frames in frame-number order (see the module docstring).
    Images are read one at a time, while iterating, so a long flight does not sit in memory. Iterating again starts
    again from the first frame.

    fps: frames per second for ``t = idx / fps``; None takes ``fps`` from ``meta.json`` in the folder or its parent
    (an Irpin flight keeps it next to ``frames/``), else 1.
    undistort: undistort each frame with the ``camera`` block of that same ``meta.json`` (``self.camera``, None
    when there is none or ``undistort`` is False). Off only to look at the raw frames: the checks and the tilt
    assume pinhole frames, and raw ones cost about 1° of tilt and 0.5 px of position on the Irpin flights.
    """

    def __init__(self, folder: str | Path, fps: float | None = None, undistort: bool = True):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"no frames folder {self.folder}")
        self.paths = sorted((p for p in self.folder.iterdir()
                             if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTENSIONS),
                            key=lambda p: _natural_key(p.name))
        if not self.paths:
            raise ValueError(f"no images in {self.folder} (looked for {', '.join(sorted(IMAGE_EXTENSIONS))})")
        if fps is not None and not _is_rate(fps):
            raise ValueError(f"fps must be a positive, finite number, got {fps!r}")
        self.fps = float(fps) if fps is not None else _meta_fps(self.folder)
        self.camera = _meta_camera(self.folder) if undistort else None
        self.paths, self.idxs = _frame_numbers(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[Frame]:
        for idx, path in zip(self.idxs, self.paths, strict=True):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)     # grey and alpha come back as BGR
            if image is None:
                raise ValueError(f"cannot read the frame {path}")
            if self.camera is not None:
                try:
                    image = self.camera.undistort_image(image)
                except ValueError as e:
                    raise ValueError(f"{path}: {e}") from None
            yield Frame(idx=idx, t=idx / self.fps, image=image, source=str(path))


def _frame_numbers(paths: list[Path]) -> tuple[list[Path], list[int]]:
    """The frames in order with their numbers: by the last number in each name when every name has one and they
    are all different; 0, 1, 2 ... in natural order when no name has one or they repeat."""
    found = [_DIGITS.findall(p.stem) for p in paths]
    missing = [p.name for p, f in zip(paths, found, strict=True) if not f]
    if not missing:
        nums = [int(f[-1]) for f in found]
        if len(set(nums)) == len(nums):                  # frame numbers: the order of time
            pairs = sorted(zip(nums, paths, strict=True), key=lambda np_: (np_[0], np_[1].name))
            return [p for _, p in pairs], [n for n, _ in pairs]
        return paths, list(range(len(paths)))             # repeated numbers (take1_a, take1_b): not frame numbers
    if len(missing) == len(paths):
        return paths, list(range(len(paths)))
    shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
    raise ValueError(f"{len(missing)} of {len(paths)} images in {paths[0].parent} have no frame number in their "
                     f"name ({shown}): move them out, or name every frame with its number")


def _is_rate(v) -> bool:
    """A positive, finite number; not a bool (True would mean 1 fps), numpy numbers welcome."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        return False
    try:
        return math.isfinite(v) and v > 0
    except OverflowError:                                  # an int too large for a float
        return False


def _metas(folder: Path) -> Iterator[tuple[Path, dict]]:
    """The meta.json in the folder, then the one in its parent, that exist, as (path, parsed object)."""
    for meta in (folder / "meta.json", folder.parent / "meta.json"):
        if not meta.is_file():
            continue
        try:
            data = json.loads(meta.read_bytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"{meta} is not valid UTF-8 JSON: {e}") from None
        if not isinstance(data, dict):
            raise ValueError(f"{meta} must be a JSON object")   # noqa: TRY004 - bad file, not a bad argument
        yield meta, data


def _meta_camera(folder: Path) -> Camera | None:
    """The camera from the first meta.json (folder, then parent) that has a "camera" block; None when none has
    one. A broken block is an error naming the file, not silently raw frames."""
    for meta, data in _metas(folder):
        if "camera" not in data:
            continue
        try:
            return Camera.from_meta(data)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{meta} has a broken camera block (needs K, dist and size): {e!r}") from None
    return None


def _meta_fps(folder: Path) -> float:
    """fps from meta.json in the folder, else in its parent; 1 when neither has one. A meta.json without an "fps"
    key is about something else and is skipped; one whose fps is broken is an error, not a silent 1 fps."""
    for meta, data in _metas(folder):
        if "fps" not in data:
            continue
        fps = data["fps"]
        if not _is_rate(fps):
            shown = repr(fps) if len(repr(fps)) < 40 else repr(fps)[:37] + "..."
            raise ValueError(f"{meta} has no usable fps (a positive, finite number), got {shown}")
        return float(fps)
    return 1.0


# --- maps ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class MapImage:
    """A map image and what its sidecar says about it.

    image: BGR uint8. sidecar: the whole sidecar JSON, or None (the task's ``global_map.png`` has none).
    level: the sidecar block for this image: the one whose ``file`` names it, else the one whose ``size_px`` is
    its size (``map`` or ``hires``); None without a sidecar. Its size always matches the image.
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
    # as stored: map pixels are coordinates, an EXIF rotation would turn the map away from its sidecar
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"cannot read the map {path}")
    h, w = image.shape[:2]
    found = _find_sidecar(path)
    if found is None:
        return MapImage(image=image, path=path, sidecar=None, level=None)
    sidecar_path, sidecar = found
    blocks = {k: b for k, b in sidecar.items() if isinstance(b, dict) and "size_px" in b}
    for k, b in blocks.items():
        size = b["size_px"]
        if not (isinstance(size, list) and len(size) == 2
                and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in size)):
            raise ValueError(f"the sidecar {sidecar_path}: {k}.size_px must be [w, h] in px, got {size!r}")
    named = [b for b in blocks.values() if b.get("file") == _rel_name(path)]   # the block that names this file
    candidates = named or [b for b in blocks.values() if b["size_px"] == [w, h]]
    level = candidates[0] if candidates else None
    if level is None or level["size_px"] != [w, h]:
        sizes = ", ".join(f"{b['size_px'][0]} x {b['size_px'][1]}" for b in (named or blocks.values())) or "none"
        raise ValueError(f"the map {path.name} is {w} x {h} px but its sidecar {sidecar_path} describes {sizes}")
    return MapImage(image=image, path=path, sidecar=sidecar, level=level)


def _rel_name(path: Path) -> str:
    """How a sidecar names a layer: its folder and file, e.g. hires/irpin_2025_z18.png. Absolute but not resolved:
    a symlinked hires/ keeps its name."""
    path = path.absolute()
    return f"{path.parent.name}/{path.name}"


def _find_sidecar(path: Path) -> tuple[Path, dict] | None:
    """The map's own ``<stem>.json`` (a broken one is an error), else the one sidecar in the dataset's ``maps/``
    that names this layer (other JSON files there are skipped; two claiming it is an error)."""
    beside = path.with_suffix(".json")
    if beside.is_file():
        return beside, _read_sidecar(beside)
    maps = path.absolute().parent.parent / "maps"
    wanted = _rel_name(path)
    claims = []
    for candidate in sorted(maps.glob("*.json")) if maps.is_dir() else []:
        try:
            sidecar = _read_sidecar(candidate)
        except ValueError:                                 # not a sidecar (notes, lists, broken files): not ours
            continue
        if any(isinstance(b, dict) and b.get("file") == wanted for b in sidecar.values()):
            claims.append((candidate, sidecar))
    if len(claims) > 1:
        raise ValueError(f"{wanted} is claimed by more than one sidecar: " + ", ".join(str(c) for c, _ in claims))
    return claims[0] if claims else None


def _read_sidecar(path: Path) -> dict:
    try:
        sidecar = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"the sidecar {path} is not valid UTF-8 JSON: {e}") from None
    if not isinstance(sidecar, dict):
        raise ValueError(f"the sidecar {path} must be a JSON object")   # noqa: TRY004 - bad file, not a bad argument
    return sidecar
