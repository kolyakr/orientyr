"""MP4 "fast start": move the index (``moov``) in front of the media data (``mdat``).

OpenCV writes the index at the end, which plays from a local file but not while a browser is still downloading
it (the web viewer, a link in the README). Moving it to the front shifts the media data by the index's size, so
every chunk offset in the index (``stco`` / ``co64`` tables) moves by the same amount.
"""
from __future__ import annotations

import struct
from pathlib import Path

__all__ = ["faststart"]

_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"mvex", b"udta"}


def _boxes(data: bytes | bytearray, start: int, end: int):
    """(type, start, header size, total size) of each box in data[start:end]."""
    i = start
    while i + 8 <= end:
        size, kind = struct.unpack(">I4s", data[i:i + 8])
        header = 8
        if size == 1:                                   # 64-bit size follows the type
            if i + 16 > end:
                raise ValueError(f"truncated MP4 box {kind!r} at byte {i}")
            size = struct.unpack(">Q", data[i + 8:i + 16])[0]
            header = 16
        elif size == 0:                                 # to the end of the file
            size = end - i
        if size < header or i + size > end:
            raise ValueError(f"malformed MP4 box {kind!r} at byte {i}")
        yield kind, i, header, size
        i += size


def _shift_offsets(moov: bytearray, start: int, end: int, delta: int, below: int) -> None:
    """Add ``delta`` to every chunk offset that points before byte ``below`` (media that moves); media after the
    old index stays where it was."""
    for kind, i, header, size in _boxes(moov, start, end):
        if kind in _CONTAINERS:
            _shift_offsets(moov, i + header, i + size, delta, below)
        elif kind in (b"stco", b"co64"):
            n = struct.unpack(">I", moov[i + header + 4:i + header + 8])[0]
            width, fmt = (4, ">I") if kind == b"stco" else (8, ">Q")
            for k in range(n):
                p = i + header + 8 + k * width
                value = struct.unpack(fmt, moov[p:p + width])[0]
                value += delta if value < below else 0
                if kind == b"stco" and value >= 2 ** 32:
                    raise ValueError("chunk offsets past 4 GB: cannot move the index of this file")
                moov[p:p + width] = struct.pack(fmt, value)


def faststart(path: str | Path) -> bool:
    """Rewrite ``path`` in place with the index first; returns False if it already was. Raises ValueError for a
    file that is not a well-formed MP4."""
    path = Path(path)
    data = path.read_bytes()
    top = list(_boxes(data, 0, len(data)))
    kinds = [k for k, *_ in top]
    if b"moov" not in kinds or b"mdat" not in kinds:
        raise ValueError(f"{path} has no moov/mdat boxes: not an MP4 written by a video encoder")
    if kinds.index(b"moov") < kinds.index(b"mdat"):
        return False
    _, mi, mheader, msize = top[kinds.index(b"moov")]
    moov = bytearray(data[mi:mi + msize])
    _shift_offsets(moov, mheader, len(moov), msize, mi)  # media before the old index moves by its size
    first_media = top[kinds.index(b"mdat")][1]
    out = data[:first_media] + bytes(moov) + b"".join(
        data[i:i + size] for kind, i, _, size in top if i >= first_media and kind != b"moov")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(out)
    tmp.replace(path)                                    # atomic: a failed rewrite leaves the original
    return True
