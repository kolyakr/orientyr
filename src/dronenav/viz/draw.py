"""Drawing pieces shared by the renderers (trajectory image, trajectory video): one style, one set of helpers.

Coordinates follow dronenav.types: pixels with the origin at the top-left corner of the top-left pixel. OpenCV
puts pixel centres at integers, so :func:`px` shifts by half a pixel and draws with sub-pixel precision.
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["FINISH", "FIXED", "FONT", "FOOTPRINT", "MISSED", "OUTLINE", "SHIFT", "START", "STATE_COLOURS", "TEXT",
           "dashed", "is_missed", "px"]

# Colours, BGR. Chosen to stand out on satellite imagery (greens, greys, browns) and from each other.
FIXED = (255, 170, 0)        # azure: route and points with an accepted fix
MISSED = (0, 140, 255)       # orange: no accepted fix, or interpolated
FOOTPRINT = (235, 235, 235)  # near white, translucent
START = (60, 210, 60)        # green
FINISH = (60, 40, 230)       # red
TEXT = (255, 255, 255)
OUTLINE = (0, 0, 0)
# Tracker states (TrackMode values): tracking looks like the fixed route, lost like the missed one.
STATE_COLOURS = {"TRACKING": FIXED, "LOST": MISSED, "RELOCALIZING": (0, 215, 255)}

FONT = cv2.FONT_HERSHEY_SIMPLEX
SHIFT = 4                    # sub-pixel drawing: OpenCV coordinates in 1/16 px


def px(x: float, y: float) -> tuple[int, int]:
    """Image px (origin at the top-left corner of the top-left pixel) to OpenCV's sub-pixel integer coordinates
    (pixel centres at integers, SHIFT fractional bits)."""
    return round((x - 0.5) * (1 << SHIFT)), round((y - 0.5) * (1 << SHIFT))


def dashed(img: np.ndarray, a, b, colour, thick: int, dash: float, gap: float, phase: float = 0.0) -> float:
    """Dashes from a to b, starting ``phase`` px into the dash+gap pattern; returns the phase at b, so a dashed
    polyline keeps one pattern across its segments (short segments would otherwise look solid)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = float(np.hypot(*(b - a)))
    period = dash + gap
    if length == 0:
        return phase
    d = (b - a) / length
    s = -phase                                   # where the current pattern period started, relative to a
    while s < length:
        ds, de = max(s, 0.0), min(s + dash, length)
        if de > ds:
            cv2.line(img, px(*(a + d * ds)), px(*(a + d * de)), colour, thick, cv2.LINE_AA, SHIFT)
        s += period
    return (length + phase) % period


def is_missed(record) -> bool:
    """No accepted fix on this frame, or its position was filled in rather than measured."""
    fix = record.track.fix
    return record.interpolated or fix is None or not fix.accepted
