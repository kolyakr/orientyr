"""The true pose of a simulated frame, shared by both sources (FlightSource, CutoutSource), so consumers (the
chaos layer, the benchmark, the telemetry) read one type."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["FlightTruth", "centre_scale"]


def centre_scale(H: np.ndarray, x: float, y: float) -> float:
    """Map px per frame px at frame point (x, y): the square root of the homography's local area scale. For a
    tilted view this is not the top-left block of H (that one depends on where the frame sits on the map)."""
    q = H @ [x, y, 1.0]
    m = q[:2] / q[2]
    J = (H[:2, :2] - np.outer(m, H[2, :2])) / q[2]          # d(map px) / d(frame px)
    return math.sqrt(abs(np.linalg.det(J)))


@dataclass(frozen=True, eq=False)
class FlightTruth:
    """Frame ``i``'s true pose. H maps (undistorted) frame px to map px; center is H at the principal point;
    footprint is TL TR BR BL; altitude, pitch, roll and speed are None when the source has no camera model."""

    i: int
    t: float
    H: np.ndarray
    center: tuple[float, float]
    footprint: np.ndarray
    heading_deg: float
    altitude_m: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    speed_mps: float | None = None

    def __post_init__(self):
        H = np.asarray(self.H, float)
        fp = np.asarray(self.footprint, float)
        if H.shape != (3, 3) or fp.shape != (4, 2) or len(self.center) != 2:
            raise ValueError(f"frame {self.i}: H must be 3x3, footprint 4x2, center 2 numbers")
        values = {"t": self.t, "heading_deg": self.heading_deg, "center": self.center, "H_frame_to_map": H,
                  "footprint_map_px": fp, "altitude_m": self.altitude_m, "pitch_deg": self.pitch_deg,
                  "roll_deg": self.roll_deg, "speed_mps": self.speed_mps}
        for name, v in values.items():
            if v is not None and not np.isfinite(np.asarray(v, float)).all():
                raise ValueError(f"frame {self.i}: {name} must be finite, got {v!r}")
        object.__setattr__(self, "H", H)
        object.__setattr__(self, "footprint", fp)
        object.__setattr__(self, "center", (float(self.center[0]), float(self.center[1])))

    def extra(self) -> dict:
        """JSON-safe values for ``Record.extra["truth"]``: x, y, heading_deg, and what the source knows besides."""
        out = {"x": self.center[0], "y": self.center[1], "heading_deg": float(self.heading_deg)}
        for k in ("altitude_m", "pitch_deg", "roll_deg", "speed_mps"):
            if getattr(self, k) is not None:
                out[k] = float(getattr(self, k))
        return out
