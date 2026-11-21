"""The Irpin dataset as the benchmarks see it (M6-03): scenarios, flight folders, and each flight's truth on a map.

Layout (data/irpin, from the team Drive): ``maps/irpin_<year>.png`` with a georeference sidecar ``.json`` (M0-05);
``flights/<preset>[_30mps]/<year>/`` with ``frames/``, ``meta.json`` and ``truth.jsonl``. A flight's truth is in the
px of the map it was rendered from (``meta["map"]``). The years' maps share one grid up to a measured shift,
``alignment_to_2025.shift_px``: a point at p on year Y's map is at p + shift_Y on the 2025 map. So the truth on
another map M is ``c + shift_flight - shift_M`` (about 3 px, 2 m, at most: small, but not below what we measure).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dronenav.accuracy import load_truth
from dronenav.georef import load_georef

__all__ = ["PRESETS", "SCENARIOS", "SPEEDS", "Scenario", "flight_dir", "m_per_px", "map_path", "truth_on_map"]


@dataclass(frozen=True)
class Scenario:
    flight: str     # the year of the ground the drone flies over
    map: str        # the year of the map it carries
    shows: str


# The plan's scenarios on Irpin data (issue #55, 20 Nov); A (task data) and B (phone take) have no exact truth.
SCENARIOS = {
    "C": Scenario("2025", "2025", "post-war ground, post-war map"),
    "D": Scenario("2025", "2022", "post-war ground, war-time map"),
    "E": Scenario("2025", "2017", "post-war ground, pre-war map"),
    "F": Scenario("2022", "2017", "war-time ground, pre-war map"),
}
PRESETS = ("mild", "harsh")
SPEEDS = (15, 30)          # m/s; the 30 m/s flights live in "<preset>_30mps"


def flight_dir(data: Path, preset: str, speed: int, year: str) -> Path:
    if speed not in SPEEDS:
        raise ValueError(f"no {speed} m/s flights (known: {', '.join(map(str, SPEEDS))})")
    return Path(data) / "irpin" / "flights" / (preset if speed == 15 else f"{preset}_30mps") / str(year)


def map_path(data: Path, year: str) -> Path:
    return Path(data) / "irpin" / "maps" / f"irpin_{year}.png"


def _block(map_png: Path):
    return load_georef(Path(map_png).with_suffix(".json")).map


def m_per_px(map_png: Path) -> float:
    """The map's measured ground resolution, metres per px, from its sidecar."""
    return float(_block(map_png).m_per_px)


def _shift(map_png: Path) -> tuple[float, float]:
    dx, dy = _block(map_png).alignment_to_2025["shift_px"]
    return float(dx), float(dy)


def truth_on_map(flight: Path, map_png: Path) -> dict[int, tuple[float, float]]:
    """The flight's true frame centres (frame idx -> x, y) in the px of ``map_png``."""
    flight = Path(flight)
    own = flight.parents[2] / json.loads((flight / "meta.json").read_text())["map"]   # irpin/ + maps/...
    (fx, fy), (mx, my) = _shift(own), _shift(map_png)
    return {i: (x + fx - mx, y + fy - my) for i, (x, y) in load_truth(flight / "truth.jsonl").items()}
