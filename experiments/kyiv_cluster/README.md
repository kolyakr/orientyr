# Kyiv-oblast cluster: multi-epoch maps (Irpin, Bucha, Hostomel, Moschun, Vorzel)

Three Esri World Imagery Wayback maps of one 15.6 km square (centre 50.545, 30.235; z16, 1.52 m/px, 10240² px),
stored in `data/kyiv_cluster/maps/` (gitignored):

| map | release | captures (9x9 metadata grid) |
|---|---|---|
| `cluster_2020_prewar` | 15423 | patchwork 2017-05-05 / 2019-04-03 / 2020-03-03, visible seams + one grey patch (SE) |
| `cluster_2022_war` | 45134 | 2022-03-18/21/31 (+ 02-28, 05-09 at the edges); smoke plumes; matches UNOSAT 31 Mar 2022 |
| `cluster_2025_rebuilt` | 20512 | 2025-04-11, but ~6 % still falls back to 2022-03-21; cloud shadows |

The epochs agree geometrically to <0.5 px (phase correlation), so ground truth carries across maps.

## Drift matrix (`drift_matrix.py`)

Synthetic 640x480 frames are cut from the flight epoch, with random rotation, 0.8–1.25 scale and blur. Each frame is registered against the map epoch with SIFT + MAGSAC inside a 1400 px window around a σ = 150 px prior. A frame counts as a success if the error is under 15 px (≈ 23 m) and it has at least 12 inliers. The test uses 40 frames centred on UNOSAT destroyed/severe labels and 40 at random positions.

| flight → map | damaged: success / med. inliers | random: success / med. inliers |
|---|---|---|
| 2020 → 2020 | 98 % / 348 | 95 % / 237 |
| 2022 → 2022 | 100 % / 474 | 98 % / 240 |
| 2025 → 2025 | 100 % / 436 | 100 % / 266 |
| 2022 → 2020 | 12 % / 5 | 2 % / 5 |
| 2020 → 2022 | 28 % / 6 | 10 % / 5 |
| 2025 → 2022 | 52 % / 13 | 15 % / 5 |
| 2022 → 2025 | 42 % / 9 | 12 % / 5 |
| 2020 → 2025 | 10 % / 5 | 0 % / 5 |
| 2025 → 2020 | 2 % / 5 | 0 % / 5 |

Results:

- Classic SIFT works only when the map and the flight come from the same epoch.
- Across epochs it collapses: 0–52 % success.
- The 5-year gap (2020 ↔ 2025) is the hardest.
- Damaged frames do better than random ones. The strata are confounded by land cover: damaged frames are urban, with persistent road and block structure, while random frames are mostly pine forest and fields that change with season and harvest. Destruction is not the dominant source of drift here; time and land cover are.

Rerun: `python drift_matrix.py ../../data/kyiv_cluster/maps ../../data/kyiv_cluster/unosat_labels_cluster.geojson results 40`
Overview: `results/cluster_overview_3epochs.jpg`; per-town triptychs in `results/`.

## Seams: audit, failed harmonisation, split

Source captures per map, from `coverage.py`, which rasterises the release's metadata polygons into `data/kyiv_cluster/coverage/`:

- **Pre-war:** 2017-05-05 (35 %), 2019-04-03 (30 %), 2020-03-03 (32 % hazy and 3 % grey).
  Every release from 2020 to 24 Feb 2022 is a patchwork like this; there is no clean pre-war release.
- **War:** 2022-03-31 (57 %), 2022-03-21 (24 %), 2022-03-18 (13 %), plus slivers from 02-28 and 05-09.
- **Rebuilt:** 2025-04-11 (96 %). Esri filled the cloudy spots with 2022-03-21 imagery, and every release up to 2026-11 still has these patches.

**Geometry is not the problem.** Per-tile phase correlation against 2025 puts every patch within about 1–2 px of it on open ground. Forest tiles in the off-nadir 2019 capture read +5–8 px, which is canopy relief displacement, not a misplaced patch.

**Colour harmonisation failed** (`harmonize.py`: one quantile tone curve per patch fitted to 2025, Gaussian-blended across seams). The seam colour step went from 1.6× to only 1.4× the natural neighbour step. The uneven haze of the 2020-03-03 capture cannot be fixed with one curve per patch, and the output looked blotchy. The harmonised maps were deleted.

**Split** (`split_maps.py`). Each epoch is cut into the largest rectangles inside a single capture, at least 32 px from any seam, with a minimum side of 2.3 km. Results are in `data/kyiv_cluster/maps/split/` (sidecar `offset_px` maps them back to the shared grid) and in `results/split_maps_overview.jpg`.

| epoch | maps | share of UNOSAT destroyed/severe labels a 640 px frame can be centred on |
|---|---|---|
| pre-war | 2017 8.1×7.7 km, 2020 6.7×6.4 km (hazy), 2019 3.8×10 km | 21 % |
| war | 03-31 7.4×14.6 km, 03-21 5.9×8.2 km | 45 % |
| rebuilt | 4 pieces of 2025-04-11 (7.0×7.2, 9.0×4.4, 3.5×9.8, 9.0×2.6 km) | 75 % |

## Town tiles and full 2025 map (`town_tiles.py`, `town_drift.py`)

- **Full 2025 map kept whole.** It is 95.6 % April 2025 imagery. `maps/cluster_2025_rebuilt_valid_mask.png` (255 = 2025) marks Esri's cloud-fill patches, which are 2022 imagery.
- **Seam-free town tiles.** Each tile is a single capture in the pre-war and war maps. They are in `data/kyiv_cluster/towns_seamfree/` and use the same pixel frame across epochs (`offset_px` in the sidecar).

| town | tile | destroyed/severe labels inside | pre-war capture |
|---|---|---|---|
| Irpin | 3.0 km | 357 / 813 | 2020-03-03 (hazy) |
| Hostomel | 2.4 km | 126 / 381 | 2019-04-03 |
| Moschun | 1.55 km | 263 / 452 | 2019-04-03 |
| Bucha | 3.0 km | 26 / 91 (sparse damage) | 2020-03-03 (hazy) |

Cross-epoch SIFT on these tiles (20 frames × 4 towns per pairing; success / median inliers):

| flight ↓ map → | pre-war | war | rebuilt |
|---|---|---|---|
| pre-war | **98 % / 200** | 5 % / 5 | 0 % / 5 |
| war | 5 % / 5 | **100 % / 286** | 2 % / 5 |
| rebuilt | 0 % / 5 | 5 % / 5 | **100 % / 246** |

The tiles were chosen for maximum damage density, so cross-epoch matching is even harder here than on the whole map (2–52 %). Seams are removed, so what remains is pure temporal drift.

Figures:
- `results/town_tiles_overview.jpg`
- `results/towns_seamfree_all.jpg`
- `results/<town>_seamfree_triptych.jpg`

## Single-date maps (`single_date_map.py`), replacing the patchwork epochs

A Wayback release shows each capture only where nothing newer covers it, but the capture itself is the same photograph in every release. `single_date_map.py` queries all releases' metadata, rasterises one capture's footprint at full resolution (eroded 4 px), and takes the pixels only from there. Where two releases overlap, the pixels agree to about 3 grey levels (JPEG noise), which shows the result is one photo. Each map was then shifted onto the 2025 grid with a single sub-pixel translation (phase correlation).

| map | capture | covers | towns fully inside | offset vs 2025 after alignment |
|---|---|---|---|---|
| `cluster_2017-05-05` | WV02, one release (2019-09-18) | 96.8 % (SE corner missing) | all four | median 0.5 px on built-up/open ground; pine forest up to ~8 px (canopy parallax) |
| `cluster_2022-03-31` | WV03, one release (2024-02-08) | 56.6 % (west) | Irpin, Bucha | median 0.24 px shift, 0.55 px spread |
| `cluster_2022-03-21` | WV02, two releases | 28.1 % (east) | Hostomel, Moschun | median 0.6 px shift, 0.5 px spread |
| `cluster_2025_rebuilt` | WV02 2025-04-11 (already single-date) | 95.6 % (cloud holes masked) | all four (Bucha 98 %) | reference |

Each map has a `_valid_mask.png` (255 = that capture) and a JSON sidecar with provenance. No single war-time capture covers all four towns. In the area where 2017, 2022-03-31 and 2025 are all valid (53.6 % of the cluster), a flight crosses Irpin and Bucha with exact single-date imagery in each year.

Overview: `results/single_date_maps.jpg`.
Rebuild: `python single_date_map.py ../../data/kyiv_cluster/maps/cluster_2025_rebuilt.json 2017-05-05 WV02 ../../data/kyiv_cluster/maps/cluster_2017-05-05`. The alignment shift is a separate step, stored in each sidecar as `aligned_to_2025_shift_px`.
