# Data card

Where our showcase imagery comes from, and why. Each section documents one dataset and the
candidates it was picked from, so the choice can be checked rather than taken on faith. M3-07 (#34)
adds the fetch script and the rest of this card; this section is the candidates considered for the
showcase area (M3-01 / M3-02).

## Candidates considered

**Method.** `experiments/ukraine_scan/scan.py` queries every Esri Wayback release's metadata for
each of the 18 UNOSAT damage-assessment areas (UKR1–UKR18; the search box is the extent of that
area's own UNOSAT labels, `+4 km` — there is no separate AOI-polygon file in our data). For every
square size (3/5/8 km) it looks for the square that lies entirely inside one *pre-war*, one *war*
(2022-02-24 to 2022-12-31) and one *post-war* (from 2023-04-01) single-date capture, and ranks the
feasible squares by `score()`: the destroyed/severe UNOSAT label count inside the square, times a
penalty per risk factor (see `penalties()` in `scan.py`) — a pre-2019 "before" capture, a war
capture that isn't UNOSAT's own assessment date, a season gap between captures, mixed sensors,
resolution worse than 0.6 m, and whether the area is occupied or front-line today (2026-11-21,
ISW/Critical Threats). Run: `python experiments/ukraine_scan/scan.py
data/irpin/source/unosat_ukraine_zenodo15088349.geojson <cache> <out>`; the best candidate per
area and size is committed at `experiments/ukraine_scan/results/candidates_summary.json`.

**Best candidate per area**, highest-scoring size first:

| Area | Cities | Status | Size | Pre / War / Post | Worst res | Destroyed+severe | Score |
|---|---|---|---|---|---|---|---|
| UKR2 | Bucha, Hostomel, Irpin, Moschun, Vorzel, Antonov Airport | none | 3 km | 2020-03-03 WV03 / 2022-03-31 WV03 / 2025-04-11 WV02 | 0.5 m | 369 | **264.0** |
| UKR3 | Lysychansk, Rubizhne, Sievierodonetsk | occupied | 3 km | 2020-10-13 WV03 / 2022-07-02 WV03 / 2023-09-01 GE01 | 0.46 m | 1496 | 100.9 |
| UKR4 | Volnovakha | occupied | 5 km | 2021-07-18 WV02 / 2022-09-02 WV03 / 2023-08-07 GE01 | 0.5 m | 789 | 75.9 |
| UKR1 | Mariupol | occupied | 3 km | 2021-06-21 WV02 / 2022-07-31 WV02 / 2024-06-24 WV03 | 0.5 m | 678 | 67.2 |
| UKR8 | Borodyanka | none | 3 km | 2010-06-11 WV02 / 2022-05-02 WV03 / 2023-09-13 GE01 | 0.5 m | 78 | 23.2 |
| UKR5 | Avdiivka | occupied | 3 km | 2021-10-16 WV03 / 2022-08-24 GE01 / 2023-09-25 WV03 | 0.46 m | 117 | 10.8 |
| UKR10 | Mykolaiv | none | 3 km | 2020-04-12 WV02 / 2022-05-21 WV02 / 2023-06-05 WV02 | 0.5 m | 16 | 8.7 |
| UKR16 | Antonivka, Kherson | front_line | 8 km | 2019-08-19 WV03 / 2022-09-25 WV02 / 2024-07-09 WV02 | 0.5 m | 47 | 9.5 |
| UKR14 | Kramatorsk | front_line | 5 km | 2019-06-27 WV02 / 2022-08-08 WV03 / 2024-07-10 WV02 | 0.5 m | 9 | 2.2 |
| UKR18 | Melitopol | occupied | 3 km | 2014-07-29 GE01 / 2022-07-02 GE01 / 2023-07-27 WV02 | 0.5 m | 9 | 0.7 |
| UKR11 | Shchastia | occupied | 3 km | 2021-06-19 GE01 / 2022-07-07 WV02 / 2024-05-28 WV03 | 0.5 m | 8 | 1.1 |
| UKR17 | Kremenchuk | none | 3 km | 2019-09-07 WV03 / 2022-08-27 WV02 / 2025-08-17 WV03 | 0.5 m | 1 | 0.5 |
| UKR6 | Chernihiv | none | — | no feasible square | — | — | — |
| UKR7 | Kharkiv | front_line | — | no feasible square | — | — | — |
| UKR9 | Makariv | none | — | no feasible square | — | — | — |
| UKR12 | Sumy | front_line | — | no feasible square | — | — | — |
| UKR13 | Trostianets | none | — | no feasible square | — | — | — |
| UKR15 | Okhtyrka | none | — | no feasible square | — | — | — |

**Rejected for no feasible square** (UKR6, UKR7, UKR9, UKR12, UKR13, UKR15): no single pre/war/post
triple of Wayback captures fully covers even the smallest (3 km) search box for that area — the
releases that do cover it don't line up into a clean three-epoch set. A coarser grid or a relaxed
cover threshold could recover some of these; out of scope here.

**Rejected on status** (UKR1, UKR3, UKR4, UKR5, UKR11, UKR18 occupied; UKR7, UKR12, UKR14, UKR16
front-line): several of these outrank UKR2 on raw destroyed/severe count (Lysychansk cluster:
1496; Volnovakha: 789; Mariupol: 678), but filming, visiting or even safely referencing an occupied
or actively contested area isn't something we can do for a showcase demo — `penalties()`'s
`status` factor (×0.2 occupied, ×0.5 front-line) reflects that, not just image quality.

**Kept: UKR2** (Bucha/Hostomel/Irpin/Moschun/Vorzel) is both the top-scoring *and* the only
`none`-status candidate in the top half of the table: liberated since March 2022, no current
front-line activity, and the second-highest destroyed/severe count of any area after the three
occupied ones above it.

## Why West Irpin specifically

UKR2's own best candidate above (3 km, 369 labels, 2020/2022/2025) uses this scan's three-epoch
method. The area that was actually built and shipped, **West Irpin**, instead comes from
`kyiv_years.py` + `results/kyiv_years.json`: within the same Kyiv-region cluster (UKR2/8/9, all
`none` status), it looks for the largest square shared by **four** single-date captures, one per
year (2017, 2020, 2022, 2025), rather than three epochs — a longer, more demonstrable before/
during/after timeline for the pitch. Of its three largest common squares, the two biggest (6.76 km
and 5.01 km) hold 0 UNOSAT labels; the third, **3.84 km centred at 50.501°N 30.2216°E**, holds
**310** labels and is the one used: 2017-05-05 (WV02, 0.5 m), 2020-03-03 (WV03, 0.31 m), 2022-03-31
(WV03, 0.31 m), 2025-04-11 (WV02, 0.5 m) — four sharp, single-sensor-per-date captures, 310 vs.
369 labels being a minor difference against a meaningfully longer timeline. `west_irpin*.py` and
`preview.py` build, cut and preview this square.

**M3-02 decision record.** The two sections above are the review M3-02 (#29) asks for: the choice
(West Irpin, backed by UKR2 as the scan's own top-ranked candidate and backup) and the reasons for
each (status/feasibility ranking, then the four-epoch timeline trade-off).
