# Data probe

These are throwaway experiment scripts, not project code. They produced the measurements in [docs/UPGRADES.md §1](../../docs/UPGRADES.md#1-what-our-data-says): what the 50 given crops are, how far classic CV gets on them, and where it breaks.

They need only OpenCV, NumPy and SciPy, and run on the CPU.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install opencv-contrib-python numpy scipy   # tested with opencv-contrib-python-headless 5.1.0
```

## Run (from the repo root)

```bash
mkdir -p outputs/data_probe
python experiments/data_probe/probe.py data/Trainee_2025_classic_cv outputs/data_probe             # registers the 50 crops (~0.6 s each)
python experiments/data_probe/fit.py outputs/data_probe/probe.json                                  # fits the task's flight generator
python experiments/data_probe/stress.py data/Trainee_2025_classic_cv outputs/data_probe/probe.json  # global search: 7 degradations × 3 levels
python experiments/data_probe/track.py data/Trainee_2025_classic_cv outputs/data_probe/probe.json   # the same frames with a tracking prior (~3 min; --quick = smoke run)
python experiments/data_probe/summarize_track.py outputs/data_probe > outputs/data_probe/track_summary.md
```

## What each script answers

| Script | Question | Writes (next to probe.json) |
|---|---|---|
| `probe.py` | Where is each crop on the map? How do rotation, scale, perspective, overlap and imagery differ? | `probe.json`, `route_probe.jpg`, `side_*.jpg` |
| `fit.py` | What generated the 50 crops? | stdout (kept as `results/fit.txt`) |
| `stress.py` | When does global search with no prior break? Does it ever return a wrong fix? | `stress.json`, `stress_*.jpg` |
| `track.py` | A tracker always has a prior. With one (tight, loose, or after coasting), which local method recovers the frames that global search loses? | `track.json` |
| `summarize_track.py` | Tables of success and wrong fixes per method and prior, accept-test separation, agreement between methods (case by case), inlier-threshold sweeps, accuracy | markdown on stdout |

## Definitions

- **Frames:** each degraded condition uses 17 crops (ids 0, 3, …, 48). There are 7 kinds × 3 levels = 357 trials. `stress.py` and `track.py` use the same degraded images; `summarize_track.py` asserts this.
- **Success / wrong:** success means the fix is < 5 map px from the truth. Wrong means a fix was declared but it is > 20 px off. The truth is the clean registration in `probe.json`.
- **Priors** (σ of translation px, rotation °, scale, tilt °):
  - tight = 8, 5, 0.05, 3 (the previous frame was tracked);
  - loose = 20, 15, 0.15, 8;
  - coast = 40, 30, 0.25, 12 (several frames were lost).
- **Accept tests, declared before the run:** global and windowed SIFT ≥ 25 inliers; NCC ≥ 0.4; ECC and NCC→ECC ≥ 0.5.

## Results kept here

`results/` holds the outputs of the runs quoted in UPGRADES.md:
- `probe.json`: the registration of every crop. It can serve as ground truth, since the generator fit leaves only 0.15 px of residual.
- `fit.txt`, `stress.json`, `track_log.txt` and `track_summary.md`.

`track.json` (437 KB) is not kept; it takes about 3 minutes to regenerate.

## Key results

- **All 50 crops register.** They have 120–384 inliers, and the median reprojection error is 0.16 px.
- **What the crops are:**
  - the map's own imagery at 4× resolution (scale 0.2535 map px per crop px; NCC 0.976 after matching blur);
  - no perspective;
  - rotated to the direction of flight, within 0.34°;
  - along a sine path: x = 319.9 + 63.41·t, y = 987.6 − 492.8·sin(2πt / 49.0), with a residual of 0.15 px RMS.
- **Global search with no prior** localizes 272 of 357 degraded frames, with zero wrong fixes. Strong Gaussian blur, strong motion blur and the strongest tilt levels defeat it.
- **With a tracking prior,** a cascade recovers 348/357 frames with a tight prior, 319 with a loose one and 301 after coasting, all with zero wrong fixes.
  - The cascade tries windowed SIFT, then NCC→ECC, then global search.
  - Each tracking step costs 13–35 ms, against 244 ms for global search.
- **On the 102 hardest frames,** global search finds the right place in 70 cases, but its 25-inlier accept test keeps only 17.
  - At 8 inliers it keeps 66, still with zero wrong fixes.
  - The margin is thin, though: one wrong fix had 6 inliers. The sweep is also scored on the same frames it would be tuned on, so it is optimistic.
- **When windowed SIFT and NCC→ECC both accept a frame but disagree** (5 cases, all with a loose or coasting prior), SIFT is the accurate one every time: 0.2–0.6 px against 6–27 px.

## Caveat: the strongest tilt levels are degenerate

`tilt_h()` in `stress.py` rotates a camera with a 60° field of view, then rescales the warped image to fit in 640 px. At 50–60° of tilt the far corners reach or pass the horizon, and the frame collapses into a thin sliver (see `stress_phone_combo.jpg`).

So treat these two levels as an extreme bound, not a realistic phone angle:
- tilt L2 (60°);
- phone_combo L2 (50° plus all the other degradations).

The real benchmark (UPGRADES, U2) should render views from a camera model: fixed intrinsics, the image cropped rather than rescaled, and views past the horizon clipped.
