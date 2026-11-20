# Feasibility micro-benchmarks (M1 Pro, 16 GB)

These are throwaway timing probes from the research pass, not project code. They back the speed and rotation numbers quoted in [docs/UPGRADES.md](../../docs/UPGRADES.md).

Two things to know before running them:
- Some of them time networks that have the right shape but random weights. Only the timing matters there.
- The matcher benchmarks need a local clone of XFeat, passed as the first argument.

**Dependencies:**
- `torch` (MPS), `torchvision`, `timm`, `transformers`;
- LightGlue (`pip install git+https://github.com/cvg/LightGlue`);
- a clone of [verlab/accelerated_features](https://github.com/verlab/accelerated_features) (XFeat);
- `opencv-contrib-python`, `numpy`.

Weights download on the first run.

| File | Measures | Output |
|---|---|---|
| `bench_matchers.py` | XFeat, XFeat + LighterGlue, SuperPoint / ALIKED / DISK + LightGlue, and SIFT on MPS and CPU, for one tracking pair | `results/results_matchers.json`, `results/bench_log.txt` |
| `bench_rot.py` | Which matchers survive in-plane rotations from 0° to 180°, against a local window and against the whole map | `results/results_rot.json`, `results/rot_log.txt` |
| `bench_reloc.py` | Relocalization budget: XFeat against the whole map, EfficientLoFTR, the DINOv2 forward pass | `results/results_reloc.json`, `results/reloc_log.txt` |
| `bench_reloc_rot.py` | Relocalization at 110°: XFeat on 8 rotated copies against SIFT on the whole map | stdout |
| `corr_bench.py`, `corr_bench2.py`, `corr_bench3.py` | Dense (x, y, heading, scale) correlation of frame features against pre-rotated map feature stacks, by FFT on MPS; the histogram filter; ViT encoder cost | stdout |
| `mps_bench.py`, `mps_vitb.py` | MPS capabilities (autocast, missing ops) and ViT-S/B/L throughput | stdout |
| `scale_check.py` | Crop → map scale with SIFT | stdout |

## Headline numbers

From the files in `results/`, for a 640×480 frame:

| Method | Median time |
|---|---|
| SIFT detect + describe (CPU) | 23 ms |
| SIFT FLANN + ratio match (CPU) | 28 ms |
| XFeat detection (MPS) | 8.5 ms |
| XFeat detect + match of a frame–window pair (MPS) | 19–23 ms |
| LighterGlue match only (MPS) | 26 ms (1024 kp) / 46 ms (2048 kp) |
| SuperPoint + LightGlue match only (MPS) | 42 / 91 ms |
| EfficientLoFTR (MPS / CPU) | 300 / 978 ms |

**Rotation** (3 frames per angle, at 0, 30, 45, 60, 90, 135 and 180°):
- SIFT localizes 21/21, both in a local window and against the whole map, with errors ≤ 1.3 px.
- XFeat, XFeat + LighterGlue, SuperPoint + LightGlue and ALIKED + LightGlue each get 0/9 at 90° and beyond.
- XFeat already drops to 1/3 at 60°.

**Relocalization:** XFeat against the whole map fails on a frame rotated by 70°: 5–6 inliers, errors of 300–2000 px.

The dense-correlation and ViT figures quoted in UPGRADES.md were printed by `corr_bench*.py` and `mps_*.py` during the research pass:
- 90 ms for the full map at 36 headings × 3 scales;
- ViT-S/B/L at 179 / 60 / 18.6 images/s at 224 px.

Rerun the scripts to reproduce them.
