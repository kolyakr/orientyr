# AGENTS.md — Orientyr (Spectron hackathon): GNSS-denied drone navigation

Shared context for any coding agent (Claude Code, Codex, Cursor, …) working in this repo.
Full task statement: [docs/TASK.md](docs/TASK.md) (extracted from the IT-JIM PDF in the repo root).

## Mission

We are competing in a hackathon. The task: given a **satellite panorama** and **50 consecutive UAV frames**
(rotated, different scale), recover and visualise the drone's route by matching each frame to the map —
i.e. visual navigation for a drone that has lost GPS (electronic-warfare resistant).

**Chosen level: ADVANCED.** Our main input is our own video, filmed with a phone/webcam over the map
shown full-screen (real camera blur, glare, perspective, screen moiré). The provided 50 frames are the
standard-level dataset — we support them too, as a sanity check and benchmark.

**Winning criterion for us = impression.** A correct trajectory is the baseline; judges remember
what looks and feels like a real product. Every decision should serve one of:

1. **It works, visibly** — every frame localized, with confidence, on the real data and on our own video.
2. **It is sophisticated** — a principled pipeline (coarse→fine, temporal filtering, failure recovery), not one `findHomography` call.
3. **It is measured** — numbers (localization error, success rate, fps) from a ground-truth benchmark we build ourselves.
4. **It looks spectacular** — a "mission control" video/demo that a judge watches once and remembers.

## Deliverables (hard requirements — never break these)

- Code reads the **map** and a **folder of frames** (or a **video**) from CLI-specified paths.
- Output: a **trajectory image** (map + route + start/finish + numbered points) **and** a **video** with live UAV position.
- `requirements.txt` (or `environment.yml`), `README.md` describing the approach, well-commented code.
- **Advanced level (chosen, required for us):** our own phone/webcam video filmed over the map on a screen,
  processed by the same pipeline. Video frames need extra preprocessing: sampling every N-th frame, blur
  rejection, screen-glare/moiré handling, and a perspective (homography, not just similarity) model.

## Planned pipeline (target architecture)

```
frames/video ─► preprocess ─► coarse localization ─► fine registration ─► pose ─► temporal filter ─► viz
                 (resize,       (where on the map?     (homography with     (x, y,      (Kalman / outlier
                  CLAHE)         retrieval or search    RANSAC/MAGSAC++)     heading,     rejection,
                                 window from prior)                          scale)       smoothing)
```

- **Matching backends, pluggable** behind one interface:
  - classic baseline: SIFT / RootSIFT + FLANN + Lowe ratio, ORB/AKAZE for speed;
  - DL: SuperPoint + LightGlue (via `kornia` or `lightglue`), optionally LoFTR / RoMa for hard frames.
- **Coarse-to-fine**: downscale the map for a global search (first frame / after loss), then refine
  in a local window around the predicted position at higher resolution.
- **Pose from homography**: frame center → map point, heading from rotation, scale ≈ altitude proxy,
  footprint polygon = warped frame corners. Reject degenerate homographies (det, aspect, inlier ratio).
- **Temporal reasoning**: motion prior to restrict search, Kalman filter / smoothing, interpolation for
  failed frames, explicit "LOST → RELOCALIZING → TRACKING" state.
- **Confidence** per frame (inliers, reprojection error) — shown in the viz, used by the filter.

## "Wow" layer (pick in this order, stop when time runs out)

1. Mission-control video: map with trail colored by confidence, rotated footprint boxes, heading arrow,
   picture-in-picture of the current drone frame + match lines, HUD (frame #, x/y, heading, inliers, state).
2. Synthetic benchmark: generate flights from the map with known ground truth (random path, rotation,
   scale, blur, brightness, perspective) → report RMSE in px and % localized; compare classic vs DL.
3. Robustness demo: simulated "jamming" — dropped frames, noise, occlusion — and the tracker recovers.
4. (Optional) interactive web viewer to scrub the flight.

## Repo layout (intended — create as needed)

```
docs/TASK.md            task statement (+ docs/task_images/)
data/                   input map + frames (gitignored; download link in docs/TASK.md)
src/dronenav/           package: io, preprocess, matching/, localize, tracking, viz
scripts/                CLI entry points (run.py, make_video.py, benchmark.py, synth_flight.py)
outputs/                generated images/videos (gitignored, except curated results/)
results/                final images/videos used in README and pitch
tests/                  quick sanity tests (homography math, synthetic GT)
```

## Engineering conventions

- Python 3.11+, `opencv-contrib-python`, `numpy`, `torch` (use `mps` on Apple Silicon, fall back to CPU), `kornia`.
- Type hints and short docstrings; comments explain *why*, not *what*. Code is part of the grade.
- Coordinates: always **map pixel coords (x, y)** at full map resolution; convert at the boundaries only.
- Everything deterministic and runnable with one command; config via CLI args / a small dataclass, no hard-coded paths.
- Cache expensive things (map keypoints/descriptors) to `outputs/cache/`.
- Never commit `data/`, large videos, or model weights; keep curated results small.
- Prefer a working end-to-end slice first, then improve components. Always keep `main` demo-able.

## How we work: tasks, branches, PRs

Every change belongs to a task. Tasks are GitHub issues titled `M<n>-<nn> · <title>` (e.g. `M1-04 · Geometry: homography, checks, pose`),
tracked on the board <https://github.com/users/kolyakr/projects/1>. An issue's "blocked by" links show what must be merged first.

### Branches

```
main ◄──── release PR (owner only) ──── develop ◄──── PR ──── M1-04-geometry
                                                 ◄──── PR ──── M2-03-quality-gate
                                                 ◄──── PR ──── …
```

| Branch | Role | Who changes it |
|---|---|---|
| `main` | Release: always runnable and demo-able; what the judges see. | Only the repo owner (@kolyakr), only by merging a PR **from `develop`**. No direct pushes. |
| `develop` | Integration: the default branch. All finished tasks land here. | Anyone, **only through a PR** from a task branch. No direct pushes, no force-push. |
| `M<n>-<nn>-<slug>` | One task = one branch, e.g. `M1-04-geometry`. | Its owner. Branch off `develop`, PR back into `develop`. Deleted after merge. |

### Workflow for a task (humans and agents alike)

1. **Pick a task** on the board whose blockers are all closed; assign yourself and move it to *In Progress*.
2. **Branch off the latest `develop`:** `git switch develop && git pull && git switch -c M1-04-geometry`.
3. **Work only within the task's scope.** Don't edit another milestone's module without a linked task; if you find
   needed work outside the scope, open or reference an issue instead of doing it silently.
4. **Commit** with the ID first: `M1-04: add MAGSAC++ homography and sanity checks`.
5. **Keep up to date:** `git fetch && git rebase origin/develop` (or merge `develop` into your branch) before opening the PR.
6. **Open the PR into `develop`** (never into `main`): title `M1-04 · <title>`, body with what changed, how it was
   tested, and `Closes #<issue>`. CI must pass.
7. **Done means the issue's "Done when" is met**, not "code exists". Merge (squash), delete the branch, the issue closes.

### Releasing to `main`

- Only the owner opens and merges a PR **`develop` → `main`** (merge commit, not squash, so history stays shared),
  after checking that `develop` runs end to end (`scripts/run.py` on the task data at least).
- Hotfix on `main`: branch `hotfix-<slug>` off `main`, PR into `main`, then merge `main` back into `develop`.

### Rules for coding agents

- Never commit or push to `main` or `develop` directly; never force-push a shared branch; never merge into `main`.
- Before starting, confirm the task ID with the human; if there is none, ask for one rather than inventing work.
- Outputs go to `outputs/` (gitignored); only curated, small ones go to `results/`.
- When a command becomes real, update the **Commands** section below in the same PR.

## Commands (fill in as they become real)

```bash
pip install -r requirements.txt
python scripts/run.py --map data/map.png --frames data/frames --out outputs/run1
python scripts/run.py --map data/map.png --video data/phone.mp4 --out outputs/phone
python scripts/benchmark.py --map data/map.png
```
