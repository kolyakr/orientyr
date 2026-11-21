# Contributing to Orientyr

How the team works on this repo: conventions, branches, tests, data and commands.

## Engineering conventions

- Python 3.11+, `opencv-contrib-python`, `numpy`, `torch` (use `mps` on Apple Silicon, fall back to CPU), `kornia`.
- Type hints and short docstrings; comments explain *why*, not *what*.
- Coordinates: always **map pixel coords (x, y)** at full map resolution; convert at the boundaries only.
- Everything deterministic and runnable with one command; config via CLI args / a small dataclass, no hard-coded paths.
- Cache expensive things (map keypoints/descriptors) to `outputs/cache/`.
- Never commit `data/`, large videos, or model weights; keep curated results small.
- Prefer a working end-to-end slice first, then improve components. Always keep `main` demo-able.

## How we work: features and fixes

Every change is a **feature** or a **fix**, built on its own branch and merged into `develop` through a PR.
There is no task board any more (all task issues were closed on 21 Nov 2026). Issues are optional: open one for a
bug you can't fix right away, or to talk a feature over before building it.

### Branches

```
main ◄──── release PR (owner only) ──── develop ◄──── PR ──── feat/phone-glare-mask
                                                 ◄──── PR ──── fix/blank-frame-crash
                                                 ◄──── PR ──── …
```

| Branch | Role | Who changes it |
|---|---|---|
| `main` | Release: always runnable and demo-able. | Only the repo owner (@kolyakr), only by merging a PR **from `develop`**. No direct pushes. |
| `develop` | Integration: the default branch. Every finished feature and fix lands here. | Anyone, **only through a PR**. No direct pushes, no force-push. |
| `feat/<slug>` | A new feature, e.g. `feat/phone-glare-mask`. | Its author. Branch off `develop`, PR back into `develop`. Deleted after merge. |
| `fix/<slug>` | A bug fix, e.g. `fix/blank-frame-crash`. | Same as `feat/`. |
| `chore/<slug>` | Docs, tooling, clean-up: no change to what the pipeline does. | Same as `feat/`. |

`<slug>` is a few lowercase words joined by hyphens. One branch holds one change: something else you notice on the
way gets its own branch, or an issue.

### One-time setup per person

```bash
gh auth login                      # as your own GitHub user
git clone git@github.com:kolyakr/orientyr.git && cd orientyr   # lands on develop (the default branch)
git config user.name "<you>" && git config user.email "<your GitHub email>"
```

### The flow

| Step | Command | What it does |
|---|---|---|
| 1. Branch | `git switch develop && git pull && git switch -c feat/<slug>` | a fresh branch off the latest `develop` (`fix/` or `chore/` as fits) |
| 2. Say what "done" means | a few lines: **What** changes, **Done when** (what you can check) | written before any code; it becomes the PR description |
| 3. Tests (red) | write tests, `tools/task.sh test` shows them fail, `git commit -m "test: … (red)"` | a **fix** starts with a test that reproduces the bug; a **feature** with tests of its contract. See TDD below |
| 4. Implement (green) | code until `tools/task.sh test` passes; `git commit -m "feat: …"` (or `fix:`, `chore:`) | small commits, one logical change each |
| 5. PR | `tools/task.sh pr [body.md]` | rebases on `develop`, runs the test gate (refuses if red), pushes, opens the PR into `develop`, titled from the branch (`feat/phone-glare-mask` → `feat: phone glare mask`); rerun to update it. `TASK_FULL=1 tools/task.sh pr` adds the `data` tests |
| 6. Merge | `tools/task.sh finish` | refuses on red tests, unpushed commits, failing checks or a PR not into `develop`; squash-merges, deletes the branch, back on fresh `develop`. Tests already green on the same committed tree (the run `pr` just made) are not run again; any new commit brings the run back |

- **Commit messages** start with the kind of change: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`.
- **The PR description** holds What, Done when, and which test proves each "Done when" item; a screenshot for
  anything visual. The tests commit comes before the implementation commits. CI must pass.
- **Done means "Done when" is met and proven by tests**, not "code exists".

### Spec-driven development and TDD

No implementation without a failing test that the "Done when" demands. Every "Done when" item maps to at least one test.
A fix always comes with the test that would have caught the bug.

What makes a test reliable here:

- **Derived from the spec, not from the code.** Test the contract (inputs → outputs, errors), not internals, so a
  refactor doesn't break it. It must fail before the implementation exists; a test that can't fail is not a test.
- **Ground truth you control.** Prefer synthetic data with exact answers: warp a map patch with a known homography,
  generate a flight with a known route (seeded), then assert the recovered pose within an **explicit tolerance**
  (e.g. `≤ 2 px`, `≤ 1°`) with the reason for that tolerance in a comment.
- **Edge and failure cases too:** degenerate / mirrored homographies, empty or blank frames, no matches, lost track,
  wrong input paths. The rejection path is as much part of the spec as the happy path.
- **Deterministic and fast:** fixed seeds, CPU only, no network, no full dataset. Unit tests run in seconds; tiny
  fixtures live in `tests/fixtures/` (a few KB). Tests that need the full data are marked `@pytest.mark.data` and
  skip cleanly when `data/` is missing; CI runs everything else.
- **Never weaken a test to make it pass.** If a test or the "Done when" is wrong, fix that first, then the test,
  and say so in the PR.
- **Visual and data changes** (renderers, UI, recordings): test the contract that can be checked (file exists, size,
  frame count, codec, schema, runs on the fake `run.jsonl`) and add a screenshot or a checklist in the PR for the
  part only a human can judge.

### Releasing to `main`

- Only the owner opens and merges a PR **`develop` → `main`** (merge commit, not squash, so history stays shared),
  after checking that `develop` runs end to end (`scripts/run.py` on the task data at least).
- Hotfix on `main`: branch `hotfix/<slug>` off `main`, PR into `main`, then merge `main` back into `develop`.

### Rules

- Never commit or push to `main` or `develop` directly; never force-push a shared branch; never merge into `main`.
- Do what was asked and no more. A bug or a better idea found on the way goes into an issue, not into the
  same branch.
- Commits: message starts with its kind (`feat: …`, `fix: …`), one logical change each, made in your own git
  identity. Never commit `data/`, outputs, videos, weights or secrets; check `git status` before every commit.
- Run tests only via `tools/task.sh test` (it puts the repo root and `src/` on the import path, as CI must). With no
  arguments it is the gate `pr` and `finish` run: the suite in parallel (`-n auto`), without the `data` tests. Run
  `tools/task.sh test --full` when the change touches code the `data` tests cover and `data/` is on the machine; never merge red.
- Tests run in parallel, in any order: write only under `tmp_path`, never to a fixed path in `outputs/` or the repo.
- `finish` merges into `develop` only after the PR is reviewed.
- Outputs go to `outputs/` (gitignored); only curated, small ones go to `results/`.
- When a command becomes real, update the **Commands** section below in the same PR.
- Data moves only through `tools/data.sh` (see **Data** below). `ls`, `pull` and `check` are always fine;
  `push` uploads to the shared team folder, so look at the dry run first
  and pass `--yes` only when it shows what you expect. Never delete or rename anything on the Drive.

## Data: the team Google Drive

All data lives in one shared Google Drive folder, which mirrors the local, gitignored `data/` folder:
<https://drive.google.com/drive/folders/1Z5iIhFekHIFkAjH49jxUdf3k4SGw7voe>. Git never holds data; the Drive is
the single source, and everyone keeps a local copy in `data/`.

```
data/hackatons-data/   global_map.png, crops/crop_0.png … crop_49.png        (organisers' task data)
data/irpin/            maps/ hires/ unosat.geojson flights/ drone/ preview/ source/   (our showcase data)
data/video/            phone takes (M2-01), added when filmed
```

`tools/data.sh` wraps rclone, with the remote rooted at the shared folder (paths are relative to `data/`):

| Command | What it does |
|---|---|
| `tools/data.sh setup` | one time: `brew install rclone` first; creates the rclone remote `orientyr` (browser Google login) |
| `tools/data.sh ls [path]` | list the Drive folder or a subfolder, e.g. `irpin/flights` |
| `tools/data.sh pull [path]` | Drive → `data/[path]`: new and changed files only, never deletes local files |
| `tools/data.sh check [path]` | compare `data/[path]` with the Drive by size; lists missing and different files |
| `tools/data.sh push <path> [--yes]` | `data/<path>` → Drive: dry run without `--yes`; never deletes on the Drive; a path is required |

Rules: pull before you start work that reads data. Add new data under a new subfolder (or new files),
never overwrite someone else's files in place; if data must change, add a versioned copy and say so in the
PR. Push needs edit access to the folder (ask @kolyakr). Without rclone, download the folder in the
browser and unzip it into `data/` so the layout above matches (`scripts/check_data.py`, M0-06, will verify it).

## Commands (fill in as they become real)

```bash
pip install -r requirements.txt && pip install -e .   # stack, then the dronenav package (src/ layout)
tools/data.sh setup && tools/data.sh pull               # get the data from the team Drive into data/
python scripts/check_data.py                             # verify data/ layout; names any missing/truncated file
tools/task.sh test                                       # the test gate of pr/finish: pytest -n auto, without the `data` tests
tools/task.sh test --full                                # the gate plus the `data` tests (needs data/); TASK_FULL=1 does it for pr/finish
tools/task.sh test tests/test_geometry.py -k pose        # any pytest args: passed through as they are (serial, nothing added)
ruff check .                                             # lint, as CI does (experiments/ excluded)
python tools/make_ci_fixtures.py --data data             # regenerate tests/fixtures/irpin_mild_2025 (CI fixture)
python scripts/fake_run.py --map data/hackatons-data/global_map.png --out outputs/fake   # run folder with no engine
(cd ui && npm install && npm run dev)                    # the console: http://localhost:3000 replays any run folder in outputs/ (/?run=<name>, or the Run list); ORIENTYR_RUNS=<dir of runs> or ORIENTYR_RUN=<run dir> for others
(cd ui && npm test)                                      # the console's own tests (vitest); pytest also runs them on a fresh fake run
python scripts/render_image.py --run outputs/fake          # trajectory.png + trajectory_preview.jpg (1600 px)
python scripts/pack_run.py --run outputs/irpin --out ui/public/runs/irpin   # a small self-contained run for the hosted console (ui/README.md, "Hosting")
python scripts/chaos_sheet.py --flight data/irpin/flights/mild/2022 --out outputs/chaos_sheets   # one check sheet per chaos degradation (no --flight: synthetic cut-outs, no data/)
python scripts/render_video.py --run outputs/fake          # trajectory.mp4, HUD video: map stage, camera projected onto its footprint, state lamp, captions, timeline
python scripts/run.py --map data/irpin/maps/irpin_2022.png --frames data/irpin/flights/mild/2022/frames --out outputs/jam --jam 150:30   # M7-02: the same run with the camera jammed for 30 s from 150 s of flight
python scripts/render_video.py --run outputs/jam --out outputs/demo --speed 6 --size 1920x1080   # the pitch video of that run
python scripts/render_video.py --run outputs/fake --style classic   # the plain M1 video; for a demo add --size 1920x1080 --speed 6
python scripts/scale_table.py --task data/hackatons-data --probe experiments/data_probe/results/probe.json \
    --irpin data/irpin/flights/mild/2025 --irpin-map data/irpin/maps/irpin_2025.png   # M1-03 scale table (README notes)
python scripts/search_run.py --irpin data/irpin/flights/mild/2025 --irpin-map data/irpin/maps/irpin_2025.png --miss 200   # M1-05 search modes
python scripts/confidence_check.py --irpin data/irpin --out outputs/confidence   # M4-08 confidence table + reliability.png
python scripts/rotation_check.py --irpin data/irpin --out outputs/rotation   # M2-05 success by heading bin (table + rotation.png)
python scripts/preprocess_ablation.py --flight data/irpin/flights/mild/2022 \
    --maps data/irpin/maps/irpin_2025.png data/irpin/maps/irpin_2022.png data/irpin/maps/irpin_2017.png   # M4-05 preprocessing ablation (README notes)
python scripts/run.py --map data/hackatons-data/global_map.png --frames data/hackatons-data/crops --out outputs/run1   # the deliverable: run folder, trajectory image, video (--tracker none: M1's interim states)
python scripts/run.py --map data/hackatons-data/global_map.png --frames data/hackatons-data/crops --out outputs/vo --vo --jam-match 20-29   # M9-02: visual odometry through a simulated map-match jam
python scripts/run.py --map data/irpin/maps/irpin_2025.png --video data/video/take.mp4 --every 10 --out outputs/phone   # M2-09: a video, every 10th frame (default: ~3 frames/s); frames/ holds the sampled frames
python scripts/calibrate_camera.py --images "data/video/calib/*.jpg" --cols 9 --rows 6 --out outputs/camera.json   # M9-03: once-off checkerboard calibration -> a camera.json Camera.from_meta reads straight away
python scripts/run.py --map data/irpin/maps/irpin_2025.png --video data/video/take.mp4 --camera outputs/camera.json --m-per-px 0.1 --out outputs/pose6dof   # M9-03: altitude_m/roll_deg/pitch_deg per accepted fix (--m-per-px: the map's scale, or its own georef sidecar)
python scripts/run.py --map data/irpin/maps/irpin_2022.png --frames data/irpin/flights/mild/2022/frames --matcher xfeat --out outputs/xfeat   # M4-03: the XFeat backend (--device cpu|mps|cuda; the first run downloads its weights into the torch cache)
python scripts/xfeat_timing.py --flight data/irpin/flights/mild/2022 --map data/irpin/maps/irpin_2022.png   # M4-03: ms per frame, XFeat against RootSIFT (no arguments: synthetic cut-outs, no data/)
python scripts/live.py --flight data/irpin/flights/mild/2022 --map data/irpin/maps/irpin_2022.png --out outputs/live   # M7-04 live mode: the flight in real time, records streamed; watch it at /?run=live in the console (--start/--seconds a stretch, --jam START:LENGTH; M7-05 judge controls in the console: jam camera, spoof GPS, map year, new flight; --maps/--flights choose what is offered, --spoof-speed PX_S)
python scripts/accuracy.py --run outputs/run1 --truth experiments/data_probe/results/probe.json   # M1-10: error per frame vs truth (add --map/--frames to localize first)
python scripts/mavlink_log.py --run outputs/irpin --utc-start 2025-06-01T10:00:00Z   # M9-01 stub: GPS_INPUT .tlog (map needs a georef sidecar; --udp HOST:PORT feeds a SITL)
python scripts/benchmark.py --data data --out outputs/bench   # M6-03: scenarios C-F x presets x speeds x trackers -> bench.json + bench.md (--frames N: quick; --reuse: table only)
python scripts/spoof_check.py --flight data/irpin/flights/mild/2022 --map data/irpin/maps/irpin_2022.png   # GPS spoof detector: false-alarm rate, detection delay (--run <run dir> to use a real run's fixes)
python scripts/screen_video.py --map data/hackatons-data/global_map.png --out data/sim_phone/task_calm --preset calm   # M2-08: SIMULATED phone take with exact truth (--sheet PNG: effects)
```
