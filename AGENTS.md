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
data/                   all data (gitignored; from the team Google Drive via tools/data.sh, see "Data")
src/dronenav/           package: io, preprocess, matching/, localize, tracking, viz
scripts/                CLI entry points (run.py, make_video.py, benchmark.py, synth_flight.py)
outputs/                generated images/videos (gitignored, except curated results/)
results/                final images/videos used in README and pitch
tests/                  quick sanity tests (homography math, synthetic GT)
ui/                     mission-control console: a Next.js app that replays a run folder (see ui/README.md)
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

### One-time setup per person

```bash
gh auth login                      # as your own GitHub user
gh auth refresh -s project         # lets gh move cards on the board
git clone git@github.com:kolyakr/orientyr.git && cd orientyr   # lands on develop (the default branch)
git config user.name "<you>" && git config user.email "<your GitHub email>"
```

Open Claude Code (or any agent) at the repo root: `CLAUDE.md` imports this file, so the agent starts with these rules.

### Task lifecycle: always through `tools/task.sh`

Every step that touches GitHub goes through [`tools/task.sh`](tools/task.sh), so the board, issues and branches never
drift apart. In Claude Code the same flow is `/start-task` and `/finish-task` (in `.claude/skills/`).

| Step | Command | What it does |
|---|---|---|
| 1. Pick | `tools/task.sh ready` | tasks you can start now: unblocked and unassigned, or unblocked and claimed by you; prefer the critical path M0 → M1 → M4 → M5 → M7 |
| 2. Start | `tools/task.sh start <issue#>` | refuses if blocked, taken or the tree is dirty; assigns you; board → **In Progress**; branch `M1-04-<slug>` off fresh `develop`, pushed (resumes it if it exists); prints the spec |
| 3. Tests (red) | write tests, `tools/task.sh test` shows them fail, `git commit -m "M1-04: tests for … (red)"` | see TDD rules below |
| 4. Implement (green) | code until `tools/task.sh test` passes; `git commit -m "M1-04: …"` | small commits, ID first |
| 5. PR | `tools/task.sh pr [body.md]` | rebases on `develop`, runs the test gate on the result (refuses if red), pushes, opens the PR into `develop` titled like the issue with `Closes #N`; rerun to update it. The gate is the fast suite: in parallel, without the `data` tests; `TASK_FULL=1 tools/task.sh pr` (like `tools/task.sh test --full`) runs them too, and the PR body says which it was |
| 6. Merge | `tools/task.sh finish` | refuses on red tests, unpushed commits, failing checks or a PR not into `develop`; squash-merges, deletes the branch, board → **Done**, back on fresh `develop`. Tests already green on the same committed tree (the run `pr` just made) are not run again; any new commit brings the run back |

Other commands: `tools/task.sh mine` (my tasks and what blocks them), `claim <issue#>` (reserve a task, blocked or
not), `drop` (give the current task back: board → Todo, unassigned; then say why in the issue),
`status <issue#> <Todo|In Progress|Done>`, `notify` (tasks just unblocked for me; the Claude Code hooks run it),
`test` (the gate; `test --full` adds the `data` tests; `test <pytest args>` runs pytest with exactly those args).
One task in progress per person at a time.

**Owner-only tasks** (label `owner-only`, e.g. M8-07: repo clean-up and publishing) are done by @kolyakr alone.
`tools/task.sh` refuses them for anyone else and hides them from `ready`; the `owner-only` workflow unassigns
anyone else and fails their PR's check.

### Blocked tasks and notifications

- A task's blockers are its GitHub "blocked by" links (from "Depends on"). Open tasks with an open blocker carry the
  label **`blocked`**, kept in sync automatically: filter the board by it.
- **Want a blocked task? Claim it:** `tools/task.sh claim <issue#>`. It's assigned to you but stays in Todo.
- **When a task closes** (its PR merged with `Closes #N`), the workflow `.github/workflows/unblock.yml` checks every
  task it was blocking. If that was the last open blocker, it removes `blocked` and comments on the task
  **@mentioning the assignee** ("🔓 Unblocked … start with `tools/task.sh start N`"), which arrives as a GitHub
  notification and email. An unclaimed task gets a "free to take" comment instead, and the closed task gets a list of
  what it unblocked. Reopening a task re-blocks its dependants.
- Make sure GitHub notifications for this repo reach you (Watch → Participating and @mentions, or All Activity).
- **Your Claude hears it too.** The shared [.claude/settings.json](.claude/settings.json) runs `tools/task.sh notify` when
  a Claude Code session starts, and again on your prompts at most every 10 minutes. It lists tasks the `unblock`
  workflow has opened since the last check that are claimed by you or unclaimed, and the agent tells you about them
  before anything else. It prints nothing when there is no news and never blocks the session. You can also run it by hand.
- Edited a "blocked by" link by hand? Labels re-sync every 6 hours, or at once with `gh workflow run unblock.yml`.

### Workflow for a task: the rules behind the commands (humans and agents alike)

1. **Pick a task** on the board whose blockers are all closed; assign yourself and move it to *In Progress*.
2. **Branch off the latest `develop`:** `git switch develop && git pull && git switch -c M1-04-geometry`.
3. **Spec first.** The issue (What / Depends on / Done when) is the spec. If it is ambiguous or wrong, settle it in
   the issue *before* coding and edit the issue text, so the spec always matches what gets built.
4. **Tests first (TDD), see below.** Write the tests that encode "Done when", run them, see them **fail**, and commit
   them before any implementation: `M1-04: tests for homography checks (red)`.
5. **Implement until the tests pass**, then refactor with the tests green. Work only within the task's scope; don't
   edit another milestone's module without a linked task; open or reference an issue for anything outside it.
6. **Commit** with the ID first: `M1-04: add MAGSAC++ homography and sanity checks`.
7. **Keep up to date:** `git fetch && git rebase origin/develop` (or merge `develop` into your branch) before opening the PR.
8. **Open the PR into `develop`** (never into `main`): title `M1-04 · <title>`, body with what changed, which tests
   cover each "Done when" item, and `Closes #<issue>`. The tests commit must come before the implementation commits.
   CI must pass.
9. **Done means the issue's "Done when" is met and proven by tests**, not "code exists". Merge (squash), delete the
   branch, the issue closes.

### Spec-driven development and TDD

No implementation without a failing test that the spec demands. Every "Done when" item maps to at least one test.

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
- **Never weaken a test to make it pass.** If a test or the spec is wrong, say so in the issue, fix the spec first,
  then the test, and mention it in the PR.
- **Visual and data tasks** (renderers, UI, recordings, area choice): test the contract that can be checked (file
  exists, size, frame count, codec, schema, runs on the fake `run.jsonl`) and add a screenshot or a checklist in the PR
  for the part only a human can judge.

### Releasing to `main`

- Only the owner opens and merges a PR **`develop` → `main`** (merge commit, not squash, so history stays shared),
  after checking that `develop` runs end to end (`scripts/run.py` on the task data at least).
- Hotfix on `main`: branch `hotfix-<slug>` off `main`, PR into `main`, then merge `main` back into `develop`.

### Rules for coding agents

- Never commit or push to `main` or `develop` directly; never force-push a shared branch; never merge into `main`.
- At the start of a session, run `git status` and `tools/task.sh mine`. If the human already has a task in
  progress, continue it on its branch; otherwise propose tasks from `tools/task.sh ready` and let the human choose.
  If the work asked for matches no task, ask for one (or ask the human to open an issue) rather than inventing work.
- When a "🔓 Orientyr board news" block appears in your context (from the `notify` hook), tell the human about those
  tasks first and offer to start one, but don't drop a task already in progress without asking.
- Use `tools/task.sh` for start / PR / finish / status; don't hand-edit the board or invent branch names.
- Never propose, claim or work on an `owner-only` task unless the human is @kolyakr.
- Commits: message starts with the task ID (`M1-04: …`), one logical change each, made in the human's git identity.
  Never commit `data/`, outputs, videos, weights or secrets; check `git status` before every commit.
- Run tests only via `tools/task.sh test` (it puts the repo root and `src/` on the import path, as CI must). With no
  arguments it is the gate `pr` and `finish` run: the suite in parallel (`-n auto`), without the `data` tests. Run
  `tools/task.sh test --full` when the task touches code the `data` tests cover and `data/` is on the machine; never merge red.
- Tests run in parallel, in any order: write only under `tmp_path`, never to a fixed path in `outputs/` or the repo.
- `finish` merges into `develop` only after the human says so.
- Outputs go to `outputs/` (gitignored); only curated, small ones go to `results/`.
- When a command becomes real, update the **Commands** section below in the same PR.
- Data moves only through `tools/data.sh` (see **Data** below). `ls`, `pull` and `check` are always fine;
  `push` uploads to the shared team folder, so run it only when the human asks, show the dry run first,
  and pass `--yes` after they confirm. Never delete or rename anything on the Drive.

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
task's issue. Push needs edit access to the folder (ask @kolyakr). Without rclone, download the folder in the
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
python scripts/scenarios.py run all                       # M8-01: every demo scenario (A-F) from one command each; `list` shows them
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
```
