---
name: finish-task
description: Finish the current Orientyr feature or fix the team's way - verify tests, open the PR into develop, and after the user approves, squash-merge it. Use when the user says the change is done, wants a PR, or wants to merge.
---

Follow AGENTS.md, "How we work". Run from the work branch (`feat/…`, `fix/…` or `chore/…`). Steps:

1. `git status`: everything relevant committed, nothing from `data/`, `outputs/`, videos or weights staged.
2. Run the suite: `tools/task.sh test`. Red → stop and fix; never weaken a test to get green.
3. Check each "Done when" item agreed with the user before coding against a test or, for visual and data changes,
   a screenshot or checklist. Anything unmet → the change is not done; say what is missing.
4. If a command became real, update the **Commands** section of AGENTS.md in this branch.
5. Write the PR body to a temp file, describing only what `git diff --stat origin/develop...HEAD` shows: **What**
   changed and why; **Done when** item → test; anything the reviewer should look at.
   Run `tools/task.sh pr <that file>` (it rebases on develop, runs the tests, pushes and opens the PR into develop,
   titled from the branch: `feat/phone-glare-mask` → `feat: phone glare mask`).
6. Give the user the PR link. Only when they say merge (and CI is green): `tools/task.sh finish`
   (squash-merge into develop, branch deleted, local develop refreshed).
7. Never open or merge a PR into `main`; releases to `main` are done by the repo owner only.
