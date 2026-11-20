---
name: finish-task
description: Finish the current Orientyr task the team's way - verify tests, open the PR into develop, and after the user approves, squash-merge and move the card to Done. Use when the user says the task is done, wants a PR, or wants to merge.
---

Follow AGENTS.md, "How we work". Run from the task branch (`M1-04-…`). Steps:

1. `git status`: everything relevant committed, nothing from `data/`, `outputs/`, videos or weights staged.
2. Run the full `pytest`. Red → stop and fix; never weaken a test to get green.
3. Check each "Done when" item of the issue (`gh issue view <n> -R kolyakr/orientyr`) against a test or, for visual
   and data tasks, a screenshot or checklist. Anything unmet → the task is not done; say what is missing.
4. If a command became real, update the **Commands** section of AGENTS.md in this branch.
5. Write the PR body to a temp file: what changed; "Done when" item → test; anything the reviewer should look at.
   Run `tools/task.sh pr <that file>` (it rebases on develop, pushes and opens the PR with `Closes #N`).
6. Give the user the PR link. Only when they say merge (and CI is green): `tools/task.sh finish`
   (squash-merge into develop, branch deleted, card → Done, local develop refreshed).
7. Never open or merge a PR into `main`; releases to `main` are done by the repo owner only.
