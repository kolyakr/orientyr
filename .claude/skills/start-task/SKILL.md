---
name: start-task
description: Pick up an Orientyr task from the board and start it the team's way - check blockers, assign, move the card to In Progress, branch off develop, then write failing tests from the spec before any code. Use at the start of work, or when the user says "start/take/pick a task" or names a task ID or issue number.
argument-hint: "[issue# or task ID, e.g. 12 or M1-04]"
---

Follow AGENTS.md, "How we work". Steps:

1. Run `git status` and `tools/task.sh mine`. If the user already has a task in progress, offer to continue it
   (switch to its branch) instead of starting a new one. Uncommitted tracked changes: stop and ask.
2. If no task was given (`$ARGUMENTS` empty), run `tools/task.sh ready` and suggest 2–3 tasks, preferring the
   critical path (M0 → M1 → M4 → M5 → M7) and small sizes; let the user choose. A task ID like `M1-04` is resolved
   to its issue number from that list or `gh issue list --search "M1-04 in:title"`.
3. Run `tools/task.sh start <issue#>`. If it refuses (blocked, taken), report why and stop.
4. Read the printed spec. If "What" or "Done when" is ambiguous, ask the user and, once settled, update the issue
   text (`gh issue edit`) before coding.
5. TDD, red first: write tests in `tests/` that encode every "Done when" item (rules in AGENTS.md, "Spec-driven
   development and TDD"). Run `tools/task.sh test`, confirm they fail for the right reason, and commit:
   `M1-04: tests for <what> (red)`.
6. Show the user the list "Done when item → test", then implement until green. Commit with the task ID first.
7. When everything passes, tell the user the next step is `/finish-task`.
