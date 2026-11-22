# Contributing

Create a `feat/`, `fix/`, or `chore/` branch from `develop` and open a pull request into `develop`.
Keep each PR focused and use commit prefixes such as `feat:`, `fix:`, `docs:`, or `test:`.

- `tools/task.sh test` runs the parallel test gate used for PRs.
- `tools/task.sh test --full` includes tests that require the benchmark datasets. To use that gate in a PR, set
  `TASK_FULL=1`.
- `tools/task.sh pr` tests the branch, pushes it, and opens or updates the PR using the configured GitHub remote.
- `tools/task.sh finish` checks CI before merging. It reuses a passing local result for the same committed tree.

Keep local datasets, run outputs, and model weights out of Git. Put small test fixtures in `tests/fixtures/`.
