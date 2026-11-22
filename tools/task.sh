#!/usr/bin/env bash
# The test gate and the PR flow for Orientyr: one command per step, so humans and agents follow the same flow.
# See CONTRIBUTING.md for the branch and PR workflow.
# Needs: gh logged in. Works on macOS bash 3.2.
#
#   tools/task.sh test               the gate pr/finish run: the suite in parallel, without the `data` tests
#   tools/task.sh test --full        the gate with the `data` tests too (TASK_FULL=1 does the same for pr/finish)
#   tools/task.sh test <pytest args> pytest with exactly these args (repo root and src/ importable)
#   tools/task.sh pr [body-file]     rebase on develop, tests green, push, open the PR into develop (title from the branch)
#   tools/task.sh finish             tests + checks green, squash-merge into develop, branch deleted, fresh develop
set -euo pipefail

ROOT=$(cd "$(git rev-parse --show-toplevel)" && pwd)  # posix form: git-bash on Windows mis-splits PYTHONPATH otherwise
cd "$ROOT"

die() { echo "task.sh: $*" >&2; exit 1; }

KINDS="feat|fix|chore"   # the branch prefixes `pr` and `finish` accept

work_branch() {  # the current branch, if it is feat/<slug>, fix/<slug> or chore/<slug>; refuses anything else
  local branch; branch=$(git branch --show-current)
  [[ "$branch" =~ ^($KINDS)/[A-Za-z0-9._-]+$ ]] \
    || die "branch '$branch' is not a feature or fix branch: name it feat/<slug>, fix/<slug> or chore/<slug>, off develop"
  echo "$branch"
}

pr_title() {  # feat/phone-glare-mask → "feat: phone glare mask"
  local slug="${1#*/}"
  echo "${1%%/*}: ${slug//-/ }"
}

# --- the test gate ----------------------------------------------------------------------------------------------
# What `pr` and `finish` (and a bare `test`) run. Fast by default: in parallel (pytest-xdist, TASK_JOBS workers,
# default one per core) and without the tests marked `data`, which need the full dataset, take most of the serial
# time and are skipped by CI anyway. TASK_FULL=1 (or `test --full`) adds them.
# A green gate on a clean tree leaves a stamp, "<tree hash> <fast|full>", in this worktree's git dir, so `finish`
# does not repeat the run `pr` has just made on the very same tree.
STAMP="$(git -C "$ROOT" rev-parse --absolute-git-dir)/task-tests-passed"

gate_mode() { if [[ "${TASK_FULL:-0}" == 1 ]]; then echo full; else echo fast; fi; }

tree_id() { git -C "$ROOT" rev-parse 'HEAD^{tree}'; }

tree_is_clean() { git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet; }

in_root() {  # repo root and src/ importable, whatever pytest's rootdir/import mode decides (src layout: src/dronenav)
  (cd "$ROOT" && PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$@")
}

run_tests() {  # the suite must be green before a PR or a merge; no tests yet → say so, don't block
  if [[ -d "$ROOT/tests" ]] && ls "$ROOT"/tests/test_*.py >/dev/null 2>&1; then
    command -v pytest >/dev/null || die "pytest not found; install requirements first"
    local mode args started=$SECONDS
    mode=$(gate_mode); args=(-q)
    # asked through the help text and not `python -c "import xdist"`: pytest may live in another interpreter
    if [[ "$(in_root pytest -h 2>/dev/null)" == *--numprocesses* ]]; then args+=(-n "${TASK_JOBS:-auto}")
    else echo "task.sh: pytest-xdist is not installed, the tests run one by one (pip install -r requirements.txt)" >&2; fi
    [[ "$mode" == full ]] || args+=(-m "not data")
    rm -f "$STAMP"                                    # a red run takes back an earlier green on the same tree
    in_root pytest "${args[@]}" || die "tests are red: fix them (never weaken a test to pass)"
    echo "task.sh: $mode gate green in $((SECONDS - started)) s ($(tests_note))" >&2
    # the stamp speaks for the commit: with uncommitted changes the run says nothing about HEAD's tree
    if tree_is_clean; then echo "$(tree_id) $mode" >"$STAMP"; fi
  else
    echo "task.sh: no tests in tests/ yet, nothing to run" >&2
  fi
}

tests_already_green() {  # did HEAD's tree already pass this gate (or the fuller one)? Untracked files are not seen
  local tree mode
  [[ -f "$STAMP" ]] && read -r tree mode <"$STAMP" || return 1
  [[ "$tree" == "$(tree_id)" ]] && [[ "$mode" == full || "$(gate_mode)" == fast ]]
}

tests_note() {  # for the log and the PR body: were the `data` tests part of the run?
  if [[ "$(gate_mode)" == full ]]; then echo 'the full suite, with the `data` tests'
  else echo 'the fast suite; `data` tests not run (`TASK_FULL=1 tools/task.sh pr` runs them)'; fi
}

TESTS_LINE='**Local tests:**'    # starts the one line of a PR body that `pr` owns and rewrites on every run

with_tests_line() {  # stdin: a PR body → stdout: the same body with the tests line replaced, or added at the end
  awk -v mark="$TESTS_LINE" -v line="$1" '
    index($0, mark) == 1 { if (!done) print line; done = 1; next }
    { print }
    END { if (!done) { print ""; print line } }'
}

require_clean() {
  tree_is_clean || die "uncommitted changes in tracked files; commit or stash first"
}

case "${1:-}" in
  pr)
    branch=$(work_branch)
    require_clean
    git fetch -q origin
    git rebase -q origin/develop || die "rebase on develop has conflicts: resolve, 'git rebase --continue', rerun"
    run_tests                                         # on the rebased branch: that is what will land in develop
    git push -q --force-with-lease -u origin HEAD     # only ever our own branch, after the rebase
    if [[ -n "${2:-}" ]]; then body=$(cat "$2")
    else body=$'## What\n\n## Done when\n\n## Tests: one per "Done when" item\n'; fi
    tests_line="$TESTS_LINE $(tests_note), green on \`$(git rev-parse --short HEAD)\`."
    existing=$(gh pr list --head "$branch" --state open --json url --jq '.[0].url')
    if [[ -n "$existing" ]]; then
      # the body is the author's by now: only the tests line is ours, so it stays true after every push
      { old=$(gh pr view "$existing" --json body --jq .body) \
          && gh pr edit "$existing" --body "$(with_tests_line "$tests_line" <<<"$old")" >/dev/null; } \
        || echo "task.sh: could not update the tests line in the PR body" >&2
      echo "PR already open, branch updated: $existing"
    else gh pr create --base develop --head "$branch" --title "$(pr_title "$branch")" \
           --body "$body"$'\n\n'"$tests_line"; fi
    ;;

  finish)
    branch=$(work_branch)
    require_clean
    [[ -z "$(git log --oneline "origin/$branch..HEAD" 2>/dev/null)" ]] || die "local commits not pushed: run tools/task.sh pr"
    pr=$(gh pr list --head "$branch" --state open --json number,baseRefName --jq '.[0] | "\(.number) \(.baseRefName)"')
    [[ -n "$pr" && "$pr" != "null null" ]] || die "no open PR for $branch: run tools/task.sh pr"
    [[ "${pr#* }" == develop ]] || die "PR #${pr% *} targets ${pr#* }, not develop"
    if tests_already_green; then
      echo "task.sh: tests already green on this tree ($(git rev-parse --short 'HEAD^{tree}'), $(cut -d' ' -f2 "$STAMP") gate): not run again"
    else run_tests; fi
    checks=$(gh pr checks "${pr% *}" --json bucket --jq '[.[] | select(.bucket=="fail" or .bucket=="pending")] | length' 2>/dev/null || echo 0)
    [[ "$checks" == 0 ]] || die "PR #${pr% *} has failing or pending checks: gh pr checks ${pr% *}"
    gh pr merge "${pr% *}" --squash --delete-branch
    git switch -q develop 2>/dev/null || true
    git pull -q --ff-only origin develop
    git branch -D "$branch" >/dev/null 2>&1 || true
    echo "Merged PR #${pr% *} ($branch) into develop. Back on fresh develop."
    ;;

  test)
    shift
    if [[ "${1:-}" == --full ]]; then
      [[ $# -eq 1 ]] || die "'test --full' takes no other arguments (pytest args alone run pytest as is)"
      TASK_FULL=1; shift
    fi
    [[ $# -eq 0 ]] && { run_tests; exit; }
    cd "$ROOT" && PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" exec pytest "$@"
    ;;

  *)
    sed -n '2,10p' "$0" >&2; exit 1
    ;;
esac
