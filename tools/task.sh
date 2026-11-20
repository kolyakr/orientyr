#!/usr/bin/env bash
# Task lifecycle for Orientyr: one command per step, so humans and agents follow the same flow.
# Needs: gh logged in with the `project` scope (`gh auth refresh -s project`). Works on macOS bash 3.2.
#
#   tools/task.sh ready              tasks I can start now: unblocked + unassigned, or unblocked + claimed by me
#   tools/task.sh mine               my open tasks, with their status and open blockers
#   tools/task.sh claim <issue#>     reserve a (possibly blocked) task; you get pinged when it unblocks
#   tools/task.sh start <issue#>     assign me, board → In Progress, branch M1-04-<slug> off fresh develop
#   tools/task.sh pr [body-file]     tests green, rebase on develop, push, open the PR into develop ("Closes #N")
#   tools/task.sh finish             tests + checks green, squash-merge into develop, board → Done, fresh develop
#   tools/task.sh test [pytest args]  run the tests the way pr/finish do (repo root and src/ importable)
#   tools/task.sh status <issue#> <Todo|In Progress|Done>
#   tools/task.sh drop              give the current task back: board → Todo, unassign, back on develop
#   tools/task.sh notify [--throttle SEC]  tasks unblocked for me since the last check (run by Claude Code hooks)
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
# shellcheck source=tools/task.env
source "${TASK_ENV:-$ROOT/tools/task.env}"
OWNER="${REPO%/*}"; NAME="${REPO#*/}"
TASK_OWNER="${TASK_OWNER:-$OWNER}"   # the person who alone may take owner-only tasks

die() { echo "task.sh: $*" >&2; exit 1; }

status_option() {
  case "$1" in
    "Todo") echo "$STATUS_TODO" ;; "In Progress") echo "$STATUS_IN_PROGRESS" ;; "Done") echo "$STATUS_DONE" ;;
    *) die "unknown status '$1' (Todo | In Progress | Done)" ;;
  esac
}

gql() {  # gql <query> <jq> [extra -F args…]: GraphQL on our repo; $owner, $name are always bound
  local q="$1" f="$2"; shift 2
  gh api graphql -f owner="$OWNER" -f name="$NAME" "$@" -f query="$q" --jq "$f"
}

issue_q() {  # issue_q <issue#> <jq filter applied to the issue>
  gql 'query($owner:String!,$name:String!,$n:Int!){repository(owner:$owner,name:$name){issue(number:$n){
    number title state body labels(first:20){nodes{name}} assignees(first:10){nodes{login}} blockedBy(first:50){nodes{number title state}}
    projectItems(first:10){nodes{id project{id}}}}}}' ".data.repository.issue | $2" -F n="$1"
}

me() { gh api user --jq .login; }

check_owner_only() {  # tasks labelled owner-only (e.g. M8-07, publishing) are for the repo owner alone
  local labels; labels=$(issue_q "$1" '[.labels.nodes[].name] | join(",")')
  if [[ ",$labels," == *",owner-only,"* && "$(me)" != "$TASK_OWNER" ]]; then
    die "#$1 is owner-only: only @$TASK_OWNER can take it"
  fi
}

set_status() {  # set_status <issue#> <status name>
  local opt item
  opt=$(status_option "$2")
  item=$(issue_q "$1" ".projectItems.nodes[] | select(.project.id==\"$PROJECT_ID\") | .id")
  [[ -n "$item" ]] || die "issue #$1 is not on the board"
  gh project item-edit --project-id "$PROJECT_ID" --id "$item" --field-id "$STATUS_FIELD" \
    --single-select-option-id "$opt" >/dev/null
  echo "#$1 → $2"
}

current_issue() {  # issue number of the current task branch, found by its task ID
  local branch id n
  branch=$(git branch --show-current)
  id=$(grep -oE '^M[0-9]-[0-9]{2}' <<<"$branch") || die "branch '$branch' is not a task branch (M1-04-…)"
  n=$(gh issue list -R "$REPO" --state all --limit 300 --json number,title \
      --jq ".[] | select(.title | startswith(\"$id \")) | .number" | head -1)
  [[ -n "$n" ]] || die "no issue titled '$id · …' for branch $branch"
  echo "$n"
}

run_tests() {  # the suite must be green before a PR or a merge; no tests yet → say so, don't block
  if [[ -d "$ROOT/tests" ]] && ls "$ROOT"/tests/test_*.py >/dev/null 2>&1; then
    command -v pytest >/dev/null || die "pytest not found; install requirements first"
    # repo root and src/ importable, whatever pytest's rootdir/import mode decides (src layout: src/dronenav)
    (cd "$ROOT" && PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" pytest -q) \
      || die "tests are red: fix them (never weaken a test to pass)"
  else
    echo "task.sh: no tests in tests/ yet, nothing to run" >&2
  fi
}

require_clean() {
  git -C "$ROOT" diff --quiet && git -C "$ROOT" diff --cached --quiet \
    || die "uncommitted changes in tracked files; commit or stash first"
}

all_issues='query($owner:String!,$name:String!){repository(owner:$owner,name:$name){issues(states:OPEN,first:100){nodes{
  number title labels(first:10){nodes{name}} assignees(first:10){nodes{login}} blockedBy(first:50){nodes{number state}}}}}}'

case "${1:-}" in
  ready)
    m=$(me)
    gql "$all_issues" ".data.repository.issues.nodes[]
      | select(([.blockedBy.nodes[] | select(.state==\"OPEN\")] | length)==0)
      | select((.assignees.nodes | length)==0 or ([.assignees.nodes[].login] | index(\"$m\")))
      | select(([.labels.nodes[].name] | index(\"owner-only\") | not) or \"$m\" == \"$TASK_OWNER\")
      | \"#\(.number)\t\(.title)\t\([.labels.nodes[].name | select(. != \"blocked\")] | join(\",\"))\t\(if (.assignees.nodes|length)>0 then \"claimed by you\" else \"\" end)\""
    ;;

  mine)
    m=$(me)
    gql "$all_issues" ".data.repository.issues.nodes[]
      | select([.assignees.nodes[].login] | index(\"$m\"))
      | \"#\(.number)\t\(.title)\t\([.blockedBy.nodes[] | select(.state==\"OPEN\") | \"#\(.number)\"] | if length>0 then \"blocked by \" + join(\",\") else \"ready\" end)\""
    ;;

  claim)
    n="${2:?usage: task.sh claim <issue#>}"
    [[ $(issue_q "$n" .state) == OPEN ]] || die "#$n is not open"
    check_owner_only "$n"
    taken=$(issue_q "$n" '[.assignees.nodes[].login] | join(",")')
    [[ -z "$taken" || "$taken" == "$(me)" ]] || die "#$n is already taken by $taken"
    gh issue edit "$n" -R "$REPO" --add-assignee @me >/dev/null
    blockers=$(issue_q "$n" '[.blockedBy.nodes[] | select(.state=="OPEN") | "#\(.number)"] | join(", ")')
    if [[ -n "$blockers" ]]; then echo "Claimed #$n. Blocked by $blockers: you'll be @mentioned on #$n when it unblocks."
    else echo "Claimed #$n. It is not blocked: tools/task.sh start $n"; fi
    ;;

  start)
    n="${2:?usage: task.sh start <issue#>}"
    [[ $(issue_q "$n" .state) == OPEN ]] || die "#$n is not open"
    check_owner_only "$n"
    m=$(me)
    taken=$(issue_q "$n" '[.assignees.nodes[].login] | join(",")')
    [[ -z "$taken" || "$taken" == "$m" ]] || die "#$n is already taken by $taken"
    blockers=$(issue_q "$n" '.blockedBy.nodes[] | select(.state=="OPEN") | "#\(.number) \(.title)"')
    [[ -z "$blockers" ]] || die "#$n is blocked by:"$'\n'"$blockers"$'\n'"Claim it instead: tools/task.sh claim $n"
    require_clean
    title=$(issue_q "$n" .title)
    id=$(grep -oE '^M[0-9]-[0-9]{2}' <<<"$title") || die "issue title has no task ID"
    slug=$(sed -E 's/^M[0-9]-[0-9]{2} · //; s/[^A-Za-z0-9]+/-/g; s/^-+|-+$//g' <<<"$title" \
           | tr '[:upper:]' '[:lower:]' | cut -d- -f1-4)
    branch="$id-$slug"
    git fetch -q origin
    if git show-ref --verify -q "refs/remotes/origin/$branch"; then
      git switch -q "$branch" 2>/dev/null || git switch -q -c "$branch" --track "origin/$branch"
      echo "Branch $branch already exists: resumed it."
    else
      git switch -q develop && git pull -q --ff-only origin develop
      git switch -q -c "$branch"
      git push -q -u origin "$branch"
    fi
    gh issue edit "$n" -R "$REPO" --add-assignee @me >/dev/null
    set_status "$n" "In Progress"
    echo "Started #$n on $branch. Next: failing tests for every 'Done when' item (TDD), then implement."
    echo "----- spec -----"
    issue_q "$n" .body
    ;;

  pr)
    n=$(current_issue)
    require_clean
    title=$(issue_q "$n" .title)
    git fetch -q origin
    git rebase -q origin/develop || die "rebase on develop has conflicts: resolve, 'git rebase --continue', rerun"
    run_tests                                         # on the rebased branch: that is what will land in develop
    git push -q --force-with-lease -u origin HEAD     # only ever our own task branch, after the rebase
    if [[ -n "${2:-}" ]]; then body=$(cat "$2")
    else body=$'## What changed\n\n## Tests: one per "Done when" item\n'; fi
    existing=$(gh pr list -R "$REPO" --head "$(git branch --show-current)" --state open --json url --jq '.[0].url')
    if [[ -n "$existing" ]]; then echo "PR already open, branch updated: $existing"
    else gh pr create -R "$REPO" --base develop --head "$(git branch --show-current)" --title "$title" \
           --body "$body"$'\n\n'"Closes #$n"; fi
    ;;

  finish)
    n=$(current_issue)
    branch=$(git branch --show-current)
    require_clean
    [[ -z "$(git log --oneline "origin/$branch..HEAD" 2>/dev/null)" ]] || die "local commits not pushed: run tools/task.sh pr"
    pr=$(gh pr list -R "$REPO" --head "$branch" --state open --json number,baseRefName --jq '.[0] | "\(.number) \(.baseRefName)"')
    [[ -n "$pr" && "$pr" != "null null" ]] || die "no open PR for $branch: run tools/task.sh pr"
    [[ "${pr#* }" == develop ]] || die "PR #${pr% *} targets ${pr#* }, not develop"
    run_tests
    checks=$(gh pr checks "${pr% *}" -R "$REPO" --json bucket --jq '[.[] | select(.bucket=="fail" or .bucket=="pending")] | length' 2>/dev/null || echo 0)
    [[ "$checks" == 0 ]] || die "PR #${pr% *} has failing or pending checks: gh pr checks ${pr% *}"
    gh pr merge "${pr% *}" -R "$REPO" --squash --delete-branch
    set_status "$n" "Done"
    git switch -q develop 2>/dev/null || true
    git pull -q --ff-only origin develop
    git branch -D "$branch" >/dev/null 2>&1 || true
    echo "Merged #$n into develop, board → Done. Back on fresh develop."
    echo "Tasks this unblocked are announced on their issues by the 'unblock' workflow."
    ;;

  test)
    shift; [[ $# -eq 0 ]] && { run_tests; exit; }
    cd "$ROOT" && PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}" exec pytest "$@"
    ;;

  status)
    set_status "${2:?usage: task.sh status <issue#> <status>}" "${3:?status}"
    ;;

  drop)
    n=$(current_issue)
    require_clean
    set_status "$n" "Todo"
    gh issue edit "$n" -R "$REPO" --remove-assignee @me >/dev/null
    git switch -q develop
    echo "Dropped #$n (its branch stays on GitHub). Say why in the issue: gh issue comment $n"
    ;;

  notify)
    # Called by the hooks in .claude/settings.json, so every teammate's Claude hears about tasks the `unblock`
    # workflow has just opened for them (claimed by them, or unclaimed). Prints nothing when there is no news, never
    # fails (a hook must not break the session), and remembers what was already reported in the local git dir.
    [[ "${2:-}" == --throttle ]] && throttle="${3:-600}" || throttle=0
    state="$(git rev-parse --git-dir)/task-notify"
    now=$(date +%s)
    last_check=$(cat "$state.checked" 2>/dev/null || echo 0)
    (( now - last_check >= throttle )) || exit 0
    echo "$now" >"$state.checked"
    # first run: report the last 3 days, so a fresh clone still hears about recent news
    since=$(cat "$state.since" 2>/dev/null || date -u -v-3d +%Y-%m-%dT%H:%M:%SZ 2>/dev/null \
            || date -u -d '3 days ago' +%Y-%m-%dT%H:%M:%SZ)
    stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    m=$(me 2>/dev/null) || exit 0
    news=$(gql 'query($owner:String!,$name:String!){repository(owner:$owner,name:$name){issues(states:OPEN,first:100){nodes{
        number title labels(first:10){nodes{name}} assignees(first:10){nodes{login}}
        comments(last:10){nodes{author{login} createdAt body}}}}}}' \
      ".data.repository.issues.nodes[]
       | select([.comments.nodes[] | select(.author.login==\"github-actions\" and .createdAt > \"$since\"
                                           and (.body | startswith(\"🔓 **Unblocked.**\")))] | length > 0)
       | select(([.labels.nodes[].name] | index(\"owner-only\") | not) or \"$m\" == \"$TASK_OWNER\")
       | [.assignees.nodes[].login] as \$who
       | if (\$who | index(\"$m\")) then \"- #\(.number) \(.title): unblocked and claimed by you → tools/task.sh start \(.number)\"
         elif (\$who | length) == 0 then \"- #\(.number) \(.title): unblocked, nobody has claimed it → tools/task.sh start \(.number)\"
         else empty end" 2>/dev/null) || exit 0
    echo "$stamp" >"$state.since"
    if [[ -n "$news" ]]; then
      echo "🔓 Orientyr board news for @$m: tasks just unblocked (their blockers were merged)."
      echo "Tell the user about these first, before anything else, and offer to start one:"
      echo "$news"
    fi
    ;;

  *)
    sed -n '2,15p' "$0"; exit 1
    ;;
esac
