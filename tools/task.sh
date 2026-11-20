#!/usr/bin/env bash
# Task lifecycle for Orientyr: one command per step, so humans and agents follow the same flow.
# Needs: gh logged in with the `project` scope (`gh auth refresh -s project`). Works on macOS bash 3.2.
#
#   tools/task.sh ready              open, unassigned tasks whose blockers are all closed
#   tools/task.sh mine               my open tasks
#   tools/task.sh start <issue#>     assign me, board → In Progress, branch M1-04-<slug> off fresh develop
#   tools/task.sh pr [body-file]     rebase on develop, push, open the PR into develop ("Closes #N")
#   tools/task.sh finish             squash-merge my PR into develop, board → Done, back on fresh develop
#   tools/task.sh status <issue#> <Todo|In Progress|Done>
set -euo pipefail

REPO="kolyakr/orientyr"
PROJECT_ID="PVT_kwHOCW-k8M4Bseh_"            # board: https://github.com/users/kolyakr/projects/1
STATUS_FIELD="PVTSSF_lAHOCW-k8M4Bseh_zhqshhs"

die() { echo "task.sh: $*" >&2; exit 1; }

status_option() {
  case "$1" in
    "Todo") echo f75ad846 ;; "In Progress") echo 47fc9ee4 ;; "Done") echo 98236657 ;;
    *) die "unknown status '$1' (Todo | In Progress | Done)" ;;
  esac
}

issue_q() {  # issue_q <issue#> <jq filter applied to the issue>
  gh api graphql -F n="$1" -f query='query($n:Int!){repository(owner:"kolyakr",name:"orientyr"){issue(number:$n){
    number title state body assignees(first:5){nodes{login}} blockedBy(first:30){nodes{number title state}}
    projectItems(first:5){nodes{id project{id}}}}}}' --jq ".data.repository.issue | $2"
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
  local branch id
  branch=$(git branch --show-current)
  id=$(grep -oE '^M[0-9]-[0-9]{2}' <<<"$branch") || die "branch '$branch' is not a task branch (M1-04-…)"
  gh issue list -R "$REPO" --state all --limit 200 --json number,title \
    --jq ".[] | select(.title | startswith(\"$id \")) | .number" | head -1
}

case "${1:-}" in
  ready)
    gh api graphql -f query='{repository(owner:"kolyakr",name:"orientyr"){issues(states:OPEN,first:100){nodes{
      number title labels(first:5){nodes{name}} assignees(first:1){totalCount} blockedBy(first:30){nodes{state}}}}}}' \
      --jq '.data.repository.issues.nodes[]
            | select(.assignees.totalCount==0 and ([.blockedBy.nodes[] | select(.state=="OPEN")] | length)==0)
            | "#\(.number)\t\(.title)\t\([.labels.nodes[].name] | join(","))"'
    ;;

  mine)
    gh issue list -R "$REPO" --assignee @me --state open
    ;;

  start)
    n="${2:?usage: task.sh start <issue#>}"
    [[ $(issue_q "$n" .state) == OPEN ]] || die "#$n is not open"
    taken=$(issue_q "$n" '[.assignees.nodes[].login] | join(",")')
    me=$(gh api user --jq .login)
    [[ -z "$taken" || "$taken" == "$me" ]] || die "#$n is already taken by $taken"
    blockers=$(issue_q "$n" '.blockedBy.nodes[] | select(.state=="OPEN") | "#\(.number) \(.title)"')
    [[ -z "$blockers" ]] || die "#$n is blocked by:"$'\n'"$blockers"
    git diff --quiet && git diff --cached --quiet || die "uncommitted changes in tracked files; commit or stash first"
    title=$(issue_q "$n" .title)
    id=$(grep -oE '^M[0-9]-[0-9]{2}' <<<"$title") || die "issue title has no task ID"
    slug=$(sed -E 's/^M[0-9]-[0-9]{2} · //; s/[^A-Za-z0-9]+/-/g; s/^-+|-+$//g' <<<"$title" \
           | tr '[:upper:]' '[:lower:]' | cut -d- -f1-4)
    branch="$id-$slug"
    gh issue edit "$n" -R "$REPO" --add-assignee @me >/dev/null
    set_status "$n" "In Progress"
    git fetch -q origin
    git switch -q develop && git pull -q --ff-only origin develop
    git switch -c "$branch"
    git push -q -u origin "$branch"
    echo "Started #$n on $branch. Next: failing tests for every 'Done when' item (TDD), then implement."
    echo "----- spec -----"
    issue_q "$n" .body
    ;;

  pr)
    n=$(current_issue); [[ -n "$n" ]] || die "no issue found for this branch"
    title=$(issue_q "$n" .title)
    git fetch -q origin
    git rebase -q origin/develop || die "rebase on develop has conflicts: resolve, 'git rebase --continue', rerun"
    git push -q --force-with-lease -u origin HEAD   # only ever our own task branch, after the rebase
    if [[ -n "${2:-}" ]]; then body=$(cat "$2")
    else body=$'## What changed\n\n## Tests: one per "Done when" item\n'; fi
    gh pr create -R "$REPO" --base develop --head "$(git branch --show-current)" --title "$title" \
      --body "$body"$'\n\n'"Closes #$n"
    ;;

  finish)
    n=$(current_issue); [[ -n "$n" ]] || die "no issue found for this branch"
    gh pr merge -R "$REPO" --squash --delete-branch
    set_status "$n" "Done"
    git switch -q develop && git pull -q --ff-only origin develop
    echo "Merged #$n into develop, board → Done. Back on fresh develop."
    ;;

  status)
    set_status "${2:?usage: task.sh status <issue#> <status>}" "${3:?status}"
    ;;

  *)
    sed -n '2,10p' "$0"; exit 1
    ;;
esac
