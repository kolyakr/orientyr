#!/usr/bin/env bash
# Team data on Google Drive <-> local data/ (gitignored). The Drive folder is the one source of the data;
# git never holds it. Uses rclone (brew install rclone / https://rclone.org/install/). macOS bash 3.2 OK.
#
#   tools/data.sh setup            one-time: create the rclone remote (opens the browser to log in to Google)
#   tools/data.sh ls [path]        list the Drive folder (or a subfolder, e.g. irpin/flights)
#   tools/data.sh pull [path]      Drive -> data/[path]: copies new and changed files, never deletes local ones
#   tools/data.sh check [path]     compare data/[path] with Drive (sizes); prints what differs
#   tools/data.sh push <path> [--yes]   data/<path> -> Drive: dry run first; uploads only with --yes;
#                                  never deletes on Drive; a path is required (no accidental full upload)
#
# Folder: https://drive.google.com/drive/folders/1Z5iIhFekHIFkAjH49jxUdf3k4SGw7voe
# Its layout mirrors data/: hackatons-data/ (organisers' map + crops), irpin/ (maps, hires, flights, ...).
# Override with ORIENTYR_REMOTE (rclone remote name) and ORIENTYR_DRIVE_ID (folder id).
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
REMOTE="${ORIENTYR_REMOTE:-orientyr}"
FOLDER_ID="${ORIENTYR_DRIVE_ID:-1Z5iIhFekHIFkAjH49jxUdf3k4SGw7voe}"
DATA="$ROOT/data"

die() { echo "data.sh: $*" >&2; exit 1; }
command -v rclone >/dev/null || die "rclone not found: brew install rclone (or https://rclone.org/install/)"

# The remote is rooted at the shared folder, so paths below are relative to it, like data/.
rc() { rclone "$@" --drive-root-folder-id "$FOLDER_ID" --progress --transfers 8; }
need_remote() {
  rclone listremotes | grep -qx "$REMOTE:" || die "no rclone remote '$REMOTE': run tools/data.sh setup"
}
clean() { local p="${1:-}"; p="${p#data/}"; p="${p%/}"; case "$p" in *..*) die "bad path '$1'";; esac; echo "$p"; }

cmd="${1:-}"; shift || true
case "$cmd" in
  setup)
    if rclone listremotes | grep -qx "$REMOTE:"; then echo "remote '$REMOTE' already exists"
    else rclone config create "$REMOTE" drive scope=drive; fi
    rclone lsf "$REMOTE:" --drive-root-folder-id "$FOLDER_ID" --max-depth 1
    ;;
  ls)
    need_remote; p=$(clean "${1:-}")
    rclone lsf "$REMOTE:$p" --drive-root-folder-id "$FOLDER_ID" --max-depth 1 --format "sp" --separator "  "
    ;;
  pull)
    need_remote; p=$(clean "${1:-}"); mkdir -p "$DATA/$p"
    rc copy "$REMOTE:$p" "$DATA/$p" --exclude ".DS_Store"
    ;;
  check)
    need_remote; p=$(clean "${1:-}")
    rclone check "$DATA/$p" "$REMOTE:$p" --drive-root-folder-id "$FOLDER_ID" --size-only --exclude ".DS_Store"
    ;;
  push)
    need_remote; [ -n "${1:-}" ] || die "push needs a path under data/, e.g. tools/data.sh push irpin/flights/harsh"
    p=$(clean "$1"); [ -n "$p" ] || die "refusing to push all of data/: name a subfolder"
    [ -e "$DATA/$p" ] || die "data/$p does not exist"
    if [ "${2:-}" = "--yes" ]; then rc copy "$DATA/$p" "$REMOTE:$p" --exclude ".DS_Store"
    else
      rclone copy "$DATA/$p" "$REMOTE:$p" --drive-root-folder-id "$FOLDER_ID" --exclude ".DS_Store" --dry-run
      echo "dry run only: rerun with --yes to upload"
    fi
    ;;
  *) sed -n '2,14p' "$0"; [ -z "$cmd" ] || exit 1 ;;
esac
