#!/usr/bin/env bash
# Push a model snapshot into the smoke app's Documents container.
#
# Usage:  scripts/ios/push_model.sh <local-snapshot-dir> [<remote-subdir>] [--device <udid>]
#
#   scripts/ios/push_model.sh ~/models/ios-eval/Qwen3-4B-Instruct-2507-q4
#   scripts/ios/push_model.sh ~/models/ios-eval/Sana_q4_embedq4 sana
#
# With no <remote-subdir> the snapshot's CONTENTS land directly in Documents/, which is where
# `find_snapshot` looks for the LLM (a root `config.json`). With one, they land in
# Documents/<subdir>/, which is where `find_media_snapshot` looks for the SANA component tree.
# Both can coexist: SANA has no root config.json, so it is never mistaken for the LLM.
#
# Inference receives every model component as a caller-provisioned local path (epic-13657) — this
# script is the consumer side of that boundary, moving already-fetched weights onto the device. It
# downloads nothing.
set -euo pipefail

BUNDLE_ID=com.idkplay.SceneWorksSmoke

SRC=""
REMOTE=""
DEVICE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    -*) echo "unknown argument: $1" >&2; exit 2 ;;
    *)
      if [ -z "$SRC" ]; then SRC="$1"; else REMOTE="$1"; fi
      shift ;;
  esac
done
[ -n "$SRC" ] || { echo "usage: push_model.sh <local-snapshot-dir> [<remote-subdir>]" >&2; exit 2; }
[ -d "$SRC" ] || { echo "not a directory: $SRC" >&2; exit 1; }

say() { printf '\n=== %s\n' "$1"; }

if [ -z "$DEVICE" ]; then
  # By UUID shape, not column position -- see the same note in run_smoke.sh. Counting fields from
  # the end lands inside the Model column ("iPhone 17 Pro Max (iPhone18,2)") and yields "17".
  DEVICE=$(xcrun devicectl list devices 2>/dev/null \
    | awk '/available \(paired\)/ {print; exit}' \
    | grep -oE '[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}' \
    | head -1)
fi
[ -n "$DEVICE" ] || { echo "no paired device found" >&2; exit 1; }
say "device $DEVICE"

# --- resolve symlinks -------------------------------------------------------------------------
# A snapshot assembled by hand is often a tree of symlinks into other snapshots (sharing a 2 GB
# trunk between a dense and a quantized variant, say). `devicectl copy to` does not follow them: it
# pushes the LINK, which lands on the device as a dangling path and fails at load with a confusing
# "file not found" naming a directory that exists. Stage a dereferenced copy instead.
STAGE=""
if [ -n "$(find "$SRC" -type l -print -quit)" ]; then
  STAGE=$(mktemp -d)
  trap 'rm -rf "$STAGE"' EXIT
  say "resolving $(find "$SRC" -type l | wc -l | tr -d ' ') symlink(s) into a staging copy"
  cp -RL "$SRC" "$STAGE/payload"
  SRC="$STAGE/payload"
fi

BYTES=$(find "$SRC" -type f -exec ls -l {} \; | awk '{s+=$5} END {print s+0}')
say "pushing $(echo "$BYTES" | awk '{printf "%.2f GB", $1/1e9}') from $SRC"

# --- push -------------------------------------------------------------------------------------
# `copy to` with a directory source creates <destination>/<basename>. To land the CONTENTS at the
# Documents root (the LLM layout) each entry is pushed individually; to land them in a subdirectory
# the directory is pushed once and renamed by choosing the destination.
DEST="Documents${REMOTE:+/$REMOTE}"
say "destination $DEST/"
for entry in "$SRC"/*; do
  # NOT `|| true`. The first version swallowed every copy failure here and then reported success,
  # because the verification below could not distinguish "listing is empty" from "listing failed".
  # Three copies errored with "device not found" and the script still exited 0. Fail on the spot.
  if ! xcrun devicectl device copy to --device "$DEVICE" \
      --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
      --source "$entry" --destination "$DEST" 2>&1 \
      | grep -vE '^[0-9]{2}:[0-9]{2}:[0-9]{2}'; then
    : # grep exits 1 when it filtered every line, which is the quiet success case.
  fi
  printf '  pushed %s\n' "$(basename "$entry")"
done

# --- verify -----------------------------------------------------------------------------------
# A push that silently dropped a file produces a load failure on device that reads like a code bug.
# Compare the device-side listing against what was sent.
say "verifying"
LISTING=$(xcrun devicectl device info files --device "$DEVICE" \
  --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
  --subdirectory "$DEST" 2>&1)
# An empty or error listing is a FAILURE, not an empty success — the distinction the first version
# missed. `info files` prints an "N files:" header on success.
echo "$LISTING" | grep -qE '^[0-9]+ files?:' \
  || { echo "could not list $DEST on device:" >&2; echo "$LISTING" >&2; exit 1; }
echo "$LISTING" | tail -n +4

MISSING=0
for entry in "$SRC"/*; do
  name=$(basename "$entry")
  echo "$LISTING" | grep -qF -- "$name" || { echo "MISSING on device: $name" >&2; MISSING=1; }
done
[ "$MISSING" -eq 0 ] || { echo "push incomplete" >&2; exit 1; }
say "OK"
