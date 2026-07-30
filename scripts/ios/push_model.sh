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
# THE RULE, verified against the device both ways: `--destination` is the full path the source
# BECOMES. It is not a containing directory.
#
#   --source inner/ --destination Documents/dirprobe    -> Documents/dirprobe/probe.txt
#   --source f.txt  --destination Documents/f.txt       -> Documents/f.txt
#   --source f.txt  --destination Documents             -> Documents  IS NOW THE FILE
#
# That last line is not hypothetical. Pushing five files with `--destination Documents` replaced
# the Documents DIRECTORY with a 738-byte tokenizer_config.json, wiping the container — and
# devicectl exited 0 every time. The corresponding directory mistake is just as quiet: SANA's
# `transformer/` and `vae/` both hold a file named `diffusion_pytorch_model.safetensors`, so
# pushing both to one destination silently overwrote the 1.99 GB trunk with the 1.25 GB decoder.
#
# One rule covers both: the destination always names the entry.
DEST="Documents${REMOTE:+/$REMOTE}"
say "destination $DEST/"
for entry in "$SRC"/*; do
  name=$(basename "$entry")
  target="$DEST/$name"
  # Capture, THEN inspect. Piping devicectl straight into grep makes the pipeline's status grep's,
  # discarding devicectl's own — which is how the first version reported five successful pushes
  # that had not happened. Scanning the text for "Error" is not enough either: the destination
  # mistake above exits 0 and prints a perfectly ordinary success block.
  if ! COPY_OUT=$(xcrun devicectl device copy to --device "$DEVICE" \
      --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
      --source "$entry" --destination "$target" 2>&1); then
    echo "copy failed for $name:" >&2; echo "$COPY_OUT" | tail -5 >&2; exit 1
  fi
  # devicectl echoes the resulting device path; require it to end in the entry we asked for. This
  # is what catches a silently-wrong destination, which no exit code will.
  if ! echo "$COPY_OUT" | grep -qE "^Path: .*/${name}$"; then
    echo "copy landed somewhere unexpected for $name (wanted a path ending in /$name):" >&2
    echo "$COPY_OUT" | grep -E '^Path:' >&2 || echo "  (devicectl printed no Path line)" >&2
    exit 1
  fi
  printf '  pushed %s -> %s\n' "$name" "$target"
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
  echo "$LISTING" | grep -qF -- "$name" || { echo "MISSING on device: $name" >&2; MISSING=1; continue; }
  [ -d "$entry" ] || continue

  # Descend. A top-level check alone is not enough: the flattening bug produced a `sana/` that
  # listed the right NAMES while the files inside were wrong (one component's weights overwritten
  # by another's, because both are called diffusion_pytorch_model.safetensors). Compare each
  # component's file names AND sizes against the local copy.
  SUB=$(xcrun devicectl device info files --device "$DEVICE" \
    --domain-type appDataContainer --domain-identifier "$BUNDLE_ID" \
    --subdirectory "$DEST/$name" 2>&1)
  echo "$SUB" | grep -qE '^[0-9]+ files?:' \
    || { echo "MISSING on device: $name/ could not be listed" >&2; MISSING=1; continue; }
  for f in "$entry"/*; do
    fname=$(basename "$f")
    # GiB, not GB. devicectl LABELS its size column "GB" and reports GiB: a 1988912542-byte file
    # lists as "1.85 GB" (1.99 decimal GB / 1.85 GiB). Comparing decimal GB against it flagged the
    # transformer as a mismatch when the push was fine — and, worse, let the other two through only
    # because their error happened to fall inside the tolerance. Match its units and the agreement
    # is exact to two decimals, so the tolerance can be tight enough to actually mean something.
    local_gib=$(ls -l "$f" | awk '{printf "%.2f", $5/1073741824}')
    line=$(echo "$SUB" | grep -F -- "$fname" || true)
    if [ -z "$line" ]; then
      echo "MISSING on device: $name/$fname" >&2; MISSING=1; continue
    fi
    dev_gib=$(echo "$line" | grep -oE '[0-9.]+ GB' | grep -oE '[0-9.]+' || echo "")
    if [ -n "$dev_gib" ]; then
      awk -v a="$local_gib" -v b="$dev_gib" 'BEGIN { exit !((a-b < -0.02) || (a-b > 0.02)) }' \
        && { echo "SIZE MISMATCH: $name/$fname local ${local_gib} GiB vs device ${dev_gib} GiB" >&2; MISSING=1; }
      printf '  ok %s/%s (%s GiB)\n' "$name" "$fname" "$dev_gib"
    else
      printf '  ok %s/%s\n' "$name" "$fname"
    fi
  done
done
[ "$MISSING" -eq 0 ] || { echo "push incomplete" >&2; exit 1; }
say "OK"
