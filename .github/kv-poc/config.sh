#!/usr/bin/env bash
# Resolve the campaign parameters for kv-poc-campaign.yml (runs on the hosted `config` job).
#
# workflow_dispatch: the dispatch inputs (DISPATCH_* env).
# push to kv-poc-run/**: .github/kv-poc-run.json at the pushed commit, with the same five keys.
# Either way every value is validated here, so a typo fails in seconds on a hosted runner instead
# of queueing a self-hosted job on a label no runner carries (which waits silently forever).
set -euo pipefail

if [ "$EVENT_NAME" = "push" ]; then
  file=.github/kv-poc-run.json
  [ -f "$file" ] || { echo "::error title=no run file::push runs read $file, which this commit does not carry"; exit 1; }
  read_key() { jq -er --arg k "$1" --arg d "$2" '.[$k] // $d | tostring' "$file"; }
  mode="$(read_key mode probe)"
  inference_sha="$(read_key inference_ref "")"
  sceneworks_sha="$(read_key sceneworks_ref "")"
  label="$(read_key runner_label rw-krea)"
  phases="$(read_key phases "")"
  source_desc="$file @ ${GITHUB_SHA}"
else
  mode="$DISPATCH_MODE"
  inference_sha="$DISPATCH_INFERENCE_REF"
  sceneworks_sha="$DISPATCH_SCENEWORKS_REF"
  label="$DISPATCH_RUNNER_LABEL"
  phases="$DISPATCH_PHASES"
  source_desc="workflow_dispatch inputs"
fi

fail() { echo "::error title=bad campaign parameter::$1"; exit 1; }
case "$mode" in probe|w1|w2) ;; *) fail "mode must be probe, w1 or w2, got '$mode'" ;; esac
[[ "$inference_sha" =~ ^[0-9a-f]{40}$ ]] || fail "inference_ref must be a full 40-hex commit id, got '$inference_sha'"
[[ "$sceneworks_sha" =~ ^[0-9a-f]{40}$ ]] || fail "sceneworks_ref must be a full 40-hex commit id, got '$sceneworks_sha'"
# A closed set: rw-krea is nax-macos-2 (the second Mac); nax is Michael's dev Mac.
case "$label" in rw-krea|nax) ;; *) fail "runner_label must be rw-krea or nax, got '$label'" ;; esac

# Each mode owns a fixed phase order; an empty list means all of that mode's phases. W2 also takes
# `none`: run the asset prep and the build only (e.g. to see a host's disk shortfall first).
case "$mode" in
  w2) order="c d d-control" ;;
  *) order="a1 a3 a2 b" ;;
esac
phases="${phases// /}"
[ -n "$phases" ] || phases="${order// /,}"
canonical=""
if [ "$mode" = w2 ] && [ "$phases" = none ]; then
  requested=()
else
  IFS=',' read -r -a requested <<< "$phases"
fi
for p in ${requested[@]+"${requested[@]}"}; do
  [ -n "$p" ] || continue
  case " $order " in *" $p "*) ;; *) fail "unknown phase '$p' for mode $mode (allowed: ${order// /,})" ;; esac
done
# Run order is fixed (A1 -> A3 -> A2 -> B; C -> D -> D-control) whatever order the list was typed in.
for p in $order; do
  for q in ${requested[@]+"${requested[@]}"}; do [ "$q" = "$p" ] && canonical="$canonical,$p" && break; done
done
[ -n "$canonical" ] || [ "$mode" != w1 ] || fail "mode w1 needs at least one phase"

runs_on="$(jq -cn --arg l "$label" '["self-hosted","macOS","ARM64",$l]')"
{
  echo "mode=$mode"
  echo "inference_sha=$inference_sha"
  echo "sceneworks_sha=$sceneworks_sha"
  echo "runner_label=$label"
  # Bounded by commas so `contains(phases, ',a1,')` can never match a prefix.
  echo "phases=${canonical},"
  echo "runs_on=$runs_on"
} >> "$GITHUB_OUTPUT"

{
  echo "## KV PoC campaign parameters"
  echo ""
  echo "| key | value |"
  echo "|---|---|"
  echo "| source | $source_desc |"
  echo "| mode | \`$mode\` |"
  echo "| inference | \`$inference_sha\` |"
  echo "| SceneWorks | \`$sceneworks_sha\` |"
  echo "| runner label | \`$label\` |"
  echo "| phases | \`${canonical#,}\` |"
} >> "$GITHUB_STEP_SUMMARY"
cat "$GITHUB_STEP_SUMMARY"
