#!/usr/bin/env bash
# Resolve the campaign parameters for kv-poc-campaign.yml (runs on the hosted `config` job).
#
# workflow_dispatch: the dispatch inputs (DISPATCH_* env).
# push to kv-poc-run/**: .github/kv-poc-run.json at the pushed commit, with the same keys.
# baseline_evidence_ref (optional, default inference_ref): the inference SHA whose completed A1
# evidence ($HOME/kv-poc/<sha>-runs/evidence/sc20671-dense) A3 binds. sc20676 itself refuses it
# unless that SHA is an ancestor of inference_ref with an unchanged SC-20671 dense closure.
# a3_bits (optional, default "2"): comma list of A3 `--kv-bits` values (2, 4), run in list order.
# a2_methods (optional, default "group-affine"): comma list of A2 `--kv-method` values
# (group-affine, group-affine-4), run in list order. At most 2 values each: the a3/a2 jobs' hard
# timeouts are sized for two sequential invocations (kv-poc-campaign.yml TIMEOUTS).
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
  baseline_sha="$(read_key baseline_evidence_ref "")"
  a3_bits="$(read_key a3_bits 2)"
  a2_methods="$(read_key a2_methods group-affine)"
  source_desc="$file @ ${GITHUB_SHA}"
else
  mode="$DISPATCH_MODE"
  inference_sha="$DISPATCH_INFERENCE_REF"
  sceneworks_sha="$DISPATCH_SCENEWORKS_REF"
  label="$DISPATCH_RUNNER_LABEL"
  phases="$DISPATCH_PHASES"
  baseline_sha="$DISPATCH_BASELINE_EVIDENCE_REF"
  a3_bits="${DISPATCH_A3_BITS:-2}"
  a2_methods="${DISPATCH_A2_METHODS:-group-affine}"
  source_desc="workflow_dispatch inputs"
fi

fail() { echo "::error title=bad campaign parameter::$1"; exit 1; }
case "$mode" in probe|w1|w2) ;; *) fail "mode must be probe, w1 or w2, got '$mode'" ;; esac
[[ "$inference_sha" =~ ^[0-9a-f]{40}$ ]] || fail "inference_ref must be a full 40-hex commit id, got '$inference_sha'"
[[ "$sceneworks_sha" =~ ^[0-9a-f]{40}$ ]] || fail "sceneworks_ref must be a full 40-hex commit id, got '$sceneworks_sha'"
[ -n "$baseline_sha" ] || baseline_sha="$inference_sha"
[[ "$baseline_sha" =~ ^[0-9a-f]{40}$ ]] || fail "baseline_evidence_ref must be empty or a full 40-hex commit id, got '$baseline_sha'"
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

# The A3 --kv-bits / A2 --kv-method lists: known values only (they also name the evidence dirs),
# no duplicates (two invocations would share one resume dir), 1..2 values (the hard timeouts).
value_list() { # <key> <list> <allowed values...>; sets $listed to the canonical comma list
  local key="$1" list="${2// /}" seen="" v vals
  shift 2
  [ -n "$list" ] || fail "$key must name at least one value (allowed: $*)"
  IFS=',' read -r -a vals <<< "$list"
  for v in "${vals[@]}"; do
    case " $* " in *" $v "*) ;; *) fail "$key: unknown value '$v' (allowed: $*)" ;; esac
    case ",$seen," in *",$v,"*) fail "$key: '$v' is listed twice" ;; esac
    seen="${seen:+$seen,}$v"
  done
  [ "${#vals[@]}" -le 2 ] || fail "$key takes at most 2 values (the job timeouts are sized for 2), got '$list'"
  listed="$seen"
}
# Not in $(...): `fail` must print its ::error line to the job log, not into a variable.
value_list a3_bits "$a3_bits" 2 4; a3_bits="$listed"
value_list a2_methods "$a2_methods" group-affine group-affine-4; a2_methods="$listed"

runs_on="$(jq -cn --arg l "$label" '["self-hosted","macOS","ARM64",$l]')"
{
  echo "mode=$mode"
  echo "inference_sha=$inference_sha"
  echo "sceneworks_sha=$sceneworks_sha"
  echo "baseline_evidence_sha=$baseline_sha"
  echo "runner_label=$label"
  # Bounded by commas so `contains(phases, ',a1,')` can never match a prefix.
  echo "phases=${canonical},"
  echo "a3_bits=$a3_bits"
  echo "a2_methods=$a2_methods"
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
  echo "| A3 dense baseline evidence | \`$baseline_sha\` |"
  echo "| runner label | \`$label\` |"
  echo "| phases | \`${canonical#,}\` |"
  echo "| A3 --kv-bits | \`$a3_bits\` |"
  echo "| A2 --kv-method | \`$a2_methods\` |"
} >> "$GITHUB_STEP_SUMMARY"
cat "$GITHUB_STEP_SUMMARY"
