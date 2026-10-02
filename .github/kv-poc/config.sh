#!/usr/bin/env bash
# Resolve the campaign parameters for kv-poc-campaign.yml (runs on the hosted `config` job).
#
# workflow_dispatch: the dispatch inputs (DISPATCH_* env).
# push to kv-poc-run/**: .github/kv-poc-run.json at the pushed commit, with the same keys.
# baseline_evidence_ref (optional, default inference_ref): the inference SHA whose completed A1
# evidence ($HOME/kv-poc/<sha>-runs/evidence/sc20671-dense) A3 binds. sc20676 itself refuses it
# unless that SHA is an ancestor of inference_ref with an unchanged SC-20671 dense closure.
# a3_bits (optional, default "2"): comma list of A3 `--kv-bits` values (2, 4, 8), run in list order.
# a2_methods (optional, default "group-affine"): comma list of A2 `--kv-method` values
# (group-affine, group-affine-4, group-affine-8), run in list order. At most 2 values each: the
# a3/a2 jobs' hard timeouts are sized for two sequential invocations (kv-poc-campaign.yml TIMEOUTS).
# a2_only_coordinate (optional, default ""): run A2 as `--only-coordinate <name>`, one of the eight
# scheduled SC-20671 coordinates. The run publishes a partial, non-publishable manifest (never a
# campaign) into its own `-only-<name>` resume + evidence dirs, so it never touches a full A2.
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
  a2_only_coordinate="$(read_key a2_only_coordinate "")"
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
  a2_only_coordinate="${DISPATCH_A2_ONLY_COORDINATE:-}"
  source_desc="workflow_dispatch inputs"
fi

fail() { echo "::error title=bad campaign parameter::$1"; exit 1; }
case "$mode" in probe|w1|w2) ;; *) fail "mode must be probe, w1 or w2, got '$mode'" ;; esac
[[ "$inference_sha" =~ ^[0-9a-f]{40}$ ]] || fail "inference_ref must be a full 40-hex commit id, got '$inference_sha'"
[[ "$sceneworks_sha" =~ ^[0-9a-f]{40}$ ]] || fail "sceneworks_ref must be a full 40-hex commit id, got '$sceneworks_sha'"
[ -n "$baseline_sha" ] || baseline_sha="$inference_sha"
[[ "$baseline_sha" =~ ^[0-9a-f]{40}$ ]] || fail "baseline_evidence_ref must be empty or a full 40-hex commit id, got '$baseline_sha'"
# A closed set naming a HOST, each mapped to the runner it must land on and to a label set only
# that runner carries. Every campaign job reuses state the previous job left under that host's
# $HOME/kv-poc, so all of a run's macOS jobs must land on the SAME Mac. The `nax` label is
# registered on BOTH Macs (run 36816509385: its probe + w2-assets ran on nax-macos-2, which
# cloned /Users/MTrefry/kv-poc/<sha>, and its w2-build on nax-macos, where /Users/michael/kv-poc/<sha>
# never existed), so it cannot select the dev Mac by itself; `rw-starvector` is the label only
# nax-macos carries (every rw-starvector job since 2026-08-31 ran there), as `rw-krea` is
# nax-macos-2's. Labels are operator-movable, so each job also asserts its runner name
# (common.sh KV_EXPECTED_RUNNER) and fails before touching state if the pool ever drifts.
case "$label" in
  rw-krea) host_labels=rw-krea; runner_name=nax-macos-2 ;;
  nax) host_labels="nax rw-starvector"; runner_name=nax-macos ;;
  *) fail "runner_label must be rw-krea or nax, got '$label'" ;;
esac

# Each mode owns a fixed phase order; an empty list means the mode's default phases (every W2
# phase; W1's a1,a3,a2,b). W1's `nf` (the SC-20669 dense noise floor, ~30 min) runs only when
# listed, e.g. "phases": "nf" alone. W2 also takes `none`: run the asset prep and the build only
# (e.g. to see a host's disk shortfall first).
case "$mode" in
  w2) order="c d d-control"; default_order="$order" ;;
  *) order="a1 a3 a2 b nf"; default_order="a1 a3 a2 b" ;;
esac
phases="${phases// /}"
[ -n "$phases" ] || phases="${default_order// /,}"
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
# Run order is fixed (A1 -> A3 -> A2 -> B -> NF; C -> D -> D-control) whatever order the list was typed in.
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
value_list a3_bits "$a3_bits" 2 4 8; a3_bits="$listed"
value_list a2_methods "$a2_methods" group-affine group-affine-4 group-affine-8; a2_methods="$listed"
# The frozen SC-20671 schedule (campaign.rs required_coordinates); sc20671 refuses any other name.
a2_only_coordinate="${a2_only_coordinate// /}"
case " $a2_only_coordinate " in
  "  "|" llama-short-single-chunked-cold "|" llama-medium-supported-batch-single-shot-warm "\
  |" llama-memory-material-single-single-shot-warm "|" llama-fit-boundary-single-chunked-cold "\
  |" qwen-short-single-single-shot-cold "|" qwen-medium-supported-batch-chunked-warm "\
  |" qwen-memory-material-single-single-shot-warm "|" qwen-fit-boundary-single-chunked-cold ") ;;
  *) fail "a2_only_coordinate must be empty or one scheduled SC-20671 coordinate, got '$a2_only_coordinate'" ;;
esac

runs_on="$(jq -cn --arg l "$host_labels" '["self-hosted","macOS","ARM64"] + ($l | split(" "))')"
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
  echo "a2_only_coordinate=$a2_only_coordinate"
  echo "runs_on=$runs_on"
  echo "runner_name=$runner_name"
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
  echo "| runner label | \`$label\` (runs-on \`$host_labels\`, must land on \`$runner_name\`) |"
  echo "| phases | \`${canonical#,}\` |"
  echo "| A3 --kv-bits | \`$a3_bits\` |"
  echo "| A2 --kv-method | \`$a2_methods\` |"
  echo "| A2 --only-coordinate | ${a2_only_coordinate:+\`$a2_only_coordinate\` (PARTIAL, non-publishable)}${a2_only_coordinate:-all eight rows} |"
} >> "$GITHUB_STEP_SUMMARY"
cat "$GITHUB_STEP_SUMMARY"
