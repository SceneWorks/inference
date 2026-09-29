#!/usr/bin/env bash
# One W1 campaign phase for kv-poc-campaign.yml:  phase.sh run <a1|a3|a2|b>  |  phase.sh collect <phase>
#
# `run` = precheck (fails, never kills) -> the exact W1-COMMANDS.sh command for the phase -> exit.
# Exit 0 = phase complete (or already complete); exit 0 with output stopped=true = the parent
# honoured a stop request and exited 75 after its in-flight row; anything else = failure.
#
# SAFE STOP. A background watcher polls every 60 s for `refs/heads/kv-poc-stop/<run_id>` on the
# inference remote, and also enforces this job's soft budget ($KV_SOFT_BUDGET_MIN). Either one
# touches <resume-dir>/STOP, and the campaign parent exits 75 after the row it is running; it never
# interrupts a row. B has no resume dir, so it checks the same request between its commands.
# SIGINT/SIGTERM also touch STOP, but that is best effort only: a GitHub cancel (or a job timeout)
# force-kills the process tree about 10 s later, mid-row, which can wedge the GPU. Cancel between
# jobs only; to stop mid-job push the stop branch.
set -uo pipefail
# shellcheck source=.github/kv-poc/common.sh
source "$(dirname "$0")/common.sh"

action="${1:?usage: phase.sh run|collect <phase>}"
phase="${2:?usage: phase.sh run|collect <phase>}"

RESUME=""
OUT=""
case "$phase" in
  a1) RESUME="$R/sc20671-dense-resume"; OUT="$R/evidence/sc20671-dense" ;;
  a3) RESUME="$R/sc20676-resume"; OUT="$R/evidence/sc20676-packed" ;;
  a2) RESUME="$R/sc20671-compressed-resume"; OUT="$R/evidence/sc20671-compressed" ;;
  b) ;;
  *) echo "unknown phase $phase" >&2; exit 2 ;;
esac
LOG_DIR="$R/logs"
LOG="$LOG_DIR/$phase-run${GITHUB_RUN_ID:-local}-attempt${GITHUB_RUN_ATTEMPT:-1}.log"
CTL="${RUNNER_TEMP:-/tmp}/kv-poc-ctl-$phase"
ART="${RUNNER_TEMP:-/tmp}/kv-poc-artifact/$phase"

collect() {
  mkdir -p "$ART"
  local src sources="$LOG_DIR"
  if [ "$phase" = b ]; then
    for n in llama qwen; do
      sources="$sources $R/evidence/sc20677-kv-$n $R/evidence/sc20677-kv-$n.partial $R/evidence/sc20677-comparison-$n.json"
    done
  else
    sources="$sources $RESUME $OUT"
  fi
  for src in $sources; do
    [ -e "$src" ] || continue
    python3.12 "$KV_DIR/summarize.py" copy --src "$src" --dest "$ART/$(basename "$src")" --max-bytes 1073741824
  done
  python3.12 "$KV_DIR/summarize.py" report --phase "$phase" --root "$R" \
    ${RESUME:+--dir "$RESUME"} ${OUT:+--dir "$OUT"} >> "${GITHUB_STEP_SUMMARY:-/dev/stdout}" || true
}

if [ "$action" = collect ]; then collect; exit 0; fi
[ "$action" = run ] || { echo "unknown action $action" >&2; exit 2; }

finish() { # <exit_code> <stopped true|false> <message>
  output exit_code "$1"
  output stopped "$2"
  summary "### Phase $phase: $3 (exit $1)"
  summary ""
  summary "Stop command for this run: \`$(stop_command)\`"
  echo "phase $phase: $3 (exit $1)"
}

mkdir -p "$CTL" "$LOG_DIR" "$R/evidence"
rm -f "$CTL/stop-requested"

# 0. An operator stop pushed before this job started means: do not start.
if stop_branch_present; then
  finish 0 true "not started: $(stop_ref) exists"
  exit 0
fi

# 1. Already complete? A campaign --out appears only after every row is accepted, and the parent
#    refuses an existing --out, so rerunning a finished phase is a skip, not an error.
if [ -n "$OUT" ] && [ -e "$OUT" ]; then
  finish 0 false "already complete ($OUT exists); skipped"
  exit 0
fi
if [ "$phase" = b ] && [ -e "$R/evidence/sc20677-comparison-llama.json" ] && [ -e "$R/evidence/sc20677-comparison-qwen.json" ]; then
  finish 0 false "already complete (both comparisons exist); skipped"
  exit 0
fi

# 2. Precheck. Fails the job with the reason; never stops or kills anything.
problems=""
gib="$(free_spec_gib)"
echo "free+speculative RAM: ${gib} GiB (need >= 84 = 68 cap + 16 reserve)"
[ "$gib" -ge 84 ] || problems="$problems; free+speculative RAM is ${gib} GiB (< 84)"
lms="$(lms_bin)"
if [ -n "$lms" ]; then
  lms_out="$("$lms" ps --json 2>&1 || true)"
  echo "lms ps --json: $lms_out"
  [ "$(printf '%s' "$lms_out" | tr -d '[:space:]')" = "[]" ] || problems="$problems; LM Studio has a loaded model (or lms failed): $lms_out"
else
  echo "lms: not installed"
fi
busy="$(busy_processes)"
[ -z "$busy" ] || problems="$problems; other MLX/cargo processes are running: $(printf '%s' "$busy" | tr '\n' ' ')"
verify_tree "$INF" "$INFERENCE_URL" "$INFERENCE_SHA" || problems="$problems; inference tree is not clean at $INFERENCE_SHA"
verify_tree "$SW" "$SCENEWORKS_URL" "$SCENEWORKS_SHA" || problems="$problems; SceneWorks tree is not clean at $SCENEWORKS_SHA"
verify_frozen || problems="$problems; frozen dir does not verify"
python3.12 "$KV_DIR/models.py" --check-only --hub "$KV_HF_HUB" --pins "$KV_DIR/models.tsv" \
  || problems="$problems; pinned snapshots are missing or wrong"
if [ "$phase" = a3 ] && [ ! -d "$R/evidence/sc20671-dense" ]; then
  problems="$problems; A3 needs the completed A1 evidence $R/evidence/sc20671-dense"
fi
if [ -n "$problems" ]; then
  echo "::error title=precheck NO-GO ($phase)::${problems#; }"
  finish 1 false "precheck NO-GO: ${problems#; }"
  exit 1
fi
echo "precheck: GO"

export PMETAL_METALLIB_PATH="$F/mlx.metallib"
export SCENEWORKS_ROOT="$SW"

# 3. Stop plumbing.
if [ -n "$RESUME" ] && [ -e "$RESUME/STOP" ]; then
  echo "removing the stale stop file $RESUME/STOP left by an earlier run (this dispatch resumes)"
  rm -f "$RESUME/STOP"
fi
request_stop() {
  [ -f "$CTL/stop-requested" ] || { echo "$1" > "$CTL/stop-requested"; echo "::warning title=stop requested::$1; the parent exits 75 after its in-flight row"; }
  if [ -n "$RESUME" ] && [ -d "$RESUME" ] && [ ! -e "$RESUME/STOP" ]; then touch "$RESUME/STOP"; echo "touched $RESUME/STOP"; fi
}
started="$(date +%s)"
budget="${KV_SOFT_BUDGET_MIN:-0}"
(
  while :; do
    sleep 60
    if [ ! -f "$CTL/stop-requested" ]; then
      if stop_branch_present; then
        request_stop "stop branch $(stop_ref) present"
      elif [ "$budget" -gt 0 ] && [ $(( ($(date +%s) - started) / 60 )) -ge "$budget" ]; then
        request_stop "soft budget of ${budget} min reached"
      fi
    else
      request_stop "$(cat "$CTL/stop-requested")" # the resume dir may appear after the request
    fi
  done
) &
watcher=$!
trap 'kill "$watcher" 2>/dev/null' EXIT
trap 'request_stop "signal received (best effort: a cancel force-kills this job in ~10 s)"' INT TERM

stop_requested() { [ -f "$CTL/stop-requested" ]; }

# Run one command in the background (async children of a non-interactive shell ignore SIGINT, so a
# stray cancel signal reaches this script, not the MLX parent) and wait through trapped signals.
run_cmd() {
  echo "+ $*" | tee -a "$LOG"
  ( set -o pipefail; "$@" 2>&1 | tee -a "$LOG" ) &
  local pid=$! rc
  while :; do
    wait "$pid"; rc=$?
    kill -0 "$pid" 2>/dev/null || break
  done
  echo "exit $rc: $1" | tee -a "$LOG"
  return "$rc"
}

LLM_ARGS_TEXT="--llama-snapshot $LQ --qwen-snapshot $QQ --llama-fp32-reference-snapshot $LB --qwen-fp32-reference-snapshot $QB --prompt-file $F/inputs/prompt.txt --safety-policy $F/policies/llm.json"
# shellcheck disable=SC2206  # every element is a space-free absolute path or flag
LLM_ARGS=($LLM_ARGS_TEXT)

rc=0
case "$phase" in
  a1)
    run_cmd "$F/sc20671_kv_baseline" parent "${LLM_ARGS[@]}" \
      --resume-dir "$RESUME" --out "$OUT" || rc=$?
    ;;
  a3)
    run_cmd "$F/sc20676_packed_evidence" parent --llama-snapshot "$LQ" --qwen-snapshot "$QQ" \
      --llama-baseline-campaign "$R/evidence/sc20671-dense" --qwen-baseline-campaign "$R/evidence/sc20671-dense" \
      --safety-policy "$F/policies/llm.json" --resume-dir "$RESUME" --out "$OUT" || rc=$?
    ;;
  a2)
    run_cmd "$F/sc20671_kv_baseline" parent --mode compressed --kv-method group-affine "${LLM_ARGS[@]}" \
      --resume-dir "$RESUME" --out "$OUT" || rc=$?
    ;;
  b)
    for fam in "llama:$LQ" "qwen:$QQ"; do
      n="${fam%%:*}"
      kvdir="$R/evidence/sc20677-kv-$n"
      cmp="$R/evidence/sc20677-comparison-$n.json"
      if [ -e "$cmp" ]; then echo "$n: comparison exists; skipped"; continue; fi
      if stop_requested; then rc=75; break; fi
      if [ -e "$kvdir.partial" ]; then
        echo "::error title=partial capture::$kvdir.partial exists from an earlier failed capture; inspect it and remove it by hand, then re-dispatch"
        rc=1; break
      fi
      if [ ! -e "$kvdir" ]; then
        run_cmd "$F/sc20677_capture_kv" parent --snapshot "${fam#*:}" --prompt-file "$F/inputs/prompt.txt" --tokens 8192 \
          --layers 0,mid,last --safety-policy "$F/policies/capture.json" --out "$kvdir" || { rc=$?; break; }
      fi
      if stop_requested; then rc=75; break; fi
      kv=""
      for f in "$kvdir"/*.safetensors; do [ -e "$f" ] && kv="$kv --kv $f"; done
      [ -n "$kv" ] || { echo "::error title=empty capture::$kvdir holds no .safetensors"; rc=1; break; }
      # shellcheck disable=SC2086  # deliberate word splitting of the space-free --kv list
      run_cmd "$F/sc20677_kv_candidates" $kv --out "$cmp" || { rc=$?; break; }
    done
    ;;
esac

case "$rc" in
  0) finish 0 false "completed" ;;
  75)
    reason="$(cat "$CTL/stop-requested" 2>/dev/null || echo "stop file present")"
    echo "::notice title=phase $phase stopped::$reason; re-dispatch the same parameters to resume"
    finish 75 true "stopped by operator request ($reason); later phases skip; re-dispatch to resume"
    rc=0
    ;;
  *) finish "$rc" false "FAILED" ;;
esac
exit "$rc"
