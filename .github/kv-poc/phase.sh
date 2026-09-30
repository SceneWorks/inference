#!/usr/bin/env bash
# One campaign phase for kv-poc-campaign.yml:  phase.sh run <phase>  |  phase.sh collect <phase>
#   W1 (LLM KV):  a1 | a3 | a2 | b
#   W2 (media):   c  = SC-20684 Krea Realtime six-cell (T2V/I2V/V2V x Q8/Q4 KV), sc20684_krea_realtime_campaign.py
#                 d  = SC-20686 Metal matrix (18 coordinates x normal/cancel), sc20686_campaign_adapter.py
#                 d-control = the same matrix's --schedule-control arm (one product-schedule arm each)
#
# `run` = precheck (fails, never kills) -> the exact launch command for the phase -> exit.
# Exit 0 = phase complete (or already complete); exit 0 with output stopped=true = the parent
# honoured a stop request and exited 75 after its in-flight row; anything else = failure.
#
# SAFE STOP. A background watcher polls every 60 s for `refs/heads/kv-poc-stop/<run_id>` on the
# inference remote, and also enforces this job's soft budget ($KV_SOFT_BUDGET_MIN). Either one
# touches <resume-dir>/STOP, and the campaign parent exits 75 after the row it is running; it never
# interrupts a row. B has no resume dir, so it checks the same request between its commands. The W2
# parents also get `--stop-file <ctl>/stop-requested` (never part of a resume identity), so a request
# lands even before their resume dir exists.
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
# A3's SC-20671 dense baseline: this run's A1, or the A1 of an ancestor inference revision
# (BASELINE_EVIDENCE_SHA) that sc20676 accepts only with an unchanged dense closure.
BASELINE_SHA="${BASELINE_EVIDENCE_SHA:-$INFERENCE_SHA}"
DENSE_BASELINE="$KV_ROOT/$BASELINE_SHA-runs/evidence/sc20671-dense"
case "$phase" in
  a1) RESUME="$R/sc20671-dense-resume"; OUT="$R/evidence/sc20671-dense" ;;
  a3) RESUME="$R/sc20676-resume"; OUT="$R/evidence/sc20676-packed" ;;
  a2) RESUME="$R/sc20671-compressed-resume"; OUT="$R/evidence/sc20671-compressed" ;;
  b) ;;
  c) RESUME="$R/sc20684-resume"; OUT="$R/evidence/sc20684-krea" ;;
  d) RESUME="$R/sc20686-mlx-resume"; OUT="$R/evidence/sc20686-mlx" ;;
  d-control) RESUME="$R/sc20686-mlx-control-resume"; OUT="$R/evidence/sc20686-mlx-control" ;;
  *) echo "unknown phase $phase" >&2; exit 2 ;;
esac
W2=0
case "$phase" in c|d|d-control) W2=1 ;; esac
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
# W1: llm.json 68 GiB cap + 16 reserve. W2: the media policy's cap + reserve (media-64: 64 + 16).
need=84
if [ "$W2" = 1 ]; then
  need="$(policy_need_gib "$F2/policies/$W2_POLICY" 2>/dev/null)" || { need=0; problems="$problems; $F2/policies/$W2_POLICY is unreadable"; }
fi
if measured="$(vm_stat | host_memory_from_vm_stat)" && [ -n "$measured" ]; then
  gib=$(( ${measured%% *} / 1073741824 ))
  echo "available RAM (free + speculative + purgeable + file cache): ${gib} GiB (need >= ${need} = cap + reserve)"
  echo "  components (available-bytes page-size free speculative purgeable inactive file-backed anonymous throttled active file-cache pages): $measured"
  [ "$gib" -ge "$need" ] || problems="$problems; available RAM is ${gib} GiB (< ${need})"
else
  problems="$problems; vm_stat host memory could not be measured"
fi
lms_idle || problems="$problems; LM Studio is not idle: $LMS_STATE"
echo "LM Studio: $LMS_STATE"
busy="$(busy_processes)"
[ -z "$busy" ] || problems="$problems; other MLX/cargo processes are running: $(printf '%s' "$busy" | tr '\n' ' ')"
verify_tree "$INF" "$INFERENCE_URL" "$INFERENCE_SHA" || problems="$problems; inference tree is not clean at $INFERENCE_SHA"
if [ "$W2" = 1 ]; then
  # The W2 parents read the inference checkout (source identity, git HEAD), never SceneWorks.
  verify_frozen "$F2" || problems="$problems; W2 frozen dir does not verify"
  case "$F2" in *[[:space:]]*) problems="$problems; $F2 contains whitespace (the Krea launcher shlex-splits its product command)" ;; esac
  python3.12 "$KV_DIR/models.py" --check-only --hub "$KV_HF_HUB" --pins "$KV_DIR/models-w2.tsv" \
    || problems="$problems; W2 pinned snapshots are missing or wrong"
  if [ "$phase" != c ]; then
    for route in wan_vace wan_vace_fun; do
      problem="$(verify_w2_vace "$route")" || problems="$problems; assembled $route: $problem"
    done
    python3.12 "$KV_DIR/fixtures.py" verify --out "$W2_FIXTURES" --pins "$KV_DIR/fixtures-w2.tsv" \
      || problems="$problems; VACE fixtures do not verify"
  fi
else
  verify_tree "$SW" "$SCENEWORKS_URL" "$SCENEWORKS_SHA" || problems="$problems; SceneWorks tree is not clean at $SCENEWORKS_SHA"
  verify_frozen || problems="$problems; frozen dir does not verify"
  python3.12 "$KV_DIR/models.py" --check-only --hub "$KV_HF_HUB" --pins "$KV_DIR/models.tsv" \
    || problems="$problems; pinned snapshots are missing or wrong"
fi
if [ "$phase" = a3 ] && [ ! -d "$DENSE_BASELINE" ]; then
  problems="$problems; A3 needs the completed A1 evidence $DENSE_BASELINE"
fi
if [ -n "$problems" ]; then
  echo "::error title=precheck NO-GO ($phase)::${problems#; }"
  finish 1 false "precheck NO-GO: ${problems#; }"
  exit 1
fi
echo "precheck: GO"

if [ "$W2" = 1 ]; then
  export PMETAL_METALLIB_PATH="$F2/mlx.metallib"
else
  export PMETAL_METALLIB_PATH="$F/mlx.metallib"
  export SCENEWORKS_ROOT="$SW"
fi

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

# W2: the SC-20686 Wan manifest's variables (scripts/sc20686_mlx_wan_campaign_manifest.example.json)
# and the matrix arguments shared by d and d-control (SC_20686_PERSISTENT_KV_CAMPAIGN.md).
if [ "$W2" = 1 ]; then
  export SC20686_MLX_BIN_DIR="$F2" \
    SC20686_MLX_WAN_TI2V_5B_SNAPSHOT="$W2_TI2V" SC20686_MLX_WAN_T2V_14B_SNAPSHOT="$W2_T2V" \
    SC20686_MLX_WAN_I2V_14B_SNAPSHOT="$W2_I2V" SC20686_MLX_WAN_VACE_SNAPSHOT="$W2_VACE" \
    SC20686_MLX_WAN_VACE_FUN_14B_SNAPSHOT="$W2_VACE_FUN" \
    SC20686_WAN_I2V_REFERENCE="$F2/inputs/dog.jpg" SC20686_VACE_REFERENCE="$F2/inputs/dog.jpg" \
    SC20686_VACE_CONTROL_17_DIR="$W2_FIXTURES/512x512-17/control" SC20686_VACE_MASK_17_DIR="$W2_FIXTURES/512x512-17/mask" \
    SC20686_VACE_CONTROL_33_DIR="$W2_FIXTURES/768x512-33/control" SC20686_VACE_MASK_33_DIR="$W2_FIXTURES/768x512-33/mask" \
    SC20686_WAN_T2V_LIGHTNING_HIGH="$W2_LIGHTNING/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/high_noise_model.safetensors" \
    SC20686_WAN_T2V_LIGHTNING_LOW="$W2_LIGHTNING/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/low_noise_model.safetensors" \
    SC20686_WAN_I2V_LIGHTNING_HIGH="$W2_LIGHTNING/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors" \
    SC20686_WAN_I2V_LIGHTNING_LOW="$W2_LIGHTNING/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors"
fi
D_ARGS=(--campaign --matrix --inference-revision "$INFERENCE_SHA" --safety-policy "$F2/policies/$W2_POLICY"
  --stop-file "$CTL/stop-requested" --wan-manifest "$INF/scripts/sc20686_mlx_wan_campaign_manifest.example.json"
  --flux-entrypoint "$F2/sc20686_flux2_edit" --flux-snapshot "$W2_FLUX" --flux-kv-snapshot "$W2_FLUX_KV"
  --flux-reference "$F2/inputs/dog.jpg" --flux-reference2 "$F2/inputs/pulid-reference.png")

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
    if [ "$BASELINE_SHA" != "$INFERENCE_SHA" ]; then
      # The shallow clone has only $INFERENCE_SHA; sc20676 walks the history back to the baseline
      # (ancestry + closure diff), so deepen until it is reachable. sc20676 refuses if it is not.
      for depth in 16 256 4096; do
        git -C "$INF" merge-base --is-ancestor "$BASELINE_SHA" "$INFERENCE_SHA" 2>/dev/null && break
        git -C "$INF" fetch --quiet --no-tags --depth "$depth" origin "$INFERENCE_SHA" \
          || echo "::warning title=history fetch failed::depth $depth of $INFERENCE_SHA"
      done
    fi
    run_cmd "$F/sc20676_packed_evidence" parent --llama-snapshot "$LQ" --qwen-snapshot "$QQ" \
      --llama-baseline-campaign "$DENSE_BASELINE" --qwen-baseline-campaign "$DENSE_BASELINE" \
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
  c)
    # The launcher shlex-splits --product-command: $F2 must stay free of spaces (it is under $HOME).
    run_cmd python3.12 "$INF/scripts/sc20684_krea_realtime_campaign.py" --snapshot "$W2_KREA" --output "$OUT" \
      --safety-policy "$F2/policies/$W2_POLICY" --resume-dir "$RESUME" --stop-file "$CTL/stop-requested" \
      --product-command "$F2/krea-integration --exact --ignored --nocapture $W2_KREA_OBSERVER" || rc=$?
    ;;
  d)
    run_cmd python3.12 "$INF/scripts/sc20686_campaign_adapter.py" "${D_ARGS[@]}" \
      --resume-dir "$RESUME" --matrix-output "$OUT" || rc=$?
    ;;
  d-control)
    run_cmd python3.12 "$INF/scripts/sc20686_campaign_adapter.py" "${D_ARGS[@]}" --schedule-control \
      --resume-dir "$RESUME" --matrix-output "$OUT" || rc=$?
    ;;
esac

# A3 publishes a measured quality-gate miss as evidence; it is never reported as a pass.
gate_note=""
if [ "$rc" = 0 ] && [ "$phase" = a3 ] && [ -f "$OUT/complete-matrix.json" ]; then
  # Families whose gate did not pass (empty = every gate passed); an unreadable matrix never passes.
  failed="$(python3.12 -c 'import json, sys; m = json.load(open(sys.argv[1])); print(",".join(r["family"] for r in m["receipts"] if r.get("qualityGatePassed") is not True) or ("" if m.get("qualityGatePassed") is True else "matrix"))' "$OUT/complete-matrix.json")" \
    || failed="an unreadable complete-matrix.json"
  if [ -n "$failed" ]; then
    echo "::warning title=A3 quality gate FAILED::packed quality gate failed for ${failed}; the receipts record each failed metric (qualityGate.failures). Evidence, NOT a pass."
    gate_note=" -- QUALITY GATE FAILED for ${failed} (evidence recorded, NOT a pass)"
  else
    gate_note=" -- quality gate passed"
  fi
fi

case "$rc" in
  0) finish 0 false "completed${gate_note}" ;;
  75)
    reason="$(cat "$CTL/stop-requested" 2>/dev/null || echo "stop file present")"
    echo "::notice title=phase $phase stopped::$reason; re-dispatch the same parameters to resume"
    finish 75 true "stopped by operator request ($reason); later phases skip; re-dispatch to resume"
    rc=0
    ;;
  *) finish "$rc" false "FAILED" ;;
esac
exit "$rc"
