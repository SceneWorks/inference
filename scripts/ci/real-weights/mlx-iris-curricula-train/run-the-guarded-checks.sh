# Each check runs under the external footprint guard (scripts/ci/iris_restore_footprint_guard.py).
# The release REPA teacher (dinov2_vitb14, ~0.35 GB f32) against the committed upstream tokens:
# cap 20 GB, 15 min wall.
# The bounded curriculum (stage1_256 with REPA → stage2_512 new phase, full fine-tune of the 3B
# generator, Muon, bf16, batch 1): cap 100 GB, 90 min wall — f32 masters (~12 GB) + Muon momentum
# (~12 GB) + f32 grads (~12 GB) + the per-step bf16 copy (~6 GB) + the frozen teacher + Qwen3-VL
# conditioning + 256² / 512² activations (stage 2 under full activation checkpointing). It starts
# only when `memory_pressure` reports >= 120 GB free (free % × total RAM, decimal GB) right before
# it; otherwise it is skipped — green, as the restore lane's full fine-tune — with the measured
# numbers in the log, a warning annotation and `curriculum-admission.txt`. Its checkpoints (~36 GB each, one kept
# per stage) live under the runner's temp dir and are removed with the test's tempdir.
bin="$(cat "$RUNNER_TEMP/iris-curricula-test-binary")"
failed=0
run_guarded() { # $1 cap GB, $2 wall s, $3 test path
  local t="$3" log="$IRIS_OUT/${3##*::}.log"
  TMPDIR="$RUNNER_TEMP" python3.12 scripts/ci/iris_restore_footprint_guard.py "$1" "$2" \
    "$bin" "$t" --ignored --exact --nocapture --test-threads 1 > "$log" 2>&1 || failed=1
  cat "$log"
  grep -q "test result: ok. 1 passed" "$log" || { echo "::error::$t did not run exactly one passing test"; failed=1; }
}
# The teacher check is an FP32 parity on the CPU device: the test runs its body in a CPU-default
# child of itself; naming the child here runs that body in the guarded process directly, so the
# guard measures it (0.41 GB on CPU, measured).
teacher_test=curricula_parity::the_release_teacher_matches_upstream_on_real_weights
IRIS_TEST_CPU_CHILD="$teacher_test" IRIS_TEST_CPU_CHILD_DONE="$IRIS_OUT/teacher-child.done" \
  run_guarded 20 900 "$teacher_test"

mem_bytes="$(sysctl -n hw.memsize)"
free_pct="$(/usr/bin/memory_pressure -Q 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')"
free_gb=$(( mem_bytes / 1000000000 * ${free_pct:-0} / 100 ))
echo "curriculum admission: RAM $((mem_bytes / 1000000000)) GB, memory_pressure free ${free_pct:-?}% = ${free_gb} GB (need >= 120 GB)" \
  | tee "$IRIS_OUT/curriculum-admission.txt"
if [[ -z "$free_pct" || "$free_gb" -lt 120 ]]; then
  echo "::warning::skipped (not run, not failed) the curriculum run: ${free_gb} GB free (memory_pressure ${free_pct:-?}%) is under 120 GB" \
    | tee -a "$IRIS_OUT/curriculum-admission.txt"
else
  run_guarded 100 5400 curricula_real_weights::real_stage_presets_resume_256_to_512_with_repa
fi
exit "$failed"
