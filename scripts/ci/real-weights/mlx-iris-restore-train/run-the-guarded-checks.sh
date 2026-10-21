# Each check runs under the external footprint guard (scripts/ci/iris_restore_footprint_guard.py).
# The reference check and the LoRA run: cap 40 GB, 30 min wall — the bf16 3B restoration backbone
# (~6 GB) + the bf16 ConvNeXt-XXL critic trunk (~1.7 GB) + f32 VGG16 (~0.55 GB) + rank-8 LoRA factors
# / AdamW / EMA + 256² activations (measured 25 GB on nax-macos-2, 2026-10-11).
# The paper's FULL fine-tune: cap 100 GB, 60 min wall — f32 masters (~12 GB) + AdamW moments
# (~24 GB) + f32 grads (~12 GB) + f32 EMA (~12 GB) + the per-step bf16 copy (~6 GB) + the frozen
# critic trunk / VGG + activations. It starts only when `memory_pressure` reports >= 120 GB free
# (free % × total RAM, decimal GB) right before it; otherwise it is skipped with the measured numbers.
bin="$(cat "$RUNNER_TEMP/iris-restore-test-binary")"
failed=0
run_guarded() { # $1 cap GB, $2 wall s, $3 test path
  local t="$3" log="$IRIS_OUT/${3##*::}.log"
  python3.12 scripts/ci/iris_restore_footprint_guard.py "$1" "$2" \
    "$bin" "$t" --ignored --exact --nocapture --test-threads 1 > "$log" 2>&1 || failed=1
  cat "$log"
  grep -q "test result: ok. 1 passed" "$log" || { echo "::error::$t did not run exactly one passing test"; failed=1; }
}
run_guarded 40 1800 restoration_train_real_weights::real_lpips_and_critic_trunk_match_the_reference
run_guarded 40 1800 restoration_train_real_weights::real_restoration_trains_exports_and_restores

mem_bytes="$(sysctl -n hw.memsize)"
free_pct="$(/usr/bin/memory_pressure -Q 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')"
free_gb=$(( mem_bytes / 1000000000 * ${free_pct:-0} / 100 ))
echo "full fine-tune admission: RAM $((mem_bytes / 1000000000)) GB, memory_pressure free ${free_pct:-?}% = ${free_gb} GB (need >= 120 GB)" \
  | tee "$IRIS_OUT/full-finetune-admission.txt"
if [[ -z "$free_pct" || "$free_gb" -lt 120 ]]; then
  echo "::warning::skipping the full fine-tune case: ${free_gb} GB free (memory_pressure ${free_pct:-?}%) is under 120 GB"
else
  run_guarded 100 3600 restoration_train_real_weights::real_restoration_full_finetune_trains_exports_and_restores
fi
exit "$failed"
