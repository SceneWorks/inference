# Each check runs under the external footprint guard: cap 40 GB, 30 min wall. The training run's
# budget: the bf16 3B restoration backbone (~6 GB) + the bf16 ConvNeXt-XXL critic trunk (~1.7 GB)
# + f32 VGG16 (~0.55 GB) + rank-8 LoRA factors / AdamW / EMA + 256² activations.
bin="$(cat "$RUNNER_TEMP/iris-restore-test-binary")"
failed=0
for t in restoration_train_real_weights::real_lpips_and_critic_trunk_match_the_reference \
         restoration_train_real_weights::real_restoration_trains_exports_and_restores; do
  log="$IRIS_OUT/${t##*::}.log"
  python3.12 scripts/ci/iris_restore_footprint_guard.py 40 1800 \
    "$bin" "$t" --ignored --exact --nocapture --test-threads 1 > "$log" 2>&1 || failed=1
  cat "$log"
  grep -q "test result: ok. 1 passed" "$log" || { echo "::error::$t did not run exactly one passing test"; failed=1; }
done
exit "$failed"
