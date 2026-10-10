set -o pipefail
cargo test --locked --release -p mlx-gen-qwen-image-2-1 --lib training::tests::trained_factors_preserve_velocity_through_save_and_reload -- --exact --nocapture --test-threads 1 2>&1 | tee "$QWEN_IMAGE_2_1_RENDER_OUT/velocity-roundtrip.log"
grep -qE 'test result: ok\. 1 passed' "$QWEN_IMAGE_2_1_RENDER_OUT/velocity-roundtrip.log"
if [[ "${QWEN_IMAGE_2_1_LORA_PHASE:-full}" == q4-numeric ]]; then
  # Retain the original twelve combinations above; this closes the width-eight no-pack blindspot.
  cargo test --locked --release -p mlx-gen-qwen-image-2-1 --lib training::tests::fully_packed_trained_factors_preserve_direct_delta_through_export -- --exact --nocapture --test-threads 1 2>&1 | tee "$QWEN_IMAGE_2_1_RENDER_OUT/packed-velocity-roundtrip.log"
  grep -qE 'test result: ok\. 1 passed' "$QWEN_IMAGE_2_1_RENDER_OUT/packed-velocity-roundtrip.log"
fi
