set -o pipefail
cargo test --locked --release -p mlx-gen-qwen-image-2-1 --lib training::tests::trained_factors_preserve_velocity_through_save_and_reload -- --exact --nocapture --test-threads 1 2>&1 | tee "$QWEN_IMAGE_2_1_RENDER_OUT/velocity-roundtrip.log"
grep -qE 'test result: ok\. 1 passed' "$QWEN_IMAGE_2_1_RENDER_OUT/velocity-roundtrip.log"
