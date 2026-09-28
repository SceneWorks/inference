set -o pipefail
run_one() {
  local name="$1" out
  out="$(MLX_GEN_QWEN_SNAPSHOT="$QWEN_IMAGE_MLX_SNAPSHOT/bf16" \
    PID_QWEN_SAFETENSORS="$PID_QWEN_SNAPSHOT/pid_qwenimage_2kto4k.safetensors" \
    PID_GEMMA_DIR="$PID_GEMMA_SNAPSHOT" \
    cargo test --locked --release -p mlx-gen-qwen-image \
    --test integration \
    pid_decode_real_weights::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
  if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
    return 1
  fi
}
run_one use_pid_without_loaded_pid_errors
run_one qwen_image_pid_decode_vs_vae
run_one qwen_image_pid_from_ldm_early_stop
