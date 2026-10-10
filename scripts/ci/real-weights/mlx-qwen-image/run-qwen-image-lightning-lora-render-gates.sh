set -o pipefail
run_one() {
  local name="$1" out
  out="$(MLX_GEN_QWEN_SNAPSHOT="$QWEN_IMAGE_MLX_SNAPSHOT/bf16" \
    QWEN_IMAGE_EDIT_SNAPSHOT="$QWEN_IMAGE_EDIT_MLX_SNAPSHOT/q8" \
    cargo test --locked --release -p mlx-gen-qwen-image \
    --test integration \
    lightning_render_real_weights::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
  if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
    return 1
  fi
}
run_one lightning_render_is_coherent
run_one edit_lightning_render_is_coherent
