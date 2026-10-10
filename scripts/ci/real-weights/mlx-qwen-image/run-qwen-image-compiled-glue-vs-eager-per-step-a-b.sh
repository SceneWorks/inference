set -o pipefail
run_one() {
  local name="$1" out
  out="$(MLX_GEN_QWEN_SNAPSHOT="$QWEN_IMAGE_MLX_SNAPSHOT/bf16" \
    QWEN_IMAGE_EDIT_SNAPSHOT="$QWEN_IMAGE_EDIT_MLX_SNAPSHOT/q8" \
    cargo test --locked --release -p mlx-gen-qwen-image --test integration \
    perf::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
  if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
    return 1
  fi
}
run_one qwen_t2i_per_step_compiled_vs_eager
run_one qwen_edit_per_step_compiled_vs_eager
