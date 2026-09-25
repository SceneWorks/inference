set -o pipefail
run_one() {
  local name="$1" out
  out="$(MLX_GEN_QWEN_SNAPSHOT="$QWEN_IMAGE_MLX_SNAPSHOT/bf16" \
    QWEN_CONTROL_WEIGHTS="$QWEN_CONTROL_UNION_SNAPSHOT/bf16/model.safetensors" \
    cargo test --locked --release -p mlx-gen-qwen-image --test integration \
    control_real_weights::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
  if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
    return 1
  fi
}
run_one control_loads_and_emits_hints
run_one scale_zero_matches_base
run_one scale_one_changes_output
run_one public_generate_runs
