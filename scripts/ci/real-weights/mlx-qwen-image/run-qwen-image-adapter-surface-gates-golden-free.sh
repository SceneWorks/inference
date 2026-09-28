set -o pipefail
run_one() {
  local name="$1" out
  out="$(MLX_GEN_QWEN_SNAPSHOT="$QWEN_IMAGE_MLX_SNAPSHOT/bf16" \
    cargo test --locked --release -p mlx-gen-qwen-image \
    --test integration \
    adapter_real_weights::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
  if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
    return 1
  fi
}
run_one routing_map_covers_full_fork_surface
run_one kohya_matches_peft_on_real_tree
run_one lightning_loras_apply_cleanly
