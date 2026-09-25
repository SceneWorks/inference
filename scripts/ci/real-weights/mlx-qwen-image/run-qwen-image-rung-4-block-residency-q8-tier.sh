set -o pipefail
name=bounded_window_is_distinct_from_the_unbounded_stream_control
out="$(MLX_GEN_QWEN_SNAPSHOT="$QWEN_IMAGE_MLX_SNAPSHOT/q8" QWEN_RUNG4_TIER=q8 \
  cargo test --locked --release -p mlx-gen-qwen-image \
  --test integration \
  block_residency_real_weights::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
  echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
  exit 1
fi
