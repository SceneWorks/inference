set -o pipefail
name=fit_preview_rgb_factors
out="$(MLX_GEN_QWEN_SNAPSHOT="$QWEN_IMAGE_MLX_SNAPSHOT/bf16" \
  QWEN_LIGHTNING_SNAPSHOT="$QWEN_LIGHTNING_SNAPSHOT" \
  cargo test --locked --release -p mlx-gen-qwen-image \
  --test integration \
  fit_preview_rgb::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
  echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
  exit 1
fi
