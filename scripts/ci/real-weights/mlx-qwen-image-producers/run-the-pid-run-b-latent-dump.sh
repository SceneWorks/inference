set -o pipefail
name=dump_runb_latents
out="$(MLX_GEN_QWEN_SNAPSHOT="$QWEN_IMAGE_MLX_SNAPSHOT/bf16" \
  cargo test --locked --release -p mlx-gen-qwen-image --test integration \
  dump_runb_latents::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
  echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
  exit 1
fi
