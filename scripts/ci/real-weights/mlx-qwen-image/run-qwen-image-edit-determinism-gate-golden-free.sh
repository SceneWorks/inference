set -o pipefail
name=edit_generate_is_deterministic_rust
out="$(QWEN_IMAGE_EDIT_SNAPSHOT="$QWEN_IMAGE_EDIT_MLX_SNAPSHOT/q8" \
  cargo test --locked --release -p mlx-gen-qwen-image \
  --test integration \
  edit_real_weights::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
  echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
  exit 1
fi
