set -o pipefail
out="$(MLX_GEN_QWEN_SNAPSHOT="$QWEN_IMAGE_MLX_SNAPSHOT/bf16" \
  cargo test --locked --release -p mlx-gen-qwen-image --test integration \
  vae_real_weights::native_decode_seam_is_byte_exact_and_precancelled \
  -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
grep -qE "test result: ok\. 1 passed" <<<"$out"
