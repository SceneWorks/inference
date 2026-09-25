set -o pipefail
out="$(SDXL_SNAPSHOT="$SDXL_N1_SNAPSHOT/bf16" \
  cargo test --locked --release -p mlx-gen-sdxl --test integration \
  vae_real_weights::native_decode_seam_is_byte_exact_to_pre_seam_engine \
  -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
grep -qE "test result: ok\. 1 passed" <<<"$out"
