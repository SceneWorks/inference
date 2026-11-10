# SW-7089 — AC1 parity vs diffusers and AC2 16 MP bounded-decode peak, MLX. Name-selected with
# `--exact` after the `--` plus a pass-count assertion, so a rename cannot go vacuously green. Two
# processes, so the parity run's buffers are reaped before the 16 MP peaks are measured.
set -o pipefail
for name in image_vae_decode_matches_the_official_diffusers_vae image_vae_tiled_decode_lowers_the_16mp_peak; do
  out="$(cargo test --locked --release -p mlx-gen-minimax-h3 --test integration \
    real_weights::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
  if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
    exit 1
  fi
done
