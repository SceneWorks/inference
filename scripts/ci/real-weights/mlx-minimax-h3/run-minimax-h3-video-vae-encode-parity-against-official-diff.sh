set -o pipefail
name=real_weight_encode_matches_the_official_diffusers_vae
out="$(cargo test --locked --release -p mlx-gen-minimax-h3 --test integration \
  real_weights::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
  echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
  exit 1
fi
