# SW-7089 — MiniMax-H3-Image 2-D VAE decoder on real weights (MLX), `profile: minimax-h3-image`.
#
# MINIMAX_H3_IMAGE_VAE_REFERENCE is OPERATOR-PROVISIONED (it cannot self-heal, like the H3 video
# reference). Produce it on a box holding the snapshot, from a torch + diffusers environment with
# `AutoencoderKLMiniMaxH3Image`:
#   MINIMAX_H3_IMAGE_SNAPSHOT=<root> python3 crates/media/mlx-gen/tools/dump_minimax_h3_image_vae_real.py \
#     --out ~/minimax-h3-image-real-vae-reference.safetensors
# then set the repository variable to that `~/`-relative path and place a copy on every `rw-mage`
# box. The gate refuses a reference whose `producer_sha256` is not the committed producer's.
set -o pipefail
if [[ -z "$MINIMAX_H3_IMAGE_SNAPSHOT" || "$MINIMAX_H3_IMAGE_SNAPSHOT" != /* ]]; then
  echo "Repository variable MINIMAX_H3_IMAGE_SNAPSHOT must be an absolute path, got: '$MINIMAX_H3_IMAGE_SNAPSHOT'" >&2
  exit 1
fi
python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model minimax-h3-image --snapshot "$MINIMAX_H3_IMAGE_SNAPSHOT"
if [[ -z "$MINIMAX_H3_IMAGE_VAE_REFERENCE" || "$MINIMAX_H3_IMAGE_VAE_REFERENCE" != /* || ! -s "$MINIMAX_H3_IMAGE_VAE_REFERENCE" ]]; then
  echo "MINIMAX_H3_IMAGE_VAE_REFERENCE must name a non-empty file on this runner, got: '$MINIMAX_H3_IMAGE_VAE_REFERENCE'; the parity gate refuses to report green without it" >&2
  exit 1
fi
# Name-selected with `--exact` after the `--` plus a pass-count assertion, so a rename cannot go
# vacuously green. Two processes, so the parity run's buffers are reaped before the 16 MP peaks.
for name in image_vae_decode_matches_the_official_diffusers_vae image_vae_tiled_decode_lowers_the_16mp_peak; do
  out="$(cargo test --locked --release -p mlx-gen-minimax-h3 --test integration \
    real_weights::"$name" -- --exact --ignored --nocapture 2>&1 | tee /dev/stderr)"
  if ! grep -qE "test result: ok\. 1 passed" <<<"$out"; then
    echo "::error::'$name' did not run exactly one passing test — a rename would make this step vacuously green" >&2
    exit 1
  fi
done
