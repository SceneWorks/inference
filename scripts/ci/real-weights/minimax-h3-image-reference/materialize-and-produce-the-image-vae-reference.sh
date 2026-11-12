# SW-7089 — provision the MiniMax-H3-Image VAE inputs with no operator step. Runs only on a
# reference cache miss; the MLX and CUDA gate jobs read the result from this job's artifact.
#
# 1. Materialize ONLY `vae_diffusers/` of MiniMaxAI/MiniMax-H3-Image @492fd8c4 (the `minimax-h3-image`
#    row, 2.34 GB) into this box's internal task-owned hub, as the other per-run-materializing MLX
#    lanes do (plain HTTPS, two retries; a pinned, verified snapshot is a no-op).
# 2. Produce the diffusers reference with the hash-pinned producer, in the reviewed hash-locked Mage
#    reference environment (CPU torch + diffusers 0.38.0; the producer loads the pinned upstream
#    `AutoencoderKLMiniMaxH3Image` module by SHA-256). The reference lands in the workspace-relative
#    `minimax-h3-image-vae-reference/`, the directory the runner store publishes and the artifact uploads.
set -o pipefail
python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
export HF_HUB_DISABLE_XET=1
lane_root="$HOME/sceneworks-rw-weights"
lane_hub="$lane_root/hub"
if [[ -L "$lane_root" ]]; then
  echo "task-owned weights root must be an internal directory: $lane_root" >&2; exit 1
fi
mkdir -p "$lane_hub"
python3.12 scripts/ci/qwen21_weights_root.py "$lane_root"
# The pinned revision, laid out the way MLX_GEN_MODELS_ROOT lays out every hub snapshot.
export MINIMAX_H3_IMAGE_SNAPSHOT="$lane_hub/models--MiniMaxAI--MiniMax-H3-Image/snapshots/492fd8c4eca666f1085f57e30738c1e0d86b3439"
echo "MINIMAX_H3_IMAGE_SNAPSHOT=$MINIMAX_H3_IMAGE_SNAPSHOT" >> "$GITHUB_ENV"
for attempt in 1 2 3; do
  PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model minimax-h3-image --snapshot "$MINIMAX_H3_IMAGE_SNAPSHOT" && break
  [[ $attempt == 3 ]] && exit 1
  echo "::warning::minimax-h3-image materialization attempt $attempt failed"
  sleep 30
done
if [[ ! -x "$RUNNER_TEMP/mage-reference/bin/python" ]]; then
  python3.12 -m venv "$RUNNER_TEMP/mage-reference"
  "$RUNNER_TEMP/mage-reference/bin/python" -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes -r crates/media/mlx-gen/_vendor/mage_flow/requirements-oracles.txt
fi
reference_dir="$GITHUB_WORKSPACE/minimax-h3-image-vae-reference"
mkdir -p "$reference_dir"
"$RUNNER_TEMP/mage-reference/bin/python" -I crates/media/mlx-gen/tools/dump_minimax_h3_image_vae_real.py \
  --out "$reference_dir/reference.safetensors"
