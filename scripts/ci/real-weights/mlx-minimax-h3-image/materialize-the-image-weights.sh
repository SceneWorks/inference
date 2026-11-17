# SW-7089/7090/7092 — provision the MiniMax-H3-Image gate inputs with no operator step. The
# diffusers references come from the `minimax-h3-image-reference` job (the downloaded
# `minimax-h3-image-vae-reference` artifact); this step materializes the weights the gates read and
# exports the snapshot and reference paths for the later steps.
#
# Into this box's internal task-owned hub (plain HTTPS, two retries; a pinned, verified snapshot is a
# no-op): `vae_diffusers/`, `transformer_diffusers/` and `tokenizer/` of MiniMaxAI/MiniMax-H3-Image
# @492fd8c4 (the `minimax-h3-image` and `minimax-h3-image-dit` rows), and MiniMax-H3's text encoder
# shards 1-12 and vision tower (shard 14) (the `minimax-h3-text-encoder` and
# `minimax-h3-vision-tower` rows), which the renders read.
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
# The pinned revisions, laid out the way MLX_GEN_MODELS_ROOT lays out every hub snapshot.
export MINIMAX_H3_IMAGE_SNAPSHOT="$lane_hub/models--MiniMaxAI--MiniMax-H3-Image/snapshots/492fd8c4eca666f1085f57e30738c1e0d86b3439"
export MINIMAX_H3_SNAPSHOT="$lane_hub/models--MiniMaxAI--MiniMax-H3/snapshots/939557dc319dd91227e30195a763f272ba7f8765"
{
  echo "MINIMAX_H3_IMAGE_SNAPSHOT=$MINIMAX_H3_IMAGE_SNAPSHOT"
  echo "MINIMAX_H3_SNAPSHOT=$MINIMAX_H3_SNAPSHOT"
} >> "$GITHUB_ENV"
for model in "minimax-h3-image:$MINIMAX_H3_IMAGE_SNAPSHOT" "minimax-h3-image-dit:$MINIMAX_H3_IMAGE_SNAPSHOT" "minimax-h3-text-encoder:$MINIMAX_H3_SNAPSHOT" "minimax-h3-vision-tower:$MINIMAX_H3_SNAPSHOT"; do
  for attempt in 1 2 3; do
    PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model "${model%%:*}" --snapshot "${model#*:}" && break
    [[ $attempt == 3 ]] && exit 1
    echo "::warning::${model%%:*} materialization attempt $attempt failed"
    sleep 30
  done
done
reference_dir="$RUNNER_TEMP/minimax-h3-image-vae-reference"
for entry in VAE:reference DIT:dit-reference EDIT:edit-reference; do
  reference="$reference_dir/${entry#*:}.safetensors"
  if [[ ! -f "$reference" ]]; then
    echo "::error::the reference job's artifact is missing $reference" >&2; exit 1
  fi
  echo "MINIMAX_H3_IMAGE_${entry%%:*}_REFERENCE=$reference" >> "$GITHUB_ENV"
done
