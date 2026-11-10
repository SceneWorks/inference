# SW-7090 — provision the MiniMax-H3-Image DiT inputs and its diffusers first-step reference.
#
# Runs after the image VAE step in the same job, so the task-owned hub, the pinned
# huggingface-hub install and the reviewed Mage reference venv already exist.
# 1. Materialize `transformer_diffusers/` + `tokenizer/` of MiniMaxAI/MiniMax-H3-Image @492fd8c4
#    (the `minimax-h3-image-dit` row, ~66 GB) into the same snapshot, and MiniMax-H3's text encoder
#    shards 1-12 (the `minimax-h3-text-encoder` row, ~58.6 GB) into its own pinned snapshot.
#    SW-7092: also MiniMax-H3's vision tower (text-encoder shard 14, the `minimax-h3-vision-tower`
#    row), which the `edit` render reads.
# 2. Produce the DiT reference with the hash-pinned producer (CPU torch, bf16, ~70 GB host RAM).
#    It lands beside the VAE reference so the job's existing artifact upload carries it to the
#    CUDA job as `minimax-h3-image-vae-reference/dit-reference.safetensors`.
# 3. SW-7092: produce the 3-reference `edit` reference (presentation ids/tags, reference VAE
#    latents, first-step DiT outputs) with its own hash-pinned producer, beside the other two as
#    `edit-reference.safetensors`.
set -o pipefail
export HF_HUB_DISABLE_XET=1
lane_hub="$HOME/sceneworks-rw-weights/hub"
export MINIMAX_H3_SNAPSHOT="$lane_hub/models--MiniMaxAI--MiniMax-H3/snapshots/939557dc319dd91227e30195a763f272ba7f8765"
echo "MINIMAX_H3_SNAPSHOT=$MINIMAX_H3_SNAPSHOT" >> "$GITHUB_ENV"
for model in "minimax-h3-image-dit:$MINIMAX_H3_IMAGE_SNAPSHOT" "minimax-h3-text-encoder:$MINIMAX_H3_SNAPSHOT" "minimax-h3-vision-tower:$MINIMAX_H3_SNAPSHOT"; do
  for attempt in 1 2 3; do
    PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model "${model%%:*}" --snapshot "${model#*:}" && break
    [[ $attempt == 3 ]] && exit 1
    echo "::warning::${model%%:*} materialization attempt $attempt failed"
    sleep 30
  done
done
reference="$RUNNER_TEMP/minimax-h3-image-vae-reference/dit-reference.safetensors"
"$RUNNER_TEMP/mage-reference/bin/python" -I crates/media/mlx-gen/tools/dump_minimax_h3_image_dit_real.py --out "$reference"
echo "MINIMAX_H3_IMAGE_DIT_REFERENCE=$reference" >> "$GITHUB_ENV"
edit_reference="$RUNNER_TEMP/minimax-h3-image-vae-reference/edit-reference.safetensors"
"$RUNNER_TEMP/mage-reference/bin/python" -I crates/media/mlx-gen/tools/dump_minimax_h3_image_edit_real.py --out "$edit_reference"
echo "MINIMAX_H3_IMAGE_EDIT_REFERENCE=$edit_reference" >> "$GITHUB_ENV"
