# SW-7090 — provision the MiniMax-H3-Image DiT inputs and its diffusers first-step reference.
# Runs only on a reference cache miss.
#
# Runs after the image VAE step in the same job, so the task-owned hub, the pinned
# huggingface-hub install and the reviewed Mage reference venv already exist.
# 1. Materialize `transformer_diffusers/` + `tokenizer/` of MiniMaxAI/MiniMax-H3-Image @492fd8c4
#    (the `minimax-h3-image-dit` row, ~66 GB) into the same snapshot. Neither producer reads
#    MiniMax-H3's text encoder or vision tower (the DiT context is seeded noise); the MLX gate job
#    materializes those for its own renders.
# 2. Produce the DiT reference with the hash-pinned producer (CPU torch, bf16, ~70 GB host RAM)
#    beside the VAE reference, so the artifact carries it to both gate jobs as
#    `minimax-h3-image-vae-reference/dit-reference.safetensors`.
# 3. SW-7092: produce the 3-reference `edit` reference (presentation ids/tags, reference VAE
#    latents, first-step DiT outputs) with its own hash-pinned producer, beside the other two as
#    `edit-reference.safetensors`.
set -o pipefail
export HF_HUB_DISABLE_XET=1
for attempt in 1 2 3; do
  PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model minimax-h3-image-dit --snapshot "$MINIMAX_H3_IMAGE_SNAPSHOT" && break
  [[ $attempt == 3 ]] && exit 1
  echo "::warning::minimax-h3-image-dit materialization attempt $attempt failed"
  sleep 30
done
reference_dir="$GITHUB_WORKSPACE/minimax-h3-image-vae-reference"
"$RUNNER_TEMP/mage-reference/bin/python" -I crates/media/mlx-gen/tools/dump_minimax_h3_image_dit_real.py --out "$reference_dir/dit-reference.safetensors"
"$RUNNER_TEMP/mage-reference/bin/python" -I crates/media/mlx-gen/tools/dump_minimax_h3_image_edit_real.py --out "$reference_dir/edit-reference.safetensors"
