python3.12 scripts/release/ensure_model_snapshot_file.py \
  --model krea-2-turbo-mlx-q4 \
  --file LICENSE.pdf \
  --cache-root "$MLX_GEN_MODELS_ROOT"
python3.12 scripts/release/ensure_model_snapshot_file.py \
  --model krea-realtime-14b-mlx-wan-z16-vae-q8 \
  --file q8/vae.safetensors \
  --cache-root "$MLX_GEN_MODELS_ROOT"
