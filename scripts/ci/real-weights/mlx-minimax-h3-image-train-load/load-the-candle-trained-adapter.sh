set -o pipefail
export MINIMAX_H3_IMAGE_CANDLE_TRAINED_LORA="$RUNNER_TEMP/minimax-h3-image-candle-trained-lora/h3_image_lora.safetensors"
[[ -f "$MINIMAX_H3_IMAGE_CANDLE_TRAINED_LORA" ]] || { echo "::error::no Candle-trained adapter at $MINIMAX_H3_IMAGE_CANDLE_TRAINED_LORA" >&2; exit 1; }
name=h3_image_candle_lora_load_real::h3_image_candle_trained_lora_installs_on_the_real_dit
log="$RUNNER_TEMP/mlx-load.log"
status=0
MODEL_INFERENCE_CASE_FQN="$name" cargo test --locked --release -p mlx-gen-minimax-h3 \
  --features model-inference-proof --test integration \
  "$name" -- --exact --ignored --nocapture 2>&1 | tee "$log" >&2 || status=$?
if ((status)) || ! python3.12 -I scripts/ci/verify_model_inference_log.py --package mlx-gen-minimax-h3 \
  --backend mlx --fqn "$name" --log "$log"; then
  echo "::error::'$name' failed or lacks its registered proof evidence (exit $status)" >&2
  exit 1
fi
