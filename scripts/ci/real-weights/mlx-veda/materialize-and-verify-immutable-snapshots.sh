python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# As the Iris MLX lanes: plain HTTPS, into the runner's Hugging Face cache (the resolved
# `MLX_GEN_MODELS_ROOT` paths), retrying a transient failure twice; a pinned, verified snapshot is a
# no-op. Phase `device` reads only the Veda predictor; `ab` / `all` also render, so they add the upstream MiniMax-H3 slice
# (VAEs, FL2VA audio VAE, tokenizer, TE index), the SceneWorks q4 DiT/TE tier and the turbo LoRAs.
export HF_HUB_DISABLE_XET=1
df -h "$MLX_GEN_MODELS_ROOT" | tee "$VEDA_OUT/hub-df.txt"
ensure() {
  # Anonymous Hub API calls are rate-limited (HTTP 429 seen on nax-macos-2), so back off longer.
  for attempt in 1 2 3 4 5; do
    PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model "$1" --snapshot "$2" && return 0
    echo "::warning::$1 materialization attempt $attempt failed"
    sleep $((90 * attempt))
  done
  return 1
}
ensure minimax-h3-veda-predictor "$VEDA_AB_PREDICTOR_SNAPSHOT"
if [[ "$VEDA_MLX_PHASE" != "device" ]]; then
  ensure minimax-h3 "$SCENEWORKS_MLX_MINIMAX_H3_BASE_DIR"
  # The q4 row materializes `q4/*` (transformer, transformer_ref, text_encoder); the tests take the
  # snapshot ROOT that holds `q4/`.
  ensure minimax-h3-mlx-q4 "$SCENEWORKS_MLX_MINIMAX_H3_TIER_DIR"
  ensure minimax-h3-turbo-loras "$VEDA_AB_TURBO_SNAPSHOT"
fi
predictor="$VEDA_AB_PREDICTOR_SNAPSHOT/minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors"
test -f "$predictor"
echo "VEDA_PREDICTOR=$predictor" >> "$GITHUB_ENV"
df -h "$MLX_GEN_MODELS_ROOT" | tee -a "$VEDA_OUT/hub-df.txt"
