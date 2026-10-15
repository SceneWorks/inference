python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# As the Iris MLX lanes: plain HTTPS, into the box's internal task-owned hub (never the shared app
# cache), retrying a transient failure twice; a pinned, verified snapshot is a no-op. Phase `device`
# reads only the Veda predictor; `ab` / `all` also render, so they add the upstream MiniMax-H3 slice
# (VAEs, FL2VA audio VAE, tokenizer, TE index), the SceneWorks q4 DiT/TE tier and the turbo LoRAs.
export HF_HUB_DISABLE_XET=1
lane_root="$HOME/sceneworks-rw-weights"
lane_hub="$lane_root/hub"
if [[ -L "$lane_root" ]]; then
  echo "task-owned weights root must be an internal directory: $lane_root" >&2; exit 1
fi
mkdir -p "$lane_hub"
python3.12 scripts/ci/qwen21_weights_root.py "$lane_root"
df -h "$lane_hub" | tee "$VEDA_OUT/lane-hub-df.txt"
vars=(VEDA_AB_PREDICTOR_SNAPSHOT)
if [[ "$VEDA_MLX_PHASE" != "device" ]]; then
  vars+=(SCENEWORKS_MLX_MINIMAX_H3_BASE_DIR SCENEWORKS_MLX_MINIMAX_H3_TIER_DIR VEDA_AB_TURBO_SNAPSHOT)
fi
for v in "${vars[@]}"; do
  tail_path="${!v#"$MLX_GEN_MODELS_ROOT"/}"
  if [[ "$tail_path" == "${!v}" || "$tail_path" == *..* ]]; then
    echo "snapshot must stay beneath the resolved hub: $v=${!v}" >&2; exit 1
  fi
  printf -v "$v" '%s' "$lane_hub/$tail_path"
  export "$v"
  echo "$v=${!v}" >> "$GITHUB_ENV"
done
ensure() {
  for attempt in 1 2 3; do
    PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model "$1" --snapshot "$2" && return 0
    echo "::warning::$1 materialization attempt $attempt failed"
    sleep 30
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
df -h "$lane_hub" | tee -a "$VEDA_OUT/lane-hub-df.txt"
