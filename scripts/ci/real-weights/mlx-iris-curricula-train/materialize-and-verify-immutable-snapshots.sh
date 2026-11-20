python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# As `mlx-iris-restore-train`: plain HTTPS (the Xet writer dropped mid-way through a 12 GB file on
# this box), into the runner's Hugging Face cache (the resolved `MLX_GEN_MODELS_ROOT` paths),
# retrying a transient failure twice (a verified snapshot is a no-op).
export HF_HUB_DISABLE_XET=1
hub_real="$(cd "$MLX_GEN_MODELS_ROOT" && pwd -P)"
df -h "$hub_real" | tee "$IRIS_OUT/shared-hub-df.txt"
ensure() {
  for attempt in 1 2 3; do
    PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model "$1" --snapshot "$2" && return 0
    echo "::warning::$1 materialization attempt $attempt failed"
    sleep 30
  done
  return 1
}
ensure iris-3b-generation-mlx "$IRIS_WEIGHTS_DIR"
ensure qwen3-vl-4b-instruct-mlx "$IRIS_TEXT_ENCODER_DIR"
ensure dinov2-base-mlx "$IRIS_REPA_TEACHER_DIR"
