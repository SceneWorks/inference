python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# As the other Iris MLX lanes: plain HTTPS, into the runner's Hugging Face cache (the resolved
# `MLX_GEN_MODELS_ROOT` paths), retrying a transient failure twice; a pinned, verified snapshot is a
# no-op.
export HF_HUB_DISABLE_XET=1
df -h "$MLX_GEN_MODELS_ROOT" | tee "$IRIS_OUT/hub-df.txt"
ensure() {
  for attempt in 1 2 3; do
    PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model "$1" --snapshot "$2" && return 0
    echo "::warning::$1 materialization attempt $attempt failed"
    sleep 30
  done
  return 1
}
# ONE entry for the three task exports of the one snapshot root (backbone, depth/, upscaler/), so no
# materialization of a subset can replace the others.
ensure iris-3b-all-tasks-mlx "$IRIS_WEIGHTS_DIR"
ensure qwen3-vl-4b-instruct-mlx "$IRIS_TEXT_ENCODER_DIR"
ensure openclip-convnext-xxlarge-mlx "$IRIS_CRITIC_TRUNK_DIR"
ensure vgg16-tv-in1k-mlx "$IRIS_LPIPS_VGG16_DIR"
ensure lpips-vgg-lin-mlx "$IRIS_LPIPS_LIN_DIR"
echo "IRIS_UPSCALER_DIR=$IRIS_WEIGHTS_DIR/upscaler" >> "$GITHUB_ENV"
