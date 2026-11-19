python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# The Xet writer dropped its channel mid-way through the 12 GB `upscaler/model.safetensors` on this
# box (2026-10-11): fetch over plain HTTPS, and retry a transient failure twice (the fetch resumes
# from the cache; a pinned, verified snapshot is a no-op).
export HF_HUB_DISABLE_XET=1
# Into the runner's Hugging Face cache (the resolved `MLX_GEN_MODELS_ROOT` paths), where a complete
# snapshot is reused by every later run.
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
ensure iris-3b-upscaler-mlx "$IRIS_WEIGHTS_DIR"
ensure openclip-convnext-xxlarge-mlx "$IRIS_CRITIC_TRUNK_DIR"
ensure vgg16-tv-in1k-mlx "$IRIS_LPIPS_VGG16_DIR"
ensure lpips-vgg-lin-mlx "$IRIS_LPIPS_LIN_DIR"
echo "IRIS_UPSCALER_DIR=$IRIS_WEIGHTS_DIR/upscaler" >> "$GITHUB_ENV"
