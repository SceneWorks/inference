python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# As the Iris MLX lanes: plain HTTPS, into the runner's Hugging Face cache (the resolved
# `MLX_GEN_MODELS_ROOT` paths; the prompt-upsampler package of sc-26238 too), retrying a transient
# failure twice; a pinned, verified snapshot is a no-op. The package
# is GATED (FLUX Kommunity License): the manifest row carries `requires_auth = true`, so the fetch
# uses the runner's own configured Hugging Face credential and nothing is interpolated here.
export HF_HUB_DISABLE_XET=1
ensure() {
  for attempt in 1 2 3; do
    PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model "$1" --snapshot "$2" && return 0
    echo "::warning::$1 materialization attempt $attempt failed"
    sleep 30
  done
  return 1
}
ensure flux3-image-klein-mlx "$FLUX3_PACKAGE_DIR"
ensure flux3-image-klein-pup-mlx "$FLUX3_PUP_DIR"
