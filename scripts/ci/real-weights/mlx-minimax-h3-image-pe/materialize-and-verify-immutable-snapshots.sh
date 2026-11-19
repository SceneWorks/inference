python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# As the other per-run-materializing MLX lanes: plain HTTPS, into the runner's Hugging Face cache (the
# resolved `MLX_GEN_MODELS_ROOT` path), retrying a transient failure twice; a pinned, verified
# snapshot is a no-op.
export HF_HUB_DISABLE_XET=1
ensure() {
  for attempt in 1 2 3; do
    PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model "$1" --snapshot "$2" && return 0
    echo "::warning::$1 materialization attempt $attempt failed"
    sleep 30
  done
  return 1
}
# The key is the `release/real-weight-models.toml` pin, shared with `candle-minimax-h3-image-pe`.
ensure minimax-h3-image-pe "$MINIMAX_PE_SNAPSHOT"
