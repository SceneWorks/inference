python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# As the other per-run-materializing MLX lanes: plain HTTPS, into the runner's Hugging Face cache (the
# resolved `MLX_GEN_MODELS_ROOT` paths), retrying a transient failure twice; a pinned, verified
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
# The keys are the `gen_core::supra2` revision pins, shared with `candle-supra2`.
ensure supra2-img-cuda "$SUPRA2_SOURCE_SNAPSHOT"
ensure flan-t5-base-cuda "$SUPRA2_T5_SNAPSHOT"
ensure sd-vae-ft-mse-cuda "$SUPRA2_VAE_SNAPSHOT"
