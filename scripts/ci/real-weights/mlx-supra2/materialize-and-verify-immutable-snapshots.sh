python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# As the other per-run-materializing MLX lanes: plain HTTPS, into the box's internal task-owned hub (never the
# shared app cache), retrying a transient failure twice; a pinned, verified snapshot is a no-op.
export HF_HUB_DISABLE_XET=1
lane_root="$HOME/sceneworks-rw-weights"
lane_hub="$lane_root/hub"
if [[ -L "$lane_root" ]]; then
  echo "task-owned weights root must be an internal directory: $lane_root" >&2; exit 1
fi
mkdir -p "$lane_hub"
python3.12 scripts/ci/qwen21_weights_root.py "$lane_root"
for v in SUPRA2_SOURCE_SNAPSHOT SUPRA2_T5_SNAPSHOT SUPRA2_VAE_SNAPSHOT; do
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
# The keys are the `gen_core::supra2` revision pins, shared with `candle-supra2`.
ensure supra2-img-cuda "$SUPRA2_SOURCE_SNAPSHOT"
ensure flan-t5-base-cuda "$SUPRA2_T5_SNAPSHOT"
ensure sd-vae-ft-mse-cuda "$SUPRA2_VAE_SNAPSHOT"
