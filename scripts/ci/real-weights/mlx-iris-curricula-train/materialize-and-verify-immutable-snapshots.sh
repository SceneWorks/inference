python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# As `mlx-iris-restore-train`: plain HTTPS (the Xet writer dropped mid-way through a 12 GB file on
# this box), into the internal task-owned hub this box's real-weight lanes use (the shared app hub
# ran out of space mid-fetch), retrying a transient failure twice (a verified snapshot is a no-op).
export HF_HUB_DISABLE_XET=1
hub_real="$(cd "$MLX_GEN_MODELS_ROOT" && pwd -P)"
df -h "$hub_real" | tee "$IRIS_OUT/shared-hub-df.txt"
lane_root="$HOME/sceneworks-rw-weights"
lane_hub="$lane_root/hub"
if [[ -L "$lane_root" ]]; then
  echo "task-owned weights root must be an internal directory: $lane_root" >&2; exit 1
fi
mkdir -p "$lane_hub"
python3.12 scripts/ci/qwen21_weights_root.py "$lane_root"
df -h "$lane_hub" | tee "$IRIS_OUT/lane-hub-df.txt"
for v in IRIS_WEIGHTS_DIR IRIS_TEXT_ENCODER_DIR IRIS_REPA_TEACHER_DIR; do
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
ensure iris-3b-generation-mlx "$IRIS_WEIGHTS_DIR"
ensure qwen3-vl-4b-instruct-mlx "$IRIS_TEXT_ENCODER_DIR"
ensure dinov2-base-mlx "$IRIS_REPA_TEACHER_DIR"
