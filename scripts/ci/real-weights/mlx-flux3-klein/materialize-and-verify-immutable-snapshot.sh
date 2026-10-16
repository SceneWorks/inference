python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# As the Iris MLX lanes: plain HTTPS, into the box's internal task-owned hub (never the shared app
# cache), retrying a transient failure twice; a pinned, verified snapshot is a no-op. The package
# is GATED (FLUX Kommunity License): the manifest row carries `requires_auth = true`, so the fetch
# uses the runner's own configured Hugging Face credential and nothing is interpolated here.
export HF_HUB_DISABLE_XET=1
lane_root="$HOME/sceneworks-rw-weights"
lane_hub="$lane_root/hub"
if [[ -L "$lane_root" ]]; then
  echo "task-owned weights root must be an internal directory: $lane_root" >&2; exit 1
fi
mkdir -p "$lane_hub"
python3.12 scripts/ci/qwen21_weights_root.py "$lane_root"
df -h "$lane_hub" | tee "$FLUX3_OUT/lane-hub-df.txt"
tail_path="${FLUX3_PACKAGE_DIR#"$MLX_GEN_MODELS_ROOT"/}"
if [[ "$tail_path" == "$FLUX3_PACKAGE_DIR" || "$tail_path" == *..* ]]; then
  echo "snapshot must stay beneath the resolved hub: FLUX3_PACKAGE_DIR=$FLUX3_PACKAGE_DIR" >&2; exit 1
fi
FLUX3_PACKAGE_DIR="$lane_hub/$tail_path"
export FLUX3_PACKAGE_DIR
echo "FLUX3_PACKAGE_DIR=$FLUX3_PACKAGE_DIR" >> "$GITHUB_ENV"
ensure() {
  for attempt in 1 2 3; do
    PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model "$1" --snapshot "$2" && return 0
    echo "::warning::$1 materialization attempt $attempt failed"
    sleep 30
  done
  return 1
}
ensure flux3-image-klein-mlx "$FLUX3_PACKAGE_DIR"
