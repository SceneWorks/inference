python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# As the other Iris MLX lanes: plain HTTPS, into the box's internal task-owned hub (never the shared
# app cache), retrying a transient failure twice; a pinned, verified snapshot is a no-op.
export HF_HUB_DISABLE_XET=1
lane_root="$HOME/sceneworks-rw-weights"
lane_hub="$lane_root/hub"
if [[ -L "$lane_root" ]]; then
  echo "task-owned weights root must be an internal directory: $lane_root" >&2; exit 1
fi
mkdir -p "$lane_hub"
python3.12 scripts/ci/qwen21_weights_root.py "$lane_root"
df -h "$lane_hub" | tee "$IRIS_OUT/lane-hub-df.txt"
for v in IRIS_WEIGHTS_DIR IRIS_TEXT_ENCODER_DIR IRIS_CRITIC_TRUNK_DIR IRIS_LPIPS_VGG16_DIR IRIS_LPIPS_LIN_DIR; do
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
# ONE entry for the three task exports of the one snapshot root (backbone, depth/, upscaler/), so no
# materialization of a subset can replace the others.
ensure iris-3b-all-tasks-mlx "$IRIS_WEIGHTS_DIR"
ensure qwen3-vl-4b-instruct-mlx "$IRIS_TEXT_ENCODER_DIR"
ensure openclip-convnext-xxlarge-mlx "$IRIS_CRITIC_TRUNK_DIR"
ensure vgg16-tv-in1k-mlx "$IRIS_LPIPS_VGG16_DIR"
ensure lpips-vgg-lin-mlx "$IRIS_LPIPS_LIN_DIR"
echo "IRIS_UPSCALER_DIR=$IRIS_WEIGHTS_DIR/upscaler" >> "$GITHUB_ENV"
