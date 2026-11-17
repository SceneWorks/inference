set -o pipefail
python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
export HF_HUB_DISABLE_XET=1
lane_root="$HOME/sceneworks-rw-weights"
if [[ -L "$lane_root" ]]; then
  echo "task-owned weights root must be an internal directory: $lane_root" >&2; exit 1
fi
mkdir -p "$lane_root/hub"
python3.12 scripts/ci/qwen21_weights_root.py "$lane_root"
image="$lane_root/hub/models--MiniMaxAI--MiniMax-H3-Image/snapshots/492fd8c4eca666f1085f57e30738c1e0d86b3439"
h3="$lane_root/hub/models--MiniMaxAI--MiniMax-H3/snapshots/939557dc319dd91227e30195a763f272ba7f8765"
for model in "minimax-h3-image:$image" "minimax-h3-image-dit:$image" "minimax-h3-text-encoder:$h3" "minimax-h3-vision-tower:$h3"; do
  for attempt in 1 2 3; do
    PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model "${model%%:*}" --snapshot "${model#*:}" && break
    [[ $attempt == 3 ]] && exit 1
    echo "::warning::${model%%:*} materialization attempt $attempt failed"
    sleep 30
  done
done
{
  echo "MINIMAX_H3_IMAGE_SNAPSHOT=$image"
  echo "MINIMAX_H3_SNAPSHOT=$h3"
  echo "MINIMAX_H3_IMAGE_TIERS_ROOT=$lane_root/minimax-h3-image-train-tiers"
} >> "$GITHUB_ENV"
