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
for attempt in 1 2 3; do
  PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py --model minimax-h3-image-dit --snapshot "$image" && break
  [[ $attempt == 3 ]] && exit 1
  echo "::warning::minimax-h3-image-dit materialization attempt $attempt failed"
  sleep 30
done
echo "MINIMAX_H3_IMAGE_SNAPSHOT=$image" >> "$GITHUB_ENV"
