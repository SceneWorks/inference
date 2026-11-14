# SW-7104: install the hash-locked Hugging Face client, require the runner's own Hugging Face login
# (the identity publish_hf_tiers.py uploads as; the token is never printed), and create this run's
# scratch directory in the internal weights root. Only this run's scratch is ever removed.
python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 -c 'from huggingface_hub import HfApi; n = HfApi().whoami().get("name"); assert n, "no Hugging Face login on this runner"; print("Hugging Face identity:", n)'
lane_root="$HOME/sceneworks-rw-weights"
if [[ -L "$lane_root" ]]; then
  echo "task-owned weights root must be an internal directory: $lane_root" >&2; exit 1
fi
mkdir -p "$lane_root/hub"
python3.12 scripts/ci/qwen21_weights_root.py "$lane_root"
scratch="$lane_root/sw-7104-publish-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
reports="$RUNNER_TEMP/sw-7104-publication-reports"
rm -rf "$reports"
mkdir -p "$scratch" "$reports"
{
  echo "PUBLISH_SCRATCH=$scratch"
  echo "PUBLISH_REPORTS=$reports"
  echo "LANE_ROOT=$lane_root"
} >> "$GITHUB_ENV"
