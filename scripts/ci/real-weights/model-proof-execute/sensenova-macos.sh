set -euo pipefail
hub_lib="$RUNNER_TEMP/sensenova-hub-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
if [[ "${1:-}" == publish ]]; then
  PYTHONPATH="$hub_lib" python3.12 scripts/ci/sensenova_loop_publish.py
  return
fi
python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$hub_lib" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
PYTHONPATH="$hub_lib" python3.12 scripts/ci/sensenova_loop_assets.py --backend mlx --expected-sha "$GITHUB_SHA" \
  --work "$RUNNER_TEMP/sensenova-native-work-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT" \
  --report "$SENSENOVA_NATIVE_REPORT" --oracle-dir "$SENSENOVA_REFERENCE_REPORT"
