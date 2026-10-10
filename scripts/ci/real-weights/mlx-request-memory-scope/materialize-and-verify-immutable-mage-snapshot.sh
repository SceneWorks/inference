if [[ -z "$MAGE_REQUEST_SCOPE_SNAPSHOT" || "$MAGE_REQUEST_SCOPE_SNAPSHOT" != /* ]]; then
  echo "MAGE_SNAPSHOT must be an operator-provided absolute path" >&2
  exit 1
fi
python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py \
  --model mage-flow \
  --snapshot "$MAGE_REQUEST_SCOPE_SNAPSHOT" \
  --require-materialization-provenance
