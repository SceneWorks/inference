if [[ -z "$SANA_LADDER_1600M" || "$SANA_LADDER_1600M" != /* ]]; then
  echo "SANA_LADDER_1600M must be an operator-provided absolute path" >&2
  exit 1
fi
python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py \
  --model sana-1600m-mlx \
  --snapshot "$SANA_LADDER_1600M"
python3.12 scripts/release/verify_model_snapshot.py \
  --model sana-1600m-mlx \
  --snapshot "$SANA_LADDER_1600M" \
  --inventory-output "$SANA_MODEL_INVENTORY"
