if [[ -z "$ZIMAGE_SNAPSHOT" || "$ZIMAGE_SNAPSHOT" != /* ]]; then
  echo "ZIMAGE_SNAPSHOT must be an operator-provided absolute path" >&2
  exit 1
fi
python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$RUNNER_TEMP/huggingface-hub" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
PYTHONPATH="$RUNNER_TEMP/huggingface-hub" python3.12 scripts/release/ensure_model_snapshot.py \
  --model z-image-turbo \
  --snapshot "$ZIMAGE_SNAPSHOT"
python3.12 scripts/release/verify_model_snapshot.py \
  --model z-image-turbo \
  --snapshot "$ZIMAGE_SNAPSHOT" \
  --inventory-output "$MEMORY_MODEL_INVENTORY"
inventory_sha="$(python3.12 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["inventory_sha256"])' "$MEMORY_MODEL_INVENTORY")"
model_revision="$(python3.12 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["revision"])' "$MEMORY_MODEL_INVENTORY")"
if [[ ! "$inventory_sha" =~ ^[0-9a-f]{64}$ ]]; then
  echo "model inventory did not produce an exact SHA-256" >&2
  exit 1
fi
if [[ ! "$model_revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "model inventory did not bind an exact revision" >&2
  exit 1
fi
echo "MEMORY_MODEL_INVENTORY_SHA256=$inventory_sha" >> "$GITHUB_ENV"
echo "MEMORY_MODEL_REVISION=$model_revision" >> "$GITHUB_ENV"
