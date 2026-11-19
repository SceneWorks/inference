python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$PYTHONPATH" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# Through the runner's Hugging Face hub cache (`cache_dir`, never a per-run directory), so a
# complete snapshot transfers nothing on a rerun. The snapshot path comes back on stdout.
IRIS_WEIGHTS_DIR="$(python3.12 - <<'PY'
import os
import sys
sys.path.insert(0, "scripts/release")
from ensure_model_snapshot import download_to_runner_cache

print(download_to_runner_cache(
    repo_id="speridlabs/iris-3b",
    revision="7445443349bc9abe3c96f01ff793e2098ca012b3",
    allow_patterns=["depth/*"],
    token=False,
))
PY
)"
export IRIS_WEIGHTS_DIR
echo "IRIS_WEIGHTS_DIR=$IRIS_WEIGHTS_DIR" >> "$GITHUB_ENV"
test -f "$IRIS_WEIGHTS_DIR/depth/config.yaml"
test -f "$IRIS_WEIGHTS_DIR/depth/model.safetensors"
test -f "$IRIS_WEIGHTS_DIR/depth/empty_prompt.safetensors"
du -sh "$IRIS_WEIGHTS_DIR/depth"
