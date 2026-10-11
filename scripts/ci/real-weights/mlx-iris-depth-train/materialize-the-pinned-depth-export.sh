python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$PYTHONPATH" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
python3.12 - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="speridlabs/iris-3b",
    revision="7445443349bc9abe3c96f01ff793e2098ca012b3",
    allow_patterns=["depth/*"],
    local_dir=os.environ["IRIS_WEIGHTS_DIR"],
    token=False,
)
PY
test -f "$IRIS_WEIGHTS_DIR/depth/config.yaml"
test -f "$IRIS_WEIGHTS_DIR/depth/model.safetensors"
test -f "$IRIS_WEIGHTS_DIR/depth/empty_prompt.safetensors"
du -sh "$IRIS_WEIGHTS_DIR/depth"
