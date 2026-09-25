set -euo pipefail
python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$PYTHONPATH" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
python3.12 - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["QUALITY_REPOSITORY"],
    revision=os.environ["QUALITY_SOURCE_REVISION"],
    allow_patterns=["q4/**"],
    local_dir=os.environ["QUALITY_ROOT"],
    token=False,
)
PY
test -d "$QUALITY_ROOT/q4/transformer"
test -d "$QUALITY_ROOT/q4/text_encoder"
test -d "$QUALITY_ROOT/q4/vae"
echo "CHROMA_QUALITY_ROOT=$QUALITY_ROOT" >> "$GITHUB_ENV"
