python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$PYTHONPATH" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
python3.12 - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["CHROMA_REPOSITORY"],
    revision=os.environ["CHROMA_SOURCE_REVISION"],
    allow_patterns=["bf16/**", "q4/**", "q8/**"],
    local_dir=os.environ["CHROMA_SNAPSHOT"],
    token=False,
)
PY
test -d "$CHROMA_SNAPSHOT/bf16/transformer"
test -d "$CHROMA_SNAPSHOT/bf16/text_encoder"
test -d "$CHROMA_SNAPSHOT/bf16/vae"
test -d "$CHROMA_SNAPSHOT/q4/transformer"
test -d "$CHROMA_SNAPSHOT/q8/transformer"
