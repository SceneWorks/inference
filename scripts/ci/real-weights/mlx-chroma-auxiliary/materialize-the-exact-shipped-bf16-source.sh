python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$PYTHONPATH" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# Through the runner's Hugging Face hub cache (`cache_dir`, never a per-run directory), so a
# complete snapshot transfers nothing on a rerun. The snapshot path comes back on stdout.
CHROMA_SNAPSHOT="$(python3.12 - <<'PY'
import os
import sys
sys.path.insert(0, "scripts/release")
from ensure_model_snapshot import download_to_runner_cache

print(download_to_runner_cache(
    repo_id=os.environ["CHROMA_REPOSITORY"],
    revision=os.environ["CHROMA_SOURCE_REVISION"],
    allow_patterns=["bf16/**", "q4/**", "q8/**"],
    token=False,
))
PY
)"
export CHROMA_SNAPSHOT
echo "CHROMA_SNAPSHOT=$CHROMA_SNAPSHOT" >> "$GITHUB_ENV"
test -d "$CHROMA_SNAPSHOT/bf16/transformer"
test -d "$CHROMA_SNAPSHOT/bf16/text_encoder"
test -d "$CHROMA_SNAPSHOT/bf16/vae"
test -d "$CHROMA_SNAPSHOT/q4/transformer"
test -d "$CHROMA_SNAPSHOT/q8/transformer"
