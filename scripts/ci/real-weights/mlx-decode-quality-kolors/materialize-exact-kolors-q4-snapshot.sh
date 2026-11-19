set -euo pipefail
python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$PYTHONPATH" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# Through the runner's Hugging Face hub cache (`cache_dir`, never a per-run directory), so a
# complete snapshot transfers nothing on a rerun. The snapshot path comes back on stdout.
QUALITY_ROOT="$(python3.12 - <<'PY'
import os
import sys
sys.path.insert(0, "scripts/release")
from ensure_model_snapshot import runner_hub_cache
from huggingface_hub import snapshot_download

print(snapshot_download(
    repo_id=os.environ["QUALITY_REPOSITORY"],
    revision=os.environ["QUALITY_SOURCE_REVISION"],
    allow_patterns=["q4/**"],
    cache_dir=str(runner_hub_cache()),
    token=False,
))
PY
)"
export QUALITY_ROOT
test -d "$QUALITY_ROOT/q4/unet"
test -d "$QUALITY_ROOT/q4/text_encoder"
test -d "$QUALITY_ROOT/q4/vae"
echo "KOLORS_QUALITY_ROOT=$QUALITY_ROOT" >> "$GITHUB_ENV"
