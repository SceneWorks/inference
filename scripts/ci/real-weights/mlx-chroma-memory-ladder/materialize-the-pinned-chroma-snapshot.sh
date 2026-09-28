set -euo pipefail
python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$PYTHONPATH" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
python3.12 - <<'PY'
import os
from huggingface_hub import snapshot_download

tiers = os.environ["CHROMA_LADDER_TIERS"].split()
snapshot_download(
    repo_id=os.environ["CHROMA_REPOSITORY"],
    revision=os.environ["CHROMA_SOURCE_REVISION"],
    allow_patterns=[f"{tier}/**" for tier in tiers],
    local_dir=os.environ["CHROMA_LADDER_ROOT"],
    token=False,
)
PY
# The harness takes a snapshot ROOT and joins the tier itself, so assert the layout it
# expects rather than discovering a `SKIPPED-BY-ABSENCE` panic hours into the ladder.
for tier in $CHROMA_LADDER_TIERS; do
  test -d "$CHROMA_LADDER_ROOT/$tier/transformer"
  test -d "$CHROMA_LADDER_ROOT/$tier/text_encoder"
  test -d "$CHROMA_LADDER_ROOT/$tier/vae"
done
# The harness takes ONE env var per catalog entry. Only this leg's entry is
# materialized, so the other two stay unset and the per-cell coverage test reports them
# as absent rather than measuring them — its documented contract, not a gap. Exported
# here rather than as a job `env:` ternary so the path expression exists once.
case "$CHROMA_MODEL" in
  chroma1_base)  entry_var=CHROMA_LADDER_BASE ;;
  chroma1_hd)    entry_var=CHROMA_LADDER_HD ;;
  chroma1_flash) entry_var=CHROMA_LADDER_FLASH ;;
  *) echo "::error::unknown Chroma entry '$CHROMA_MODEL'" >&2; exit 1 ;;
esac
echo "$entry_var=$CHROMA_LADDER_ROOT" >> "$GITHUB_ENV"
