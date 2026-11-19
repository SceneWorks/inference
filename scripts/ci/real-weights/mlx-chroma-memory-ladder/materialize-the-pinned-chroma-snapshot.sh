set -euo pipefail
python3.12 -m pip install --disable-pip-version-check --only-binary=:all: --require-hashes --target "$PYTHONPATH" -r .github/requirements/real-weights-huggingface-hub-macos-arm64-py312.txt
# Through the runner's Hugging Face hub cache (`cache_dir`, never a per-run directory), so a
# complete snapshot transfers nothing on a rerun. The snapshot path comes back on stdout.
CHROMA_LADDER_ROOT="$(python3.12 - <<'PY'
import os
import sys
sys.path.insert(0, "scripts/release")
from ensure_model_snapshot import download_to_runner_cache

tiers = os.environ["CHROMA_LADDER_TIERS"].split()
print(download_to_runner_cache(
    repo_id=os.environ["CHROMA_REPOSITORY"],
    revision=os.environ["CHROMA_SOURCE_REVISION"],
    allow_patterns=[f"{tier}/**" for tier in tiers],
    token=False,
))
PY
)"
# The shared cache snapshot also holds whatever tiers other Chroma lanes fetched at this revision,
# and the harness measures every tier present under its root. Give it a per-run view holding only
# the requested tiers (links into the cache, no copy), so `chroma_ladder_tiers` keeps bounding what
# this leg measures.
cache_snapshot="$CHROMA_LADDER_ROOT"
CHROMA_LADDER_ROOT="$RUNNER_TEMP/chroma-ladder-view-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT-$CHROMA_MODEL"
rm -rf "$CHROMA_LADDER_ROOT"
mkdir -p "$CHROMA_LADDER_ROOT"
for tier in $CHROMA_LADDER_TIERS; do
  ln -s "$cache_snapshot/$tier" "$CHROMA_LADDER_ROOT/$tier"
done
export CHROMA_LADDER_ROOT
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
