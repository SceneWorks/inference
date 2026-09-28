set -euo pipefail
# The ladder's tests are exactly the `#[ignore]`d ones; helpers are not dispatchable.
all_tests="$(python3.12 - <<'PY'
import re, pathlib
src = pathlib.Path("crates/media/mlx-gen/mlx-gen-chroma/tests/memory_ladder_real_weights.rs").read_text()
names = re.findall(r'#\[test\]\s*\n\s*#\[ignore[^\]]*\]\s*\nfn ([a-z0-9_]+)\(', src)
print("\n".join(names))
PY
)"
n_all="$(grep -c . <<<"$all_tests")"
if [[ "$n_all" -ne 16 ]]; then
  echo "::error::expected 16 ignored ladder tests, found $n_all — the list this lane dispatches has drifted from the file" >&2
  exit 1
fi
for t in $CHROMA_LADDER_TESTS; do
  if ! grep -qxF "$t" <<<"$all_tests"; then
    echo "::error::chroma_ladder_tests names '$t', which is not a test in memory_ladder_real_weights.rs" >&2
    exit 1
  fi
done
for tier in $CHROMA_LADDER_TIERS; do
  case "$tier" in
    q4|q8|bf16) ;;
    *) echo "::error::chroma_ladder_tiers names '$tier'; valid tiers are q4 q8 bf16" >&2; exit 1 ;;
  esac
done
if ! tr ' ' '\n' <<<"$CHROMA_LADDER_TIERS" | grep -qx q4; then
  echo "::error::chroma_ladder_tiers must include q4 — DEFAULT_TIER is what every asserted row measures" >&2
  exit 1
fi
{ echo "CHROMA_LADDER_ALL_TESTS<<EOF"; echo "$all_tests"; echo "EOF"; } >> "$GITHUB_ENV"
