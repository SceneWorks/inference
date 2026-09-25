set -o pipefail
: > "$RUNNER_TEMP/chroma-sc16462-identity.jsonl"
: > "$RUNNER_TEMP/chroma-sc16462-comparison.jsonl"
for bits in 4 8; do
  # The shipped tier IS the baseline: it already carries the exact packed transformer plus
  # the dense bf16 auxiliaries. `repack_auxiliaries` derives the auxiliary width from that
  # transformer, so a tier can never be built whose text encoder sits above its own tier.
  export SC16462_BASELINE="$CHROMA_SNAPSHOT/q$bits"
  test -d "$SC16462_BASELINE/transformer"
  test -d "$SC16462_BASELINE/text_encoder"

  # Gate 1 — conversion faithfulness: the published pack must be BIT-identical to the
  # in-app load-time pack, or the shipped tier and a dense-source render diverge.
  cargo test --locked --release -p mlx-gen-chroma --test integration \
    auxiliary_pack_identity::packed_auxiliaries_match_load_time_quantization -- --ignored --exact --nocapture \
    --test-threads=1 | tee "$RUNNER_TEMP/chroma-q$bits-identity.log"
  sed -n 's/^.*SC16462_IDENTITY //p' "$RUNNER_TEMP/chroma-q$bits-identity.log" \
    >> "$RUNNER_TEMP/chroma-sc16462-identity.jsonl"

  # Gate 2 — the shipped-vs-at-tier comparison: transformer bytes byte-identical, render
  # coherent (a missed packed site reads codes as floats and collapses), auxiliaries
  # actually smaller. Deliberately NO pixel-identity threshold against the bf16-auxiliary
  # render: packing the text encoder to the selected tier is supposed to move the
  # conditioning, and gating on identity would only pass by not doing the work.
  cargo test --locked --release -p mlx-gen-chroma --test integration \
    auxiliary_tier_comparison::compare_auxiliary_widths -- --ignored --exact --nocapture --test-threads=1 \
    | tee "$RUNNER_TEMP/chroma-q$bits-comparison.log"
  sed -n 's/^.*auxiliaries @ //p' "$RUNNER_TEMP/chroma-q$bits-comparison.log" \
    >> "$RUNNER_TEMP/chroma-sc16462-comparison.jsonl"

  # Stage the tier that will actually be published, from the same derived-width path.
  rm -rf "$CHROMA_PACKED_ROOT/q$bits"
  mkdir -p "$CHROMA_PACKED_ROOT"
  cp -R "$SC16462_OUT/tier-aux-at-tier" "$CHROMA_PACKED_ROOT/q$bits"
  diff -qr -x text_encoder -x vae "$SC16462_BASELINE" "$CHROMA_PACKED_ROOT/q$bits"
done
test "$(wc -l < "$RUNNER_TEMP/chroma-sc16462-identity.jsonl" | tr -d ' ')" = 2
python3.12 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["CHROMA_PACKED_ROOT"])
files = {}
t5_quantization = {}
for tier in ("q4", "q8"):
    tier_root = root / tier
    expected_bits = int(tier[1:])
    quantization = json.loads(
        (tier_root / "text_encoder/config.json").read_text(encoding="utf-8")
    )["quantization"]
    t5_quantization[tier] = quantization
    # The invariant this whole story exists to restore: the auxiliaries sit AT the tier the
    # user selected, never above it.
    if quantization.get("bits") != expected_bits:
        raise SystemExit(
            f"{tier}: text encoder declares Q{quantization.get('bits')}, expected "
            f"Q{expected_bits} -- an auxiliary above the selected tier is the defect"
        )
    if quantization.get("group_size") != 32:
        raise SystemExit(f"{tier}: T5 group_size {quantization.get('group_size')} != 32")
    if "residual_bits" in quantization:
        raise SystemExit(f"{tier}: residual packing was removed; artifact is stale")
    for path in sorted(tier_root.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            relative = f"{tier}/{path.relative_to(tier_root).as_posix()}"
            files[relative] = {
                "size": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
manifest = {
    "model": os.environ["CHROMA_MODEL"],
    "inferenceRevision": os.environ["GITHUB_SHA"],
    "sourceRevision": os.environ["CHROMA_SOURCE_REVISION"],
    "t5Quantization": t5_quantization,
    "files": files,
}
Path(os.environ["CHROMA_PAYLOAD_MANIFEST"]).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
