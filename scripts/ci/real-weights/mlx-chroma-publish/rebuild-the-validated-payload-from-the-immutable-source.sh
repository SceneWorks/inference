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
mkdir -p "$CHROMA_PACKED"
for bits in 4 8; do
  # `repack_auxiliaries` copies transformer/tokenizer/scheduler/top-level bytes verbatim and
  # rewrites ONLY text_encoder/ + vae/, at the width it derives from the tier's own packed
  # transformer. The publish path therefore cannot move transformer bytes, and cannot mint
  # an auxiliary above the tier the user selected. The `diff` below re-proves the first
  # property independently of the Rust code that promised it.
  SC16462_BASELINE="$CHROMA_SNAPSHOT/q$bits" \
  SC16462_OUT="$CHROMA_BUILT/q$bits" \
  SC16462_MODEL="$CHROMA_MODEL" \
    cargo test --locked --release -p mlx-gen-chroma --test integration \
      auxiliary_pack_identity::packed_auxiliaries_match_load_time_quantization -- --ignored --exact --nocapture \
      --test-threads=1
  cp -R "$CHROMA_BUILT/q$bits/identity-q$bits" "$CHROMA_PACKED/q$bits"
  diff -qr -x text_encoder -x vae "$CHROMA_SNAPSHOT/q$bits" "$CHROMA_PACKED/q$bits"
done
python3.12 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["CHROMA_PACKED"])
expected = json.loads(Path(os.environ["CHROMA_VALIDATED_MANIFEST"]).read_text())
# Each tier's auxiliaries must declare that tier's own width. q4 and q8 therefore DIFFER
# here by construction: equal policies across tiers is precisely the above-tier defect
# sc-16462 removes (the old lane asserted they were equal, which is how a uniform Q8
# auxiliary rode along on the q4 route).
t5_quantization = {}
for tier in ("q4", "q8"):
    quantization = json.loads(
        (root / tier / "text_encoder/config.json").read_text(encoding="utf-8")
    )["quantization"]
    expected_bits = int(tier[1:])
    if quantization.get("bits") != expected_bits:
        raise SystemExit(
            f"{tier}: rebuilt text encoder declares Q{quantization.get('bits')}, expected "
            f"Q{expected_bits} -- an auxiliary above the selected tier is the defect"
        )
    if quantization.get("group_size") != 32:
        raise SystemExit(f"{tier}: T5 group_size {quantization.get('group_size')} != 32")
    if "residual_bits" in quantization:
        raise SystemExit(f"{tier}: residual packing was removed; artifact is stale")
    t5_quantization[tier] = quantization
identity = {
    "model": os.environ["CHROMA_MODEL"],
    "inferenceRevision": os.environ["GITHUB_SHA"],
    "sourceRevision": os.environ["CHROMA_SOURCE_REVISION"],
    "t5Quantization": t5_quantization,
}
for key, value in identity.items():
    if expected.get(key) != value:
        raise SystemExit(
            f"validated payload {key} mismatch: {expected.get(key)!r} != {value!r}"
        )
files = {}
for tier in ("q4", "q8"):
    tier_root = root / tier
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
if files != expected.get("files"):
    missing = sorted(set(expected.get("files", {})) - set(files))
    extra = sorted(set(files) - set(expected.get("files", {})))
    changed = sorted(
        key
        for key in set(files) & set(expected.get("files", {}))
        if files[key] != expected["files"][key]
    )
    raise SystemExit(
        f"rebuilt payload differs from validation: missing={missing}, extra={extra}, changed={changed}"
    )
print(f"verified {len(files)} rebuilt files against validation SHA-256 manifest")
PY
