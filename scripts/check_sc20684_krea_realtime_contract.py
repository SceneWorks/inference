#!/usr/bin/env python3
"""Fail closed when the SC-20684 Krea source-only POC contract drifts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "sc-20684-krea-realtime-compressed-kv-contract.json"
ADR_NAME = "SC_20684_KREA_REALTIME_COMPRESSED_KV_SOURCE_MAP.md"
REQUIRED_MAPPINGS = {
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/config.rs", "pub struct KvCacheQuant"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/config.rs", "pub fn row_bytes"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/causal.rs", "struct PackedKv"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/causal.rs", "fn dense(&self, quant: Option<KvCacheQuant>)"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/causal.rs", "pub fn window_prev"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/causal.rs", "pub fn append"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/causal.rs", "fn denoise_chunk_inner"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/t2v.rs", "fn resolve_request_config"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/generate.rs", "fn run_ar_loop_conditioned"),
    ("crates/media/mlx-gen/mlx-gen-wan/src/transformer.rs", "fn forward_causal("),
    ("crates/media/mlx-gen/mlx-gen-wan/src/transformer.rs", "pub fn forward_causal_chunk"),
}
REQUIRED_FALLBACKS = {"format", "mask", "head-dimension", "lifecycle", "receipt"}
REQUIRED_RECEIPT_FIELDS = {
    "modelIdentity", "requestGeometry", "cacheGeometry", "maskCapability",
    "representationIdentity", "persistentBytes", "transientBytes", "fallbackReason",
    "parity", "quality", "cancellation",
}
REQUIRED_UPSTREAM_MECHANISMS = {"scalar_fused_decode_attend", "rabitq_prefill_attend"}
REQUIRED_HEADINGS = (
    "## Proven current route",
    "## Existing evidence boundary",
    "## Frozen upstream comparison and decision",
    "## First implementation blocker",
)


def errors_for(data: dict, source_root: Path) -> list[str]:
    errors: list[str] = []
    if data.get("schemaVersion") != 1 or data.get("story") != "SC-20684":
        errors.append("schema or story mismatch")
    if data.get("validation") != "checked-out-current-source":
        errors.append("validation must bind checked-out current source")
    if data.get("decision") != "do-not-implement-until-format-and-mask-route-are-proven":
        errors.append("decision must remain fail-closed")
    upstream = data.get("upstream")
    if (
        not isinstance(upstream, dict)
        or upstream.get("tag") != "v0.65.0"
        or upstream.get("commit") != "54989ee223611627592f7f9bd925e924658f1f22"
        or set(upstream.get("mechanisms", [])) != REQUIRED_UPSTREAM_MECHANISMS
    ):
        errors.append("immutable upstream mismatch")
    poc = data.get("poc")
    if (
        not isinstance(poc, dict)
        or poc.get("status") != "blocked-before-implementation"
        or poc.get("nonGoal") != "full-cache dequantize-then-SDPA is not compressed-domain execution"
    ):
        errors.append("POC boundary mismatch")
    fallbacks = data.get("rejectionFallbacks")
    fallback_ids = (
        {item.get("id") for item in fallbacks if isinstance(item, dict)}
        if isinstance(fallbacks, list)
        else set()
    )
    invalid_reason = not isinstance(fallbacks, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("reason"), str)
        or not item["reason"]
        for item in fallbacks
    )
    if fallback_ids != REQUIRED_FALLBACKS or invalid_reason or len(fallbacks) != len(REQUIRED_FALLBACKS):
        errors.append("fallback taxonomy must be complete and reasoned")
    receipt = data.get("receipt")
    fields = set(receipt.get("requiredFields", [])) if isinstance(receipt, dict) else set()
    if (
        not isinstance(receipt, dict)
        or receipt.get("status") != "not-produced"
        or fields != REQUIRED_RECEIPT_FIELDS
        or len(receipt.get("requiredFields", [])) != len(REQUIRED_RECEIPT_FIELDS)
    ):
        errors.append("receipt contract must remain unproduced and complete")
    mappings = data.get("sourceMappings")
    declared = (
        {(item.get("path"), item.get("needle")) for item in mappings if isinstance(item, dict)}
        if isinstance(mappings, list)
        else set()
    )
    if (
        not isinstance(mappings, list)
        or declared != REQUIRED_MAPPINGS
        or len(mappings) != len(REQUIRED_MAPPINGS)
    ):
        errors.append("sourceMappings must cover exactly the current Krea seams")
    for path, needle in declared & REQUIRED_MAPPINGS:
        source = source_root / path
        if not source.is_file():
            errors.append(f"stale source mapping: {path}")
        elif needle not in source.read_text(encoding="utf-8"):
            errors.append(f"stale source needle: {path}: {needle}")
    return errors


def validate(root: Path, source_root: Path) -> list[str]:
    docs = root / "docs/architecture"
    manifest = docs / MANIFEST_NAME
    sidecar = docs / f"{MANIFEST_NAME}.sha256"
    adr = docs / ADR_NAME
    try:
        raw = manifest.read_bytes()
        checksum = sidecar.read_text(encoding="utf-8").strip()
        adr_text = adr.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"missing contract artifact: {exc}"]
    errors = [] if raw.endswith(b"\n") and b"\r\n" not in raw else ["manifest must be LF terminated"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [*errors, f"invalid JSON: {exc}"]
    errors.extend(errors_for(data, source_root))
    expected = f"{hashlib.sha256(raw).hexdigest()}  {MANIFEST_NAME}"
    if checksum != expected:
        errors.append("manifest checksum mismatch")
    errors.extend(f"source map missing required section: {heading}" for heading in REQUIRED_HEADINGS if heading not in adr_text)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args = parser.parse_args()
    errors = validate(args.root, args.source_root)
    if errors:
        print("\n".join(errors))
        return 1
    print("SC-20684 Krea Realtime compressed-KV source contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
