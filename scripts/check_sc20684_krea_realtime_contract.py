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
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/compressed_kv.rs", "pub struct ExperimentalCompressedKvConfig"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/compressed_kv.rs", "pub trait RetainedKernelHandle"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/compressed_kv.rs", "pub fn tiled_online_attention"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/compressed_kv.rs", "pub fn append_after_decision"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/compressed_kv.rs", "pub fn trim_prefix"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/compressed_kv.rs", "pub struct KreaPackedMetalKernel"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/causal.rs", "pub fn enable_experimental_packed_metal"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/causal.rs", "fn prepare_packed_window"),
    ("crates/media/mlx-gen/mlx-gen-wan/src/transformer.rs", "pub trait CausalPackedAttention"),
    ("crates/media/mlx-gen/mlx-gen-wan/src/transformer.rs", "pub fn forward_causal_chunk_with_packed_attention"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/causal.rs", "pub fn packed_metal_route_receipt"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/tests/generate_smoke.rs", "fn sc20684_packed_campaign_observer()"),
}
REQUIRED_FALLBACKS = {"disabled", "q4-quality", "handle", "geometry", "mask", "cancellation", "receipt"}
REQUIRED_RECEIPT_FIELDS = {
    "modelIdentity", "requestGeometry", "cacheGeometry", "maskCapability",
    "representationIdentity", "compiledHandleIdentity", "persistentBytes", "retainedHandleBytes",
    "boundedScratchBytes", "denseWindowBytes", "scoreMatrixBytes", "fallbackReason", "parity",
    "quality", "cancellation", "timingLabel",
}
REQUIRED_UPSTREAM_MECHANISMS = {"scalar_fused_decode_attend", "rabitq_prefill_attend"}
REQUIRED_HEADINGS = (
    "## Proven current route",
    "## Existing evidence boundary",
    "## Frozen upstream comparison and decision",
    "## Experimental source POC",
)


def errors_for(data: dict, source_root: Path) -> list[str]:
    errors: list[str] = []
    if data.get("schemaVersion") != 1 or data.get("story") != "SC-20684":
        errors.append("schema or story mismatch")
    if data.get("validation") != "checked-out-current-source":
        errors.append("validation must bind checked-out current source")
    if data.get("decision") != "experimental-krea-owned-packed-affine-online-softmax-poc":
        errors.append("decision must name the checked-in experimental POC")
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
        or poc.get("status") != "implemented-source-only-device-unverified"
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
        or receipt.get("status") != "schema-only-not-produced"
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
    poc_source = source_root / "crates/media/mlx-gen/mlx-gen-krea-realtime/src/compressed_kv.rs"
    if poc_source.is_file():
        source = poc_source.read_text(encoding="utf-8")
        production_source = source.split("#[cfg(test)]", 1)[0]
        forbidden = ("dequantize(", "scaled_dot_product_attention", "build_block_causal_mask(")
        for needle in (*forbidden, "let mut scores"):
            if needle in production_source:
                errors.append(f"compressed POC must not allocate a dense K/V window or score route: {needle}")
        required = (
            "TILE_ROWS", "dense_window_bytes = 0", "score_matrix_bytes = 0", "DispatchDecision",
            "simdgroup_matrix", "simdgroup_load", "simdgroup_multiply_accumulate",
            "KreaPackedMetalKernel", "current_k", "current_v", "acc0 *= old_weight",
            "lane * 4", ".thread_group(256, 1, 1)",
        )
        for needle in required:
            if needle not in source:
                errors.append(f"compressed POC structural guard missing: {needle}")
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
