#!/usr/bin/env python3
"""Fail closed when the SC-20674 compressed-KV contract loses scope or seams."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/architecture/sc-20674-compressed-kv-contract.json"
SIDECAR = ROOT / "docs/architecture/sc-20674-compressed-kv-contract.json.sha256"
ADR = ROOT / "docs/architecture/SC_20674_COMPRESSED_KV_CONTRACT.md"
HEADINGS = ("## Decision and current seams", "## State diagrams", "## Compatibility and migration", "## Capability, error, and fallback taxonomy", "## Exhaustive lifecycle-to-dense-equivalent plan", "## Candidate boundary and first-push gate")
REQUIRED_MAPPINGS = {
    ("crates/llm/mlx-llm/src/primitives/kv_cache.rs", "pub trait KvCache"),
    ("crates/llm/mlx-llm/src/primitives/kv_cache.rs", "fn retain_sequences"),
    ("crates/llm/mlx-llm/src/primitives/paged_kv_cache.rs", "pub struct PagedKvCache"),
    ("crates/llm/mlx-llm/src/primitives/paged_kv_cache.rs", "pub fn shareable_prefix_blocks"),
    ("crates/llm/mlx-llm/src/primitives/attention.rs", "pub fn sdpa_capped"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/causal.rs", "pub fn window_prev"),
    ("crates/media/mlx-gen/mlx-gen-krea-realtime/src/causal.rs", "pub fn append"),
}


def errors_for(data: dict, source_root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    required = data.get("requiredOperations")
    routes = data.get("operationRoutes")
    if not isinstance(required, list) or len(required) != len(set(required)):
        return ["requiredOperations must be a unique list"]
    route_map = {route.get("operation"): route for route in routes if isinstance(route, dict)} if isinstance(routes, list) else {}
    if set(route_map) != set(required):
        errors.append("operationRoutes must cover exactly requiredOperations")
    for operation in required:
        route = route_map.get(operation, {})
        if route.get("status") not in {"compressed-supported", "dense-fallback-before-mutation"}:
            errors.append(f"{operation}: missing valid route status")
        if not isinstance(route.get("reason"), str) or not route["reason"]:
            errors.append(f"{operation}: missing deterministic reason")
    mappings = data.get("sourceMappings")
    if not isinstance(mappings, list):
        errors.append("sourceMappings must be a list")
        mappings = []
    declared = {(mapping.get("path"), mapping.get("needle")) for mapping in mappings if isinstance(mapping, dict)}
    if declared != REQUIRED_MAPPINGS:
        errors.append("sourceMappings must cover exactly the required current seams")
    for mapping in mappings:
        path, needle = mapping.get("path"), mapping.get("needle")
        if not isinstance(path, str) or not isinstance(needle, str) or not path or not needle:
            errors.append("source mapping needs path and needle")
            continue
        relative = Path(path)
        source = source_root / relative
        if relative.is_absolute() or ".." in relative.parts or not source.is_file():
            errors.append(f"stale source mapping: {path}")
        elif needle not in source.read_text(encoding="utf-8"):
            errors.append(f"stale source needle: {path}: {needle}")
    return errors


def main() -> int:
    raw = MANIFEST.read_bytes()
    errors = [] if raw.endswith(b"\n") and b"\r\n" not in raw else ["manifest must be LF terminated"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        data, errors = {}, [*errors, f"invalid JSON: {exc}"]
    if data.get("story") != "SC-20674" or data.get("base") != "3deb898c8dfa572e939ba9705adfe311dd6d43f0":
        errors.append("story or immutable base mismatch")
    if data.get("validation") != "checked-out-current-source":
        errors.append("source validation must bind current checkout")
    errors.extend(errors_for(data))
    if SIDECAR.read_text(encoding="utf-8").strip().split(maxsplit=1) != [hashlib.sha256(raw).hexdigest(), MANIFEST.name]:
        errors.append("manifest checksum mismatch")
    adr = ADR.read_text(encoding="utf-8")
    errors.extend(f"ADR missing required section: {heading}" for heading in HEADINGS if heading not in adr)
    if errors:
        print("\n".join(errors))
        return 1
    print("SC-20674 compressed-KV contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
