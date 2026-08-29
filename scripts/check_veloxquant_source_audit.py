#!/usr/bin/env python3
"""Validate SC-20672's byte-sealed VeloxQuant source-audit provenance."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/architecture/sc-20672-veloxquant-provenance.json"
SIDECAR = ROOT / "docs/architecture/sc-20672-veloxquant-provenance.json.sha256"
AUDIT = ROOT / "docs/architecture/SC_20672_VELOXQUANT_SOURCE_AUDIT.md"
SHA = re.compile(r"^[0-9a-f]{40}$")

EXPECTED = {
    "story": "SC-20672",
    "upstream.commit": "54989ee223611627592f7f9bd925e924658f1f22",
    "upstream.planningBaseline.commit": "92909d441cfe1cad6693d9eec5cbf6f57a1d8ff4",
    "localProvenance.productInferenceRevision": "3775a5f80a07a38071c7859f6ac565bcab5d1c7b",
}


def get(data: dict, path: str):
    value = data
    for part in path.split("."):
        value = value[part]
    return value


def source_mapping_errors(data: dict, source_root: Path = ROOT) -> list[str]:
    """Bind mappings to the checked-out source, not an archival product pin.

    ``productInferenceRevision`` is immutable provenance for the SceneWorks
    product pin. GitHub PR checkouts deliberately fetch only the current head,
    so requiring that historical object would make this source audit depend on
    unrelated checkout depth instead of the files the gate is validating.
    """
    errors: list[str] = []
    for mapping in data.get("localSourceMappings", []):
        path = mapping.get("path")
        symbol = mapping.get("symbol")
        source_needle = mapping.get("sourceNeedle")
        if not isinstance(path, str) or not path:
            errors.append("local source mapping has no path")
            continue
        relative_path = Path(path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"local source mapping is not a repository path: {path}")
            continue
        source = source_root / relative_path
        if not source.is_file():
            errors.append(f"local source mapping is absent in checked-out source: {path}")
            continue
        if not isinstance(symbol, str) or not symbol:
            errors.append(f"local source mapping has no symbol: {path}")
        if not isinstance(source_needle, str) or not source_needle:
            errors.append(f"local source mapping has no source needle: {path}")
            continue
        try:
            source_text = source.read_text(encoding="utf-8")
        except OSError:
            errors.append(f"local source mapping cannot be read in checked-out source: {path}")
            continue
        except UnicodeDecodeError:
            errors.append(f"local source mapping is not UTF-8 source: {path}")
            continue
        if source_needle not in source_text:
            errors.append(
                f"local source mapping is missing {source_needle!r} in checked-out source: {path}"
            )
    return errors


def main() -> int:
    errors: list[str] = []
    raw = MANIFEST.read_bytes()
    if b"\r\n" in raw or not raw.endswith(b"\n"):
        errors.append("manifest must use LF line endings and end with one newline")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        errors.append(f"manifest is not valid JSON: {exc}")
        data = {}

    for path, expected in EXPECTED.items():
        try:
            actual = get(data, path)
        except (KeyError, TypeError):
            errors.append(f"manifest is missing {path}")
            continue
        if actual != expected:
            errors.append(f"{path} must be {expected}, got {actual}")

    for path in ("upstream.commit", "upstream.planningBaseline.commit", "localProvenance.productInferenceRevision"):
        try:
            if not SHA.fullmatch(get(data, path)):
                errors.append(f"{path} must be a lowercase 40-character SHA")
        except (KeyError, TypeError):
            pass

    try:
        validation_mode = get(data, "localProvenance.sourceMappingValidation.mode")
    except (KeyError, TypeError):
        errors.append("manifest is missing localProvenance.sourceMappingValidation.mode")
    else:
        if validation_mode != "checked-out-current-source":
            errors.append("source mappings must validate checked-out current source")
    errors.extend(source_mapping_errors(data))

    for mapping in data.get("upstreamMechanisms", []):
        if not all(isinstance(mapping.get(key), str) and mapping[key] for key in ("path", "symbol", "classification", "finding")):
            errors.append("upstream mechanism mapping must have path, symbol, classification, and finding")

    sidecar = SIDECAR.read_text(encoding="utf-8").strip().split(maxsplit=1)
    digest = hashlib.sha256(raw).hexdigest()
    if len(sidecar) != 2 or sidecar[0] != digest or sidecar[1] != MANIFEST.name:
        errors.append("manifest SHA-256 sidecar is missing or does not match exact manifest bytes")

    audit_text = AUDIT.read_text(encoding="utf-8")
    for required in (MANIFEST.name, "54989ee223611627592f7f9bd925e924658f1f22", "92909d441cfe1cad6693d9eec5cbf6f57a1d8ff4"):
        if required not in audit_text:
            errors.append(f"audit document does not cite required provenance: {required}")

    if errors:
        print("\n".join(errors))
        return 1
    print("SC-20672 VeloxQuant source audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
