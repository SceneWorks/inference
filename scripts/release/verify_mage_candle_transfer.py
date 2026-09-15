#!/usr/bin/env python3
"""Write or verify the exact hash manifest for the transferred Candle Mage bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


MANIFEST = "mage_candle_transfer_manifest.json"
FILES = (
    "mage_flow_te_golden.safetensors",
    "mage_flow_dit_golden.safetensors",
    "mage_flow_vae_f32_1024.safetensors",
    "mage_flow_e2e_golden.safetensors",
    "mage_flow_edit_golden.safetensors",
    "mage_flow_edit_base_golden.safetensors",
    "mage_flow_edit_turbo_golden.safetensors",
    "mage_edit_oracle_manifest.json",
    "mage_edit_variants_manifest.json",
    "mage_candle_oracles_manifest.json",
)
REVISION_MARKER = ".sceneworks-model-revision"


class InvalidTransfer(RuntimeError):
    pass


def require_exclusive_regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise InvalidTransfer(f"{label} must be a regular, non-symlink file")
    try:
        link_count = path.stat().st_nlink
    except OSError as error:
        raise InvalidTransfer(f"cannot inspect {label}: {error}") from error
    if link_count != 1:
        raise InvalidTransfer(f"{label} must have exactly one hard link, found {link_count}")


def revision(path: Path) -> str:
    resolved = path.resolve()
    marker = resolved / REVISION_MARKER
    value = marker.read_text(encoding="utf-8").strip() if marker.is_file() else resolved.name
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise InvalidTransfer(f"snapshot is not pinned to a 40-hex revision: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document(
    output: Path,
    generation_snapshot: Path,
    edit_snapshot: Path,
    edit_base_snapshot: Path,
    edit_turbo_snapshot: Path,
) -> dict:
    records = []
    for name in FILES:
        path = output / name
        if not path.is_file():
            raise InvalidTransfer(f"transferred Mage oracle is missing: {path}")
        size = path.stat().st_size
        if type(size) is not int or size <= 0:
            raise InvalidTransfer(f"transferred Mage oracle is empty: {path}")
        records.append({"name": name, "bytes": size, "sha256": sha256(path)})
    return {
        "schema": 1,
        "generationSnapshotRevision": revision(generation_snapshot),
        "editSnapshotRevision": revision(edit_snapshot),
        "editBaseSnapshotRevision": revision(edit_base_snapshot),
        "editTurboSnapshotRevision": revision(edit_turbo_snapshot),
        "files": records,
    }


def validate_manifest(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "generationSnapshotRevision",
        "editSnapshotRevision",
        "editBaseSnapshotRevision",
        "editTurboSnapshotRevision",
        "files",
    }:
        raise InvalidTransfer("Candle transfer manifest header population is stale")
    if type(value["schema"]) is not int or value["schema"] != 1:
        raise InvalidTransfer("Candle transfer manifest schema is stale")
    for key in (
        "generationSnapshotRevision",
        "editSnapshotRevision",
        "editBaseSnapshotRevision",
        "editTurboSnapshotRevision",
    ):
        if not isinstance(value[key], str) or re.fullmatch(r"[0-9a-f]{40}", value[key]) is None:
            raise InvalidTransfer(f"Candle transfer manifest {key} is stale")
    records = value["files"]
    if (
        not isinstance(records, list)
        or len(records) != len(FILES)
        or any(
            not isinstance(record, dict)
            or set(record) != {"name", "bytes", "sha256"}
            or type(record["name"]) is not str
            or type(record["bytes"]) is not int
            or record["bytes"] <= 0
            or type(record["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
            for record in records
        )
        or [record["name"] for record in records] != list(FILES)
    ):
        raise InvalidTransfer("Candle transfer manifest file population is stale")
    return value


def verify(
    output: Path,
    generation_snapshot: Path,
    edit_snapshot: Path,
    edit_base_snapshot: Path,
    edit_turbo_snapshot: Path,
    write_manifest: bool,
) -> None:
    actual = document(
        output,
        generation_snapshot,
        edit_snapshot,
        edit_base_snapshot,
        edit_turbo_snapshot,
    )
    manifest_path = output / MANIFEST
    if write_manifest:
        manifest_path.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
    try:
        expected = validate_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidTransfer(f"invalid Candle transfer manifest: {error}") from error
    if expected != actual:
        raise InvalidTransfer("Candle transfer manifest revisions/population/hashes are stale")
    print(f"verified exact {len(FILES)}-file Candle Mage transfer bundle under {output}")


def migrate_edit_variant_manifest_hash_only(
    output: Path,
    generation_snapshot: Path,
    edit_snapshot: Path,
    edit_base_snapshot: Path,
    edit_turbo_snapshot: Path,
) -> None:
    """Rebind only the nested edit-variant manifest after its strict migration."""
    actual = document(
        output,
        generation_snapshot,
        edit_snapshot,
        edit_base_snapshot,
        edit_turbo_snapshot,
    )
    manifest_path = output / MANIFEST
    require_exclusive_regular_file(manifest_path, "Candle transfer manifest")
    try:
        expected = validate_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidTransfer(f"invalid Candle transfer manifest: {error}") from error

    target = "mage_edit_variants_manifest.json"
    if {
        key: value for key, value in expected.items() if key != "files"
    } != {
        key: value for key, value in actual.items() if key != "files"
    }:
        raise InvalidTransfer("legacy Candle transfer manifest revisions are not exact")
    expected_files = {record["name"]: record for record in expected["files"]}
    actual_files = {record["name"]: record for record in actual["files"]}
    mismatches = {
        name for name in FILES if expected_files[name] != actual_files[name]
    }
    if mismatches != {target}:
        raise InvalidTransfer(
            "legacy Candle transfer manifest must differ only at "
            f"{target}, found {sorted(mismatches)}"
        )

    migrated = dict(expected)
    migrated["files"] = [actual_files[name] for name in FILES]
    manifest_path.write_text(json.dumps(migrated, indent=2) + "\n", encoding="utf-8")
    verify(
        output,
        generation_snapshot,
        edit_snapshot,
        edit_base_snapshot,
        edit_turbo_snapshot,
        False,
    )
    print(f"migrated only {target} in {manifest_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen", required=True, type=Path)
    parser.add_argument("--edit", required=True, type=Path)
    parser.add_argument("--edit-base", required=True, type=Path)
    parser.add_argument("--edit-turbo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write-manifest", action="store_true")
    mode.add_argument("--migrate-edit-variant-manifest-hash-only", action="store_true")
    args = parser.parse_args()
    try:
        paths = (
            args.output.resolve(),
            args.gen.resolve(),
            args.edit.resolve(),
            args.edit_base.resolve(),
            args.edit_turbo.resolve(),
        )
        if args.migrate_edit_variant_manifest_hash_only:
            migrate_edit_variant_manifest_hash_only(*paths)
        else:
            verify(*paths, args.write_manifest)
    except InvalidTransfer as error:
        print(f"Mage Candle transfer verification FAILED: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
