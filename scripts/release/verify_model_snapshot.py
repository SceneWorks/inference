#!/usr/bin/env python3
"""Verify a runner-provisioned model snapshot against the pinned fixture manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path


MARKER = ".sceneworks-model-revision"
MATERIALIZATION_RECEIPT = ".sceneworks-model-materialization.json"
MATERIALIZATION_INCOMPLETE = ".sceneworks-model-materialization.incomplete"


def expected_materialization_receipt(model: dict) -> dict | None:
    """Return the exact receipt for a pinned alternate materialization source."""
    repository = model.get("materialization_repository")
    revision = model.get("materialization_revision")
    prefix = model.get("materialization_path_prefix")
    if (repository is None) != (revision is None):
        raise RuntimeError(
            f"{model['key']} must declare materialization_repository and "
            "materialization_revision together"
        )
    if repository is None:
        if (
            prefix is not None
            or "materialization_requires_auth" in model
            or "materialization_expected_files" in model
        ):
            raise RuntimeError(
                f"{model['key']} declares alternate materialization settings without a source"
            )
        return None
    if not isinstance(repository, str) or not repository.strip():
        raise RuntimeError(f"{model['key']} materialization_repository is empty")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise RuntimeError(
            f"{model['key']} materialization_revision must be an immutable 40-hex commit"
        )
    if prefix is not None:
        if (
            not isinstance(prefix, str)
            or not prefix
            or prefix.startswith("/")
            or "\\" in prefix
            or any(part in ("", ".", "..") for part in prefix.split("/"))
        ):
            raise RuntimeError(
                f"{model['key']} materialization_path_prefix must be a normalized relative path"
            )
    requires_auth = model.get("materialization_requires_auth", False)
    if not isinstance(requires_auth, bool):
        raise RuntimeError(f"{model['key']} materialization_requires_auth must be a boolean")
    download_files = model.get("download_files")
    if download_files is not None:
        if (
            not isinstance(download_files, list)
            or not download_files
            or any(
                not isinstance(pattern, str)
                or not pattern
                or pattern.startswith("/")
                or "\\" in pattern
                or any(part in ("", ".", "..") for part in pattern.split("/"))
                for pattern in download_files
            )
        ):
            raise RuntimeError(
                f"{model['key']} download_files must be normalized relative patterns"
            )
    expected_files = model.get("materialization_expected_files")
    if (
        not isinstance(expected_files, list)
        or not expected_files
        or any(
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or "\\" in path
            or any(character in path for character in "*?[")
            or any(part in ("", ".", "..") for part in path.split("/"))
            for path in expected_files
        )
        or expected_files != sorted(set(expected_files))
    ):
        raise RuntimeError(
            f"{model['key']} materialization_expected_files must be a sorted, unique list "
            "of normalized exact relative paths"
        )
    return {
        "schema_version": 2,
        "canonical_repository": model["repository"],
        "canonical_revision": model["revision"],
        "materialization_repository": repository,
        "materialization_revision": revision,
        "materialization_path_prefix": prefix,
        "materialization_download_files": download_files,
        "materialization_expected_files": expected_files,
    }


def materialization_payload_files(model: dict, snapshot: Path) -> list[Path]:
    """Return and exact-set-validate every regular alternate-source payload file."""
    root = snapshot.resolve()
    files: list[Path] = []
    provenance_names = {MARKER, MATERIALIZATION_RECEIPT, MATERIALIZATION_INCOMPLETE}
    for item in sorted(root.rglob("*")):
        relative = item.relative_to(root).as_posix()
        if relative in provenance_names or relative.startswith(".cache/"):
            continue
        if item.is_symlink():
            raise RuntimeError(f"materialized payload contains a symlink: {relative}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise RuntimeError(f"materialized payload contains a non-regular entry: {relative}")
        files.append(item)
    if not files:
        raise RuntimeError(f"materialized payload has no regular files: {root}")
    expected = model.get("materialization_expected_files")
    actual = [item.relative_to(root).as_posix() for item in files]
    if expected is not None and actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise RuntimeError(
            f"{model['key']} materialized payload does not match its exact projected file set; "
            + "; ".join(details)
        )
    return files


def materialization_payload_inventory(model: dict, snapshot: Path) -> dict:
    """Bind every regular payload file in an alternate-source materialization."""
    root = snapshot.resolve()
    files = [
        {
            "path": item.relative_to(root).as_posix(),
            "size": item.stat().st_size,
            "sha256": _sha256(item),
        }
        for item in materialization_payload_files(model, root)
    ]
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "materialization_files": files,
        "materialization_inventory_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def completed_materialization_receipt(model: dict, snapshot: Path) -> dict:
    """Return the exact identity plus content inventory for a completed materialization."""
    expected = expected_materialization_receipt(model)
    if expected is None:
        raise RuntimeError(f"{model['key']} does not declare an alternate materialization source")
    return {**expected, **materialization_payload_inventory(model, snapshot)}


def _is_standard_hf_canonical_snapshot(model: dict, snapshot: Path) -> bool:
    """Return whether a path is the pinned repo's standard Hugging Face cache location."""
    repository_cache = "models--" + model["repository"].replace("/", "--")
    return (
        snapshot.name == model["revision"]
        and snapshot.parent.name == "snapshots"
        and snapshot.parent.parent.name == repository_cache
    )


def verify_materialization_provenance(
    model: dict, snapshot: Path, *, required: bool = False
) -> None:
    """Validate an alternate-source receipt or a pristine canonical HF cache path.

    Existing operator caches predate materialization receipts. A standard
    ``snapshots/<canonical revision>`` directory without our marker remains valid canonical
    evidence. Any directory written by ``ensure_model_snapshot.py`` carries the marker and must
    also carry the exact alternate-source receipt when this stricter gate is requested.
    """
    expected = expected_materialization_receipt(model)
    snapshot = snapshot.resolve()
    incomplete_path = snapshot / MATERIALIZATION_INCOMPLETE
    if incomplete_path.is_symlink() or incomplete_path.exists():
        raise RuntimeError(f"{model['key']} has an incomplete model materialization")
    receipt_path = snapshot / MATERIALIZATION_RECEIPT
    if receipt_path.is_symlink() or (receipt_path.exists() and not receipt_path.is_file()):
        raise RuntimeError(f"{model['key']} has an unsafe materialization receipt")
    if receipt_path.is_file():
        try:
            actual = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"{model['key']} has an invalid materialization receipt: {error}"
            ) from error
        if expected is None:
            raise RuntimeError(
                f"{model['key']} has an unexpected alternate-source materialization receipt"
            )
        identity_keys = set(expected)
        inventory_keys = {"materialization_files", "materialization_inventory_sha256"}
        if not isinstance(actual, dict) or set(actual) != identity_keys | inventory_keys:
            raise RuntimeError(f"{model['key']} materialization receipt has an invalid schema")
        if {key: actual[key] for key in identity_keys} != expected:
            raise RuntimeError(f"{model['key']} materialization receipt does not match the pin")
        inventory = materialization_payload_inventory(model, snapshot)
        if any(actual[key] != inventory[key] for key in inventory_keys):
            raise RuntimeError(
                f"{model['key']} materialized payload does not match its receipt inventory"
            )
        return
    if not required:
        return
    if _is_standard_hf_canonical_snapshot(model, snapshot) and not (snapshot / MARKER).exists():
        return
    raise RuntimeError(
        f"{model['key']} requires either an exact alternate-source materialization receipt "
        f"or a pristine snapshots/{model['revision']} canonical cache"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_inventory(model: dict, snapshot: Path) -> dict:
    """Return a canonical content inventory after validating the pinned snapshot claim."""
    verify_snapshot(model, snapshot)
    root = snapshot.resolve()
    files = []
    for item in sorted(root.rglob("*")):
        if not (item.is_file() or item.is_symlink()):
            continue
        relative = item.relative_to(root).as_posix()
        if relative in (
            MARKER,
            MATERIALIZATION_RECEIPT,
            MATERIALIZATION_INCOMPLETE,
        ) or relative.startswith(".cache/"):
            continue
        try:
            size = item.stat().st_size
            digest = _sha256(item)
        except OSError as error:
            raise RuntimeError(f"cannot inventory model file {item}: {error}") from error
        files.append(
            {
                "path": item.relative_to(root).as_posix(),
                "kind": "symlink" if item.is_symlink() else "file",
                "size": size,
                "sha256": digest,
            }
        )
    if not files:
        raise RuntimeError(f"snapshot has no files to inventory: {root}")
    content_projection = [
        {"path": item["path"], "size": item["size"], "sha256": item["sha256"]}
        for item in files
    ]
    encoded = json.dumps(content_projection, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "model": model["key"],
        "repository": model["repository"],
        "revision": model["revision"],
        "inventory_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def load_model(manifest: Path, key: str) -> dict:
    policy = tomllib.loads(manifest.read_text(encoding="utf-8"))
    matches = [model for model in policy.get("models", []) if model["key"] == key]
    if len(matches) != 1:
        raise RuntimeError(f"expected one model policy for {key!r}, found {len(matches)}")
    return matches[0]


def _verified_snapshot_root(model: dict, snapshot: Path) -> Path:
    if snapshot.is_symlink():
        raise RuntimeError(f"{model['key']} has an unsafe snapshot directory symlink")
    snapshot = snapshot.resolve()
    if not snapshot.is_dir():
        raise RuntimeError(f"snapshot directory does not exist: {snapshot}")
    marker = snapshot / MARKER
    actual_revision = snapshot.name
    if marker.is_symlink() or (marker.exists() and not marker.is_file()):
        raise RuntimeError(f"{model['key']} has an unsafe revision marker")
    if marker.is_file():
        actual_revision = marker.read_text(encoding="utf-8").strip()
    if actual_revision != model["revision"]:
        raise RuntimeError(
            f"{model['key']} revision mismatch: {actual_revision!r} != {model['revision']!r}; "
            f"use a standard HF snapshots/<revision> path or add {MARKER}"
        )
    return snapshot


def verify_model_payload_files(model: dict, root: Path) -> None:
    """Verify required model files and every shard referenced by a weight index."""
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"model payload directory does not exist: {root}")
    missing = [name for name in model["expected_files"] if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"{model['key']} snapshot is incomplete; missing: {', '.join(missing)}")
    for name in model["expected_files"]:
        if not name.endswith(".index.json"):
            continue
        index_path = root / name
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
            if not isinstance(weight_map, dict):
                raise TypeError("weight_map is not an object")
            shards = set(weight_map.values())
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError(
                f"{model['key']} has an invalid weight index {name}: {error}"
            ) from error
        if not shards or not all(isinstance(shard, str) for shard in shards):
            raise RuntimeError(
                f"{model['key']} has an invalid weight index {name}: empty/non-string weight_map"
            )
        component = index_path.parent
        missing_shards = [shard for shard in sorted(shards) if not (component / shard).is_file()]
        if missing_shards:
            raise RuntimeError(
                f"{model['key']} snapshot is incomplete; {name} references missing shards: "
                + ", ".join(missing_shards)
            )


def verify_snapshot_payload(model: dict, snapshot: Path) -> None:
    """Verify the pinned identity and required model payload, ignoring in-progress provenance.

    Only the materializer calls this while its explicit incomplete marker is present. Public
    consumers must use :func:`verify_snapshot`, which also requires a completed provenance state.
    """
    snapshot = _verified_snapshot_root(model, snapshot)
    verify_model_payload_files(model, snapshot)


def verify_snapshot(
    model: dict,
    snapshot: Path,
    *,
    require_materialization_provenance: bool = False,
) -> None:
    snapshot = _verified_snapshot_root(model, snapshot)
    verify_materialization_provenance(
        model,
        snapshot,
        required=require_materialization_provenance,
    )
    verify_snapshot_payload(model, snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model key in real-weight-models.toml")
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("release/real-weight-models.toml"),
    )
    parser.add_argument(
        "--inventory-output",
        type=Path,
        help="write a canonical dereferenced content inventory for evidence binding",
    )
    args = parser.parse_args()
    model = load_model(args.manifest, args.model)
    verify_snapshot(model, args.snapshot)
    suffix = ""
    if args.inventory_output is not None:
        inventory = snapshot_inventory(model, args.snapshot)
        args.inventory_output.parent.mkdir(parents=True, exist_ok=True)
        args.inventory_output.write_text(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        suffix = f" inventory_sha256={inventory['inventory_sha256']}"
    print(f"model snapshot: OK ({model['key']}@{model['revision']}){suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
