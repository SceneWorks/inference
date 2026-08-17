#!/usr/bin/env python3
"""Verify a runner-provisioned model snapshot against the pinned fixture manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path


MARKER = ".sceneworks-model-revision"


def snapshot_path(value: str) -> Path:
    """Expand a leading `~` so a per-user snapshot variable resolves on any runner account.

    The workflow's snapshot variables are moving to the `~/`-relative convention documented in
    `resolve_snapshot_paths.py`: one repository variable names the same location on both macOS
    boxes, whose homes differ (`/Users/michael` vs `/Users/MTrefry`), so a baked absolute value is
    only correct on whichever box happens to claim the job. That workflow step expands the value
    for the rest of a job, but these scripts are also run by hand and by an operator reproducing a
    lane, and an unexpanded `~` reaching `Path` becomes a literal directory named `~` under the
    checkout rather than an error naming the real problem.
    """
    return Path(value).expanduser()


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
        if relative == MARKER or relative.startswith(".cache/"):
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


def verify_snapshot(model: dict, snapshot: Path) -> None:
    snapshot = snapshot.resolve()
    if not snapshot.is_dir():
        raise RuntimeError(f"snapshot directory does not exist: {snapshot}")
    marker = snapshot / MARKER
    actual_revision = snapshot.name
    if marker.is_file():
        actual_revision = marker.read_text(encoding="utf-8").strip()
    if actual_revision != model["revision"]:
        raise RuntimeError(
            f"{model['key']} revision mismatch: {actual_revision!r} != {model['revision']!r}; "
            f"use a standard HF snapshots/<revision> path or add {MARKER}"
        )
    missing = [name for name in model["expected_files"] if not (snapshot / name).is_file()]
    if missing:
        raise RuntimeError(f"{model['key']} snapshot is incomplete; missing: {', '.join(missing)}")
    for name in model["expected_files"]:
        if not name.endswith(".index.json"):
            continue
        index_path = snapshot / name
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
            if not isinstance(weight_map, dict):
                raise TypeError("weight_map is not an object")
            shards = set(weight_map.values())
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise RuntimeError(f"{model['key']} has an invalid weight index {name}: {error}") from error
        if not shards or not all(isinstance(shard, str) for shard in shards):
            raise RuntimeError(f"{model['key']} has an invalid weight index {name}: empty/non-string weight_map")
        component = index_path.parent
        missing_shards = [shard for shard in sorted(shards) if not (component / shard).is_file()]
        if missing_shards:
            raise RuntimeError(
                f"{model['key']} snapshot is incomplete; {name} references missing shards: "
                + ", ".join(missing_shards)
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model key in real-weight-models.toml")
    parser.add_argument("--snapshot", required=True, type=snapshot_path)
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
