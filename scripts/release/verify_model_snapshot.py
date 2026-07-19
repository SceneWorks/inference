#!/usr/bin/env python3
"""Verify a runner-provisioned model snapshot against the pinned fixture manifest."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path


MARKER = ".sceneworks-model-revision"


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
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("release/real-weight-models.toml"),
    )
    args = parser.parse_args()
    model = load_model(args.manifest, args.model)
    verify_snapshot(model, args.snapshot)
    print(f"model snapshot: OK ({model['key']}@{model['revision']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
