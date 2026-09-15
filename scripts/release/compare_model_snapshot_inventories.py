#!/usr/bin/env python3
"""Fail closed unless two independently generated model inventories name identical bytes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SHA256 = re.compile(r"^[0-9a-f]{64}$")
IDENTITY_KEYS = ("schema_version", "model", "repository", "revision", "inventory_sha256")


def load_inventory(path: Path, expected_model: str) -> dict:
    metadata = path.lstat()
    if path.is_symlink() or not path.is_file() or metadata.st_size == 0:
        raise RuntimeError(f"inventory is not a non-empty regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"inventory is not an object: {path}")
    if set(value) != {*IDENTITY_KEYS, "files"}:
        raise RuntimeError(f"inventory has an unexpected schema: {path}")
    if value["schema_version"] != 1 or value["model"] != expected_model:
        raise RuntimeError(f"inventory identity mismatch: {path}")
    if not isinstance(value["repository"], str) or not value["repository"]:
        raise RuntimeError(f"inventory repository is missing: {path}")
    if not isinstance(value["revision"], str) or re.fullmatch(r"[0-9a-f]{40}", value["revision"]) is None:
        raise RuntimeError(f"inventory revision is not a pinned commit: {path}")
    if not isinstance(value["inventory_sha256"], str) or SHA256.fullmatch(value["inventory_sha256"]) is None:
        raise RuntimeError(f"inventory digest is malformed: {path}")
    if not isinstance(value["files"], list) or not value["files"]:
        raise RuntimeError(f"inventory file list is empty: {path}")
    return value


def compare(expected: Path, actual: Path, model: str) -> str:
    expected_value = load_inventory(expected, model)
    actual_value = load_inventory(actual, model)
    expected_identity = {key: expected_value[key] for key in IDENTITY_KEYS}
    actual_identity = {key: actual_value[key] for key in IDENTITY_KEYS}
    if actual_identity != expected_identity:
        raise RuntimeError(
            f"{model} inventory identity differs across native hosts: "
            f"expected {expected_identity!r}, found {actual_identity!r}"
        )
    return expected_value["inventory_sha256"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    digest = compare(args.expected, args.actual, args.model)
    print(f"{args.model} native-host inventory parity: {digest}")


if __name__ == "__main__":
    main()
