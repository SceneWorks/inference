"""Publisher-bound asset verification and cache-preserving provision helpers for SC-23935."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

DEFAULT_CLOSURE = Path(__file__).resolve().parents[2] / "release/qwen38-bonsai-artifacts.json"


def frozen_model(model: dict, closure: Path = DEFAULT_CLOSURE) -> dict:
    document = json.loads(closure.read_text(encoding="utf-8"))
    frozen = document["models"][model["key"]]
    if document.get("schema_version") != 1 or any(
        frozen.get(key) != model.get(key) for key in ("repository", "revision")
    ):
        raise ValueError("frozen publisher closure does not match model identity")
    seen = set()
    for item in frozen["files"]:
        path = item["path"]
        if path in seen or Path(path).is_absolute() or ".." in Path(path).parts or "\\" in path:
            raise ValueError("frozen closure contains duplicate or unsafe paths")
        seen.add(path)
        sha = item["sha256"]
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("invalid frozen SHA256")
        lfs, blob = item.get("lfs_sha256"), item.get("git_blob_sha1")
        if lfs:
            if lfs != sha or blob is not None:
                raise ValueError("frozen LFS identity does not match SHA256")
        elif not isinstance(blob, str) or len(blob) != 40 or any(c not in "0123456789abcdef" for c in blob):
            raise ValueError("frozen file lacks publisher Git/LFS identity")
    return frozen


def verify_file(path: Path, expected: dict) -> dict:
    size = path.stat().st_size
    if size != expected["bytes"]:
        raise ValueError(f"publisher size mismatch: {expected['path']}")
    digest = hashlib.sha256()
    git = hashlib.sha1(f"blob {size}\0".encode()) if expected.get("git_blob_sha1") else None
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            if git is not None:
                git.update(chunk)
    actual = digest.hexdigest()
    if actual != expected["sha256"]:
        raise ValueError(f"publisher SHA256 mismatch: {expected['path']}; existing file retained")
    if git is not None and git.hexdigest() != expected["git_blob_sha1"]:
        raise ValueError(f"publisher Git blob mismatch: {expected['path']}")
    return {"path": expected["path"], "bytes": size, "sha256": actual,
            "publisher_identity_verified": True}


def selected_files(frozen: dict, cells: list[dict]) -> list[dict]:
    """Only admitted GGUF variants are eligible; support/license files accompany them."""
    if not any("language_variant" in cell for cell in cells):
        return list(frozen["files"])
    languages = {cell["language_variant"].upper() for cell in cells}
    visions = {cell["vision_variant"].upper() for cell in cells}
    return [item for item in frozen["files"] if not item["path"].endswith(".gguf") or any(
        item["path"] == f"Ternary-Bonsai-2-27B-{language}_0.gguf" for language in languages
    ) or any(item["path"] == f"Ternary-Bonsai-2-27B-mmproj-{vision}{'_0' if vision == 'Q8' else ''}.gguf"
             for vision in visions)]


def provision_snapshot(model: dict, snapshot: Path, files: list[dict], download: Callable[..., Any]) -> list[dict]:
    """Verify existing files first, fetch only missing pinned files, then verify publisher hashes.

    Corrupt existing files fail closed and are retained. No cache eviction or replacement occurs.
    Hugging Face owns resume of missing/incomplete downloads and its normal cache bookkeeping.
    """
    expected_repo = "models--" + model["repository"].replace("/", "--")
    if snapshot.name != model["revision"] or snapshot.parent.name != "snapshots" or snapshot.parent.parent.name != expected_repo:
        raise ValueError("provisioning requires the exact publisher repository/revision HF cache path")
    inventory = []
    missing = []
    for item in files:
        path = snapshot / item["path"]
        if path.exists():
            inventory.append(verify_file(path, item))
        elif path.is_symlink():
            raise ValueError(f"broken cache symlink retained: {item['path']}")
        else:
            missing.append(item)
    required = sum(item["bytes"] for item in missing)
    existing_parent = snapshot
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    import shutil
    if shutil.disk_usage(existing_parent).free < required:
        raise ValueError("insufficient disk space for selected missing pinned assets")
    for item in missing:
        downloaded = Path(download(repo_id=model["repository"], revision=model["revision"],
                                   filename=item["path"], cache_dir=str(snapshot.parent.parent.parent), token=False))
        expected_path = snapshot / item["path"]
        if downloaded.absolute() != expected_path.absolute():
            raise ValueError("publisher download returned an unexpected snapshot path")
        inventory.append(verify_file(expected_path, item))
    return sorted(inventory, key=lambda item: item["path"])


def verify_inventory(model: dict, inventory: dict, closure: Path = DEFAULT_CLOSURE) -> None:
    """Bind the already hashed native-run inventory to the independently frozen publisher closure."""
    frozen = frozen_model(model, closure)
    actual = {item["path"]: item for item in inventory["files"]}
    for expected in frozen["files"]:
        row = actual.get(expected["path"], {})
        if row.get("size") != expected["bytes"] or row.get("sha256") != expected["sha256"]:
            raise ValueError(f"native snapshot differs from frozen publisher identity: {expected['path']}")
