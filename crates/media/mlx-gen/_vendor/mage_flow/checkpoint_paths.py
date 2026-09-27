"""Validate sharded checkpoint paths before a loader opens index-controlled files.

Accelerate 1.13.0 joins ``weight_map`` values without containment checks. MageFlow
preflights its own index loader and the Qwen ``from_pretrained`` entry point here.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path, PureWindowsPath


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _hub_blob_root(directory: Path) -> Path | None:
    """Recognize the immutable snapshot layout used by huggingface_hub."""
    for parent in (directory, *directory.parents):
        if parent.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40,64}", parent.name):
            blobs = parent.parent.parent / "blobs"
            if blobs.is_dir() and not blobs.is_symlink():
                return blobs.resolve(strict=True)
    return None


def resolve_checkpoint_shard(directory: str | os.PathLike, name: str) -> Path:
    """Return an authorized regular shard, including Hub snapshot blob symlinks."""
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"checkpoint directory is not a directory: {directory}")
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError(f"invalid checkpoint shard path: {name!r}")
    parts = name.split("/")
    if (name.startswith("/") or PureWindowsPath(name).drive or
            any(part in ("", ".", "..") for part in parts)):
        raise ValueError(f"checkpoint shard must be a relative path inside its directory: {name!r}")
    candidate = root.joinpath(*parts)
    try:
        resolved = candidate.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise ValueError(f"checkpoint shard is missing or unreadable: {name!r}") from exc
    blob_root = _hub_blob_root(root)
    if not _within(resolved, root) and (blob_root is None or not _within(resolved, blob_root)):
        raise ValueError(f"checkpoint shard symlink escapes authorized roots: {name!r}")
    if not stat.S_ISREG(mode):
        raise ValueError(f"checkpoint shard is not a regular file: {name!r}")
    return resolved


def validate_checkpoint_index(index_path: str | os.PathLike) -> list[Path]:
    """Validate every named shard in an index before any shard is loaded."""
    index_path = Path(index_path)
    authorized_index = resolve_checkpoint_shard(index_path.parent, index_path.name)
    try:
        with authorized_index.open(encoding="utf-8") as handle:
            index = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid checkpoint index: {index_path}") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"checkpoint index has no weight_map: {index_path}")
    if not all(isinstance(key, str) and key and isinstance(name, str)
               for key, name in weight_map.items()):
        raise ValueError(f"checkpoint index has invalid weight_map entries: {index_path}")
    return [resolve_checkpoint_shard(index_path.parent, name)
            for name in sorted(set(weight_map.values()))]


def preflight_checkpoint_directory(directory: str | os.PathLike) -> None:
    """Check all sharded indexes in a local model directory before HF loading."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"checkpoint directory is not a directory: {directory}")
    for index_path in sorted(root.glob("*.index.json")):
        validate_checkpoint_index(index_path)


def preflight_model_source(
    source: str | os.PathLike, *, revision: str | None = None, subfolder: str = ""
) -> str:
    """Resolve a local or Hub model source, then preflight any shard index."""
    source = os.fspath(source)
    if os.path.isdir(source):
        directory = source
    else:
        from huggingface_hub import snapshot_download

        directory = snapshot_download(repo_id=source, revision=revision)
    model_root = Path(directory).resolve(strict=True)
    if subfolder:
        parts = subfolder.split("/")
        if (subfolder.startswith("/") or "\\" in subfolder or PureWindowsPath(subfolder).drive
                or any(part in ("", ".", "..") for part in parts)):
            raise ValueError(f"invalid model subfolder: {subfolder!r}")
        checkpoint_dir = model_root.joinpath(*parts).resolve(strict=True)
        if not _within(checkpoint_dir, model_root):
            raise ValueError(f"model subfolder escapes checkpoint directory: {subfolder!r}")
    else:
        checkpoint_dir = model_root
    preflight_checkpoint_directory(checkpoint_dir)
    return str(model_root)
