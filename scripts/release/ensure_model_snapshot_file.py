#!/usr/bin/env python3
"""Repair one manifest-required file in a canonical Hugging Face snapshot."""

from __future__ import annotations

import argparse
import fnmatch
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__:
    from .verify_model_snapshot import load_model, verify_snapshot
else:
    from verify_model_snapshot import load_model, verify_snapshot


Download = Callable[..., Any]


def canonical_snapshot_path(cache_root: Path, model: dict) -> Path:
    repository = model["repository"]
    revision = model["revision"]
    if (
        not isinstance(repository, str)
        or repository.count("/") != 1
        or any(part in ("", ".", "..") for part in repository.split("/"))
    ):
        raise RuntimeError(f"{model['key']} repository is not a canonical owner/name")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise RuntimeError(f"{model['key']} revision must be an immutable 40-hex commit")
    if not cache_root.is_absolute():
        raise RuntimeError(f"model cache root must be absolute: {cache_root}")
    repository_cache = "models--" + repository.replace("/", "--")
    return cache_root / repository_cache / "snapshots" / revision


def ensure_expected_file(
    model: dict,
    cache_root: Path,
    relative_file: str,
    download: Download,
) -> Path:
    """Force-fetch one expected file at the manifest pin and verify the complete snapshot."""
    relative = Path(relative_file)
    if (
        not relative_file
        or relative.is_absolute()
        or relative.as_posix() != relative_file
        or "\\" in relative_file
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        raise RuntimeError(
            f"required file must be a normalized relative path: {relative_file!r}"
        )
    expected_files = model.get("expected_files", [])
    if relative_file not in expected_files:
        raise RuntimeError(
            f"{model['key']} does not list {relative_file!r} as an expected file"
        )
    download_files = model.get("download_files")
    if not download_files or not any(
        fnmatch.fnmatchcase(relative_file, pattern) for pattern in download_files
    ):
        raise RuntimeError(
            f"{model['key']} does not permit downloading expected file {relative_file!r}"
        )

    snapshot = canonical_snapshot_path(cache_root, model)
    # Prove the immutable snapshot identity and every other required payload before repairing the
    # named leaf. This keeps a broad or corrupt cache from being blessed by a one-file download.
    reduced_model = {
        **model,
        "expected_files": [name for name in expected_files if name != relative_file],
    }
    verify_snapshot(reduced_model, snapshot)

    downloaded = Path(
        download(
            repo_id=model["repository"],
            filename=relative_file,
            revision=model["revision"],
            cache_dir=str(cache_root),
            token=bool(model.get("requires_auth", False)),
            force_download=True,
        )
    )
    target = snapshot / relative
    if downloaded.absolute() != target.absolute():
        raise RuntimeError(
            f"pinned file materializer returned {downloaded}, expected canonical path {target}"
        )
    verify_snapshot(model, snapshot)
    return target


def download_model_file(**kwargs: Any) -> Any:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required to repair a snapshot file") from error
    return hf_hub_download(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("release/real-weight-models.toml"),
    )
    args = parser.parse_args()
    model = load_model(args.manifest, args.model)
    target = ensure_expected_file(model, args.cache_root, args.file, download_model_file)
    print(f"model snapshot file: OK ({model['key']}@{model['revision']} {target})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
