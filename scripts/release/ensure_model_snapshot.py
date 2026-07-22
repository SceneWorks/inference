#!/usr/bin/env python3
"""Materialize and verify an immutable model snapshot when a runner cache is empty."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__:
    from .verify_model_snapshot import MARKER, load_model, verify_snapshot
else:
    from verify_model_snapshot import MARKER, load_model, verify_snapshot


Download = Callable[..., Any]


def ensure_snapshot(model: dict, snapshot: Path, download: Download) -> bool:
    """Return true after downloading, or false when an existing snapshot is valid."""
    try:
        verify_snapshot(model, snapshot)
        return False
    except RuntimeError as initial_error:
        if snapshot.exists() and not snapshot.is_dir():
            raise initial_error
        if snapshot.is_dir():
            marker = snapshot / MARKER
            actual_revision = snapshot.name
            if marker.is_file():
                actual_revision = marker.read_text(encoding="utf-8").strip()
            if actual_revision != model["revision"]:
                raise initial_error

    snapshot.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"materializing {model['repository']}@{model['revision']} in {snapshot.resolve()}",
        flush=True,
    )
    download_kwargs = {
        "repo_id": model["repository"],
        "revision": model["revision"],
        "local_dir": str(snapshot),
        "token": False,
    }
    # Optional per-model download allow-list. When set, materialize ONLY these repo-relative paths
    # (snapshot_download `allow_patterns`) instead of the whole repo — for repos whose pinned
    # checkpoints are a small fraction of a large repo (e.g. `hkchengrex/MMAudio` ships ~46 GB of
    # training checkpoints + weight variants the inference stack never loads). Absent ⇒ whole-repo
    # download, the default for every other model. `verify_snapshot` still enforces `expected_files`,
    # so an under-fetch (a needed file left off the list) fails loudly right after download.
    allow_patterns = model.get("download_files")
    if allow_patterns:
        download_kwargs["allow_patterns"] = list(allow_patterns)
    download(**download_kwargs)
    (snapshot / MARKER).write_text(model["revision"] + "\n", encoding="utf-8")
    try:
        verify_snapshot(model, snapshot)
    except RuntimeError as error:
        raise RuntimeError(f"downloaded snapshot failed verification: {error}") from error
    return True


def download_snapshot(**kwargs: Any) -> Any:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required only when a pinned snapshot is absent"
        ) from error
    return snapshot_download(**kwargs)


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
    downloaded = ensure_snapshot(model, args.snapshot, download_snapshot)
    source = "downloaded" if downloaded else "cached"
    print(f"model snapshot: OK ({model['key']}@{model['revision']}, {source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
