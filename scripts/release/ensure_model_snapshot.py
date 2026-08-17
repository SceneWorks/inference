#!/usr/bin/env python3
"""Materialize and verify an immutable model snapshot when a runner cache is empty."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__:
    from .verify_model_snapshot import MARKER, load_model, snapshot_path, verify_snapshot
else:
    from verify_model_snapshot import MARKER, load_model, snapshot_path, verify_snapshot


Download = Callable[..., Any]


def is_hf_cache_snapshot(snapshot: Path) -> bool:
    """Return true when the path is a Hugging Face cache's own `models--*/snapshots/<revision>`.

    The hub cache owns that layout: blobs live in a sibling `blobs/`, `snapshots/<revision>/`
    holds symlinks into them, and `refs/` records what resolves where. Downloading into it with
    `local_dir` pointed at the snapshot directory makes `huggingface_hub` copy each freshly
    cached blob onto itself (`shutil.SameFileError`), and any file that did not collide would be
    written outside the cache's bookkeeping.

    The `models--<org>--<name>/snapshots/<revision>` pair is the whole test; a `hub/` ancestor is
    NOT required, because `HF_HUB_CACHE` can put that layout under any directory name.
    """
    parts = snapshot.parts
    return any(
        part == "snapshots"
        and index >= 1
        and index + 1 < len(parts)
        and parts[index - 1].startswith("models--")
        for index, part in enumerate(parts)
    )


def ensure_snapshot(model: dict, snapshot: Path, download: Download) -> bool:
    """Return true after downloading, or false when an existing snapshot is valid."""
    try:
        verify_snapshot(model, snapshot)
        return False
    except RuntimeError as initial_error:
        # `except ... as` unbinds the name at the end of the clause, so the reason has to be
        # carried out by hand for the refusal below to be able to name what was wrong.
        shortfall = str(initial_error)
        if snapshot.exists() and not snapshot.is_dir():
            raise initial_error
        if snapshot.is_dir():
            marker = snapshot / MARKER
            actual_revision = snapshot.name
            if marker.is_file():
                actual_revision = marker.read_text(encoding="utf-8").strip()
            if actual_revision != model["revision"]:
                raise initial_error

    # A COMPLETE cache-resident snapshot already returned above, from `verify_snapshot`. Reaching
    # here with a cache path means the pinned revision is absent or short of `expected_files`, and
    # the download that would follow is the one that exploded on the Windows CUDA box: `local_dir`
    # aimed at `<cache>/hub/models--*/snapshots/<revision>` resolved the blob into that very
    # directory and then copied it onto itself. Refuse with the missing set instead -- the cache is
    # `huggingface_hub`'s to write, and a lane that spent hours materializing into it would leave
    # an unreferenced snapshot behind either way.
    if is_hf_cache_snapshot(snapshot):
        raise RuntimeError(
            f"{model['key']} snapshot path is inside a Hugging Face cache "
            f"({snapshot}) and does not satisfy the pin: {shortfall}. "
            "Refusing to download into the cache's own snapshots directory -- "
            "huggingface_hub would copy each blob onto itself (shutil.SameFileError) and "
            "bypass the cache's blobs/refs bookkeeping. Either populate the cache on the "
            f"runner with `huggingface-cli download {model['repository']} "
            f"--revision {model['revision']}`, which writes through it, or point this lane's "
            "snapshot variable at a plain directory outside any models--*/snapshots/* layout "
            "and let this script materialize it."
        )

    snapshot.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"materializing {model['repository']}@{model['revision']} in {snapshot.resolve()}",
        flush=True,
    )
    download_kwargs = {
        "repo_id": model["repository"],
        "revision": model["revision"],
        "local_dir": str(snapshot),
        # Public release fixtures stay explicitly anonymous. Gated checkpoints
        # opt in to the runner's configured Hugging Face credential without
        # placing the token in the workflow or command line.
        "token": bool(model.get("requires_auth", False)),
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
    parser.add_argument("--snapshot", required=True, type=snapshot_path)
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
