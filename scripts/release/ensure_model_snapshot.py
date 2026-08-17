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


def hf_cache_root(snapshot: Path) -> Path | None:
    """Return the hub-cache root owning this path, or None for a plain materialize directory.

    A hub cache lays a repo out as `<root>/models--<org>--<name>/{blobs,refs,snapshots}`, so the
    root is simply the grandparent of the `snapshots/<revision>` directory. Recovering it is what
    makes the cache-resident case self-healing rather than an operator chore: `snapshot_download`
    given `cache_dir=<root>` and NO `local_dir` writes *through* the cache -- blob, ref and
    snapshot symlink together -- and lands the result at exactly this path.

    That is the distinction the Windows CUDA lane fell down on. `local_dir=<this path>` makes
    `huggingface_hub` resolve each blob into the cache (i.e. into this very directory) and then
    copy it to `local_dir`, which is the same file: `shutil.SameFileError`. Anything that did not
    collide would land outside the cache's own bookkeeping.

    The `models--<org>--<name>/snapshots/<revision>` pair is the whole test; a `hub/` ancestor is
    NOT required, because `HF_HUB_CACHE` can put that layout under any directory name.
    """
    parts = snapshot.parts
    for index, part in enumerate(parts):
        if part != "snapshots" or index < 1 or index + 1 >= len(parts):
            continue
        if parts[index - 1].startswith("models--"):
            # `Path(*parts[:0])` is `.`, the right root for a cache named relative to the cwd.
            return Path(*parts[: index - 1])
    return None


def _download_kwargs(model: dict) -> dict:
    """Return the manifest-derived kwargs shared by the cache-heal and plain-directory fetches."""
    kwargs = {
        "repo_id": model["repository"],
        "revision": model["revision"],
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
    #
    # The allow-list is scoped by the MODEL, not by where the bytes land, so a cache-resident
    # target is constrained identically -- an MMAudio-shaped row must not become a whole-repo
    # fetch just because the lane points at a cache.
    allow_patterns = model.get("download_files")
    if allow_patterns:
        kwargs["allow_patterns"] = list(allow_patterns)
    return kwargs


def _heal_in_cache(model: dict, snapshot: Path, cache_root: Path, download: Download) -> bool:
    """Materialize a cache-resident snapshot through the cache itself, then re-verify.

    The refusal below is a BACKSTOP, not the ordinary path: it fires only when the cache-correct
    fetch has already run and the pin is still unsatisfied, so it can no longer be repaired by
    telling the operator to run the fetch by hand.
    """
    print(
        f"materializing {model['repository']}@{model['revision']} through the Hugging Face "
        f"cache at {cache_root} (cache_dir, no local_dir)",
        flush=True,
    )
    try:
        download(cache_dir=str(cache_root), **_download_kwargs(model))
    except Exception as error:
        raise RuntimeError(
            f"{model['key']} snapshot at {snapshot} is cache-resident and the cache-correct "
            f"fetch into {cache_root} failed: {error}. Check that "
            f"{model['repository']}@{model['revision']} is reachable from this runner's "
            "credential and that the volume has room, or point this lane's snapshot variable "
            "at a plain directory outside any models--*/snapshots/* layout."
        ) from error
    # Nothing is written into the snapshot directory by hand -- no revision marker. The cache
    # names the revision in the directory itself, and `verify_snapshot` reads it from there.
    try:
        verify_snapshot(model, snapshot)
    except RuntimeError as error:
        raise RuntimeError(
            f"{model['key']} snapshot at {snapshot} still does not satisfy the pin after "
            f"materializing through the Hugging Face cache at {cache_root}: {error}. The "
            "cache-correct fetch already ran, so this is not the SameFileError shape it "
            f"replaced -- verify that {model['repository']}@{model['revision']} publishes "
            "every path in the manifest's expected_files (and that download_files, if set, "
            "covers them), or point this lane's snapshot variable at a plain directory "
            "outside any models--*/snapshots/* layout."
        ) from error
    return True


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

    # A COMPLETE snapshot already returned above, from `verify_snapshot`. Reaching here means the
    # pinned revision is absent or short of `expected_files`, and WHERE it lives decides how it is
    # fetched. A cache-resident target is materialized through the cache; only a plain directory
    # takes `local_dir`, which is what exploded on the Windows CUDA box when it was aimed at
    # `<cache>/hub/models--*/snapshots/<revision>`.
    cache_root = hf_cache_root(snapshot)
    if cache_root is not None:
        return _heal_in_cache(model, snapshot, cache_root, download)

    snapshot.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"materializing {model['repository']}@{model['revision']} in {snapshot.resolve()}",
        flush=True,
    )
    download(local_dir=str(snapshot), **_download_kwargs(model))
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
