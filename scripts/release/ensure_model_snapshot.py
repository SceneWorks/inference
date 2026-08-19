#!/usr/bin/env python3
"""Materialize and verify an immutable model snapshot when a runner cache is empty."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

if __package__:
    from .verify_model_snapshot import (
        MARKER,
        MATERIALIZATION_INCOMPLETE,
        MATERIALIZATION_RECEIPT,
        completed_materialization_receipt,
        expected_materialization_receipt,
        load_model,
        materialization_payload_files,
        snapshot_path,
        verify_model_payload_files,
        verify_snapshot,
        verify_snapshot_payload,
    )
else:
    from verify_model_snapshot import (
        MARKER,
        MATERIALIZATION_INCOMPLETE,
        MATERIALIZATION_RECEIPT,
        completed_materialization_receipt,
        expected_materialization_receipt,
        load_model,
        materialization_payload_files,
        snapshot_path,
        verify_model_payload_files,
        verify_snapshot,
        verify_snapshot_payload,
    )


Download = Callable[..., Any]


def hf_cache_location(snapshot: Path) -> tuple[Path, str, str] | None:
    """Split a hub-cache path into `(root, repo_directory, revision_directory)`, else None.

    A hub cache lays a repo out as `<root>/models--<org>--<name>/{blobs,refs,snapshots}`, so the
    root is simply the grandparent of the `snapshots/<revision>` directory. Recovering it is what
    makes the cache-resident case self-healing rather than an operator chore: `snapshot_download`
    given `cache_dir=<root>` and NO `local_dir` writes *through* the cache -- blob, ref and
    snapshot symlink together -- and lands the result at exactly this path.

    That is the distinction the Windows CUDA lane fell down on. `local_dir=<this path>` makes
    `huggingface_hub` resolve each blob into the cache (i.e. into this very directory) and then
    copy it to `local_dir`, which is the same file: `shutil.SameFileError`. Anything that did not
    collide would land outside the cache's own bookkeeping.

    The two NAME segments come back with the root because the caller has to check them: the hub
    derives both from the repo id and the revision, so a path whose `models--*` or terminal
    segment disagrees with the manifest is a mispointed variable, and healing it would fetch into
    a sibling directory rather than this one.

    Anchored at the TAIL, not at any `snapshots` component: the path a lane hands over IS the
    snapshot directory, so the layout has to be its last three segments. A `hub/` ancestor is NOT
    required, because `HF_HUB_CACHE` can put that layout under any directory name.
    """
    parts = snapshot.parts
    if len(parts) < 3 or parts[-2] != "snapshots" or not parts[-3].startswith("models--"):
        return None
    # `Path(*parts[:0])` is `.`, the right root for a cache named relative to the cwd.
    return Path(*parts[:-3]), parts[-3], parts[-1]


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


def _heal_in_cache(
    model: dict,
    snapshot: Path,
    location: tuple[Path, str, str],
    download: Download,
    *,
    require_materialization_provenance: bool = False,
) -> bool:
    """Materialize a cache-resident snapshot through the cache itself, then re-verify.

    The FIRST refusal is a precondition, checked before a byte moves. The hub derives both name
    segments deterministically -- `models--<org>--<name>` from the repo id, and the terminal
    directory from the resolved revision -- so a path whose segments disagree with the manifest
    row is a mispointed variable. Healing it would still succeed: `cache_dir` sends the fetch to
    the repo's OWN directory under this root, filling a sibling of the path the lane asked for,
    which then fails verification after a multi-GB download. Refusing up front costs nothing and
    names the discrepancy instead of reporting it as a missing-files shortfall.

    The refusals AFTER the download are backstops, not the ordinary path: they fire only when the
    cache-correct fetch has already run, so the problem can no longer be repaired by telling the
    operator to run that fetch by hand.
    """
    cache_root, repo_directory, revision_directory = location
    expected_repo_directory = "models--" + model["repository"].replace("/", "--")
    if repo_directory != expected_repo_directory or revision_directory != model["revision"]:
        raise RuntimeError(
            f"{model['key']} snapshot variable points into the Hugging Face cache at "
            f"{cache_root} at the wrong repository/revision: expected "
            f"{expected_repo_directory}/snapshots/{model['revision']}, path says "
            f"{repo_directory}/snapshots/{revision_directory} ({snapshot}). Refusing to fetch -- "
            f"materializing {model['repository']}@{model['revision']} through this cache would "
            "fill that repo's own sibling directory and leave this path exactly as wrong as it "
            "is. Correct the lane's snapshot variable."
        )
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
        verify_snapshot(
            model,
            snapshot,
            require_materialization_provenance=require_materialization_provenance,
        )
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


def _atomic_write(path: Path, content: str) -> None:
    """Publish one small provenance file atomically in its destination directory."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _managed_staging_paths(snapshot: Path) -> tuple[Path, Path, dict]:
    """Return one deterministic, provenance-marked staging location for a snapshot."""
    destination = str(snapshot.resolve())
    digest = hashlib.sha256(destination.encode()).hexdigest()[:20]
    staging = snapshot.parent / f".sceneworks-model-materialization-{digest}"
    claim = snapshot.parent / f".sceneworks-model-materialization-{digest}.json"
    document = {
        "schema_version": 1,
        "managed_by": "sceneworks.ensure_model_snapshot",
        "destination": destination,
    }
    return staging, claim, document


def _remove_managed_staging(staging: Path, claim: Path, expected_claim: dict) -> None:
    """Remove only an exact staging tree previously claimed for this destination."""
    if claim.is_symlink() or (claim.exists() and not claim.is_file()):
        raise RuntimeError(f"unsafe model materialization staging claim: {claim}")
    if not claim.exists():
        if staging.is_symlink() or staging.exists():
            raise RuntimeError(f"unclaimed model materialization staging path: {staging}")
        return
    try:
        actual_claim = json.loads(claim.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid model materialization staging claim: {claim}") from error
    if actual_claim != expected_claim:
        raise RuntimeError(f"model materialization staging claim does not match: {claim}")
    if staging.is_symlink() or (staging.exists() and not staging.is_dir()):
        raise RuntimeError(f"unsafe claimed model materialization staging path: {staging}")
    if staging.is_dir():
        shutil.rmtree(staging)
    claim.unlink()


def _prepare_managed_staging(snapshot: Path) -> tuple[Path, Path, dict]:
    """Reclaim an interrupted attempt and create a fresh empty staging directory."""
    staging, claim, document = _managed_staging_paths(snapshot)
    _remove_managed_staging(staging, claim, document)
    _atomic_write(claim, json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
    staging.mkdir(mode=0o700)
    return staging, claim, document


def _clear_snapshot_payload(snapshot: Path) -> None:
    """Remove every prior payload entry while retaining only our provenance state."""
    snapshot = snapshot.resolve()
    if snapshot == Path(snapshot.anchor) or snapshot == Path.home().resolve():
        raise RuntimeError(f"refusing to clear an unsafe snapshot directory: {snapshot}")
    provenance_names = {MARKER, MATERIALIZATION_RECEIPT, MATERIALIZATION_INCOMPLETE}
    entries = [entry for entry in snapshot.iterdir() if entry.name not in provenance_names]
    for entry in entries:
        if not (entry.is_symlink() or entry.is_file() or entry.is_dir()):
            raise RuntimeError(f"snapshot contains an unsupported payload entry: {entry}")
    for entry in entries:
        if entry.is_symlink() or entry.is_file():
            entry.unlink()
        else:
            shutil.rmtree(entry)


def _verified_materialization_source(source_root: Path, prefix: str | None) -> Path:
    """Return a nonempty, symlink-free materialization source tree."""
    if source_root.is_symlink():
        raise RuntimeError(f"materialization source root is a symlink: {source_root}")
    source = source_root.resolve()
    if prefix is not None:
        for part in prefix.split("/"):
            source /= part
            if source.is_symlink() or not source.is_dir():
                raise RuntimeError(
                    f"materialization source prefix was not downloaded safely: {prefix}"
                )
    files = []
    for item in sorted(source.rglob("*")):
        relative = item.relative_to(source)
        if relative.parts[0] == ".cache":
            continue
        if item.is_file() or item.is_symlink():
            files.append(item)
    if not files:
        label = prefix or "root"
        raise RuntimeError(f"materialization source tree is empty: {label}")
    for item in files:
        if item.is_symlink():
            raise RuntimeError(f"materialization source tree contains a symlink: {item}")
    return source


def _project_materialization_tree(source_root: Path, snapshot: Path, prefix: str | None) -> None:
    """Move a verified staging tree (or one of its subdirectories) into the snapshot."""
    snapshot = snapshot.resolve()
    source = _verified_materialization_source(source_root, prefix)
    files = [
        item
        for item in sorted(source.rglob("*"))
        if item.relative_to(source).parts[0] != ".cache"
        and (item.is_file() or item.is_symlink())
    ]

    # Preflight the complete move before changing the persistent cache. A late destination
    # collision must not leave half of a staged model installed.
    for item in files:
        relative = item.relative_to(source)
        destination_parent = snapshot
        for part in relative.parts[:-1]:
            destination_parent /= part
            if destination_parent.is_symlink():
                raise RuntimeError(
                    f"materialization projection crosses a destination symlink: {relative}"
                )
            if destination_parent.exists() and not destination_parent.is_dir():
                raise RuntimeError(
                    f"materialization projection collides with a destination file: {relative}"
                )
        destination = destination_parent / relative.name
        if destination.is_symlink() or destination.is_dir():
            raise RuntimeError(
                f"materialization projection collides with an unsafe destination: {relative}"
            )
        if destination.name in (MARKER, MATERIALIZATION_RECEIPT, MATERIALIZATION_INCOMPLETE):
            raise RuntimeError(
                f"materialization projection collides with a provenance file: {relative}"
            )

    for item in files:
        relative = item.relative_to(source)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(item, destination)


def ensure_snapshot(
    model: dict,
    snapshot: Path,
    download: Download,
    *,
    require_materialization_provenance: bool = False,
) -> bool:
    """Return true after downloading, or false when an existing snapshot is valid."""
    try:
        verify_snapshot(
            model,
            snapshot,
            require_materialization_provenance=require_materialization_provenance,
        )
        return False
    except RuntimeError as initial_error:
        if snapshot.is_symlink() or (snapshot.exists() and not snapshot.is_dir()):
            raise initial_error
        if snapshot.is_dir():
            marker = snapshot / MARKER
            actual_revision = snapshot.name
            if marker.is_symlink() or (marker.exists() and not marker.is_file()):
                raise initial_error
            if marker.is_file():
                actual_revision = marker.read_text(encoding="utf-8").strip()
            if actual_revision != model["revision"]:
                raise initial_error

    # A COMPLETE snapshot already returned above, from `verify_snapshot`. Reaching here means the
    # pinned revision is absent or short of `expected_files`, and WHERE it lives decides how it is
    # fetched. A cache-resident target is materialized through the cache; only a plain directory
    # takes `local_dir`, which is what exploded on the Windows CUDA box when it was aimed at
    # `<cache>/hub/models--*/snapshots/<revision>`.
    # Alternate-source (receipt) models never take the heal: their bytes come from a different
    # repository/revision than the canonical layout the cache directory names, so a `cache_dir`
    # fetch could not land them at this path. The staging projection below handles a cache-shaped
    # destination for them (see
    # `test_materialization_replaces_stale_destination_symlinks_and_extra_payload`).
    location = hf_cache_location(snapshot)
    if location is not None and expected_materialization_receipt(model) is None:
        return _heal_in_cache(
            model,
            snapshot,
            location,
            download,
            require_materialization_provenance=require_materialization_provenance,
        )

    receipt = expected_materialization_receipt(model)
    source_repository = (
        receipt["materialization_repository"] if receipt is not None else model["repository"]
    )
    source_revision = (
        receipt["materialization_revision"] if receipt is not None else model["revision"]
    )
    source_prefix = receipt["materialization_path_prefix"] if receipt is not None else None
    snapshot.mkdir(parents=True, exist_ok=True)
    receipt_path = snapshot / MATERIALIZATION_RECEIPT
    incomplete_path = snapshot / MATERIALIZATION_INCOMPLETE
    if receipt_path.is_symlink() or (receipt_path.exists() and not receipt_path.is_file()):
        raise RuntimeError(f"{model['key']} has an unsafe materialization receipt")
    # The incomplete marker is published before any transfer and the receipt only after the
    # downloader returns and the payload verifies. Even an interruption after every currently
    # required file arrives therefore cannot masquerade as a completed materialization.
    _atomic_write(snapshot / MARKER, model["revision"] + "\n")
    attempt = receipt or {
        "schema_version": 2,
        "canonical_repository": model["repository"],
        "canonical_revision": model["revision"],
    }
    _atomic_write(
        incomplete_path,
        json.dumps(attempt, sort_keys=True, separators=(",", ":")) + "\n",
    )
    receipt_path.unlink(missing_ok=True)
    print(
        f"materializing {source_repository}@{source_revision} for "
        f"{model['repository']}@{model['revision']} in {snapshot.resolve()}",
        flush=True,
    )
    download_kwargs = {
        "repo_id": source_repository,
        "revision": source_revision,
        # Public release fixtures stay explicitly anonymous. Gated checkpoints
        # opt in to the runner's configured Hugging Face credential without
        # placing the token in the workflow or command line.
        "token": bool(
            model.get("materialization_requires_auth", False)
            if receipt is not None
            else model.get("requires_auth", False)
        ),
    }
    # Optional per-model download allow-list. When set, materialize ONLY these repo-relative paths
    # (snapshot_download `allow_patterns`) instead of the whole repo — for repos whose pinned
    # checkpoints are a small fraction of a large repo (e.g. `hkchengrex/MMAudio` ships ~46 GB of
    # training checkpoints + weight variants the inference stack never loads). Absent ⇒ whole-repo
    # download, the default for every other model. `verify_snapshot` still enforces `expected_files`,
    # so an under-fetch (a needed file left off the list) fails loudly right after download.
    allow_patterns = model.get("download_files")
    if source_prefix is not None:
        allow_patterns = (
            [f"{source_prefix}/{pattern}" for pattern in allow_patterns]
            if allow_patterns
            else [f"{source_prefix}/**"]
        )
    if allow_patterns:
        download_kwargs["allow_patterns"] = list(allow_patterns)
    try:
        # A fresh, empty staging directory is mandatory. huggingface_hub intentionally returns a
        # non-empty local_dir when its repo-info request fails, without proving revision or
        # completeness. Staging prevents an old persistent cache from being blessed as the mirror
        # merely because the Hub was unreachable.
        staging, staging_claim, staging_document = _prepare_managed_staging(snapshot)
        try:
            download_kwargs["local_dir"] = str(staging)
            downloaded = download(**download_kwargs)
            if downloaded is None or Path(downloaded).resolve() != staging.resolve():
                raise RuntimeError(
                    f"materializer returned an unexpected staging path: {downloaded!r}"
                )
            staged_payload = _verified_materialization_source(staging, source_prefix)
            verify_model_payload_files(model, staged_payload)
            materialization_payload_files(model, staged_payload)
            _clear_snapshot_payload(snapshot)
            _project_materialization_tree(staging, snapshot, source_prefix)
        finally:
            _remove_managed_staging(staging, staging_claim, staging_document)
        verify_snapshot_payload(model, snapshot)
        if receipt is not None:
            receipt = completed_materialization_receipt(model, snapshot)
            _atomic_write(
                receipt_path,
                json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
            )
        incomplete_path.unlink()
        verify_snapshot(
            model,
            snapshot,
            require_materialization_provenance=require_materialization_provenance,
        )
    except Exception as error:
        if receipt_path.is_file() and not receipt_path.is_symlink():
            receipt_path.unlink()
        if not incomplete_path.exists() and not incomplete_path.is_symlink():
            _atomic_write(
                incomplete_path,
                json.dumps(attempt, sort_keys=True, separators=(",", ":")) + "\n",
            )
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
    parser.add_argument(
        "--require-materialization-provenance",
        action="store_true",
        help=(
            "require either the exact alternate-source receipt or a pristine canonical "
            "snapshots/<revision> cache"
        ),
    )
    args = parser.parse_args()
    model = load_model(args.manifest, args.model)
    downloaded = ensure_snapshot(
        model,
        args.snapshot,
        download_snapshot,
        require_materialization_provenance=args.require_materialization_provenance,
    )
    source = "downloaded" if downloaded else "cached"
    print(f"model snapshot: OK ({model['key']}@{model['revision']}, {source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
