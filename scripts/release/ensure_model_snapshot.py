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
        verify_model_payload_files,
        verify_snapshot,
        verify_snapshot_payload,
    )


Download = Callable[..., Any]


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
    parser.add_argument("--snapshot", required=True, type=Path)
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
