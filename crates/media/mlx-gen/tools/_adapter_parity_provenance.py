"""Shared provenance guards for the licensed adapter-parity dump scripts.

The real-weight artifacts are intentionally gitignored, so their producing
source must be pinned in the tracked scripts/manifest instead of inferred from
whichever checkout happens to be on disk.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

MFLUX_REPOSITORY = "https://github.com/michaeltrefry/mflux.git"
MFLUX_REVISION = "81106c833435d1a1f62ca04a8c01c1d380d39272"
MFLUX_REPOSITORY_IDENTITY = "github.com/michaeltrefry/mflux"
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _repository_identity(remote: str) -> str:
    value = remote.strip().removesuffix(".git")
    if value.startswith("git@"):
        value = value[4:].replace(":", "/", 1)
    elif "://" in value:
        value = value.split("://", 1)[1]
        value = value.split("@", 1)[-1]
    return value.lower()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frozen_repository(
    root: str | Path,
    *,
    expected_revision: str = MFLUX_REVISION,
    expected_identity: str = MFLUX_REPOSITORY_IDENTITY,
) -> str:
    root = Path(root)
    revision = _git(root, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise RuntimeError(
            f"adapter parity requires frozen mflux {expected_revision}, got {revision} at {root}"
        )
    remote = _git(root, "remote", "get-url", "origin")
    if _repository_identity(remote) != expected_identity:
        raise RuntimeError(
            f"adapter parity requires {MFLUX_REPOSITORY}, got origin {remote!r} at {root}"
        )
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=normal")
    dirty = []
    for line in status.splitlines():
        code, path = line[:2], line[3:]
        if code == "??" and path.split("/", 1)[0].startswith(".venv"):
            continue
        dirty.append(line)
    if dirty:
        raise RuntimeError(
            "adapter parity requires clean tracked/index state and no untracked source; "
            f"dirty entries at {root}:\n" + "\n".join(dirty)
        )
    return revision


def assert_frozen_mflux() -> str:
    import mflux

    root = Path(mflux.__file__).resolve().parents[2]
    return assert_frozen_repository(root)


def hf_snapshot_inventory(path: str | Path) -> str:
    root = Path(path).expanduser().absolute()
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*")):
        if not (item.is_file() or item.is_symlink()):
            continue
        relative = item.relative_to(root).as_posix()
        if item.is_symlink():
            record = f"{relative}\0symlink\0{sha256(item)}\0{item.stat().st_size}\n"
        else:
            record = f"{relative}\0file\0{sha256(item)}\0{item.stat().st_size}\n"
        digest.update(record.encode())
    return digest.hexdigest()


def assert_hf_snapshot(
    path: str | Path,
    *,
    repository: str,
    revision: str,
    subdirectory: str = ".",
    reference: str = "main",
) -> tuple[str, str]:
    if COMMIT_SHA.fullmatch(revision) is None:
        raise RuntimeError(f"model revision must be a full 40-character commit SHA: {revision!r}")
    if not reference or reference.startswith("/") or ".." in Path(reference).parts:
        raise RuntimeError(f"invalid Hugging Face reference: {reference!r}")
    loaded = Path(path).expanduser().absolute()
    if not loaded.is_dir():
        raise RuntimeError(
            "licensed parity generation requires an explicit local Hugging Face snapshot path, "
            f"got {path!r}"
        )
    parts = loaded.parts
    try:
        snapshots_index = len(parts) - 1 - list(reversed(parts)).index("snapshots")
    except ValueError as error:
        raise RuntimeError(f"model path is not inside an HF snapshots directory: {loaded}") from error
    expected_cache_repo = f"models--{repository.replace('/', '--')}"
    actual_cache_repo = parts[snapshots_index - 1] if snapshots_index else ""
    actual_revision = parts[snapshots_index + 1] if snapshots_index + 1 < len(parts) else ""
    relative = Path(*parts[snapshots_index + 2 :]).as_posix() or "."
    if actual_cache_repo != expected_cache_repo or actual_revision != revision:
        raise RuntimeError(
            "loaded model path does not match the claimed repository/revision: "
            f"{loaded} != {repository}@{revision}"
        )
    if relative != subdirectory:
        raise RuntimeError(
            f"loaded model subdirectory {relative!r} does not match claimed {subdirectory!r}"
        )
    repository_cache = Path(*parts[:snapshots_index])
    reference_path = repository_cache / "refs" / reference
    if not reference_path.is_file():
        raise RuntimeError(
            f"model cache has no local {reference!r} reference proving {repository}@{revision}: "
            f"{reference_path}"
        )
    reference_revision = reference_path.read_text(encoding="utf-8").strip()
    if reference_revision != revision:
        raise RuntimeError(
            f"model cache reference {repository}@{reference} resolves to "
            f"{reference_revision}, not claimed {revision}"
        )
    return str(loaded), hf_snapshot_inventory(loaded)


def assert_hf_file(
    path: str | Path,
    *,
    repository: str,
    revision: str,
    file: str,
    reference: str = "main",
) -> str:
    loaded = Path(path).expanduser().absolute()
    parts = loaded.parts
    try:
        snapshots_index = len(parts) - 1 - list(reversed(parts)).index("snapshots")
    except ValueError as error:
        raise RuntimeError(f"artifact path is not inside an HF snapshots directory: {loaded}") from error
    snapshot = Path(*parts[: snapshots_index + 2])
    relative = Path(*parts[snapshots_index + 2 :]).as_posix()
    assert_hf_snapshot(
        snapshot,
        repository=repository,
        revision=revision,
        reference=reference,
    )
    if relative != file:
        raise RuntimeError(
            f"loaded artifact file {relative!r} does not match claimed {file!r}"
        )
    if not loaded.is_file():
        raise RuntimeError(f"missing Hugging Face artifact: {loaded}")
    return str(loaded)


def runtime_versions() -> str:
    names = (
        "mlx",
        "mlx-metal",
        "numpy",
        "safetensors",
        "diffusers",
        "transformers",
        "torch",
        "peft",
    )
    versions = {
        "python": platform.python_version(),
        "platform": sys.platform,
    }
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return json.dumps(versions, sort_keys=True, separators=(",", ":"))


def golden_metadata(
    *,
    script: str | Path,
    model_path: str | Path,
    model_repository: str,
    model_revision: str,
    model_subdirectory: str = ".",
    model_reference: str = "main",
) -> dict[str, str]:
    loaded_path, inventory = assert_hf_snapshot(
        model_path,
        repository=model_repository,
        revision=model_revision,
        subdirectory=model_subdirectory,
        reference=model_reference,
    )
    return {
        "reference_mflux_repository": MFLUX_REPOSITORY,
        "reference_mflux_revision": assert_frozen_mflux(),
        "reference_model_repository": model_repository,
        "reference_model_revision": model_revision,
        "reference_model_ref": model_reference,
        "reference_model_subdirectory": model_subdirectory,
        "reference_model_path": loaded_path,
        "reference_model_inventory_sha256": inventory,
        "reference_runtime": runtime_versions(),
        "reference_script_sha256": sha256(script),
        "reference_provenance_sha256": sha256(__file__),
    }
