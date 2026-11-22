#!/usr/bin/env python3
"""Stage and bind the product-pinned Hub snapshots a registered model proof declares.

A case's `hub_snapshots` slots (scripts/ci/model_inference_proofs.py `validate_hub_snapshot_slots`)
resolve, in the validated plan, to `environment name -> {repository, revision, subfolder}`. For each
pin this setup:

1. reads the Hub's expanded inventory of that public revision and selects the files under the
   subfolder (the whole revision when there is none);
2. stats each file at `<hub>/models--org--name/snapshots/<revision>/<file>` — it never lists a
   directory (a Hub listing blocked forever on nax-macos-2) — and fetches only the absent ones,
   after refusing when their bytes plus a reserve exceed the free space; an existing file is
   never replaced. Several runner listeners share one host's hub, so each absent file is fetched
   under a host-wide per-file lock (`<hub>/.locks/<repo>/model-proof-staging/...`, released by
   the OS when a job dies) and re-checked once the lock is held: a job that waited finds the
   file the other job staged instead of downloading it again;
3. authenticates every file of the subfolder against the Hub's LFS SHA-256 / Git blob digests
   (scripts/ci/pinned_hub_source.py), whether staged now or already present;
4. appends `NAME=<snapshot>/<subfolder>` for each slot and the receipt path to `GITHUB_ENV`.

`<hub>` is `MODEL_PROOF_HUB_ROOT` (the runner's model root, `~` expanded, real path). No model is
loaded; the one manual proof invocation that follows re-checks the receipt against its bindings.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
from functools import partial
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import sys
import tempfile
import time
import urllib.parse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci.file_lock import exclusive_file_lock, try_lock_handle  # noqa: E402
from scripts.ci.flux1_conditional_asset_inventory import confined, publish_download  # noqa: E402
from scripts.ci.model_inference_proofs import unlinked_path  # noqa: E402
from scripts.ci.pinned_hub_source import pinned_file_identity, pinned_manifest, verify_pinned_hub_files  # noqa: E402

GIB = 1024**3
# Free space a staging run must leave on the model root's volume.
DISK_RESERVE_BYTES = 20 * GIB
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{name}"
LOCK_POLL_SECONDS = 5
# A holder may be downloading tens of GiB; say so periodically rather than going silent.
LOCK_LOG_SECONDS = 180


def snapshot_dir(hub: Path, repository: str, revision: str) -> Path:
    return hub / ("models--" + repository.replace("/", "--")) / "snapshots" / revision


def _try_lock(handle) -> bool:
    return try_lock_handle(handle, is_windows=(os.name == "nt"))


def staging_lock_path(hub: Path, repository: str, revision: str, file: str) -> Path:
    return (hub / ".locks" / ("models--" + repository.replace("/", "--")) / "model-proof-staging"
            / revision / (file + ".lock"))


def _guard_lock(path: Path, hub: Path) -> None:
    unlinked_path(path, hub)
    if path.exists() and (not path.is_file() or path.stat().st_nlink != 1):
        raise ValueError(f"staging lock must be a regular unshared file: {path}")


def _guard_directory(path: Path, hub: Path) -> None:
    unlinked_path(path, hub)
    if path.exists() and not path.is_dir():
        raise ValueError(f"snapshot asset parent is not a directory: {path}")


def _asset_present(path: Path, hub: Path) -> bool:
    # Hugging Face uses symlink leaves to in-hub blobs. Writes and directory
    # creation must always use real, unlinked ancestors inside the hub.
    _guard_directory(path.parent, hub)
    if not os.path.lexists(path):
        return False
    try:
        resolved = confined(path.resolve(strict=True), hub, path)
    except FileNotFoundError as error:
        raise ValueError(f"snapshot asset has a dangling link: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"snapshot asset is not a regular file: {path}")
    return True


def _publish_verified_symlink(temporary: Path, destination: Path, *, hub: Path, blobs: Path) -> None:
    """Place a verified payload in private Hub storage, then create a no-clobber snapshot leaf."""
    _guard_directory(blobs, hub)
    blobs.mkdir(parents=True, exist_ok=True)
    _guard_directory(blobs, hub)
    identity = temporary.lstat()
    owned = Path(tempfile.mkdtemp(prefix=".model-proof-", dir=blobs))
    blob = owned / destination.name
    target = os.path.relpath(blob, destination.parent)
    try:
        # This job exclusively created the private directory. The payload was already fsynced
        # and authenticated before publication, and the move leaves its bytes untouched.
        _guard_directory(owned, hub)
        os.rename(temporary, blob)
        _guard_directory(owned, hub)
        _asset_present(destination, hub)
        # Creating a link fails if any writer has published this leaf, even after the guard.
        os.symlink(target, destination)
    except BaseException:
        try:
            _discard_unpublished_blob(blob, identity, destination, hub)
        except (OSError, ValueError):
            # Preserve ambiguous state and the original failure.
            pass
        raise


def _discard_unpublished_blob(blob, identity, destination, hub):
    _guard_directory(blob.parent, hub)
    _guard_directory(destination.parent, hub)
    try:
        target = os.readlink(destination)
    except OSError as error:
        if error.errno not in (errno.ENOENT, errno.EINVAL):
            raise
    else:
        if os.path.abspath(destination.parent / target) == str(blob):
            return  # Publication may have completed before Python received the interruption.
    try:
        current = blob.lstat()
    except FileNotFoundError:
        pass
    else:
        if os.path.samestat(identity, current):
            blob.unlink()
    blob.parent.rmdir()


@contextmanager
def staging_lock(hub: Path, repository: str, revision: str, file: str):
    """Hold the host-wide lock for one pinned file. The lock file is never deleted, because a
    waiter could otherwise lock a different inode; the OS releases the lock if a job dies."""
    hub = hub.resolve(strict=True)
    path = staging_lock_path(hub, repository, revision, file)

    def open_lock(name, flags):
        return os.open(name, flags | getattr(os, "O_NOFOLLOW", 0), 0o600)

    @contextmanager
    def open_guarded(path):
        _guard_lock(path, hub)
        path.parent.mkdir(parents=True, exist_ok=True)
        _guard_lock(path, hub)
        with open(path, "a+b", opener=open_lock) as handle:
            _guard_lock(path, hub)
            opened = os.fstat(handle.fileno())
            if opened.st_nlink != 1 or not os.path.samestat(opened, path.lstat()):
                raise ValueError(f"staging lock changed during open: {path}")
            yield handle

    started = logged = 0.0

    def on_wait(wait_count):
        nonlocal started, logged
        now = time.monotonic()
        if wait_count == 0:
            started = logged = now
            print(f"waiting for another job staging {repository}@{revision}/{file}", flush=True)
        elif now - logged >= LOCK_LOG_SECONDS:
            logged = now
            print(f"still waiting ({logged - started:.0f} s) for another job staging "
                  f"{repository}@{revision}/{file}", flush=True)

    with exclusive_file_lock(path, poll_seconds=LOCK_POLL_SECONDS, on_wait=on_wait,
                             try_lock=_try_lock, monotonic=time.monotonic, sleep=time.sleep,
                             open_persistent=open_guarded):
        yield


def pinned_files(inventory: dict[str, dict], subfolder: str) -> dict[str, int]:
    """Every file of the pinned revision under `subfolder` -> its Hub size."""
    prefix = subfolder + "/" if subfolder else ""
    files = {name: sibling.get("size") for name, sibling in inventory.items() if name.startswith(prefix)}
    if not files:
        raise ValueError(f"pinned revision has no files under {subfolder or '<root>'}")
    if any(type(size) is not int or size < 0 for size in files.values()):
        raise ValueError("Hub inventory lacks a file size")
    return files


def resolve_hub_root(env: dict[str, str], *, platform: str | None = None) -> Path:
    configured = env.get("MODEL_PROOF_HUB_ROOT", "").strip()
    if not configured:
        raise ValueError("MODEL_PROOF_HUB_ROOT (the runner model root variable) is unset")
    expanded = os.path.expanduser(configured)
    root = Path(expanded)
    if not (root.is_absolute() or PureWindowsPath(expanded).is_absolute()):
        raise ValueError("MODEL_PROOF_HUB_ROOT must be an absolute or ~ path")
    # On macOS, repo variables point at an external volume (for example /Volumes/Models/...).
    # When that external volume is not mounted on the runner, /Volumes is root-owned and mkdir
    # fails with EACCES; fall back to the runner account's local macOS cache directory.
    active_platform = sys.platform if platform is None else platform
    if active_platform == "darwin" and len(root.parts) >= 3 and root.parts[:2] == ("/", "Volumes"):
        if not Path("/", "Volumes", root.parts[2]).exists():
            home = Path(os.path.expanduser(env.get("HOME") or "~"))
            return home / "Library/Caches/SceneWorks/model-proof/huggingface/hub"
    return root


def hub_root(env: dict[str, str]) -> Path:
    root = resolve_hub_root(env)
    root.mkdir(parents=True, exist_ok=True)
    return Path(os.path.realpath(root))


def stage(snapshots: dict[str, dict], hub: Path, *, manifest=pinned_manifest, fetch=None,
          verify=verify_pinned_hub_files, free_bytes=None) -> dict:
    """Fetch every absent pinned file, then authenticate every pinned file; return the receipt."""
    if fetch is None:
        from scripts.ci.flux1_conditional_asset_inventory import fetch_pinned as fetch
    hub = hub.resolve(strict=True)
    free_bytes = free_bytes or (lambda path: shutil.disk_usage(path).free)
    plans = []
    for name, pin in sorted(snapshots.items()):
        repository, revision, subfolder = pin["repository"], pin["revision"], pin["subfolder"]
        _, _, inventory = manifest(repository, revision)
        files = pinned_files(inventory, subfolder)
        base = snapshot_dir(hub, repository, revision)
        absent = {}
        for file in files:
            if ("\\" in file or ":" in file or any(part in ("", ".", "..") for part in file.split("/"))):
                raise ValueError(f"unsafe pinned snapshot asset path: {file}")
            identity = pinned_file_identity(file, inventory[file])
            if not _asset_present(base / file, hub):
                _guard_lock(staging_lock_path(hub, repository, revision, file), hub)
                absent[file] = identity
        plans.append((name, pin, base, files, absent))
    needed = sum(identity["bytes"] for *_, absent in plans for identity in absent.values())
    free = free_bytes(hub)
    if needed + DISK_RESERVE_BYTES > free:
        raise ValueError(f"staging needs {needed / GIB:.1f} GiB plus a {DISK_RESERVE_BYTES / GIB:.0f} GiB reserve "
                         f"but {hub} has {free / GIB:.1f} GiB free")
    receipt: dict[str, dict] = {}
    for name, pin, base, files, absent in plans:
        fetched, placed_by_another_job = [], []
        for file in sorted(absent):
            destination = base / file
            with staging_lock(hub, pin["repository"], pin["revision"], file):
                # Staged for an earlier slot of the same revision, or by another job on this host.
                if _asset_present(destination, hub):
                    continue
                url = HF_RESOLVE.format(repo=pin["repository"], revision=pin["revision"], name=urllib.parse.quote(file))
                identity = absent[file]
                size = identity["bytes"]
                print(f"staging {pin['repository']}@{pin['revision']}/{file} ({size / GIB:.2f} GiB)", flush=True)
                digest_arg = "expected_sha256" if identity["kind"] == "lfs_sha256" else "expected_git_blob"
                publish = partial(publish_download, guard=partial(_asset_present, hub=hub),
                                  without_hardlinks=partial(_publish_verified_symlink, hub=hub,
                                                            blobs=base.parent.parent / "blobs"))
                try:
                    _, _, _, placed = fetch(url, destination, expected_bytes=size, publish=publish,
                                            **{digest_arg: identity["digest"]})
                except FileExistsError:
                    # A peer using the Hub outside this stager can win before the
                    # transfer starts. The whole-snapshot check below authenticates it.
                    if not _asset_present(destination, hub):
                        raise
                    placed = True
                (placed_by_another_job if placed else fetched).append(file)
        for file in files:
            _asset_present(base / file, hub)
        verified = verify(pin["repository"], pin["revision"], base, tuple(sorted(files)))
        path = base / pin["subfolder"] if pin["subfolder"] else base
        receipt[name] = {**pin, "path": str(path), "status": verified["status"], "fetched": fetched,
                         "kept_existing": sorted(set(files) - set(fetched)),
                         "placed_by_another_job": placed_by_another_job,
                         "manifest_sha256": verified["manifest_sha256"], "files": verified["files"]}
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    snapshots = plan.get("hub_snapshots")
    if plan.get("asset_setup") != "hub-snapshots" or not isinstance(snapshots, dict) or not snapshots:
        raise ValueError("Hub snapshot staging requires a validated plan with hub_snapshots")
    env = dict(os.environ)
    github_env, temporary = env.get("GITHUB_ENV", ""), env.get("RUNNER_TEMP", "")
    if not github_env or not temporary:
        raise ValueError("GitHub Actions environment file and RUNNER_TEMP are required")
    hub = hub_root(env)
    receipt = {"hub_root": str(hub), "snapshots": stage(snapshots, hub)}
    receipt_path = Path(temporary) / "model-proof-hub-snapshots.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    with open(github_env, "a", encoding="utf-8") as out:
        for name, entry in receipt["snapshots"].items():
            if "\n" in entry["path"] or not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                raise ValueError(f"invalid binding {name}")
            out.write(f"{name}={entry['path']}\n")
        out.write(f"MODEL_PROOF_HUB_SNAPSHOT_RECEIPT={receipt_path}\n")
    print(json.dumps({name: {key: entry[key] for key in ("repository", "revision", "subfolder", "path", "fetched")}
                      for name, entry in receipt["snapshots"].items()}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print("Hub snapshot staging rejected: " + str(error), file=sys.stderr)
        raise SystemExit(1)
