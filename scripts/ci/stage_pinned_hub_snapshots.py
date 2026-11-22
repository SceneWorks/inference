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
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.parse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci.pinned_hub_source import pinned_manifest, verify_pinned_hub_files  # noqa: E402

GIB = 1024**3
# Free space a staging run must leave on the model root's volume.
DISK_RESERVE_BYTES = 20 * GIB
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{name}"
LOCK_POLL_SECONDS = 5
# A holder may be downloading tens of GiB; say so periodically rather than going silent.
LOCK_LOG_SECONDS = 180
# Only these mean "another holder has it"; any other lock error (ENOLCK/ENOTSUP on a network
# hub, EBADF, EINVAL) is raised instead of spinning forever.
POSIX_LOCK_BUSY = frozenset((errno.EWOULDBLOCK, errno.EAGAIN))
MSVCRT_LOCK_BUSY = frozenset((errno.EACCES, errno.EDEADLK))


def snapshot_dir(hub: Path, repository: str, revision: str) -> Path:
    return hub / ("models--" + repository.replace("/", "--")) / "snapshots" / revision


def _try_lock(handle) -> bool:
    windows = os.name == "nt"
    try:
        if windows:
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as error:
        if error.errno in (MSVCRT_LOCK_BUSY if windows else POSIX_LOCK_BUSY):
            return False
        raise


@contextmanager
def staging_lock(hub: Path, repository: str, revision: str, file: str):
    """Hold the host-wide lock for one pinned file. The lock file is never deleted, because a
    waiter could otherwise lock a different inode; the OS releases the lock if a job dies."""
    path = (hub / ".locks" / ("models--" + repository.replace("/", "--")) / "model-proof-staging"
            / revision / (file + ".lock"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        if not _try_lock(handle):
            started = logged = time.monotonic()
            print(f"waiting for another job staging {repository}@{revision}/{file}", flush=True)
            while not _try_lock(handle):
                time.sleep(LOCK_POLL_SECONDS)
                if time.monotonic() - logged >= LOCK_LOG_SECONDS:
                    logged = time.monotonic()
                    print(f"still waiting ({logged - started:.0f} s) for another job staging "
                          f"{repository}@{revision}/{file}", flush=True)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
    root = Path(os.path.expanduser(configured))
    if not root.is_absolute():
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
    free_bytes = free_bytes or (lambda path: shutil.disk_usage(path).free)
    plans = []
    for name, pin in sorted(snapshots.items()):
        repository, revision, subfolder = pin["repository"], pin["revision"], pin["subfolder"]
        _, _, inventory = manifest(repository, revision)
        files = pinned_files(inventory, subfolder)
        base = snapshot_dir(hub, repository, revision)
        absent = {file: size for file, size in files.items() if not (base / file).exists()}
        plans.append((name, pin, base, files, absent))
    needed = sum(size for *_, absent in plans for size in absent.values())
    free = free_bytes(hub)
    if needed + DISK_RESERVE_BYTES > free:
        raise ValueError(f"staging needs {needed / GIB:.1f} GiB plus a {DISK_RESERVE_BYTES / GIB:.0f} GiB reserve "
                         f"but {hub} has {free / GIB:.1f} GiB free")
    receipt: dict[str, dict] = {}
    for name, pin, base, files, absent in plans:
        fetched, placed_by_another_job = [], []
        for file in sorted(absent):
            with staging_lock(hub, pin["repository"], pin["revision"], file):
                # Staged for an earlier slot of the same revision, or by another job on this host.
                if (base / file).exists():
                    continue
                url = HF_RESOLVE.format(repo=pin["repository"], revision=pin["revision"], name=urllib.parse.quote(file))
                print(f"staging {pin['repository']}@{pin['revision']}/{file} ({absent[file] / GIB:.2f} GiB)", flush=True)
                _, _, _, placed = fetch(url, base / file)
                (placed_by_another_job if placed else fetched).append(file)
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
