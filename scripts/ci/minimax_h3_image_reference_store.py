#!/usr/bin/env python3
"""Runner-local, branch-independent store for the MiniMax-H3-Image diffusers reference.

GitHub scopes `actions/cache` entries by branch: a run on a new story branch restores only its own
branch's entries and the default branch's, so a reference saved on a sibling branch was never
found. This store lives on the runner's task-owned weights root instead and is keyed by the same
content hash, so every branch that computes the same key reuses the same entry.

Layout under the store root: one complete entry per key, `<key>/` holding the reference files and
`manifest.json` (each file's SHA-256, written at publish time). Publishing writes into
`<key>.partial-*` inside the root and renames it to `<key>/` only once the manifest exists, so a
concurrent or failed run never exposes a partial entry. Pruning keeps the newest `KEEP` entries and
deletes only directories whose names match `h3img-ref-v*`; it never touches anything else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

MANIFEST = "manifest.json"
KEEP = 3
PARTIAL_MAX_AGE_SECONDS = 24 * 3600
ENTRY = re.compile(r"^h3img-ref-v[0-9]+-[0-9a-f]{64}$")
PARTIAL = re.compile(r"^h3img-ref-v[0-9]+-[0-9a-f]{64}\.partial-[0-9]+-[0-9a-f]{8}$")
CHUNK = 8 << 20


def _key(key: str) -> str:
    if not ENTRY.match(key):
        raise ValueError(f"not a reference store key: {key!r}")
    return key


def _copy_hashing(source: Path, target: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as reader, target.open("wb") as writer:
        while chunk := reader.read(CHUNK):
            digest.update(chunk)
            writer.write(chunk)
    return digest.hexdigest()


def restore(root: Path, key: str, dest: Path) -> bool:
    """Copy a verified entry into a fresh `dest`. A missing or failing entry is a miss: `dest` is
    left absent, and a failing entry is deleted (only that key's directory)."""
    entry = root / _key(key)
    shutil.rmtree(dest, ignore_errors=True)
    if not entry.is_dir():
        print(f"reference store miss: no entry at {entry}")
        return False
    reason = None
    try:
        expected = json.loads((entry / MANIFEST).read_text(encoding="utf-8"))["files"]
        present = sorted(path.name for path in entry.iterdir() if path.name != MANIFEST)
        if not expected or present != sorted(expected) or not all(
            (entry / name).is_file() and not (entry / name).is_symlink() for name in present
        ):
            reason = f"files {present} do not match the manifest {sorted(expected)}"
        else:
            dest.mkdir(parents=True)
            for name in present:
                actual = _copy_hashing(entry / name, dest / name)
                if actual != expected[name]:
                    reason = f"{name} sha256 {actual} != manifest {expected[name]}"
                    break
    except (OSError, ValueError, KeyError, TypeError) as error:
        reason = f"unreadable entry: {error}"
    if reason:
        print(f"::warning::reference store entry {entry} failed verification ({reason}); deleting it")
        shutil.rmtree(entry, ignore_errors=True)
        shutil.rmtree(dest, ignore_errors=True)
        return False
    os.utime(entry)
    print(f"reference store hit: {entry} verified against {MANIFEST}")
    return True


def publish(root: Path, key: str, source: Path) -> Path:
    """Copy `source` into a partial directory, write its manifest, and rename it into place."""
    entry = root / _key(key)
    files = sorted(path for path in source.iterdir())
    if not files or not all(path.is_file() and not path.is_symlink() for path in files):
        raise ValueError(f"{source} must hold only regular files, and at least one")
    if entry.is_dir():
        print(f"reference store entry already present: {entry}")
        return entry
    root.mkdir(parents=True, exist_ok=True)
    partial = root / f"{key}.partial-{os.getpid()}-{os.urandom(4).hex()}"
    partial.mkdir()
    try:
        hashes = {path.name: _copy_hashing(path, partial / path.name) for path in files}
        (partial / MANIFEST).write_text(json.dumps({"key": key, "files": hashes}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            os.rename(partial, entry)
        except OSError:
            if not entry.is_dir():
                raise
            print(f"reference store entry published concurrently: {entry}")
    finally:
        shutil.rmtree(partial, ignore_errors=True)
    print(f"reference store published: {entry}")
    return entry


def prune(root: Path, keep: int = KEEP, now: float | None = None) -> list[Path]:
    """Delete all but the newest `keep` entries, and partials older than a day. Only names that
    match `h3img-ref-v*` are ever candidates."""
    if not root.is_dir():
        return []
    now = time.time() if now is None else now
    entries, removed = [], []
    for path in root.iterdir():
        if path.is_symlink() or not path.is_dir():
            continue
        if ENTRY.match(path.name):
            entries.append(path)
        elif PARTIAL.match(path.name) and now - path.stat().st_mtime > PARTIAL_MAX_AGE_SECONDS:
            removed.append(path)
    entries.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    removed.extend(entries[keep:])
    for path in removed:
        assert path.name.startswith("h3img-ref-v") and path.parent == root
        print(f"reference store prune: {path}")
        shutil.rmtree(path)
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("restore", "publish"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--dir", type=Path, required=True, help="workspace reference directory")
    args = parser.parse_args(argv)
    if args.command == "restore":
        hit = restore(args.root, args.key, args.dir)
        output = os.environ.get("GITHUB_OUTPUT")
        if output:
            with open(output, "a", encoding="utf-8") as handle:
                handle.write(f"hit={'true' if hit else 'false'}\n")
    else:
        publish(args.root, args.key, args.dir)
    prune(args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
