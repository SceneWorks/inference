#!/usr/bin/env python3.12
"""Materialize the four pinned W1 snapshots in the hub cache and verify them against models.tsv.

For each (repo, revision) in models.tsv: if `<hub>/models--<org>--<name>/snapshots/<revision>`
already holds every pinned file at its pinned byte size, nothing is fetched. Otherwise the
revision is downloaded THROUGH the cache (`snapshot_download(cache_dir=<hub>)`, never
`local_dir`) and verified again. Byte sizes are the cheap identity check here; the campaign
parent re-inventories every snapshot (sha256 + bytes) before it starts a row, so a wrong byte
that happens to keep its size is still refused there.

`--check-only` never touches the network (used by every campaign job before it launches).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def load_pins(path: Path) -> dict[tuple[str, str], list[tuple[str, int]]]:
    pins: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        repo, revision, name, size, _sha256 = line.split("\t")
        pins.setdefault((repo, revision), []).append((name, int(size)))
    return pins


def snapshot_dir(hub: Path, repo: str, revision: str) -> Path:
    return hub / ("models--" + repo.replace("/", "--")) / "snapshots" / revision


def problems(snapshot: Path, files: list[tuple[str, int]]) -> list[str]:
    found = []
    for name, size in files:
        path = snapshot / name
        if not path.is_file():
            found.append(f"{name}: missing")
        elif path.stat().st_size != size:
            found.append(f"{name}: {path.stat().st_size} bytes, pinned {size}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", required=True, type=Path)
    parser.add_argument("--pins", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    failed = False
    for (repo, revision), files in load_pins(args.pins).items():
        snapshot = snapshot_dir(args.hub, repo, revision)
        issues = problems(snapshot, files)
        if issues and not args.check_only:
            from huggingface_hub import snapshot_download

            print(f"{repo}@{revision}: fetching ({'; '.join(issues)})", flush=True)
            got = Path(snapshot_download(repo_id=repo, revision=revision, cache_dir=str(args.hub)))
            if got.resolve() != snapshot.resolve():
                print(f"::error title=unexpected snapshot path::{repo} landed at {got}, expected {snapshot}")
                failed = True
                continue
            issues = problems(snapshot, files)
        if issues:
            print(f"::error title=pinned snapshot mismatch::{repo}@{revision}: {'; '.join(issues)}")
            failed = True
        else:
            total = sum(size for _, size in files)
            print(f"{repo}@{revision}: OK ({len(files)} pinned files, {total} bytes) at {snapshot}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
