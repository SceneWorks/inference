#!/usr/bin/env python3.12
"""Materialize pinned snapshots in the hub cache and verify them against a pins TSV.

Pins TSV rows: ``repo<TAB>revision<TAB>path<TAB>bytes<TAB>sha256`` (``#`` lines are comments).

Default (W1, models.tsv): FIRST a disk precheck (``disk_precheck``): the bytes of every pinned
file still missing or of the wrong size, plus ``--reserve-gib`` (build.sh passes 20), against the
hub volume's free space; free and needed bytes are logged either way and a shortfall fails before
any download. Then, for each (repo, revision), if `<hub>/models--<org>--<name>/snapshots/
<revision>` already holds every pinned file at its pinned byte size, nothing is fetched. Otherwise the
revision is downloaded THROUGH the cache (`snapshot_download(cache_dir=<hub>)`, never `local_dir`)
and verified again. Byte sizes are the cheap identity check here; the campaign parent re-inventories
every snapshot (sha256 + bytes) before it starts a row, so a wrong byte that happens to keep its size
is still refused there.

`--only-pinned` (W2, models-w2.tsv): fetch exactly the pinned files that are missing or have the
wrong size, one `hf_hub_download` each (through the cache), and nothing else. BEFORE any network
access it totals the bytes still to fetch and compares them (plus `--reserve-gib`) with the free
space of the hub's volume; when they do not fit it lists every missing file with its size and fails
without downloading anything. `--verify-sha256` then hashes every pinned file (pre-existing ones
included) against the pinned sha256.

`--check-only` never touches the network (used by every campaign job before it launches).
`--report` prints the missing-bytes table against free space and exits 0; it never downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

GIB = 1 << 30


def load_pins(path: Path) -> dict[tuple[str, str], list[tuple[str, int, str]]]:
    pins: dict[tuple[str, str], list[tuple[str, int, str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        repo, revision, name, size, sha256 = line.split("\t")
        pins.setdefault((repo, revision), []).append((name, int(size), sha256))
    return pins


def snapshot_dir(hub: Path, repo: str, revision: str) -> Path:
    return hub / ("models--" + repo.replace("/", "--")) / "snapshots" / revision


def problems(snapshot: Path, files: list[tuple[str, int, str]]) -> list[str]:
    found = []
    for name, size, _sha256 in files:
        path = snapshot / name
        if not path.is_file():
            found.append(f"{name}: missing")
        elif path.stat().st_size != size:
            found.append(f"{name}: {path.stat().st_size} bytes, pinned {size}")
    return found


def missing_files(hub: Path, pins) -> list[tuple[str, str, str, int, bool]]:
    """(repo, revision, path, bytes, present-with-wrong-size) for every pinned file not in place."""
    missing = []
    for (repo, revision), files in pins.items():
        snapshot = snapshot_dir(hub, repo, revision)
        for name, size, _sha256 in files:
            path = snapshot / name
            if not path.is_file():
                missing.append((repo, revision, name, size, False))
            elif path.stat().st_size != size:
                missing.append((repo, revision, name, size, True))
    return missing


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disk_table(hub: Path, pins, reserve_gib: float) -> tuple[bool, list[str]]:
    """Missing bytes vs free space on the hub's volume. Returns (fits, report lines)."""
    missing = missing_files(hub, pins)
    need = sum(item[3] for item in missing)
    total = sum(size for files in pins.values() for _, size, _ in files)
    free = shutil.disk_usage(hub).free
    reserve = int(reserve_gib * GIB)
    fits = need == 0 or need + reserve <= free
    lines = [
        f"pinned: {sum(len(f) for f in pins.values())} files, {total} bytes ({total / GIB:.2f} GiB)",
        f"missing or wrong-size: {len(missing)} files, {need} bytes ({need / GIB:.2f} GiB)",
        f"free on the hub volume ({hub}): {free} bytes ({free / GIB:.2f} GiB); reserve kept free: {reserve_gib:g} GiB",
    ]
    by_repo: dict[tuple[str, str], int] = {}
    for repo, revision, _name, size, _wrong in missing:
        by_repo[(repo, revision)] = by_repo.get((repo, revision), 0) + size
    for (repo, revision), size in sorted(by_repo.items()):
        lines.append(f"  {repo}@{revision[:12]}: {size / GIB:.2f} GiB to fetch")
    for repo, revision, name, size, wrong in missing:
        lines.append(f"    {repo}@{revision[:12]} {name} {size} bytes{' (present, WRONG SIZE)' if wrong else ''}")
    if not fits:
        lines.append(
            f"SHORTFALL: need {(need + reserve - free) / GIB:.2f} GiB more free space on {hub} "
            "(free space, move the hub, or pre-seed the files) -- nothing was downloaded"
        )
    return fits, lines


def disk_precheck(hub: Path, pins, reserve_gib: float) -> bool:
    """Log the missing-bytes table against free space; False (with an error) when it does not fit."""
    fits, lines = disk_table(hub, pins, reserve_gib)
    for line in lines:
        print(line, flush=True)
    if not fits:
        print("::error title=not enough disk for the pinned W1 snapshots::" + " | ".join(
            line.strip() for line in lines if not line.startswith("    ")))
    return fits


def fetch_only_pinned(hub: Path, pins, reserve_gib: float) -> bool:
    fits, lines = disk_table(hub, pins, reserve_gib)
    for line in lines:
        print(line, flush=True)
    if not fits:
        print("::error title=not enough disk for the pinned assets::" + " | ".join(
            line.strip() for line in lines if not line.startswith("    ")))
        return False
    missing = missing_files(hub, pins)
    if not missing:
        return True
    from huggingface_hub import hf_hub_download

    for repo, revision, name, size, wrong in missing:
        note = ", replacing a wrong-size copy" if wrong else ""
        print(f"{repo}@{revision}: fetching {name} ({size} bytes{note})", flush=True)
        got = Path(hf_hub_download(
            repo_id=repo, filename=name, revision=revision, cache_dir=str(hub), force_download=wrong,
        ))
        expected = snapshot_dir(hub, repo, revision) / name
        if got.absolute() != expected.absolute():
            print(f"::error title=unexpected snapshot path::{repo} {name} landed at {got}, expected {expected}")
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", required=True, type=Path)
    parser.add_argument("--pins", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--report", action="store_true")
    mode.add_argument("--only-pinned", action="store_true")
    parser.add_argument("--reserve-gib", type=float, default=10.0)
    parser.add_argument("--verify-sha256", action="store_true")
    args = parser.parse_args()
    pins = load_pins(args.pins)

    if args.report:
        if not args.hub.is_dir():
            print(f"hub cache {args.hub} does not exist")
            return 0
        _fits, lines = disk_table(args.hub, pins, args.reserve_gib)
        print("\n".join(lines))
        return 0
    if args.only_pinned and not fetch_only_pinned(args.hub, pins, args.reserve_gib):
        return 1
    if not (args.check_only or args.only_pinned) and not disk_precheck(args.hub, pins, args.reserve_gib):
        return 1

    failed = False
    for (repo, revision), files in pins.items():
        snapshot = snapshot_dir(args.hub, repo, revision)
        issues = problems(snapshot, files)
        if issues and not args.check_only and not args.only_pinned:
            from huggingface_hub import snapshot_download

            print(f"{repo}@{revision}: fetching ({'; '.join(issues)})", flush=True)
            got = Path(snapshot_download(repo_id=repo, revision=revision, cache_dir=str(args.hub)))
            if got.resolve() != snapshot.resolve():
                print(f"::error title=unexpected snapshot path::{repo} landed at {got}, expected {snapshot}")
                failed = True
                continue
            issues = problems(snapshot, files)
        if not issues and args.verify_sha256:
            issues = [
                f"{name}: sha256 differs from the pin" for name, _size, sha256 in files
                if sha256_file(snapshot / name) != sha256
            ]
        if issues:
            print(f"::error title=pinned snapshot mismatch::{repo}@{revision}: {'; '.join(issues)}")
            failed = True
        else:
            total = sum(size for _, size, _ in files)
            checked = "sha256 + bytes" if args.verify_sha256 else "bytes"
            print(f"{repo}@{revision}: OK ({len(files)} pinned files, {total} bytes, {checked}) at {snapshot}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
