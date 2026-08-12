#!/usr/bin/env python3
"""Atomically promote verified Mage manifest migrations into an operator seed."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any


FILES = (
    "mage_flow_te_golden.safetensors",
    "mage_flow_edit_golden.safetensors",
    "mage_flow_edit_base_golden.safetensors",
    "mage_flow_edit_turbo_golden.safetensors",
    "mage_flow_vae_f32_256.safetensors",
    "mage_flow_vae_f32_992.safetensors",
    "mage_flow_vae_f32_1024.safetensors",
    "mage_flow_vae_f32_2048.safetensors",
    "mage_flow_vae_f32_512x2048.safetensors",
    "mage_flow_vae_f32_768x1280.safetensors",
    "mage_flow_vae_f32_768x1152.safetensors",
    "mage_flow_dit_golden.safetensors",
    "mage_flow_e2e_golden.safetensors",
    "mage_flow_e2e_golden.png",
    "mage_flow_edit_golden.png",
    "mage_oracles_manifest.json",
    "mage_edit_oracle_manifest.json",
    "mage_edit_variants_manifest.json",
    "mage_candle_oracles_manifest.json",
    "mage_candle_transfer_manifest.json",
)
TARGETS = (
    "mage_edit_variants_manifest.json",
    "mage_candle_transfer_manifest.json",
)
UNCHANGED = tuple(name for name in FILES if name not in TARGETS)
SCHEMA = 1
MANAGED_BY = "sceneworks.promote_mage_oracle_seed"
CLAIM = "claim.json"
READY = "ready.json"


class PromotionError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    # ast-encoding-guard: os.open is descriptor-level binary I/O, not text decoding.
    descriptor = os.open(path, os.O_RDONLY)  # type: ignore[arg-type]
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, content: bytes, mode: int = 0o600) -> None:
    temporary = path.parent / f".{path.name}.tmp"
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        # ast-encoding-guard: os.open is descriptor-level binary I/O, not text decoding.
        descriptor = os.open(temporary, flags, mode)  # type: ignore[arg-type]
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise PromotionError(f"short write while publishing {path}")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        temporary.unlink(missing_ok=True)
        raise PromotionError(f"cannot atomically publish {path}: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _require_root(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(path))
    if path.is_symlink() or not path.is_dir():
        raise PromotionError(f"{label} must be an existing, non-symlink directory: {path}")
    resolved = path.resolve(strict=True)
    if resolved == Path(resolved.anchor):
        raise PromotionError(f"{label} cannot be a filesystem root: {resolved}")
    return resolved


def _require_exclusive_regular(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PromotionError(f"cannot inspect {label}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise PromotionError(f"{label} must be a regular, non-symlink file: {path}")
    if metadata.st_nlink != 1:
        raise PromotionError(
            f"{label} must have exactly one hard link, found {metadata.st_nlink}: {path}"
        )
    return metadata


def inventory(root: Path, label: str) -> dict[str, dict[str, int | str]]:
    try:
        names = {entry.name for entry in root.iterdir()}
    except OSError as error:
        raise PromotionError(f"cannot enumerate {label}: {error}") from error
    expected = set(FILES)
    if names != expected:
        raise PromotionError(
            f"{label} inventory is not exact; missing={sorted(expected - names)}, "
            f"extra={sorted(names - expected)}"
        )
    result: dict[str, dict[str, int | str]] = {}
    for name in FILES:
        path = root / name
        metadata = _require_exclusive_regular(path, f"{label} file {name}")
        result[name] = {"bytes": metadata.st_size, "sha256": sha256(path)}
    return result


def unchanged_digest(records: dict[str, dict[str, int | str]]) -> str:
    digest = hashlib.sha256()
    for name in UNCHANGED:
        record = records[name]
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode())
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _managed_paths(seed: Path) -> tuple[Path, Path]:
    suffix = hashlib.sha256(str(seed).encode()).hexdigest()[:20]
    return (
        seed.parent / f".sceneworks-mage-seed-promotion-{suffix}",
        seed.parent / f".sceneworks-mage-seed-promotion-{suffix}.lock",
    )


def _claim_document(seed: Path) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "managedBy": MANAGED_BY,
        "seed": str(seed),
        "targets": list(TARGETS),
    }


def _blob_name(kind: str, target: str) -> str:
    return f"{kind}-{target}"


def _install_name(target: str) -> str:
    return f"install-{target}"


def _allowed_transaction_names() -> set[str]:
    names = {CLAIM, READY}
    for target in TARGETS:
        names.update(
            {
                _blob_name("old", target),
                _blob_name("new", target),
                _install_name(target),
            }
        )
    return names | {f".{name}.tmp" for name in names}


def _discard_atomic_temps(transaction: Path) -> None:
    names = _transaction_entries(transaction)
    temporary_names = sorted(name for name in names if name.startswith(".") and name.endswith(".tmp"))
    for name in temporary_names:
        (transaction / name).unlink()
    if temporary_names:
        _fsync_directory(transaction)


def _load_json(path: Path, label: str) -> Any:
    _require_exclusive_regular(path, label)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PromotionError(f"invalid {label}: {error}") from error


def _transaction_entries(transaction: Path) -> set[str]:
    if transaction.is_symlink() or not transaction.is_dir():
        raise PromotionError(f"unsafe Mage seed promotion transaction: {transaction}")
    names = {entry.name for entry in transaction.iterdir()}
    unknown = names - _allowed_transaction_names()
    if unknown:
        raise PromotionError(
            f"Mage seed promotion transaction contains unknown entries: {sorted(unknown)}"
        )
    for name in names:
        _require_exclusive_regular(
            transaction / name, f"Mage seed promotion transaction entry {name}"
        )
    return names


def _remove_transaction(transaction: Path) -> None:
    names = _transaction_entries(transaction)
    # Once both persistent targets are verified, remove READY first. A cleanup interruption then
    # looks like a claimed pre-ready transaction: recovery removes the remaining private blobs
    # without touching the already-consistent seed. While READY exists, every blob it needs must
    # remain present so a hard interruption can always complete forward.
    ordered: list[str] = []
    if READY in names:
        ordered.append(READY)
    ordered.extend(name for name in sorted(names) if name not in {READY, CLAIM})
    if CLAIM in names:
        ordered.append(CLAIM)
    for index, name in enumerate(ordered):
        (transaction / name).unlink()
        if index == 0 and name == READY:
            _fsync_directory(transaction)
    transaction.rmdir()
    _fsync_directory(transaction.parent)


def _validate_ready(value: Any, seed: Path) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "managedBy",
        "seed",
        "source",
        "runnerName",
        "slot",
        "revision",
        "targets",
        "unchangedDigest",
    }:
        raise PromotionError("Mage seed promotion ready journal population is invalid")
    if (
        value["schema"] != SCHEMA
        or value["managedBy"] != MANAGED_BY
        or value["seed"] != str(seed)
        or not isinstance(value["source"], str)
        or not value["source"].startswith("/")
        or not isinstance(value["runnerName"], str)
        or not value["runnerName"].strip()
        or value["slot"] not in {"primary", "secondary", "single"}
        or not isinstance(value["revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", value["revision"]) is None
        or not isinstance(value["unchangedDigest"], str)
        or re.fullmatch(r"[0-9a-f]{64}", value["unchangedDigest"]) is None
    ):
        raise PromotionError("Mage seed promotion ready journal identity is invalid")
    targets = value["targets"]
    if (
        not isinstance(targets, list)
        or any(not isinstance(record, dict) for record in targets)
        or [record.get("name") for record in targets] != list(TARGETS)
    ):
        raise PromotionError("Mage seed promotion ready journal target order is invalid")
    for record in targets:
        if not isinstance(record, dict) or set(record) != {"name", "old", "new"}:
            raise PromotionError("Mage seed promotion target record is invalid")
        for side in ("old", "new"):
            payload = record[side]
            if (
                not isinstance(payload, dict)
                or set(payload) != {"bytes", "sha256"}
                or type(payload["bytes"]) is not int
                or payload["bytes"] <= 0
                or not isinstance(payload["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", payload["sha256"]) is None
            ):
                raise PromotionError("Mage seed promotion target payload is invalid")
    return value


def _validate_transaction_blobs(transaction: Path, ready: dict[str, Any]) -> None:
    names = _transaction_entries(transaction)
    required = {CLAIM, READY}
    for record in ready["targets"]:
        for side in ("old", "new"):
            name = _blob_name(side, record["name"])
            required.add(name)
            metadata = _require_exclusive_regular(
                transaction / name, f"Mage seed promotion {side} blob"
            )
            expected = record[side]
            if metadata.st_size != expected["bytes"] or sha256(transaction / name) != expected[
                "sha256"
            ]:
                raise PromotionError(f"Mage seed promotion {side} blob is stale: {name}")
    missing = required - names
    if missing:
        raise PromotionError(
            f"Mage seed promotion transaction is incomplete: {sorted(missing)}"
        )


def _replace_target(seed: Path, transaction: Path, record: dict[str, Any]) -> None:
    target = seed / record["name"]
    if target.exists() or target.is_symlink():
        _require_exclusive_regular(target, f"operator seed target {record['name']}")
        current = sha256(target)
        if current == record["new"]["sha256"]:
            return
        if current != record["old"]["sha256"]:
            raise PromotionError(
                f"operator seed target changed outside the transaction: {record['name']}"
            )
    source = transaction / _blob_name("new", record["name"])
    _require_exclusive_regular(source, f"new promotion blob for {record['name']}")
    install = transaction / _install_name(record["name"])
    _atomic_write(install, source.read_bytes(), 0o644)
    os.replace(install, target)
    _fsync_directory(seed)
    _require_exclusive_regular(target, f"promoted operator seed target {record['name']}")
    if sha256(target) != record["new"]["sha256"]:
        raise PromotionError(f"promoted operator seed target is stale: {record['name']}")


def _complete_ready_transaction(
    seed: Path, transaction: Path, ready: dict[str, Any]
) -> None:
    _validate_transaction_blobs(transaction, ready)
    # Ordering is part of the contract: publish the inner manifest before the transfer receipt
    # that hashes it. A crash between the two leaves READY in place and is completed forward.
    for record in ready["targets"]:
        _replace_target(seed, transaction, record)
    records = inventory(seed, "promoted operator Mage oracle seed")
    if unchanged_digest(records) != ready["unchangedDigest"]:
        raise PromotionError("unchanged Mage oracle payload drifted during promotion")
    for record in ready["targets"]:
        if records[record["name"]] != record["new"]:
            raise PromotionError(f"promoted target does not match journal: {record['name']}")


def _recover_locked(
    seed: Path, transaction: Path, *, cleanup_ready: bool = True
) -> tuple[bool, dict[str, Any] | None]:
    if not transaction.exists() and not transaction.is_symlink():
        return False, None
    if transaction.is_symlink() or not transaction.is_dir():
        raise PromotionError(f"unsafe Mage seed promotion transaction: {transaction}")
    names = _transaction_entries(transaction)
    if CLAIM not in names:
        # The first claim publication can itself be interrupted before its atomic rename. This
        # exact deterministic temp is the only unclaimed content we can attribute to this tool;
        # reclaim it without touching the persistent seed. Every other unclaimed population stays
        # fail-closed for operator inspection.
        if names == {f".{CLAIM}.tmp"}:
            (transaction / f".{CLAIM}.tmp").unlink()
            transaction.rmdir()
            _fsync_directory(transaction.parent)
            return True, None
        if names:
            raise PromotionError(
                f"refusing to remove an unclaimed Mage seed promotion transaction: {transaction}"
            )
        transaction.rmdir()
        _fsync_directory(transaction.parent)
        return True, None
    if _load_json(transaction / CLAIM, "Mage seed promotion claim") != _claim_document(seed):
        raise PromotionError("Mage seed promotion transaction claim does not match this seed")
    _discard_atomic_temps(transaction)
    names = _transaction_entries(transaction)
    if READY not in names:
        # No persistent seed mutation can happen before READY is fsynced. Reclaim only this exact,
        # claimed transaction and leave the seed untouched.
        _remove_transaction(transaction)
        return True, None
    ready = _validate_ready(
        _load_json(transaction / READY, "Mage seed promotion ready journal"), seed
    )
    _complete_ready_transaction(seed, transaction, ready)
    if cleanup_ready:
        _remove_transaction(transaction)
    return True, ready


class _SeedLock:
    def __init__(self, path: Path):
        self.path = path
        self.descriptor: int | None = None

    def __enter__(self) -> "_SeedLock":
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            # ast-encoding-guard: os.open is descriptor-level binary I/O, not text decoding.
            descriptor = os.open(self.path, flags, 0o600)  # type: ignore[arg-type]
        except OSError as error:
            raise PromotionError(f"cannot open Mage seed promotion lock: {error}") from error
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            raise PromotionError("Mage seed promotion lock is not an exclusive regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            path_metadata = self.path.lstat()
        except OSError as error:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            raise PromotionError(f"Mage seed promotion lock path changed: {error}") from error
        if (
            not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_dev != metadata.st_dev
            or path_metadata.st_ino != metadata.st_ino
            or path_metadata.st_nlink != 1
        ):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            raise PromotionError("Mage seed promotion lock path changed while acquiring it")
        self.descriptor = descriptor
        return self

    def __exit__(self, *_args: object) -> None:
        assert self.descriptor is not None
        fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        os.close(self.descriptor)
        self.descriptor = None


def _receipt_document(
    ready: dict[str, Any],
    *,
    status: str,
    recovered_by_runner: str | None = None,
    recovered_by_slot: str | None = None,
    recovered_by_revision: str | None = None,
) -> dict[str, Any]:
    receipt = {
        "schema": SCHEMA,
        "operation": MANAGED_BY,
        "status": status,
        "runnerName": ready["runnerName"],
        "slot": ready["slot"],
        "revision": ready["revision"],
        "source": ready["source"],
        "seed": ready["seed"],
        "unchangedFileCount": len(UNCHANGED),
        "unchangedDigest": ready["unchangedDigest"],
        "targets": ready["targets"],
    }
    if status == "recovered":
        receipt["recoveredBy"] = {
            "runnerName": recovered_by_runner,
            "slot": recovered_by_slot,
            "revision": recovered_by_revision,
        }
    return receipt


def _validate_receipt(value: Any) -> dict[str, Any]:
    base_keys = {
        "schema",
        "operation",
        "status",
        "runnerName",
        "slot",
        "revision",
        "source",
        "seed",
        "unchangedFileCount",
        "unchangedDigest",
        "targets",
    }
    if not isinstance(value, dict) or value.get("status") not in {
        "promoted",
        "already-current",
        "recovered",
    }:
        raise PromotionError("Mage seed promotion receipt status is invalid")
    expected_keys = base_keys | ({"recoveredBy"} if value["status"] == "recovered" else set())
    if set(value) != expected_keys:
        raise PromotionError("Mage seed promotion receipt population is invalid")
    if not isinstance(value["seed"], str) or not value["seed"].startswith("/"):
        raise PromotionError("Mage seed promotion receipt seed is invalid")
    ready = _validate_ready(
        {
            "schema": value["schema"],
            "managedBy": value["operation"],
            "seed": value["seed"],
            "source": value["source"],
            "runnerName": value["runnerName"],
            "slot": value["slot"],
            "revision": value["revision"],
            "targets": value["targets"],
            "unchangedDigest": value["unchangedDigest"],
        },
        Path(value["seed"]),
    )
    if value["unchangedFileCount"] != len(UNCHANGED):
        raise PromotionError("Mage seed promotion receipt unchanged-file count is invalid")
    if value["status"] == "recovered":
        recovered_by = value["recoveredBy"]
        if (
            not isinstance(recovered_by, dict)
            or set(recovered_by) != {"runnerName", "slot", "revision"}
            or not isinstance(recovered_by["runnerName"], str)
            or not recovered_by["runnerName"].strip()
            or recovered_by["slot"] not in {"primary", "secondary", "single"}
            or not isinstance(recovered_by["revision"], str)
            or re.fullmatch(r"[0-9a-f]{40}", recovered_by["revision"]) is None
        ):
            raise PromotionError("Mage seed promotion recovery receipt identity is invalid")
    return {**value, "targets": ready["targets"]}


def verify_receipt(receipt_dir_path: Path, revision: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise PromotionError(f"receipt verification revision is not exact 40-hex: {revision}")
    receipt_dir = _require_root(receipt_dir_path, "Mage seed promotion receipt directory")
    expected_names = {"mage-seed-promotion-single.json"}
    names = {entry.name for entry in receipt_dir.iterdir()}
    if names != expected_names:
        raise PromotionError(
            "Mage seed promotion receipt inventory is not exact; "
            f"missing={sorted(expected_names - names)}, extra={sorted(names - expected_names)}"
        )

    receipt = _validate_receipt(
        _load_json(
            receipt_dir / "mage-seed-promotion-single.json",
            "Mage seed promotion receipt",
        )
    )
    certifier = (
        receipt["recoveredBy"]
        if receipt["status"] == "recovered"
        else {
            "runnerName": receipt["runnerName"],
            "slot": receipt["slot"],
            "revision": receipt["revision"],
        }
    )
    if certifier["slot"] != "single" or certifier["revision"] != revision:
        raise PromotionError("Mage seed promotion receipt does not certify this exact run")


def _verified_receipt_destination(
    path: Path,
    *,
    forbidden_roots: tuple[Path, ...],
    forbidden_files: tuple[Path, ...] = (),
) -> Path:
    path = Path(os.path.abspath(path))
    if path.exists() or path.is_symlink():
        raise PromotionError(f"promotion receipt destination must not already exist: {path}")
    parent = _require_root(path.parent, "promotion receipt parent")
    destination = parent / path.name
    if any(destination == root or destination.is_relative_to(root) for root in forbidden_roots):
        raise PromotionError(f"promotion receipt destination overlaps managed payload: {destination}")
    if destination in forbidden_files:
        raise PromotionError(f"promotion receipt destination overlaps managed state: {destination}")
    return destination


def recover_only(
    seed_path: Path,
    *,
    receipt_path: Path | None = None,
    runner_name: str | None = None,
    slot: str | None = None,
    revision: str | None = None,
) -> bool:
    seed = _require_root(seed_path, "operator Mage oracle seed")
    transaction, lock = _managed_paths(seed)
    receipt = (
        _verified_receipt_destination(
            receipt_path,
            forbidden_roots=(seed, transaction),
            forbidden_files=(lock,),
        )
        if receipt_path is not None
        else None
    )
    with _SeedLock(lock):
        recovered, ready = _recover_locked(seed, transaction, cleanup_ready=False)
        inventory(seed, "operator Mage oracle seed")
        if ready is not None:
            if (
                receipt is None
                or not runner_name
                or slot not in {"primary", "secondary", "single"}
                or revision is None
                or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            ):
                raise PromotionError(
                    "a committed promotion recovery requires an exact recovery receipt identity"
                )
            _atomic_write(
                receipt,
                _json_bytes(
                    _receipt_document(
                        ready,
                        status="recovered",
                        recovered_by_runner=runner_name,
                        recovered_by_slot=slot,
                        recovered_by_revision=revision,
                    )
                ),
                0o644,
            )
            _remove_transaction(transaction)
    return recovered


def promote(
    source_path: Path,
    seed_path: Path,
    expected_old: dict[str, str],
    *,
    runner_name: str,
    slot: str,
    revision: str,
    receipt_path: Path,
    allow_already_current: bool = False,
) -> dict[str, Any]:
    source = _require_root(source_path, "verified temporary Mage oracle bundle")
    seed = _require_root(seed_path, "operator Mage oracle seed")
    try:
        aliases = os.path.samefile(source, seed)
    except OSError as error:
        raise PromotionError(f"cannot compare Mage oracle roots: {error}") from error
    if aliases:
        raise PromotionError("temporary Mage oracle bundle must not alias the operator seed")
    if not runner_name.strip():
        raise PromotionError("runner name is required for a promotion receipt")
    if slot not in {"primary", "secondary", "single"}:
        raise PromotionError(f"invalid Mage seed promotion slot: {slot}")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise PromotionError(f"promotion revision is not exact 40-hex: {revision}")
    if set(expected_old) != set(TARGETS) or any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None for value in expected_old.values()
    ):
        raise PromotionError("expected legacy target hashes are incomplete or malformed")

    transaction, lock = _managed_paths(seed)
    receipt_path = _verified_receipt_destination(
        receipt_path,
        forbidden_roots=(source, seed, transaction),
        forbidden_files=(lock,),
    )
    with _SeedLock(lock):
        _recover_locked(seed, transaction)
        source_records = inventory(source, "verified temporary Mage oracle bundle")
        seed_records = inventory(seed, "operator Mage oracle seed")
        for name in UNCHANGED:
            if source_records[name] != seed_records[name]:
                raise PromotionError(f"non-target Mage oracle differs from operator seed: {name}")
        current_targets = [source_records[name] == seed_records[name] for name in TARGETS]
        if any(current_targets) and not all(current_targets):
            raise PromotionError("operator seed has a mixed legacy/current target population")
        if all(current_targets):
            if not allow_already_current:
                raise PromotionError("Mage seed promotion is a forbidden no-op")
            ready = {
                "schema": SCHEMA,
                "managedBy": MANAGED_BY,
                "seed": str(seed),
                "source": str(source),
                "runnerName": runner_name,
                "slot": slot,
                "revision": revision,
                "targets": [
                    {
                        "name": name,
                        "old": seed_records[name],
                        "new": source_records[name],
                    }
                    for name in TARGETS
                ],
                "unchangedDigest": unchanged_digest(source_records),
            }
            receipt = _receipt_document(ready, status="already-current")
            _atomic_write(receipt_path, _json_bytes(receipt), 0o644)
            return receipt

        for name in TARGETS:
            if seed_records[name]["sha256"] != expected_old[name]:
                raise PromotionError(f"operator seed target changed since import: {name}")

        transaction.mkdir(mode=0o700)
        _fsync_directory(transaction.parent)
        _atomic_write(transaction / CLAIM, _json_bytes(_claim_document(seed)))
        target_records = []
        for name in TARGETS:
            old = seed_records[name]
            new = source_records[name]
            _atomic_write(
                transaction / _blob_name("old", name), (seed / name).read_bytes()
            )
            _atomic_write(
                transaction / _blob_name("new", name), (source / name).read_bytes()
            )
            target_records.append({"name": name, "old": old, "new": new})
        ready = {
            "schema": SCHEMA,
            "managedBy": MANAGED_BY,
            "seed": str(seed),
            "source": str(source),
            "runnerName": runner_name,
            "slot": slot,
            "revision": revision,
            "targets": target_records,
            "unchangedDigest": unchanged_digest(source_records),
        }
        # READY is the write-ahead commit point and must be the final transaction write.
        _atomic_write(transaction / READY, _json_bytes(ready))
        _complete_ready_transaction(seed, transaction, ready)

        receipt = _receipt_document(ready, status="promoted")
        _atomic_write(receipt_path, _json_bytes(receipt), 0o644)
        _remove_transaction(transaction)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--recover-only", action="store_true")
    parser.add_argument("--verify-receipts", type=Path)
    parser.add_argument("--allow-already-current", action="store_true")
    parser.add_argument("--expected-old-edit-sha")
    parser.add_argument("--expected-old-transfer-sha")
    parser.add_argument("--runner-name")
    parser.add_argument("--slot", choices=("primary", "secondary", "single"))
    parser.add_argument("--revision")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        if args.verify_receipts is not None:
            if any(
                value is not None
                for value in (
                    args.seed,
                    args.source,
                    args.expected_old_edit_sha,
                    args.expected_old_transfer_sha,
                    args.runner_name,
                    args.slot,
                    args.receipt,
                )
            ) or args.recover_only or args.allow_already_current:
                raise PromotionError(
                    "verify-receipts accepts only the receipt directory and revision"
                )
            if args.revision is None:
                raise PromotionError("verify-receipts requires an exact revision")
            verify_receipt(args.verify_receipts, args.revision)
            print("verified exact Mage seed promotion receipt from the active runner")
            return 0
        if args.recover_only:
            if any(
                value is not None
                for value in (
                    args.source,
                    args.expected_old_edit_sha,
                    args.expected_old_transfer_sha,
                )
            ) or args.allow_already_current or args.seed is None:
                raise PromotionError("recover-only does not accept source or legacy hashes")
            receipt_identity = (args.receipt, args.runner_name, args.slot, args.revision)
            if any(value is not None for value in receipt_identity) and not all(
                value is not None for value in receipt_identity
            ):
                raise PromotionError(
                    "recover-only receipt, runner, slot, and revision must be supplied together"
                )
            recovered = recover_only(
                args.seed,
                receipt_path=args.receipt,
                runner_name=args.runner_name,
                slot=args.slot,
                revision=args.revision,
            )
            print(
                "recovered interrupted Mage seed promotion"
                if recovered
                else "Mage seed promotion state is clean"
            )
            return 0
        required = {
            "source": args.source,
            "seed": args.seed,
            "expected-old-edit-sha": args.expected_old_edit_sha,
            "expected-old-transfer-sha": args.expected_old_transfer_sha,
            "runner-name": args.runner_name,
            "slot": args.slot,
            "revision": args.revision,
            "receipt": args.receipt,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise PromotionError(f"promotion arguments are missing: {missing}")
        promote(
            args.source,
            args.seed,
            {
                TARGETS[0]: args.expected_old_edit_sha,
                TARGETS[1]: args.expected_old_transfer_sha,
            },
            runner_name=args.runner_name,
            slot=args.slot,
            revision=args.revision,
            receipt_path=args.receipt,
            allow_already_current=args.allow_already_current,
        )
        print(f"promoted exact Mage oracle manifests into {args.seed}")
        return 0
    except PromotionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
