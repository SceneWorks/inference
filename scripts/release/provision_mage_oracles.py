#!/usr/bin/env python3
"""Regenerate and verify the shared CPU-only Mage TE + VAE real-weight oracle bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[2]
MLX = ROOT / "crates/media/mlx-gen"
GEOMETRIES = ("256", "992", "1024", "2048", "512x2048", "768x1280", "768x1152")
TE_FILE = "mage_flow_te_golden.safetensors"
VAE_FILES = tuple(f"mage_flow_vae_f32_{geometry}.safetensors" for geometry in GEOMETRIES)
EXPECTED_FILES = (TE_FILE, *VAE_FILES)
MANIFEST = "mage_oracles_manifest.json"
TE_KEYS = {"gen_hidden_full", "gen_txt", "neg_txt", "edit_hidden_full", "edit_txt"}
VAE_KEYS = {"geometry", "enc_mean", "enc_logvar", "enc_latent", "synth_latent",
            "dec_from_latent", "dec_from_synth", "pixels", "image_u8"}
REFERENCE_PACKAGES = {
    "diffusers": "0.38.0",
    "safetensors": "0.8.0",
    "torch": "2.13.0",
    "transformers": "5.5.0",
}


class InvalidOracle(RuntimeError):
    pass


def _validate_reference_environment() -> dict[str, str]:
    actual = {}
    for package, expected in REFERENCE_PACKAGES.items():
        try:
            actual[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise InvalidOracle(f"pinned reference package is missing: {package}=={expected}") from error
        if actual[package] != expected:
            raise InvalidOracle(
                f"pinned reference package mismatch: {package}=={actual[package]}, expected {expected}"
            )
    return actual


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision(snapshot: Path) -> str:
    revision = snapshot.resolve().name
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise InvalidOracle(f"MAGE_SNAPSHOT must resolve to a 40-hex immutable snapshot, got {snapshot}")
    return revision


def _geometry(value: str) -> tuple[int, int]:
    if "x" in value:
        height, width = value.split("x", 1)
        return int(height), int(width)
    return int(value), int(value)


def _inspect(path: Path) -> tuple[dict[str, str], set[str], dict[str, list[int]]]:
    try:
        with safe_open(path, framework="numpy") as handle:
            metadata = handle.metadata() or {}
            keys = set(handle.keys())
            shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
    except Exception as error:
        raise InvalidOracle(f"{path.name} is not a readable safetensors bundle: {error}") from error
    return metadata, keys, shapes


def _validate_files(output: Path, revision: str) -> list[dict[str, object]]:
    records = []
    for name in EXPECTED_FILES:
        path = output / name
        if not path.is_file():
            raise InvalidOracle(f"required Mage oracle is missing: {path}")
        metadata, keys, shapes = _inspect(path)
        if name == TE_FILE:
            missing = TE_KEYS - keys
            if metadata.get("device") != "cpu" or metadata.get("gen_revision") != revision:
                raise InvalidOracle(
                    f"{name} metadata mismatch: device={metadata.get('device')!r}, "
                    f"gen_revision={metadata.get('gen_revision')!r}, expected cpu/{revision}"
                )
            if missing:
                raise InvalidOracle(f"{name} lacks required tensors: {sorted(missing)}")
        else:
            geometry = name.removeprefix("mage_flow_vae_f32_").removesuffix(".safetensors")
            missing = VAE_KEYS - keys
            expected_hw = list(_geometry(geometry))
            if (
                metadata.get("device") != "cpu"
                or metadata.get("dtype") != "float32"
                or metadata.get("revision") != revision
                or shapes.get("geometry") != [2]
            ):
                raise InvalidOracle(
                    f"{name} metadata/schema mismatch for cpu/f32 revision {revision}"
                )
            with safe_open(path, framework="numpy") as handle:
                actual_hw = handle.get_tensor("geometry").astype(np.int64).tolist()
            if actual_hw != expected_hw:
                raise InvalidOracle(f"{name} geometry is {actual_hw}, expected {expected_hw}")
            if missing:
                raise InvalidOracle(f"{name} lacks required tensors: {sorted(missing)}")
        records.append({"name": name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return records


def _write_manifest(
    output: Path,
    revision: str,
    records: list[dict[str, object]],
    seconds: float,
    reference_environment: dict[str, str] | None = None,
) -> None:
    document = {
        "schema": 1,
        "reference": "microsoft/Mage frozen vendored reference",
        "snapshotRevision": revision,
        "device": "cpu",
        "vaeGeometries": list(GEOMETRIES),
        "generationSeconds": round(seconds, 3),
        "referenceEnvironment": reference_environment or {},
        "files": records,
    }
    (output / MANIFEST).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def verify(output: Path, snapshot: Path) -> None:
    revision = _revision(snapshot)
    manifest_path = output / MANIFEST
    if not manifest_path.is_file():
        raise InvalidOracle(f"required Mage oracle manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise InvalidOracle(f"invalid Mage oracle manifest: {error}") from error
    if (
        manifest.get("schema") != 1
        or manifest.get("device") != "cpu"
        or manifest.get("snapshotRevision") != revision
        or manifest.get("vaeGeometries") != list(GEOMETRIES)
    ):
        raise InvalidOracle(f"Mage oracle manifest does not match cpu/{revision}/{GEOMETRIES}")
    records = _validate_files(output, revision)
    expected = {record["name"]: record for record in manifest.get("files", [])}
    if set(expected) != set(EXPECTED_FILES):
        raise InvalidOracle("Mage oracle manifest file population is incomplete or stale")
    for record in records:
        manifest_record = expected[record["name"]]
        if (
            manifest_record.get("sha256") != record["sha256"]
            or manifest_record.get("bytes") != record["bytes"]
        ):
            raise InvalidOracle(f"{record['name']} hash/size differs from the manifest")
    print(f"verified {len(records)} CPU Mage oracles for revision {revision} under {output}")


def provision(output: Path, snapshot: Path) -> None:
    revision = _revision(snapshot)
    reference_environment = _validate_reference_environment()
    output.mkdir(parents=True, exist_ok=True)
    for name in (*EXPECTED_FILES, MANIFEST):
        (output / name).unlink(missing_ok=True)
    env = {
        **os.environ,
        "MAGE_DEVICE": "cpu",
        "MAGE_SNAPSHOT": str(snapshot),
        "MAGE_GOLDEN_DIR": str(output),
        "MAGE_VAE_SIZES": ",".join(GEOMETRIES),
        "PYTHONPATH": str(MLX / "_vendor"),
    }
    started = time.monotonic()
    subprocess.run(
        [sys.executable, str(MLX / "tools/dump_mage_flow_golden.py"), "--stage", "te"],
        cwd=MLX,
        env=env,
        check=True,
    )
    subprocess.run(
        [sys.executable, str(MLX / "tools/dump_mage_vae_sizes.py")],
        cwd=MLX,
        env=env,
        check=True,
    )
    seconds = time.monotonic() - started
    records = _validate_files(output, revision)
    _write_manifest(output, revision, records, seconds, reference_environment)
    verify(output, snapshot)
    print(f"regenerated shared CPU Mage oracle bundle in {seconds:.1f}s")


def _self_test() -> None:
    revision = "a" * 40
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        snapshot = root / revision
        snapshot.mkdir()
        output = root / "oracles"
        output.mkdir()
        tensors = {
            "gen_hidden_full": np.zeros((3, 2), np.float32),
            "gen_txt": np.zeros((1, 2), np.float32),
            "neg_txt": np.zeros((1, 2), np.float32),
            "edit_hidden_full": np.zeros((2, 2), np.float32),
            "edit_txt": np.zeros((1, 2), np.float32),
        }
        save_file(tensors, output / TE_FILE, metadata={"device": "cpu", "gen_revision": revision})
        for geometry, name in zip(GEOMETRIES, VAE_FILES, strict=True):
            height, width = _geometry(geometry)
            arrays = {key: np.zeros((1,), np.float32) for key in VAE_KEYS - {"geometry"}}
            arrays["geometry"] = np.array([height, width], np.int32)
            save_file(
                arrays,
                output / name,
                metadata={"device": "cpu", "dtype": "float32", "revision": revision},
            )
        records = _validate_files(output, revision)
        _write_manifest(output, revision, records, 0)
        verify(output, snapshot)

        cases = []
        missing = root / "missing"
        shutil.copytree(output, missing)
        (missing / VAE_FILES[-1]).unlink()
        cases.append(("absent", missing))
        corrupt = root / "corrupt"
        shutil.copytree(output, corrupt)
        with (corrupt / TE_FILE).open("r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            handle.write(b"\xff")
        cases.append(("corrupt", corrupt))
        stale = root / "stale"
        shutil.copytree(output, stale)
        document = json.loads((stale / MANIFEST).read_text(encoding="utf-8"))
        document["snapshotRevision"] = "b" * 40
        (stale / MANIFEST).write_text(json.dumps(document), encoding="utf-8")
        cases.append(("revision mismatch", stale))
        for label, candidate in cases:
            try:
                verify(candidate, snapshot)
            except InvalidOracle:
                continue
            raise AssertionError(f"self-test failed to reject {label}")
    print("Mage oracle provisioning self-test PASS: absent/corrupt/revision mismatch rejected")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if not args.snapshot or not args.output:
        parser.error("--snapshot and --output are required")
    try:
        if args.verify_only:
            verify(args.output.resolve(), args.snapshot.resolve())
        else:
            provision(args.output.resolve(), args.snapshot.resolve())
    except InvalidOracle as error:
        print(f"Mage oracle provisioning FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
