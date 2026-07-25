#!/usr/bin/env python3
"""Regenerate and verify the shared CPU-only Mage TE + VAE real-weight oracle bundle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
MLX = ROOT / "crates/media/mlx-gen"
GEOMETRIES = ("256", "992", "1024", "2048", "512x2048", "768x1280", "768x1152")
TE_FILE = "mage_flow_te_golden.safetensors"
VAE_FILES = tuple(f"mage_flow_vae_f32_{geometry}.safetensors" for geometry in GEOMETRIES)
EXPECTED_FILES = (TE_FILE, *VAE_FILES)
MANIFEST = "mage_oracles_manifest.json"
TE_SCHEMA = {
    "cfg": ("F32", [1]),
    "drop_idx": ("I32", [2]),
    "edit_drop_idx": ("I32", [1]),
    "edit_hidden_full": ("F32", [157, 2560]),
    "edit_image_grid_thw": ("I64", [1, 3]),
    "edit_input_ids": ("I64", [157]),
    "edit_pixel_values": ("F32", [288, 1536]),
    "edit_txt": ("F32", [93, 2560]),
    "edit_txt_len": ("I32", [1]),
    "edit_vec": ("F32", [1, 2560]),
    "edit_vl_ref_u8": ("U8", [384, 192, 3]),
    "gen_drop_idx": ("I32", [1]),
    "gen_hidden_full": ("F32", [94, 2560]),
    "gen_input_ids": ("I64", [54]),
    "gen_txt": ("F32", [20, 2560]),
    "gen_txt_len": ("I32", [1]),
    "gen_vec": ("F32", [1, 2560]),
    "geometry": ("I32", [4]),
    "gs_key": ("I64", [1]),
    "neg_txt": ("F32", [6, 2560]),
    "neg_txt_len": ("I32", [1]),
    "neg_vec": ("F32", [1, 2560]),
    "seed": ("I64", [1]),
    "static_shift": ("F32", [1]),
}
REFERENCE_PACKAGES = {
    "accelerate": "1.13.0",
    "diffusers": "0.38.0",
    "einops": "0.8.2",
    "loguru": "0.7.3",
    "numpy": "2.4.3",
    "pillow": "12.3.0",
    "pydantic": "2.12.5",
    "safetensors": "0.8.0",
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "transformers": "5.5.0",
    "typing_extensions": "4.15.0",
}
REFERENCE_PYTHON = (3, 12, 11)


class InvalidOracle(RuntimeError):
    pass


def _validate_python_version(version: tuple[int, int, int]) -> None:
    if version != REFERENCE_PYTHON:
        raise InvalidOracle(
            f"reference Python is {'.'.join(map(str, version))}, "
            f"expected {'.'.join(map(str, REFERENCE_PYTHON))}"
        )


def _validate_reference_environment() -> dict[str, str]:
    _validate_python_version(sys.version_info[:3])
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


def _inspect(
    path: Path,
) -> tuple[dict[str, str], set[str], dict[str, list[int]], dict[str, str]]:
    try:
        with safe_open(path, framework="numpy") as handle:
            metadata = handle.metadata() or {}
            keys = set(handle.keys())
            shapes = {key: list(handle.get_slice(key).get_shape()) for key in keys}
            dtypes = {key: handle.get_slice(key).get_dtype() for key in keys}
    except Exception as error:
        raise InvalidOracle(f"{path.name} is not a readable safetensors bundle: {error}") from error
    return metadata, keys, shapes, dtypes


def _vae_schema(height: int, width: int) -> dict[str, tuple[str, list[int]]]:
    latent = [1, 128, height // 16, width // 16]
    image = [1, 3, height, width]
    return {
        "dec_from_latent": ("F32", image),
        "dec_from_synth": ("F32", image),
        "enc_latent": ("F32", latent),
        "enc_logvar": ("F32", latent),
        "enc_mean": ("F32", latent),
        "geometry": ("I32", [2]),
        "image_u8": ("U8", [height, width, 3]),
        "pixels": ("F32", image),
        "seed": ("I64", [1]),
        "synth_latent": ("F32", latent),
    }


def _validate_schema(
    name: str,
    schema: dict[str, tuple[str, list[int]]],
    keys: set[str],
    shapes: dict[str, list[int]],
    dtypes: dict[str, str],
) -> None:
    if keys != set(schema):
        raise InvalidOracle(
            f"{name} tensor population mismatch: missing={sorted(set(schema) - keys)}, "
            f"unexpected={sorted(keys - set(schema))}"
        )
    for key, (expected_dtype, expected_shape) in schema.items():
        if dtypes.get(key) != expected_dtype or shapes.get(key) != expected_shape:
            raise InvalidOracle(
                f"{name}:{key} is {dtypes.get(key)}/{shapes.get(key)}, "
                f"expected {expected_dtype}/{expected_shape}"
            )


def _validate_files(output: Path, revision: str) -> list[dict[str, object]]:
    records = []
    for name in EXPECTED_FILES:
        path = output / name
        if not path.is_file():
            raise InvalidOracle(f"required Mage oracle is missing: {path}")
        metadata, keys, shapes, dtypes = _inspect(path)
        if name == TE_FILE:
            if metadata.get("device") != "cpu" or metadata.get("gen_revision") != revision:
                raise InvalidOracle(
                    f"{name} metadata mismatch: device={metadata.get('device')!r}, "
                    f"gen_revision={metadata.get('gen_revision')!r}, expected cpu/{revision}"
                )
            _validate_schema(name, TE_SCHEMA, keys, shapes, dtypes)
        else:
            geometry = name.removeprefix("mage_flow_vae_f32_").removesuffix(".safetensors")
            expected_hw = list(_geometry(geometry))
            if (
                metadata.get("device") != "cpu"
                or metadata.get("dtype") != "float32"
                or metadata.get("revision") != revision
            ):
                raise InvalidOracle(
                    f"{name} metadata/schema mismatch for cpu/f32 revision {revision}"
                )
            with safe_open(path, framework="numpy") as handle:
                actual_hw = handle.get_tensor("geometry").astype(np.int64).tolist()
            if actual_hw != expected_hw:
                raise InvalidOracle(f"{name} geometry is {actual_hw}, expected {expected_hw}")
            _validate_schema(
                name, _vae_schema(*expected_hw), keys, shapes, dtypes
            )
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
        schema = _vae_schema(256, 256)
        shapes = {key: shape for key, (_, shape) in schema.items()}
        dtypes = {key: dtype for key, (dtype, _) in schema.items()}
        _validate_schema("synthetic", schema, set(schema), shapes, dtypes)

        wrong_dtype = dict(dtypes)
        wrong_dtype["pixels"] = "F16"
        wrong_shape = dict(shapes)
        wrong_shape["enc_mean"] = [1, 128, 15, 16]

        missing = root / "missing"
        missing.mkdir()
        corrupt = root / "corrupt.safetensors"
        corrupt.write_bytes(b"not safetensors")
        stale = root / "stale"
        stale.mkdir()
        _write_manifest(stale, "b" * 40, [], 0)

        cases = (
            ("Python patch mismatch", lambda: _validate_python_version((3, 12, 12))),
            ("absent", lambda: _validate_files(missing, revision)),
            ("corrupt", lambda: _inspect(corrupt)),
            ("revision mismatch", lambda: verify(stale, snapshot)),
            (
                "wrong dtype",
                lambda: _validate_schema("synthetic", schema, set(schema), shapes, wrong_dtype),
            ),
            (
                "wrong shape",
                lambda: _validate_schema("synthetic", schema, set(schema), wrong_shape, dtypes),
            ),
        )
        for label, mutation in cases:
            try:
                mutation()
            except InvalidOracle:
                continue
            raise AssertionError(f"self-test failed to reject {label}")
    print(
        "Mage oracle provisioning self-test PASS: "
        "Python-patch/absent/corrupt/revision/dtype/shape mutations rejected"
    )


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
