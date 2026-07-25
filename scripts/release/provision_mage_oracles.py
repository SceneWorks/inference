#!/usr/bin/env python3
"""Regenerate and verify the shared CPU-only Mage TE + VAE + edit oracle bundle."""

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
EDIT_FILE = "mage_flow_edit_golden.safetensors"
VAE_FILES = tuple(f"mage_flow_vae_f32_{geometry}.safetensors" for geometry in GEOMETRIES)
EXPECTED_FILES = (TE_FILE, EDIT_FILE, *VAE_FILES)
MANIFEST = "mage_oracles_manifest.json"
EDIT_MANIFEST = "mage_edit_oracle_manifest.json"
TE_SCHEMA = {
    "cfg": ("F32", [1]),
    "drop_idx": ("I32", [2]),
    "edit_drop_idx": ("I32", [1]),
    "edit_deepstack_0": ("F32", [72, 2560]),
    "edit_deepstack_1": ("F32", [72, 2560]),
    "edit_deepstack_2": ("F32", [72, 2560]),
    "edit_hidden_full": ("F32", [157, 2560]),
    "edit_image_grid_thw": ("I64", [1, 3]),
    "edit_input_ids": ("I64", [157]),
    "edit_lm_layer_0_pre_inject": ("F32", [1, 157, 2560]),
    "edit_lm_layer_1_pre_inject": ("F32", [1, 157, 2560]),
    "edit_lm_layer_2_pre_inject": ("F32", [1, 157, 2560]),
    "edit_pixel_values": ("F32", [288, 1536]),
    "edit_position_ids": ("I64", [3, 1, 157]),
    "edit_txt": ("F32", [93, 2560]),
    "edit_txt_len": ("I32", [1]),
    "edit_vec": ("F32", [1, 2560]),
    "edit_vision_embeds": ("F32", [72, 2560]),
    "edit_vl_ref_u8": ("U8", [384, 192, 3]),
    "edit_vl_source_u8": ("U8", [2048, 1024, 3]),
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
EDIT_SCHEMA = {
    "cfg": ("F32", [1]),
    "drop_idx": ("I32", [2]),
    "final_latent": ("F32", [1, 128, 16, 16]),
    "final_tokens": ("F32", [1, 256, 128]),
    "geometry": ("I32", [4]),
    "gs_key": ("I64", [1]),
    "image_u8": ("U8", [256, 256, 3]),
    "img_shapes": ("I32", [4, 3]),
    "ref_u8": ("U8", [256, 256, 3]),
    "seed": ("I64", [1]),
    "seq_step0": ("F32", [1, 1024, 128]),
    "seq_step1": ("F32", [1, 1024, 128]),
    "sigmas_30": ("F32", [31]),
    "static_shift": ("F32", [1]),
    "target_tokens": ("I32", [1]),
    "timesteps_30": ("F32", [30]),
}
EDIT_IMG_SHAPES = [[1, 16, 16]] * 4
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
# Keep this aligned with real-weights.yml. The self-hosted macOS oracle runner installs this exact
# standalone interpreter in runner temp via uv, avoiding system Python and global tool caches.
REFERENCE_PYTHON = (3, 12, 10)


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
    resolved = snapshot.resolve()
    marker = resolved / ".sceneworks-model-revision"
    revision = (
        marker.read_text(encoding="utf-8").strip()
        if marker.is_file()
        else resolved.name
    )
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise InvalidOracle(
            "MAGE_SNAPSHOT must resolve to a 40-hex immutable snapshot or carry "
            f".sceneworks-model-revision, got {snapshot}"
        )
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


def _expected_schedule(steps: int) -> tuple[list[float], list[float]]:
    base = [
        1.0 - index * (1.0 - 1.0 / steps) / (steps - 1)
        for index in range(steps)
    ]
    shifted = [6.0 * sigma / (1.0 + 5.0 * sigma) for sigma in base]
    return shifted + [0.0], [sigma * 1000.0 for sigma in shifted]


def _require_ordered_values(
    values: dict[str, object], key: str, expected: list[float], tolerance: float
) -> None:
    actual = values.get(key)
    if (
        not isinstance(actual, list)
        or len(actual) != len(expected)
        or any(abs(float(left) - float(right)) > tolerance for left, right in zip(actual, expected))
    ):
        raise InvalidOracle(f"{key} ordered values are stale")


def _validate_edit_policy_record(values: dict[str, object], checks: dict[str, bool]) -> None:
    expected_values = {
        "geometry": [256, 256, 4, 30],
        "seed": [42],
        "cfg": [5.0],
        "gs_key": [20260720],
        "drop_idx": [34, 64],
        "static_shift": [6.0],
        "target_tokens": [256],
        "img_shapes": EDIT_IMG_SHAPES,
    }
    for key, expected in expected_values.items():
        if values.get(key) != expected:
            raise InvalidOracle(f"{key} value is stale")
    sigmas, timesteps = _expected_schedule(30)
    _require_ordered_values(values, "sigmas_30", sigmas, 1.0e-4)
    _require_ordered_values(values, "timesteps_30", timesteps, 1.0e-3)
    failed = [name for name, passed in checks.items() if passed is not True]
    if failed:
        raise InvalidOracle(f"load-bearing edit mutation/value checks failed: {failed}")


def _validate_primary_edit_policy(path: Path) -> None:
    with safe_open(path, framework="numpy") as handle:
        values = {
            key: handle.get_tensor(key).tolist()
            for key in (
                "geometry",
                "seed",
                "cfg",
                "gs_key",
                "drop_idx",
                "static_shift",
                "target_tokens",
                "img_shapes",
                "sigmas_30",
                "timesteps_30",
            )
        }
        final = handle.get_tensor("final_tokens")
        image = handle.get_tensor("image_u8")
        reference = handle.get_tensor("ref_u8")
        step0 = handle.get_tensor("seq_step0")
        step1 = handle.get_tensor("seq_step1")
    checks = {
        "trajectoryChanges": not np.array_equal(step0, step1),
        "targetDiffersReference": not np.array_equal(
            step0[:, :256, :], step0[:, 256:512, :]
        ),
        "finalChangesInitial": not np.array_equal(final, step0[:, :256, :]),
        "imageDiscriminating": (
            int(image.max()) - int(image.min()) >= 64 and float(image.std()) > 5.0
        ),
        "referenceDiscriminating": (
            int(reference.max()) - int(reference.min()) >= 64
            and float(reference.std()) > 5.0
        ),
        "imageDiffersReference": not np.array_equal(image, reference),
    }
    _validate_edit_policy_record(values, checks)


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


def _validate_files(
    output: Path, revision: str, edit_revision: str
) -> list[dict[str, object]]:
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
        elif name == EDIT_FILE:
            if metadata.get("device") != "cpu" or metadata.get("edit_revision") != edit_revision:
                raise InvalidOracle(
                    f"{name} metadata mismatch: device={metadata.get('device')!r}, "
                    f"edit_revision={metadata.get('edit_revision')!r}, "
                    f"expected cpu/{edit_revision}"
                )
            _validate_schema(name, EDIT_SCHEMA, keys, shapes, dtypes)
            _validate_primary_edit_policy(path)
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
    edit_revision: str,
    records: list[dict[str, object]],
    seconds: float,
    reference_environment: dict[str, str] | None = None,
) -> None:
    document = {
        "schema": 1,
        "reference": "microsoft/Mage frozen vendored reference",
        "snapshotRevision": revision,
        "editSnapshotRevision": edit_revision,
        "device": "cpu",
        "vaeGeometries": list(GEOMETRIES),
        "generationSeconds": round(seconds, 3),
        "referenceEnvironment": reference_environment or {},
        "files": records,
    }
    (output / MANIFEST).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    edit_document = {
        **document,
        "files": [record for record in records if record["name"] == EDIT_FILE],
    }
    (output / EDIT_MANIFEST).write_text(
        json.dumps(edit_document, indent=2) + "\n", encoding="utf-8"
    )


def verify(output: Path, snapshot: Path, edit_snapshot: Path) -> None:
    revision = _revision(snapshot)
    edit_revision = _revision(edit_snapshot)
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
        or manifest.get("editSnapshotRevision") != edit_revision
        or manifest.get("vaeGeometries") != list(GEOMETRIES)
    ):
        raise InvalidOracle(f"Mage oracle manifest does not match cpu/{revision}/{GEOMETRIES}")
    records = _validate_files(output, revision, edit_revision)
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


def verify_edit_artifact(output: Path, edit_snapshot: Path) -> None:
    edit_revision = _revision(edit_snapshot)
    try:
        document = json.loads((output / EDIT_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidOracle(f"invalid Mage edit artifact manifest: {error}") from error
    records = document.get("files", [])
    if (
        document.get("device") != "cpu"
        or document.get("editSnapshotRevision") != edit_revision
        or len(records) != 1
        or records[0].get("name") != EDIT_FILE
    ):
        raise InvalidOracle("Mage edit artifact manifest is incomplete or stale")
    path = output / EDIT_FILE
    metadata, keys, shapes, dtypes = _inspect(path)
    if metadata.get("device") != "cpu" or metadata.get("edit_revision") != edit_revision:
        raise InvalidOracle("Mage edit artifact metadata does not match its CPU snapshot")
    _validate_schema(EDIT_FILE, EDIT_SCHEMA, keys, shapes, dtypes)
    _validate_primary_edit_policy(path)
    if records[0].get("sha256") != _sha256(path) or records[0].get("bytes") != path.stat().st_size:
        raise InvalidOracle("Mage edit artifact hash/size differs from its manifest")


def provision(output: Path, snapshot: Path, edit_snapshot: Path) -> None:
    revision = _revision(snapshot)
    edit_revision = _revision(edit_snapshot)
    reference_environment = _validate_reference_environment()
    output.mkdir(parents=True, exist_ok=True)
    for name in (*EXPECTED_FILES, MANIFEST, EDIT_MANIFEST):
        (output / name).unlink(missing_ok=True)
    env = {
        **os.environ,
        "MAGE_DEVICE": "cpu",
        "MAGE_SNAPSHOT": str(snapshot),
        "MAGE_EDIT_SNAPSHOT": str(edit_snapshot),
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
        [sys.executable, str(MLX / "tools/dump_mage_flow_golden.py"), "--stage", "edit"],
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
    records = _validate_files(output, revision, edit_revision)
    _write_manifest(
        output, revision, edit_revision, records, seconds, reference_environment
    )
    verify_edit_artifact(output, edit_snapshot)
    verify(output, snapshot, edit_snapshot)
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
        wrong_edit_shape = {
            key: shape for key, (_, shape) in EDIT_SCHEMA.items()
        }
        wrong_edit_shape["img_shapes"] = [2, 3]
        edit_dtypes = {key: dtype for key, (dtype, _) in EDIT_SCHEMA.items()}

        missing = root / "missing"
        missing.mkdir()
        corrupt = root / "corrupt.safetensors"
        corrupt.write_bytes(b"not safetensors")
        stale = root / "stale"
        stale.mkdir()
        _write_manifest(stale, "b" * 40, "c" * 40, [], 0)
        corrupt_edit = root / "corrupt-edit"
        corrupt_edit.mkdir()
        (corrupt_edit / EDIT_MANIFEST).write_text("{", encoding="utf-8")

        cases = (
            ("Python patch mismatch", lambda: _validate_python_version((3, 12, 12))),
            ("absent", lambda: _validate_files(missing, revision, revision)),
            ("corrupt", lambda: _inspect(corrupt)),
            ("revision mismatch", lambda: verify(stale, snapshot, snapshot)),
            (
                "stale standalone edit manifest",
                lambda: verify_edit_artifact(stale, snapshot),
            ),
            (
                "corrupt standalone edit manifest",
                lambda: verify_edit_artifact(corrupt_edit, snapshot),
            ),
            (
                "wrong dtype",
                lambda: _validate_schema("synthetic", schema, set(schema), shapes, wrong_dtype),
            ),
            (
                "wrong shape",
                lambda: _validate_schema("synthetic", schema, set(schema), wrong_shape, dtypes),
            ),
            (
                "wrong edit topology",
                lambda: _validate_schema(
                    EDIT_FILE,
                    EDIT_SCHEMA,
                    set(EDIT_SCHEMA),
                    wrong_edit_shape,
                    edit_dtypes,
                ),
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
    parser.add_argument("--edit-snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--verify-edit-artifact", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    if args.verify_edit_artifact:
        if not args.edit_snapshot or not args.output:
            parser.error("--verify-edit-artifact requires --edit-snapshot and --output")
        try:
            verify_edit_artifact(args.output.resolve(), args.edit_snapshot.resolve())
        except (InvalidOracle, OSError, ValueError, json.JSONDecodeError) as error:
            print(f"Mage edit artifact verification FAILED: {error}", file=sys.stderr)
            return 1
        print(f"verified standalone Mage edit artifact under {args.output.resolve()}")
        return 0
    if not args.snapshot or not args.edit_snapshot or not args.output:
        parser.error("--snapshot, --edit-snapshot, and --output are required")
    try:
        if args.verify_only:
            verify(
                args.output.resolve(),
                args.snapshot.resolve(),
                args.edit_snapshot.resolve(),
            )
        else:
            provision(
                args.output.resolve(),
                args.snapshot.resolve(),
                args.edit_snapshot.resolve(),
            )
    except InvalidOracle as error:
        print(f"Mage oracle provisioning FAILED: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
