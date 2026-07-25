#!/usr/bin/env python3
"""Fail-closed verifier and hash manifest for Mage's 1024px Candle acceptance oracles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from safetensors import safe_open


FILES = (
    "mage_flow_dit_golden.safetensors",
    "mage_flow_e2e_golden.safetensors",
)
MANIFEST = "mage_candle_oracles_manifest.json"
REVISION_MARKER = ".sceneworks-model-revision"
GEOMETRY = [1024, 1024, 20, 4]
PROMPT = "a calico kitten sitting on a wooden windowsill beside a blue ceramic mug"
META_KEYS = {"geometry", "seed", "cfg", "gs_key", "drop_idx", "static_shift"}
DIT_KEYS = {
    "dit_in.img",
    "dit_in.txt",
    "dit_in.timesteps",
    "dit_in.img_cu_seqlens",
    "dit_in.txt_cu_seqlens",
    "dit_out",
    "img_shapes",
    *META_KEYS,
}
E2E_KEYS = {
    "final_tokens",
    "final_latent",
    "image_u8",
    "traj_step0",
    "traj_step1",
    "img_shapes",
    "sigmas_20",
    "timesteps_20",
    "sigmas_4",
    "timesteps_4",
    "sigmas_30",
    "timesteps_30",
    *META_KEYS,
}
EXPECTED_IMG_SHAPES = [[1, 64, 64], [1, 64, 64]]


class InvalidOracle(RuntimeError):
    pass


def revision(path: Path) -> str:
    resolved = path.resolve()
    marker = resolved / REVISION_MARKER
    value = marker.read_text(encoding="utf-8").strip() if marker.is_file() else resolved.name
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise InvalidOracle(f"snapshot is not pinned to a 40-hex revision: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(path: Path) -> dict:
    if not path.is_file():
        raise InvalidOracle(f"required acceptance oracle is missing: {path}")
    try:
        with safe_open(path, framework="numpy") as handle:
            tensors = {}
            values = {}
            selected = {}
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                tensors[key] = {
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                }
                if (
                    key in META_KEYS
                    or key.startswith(("sigmas_", "timesteps_"))
                    or key
                    in {
                        "dit_in.timesteps",
                        "dit_in.img_cu_seqlens",
                        "dit_in.txt_cu_seqlens",
                        "img_shapes",
                    }
                ):
                    values[key] = tensor.tolist()
                if key in {
                    "dit_in.img",
                    "dit_out",
                    "final_tokens",
                    "image_u8",
                    "traj_step0",
                    "traj_step1",
                }:
                    selected[key] = tensor
            metadata = handle.metadata() or {}
    except Exception as error:
        raise InvalidOracle(f"cannot inspect {path.name}: {error}") from error
    checks = {
        "ditChangesInput": (
            not np.array_equal(selected["dit_in.img"], selected["dit_out"])
            if {"dit_in.img", "dit_out"} <= set(selected)
            else None
        ),
        "trajectoryChanges": (
            not np.array_equal(selected["traj_step0"], selected["traj_step1"])
            if {"traj_step0", "traj_step1"} <= set(selected)
            else None
        ),
        "finalChangesInitial": None,
        "imageDiscriminating": None,
    }
    if {"final_tokens", "traj_step0"} <= set(selected):
        final_tokens = selected["final_tokens"]
        checks["finalChangesInitial"] = not np.array_equal(
            final_tokens, selected["traj_step0"][:, : final_tokens.shape[1], :]
        )
    if "image_u8" in selected:
        image = selected["image_u8"]
        checks["imageDiscriminating"] = (
            int(image.max()) - int(image.min()) >= 64 and float(image.std()) > 5.0
        )
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "metadata": metadata,
        "tensors": tensors,
        "values": values,
        "mutationChecks": checks,
    }


def expected_schedule(steps: int) -> tuple[list[float], list[float]]:
    base = [
        1.0 - index * (1.0 - 1.0 / steps) / (steps - 1)
        for index in range(steps)
    ]
    shifted = [6.0 * sigma / (1.0 + 5.0 * sigma) for sigma in base]
    return shifted + [0.0], [sigma * 1000.0 for sigma in shifted]


def require_values(record: dict, key: str, expected: object, tolerance: float = 0.0) -> None:
    actual = record["values"].get(key)
    if tolerance:
        if (
            not isinstance(actual, list)
            or not isinstance(expected, list)
            or len(actual) != len(expected)
            or any(abs(float(left) - float(right)) > tolerance for left, right in zip(actual, expected))
        ):
            raise InvalidOracle(f"{record['file']}:{key} ordered values are stale")
    elif actual != expected:
        raise InvalidOracle(f"{record['file']}:{key} value is stale")


def require_tensor(
    record: dict,
    key: str,
    dtype: str,
    shape: list[int] | None = None,
) -> list[int]:
    tensor = record["tensors"][key]
    if tensor["dtype"] != dtype:
        raise InvalidOracle(f"{record['file']}:{key} dtype {tensor['dtype']} != {dtype}")
    actual = tensor["shape"]
    if shape is not None and actual != shape:
        raise InvalidOracle(f"{record['file']}:{key} shape {actual} != {shape}")
    return actual


def validate_common(record: dict, expected_revision: str, keys: set[str]) -> None:
    if set(record["tensors"]) != keys:
        raise InvalidOracle(f"{record['file']} tensor population is incomplete or stale")
    metadata = record["metadata"]
    if (
        metadata.get("device") != "cpu"
        or metadata.get("gen_revision") != expected_revision
        or metadata.get("prompt") != PROMPT
    ):
        raise InvalidOracle(f"{record['file']} metadata does not pin cpu/{expected_revision}")
    for key, dtype, shape, value in (
        ("geometry", "int32", [4], GEOMETRY),
        ("seed", "int64", [1], [42]),
        ("cfg", "float32", [1], [5.0]),
        ("gs_key", "int64", [1], [20260720]),
        ("drop_idx", "int32", [2], [34, 64]),
        ("static_shift", "float32", [1], [6.0]),
    ):
        require_tensor(record, key, dtype, shape)
        require_values(record, key, value)


def validate_dit(record: dict, expected_revision: str) -> None:
    validate_common(record, expected_revision, DIT_KEYS)
    require_tensor(record, "dit_in.img", "float32", [1, 8192, 128])
    require_tensor(record, "dit_out", "float32", [1, 8192, 128])
    require_tensor(record, "dit_in.txt", "float32", [1, 26, 2560])
    require_tensor(record, "dit_in.timesteps", "float32", [2])
    require_tensor(record, "dit_in.img_cu_seqlens", "int64", [3])
    require_tensor(record, "dit_in.txt_cu_seqlens", "int64", [3])
    require_tensor(record, "img_shapes", "int32", [2, 3])
    require_values(record, "dit_in.timesteps", [1.0, 1.0], 1.0e-6)
    require_values(record, "dit_in.img_cu_seqlens", [0, 4096, 8192])
    require_values(record, "dit_in.txt_cu_seqlens", [0, 20, 26])
    require_values(record, "img_shapes", EXPECTED_IMG_SHAPES)
    if record["mutationChecks"].get("ditChangesInput") is not True:
        raise InvalidOracle("DiT output equals its input; mutation gate is non-load-bearing")


def validate_e2e(record: dict, expected_revision: str) -> None:
    validate_common(record, expected_revision, E2E_KEYS)
    require_tensor(record, "final_tokens", "float32", [1, 4096, 128])
    require_tensor(record, "final_latent", "float32", [1, 128, 64, 64])
    require_tensor(record, "traj_step0", "float32", [1, 8192, 128])
    require_tensor(record, "traj_step1", "float32", [1, 8192, 128])
    require_tensor(record, "image_u8", "uint8", [1024, 1024, 3])
    require_tensor(record, "img_shapes", "int32", [2, 3])
    require_values(record, "img_shapes", EXPECTED_IMG_SHAPES)
    if record["mutationChecks"].get("trajectoryChanges") is not True:
        raise InvalidOracle("E2E step0 equals step1; trajectory gate is non-load-bearing")
    if record["mutationChecks"].get("finalChangesInitial") is not True:
        raise InvalidOracle("E2E final tokens equal their initial trajectory slice")
    if record["mutationChecks"].get("imageDiscriminating") is not True:
        raise InvalidOracle("E2E image is constant or non-discriminating")
    for steps in (20, 4, 30):
        require_tensor(record, f"sigmas_{steps}", "float32", [steps + 1])
        require_tensor(record, f"timesteps_{steps}", "float32", [steps])
        sigmas, timesteps = expected_schedule(steps)
        require_values(record, f"sigmas_{steps}", sigmas, 1.0e-4)
        require_values(record, f"timesteps_{steps}", timesteps, 1.0e-3)


def records(output: Path, expected_revision: str) -> list[dict]:
    found = [inspect(output / filename) for filename in FILES]
    validate_dit(found[0], expected_revision)
    validate_e2e(found[1], expected_revision)
    return found


def document(output: Path, snapshot: Path) -> dict:
    pinned = revision(snapshot)
    return {
        "schema": 1,
        "reference": "microsoft/Mage frozen vendored CPU reference",
        "snapshotRevision": pinned,
        "geometry": GEOMETRY,
        "files": records(output, pinned),
    }


def verify(output: Path, snapshot: Path, write_manifest: bool) -> None:
    actual = document(output, snapshot)
    manifest_path = output / MANIFEST
    if write_manifest:
        manifest_path.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
    try:
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidOracle(f"invalid Candle acceptance manifest: {error}") from error
    if expected != actual:
        raise InvalidOracle("Candle acceptance manifest revision/schema/value/hash is stale")
    print(f"verified {len(FILES)} Mage Candle acceptance oracles under {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    try:
        verify(args.output.resolve(), args.snapshot.resolve(), args.write_manifest)
    except InvalidOracle as error:
        print(f"Mage Candle oracle verification FAILED: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
