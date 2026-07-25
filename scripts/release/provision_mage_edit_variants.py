#!/usr/bin/env python3
"""Generate and verify frozen-Torch Mage Edit/BASE/Turbo parity bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
MLX = ROOT / "crates/media/mlx-gen"
CASES = (
    ("edit", "mage_flow_edit_golden.safetensors", 30, 5.0),
    ("edit-base", "mage_flow_edit_base_golden.safetensors", 30, 5.0),
    ("edit-turbo", "mage_flow_edit_turbo_golden.safetensors", 4, 1.0),
)
REVISION_MARKER = ".sceneworks-model-revision"
PROMPT = "a calico kitten sitting on a wooden windowsill beside a blue ceramic mug"
EDIT_INSTRUCTION = "Replace the background with a field of sunflowers"
def expected_img_shapes(cfg: float) -> list[list[int]]:
    return [[1, 16, 16]] * (4 if cfg > 1.0 else 2)


def edit_schema(steps: int, cfg: float) -> dict[str, tuple[str, list[int]]]:
    sequence_tokens = 1024 if cfg > 1.0 else 512
    return {
        "cfg": ("float32", [1]),
        "drop_idx": ("int32", [2]),
        "final_latent": ("float32", [1, 128, 16, 16]),
        "final_tokens": ("float32", [1, 256, 128]),
        "geometry": ("int32", [4]),
        "gs_key": ("int64", [1]),
        "image_u8": ("uint8", [256, 256, 3]),
        "img_shapes": ("int32", [len(expected_img_shapes(cfg)), 3]),
        "ref_u8": ("uint8", [256, 256, 3]),
        "seed": ("int64", [1]),
        "seq_step0": ("float32", [1, sequence_tokens, 128]),
        "seq_step1": ("float32", [1, sequence_tokens, 128]),
        f"sigmas_{steps}": ("float32", [steps + 1]),
        "static_shift": ("float32", [1]),
        "target_tokens": ("int32", [1]),
        f"timesteps_{steps}": ("float32", [steps]),
    }


def expected_schedule(steps: int) -> tuple[list[float], list[float]]:
    base = [
        1.0 - index * (1.0 - 1.0 / steps) / (steps - 1)
        for index in range(steps)
    ]
    shifted = [6.0 * sigma / (1.0 + 5.0 * sigma) for sigma in base]
    return shifted + [0.0], [sigma * 1000.0 for sigma in shifted]


def revision(path: Path) -> str:
    resolved = path.resolve()
    marker = resolved / REVISION_MARKER
    value = (
        marker.read_text(encoding="utf-8").strip()
        if marker.is_file()
        else resolved.name
    )
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise RuntimeError(
            f"snapshot must resolve to a 40-hex revision directory or carry {REVISION_MARKER}: {path}"
        )
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(path: Path) -> dict:
    try:
        with safe_open(path, framework="numpy") as handle:
            metadata = handle.metadata() or {}
            tensors = {}
            values = {}
            selected = {}
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                tensors[key] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
                if key in {
                    "cfg",
                    "drop_idx",
                    "geometry",
                    "gs_key",
                    "img_shapes",
                    "seed",
                    "static_shift",
                    "target_tokens",
                } or key.startswith(("sigmas_", "timesteps_")):
                    values[key] = tensor.tolist()
                if key in {
                    "final_tokens",
                    "image_u8",
                    "ref_u8",
                    "seq_step0",
                    "seq_step1",
                }:
                    selected[key] = tensor
    except Exception as error:
        raise RuntimeError(f"{path.name} is not a readable safetensors bundle: {error}") from error
    image = selected.get("image_u8")
    reference = selected.get("ref_u8")
    step0 = selected.get("seq_step0")
    step1 = selected.get("seq_step1")
    final = selected.get("final_tokens")
    checks = {
        "trajectoryChanges": (
            not np.array_equal(step0, step1) if step0 is not None and step1 is not None else None
        ),
        "targetDiffersReference": (
            not np.array_equal(step0[:, :256, :], step0[:, 256:512, :])
            if step0 is not None
            else None
        ),
        "finalChangesInitial": (
            not np.array_equal(final, step0[:, :256, :])
            if final is not None and step0 is not None
            else None
        ),
        "imageDiscriminating": (
            int(image.max()) - int(image.min()) >= 64 and float(image.std()) > 5.0
            if image is not None
            else None
        ),
        "referenceDiscriminating": (
            int(reference.max()) - int(reference.min()) >= 64 and float(reference.std()) > 5.0
            if reference is not None
            else None
        ),
        "imageDiffersReference": (
            not np.array_equal(image, reference)
            if image is not None and reference is not None
            else None
        ),
    }
    return {
        "metadata": metadata,
        "tensors": tensors,
        "values": values,
        "mutationChecks": checks,
    }


def require_values(record: dict, key: str, expected: object, tolerance: float = 0.0) -> None:
    actual = record["values"].get(key)
    if tolerance:
        if (
            not isinstance(actual, list)
            or not isinstance(expected, list)
            or len(actual) != len(expected)
            or any(abs(float(left) - float(right)) > tolerance for left, right in zip(actual, expected))
        ):
            raise RuntimeError(f"{key} ordered values are stale")
    elif actual != expected:
        raise RuntimeError(f"{key} value is stale")


def validate_record(
    record: dict, expected_revision: str, expected_steps: int, expected_cfg: float
) -> None:
    schema = edit_schema(expected_steps, expected_cfg)
    if set(record["tensors"]) != set(schema):
        raise RuntimeError("tensor population mismatch")
    for key, (dtype, shape) in schema.items():
        actual = record["tensors"][key]
        if actual != {"dtype": dtype, "shape": shape}:
            raise RuntimeError(f"{key} dtype/shape is stale")
    metadata = record["metadata"]
    if (
        metadata.get("device") != "cpu"
        or metadata.get("edit_revision") != expected_revision
        or metadata.get("prompt") != PROMPT
        or metadata.get("edit_instruction") != EDIT_INSTRUCTION
    ):
        raise RuntimeError(f"metadata does not pin cpu/{expected_revision}")
    for key, expected in (
        ("geometry", [256, 256, 4, expected_steps]),
        ("seed", [42]),
        ("cfg", [expected_cfg]),
        ("gs_key", [20260720]),
        ("drop_idx", [34, 64]),
        ("static_shift", [6.0]),
        ("target_tokens", [256]),
        ("img_shapes", expected_img_shapes(expected_cfg)),
    ):
        require_values(record, key, expected)
    sigmas, timesteps = expected_schedule(expected_steps)
    require_values(record, f"sigmas_{expected_steps}", sigmas, 1.0e-4)
    require_values(record, f"timesteps_{expected_steps}", timesteps, 1.0e-3)
    failed = [name for name, passed in record["mutationChecks"].items() if passed is not True]
    if failed:
        raise RuntimeError(f"load-bearing mutation/value checks failed: {failed}")


def validate(path: Path, expected_revision: str, expected_steps: int, expected_cfg: float) -> None:
    validate_record(inspect(path), expected_revision, expected_steps, expected_cfg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit", required=True, type=Path)
    parser.add_argument("--edit-base", required=True, type=Path)
    parser.add_argument("--edit-turbo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="validate existing files and their exact hash manifest without generating",
    )
    args = parser.parse_args()
    snapshots = {
        "edit": args.edit,
        "edit-base": args.edit_base,
        "edit-turbo": args.edit_turbo,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for label, filename, steps, cfg in CASES:
        snapshot = snapshots[label]
        pinned = revision(snapshot)
        destination = args.output / filename
        # provision_mage_oracles.py owns the primary Edit file and its two immutable manifests.
        # Reuse it here so adding Base/Turbo cannot overwrite a previously verified artifact.
        if label != "edit" and not args.verify_only:
            with tempfile.TemporaryDirectory(prefix=f"mage-{label}-") as temporary:
                temp = Path(temporary)
                env = {
                    **os.environ,
                    "MAGE_DEVICE": "cpu",
                    "MAGE_EDIT_SNAPSHOT": str(snapshot),
                    "MAGE_GOLDEN_DIR": str(temp),
                    "MAGE_H": "256",
                    "MAGE_W": "256",
                    "MAGE_STEPS": "4",
                    "MAGE_EDIT_STEPS": str(steps),
                    "MAGE_CFG": str(cfg),
                    "PYTHONPATH": str(MLX / "_vendor"),
                }
                subprocess.run(
                    [
                        sys.executable,
                        str(MLX / "tools/dump_mage_flow_golden.py"),
                        "--stage",
                        "edit",
                    ],
                    cwd=MLX,
                    env=env,
                    check=True,
                )
                generated = temp / "mage_flow_edit_golden.safetensors"
                shutil.copy2(generated, destination)
        elif not destination.is_file():
            raise RuntimeError(
                f"{destination.name} must already exist before variant verification"
            )
        validate(destination, pinned, steps, cfg)
        records.append(
            {
                "variant": label,
                "snapshotRevision": pinned,
                "file": filename,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
                "cfg": cfg,
                "steps": steps,
            }
        )
    manifest = {
        "schema": 1,
        "reference": "microsoft/Mage frozen vendored reference",
        "device": "cpu",
        "files": records,
    }
    manifest_path = args.output / "mage_edit_variants_manifest.json"
    if args.verify_only:
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid Mage edit variant manifest: {error}") from error
        if existing != manifest:
            raise RuntimeError(
                "Mage edit variant manifest revisions/geometry/cfg/hash population is stale"
            )
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    print(f"verified {len(records)} Mage edit variant oracles under {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
