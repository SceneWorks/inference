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


def validate(path: Path, expected_revision: str, expected_steps: int, expected_cfg: float) -> None:
    with safe_open(path, framework="numpy") as handle:
        metadata = handle.metadata() or {}
        required = {
            "cfg",
            "final_tokens",
            "geometry",
            "image_u8",
            "img_shapes",
            "ref_u8",
            "seed",
            "seq_step0",
            "seq_step1",
            "target_tokens",
        }
        missing = required - set(handle.keys())
        if missing:
            raise RuntimeError(f"{path.name} is missing {sorted(missing)}")
        if metadata.get("device") != "cpu" or metadata.get("edit_revision") != expected_revision:
            raise RuntimeError(f"{path.name} metadata does not pin cpu/{expected_revision}")
        geometry = handle.get_tensor("geometry").astype(np.int64)
        cfg = float(handle.get_tensor("cfg")[0])
        image = handle.get_tensor("image_u8")
        step0 = handle.get_tensor("seq_step0")
        step1 = handle.get_tensor("seq_step1")
    expected_geometry = [256, 256, 4, expected_steps]
    if geometry.tolist() != expected_geometry:
        raise RuntimeError(
            f"{path.name} geometry is {geometry.tolist()}, expected {expected_geometry}"
        )
    if cfg != expected_cfg:
        raise RuntimeError(f"{path.name} cfg is {cfg}, expected {expected_cfg}")
    if int(image.max()) - int(image.min()) < 64 or float(image.std()) <= 5.0:
        raise RuntimeError(f"{path.name} image is non-discriminating")
    if np.array_equal(step0, step1):
        raise RuntimeError(f"{path.name} mutation guard failed: step0 equals step1")


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
