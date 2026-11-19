#!/usr/bin/env python3
"""Bind the pinned inputs of the Windows CUDA Krea proofs, downloading a missing one into the
runner's Hugging Face hub cache first.

These bindings only select inputs; the source-owned readers inside the one manual invocation
record the bytes they consume, so a path is not a substitute for those receipts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ci.proof_asset_snapshots import Input, manifest_bindings, manifest_keys, materialize  # noqa: E402

# `asset_setup` (scripts/ci/model_inference_proofs.py) -> {environment variable: pinned input}.
KREA_PROOFS = {
    "krea-native-nvfp4-pose": {
        "KREA_IMPORTED_DIT": Input("krea-2-comfy-turbo-nvfp4-proof",
                                   "diffusion_models/krea2_turbo_nvfp4.safetensors"),
        "KREA_BASE_SNAPSHOT": Input("krea-2-turbo-mlx-bf16-pose-proof", "bf16"),
        "KREA_CONTROL_BRANCH": Input("krea2-pose-controlnet-beta-proof", "control_step5000.safetensors"),
    },
    "krea-training": {"KREA_RAW_DIR": Input("krea-2-raw-train-proof")},
}
MANIFEST_BINDINGS = manifest_bindings(KREA_PROOFS)
MANIFEST_KEYS = manifest_keys(KREA_PROOFS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", choices=sorted(KREA_PROOFS), required=True)
    args = parser.parse_args()
    github_env = os.environ.get("GITHUB_ENV", "")
    if not github_env:
        raise ValueError("GitHub Actions environment file is required")
    paths = materialize(KREA_PROOFS[args.setup])
    for name, path in paths.items():
        if not path.exists() or (path.is_file() and path.stat().st_size == 0):
            raise ValueError(f"missing pinned Krea input {name}: {path}")
        if "\n" in str(path):
            raise ValueError(f"invalid newline in {name}")
    with open(github_env, "a", encoding="utf-8") as out:
        for name, path in paths.items():
            out.write(f"{name}={path}\n")
    print(json.dumps({"setup": args.setup, "installed_paths": {k: str(v) for k, v in paths.items()}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
