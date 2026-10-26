#!/usr/bin/env python3
"""Bind already provisioned, pinned Iris snapshots for an exact manual proof."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


IRIS_REVISION = "7445443349bc9abe3c96f01ff793e2098ca012b3"
QWEN_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def snapshot_paths(backend: str, env: dict[str, str]) -> tuple[tuple[str, str, Path], ...]:
    if backend == "mlx":
        if env.get("RUNNER_OS") != "macOS" or env.get("RUNNER_NAME") != "nax-macos-2":
            raise ValueError("Iris MLX quant proofs require the provisioned secondary Mac")
        # The existing Iris curricula/memory lanes use this runner-local internal hub.
        root = Path.home() / "sceneworks-rw-weights" / "hub"
        models = (("IRIS_WEIGHTS_DIR", "iris-3b-generation-mlx", "speridlabs--iris-3b", IRIS_REVISION),
                  ("IRIS_TEXT_ENCODER_DIR", "qwen3-vl-4b-instruct-mlx", "Qwen--Qwen3-VL-4B-Instruct", QWEN_REVISION))
    elif backend == "candle-cuda":
        if env.get("RUNNER_OS") != "Windows":
            raise ValueError("Iris Candle CUDA quant proofs require a Windows runner")
        value = env.get("CANDLE_GEN_MODELS_ROOT", "")
        if not value or not Path(value).is_absolute():
            raise ValueError("CANDLE_GEN_MODELS_ROOT must be an absolute, operator-provided Windows cache path")
        root = Path(value)
        models = (("IRIS_WEIGHTS_DIR", "iris-3b-cuda", "speridlabs--iris-3b", IRIS_REVISION),
                  ("IRIS_TEXT_ENCODER_DIR", "qwen3-vl-4b-instruct-cuda", "Qwen--Qwen3-VL-4B-Instruct", QWEN_REVISION))
    else:
        raise ValueError(f"unsupported Iris quant proof backend: {backend}")
    if backend == "mlx" and root.parent.is_symlink():
        raise ValueError(f"secondary Mac task-owned weights root must not be a symlink: {root.parent}")
    if not root.is_dir():
        raise ValueError(f"runner-local model hub is missing: {root}")
    return tuple((name, model, root / f"models--{repository}" / "snapshots" / revision)
                 for name, model, repository, revision in models)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("mlx", "candle-cuda"), required=True)
    args = parser.parse_args()
    github_env = os.environ.get("GITHUB_ENV", "")
    if not github_env:
        raise ValueError("GitHub Actions environment file is required")
    paths = snapshot_paths(args.backend, os.environ)
    verifier = Path(__file__).resolve().parents[1] / "release" / "verify_model_snapshot.py"
    for name, model, path in paths:
        subprocess.run([sys.executable, "-I", str(verifier), "--model", model,
                        "--snapshot", str(path)], check=True)
        print(f"{name} verified at {path}", flush=True)
    with Path(github_env).open("a", encoding="utf-8") as handle:
        for name, _, path in paths:
            if "\n" in str(path):
                raise ValueError(f"invalid newline in {name}")
            handle.write(f"{name}={path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
