#!/usr/bin/env python3
"""Bind already provisioned, pinned Iris snapshots for an exact manual proof."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pinned_hub_source import verify_pinned_hub_files


IRIS_REVISION = "7445443349bc9abe3c96f01ff793e2098ca012b3"
QWEN_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
IRIS_FILES = ("config.yaml", "model.safetensors")
QWEN_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json",
              "model.safetensors.index.json", "model-00001-of-00002.safetensors",
              "model-00002-of-00002.safetensors")


def validate_receipt(value: dict, backend: str, expected_paths: dict[str, str] | None = None) -> dict:
    """Check the exact upstream claims carried into an Iris case evidence record."""
    if (not isinstance(value, dict) or set(value) != {"schema_version", "backend", "assets"}
            or value["schema_version"] != 1 or value["backend"] != backend
            or not isinstance(value["assets"], list) or len(value["assets"]) != 2):
        raise ValueError("Iris asset preflight receipt is missing or malformed")
    expected = (("IRIS_WEIGHTS_DIR", "speridlabs/iris-3b", IRIS_REVISION, IRIS_FILES),
                ("IRIS_TEXT_ENCODER_DIR", "Qwen/Qwen3-VL-4B-Instruct", QWEN_REVISION, QWEN_FILES))
    for item, (name, repository, revision, files) in zip(value["assets"], expected):
        if not isinstance(item, dict) or set(item) != {"environment", "path", "upstream"}:
            raise ValueError("Iris asset preflight entry is malformed")
        path = item["path"]
        suffix = f"models--{repository.replace('/', '--')}/snapshots/{revision}"
        if (item["environment"] != name or not isinstance(path, str)
                or not path.replace("\\", "/").endswith("/" + suffix)
                or expected_paths is not None and expected_paths.get(name) != path):
            raise ValueError(f"Iris asset preflight path/revision mismatch: {name}")
        upstream = item["upstream"]
        url = f"https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true"
        if (not isinstance(upstream, dict)
                or set(upstream) != {"status", "repository", "revision", "readback_url", "manifest_sha256", "files"}
                or upstream["status"] != "hub_content_verified" or upstream["repository"] != repository
                or upstream["revision"] != revision or upstream["readback_url"] != url
                or not isinstance(upstream["manifest_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", upstream["manifest_sha256"])
                or not isinstance(upstream["files"], list)
                or tuple(file.get("path") for file in upstream["files"] if isinstance(file, dict)) != files):
            raise ValueError(f"Iris asset preflight upstream claim mismatch: {name}")
        for file in upstream["files"]:
            if not isinstance(file, dict) or set(file) != {"path", "bytes", "kind", "digest"}:
                raise ValueError("Iris asset preflight file receipt is malformed")
            kind, digest = file["kind"], file["digest"]
            if (type(file["bytes"]) is not int or file["bytes"] < 0
                    or (kind == "lfs_sha256" and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)))
                    or (kind == "git_blob_sha1" and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{40}", digest)))
                    or kind not in ("lfs_sha256", "git_blob_sha1")):
                raise ValueError("Iris asset preflight file digest is malformed")
    return value


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
    attestations = []
    for name, model, path in paths:
        subprocess.run([sys.executable, "-I", str(verifier), "--model", model,
                        "--snapshot", str(path)], check=True)
        repository = "speridlabs/iris-3b" if name == "IRIS_WEIGHTS_DIR" else "Qwen/Qwen3-VL-4B-Instruct"
        revision = IRIS_REVISION if name == "IRIS_WEIGHTS_DIR" else QWEN_REVISION
        files = IRIS_FILES if name == "IRIS_WEIGHTS_DIR" else QWEN_FILES
        attestation = verify_pinned_hub_files(repository, revision, path, files)
        attestations.append({"environment": name, "path": str(path), "upstream": attestation})
        print(f"{name} content authenticated against {repository}@{revision}", flush=True)
    temporary = os.environ.get("RUNNER_TEMP", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if not temporary or not run_id.isdecimal() or not run_attempt.isdecimal():
        raise ValueError("owned Actions temp and numeric run identity are required")
    receipt = Path(temporary) / f"iris-quant-assets-{run_id}-{run_attempt}.json"
    value = validate_receipt({"schema_version": 1, "backend": args.backend,
                              "assets": attestations}, args.backend,
                             {name: str(path) for name, _, path in paths})
    with receipt.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
    with Path(github_env).open("a", encoding="utf-8") as handle:
        for name, _, path in paths:
            if "\n" in str(path):
                raise ValueError(f"invalid newline in {name}")
            handle.write(f"{name}={path}\n")
        handle.write(f"MODEL_PROOF_ASSET_RECEIPT={receipt}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
