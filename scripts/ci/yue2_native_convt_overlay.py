#!/usr/bin/env python3
"""Stage exact M3 and pinned Candle archives with declared diagnostic-only patches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from yue2_decoder_trace_overlay import (ENGINE_SHA, LOCK_SHA, VAE_PATH,
                                         stage as stage_vae, tree_digest)
from yue2_precision_proof import require, sha256, write_json

CANDLE_SHA = "1e6aa85e867eb007cba1b8bae517a10d1aaf0c0d"
CANDLE_TREE = "7ff76c6176ee6508f027bd2b177087a9da5edddf"
CANDLE_BACKEND = Path("candle-core/src/cuda_backend/mod.rs")
CANDLE_BACKEND_SHA = "bc2bd5fe3702bfccb49d64af07dc4197013d6617c720643cd2a14b718f4134e2"
CANDLE_ROOT_MANIFEST_SHA = "dc8610cd8fbdbd87206a31b7ad416e1bc35d597ebcb83a03e86b2fe25ab326f7"


def exact_checkout(source: Path, sha: str, tree: str | None = None) -> None:
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                   text=True, encoding="utf-8").strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain",
                                     "--untracked-files=normal"], text=True,
                                    encoding="utf-8")
    require(head == sha and not dirty, "diagnostic source checkout changed")
    if tree:
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD^{tree}"],
                                         text=True, encoding="utf-8").strip()
        require(actual == tree, "pinned Candle tree changed")


def apply_one_patch(root: Path, patch: Path, path: Path) -> None:
    content = patch.read_text(encoding="utf-8")
    name = path.as_posix()
    require(content.startswith(f"--- a/{name}\n+++ b/{name}\n") and
            content.count("\n--- ") == 0 and content.count("\n+++ ") == 1,
            "diagnostic patch touches an unexpected source")
    subprocess.run(["git", "apply", "--check", "--unidiff-zero", str(patch)],
                   cwd=root, check=True)
    subprocess.run(["git", "apply", "--unidiff-zero", str(patch)], cwd=root, check=True)


def extract_exact(source: Path, sha: str, archive: Path, destination: Path) -> None:
    require(not archive.exists() and not destination.exists(), "source staging must be fresh")
    subprocess.run(["git", "-C", str(source), "archive", "--format=tar.gz", "--output",
                    str(archive), sha], check=True)
    destination.mkdir(parents=True)
    with tarfile.open(archive) as package:
        for member in package:
            require(member.isfile() or member.isdir(), "source archive contains a link or special member")
            require(not Path(member.name).is_absolute() and ".." not in Path(member.name).parts,
                    "source archive path escapes destination")
        package.extractall(destination, filter="data")


def stage(engine: Path, candle: Path, root: Path, trace_patch: Path,
          weight_patch: Path, backend_patch: Path, kernel_path_patch: Path,
          control_sha: str) -> Path:
    require(not root.exists(), "native column source root must be fresh")
    root.mkdir(parents=True)
    engine = engine.resolve(strict=True)
    candle = candle.resolve(strict=True)
    exact_checkout(engine, ENGINE_SHA)
    exact_checkout(candle, CANDLE_SHA, CANDLE_TREE)
    trace_patch = trace_patch.resolve(strict=True)
    weight_patch = weight_patch.resolve(strict=True)
    backend_patch = backend_patch.resolve(strict=True)
    kernel_path_patch = kernel_path_patch.resolve(strict=True)
    engine_overlay = root / "engine-overlay"
    stage_vae(engine, engine_overlay, trace_patch, ENGINE_SHA, control_sha)
    existing = json.loads((root / "overlay-provenance.json").read_text(encoding="utf-8"))
    prior_vae = existing["derivative_vae_sha256"]
    prior_tree = existing["derivative_tree_sha256"]
    apply_one_patch(engine_overlay, weight_patch, VAE_PATH)
    shutil.copy2(weight_patch, root / "native-convt-vae-weight.patch")
    require(sha256(engine_overlay / "Cargo.lock") == LOCK_SHA,
            "native weight accessor changed the M3 lock")
    candle_overlay = root / "candle-overlay"
    candle_archive = root / "pinned-candle-tracked-source.tar.gz"
    extract_exact(candle, CANDLE_SHA, candle_archive, candle_overlay)
    original_backend = subprocess.check_output(["git", "-C", str(candle), "show",
                                                f"{CANDLE_SHA}:{CANDLE_BACKEND.as_posix()}"])
    require(sha256(candle_overlay / CANDLE_BACKEND) ==
            hashlib.sha256(original_backend).hexdigest() == CANDLE_BACKEND_SHA and
            sha256(candle_overlay / "Cargo.toml") == CANDLE_ROOT_MANIFEST_SHA,
            "pinned Candle backend blob changed")
    candle_base_tree = tree_digest(candle_overlay)
    apply_one_patch(candle_overlay, backend_patch, CANDLE_BACKEND)
    apply_one_patch(candle_overlay, kernel_path_patch, Path("Cargo.toml"))
    shutil.copy2(backend_patch, root / "native-convt-candle-core.patch")
    shutil.copy2(kernel_path_patch, root / "native-convt-candle-kernel-path.patch")
    provenance = {**existing, "schema": "yue2-native-convt-source-v1",
                  "trace_derivative_vae_sha256": prior_vae,
                  "trace_derivative_tree_sha256": prior_tree,
                  "native_vae_patch_sha256": sha256(weight_patch),
                  "derivative_vae_sha256": sha256(engine_overlay / VAE_PATH),
                  "derivative_tree_sha256": tree_digest(engine_overlay),
                  "candle_git_sha": CANDLE_SHA, "candle_git_tree": CANDLE_TREE,
                  "candle_archive_sha256": sha256(candle_archive),
                  "candle_base_tree_sha256": candle_base_tree,
                  "candle_backend_base_sha256": CANDLE_BACKEND_SHA,
                  "candle_backend_patch_sha256": sha256(backend_patch),
                  "candle_root_manifest_base_sha256": CANDLE_ROOT_MANIFEST_SHA,
                  "candle_kernel_path_patch_sha256": sha256(kernel_path_patch),
                  "candle_root_manifest_derivative_sha256": sha256(candle_overlay / "Cargo.toml"),
                  "candle_backend_derivative_sha256": sha256(candle_overlay / CANDLE_BACKEND),
                  "candle_derivative_tree_sha256": tree_digest(candle_overlay),
                  "candle_core_source_exception": "path-patched pinned Candle core only"}
    write_json(root / "native-provenance.json", provenance)
    return root / "native-provenance.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "candle", "root", "trace-patch", "weight-patch", "backend-patch",
                 "kernel-path-patch"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--control-sha", required=True)
    args = parser.parse_args()
    print(json.dumps({"provenance": str(stage(args.engine, args.candle, args.root,
                                              args.trace_patch, args.weight_patch,
                                              args.backend_patch, args.kernel_path_patch,
                                              args.control_sha))}))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError, tarfile.TarError) as error:
        raise SystemExit(f"yue2-native-convt-overlay: {error}") from error
