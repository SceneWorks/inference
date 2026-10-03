#!/usr/bin/env python3
"""Stage exact M4 and pinned Candle for a diagnostic-only BF16 stage-2 experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tomllib

from yue2_native_convt_overlay import (CANDLE_SHA, CANDLE_TREE, CANDLE_BACKEND,
                                        CANDLE_BACKEND_SHA, CANDLE_ROOT_MANIFEST_SHA,
                                        apply_one_patch, exact_checkout, extract_exact)
from yue2_decoder_trace_overlay import tree_digest
from yue2_precision_proof import require, sha256, write_json

M4 = "825341ff8d0110ea448213485891b39d57806fa4"
ROOT_LOCK = "e5af5a1126d3bd5374104db537e9dfbd5f2dbc7a579185ce4cd63954d81fbe75"
VAE = Path("crates/audio/candle-audio-yue2/src/vae.rs")


def locked_tuples(path: Path) -> set[tuple[str, str, str | None, str | None]]:
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    return {(p["name"], p["version"], p.get("source"), p.get("checksum"))
            for p in document["package"]}


def stage(engine: Path, candle: Path, root: Path, harness: Path,
          control_sha: str) -> Path:
    engine, candle, root, harness = (path.resolve() for path in
                                     (engine, candle, root, harness))
    require(not root.exists(), "M4 diagnostic staging root must be fresh")
    exact_checkout(engine, M4)
    exact_checkout(candle, CANDLE_SHA, CANDLE_TREE)
    require(sha256(engine / "Cargo.lock") == ROOT_LOCK, "M4 root lock changed")
    require(control_sha and len(control_sha) == 40, "control source SHA absent")
    root.mkdir(parents=True)
    destination = root / "engine-overlay"
    engine_archive = root / "m4-tracked-source.tar.gz"
    extract_exact(engine, M4, engine_archive, destination)
    require(sha256(destination / "Cargo.lock") == ROOT_LOCK and
            sha256(destination / VAE) == sha256(engine / VAE),
            "exact tracked M4 source changed")
    base_tree = tree_digest(destination)
    vae_patch = harness / "m4-vae-observation.patch"
    apply_one_patch(destination, vae_patch, VAE)
    shutil.copy2(vae_patch, root / vae_patch.name)
    require(sha256(destination / "Cargo.lock") == ROOT_LOCK,
            "diagnostic VAE patch changed M4 lock")

    candle_overlay = root / "candle-overlay"
    candle_archive = root / "pinned-candle-tracked-source.tar.gz"
    extract_exact(candle, CANDLE_SHA, candle_archive, candle_overlay)
    require(sha256(candle_overlay / CANDLE_BACKEND) == CANDLE_BACKEND_SHA and
            sha256(candle_overlay / "Cargo.toml") == CANDLE_ROOT_MANIFEST_SHA,
            "pinned Candle backend/manifest changed")
    candle_base_tree = tree_digest(candle_overlay)
    backend_patch = harness / "candle-column.patch"
    kernel_patch = harness / "candle-kernel-path.patch"
    apply_one_patch(candle_overlay, backend_patch, CANDLE_BACKEND)
    apply_one_patch(candle_overlay, kernel_patch, Path("Cargo.toml"))
    for patch in (backend_patch, kernel_patch):
        shutil.copy2(patch, root / patch.name)

    original = locked_tuples(destination / "Cargo.lock")
    selected = locked_tuples(harness / "Cargo.lock.snapshot")
    exceptions = selected - original
    require(len(original) == 374 and len(selected) >= 206 and
            all(item[0] in ("yue2-bf16-tile-diagnostic", "candle-core") for item in exceptions) and
            len(selected & original) >= 204,
            "diagnostic dependency graph drifted from exact M4")
    # Materialize a standalone, run-owned Cargo project outside both checkouts.
    # The fixed lock is copied, never regenerated on the runner.
    build = root / "harness"
    build.mkdir()
    shutil.copytree(harness / "src", build / "src")
    template = (harness / "Cargo.toml.in").read_text(encoding="utf-8")
    require(template.count("__ENGINE_ROOT__") == 3 and
            template.count("__CANDLE_ROOT__") == 1,
            "standalone manifest path placeholders changed")
    manifest = template.replace("__ENGINE_ROOT__", destination.as_posix())
    manifest = manifest.replace("__CANDLE_ROOT__", candle_overlay.as_posix())
    (build / "Cargo.toml").write_text(manifest, encoding="utf-8")
    shutil.copy2(harness / "Cargo.lock.snapshot", build / "Cargo.lock")
    require(locked_tuples(build / "Cargo.lock") == selected,
            "materialized diagnostic lock changed")
    manifest = {
        "schema": "yue2-m4-pedantic-source-v1", "engineSha": M4,
        "controlSha": control_sha, "engineArchiveSha256": sha256(engine_archive),
        "engineBaseTreeSha256": base_tree,
        "engineDerivativeTreeSha256": tree_digest(destination),
        "baseVaeSha256": sha256(engine / VAE),
        "vaePatchSha256": sha256(vae_patch),
        "derivativeVaeSha256": sha256(destination / VAE),
        "m4RootLockSha256": ROOT_LOCK, "m4RootTupleCount": len(original),
        "harnessLockSha256": sha256(harness / "Cargo.lock.snapshot"),
        "harnessSourceTreeSha256": tree_digest(build),
        "harnessManifestSha256": sha256(build / "Cargo.toml"),
        "harnessTupleCount": len(selected), "unchangedM4TupleCount": len(selected & original),
        "candleSha": CANDLE_SHA, "candleTree": CANDLE_TREE,
        "candleArchiveSha256": sha256(candle_archive),
        "candleBaseTreeSha256": candle_base_tree,
        "candleDerivativeTreeSha256": tree_digest(candle_overlay),
        "candleBackendPatchSha256": sha256(backend_patch),
        "candleKernelPathPatchSha256": sha256(kernel_patch),
        "candleBackendDerivativeSha256": sha256(candle_overlay / CANDLE_BACKEND),
        "diagnosticOnly": True,
    }
    path = root / "m4-pedantic-source.json"
    write_json(path, manifest)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "candle", "root", "harness"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--control-sha", required=True)
    args = parser.parse_args()
    print(json.dumps({"provenance": str(stage(args.engine, args.candle,
                                              args.root, args.harness,
                                              args.control_sha))}))


if __name__ == "__main__":
    main()
