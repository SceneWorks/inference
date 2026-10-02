"""Stage an exact M3 tracked archive plus one declared, observational VAE source patch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tomllib

from yue2_precision_proof import require, sha256, write_json

ENGINE_SHA = "4127a675fc8575555e029e01b7f6867488880a8f"
LOCK_SHA = "e5af5a1126d3bd5374104db537e9dfbd5f2dbc7a579185ce4cd63954d81fbe75"
VAE_PATH = Path("crates/audio/candle-audio-yue2/src/vae.rs")


def tree_digest(root: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        require(path.is_file() and not path.is_symlink(), "overlay tree contains a non-file")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def stage(source: Path, destination: Path, patch: Path, engine_sha: str,
          control_sha: str) -> Path:
    require(engine_sha == ENGINE_SHA, "overlay source must be exact frozen M3")
    require(not destination.exists(), "overlay destination must be fresh")
    source = source.resolve(strict=True)
    patch = patch.resolve(strict=True)
    destination = destination.resolve()
    require(not destination.is_relative_to(source) and not source.is_relative_to(destination),
            "overlay must be outside the stationary engine source")
    head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True, encoding="utf-8").stdout.strip()
    dirty = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=normal"],
                           capture_output=True, text=True, check=True, encoding="utf-8").stdout
    require(head == ENGINE_SHA and not dirty, "stationary M3 source changed")
    require(sha256(source / "Cargo.lock") == LOCK_SHA, "M3 locked graph changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / "m3-tracked-source.tar.gz"
    require(not archive.exists(), "tracked archive must be fresh")
    subprocess.run(["git", "-C", str(source), "archive", "--format=tar.gz", "--output", str(archive),
                    "HEAD"], check=True)
    destination.mkdir(parents=True)
    with tarfile.open(archive) as package:
        for item in package:
            require(item.isfile() or item.isdir(), "M3 archive contains a link or special member")
            require(not Path(item.name).is_absolute() and ".." not in Path(item.name).parts,
                    "M3 archive path escapes destination")
        package.extractall(destination, filter="data")
    require(sha256(destination / "Cargo.lock") == LOCK_SHA and
            sha256(destination / VAE_PATH) == sha256(source / VAE_PATH),
            "tracked M3 archive changed locked source")
    baseline_tree = tree_digest(destination)
    patch_text = patch.read_text(encoding="utf-8")
    require(patch_text.startswith(f"--- a/{VAE_PATH.as_posix()}\n+++ b/{VAE_PATH.as_posix()}\n") and
            patch_text.count("\n--- ") == 0 and patch_text.count("\n+++ ") == 1,
            "overlay patch must touch only M3 vae.rs")
    subprocess.run(["git", "apply", "--check", "--unidiff-zero", str(patch)], cwd=destination, check=True)
    subprocess.run(["git", "apply", "--unidiff-zero", str(patch)], cwd=destination, check=True)
    shutil.copy2(patch, destination.parent / "decoder-trace-vae.patch")
    require(sha256(destination / "Cargo.lock") == LOCK_SHA, "overlay changed locked dependencies")
    lock = tomllib.loads((destination / "Cargo.lock").read_text(encoding="utf-8"))
    require(len(lock["package"]) == 374, "M3 root locked graph package count changed")
    provenance = {
        "schema": "yue2-decoder-trace-source-v1", "engine_sha": ENGINE_SHA,
        "control_sha": control_sha, "tracked_archive_sha256": sha256(archive),
        "base_tree_sha256": baseline_tree, "base_vae_sha256": sha256(source / VAE_PATH),
        "overlay_patch_sha256": sha256(patch),
        "derivative_vae_sha256": sha256(destination / VAE_PATH),
        "derivative_tree_sha256": tree_digest(destination),
        "cargo_lock_sha256": LOCK_SHA, "root_locked_packages": len(lock["package"]),
        "source_path": VAE_PATH.as_posix(), "derivative_only": True,
    }
    path = destination.parent / "overlay-provenance.json"
    require(not path.exists(), "overlay provenance path must be fresh")
    write_json(path, provenance)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "destination", "patch"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("engine-sha", "control-sha"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    print(json.dumps({"provenance": str(stage(args.source, args.destination, args.patch,
                                            args.engine_sha, args.control_sha))}))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError, tarfile.TarError) as error:
        raise SystemExit(f"yue2-decoder-trace-overlay: {error}") from error
