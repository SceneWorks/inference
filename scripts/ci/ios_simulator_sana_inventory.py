#!/usr/bin/env python3
"""Bind a copied SANA BF16 component tree to a verified canonical snapshot inventory.

CI_IMPACT: one manual simulator workflow only; hashes the copied tier once after transfer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ci.pinned_hub_source import verify_pinned_hub_files

REQUIRED = {
    "transformer/diffusion_pytorch_model.safetensors",
    "text_encoder/gemma-2-2b-it.safetensors",
    "text_encoder/tokenizer.json",
    "vae/diffusion_pytorch_model.safetensors",
}
REPOSITORY = "SceneWorks/Sana_1600M_1024px_mlx"
REVISION = "ac421696dd6eb0b41f446f4c45a53ccc057d82a1"


def materialize_source_bf16(snapshot: Path, download: Callable[..., str] | None = None) -> dict:
    """Fetch only the pinned BF16 projection through an actual runner's Hub cache."""
    repository_directory = "models--" + REPOSITORY.replace("/", "--")
    if (not snapshot.is_absolute() or snapshot.name != REVISION
            or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name != repository_directory):
        raise ValueError("SANA BF16 materialization requires the pinned canonical Hub cache path")
    files = tuple("bf16/" + name for name in sorted(REQUIRED))
    missing = [name for name in files if not (snapshot / name).is_file()]
    if missing:
        if download is None:
            from huggingface_hub import snapshot_download
            download = snapshot_download
        fetched = Path(download(repo_id=REPOSITORY, revision=REVISION, token=False,
                                cache_dir=str(snapshot.parent.parent.parent),
                                allow_patterns=list(files), max_workers=4))
        if fetched != snapshot:
            raise ValueError("Hub fetched BF16 into a different snapshot path")
    if any(not (snapshot / name).is_file() for name in files):
        raise ValueError("pinned BF16 materialization left required files absent")
    return {"repository": REPOSITORY, "revision": REVISION, "snapshot": str(snapshot),
            "required_files": files, "missing_before_fetch": missing}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_digest(files: list[dict]) -> str:
    projection = [{"path": row["path"], "size": row["size"], "sha256": row["sha256"]}
                  for row in files]
    return hashlib.sha256(json.dumps(projection, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def authenticate_source_bf16(source: dict, snapshot: Path) -> dict:
    if (source.get("model") != "sana-1600m-mlx"
            or source.get("repository") != REPOSITORY or source.get("revision") != REVISION
            or snapshot.name != REVISION):
        raise ValueError("SANA BF16 source must be the pinned canonical snapshot")
    return verify_pinned_hub_files(REPOSITORY, REVISION, snapshot,
                                   tuple("bf16/" + name for name in sorted(REQUIRED)))


def verify_copied_bf16(source: dict, copied: Path) -> dict:
    if source.get("model") != "sana-1600m-mlx" or not source.get("revision"):
        raise ValueError("canonical SANA source inventory required")
    if not copied.is_dir() or copied.is_symlink():
        raise ValueError("copied Documents/sana must be a real directory")
    prefix = "bf16/"
    expected = sorted(({"path": row["path"][len(prefix):], "size": row["size"],
                        "sha256": row["sha256"]} for row in source["files"]
                       if row["path"].startswith(prefix)), key=lambda row: row["path"])
    paths = {row["path"] for row in expected}
    if not REQUIRED.issubset(paths) or len(paths) != len(expected):
        raise ValueError("verified source lacks a complete unique BF16 component tree")
    actual = []
    for entry in sorted(copied.rglob("*")):
        if entry.is_symlink():
            raise ValueError("copied BF16 tree contains a symlink")
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise ValueError("copied BF16 tree contains a non-file")
        actual.append({"path": entry.relative_to(copied).as_posix(),
                       "size": entry.stat().st_size, "sha256": sha256_file(entry)})
    if actual != expected:
        raise ValueError("copied BF16 content differs from verified canonical snapshot")
    return {"schema_version": 1, "model": source["model"], "revision": source["revision"],
            "tier": "bf16", "source_inventory_sha256": source["inventory_sha256"],
            "tier_inventory_sha256": content_digest(actual), "files": actual}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-inventory", type=Path)
    parser.add_argument("--source-snapshot", type=Path)
    parser.add_argument("--copied", type=Path)
    parser.add_argument("--materialize-bf16", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.materialize_bf16 is not None:
        if (args.source_inventory is not None or args.source_snapshot is not None
                or args.copied is not None):
            parser.error("BF16 materialization cannot combine with inventory verification")
        result = materialize_source_bf16(args.materialize_bf16)
        args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
        print("pinned SANA BF16 projection materialized through runner Hub cache")
        return 0
    if args.source_inventory is None or (args.source_snapshot is None) == (args.copied is None):
        parser.error("select exactly one of --source-snapshot or --copied")
    source = json.loads(args.source_inventory.read_text())
    result = (authenticate_source_bf16(source, args.source_snapshot)
              if args.source_snapshot is not None else verify_copied_bf16(source, args.copied))
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    print("SANA BF16 source authenticated against pinned Hub" if args.source_snapshot is not None
          else f"copied SANA BF16 content: OK tier_inventory_sha256={result['tier_inventory_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
