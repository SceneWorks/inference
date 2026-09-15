#!/usr/bin/env python3
"""Validate every MiniMax-H3 VRAM input before Cargo can create a CUDA context."""

from __future__ import annotations

import argparse
import json
import os
import stat
from pathlib import Path


def reparse(path: Path) -> bool:
    info = os.lstat(path)
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def real_file(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"missing required file: {path}")
    if reparse(path):
        raise ValueError(f"Windows reparse point is not a real staged file: {path}")


def real_dir(path: Path) -> None:
    if not path.is_dir():
        raise ValueError(f"missing required directory: {path}")
    if reparse(path):
        raise ValueError(f"Windows reparse point is not a real staged directory: {path}")


def indexed(directory: Path, index: str) -> None:
    real_dir(directory)
    index_path = directory / index
    real_file(index_path)
    try:
        weights = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values()
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"invalid shard index: {index_path}: {error}") from error
    names = set(weights)
    if not names:
        raise ValueError(f"empty shard index: {index_path}")
    for name in names:
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"unsafe shard name {name!r} in {index_path}")
        real_file(directory / name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dit", type=Path, required=True)
    parser.add_argument("--te", type=Path, required=True)
    parser.add_argument("--tier", choices=("q4", "q8", "bf16"), required=True)
    args = parser.parse_args()
    real_dir(args.root)
    indexed(args.dit, "diffusion_pytorch_model.safetensors.index.json")
    indexed(args.te, "model.safetensors.index.json")
    indexed(args.root / "vae", "diffusion_pytorch_model.safetensors.index.json")
    for relative in (
        "audio_vae/config.json", "audio_vae/diffusion_pytorch_model.safetensors",
        "FL2VA/audio_vae/config.json", "FL2VA/audio_vae/config.yaml", "FL2VA/audio_vae/metadata.json",
        "FL2VA/audio_vae/model.safetensors", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
    ):
        real_file(args.root / relative)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"MiniMax-H3 VRAM staging preflight failed: {error}") from error
