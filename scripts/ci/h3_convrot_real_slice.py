#!/usr/bin/env python3
"""Resolve a real MiniMax-H3 ConvRot INT8 checkpoint slice for the tiled INT8 GEMM parity proof.

`mlx-gen-minimax-h3`'s `int8_gemm_parity::real_checkpoint_slice_matches_serial_reference`
(sc-26691) compares the tiled Metal kernel with the serial reference on a published checkpoint's
own I8 codes, F32 scales and `.comfy_quant` descriptors. It needs only four projections of one
block, not the ~21 GB file, so:

1. a complete local copy in a Hugging Face hub cache is used in place (nothing is copied); else
2. exactly the selected tensors are fetched by HTTP byte range from the pinned public revision and
   written as a compact, valid safetensors file (header offsets re-based), after checking the
   remote header's declared byte ranges and the served file length against the pinned size.

The tensor bytes are data only; nothing downloaded is executed. Prints the resolved path on the
last stdout line. Standard library only; run with `python3 -I`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request
from pathlib import Path

REPOSITORY = "PulpCut/MiniMax-H3-Turbo-INT8-ConvRot"
REVISION = "e47c20aa569b03dabc37808f4f978a3988b959f7"
FILENAME = "minimax_h3_fl2va_pruned_turbo_int8_convrot.safetensors"
# Pinned in docs/testing of the SceneWorks sc-26691 campaign casebook (`pulpcut` asset set).
FILE_BYTES = 20_970_379_854
BASES = (
    "blocks.10.attn.qkv_proj",
    "blocks.10.attn.out_proj",
    "blocks.10.mlp.fc1",
    "blocks.10.mlp.fc2",
)
SUFFIXES = ("weight", "weight_scale", "comfy_quant")
MAX_HEADER = 64 * 1024 * 1024


def cache_candidates() -> list[Path]:
    roots = []
    for name in ("H3_INT8_GEMM_HUB_CACHE", "HF_HUB_CACHE"):
        if os.environ.get(name):
            roots.append(Path(os.environ[name]))
    roots += [Path("/Volumes/Models/huggingface/hub"), Path.home() / ".cache/huggingface/hub"]
    folder = "models--" + REPOSITORY.replace("/", "--")
    return [root / folder / "snapshots" / REVISION / FILENAME for root in roots]


def fetch(url: str, start: int, end: int) -> tuple[bytes, int | None]:
    """Bytes [start, end] inclusive, plus the total length the server reports."""
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        if response.status != 206:
            raise SystemExit(f"expected a 206 partial response, got {response.status}")
        data = response.read()
        total = response.headers.get("Content-Range", "").rpartition("/")[2]
    if len(data) != end - start + 1:
        raise SystemExit(f"short range read {len(data)} != {end - start + 1}")
    return data, int(total) if total.isdigit() else None


def materialize(output: Path) -> Path:
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}"
    prefix, total = fetch(url, 0, 7)
    if total != FILE_BYTES:
        raise SystemExit(f"served file length {total} != pinned {FILE_BYTES}")
    (header_len,) = struct.unpack("<Q", prefix)
    if not 0 < header_len <= MAX_HEADER:
        raise SystemExit(f"implausible header length {header_len}")
    header_bytes, _ = fetch(url, 8, 8 + header_len - 1)
    header = json.loads(header_bytes)
    data_start = 8 + header_len
    selected = {}
    for base in BASES:
        for suffix in SUFFIXES:
            name = f"{base}.{suffix}"
            if name not in header:
                raise SystemExit(f"pinned checkpoint has no {name}")
            selected[name] = header[name]
    out_header: dict = {"__metadata__": {"source": f"{REPOSITORY}@{REVISION}/{FILENAME}"}}
    chunks = []
    offset = 0
    for name, entry in selected.items():
        begin, end = entry["data_offsets"]
        if not 0 <= begin < end or data_start + end > FILE_BYTES:
            raise SystemExit(f"{name}: invalid data range {begin}..{end}")
        data, _ = fetch(url, data_start + begin, data_start + end - 1)
        chunks.append(data)
        out_header[name] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [offset, offset + len(data)],
        }
        print(f"{name} {entry['dtype']} {entry['shape']} sha256={hashlib.sha256(data).hexdigest()}")
        offset += len(data)
    encoded = json.dumps(out_header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for data in chunks:
            handle.write(data)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True, help="slice path if no cache copy")
    args = parser.parse_args()
    for candidate in cache_candidates():
        if candidate.is_file() and candidate.stat().st_size == FILE_BYTES:
            print(f"cached complete checkpoint: {candidate}")
            print(candidate)
            return 0
    print(f"no complete cached copy; fetching {len(BASES) * len(SUFFIXES)} tensors by range")
    print(materialize(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main())
