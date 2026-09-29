#!/usr/bin/env python3
"""Check the exact-M2 F32 quality reference header without loading tensor payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def expected_shapes(fixture: dict) -> dict[str, list[int]]:
    expected = {}
    for mode, record in fixture["modes"].items():
        for phase, width in (("abc", 151644), ("semantic", 32769)):
            if record.get(phase) is not None:
                expected[f"ar/{mode}/{phase}"] = [len(record[phase]["emitted"]), width]
    expected.update(
        {
            "audio/multi_chunk_32": [383872],
            "audio/single_chunk_5": [92032],
            "latents/multi_chunk_32": [6400],
            "latents/single_chunk_5": [1536],
        }
    )
    return expected


def verify(reference: Path, fixture: Path) -> dict:
    size = reference.stat().st_size
    with reference.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("missing safetensors length")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > size - 8:
            raise ValueError("invalid safetensors header length")
        header = json.loads(stream.read(header_size))
    expected = expected_shapes(json.loads(fixture.read_text()))
    missing = sorted(expected.keys() - header.keys())
    if missing:
        raise ValueError(f"missing current-M2 fixture tensors: {missing}")
    payload_size = size - 8 - header_size
    for key, shape in expected.items():
        tensor = header[key]
        if tensor["dtype"] != "F32" or tensor["shape"] != shape:
            raise ValueError(f"{key}: incorrect dtype or shape")
        start, end = tensor["data_offsets"]
        count = 1
        for dimension in shape:
            count *= dimension
        if not (0 <= start <= end <= payload_size and end - start == 4 * count):
            raise ValueError(f"{key}: invalid or truncated tensor payload")
    digest = hashlib.sha256()
    with reference.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "sha256": digest.hexdigest(),
        "bytes": size,
        "required_tensors": len(expected),
        "fixture": str(fixture),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    args = parser.parse_args()
    identity = verify(args.reference, args.fixture)
    args.identity.write_text(json.dumps(identity, indent=2) + "\n")
    print(f"PASS: {identity['required_tensors']} F32 tensors, SHA-256 {identity['sha256']}")


if __name__ == "__main__":
    main()
