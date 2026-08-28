#!/usr/bin/env python3
"""Compact the four duplicate synthetic SenseNova integration fixtures.

The T2I, IT2I, interleave, and VQA dump scripts all construct the same seeded tiny
NEOChat model.  They need different inputs and reference outputs, but previously
each committed a byte-identical copy of the model weights.  This tool retains one
shared ``sensenova_common_golden.safetensors`` and writes each case file with only
its own inputs, expected outputs, and metadata.

Regenerate the compact corpus from the vendored reference checkout:

    cd crates/media/mlx-gen/_vendor/sensenova_u1
    PYTHONPATH=src .venv/bin/python ../../tools/dump_sensenova_t2i_golden.py
    PYTHONPATH=src .venv/bin/python ../../tools/dump_sensenova_it2i_golden.py
    PYTHONPATH=src .venv/bin/python ../../tools/dump_sensenova_interleave_golden.py
    PYTHONPATH=src .venv/bin/python ../../tools/dump_sensenova_vqa_golden.py
    .venv/bin/python ../../tools/compact_sensenova_goldens.py

The dump scripts must run before this tool because they deliberately recreate the
four complete source fixtures.  The compacting pass is deterministic and rejects
any supposed common tensor whose dtype, shape, or bytes differ between cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file


CASES = ("t2i", "it2i", "interleave", "vqa")
COMMON = "sensenova_common_golden.safetensors"
HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURES = HERE.parent / "mlx-gen-sensenova" / "tests" / "fixtures"


def fixture_path(directory: Path, case: str) -> Path:
    return directory / f"{case}_golden.safetensors"


def read_fixture(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(path, framework="pt", device="cpu") as source:
        tensors = {
            name: source.get_tensor(name).contiguous() for name in sorted(source.keys())
        }
        # `safe_open` does not promise the metadata-map iteration order.  Canonicalize it before
        # serialization so two regenerations have the same safetensors header bytes as well as the
        # same tensor payloads.
        return tensors, dict(sorted((source.metadata() or {}).items()))


def tensors_match(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(dict(sorted(tensors.items())), temporary, metadata=dict(sorted(metadata.items())))
    # safetensors' Rust serializer stores metadata in a HashMap, so its header-key order varies
    # even when Python receives a sorted mapping.  Re-encode only the header with sorted metadata;
    # tensor payload bytes and their offsets are preserved exactly.
    raw = temporary.read_bytes()
    header_size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8 : 8 + header_size])
    header["__metadata__"] = dict(sorted(header.get("__metadata__", {}).items()))
    canonical = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode()
    canonical += b" " * (-len(canonical) % 8)
    temporary.write_bytes(len(canonical).to_bytes(8, "little") + canonical + raw[8 + header_size :])
    os.replace(temporary, path)


def compact(source_dir: Path, output_dir: Path) -> None:
    sources: dict[str, tuple[dict[str, torch.Tensor], dict[str, str]]] = {}
    for case in CASES:
        path = fixture_path(source_dir, case)
        if not path.is_file():
            raise SystemExit(f"missing full {case} fixture: {path}")
        sources[case] = read_fixture(path)

    reference, _ = sources[CASES[0]]
    common_names = {
        name
        for name, tensor in reference.items()
        if all(
            name in sources[case][0] and tensors_match(tensor, sources[case][0][name])
            for case in CASES[1:]
        )
    }
    if not common_names:
        raise SystemExit("no byte-identical shared tensors found; refusing to weaken fixture parity")

    common = {name: reference[name] for name in sorted(common_names)}
    cases = {
        case: {name: tensor for name, tensor in tensors.items() if name not in common_names}
        for case, (tensors, _) in sources.items()
    }
    if any(not tensors for tensors in cases.values()):
        raise SystemExit("a compact case has no load-bearing inputs or expectations")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_fixture(output_dir / COMMON, common, {"fixture_layout": "sensenova-shared-v1"})
    for case in CASES:
        _, metadata = sources[case]
        write_fixture(fixture_path(output_dir, case), cases[case], metadata)

    # Verify an exact lossless partition before reporting success.  This is deliberately stronger
    # than shape/statistics checks: every original tensor byte remains a test input or expectation.
    written_common, _ = read_fixture(output_dir / COMMON)
    for case in CASES:
        written_case, written_metadata = read_fixture(fixture_path(output_dir, case))
        original, original_metadata = sources[case]
        if written_metadata != original_metadata:
            raise SystemExit(f"{case}: metadata changed during compaction")
        rebuilt = written_common | written_case
        if rebuilt.keys() != original.keys() or any(
            not tensors_match(rebuilt[name], original[name]) for name in original
        ):
            raise SystemExit(f"{case}: compact fixture is not a lossless partition")

    before = sum(fixture_path(source_dir, case).stat().st_size for case in CASES)
    after = (output_dir / COMMON).stat().st_size + sum(
        fixture_path(output_dir, case).stat().st_size for case in CASES
    )
    print(f"common tensors={len(common)} bytes={(output_dir / COMMON).stat().st_size}")
    for case in CASES:
        path = fixture_path(output_dir, case)
        print(f"{case}: tensors={len(cases[case])} bytes={path.stat().st_size} sha256={digest(path)}")
    print(f"total: {before} -> {after} bytes ({(1 - after / before) * 100:.2f}% reduction)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIXTURES)
    args = parser.parse_args()
    compact(args.source_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
