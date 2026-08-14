#!/usr/bin/env python3
"""Derive the conservative decode-quality implementation source closure.

The closure recursively hashes a fail-closed superset of the involved shared and provider crate
sources, manifests, and embedded assets. It intentionally over-invalidates when an included source
is not exercised by a particular route. This script only reads source bytes: it never executes those
sources, loads a model, samples memory, reads timing, or consumes calibration artifacts. The embedded
Rust constant is normalized to zeros before hashing so the digest can self-identify without a
recursive fixed-point problem.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = (
    ".github/workflows/real-weights.yml",
    "Cargo.lock",
    "Cargo.toml",
    "crates/contracts/gen-core/Cargo.toml",
    "crates/contracts/gen-core/src",
    "crates/media/mlx-gen/Cargo.toml",
    "crates/media/mlx-gen/src",
    "crates/media/mlx-gen/mlx-gen-z-image/Cargo.toml",
    "crates/media/mlx-gen/mlx-gen-z-image/src",
    "crates/media/mlx-gen/mlx-gen-flux/Cargo.toml",
    "crates/media/mlx-gen/mlx-gen-flux/assets",
    "crates/media/mlx-gen/mlx-gen-flux/src",
    "crates/media/mlx-gen/mlx-gen-kolors/Cargo.toml",
    "crates/media/mlx-gen/mlx-gen-kolors/src",
    "crates/media/mlx-gen/mlx-gen-kolors/tests/decode_quality_admission.rs",
    "crates/media/mlx-gen/mlx-gen-sdxl/Cargo.toml",
    "crates/media/mlx-gen/mlx-gen-sdxl/src",
    "crates/media/mlx-gen/mlx-gen-sdxl/tests/decode_quality_admission.rs",
    "crates/media/mlx-gen/mlx-gen-chroma/Cargo.toml",
    "crates/media/mlx-gen/mlx-gen-chroma/assets",
    "crates/media/mlx-gen/mlx-gen-chroma/src",
    "crates/media/mlx-gen/mlx-gen-chroma/tests/decode_quality_admission.rs",
    "scripts/ci/collect_decode_quality_admission.py",
    "scripts/ci/decode_quality_implementation_fingerprint.py",
)
EXCLUDED_PARTS = frozenset({".git", "__pycache__", "target"})
EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo"})


def semantic_source_closure(
    root: Path = ROOT, source_roots: tuple[str, ...] = SOURCE_ROOTS
) -> tuple[str, ...]:
    selected: set[str] = set()
    for relative in source_roots:
        path = root / relative
        if path.is_file():
            candidates = (path,)
        elif path.is_dir():
            candidates = (candidate for candidate in path.rglob("*") if candidate.is_file())
        else:
            raise ValueError(f"decode-quality source root does not exist: {relative}")
        for candidate in candidates:
            candidate_relative = candidate.relative_to(root)
            if EXCLUDED_PARTS.intersection(candidate_relative.parts):
                continue
            if candidate.suffix in EXCLUDED_SUFFIXES:
                continue
            selected.add(candidate_relative.as_posix())
    return tuple(sorted(selected))


FILES = semantic_source_closure()
CONSTANT = re.compile(
    rb'(MEMORY_DECODE_QUALITY_IMPLEMENTATION_FINGERPRINT: &str =\s*\n?\s*")[0-9a-f]{64}(";)'
)
CANONICAL_FIXTURE = re.compile(
    rb'(MEMORY_DECODE_QUALITY_CANONICAL_FIXTURE_SHA256: &str =\s*\n?\s*")[0-9a-f]{64}(";)'
)


def fingerprint() -> str:
    digest = hashlib.sha256()
    for relative in FILES:
        path = ROOT / relative
        payload = path.read_bytes()
        if relative.endswith("memory_strategy.rs") and "gen-core" in relative:
            payload, replacements = CONSTANT.subn(rb"\g<1>" + b"0" * 64 + rb"\g<2>", payload)
            if replacements != 1:
                raise ValueError("could not normalize the embedded decode-quality fingerprint")
            payload, replacements = CANONICAL_FIXTURE.subn(
                rb"\g<1>" + b"0" * 64 + rb"\g<2>", payload
            )
            if replacements != 1:
                raise ValueError("could not normalize the canonical decode-quality fixture hash")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\xff")
    return digest.hexdigest()


def embedded() -> str:
    payload = (ROOT / "crates/contracts/gen-core/src/memory_strategy.rs").read_bytes()
    match = CONSTANT.search(payload)
    if match is None:
        raise ValueError("could not read the embedded decode-quality fingerprint")
    return payload[match.start(0) : match.end(0)].decode("utf-8").split('"')[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    actual = fingerprint()
    if args.check and embedded() != actual:
        raise SystemExit(
            f"decode-quality implementation fingerprint is stale: embedded={embedded()} actual={actual}"
        )
    print(actual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
