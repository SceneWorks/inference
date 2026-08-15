#!/usr/bin/env python3
"""Derive the decode-quality implementation source closure recorded on each measurement.

The digest names the source tree that produced a measurement. It is a **forensic stamp, not an
admission gate** (sc-19728): the real-weight harness records it on every receipt, and nothing
refuses a row for carrying a different one. Measurements stand until they are remeasured.

That is what lets the closure stay a deliberately broad, fail-closed superset of the involved shared
and provider crate sources, manifests, and embedded assets. As a gate the breadth was the defect —
`Cargo.lock` and whole-crate `src` trees meant every dependency bump and main sync invalidated every
measurement, and the constant it was compared against was re-derived by hand each time. As a stamp
the same breadth is free: over-recording costs nothing and never blocks anyone.

This script only reads source bytes. It never executes those sources, loads a model, samples memory,
reads timing, or consumes calibration artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
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


def fingerprint() -> str:
    """Hash the closure verbatim.

    No byte is normalized away. The predecessor blanked two embedded Rust constants before hashing,
    because the digest was written back into one of them and the other tracked it — a fixed point
    that only existed because the value had to live in the binary to be compared against. Nothing
    compares it now, so nothing writes it back, so the hash is over the sources as they are.
    """
    digest = hashlib.sha256()
    for relative in FILES:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
        digest.update(b"\xff")
    return digest.hexdigest()


def main() -> int:
    # Deliberately no flags. Parsing anyway rejects stray arguments, so a caller still passing the
    # removed `--check` fails loudly instead of silently printing a digest it meant to verify.
    argparse.ArgumentParser().parse_args()
    print(fingerprint())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
