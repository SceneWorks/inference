#!/usr/bin/env python3
"""Derive the semantic decode-quality implementation source closure.

This hashes only correctness-relevant source. It never loads a model, samples memory, reads timing,
or consumes calibration artifacts. The embedded Rust constant is normalized to zeros before hashing
so the digest can self-identify without a recursive fixed-point problem.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FILES = (
    ".github/workflows/real-weights.yml",
    "crates/contracts/gen-core/src/memory_strategy.rs",
    "crates/contracts/gen-core/src/runtime.rs",
    "crates/media/mlx-gen/src/request_scope.rs",
    "crates/media/mlx-gen/src/diagnostics.rs",
    "crates/media/mlx-gen/src/image.rs",
    "crates/media/mlx-gen/src/vae_tiling.rs",
    "crates/media/mlx-gen/mlx-gen-z-image/src/vae/mod.rs",
    "crates/media/mlx-gen/mlx-gen-z-image/src/vae/attention.rs",
    "crates/media/mlx-gen/mlx-gen-z-image/src/vae/conv_layers.rs",
    "crates/media/mlx-gen/mlx-gen-z-image/src/vae/decoder.rs",
    "crates/media/mlx-gen/mlx-gen-z-image/src/vae/down_encoder_block.rs",
    "crates/media/mlx-gen/mlx-gen-z-image/src/vae/down_sampler.rs",
    "crates/media/mlx-gen/mlx-gen-z-image/src/vae/encoder.rs",
    "crates/media/mlx-gen/mlx-gen-z-image/src/vae/mid_block.rs",
    "crates/media/mlx-gen/mlx-gen-z-image/src/vae/resnet_block.rs",
    "crates/media/mlx-gen/mlx-gen-z-image/src/vae/up_decoder_block.rs",
    "crates/media/mlx-gen/mlx-gen-z-image/src/vae/up_sampler.rs",
    "crates/media/mlx-gen/mlx-gen-kolors/src/memory_strategy.rs",
    "crates/media/mlx-gen/mlx-gen-kolors/src/registry.rs",
    "crates/media/mlx-gen/mlx-gen-kolors/tests/decode_quality_admission.rs",
    "crates/media/mlx-gen/mlx-gen-sdxl/src/memory_strategy.rs",
    "crates/media/mlx-gen/mlx-gen-sdxl/src/model.rs",
    "crates/media/mlx-gen/mlx-gen-sdxl/src/pipeline.rs",
    "crates/media/mlx-gen/mlx-gen-sdxl/src/vae.rs",
    "crates/media/mlx-gen/mlx-gen-sdxl/tests/decode_quality_admission.rs",
    "crates/media/mlx-gen/mlx-gen-chroma/src/memory_strategy.rs",
    "crates/media/mlx-gen/mlx-gen-chroma/src/model.rs",
    "crates/media/mlx-gen/mlx-gen-chroma/tests/decode_quality_admission.rs",
    "scripts/ci/collect_decode_quality_admission.py",
)
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
