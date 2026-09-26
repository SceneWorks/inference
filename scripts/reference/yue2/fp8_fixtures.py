#!/usr/bin/env python3
"""Upstream FP8 E4M3 per-tensor quantization fixture for the native FP8 AR mode (sc-22995).

Evaluates the pinned upstream ``yue2.quantization.quantize_tensor`` (the function ``FP8Linear``
applies to every AR weight at preparation and to every activation per forward) on small,
deterministic BF16 tensors, and records the exact E4M3 codes (as the float values they decode to)
and the F32 scale. The native ``fp8::quantize_e4m3`` must reproduce both bit for bit
(``fp8::tests::e4m3_quantization_matches_upstream``).

No weights are loaded (the inputs are generated here); peak RSS ~0.4 GB. Run with the pinned
reference environment (see README.md):

    ~/.cache/sceneworks-yue2-ref/venv/bin/python scripts/reference/yue2/fp8_fixtures.py
"""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import torch
from yue2 import quantization

OUT = Path(__file__).resolve().parents[3] / "crates/audio/candle-audio-yue2/tests/fixtures/fp8_quantize.json"
COMMIT = "92a73cc7652fcc1f937855e4b765e0a0edd7ff2e"


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def cases():
    # A weight-like matrix: smooth values over several decades, both signs.
    n = 16 * 48
    w = [math.sin(i * 0.37) * (0.02 + 0.3 * ((i * 7919) % 97) / 97) for i in range(n)]
    yield "weight_like", [16, 48], w
    # An activation-like block with outliers (the dynamic scale is set by one element).
    a = [math.cos(i * 1.3) * 0.5 for i in range(32 * 16)]
    a[5] = 37.25
    a[300] = -41.5
    yield "activation_outliers", [32, 16], a
    # Tiny magnitudes near the E4M3 subnormal range after scaling, plus exact zeros.
    t = [((i % 11) - 5) * 1e-3 * (1 + (i % 3)) for i in range(16 * 16)]
    yield "small_with_zeros", [16, 16], t
    # All zeros: upstream clamps amax at 1e-12 before dividing.
    yield "zeros", [16, 16], [0.0] * 256


def main() -> None:
    out = {
        "generator": "scripts/reference/yue2/fp8_fixtures.py",
        "upstream_commit": COMMIT,
        "torch": torch.__version__,
        "function": "yue2.quantization.quantize_tensor",
        "cases": [],
    }
    for name, shape, values in cases():
        x = torch.tensor(values, dtype=torch.float32).reshape(shape).to(torch.bfloat16)
        q, scale = quantization.quantize_tensor(x)
        assert q.dtype == torch.float8_e4m3fn
        out["cases"].append({
            "name": name,
            "shape": shape,
            "input_bf16_bits": x.view(torch.int16).flatten().tolist(),
            "scale_f32_bits": f32_bits(float(scale.item())),
            "q_values": q.float().flatten().tolist(),
        })
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT} ({len(out['cases'])} cases)")


if __name__ == "__main__":
    main()
