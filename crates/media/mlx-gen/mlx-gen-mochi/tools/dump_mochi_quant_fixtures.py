#!/usr/bin/env python3
"""Dump the Mochi 1 Q4/Q8 **transformer-Linear** quant parity fixture (sc-11990, story A6).

This is the small, **committed**, weight-free reference-`mx.quantize` golden that gates the crate's
packed-load path (`tests/quant_parity.rs`). It quantizes a deterministic bf16 weight the exact way
`convert.rs` (MLX `quantize`, group 64) and the packed loader (`MochiLinear::Quant` →
`quantized_matmul`, fp32-accumulate over f32 activations) do, then dumps both the packed parts and the
reference forward output. The Rust gate reproduces this **bit-for-bit** via BOTH routes:

  * **convert-then-load** — `convert::quantize_transformer_map` packs the same bf16 `w`, and
    `MochiLinear::load` consumes it; and
  * **consume-prequantized** — `MochiLinear::load` reads the packed `wq`/`scales`/`biases` dumped here.

The two routes agree with each other **bit-exact** on any single platform (byte-identical packs +
deterministic `quantized_matmul`), and each matches the dumped `q{bits}.y` within a tight ULP
tolerance — the small slack absorbs the NAX/non-NAX forward drift described below.

The op surface is stable MLX (`quantize`/`quantized_matmul`) — no torch/diffusers, no real Mochi
weights.

`quantized_matmul`'s f32 forward `q{bits}.y` is version- AND hardware-path-sensitive: under the
0.32.0 pin (epic 12742) the Apple-matrix-unit "NAX" path (deployment-target 26.2, self-hosted M-series)
and the non-NAX path (deployment-target 15.0, hosted PR CI) differ ~1-2 ULP-f32 (Q4 1.31e-6 / Q8
9.54e-7 on this fixture; MLX #3631/#3632/#3810). Only `q{bits}.y` moves — the packs `wq`/`scales`/
`biases` (and `x`/`w`) are byte-identical across 0.31.2->0.32.0 AND across NAX/non-NAX. On 0.31.2 both
Metal paths were bit-identical, so one golden served both; 0.32.0's NAX quant fixes broke that tie.

**The committed golden must be the NON-NAX (CI-matching) value**, which is identical to the original
0.31.2 dump (the non-NAX path did not move on the bump). The Rust gate (`tests/quant_parity.rs`)
compares against it with a tight relative ULP tolerance (`MOCHI_QUANT_GOLDEN_ULP_TOL`) so the
self-hosted NAX runner passes too. DO NOT re-dump this fixture on a NAX host (M-series at dt26.2) and
commit it — that captures the NAX value and re-breaks non-NAX CI (sc-12747). Regenerate only on a
non-NAX MLX build (hosted-CI-class Metal, dt15.0), or leave the 0.31.2-equivalent golden in place:

    uv run --with "mlx==0.32.0" python tools/dump_mochi_quant_fixtures.py   # ONLY on a non-NAX host

Writes (committed; ~0.3 MB):
  tests/fixtures/mochi_quant_slice.safetensors
    x           [B, in]   f32   — activations (Mochi's f32 DiT compute)
    w           [out, in] bf16  — the dense Linear weight (a `to_q`-shaped slice)
    q{4,8}.wq       u32   — packed codes
    q{4,8}.scales   bf16  — group scales
    q{4,8}.biases   bf16  — group biases
    q{4,8}.y    [B, out]  f32   — reference quantized_matmul(x, wq, scales, biases, transpose=True)
"""
import os

import mlx.core as mx

GROUP_SIZE = 64
OUT_F, IN_F, B = 128, 256, 5  # in divisible by group_size; f32 activations, [out, in] weight


def main() -> None:
    mx.random.seed(0)
    # A `to_q`-shaped Linear weight (bf16 — convert casts to bf16 before quantize) and f32 activations.
    w = (mx.random.normal((OUT_F, IN_F)) * 0.1).astype(mx.bfloat16)
    x = (mx.random.normal((B, IN_F)) * 1.0).astype(mx.float32)

    fixture = {"x": x, "w": w}
    for bits in (4, 8):
        wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=bits)
        # transpose=True → y = x @ dequant(w)^T, i.e. the `[out, in]` Linear applied to `[B, in]`.
        y = mx.quantized_matmul(
            x, wq, scales=scales, biases=biases, transpose=True, group_size=GROUP_SIZE, bits=bits
        ).astype(mx.float32)
        fixture[f"q{bits}.wq"] = wq
        fixture[f"q{bits}.scales"] = scales
        fixture[f"q{bits}.biases"] = biases
        fixture[f"q{bits}.y"] = y
        print(f"q{bits}: wq{list(wq.shape)} scales{list(scales.shape)} y{list(y.shape)}")

    dst = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(dst, exist_ok=True)
    path = os.path.abspath(os.path.join(dst, "mochi_quant_slice.safetensors"))
    mx.eval(list(fixture.values()))
    mx.save_safetensors(path, fixture)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
