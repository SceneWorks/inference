#!/usr/bin/env python3
"""Generate tests/fixtures/minimax_h3_w4a8.json -- the sc-26685 independent W4A8 reference.

Golden values are produced by *executing* the pinned comfy-kitchen
``AsymW4A8Int8Layout`` Python layout over owned synthetic operands and by an
explicit, independent CPU replay of the three hashed CUDA kernels
(``w4a8_gemm.cu`` weight decode, ``int8_linear.cu`` ConvRot activation quant,
``cutlass_gemm_int8.cu``/``int8_linear.cu`` F32 epilogue).  They never come from
production Rust math and never from a dense floating fallback.

The literal ``QuantizedTensor`` + ``F.linear`` W4A8 INT8 dispatch is not
invocable on this CPU host (see ``reference.literalQuantizedTensorDispatch``);
when that is the case the pinned arithmetic is called directly with the explicit
CPU kernel replay above and the fixture is labelled ``cpu-kernel-replay``.  No
upstream CUDA execution is claimed.

Every operand is owned synthetic data.  No model payload, no HF cache and no
network are touched at execution time; the pinned source text is supplied
separately through ``--reference-dir`` and is hash-verified (git blob SHA-1 and
SHA-256) before any of it is executed.

Besides one entry per successful forward, the fixture pins what the published
contract must refuse (``refusedOperands``: negative/NaN ``s_rel`` and a
non-positive ``s_channel``) and one operand set carrying a W4A8 layer next to a
ConvRot INT8 layer (``mixedLayer``), whose ConvRot half reuses the external
``h3_convrot_int8.json`` activation verbatim.  Every successful case is generated
with an output width that is a multiple of 16, the CUDA INT8 IGEMM's geometry, so
no native comparison has to skip a case.

Test oracle only: never an inference dependency.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import platform
import struct
import sys
from fractions import Fraction
from pathlib import Path

import torch

# --------------------------------------------------------------------------------------
# Pinned source authority (comfy-kitchen PR#90 -> merged tree; see the worker brief).
# --------------------------------------------------------------------------------------
STORY = "sc-26685"
SCHEMA_VERSION = 1
LOADER_REVISION = "ad1d4e072bd2ed5eec1d30d07ec390f327a3af52"
KITCHEN_MAIN = "78b7fe78552944ffc5763d680ba57f023e7ab5f4"
KITCHEN_PR_HEAD = "d908d589fbda44595f6d78d9c47ef42ba4aa8fad"
KITCHEN_MERGED_TREE = "06a261f8917d2e0486eee14fe665b545a4419ca6"
FROZEN_ARTIFACT_TREE = "a56850c870825d6a8ae501a595c4dbbbc77bbc7e"

GROUP_SIZE = 16
CONVROT_GROUP = 256
CANDLE_OUTPUT_ROW_TILE = 64
MLX_COLUMN_TILE = 256

# path -> (git blob sha1, sha256, role)
PINNED_SOURCES: dict[str, tuple[str, str, str]] = {
    "comfy_kitchen/tensor/w4a8_int8.py": (
        "3d9c0fa5d282a970ed52237a7fcd67a71642a21f",
        "9a1a7cc80798035528d8c9b443df1f22ad1abe74d46353dd282749e3579c868c",
        "pinned-layout",
    ),
    "comfy_kitchen/backends/cuda/ops/w4a8_gemm.cu": (
        "21002ea10bf0d05d945083c61f79098324b76bb1",
        "7c8bfed0b6c05f67214b776c655aff1ab24018cee03e889346f5f9168273a58a",
        "arithmetic-authority",
    ),
    "comfy_kitchen/backends/cuda/ops/cutlass_gemm_int8.cu": (
        "f7142fa41973a808cd6a5fe4b0b1cd063a7b2a3b",
        "7e5a0da97cd50e14a503eaa6bb541b7e161b5d4197b9bdde70cb9e32e7102e27",
        "arithmetic-authority",
    ),
    "comfy_kitchen/backends/cuda/ops/int8_linear.cu": (
        "4c7d49e72425aa868d388a5b73b98c585e1fbded",
        "f7ad0f88f7118e4c4b318b0f5ddb27684eaead5f34efd9fb780d12680b2c3f8c",
        "arithmetic-authority",
    ),
    "comfy_kitchen/tensor/int8_utils.py": (
        "beff8237e3b37221ac0d88e1d4712f00d16b9daa",
        "4776532e7394c56262cc02c48217a20b520d123c2bf9539eaf77cf0ff32ff1b8",
        "pinned-layout-helper",
    ),
    "comfy_kitchen/backends/eager/w4a8_int8.py": (
        "814e4d707277e3ffd9344a42ef461a1bb6038a9e",
        "a6f48b1039acf539db95cb82de11bebc60a185c3ff2bcdc9982125a459648f71",
        "cpu-fallback-label",
    ),
    "comfy_kitchen/backends/eager/quantization.py": (
        "dc3786b71282ce4510b0e3a2ed7f1a8e44157044",
        "13a3366a200d71b79d6bb8e53bc5bf8dcafa0f5d0410775843809df201f36e53",
        "cpu-fallback-label",
    ),
    "comfy_kitchen/backends/triton/w4a8_int8.py": (
        "376d4d65942fef41fa90bf130d84bc3caef19ea6",
        "249dc77e83f974e171d178cb1984ff1be607f6539f476da4c535457c0d3513a6",
        "cpu-fallback-label",
    ),
    "comfy_kitchen/registry.py": (
        "94d0aadf76fef5b09d7a25ba28e9045ac0dd8cd0",
        "181b02ee1ee60154766cebb3cad6dc5d2015c38551e10071303a5a2a897f7015",
        "cpu-fallback-label",
    ),
}

REGEN_COMMAND = (
    "python3 scripts/ci/generate_minimax_h3_w4a8_fixture.py "
    "--reference-dir <pinned-comfy-kitchen-source-dir> "
    "--convrot-fixture tests/fixtures/h3_convrot_int8.json "
    "--output tests/fixtures/minimax_h3_w4a8.json --receipt <receipt.json>"
)

# --------------------------------------------------------------------------------------
# Exact scalar arithmetic.  Every CUDA float op is replayed as one correctly rounded
# binary32 operation; the only inexact-in-binary64 op (division) is rounded from an
# exact rational so no double rounding can occur.
# --------------------------------------------------------------------------------------
_NUL = bytes([0])


def f32(x) -> float:
    """Round to binary32, ties to even (the IEEE-754 default CUDA float rounding)."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def f32_div(a: float, b: float) -> float:
    """binary32 division rounded from the exact rational quotient."""
    return f32(Fraction(a) / Fraction(b))


def f32_mul(a: float, b: float) -> float:
    """Correctly rounded binary32 multiply.  Both operands are binary32, so their exact
    product needs 48 significand bits: the binary64 product is exact and the single
    rounding below is the IEEE binary32 result (no double rounding)."""
    return f32(float(a) * float(b))


def f32_add(a: float, b: float) -> float:
    """Correctly rounded binary32 add.  The exact sum of two binary32 values fits in
    binary64 (53 >= 2*24 + 2), so the single rounding below is the IEEE result."""
    return f32(float(a) + float(b))


def rn_int(x: float) -> int:
    """__float2int_rn / nearbyintf: round to nearest integer, ties to even."""
    return round(x)


def cuda_fminf(a: float, b: float) -> float:
    if a != a:
        return b
    if b != b:
        return a
    return a if a < b else b


def cuda_fmaxf(a: float, b: float) -> float:
    if a != a:
        return b
    if b != b:
        return a
    return a if a > b else b


def bf16_bits(value: float) -> int:
    """__float2bfloat16_rn on a binary32 value -> 16 raw bits, ties to even."""
    u = struct.unpack("<I", struct.pack("<f", value))[0]
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF


def bf16_rn(value: float) -> float:
    return struct.unpack("<f", struct.pack("<I", (bf16_bits(value) << 16) & 0xFFFFFFFF))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def fp8_e4m3fn_to_f32(bits: int) -> float:
    """Exact __nv_cvt_fp8_to_halfraw(v, __NV_E4M3) then __half2float.

    Every finite e4m3fn value is exactly representable in binary16 and binary32, so the
    two-step CUDA conversion widens without loss; this decodes the same value directly.
    0x7f/0xff are NaN in e4m3fn (there is no infinity).
    """
    b = int(bits) & 0xFF
    sign = -1.0 if b & 0x80 else 1.0
    exponent = (b >> 3) & 0xF
    mantissa = b & 0x7
    if exponent == 0xF and mantissa == 0x7:
        return float("nan")
    magnitude = (mantissa * 2.0 ** -9) if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
    return sign * magnitude


def f32_to_fp8_e4m3fn_bits(value: float) -> int:
    """binary32 -> e4m3fn raw bits by nearest representable magnitude (ties to even).

    Only used to build relative-scale operands from exact encodings (1.0, 0.5, 2.0, 3.0,
    64.0, 448.0, 2**-9), so ``verify_fp8_decode`` asserts the round trip on every pattern
    this generator can emit.
    """
    v = f32(value)
    if v != v:
        return 0x7F
    sign = 0x80 if math.copysign(1.0, v) < 0 else 0
    a = abs(v)
    if a == 0.0:
        return sign
    best_key = best_bits = None
    for exponent in range(16):
        for mantissa in range(8):
            if exponent == 0xF and mantissa == 0x7:
                continue                       # 0x7f/0xff are NaN in e4m3fn
            magnitude = ((mantissa * 2.0 ** -9) if exponent == 0
                         else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
            key = (abs(magnitude - a), exponent & 1, mantissa & 1, exponent, mantissa)
            if best_key is None or key < best_key:
                best_key, best_bits = key, sign | (exponent << 3) | mantissa
    return best_bits


# --------------------------------------------------------------------------------------
# Hashed CUDA kernel replay (independent CPU ports; the production Rust math is never
# consulted and no dense floating fallback is used anywhere on this path).
# --------------------------------------------------------------------------------------


def h4_row_dot(d: int, x0: float, x1: float, x2: float, x3: float) -> float:
    """int8_linear.cu h4_row_dot: row `d` of the regular (non-Walsh) H4, in source order."""
    if d == 0:
        return f32_add(f32_add(f32_add(x0, x1), x2), -x3)
    if d == 1:
        return f32_add(f32_add(f32_add(x0, x1), -x2), x3)
    if d == 2:
        return f32_add(f32_add(f32_add(x0, -x1), x2), x3)
    return f32_add(f32_add(f32_add(-x0, x1), x2), x3)


def convrot_fht_256(block: list[float]) -> list[float]:
    """quantize_int8_rowwise_convrot_kernel FHT: 4 stages at strides 1/4/16/64, 0.5 each."""
    src = list(block)
    dst = [0.0] * CONVROT_GROUP
    for stride in (1, 4, 16, 64):
        for i in range(CONVROT_GROUP):
            d = (i // stride) & 3
            base = i - d * stride
            dst[i] = f32(0.5 * h4_row_dot(d, src[base], src[base + stride], src[base + 2 * stride], src[base + 3 * stride]))
        src, dst = dst, src
    return src


BF16_MAX = f32(3.38953139e38)          # int8_linear.cu finite_max_for_dtype<nv_bfloat16>
ROW_SCALE_RECIP = f32(1.0 / 127.0)     # int8_linear.cu `* (1.0f / 127.0f)` compile-time constant
ROW_SCALE_FLOOR = 1.0e-30


def rotate_rows(rows: list[list[float]]) -> list[list[float]]:
    """Block-diagonal ConvRot (regular H4, groups of 256) exactly as the CUDA FHT."""
    out = []
    for row in rows:
        assert len(row) % CONVROT_GROUP == 0, "K must be a multiple of the ConvRot group"
        rotated: list[float] = []
        for g in range(len(row) // CONVROT_GROUP):
            rotated.extend(convrot_fht_256(row[g * CONVROT_GROUP:(g + 1) * CONVROT_GROUP]))
        out.append(rotated)
    return out


def quantize_int8_rowwise_convrot_replay(rows_bf16: list[list[float]]):
    """int8_linear.cu quantize_int8_rowwise_convrot_kernel (InputType = nv_bfloat16).

    One global row scale over the complete row (all ConvRot groups), scale rounded to
    BF16 before the division, quotient rounded to BF16 before nearbyintf, clamp [-128,127].
    """
    rotated = rotate_rows(rows_bf16)
    codes: list[list[int]] = []
    scales: list[float] = []
    group_absmax: list[list[float]] = []
    for row in rotated:
        abs_max = 0.0
        per_group = []
        for g in range(len(row) // CONVROT_GROUP):
            block_max = max(abs(v) for v in row[g * CONVROT_GROUP:(g + 1) * CONVROT_GROUP])
            per_group.append(block_max)
            abs_max = max(abs_max, block_max)
        group_absmax.append(per_group)
        scale = cuda_fmaxf(f32(cuda_fminf(abs_max, BF16_MAX) * ROW_SCALE_RECIP), ROW_SCALE_FLOOR)
        scales.append(scale)
        scale_t = bf16_rn(scale)
        row_codes = []
        for value in row:
            scaled = bf16_rn(f32_div(bf16_rn(value), scale_t))
            # int8_linear.cu: quantized = nearbyintf(scaled); THEN the float clamp
            # [-128,127]; THEN the int8 cast.  Rounding before clamping is what makes
            # xq == -128 reachable at all (a -127.5 quotient rounds to -128 first).
            quantized = float(rn_int(scaled))
            row_codes.append(int(cuda_fminf(127.0, cuda_fmaxf(-128.0, quantized))))
        codes.append(row_codes)
    return rotated, codes, scales, group_absmax


def decode_weight_replay(packed: list[list[int]], srel_bits: list[list[int]],
                         codebook: list[float] | None, k: int, group: int = GROUP_SIZE):
    """w4a8_gemm.cu dequant_int4_grouped_to_int8_kernel.

    Even column -> LOW nibble, odd column -> HIGH nibble; the scale is decoded from raw
    e4m3 bits per group; level = codebook[code] or (code - 8); round to nearest even of
    the binary32 product, THEN clamp [-127,127] as integers (deliberately not [-128,127]).
    """
    n = len(packed)
    out: list[list[int]] = []
    for i in range(n):
        row = []
        for j in range(k // 2):
            byte = int(packed[i][j]) & 0xFF
            even_code = byte & 0xF
            odd_code = (byte >> 4) & 0xF
            for offset, code in ((0, even_code), (1, odd_code)):
                col = 2 * j + offset
                s = fp8_e4m3fn_to_f32(srel_bits[i][col // group])
                level = (codebook[code] if codebook is not None else float(code) - 8.0)
                value = rn_int(f32_mul(level, s))
                row.append(max(-127, min(127, value)))
        out.append(row)
    return out


def int_accumulate_replay(xq: list[list[int]], w: list[list[int]], k_tile: int = CONVROT_GROUP):
    """cutlass_gemm_int8.cu ElementAcc = int32_t over the complete K.

    Exact integer accumulation (int32 addition is exact and order independent as long as
    no overflow); per-k_tile partial sums are recorded so a consumer can prove the total
    is the exact complete-K dot and that no per-chunk rounding was applied.
    """
    m, n = len(xq), len(w)
    acc: list[list[int]] = []
    partials: list[int] = []
    terms = 0
    for r in range(m):
        row_out = []
        for c in range(n):
            total = 0
            for t in range(0, len(xq[r]), k_tile):
                part = 0
                for k in range(t, min(t + k_tile, len(xq[r]))):
                    part += int(xq[r][k]) * int(w[c][k])
                terms += min(k_tile, len(xq[r]) - t)
                partials.append(part)
                total += part
            if not (-2 ** 31 <= total <= 2 ** 31 - 1):
                raise OverflowError(f"int32 accumulator overflow: {total}")
            row_out.append(total)
        acc.append(row_out)
    return acc, partials, terms


def epilogue_replay(acc: int, x_scale: float, w_scale: float, bias: float | None) -> float:
    """cutlass_gemm_int8.cu EVT Accum -> XScale -> Mul0 -> WScale -> Mul1 -> Add2 -> StoreD.

    EXACTLY ((acc * xs) * ws) + bias in binary32 with three separate roundings and a
    single round-to-nearest at the store (the F32 product chain is never fused into an
    FMA and the bias is a binary32 add, not a BF16 one).
    """
    p1 = f32_mul(f32(float(acc)), x_scale)
    p2 = f32_mul(p1, w_scale)
    p3 = f32_add(p2, bias) if bias is not None else p2
    return bf16_rn(p3)


# --------------------------------------------------------------------------------------
# Deterministic owned synthetic data (no HF payload, no model weights).
# --------------------------------------------------------------------------------------
_MASK64 = (1 << 64) - 1


class SplitMix64:
    """Portable deterministic stream; independent of any library RNG version."""

    def __init__(self, seed: int):
        self.state = seed & _MASK64

    def u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
        return (z ^ (z >> 31)) & _MASK64

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * ((self.u64() >> 11) / float(1 << 53))

    def approx_gaussian(self) -> float:
        # Irwin-Hall(4) approximation: deterministic, symmetric, unimodal.
        return sum(self.uniform(-1.0, 1.0) for _ in range(4)) / 2.0

    def int_in(self, lo: int, hi: int) -> int:
        return lo + self.u64() % (hi - lo + 1)


def to_bf16_rows(rows: list[list[float]]) -> list[list[float]]:
    return [[bf16_rn(v) for v in row] for row in rows]


# --------------------------------------------------------------------------------------
# Pinned Python layout extraction (the arithmetic bodies are the pinned source text).
# --------------------------------------------------------------------------------------


def _pinned_definitions(path: Path, names: set[str], namespace: dict) -> dict:
    """Extract and execute pinned source definitions verbatim (no rewriting).

    Functions, classes and module-level literal constants named in `names` are compiled
    straight from the pinned AST in source order.  Everything else the bodies reference is
    supplied by the caller: the standard library, torch, and the small non-arithmetic
    shims below (a base dataclass field surface and the CUDA-extension availability flag).
    """
    tree = ast.parse(path.read_bytes(), filename=str(path))
    literal_values = (ast.Constant, ast.List, ast.Dict, ast.Tuple, ast.Set)
    selected: dict = {}

    def walk(nodes):
        for node in nodes:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
                selected[node.name] = node
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names
                  and isinstance(node.value, literal_values)):
                selected[node.targets[0].id] = node
            elif isinstance(node, ast.ClassDef):
                walk(node.body)

    walk(tree.body)
    if set(selected) != names:
        raise ValueError(f"{path}: missing pinned definitions {sorted(names - set(selected))}")
    for name, node in selected.items():
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return selected


@dataclasses.dataclass(frozen=True)
class _BaseLayoutParams:
    """Field surface of comfy_kitchen.tensor.base.BaseLayoutParams used by the pinned
    Params dataclass.  Supplied locally; contains no arithmetic."""

    scale: object = None
    orig_dtype: object = None
    orig_shape: object = None


def build_pinned_layout(reference_dir: Path):
    """Execute the pinned AsymW4A8Int8Layout encoder/decoder over owned synthetic data."""
    namespace = {
        "torch": torch,
        "math": math,
        "dataclass": dataclasses.dataclass,
        "BaseLayoutParams": _BaseLayoutParams,
        "QuantizedLayout": object,
        "_CUDA_AVAILABLE": False,   # torch.version.cuda is None on this CPU host
    }
    _pinned_definitions(reference_dir / "comfy_kitchen/tensor/int8_utils.py",
                        {"_HADAMARD_CACHE", "_build_hadamard", "_rotate_weight",
                         "_rotate_activation"}, namespace)
    _pinned_definitions(reference_dir / "comfy_kitchen/tensor/w4a8_int8.py",
                        {"_convrot_rotate", "AsymW4A8Int8Layout"}, namespace)
    return namespace


# --------------------------------------------------------------------------------------
# Case construction
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class Operands:
    case_id: str
    purpose: str
    params_origin: str
    m: int
    n: int
    k: int
    x_bf16: list[list[float]]
    packed: list[list[int]]          # unsigned bytes 0..255, [n, k/2]
    srel_bits: list[list[int]]       # raw e4m3 bytes 0..255, [n, k/group]
    s_channel: list[float]
    codebook: list[float] | None
    bias: list[float] | None
    note: str = ""


def pack_codes(q_u: list[list[int]], k: int) -> list[list[int]]:
    """Pinned packing: even column -> low nibble, odd column -> high nibble, stored int8."""
    packed = []
    for row in q_u:
        assert len(row) == k
        out = []
        for j in range(k // 2):
            byte = (int(row[2 * j]) & 0xF) | ((int(row[2 * j + 1]) & 0xF) << 4)
            out.append(byte)
        packed.append(out)
    return packed


def synthetic_weight(rng: SplitMix64, n: int, k: int, row_span: float, group_span: float) -> list[list[float]]:
    rows = []
    for i in range(n):
        row_amp = 10.0 ** rng.uniform(-row_span, row_span)
        groups = k // GROUP_SIZE
        group_amps = [10.0 ** rng.uniform(-group_span, group_span) for _ in range(groups)]
        row = []
        for j in range(k):
            row.append(row_amp * group_amps[j // GROUP_SIZE] * rng.approx_gaussian())
        rows.append(row)
    return rows


def operands_from_pinned_quantize(ns, case_id, purpose, note, weight_f64, x_f64, bias_f64):
    n, k = len(weight_f64), len(weight_f64[0])
    m = len(x_f64)
    w = torch.tensor([row[:] for row in weight_f64], dtype=torch.float32).to(torch.bfloat16)
    with torch.no_grad():
        qpacked, params = ns["AsymW4A8Int8Layout"].quantize(
            w, group_size=GROUP_SIZE, convrot_groupsize=CONVROT_GROUP,
            symmetric=True, scale_dtype=torch.float8_e4m3fn, codebook=True,
        )
    packed = [[int(b) & 0xFF for b in row.tolist()] for row in qpacked.to(torch.int32).view(n, k // 2)]
    srel_bits = [[int(b) & 0xFF for b in row.tolist()] for row in params.scale.view(torch.uint8)]
    s_channel = [float(v) for v in params.s_channel.tolist()]
    codebook = [float(v) for v in params.codebook.tolist()]
    return Operands(case_id, purpose, "pinned-AsymW4A8Int8Layout.quantize", m, n, k,
                    to_bf16_rows(x_f64), packed, srel_bits, s_channel, codebook,
                    bias_f64, note)


def operands_constructed(case_id, purpose, note, x_f64, q_u, srel_bits, s_channel, codebook, bias):
    n, k = len(q_u), len(q_u[0])
    m = len(x_f64)
    return Operands(case_id, purpose, "constructed", m, n, k, to_bf16_rows(x_f64),
                    pack_codes(q_u, k), srel_bits, s_channel, codebook, bias, note)


# --------------------------------------------------------------------------------------
# Counterfactual variants (each is a wrong implementation the fixture must distinguish).
# --------------------------------------------------------------------------------------


def cf_dense_float(op: Operands, ns, x_bf16):
    """Dense floating fallback: dequantized original-basis weight x unrotated f32 input."""
    qpacked = torch.tensor(op.packed, dtype=torch.uint8).view(torch.int8)
    params = ns["AsymW4A8Int8Layout"].Params(
        scale=torch.tensor(op.srel_bits, dtype=torch.uint8).view(torch.float8_e4m3fn),
        s_channel=torch.tensor(op.s_channel, dtype=torch.float32),
        correction=None,
        codebook=torch.tensor(op.codebook, dtype=torch.float32),
        orig_dtype=torch.bfloat16,
        orig_shape=(op.n, op.k),
        group_size=GROUP_SIZE,
        convrot_groupsize=CONVROT_GROUP,
    )
    w_orig = ns["AsymW4A8Int8Layout"].dequantize(qpacked, params).float()
    out = []
    for r in range(op.m):
        row = []
        for c in range(op.n):
            total = 0.0
            for k in range(op.k):
                total = f32_add(total, f32_mul(float(x_bf16[r][k]), float(w_orig[c][k])))
            row.append(bf16_rn(f32_add(total, op.bias[c] if op.bias else 0.0)))
        out.append(row)
    return out


def cf_no_activation_quant(rotated, acc, xs, decoded, s_channel, bias, m, n):
    out = []
    for r in range(m):
        row = []
        for c in range(n):
            total = 0.0
            for k in range(len(rotated[r])):
                total = f32_add(total, f32_mul(rotated[r][k], float(decoded[c][k])))
            row.append(bf16_rn(f32_add(total, bias[c] if bias else 0.0)))
        out.append(row)
    return out


def cf_per_tile_requant(x_bf16, decoded, s_channel, bias, m, n, k):
    """Wrong per-K-tile activation requant: each 256-column ConvRot tile gets its own int8
    codes and its own row scale, and the per-tile scaled partials are summed in binary32."""
    out = [[0.0] * n for _ in range(m)]
    tiles = k // CONVROT_GROUP
    for r in range(m):
        tile_codes, tile_scales = [], []
        for g in range(tiles):
            block = x_bf16[r][g * CONVROT_GROUP:(g + 1) * CONVROT_GROUP]
            _, codes, scales, _ = quantize_int8_rowwise_convrot_replay([block])
            tile_codes.append(codes[0])
            tile_scales.append(scales[0])
        for c in range(n):
            total = 0.0
            for g in range(tiles):
                part = 0
                for j in range(CONVROT_GROUP):
                    part += int(tile_codes[g][j]) * int(decoded[c][g * CONVROT_GROUP + j])
                total = f32_add(total, f32_mul(f32(float(part)), tile_scales[g]))
            out[r][c] = bf16_rn(f32_add(f32_mul(total, s_channel[c]), bias[c] if bias else 0.0))
    return out



def cf_nibble_swapped(op: Operands):
    """Wrong packing: even column -> HIGH nibble instead of LOW."""
    swapped = [[((b & 0xF) << 4) | ((b >> 4) & 0xF) for b in row] for row in op.packed]
    return decode_weight_replay(swapped, op.srel_bits, op.codebook, op.k)


def cf_weight_clamp_sym128(op: Operands):
    """Wrong weight-decode clamp: [-128,128] instead of the kernel's [-127,127]."""
    out = []
    for i in range(op.n):
        new = []
        for j in range(op.k):
            byte = op.packed[i][j // 2]
            code = (byte & 0xF) if j % 2 == 0 else ((byte >> 4) & 0xF)
            s = fp8_e4m3fn_to_f32(op.srel_bits[i][j // GROUP_SIZE])
            level = op.codebook[code] if op.codebook else float(code) - 8.0
            raw = rn_int(f32_mul(level, s))
            new.append(max(-128, min(128, raw)))
        out.append(new)
    return out


def cf_weight_round_half_up(op: Operands):
    out = []
    for i in range(op.n):
        row = []
        for j in range(op.k):
            byte = op.packed[i][j // 2]
            code = (byte & 0xF) if j % 2 == 0 else ((byte >> 4) & 0xF)
            s = fp8_e4m3fn_to_f32(op.srel_bits[i][j // GROUP_SIZE])
            level = op.codebook[code] if op.codebook else float(code) - 8.0
            product = f32_mul(level, s)
            value = math.floor(product + 0.5) if product >= 0 else math.ceil(product - 0.5)
            row.append(max(-127, min(127, int(value))))
        out.append(row)
    return out


# --------------------------------------------------------------------------------------
# Internal verification
# --------------------------------------------------------------------------------------


def verify_pinned_sources(reference_dir: Path) -> list[dict]:
    records = []
    for rel, (want_blob, want_sha, role) in PINNED_SOURCES.items():
        path = reference_dir / rel
        if not path.is_file():
            raise FileNotFoundError(f"pinned source missing: {path}")
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + _NUL + data).hexdigest()
        sha = hashlib.sha256(data).hexdigest()
        if blob != want_blob or sha != want_sha:
            raise ValueError(
                f"pinned source hash mismatch for {rel}: blob {blob} (want {want_blob}), "
                f"sha256 {sha} (want {want_sha})")
        records.append({"path": rel, "blob": blob, "sha256": sha, "bytes": len(data), "role": role})
    return records


def verify_fp8_decode() -> dict:
    """Cross-check the e4m3 decode against torch's own float8_e4m3fn on all 256 patterns."""
    mismatches = []
    for bits in range(256):
        t = torch.tensor([bits], dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
        mine = fp8_e4m3fn_to_f32(bits)
        theirs = float(t[0])
        same = (mine != mine and theirs != theirs) or mine == theirs
        if not same:
            mismatches.append(bits)
    if mismatches:
        raise ValueError(f"fp8 e4m3 decode disagrees with torch on {mismatches}")
    round_trip_bad = []
    for bits in range(256):
        value = fp8_e4m3fn_to_f32(bits)
        if value != value:
            continue
        again = fp8_e4m3fn_to_f32(f32_to_fp8_e4m3fn_bits(value))
        if again != value and not (again != again):
            round_trip_bad.append(bits)
    if round_trip_bad:
        raise ValueError(f"fp8 e4m3 encode/decode round trip failed on {round_trip_bad}")
    # The exact encodings this generator emits must match torch's own f32 -> e4m3fn cast.
    emitted = [0.0, 1.0, 0.5, 2.0, 3.0, 64.0, 448.0, 2.0 ** -9, -1.0, -448.0]
    encode_mismatches = []
    for value in emitted:
        theirs = int(torch.tensor([value], dtype=torch.float32).to(torch.float8_e4m3fn)
                     .view(torch.uint8)[0])
        mine = f32_to_fp8_e4m3fn_bits(value)
        if mine != theirs:
            encode_mismatches.append({"value": value, "mine": mine, "torch": theirs})
        if fp8_e4m3fn_to_f32(mine) != f32(value):
            encode_mismatches.append({"value": value, "note": "not an exact e4m3 encoding"})
    if encode_mismatches:
        raise ValueError(f"fp8 e4m3 encode disagrees with torch: {encode_mismatches}")
    return {"patterns": 256, "torchDecodeMismatches": 0, "roundTripMismatches": [],
            "torchEncodeMismatches": 0, "emittedEncodings": len(emitted)}


def verify_scale_constant() -> dict:
    """The hashed kernel computes the row scale as
    ``fmaxf(fminf(abs_max, BF16_MAX) * (1.0f / 127.0f), 1.0e-30f)``.  The brief and the
    primary-evidence summary describe it as a division by 127.  Those two forms are NOT bit
    equivalent for binary32, so the fixture pins the source form and records exactly where
    the alternative would disagree."""
    rng = SplitMix64(0x5C266851)
    differing = 0
    checked = 0
    examples = []
    probes = [0.0, ROW_SCALE_FLOOR, 1e-30, 1e-20, 1.0, 127.0, 128.0, 65504.0,
              3.38953139e38, 3.4028234663852886e38]
    for _ in range(20000):
        probes.append(f32(10.0 ** rng.uniform(-38, 38) * rng.uniform(-1.0, 1.0)))
    for value in probes:
        clamped = cuda_fminf(f32(value), BF16_MAX)
        by_reciprocal = f32_mul(clamped, ROW_SCALE_RECIP)
        by_division = f32_div(clamped, 127.0)
        checked += 1
        if by_reciprocal != by_division:
            differing += 1
            if len(examples) < 5:
                examples.append({"absmax": clamped, "multiplyByReciprocal": by_reciprocal,
                                 "divideBy127": by_division})
    return {"checked": checked,
            "multiplyByReciprocalVsDivideBy127Differing": differing,
            "fixtureUses": "absmax_clamped * f32(1/127) -- the compile-time constant form "
                           "written in int8_linear.cu quantize_int8_rowwise_convrot_kernel",
            "examples": examples}


def verify_source_constant_forms(reference_dir: Path) -> dict:
    """Assert the replayed constants actually appear in the hashed CUDA source text."""
    linear = (reference_dir / "comfy_kitchen/backends/cuda/ops/int8_linear.cu").read_text(
        encoding="utf-8")
    gemm = (reference_dir / "comfy_kitchen/backends/cuda/ops/w4a8_gemm.cu").read_text(
        encoding="utf-8")
    cutlass = (reference_dir / "comfy_kitchen/backends/cuda/ops/cutlass_gemm_int8.cu").read_text(
        encoding="utf-8")
    checks = {
        "int8_linear.h4_row_dot": "x0 + x1 + x2 - x3" in linear,
        "int8_linear.stage_strides": "const int s = (stage == 0) ? 1 : (stage == 1) ? 4 : (stage == 2) ? 16 : 64" in linear,
        "int8_linear.stage_scale": "const float v = 0.5f * h4_row_dot(" in linear,
        "int8_linear.row_scale": "(1.0f / 127.0f)" in linear,
        "int8_linear.row_scale_floor": "1.0e-30f" in linear,
        "int8_linear.bf16_finite_max": "3.38953139e38f" in linear,
        "int8_linear.activation_clamp": "fminf(127.0f, fmaxf(-128.0f, quantized))" in linear,
        "int8_linear.nearbyintf": "quantized = nearbyintf(scaled);" in linear,
        "int8_linear.round_before_clamp": (
            linear.index("quantized = nearbyintf(scaled);")
            < linear.index("fminf(127.0f, fmaxf(-128.0f, quantized))")),
        "int8_linear.quant_div_bf16": "quant_div_float_to_float" in linear,
        "int8_linear.epilogue_fold": "static_cast<float>(input[idx]) * x_scales[row] * weight_scale" in linear,
        "w4a8_gemm.low_nibble_even": "const unsigned c0 = byte & 0xF, c1 = (byte >> 4) & 0xF" in gemm,
        "w4a8_gemm.level": "const float v0 = codebook ? cb[c0] : (static_cast<float>(c0) - 8.0f)" in gemm,
        "w4a8_gemm.decode_clamp": "max(-127, min(127, __float2int_rn(v0 * s)))" in gemm,
        "w4a8_gemm.fp8_load": "__nv_cvt_fp8_to_halfraw(v, __NV_E4M3)" in gemm,
        "cutlass.acc_int32": "using ElementAcc = int32_t; using ElementCompute = float;" in cutlass,
        "cutlass.mul0_round": "FloatRoundStyle::round_to_nearest" in cutlass,
    }
    missing = sorted(name for name, ok in checks.items() if not ok)
    if missing:
        raise ValueError(f"hashed CUDA source does not contain replayed forms: {missing}")
    return {"checks": len(checks), "missing": missing}


# --------------------------------------------------------------------------------------
# Fixture assembly
# --------------------------------------------------------------------------------------


def build_cases(ns) -> list[Operands]:
    cases: list[Operands] = []

    # 1 -- cross-tile shape at n=80 (Candle 64-row tile) and k=1024 (4 ConvRot groups,
    #     4 MLX 256-column tiles, 16-group boundary crossed), fully non-uniform, biased.
    rng = SplitMix64(0x26685_0001)
    weight = synthetic_weight(rng, 80, 1024, row_span=3.0, group_span=1.5)
    # Adjacent Candle-tile boundary columns 63|64 get very different scales.
    for j in range(1024):
        weight[63][j] *= 1e-3
        weight[64][j] *= 1e3
    x = []
    for r in range(2):
        row = []
        block_amp = [1.0, 1e-3, 1e-6, 1e2] if r == 0 else [1e-2, 1e3, 1e-5, 1.0]
        for k in range(1024):
            row.append(block_amp[k // 256] * rng.approx_gaussian())
        x.append(row)
    bias = [rng.uniform(-2.0, 2.0) for _ in range(80)]
    cases.append(operands_from_pinned_quantize(
        ns, "native_tiles_n80_k1024",
        "crosses the Candle 64-output-row tile and the MLX 256-column tile; 4 ConvRot "
        "groups with very different maxima (global row scale); adjacent chunk columns "
        "with very different s_rel/codebook usage; non-uniform codes/codebook/channels/"
        "relative scales; biased",
        "weight row 63/64 are scaled 1e-3/1e3 so the two sides of the native output-row "
        "tile boundary carry very different s_rel and s_channel",
        weight, x, bias))

    # 2 -- second required cross-tile shape at k=512 (2 ConvRot groups, 2 MLX column tiles).
    rng = SplitMix64(0x26685_0002)
    weight = synthetic_weight(rng, 80, 512, row_span=2.5, group_span=1.25)
    for j in range(512):
        weight[79][j] *= 5e2
    x = [[(1.0 if (k // 256) == 0 else 1e-4) * rng.approx_gaussian() for k in range(512)]]
    bias = [rng.uniform(-1.0, 1.0) for _ in range(80)]
    cases.append(operands_from_pinned_quantize(
        ns, "native_tiles_n80_k512",
        "second cross-tile shape (k=512): 2 ConvRot groups and the 16-group boundary, "
        "n=80 across the Candle 64-row tile; biased path",
        "", weight, x, bias))

    # 3 -- global row scale across ConvRot groups with very different maxima; no bias.
    rng = SplitMix64(0x26685_0003)
    weight = synthetic_weight(rng, 16, 512, row_span=1.5, group_span=1.0)
    x = []
    for r in range(2):
        row = []
        for k in range(512):
            amp = [1.0, 1e-6][k // 256] if r == 0 else [1e-7, 3.0][k // 256]
            row.append(amp * rng.approx_gaussian())
        x.append(row)
    cases.append(operands_from_pinned_quantize(
        ns, "convrot_global_row_scale",
        "two ConvRot groups whose rotated maxima differ by ~1e6: the activation row scale "
        "must be global over the complete row, never per group/tile; bias is null here",
        "bias=null exercises the no-bias epilogue branch",
        weight, x, None))

    # 4 -- FP8 e4m3 relative-scale extremes, NON-NEGATIVE only: the published contract's
    #      `W4A8Int8::validate_s_rel` refuses a negative or NaN relative scale (the converter emits
    #      only non-negative s_rel), so a successful-forward case must stay inside that domain.
    #      The refused encodings live in `refusedOperands` instead.
    rng = SplitMix64(0x26685_0004)
    k, n = 256, 16
    codebook = [0.0, 1.0, -1.0, 2.5, -2.5, 3.0, -3.0, 7.75, -7.75, 11.5, -11.5,
                31.25, -31.25, 63.75, -63.75, 127.0]
    srel_bits = []
    for i in range(n):
        row = []
        for g in range(k // GROUP_SIZE):
            row.append([0x00, 0x01, 0x7E, 0x38, 0x44, 0x00, 0x7E, 0x38][(i + g) % 8])
        srel_bits.append(row)
    q_u = [[rng.int_in(0, 15) for _ in range(k)] for _ in range(n)]
    s_channel = [10.0 ** rng.uniform(-3, 3) for _ in range(n)]
    x = [[rng.approx_gaussian() for _ in range(k)]]
    bias = [rng.uniform(-0.5, 0.5) for _ in range(n)]
    cases.append(operands_constructed(
        "fp8_srel_extremes",
        "relative-scale FP8 e4m3 extremes the production contract accepts: +0 (0x00), min "
        "subnormal 2^-9 (0x01), max finite 448 (0x7E) and exact 1.0/3.0 (0x38/0x44); drives "
        "weight-decode saturation at +127 and the zero/subnormal scale paths. Negative and NaN "
        "s_rel bytes are production-invalid and are pinned by `refusedOperands`, never here",
        "sRelBits cycle 0x00,0x01,0x7E,0x38,0x44,0x00,0x7E,0x38 across rows and groups",
        x, q_u, srel_bits, s_channel, codebook, bias))

    # 5 -- half-even weight ties and asymmetric [-127,127] saturation.
    k, n = 256, 16
    codebook = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 127.5, -127.5, 128.0, -128.0,
                200.0, -200.0, 3.5, -3.5, 255.5, -255.5]
    srel_bits = [[f32_to_fp8_e4m3fn_bits([1.0, 0.5, 2.0, 3.0][g % 4]) for g in range(k // GROUP_SIZE)]
                 for _ in range(n)]
    q_u = [[(j + i * 3) % 16 for j in range(k)] for i in range(n)]
    s_channel = [float(i + 1) / 64.0 for i in range(n)]
    x = [[rng.approx_gaussian() * (1.0 if (j // 256) == 0 else 1e-5) for j in range(k)]]
    bias = [rng.uniform(-0.25, 0.25) for _ in range(n)]
    cases.append(operands_constructed(
        "weight_half_even_ties",
        "weight decode ties at exactly n.5 (0.5,1.5,2.5,3.5 and their negatives) must round "
        "half to EVEN, and the asymmetric integer clamp [-127,127] saturates 127.5/128/200/255.5 "
        "to +127 and their negatives to -127 (a symmetric [-128,128] clamp would give -128)",
        "codebook entries are exact n.5 values and s_rel is an exact power of two or 3.0, "
        "so every product is an exact tie",
        x, q_u, srel_bits, s_channel, codebook, bias))

    # 6 -- packed nibble order (even column = low nibble).
    k, n = 256, 16
    codebook = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    srel_bits = [[0x38 for _ in range(k // GROUP_SIZE)] for _ in range(n)]
    q_u = []
    for i in range(n):
        row = []
        for j in range(k):
            row.append((i * 5 + 1) % 16 if j % 2 == 0 else (i * 5 + 14) % 16)
        q_u.append(row)
    s_channel = [float(i + 1) / 32.0 for i in range(n)]
    x = [[rng.approx_gaussian() for _ in range(k)]]
    bias = [0.0] * n
    cases.append(operands_constructed(
        "packed_nibble_order",
        "asymmetric codebook with different even/odd column codes: swapping the packed "
        "nibble order (even=high) must change decodedWeightI8 and the output",
        "", x, q_u, srel_bits, s_channel, codebook, bias))

    # 7 -- complete-K chunk cancellation: the two 512-column halves cancel exactly.
    rng = SplitMix64(0x26685_0007)
    k, n = 1024, 16
    codebook = [0.0, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0, 16.0, -16.0,
                0.25, -0.25, 0.5, -0.5, 1.5]
    srel_bits = [[0x68 for _ in range(k // GROUP_SIZE)]] * n     # 64.0 exactly
    srel_bits = [list(row) for row in srel_bits]
    q_u = []
    for i in range(n):
        row = []
        for j in range(k):
            if i == 0:
                row.append(1 if j < 512 else 2)          # +64 / -64
            elif i == 1:
                row.append(1)                            # +64 everywhere
            elif i == 2:
                row.append(2 if j < 512 else 1)          # -64 / +64
            else:
                row.append(1 if (j // 256) % 2 == 0 else 2)
        q_u.append(row)
    s_channel = [1.0] * n
    # x has period 512 -> xq[k] == xq[k+512] exactly, so the two halves cancel in int32.
    base = [rng.approx_gaussian() for _ in range(512)]
    x = [base + base]
    bias = [0.0] * n
    cases.append(operands_constructed(
        "chunk_cancellation_k1024",
        "the int32 dot over the complete K must cancel exactly across the two 512-column "
        "halves while each half contributes a large partial sum; any per-K-chunk scaling, "
        "per-chunk BF16 cast or per-tile requant destroys the exact zero",
        "x is 512-periodic so xq[k] == xq[k+512]; weights are +64/-64 (codebook 1.0/-1.0, "
        "s_rel 64.0) in the two halves",
        x, q_u, srel_bits, s_channel, codebook, bias))

    # 8 -- bias FMA cancellation: separate roundings give exactly 0, a fused FMA does not.
    rng = SplitMix64(0x26685_0008)
    k, n = 256, 16
    codebook = [0.0, 1.0, 2.0, 3.0, 5.5, -1.25, 7.75, -3.5, 11.0, -6.25, 13.5, -9.0,
                16.25, -12.5, 19.5, -16.75]
    srel_bits = [[0x38 for _ in range(k // GROUP_SIZE)]] * n     # s_rel = 1.0
    srel_bits = [list(row) for row in srel_bits]
    q_u = [[3 if j == 0 else 0 for j in range(k)] for _ in range(n)]   # decoded weight 3 / 0
    s_channel = [0.0] * n
    bias = [-1.0] * n
    found = None
    for trial in range(4000):
        amp = 10.0 ** rng.uniform(-2.0, 2.0)
        row = [bf16_rn(amp * rng.approx_gaussian()) for _ in range(k)]
        rotated, codes, scales, _ = quantize_int8_rowwise_convrot_replay([row])
        if codes[0][0] != 1:
            continue
        acc = 3 * codes[0][0]
        p1 = f32_mul(f32(float(acc)), scales[0])
        channel = f32(Fraction(1.0) / Fraction(p1))
        if f32_mul(p1, channel) != 1.0:
            continue
        if math.fma(p1, channel, -1.0) == 0.0:
            continue
        if p1 in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            continue
        found = (row, acc, p1, channel, trial)
        break
    if found is None:
        raise RuntimeError("bias-FMA counterexample search failed")
    row, acc, p1, channel, trial = found
    s_channel = [channel] * n
    cases.append(operands_constructed(
        "bias_fma_cancellation",
        "constant positive int32 accumulator with p1=f32(acc*xs), s_channel=f32(1/p1) and "
        "bias=-1.0 (BF16-exact): the ordered F32 chain ((acc*xs)*s_channel)+bias is exactly "
        "0.0 while a fused mul+bias FMA retains a nonzero residual",
        f"searched {trial + 1} deterministic synthetic rows; p1={p1!r}, s_channel={channel!r}, "
        f"acc={acc} constant across all {n} output rows",
        [row], q_u, srel_bits, s_channel, codebook, bias))

    # 9 -- epilogue store rounding at an EXACT BF16 halfway (ties to EVEN, both tie
    #      directions), plus reassociation / early-BF16-cast flips of that rounding.  The
    #      halfways sit in the 2**20 binade, where one binary32 ulp is small relative to the
    #      BF16 ulp, so a one-ulp perturbation of the rounded product crosses the halfway and
    #      the wrong ordering becomes detectable after the single BF16 round at the store.
    rng = SplitMix64(0x26685_0009)
    k, n = 256, 16
    codebook = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    srel_bits = [[0x38 for _ in range(k // GROUP_SIZE)]] * n
    srel_bits = [list(row) for row in srel_bits]
    q_u = [[3 if j == 0 else 0 for j in range(k)] for _ in range(n)]
    # 2**20 + 4096 is the halfway between BF16 0x4980 and 0x4981 -> ties to even gives
    # 0x4980 (2**20).  2**20 + 12288 is the halfway between 0x4981 and 0x4982 -> ties to
    # even gives 0x4982 (2**20 + 8192).  Both tie directions are therefore pinned.
    specs = [
        (1052672.0, 1024.0, None),
        (1060864.0, -2048.0, None),
        (1052672.0, 1024.0, "reassociatedScales"),
        (1060864.0, 16384.0, "earlyBf16Cast"),
    ]

    def epilogue_variants(acc_v, x_scale, w_scale, bias_v):
        p1 = f32_mul(f32(float(acc_v)), x_scale)
        p2 = f32_mul(p1, w_scale)
        return {
            "correct": bf16_rn(f32_add(p2, bias_v)),
            "reassociatedScales": bf16_rn(f32_add(
                f32_mul(f32(float(acc_v)), f32_mul(x_scale, w_scale)), bias_v)),
            "earlyBf16Cast": bf16_rn(f32_add(f32_mul(bf16_rn(p1), w_scale), bias_v)),
            "biasAddedInBf16": bf16_rn(f32_add(bf16_rn(p2), bf16_rn(bias_v))),
            "fusedBiasFma": bf16_rn(math.fma(p1, w_scale, bias_v)),
            "cpuPortableEager": bf16_rn(f32_add(
                bf16_rn(f32_mul(f32(float(acc_v)), f32_mul(x_scale, w_scale))), bf16_rn(bias_v))),
        }

    def find_halfway_scale(acc_v, x_scale, target, bias_v, required):
        """binary32 channel scale with ((acc*xs)*s_channel)+bias landing exactly on the
        halfway, and `required` (if any) landing on the other side of it."""
        p1 = f32_mul(f32(float(acc_v)), x_scale)
        if p1 == 0.0:
            return None
        want = f32(target - bias_v)
        if f32_add(want, bias_v) != target:
            return None
        base = f32_bits(f32(want / p1))
        for delta in range(-30000, 30001):
            bits = base + delta
            if not (0 < bits < 0x7F800000):
                continue
            w_scale = f32_from_bits(bits)
            if w_scale == 0.0 or f32_mul(p1, w_scale) != want:
                continue
            values = epilogue_variants(acc_v, x_scale, w_scale, bias_v)
            if values["correct"] != bf16_rn(target):
                continue
            if required is None or values[required] != values["correct"]:
                return w_scale
        return None

    chosen, row, acc_used, trial = None, None, 0, 0
    for trial in range(1500):
        amp = 10.0 ** rng.uniform(-3.0, 1.0)
        row = [bf16_rn(amp * rng.approx_gaussian()) for _ in range(k)]
        _, codes, scales, _ = quantize_int8_rowwise_convrot_replay([row])
        acc_used = 3 * codes[0][0]
        if acc_used == 0:
            continue
        found = [find_halfway_scale(acc_used, scales[0], t, b, req) for t, b, req in specs]
        if all(w is not None for w in found):
            chosen = found
            break
    if chosen is None:
        raise RuntimeError("bf16 halfway construction failed")
    # 16 output rows = the four halfway specs repeated, so every row (not just the first four)
    # sits on a tie and the CUDA lane's 16-wide output alignment is satisfiable.
    s_channel = list(chosen) * (n // len(specs))
    bias = [b for _t, b, _req in specs] * (n // len(specs))
    cases.append(operands_constructed(
        "bf16_halfway_and_early_cast",
        "the final store lands exactly on a BF16 halfway in the 2**20 binade: 2**20+4096 must "
        "round to 2**20 and 2**20+12288 must round to 2**20+8192 (round to nearest, ties to "
        "EVEN, a single rounding at the store, both tie directions). Detects half-up/half-away "
        "rounding, an early BF16 cast of acc*xs, and reassociating the scale product",
        f"searched {trial + 1} deterministic synthetic rows plus a window of binary32 channel "
        "scales so the ordered chain lands exactly on the halfway while the wrong ordering "
        f"crosses it; acc={acc_used} constant across all {n} output rows",
        [row], q_u, srel_bits, s_channel, codebook, bias))

    # 10 -- activation quantization vs a dense floating fallback, plus xq == -128 saturation.
    rng = SplitMix64(0x26685_0010)
    weight = synthetic_weight(rng, 16, 512, row_span=1.0, group_span=0.75)
    # Weight columns in the second ConvRot block carry the dominant output contribution, so
    # the int8 path (which quantizes that block's tiny activation to zero) must lose it while
    # a dense floating fallback would keep it.  This makes the discriminator material.
    for i in range(16):
        for j in range(CONVROT_GROUP, 512):
            weight[i][j] *= 1e3
    # Row 0 is searched so the asymmetric activation clamp [-128,127] is actually exercised
    # (a symmetric [-127,127] clamp would change these codes) over a wide quantization span.
    chosen, trial = None, 0
    for trial in range(20000):
        amp = 10.0 ** rng.uniform(-2.0, 2.0)
        row0 = [bf16_rn(amp * rng.approx_gaussian()) for _ in range(512)]
        _, codes, _, _ = quantize_int8_rowwise_convrot_replay([row0])
        if min(codes[0]) == -128 and max(codes[0]) - min(codes[0]) >= 200:
            chosen = row0
            break
    if chosen is None:
        raise RuntimeError("activation -128 saturation search failed")
    # Row 1 is a fixed construction: one dominant outlier plus many tiny values, so the int8
    # path discards information the dense floating path would keep.
    row1 = [bf16_rn(2.0 ** -14 * rng.approx_gaussian()) for _ in range(512)]
    row1[11] = bf16_rn(2.0 ** 6 * rng.approx_gaussian())
    bias = [rng.uniform(-1.0, 1.0) for _ in range(16)]
    cases.append(operands_from_pinned_quantize(
        ns, "activation_quant_vs_dense",
        "activation quantized ONCE per row versus a dense floating fallback / no activation "
        "quantization: the int8 path must differ materially from both. Row 0 also drives the "
        "asymmetric activation clamp [-128,127] (xq reaches -128); row 1 has one dominant "
        "outlier so quantization discards the small values",
        f"row 0 searched over {trial + 1} deterministic synthetic rows to reach xq == -128 with a "
        ">=200 quantization span; row 1 is a fixed outlier construction",
        weight, [chosen, row1], bias))

    # 11 -- MLX output-column tile boundary: N = 320 > the 256-column weight tile, so the forward
    #       must produce a full 256-wide tile plus a 64-wide remainder tile, and Candle's 64-row
    #       tile five times. The rows either side of the 256 boundary carry very different weight
    #       magnitudes, so a tile-join that reordered/dropped columns cannot pass.
    rng = SplitMix64(0x26685_0011)
    weight = synthetic_weight(rng, 320, 512, row_span=2.0, group_span=1.0)
    for j in range(512):
        weight[255][j] *= 2e2
        weight[256][j] *= 2e-3
    x = [[[1.0, 1e-3][r] * rng.approx_gaussian() for _ in range(512)] for r in range(2)]
    bias = [rng.uniform(-1.5, 1.5) for _ in range(320)]
    cases.append(operands_from_pinned_quantize(
        ns, "mlx_tile_n320_k512",
        "N=320 crosses the MLX 256-output-column weight tile boundary (a full 256-wide tile and "
        "a 64-wide remainder) and the Candle 64-row tile five times; output rows 255/256 are "
        "scaled 2e2/2e-3 so the tile join is numerically load-bearing; 2 ConvRot groups, biased",
        "the second MLX column tile is a partial 64-wide tile",
        weight, x, bias))

    # 12 -- F32 bias precision: cutlass_gemm_int8.cu adds the STORED bias (`bias.float()`) in F32
    #       and rounds ONCE at the store. A native epilogue that first casts the bias to the
    #       activation dtype loses the bias's low bits — invisible unless the pre-bias product lands
    #       near the rounded bias, so this row is searched for `(acc*xs)*s_channel == 1.0` exactly
    #       with bias -0.999f (bf16(-0.999f) == -1.0f), making the two results 0.001 vs 0.0.
    rng = SplitMix64(0x26685_0012)
    k, n = 256, 16
    codebook = [0.0, 1.0, 2.0, 3.0, 5.5, -1.25, 7.75, -3.5, 11.0, -6.25, 13.5, -9.0,
                16.25, -12.5, 19.5, -16.75]
    srel_bits = [[0x38 for _ in range(k // GROUP_SIZE)] for _ in range(n)]
    q_u = [[3 if j == 0 else 0 for j in range(k)] for _ in range(n)]
    bias_value = f32(-0.999)
    if bf16_rn(bias_value) != -1.0:
        raise RuntimeError("the bias-precision case needs a bias that rounds to -1.0 in BF16")
    chosen, chosen_channel, trial = None, None, 0
    for trial in range(8000):
        amp = 10.0 ** rng.uniform(-2.0, 2.0)
        row = [bf16_rn(amp * rng.approx_gaussian()) for _ in range(k)]
        _, codes, scales, _ = quantize_int8_rowwise_convrot_replay([row])
        if codes[0][0] == 0:
            continue
        acc = 3 * codes[0][0]
        p1 = f32_mul(f32(float(acc)), scales[0])
        if p1 <= 0.0:
            continue
        channel = f32(Fraction(1.0) / Fraction(p1))
        if f32_mul(p1, channel) != 1.0:
            continue
        right = bf16_rn(f32_add(1.0, bias_value))
        wrong = bf16_rn(f32_add(1.0, bf16_rn(bias_value)))
        if right == wrong:
            continue
        chosen, chosen_channel = row, channel
        break
    if chosen is None:
        raise RuntimeError("bias F32-precision counterexample search failed")
    cases.append(operands_constructed(
        "bias_f32_precision",
        "the epilogue must add the stored F32 bias and round once at the store: this row lands "
        "the pre-bias product on exactly 1.0 with bias -0.999f, so the reference stores ~0.001 "
        "while an epilogue that first casts the bias to BF16 (-1.0) stores exactly 0.0",
        f"searched {trial + 1} deterministic synthetic rows for (acc*xs)*s_channel == 1.0 exactly; "
        f"bias={bias_value!r}",
        [chosen], q_u, srel_bits, [chosen_channel] * n, codebook, [bias_value] * n))

    return cases


def build_refusals(cases: list[Operands]) -> list[dict]:
    """Operand sets the published contract must REFUSE, with the rule that refuses them.

    The production loader (`W4A8Int8::validate_s_rel` / `validate_s_channel`) refuses a negative or
    NaN relative scale (the converter emits only non-negative `s_rel`) and a non-positive channel
    scale. These are the same bytes the old success case wrongly carried: a successful-forward case
    cannot contain them, so they are covered as explicit expected refusals instead.
    """
    base = next(op for op in cases if op.case_id == "packed_nibble_order")
    groups = base.k // GROUP_SIZE

    def variant(case_id, purpose, srel_row0, s_channel, rule, expected_error):
        srel = [list(row) for row in base.srel_bits]
        srel[0] = list(srel_row0)
        return {
            "id": case_id,
            "purpose": purpose,
            "rule": rule,
            "expectedError": expected_error,
            "m": base.m, "n": base.n, "k": base.k,
            "groupSize": GROUP_SIZE,
            "convrotGroup": CONVROT_GROUP,
            "inputDtype": "BF16",
            "outputDtype": "BF16",
            "x": [v for row in base.x_bf16 for v in row],
            "packedWeight": [b for row in base.packed for b in row],
            "sRelBits": [b for row in srel for b in row],
            "sChannel": s_channel,
            "codebook": base.codebook,
            "bias": base.bias,
        }

    negative = [0x81 if g % 2 == 0 else 0xFE for g in range(groups)]
    nan = [0x7F] * groups
    zero_channel = list(base.s_channel)
    zero_channel[0] = 0.0
    return [
        variant("refused_negative_s_rel",
                "E4M3FN relative-scale bytes 0x81 (-2^-9) and 0xFE (-448): the converter only "
                "emits non-negative s_rel, so a negative scale is not this dialect",
                negative, list(base.s_channel), "s_rel must be non-negative finite E4M3FN",
                "W4A8 weight_s_rel must be non-negative finite E4M3FN"),
        variant("refused_nan_s_rel",
                "E4M3FN NaN relative-scale byte 0x7F: there is no infinity and a NaN scale has no "
                "decode",
                nan, list(base.s_channel), "s_rel must be non-negative finite E4M3FN",
                "W4A8 weight_s_rel must be non-negative finite E4M3FN"),
        variant("refused_zero_s_channel",
                "s_channel byte 0.0: the per-output-channel epilogue scale must be finite positive, "
                "otherwise the whole output row is annihilated",
                [0x38] * groups, zero_channel, "s_channel must be finite positive F32",
                "W4A8 weight_s_channel must be finite positive F32"),
    ]


def build_mixed_layer(convrot_fixture: Path, w4a8_case_id: str) -> dict:
    """One published operand set carrying BOTH native formats in adjacent layers.

    Each half stays pinned to its own external authority: the W4A8 half is `w4a8_case_id` above
    (comfy-kitchen PR#90 CPU kernel replay), and the ConvRot INT8 half reuses the exact pinned
    activation — input, int8 codes and row scale — of the S4 fixture (`h3_convrot_int8.json`,
    kitchen `f8b7e97a`), so the mixed case adds routing and an int8 GEMM/epilogue numeric check
    rather than a second interpretation of the activation arithmetic. Its weights/scales/bias are
    owned synthetic constants; the expected output is the pinned cutlass epilogue
    `((acc*xs)*s_channel)+bias` in F32 over the pinned codes (the `f32` output dtype stores without
    a BF16 round).
    """
    raw = convrot_fixture.read_bytes()
    fx = json.loads(raw)
    index = 3
    case = fx["cases"][index]
    if case["group"] != CONVROT_GROUP or case["dtype"] != "f32":
        raise ValueError("mixed-layer authority case must be the group-256 f32 case")
    m, n, k = case["m"], case["n"], case["k"]
    codes = [[((i * 37 + j * 11) % 255) - 127 for j in range(k)] for i in range(n)]
    weight_scale = [f32(0.5 + 0.25 * i) for i in range(n)]
    bias = [f32(0.125 * (i - 8)) for i in range(n)]
    xs = case["activation_scales"]
    act = case["activation_codes"]
    accumulators = [[sum(int(act[r * k + j]) * codes[c][j] for j in range(k))
                     for c in range(n)] for r in range(m)]
    expected = []
    for r in range(m):
        for c in range(n):
            p = f32_mul(f32(float(accumulators[r][c])), xs[r])
            expected.append(f32_add(f32_mul(p, weight_scale[c]), bias[c]))
    # Numeric materiality: the pinned activation's row 0 is the all-zero row (bias passthrough at
    # the 1e-30 scale floor), so at least one other row must carry a real accumulator and its
    # output must differ from the bias-only result for every column.
    if all(acc == 0 for row in accumulators for acc in row):
        raise AssertionError("mixed-layer accumulator is identically zero")
    material = sum(1 for r in range(m) for c in range(n)
                   if accumulators[r][c] != 0 and expected[r * n + c] != bias[c])
    if material < n:
        raise AssertionError("mixed-layer expected output is bias-only for too many columns")
    return {
        "purpose": "one operand set carrying a packed W4A8 layer and a ConvRot INT8 layer: the "
                   "loader must keep each in its own format (never densify one into the other) and "
                   "each forward must reproduce its own pinned external arithmetic",
        "w4a8Case": w4a8_case_id,
        "convrot": {
            "authority": {
                "fixture": "tests/fixtures/h3_convrot_int8.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "caseIndex": index,
                "kitchenRevision": fx["kitchen_revision"],
                "referenceCuda": fx["reference_cuda"],
            },
            "m": m, "n": n, "k": k, "group": CONVROT_GROUP, "dtype": "F32",
            "note": "row 0 of the pinned activation is its all-zero row (every code 0, scale at the "
                    "1e-30 floor), so it pins the bias passthrough; rows 1..4 carry the int32 "
                    "accumulator and the ((acc*xs)*s_channel)+bias chain, stored without a BF16 "
                    "round because the output dtype is F32",
            "input": case["input"],
            "activationCodes": act,
            "activationScales": xs,
            "weight": [v for row in codes for v in row],
            "weightScale": weight_scale,
            "bias": bias,
            "expected": expected,
        },
    }


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="directory holding the hash-pinned comfy-kitchen source text")
    parser.add_argument("--convrot-fixture", type=Path,
                        default=Path("tests/fixtures/h3_convrot_int8.json"),
                        help="the S4 ConvRot INT8 fixture whose pinned activation the mixed-layer "
                             "case reuses (sha256 recorded in the output)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True,
                        help="audit receipt (kept outside the checkout)")
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    if torch.cuda.is_available():
        raise RuntimeError("this oracle is CPU-only; CUDA must be hidden in the child environment")

    source_records = verify_pinned_sources(args.reference_dir)
    fp8_check = verify_fp8_decode()
    scale_check = verify_scale_constant()
    form_check = verify_source_constant_forms(args.reference_dir)
    ns = build_pinned_layout(args.reference_dir)

    # Literal QuantizedTensor + F.linear W4A8 dispatch probe (attempted, recorded truthfully).
    spec = importlib.util.find_spec("comfy_kitchen")
    literal = {
        "attempted": True,
        "comfyKitchenInstalled": spec is not None,
        "probeError": None if spec is not None else "ModuleNotFoundError: No module named 'comfy_kitchen'",
        "cudaExtension": "unavailable: torch.version.cuda is None, torch.cuda.is_available() is False",
        "cpuPortablePathWouldBe": (
            "AsymW4A8Int8Layout -> _w4a8_int8_matmul_portable -> registry 'eager' "
            "comfy_kitchen.backends.eager.w4a8_int8.w4a8_int8_linear -> "
            "comfy_kitchen.backends.eager.quantization.int8_linear"),
        "cpuPortablePathMatchesHashedCudaArithmetic": False,
        "cpuPortablePathDeviations": [
            "activation rotated by the Python _build_hadamard/_rotate_activation matmul, not the hashed CUDA FHT",
            "row scale computed as abs_max.float()/127.0 (division) then x/scale in the input dtype, not "
            "quant_div_float_to_float (BF16-rounded operands and BF16-rounded quotient)",
            "epilogue is acc * (x_scale * weight_scale) cast to out_dtype, THEN + bias added in out_dtype -- "
            "not ((acc*xs)*s_channel) + bias in F32 with one rounding at the store",
        ],
        "reason": (
            "comfy-kitchen is not installed in the qualified CPU venv and installing into that user "
            "environment is forbidden by the epic contract; even if it were, the CPU portable path is not "
            "the hashed CUDA arithmetic. Golden values therefore come from the explicit CPU replay of the "
            "hashed CUDA kernels, labelled 'cpu-kernel-replay', with no upstream CUDA claim."),
    }

    cases = build_cases(ns)
    refusals = build_refusals(cases)
    mixed = build_mixed_layer(args.convrot_fixture, "native_tiles_n80_k512")

    counters = {
        "pinnedLayoutEncodeCalls": 0,
        "pinnedLayoutDecodeCrossChecks": 0,
        "activationQuantKernelCalls": 0,
        "weightDecodeKernelCalls": 0,
        "intAccumulateDots": 0,
        "intAccumulateTerms": 0,
        "epilogueF32Ops": 0,
        "floatMatmulCalls": 0,
        "denseDequantizeFallbacks": 0,
    }
    discriminators: dict[str, list[str]] = {}
    counterfactual_summary = []
    out_cases = []

    def note_discriminator(name: str, case_id: str):
        discriminators.setdefault(name, [])
        if case_id not in discriminators[name]:
            discriminators[name].append(case_id)

    for op in cases:
        rotated, xq, xs, group_absmax = quantize_int8_rowwise_convrot_replay(op.x_bf16)
        counters["activationQuantKernelCalls"] += 1
        decoded = decode_weight_replay(op.packed, op.srel_bits, op.codebook, op.k)
        counters["weightDecodeKernelCalls"] += 1
        acc, partials, terms = int_accumulate_replay(xq, decoded)
        counters["intAccumulateDots"] += op.m * op.n
        counters["intAccumulateTerms"] += terms
        counters["epilogueF32Ops"] += op.m * op.n * (3 if op.bias is not None else 2)
        out = [[epilogue_replay(acc[r][c], xs[r], op.s_channel[c],
                                op.bias[c] if op.bias else None) for c in range(op.n)]
               for r in range(op.m)]

        # --- pinned-layout cross-checks (the pinned Python decode must agree with the replay) ---
        if op.params_origin == "pinned-AsymW4A8Int8Layout.quantize":
            counters["pinnedLayoutEncodeCalls"] += 1
        qpacked = torch.tensor(op.packed, dtype=torch.uint8).view(torch.int8)
        params = ns["AsymW4A8Int8Layout"].Params(
            scale=torch.tensor(op.srel_bits, dtype=torch.uint8).view(torch.float8_e4m3fn),
            s_channel=torch.tensor(op.s_channel, dtype=torch.float32),
            correction=None,
            codebook=torch.tensor(op.codebook, dtype=torch.float32) if op.codebook else None,
            orig_dtype=torch.bfloat16,
            orig_shape=(op.n, op.k),
            group_size=GROUP_SIZE,
            convrot_groupsize=CONVROT_GROUP,
        )
        pinned_w = ns["AsymW4A8Int8Layout"].dequantize(qpacked, params).float()
        replay_w = torch.tensor([[v * op.s_channel[c] for v in decoded[c]] for c in range(op.n)],
                                dtype=torch.float32)
        replay_unrotated = ns["_convrot_rotate"](replay_w.to(torch.bfloat16), CONVROT_GROUP).float()
        if not torch.equal(pinned_w, replay_unrotated):
            diff = (pinned_w - replay_unrotated).abs().max().item()
            raise AssertionError(f"{op.case_id}: pinned decode disagrees with kernel replay (max {diff})")
        counters["pinnedLayoutDecodeCrossChecks"] += 1

        # --- counterfactual discriminators ---
        def cf(name, values, description):
            differs = any(values[r][c] != out[r][c] for r in range(op.m) for c in range(op.n))
            if differs:
                note_discriminator(name, op.case_id)
            per_row = []
            for r in range(op.m):
                diff = max(abs(values[r][c] - out[r][c]) for c in range(op.n))
                scale = max(max(abs(out[r][c]) for c in range(op.n)), 1e-30)
                per_row.append({"row": r, "maxAbsDiff": diff,
                                "expectedRowScale": scale, "relativeDiff": diff / scale})
            counterfactual_summary.append({
                "case": op.case_id, "counterfactual": name, "description": description,
                "differsFromExpected": differs,
                "maxAbsDiff": max(item["maxAbsDiff"] for item in per_row),
                "maxRelativeDiff": max(item["relativeDiff"] for item in per_row),
                "perRow": per_row,
            })

        cf("packedNibbleOrderSwapped",
           [[epilogue_replay(sum(int(xq[r][k]) * v for k, v in enumerate(row)), xs[r],
                             op.s_channel[c], op.bias[c] if op.bias else None) for c, row in enumerate(cf_nibble_swapped(op))]
            for r in range(op.m)],
           "decode with even column = HIGH nibble instead of LOW")
        cf("denseFloatingFallback", cf_dense_float(op, ns, op.x_bf16),
           "F.linear over the dequantized original-basis weight and unrotated f32 input")
        cf("noActivationQuantization",
           cf_no_activation_quant(rotated, acc, xs, decoded, op.s_channel, op.bias, op.m, op.n),
           "rotated activation used unquantized against the decoded int8 weight")
        per_tile = cf_per_tile_requant(op.x_bf16, decoded, op.s_channel, op.bias, op.m, op.n, op.k)
        cf("perTileRequant", per_tile,
           "activation re-quantized per 256-column ConvRot tile instead of once per row")
        chunked = [[epilogue_replay(acc[r][c], xs[r], op.s_channel[c] * (2.0 if c % 2 else 0.5),
                                   op.bias[c] if op.bias else None) for c in range(op.n)]
                   for r in range(op.m)]
        cf("perChunkWeightScale", chunked,
           "per output-column chunk weight rescale (each chunk renormalized to its own max)")
        cf("weightClampSymmetric128",
           [[epilogue_replay(sum(int(xq[r][k]) * v for k, v in enumerate(row)), xs[r],
                             op.s_channel[c], op.bias[c] if op.bias else None)
             for c, row in enumerate(cf_weight_clamp_sym128(op))] for r in range(op.m)],
           "weight decode clamps [-128,128] instead of the kernel's [-127,127]")
        cf("weightRoundHalfAway",
           [[epilogue_replay(sum(int(xq[r][k]) * v for k, v in enumerate(row)), xs[r],
                             op.s_channel[c], op.bias[c] if op.bias else None)
             for c, row in enumerate(cf_weight_round_half_up(op))] for r in range(op.m)],
           "weight decode rounds half away from zero instead of half to even")
        cf("fusedBiasFma",
           [[bf16_rn(math.fma(f32_mul(f32(float(acc[r][c])), xs[r]), op.s_channel[c],
                              op.bias[c] if op.bias else 0.0)) for c in range(op.n)]
            for r in range(op.m)],
           "second multiply fused into the bias add (FMA)")
        cf("reassociatedScales",
           [[bf16_rn(f32_add(f32_mul(f32(float(acc[r][c])), f32_mul(xs[r], op.s_channel[c])),
                             op.bias[c] if op.bias else 0.0)) for c in range(op.n)]
            for r in range(op.m)],
           "acc * (xs * s_channel) instead of (acc * xs) * s_channel")
        cf("earlyBf16Cast",
           [[bf16_rn(f32_add(f32_mul(bf16_rn(f32_mul(f32(float(acc[r][c])), xs[r])), op.s_channel[c]),
                             op.bias[c] if op.bias else 0.0)) for c in range(op.n)]
            for r in range(op.m)],
           "acc*xs cast to BF16 before the s_channel multiply")
        cf("biasRoundedToActivationDtype",
           [[bf16_rn(f32_add(f32_mul(f32_mul(f32(float(acc[r][c])), xs[r]), op.s_channel[c]),
                              bf16_rn(op.bias[c]) if op.bias else 0.0)) for c in range(op.n)]
            for r in range(op.m)],
           "the stored F32 bias cast to the activation dtype (BF16) before the F32 epilogue add")
        cf("biasAddedInBf16",
           [[bf16_rn(f32_add(f32_mul(f32_mul(f32(float(acc[r][c])), xs[r]), op.s_channel[c]), 0.0))
             if op.bias is None else
             bf16_rn(f32_add(bf16_rn(f32_mul(f32_mul(f32(float(acc[r][c])), xs[r]), op.s_channel[c])),
                             bf16_rn(op.bias[c]))) for c in range(op.n)]
            for r in range(op.m)],
           "bias added after the BF16 cast, in BF16 (the eager CPU portable epilogue)")
        cf("cpuPortableEager",
           [[bf16_rn(f32_add(bf16_rn(f32_mul(f32(float(acc[r][c])), f32_mul(xs[r], op.s_channel[c]))),
                             bf16_rn(op.bias[c]) if op.bias else 0.0)) for c in range(op.n)]
            for r in range(op.m)],
           "the complete eager CPU portable epilogue: acc * (xs*s_channel) -> BF16, then + bias in BF16")

        row_absmax = [max(abs(v) for v in row) for row in rotated]
        out_cases.append({
            "id": op.case_id,
            "m": op.m, "n": op.n, "k": op.k,
            "groupSize": GROUP_SIZE,
            "convrotGroup": CONVROT_GROUP,
            "inputDtype": "BF16",
            "outputDtype": "BF16",
            "purpose": op.purpose,
            "paramsOrigin": op.params_origin,
            "x": [v for row in op.x_bf16 for v in row],
            "packedWeight": [b for row in op.packed for b in row],
            "sRelBits": [b for row in op.srel_bits for b in row],
            "sChannel": op.s_channel,
            "codebook": op.codebook,
            "bias": op.bias,
            "expected": {
                "xQuantized": [v for row in xq for v in row],
                "xScale": xs,
                "decodedWeightI8": [v for row in decoded for v in row],
                "accumulatorI32": [v for row in acc for v in row],
                "output": [v for row in out for v in row],
                "rotatedF32": [v for row in rotated for v in row],
                "rotGroupAbsmax": [v for row in group_absmax for v in row],
                "rowAbsmax": row_absmax,
                "chunkPartialI32": partials,
            },
        })

    # --- fixture-level invariants a consumer can rely on -----------------------------
    by_id = {c["id"]: c for c in out_cases}
    for required in ("native_tiles_n80_k1024", "native_tiles_n80_k512", "bias_fma_cancellation",
                     "activation_quant_vs_dense", "chunk_cancellation_k1024",
                     "convrot_global_row_scale", "fp8_srel_extremes", "weight_half_even_ties",
                     "packed_nibble_order", "bf16_halfway_and_early_cast", "mlx_tile_n320_k512",
                     "bias_f32_precision"):
        if required not in by_id:
            raise AssertionError(f"missing required case {required}")

    # Every successful forward must be loadable by the CUDA lane, whose INT8 IGEMM requires an
    # output width that is a multiple of 16. A case the CUDA comparison silently skips is a case
    # that asserts nothing, so the whole set is pinned to that geometry instead.
    for case in out_cases:
        if case["n"] % 16:
            raise AssertionError(f"case {case['id']} n={case['n']} is not CUDA-loadable (n % 16)")
    if not any(case["n"] > MLX_COLUMN_TILE for case in out_cases):
        raise AssertionError("no case crosses the MLX 256-output-column tile boundary")

    if len(mixed["convrot"]["expected"]) != mixed["convrot"]["m"] * mixed["convrot"]["n"]:
        raise AssertionError("mixed-layer expected output is not [m, n]")
    if mixed["w4a8Case"] not in by_id:
        raise AssertionError("mixed-layer W4A8 half does not name a fixture case")
    for refusal in refusals:
        if not refusal["rule"] or not refusal["purpose"] or not refusal["expectedError"]:
            raise AssertionError("refusal entry must name the contract rule it exercises")

    tile = by_id["native_tiles_n80_k1024"]
    if not (tile["n"] > CANDLE_OUTPUT_ROW_TILE and tile["k"] > MLX_COLUMN_TILE):
        raise AssertionError("tile-crossing case does not cross the native tile boundaries")
    if len(set(tile["sChannel"])) < 8 or len(set(tile["codebook"])) != 16:
        raise AssertionError("tile-crossing case is not non-uniform")
    if len({tuple(tile["sRelBits"][i * (tile["k"] // GROUP_SIZE):(i + 1) * (tile["k"] // GROUP_SIZE)])
            for i in range(tile["n"])}) < 8:
        raise AssertionError("tile-crossing case s_rel is not non-uniform")

    cancel = by_id["chunk_cancellation_k1024"]
    cancel_acc = cancel["expected"]["accumulatorI32"]
    if cancel_acc[0] != 0 or cancel_acc[2] != 0:
        raise AssertionError("chunk cancellation case does not cancel to exactly zero")
    cancel_partial = cancel["expected"]["chunkPartialI32"]
    width = cancel["k"] // CONVROT_GROUP
    for r in range(cancel["m"]):
        for c in range(cancel["n"]):
            total = sum(cancel_partial[(r * cancel["n"] + c) * width:(r * cancel["n"] + c) * width + width])
            if total != cancel_acc[r * cancel["n"] + c]:
                raise AssertionError("chunk partial sums do not sum to the complete-K accumulator")
    if max(abs(v) for v in cancel_partial[:width]) < 1000:
        raise AssertionError("chunk cancellation partial sums are not large enough to discriminate")

    fma_case = by_id["bias_fma_cancellation"]
    fma_out = fma_case["expected"]["output"]
    if any(v != 0.0 for v in fma_out):
        raise AssertionError("bias FMA cancellation case must produce exactly 0.0")
    fma_residual = 0.0
    for r in range(fma_case["m"]):
        for c in range(fma_case["n"]):
            acc_v = fma_case["expected"]["accumulatorI32"][r * fma_case["n"] + c]
            p1 = f32_mul(f32(float(acc_v)), fma_case["expected"]["xScale"][r])
            if p1 <= 0.0 or p1 in (0.5, 1.0, 2.0, 4.0, 8.0):
                raise AssertionError("bias FMA counterexample needs a positive non-power-of-two p1")
            if f32_add(f32_mul(p1, fma_case["sChannel"][c]), fma_case["bias"][c]) != 0.0:
                raise AssertionError("ordered F32 chain must land on exactly 0.0")
            residual = math.fma(p1, fma_case["sChannel"][c], fma_case["bias"][c])
            if residual == 0.0:
                raise AssertionError("bias FMA counterexample does not discriminate")
            fma_residual = max(fma_residual, abs(residual))

    halfway_case = by_id["bf16_halfway_and_early_cast"]
    halfway = halfway_case["expected"]["output"]
    # 2**20+4096 ties to even -> 2**20 ; 2**20+12288 ties to even -> 2**20+8192.
    if halfway[0] != 1048576.0 or halfway[1] != 1064960.0:
        raise AssertionError(f"bf16 halfway case did not round ties to even: {halfway}")
    for index, (target, bias_v) in enumerate(
            zip((1052672.0, 1060864.0, 1052672.0, 1060864.0), halfway_case["bias"])):
        acc_v = halfway_case["expected"]["accumulatorI32"][index]
        p2 = f32_mul(f32_mul(f32(float(acc_v)), halfway_case["expected"]["xScale"][0]),
                     halfway_case["sChannel"][index])
        if f32_add(p2, bias_v) != target:
            raise AssertionError("epilogue does not land exactly on the BF16 halfway")

    global_case = by_id["convrot_global_row_scale"]
    global_row = global_case["expected"]
    groups_per_row = global_case["k"] // CONVROT_GROUP
    for r in range(global_case["m"]):
        groups = global_row["rotGroupAbsmax"][r * groups_per_row:(r + 1) * groups_per_row]
        if max(groups) / max(min(groups), 1e-300) < 1e4:
            raise AssertionError("global row scale case does not span ConvRot-group maxima")
        expected_scale = cuda_fmaxf(
            f32(cuda_fminf(global_row["rowAbsmax"][r], BF16_MAX) * ROW_SCALE_RECIP), ROW_SCALE_FLOOR)
        if global_row["xScale"][r] != expected_scale:
            raise AssertionError("x scale is not the global row scale")

    dense_case = by_id["activation_quant_vs_dense"]
    act = dense_case["expected"]["xQuantized"]
    if min(act) != -128:
        raise AssertionError("activation saturation case does not reach xq == -128")
    dense_cf = [c for c in counterfactual_summary
                if c["case"] == "activation_quant_vs_dense"
                and c["counterfactual"] == "denseFloatingFallback"][0]
    if dense_cf["maxRelativeDiff"] <= 0.05:
        raise AssertionError(
            "dense-fallback discriminator is not material: max relative difference "
            f"{dense_cf['maxRelativeDiff']} (abs {dense_cf['maxAbsDiff']})")

    # The row-scale constant form is a real bit-level fork (see internalChecks
    # .rowScaleConstantForm): a consumer dividing by 127 instead of multiplying by the
    # compile-time constant gets a different xScale for some rows, so pin it with a case.
    for case in out_cases:
        for value in case["expected"]["rowAbsmax"]:
            clamped = cuda_fminf(value, BF16_MAX)
            if f32_mul(clamped, ROW_SCALE_RECIP) != f32_div(clamped, 127.0):
                note_discriminator("rowScaleConstantForm", case["id"])
                break

    required_discriminators = [
        "packedNibbleOrderSwapped", "denseFloatingFallback", "noActivationQuantization",
        "perTileRequant", "perChunkWeightScale", "weightClampSymmetric128", "weightRoundHalfAway",
        "fusedBiasFma", "reassociatedScales", "earlyBf16Cast", "biasAddedInBf16", "cpuPortableEager",
        "biasRoundedToActivationDtype", "rowScaleConstantForm",
    ]
    uncovered = [name for name in required_discriminators if not discriminators.get(name)]
    if uncovered:
        raise AssertionError(f"no case discriminates {uncovered}")

    fixture = {
        "schemaVersion": SCHEMA_VERSION,
        "reference": {
            "story": STORY,
            "loaderRevision": LOADER_REVISION,
            "kitchenRevision": {"main": KITCHEN_MAIN, "prHead": KITCHEN_PR_HEAD,
                                "mergedTree": KITCHEN_MERGED_TREE},
            "frozenArtifactTree": FROZEN_ARTIFACT_TREE,
            "kernelBlobs": source_records,
            "executionBackend": "cpu-kernel-replay",
            "upstreamCudaExecuted": False,
            "noDenseFallback": True,
            "literalQuantizedTensorDispatch": literal,
            "arithmetic": {
                "weightDecode": "w4a8_gemm.cu dequant_int4_grouped_to_int8_kernel: even column low "
                                "nibble, odd column high nibble; level = codebook[code] else code-8; "
                                "round-to-nearest-even of the binary32 level*s_rel product, then the "
                                "integer clamp [-127,127]",
                "activationQuant": "int8_linear.cu quantize_int8_rowwise_convrot_kernel: ConvRot "
                                   "(regular H4, strides 1/4/16/64, 0.5 per stage) over 256-column "
                                   "groups; ONE global row scale max(min(absmax,BF16_MAX)*(1/127),1e-30) "
                                   "over the complete row; quant_div_float_to_float<bf16> (rot and scale "
                                   "rounded to BF16, quotient rounded to BF16), nearbyintf, clamp [-128,127]",
                "accumulator": "cutlass_gemm_int8.cu ElementAcc=int32_t exact integer dot over the "
                               "complete K (per-256-column partial sums recorded in chunkPartialI32)",
                "epilogue": "cutlass_gemm_int8.cu EVT Accum->XScale->Mul0->WScale->Mul1->Add2->StoreD: "
                            "EXACTLY ((acc*xs)*s_channel)+bias as three separate binary32 roundings, "
                            "bias is binary32, one round-to-nearest to BF16 at the store",
                "rotation": "EXACT non-Walsh regular H4 [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]], "
                            "block-diagonal groups of 256 along K, applied to the activation only",
            },
            "torch": torch.__version__,
            "device": "cpu",
            "nativeTileBoundaries": {
                "candleOutputRowTile": CANDLE_OUTPUT_ROW_TILE,
                "mlxColumnTile": MLX_COLUMN_TILE,
                "convrotGroup": CONVROT_GROUP,
                "groupSize": GROUP_SIZE,
            },
            "kernelReplayCounters": counters,
            "regeneration": {"command": REGEN_COMMAND,
                             "note": "pinned source text is supplied through --reference-dir and is "
                                     "hash-verified (git blob SHA-1 + SHA-256) before execution; "
                                     "no network and no model payload are used"},
            "synthetic": "All operands are owned synthetic data produced by a portable SplitMix64 "
                         "stream; no model weights, no HF payload and no cache are read.",
        },
        "cases": out_cases,
        "refusedOperands": refusals,
        "mixedLayer": mixed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, separators=(",", ":")) + "\n", encoding="utf-8")
    fixture_sha = hashlib.sha256(args.output.read_bytes()).hexdigest()

    receipt = {
        "story": STORY,
        "fixture": {"path": str(args.output), "bytes": args.output.stat().st_size,
                    "sha256": fixture_sha, "cases": len(out_cases),
                    "regeneration": REGEN_COMMAND},
        "sourceVerification": {
            "referenceDir": str(args.reference_dir),
            "blobs": source_records,
            "license": "Apache-2.0 (comfy-kitchen LICENSE, SPDX-License-Identifier: Apache-2.0 on "
                       "w4a8_gemm.cu / cutlass_gemm_int8.cu / int8_linear.cu); no upstream source text "
                       "is copied into this repository",
        },
        "execution": {
            "backend": "cpu-kernel-replay",
            "upstreamCudaExecuted": False,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": "cpu",
            "cudaAvailable": torch.cuda.is_available(),
            "torchVersionCuda": torch.version.cuda,
            "env": {k: os.environ.get(k) for k in
                    ("CUDA_VISIBLE_DEVICES", "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")},
        },
        "literalQuantizedTensorDispatch": literal,
        "refusedOperands": [r["id"] for r in refusals],
        "mixedLayer": {
            "w4a8Case": mixed["w4a8Case"],
            "convrotAuthority": mixed["convrot"]["authority"],
        },
        "kernelReplayCounters": counters,
        "internalChecks": {
            "fp8E4M3Decode": fp8_check,
            "rowScaleConstantForm": scale_check,
            "hashedCudaSourceForms": form_check,
            "pinnedLayoutDecodeCrossChecks": counters["pinnedLayoutDecodeCrossChecks"],
            "noFloatMatmul": counters["floatMatmulCalls"] == 0,
            "noDenseDequantizeFallback": counters["denseDequantizeFallbacks"] == 0,
        },
        "caseConstruction": {op.case_id: op.note for op in cases if op.note},
        "counterfactualDiscriminators": discriminators,
        "counterfactuals": counterfactual_summary,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {len(out_cases)} CPU-kernel-replay W4A8 cases to {args.output}")
    print(f"fixture sha256 {fixture_sha}")
    print(f"receipt {args.receipt}")
    print(f"counters {counters}")
    return 0


if __name__ == "__main__":
    sys.exit(main())