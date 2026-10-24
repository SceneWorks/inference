// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org — derived from comfy-kitchen
// `w4a8_gemm.cu` `dequant_int4_grouped_to_int8_kernel` (PR#90, blob 21002ea10). Apache-2.0.
//
// Packed W4A8 weight decode (sc-26685): two int4 codes per I8 byte (even column = LOW nibble),
// level = codebook[nibble], per-group fp8 (OCP E4M3FN) relative scale, and
// `int8 = round_ties_even(level * scale)` clamped to [-127, 127] — the int8 weight the shared
// cuBLASLt INT8 IGEMM multiplies. Compiled once per device through the nvrtc compile-once seam
// (`nvrtc.rs`), standard intrinsics only, no headers and no build.rs (the same contract as
// `kitchen_int8.cu`). `round_ties_even` is `rintf` (round-half-to-even), matching the pinned
// `__float2int_rn`.
//
// The activation side is untouched by this file: the full row is ConvRot-rotated and INT8-quantized
// ONCE by `kitchen_int8.cu` (`quantize_kitchen_int8`), and this decode runs per bounded output-row
// tile of the weight, so the resident format stays the packed `[out, in/2]` codes + F8 s_rel + F32
// s_channel + F32[16] codebook — never a dense INT8 model shadow.

typedef unsigned char uint8_t;
typedef signed char int8_t;

// OCP E4M3FN byte -> f32, bit-identical to `gen_core::comfy_quant::fp8_e4m3fn_to_f32` and the MLX
// DEQUANT kernel: exponent 0 is subnormal (mantissa/8 · 2^-6); 0x7F/0xFF are NaN (which
// `validate_s_rel` refuses at load, but is decoded to 0 defensively as the host `NaN as i8` does).
__device__ __forceinline__ float w4a8_e4m3_to_f32(uint8_t scale_code) {
    uint exponent = (scale_code >> 3) & 0x0Fu;
    uint mantissa = scale_code & 0x07u;
    if (exponent == 0x0Fu && mantissa == 0x07u) {
        return 0.0f;
    }
    float magnitude = exponent == 0u
        ? (float)mantissa * 0.125f * 0.015625f
        : (1.0f + (float)mantissa * 0.125f) * exp2f((float)exponent - 7.0f);
    return (scale_code & 0x80u) == 0u ? magnitude : -magnitude;
}

// One thread per decoded element. `out0` is the first output feature of this tile, so a tile is a
// contiguous `[rows, k]` slice of the resident packed `[out, k]` weight.
extern "C" __global__ void w4a8_dequant_int8(
    const uint8_t* __restrict__ packed,   // [out, k/2]
    const uint8_t* __restrict__ s_rel,    // [out, k/g]
    const float*   __restrict__ codebook, // [16]
    int8_t*        __restrict__ out,      // [rows, k]
    int k, int g, int out0)
{
    long work = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int c = (int)(work % (long)k);
    int row = (int)(work / (long)k);
    int ofeat = out0 + row;
    uint packed_byte = packed[(long)ofeat * (k / 2) + (c / 2)];
    uint nibble = (c & 1) == 0 ? (packed_byte & 0xFu) : (packed_byte >> 4);
    float scale = w4a8_e4m3_to_f32(s_rel[(long)ofeat * (k / g) + (c / g)]);
    out[work] = (int8_t)max(-127.0f, min(127.0f, rintf(codebook[nibble] * scale)));
}