// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org — the W4A8 scheme and its int4->int8 weight
// decode are derived from comfy-kitchen `comfy_kitchen/backends/cuda/ops/w4a8_gemm.cu`
// (`dequant_int4_grouped_to_int8_kernel`, PR#90 blob 21002ea10) and the int8 GEMM epilogue of
// `cutlass_gemm_int8.cu` (blob f7142fa41). Apache-2.0. Adapted for SceneWorks (SW-6134): this file
// is new code — a single-file int8 tensor-core GEMM that decodes the packed weight inside the
// kernel instead of materializing an int8 weight tile first.
//
// Fused W4A8 GEMM: Y[m, n] = ((float(sum_k Qx[m,k] * Wq[n,k]) * xs[m]) * s_channel[n]) + bias[n]
//
//   Qx        int8 [M, K]   the ConvRot-rotated, row-quantized activation (kitchen_int8.cu)
//   xs        f32  [M]      its per-row scale
//   packed    u8   [N, K/2] two int4 codes per byte, even column in the LOW nibble
//   s_rel     u8   [N, K/16] OCP E4M3FN per-group relative scale
//   codebook  f32  [16]     the level table the codes index
//   Wq[n,k] = clamp(rintf(codebook[code] * e4m3(s_rel[n, k/16])), -127, 127)   (exactly the
//             w4a8_dequant.cu decode, same helper, same operation order)
//
// The int32 accumulation is exact (|acc| <= K * 127 * 128 < 2^31 for K <= 131072, checked on the
// host), so the result is independent of summation order: this kernel is bit-identical to the
// dequant-tile + cuBLASLt IGEMM reference whenever the epilogue repeats its float ops in order.
// The epilogue uses __fmul_rn / __fadd_rn so nvrtc cannot contract them into an FMA.
//
// Tensor cores: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 (sm_80+; the host floor).
// Tiling: a 128x128 output tile per 256-thread block (8 warps as 2 (M) x 4 (N), 64x32 per warp),
// K stepped by 64 with a register-prefetched single shared buffer. Shared rows are padded to 80
// bytes so the fragment loads are bank-conflict free. Standard intrinsics only; no headers.

typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned int uint;

#define W4A8_BM 128
#define W4A8_BN 128
#define W4A8_BK 64
#define W4A8_LDS 80
#define W4A8_THREADS 256

// Bit-identical to w4a8_dequant.cu's `w4a8_e4m3_to_f32` (and gen_core::comfy_quant): exponent 0 is
// subnormal, 0x7F/0xFF (refused at load) decode to 0 defensively.
__device__ __forceinline__ float w4a8g_e4m3_to_f32(uint8_t scale_code) {
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

// Decode the 8 int4 codes of one packed 32-bit word (columns 0..7, nibble j = column j) to 8 int8
// values, returned as two little-endian words.
__device__ __forceinline__ void w4a8g_decode_word(
    uint word, float scale, const float* __restrict__ cb, uint* lo, uint* hi)
{
    uint a = 0u, b = 0u;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        uint nibble = (word >> (4 * j)) & 0xFu;
        int v = (int)(int8_t)max(-127.0f, min(127.0f, rintf(cb[nibble] * scale)));
        uint byte = (uint)v & 0xFFu;
        if (j < 4) {
            a |= byte << (8 * j);
        } else {
            b |= byte << (8 * (j - 4));
        }
    }
    *lo = a;
    *hi = b;
}

__device__ __forceinline__ uint w4a8g_lds32(const uint8_t* p) {
    return *reinterpret_cast<const uint*>(p);
}

__device__ __forceinline__ unsigned short w4a8g_f32_to_bf16(float x) {
    uint u = __float_as_uint(x);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (unsigned short)((u >> 16) | 0x40u);
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}

__device__ __forceinline__ unsigned short w4a8g_f32_to_f16(float x) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    return h;
}

extern "C" __global__ void __launch_bounds__(W4A8_THREADS) w4a8_int4_gemm(
    const uint8_t* __restrict__ qx,        // [M, K] int8 bytes
    const float*   __restrict__ xs,        // [M]
    const uint8_t* __restrict__ packed,    // [N, K/2]
    const uint8_t* __restrict__ s_rel,     // [N, K/16]
    const float*   __restrict__ codebook,  // [16]
    const float*   __restrict__ s_channel, // [N]
    const float*   __restrict__ bias,      // [N] (read only when has_bias)
    void*          __restrict__ out,       // [M, N] f32 / bf16 / f16
    int M, int N, int K, int has_bias, int out_dtype)
{
    __shared__ __align__(16) uint8_t As[W4A8_BM * W4A8_LDS];
    __shared__ __align__(16) uint8_t Bs[W4A8_BN * W4A8_LDS];
    __shared__ float cb[16];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = warp >> 2;  // 0..1 -> 64 rows each
    const int warp_n = warp & 3;   // 0..3 -> 32 columns each
    const int g = lane >> 2;       // mma groupID
    const int t4 = lane & 3;       // mma threadID_in_group
    const int bm0 = blockIdx.y * W4A8_BM;
    const int bn0 = blockIdx.x * W4A8_BN;

    if (tid < 16) {
        cb[tid] = codebook[tid];
    }

    int acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r)
                acc[i][j][r] = 0;

    // Global -> register staging. A: 128 rows x 64 bytes = 512 16-byte chunks, two per thread.
    // B: 128 weight rows x 32 packed bytes = 256 chunks, one per thread (two 16-column groups).
    const int a_row0 = tid >> 2, a_col = (tid & 3) * 16;          // chunk tid
    const int a_row1 = (tid + W4A8_THREADS) >> 2;                  // chunk tid + 256
    const int b_row = tid >> 1, b_half = tid & 1;
    const long long kh = (long long)(K / 2), kg = (long long)(K / 16);
    uint4 ra0, ra1, rb;
    uint8_t sc0 = 0, sc1 = 0;
    const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);

    const int ktiles = K / W4A8_BK;

#define W4A8_LOAD(k0)                                                                           \
    {                                                                                           \
        int gm0 = bm0 + a_row0, gm1 = bm0 + a_row1;                                             \
        ra0 = gm0 < M ? *reinterpret_cast<const uint4*>(qx + (long long)gm0 * K + (k0) + a_col) \
                      : zero4;                                                                  \
        ra1 = gm1 < M ? *reinterpret_cast<const uint4*>(qx + (long long)gm1 * K + (k0) + a_col) \
                      : zero4;                                                                  \
        int gn = bn0 + b_row;                                                                   \
        if (gn < N) {                                                                           \
            rb = *reinterpret_cast<const uint4*>(packed + (long long)gn * kh + (k0) / 2         \
                                                 + b_half * 16);                                \
            const uint8_t* sp = s_rel + (long long)gn * kg + (k0) / 16 + b_half * 2;            \
            sc0 = sp[0];                                                                        \
            sc1 = sp[1];                                                                        \
        } else {                                                                                \
            rb = zero4;                                                                         \
            sc0 = 0;                                                                            \
            sc1 = 0;                                                                            \
        }                                                                                       \
    }

    W4A8_LOAD(0);
    __syncthreads();  // codebook visible

    for (int kt = 0; kt < ktiles; ++kt) {
        // Store the staged tile (decoding the weight on the way) into the shared buffer.
        *reinterpret_cast<uint4*>(As + a_row0 * W4A8_LDS + a_col) = ra0;
        *reinterpret_cast<uint4*>(As + a_row1 * W4A8_LDS + a_col) = ra1;
        {
            float s0 = w4a8g_e4m3_to_f32(sc0), s1 = w4a8g_e4m3_to_f32(sc1);
            uint w[8];
            w4a8g_decode_word(rb.x, s0, cb, &w[0], &w[1]);
            w4a8g_decode_word(rb.y, s0, cb, &w[2], &w[3]);
            w4a8g_decode_word(rb.z, s1, cb, &w[4], &w[5]);
            w4a8g_decode_word(rb.w, s1, cb, &w[6], &w[7]);
            uint8_t* dst = Bs + b_row * W4A8_LDS + b_half * 32;
            *reinterpret_cast<uint4*>(dst) = make_uint4(w[0], w[1], w[2], w[3]);
            *reinterpret_cast<uint4*>(dst + 16) = make_uint4(w[4], w[5], w[6], w[7]);
        }
        __syncthreads();
        if (kt + 1 < ktiles) {
            W4A8_LOAD((kt + 1) * W4A8_BK);
        }
#pragma unroll
        for (int ks = 0; ks < W4A8_BK; ks += 32) {
            uint af[4][4];
            uint bf[4][2];
#pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                const uint8_t* base = As + (warp_m * 64 + mt * 16 + g) * W4A8_LDS + ks + t4 * 4;
                af[mt][0] = w4a8g_lds32(base);
                af[mt][1] = w4a8g_lds32(base + 8 * W4A8_LDS);
                af[mt][2] = w4a8g_lds32(base + 16);
                af[mt][3] = w4a8g_lds32(base + 8 * W4A8_LDS + 16);
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const uint8_t* base = Bs + (warp_n * 32 + nt * 8 + g) * W4A8_LDS + ks + t4 * 4;
                bf[nt][0] = w4a8g_lds32(base);
                bf[nt][1] = w4a8g_lds32(base + 16);
            }
#pragma unroll
            for (int mt = 0; mt < 4; ++mt)
#pragma unroll
                for (int nt = 0; nt < 4; ++nt)
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+r"(acc[mt][nt][0]), "+r"(acc[mt][nt][1]),
                          "+r"(acc[mt][nt][2]), "+r"(acc[mt][nt][3])
                        : "r"(af[mt][0]), "r"(af[mt][1]), "r"(af[mt][2]), "r"(af[mt][3]),
                          "r"(bf[nt][0]), "r"(bf[nt][1]));
        }
        __syncthreads();
    }
#undef W4A8_LOAD

    // Epilogue: ((float(acc) * xs[m]) * s_channel[n]) + bias[n], each op round-to-nearest and
    // unfused, then one rounding to the output dtype — the reference's eager F32 op order.
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int row = bm0 + warp_m * 64 + mt * 16 + g + half * 8;
            if (row >= M) {
                continue;
            }
            float xsr = xs[row];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    int col = bn0 + warp_n * 32 + nt * 8 + t4 * 2 + e;
                    if (col >= N) {
                        continue;
                    }
                    float v = __fmul_rn(__int2float_rn(acc[mt][nt][half * 2 + e]), xsr);
                    v = __fmul_rn(v, s_channel[col]);
                    if (has_bias) {
                        v = __fadd_rn(v, bias[col]);
                    }
                    long long o = (long long)row * N + col;
                    if (out_dtype == 0) {
                        reinterpret_cast<float*>(out)[o] = v;
                    } else if (out_dtype == 1) {
                        reinterpret_cast<unsigned short*>(out)[o] = w4a8g_f32_to_bf16(v);
                    } else {
                        reinterpret_cast<unsigned short*>(out)[o] = w4a8g_f32_to_f16(v);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Fused activation prep: the pinned ConvRot FHT (block 256) + Kitchen's row-wise int8 quantizer in
// one launch (comfy-kitchen `int8_linear.cu::quantize_int8_rowwise_convrot_kernel`, PR#90 blob
// 4c7d49e72, which rotates inside the quantizer). It replaces the eager reference — the F32
// butterfly as separate candle ops (`convrot::convrot_fht_f32`) followed by `kitchen_int8.cu` —
// and must produce byte-identical codes and bit-identical scales:
//   * the butterfly is the same four radix-4 stages (strides 1, 4, 16, 64), each output
//     `((a+b)+c)-d`, `((a+b)-c)+d`, `((a-b)+c)+d`, `((-a+b)+c)+d` then `* 0.5f`, every op a
//     separately rounded binary32 op (__fadd_rn / __fsub_rn / __fmul_rn: no contraction);
//   * the row absmax is taken over those F32 values (max is order independent);
//   * the scale and per-element quotient repeat kitchen_int8.cu's arithmetic verbatim.
// One block of 256 threads per row; each 256-column group is rotated in shared memory twice (once
// for the absmax, once to quantize) so no row-sized buffer is needed for any K.

__device__ __forceinline__ float w4a8g_source_round(float x, int dtype) {
    if (dtype == 1) {
        uint u = __float_as_uint(x);
        if ((u & 0x7fffffffU) > 0x7f800000U) return x;
        u += 0x7fffU + ((u >> 16) & 1U);
        return __uint_as_float(u & 0xffff0000U);
    }
    if (dtype == 2) {
        unsigned short h;
        float y;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
        asm("cvt.f32.f16 %0, %1;" : "=f"(y) : "h"(h));
        return y;
    }
    return x;
}

__device__ __forceinline__ float w4a8g_load(const void* x, long long i, int dtype) {
    if (dtype == 1) {
        uint bits = (uint)reinterpret_cast<const unsigned short*>(x)[i];
        return __uint_as_float(bits << 16);
    }
    if (dtype == 2) {
        unsigned short h = reinterpret_cast<const unsigned short*>(x)[i];
        float y;
        asm("cvt.f32.f16 %0, %1;" : "=f"(y) : "h"(h));
        return y;
    }
    return reinterpret_cast<const float*>(x)[i];
}

// Rotate the 256 values in `buf` in place; thread `tid` owns element `tid` of every stage.
__device__ __forceinline__ void w4a8g_fht256(float* buf, int tid) {
#pragma unroll
    for (int s = 1; s < 256; s *= 4) {
        int j = (tid / s) & 3;
        int base = (tid / (4 * s)) * (4 * s) + (tid % s);
        float a = buf[base], b = buf[base + s], c = buf[base + 2 * s], d = buf[base + 3 * s];
        float o;
        if (j == 0) {
            o = __fsub_rn(__fadd_rn(__fadd_rn(a, b), c), d);
        } else if (j == 1) {
            o = __fadd_rn(__fsub_rn(__fadd_rn(a, b), c), d);
        } else if (j == 2) {
            o = __fadd_rn(__fadd_rn(__fsub_rn(a, b), c), d);
        } else {
            o = __fadd_rn(__fadd_rn(__fadd_rn(-a, b), c), d);
        }
        __syncthreads();
        buf[tid] = __fmul_rn(o, 0.5f);
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(256) w4a8_quantize_fht256(
    const void* __restrict__ x,   // [M, K] in `dtype`
    uint8_t* __restrict__ q,      // [M, K] int8 codes
    float* __restrict__ scales,   // [M]
    int k, int dtype)
{
    __shared__ float buf[256];
    __shared__ float peaks[256];
    const int tid = threadIdx.x;
    const long long offset = (long long)blockIdx.x * k;
    float peak = 0.0f;
    for (int c0 = 0; c0 < k; c0 += 256) {
        buf[tid] = w4a8g_load(x, offset + c0 + tid, dtype);
        __syncthreads();
        w4a8g_fht256(buf, tid);
        peak = fmaxf(peak, fabsf(buf[tid]));
        __syncthreads();
    }
    peaks[tid] = peak;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (tid < stride) peaks[tid] = fmaxf(peaks[tid], peaks[tid + stride]);
        __syncthreads();
    }
    float limit = dtype == 1 ? 3.38953139e38f : dtype == 2 ? 65504.0f : 3.402823466e38f;
    float scale = fmaxf(fminf(peaks[0], limit) * (1.0f / 127.0f), 1.0e-30f);
    if (tid == 0) scales[blockIdx.x] = scale;
    float divisor = w4a8g_source_round(scale, dtype);
    for (int c0 = 0; c0 < k; c0 += 256) {
        buf[tid] = w4a8g_load(x, offset + c0 + tid, dtype);
        __syncthreads();
        w4a8g_fht256(buf, tid);
        float z = w4a8g_source_round(w4a8g_source_round(buf[tid], dtype) / divisor, dtype);
        float rounded = fminf(127.0f, fmaxf(-128.0f, nearbyintf(z)));
        q[offset + c0 + tid] = (uint8_t)(int8_t)rounded;
        __syncthreads();
    }
}
