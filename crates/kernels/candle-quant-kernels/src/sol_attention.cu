// SW-6137 (SW-F287 S12): Sol-Attn sparse block-routed attention, opt-in and lossy.
//
// Ported source, single file for the nvrtc seam (no includes; the shared INT8-attention primitives of
// int8_attention_common.cuh are prepended at compile time). Attribution (E10):
//   * comfy-kitchen @ 6022b48b7e39fdd31f0124de268a8267af46d4e0
//     comfy_kitchen/backends/cuda/sage_attention/{sol_layout.cuh, sol_attn.cu,
//     sol_attn_preprocess.cu, sol_attn_vtranspose.cu, sol_attn_route.cu, sol_attn_exact.cu} and the
//     top-k threshold of comfy_kitchen/backends/cuda/__init__.py (`_topk_from_pooled`) -
//     Apache-2.0, Copyright (c) 2025 Comfy Org. All rights reserved. Licensed under the Apache
//     License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0); distributed on an "AS IS"
//     BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
//   * The algorithm: Sol-Attn, training-free sparse attention for video diffusion (arXiv 2607.24027),
//     as comfy-kitchen implements it; the top-k selection is its SLA-style mode.
// See candle-quant-kernels/src/sol_attention-NOTICE.md and docs/licensing/third-party-kernel-sources.md.
//
// Pipeline (one call; host orchestration in sol_attention.rs):
//   sol_reduce_kv      K block means, V block sums (bf16), V per-channel |max|      [preprocess pass 1]
//   sol_pooled_stats   K mean over blocks, K block-mean variance, V scale            [pass 2]
//   sol_pooled_quant   centred pooled keys -> INT8                                   [pass 3]
//   sol_prep_q         Q -> INT8 per token, query-block centroid -> INT8, tau thr    [pass 4]
//   sol_prep_k         centred K -> INT8 per token, permuted for the P.V repack      [pass 5]
//   sol_vtranspose     V -> INT8 per channel, transposed to [D, keys]
//   sol_topk_threshold per query block, the k-th best pooled score (top-k mode; replaces tau's)
//   sol_route          routed block lists + the pooled tail's online-softmax state, per query block
//   sol_exact          each query block's routed key blocks exactly (INT8 Q.K, U8 P.V), resuming
//                      route's softmax state
// route and exact run per query-block window [qb0, qb0 + nqc) (the chunked-query variant): every
// per-query-block quantity is independent of the window, so a chunked call is bitwise the unchunked.
//
// Differences from comfy-kitchen (all host-refused or absent, never silently different):
//   * no token augmentation (sol_attn_token.cu), no chunked fused-QKV producer
//     (sol_attn_producer.cu: per-head RMSNorm + RoPE over fused qkv chunks), no key bias, no
//     block_len (zero-padded tiles), no coarse gate, no key prefix (q_offset): Sq == Sk.
//   * Q carriers and the output are [B*H, T, D] (the [B, H, S, D] layout the dispatch hands over)
//     instead of [B, T, H, D].
//   * the top-k threshold is computed on the device from the route kernel's own integer scores
//     (bitwise the scores route compares, so no ulp back-off), and `topk_ratio >= 1` routes every
//     key block (comfy-kitchen caps the budget at n - 1 blocks and accepts ratios in (0, 1) only).
//   * byte rows are packed into 32-bit words in registers (no addressable byte arrays: the seam's
//     zero-local-memory rule), the 16-bit loads/stores use the shared ia_* helpers, and
//     i2f_exact is a plain int->float conversion (exact: |Q.K| <= 127*127*128 < 2^22).
//
// Layouts (element strides host-validated: unit last dim, row/head/batch strides and the start
// offset multiples of 8 elements, so every 16-byte load is aligned):
//   q / k / v   bf16|f16, logical [B, H, T, 128], read through (batch, token, head) strides.
//   qiP [BH, T, D] int8 (perm_d)             qs  [BH, T] f32
//   kiP [BH, Tp, D] int8 (perm_key + perm_d) ksb [BH, Tp] float2 (scale, bias)
//   vTi [BH, D, Tp] int8 (perm_d on keys)    vsc [BH, D] f32
//   kc (scratch) [BH, NPAD, D] f32 K block means; kmean / kcvar [BH, D] f32
//   kciP [BH, NPAD, D] int8 (perm_d)         kcs [BH, NPAD] f32
//   vcT [BH, D, NPAD] bf16 (V block sums)    thr [BH, NQ] f32
//   cen8 [BH, NQ, D] int8 (perm_d)           cens [BH, NQ] f32
//   idx [BH, nqc, NTB] u16; cnt [BH, nqc] i32; oPart [BH, nqc, D] bf16; mPart / lPart [BH, nqc] f32
//   out [BH, T, D] bf16|f16
// T = tokens (Sq == Sk), Tp = 64-padded, NTB = Tp / 64 key blocks, NQ = NTB query blocks,
// NPAD = NTB rounded up to 64. All offsets are 64-bit.

#define SOL_HD 128
#define SOL_BLK 64
#define SOL_NEG (-3.0e38f)          // finite, so NEG - NEG == 0 (unlike -inf)
#define SOL_LD_TILE (SOL_HD + 8)    // 16-bit row stride of a staged tile; keeps uint4 stores aligned
#define SOL_TILE_MASKED (-1.0e37f)  // scores at or below this are masked (NEG lands here)

typedef unsigned char u8;

// Contraction-axis permutation: each lane's two MMA operand words become one 8-byte load. Applied to
// Q/K/pooled-K d axes and V^T's key axis.
__device__ __forceinline__ int sol_perm_d(int d) {
    const int kc = d >> 5, rem = d & 31, h = rem >> 4, r2 = rem & 15;
    return kc * 32 + 8 * (r2 >> 2) + 4 * h + (r2 & 3);
}
// Which source channel lands at permuted position p (sol_perm_d's inverse).
__device__ __forceinline__ int sol_perm_d_inv(int p) {
    const int kc = p >> 5, rem = p & 31, a = rem >> 3, h = (rem >> 2) & 1, b = rem & 3;
    return kc * 32 + h * 16 + 4 * a + b;
}
// Key relabelling per 64-block so the INT8 P.V A operand needs no shuffles. Applied to K rows +
// scales; NOT to V^T.
__device__ __forceinline__ int sol_perm_key(int p) {
    return 16 * (p >> 4) + 4 * ((p & 7) >> 1) + 2 * ((p >> 3) & 1) + (p & 1);
}

// Smem XOR swizzles (K tile 64 x 128 B, V^T tile 128 x 64 B).
__device__ __forceinline__ int sol_swz_k(int row) { return (row & 3) * 2; }
__device__ __forceinline__ int sol_swz_v(int col) { return ((col >> 2) ^ col) & 3; }

__device__ __forceinline__ int sol_q8(float x, float inv) {
    return max(-127, min(127, __float2int_rn(x * inv)));
}
__device__ __forceinline__ u32 sol_byte(int q, int j) { return ((u32)(q & 0xff)) << (8 * j); }

template <bool BF>
__device__ __forceinline__ u16 sol_store16(float x) {
    u16 r;
    if (BF) asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
    else asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(x));
    return r;
}

// 128-thread reductions, one thread per channel; every thread gets the result and `s` is free on
// return.
__device__ __forceinline__ float sol_block_max128(float x, float* s) {
    const int d = threadIdx.x;
    s[d] = x;
    __syncthreads();
    for (int w = 64; w; w >>= 1) {
        if (d < w) s[d] = fmaxf(s[d], s[d + w]);
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}
__device__ __forceinline__ float sol_block_sum128(float x, float* s) {
    const int d = threadIdx.x;
    s[d] = x;
    __syncthreads();
    for (int w = 64; w; w >>= 1) {
        if (d < w) s[d] += s[d + w];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}

// Live tokens at the front of 64-block n (the ragged tail only; no block_len table).
__device__ __forceinline__ int sol_block_len(int n, int T) { return min(SOL_BLK, T - n * SOL_BLK); }

// Stage rows 0..len-1 (row t at src + t*stride, 16 B loads) into the tile, zero past len.
__device__ __forceinline__ void sol_stage_tile64(u16* tile, const u16* __restrict__ src, i64 stride,
                                                 int len) {
    for (int idx = threadIdx.x; idx < SOL_BLK * (SOL_HD / 8); idx += SOL_HD) {
        const int t = idx / (SOL_HD / 8), c8 = (idx % (SOL_HD / 8)) * 8;
        uint4 val = make_uint4(0u, 0u, 0u, 0u);
        if (t < len) val = *reinterpret_cast<const uint4*>(src + (i64)t * stride + c8);
        *reinterpret_cast<uint4*>(tile + t * SOL_LD_TILE + c8) = val;
    }
}

// Per-token absmax scale + perm_d'd int8 row, one thread per token; rows [len, nrows) exist but are
// dead (zeros). qiP / qs point at this tile's first token; consecutive tokens are SOL_HD bytes /
// one float apart. The permuted row is built as 32 words in registers.
template <bool BF>
__device__ __forceinline__ void sol_quant_q_rows(const u16* tile, int len, int nrows,
                                                 i8* __restrict__ qiP, float* __restrict__ qs) {
    for (int t = threadIdx.x; t < nrows; t += SOL_HD) {
        const u16* row = tile + t * SOL_LD_TILE;  // zero-staged past len
        const bool live = t < len;
        float a = 0.f;
#pragma unroll 8
        for (int d = 0; d < SOL_HD; ++d) a = fmaxf(a, fabsf(ia_load(row + d, BF)));
        const float sc = live ? fmaxf(a / 127.0f, 1e-8f) : 0.f;  // dead rows: deterministic zeros
        qs[t] = sc;
        const float inv = live ? 1.f / sc : 0.f;
        u32* dst = reinterpret_cast<u32*>(qiP + (i64)t * SOL_HD);
#pragma unroll
        for (int w = 0; w < SOL_HD / 4; ++w) {
            u32 word = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j)
                word |= sol_byte(sol_q8(ia_load(row + sol_perm_d_inv(4 * w + j), BF), inv), j);
            dst[w] = word;
        }
    }
}

// Query-block centroid, quantized like a pseudo-row with the pooled keys' perm_d. One thread per
// channel; returns this thread's channel mean. `sred` holds bytes on return - sync before reuse.
template <bool BF>
__device__ __forceinline__ float sol_centroid_quant(const u16* tile, int len, float* sred,
                                                    i8* __restrict__ cen8,
                                                    float* __restrict__ cens) {
    const int d = threadIdx.x;
    float c = 0.f;
    for (int t = 0; t < len; ++t) c += ia_load(tile + t * SOL_LD_TILE + d, BF);
    c /= (float)len;
    const float csc = fmaxf(sol_block_max128(fabsf(c), sred) / 127.0f, 1e-8f);
    char* s8 = reinterpret_cast<char*>(sred);
    s8[sol_perm_d(d)] = (char)sol_q8(c, 1.f / csc);
    __syncthreads();
    if (d < SOL_HD / 16) reinterpret_cast<uint4*>(cen8)[d] = reinterpret_cast<const uint4*>(s8)[d];
    if (d == 0) *cens = csc;
    return c;
}

// Centred per-key scale + perm_d'd int8 row; destination row p takes SOURCE row perm_key(p). Dead
// rows get a zero scale, NEG bias and zero bytes.
template <bool BF>
__device__ __forceinline__ void sol_quant_k_rows(const u16* tile, int len,
                                                 const float* __restrict__ kmean,
                                                 i8* __restrict__ kiP, float2* __restrict__ ksb) {
    for (int p = threadIdx.x; p < SOL_BLK; p += SOL_HD) {
        const int s = sol_perm_key(p);
        const bool live = s < len;
        const u16* row = tile + s * SOL_LD_TILE;
        float a = 0.f;
        for (int d = 0; d < SOL_HD; ++d) a = fmaxf(a, fabsf(ia_load(row + d, BF) - kmean[d]));
        const float sc = fmaxf(a / 127.0f, 1e-8f);
        ksb[p] = make_float2(live ? sc : 0.f, live ? 0.f : SOL_NEG);
        const float inv = 1.f / sc;
        u32* dst = reinterpret_cast<u32*>(kiP + (i64)p * SOL_HD);
#pragma unroll
        for (int w = 0; w < SOL_HD / 4; ++w) {
            u32 word = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int d = sol_perm_d_inv(4 * w + j);
                word |= live ? sol_byte(sol_q8(ia_load(row + d, BF) - kmean[d], inv), j) : 0u;
            }
            dst[w] = word;
        }
    }
}

// ---- preprocess pass 1: K/V block reductions + V's per-channel |max| ----
// One CTA per (pooled block, bh), thread == channel. `vamax` must be zeroed.
template <bool BF>
__device__ void sol_reduce_kv(const u16* __restrict__ k, const u16* __restrict__ v,
                              float* __restrict__ kc, u16* __restrict__ vcT,
                              float* __restrict__ vamax, int T, int H, int NTB, int NPAD, i64 sb,
                              i64 st, i64 sh, i64 vb, i64 vt, i64 vh) {
    const int n = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H, d = threadIdx.x;
    const i64 o = ((i64)bh * NPAD + n) * SOL_HD + d;
    if (n >= NTB) {
        kc[o] = 0.f;
        vcT[((i64)bh * SOL_HD + d) * NPAD + n] = 0;
        return;
    }
    const int t0 = n * SOL_BLK, len = sol_block_len(n, T);
    float sk = 0.f, sv = 0.f, av = 0.f;
    for (int i = 0; i < len; ++i) {
        const float kk = ia_load(k + (i64)batch * sb + (i64)(t0 + i) * st + (i64)head * sh + d, BF);
        const float vv = ia_load(v + (i64)batch * vb + (i64)(t0 + i) * vt + (i64)head * vh + d, BF);
        sk += kk;
        sv += vv;
        av = fmaxf(av, fabsf(vv));
    }
    kc[o] = sk / (float)len;                                           // block MEAN of K
    vcT[((i64)bh * SOL_HD + d) * NPAD + n] = sol_store16<true>(sv);    // block SUM of V (bf16)
    atomicMax(reinterpret_cast<unsigned int*>(&vamax[(i64)bh * SOL_HD + d]),
              __float_as_uint(av));  // av >= 0, so the bit pattern orders correctly
}

// ---- pass 2: reductions over the pooled tensors; vsc holds pass 1's V |max| and becomes the V scale ----
extern "C" __global__ void __launch_bounds__(SOL_HD)
    sol_pooled_stats(const float* __restrict__ kc, float* __restrict__ kmean, float* vsc,
                     float* __restrict__ kcvar, int NTB, int NPAD) {
    const int bh = blockIdx.x, d = threadIdx.x;
    float sm = 0.f, ss = 0.f;
    for (int n = 0; n < NTB; ++n) {
        const float x = kc[((i64)bh * NPAD + n) * SOL_HD + d];
        sm += x;
        ss = fmaf(x, x, ss);
    }
    const float m = sm / (float)NTB;
    kmean[(i64)bh * SOL_HD + d] = m;
    vsc[(i64)bh * SOL_HD + d] = fmaxf(vsc[(i64)bh * SOL_HD + d] / 127.0f, 1e-8f);  // |max| in place
    kcvar[(i64)bh * SOL_HD + d] = fmaxf(ss / (float)NTB - m * m, 0.f);
}

// ---- pass 3: centre + quantize the pooled keys ----
extern "C" __global__ void __launch_bounds__(SOL_HD)
    sol_pooled_quant(const float* __restrict__ kc, const float* __restrict__ kmean,
                     i8* __restrict__ kciP, float* __restrict__ kcs, int NTB, int NPAD) {
    __shared__ float sred[SOL_HD];
    const int n = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const i64 o = ((i64)bh * NPAD + n) * SOL_HD + d;
    const bool live = n < NTB;
    const float x = live ? (kc[o] - kmean[(i64)bh * SOL_HD + d]) : 0.f;
    const float sc = fmaxf(sol_block_max128(fabsf(x), sred) / 127.0f, 1e-12f);
    if (d == 0) kcs[(i64)bh * NPAD + n] = live ? sc : 0.f;
    kciP[((i64)bh * NPAD + n) * SOL_HD + sol_perm_d(d)] = live ? (i8)sol_q8(x, 1.f / sc) : (i8)0;
}

// ---- pass 4: quantize Q, centroid, tau routing threshold from one staged tile ----
template <bool BF>
__device__ void sol_prep_q(const u16* __restrict__ q, const float* __restrict__ kcvar,
                           i8* __restrict__ qiP, float* __restrict__ qs, float* __restrict__ thr,
                           i8* __restrict__ cen8, float* __restrict__ cens, int T, int H, int NQ,
                           float tau, float log2s, i64 sb, i64 st, i64 sh) {
    __shared__ __align__(16) u16 sQ[SOL_BLK * SOL_LD_TILE];
    __shared__ __align__(16) float sred[SOL_HD];
    const int qb = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H;
    const int t0 = qb * SOL_BLK, len = sol_block_len(qb, T), nrows = len;
    sol_stage_tile64(sQ, q + (i64)batch * sb + (i64)t0 * st + (i64)head * sh, st, len);
    __syncthreads();
    const i64 tok0 = (i64)bh * T + t0;
    sol_quant_q_rows<BF>(sQ, len, nrows, qiP + tok0 * SOL_HD, qs + tok0);
    __syncthreads();
    const i64 qrow = (i64)bh * NQ + qb;
    const float c = sol_centroid_quant<BF>(sQ, len, sred, cen8 + qrow * SOL_HD, cens + qrow);
    __syncthreads();  // sred held the centroid bytes
    const float var = sol_block_sum128(c * c * kcvar[(i64)bh * SOL_HD + threadIdx.x], sred);
    // tau sigma of the proxy row, from the centroid's proxy variance
    if (threadIdx.x == 0) thr[qrow] = tau * sqrtf(var * log2s * log2s + 1e-6f);
}

// ---- pass 5: centre + quantize K into the permuted layout ----
template <bool BF>
__device__ void sol_prep_k(const u16* __restrict__ k, const float* __restrict__ kmean,
                           i8* __restrict__ kiP, float2* __restrict__ ksb, int T, int Tp, int H,
                           i64 sb, i64 st, i64 sh) {
    __shared__ __align__(16) u16 sK[SOL_BLK * SOL_LD_TILE];
    const int n = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H;
    const int t0 = n * SOL_BLK, len = sol_block_len(n, T);
    sol_stage_tile64(sK, k + (i64)batch * sb + (i64)t0 * st + (i64)head * sh, st, len);
    __syncthreads();
    const i64 dst0 = (i64)bh * Tp + t0;
    sol_quant_k_rows<BF>(sK, len, kmean + (i64)bh * SOL_HD, kiP + dst0 * SOL_HD, ksb + dst0);
}

// ---- V quantize + transpose: [B*H, D, Tp] int8, perm_d on the key axis per 64-block ----
#define SOL_VT_TT 256
#define SOL_VT_LDS (SOL_HD + 16)
template <bool BF>
__device__ void sol_vtranspose(const u16* __restrict__ v, const float* __restrict__ vsc,
                               i8* __restrict__ vT, int T, int Tp, int H, i64 sb, i64 st, i64 sh) {
    __shared__ __align__(16) i8 sV[SOL_VT_TT * SOL_VT_LDS];
    const int bh = blockIdx.y, head = bh % H, batch = bh / H;
    const int t0 = blockIdx.x * SOL_VT_TT;
    // phase 1: coalesced read (D contiguous) -> smem rows
    for (int idx = threadIdx.x; idx < SOL_VT_TT * (SOL_HD / 16); idx += 256) {
        const int t = idx / (SOL_HD / 16), c16 = (idx % (SOL_HD / 16)) * 16;
        uint4 out = make_uint4(0u, 0u, 0u, 0u);
        if (t0 + t < T) {
            const u16* src = v + (i64)batch * sb + (i64)(t0 + t) * st + (i64)head * sh + c16;
            u32 w[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                u32 word = 0;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int c = 4 * i + j;
                    word |= sol_byte(sol_q8(ia_load(src + c, BF), 1.f / vsc[(i64)bh * SOL_HD + c16 + c]), j);
                }
                w[i] = word;
            }
            out = make_uint4(w[0], w[1], w[2], w[3]);
        }
        const int tp = (t / 64) * 64 + sol_perm_d(t % 64);
        *reinterpret_cast<uint4*>(sV + tp * SOL_VT_LDS + c16) = out;
    }
    __syncthreads();
    // phase 2: 4x4 byte register transposes -> coalesced write (T contiguous)
    i8* dst0 = vT + (i64)bh * SOL_HD * Tp + t0;
    for (int idx = threadIdx.x; idx < (SOL_HD / 4) * (SOL_VT_TT / 4); idx += 256) {
        const int cg = idx / (SOL_VT_TT / 4), tg = (idx % (SOL_VT_TT / 4)) * 4;
        if (t0 + tg >= Tp) continue;
        u32 r[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            r[j] = *reinterpret_cast<const u32*>(sV + (tg + j) * SOL_VT_LDS + cg * 4);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const u32 w = ((r[0] >> (i * 8)) & 0xffu) | (((r[1] >> (i * 8)) & 0xffu) << 8) |
                          (((r[2] >> (i * 8)) & 0xffu) << 16) | (((r[3] >> (i * 8)) & 0xffu) << 24);
            *reinterpret_cast<u32*>(dst0 + (i64)(cg * 4 + i) * Tp + tg) = w;
        }
    }
}

// ---- top-k threshold (comfy-kitchen `_topk_from_pooled`, on the device) ----
// One CTA (128 threads) per (query block, bh). The pooled score of key block b is the route
// kernel's own arithmetic: (float)(cen8 . kciP[b]) * (cens * scale_log2) * kcs[b] - the integer dot
// is exact, so these are bitwise the scores route compares. The threshold is the kk-th largest
// non-sink score (route keeps score >= threshold, so a tied group at the boundary is kept); kk == 0
// stores +inf (nothing routes beyond the forced blocks). The k-th value is found by a 32-step
// radix descent over the order-preserving u32 image of the scores, held in dynamic smem.
__device__ __forceinline__ u32 sol_order_key(float x) {
    const u32 b = __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
__device__ __forceinline__ float sol_order_value(u32 k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

extern "C" __global__ void __launch_bounds__(SOL_HD)
    sol_topk_threshold(const i8* __restrict__ cen8, const float* __restrict__ cens,
                       const i8* __restrict__ kciP, const float* __restrict__ kcs,
                       float* __restrict__ thr, int NQ, int NTB, int NPAD, int kk, int sink_s,
                       int sink_e, float scale_log2) {
    extern __shared__ u32 skey[];
    __shared__ int scount[SOL_HD];
    __shared__ int sc8[SOL_HD / 4];
    const int qb = blockIdx.x, bh = blockIdx.y, tid = threadIdx.x;
    const i64 qrow = (i64)bh * NQ + qb;
    if (kk <= 0) {
        if (tid == 0) thr[qrow] = __int_as_float(0x7f800000);
        return;
    }
    if (tid < SOL_HD / 4) sc8[tid] = reinterpret_cast<const int*>(cen8 + qrow * SOL_HD)[tid];
    __syncthreads();
    const float qsc = cens[qrow] * scale_log2;
    for (int b = tid; b < NTB; b += SOL_HD) {
        const int* krow = reinterpret_cast<const int*>(kciP + ((i64)bh * NPAD + b) * SOL_HD);
        int dot = 0;
#pragma unroll 8
        for (int w = 0; w < SOL_HD / 4; ++w) dot = __dp4a(sc8[w], krow[w], dot);
        const float score = (float)dot * qsc * kcs[(i64)bh * NPAD + b];
        const bool sink = b >= sink_s && b < sink_e;
        skey[b] = sink ? 0u : sol_order_key(score);
    }
    __syncthreads();
    u32 prefix = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const u32 cand = prefix | (1u << bit);
        int c = 0;
        for (int b = tid; b < NTB; b += SOL_HD) c += skey[b] >= cand ? 1 : 0;
        scount[tid] = c;
        __syncthreads();
        for (int w = 64; w; w >>= 1) {
            if (tid < w) scount[tid] += scount[tid + w];
            __syncthreads();
        }
        if (scount[0] >= kk) prefix = cand;
        __syncthreads();
    }
    if (tid == 0) thr[qrow] = sol_order_value(prefix);
}

// ---- routing + approximate tail ----
// 64 query centroids per CTA (4 warps x 16); proxy QK on INT8 MMA, pooled P.V on bf16 MMA. Emits
// per (bh, query block in the window): the routed list, its length, and one online-softmax state
// (o / vsc and l both x255, the exact kernel's units).
#define SOL_R_LDV (SOL_BLK + 8)
extern "C" __global__ void __launch_bounds__(128)
    sol_route(const i8* __restrict__ cen8, const float* __restrict__ cens,
              const i8* __restrict__ kciP, const float* __restrict__ kcs,
              const u16* __restrict__ vcT, const float* __restrict__ vsc,
              const float* __restrict__ threshold, u16* __restrict__ blk_idx,
              int* __restrict__ blk_cnt, u16* __restrict__ o_part, float* __restrict__ m_part,
              float* __restrict__ l_part, int tail, int T, int NTB, int NPAD, int NQ, int qb0,
              int nqc, int sink_s, int sink_e, int sink_qs, int sink_qe, int diag_lo,
              int diag_span, float scale_log2) {
    constexpr int KC = SOL_HD / 32, NKT = SOL_BLK / 8, NT = SOL_HD / 8, PKC = SOL_BLK / 16;
    __shared__ __align__(16) i8 sKc[SOL_BLK * SOL_HD];
    __shared__ __align__(16) u16 sVcT[SOL_HD * SOL_R_LDV];
    __shared__ float sLen[SOL_BLK];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, qd = lane & 3;
    const int bh = blockIdx.y;
    const int l0 = blockIdx.x * SOL_BLK + warp * 16 + g;  // window-local query blocks
    const int lr[2] = {l0, l0 + 8};
    const bool live[2] = {lr[0] < nqc, lr[1] < nqc};
    const int qr[2] = {qb0 + lr[0], qb0 + lr[1]};  // absolute query blocks

    u32 qa[KC][4];
    float qsc[2], thr[2];
    bool q_in_sink[2];
    {
        const int r0 = qb0 + min(lr[0], nqc - 1), r1 = qb0 + min(lr[1], nqc - 1);
        const i8* p0 = cen8 + ((i64)bh * NQ + r0) * SOL_HD;
        const i8* p1 = cen8 + ((i64)bh * NQ + r1) * SOL_HD;
#pragma unroll
        for (int kc = 0; kc < KC; ++kc) {
            const int c0 = kc * 32 + qd * 8;
            const uint2 a0 = *reinterpret_cast<const uint2*>(p0 + c0);
            const uint2 a1 = *reinterpret_cast<const uint2*>(p1 + c0);
            qa[kc][0] = a0.x;
            qa[kc][2] = a0.y;
            qa[kc][1] = a1.x;
            qa[kc][3] = a1.y;
        }
        qsc[0] = live[0] ? cens[(i64)bh * NQ + r0] * scale_log2 : 0.f;
        qsc[1] = live[1] ? cens[(i64)bh * NQ + r1] * scale_log2 : 0.f;
        thr[0] = live[0] ? threshold[(i64)bh * NQ + r0] : 0.f;
        thr[1] = live[1] ? threshold[(i64)bh * NQ + r1] : 0.f;
        q_in_sink[0] = live[0] && qr[0] >= sink_qs && qr[0] < sink_qe;
        q_in_sink[1] = live[1] && qr[1] >= sink_qs && qr[1] < sink_qe;
    }

    // sink blocks lead every routed list
    const int S = max(0, min(sink_e, NTB) - sink_s);
    int cnt[2] = {live[0] ? S : 0, live[1] ? S : 0};
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        if (live[rr]) {
            u16* row = blk_idx + ((i64)bh * nqc + lr[rr]) * NTB;
            for (int i = qd; i < S; i += 4) row[i] = (u16)(sink_s + i);
        }
    }

    float o_acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) o_acc[nt][0] = o_acc[nt][1] = o_acc[nt][2] = o_acc[nt][3] = 0.f;
    float m_r[2] = {SOL_NEG, SOL_NEG}, l_r[2] = {0.f, 0.f};

    for (int gs = 0; gs < NTB; gs += SOL_BLK) {
        __syncthreads();
        for (int idx = tid; idx < SOL_BLK * (SOL_HD / 16); idx += 128) {
            const int p = idx / (SOL_HD / 16), c16 = idx % (SOL_HD / 16);
            ia_cp16_ca(sKc + p * SOL_HD + ((c16 ^ sol_swz_k(p)) << 4),
                       kciP + ((i64)bh * NPAD + gs + p) * SOL_HD + c16 * 16);
        }
        for (int idx = tid; idx < SOL_HD * (SOL_BLK / 8); idx += 128) {
            const int c = idx / (SOL_BLK / 8), part = idx % (SOL_BLK / 8);
            ia_cp16_ca(sVcT + c * SOL_R_LDV + part * 8,
                       vcT + ((i64)bh * SOL_HD + c) * NPAD + gs + part * 8);
        }
        if (tid < SOL_BLK) sLen[tid] = (gs + tid < NTB) ? (float)sol_block_len(gs + tid, T) : 0.f;
        ia_commit();
        ia_wait<0>();
        __syncthreads();

        int s_acc[NKT][4];
#pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
            s_acc[nt][0] = s_acc[nt][1] = s_acc[nt][2] = s_acc[nt][3] = 0;
            const int R = nt * 8 + g;
            const i8* krow = sKc + R * SOL_HD + ((qd & 1) << 3);
            const int swk = sol_swz_k(R), qhi = qd >> 1;
#pragma unroll
            for (int kc = 0; kc < KC; ++kc) {
                const uint2 kb = *reinterpret_cast<const uint2*>(krow + (((kc * 2 + qhi) ^ swk) << 4));
                ia_mma_s8(s_acc[nt], qa[kc], kb.x, kb.y);
            }
        }

        float pv[NKT][4];
#pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
            const int c0 = nt * 8 + qd * 2;
            const float ks0 = kcs[(i64)bh * NPAD + gs + c0];
            const float ks1 = kcs[(i64)bh * NPAD + gs + c0 + 1];
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                bool cand[2], pre[2], valid[2];
                float score[2];
#pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int b = gs + c0 + cc;
                    valid[cc] = live[rr] && b < NTB;
                    pre[cc] = valid[cc] && b >= sink_s && b < sink_e;
                    score[cc] = valid[cc] ? (float)s_acc[nt][rr * 2 + cc] * qsc[rr] * (cc ? ks1 : ks0)
                                          : SOL_NEG;
                    const bool routed =
                        valid[cc] && ((score[cc] >= thr[rr]) ||
                                      (unsigned)(b - qr[rr] - diag_lo) <= (unsigned)diag_span);
                    cand[cc] = !pre[cc] && (q_in_sink[rr] ? valid[cc] : routed);
                }
                int prefix = (int)cand[0] + (int)cand[1];
                int x = __shfl_up_sync(0xffffffffu, prefix, 1, 4);
                if (qd >= 1) prefix += x;
                x = __shfl_up_sync(0xffffffffu, prefix, 2, 4);
                if (qd >= 2) prefix += x;
                const int before = prefix - (int)cand[0] - (int)cand[1];
                const int total = __shfl_sync(0xffffffffu, prefix, 3, 4);
                const int slot0 = cnt[rr] + before;
                const int slot1 = slot0 + (int)cand[0];
                if (live[rr]) {
                    u16* row = blk_idx + ((i64)bh * nqc + lr[rr]) * NTB;
                    if (cand[0]) row[slot0] = (u16)(gs + c0);
                    if (cand[1]) row[slot1] = (u16)(gs + c0 + 1);
                }
                cnt[rr] += total;
                // exact-routed and sink blocks leave the tail; tail == 0 hands the exact kernel an
                // empty state (softmax over the routed blocks only)
                pv[nt][rr * 2] = tail && valid[0] && !(pre[0] || cand[0]) ? score[0] : SOL_NEG;
                pv[nt][rr * 2 + 1] = tail && valid[1] && !(pre[1] || cand[1]) ? score[1] : SOL_NEG;
            }
        }

        float m_new[2] = {m_r[0], m_r[1]};
#pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) m_new[e >> 1] = fmaxf(m_new[e >> 1], pv[nt][e]);
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            m_new[0] = fmaxf(m_new[0], __shfl_xor_sync(0xffffffffu, m_new[0], off));
            m_new[1] = fmaxf(m_new[1], __shfl_xor_sync(0xffffffffu, m_new[1], off));
        }
        const float alpha0 = exp2f(m_r[0] - m_new[0]);
        const float alpha1 = exp2f(m_r[1] - m_new[1]);
        m_r[0] = m_new[0];
        m_r[1] = m_new[1];

        float l_add[2] = {0.f, 0.f};
#pragma unroll
        for (int nt = 0; nt < NKT; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int rr = e >> 1;
                const int bl = nt * 8 + qd * 2 + (e & 1);
                const float p = pv[nt][e] <= SOL_NEG ? 0.f : exp2f(pv[nt][e] - m_new[rr]);
                pv[nt][e] = p;
                l_add[rr] += p * sLen[bl];
            }
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            l_add[0] += __shfl_xor_sync(0xffffffffu, l_add[0], off);
            l_add[1] += __shfl_xor_sync(0xffffffffu, l_add[1], off);
        }
        l_r[0] = l_r[0] * alpha0 + l_add[0];
        l_r[1] = l_r[1] * alpha1 + l_add[1];

        if (!tail) continue;
        u32 pa[PKC][4];
#pragma unroll
        for (int kk = 0; kk < PKC; ++kk) {
            pa[kk][0] = ia_pack2<true>(pv[2 * kk][0], pv[2 * kk][1]);
            pa[kk][1] = ia_pack2<true>(pv[2 * kk][2], pv[2 * kk][3]);
            pa[kk][2] = ia_pack2<true>(pv[2 * kk + 1][0], pv[2 * kk + 1][1]);
            pa[kk][3] = ia_pack2<true>(pv[2 * kk + 1][2], pv[2 * kk + 1][3]);
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            o_acc[nt][0] *= alpha0;
            o_acc[nt][1] *= alpha0;
            o_acc[nt][2] *= alpha1;
            o_acc[nt][3] *= alpha1;
            const u16* vcol = sVcT + (nt * 8 + g) * SOL_R_LDV;
#pragma unroll
            for (int kk = 0; kk < PKC; ++kk) {
                const u32 vb0 = *reinterpret_cast<const u32*>(vcol + kk * 16 + qd * 2);
                const u32 vb1 = *reinterpret_cast<const u32*>(vcol + kk * 16 + qd * 2 + 8);
                ia_mma_16<true>(o_acc[nt], pa[kk], vb0, vb1);
            }
        }
    }

#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        if (!live[rr]) continue;
        const i64 qs = (i64)bh * nqc + lr[rr];
        if (qd == 0) {
            blk_cnt[qs] = cnt[rr];
            m_part[qs] = m_r[rr];
            l_part[qs] = l_r[rr] * 255.0f;
        }
        u16* orow = o_part + qs * SOL_HD;
        const float* vsrow = vsc + (i64)bh * SOL_HD;
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int c = nt * 8 + qd * 2;
            orow[c] = sol_store16<true>(o_acc[nt][rr * 2] * (255.0f / vsrow[c]));
            orow[c + 1] = sol_store16<true>(o_acc[nt][rr * 2 + 1] * (255.0f / vsrow[c + 1]));
        }
    }
}

// ---- shared 64-key tile body of the exact kernel ----
// Staged K tile (64 x 128 int8) and V^T tile (128 x 64 int8), XOR-swizzled 16-byte chunks. One warp
// owns 16 rows: g = lane >> 2 picks rows g and g + 8, qd = lane & 3 the column pair.
#define SOL_TILE_KC (SOL_HD / 32)
#define SOL_TILE_NKT (SOL_BLK / 8)
#define SOL_TILE_NT (SOL_HD / 8)
#define SOL_TILE_PKC (SOL_BLK / 32)

// S = Q.K^T: the warp's 16 rows against the tile's 64 keys, int32 accumulators.
__device__ __forceinline__ void sol_tile_qk(const i8* sK_tile, const u32 (&qa)[SOL_TILE_KC][4],
                                            int g, int qd, int (&s_acc)[SOL_TILE_NKT][4]) {
#pragma unroll
    for (int nt = 0; nt < SOL_TILE_NKT; ++nt) {
        s_acc[nt][0] = s_acc[nt][1] = s_acc[nt][2] = s_acc[nt][3] = 0;
        const int R = nt * 8 + g;
        const i8* krow = sK_tile + R * SOL_HD + ((qd & 1) << 3);
        const int swk = sol_swz_k(R), qhi = qd >> 1;
#pragma unroll
        for (int kc = 0; kc < SOL_TILE_KC; ++kc) {
            const uint2 kb = *reinterpret_cast<const uint2*>(krow + (((kc * 2 + qhi) ^ swk) << 4));
            ia_mma_s8(s_acc[nt], qa[kc], kb.x, kb.y);
        }
    }
}

// Scores in log2 units: (int dot) * qsc[row] * ks + bias, with the tile's 64 (ks, bias) pairs at
// kb_src, and the row max folded into the same loop.
__device__ __forceinline__ void sol_tile_scores(const int (&s_acc)[SOL_TILE_NKT][4],
                                                const float2* kb_src, const float (&qsc)[2], int qd,
                                                float (&p_val)[SOL_TILE_NKT][4], float (&bmax)[2]) {
#pragma unroll
    for (int nt = 0; nt < SOL_TILE_NKT; ++nt) {
        const int c0 = nt * 8 + qd * 2;
        const float4 kb4 = *reinterpret_cast<const float4*>(kb_src + c0);
        const float k0s = kb4.x, m0 = kb4.y, k1s = kb4.z, m1 = kb4.w;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = e >> 1;
            const float s = (e & 1) ? fmaf((float)s_acc[nt][e], qsc[row] * k1s, m1)
                                    : fmaf((float)s_acc[nt][e], qsc[row] * k0s, m0);
            p_val[nt][e] = s;
            bmax[row] = fmaxf(bmax[row], s);
        }
    }
}

// Online softmax over the tile's scores and O += P.V. P is u8 against the block max bmax; l sums
// the packed bytes so numerator and denominator quantize identically. (m_r, l_r, c_r) is the row
// state, c_r the scale of o_acc and l_r. Routed blocks always hold live keys (no guard).
__device__ __forceinline__ void sol_tile_softmax_pv(float (&p_val)[SOL_TILE_NKT][4], float (&bmax)[2],
                                                    const i8* sVt_tile, int g, int qd,
                                                    float (&m_r)[2], float (&l_r)[2],
                                                    float (&c_r)[2],
                                                    float (&o_acc)[SOL_TILE_NT][4]) {
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        bmax[0] = fmaxf(bmax[0], __shfl_xor_sync(0xffffffffu, bmax[0], off));
        bmax[1] = fmaxf(bmax[1], __shfl_xor_sync(0xffffffffu, bmax[1], off));
    }
    const float alpha0 = exp2f(c_r[0] - bmax[0]);
    const float alpha1 = exp2f(c_r[1] - bmax[1]);
    c_r[0] = bmax[0];
    c_r[1] = bmax[1];
    m_r[0] = fmaxf(m_r[0], bmax[0]);
    m_r[1] = fmaxf(m_r[1], bmax[1]);

    // u8 P scale folded into the exponent (+log2 255); l carries it too
    const float m_off[2] = {bmax[0] - 7.99435344f, bmax[1] - 7.99435344f};
#pragma unroll
    for (int nt = 0; nt < SOL_TILE_NKT; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) p_val[nt][e] = exp2f(p_val[nt][e] - m_off[e >> 1]);
    }

    // free repack (perm_key): n-tiles (4kk, 4kk+1) -> keys 32kk+4q..+3
    u32 pa[SOL_TILE_PKC][4];
#pragma unroll
    for (int kk = 0; kk < SOL_TILE_PKC; ++kk) {
        const int b0 = 4 * kk, b1 = b0 + 1, b2 = b0 + 2, b3 = b0 + 3;
        pa[kk][0] = ia_pack_u8x4(p_val[b0][0], p_val[b0][1], p_val[b1][0], p_val[b1][1]);
        pa[kk][1] = ia_pack_u8x4(p_val[b0][2], p_val[b0][3], p_val[b1][2], p_val[b1][3]);
        pa[kk][2] = ia_pack_u8x4(p_val[b2][0], p_val[b2][1], p_val[b3][0], p_val[b3][1]);
        pa[kk][3] = ia_pack_u8x4(p_val[b2][2], p_val[b2][3], p_val[b3][2], p_val[b3][3]);
    }
    u32 li[2] = {0, 0};
#pragma unroll
    for (int kk = 0; kk < SOL_TILE_PKC; ++kk) {
        li[0] = __dp4a(pa[kk][0], 0x01010101u, li[0]);
        li[0] = __dp4a(pa[kk][2], 0x01010101u, li[0]);
        li[1] = __dp4a(pa[kk][1], 0x01010101u, li[1]);
        li[1] = __dp4a(pa[kk][3], 0x01010101u, li[1]);
    }
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
        li[0] += __shfl_xor_sync(0xffffffffu, li[0], off);
        li[1] += __shfl_xor_sync(0xffffffffu, li[1], off);
    }
    l_r[0] = l_r[0] * alpha0 + (float)(int)li[0];
    l_r[1] = l_r[1] * alpha1 + (float)(int)li[1];

#pragma unroll
    for (int nt = 0; nt < SOL_TILE_NT; ++nt) {
        int d[4] = {0, 0, 0, 0};
        const int C = nt * 8 + g;
        const i8* vcol = sVt_tile + C * SOL_BLK + ((qd & 1) << 3);
        const int swv = sol_swz_v(C), qhi2 = qd >> 1;
#pragma unroll
        for (int kk = 0; kk < SOL_TILE_PKC; ++kk) {
            const uint2 vb = *reinterpret_cast<const uint2*>(vcol + (((kk * 2 + qhi2) ^ swv) << 4));
            ia_mma_u8s8(d, pa[kk], vb.x, vb.y);
        }
        o_acc[nt][0] = fmaf(o_acc[nt][0], alpha0, (float)d[0]);
        o_acc[nt][1] = fmaf(o_acc[nt][1], alpha0, (float)d[1]);
        o_acc[nt][2] = fmaf(o_acc[nt][2], alpha1, (float)d[2]);
        o_acc[nt][3] = fmaf(o_acc[nt][3], alpha1, (float)d[3]);
    }
}

// ---- exact branch: each query block walks its routed key-block list ----
// comfy-kitchen bounds sm_120 at 3 CTAs/SM (nvcc fits it without spilling); under nvrtc that cap
// (170 registers) spills 8 B/thread, so sm_120 is bounded at 2 CTAs/SM (zero local memory, the
// seam's rule); elsewhere unbounded.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define SOL_EXACT_BOUNDS __launch_bounds__(128, 2)
#else
#define SOL_EXACT_BOUNDS __launch_bounds__(128)
#endif
#define SOL_NSTAGE 2

template <bool BF>
__device__ void sol_exact(const i8* __restrict__ qi, const float* __restrict__ qs,
                          const i8* __restrict__ kiP, const float2* __restrict__ ksb,
                          const i8* __restrict__ vTi, const float* __restrict__ vsc,
                          const u16* __restrict__ blk_idx, const int* __restrict__ blk_cnt,
                          const u16* __restrict__ o_part, const float* __restrict__ m_part,
                          const float* __restrict__ l_part, u16* __restrict__ out, int T, int Tp,
                          int NTB, int qb0, int nqc, float scale_log2) {
    __shared__ __align__(16) i8 sK[SOL_NSTAGE * SOL_BLK * SOL_HD];
    __shared__ __align__(16) i8 sVt[SOL_NSTAGE * SOL_HD * SOL_BLK];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, qd = lane & 3;
    const int lq = blockIdx.x, bh = blockIdx.y;  // window-local query block
    const i64 vT_base = (i64)bh * SOL_HD * Tp;
    const i64 kp_base = (i64)bh * Tp;
    const i64 kd_base = (i64)bh * Tp * SOL_HD;
    const int q_row0 = (qb0 + lq) * SOL_BLK + warp * 16 + g;

    u32 qa[SOL_TILE_KC][4];
    float qsc[2];
    {
        const int r0 = min(q_row0, T - 1), r1 = min(q_row0 + 8, T - 1);
        const i8* p0 = qi + ((i64)bh * T + r0) * SOL_HD;
        const i8* p1 = qi + ((i64)bh * T + r1) * SOL_HD;
#pragma unroll
        for (int kc = 0; kc < SOL_TILE_KC; ++kc) {
            const int c0 = kc * 32 + qd * 8;  // perm_d put a0 and a2 side by side
            const uint2 a0 = *reinterpret_cast<const uint2*>(p0 + c0);
            const uint2 a1 = *reinterpret_cast<const uint2*>(p1 + c0);
            qa[kc][0] = a0.x;
            qa[kc][2] = a0.y;
            qa[kc][1] = a1.x;
            qa[kc][3] = a1.y;
        }
        qsc[0] = qs[(i64)bh * T + r0] * scale_log2;
        qsc[1] = qs[(i64)bh * T + r1] * scale_log2;
    }

    const i64 qb_s = (i64)bh * nqc + lq;
    const u16* my_idx = blk_idx + qb_s * NTB;
    const int n_blocks = blk_cnt[qb_s];

    // resume route's state: one (o, m, l) per (bh, query block)
    float o_acc[SOL_TILE_NT][4];
    float m_r[2], l_r[2], c_r[2];
    {
        const u16* orow = o_part + qb_s * SOL_HD;
#pragma unroll
        for (int nt = 0; nt < SOL_TILE_NT; ++nt) {
            const int c = nt * 8 + qd * 2;
            const float v0 = ia_load(orow + c, true), v1 = ia_load(orow + c + 1, true);
            o_acc[nt][0] = v0;
            o_acc[nt][2] = v0;
            o_acc[nt][1] = v1;
            o_acc[nt][3] = v1;
        }
        m_r[0] = m_r[1] = m_part[qb_s];
        c_r[0] = c_r[1] = m_part[qb_s];
        l_r[0] = l_r[1] = l_part[qb_s];
    }

    auto stage = [&](int kbi, int buf) {
        const i64 k0 = (i64)kbi * SOL_BLK;
        i8* sk = sK + buf * SOL_BLK * SOL_HD;
        i8* sv = sVt + buf * SOL_HD * SOL_BLK;
        for (int idx = tid; idx < SOL_BLK * (SOL_HD / 16); idx += 128) {
            const int p = idx / (SOL_HD / 16), c16 = idx % (SOL_HD / 16);
            ia_cp16(sk + p * SOL_HD + ((c16 ^ sol_swz_k(p)) << 4), kiP + kd_base + (k0 + p) * SOL_HD + c16 * 16, 16);
        }
        for (int idx = tid; idx < SOL_HD * 4; idx += 128) {
            const int c = idx >> 2, part = idx & 3;
            ia_cp16(sv + c * SOL_BLK + ((part ^ sol_swz_v(c)) << 4), vTi + vT_base + (i64)c * Tp + k0 + part * 16, 16);
        }
    };

#pragma unroll
    for (int st = 0; st < SOL_NSTAGE - 1; ++st) {
        if (st < n_blocks) stage(my_idx[st], st);
        ia_commit();
    }
    for (int kb = 0; kb < n_blocks; ++kb) {
        const int cur = kb % SOL_NSTAGE;
        const i64 cur_k0 = (i64)my_idx[kb] * SOL_BLK;
        const int ahead = kb + SOL_NSTAGE - 1;
        if (ahead < n_blocks) stage(my_idx[ahead], ahead % SOL_NSTAGE);
        ia_commit();
        ia_wait<SOL_NSTAGE - 1>();
        __syncthreads();
        int s_acc[SOL_TILE_NKT][4];
        sol_tile_qk(sK + cur * SOL_BLK * SOL_HD, qa, g, qd, s_acc);
        float p_val[SOL_TILE_NKT][4];
        float bmax[2] = {m_r[0] - 20.f, m_r[1] - 20.f};  // floor: 2^-20 under the running max
        sol_tile_scores(s_acc, ksb + kp_base + cur_k0, qsc, qd, p_val, bmax);
        sol_tile_softmax_pv(p_val, bmax, sVt + cur * SOL_HD * SOL_BLK, g, qd, m_r, l_r, c_r, o_acc);
        __syncthreads();  // next iteration refills `cur`
    }

    // l is zero only when a row got no mass from either branch; zeros are right there
    const float inv0 = 1.f / fmaxf(l_r[0], 1e-30f);
    const float inv1 = 1.f / fmaxf(l_r[1], 1e-30f);
    const float* vs = vsc + (i64)bh * SOL_HD;
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const int r = q_row0 + rr * 8;
        if (r >= T) continue;
        const float inv = rr ? inv1 : inv0;
        u16* orow = out + ((i64)bh * T + r) * SOL_HD;
#pragma unroll
        for (int nt = 0; nt < SOL_TILE_NT; ++nt) {
            const int c = nt * 8 + qd * 2;
            const u32 lo = sol_store16<BF>(o_acc[nt][rr * 2] * inv * vs[c]);
            const u32 hi = sol_store16<BF>(o_acc[nt][rr * 2 + 1] * inv * vs[c + 1]);
            *reinterpret_cast<u32*>(orow + c) = lo | (hi << 16);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Entry points. The suffix selects bf16 or f16 operands.
// ---------------------------------------------------------------------------------------------

#define SOL_ENTRIES(SUFFIX, BF)                                                                    \
    extern "C" __global__ void __launch_bounds__(SOL_HD) sol_reduce_kv_##SUFFIX(                   \
        const u16* k, const u16* v, float* kc, u16* vcT, float* vamax, int T, int H, int NTB,      \
        int NPAD, i64 sb, i64 st, i64 sh, i64 vb, i64 vt, i64 vh) {                                \
        sol_reduce_kv<BF>(k, v, kc, vcT, vamax, T, H, NTB, NPAD, sb, st, sh, vb, vt, vh);          \
    }                                                                                              \
    extern "C" __global__ void __launch_bounds__(SOL_HD) sol_prep_q_##SUFFIX(                      \
        const u16* q, const float* kcvar, i8* qiP, float* qs, float* thr, i8* cen8, float* cens,   \
        int T, int H, int NQ, float tau, float log2s, i64 sb, i64 st, i64 sh) {                    \
        sol_prep_q<BF>(q, kcvar, qiP, qs, thr, cen8, cens, T, H, NQ, tau, log2s, sb, st, sh);      \
    }                                                                                              \
    extern "C" __global__ void __launch_bounds__(SOL_HD) sol_prep_k_##SUFFIX(                      \
        const u16* k, const float* kmean, i8* kiP, float2* ksb, int T, int Tp, int H, i64 sb,      \
        i64 st, i64 sh) {                                                                          \
        sol_prep_k<BF>(k, kmean, kiP, ksb, T, Tp, H, sb, st, sh);                                  \
    }                                                                                              \
    extern "C" __global__ void __launch_bounds__(256) sol_vtranspose_##SUFFIX(                     \
        const u16* v, const float* vsc, i8* vT, int T, int Tp, int H, i64 sb, i64 st, i64 sh) {    \
        sol_vtranspose<BF>(v, vsc, vT, T, Tp, H, sb, st, sh);                                      \
    }                                                                                              \
    extern "C" __global__ void SOL_EXACT_BOUNDS sol_exact_##SUFFIX(                                \
        const i8* qi, const float* qs, const i8* kiP, const float2* ksb, const i8* vTi,            \
        const float* vsc, const u16* blk_idx, const int* blk_cnt, const u16* o_part,               \
        const float* m_part, const float* l_part, u16* out, int T, int Tp, int NTB, int qb0,       \
        int nqc, float scale_log2) {                                                               \
        sol_exact<BF>(qi, qs, kiP, ksb, vTi, vsc, blk_idx, blk_cnt, o_part, m_part, l_part, out,  \
                      T, Tp, NTB, qb0, nqc, scale_log2);                                           \
    }

SOL_ENTRIES(bf16, true)
SOL_ENTRIES(f16, false)
