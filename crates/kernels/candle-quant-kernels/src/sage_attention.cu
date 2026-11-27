// SW-6131 (SW-F287 S6): SageAttention-style INT8 QK attention, dense and unmasked.
//
// Ported design, single source for the nvrtc seam (no includes). Attribution (E10):
//   * comfy-kitchen @ 6022b48b7e39fdd31f0124de268a8267af46d4e0
//     comfy_kitchen/backends/cuda/sage_attention/{quant_qk_int8.cu, sage_attn_launcher.cu,
//     qk_int_sv_i8_cuda.cuh} - Apache-2.0, Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//     Taken from it: the fused signed block-Hadamard (H4 in-thread + lane butterflies) rotation of
//     Q and K before INT8 quantization, its fixed sign diagonal (the four 32-bit sign words below
//     are comfy-kitchen's `apply_convrot_sign128` constants), the softmax-invariant K centering,
//     and the softmax-scale-times-log2(e) fold into the Q scale.
//   * thu-ml/SageAttention @ d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 - Apache-2.0,
//     Copyright (c) 2024 Jintao Zhang, Haofeng Huang et al. The algorithm: INT8 Q.K^T on the
//     integer tensor cores with fp32 softmax and a 16-bit P.V ("qk_int8_pv_fp16").
// See candle-quant-kernels/src/sage_attention-NOTICE.md and docs/licensing/third-party-kernel-sources.md.
//
// Deliberate differences from comfy-kitchen's `qk_int_sv_i8` path (story SW-6131 scope):
//   * V stays bf16/f16 and P.V runs on the 16-bit tensor cores with fp32 accumulation (comfy-kitchen
//     also quantizes V and P to INT8); only Q.K^T is lossy.
//   * Q and K are quantized per token (one scale per row) - a finer block than comfy-kitchen's
//     per-thread groups; the K centering offset is the per-channel mean of K over the sequence.
//   * Head dims 64 and 128 only (others are refused by the host plan before launch).
//
// Layouts (element strides are host-validated; every pointer is 16-byte aligned):
//   x (Q or K input) : bf16/f16 [B, H, S, D] with arbitrary batch/head/row strides, unit last dim.
//   codes            : int8 [B, H, S_pad, D] contiguous; rows >= S are zero.
//   scales           : f32 [B, H, S_pad]; rows >= S are zero.
//   V                : bf16/f16 [B, HK, Sk, D] with arbitrary batch/head/row strides, unit last dim.
//   O                : bf16/f16 [B, H, Sq, D] contiguous.
// Every index is computed in 64 bits.

typedef unsigned int u32;
typedef unsigned short u16;
typedef signed char i8;
typedef long long i64;

#define SAGE_NEG_INF __int_as_float(0xff800000)

// comfy-kitchen `apply_convrot_sign128` words: channel c is negated when bit (c & 31) of word
// (c >> 5) is clear.
__device__ __forceinline__ u32 sage_sign_word(int w) {
    return w == 0 ? 0x1035997bu : w == 1 ? 0x8087f5eeu : w == 2 ? 0xee2e4e1au : 0x71132418u;
}

__device__ __forceinline__ float sage_load(const u16* p, bool bf) {
    u16 h = *p;
    if (bf) return __uint_as_float(((u32)h) << 16);
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}

// Pack (lo, hi) into a bf16x2 / f16x2 register, round to nearest even.
template <bool BF>
__device__ __forceinline__ u32 sage_pack2(float lo, float hi) {
    u32 r;
    if (BF) asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    else asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}

// 2^x on the SFU (ex2.approx, ~2 ulp; exactly 0 for -inf), as FlashAttention-2's fast-math build.
__device__ __forceinline__ float sage_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float sage_warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float sage_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// One row of D channels held as D/32 consecutive channels per lane (channel = lane*EPT + c).
// Applies the sign diagonal, then the normalized Walsh-Hadamard transform: an in-thread H_EPT on
// the low channel bits, then five xor-shuffle butterflies on the lane bits. Orthogonal, so Q and K
// rotated alike keep every dot product.
template <int D>
__device__ __forceinline__ void sage_rotate(float* v, int lane) {
    constexpr int EPT = D / 32;
#pragma unroll
    for (int c = 0; c < EPT; ++c) {
        const int ch = lane * EPT + c;
        const u32 flip = ((sage_sign_word(ch >> 5) >> (ch & 31)) & 1u) ^ 1u;
        v[c] = __uint_as_float(__float_as_uint(v[c]) ^ (flip << 31));
    }
    if (EPT == 4) {
        const float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
        v[0] = a0 + a2;
        v[1] = a1 + a3;
        v[2] = a0 - a2;
        v[3] = a1 - a3;
    } else {
        const float a0 = v[0] + v[1], a1 = v[0] - v[1];
        v[0] = a0;
        v[1] = a1;
    }
#pragma unroll
    for (int bit = 1; bit < 32; bit <<= 1) {
#pragma unroll
        for (int c = 0; c < EPT; ++c) {
            const float other = __shfl_xor_sync(0xffffffffu, v[c], bit);
            v[c] = (lane & bit) ? other - v[c] : v[c] + other;
        }
    }
    const float norm = D == 128 ? __int_as_float(0x3db504f3) : 0.125f;  // 1/sqrt(128), 1/sqrt(64)
#pragma unroll
    for (int c = 0; c < EPT; ++c) v[c] *= norm;
}

// Load one row (zero beyond S), optionally minus the per-channel mean.
template <int D>
__device__ __forceinline__ void sage_load_row(const u16* x, i64 row_off, bool live, bool bf,
                                              const float* mean, bool center, int lane, float* v) {
    constexpr int EPT = D / 32;
#pragma unroll
    for (int c = 0; c < EPT; ++c) {
        v[c] = live ? sage_load(x + row_off + lane * EPT + c, bf) : 0.0f;
        if (center && live) v[c] -= mean[lane * EPT + c];
    }
}

// Per-channel partial sums of K over one chunk of rows (the centering mean, before the fixed-order
// chunk reduction in `sage_k_center`; no atomics, so the mean is deterministic). One thread per
// channel. grid (B*H, chunks); `partial` is [B*H, chunks, D].
template <int D>
__device__ void sage_k_sum(const u16* x, i64 sb, i64 sh, i64 ss, int H, int S, int rows_per_cta,
                           bool bf, float* partial) {
    const int bh = blockIdx.x, b = bh / H, h = bh - b * H, c = threadIdx.x;
    const int r0 = blockIdx.y * rows_per_cta;
    const int r1 = min(S, r0 + rows_per_cta);
    const i64 base = (i64)b * sb + (i64)h * sh + c;
    float acc = 0.0f;
#pragma unroll 8
    for (int r = r0; r < r1; ++r) acc += sage_load(x + base + (i64)r * ss, bf);
    partial[((i64)bh * gridDim.y + blockIdx.y) * D + c] = acc;
}

// The per-channel mean (the chunk partial sums reduced in a fixed order) and the centering decision
// per (b, head): `adaptive` samples up to 256 rows and engages centering when the summed per-row
// rotated range shrinks; otherwise it is always engaged. Writes `mean[bh]` and `flag[bh]`.
// grid (B*H), block 256.
template <int D>
__device__ void sage_k_center(const u16* x, i64 sb, i64 sh, i64 ss, int H, int S, bool bf,
                              int adaptive, const float* partial, int chunks, float* mean,
                              int* flag) {
    constexpr int EPT = D / 32;
    __shared__ float mean_s[D];
    __shared__ float red[2][8];
    const int bh = blockIdx.x, b = bh / H, h = bh - b * H;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const float inv_s = 1.0f / (float)S;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < chunks; ++j) sum += partial[((i64)bh * chunks + j) * D + c];
        mean_s[c] = sum * inv_s;
    }
    __syncthreads();
    if (adaptive) {
        const int samples = min(S, 256);
        float raw = 0.0f, cen = 0.0f;
        for (int i = warp; i < samples; i += 8) {
            const int r = (int)(((i64)i * S) / samples);
            const i64 off = (i64)b * sb + (i64)h * sh + (i64)r * ss;
            float v[EPT], w[EPT];
            sage_load_row<D>(x, off, true, bf, mean_s, false, lane, v);
#pragma unroll
            for (int c = 0; c < EPT; ++c) w[c] = v[c] - mean_s[lane * EPT + c];
            sage_rotate<D>(v, lane);
            sage_rotate<D>(w, lane);
            float av = 0.0f, aw = 0.0f;
#pragma unroll
            for (int c = 0; c < EPT; ++c) {
                av = fmaxf(av, fabsf(v[c]));
                aw = fmaxf(aw, fabsf(w[c]));
            }
            raw += sage_warp_max(av);
            cen += sage_warp_max(aw);
        }
        if (lane == 0) {
            red[0][warp] = raw;
            red[1][warp] = cen;
        }
    }
    __syncthreads();
    if (threadIdx.x < D) mean[(i64)bh * D + threadIdx.x] = mean_s[threadIdx.x];
    if (threadIdx.x == 0) {
        int on = 1;
        if (adaptive) {
            float raw = 0.0f, cen = 0.0f;
            for (int w = 0; w < 8; ++w) {
                raw += red[0][w];
                cen += red[1][w];
            }
            on = cen < raw ? 1 : 0;
        }
        flag[bh] = on;
    }
}

// Rotate + per-row INT8 quantization of Q or K. One warp per row, 8 rows per CTA.
// grid (S_pad / 8, H, B), block 256. `pre` multiplies only the stored scale (Q: softmax scale *
// log2(e); K: 1). `center`: 0 = off, 1 = subtract `mean[bh]` where `flag[bh]` is set.
template <int D>
__device__ void sage_quant(const u16* x, i64 sb, i64 sh, i64 ss, int H, int S, int s_pad, bool bf,
                           float pre, const float* mean, const int* flag, int center, i8* codes,
                           float* scales) {
    constexpr int EPT = D / 32;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int h = blockIdx.y, b = blockIdx.z, bh = b * H + h;
    if (row >= s_pad) return;
    const bool live = row < S;
    const bool cen = center != 0 && flag[bh] != 0;
    float v[EPT];
    sage_load_row<D>(x, (i64)b * sb + (i64)h * sh + (i64)row * ss, live, bf, mean + (i64)bh * D, cen,
                     lane, v);
    sage_rotate<D>(v, lane);
    float amax = 0.0f;
#pragma unroll
    for (int c = 0; c < EPT; ++c) amax = fmaxf(amax, fabsf(v[c]));
    amax = sage_warp_max(amax);
    const float step = amax / 127.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    u32 packed = 0;
#pragma unroll
    for (int c = 0; c < EPT; ++c) {
        int q = __float2int_rn(v[c] * inv);
        q = max(-127, min(127, q));
        packed |= ((u32)(q & 0xff)) << (8 * c);
    }
    const i64 out = ((i64)bh * s_pad + row) * D + lane * EPT;
    if (EPT == 4) *(u32*)(codes + out) = packed;
    else *(u16*)(codes + out) = (u16)packed;
    if (lane == 0) scales[(i64)bh * s_pad + row] = step * pre;
}

// ---------------------------------------------------------------------------------------------
// The attention kernel: one 256-thread CTA per (128 query rows, head, batch); warp w owns query
// rows 16w..16w+15. Per 64-key block: S = Q.K^T on mma.m16n8k32.s8 (exact int32), dequantized by
// the per-row Q scale and per-key K scale (log2 domain), online softmax in fp32 with exp2, P
// rounded to the V dtype and P.V on mma.m16n8k16 with fp32 accumulation. K and V tiles stream
// through shared memory with cp.async, FlashAttention-2's K/V overlap.
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ u32 sage_smem(const void* p) {
    return (u32)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void sage_cp16(void* dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sage_smem(dst)), "l"(src),
                 "r"(bytes));
}

__device__ __forceinline__ void sage_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void sage_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void sage_ldm4(u32* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sage_smem(p)));
}

__device__ __forceinline__ void sage_ldm4t(u32* r, const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sage_smem(p)));
}

__device__ __forceinline__ void sage_mma_s8(int* c, const u32* a, u32 b0, u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <bool BF>
__device__ __forceinline__ void sage_mma_16(float* c, const u32* a, u32 b0, u32 b1) {
    if (BF)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    else
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
            "{%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

#define SAGE_BLOCK_N 64

template <int D, bool BF>
__device__ void sage_attn(const i8* __restrict__ q, const float* __restrict__ qs,
                          const i8* __restrict__ k, const float* __restrict__ ks,
                          const u16* __restrict__ v, i64 vsb, i64 vsh, i64 vss,
                          u16* __restrict__ o, int H, int HK, int sq, int sq_pad, int sk,
                          int sk_pad) {
    constexpr int KSTR = D + 16;     // bytes per K smem row (ldmatrix rows on distinct banks)
    constexpr int VSTR = D + 8;      // u16 elements per V smem row
    constexpr int KT = D / 32;       // int8 k-steps of 32 channels
    constexpr int NT = D / 8;        // output n-tiles of 8 channels
    __shared__ __align__(16) i8 k_s[SAGE_BLOCK_N * KSTR];
    __shared__ __align__(16) u16 v_s[SAGE_BLOCK_N * VSTR];
    __shared__ __align__(16) float ksc_s[SAGE_BLOCK_N];

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int h = blockIdx.y, b = blockIdx.z;
    const int hk = h / (H / HK);
    const i64 bh = (i64)b * H + h, bhk = (i64)b * HK + hk;
    const int r0 = blockIdx.x * 128 + warp * 16;

    // Q fragments for the whole head dim, straight from global (A layout of m16n8k32).
    u32 qa[KT][4];
    {
        const i8* q0 = q + (bh * sq_pad + r0 + g) * D + 4 * t;
        const i8* q1 = q0 + 8 * D;
#pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            qa[kt][0] = *(const u32*)(q0 + kt * 32);
            qa[kt][1] = *(const u32*)(q1 + kt * 32);
            qa[kt][2] = *(const u32*)(q0 + kt * 32 + 16);
            qa[kt][3] = *(const u32*)(q1 + kt * 32 + 16);
        }
    }
    const float qs0 = qs[bh * sq_pad + r0 + g];
    const float qs1 = qs[bh * sq_pad + r0 + g + 8];

    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    float m0 = SAGE_NEG_INF, m1 = SAGE_NEG_INF, l0 = 0.0f, l1 = 0.0f;

    const int blocks = sk_pad / SAGE_BLOCK_N;
    const i8* kbase = k + bhk * (i64)sk_pad * D;
    const float* ksbase = ks + bhk * (i64)sk_pad;
    const u16* vbase = v + (i64)b * vsb + (i64)hk * vsh;

    auto load_k = [&](int blk) {
        constexpr int CHUNKS = SAGE_BLOCK_N * D / 16;
#pragma unroll
        for (int i = tid; i < CHUNKS; i += 256) {
            const int r = i / (D / 16), c = i - r * (D / 16);
            sage_cp16(k_s + r * KSTR + c * 16, kbase + ((i64)blk * SAGE_BLOCK_N + r) * D + c * 16, 16);
        }
        if (tid < SAGE_BLOCK_N / 4)
            sage_cp16(ksc_s + tid * 4, ksbase + (i64)blk * SAGE_BLOCK_N + tid * 4, 16);
    };
    auto load_v = [&](int blk) {
        constexpr int CHUNKS = SAGE_BLOCK_N * D / 8;
#pragma unroll
        for (int i = tid; i < CHUNKS; i += 256) {
            const int r = i / (D / 8), c = i - r * (D / 8);
            const int key = blk * SAGE_BLOCK_N + r;
            const bool live = key < sk;
            const u16* src = vbase + (i64)(live ? key : 0) * vss + c * 8;
            sage_cp16(v_s + r * VSTR + c * 8, src, live ? 16 : 0);
        }
    };

    load_k(0);
    sage_commit();
    for (int blk = 0; blk < blocks; ++blk) {
        load_v(blk);
        sage_commit();
        sage_wait<1>();
        __syncthreads();

        // S = Q.K^T for this warp's 16 rows x 64 keys.
        int s[8][4];
#pragma unroll
        for (int n = 0; n < 8; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int kc = 0; kc < KT / 2; ++kc) {
                u32 bf[4];
                sage_ldm4(bf, k_s + (n * 8 + (lane & 7)) * KSTR + kc * 64 + (lane >> 3) * 16);
                sage_mma_s8(s[n], qa[2 * kc], bf[0], bf[1]);
                sage_mma_s8(s[n], qa[2 * kc + 1], bf[2], bf[3]);
            }
        }
        float ksc[8][2];
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            ksc[n][0] = ksc_s[n * 8 + 2 * t];
            ksc[n][1] = ksc_s[n * 8 + 2 * t + 1];
        }
        __syncthreads();  // every warp is done with k_s / ksc_s
        if (blk + 1 < blocks) {
            load_k(blk + 1);
            sage_commit();
        }

        // Online softmax (log2 domain: the Q scale carries softmax_scale * log2(e)).
        float p[8][4];
        const int key0 = blk * SAGE_BLOCK_N;
        const bool tail = key0 + SAGE_BLOCK_N > sk;
        float mx0 = SAGE_NEG_INF, mx1 = SAGE_NEG_INF;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const bool dead = tail && key0 + n * 8 + 2 * t + j >= sk;
                p[n][j] = dead ? SAGE_NEG_INF : (float)s[n][j] * qs0 * ksc[n][j];
                p[n][2 + j] = dead ? SAGE_NEG_INF : (float)s[n][2 + j] * qs1 * ksc[n][j];
                mx0 = fmaxf(mx0, p[n][j]);
                mx1 = fmaxf(mx1, p[n][2 + j]);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float a0 = sage_exp2(m0 - n0), a1 = sage_exp2(m1 - n1);
        m0 = n0;
        m1 = n1;
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            p[n][0] = sage_exp2(p[n][0] - n0);
            p[n][1] = sage_exp2(p[n][1] - n0);
            p[n][2] = sage_exp2(p[n][2] - n1);
            p[n][3] = sage_exp2(p[n][3] - n1);
            sum0 += p[n][0] + p[n][1];
            sum1 += p[n][2] + p[n][3];
        }
        l0 = l0 * a0 + sum0;
        l1 = l1 * a1 + sum1;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            acc[n][0] *= a0;
            acc[n][1] *= a0;
            acc[n][2] *= a1;
            acc[n][3] *= a1;
        }
        u32 pa[4][4];
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            pa[kk][0] = sage_pack2<BF>(p[2 * kk][0], p[2 * kk][1]);
            pa[kk][1] = sage_pack2<BF>(p[2 * kk][2], p[2 * kk][3]);
            pa[kk][2] = sage_pack2<BF>(p[2 * kk + 1][0], p[2 * kk + 1][1]);
            pa[kk][3] = sage_pack2<BF>(p[2 * kk + 1][2], p[2 * kk + 1][3]);
        }

        if (blk + 1 < blocks) sage_wait<1>();
        else sage_wait<0>();
        __syncthreads();  // V(blk) landed for every thread's copies

        // O += P.V for this warp's 16 rows x D channels.
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int np = 0; np < NT / 2; ++np) {
                u32 bv[4];
                sage_ldm4t(bv, v_s + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * VSTR + np * 16 +
                                   (lane >> 4) * 8);
                sage_mma_16<BF>(acc[2 * np], pa[kk], bv[0], bv[1]);
                sage_mma_16<BF>(acc[2 * np + 1], pa[kk], bv[2], bv[3]);
            }
        }
        __syncthreads();  // every warp is done with v_s before the next V copy
    }

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float inv0 = 1.0f / l0, inv1 = 1.0f / l1;
    const int row0 = r0 + g, row1 = r0 + g + 8;
    u16* o0 = o + (bh * sq + row0) * D + 2 * t;
    u16* o1 = o + (bh * sq + row1) * D + 2 * t;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        if (row0 < sq) *(u32*)(o0 + n * 8) = sage_pack2<BF>(acc[n][0] * inv0, acc[n][1] * inv0);
        if (row1 < sq) *(u32*)(o1 + n * 8) = sage_pack2<BF>(acc[n][2] * inv1, acc[n][3] * inv1);
    }
}

// ---------------------------------------------------------------------------------------------
// Entry points. `bf` / the name suffix select bf16 (1) or f16 (0).
// ---------------------------------------------------------------------------------------------

#define SAGE_PREP(D)                                                                              \
    extern "C" __global__ void __launch_bounds__(D) sage_k_sum_d##D(                             \
        const u16* x, i64 sb, i64 sh, i64 ss, int H, int S, int rows, int bf, float* partial) {  \
        sage_k_sum<D>(x, sb, sh, ss, H, S, rows, bf != 0, partial);                               \
    }                                                                                             \
    extern "C" __global__ void __launch_bounds__(256) sage_k_center_d##D(                        \
        const u16* x, i64 sb, i64 sh, i64 ss, int H, int S, int bf, int adaptive,                 \
        const float* partial, int chunks, float* mean, int* flag) {                               \
        sage_k_center<D>(x, sb, sh, ss, H, S, bf != 0, adaptive, partial, chunks, mean, flag);    \
    }                                                                                             \
    extern "C" __global__ void __launch_bounds__(256) sage_quant_d##D(                           \
        const u16* x, i64 sb, i64 sh, i64 ss, int H, int S, int s_pad, int bf, float pre,         \
        const float* mean, const int* flag, int center, i8* codes, float* scales) {               \
        sage_quant<D>(x, sb, sh, ss, H, S, s_pad, bf != 0, pre, mean, flag, center, codes,        \
                      scales);                                                                    \
    }

#define SAGE_ATTN(D, BF, SUFFIX)                                                                  \
    extern "C" __global__ void __launch_bounds__(256, 1) sage_attn_d##D##_##SUFFIX(              \
        const i8* q, const float* qs, const i8* k, const float* ks, const u16* v, i64 vsb,        \
        i64 vsh, i64 vss, u16* o, int H, int HK, int sq, int sq_pad, int sk, int sk_pad) {        \
        sage_attn<D, BF>(q, qs, k, ks, v, vsb, vsh, vss, o, H, HK, sq, sq_pad, sk, sk_pad);       \
    }

SAGE_PREP(64)
SAGE_PREP(128)
SAGE_ATTN(64, true, bf16)
SAGE_ATTN(64, false, f16)
SAGE_ATTN(128, true, bf16)
SAGE_ATTN(128, false, f16)
