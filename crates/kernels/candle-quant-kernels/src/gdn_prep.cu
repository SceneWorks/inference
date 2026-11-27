// Provenance (SW-F287 E10): an original implementation of the published Gated DeltaNet recurrence
// (Yang et al., "Gated Delta Networks"), written against this repository's candle op chain bit for
// bit (inference PR #1469). No source is copied from comfy-kitchen's or mlx_lm's kernels, so no
// third-party licence applies.
//
// Fused Gated DeltaNet input preparation (Qwen3.5 / Qwen3.8 linear-attention layers): the short
// causal conv + SiLU, the q/k L2 norms, and the decay / delta-strength gates.
//
// NUMERICS CONTRACT (as fused_decode.cu): every kernel reproduces candle's op chain bit for bit
// on bf16 — each candle kernel's result is rounded to bf16 exactly where candle stores a bf16
// tensor, f32 steps use the `__f*_rn` intrinsics (candle compiles without fast-math; its f32
// `affine` contracts `x * mul + add` into an fma, reproduced with `__fmaf_rn`), `expf` / `logf` /
// `powf` are the same libdevice functions nvcc links into candle's kernels, and the bf16
// transcendental helpers replicate `hexp` / `__hdiv` from CUDA 12.9's `cuda_bf16.hpp` (see
// fused_decode.cu's TOOLKIT DEPENDENCY note; the GPU parity tests guard it).

typedef unsigned short bf16_t;

__device__ __forceinline__ float bf16_to_f32(bf16_t h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

__device__ __forceinline__ bf16_t f32_to_bf16(float f) {
    unsigned int u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (bf16_t)((u >> 16) | 0x0040u);
    }
    unsigned int lsb = (u >> 16) & 1u;
    return (bf16_t)((u + 0x7fffu + lsb) >> 16);
}

__device__ __forceinline__ float rb(float f) { return bf16_to_f32(f32_to_bf16(f)); }

__device__ __forceinline__ float ex2_approx(float x) {
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float div_approx(float a, float b) {
    float r;
    asm("div.approx.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// bf16 hexp(x): ex2.approx(x * log2e) rounded to bf16.
__device__ __forceinline__ float hexp_bf16(float x) {
    const float log2e_up = __uint_as_float(0x3FB8AA3Cu);
    return rb(ex2_approx(__fmul_rn(x, log2e_up)));
}

// bf16 __hdiv(a, d): div.approx.f32 on the widened operands with the 2^126 guard, rounded.
__device__ __forceinline__ float hdiv_bf16(float a, float d) {
    const float two_126 = __uint_as_float(0x7E800000u);
    bool big = fabsf(d) >= two_126;
    if (big) d = __fmul_rn(d, 0.25f);
    float r = div_approx(a, d);
    if (big) r = __fmaf_rn(r, 0.25f, -0.0f);
    return rb(r);
}

// candle usilu_bf16: x / (1 + hexp(-x)).
__device__ __forceinline__ float silu_bf16(float x) {
    float d = rb(__fadd_rn(1.0f, hexp_bf16(-x)));
    return hdiv_bf16(x, d);
}

// candle usigmoid_bf16: 1 / (1 + hexp(-x)).
__device__ __forceinline__ float sigmoid_bf16(float x) {
    float d = rb(__fadd_rn(1.0f, hexp_bf16(-x)));
    return hdiv_bf16(1.0f, d);
}

// ---------------------------------------------------------------------------------------------
// Causal depthwise conv + SiLU (candle-llm `causal_depthwise_conv_traced`):
//   cat = [state ; x]                         [B, S + K - 1, C]
//   acc = sum_j cat[:, j:j+S] * w[:, j]       bf16 bmul then bf16 badd, j = 0 .. K-1 in order
//   out = silu(acc)
// Writes `cat` (the trace the checkpoint ring's conv tails are read from) and `out`.
// One thread per (b, position, channel) of `cat`; the S output positions also compute `out`.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void gdn_conv_silu_bf16(const bf16_t* __restrict__ state,
                                              const bf16_t* __restrict__ x,
                                              const bf16_t* __restrict__ w,
                                              bf16_t* __restrict__ cat, bf16_t* __restrict__ out,
                                              int B, int S, int C, int K) {
    const int L = S + K - 1;
    const long long n = (long long)B * L * C;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < n;
         e += (long long)gridDim.x * blockDim.x) {
        const int c = (int)(e % C);
        const int p = (int)((e / C) % L);
        const int b = (int)(e / ((long long)C * L));
        cat[e] = p < K - 1 ? state[((long long)b * (K - 1) + p) * C + c]
                           : x[((long long)b * S + (p - (K - 1))) * C + c];
        if (p >= K - 1) {
            const int s = p - (K - 1);  // output position: window cat[s .. s + K - 1]
            float acc = 0.0f;
            for (int j = 0; j < K; ++j) {
                const int q = s + j;
                const float v = bf16_to_f32(q < K - 1 ? state[((long long)b * (K - 1) + q) * C + c]
                                                      : x[((long long)b * S + (q - (K - 1))) * C + c]);
                const float term = rb(__fmul_rn(v, bf16_to_f32(w[(long long)c * K + j])));
                acc = j == 0 ? term : rb(__fadd_rn(acc, term));
            }
            out[((long long)b * S + s) * C + c] = f32_to_bf16(silu_bf16(acc));
        }
    }
}

// fast_sum's tree in bf16 over a power-of-two row of DK values (DK = 32 * PER_LANE): every add
// rounds to bf16 (candle's bf16 `fast_sum` accumulates in its shared bf16 array); lane 0 holds
// the tree's value, broadcast to the warp.
template <int PER_LANE>
__device__ __forceinline__ float bf16_tree_sum(float (&p)[PER_LANE]) {
#pragma unroll
    for (int m = PER_LANE / 2; m >= 1; m >>= 1) {
#pragma unroll
        for (int i = 0; i < m; ++i) {
            p[i] = rb(__fadd_rn(p[i], p[i + m]));
        }
    }
    float x = p[0];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        x = rb(__fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, o)));
    }
    return __shfl_sync(0xffffffffu, x, 0);
}

// ---------------------------------------------------------------------------------------------
// q / k / v from the conv output [B, S, 2 * Hk * Dk + Hv * Dv] (q ‖ k ‖ v channels):
//   q = l2norm(q) * scale,  k = l2norm(k),  v as is      — all three widened to f32
// with candle-llm `l2norm`: ss = sum(x * x) (bf16 usqr, bf16 fast_sum), inv = (ss + eps)^-0.5
// (bf16 affine, bf16 upowf), x * inv (bf16 bmul); q then bf16 affine(scale, 0).
// One warp per (b, s, head) row of q and k (2 * Hk rows per token); v rows are copied by the
// same grid after the q/k rows (one warp per Dv-wide v head).
// eps and scale arrive as the bf16 values candle's affine kernels receive.
// ---------------------------------------------------------------------------------------------
template <int PER_LANE>
__device__ void gdn_qkv(const bf16_t* __restrict__ conv, float* __restrict__ q,
                        float* __restrict__ k, float* __restrict__ v, int B, int S, int Hk,
                        int Hv, int Dv, float eps, float scale) {
    constexpr int DK = PER_LANE * 32;
    const int lane = threadIdx.x % 32;
    const long long warp = (long long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const long long tokens = (long long)B * S;
    const long long qk_rows = tokens * 2 * Hk;
    const long long v_rows = tokens * Hv;
    const long long C = 2LL * Hk * DK + (long long)Hv * Dv;
    if (warp < qk_rows) {
        const long long tok = warp / (2 * Hk);
        const int r = (int)(warp % (2 * Hk));  // 0..Hk-1 = q heads, Hk..2Hk-1 = k heads
        const bf16_t* src = conv + tok * C + (long long)r * DK;
        float x[PER_LANE], p[PER_LANE];
#pragma unroll
        for (int i = 0; i < PER_LANE; ++i) {
            x[i] = bf16_to_f32(src[lane + 32 * i]);
            p[i] = rb(__fadd_rn(0.0f, rb(__fmul_rn(x[i], x[i]))));
        }
        const float ss = bf16_tree_sum<PER_LANE>(p);
        const float plus = rb(__fmaf_rn(ss, 1.0f, eps));
        const float inv = rb(powf(plus, -0.5f));
        const bool is_q = r < Hk;
        const int head = is_q ? r : r - Hk;
        float* dst = (is_q ? q : k) + (tok * Hk + head) * DK;
#pragma unroll
        for (int i = 0; i < PER_LANE; ++i) {
            float y = rb(__fmul_rn(x[i], inv));
            if (is_q) {
                y = rb(__fmaf_rn(y, scale, 0.0f));
            }
            dst[lane + 32 * i] = y;
        }
    } else if (warp < qk_rows + v_rows) {
        const long long vr = warp - qk_rows;
        const long long tok = vr / Hv;
        const int head = (int)(vr % Hv);
        const bf16_t* src = conv + tok * C + 2LL * Hk * DK + (long long)head * Dv;
        float* dst = v + (tok * Hv + head) * Dv;
        for (int i = lane; i < Dv; i += 32) {
            dst[i] = bf16_to_f32(src[i]);
        }
    }
}

#define GDN_QKV_ENTRY(DK)                                                                      \
    extern "C" __global__ void gdn_qkv_bf16_dk##DK(const bf16_t* __restrict__ conv,          \
                                                   float* __restrict__ q,                     \
                                                   float* __restrict__ k,                     \
                                                   float* __restrict__ v, int B, int S,       \
                                                   int Hk, int Hv, int Dv, float eps,         \
                                                   float scale) {                             \
        gdn_qkv<DK / 32>(conv, q, k, v, B, S, Hk, Hv, Dv, eps, scale);                         \
    }

GDN_QKV_ENTRY(32)
GDN_QKV_ENTRY(64)
GDN_QKV_ENTRY(128)
GDN_QKV_ENTRY(256)

// ---------------------------------------------------------------------------------------------
// Gates, per (b, s, value head):
//   g    = bf16( exp(-(exp(A_log) * softplus(a + dt_bias))) )   (f32 chain, candle-llm compute_g)
//   beta = sigmoid(b)                                           (bf16 usigmoid)
// both widened to f32. softplus(x) = relu(x) + log(1 + exp(-|x|)) as candle-llm evaluates it.
// a, b: [B, S, Hv] bf16; a_log, dt_bias: [Hv] bf16.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void gdn_gates_bf16(const bf16_t* __restrict__ a,
                                          const bf16_t* __restrict__ b,
                                          const bf16_t* __restrict__ a_log,
                                          const bf16_t* __restrict__ dt_bias,
                                          float* __restrict__ g, float* __restrict__ beta,
                                          long long n, int Hv) {
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x; e < n;
         e += (long long)gridDim.x * blockDim.x) {
        const int h = (int)(e % Hv);
        const float s = __fadd_rn(bf16_to_f32(a[e]), bf16_to_f32(dt_bias[h]));
        const float relu = fmaxf(s, 0.0f);
        const float neg_abs = __fmaf_rn(fabsf(s), -1.0f, 0.0f);
        const float log1p = logf(__fmaf_rn(expf(neg_abs), 1.0f, 1.0f));
        const float sp = __fadd_rn(relu, log1p);
        const float coeff = __fmul_rn(expf(bf16_to_f32(a_log[h])), sp);
        const float gate = expf(__fmaf_rn(coeff, -1.0f, 0.0f));
        g[e] = rb(gate);
        beta[e] = sigmoid_bf16(bf16_to_f32(b[e]));
    }
}
