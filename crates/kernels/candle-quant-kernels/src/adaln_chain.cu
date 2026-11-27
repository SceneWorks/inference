// SW-6129 (SW-F287 S4): the fused AdaLN for the COMPOSABLE norm chains the DiT providers carry —
// `norm(x) * (1 + scale) + shift` (or `* scale`) in one launch, bit-identical to the candle op
// chain each crate runs today:
//
//   xc   = x.to_dtype(F32)                        (precision F32; the identity for an f32 x)
//        | x                                      (precision native: every op in x's dtype)
//   Layer:    mean = xc.mean_keepdim(-1); c = xc - mean; var = c.sqr().mean_keepdim(-1);
//             y = c / (var + eps).sqrt()
//   Rms:      y = xc / (xc.sqr().mean_keepdim(-1) + eps).sqrt()
//   RmsRecip: y = xc * (xc.sqr().mean_keepdim(-1) + eps).sqrt().recip()
//   [y = y * weight] [y = y + bias]               (separate broadcast ops, compute dtype)
//   [y = y.to_dtype(T)]                           (cast through T, e.g. back to x's dtype)
//   [y = y.to_dtype(W) * post_weight]             (a norm weight applied after the cast, in W)
//   out = (y.to_dtype(M) * (scale + 1) + shift).to_dtype(O)
//
// (candle_nn's LayerNorm without a bias is the Layer chain: its `sum_keepdim / n` is the same
// affine as `mean_keepdim`.) Written from the in-tree candle-kernels sources (vendored candle 0.10.2:
// `reduce.cu` fast_sum, `affine.cu`, `binary.cu`, `unary.cu`, `cast.cu`) and CUDA's
// `cuda_bf16.hpp` (hsqrt / __hdiv); nothing is copied from comfy-kitchen beyond the one-launch
// AdaLN idea S2's `adaln.cu` already credits.
//
// BIT IDENTITY CONTRACT — each step is the reference kernel's own arithmetic, in its order:
//   mean_keepdim  candle `fast_sum` (one block per row, blockDim = next_pow2(min(1024, n)); thread
//                 t accumulates columns t, t+blockDim, ... into its slot in that order; then the
//                 shared-memory tree `slot[t] += slot[t + s]` for s = blockDim/2 .. 1; every add
//                 rounded to the compute dtype), then `affine(1/n, 0)`: f32 `fma(sum, inv_n, 0)`
//                 (nvcc contracts `x*mul + add`), bf16 `round(sum * bf16(1/n))`.
//   sub / sqr / mul / add   one correctly rounded op each (separate kernels, never contracted).
//                 For bf16 a product or sum of two bf16 values computed in f32 and rounded once is
//                 the hardware's rounded bf16 op.
//   + eps         `affine(1, eps)`: f32 fma(v, 1, eps) == round(v + eps); bf16 eps itself rounded.
//   sqrt          f32 `sqrtf` (IEEE, sqrt.rn); bf16 `hsqrt` = sqrt.approx.f32 then round.
//   div           f32 IEEE `x / y`; bf16 `__hdiv` = div.approx.f32 (with its 2^126 rescale) then
//                 round.
//   recip         f32 `recipg` = `1.0 / a`, which nvcc lowers to rcp.rn.f32; bf16 `__hdiv(1, a)`.
//   casts         round-to-nearest-even (cvt.rn).
//   modulation    as S2: s = round_M(scale + 1); p = round_M(y * s); out = round_M(p + shift).
//
// Indices are 64-bit; the Rust plan refuses tensors above i32::MAX elements (epic E6).
// Single file, no includes (nvrtc seam rule). No per-thread local arrays.

#define CHAIN_MAX_BLOCK 1024

typedef unsigned short chain_u16;

// dtype codes: 0 = f32, 1 = bf16, 2 = f16.
__device__ __forceinline__ float chain_bf16_to_f32(chain_u16 h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

__device__ __forceinline__ float chain_f16_to_f32(chain_u16 h) {
    float y;
    asm("cvt.f32.f16 %0, %1;" : "=f"(y) : "h"(h));
    return y;
}

__device__ __forceinline__ chain_u16 chain_f32_to_bf16(float f) {
#if __CUDA_ARCH__ >= 800
    chain_u16 r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return r;
#else
    unsigned int u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (chain_u16)0x7fffu;
    unsigned int lsb = (u >> 16) & 1u;
    return (chain_u16)((u + 0x7fffu + lsb) >> 16);
#endif
}

__device__ __forceinline__ chain_u16 chain_f32_to_f16(float f) {
    chain_u16 r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return r;
}

__device__ __forceinline__ float chain_load(const void* p, int dt, long long i) {
    if (dt == 0) return ((const float*)p)[i];
    if (dt == 1) return chain_bf16_to_f32(((const chain_u16*)p)[i]);
    return chain_f16_to_f32(((const chain_u16*)p)[i]);
}

__device__ __forceinline__ float chain_round(float v, int dt) {
    if (dt == 0) return v;
    if (dt == 1) return chain_bf16_to_f32(chain_f32_to_bf16(v));
    return chain_f16_to_f32(chain_f32_to_f16(v));
}

__device__ __forceinline__ void chain_store(void* p, int dt, long long i, float v) {
    if (dt == 0) {
        ((float*)p)[i] = v;
    } else if (dt == 1) {
        ((chain_u16*)p)[i] = chain_f32_to_bf16(v);
    } else {
        ((chain_u16*)p)[i] = chain_f32_to_f16(v);
    }
}

// Compute-dtype arithmetic: `bf` = 0 for f32 (IEEE ops), 1 for bf16 (op in f32, rounded once).
__device__ __forceinline__ float chain_add(float a, float b, int bf) {
    const float r = __fadd_rn(a, b);
    return bf ? chain_round(r, 1) : r;
}

__device__ __forceinline__ float chain_sub(float a, float b, int bf) {
    const float r = __fsub_rn(a, b);
    return bf ? chain_round(r, 1) : r;
}

__device__ __forceinline__ float chain_mul(float a, float b, int bf) {
    const float r = __fmul_rn(a, b);
    return bf ? chain_round(r, 1) : r;
}

// cuda_bf16.hpp `__internal_device_hdiv`.
__device__ __forceinline__ float chain_bf16_div(float a, float b) {
    const float two_126 = __uint_as_float(0x7E800000U);
    const bool big = fabsf(b) >= two_126;
    if (big) b = __fmul_rn(b, 0.25f);
    float ans;
    asm("{ div.approx.f32 %0, %1, %2; }" : "=f"(ans) : "f"(a), "f"(b));
    if (big) ans = __fmul_rn(ans, 0.25f);
    return chain_round(ans, 1);
}

__device__ __forceinline__ float chain_div(float a, float b, int bf) {
    return bf ? chain_bf16_div(a, b) : __fdiv_rn(a, b);
}

__device__ __forceinline__ float chain_sqrt(float a, int bf) {
    if (!bf) return __fsqrt_rn(a);
    float r = a;
    asm("{ sqrt.approx.f32 %0, %0; }" : "+f"(r));
    return chain_round(r, 1);
}

__device__ __forceinline__ float chain_recip(float a, int bf) {
    if (bf) return chain_bf16_div(1.0f, a);
    // `1.0 / a` with a double literal; nvcc lowers it to the exactly rounded rcp.rn.f32.
    return __frcp_rn(a);
}

// candle `fast_sum` of one row's `v(col)` values, `bd` threads; returns the row sum to every
// thread. `slot` is the block's CHAIN_MAX_BLOCK-entry scratch.
__device__ __forceinline__ float chain_row_sum(float acc, float* slot, int bf) {
    const int tid = threadIdx.x;
    slot[tid] = acc;
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) slot[tid] = chain_add(slot[tid], slot[tid + s], bf);
    }
    __syncthreads();
    const float total = slot[0];
    __syncthreads();
    return total;
}

// One block per row of `n` contiguous columns of `x` (dtype `dt_x`). `norm`: 0 Layer, 1 Rms,
// 2 RmsRecip. `bf`: compute in bf16 (native precision on a bf16 x), else f32. `inv_n` / `eps` are
// the affine operands already in the compute dtype. weight / bias (`kind` 0 absent, 1 present) are
// compute-dtype `[n]`. scale / shift are `dt_m` with row offsets `b * sb + s * ss`; out is `dt_o`.
extern "C" __global__ void adaln_chain_modulate(
    const void* x, int dt_x, const void* weight, int weight_kind, const void* bias, int bias_kind,
    const void* scale, const void* shift, int dt_m, void* out, int dt_o, int n, int norm, int bf,
    float inv_n, float eps, int one_plus, long long seq, long long scale_sb, long long scale_ss,
    long long shift_sb, long long shift_ss, int has_shift, int dt_mid, const void* post_weight,
    int dt_pw) {
    __shared__ float slot[CHAIN_MAX_BLOCK];
    const long long row = blockIdx.x;
    const long long base = row * (long long)n;
    const int tid = threadIdx.x;
    const int step = blockDim.x;
    const int wdt = bf ? 1 : 0;

    float mean = 0.0f;
    if (norm == 0) {
        float acc = 0.0f;
        for (int col = tid; col < n; col += step) {
            acc = chain_add(acc, chain_load(x, dt_x, base + col), bf);
        }
        const float sum = chain_row_sum(acc, slot, bf);
        mean = bf ? chain_add(chain_round(__fmul_rn(sum, inv_n), 1), 0.0f, 1)
                  : __fmaf_rn(sum, inv_n, 0.0f);
    }
    float acc = 0.0f;
    for (int col = tid; col < n; col += step) {
        float v = chain_load(x, dt_x, base + col);
        if (norm == 0) v = chain_sub(v, mean, bf);
        acc = chain_add(acc, chain_mul(v, v, bf), bf);
    }
    const float sumsq = chain_row_sum(acc, slot, bf);
    const float ms = bf ? chain_add(chain_round(__fmul_rn(sumsq, inv_n), 1), 0.0f, 1)
                        : __fmaf_rn(sumsq, inv_n, 0.0f);
    const float den = chain_sqrt(chain_add(ms, eps, bf), bf);
    const float inv = norm == 2 ? chain_recip(den, bf) : 0.0f;

    const long long b = row / seq;
    const long long s = row - b * seq;
    const long long scale_row = b * scale_sb + s * scale_ss;
    const long long shift_row = b * shift_sb + s * shift_ss;
    for (int col = tid; col < n; col += step) {
        float v = chain_load(x, dt_x, base + col);
        float y;
        if (norm == 0) {
            y = chain_div(chain_sub(v, mean, bf), den, bf);
        } else if (norm == 1) {
            y = chain_div(v, den, bf);
        } else {
            y = chain_mul(v, inv, bf);
        }
        if (weight_kind) y = chain_mul(y, chain_load(weight, wdt, col), bf);
        if (bias_kind) y = chain_add(y, chain_load(bias, wdt, col), bf);
        // Optional cast-through (`dt_mid` >= 0) and post-cast weight in its own dtype (`dt_pw`).
        if (dt_mid >= 0) y = chain_round(y, dt_mid);
        if (dt_pw >= 0) {
            y = chain_round(y, dt_pw);
            y = chain_round(__fmul_rn(y, chain_load(post_weight, dt_pw, col)), dt_pw);
        }
        y = chain_round(y, dt_m);
        float sc = chain_load(scale, dt_m, scale_row + col);
        if (one_plus) sc = chain_round(__fadd_rn(sc, 1.0f), dt_m);
        const float p = chain_round(__fmul_rn(y, sc), dt_m);
        // Scale-only (`norm * (1 + scale)`, no shift): the product is the result.
        const float o = has_shift
            ? chain_round(__fadd_rn(p, chain_load(shift, dt_m, shift_row + col)), dt_m)
            : p;
        chain_store(out, dt_o, base + col, o);
    }
}
