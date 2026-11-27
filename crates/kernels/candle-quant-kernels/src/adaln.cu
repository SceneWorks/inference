// SW-6127 (SW-F287 S2): fused AdaLN / RMS-AdaLN — `norm(x) * (1 + scale) + shift` (or `* scale`)
// in one launch, for RMSNorm and LayerNorm with an optional learned norm weight (and LayerNorm
// bias), modulation broadcast per sample `[B,1,D]` or per token `[B,S,D]`. Written from the
// in-tree reference semantics (candle-kernels `reduce.cu` `rmsnorm` / `layernorm`, candle's
// `affine` and broadcast binary kernels, as `candle_gen::modulate::adaln_reference` chains them);
// the one-launch norm + modulate structure follows comfy-kitchen's `adaln.cu` design, but no
// comfy-kitchen source is copied (its reduction order differs from candle's, which this kernel
// must reproduce to be bit-identical).
//
// BIT IDENTITY CONTRACT (each step is the reference's own operation, in the reference's order):
//   reduce  candle's rmsnorm / layernorm launch: one block per row, blockDim = 32 below 1024
//           columns else 1024; per-thread strided accumulation (col = tid; col += blockDim) with
//           `acc += x*x` contracted to one fma (nvcc -O3 default) and LayerNorm's `sum += x`;
//           warp xor-butterfly (mask 16..1), then for 1024 threads the 32 warp partials through
//           shared memory and a second butterfly. Every thread ends with the same row sums.
//   stats   rms: mean = sum / n (IEEE divide), rs = rsqrtf(mean + eps).
//           layer: mean = sum / n, var = sumsq / n - mean*mean (contracted fma), rs =
//           rsqrtf(var + eps).
//   norm    rms: (rs * x) * w; layer: fma((x - mean) * rs, w, b) — w = 1 / b = 0 when absent,
//           which equals the reference's ones / zeros tensors. Computed in f32 and rounded once
//           to x's dtype: the reference either stores the norm in that dtype directly (native
//           precision) or computes it in f32 and casts (`norm_in_f32`), the same value.
//   mod     one-plus: s = round(scale + 1) (candle's affine(1, 1): `x*1 + 1` is exact up to the
//           one rounding); p = round(norm * s); out = round(p + shift). A product or sum of two
//           f32-exact 16-bit values computed in f32 and rounded once equals the hardware's exactly
//           rounded bf16 / f16 multiply and add, so the elementwise ops match too. f32: plain
//           __fmul_rn / __fadd_rn (separate candle kernels, never contracted).
//
// Indices are 64-bit; the Rust plan refuses tensors above i32::MAX elements (epic E6) anyway.
// Single file, no includes (nvrtc seam rule). No per-thread local arrays.

#define ADALN_WARP 32

typedef unsigned short adaln_u16;

__device__ __forceinline__ float adaln_bf16_to_f32(adaln_u16 h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

__device__ __forceinline__ float adaln_f16_to_f32(adaln_u16 h) {
    float y;
    asm("cvt.f32.f16 %0, %1;" : "=f"(y) : "h"(h));
    return y;
}

// `__float2bfloat16`: cvt.rn.bf16.f32 where the ISA has it (sm_80+, the only arch candle's bf16
// kernels exist on); a bit-equivalent software round-to-nearest-even below it.
__device__ __forceinline__ adaln_u16 adaln_f32_to_bf16(float f) {
#if __CUDA_ARCH__ >= 800
    adaln_u16 r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return r;
#else
    unsigned int u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (adaln_u16)0x7fffu;
    unsigned int lsb = (u >> 16) & 1u;
    return (adaln_u16)((u + 0x7fffu + lsb) >> 16);
#endif
}

__device__ __forceinline__ adaln_u16 adaln_f32_to_f16(float f) {
    adaln_u16 r;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return r;
}

// Element access per storage dtype: 0 = f32, 1 = bf16, 2 = f16.
template <int DT>
__device__ __forceinline__ float adaln_load(const void* p, long long i) {
    if (DT == 0) return ((const float*)p)[i];
    if (DT == 1) return adaln_bf16_to_f32(((const adaln_u16*)p)[i]);
    return adaln_f16_to_f32(((const adaln_u16*)p)[i]);
}

template <int DT>
__device__ __forceinline__ float adaln_round(float v) {
    if (DT == 0) return v;
    if (DT == 1) return adaln_bf16_to_f32(adaln_f32_to_bf16(v));
    return adaln_f16_to_f32(adaln_f32_to_f16(v));
}

template <int DT>
__device__ __forceinline__ void adaln_store(void* p, long long i, float v) {
    if (DT == 0) {
        ((float*)p)[i] = v;
    } else if (DT == 1) {
        ((adaln_u16*)p)[i] = adaln_f32_to_bf16(v);
    } else {
        ((adaln_u16*)p)[i] = adaln_f32_to_f16(v);
    }
}

// An optional per-column parameter (norm weight or bias): kind 0 = absent (`dflt`), 1 = x's
// dtype, 2 = f32.
template <int DT>
__device__ __forceinline__ float adaln_param(const void* p, int kind, int col, float dflt) {
    if (kind == 0) return dflt;
    if (kind == 2) return ((const float*)p)[col];
    return adaln_load<DT>(p, col);
}

__device__ __forceinline__ float adaln_warp_sum(float v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, mask, 32));
    }
    return v;
}

// candle's two-level block sum (`block_size > WARP_SIZE` branch); `s` is a 32-slot scratch.
__device__ __forceinline__ float adaln_block_sum(float v, float* s) {
    v = adaln_warp_sum(v);
    if (blockDim.x > ADALN_WARP) {
        const int warp = threadIdx.x / ADALN_WARP;
        const int lane = threadIdx.x % ADALN_WARP;
        if (lane == 0) s[warp] = v;
        __syncthreads();
        v = s[lane];
        v = adaln_warp_sum(v);
    }
    return v;
}

// One block per row of `n` columns. `x` and `out` are contiguous `[rows, n]`; row r belongs to
// sample r / seq at position r % seq, and its modulation rows start at
// `b * scale_sb + s * scale_ss` (element strides; 0 broadcasts that axis). `layer` selects
// LayerNorm (else RMSNorm); `one_plus` selects `(1 + scale)` (else `scale`).
template <int DT>
__device__ __forceinline__ void adaln_row(const void* x, const void* weight, int weight_kind,
                                          const void* bias, int bias_kind, const void* scale,
                                          const void* shift, void* out, int n, float eps,
                                          int layer, int one_plus, long long seq,
                                          long long scale_sb, long long scale_ss,
                                          long long shift_sb, long long shift_ss, int has_shift) {
    __shared__ float s_sum[ADALN_WARP];
    __shared__ float s_sumsq[ADALN_WARP];
    const long long row = blockIdx.x;
    const long long base = row * (long long)n;
    const int tid = threadIdx.x;
    const int step = blockDim.x;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int col = tid; col < n; col += step) {
        const float xi = adaln_load<DT>(x, base + col);
        if (layer) sum = __fadd_rn(sum, xi);
        sumsq = __fmaf_rn(xi, xi, sumsq);
    }
    float mean = 0.0f;
    float rs;
    if (layer) {
        // candle's float2 warp reduce sums .x then .y per butterfly step; the two lanes are
        // independent, so reducing them one after the other is the same arithmetic.
        sum = adaln_block_sum(sum, s_sum);
        sumsq = adaln_block_sum(sumsq, s_sumsq);
        mean = __fdiv_rn(sum, (float)n);
        const float var = __fmaf_rn(-mean, mean, __fdiv_rn(sumsq, (float)n));
        rs = rsqrtf(__fadd_rn(var, eps));
    } else {
        sumsq = adaln_block_sum(sumsq, s_sumsq);
        rs = rsqrtf(__fadd_rn(__fdiv_rn(sumsq, (float)n), eps));
    }

    const long long b = row / seq;
    const long long s = row - b * seq;
    const long long scale_row = b * scale_sb + s * scale_ss;
    const long long shift_row = b * shift_sb + s * shift_ss;
    for (int col = tid; col < n; col += step) {
        const float xi = adaln_load<DT>(x, base + col);
        const float w = adaln_param<DT>(weight, weight_kind, col, 1.0f);
        float y;
        if (layer) {
            const float bb = adaln_param<DT>(bias, bias_kind, col, 0.0f);
            y = __fmaf_rn(__fmul_rn(__fsub_rn(xi, mean), rs), w, bb);
        } else {
            y = __fmul_rn(__fmul_rn(rs, xi), w);
        }
        y = adaln_round<DT>(y);
        float sc = adaln_load<DT>(scale, scale_row + col);
        if (one_plus) sc = adaln_round<DT>(__fadd_rn(sc, 1.0f));
        const float p = adaln_round<DT>(__fmul_rn(y, sc));
        // Scale-only (`norm * (1 + scale)`, no shift): the product is the result.
        adaln_store<DT>(out, base + col,
                        has_shift ? __fadd_rn(p, adaln_load<DT>(shift, shift_row + col)) : p);
    }
}

#define ADALN_ENTRY(NAME, DT)                                                                    \
    extern "C" __global__ void NAME(const void* x, const void* weight, int weight_kind,          \
                                    const void* bias, int bias_kind, const void* scale,          \
                                    const void* shift, void* out, int n, float eps, int layer,   \
                                    int one_plus, long long seq, long long scale_sb,             \
                                    long long scale_ss, long long shift_sb, long long shift_ss,  \
                                    int has_shift) {                                             \
        adaln_row<DT>(x, weight, weight_kind, bias, bias_kind, scale, shift, out, n, eps, layer, \
                      one_plus, seq, scale_sb, scale_ss, shift_sb, shift_ss, has_shift);         \
    }

ADALN_ENTRY(adaln_modulate_f32, 0)
ADALN_ENTRY(adaln_modulate_bf16, 1)
ADALN_ENTRY(adaln_modulate_f16, 2)
