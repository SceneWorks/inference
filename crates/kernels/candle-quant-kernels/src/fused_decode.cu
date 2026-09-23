// Fused decode primitives (sc-24137, epic sc-24128): RMSNorm(+residual), SwiGLU, QK-norm+RoPE.
//
// Compiled at runtime through the nvrtc compile-once seam (`nvrtc.rs`): builtins only, no
// headers, no build.rs. bf16 is handled as raw `unsigned short` with hand-written conversions so
// the source needs neither `cuda_bf16.h` nor an include path.
//
// NUMERICS CONTRACT: each kernel reproduces candle's op chain *bit for bit* on bf16 and f32, so
// the fused path is a pure speed change (the story's AC2 is token-identical greedy decode with the
// kernels on vs off). That is why the code below looks pedantic:
//
// - The row reduction uses candle's `fast_sum` launch shape (one block per row, blockDim =
//   min(1024, next_pow2(n))), its strided per-thread accumulation and its shared-memory tree, so
//   the f32 summation order is identical.
// - Every f32 op that candle performs as a separate kernel is written with the `__f*_rn`
//   intrinsics, which nvrtc never contracts into an fma (candle's affine kernel *does* contract
//   `x*mul+add` into an fma, but with add == 0 or mul == 1 that is the same rounded value).
// - bf16 results are rounded exactly where candle's op chain stores a bf16 tensor. A bf16 product
//   or sum of two bf16 values computed in f32 and rounded once equals the hardware's exactly
//   rounded `__hmul` / `__hadd`, so the elementwise ops match too.
// - bf16 SiLU replicates `usilu_bf16` (`x / (1 + hexp(-x))` in native bf16): `hexp` is
//   `ex2.approx.f32(x * log2e)` rounded to bf16, and `__hdiv` is `div.approx.f32` on the widened
//   operands with the 2^126 guard — the same PTX candle's kernel executes.

typedef unsigned short bf16_t;

__device__ __forceinline__ float bf16_to_f32(bf16_t h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

// Round-to-nearest-even, no flush-to-zero: the same value as `cvt.rn.bf16.f32` for every finite
// input (NaN stays a quiet NaN).
__device__ __forceinline__ bf16_t f32_to_bf16(float f) {
    unsigned int u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (bf16_t)((u >> 16) | 0x0040u);
    }
    unsigned int lsb = (u >> 16) & 1u;
    return (bf16_t)((u + 0x7fffu + lsb) >> 16);
}

__device__ __forceinline__ float round_bf16(float f) { return bf16_to_f32(f32_to_bf16(f)); }

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

// candle `usilu_f32`: x / (1 + expf(-x)).
__device__ __forceinline__ float silu_f32(float x) {
    return __fdiv_rn(x, __fadd_rn(1.0f, expf(-x)));
}

// candle `usilu_bf16` on sm_80+: x / (1 + hexp(-x)) with every op in bf16.
__device__ __forceinline__ float silu_bf16(float x) {
    const float log2e_up = __uint_as_float(0x3FB8AA3Cu);
    float e = round_bf16(ex2_approx(__fmul_rn(-x, log2e_up)));  // hexp(-x)
    float d = round_bf16(__fadd_rn(1.0f, e));                    // __hadd(1, e)
    const float two_126 = __uint_as_float(0x7E800000u);           // __hdiv(x, d)
    bool b_big = fabsf(d) >= two_126;
    if (b_big) d = __fmul_rn(d, 0.25f);
    float r = div_approx(x, d);
    if (b_big) r = __fmaf_rn(r, 0.25f, -0.0f);
    return round_bf16(r);
}

// ---------------------------------------------------------------------------------------------
// Row RMS: sum(x^2) in candle's fast_sum order, then mean = sum*scale, denom = sqrt(mean + eps).
// `vals` supplies the (already rounded) row value for index i; `n` is the row length. All
// threads leave with the row's `denom`.
// ---------------------------------------------------------------------------------------------
template <typename Load>
__device__ __forceinline__ float row_rms_denom(float* shr, Load load, int n, float scale,
                                               float eps) {
    int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = load(i);
        acc = __fadd_rn(acc, __fmul_rn(v, v));
    }
    shr[tid] = acc;
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) shr[tid] = __fadd_rn(shr[tid], shr[tid + s]);
    }
    __syncthreads();
    float sum = shr[0];
    __syncthreads();  // shr is reused by the next row/kernel phase
    float mean = __fmul_rn(sum, scale);
    return sqrtf(__fadd_rn(mean, eps));
}

// ---------------------------------------------------------------------------------------------
// RMSNorm (+ optional residual). One block per row.
//   h = x + residual (rounded to the dtype)      -- written to `h_out` when `residual` != null
//   y = (h / sqrt(mean(h^2) + eps)) * w           -- computed in f32, rounded to the dtype
// Args: x[rows*n], residual[rows*n] | null, w[n], h_out[rows*n] | null, y[rows*n], n, scale, eps
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void rms_norm_residual_f32(const float* x, const float* residual,
                                                 const float* w, float* h_out, float* y, int n,
                                                 float scale, float eps) {
    __shared__ float shr[1024];
    size_t row = blockIdx.x;
    const float* xr = x + row * (size_t)n;
    const float* rr = residual ? residual + row * (size_t)n : 0;
    float* hr = h_out ? h_out + row * (size_t)n : 0;
    float* yr = y + row * (size_t)n;
    if (rr) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) hr[i] = __fadd_rn(xr[i], rr[i]);
    }
    const float* src = rr ? hr : xr;
    float denom = row_rms_denom(shr, [&](int i) { return src[i]; }, n, scale, eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        yr[i] = __fmul_rn(__fdiv_rn(src[i], denom), w[i]);
    }
}

extern "C" __global__ void rms_norm_residual_bf16(const bf16_t* x, const bf16_t* residual,
                                                  const bf16_t* w, bf16_t* h_out, bf16_t* y,
                                                  int n, float scale, float eps) {
    __shared__ float shr[1024];
    size_t row = blockIdx.x;
    const bf16_t* xr = x + row * (size_t)n;
    const bf16_t* rr = residual ? residual + row * (size_t)n : 0;
    bf16_t* hr = h_out ? h_out + row * (size_t)n : 0;
    bf16_t* yr = y + row * (size_t)n;
    if (rr) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            hr[i] = f32_to_bf16(__fadd_rn(bf16_to_f32(xr[i]), bf16_to_f32(rr[i])));
        }
    }
    const bf16_t* src = rr ? hr : xr;
    float denom = row_rms_denom(shr, [&](int i) { return bf16_to_f32(src[i]); }, n, scale, eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = __fmul_rn(__fdiv_rn(bf16_to_f32(src[i]), denom), bf16_to_f32(w[i]));
        yr[i] = f32_to_bf16(v);
    }
}

// ---------------------------------------------------------------------------------------------
// SwiGLU: out = silu(gate) * up, elementwise over `n` values.
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void swiglu_f32(const float* gate, const float* up, float* out,
                                      unsigned int n) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        out[i] = __fmul_rn(silu_f32(gate[i]), up[i]);
    }
}

extern "C" __global__ void swiglu_bf16(const bf16_t* gate, const bf16_t* up, bf16_t* out,
                                       unsigned int n) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        float g = silu_bf16(bf16_to_f32(gate[i]));
        out[i] = f32_to_bf16(__fmul_rn(g, bf16_to_f32(up[i])));
    }
}

// ---------------------------------------------------------------------------------------------
// QK-norm + RoPE. One block per (batch, seq, head) row of `hd` values:
//   q = rms_norm(x, w)                                   (per head, rounded to the dtype)
//   q[:rd] = q[:rd] * cos + rotate_half(q[:rd]) * sin    (NeoX half-split or GPT-J interleaved)
//   q[rd:] passes through.
// cos/sin are [cb, s, rd] with cb in {1, b}; rows are laid out as [b, s, heads].
// ---------------------------------------------------------------------------------------------
extern "C" __global__ void rms_norm_rope_f32(const float* x, const float* w, const float* cs,
                                             const float* sn, float* y, int hd, int rd, int heads,
                                             int seq, int cos_batched, int interleaved,
                                             float scale, float eps) {
    __shared__ float shr[1024];
    __shared__ float nrm[1024];
    size_t row = blockIdx.x;
    const float* xr = x + row * (size_t)hd;
    float* yr = y + row * (size_t)hd;
    int s_idx = (int)((row / (size_t)heads) % (size_t)seq);
    int b_idx = (int)(row / ((size_t)heads * (size_t)seq));
    size_t cs_row = ((size_t)(cos_batched ? b_idx : 0) * (size_t)seq + (size_t)s_idx) * (size_t)rd;
    const float* cr = cs + cs_row;
    const float* sr = sn + cs_row;

    float denom = row_rms_denom(shr, [&](int i) { return xr[i]; }, hd, scale, eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        nrm[i] = __fmul_rn(__fdiv_rn(xr[i], denom), w[i]);
    }
    __syncthreads();
    int half = rd / 2;
    for (int j = threadIdx.x; j < hd; j += blockDim.x) {
        float v = nrm[j];
        if (j < rd) {
            float rot;
            if (interleaved) {
                rot = (j & 1) ? nrm[j - 1] : -nrm[j + 1];
            } else {
                rot = (j < half) ? -nrm[j + half] : nrm[j - half];
            }
            float t1 = __fmul_rn(v, cr[j]);
            float t2 = __fmul_rn(rot, sr[j]);
            v = __fadd_rn(t1, t2);
        }
        yr[j] = v;
    }
}

extern "C" __global__ void rms_norm_rope_bf16(const bf16_t* x, const bf16_t* w, const bf16_t* cs,
                                              const bf16_t* sn, bf16_t* y, int hd, int rd,
                                              int heads, int seq, int cos_batched,
                                              int interleaved, float scale, float eps) {
    __shared__ float shr[1024];
    __shared__ float nrm[1024];
    size_t row = blockIdx.x;
    const bf16_t* xr = x + row * (size_t)hd;
    bf16_t* yr = y + row * (size_t)hd;
    int s_idx = (int)((row / (size_t)heads) % (size_t)seq);
    int b_idx = (int)(row / ((size_t)heads * (size_t)seq));
    size_t cs_row = ((size_t)(cos_batched ? b_idx : 0) * (size_t)seq + (size_t)s_idx) * (size_t)rd;
    const bf16_t* cr = cs + cs_row;
    const bf16_t* sr = sn + cs_row;

    float denom = row_rms_denom(shr, [&](int i) { return bf16_to_f32(xr[i]); }, hd, scale, eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        nrm[i] = round_bf16(__fmul_rn(__fdiv_rn(bf16_to_f32(xr[i]), denom), bf16_to_f32(w[i])));
    }
    __syncthreads();
    int half = rd / 2;
    for (int j = threadIdx.x; j < hd; j += blockDim.x) {
        float v = nrm[j];
        if (j < rd) {
            float rot;
            if (interleaved) {
                rot = (j & 1) ? nrm[j - 1] : -nrm[j + 1];
            } else {
                rot = (j < half) ? -nrm[j + half] : nrm[j - half];
            }
            float t1 = round_bf16(__fmul_rn(v, bf16_to_f32(cr[j])));
            float t2 = round_bf16(__fmul_rn(rot, bf16_to_f32(sr[j])));
            v = round_bf16(__fadd_rn(t1, t2));
        }
        yr[j] = f32_to_bf16(v);
    }
}
