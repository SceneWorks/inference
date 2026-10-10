// Indexed (gathered) MoE GEMV — sc-24440, epic sc-24432.
//
// A decode step through a Mixture-of-Experts layer runs every (token, slot) pair's routed expert
// on one activation row. These kernels take the routes as DEVICE data: `ids[task]` (u32, one per
// (token, slot) pair, `task = token * slots + slot`) selects expert `e`, and `experts[e]` — a
// device table of u64 base addresses built once at load — is where that expert's weight for this
// projection lives. The weights are read IN PLACE: nothing is gathered or copied, the host never
// reads a route, and every launch shape depends only on the step shape, so a CUDA graph that
// recorded one step replays at any routing.
//
// Provenance: the DERIVED sections below are candle's `candle-kernels/src/mmvq_gguf.cu` (Apache-2.0
// OR MIT, huggingface/candle), itself "adapted from llama.cpp's CUDA mmvq path" — llama.cpp /
// ggml, MIT License, Copyright (c) 2023-2024 The ggml authors.
//
// Compiled at runtime through the nvrtc compile-once seam (`nvrtc.rs`): builtins and inline PTX
// only, no headers. The few fp16 helpers candle's `cuda_fp16.h` would provide are defined below
// as `gg_half` / `gg_half2` (exact `cvt` conversions).
//
// Every kernel here:
//
// * `moe_mmvq_<type>` — GGML block-quantized weights (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K..Q6_K)
//   times a Q8_1-quantized activation, f32 out. The vec-dot functions, block structs and the
//   mat-vec core `mmvq_core_impl` are candle's own `mmvq_gguf.cu` (the vendored candle-kernels'
//   fast MMVQ path that `QMatMul` runs at decode), copied verbatim between the DERIVED markers
//   below with only the fp16 type/helper names renamed — a CPU test in `moe_gemv.rs`
//   re-derives that text from the vendored file and fails on any drift. Each task runs
//   `mmvq_core_impl<..., ncols_dst = 1>` with candle's own launch geometry (grid = rows,
//   block = 32 x 4), so a pair's output is BIT-IDENTICAL to `QMatMul::forward` of that expert on
//   that one row (candle's own upstream `indexed_moe_forward` is not reused: its expert stride is
//   `n * k / QK_K * sizeof(block)`, wrong for the 32-element block types, and it has no Q4_0 kernel).
// * `mmvq_gguf_quantize_q8_1_f32` — candle's activation quantizer, verbatim (DERIVED).
// * `moe_gemv_dense_<t>` — dense f32 / bf16 / f16 expert weights times an activation of the same
//   dtype: f32 accumulate in a fixed order (each lane strides K, then a fixed xor-shuffle tree),
//   one rounding of the output to the dtype.
// * `moe_gemv_q8_0_<t>` — the MLX-affine Q8 tier (`QuantizedLinear`'s dequantize-per-forward
//   form): Q8_0 blocks dequantized in registers to `d * q` (exact in f32), rounded ONCE to the
//   activation dtype exactly as `dequantize(..).to_dtype(t)` rounds, then the dense GEMV above.

typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

// IEEE binary16 as raw bits, with the exact conversions candle's `half` uses (`cvt.f32.f16`;
// `cvt.rn.f16.f32` — round to nearest even — for `(half)f`).
struct alignas(2) gg_half {
  unsigned short bits;
  gg_half() = default;
  __device__ __forceinline__ explicit gg_half(float f) {
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(bits) : "f"(f));
  }
  __device__ __forceinline__ operator float() const {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
  }
};
struct alignas(4) gg_half2 {
  gg_half x;
  gg_half y;
};
static __device__ __forceinline__ float gg_half2float(const gg_half h) { return (float)h; }
static __device__ __forceinline__ float2 gg_half22float2(const gg_half2 h) {
  return make_float2((float)h.x, (float)h.y);
}
static __device__ __forceinline__ gg_half gg_low2half(const gg_half2 h) { return h.x; }
static __device__ __forceinline__ float gg_low2float(const gg_half2 h) { return (float)h.x; }

// ===== BEGIN DERIVED: candle-kernels mmvq_gguf.cu, constants .. mmvq_core_impl =====
// Constants, types, and helpers shared with the indexed MoE kernels.

#define WARP_SIZE 32
#define CUDA_QUANTIZE_BLOCK_SIZE 256
#define K_QUANTS_PER_ITERATION 2
#define QK_K 256
#define K_SCALE_SIZE 12

// Matches candle's MATRIX_ROW_PADDING.
#define MATRIX_ROW_PADDING 512

typedef uint16_t ggml_fp16_t;

static __device__ __forceinline__ float warp_reduce_sum_f32(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, mask, WARP_SIZE);
  }
  return x;
}

static __device__ __forceinline__ float warp_reduce_max_f32(float x) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, WARP_SIZE));
  }
  return x;
}

static __device__ __forceinline__ int get_int_from_int8(const int8_t *x8,
                                                        const int &i32) {
  const uint16_t *x16 = (const uint16_t *)(x8 + sizeof(int) * i32);
  int x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;
  return x32;
}

static __device__ __forceinline__ int get_int_from_uint8(const uint8_t *x8,
                                                         const int &i32) {
  const uint16_t *x16 = (const uint16_t *)(x8 + sizeof(int) * i32);
  int x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;
  return x32;
}

static __device__ __forceinline__ int
get_int_from_int8_aligned(const int8_t *x8, const int &i32) {
  return *((const int *)(x8 + sizeof(int) * i32));
}

static __device__ __forceinline__ int
get_int_from_uint8_aligned(const uint8_t *x8, const int &i32) {
  return *((const int *)(x8 + sizeof(int) * i32));
}

#define MIN_CC_DP4A 610

static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b,
                                                     int c) {
#if __CUDA_ARCH__ >= MIN_CC_DP4A
  return __dp4a(a, b, c);
#else
  const int8_t *a8 = (const int8_t *)&a;
  const int8_t *b8 = (const int8_t *)&b;
  return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif
}

// ---------------------------------------------------------------------------
// Block structs for each supported quantization type
// ---------------------------------------------------------------------------

#define QK8_0 32
#define QR8_0 1
#define QI8_0 (QK8_0 / (4 * QR8_0))
typedef struct {
  gg_half d;
  int8_t qs[QK8_0];
} block_q8_0;

#define QK8_1 32
#define QR8_1 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
typedef struct {
  gg_half2 ds;
  int8_t qs[QK8_0];
} block_q8_1;

#define QR2_K 4
#define QI2_K (QK_K / (4 * QR2_K))
typedef struct {
  uint8_t scales[QK_K / 16];
  uint8_t qs[QK_K / 4];
  gg_half2 dm;
} block_q2_K;

#define QR3_K 4
#define QI3_K (QK_K / (4 * QR3_K))
typedef struct {
  uint8_t hmask[QK_K / 8];
  uint8_t qs[QK_K / 4];
  uint8_t scales[K_SCALE_SIZE];
  gg_half d;
} block_q3_K;

#define QR4_K 2
#define QI4_K (QK_K / (4 * QR4_K))
typedef struct {
  gg_half2 dm;
  uint8_t scales[3 * QK_K / 64];
  uint8_t qs[QK_K / 2];
} block_q4_K;

#define QR5_K 2
#define QI5_K (QK_K / (4 * QR5_K))
typedef struct {
  gg_half2 dm;
  uint8_t scales[K_SCALE_SIZE];
  uint8_t qh[QK_K / 8];
  uint8_t qs[QK_K / 2];
} block_q5_K;

#define QR6_K 2
#define QI6_K (QK_K / (4 * QR6_K))
typedef struct {
  uint8_t ql[QK_K / 2];
  uint8_t qh[QK_K / 4];
  int8_t scales[QK_K / 16];
  gg_half d;
} block_q6_K;

#define QK4_0 32
#define QR4_0 2
#define QI4_0 (QK4_0 / (4 * QR4_0))
typedef struct {
  gg_half d;
  uint8_t qs[QK4_0 / 2];
} block_q4_0;

#define QK4_1 32
#define QR4_1 2
#define QI4_1 (QK4_1 / (4 * QR4_1))
typedef struct {
  gg_half2 dm;
  uint8_t qs[QK4_1 / 2];
} block_q4_1;

#define QK5_0 32
#define QR5_0 2
#define QI5_0 (QK5_0 / (4 * QR5_0))
typedef struct {
  gg_half d;
  uint8_t qh[4];
  uint8_t qs[QK5_0 / 2];
} block_q5_0;

#define QK5_1 32
#define QR5_1 2
#define QI5_1 (QK5_1 / (4 * QR5_1))
typedef struct {
  gg_half2 dm;
  uint8_t qh[4];
  uint8_t qs[QK5_1 / 2];
} block_q5_1;

// VDR = vec-dot unroll factor per type.
#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_1_Q8_1_MMVQ 2
#define VDR_Q2_K_Q8_1_MMVQ 1
#define VDR_Q3_K_Q8_1_MMVQ 1
#define VDR_Q4_K_Q8_1_MMVQ 2
#define VDR_Q5_K_Q8_1_MMVQ 2
#define VDR_Q6_K_Q8_1_MMVQ 1

// ---------------------------------------------------------------------------
// vec_dot impl helpers (per-quant-type)
// ---------------------------------------------------------------------------

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q4_0_q8_1_impl(const int *v, const int *u, const float &d4,
                       const gg_half2 &ds8) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
    const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
    sumi = ggml_cuda_dp4a(vi0, u[2 * i + 0], sumi);
    sumi = ggml_cuda_dp4a(vi1, u[2 * i + 1], sumi);
  }
  const float2 ds8f = gg_half22float2(ds8);
  return d4 * (sumi * ds8f.x - (8 * vdr / QI4_0) * ds8f.y);
}

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q4_1_q8_1_impl(const int *v, const int *u, const gg_half2 &dm4,
                       const gg_half2 &ds8) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
    const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
    sumi = ggml_cuda_dp4a(vi0, u[2 * i + 0], sumi);
    sumi = ggml_cuda_dp4a(vi1, u[2 * i + 1], sumi);
  }
  const float2 dm4f = gg_half22float2(dm4);
  const float2 ds8f = gg_half22float2(ds8);
  const float d4d8 = dm4f.x * ds8f.x;
  const float m4s8 = dm4f.y * ds8f.y;
  return sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q5_0_q8_1_impl(const int *vl, const int *vh, const int *u,
                       const float &d5, const gg_half2 &ds8) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    int vi0 = (vl[i] >> 0) & 0x0F0F0F0F;
    vi0 |= (vh[i] << 4) & 0x00000010;
    vi0 |= (vh[i] << 11) & 0x00001000;
    vi0 |= (vh[i] << 18) & 0x00100000;
    vi0 |= (vh[i] << 25) & 0x10000000;
    sumi = ggml_cuda_dp4a(vi0, u[2 * i + 0], sumi);

    int vi1 = (vl[i] >> 4) & 0x0F0F0F0F;
    vi1 |= (vh[i] >> 12) & 0x00000010;
    vi1 |= (vh[i] >> 5) & 0x00001000;
    vi1 |= (vh[i] << 2) & 0x00100000;
    vi1 |= (vh[i] << 9) & 0x10000000;
    sumi = ggml_cuda_dp4a(vi1, u[2 * i + 1], sumi);
  }
  const float2 ds8f = gg_half22float2(ds8);
  return d5 * (sumi * ds8f.x - (16 * vdr / QI5_0) * ds8f.y);
}

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q5_1_q8_1_impl(const int *vl, const int *vh, const int *u,
                       const gg_half2 &dm5, const gg_half2 &ds8) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    int vi0 = (vl[i] >> 0) & 0x0F0F0F0F;
    vi0 |= (vh[i] << 4) & 0x00000010;
    vi0 |= (vh[i] << 11) & 0x00001000;
    vi0 |= (vh[i] << 18) & 0x00100000;
    vi0 |= (vh[i] << 25) & 0x10000000;
    sumi = ggml_cuda_dp4a(vi0, u[2 * i + 0], sumi);

    int vi1 = (vl[i] >> 4) & 0x0F0F0F0F;
    vi1 |= (vh[i] >> 12) & 0x00000010;
    vi1 |= (vh[i] >> 5) & 0x00001000;
    vi1 |= (vh[i] << 2) & 0x00100000;
    vi1 |= (vh[i] << 9) & 0x10000000;
    sumi = ggml_cuda_dp4a(vi1, u[2 * i + 1], sumi);
  }
  const float2 dm5f = gg_half22float2(dm5);
  const float2 ds8f = gg_half22float2(ds8);
  const float d5d8 = dm5f.x * ds8f.x;
  const float m5s8 = dm5f.y * ds8f.y;
  return sumi * d5d8 + m5s8 / (QI5_1 / vdr);
}

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q8_0_q8_1_impl(const int *v, const int *u, const gg_half &d8_0,
                       const gg_half &d8_1) {
  int sumi = 0;
#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
  }
  return sumi * gg_half2float(d8_0) * gg_half2float(d8_1);
}

static __device__ __forceinline__ float
vec_dot_q2_K_q8_1_impl_mmvq(const int &v, const int *__restrict__ u,
                            const uint8_t *__restrict__ scales,
                            const gg_half2 &dm2, const float *__restrict__ d8) {
  float sumf_d = 0.0f;
  float sumf_m = 0.0f;
#pragma unroll
  for (int i = 0; i < QR2_K; ++i) {
    const int sc = scales[2 * i];
    const int vi = (v >> (2 * i)) & 0x03030303;
    sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF));
    int m = sc >> 4;
    m |= m << 8;
    m |= m << 16;
    sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0);
  }
  const float2 dm2f = gg_half22float2(dm2);
  return dm2f.x * sumf_d - dm2f.y * sumf_m;
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int &vl, const int &vh, const int *__restrict__ u,
    const uint8_t *__restrict__ scales, const int &scale_offset,
    const float &d3, const float *__restrict__ d8) {
  float sumf = 0.0f;
#pragma unroll
  for (int i = 0; i < QR3_K; ++i) {
    const int isc = scale_offset + 2 * i;
    const int isc_low = isc % (QK_K / 32);
    const int sc_shift_low = 4 * (isc / (QK_K / 32));
    const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xF;
    const int isc_high = isc % (QK_K / 64);
    const int sc_shift_high = 2 * (isc / (QK_K / 64));
    const int sc_high = ((scales[(QK_K / 32) + isc_high] >> sc_shift_high) & 3)
                        << 4;
    const int sc = (sc_low | sc_high) - 32;
    const int vil = (vl >> (2 * i)) & 0x03030303;
    const int vih = ((vh >> i) << 2) & 0x04040404;
    const int vi = __vsubss4(vil, vih);
    sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
  }
  return d3 * sumf;
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int *__restrict__ v, const int *__restrict__ u,
    const uint8_t *__restrict__ sc, const uint8_t *__restrict__ m,
    const gg_half2 &dm4, const float *__restrict__ d8) {
  float sumf_d = 0.0f;
  float sumf_m = 0.0f;
#pragma unroll
  for (int i = 0; i < QR4_K; ++i) {
    const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
    const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
    const int dot1 =
        ggml_cuda_dp4a(v1i, u[2 * i + 1], ggml_cuda_dp4a(v0i, u[2 * i + 0], 0));
    const int dot2 = ggml_cuda_dp4a(
        0x01010101, u[2 * i + 1], ggml_cuda_dp4a(0x01010101, u[2 * i + 0], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
  }
  const float2 dm4f = gg_half22float2(dm4);
  return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int *__restrict__ vl, const int *__restrict__ vh,
    const int *__restrict__ u, const uint8_t *__restrict__ sc,
    const uint8_t *__restrict__ m, const gg_half2 &dm5,
    const float *__restrict__ d8) {
  float sumf_d = 0.0f;
  float sumf_m = 0.0f;
#pragma unroll
  for (int i = 0; i < QR5_K; ++i) {
    const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
    const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
    const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
    const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
    const int v0i = vl0i | vh0i;
    const int v1i = vl1i | vh1i;
    const int dot1 =
        ggml_cuda_dp4a(v0i, u[2 * i + 0], ggml_cuda_dp4a(v1i, u[2 * i + 1], 0));
    const int dot2 = ggml_cuda_dp4a(
        0x01010101, u[2 * i + 0], ggml_cuda_dp4a(0x01010101, u[2 * i + 1], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
  }
  const float2 dm5f = gg_half22float2(dm5);
  return dm5f.x * sumf_d - dm5f.y * sumf_m;
}

static __device__ __forceinline__ float
vec_dot_q6_K_q8_1_impl_mmvq(const int &vl, const int &vh,
                            const int *__restrict__ u,
                            const int8_t *__restrict__ scales, const float &d,
                            const float *__restrict__ d8) {
  float sumf = 0.0f;
#pragma unroll
  for (int i = 0; i < QR6_K; ++i) {
    const int sc = scales[4 * i];
    const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
    const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
    const int vi = __vsubss4((vil | vih), 0x20202020);
    sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc);
  }
  return d * sumf;
}

// vec_dot wrappers for each quant type.

typedef float (*vec_dot_q_cuda_t)(const void *__restrict__ vbq,
                                  const block_q8_1 *__restrict__ bq8_1,
                                  const int &kbx, const int &iqs);

static __device__ __forceinline__ float
vec_dot_q4_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q4_0 *bq4_0 = (const block_q4_0 *)vbq + kbx;
  int v[VDR_Q4_0_Q8_1_MMVQ];
  int u[2 * VDR_Q4_0_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_uint8(bq4_0->qs, iqs + i);
    u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
  }
  return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

static __device__ __forceinline__ float
vec_dot_q4_1_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q4_1 *bq4_1 = (const block_q4_1 *)vbq + kbx;
  int v[VDR_Q4_1_Q8_1_MMVQ];
  int u[2 * VDR_Q4_1_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
    u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
  }
  return vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

static __device__ __forceinline__ float
vec_dot_q5_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q5_0 *bq5_0 = (const block_q5_0 *)vbq + kbx;
  int vl[VDR_Q5_0_Q8_1_MMVQ];
  int vh[VDR_Q5_0_Q8_1_MMVQ];
  int u[2 * VDR_Q5_0_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
    vl[i] = get_int_from_uint8(bq5_0->qs, iqs + i);
    vh[i] = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
    u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
  }
  return vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d,
                                                     bq8_1->ds);
}

static __device__ __forceinline__ float
vec_dot_q5_1_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q5_1 *bq5_1 = (const block_q5_1 *)vbq + kbx;
  int vl[VDR_Q5_1_Q8_1_MMVQ];
  int vh[VDR_Q5_1_Q8_1_MMVQ];
  int u[2 * VDR_Q5_1_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
    vl[i] = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
    vh[i] = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
    u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
  }
  return vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm,
                                                     bq8_1->ds);
}

static __device__ __forceinline__ float
vec_dot_q8_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q8_0 *bq8_0 = (const block_q8_0 *)vbq + kbx;
  int v[VDR_Q8_0_Q8_1_MMVQ];
  int u[VDR_Q8_0_Q8_1_MMVQ];
#pragma unroll
  for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
    v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
    u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
  }
  return vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d,
                                                     gg_low2half(bq8_1->ds));
}

static __device__ __forceinline__ float
vec_dot_q2_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q2_K *bq2_K = (const block_q2_K *)vbq + kbx;
  const int bq8_offset = QR2_K * (iqs / QI8_1);
  const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
  const uint8_t *scales = bq2_K->scales + scale_offset;
  const int v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
  int u[QR2_K];
  float d8[QR2_K];
#pragma unroll
  for (int i = 0; i < QR2_K; ++i) {
    u[i] = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
    d8[i] = gg_low2float(bq8_1[bq8_offset + i].ds);
  }
  return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

static __device__ __forceinline__ float
vec_dot_q3_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q3_K *bq3_K = (const block_q3_K *)vbq + kbx;
  const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
  const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
  const float d = bq3_K->d;
  const int vl = get_int_from_uint8(bq3_K->qs, iqs);
  const int vh =
      ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K / 2)) >> bq8_offset;
  int u[QR3_K];
  float d8[QR3_K];
#pragma unroll
  for (int i = 0; i < QR3_K; ++i) {
    u[i] = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
    d8[i] = gg_low2float(bq8_1[bq8_offset + i].ds);
  }
  return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d,
                                      d8);
}

static __device__ __forceinline__ float
vec_dot_q4_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q4_K *bq4_K = (const block_q4_K *)vbq + kbx;
  int v[2];
  int u[2 * QR4_K];
  float d8[QR4_K];
  const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
  const int *q4 =
      (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  v[0] = q4[0];
  v[1] = q4[4];
  const uint16_t *scales = (const uint16_t *)bq4_K->scales;
  uint16_t aux[2];
  const int j = bq8_offset / 2;
  if (j < 2) {
    aux[0] = scales[j + 0] & 0x3f3f;
    aux[1] = scales[j + 2] & 0x3f3f;
  } else {
    aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
  }
  const uint8_t *sc = (const uint8_t *)aux;
  const uint8_t *m = sc + 2;
  for (int i = 0; i < QR4_K; ++i) {
    const block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
    d8[i] = gg_low2float(bq8i->ds);
    const int *q8 = (const int *)bq8i->qs + ((iqs / 2) % 4);
    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
  }
  return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

static __device__ __forceinline__ float
vec_dot_q5_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q5_K *bq5_K = (const block_q5_K *)vbq + kbx;
  int vl[2];
  int vh[2];
  int u[2 * QR5_K];
  float d8[QR5_K];
  const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
  const int *ql =
      (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  const int *qh = (const int *)(bq5_K->qh + 4 * ((iqs / 2) % 4));
  vl[0] = ql[0];
  vl[1] = ql[4];
  vh[0] = qh[0] >> bq8_offset;
  vh[1] = qh[4] >> bq8_offset;
  const uint16_t *scales = (const uint16_t *)bq5_K->scales;
  uint16_t aux[2];
  const int j = bq8_offset / 2;
  if (j < 2) {
    aux[0] = scales[j + 0] & 0x3f3f;
    aux[1] = scales[j + 2] & 0x3f3f;
  } else {
    aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
    aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
  }
  const uint8_t *sc = (const uint8_t *)aux;
  const uint8_t *m = sc + 2;
#pragma unroll
  for (int i = 0; i < QR5_K; ++i) {
    const block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
    d8[i] = gg_low2float(bq8i->ds);
    const int *q8 = (const int *)bq8i->qs + ((iqs / 2) % 4);
    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
  }
  return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

static __device__ __forceinline__ float
vec_dot_q6_K_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {
  const block_q6_K *bq6_K = (const block_q6_K *)vbq + kbx;
  const int bq8_offset =
      2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
  const int scale_offset =
      (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
  const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
  const int vl = get_int_from_uint8(bq6_K->ql, iqs);
  const int vh =
      get_int_from_uint8(bq6_K->qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) +
                                        iqs % (QI6_K / 4)) >>
      vh_shift;
  const int8_t *scales = bq6_K->scales + scale_offset;
  int u[QR6_K];
  float d8[QR6_K];
#pragma unroll
  for (int i = 0; i < QR6_K; ++i) {
    u[i] = get_int_from_int8_aligned(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
    d8[i] = gg_low2float(bq8_1[bq8_offset + 2 * i].ds);
  }
  return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, bq6_K->d, d8);
}

// Core mat-vec-q template.

static constexpr __device__ int mmvq_nwarps_for(int ncols_dst) {
  return (ncols_dst <= 4) ? 4 : 2;
}

static constexpr __device__ int mmvq_rows_per_cuda_block_for(int ncols_dst) {
  return (ncols_dst == 1) ? 1 : 2;
}

template <typename dst_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_dst>
static __device__ void mmvq_core_impl(
    const void *__restrict__ vx,
    const block_q8_1 *__restrict__ y,
    dst_t *__restrict__ dst,
    const int ncols_x, const int nrows_x,
    const int stride_col_y, const int stride_col_dst) {

  constexpr int nwarps = mmvq_nwarps_for(ncols_dst);
  constexpr int rows_per_cuda_block = mmvq_rows_per_cuda_block_for(ncols_dst);

  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cuda_block * blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

  // Partial sums.
  float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);
    const int kqs = vdr * (tid % (qi / vdr));

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        const int row = row0 + i;
        const int weight_kbx = row * blocks_per_row_x + kbx;
        tmp[j][i] +=
            vec_dot_q_cuda(vx, &y[j * stride_col_y + kby], weight_kbx, kqs);
      }
    }
  }

  __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst]
                              [rows_per_cuda_block][WARP_SIZE];

  if (threadIdx.y > 0) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }

#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
      for (int l = 0; l < nwarps - 1; ++l) {
        tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
      }
      tmp[j][i] = warp_reduce_sum_f32(tmp[j][i]);
    }
    if (threadIdx.x < rows_per_cuda_block &&
        (rows_per_cuda_block == 1 ||
         uint32_t(row0 + threadIdx.x) < (uint32_t)nrows_x)) {
      dst[j * stride_col_dst + row0 + threadIdx.x] =
          (dst_t)tmp[j][threadIdx.x];
    }
  }
}

// ===== END DERIVED =====

// ===== BEGIN DERIVED: candle-kernels mmvq_gguf.cu, mmvq_gguf_quantize_q8_1_f32 =====
extern "C" __global__ void
mmvq_gguf_quantize_q8_1_f32(const float *__restrict__ x,
                            void *__restrict__ vy, const int kx,
                            const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1 *y = (block_q8_1 *)vy;
  const int ib = i_padded / QK8_1;
  const int iqs = i_padded % QK8_1;

  const float xi = (ix < kx) ? x[iy * kx + ix] : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max_f32(amax);
  sum = warp_reduce_sum_f32(sum);

  const float d = amax / 127.0f;
  const int8_t q = (amax == 0.0f) ? 0 : (int8_t)roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }
  reinterpret_cast<gg_half &>(y[ib].ds.x) = (gg_half)d;
  reinterpret_cast<gg_half &>(y[ib].ds.y) = (gg_half)sum;
}
// ===== END DERIVED =====

// ---------------------------------------------------------------------------------------------
// GGML: one task per blockIdx.y, candle's single-row MMVQ geometry in x/threads.
//
// `vy` holds the Q8_1 activation rows (`stride_col_y` blocks each): one per token
// (`per_slot == 0`, the gate / up projections — every slot of a token reads its token's row) or
// one per task (`per_slot != 0`, the down projection).
// ---------------------------------------------------------------------------------------------

// The expert a task routes to, bounds-checked: an id past the bank (only a corrupted route table
// can hold one) traps the launch rather than reading another allocation through the table.
static __device__ __forceinline__ unsigned int moe_expert(const unsigned int *__restrict__ ids,
                                                          int task, unsigned int n_experts) {
  const unsigned int e = ids[task];
  if (e >= n_experts) {
    __trap();
  }
  return e;
}

#define MOE_MMVQ_ENTRY(tag, block_q_t, qk_val, qi_val, vdr_val, vec_dot)                        \
  extern "C" __global__ void moe_mmvq_##tag(                                                  \
      const unsigned long long *__restrict__ experts, const void *__restrict__ vy,            \
      const unsigned int *__restrict__ ids, float *__restrict__ dst, const int ncols_x,       \
      const int nrows_x, const int stride_col_y, const int slots, const int per_slot,         \
      const unsigned int n_experts) {                                                         \
    const int task = blockIdx.y;                                                              \
    const void *vx = (const void *)experts[moe_expert(ids, task, n_experts)];                 \
    const int yrow = per_slot ? task : task / slots;                                          \
    mmvq_core_impl<float, qk_val, qi_val, block_q_t, vdr_val, vec_dot, 1>(                    \
        vx, (const block_q8_1 *)vy + (size_t)yrow * (size_t)stride_col_y,                     \
        dst + (size_t)task * (size_t)nrows_x, ncols_x, nrows_x, stride_col_y, nrows_x);       \
  }

MOE_MMVQ_ENTRY(q4_0, block_q4_0, QK4_0, QI4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1)
MOE_MMVQ_ENTRY(q4_1, block_q4_1, QK4_1, QI4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1)
MOE_MMVQ_ENTRY(q5_0, block_q5_0, QK5_0, QI5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1)
MOE_MMVQ_ENTRY(q5_1, block_q5_1, QK5_1, QI5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1)
MOE_MMVQ_ENTRY(q8_0, block_q8_0, QK8_0, QI8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1)
MOE_MMVQ_ENTRY(q2_k, block_q2_K, QK_K, QI2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1)
MOE_MMVQ_ENTRY(q3_k, block_q3_K, QK_K, QI3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1)
MOE_MMVQ_ENTRY(q4_k, block_q4_K, QK_K, QI4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1)
MOE_MMVQ_ENTRY(q5_k, block_q5_K, QK_K, QI5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1)
MOE_MMVQ_ENTRY(q6_k, block_q6_K, QK_K, QI6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1)

// ---------------------------------------------------------------------------------------------
// Dense and Q8_0-dequant GEMV: MOE_GEMV_WARPS warps per block, one output row per warp.
// ---------------------------------------------------------------------------------------------

#define MOE_GEMV_WARPS 4

typedef unsigned short u16;

static __device__ __forceinline__ float bf16_to_f32(u16 h) {
  return __uint_as_float(((unsigned int)h) << 16);
}

// Round-to-nearest-even, NaN stays a quiet NaN (candle's `__float2bfloat16`).
static __device__ __forceinline__ u16 f32_to_bf16(float f) {
  unsigned int u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) {
    return (u16)((u >> 16) | 0x0040u);
  }
  unsigned int lsb = (u >> 16) & 1u;
  return (u16)((u + 0x7fffu + lsb) >> 16);
}

static __device__ __forceinline__ float f16_to_f32(u16 h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}

static __device__ __forceinline__ u16 f32_to_f16(float f) {
  u16 h;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
  return h;
}

// The activation / output element types: load to f32 exactly, store rounded once.
struct TF32 {
  typedef float T;
  static const int VEC = 4;
  static __device__ __forceinline__ float dot16(uint4 w, uint4 x, float acc) {
    acc = fmaf(__uint_as_float(w.x), __uint_as_float(x.x), acc);
    acc = fmaf(__uint_as_float(w.y), __uint_as_float(x.y), acc);
    acc = fmaf(__uint_as_float(w.z), __uint_as_float(x.z), acc);
    acc = fmaf(__uint_as_float(w.w), __uint_as_float(x.w), acc);
    return acc;
  }
  static __device__ __forceinline__ float load(const float *p, size_t i) { return p[i]; }
  static __device__ __forceinline__ void store(float *p, size_t i, float v) { p[i] = v; }
  // `v` rounded to this type and widened back: what a tensor of this dtype holds.
  static __device__ __forceinline__ float round(float v) { return v; }
};
// Eight 16-bit elements of a uint4 in memory order, widened by `cvt`.
#define DOT16_U16(CVT)                                                                           \
  static __device__ __forceinline__ float dot16(uint4 w, uint4 x, float acc) {                   \
    const unsigned int ws[4] = {w.x, w.y, w.z, w.w};                                             \
    const unsigned int xs[4] = {x.x, x.y, x.z, x.w};                                             \
    _Pragma("unroll") for (int i = 0; i < 4; ++i) {                                              \
      acc = fmaf(CVT((u16)(ws[i] & 0xffffu)), CVT((u16)(xs[i] & 0xffffu)), acc);                 \
      acc = fmaf(CVT((u16)(ws[i] >> 16)), CVT((u16)(xs[i] >> 16)), acc);                         \
    }                                                                                            \
    return acc;                                                                                  \
  }

struct TBF16 {
  typedef u16 T;
  static const int VEC = 8;
  DOT16_U16(bf16_to_f32)
  static __device__ __forceinline__ float load(const u16 *p, size_t i) { return bf16_to_f32(p[i]); }
  static __device__ __forceinline__ void store(u16 *p, size_t i, float v) { p[i] = f32_to_bf16(v); }
  static __device__ __forceinline__ float round(float v) { return bf16_to_f32(f32_to_bf16(v)); }
};
struct TF16 {
  typedef u16 T;
  static const int VEC = 8;
  DOT16_U16(f16_to_f32)
  static __device__ __forceinline__ float load(const u16 *p, size_t i) { return f16_to_f32(p[i]); }
  static __device__ __forceinline__ void store(u16 *p, size_t i, float v) { p[i] = f32_to_f16(v); }
  static __device__ __forceinline__ float round(float v) { return f16_to_f32(f32_to_f16(v)); }
};

static __device__ __forceinline__ float moe_warp_sum(float v) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    v += __shfl_xor_sync(0xffffffff, v, mask, 32);
  }
  return v;
}

// Dense: weight row `row` of expert `e` is `k` contiguous elements of the activation's dtype.
// `vec` (host-decided per bank and shape, never per call: K a multiple of D::VEC and every expert
// base 16-byte aligned; the activation is always passed 16-byte aligned) selects 16-byte loads.
template <typename D>
static __device__ __forceinline__ void moe_gemv_dense(const unsigned long long *__restrict__ experts,
                                                      const typename D::T *__restrict__ x,
                                                      const unsigned int *__restrict__ ids,
                                                      typename D::T *__restrict__ y, int n, int k,
                                                      int slots, int per_slot, int vec,
                                                      unsigned int n_experts) {
  const int task = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * MOE_GEMV_WARPS + warp;
  if (row >= n) {
    return;
  }
  const typename D::T *w =
      (const typename D::T *)experts[moe_expert(ids, task, n_experts)] + (size_t)row * (size_t)k;
  const typename D::T *xr = x + (size_t)(per_slot ? task : task / slots) * (size_t)k;
  float acc = 0.0f;
  if (vec) {
    // 16-byte loads: D::VEC elements per lane per step, in element order.
    for (int c = lane * D::VEC; c < k; c += 32 * D::VEC) {
      const uint4 wv = *(const uint4 *)(w + c);
      const uint4 xv = *(const uint4 *)(xr + c);
      acc = D::dot16(wv, xv, acc);
    }
  } else {
    for (int c = lane; c < k; c += 32) {
      acc = fmaf(D::load(w, c), D::load(xr, c), acc);
    }
  }
  acc = moe_warp_sum(acc);
  if (lane == 0) {
    D::store(y, (size_t)task * (size_t)n + row, acc);
  }
}

// Q8_0 (the MLX-affine Q8 tier): 34-byte blocks of {f16 d, int8 qs[32]} along K. Each lane takes
// whole blocks; an element is `round_D((float)q * d)` — `d * q` is exact in f32 (an int8 times
// an f16 value), and `round_D` is the one rounding `dequantize(..).to_dtype(D)` applies.
template <typename D>
static __device__ __forceinline__ void moe_gemv_q8_0(const unsigned long long *__restrict__ experts,
                                                     const typename D::T *__restrict__ x,
                                                     const unsigned int *__restrict__ ids,
                                                     typename D::T *__restrict__ y, int n, int k,
                                                     int slots, int per_slot,
                                                     unsigned int n_experts) {
  const int task = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * MOE_GEMV_WARPS + warp;
  if (row >= n) {
    return;
  }
  const int blocks = k / QK8_0;
  const block_q8_0 *w =
      (const block_q8_0 *)experts[moe_expert(ids, task, n_experts)] + (size_t)row * (size_t)blocks;
  const typename D::T *xr = x + (size_t)(per_slot ? task : task / slots) * (size_t)k;
  float acc = 0.0f;
  for (int b = lane; b < blocks; b += 32) {
    const float d = (float)w[b].d;
    const typename D::T *xb = xr + (size_t)b * QK8_0;
#pragma unroll 8
    for (int i = 0; i < QK8_0; ++i) {
      acc = fmaf(D::round((float)w[b].qs[i] * d), D::load(xb, i), acc);
    }
  }
  acc = moe_warp_sum(acc);
  if (lane == 0) {
    D::store(y, (size_t)task * (size_t)n + row, acc);
  }
}

#define MOE_GEMV_ENTRIES(tag, D)                                                                \
  extern "C" __global__ void __launch_bounds__(MOE_GEMV_WARPS * 32) moe_gemv_dense_##tag(      \
      const unsigned long long *__restrict__ experts, const D::T *__restrict__ x,               \
      const unsigned int *__restrict__ ids, D::T *__restrict__ y, int n, int k, int slots,      \
      int per_slot, int vec, unsigned int n_experts) {                                          \
    moe_gemv_dense<D>(experts, x, ids, y, n, k, slots, per_slot, vec, n_experts);               \
  }                                                                                             \
  extern "C" __global__ void __launch_bounds__(MOE_GEMV_WARPS * 32) moe_gemv_q8_0_##tag(       \
      const unsigned long long *__restrict__ experts, const D::T *__restrict__ x,               \
      const unsigned int *__restrict__ ids, D::T *__restrict__ y, int n, int k, int slots,      \
      int per_slot, unsigned int n_experts) {                                                   \
    moe_gemv_q8_0<D>(experts, x, ids, y, n, k, slots, per_slot, n_experts);                     \
  }

MOE_GEMV_ENTRIES(f32, TF32)
MOE_GEMV_ENTRIES(bf16, TBF16)
MOE_GEMV_ENTRIES(f16, TF16)
