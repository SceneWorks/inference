// SW-6128 (SW-F287 S3): fused strided RoPE and fused QK-RMSNorm + RoPE for the media lane.
//
// Written from the in-tree reference semantics (candle-gen `rope::apply_rope_reference` /
// `rope::rms_norm_rope_reference`, which are the narrow/mul/cat chains Krea, the HF half-split
// copies and FLUX's freqs_cis apply carried, and candle-kernels' `rmsnorm`). The design follows
// comfy-kitchen's `rms_rope.cu` / `rope_device.cuh` (q/k addressed through their own strides so a
// permuted or fused-QKV view needs no copy, an in-place form, both pair layouts, partial rotary
// dim), but no comfy-kitchen source is copied.
//
// LAYOUT. A launch sees x as rows of `d` lanes (lane stride 1). A row is indexed by three row
// coordinates (i0, i1, i2) — the host maps (batch, seq, heads) onto them in memory order — each
// with its own element stride for x, for the output and for the cos/sin tables (a table stride of
// 0 broadcasts, e.g. over heads). Offsets are 64-bit, so a view deep into a large buffer (q/k of a
// fused QKV projection) is addressed in place. The output may be x itself (the in-place form: the
// host has checked the view does not overlap itself) or a fresh contiguous buffer.
//
// PAIRING. `split == 0`: interleaved (GPT-J / complex) pairs (2p, 2p+1). `split == 1`: NeoX /
// rotate-half pairs (p, p + rd/2). Lanes >= rd pass through. A table is either half width (one
// cos/sin per pair, `full == 0`) or full width (one per lane, `full == 1`: lane l reads entry l).
//
// NUMERICS CONTRACT (the reference's own operations, in its order; `__f*_rn` never contract):
//   F32 math (`native == 0`): x and the tables widen to f32, then
//       interleaved  y0 = x0*c0 - x1*s0         y1 = x0*s1 + x1*c1
//       split        y0 = x0*c0 + (-x1)*s0      y1 = x1*c1 + x0*s1
//     each product and the sum rounded in f32, the result rounded once to x's dtype. That is the
//     reference chain `to_dtype(F32) -> broadcast_mul x2 -> sub/add -> to_dtype` bit for bit.
//   Native math (`native == 1`): the reference casts the tables to x's dtype and multiplies / adds
//     in x's dtype; every product and the sum are rounded to the dtype here. A bf16 / f16 product
//     or sum of two dtype values computed in f32 and rounded once equals the hardware's correctly
//     rounded half op (the f32 result is exact, or far from a dtype tie), so this is exact too.
//   RMSNorm (rms kernel): candle-kernels `rmsnorm` for d < 1024 — one warp per row, lane t
//     accumulates columns t, t+32, ... as fma(x, x, acc) (nvcc contracts candle's `tmp += xi*xi`),
//     a shfl_xor butterfly, mean = acc / d (IEEE), r = rsqrtf(mean + eps), y = (r * x) * w, the
//     normalized value rounded to x's dtype (the reference's `rms_norm(...).to_dtype(dt)`), then
//     the RoPE above on the rounded values. In native math the weight is rounded to the dtype
//     first (the reference's `w.to_dtype(dt)`).
//
// Single file, no includes (nvrtc seam rule). No per-thread local arrays.

#define DT_F32 0
#define DT_BF16 1
#define DT_F16 2
#define RMS_ROPE_WARPS 4

__device__ __forceinline__ float mr_bf16_to_f32(unsigned short h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

// Round-to-nearest-even, no flush-to-zero: `__float2bfloat16` for every finite input (NaN stays a
// quiet NaN). Software so the source needs no sm_80 cvt.
__device__ __forceinline__ unsigned short mr_f32_to_bf16(float f) {
    unsigned int u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (unsigned short)((u >> 16) | 0x0040u);
    }
    unsigned int lsb = (u >> 16) & 1u;
    return (unsigned short)((u + 0x7fffu + lsb) >> 16);
}

__device__ __forceinline__ float mr_f16_to_f32(unsigned short h) {
    float y;
    asm("cvt.f32.f16 %0, %1;" : "=f"(y) : "h"(h));
    return y;
}

__device__ __forceinline__ unsigned short mr_f32_to_f16(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

template <int DT> struct Elem;

template <> struct Elem<DT_F32> {
    typedef float S;
    static __device__ __forceinline__ float load(S v) { return v; }
    static __device__ __forceinline__ S store(float v) { return v; }
    static __device__ __forceinline__ float rnd(float v) { return v; }
};

template <> struct Elem<DT_BF16> {
    typedef unsigned short S;
    static __device__ __forceinline__ float load(S v) { return mr_bf16_to_f32(v); }
    static __device__ __forceinline__ S store(float v) { return mr_f32_to_bf16(v); }
    static __device__ __forceinline__ float rnd(float v) { return mr_bf16_to_f32(mr_f32_to_bf16(v)); }
};

template <> struct Elem<DT_F16> {
    typedef unsigned short S;
    static __device__ __forceinline__ float load(S v) { return mr_f16_to_f32(v); }
    static __device__ __forceinline__ S store(float v) { return mr_f32_to_f16(v); }
    static __device__ __forceinline__ float rnd(float v) { return mr_f16_to_f32(mr_f32_to_f16(v)); }
};

// The rotation of one pair (x0 at lane l0, x1 at lane l1) with its table entries, per the
// numerics contract above: y0 = x0*c0 - x1*s0, y1 = x0*s1 + x1*c1 for both pairings (the split
// chain's `(-x1)*s0` is the same IEEE value as `-(x1*s0)`, and its `x1*c1 + x0*s1` the same sum).
// `native` rounds each table entry, each product and the sum to the dtype.
template <int DT>
__device__ __forceinline__ void mr_rotate(float x0, float x1, float c0, float s0, float c1,
                                          float s1, int native, float* y0, float* y1) {
    typedef Elem<DT> E;
    if (native) {
        c0 = E::rnd(c0);
        s0 = E::rnd(s0);
        c1 = E::rnd(c1);
        s1 = E::rnd(s1);
    }
    float a = __fmul_rn(x0, c0);
    float b = __fmul_rn(x1, s0);
    float c = __fmul_rn(x0, s1);
    float e = __fmul_rn(x1, c1);
    if (native) {
        a = E::rnd(a);
        b = E::rnd(b);
        c = E::rnd(c);
        e = E::rnd(e);
    }
    *y0 = __fsub_rn(a, b);
    *y1 = __fadd_rn(c, e);
}

// Lanes of item `p` of a row: a rotated pair (p < rd/2) or up to two pass-through lanes.
__device__ __forceinline__ void mr_item_lanes(int p, int d, int rd, int split, int* l0, int* l1,
                                              int* rotate) {
    int half = rd >> 1;
    if (p < half) {
        *rotate = 1;
        if (split) {
            *l0 = p;
            *l1 = p + half;
        } else {
            *l0 = 2 * p;
            *l1 = 2 * p + 1;
        }
    } else {
        *rotate = 0;
        *l0 = rd + 2 * (p - half);
        *l1 = (*l0 + 1 < d) ? *l0 + 1 : -1;
    }
}

// ---------------------------------------------------------------------------------------------
// RoPE: one thread per item (a rotated pair, or a pass-through lane pair when the output is not
// x itself). Grid-stride over rows * items.
// ---------------------------------------------------------------------------------------------
template <int DT>
__device__ __forceinline__ void mr_rope(const typename Elem<DT>::S* x, typename Elem<DT>::S* y,
                                        const float* cs, const float* sn, long long n_items,
                                        int items, int n1, int n2, long long xo, long long xs0,
                                        long long xs1, long long xs2, long long yo, long long ys0,
                                        long long ys1, long long ys2, long long to, long long ts0,
                                        long long ts1, long long ts2, long long tl, int d, int rd,
                                        int split, int full, int native) {
    typedef Elem<DT> E;
    // Index math in 32 bits (the host caps rows * items at i32::MAX, E6); offsets in 64.
    const unsigned total = (unsigned)n_items;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        unsigned row = i / (unsigned)items;
        int p = (int)(i - row * (unsigned)items);
        long long i2 = row % (unsigned)n2;
        unsigned r1 = row / (unsigned)n2;
        long long i1 = r1 % (unsigned)n1;
        long long i0 = r1 / (unsigned)n1;
        long long xb = xo + i0 * xs0 + i1 * xs1 + i2 * xs2;
        long long yb = yo + i0 * ys0 + i1 * ys1 + i2 * ys2;
        int l0, l1, rotate;
        mr_item_lanes(p, d, rd, split, &l0, &l1, &rotate);
        if (rotate) {
            long long tb = to + i0 * ts0 + i1 * ts1 + i2 * ts2;
            long long t0 = tb + (long long)(full ? l0 : p) * tl;
            long long t1 = tb + (long long)(full ? l1 : p) * tl;
            float y0, y1;
            mr_rotate<DT>(E::load(x[xb + l0]), E::load(x[xb + l1]), cs[t0], sn[t0], cs[t1],
                          sn[t1], native, &y0, &y1);
            y[yb + l0] = E::store(y0);
            y[yb + l1] = E::store(y1);
        } else {
            y[yb + l0] = x[xb + l0];
            if (l1 >= 0) y[yb + l1] = x[xb + l1];
        }
    }
}

#define MR_ROPE_ENTRY(NAME, DT)                                                                 \
    extern "C" __global__ void NAME(                                                            \
        const Elem<DT>::S* x, Elem<DT>::S* y, const float* cs, const float* sn,                 \
        long long n_items, int items, int n1, int n2, long long xo, long long xs0,              \
        long long xs1, long long xs2, long long yo, long long ys0, long long ys1,               \
        long long ys2, long long to, long long ts0, long long ts1, long long ts2, long long tl, \
        int d, int rd, int split, int full, int native) {                                       \
        mr_rope<DT>(x, y, cs, sn, n_items, items, n1, n2, xo, xs0, xs1, xs2, yo, ys0, ys1, ys2, \
                    to, ts0, ts1, ts2, tl, d, rd, split, full, native);                         \
    }

MR_ROPE_ENTRY(media_rope_f32, DT_F32)
MR_ROPE_ENTRY(media_rope_bf16, DT_BF16)
MR_ROPE_ENTRY(media_rope_f16, DT_F16)

// ---------------------------------------------------------------------------------------------
// RMSNorm + RoPE: one warp per row (blockDim = (32, RMS_ROPE_WARPS)), d < 1024. Phase 1 is
// candle's rmsnorm reduction; phase 2 re-reads the lanes a thread owns, normalizes and rounds
// them, and rotates. A thread writes only lanes it alone reads in phase 2, and every lane's phase
// 1 read precedes the warp shuffle, so the in-place form (y == x) is race-free.
// ---------------------------------------------------------------------------------------------
template <int DT>
__device__ __forceinline__ void mr_rms_rope(const typename Elem<DT>::S* x, typename Elem<DT>::S* y,
                                            const float* w, const float* cs, const float* sn,
                                            long long rows, int n1, int n2, long long xo,
                                            long long xs0, long long xs1, long long xs2,
                                            long long yo, long long ys0, long long ys1,
                                            long long ys2, long long to, long long ts0,
                                            long long ts1, long long ts2, long long tl, int d,
                                            int rd, int split, int full, int native, float eps) {
    typedef Elem<DT> E;
    unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= (unsigned)rows) return;  // warp-uniform
    int lane = threadIdx.x;
    long long i2 = row % (unsigned)n2;
    unsigned r1 = row / (unsigned)n2;
    long long i1 = r1 % (unsigned)n1;
    long long i0 = r1 / (unsigned)n1;
    long long xb = xo + i0 * xs0 + i1 * xs1 + i2 * xs2;
    long long yb = yo + i0 * ys0 + i1 * ys1 + i2 * ys2;
    long long tb = to + i0 * ts0 + i1 * ts1 + i2 * ts2;

    float acc = 0.0f;
    for (int col = lane; col < d; col += 32) {
        float v = E::load(x[xb + col]);
        acc = __fmaf_rn(v, v, acc);
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, mask, 32));
    }
    float mean = __fdiv_rn(acc, (float)d);
    float r = rsqrtf(__fadd_rn(mean, eps));

    int items = (d + 1) >> 1;
    for (int p = lane; p < items; p += 32) {
        int l0, l1, rotate;
        mr_item_lanes(p, d, rd, split, &l0, &l1, &rotate);
        float w0 = w[l0];
        float n0 = E::rnd(__fmul_rn(__fmul_rn(r, E::load(x[xb + l0])), native ? E::rnd(w0) : w0));
        float n1v = 0.0f;
        if (l1 >= 0) {
            float w1 = w[l1];
            n1v = E::rnd(__fmul_rn(__fmul_rn(r, E::load(x[xb + l1])), native ? E::rnd(w1) : w1));
        }
        if (rotate) {
            long long t0 = tb + (long long)(full ? l0 : p) * tl;
            long long t1 = tb + (long long)(full ? l1 : p) * tl;
            float y0, y1;
            mr_rotate<DT>(n0, n1v, cs[t0], sn[t0], cs[t1], sn[t1], native, &y0, &y1);
            y[yb + l0] = E::store(y0);
            y[yb + l1] = E::store(y1);
        } else {
            y[yb + l0] = E::store(n0);
            if (l1 >= 0) y[yb + l1] = E::store(n1v);
        }
    }
}

#define MR_RMS_ROPE_ENTRY(NAME, DT)                                                              \
    extern "C" __global__ void NAME(                                                             \
        const Elem<DT>::S* x, Elem<DT>::S* y, const float* w, const float* cs, const float* sn,  \
        long long rows, int n1, int n2, long long xo, long long xs0, long long xs1,              \
        long long xs2, long long yo, long long ys0, long long ys1, long long ys2, long long to,  \
        long long ts0, long long ts1, long long ts2, long long tl, int d, int rd, int split,     \
        int full, int native, float eps) {                                                       \
        mr_rms_rope<DT>(x, y, w, cs, sn, rows, n1, n2, xo, xs0, xs1, xs2, yo, ys0, ys1, ys2, to, \
                        ts0, ts1, ts2, tl, d, rd, split, full, native, eps);                     \
    }

MR_RMS_ROPE_ENTRY(media_rms_norm_rope_f32, DT_F32)
MR_RMS_ROPE_ENTRY(media_rms_norm_rope_bf16, DT_BF16)
MR_RMS_ROPE_ENTRY(media_rms_norm_rope_f16, DT_F16)
