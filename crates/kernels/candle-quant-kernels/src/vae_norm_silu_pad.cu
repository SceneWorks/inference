// SW-6133 (SW-F287 S8): fused video-VAE channel norm + SiLU + temporal pad. One pass reads the
// raw activation x [B, C, T, H, W] and writes the temporally padded conv input that the VAE's
// causal 3-D convolution consumes, so neither the normalized tensor, the SiLU tensor nor the
// concatenated / padded copies of the op chain are ever materialized.
//
// Written from the in-tree reference semantics: candle-gen-wan `vae.rs` (`ChanNorm` -> `.silu()`
// -> `conv3d.rs` `CausalConv3d` zero pad / `feat_cache` concat) and candle-gen-ltx `vae.rs`
// (`pixel_norm` -> SiLU -> `conv3d.rs` frame-replication pad). The idea of writing norm + SiLU
// straight into the padded conv buffer follows comfy-kitchen's `group_norm_pad3d.cu` design, but
// that kernel is a GroupNorm over NDHWC; no comfy-kitchen source is copied.
//
// Output frame j of the padded time axis (Tp frames) is, in order:
//   [0, zero_front)                       zeros (Wan's causal pad before the cache has warmed)
//   [zero_front, zero_front + n_cache)    the cached frames, already activated, copied bit for bit
//   [.., Tp - zero_back)                  act(x[t]) with t = clamp(j - zero_front - n_cache -
//                                         rep_front, 0, T - 1) (LTX's first/last-frame
//                                         replication is the clamp)
//   [Tp - zero_back, Tp)                  zeros
// act(v)_c = silu(norm_c(v)), computed in f32 and rounded once to the output dtype:
//   norm_kind 0 (Wan channel L2):  y = (v_c / min(max(sqrt(ss), eps), 1e30)) * scale * gamma_c
//   norm_kind 1 (LTX pixel RMS):   y = v_c / sqrt(ss * inv_c + eps)
//   silu(y) = y / (1 + expf(-y))            (candle's `silu_fwd`)
// with every operation the op chain's own, in its order, with no contraction (__f*_rn): the f32
// result is the op chain's bit for bit. ss = sum_c v_c^2 is the f32 chain's
// `to_dtype(F32).sqr().sum_keepdim(1)`: the rounded squares added in candle's `fast_sum` order (one
// block of P = next_pow2(min(1024, C)) threads, thread i summing squares i, i + P, ... from 0, then
// the shared-memory tree `shr[i] += shr[i + s]` for s = P/2 .. 1). That tree visits its leaves in
// bit-reversed index order, so a binary-counter pairwise accumulation over leaf bitrev(k), k = 0 ..
// P-1, performs exactly its additions (IEEE addition commutes) in ten named registers.
// Every tensor is addressed by its own (batch, channel, time) element strides with the H*W plane
// contiguous, so the cache and x may be any view and the output may be channel-major
// [B, C, Tp, H, W] or frame-major [B, Tp, C, H, W] (the merged conv2d batch layout). All index
// math is 64-bit.
//
// Single file, no includes (nvrtc seam rule). No per-thread local arrays.

#define NSP_THREADS 256

typedef unsigned short nsp_u16;

__device__ __forceinline__ float nsp_ld(const float* p, long long i, int) { return p[i]; }

// A 16-bit element as f32 exactly as candle's cast does it: bf16 is the high half of the binary32
// word; f16 widens with cvt.f32.f16.
__device__ __forceinline__ float nsp_ld(const nsp_u16* p, long long i, int bf16) {
    const nsp_u16 h = p[i];
    if (bf16) return __uint_as_float(((unsigned)h) << 16);
    float y;
    asm("cvt.f32.f16 %0, %1;" : "=f"(y) : "h"(h));
    return y;
}

__device__ __forceinline__ void nsp_st(float* p, long long i, float v, int) { p[i] = v; }

// Round-to-nearest-even narrowing (NaN kept quiet), the same rounding candle's casts use.
__device__ __forceinline__ void nsp_st(nsp_u16* p, long long i, float v, int bf16) {
    nsp_u16 h;
    if (bf16) {
        const unsigned bits = __float_as_uint(v);
        if ((bits & 0x7fffffffu) > 0x7f800000u) {
            h = (nsp_u16)((bits >> 16) | 0x0040u);
        } else {
            const unsigned lsb = (bits >> 16) & 1u;
            h = (nsp_u16)((bits + 0x7fffu + lsb) >> 16);
        }
    } else {
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    }
    p[i] = h;
}

// One level of the binary-counter pairwise sum: a set bit merges the pending left subtree of this
// level into the carry; a clear bit parks the carry there and ends this leaf's merging. Past the
// tree's depth the carry is the finished sum.
#define NSP_LEVEL(l, p)                                                                            \
    if (tree_levels == (l)) {                                                                      \
        ss = carry;                                                                                \
        continue;                                                                                  \
    }                                                                                              \
    if (k & (1 << (l))) {                                                                          \
        carry = __fadd_rn(p, carry);                                                               \
    } else {                                                                                       \
        p = carry;                                                                                 \
        continue;                                                                                  \
    }

template <typename T>
__device__ __forceinline__ void nsp_body(
    const T* x, long long x_off, long long xsb, long long xsc, long long xst,
    const T* gamma, long long g_off,
    const T* cache, long long c_off, long long csb, long long csc, long long cst, int n_cache,
    T* out, long long osb, long long osc, long long ost,
    int batch, int channels, int frames, int padded_frames, long long hw,
    int zero_front, int rep_front, int zero_back,
    int norm_kind, float scale, float eps, float inv_c, int bf16) {
    // candle's fast_sum block width for a C-element reduction, and its log2 (<= 10).
    int tree_width = 1, tree_levels = 0;
    while (tree_width < channels && tree_width < 1024) {
        tree_width <<= 1;
        ++tree_levels;
    }
    const unsigned long long positions =
        (unsigned long long)batch * (unsigned long long)padded_frames * (unsigned long long)hw;
    const unsigned long long stride = (unsigned long long)gridDim.x * NSP_THREADS;
    for (unsigned long long p = (unsigned long long)blockIdx.x * NSP_THREADS + threadIdx.x;
         p < positions; p += stride) {
        const long long s = (long long)(p % (unsigned long long)hw);
        const unsigned long long bj = p / (unsigned long long)hw;
        const int j = (int)(bj % (unsigned long long)padded_frames);
        const long long b = (long long)(bj / (unsigned long long)padded_frames);
        const long long o = b * osb + (long long)j * ost + s;
        if (j < zero_front || j >= padded_frames - zero_back) {
            for (int c = 0; c < channels; ++c) out[o + (long long)c * osc] = (T)0;
            continue;
        }
        if (j < zero_front + n_cache) {
            const long long src = c_off + b * csb + (long long)(j - zero_front) * cst + s;
            for (int c = 0; c < channels; ++c) out[o + (long long)c * osc] = cache[src + (long long)c * csc];
            continue;
        }
        int t = j - zero_front - n_cache - rep_front;
        t = t < 0 ? 0 : (t >= frames ? frames - 1 : t);
        const long long xi = x_off + b * xsb + (long long)t * xst + s;
        float ss = 0.0f;
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f, p4 = 0.0f;
        float p5 = 0.0f, p6 = 0.0f, p7 = 0.0f, p8 = 0.0f, p9 = 0.0f;
        for (int k = 0; k < tree_width; ++k) {
            const int i = tree_levels ? (int)(__brev((unsigned)k) >> (32 - tree_levels)) : 0;
            float carry = 0.0f;
            for (int c = i; c < channels; c += tree_width) {
                const float v = nsp_ld(x, xi + (long long)c * xsc, bf16);
                carry = __fadd_rn(carry, __fmul_rn(v, v));
            }
            NSP_LEVEL(0, p0) NSP_LEVEL(1, p1) NSP_LEVEL(2, p2) NSP_LEVEL(3, p3) NSP_LEVEL(4, p4)
            NSP_LEVEL(5, p5) NSP_LEVEL(6, p6) NSP_LEVEL(7, p7) NSP_LEVEL(8, p8) NSP_LEVEL(9, p9)
            ss = carry;
        }
        float denom;
        if (norm_kind == 0) {
            denom = fminf(fmaxf(sqrtf(ss), eps), 1e30f);
        } else {
            // candle's `sum / c` then `+ eps` are two affine launches: two roundings, no fma.
            denom = sqrtf(__fadd_rn(__fmul_rn(ss, inv_c), eps));
        }
        for (int c = 0; c < channels; ++c) {
            float y = __fdiv_rn(nsp_ld(x, xi + (long long)c * xsc, bf16), denom);
            if (norm_kind == 0) {
                y = __fmul_rn(__fmul_rn(y, scale), nsp_ld(gamma, g_off + c, bf16));
            }
            nsp_st(out, o + (long long)c * osc, __fdiv_rn(y, __fadd_rn(1.0f, expf(-y))), bf16);
        }
    }
}

#define NSP_ENTRY(NAME, T, BF16)                                                                   \
    extern "C" __global__ void NAME(                                                               \
        const T* x, long long x_off, long long xsb, long long xsc, long long xst,                 \
        const T* gamma, long long g_off,                                                           \
        const T* cache, long long c_off, long long csb, long long csc, long long cst, int n_cache, \
        T* out, long long osb, long long osc, long long ost,                                       \
        int batch, int channels, int frames, int padded_frames, long long hw,                      \
        int zero_front, int rep_front, int zero_back,                                              \
        int norm_kind, float scale, float eps, float inv_c) {                                      \
        nsp_body<T>(x, x_off, xsb, xsc, xst, gamma, g_off, cache, c_off, csb, csc, cst, n_cache,   \
                    out, osb, osc, ost, batch, channels, frames, padded_frames, hw, zero_front,   \
                    rep_front, zero_back, norm_kind, scale, eps, inv_c, BF16);                     \
    }

NSP_ENTRY(vae_norm_silu_pad_f32, float, 0)
NSP_ENTRY(vae_norm_silu_pad_bf16, nsp_u16, 1)
NSP_ENTRY(vae_norm_silu_pad_f16, nsp_u16, 0)
