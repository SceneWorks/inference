// Fused NVFP4 decode GEMV (sc-24136, epic sc-24128): y[m, n] = gs * sum_k x[m, k] * W[n, k] for
// the M = 1..=8 token rows of a decode step, W a resident packed NVFP4 weight, x a bf16 activation
// that is NOT quantized (W4A16 math on the packed weight; the cuBLASLt W4A4 path quantizes x to
// FP4 first). One kernel serves every M (a runtime argument).
//
// Compiled at runtime through the nvrtc compile-once seam (`nvrtc.rs`): builtins and inline PTX
// only, no headers. Needs sm_80 (`mma.sync ... .bf16`, `fma.rn.bf16x2`).
//
// WEIGHT LAYOUT (read exactly as `Nvfp4Weight` stages it; see `nvfp4.rs` and `cublaslt.rs`
// `cublaslt_scale_layout`):
//
// - `packed`: row-major [N, cols_padded / 2] bytes, cols_padded a multiple of 32. Byte j of a row
//   holds column 2j in its LOW nibble and column 2j+1 in its HIGH nibble. E2M1 code: sign in bit
//   3, magnitude {0, 0.5, 1, 1.5, 2, 3, 4, 6}. Columns >= K are zero nibbles.
// - `scales`: one OCP FP8 E4M3 byte per (row, 16-column block), in cuBLASLt's ROW-MAJOR
//   scale-factor-atom layout: atoms of 128 rows x 4 blocks (512 bytes), atom index
//   `k_atom + num_k_atoms * m_atom` (k-atom fastest), intra-atom offset
//   `(r % 32) * 16 + ((r % 128) / 32) * 4 + (block % 4)`. So the scale bytes of blocks 4u..4u+3 of
//   one row are 4 CONSECUTIVE bytes: one aligned u32 per row per 64-column unit.
// - `gs`: the FP32 per-tensor scale. Dequant: W[n, k] = E2M1(nib) * E4M3(scale[n, k / 16]) * gs.
//
// WORK SPLIT: a block of MMA_WARPS warps owns 16 output rows (N is a multiple of 16 for every
// NVFP4 weight); its warps split K (warp w takes 64-column units w, w + MMA_WARPS, ...), software
// pipelined one unit ahead, and their 16 x 8 f32 tiles are summed in shared memory at the end. Per
// unit a warp issues four `mma.m16n8k16` (bf16 in, f32 accumulate): A = 16 weight rows x 16 K,
// B = 16 K x 8 activation rows (rows >= M are zero), C = 16 x 8 f32.
//
// The mma's K order is free as long as A and B agree, so each lane takes the K positions it can
// load contiguously: lane (g = lane / 4, t = lane % 4) owns the 16 columns 16t..16t+15 of the unit
// (one 16-column block, so one scale byte per row) as 8 packed bytes of rows g and g + 8 (the four
// lanes of a group read 32 contiguous bytes of a row) and 16 bf16 of activation row g. At mma step
// s (0..3) its A registers hold columns 16t + 4s + {0,1} (byte 2s) and 16t + 4s + {2,3} (byte
// 2s + 1); its B registers the same columns of x. The activation is read straight from global
// memory (tiny, L1/L2-resident): 16-byte loads when K % 8 == 0 (`vec_x`), element-wise with bounds
// checks otherwise; columns >= K read as zero.
//
// NUMERICS: dequant is EXACT in bf16 — E2M1 (2 significant bits) x E4M3 (4) fits bf16's 8 and the
// magnitude range [2^-10, 2688] is in range, so `bf16(E2M1 * E4M3)` (a 256-entry byte -> bf16x2
// E2M1 table in shared memory, times the block scale with `fma.rn.bf16x2`) loses nothing. The tensor
// core forms exact bf16 x bf16 products and accumulates in f32; the warps' partial tiles are summed
// in f32, scaled by `gs` and rounded ONCE to bf16 (round-to-nearest-even). No activation
// quantization.

typedef unsigned short bf16_t;
typedef unsigned int u32;

// Round-to-nearest-even, NaN stays a quiet NaN (same as `cvt.rn.bf16.f32`).
__device__ __forceinline__ bf16_t f32_to_bf16(float f) {
    u32 u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (bf16_t)((u >> 16) | 0x0040u);
    }
    u32 lsb = (u >> 16) & 1u;
    return (bf16_t)((u + 0x7fffu + lsb) >> 16);
}

// OCP FP8 E4M3 (bias 7, subnormals at E == 0, S.1111.111 = NaN) — the codec's `e4m3_to_f32`.
__device__ __forceinline__ float e4m3_to_f32(u32 b) {
    u32 e = (b >> 3) & 0xfu;
    u32 m = b & 7u;
    float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0.001953125f;
    if ((b & 0x7fu) == 0x7fu) {
        v = __uint_as_float(0x7fc00000u);
    }
    return (b & 0x80u) ? -v : v;
}

// E2M1 code (low 4 bits) -> f32 — the codec's `E2M1_LUT`.
__device__ __forceinline__ float e2m1_to_f32(u32 c) {
    u32 e = (c >> 1) & 3u;
    u32 m = c & 1u;
    float v = e ? __uint_as_float(((e + 126u) << 23) | (m << 22)) : (m ? 0.5f : 0.0f);
    return (c & 8u) ? -v : v;
}

#define MMA_WARPS 8
#define MMA_THREADS (MMA_WARPS * 32)

// The high half of an f32 whose low 16 bits are zero: its exact bf16.
__device__ __forceinline__ u32 f32_bits_to_bf16_hi(float f) { return __float_as_uint(f) >> 16; }

__device__ __forceinline__ u32 bf16x2_mul_exact(u32 a, u32 b) {
    // a * b + (-0.0): exact here, so the rounding mode never acts. `fma.rn.bf16x2` is sm_80+.
    u32 d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(0x80008000u));
    return d;
}

__device__ __forceinline__ void mma_bf16_16816(float c[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0,
                                               u32 b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

struct MmaUnit {
    uint2 w0;  // row g: 8 packed bytes (16 columns)
    uint2 w1;  // row g + 8
    u32 s0;    // bf16x2 broadcast of row g's block scale
    u32 s1;    // row g + 8
    u32 x[8];  // activation row g: 16 bf16 (zero for g >= M or columns >= K)
};

__device__ __forceinline__ void mma_load_unit(MmaUnit& un, int u, const unsigned char* w_g,
                                              const unsigned char* w_g8, const u32* s_g,
                                              const u32* s_g8, const bf16_t* x_g, int k,
                                              int cols_padded, bool x_row, int vec_x, int t) {
    const int col = (u << 6) + (t << 4);
    if (col < cols_padded) {
        un.w0 = __ldg((const uint2*)(w_g + (col >> 1)));
        un.w1 = __ldg((const uint2*)(w_g8 + (col >> 1)));
        const u32 sh = 8u * (u32)t;
        const u32 b0 =
            f32_bits_to_bf16_hi(e4m3_to_f32((__ldg(s_g + (size_t)u * 128u) >> sh) & 0xffu));
        const u32 b1 =
            f32_bits_to_bf16_hi(e4m3_to_f32((__ldg(s_g8 + (size_t)u * 128u) >> sh) & 0xffu));
        un.s0 = b0 | (b0 << 16);
        un.s1 = b1 | (b1 << 16);
    } else {
        un.w0 = make_uint2(0u, 0u);
        un.w1 = make_uint2(0u, 0u);
        un.s0 = 0u;
        un.s1 = 0u;
    }
    if (x_row && vec_x && col + 16 <= k) {
        const uint4* p = (const uint4*)(x_g + col);
        uint4 a = __ldg(p);
        uint4 b = __ldg(p + 1);
        un.x[0] = a.x;
        un.x[1] = a.y;
        un.x[2] = a.z;
        un.x[3] = a.w;
        un.x[4] = b.x;
        un.x[5] = b.y;
        un.x[6] = b.z;
        un.x[7] = b.w;
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int c0 = col + 2 * i;
            u32 lo = (x_row && c0 < k) ? (u32)x_g[c0] : 0u;
            u32 hi = (x_row && c0 + 1 < k) ? (u32)x_g[c0 + 1] : 0u;
            un.x[i] = lo | (hi << 16);
        }
    }
}

__device__ __forceinline__ void mma_compute_unit(const MmaUnit& un, const u32* lut, float c[4]) {
    const u32 wg[2] = {un.w0.x, un.w0.y};
    const u32 wg8[2] = {un.w1.x, un.w1.y};
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        // Bytes 2s and 2s + 1 of the 8-byte chunk: word s / 2, byte offset (s % 2) * 2.
        const u32 sh = 16u * (u32)(s & 1);
        const u32 bg = wg[s >> 1] >> sh;
        const u32 bg8 = wg8[s >> 1] >> sh;
        const u32 a0 = bf16x2_mul_exact(lut[bg & 0xffu], un.s0);
        const u32 a2 = bf16x2_mul_exact(lut[(bg >> 8) & 0xffu], un.s0);
        const u32 a1 = bf16x2_mul_exact(lut[bg8 & 0xffu], un.s1);
        const u32 a3 = bf16x2_mul_exact(lut[(bg8 >> 8) & 0xffu], un.s1);
        mma_bf16_16816(c, a0, a1, a2, a3, un.x[2 * s], un.x[2 * s + 1]);
    }
}

extern "C" __global__ void __launch_bounds__(MMA_THREADS) nvfp4_gemv_bf16(
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scales,
    const bf16_t* __restrict__ x,
    bf16_t* __restrict__ y,
    int n_rows,
    int k,
    int cols_padded,
    int num_k_atoms,
    float gs,
    int vec_x,
    int m_rows) {
    __shared__ u32 lut[256];
    __shared__ float red[MMA_WARPS][32][4];
    for (int i = threadIdx.x; i < 256; i += MMA_THREADS) {
        const u32 lo = f32_bits_to_bf16_hi(e2m1_to_f32((u32)i & 15u));
        const u32 hi = f32_bits_to_bf16_hi(e2m1_to_f32((u32)i >> 4));
        lut[i] = lo | (hi << 16);
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int row0 = blockIdx.x * 16;
    const int n_g = row0 + g;
    const int n_g8 = row0 + g + 8;
    const int row_bytes = cols_padded >> 1;
    const unsigned char* w_g = packed + (size_t)n_g * (size_t)row_bytes;
    const unsigned char* w_g8 = packed + (size_t)n_g8 * (size_t)row_bytes;
    const size_t atom_row_bytes = (size_t)num_k_atoms * 512u;
    const u32* s_g = (const u32*)(scales + (size_t)(n_g >> 7) * atom_row_bytes +
                                  ((n_g & 31) << 4) + (((n_g >> 5) & 3) << 2));
    const u32* s_g8 = (const u32*)(scales + (size_t)(n_g8 >> 7) * atom_row_bytes +
                                   ((n_g8 & 31) << 4) + (((n_g8 >> 5) & 3) << 2));
    const bool x_row = g < m_rows;
    const bf16_t* x_g = x + (size_t)(x_row ? g : 0) * (size_t)k;
    const int units = (cols_padded + 63) >> 6;

    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int u = warp;
    if (u < units) {
        MmaUnit cur;
        mma_load_unit(cur, u, w_g, w_g8, s_g, s_g8, x_g, k, cols_padded, x_row, vec_x, t);
        for (; u + MMA_WARPS < units; u += MMA_WARPS) {
            MmaUnit next;
            mma_load_unit(next, u + MMA_WARPS, w_g, w_g8, s_g, s_g8, x_g, k, cols_padded, x_row,
                          vec_x, t);
            mma_compute_unit(cur, lut, c);
            cur = next;
        }
        mma_compute_unit(cur, lut, c);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        red[warp][lane][i] = c[i];
    }
    __syncthreads();
    if (warp == 0) {
        float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int w = 0; w < MMA_WARPS; ++w) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                v[i] += red[w][lane][i];
            }
        }
        // c0, c1: weight row g, activation rows 2t, 2t + 1; c2, c3: weight row g + 8.
        const int m0 = 2 * t;
        if (m0 < m_rows) {
            y[(size_t)m0 * (size_t)n_rows + n_g] = f32_to_bf16(v[0] * gs);
            y[(size_t)m0 * (size_t)n_rows + n_g8] = f32_to_bf16(v[2] * gs);
        }
        if (m0 + 1 < m_rows) {
            y[(size_t)(m0 + 1) * (size_t)n_rows + n_g] = f32_to_bf16(v[1] * gs);
            y[(size_t)(m0 + 1) * (size_t)n_rows + n_g8] = f32_to_bf16(v[3] * gs);
        }
    }
}
