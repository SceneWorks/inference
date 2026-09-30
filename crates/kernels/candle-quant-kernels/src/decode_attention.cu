// Length-aware decode attention and device-indexed copies (sc-24441, epic sc-24432).
//
// Compiled at runtime through the nvrtc compile-once seam (`nvrtc.rs`): builtins only, no headers.
// bf16 is raw `unsigned short` with hand-written conversions (as in `fused_decode.cu`).
//
// Every kernel here reads its sequence position from DEVICE memory (`start[0]`, a u32 the caller
// staged before the step), never from a kernel argument, so a CUDA graph that recorded one step
// replays correctly at any later position. The launch geometry depends only on the step shape and
// the cache CAPACITY, never on how full the cache is.
//
// DETERMINISM CONTRACT (decode attention). The result depends only on the operands and the fixed
// key chunking (`DECODE_ATTN_CHUNK` keys per chunk): each chunk's partial softmax is reduced in a
// fixed order (per-key dot product: lane-strided fma then a fixed xor-shuffle tree read from lane
// 0; max/sum: a fixed shared-memory tree; value accumulation: keys in ascending order), and the
// combine walks the chunks in ascending order. No atomics, no order that depends on scheduling —
// so an eager launch and a graph replay at the same position are bit-identical by construction.
//
// COST CONTRACT. The partial kernel's grid covers the capacity, but a block whose chunk holds no
// visible key (past the query's position, or before its sliding window) returns before touching
// memory; the combine reads only the chunks that hold visible keys. Work follows the fill. A
// partial block serves a tile of the query heads that share one KV head, so each K/V chunk is
// read once per tile (once in all when the group fits a tile), not once per query head.

typedef unsigned short bf16_t;

#define DECODE_ATTN_CHUNK 256
#define DECODE_ATTN_THREADS 128
// Most query heads one partial block serves (its per-head accumulators live in registers).
#define DECODE_ATTN_MAX_GROUP_TILE 16

__device__ __forceinline__ float bf16_to_f32(bf16_t h) {
    return __uint_as_float(((unsigned int)h) << 16);
}

// Round-to-nearest-even, no flush-to-zero (NaN stays a quiet NaN).
__device__ __forceinline__ bf16_t f32_to_bf16(float f) {
    unsigned int u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) {
        return (bf16_t)((u >> 16) | 0x0040u);
    }
    unsigned int lsb = (u >> 16) & 1u;
    return (bf16_t)((u + 0x7fffu + lsb) >> 16);
}

__device__ __forceinline__ float load_val(const float* p, size_t i) { return p[i]; }
__device__ __forceinline__ float load_val(const bf16_t* p, size_t i) { return bf16_to_f32(p[i]); }
__device__ __forceinline__ void store_val(float* p, size_t i, float v) { p[i] = v; }
__device__ __forceinline__ void store_val(bf16_t* p, size_t i, float v) { p[i] = f32_to_bf16(v); }

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

// The visible key range [lo, hi) of query `i` of a step starting at `start`: causal (keys up to
// and including the query's own position), clipped to the capacity, and — with `window > 0` —
// no further back than `window - 1` positions.
__device__ __forceinline__ void visible_range(unsigned int start, int i, int cap, int window,
                                              int* lo, int* hi) {
    int pos = (int)start + i;
    int h = pos + 1;
    if (h > cap) h = cap;
    int l = 0;
    if (window > 0) {
        l = pos - window + 1;
        if (l < 0) l = 0;
    }
    *lo = l;
    *hi = h;
}

// Fixed-order shared-memory tree over DECODE_ATTN_THREADS partial values in `red`; returns the
// combined value to every thread. `is_max` selects max vs sum.
__device__ __forceinline__ float block_tree(float* red, float v, bool is_max) {
    int tid = threadIdx.x;
    red[tid] = v;
    for (int s = DECODE_ATTN_THREADS / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            float a = red[tid];
            float b = red[tid + s];
            red[tid] = is_max ? fmaxf(a, b) : __fadd_rn(a, b);
        }
    }
    __syncthreads();
    float out = red[0];
    __syncthreads();
    return out;
}

// ---------------------------------------------------------------------------------------------
// Phase 1: one block per (chunk, KV head x group tile, batch·query). A block serves `tile` query
// heads that share one KV head (grouped-query attention: `groups = heads / kv_heads` query heads
// per KV head, split into ceil(groups / tile) tiles), so each K/V row of the chunk is read from
// memory once per tile rather than once per query head. Writes each head's partial softmax state
// to `ws[(bm, h, c)] = [max, sum, acc[0..dv)]` (f32). Chunks with no visible key return
// immediately and are never read by the combine.
//
// Every per-head reduction runs in exactly the order a one-head block would run it — the dot
// product's lane-strided fma chain and xor-shuffle tree, the max / sum trees, the ascending-key
// value fma chain — so the result is independent of the tile: bit-identical to a launch with one
// query head per block.
//
// q:   [B, H, M, dk]        k: [B, Hkv, cap, dk]      v: [B, Hkv, cap, dv]   (all contiguous)
// Dynamic shared memory: (tile·dk + tile·DECODE_ATTN_CHUNK + DECODE_ATTN_THREADS) floats.
// ---------------------------------------------------------------------------------------------
template <typename T>
__device__ void decode_attn_partial(const T* q, const T* k, const T* v,
                                    const unsigned int* start, float* ws, int heads,
                                    int kv_heads, int queries, int cap, int dk, int dv,
                                    float scale, float softcap, int window, int n_chunks,
                                    int tile) {
    extern __shared__ float smem[];
    float* q_s = smem;                            // [tile][dk]
    float* s_s = smem + tile * dk;                // [tile][DECODE_ATTN_CHUNK]
    float* red = s_s + tile * DECODE_ATTN_CHUNK;  // [DECODE_ATTN_THREADS]

    const int groups = heads / kv_heads;
    const int n_tiles = (groups + tile - 1) / tile;
    const int c = blockIdx.x;
    const int kvh = blockIdx.y / n_tiles;
    const int g0 = (blockIdx.y % n_tiles) * tile;
    const int gn = groups - g0 < tile ? groups - g0 : tile;  // query heads this block serves
    const int h0 = kvh * groups + g0;       // the first of them
    const int bm = blockIdx.z;
    const int b = bm / queries;
    const int i = bm % queries;
    const int tid = threadIdx.x;

    int lo, hi;
    visible_range(start[0], i, cap, window, &lo, &hi);
    const int k0 = c * DECODE_ATTN_CHUNK;
    int k1 = k0 + DECODE_ATTN_CHUNK;
    if (k1 > hi) k1 = hi;
    const int ks = k0 > lo ? k0 : lo;
    if (ks >= k1) return;  // no visible key in this chunk: never read by the combine

    for (int g = 0; g < gn; ++g) {
        const T* qrow = q + (((size_t)b * heads + h0 + g) * queries + i) * (size_t)dk;
        for (int d = tid; d < dk; d += DECODE_ATTN_THREADS) q_s[g * dk + d] = load_val(qrow, d);
    }
    __syncthreads();

    const T* kbase = k + ((size_t)b * kv_heads + kvh) * (size_t)cap * dk;
    const T* vbase = v + ((size_t)b * kv_heads + kvh) * (size_t)cap * dv;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int n_warps = DECODE_ATTN_THREADS / 32;
    for (int j = ks + warp; j < k1; j += n_warps) {
        const T* kr = kbase + (size_t)j * dk;
        float acc[DECODE_ATTN_MAX_GROUP_TILE];
#pragma unroll
        for (int g = 0; g < DECODE_ATTN_MAX_GROUP_TILE; ++g) acc[g] = 0.0f;
        for (int d = lane; d < dk; d += 32) {
            const float kd = load_val(kr, d);  // one read of the key serves every head
#pragma unroll
            for (int g = 0; g < DECODE_ATTN_MAX_GROUP_TILE; ++g) {
                if (g < gn) acc[g] = fmaf(q_s[g * dk + d], kd, acc[g]);
            }
        }
#pragma unroll
        for (int g = 0; g < DECODE_ATTN_MAX_GROUP_TILE; ++g) {
            if (g < gn) {  // `gn` is uniform across the block, so every lane shuffles
                float a = acc[g];
                for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xffffffffu, a, off);
                if (lane == 0) {
                    float s = a * scale;
                    if (softcap > 0.0f) s = softcap * tanhf(s / softcap);
                    s_s[g * DECODE_ATTN_CHUNK + (j - k0)] = s;
                }
            }
        }
    }
    __syncthreads();

    const int a0 = ks - k0;
    const int a1 = k1 - k0;
    for (int g = 0; g < gn; ++g) {
        float* sg = s_s + g * DECODE_ATTN_CHUNK;
        // Max over the chunk's visible scores (strided per thread, then the fixed tree).
        float m = neg_inf();
        for (int t = a0 + tid; t < a1; t += DECODE_ATTN_THREADS) m = fmaxf(m, sg[t]);
        m = block_tree(red, m, true);

        // p_j = exp(s_j - m), written back in place; the chunk sum through the fixed tree.
        float sum = 0.0f;
        for (int t = a0 + tid; t < a1; t += DECODE_ATTN_THREADS) {
            float p = expf(sg[t] - m);
            sg[t] = p;
            sum = __fadd_rn(sum, p);
        }
        sum = block_tree(red, sum, false);
        if (tid == 0) {
            float* w = ws + (((size_t)bm * heads + h0 + g) * n_chunks + c) * (size_t)(2 + dv);
            w[0] = m;
            w[1] = sum;
        }
    }
    __syncthreads();

    for (int d = tid; d < dv; d += DECODE_ATTN_THREADS) {
        float a[DECODE_ATTN_MAX_GROUP_TILE];
#pragma unroll
        for (int g = 0; g < DECODE_ATTN_MAX_GROUP_TILE; ++g) a[g] = 0.0f;
        for (int j = ks; j < k1; ++j) {
            const float vd = load_val(vbase + (size_t)j * dv, d);  // one read serves every head
#pragma unroll
            for (int g = 0; g < DECODE_ATTN_MAX_GROUP_TILE; ++g) {
                if (g < gn) a[g] = fmaf(s_s[g * DECODE_ATTN_CHUNK + (j - k0)], vd, a[g]);
            }
        }
#pragma unroll
        for (int g = 0; g < DECODE_ATTN_MAX_GROUP_TILE; ++g) {
            if (g < gn) {
                ws[(((size_t)bm * heads + h0 + g) * n_chunks + c) * (size_t)(2 + dv) + 2 + d] =
                    a[g];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Phase 2: one block per (query head, batch·query). Combines the visible chunks in ascending
// order: M = max_c m_c, l = Σ l_c·e^(m_c−M), out = Σ acc_c·e^(m_c−M) / l.
// out: [B, H, M, dv]
// ---------------------------------------------------------------------------------------------
template <typename T>
__device__ void decode_attn_combine(const float* ws, const unsigned int* start, T* out,
                                    int heads, int queries, int cap, int dv, int window,
                                    int n_chunks) {
    const int h = blockIdx.x;
    const int bm = blockIdx.y;
    const int i = bm % queries;
    const int tid = threadIdx.x;

    int lo, hi;
    visible_range(start[0], i, cap, window, &lo, &hi);
    const int c_lo = lo / DECODE_ATTN_CHUNK;
    const int c_hi = (hi - 1) / DECODE_ATTN_CHUNK;
    const float* wbase = ws + ((size_t)bm * heads + h) * n_chunks * (size_t)(2 + dv);

    float mx = neg_inf();
    for (int c = c_lo; c <= c_hi; ++c) mx = fmaxf(mx, wbase[(size_t)c * (2 + dv)]);
    float l = 0.0f;
    for (int c = c_lo; c <= c_hi; ++c) {
        const float* w = wbase + (size_t)c * (2 + dv);
        l = fmaf(w[1], expf(w[0] - mx), l);
    }
    // `bm = b·M + i` and out is [B, H, M, dv]: re-index to (b, h, i).
    const int b = bm / queries;
    T* orow = out + (((size_t)b * heads + h) * queries + i) * (size_t)dv;
    for (int d = tid; d < dv; d += DECODE_ATTN_THREADS) {
        float a = 0.0f;
        for (int c = c_lo; c <= c_hi; ++c) {
            const float* w = wbase + (size_t)c * (2 + dv);
            a = fmaf(w[2 + d], expf(w[0] - mx), a);
        }
        store_val(orow, d, a / l);
    }
}

#define DECODE_ATTN_ENTRY(SUFFIX, T)                                                              \
    extern "C" __global__ void decode_attn_partial_##SUFFIX(                                     \
        const T* q, const T* k, const T* v, const unsigned int* start, float* ws, int heads,     \
        int kv_heads, int queries, int cap, int dk, int dv, float scale, float softcap,          \
        int window, int n_chunks, int tile) {                                                    \
        decode_attn_partial<T>(q, k, v, start, ws, heads, kv_heads, queries, cap, dk, dv, scale, \
                               softcap, window, n_chunks, tile);                                  \
    }                                                                                            \
    extern "C" __global__ void decode_attn_combine_##SUFFIX(                                     \
        const float* ws, const unsigned int* start, T* out, int heads, int queries, int cap,     \
        int dv, int window, int n_chunks) {                                                      \
        decode_attn_combine<T>(ws, start, out, heads, queries, cap, dv, window, n_chunks);       \
    }

DECODE_ATTN_ENTRY(f32, float)
DECODE_ATTN_ENTRY(bf16, bf16_t)

// ---------------------------------------------------------------------------------------------
// Device-indexed copies. `index[0]` is read on the device, so a recorded launch writes / reads
// wherever the index points at replay time.
//
// write_rows_at: dst [outer, cap, inner] <- src [outer, rows, inner] at rows index[0]..+rows.
//   Rows past `cap` are dropped (the host refuses a step past the capacity before it launches).
// read_slot:     out [n] <- src [slots, n] row index[0] (an out-of-range index yields zeros).
// Element width only matters, so each comes in a 2-byte and a 4-byte flavour.
// ---------------------------------------------------------------------------------------------
template <typename E>
__device__ void write_rows_at(E* dst, const E* src, const unsigned int* index, size_t outer,
                              size_t rows, size_t inner, size_t cap) {
    const size_t n = outer * rows * inner;
    const size_t at = index[0];
    for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n;
         e += (size_t)gridDim.x * blockDim.x) {
        const size_t o = e / (rows * inner);
        const size_t r = (e / inner) % rows;
        const size_t x = e % inner;
        const size_t row = at + r;
        if (row < cap) dst[(o * cap + row) * inner + x] = src[e];
    }
}

template <typename E>
__device__ void read_slot(const E* src, const unsigned int* index, E* out, size_t n,
                          size_t slots) {
    const size_t at = index[0];
    for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n;
         e += (size_t)gridDim.x * blockDim.x) {
        out[e] = at < slots ? src[at * n + e] : (E)0;
    }
}

extern "C" __global__ void write_rows_at_u16(unsigned short* dst, const unsigned short* src,
                                             const unsigned int* index, size_t outer,
                                             size_t rows, size_t inner, size_t cap) {
    write_rows_at<unsigned short>(dst, src, index, outer, rows, inner, cap);
}

extern "C" __global__ void write_rows_at_u32(unsigned int* dst, const unsigned int* src,
                                             const unsigned int* index, size_t outer,
                                             size_t rows, size_t inner, size_t cap) {
    write_rows_at<unsigned int>(dst, src, index, outer, rows, inner, cap);
}

extern "C" __global__ void read_slot_u16(const unsigned short* src, const unsigned int* index,
                                         unsigned short* out, size_t n, size_t slots) {
    read_slot<unsigned short>(src, index, out, n, slots);
}

extern "C" __global__ void read_slot_u32(const unsigned int* src, const unsigned int* index,
                                         unsigned int* out, size_t n, size_t slots) {
    read_slot<unsigned int>(src, index, out, n, slots);
}
