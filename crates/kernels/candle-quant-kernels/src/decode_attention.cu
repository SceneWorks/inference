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
// key chunking (`DECODE_ATTN_CHUNK` keys per chunk, aligned to absolute key positions, so the
// capacity never changes it): every reduction runs in an order fixed by the code —
//   * per-key dot product: lane `l` owns the key elements `d` with `(d / W) % 32 == l` (`W` the
//     16-byte vector width: 8 bf16 or 4 f32), fma'd in ascending `d`; then a fixed xor-shuffle
//     tree, read from lane 0. A 16-byte load and the scalar fallback (an unaligned buffer, or a
//     width that is not a multiple of `W`) run exactly the same fma sequence;
//   * per-chunk max / sum: one warp per query head, lane-strided then a fixed xor-shuffle tree,
//     read from lane 0;
//   * value accumulation: the chunk's keys split into `S` fixed slices (a function of the value
//     width only), each slice's keys fma'd in ascending order, the slices added in ascending order;
//   * the combine walks the chunks in ascending order.
// No atomics, no order that depends on scheduling — so an eager launch and a graph replay at the
// same position are bit-identical by construction.
//
// COST CONTRACT. The partial kernel's grid covers the capacity, but a block whose chunk holds no
// visible key (past the query's position, or before its sliding window) returns before touching
// memory; the combine reads only the chunks that hold visible keys. Work follows the fill. A
// partial block serves a tile of the query heads that share one KV head, so each K/V chunk is
// read once per tile (once in all when the group fits a tile), not once per query head.
//
// LATENCY (sc-24446). At decode shapes the attention is latency-bound, not bandwidth-bound, so a
// chunk is small (more blocks in flight) and every warp issues all its loads of a round before it
// consumes any: a round of the score pass loads `DECODE_ATTN_KEY_UNROLL` whole key rows per warp
// (16-byte vectors), a round of the value pass `DECODE_ATTN_VALUE_UNROLL` keys per thread.

typedef unsigned short bf16_t;

#define DECODE_ATTN_CHUNK 64
#define DECODE_ATTN_THREADS 256
#define DECODE_ATTN_WARPS (DECODE_ATTN_THREADS / 32)
// Most query heads one partial block serves (its per-head accumulators live in registers).
#define DECODE_ATTN_MAX_GROUP_TILE 8
// Keys a warp scores per round of the score pass (all their loads in flight together).
#define DECODE_ATTN_KEY_UNROLL 8
// Keys a thread accumulates per round of the value pass.
#define DECODE_ATTN_VALUE_UNROLL 16
// Widest key / value head (matches `DECODE_ATTN_MAX_HEAD_DIM` on the host).
#define DECODE_ATTN_MAX_HEAD_DIM 1024
// Value elements per combine thread.
#define DECODE_ATTN_COMBINE_DIMS (DECODE_ATTN_MAX_HEAD_DIM / DECODE_ATTN_THREADS)
// Chunks the combine stages per pass, and its loads in flight per thread.
#define DECODE_ATTN_COMBINE_TILE 256
#define DECODE_ATTN_COMBINE_UNROLL 8

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

// Elements per 16-byte vector.
template <typename T> struct VecWidth;
template <> struct VecWidth<float> { static const int W = 4; };
template <> struct VecWidth<bf16_t> { static const int W = 8; };

// Elements `d0 .. d0 + 4` of a row (zeros at and past `n`). `vec`: the caller proved a whole
// 16-byte load is aligned and in bounds whenever `d0 < n` (`n % 4 == 0`, 16-byte aligned rows).
__device__ __forceinline__ void load_key_vec(const float* row, int d0, int n, bool vec, float* o) {
    if (d0 >= n) {
#pragma unroll
        for (int e = 0; e < 4; ++e) o[e] = 0.0f;
    } else if (vec) {
        const float4 x = *(const float4*)(row + d0);
        o[0] = x.x;
        o[1] = x.y;
        o[2] = x.z;
        o[3] = x.w;
    } else {
#pragma unroll
        for (int e = 0; e < 4; ++e) o[e] = d0 + e < n ? row[d0 + e] : 0.0f;
    }
}

// As above for bf16: elements `d0 .. d0 + 8` (one 16-byte load when `vec`).
__device__ __forceinline__ void load_key_vec(const bf16_t* row, int d0, int n, bool vec,
                                             float* o) {
    if (d0 >= n) {
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] = 0.0f;
    } else if (vec) {
        const uint4 x = *(const uint4*)(row + d0);
        o[0] = __uint_as_float(x.x << 16);
        o[1] = __uint_as_float(x.x & 0xffff0000u);
        o[2] = __uint_as_float(x.y << 16);
        o[3] = __uint_as_float(x.y & 0xffff0000u);
        o[4] = __uint_as_float(x.z << 16);
        o[5] = __uint_as_float(x.z & 0xffff0000u);
        o[6] = __uint_as_float(x.w << 16);
        o[7] = __uint_as_float(x.w & 0xffff0000u);
    } else {
#pragma unroll
        for (int e = 0; e < 8; ++e) o[e] = d0 + e < n ? bf16_to_f32(row[d0 + e]) : 0.0f;
    }
}

// Elements `d, d + 1` of a row (`d < n`; zero past `n`). `vec`: one aligned pair load.
__device__ __forceinline__ void load_pair(const float* row, int d, int n, bool vec, float* o) {
    if (vec) {
        const float2 x = *(const float2*)(row + d);
        o[0] = x.x;
        o[1] = x.y;
    } else {
        o[0] = row[d];
        o[1] = d + 1 < n ? row[d + 1] : 0.0f;
    }
}

__device__ __forceinline__ void load_pair(const bf16_t* row, int d, int n, bool vec, float* o) {
    if (vec) {
        const unsigned int x = *(const unsigned int*)(row + d);
        o[0] = __uint_as_float(x << 16);
        o[1] = __uint_as_float(x & 0xffff0000u);
    } else {
        o[0] = bf16_to_f32(row[d]);
        o[1] = d + 1 < n ? bf16_to_f32(row[d + 1]) : 0.0f;
    }
}

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

// The value pass's key slices for value width `dv`: a thread owns one pair of value elements, so
// `ceil(dv / 2)` threads cover a row and the block's remaining threads take further slices of the
// chunk's keys. Must equal `value_slices` on the host (it sizes the shared memory).
__device__ __forceinline__ int value_slices(int dv) {
    const int pairs = (dv + 1) / 2;
    return pairs >= DECODE_ATTN_THREADS ? 1 : DECODE_ATTN_THREADS / pairs;
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
// Every per-head reduction runs in exactly the order a one-head block would run it (see the
// determinism contract above), so the result is independent of the tile: bit-identical to a
// launch with one query head per block.
//
// q:   [B, H, M, dk]        k: [B, Hkv, cap, dk]      v: [B, Hkv, cap, dv]   (all contiguous)
// Dynamic shared memory: tile·(max(dk, S·dv) + DECODE_ATTN_CHUNK) floats — the query rows (the
// value pass reuses that region for its per-slice sums) and the chunk's scores.
// ---------------------------------------------------------------------------------------------
template <typename T>
__device__ void decode_attn_partial(const T* q, const T* k, const T* v,
                                    const unsigned int* start, float* ws, int heads,
                                    int kv_heads, int queries, int cap, int dk, int dv,
                                    float scale, float softcap, int window, int n_chunks,
                                    int tile) {
    extern __shared__ float smem[];
    const int slices = value_slices(dv);
    const int region = tile * (dk > slices * dv ? dk : slices * dv);
    float* q_s = smem;              // [tile][dk] (score pass)
    float* vred = smem;             // [slices][tile][dv] (value pass; the query rows are done)
    float* s_s = smem + region;     // [tile][DECODE_ATTN_CHUNK]

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
    const int warp = tid / 32;
    const int lane = tid % 32;

    int lo, hi;
    visible_range(start[0], i, cap, window, &lo, &hi);
    const int k0 = c * DECODE_ATTN_CHUNK;
    int k1 = k0 + DECODE_ATTN_CHUNK;
    if (k1 > hi) k1 = hi;
    const int ks = k0 > lo ? k0 : lo;
    if (ks >= k1) return;  // no visible key in this chunk: never read by the combine
    const int a0 = ks - k0;  // the visible chunk offsets [a0, a1)
    const int a1 = k1 - k0;

    for (int e = tid; e < gn * dk; e += DECODE_ATTN_THREADS) {
        const int g = e / dk;
        const int d = e - g * dk;
        q_s[g * dk + d] = load_val(q + (((size_t)b * heads + h0 + g) * queries + i) * (size_t)dk, d);
    }
    __syncthreads();

    const T* kbase = k + ((size_t)b * kv_heads + kvh) * (size_t)cap * dk;
    const T* vbase = v + ((size_t)b * kv_heads + kvh) * (size_t)cap * dv;

    // Score pass: warp `w` scores the chunk offsets w, w + WARPS, ... — a round loads
    // DECODE_ATTN_KEY_UNROLL whole key rows (lane-owned 16-byte vectors) before any fma.
    const int W = VecWidth<T>::W;
    const int seg = 32 * W;  // key elements one warp-wide vector load covers
    const bool kvec = (dk % W) == 0 && (((size_t)k) & 15) == 0;
    for (int t0 = a0 + warp; t0 < a1; t0 += DECODE_ATTN_WARPS * DECODE_ATTN_KEY_UNROLL) {
        float acc[DECODE_ATTN_KEY_UNROLL][DECODE_ATTN_MAX_GROUP_TILE];
#pragma unroll
        for (int u = 0; u < DECODE_ATTN_KEY_UNROLL; ++u) {
#pragma unroll
            for (int g = 0; g < DECODE_ATTN_MAX_GROUP_TILE; ++g) acc[u][g] = 0.0f;
        }
        for (int d0 = lane * W; d0 < dk; d0 += seg) {
            float kv[DECODE_ATTN_KEY_UNROLL][VecWidth<T>::W];
#pragma unroll
            for (int u = 0; u < DECODE_ATTN_KEY_UNROLL; ++u) {
                const int t = t0 + u * DECODE_ATTN_WARPS;
                // A key past the chunk's visible range loads as zeros and is never stored.
                const int row = t < a1 ? k0 + t : k0 + a0;
                load_key_vec(kbase + (size_t)row * dk, t < a1 ? d0 : dk, dk, kvec, kv[u]);
            }
#pragma unroll
            for (int g = 0; g < DECODE_ATTN_MAX_GROUP_TILE; ++g) {
                if (g < gn) {
                    float qv[VecWidth<T>::W];
#pragma unroll
                    for (int e = 0; e < W; ++e) qv[e] = d0 + e < dk ? q_s[g * dk + d0 + e] : 0.0f;
#pragma unroll
                    for (int u = 0; u < DECODE_ATTN_KEY_UNROLL; ++u) {
#pragma unroll
                        for (int e = 0; e < W; ++e) {
                            if (d0 + e < dk) acc[u][g] = fmaf(qv[e], kv[u][e], acc[u][g]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int u = 0; u < DECODE_ATTN_KEY_UNROLL; ++u) {
            const int t = t0 + u * DECODE_ATTN_WARPS;
            if (t < a1) {  // uniform across the warp, so every lane shuffles
#pragma unroll
                for (int g = 0; g < DECODE_ATTN_MAX_GROUP_TILE; ++g) {
                    if (g < gn) {  // `gn` is uniform across the block
                        float a = acc[u][g];
#pragma unroll
                        for (int off = 16; off > 0; off >>= 1) {
                            a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, off));
                        }
                        if (lane == 0) {
                            float s = a * scale;
                            if (softcap > 0.0f) s = softcap * tanhf(s / softcap);
                            s_s[g * DECODE_ATTN_CHUNK + t] = s;
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    // Softmax pass: warp `g` takes query head `g` — the chunk max, p_j = exp(s_j - max) written
    // back in place, and the chunk sum (lane-strided, then the xor-shuffle tree from lane 0).
    for (int g = warp; g < gn; g += DECODE_ATTN_WARPS) {
        float* sg = s_s + g * DECODE_ATTN_CHUNK;
        float m = neg_inf();
        for (int t = a0 + lane; t < a1; t += 32) m = fmaxf(m, sg[t]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        m = __shfl_sync(0xffffffffu, m, 0);
        float sum = 0.0f;
        for (int t = a0 + lane; t < a1; t += 32) {
            const float p = expf(sg[t] - m);
            sg[t] = p;
            sum = __fadd_rn(sum, p);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, off));
        }
        if (lane == 0) {
            float* w = ws + (((size_t)bm * heads + h0 + g) * n_chunks + c) * (size_t)(2 + dv);
            w[0] = m;
            w[1] = sum;
        }
    }
    __syncthreads();

    // Value pass: thread (slice s, pair p) accumulates value elements 2p, 2p + 1 over the keys of
    // its slice — chunk offsets [s·C/S, (s+1)·C/S) ∩ [a0, a1) — in ascending order, a round of
    // DECODE_ATTN_VALUE_UNROLL loads in flight; one read of a value row serves every head.
    const int pairs = (dv + 1) / 2;
    const int sl = tid / pairs;
    const bool vvec = (dv % 2) == 0 && (((size_t)v) & (2 * sizeof(T) - 1)) == 0;
    if (sl < slices) {
        int j0 = sl * DECODE_ATTN_CHUNK / slices;
        int j1 = (sl + 1) * DECODE_ATTN_CHUNK / slices;
        if (j0 < a0) j0 = a0;
        if (j1 > a1) j1 = a1;
        for (int p = tid - sl * pairs; p < pairs; p += DECODE_ATTN_THREADS) {
            const int d = 2 * p;
            float acc[DECODE_ATTN_MAX_GROUP_TILE][2];
#pragma unroll
            for (int g = 0; g < DECODE_ATTN_MAX_GROUP_TILE; ++g) {
                acc[g][0] = 0.0f;
                acc[g][1] = 0.0f;
            }
            for (int t0 = j0; t0 < j1; t0 += DECODE_ATTN_VALUE_UNROLL) {
                float vv[DECODE_ATTN_VALUE_UNROLL][2];
#pragma unroll
                for (int u = 0; u < DECODE_ATTN_VALUE_UNROLL; ++u) {
                    const int t = t0 + u < j1 ? t0 + u : j0;  // past the slice: a harmless reload
                    load_pair(vbase + (size_t)(k0 + t) * dv, d, dv, vvec, vv[u]);
                }
#pragma unroll
                for (int u = 0; u < DECODE_ATTN_VALUE_UNROLL; ++u) {
                    if (t0 + u < j1) {
#pragma unroll
                        for (int g = 0; g < DECODE_ATTN_MAX_GROUP_TILE; ++g) {
                            if (g < gn) {
                                const float pw = s_s[g * DECODE_ATTN_CHUNK + t0 + u];
                                acc[g][0] = fmaf(pw, vv[u][0], acc[g][0]);
                                acc[g][1] = fmaf(pw, vv[u][1], acc[g][1]);
                            }
                        }
                    }
                }
            }
#pragma unroll
            for (int g = 0; g < DECODE_ATTN_MAX_GROUP_TILE; ++g) {
                if (g < gn) {
                    float* r = vred + ((size_t)sl * tile + g) * dv;
                    r[d] = acc[g][0];
                    if (d + 1 < dv) r[d + 1] = acc[g][1];
                }
            }
        }
    }
    __syncthreads();

    // The slices' sums, added in ascending slice order.
    for (int e = tid; e < gn * dv; e += DECODE_ATTN_THREADS) {
        const int g = e / dv;
        const int d = e - g * dv;
        float a = vred[(size_t)g * dv + d];
        for (int r = 1; r < slices; ++r) a = __fadd_rn(a, vred[((size_t)r * tile + g) * dv + d]);
        ws[(((size_t)bm * heads + h0 + g) * n_chunks + c) * (size_t)(2 + dv) + 2 + d] = a;
    }
}

// ---------------------------------------------------------------------------------------------
// Phase 2: one block per (query head, batch·query). Combines the visible chunks in ascending
// order: M = max_c m_c, l = Σ l_c·e^(m_c−M), out = Σ acc_c·e^(m_c−M) / l. The per-chunk factors
// e^(m_c−M) are staged in shared memory once per chunk (a tile of chunks at a time) rather than
// recomputed per element; the sums keep the ascending fma order.
// out: [B, H, M, dv]
// ---------------------------------------------------------------------------------------------
template <typename T>
__device__ void decode_attn_combine(const float* ws, const unsigned int* start, T* out,
                                    int heads, int queries, int cap, int dv, int window,
                                    int n_chunks) {
    __shared__ float e_s[DECODE_ATTN_COMBINE_TILE];
    __shared__ float l_s[DECODE_ATTN_COMBINE_TILE];
    __shared__ float red[DECODE_ATTN_THREADS];
    const int h = blockIdx.x;
    const int bm = blockIdx.y;
    const int i = bm % queries;
    const int tid = threadIdx.x;

    int lo, hi;
    visible_range(start[0], i, cap, window, &lo, &hi);
    const int c_lo = lo / DECODE_ATTN_CHUNK;
    const int c_hi = (hi - 1) / DECODE_ATTN_CHUNK;
    const size_t stride = (size_t)(2 + dv);
    const float* wbase = ws + ((size_t)bm * heads + h) * n_chunks * stride;

    float mx = neg_inf();
    for (int c = c_lo + tid; c <= c_hi; c += DECODE_ATTN_THREADS) mx = fmaxf(mx, wbase[c * stride]);
    mx = block_tree(red, mx, true);  // a max is exact: any tree order gives the same value

    float l = 0.0f;
    float a[DECODE_ATTN_COMBINE_DIMS];
#pragma unroll
    for (int r = 0; r < DECODE_ATTN_COMBINE_DIMS; ++r) a[r] = 0.0f;
    for (int t0 = c_lo; t0 <= c_hi; t0 += DECODE_ATTN_COMBINE_TILE) {
        const int n = c_hi - t0 + 1 < DECODE_ATTN_COMBINE_TILE ? c_hi - t0 + 1
                                                               : DECODE_ATTN_COMBINE_TILE;
        for (int u = tid; u < n; u += DECODE_ATTN_THREADS) {
            const float* w = wbase + (size_t)(t0 + u) * stride;
            e_s[u] = expf(w[0] - mx);
            l_s[u] = w[1];
        }
        __syncthreads();
        for (int u = 0; u < n; ++u) l = fmaf(l_s[u], e_s[u], l);
#pragma unroll
        for (int r = 0; r < DECODE_ATTN_COMBINE_DIMS; ++r) {
            const int d = tid + r * DECODE_ATTN_THREADS;
            if (d < dv) {
                const float* col = wbase + (size_t)t0 * stride + 2 + d;
                for (int u0 = 0; u0 < n; u0 += DECODE_ATTN_COMBINE_UNROLL) {
                    float x[DECODE_ATTN_COMBINE_UNROLL];
#pragma unroll
                    for (int u = 0; u < DECODE_ATTN_COMBINE_UNROLL; ++u) {
                        x[u] = u0 + u < n ? col[(size_t)(u0 + u) * stride] : 0.0f;
                    }
#pragma unroll
                    for (int u = 0; u < DECODE_ATTN_COMBINE_UNROLL; ++u) {
                        if (u0 + u < n) a[r] = fmaf(x[u], e_s[u0 + u], a[r]);
                    }
                }
            }
        }
        __syncthreads();
    }
    // `bm = b·M + i` and out is [B, H, M, dv]: re-index to (b, h, i).
    const int b = bm / queries;
    T* orow = out + (((size_t)b * heads + h) * queries + i) * (size_t)dv;
#pragma unroll
    for (int r = 0; r < DECODE_ATTN_COMBINE_DIMS; ++r) {
        const int d = tid + r * DECODE_ATTN_THREADS;
        if (d < dv) store_val(orow, d, a[r] / l);
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
