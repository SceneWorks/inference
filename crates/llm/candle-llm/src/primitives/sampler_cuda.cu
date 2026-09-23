// On-device token sampler (epic sc-24128, story sc-24133).
//
// One block of SAMPLER_THREADS threads samples one logits row: temperature, top-k and top-p are
// applied as *thresholds* found by radix selection over the weight bits (no vocabulary sort), and
// the categorical draw is an inverse-CDF walk in index order over fixed-point weights. Fixed-point
// (w * 2^40 as u64) makes every sum associative, so a seeded draw is bit-reproducible regardless of
// the order the atomics land in. The distribution is the host reference's
// (`primitives::sampler::sample_host`): w_i = exp((x_i - max) / T); top-k keeps the k largest
// weights, ties to the lower index; top-p keeps the shortest descending prefix whose mass reaches
// top_p * total, ties to the lower index, at least one token; any NaN logit or a non-finite max
// falls back to the argmax (first maximum), as the host does.
//
// NVRTC is invoked with no SDK include paths, so this source is self-contained.

typedef unsigned int u32;
typedef unsigned long long u64;
static_assert(sizeof(u32) == 4, "u32 ABI");
static_assert(sizeof(u64) == 8, "u64 ABI");

#define SAMPLER_THREADS 1024
#define SAMPLER_WARPS 32
#define FULL_MASK 0xffffffffu
#define NO_INDEX 0xffffffffu
#define SPLITMIX_INCREMENT 0x9E3779B97F4A7C15ull
// 2^40: a weight in [0, 1] becomes an integer in [0, 2^40]; a row of up to 2^23 tokens sums
// below 2^63. The Rust side refuses larger vocabularies.
#define FIX_ONE 1099511627776.0

// SplitMix64's output function applied to an already-advanced state. Draw i (0-based) of a stream
// whose state was `s` is `splitmix_uniform(s + (i + 1) * SPLITMIX_INCREMENT)` - bit-identical to
// the host `SplitMix64::next_f32` sequence.
__device__ __forceinline__ float splitmix_uniform(u64 z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);
    return (float)(z >> 40) * (1.0f / 16777216.0f);
}

extern "C" __global__ void candle_llm_splitmix_uniform_f32(u64 state, u32 n, float* out) {
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = splitmix_uniform(state + (u64)(i + 1) * SPLITMIX_INCREMENT);
    }
}

__device__ __forceinline__ bool is_nan(float v) { return v != v; }

__device__ __forceinline__ bool is_finite(float v) {
    return (__float_as_uint(v) & 0x7f800000u) != 0x7f800000u;
}

__device__ __forceinline__ float weight_of(float x, float mx, float inv_t) {
    return expf((x - mx) * inv_t);
}

// Non-negative floats order like their bit patterns.
__device__ __forceinline__ u32 key_of(float w) { return __float_as_uint(w); }

__device__ __forceinline__ u64 fixed_of(float w) {
    return __double2ull_rz((double)w * FIX_ONE);
}

template <typename T>
__device__ __forceinline__ T warp_inclusive_scan(T v) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        T n = __shfl_up_sync(FULL_MASK, v, offset);
        if (lane >= offset) v += n;
    }
    return v;
}

// Block-wide inclusive scan; `*total` receives the block sum. Ends on a barrier so `scratch` (32
// entries) may be reused immediately.
template <typename T>
__device__ T block_inclusive_scan(T v, T* scratch, T* total) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_inclusive_scan(v);
    if (lane == 31) scratch[warp] = v;
    __syncthreads();
    if (warp == 0) {
        T s = warp_inclusive_scan(scratch[lane]);
        scratch[lane] = s;
    }
    __syncthreads();
    if (warp > 0) v += scratch[warp - 1];
    *total = scratch[SAMPLER_WARPS - 1];
    __syncthreads();
    return v;
}

template <typename T>
__device__ T block_sum(T v, T* scratch) {
    T total;
    block_inclusive_scan(v, scratch, &total);
    return total;
}

// Block-wide (max value, lowest index) reduction, broadcast to every thread.
__device__ void block_argmax(float* value, u32* index, float* sv, u32* si) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float v = *value;
    u32 i = *index;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(FULL_MASK, v, offset);
        u32 oi = __shfl_down_sync(FULL_MASK, i, offset);
        if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    if (lane == 0) { sv[warp] = v; si[warp] = i; }
    __syncthreads();
    if (warp == 0) {
        v = sv[lane];
        i = si[lane];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(FULL_MASK, v, offset);
            u32 oi = __shfl_down_sync(FULL_MASK, i, offset);
            if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
        }
        if (lane == 0) { sv[0] = v; si[0] = i; }
    }
    __syncthreads();
    *value = sv[0];
    *index = si[0];
    __syncthreads();
}

#define NO_BUCKET 256u

// Add each lane's `slot` (NO_BUCKET = nothing) to a shared 256-bin histogram, with its `mass`
// when `hist_mass` is non-null. Must be called by all 32 lanes. A logits row's weights pile into a
// handful of exponent buckets, so per-lane shared atomics serialize on a few addresses; instead
// the warp groups lanes by bucket (one ballot per distinct bucket) and one leader adds the
// group's count and mass with one shared atomic (other warps add to the same bins). Integer
// adds keep the sums order-independent.
__device__ __forceinline__ void warp_histogram_add(u32 slot, u64 mass, u32* hist_count,
                                                   u64* hist_mass) {
    const u32 lane = threadIdx.x & 31;
    unsigned pending = __ballot_sync(FULL_MASK, slot != NO_BUCKET);
    while (pending) {
        const int leader = __ffs(pending) - 1;
        const u32 bucket = __shfl_sync(FULL_MASK, slot, leader);
        const unsigned group = __ballot_sync(FULL_MASK, slot == bucket);
        if (hist_mass) {
            u64 m = slot == bucket ? mass : 0;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                m += __shfl_xor_sync(FULL_MASK, m, offset);
            }
            if (lane == (u32)leader) atomicAdd(&hist_mass[bucket], m);
        }
        if (lane == (u32)leader) atomicAdd(&hist_count[bucket], (u32)__popc(group));
        pending &= ~group;
    }
}

// The kept-set predicate: above the top-k threshold (ties up to an index cutoff) and above the
// top-p threshold (likewise). A `(0, NO_INDEX)` threshold keeps everything.
__device__ __forceinline__ bool kept_by(u32 key, u32 i, u32 t_key, u32 t_cut) {
    return key > t_key || (key == t_key && i <= t_cut);
}

extern "C" __global__ void candle_llm_sample_rows_f32(
    const float* logits,
    u32 vocab,
    float inv_t,
    u32 top_k,
    float top_p,
    u64 rng_state,
    float u_override,
    u32* out) {
    __shared__ u32 hist_count[256];
    __shared__ u64 hist_mass[256];
    __shared__ u32 scan32[SAMPLER_WARPS];
    __shared__ u64 scan64[SAMPLER_WARPS];
    __shared__ float arg_v[SAMPLER_WARPS];
    __shared__ u32 arg_i[SAMPLER_WARPS];
    __shared__ u32 sel32[4];
    __shared__ u64 sel64;

    const u32 row = blockIdx.x;
    const u32 tid = threadIdx.x;
    const float* x = logits + (u64)row * vocab;

    // 1. Argmax (first maximum) and NaN detection.
    float best = __uint_as_float(0xff800000u);  // -inf
    u32 best_i = NO_INDEX;
    u32 nan_seen = 0;
    for (u32 i = tid; i < vocab; i += SAMPLER_THREADS) {
        float v = x[i];
        if (is_nan(v)) {
            nan_seen = 1;
        } else if (v > best) {
            best = v;
            best_i = i;
        }
    }
    const u32 nans = block_sum<u32>(nan_seen, scan32);
    block_argmax(&best, &best_i, arg_v, arg_i);
    if (best_i == NO_INDEX) best_i = 0;  // every logit -inf: the host argmax answers 0
    if (nans > 0 || !is_finite(best)) {
        if (tid == 0) out[row] = best_i;
        return;
    }
    const float mx = best;
    // Each thread's contiguous index segment, for the index-order scans (one block scan each
    // instead of one per 1024-token chunk).
    const u32 seg = (vocab + SAMPLER_THREADS - 1) / SAMPLER_THREADS;
    const u32 seg_lo = min(tid * seg, vocab);
    const u32 seg_hi = min(seg_lo + seg, vocab);

    // 2. top-k: radix-select the k-th largest weight key, then the index cutoff among its ties.
    u32 k_key = 0, k_cut = NO_INDEX;
    if (top_k > 0 && top_k < vocab) {
        u32 prefix = 0, mask = 0, remaining = top_k;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (u32 b = tid; b < 256; b += SAMPLER_THREADS) hist_count[b] = 0;
            __syncthreads();
            for (u32 base = 0; base < vocab; base += SAMPLER_THREADS) {
                const u32 i = base + tid;
                u32 slot = NO_BUCKET;
                if (i < vocab) {
                    u32 key = key_of(weight_of(x[i], mx, inv_t));
                    if ((key & mask) == prefix) slot = (key >> shift) & 255u;
                }
                warp_histogram_add(slot, 0, hist_count, 0);
            }
            __syncthreads();
            if (tid == 0) {
                u32 rem = remaining, digit = 0;
                for (int b = 255; b >= 0; --b) {
                    u32 c = hist_count[b];
                    if (rem <= c) { digit = (u32)b; break; }
                    rem -= c;
                }
                sel32[0] = digit;
                sel32[1] = rem;
            }
            __syncthreads();
            prefix |= sel32[0] << shift;
            mask |= 255u << shift;
            remaining = sel32[1];
            __syncthreads();
        }
        k_key = prefix;
        // The `remaining`-th token (index order) whose key equals the threshold: count per
        // contiguous segment, scan the counts, and let the owning thread walk its segment.
        u32 count = 0;
        for (u32 i = seg_lo; i < seg_hi; ++i) {
            if (key_of(weight_of(x[i], mx, inv_t)) == k_key) ++count;
        }
        u32 total;
        const u32 incl = block_inclusive_scan<u32>(count, scan32, &total);
        u32 seen = incl - count;
        if (seen < remaining && remaining <= incl) {
            for (u32 i = seg_lo; i < seg_hi; ++i) {
                if (key_of(weight_of(x[i], mx, inv_t)) == k_key && ++seen == remaining) {
                    sel32[2] = i;
                    break;
                }
            }
        }
        __syncthreads();
        k_cut = sel32[2];
        __syncthreads();
    }

    // 3. top-p: radix-select, by mass, the weight key where the descending prefix reaches
    //    top_p * total, then how many of its ties (index order) the prefix needs.
    u32 p_key = 0, p_cut = NO_INDEX;
    if (top_p < 1.0f) {
        u64 local = 0;
        for (u32 i = tid; i < vocab; i += SAMPLER_THREADS) {
            float w = weight_of(x[i], mx, inv_t);
            if (kept_by(key_of(w), i, k_key, k_cut)) local += fixed_of(w);
        }
        const u64 mass = block_sum<u64>(local, scan64);
        const double threshold = (double)fmaxf(top_p, 0.0f) * (double)mass;
        // Pruning: the max weight is 1, so the mass S >= 1, and the tokens lighter than
        // (1 - top_p) / vocab weigh less than (1 - top_p) * S together - the nucleus reaches
        // top_p * S before it gets to any of them. They are left out of the radix histograms
        // (their mass is already in `mass`); on real logits that is almost the whole vocabulary.
        // Half the bound is kept as margin, and the test is on the exponent (no expf) so a
        // pruned token costs one load and one multiply.
        const float log_floor = logf(0.5f * (1.0f - fmaxf(top_p, 0.0f)) / (float)vocab);
        u32 prefix = 0, mask = 0;
        u64 above = 0;  // mass strictly above the current prefix range
        u32 ties = 0;
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (u32 b = tid; b < 256; b += SAMPLER_THREADS) {
                hist_count[b] = 0;
                hist_mass[b] = 0;
            }
            __syncthreads();
            for (u32 base = 0; base < vocab; base += SAMPLER_THREADS) {
                const u32 i = base + tid;
                u32 slot = NO_BUCKET;
                u64 f = 0;
                const float z = i < vocab ? (x[i] - mx) * inv_t : log_floor - 1.0f;
                if (z >= log_floor) {
                    float w = expf(z);
                    u32 key = key_of(w);
                    if (kept_by(key, i, k_key, k_cut) && (key & mask) == prefix) {
                        slot = (key >> shift) & 255u;
                        f = fixed_of(w);
                    }
                }
                warp_histogram_add(slot, f, hist_count, hist_mass);
            }
            __syncthreads();
            if (tid == 0) {
                u64 acc = above;
                u32 digit = NO_INDEX, last = 0;
                for (int b = 255; b >= 0; --b) {
                    if (hist_count[b] == 0) continue;
                    last = (u32)b;
                    if ((double)(acc + hist_mass[b]) >= threshold) { digit = (u32)b; break; }
                    acc += hist_mass[b];
                }
                if (digit == NO_INDEX) {
                    // Rounding left the threshold unreached: take the lightest non-empty bucket.
                    digit = last;
                    acc -= hist_mass[last];
                }
                sel32[0] = digit;
                sel32[1] = hist_count[digit];
                sel64 = acc;
            }
            __syncthreads();
            prefix |= sel32[0] << shift;
            mask |= 255u << shift;
            above = sel64;
            ties = sel32[1];
            __syncthreads();
        }
        p_key = prefix;
        // Every tie has the same weight, so the count the prefix needs is closed-form.
        const u64 f = fixed_of(__uint_as_float(p_key));
        u32 need = ties;
        if (f > 0) {
            double more = (threshold - (double)above) / (double)f;
            need = more <= 1.0 ? 1u : (u32)ceil(more);
            if (need > ties) need = ties;
        }
        u32 count = 0;
        for (u32 i = seg_lo; i < seg_hi; ++i) {
            u32 key = key_of(weight_of(x[i], mx, inv_t));
            if (key == p_key && kept_by(key, i, k_key, k_cut)) ++count;
        }
        u32 total;
        const u32 incl = block_inclusive_scan<u32>(count, scan32, &total);
        u32 seen = incl - count;
        if (seen < need && need <= incl) {
            for (u32 i = seg_lo; i < seg_hi; ++i) {
                u32 key = key_of(weight_of(x[i], mx, inv_t));
                if (key == p_key && kept_by(key, i, k_key, k_cut) && ++seen == need) {
                    sel32[2] = i;
                    break;
                }
            }
        }
        __syncthreads();
        p_cut = sel32[2];
        __syncthreads();
    }

    // 4. Categorical inverse-CDF draw in index order over the kept fixed-point weights: the
    //    segment whose exclusive..inclusive mass range brackets the target walks itself.
    u64 local = 0;
    for (u32 i = seg_lo; i < seg_hi; ++i) {
        float w = weight_of(x[i], mx, inv_t);
        u32 key = key_of(w);
        if (kept_by(key, i, k_key, k_cut) && kept_by(key, i, p_key, p_cut)) local += fixed_of(w);
    }
    if (tid == 0) sel32[3] = best_i;  // the argmax is always kept: the rounding fallback
    u64 kept_mass;
    const u64 incl = block_inclusive_scan<u64>(local, scan64, &kept_mass);
    const float u = u_override >= 0.0f
        ? u_override
        : splitmix_uniform(rng_state + (u64)(row + 1) * SPLITMIX_INCREMENT);
    const double target = (double)u * (double)kept_mass;
    u64 cum = incl - local;
    if (local > 0 && (double)cum <= target && (double)incl > target) {
        for (u32 i = seg_lo; i < seg_hi; ++i) {
            float w = weight_of(x[i], mx, inv_t);
            u32 key = key_of(w);
            if (!(kept_by(key, i, k_key, k_cut) && kept_by(key, i, p_key, p_cut))) continue;
            u64 f = fixed_of(w);
            if (f == 0) continue;
            cum += f;
            if ((double)cum > target) {
                sel32[3] = i;
                break;
            }
        }
    }
    __syncthreads();
    if (tid == 0) out[row] = sel32[3];
}
