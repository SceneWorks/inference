// Provenance (SW-F287 E10): an original implementation of the published Gated DeltaNet recurrence
// (Yang et al., "Gated Delta Networks"), written against this repository's candle op chain bit for
// bit (inference PR #1469). No source is copied from comfy-kitchen's or mlx_lm's kernels, so no
// third-party licence applies.
//
// Fused gated delta recurrence for decode and speculative verify (Gated DeltaNet linear attention).
//
// One warp owns one (batch, value head, value row) row of the recurrent state S [B, Hv, Dv, Dk]
// (Dk / 32 values per lane), keeps it in registers across the step's T tokens and, per token t:
//
//   S   = S * g[t]                    (decay)
//   kv  = sum_dk S * k[t]             (S . k)
//   d   = (v[t] - kv) * beta[t]
//   S   = S + k[t] * d                (rank-1 delta update)
//   y   = sum_dk S * q[t]             (S . q)
//
// writing y and, for t >= first_write, the post-token row into the checkpoint ring slot
// write_index[t].
//
// Bit-identical to the op-chain reference (candle elementwise kernels + `fast_sum`): every
// elementwise step is the same single IEEE operation (__fmul_rn / __fadd_rn / __fsub_rn, no
// contraction), and each dot product sums in exactly `fast_sum`'s order for a Dk-wide row (one
// block of Dk threads, `shr[t] = 0 + x[t]`, then `shr[t] += shr[t + s]` for s = Dk/2 .. 1): the
// s >= 32 levels pair a lane's own registers, the s < 32 levels are a warp butterfly whose lane 0
// holds the tree's value (IEEE addition is commutative), broadcast to the warp.
//
// GQA: query/key head hk serves value heads hk*r .. hk*r + r - 1 (r = Hv / Hk), the reference's
// contiguous head repeat.

#define GD_WARP 32
#define GD_ROWS_PER_BLOCK 4

// fast_sum's tree over the row's products; p[i] holds element lane + 32 * i.
template <int PER_LANE>
__device__ __forceinline__ float gd_tree_sum(float (&p)[PER_LANE]) {
#pragma unroll
    for (int m = PER_LANE / 2; m >= 1; m >>= 1) {
#pragma unroll
        for (int i = 0; i < m; ++i) {
            p[i] = __fadd_rn(p[i], p[i + m]);
        }
    }
    float x = p[0];
#pragma unroll
    for (int o = GD_WARP / 2; o > 0; o >>= 1) {
        x = __fadd_rn(x, __shfl_xor_sync(0xffffffffu, x, o));
    }
    return __shfl_sync(0xffffffffu, x, 0);
}

// src: the live state. With src_index, slot src_index[0] of a [src_slots, B, Hv, Dv, Dk] ring
// (an index >= src_slots reads zeros); without, a [B, Hv, Dv, Dk] tensor. has_state == 0 starts
// from zeros (the first step of a sequence).
// ring: [ring_slots, B, Hv, Dv, Dk]; token t >= first_write writes slot write_index[t] (an index
// >= ring_slots is skipped). src and ring may be the same buffer: a warp reads its row before it
// writes any, and no other warp touches that row.
template <int PER_LANE>
__device__ void gated_delta_step(
    const float* src, const unsigned int* __restrict__ src_index,
    unsigned long long src_slots, int has_state,
    float* ring, const unsigned int* __restrict__ write_index,
    unsigned long long ring_slots, int first_write,
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta, float* __restrict__ y,
    int B, int T, int Hk, int Hv, int Dv) {
    constexpr int Dk = PER_LANE * GD_WARP;
    const int lane = threadIdx.x % GD_WARP;
    const long long row = (long long)blockIdx.x * GD_ROWS_PER_BLOCK + threadIdx.x / GD_WARP;
    const long long rows = (long long)B * Hv * Dv;
    if (row >= rows) {
        return;
    }
    const int dv = (int)(row % Dv);
    const int hv = (int)((row / Dv) % Hv);
    const int b = (int)(row / ((long long)Dv * Hv));
    const int hk = hv / (Hv / Hk);
    const unsigned long long slot_elems = (unsigned long long)rows * Dk;
    const unsigned long long row_off = (unsigned long long)row * Dk;

    float s[PER_LANE];
    const float* src_row = nullptr;
    if (has_state) {
        if (src_index != nullptr) {
            const unsigned long long at = src_index[0];
            if (at < src_slots) {
                src_row = src + at * slot_elems + row_off;
            }
        } else {
            src_row = src + row_off;
        }
    }
#pragma unroll
    for (int i = 0; i < PER_LANE; ++i) {
        s[i] = src_row != nullptr ? src_row[lane + i * GD_WARP] : 0.0f;
    }

    for (int t = 0; t < T; ++t) {
        const long long bt = (long long)b * T + t;
        const float* kt = k + (bt * Hk + hk) * Dk;
        const float* qt = q + (bt * Hk + hk) * Dk;
        const float gt = g[bt * Hv + hv];
        const float bet = beta[bt * Hv + hv];
        const float vt = v[(bt * Hv + hv) * Dv + dv];

        float kr[PER_LANE];
        float p[PER_LANE];
#pragma unroll
        for (int i = 0; i < PER_LANE; ++i) {
            kr[i] = kt[lane + i * GD_WARP];
            s[i] = __fmul_rn(s[i], gt);
            p[i] = __fadd_rn(0.0f, __fmul_rn(s[i], kr[i]));
        }
        const float kv = gd_tree_sum<PER_LANE>(p);
        const float d = __fmul_rn(__fsub_rn(vt, kv), bet);
#pragma unroll
        for (int i = 0; i < PER_LANE; ++i) {
            s[i] = __fadd_rn(s[i], __fmul_rn(kr[i], d));
            p[i] = __fadd_rn(0.0f, __fmul_rn(s[i], qt[lane + i * GD_WARP]));
        }
        const float out = gd_tree_sum<PER_LANE>(p);
        if (lane == 0) {
            y[(bt * Hv + hv) * Dv + dv] = out;
        }
        if (t >= first_write) {
            const unsigned long long at = write_index[t];
            if (at < ring_slots) {
                float* dst = ring + at * slot_elems + row_off;
#pragma unroll
                for (int i = 0; i < PER_LANE; ++i) {
                    dst[lane + i * GD_WARP] = s[i];
                }
            }
        }
    }
}

#define GD_ENTRY(DK)                                                                            \
    extern "C" __global__ void gated_delta_step_f32_dk##DK(                                     \
        const float* src, const unsigned int* __restrict__ src_index,                           \
        unsigned long long src_slots, int has_state, float* ring,                               \
        const unsigned int* __restrict__ write_index, unsigned long long ring_slots,            \
        int first_write, const float* __restrict__ q, const float* __restrict__ k,              \
        const float* __restrict__ v, const float* __restrict__ g,                               \
        const float* __restrict__ beta, float* __restrict__ y, int B, int T, int Hk, int Hv,   \
        int Dv) {                                                                               \
        gated_delta_step<DK / GD_WARP>(src, src_index, src_slots, has_state, ring, write_index, \
                                       ring_slots, first_write, q, k, v, g, beta, y, B, T, Hk,  \
                                       Hv, Dv);                                                 \
    }

GD_ENTRY(32)
GD_ENTRY(64)
GD_ENTRY(128)
GD_ENTRY(256)
