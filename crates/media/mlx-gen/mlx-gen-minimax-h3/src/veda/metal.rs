//! Metal kernels of the MLX Veda route (sc-26214): the frozen tile scorer, bit for bit, and the
//! block-sparse attention over each query tile's selected key tiles.
//!
//! Both read the caller's Q/K/V in their native packed `[seq, heads, dim]` layout through the
//! layout's tile permutation, so no tiled, gathered or padded copy of Q, K or V is ever built: a
//! key tile shared by many query tiles (every dense global tile, notably) is read in place by each
//! of them rather than copied once per query tile.
use mlx_gen::{Error, Result};
use mlx_rs::fast::MetalKernel;
use mlx_rs::{Array, Dtype};
use std::cell::OnceCell;

/// SIMD groups of one attention threadgroup, eight query rows each: a quarter of a 128-row tile.
const SIMD_GROUPS: i32 = 4;
/// Head dimension the attention kernel is specialised for (MiniMax-H3).
pub const HEAD_DIM: usize = 128;

const HEADER: &str = r#"#include <metal_stdlib>
using namespace metal;

template <typename T> struct veda_element;
template <typename T> struct veda_element<device T*> { using type = T; };

// Row and first column of this lane's two elements of an 8x8 simdgroup matrix (the Apple GPU
// fragment layout MLX's steel kernels use).
inline ushort2 veda_frag(uint lane) {
    const uint qid = lane / 4u;
    return ushort2((qid & 4u) + ((lane / 2u) % 4u), (qid & 2u) * 2u + (lane % 2u) * 2u);
}

// A correctly rounded product that no compiler may fuse into a following add: `fma(a, b, +0)`
// rounds `a * b` once, and an `fma` followed by an add is never contracted. Every sum below is a
// sequential FP32 accumulation from 0.0f in the host contract's exact operation order, so a result
// depends only on its own operands, never on how many heads or tiles share the dispatch.
inline float veda_mul(float a, float b) { return fma(a, b, 0.0f); }
"#;

/// `pool` / `project` / `logits` of gen-core's host `pool_video_tiles` + `tile_logits`.
const POOL: &str = r#"
    // params: heads_total, video, dim
    const int H = params[0], video = params[1], D = params[2];
    const uint i = thread_position_in_grid.x;
    const uint g = thread_position_in_grid.y;
    if (int(i) >= video * D) return;
    const int d = int(i) % D, t = int(i) / D;
    const int head = heads[g];
    const uint count = valid[t];
    const uint out = ((g * uint(video) + uint(t)) * 3u) * uint(D) + uint(d);
    if (count == 0u) { f[out] = 0.0f; f[out + D] = 0.0f; f[out + 2 * D] = 0.0f; return; }
    float sum = 0.0f;
    float mx = -INFINITY;
    float mn = INFINITY;
    for (int r = 0; r < 128; ++r) {
        const int row = perm[t * 128 + r];
        // Padding slots are zero in the host's tiled operand, and are summed like any row.
        const float v = row >= 0 ? float(x[(uint(row) * uint(H) + uint(head)) * uint(D) + uint(d)]) : 0.0f;
        sum = sum + v;
        if (uint(r) < count) { mx = fmax(mx, v); mn = fmin(mn, v); }
    }
    f[out] = precise::divide(sum, float(count));
    f[out + D] = mx;
    f[out + 2 * D] = mn;
"#;

const PROJECT: &str = r#"
    // params: heads_total, video, dim, layer.  proj_*: [layers, heads_total, 3*dim, dim] bf16.
    const int H = params[0], video = params[1], D = params[2], layer = params[3];
    const uint i = thread_position_in_grid.x;
    const uint g = thread_position_in_grid.y;
    if (int(i) >= video * D) return;
    const int d = int(i) % D, t = int(i) / D;
    const uint head = uint(heads[g]);
    const uint base = ((uint(layer) * uint(H) + head) * 3u * uint(D)) * uint(D) + uint(d);
    const device float* qr = fq + (g * uint(video) + uint(t)) * 3u * uint(D);
    const device float* kr = fk + (g * uint(video) + uint(t)) * 3u * uint(D);
    float vq = 0.0f;
    float vk = 0.0f;
    for (int n = 0; n < 3 * D; ++n) {
        vq = vq + veda_mul(qr[n], float(proj_q[base + uint(n) * uint(D)]));
        vk = vk + veda_mul(kr[n], float(proj_k[base + uint(n) * uint(D)]));
    }
    const uint out = (g * uint(video) + uint(t)) * uint(D) + uint(d);
    oq[out] = vq + qr[d];
    ok[out] = vk + kr[d];
"#;

const LOGITS: &str = r#"
    // params: video, dim.  divisor: [1] FP32 sqrt(dim), computed on the host.
    const int video = params[0], D = params[1];
    const uint i = thread_position_in_grid.x;
    const uint g = thread_position_in_grid.y;
    if (int(i) >= video * video) return;
    const int c = int(i) % video, r = int(i) / video;
    const device float* qr = q + (g * uint(video) + uint(r)) * uint(D);
    const device float* kr = k + (g * uint(video) + uint(c)) * uint(D);
    float v = 0.0f;
    for (int d = 0; d < D; ++d) v = v + veda_mul(qr[d], kr[d]);
    s[(g * uint(video) + uint(r)) * uint(video) + uint(c)] = precise::divide(v, divisor[0]);
"#;

/// One threadgroup = 32 query rows (one quarter of a video query tile) of one head; it streams the
/// tile's selected key tiles in 16-token blocks through threadgroup memory with an FP32 online
/// softmax, `simdgroup_matrix` products and per-tile validity (partial tiles mask their padding).
const ATTEND: &str = r#"
    // params: heads_total, video, kmax
    constexpr int D = 128;
    constexpr int BK = 16;
    constexpr int DT = D / 8;
    constexpr int KT8 = BK / 8;
    constexpr uint LD = D + 4;
    constexpr uint QUADS = BK * D / 4;
    using T = typename veda_element<decltype(out)>::type;
    threadgroup float k_tile[BK * LD];
    threadgroup float v_tile[BK * LD];

    const uint H = uint(params[0]);
    const uint video = uint(params[1]);
    const uint kmax = uint(params[2]);
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint qtile = threadgroup_position_in_grid.x / 4u;
    const uint quarter = threadgroup_position_in_grid.x % 4u;
    const uint g = threadgroup_position_in_grid.y;
    const uint head = uint(heads[g]);
    const ushort2 frag = veda_frag(lane);
    const uint local = quarter * 32u + sg * 8u + frag.x;
    const int qrow = perm[qtile * 128u + local];
    const bool live = local < valid[qtile];

    const float q_scale = rsqrt(float(D)) * M_LOG2E_F;
    simdgroup_float8x8 qm[DT];
    simdgroup_float8x8 om[DT];
#pragma clang loop unroll(full)
    for (int d = 0; d < DT; ++d) {
        float a = 0.0f, b = 0.0f;
        if (live) {
            const device T* qp = q + (uint(qrow) * H + head) * uint(D) + uint(d * 8) + frag.y;
            a = float(qp[0]) * q_scale;
            b = float(qp[1]) * q_scale;
        }
        qm[d].thread_elements()[0] = a;
        qm[d].thread_elements()[1] = b;
        om[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float run_max = -INFINITY;
    float run_sum = 0.0f;
    const uint list = g * video + qtile;
    const uint n = counts[list];
    for (uint i = 0; i < n; ++i) {
        const uint kt = uint(keys[list * kmax + i]);
        const uint kv = valid[kt];
        for (uint t0 = 0; t0 < kv; t0 += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint idx = tid; idx < QUADS; idx += 32u * 4u) {
                const uint tt = idx / (D / 4);
                const uint c0 = (idx % (D / 4)) * 4;
                float4 kq = float4(0.0f);
                float4 vq = float4(0.0f);
                if (t0 + tt < kv) {
                    const uint row = uint(perm[kt * 128u + t0 + tt]);
                    const uint at = (row * H + head) * uint(D) + c0;
                    kq = float4(float(k[at]), float(k[at + 1]), float(k[at + 2]), float(k[at + 3]));
                    vq = float4(float(v[at]), float(v[at + 1]), float(v[at + 2]), float(v[at + 3]));
                }
                *(threadgroup float4*)(k_tile + tt * LD + c0) = kq;
                *(threadgroup float4*)(v_tile + tt * LD + c0) = vq;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 sm[KT8];
#pragma clang loop unroll(full)
            for (int j = 0; j < KT8; ++j) sm[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#pragma clang loop unroll(full)
            for (int d = 0; d < DT; ++d) {
#pragma clang loop unroll(full)
                for (int j = 0; j < KT8; ++j) {
                    simdgroup_float8x8 km;
                    simdgroup_load(km, k_tile + j * 8 * LD + d * 8, LD, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(sm[j], qm[d], km, sm[j]);
                }
            }
            const bool full = t0 + BK <= kv;
            float block_max = -INFINITY;
#pragma clang loop unroll(full)
            for (int j = 0; j < KT8; ++j) {
#pragma clang loop unroll(full)
                for (int e = 0; e < 2; ++e) {
                    float s = sm[j].thread_elements()[e];
                    if (!full && t0 + uint(j * 8) + frag.y + uint(e) >= kv) s = -INFINITY;
                    sm[j].thread_elements()[e] = s;
                    block_max = max(block_max, s);
                }
            }
            block_max = max(block_max, simd_shuffle_xor(block_max, 1));
            block_max = max(block_max, simd_shuffle_xor(block_max, 8));
            const float next_max = max(run_max, block_max);
            const float factor = run_max == -INFINITY ? 0.0f : fast::exp2(run_max - next_max);
            float block_sum = 0.0f;
#pragma clang loop unroll(full)
            for (int j = 0; j < KT8; ++j) {
#pragma clang loop unroll(full)
                for (int e = 0; e < 2; ++e) {
                    const float s = sm[j].thread_elements()[e];
                    const float p = s == -INFINITY ? 0.0f : fast::exp2(s - next_max);
                    sm[j].thread_elements()[e] = p;
                    block_sum += p;
                }
            }
            block_sum += simd_shuffle_xor(block_sum, 1);
            block_sum += simd_shuffle_xor(block_sum, 8);
            run_sum = run_sum * factor + block_sum;
            run_max = next_max;
#pragma clang loop unroll(full)
            for (int d = 0; d < DT; ++d) {
                om[d].thread_elements()[0] *= factor;
                om[d].thread_elements()[1] *= factor;
            }
#pragma clang loop unroll(full)
            for (int j = 0; j < KT8; ++j) {
#pragma clang loop unroll(full)
                for (int d = 0; d < DT; ++d) {
                    simdgroup_float8x8 vm;
                    simdgroup_load(vm, v_tile + j * 8 * LD + d * 8, LD);
                    simdgroup_multiply_accumulate(om[d], sm[j], vm, om[d]);
                }
            }
        }
    }
    // Tiled output `[heads_total, video * 128, D]` at this head; padding rows are written as zero.
    const uint base = ((head * video + qtile) * 128u + local) * uint(D) + frag.y;
#pragma clang loop unroll(full)
    for (int d = 0; d < DT; ++d) {
#pragma clang loop unroll(full)
        for (int e = 0; e < 2; ++e) {
            const float o = live && run_sum > 0.0f ? om[d].thread_elements()[e] / run_sum : 0.0f;
            out[base + uint(d * 8 + e)] = static_cast<T>(o);
        }
    }
"#;

/// NAX header: MLX's `BaseNAXFrag` fragment layout and one `matmul2d` product on the GPU's
/// neural accelerators (MetalPerformancePrimitives), as inference's packed-KV NAX attention uses.
const NAX_HEADER: &str = r#"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// (column, row) of this lane's first element of a 16x16 fragment; the lane holds rows `fm` and
// `fm + 8`, columns `fn..fn+3`.
inline __attribute__((always_inline)) short2 veda_nax_coord(uint lane) {
    const short qid = short(lane >> 2);
    const short fm = (qid & 4) | short((lane >> 1) & 3u);
    const short fn = ((qid & 2) | short(lane & 1u)) * 4;
    return short2(fn, fm);
}

// C[16x32] += A[16x16] . B[16x32], B given as two 16x16 fragments (rows of B^T when `TB`).
template <bool TB, typename AT, typename BT>
inline __attribute__((always_inline)) void veda_nax_mma(thread vec<float, 8>& c0, thread vec<float, 8>& c1,
                            thread const vec<AT, 8>& a, thread const vec<BT, 8>& b0,
                            thread const vec<BT, 8>& b1) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, TB, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
    auto ct_a = op.template get_left_input_cooperative_tensor<AT, BT, float>();
    auto ct_b = op.template get_right_input_cooperative_tensor<AT, BT, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<decltype(ct_a), decltype(ct_b), float>();
#pragma clang loop unroll(full)
    for (short i = 0; i < 8; ++i) {
        ct_a[i] = a[i];
        ct_b[i] = b0[i];
        ct_b[8 + i] = b1[i];
        ct_c[i] = c0[i];
        ct_c[8 + i] = c1[i];
    }
    op.run(ct_a, ct_b, ct_c);
#pragma clang loop unroll(full)
    for (short i = 0; i < 8; ++i) {
        c0[i] = ct_c[i];
        c1[i] = ct_c[8 + i];
    }
}

template <typename T, int LD>
inline __attribute__((always_inline)) vec<T, 8> veda_nax_frag(const threadgroup T* tile, short2 coord, uint row0, uint col0) {
    const threadgroup T* base = tile + (row0 + coord.y) * LD + col0 + coord.x;
    const vec<T, 4> lo = *(const threadgroup vec<T, 4>*)(base);
    const vec<T, 4> hi = *(const threadgroup vec<T, 4>*)(base + 8 * LD);
    return vec<T, 8>(lo, hi);
}
"#;

/// The block-sparse attention on the neural accelerators: one threadgroup of eight SIMD groups
/// owns a whole 128-row query tile of one head (16 rows per SIMD group) and streams each selected
/// key tile in 32-token blocks through 16-bit threadgroup tiles; `Q.K^T` and `P.V` run as
/// `matmul2d` products with FP32 accumulation and an FP32 log2-domain online softmax.
const ATTEND_NAX: &str = r#"
    // params: heads_total, video, kmax
    constexpr int D = 128;
    constexpr int BK = 32;
    constexpr int LD = D + 8;
    constexpr int DT = D / 16;
    constexpr int KF = BK / 16;
    constexpr uint TSTEP = 256u / uint(D / 4);
    using T = metal::remove_cv_t<typename veda_element<decltype(q)>::type>;
    threadgroup T k_tile[BK * LD];
    threadgroup T v_tile[BK * LD];

    const uint H = uint(params[0]);
    const uint video = uint(params[1]);
    const uint kmax = uint(params[2]);
    const uint tid = thread_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint qtile = threadgroup_position_in_grid.x;
    const uint g = threadgroup_position_in_grid.y;
    const uint head = uint(heads[g]);
    const short2 coord = veda_nax_coord(lane);
    const uint c_quad = (tid % uint(D / 4)) * 4u;
    const uint t_row = tid / uint(D / 4);
    const uint qvalid = valid[qtile];
    uint local[2];
    bool live[2];
    const device T* q_row[2];
#pragma clang loop unroll(full)
    for (short i = 0; i < 2; ++i) {
        local[i] = sg * 16u + uint(coord.y) + 8u * uint(i);
        live[i] = local[i] < qvalid;
        const int row = live[i] ? perm[qtile * 128u + local[i]] : perm[qtile * 128u];
        q_row[i] = q + (uint(row) * H + head) * uint(D) + uint(coord.x);
    }
    const float scale2 = rsqrt(float(D)) * M_LOG2E_F;
    vec<float, 8> o[DT];
#pragma clang loop unroll(full)
    for (int d = 0; d < DT; ++d) o[d] = vec<float, 8>(0.0f);
    float run_max[2] = {-INFINITY, -INFINITY};
    float run_sum[2] = {0.0f, 0.0f};

    const uint list = g * video + qtile;
    const uint n = counts[list];
    for (uint i = 0; i < n; ++i) {
        const uint kt = uint(keys[list * kmax + i]);
        const uint kv = valid[kt];
        for (uint t0 = 0; t0 < kv; t0 += BK) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
            for (uint r = 0; r < uint(BK) / TSTEP; ++r) {
                const uint tt = t_row + r * TSTEP;
                vec<T, 4> kq = vec<T, 4>(0);
                vec<T, 4> vq = vec<T, 4>(0);
                if (t0 + tt < kv) {
                    const uint row = uint(perm[kt * 128u + t0 + tt]);
                    const uint at = (row * H + head) * uint(D) + c_quad;
                    kq = *(const device vec<T, 4>*)(k + at);
                    vq = *(const device vec<T, 4>*)(v + at);
                }
                *(threadgroup vec<T, 4>*)(k_tile + tt * LD + c_quad) = kq;
                *(threadgroup vec<T, 4>*)(v_tile + tt * LD + c_quad) = vq;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            vec<float, 8> s[KF];
#pragma clang loop unroll(full)
            for (int f = 0; f < KF; ++f) s[f] = vec<float, 8>(0.0f);
#pragma clang loop unroll(full)
            for (int f = 0; f < KF; f += 2) {
#pragma clang loop unroll(full)
                for (int d = 0; d < DT; ++d) {
                    const vec<T, 8> qd = vec<T, 8>(*(const device vec<T, 4>*)(q_row[0] + d * 16),
                                                   *(const device vec<T, 4>*)(q_row[1] + d * 16));
                    const vec<T, 8> k0 = veda_nax_frag<T, LD>(k_tile, coord, f * 16, d * 16);
                    const vec<T, 8> k1 = veda_nax_frag<T, LD>(k_tile, coord, f * 16 + 16, d * 16);
                    veda_nax_mma<true>(s[f], s[f + 1], qd, k0, k1);
                }
            }
            const bool full = t0 + uint(BK) <= kv;
            float block_max[2] = {-INFINITY, -INFINITY};
#pragma clang loop unroll(full)
            for (int f = 0; f < KF; ++f) {
#pragma clang loop unroll(full)
                for (short a = 0; a < 2; ++a) {
#pragma clang loop unroll(full)
                    for (short j = 0; j < 4; ++j) {
                        float x = s[f][4 * a + j] * scale2;
                        if (!full && t0 + uint(f * 16 + coord.x + j) >= kv) x = -INFINITY;
                        s[f][4 * a + j] = x;
                        block_max[a] = max(block_max[a], x);
                    }
                }
            }
            float factor[2];
#pragma clang loop unroll(full)
            for (short a = 0; a < 2; ++a) {
                block_max[a] = max(block_max[a], simd_shuffle_xor(block_max[a], 1));
                block_max[a] = max(block_max[a], simd_shuffle_xor(block_max[a], 8));
                const float next_max = max(run_max[a], block_max[a]);
                factor[a] = run_max[a] == -INFINITY ? 0.0f : fast::exp2(run_max[a] - next_max);
                float block_sum = 0.0f;
#pragma clang loop unroll(full)
                for (int f = 0; f < KF; ++f) {
#pragma clang loop unroll(full)
                    for (short j = 0; j < 4; ++j) {
                        const float x = s[f][4 * a + j];
                        const float p = x == -INFINITY ? 0.0f : fast::exp2(x - next_max);
                        s[f][4 * a + j] = p;
                        block_sum += p;
                    }
                }
                block_sum += simd_shuffle_xor(block_sum, 1);
                block_sum += simd_shuffle_xor(block_sum, 8);
                run_sum[a] = run_sum[a] * factor[a] + block_sum;
                run_max[a] = next_max;
            }
#pragma clang loop unroll(full)
            for (int d = 0; d < DT; ++d) {
#pragma clang loop unroll(full)
                for (short j = 0; j < 4; ++j) {
                    o[d][j] *= factor[0];
                    o[d][4 + j] *= factor[1];
                }
            }
#pragma clang loop unroll(full)
            for (int d = 0; d < DT; d += 2) {
#pragma clang loop unroll(full)
                for (int f = 0; f < KF; ++f) {
                    const vec<T, 8> v0 = veda_nax_frag<T, LD>(v_tile, coord, f * 16, d * 16);
                    const vec<T, 8> v1 = veda_nax_frag<T, LD>(v_tile, coord, f * 16, d * 16 + 16);
                    veda_nax_mma<false>(o[d], o[d + 1], s[f], v0, v1);
                }
            }
        }
    }
    // Tiled output `[heads_total, video * 128, D]` at this head; padding rows are written as zero.
#pragma clang loop unroll(full)
    for (short a = 0; a < 2; ++a) {
        const uint base = ((head * video + qtile) * 128u + local[a]) * uint(D);
#pragma clang loop unroll(full)
        for (int d = 0; d < DT; ++d) {
#pragma clang loop unroll(full)
            for (short j = 0; j < 4; ++j) {
                const float x = live[a] && run_sum[a] > 0.0f ? o[d][4 * a + j] / run_sum[a] : 0.0f;
                out[base + uint(d * 16 + coord.x + j)] = static_cast<T>(x);
            }
        }
    }
"#;

struct Kernels {
    pool: MetalKernel,
    project: MetalKernel,
    logits: MetalKernel,
    attend: MetalKernel,
    attend_nax: MetalKernel,
}

thread_local! {
    // `MetalKernel` is thread-affine (MLX's shared Metal device is not thread-safe), so each thread
    // builds its own handles — the core crate's `convrot_int8` / `w4a8_int8` pattern — instead of
    // asserting `Send`/`Sync` for a process-wide static. The handles capture no arrays, and MLX
    // caches the compiled pipelines itself (keyed on name/source/template arguments), so a second
    // thread's handles do not recompile Metal.
    static KERNELS: OnceCell<std::result::Result<Kernels, String>> = const { OnceCell::new() };
}

/// Run `f` with this thread's Veda kernels, building them on first use. A build failure is cached
/// for the thread and reported on every call.
fn with_kernels<T>(f: impl FnOnce(&Kernels) -> Result<T>) -> Result<T> {
    KERNELS.with(|cell| {
        let kernels = cell
            .get_or_init(|| {
                let nax_header = format!("{HEADER}{NAX_HEADER}");
                let with =
                    |name: &str, inputs: &[&str], outputs: &[&str], body: &str, header: &str| {
                        MetalKernel::with_options(
                            &format!("sc26214_veda_{name}"),
                            inputs,
                            outputs,
                            body,
                            header,
                            true,
                            false,
                        )
                        .map_err(|e| e.to_string())
                    };
                let make = |name: &str, inputs: &[&str], outputs: &[&str], body: &str| {
                    with(name, inputs, outputs, body, HEADER)
                };
                const ATTEND_INPUTS: [&str; 9] = [
                    "q", "k", "v", "perm", "valid", "heads", "keys", "counts", "params",
                ];
                Ok(Kernels {
                    pool: make(
                        "pool",
                        &["x", "perm", "valid", "heads", "params"],
                        &["f"],
                        POOL,
                    )?,
                    project: make(
                        "project",
                        &["fq", "fk", "proj_q", "proj_k", "heads", "params"],
                        &["oq", "ok"],
                        PROJECT,
                    )?,
                    logits: make("logits", &["q", "k", "divisor", "params"], &["s"], LOGITS)?,
                    attend: make("attend", &ATTEND_INPUTS, &["out"], ATTEND)?,
                    // Compiled on first dispatch only, so a GPU without the neural accelerators
                    // (whose Metal compiler may lack MetalPerformancePrimitives) never builds it.
                    attend_nax: with(
                        "attend_nax",
                        &ATTEND_INPUTS,
                        &["out"],
                        ATTEND_NAX,
                        &nax_header,
                    )?,
                })
            })
            .as_ref()
            .map_err(|e| Error::Msg(format!("Veda Metal kernels: {e}")))?;
        f(kernels)
    })
}

fn ints(values: &[i32]) -> Array {
    Array::from_slice(values, &[values.len() as i32])
}

/// Device-resident layout operands of one tile shape: permutation, per-tile valid counts, the
/// real global rows (in tile order) and the map from each native row to its position in
/// `concat(tiled video slots, global rows)`.
pub struct DeviceLayout {
    pub perm: Array,
    pub valid: Array,
    pub globals: Option<Array>,
    pub compact: Array,
    pub tiles: usize,
    pub video: usize,
}
impl DeviceLayout {
    pub fn new(layout: &super::contract::Layout) -> Self {
        let valid: Vec<u32> = layout.valid.iter().map(|&v| v as u32).collect();
        let video_slots = layout.video * super::contract::TILE;
        let globals: Vec<i32> = layout.perm[video_slots..]
            .iter()
            .copied()
            .filter(|&r| r >= 0)
            .collect();
        let mut compact = vec![0i32; layout.seq];
        for (slot, &row) in layout.perm[..video_slots].iter().enumerate() {
            if row >= 0 {
                compact[row as usize] = slot as i32;
            }
        }
        for (i, &row) in globals.iter().enumerate() {
            compact[row as usize] = (video_slots + i) as i32;
        }
        Self {
            perm: ints(&layout.perm),
            valid: Array::from_slice(&valid, &[valid.len() as i32]),
            globals: (!globals.is_empty()).then(|| ints(&globals)),
            compact: ints(&compact),
            tiles: layout.valid.len(),
            video: layout.video,
        }
    }
}

/// Frozen learned tile logits `[heads, video, video]` of one same-shape head batch, equal bit for
/// bit to gen-core's host `pool_video_tiles` + `tile_logits` for every head on its own, with one
/// evaluation and one host readback for the whole batch. `x_q`/`x_k` are the native
/// `[seq, heads_total, dim]` Q and K; `proj_*` device-resident BF16 projections
/// `[layers, heads, 3*dim, dim]` indexed at `layer` (the runtime passes one layer's
/// `[heads, 3*dim, dim]` with `layer = 0`).
#[allow(clippy::too_many_arguments)]
pub fn scores(
    layout: &DeviceLayout,
    x_q: &Array,
    x_k: &Array,
    proj_q: &Array,
    proj_k: &Array,
    heads: &[i32],
    layer: usize,
    dim: usize,
) -> Result<Vec<f32>> {
    let total = x_q.shape()[1];
    let (g, video, d) = (heads.len() as i32, layout.video as i32, dim as i32);
    let ids = ints(heads);
    with_kernels(|k| {
        let pool = |x: &Array| -> Result<Array> {
            Ok(k.pool
                .apply()
                .inputs([
                    x,
                    &layout.perm,
                    &layout.valid,
                    &ids,
                    &ints(&[total, video, d]),
                ])
                .output_shape(vec![g, video, 3 * d], Dtype::Float32)
                .grid(video * d, g, 1)
                .thread_group(256.min(video * d), 1, 1)
                .run()?
                .remove(0))
        };
        let (fq, fk) = (pool(x_q)?, pool(x_k)?);
        let mut projected = k
            .project
            .apply()
            .inputs([
                &fq,
                &fk,
                proj_q,
                proj_k,
                &ids,
                &ints(&[total, video, d, layer as i32]),
            ])
            .output_shape(vec![g, video, d], Dtype::Float32)
            .output_shape(vec![g, video, d], Dtype::Float32)
            .grid(video * d, g, 1)
            .thread_group(256.min(video * d), 1, 1)
            .run()?;
        let (pk, pq) = (projected.pop().unwrap(), projected.pop().unwrap());
        let divisor = Array::from_slice(&[(dim as f32).sqrt()], &[1]);
        let s = k
            .logits
            .apply()
            .inputs([&pq, &pk, &divisor, &ints(&[video, d])])
            .output_shape(vec![g, video, video], Dtype::Float32)
            .grid(video * video, g, 1)
            .thread_group(256.min(video * video), 1, 1)
            .run()?
            .remove(0);
        mlx_rs::transforms::eval([&s])?;
        Ok(s.as_slice::<f32>().to_vec())
    })
}

/// Sparse attention of every video query tile of `heads` (all heads of one tile shape) over
/// their selected key tiles, returned in tiled order `[total_heads, video * 128, dim]` with
/// padding rows zero. Only the listed heads' rows are written; every other head's rows are left
/// unspecified for the caller to mask. `rows` are the heads' head-major selected rows
/// (`heads * tiles` entries, ascending key tiles), each at most `kmax` key tiles.
///
/// Every buffer this allocates is sized by the layer-invariant `total_heads` and `kmax`, never by
/// how many heads the plan gives this shape in this layer, so MLX's buffer cache reuses them
/// across layers instead of accumulating a distinct size per layer.
pub fn attend(
    layout: &DeviceLayout,
    q: &Array,
    k: &Array,
    v: &Array,
    heads: &[i32],
    rows: &[Vec<usize>],
    kmax: usize,
) -> Result<Array> {
    attend_route(layout, q, k, v, heads, rows, kmax, nax_available())
}
/// [`attend`] on an explicit route: `nax` selects the neural-accelerator kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attend_route(
    layout: &DeviceLayout,
    q: &Array,
    k: &Array,
    v: &Array,
    heads: &[i32],
    rows: &[Vec<usize>],
    kmax: usize,
    nax: bool,
) -> Result<Array> {
    let total = q.shape()[1];
    let (g, tiles, video) = (heads.len(), layout.tiles, layout.video);
    if g == 0
        || g > total as usize
        || rows.len() != g * tiles
        || q.shape()[2] != HEAD_DIM as i32
        || heads.iter().any(|&h| h < 0 || h >= total)
    {
        return Err(Error::Msg("Veda attention operand mismatch".into()));
    }
    let longest = (0..g)
        .flat_map(|h| rows[h * tiles..h * tiles + video].iter().map(Vec::len))
        .max()
        .unwrap_or(0);
    if longest == 0 {
        return Err(Error::Msg("Veda empty selection".into()));
    }
    if longest > kmax {
        return Err(Error::Msg(
            "Veda selection exceeds admitted gather bound".into(),
        ));
    }
    let mut keys = vec![0i32; total as usize * video * kmax];
    let mut counts = vec![0u32; total as usize * video];
    for h in 0..g {
        for r in 0..video {
            let selected = &rows[h * tiles + r];
            let at = (h * video + r) * kmax;
            for (slot, &key) in keys[at..at + selected.len()].iter_mut().zip(selected) {
                *slot = key as i32;
            }
            counts[h * video + r] = selected.len() as u32;
        }
    }
    let keys = Array::from_slice(&keys, &[keys.len() as i32]);
    let counts = Array::from_slice(&counts, &[counts.len() as i32]);
    // NAX: one 256-thread threadgroup per 128-row query tile; otherwise four 128-thread quarters.
    // The NAX products are 16-bit: the runtime narrows FP32 operands once per layer; this keeps a
    // direct FP32 caller (tests) on the same single route rather than a dtype-chosen second one.
    let narrow = |a: &Array| -> Result<Array> {
        Ok(if nax && a.dtype() == Dtype::Float32 {
            a.as_dtype(Dtype::Bfloat16)?
        } else {
            a.clone()
        })
    };
    let (q16, k16, v16) = (narrow(q)?, narrow(k)?, narrow(v)?);
    let out = with_kernels(|kernels| {
        let (kernel, per_tile, threads) = if nax {
            (&kernels.attend_nax, 1, 256)
        } else {
            (&kernels.attend, 4, 32 * SIMD_GROUPS)
        };
        Ok(kernel
            .apply()
            .inputs([
                &q16,
                &k16,
                &v16,
                &layout.perm,
                &layout.valid,
                &ints(heads),
                &keys,
                &counts,
                &ints(&[total, video as i32, kmax as i32]),
            ])
            .output_shape(
                vec![total, (video * 128) as i32, HEAD_DIM as i32],
                q16.dtype(),
            )
            .grid(threads * per_tile * video as i32, g as i32, 1)
            .thread_group(threads, 1, 1)
            .run()?
            .remove(0))
    })?;
    Ok(if out.dtype() == q.dtype() {
        out
    } else {
        out.as_dtype(q.dtype())?
    })
}

/// Whether MLX runs its neural-accelerator (NAX) kernels on this device — MLX's own predicate,
/// over the already-created Metal device.
pub fn nax_available() -> bool {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            #[link_name = "_ZN3mlx4core5metal16is_nax_availableEv"]
            fn mlx_metal_is_nax_available() -> bool;
        }
        // SAFETY: a no-argument predicate whose result MLX caches in a function-local static.
        unsafe { mlx_metal_is_nax_available() }
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}
