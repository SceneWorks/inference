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
use std::sync::OnceLock;

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
    // Tiled output `[g, video * 128, D]`; padding rows are written as zero.
    const uint base = ((g * video + qtile) * 128u + local) * uint(D) + frag.y;
#pragma clang loop unroll(full)
    for (int d = 0; d < DT; ++d) {
#pragma clang loop unroll(full)
        for (int e = 0; e < 2; ++e) {
            const float o = live && run_sum > 0.0f ? om[d].thread_elements()[e] / run_sum : 0.0f;
            out[base + uint(d * 8 + e)] = static_cast<T>(o);
        }
    }
"#;

struct Kernels {
    pool: MetalKernel,
    project: MetalKernel,
    logits: MetalKernel,
    attend: MetalKernel,
}
// MLX kernel handles are immutable descriptions (source + names); MLX compiles and caches the
// pipelines itself, keyed on name/source/template arguments, behind its own device lock.
unsafe impl Send for Kernels {}
unsafe impl Sync for Kernels {}

fn kernels() -> Result<&'static Kernels> {
    static KERNELS: OnceLock<std::result::Result<Kernels, String>> = OnceLock::new();
    KERNELS
        .get_or_init(|| {
            let make = |name: &str, inputs: &[&str], outputs: &[&str], body: &str| {
                MetalKernel::with_options(
                    &format!("sc26214_veda_{name}"),
                    inputs,
                    outputs,
                    body,
                    HEADER,
                    true,
                    false,
                )
                .map_err(|e| e.to_string())
            };
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
                attend: make(
                    "attend",
                    &[
                        "q", "k", "v", "perm", "valid", "heads", "keys", "counts", "params",
                    ],
                    &["out"],
                    ATTEND,
                )?,
            })
        })
        .as_ref()
        .map_err(|e| Error::Msg(format!("Veda Metal kernels: {e}")))
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
/// `[seq, heads_total, dim]` Q and K; `proj_*` the device-resident `[layers, heads, 3*dim, dim]`
/// BF16 projections.
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
    let k = kernels()?;
    let total = x_q.shape()[1];
    let (g, video, d) = (heads.len() as i32, layout.video as i32, dim as i32);
    let ids = ints(heads);
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
}

/// Sparse attention of every video query tile of one head batch over its selected key tiles,
/// returned in tiled order `[heads, video * 128, dim]` (padding rows zero). `rows` are the
/// batch's head-major selected rows (`heads * tiles` entries, ascending key tiles).
pub fn attend(
    layout: &DeviceLayout,
    q: &Array,
    k: &Array,
    v: &Array,
    heads: &[i32],
    rows: &[Vec<usize>],
) -> Result<Array> {
    let kernels = kernels()?;
    let total = q.shape()[1];
    let (g, tiles, video) = (heads.len(), layout.tiles, layout.video);
    if rows.len() != g * tiles || q.shape()[2] != HEAD_DIM as i32 {
        return Err(Error::Msg("Veda attention operand mismatch".into()));
    }
    let kmax = (0..g)
        .flat_map(|h| rows[h * tiles..h * tiles + video].iter().map(Vec::len))
        .max()
        .unwrap_or(0);
    if kmax == 0 {
        return Err(Error::Msg("Veda empty selection".into()));
    }
    let mut keys = vec![0i32; g * video * kmax];
    let mut counts = vec![0u32; g * video];
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
    Ok(kernels
        .attend
        .apply()
        .inputs([
            q,
            k,
            v,
            &layout.perm,
            &layout.valid,
            &ints(heads),
            &keys,
            &counts,
            &ints(&[total, video as i32, kmax as i32]),
        ])
        .output_shape(
            vec![g as i32, (video * 128) as i32, HEAD_DIM as i32],
            q.dtype(),
        )
        .grid(32 * SIMD_GROUPS * 4 * video as i32, g as i32, 1)
        .thread_group(32 * SIMD_GROUPS, 1, 1)
        .run()?
        .remove(0))
}
