//! Retained MLX Metal reader for SC-20675's packed K/V layout.
//!
//! K codes are token-group packed (`[B,H,groups,G·D/4]`, one f16 scale/zero per group and channel)
//! and V codes are channel-group packed (`[B,H,tokens,D/4]`, one f16 scale/zero per token and
//! channel group). The not-yet-quantized residual of each (at most one quantization group of the
//! most recent tokens) is a separate dense `[B,H,residual_capacity,D]` input, so the reader never
//! receives, and never builds, a concatenation of history and residual. Buffers may be
//! block-preallocated: the live extents are kernel parameters, not buffer shapes.
//!
//! Kernel design. Each threadgroup owns one query row (batch, query head, query position) and one
//! block-aligned slice of the visible KV range. Its SIMD groups stride over 32-token blocks; inside a
//! SIMD group every lane owns `D/32` contiguous channels, so one lane reads one packed byte per
//! token for K and for V, a score is one `simd_sum`, and the online-softmax state (running max,
//! normalizer, `D/32` accumulators) lives in registers with no barrier per token. SIMD groups merge
//! once through threadgroup memory at the end. A long history is split across several threadgroups
//! per row (split-KV); their partial `(max, sum, acc)` are merged by a second one-SIMD-group pass.
//! The causal/sliding bound is applied to the range before it is split, so masked tokens are never
//! visited. The kernel never materializes dense historical K/V or an `S_q×S_kv` score tensor.
use crate::error::{Error, Result};
use crate::primitives::packed_group_affine_kv::{
    packed_metal_head_dimension_supported, PACKED_CODES_PER_BYTE, PACKED_METAL_QUANT_GROUP_SIZE,
};
use mlx_rs::fast::{MetalKernel, OutputArg};
use mlx_rs::{Array, Dtype};

fn checked_shape(shape: &[i32], tensor: &str) -> Result<[usize; 4]> {
    let [batch, heads, tokens, width] = shape else {
        return Err(Error::Unsupported(format!("SC-20676 {tensor} buffer rank")));
    };
    [*batch, *heads, *tokens, *width]
        .map(|dimension| {
            usize::try_from(dimension).map_err(|_| {
                Error::Unsupported(format!("SC-20676 {tensor} buffer has a negative dimension"))
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()
        .and_then(|dimensions| {
            dimensions
                .try_into()
                .map_err(|_| Error::Msg("internal SC-20676 shape conversion failed".into()))
        })
}

fn checked_msl_i32(value: usize, name: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| Error::Unsupported(format!("SC-20676 {name} exceeds MSL i32 range")))
}

/// Mask forms the retained reader can prove without allocating a score matrix.  Arbitrary
/// additive masks deliberately select the observable dense fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedMask {
    None,
    Causal,
    SlidingWindow(usize),
    AdditiveUnsupported,
}

/// Explicit GPU-family tuning boundary. Unknown Apple GPUs run one SIMD group per threadgroup and
/// rely on split-KV threadgroups for parallelism; qualified recent families use eight cooperating
/// SIMD groups per threadgroup. No family outside this enum is silently assigned a geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PackedMetalGpuFamily {
    #[default]
    ConservativeUnknownApple,
    Apple7OrNewer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedMetalTuning {
    threads: usize,
    simd_groups: usize,
    values_per_thread: usize,
}

/// Dispatch geometry selected by the retained packed Metal reader. Evidence harnesses expose this
/// profile alongside the probed device name so performance receipts cannot silently change tuning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedMetalTuningProfile {
    pub gpu_family: &'static str,
    pub threads: usize,
    pub simd_groups: usize,
    pub values_per_thread: usize,
}

impl PackedMetalGpuFamily {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ConservativeUnknownApple => "conservative-unknown-apple",
            Self::Apple7OrNewer => "apple7-or-newer",
        }
    }

    fn tuning(self, head_dimension: usize) -> Option<PackedMetalTuning> {
        if !packed_metal_head_dimension_supported(head_dimension) {
            return None;
        }
        let simd_groups = match self {
            Self::ConservativeUnknownApple => 1,
            Self::Apple7OrNewer => 8,
        };
        Some(PackedMetalTuning {
            threads: simd_groups * SIMD_WIDTH,
            simd_groups,
            values_per_thread: head_dimension / SIMD_WIDTH,
        })
    }
}

const SIMD_WIDTH: usize = 32;
/// Register budget per lane for the GQA-shared accumulators (`heads_per_row · D/32`).
const MAX_LANE_ACCUMULATORS: usize = 32;
/// Resident SIMD groups a split-KV dispatch aims for: several waves over a 30–40-core Apple GPU.
const TARGET_RESIDENT_SIMD_GROUPS: usize = 4096;
/// Every split owns at least `max(MIN_BLOCKS_PER_SPLIT, 2 · simd_groups)` 32-token blocks, so the
/// per-split merge and the reduction pass stay amortized.
const MIN_BLOCKS_PER_SPLIT: usize = 4;
/// Upper bound on KV splits per query row (bounds the partial-result scratch and the reduction).
const MAX_KV_SPLITS: usize = 128;

/// Split-KV heuristic. One threadgroup serves one (row, KV split), where a row is `B · Sq` times
/// the query-head blocks sharing a KV head (8 rows for a Llama-3.2-3B or Qwen3-1.7B decode token).
/// That is far too few threadgroups to occupy the GPU while each walks the whole history, so the
/// visible range is cut into block-aligned splits such that (1) `rows · splits · simd_groups`
/// reaches 4096 resident SIMD groups, (2) each split still owns at least `max(4, 2 · simd_groups)`
/// blocks, and (3) at most 128 splits exist. One split selects the single-pass kernel (prefill
/// chunks, short histories); more select the partial pass plus the reduction pass. The constants
/// were chosen from the SC-20676 synthetic decode sweep on an M5 Max (4k–131k tokens, Hkv = 8,
/// D = 128, both tuning profiles).
pub fn packed_kv_split_count(rows: usize, visible_tokens: usize, simd_groups: usize) -> usize {
    let rows = rows.max(1);
    let simd_groups = simd_groups.max(1);
    let blocks = visible_tokens.div_ceil(PACKED_METAL_QUANT_GROUP_SIZE);
    let for_occupancy = TARGET_RESIDENT_SIMD_GROUPS.div_ceil(rows.saturating_mul(simd_groups));
    let for_amortization = blocks / MIN_BLOCKS_PER_SPLIT.max(2 * simd_groups);
    for_occupancy.min(for_amortization).clamp(1, MAX_KV_SPLITS)
}

const HEADER: &str = r#"#include <metal_stdlib>
using namespace metal;

template <typename T> struct sc20676_device_element;
template <typename T> struct sc20676_device_element<device T*> { using type = T; };

// Four 2-bit codes per byte; `index` is the code index of the lane's first channel, a multiple of
// the lane's channel count.
template <int EPL>
inline void sc20676_unpack(const device uint8_t* base, uint index, thread uint* codes) {
    if (EPL == 2) {
        const uint byte = base[index >> 2];
        const uint shift = (index & 3u) * 2u;
        codes[0] = (byte >> shift) & 3u;
        codes[1] = (byte >> (shift + 2u)) & 3u;
    } else {
        for (uint word = 0; word < uint(EPL) / 4u; ++word) {
            const uint byte = base[(index >> 2) + word];
            for (uint k = 0; k < 4u; ++k) codes[word * 4u + k] = (byte >> (2u * k)) & 3u;
        }
    }
}

// Fused packed attention for one threadgroup: one row of `QG` query heads sharing a KV head, one
// block-aligned KV split. SIMD groups stride over 32-token blocks; each lane owns `EPL` contiguous
// channels, so a score is one `simd_sum` and the online softmax lives in registers. Scores are in
// the log2 domain (`q` pre-scaled by log2(e)/sqrt(D)), so `exp2` is the softmax exponential. SIMD
// group 0 returns true holding the merged (unnormalized) accumulators, maxima, and sums for the
// row's heads; `qrow0` is the output row of the first head (head `g` is `qrow0 + g * SQ`).
template <int D, int QG, int BN, int MASK_MODE, int WINDOW, typename QT, typename KT, typename VT>
inline bool sc20676_attend(
    const device QT* q, const device uint8_t* k_codes, const device half* k_scale,
    const device half* k_zero, const device KT* k_tail, const device uint8_t* v_codes,
    const device half* v_scale, const device half* v_zero, const device VT* v_tail,
    uint k_packed, uint v_packed, uint kv_len, uint HQ, uint SQ, uint HKV, uint KG_CAP,
    uint KT_CAP, uint V_CAP, uint VT_CAP, threadgroup float* tg_max, threadgroup float* tg_sum,
    threadgroup float* tg_acc, uint lane, uint sg, uint split, uint splits, uint row,
    thread float* merged, thread float* out_max, thread float* out_sum, thread uint& qrow0) {
    constexpr int EPL = D / 32;
    constexpr int G = 32;
    constexpr int KW = G * D / 4;
    constexpr int VW = D / 4;
    constexpr int VG = D / G;
    const uint head_blocks = HQ / QG;
    const uint qi = row % SQ;
    const uint qh0 = ((row / SQ) % head_blocks) * QG;
    const uint b = row / (SQ * head_blocks);
    const uint kv_row = b * HKV + qh0 / (HQ / HKV);
    const uint qpos = kv_len - SQ + qi;
    const uint hi = MASK_MODE == 0 ? kv_len : qpos + 1;
    const uint lo = (MASK_MODE == 2 && hi > uint(WINDOW)) ? hi - uint(WINDOW) : 0;
    const uint first_block = lo / G;
    const uint end_block = (hi + G - 1) / G;
    const uint per_split = (end_block - first_block + splits - 1) / splits;
    const uint split_begin = min(end_block, first_block + split * per_split);
    const uint split_end = min(end_block, split_begin + per_split);
    const uint ch = lane * EPL;
    qrow0 = (b * HQ + qh0) * SQ + qi;

    const float q_scale = rsqrt(float(D)) * M_LOG2E_F;
    float qv[QG][EPL];
    float acc[QG][EPL];
    float run_max[QG];
    float run_sum[QG];
    uint codes[EPL];
    float vval[EPL];
    for (uint g = 0; g < QG; ++g) {
        for (uint j = 0; j < EPL; ++j) {
            qv[g][j] = float(q[(qrow0 + g * SQ) * D + ch + j]) * q_scale;
            acc[g][j] = 0.0f;
        }
        run_max[g] = -INFINITY;
        run_sum[g] = 0.0f;
    }

    for (uint blk = split_begin + sg; blk < split_end; blk += BN) {
        const uint t_begin = max(blk * G, lo);
        const uint t_end = min(blk * G + G, hi);
        const bool k_quantized = (blk + 1) * G <= k_packed;
        const device uint8_t* kc = k_codes + (kv_row * KG_CAP + blk) * KW;
        float qs[QG][EPL];
        float qz[QG];
        if (k_quantized) {
            const uint meta = (kv_row * KG_CAP + blk) * D + ch;
            float ks[EPL];
            float kz[EPL];
            for (uint j = 0; j < EPL; ++j) {
                ks[j] = float(k_scale[meta + j]);
                kz[j] = float(k_zero[meta + j]);
            }
            for (uint g = 0; g < QG; ++g) {
                qz[g] = 0.0f;
                for (uint j = 0; j < EPL; ++j) {
                    qs[g][j] = qv[g][j] * ks[j];
                    qz[g] += qv[g][j] * kz[j];
                }
            }
        }
        for (uint t = t_begin; t < t_end; ++t) {
            float score[QG];
            if (k_quantized) {
                sc20676_unpack<EPL>(kc, (t - blk * G) * D + ch, codes);
                float kf[EPL];
                for (uint j = 0; j < EPL; ++j) kf[j] = float(codes[j]);
                for (uint g = 0; g < QG; ++g) {
                    float partial = qz[g];
                    for (uint j = 0; j < EPL; ++j) partial += qs[g][j] * kf[j];
                    score[g] = simd_sum(partial);
                }
            } else {
                const uint base = (kv_row * KT_CAP + (t - k_packed)) * D + ch;
                float kf[EPL];
                for (uint j = 0; j < EPL; ++j) kf[j] = float(k_tail[base + j]);
                for (uint g = 0; g < QG; ++g) {
                    float partial = 0.0f;
                    for (uint j = 0; j < EPL; ++j) partial += qv[g][j] * kf[j];
                    score[g] = simd_sum(partial);
                }
            }
            if (t < v_packed) {
                const uint vrow = kv_row * V_CAP + t;
                const uint meta = vrow * VG + ch / G;
                const float vs = float(v_scale[meta]);
                const float vz = float(v_zero[meta]);
                sc20676_unpack<EPL>(v_codes + vrow * VW, ch, codes);
                for (uint j = 0; j < EPL; ++j) vval[j] = vz + vs * float(codes[j]);
            } else {
                const uint base = (kv_row * VT_CAP + (t - v_packed)) * D + ch;
                for (uint j = 0; j < EPL; ++j) vval[j] = float(v_tail[base + j]);
            }
            for (uint g = 0; g < QG; ++g) {
                const float next_max = max(run_max[g], score[g]);
                const float factor =
                    run_max[g] == -INFINITY ? 0.0f : fast::exp2(run_max[g] - next_max);
                const float weight = fast::exp2(score[g] - next_max);
                run_sum[g] = run_sum[g] * factor + weight;
                run_max[g] = next_max;
                for (uint j = 0; j < EPL; ++j) acc[g][j] = acc[g][j] * factor + weight * vval[j];
            }
        }
    }

    if (lane == 0) {
        for (uint g = 0; g < QG; ++g) {
            tg_max[sg * QG + g] = run_max[g];
            tg_sum[sg * QG + g] = run_sum[g];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint g = 0; g < QG; ++g) {
        float group_max = -INFINITY;
        for (uint s = 0; s < BN; ++s) group_max = max(group_max, tg_max[s * QG + g]);
        const float own = run_max[g] == -INFINITY ? 0.0f : fast::exp2(run_max[g] - group_max);
        for (uint j = 0; j < EPL; ++j) tg_acc[sg * D + ch + j] = acc[g][j] * own;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float group_sum = 0.0f;
            for (uint s = 0; s < BN; ++s) {
                const float m = tg_max[s * QG + g];
                group_sum += m == -INFINITY ? 0.0f : tg_sum[s * QG + g] * fast::exp2(m - group_max);
            }
            for (uint j = 0; j < EPL; ++j) {
                float lane_acc = 0.0f;
                for (uint s = 0; s < BN; ++s) lane_acc += tg_acc[s * D + ch + j];
                merged[g * EPL + j] = lane_acc;
            }
            out_max[g] = group_max;
            out_sum[g] = group_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return sg == 0;
}
"#;

/// Per-call kernel body: MLX regenerates and scans the body source on every dispatch, so the
/// attention core lives in [`HEADER`] and the body only binds buffers, shapes, and thread indices.
const ATTEND_BODY: &str = r#"
    constexpr int EPL = D / 32;
    threadgroup float tg_max[BN * QG];
    threadgroup float tg_sum[BN * QG];
    threadgroup float tg_acc[BN * D];
    float merged[QG * EPL];
    float group_max[QG];
    float group_sum[QG];
    uint qrow0;
    const uint splits = threadgroups_per_grid.x;
    const uint split = threadgroup_position_in_grid.x;
    const bool writer = sc20676_attend<D, QG, BN, MASK_MODE, WINDOW>(
        q, k_codes, k_scale, k_zero, k_tail, v_codes, v_scale, v_zero, v_tail, uint(params[0]),
        uint(params[1]), uint(params[2]), uint(q_shape[1]), uint(q_shape[2]), uint(v_codes_shape[1]), uint(k_codes_shape[2]),
        uint(k_tail_shape[2]), uint(v_codes_shape[2]), uint(v_tail_shape[2]), tg_max, tg_sum,
        tg_acc, thread_index_in_simdgroup, simdgroup_index_in_threadgroup, split, splits,
        threadgroup_position_in_grid.y, merged, group_max, group_sum, qrow0);
    if (!writer) return;
    const uint ch = thread_index_in_simdgroup * EPL;
    const uint SQ = uint(q_shape[2]);
    for (uint g = 0; g < QG; ++g) {
        const uint qrow = qrow0 + g * SQ;
        SC20676_EPILOGUE
    }
"#;

const SINGLE_EPILOGUE: &str = r#"
        using OutputT = typename sc20676_device_element<decltype(out)>::type;
        for (uint j = 0; j < EPL; ++j) {
            out[qrow * D + ch + j] = static_cast<OutputT>(
                group_sum[g] > 0.0f ? merged[g * EPL + j] / group_sum[g] : 0.0f);
        }
"#;

const PARTIAL_EPILOGUE: &str = r#"
        const uint slot = qrow * splits + split;
        for (uint j = 0; j < EPL; ++j) part_acc[slot * D + ch + j] = merged[g * EPL + j];
        if (thread_index_in_simdgroup == 0) {
            part_max[slot] = group_max[g];
            part_sum[slot] = group_sum[g];
        }
"#;

/// Second split-KV pass: one SIMD group per query row rescales and sums the split partials.
const REDUCE_BODY: &str = r#"
    constexpr int EPL = D / 32;
    const uint lane = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.y;
    const uint splits = uint(part_max_shape[1]);
    const uint ch = lane * EPL;
    // Lane-parallel maximum and normalizer over the splits; each split's weight is broadcast from
    // the lane that computed it, so the serial part is one shuffle and EPL FMAs per split.
    float lane_max = -INFINITY;
    for (uint s = lane; s < splits; s += 32) lane_max = max(lane_max, part_max[row * splits + s]);
    const float global_max = simd_max(lane_max);
    float lane_sum = 0.0f;
    for (uint s = lane; s < splits; s += 32) {
        const float split_max = part_max[row * splits + s];
        lane_sum += split_max == -INFINITY ? 0.0f
                                           : part_sum[row * splits + s] * fast::exp2(split_max - global_max);
    }
    const float global_sum = simd_sum(lane_sum);
    float merged[EPL];
    for (uint j = 0; j < EPL; ++j) merged[j] = 0.0f;
    for (uint base = 0; base < splits; base += 32) {
        const uint mine = base + lane;
        const float split_max = mine < splits ? part_max[row * splits + mine] : -INFINITY;
        const float weight = split_max == -INFINITY ? 0.0f : fast::exp2(split_max - global_max);
        const uint count = min(32u, splits - base);
        for (uint k = 0; k < count; ++k) {
            const float w = simd_shuffle(weight, k);
            const uint slot = row * splits + base + k;
            for (uint j = 0; j < EPL; ++j) merged[j] += part_acc[slot * D + ch + j] * w;
        }
    }
    using OutputT = typename sc20676_device_element<decltype(out)>::type;
    for (uint j = 0; j < EPL; ++j) {
        out[row * D + ch + j] = static_cast<OutputT>(global_sum > 0.0f ? merged[j] / global_sum : 0.0f);
    }
"#;

const ATTEND_INPUTS: [&str; 10] = [
    "q", "k_codes", "k_scale", "k_zero", "k_tail", "v_codes", "v_scale", "v_zero", "v_tail",
    "params",
];

/// Arguments of one fused packed-attention dispatch. Buffers may be larger than their live
/// extents (block preallocation); `key_packed_tokens`, `value_packed_tokens`, and `kv_tokens` are
/// the live extents. Tokens `key_packed_tokens..kv_tokens` of K are read from `key_tail` rows
/// `0..`, and likewise for V from `value_tail`.
#[derive(Clone, Copy, Debug)]
pub struct PackedAttentionArgs<'a> {
    /// `[B, Hq, Sq, D]`, f16/bf16/f32. Queries are the last `Sq` positions of the KV range.
    pub query: &'a Array,
    /// `[B, Hkv, key_group_capacity, G·D/4]` Uint8.
    pub key_codes: &'a Array,
    /// `[B, Hkv, key_group_capacity, D]` Float16 scales.
    pub key_scales: &'a Array,
    /// `[B, Hkv, key_group_capacity, D]` Float16 zeros.
    pub key_zeros: &'a Array,
    /// `[B, Hkv, key_residual_capacity, D]` dense residual keys, any float dtype.
    pub key_tail: &'a Array,
    /// `[B, Hkv, value_capacity, D/4]` Uint8.
    pub value_codes: &'a Array,
    /// `[B, Hkv, value_capacity, D/G]` Float16 scales.
    pub value_scales: &'a Array,
    /// `[B, Hkv, value_capacity, D/G]` Float16 zeros.
    pub value_zeros: &'a Array,
    /// `[B, Hkv, value_residual_capacity, D]` dense residual values, any float dtype.
    pub value_tail: &'a Array,
    /// Leading KV tokens held as quantized K groups (a multiple of the quantization group).
    pub key_packed_tokens: usize,
    /// Leading KV tokens held as quantized V rows.
    pub value_packed_tokens: usize,
    /// Live KV tokens.
    pub kv_tokens: usize,
    pub mask: PackedMask,
}

/// Retained kernel objects; MLX performs cold compilation on first `.run()` and reuses the same
/// compiled pipelines for subsequent dispatches.
pub struct PackedMetalKernel {
    single: MetalKernel,
    partial: MetalKernel,
    reduce: MetalKernel,
    identity: String,
    gpu_family: PackedMetalGpuFamily,
}

impl crate::primitives::packed_group_affine_kv::RetainedPackedKernel for PackedMetalKernel {
    fn cache_identity(&self) -> &str {
        &self.identity
    }

    fn backend(&self) -> &str {
        "mlx-metal"
    }

    fn retained_host_bytes_estimate(&self) -> usize {
        std::mem::size_of::<Self>()
    }

    fn dispatch(&self, args: &PackedAttentionArgs<'_>) -> Result<Array> {
        PackedMetalKernel::dispatch(self, args)
    }
}

impl std::fmt::Debug for PackedMetalKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedMetalKernel")
            .field("gpu_family", &self.gpu_family)
            .finish_non_exhaustive()
    }
}

/// Live-extent and shape contract of one dispatch, validated before any kernel is encoded.
struct ValidatedDispatch {
    /// Threadgroup rows: `B · (Hq / heads_per_row) · Sq`.
    rows: usize,
    /// Output rows: `B · Hq · Sq`.
    query_rows: usize,
    /// Query heads sharing one KV head that one threadgroup row serves.
    heads_per_row: usize,
    head_dimension: usize,
    visible_tokens: usize,
    mask_mode: i32,
    window: i32,
}

fn validate_dispatch(args: &PackedAttentionArgs<'_>) -> Result<ValidatedDispatch> {
    let group = PACKED_METAL_QUANT_GROUP_SIZE;
    let [batch, query_heads, query_tokens, head_dimension] =
        checked_shape(args.query.shape(), "query")?;
    let kc = checked_shape(args.key_codes.shape(), "key codes")?;
    let ks = checked_shape(args.key_scales.shape(), "key scales")?;
    let kz = checked_shape(args.key_zeros.shape(), "key zeros")?;
    let kt = checked_shape(args.key_tail.shape(), "key residual")?;
    let vc = checked_shape(args.value_codes.shape(), "value codes")?;
    let vs = checked_shape(args.value_scales.shape(), "value scales")?;
    let vz = checked_shape(args.value_zeros.shape(), "value zeros")?;
    let vt = checked_shape(args.value_tail.shape(), "value residual")?;
    let kv_heads = kc[1];
    let float = |dtype: Dtype| matches!(dtype, Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32);
    let key_words = group * head_dimension / PACKED_CODES_PER_BYTE;
    let value_words = head_dimension / PACKED_CODES_PER_BYTE;
    let value_groups = head_dimension / group;
    let kv = args.kv_tokens;
    let valid = batch != 0
        && query_heads != 0
        && query_tokens != 0
        && packed_metal_head_dimension_supported(head_dimension)
        && kv_heads != 0
        && query_heads % kv_heads == 0
        && kc == [batch, kv_heads, kc[2], key_words]
        && ks == [batch, kv_heads, kc[2], head_dimension]
        && kz == ks
        && kt == [batch, kv_heads, kt[2], head_dimension]
        && vc == [batch, kv_heads, vc[2], value_words]
        && vs == [batch, kv_heads, vc[2], value_groups]
        && vz == vs
        && vt == [batch, kv_heads, vt[2], head_dimension]
        && kc[2] != 0
        && kt[2] != 0
        && vc[2] != 0
        && vt[2] != 0
        && kv != 0
        && query_tokens <= kv
        && args.key_packed_tokens.is_multiple_of(group)
        && args.key_packed_tokens <= kv
        && args.key_packed_tokens / group <= kc[2]
        && kv - args.key_packed_tokens <= kt[2]
        && args.value_packed_tokens <= kv
        && args.value_packed_tokens <= vc[2]
        && kv - args.value_packed_tokens <= vt[2]
        && args.key_codes.dtype() == Dtype::Uint8
        && args.value_codes.dtype() == Dtype::Uint8
        && args.key_scales.dtype() == Dtype::Float16
        && args.key_zeros.dtype() == Dtype::Float16
        && args.value_scales.dtype() == Dtype::Float16
        && args.value_zeros.dtype() == Dtype::Float16
        && float(args.key_tail.dtype())
        && float(args.value_tail.dtype())
        && float(args.query.dtype());
    if !valid {
        return Err(Error::Unsupported(
            "SC-20676 packed buffers do not match query/cache geometry".into(),
        ));
    }
    let (mask_mode, window) = match args.mask {
        PackedMask::None => (0, 0),
        PackedMask::Causal => (1, 0),
        PackedMask::SlidingWindow(window) if window > 0 => (
            2,
            i32::try_from(window)
                .map_err(|_| Error::Unsupported("sliding window exceeds i32".into()))?,
        ),
        PackedMask::SlidingWindow(_) => {
            return Err(Error::Unsupported("empty sliding window".into()))
        }
        PackedMask::AdditiveUnsupported => {
            return Err(Error::Unsupported(
                "additive mask requires dense fallback".into(),
            ))
        }
    };
    checked_msl_i32(kv, "KV length")?;
    // Serve as many GQA query heads per row as keep `heads · D/32` accumulators in registers.
    let gqa = query_heads / kv_heads;
    let lane_values = head_dimension / SIMD_WIDTH;
    let heads_per_row = (1..=gqa)
        .rev()
        .find(|heads| gqa.is_multiple_of(*heads) && heads * lane_values <= MAX_LANE_ACCUMULATORS)
        .unwrap_or(1);
    let visible_tokens = match args.mask {
        PackedMask::SlidingWindow(window) => window.min(kv),
        _ => kv,
    };
    Ok(ValidatedDispatch {
        rows: batch * (query_heads / heads_per_row) * query_tokens,
        query_rows: batch * query_heads * query_tokens,
        heads_per_row,
        head_dimension,
        visible_tokens,
        mask_mode,
        window,
    })
}

impl PackedMetalKernel {
    pub fn new() -> Result<Self> {
        Self::for_identity("sc-20676-packed-group-affine-v1")
    }

    /// Construct the retained reader for one cache identity.  The identity is part of the
    /// compiled-handle binding, preventing a pipeline from being reused with another cache's
    /// layout or quantization contract.
    pub fn for_identity(identity: impl Into<String>) -> Result<Self> {
        Self::for_identity_and_family(identity, PackedMetalGpuFamily::ConservativeUnknownApple)
    }

    /// Bind a cache identity to an explicit GPU-family tuning profile. Callers may select the
    /// qualified recent-family profile only after their device probe; unknown devices retain the
    /// conservative one-SIMD-group geometry.
    pub fn for_identity_and_family(
        identity: impl Into<String>,
        gpu_family: PackedMetalGpuFamily,
    ) -> Result<Self> {
        let single = ATTEND_BODY.replace("SC20676_EPILOGUE", SINGLE_EPILOGUE);
        let partial = ATTEND_BODY.replace("SC20676_EPILOGUE", PARTIAL_EPILOGUE);
        Ok(Self {
            single: MetalKernel::with_options(
                "sc20676_attend",
                &ATTEND_INPUTS,
                &["out"],
                &single,
                HEADER,
                true,
                false,
            )?,
            partial: MetalKernel::with_options(
                "sc20676_split",
                &ATTEND_INPUTS,
                &["part_acc", "part_max", "part_sum"],
                &partial,
                HEADER,
                true,
                false,
            )?,
            reduce: MetalKernel::with_options(
                "sc20676_reduce",
                &["part_acc", "part_max", "part_sum"],
                &["out"],
                REDUCE_BODY,
                HEADER,
                true,
                false,
            )?,
            identity: identity.into(),
            gpu_family,
        })
    }

    pub fn gpu_family(&self) -> PackedMetalGpuFamily {
        self.gpu_family
    }

    pub fn tuning_profile(&self, head_dimension: usize) -> Option<PackedMetalTuningProfile> {
        self.gpu_family
            .tuning(head_dimension)
            .map(|tuning| PackedMetalTuningProfile {
                gpu_family: self.gpu_family.as_str(),
                threads: tuning.threads,
                simd_groups: tuning.simd_groups,
                values_per_thread: tuning.values_per_thread,
            })
    }

    /// KV splits the heuristic selects for these arguments (1 = single pass).
    pub fn planned_splits(&self, args: &PackedAttentionArgs<'_>) -> Result<usize> {
        let validated = validate_dispatch(args)?;
        let tuning = self.required_tuning(validated.head_dimension)?;
        Ok(packed_kv_split_count(
            validated.rows,
            validated.visible_tokens,
            tuning.simd_groups,
        ))
    }

    fn required_tuning(&self, head_dimension: usize) -> Result<PackedMetalTuning> {
        self.gpu_family.tuning(head_dimension).ok_or_else(|| {
            Error::Unsupported(
                "SC-20676 has no conservative tuning for this device/geometry".into(),
            )
        })
    }

    pub fn dispatch(&self, args: &PackedAttentionArgs<'_>) -> Result<Array> {
        self.dispatch_with_splits(args, None)
    }

    /// Dispatch with an explicit KV split count (`None` = the documented heuristic). Test and
    /// evidence seam for exercising both the single-pass and the split-KV kernels.
    pub fn dispatch_with_splits(
        &self,
        args: &PackedAttentionArgs<'_>,
        splits: Option<usize>,
    ) -> Result<Array> {
        let validated = validate_dispatch(args)?;
        let tuning = self.required_tuning(validated.head_dimension)?;
        let splits = splits
            .unwrap_or_else(|| {
                packed_kv_split_count(validated.rows, validated.visible_tokens, tuning.simd_groups)
            })
            .clamp(1, MAX_KV_SPLITS);
        let head_dimension = validated.head_dimension;
        let params = Array::from_slice(
            &[
                checked_msl_i32(args.key_packed_tokens, "packed key tokens")?,
                checked_msl_i32(args.value_packed_tokens, "packed value tokens")?,
                checked_msl_i32(args.kv_tokens, "KV tokens")?,
            ],
            &[3],
        );
        let threads = checked_msl_i32(tuning.threads, "thread-group width")?;
        let rows = checked_msl_i32(validated.rows, "threadgroup rows")?;
        let query_rows = checked_msl_i32(validated.query_rows, "query rows")?;
        let heads_per_row = checked_msl_i32(validated.heads_per_row, "heads per row")?;
        let split_threads = splits
            .checked_mul(tuning.threads)
            .ok_or_else(|| Error::Unsupported("SC-20676 Metal grid dimension overflow".into()))
            .and_then(|value| checked_msl_i32(value, "Metal grid width"))?;
        let head_i32 = checked_msl_i32(head_dimension, "head dimension")?;
        // Lane width (D/32), group size, and packed row widths are derived from `D` in the MSL.
        let bn = checked_msl_i32(tuning.simd_groups, "SIMD groups")?;
        let kernel = if splits == 1 {
            &self.single
        } else {
            &self.partial
        };
        let attend = kernel
            .apply()
            .input(args.query)
            .input(args.key_codes)
            .input(args.key_scales)
            .input(args.key_zeros)
            .input(args.key_tail)
            .input(args.value_codes)
            .input(args.value_scales)
            .input(args.value_zeros)
            .input(args.value_tail)
            .input(&params)
            .template_arg("D", head_i32)
            .template_arg("QG", heads_per_row)
            .template_arg("BN", bn)
            .template_arg("MASK_MODE", validated.mask_mode)
            .template_arg("WINDOW", validated.window);
        let query_shape = args.query.shape().to_vec();
        if splits == 1 {
            return attend
                .output(OutputArg {
                    shape: query_shape,
                    dtype: args.query.dtype(),
                })
                .grid(threads, rows, 1)
                .thread_group(threads, 1, 1)
                .run()?
                .into_iter()
                .next()
                .ok_or_else(|| Error::Msg("SC-20676 kernel returned no output".into()));
        }
        let splits_i32 = checked_msl_i32(splits, "KV splits")?;
        let mut partials = attend
            .output(OutputArg {
                shape: vec![query_rows, splits_i32, head_i32],
                dtype: Dtype::Float32,
            })
            .output(OutputArg {
                shape: vec![query_rows, splits_i32],
                dtype: Dtype::Float32,
            })
            .output(OutputArg {
                shape: vec![query_rows, splits_i32],
                dtype: Dtype::Float32,
            })
            .grid(split_threads, rows, 1)
            .thread_group(threads, 1, 1)
            .run()?
            .into_iter();
        let (Some(part_acc), Some(part_max), Some(part_sum)) =
            (partials.next(), partials.next(), partials.next())
        else {
            return Err(Error::Msg(
                "SC-20676 split kernel returned too few outputs".into(),
            ));
        };
        let simd = checked_msl_i32(SIMD_WIDTH, "SIMD width")?;
        self.reduce
            .apply()
            .input(&part_acc)
            .input(&part_max)
            .input(&part_sum)
            .output(OutputArg {
                shape: query_shape,
                dtype: args.query.dtype(),
            })
            .grid(simd, query_rows, 1)
            .thread_group(simd, 1, 1)
            .template_arg("D", head_i32)
            .run()?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Msg("SC-20676 reduce kernel returned no output".into()))
    }
}

/// Quantize completed K token groups per channel exactly as the CPU reference
/// (`TokenGroupKeyTensor::flush_pending_group`): f32 min/max, `scale = max((max−min)/3, ε)`, codes
/// `clamp(round((x−min)/scale), 0, 3)` with IEEE division and round-half-away-from-zero, and f16
/// scale/zero. The source is the virtual sequence `residual[0..p] ++ fresh`.
const QUANTIZE_KEYS_BODY: &str = r#"
    constexpr int G = 32;
    constexpr int KW = G * D / 4;
    const uint gid = thread_position_in_grid.x;
    const uint quads = uint(D) / 4u;
    const uint p = uint(params[0]);
    const uint groups = uint(params[1]);
    const uint BH = uint(fresh_shape[0]) * uint(fresh_shape[1]);
    const uint STEP = uint(fresh_shape[2]);
    const uint TCAP = uint(tail_shape[2]);
    const uint quad = gid % quads;
    const uint g = (gid / quads) % groups;
    const uint bh = gid / (quads * groups);
    if (bh >= BH) return;
    const uint c0 = quad * 4u;
    float lo[4];
    float hi[4];
    float sc[4];
    for (uint k = 0; k < 4u; ++k) {
        lo[k] = INFINITY;
        hi[k] = -INFINITY;
    }
    for (uint t = 0; t < uint(G); ++t) {
        const uint vt = g * uint(G) + t;
        for (uint k = 0; k < 4u; ++k) {
            const float x = vt < p ? float(tail[(bh * TCAP + vt) * D + c0 + k])
                                   : float(fresh[(bh * STEP + vt - p) * D + c0 + k]);
            lo[k] = fmin(lo[k], x);
            hi[k] = fmax(hi[k], x);
        }
    }
    for (uint k = 0; k < 4u; ++k) {
        sc[k] = fmax(precise::divide(hi[k] - lo[k], 3.0f), FLT_EPSILON);
        scales[(bh * groups + g) * D + c0 + k] = half(sc[k]);
        zeros[(bh * groups + g) * D + c0 + k] = half(lo[k]);
    }
    for (uint t = 0; t < uint(G); ++t) {
        const uint vt = g * uint(G) + t;
        uint byte = 0;
        for (uint k = 0; k < 4u; ++k) {
            const float x = vt < p ? float(tail[(bh * TCAP + vt) * D + c0 + k])
                                   : float(fresh[(bh * STEP + vt - p) * D + c0 + k]);
            const uint code = uint(clamp(round(precise::divide(x - lo[k], sc[k])), 0.0f, 3.0f));
            byte |= code << (2u * k);
        }
        codes[(bh * groups + g) * KW + (t * uint(D) + c0) / 4u] = uint8_t(byte);
    }
"#;

/// Quantize V rows per token over 32-channel groups exactly as the CPU reference
/// (`PackedTensor::append`). The source is the virtual sequence `residual[0..p] ++ fresh`.
const QUANTIZE_VALUES_BODY: &str = r#"
    constexpr int G = 32;
    constexpr int VW = D / 4;
    constexpr int VG = D / G;
    const uint gid = thread_position_in_grid.x;
    const uint p = uint(params[0]);
    const uint rows = uint(params[1]);
    const uint BH = uint(fresh_shape[0]) * uint(fresh_shape[1]);
    const uint STEP = uint(fresh_shape[2]);
    const uint TCAP = uint(tail_shape[2]);
    const uint grp = gid % uint(VG);
    const uint r = (gid / uint(VG)) % rows;
    const uint bh = gid / (uint(VG) * rows);
    if (bh >= BH) return;
    const uint c0 = grp * uint(G);
    float lo = INFINITY;
    float hi = -INFINITY;
    for (uint c = 0; c < uint(G); ++c) {
        const float x = r < p ? float(tail[(bh * TCAP + r) * D + c0 + c])
                              : float(fresh[(bh * STEP + r - p) * D + c0 + c]);
        lo = fmin(lo, x);
        hi = fmax(hi, x);
    }
    const float sc = fmax(precise::divide(hi - lo, 3.0f), FLT_EPSILON);
    scales[(bh * rows + r) * VG + grp] = half(sc);
    zeros[(bh * rows + r) * VG + grp] = half(lo);
    for (uint word = 0; word < uint(G) / 4u; ++word) {
        uint byte = 0;
        for (uint k = 0; k < 4u; ++k) {
            const uint c = c0 + word * 4u + k;
            const float x = r < p ? float(tail[(bh * TCAP + r) * D + c])
                                  : float(fresh[(bh * STEP + r - p) * D + c]);
            const uint code = uint(clamp(round(precise::divide(x - lo, sc)), 0.0f, 3.0f));
            byte |= code << (2u * k);
        }
        codes[(bh * rows + r) * VW + c0 / 4u + word] = uint8_t(byte);
    }
"#;

thread_local! {
    static QUANTIZERS: std::cell::OnceCell<(MetalKernel, MetalKernel)> =
        const { std::cell::OnceCell::new() };
}

fn with_quantizers<T>(f: impl FnOnce(&MetalKernel, &MetalKernel) -> Result<T>) -> Result<T> {
    QUANTIZERS.with(|cell| {
        if cell.get().is_none() {
            let keys = MetalKernel::with_options(
                "sc20676_quant_k",
                &["tail", "fresh", "params"],
                &["codes", "scales", "zeros"],
                QUANTIZE_KEYS_BODY,
                HEADER,
                true,
                false,
            )?;
            let values = MetalKernel::with_options(
                "sc20676_quant_v",
                &["tail", "fresh", "params"],
                &["codes", "scales", "zeros"],
                QUANTIZE_VALUES_BODY,
                HEADER,
                true,
                false,
            )?;
            let _ = cell.set((keys, values));
        }
        let (keys, values) = cell
            .get()
            .ok_or_else(|| Error::Msg("SC-20676 quantizer initialization failed".into()))?;
        f(keys, values)
    })
}

/// Packed group-affine quantization of the first `groups · G` tokens of the virtual sequence
/// `residual[.., .., 0..residual_rows, ..] ++ fresh`, on the GPU and lazily. Returns
/// `[key_codes [B,H,groups,G·D/4], key_scales, key_zeros [B,H,groups,D],
///   value_codes [B,H,groups·G,D/4], value_scales, value_zeros [B,H,groups·G,D/G]]`.
pub(crate) fn quantize_group_affine_flush(
    key_residual: &Array,
    value_residual: &Array,
    residual_rows: usize,
    fresh_keys: &Array,
    fresh_values: &Array,
    groups: usize,
) -> Result<[Array; 6]> {
    let [batch, heads, _, head_dimension] = checked_shape(fresh_keys.shape(), "fresh keys")?;
    if groups == 0 || !packed_metal_head_dimension_supported(head_dimension) {
        return Err(Error::Unsupported(
            "SC-20676 quantization flush geometry".into(),
        ));
    }
    let group = PACKED_METAL_QUANT_GROUP_SIZE;
    let rows = groups * group;
    let residual = checked_msl_i32(residual_rows, "residual rows")?;
    let key_params = Array::from_slice(&[residual, checked_msl_i32(groups, "groups")?], &[2]);
    let value_params = Array::from_slice(&[residual, checked_msl_i32(rows, "rows")?], &[2]);
    let (b, h, g, d, r) = (
        checked_msl_i32(batch, "batch")?,
        checked_msl_i32(heads, "heads")?,
        checked_msl_i32(groups, "groups")?,
        checked_msl_i32(head_dimension, "head dimension")?,
        checked_msl_i32(rows, "flushed rows")?,
    );
    let key_words = checked_msl_i32(group * head_dimension / PACKED_CODES_PER_BYTE, "key words")?;
    let value_words = checked_msl_i32(head_dimension / PACKED_CODES_PER_BYTE, "value words")?;
    let value_groups = checked_msl_i32(head_dimension / group, "value groups")?;
    let key_threads = checked_msl_i32(
        batch * heads * groups * (head_dimension / PACKED_CODES_PER_BYTE),
        "key quantizer grid",
    )?;
    let value_threads = checked_msl_i32(
        batch * heads * rows * (head_dimension / group),
        "value quantizer grid",
    )?;
    let triple = |shape_codes: Vec<i32>, shape_metadata: Vec<i32>| {
        [
            OutputArg {
                shape: shape_codes,
                dtype: Dtype::Uint8,
            },
            OutputArg {
                shape: shape_metadata.clone(),
                dtype: Dtype::Float16,
            },
            OutputArg {
                shape: shape_metadata,
                dtype: Dtype::Float16,
            },
        ]
    };
    with_quantizers(|keys, values| {
        let [kc, ks, kz] = triple(vec![b, h, g, key_words], vec![b, h, g, d]);
        let key_out = keys
            .apply()
            .input(key_residual)
            .input(fresh_keys)
            .input(&key_params)
            .output(kc)
            .output(ks)
            .output(kz)
            .grid(key_threads, 1, 1)
            .thread_group(key_threads.min(256), 1, 1)
            .template_arg("D", d)
            .run()?;
        let [vc, vs, vz] = triple(vec![b, h, r, value_words], vec![b, h, r, value_groups]);
        let value_out = values
            .apply()
            .input(value_residual)
            .input(fresh_values)
            .input(&value_params)
            .output(vc)
            .output(vs)
            .output(vz)
            .grid(value_threads, 1, 1)
            .thread_group(value_threads.min(256), 1, 1)
            .template_arg("D", d)
            .run()?;
        let mut all = key_out.into_iter().chain(value_out);
        let mut next = || {
            all.next()
                .ok_or_else(|| Error::Msg("SC-20676 quantizer returned too few outputs".into()))
        };
        Ok([next()?, next()?, next()?, next()?, next()?, next()?])
    })
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use mlx_rs::Dtype;

    fn half(values: Vec<f32>, shape: &[i32]) -> Array {
        Array::from_slice(&values, shape)
            .as_dtype(Dtype::Float16)
            .unwrap()
    }

    #[test]
    fn gpu_family_tuning_is_conservative_and_geometry_explicit() {
        for dimension in [64, 128, 256] {
            let conservative = PackedMetalGpuFamily::ConservativeUnknownApple
                .tuning(dimension)
                .unwrap();
            assert_eq!(conservative.threads, 32);
            assert_eq!(conservative.simd_groups, 1);
            assert_eq!(conservative.values_per_thread, dimension / 32);

            let recent = PackedMetalGpuFamily::Apple7OrNewer
                .tuning(dimension)
                .unwrap();
            assert_eq!(recent.threads, 256);
            assert_eq!(recent.simd_groups, 8);
            assert_eq!(recent.values_per_thread, dimension / 32);
        }
        assert!(PackedMetalGpuFamily::ConservativeUnknownApple
            .tuning(96)
            .is_none());
        assert_eq!(
            PackedMetalGpuFamily::ConservativeUnknownApple.as_str(),
            "conservative-unknown-apple"
        );
        assert_eq!(
            PackedMetalGpuFamily::Apple7OrNewer.as_str(),
            "apple7-or-newer"
        );
    }

    #[test]
    fn split_heuristic_splits_long_decode_and_keeps_prefill_single_pass() {
        // Short history: never split below the per-split amortization floor.
        assert_eq!(packed_kv_split_count(8, 64, 1), 1);
        assert_eq!(packed_kv_split_count(8, 32 * 8, 1), 2);
        assert_eq!(packed_kv_split_count(8, 32 * 32, 8), 2);
        // Llama-3.2-3B / Qwen3-1.7B decode rows over long histories split for occupancy.
        assert_eq!(packed_kv_split_count(8, 10_240, 1), 80);
        assert_eq!(packed_kv_split_count(8, 32_768, 1), MAX_KV_SPLITS);
        assert_eq!(packed_kv_split_count(8, 32_768, 8), 64);
        // A prefill chunk already has enough rows to occupy the GPU.
        assert_eq!(packed_kv_split_count(8 * 512, 32_768, 1), 1);
        assert!(packed_kv_split_count(1, usize::MAX / 2, 8) <= MAX_KV_SPLITS);
    }

    #[test]
    fn sliding_window_larger_than_msl_i32_fails_before_dispatch() {
        let q = Array::from_slice(&[0.0f32; 64], &[1, 1, 1, 64]);
        let key_codes = Array::from_slice(&[0u8; 512], &[1, 1, 1, 512]);
        let key_scale = half(vec![0.0; 64], &[1, 1, 1, 64]);
        let key_zero = half(vec![0.0; 64], &[1, 1, 1, 64]);
        let key_tail = Array::from_slice(&[0.0f32; 64], &[1, 1, 1, 64]);
        let value_codes = Array::from_slice(&[0u8; 16], &[1, 1, 1, 16]);
        let value_scale = half(vec![0.0; 2], &[1, 1, 1, 2]);
        let value_zero = half(vec![0.0; 2], &[1, 1, 1, 2]);
        let value_tail = Array::from_slice(&[0.0f32; 64], &[1, 1, 1, 64]);
        let error = PackedMetalKernel::new()
            .unwrap()
            .dispatch(&PackedAttentionArgs {
                query: &q,
                key_codes: &key_codes,
                key_scales: &key_scale,
                key_zeros: &key_zero,
                key_tail: &key_tail,
                value_codes: &value_codes,
                value_scales: &value_scale,
                value_zeros: &value_zero,
                value_tail: &value_tail,
                key_packed_tokens: 0,
                value_packed_tokens: 1,
                kv_tokens: 1,
                mask: PackedMask::SlidingWindow(usize::MAX),
            })
            .unwrap_err();
        assert!(error.to_string().contains("sliding window exceeds i32"));
    }
}
