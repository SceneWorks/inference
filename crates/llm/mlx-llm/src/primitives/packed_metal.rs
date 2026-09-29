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
//!
//! On the qualified [`PackedMetalGpuFamily::Apple7OrNewer`] profile, multi-row steps (chunked
//! prefill after the first chunk, speculative verify) use a tiled flash-attention kernel instead
//! once `S_q` reaches [`packed_tiled_min_query_tokens`]: the per-row
//! kernel re-decodes the whole compressed history for every query row. A tiled threadgroup owns 32
//! query rows (query position × GQA head sharing one KV head) and streams the visible KV range in
//! blocks; each packed K/V block is dequantized once into block-sized threadgroup tiles and reused by
//! every row of the tile through `simdgroup_matrix` products, with the online softmax per row in
//! registers. Long ranges may be split across threadgroups and merged by the same reduction pass.
//!
//! Where MLX itself runs its Neural-Accelerator kernels ([`mlx_nax_available`]), multi-row steps
//! with bf16/f16 queries at D = 64/128 run the NAX tiled kernel instead: the same block-shared
//! dequantization, but into 16-bit tiles, with `Q·Kᵀ` and `P·V` on the matrix unit through MLX's
//! `steel_attention_nax` fragment products (`mpp::tensor_ops::matmul2d`). f32 queries and D = 256
//! keep the fp32 tiled kernel; the choice and its reason are reported by
//! [`PackedMetalKernel::nax_selection`] and [`PackedMetalKernel::kernel_descriptor`].
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

/// Explicit GPU-family tuning boundary. Unknown Apple GPUs run only the per-row kernel with one SIMD
/// group per threadgroup and rely on split-KV threadgroups for parallelism; qualified recent
/// families use eight cooperating SIMD groups per threadgroup for the per-row kernel and select the
/// tiled multi-row kernel (four SIMD groups, `simdgroup_matrix`) for multi-row steps. No family
/// outside this enum is silently assigned a geometry.
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

/// SIMD groups per tiled threadgroup; each owns eight query rows (one 8×8 fragment row block).
const TILED_SIMD_GROUPS: usize = 4;
/// Query rows (position × GQA head) one tiled threadgroup serves.
const TILED_ROWS: usize = TILED_SIMD_GROUPS * 8;
/// KV tokens per tiled block: the fp32 K and V tiles together stay near 16 KiB of threadgroup
/// memory (`2 · BK · (D + 4) · 4` bytes), i.e. 32/16/8 tokens for D = 64/128/256.
const fn tiled_block_tokens(head_dimension: usize) -> usize {
    2048 / head_dimension
}
/// Resident tiled threadgroups a split dispatch aims for (four SIMD groups each).
const TILED_TARGET_THREADGROUPS: usize = 512;
/// A tiled KV split owns at least this many visible tokens, so the extra partial write and the
/// reduction stay amortized.
const TILED_MIN_TOKENS_PER_SPLIT: usize = 2048;

/// Query tokens from which [`PackedMetalKernel::dispatch`] selects the tiled multi-row kernel on the
/// qualified family; fewer rows keep the per-row split-KV kernel. Chosen per head dimension from the
/// SC-20676 synthetic crossover sweep (`tiled_threshold_crossover_sweep`, M5 Max, bf16, Hq = 24,
/// Hkv = 8, per-row kernel on the Apple7OrNewer profile), median ms per-row / tiled at `S_q = 16`:
///
/// * D = 64: 1.24 / 1.26 over 4k, 2.28 / 0.97 over 32k → 16.
/// * D = 128: 0.64 / 0.73 over 4k, 2.43 / 1.34 over 32k → 16.
/// * D = 256 (8-token blocks, 64 fp32 fragments per lane): 0.67 / 1.63 over 4k and 4.08 / 3.49
///   over 32k at 16, but 1.23 / 1.65 and 7.85 / 5.01 at 32 → 32.
///
/// The NAX tiled kernel (bf16/f16 at D = 64/128) is past its crossover at the same threshold
/// (release build, median ms per-row / NAX at `S_q = 16`): D = 64 0.33 / 0.22 over 4k and
/// 1.68 / 0.36 over 32k; D = 128 0.40 / 0.31 and 2.08 / 0.63.
pub const fn packed_tiled_min_query_tokens(head_dimension: usize) -> usize {
    if head_dimension >= 256 {
        32
    } else {
        16
    }
}

/// SIMD groups per NAX threadgroup; each owns sixteen query rows (one 16-row NAX fragment, as in
/// MLX's `steel_attention_nax`). Eight rather than MLX's four: each dequantized block is shared by
/// 128 rows instead of 64. SC-20676 synthetic chunked-prefill sweep (M5 Max, bf16, Hq = 24,
/// Hkv = 8, D = 128, 2048-row chunks, all chunks evaluated together), NAX ms / dense SDPA ms:
/// 4 groups 10.7 / 8.4 over 8k and 165.6 / 141.6 over 32k; 8 groups 9.9 / 8.3 and 160.5 / 150.0;
/// 16 groups 11.5 / 8.4 and 176.4 / 157.7.
const NAX_SIMD_GROUPS: usize = 8;
/// Query rows (position × GQA head) one NAX threadgroup serves.
const NAX_ROWS: usize = NAX_SIMD_GROUPS * 16;
/// KV tokens per NAX block: one packed K token group, MLX's `bk = 32`. The bf16/f16 K and V tiles
/// take `2 · 32 · (D + 8) · 2` bytes of threadgroup memory (9 KiB at D = 64, 17 KiB at D = 128).
const NAX_BLOCK_TOKENS: usize = 32;

/// Whether MLX's own Metal backend runs its Neural-Accelerator kernels in this process: the linked,
/// pinned MLX's `mlx::core::metal::is_nax_available()`. It is false when MLX was built with
/// `MLX_METAL_NO_NAX` (Metal < 4.0, SDK or deployment target below macOS 26.2), when the running
/// OS is older than macOS 26.2, or when the GPU architecture generation — parsed from
/// `MLX_METAL_GPU_ARCH` or `MTLDevice.architecture.name` — is below 17 (18 for phone GPUs). The
/// packed reader calls the predicate instead of re-deriving it, so it can never disagree with
/// MLX's dense `steel_attention_nax` selection (which additionally requires 16-bit inputs unless
/// `MLX_ENABLE_TF32`; see [`PackedNaxSelection`]).
#[cfg(target_os = "macos")]
pub fn mlx_nax_available() -> bool {
    // MLX keeps this predicate in its C++ Metal backend (no mlx-c binding); the static library is
    // the exact pinned build every MLX dispatch in this process uses. `bool f()` has the same
    // AArch64 calling convention in C and C++.
    unsafe extern "C" {
        #[link_name = "_ZN3mlx4core5metal16is_nax_availableEv"]
        fn mlx_metal_is_nax_available() -> bool;
    }
    // SAFETY: a no-argument predicate; MLX caches its result in a function-local static.
    unsafe { mlx_metal_is_nax_available() }
}

/// Non-macOS builds have no MLX Metal backend, hence no Neural Accelerator.
#[cfg(not(target_os = "macos"))]
pub fn mlx_nax_available() -> bool {
    false
}

/// Head dimensions MLX instantiates `steel_attention_nax` for (`bd` 64 and 128).
pub const fn packed_nax_head_dimension_supported(head_dimension: usize) -> bool {
    matches!(head_dimension, 64 | 128)
}

/// The NAX decision for one multi-row step and why, recorded in [`PackedKernelDescriptor`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedNaxSelection {
    pub selected: bool,
    pub reason: &'static str,
}

const NAX_SELECTED: &str = "mlx::core::metal::is_nax_available() and 16-bit queries at D 64/128";
const NAX_CONSERVATIVE: &str = "conservative GPU family never runs the tiled kernels";
const NAX_UNAVAILABLE: &str = "mlx::core::metal::is_nax_available() is false (MLX built without \
     NAX, macOS < 26.2, or GPU generation < 17)";
const NAX_F32_QUERY: &str = "f32 queries keep the fp32 tiled kernel: NAX products are 16-bit (MLX \
     runs f32 on NAX only as TF32, which the f32 1e-4 contract excludes)";
const NAX_HEAD_DIMENSION: &str = "MLX instantiates NAX attention for D 64/128 only";

/// Static geometry of one packed kernel path, for evidence and comparison receipts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedKernelDescriptor {
    pub kernel: &'static str,
    /// Why this kernel was selected over the NAX tiled kernel (or that NAX was selected).
    pub selection: &'static str,
    pub threads: usize,
    pub simd_groups: usize,
    /// KV tokens one barrier-delimited step covers.
    pub kv_block_tokens: usize,
    /// Threadgroup barriers per KV block in the streaming loop (the per-row kernel has none; its
    /// only barriers are in the once-per-threadgroup merge).
    pub threadgroup_barriers_per_kv_block: usize,
    pub softmax_state: &'static str,
}

/// KV splits for the tiled kernel: enough threadgroups for occupancy, never fewer than
/// 2048 visible tokens per split, at most 128 splits.
pub fn packed_tiled_split_count(threadgroups: usize, visible_tokens: usize) -> usize {
    let for_occupancy = TILED_TARGET_THREADGROUPS.div_ceil(threadgroups.max(1));
    let for_amortization = visible_tokens / TILED_MIN_TOKENS_PER_SPLIT;
    for_occupancy.min(for_amortization).clamp(1, MAX_KV_SPLITS)
}

/// Kernel a dispatch selects (see [`PackedMetalKernel::planned_path`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedKernelPath {
    /// One threadgroup per (row of GQA heads, KV split); decode and short verify steps.
    PerRow { splits: usize },
    /// Tiled multi-row flash attention with block-shared dequantized K/V.
    Tiled { splits: usize },
    /// The tiled mechanism with 16-bit K/V tiles and Neural-Accelerator matrix products.
    NaxTiled { splits: usize },
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

/// Tiled multi-row core. A threadgroup of `WM` SIMD groups owns `8·WM` query rows of one KV head:
/// row `m` is query position `m / gqa` and GQA head `m % gqa`, so every query head sharing the KV
/// head reuses the same dequantized block. Per KV block of `BK` tokens the threadgroup dequantizes
/// packed K and V once into fp32 tiles (tail tokens come from the dense residual inputs, tokens past
/// the live extent are zero), then each SIMD group computes its 8×BK score fragment with
/// `simdgroup_matrix` products, masks it per row, updates the online softmax in registers (the four
/// lanes sharing a fragment row reduce with two shuffles), rescales its 8×D accumulator, and adds
/// `P·V`. Blocks outside the tile's causal/sliding range are never visited, and blocks visible to
/// every row of the tile skip the per-element mask. Scores are in the log2 domain like the per-row
/// kernel, so split partials use the shared reduction pass.
const TILED_HEADER: &str = r#"
// Row and first column of this lane's two elements of an 8×8 simdgroup matrix: the Apple GPU
// fragment layout MLX's steel GEMM and attention kernels are built on.
inline ushort2 sc20676_frag(uint lane) {
    const uint qid = lane / 4u;
    return ushort2((qid & 4u) + ((lane / 2u) % 4u), (qid & 2u) * 2u + (lane % 2u) * 2u);
}

template <typename T>
inline float4 sc20676_load4(const device T* src) {
    return float4(float(src[0]), float(src[1]), float(src[2]), float(src[3]));
}

inline float4 sc20676_codes4(uint byte) {
    return float4(float(byte & 3u), float((byte >> 2) & 3u), float((byte >> 4) & 3u),
                  float((byte >> 6) & 3u));
}

template <int D, int BK, int WM, int MASK_MODE, int WINDOW, typename QT, typename KT, typename VT>
inline void sc20676_tiled(
    const device QT* q, const device uint8_t* k_codes, const device half* k_scale,
    const device half* k_zero, const device KT* k_tail, const device uint8_t* v_codes,
    const device half* v_scale, const device half* v_zero, const device VT* v_tail,
    uint k_packed, uint v_packed, uint kv_len, uint HQ, uint SQ, uint HKV, uint KG_CAP,
    uint KT_CAP, uint V_CAP, uint VT_CAP, threadgroup float* k_tile, threadgroup float* v_tile,
    uint tid, uint lane, uint sg, uint tile, uint kv_row, uint split, uint splits,
    thread float* o, thread float& row_max, thread float& row_sum, thread uint& out_row,
    thread bool& valid) {
    constexpr uint G = 32;
    constexpr uint KW = G * D / 4;
    constexpr uint VW = D / 4;
    constexpr uint VG = D / G;
    constexpr uint LD = D + 4;
    constexpr int DT = D / 8;
    constexpr int KT8 = BK / 8;
    constexpr uint BQ = 8 * WM;
    constexpr uint QUADS = BK * D / 4;
    const uint gqa = HQ / HKV;
    const uint rows = SQ * gqa;
    const uint b = kv_row / HKV;
    const uint kvh = kv_row % HKV;
    const ushort2 frag = sc20676_frag(lane);
    const uint m = tile * BQ + sg * 8 + frag.x;
    valid = m < rows;
    const uint mc = min(m, rows - 1);
    const uint qi = mc / gqa;
    out_row = (b * HQ + kvh * gqa + mc % gqa) * SQ + qi;
    const uint qoff = kv_len - SQ;
    const uint hi = MASK_MODE == 0 ? kv_len : qoff + qi + 1;
    const uint lo = (MASK_MODE == 2 && hi > uint(WINDOW)) ? hi - uint(WINDOW) : 0;
    // Tile-wide bounds: the union of the rows' ranges is visited; the intersection needs no mask.
    const uint first_qi = (tile * BQ) / gqa;
    const uint last_qi = (min(tile * BQ + BQ, rows) - 1) / gqa;
    const uint tile_hi = MASK_MODE == 0 ? kv_len : qoff + last_qi + 1;
    const uint first_hi = MASK_MODE == 0 ? kv_len : qoff + first_qi + 1;
    const uint tile_lo = (MASK_MODE == 2 && first_hi > uint(WINDOW)) ? first_hi - uint(WINDOW) : 0;
    const uint full_hi = first_hi;
    const uint full_lo = (MASK_MODE == 2 && tile_hi > uint(WINDOW)) ? tile_hi - uint(WINDOW) : 0;
    const uint first_block = tile_lo / BK;
    const uint end_block = (tile_hi + BK - 1) / BK;
    const uint per_split = (end_block - first_block + splits - 1) / splits;
    const uint block_begin = min(end_block, first_block + split * per_split);
    const uint block_end = min(end_block, block_begin + per_split);

    const float q_scale = rsqrt(float(D)) * M_LOG2E_F;
    const device QT* q_row = q + out_row * D + frag.y;
    simdgroup_float8x8 qm[DT];
    simdgroup_float8x8 om[DT];
#pragma clang loop unroll(full)
    for (int d = 0; d < DT; ++d) {
        qm[d].thread_elements()[0] = float(q_row[d * 8]) * q_scale;
        qm[d].thread_elements()[1] = float(q_row[d * 8 + 1]) * q_scale;
        om[d] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float run_max = -INFINITY;
    float run_sum = 0.0f;

    for (uint blk = block_begin; blk < block_end; ++blk) {
        const uint t0 = blk * BK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < QUADS; idx += 32 * WM) {
            const uint tt = idx / (D / 4);
            const uint c0 = (idx % (D / 4)) * 4;
            const uint t = t0 + tt;
            float4 kv = float4(0.0f);
            float4 vv = float4(0.0f);
            if (t < kv_len) {
                if (t < k_packed) {
                    const uint grp = kv_row * KG_CAP + t / G;
                    const float4 codes = sc20676_codes4(k_codes[grp * KW + ((t % G) * D + c0) / 4]);
                    kv = sc20676_load4(k_zero + grp * D + c0)
                        + sc20676_load4(k_scale + grp * D + c0) * codes;
                } else {
                    kv = sc20676_load4(k_tail + (kv_row * KT_CAP + (t - k_packed)) * D + c0);
                }
                if (t < v_packed) {
                    const uint vrow = kv_row * V_CAP + t;
                    const uint meta = vrow * VG + c0 / G;
                    vv = float(v_zero[meta])
                        + float(v_scale[meta]) * sc20676_codes4(v_codes[vrow * VW + c0 / 4]);
                } else {
                    vv = sc20676_load4(v_tail + (kv_row * VT_CAP + (t - v_packed)) * D + c0);
                }
            }
            *(threadgroup float4*)(k_tile + tt * LD + c0) = kv;
            *(threadgroup float4*)(v_tile + tt * LD + c0) = vv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 sm[KT8];
#pragma clang loop unroll(full)
        for (int j = 0; j < KT8; ++j) sm[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#pragma clang loop unroll(full)
        for (int d = 0; d < DT; ++d) {
#pragma clang loop unroll(full)
            for (int j = 0; j < KT8; ++j) {
                simdgroup_float8x8 kt;
                simdgroup_load(kt, k_tile + j * 8 * LD + d * 8, LD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sm[j], qm[d], kt, sm[j]);
            }
        }
        const bool full = t0 >= full_lo && t0 + BK <= full_hi;
        float block_max = -INFINITY;
#pragma clang loop unroll(full)
        for (int j = 0; j < KT8; ++j) {
#pragma clang loop unroll(full)
            for (int e = 0; e < 2; ++e) {
                const uint t = t0 + j * 8 + frag.y + e;
                float s = sm[j].thread_elements()[e];
                if (!full && (t >= hi || t < lo)) s = -INFINITY;
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
#pragma clang loop unroll(full)
    for (int d = 0; d < DT; ++d) {
        o[2 * d] = om[d].thread_elements()[0];
        o[2 * d + 1] = om[d].thread_elements()[1];
    }
    row_max = run_max;
    row_sum = run_sum;
}
"#;

/// Per-call tiled body: grid `x` = query-row tiles, `y` = batch × KV head, `z` = KV splits.
const TILED_BODY: &str = r#"
    threadgroup float k_tile[BK * (D + 4)];
    threadgroup float v_tile[BK * (D + 4)];
    float o[D / 4];
    float row_max;
    float row_sum;
    uint out_row;
    bool valid;
    const uint split = threadgroup_position_in_grid.z;
    const uint splits = threadgroups_per_grid.z;
    sc20676_tiled<D, BK, WM, MASK_MODE, WINDOW>(
        q, k_codes, k_scale, k_zero, k_tail, v_codes, v_scale, v_zero, v_tail, uint(params[0]),
        uint(params[1]), uint(params[2]), uint(q_shape[1]), uint(q_shape[2]), uint(v_codes_shape[1]),
        uint(k_codes_shape[2]), uint(k_tail_shape[2]), uint(v_codes_shape[2]), uint(v_tail_shape[2]),
        k_tile, v_tile, thread_index_in_threadgroup, thread_index_in_simdgroup,
        simdgroup_index_in_threadgroup, threadgroup_position_in_grid.x,
        threadgroup_position_in_grid.y, split, splits, o, row_max, row_sum, out_row, valid);
    if (!valid) return;
    const uint col = sc20676_frag(thread_index_in_simdgroup).y;
    for (uint d = 0; d < uint(D / 8); ++d) {
        for (uint e = 0; e < 2u; ++e) {
            SC20676_TILED_EPILOGUE
        }
    }
"#;

const TILED_SINGLE_EPILOGUE: &str = r#"
            using OutputT = typename sc20676_device_element<decltype(out)>::type;
            out[out_row * D + d * 8 + col + e] =
                static_cast<OutputT>(row_sum > 0.0f ? o[2 * d + e] / row_sum : 0.0f);
"#;

// One lane per fragment row (column 0) writes the row's maximum and normalizer.
const TILED_PARTIAL_EPILOGUE: &str = r#"
            const uint slot = out_row * splits + split;
            part_acc[slot * D + d * 8 + col + e] = o[2 * d + e];
            if (col == 0 && e == 0 && d == 0) {
                part_max[slot] = row_max;
                part_sum[slot] = row_sum;
            }
"#;

/// Neural-Accelerator (NAX) tiled core: the tiled kernel's mechanism with MLX's
/// `steel_attention_nax` matrix products (`mpp::tensor_ops::matmul2d`, 16×32×16 per SIMD group, the
/// fragment layout of MLX's `BaseNAXFrag`). A threadgroup of `WM` SIMD groups owns `16·WM` query
/// rows of one KV head (row `m` = query position `m / gqa`, GQA head `m % gqa`); each lane holds rows
/// `fm` and `fm + 8` of its SIMD group's 16-row fragment and columns `fn..fn+3`. Per KV block of
/// `BK` tokens the threadgroup dequantizes packed K and V once into `T` (bf16/f16, the query dtype)
/// threadgroup tiles (tail tokens from the dense residual inputs, tokens past the live extent
/// zero); every SIMD group then runs `S = Q·Kᵀ` and `O += P·V` on the accelerator with fp32
/// accumulation, and keeps the online softmax (log2 domain, fp32) in registers. Blocks outside the
/// tile's causal/sliding range are never visited, blocks visible to every row skip the
/// per-element mask, and split partials use the shared reduction pass.
///
/// Every helper is force-inlined and every fixed-trip loop fully unrolled (MLX's `METAL_FUNC` and
/// `STEEL_PRAGMA_UNROLL`): one dynamically indexed fragment array (e.g. the output accumulators in
/// a rolled epilogue loop) moves it from registers to stack memory for the whole kernel, which made
/// a verbatim port of MLX's dense NAX loop three times slower than MLX's own build of it.
const NAX_HEADER: &str = r#"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

// MLX BaseNAXFrag::get_coord: (column, row) of this lane's first element of a 16×16 fragment; the
// lane holds rows `fm` and `fm + 8`, columns `fn..fn+3` of each.
inline __attribute__((always_inline)) short2 sc20676_nax_coord(uint lane) {
    const short qid = short(lane >> 2);
    const short fm = (qid & 4) | short((lane >> 1) & 3u);
    const short fn = ((qid & 2) | short(lane & 1u)) * 4;
    return short2(fn, fm);
}

// C[16×32] += A[16×16] · B[16×32] on the matrix unit, B given as two 16×16 fragments (rows of `Bᵀ`
// when `TB`), exactly MLX BaseNAXFrag::mma's first form.
template <bool TB, typename AT, typename BT>
inline __attribute__((always_inline)) void sc20676_nax_mma(thread vec<float, 8>& c0, thread vec<float, 8>& c1,
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

// Rows fm, fm+8 × columns fn..fn+3 of the 16×16 fragment at (`row0`, `col0`) of a threadgroup tile.
template <typename T, int LD>
inline __attribute__((always_inline)) vec<T, 8> sc20676_nax_tile_frag(const threadgroup T* tile, short2 coord, uint row0, uint col0) {
    const threadgroup T* base = tile + (row0 + coord.y) * LD + col0 + coord.x;
    const vec<T, 4> lo = *(const threadgroup vec<T, 4>*)(base);
    const vec<T, 4> hi = *(const threadgroup vec<T, 4>*)(base + 8 * LD);
    return vec<T, 8>(lo, hi);
}

template <int D, int BK, int WM, int MASK_MODE, int WINDOW, typename T, typename KT, typename VT>
inline __attribute__((always_inline)) void sc20676_nax(
    const device T* q, const device uint8_t* k_codes, const device half* k_scale,
    const device half* k_zero, const device KT* k_tail, const device uint8_t* v_codes,
    const device half* v_scale, const device half* v_zero, const device VT* v_tail,
    uint k_packed, uint v_packed, uint kv_len, uint HQ, uint SQ, uint HKV, uint KG_CAP,
    uint KT_CAP, uint V_CAP, uint VT_CAP, threadgroup T* k_tile, threadgroup T* v_tile,
    uint tid, uint lane, uint sg, uint tile, uint kv_row, uint split, uint splits,
    thread vec<float, 8>* o, thread float* row_max, thread float* row_sum,
    thread uint* out_row, thread bool* valid) {
    constexpr uint G = 32;
    constexpr uint KW = G * D / 4;
    constexpr uint VW = D / 4;
    constexpr uint VG = D / G;
    constexpr int LD = D + 8;
    constexpr int DT = D / 16;
    constexpr int KF = BK / 16;
    constexpr uint BQ = 16 * WM;
    // Dequantization ownership: thread `tid` fills channel quad `c_quad` of token rows
    // `t_row, t_row + TSTEP, ...` of each block.
    constexpr uint TSTEP = 32 * WM / (D / 4);
    static_assert(BK == G, "a NAX block is one packed K token group");
    static_assert((32 * WM) % (D / 4) == 0 && BK % TSTEP == 0, "dequantization ownership");
    const uint c_quad = (tid % (D / 4)) * 4;
    const uint t_row = tid / (D / 4);
    const uint gqa = HQ / HKV;
    const uint rows = SQ * gqa;
    const uint b = kv_row / HKV;
    const uint kvh = kv_row % HKV;
    const short2 coord = sc20676_nax_coord(lane);
    const uint qoff = kv_len - SQ;
    uint hi[2];
    uint lo[2];
    const device T* q_row[2];
#pragma clang loop unroll(full)
    for (short i = 0; i < 2; ++i) {
        const uint m = tile * BQ + sg * 16 + coord.y + 8 * i;
        valid[i] = m < rows;
        const uint mc = min(m, rows - 1);
        const uint qi = mc / gqa;
        out_row[i] = (b * HQ + kvh * gqa + mc % gqa) * SQ + qi;
        hi[i] = MASK_MODE == 0 ? kv_len : qoff + qi + 1;
        lo[i] = (MASK_MODE == 2 && hi[i] > uint(WINDOW)) ? hi[i] - uint(WINDOW) : 0;
        q_row[i] = q + out_row[i] * D + coord.x;
    }
    // Tile-wide bounds: the union of the rows' ranges is visited; the intersection needs no mask.
    const uint first_qi = (tile * BQ) / gqa;
    const uint last_qi = (min(tile * BQ + BQ, rows) - 1) / gqa;
    const uint tile_hi = MASK_MODE == 0 ? kv_len : qoff + last_qi + 1;
    const uint first_hi = MASK_MODE == 0 ? kv_len : qoff + first_qi + 1;
    const uint tile_lo = (MASK_MODE == 2 && first_hi > uint(WINDOW)) ? first_hi - uint(WINDOW) : 0;
    const uint full_hi = first_hi;
    const uint full_lo = (MASK_MODE == 2 && tile_hi > uint(WINDOW)) ? tile_hi - uint(WINDOW) : 0;
    const uint first_block = tile_lo / BK;
    const uint end_block = (tile_hi + BK - 1) / BK;
    const uint per_split = (end_block - first_block + splits - 1) / splits;
    const uint block_begin = min(end_block, first_block + split * per_split);
    const uint block_end = min(end_block, block_begin + per_split);

    const float scale2 = rsqrt(float(D)) * M_LOG2E_F;
#pragma clang loop unroll(full)
    for (int d = 0; d < DT; ++d) o[d] = vec<float, 8>(0.0f);
    float run_max[2] = {-INFINITY, -INFINITY};
    float run_sum[2] = {0.0f, 0.0f};

    for (uint blk = block_begin; blk < block_end; ++blk) {
        const uint t0 = blk * BK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t0 + BK <= k_packed && t0 + BK <= v_packed) {
            // Whole block packed (the prefill steady state): each thread owns one channel quad
            // for the block, so the block's K scale/zero (one token group) load once per thread.
            const uint grp = kv_row * KG_CAP + blk;
            const float4 ks = sc20676_load4(k_scale + grp * D + c_quad);
            const float4 kz = sc20676_load4(k_zero + grp * D + c_quad);
            const device uint8_t* kc = k_codes + grp * KW + c_quad / 4;
#pragma clang loop unroll(full)
            for (uint r = 0; r < BK / TSTEP; ++r) {
                const uint tt = t_row + r * TSTEP;
                const float4 kv = kz + ks * sc20676_codes4(kc[tt * (D / 4)]);
                *(threadgroup vec<T, 4>*)(k_tile + tt * LD + c_quad) = vec<T, 4>(kv);
                const uint vrow = kv_row * V_CAP + t0 + tt;
                const uint meta = vrow * VG + c_quad / G;
                const float4 vv = float(v_zero[meta])
                    + float(v_scale[meta]) * sc20676_codes4(v_codes[vrow * VW + c_quad / 4]);
                *(threadgroup vec<T, 4>*)(v_tile + tt * LD + c_quad) = vec<T, 4>(vv);
            }
        } else {
#pragma clang loop unroll(full)
            for (uint r = 0; r < BK / TSTEP; ++r) {
                const uint tt = t_row + r * TSTEP;
                const uint t = t0 + tt;
                float4 kv = float4(0.0f);
                float4 vv = float4(0.0f);
                if (t < kv_len) {
                    if (t < k_packed) {
                        const uint grp = kv_row * KG_CAP + t / G;
                        const float4 codes =
                            sc20676_codes4(k_codes[grp * KW + ((t % G) * D + c_quad) / 4]);
                        kv = sc20676_load4(k_zero + grp * D + c_quad)
                            + sc20676_load4(k_scale + grp * D + c_quad) * codes;
                    } else {
                        kv = sc20676_load4(k_tail + (kv_row * KT_CAP + (t - k_packed)) * D + c_quad);
                    }
                    if (t < v_packed) {
                        const uint vrow = kv_row * V_CAP + t;
                        const uint meta = vrow * VG + c_quad / G;
                        vv = float(v_zero[meta])
                            + float(v_scale[meta]) * sc20676_codes4(v_codes[vrow * VW + c_quad / 4]);
                    } else {
                        vv = sc20676_load4(v_tail + (kv_row * VT_CAP + (t - v_packed)) * D + c_quad);
                    }
                }
                *(threadgroup vec<T, 4>*)(k_tile + tt * LD + c_quad) = vec<T, 4>(kv);
                *(threadgroup vec<T, 4>*)(v_tile + tt * LD + c_quad) = vec<T, 4>(vv);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        vec<float, 8> s[KF];
#pragma clang loop unroll(full)
        for (int f = 0; f < KF; ++f) s[f] = vec<float, 8>(0.0f);
#pragma clang loop unroll(full)
        for (int f = 0; f < KF; f += 2) {
#pragma clang loop unroll(full)
            for (int d = 0; d < DT; ++d) {
                // Query fragments are re-read per block (cache-resident), as MLX's NAX attention
                // does, rather than held in registers across the KV loop.
                const vec<T, 8> qd = vec<T, 8>(*(const device vec<T, 4>*)(q_row[0] + d * 16),
                                               *(const device vec<T, 4>*)(q_row[1] + d * 16));
                const vec<T, 8> k0 = sc20676_nax_tile_frag<T, LD>(k_tile, coord, f * 16, d * 16);
                const vec<T, 8> k1 = sc20676_nax_tile_frag<T, LD>(k_tile, coord, f * 16 + 16, d * 16);
                sc20676_nax_mma<true>(s[f], s[f + 1], qd, k0, k1);
            }
        }
        const bool full = t0 >= full_lo && t0 + BK <= full_hi;
        float block_max[2] = {-INFINITY, -INFINITY};
#pragma clang loop unroll(full)
        for (int f = 0; f < KF; ++f) {
#pragma clang loop unroll(full)
            for (short i = 0; i < 2; ++i) {
#pragma clang loop unroll(full)
                for (short j = 0; j < 4; ++j) {
                    const uint t = t0 + f * 16 + coord.x + j;
                    float v = s[f][4 * i + j] * scale2;
                    if (!full && (t >= hi[i] || t < lo[i])) v = -INFINITY;
                    s[f][4 * i + j] = v;
                    block_max[i] = max(block_max[i], v);
                }
            }
        }
        float factor[2];
#pragma clang loop unroll(full)
        for (short i = 0; i < 2; ++i) {
            block_max[i] = max(block_max[i], simd_shuffle_xor(block_max[i], 1));
            block_max[i] = max(block_max[i], simd_shuffle_xor(block_max[i], 8));
            const float next_max = max(run_max[i], block_max[i]);
            factor[i] = run_max[i] == -INFINITY ? 0.0f : fast::exp2(run_max[i] - next_max);
            float block_sum = 0.0f;
#pragma clang loop unroll(full)
            for (int f = 0; f < KF; ++f) {
#pragma clang loop unroll(full)
                for (short j = 0; j < 4; ++j) {
                    const float v = s[f][4 * i + j];
                    const float p = v == -INFINITY ? 0.0f : fast::exp2(v - next_max);
                    s[f][4 * i + j] = p;
                    block_sum += p;
                }
            }
            block_sum += simd_shuffle_xor(block_sum, 1);
            block_sum += simd_shuffle_xor(block_sum, 8);
            run_sum[i] = run_sum[i] * factor[i] + block_sum;
            run_max[i] = next_max;
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
                const vec<T, 8> v0 = sc20676_nax_tile_frag<T, LD>(v_tile, coord, f * 16, d * 16);
                const vec<T, 8> v1 = sc20676_nax_tile_frag<T, LD>(v_tile, coord, f * 16, d * 16 + 16);
                sc20676_nax_mma<false>(o[d], o[d + 1], s[f], v0, v1);
            }
        }
    }
#pragma clang loop unroll(full)
    for (short i = 0; i < 2; ++i) {
        row_max[i] = run_max[i];
        row_sum[i] = run_sum[i];
    }
}
"#;

/// Per-call NAX body: grid `x` = query-row tiles, `y` = batch × KV head, `z` = KV splits.
const NAX_BODY: &str = r#"
    using T = metal::remove_cv_t<typename sc20676_device_element<decltype(q)>::type>;
    threadgroup T k_tile[BK * (D + 8)];
    threadgroup T v_tile[BK * (D + 8)];
    vec<float, 8> o[D / 16];
    float row_max[2];
    float row_sum[2];
    uint out_row[2];
    bool valid[2];
    const uint split = threadgroup_position_in_grid.z;
    const uint splits = threadgroups_per_grid.z;
    sc20676_nax<D, BK, WM, MASK_MODE, WINDOW>(
        q, k_codes, k_scale, k_zero, k_tail, v_codes, v_scale, v_zero, v_tail, uint(params[0]),
        uint(params[1]), uint(params[2]), uint(q_shape[1]), uint(q_shape[2]), uint(v_codes_shape[1]),
        uint(k_codes_shape[2]), uint(k_tail_shape[2]), uint(v_codes_shape[2]), uint(v_tail_shape[2]),
        k_tile, v_tile, thread_index_in_threadgroup, thread_index_in_simdgroup,
        simdgroup_index_in_threadgroup, threadgroup_position_in_grid.x,
        threadgroup_position_in_grid.y, split, splits, o, row_max, row_sum, out_row, valid);
    const short col = sc20676_nax_coord(thread_index_in_simdgroup).x;
#pragma clang loop unroll(full)
    for (short i = 0; i < 2; ++i) {
        if (!valid[i]) continue;
#pragma clang loop unroll(full)
        for (uint d = 0; d < uint(D / 16); ++d) {
#pragma clang loop unroll(full)
            for (short j = 0; j < 4; ++j) {
                const uint c = d * 16 + col + j;
                const float value = o[d][4 * i + j];
                SC20676_NAX_EPILOGUE
            }
        }
    }
"#;

const NAX_SINGLE_EPILOGUE: &str = r#"
                using OutputT = typename sc20676_device_element<decltype(out)>::type;
                out[out_row[i] * D + c] =
                    static_cast<OutputT>(row_sum[i] > 0.0f ? value / row_sum[i] : 0.0f);
"#;

// The lane holding column 0 of a fragment row writes the row's maximum and normalizer.
const NAX_PARTIAL_EPILOGUE: &str = r#"
                const uint slot = out_row[i] * splits + split;
                part_acc[slot * D + c] = value;
                if (c == 0) {
                    part_max[slot] = row_max[i];
                    part_sum[slot] = row_sum[i];
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
    tiled_single: MetalKernel,
    tiled_partial: MetalKernel,
    nax_single: MetalKernel,
    nax_partial: MetalKernel,
    identity: String,
    gpu_family: PackedMetalGpuFamily,
    /// [`mlx_nax_available`], probed once at construction on the qualified family only.
    nax_available: bool,
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
            .field("nax_available", &self.nax_available)
            .finish_non_exhaustive()
    }
}

/// Live-extent and shape contract of one dispatch, validated before any kernel is encoded.
struct ValidatedDispatch {
    batch: usize,
    kv_heads: usize,
    query_heads: usize,
    query_tokens: usize,
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
        batch,
        kv_heads,
        query_heads,
        query_tokens,
        rows: batch * (query_heads / heads_per_row) * query_tokens,
        query_rows: batch * query_heads * query_tokens,
        heads_per_row,
        head_dimension,
        visible_tokens,
        mask_mode,
        window,
    })
}

/// Cache identity of the default SC-20676 packed group-affine reader.
pub const PACKED_METAL_DEFAULT_IDENTITY: &str = "sc-20676-packed-group-affine-v1";

impl PackedMetalKernel {
    pub fn new() -> Result<Self> {
        Self::for_identity(PACKED_METAL_DEFAULT_IDENTITY)
    }

    /// Construct the retained reader for one cache identity.  The identity is part of the
    /// compiled-handle binding, preventing a pipeline from being reused with another cache's
    /// layout or quantization contract.
    pub fn for_identity(identity: impl Into<String>) -> Result<Self> {
        Self::for_identity_and_family(identity, PackedMetalGpuFamily::ConservativeUnknownApple)
    }

    /// Bind a cache identity to an explicit GPU-family tuning profile. Callers may select the
    /// qualified recent-family profile only after their device probe; unknown devices retain the
    /// conservative one-SIMD-group per-row geometry for every step (never the tiled kernel).
    pub fn for_identity_and_family(
        identity: impl Into<String>,
        gpu_family: PackedMetalGpuFamily,
    ) -> Result<Self> {
        let single = ATTEND_BODY.replace("SC20676_EPILOGUE", SINGLE_EPILOGUE);
        let partial = ATTEND_BODY.replace("SC20676_EPILOGUE", PARTIAL_EPILOGUE);
        let tiled_header = format!("{HEADER}{TILED_HEADER}");
        let tiled_single = TILED_BODY.replace("SC20676_TILED_EPILOGUE", TILED_SINGLE_EPILOGUE);
        let tiled_partial = TILED_BODY.replace("SC20676_TILED_EPILOGUE", TILED_PARTIAL_EPILOGUE);
        let nax_header = format!("{tiled_header}{NAX_HEADER}");
        let nax_single = NAX_BODY.replace("SC20676_NAX_EPILOGUE", NAX_SINGLE_EPILOGUE);
        let nax_partial = NAX_BODY.replace("SC20676_NAX_EPILOGUE", NAX_PARTIAL_EPILOGUE);
        Ok(Self {
            // Compiled on first run only, so a device without the Neural Accelerator (whose Metal
            // compiler may lack MetalPerformancePrimitives) never builds these pipelines.
            nax_single: MetalKernel::with_options(
                "sc20676_nax",
                &ATTEND_INPUTS,
                &["out"],
                &nax_single,
                &nax_header,
                true,
                false,
            )?,
            nax_partial: MetalKernel::with_options(
                "sc20676_nax_split",
                &ATTEND_INPUTS,
                &["part_acc", "part_max", "part_sum"],
                &nax_partial,
                &nax_header,
                true,
                false,
            )?,
            tiled_single: MetalKernel::with_options(
                "sc20676_tiled",
                &ATTEND_INPUTS,
                &["out"],
                &tiled_single,
                &tiled_header,
                true,
                false,
            )?,
            tiled_partial: MetalKernel::with_options(
                "sc20676_tiled_split",
                &ATTEND_INPUTS,
                &["part_acc", "part_max", "part_sum"],
                &tiled_partial,
                &tiled_header,
                true,
                false,
            )?,
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
            nax_available: gpu_family == PackedMetalGpuFamily::Apple7OrNewer && mlx_nax_available(),
        })
    }

    /// Test seam: behave as on a device where MLX reports no Neural Accelerator.
    #[cfg(test)]
    pub(crate) fn without_nax(mut self) -> Self {
        self.nax_available = false;
        self
    }

    /// Whether this reader may run the NAX tiled kernel at all ([`mlx_nax_available`] on the
    /// qualified family).
    pub fn nax_available(&self) -> bool {
        self.nax_available
    }

    /// Whether a multi-row step with `query_dtype` queries at `head_dimension` runs the NAX tiled
    /// kernel rather than the fp32 tiled kernel, and why.
    pub fn nax_selection(&self, head_dimension: usize, query_dtype: Dtype) -> PackedNaxSelection {
        let rejected = |reason| PackedNaxSelection {
            selected: false,
            reason,
        };
        if self.gpu_family != PackedMetalGpuFamily::Apple7OrNewer {
            rejected(NAX_CONSERVATIVE)
        } else if !self.nax_available {
            rejected(NAX_UNAVAILABLE)
        } else if !matches!(query_dtype, Dtype::Bfloat16 | Dtype::Float16) {
            rejected(NAX_F32_QUERY)
        } else if !packed_nax_head_dimension_supported(head_dimension) {
            rejected(NAX_HEAD_DIMENSION)
        } else {
            PackedNaxSelection {
                selected: true,
                reason: NAX_SELECTED,
            }
        }
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

    /// KV splits of the kernel [`Self::planned_path`] selects (1 = single pass).
    pub fn planned_splits(&self, args: &PackedAttentionArgs<'_>) -> Result<usize> {
        Ok(match self.planned_path(args)? {
            PackedKernelPath::PerRow { splits }
            | PackedKernelPath::Tiled { splits }
            | PackedKernelPath::NaxTiled { splits } => splits,
        })
    }

    /// Whether a step of `query_tokens` rows at `head_dimension` runs the tiled kernel: only on the
    /// qualified family, from [`packed_tiled_min_query_tokens`].
    pub fn selects_tiled(&self, query_tokens: usize, head_dimension: usize) -> bool {
        self.gpu_family == PackedMetalGpuFamily::Apple7OrNewer
            && query_tokens >= packed_tiled_min_query_tokens(head_dimension)
    }

    /// Geometry of the kernel [`Self::dispatch`] runs for a step of `query_tokens` rows of
    /// `query_dtype` queries, or `None` for an unsupported head dimension.
    pub fn kernel_descriptor(
        &self,
        query_tokens: usize,
        head_dimension: usize,
        query_dtype: Dtype,
    ) -> Option<PackedKernelDescriptor> {
        let tuning = self.gpu_family.tuning(head_dimension)?;
        let nax = self.nax_selection(head_dimension, query_dtype);
        let tiled = self.selects_tiled(query_tokens, head_dimension);
        Some(if tiled && nax.selected {
            PackedKernelDescriptor {
                kernel: "sc20676_nax_tiled_matmul2d",
                selection: nax.reason,
                threads: NAX_SIMD_GROUPS * SIMD_WIDTH,
                simd_groups: NAX_SIMD_GROUPS,
                kv_block_tokens: NAX_BLOCK_TOKENS,
                threadgroup_barriers_per_kv_block: 2,
                softmax_state: "fp32 registers per 16-row NAX fragment (two shuffles per row \
                                reduction); K/V blocks dequantized once into 16-bit threadgroup \
                                tiles; S = Q·Kᵀ and O += P·V on mpp matmul2d; split-KV partials \
                                merged by a reduce pass",
            }
        } else if tiled {
            PackedKernelDescriptor {
                kernel: "sc20676_tiled_multi_row_simdgroup_matrix",
                selection: nax.reason,
                threads: TILED_SIMD_GROUPS * SIMD_WIDTH,
                simd_groups: TILED_SIMD_GROUPS,
                kv_block_tokens: tiled_block_tokens(head_dimension),
                threadgroup_barriers_per_kv_block: 2,
                softmax_state: "registers per 8-row simdgroup_matrix fragment (two shuffles per \
                                row reduction); K/V blocks dequantized once into threadgroup tiles; \
                                split-KV partials merged by a reduce pass",
            }
        } else {
            PackedKernelDescriptor {
                kernel: "sc20676_split_kv_simdgroup",
                selection: "fewer query rows than packed_tiled_min_query_tokens, or the \
                            conservative family",
                threads: tuning.threads,
                simd_groups: tuning.simd_groups,
                kv_block_tokens: PACKED_METAL_QUANT_GROUP_SIZE,
                threadgroup_barriers_per_kv_block: 0,
                softmax_state: "registers per SIMD group (simd_sum), merged once through \
                                threadgroup memory; split-KV partials merged by a reduce pass",
            }
        })
    }

    fn required_tuning(&self, head_dimension: usize) -> Result<PackedMetalTuning> {
        self.gpu_family.tuning(head_dimension).ok_or_else(|| {
            Error::Unsupported(
                "SC-20676 has no conservative tuning for this device/geometry".into(),
            )
        })
    }

    /// Kernel and KV split count [`Self::dispatch`] selects (see [`Self::selects_tiled`]).
    pub fn planned_path(&self, args: &PackedAttentionArgs<'_>) -> Result<PackedKernelPath> {
        let validated = validate_dispatch(args)?;
        if self.selects_tiled(validated.query_tokens, validated.head_dimension) {
            if self
                .nax_selection(validated.head_dimension, args.query.dtype())
                .selected
            {
                return Ok(PackedKernelPath::NaxTiled {
                    splits: nax_splits(&validated),
                });
            }
            return Ok(PackedKernelPath::Tiled {
                splits: tiled_splits(&validated),
            });
        }
        let tuning = self.required_tuning(validated.head_dimension)?;
        Ok(PackedKernelPath::PerRow {
            splits: packed_kv_split_count(
                validated.rows,
                validated.visible_tokens,
                tuning.simd_groups,
            ),
        })
    }

    pub fn dispatch(&self, args: &PackedAttentionArgs<'_>) -> Result<Array> {
        match self.planned_path(args)? {
            PackedKernelPath::NaxTiled { .. } => self.dispatch_nax(args, None),
            PackedKernelPath::Tiled { .. } => self.dispatch_tiled(args, None),
            PackedKernelPath::PerRow { .. } => self.dispatch_with_splits(args, None),
        }
    }

    /// Run the NAX tiled kernel regardless of `S_q`, with an explicit KV split count (`None` =
    /// [`packed_tiled_split_count`] over 64-row tiles). Test, benchmark, and evidence seam; refused
    /// wherever [`Self::nax_selection`] rejects NAX for the query dtype and head dimension.
    pub fn dispatch_nax(
        &self,
        args: &PackedAttentionArgs<'_>,
        splits: Option<usize>,
    ) -> Result<Array> {
        let validated = validate_dispatch(args)?;
        let selection = self.nax_selection(validated.head_dimension, args.query.dtype());
        if !selection.selected {
            return Err(Error::Unsupported(format!(
                "SC-20676 NAX tiled kernel refused: {}",
                selection.reason
            )));
        }
        let splits = splits
            .unwrap_or_else(|| nax_splits(&validated))
            .clamp(1, MAX_KV_SPLITS);
        let tiles = (validated.query_tokens * (validated.query_heads / validated.kv_heads))
            .div_ceil(NAX_ROWS);
        self.dispatch_tiles(
            args,
            &validated,
            [&self.nax_single, &self.nax_partial],
            tiles,
            NAX_SIMD_GROUPS,
            NAX_BLOCK_TOKENS,
            splits,
        )
    }

    /// Run the tiled multi-row kernel regardless of `S_q`, with an explicit KV split count
    /// (`None` = [`packed_tiled_split_count`]). Test, benchmark, and evidence seam; refused on the
    /// conservative family, which never runs the tiled geometry.
    pub fn dispatch_tiled(
        &self,
        args: &PackedAttentionArgs<'_>,
        splits: Option<usize>,
    ) -> Result<Array> {
        if self.gpu_family != PackedMetalGpuFamily::Apple7OrNewer {
            return Err(Error::Unsupported(
                "SC-20676 tiled kernel requires the qualified Apple7OrNewer profile".into(),
            ));
        }
        let validated = validate_dispatch(args)?;
        let splits = splits
            .unwrap_or_else(|| tiled_splits(&validated))
            .clamp(1, MAX_KV_SPLITS);
        let tiles = (validated.query_tokens * (validated.query_heads / validated.kv_heads))
            .div_ceil(TILED_ROWS);
        self.dispatch_tiles(
            args,
            &validated,
            [&self.tiled_single, &self.tiled_partial],
            tiles,
            TILED_SIMD_GROUPS,
            tiled_block_tokens(validated.head_dimension),
            splits,
        )
    }

    /// Encode one tiled-family dispatch (fp32 tiled or NAX): grid `x` = row tiles, `y` = batch ×
    /// KV head, `z` = KV splits, then the shared reduction when split.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_tiles(
        &self,
        args: &PackedAttentionArgs<'_>,
        validated: &ValidatedDispatch,
        [single, partial]: [&MetalKernel; 2],
        tiles: usize,
        simd_groups: usize,
        block_tokens: usize,
        splits: usize,
    ) -> Result<Array> {
        let head_dimension = validated.head_dimension;
        let params = dispatch_params(args)?;
        let threads = simd_groups * SIMD_WIDTH;
        let grid_x = tiles
            .checked_mul(threads)
            .ok_or_else(|| Error::Unsupported("SC-20676 Metal grid dimension overflow".into()))
            .and_then(|value| checked_msl_i32(value, "tiled grid width"))?;
        let kv_rows = checked_msl_i32(validated.batch * validated.kv_heads, "KV rows")?;
        let splits_i32 = checked_msl_i32(splits, "KV splits")?;
        let threads = checked_msl_i32(threads, "thread-group width")?;
        let head_i32 = checked_msl_i32(head_dimension, "head dimension")?;
        let block = checked_msl_i32(block_tokens, "tiled block")?;
        let wm = checked_msl_i32(simd_groups, "tiled SIMD groups")?;
        let kernel = if splits == 1 { single } else { partial };
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
            .template_arg("BK", block)
            .template_arg("WM", wm)
            .template_arg("MASK_MODE", validated.mask_mode)
            .template_arg("WINDOW", validated.window);
        let query_shape = args.query.shape().to_vec();
        if splits == 1 {
            return attend
                .output(OutputArg {
                    shape: query_shape,
                    dtype: args.query.dtype(),
                })
                .grid(grid_x, kv_rows, 1)
                .thread_group(threads, 1, 1)
                .run()?
                .into_iter()
                .next()
                .ok_or_else(|| Error::Msg("SC-20676 tiled kernel returned no output".into()));
        }
        let query_rows = checked_msl_i32(validated.query_rows, "query rows")?;
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
            .grid(grid_x, kv_rows, splits_i32)
            .thread_group(threads, 1, 1)
            .run()?
            .into_iter();
        let (Some(part_acc), Some(part_max), Some(part_sum)) =
            (partials.next(), partials.next(), partials.next())
        else {
            return Err(Error::Msg(
                "SC-20676 tiled split kernel returned too few outputs".into(),
            ));
        };
        self.reduce_partials(
            [&part_acc, &part_max, &part_sum],
            args.query,
            query_rows,
            head_i32,
        )
    }

    /// Second split-KV pass shared by both kernels: merge `[rows, splits]` partials into the
    /// query-shaped, query-dtype output.
    fn reduce_partials(
        &self,
        [part_acc, part_max, part_sum]: [&Array; 3],
        query: &Array,
        query_rows: i32,
        head_dimension: i32,
    ) -> Result<Array> {
        let simd = checked_msl_i32(SIMD_WIDTH, "SIMD width")?;
        self.reduce
            .apply()
            .input(part_acc)
            .input(part_max)
            .input(part_sum)
            .output(OutputArg {
                shape: query.shape().to_vec(),
                dtype: query.dtype(),
            })
            .grid(simd, query_rows, 1)
            .thread_group(simd, 1, 1)
            .template_arg("D", head_dimension)
            .run()?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Msg("SC-20676 reduce kernel returned no output".into()))
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
        let params = dispatch_params(args)?;
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
        self.reduce_partials(
            [&part_acc, &part_max, &part_sum],
            args.query,
            query_rows,
            head_i32,
        )
    }
}

/// Live extents `[key_packed, value_packed, kv]` as the kernels' `params` input.
fn dispatch_params(args: &PackedAttentionArgs<'_>) -> Result<Array> {
    Ok(Array::from_slice(
        &[
            checked_msl_i32(args.key_packed_tokens, "packed key tokens")?,
            checked_msl_i32(args.value_packed_tokens, "packed value tokens")?,
            checked_msl_i32(args.kv_tokens, "KV tokens")?,
        ],
        &[3],
    ))
}

/// Tiled split count for a validated dispatch (threadgroups = row tiles × batch × KV heads).
fn tiled_splits(validated: &ValidatedDispatch) -> usize {
    tile_splits(validated, TILED_ROWS)
}

/// NAX split count: the tiled heuristic over 64-row tiles.
fn nax_splits(validated: &ValidatedDispatch) -> usize {
    tile_splits(validated, NAX_ROWS)
}

fn tile_splits(validated: &ValidatedDispatch, rows_per_tile: usize) -> usize {
    let tiles = (validated.query_tokens * (validated.query_heads / validated.kv_heads))
        .div_ceil(rows_per_tile);
    packed_tiled_split_count(
        tiles * validated.batch * validated.kv_heads,
        validated.visible_tokens,
    )
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

    /// Synthetic packed buffers (random codes and metadata, no model weights) for `total` tokens
    /// of one layer, all tokens quantized, plus one-group residual tails.
    struct SyntheticPackedLayer {
        key_codes: Array,
        key_scales: Array,
        key_zeros: Array,
        key_tail: Array,
        value_codes: Array,
        value_scales: Array,
        value_zeros: Array,
        value_tail: Array,
    }

    const BENCH_QUERY_HEADS: i32 = 24;
    const BENCH_KV_HEADS: i32 = 8;
    const BENCH_DIM: i32 = 128;

    fn synthetic_packed_layer(total: i32, d: i32) -> SyntheticPackedLayer {
        use mlx_rs::random::{normal, randint};
        let h = BENCH_KV_HEADS;
        let groups = total / 32;
        let codes = |shape: &[i32]| {
            randint::<_, i32>(0, 256, shape, None)
                .unwrap()
                .as_dtype(Dtype::Uint8)
                .unwrap()
        };
        let metadata = |shape: &[i32], scale: f32| {
            normal::<f32>(shape, None, scale, None)
                .unwrap()
                .abs()
                .unwrap()
                .as_dtype(Dtype::Float16)
                .unwrap()
        };
        let tail = || {
            normal::<f32>(&[1, h, 32, d], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        let layer = SyntheticPackedLayer {
            key_codes: codes(&[1, h, groups, 32 * d / 4]),
            key_scales: metadata(&[1, h, groups, d], 0.3),
            key_zeros: metadata(&[1, h, groups, d], 0.5),
            key_tail: tail(),
            value_codes: codes(&[1, h, total, d / 4]),
            value_scales: metadata(&[1, h, total, d / 32], 0.3),
            value_zeros: metadata(&[1, h, total, d / 32], 0.5),
            value_tail: tail(),
        };
        for array in [
            &layer.key_codes,
            &layer.key_scales,
            &layer.key_zeros,
            &layer.key_tail,
            &layer.value_codes,
            &layer.value_scales,
            &layer.value_zeros,
            &layer.value_tail,
        ] {
            array.eval().unwrap();
        }
        layer
    }

    impl SyntheticPackedLayer {
        fn args<'a>(&'a self, query: &'a Array, kv: usize) -> PackedAttentionArgs<'a> {
            PackedAttentionArgs {
                query,
                key_codes: &self.key_codes,
                key_scales: &self.key_scales,
                key_zeros: &self.key_zeros,
                key_tail: &self.key_tail,
                value_codes: &self.value_codes,
                value_scales: &self.value_scales,
                value_zeros: &self.value_zeros,
                value_tail: &self.value_tail,
                key_packed_tokens: kv,
                value_packed_tokens: kv,
                kv_tokens: kv,
                mask: PackedMask::Causal,
            }
        }
    }

    fn bench_query(rows: i32, d: i32) -> Array {
        let query =
            mlx_rs::random::normal::<f32>(&[1, BENCH_QUERY_HEADS, rows, d], None, None, None)
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
        query.eval().unwrap();
        query
    }

    /// Wall milliseconds of one evaluated attention call.
    fn timed(run: impl Fn() -> Array) -> f64 {
        let started = std::time::Instant::now();
        run().eval().unwrap();
        started.elapsed().as_secs_f64() * 1000.0
    }

    /// SC-20676 threshold sweep (synthetic, one layer, bf16, Hq = 24, Hkv = 8, D = 64/128/256):
    /// per-row kernel (both tuning families) vs tiled kernel for small `S_q` over 4k and 32k
    /// histories. Median of five evaluated calls each after one warm call.
    #[test]
    #[ignore = "GPU micro-benchmark; run explicitly with --ignored --nocapture"]
    fn tiled_threshold_crossover_sweep() {
        let conservative = PackedMetalKernel::new().unwrap();
        let recent = PackedMetalKernel::for_identity_and_family(
            "bench",
            PackedMetalGpuFamily::Apple7OrNewer,
        )
        .unwrap();
        let median = |run: &dyn Fn() -> Array| {
            run().eval().unwrap();
            let mut samples = (0..5).map(|_| timed(run)).collect::<Vec<_>>();
            samples.sort_by(f64::total_cmp);
            samples[2]
        };
        eprintln!("D\tkv\tS_q\tper-row conservative ms\tper-row apple7 ms\ttiled ms\tNAX tiled ms");
        for d in [64, 128, 256] {
            let layer = synthetic_packed_layer(32_768, d);
            for kv in [4096usize, 32_768] {
                for rows in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64] {
                    let query = bench_query(rows, d);
                    let args = layer.args(&query, kv);
                    let per_row_c =
                        median(&|| conservative.dispatch_with_splits(&args, None).unwrap());
                    let per_row_r = median(&|| recent.dispatch_with_splits(&args, None).unwrap());
                    let tiled = median(&|| recent.dispatch_tiled(&args, None).unwrap());
                    let nax = if packed_nax_head_dimension_supported(d as usize) {
                        format!(
                            "{:.3}",
                            median(&|| recent.dispatch_nax(&args, None).unwrap())
                        )
                    } else {
                        "-".into()
                    };
                    eprintln!(
                        "{d}\t{kv}\t{rows}\t{per_row_c:.3}\t{per_row_r:.3}\t{tiled:.3}\t{nax}"
                    );
                }
            }
        }
    }

    fn chunked_prefill_benchmark(total: i32) {
        use mlx_rs::fast::{scaled_dot_product_attention, ScaledDotProductAttentionMask};
        use mlx_rs::ops::indexing::TryIndexOp;
        let kernel = PackedMetalKernel::for_identity_and_family(
            "bench",
            PackedMetalGpuFamily::Apple7OrNewer,
        )
        .unwrap();
        let layer = synthetic_packed_layer(total, BENCH_DIM);
        let dense = || {
            let array = mlx_rs::random::normal::<f32>(
                &[1, BENCH_KV_HEADS, total, BENCH_DIM],
                None,
                None,
                None,
            )
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
            array.eval().unwrap();
            array
        };
        let (dense_keys, dense_values) = (dense(), dense());
        let scale = (BENCH_DIM as f32).powf(-0.5);
        eprintln!("total\tchunk\tpath\tchunks\tmean ms/chunk\tlast-chunk ms\ttotal ms\tbatched ms");
        for chunk in [512, 2048] {
            let query = bench_query(chunk, BENCH_DIM);
            let chunks = total / chunk;
            // Warm every pipeline once on the first chunk's geometry.
            let warm = layer.args(&query, chunk as usize);
            kernel.dispatch_tiled(&warm, None).unwrap().eval().unwrap();
            kernel.dispatch_nax(&warm, None).unwrap().eval().unwrap();
            kernel
                .dispatch_with_splits(&warm, None)
                .unwrap()
                .eval()
                .unwrap();
            let paths: [(&str, &dyn Fn(usize) -> Array); 4] = [
                ("dense-sdpa", &|kv| {
                    let end = kv as i32;
                    let keys = dense_keys.try_index((.., .., 0..end, ..)).unwrap();
                    let values = dense_values.try_index((.., .., 0..end, ..)).unwrap();
                    scaled_dot_product_attention(
                        &query,
                        &keys,
                        &values,
                        scale,
                        ScaledDotProductAttentionMask::Causal,
                        None,
                    )
                    .unwrap()
                }),
                ("nax-tiled-packed", &|kv| {
                    kernel.dispatch_nax(&layer.args(&query, kv), None).unwrap()
                }),
                ("tiled-packed", &|kv| {
                    kernel
                        .dispatch_tiled(&layer.args(&query, kv), None)
                        .unwrap()
                }),
                ("per-row-packed", &|kv| {
                    kernel
                        .dispatch_with_splits(&layer.args(&query, kv), None)
                        .unwrap()
                }),
            ];
            for (name, run) in paths {
                run(chunk as usize).eval().unwrap();
                // Per chunk, the fastest of three evaluated calls (a shared GPU adds noise).
                let samples = (1..=chunks)
                    .map(|index| {
                        (0..3)
                            .map(|_| timed(|| run((index * chunk) as usize)))
                            .fold(f64::INFINITY, f64::min)
                    })
                    .collect::<Vec<_>>();
                let sum = samples.iter().sum::<f64>();
                // Every chunk encoded lazily and evaluated together: GPU time without the
                // per-call host round trip (fastest of three). The per-row path is left out: its
                // chunks together would sit in one command buffer for seconds.
                let batched = if name == "per-row-packed" {
                    f64::NAN
                } else {
                    (0..3)
                        .map(|_| {
                            let started = std::time::Instant::now();
                            let outputs = (1..=chunks)
                                .map(|index| run((index * chunk) as usize))
                                .collect::<Vec<_>>();
                            mlx_rs::transforms::eval(&outputs).unwrap();
                            started.elapsed().as_secs_f64() * 1000.0
                        })
                        .fold(f64::INFINITY, f64::min)
                };
                eprintln!(
                    "{total}\t{chunk}\t{name}\t{chunks}\t{:.2}\t{:.2}\t{sum:.1}\t{batched:.1}",
                    sum / f64::from(chunks),
                    samples.last().copied().unwrap_or_default()
                );
            }
        }
    }

    /// SC-20676 chunked-prefill micro-benchmark over an 8k history (see
    /// [`chunked_prefill_benchmark`]): dense MLX SDPA on the full chunk vs the tiled and per-row
    /// packed readers, synthetic bf16 tensors, one layer.
    #[test]
    #[ignore = "GPU micro-benchmark; run explicitly with --ignored --nocapture"]
    fn tiled_chunked_prefill_benchmark_8k() {
        chunked_prefill_benchmark(8192);
    }

    /// The 32k-token variant of [`tiled_chunked_prefill_benchmark_8k`].
    #[test]
    #[ignore = "GPU micro-benchmark; run explicitly with --ignored --nocapture"]
    fn tiled_chunked_prefill_benchmark_32k() {
        chunked_prefill_benchmark(32_768);
    }
}
