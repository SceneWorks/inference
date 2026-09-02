//! Experimental compressed-domain attention for Krea Realtime's persistent KV cache.
//!
//! This module owns both a source-level CPU oracle and an off-by-default MLX custom-Metal
//! dispatch. [`CompressedKvCache`] packs D-axis affine rows compatible with the existing
//! `crate::causal::PackedKv` layout, streams those rows through tiled online softmax, and gives
//! device code an exact fail-closed selection boundary. The public API remains disabled until a
//! caller explicitly retains a Metal handle and produces the SC-20684 device receipt.
//!
//! No cached K/V row is expanded into a dense window and no score matrix is allocated.
//! The only dynamic result allocation is the required output tensor; the device analogue
//! owns at most `TILE_ROWS * head_dim * sizeof(f32)` scratch per invocation.

use mlx_gen::Result as MlxGenResult;
use mlx_rs::fast::{MetalKernel, OutputArg};
use mlx_rs::{Array, Dtype};
use std::{
    fmt,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Instant,
};

/// The one opt-in source POC representation.  Q8 is the default candidate; Q4 remains
/// deliberately gated behind a separate quality-arm acknowledgement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressedTier {
    Q8,
    Q4,
}

impl CompressedTier {
    const fn bits(self) -> usize {
        match self {
            Self::Q8 => 8,
            Self::Q4 => 4,
        }
    }
}

/// Explicit experimental selection.  Constructing a [`CompressedKvCache`] with the
/// default value returns a dense fallback *before* creating or changing packed state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExperimentalCompressedKvConfig {
    pub enabled: bool,
    pub tier: CompressedTier,
    pub allow_q4_quality_arm: bool,
    pub group_size: usize,
}

impl Default for ExperimentalCompressedKvConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            tier: CompressedTier::Q8,
            allow_q4_quality_arm: false,
            group_size: 64,
        }
    }
}

/// Identity-safe opaque ownership for a compiled Metal kernel.  A production binding
/// stores its device pipeline object behind this trait; the reference tests use a small
/// named handle and never make a device call.
pub trait RetainedKernelHandle: fmt::Debug {
    fn identity(&self) -> &str;
    fn retained_bytes(&self) -> usize;
}

/// Retained MLX custom-kernel object for the device POC.  Construction only records MSL with
/// MLX; compilation occurs lazily at its first dispatch.  It is deliberately owned by the cache,
/// not reconstructed per attention call, so MLX's JIT cache and the object identity stay coupled.
#[derive(Debug)]
pub struct KreaPackedMetalKernel {
    kernel: MetalKernel,
    tier: CompressedTier,
}

impl KreaPackedMetalKernel {
    pub fn new(tier: CompressedTier) -> MlxGenResult<Self> {
        let kernel = MetalKernel::with_options(
            match tier {
                CompressedTier::Q8 => "sc20684_krea_packed_q8_online",
                CompressedTier::Q4 => "sc20684_krea_packed_q4_online",
            },
            &[
                "q",
                "kw",
                "ks",
                "kb",
                "vw",
                "vs",
                "vb",
                "current_k",
                "current_v",
                "query_positions",
                "key_positions",
            ],
            &["out"],
            KREA_PACKED_ONLINE_SOFTMAX_MSL,
            KREA_PACKED_ONLINE_SOFTMAX_HEADER,
            true,
            false,
        )?;
        Ok(Self { kernel, tier })
    }

    pub const fn identity_for(tier: CompressedTier) -> &'static str {
        match tier {
            CompressedTier::Q8 => "sc20684/krea-packed-affine-q8-d128-g64-v1",
            CompressedTier::Q4 => "sc20684/krea-packed-affine-q4-d128-g64-v1",
        }
    }

    /// Dispatch packed historical K/V plus the current chunk. `query_positions` and
    /// `key_positions` are O(S) position vectors; there is no `[Sq,Sk]` mask or score input.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dispatch(
        &self,
        q: &Array,
        kw: &Array,
        ks: &Array,
        kb: &Array,
        vw: &Array,
        vs: &Array,
        vb: &Array,
        current_k: &Array,
        current_v: &Array,
        query_positions: &Array,
        key_positions: &Array,
        block_size: usize,
    ) -> MlxGenResult<Array> {
        let shape = q.shape();
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 40 || shape[3] != 128 {
            return Err(mlx_gen::Error::Msg(
                "krea packed Metal POC supports only B=1/H=40/D=128".into(),
            ));
        }
        let history_tokens = kw.shape().get(2).copied().unwrap_or_default();
        let current_tokens = current_k.shape().get(2).copied().unwrap_or_default();
        let words_per_row = 128 * self.tier.bits() / 32;
        let metadata_per_row = 128 / 64;
        let packed_shape = [1, 40, history_tokens, words_per_row as i32];
        let metadata_shape = [1, 40, history_tokens, metadata_per_row];
        if current_k.shape() != current_v.shape()
            || current_k.shape().len() != 4
            || current_k.shape()[0] != 1
            || current_k.shape()[1] != 40
            || current_k.shape()[3] != 128
            || current_tokens != shape[2]
            || kw.shape() != packed_shape
            || vw.shape() != packed_shape
            || ks.shape() != metadata_shape
            || kb.shape() != metadata_shape
            || vs.shape() != metadata_shape
            || vb.shape() != metadata_shape
            || key_positions.shape().len() != 1
            || query_positions.shape().len() != 1
            || query_positions.shape()[0] != shape[2]
            || key_positions.shape()[0] != history_tokens + current_tokens
            || block_size == 0
            || kw.dtype() != Dtype::Uint32
            || vw.dtype() != Dtype::Uint32
            || ks.dtype() != Dtype::Bfloat16
            || kb.dtype() != Dtype::Bfloat16
            || vs.dtype() != Dtype::Bfloat16
            || vb.dtype() != Dtype::Bfloat16
            || q.dtype() != Dtype::Bfloat16
            || current_k.dtype() != Dtype::Bfloat16
            || current_v.dtype() != Dtype::Bfloat16
            || query_positions.dtype() != Dtype::Int64
            || key_positions.dtype() != Dtype::Int64
        {
            return Err(mlx_gen::Error::Msg(
                "krea packed Metal POC received incompatible current-K/V or position geometry"
                    .into(),
            ));
        }
        let (batch, heads, sq, dim) = (shape[0], shape[1], shape[2], shape[3]);
        let output = self
            .kernel
            .apply()
            .input(q)
            .input(kw)
            .input(ks)
            .input(kb)
            .input(vw)
            .input(vs)
            .input(vb)
            .input(current_k)
            .input(current_v)
            .input(query_positions)
            .input(key_positions)
            .output(OutputArg {
                shape: vec![batch, heads, sq, dim],
                dtype: q.dtype(),
            })
            .grid(((sq + 7) / 8) * heads * 256, batch, 1)
            .thread_group(256, 1, 1)
            .template_arg("BITS", self.tier.bits() as i32)
            .template_arg("HEAD_DIM", dim)
            .template_arg("BLOCK_SIZE", block_size as i32)
            .run()?;
        output.into_iter().next().ok_or_else(|| {
            mlx_gen::Error::Msg("krea packed Metal kernel returned no output".into())
        })
    }
}

impl RetainedKernelHandle for KreaPackedMetalKernel {
    fn identity(&self) -> &str {
        Self::identity_for(self.tier)
    }
    fn retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

// MLX synthesizes the argument list from `MetalKernel::with_options`.  Each threadgroup owns a
// 8-row query tile; packed history is decoded from Krea's actual uint32/bf16 D-axis rows.  The
// simdgroup fragments are register/tile-local, while max/sum/value are streamed over keys: no
// dense historical K/V buffer and no `Sq × Sk` score allocation are representable by this source.
const KREA_PACKED_ONLINE_SOFTMAX_HEADER: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
"#;

const KREA_PACKED_ONLINE_SOFTMAX_MSL: &str = r#"
    // One 256-thread group owns an 8-query tile: eight simdgroups, each with 32 lanes.
    // A lane owns four D values, covering all 128 output channels.  Scores are streamed in
    // 8-key blocks; `scores` is a bounded 8x8 threadgroup tile, never an Sq-by-Sk allocation.
    constexpr uint Q_TILE = 8, K_TILE = 8, GROUP = 64;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint q_row = simdgroup_index_in_threadgroup;
    const uint group = threadgroup_position_in_grid.x;
    const uint tiles_per_head = (q_shape[2] + Q_TILE - 1) / Q_TILE;
    const uint h = group / tiles_per_head;
    const uint q_base = (group % tiles_per_head) * Q_TILE;
    const uint qi = q_base + q_row;
    const uint b = threadgroup_position_in_grid.y;
    threadgroup float q_tile[Q_TILE][HEAD_DIM];
    threadgroup float k_tile[K_TILE][HEAD_DIM];
    threadgroup float v_tile[K_TILE][HEAD_DIM];
    threadgroup float scores[Q_TILE][K_TILE];
    threadgroup float row_max[Q_TILE];
    threadgroup float row_sum[Q_TILE];
    threadgroup float old_weight[Q_TILE];
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    if (q_row < Q_TILE && qi < q_shape[2]) {
        for (uint j = 0; j < 4; ++j) {
            const uint d = lane * 4 + j;
            q_tile[q_row][d] = float(q[((b * q_shape[1] + h) * q_shape[2] + qi) * HEAD_DIM + d]);
        }
    } else if (q_row < Q_TILE) {
        // Matrix loads cover the complete 8-row fragment. Zero inactive tail rows so the final
        // partial query tile never reads indeterminate threadgroup storage.
        for (uint j = 0; j < 4; ++j) q_tile[q_row][lane * 4 + j] = 0.0f;
    }
    if (lane == 0) { row_max[q_row] = -INFINITY; row_sum[q_row] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint key_base = 0; key_base < key_positions_shape[0]; key_base += K_TILE) {
        for (uint i = tid; i < K_TILE * HEAD_DIM; i += 256) {
            const uint kr = i / HEAD_DIM, d = i % HEAD_DIM, key = key_base + kr;
            if (key < key_positions_shape[0]) {
                const bool historical = key < kw_shape[2];
                const uint word = d * BITS / 32, shift = (d * BITS) % 32, mask = (1u << BITS) - 1u;
                const uint kc = historical ? ((kw[((b * q_shape[1] + h) * kw_shape[2] + key) * (HEAD_DIM * BITS / 32) + word] >> shift) & mask) : 0u;
                const uint vc = historical ? ((vw[((b * q_shape[1] + h) * vw_shape[2] + key) * (HEAD_DIM * BITS / 32) + word] >> shift) & mask) : 0u;
                k_tile[kr][d] = historical ? float(ks[((b * q_shape[1] + h) * kw_shape[2] + key) * (HEAD_DIM / GROUP) + d / GROUP]) * float(kc) + float(kb[((b * q_shape[1] + h) * kw_shape[2] + key) * (HEAD_DIM / GROUP) + d / GROUP]) : float(current_k[((b * q_shape[1] + h) * current_k_shape[2] + key - kw_shape[2]) * HEAD_DIM + d]);
                v_tile[kr][d] = historical ? float(vs[((b * q_shape[1] + h) * vw_shape[2] + key) * (HEAD_DIM / GROUP) + d / GROUP]) * float(vc) + float(vb[((b * q_shape[1] + h) * vw_shape[2] + key) * (HEAD_DIM / GROUP) + d / GROUP]) : float(current_v[((b * q_shape[1] + h) * current_v_shape[2] + key - vw_shape[2]) * HEAD_DIM + d]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // simdgroup 0 computes the bounded Q_TILE x K_TILE score block with real MMA fragments.
        if (q_row == 0) {
            simdgroup_matrix<float, 8, 8> a, bmat, c(0.0f);
            for (uint d0 = 0; d0 < HEAD_DIM; d0 += 8) {
                simdgroup_load(a, &q_tile[0][d0], HEAD_DIM);
                simdgroup_load(bmat, &k_tile[0][d0], HEAD_DIM, true);
                simdgroup_multiply_accumulate(c, a, bmat, c);
            }
            simdgroup_store(c, &scores[0][0], K_TILE);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0 && qi < q_shape[2]) {
            const uint valid = min(K_TILE, key_positions_shape[0] - key_base);
            float next_max = row_max[q_row];
            for (uint kr = 0; kr < valid; ++kr) {
                const long q_position = query_positions[qi];
                const long k_position = key_positions[key_base + kr];
                if (k_position / long(BLOCK_SIZE) > q_position / long(BLOCK_SIZE)) continue;
                next_max = max(next_max, scores[q_row][kr] * rsqrt(float(HEAD_DIM)));
            }
            old_weight[q_row] = isfinite(row_max[q_row]) ? exp(row_max[q_row] - next_max) : 0.0f;
            row_sum[q_row] *= old_weight[q_row];
            for (uint kr = 0; kr < valid; ++kr) {
                const long q_position = query_positions[qi];
                const long k_position = key_positions[key_base + kr];
                if (k_position / long(BLOCK_SIZE) > q_position / long(BLOCK_SIZE)) continue;
                row_sum[q_row] += exp(scores[q_row][kr] * rsqrt(float(HEAD_DIM)) - next_max);
            }
            row_max[q_row] = next_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (qi < q_shape[2]) {
            const uint valid = min(K_TILE, key_positions_shape[0] - key_base);
            acc0 *= old_weight[q_row]; acc1 *= old_weight[q_row]; acc2 *= old_weight[q_row]; acc3 *= old_weight[q_row];
            for (uint kr = 0; kr < valid; ++kr) {
                const long q_position = query_positions[qi];
                const long k_position = key_positions[key_base + kr];
                if (k_position / long(BLOCK_SIZE) > q_position / long(BLOCK_SIZE)) continue;
                const float w = exp(scores[q_row][kr] * rsqrt(float(HEAD_DIM)) - row_max[q_row]);
                acc0 += w * v_tile[kr][lane * 4 + 0]; acc1 += w * v_tile[kr][lane * 4 + 1];
                acc2 += w * v_tile[kr][lane * 4 + 2]; acc3 += w * v_tile[kr][lane * 4 + 3];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (qi < q_shape[2]) {
        const uint o = ((b * q_shape[1] + h) * q_shape[2] + qi) * HEAD_DIM + lane * 4;
        out[o + 0] = bfloat16_t(acc0 / row_sum[q_row]);
        out[o + 1] = bfloat16_t(acc1 / row_sum[q_row]);
        out[o + 2] = bfloat16_t(acc2 / row_sum[q_row]);
        out[o + 3] = bfloat16_t(acc3 / row_sum[q_row]);
    }
"#;

#[derive(Clone, Debug)]
pub struct CompiledKernelBinding {
    handle: Arc<dyn RetainedKernelHandle>,
}

impl CompiledKernelBinding {
    pub fn new(handle: Arc<dyn RetainedKernelHandle>) -> Result<Self, DispatchFailure> {
        if handle.identity().is_empty() {
            return Err(DispatchFailure::InvalidKernelIdentity);
        }
        Ok(Self { handle })
    }

    pub fn identity(&self) -> &str {
        self.handle.identity()
    }

    pub fn retained_bytes(&self) -> usize {
        self.handle.retained_bytes()
    }
}

/// Geometry and mask metadata must be present before the cache is mutated.  `BlockCausal`
/// is Krea's existing block mask expressed analytically, not as an allocated mask tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionGeometry {
    pub batch: usize,
    pub heads: usize,
    pub query_tokens: usize,
    pub key_tokens: usize,
    pub head_dim: usize,
    pub query_start: usize,
    pub key_start: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KreaMask {
    None,
    BlockCausal { block_size: usize },
}

impl KreaMask {
    fn allows(self, geometry: AttentionGeometry, q: usize, k: usize) -> bool {
        match self {
            Self::None => true,
            Self::BlockCausal { block_size } => {
                block_size != 0
                    && (geometry.key_start + k) / block_size
                        <= (geometry.query_start + q) / block_size
            }
        }
    }
}

/// Why the unchanged dense route was selected.  These values are receipt fields, never
/// guessed by a caller after the fact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DispatchFailure {
    Disabled,
    Q4QualityArmNotAcknowledged,
    MissingCompiledHandle,
    InvalidKernelIdentity,
    UnsupportedBatch,
    UnsupportedHeadCount,
    UnsupportedHeadDimension,
    UnsupportedGroupSize,
    UnsupportedMask,
    GeometryMismatch,
    Cancelled,
}

impl DispatchFailure {
    pub const fn as_receipt_reason(&self) -> &'static str {
        match self {
            Self::Disabled => "experimental-disabled",
            Self::Q4QualityArmNotAcknowledged => "q4-quality-arm-not-acknowledged",
            Self::MissingCompiledHandle => "compiled-handle-missing",
            Self::InvalidKernelIdentity => "compiled-handle-identity-invalid",
            Self::UnsupportedBatch => "batch-unsupported",
            Self::UnsupportedHeadCount => "head-count-unsupported",
            Self::UnsupportedHeadDimension => "head-dimension-unsupported",
            Self::UnsupportedGroupSize => "group-size-unsupported",
            Self::UnsupportedMask => "mask-unsupported",
            Self::GeometryMismatch => "geometry-mismatch",
            Self::Cancelled => "cancelled",
        }
    }
}

/// The cache decision is made before a write, trim, or append.  Only `Compressed` permits
/// packed-state mutation; `DenseFallback` returns the exact reason to the existing route.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DispatchDecision {
    Compressed {
        geometry: AttentionGeometry,
        mask: KreaMask,
    },
    DenseFallback(DispatchFailure),
}

/// Source-owned numbers for a device receipt.  `elapsed_ns` measures only this reference
/// call and must never be presented as a Metal performance result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CompressedInstrumentation {
    pub persistent_bytes: usize,
    pub retained_handle_bytes: usize,
    pub bounded_scratch_bytes: usize,
    pub dense_window_bytes: usize,
    pub score_matrix_bytes: usize,
    pub compressed_dispatches: usize,
    pub dense_fallbacks: usize,
    pub cancellation_count: usize,
    pub elapsed_ns: u128,
    pub last_fallback_reason: Option<&'static str>,
}

/// A compact, owned `[B,H,S,D]` f32 tensor used only by deterministic source tests and
/// the independent CPU parity oracle.  It is not an MLX cache representation.
#[derive(Clone, Debug, PartialEq)]
pub struct CpuTensor {
    pub shape: [usize; 4],
    pub values: Vec<f32>,
}

impl CpuTensor {
    pub fn zeros(shape: [usize; 4]) -> Self {
        Self {
            shape,
            values: vec![0.0; shape.iter().product()],
        }
    }

    pub fn from_fn(
        shape: [usize; 4],
        mut f: impl FnMut(usize, usize, usize, usize) -> f32,
    ) -> Self {
        let mut out = Self::zeros(shape);
        for b in 0..shape[0] {
            for h in 0..shape[1] {
                for s in 0..shape[2] {
                    for d in 0..shape[3] {
                        out.set(b, h, s, d, f(b, h, s, d));
                    }
                }
            }
        }
        out
    }

    fn index(&self, b: usize, h: usize, s: usize, d: usize) -> usize {
        (((b * self.shape[1] + h) * self.shape[2] + s) * self.shape[3]) + d
    }

    pub fn get(&self, b: usize, h: usize, s: usize, d: usize) -> f32 {
        self.values[self.index(b, h, s, d)]
    }

    pub fn set(&mut self, b: usize, h: usize, s: usize, d: usize, value: f32) {
        let i = self.index(b, h, s, d);
        self.values[i] = value;
    }

    pub fn bytes(&self) -> usize {
        self.values.len() * std::mem::size_of::<f32>()
    }
}

/// Physical Krea affine representation: `words [B,H,S,D*bits/32]` plus BF16 scale/bias
/// `[B,H,S,D/group]`.  K and V intentionally share the D-axis grouping used by the live
/// cache, rather than impersonating VeloxQuant's token-axis K metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PackedAffineRows {
    batch: usize,
    heads: usize,
    tokens: usize,
    head_dim: usize,
    tier: CompressedTier,
    group_size: usize,
    words: Vec<u32>,
    scales_bf16: Vec<u16>,
    biases_bf16: Vec<u16>,
}

impl PackedAffineRows {
    fn pack(
        dense: &CpuTensor,
        tier: CompressedTier,
        group_size: usize,
    ) -> Result<Self, DispatchFailure> {
        let [batch, heads, tokens, head_dim] = dense.shape;
        if group_size == 0 || head_dim == 0 || !head_dim.is_multiple_of(group_size) {
            return Err(DispatchFailure::UnsupportedGroupSize);
        }
        let bits = tier.bits();
        if !(head_dim * bits).is_multiple_of(32) {
            return Err(DispatchFailure::UnsupportedHeadDimension);
        }
        let words_per_row = head_dim * bits / 32;
        let groups_per_row = head_dim / group_size;
        let rows = batch * heads * tokens;
        let mut packed = Self {
            batch,
            heads,
            tokens,
            head_dim,
            tier,
            group_size,
            words: vec![0; rows * words_per_row],
            scales_bf16: vec![0; rows * groups_per_row],
            biases_bf16: vec![0; rows * groups_per_row],
        };
        let levels = ((1usize << bits) - 1) as f32;
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..tokens {
                    for g in 0..groups_per_row {
                        let begin = g * group_size;
                        let mut min = f32::INFINITY;
                        let mut max = f32::NEG_INFINITY;
                        for d in begin..begin + group_size {
                            let value = dense.get(b, h, s, d);
                            min = min.min(value);
                            max = max.max(value);
                        }
                        let scale = if max > min { (max - min) / levels } else { 1.0 };
                        let meta = packed.meta_index(b, h, s, g);
                        packed.scales_bf16[meta] = f32_to_bf16(scale);
                        packed.biases_bf16[meta] = f32_to_bf16(min);
                        for d in begin..begin + group_size {
                            let raw = ((dense.get(b, h, s, d) - min) / scale).round();
                            let code = raw.clamp(0.0, levels) as u32;
                            packed.set_code(b, h, s, d, code);
                        }
                    }
                }
            }
        }
        Ok(packed)
    }

    fn words_per_row(&self) -> usize {
        self.head_dim * self.tier.bits() / 32
    }

    fn groups_per_row(&self) -> usize {
        self.head_dim / self.group_size
    }

    fn row_index(&self, b: usize, h: usize, s: usize) -> usize {
        s * self.batch * self.heads + b * self.heads + h
    }

    fn word_index(&self, b: usize, h: usize, s: usize, d: usize) -> (usize, u32) {
        let bits = self.tier.bits();
        let element_bit = d * bits;
        (
            self.row_index(b, h, s) * self.words_per_row() + element_bit / 32,
            (element_bit % 32) as u32,
        )
    }

    fn meta_index(&self, b: usize, h: usize, s: usize, group: usize) -> usize {
        self.row_index(b, h, s) * self.groups_per_row() + group
    }

    fn set_code(&mut self, b: usize, h: usize, s: usize, d: usize, code: u32) {
        let (word, shift) = self.word_index(b, h, s, d);
        let mask = (1u32 << self.tier.bits()) - 1;
        self.words[word] = (self.words[word] & !(mask << shift)) | ((code & mask) << shift);
    }

    fn decode(&self, b: usize, h: usize, s: usize, d: usize) -> f32 {
        let (word, shift) = self.word_index(b, h, s, d);
        let mask = (1u32 << self.tier.bits()) - 1;
        let code = (self.words[word] >> shift) & mask;
        let meta = self.meta_index(b, h, s, d / self.group_size);
        bf16_to_f32(self.scales_bf16[meta]) * code as f32 + bf16_to_f32(self.biases_bf16[meta])
    }

    fn bytes(&self) -> usize {
        self.words.len() * 4 + (self.scales_bf16.len() + self.biases_bf16.len()) * 2
    }

    fn validate_append(&self, next: &Self) -> Result<(), DispatchFailure> {
        if self.batch != next.batch
            || self.heads != next.heads
            || self.head_dim != next.head_dim
            || self.tier != next.tier
            || self.group_size != next.group_size
            || self.tokens.checked_add(next.tokens).is_none()
        {
            return Err(DispatchFailure::GeometryMismatch);
        }
        Ok(())
    }

    /// Extend the token axis without rebuilding the existing packed history. Rows are physically
    /// `[token, batch, head, payload]`, so a new chunk is contiguous and can be extended in O(chunk).
    fn append_rows(&mut self, next: &Self) -> Result<(), DispatchFailure> {
        self.validate_append(next)?;
        self.words.extend_from_slice(&next.words);
        self.scales_bf16.extend_from_slice(&next.scales_bf16);
        self.biases_bf16.extend_from_slice(&next.biases_bf16);
        self.tokens += next.tokens;
        Ok(())
    }
}

/// Persistent K/V state.  Append is transactional: both K and V are packed before the
/// state is changed, so a validation or cancellation error leaves the old cache untouched.
#[derive(Clone, Debug)]
pub struct CompressedKvCache {
    config: ExperimentalCompressedKvConfig,
    k: Option<PackedAffineRows>,
    v: Option<PackedAffineRows>,
    handle: Option<CompiledKernelBinding>,
    instrumentation: CompressedInstrumentation,
}

impl CompressedKvCache {
    pub fn new(config: ExperimentalCompressedKvConfig) -> Self {
        Self {
            config,
            k: None,
            v: None,
            handle: None,
            instrumentation: CompressedInstrumentation::default(),
        }
    }

    /// Bind only the pipeline compiled for this exact packed representation.  A nonempty
    /// arbitrary label is not enough: it would let a Q4 or different-group kernel read Q8 rows.
    pub fn bind_compiled_handle(
        &mut self,
        handle: CompiledKernelBinding,
    ) -> Result<(), DispatchFailure> {
        if handle.identity() != self.expected_handle_identity() {
            return Err(DispatchFailure::InvalidKernelIdentity);
        }
        self.instrumentation.retained_handle_bytes = handle.retained_bytes();
        self.handle = Some(handle);
        Ok(())
    }

    fn expected_handle_identity(&self) -> &'static str {
        match self.config.tier {
            CompressedTier::Q8 => "sc20684/krea-packed-affine-q8-d128-g64-v1",
            CompressedTier::Q4 => "sc20684/krea-packed-affine-q4-d128-g64-v1",
        }
    }

    pub fn stored_tokens(&self) -> usize {
        self.k.as_ref().map_or(0, |k| k.tokens)
    }

    pub fn instrumentation(&self) -> &CompressedInstrumentation {
        &self.instrumentation
    }

    /// Run an accepted fused dispatch with a dense retry seam. The packed callback is allowed to
    /// stage state, but any failure restores the pre-dispatch snapshot before the dense callback
    /// runs, so a backend fault cannot leave K/V half-updated or charge packed telemetry.
    pub fn dispatch_with_dense_retry<P, D>(
        &mut self,
        packed: P,
        dense: D,
    ) -> Result<CpuTensor, DispatchFailure>
    where
        P: FnOnce(&mut Self) -> Result<CpuTensor, DispatchFailure>,
        D: FnOnce(&Self) -> Result<CpuTensor, DispatchFailure>,
    {
        let before = self.clone();
        match packed(self) {
            Ok(result) => Ok(result),
            Err(error) => {
                *self = before;
                self.instrumentation.dense_fallbacks += 1;
                self.instrumentation.last_fallback_reason = Some(error.as_receipt_reason());
                dense(self)
            }
        }
    }

    pub fn decide(&mut self, geometry: AttentionGeometry, mask: KreaMask) -> DispatchDecision {
        let failure = (!self.config.enabled)
            .then_some(DispatchFailure::Disabled)
            .or_else(|| {
                (self.config.tier == CompressedTier::Q4 && !self.config.allow_q4_quality_arm)
                    .then_some(DispatchFailure::Q4QualityArmNotAcknowledged)
            })
            .or_else(|| {
                self.handle
                    .is_none()
                    .then_some(DispatchFailure::MissingCompiledHandle)
            })
            .or_else(|| (geometry.batch != 1).then_some(DispatchFailure::UnsupportedBatch))
            .or_else(|| (geometry.heads != 40).then_some(DispatchFailure::UnsupportedHeadCount))
            .or_else(|| {
                (geometry.head_dim != 128).then_some(DispatchFailure::UnsupportedHeadDimension)
            })
            .or_else(|| {
                (self.config.group_size != 64).then_some(DispatchFailure::UnsupportedGroupSize)
            })
            .or(match mask {
                KreaMask::None | KreaMask::BlockCausal { block_size: 1.. } => None,
                KreaMask::BlockCausal { block_size: 0 } => Some(DispatchFailure::UnsupportedMask),
            });
        match failure {
            Some(reason) => {
                self.instrumentation.dense_fallbacks += 1;
                self.instrumentation.last_fallback_reason = Some(reason.as_receipt_reason());
                DispatchDecision::DenseFallback(reason)
            }
            None => DispatchDecision::Compressed { geometry, mask },
        }
    }

    /// Commit a complete post-RoPE K/raw-V chunk.  The caller must have already received a
    /// `Compressed` decision; this method still verifies shape identity fail closed.
    pub fn append_after_decision(
        &mut self,
        decision: &DispatchDecision,
        keys: &CpuTensor,
        values: &CpuTensor,
    ) -> Result<(), DispatchFailure> {
        let DispatchDecision::Compressed { geometry, .. } = decision else {
            return Err(DispatchFailure::GeometryMismatch);
        };
        if keys.shape != values.shape
            || keys.shape
                != [
                    geometry.batch,
                    geometry.heads,
                    geometry.key_tokens,
                    geometry.head_dim,
                ]
        {
            return Err(DispatchFailure::GeometryMismatch);
        }
        let next_k = PackedAffineRows::pack(keys, self.config.tier, self.config.group_size)?;
        let next_v = PackedAffineRows::pack(values, self.config.tier, self.config.group_size)?;
        if let Some(old) = &self.k {
            if old.batch != next_k.batch
                || old.heads != next_k.heads
                || old.head_dim != next_k.head_dim
            {
                return Err(DispatchFailure::GeometryMismatch);
            }
        }
        if let Some(old) = &self.v {
            if old.batch != next_v.batch
                || old.heads != next_v.heads
                || old.head_dim != next_v.head_dim
            {
                return Err(DispatchFailure::GeometryMismatch);
            }
        }
        if let Some(old) = &self.k {
            old.validate_append(&next_k)?;
        }
        if let Some(old) = &self.v {
            old.validate_append(&next_v)?;
        }
        // Packing both chunks above is the validation/transaction boundary. Append each
        // already-packed head block directly; rebuilding the complete history on every decode
        // step would make repeated updates O(history²) and allocate a full-history copy.
        if let Some(existing) = &mut self.k {
            existing.append_rows(&next_k)?;
        } else {
            self.k = Some(next_k);
        }
        if let Some(existing) = &mut self.v {
            existing.append_rows(&next_v)?;
        } else {
            self.v = Some(next_v);
        }
        self.instrumentation.persistent_bytes = self.k.as_ref().map_or(0, PackedAffineRows::bytes)
            + self.v.as_ref().map_or(0, PackedAffineRows::bytes);
        Ok(())
    }

    /// Cancellation-safe prefix trim.  It copies only packed words/metadata needed for the
    /// retained tokens and never expands a K/V window.  A cancelled operation leaves state intact.
    pub fn trim_prefix(
        &mut self,
        keep_from: usize,
        cancel: &impl CancellationProbe,
    ) -> Result<(), DispatchFailure> {
        let (Some(k), Some(v)) = (&self.k, &self.v) else {
            return Ok(());
        };
        if cancel.cancelled() {
            self.instrumentation.cancellation_count += 1;
            return Err(DispatchFailure::Cancelled);
        }
        let start = keep_from.min(k.tokens);
        let next_k = slice_packed(k, start, cancel)?;
        let next_v = slice_packed(v, start, cancel)?;
        self.k = Some(next_k);
        self.v = Some(next_v);
        self.instrumentation.persistent_bytes = self.k.as_ref().map_or(0, PackedAffineRows::bytes)
            + self.v.as_ref().map_or(0, PackedAffineRows::bytes);
        Ok(())
    }

    /// CPU oracle for the retained kernel contract.  It consumes packed rows directly and keeps
    /// running max/sum/output for each query; it never allocates a dense K/V read or score matrix.
    pub fn tiled_online_attention(
        &mut self,
        query: &CpuTensor,
        geometry: AttentionGeometry,
        mask: KreaMask,
        cancel: &impl CancellationProbe,
    ) -> Result<CpuTensor, DispatchFailure> {
        if !matches!(
            self.decide(geometry, mask),
            DispatchDecision::Compressed { .. }
        ) {
            return Err(DispatchFailure::GeometryMismatch);
        }
        let (Some(keys), Some(values)) = (&self.k, &self.v) else {
            return Err(DispatchFailure::GeometryMismatch);
        };
        if query.shape
            != [
                geometry.batch,
                geometry.heads,
                geometry.query_tokens,
                geometry.head_dim,
            ]
            || keys.tokens != geometry.key_tokens
            || values.tokens != geometry.key_tokens
            || keys.heads != geometry.heads
            || values.heads != geometry.heads
        {
            return Err(DispatchFailure::GeometryMismatch);
        }
        let started = Instant::now();
        let _guard = ScratchGuard::new();
        self.instrumentation.bounded_scratch_bytes =
            TILE_ROWS * geometry.head_dim * std::mem::size_of::<f32>();
        self.instrumentation.dense_window_bytes = 0;
        self.instrumentation.score_matrix_bytes = 0;
        let mut out = CpuTensor::zeros(query.shape);
        for b in 0..geometry.batch {
            for h in 0..geometry.heads {
                for tile_start in (0..geometry.query_tokens).step_by(TILE_ROWS) {
                    let tile_end = (tile_start + TILE_ROWS).min(geometry.query_tokens);
                    for q in tile_start..tile_end {
                        if cancel.cancelled() {
                            self.instrumentation.cancellation_count += 1;
                            return Err(DispatchFailure::Cancelled);
                        }
                        let mut max = f32::NEG_INFINITY;
                        let mut sum = 0.0f32;
                        let mut acc = vec![0.0f32; geometry.head_dim];
                        for k in 0..geometry.key_tokens {
                            if !mask.allows(geometry, q, k) {
                                continue;
                            }
                            let mut dot = 0.0;
                            for d in 0..geometry.head_dim {
                                dot += query.get(b, h, q, d) * keys.decode(b, h, k, d);
                            }
                            let score = dot / (geometry.head_dim as f32).sqrt();
                            let next_max = max.max(score);
                            let old_weight = if max.is_finite() {
                                (max - next_max).exp()
                            } else {
                                0.0
                            };
                            let new_weight = (score - next_max).exp();
                            sum = sum * old_weight + new_weight;
                            for (d, accumulated) in acc.iter_mut().enumerate() {
                                *accumulated = *accumulated * old_weight
                                    + new_weight * values.decode(b, h, k, d);
                            }
                            max = next_max;
                        }
                        if sum != 0.0 {
                            for (d, accumulated) in acc.iter().enumerate() {
                                out.set(b, h, q, d, *accumulated / sum);
                            }
                        }
                    }
                }
            }
        }
        self.instrumentation.compressed_dispatches += 1;
        self.instrumentation.elapsed_ns = started.elapsed().as_nanos();
        Ok(out)
    }
}

/// The bounded number of query rows a retained threadgroup owns at once.
pub const TILE_ROWS: usize = 8;
/// The bounded number of packed-history key rows streamed through a threadgroup at once.
pub const TILE_KEYS: usize = 8;
/// Exact threadgroup storage declared by the fixed D=128 Metal source: Q/K/V tiles, the 8x8 score
/// tile, and three 8-row online-softmax vectors. It is attributable to a receipt only after the
/// corresponding packed output has materialized successfully.
pub const PACKED_METAL_THREADGROUP_SCRATCH_BYTES: usize =
    (3 * TILE_ROWS * 128 + TILE_ROWS * TILE_KEYS + 3 * TILE_ROWS) * std::mem::size_of::<f32>();

pub trait CancellationProbe {
    fn cancelled(&self) -> bool;
}

impl CancellationProbe for bool {
    fn cancelled(&self) -> bool {
        *self
    }
}

/// Test-visible accounting for automatic scratch cleanup on cancellation and errors.
static OUTSTANDING_SCRATCH: AtomicUsize = AtomicUsize::new(0);
struct ScratchGuard;
impl ScratchGuard {
    fn new() -> Self {
        OUTSTANDING_SCRATCH.fetch_add(1, Ordering::SeqCst);
        Self
    }
}
impl Drop for ScratchGuard {
    fn drop(&mut self) {
        OUTSTANDING_SCRATCH.fetch_sub(1, Ordering::SeqCst);
    }
}
#[cfg(test)]
fn outstanding_scratch() -> usize {
    OUTSTANDING_SCRATCH.load(Ordering::SeqCst)
}

fn slice_packed(
    rows: &PackedAffineRows,
    start: usize,
    cancel: &impl CancellationProbe,
) -> Result<PackedAffineRows, DispatchFailure> {
    let tokens = rows.tokens - start;
    let mut next = PackedAffineRows {
        batch: rows.batch,
        heads: rows.heads,
        tokens,
        head_dim: rows.head_dim,
        tier: rows.tier,
        group_size: rows.group_size,
        words: Vec::with_capacity(rows.batch * rows.heads * tokens * rows.words_per_row()),
        scales_bf16: Vec::with_capacity(rows.batch * rows.heads * tokens * rows.groups_per_row()),
        biases_bf16: Vec::with_capacity(rows.batch * rows.heads * tokens * rows.groups_per_row()),
    };
    for s in start..rows.tokens {
        for b in 0..rows.batch {
            for h in 0..rows.heads {
                if cancel.cancelled() {
                    return Err(DispatchFailure::Cancelled);
                }
                let row = rows.row_index(b, h, s);
                let word_start = row * rows.words_per_row();
                next.words
                    .extend_from_slice(&rows.words[word_start..word_start + rows.words_per_row()]);
                let meta_start = row * rows.groups_per_row();
                next.scales_bf16.extend_from_slice(
                    &rows.scales_bf16[meta_start..meta_start + rows.groups_per_row()],
                );
                next.biases_bf16.extend_from_slice(
                    &rows.biases_bf16[meta_start..meta_start + rows.groups_per_row()],
                );
            }
        }
    }
    Ok(next)
}

fn f32_to_bf16(value: f32) -> u16 {
    ((value
        .to_bits()
        .wrapping_add(0x7fff + ((value.to_bits() >> 16) & 1)))
        >> 16) as u16
}
fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Handle(CompressedTier);
    impl RetainedKernelHandle for Handle {
        fn identity(&self) -> &str {
            match self.0 {
                CompressedTier::Q8 => "sc20684/krea-packed-affine-q8-d128-g64-v1",
                CompressedTier::Q4 => "sc20684/krea-packed-affine-q4-d128-g64-v1",
            }
        }
        fn retained_bytes(&self) -> usize {
            128
        }
    }
    fn cache(tier: CompressedTier) -> CompressedKvCache {
        let mut cache = CompressedKvCache::new(ExperimentalCompressedKvConfig {
            enabled: true,
            tier,
            allow_q4_quality_arm: tier == CompressedTier::Q4,
            group_size: 64,
        });
        cache
            .bind_compiled_handle(CompiledKernelBinding::new(Arc::new(Handle(tier))).unwrap())
            .unwrap();
        cache
    }
    fn tensor(tokens: usize, salt: f32) -> CpuTensor {
        CpuTensor::from_fn([1, 40, tokens, 128], |_, h, s, d| {
            let x = ((s * 37 + d * 11 + h * 19) % 97) as f32 / 31.0 - 1.5;
            if d % 63 == 0 {
                x * 17.0 + salt
            } else {
                x + salt
            }
        })
    }
    fn geometry(q: usize, k: usize) -> AttentionGeometry {
        AttentionGeometry {
            batch: 1,
            heads: 40,
            query_tokens: q,
            key_tokens: k,
            head_dim: 128,
            query_start: 0,
            key_start: 0,
        }
    }
    fn append(cache: &mut CompressedKvCache, keys: &CpuTensor, values: &CpuTensor) {
        let decision = cache.decide(geometry(1, keys.shape[2]), KreaMask::None);
        cache
            .append_after_decision(&decision, keys, values)
            .unwrap();
    }

    #[test]
    fn q8_packed_attention_matches_independent_dequantize_then_fp32_reference_for_tiles_tails_masks_and_outliers(
    ) {
        let keys = tensor(35, 0.0);
        let values = tensor(35, 0.5);
        let query = tensor(37, -0.25);
        let mut cache = cache(CompressedTier::Q8);
        append(&mut cache, &keys, &values);
        let actual = cache
            .tiled_online_attention(
                &query,
                geometry(37, 35),
                KreaMask::BlockCausal { block_size: 7 },
                &false,
            )
            .unwrap();
        // The fused-path contract is parity with an independent dequantize-then-fp32-attend
        // reference. Comparing directly with the pre-quantized f32 fixtures would instead fold
        // Q8/BF16-metadata quality loss into the kernel-parity assertion and can hide a correct
        // packed reader behind an invalid tolerance.
        let decoded_keys = CpuTensor::from_fn(keys.shape, |b, h, s, d| {
            cache.k.as_ref().unwrap().decode(b, h, s, d)
        });
        let decoded_values = CpuTensor::from_fn(values.shape, |b, h, s, d| {
            cache.v.as_ref().unwrap().decode(b, h, s, d)
        });
        let expected = dense_reference(
            &query,
            &decoded_keys,
            &decoded_values,
            geometry(37, 35),
            KreaMask::BlockCausal { block_size: 7 },
        );
        assert_max_error(&actual, &expected, 1e-5);
        let stats = cache.instrumentation();
        assert_eq!(stats.dense_window_bytes, 0);
        assert_eq!(stats.score_matrix_bytes, 0);
        assert_eq!(stats.bounded_scratch_bytes, TILE_ROWS * 128 * 4);
    }

    #[test]
    fn arbitrary_append_boundaries_and_trim_are_bit_stable() {
        let keys = tensor(19, 0.0);
        let values = tensor(19, 0.5);
        let mut one = cache(CompressedTier::Q8);
        append(&mut one, &keys, &values);
        let mut chunks = cache(CompressedTier::Q8);
        for &(start, end) in &[(0, 1), (1, 8), (8, 11), (11, 19)] {
            append(
                &mut chunks,
                &slice_dense(&keys, start, end),
                &slice_dense(&values, start, end),
            );
        }
        assert_eq!(one.k, chunks.k);
        assert_eq!(one.v, chunks.v);
        chunks.trim_prefix(3, &false).unwrap();
        assert_eq!(chunks.stored_tokens(), 16);
        assert!(chunks.instrumentation().persistent_bytes > 0);
    }

    #[test]
    fn unsupported_or_disabled_paths_do_not_mutate_packed_state_and_cancellation_cleans_scratch() {
        let mut disabled = CompressedKvCache::new(ExperimentalCompressedKvConfig::default());
        assert_eq!(
            disabled.decide(geometry(1, 1), KreaMask::None),
            DispatchDecision::DenseFallback(DispatchFailure::Disabled)
        );
        assert_eq!(disabled.stored_tokens(), 0);
        let keys = tensor(2, 0.0);
        let values = tensor(2, 0.0);
        let query = tensor(2, 0.0);
        let mut live = cache(CompressedTier::Q8);
        append(&mut live, &keys, &values);
        assert_eq!(
            live.tiled_online_attention(&query, geometry(2, 2), KreaMask::None, &true),
            Err(DispatchFailure::Cancelled)
        );
        assert_eq!(outstanding_scratch(), 0);
    }

    #[test]
    fn injected_dispatch_fault_retries_dense_without_partial_packed_mutation() {
        let keys = tensor(3, 0.0);
        let values = tensor(3, 0.5);
        let mut cache = cache(CompressedTier::Q8);
        append(&mut cache, &keys, &values);
        let stored_before = cache.stored_tokens();
        let packed_before = cache.instrumentation().compressed_dispatches;
        let dense_result = tensor(1, 9.0);
        let actual = cache
            .dispatch_with_dense_retry(
                |state| {
                    state.k = None;
                    state.instrumentation.compressed_dispatches += 1;
                    Err(DispatchFailure::GeometryMismatch)
                },
                |_| Ok(dense_result.clone()),
            )
            .unwrap();
        assert_eq!(actual, dense_result);
        assert_eq!(cache.stored_tokens(), stored_before);
        assert_eq!(cache.instrumentation().compressed_dispatches, packed_before);
        assert_eq!(cache.instrumentation().dense_fallbacks, 1);
        assert_eq!(
            cache.instrumentation().last_fallback_reason,
            Some("geometry-mismatch")
        );
    }

    #[test]
    fn handle_identity_is_bound_to_the_exact_packed_representation() {
        let mut cache = CompressedKvCache::new(ExperimentalCompressedKvConfig {
            enabled: true,
            tier: CompressedTier::Q8,
            allow_q4_quality_arm: false,
            group_size: 64,
        });
        let q4_handle = CompiledKernelBinding::new(Arc::new(Handle(CompressedTier::Q4))).unwrap();
        assert_eq!(
            cache.bind_compiled_handle(q4_handle),
            Err(DispatchFailure::InvalidKernelIdentity)
        );
        assert!(matches!(
            cache.decide(geometry(1, 1), KreaMask::None),
            DispatchDecision::DenseFallback(DispatchFailure::MissingCompiledHandle)
        ));
    }

    #[test]
    fn q4_requires_a_separate_quality_arm_even_with_a_retained_handle() {
        let mut cache = CompressedKvCache::new(ExperimentalCompressedKvConfig {
            enabled: true,
            tier: CompressedTier::Q4,
            allow_q4_quality_arm: false,
            group_size: 64,
        });
        cache
            .bind_compiled_handle(
                CompiledKernelBinding::new(Arc::new(Handle(CompressedTier::Q4))).unwrap(),
            )
            .unwrap();
        assert_eq!(
            cache.decide(geometry(1, 1), KreaMask::None),
            DispatchDecision::DenseFallback(DispatchFailure::Q4QualityArmNotAcknowledged)
        );
        assert_eq!(cache.stored_tokens(), 0);
    }

    #[test]
    fn real_metal_kernel_jits_and_matches_block_causal_uniform_value_oracle() {
        const HEADS: usize = 40;
        const DIM: usize = 128;
        const HISTORY: usize = 9;
        const SQ: usize = 9;
        const GROUPS: usize = DIM / 64;

        let bf16 = |values: Vec<f32>, shape: &[i32]| {
            Array::from_slice(&values, shape)
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        let q = bf16(vec![0.0; HEADS * SQ * DIM], &[1, 40, 9, 128]);
        let current_k = bf16(vec![0.0; HEADS * SQ * DIM], &[1, 40, 9, 128]);
        let mut current_v_values = vec![0.0; HEADS * SQ * DIM];
        for h in 0..HEADS {
            for s in 0..SQ {
                for d in 0..DIM {
                    current_v_values[(h * SQ + s) * DIM + d] = 100.0 + s as f32;
                }
            }
        }
        let current_v = bf16(current_v_values, &[1, 40, 9, 128]);
        let query_positions = Array::from_slice(&(9i64..18).collect::<Vec<_>>(), &[9]);
        let key_positions = Array::from_slice(&(0i64..18).collect::<Vec<_>>(), &[18]);

        for (tier, bits) in [(CompressedTier::Q8, 8usize), (CompressedTier::Q4, 4)] {
            let words = DIM * bits / 32;
            let kw = Array::from_slice(
                &vec![0u32; HEADS * HISTORY * words],
                &[1, 40, 9, words as i32],
            );
            let mut packed_values = vec![0u32; HEADS * HISTORY * words];
            for h in 0..HEADS {
                for s in 0..HISTORY {
                    let code = (s + 1) as u32;
                    let mut word = 0u32;
                    for field in 0..(32 / bits) {
                        word |= code << (field * bits);
                    }
                    packed_values[(h * HISTORY + s) * words..(h * HISTORY + s + 1) * words]
                        .fill(word);
                }
            }
            let vw = Array::from_slice(&packed_values, &[1, 40, 9, words as i32]);
            let zero_meta = bf16(vec![0.0; HEADS * HISTORY * GROUPS], &[1, 40, 9, 2]);
            let unit_scales = bf16(vec![1.0; HEADS * HISTORY * GROUPS], &[1, 40, 9, 2]);
            let kernel = KreaPackedMetalKernel::new(tier).unwrap();
            let dispatch = || {
                kernel
                    .dispatch(
                        &q,
                        &kw,
                        &zero_meta,
                        &zero_meta,
                        &vw,
                        &unit_scales,
                        &zero_meta,
                        &current_k,
                        &current_v,
                        &query_positions,
                        &key_positions,
                        4,
                    )
                    .unwrap()
            };

            // The first evaluation forces lazy Metal compilation; the second covers the retained
            // steady dispatch. SQ=9 exercises both a full query tile and the zero-filled tail tile.
            for output in [dispatch(), dispatch()] {
                let output = output.as_dtype(Dtype::Float32).unwrap();
                output.eval().unwrap();
                let values = output.as_slice::<f32>();
                for h in 0..HEADS {
                    for qi in 0..SQ {
                        let q_position = 9 + qi;
                        let allowed_current = ((q_position / 4 + 1) * 4).min(18) - 9;
                        let expected_sum = (1..=HISTORY).map(|v| v as f32).sum::<f32>()
                            + (0..allowed_current).map(|s| 100.0 + s as f32).sum::<f32>();
                        let expected = expected_sum / (HISTORY + allowed_current) as f32;
                        for d in 0..DIM {
                            let actual = values[(h * SQ + qi) * DIM + d];
                            assert!(
                                (actual - expected).abs() <= 0.25,
                                "tier={tier:?} h={h} qi={qi} d={d}: {actual} != {expected}"
                            );
                        }
                    }
                }
            }
        }
    }

    fn dense_reference(
        q: &CpuTensor,
        k: &CpuTensor,
        v: &CpuTensor,
        g: AttentionGeometry,
        mask: KreaMask,
    ) -> CpuTensor {
        let mut out = CpuTensor::zeros(q.shape);
        for h in 0..g.heads {
            for qi in 0..g.query_tokens {
                let mut scores = Vec::new();
                for ki in 0..g.key_tokens {
                    if mask.allows(g, qi, ki) {
                        let mut dot = 0.0;
                        for d in 0..g.head_dim {
                            dot += q.get(0, h, qi, d) * k.get(0, h, ki, d);
                        }
                        scores.push((ki, dot / (g.head_dim as f32).sqrt()));
                    }
                }
                let max = scores
                    .iter()
                    .map(|(_, s)| *s)
                    .fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|(_, s)| (*s - max).exp()).sum();
                for d in 0..g.head_dim {
                    let value: f32 = scores
                        .iter()
                        .map(|(ki, s)| (*s - max).exp() * v.get(0, h, *ki, d))
                        .sum();
                    out.set(0, h, qi, d, value / sum);
                }
            }
        }
        out
    }
    fn slice_dense(input: &CpuTensor, start: usize, end: usize) -> CpuTensor {
        CpuTensor::from_fn(
            [1, input.shape[1], end - start, input.shape[3]],
            |b, h, s, d| input.get(b, h, start + s, d),
        )
    }
    fn assert_max_error(a: &CpuTensor, b: &CpuTensor, tolerance: f32) {
        let max = a
            .values
            .iter()
            .zip(&b.values)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(max <= tolerance, "max error {max} > {tolerance}");
    }
}
