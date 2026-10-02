//! Backend-neutral compressed-KV-cache policy (epic sc-20669, story sc-20679).
//!
//! A product opts a generation into the compressed KV cache with
//! [`TextLlmRequest::kv_compression`](crate::TextLlmRequest::kv_compression) (off by default). The
//! backend then runs it compressed **only** when [`qualify_kv_compression`] admits the request
//! against the one checked-in table, [`KV_COMPRESSION_QUALIFICATIONS`]; every other request runs
//! dense and says why with a [`KvCacheFallbackReason`]. The outcome is reported per generation on
//! [`TextLlmOutput::kv_cache`](crate::TextLlmOutput::kv_cache) as a [`KvCacheReport`].
//!
//! Every type here is tensor-free so the Candle backend and the product can share the exact
//! eligibility semantics; only a backend with a fused compressed-domain reader (MLX Metal today)
//! can run the compressed path. There is no universal default: a model family, format and context
//! range absent from the table is never compressed.

/// Version of the compressed KV representation and fused-reader semantics the qualification table
/// speaks about: packed group-affine codes, 32-element groups (K groups span tokens, V groups span
/// channels), f16 scale/zero rounded as the SC-20676 quantizer stores them, read by the fused
/// decode kernels. A change to any of those is a new format version, and every row of
/// [`KV_COMPRESSION_QUALIFICATIONS`] must be requalified before it applies to it.
pub const KV_CACHE_FORMAT_VERSION: u32 = 1;

/// The product's opt-in for the compressed KV cache on one generation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KvCompressionPolicy {
    /// Always dense (the default). Reported as [`KvCacheFallbackReason::PolicyDisabled`].
    #[default]
    Off,
    /// Compressed where [`KV_COMPRESSION_QUALIFICATIONS`] admits the request; dense, with a
    /// reason, everywhere else.
    Qualified,
}

/// A compressed KV representation a backend can run with a fused compressed-domain reader.
///
/// Only formats with qualifying evidence are listed: the 4-bit (K4V4) and 2-bit group-affine
/// candidates were No-Go in the SC-20669 proof of concept and are deliberately not advertised.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvCompressionFormat {
    /// Packed group-affine, 8-bit key and value codes, group 32, f16 scale/zero.
    GroupAffineK8V8,
}

impl KvCompressionFormat {
    /// Stable lower-case label.
    pub const fn id(self) -> &'static str {
        match self {
            Self::GroupAffineK8V8 => "group-affine-k8v8",
        }
    }

    /// Code width of the stored keys.
    pub const fn key_bits(self) -> u8 {
        match self {
            Self::GroupAffineK8V8 => 8,
        }
    }

    /// Code width of the stored values.
    pub const fn value_bits(self) -> u8 {
        match self {
            Self::GroupAffineK8V8 => 8,
        }
    }

    /// Elements sharing one scale/zero pair.
    pub const fn group_size(self) -> usize {
        match self {
            Self::GroupAffineK8V8 => 32,
        }
    }

    /// Bytes of one group's affine metadata: an f16 scale and an f16 zero.
    pub const fn group_metadata_bytes(self) -> u64 {
        match self {
            Self::GroupAffineK8V8 => 2 * 2,
        }
    }

    /// Bytes the packed representation stores for one layer and one KV head at
    /// `capacity_tokens` positions: key and value codes plus their scale/zero metadata. K groups
    /// span tokens (one scale/zero pair per channel per `group_size` tokens); V groups span
    /// channels (one pair per `group_size` channels of every token). `capacity_tokens` is a
    /// multiple of [`Self::group_size`] for a block-allocated cache; it is rounded up here so a
    /// partial group is never under-priced.
    pub fn packed_bytes_per_head(self, capacity_tokens: u64, head_dim: u64) -> Option<u64> {
        let group = self.group_size() as u64;
        let key_groups = capacity_tokens.div_ceil(group);
        let key_codes =
            key_groups.checked_mul(code_bytes(group.checked_mul(head_dim)?, self.key_bits())?)?;
        let key_metadata = key_groups
            .checked_mul(head_dim)?
            .checked_mul(self.group_metadata_bytes())?;
        let value_codes = capacity_tokens.checked_mul(code_bytes(head_dim, self.value_bits())?)?;
        let value_metadata = capacity_tokens
            .checked_mul(head_dim.div_ceil(group))?
            .checked_mul(self.group_metadata_bytes())?;
        key_codes
            .checked_add(key_metadata)?
            .checked_add(value_codes)?
            .checked_add(value_metadata)
    }
}

/// Bytes holding `codes` packed codes of `bits` width.
fn code_bytes(codes: u64, bits: u8) -> Option<u64> {
    Some(codes.checked_mul(u64::from(bits))?.div_ceil(8))
}

/// The KV geometry of a loaded decoder, as cache pricing reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvCacheShape {
    /// Decoder layers holding a K/V cache.
    pub layers: u64,
    /// K/V heads per layer.
    pub kv_heads: u64,
    /// Channels per head.
    pub head_dim: u64,
    /// Scalar width of the dense K/V (the decoder's compute dtype). A compressed cache keeps its
    /// incomplete group's residual rows at this width.
    pub element_bytes: u64,
}

impl KvCacheShape {
    /// Bytes of a dense K/V cache holding `tokens` positions (K and V for every layer and head).
    pub fn dense_bytes(self, tokens: u64) -> Option<u64> {
        tokens
            .checked_mul(self.layers)?
            .checked_mul(self.kv_heads)?
            .checked_mul(self.head_dim)?
            .checked_mul(self.element_bytes)?
            .checked_mul(2)
    }
}

/// How a backend's compressed cache allocates and reads, beyond what the format itself fixes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompressedKvAllocation {
    /// Positions the packed arrays grow by at a time; capacity is a whole number of these blocks
    /// (rounded up to whole groups), never smaller than one block.
    pub growth_block_tokens: u64,
    /// Layers whose per-layer packed transient (a block growth's pre-growth arrays beside their
    /// successors, or a group flush's freshly quantized codes before they are written in place)
    /// the backend's evaluator can hold at once.
    pub coexisting_layer_transients: u64,
    /// Scratch one layer's fused-reader dispatch holds (split-KV partials), priced for every
    /// layer.
    pub reader_scratch_bytes_per_layer: u64,
}

/// Bytes a compressed KV cache of `format` needs to serve `tokens` positions (prompt plus every
/// generated token), derived from the format rather than a fixed ratio:
///
/// * resident: per layer and KV head, the packed codes and scale/zero metadata at the
///   block-rounded capacity ([`KvCompressionFormat::packed_bytes_per_head`]) plus one group of
///   dense K and V residual rows at the decoder's compute width;
/// * transient: `coexisting_layer_transients` layers' packed arrays at full capacity plus one block
///   (the pre-growth copy or the flush output beside the live arrays), every layer's residual
///   rows once more (a step's rollback point holds the residuals a flush replaced), and every
///   layer's fused-reader scratch.
///
/// `None` on overflow or a zero growth block.
pub fn compressed_kv_cache_bytes(
    format: KvCompressionFormat,
    shape: KvCacheShape,
    tokens: u64,
    allocation: CompressedKvAllocation,
) -> Option<u64> {
    let group = format.group_size() as u64;
    if allocation.growth_block_tokens == 0 {
        return None;
    }
    let block = allocation
        .growth_block_tokens
        .div_ceil(group)
        .checked_mul(group)?;
    let capacity = tokens.div_ceil(block).max(1).checked_mul(block)?;
    let packed_layer = format
        .packed_bytes_per_head(capacity, shape.head_dim)?
        .checked_mul(shape.kv_heads)?;
    let residual_layer = group
        .checked_mul(shape.head_dim)?
        .checked_mul(shape.element_bytes)?
        .checked_mul(2)?
        .checked_mul(shape.kv_heads)?;
    let resident = packed_layer
        .checked_add(residual_layer)?
        .checked_mul(shape.layers)?;
    let block_layer = format
        .packed_bytes_per_head(block, shape.head_dim)?
        .checked_mul(shape.kv_heads)?;
    let transient = packed_layer
        .checked_add(block_layer)?
        .checked_mul(allocation.coexisting_layer_transients.min(shape.layers))?
        .checked_add(residual_layer.checked_mul(shape.layers)?)?
        .checked_add(
            allocation
                .reader_scratch_bytes_per_layer
                .checked_mul(shape.layers)?,
        )?;
    resident.checked_add(transient)
}

/// A decoder family the qualification table can name. A backend maps its loaded architecture onto
/// one of these only when the checkpoint is that family's plain causal decoder; anything else
/// (hybrid, multimodal-wrapped, Prism, other architectures) has no family and is never qualified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvModelFamily {
    /// Llama (`LlamaForCausalLM`).
    Llama,
    /// Qwen3 dense causal decoder (`Qwen3ForCausalLM`) — not the hybrid Qwen3.5/3.6 decoder.
    Qwen3,
}

impl KvModelFamily {
    /// Stable lower-case label.
    pub const fn id(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Qwen3 => "qwen3",
        }
    }
}

/// One qualified (family × format × context range) combination and the evidence behind it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvQualification {
    /// The decoder family.
    pub family: KvModelFamily,
    /// The compressed representation it qualified with.
    pub format: KvCompressionFormat,
    /// Smallest context (tokens prefilled before decode starts) that runs compressed. Shorter
    /// contexts decode slower compressed than dense, so they stay dense.
    pub min_context_tokens: u64,
    /// Exclusive upper bound of the qualified context, `None` when the evidence covers the model's
    /// own context window.
    pub max_context_tokens: Option<u64>,
    /// Where the qualification comes from.
    pub evidence: &'static str,
}

impl KvQualification {
    /// Whether `context_tokens` lies in this row's qualified range.
    pub fn admits(&self, context_tokens: u64) -> bool {
        context_tokens >= self.min_context_tokens
            && self
                .max_context_tokens
                .is_none_or(|max| context_tokens < max)
    }
}

/// The one checked-in qualification table (format version [`KV_CACHE_FORMAT_VERSION`]).
///
/// Thresholds are the SC-20671 campaign coordinates the evidence was measured at
/// (`context_band_target` in `mlx-llm`'s campaign): memory-material is a quarter of the evidence
/// model's native window and fit-boundary is `max(window − 512, ⌈0.9 · window⌉)`. Qwen3-1.7B
/// (40 960-token window) qualified at both coordinates, so it runs compressed from its
/// memory-material coordinate (10 240) up. Llama-3.2-3B (131 072-token window) qualified at
/// memory-material (32 768); its fit-boundary coordinate (130 560) awaits a dense multi-turn
/// noise-floor run, so contexts from there up stay dense until that row is flipped to `None`.
pub const KV_COMPRESSION_QUALIFICATIONS: &[KvQualification] = &[
    KvQualification {
        family: KvModelFamily::Llama,
        format: KvCompressionFormat::GroupAffineK8V8,
        min_context_tokens: 32_768,
        max_context_tokens: Some(130_560),
        evidence: "sc-20669 A2 run 37004025116 + dense noise floor 37035827730: \
                   Llama-3.2-3B-Instruct-4bit memory-material (32768) passes at group-affine-8; \
                   fit-boundary (130560) pending a dense multi-turn noise-floor run",
    },
    KvQualification {
        family: KvModelFamily::Qwen3,
        format: KvCompressionFormat::GroupAffineK8V8,
        min_context_tokens: 10_240,
        max_context_tokens: None,
        evidence: "sc-20669 A2 run 37004025116 + dense noise floor 37035827730: \
                   Qwen3-1.7B-4bit memory-material (10240) and fit-boundary (40448) pass at \
                   group-affine-8",
    },
];

/// Why a generation ran (or finished) on the dense KV cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvCacheFallbackReason {
    /// The request did not opt in ([`KvCompressionPolicy::Off`]).
    PolicyDisabled,
    /// The loaded model's family has no row in the qualification table.
    UnqualifiedModel,
    /// The request takes a path the compressed cache is not wired through (multimodal prefill,
    /// multi-token prediction, a hybrid recurrent decoder).
    UnsupportedRequest,
    /// More than one sequence decodes together: batched prefill attends through additive padding
    /// masks, which the fused compressed reader cannot apply.
    BatchedDecode,
    /// The context is below the family's qualified minimum (short contexts decode slower
    /// compressed than dense).
    BelowMinimumContext,
    /// The context is at or above the family's qualified range.
    AboveQualifiedContext,
    /// The model's attention geometry is outside what the fused reader implements (head
    /// dimension, attention-score soft-cap, shared K/V layers).
    UnsupportedGeometry,
    /// No fused compressed reader is available on this backend/host (a backend without one, or
    /// a reader that failed to build or bind).
    ReaderUnavailable,
    /// The generation started compressed and the cache explicitly transitioned to dense part-way
    /// (for example a reader dispatch fault); the backend's detail names the operation.
    RuntimeFallback,
}

impl KvCacheFallbackReason {
    /// Every reason, in declaration order.
    pub const ALL: [Self; 9] = [
        Self::PolicyDisabled,
        Self::UnqualifiedModel,
        Self::UnsupportedRequest,
        Self::BatchedDecode,
        Self::BelowMinimumContext,
        Self::AboveQualifiedContext,
        Self::UnsupportedGeometry,
        Self::ReaderUnavailable,
        Self::RuntimeFallback,
    ];

    /// Stable lower-case label a product renders as-is.
    pub const fn id(self) -> &'static str {
        match self {
            Self::PolicyDisabled => "policy_disabled",
            Self::UnqualifiedModel => "unqualified_model",
            Self::UnsupportedRequest => "unsupported_request",
            Self::BatchedDecode => "batched_decode",
            Self::BelowMinimumContext => "below_minimum_context",
            Self::AboveQualifiedContext => "above_qualified_context",
            Self::UnsupportedGeometry => "unsupported_geometry",
            Self::ReaderUnavailable => "reader_unavailable",
            Self::RuntimeFallback => "runtime_fallback",
        }
    }
}

/// Decide one request against [`KV_COMPRESSION_QUALIFICATIONS`]: the qualifying row, or why the
/// request stays dense. `family` is `None` for a model the backend cannot name as a table family;
/// `context_tokens` is the number of tokens prefilled before decoding starts; `batch` the number
/// of sequences decoding together. The checks run in a fixed order — policy, batch, family,
/// context — so the reported reason is deterministic.
pub fn qualify_kv_compression(
    policy: KvCompressionPolicy,
    family: Option<KvModelFamily>,
    context_tokens: u64,
    batch: u64,
) -> Result<&'static KvQualification, KvCacheFallbackReason> {
    qualify_against(
        KV_COMPRESSION_QUALIFICATIONS,
        policy,
        family,
        context_tokens,
        batch,
    )
}

fn qualify_against(
    table: &'static [KvQualification],
    policy: KvCompressionPolicy,
    family: Option<KvModelFamily>,
    context_tokens: u64,
    batch: u64,
) -> Result<&'static KvQualification, KvCacheFallbackReason> {
    if policy == KvCompressionPolicy::Off {
        return Err(KvCacheFallbackReason::PolicyDisabled);
    }
    if batch != 1 {
        return Err(KvCacheFallbackReason::BatchedDecode);
    }
    let rows = table
        .iter()
        .filter(|row| Some(row.family) == family)
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return Err(KvCacheFallbackReason::UnqualifiedModel);
    }
    if let Some(row) = rows.iter().find(|row| row.admits(context_tokens)) {
        return Ok(row);
    }
    if rows
        .iter()
        .all(|row| context_tokens < row.min_context_tokens)
    {
        Err(KvCacheFallbackReason::BelowMinimumContext)
    } else {
        Err(KvCacheFallbackReason::AboveQualifiedContext)
    }
}

/// Counters of one generation's KV cache, measured by the backend. All zero on a dense run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvCacheCounters {
    /// Attention calls the fused compressed-domain reader served.
    pub fused_attention_calls: u64,
    /// Explicit dense transitions the compressed cache recorded.
    pub dense_fallback_events: u64,
    /// Full-cache dense reconstructions (a dense transition rebuilding resident history).
    pub full_cache_dequantizations: u64,
    /// Device bytes the compressed representation retained at the end of the generation (codes,
    /// scale/zero metadata and the bounded not-yet-quantized residual).
    pub compressed_cache_bytes: u64,
}

/// The KV cache one generation ran on, carried on
/// [`TextLlmOutput::kv_cache`](crate::TextLlmOutput::kv_cache).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvCacheReport {
    /// [`KV_CACHE_FORMAT_VERSION`] of the backend that produced this report.
    pub format_version: u32,
    /// The compressed format the generation was started on; `None` when it ran dense throughout.
    pub format: Option<KvCompressionFormat>,
    /// Why the generation ran (or finished) dense; `None` exactly when it ran wholly compressed.
    pub fallback: Option<KvCacheFallbackReason>,
    /// The backend's own words for the fallback, when it has more than the reason.
    pub detail: Option<String>,
    /// Measured cache counters.
    pub counters: KvCacheCounters,
}

impl KvCacheReport {
    /// A generation that ran dense throughout for `reason`.
    pub fn dense(reason: KvCacheFallbackReason, detail: Option<String>) -> Self {
        Self {
            format_version: KV_CACHE_FORMAT_VERSION,
            format: None,
            fallback: Some(reason),
            detail,
            counters: KvCacheCounters::default(),
        }
    }

    /// Whether the whole generation ran on the compressed cache with no dense fallback.
    pub fn ran_compressed(&self) -> bool {
        self.format.is_some() && self.fallback.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ON: KvCompressionPolicy = KvCompressionPolicy::Qualified;

    #[test]
    fn the_policy_is_off_by_default_and_off_is_always_dense() {
        assert_eq!(KvCompressionPolicy::default(), KvCompressionPolicy::Off);
        for family in [None, Some(KvModelFamily::Llama), Some(KvModelFamily::Qwen3)] {
            assert_eq!(
                qualify_kv_compression(KvCompressionPolicy::Off, family, 50_000, 1),
                Err(KvCacheFallbackReason::PolicyDisabled)
            );
        }
    }

    #[test]
    fn the_table_holds_one_k8v8_row_per_family() {
        for family in [KvModelFamily::Llama, KvModelFamily::Qwen3] {
            let rows = KV_COMPRESSION_QUALIFICATIONS
                .iter()
                .filter(|row| row.family == family)
                .collect::<Vec<_>>();
            assert_eq!(rows.len(), 1, "{family:?}");
            assert_eq!(rows[0].format, KvCompressionFormat::GroupAffineK8V8);
            assert!(rows[0]
                .max_context_tokens
                .is_none_or(|max| max > rows[0].min_context_tokens));
        }
        let format = KvCompressionFormat::GroupAffineK8V8;
        assert_eq!(
            (format.key_bits(), format.value_bits(), format.group_size()),
            (8, 8, 32)
        );
    }

    #[test]
    fn qwen3_runs_compressed_from_its_memory_material_coordinate_up() {
        let qwen = Some(KvModelFamily::Qwen3);
        assert_eq!(
            qualify_kv_compression(ON, qwen, 10_239, 1),
            Err(KvCacheFallbackReason::BelowMinimumContext)
        );
        for context in [10_240, 40_448, 1 << 40] {
            let row = qualify_kv_compression(ON, qwen, context, 1).unwrap();
            assert_eq!(
                (row.family, row.format),
                (KvModelFamily::Qwen3, KvCompressionFormat::GroupAffineK8V8)
            );
        }
    }

    #[test]
    fn llama_runs_compressed_from_memory_material_up_to_its_pending_fit_boundary() {
        let llama = Some(KvModelFamily::Llama);
        assert_eq!(
            qualify_kv_compression(ON, llama, 32_767, 1),
            Err(KvCacheFallbackReason::BelowMinimumContext)
        );
        assert!(qualify_kv_compression(ON, llama, 32_768, 1).is_ok());
        assert!(qualify_kv_compression(ON, llama, 130_559, 1).is_ok());
        assert_eq!(
            qualify_kv_compression(ON, llama, 130_560, 1),
            Err(KvCacheFallbackReason::AboveQualifiedContext)
        );
    }

    #[test]
    fn batch_and_unknown_families_stay_dense_in_a_fixed_order() {
        assert_eq!(
            qualify_kv_compression(ON, Some(KvModelFamily::Qwen3), 20_000, 2),
            Err(KvCacheFallbackReason::BatchedDecode)
        );
        // Batch is decided before the family, so a batched unknown model reports the batch.
        assert_eq!(
            qualify_kv_compression(ON, None, 20_000, 4),
            Err(KvCacheFallbackReason::BatchedDecode)
        );
        assert_eq!(
            qualify_kv_compression(ON, None, 20_000, 1),
            Err(KvCacheFallbackReason::UnqualifiedModel)
        );
        assert_eq!(
            qualify_kv_compression(ON, Some(KvModelFamily::Llama), 20_000, 0),
            Err(KvCacheFallbackReason::BatchedDecode)
        );
    }

    #[test]
    fn a_table_without_a_family_row_is_unqualified_for_it() {
        static ONLY_QWEN: &[KvQualification] = &[KvQualification {
            family: KvModelFamily::Qwen3,
            format: KvCompressionFormat::GroupAffineK8V8,
            min_context_tokens: 1,
            max_context_tokens: None,
            evidence: "test",
        }];
        assert_eq!(
            qualify_against(ONLY_QWEN, ON, Some(KvModelFamily::Llama), 50_000, 1),
            Err(KvCacheFallbackReason::UnqualifiedModel)
        );
        assert!(qualify_against(ONLY_QWEN, ON, Some(KvModelFamily::Qwen3), 1, 1).is_ok());
    }

    #[test]
    fn reason_and_format_labels_are_stable_and_distinct() {
        let ids = KvCacheFallbackReason::ALL.map(KvCacheFallbackReason::id);
        let unique = ids.iter().collect::<std::collections::BTreeSet<_>>();
        assert_eq!(unique.len(), ids.len());
        assert_eq!(
            KvCacheFallbackReason::PolicyDisabled.id(),
            "policy_disabled"
        );
        assert_eq!(
            KvCompressionFormat::GroupAffineK8V8.id(),
            "group-affine-k8v8"
        );
        assert_eq!(
            (KvModelFamily::Llama.id(), KvModelFamily::Qwen3.id()),
            ("llama", "qwen3")
        );
    }

    /// Llama-3.2-3B / Qwen3-1.7B decoder KV geometry at bf16.
    const EVIDENCE_SHAPE: KvCacheShape = KvCacheShape {
        layers: 28,
        kv_heads: 8,
        head_dim: 128,
        element_bytes: 2,
    };
    const NO_TRANSIENT: CompressedKvAllocation = CompressedKvAllocation {
        growth_block_tokens: 256,
        coexisting_layer_transients: 0,
        reader_scratch_bytes_per_layer: 0,
    };

    #[test]
    fn k8v8_packs_one_byte_codes_and_a_scale_zero_pair_per_32_elements() {
        let format = KvCompressionFormat::GroupAffineK8V8;
        assert_eq!(format.group_metadata_bytes(), 4);
        // Per head and position: K and V each store D one-byte codes plus D/32 f16 pairs.
        for (capacity, head_dim) in [(256, 128), (32, 64), (130_560, 256)] {
            let per_side = head_dim + head_dim / 32 * 4;
            assert_eq!(
                format.packed_bytes_per_head(capacity, head_dim),
                Some(capacity * per_side * 2),
                "{capacity} x {head_dim}"
            );
        }
        // A partial K group is priced as a whole group.
        assert_eq!(
            format.packed_bytes_per_head(33, 128).unwrap(),
            2 * (32 * 128 + 128 * 4) + 33 * (128 + 4 * 4)
        );
    }

    #[test]
    fn compressed_cache_bytes_price_block_capacity_residuals_and_transients() {
        let format = KvCompressionFormat::GroupAffineK8V8;
        let shape = EVIDENCE_SHAPE;
        let packed_layer =
            |tokens| format.packed_bytes_per_head(tokens, 128).unwrap() * shape.kv_heads;
        let residual_layer = 32 * 128 * 2 * 2 * shape.kv_heads;
        // Resident only: block-rounded packed arrays plus one group of dense residual rows.
        let resident = |tokens| (packed_layer(tokens) + residual_layer) * shape.layers;
        assert_eq!(
            compressed_kv_cache_bytes(format, shape, 32_768 + 128, NO_TRANSIENT),
            Some(resident(32_768 + 256) + residual_layer * shape.layers)
        );
        // An empty request still holds one block.
        assert_eq!(
            compressed_kv_cache_bytes(format, shape, 0, NO_TRANSIENT),
            compressed_kv_cache_bytes(format, shape, 256, NO_TRANSIENT)
        );
        // Growth/flush transients scale with the coexisting layers (capped at the layer count)
        // and the reader scratch with every layer.
        let with = |coexisting, scratch| {
            compressed_kv_cache_bytes(
                format,
                shape,
                40_960,
                CompressedKvAllocation {
                    coexisting_layer_transients: coexisting,
                    reader_scratch_bytes_per_layer: scratch,
                    ..NO_TRANSIENT
                },
            )
            .unwrap()
        };
        let base = with(0, 0);
        assert_eq!(
            with(3, 0) - base,
            3 * (packed_layer(40_960) + packed_layer(256))
        );
        assert_eq!(with(1_000, 0), with(28, 0));
        assert_eq!(with(0, 7) - base, 7 * shape.layers);
        assert_eq!(
            compressed_kv_cache_bytes(
                format,
                shape,
                1,
                CompressedKvAllocation {
                    growth_block_tokens: 0,
                    ..NO_TRANSIENT
                }
            ),
            None
        );
        assert_eq!(
            compressed_kv_cache_bytes(format, shape, u64::MAX, NO_TRANSIENT),
            None
        );
    }

    /// The point of the compressed cache: at every qualified context of the evidence geometry,
    /// even with a full evaluator window of layer transients it prices below the dense cache.
    #[test]
    fn qualified_contexts_price_below_dense() {
        let format = KvCompressionFormat::GroupAffineK8V8;
        let allocation = CompressedKvAllocation {
            growth_block_tokens: 256,
            coexisting_layer_transients: 11,
            reader_scratch_bytes_per_layer: 32 * 128 * 130 * 4,
        };
        for row in KV_COMPRESSION_QUALIFICATIONS {
            for tokens in [
                row.min_context_tokens,
                row.max_context_tokens.map_or(40_960, |max| max - 1),
            ] {
                let compressed =
                    compressed_kv_cache_bytes(format, EVIDENCE_SHAPE, tokens, allocation).unwrap();
                let dense = EVIDENCE_SHAPE.dense_bytes(tokens).unwrap();
                assert!(compressed < dense, "{tokens}: {compressed} vs {dense}");
                assert!(
                    compressed > dense / 2,
                    "{tokens}: never cheaper than codes alone"
                );
            }
        }
    }

    #[test]
    fn a_dense_report_is_never_compressed() {
        let report = KvCacheReport::dense(KvCacheFallbackReason::BelowMinimumContext, None);
        assert_eq!(report.format_version, KV_CACHE_FORMAT_VERSION);
        assert!(!report.ran_compressed());
        assert_eq!(report.counters, KvCacheCounters::default());
        let compressed = KvCacheReport {
            format: Some(KvCompressionFormat::GroupAffineK8V8),
            fallback: None,
            ..report.clone()
        };
        assert!(compressed.ran_compressed());
        let interrupted = KvCacheReport {
            fallback: Some(KvCacheFallbackReason::RuntimeFallback),
            ..compressed
        };
        assert!(!interrupted.ran_compressed());
    }
}
