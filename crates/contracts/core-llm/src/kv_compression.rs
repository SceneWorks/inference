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
    /// Smallest prompt (tokens prefilled before decode starts) that runs compressed. Shorter
    /// contexts decode slower compressed than dense, so they stay dense.
    pub min_context_tokens: u64,
    /// Exclusive upper bound of the request's final context — prompt plus every token it may
    /// generate — so a compressed decode never grows past the evidenced range. `None` only when
    /// the evidence covers every context the family can reach.
    pub max_context_tokens: Option<u64>,
    /// Where the qualification comes from.
    pub evidence: &'static str,
}

impl KvQualification {
    /// Whether a request prefilling `prompt_tokens` and decoding up to `max_new_tokens` more lies
    /// in this row's qualified range: its prompt at or above the minimum and its final context
    /// below the maximum.
    pub fn admits(&self, prompt_tokens: u64, max_new_tokens: u64) -> bool {
        prompt_tokens >= self.min_context_tokens
            && self.max_context_tokens.is_none_or(|max| {
                prompt_tokens
                    .checked_add(max_new_tokens)
                    .is_some_and(|final_tokens| final_tokens < max)
            })
    }
}

/// The one checked-in qualification table (format version [`KV_CACHE_FORMAT_VERSION`]).
///
/// Thresholds are the SC-20671 campaign coordinates the evidence was measured at
/// (`context_band_target` in `mlx-llm`'s campaign): memory-material is a quarter of the evidence
/// model's native window and fit-boundary is `max(window − 512, ⌈0.9 · window⌉)`. The minimum
/// bounds the prompt; the maximum bounds the final context (prompt + generated tokens).
/// Qwen3-1.7B (40 960-token window) qualified at both coordinates, so it runs compressed from its
/// memory-material coordinate (10 240) to its evidenced window (final context ≤ 40 960): a Qwen3
/// checkpoint with a longer window (Qwen3-2507, YaRN) stays dense beyond it. Llama-3.2-3B
/// (131 072-token window) qualified at memory-material (32 768); its fit-boundary coordinate
/// (130 560) awaits a dense multi-turn noise-floor run, so final contexts from there up stay
/// dense until that row's maximum is raised.
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
        max_context_tokens: Some(40_961),
        evidence: "sc-20669 A2 run 37004025116 + dense noise floor 37035827730: \
                   Qwen3-1.7B-4bit memory-material (10240) and fit-boundary (40448) pass at \
                   group-affine-8 within its 40960-token window",
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
    /// The prompt is below the family's qualified minimum (short contexts decode slower
    /// compressed than dense).
    BelowMinimumContext,
    /// The final context (prompt + maximum new tokens) reaches past the family's qualified range.
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
/// `context_tokens` is the number of tokens prefilled before decoding starts; `max_new_tokens`
/// the most tokens the request may generate after them (the final context is their sum);
/// `batch` the number of sequences decoding together. The checks run in a fixed order — policy,
/// batch, family, prompt minimum, final-context maximum — so the reported reason is
/// deterministic.
pub fn qualify_kv_compression(
    policy: KvCompressionPolicy,
    family: Option<KvModelFamily>,
    context_tokens: u64,
    max_new_tokens: u64,
    batch: u64,
) -> Result<&'static KvQualification, KvCacheFallbackReason> {
    qualify_against(
        KV_COMPRESSION_QUALIFICATIONS,
        policy,
        family,
        context_tokens,
        max_new_tokens,
        batch,
    )
}

fn qualify_against(
    table: &'static [KvQualification],
    policy: KvCompressionPolicy,
    family: Option<KvModelFamily>,
    context_tokens: u64,
    max_new_tokens: u64,
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
    if let Some(row) = rows
        .iter()
        .find(|row| row.admits(context_tokens, max_new_tokens))
    {
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

    /// The report of one single-sequence generation on a provider whose model has no table family
    /// (a multimodal-wrapped or other non-table decoder, on any backend): the [`plan_kv_cache`]
    /// decision for `family: None` — [`KvCacheFallbackReason::PolicyDisabled`] when the request
    /// did not opt in, else [`KvCacheFallbackReason::UnqualifiedModel`].
    pub fn without_table_family(policy: KvCompressionPolicy) -> Self {
        plan_kv_cache_without_reader(
            KvCacheRequest {
                policy,
                family: None,
                batch: 1,
                ..KvCacheRequest::default()
            },
            "this provider",
        )
    }
}

/// The table family of a loaded decoder (sc-20683), shared by every backend so the same checkpoint
/// names the same family on MLX and Candle.
///
/// `decoder` is the backend's decoder-dispatch tag (`Architecture::family()` in both engines:
/// `"llama"`, `"qwen3"`, `"qwen3_5"`, …); `architecture` and `model_type` are the loaded
/// checkpoint config's `architectures[0]` and `model_type`. The Qwen3 dispatch is the family
/// itself. The Llama dispatch also serves Mistral and dense Qwen2, so a Llama-dispatched
/// checkpoint is the Llama family only when its config names llama. Every other dispatch (hybrid,
/// multimodal-wrapped, other architectures) has no family, and neither has a decoder assembled
/// without a config (a backend's `from_parts`), which cannot tell Llama from Mistral.
pub fn kv_model_family(
    decoder: &str,
    architecture: &str,
    model_type: &str,
) -> Option<KvModelFamily> {
    match decoder {
        "qwen3" => Some(KvModelFamily::Qwen3),
        "llama" => (architecture.to_ascii_lowercase().contains("llama")
            || model_type.eq_ignore_ascii_case("llama"))
        .then_some(KvModelFamily::Llama),
        _ => None,
    }
}

/// What a backend knows about one generation when it decides its KV cache (sc-20683).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KvCacheRequest {
    /// The request's opt-in.
    pub policy: KvCompressionPolicy,
    /// The loaded model's table family ([`kv_model_family`]).
    pub family: Option<KvModelFamily>,
    /// Tokens prefilled before decoding starts.
    pub context_tokens: u64,
    /// The most tokens the request may generate after them (the final context is the sum).
    pub max_new_tokens: u64,
    /// Sequences decoding together.
    pub batch: u64,
    /// Why this request takes a path the compressed cache is not wired through (multimodal
    /// prefill, a hybrid recurrent decoder), reported as
    /// [`KvCacheFallbackReason::UnsupportedRequest`]; `None` for a plain causal text decode.
    pub unsupported_request: Option<String>,
}

/// The KV cache a backend decided for one generation with [`plan_kv_cache`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KvCachePlan<R> {
    /// Run compressed on the qualifying row's format, read by the backend's `reader`.
    Compressed {
        /// The qualifying row.
        qualification: &'static KvQualification,
        /// The backend's fused compressed-domain reader.
        reader: R,
    },
    /// Run dense; the report (and its reason) the output carries.
    Dense(KvCacheReport),
}

/// Decide one generation's KV cache in the fixed order every backend shares: the qualification
/// table ([`qualify_kv_compression`]: policy, batch, family, prompt minimum, final-context
/// maximum), then the request shape
/// ([`KvCacheRequest::unsupported_request`]), then the backend's own `reader` stage — which refuses
/// with its reason ([`KvCacheFallbackReason::UnsupportedGeometry`] or
/// [`KvCacheFallbackReason::ReaderUnavailable`]) and detail, or hands back the reader that serves the
/// qualifying row. The reader stage runs only for a request every earlier stage admits.
pub fn plan_kv_cache<R>(
    request: KvCacheRequest,
    reader: impl FnOnce(&'static KvQualification) -> Result<R, (KvCacheFallbackReason, String)>,
) -> KvCachePlan<R> {
    let qualification = match qualify_kv_compression(
        request.policy,
        request.family,
        request.context_tokens,
        request.max_new_tokens,
        request.batch,
    ) {
        Ok(row) => row,
        Err(reason) => return KvCachePlan::Dense(KvCacheReport::dense(reason, None)),
    };
    if let Some(detail) = request.unsupported_request {
        return KvCachePlan::Dense(KvCacheReport::dense(
            KvCacheFallbackReason::UnsupportedRequest,
            Some(detail),
        ));
    }
    match reader(qualification) {
        Ok(reader) => KvCachePlan::Compressed {
            qualification,
            reader,
        },
        Err((reason, detail)) => KvCachePlan::Dense(KvCacheReport::dense(reason, Some(detail))),
    }
}

/// [`plan_kv_cache`] for a backend with no fused compressed-domain reader (Candle, on every
/// device): the same decision, except that a request the table and request shape admit — one a
/// reader-backed backend would run compressed — runs dense as
/// [`KvCacheFallbackReason::ReaderUnavailable`], naming `backend`. The report never claims a
/// compressed format or compressed counters.
pub fn plan_kv_cache_without_reader(request: KvCacheRequest, backend: &str) -> KvCacheReport {
    match plan_kv_cache(request, |_| {
        Err::<std::convert::Infallible, _>((
            KvCacheFallbackReason::ReaderUnavailable,
            format!("{backend} has no fused compressed-domain KV reader; the cache runs dense"),
        ))
    }) {
        KvCachePlan::Dense(report) => report,
        KvCachePlan::Compressed { reader, .. } => match reader {},
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
                qualify_kv_compression(KvCompressionPolicy::Off, family, 20_000, 0, 1),
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
    fn qwen3_runs_compressed_from_memory_material_within_its_evidenced_window() {
        let qwen = Some(KvModelFamily::Qwen3);
        assert_eq!(
            qualify_kv_compression(ON, qwen, 10_239, 64, 1),
            Err(KvCacheFallbackReason::BelowMinimumContext)
        );
        // Prompt at memory-material and at fit-boundary, final context up to the 40 960 window.
        for (prompt, new) in [(10_240, 0), (10_240, 30_720), (40_448, 512)] {
            let row = qualify_kv_compression(ON, qwen, prompt, new, 1).unwrap();
            assert_eq!(
                (row.family, row.format),
                (KvModelFamily::Qwen3, KvCompressionFormat::GroupAffineK8V8)
            );
        }
        // A longer-window Qwen3 (2507, YaRN) has no evidence past 40 960 tokens.
        for (prompt, new) in [(40_961, 0), (40_448, 513), (200_000, 0)] {
            assert_eq!(
                qualify_kv_compression(ON, qwen, prompt, new, 1),
                Err(KvCacheFallbackReason::AboveQualifiedContext),
                "{prompt} + {new}"
            );
        }
    }

    #[test]
    fn llama_runs_compressed_from_memory_material_up_to_its_pending_fit_boundary() {
        let llama = Some(KvModelFamily::Llama);
        assert_eq!(
            qualify_kv_compression(ON, llama, 32_767, 0, 1),
            Err(KvCacheFallbackReason::BelowMinimumContext)
        );
        assert!(qualify_kv_compression(ON, llama, 32_768, 512, 1).is_ok());
        assert!(qualify_kv_compression(ON, llama, 130_559, 0, 1).is_ok());
        assert_eq!(
            qualify_kv_compression(ON, llama, 130_560, 0, 1),
            Err(KvCacheFallbackReason::AboveQualifiedContext)
        );
        // A prompt admitted below the bound whose decode would grow into the unevidenced
        // fit-boundary band stays dense from the start.
        assert_eq!(
            qualify_kv_compression(ON, llama, 130_000, 1_000, 1),
            Err(KvCacheFallbackReason::AboveQualifiedContext)
        );
        assert_eq!(
            qualify_kv_compression(ON, llama, 32_768, u64::MAX, 1),
            Err(KvCacheFallbackReason::AboveQualifiedContext),
            "an overflowing final context is never admitted"
        );
    }

    #[test]
    fn batch_and_unknown_families_stay_dense_in_a_fixed_order() {
        assert_eq!(
            qualify_kv_compression(ON, Some(KvModelFamily::Qwen3), 20_000, 0, 2),
            Err(KvCacheFallbackReason::BatchedDecode)
        );
        // Batch is decided before the family, so a batched unknown model reports the batch.
        assert_eq!(
            qualify_kv_compression(ON, None, 20_000, 0, 4),
            Err(KvCacheFallbackReason::BatchedDecode)
        );
        assert_eq!(
            qualify_kv_compression(ON, None, 20_000, 0, 1),
            Err(KvCacheFallbackReason::UnqualifiedModel)
        );
        assert_eq!(
            qualify_kv_compression(ON, Some(KvModelFamily::Llama), 40_000, 0, 0),
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
            qualify_against(ONLY_QWEN, ON, Some(KvModelFamily::Llama), 50_000, 0, 1),
            Err(KvCacheFallbackReason::UnqualifiedModel)
        );
        assert!(qualify_against(ONLY_QWEN, ON, Some(KvModelFamily::Qwen3), 1, 1 << 40, 1).is_ok());
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

    fn request(family: Option<KvModelFamily>, context_tokens: u64) -> KvCacheRequest {
        KvCacheRequest {
            policy: ON,
            family,
            context_tokens,
            max_new_tokens: 64,
            batch: 1,
            unsupported_request: None,
        }
    }

    #[test]
    fn a_family_is_named_by_dispatch_and_for_llama_by_the_config_identity() {
        use KvModelFamily::{Llama, Qwen3};
        assert_eq!(
            kv_model_family("qwen3", "Qwen3ForCausalLM", "qwen3"),
            Some(Qwen3)
        );
        assert_eq!(
            kv_model_family("llama", "LlamaForCausalLM", ""),
            Some(Llama)
        );
        assert_eq!(kv_model_family("llama", "", "LLAMA"), Some(Llama));
        // The Llama dispatch also serves Mistral and dense Qwen2, which have no row.
        assert_eq!(
            kv_model_family("llama", "MistralForCausalLM", "mistral"),
            None
        );
        assert_eq!(kv_model_family("llama", "Qwen2ForCausalLM", "qwen2"), None);
        assert_eq!(kv_model_family("llama", "", ""), None);
        for decoder in ["qwen3_5", "qwen3_vl", "gemma2", "phi3", "deepseek_v2", ""] {
            assert_eq!(
                kv_model_family(decoder, "LlamaForCausalLM", "llama"),
                None,
                "{decoder}"
            );
        }
    }

    #[test]
    fn the_plan_runs_the_table_then_the_request_shape_then_the_reader() {
        let qwen = Some(KvModelFamily::Qwen3);
        let reader_ran = std::cell::Cell::new(false);
        let reader = |row: &'static KvQualification| {
            reader_ran.set(true);
            Ok::<_, (KvCacheFallbackReason, String)>(row.format)
        };
        // Table refusals win over the request shape and never reach the reader.
        let short_multimodal = KvCacheRequest {
            unsupported_request: Some("multimodal".into()),
            ..request(qwen, 100)
        };
        assert_eq!(
            plan_kv_cache(short_multimodal, reader),
            KvCachePlan::Dense(KvCacheReport::dense(
                KvCacheFallbackReason::BelowMinimumContext,
                None
            ))
        );
        let multimodal = KvCacheRequest {
            unsupported_request: Some("multimodal".into()),
            ..request(qwen, 20_000)
        };
        assert_eq!(
            plan_kv_cache(multimodal, reader),
            KvCachePlan::Dense(KvCacheReport::dense(
                KvCacheFallbackReason::UnsupportedRequest,
                Some("multimodal".into())
            ))
        );
        assert!(!reader_ran.get());
        let KvCachePlan::Compressed {
            qualification,
            reader: format,
        } = plan_kv_cache(request(qwen, 20_000), reader)
        else {
            panic!("a qualified plain request runs compressed");
        };
        assert!(reader_ran.get());
        assert_eq!(qualification.family, KvModelFamily::Qwen3);
        assert_eq!(format, KvCompressionFormat::GroupAffineK8V8);
        // The token budget is part of the table stage: it bounds the final context.
        let past_the_window = KvCacheRequest {
            max_new_tokens: 513,
            ..request(qwen, 40_448)
        };
        assert_eq!(
            plan_kv_cache(past_the_window, reader),
            KvCachePlan::Dense(KvCacheReport::dense(
                KvCacheFallbackReason::AboveQualifiedContext,
                None
            ))
        );
        let refused = plan_kv_cache(request(qwen, 20_000), |_| {
            Err::<(), _>((
                KvCacheFallbackReason::UnsupportedGeometry,
                "head dim".into(),
            ))
        });
        assert_eq!(
            refused,
            KvCachePlan::Dense(KvCacheReport::dense(
                KvCacheFallbackReason::UnsupportedGeometry,
                Some("head dim".into())
            ))
        );
    }

    #[test]
    fn a_backend_without_a_reader_reports_dense_with_the_shared_reason() {
        let backend = |request| plan_kv_cache_without_reader(request, "test-backend");
        let qwen = Some(KvModelFamily::Qwen3);
        let off = KvCacheRequest {
            policy: KvCompressionPolicy::Off,
            ..request(qwen, 20_000)
        };
        assert_eq!(
            backend(off).fallback,
            Some(KvCacheFallbackReason::PolicyDisabled)
        );
        assert_eq!(
            backend(request(None, 20_000)).fallback,
            Some(KvCacheFallbackReason::UnqualifiedModel)
        );
        let unavailable = backend(request(qwen, 20_000));
        assert_eq!(
            unavailable.fallback,
            Some(KvCacheFallbackReason::ReaderUnavailable)
        );
        assert!(unavailable
            .detail
            .as_deref()
            .is_some_and(|detail| detail.contains("test-backend")));
        assert_eq!(unavailable.format, None);
        assert_eq!(unavailable.counters, KvCacheCounters::default());
        assert!(!unavailable.ran_compressed());
    }

    #[test]
    fn a_provider_without_a_table_family_is_disabled_or_unqualified() {
        assert_eq!(
            KvCacheReport::without_table_family(KvCompressionPolicy::Off),
            KvCacheReport::dense(KvCacheFallbackReason::PolicyDisabled, None)
        );
        assert_eq!(
            KvCacheReport::without_table_family(ON),
            KvCacheReport::dense(KvCacheFallbackReason::UnqualifiedModel, None)
        );
    }
}
