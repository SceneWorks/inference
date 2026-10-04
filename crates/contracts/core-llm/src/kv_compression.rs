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
    /// Dense K/V the prompt step holds beside the packed store before it is quantized (the
    /// layers' fresh prompt K/V the backend lets coexist).
    pub prefill_transient_bytes: u64,
}

/// Bytes a compressed KV cache of `format` needs to serve `tokens` positions (prompt plus every
/// generated token), derived from the format rather than a fixed ratio:
///
/// * resident: per layer and KV head, the packed codes and scale/zero metadata at the
///   block-rounded capacity ([`KvCompressionFormat::packed_bytes_per_head`]) plus one group of
///   dense K and V residual rows at the decoder's compute width;
/// * transient: `coexisting_layer_transients` layers' packed arrays at full capacity plus one block
///   (the pre-growth copy or the flush output beside the live arrays), every layer's residual
///   rows once more (a step's rollback point holds the residuals a flush replaced), every
///   layer's fused-reader scratch, and the prompt step's dense K/V
///   (`prefill_transient_bytes`).
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
        )?
        .checked_add(allocation.prefill_transient_bytes)?;
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

/// The exact decoder architecture a qualification row was measured on (sc-20688 review): the
/// checkpoint config's geometry. The evidence is per model, not per family — a larger or smaller
/// sibling (Llama-3.1-8B, TinyLlama, Qwen3-0.6B/8B) shares the family's decoder code but not its
/// measured quality and speed — so only a checkpoint whose config matches a row's architecture
/// field for field is that row's model ([`qualified_kv_model_family`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KvModelArchitecture {
    /// `num_hidden_layers`.
    pub layers: u64,
    /// `hidden_size`.
    pub hidden_size: u64,
    /// `num_attention_heads`.
    pub attention_heads: u64,
    /// `num_key_value_heads` (defaults to `num_attention_heads`).
    pub kv_heads: u64,
    /// `head_dim` (defaults to `hidden_size / num_attention_heads`).
    pub head_dim: u64,
    /// `intermediate_size`.
    pub intermediate_size: u64,
    /// `vocab_size`.
    pub vocab_size: u64,
    /// `max_position_embeddings`.
    pub max_position_embeddings: u64,
    /// `rope_theta`, which must be a whole number to match.
    pub rope_theta: u64,
    /// `tie_word_embeddings` (defaults to `false`).
    pub tie_word_embeddings: bool,
}

impl KvModelArchitecture {
    /// The architecture of a checkpoint `config.json` (its `text_config` when it has one), or
    /// `None` when a required field is missing or not a whole number — such a config matches no
    /// row.
    pub fn from_config(config: &serde_json::Value) -> Option<Self> {
        let text = config.get("text_config").unwrap_or(config);
        let int = |key: &str| text.get(key).and_then(serde_json::Value::as_u64);
        let hidden_size = int("hidden_size")?;
        let attention_heads = int("num_attention_heads")?;
        let rope_theta = text.get("rope_theta").and_then(serde_json::Value::as_f64)?;
        if rope_theta.fract() != 0.0 || !(0.0..=u64::MAX as f64).contains(&rope_theta) {
            return None;
        }
        Some(Self {
            layers: int("num_hidden_layers")?,
            hidden_size,
            attention_heads,
            kv_heads: int("num_key_value_heads").unwrap_or(attention_heads),
            head_dim: match int("head_dim") {
                Some(head_dim) => head_dim,
                None => hidden_size.checked_div(attention_heads)?,
            },
            intermediate_size: int("intermediate_size")?,
            vocab_size: int("vocab_size")?,
            max_position_embeddings: int("max_position_embeddings")?,
            rope_theta: rope_theta as u64,
            tie_word_embeddings: text
                .get("tie_word_embeddings")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
        })
    }
}

/// `meta-llama/Llama-3.2-3B-Instruct` (the measured `mlx-community/Llama-3.2-3B-Instruct-4bit`
/// and its bf16 reference share it).
pub const LLAMA_3_2_3B_ARCHITECTURE: KvModelArchitecture = KvModelArchitecture {
    layers: 28,
    hidden_size: 3072,
    attention_heads: 24,
    kv_heads: 8,
    head_dim: 128,
    intermediate_size: 8192,
    vocab_size: 128_256,
    max_position_embeddings: 131_072,
    rope_theta: 500_000,
    tie_word_embeddings: true,
};

/// `Qwen/Qwen3-1.7B` (the measured `mlx-community/Qwen3-1.7B-4bit` and its bf16 reference share
/// it).
pub const QWEN3_1_7B_ARCHITECTURE: KvModelArchitecture = KvModelArchitecture {
    layers: 28,
    hidden_size: 2048,
    attention_heads: 16,
    kv_heads: 8,
    head_dim: 128,
    intermediate_size: 6144,
    vocab_size: 151_936,
    max_position_embeddings: 40_960,
    rope_theta: 1_000_000,
    tie_word_embeddings: true,
};

/// One qualified (model × format × context range) combination and the evidence behind it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvQualification {
    /// The decoder family of the measured model.
    pub family: KvModelFamily,
    /// The measured model.
    pub model: &'static str,
    /// The measured model's exact architecture: a checkpoint of the family qualifies for this row
    /// only when its config matches it.
    pub architecture: KvModelArchitecture,
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
/// Each row is one measured model ([`KvQualification::architecture`]), decoding one sequence: a
/// batch of more than one is [`KvCacheFallbackReason::BatchedDecode`] on every path.
///
/// Thresholds are the SC-20671 campaign coordinates the evidence was measured at
/// (`context_band_target` in `mlx-llm`'s campaign): memory-material is a quarter of the evidence
/// model's native window and fit-boundary is `max(window − 512, ⌈0.9 · window⌉)`. The minimum
/// bounds the prompt; the maximum bounds the final context (prompt + generated tokens).
/// Qwen3-1.7B (40 960-token window) qualified at both coordinates, so it runs compressed from its
/// memory-material coordinate (10 240) to its evidenced window (final context ≤ 40 960): a Qwen3
/// checkpoint with a longer window (Qwen3-2507, YaRN) stays dense beyond it. Llama-3.2-3B
/// (131 072-token window) qualified at memory-material (32 768) but failed fit-boundary (130 560)
/// on multi-turn (2/1024 against a dense floor of 0/1024), so its final context stays capped below
/// 130 560 and runs dense from there up.
///
/// Evidence: terminal campaign run 37201786765 (inference `32dea7a3c`, SceneWorks `1f4bb8282`),
/// with the group-affine-8 (A2) and dense noise-floor (NF) arms on the same UTC day and matching
/// prompt hashes. Llama-3.1-8B and Qwen3-8B were measured in run 37201786765 and fail the K8V8
/// gates (Llama-8B on multi-turn, Qwen3-8B on greedy); both stay dense
/// ([`KvCacheFallbackReason::UnqualifiedModel`]). History: A2 v5 run 37004025116 and noise-floor
/// runs 37035827730 and 37076039109.
pub const KV_COMPRESSION_QUALIFICATIONS: &[KvQualification] = &[
    KvQualification {
        family: KvModelFamily::Llama,
        model: "Llama-3.2-3B-Instruct",
        architecture: LLAMA_3_2_3B_ARCHITECTURE,
        format: KvCompressionFormat::GroupAffineK8V8,
        min_context_tokens: 32_768,
        max_context_tokens: Some(130_560),
        evidence: "sc-20669 terminal campaign run 37201786765 (inference 32dea7a3c, SceneWorks \
                   1f4bb8282; A2 and NF same UTC day, prompt hashes match): \
                   Llama-3.2-3B-Instruct-4bit memory-material (32768) passes at group-affine-8; \
                   fit-boundary (130560) fails multi-turn (2/1024 vs dense floor 0/1024), so \
                   final context stays < 130560; history: A2 v5 37004025116, noise floors \
                   37035827730 and 37076039109",
    },
    KvQualification {
        family: KvModelFamily::Qwen3,
        model: "Qwen3-1.7B",
        architecture: QWEN3_1_7B_ARCHITECTURE,
        format: KvCompressionFormat::GroupAffineK8V8,
        min_context_tokens: 10_240,
        max_context_tokens: Some(40_961),
        evidence: "sc-20669 terminal campaign run 37201786765 (inference 32dea7a3c, SceneWorks \
                   1f4bb8282; A2 and NF same UTC day, prompt hashes match): \
                   Qwen3-1.7B-4bit memory-material (10240) and fit-boundary (40448) pass at \
                   group-affine-8 within its 40960-token window; history: A2 v5 37004025116, \
                   noise floors 37035827730 and 37076039109",
    },
];

/// Why a generation ran (or finished) on the dense KV cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KvCacheFallbackReason {
    /// The request did not opt in ([`KvCompressionPolicy::Off`]).
    PolicyDisabled,
    /// The loaded model is not one the qualification table measured: no row of its family, or a
    /// checkpoint of the family whose architecture is not a row's ([`qualified_kv_model_family`]).
    UnqualifiedModel,
    /// The request takes a path the compressed cache is not wired through (multimodal prefill,
    /// multi-token prediction, a hybrid recurrent decoder).
    UnsupportedRequest,
    /// More than one sequence decodes together. The table's evidence is single-sequence only, so
    /// every batched path — the padded lockstep batch decoder and continuous batching over paged
    /// caches alike — runs every sequence of a batch of more than one dense with this reason.
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
    /// The generation ran on a paged compressed cache that kept its history compressed, but some
    /// attention calls were outside the fused paged reader (an additive mask, an attention scale
    /// or shape it does not implement, a caller that needs dense K/V) and were served by the
    /// cache's dense gather fallback: the sequence's pages dequantized for that one call. The
    /// backend's detail names each reason and its call count.
    DenseGather,
}

impl KvCacheFallbackReason {
    /// Every reason, in declaration order.
    pub const ALL: [Self; 10] = [
        Self::PolicyDisabled,
        Self::UnqualifiedModel,
        Self::UnsupportedRequest,
        Self::BatchedDecode,
        Self::BelowMinimumContext,
        Self::AboveQualifiedContext,
        Self::UnsupportedGeometry,
        Self::ReaderUnavailable,
        Self::RuntimeFallback,
        Self::DenseGather,
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
            Self::DenseGather => "dense_gather",
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
    /// Attention calls a paged compressed cache served through its dense gather fallback; each
    /// dequantized that sequence's pages for the one call and kept nothing dense afterwards.
    pub dense_gather_fallbacks: u64,
    /// Device bytes the compressed representation retained at the end of the generation (codes,
    /// scale/zero metadata and the bounded not-yet-quantized residual). For a paged cache this is
    /// the generation's own live pages, not the shared pool's capacity — see
    /// [`Self::pool_held_bytes`].
    pub compressed_cache_bytes: u64,
    /// Device bytes the shared page pool held when the generation finished: every sequence's
    /// pages plus the pool's unused capacity (sc-20681). What the paged cache actually holds
    /// resident, as opposed to this generation's live share. `0` for a cache without a shared
    /// pool.
    pub pool_held_bytes: u64,
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

/// The table family a loaded checkpoint plans its KV cache as (sc-20688 review): its
/// [`kv_model_family`] — read from the config's (`text_config`'s) `architectures[0]` and
/// `model_type` — but only when the config's [`KvModelArchitecture`] is that of a measured row of
/// the family. Every other checkpoint — another size of the family, a different context window or
/// RoPE base, an unreadable config — has no table family, so an opted-in request reports
/// [`KvCacheFallbackReason::UnqualifiedModel`]. `decoder` is the backend's decoder-dispatch tag.
pub fn qualified_kv_model_family(
    decoder: &str,
    config: &serde_json::Value,
) -> Option<KvModelFamily> {
    let text = config.get("text_config").unwrap_or(config);
    let field = |key: &str| {
        text.get(key)
            .and_then(|value| match value {
                serde_json::Value::Array(values) => {
                    values.first().and_then(serde_json::Value::as_str)
                }
                value => value.as_str(),
            })
            .unwrap_or("")
    };
    let family = kv_model_family(decoder, field("architectures"), field("model_type"))?;
    let architecture = KvModelArchitecture::from_config(config)?;
    KV_COMPRESSION_QUALIFICATIONS
        .iter()
        .any(|row| row.family == family && row.architecture == architecture)
        .then_some(family)
}

/// Head dimensions the fused compressed-domain reader implements.
pub const KV_FUSED_READER_HEAD_DIMS: [u64; 3] = [64, 128, 256];

/// The attention geometry of a loaded decoder, as the compressed-KV plan's geometry stage reads it
/// (sc-20688 review: one stage both backends run, so a decoder the fused reader cannot serve
/// reports [`KvCacheFallbackReason::UnsupportedGeometry`] on every backend).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvAttentionGeometry {
    /// Channels per attention head.
    pub head_dim: u64,
    /// The decoder soft-caps attention scores (Gemma-2 `attn_logit_softcapping`).
    pub attention_softcap: bool,
    /// The decoder's attention scale denominator (`query_pre_attn_scalar`); `None` is `head_dim`.
    pub query_pre_attn_scalar: Option<u64>,
    /// The decoder attends through latent (MLA) or cross-layer shared K/V.
    pub latent_or_shared_kv: bool,
    /// The decoder routes its MLP through experts (a mixture-of-experts checkpoint such as
    /// `qwen3_moe`, which dispatches as its dense family).
    pub mixture_of_experts: bool,
}

impl KvAttentionGeometry {
    /// Why the fused compressed reader cannot serve this decoder, or `None` when it can. The
    /// default (zero) geometry is refused, so a backend that does not fill it in fails closed.
    pub fn refusal(&self) -> Option<String> {
        if !KV_FUSED_READER_HEAD_DIMS.contains(&self.head_dim) {
            return Some(format!(
                "the fused compressed reader supports head dimension 64, 128 or 256, not {}",
                self.head_dim
            ));
        }
        if self.attention_softcap {
            return Some("attention-score soft-cap needs tanh before softmax".into());
        }
        if self
            .query_pre_attn_scalar
            .is_some_and(|scalar| scalar != self.head_dim)
        {
            return Some(
                "the fused reader scales scores by the inverse square-root head dimension".into(),
            );
        }
        if self.latent_or_shared_kv {
            return Some("latent or shared K/V attention has no compressed-domain reader".into());
        }
        if self.mixture_of_experts {
            return Some(
                "a mixture-of-experts decoder is outside the dense decoders the evidence measured"
                    .into(),
            );
        }
        None
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
    /// The loaded decoder's attention geometry, refused by the geometry stage
    /// ([`KvAttentionGeometry::refusal`]) as [`KvCacheFallbackReason::UnsupportedGeometry`].
    pub geometry: KvAttentionGeometry,
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
/// ([`KvCacheRequest::unsupported_request`]), then the decoder's attention geometry
/// ([`KvCacheRequest::geometry`], refused as [`KvCacheFallbackReason::UnsupportedGeometry`]), then
/// the backend's own `reader` stage — which refuses with its reason (normally
/// [`KvCacheFallbackReason::ReaderUnavailable`]) and detail, or hands back the reader that serves
/// the qualifying row. The reader stage runs only for a request every earlier stage admits.
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
    if let Some(detail) = request.geometry.refusal() {
        return KvCachePlan::Dense(KvCacheReport::dense(
            KvCacheFallbackReason::UnsupportedGeometry,
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
            model: "test",
            architecture: QWEN3_1_7B_ARCHITECTURE,
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
        assert_eq!(KvCacheFallbackReason::DenseGather.id(), "dense_gather");
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
        prefill_transient_bytes: 0,
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
        // The prompt step's dense K/V transient is charged as supplied.
        assert_eq!(
            compressed_kv_cache_bytes(
                format,
                shape,
                40_960,
                CompressedKvAllocation {
                    prefill_transient_bytes: 12_345,
                    ..NO_TRANSIENT
                },
            )
            .unwrap()
                - base,
            12_345
        );
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
        for row in KV_COMPRESSION_QUALIFICATIONS {
            for tokens in [
                row.min_context_tokens,
                row.max_context_tokens.map_or(40_960, |max| max - 1),
            ] {
                let allocation = CompressedKvAllocation {
                    growth_block_tokens: 256,
                    coexisting_layer_transients: 11,
                    reader_scratch_bytes_per_layer: 32 * 128 * 130 * 4,
                    // One layer's dense K/V of the whole context as the prompt step's transient.
                    prefill_transient_bytes: tokens * 8 * 128 * 2 * 2,
                };
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
            geometry: SUPPORTED,
        }
    }

    const SUPPORTED: KvAttentionGeometry = KvAttentionGeometry {
        head_dim: 128,
        attention_softcap: false,
        query_pre_attn_scalar: None,
        latent_or_shared_kv: false,
        mixture_of_experts: false,
    };

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

    /// The measured checkpoints' own `config.json` fields (the mlx-community 4-bit snapshots the
    /// SC-20671 campaign measured; `quantization` is the MLX conversion's and not architecture).
    fn llama_3_2_3b_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama",
            "num_hidden_layers": 28, "hidden_size": 3072, "num_attention_heads": 24,
            "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 128256,
            "intermediate_size": 8192, "rope_theta": 500000.0,
            "max_position_embeddings": 131072, "tie_word_embeddings": true,
            "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                             "original_max_position_embeddings": 8192, "rope_type": "llama3"},
            "quantization": {"group_size": 64, "bits": 4},
        })
    }

    fn qwen3_1_7b_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3",
            "num_hidden_layers": 28, "hidden_size": 2048, "num_attention_heads": 16,
            "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 151936,
            "intermediate_size": 6144, "rope_theta": 1000000, "max_position_embeddings": 40960,
            "tie_word_embeddings": true, "quantization": {"group_size": 64, "bits": 4},
        })
    }

    fn with(mut config: serde_json::Value, fields: serde_json::Value) -> serde_json::Value {
        for (key, value) in fields.as_object().unwrap() {
            config[key] = value.clone();
        }
        config
    }

    /// sc-20688 review: qualification is per measured model, not per family. The two measured
    /// configs plan as their family; a sibling of the same family (same decoder dispatch and
    /// config identity) does not, so an opted-in request on it reports `UnqualifiedModel`.
    #[test]
    fn only_the_measured_architectures_plan_as_a_table_family() {
        use KvModelFamily::{Llama, Qwen3};
        assert_eq!(
            KvModelArchitecture::from_config(&llama_3_2_3b_config()),
            Some(LLAMA_3_2_3B_ARCHITECTURE)
        );
        assert_eq!(
            KvModelArchitecture::from_config(&qwen3_1_7b_config()),
            Some(QWEN3_1_7B_ARCHITECTURE)
        );
        assert_eq!(
            qualified_kv_model_family("llama", &llama_3_2_3b_config()),
            Some(Llama)
        );
        assert_eq!(
            qualified_kv_model_family("qwen3", &qwen3_1_7b_config()),
            Some(Qwen3)
        );
        // A multimodal-style wrapper reads its text decoder's config.
        assert_eq!(
            qualified_kv_model_family(
                "qwen3",
                &serde_json::json!({"model_type": "wrapper", "text_config": qwen3_1_7b_config()})
            ),
            Some(Qwen3)
        );
        let llama_3_1_8b = with(
            llama_3_2_3b_config(),
            serde_json::json!({"num_hidden_layers": 32, "hidden_size": 4096,
                "num_attention_heads": 32, "intermediate_size": 14336,
                "tie_word_embeddings": false}),
        );
        let tinyllama = with(
            llama_3_2_3b_config(),
            serde_json::json!({"num_hidden_layers": 22, "hidden_size": 2048,
                "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 64,
                "intermediate_size": 5632, "vocab_size": 32000, "rope_theta": 10000.0,
                "max_position_embeddings": 2048, "tie_word_embeddings": false}),
        );
        let qwen3_8b = with(
            qwen3_1_7b_config(),
            serde_json::json!({"num_hidden_layers": 36, "hidden_size": 4096,
                "num_attention_heads": 32, "intermediate_size": 12288,
                "tie_word_embeddings": false}),
        );
        let qwen3_0_6b = with(
            qwen3_1_7b_config(),
            serde_json::json!({"hidden_size": 1024, "intermediate_size": 3072}),
        );
        // A long-window Qwen3-1.7B variant (YaRN / 2507-style) is not the measured window.
        let qwen3_long = with(
            qwen3_1_7b_config(),
            serde_json::json!({"max_position_embeddings": 262144}),
        );
        let llama_other_rope = with(
            llama_3_2_3b_config(),
            serde_json::json!({"rope_theta": 500000.5}),
        );
        for (decoder, config, family) in [
            ("llama", &llama_3_1_8b, Llama),
            ("llama", &tinyllama, Llama),
            ("llama", &llama_other_rope, Llama),
            ("qwen3", &qwen3_8b, Qwen3),
            ("qwen3", &qwen3_0_6b, Qwen3),
            ("qwen3", &qwen3_long, Qwen3),
        ] {
            // The family rule alone names it — the architecture is what refuses it.
            assert_eq!(
                kv_model_family(
                    decoder,
                    config["architectures"][0].as_str().unwrap(),
                    config["model_type"].as_str().unwrap()
                ),
                Some(family),
                "{config}"
            );
            assert_eq!(qualified_kv_model_family(decoder, config), None, "{config}");
        }
        // Missing or fractional fields match no row.
        let mut partial = llama_3_2_3b_config();
        partial.as_object_mut().unwrap().remove("intermediate_size");
        assert_eq!(qualified_kv_model_family("llama", &partial), None);
        // The measured architecture under another dispatch (Mistral identity) has no family.
        let mistral = with(
            llama_3_2_3b_config(),
            serde_json::json!({"architectures": ["MistralForCausalLM"], "model_type": "mistral"}),
        );
        assert_eq!(qualified_kv_model_family("llama", &mistral), None);
        // Every row names a distinct measured architecture.
        let rows = KV_COMPRESSION_QUALIFICATIONS
            .iter()
            .map(|row| (row.family, row.architecture))
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(rows.len(), KV_COMPRESSION_QUALIFICATIONS.len());
    }

    /// sc-20688 review: the geometry stage is shared, so every backend refuses the same decoders
    /// with `UnsupportedGeometry` — after the table and the request shape, before the reader.
    #[test]
    fn the_shared_geometry_stage_refuses_what_the_fused_reader_cannot_read() {
        assert_eq!(SUPPORTED.refusal(), None);
        for head_dim in KV_FUSED_READER_HEAD_DIMS {
            assert_eq!(
                KvAttentionGeometry {
                    head_dim,
                    query_pre_attn_scalar: Some(head_dim),
                    ..SUPPORTED
                }
                .refusal(),
                None
            );
        }
        let refusals = [
            (KvAttentionGeometry::default(), "head dimension"),
            (
                KvAttentionGeometry {
                    head_dim: 96,
                    ..SUPPORTED
                },
                "head dimension",
            ),
            (
                KvAttentionGeometry {
                    attention_softcap: true,
                    ..SUPPORTED
                },
                "soft-cap",
            ),
            (
                KvAttentionGeometry {
                    query_pre_attn_scalar: Some(256),
                    ..SUPPORTED
                },
                "square-root",
            ),
            (
                KvAttentionGeometry {
                    latent_or_shared_kv: true,
                    ..SUPPORTED
                },
                "latent or shared",
            ),
            (
                KvAttentionGeometry {
                    mixture_of_experts: true,
                    ..SUPPORTED
                },
                "mixture-of-experts",
            ),
        ];
        let qwen = Some(KvModelFamily::Qwen3);
        for (geometry, words) in refusals {
            let refusal = geometry.refusal().unwrap();
            assert!(refusal.contains(words), "{refusal}");
            let refused = KvCacheRequest {
                geometry,
                ..request(qwen, 20_000)
            };
            let dense = KvCacheReport::dense(
                KvCacheFallbackReason::UnsupportedGeometry,
                Some(refusal.clone()),
            );
            // With a reader and without one: the same reason and detail.
            assert_eq!(
                plan_kv_cache(refused.clone(), |row| Ok::<
                    _,
                    (KvCacheFallbackReason, String),
                >(row.format)),
                KvCachePlan::Dense(dense.clone())
            );
            assert_eq!(plan_kv_cache_without_reader(refused.clone(), "test"), dense);
            // The table and the request shape decide first.
            assert_eq!(
                plan_kv_cache_without_reader(
                    KvCacheRequest {
                        geometry,
                        ..request(qwen, 100)
                    },
                    "test"
                )
                .fallback,
                Some(KvCacheFallbackReason::BelowMinimumContext)
            );
            assert_eq!(
                plan_kv_cache_without_reader(
                    KvCacheRequest {
                        unsupported_request: Some("hybrid".into()),
                        ..refused
                    },
                    "test"
                )
                .fallback,
                Some(KvCacheFallbackReason::UnsupportedRequest)
            );
        }
    }
}
