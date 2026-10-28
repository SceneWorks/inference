//! Encoder contract records: packing, config values, tokenizer and prompt contracts, and the discovery/planning facts.

use super::*;

/// Selected-encoder behavior configs are metadata, not model payloads. Both bounded discovery and
/// executable validation apply this same limit so a candidate cannot pass one admission seam and
/// fail the other. Executable validation still seals the complete accepted config and every weights
/// shard through [`PinnedEncoderSource`].
pub(super) const MAX_SELECTED_ENCODER_CONFIG_BYTES: u64 = 4 * 1024 * 1024;

/// The architecture and output shape a generator requires from its text encoder.
///
/// `architecture` is the Hugging Face text-model `model_type` (for multimodal configs this is read
/// from `text_config.model_type`).  `output_width` is the width delivered to the denoiser after the
/// provider's fixed hidden-state selection/concatenation policy; it can therefore be wider than
/// `hidden_size` (FLUX.2 concatenates three intermediate states).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderPackingContract {
    /// The one affine group size the consuming loaders interpret. A same-bit source at any other
    /// group size is not compatible: several backends intentionally use a compile-time group size.
    pub group_size: usize,
    /// Whether the token embedding is packed with the layer projections. Norms remain dense.
    pub pack_embedding: bool,
    /// Whether the optional generation LM head is packed. FLUX.2-dev keeps this head dense.
    pub pack_lm_head: bool,
    /// Whether the backend can construct packed weights directly from one safetensors file. Candle
    /// Krea requires directory sidecars and therefore sets this false even though its MLX twin can
    /// consume the same packed tensor triple from a file.
    pub supports_file: bool,
}

/// An exactly comparable finite floating-point config value. Storing IEEE bits keeps
/// [`EncoderContract`] `Eq` while still presenting normal f64 values at construction/validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderConfigFloat(u64);

impl EncoderConfigFloat {
    pub const fn new(value: f64) -> Self {
        Self(value.to_bits())
    }

    pub const fn get(self) -> f64 {
        f64::from_bits(self.0)
    }
}

/// An execution-affecting boolean whose authored config value may be required or optional.
///
/// `Optional` does not mean unconstrained: omission selects the provider's fixed runtime behavior,
/// while any authored root or `text_config` value must still equal that effective value. This is
/// needed for published Qwen2.5-VL and Mistral3 configs which omit booleans that their concrete
/// loaders nevertheless implement deterministically.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderConfigBool {
    Required(bool),
    Optional(bool),
}

impl EncoderConfigBool {
    pub const fn effective(self) -> bool {
        match self {
            Self::Required(value) | Self::Optional(value) => value,
        }
    }

    pub const fn is_required(self) -> bool {
        matches!(self, Self::Required(_))
    }
}

/// What a text-encoder substitution is allowed to do with tokenization. Every currently supported
/// alternate is a language-weight replacement: the provider retains the base snapshot's tokenizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderTokenizerBinding {
    RetainBase,
}

/// One tokenizer literal whose numeric meaning is consumed directly by a provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderRequiredToken {
    pub role: &'static str,
    pub literal: &'static str,
    pub id: i64,
    /// Optional model-config field that must carry the same value wherever it is declared.
    pub config_field: Option<&'static str>,
}

/// Tokenizer identity and binding contract for one encoder family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderTokenizerContract {
    /// Diagnostic family name only. Compatibility is established by artifact evidence.
    pub family: &'static str,
    pub binding: EncoderTokenizerBinding,
    /// Relative candidate paths, in provider loader precedence order.
    pub artifact_candidates: &'static [&'static str],
    pub required_tokens: &'static [EncoderRequiredToken],
}

/// Stable identity for the prompt renderer used before tokenization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderPromptTemplate {
    KreaQwen3Vl,
    KreaQwen3VlEdit,
    QwenImage,
    QwenImageEdit,
    QwenInstruct,
    QwenInstructNoThink,
    Flux2DevMistral,
    Flux2DevCaptionUpsample,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderPromptLengthPolicy {
    Unbounded,
    RightTruncate { max_tokens: usize },
    RejectAbove { max_tokens: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderPromptPadding {
    None,
    RightToMax { pad_token_id: i64 },
}

/// One concrete prompt rendering, tokenization, padding, and output-selection policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderPromptExecutionContract {
    pub purpose: &'static str,
    pub template: EncoderPromptTemplate,
    pub add_special_tokens: bool,
    pub length: EncoderPromptLengthPolicy,
    pub padding: EncoderPromptPadding,
    pub prefix_trim: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderContract {
    pub architecture: &'static str,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub output_width: usize,
    /// Number of decoder blocks the concrete loader actually constructs. Prompt-only backends can
    /// intentionally stop at the highest selected hidden state even when the authored architecture
    /// declares more layers.
    pub loaded_hidden_layers: usize,
    /// Whether the concrete route reads the final decoder norm. Intermediate-state prompt routes do
    /// not; Qwen-Image and FLUX.2-dev caption generation do.
    pub requires_final_norm: bool,
    /// Whether the consuming loader also constructs the decoder LM head from
    /// `<language-model-prefix>.lm_head.weight` (FLUX.2-dev caption upsampling).
    pub requires_lm_head: bool,
    /// Exact behavior-bearing config consumed by the frozen encoder implementation.
    pub hidden_activation: &'static str,
    pub attention_dropout: EncoderConfigFloat,
    pub rms_norm_eps: EncoderConfigFloat,
    /// Runtime epsilon for per-head q/k RMSNorm when that operation exists. It is the epsilon the
    /// checkpoint's own config declares for that norm — for the Qwen3-family encoders that is
    /// `rms_norm_eps`, because `Qwen3Attention` builds `q_norm`/`k_norm` from it and the released
    /// configs declare no separate qk-norm key. A backend whose runtime uses its library's default
    /// instead has a defect here, not a variant: that is what sc-17137's review found in MLX
    /// Z-Image (1e-5 against the checkpoint's 1e-6). It is a separate field because an architecture
    /// that *does* publish a distinct qk-norm epsilon must be able to say so.
    pub qk_norm_eps: Option<EncoderConfigFloat>,
    pub rope_theta: EncoderConfigFloat,
    pub max_position_embeddings: usize,
    pub attention_bias: EncoderConfigBool,
    pub tie_word_embeddings: EncoderConfigBool,
    /// Exact retained-tokenizer identity and evidence policy.
    pub tokenizer: EncoderTokenizerContract,
    /// Exact prompt execution policies for every route/purpose sharing this encoder contract.
    pub prompt_executions: &'static [EncoderPromptExecutionContract],
    /// Behavior-bearing model-config IDs. These complement, but never replace, tokenizer-artifact
    /// validation through [`EncoderTokenizerContract::required_tokens`].
    pub bos_token_id: Option<i64>,
    pub eos_token_id: Option<i64>,
    pub image_token_id: Option<i64>,
    pub vision_start_token_id: Option<i64>,
    pub vision_end_token_id: Option<i64>,
    /// Empty for ordinary 1-D RoPE; otherwise the exact multimodal RoPE partition.
    pub mrope_section: &'static [usize],
    /// `Some(true)` for Qwen3-VL's interleaved multimodal RoPE. `None` means ordinary 1-D/default
    /// partitioning and rejects an explicit interleaved declaration.
    pub mrope_interleaved: Option<bool>,
    /// Provider-fixed hidden-state selection (one-based hidden-state list indices). This is part of
    /// the descriptor even when a component-only alternate has no pipeline manifest to restate it.
    pub selected_hidden_layers: &'static [usize],
    /// `None` is dense-only. `Some` describes the exact packed surface used by Q4/Q8 tiers: all
    /// decoder projection matrices plus the explicitly marked embedding/LM-head fields.
    pub packing: Option<EncoderPackingContract>,
    /// A tensor whose storage dtype controls the entire dense matrix/embedding store in a backend.
    /// Only Candle Krea has this sentinel-wide cast behavior; norms are explicitly loaded as f32 and
    /// excluded from the equality check.
    pub dense_storage_dtype_probe: Option<&'static str>,
}

/// A caller-provided encoder source after resolving whether a directory names the component itself
/// or a complete snapshot containing `text_encoder/`.
#[derive(Clone, Debug)]
pub struct ValidatedEncoderSource {
    pub(super) requested_source: WeightsSource,
    pub(super) weights: WeightsSource,
    pub(super) config_path: Option<PathBuf>,
    pub(super) packed_quant: Option<PackedQuantization>,
    pub(super) pinned: PinnedEncoderSource,
    pub(super) tokenizer: Option<ValidatedTokenizerSource>,
}

/// Non-authorizing compatibility facts captured from bounded config/header inspection.
///
/// These values are suitable for catalog and memory projection only. They contain no paths, pins,
/// or artifact receipt and therefore cannot be used to open or materialize a source. Executable
/// preparation must still acquire a [`ValidatedEncoderSource`] and retain its complete seal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderDiscoveryFacts {
    pub(super) materialized_language_tensor_headers: Vec<SafetensorsTensorHeader>,
    pub(super) source_bytes: u64,
}

impl EncoderDiscoveryFacts {
    /// Exact language tensor surface consumed by the validated contract, excluding unused tails
    /// and unrelated tensors in a shared multimodal checkpoint.
    pub fn materialized_language_tensor_headers(&self) -> &[SafetensorsTensorHeader] {
        &self.materialized_language_tensor_headers
    }

    /// Current direct-shard bytes for the source inventory inspected by discovery.
    pub fn source_bytes(&self) -> u64 {
        self.source_bytes
    }
}

/// Non-authorizing direct-shard metadata for source classification and weights-free fallback
/// pricing. Unlike [`EncoderDiscoveryFacts`], this inventory has not asserted an architecture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextEncoderDiscoveryInventory {
    pub(super) tensor_headers: Vec<SafetensorsTensorHeader>,
    pub(super) source_bytes: u64,
}

impl TextEncoderDiscoveryInventory {
    pub fn tensor_headers(&self) -> &[SafetensorsTensorHeader] {
        &self.tensor_headers
    }

    pub fn source_bytes(&self) -> u64 {
        self.source_bytes
    }
}

/// Path-free source shape used only for selected-encoder memory planning.
///
/// This classification lets a caller preserve the loader's complete-snapshot deduplication rule
/// without receiving a resolved path or any artifact authorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextEncoderSourceLayout {
    File,
    DirectDirectory,
    CompleteSnapshot,
}

/// Non-authorizing selected-encoder facts for memory planning.
///
/// A prepared [`crate::LoadSpec`] derives these values from its retained acquisition receipt;
/// compatibility callers may derive them from bounded discovery. The fact contains no paths, pins,
/// hashes, or reopen capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextEncoderPlanningFacts {
    pub(super) source_layout: TextEncoderSourceLayout,
    pub(super) direct_shard_bytes: u64,
}

impl TextEncoderPlanningFacts {
    pub fn source_layout(&self) -> TextEncoderSourceLayout {
        self.source_layout
    }

    pub fn direct_shard_bytes(&self) -> u64 {
        self.direct_shard_bytes
    }
}

/// Exact selected-encoder source shape and direct-shard inventory exported onto a prepared
/// [`crate::LoadSpec`]. File pins guard every accepted object; this companion receipt preserves the
/// directory-enumeration invariant so a later shard addition, removal, rename, or type change cannot
/// enter a provider load without changing the prepared identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PreparedEncoderLoadReceipt {
    pub(super) requested_source: WeightsSource,
    pub(super) pinned: PinnedEncoderSource,
    pub(super) tokenizer: Option<ValidatedTokenizerSource>,
}

/// How a validated encoder source relates to the provider's retained tokenizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderTokenizerDisposition {
    /// A direct component/File is a weights-only replacement and inherits the base tokenizer.
    InheritedBase,
    /// A complete selected snapshot carried an exactly matching tokenizer artifact.
    MatchedSelectedTokenizer,
}

/// Exact tokenizer artifact(s) pinned by encoder validation. Production parsers consume the base
/// path only through [`Self::read_unchanged`], so the bytes validated for literal IDs and semantic
/// identity are the bytes parsed at runtime.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatedTokenizerSource {
    pub(super) base: PinnedTokenizerArtifact,
    pub(super) base_candidates: Vec<PathBuf>,
    pub(super) selected: Option<PinnedTokenizerArtifact>,
    pub(super) selected_candidates: Option<Vec<PathBuf>>,
    pub(super) disposition: EncoderTokenizerDisposition,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PinnedTokenizerArtifact {
    pub(super) pin: PinnedWeightsFile,
    pub(super) semantic_sha256: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct PinnedEncoderSource {
    pub(super) weights: WeightsSource,
    pub(super) shard_paths: Vec<PathBuf>,
    pub(super) shard_pins: Vec<PinnedWeightsFile>,
    pub(super) config_candidate: Option<PathBuf>,
    pub(super) config_pin: Option<PinnedWeightsFile>,
}
