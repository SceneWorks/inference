//! Text-encoder architecture contracts and header-only substitution validation.
//!
//! A generator descriptor advertises the exact conditioning encoder its denoiser was trained
//! against.  Callers may replace the bundled weights through [`crate::LoadSpec::text_encoder`], but
//! only after this module has compared both the Hugging Face `config.json` and the safetensors tensor
//! headers with that advertised contract.  Validation deliberately happens before either tensor
//! backend opens/materializes weights, so an incompatible encoder is a small, legible load error
//! rather than a late matmul-shape failure (or, worse, a plausible render with the wrong conditioning).

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::weightsmeta::Dtype;
use crate::{
    safetensors_path_tensor_headers, Error, PinnedWeightsFile, Result, SafetensorsTensorHeader,
    VisionEncoderContract, WeightsSource,
};

/// Selected-encoder behavior configs are metadata, not model payloads. Both bounded discovery and
/// executable validation apply this same limit so a candidate cannot pass one admission seam and
/// fail the other. Executable validation still seals the complete accepted config and every weights
/// shard through [`PinnedEncoderSource`].
const MAX_SELECTED_ENCODER_CONFIG_BYTES: u64 = 4 * 1024 * 1024;

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
    requested_source: WeightsSource,
    weights: WeightsSource,
    config_path: Option<PathBuf>,
    packed_quant_bits: Option<i32>,
    pinned: PinnedEncoderSource,
    tokenizer: Option<ValidatedTokenizerSource>,
}

/// Non-authorizing compatibility facts captured from bounded config/header inspection.
///
/// These values are suitable for catalog and memory projection only. They contain no paths, pins,
/// or artifact receipt and therefore cannot be used to open or materialize a source. Executable
/// preparation must still acquire a [`ValidatedEncoderSource`] and retain its complete seal.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderDiscoveryFacts {
    materialized_language_tensor_headers: Vec<SafetensorsTensorHeader>,
    source_bytes: u64,
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
    tensor_headers: Vec<SafetensorsTensorHeader>,
    source_bytes: u64,
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
    source_layout: TextEncoderSourceLayout,
    direct_shard_bytes: u64,
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
    requested_source: WeightsSource,
    pinned: PinnedEncoderSource,
    tokenizer: Option<ValidatedTokenizerSource>,
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
    base: PinnedTokenizerArtifact,
    base_candidates: Vec<PathBuf>,
    selected: Option<PinnedTokenizerArtifact>,
    selected_candidates: Option<Vec<PathBuf>>,
    disposition: EncoderTokenizerDisposition,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PinnedTokenizerArtifact {
    pin: PinnedWeightsFile,
    semantic_sha256: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PinnedEncoderSource {
    weights: WeightsSource,
    shard_paths: Vec<PathBuf>,
    shard_pins: Vec<PinnedWeightsFile>,
    config_candidate: Option<PathBuf>,
    config_pin: Option<PinnedWeightsFile>,
}

impl ValidatedEncoderSource {
    /// Prove that every lexical loader entry and exact canonical target retained by this receipt is
    /// under one of the caller's pre-authorized model roots.
    pub fn ensure_confined_to(&self, allowed_roots: &[PathBuf]) -> Result<()> {
        self.ensure_unchanged()?;
        let roots = allowed_roots
            .iter()
            .map(std::path::absolute)
            .collect::<std::io::Result<Vec<_>>>()?;
        if roots.is_empty() {
            return Err(Error::Unsupported(
                "validated text encoder has no authorized model roots".into(),
            ));
        }
        for pin in self.receipt_pins()? {
            for (kind, path) in [
                ("loader entry", pin.loader_path()),
                ("canonical target", pin.canonical_target_path()),
            ] {
                if !roots.iter().any(|root| path.starts_with(root)) {
                    return Err(Error::Unsupported(format!(
                        "validated text encoder {kind} escapes authorized model roots: {}",
                        path.display()
                    )));
                }
            }
        }
        self.ensure_unchanged()
    }

    /// Attach the exact validated source and its complete shard/config/tokenizer receipt to a load
    /// spec in one atomic operation.
    ///
    /// The requested source shape is preserved: a complete snapshot remains a complete-snapshot
    /// `Dir`, while validation continues to expose only its resolved `text_encoder/` component to
    /// providers. Every direct shard plus the behavior config and retained/selected tokenizer files
    /// enters the prepared identity set used by cache keys and provider load brackets. This is the
    /// only public receipt-export seam; callers never recreate the contract's shard inventory.
    pub fn prepare_load_spec(&self, spec: &mut crate::LoadSpec) -> Result<()> {
        self.ensure_unchanged()?;
        let mut candidate = spec.clone();
        if let Some(existing) = candidate.text_encoder.as_ref() {
            if existing != &self.requested_source {
                return Err(Error::Unsupported(format!(
                    "validated text encoder cannot replace an already selected source: existing {}, validated {}",
                    source_path(existing).display(),
                    source_path(&self.requested_source).display()
                )));
            }
        } else {
            candidate.text_encoder = Some(self.requested_source.clone());
        }

        let receipt_pins = self.receipt_pins()?;
        let receipt_by_path = receipt_pins
            .iter()
            .map(|pin| (pin.loader_path().to_path_buf(), pin))
            .collect::<BTreeMap<_, _>>();
        let mut prepared = Vec::new();
        for path in candidate.file_source_paths() {
            let absolute = std::path::absolute(path)?;
            if let Some(pin) = receipt_by_path.get(&absolute) {
                prepared.push((*pin).clone());
            } else if let Some(pin) = candidate.prepared_file_pins().get(&absolute) {
                pin.ensure_unchanged()?;
                prepared.push(pin.clone());
            } else {
                prepared.push(PinnedWeightsFile::pin(path)?);
            }
        }
        prepared.extend(receipt_pins.iter().cloned());
        let receipt_paths = receipt_pins
            .iter()
            .map(|pin| pin.loader_path().to_path_buf())
            .collect::<Vec<_>>();
        candidate.prepare_with_validated_receipt_pins(
            prepared,
            receipt_paths,
            PreparedEncoderLoadReceipt {
                requested_source: self.requested_source.clone(),
                pinned: self.pinned.clone(),
                tokenizer: self.tokenizer.clone(),
            },
        )?;
        self.ensure_unchanged()?;
        *spec = candidate;
        Ok(())
    }

    fn receipt_pins(&self) -> Result<Vec<PinnedWeightsFile>> {
        self.ensure_unchanged()?;
        let mut pins = self.pinned.shard_pins.clone();
        pins.extend(self.pinned.config_pin.iter().cloned());
        if let Some(tokenizer) = &self.tokenizer {
            pins.push(tokenizer.base.pin.clone());
            pins.extend(
                tokenizer
                    .selected
                    .iter()
                    .map(|selected| selected.pin.clone()),
            );
        }
        pins.sort_by(|left, right| left.loader_path().cmp(right.loader_path()));
        pins.dedup_by(|left, right| left.loader_path() == right.loader_path());
        self.ensure_unchanged()?;
        Ok(pins)
    }

    /// Whether an authoritative behavior config accompanied the source. The path itself stays
    /// encapsulated so callers cannot reopen it outside the unchanged-read bracket.
    pub fn has_config(&self) -> bool {
        self.config_path.is_some()
    }

    /// Derive the load-time conversion from the quantization evidence retained by this exact
    /// validated source. This prevents a swap/restore between a second metadata read and payload
    /// load from selecting the conversion for different bytes.
    pub fn load_time_quant_bits(
        &self,
        expected_bits: Option<i32>,
        provider_id: &str,
    ) -> Result<Option<i32>> {
        self.ensure_unchanged()?;
        resolve_encoder_load_time_quant_bits(self.packed_quant_bits, expected_bits, provider_id)
    }

    /// Exact bytes in the direct shard inventory accepted by both backends. Nested files are not
    /// loadable shards and therefore cannot inflate conditioning memory facts.
    pub fn source_bytes(&self) -> Result<u64> {
        self.ensure_unchanged()?;
        let bytes = self.pinned.direct_shard_bytes()?;
        self.ensure_unchanged()?;
        Ok(bytes)
    }

    /// Retained tokenizer receipt. `None` is limited to metadata-only validation helpers; every
    /// production source path binds against a base snapshot.
    pub fn tokenizer_source(&self) -> Option<&ValidatedTokenizerSource> {
        self.tokenizer.as_ref()
    }

    pub fn tokenizer_path(&self) -> Option<&Path> {
        self.tokenizer.as_ref().map(ValidatedTokenizerSource::path)
    }

    pub fn tokenizer_disposition(&self) -> Option<EncoderTokenizerDisposition> {
        self.tokenizer
            .as_ref()
            .map(ValidatedTokenizerSource::disposition)
    }

    /// Parse the retained runtime tokenizer inside the same exact pin bracket as encoder loading.
    pub fn read_tokenizer_unchanged<T, E>(
        &self,
        read: impl FnOnce(&Path) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E>
    where
        E: From<Error>,
    {
        self.tokenizer
            .as_ref()
            .ok_or_else(|| {
                Error::Unsupported(
                    "validated encoder source has no retained tokenizer receipt".into(),
                )
            })?
            .read_unchanged(read)
    }

    /// Exact tensor headers from the direct shard inventory accepted by the selected-encoder
    /// loaders. Memory projections must use this receipt instead of recursively walking the source
    /// path, because nested safetensors are not loadable shards.
    pub fn tensor_headers(&self) -> Result<Vec<SafetensorsTensorHeader>> {
        self.pinned.ensure_unchanged()?;
        let headers = self.pinned.headers()?;
        self.pinned.ensure_unchanged()?;
        Ok(headers)
    }

    /// Validate the vision half against the same pinned config and shard inventory already accepted
    /// for the language half. The returned receipt remains this source; callers must still use
    /// [`Self::read_unchanged`] for materialization.
    pub fn validate_vision(
        &self,
        vision: &VisionEncoderContract,
        language: &EncoderContract,
    ) -> Result<()> {
        self.ensure_unchanged()?;
        let config_pin = self.pinned.config_pin.as_ref().ok_or_else(|| {
            Error::Unsupported(format!(
                "multimodal vision encoder requires authoritative config.json for {}",
                source_path(&self.weights).display()
            ))
        })?;
        let config: Value = config_pin.read_unchanged(|path| {
            let bytes = std::fs::read(path).map_err(|error| {
                Error::Msg(format!(
                    "vision encoder contract: read {}: {error}",
                    path.display()
                ))
            })?;
            serde_json::from_slice(&bytes).map_err(|error| {
                Error::Msg(format!(
                    "vision encoder contract: parse {}: {error}",
                    path.display()
                ))
            })
        })?;
        vision.validate_config(&config, config_pin.loader_path(), language)?;
        let headers = self.pinned.headers()?;
        vision.validate_tensor_headers(&headers, source_path(&self.weights))?;
        self.ensure_unchanged()
    }

    /// Exact language tensor surface consumed by this contract, excluding unused authored layers
    /// and unrelated `visual.*` tensors that share the same multimodal checkpoint.
    pub fn materialized_language_tensor_headers(
        &self,
        contract: &EncoderContract,
    ) -> Result<Vec<SafetensorsTensorHeader>> {
        self.ensure_unchanged()?;
        let headers = self.pinned.headers()?;
        let packing = if self.packed_quant_bits.is_some() {
            Some(contract.packing.ok_or_else(|| {
                Error::Unsupported(format!(
                    "validated packed encoder has no packing contract for architecture {}",
                    contract.architecture
                ))
            })?)
        } else {
            None
        };
        let expected = contract.materialized_language_tensor_names(packing)?;
        let selected = headers
            .into_iter()
            .filter(|header| expected.contains(&header.name))
            .collect();
        self.ensure_unchanged()?;
        Ok(selected)
    }

    /// Exact vision tensor surface consumed by a previously validated multimodal source.
    pub fn materialized_vision_tensor_headers(
        &self,
        vision: &VisionEncoderContract,
        language: &EncoderContract,
    ) -> Result<Vec<SafetensorsTensorHeader>> {
        self.validate_vision(vision, language)?;
        let expected = vision
            .expected_headers()?
            .into_iter()
            .map(|(name, _)| name)
            .collect::<BTreeSet<_>>();
        let selected = self
            .pinned
            .headers()?
            .into_iter()
            .filter(|header| expected.contains(&header.name))
            .collect();
        self.ensure_unchanged()?;
        Ok(selected)
    }

    /// Execute one backend load against the exact config and direct-shard set that passed contract
    /// validation. This brackets the backend's own directory enumeration so a persistent shard
    /// addition, removal, retarget, or replacement cannot turn validated bytes into different bytes.
    pub fn read_unchanged<T, E>(
        &self,
        read: impl FnOnce(&WeightsSource) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E>
    where
        E: From<Error>,
    {
        self.ensure_unchanged().map_err(E::from)?;
        let result = read(&self.weights);
        self.ensure_unchanged().map_err(E::from)?;
        result
    }

    fn ensure_unchanged(&self) -> Result<()> {
        self.pinned.ensure_unchanged()?;
        if let Some(tokenizer) = &self.tokenizer {
            tokenizer.ensure_unchanged()?;
        }
        Ok(())
    }
}

impl PreparedEncoderLoadReceipt {
    pub(crate) fn ensure_unchanged_for(
        &self,
        selected_source: Option<&WeightsSource>,
    ) -> Result<()> {
        if selected_source != Some(&self.requested_source) {
            return Err(Error::Unsupported(
                "prepared text-encoder receipt no longer matches the LoadSpec source".into(),
            ));
        }
        let (resolved, _) = resolve_source(&self.requested_source)?;
        if resolved != self.pinned.weights {
            return Err(Error::Unsupported(format!(
                "prepared text-encoder source shape changed: expected {}, got {}",
                source_path(&self.pinned.weights).display(),
                source_path(&resolved).display()
            )));
        }
        self.pinned.ensure_unchanged().map_err(|error| {
            Error::Unsupported(format!("prepared text-encoder receipt changed: {error}"))
        })?;
        if let Some(tokenizer) = &self.tokenizer {
            tokenizer.ensure_unchanged().map_err(|error| {
                Error::Unsupported(format!("prepared text-encoder receipt changed: {error}"))
            })?;
        }
        Ok(())
    }

    pub(crate) fn planning_facts_for(
        &self,
        selected_source: Option<&WeightsSource>,
    ) -> Result<TextEncoderPlanningFacts> {
        self.ensure_unchanged_for(selected_source)?;
        let direct_shard_bytes = self
            .pinned
            .shard_pins
            .iter()
            .try_fold(0_u64, |total, pin| {
                total
                    .checked_add(pin.target_fingerprint().size)
                    .ok_or_else(|| {
                        Error::Unsupported(
                            "text encoder direct-shard byte total overflowed u64".into(),
                        )
                    })
            })?;
        let facts = TextEncoderPlanningFacts {
            source_layout: resolved_source_layout(&self.requested_source, &self.pinned.weights)?,
            direct_shard_bytes,
        };
        self.ensure_unchanged_for(selected_source)?;
        Ok(facts)
    }
}

impl ValidatedTokenizerSource {
    pub fn path(&self) -> &Path {
        self.base.pin.loader_path()
    }

    pub fn disposition(&self) -> EncoderTokenizerDisposition {
        self.disposition
    }

    pub fn semantic_sha256(&self) -> [u8; 32] {
        self.base.semantic_sha256
    }

    pub fn read_unchanged<T, E>(
        &self,
        read: impl FnOnce(&Path) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E>
    where
        E: From<Error>,
    {
        self.ensure_unchanged().map_err(E::from)?;
        let result = read(self.path());
        self.ensure_unchanged().map_err(E::from)?;
        result
    }

    fn ensure_unchanged(&self) -> Result<()> {
        ensure_tokenizer_resolution_unchanged(
            &self.base_candidates,
            self.base.pin.loader_path(),
            "base snapshot",
        )?;
        self.base.pin.ensure_unchanged()?;
        if let Some(selected) = &self.selected {
            ensure_tokenizer_resolution_unchanged(
                self.selected_candidates.as_deref().ok_or_else(|| {
                    Error::Unsupported(
                        "selected tokenizer receipt has no candidate inventory".into(),
                    )
                })?,
                selected.pin.loader_path(),
                "selected complete snapshot",
            )?;
            selected.pin.ensure_unchanged()?;
        }
        Ok(())
    }
}

impl PinnedEncoderSource {
    fn pin(weights: &WeightsSource, config_candidate: Option<PathBuf>) -> Result<Self> {
        let shard_paths = encoder_shard_paths(weights)?;
        let shard_pins = shard_paths
            .iter()
            .map(PinnedWeightsFile::pin)
            .collect::<Result<Vec<_>>>()?;
        let config_pin = config_candidate
            .as_ref()
            .filter(|path| path.is_file())
            .map(PinnedWeightsFile::pin)
            .transpose()?;
        let pinned = Self {
            weights: weights.clone(),
            shard_paths,
            shard_pins,
            config_candidate,
            config_pin,
        };
        pinned.ensure_unchanged()?;
        Ok(pinned)
    }

    fn ensure_unchanged(&self) -> Result<()> {
        let current = encoder_shard_paths(&self.weights)?;
        if current != self.shard_paths {
            return Err(Error::Unsupported(format!(
                "text encoder shard inventory changed after validation: expected {:?}, got {:?}",
                self.shard_paths, current
            )));
        }
        let config_present = self
            .config_candidate
            .as_ref()
            .is_some_and(|path| path.is_file());
        if config_present != self.config_pin.is_some() {
            return Err(Error::Unsupported(format!(
                "text encoder config presence changed after validation: {}",
                self.config_candidate
                    .as_deref()
                    .unwrap_or_else(|| Path::new("<none>"))
                    .display()
            )));
        }
        if let Some(pin) = &self.config_pin {
            pin.ensure_unchanged()?;
        }
        for pin in &self.shard_pins {
            pin.ensure_unchanged()?;
        }
        Ok(())
    }

    fn headers(&self) -> Result<Vec<SafetensorsTensorHeader>> {
        collect_unique_encoder_headers(
            self.shard_pins
                .iter()
                .map(|pin| pin.read_unchanged(|path| safetensors_path_tensor_headers(path))),
        )
    }

    fn direct_shard_bytes(&self) -> Result<u64> {
        self.ensure_unchanged()?;
        let mut total = 0u64;
        for pin in &self.shard_pins {
            let bytes = pin.read_unchanged(|path| {
                std::fs::metadata(path)
                    .map(|metadata| metadata.len())
                    .map_err(|error| {
                        Error::Msg(format!(
                            "text encoder contract: stat direct shard {}: {error}",
                            path.display()
                        ))
                    })
            })?;
            total = total.checked_add(bytes).ok_or_else(|| {
                Error::Unsupported("text encoder direct-shard byte total overflowed u64".into())
            })?;
        }
        self.ensure_unchanged()?;
        Ok(total)
    }

    fn read_unchanged<T>(&self, read: impl FnOnce(&WeightsSource) -> Result<T>) -> Result<T> {
        self.ensure_unchanged()?;
        let result = read(&self.weights)?;
        self.ensure_unchanged()?;
        Ok(result)
    }
}

fn collect_unique_encoder_headers(
    inventories: impl IntoIterator<Item = Result<Vec<SafetensorsTensorHeader>>>,
) -> Result<Vec<SafetensorsTensorHeader>> {
    let mut headers = BTreeMap::new();
    for inventory in inventories {
        for header in inventory? {
            if headers
                .insert(header.name.clone(), header.clone())
                .is_some()
            {
                return Err(Error::Unsupported(format!(
                    "text encoder source contains duplicate tensor key {:?} across direct shards; selected encoders must be accepted identically by the strict MLX and Candle loaders",
                    header.name
                )));
            }
        }
    }
    Ok(headers.into_values().collect())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PackedQuantization {
    bits: usize,
    group_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EncoderDtypePolicy {
    Native,
    ComfyUiFp8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EncoderConfigPolicy {
    /// Ordinary alternates must supply their own behavior-bearing config.
    Required,
    /// Legacy ComfyUI exports contain only tensors. Their consuming route freezes the provider's
    /// config/tokenizer policy, so a present sibling config is checked but its absence is allowed.
    ProviderOwnedComfyUi,
}

#[derive(Clone, Copy, Debug)]
struct MatrixExpectation<'a> {
    name: &'a str,
    field: &'static str,
    shape: [usize; 2],
    must_be_packed: bool,
}

impl EncoderDtypePolicy {
    fn accepts_dense(self, dtype: Dtype) -> bool {
        matches!(dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
            || (self == Self::ComfyUiFp8 && dtype == Dtype::F8_E4M3)
    }

    fn expected_dense(self) -> &'static str {
        match self {
            Self::Native => "F16, BF16, or F32",
            Self::ComfyUiFp8 => "F8_E4M3, F16, BF16, or F32",
        }
    }
}

/// Resolve the load-time action needed to keep a selected text encoder on the provider's numeric
/// policy. `expected_bits` is the effective base-model policy (`None` = dense, `Some(4|8)` = packed).
///
/// A matching pre-packed encoder needs no conversion, while a dense encoder inherits the effective
/// Q4/Q8 tier at load. Any packed-vs-expected mismatch is rejected because backend `quantize`
/// implementations intentionally no-op on packed weights and would otherwise silently serve the
/// wrong tier.
#[cfg(test)]
fn text_encoder_load_time_quant_bits(
    source: &WeightsSource,
    expected_bits: Option<i32>,
    provider_id: &str,
) -> Result<Option<i32>> {
    let packed_bits = text_encoder_packed_quant_bits(source)?;
    resolve_encoder_load_time_quant_bits(packed_bits, expected_bits, provider_id)
}

/// Resolve the pure packed-source versus provider-tier policy used by
/// [`ValidatedEncoderSource::load_time_quant_bits`].
///
/// Keeping this independent of filesystem receipts lets memory projection and catalog tests cover
/// the complete policy matrix from already validated header facts. Production source admission
/// still owns artifact sealing and calls the same resolver only after validation has established
/// `packed_bits` from the selected source.
pub fn resolve_encoder_load_time_quant_bits(
    packed_bits: Option<i32>,
    expected_bits: Option<i32>,
    provider_id: &str,
) -> Result<Option<i32>> {
    if let Some(bits) = expected_bits {
        if !matches!(bits, 4 | 8) {
            return Err(Error::Unsupported(format!(
                "{provider_id}: unsupported text encoder quantization policy Q{bits}; expected Q4 or Q8"
            )));
        }
    }
    match (packed_bits, expected_bits) {
        (Some(packed), Some(expected)) if packed == expected => Ok(None),
        (Some(packed), Some(expected)) => Err(Error::Unsupported(format!(
            "{provider_id}: selected text encoder is pre-quantized Q{packed} but the model policy is Q{expected}; quantize is a no-op on packed weights so this would silently serve Q{packed}. Select a Q{expected} or dense compatible encoder."
        ))),
        (Some(packed), None) => Err(Error::Unsupported(format!(
            "{provider_id}: selected text encoder is pre-quantized Q{packed} but the model policy is dense; select a dense compatible encoder."
        ))),
        (None, Some(expected)) => Ok(Some(expected)),
        (None, None) => Ok(None),
    }
}

/// Read a selected encoder component's optional Q4/Q8 marker. Missing `quantization` means dense;
/// malformed or unsupported markers fail closed. Header validation remains separately authoritative
/// for architecture and shape.
pub fn text_encoder_packed_quant_bits(source: &WeightsSource) -> Result<Option<i32>> {
    let (resolved, config_path) = resolve_source(source)?;
    let pinned = PinnedEncoderSource::pin(&resolved, config_path.clone())?;
    let declared = match pinned.config_pin.as_ref() {
        Some(pin) => {
            let bytes = pin.read_unchanged(|path| {
                std::fs::read(path).map_err(|error| {
                    Error::Msg(format!(
                        "text encoder quantization: read {}: {error}",
                        path.display()
                    ))
                })
            })?;
            let config: Value = serde_json::from_slice(&bytes).map_err(|error| {
                Error::Msg(format!(
                    "text encoder quantization: parse {}: {error}",
                    config_path
                        .as_deref()
                        .unwrap_or_else(|| Path::new("<none>"))
                        .display()
                ))
            })?;
            parse_packed_quantization(
                &config,
                config_path
                    .as_deref()
                    .unwrap_or_else(|| Path::new("<none>")),
            )?
        }
        None => None,
    };
    let headers = pinned.headers()?;
    let language_headers = language_quantization_evidence_headers(&headers);
    let quant =
        validate_quantization_evidence(&language_headers, source_path(&resolved), declared)?;
    pinned.ensure_unchanged()?;
    Ok(quant.map(|quant| quant.bits as i32))
}

/// Exact safetensors bytes for the encoder component named by `source`. A complete snapshot Dir is
/// resolved to its `text_encoder/` child, and the direct shard/config inventory is pinned around the
/// read so memory facts cannot price a different tree than the selected loader later consumes.
pub fn read_text_encoder_source_unchanged<T>(
    source: &WeightsSource,
    read: impl FnOnce(&WeightsSource) -> Result<T>,
) -> Result<T> {
    let (resolved, config_path) = resolve_source(source)?;
    let pinned = PinnedEncoderSource::pin(&resolved, config_path)?;
    pinned.read_unchanged(read)
}

pub fn text_encoder_source_bytes(source: &WeightsSource) -> Result<u64> {
    let (resolved, config_path) = resolve_source(source)?;
    PinnedEncoderSource::pin(&resolved, config_path)?.direct_shard_bytes()
}

/// Exact direct-shard tensor inventory for an encoder source, without asserting an architecture.
/// This is the header analogue of [`text_encoder_source_bytes`] for weights-free catalog pricing:
/// it resolves complete snapshots to `text_encoder/`, excludes nested files exactly like the
/// concrete loaders, and pins the inventory across inspection.
pub fn text_encoder_source_tensor_headers(
    source: &WeightsSource,
) -> Result<Vec<SafetensorsTensorHeader>> {
    let (resolved, config_path) = resolve_source(source)?;
    let pinned = PinnedEncoderSource::pin(&resolved, config_path)?;
    pinned.ensure_unchanged()?;
    let headers = pinned.headers()?;
    pinned.ensure_unchanged()?;
    Ok(headers)
}

/// Inspect one encoder source's direct shards for catalog classification or weights-free pricing
/// without acquiring an executable artifact seal. The returned inventory is non-authorizing and
/// has not asserted a particular [`EncoderContract`]; callers that have behavior config must prefer
/// [`EncoderContract::validate_source_for_discovery`].
pub fn text_encoder_source_inventory_for_discovery(
    source: &WeightsSource,
    allowed_roots: &[PathBuf],
) -> Result<TextEncoderDiscoveryInventory> {
    let inspected = inspect_encoder_source_for_discovery(source, allowed_roots, false, true)?;
    Ok(TextEncoderDiscoveryInventory {
        tensor_headers: inspected.headers,
        source_bytes: inspected.source_bytes,
    })
}

/// Inspect path-free selected-encoder planning facts without acquiring an executable artifact
/// seal. `allowed_roots` confines every lexical entry and canonical target before traversal; the
/// returned value deliberately cannot authorize a later load.
pub fn text_encoder_planning_facts_for_discovery(
    source: &WeightsSource,
    allowed_roots: &[PathBuf],
) -> Result<TextEncoderPlanningFacts> {
    let inspected = inspect_encoder_source_for_discovery(source, allowed_roots, false, false)?;
    Ok(TextEncoderPlanningFacts {
        source_layout: resolved_source_layout(source, &inspected.weights)?,
        direct_shard_bytes: inspected.source_bytes,
    })
}

impl EncoderContract {
    /// Project a previously contract-validated dense header inventory onto the exact language
    /// tensor names retained by this encoder's constructors. This is the embedded-checkpoint twin
    /// of [`ValidatedEncoderSource::materialized_language_tensor_headers`]: fused ComfyUI sources
    /// retain their own file pin, so they cannot produce a standalone encoder receipt, but memory
    /// planning still must exclude validated yet unmaterialized tails and unrelated extras.
    pub fn materialized_dense_language_tensor_headers(
        &self,
        validated_headers: &[SafetensorsTensorHeader],
    ) -> Result<Vec<SafetensorsTensorHeader>> {
        let expected = self.materialized_language_tensor_names(None)?;
        Ok(validated_headers
            .iter()
            .filter(|header| expected.contains(&header.name))
            .cloned()
            .collect())
    }

    /// Validate the provider-authored contract itself before it is advertised or used to inspect a
    /// caller's files. This keeps malformed descriptors from turning later shape arithmetic into a
    /// panic or from advertising an architecture for which no exhaustive header signature exists.
    pub fn validate_definition(&self) -> Result<()> {
        self.validate_tokenizer_definition()?;
        self.validate_prompt_execution_definition()?;
        if self.architecture.trim().is_empty()
            || self.hidden_activation.trim().is_empty()
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
            || self.vocab_size == 0
            || self.output_width == 0
            || self.loaded_hidden_layers == 0
            || self.loaded_hidden_layers > self.num_hidden_layers
            || self.max_position_embeddings == 0
            || !self.attention_dropout.get().is_finite()
            || self.attention_dropout.get() != 0.0
            || !self.rms_norm_eps.get().is_finite()
            || self.rms_norm_eps.get() <= 0.0
            || self
                .qk_norm_eps
                .is_some_and(|eps| !eps.get().is_finite() || eps.get() <= 0.0)
            || !self.rope_theta.get().is_finite()
            || self.rope_theta.get() <= 0.0
            || !self.output_width.is_multiple_of(self.hidden_size)
            || self.num_key_value_heads > self.num_attention_heads
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
            || self
                .num_attention_heads
                .checked_mul(self.head_dim)
                .is_none()
            || self
                .num_key_value_heads
                .checked_mul(self.head_dim)
                .is_none()
            || self.requires_lm_head && !self.requires_final_norm
            || self.selected_hidden_layers.is_empty()
            || self
                .selected_hidden_layers
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
            || self
                .selected_hidden_layers
                .iter()
                .any(|&layer| layer == 0 || layer > self.loaded_hidden_layers)
            || self
                .mrope_section
                .iter()
                .try_fold(0usize, |sum, &part| sum.checked_add(part))
                .is_none()
            || (!self.mrope_section.is_empty()
                && (!self.head_dim.is_multiple_of(2)
                    || self.mrope_section.iter().sum::<usize>() != self.head_dim / 2))
            || self.mrope_interleaved.is_some() && self.mrope_section.is_empty()
            || matches!(self.architecture, "qwen3" | "qwen3_vl_text") != self.qk_norm_eps.is_some()
            || [
                self.bos_token_id,
                self.eos_token_id,
                self.image_token_id,
                self.vision_start_token_id,
                self.vision_end_token_id,
            ]
            .into_iter()
            .flatten()
            .any(|token| match usize::try_from(token) {
                Ok(token) => token >= self.vocab_size,
                Err(_) => true,
            })
            || self.packing.is_some_and(|packing| {
                packing.group_size == 0
                    || !packing.group_size.is_power_of_two()
                    || !matches!(packing.group_size, 64)
            })
        {
            return Err(Error::Unsupported(format!(
                "invalid text encoder contract for {:?}: dimensions/config values and the exact loaded/selected-layer surface must be coherent, attention widths must not overflow, output_width ({}) must be a multiple of hidden_size ({}), attention heads ({}) must be divisible by key/value heads ({}), token ids and multimodal RoPE must fit their declared geometry, and packed loaders require group_size 64",
                self.architecture,
                self.output_width,
                self.hidden_size,
                self.num_attention_heads,
                self.num_key_value_heads
            )));
        }
        self.expected_header_prefix()?;
        Ok(())
    }

    fn validate_tokenizer_definition(&self) -> Result<()> {
        let tokenizer = self.tokenizer;
        if tokenizer.family.trim().is_empty() || tokenizer.artifact_candidates.is_empty() {
            return Err(Error::Unsupported(
                "text encoder tokenizer contract requires a family and at least one artifact candidate"
                    .into(),
            ));
        }
        let mut candidates = BTreeSet::new();
        for candidate in tokenizer.artifact_candidates {
            let path = Path::new(candidate);
            if candidate.trim().is_empty()
                || path.is_absolute()
                || !path
                    .components()
                    .all(|component| matches!(component, Component::Normal(_)))
                || !candidates.insert(*candidate)
            {
                return Err(Error::Unsupported(format!(
                    "invalid tokenizer artifact candidate {candidate:?} for family {}: candidates must be unique, non-empty, relative normal paths",
                    tokenizer.family
                )));
            }
        }

        let mut roles = BTreeSet::new();
        let mut literals = BTreeSet::new();
        for required in tokenizer.required_tokens {
            if required.role.trim().is_empty()
                || required.literal.is_empty()
                || required.id < 0
                || usize::try_from(required.id).map_or(true, |id| id >= self.vocab_size)
                || !roles.insert(required.role)
                || !literals.insert(required.literal)
                || required
                    .config_field
                    .is_some_and(|field| field.trim().is_empty())
            {
                return Err(Error::Unsupported(format!(
                    "invalid required tokenizer literal {:?} ({}) for family {}: roles/literals must be unique and token ids must fit vocab_size {}",
                    required.literal, required.id, tokenizer.family, self.vocab_size
                )));
            }
        }
        Ok(())
    }

    fn validate_prompt_execution_definition(&self) -> Result<()> {
        if self.prompt_executions.is_empty() {
            return Err(Error::Unsupported(
                "text encoder contract requires at least one prompt execution policy".into(),
            ));
        }
        let mut purposes = BTreeSet::new();
        for execution in self.prompt_executions {
            if execution.purpose.trim().is_empty() || !purposes.insert(execution.purpose) {
                return Err(Error::Unsupported(format!(
                    "invalid or duplicate text encoder prompt purpose {:?}",
                    execution.purpose
                )));
            }
            let finite_max = match execution.length {
                EncoderPromptLengthPolicy::Unbounded => None,
                EncoderPromptLengthPolicy::RightTruncate { max_tokens }
                | EncoderPromptLengthPolicy::RejectAbove { max_tokens } => Some(max_tokens),
            };
            if finite_max.is_some_and(|max| max == 0 || execution.prefix_trim >= max) {
                return Err(Error::Unsupported(format!(
                    "invalid text encoder prompt policy {:?}: finite max must be non-zero and exceed prefix_trim {}",
                    execution.purpose, execution.prefix_trim
                )));
            }
            if let EncoderPromptPadding::RightToMax { pad_token_id } = execution.padding {
                if !matches!(
                    execution.length,
                    EncoderPromptLengthPolicy::RightTruncate { .. }
                ) || !self
                    .tokenizer
                    .required_tokens
                    .iter()
                    .any(|required| required.id == pad_token_id)
                {
                    return Err(Error::Unsupported(format!(
                        "invalid text encoder prompt policy {:?}: right-padding requires a finite right-truncation max and a declared pad-token literal for id {pad_token_id}",
                        execution.purpose
                    )));
                }
            }
        }
        Ok(())
    }

    /// Pin and validate the tokenizer a provider retains from its base snapshot. Runtime tokenizer
    /// parsers must consume the returned receipt through [`ValidatedTokenizerSource::read_unchanged`].
    pub fn tokenizer_for_base(&self, base_root: &Path) -> Result<ValidatedTokenizerSource> {
        self.validate_definition()?;
        match self.tokenizer.binding {
            EncoderTokenizerBinding::RetainBase => {}
        }
        let base_candidates =
            tokenizer_candidate_paths(base_root, self.tokenizer.artifact_candidates)?;
        let base_path = resolve_tokenizer_artifact(&base_candidates, "base snapshot")?;
        let base = pin_tokenizer_artifact(&base_path, self.tokenizer.required_tokens)?;
        let tokenizer = ValidatedTokenizerSource {
            base,
            base_candidates,
            selected: None,
            selected_candidates: None,
            disposition: EncoderTokenizerDisposition::InheritedBase,
        };
        tokenizer.ensure_unchanged()?;
        Ok(tokenizer)
    }

    fn bind_tokenizer(
        &self,
        base_root: &Path,
        selected_source: &WeightsSource,
    ) -> Result<ValidatedTokenizerSource> {
        let mut tokenizer = self.tokenizer_for_base(base_root)?;
        if let Some(selected_root) = selected_complete_snapshot_root(selected_source) {
            let selected_candidates =
                tokenizer_candidate_paths(selected_root, self.tokenizer.artifact_candidates)?;
            let selected_path =
                resolve_tokenizer_artifact(&selected_candidates, "selected complete snapshot")?;
            let selected = pin_tokenizer_artifact(&selected_path, self.tokenizer.required_tokens)?;
            if selected.semantic_sha256 != tokenizer.base.semantic_sha256 {
                return Err(Error::Unsupported(format!(
                    "selected complete snapshot tokenizer is incompatible with retained base tokenizer for family {}: {} has semantic sha256 {}, base {} has {}; pass the selected text_encoder component directly only when it is intentionally a weights-only fine-tune that inherits the base tokenizer",
                    self.tokenizer.family,
                    selected_path.display(),
                    digest_hex(selected.semantic_sha256),
                    tokenizer.path().display(),
                    digest_hex(tokenizer.base.semantic_sha256),
                )));
            }
            tokenizer.selected = Some(selected);
            tokenizer.selected_candidates = Some(selected_candidates);
            tokenizer.disposition = EncoderTokenizerDisposition::MatchedSelectedTokenizer;
        }
        tokenizer.ensure_unchanged()?;
        Ok(tokenizer)
    }

    /// Resolve and validate the effective text-encoder component for one generator load.  The
    /// bundled component remains the zero-change source default, but it is subject to the same
    /// config-and-header contract as an explicit override.
    pub fn source_for_load(
        &self,
        spec: &crate::LoadSpec,
        base_root: &Path,
    ) -> Result<ValidatedEncoderSource> {
        spec.validate_prepared_file_pins()?;
        let builtin = WeightsSource::Dir(base_root.join("text_encoder"));
        let source = spec.text_encoder.as_ref().unwrap_or(&builtin);
        let mut validated = self.validate_source_with_policy(
            source,
            EncoderDtypePolicy::Native,
            EncoderConfigPolicy::Required,
            None,
        )?;
        validated.tokenizer = Some(self.bind_tokenizer(base_root, source)?);
        validated.ensure_unchanged()?;
        spec.validate_prepared_file_pins()?;
        Ok(validated)
    }

    /// Validate one catalog/discovery candidate without acquiring a reusable source receipt.
    ///
    /// This path is intentionally limited to direct-file inventory, bounded behavior config, and
    /// safetensors headers. It never acquires an [`crate::ArtifactSeal`] or reads tensor payloads.
    /// Every lexical loader entry and its current canonical target must remain under one of
    /// `allowed_roots`; callers cannot use discovery as an authorization bypass. The result is only
    /// compatibility information for the current call. Executable load/worker preparation must use
    /// [`Self::validate_source_against_base`] (or [`Self::source_for_load`]) to acquire the complete
    /// retained seal.
    pub fn validate_source_for_discovery(
        &self,
        source: &WeightsSource,
        allowed_roots: &[PathBuf],
    ) -> Result<EncoderDiscoveryFacts> {
        self.validate_source_for_discovery_with_policy(
            source,
            allowed_roots,
            EncoderDtypePolicy::Native,
            EncoderConfigPolicy::Required,
        )
    }

    /// Bounded discovery counterpart to [`Self::validate_comfyui_source`]. Only a direct File is
    /// accepted, and the same provider-owned-config and FP8 normalization policy is applied without
    /// acquiring an executable artifact receipt.
    pub fn validate_comfyui_source_for_discovery(
        &self,
        source: &WeightsSource,
        allowed_roots: &[PathBuf],
    ) -> Result<EncoderDiscoveryFacts> {
        if !matches!(source, WeightsSource::File(_)) {
            return Err(Error::Unsupported(
                "ComfyUI text encoder validation requires one File source".into(),
            ));
        }
        self.validate_source_for_discovery_with_policy(
            source,
            allowed_roots,
            EncoderDtypePolicy::ComfyUiFp8,
            EncoderConfigPolicy::ProviderOwnedComfyUi,
        )
    }

    fn validate_source_for_discovery_with_policy(
        &self,
        source: &WeightsSource,
        allowed_roots: &[PathBuf],
        dtype_policy: EncoderDtypePolicy,
        config_policy: EncoderConfigPolicy,
    ) -> Result<EncoderDiscoveryFacts> {
        self.validate_definition()?;
        let inspected = inspect_encoder_source_for_discovery(
            source,
            allowed_roots,
            config_policy == EncoderConfigPolicy::Required,
            true,
        )?;
        let packed_quant = self.validate_inspected_source(
            source,
            &inspected.weights,
            inspected
                .config_path
                .as_deref()
                .zip(inspected.config.as_ref()),
            &inspected.headers,
            dtype_policy,
            config_policy,
        )?;
        let packing = packed_quant
            .map(|_| {
                self.packing.ok_or_else(|| {
                    Error::Unsupported(format!(
                        "validated packed encoder has no packing contract for architecture {}",
                        self.architecture
                    ))
                })
            })
            .transpose()?;
        let expected = self.materialized_language_tensor_names(packing)?;
        let materialized_language_tensor_headers = inspected
            .headers
            .into_iter()
            .filter(|header| expected.contains(&header.name))
            .collect();
        Ok(EncoderDiscoveryFacts {
            materialized_language_tensor_headers,
            source_bytes: inspected.source_bytes,
        })
    }

    /// Resolve and validate the effective text-encoder component for metadata-only planning.
    /// Unlike [`Self::source_for_load`], this deliberately does not require or retain a tokenizer:
    /// memory admission prices tensor materialization, while the executable load remains the seam
    /// that proves tokenizer compatibility. Complete selected snapshots are accepted here so their
    /// tensor surface can be priced before that later load-time binding.
    pub fn source_for_planning(
        &self,
        spec: &crate::LoadSpec,
        base_root: &Path,
    ) -> Result<ValidatedEncoderSource> {
        let builtin = WeightsSource::Dir(base_root.join("text_encoder"));
        let source = spec.text_encoder.as_ref().unwrap_or(&builtin);
        self.validate_source_for_planning(source)
    }

    /// Validate a provider-selected encoder's behavior and tensor surface for metadata-only
    /// planning, without claiming that its tokenizer has passed the production load-time binding.
    pub fn validate_source_for_planning(
        &self,
        source: &WeightsSource,
    ) -> Result<ValidatedEncoderSource> {
        self.validate_source_with_policy(
            source,
            EncoderDtypePolicy::Native,
            EncoderConfigPolicy::Required,
            None,
        )
    }

    /// Validate a provider-selected source while binding it to the tokenizer retained from
    /// `base_root`. This is the production counterpart to metadata-only [`Self::validate_source`].
    pub fn validate_source_against_base(
        &self,
        source: &WeightsSource,
        base_root: &Path,
    ) -> Result<ValidatedEncoderSource> {
        let mut validated = self.validate_source_with_policy(
            source,
            EncoderDtypePolicy::Native,
            EncoderConfigPolicy::Required,
            None,
        )?;
        validated.tokenizer = Some(self.bind_tokenizer(base_root, source)?);
        validated.ensure_unchanged()?;
        Ok(validated)
    }

    /// Validate one substituted encoder without reading tensor payloads.
    ///
    /// A `Dir` may point directly at the encoder component or at a complete diffusers snapshot.  A
    /// `File` requires a sibling `config.json`: tensor geometry cannot prove behavior-bearing config
    /// or tokenizer identity. A caller cannot lend separately authored bytes the built-in config;
    /// each ordinary selected component must carry its own authoritative behavior evidence.
    pub fn validate_source(&self, source: &WeightsSource) -> Result<ValidatedEncoderSource> {
        if selected_complete_snapshot_root(source).is_some() {
            return Err(Error::Unsupported(
                "a complete text-encoder snapshot requires retained-tokenizer binding; use validate_source_against_base with the provider's base snapshot root"
                    .into(),
            ));
        }
        self.validate_source_with_policy(
            source,
            EncoderDtypePolicy::Native,
            EncoderConfigPolicy::Required,
            None,
        )
    }

    /// Validate a single-file ComfyUI encoder that the consuming route explicitly normalizes from
    /// plain/scaled FP8 before construction. Ordinary native routes remain stricter and cannot use
    /// this method to admit an FP8 source their backend loader does not normalize.
    pub fn validate_comfyui_source(
        &self,
        source: &WeightsSource,
    ) -> Result<ValidatedEncoderSource> {
        if !matches!(source, WeightsSource::File(_)) {
            return Err(Error::Unsupported(
                "ComfyUI text encoder validation requires one pinned File source".into(),
            ));
        }
        self.validate_source_with_policy(
            source,
            EncoderDtypePolicy::ComfyUiFp8,
            EncoderConfigPolicy::ProviderOwnedComfyUi,
            None,
        )
    }

    pub fn validate_comfyui_source_against_base(
        &self,
        source: &WeightsSource,
        base_root: &Path,
    ) -> Result<ValidatedEncoderSource> {
        if !matches!(source, WeightsSource::File(_)) {
            return Err(Error::Unsupported(
                "ComfyUI text encoder validation requires one pinned File source".into(),
            ));
        }
        let mut validated = self.validate_source_with_policy(
            source,
            EncoderDtypePolicy::ComfyUiFp8,
            EncoderConfigPolicy::ProviderOwnedComfyUi,
            None,
        )?;
        validated.tokenizer = Some(self.bind_tokenizer(base_root, source)?);
        validated.ensure_unchanged()?;
        Ok(validated)
    }

    fn validate_source_with_policy(
        &self,
        source: &WeightsSource,
        dtype_policy: EncoderDtypePolicy,
        config_policy: EncoderConfigPolicy,
        tokenizer: Option<ValidatedTokenizerSource>,
    ) -> Result<ValidatedEncoderSource> {
        self.validate_definition()?;
        let (weights, own_config) = resolve_source(source)?;
        let own_config = own_config.filter(|path| path.is_file());
        let config_path = match &weights {
            WeightsSource::File(_) => own_config,
            WeightsSource::Dir(_) => own_config,
        };
        let config_candidate = config_path.clone().or_else(|| match &weights {
            WeightsSource::File(path) => path.parent().map(|parent| parent.join("config.json")),
            WeightsSource::Dir(path) => Some(path.join("config.json")),
        });
        let pinned = PinnedEncoderSource::pin(&weights, config_candidate)?;
        let config = if let Some(config_path) = &config_path {
            let config_pin = pinned.config_pin.as_ref().ok_or_else(|| {
                Error::Unsupported(format!(
                    "text encoder config disappeared during validation: {}",
                    config_path.display()
                ))
            })?;
            let config = config_pin.read_unchanged(read_selected_encoder_config)?;
            Some(config)
        } else {
            None
        };
        let headers = pinned.headers().map_err(|error| {
            Error::Msg(format!(
                "text encoder contract: inspect {}: {error}",
                source_path(&weights).display()
            ))
        })?;
        let packed_quant = self.validate_inspected_source(
            source,
            &weights,
            config_path.as_deref().zip(config.as_ref()),
            &headers,
            dtype_policy,
            config_policy,
        )?;
        pinned.ensure_unchanged()?;
        Ok(ValidatedEncoderSource {
            requested_source: source.clone(),
            weights,
            config_path,
            packed_quant_bits: packed_quant.map(|quant| quant.bits as i32),
            pinned,
            tokenizer,
        })
    }

    fn validate_inspected_source(
        &self,
        requested_source: &WeightsSource,
        weights: &WeightsSource,
        config: Option<(&Path, &Value)>,
        headers: &[SafetensorsTensorHeader],
        dtype_policy: EncoderDtypePolicy,
        config_policy: EncoderConfigPolicy,
    ) -> Result<Option<PackedQuantization>> {
        let declared_quant = if let Some((config_path, config)) = config {
            self.validate_config(config, config_path)?;
            parse_packed_quantization(config, config_path)?
        } else if config_policy == EncoderConfigPolicy::Required {
            return Err(Error::Msg(format!(
                "text encoder substitution has no config.json (source {}); exact behavior, tokenizer, head-topology, and precision compatibility cannot be proven from tensor shapes alone",
                source_path(requested_source).display()
            )));
        } else {
            None
        };

        let language_headers = language_quantization_evidence_headers(headers);
        let packed_quant = validate_quantization_evidence(
            &language_headers,
            source_path(weights),
            declared_quant,
        )?;
        self.validate_packing_config(
            source_path(weights),
            packed_quant,
            matches!(weights, WeightsSource::File(_)),
        )?;
        self.validate_headers(headers, source_path(weights), packed_quant, dtype_policy)?;
        Ok(packed_quant)
    }

    /// Validate an encoder embedded inside a fused safetensors checkpoint. `component_prefixes`
    /// are stripped before applying this contract's normal architecture and geometry checks. The
    /// caller must retain its own [`PinnedWeightsFile`] for the fused payload; this method pins the
    /// same file across header inspection so the validation itself cannot race a replacement.
    pub fn validate_embedded_file(&self, file: &Path, component_prefixes: &[&str]) -> Result<()> {
        self.validate_embedded_file_with_policy(
            file,
            component_prefixes,
            EncoderDtypePolicy::Native,
            EncoderConfigPolicy::Required,
        )
    }

    /// Validate a fused ComfyUI checkpoint whose selected route explicitly normalizes FP8 tensors
    /// before constructing the native encoder.
    pub fn validate_embedded_comfyui_file(
        &self,
        file: &Path,
        component_prefixes: &[&str],
    ) -> Result<()> {
        self.validate_embedded_file_with_policy(
            file,
            component_prefixes,
            EncoderDtypePolicy::ComfyUiFp8,
            EncoderConfigPolicy::ProviderOwnedComfyUi,
        )
    }

    /// Validate a fused ComfyUI encoder and return the exact retained tokenizer receipt that must be
    /// used by the eventual runtime parser.
    pub fn validate_embedded_comfyui_file_against_base(
        &self,
        file: &Path,
        component_prefixes: &[&str],
        base_root: &Path,
    ) -> Result<ValidatedTokenizerSource> {
        let tokenizer = self.bind_tokenizer(base_root, &WeightsSource::File(file.to_path_buf()))?;
        self.validate_embedded_comfyui_file(file, component_prefixes)?;
        tokenizer.ensure_unchanged()?;
        Ok(tokenizer)
    }

    fn validate_embedded_file_with_policy(
        &self,
        file: &Path,
        component_prefixes: &[&str],
        dtype_policy: EncoderDtypePolicy,
        config_policy: EncoderConfigPolicy,
    ) -> Result<()> {
        self.validate_definition()?;
        let pin = PinnedWeightsFile::pin(file)?;
        let config_path = file
            .parent()
            .map(|parent| parent.join("config.json"))
            .ok_or_else(|| {
                Error::Unsupported(format!(
                "embedded text encoder {} has no parent directory for an authoritative config.json",
                file.display()
            ))
            })?;
        if config_policy == EncoderConfigPolicy::Required && !config_path.is_file() {
            return Err(Error::Unsupported(format!(
                "embedded text encoder {} requires sibling config.json; tensor headers cannot prove behavior, tokenizer, or head topology",
                file.display()
            )));
        }
        let config_pin = config_path
            .is_file()
            .then(|| PinnedWeightsFile::pin(&config_path))
            .transpose()?;
        if let Some(config_pin) = config_pin.as_ref() {
            let config: Value = config_pin.read_unchanged(|path| {
                let text = std::fs::read_to_string(path).map_err(|error| {
                    Error::Msg(format!(
                        "text encoder contract: read {}: {error}",
                        path.display()
                    ))
                })?;
                serde_json::from_str(&text).map_err(|error| {
                    Error::Msg(format!(
                        "text encoder contract: parse {}: {error}",
                        path.display()
                    ))
                })
            })?;
            self.validate_config(&config, &config_path)?;
            if parse_packed_quantization(&config, &config_path)?.is_some() {
                return Err(Error::Unsupported(format!(
                    "embedded text encoder {} cannot use a packed quantization marker; fused imports support dense or explicitly normalized ComfyUI FP8 tensors only",
                    file.display()
                )));
            }
        }
        let headers = pin.read_unchanged(|path| {
            safetensors_path_tensor_headers(path).map_err(|error| {
                Error::Msg(format!(
                    "text encoder contract: inspect embedded source {}: {error}",
                    path.display()
                ))
            })
        })?;
        let encoder_prefix = self.expected_header_prefix()?;
        let embedding_suffix = format!("{encoder_prefix}.embed_tokens.weight");
        let matching = component_prefixes
            .iter()
            .copied()
            .filter(|prefix| {
                let embedding = format!("{prefix}{embedding_suffix}");
                headers.iter().any(|header| header.name == embedding)
            })
            .collect::<Vec<_>>();
        let component_prefix = match matching.as_slice() {
            [prefix] => *prefix,
            [] => {
                return Err(self.mismatch(
                    file,
                    "embedded_architecture_header",
                    format!("one of {component_prefixes:?} followed by {embedding_suffix}"),
                    "missing",
                ))
            }
            _ => {
                return Err(self.mismatch(
                    file,
                    "embedded_architecture_header",
                    "one unambiguous text encoder component",
                    format!("multiple prefixes {matching:?}"),
                ))
            }
        };
        let embedded = headers
            .into_iter()
            .filter_map(|header| {
                header
                    .name
                    .strip_prefix(component_prefix)
                    .map(|name| SafetensorsTensorHeader {
                        name: name.to_owned(),
                        ..header
                    })
            })
            .collect::<Vec<_>>();
        self.validate_headers(&embedded, file, None, dtype_policy)?;
        if let Some(config_pin) = config_pin {
            config_pin.ensure_unchanged()?;
        } else if config_path.is_file() {
            return Err(Error::Unsupported(format!(
                "embedded text encoder config appeared during validation: {}",
                config_path.display()
            )));
        }
        pin.ensure_unchanged()?;
        Ok(())
    }

    fn validate_config(&self, config: &Value, path: &Path) -> Result<()> {
        self.validate_definition()?;
        let text = match config.get("text_config") {
            Some(text) => text
                .as_object()
                .map(|_| text)
                .ok_or_else(|| self.mismatch(path, "text_config", "object", "non-object"))?,
            None => config,
        };
        self.validate_architecture_config(text, path)?;

        self.expect_json_usize(text, path, "hidden_size", self.hidden_size)?;
        self.expect_json_usize(text, path, "intermediate_size", self.intermediate_size)?;
        self.expect_json_usize(text, path, "num_hidden_layers", self.num_hidden_layers)?;
        self.expect_json_usize(text, path, "num_attention_heads", self.num_attention_heads)?;
        self.expect_json_usize(text, path, "num_key_value_heads", self.num_key_value_heads)?;
        let config_head_dim = match text.get("head_dim") {
            None | Some(Value::Null) => self
                .hidden_size
                .is_multiple_of(self.num_attention_heads)
                .then_some(self.hidden_size / self.num_attention_heads),
            Some(value) => value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| self.mismatch(path, "head_dim", self.head_dim, value))
                .map(Some)?,
        };
        match config_head_dim {
            Some(actual) if actual == self.head_dim => {}
            Some(actual) => {
                return Err(self.mismatch(path, "head_dim", self.head_dim, actual));
            }
            None => return Err(self.mismatch(path, "head_dim", self.head_dim, "missing")),
        }
        self.expect_json_usize(text, path, "vocab_size", self.vocab_size)?;
        self.expect_json_str(text, path, "hidden_act", self.hidden_activation)?;
        self.expect_json_f64(
            text,
            path,
            "attention_dropout",
            self.attention_dropout.get(),
        )?;
        self.expect_json_f64(text, path, "rms_norm_eps", self.rms_norm_eps.get())?;
        let rope_aliases = [
            &["rope_theta"][..],
            &["rope_parameters", "rope_theta"][..],
            &["rope_scaling", "rope_theta"][..],
        ];
        self.expect_json_f64_from_all_aliases(
            text,
            path,
            "rope_theta",
            self.rope_theta.get(),
            &rope_aliases,
        )?;
        if !std::ptr::eq(config, text) {
            self.validate_optional_json_f64_aliases(
                config,
                path,
                "rope_theta",
                self.rope_theta.get(),
                &rope_aliases,
            )?;
        }
        self.expect_json_usize(
            text,
            path,
            "max_position_embeddings",
            self.max_position_embeddings,
        )?;
        self.validate_json_bool_aliases(config, text, path, "attention_bias", self.attention_bias)?;
        self.validate_json_bool_aliases(
            config,
            text,
            path,
            "tie_word_embeddings",
            self.tie_word_embeddings,
        )?;
        for (field, expected) in [
            ("bos_token_id", self.bos_token_id),
            ("eos_token_id", self.eos_token_id),
            ("image_token_id", self.image_token_id),
            ("vision_start_token_id", self.vision_start_token_id),
            ("vision_end_token_id", self.vision_end_token_id),
        ] {
            self.validate_json_i64_aliases(config, text, path, field, expected)?;
        }
        for required in self.tokenizer.required_tokens {
            if let Some(field) = required.config_field {
                self.validate_json_i64_aliases(config, text, path, field, Some(required.id))?;
            }
        }
        self.validate_mrope_section_aliases(text, path, true)?;
        self.validate_mrope_interleaving_aliases(text, path, true)?;
        self.validate_rope_type_aliases(text, path)?;
        if !std::ptr::eq(config, text) {
            self.validate_mrope_section_aliases(config, path, false)?;
            self.validate_mrope_interleaving_aliases(config, path, false)?;
            self.validate_rope_type_aliases(config, path)?;
        }
        let allowed_rope_fields = BTreeSet::from([
            "mrope_interleaved",
            "mrope_section",
            "rope_theta",
            "rope_type",
            "type",
        ]);
        for (scope_name, scope) in [("text_config", text), ("root", config)] {
            if scope_name == "root" && std::ptr::eq(config, text) {
                continue;
            }
            for object_name in ["rope_parameters", "rope_scaling"] {
                let Some(value) = scope.get(object_name) else {
                    continue;
                };
                if value.is_null() {
                    continue;
                }
                let Some(object) = value.as_object() else {
                    return Err(self.mismatch(
                        path,
                        "rope_behavior_fields",
                        format!("{object_name} object or null"),
                        format!("{scope_name}={value}"),
                    ));
                };
                let extra = object
                    .keys()
                    .filter(|field| !allowed_rope_fields.contains(field.as_str()))
                    .cloned()
                    .collect::<Vec<_>>();
                if !extra.is_empty() {
                    return Err(self.mismatch(
                        path,
                        "rope_behavior_fields",
                        format!("only {allowed_rope_fields:?}"),
                        format!("unsupported {scope_name}.{object_name} keys {extra:?}"),
                    ));
                }
            }
        }
        if let Some(layer_types) = text.get("layer_types") {
            let layer_types = layer_types.as_array().ok_or_else(|| {
                self.mismatch(
                    path,
                    "layer_types",
                    "array of full_attention strings",
                    layer_types,
                )
            })?;
            if layer_types.len() < self.loaded_hidden_layers {
                return Err(self.mismatch(
                    path,
                    "layer_types",
                    format!("at least {} entries", self.loaded_hidden_layers),
                    layer_types.len(),
                ));
            }
            for (index, value) in layer_types
                .iter()
                .take(self.loaded_hidden_layers)
                .enumerate()
            {
                if value.as_str() != Some("full_attention") {
                    return Err(self.mismatch(
                        path,
                        "layer_types",
                        "full_attention for every loaded layer",
                        format!("index {index}={value}"),
                    ));
                }
            }
        }
        match text.get("sliding_window") {
            None | Some(Value::Null) => {}
            Some(value) => {
                return Err(self.mismatch(path, "sliding_window", "missing or null", value));
            }
        }
        match text.get("use_sliding_window") {
            None | Some(Value::Null) | Some(Value::Bool(false)) => {}
            Some(value) => {
                return Err(self.mismatch(path, "use_sliding_window", false, value));
            }
        }
        Ok(())
    }

    fn expect_json_str(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: &str,
    ) -> Result<()> {
        match config.get(field).and_then(Value::as_str) {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(self.mismatch(path, field, expected, actual)),
            None => Err(self.mismatch(path, field, expected, "missing")),
        }
    }

    fn validate_architecture_config(&self, text: &Value, path: &Path) -> Result<()> {
        let mut found = false;
        if let Some(value) = text.get("model_type") {
            let actual = value.as_str().ok_or_else(|| {
                self.mismatch(path, "architecture.model_type", self.architecture, value)
            })?;
            found = true;
            if !architecture_matches(actual, self.architecture) {
                return Err(self.mismatch(
                    path,
                    "architecture.model_type",
                    self.architecture,
                    actual,
                ));
            }
        }
        if let Some(value) = text.get("architectures") {
            let values = value.as_array().ok_or_else(|| {
                self.mismatch(
                    path,
                    "architecture.architectures",
                    "array of strings",
                    value,
                )
            })?;
            for (index, value) in values.iter().enumerate() {
                let actual = value.as_str().ok_or_else(|| {
                    self.mismatch(
                        path,
                        "architecture.architectures",
                        "array of strings",
                        format!("index {index}={value}"),
                    )
                })?;
                found = true;
                if !architecture_matches(actual, self.architecture) {
                    return Err(self.mismatch(
                        path,
                        "architecture.architectures",
                        self.architecture,
                        format!("index {index}={actual}"),
                    ));
                }
            }
        }
        if found {
            Ok(())
        } else {
            Err(self.mismatch(path, "architecture", self.architecture, "missing"))
        }
    }

    fn expect_json_f64(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: f64,
    ) -> Result<()> {
        match config.get(field).and_then(Value::as_f64) {
            Some(actual) if actual.to_bits() == expected.to_bits() => Ok(()),
            Some(actual) => Err(self.mismatch(path, field, expected, actual)),
            None => Err(self.mismatch(path, field, expected, "missing")),
        }
    }

    fn expect_json_f64_from_all_aliases(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: f64,
        candidates: &[&[&str]],
    ) -> Result<()> {
        let mut found = false;
        for candidate in candidates {
            let Some(value) = json_path(config, candidate) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let actual = value.as_f64().ok_or_else(|| {
                self.mismatch(
                    path,
                    field,
                    expected,
                    format!("{}={value}", candidate.join(".")),
                )
            })?;
            found = true;
            if actual.to_bits() != expected.to_bits() {
                return Err(self.mismatch(
                    path,
                    field,
                    expected,
                    format!("{}={actual}", candidate.join(".")),
                ));
            }
        }
        if found {
            Ok(())
        } else {
            Err(self.mismatch(path, field, expected, "missing"))
        }
    }

    fn validate_optional_json_f64_aliases(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: f64,
        candidates: &[&[&str]],
    ) -> Result<()> {
        for candidate in candidates {
            let Some(value) = json_path(config, candidate) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let actual = value.as_f64().ok_or_else(|| {
                self.mismatch(
                    path,
                    field,
                    expected,
                    format!("{}={value}", candidate.join(".")),
                )
            })?;
            if actual.to_bits() != expected.to_bits() {
                return Err(self.mismatch(
                    path,
                    field,
                    expected,
                    format!("{}={actual}", candidate.join(".")),
                ));
            }
        }
        Ok(())
    }

    fn root_text_aliases<'a>(
        root: &'a Value,
        text: &'a Value,
        field: &str,
    ) -> Vec<(&'static str, &'a Value)> {
        let mut values = Vec::with_capacity(2);
        if let Some(value) = text.get(field) {
            values.push(("text_config", value));
        }
        if !std::ptr::eq(root, text) {
            if let Some(value) = root.get(field) {
                values.push(("root", value));
            }
        }
        values
    }

    fn validate_json_bool_aliases(
        &self,
        root: &Value,
        text: &Value,
        path: &Path,
        field: &'static str,
        contract: EncoderConfigBool,
    ) -> Result<()> {
        let mut first = None;
        let expected = contract.effective();
        for (location, value) in Self::root_text_aliases(root, text, field) {
            if value.is_null() {
                continue;
            }
            let actual = value.as_bool().ok_or_else(|| {
                self.mismatch(path, field, "boolean", format!("{location}={value}"))
            })?;
            if actual != expected {
                return Err(self.mismatch(path, field, expected, format!("{location}={actual}")));
            }
            if first.is_some_and(|first| first != actual) {
                return Err(self.mismatch(
                    path,
                    field,
                    "one consistent root/text_config value",
                    format!("{location}={actual}"),
                ));
            }
            first = Some(actual);
        }
        if first.is_none() && contract.is_required() {
            Err(self.mismatch(path, field, expected, "missing"))
        } else {
            Ok(())
        }
    }

    fn validate_json_i64_aliases(
        &self,
        root: &Value,
        text: &Value,
        path: &Path,
        field: &'static str,
        expected: Option<i64>,
    ) -> Result<()> {
        let mut first = None;
        for (location, value) in Self::root_text_aliases(root, text, field) {
            if value.is_null() {
                continue;
            }
            let actual = value.as_i64().ok_or_else(|| {
                self.mismatch(path, field, "integer", format!("{location}={value}"))
            })?;
            if let Some(expected) = expected {
                if actual != expected {
                    return Err(self.mismatch(
                        path,
                        field,
                        expected,
                        format!("{location}={actual}"),
                    ));
                }
            }
            if first.is_some_and(|first| first != actual) {
                return Err(self.mismatch(
                    path,
                    field,
                    "one consistent root/text_config value",
                    format!("{location}={actual}"),
                ));
            }
            first = Some(actual);
        }
        match (expected, first) {
            (Some(expected), None) => Err(self.mismatch(path, field, expected, "missing")),
            _ => Ok(()),
        }
    }

    fn validate_mrope_section_aliases(
        &self,
        text: &Value,
        path: &Path,
        required: bool,
    ) -> Result<()> {
        let aliases = [
            &["rope_parameters", "mrope_section"][..],
            &["rope_scaling", "mrope_section"][..],
        ];
        let mut found = false;
        for alias in aliases {
            let Some(value) = json_path(text, alias) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let actual = value
                .as_array()
                .and_then(|values| {
                    values
                        .iter()
                        .map(|value| value.as_u64().and_then(|value| usize::try_from(value).ok()))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or_else(|| {
                    self.mismatch(
                        path,
                        "mrope_section",
                        format!("{:?}", self.mrope_section),
                        format!("{}={value}", alias.join(".")),
                    )
                })?;
            found = true;
            if actual != self.mrope_section {
                return Err(self.mismatch(
                    path,
                    "mrope_section",
                    format!("{:?}", self.mrope_section),
                    format!("{}={actual:?}", alias.join(".")),
                ));
            }
        }
        if required && !self.mrope_section.is_empty() && !found {
            Err(self.mismatch(
                path,
                "mrope_section",
                format!("{:?}", self.mrope_section),
                "missing",
            ))
        } else {
            Ok(())
        }
    }

    fn validate_mrope_interleaving_aliases(
        &self,
        text: &Value,
        path: &Path,
        required: bool,
    ) -> Result<()> {
        let aliases = [
            &["rope_parameters", "mrope_interleaved"][..],
            &["rope_scaling", "mrope_interleaved"][..],
        ];
        let expected = self.mrope_interleaved.unwrap_or(false);
        let mut found = false;
        for alias in aliases {
            let Some(value) = json_path(text, alias) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let actual = value.as_bool().ok_or_else(|| {
                self.mismatch(
                    path,
                    "mrope_interleaved",
                    expected,
                    format!("{}={value}", alias.join(".")),
                )
            })?;
            found = true;
            if actual != expected {
                return Err(self.mismatch(
                    path,
                    "mrope_interleaved",
                    expected,
                    format!("{}={actual}", alias.join(".")),
                ));
            }
        }
        if required && self.mrope_interleaved.is_some() && !found {
            Err(self.mismatch(path, "mrope_interleaved", expected, "missing"))
        } else {
            Ok(())
        }
    }

    fn validate_rope_type_aliases(&self, text: &Value, path: &Path) -> Result<()> {
        let aliases = [
            &["rope_type"][..],
            &["rope_parameters", "rope_type"][..],
            &["rope_parameters", "type"][..],
            &["rope_scaling", "rope_type"][..],
            &["rope_scaling", "type"][..],
        ];
        for alias in aliases {
            let Some(value) = json_path(text, alias) else {
                continue;
            };
            if value.is_null() {
                continue;
            }
            let actual = value.as_str().ok_or_else(|| {
                self.mismatch(
                    path,
                    "rope_type",
                    "default",
                    format!("{}={value}", alias.join(".")),
                )
            })?;
            if actual != "default" {
                return Err(self.mismatch(
                    path,
                    "rope_type",
                    "default",
                    format!("{}={actual}", alias.join(".")),
                ));
            }
        }
        Ok(())
    }

    fn expect_json_usize(
        &self,
        config: &Value,
        path: &Path,
        field: &'static str,
        expected: usize,
    ) -> Result<()> {
        match config
            .get(field)
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
        {
            Some(actual) if actual == expected => Ok(()),
            Some(actual) => Err(self.mismatch(path, field, expected, actual)),
            None => Err(self.mismatch(path, field, expected, "missing")),
        }
    }

    fn validate_headers(
        &self,
        headers: &[SafetensorsTensorHeader],
        path: &Path,
        packed_quant: Option<PackedQuantization>,
        dtype_policy: EncoderDtypePolicy,
    ) -> Result<()> {
        let prefix = self.expected_header_prefix()?;
        self.validate_architecture_signature(headers, path, prefix)?;
        let lm_head_prefix = self
            .requires_lm_head
            .then(|| self.expected_lm_head_prefix())
            .transpose()?
            .map(|prefix| format!("{prefix}.lm_head."));
        let lm_head_name = lm_head_prefix
            .as_ref()
            .map(|prefix| format!("{prefix}weight"));
        let relevant = headers
            .iter()
            .filter(|header| {
                header.name.starts_with(&format!("{prefix}."))
                    || lm_head_prefix
                        .as_ref()
                        .is_some_and(|prefix| header.name.starts_with(prefix))
            })
            .cloned()
            .collect::<Vec<_>>();
        let headers = relevant.as_slice();
        let packing = packed_quant.and(self.packing);
        let mut allowed_packed_bases = BTreeSet::new();
        let embedding_name = format!("{prefix}.embed_tokens.weight");
        let embedding_base = embedding_name
            .strip_suffix(".weight")
            .expect("embedding name has weight suffix");
        let pack_embedding = packing.is_some_and(|packing| packing.pack_embedding);
        if pack_embedding {
            allowed_packed_bases.insert(embedding_base.to_owned());
        }
        self.validate_matrix(
            headers,
            path,
            MatrixExpectation {
                name: &embedding_name,
                field: "vocab_size",
                shape: [self.vocab_size, self.hidden_size],
                must_be_packed: pack_embedding,
            },
            packed_quant,
            dtype_policy,
        )?;
        let layer_marker = format!("{prefix}.layers.");
        let layers: BTreeSet<usize> = headers
            .iter()
            .filter_map(|header| {
                header
                    .name
                    .strip_prefix(&layer_marker)
                    .and_then(|tail| tail.split('.').next())
                    .and_then(|index| index.parse().ok())
            })
            .collect();
        let missing_loaded_layers = (0..self.loaded_hidden_layers)
            .filter(|layer| !layers.contains(layer))
            .collect::<Vec<_>>();
        let out_of_contract_layers = layers
            .iter()
            .copied()
            .filter(|&layer| layer >= self.num_hidden_layers)
            .collect::<Vec<_>>();
        if !missing_loaded_layers.is_empty() || !out_of_contract_layers.is_empty() {
            return Err(self.mismatch(
                path,
                "loaded_hidden_layers",
                format!(
                    "every layer 0..{} and no layer >= {}",
                    self.loaded_hidden_layers, self.num_hidden_layers
                ),
                format!(
                    "missing={missing_loaded_layers:?}, out_of_contract={out_of_contract_layers:?}"
                ),
            ));
        }

        let attention_width = self
            .num_attention_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| {
                self.mismatch(
                    path,
                    "attention_width",
                    "non-overflowing head count × head dimension",
                    format!("{} × {}", self.num_attention_heads, self.head_dim),
                )
            })?;
        let kv_width = self
            .num_key_value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| {
                self.mismatch(
                    path,
                    "key_value_width",
                    "non-overflowing head count × head dimension",
                    format!("{} × {}", self.num_key_value_heads, self.head_dim),
                )
            })?;
        for layer in 0..self.loaded_hidden_layers {
            let base = format!("{prefix}.layers.{layer}");
            for (suffix, field, shape) in [
                (
                    "self_attn.q_proj.weight",
                    "num_attention_heads/head_dim",
                    [attention_width, self.hidden_size],
                ),
                (
                    "self_attn.k_proj.weight",
                    "num_key_value_heads/head_dim",
                    [kv_width, self.hidden_size],
                ),
                (
                    "self_attn.v_proj.weight",
                    "num_key_value_heads/head_dim",
                    [kv_width, self.hidden_size],
                ),
                (
                    "self_attn.o_proj.weight",
                    "attention_output_width",
                    [self.hidden_size, attention_width],
                ),
                (
                    "mlp.gate_proj.weight",
                    "intermediate_size",
                    [self.intermediate_size, self.hidden_size],
                ),
                (
                    "mlp.up_proj.weight",
                    "intermediate_size",
                    [self.intermediate_size, self.hidden_size],
                ),
                (
                    "mlp.down_proj.weight",
                    "intermediate_size",
                    [self.hidden_size, self.intermediate_size],
                ),
            ] {
                let name = format!("{base}.{suffix}");
                if packing.is_some() {
                    allowed_packed_bases.insert(
                        name.strip_suffix(".weight")
                            .expect("projection name has weight suffix")
                            .to_owned(),
                    );
                }
                self.validate_matrix(
                    headers,
                    path,
                    MatrixExpectation {
                        name: &name,
                        field,
                        shape,
                        must_be_packed: packing.is_some(),
                    },
                    packed_quant,
                    dtype_policy,
                )?;
            }
            self.validate_vector(
                headers,
                path,
                &format!("{base}.input_layernorm.weight"),
                "hidden_size",
                self.hidden_size,
                dtype_policy,
            )?;
            self.validate_vector(
                headers,
                path,
                &format!("{base}.post_attention_layernorm.weight"),
                "hidden_size",
                self.hidden_size,
                dtype_policy,
            )?;
            match self.architecture {
                "qwen3" | "qwen3_vl_text" => {
                    self.validate_vector(
                        headers,
                        path,
                        &format!("{base}.self_attn.q_norm.weight"),
                        "head_dim",
                        self.head_dim,
                        dtype_policy,
                    )?;
                    self.validate_vector(
                        headers,
                        path,
                        &format!("{base}.self_attn.k_norm.weight"),
                        "head_dim",
                        self.head_dim,
                        dtype_policy,
                    )?;
                }
                "qwen2_5_vl_text" => {
                    for (projection, width) in [
                        ("q_proj", attention_width),
                        ("k_proj", kv_width),
                        ("v_proj", kv_width),
                    ] {
                        self.validate_vector(
                            headers,
                            path,
                            &format!("{base}.self_attn.{projection}.bias"),
                            "attention_bias_width",
                            width,
                            dtype_policy,
                        )?;
                    }
                }
                "mistral" => {}
                _ => unreachable!("expected_header_prefix rejects unsupported architectures"),
            }
        }
        self.validate_bias_surface(headers, path, prefix)?;
        if self.requires_final_norm {
            self.validate_vector(
                headers,
                path,
                &format!("{prefix}.norm.weight"),
                "hidden_size",
                self.hidden_size,
                dtype_policy,
            )?;
        }
        if self.requires_lm_head {
            let name = lm_head_name.expect("requires_lm_head constructed a validated name");
            let pack_lm_head = packing.is_some_and(|packing| packing.pack_lm_head);
            if pack_lm_head {
                allowed_packed_bases.insert(
                    name.strip_suffix(".weight")
                        .expect("LM-head name has weight suffix")
                        .to_owned(),
                );
            }
            self.validate_matrix(
                headers,
                path,
                MatrixExpectation {
                    name: &name,
                    field: "lm_head_output_width",
                    shape: [self.vocab_size, self.hidden_size],
                    must_be_packed: pack_lm_head,
                },
                packed_quant,
                dtype_policy,
            )?;
        }
        self.validate_all_packed_triples(headers, path, packed_quant, &allowed_packed_bases)?;
        self.validate_dense_storage_probe(headers, path)?;
        Ok(())
    }

    fn validate_dense_storage_probe(
        &self,
        headers: &[SafetensorsTensorHeader],
        path: &Path,
    ) -> Result<()> {
        let Some(probe_name) = self.dense_storage_dtype_probe else {
            return Ok(());
        };
        let probe = headers
            .iter()
            .find(|header| header.name == probe_name)
            .ok_or_else(|| {
                self.mismatch(path, "dense_storage_dtype_probe", probe_name, "missing")
            })?;
        // Candle Krea first opens at BF16 and retains that store only when this native probe is
        // itself BF16. In that branch every consumed dense matrix must already be BF16 or the open
        // would silently narrow it. A F16/F32 probe makes the loader reopen the whole component at
        // F32, where every accepted native float dtype is widened safely and need not be uniform.
        if probe.dtype != Dtype::BF16 {
            return Ok(());
        }
        let prefix = self.expected_header_prefix()?;
        let lm_head_name = self
            .requires_lm_head
            .then(|| self.expected_lm_head_prefix())
            .transpose()?
            .map(|prefix| format!("{prefix}.lm_head.weight"));
        for header in headers.iter().filter(|header| {
            self.is_consumed_dense_store_tensor(&header.name, prefix, lm_head_name.as_deref())
                && matches!(header.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
        }) {
            if header.dtype != Dtype::BF16 {
                return Err(self.mismatch(
                    path,
                    "dense_storage_dtype",
                    format!("BF16 selected by {probe_name}"),
                    format!("{}={:?}", header.name, header.dtype),
                ));
            }
        }
        Ok(())
    }

    fn is_consumed_dense_store_tensor(
        &self,
        name: &str,
        prefix: &str,
        lm_head_name: Option<&str>,
    ) -> bool {
        if name == format!("{prefix}.embed_tokens.weight") || Some(name) == lm_head_name {
            return true;
        }
        let Some(tail) = name.strip_prefix(&format!("{prefix}.layers.")) else {
            return false;
        };
        let Some((layer, suffix)) = tail.split_once('.') else {
            return false;
        };
        let Ok(layer) = layer.parse::<usize>() else {
            return false;
        };
        if layer >= self.loaded_hidden_layers {
            return false;
        }
        matches!(
            suffix,
            "self_attn.q_proj.weight"
                | "self_attn.k_proj.weight"
                | "self_attn.v_proj.weight"
                | "self_attn.o_proj.weight"
                | "mlp.gate_proj.weight"
                | "mlp.up_proj.weight"
                | "mlp.down_proj.weight"
                | "self_attn.q_proj.bias"
                | "self_attn.k_proj.bias"
                | "self_attn.v_proj.bias"
        )
    }

    fn validate_bias_surface(
        &self,
        headers: &[SafetensorsTensorHeader],
        path: &Path,
        prefix: &str,
    ) -> Result<()> {
        let mut allowed = BTreeSet::new();
        if self.architecture == "qwen2_5_vl_text" {
            for layer in 0..self.loaded_hidden_layers {
                for projection in ["q_proj", "k_proj", "v_proj"] {
                    allowed.insert(format!(
                        "{prefix}.layers.{layer}.self_attn.{projection}.bias"
                    ));
                }
            }
        }
        let actual = headers
            .iter()
            .filter(|header| {
                let Some(layer) = architecture_layer_index(&header.name, prefix) else {
                    return false;
                };
                layer < self.loaded_hidden_layers && header.name.ends_with(".bias")
            })
            .map(|header| header.name.clone())
            .collect::<BTreeSet<_>>();
        if actual != allowed {
            return Err(self.mismatch(
                path,
                "projection_bias_surface",
                format!("{allowed:?}"),
                format!("{actual:?}"),
            ));
        }
        Ok(())
    }

    fn validate_packing_config(
        &self,
        path: &Path,
        packed: Option<PackedQuantization>,
        is_file: bool,
    ) -> Result<()> {
        match (packed, self.packing) {
            (None, _) => Ok(()),
            (Some(_), Some(expected)) if is_file && !expected.supports_file => Err(self.mismatch(
                path,
                "packed_file_support",
                "directory-backed packed encoder",
                "single File source",
            )),
            (Some(actual), Some(expected)) if actual.group_size == expected.group_size => Ok(()),
            (Some(actual), Some(expected)) => Err(self.mismatch(
                path,
                "quantization.group_size",
                expected.group_size,
                actual.group_size,
            )),
            (Some(actual), None) => Err(self.mismatch(
                path,
                "quantization",
                "dense-only encoder",
                format!("Q{} group_size {}", actual.bits, actual.group_size),
            )),
        }
    }

    fn validate_vector(
        &self,
        headers: &[SafetensorsTensorHeader],
        path: &Path,
        name: &str,
        field: &'static str,
        expected: usize,
        dtype_policy: EncoderDtypePolicy,
    ) -> Result<()> {
        let Some(tensor) = headers.iter().find(|header| header.name == name) else {
            return Err(self.mismatch(
                path,
                field,
                format!("[{expected}]"),
                format!("missing {name}"),
            ));
        };
        if tensor.shape != [expected] {
            return Err(self.mismatch(
                path,
                field,
                format!("[{expected}]"),
                format!("{:?}", tensor.shape),
            ));
        }
        if !dtype_policy.accepts_dense(tensor.dtype) {
            return Err(self.mismatch(
                path,
                "tensor_dtype",
                dtype_policy.expected_dense(),
                format!("{}={:?}", tensor.name, tensor.dtype),
            ));
        }
        Ok(())
    }

    fn validate_all_packed_triples(
        &self,
        headers: &[SafetensorsTensorHeader],
        path: &Path,
        packed_quant: Option<PackedQuantization>,
        allowed_packed_bases: &BTreeSet<String>,
    ) -> Result<()> {
        let find = |name: &str| headers.iter().find(|header| header.name == name);
        let prefix = self.expected_header_prefix()?;
        let is_unloaded_architecture_layer = |name: &str| {
            architecture_layer_index(name, prefix).is_some_and(|layer| {
                layer >= self.loaded_hidden_layers && layer < self.num_hidden_layers
            })
        };
        for weight in headers.iter().filter(|header| {
            header.name.ends_with(".weight")
                && (header.dtype == Dtype::U32
                    || find(&format!(
                        "{}.scales",
                        header.name.strip_suffix(".weight").unwrap_or(&header.name)
                    ))
                    .is_some()
                    || find(&format!(
                        "{}.biases",
                        header.name.strip_suffix(".weight").unwrap_or(&header.name)
                    ))
                    .is_some())
        }) {
            let base = weight.name.strip_suffix(".weight").unwrap_or(&weight.name);
            if is_unloaded_architecture_layer(base) {
                continue;
            }
            if !allowed_packed_bases.contains(base) {
                return Err(self.mismatch(
                    path,
                    "packed_surface",
                    format!("exactly {allowed_packed_bases:?}"),
                    format!("unexpected packed tensor {base}"),
                ));
            }
            let scales = find(&format!("{base}.scales"));
            let biases = find(&format!("{base}.biases"));
            let (Some(scales), Some(biases)) = (scales, biases) else {
                return Err(self.mismatch(
                    path,
                    "packed_components",
                    format!("{base}.weight + .scales + .biases"),
                    "incomplete packed triple",
                ));
            };
            let quant = packed_quant.ok_or_else(|| {
                self.mismatch(
                    path,
                    "quantization",
                    "sibling config.json with bits/group_size",
                    format!("packed tensor {base} without authoritative metadata"),
                )
            })?;
            if weight.dtype != Dtype::U32
                || !matches!(scales.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
                || !matches!(biases.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
                || weight.shape.len() != 2
                || scales.shape.len() != 2
                || biases.shape != scales.shape
                || weight.shape[0] != scales.shape[0]
            {
                return Err(self.mismatch(
                    path,
                    "packed_triple",
                    "U32 rank-2 weight plus matching float rank-2 scales/biases",
                    format!(
                        "weight={:?}/{:?}, scales={:?}/{:?}, biases={:?}/{:?}",
                        weight.dtype,
                        weight.shape,
                        scales.dtype,
                        scales.shape,
                        biases.dtype,
                        biases.shape
                    ),
                ));
            }
            let logical_input = weight.shape[1]
                .checked_mul(32)
                .and_then(|value| value.checked_div(quant.bits))
                .ok_or_else(|| {
                    self.mismatch(path, "packed_geometry", "non-overflowing matrix", base)
                })?;
            if !logical_input.is_multiple_of(quant.group_size)
                || scales.shape[1] != logical_input / quant.group_size
            {
                return Err(self.mismatch(
                    path,
                    "packed_affine_shape",
                    format!(
                        "[{}, {}] from bits={} group_size={}",
                        weight.shape[0],
                        logical_input / quant.group_size,
                        quant.bits,
                        quant.group_size
                    ),
                    format!("scales={:?}, biases={:?}", scales.shape, biases.shape),
                ));
            }
        }
        for affine in headers
            .iter()
            .filter(|header| header.name.ends_with(".scales") || header.name.ends_with(".biases"))
        {
            let base = affine
                .name
                .strip_suffix(".scales")
                .or_else(|| affine.name.strip_suffix(".biases"))
                .expect("suffix filter guarantees one match");
            if is_unloaded_architecture_layer(base) {
                continue;
            }
            if find(&format!("{base}.weight")).is_none()
                || find(&format!("{base}.scales")).is_none()
                || find(&format!("{base}.biases")).is_none()
            {
                return Err(self.mismatch(
                    path,
                    "packed_components",
                    format!("{base}.weight + .scales + .biases"),
                    "orphaned packed component",
                ));
            }
        }
        Ok(())
    }

    fn expected_header_prefix(&self) -> Result<&'static str> {
        match self.architecture {
            "qwen3" | "qwen2_5_vl_text" => Ok("model"),
            "qwen3_vl_text" => Ok("language_model"),
            "mistral" => Ok("language_model.model"),
            architecture => Err(Error::Unsupported(format!(
                "text encoder contract has no header signature for architecture {architecture}"
            ))),
        }
    }

    fn expected_lm_head_prefix(&self) -> Result<&'static str> {
        match self.architecture {
            "mistral" => Ok("language_model"),
            architecture => Err(Error::Unsupported(format!(
                "text encoder contract requests an LM head but has no LM-head signature for architecture {architecture}"
            ))),
        }
    }

    /// Exact tensor names retained by the concrete language constructors for this contract.
    ///
    /// Packed matrices are three independent safetensors payloads. Keep the affine tables in the
    /// selected surface alongside their code tensor, while ignoring arbitrary keys that merely share
    /// a loaded layer prefix.
    fn materialized_language_tensor_names(
        &self,
        packing: Option<EncoderPackingContract>,
    ) -> Result<BTreeSet<String>> {
        fn insert_matrix(names: &mut BTreeSet<String>, base: String, packed: bool) {
            names.insert(format!("{base}.weight"));
            if packed {
                names.insert(format!("{base}.scales"));
                names.insert(format!("{base}.biases"));
            }
        }

        let prefix = self.expected_header_prefix()?;
        let mut names = BTreeSet::new();
        insert_matrix(
            &mut names,
            format!("{prefix}.embed_tokens"),
            packing.is_some_and(|packing| packing.pack_embedding),
        );
        for layer in 0..self.loaded_hidden_layers {
            let base = format!("{prefix}.layers.{layer}");
            for suffix in [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ] {
                insert_matrix(&mut names, format!("{base}.{suffix}"), packing.is_some());
            }
            names.extend([
                format!("{base}.input_layernorm.weight"),
                format!("{base}.post_attention_layernorm.weight"),
            ]);
            match self.architecture {
                "qwen3" | "qwen3_vl_text" => names.extend([
                    format!("{base}.self_attn.q_norm.weight"),
                    format!("{base}.self_attn.k_norm.weight"),
                ]),
                "qwen2_5_vl_text" => names.extend([
                    format!("{base}.self_attn.q_proj.bias"),
                    format!("{base}.self_attn.k_proj.bias"),
                    format!("{base}.self_attn.v_proj.bias"),
                ]),
                "mistral" => {}
                _ => unreachable!("expected_header_prefix rejects unsupported architectures"),
            }
        }
        if self.requires_final_norm {
            names.insert(format!("{prefix}.norm.weight"));
        }
        if self.requires_lm_head {
            let prefix = self.expected_lm_head_prefix()?;
            insert_matrix(
                &mut names,
                format!("{prefix}.lm_head"),
                packing.is_some_and(|packing| packing.pack_lm_head),
            );
        }
        Ok(names)
    }

    fn validate_architecture_signature(
        &self,
        headers: &[SafetensorsTensorHeader],
        path: &Path,
        prefix: &str,
    ) -> Result<()> {
        let has = |suffix: &str| {
            let name = format!("{prefix}.layers.0.self_attn.{suffix}");
            headers.iter().any(|header| header.name == name)
        };
        let qk_norm = has("q_norm.weight") && has("k_norm.weight");
        let qk_bias = has("q_proj.bias") && has("k_proj.bias");
        let matches = match self.architecture {
            "qwen3" | "qwen3_vl_text" => qk_norm && !qk_bias,
            "qwen2_5_vl_text" => qk_bias && !qk_norm,
            "mistral" => !qk_norm && !qk_bias,
            _ => false,
        };
        if matches {
            Ok(())
        } else {
            Err(self.mismatch(
                path,
                "architecture_header",
                self.architecture,
                format!("prefix={prefix}, qk_norm={qk_norm}, qk_bias={qk_bias}"),
            ))
        }
    }

    fn validate_matrix(
        &self,
        headers: &[SafetensorsTensorHeader],
        path: &Path,
        expectation: MatrixExpectation<'_>,
        packed_quant: Option<PackedQuantization>,
        dtype_policy: EncoderDtypePolicy,
    ) -> Result<()> {
        let MatrixExpectation {
            name,
            field,
            shape: [expected_output, expected_input],
            must_be_packed,
        } = expectation;
        let Some(weight) = headers.iter().find(|header| header.name == name) else {
            return Err(self.mismatch(path, field, expected_output, format!("missing {name}")));
        };
        let base = name.strip_suffix(".weight").unwrap_or(name);
        let scales = headers
            .iter()
            .find(|header| header.name == format!("{base}.scales"));
        let biases = headers
            .iter()
            .find(|header| header.name == format!("{base}.biases"));
        match (scales, biases) {
            (None, None) => {
                if must_be_packed {
                    return Err(self.mismatch(
                        path,
                        "packed_surface",
                        format!("packed triple for {base}"),
                        "dense weight",
                    ));
                }
                if !dtype_policy.accepts_dense(weight.dtype) {
                    return Err(self.mismatch(
                        path,
                        "tensor_dtype",
                        dtype_policy.expected_dense(),
                        format!("{}={:?}", weight.name, weight.dtype),
                    ));
                }
                if weight.shape.len() != 2 {
                    return Err(self.mismatch(
                        path,
                        field,
                        format!("[{expected_output}, {expected_input}] dense matrix"),
                        format!("{:?}", weight.shape),
                    ));
                }
                if weight.shape[0] != expected_output {
                    return Err(self.mismatch(path, field, expected_output, weight.shape[0]));
                }
                if weight.shape[1] != expected_input {
                    return Err(self.mismatch(
                        path,
                        "hidden_size",
                        expected_input,
                        weight.shape[1],
                    ));
                }
            }
            (Some(scales), Some(biases)) => {
                if !must_be_packed {
                    return Err(self.mismatch(
                        path,
                        "packed_surface",
                        format!("dense weight for {base}"),
                        "packed triple",
                    ));
                }
                let quant = packed_quant.ok_or_else(|| {
                    self.mismatch(
                        path,
                        "quantization",
                        "sibling config.json with bits/group_size",
                        format!("packed tensor {base} without authoritative metadata"),
                    )
                })?;
                if weight.dtype != Dtype::U32 {
                    return Err(self.mismatch(
                        path,
                        "packed_dtype",
                        "U32",
                        format!("{:?}", weight.dtype),
                    ));
                }
                if !matches!(scales.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
                    || !matches!(biases.dtype, Dtype::F16 | Dtype::BF16 | Dtype::F32)
                {
                    return Err(self.mismatch(
                        path,
                        "packed_scale_dtype",
                        "F16, BF16, or F32",
                        format!("scales={:?}, biases={:?}", scales.dtype, biases.dtype),
                    ));
                }
                let packed_input_bits =
                    expected_input.checked_mul(quant.bits).ok_or_else(|| {
                        self.mismatch(
                            path,
                            "packed_geometry",
                            "non-overflowing matrix",
                            format!("input={expected_input}, bits={}", quant.bits),
                        )
                    })?;
                if !expected_input.is_multiple_of(quant.group_size)
                    || !packed_input_bits.is_multiple_of(32)
                {
                    return Err(self.mismatch(
                        path,
                        "packed_geometry",
                        "group-aligned Q4/Q8 matrix",
                        format!(
                            "input={expected_input}, bits={}, group_size={}",
                            quant.bits, quant.group_size
                        ),
                    ));
                }
                let expected_weight = vec![expected_output, packed_input_bits / 32];
                let expected_affine = vec![expected_output, expected_input / quant.group_size];
                if weight.shape != expected_weight {
                    return Err(self.mismatch(
                        path,
                        "packed_weight_shape",
                        format!("{expected_weight:?}"),
                        format!("{:?}", weight.shape),
                    ));
                }
                if scales.shape != expected_affine || biases.shape != expected_affine {
                    return Err(self.mismatch(
                        path,
                        "packed_affine_shape",
                        format!("{expected_affine:?}"),
                        format!("scales={:?}, biases={:?}", scales.shape, biases.shape),
                    ));
                }
            }
            _ => {
                return Err(self.mismatch(
                    path,
                    "packed_components",
                    format!("{base}.weight + .scales + .biases"),
                    "incomplete packed triple",
                ));
            }
        }
        Ok(())
    }

    fn mismatch(
        &self,
        path: &Path,
        field: &'static str,
        expected: impl std::fmt::Display,
        actual: impl std::fmt::Display,
    ) -> Error {
        Error::Unsupported(format!(
            "text encoder contract mismatch at {}: field {field} expected {expected}, got {actual}",
            path.display()
        ))
    }
}

fn parse_packed_quantization(config: &Value, path: &Path) -> Result<Option<PackedQuantization>> {
    let Some(marker) = config.get("quantization") else {
        return Ok(None);
    };
    let marker = marker.as_object().ok_or_else(|| {
        Error::Unsupported(format!(
            "text encoder quantization: {} `quantization` must be an object",
            path.display()
        ))
    })?;
    let bits = marker
        .get("bits")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            Error::Unsupported(format!(
                "text encoder quantization: {} `quantization.bits` must be an integer",
                path.display()
            ))
        })?;
    if !matches!(bits, 4 | 8) {
        return Err(Error::Unsupported(format!(
            "text encoder quantization: {} declares unsupported `quantization.bits` {bits}; expected 4 or 8",
            path.display()
        )));
    }
    let group_size = marker
        .get("group_size")
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    Error::Unsupported(format!(
                        "text encoder quantization: {} `quantization.group_size` must be an integer when present",
                        path.display()
                    ))
                })
        })
        .transpose()?
        .unwrap_or(64);
    if group_size != 64 {
        return Err(Error::Unsupported(format!(
            "text encoder quantization: {} declares unsupported `quantization.group_size` {group_size}; every shipping selected-encoder loader requires group_size 64",
            path.display()
        )));
    }
    Ok(Some(PackedQuantization { bits, group_size }))
}

fn source_path(source: &WeightsSource) -> &Path {
    match source {
        WeightsSource::Dir(path) | WeightsSource::File(path) => path,
    }
}

fn resolved_source_layout(
    requested: &WeightsSource,
    resolved: &WeightsSource,
) -> Result<TextEncoderSourceLayout> {
    match (requested, resolved) {
        (WeightsSource::File(requested), WeightsSource::File(resolved))
            if requested == resolved =>
        {
            Ok(TextEncoderSourceLayout::File)
        }
        (WeightsSource::Dir(requested), WeightsSource::Dir(resolved)) if requested == resolved => {
            Ok(TextEncoderSourceLayout::DirectDirectory)
        }
        (WeightsSource::Dir(requested), WeightsSource::Dir(resolved))
            if requested.join("text_encoder") == *resolved =>
        {
            Ok(TextEncoderSourceLayout::CompleteSnapshot)
        }
        _ => Err(Error::Unsupported(
            "text encoder resolved source shape is inconsistent with its requested source".into(),
        )),
    }
}

#[derive(Debug)]
struct DiscoveryInspection {
    weights: WeightsSource,
    config_path: Option<PathBuf>,
    config: Option<Value>,
    headers: Vec<SafetensorsTensorHeader>,
    source_bytes: u64,
}

fn inspect_encoder_source_for_discovery(
    source: &WeightsSource,
    allowed_roots: &[PathBuf],
    require_config: bool,
    inspect_content: bool,
) -> Result<DiscoveryInspection> {
    // Confinement of the caller's original entry is deliberately first. In particular, do not ask
    // whether nested configs exist or enumerate a directory until the entry itself is authorized.
    ensure_discovery_paths_confined(&[source_path(source).to_path_buf()], allowed_roots)?;
    let (weights, config_path) = resolve_source_for_discovery(source, allowed_roots)?;
    let mut preflight = vec![source_path(&weights).to_path_buf()];
    preflight.extend(config_path.iter().cloned());
    ensure_discovery_paths_confined(&preflight, allowed_roots)?;

    if require_config && config_path.is_none() {
        return Err(missing_encoder_config_error(source));
    }

    let shard_paths = encoder_shard_paths_for_discovery(&weights, allowed_roots)?;
    ensure_discovery_paths_confined(&shard_paths, allowed_roots)?;
    let source_bytes = discovery_direct_shard_bytes(&shard_paths)?;
    let (config, headers) = if inspect_content {
        let config = config_path
            .as_deref()
            .map(read_selected_encoder_config)
            .transpose()?;
        let headers =
            collect_unique_encoder_headers(shard_paths.iter().map(safetensors_path_tensor_headers))
                .map_err(|error| {
                    Error::Msg(format!(
                        "text encoder contract: inspect {}: {error}",
                        source_path(&weights).display()
                    ))
                })?;
        (config, headers)
    } else {
        (None, Vec::new())
    };

    let (current_weights, current_config) = resolve_source_for_discovery(source, allowed_roots)?;
    if require_config && current_config.is_none() {
        return Err(missing_encoder_config_error(source));
    }
    let current_shards = encoder_shard_paths_for_discovery(&current_weights, allowed_roots)?;
    let current_bytes = discovery_direct_shard_bytes(&current_shards)?;
    if current_weights != weights
        || current_config != config_path
        || current_shards != shard_paths
        || current_bytes != source_bytes
    {
        return Err(Error::Unsupported(
            "text encoder discovery source shape changed during validation".into(),
        ));
    }
    let mut postflight = vec![source_path(&current_weights).to_path_buf()];
    postflight.extend(current_config.iter().cloned());
    postflight.extend(current_shards);
    ensure_discovery_paths_confined(&postflight, allowed_roots)?;

    Ok(DiscoveryInspection {
        weights,
        config_path,
        config,
        headers,
        source_bytes,
    })
}

fn missing_encoder_config_error(source: &WeightsSource) -> Error {
    Error::Msg(format!(
        "text encoder substitution has no config.json (source {}); exact behavior, tokenizer, head-topology, and precision compatibility cannot be proven from tensor shapes alone",
        source_path(source).display()
    ))
}

fn resolve_source_for_discovery(
    source: &WeightsSource,
    allowed_roots: &[PathBuf],
) -> Result<(WeightsSource, Option<PathBuf>)> {
    match source {
        WeightsSource::File(path) => {
            ensure_discovery_paths_confined(std::slice::from_ref(path), allowed_roots)?;
            let config = match path.parent().map(|parent| parent.join("config.json")) {
                Some(candidate) if discovery_regular_file_candidate(&candidate, allowed_roots)? => {
                    Some(candidate)
                }
                _ => None,
            };
            Ok((WeightsSource::File(path.clone()), config))
        }
        WeightsSource::Dir(path) => {
            ensure_discovery_paths_confined(std::slice::from_ref(path), allowed_roots)?;
            let nested = path.join("text_encoder");
            // Authorize the intermediate component directory itself before appending or inspecting
            // its config path. `symlink_metadata(nested/config.json)` would otherwise traverse an
            // untrusted `text_encoder` directory symlink before its target was root-confined.
            if discovery_directory_candidate(&nested, allowed_roots)? {
                let nested_config = nested.join("config.json");
                if discovery_regular_file_candidate(&nested_config, allowed_roots)? {
                    return Ok((WeightsSource::Dir(nested), Some(nested_config)));
                }
            }
            let direct = path.join("config.json");
            let config =
                discovery_regular_file_candidate(&direct, allowed_roots)?.then_some(direct);
            Ok((WeightsSource::Dir(path.clone()), config))
        }
    }
}

fn discovery_directory_candidate(path: &Path, allowed_roots: &[PathBuf]) -> Result<bool> {
    ensure_discovery_paths_lexically_confined(
        std::slice::from_ref(&path.to_path_buf()),
        allowed_roots,
    )?;
    match std::fs::symlink_metadata(path) {
        Ok(_) => {
            ensure_discovery_paths_confined(
                std::slice::from_ref(&path.to_path_buf()),
                allowed_roots,
            )?;
            Ok(std::fs::metadata(path)?.is_dir())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn discovery_regular_file_candidate(path: &Path, allowed_roots: &[PathBuf]) -> Result<bool> {
    ensure_discovery_paths_lexically_confined(
        std::slice::from_ref(&path.to_path_buf()),
        allowed_roots,
    )?;
    match std::fs::symlink_metadata(path) {
        Ok(_) => {
            ensure_discovery_paths_confined(
                std::slice::from_ref(&path.to_path_buf()),
                allowed_roots,
            )?;
            Ok(std::fs::metadata(path)?.is_file())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn ensure_discovery_paths_lexically_confined(
    paths: &[PathBuf],
    allowed_roots: &[PathBuf],
) -> Result<()> {
    let roots = allowed_roots
        .iter()
        .map(std::path::absolute)
        .collect::<std::io::Result<Vec<_>>>()?;
    if roots.is_empty() {
        return Err(Error::Unsupported(
            "validated text encoder has no authorized model roots".into(),
        ));
    }
    for path in paths {
        let loader_path = std::path::absolute(path)?;
        if !roots.iter().any(|root| loader_path.starts_with(root)) {
            return Err(Error::Unsupported(format!(
                "validated text encoder loader entry escapes authorized model roots: {}",
                loader_path.display()
            )));
        }
    }
    Ok(())
}

fn ensure_discovery_paths_confined(paths: &[PathBuf], allowed_roots: &[PathBuf]) -> Result<()> {
    ensure_discovery_paths_lexically_confined(paths, allowed_roots)?;
    let roots = allowed_roots
        .iter()
        .map(std::path::absolute)
        .collect::<std::io::Result<Vec<_>>>()?;
    for path in paths {
        let canonical_target_path = std::fs::canonicalize(std::path::absolute(path)?)?;
        if !roots
            .iter()
            .any(|root| canonical_target_path.starts_with(root))
        {
            return Err(Error::Unsupported(format!(
                "validated text encoder canonical target escapes authorized model roots: {}",
                canonical_target_path.display()
            )));
        }
    }
    Ok(())
}

fn read_selected_encoder_config(path: &Path) -> Result<Value> {
    let file = std::fs::File::open(path).map_err(|error| {
        Error::Msg(format!(
            "text encoder contract: read {}: {error}",
            path.display()
        ))
    })?;
    let mut bytes = Vec::new();
    file.take(MAX_SELECTED_ENCODER_CONFIG_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| {
            Error::Msg(format!(
                "text encoder contract: read {}: {error}",
                path.display()
            ))
        })?;
    if bytes.len() as u64 > MAX_SELECTED_ENCODER_CONFIG_BYTES {
        return Err(Error::Unsupported(format!(
            "text encoder config {} exceeds the {MAX_SELECTED_ENCODER_CONFIG_BYTES}-byte maximum",
            path.display()
        )));
    }
    serde_json::from_slice(&bytes).map_err(|error| {
        Error::Msg(format!(
            "text encoder contract: parse {}: {error}",
            path.display()
        ))
    })
}

fn json_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    path.iter()
        .try_fold(value, |current, segment| current.get(*segment))
}

fn encoder_shard_paths(source: &WeightsSource) -> Result<Vec<PathBuf>> {
    let mut paths = match source {
        WeightsSource::File(path) => vec![std::path::absolute(path)?],
        WeightsSource::Dir(path) => {
            let mut paths = Vec::new();
            for entry in std::fs::read_dir(path)? {
                let candidate = entry?.path();
                if candidate
                    .extension()
                    .and_then(|extension| extension.to_str())
                    != Some("safetensors")
                    || crate::weightsmeta::is_hidden_file(&candidate)
                {
                    continue;
                }
                let metadata = std::fs::metadata(&candidate).map_err(|error| {
                    Error::Msg(format!(
                        "text encoder direct shard candidate {} cannot be inspected: {error}",
                        candidate.display()
                    ))
                })?;
                if !metadata.is_file() {
                    return Err(Error::Unsupported(format!(
                        "text encoder direct shard candidate is not a regular file: {}",
                        candidate.display()
                    )));
                }
                paths.push(candidate);
            }
            paths
                .into_iter()
                .map(std::path::absolute)
                .collect::<std::io::Result<Vec<_>>>()?
        }
    };
    paths.sort();
    if paths.is_empty() {
        return Err(Error::Msg(format!(
            "no direct .safetensors shards in text encoder source {}",
            source_path(source).display()
        )));
    }
    Ok(paths)
}

fn encoder_shard_paths_for_discovery(
    source: &WeightsSource,
    allowed_roots: &[PathBuf],
) -> Result<Vec<PathBuf>> {
    ensure_discovery_paths_confined(&[source_path(source).to_path_buf()], allowed_roots)?;
    let mut paths = match source {
        WeightsSource::File(path) => {
            let absolute = std::path::absolute(path)?;
            ensure_discovery_paths_confined(std::slice::from_ref(&absolute), allowed_roots)?;
            if !std::fs::metadata(&absolute)?.is_file() {
                return Err(Error::Unsupported(format!(
                    "text encoder direct shard candidate is not a regular file: {}",
                    absolute.display()
                )));
            }
            vec![absolute]
        }
        WeightsSource::Dir(path) => {
            let mut paths = Vec::new();
            for entry in std::fs::read_dir(path)? {
                let candidate = entry?.path();
                if candidate
                    .extension()
                    .and_then(|extension| extension.to_str())
                    != Some("safetensors")
                    || crate::weightsmeta::is_hidden_file(&candidate)
                {
                    continue;
                }
                let absolute = std::path::absolute(candidate)?;
                // Do not follow a shard symlink for metadata until both its lexical loader entry and
                // current canonical target have passed the caller's root authorization.
                ensure_discovery_paths_confined(std::slice::from_ref(&absolute), allowed_roots)?;
                let metadata = std::fs::metadata(&absolute).map_err(|error| {
                    Error::Msg(format!(
                        "text encoder direct shard candidate {} cannot be inspected: {error}",
                        absolute.display()
                    ))
                })?;
                if !metadata.is_file() {
                    return Err(Error::Unsupported(format!(
                        "text encoder direct shard candidate is not a regular file: {}",
                        absolute.display()
                    )));
                }
                paths.push(absolute);
            }
            paths
        }
    };
    paths.sort();
    if paths.is_empty() {
        return Err(Error::Msg(format!(
            "no direct .safetensors shards in text encoder source {}",
            source_path(source).display()
        )));
    }
    Ok(paths)
}

fn discovery_direct_shard_bytes(paths: &[PathBuf]) -> Result<u64> {
    paths.iter().try_fold(0u64, |total, path| {
        let bytes = std::fs::metadata(path)
            .map_err(|error| {
                Error::Msg(format!(
                    "text encoder contract: stat direct shard {}: {error}",
                    path.display()
                ))
            })?
            .len();
        total.checked_add(bytes).ok_or_else(|| {
            Error::Unsupported("text encoder direct-shard byte total overflowed u64".into())
        })
    })
}

fn validate_quantization_evidence(
    headers: &[SafetensorsTensorHeader],
    path: &Path,
    declared: Option<PackedQuantization>,
) -> Result<Option<PackedQuantization>> {
    let names = headers
        .iter()
        .map(|header| header.name.as_str())
        .collect::<BTreeSet<_>>();
    let has_packed_evidence = headers.iter().any(|header| {
        if !header.name.ends_with(".weight") {
            return false;
        }
        let base = header.name.strip_suffix(".weight").unwrap_or(&header.name);
        header.dtype == Dtype::U32
            || names.contains(format!("{base}.scales").as_str())
            || names.contains(format!("{base}.biases").as_str())
    });
    match (declared, has_packed_evidence) {
        (Some(quant), true) => Ok(Some(quant)),
        (Some(quant), false) => Err(Error::Unsupported(format!(
            "text encoder quantization mismatch at {}: config declares Q{} group_size {} but the exact direct-shard surface is dense",
            path.display(), quant.bits, quant.group_size
        ))),
        (None, true) => Err(Error::Unsupported(format!(
            "text encoder quantization mismatch at {}: packed tensor evidence requires authoritative config.json bits/group_size",
            path.display()
        ))),
        (None, false) => Ok(None),
    }
}

/// Keep tier evidence confined to tensors every supported language loader consumes: the token
/// embedding and decoder layer zero. A provider contract subsequently checks every matrix in its
/// complete loaded layer window, so this anchor cannot make a mixed dense/packed source valid. It
/// only prevents ignored vision tensors and intentionally unloaded decoder tails from choosing the
/// effective tier before contract validation.
fn language_quantization_evidence_headers(
    headers: &[SafetensorsTensorHeader],
) -> Vec<SafetensorsTensorHeader> {
    const PREFIXES: [&str; 3] = ["language_model.model", "language_model", "model"];
    headers
        .iter()
        .filter(|header| {
            PREFIXES.iter().any(|prefix| {
                header.name.starts_with(&format!("{prefix}.embed_tokens."))
                    || header.name.starts_with(&format!("{prefix}.layers.0."))
            })
        })
        .cloned()
        .collect()
}

fn architecture_layer_index(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(&format!("{prefix}.layers."))?
        .split('.')
        .next()?
        .parse()
        .ok()
}

fn selected_complete_snapshot_root(source: &WeightsSource) -> Option<&Path> {
    match source {
        WeightsSource::Dir(path) if path.join("text_encoder/config.json").is_file() => Some(path),
        WeightsSource::Dir(_) | WeightsSource::File(_) => None,
    }
}

fn tokenizer_candidate_paths(root: &Path, candidates: &[&str]) -> Result<Vec<PathBuf>> {
    candidates
        .iter()
        .map(|candidate| std::path::absolute(root.join(candidate)).map_err(Error::from))
        .collect()
}

fn resolve_tokenizer_artifact(candidates: &[PathBuf], label: &str) -> Result<PathBuf> {
    for path in candidates {
        if path.is_file() {
            return Ok(path.clone());
        }
    }
    Err(Error::Unsupported(format!(
        "{label} has no retained tokenizer artifact for encoder compatibility: expected one of {candidates:?}"
    )))
}

fn ensure_tokenizer_resolution_unchanged(
    candidates: &[PathBuf],
    expected: &Path,
    label: &str,
) -> Result<()> {
    let current = resolve_tokenizer_artifact(candidates, label)?;
    if current != expected {
        return Err(Error::Unsupported(format!(
            "{label} tokenizer selection changed after validation: expected {}, got {}",
            expected.display(),
            current.display()
        )));
    }
    Ok(())
}

fn pin_tokenizer_artifact(
    path: &Path,
    required_tokens: &[EncoderRequiredToken],
) -> Result<PinnedTokenizerArtifact> {
    let pin = PinnedWeightsFile::pin(path)?;
    let value: Value = pin.read_unchanged(|path| {
        let bytes = std::fs::read(path).map_err(Error::from)?;
        serde_json::from_slice(&bytes).map_err(|error| {
            Error::Msg(format!(
                "text encoder tokenizer contract: parse {}: {error}",
                path.display()
            ))
        })
    })?;
    for required in required_tokens {
        match tokenizer_literal_id(&value, required.literal) {
            Some(actual) if actual == required.id => {}
            Some(actual) => {
                return Err(Error::Unsupported(format!(
                    "text encoder tokenizer contract: {} literal {:?} expected id {}, got {} in {}",
                    required.role,
                    required.literal,
                    required.id,
                    actual,
                    path.display()
                )));
            }
            None => {
                return Err(Error::Unsupported(format!(
                    "text encoder tokenizer contract: required {} literal {:?} is missing from {}",
                    required.role,
                    required.literal,
                    path.display()
                )));
            }
        }
    }
    let mut canonical = Vec::new();
    write_canonical_json(&value, &mut canonical)?;
    let semantic_sha256: [u8; 32] = Sha256::digest(&canonical).into();
    pin.ensure_unchanged()?;
    Ok(PinnedTokenizerArtifact {
        pin,
        semantic_sha256,
    })
}

fn tokenizer_literal_id(tokenizer: &Value, literal: &str) -> Option<i64> {
    tokenizer
        .get("added_tokens")
        .and_then(Value::as_array)
        .and_then(|tokens| {
            tokens.iter().find_map(|token| {
                (token.get("content").and_then(Value::as_str) == Some(literal))
                    .then(|| token.get("id").and_then(Value::as_i64))
                    .flatten()
            })
        })
        .or_else(|| {
            let vocab = tokenizer.get("model")?.get("vocab")?;
            match vocab {
                Value::Object(vocab) => vocab.get(literal).and_then(Value::as_i64),
                // Unigram tokenizers store `[token, score]` rows; the row index is the token id.
                Value::Array(vocab) => vocab.iter().enumerate().find_map(|(id, row)| {
                    (row.as_array()
                        .and_then(|row| row.first())
                        .and_then(Value::as_str)
                        == Some(literal))
                    .then(|| i64::try_from(id).ok())
                    .flatten()
                }),
                _ => None,
            }
        })
}

fn write_canonical_json(value: &Value, output: &mut Vec<u8>) -> Result<()> {
    match value {
        Value::Object(object) => {
            output.push(b'{');
            let mut entries = object.iter().collect::<Vec<_>>();
            entries.sort_by_key(|(key, _)| *key);
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                output.extend(
                    serde_json::to_vec(key).map_err(|error| {
                        Error::Msg(format!("canonicalize tokenizer key: {error}"))
                    })?,
                );
                output.push(b':');
                write_canonical_json(value, output)?;
            }
            output.push(b'}');
        }
        Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_canonical_json(value, output)?;
            }
            output.push(b']');
        }
        scalar => output.extend(
            serde_json::to_vec(scalar)
                .map_err(|error| Error::Msg(format!("canonicalize tokenizer value: {error}")))?,
        ),
    }
    Ok(())
}

fn digest_hex(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn resolve_source(source: &WeightsSource) -> Result<(WeightsSource, Option<PathBuf>)> {
    match source {
        WeightsSource::File(path) => Ok((
            WeightsSource::File(path.clone()),
            path.parent().map(|parent| parent.join("config.json")),
        )),
        WeightsSource::Dir(path) => {
            let nested = path.join("text_encoder");
            let nested_config = nested.join("config.json");
            if nested_config.is_file() {
                return Ok((WeightsSource::Dir(nested), Some(nested_config)));
            }
            let direct = path.join("config.json");
            if direct.is_file() {
                return Ok((WeightsSource::Dir(path.clone()), Some(direct)));
            }
            Ok((WeightsSource::Dir(path.clone()), Some(direct)))
        }
    }
}

fn architecture_matches(actual: &str, expected: &str) -> bool {
    fn normalize(value: &str) -> String {
        value
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect()
    }
    let actual = normalize(actual);
    let expected = normalize(expected);
    if actual == expected {
        return true;
    }
    // Canonical Qwen-Image configs nest a text-only `model_type=qwen2_5_vl_text` beside the known
    // multimodal wrapper class. This one explicit wrapper is architecture-equivalent for the text
    // contract; unrelated `ForConditionalGeneration` wrappers remain rejected.
    if expected == "qwen25vltext" && actual == "qwen25vlforconditionalgeneration" {
        return true;
    }
    actual.strip_prefix(&expected).is_some_and(|suffix| {
        matches!(
            suffix,
            "model" | "forcausallm" | "forconditionalgeneration" | "textmodel"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const TEST_TOKENIZER: EncoderTokenizerContract = EncoderTokenizerContract {
        family: "test-qwen3",
        binding: EncoderTokenizerBinding::RetainBase,
        artifact_candidates: &["tokenizer/tokenizer.json"],
        required_tokens: &[EncoderRequiredToken {
            role: "test_special",
            literal: "<test>",
            id: 1,
            config_field: None,
        }],
    };
    const TEST_PROMPTS: &[EncoderPromptExecutionContract] = &[EncoderPromptExecutionContract {
        purpose: "test_prompt",
        template: EncoderPromptTemplate::QwenInstruct,
        add_special_tokens: true,
        length: EncoderPromptLengthPolicy::RightTruncate { max_tokens: 8 },
        padding: EncoderPromptPadding::None,
        prefix_trim: 0,
    }];

    const CONTRACT: EncoderContract = EncoderContract {
        architecture: "qwen3",
        hidden_size: 8,
        intermediate_size: 12,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 4,
        vocab_size: 16,
        output_width: 8,
        loaded_hidden_layers: 2,
        requires_final_norm: true,
        requires_lm_head: false,
        hidden_activation: "silu",
        attention_dropout: EncoderConfigFloat::new(0.0),
        rms_norm_eps: EncoderConfigFloat::new(1e-6),
        qk_norm_eps: Some(EncoderConfigFloat::new(1e-6)),
        rope_theta: EncoderConfigFloat::new(1_000_000.0),
        max_position_embeddings: 4_096,
        attention_bias: EncoderConfigBool::Required(false),
        tie_word_embeddings: EncoderConfigBool::Required(true),
        tokenizer: TEST_TOKENIZER,
        prompt_executions: TEST_PROMPTS,
        bos_token_id: None,
        eos_token_id: None,
        image_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
        mrope_section: &[],
        mrope_interleaved: None,
        selected_hidden_layers: &[2],
        packing: Some(EncoderPackingContract {
            group_size: 64,
            pack_embedding: false,
            pack_lm_head: false,
            supports_file: true,
        }),
        dense_storage_dtype_probe: Some("model.layers.0.input_layernorm.weight"),
    };

    #[test]
    fn malformed_provider_contract_fails_before_source_inspection() {
        let malformed = EncoderContract {
            num_key_value_heads: 3,
            ..CONTRACT
        };
        let error = malformed
            .validate_source(&WeightsSource::Dir(PathBuf::from(
                "definitely-not-inspected",
            )))
            .expect_err("malformed provider contract must reject first")
            .to_string();
        assert!(error.contains("attention heads (2)"), "{error}");
        assert!(error.contains("key/value heads (3)"), "{error}");
        assert!(!error.contains("No such file"), "{error}");
    }

    fn write_fixture_with_shapes(root: &Path, embedding_hidden: usize, q_projection_output: usize) {
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(
            root.join("config.json"),
            serde_json::to_vec(&json!({
                "model_type": "qwen3",
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 16,
                "hidden_act": "silu",
                "attention_dropout": 0.0,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000.0,
                "max_position_embeddings": 4096,
                "attention_bias": false,
                "tie_word_embeddings": true
            }))
            .unwrap(),
        )
        .unwrap();
        let mut offset = 0usize;
        let mut header = serde_json::Map::new();
        let mut tensors = vec![(
            "model.embed_tokens.weight".to_owned(),
            vec![16, embedding_hidden],
        )];
        for layer in 0..2 {
            let base = format!("model.layers.{layer}");
            tensors.extend([
                (
                    format!("{base}.self_attn.q_proj.weight"),
                    vec![if layer == 0 { q_projection_output } else { 8 }, 8],
                ),
                (format!("{base}.self_attn.k_proj.weight"), vec![4, 8]),
                (format!("{base}.self_attn.v_proj.weight"), vec![4, 8]),
                (format!("{base}.self_attn.o_proj.weight"), vec![8, 8]),
                (format!("{base}.self_attn.q_norm.weight"), vec![4]),
                (format!("{base}.self_attn.k_norm.weight"), vec![4]),
                (format!("{base}.mlp.gate_proj.weight"), vec![12, 8]),
                (format!("{base}.mlp.up_proj.weight"), vec![12, 8]),
                (format!("{base}.mlp.down_proj.weight"), vec![8, 12]),
                (format!("{base}.input_layernorm.weight"), vec![8]),
                (format!("{base}.post_attention_layernorm.weight"), vec![8]),
            ]);
        }
        tensors.push(("model.norm.weight".to_owned(), vec![8]));
        for (name, shape) in tensors {
            let bytes = shape.iter().product::<usize>() * 4;
            header.insert(
                name,
                json!({"dtype":"F32", "shape":shape, "data_offsets":[offset, offset + bytes]}),
            );
            offset += bytes;
        }
        let encoded = serde_json::to_vec(&header).unwrap();
        let mut file = Vec::with_capacity(8 + encoded.len() + offset);
        file.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
        file.extend_from_slice(&encoded);
        file.resize(8 + encoded.len() + offset, 0);
        std::fs::write(root.join("model.safetensors"), file).unwrap();
    }

    fn write_fixture(root: &Path, embedding_hidden: usize) {
        write_fixture_with_shapes(root, embedding_hidden, 8);
    }

    fn write_tokenizer_fixture(root: &Path) {
        let tokenizer_dir = root.join("tokenizer");
        std::fs::create_dir_all(&tokenizer_dir).unwrap();
        std::fs::write(
            tokenizer_dir.join("tokenizer.json"),
            serde_json::to_vec(&json!({
                "added_tokens": [{"id": 1, "content": "<test>"}],
                "model": {"vocab": {}}
            }))
            .unwrap(),
        )
        .unwrap();
    }

    fn write_prefixed_fixture(root: &Path, outer_prefix: &str) -> PathBuf {
        let source = std::fs::read(root.join("model.safetensors")).unwrap();
        let header_len = u64::from_le_bytes(source[..8].try_into().unwrap()) as usize;
        let header: serde_json::Map<String, Value> =
            serde_json::from_slice(&source[8..8 + header_len]).unwrap();
        let mut prefixed = serde_json::Map::new();
        let mut data_len = 0usize;
        for (name, value) in header {
            data_len = data_len.max(
                value["data_offsets"][1]
                    .as_u64()
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap(),
            );
            prefixed.insert(format!("{outer_prefix}{name}"), value);
        }
        let encoded = serde_json::to_vec(&prefixed).unwrap();
        let mut file = Vec::with_capacity(8 + encoded.len() + data_len);
        file.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
        file.extend_from_slice(&encoded);
        file.resize(8 + encoded.len() + data_len, 0);
        let path = root.join("combined.safetensors");
        std::fs::write(&path, file).unwrap();
        path
    }

    fn rewrite_config_field(root: &Path, field: &str, value: usize) {
        let path = root.join("config.json");
        let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config[field] = json!(value);
        std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    }

    fn rewrite_config_quantization(root: &Path, bits: i32) {
        let path = root.join("config.json");
        let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["quantization"] = json!({"bits": bits, "group_size": 64});
        std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    }

    fn authorized_test_roots(root: &Path) -> Vec<PathBuf> {
        let mut roots = vec![
            std::path::absolute(root).unwrap(),
            std::fs::canonicalize(root).unwrap(),
        ];
        roots.sort();
        roots.dedup();
        roots
    }

    #[derive(Clone, Copy, Debug)]
    enum DiscoverySourceShape {
        File,
        DirectDirectory,
        CompleteSnapshot,
    }

    fn write_discovery_source(
        root: &Path,
        shape: DiscoverySourceShape,
        embedding_hidden: usize,
    ) -> (WeightsSource, PathBuf) {
        match shape {
            DiscoverySourceShape::File => {
                let component = root.join("file");
                write_fixture(&component, embedding_hidden);
                (
                    WeightsSource::File(component.join("model.safetensors")),
                    component,
                )
            }
            DiscoverySourceShape::DirectDirectory => {
                let component = root.join("direct");
                write_fixture(&component, embedding_hidden);
                (WeightsSource::Dir(component.clone()), component)
            }
            DiscoverySourceShape::CompleteSnapshot => {
                let snapshot = root.join("snapshot");
                let component = snapshot.join("text_encoder");
                write_fixture(&component, embedding_hidden);
                (WeightsSource::Dir(snapshot), component)
            }
        }
    }

    #[test]
    fn validates_config_and_headers_before_load() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let validated = CONTRACT
            .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
            .unwrap();
        validated
            .read_unchanged::<(), Error>(|source| match source {
                WeightsSource::Dir(_) => Ok(()),
                WeightsSource::File(path) => Err(Error::Msg(format!(
                    "unexpected file encoder source: {}",
                    path.display()
                ))),
            })
            .unwrap();
    }

    #[test]
    fn discovery_validation_does_no_full_hash_work_while_load_validation_still_seals() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("base");
        let selected = temp.path().join("selected");
        write_tokenizer_fixture(&base);
        write_fixture(&selected, 8);
        let source = WeightsSource::Dir(selected);
        let allowed_roots = authorized_test_roots(temp.path());

        let before_discovery = crate::runtime::test_full_hash_work_count();
        let facts = CONTRACT
            .validate_source_for_discovery(&source, &allowed_roots)
            .unwrap();
        assert!(!facts.materialized_language_tensor_headers().is_empty());
        assert!(facts.source_bytes() > 0);
        let inventory =
            text_encoder_source_inventory_for_discovery(&source, &allowed_roots).unwrap();
        assert!(!inventory.tensor_headers().is_empty());
        assert_eq!(inventory.source_bytes(), facts.source_bytes());
        let discovery_planning =
            text_encoder_planning_facts_for_discovery(&source, &allowed_roots).unwrap();
        assert_eq!(
            discovery_planning.source_layout(),
            TextEncoderSourceLayout::DirectDirectory
        );
        assert_eq!(
            discovery_planning.direct_shard_bytes(),
            facts.source_bytes()
        );
        let comfyui_source = WeightsSource::File(match &source {
            WeightsSource::Dir(path) => path.join("model.safetensors"),
            WeightsSource::File(_) => unreachable!("the fixture source is a directory"),
        });
        let comfyui = CONTRACT
            .validate_comfyui_source_for_discovery(&comfyui_source, &allowed_roots)
            .unwrap();
        assert_eq!(comfyui.source_bytes(), facts.source_bytes());
        assert_eq!(
            crate::runtime::test_full_hash_work_count(),
            before_discovery,
            "catalog discovery must not acquire and discard a full-content artifact seal"
        );

        let before_load = crate::runtime::test_full_hash_work_count();
        let sealed = CONTRACT
            .validate_source_against_base(&source, &base)
            .unwrap();
        assert!(
            crate::runtime::test_full_hash_work_count() > before_load,
            "executable validation must retain full acquisition sealing"
        );

        let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base));
        sealed.prepare_load_spec(&mut spec).unwrap();
        let before_prepared_planning = crate::runtime::test_full_hash_work_count();
        let prepared_planning = spec
            .prepared_text_encoder_planning_facts()
            .unwrap()
            .expect("validated source installed a receipt");
        assert_eq!(prepared_planning, discovery_planning);
        assert_eq!(
            crate::runtime::test_full_hash_work_count(),
            before_prepared_planning,
            "prepared planning must reuse the retained seal instead of hashing content again"
        );

        let selected_shard = match &source {
            WeightsSource::Dir(path) => path.join("model.safetensors"),
            WeightsSource::File(_) => unreachable!("the fixture source is a directory"),
        };
        let shard_len = std::fs::metadata(&selected_shard).unwrap().len();
        std::fs::write(&selected_shard, vec![1_u8; shard_len as usize]).unwrap();
        let error = spec
            .prepared_text_encoder_planning_facts()
            .expect_err("prepared planning must reject a mutated selected source")
            .to_string();
        assert!(error.contains("receipt changed"), "{error}");
    }

    #[test]
    fn discovery_and_sealed_validation_have_source_shape_and_contract_parity() {
        #[derive(Clone, Copy, Debug)]
        enum Mutation {
            None,
            Config,
            Header,
            Quantization,
        }

        for shape in [
            DiscoverySourceShape::File,
            DiscoverySourceShape::DirectDirectory,
            DiscoverySourceShape::CompleteSnapshot,
        ] {
            for mutation in [
                Mutation::None,
                Mutation::Config,
                Mutation::Header,
                Mutation::Quantization,
            ] {
                let temp = tempfile::tempdir().unwrap();
                let header_width = if matches!(mutation, Mutation::Header) {
                    7
                } else {
                    8
                };
                let (source, component) = write_discovery_source(temp.path(), shape, header_width);
                match mutation {
                    Mutation::None | Mutation::Header => {}
                    Mutation::Config => rewrite_config_field(&component, "hidden_size", 7),
                    Mutation::Quantization => rewrite_config_quantization(&component, 4),
                }

                let discovery = CONTRACT
                    .validate_source_for_discovery(&source, &authorized_test_roots(temp.path()));
                let sealed = CONTRACT.validate_source_for_planning(&source);
                match mutation {
                    Mutation::None => {
                        let facts = discovery.unwrap_or_else(|error| {
                            panic!("{shape:?} discovery unexpectedly failed: {error}")
                        });
                        let sealed = sealed.unwrap_or_else(|error| {
                            panic!("{shape:?} sealed validation unexpectedly failed: {error}")
                        });
                        assert_eq!(facts.source_bytes(), sealed.source_bytes().unwrap());
                        assert_eq!(
                            facts.materialized_language_tensor_headers(),
                            sealed
                                .materialized_language_tensor_headers(&CONTRACT)
                                .unwrap()
                        );
                    }
                    Mutation::Config => {
                        let discovery = discovery.unwrap_err().to_string();
                        let sealed = sealed.unwrap_err().to_string();
                        assert!(discovery.contains("field hidden_size"), "{discovery}");
                        assert!(sealed.contains("field hidden_size"), "{sealed}");
                    }
                    Mutation::Header => {
                        let discovery = discovery.unwrap_err().to_string();
                        let sealed = sealed.unwrap_err().to_string();
                        assert!(discovery.contains("field hidden_size"), "{discovery}");
                        assert!(sealed.contains("field hidden_size"), "{sealed}");
                    }
                    Mutation::Quantization => {
                        let discovery = discovery.unwrap_err().to_string();
                        let sealed = sealed.unwrap_err().to_string();
                        assert!(discovery.contains("quantization mismatch"), "{discovery}");
                        assert!(sealed.contains("quantization mismatch"), "{sealed}");
                    }
                }
            }
        }
    }

    #[test]
    fn required_config_is_rejected_before_shard_enumeration() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("selected");
        std::fs::create_dir_all(source.join("broken.safetensors")).unwrap();
        let error = CONTRACT
            .validate_source_for_discovery(
                &WeightsSource::Dir(source),
                &authorized_test_roots(temp.path()),
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("has no config.json"), "{error}");
        assert!(!error.contains("not a regular file"), "{error}");
    }

    #[test]
    fn discovery_validation_rejects_config_header_and_source_shape_mutations() {
        let temp = tempfile::tempdir().unwrap();
        let allowed_roots = authorized_test_roots(temp.path());

        let config_mutation = temp.path().join("config-mutation");
        write_fixture(&config_mutation, 8);
        rewrite_config_field(&config_mutation, "hidden_size", 7);
        let error = CONTRACT
            .validate_source_for_discovery(&WeightsSource::Dir(config_mutation), &allowed_roots)
            .unwrap_err()
            .to_string();
        assert!(error.contains("field hidden_size"), "{error}");

        let header_mutation = temp.path().join("header-mutation");
        write_fixture(&header_mutation, 7);
        let error = CONTRACT
            .validate_source_for_discovery(&WeightsSource::Dir(header_mutation), &allowed_roots)
            .unwrap_err()
            .to_string();
        assert!(error.contains("field hidden_size"), "{error}");

        let source_shape_mutation = temp.path().join("source-shape-mutation");
        write_fixture(&source_shape_mutation, 8);
        CONTRACT
            .validate_source_for_discovery(
                &WeightsSource::Dir(source_shape_mutation.clone()),
                &allowed_roots,
            )
            .unwrap();
        let nested = source_shape_mutation.join("text_encoder");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::copy(
            source_shape_mutation.join("config.json"),
            nested.join("config.json"),
        )
        .unwrap();
        let error = CONTRACT
            .validate_source_for_discovery(
                &WeightsSource::Dir(source_shape_mutation),
                &allowed_roots,
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("no direct .safetensors shards"), "{error}");
    }

    #[test]
    fn discovery_and_sealed_validation_share_the_config_size_policy() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        std::fs::OpenOptions::new()
            .write(true)
            .open(temp.path().join("config.json"))
            .unwrap()
            .set_len(MAX_SELECTED_ENCODER_CONFIG_BYTES + 1)
            .unwrap();
        let source = WeightsSource::Dir(temp.path().to_path_buf());
        let discovery_error = CONTRACT
            .validate_source_for_discovery(&source, &authorized_test_roots(temp.path()))
            .unwrap_err()
            .to_string();
        let sealed_error = CONTRACT
            .validate_source_for_planning(&source)
            .unwrap_err()
            .to_string();
        assert!(
            discovery_error.contains("exceeds the 4194304-byte maximum"),
            "{discovery_error}"
        );
        assert!(
            sealed_error.contains("exceeds the 4194304-byte maximum"),
            "{sealed_error}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn discovery_validation_rejects_canonical_target_outside_authorized_roots() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let allowed = temp.path().join("allowed");
        let outside = temp.path().join("outside");
        write_fixture(&outside, 8);
        std::fs::create_dir_all(&allowed).unwrap();
        std::fs::copy(outside.join("config.json"), allowed.join("config.json")).unwrap();
        symlink(
            outside.join("model.safetensors"),
            allowed.join("model.safetensors"),
        )
        .unwrap();

        let allowed_roots = authorized_test_roots(&allowed);
        let error = CONTRACT
            .validate_source_for_discovery(&WeightsSource::Dir(allowed.clone()), &allowed_roots)
            .unwrap_err()
            .to_string();
        assert!(error.contains("canonical target"), "{error}");
        assert!(error.contains("escapes authorized model roots"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn discovery_confinement_covers_file_direct_directory_and_complete_snapshot() {
        use std::os::unix::fs::symlink;

        for shape in [
            DiscoverySourceShape::File,
            DiscoverySourceShape::DirectDirectory,
            DiscoverySourceShape::CompleteSnapshot,
        ] {
            let temp = tempfile::tempdir().unwrap();
            let allowed = temp.path().join("allowed");
            let outside = temp.path().join("outside");
            let (source, component) = write_discovery_source(&allowed, shape, 8);
            write_fixture(&outside, 8);
            std::fs::remove_file(component.join("model.safetensors")).unwrap();
            symlink(
                outside.join("model.safetensors"),
                component.join("model.safetensors"),
            )
            .unwrap();

            let error = CONTRACT
                .validate_source_for_discovery(&source, &authorized_test_roots(&allowed))
                .unwrap_err()
                .to_string();
            assert!(error.contains("canonical target"), "{shape:?}: {error}");
            assert!(
                error.contains("escapes authorized model roots"),
                "{shape:?}: {error}"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn discovery_rejects_snapshot_component_symlink_before_inspecting_its_config() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let allowed = temp.path().join("allowed");
        let snapshot = allowed.join("snapshot");
        let outside_component = temp.path().join("outside/text_encoder");
        std::fs::create_dir_all(&snapshot).unwrap();
        write_fixture(&outside_component, 8);
        symlink(&outside_component, snapshot.join("text_encoder")).unwrap();

        let error = CONTRACT
            .validate_source_for_discovery(
                &WeightsSource::Dir(snapshot),
                &authorized_test_roots(&allowed),
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("canonical target"), "{error}");
        assert!(error.contains("escapes authorized model roots"), "{error}");
    }

    #[test]
    fn duplicate_direct_shard_keys_fail_before_backend_selection() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        std::fs::copy(
            temp.path().join("model.safetensors"),
            temp.path().join("model-00002.safetensors"),
        )
        .unwrap();

        let error = CONTRACT
            .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
            .expect_err("cross-backend duplicate semantics must fail closed")
            .to_string();
        assert!(error.contains("duplicate tensor key"), "{error}");
    }

    #[test]
    fn configless_file_rejects_unprovable_behavior_and_head_topology() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let file = temp.path().join("model.safetensors");
        std::fs::remove_file(temp.path().join("config.json")).unwrap();
        let error = CONTRACT
            .validate_source(&WeightsSource::File(file))
            .expect_err("headers cannot prove behavior or tokenizer compatibility")
            .to_string();
        assert!(error.contains("has no config.json"), "{error}");
        assert!(error.contains("tokenizer"), "{error}");
    }

    #[test]
    fn provider_owned_comfyui_file_preserves_the_legacy_configless_surface() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let file = temp.path().join("model.safetensors");
        std::fs::remove_file(temp.path().join("config.json")).unwrap();

        let validated = CONTRACT
            .validate_comfyui_source(&WeightsSource::File(file.clone()))
            .unwrap();
        validated
            .read_unchanged::<(), Error>(|source| match source {
                WeightsSource::File(actual) if actual == &file => Ok(()),
                other => Err(Error::Msg(format!("unexpected encoder source: {other:?}"))),
            })
            .unwrap();
        assert!(!validated.has_config());
    }

    #[test]
    fn provider_owned_comfyui_file_still_rejects_a_present_wrong_config() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        rewrite_config_field(temp.path(), "hidden_size", 7);

        let error = CONTRACT
            .validate_comfyui_source(&WeightsSource::File(temp.path().join("model.safetensors")))
            .unwrap_err()
            .to_string();
        assert!(error.contains("field hidden_size"), "{error}");
        assert!(error.contains("expected 8, got 7"), "{error}");
    }

    #[test]
    fn validates_dense_encoder_embedded_in_fused_checkpoint() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let file = write_prefixed_fixture(temp.path(), "text_encoder.");
        CONTRACT
            .validate_embedded_file(&file, &["text_encoder."])
            .unwrap();
    }

    #[test]
    fn provider_owned_comfyui_checkpoint_preserves_the_legacy_configless_surface() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let file = write_prefixed_fixture(temp.path(), "text_encoder.");
        std::fs::remove_file(temp.path().join("config.json")).unwrap();
        CONTRACT
            .validate_embedded_comfyui_file(&file, &["text_encoder."])
            .unwrap();
    }

    #[test]
    fn ordinary_embedded_encoder_still_requires_its_own_config() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let file = write_prefixed_fixture(temp.path(), "text_encoder.");
        std::fs::remove_file(temp.path().join("config.json")).unwrap();
        let error = CONTRACT
            .validate_embedded_file(&file, &["text_encoder."])
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires sibling config.json"), "{error}");
    }

    #[test]
    fn embedded_encoder_header_mismatch_rejects_before_payload_load() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 7);
        let file = write_prefixed_fixture(temp.path(), "text_encoder.");
        let error = CONTRACT
            .validate_embedded_file(&file, &["text_encoder."])
            .unwrap_err()
            .to_string();
        assert!(error.contains("field hidden_size"), "{error}");
        assert!(error.contains("expected 8, got 7"), "{error}");
    }

    #[test]
    fn wrong_tensor_shape_names_the_exact_contract_field() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 7);
        let error = CONTRACT
            .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("field hidden_size"), "{error}");
        assert!(error.contains("expected 8, got 7"), "{error}");
    }

    #[test]
    fn wrong_config_hidden_size_names_the_exact_contract_field() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        rewrite_config_field(temp.path(), "hidden_size", 7);
        let error = CONTRACT
            .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("field hidden_size"), "{error}");
        assert!(error.contains("expected 8, got 7"), "{error}");
    }

    #[test]
    fn wrong_config_layer_count_names_the_exact_contract_field() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        rewrite_config_field(temp.path(), "num_hidden_layers", 3);
        let error = CONTRACT
            .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("field num_hidden_layers"), "{error}");
        assert!(error.contains("expected 2, got 3"), "{error}");
    }

    #[test]
    fn wrong_projection_shape_names_the_exact_head_contract_field() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture_with_shapes(temp.path(), 8, 7);
        let error = CONTRACT
            .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("field num_attention_heads/head_dim"),
            "{error}"
        );
        assert!(error.contains("expected 8, got 7"), "{error}");
    }

    #[test]
    fn absent_override_validates_and_preserves_the_builtin_source() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path();
        write_fixture(&base.join("text_encoder"), 8);
        write_tokenizer_fixture(base);
        let source = CONTRACT
            .source_for_load(&crate::LoadSpec::new(WeightsSource::Dir(base.into())), base)
            .unwrap();
        source
            .read_unchanged::<(), Error>(|source| match source {
                WeightsSource::Dir(path) if path.as_path() == base.join("text_encoder") => Ok(()),
                other => Err(Error::Msg(format!("unexpected encoder source: {other:?}"))),
            })
            .unwrap();
    }

    #[test]
    fn retained_tokenizer_rejects_replacement_before_runtime_open() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path();
        write_fixture(&base.join("text_encoder"), 8);
        write_tokenizer_fixture(base);
        let source = CONTRACT
            .source_for_load(&crate::LoadSpec::new(WeightsSource::Dir(base.into())), base)
            .unwrap();
        let tokenizer = base.join("tokenizer/tokenizer.json");
        let replacement = base.join("tokenizer/replacement.json");
        std::fs::write(
            &replacement,
            br#"{"added_tokens":[{"id":1,"content":"<test>"}],"model":{"vocab":{"replacement":2}}}"#,
        )
        .unwrap();
        std::fs::rename(replacement, tokenizer).unwrap();

        let opened = std::cell::Cell::new(false);
        let error = source
            .read_tokenizer_unchanged::<(), Error>(|_| {
                opened.set(true);
                Ok(())
            })
            .unwrap_err()
            .to_string();
        assert!(!opened.get(), "mutated tokenizer bytes must not be opened");
        assert!(error.contains("pinned weights"), "{error}");
    }

    #[test]
    fn retained_tokenizer_rejects_swap_and_restore_inside_read_bracket() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path();
        write_fixture(&base.join("text_encoder"), 8);
        write_tokenizer_fixture(base);
        let source = CONTRACT
            .source_for_load(&crate::LoadSpec::new(WeightsSource::Dir(base.into())), base)
            .unwrap();
        let tokenizer = base.join("tokenizer/tokenizer.json");
        let original = base.join("tokenizer/original.json");

        let error = source
            .read_tokenizer_unchanged::<(), Error>(|path| {
                std::fs::rename(path, &original)?;
                std::fs::write(
                    path,
                    br#"{"added_tokens":[{"id":1,"content":"<test>"}],"model":{"vocab":{"swapped":2}}}"#,
                )?;
                std::fs::remove_file(path)?;
                std::fs::rename(&original, path)?;
                Ok(())
            })
            .unwrap_err()
            .to_string();
        assert!(error.contains("pinned weights"), "{error}");
        assert!(tokenizer.is_file());
    }

    #[cfg(unix)]
    #[test]
    fn retained_tokenizer_rejects_symlink_retarget_before_runtime_open() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let base = temp.path();
        write_fixture(&base.join("text_encoder"), 8);
        write_tokenizer_fixture(base);
        let tokenizer = base.join("tokenizer/tokenizer.json");
        let first = base.join("tokenizer/first.json");
        let second = base.join("tokenizer/second.json");
        std::fs::rename(&tokenizer, &first).unwrap();
        std::fs::copy(&first, &second).unwrap();
        symlink(&first, &tokenizer).unwrap();
        let source = CONTRACT
            .source_for_load(&crate::LoadSpec::new(WeightsSource::Dir(base.into())), base)
            .unwrap();

        std::fs::remove_file(&tokenizer).unwrap();
        symlink(&second, &tokenizer).unwrap();
        let opened = std::cell::Cell::new(false);
        let error = source
            .read_tokenizer_unchanged::<(), Error>(|_| {
                opened.set(true);
                Ok(())
            })
            .unwrap_err()
            .to_string();
        assert!(!opened.get(), "retargeted tokenizer must fail before open");
        assert!(error.contains("pinned weights"), "{error}");
    }

    #[test]
    fn semantically_identical_complete_snapshot_retains_the_base_runtime_tokenizer() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("base");
        let selected = temp.path().join("selected");
        write_fixture(&base.join("text_encoder"), 8);
        write_tokenizer_fixture(&base);
        write_fixture(&selected.join("text_encoder"), 8);
        std::fs::create_dir_all(selected.join("tokenizer")).unwrap();
        std::fs::write(
            selected.join("tokenizer/tokenizer.json"),
            br#"{ "model": { "vocab": {} }, "added_tokens": [ { "content": "<test>", "id": 1 } ] }"#,
        )
        .unwrap();
        let spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()))
            .with_text_encoder(WeightsSource::Dir(selected));
        let source = CONTRACT.source_for_load(&spec, &base).unwrap();

        assert_eq!(
            source.tokenizer_disposition(),
            Some(EncoderTokenizerDisposition::MatchedSelectedTokenizer)
        );
        let runtime_path = source
            .read_tokenizer_unchanged::<PathBuf, Error>(|path| Ok(path.to_path_buf()))
            .unwrap();
        assert_eq!(
            runtime_path,
            std::path::absolute(base.join("tokenizer/tokenizer.json")).unwrap()
        );
    }

    #[test]
    fn file_selection_prepares_shard_sibling_config_and_base_tokenizer_identity() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("base");
        let selected = temp.path().join("selected");
        std::fs::create_dir_all(&base).unwrap();
        write_tokenizer_fixture(&base);
        write_fixture(&selected, 8);
        let selected_file = selected.join("model.safetensors");
        let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
        let validated = CONTRACT
            .validate_source_against_base(&WeightsSource::File(selected_file.clone()), &base)
            .unwrap();

        validated.prepare_load_spec(&mut spec).unwrap();

        assert_eq!(
            spec.text_encoder,
            Some(WeightsSource::File(selected_file.clone()))
        );
        let paths = spec
            .prepared_file_pins()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        assert!(paths.contains(&std::path::absolute(&selected_file).unwrap()));
        assert!(paths.contains(&std::path::absolute(selected.join("config.json")).unwrap()));
        assert!(
            paths.contains(&std::path::absolute(base.join("tokenizer/tokenizer.json")).unwrap())
        );

        std::fs::write(selected.join("config.json"), b"{}").unwrap();
        let opened = std::cell::Cell::new(false);
        let error = spec
            .read_prepared_files_unchanged::<(), Error>(|| {
                opened.set(true);
                Ok(())
            })
            .unwrap_err()
            .to_string();
        assert!(
            !opened.get(),
            "mutated config must fail before provider load"
        );
        assert!(error.contains("pinned weights"), "{error}");
    }

    #[test]
    fn complete_snapshot_prepares_selected_and_base_tokenizer_identity() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("base");
        let selected = temp.path().join("selected");
        std::fs::create_dir_all(&base).unwrap();
        write_tokenizer_fixture(&base);
        write_fixture(&selected.join("text_encoder"), 8);
        write_tokenizer_fixture(&selected);
        let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
        let validated = CONTRACT
            .validate_source_against_base(&WeightsSource::Dir(selected.clone()), &base)
            .unwrap();

        validated.prepare_load_spec(&mut spec).unwrap();

        assert_eq!(
            spec.text_encoder,
            Some(WeightsSource::Dir(selected.clone()))
        );
        let paths = spec
            .prepared_file_pins()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        for path in [
            selected.join("text_encoder/model.safetensors"),
            selected.join("text_encoder/config.json"),
            selected.join("tokenizer/tokenizer.json"),
            base.join("tokenizer/tokenizer.json"),
        ] {
            assert!(
                paths.contains(&std::path::absolute(path).unwrap()),
                "missing prepared receipt path"
            );
        }

        std::fs::write(
            selected.join("tokenizer/tokenizer.json"),
            br#"{"added_tokens":[]}"#,
        )
        .unwrap();
        let opened = std::cell::Cell::new(false);
        let error = spec
            .read_prepared_files_unchanged::<(), Error>(|| {
                opened.set(true);
                Ok(())
            })
            .expect_err("mutated selected tokenizer must fail before provider load")
            .to_string();
        assert!(!opened.get());
        assert!(error.contains("pinned weights"), "{error}");
    }

    #[test]
    fn prepared_directory_receipt_rejects_shard_inventory_mutation() {
        for mutation in ["addition", "removal", "rename", "type"] {
            let temp = tempfile::tempdir().unwrap();
            let base = temp.path().join("base");
            let selected = temp.path().join("selected");
            std::fs::create_dir_all(&base).unwrap();
            write_tokenizer_fixture(&base);
            write_fixture(&selected, 8);
            let shard = selected.join("model.safetensors");
            let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
            let validated = CONTRACT
                .validate_source_against_base(&WeightsSource::Dir(selected.clone()), &base)
                .unwrap();
            validated.prepare_load_spec(&mut spec).unwrap();

            match mutation {
                "addition" => {
                    std::fs::copy(&shard, selected.join("added.safetensors")).unwrap();
                }
                "removal" => std::fs::remove_file(&shard).unwrap(),
                "rename" => {
                    std::fs::rename(&shard, selected.join("renamed.safetensors")).unwrap();
                }
                "type" => {
                    std::fs::remove_file(&shard).unwrap();
                    std::fs::create_dir(&shard).unwrap();
                }
                _ => unreachable!(),
            }

            let opened = std::cell::Cell::new(false);
            let error = spec
                .read_prepared_files_unchanged::<(), Error>(|| {
                    opened.set(true);
                    Ok(())
                })
                .expect_err("mutated direct-shard inventory must fail before provider load")
                .to_string();
            assert!(!opened.get(), "{mutation} reached the provider callback");
            assert!(
                error.contains("receipt changed") || error.contains("pinned weights"),
                "{mutation}: {error}"
            );
        }
    }

    #[test]
    fn source_for_load_rechecks_prepared_directory_inventory() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("base");
        let selected = temp.path().join("selected");
        std::fs::create_dir_all(&base).unwrap();
        write_tokenizer_fixture(&base);
        write_fixture(&selected, 8);
        let shard = selected.join("model.safetensors");
        let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
        let validated = CONTRACT
            .validate_source_against_base(&WeightsSource::Dir(selected.clone()), &base)
            .unwrap();
        validated.prepare_load_spec(&mut spec).unwrap();

        std::fs::copy(shard, selected.join("added.safetensors")).unwrap();

        let error = CONTRACT
            .source_for_load(&spec, &base)
            .expect_err("direct provider validation must retain the prepared shard inventory")
            .to_string();
        assert!(error.contains("receipt changed"), "{error}");
    }

    #[test]
    fn prepared_encoder_receipt_rejects_selected_source_replacement() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("base");
        let selected = temp.path().join("selected");
        std::fs::create_dir_all(&base).unwrap();
        write_tokenizer_fixture(&base);
        write_fixture(&selected, 8);
        let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
        let validated = CONTRACT
            .validate_source_against_base(&WeightsSource::Dir(selected), &base)
            .unwrap();
        validated.prepare_load_spec(&mut spec).unwrap();

        spec.text_encoder = Some(WeightsSource::Dir(temp.path().join("replacement")));
        let error = spec
            .validate_prepared_file_pins()
            .expect_err("prepared encoder source replacement must fail closed")
            .to_string();
        assert!(error.contains("no longer matches"), "{error}");
    }

    #[test]
    fn prepared_file_receipt_rejects_later_config_sidecar() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("base");
        let selected = temp.path().join("selected");
        std::fs::create_dir_all(&base).unwrap();
        write_tokenizer_fixture(&base);
        write_fixture(&selected, 8);
        let selected_file = selected.join("model.safetensors");
        std::fs::remove_file(selected.join("config.json")).unwrap();
        let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
        let validated = CONTRACT
            .validate_comfyui_source_against_base(&WeightsSource::File(selected_file), &base)
            .unwrap();
        validated.prepare_load_spec(&mut spec).unwrap();

        std::fs::write(selected.join("config.json"), b"{}").unwrap();
        let opened = std::cell::Cell::new(false);
        let error = spec
            .read_prepared_files_unchanged::<(), Error>(|| {
                opened.set(true);
                Ok(())
            })
            .expect_err("a newly added behavior sidecar must fail before provider load")
            .to_string();
        assert!(!opened.get());
        assert!(error.contains("receipt changed"), "{error}");
    }

    #[test]
    fn prepared_direct_directory_rejects_complete_snapshot_reinterpretation() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("base");
        let selected = temp.path().join("selected");
        std::fs::create_dir_all(&base).unwrap();
        write_tokenizer_fixture(&base);
        write_fixture(&selected, 8);
        let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
        let validated = CONTRACT
            .validate_source_against_base(&WeightsSource::Dir(selected.clone()), &base)
            .unwrap();
        validated.prepare_load_spec(&mut spec).unwrap();

        write_fixture(&selected.join("text_encoder"), 8);
        let error = spec
            .validate_prepared_file_pins()
            .expect_err("a direct component cannot become a complete snapshot after preparation")
            .to_string();
        assert!(error.contains("source shape changed"), "{error}");
    }

    #[test]
    fn prepared_receipt_rejects_higher_priority_tokenizer_candidate_addition() {
        const CANDIDATE_TOKENIZER: EncoderTokenizerContract = EncoderTokenizerContract {
            artifact_candidates: &["tokenizer/tokenizer.json", "processor/tokenizer.json"],
            ..TEST_TOKENIZER
        };
        const CANDIDATE_CONTRACT: EncoderContract = EncoderContract {
            tokenizer: CANDIDATE_TOKENIZER,
            ..CONTRACT
        };

        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("base");
        let selected = temp.path().join("selected");
        write_tokenizer_fixture(&base);
        std::fs::create_dir_all(base.join("processor")).unwrap();
        std::fs::rename(
            base.join("tokenizer/tokenizer.json"),
            base.join("processor/tokenizer.json"),
        )
        .unwrap();
        write_fixture(&selected, 8);
        let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
        let validated = CANDIDATE_CONTRACT
            .validate_source_against_base(&WeightsSource::Dir(selected), &base)
            .unwrap();
        validated.prepare_load_spec(&mut spec).unwrap();

        write_tokenizer_fixture(&base);
        let error = spec
            .validate_prepared_file_pins()
            .expect_err("a new higher-priority tokenizer must not replace the retained artifact")
            .to_string();
        assert!(error.contains("tokenizer selection changed"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn prepared_directory_receipt_rejects_shard_symlink_retarget() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("base");
        let selected = temp.path().join("selected");
        std::fs::create_dir_all(&base).unwrap();
        write_tokenizer_fixture(&base);
        write_fixture(&selected, 8);
        let selected_shard = selected.join("model.safetensors");
        let target_a = temp.path().join("target-a.safetensors");
        let target_b = temp.path().join("target-b.safetensors");
        std::fs::rename(&selected_shard, &target_a).unwrap();
        std::fs::copy(&target_a, &target_b).unwrap();
        symlink(&target_a, &selected_shard).unwrap();
        let mut spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()));
        let validated = CONTRACT
            .validate_source_against_base(&WeightsSource::Dir(selected.clone()), &base)
            .unwrap();
        validated.prepare_load_spec(&mut spec).unwrap();

        let staged = selected.join("staged.safetensors");
        symlink(&target_b, &staged).unwrap();
        std::fs::rename(staged, &selected_shard).unwrap();

        let opened = std::cell::Cell::new(false);
        let error = spec
            .read_prepared_files_unchanged::<(), Error>(|| {
                opened.set(true);
                Ok(())
            })
            .expect_err("retargeted shard symlink must fail before provider load")
            .to_string();
        assert!(!opened.get());
        assert!(error.contains("pinned weights"), "{error}");
    }

    #[test]
    fn direct_component_inherits_and_pins_the_base_runtime_tokenizer() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("base");
        let component = temp.path().join("alternate-component");
        write_fixture(&base.join("text_encoder"), 8);
        write_tokenizer_fixture(&base);
        write_fixture(&component, 8);
        let spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()))
            .with_text_encoder(WeightsSource::Dir(component));
        let source = CONTRACT.source_for_load(&spec, &base).unwrap();

        assert_eq!(
            source.tokenizer_disposition(),
            Some(EncoderTokenizerDisposition::InheritedBase)
        );
        let runtime_path = source
            .read_tokenizer_unchanged::<PathBuf, Error>(|path| Ok(path.to_path_buf()))
            .unwrap();
        assert_eq!(
            runtime_path,
            std::path::absolute(base.join("tokenizer/tokenizer.json")).unwrap()
        );
    }

    #[test]
    fn validated_inventory_and_pricing_ignore_nested_safetensors() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let direct = temp.path().join("model.safetensors");
        let direct_bytes = std::fs::metadata(&direct).unwrap().len();
        write_fixture(&temp.path().join("nested"), 8);

        let selected = CONTRACT
            .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
            .unwrap();
        assert_eq!(selected.source_bytes().unwrap(), direct_bytes);
        assert_eq!(
            selected.tensor_headers().unwrap().len(),
            safetensors_path_tensor_headers(&direct).unwrap().len()
        );
    }

    #[test]
    fn direct_safetensors_directory_is_rejected_as_a_non_file_shard() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        std::fs::create_dir(temp.path().join("bogus.safetensors")).unwrap();

        let error = CONTRACT
            .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("not a regular file"), "{error}");
        assert!(error.contains("bogus.safetensors"), "{error}");
    }

    #[cfg(unix)]
    #[test]
    fn broken_direct_safetensors_symlink_is_rejected_during_inventory() {
        use std::os::unix::fs::symlink;

        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        symlink(
            temp.path().join("missing-target"),
            temp.path().join("broken.safetensors"),
        )
        .unwrap();

        let error = CONTRACT
            .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("cannot be inspected"), "{error}");
        assert!(error.contains("broken.safetensors"), "{error}");
    }

    #[test]
    fn absent_override_rejects_a_doctored_builtin_config_and_names_the_field() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path();
        let encoder = base.join("text_encoder");
        write_fixture(&encoder, 8);
        write_tokenizer_fixture(base);
        let config_path = encoder.join("config.json");
        let mut config: Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        config.as_object_mut().unwrap().remove("num_hidden_layers");
        std::fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();

        let error = CONTRACT
            .source_for_load(&crate::LoadSpec::new(WeightsSource::Dir(base.into())), base)
            .unwrap_err()
            .to_string();
        assert!(error.contains("field num_hidden_layers"), "{error}");
        assert!(error.contains("expected 2, got missing"), "{error}");
    }

    #[test]
    fn dense_selected_encoder_inherits_the_effective_quant_tier() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        assert_eq!(
            text_encoder_load_time_quant_bits(
                &WeightsSource::Dir(temp.path().to_path_buf()),
                Some(8),
                "fixture",
            )
            .unwrap(),
            Some(8)
        );
    }

    #[test]
    fn matching_prepacked_selected_encoder_needs_no_requantization() {
        let temp = tempfile::tempdir().unwrap();
        write_packed_fixture(temp.path(), 8, true);
        assert_eq!(
            text_encoder_load_time_quant_bits(
                &WeightsSource::Dir(temp.path().to_path_buf()),
                Some(4),
                "fixture",
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn prepacked_selected_encoder_rejects_a_different_effective_tier() {
        let temp = tempfile::tempdir().unwrap();
        write_packed_fixture(temp.path(), 8, true);
        let error = text_encoder_load_time_quant_bits(
            &WeightsSource::Dir(temp.path().to_path_buf()),
            Some(8),
            "fixture",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("pre-quantized Q4"), "{error}");
        assert!(error.contains("model policy is Q8"), "{error}");
    }

    #[test]
    fn prepacked_selected_encoder_rejects_a_dense_model_policy() {
        let temp = tempfile::tempdir().unwrap();
        write_packed_fixture(temp.path(), 8, true);
        let error = text_encoder_load_time_quant_bits(
            &WeightsSource::Dir(temp.path().to_path_buf()),
            None,
            "fixture",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("pre-quantized Q4"), "{error}");
        assert!(error.contains("model policy is dense"), "{error}");
    }

    #[test]
    fn dense_headers_reject_a_false_packed_config_marker() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        rewrite_config_quantization(temp.path(), 4);

        let error = CONTRACT
            .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("config declares Q4"), "{error}");
        assert!(error.contains("direct-shard surface is dense"), "{error}");
    }

    #[test]
    fn ignored_vision_packing_does_not_determine_the_language_tier() {
        let headers = vec![
            SafetensorsTensorHeader {
                name: "model.embed_tokens.weight".into(),
                dtype: Dtype::F16,
                shape: vec![16, 64],
                data_bytes: 2_048,
            },
            SafetensorsTensorHeader {
                name: "visual.proj.weight".into(),
                dtype: Dtype::U32,
                shape: vec![64, 1],
                data_bytes: 256,
            },
            SafetensorsTensorHeader {
                name: "visual.proj.scales".into(),
                dtype: Dtype::F16,
                shape: vec![64, 1],
                data_bytes: 128,
            },
            SafetensorsTensorHeader {
                name: "visual.proj.biases".into(),
                dtype: Dtype::F16,
                shape: vec![64],
                data_bytes: 128,
            },
        ];
        let language = language_quantization_evidence_headers(&headers);
        assert_eq!(
            validate_quantization_evidence(&language, Path::new("fixture"), None).unwrap(),
            None
        );
        let error = validate_quantization_evidence(
            &language,
            Path::new("fixture"),
            Some(PackedQuantization {
                bits: 4,
                group_size: 64,
            }),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("direct-shard surface is dense"), "{error}");
    }

    #[test]
    fn ignored_decoder_tail_packing_does_not_determine_the_language_tier() {
        let headers = vec![
            SafetensorsTensorHeader {
                name: "model.embed_tokens.weight".into(),
                dtype: Dtype::F16,
                shape: vec![16, 64],
                data_bytes: 2_048,
            },
            SafetensorsTensorHeader {
                name: "model.layers.0.self_attn.q_proj.weight".into(),
                dtype: Dtype::F16,
                shape: vec![64, 64],
                data_bytes: 8_192,
            },
            SafetensorsTensorHeader {
                name: "model.layers.1.self_attn.q_proj.weight".into(),
                dtype: Dtype::U32,
                shape: vec![64, 8],
                data_bytes: 2_048,
            },
            SafetensorsTensorHeader {
                name: "model.layers.1.self_attn.q_proj.scales".into(),
                dtype: Dtype::F16,
                shape: vec![64, 1],
                data_bytes: 128,
            },
            SafetensorsTensorHeader {
                name: "model.layers.1.self_attn.q_proj.biases".into(),
                dtype: Dtype::F16,
                shape: vec![64, 1],
                data_bytes: 128,
            },
        ];
        let language = language_quantization_evidence_headers(&headers);
        assert_eq!(language.len(), 2);
        assert_eq!(
            validate_quantization_evidence(&language, Path::new("fixture"), None).unwrap(),
            None
        );
    }

    #[test]
    fn ignored_decoder_tail_packed_components_are_not_validated() {
        let contract = EncoderContract {
            loaded_hidden_layers: 1,
            ..CONTRACT
        };
        let headers = vec![
            SafetensorsTensorHeader {
                name: "model.layers.1.self_attn.q_proj.weight".into(),
                dtype: Dtype::U32,
                shape: vec![64, usize::MAX],
                data_bytes: 0,
            },
            SafetensorsTensorHeader {
                name: "model.layers.1.mlp.up_proj.scales".into(),
                dtype: Dtype::F16,
                shape: vec![1],
                data_bytes: 2,
            },
        ];
        contract
            .validate_all_packed_triples(&headers, Path::new("fixture"), None, &BTreeSet::new())
            .unwrap();
    }

    #[test]
    fn candle_krea_storage_probe_rejects_only_the_lossy_bf16_branch() {
        let header = |name: &str, dtype| SafetensorsTensorHeader {
            name: name.into(),
            dtype,
            shape: vec![8],
            data_bytes: 16,
        };
        let probe = CONTRACT.dense_storage_dtype_probe.unwrap();
        let mixed_f32_store = vec![
            header(probe, Dtype::F32),
            header("model.layers.0.self_attn.q_proj.weight", Dtype::BF16),
        ];
        CONTRACT
            .validate_dense_storage_probe(&mixed_f32_store, Path::new("fixture"))
            .unwrap();

        let lossy_bf16_store = vec![
            header(probe, Dtype::BF16),
            header("model.layers.0.self_attn.q_proj.weight", Dtype::F32),
        ];
        let error = CONTRACT
            .validate_dense_storage_probe(&lossy_bf16_store, Path::new("fixture"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("dense_storage_dtype"), "{error}");
        assert!(error.contains("q_proj.weight=F32"), "{error}");
    }

    #[test]
    fn candle_krea_storage_probe_ignores_unloaded_decoder_tail() {
        let contract = EncoderContract {
            loaded_hidden_layers: 1,
            ..CONTRACT
        };
        let header = |name: &str, dtype| SafetensorsTensorHeader {
            name: name.into(),
            dtype,
            shape: vec![8],
            data_bytes: 16,
        };
        let probe = contract.dense_storage_dtype_probe.unwrap();
        contract
            .validate_dense_storage_probe(
                &[
                    header(probe, Dtype::BF16),
                    header("model.layers.0.self_attn.q_proj.weight", Dtype::BF16),
                    header("model.layers.1.self_attn.q_proj.weight", Dtype::F32),
                ],
                Path::new("fixture"),
            )
            .unwrap();
    }

    #[test]
    fn model_type_is_authoritative_over_a_stale_matching_architectures_alias() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let path = temp.path().join("config.json");
        let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["model_type"] = json!("mistral");
        config["architectures"] = json!(["Qwen3ForCausalLM"]);

        let error = CONTRACT
            .validate_config(&config, &path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("architecture.model_type"), "{error}");
        assert!(error.contains("mistral"), "{error}");
    }

    #[test]
    fn every_architectures_alias_must_match_the_model_type_contract() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let path = temp.path().join("config.json");
        let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config["architectures"] = json!(["Qwen3ForCausalLM", "MistralForCausalLM"]);

        let error = CONTRACT
            .validate_config(&config, &path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("architecture.architectures"), "{error}");
        assert!(error.contains("MistralForCausalLM"), "{error}");
    }

    #[test]
    fn every_rope_theta_and_type_alias_must_match() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let path = temp.path().join("config.json");
        let mut theta: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        theta["rope_parameters"] = json!({"rope_theta": 500000.0});
        let error = CONTRACT
            .validate_config(&theta, &path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("rope_theta"), "{error}");
        assert!(error.contains("rope_parameters.rope_theta"), "{error}");

        let mut rope_type: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        rope_type["rope_type"] = json!("default");
        rope_type["rope_scaling"] = json!({"type": "linear"});
        let error = CONTRACT
            .validate_config(&rope_type, &path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("rope_type"), "{error}");
        assert!(error.contains("rope_scaling.type"), "{error}");
    }

    #[test]
    fn every_mrope_section_and_interleaving_alias_must_match() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let path = temp.path().join("config.json");
        let mut section: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        section["rope_parameters"] = json!({"mrope_section": []});
        section["rope_scaling"] = json!({"mrope_section": [1]});
        let error = CONTRACT
            .validate_config(&section, &path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("mrope_section"), "{error}");
        assert!(error.contains("rope_scaling.mrope_section"), "{error}");

        let mut interleaving: Value =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        interleaving["rope_parameters"] = json!({"mrope_interleaved": false});
        interleaving["rope_scaling"] = json!({"mrope_interleaved": true});
        let error = CONTRACT
            .validate_config(&interleaving, &path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("mrope_interleaved"), "{error}");
        assert!(error.contains("rope_scaling.mrope_interleaved"), "{error}");
    }

    #[test]
    fn root_and_text_config_bool_and_token_id_duplicates_must_match() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let path = temp.path().join("config.json");
        let mut nested: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        nested["eos_token_id"] = json!(2);
        let bool_conflict = json!({
            "attention_bias": true,
            "text_config": nested,
        });
        let error = CONTRACT
            .validate_config(&bool_conflict, &path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("attention_bias"), "{error}");
        assert!(error.contains("root=true"), "{error}");

        let mut nested: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        nested["eos_token_id"] = json!(2);
        let token_conflict = json!({
            "eos_token_id": 3,
            "text_config": nested,
        });
        let error = CONTRACT
            .validate_config(&token_conflict, &path)
            .unwrap_err()
            .to_string();
        assert!(error.contains("eos_token_id"), "{error}");
        assert!(error.contains("root=3"), "{error}");
    }

    #[test]
    fn optional_behavior_bool_allows_omission_but_rejects_every_authored_conflict() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let path = temp.path().join("config.json");
        let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        config.as_object_mut().unwrap().remove("attention_bias");
        let contract = EncoderContract {
            attention_bias: EncoderConfigBool::Optional(false),
            ..CONTRACT
        };
        contract
            .validate_config(&config, &path)
            .expect("omission selects the provider's fixed false runtime behavior");

        let mut root_conflict = config.clone();
        root_conflict["attention_bias"] = json!(true);
        let mut nested = config.clone();
        nested["attention_bias"] = json!(true);
        let nested_conflict = json!({ "text_config": nested });
        for authored in [root_conflict, nested_conflict] {
            let error = contract
                .validate_config(&authored, &path)
                .unwrap_err()
                .to_string();
            assert!(error.contains("attention_bias"), "{error}");
            assert!(error.contains("expected false"), "{error}");
        }
    }

    #[test]
    fn nested_text_architecture_is_authoritative_over_matching_root_wrapper() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let path = temp.path().join("config.json");
        let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let mut nested = config.clone();
        nested["model_type"] = json!("qwen2_5_vl_text");
        config["model_type"] = json!("qwen3");
        config["text_config"] = nested;
        std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();

        let error = CONTRACT
            .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("field architecture"), "{error}");
        assert!(error.contains("qwen2_5_vl_text"), "{error}");
    }

    #[test]
    fn dense_file_without_sibling_config_is_rejected() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        std::fs::remove_file(temp.path().join("config.json")).unwrap();
        let file = temp.path().join("model.safetensors");

        let error = CONTRACT
            .validate_source(&WeightsSource::File(file))
            .unwrap_err()
            .to_string();
        assert!(error.contains("has no config.json"), "{error}");
    }

    #[test]
    fn file_override_never_borrows_the_builtin_config_as_architecture_proof() {
        let temp = tempfile::tempdir().unwrap();
        let base = temp.path().join("base");
        let external = temp.path().join("external");
        write_fixture(&base.join("text_encoder"), 8);
        write_tokenizer_fixture(&base);
        write_fixture(&external, 8);
        std::fs::remove_file(external.join("config.json")).unwrap();
        let contract = EncoderContract {
            architecture: "qwen3_vl_text",
            ..CONTRACT
        };
        let base_config = base.join("text_encoder/config.json");
        let mut config: Value =
            serde_json::from_slice(&std::fs::read(&base_config).unwrap()).unwrap();
        config["model_type"] = json!("qwen3_vl_text");
        std::fs::write(&base_config, serde_json::to_vec(&config).unwrap()).unwrap();
        let spec = crate::LoadSpec::new(WeightsSource::Dir(base.clone()))
            .with_text_encoder(WeightsSource::File(external.join("model.safetensors")));

        let error = contract
            .source_for_load(&spec, &base)
            .unwrap_err()
            .to_string();
        assert!(error.contains("has no config.json"), "{error}");
        assert!(error.contains("cannot be proven"), "{error}");
    }

    fn write_packed_fixture(root: &Path, q_width: usize, include_q_biases: bool) {
        std::fs::create_dir_all(root).unwrap();
        std::fs::write(
            root.join("config.json"),
            br#"{"model_type":"qwen3","hidden_size":64,"intermediate_size":96,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,"vocab_size":16,"hidden_act":"silu","attention_dropout":0.0,"rms_norm_eps":0.000001,"rope_theta":1000000,"max_position_embeddings":4096,"attention_bias":false,"tie_word_embeddings":true,"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        let mut entries = vec![("model.embed_tokens.weight".to_owned(), "F16", vec![16, 64])];
        for layer in 0..2 {
            let base = format!("model.layers.{layer}");
            let packed = |name: String, output: usize, input: usize| {
                vec![
                    (
                        format!("{name}.weight"),
                        "U32",
                        vec![output, input * 4 / 32],
                    ),
                    (format!("{name}.scales"), "F16", vec![output, input / 64]),
                    (format!("{name}.biases"), "F16", vec![output, input / 64]),
                ]
            };
            let mut q = packed(format!("{base}.self_attn.q_proj"), 64, 64);
            if layer == 0 {
                q[0].2[1] = q_width;
                if !include_q_biases {
                    q.retain(|(name, _, _)| !name.ends_with(".biases"));
                }
            }
            entries.extend(q);
            entries.extend(packed(format!("{base}.self_attn.k_proj"), 32, 64));
            entries.extend(packed(format!("{base}.self_attn.v_proj"), 32, 64));
            entries.extend(packed(format!("{base}.self_attn.o_proj"), 64, 64));
            entries.extend(packed(format!("{base}.mlp.gate_proj"), 96, 64));
            entries.extend(packed(format!("{base}.mlp.up_proj"), 96, 64));
            entries.extend(packed(format!("{base}.mlp.down_proj"), 64, 96));
            entries.extend([
                (format!("{base}.self_attn.q_norm.weight"), "F16", vec![32]),
                (format!("{base}.self_attn.k_norm.weight"), "F16", vec![32]),
                (format!("{base}.input_layernorm.weight"), "F16", vec![64]),
                (
                    format!("{base}.post_attention_layernorm.weight"),
                    "F16",
                    vec![64],
                ),
            ]);
        }
        entries.push(("model.norm.weight".to_owned(), "F16", vec![64]));
        let mut offset = 0usize;
        let mut header = serde_json::Map::new();
        for (name, dtype, shape) in entries {
            let element_bytes = if matches!(dtype, "F16" | "BF16") {
                2
            } else {
                4
            };
            let bytes = shape.iter().product::<usize>() * element_bytes;
            header.insert(
                name,
                json!({"dtype":dtype, "shape":shape, "data_offsets":[offset, offset + bytes]}),
            );
            offset += bytes;
        }
        let encoded = serde_json::to_vec(&header).unwrap();
        let mut file = Vec::with_capacity(8 + encoded.len() + offset);
        file.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
        file.extend_from_slice(&encoded);
        file.resize(8 + encoded.len() + offset, 0);
        std::fs::write(root.join("model.safetensors"), file).unwrap();
    }

    const PACKED_CONTRACT: EncoderContract = EncoderContract {
        architecture: "qwen3",
        hidden_size: 64,
        intermediate_size: 96,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 32,
        vocab_size: 16,
        output_width: 64,
        loaded_hidden_layers: 2,
        requires_final_norm: true,
        requires_lm_head: false,
        hidden_activation: "silu",
        attention_dropout: EncoderConfigFloat::new(0.0),
        rms_norm_eps: EncoderConfigFloat::new(1e-6),
        qk_norm_eps: Some(EncoderConfigFloat::new(1e-6)),
        rope_theta: EncoderConfigFloat::new(1_000_000.0),
        max_position_embeddings: 4_096,
        attention_bias: EncoderConfigBool::Required(false),
        tie_word_embeddings: EncoderConfigBool::Required(true),
        tokenizer: TEST_TOKENIZER,
        prompt_executions: TEST_PROMPTS,
        bos_token_id: None,
        eos_token_id: None,
        image_token_id: None,
        vision_start_token_id: None,
        vision_end_token_id: None,
        mrope_section: &[],
        mrope_interleaved: None,
        selected_hidden_layers: &[2],
        packing: Some(EncoderPackingContract {
            group_size: 64,
            pack_embedding: false,
            pack_lm_head: false,
            supports_file: true,
        }),
        dense_storage_dtype_probe: Some("model.layers.0.input_layernorm.weight"),
    };

    #[test]
    fn packed_contract_rejects_corrupt_second_dimension_and_incomplete_triple() {
        let corrupt = tempfile::tempdir().unwrap();
        write_packed_fixture(corrupt.path(), 9, true);
        let error = PACKED_CONTRACT
            .validate_source(&WeightsSource::Dir(corrupt.path().to_path_buf()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("field packed_weight_shape"), "{error}");
        assert!(error.contains("expected [64, 8], got [64, 9]"), "{error}");

        let incomplete = tempfile::tempdir().unwrap();
        write_packed_fixture(incomplete.path(), 8, false);
        let error = PACKED_CONTRACT
            .validate_source(&WeightsSource::Dir(incomplete.path().to_path_buf()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("field packed_components"), "{error}");
    }

    #[test]
    fn packed_file_without_sibling_metadata_is_unprovable_and_rejected() {
        let temp = tempfile::tempdir().unwrap();
        write_packed_fixture(temp.path(), 8, true);
        std::fs::remove_file(temp.path().join("config.json")).unwrap();
        let error = PACKED_CONTRACT
            .validate_source(&WeightsSource::File(temp.path().join("model.safetensors")))
            .unwrap_err()
            .to_string();
        assert!(error.contains("has no config.json"), "{error}");
        assert!(
            error.contains("exact behavior, tokenizer, head-topology, and precision compatibility"),
            "{error}"
        );
    }

    #[test]
    fn validated_file_rejects_replacement_before_payload_open() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let file = temp.path().join("model.safetensors");
        let validated = CONTRACT
            .validate_source(&WeightsSource::File(file.clone()))
            .unwrap();
        let replacement = temp.path().join("replacement.safetensors");
        std::fs::copy(&file, &replacement).unwrap();
        std::fs::rename(&replacement, &file).unwrap();

        let error = validated
            .read_unchanged::<(), Error>(|_| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("pinned weights"), "{error}");
    }

    #[test]
    fn validated_directory_rejects_shard_inventory_and_config_mutation() {
        let added = tempfile::tempdir().unwrap();
        write_fixture(added.path(), 8);
        let validated = CONTRACT
            .validate_source(&WeightsSource::Dir(added.path().to_path_buf()))
            .unwrap();
        std::fs::copy(
            added.path().join("model.safetensors"),
            added.path().join("model-00002-of-00002.safetensors"),
        )
        .unwrap();
        let error = validated
            .read_unchanged::<(), Error>(|_| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("shard inventory changed"), "{error}");

        let config = tempfile::tempdir().unwrap();
        write_fixture(config.path(), 8);
        let validated = CONTRACT
            .validate_source(&WeightsSource::Dir(config.path().to_path_buf()))
            .unwrap();
        rewrite_config_field(config.path(), "vocab_size", 17);
        let error = validated
            .read_unchanged::<(), Error>(|_| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("pinned weights"), "{error}");
    }

    #[test]
    fn read_unchanged_brackets_backend_directory_enumeration() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let validated = CONTRACT
            .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
            .unwrap();
        let root = temp.path().to_path_buf();
        let error = validated
            .read_unchanged::<(), Error>(|_| {
                std::fs::copy(
                    root.join("model.safetensors"),
                    root.join("late-added.safetensors"),
                )
                .map_err(Error::from)?;
                Ok(())
            })
            .unwrap_err()
            .to_string();
        assert!(error.contains("shard inventory changed"), "{error}");
    }

    #[test]
    fn retained_quant_evidence_rejects_post_validation_config_mutation() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let validated = CONTRACT
            .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
            .unwrap();
        rewrite_config_quantization(temp.path(), 4);

        let error = validated
            .load_time_quant_bits(Some(4), "fixture")
            .unwrap_err()
            .to_string();
        assert!(error.contains("pinned weights"), "{error}");
    }

    #[test]
    fn selected_encoder_bytes_equal_the_direct_loader_shards_only() {
        let temp = tempfile::tempdir().unwrap();
        write_fixture(temp.path(), 8);
        let direct = temp.path().join("model.safetensors");
        let direct_bytes = std::fs::metadata(&direct).unwrap().len();
        let nested = temp.path().join("ignored/nested.safetensors");
        std::fs::create_dir_all(nested.parent().unwrap()).unwrap();
        std::fs::copy(&direct, &nested).unwrap();

        let validated = CONTRACT
            .validate_source(&WeightsSource::Dir(temp.path().to_path_buf()))
            .unwrap();
        assert_eq!(validated.source_bytes().unwrap(), direct_bytes);
        assert_eq!(
            text_encoder_source_bytes(&WeightsSource::Dir(temp.path().to_path_buf())).unwrap(),
            direct_bytes
        );
    }
}
