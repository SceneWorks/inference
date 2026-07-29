//! Tensor-neutral image-memory strategy and provider contract (SC-15449).
//!
//! This module is the inference-side authority for what an image provider can do. Providers
//! describe structure, lifecycle hooks, formula shape, backend realization, asset facts, and the
//! calibration ABI/fingerprint. Measured coefficients and envelopes remain external evidence, and
//! the caller remains the sole owner of live-budget accounting and least-cost selection.
//!
//! The separation is deliberate: a provider safety gate may reject a shared selection, but the
//! return type cannot substitute a different strategy or numeric tier. Unknown, stale,
//! fingerprint-mismatched, or out-of-envelope evidence is therefore never turned into a claimed
//! optimized fit by provider-local policy.

use crate::{Error, GenerationRequest, Precision, Quant, Result};

/// Current ABI of the provider/evidence calibration handshake.
///
/// Providers also supply a content fingerprint. The ABI changes when this contract changes how
/// formula inputs or lifecycle semantics are interpreted; the fingerprint changes whenever one
/// provider changes tensor layout, quantization floors, execution structure, or another detail that
/// invalidates its measurements.
pub const IMAGE_MEMORY_CALIBRATION_ABI: u32 = 1;

/// The normative least-cost image-memory ladder, in selection order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ImageMemoryStrategy {
    Resident = 0,
    StagedResidency = 1,
    BoundedDecode = 2,
    BoundedAttention = 3,
    BoundedTransformerResidency = 4,
}

impl ImageMemoryStrategy {
    pub const ALL: [Self; 5] = [
        Self::Resident,
        Self::StagedResidency,
        Self::BoundedDecode,
        Self::BoundedAttention,
        Self::BoundedTransformerResidency,
    ];

    /// `true` for a constrained strategy. The resident baseline is not an optimized fit.
    pub const fn is_optimized(self) -> bool {
        !matches!(self, Self::Resident)
    }
}

/// Static provider disposition for one rung. Dynamic verification never belongs here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageMemoryStrategySupport {
    Implemented,
    /// The architecture lacks the component or independent work the rung would optimize.
    StructurallyNotApplicable {
        reason: String,
    },
    Missing,
}

/// Provider-declared support and parameter domain for one ladder rung.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageMemoryStrategyCapability {
    pub strategy: ImageMemoryStrategy,
    pub support: ImageMemoryStrategySupport,
    /// Production parameter candidates accepted by the provider. Empty fields mean the strategy
    /// does not use that parameter; evidence covers only the exact values it exercised.
    pub parameters: ImageMemoryParameterRanges,
}

/// Which transformer(s) rung 4's block window applies to (SC-15794).
///
/// Rung 4 is defined as *"only an active transformer block or bounded block window is
/// wired/materialized at once"* — which says nothing about **which** transformer. It was implemented
/// for the DiT by convention, not by definition, and a text encoder is a transformer. This is a
/// **scope on rung 4, not a new rung**: a fifth rung would cost new conformance cells across every
/// catalog entry, new contract vocabulary, and an incoherent cost ordering (there is no sensible
/// request that wants rung 5 without rung 4).
///
/// Ordered cheapest-scope-first so a selector can widen the scope monotonically.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TransformerComponent {
    /// The denoising transformer only — rung 4's original by-convention scope, and the default so an
    /// existing provider's behaviour is unchanged until it declares otherwise.
    #[default]
    Dit,
    /// The text encoder only. Worth scoping separately because the encoder re-materializes **once per
    /// generation** while the DiT re-materializes once per step, so the two have very different
    /// cost sides for the same mechanism.
    TextEncoder,
    /// Both transformers stream.
    Both,
}

impl TransformerComponent {
    /// Whether this scope streams the denoising transformer.
    pub const fn includes_dit(self) -> bool {
        matches!(self, Self::Dit | Self::Both)
    }

    /// Whether this scope streams the text encoder.
    pub const fn includes_text_encoder(self) -> bool {
        matches!(self, Self::TextEncoder | Self::Both)
    }
}

/// Production parameter domains. The values are candidates, not calibration evidence.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImageMemoryParameterRanges {
    pub decode_tile_edges: Vec<u32>,
    pub decode_overlaps: Vec<u32>,
    pub attention_chunk_sizes: Vec<u32>,
    pub transformer_window_sizes: Vec<u32>,
    /// Component scopes the provider actually implements for rung 4 (SC-15794). Empty means the
    /// provider streams only the DiT, which is how every pre-SC-15794 provider behaves.
    pub transformer_window_components: Vec<TransformerComponent>,
}

/// Concrete parameters selected for one request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImageMemoryStrategyParameters {
    pub decode_tile_edge: Option<u32>,
    pub decode_overlap: Option<u32>,
    pub attention_chunk_size: Option<u32>,
    pub transformer_window_size: Option<u32>,
    /// Which transformer(s) the window applies to. `None` ⇒ [`TransformerComponent::Dit`], so a
    /// selection written before SC-15794 keeps its exact previous meaning.
    pub transformer_window_component: Option<TransformerComponent>,
}

impl ImageMemoryStrategyParameters {
    /// The effective rung-4 component scope: the declared one, or the DiT-only default.
    pub fn window_component(&self) -> TransformerComponent {
        self.transformer_window_component.unwrap_or_default()
    }
}

/// Provider lifecycle phases whose scarce backend residency may be separated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageMemoryPhase {
    Conditioning,
    Denoise,
    Decode,
}

/// Lifecycle and bounded-work hooks implemented by a provider.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImageMemoryLifecycleCapabilities {
    pub phases: Vec<ImageMemoryPhase>,
    /// Completed phases are synchronized before their scarce residency is released.
    pub synchronized_phase_release: bool,
    pub decode_tiling: bool,
    pub attention_chunking: bool,
    pub transformer_window_materialization: bool,
}

/// Formula inputs whose coefficients live in manifest/generated evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageMemoryFormulaVariable {
    AssetBytes,
    PixelCount,
    LatentPixelCount,
    BatchCount,
    FrameCount,
    ConditioningTokenCount,
    OverlayBytes,
    DecodeTileArea,
    AttentionChunkSize,
    TransformerWindowSize,
}

/// Provider-owned shape of the peak-memory formula. Coefficients do not live here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageMemoryFormulaKind {
    /// Generic MLX-compatible seam: materialized asset bytes plus calibrated headroom.
    AssetBytesPlusHeadroom,
    /// One affine expression over the declared request/strategy variables.
    Affine {
        variables: Vec<ImageMemoryFormulaVariable>,
    },
    /// Maximum of independently calibrated lifecycle phase expressions (the generalized Krea
    /// phase-curve shape, also suitable for request-aware provider estimators such as Mage).
    PhaseEnvelope {
        phases: Vec<ImageMemoryPhase>,
        variables: Vec<ImageMemoryFormulaVariable>,
    },
}

/// Backend-specific realization without imposing CUDA transfer language on unified memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageMemoryBackendRealization {
    CandleCuda {
        device_residency: bool,
        host_backed_weights: bool,
        host_to_device_block_materialization: bool,
    },
    MlxMetal {
        bounded_wired_residency: bool,
        lazy_or_mmap_materialization: bool,
        explicit_evaluation_and_synchronization: bool,
        cache_eviction: bool,
    },
}

impl ImageMemoryBackendRealization {
    pub const fn backend_id(&self) -> &'static str {
        match self {
            Self::CandleCuda { .. } => "candle",
            Self::MlxMetal { .. } => "mlx",
        }
    }
}

/// Provider-owned calibration identity.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageMemoryCalibrationIdentity {
    pub abi: u32,
    pub fingerprint: String,
}

impl ImageMemoryCalibrationIdentity {
    pub fn new(fingerprint: impl Into<String>) -> Self {
        Self {
            abi: IMAGE_MEMORY_CALIBRATION_ABI,
            fingerprint: fingerprint.into(),
        }
    }
}

/// Provider-owned, load-exact asset facts used as formula inputs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImageMemoryAssetFacts {
    pub base_bytes: u64,
    pub conditioning_bytes: u64,
    pub transformer_bytes: u64,
    pub decoder_bytes: u64,
    pub overlay_bytes: u64,
}

/// Cache keys must include every axis that can change residency or execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageMemoryCacheSemantics {
    StrategyTierParametersGeometryAndOverlay,
}

/// Warm generators must not inherit a prior request's memory decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageMemoryWarmRunSemantics {
    RevalidateBudgetAndReapplyRequestState,
}

/// Cancellation and error cleanup must synchronize backend work before releasing active state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageMemoryCleanupSemantics {
    SynchronizeAndReleaseActivePhasesAndWindows,
}

/// Runtime semantics required of every adopting provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageMemoryRuntimeSemantics {
    pub cache: ImageMemoryCacheSemantics,
    pub warm_run: ImageMemoryWarmRunSemantics,
    pub cancellation: ImageMemoryCleanupSemantics,
    pub error: ImageMemoryCleanupSemantics,
}

impl Default for ImageMemoryRuntimeSemantics {
    fn default() -> Self {
        Self {
            cache: ImageMemoryCacheSemantics::StrategyTierParametersGeometryAndOverlay,
            warm_run: ImageMemoryWarmRunSemantics::RevalidateBudgetAndReapplyRequestState,
            cancellation: ImageMemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows,
            error: ImageMemoryCleanupSemantics::SynchronizeAndReleaseActivePhasesAndWindows,
        }
    }
}

/// Static provider contract returned before weights are loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageMemoryProviderContract {
    pub provider_id: String,
    pub backend: ImageMemoryBackendRealization,
    pub strategies: Vec<ImageMemoryStrategyCapability>,
    pub lifecycle: ImageMemoryLifecycleCapabilities,
    pub formula: ImageMemoryFormulaKind,
    /// `None` is the compatibility default for a provider that has not adopted calibration yet.
    /// Such a provider can run its resident path but can never claim a verified optimized fit.
    pub calibration: Option<ImageMemoryCalibrationIdentity>,
    pub asset_facts: ImageMemoryAssetFacts,
    pub runtime: ImageMemoryRuntimeSemantics,
}

impl ImageMemoryProviderContract {
    /// Safe compatibility view for an existing provider: resident only, optimized rungs missing,
    /// and no calibration identity. This preserves existing behavior without fabricating evidence.
    pub fn compatibility_default(
        provider_id: impl Into<String>,
        backend: ImageMemoryBackendRealization,
    ) -> Self {
        Self {
            provider_id: provider_id.into(),
            backend,
            strategies: ImageMemoryStrategy::ALL
                .into_iter()
                .map(|strategy| ImageMemoryStrategyCapability {
                    strategy,
                    support: if strategy == ImageMemoryStrategy::Resident {
                        ImageMemoryStrategySupport::Implemented
                    } else {
                        ImageMemoryStrategySupport::Missing
                    },
                    parameters: ImageMemoryParameterRanges::default(),
                })
                .collect(),
            lifecycle: ImageMemoryLifecycleCapabilities::default(),
            formula: ImageMemoryFormulaKind::AssetBytesPlusHeadroom,
            calibration: None,
            asset_facts: ImageMemoryAssetFacts::default(),
            runtime: ImageMemoryRuntimeSemantics::default(),
        }
    }

    pub fn capability(
        &self,
        strategy: ImageMemoryStrategy,
    ) -> Option<&ImageMemoryStrategyCapability> {
        self.strategies
            .iter()
            .find(|capability| capability.strategy == strategy)
    }

    /// Static conformance errors. An empty result means the provider declaration is internally
    /// coherent; it does not mean measurements are verified.
    pub fn conformance_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.provider_id.trim().is_empty() {
            errors.push("provider_id must be non-empty".to_owned());
        }

        for strategy in ImageMemoryStrategy::ALL {
            let count = self
                .strategies
                .iter()
                .filter(|capability| capability.strategy == strategy)
                .count();
            if count != 1 {
                errors.push(format!(
                    "strategy {strategy:?} must appear exactly once (found {count})"
                ));
            }
        }
        if self.strategies.len() != ImageMemoryStrategy::ALL.len() {
            errors.push(format!(
                "strategy table must contain exactly five entries (found {})",
                self.strategies.len()
            ));
        }

        let implemented = |strategy| {
            matches!(
                self.capability(strategy).map(|c| &c.support),
                Some(ImageMemoryStrategySupport::Implemented)
            )
        };
        if implemented(ImageMemoryStrategy::StagedResidency) {
            for phase in [
                ImageMemoryPhase::Conditioning,
                ImageMemoryPhase::Denoise,
                ImageMemoryPhase::Decode,
            ] {
                if !self.lifecycle.phases.contains(&phase) {
                    errors.push(format!(
                        "StagedResidency requires a {phase:?} lifecycle phase"
                    ));
                }
            }
            if !self.lifecycle.synchronized_phase_release {
                errors.push(
                    "StagedResidency requires synchronized release of completed phases".to_owned(),
                );
            }
        }
        if implemented(ImageMemoryStrategy::BoundedDecode) && !self.lifecycle.decode_tiling {
            errors.push("BoundedDecode requires the decode_tiling hook".to_owned());
        }
        if implemented(ImageMemoryStrategy::BoundedAttention) && !self.lifecycle.attention_chunking
        {
            errors.push("BoundedAttention requires the attention_chunking hook".to_owned());
        }
        if implemented(ImageMemoryStrategy::BoundedTransformerResidency)
            && !self.lifecycle.transformer_window_materialization
        {
            errors.push(
                "BoundedTransformerResidency requires the transformer-window hook".to_owned(),
            );
        }

        for capability in &self.strategies {
            if let ImageMemoryStrategySupport::StructurallyNotApplicable { reason } =
                &capability.support
            {
                if reason.trim().is_empty() {
                    errors.push(format!(
                        "{:?} is StructurallyNotApplicable without a reason",
                        capability.strategy
                    ));
                }
            }
            validate_ranges(capability, &mut errors);
            validate_owned_parameter_domain(capability, &mut errors);
        }

        if let Some(calibration) = &self.calibration {
            if calibration.abi != IMAGE_MEMORY_CALIBRATION_ABI {
                errors.push(format!(
                    "calibration ABI {} does not match contract ABI {}",
                    calibration.abi, IMAGE_MEMORY_CALIBRATION_ABI
                ));
            }
            if calibration.fingerprint.trim().is_empty() {
                errors.push("calibration fingerprint must be non-empty".to_owned());
            }
        }

        if self.backend.backend_id().is_empty() {
            errors.push("backend realization must have an id".to_owned());
        }
        errors
    }

    /// Validate a worker-owned selection against static provider capability and parameter ranges.
    pub fn validate_selection(&self, selection: &ImageMemorySelection) -> Result<()> {
        let capability = self.capability(selection.strategy).ok_or_else(|| {
            Error::Unsupported(format!(
                "{} does not declare {:?}",
                self.provider_id, selection.strategy
            ))
        })?;
        if !matches!(capability.support, ImageMemoryStrategySupport::Implemented) {
            return Err(Error::Unsupported(format!(
                "{} cannot execute {:?}: {:?}",
                self.provider_id, selection.strategy, capability.support
            )));
        }
        for rung in ImageMemoryStrategy::ALL
            .into_iter()
            .filter(|rung| *rung < selection.strategy)
        {
            let support = self.capability(rung).map(|capability| &capability.support);
            if matches!(support, Some(ImageMemoryStrategySupport::Missing) | None) {
                return Err(Error::Unsupported(format!(
                    "{} cannot execute cumulative {:?}: prerequisite rung {rung:?} is missing",
                    self.provider_id, selection.strategy
                )));
            }
        }
        validate_selected_parameters(selection, self)
            .map_err(|message| Error::Unsupported(format!("{}: {message}", self.provider_id)))
    }
}

fn validate_ranges(capability: &ImageMemoryStrategyCapability, errors: &mut Vec<String>) {
    let ranges = &capability.parameters;
    for (name, values) in [
        ("decode_tile_edges", &ranges.decode_tile_edges),
        ("decode_overlaps", &ranges.decode_overlaps),
        ("attention_chunk_sizes", &ranges.attention_chunk_sizes),
        ("transformer_window_sizes", &ranges.transformer_window_sizes),
    ] {
        if values.contains(&0) {
            errors.push(format!("{:?} {name} contains zero", capability.strategy));
        }
        let mut unique = values.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != values.len() {
            errors.push(format!(
                "{:?} {name} contains duplicate candidates",
                capability.strategy
            ));
        }
    }
}

fn validate_owned_parameter_domain(
    capability: &ImageMemoryStrategyCapability,
    errors: &mut Vec<String>,
) {
    if !matches!(capability.support, ImageMemoryStrategySupport::Implemented) {
        return;
    }
    let ranges = &capability.parameters;
    let (decode, attention, transformer) = match capability.strategy {
        ImageMemoryStrategy::Resident | ImageMemoryStrategy::StagedResidency => {
            (false, false, false)
        }
        ImageMemoryStrategy::BoundedDecode => (true, false, false),
        ImageMemoryStrategy::BoundedAttention => (false, true, false),
        ImageMemoryStrategy::BoundedTransformerResidency => (false, false, true),
    };
    if decode && (ranges.decode_tile_edges.is_empty() || ranges.decode_overlaps.is_empty()) {
        errors.push(
            "BoundedDecode requires non-empty decode tile-edge and overlap candidates".to_owned(),
        );
    }
    if attention && ranges.attention_chunk_sizes.is_empty() {
        errors
            .push("BoundedAttention requires non-empty attention chunk-size candidates".to_owned());
    }
    if transformer && ranges.transformer_window_sizes.is_empty() {
        errors.push(
            "BoundedTransformerResidency requires non-empty transformer window candidates"
                .to_owned(),
        );
    }
    if !decode && (!ranges.decode_tile_edges.is_empty() || !ranges.decode_overlaps.is_empty()) {
        errors.push(format!(
            "{:?} must not own decode parameter candidates",
            capability.strategy
        ));
    }
    if !attention && !ranges.attention_chunk_sizes.is_empty() {
        errors.push(format!(
            "{:?} must not own attention parameter candidates",
            capability.strategy
        ));
    }
    if !transformer && !ranges.transformer_window_sizes.is_empty() {
        errors.push(format!(
            "{:?} must not own transformer-window candidates",
            capability.strategy
        ));
    }
    // The component scope is owned by the same rung as the window it scopes (SC-15794). It is
    // deliberately allowed to be empty on an implementing provider — that reads as the pre-SC-15794
    // DiT-only behaviour rather than as an incomplete declaration, so existing providers stay valid.
    if !transformer && !ranges.transformer_window_components.is_empty() {
        errors.push(format!(
            "{:?} must not own transformer-window component candidates",
            capability.strategy
        ));
    }
    let components = &ranges.transformer_window_components;
    let mut unique = components.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != components.len() {
        errors.push(format!(
            "{:?} transformer_window_components contains duplicate candidates",
            capability.strategy
        ));
    }
}

fn validate_selected_parameters(
    selection: &ImageMemorySelection,
    contract: &ImageMemoryProviderContract,
) -> std::result::Result<(), String> {
    let selected = selection.parameters;
    let implemented = |strategy| {
        matches!(
            contract
                .capability(strategy)
                .map(|capability| &capability.support),
            Some(ImageMemoryStrategySupport::Implemented)
        )
    };
    let requires_decode = selection.strategy >= ImageMemoryStrategy::BoundedDecode
        && implemented(ImageMemoryStrategy::BoundedDecode);
    let requires_attention = selection.strategy >= ImageMemoryStrategy::BoundedAttention
        && implemented(ImageMemoryStrategy::BoundedAttention);
    let requires_transformer = selection.strategy
        >= ImageMemoryStrategy::BoundedTransformerResidency
        && implemented(ImageMemoryStrategy::BoundedTransformerResidency);

    validate_required_parameter(
        "decode_tile_edge",
        selected.decode_tile_edge,
        requires_decode,
        contract
            .capability(ImageMemoryStrategy::BoundedDecode)
            .map(|capability| capability.parameters.decode_tile_edges.as_slice())
            .unwrap_or_default(),
    )?;
    validate_required_parameter(
        "decode_overlap",
        selected.decode_overlap,
        requires_decode,
        contract
            .capability(ImageMemoryStrategy::BoundedDecode)
            .map(|capability| capability.parameters.decode_overlaps.as_slice())
            .unwrap_or_default(),
    )?;
    validate_required_parameter(
        "attention_chunk_size",
        selected.attention_chunk_size,
        requires_attention,
        contract
            .capability(ImageMemoryStrategy::BoundedAttention)
            .map(|capability| capability.parameters.attention_chunk_sizes.as_slice())
            .unwrap_or_default(),
    )?;
    validate_required_parameter(
        "transformer_window_size",
        selected.transformer_window_size,
        requires_transformer,
        contract
            .capability(ImageMemoryStrategy::BoundedTransformerResidency)
            .map(|capability| capability.parameters.transformer_window_sizes.as_slice())
            .unwrap_or_default(),
    )?;
    // The component scope (SC-15794) is validated separately from the numeric parameters: unlike a
    // tile edge or a window size it has a meaningful DEFAULT (DiT-only), so `None` is legal at every
    // rung and only an explicitly-declared scope is checked. That keeps a pre-SC-15794 selection —
    // which cannot carry a component — valid rather than retroactively incomplete.
    if let Some(component) = selected.transformer_window_component {
        if !requires_transformer {
            return Err(format!(
                "transformer_window_component={component:?} is irrelevant below its owning strategy \
                 rung"
            ));
        }
        let allowed = contract
            .capability(ImageMemoryStrategy::BoundedTransformerResidency)
            .map(|capability| {
                capability
                    .parameters
                    .transformer_window_components
                    .as_slice()
            })
            .unwrap_or_default();
        // An empty declaration means the provider implements the DiT-only default and nothing else,
        // so that is the one scope a request may still name explicitly.
        let permitted = if allowed.is_empty() {
            component == TransformerComponent::Dit
        } else {
            allowed.contains(&component)
        };
        if !permitted {
            return Err(format!(
                "transformer_window_component={component:?} is outside the declared production \
                 candidates {allowed:?}"
            ));
        }
    }
    Ok(())
}

fn validate_required_parameter(
    name: &str,
    value: Option<u32>,
    required: bool,
    allowed: &[u32],
) -> std::result::Result<(), String> {
    match (required, value) {
        (true, None) => Err(format!(
            "{name} is required by the selected cumulative strategy"
        )),
        (false, Some(value)) => Err(format!(
            "{name}={value} is irrelevant below its owning strategy rung"
        )),
        (true, Some(value)) if !allowed.contains(&value) => Err(format!(
            "{name}={value} is outside the declared production candidates {allowed:?}"
        )),
        _ => Ok(()),
    }
}

/// Numeric tier is one immutable axis of a selection. Strategies cannot change it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageMemoryNumericTier {
    pub precision: Precision,
    pub quant: Option<Quant>,
}

/// Shared-worker selection presented to the provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageMemorySelection {
    pub strategy: ImageMemoryStrategy,
    pub parameters: ImageMemoryStrategyParameters,
    pub tier: ImageMemoryNumericTier,
}

/// Request geometry used by evidence and cache identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageMemoryGeometry {
    pub width: u32,
    pub height: u32,
    pub batch: u32,
    pub frames: u32,
}

/// Canonical live-budget accounting owned by the caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageMemoryBudget {
    pub total_bytes: u64,
    pub committed_bytes: u64,
    pub reclaimable_bytes: u64,
    pub reserved_headroom_bytes: u64,
}

impl ImageMemoryBudget {
    /// Effective request budget. Reserved headroom is removed from the currently free plus
    /// reclaimable memory, while the total-minus-headroom ceiling prevents reclaimable accounting
    /// from exceeding physical capacity. All arithmetic is saturating.
    pub fn effective_bytes(self) -> u64 {
        let ceiling = self
            .total_bytes
            .saturating_sub(self.reserved_headroom_bytes);
        self.total_bytes
            .saturating_sub(self.committed_bytes)
            .saturating_add(self.reclaimable_bytes)
            .saturating_sub(self.reserved_headroom_bytes)
            .min(ceiling)
    }

    /// Exact-boundary fits are accepted.
    pub fn fits(self, predicted_peak_bytes: u64) -> bool {
        predicted_peak_bytes <= self.effective_bytes()
    }
}

/// Cold/warm state is request-scoped and never implies reuse of a prior selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageMemoryCacheState {
    Cold,
    Warm,
}

/// Advertised request surface. Optimized evidence never transfers between these modes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageMemoryMode {
    TextToImage,
    ImageToImage,
    Edit,
    Other(String),
}

/// Context for provider safety and lifecycle hooks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageMemoryRunContext {
    pub selection: ImageMemorySelection,
    /// Provider calibration identity that the caller used for admission. Resident requests carry
    /// this handshake too; optimized-only evidence validation is not sufficient.
    pub calibration_abi: u32,
    pub calibration_fingerprint: String,
    pub mode: ImageMemoryMode,
    pub has_reference: bool,
    pub use_pid: bool,
    pub has_phases: bool,
    pub geometry: ImageMemoryGeometry,
    pub overlay: Option<String>,
    pub budget: ImageMemoryBudget,
    pub predicted_peak_bytes: u64,
    pub cache_state: ImageMemoryCacheState,
    pub evidence_revision: String,
}

/// Defense-in-depth result. It can accept or reject; it cannot replace the worker's selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageMemorySafetyDecision {
    Accept,
    Reject { reason: String },
}

/// Terminal cleanup reason supplied to a request scope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageMemoryRunOutcome {
    Complete,
    Canceled,
    Error { message: String },
}

/// Tensor-neutral lifecycle scope implemented by adopting providers.
pub trait ImageMemoryRequestScope {
    /// Translate the selected strategy into the provider's existing request controls. This is the
    /// executable bridge for providers that predate the shared contract.
    fn configure_request(&mut self, request: &mut GenerationRequest) -> Result<()>;
    fn enter_phase(&mut self, phase: ImageMemoryPhase) -> Result<()>;
    fn leave_phase(&mut self, phase: ImageMemoryPhase) -> Result<()>;
    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        geometry: ImageMemoryGeometry,
    ) -> Result<()>;
    fn configure_attention(&mut self, chunk_size: u32) -> Result<()>;
    fn materialize_transformer_window(&mut self, first_block: u32, block_count: u32) -> Result<()>;
    /// Must be called exactly once on success, cancellation, or error. Providers synchronize and
    /// release active phases/windows according to [`ImageMemoryRuntimeSemantics`].
    fn finish(&mut self, outcome: ImageMemoryRunOutcome) -> Result<()>;
}

/// Five catalog conformance states from epic SC-15448.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageMemoryConformanceState {
    Verified,
    ImplementedUnverified,
    StructurallyNotApplicable,
    Missing,
    RouteUnavailableOrBroken,
}

/// Six independent evidence dimensions from epic SC-15448.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ImageMemoryEvidenceDimension {
    StaticImplementation,
    DeclaredCalibration,
    HistoricalVerification,
    CurrentEnvironmentVerification,
    CanonicalRouteLoadability,
    ExactStrategyParameters,
}

impl ImageMemoryEvidenceDimension {
    pub const ALL: [Self; 6] = [
        Self::StaticImplementation,
        Self::DeclaredCalibration,
        Self::HistoricalVerification,
        Self::CurrentEnvironmentVerification,
        Self::CanonicalRouteLoadability,
        Self::ExactStrategyParameters,
    ];
}

/// Why one evidence dimension does or does not cover a request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageMemoryEvidenceVerdict {
    Satisfied,
    Missing,
    Unverified,
    Stale,
    FingerprintMismatch,
    OutOfEnvelope,
    Invalid,
}

/// Explicit six-dimensional evidence result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageMemoryEvidenceDimensions {
    pub static_implementation: ImageMemoryEvidenceVerdict,
    pub declared_calibration: ImageMemoryEvidenceVerdict,
    pub historical_verification: ImageMemoryEvidenceVerdict,
    pub current_environment_verification: ImageMemoryEvidenceVerdict,
    pub canonical_route_loadability: ImageMemoryEvidenceVerdict,
    pub exact_strategy_parameters: ImageMemoryEvidenceVerdict,
}

impl ImageMemoryEvidenceDimensions {
    pub const VERIFIED: Self = Self {
        static_implementation: ImageMemoryEvidenceVerdict::Satisfied,
        declared_calibration: ImageMemoryEvidenceVerdict::Satisfied,
        historical_verification: ImageMemoryEvidenceVerdict::Satisfied,
        current_environment_verification: ImageMemoryEvidenceVerdict::Satisfied,
        canonical_route_loadability: ImageMemoryEvidenceVerdict::Satisfied,
        exact_strategy_parameters: ImageMemoryEvidenceVerdict::Satisfied,
    };

    pub fn verdict(self, dimension: ImageMemoryEvidenceDimension) -> ImageMemoryEvidenceVerdict {
        match dimension {
            ImageMemoryEvidenceDimension::StaticImplementation => self.static_implementation,
            ImageMemoryEvidenceDimension::DeclaredCalibration => self.declared_calibration,
            ImageMemoryEvidenceDimension::HistoricalVerification => self.historical_verification,
            ImageMemoryEvidenceDimension::CurrentEnvironmentVerification => {
                self.current_environment_verification
            }
            ImageMemoryEvidenceDimension::CanonicalRouteLoadability => {
                self.canonical_route_loadability
            }
            ImageMemoryEvidenceDimension::ExactStrategyParameters => self.exact_strategy_parameters,
        }
    }

    pub fn all_satisfied(self) -> bool {
        ImageMemoryEvidenceDimension::ALL
            .into_iter()
            .all(|dimension| self.verdict(dimension) == ImageMemoryEvidenceVerdict::Satisfied)
    }
}

/// Fully-qualified key for one evidence cell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageMemoryEvidenceKey {
    pub resolved_route: String,
    pub backend: String,
    pub tier: ImageMemoryNumericTier,
    pub mode: String,
    pub overlay: Option<String>,
    pub geometry: ImageMemoryGeometry,
    pub strategy: ImageMemoryStrategy,
    pub parameters: ImageMemoryStrategyParameters,
}

/// Numerical parity contract for an evidence record.
#[derive(Clone, Debug, PartialEq)]
pub enum ImageMemoryParityContract {
    Exact,
    Tolerance {
        metric: String,
        maximum_error: f64,
    },
    Golden {
        fixture: String,
        metric: String,
        maximum_error: f64,
    },
}

/// Result of executing the declared numerical contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImageMemoryParityResult {
    Passed,
    Failed { reason: String },
    NotRun,
}

/// Dynamic/static evidence handshake consumed by the shared selector.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageMemoryEvidence {
    pub key: ImageMemoryEvidenceKey,
    pub conformance: ImageMemoryConformanceState,
    pub dimensions: ImageMemoryEvidenceDimensions,
    pub calibration_abi: u32,
    pub calibration_fingerprint: String,
    pub sceneworks_revision: String,
    pub inference_revision: String,
    pub harness_version: String,
    pub predicted_peak_bytes: u64,
    pub observed_peak_bytes: Option<u64>,
    pub parity: ImageMemoryParityContract,
    pub parity_result: ImageMemoryParityResult,
}

impl ImageMemoryEvidence {
    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.observed_peak_bytes.is_none() {
            errors.push("verified evidence requires an observed peak".to_owned());
        }
        match &self.parity {
            ImageMemoryParityContract::Exact => {}
            ImageMemoryParityContract::Tolerance {
                metric,
                maximum_error,
            } => validate_parity_limit(metric, *maximum_error, &mut errors),
            ImageMemoryParityContract::Golden {
                fixture,
                metric,
                maximum_error,
            } => {
                if fixture.trim().is_empty() {
                    errors.push("golden parity fixture must be non-empty".to_owned());
                }
                validate_parity_limit(metric, *maximum_error, &mut errors);
            }
        }
        match &self.parity_result {
            ImageMemoryParityResult::Passed => {}
            ImageMemoryParityResult::Failed { reason } if reason.trim().is_empty() => {
                errors.push("failed parity result requires a reason".to_owned());
            }
            ImageMemoryParityResult::Failed { .. } | ImageMemoryParityResult::NotRun => {}
        }
        errors
    }

    /// Only a fully verified, six-dimension record matching the current provider handshake may
    /// authorize an optimized fit.
    pub fn optimized_eligibility(
        &self,
        contract: &ImageMemoryProviderContract,
    ) -> std::result::Result<(), ImageMemoryEvidenceVerdict> {
        if !self.key.strategy.is_optimized() {
            return Ok(());
        }
        if self.conformance != ImageMemoryConformanceState::Verified {
            return Err(ImageMemoryEvidenceVerdict::Unverified);
        }
        if !self.validation_errors().is_empty() {
            return Err(ImageMemoryEvidenceVerdict::Invalid);
        }
        if self.parity_result != ImageMemoryParityResult::Passed {
            return Err(ImageMemoryEvidenceVerdict::Unverified);
        }
        for dimension in ImageMemoryEvidenceDimension::ALL {
            let verdict = self.dimensions.verdict(dimension);
            if verdict != ImageMemoryEvidenceVerdict::Satisfied {
                return Err(verdict);
            }
        }
        let Some(identity) = &contract.calibration else {
            return Err(ImageMemoryEvidenceVerdict::Unverified);
        };
        if self.calibration_abi != identity.abi
            || self.calibration_fingerprint != identity.fingerprint
        {
            return Err(ImageMemoryEvidenceVerdict::FingerprintMismatch);
        }
        Ok(())
    }
}

fn validate_parity_limit(metric: &str, maximum_error: f64, errors: &mut Vec<String>) {
    if metric.trim().is_empty() {
        errors.push("parity metric must be non-empty".to_owned());
    }
    if !maximum_error.is_finite() || maximum_error < 0.0 {
        errors.push("parity maximum_error must be finite and non-negative".to_owned());
    }
}

/// Truthful pre-OOM rejection payload. The caller may populate a measured smaller geometry; it must
/// not invent advice from unknown or unverified evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageMemoryRejection {
    pub available_bytes: u64,
    pub minimum_verified_peak_bytes: Option<u64>,
    pub smaller_verified_geometry: Option<ImageMemoryGeometry>,
    pub exclusion_reason: ImageMemoryEvidenceVerdict,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mlx_backend() -> ImageMemoryBackendRealization {
        ImageMemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        }
    }

    fn implemented(strategy: ImageMemoryStrategy) -> ImageMemoryStrategyCapability {
        ImageMemoryStrategyCapability {
            strategy,
            support: ImageMemoryStrategySupport::Implemented,
            parameters: ImageMemoryParameterRanges::default(),
        }
    }

    fn adopted_contract() -> ImageMemoryProviderContract {
        let mut strategies: Vec<_> = ImageMemoryStrategy::ALL
            .into_iter()
            .map(implemented)
            .collect();
        strategies[2].parameters.decode_tile_edges = vec![512, 384];
        strategies[2].parameters.decode_overlaps = vec![64];
        strategies[3].parameters.attention_chunk_sizes = vec![256, 128];
        strategies[4].parameters.transformer_window_sizes = vec![2, 1];
        ImageMemoryProviderContract {
            provider_id: "test-provider".to_owned(),
            backend: mlx_backend(),
            strategies,
            lifecycle: ImageMemoryLifecycleCapabilities {
                phases: vec![
                    ImageMemoryPhase::Conditioning,
                    ImageMemoryPhase::Denoise,
                    ImageMemoryPhase::Decode,
                ],
                synchronized_phase_release: true,
                decode_tiling: true,
                attention_chunking: true,
                transformer_window_materialization: true,
            },
            formula: ImageMemoryFormulaKind::PhaseEnvelope {
                phases: vec![
                    ImageMemoryPhase::Conditioning,
                    ImageMemoryPhase::Denoise,
                    ImageMemoryPhase::Decode,
                ],
                variables: vec![
                    ImageMemoryFormulaVariable::AssetBytes,
                    ImageMemoryFormulaVariable::PixelCount,
                ],
            },
            calibration: Some(ImageMemoryCalibrationIdentity::new("test-layout-v1")),
            asset_facts: ImageMemoryAssetFacts::default(),
            runtime: ImageMemoryRuntimeSemantics::default(),
        }
    }

    #[test]
    fn compatibility_default_never_advertises_an_optimized_rung() {
        let contract = ImageMemoryProviderContract::compatibility_default("legacy", mlx_backend());
        assert!(contract.conformance_errors().is_empty());
        assert!(contract.calibration.is_none());
        for strategy in ImageMemoryStrategy::ALL {
            let support = &contract.capability(strategy).unwrap().support;
            if strategy == ImageMemoryStrategy::Resident {
                assert_eq!(support, &ImageMemoryStrategySupport::Implemented);
            } else {
                assert_eq!(support, &ImageMemoryStrategySupport::Missing);
            }
        }
    }

    #[test]
    fn optimized_evidence_requires_every_dimension_and_matching_fingerprint() {
        let contract = adopted_contract();
        let mut evidence = ImageMemoryEvidence {
            key: ImageMemoryEvidenceKey {
                resolved_route: "test".to_owned(),
                backend: "mlx".to_owned(),
                tier: ImageMemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                },
                mode: "text_to_image".to_owned(),
                overlay: None,
                geometry: ImageMemoryGeometry {
                    width: 1024,
                    height: 1024,
                    batch: 1,
                    frames: 1,
                },
                strategy: ImageMemoryStrategy::BoundedDecode,
                parameters: ImageMemoryStrategyParameters {
                    decode_tile_edge: Some(512),
                    decode_overlap: Some(64),
                    ..Default::default()
                },
            },
            conformance: ImageMemoryConformanceState::Verified,
            dimensions: ImageMemoryEvidenceDimensions::VERIFIED,
            calibration_abi: IMAGE_MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: "test-layout-v1".to_owned(),
            sceneworks_revision: "scene".to_owned(),
            inference_revision: "inference".to_owned(),
            harness_version: "v1".to_owned(),
            predicted_peak_bytes: 10,
            observed_peak_bytes: Some(9),
            parity: ImageMemoryParityContract::Exact,
            parity_result: ImageMemoryParityResult::Passed,
        };
        assert_eq!(evidence.optimized_eligibility(&contract), Ok(()));

        evidence.dimensions.current_environment_verification = ImageMemoryEvidenceVerdict::Stale;
        assert_eq!(
            evidence.optimized_eligibility(&contract),
            Err(ImageMemoryEvidenceVerdict::Stale)
        );
        evidence.dimensions = ImageMemoryEvidenceDimensions::VERIFIED;
        evidence.calibration_fingerprint = "old-layout".to_owned();
        assert_eq!(
            evidence.optimized_eligibility(&contract),
            Err(ImageMemoryEvidenceVerdict::FingerprintMismatch)
        );
    }

    #[test]
    fn effective_budget_reserves_headroom_from_free_and_reclaimable_memory() {
        let budget = ImageMemoryBudget {
            total_bytes: 100,
            committed_bytes: 80,
            reclaimable_bytes: 50,
            reserved_headroom_bytes: 10,
        };
        assert_eq!(budget.effective_bytes(), 60);
        assert!(budget.fits(60));
        assert!(!budget.fits(61));

        assert_eq!(
            ImageMemoryBudget {
                reclaimable_bytes: 0,
                ..budget
            }
            .effective_bytes(),
            10
        );
    }

    #[test]
    fn effective_budget_saturates_at_zero_and_total_minus_headroom() {
        assert_eq!(
            ImageMemoryBudget {
                total_bytes: 100,
                committed_bytes: 110,
                reclaimable_bytes: 5,
                reserved_headroom_bytes: 10,
            }
            .effective_bytes(),
            0
        );
        assert_eq!(
            ImageMemoryBudget {
                total_bytes: 100,
                committed_bytes: 0,
                reclaimable_bytes: u64::MAX,
                reserved_headroom_bytes: 10,
            }
            .effective_bytes(),
            90
        );
        assert_eq!(
            ImageMemoryBudget {
                total_bytes: 100,
                committed_bytes: 0,
                reclaimable_bytes: u64::MAX,
                reserved_headroom_bytes: 101,
            }
            .effective_bytes(),
            0
        );
    }

    #[test]
    fn provider_contract_requires_hooks_for_implemented_rungs() {
        let mut contract = adopted_contract();
        contract.lifecycle.attention_chunking = false;
        assert!(contract
            .conformance_errors()
            .iter()
            .any(|error| error.contains("attention_chunking")));
    }

    #[test]
    fn selection_preserves_tier_and_rejects_unknown_parameter_values() {
        let contract = adopted_contract();
        let tier = ImageMemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
        };
        let mut selection = ImageMemorySelection {
            strategy: ImageMemoryStrategy::BoundedDecode,
            parameters: ImageMemoryStrategyParameters {
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                ..Default::default()
            },
            tier,
        };
        contract.validate_selection(&selection).unwrap();
        assert_eq!(selection.tier, tier);

        selection.parameters.decode_tile_edge = Some(320);
        assert!(matches!(
            contract.validate_selection(&selection),
            Err(Error::Unsupported(_))
        ));
    }

    #[test]
    fn cumulative_parameters_are_required_and_irrelevant_parameters_are_rejected() {
        let contract = adopted_contract();
        let tier = ImageMemoryNumericTier {
            precision: Precision::Bf16,
            quant: None,
        };
        let mut selection = ImageMemorySelection {
            strategy: ImageMemoryStrategy::BoundedAttention,
            parameters: ImageMemoryStrategyParameters {
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                attention_chunk_size: Some(128),
                transformer_window_size: None,
                transformer_window_component: None,
            },
            tier,
        };
        contract.validate_selection(&selection).unwrap();

        selection.parameters.decode_overlap = None;
        assert!(contract
            .validate_selection(&selection)
            .unwrap_err()
            .to_string()
            .contains("decode_overlap is required"));

        selection.strategy = ImageMemoryStrategy::Resident;
        selection.parameters = ImageMemoryStrategyParameters {
            attention_chunk_size: Some(128),
            ..Default::default()
        };
        assert!(contract
            .validate_selection(&selection)
            .unwrap_err()
            .to_string()
            .contains("irrelevant"));
    }

    #[test]
    fn optimized_evidence_requires_valid_passing_parity_and_an_observed_peak() {
        let contract = adopted_contract();
        let mut evidence = ImageMemoryEvidence {
            key: ImageMemoryEvidenceKey {
                resolved_route: "test".to_owned(),
                backend: "mlx".to_owned(),
                tier: ImageMemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: None,
                },
                mode: "text_to_image".to_owned(),
                overlay: None,
                geometry: ImageMemoryGeometry {
                    width: 512,
                    height: 512,
                    batch: 1,
                    frames: 1,
                },
                strategy: ImageMemoryStrategy::BoundedDecode,
                parameters: ImageMemoryStrategyParameters {
                    decode_tile_edge: Some(512),
                    decode_overlap: Some(64),
                    ..Default::default()
                },
            },
            conformance: ImageMemoryConformanceState::Verified,
            dimensions: ImageMemoryEvidenceDimensions::VERIFIED,
            calibration_abi: IMAGE_MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: "test-layout-v1".to_owned(),
            sceneworks_revision: "scene".to_owned(),
            inference_revision: "inference".to_owned(),
            harness_version: "v1".to_owned(),
            predicted_peak_bytes: 10,
            observed_peak_bytes: Some(9),
            parity: ImageMemoryParityContract::Tolerance {
                metric: "max_abs".to_owned(),
                maximum_error: 0.001,
            },
            parity_result: ImageMemoryParityResult::Passed,
        };
        assert_eq!(evidence.optimized_eligibility(&contract), Ok(()));

        evidence.parity = ImageMemoryParityContract::Tolerance {
            metric: String::new(),
            maximum_error: f64::NAN,
        };
        assert_eq!(
            evidence.optimized_eligibility(&contract),
            Err(ImageMemoryEvidenceVerdict::Invalid)
        );
        evidence.parity = ImageMemoryParityContract::Golden {
            fixture: String::new(),
            metric: "max_abs".to_owned(),
            maximum_error: -1.0,
        };
        assert_eq!(
            evidence.optimized_eligibility(&contract),
            Err(ImageMemoryEvidenceVerdict::Invalid)
        );
        evidence.parity = ImageMemoryParityContract::Exact;
        evidence.observed_peak_bytes = None;
        assert_eq!(
            evidence.optimized_eligibility(&contract),
            Err(ImageMemoryEvidenceVerdict::Invalid)
        );
        evidence.observed_peak_bytes = Some(9);
        evidence.parity_result = ImageMemoryParityResult::NotRun;
        assert_eq!(
            evidence.optimized_eligibility(&contract),
            Err(ImageMemoryEvidenceVerdict::Unverified)
        );
    }
    /// SC-15794: the component scope's accessors and its DiT-only default.
    ///
    /// The default is the whole backward-compatibility story — every provider and every selection
    /// written before this type existed must keep meaning "stream the DiT", so `None` resolving to
    /// anything else would silently re-scope work already in production.
    #[test]
    fn the_transformer_component_defaults_to_dit_and_reports_its_membership() {
        assert_eq!(TransformerComponent::default(), TransformerComponent::Dit);
        assert_eq!(
            ImageMemoryStrategyParameters::default().window_component(),
            TransformerComponent::Dit
        );
        let explicit = ImageMemoryStrategyParameters {
            transformer_window_component: Some(TransformerComponent::TextEncoder),
            ..Default::default()
        };
        assert_eq!(
            explicit.window_component(),
            TransformerComponent::TextEncoder
        );

        // Membership, asserted per variant rather than via a loop so a swapped pair cannot pass.
        assert!(TransformerComponent::Dit.includes_dit());
        assert!(!TransformerComponent::Dit.includes_text_encoder());
        assert!(!TransformerComponent::TextEncoder.includes_dit());
        assert!(TransformerComponent::TextEncoder.includes_text_encoder());
        assert!(TransformerComponent::Both.includes_dit());
        assert!(TransformerComponent::Both.includes_text_encoder());
    }

    /// The selected scope is validated against what the provider actually declared — the check that
    /// stops a selector asking for an encoder stream from a provider that only implements the DiT.
    #[test]
    fn a_selected_component_must_be_one_the_provider_declared() {
        let contract = |components: Vec<TransformerComponent>| {
            let mut c = adopted_contract();
            let rung4 = c
                .strategies
                .iter_mut()
                .find(|s| s.strategy == ImageMemoryStrategy::BoundedTransformerResidency)
                .expect("rung 4 capability");
            rung4.parameters.transformer_window_sizes = vec![1];
            rung4.parameters.transformer_window_components = components;
            c
        };
        let select = |component| ImageMemorySelection {
            strategy: ImageMemoryStrategy::BoundedTransformerResidency,
            parameters: ImageMemoryStrategyParameters {
                // Rung 4 is cumulative, so the lower rungs' parameters are required too; these are
                // `adopted_contract`'s own declared candidates.
                decode_tile_edge: Some(512),
                decode_overlap: Some(64),
                attention_chunk_size: Some(256),
                transformer_window_size: Some(1),
                transformer_window_component: Some(component),
            },
            tier: ImageMemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
            },
        };

        let dit_only = contract(vec![TransformerComponent::Dit]);
        assert!(
            validate_selected_parameters(&select(TransformerComponent::Dit), &dit_only).is_ok()
        );
        let err =
            validate_selected_parameters(&select(TransformerComponent::TextEncoder), &dit_only)
                .expect_err(
                    "a provider that declares only Dit must reject a TextEncoder selection",
                );
        assert!(err.contains("TextEncoder"), "{err}");

        // An empty declaration means DiT-only, so DiT is still selectable and nothing else is.
        let empty = contract(Vec::new());
        assert!(validate_selected_parameters(&select(TransformerComponent::Dit), &empty).is_ok());
        assert!(validate_selected_parameters(&select(TransformerComponent::Both), &empty).is_err());

        // A provider that declares the encoder scope accepts it.
        let both = contract(vec![TransformerComponent::Dit, TransformerComponent::Both]);
        assert!(validate_selected_parameters(&select(TransformerComponent::Both), &both).is_ok());
    }

    /// A provider may only own component candidates on the rung that owns the window they scope.
    #[test]
    fn only_rung_four_may_own_component_candidates() {
        let with_components = |strategy| ImageMemoryStrategyCapability {
            strategy,
            support: ImageMemoryStrategySupport::Implemented,
            parameters: ImageMemoryParameterRanges {
                transformer_window_components: vec![TransformerComponent::Both],
                ..Default::default()
            },
        };
        let mut errors = Vec::new();
        validate_owned_parameter_domain(
            &with_components(ImageMemoryStrategy::BoundedDecode),
            &mut errors,
        );
        assert!(
            errors.iter().any(|e| e.contains("component candidates")),
            "a non-rung-4 strategy owning component candidates must be rejected, got {errors:?}"
        );

        // Duplicates are a declaration bug the selector would silently tolerate.
        let mut errors = Vec::new();
        validate_owned_parameter_domain(
            &ImageMemoryStrategyCapability {
                strategy: ImageMemoryStrategy::BoundedTransformerResidency,
                support: ImageMemoryStrategySupport::Implemented,
                parameters: ImageMemoryParameterRanges {
                    transformer_window_sizes: vec![1],
                    transformer_window_components: vec![
                        TransformerComponent::Dit,
                        TransformerComponent::Dit,
                    ],
                    ..Default::default()
                },
            },
            &mut errors,
        );
        assert!(
            errors.iter().any(|e| e.contains("duplicate")),
            "duplicate component candidates must be rejected, got {errors:?}"
        );

        // An EMPTY declaration is legal and means DiT-only — that is what keeps every pre-SC-15794
        // provider valid rather than retroactively incomplete.
        let mut errors = Vec::new();
        validate_owned_parameter_domain(
            &ImageMemoryStrategyCapability {
                strategy: ImageMemoryStrategy::BoundedTransformerResidency,
                support: ImageMemoryStrategySupport::Implemented,
                parameters: ImageMemoryParameterRanges {
                    transformer_window_sizes: vec![1],
                    ..Default::default()
                },
            },
            &mut errors,
        );
        assert!(
            errors.is_empty(),
            "an empty component declaration must be legal, got {errors:?}"
        );
    }
}
