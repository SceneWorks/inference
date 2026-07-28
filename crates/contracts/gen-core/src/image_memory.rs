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

use crate::{Error, Precision, Quant, Result};

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

/// Production parameter domains. The values are candidates, not calibration evidence.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ImageMemoryParameterRanges {
    pub decode_tile_edges: Vec<u32>,
    pub decode_overlaps: Vec<u32>,
    pub attention_chunk_sizes: Vec<u32>,
    pub transformer_window_sizes: Vec<u32>,
}

/// Concrete parameters selected for one request.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImageMemoryStrategyParameters {
    pub decode_tile_edge: Option<u32>,
    pub decode_overlap: Option<u32>,
    pub attention_chunk_size: Option<u32>,
    pub transformer_window_size: Option<u32>,
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
        validate_selected_parameters(selection, capability)
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

fn validate_selected_parameters(
    selection: &ImageMemorySelection,
    capability: &ImageMemoryStrategyCapability,
) -> std::result::Result<(), String> {
    let selected = selection.parameters;
    let ranges = &capability.parameters;
    for (name, value, allowed) in [
        (
            "decode_tile_edge",
            selected.decode_tile_edge,
            &ranges.decode_tile_edges,
        ),
        (
            "decode_overlap",
            selected.decode_overlap,
            &ranges.decode_overlaps,
        ),
        (
            "attention_chunk_size",
            selected.attention_chunk_size,
            &ranges.attention_chunk_sizes,
        ),
        (
            "transformer_window_size",
            selected.transformer_window_size,
            &ranges.transformer_window_sizes,
        ),
    ] {
        if let Some(value) = value {
            if !allowed.contains(&value) {
                return Err(format!(
                    "{name}={value} is outside the declared production candidates {allowed:?}"
                ));
            }
        }
    }
    Ok(())
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
    /// Effective request budget. Reclaimable memory cannot raise the result above total minus
    /// reserved headroom, and all arithmetic is saturating.
    pub fn effective_bytes(self) -> u64 {
        let ceiling = self
            .total_bytes
            .saturating_sub(self.reserved_headroom_bytes);
        self.total_bytes
            .saturating_sub(self.committed_bytes)
            .saturating_add(self.reclaimable_bytes)
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

/// Context for provider safety and lifecycle hooks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageMemoryRunContext {
    pub selection: ImageMemorySelection,
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
}

impl ImageMemoryEvidence {
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
    fn exact_budget_boundary_fits_and_reclaimable_is_capped() {
        let budget = ImageMemoryBudget {
            total_bytes: 100,
            committed_bytes: 80,
            reclaimable_bytes: 50,
            reserved_headroom_bytes: 10,
        };
        assert_eq!(budget.effective_bytes(), 70);
        assert!(budget.fits(70));
        assert!(!budget.fits(71));
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
}
