//! The shared image-memory ladder for **Chroma1 HD / Base / Flash** on MLX/Metal (SC-15520).
//!
//! Chroma is the FLUX MMDiT skeleton with pruned adaLN — 19 double + 38 single blocks whose
//! modulation comes from a dense `distilled_guidance_layer` Approximator rather than from per-block
//! `norm*.linear` weights — plus **T5-XXL-only** conditioning and the FLUX.1 16-channel VAE. All
//! three catalog entries (`chroma1_hd`, `chroma1_base`, `chroma1_flash`) are the *same* provider
//! route: one architecture, one loader, one contract shape; they differ only in checkpoint, sampler
//! default and schedule. Nothing here is inherited from FLUX.1 or from any other family — the epic's
//! standing rule is that a rung's presence, magnitude and candidate set are per family per backend,
//! so every number this module publishes comes from
//! `tests/memory_ladder_real_weights.rs` on this Mac's Metal GPU.
//!
//! ## What each rung does here
//!
//! | rung | mechanism | state |
//! |---|---|---|
//! | 1 staged residency | shed the T5-XXL encoder before the DiT/VAE load, per request | Implemented |
//! | 2 bounded decode | `Vae::decode_tiled` over the FLUX.1 16-ch AutoencoderKL | see [`DECODE_SUPPORT`] |
//! | 3 bounded attention | `sdpa_budgeted_bhsd` on both block stacks | see [`ATTENTION_SUPPORT`] |
//! | 4 bounded transformer residency | `run_windowed` over the two DiT sub-stacks | see [`WINDOW_SUPPORT`] |
//!
//! ## The T5 is the conditioning phase, and it is large
//!
//! Chroma's shipped q4 tier packs only the DiT block linears; the T5-XXL encoder ships **dense bf16**
//! at 9.08 GiB against a 5.18 GiB packed transformer. Under rung 1 the request peak is therefore
//! conditioning-bound rather than denoise-bound, which is why
//! [`TRANSFORMER_WINDOW_COMPONENT`] is a **measured** choice rather than the `Dit` default every
//! provider had before the component scope existed (SC-15794, and the Kolors precedent in SC-15521).
//! sc-16462 (inference PR #443) is the story that packs those auxiliaries; when it lands, every
//! conditioning-phase number in this module's tests is re-derived rather than inherited.

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::gen_core::{
    standard_memory_strategy_safety_check, Error as CoreError, LoadShape, LoadSpec,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyPrerequisite, MemoryStrategySupport,
    TransformerComponent,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{GenerationRequest, OffloadPolicy, WeightsSource};

use crate::config::{ChromaTransformerConfig, ChromaVariant};

/// Static, weights-free identity for the registry conformance walk. Never a production calibration.
const STATIC_CALIBRATION: &str = "chroma1-mlx-registry-behavior-v1";

/// The production calibration key. Bound to the measured entry/tier/geometry in
/// [`production_calibration_fingerprint`]; every other axis fails closed.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "chroma1-base-q4-mlx-shared-ladder-2026-08-05-v1";

// ── Rung 2: the native VAE decode geometry ───────────────────────────────────────────────────────

/// Tile edges swept against the exact untiled decode of the **production** latent, in output pixels.
///
/// Recorded here rather than in a comment so `the_rejected_decode_geometries_are_refused_by_the_production_path`
/// can re-assert every rejection against the production admission path.
pub const DECODE_TILE_EDGES_SWEPT: &[u32] = &[768, 640, 512, 384];
/// Feather overlaps swept beside [`DECODE_TILE_EDGES_SWEPT`].
pub const DECODE_OVERLAPS_SWEPT: &[u32] = &[64, 96, 128];

/// Whether the swept rung-2 geometry survived measurement on the production path.
///
/// `false` publishes rung 2 as `Missing` **with its numbers recorded** in the harness, which is the
/// epic's requirement for a measured-and-withheld rung — never a silent omission.
pub const DECODE_SUPPORT: bool = true;
/// The published native tile-edge domain (descending). Empty when [`DECODE_SUPPORT`] is `false`.
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512];
/// The default edge inside [`DECODE_TILE_EDGES`].
pub const DECODE_TILE_EDGE: u32 = 512;
/// The one published feather overlap.
pub const DECODE_OVERLAP: u32 = 128;

// ── Rung 3: the bounded-attention budget ─────────────────────────────────────────────────────────

/// The single published attention-score budget — the shared `CONSTRAINED_ATTN_SCORES_BUDGET`.
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
/// Whether bounded attention survived measurement on the production path.
pub const ATTENTION_SUPPORT: bool = true;

// ── Rung 4: the transformer window ───────────────────────────────────────────────────────────────

/// Window cadences swept over the DiT sub-stacks.
pub const TRANSFORMER_WINDOW_SIZES_SWEPT: &[u32] = &[1, 2, 5, 10];
/// Whether bounded transformer residency survived measurement on the production path.
pub const WINDOW_SUPPORT: bool = true;
/// The published window cadence domain.
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1];
/// The default cadence inside [`TRANSFORMER_WINDOW_SIZES`].
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;
/// The **measured** default component scope, and the one a request that names none receives.
///
/// This is not a `Dit`-by-inheritance choice. Measured on Chroma1-Base q4 at 1024² (Apple/Metal,
/// `the_request_peak_bearing_phase_is_measured_not_assumed`), the phases are conditioning 10.15 GiB
/// / denoise 7.14 GiB / decode 14.14 GiB untiled — and rung 4's own composition engages rung 2, so
/// its decode is already bounded to 4.37 GiB and **conditioning binds**. A `Dit`-scoped window
/// bounds the 7.14 GiB denoise, which is below the binding phase, and therefore moves the request
/// peak by nothing. `TextEncoder` is the scope that addresses the binding phase; Kolors reached the
/// same conclusion first, for the same reason and a different encoder (SC-15521).
pub const TRANSFORMER_WINDOW_COMPONENT: TransformerComponent = TransformerComponent::TextEncoder;
/// The component scopes this provider will admit. All three are implemented; the harness measures
/// each and `the_window_component_scopes_are_measured_not_inherited` pins what each one buys.
/// [`TRANSFORMER_WINDOW_COMPONENT`] leads the list: the shared behavior fixture takes the first
/// published candidate as its representative, so a domain whose head is not the default would have
/// the conformance walk exercising a scope no unqualified request receives.
pub const TRANSFORMER_WINDOW_COMPONENTS: &[TransformerComponent] = &[
    TransformerComponent::TextEncoder,
    TransformerComponent::Dit,
    TransformerComponent::Both,
];

/// The depth the shared request scope validates window alignment against: the **longest** windowable
/// sub-stack across every admitted component scope, since each stack is windowed independently over
/// the same cadence. Chroma's are 19 double + 38 single DiT blocks and 24 T5 encoder blocks.
pub(crate) fn window_stack_depth() -> usize {
    let cfg = ChromaTransformerConfig::default();
    cfg.num_layers
        .max(cfg.num_single_layers)
        .max(mlx_gen_flux::T5_BLOCKS)
}

/// The native VAE decode ladder, declared through the checked shared constructor so it can never
/// overlap the PiD student's disjoint domain (sc-15775).
fn decode_routes(provider_id: &str) -> mlx_gen::gen_core::Result<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(
        provider_id,
        DECODE_TILE_EDGES.iter().copied(),
        DECODE_OVERLAP,
    )
}

fn variant_for(provider_id: &str) -> mlx_gen::gen_core::Result<ChromaVariant> {
    match provider_id {
        crate::CHROMA1_HD_ID => Ok(ChromaVariant::Hd),
        crate::CHROMA1_BASE_ID => Ok(ChromaVariant::Base),
        crate::CHROMA1_FLASH_ID => Ok(ChromaVariant::Flash),
        _ => Err(CoreError::Unsupported(format!(
            "unknown Chroma1 memory provider {provider_id}"
        ))),
    }
}

/// The `'static` provider id for a route, so every message names the same string the registry does.
fn provider_static(provider_id: &str) -> mlx_gen::gen_core::Result<&'static str> {
    Ok(variant_for(provider_id)?.id())
}

/// The overlay axes a request carries, joined into the contract's route key. `None` is the clean
/// base route — the only one whose quality-sensitive rungs are admitted.
///
/// Chroma's crate reads only `adapters` and `pid` today; the remaining axes are listed anyway so a
/// spec that carries one fails **closed** here rather than being silently treated as clean by a
/// provider that would ignore it.
fn route_overlay(spec: &LoadSpec) -> Option<String> {
    let mut axes = Vec::new();
    if !spec.adapters.is_empty() {
        axes.push("adapters");
    }
    if spec.control.is_some() {
        axes.push("control");
    }
    if !spec.extra_controls.is_empty() {
        axes.push("extra-controls");
    }
    if spec.ip_adapter.is_some() {
        axes.push("ip-adapter");
    }
    if spec.identity.is_some() {
        axes.push("identity");
    }
    if spec.text_encoder.is_some() {
        axes.push("external-text-encoder");
    }
    if spec.pid.is_some() {
        axes.push("pid");
    }
    (!axes.is_empty()).then(|| axes.join("-"))
}

fn clean(spec: &LoadSpec) -> bool {
    route_overlay(spec).is_none()
}

/// Chroma is text-to-image only (`conditioning: vec![]`), so the route is fixed.
fn route_mode_and_references(spec: &LoadSpec) -> (MemoryMode, u32) {
    let _ = spec;
    (MemoryMode::TextToImage, 0)
}

/// Whether this load can rebuild individual DiT blocks from the snapshot on demand.
///
/// `spec.quantize.is_some()` is refused deliberately: that combination means a **dense** source the
/// loader packs in place, so a window would re-quantize its blocks on every re-materialization —
/// a host-format conversion per window, which is not what
/// [`MemoryWindowMaterialization::DeviceFormatTransfer`](mlx_gen::gen_core::MemoryWindowMaterialization)
/// declares. A shipped packed tier loads with `quantize == None` (the loader packed-detects
/// `{base}.scales`), so the production Q4/Q8 route is admitted and only the re-quantizing one is not.
pub fn structurally_streamable(spec: &LoadSpec) -> bool {
    WINDOW_SUPPORT
        && spec.offload_policy == OffloadPolicy::Sequential
        && spec.load_shape == LoadShape::DeferredMaterialization
        && spec.quantize.is_none()
        && clean(spec)
        && matches!(spec.weights, WeightsSource::Dir(_))
}

#[allow(clippy::too_many_arguments)]
fn build_contract(
    provider_id: &str,
    spec: &LoadSpec,
    streamable: bool,
    calibration: Option<MemoryCalibrationIdentity>,
    footprint: mlx_gen::PerComponentBytes,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    variant_for(provider_id)?;
    let staged = spec.offload_policy == OffloadPolicy::Sequential;
    let clean = clean(spec);
    let decode = DECODE_SUPPORT && clean;
    let attention = ATTENTION_SUPPORT && clean;
    let routes = decode_routes(provider_id)?;
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.calibration = calibration;
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::DecodeTileArea,
            MemoryFormulaVariable::AttentionChunkSize,
            MemoryFormulaVariable::TransformerWindowSize,
        ],
    };
    contract.asset_facts.base_bytes = footprint
        .text_encoder
        .saturating_add(footprint.dit)
        .saturating_add(footprint.vae);
    contract.asset_facts.conditioning_bytes = footprint.text_encoder;
    contract.asset_facts.transformer_bytes = footprint.dit;
    contract.asset_facts.decoder_bytes = footprint.vae;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: staged,
        decode_tiling: decode,
        attention_chunking: attention,
        transformer_window_materialization: streamable,
    };
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency if staged => MemoryStrategySupport::Implemented,
            MemoryStrategy::BoundedDecode if decode => {
                capability.parameters.decode_tile_edges = routes.native_edges().to_vec();
                capability.parameters.decode_overlaps = vec![DECODE_OVERLAP];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedAttention if attention => {
                capability.parameters.attention_chunk_sizes = vec![ATTENTION_CHUNK_SIZE];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency if streamable => {
                capability.parameters.transformer_window_sizes = TRANSFORMER_WINDOW_SIZES.to_vec();
                capability.parameters.transformer_window_components =
                    TRANSFORMER_WINDOW_COMPONENTS.to_vec();
                MemoryStrategySupport::Implemented
            }
            _ => MemoryStrategySupport::Missing,
        };
    }
    if streamable {
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
    }
    Ok(contract)
}

/// The measured production key. Every axis is exact; anything else stays uncalibrated so the
/// selector cannot reach an optimized strategy on evidence that was never taken.
fn production_calibration_fingerprint(provider_id: &str, spec: &LoadSpec) -> Option<&'static str> {
    (provider_id == crate::CHROMA1_BASE_ID
        && spec.precision == mlx_gen::Precision::Bf16
        && spec.quantize.is_none()
        && spec.offload_policy == OffloadPolicy::Sequential
        && clean(spec))
    .then_some(MEMORY_CALIBRATION_FINGERPRINT)
}

/// The production contract, with filesystem-backed asset facts.
pub fn contract_for(
    provider_id: &str,
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    let calibration = production_calibration_fingerprint(provider_id, spec)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape));
    build_contract(
        provider_id,
        spec,
        structurally_streamable(spec),
        calibration,
        crate::model::component_footprint(spec)?,
    )
}

/// Declaration-equivalent, zero-filesystem contract used only by registry conformance.
#[doc(hidden)]
pub fn weights_free_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    let route = variant_for(provider_id)?.id().replace('_', "-");
    build_contract(
        provider_id,
        spec,
        structurally_streamable(spec),
        Some(MemoryCalibrationIdentity::new(
            format!("{STATIC_CALIBRATION}-{route}"),
            spec.load_shape,
        )),
        Default::default(),
    )
}

pub(crate) fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        let (expected_mode, expected_references) = route_mode_and_references(spec);
        if context.mode != expected_mode
            || context.geometry.reference_count != expected_references
            || context.overlay != route_overlay(spec)
        {
            return Err(CoreError::Unsupported(format!(
                "{}: memory route does not match the loaded mode/overlay",
                contract.provider_id
            )));
        }
        if context.use_pid && spec.pid.is_none() {
            return Err(CoreError::Unsupported(format!(
                "{}: PiD route requested without a loaded PiD overlay",
                contract.provider_id
            )));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            decode_routes(&contract.provider_id)?
                .validate(
                    context.use_pid,
                    context.selection.parameters.decode_tile_edge,
                    context.selection.parameters.decode_overlap,
                )
                .map_err(CoreError::Unsupported)?;
        }
        if contract.engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        ) {
            if !contract.lifecycle.transformer_window_materialization || !context.has_phases {
                return Err(CoreError::Unsupported(format!(
                    "{}: bounded transformer residency requires the Sequential + \
                     DeferredMaterialization route with rung 1 engaged in the same request",
                    contract.provider_id
                )));
            }
            let component = context.selection.parameters.window_component();
            if !TRANSFORMER_WINDOW_COMPONENTS.contains(&component) {
                return Err(CoreError::Unsupported(format!(
                    "{}: transformer window component {component:?} is outside the published \
                     domain {TRANSFORMER_WINDOW_COMPONENTS:?}",
                    contract.provider_id
                )));
            }
        }
        Ok(())
    };
    standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        }),
        Some(&route_gate),
    )
}

pub(crate) fn registered_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> mlx_gen::gen_core::Result<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    if contract
        .capability(strategy)
        .is_none_or(|capability| capability.support != MemoryStrategySupport::Implemented)
    {
        return Ok(Vec::new());
    }
    let (mode, reference_count) = route_mode_and_references(spec);
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode,
            reference_count,
            use_pid: false,
            has_phases: spec.offload_policy == OffloadPolicy::Sequential,
            overlay: route_overlay(spec),
        },
    )?;
    Ok(vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(context)])
}

fn begin_request_with_cleanup(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let provider = provider_static(&contract.provider_id)?;
    let routes = decode_routes(provider)?;
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        window_stack_depth(),
        move |use_pid, edge, overlap| {
            routes
                .validate(use_pid, Some(edge), Some(overlap))
                .map_err(CoreError::Unsupported)
        },
    )?;
    config.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
    config.transformer_window = contract
        .engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .then_some(context.selection.parameters.transformer_window_size)
        .flatten();
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    )))
}

pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub(crate) fn begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

// ── Request-local resolution of the selected parameters ──────────────────────────────────────────

/// Rung 3's plan. An unselected request returns the exact historical unbounded/uncancellable plan.
pub(crate) fn attention_plan(req: &GenerationRequest) -> AttentionPlan<'_> {
    match req.memory {
        Some(memory) if memory.chunk_attention => {
            AttentionPlan::budgeted(AttentionBudget::CONSTRAINED).with_cancel(&req.cancel)
        }
        _ => AttentionPlan::UNBOUNDED,
    }
}

/// Rung 3's parameter domain check, on the same layer as its rung-2 and rung-4 siblings.
pub(crate) fn validate_attention_chunk(
    req: &GenerationRequest,
    provider_id: &str,
) -> mlx_gen::Result<()> {
    let Some(memory) = req.memory.filter(|memory| memory.chunk_attention) else {
        return Ok(());
    };
    match memory.attention_chunk_size {
        None | Some(ATTENTION_CHUNK_SIZE) => Ok(()),
        Some(other) => Err(mlx_gen::Error::Unsupported(format!(
            "{provider_id}: attention chunk size {other} is outside the published production \
             domain [{ATTENTION_CHUNK_SIZE}]"
        ))),
    }
}

/// Rung 4's window, or `None` when unselected. Validates the cadence and the component scope
/// against the published domains, on the production path.
pub(crate) fn transformer_window(
    req: &GenerationRequest,
    provider_id: &str,
) -> mlx_gen::Result<Option<usize>> {
    let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
        return Ok(None);
    };
    if !memory.stage_residency {
        return Err(mlx_gen::Error::Unsupported(format!(
            "{provider_id}: bounded transformer residency requires staged residency in the same \
             request"
        )));
    }
    if req.use_pid {
        return Err(mlx_gen::Error::Unsupported(format!(
            "{provider_id}: bounded transformer residency is not implemented for the PiD decode \
             overlay"
        )));
    }
    let component = memory
        .transformer_window_component
        .unwrap_or(TRANSFORMER_WINDOW_COMPONENT);
    if !TRANSFORMER_WINDOW_COMPONENTS.contains(&component) {
        return Err(mlx_gen::Error::Unsupported(format!(
            "{provider_id}: transformer window component {component:?} is outside the published \
             production domain {TRANSFORMER_WINDOW_COMPONENTS:?}"
        )));
    }
    let size = memory
        .transformer_window_size
        .unwrap_or(TRANSFORMER_WINDOW_SIZE);
    if !TRANSFORMER_WINDOW_SIZES.contains(&size) {
        return Err(mlx_gen::Error::Unsupported(format!(
            "{provider_id}: transformer window {size} is outside the published production domain \
             {TRANSFORMER_WINDOW_SIZES:?}"
        )));
    }
    Ok(Some(size as usize))
}

/// Rung 2's plan. Unselected requests return `None`, keeping the historical one-pass decode exactly
/// intact. PiD is handled before this plan at the decode call site and never inherits native VAE
/// geometry.
pub(crate) fn decode_tiling(
    req: &GenerationRequest,
    provider_id: &str,
) -> mlx_gen::Result<Option<TilingConfig>> {
    let Some(memory) = req.memory.filter(|memory| memory.tile_vae_decode) else {
        return Ok(None);
    };
    if req.cancel.is_cancelled() {
        return Err(mlx_gen::Error::Canceled);
    }
    let edge = memory.decode_tile_edge.unwrap_or(DECODE_TILE_EDGE);
    let overlap = memory.decode_overlap.unwrap_or(DECODE_OVERLAP);
    // The same checked route set admission used, so an out-of-domain value is refused on the
    // production render path and not only at selection time.
    decode_routes(provider_id)
        .map_err(|error| mlx_gen::Error::Unsupported(error.to_string()))?
        .validate(req.use_pid, Some(edge), Some(overlap))
        .map_err(mlx_gen::Error::Unsupported)?;
    Ok(Some(TilingConfig::spatial_only(
        edge as i32,
        overlap as i32,
    )))
}

/// Request-local conformance fault at a completed physical phase boundary (SC-15449). The shared
/// request floor authorizes this pair; production requests leave both fields unset.
pub(crate) fn calibration_fault(
    req: &GenerationRequest,
    phase: MemoryPhase,
    provider_id: &str,
) -> mlx_gen::Result<()> {
    match req.memory {
        Some(memory)
            if memory.calibration_fault_harness_authorized
                && memory.calibration_error_phase == Some(phase) =>
        {
            Err(mlx_gen::Error::Msg(format!(
                "{provider_id}: injected memory-strategy calibration error at {phase:?}"
            )))
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::GenerationMemory;
    use mlx_gen::{AdapterKind, AdapterSpec, Quant};

    fn sequential_spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
    }

    fn streamable_spec() -> LoadSpec {
        sequential_spec().with_load_shape(LoadShape::DeferredMaterialization)
    }

    #[test]
    fn every_entry_publishes_the_same_ladder_and_is_internally_coherent() {
        for provider in [
            crate::CHROMA1_HD_ID,
            crate::CHROMA1_BASE_ID,
            crate::CHROMA1_FLASH_ID,
        ] {
            let contract = weights_free_contract(provider, &streamable_spec()).unwrap();
            assert!(
                contract.conformance_errors().is_empty(),
                "{provider}: {:?}",
                contract.conformance_errors()
            );
            assert_eq!(contract.asset_facts, Default::default());
            for (strategy, expected) in [
                (MemoryStrategy::Resident, true),
                (MemoryStrategy::StagedResidency, true),
                (MemoryStrategy::BoundedDecode, DECODE_SUPPORT),
                (MemoryStrategy::BoundedAttention, ATTENTION_SUPPORT),
                (MemoryStrategy::BoundedTransformerResidency, WINDOW_SUPPORT),
            ] {
                let support = &contract.capability(strategy).unwrap().support;
                assert_eq!(
                    matches!(support, MemoryStrategySupport::Implemented),
                    expected,
                    "{provider} {strategy:?}"
                );
            }
        }
        // Sibling entries must not be Verified by sharing code: only the measured entry carries a
        // production calibration identity.
        assert_eq!(
            production_calibration_fingerprint(crate::CHROMA1_BASE_ID, &sequential_spec()),
            Some(MEMORY_CALIBRATION_FINGERPRINT)
        );
        for provider in [crate::CHROMA1_HD_ID, crate::CHROMA1_FLASH_ID] {
            assert!(production_calibration_fingerprint(provider, &sequential_spec()).is_none());
        }
    }

    #[test]
    fn the_calibration_key_is_exact_and_every_other_axis_fails_closed() {
        let exact = sequential_spec();
        for changed in [
            exact.clone().with_quant(Quant::Q4),
            exact.clone().with_offload_policy(OffloadPolicy::Resident),
            {
                let mut spec = exact.clone();
                spec.precision = mlx_gen::Precision::Fp32;
                spec
            },
            {
                let mut spec = exact.clone();
                spec.adapters.push(AdapterSpec::new(
                    "/adapter.safetensors".into(),
                    1.0,
                    AdapterKind::Lora,
                ));
                spec
            },
            exact.clone().with_pid(
                WeightsSource::File("/pid.safetensors".into()),
                WeightsSource::Dir("/gemma".into()),
            ),
        ] {
            assert!(
                production_calibration_fingerprint(crate::CHROMA1_BASE_ID, &changed).is_none(),
                "a changed axis must not inherit the calibrated key"
            );
        }
    }

    #[test]
    fn every_overlay_withdraws_the_quality_sensitive_rungs() {
        let cases = [
            {
                let mut spec = streamable_spec();
                spec.adapters.push(AdapterSpec::new(
                    "/adapter.safetensors".into(),
                    1.0,
                    AdapterKind::Lora,
                ));
                spec
            },
            {
                let mut spec = streamable_spec();
                spec.control = Some(WeightsSource::File("/control.safetensors".into()));
                spec
            },
            {
                let mut spec = streamable_spec();
                spec.extra_controls
                    .push(WeightsSource::File("/control-2.safetensors".into()));
                spec
            },
            {
                let mut spec = streamable_spec();
                spec.ip_adapter = Some(WeightsSource::Dir("/ip".into()));
                spec
            },
            {
                let mut spec = streamable_spec();
                spec.identity = Some(Default::default());
                spec
            },
            {
                let mut spec = streamable_spec();
                spec.text_encoder = Some(WeightsSource::Dir("/external-text".into()));
                spec
            },
            streamable_spec().with_pid(
                WeightsSource::File("/pid.safetensors".into()),
                WeightsSource::Dir("/gemma".into()),
            ),
        ];
        for spec in cases {
            let contract = weights_free_contract(crate::CHROMA1_BASE_ID, &spec).unwrap();
            assert!(contract.conformance_errors().is_empty());
            for strategy in [
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ] {
                assert_eq!(
                    contract.capability(strategy).unwrap().support,
                    MemoryStrategySupport::Missing,
                    "{strategy:?} must be withdrawn for {:?}",
                    route_overlay(&spec)
                );
            }
            assert!(!contract.lifecycle.decode_tiling);
            assert!(!contract.lifecycle.attention_chunking);
            assert!(!contract.lifecycle.transformer_window_materialization);
        }
    }

    #[test]
    fn rung_four_needs_sequential_deferred_and_a_non_requantizing_tier() {
        assert!(structurally_streamable(&streamable_spec()));
        for spec in [
            // Eager materialization: no reopenable stream.
            sequential_spec(),
            // Resident: rung 1 cannot be engaged in the same request.
            streamable_spec().with_offload_policy(OffloadPolicy::Resident),
            // A dense source the loader would re-quantize per window.
            streamable_spec().with_quant(Quant::Q4),
            // A single-file checkpoint has no component tree to reopen.
            LoadSpec::new(WeightsSource::File("/one.safetensors".into()))
                .with_offload_policy(OffloadPolicy::Sequential)
                .with_load_shape(LoadShape::DeferredMaterialization),
        ] {
            assert!(!structurally_streamable(&spec));
            assert_eq!(
                weights_free_contract(crate::CHROMA1_BASE_ID, &spec)
                    .unwrap()
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing
            );
        }
    }

    #[test]
    fn the_request_scope_admits_only_the_published_parameter_domains() {
        let spec = streamable_spec();
        let contract = weights_free_contract(crate::CHROMA1_BASE_ID, &spec).unwrap();

        let mut fixture = registered_fixture(&spec, &contract, MemoryStrategy::BoundedDecode)
            .unwrap()
            .remove(0);
        let mut scope = registered_begin_request(&spec, &contract, &fixture.context)
            .unwrap()
            .unwrap();
        scope.configure_request(&mut fixture.request).unwrap();
        let memory = fixture
            .request
            .memory
            .expect("bounded decode request memory");
        assert!(memory.tile_vae_decode);
        let edge = memory.decode_tile_edge.expect("selected edge");
        let overlap = memory.decode_overlap.expect("selected overlap");
        scope
            .configure_decode(edge, overlap, fixture.context.geometry)
            .unwrap();
        assert!(scope
            .configure_decode(edge + 1, overlap, fixture.context.geometry)
            .is_err());
        assert!(scope
            .configure_decode(edge, overlap + 1, fixture.context.geometry)
            .is_err());

        let mut fixture = registered_fixture(&spec, &contract, MemoryStrategy::BoundedAttention)
            .unwrap()
            .remove(0);
        let mut scope = registered_begin_request(&spec, &contract, &fixture.context)
            .unwrap()
            .unwrap();
        scope.configure_request(&mut fixture.request).unwrap();
        assert!(fixture.request.memory.unwrap().chunk_attention);
        scope.configure_attention(ATTENTION_CHUNK_SIZE).unwrap();
        assert!(scope.configure_attention(ATTENTION_CHUNK_SIZE - 1).is_err());

        let mut fixture = registered_fixture(
            &spec,
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .remove(0);
        let mut scope = registered_begin_request(&spec, &contract, &fixture.context)
            .unwrap()
            .unwrap();
        scope.configure_request(&mut fixture.request).unwrap();
        let memory = fixture.request.memory.expect("rung 4 request memory");
        assert!(memory.stage_residency, "rung 4 must engage rung 1");
        assert!(memory.stream_transformer_blocks);
        assert_eq!(
            memory.transformer_window_component,
            Some(TRANSFORMER_WINDOW_COMPONENT)
        );
        let window = memory.transformer_window_size.expect("selected cadence");
        scope.materialize_transformer_window(0, window).unwrap();
        assert!(scope.materialize_transformer_window(1, window).is_err() || window == 1);
    }

    #[test]
    fn native_and_pid_decode_routes_are_disjoint_and_checked() {
        let routes = decode_routes(crate::CHROMA1_BASE_ID).unwrap();
        assert_eq!(routes.native_edges(), DECODE_TILE_EDGES);
        routes
            .validate(false, Some(DECODE_TILE_EDGE), Some(DECODE_OVERLAP))
            .unwrap();
        // A swept-but-rejected edge is refused by the same checked route set the production render
        // path uses — the rejection is enforced, not documented.
        for edge in DECODE_TILE_EDGES_SWEPT
            .iter()
            .filter(|edge| !DECODE_TILE_EDGES.contains(edge))
        {
            assert!(routes
                .validate(false, Some(*edge), Some(DECODE_OVERLAP))
                .is_err());
        }
        for overlap in DECODE_OVERLAPS_SWEPT
            .iter()
            .filter(|overlap| **overlap != DECODE_OVERLAP)
        {
            assert!(routes
                .validate(false, Some(DECODE_TILE_EDGE), Some(*overlap))
                .is_err());
        }
        assert!(routes
            .validate(true, Some(DECODE_TILE_EDGE), Some(DECODE_OVERLAP))
            .is_err());
        let pid_edge = mlx_gen_pid::DecodeRoutes::pid_edges()[0];
        let pid_overlap = mlx_gen_pid::DecodeRoutes::pid_overlap();
        routes
            .validate(true, Some(pid_edge), Some(pid_overlap))
            .unwrap();
        assert!(routes
            .validate(false, Some(pid_edge), Some(pid_overlap))
            .is_err());
    }

    #[test]
    fn request_local_resolvers_are_exact_and_default_to_the_historical_path() {
        let plain = GenerationRequest::default();
        assert_eq!(attention_plan(&plain).budget, AttentionBudget::UNBOUNDED);
        assert!(attention_plan(&plain).cancel.is_none());
        assert!(decode_tiling(&plain, crate::CHROMA1_BASE_ID)
            .unwrap()
            .is_none());
        assert!(transformer_window(&plain, crate::CHROMA1_BASE_ID)
            .unwrap()
            .is_none());
        validate_attention_chunk(&plain, crate::CHROMA1_BASE_ID).unwrap();

        let chunked = GenerationRequest {
            memory: Some(GenerationMemory {
                chunk_attention: true,
                attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            attention_plan(&chunked).budget,
            AttentionBudget::CONSTRAINED
        );
        assert!(attention_plan(&chunked).cancel.is_some());
        validate_attention_chunk(&chunked, crate::CHROMA1_BASE_ID).unwrap();
        let out_of_domain = GenerationRequest {
            memory: Some(GenerationMemory {
                chunk_attention: true,
                attention_chunk_size: Some(ATTENTION_CHUNK_SIZE - 1),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(validate_attention_chunk(&out_of_domain, crate::CHROMA1_BASE_ID).is_err());

        let tiled = GenerationRequest {
            memory: Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(DECODE_TILE_EDGE),
                decode_overlap: Some(DECODE_OVERLAP),
                ..Default::default()
            }),
            ..Default::default()
        };
        let plan = decode_tiling(&tiled, crate::CHROMA1_BASE_ID)
            .unwrap()
            .unwrap();
        let spatial = plan.spatial.expect("spatial-only plan");
        assert_eq!(spatial.tile_px, DECODE_TILE_EDGE as i32);
        assert_eq!(spatial.overlap_px, DECODE_OVERLAP as i32);
        assert!(plan.temporal.is_none());
        let canceled = tiled.clone();
        canceled.cancel.cancel();
        assert!(matches!(
            decode_tiling(&canceled, crate::CHROMA1_BASE_ID),
            Err(mlx_gen::Error::Canceled)
        ));
        let bad_edge = GenerationRequest {
            memory: Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(DECODE_TILE_EDGE + 1),
                decode_overlap: Some(DECODE_OVERLAP),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(decode_tiling(&bad_edge, crate::CHROMA1_BASE_ID).is_err());

        let windowed = GenerationRequest {
            memory: Some(GenerationMemory {
                stage_residency: true,
                stream_transformer_blocks: true,
                transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
                transformer_window_component: Some(TRANSFORMER_WINDOW_COMPONENT),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            transformer_window(&windowed, crate::CHROMA1_BASE_ID).unwrap(),
            Some(TRANSFORMER_WINDOW_SIZE as usize)
        );
        // Rung 4 without rung 1 in the same request is refused on the production path.
        let unstaged = GenerationRequest {
            memory: Some(GenerationMemory {
                stage_residency: false,
                ..windowed.memory.unwrap()
            }),
            ..Default::default()
        };
        assert!(transformer_window(&unstaged, crate::CHROMA1_BASE_ID).is_err());
        // An out-of-domain cadence is refused on the production path.
        let out_of_domain = GenerationRequest {
            memory: Some(GenerationMemory {
                transformer_window_size: Some(3),
                ..windowed.memory.unwrap()
            }),
            ..Default::default()
        };
        assert!(transformer_window(&out_of_domain, crate::CHROMA1_BASE_ID).is_err());
        // Every published scope is reachable, and each resolves to the same cadence.
        for component in TRANSFORMER_WINDOW_COMPONENTS {
            let req = GenerationRequest {
                memory: Some(GenerationMemory {
                    transformer_window_component: Some(*component),
                    ..windowed.memory.unwrap()
                }),
                ..Default::default()
            };
            assert_eq!(
                transformer_window(&req, crate::CHROMA1_BASE_ID).unwrap(),
                Some(TRANSFORMER_WINDOW_SIZE as usize),
                "published scope {component:?} must be reachable"
            );
        }
    }

    #[test]
    fn the_calibration_fault_is_authorized_phase_exact_and_request_local() {
        let mut memory = GenerationMemory::default();
        memory.authorize_calibration_fault(MemoryPhase::Decode);
        let injected = GenerationRequest {
            memory: Some(memory),
            ..Default::default()
        };
        assert!(calibration_fault(&injected, MemoryPhase::Denoise, crate::CHROMA1_BASE_ID).is_ok());
        let error = calibration_fault(&injected, MemoryPhase::Decode, crate::CHROMA1_BASE_ID)
            .unwrap_err()
            .to_string();
        assert!(error.contains(crate::CHROMA1_BASE_ID));
        assert!(error.contains("Decode"));
        assert!(calibration_fault(
            &GenerationRequest::default(),
            MemoryPhase::Decode,
            crate::CHROMA1_BASE_ID
        )
        .is_ok());
        // An unauthorized phase is inert: the pair is what the shared floor accepts.
        let unauthorized = GenerationRequest {
            memory: Some(GenerationMemory {
                calibration_error_phase: Some(MemoryPhase::Decode),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(
            calibration_fault(&unauthorized, MemoryPhase::Decode, crate::CHROMA1_BASE_ID).is_ok()
        );
    }

    #[test]
    fn the_window_stack_depth_comes_from_the_model_config() {
        let cfg = ChromaTransformerConfig::default();
        assert_eq!(window_stack_depth(), cfg.num_single_layers);
        assert!(window_stack_depth() >= cfg.num_layers);
        assert!(window_stack_depth() >= mlx_gen_flux::T5_BLOCKS);
    }
}
