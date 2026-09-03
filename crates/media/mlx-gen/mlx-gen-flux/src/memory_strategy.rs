//! FLUX.1 MLX shared memory-provider contract foundation (SC-15514).
//!
//! This slice exposes the existing two-phase `Residency` lifecycle through the shared provider
//! contract. The clean schnell/dev routes use the shared head-once/tail-tiled native VAE decode and
//! thread the shared bounded-attention kernel through every double- and single-stream block. Control
//! and every loaded overlay remain `Missing` until their additional paths have independent coverage.
//! Production calibration is limited to the exact measured FLUX.1-dev Q4 deferred artifact and
//! request geometry; weights-free registry conformance receives an isolated synthetic identity.
//!
//! ## Envelope vs. structure in the request route gate (sc-20569 twin)
//!
//! The route gate carries two kinds of clause and they must not be confused — the same distinction
//! SC-20569 established for mlx-gen-sensenova. A **structural** clause (loaded mode/overlay/reference
//! match, PiD requested without a loaded overlay, request phases, decode-tile geometry, transformer
//! streaming eligibility) states what THIS loaded route can do at all — no authority changes that —
//! and stays fail-closed unconditionally. The **envelope** clause (the calibrated-geometry block)
//! states what the `2026-08-03` FLUX.1-dev Q4 campaign MEASURED: clean text-to-image at 1024x1024,
//! batch 1, one frame, zero references. That is a statement about which evidence exists, not about
//! what the engine can render — the shipped `flux_dev` manifest advertises 768x768, 1280x720, and
//! 720x1280 alongside 1024x1024, and counts 1/2/4, and this clause fired on ANY optimized-rung
//! selection regardless of authority, so a legacy/estimated admission at any of those off-cell
//! combinations was refused. Only a context that also claims measured evidence
//! (`optimization_authority == Calibrated`) is held to the measured cell; an `Estimated` claim —
//! exactly what `AdmissionPath::Legacy` in the SceneWorks fit gate carries for an out-of-envelope
//! request — must degrade to the caller's legacy/estimated admission instead of refusing.
//!
//! Every clause inside `route_gate` therefore builds a `CoreError::Msg`, not `CoreError::Unsupported`.
//! `standard_memory_strategy_safety_check` stringifies a route-gate error into
//! `MemorySafetyDecision::Reject` immediately (`error.to_string()`), so nothing downstream can read
//! the type, while `begin_request_with_cleanup` types the surviving string as `Unsupported` once at
//! the request boundary. Building `Unsupported` inside the closure too doubled the prefix into
//! `unsupported: unsupported: …` — the same secondary defect SC-20569 fixed in mlx-gen-sensenova.

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::gen_core::{
    Error as CoreError, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier,
    MemoryOptimizationAuthority, MemoryPhase, MemoryPrerequisiteScope, MemoryProviderContract,
    MemoryRequestScope, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyPrerequisite, MemoryStrategySupport, Result as CoreResult, TransformerComponent,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{GenerationRequest, LoadSpec, OffloadPolicy};

pub const STATIC_BEHAVIOR_FINGERPRINT: &str = "flux-one-static-registry-behavior-v2";
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "flux1-dev-q4-mlx-shared-ladder-2026-08-03-v1";
pub const CALIBRATED_Q4_COMPOSITE_SHA256: &str =
    "9dbbfeec18eb1fb137d264fe74777fe01f2f15cb0a1402f1e47c76c795463fbe";
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
/// Native FLUX.1/Z-Image VAE tile geometry in output pixels. These are the production candidates
/// supported by the shared head-once/tail-tiled decoder; PiD owns a separate, disjoint domain.
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512];
pub const DECODE_OVERLAP: u32 = 64;
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;

fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(
        provider_id,
        DECODE_TILE_EDGES.iter().copied(),
        DECODE_OVERLAP,
    )
}

fn is_known_provider(provider_id: &str) -> bool {
    matches!(
        provider_id,
        crate::FLUX1_SCHNELL_ID | crate::FLUX1_DEV_ID | crate::FLUX1_DEV_CONTROL_ID
    )
}

/// Latent channels the FLUX.1 autoencoder produces and the DiT consumes.
///
/// Not a config field in this crate: it is the shape literal the pack/unpack pair in
/// [`crate::pipeline`] is written against (`[1, 16, h, w]`). Naming it here keeps the published axis
/// and that geometry in one place.
const LATENT_CHANNELS: u32 = 16;
/// Pixels per latent unit on each spatial axis — the four-stage FLUX.1 AutoencoderKL.
const VAE_SPATIAL_SCALE: u32 = 8;
/// The 2x2 latent packing `pipeline::pack_latents` applies before the DiT. Together with
/// [`VAE_SPATIAL_SCALE`] this is exactly [`crate::SIZE_MULTIPLE`], which the tests pin.
const LATENT_PATCH_SIZE: u32 = 2;

/// Architecture axes shared by all three registered FLUX.1 routes (epic SC-22657, E2).
///
/// The DiT axes are this crate's own transformer constants — `transformer::HEADS`,
/// `transformer::HEAD_DIM` and `FluxTransformerConfig`, which the loader builds every variant from.
/// Schnell, Dev and Dev-Control share one DiT and differ only in guidance support and the control
/// overlay, so they publish one set of axes.
///
/// `transformer_blocks` is the **sum** of the joint and single stacks (19 + 38): both are
/// transformer blocks the denoiser traverses on every step, and publishing only the joint half would
/// understate the trunk by two thirds.
///
/// `vae_temporal_scale` stays `None` — FLUX.1 is an image model whose autoencoder has no temporal
/// axis, and a structurally absent axis is declared absent, never zero.
fn architecture_facts() -> mlx_gen::gen_core::MemoryArchitectureFacts {
    let dit =
        crate::transformer::FluxTransformerConfig::for_variant(crate::config::FluxVariant::Dev);
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(crate::transformer::HEADS),
        head_dim: mlx_gen::architecture_facts::axis(crate::transformer::HEAD_DIM),
        transformer_blocks: mlx_gen::architecture_facts::axis(
            dit.num_layers.saturating_add(dit.num_single_layers),
        ),
        patch_size: mlx_gen::architecture_facts::axis(LATENT_PATCH_SIZE),
        latent_channels: mlx_gen::architecture_facts::axis(LATENT_CHANNELS),
        vae_spatial_scale: mlx_gen::architecture_facts::axis(VAE_SPATIAL_SCALE),
        vae_temporal_scale: None,
        // The DiT's main residual stream is f32; only the modulation path is bf16
        // (`transformer.rs`), so f32 is the activation width the peak is built on.
        activation_dtype_width: Some(mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH),
    }
}

fn validate_load_contract(provider_id: &str, spec: &LoadSpec) -> CoreResult<()> {
    if !is_known_provider(provider_id) {
        return Err(CoreError::Unsupported(format!(
            "unknown FLUX.1 provider {provider_id}"
        )));
    }
    if !matches!(spec.weights, mlx_gen::WeightsSource::Dir(_))
        || spec.precision != mlx_gen::Precision::Bf16
        || !matches!(
            spec.quantize,
            None | Some(mlx_gen::Quant::Q4 | mlx_gen::Quant::Q8)
        )
    {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: FLUX.1 memory routes require a bf16/Q4/Q8 snapshot directory"
        )));
    }
    let unsupported_common = !spec.extra_controls.is_empty()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
        || !spec.components.is_empty();
    let valid = match provider_id {
        crate::FLUX1_SCHNELL_ID | crate::FLUX1_DEV_ID => {
            !unsupported_common && spec.control.is_none()
        }
        crate::FLUX1_DEV_CONTROL_ID => {
            !unsupported_common
                && spec.control.is_some()
                && spec.ip_adapter.is_none()
                && spec.pid.is_none()
        }
        _ => false,
    };
    if !valid {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: FLUX.1 memory route does not support the loaded component composition"
        )));
    }
    Ok(())
}

fn route_overlay(provider_id: &str, spec: &LoadSpec) -> Option<String> {
    let mut axes = Vec::new();
    if provider_id == crate::FLUX1_DEV_CONTROL_ID || spec.control.is_some() {
        axes.push("control");
    }
    if !spec.extra_controls.is_empty() {
        axes.push("extra-controls");
    }
    if !spec.adapters.is_empty() {
        axes.push("adapters");
    }
    if spec.ip_adapter.is_some() {
        axes.push("ip-adapter");
    }
    if spec.pid.is_some() {
        axes.push("pid");
    }
    if spec.identity.is_some() {
        axes.push("identity");
    }
    if spec.text_encoder.is_some() {
        axes.push("external-text-encoder");
    }
    (!axes.is_empty()).then(|| axes.join("-"))
}

fn route_mode_and_references(provider_id: &str, spec: &LoadSpec) -> (MemoryMode, u32) {
    if provider_id == crate::FLUX1_DEV_CONTROL_ID
        || spec.control.is_some()
        || spec.ip_adapter.is_some()
        || spec.identity.is_some()
    {
        (MemoryMode::ImageToImage, 1)
    } else {
        (MemoryMode::TextToImage, 0)
    }
}

fn build_contract(
    provider_id: &str,
    spec: &LoadSpec,
    footprint: mlx_gen::PerComponentBytes,
    streamable: bool,
    calibration: Option<MemoryCalibrationIdentity>,
) -> CoreResult<MemoryProviderContract> {
    if !is_known_provider(provider_id) {
        return Err(CoreError::Unsupported(format!(
            "unknown FLUX.1 provider {provider_id}"
        )));
    }
    let staged = matches!(spec.offload_policy, OffloadPolicy::Sequential);
    let clean_base =
        provider_id != crate::FLUX1_DEV_CONTROL_ID && route_overlay(provider_id, spec).is_none();
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
    contract.architecture_facts = architecture_facts();
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
            MemoryFormulaVariable::OverlayBytes,
            MemoryFormulaVariable::DecodeTileArea,
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
        decode_tiling: clean_base,
        attention_chunking: clean_base,
        transformer_window_materialization: streamable,
    };
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency if staged => MemoryStrategySupport::Implemented,
            MemoryStrategy::BoundedDecode if clean_base => {
                capability.parameters.decode_tile_edges = routes.native_edges().to_vec();
                capability.parameters.decode_overlaps = vec![DECODE_OVERLAP];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedAttention if clean_base => {
                capability.parameters.attention_chunk_sizes = vec![ATTENTION_CHUNK_SIZE];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency if streamable => {
                capability.parameters.transformer_window_sizes = vec![TRANSFORMER_WINDOW_SIZE];
                capability.parameters.transformer_window_components =
                    vec![TransformerComponent::Dit];
                MemoryStrategySupport::Implemented
            }
            _ => MemoryStrategySupport::Missing,
        };
    }
    contract.additional_prerequisites.push((
        MemoryStrategy::BoundedTransformerResidency,
        MemoryStrategyPrerequisite::Rung {
            rung: MemoryStrategy::StagedResidency,
            scope: MemoryPrerequisiteScope::EngagedInSameRequest,
        },
    ));
    Ok(contract)
}

/// Production contract. Filesystem-backed asset facts are real; rung 4 is declared loadable only
/// for an exact pinned packed inventory, while calibration remains absent until real measurement.
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    validate_load_contract(provider_id, spec)?;
    let inventory = crate::artifact_inventory::verified_stream_inventory(provider_id, spec);
    memory_strategy_contract_with_inventory(provider_id, spec, inventory.as_ref())
}

/// Production admission with an inventory already resolved by the loader. Keeping validation in
/// this wrapper prevents loaded generators from bypassing the exact source/component contract while
/// still letting the base loader reuse its single inventory snapshot.
pub(crate) fn validated_memory_strategy_contract_with_inventory(
    provider_id: &str,
    spec: &LoadSpec,
    inventory: Option<&crate::artifact_inventory::PackedArtifactInventory>,
) -> CoreResult<MemoryProviderContract> {
    validate_load_contract(provider_id, spec)?;
    memory_strategy_contract_with_inventory(provider_id, spec, inventory)
}

/// Verify and identify the exact deferred packed artifact used by a real-weight evidence runner.
///
/// This is deliberately evidence-only: it performs the same full pinned inventory/content admission
/// as production rung 4 and returns that inventory's composite SHA-256, but it does not grant or
/// mutate a production calibration identity.
#[doc(hidden)]
pub fn verified_runner_artifact(provider_id: &str, spec: &LoadSpec) -> CoreResult<String> {
    if !crate::artifact_inventory::structurally_streamable(provider_id, spec) {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: runner artifact verification requires Sequential + DeferredMaterialization on a clean bf16/Q4/Q8 base route"
        )));
    }
    let inventory = crate::artifact_inventory::verified_stream_inventory(provider_id, spec)
        .ok_or_else(|| {
            CoreError::Unsupported(format!(
                "{provider_id}: runner artifact failed exact pinned inventory/content verification"
            ))
        })?;
    let contract = memory_strategy_contract_with_inventory(provider_id, spec, Some(&inventory))?;
    validate_runner_gate(provider_id, inventory.composite_sha256(), &contract)?;
    inventory.ensure_unchanged()?;
    Ok(inventory.composite_sha256().to_owned())
}

fn production_calibration_identity(
    provider_id: &str,
    spec: &LoadSpec,
    inventory: Option<&crate::artifact_inventory::PackedArtifactInventory>,
) -> Option<MemoryCalibrationIdentity> {
    let inventory = inventory?;
    production_calibration_fingerprint(provider_id, spec, inventory.composite_sha256())
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape))
}

fn production_calibration_fingerprint(
    provider_id: &str,
    spec: &LoadSpec,
    composite_sha256: &str,
) -> Option<&'static str> {
    (provider_id == crate::FLUX1_DEV_ID
        && composite_sha256 == CALIBRATED_Q4_COMPOSITE_SHA256
        && spec.precision == mlx_gen::Precision::Bf16
        && spec.quantize == Some(mlx_gen::Quant::Q4)
        && spec.offload_policy == OffloadPolicy::Sequential
        && spec.load_shape == mlx_gen::LoadShape::DeferredMaterialization
        && route_overlay(provider_id, spec).is_none())
    .then_some(MEMORY_CALIBRATION_FINGERPRINT)
}

/// Fail closed unless the real-weight runner is bound to the exact measured production key.
pub fn validate_runner_gate(
    provider_id: &str,
    artifact_sha256: &str,
    contract: &MemoryProviderContract,
) -> CoreResult<()> {
    let valid = provider_id == crate::FLUX1_DEV_ID
        && artifact_sha256 == CALIBRATED_Q4_COMPOSITE_SHA256
        && contract.provider_id == provider_id
        && contract.calibration.as_ref().is_some_and(|identity| {
            identity.fingerprint == MEMORY_CALIBRATION_FINGERPRINT
                && identity.load_shape == mlx_gen::LoadShape::DeferredMaterialization
                && identity.load_shape == contract.load_shape
        });
    if !valid {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: runner artifact/contract does not match the calibrated FLUX.1-dev Q4 key"
        )));
    }
    Ok(())
}

pub(crate) fn memory_strategy_contract_with_inventory(
    provider_id: &str,
    spec: &LoadSpec,
    inventory: Option<&crate::artifact_inventory::PackedArtifactInventory>,
) -> CoreResult<MemoryProviderContract> {
    if let Some(inventory) = inventory {
        inventory.ensure_unchanged()?;
        if inventory.composite_sha256().len() != 64
            || !inventory
                .transformer_source()
                .canonical_path()
                .is_absolute()
        {
            return Err(CoreError::Unsupported(
                "flux1: verified packed inventory has an invalid composite or transformer pin"
                    .to_owned(),
            ));
        }
    }
    let calibration = production_calibration_identity(provider_id, spec, inventory);
    build_contract(
        provider_id,
        spec,
        crate::model::component_footprint(spec)?,
        inventory.is_some(),
        calibration,
    )
}

/// Declaration-equivalent, zero-filesystem contract used only by registry conformance.
#[doc(hidden)]
pub fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    // The legacy registry-conformance probe supplies one caller-owned LoadSpec to every provider.
    // Authenticate all of its axes, but supply the control route's intrinsic source only for this
    // zero-filesystem fixture. The selector-aware resolver below remains the authoritative finite
    // surface, while production `memory_strategy_contract` always requires the real control source.
    let mut validation_spec = spec.clone();
    if provider_id == crate::FLUX1_DEV_CONTROL_ID && validation_spec.control.is_none() {
        validation_spec.control = Some(mlx_gen::WeightsSource::File(
            "/weights-free-flux1-control.safetensors".into(),
        ));
    }
    validate_load_contract(provider_id, &validation_spec)?;
    let route = match provider_id {
        crate::FLUX1_SCHNELL_ID => "schnell",
        crate::FLUX1_DEV_ID => "dev",
        crate::FLUX1_DEV_CONTROL_ID => "dev-control",
        _ => "unknown",
    };
    build_contract(
        provider_id,
        spec,
        mlx_gen::PerComponentBytes::default(),
        crate::artifact_inventory::structurally_streamable(provider_id, spec),
        Some(MemoryCalibrationIdentity::new(
            format!("{STATIC_BEHAVIOR_FINGERPRINT}-{route}"),
            spec.load_shape,
        )),
    )
}

fn surface_selector_matches_spec(
    surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec,
) -> CoreResult<()> {
    use mlx_gen::gen_core::MemoryContractSurfaceTier;

    let tier_matches = match surface.resolved_artifact_tier() {
        MemoryContractSurfaceTier::Bf16 => {
            surface.spec.precision == mlx_gen::Precision::Bf16 && surface.spec.quantize.is_none()
        }
        MemoryContractSurfaceTier::Q4 => surface.spec.quantize == Some(mlx_gen::Quant::Q4),
        MemoryContractSurfaceTier::Q8 => surface.spec.quantize == Some(mlx_gen::Quant::Q8),
        MemoryContractSurfaceTier::Nvfp4 => false,
    };
    let plain = surface.spec.adapters.is_empty()
        && surface.spec.control.is_none()
        && surface.spec.extra_controls.is_empty()
        && surface.spec.ip_adapter.is_none()
        && surface.spec.pid.is_none()
        && surface.spec.identity.is_none()
        && surface.spec.text_encoder.is_none()
        && surface.spec.components.is_empty();
    if tier_matches
        && plain
        && matches!(surface.spec.weights, mlx_gen::WeightsSource::Dir(_))
        && surface.selector.offload_policy == surface.spec.offload_policy
        && surface.selector.load_shape == surface.spec.load_shape
    {
        Ok(())
    } else {
        Err(CoreError::Unsupported(format!(
            "FLUX.1 memory surface selector '{}' does not match its plain registry LoadSpec",
            surface.selector.id()
        )))
    }
}

/// Resolve the finite registry surface from the selector's explicit artifact tier.
///
/// FLUX.1 keeps its historical matching `LoadSpec::quantize` value for a prepacked Q4/Q8 turnkey:
/// the loader treats it as a checked no-op when the component marker matches. The selector remains
/// authoritative so facts never infer the resolved tier from a future load-time conversion shape.
pub(crate) fn weights_free_memory_surface_contract(
    provider_id: &str,
    surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec,
) -> CoreResult<MemoryProviderContract> {
    use mlx_gen::gen_core::MemoryContractSurfaceTier;

    surface_selector_matches_spec(surface)?;
    let supported_tier = matches!(
        surface.resolved_artifact_tier(),
        MemoryContractSurfaceTier::Bf16
            | MemoryContractSurfaceTier::Q4
            | MemoryContractSurfaceTier::Q8
    );
    let mut spec = surface.spec.clone();
    if provider_id == crate::FLUX1_DEV_CONTROL_ID {
        spec.control = Some(mlx_gen::WeightsSource::File(
            "/weights-free-flux1-control.safetensors".into(),
        ));
    }
    validate_load_contract(provider_id, &spec)?;
    let streamable = supported_tier
        && matches!(provider_id, crate::FLUX1_SCHNELL_ID | crate::FLUX1_DEV_ID)
        && spec.offload_policy == OffloadPolicy::Sequential
        && spec.load_shape == mlx_gen::LoadShape::DeferredMaterialization;
    let route = match provider_id {
        crate::FLUX1_SCHNELL_ID => "schnell",
        crate::FLUX1_DEV_ID => "dev",
        crate::FLUX1_DEV_CONTROL_ID => "dev-control",
        _ => "unknown",
    };
    build_contract(
        provider_id,
        &spec,
        Default::default(),
        streamable,
        Some(MemoryCalibrationIdentity::new(
            format!("{STATIC_BEHAVIOR_FINGERPRINT}-{route}"),
            spec.load_shape,
        )),
    )
}

pub(crate) fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        let (expected_mode, expected_references) =
            route_mode_and_references(&contract.provider_id, spec);
        if context.mode != expected_mode
            || context.geometry.reference_count != expected_references
            || context.overlay != route_overlay(&contract.provider_id, spec)
        {
            return Err(CoreError::Msg(format!(
                "{}: memory route does not match the loaded mode/overlay",
                contract.provider_id
            )));
        }
        if context.use_pid && spec.pid.is_none() {
            return Err(CoreError::Msg(format!(
                "{}: PiD was requested without a loaded PiD overlay",
                contract.provider_id
            )));
        }
        if context.has_phases {
            return Err(CoreError::Msg(format!(
                "{}: FLUX.1 memory routes are single-phase",
                contract.provider_id
            )));
        }
        // sc-20569 twin: `claims_measured_evidence` is the same envelope/structure discriminator the
        // SenseNova route gate uses. An optimized rung admitted behind a legacy/estimated authority
        // has NOT claimed to be graded against the measured cell, so it must degrade to admission
        // instead of refusing; only a `Calibrated` claim off the measured cell fails closed, because
        // admitting THAT would grade a request against evidence captured at a different geometry.
        let claims_measured_evidence =
            context.optimization_authority == MemoryOptimizationAuthority::Calibrated;
        if contract
            .calibration
            .as_ref()
            .is_some_and(|identity| identity.fingerprint == MEMORY_CALIBRATION_FINGERPRINT)
            && context.selection.strategy.is_optimized()
            && claims_measured_evidence
            && (context.mode != MemoryMode::TextToImage
                || context.geometry.reference_count != 0
                || context.geometry.width != 1024
                || context.geometry.height != 1024
                || context.geometry.batch != 1
                || context.geometry.frames != 1
                || context.use_pid
                || context.overlay.is_some())
        {
            return Err(CoreError::Msg(format!(
                "{}: calibrated memory geometry is exactly clean text-to-image 1024x1024, batch 1, one frame, and zero references",
                contract.provider_id
            )));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            let routes = decode_routes(&contract.provider_id)?;
            routes
                .validate(
                    context.use_pid,
                    context.selection.parameters.decode_tile_edge,
                    context.selection.parameters.decode_overlap,
                )
                .map_err(CoreError::Msg)?;
        }
        if contract.engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        ) && !contract.lifecycle.transformer_window_materialization
        {
            return Err(CoreError::Msg(format!(
                "{}: transformer streaming requires the verified Sequential + DeferredMaterialization route",
                contract.provider_id
            )));
        }
        Ok(())
    };
    mlx_gen::gen_core::standard_memory_strategy_safety_check(
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

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    safety_check(spec, contract, context)
}

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let (mode, reference_count) = route_mode_and_references(&contract.provider_id, spec);
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
            use_pid: spec.pid.is_some(),
            has_phases: false,
            overlay: route_overlay(&contract.provider_id, spec),
        },
    )?;
    let mut fixture = mlx_gen::gen_core::MemoryBehaviorFixture::new(context);
    fixture.request.prompt = "weights-free FLUX.1 memory behavior".to_owned();
    fixture.request.phases = None;
    fixture.request.use_pid = spec.pid.is_some();
    let reference = || mlx_gen::media::Image {
        width: 1,
        height: 1,
        pixels: vec![0, 0, 0],
    };
    fixture.request.conditioning = match contract.provider_id.as_str() {
        crate::FLUX1_DEV_CONTROL_ID => vec![mlx_gen::Conditioning::Control {
            image: reference(),
            kind: mlx_gen::ControlKind::Pose,
            scale: Some(1.0),
        }],
        _ if spec.ip_adapter.is_some() => vec![mlx_gen::Conditioning::Reference {
            image: reference(),
            strength: Some(1.0),
        }],
        _ => Vec::new(),
    };
    Ok(vec![fixture])
}

pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        provider_id,
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub(crate) fn begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    begin_request_with_cleanup(
        provider_id,
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

fn begin_request_with_cleanup(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    // The one place a refused route becomes a TYPED error. `MemorySafetyDecision::Reject` carries an
    // already-rendered string, so the route gate deliberately hands back plain reasons (see
    // `safety_check`) and the `unsupported: ` prefix is applied exactly once, here.
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let routes = decode_routes(provider_id)?;
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        57,
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

/// Resolve the clean-base request's attention plan. An unselected request returns the exact
/// historical unbounded/uncancellable plan; the shared request scope supplies the only accepted
/// bounded score budget when rung 3 is selected.
pub(crate) fn attention_plan(req: &GenerationRequest) -> AttentionPlan<'_> {
    match req.memory {
        Some(memory) if memory.chunk_attention => {
            AttentionPlan::budgeted(AttentionBudget::CONSTRAINED).with_cancel(&req.cancel)
        }
        _ => AttentionPlan::UNBOUNDED,
    }
}

/// Resolve an explicitly selected native VAE tile plan. Unselected requests return `None`, keeping
/// the historical one-pass decode exactly intact. PiD is deliberately handled before this plan at
/// the decode call site and therefore never inherits native VAE geometry.
pub(crate) fn decode_tiling(req: &GenerationRequest) -> mlx_gen::Result<Option<TilingConfig>> {
    let Some(memory) = req.memory.filter(|memory| memory.tile_vae_decode) else {
        return Ok(None);
    };
    if req.cancel.is_cancelled() {
        return Err(mlx_gen::Error::Canceled);
    }
    Ok(Some(TilingConfig::spatial_only(
        memory.decode_tile_edge.unwrap_or(DECODE_TILE_EDGE) as i32,
        memory.decode_overlap.unwrap_or(DECODE_OVERLAP) as i32,
    )))
}

/// Request-local conformance fault at a completed physical phase boundary. The shared request floor
/// authorizes this pair; production requests leave both fields unset.
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
    use mlx_gen::WeightsSource;

    fn sequential_spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
    }

    fn sequential_spec_for(provider_id: &str) -> LoadSpec {
        let mut spec = sequential_spec();
        if provider_id == crate::FLUX1_DEV_CONTROL_ID {
            spec.control = Some(WeightsSource::File("/control.safetensors".into()));
        }
        spec
    }

    fn write_exact_snapshot(root: &std::path::Path, quant: Option<mlx_gen::Quant>) {
        crate::artifact_inventory::write_test_snapshot(root, quant);
    }

    /// AC (SC-22662): every registered FLUX.1 route publishes the axes of the DiT it runs, derived
    /// from this crate's own transformer constants, and passes the shared facts conformance check.
    #[test]
    fn architecture_facts_follow_the_crate_transformer_constants() {
        for provider in [
            crate::FLUX1_SCHNELL_ID,
            crate::FLUX1_DEV_ID,
            crate::FLUX1_DEV_CONTROL_ID,
        ] {
            let spec = sequential_spec_for(provider);
            let contract = weights_free_memory_strategy_contract(provider, &spec).unwrap();
            assert_eq!(
                contract.architecture_facts,
                mlx_gen::gen_core::MemoryArchitectureFacts {
                    attention_heads: Some(24),
                    head_dim: Some(128),
                    // 19 joint + 38 single blocks.
                    transformer_blocks: Some(57),
                    patch_size: Some(2),
                    latent_channels: Some(16),
                    vae_spatial_scale: Some(8),
                    vae_temporal_scale: None,
                    activation_dtype_width: Some(4),
                },
                "{provider} architecture facts"
            );
            assert!(contract.architecture_facts.has_snapshot_read_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
    }

    /// The published spatial geometry is the one the request validator enforces: the VAE scale times
    /// the latent packing IS [`crate::SIZE_MULTIPLE`], so neither axis can drift alone.
    #[test]
    fn the_published_spatial_axes_multiply_to_the_enforced_size_multiple() {
        assert_eq!(VAE_SPATIAL_SCALE * LATENT_PATCH_SIZE, crate::SIZE_MULTIPLE);
    }

    #[test]
    fn static_contract_declares_native_decode_and_attention_only_for_clean_base_routes() {
        for provider in [
            crate::FLUX1_SCHNELL_ID,
            crate::FLUX1_DEV_ID,
            crate::FLUX1_DEV_CONTROL_ID,
        ] {
            let spec = sequential_spec_for(provider);
            let contract = weights_free_memory_strategy_contract(provider, &spec).unwrap();
            assert_eq!(contract.asset_facts, Default::default());
            assert!(contract.conformance_errors().is_empty());
            assert_eq!(
                contract
                    .capability(MemoryStrategy::StagedResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Implemented
            );
            let attention = contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap();
            let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
            if provider == crate::FLUX1_DEV_CONTROL_ID {
                assert_eq!(attention.support, MemoryStrategySupport::Missing);
                assert_eq!(decode.support, MemoryStrategySupport::Missing);
            } else {
                assert_eq!(attention.support, MemoryStrategySupport::Implemented);
                assert_eq!(
                    attention.parameters.attention_chunk_sizes,
                    [ATTENTION_CHUNK_SIZE]
                );
                let fixture =
                    registered_valid_fixture(&spec, &contract, MemoryStrategy::BoundedAttention)
                        .unwrap()
                        .remove(0);
                assert_eq!(
                    registered_safety_check(&spec, &contract, &fixture.context),
                    MemorySafetyDecision::Accept
                );
                assert_eq!(decode.support, MemoryStrategySupport::Implemented);
                assert_eq!(decode.parameters.decode_tile_edges, DECODE_TILE_EDGES);
                assert_eq!(decode.parameters.decode_overlaps, [DECODE_OVERLAP]);
                let fixture =
                    registered_valid_fixture(&spec, &contract, MemoryStrategy::BoundedDecode)
                        .unwrap()
                        .remove(0);
                assert_eq!(
                    registered_safety_check(&spec, &contract, &fixture.context),
                    MemorySafetyDecision::Accept
                );
            }
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing
            );
            let fixtures =
                registered_valid_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
                    .unwrap();
            assert_eq!(fixtures.len(), 1);
            assert_eq!(
                registered_safety_check(&spec, &contract, &fixtures[0].context),
                MemorySafetyDecision::Accept
            );
        }
    }

    #[test]
    fn static_rung4_fixture_is_zero_filesystem_and_requires_structural_eligibility() {
        let eligible = sequential_spec()
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization)
            .with_quant(mlx_gen::Quant::Q4);
        for provider in [crate::FLUX1_SCHNELL_ID, crate::FLUX1_DEV_ID] {
            let contract = weights_free_memory_strategy_contract(provider, &eligible).unwrap();
            assert_eq!(contract.asset_facts, Default::default());
            assert!(contract.lifecycle.transformer_window_materialization);
            let capability = contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap();
            assert_eq!(capability.support, MemoryStrategySupport::Implemented);
            assert_eq!(
                capability.parameters.transformer_window_sizes,
                [TRANSFORMER_WINDOW_SIZE]
            );
            assert_eq!(
                capability.parameters.transformer_window_components,
                [TransformerComponent::Dit]
            );
            let mut fixture = registered_valid_fixture(
                &eligible,
                &contract,
                MemoryStrategy::BoundedTransformerResidency,
            )
            .unwrap()
            .remove(0);
            let mut scope =
                registered_begin_request(provider, &eligible, &contract, &fixture.context)
                    .unwrap()
                    .unwrap();
            scope.configure_request(&mut fixture.request).unwrap();
            let memory = fixture.request.memory.expect("rung4 request memory");
            assert!(memory.stage_residency);
            assert!(memory.stream_transformer_blocks);
            assert_eq!(
                memory.transformer_window_size,
                Some(TRANSFORMER_WINDOW_SIZE)
            );
            assert_eq!(
                memory.transformer_window_component,
                Some(TransformerComponent::Dit)
            );
            scope
                .materialize_transformer_window(0, TRANSFORMER_WINDOW_SIZE)
                .unwrap();
        }

        let resident = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization)
            .with_quant(mlx_gen::Quant::Q4);
        assert_eq!(
            weights_free_memory_strategy_contract(crate::FLUX1_DEV_ID, &resident)
                .unwrap()
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
    }

    #[test]
    fn explicit_surface_resolver_preserves_matching_prepacked_tier_identity() {
        use mlx_gen::gen_core::MemoryContractSurfaceTier;

        for surface in mlx_gen::gen_core::mlx_memory_contract_surface_specs() {
            let tier = surface.resolved_artifact_tier();
            let expected_quant = match tier {
                MemoryContractSurfaceTier::Bf16 => None,
                MemoryContractSurfaceTier::Q4 => Some(mlx_gen::Quant::Q4),
                MemoryContractSurfaceTier::Q8 => Some(mlx_gen::Quant::Q8),
                MemoryContractSurfaceTier::Nvfp4 => unreachable!("MLX facts omit NVFP4"),
            };
            for provider in [
                crate::FLUX1_SCHNELL_ID,
                crate::FLUX1_DEV_ID,
                crate::FLUX1_DEV_CONTROL_ID,
            ] {
                let contract = weights_free_memory_surface_contract(provider, &surface).unwrap();
                assert_eq!(surface.spec.quantize, expected_quant, "{provider}");
                assert_eq!(contract.provider_id, provider);
            }

            let mut crossed_selector = surface.selector;
            crossed_selector.tier = match tier {
                MemoryContractSurfaceTier::Bf16 => MemoryContractSurfaceTier::Q4,
                _ => MemoryContractSurfaceTier::Bf16,
            };
            let crossed = mlx_gen::gen_core::MemoryContractSurfaceSpec {
                selector: crossed_selector,
                spec: surface.spec.clone(),
            };
            assert!(weights_free_memory_surface_contract(crate::FLUX1_DEV_ID, &crossed).is_err());
        }
    }

    #[test]
    fn production_rung4_needs_verified_exact_inventory_and_unknown_fixture_stays_uncalibrated() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_exact_snapshot(&root, Some(mlx_gen::Quant::Q4));
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization)
            .with_quant(mlx_gen::Quant::Q4);
        let contract = memory_strategy_contract(crate::FLUX1_DEV_ID, &spec).unwrap();
        let artifact =
            crate::artifact_inventory::verified_stream_inventory(crate::FLUX1_DEV_ID, &spec)
                .unwrap()
                .composite_sha256()
                .to_owned();
        assert_eq!(artifact.len(), 64);
        assert!(contract.calibration.is_none());
        assert!(verified_runner_artifact(crate::FLUX1_DEV_ID, &spec).is_err());
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );

        std::fs::write(
            root.join("transformer/model-00002-of-00002.safetensors"),
            [5_u8; 8],
        )
        .unwrap();
        let sharded = memory_strategy_contract(crate::FLUX1_DEV_ID, &spec).unwrap();
        assert!(sharded.calibration.is_none());
        assert_eq!(
            sharded
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert!(verified_runner_artifact(crate::FLUX1_DEV_ID, &spec).is_err());
        let resident = spec.clone().with_offload_policy(OffloadPolicy::Resident);
        assert!(verified_runner_artifact(crate::FLUX1_DEV_ID, &resident).is_err());
    }

    #[test]
    fn production_calibration_key_is_exact_and_all_other_axes_fail_closed() {
        let exact = LoadSpec::new(WeightsSource::Dir("/exact-q4".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization)
            .with_quant(mlx_gen::Quant::Q4);
        assert_eq!(
            production_calibration_fingerprint(
                crate::FLUX1_DEV_ID,
                &exact,
                CALIBRATED_Q4_COMPOSITE_SHA256,
            ),
            Some(MEMORY_CALIBRATION_FINGERPRINT)
        );
        assert!(production_calibration_fingerprint(
            crate::FLUX1_SCHNELL_ID,
            &exact,
            CALIBRATED_Q4_COMPOSITE_SHA256,
        )
        .is_none());
        assert!(production_calibration_fingerprint(crate::FLUX1_DEV_ID, &exact, "stale").is_none());
        for changed in [
            exact.clone().with_quant(mlx_gen::Quant::Q8),
            exact.clone().with_offload_policy(OffloadPolicy::Resident),
            exact
                .clone()
                .with_load_shape(mlx_gen::LoadShape::EagerMaterialization),
        ] {
            assert!(production_calibration_fingerprint(
                crate::FLUX1_DEV_ID,
                &changed,
                CALIBRATED_Q4_COMPOSITE_SHA256,
            )
            .is_none());
        }
        let mut overlay = exact;
        overlay.adapters.push(mlx_gen::AdapterSpec::new(
            "/adapter.safetensors".into(),
            1.0,
            mlx_gen::AdapterKind::Lora,
        ));
        assert!(production_calibration_fingerprint(
            crate::FLUX1_DEV_ID,
            &overlay,
            CALIBRATED_Q4_COMPOSITE_SHA256,
        )
        .is_none());
    }

    #[test]
    fn calibrated_contract_admits_only_the_measured_geometry_and_runner_key() {
        let spec = LoadSpec::new(WeightsSource::Dir("/exact-q4".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization)
            .with_quant(mlx_gen::Quant::Q4);
        let mut contract =
            weights_free_memory_strategy_contract(crate::FLUX1_DEV_ID, &spec).unwrap();
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            MEMORY_CALIBRATION_FINGERPRINT,
            spec.load_shape,
        ));
        validate_runner_gate(
            crate::FLUX1_DEV_ID,
            CALIBRATED_Q4_COMPOSITE_SHA256,
            &contract,
        )
        .unwrap();
        assert!(validate_runner_gate(crate::FLUX1_DEV_ID, "stale", &contract).is_err());

        let context = registered_valid_fixture(
            &spec,
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .remove(0)
        .context;
        assert_eq!(
            registered_safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Accept
        );
        for mutation in 0..6 {
            let mut changed = context.clone();
            match mutation {
                0 => changed.geometry.width = 768,
                1 => changed.geometry.height = 768,
                2 => changed.geometry.batch = 2,
                3 => changed.geometry.frames = 2,
                4 => changed.geometry.reference_count = 1,
                _ => changed.mode = MemoryMode::Edit,
            }
            assert!(matches!(
                registered_safety_check(&spec, &contract, &changed),
                MemorySafetyDecision::Reject { .. }
            ));
        }

        let mut resident = context;
        resident.selection.strategy = MemoryStrategy::Resident;
        resident.selection.parameters = Default::default();
        resident.geometry.width = 768;
        resident.geometry.height = 768;
        assert_eq!(
            registered_safety_check(&spec, &contract, &resident),
            MemorySafetyDecision::Accept,
            "the optimized exact-geometry calibration must not constrain Resident requests"
        );
    }

    /// The exact geometries `config/manifests/builtin.models.jsonc` advertises for `flux_dev`
    /// (`limits.resolutions`). Only `1024x1024` is the measured cell; the other three are shipped,
    /// product-legal geometries the calibrated-geometry envelope clause used to refuse outright.
    const MANIFEST_RESOLUTIONS: [(u32, u32); 4] =
        [(768, 768), (1024, 1024), (1280, 720), (720, 1280)];

    /// `flux_dev`'s manifest `limits.count`. SceneWorks pins the provider-facing `batch` to one
    /// forward pass today, but the gate is a provider-owned seam and any caller may set it, so the
    /// degrade is proven across the full advertised count axis rather than at the worker's current
    /// pin.
    const MANIFEST_COUNTS: [u32; 3] = [1, 2, 4];

    fn calibrated_q4_contract(spec: &LoadSpec) -> MemoryProviderContract {
        let mut contract =
            weights_free_memory_strategy_contract(crate::FLUX1_DEV_ID, spec).unwrap();
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            MEMORY_CALIBRATION_FINGERPRINT,
            spec.load_shape,
        ));
        contract
    }

    fn t2i_route() -> mlx_gen::gen_core::MemoryBehaviorRoute {
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        }
    }

    fn route_context(
        contract: &MemoryProviderContract,
        strategy: MemoryStrategy,
        tier: MemoryNumericTier,
    ) -> MemoryRunContext {
        mlx_gen::gen_core::standard_memory_behavior_context(contract, strategy, tier, t2i_route())
            .unwrap()
    }

    /// sc-20569 twin (mlx-gen-flux): every geometry/count `flux_dev` advertises off the measured
    /// 1024x1024/batch-1/frame-1 cell used to be refused unconditionally by the calibrated-geometry
    /// envelope clause whenever an optimized rung was selected, regardless of authority. A context
    /// that does NOT claim measured evidence — `AdmissionPath::Legacy` in the SceneWorks fit gate,
    /// whether it synthesized an estimate ladder or froze to the resident baseline — must be
    /// ADMITTED at every manifest-declared geometry and count, and must be able to open a request
    /// scope.
    #[test]
    fn every_manifest_geometry_and_count_degrades_to_legacy_admission() {
        let spec = LoadSpec::new(WeightsSource::Dir("/exact-q4".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization)
            .with_quant(mlx_gen::Quant::Q4);
        let contract = calibrated_q4_contract(&spec);
        let tier = MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        };

        let mut admitted = 0;
        for (width, height) in MANIFEST_RESOLUTIONS {
            for batch in MANIFEST_COUNTS {
                for (authority, strategy) in [
                    (
                        MemoryOptimizationAuthority::Estimated,
                        MemoryStrategy::BoundedAttention,
                    ),
                    (
                        MemoryOptimizationAuthority::Estimated,
                        MemoryStrategy::BoundedTransformerResidency,
                    ),
                    (
                        MemoryOptimizationAuthority::Resident,
                        MemoryStrategy::Resident,
                    ),
                ] {
                    let label =
                        format!("{width}x{height} batch {batch} {authority:?}/{strategy:?}");
                    let mut context = route_context(&contract, strategy, tier);
                    context.optimization_authority = authority;
                    context.geometry.width = width;
                    context.geometry.height = height;
                    context.geometry.batch = batch;
                    assert_eq!(
                        registered_safety_check(&spec, &contract, &context),
                        MemorySafetyDecision::Accept,
                        "{label}"
                    );
                    assert!(
                        begin_request(crate::FLUX1_DEV_ID, &spec, &contract, &context)
                            .unwrap_or_else(|error| panic!("{label}: {error}"))
                            .is_some(),
                        "{label}"
                    );
                    admitted += 1;
                }
            }
        }
        assert_eq!(
            admitted,
            MANIFEST_RESOLUTIONS.len() * MANIFEST_COUNTS.len() * 3,
            "four manifest resolutions x three counts x three legacy dispositions"
        );
    }

    /// The degrade must not weaken the STRUCTURAL refusals. No amount of estimate authority makes a
    /// clean `flux_dev` load answer a mismatched mode, admit PiD it never loaded, or run a
    /// multi-phase trajectory. Each axis is mutated on its own so every guard is asked its own
    /// question.
    #[test]
    fn structural_refusals_are_not_weakened_by_the_legacy_degrade() {
        let spec = LoadSpec::new(WeightsSource::Dir("/exact-q4".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization)
            .with_quant(mlx_gen::Quant::Q4);
        let contract = calibrated_q4_contract(&spec);
        let tier = MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        };
        let mut legacy = route_context(&contract, MemoryStrategy::BoundedAttention, tier);
        legacy.optimization_authority = MemoryOptimizationAuthority::Estimated;
        // The unmutated legacy context is admitted, so each rejection below is attributable to the
        // one axis it mutates.
        assert_eq!(
            registered_safety_check(&spec, &contract, &legacy),
            MemorySafetyDecision::Accept
        );

        let mut wrong_mode = legacy.clone();
        wrong_mode.mode = MemoryMode::Edit;
        let mut wrong_reference = legacy.clone();
        wrong_reference.geometry.reference_count = 1;
        wrong_reference.has_reference = true;
        let mut use_pid = legacy.clone();
        use_pid.use_pid = true;
        let mut has_phases = legacy.clone();
        has_phases.has_phases = true;
        for (label, context, needle) in [
            (
                "wrong_mode",
                wrong_mode,
                "does not match the loaded mode/overlay",
            ),
            (
                "wrong_reference",
                wrong_reference,
                "does not match the loaded mode/overlay",
            ),
            (
                "use_pid",
                use_pid,
                "PiD was requested without a loaded PiD overlay",
            ),
            (
                "has_phases",
                has_phases,
                "FLUX.1 memory routes are single-phase",
            ),
        ] {
            assert!(
                matches!(
                    registered_safety_check(&spec, &contract, &context),
                    MemorySafetyDecision::Reject { reason } if reason.contains(needle)
                ),
                "{label}: structural refusal must survive the legacy degrade"
            );
            assert!(
                begin_request(crate::FLUX1_DEV_ID, &spec, &contract, &context).is_err(),
                "{label}"
            );
        }
    }

    /// sc-20569 twin secondary: `begin_request` used to double-stringify a route-gate refusal —
    /// `standard_memory_strategy_safety_check` renders the route gate's error into
    /// `MemorySafetyDecision::Reject` once (`error.to_string()`), and `begin_request_with_cleanup`
    /// typed that already-rendered string as `Unsupported` again, printing
    /// `unsupported: unsupported: flux1_dev: …`. Exactly one prefix must reach the caller.
    #[test]
    fn a_route_refusal_surfaces_the_unsupported_prefix_exactly_once() {
        let spec = LoadSpec::new(WeightsSource::Dir("/exact-q4".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization)
            .with_quant(mlx_gen::Quant::Q4);
        let contract = calibrated_q4_contract(&spec);
        let tier = MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        };
        let mut context = route_context(&contract, MemoryStrategy::BoundedAttention, tier);
        context.geometry.width = 768;
        context.geometry.height = 768;
        let error = match begin_request(crate::FLUX1_DEV_ID, &spec, &contract, &context) {
            Ok(_) => panic!("a measured claim off the campaign cell must still refuse"),
            Err(error) => error.to_string(),
        };
        assert_eq!(
            error.matches("unsupported: ").count(),
            1,
            "the rendered refusal must carry one `unsupported: ` prefix: {error}"
        );
        assert!(
            error.starts_with(&format!("unsupported: {}: ", crate::FLUX1_DEV_ID)),
            "{error}"
        );
    }

    #[test]
    fn request_scope_configures_only_the_declared_attention_budget() {
        let spec = sequential_spec();
        let contract = weights_free_memory_strategy_contract(crate::FLUX1_DEV_ID, &spec).unwrap();
        let mut fixture =
            registered_valid_fixture(&spec, &contract, MemoryStrategy::BoundedAttention)
                .unwrap()
                .remove(0);
        let mut scope =
            registered_begin_request(crate::FLUX1_DEV_ID, &spec, &contract, &fixture.context)
                .unwrap()
                .unwrap();
        scope.configure_request(&mut fixture.request).unwrap();
        let memory = fixture.request.memory.expect("bounded request memory");
        assert!(memory.chunk_attention);
        assert_eq!(memory.attention_chunk_size, Some(ATTENTION_CHUNK_SIZE));
        scope.configure_attention(ATTENTION_CHUNK_SIZE).unwrap();
        assert!(scope.configure_attention(ATTENTION_CHUNK_SIZE - 1).is_err());
    }

    #[test]
    fn request_scope_configures_only_native_decode_geometry() {
        let spec = sequential_spec();
        let contract = weights_free_memory_strategy_contract(crate::FLUX1_DEV_ID, &spec).unwrap();
        let mut fixture = registered_valid_fixture(&spec, &contract, MemoryStrategy::BoundedDecode)
            .unwrap()
            .remove(0);
        let mut scope =
            registered_begin_request(crate::FLUX1_DEV_ID, &spec, &contract, &fixture.context)
                .unwrap()
                .unwrap();
        scope.configure_request(&mut fixture.request).unwrap();
        let memory = fixture.request.memory.expect("bounded request memory");
        assert!(memory.tile_vae_decode);
        let edge = memory.decode_tile_edge.expect("selected native edge");
        let overlap = memory.decode_overlap.expect("selected native overlap");
        scope
            .configure_decode(edge, overlap, fixture.context.geometry)
            .unwrap();
        assert!(scope
            .configure_decode(edge + 1, overlap, fixture.context.geometry)
            .is_err());
        assert!(scope
            .configure_decode(edge, overlap + 1, fixture.context.geometry)
            .is_err());
    }

    #[test]
    fn native_and_pid_decode_routes_are_disjoint_and_checked() {
        let routes = decode_routes(crate::FLUX1_DEV_ID).unwrap();
        assert_eq!(routes.native_edges(), DECODE_TILE_EDGES);
        routes
            .validate(false, Some(DECODE_TILE_EDGE), Some(DECODE_OVERLAP))
            .unwrap();
        assert!(routes
            .validate(false, Some(448), Some(DECODE_OVERLAP))
            .is_err());
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
    fn decode_tiling_is_request_local_exact_and_cancellable() {
        assert!(decode_tiling(&GenerationRequest::default())
            .unwrap()
            .is_none());
        let selected = GenerationRequest {
            memory: Some(mlx_gen::gen_core::GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(640),
                decode_overlap: Some(DECODE_OVERLAP),
                ..Default::default()
            }),
            ..Default::default()
        };
        let tiling = decode_tiling(&selected).unwrap().unwrap();
        let spatial = tiling.spatial.expect("spatial-only plan");
        assert_eq!(spatial.tile_px, 640);
        assert_eq!(spatial.overlap_px, DECODE_OVERLAP as i32);
        assert!(tiling.temporal.is_none());

        let canceled = selected.clone();
        canceled.cancel.cancel();
        assert!(matches!(
            decode_tiling(&canceled),
            Err(mlx_gen::Error::Canceled)
        ));
        assert!(decode_tiling(&GenerationRequest::default())
            .unwrap()
            .is_none());
    }

    #[test]
    fn decode_fault_is_authorized_phase_exact_and_request_local() {
        let mut memory = mlx_gen::gen_core::GenerationMemory::default();
        memory.authorize_calibration_fault(MemoryPhase::Decode);
        let injected = GenerationRequest {
            memory: Some(memory),
            ..Default::default()
        };
        assert!(calibration_fault(&injected, MemoryPhase::Denoise, crate::FLUX1_DEV_ID).is_ok());
        let error = calibration_fault(&injected, MemoryPhase::Decode, crate::FLUX1_DEV_ID)
            .unwrap_err()
            .to_string();
        assert!(error.contains(crate::FLUX1_DEV_ID));
        assert!(error.contains("Decode"));
        assert!(calibration_fault(
            &GenerationRequest::default(),
            MemoryPhase::Decode,
            crate::FLUX1_DEV_ID
        )
        .is_ok());
    }

    #[test]
    fn decode_error_finishes_scope_and_a_fresh_request_can_follow() {
        let spec = sequential_spec();
        let contract = weights_free_memory_strategy_contract(crate::FLUX1_DEV_ID, &spec).unwrap();
        let fixture = registered_valid_fixture(&spec, &contract, MemoryStrategy::BoundedDecode)
            .unwrap()
            .remove(0);
        let open = || {
            registered_begin_request(crate::FLUX1_DEV_ID, &spec, &contract, &fixture.context)
                .unwrap()
                .unwrap()
        };

        let mut failed = open();
        failed
            .finish(mlx_gen::gen_core::MemoryRunOutcome::Error {
                message: "injected decode fault".into(),
            })
            .unwrap();
        assert!(failed
            .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, fixture.context.geometry)
            .is_err());

        let mut follow_up = open();
        follow_up
            .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, fixture.context.geometry)
            .unwrap();
        follow_up
            .finish(mlx_gen::gen_core::MemoryRunOutcome::Complete)
            .unwrap();
    }

    #[test]
    fn attention_plan_is_request_local_and_unselected_is_exactly_unbounded() {
        let plain = GenerationRequest::default();
        let plan = attention_plan(&plain);
        assert_eq!(plan.budget, AttentionBudget::UNBOUNDED);
        assert!(plan.cancel.is_none());

        let selected = GenerationRequest {
            memory: Some(mlx_gen::gen_core::GenerationMemory {
                chunk_attention: true,
                attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                ..Default::default()
            }),
            ..Default::default()
        };
        let plan = attention_plan(&selected);
        assert_eq!(plan.budget, AttentionBudget::CONSTRAINED);
        assert!(plan.cancel.is_some());

        let follow_up = GenerationRequest::default();
        assert_eq!(
            attention_plan(&follow_up).budget,
            AttentionBudget::UNBOUNDED
        );
        assert!(attention_plan(&follow_up).cancel.is_none());
    }

    #[test]
    fn every_supported_base_overlay_keeps_upper_rungs_missing() {
        let cases = [
            {
                let mut spec = sequential_spec();
                spec.adapters.push(mlx_gen::AdapterSpec::new(
                    "/adapter.safetensors".into(),
                    1.0,
                    mlx_gen::AdapterKind::Lora,
                ));
                spec
            },
            {
                let mut spec = sequential_spec();
                spec.ip_adapter = Some(WeightsSource::Dir("/ip".into()));
                spec
            },
            sequential_spec().with_pid(
                WeightsSource::File("/pid.safetensors".into()),
                WeightsSource::Dir("/gemma".into()),
            ),
        ];
        for spec in cases {
            let contract =
                weights_free_memory_strategy_contract(crate::FLUX1_DEV_ID, &spec).unwrap();
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedAttention)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing
            );
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedDecode)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing
            );
            assert!(!contract.lifecycle.decode_tiling);
            assert!(!contract.lifecycle.attention_chunking);
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing
            );
        }
    }

    #[test]
    fn load_contract_rejects_every_unsupported_provider_source_and_component_axis() {
        let mut cases = Vec::new();
        cases.push(("unknown", sequential_spec()));
        cases.push((
            crate::FLUX1_DEV_ID,
            LoadSpec::new(WeightsSource::File("/weights.safetensors".into()))
                .with_offload_policy(OffloadPolicy::Sequential),
        ));
        let mut fp32 = sequential_spec();
        fp32.precision = mlx_gen::Precision::Fp32;
        cases.push((crate::FLUX1_DEV_ID, fp32));
        cases.push((
            crate::FLUX1_DEV_ID,
            sequential_spec().with_quant(mlx_gen::Quant::Nvfp4),
        ));
        let mut control_on_base = sequential_spec();
        control_on_base.control = Some(WeightsSource::File("/control.safetensors".into()));
        cases.push((crate::FLUX1_DEV_ID, control_on_base));
        let mut extra_control = sequential_spec();
        extra_control
            .extra_controls
            .push(WeightsSource::File("/extra.safetensors".into()));
        cases.push((crate::FLUX1_DEV_ID, extra_control));
        let mut identity = sequential_spec();
        identity.identity = Some(Default::default());
        cases.push((crate::FLUX1_DEV_ID, identity));
        let mut text_encoder = sequential_spec();
        text_encoder.text_encoder = Some(WeightsSource::Dir("/external-text".into()));
        cases.push((crate::FLUX1_DEV_ID, text_encoder));
        let mut component = sequential_spec();
        component.components.insert(
            "unknown".to_owned(),
            WeightsSource::File("/unknown.safetensors".into()),
        );
        cases.push((crate::FLUX1_DEV_ID, component));
        assert!(memory_strategy_contract(crate::FLUX1_DEV_CONTROL_ID, &sequential_spec()).is_err());
        let mut control_ip = sequential_spec_for(crate::FLUX1_DEV_CONTROL_ID);
        control_ip.ip_adapter = Some(WeightsSource::Dir("/ip".into()));
        cases.push((crate::FLUX1_DEV_CONTROL_ID, control_ip));
        let control_pid = sequential_spec_for(crate::FLUX1_DEV_CONTROL_ID).with_pid(
            WeightsSource::File("/pid.safetensors".into()),
            WeightsSource::Dir("/gemma".into()),
        );
        cases.push((crate::FLUX1_DEV_CONTROL_ID, control_pid));

        for (provider, spec) in cases {
            assert!(weights_free_memory_strategy_contract(provider, &spec).is_err());
        }
    }

    #[test]
    fn route_tier_overlay_and_load_shape_are_fail_closed() {
        let mut spec = sequential_spec();
        spec.adapters.push(mlx_gen::AdapterSpec::new(
            "/adapter.safetensors".into(),
            1.0,
            mlx_gen::AdapterKind::Lora,
        ));
        spec.ip_adapter = Some(WeightsSource::Dir("/ip".into()));
        spec = spec.with_pid(
            WeightsSource::File("/pid.safetensors".into()),
            WeightsSource::Dir("/gemma".into()),
        );
        let contract = weights_free_memory_strategy_contract(crate::FLUX1_DEV_ID, &spec).unwrap();
        let mut context =
            registered_valid_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
                .unwrap()
                .remove(0)
                .context;
        assert_eq!(
            registered_safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Accept
        );
        for mutation in 0..6 {
            let mut changed = context.clone();
            match mutation {
                0 => changed.overlay = Some("different".to_owned()),
                1 => changed.geometry.reference_count = 0,
                2 => changed.selection.tier.precision = mlx_gen::Precision::Fp32,
                3 => {
                    changed.load_shape = mlx_gen::LoadShape::DeferredMaterialization;
                }
                4 => changed.mode = MemoryMode::TextToImage,
                _ => changed.has_phases = true,
            }
            assert!(matches!(
                registered_safety_check(&spec, &contract, &changed),
                MemorySafetyDecision::Reject { .. }
            ));
        }
        let mut native_request = context.clone();
        native_request.use_pid = false;
        assert_eq!(
            registered_safety_check(&spec, &contract, &native_request),
            MemorySafetyDecision::Accept,
            "a loaded PiD overlay is optional per request"
        );
        let spec_without_pid = sequential_spec();
        let contract_without_pid =
            weights_free_memory_strategy_contract(crate::FLUX1_DEV_ID, &spec_without_pid).unwrap();
        let mut impossible_pid = registered_valid_fixture(
            &spec_without_pid,
            &contract_without_pid,
            MemoryStrategy::StagedResidency,
        )
        .unwrap()
        .remove(0)
        .context;
        impossible_pid.use_pid = true;
        assert!(matches!(
            registered_safety_check(&spec_without_pid, &contract_without_pid, &impossible_pid),
            MemorySafetyDecision::Reject { .. }
        ));
        let mut scope = registered_begin_request(crate::FLUX1_DEV_ID, &spec, &contract, &context)
            .unwrap()
            .unwrap();
        let mut request =
            registered_valid_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
                .unwrap()
                .remove(0)
                .request;
        scope.configure_request(&mut request).unwrap();
        assert!(request.memory.unwrap().stage_residency);
        context.calibration_fingerprint.push_str("-stale");
        assert!(matches!(
            registered_safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn production_unknown_artifact_has_no_calibration_and_rejects_static_context() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        for component in ["text_encoder", "text_encoder_2", "transformer", "vae"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
        }
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(OffloadPolicy::Sequential);
        let runtime = memory_strategy_contract(crate::FLUX1_DEV_ID, &spec).unwrap();
        assert!(runtime.calibration.is_none());
        let fixture = weights_free_memory_strategy_contract(crate::FLUX1_DEV_ID, &spec).unwrap();
        let context = registered_valid_fixture(&spec, &fixture, MemoryStrategy::StagedResidency)
            .unwrap()
            .remove(0)
            .context;
        assert!(matches!(
            registered_safety_check(&spec, &runtime, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }
}
