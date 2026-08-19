//! FLUX.2-dev provider-side memory-safety contracts.
//!
//! Text-to-image and edit share weights and the resident execution path, but they are distinct
//! provider registrations with distinct route gates and calibration identities. The edit provider
//! already bounds long multi-reference sequences internally with `MemoryConfig::LONG_SEQ`.
//! SceneWorks supplies request geometry, numeric tier, incremental live demand derived from each
//! route's evidence-owned absolute peak, and the live unified-memory budget. This module validates
//! the exact provider route and tier, then delegates the canonical budget comparison to `gen-core`;
//! calibration coefficients never live in a provider.

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::gen_core::{
    standard_memory_strategy_safety_check, Error as CoreError, LoadShape, LoadSpec,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyPrerequisite, MemoryStrategySupport, Quant,
    TransformerComponent,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{GenerationRequest, OffloadPolicy, WeightsSource};

use crate::config::{Flux2Variant, FLUX2_DEV_CONTROL_ID, FLUX2_DEV_EDIT_ID, FLUX2_DEV_ID};

pub const CALIBRATION_FINGERPRINT: &str = "sc-16593-flux2-dev-edit-evidence-v2";
pub const DEV_T2I_CALIBRATION_FINGERPRINT: &str = "sc-18218-flux2-dev-t2i-resident-evidence-v1";
pub const DEV_CONTROL_OVERLAY: &str = "control";

pub fn contract_for_variant(
    variant: Flux2Variant,
    spec: &LoadSpec,
) -> mlx_gen::Result<Option<MemoryProviderContract>> {
    if variant == Flux2Variant::Dev {
        return Ok(Some(build_dev_t2i_contract()));
    }
    if variant == Flux2Variant::DevEdit {
        return Ok(Some(build_contract()));
    }
    if matches!(
        variant,
        Flux2Variant::Klein9b | Flux2Variant::Klein9bEdit | Flux2Variant::Klein9bKvEdit
    ) {
        return Ok(Some(klein_contract_for(variant.id(), spec)?));
    }
    Ok(None)
}

fn build_contract() -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        FLUX2_DEV_EDIT_ID,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = LoadShape::EagerMaterialization;
    contract.formula = MemoryFormulaKind::Affine {
        variables: vec![
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::ConditioningTokenCount,
        ],
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        CALIBRATION_FINGERPRINT,
        LoadShape::EagerMaterialization,
    ));
    for capability in &mut contract.strategies {
        if capability.strategy != MemoryStrategy::Resident {
            capability.support = MemoryStrategySupport::Missing;
        }
    }
    contract
}

fn build_dev_t2i_contract() -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        FLUX2_DEV_ID,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = LoadShape::EagerMaterialization;
    contract.formula = MemoryFormulaKind::Affine {
        variables: vec![
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::ConditioningTokenCount,
        ],
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        DEV_T2I_CALIBRATION_FINGERPRINT,
        LoadShape::EagerMaterialization,
    ));
    contract
}

/// Honest pre-calibration contract for the standalone MLX Fun-Controlnet route.
///
/// The provider has a real resident execution path and a registered component-footprint fallback,
/// but no control-specific optimization evidence. Keep every optimized rung Missing and calibration
/// absent rather than borrowing the base Dev receipt. The route/tier safety gate below remains fully
/// load-exact. Asset facts deliberately remain empty: the registry fallback prices the base split,
/// while its consumer adds the control checkpoint. Relabelling that estimate as provider load-exact
/// facts would hide the runtime distinction between an effective packed base and a dense control
/// branch when no explicit quantization request was made.
fn build_dev_control_contract(spec: &LoadSpec) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        FLUX2_DEV_CONTROL_ID,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: false,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: matches!(spec.offload_policy, OffloadPolicy::Sequential),
        ..Default::default()
    };
    contract
}

pub fn dev_t2i_safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    expected_tier: MemoryNumericTier,
) -> MemorySafetyDecision {
    let route_gate = || {
        if context.mode != MemoryMode::TextToImage
            || context.has_reference
            || context.geometry.reference_count != 0
        {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_ID}: memory-safety context must describe reference-free text-to-image"
            )));
        }
        if expected_tier.quant == Some(Quant::Nvfp4) {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_ID}: NVFP4 is not implemented by the MLX provider"
            )));
        }
        Ok(())
    };
    standard_memory_strategy_safety_check(contract, context, Some(expected_tier), Some(&route_gate))
}

pub fn safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    expected_tier: MemoryNumericTier,
) -> MemorySafetyDecision {
    let route_accepted = std::cell::Cell::new(false);
    let route_gate = || {
        if context.mode != MemoryMode::Edit || !context.has_reference {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_EDIT_ID}: memory-safety context must describe a referenced edit"
            )));
        }
        if context.geometry.reference_count < 2 {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_EDIT_ID}: memory-safety context must name at least two references"
            )));
        }
        if expected_tier.quant == Some(Quant::Nvfp4) {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_EDIT_ID}: NVFP4 is not implemented by the MLX provider"
            )));
        }
        route_accepted.set(true);
        Ok(())
    };
    match standard_memory_strategy_safety_check(
        contract,
        context,
        Some(expected_tier),
        Some(&route_gate),
    ) {
        MemorySafetyDecision::Accept => MemorySafetyDecision::Accept,
        MemorySafetyDecision::Reject { reason }
            if route_accepted.get()
                && reason.contains("incremental live demand")
                && reason.contains("exceeds effective budget") =>
        {
            let reference_count = context.geometry.reference_count;
            let gib = 1024.0 * 1024.0 * 1024.0;
            MemorySafetyDecision::Reject {
                reason: format!(
                    "FLUX.2-dev multi-reference edit at {}×{} with {reference_count} reference \
                     images needs ~{} GB of unified memory (with headroom) but this machine has \
                     ~{} GB. Lower the output resolution, use a single reference image, choose a \
                     smaller numeric tier, or run on a Mac with more memory.",
                    context.geometry.width,
                    context.geometry.height,
                    (context
                        .budget
                        .required_total_bytes(context.predicted_peak_bytes)
                        as f64
                        / gib)
                        .round() as i64,
                    (context.budget.total_bytes as f64 / gib).round() as i64,
                ),
            }
        }
        MemorySafetyDecision::Reject { reason } => MemorySafetyDecision::Reject { reason },
    }
}

pub fn dev_control_safety_check(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    expected_tier: MemoryNumericTier,
) -> MemorySafetyDecision {
    let route_gate = || {
        if contract.provider_id != FLUX2_DEV_CONTROL_ID {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_CONTROL_ID}: memory contract resolved the wrong provider route {}",
                contract.provider_id
            )));
        }
        if context.mode != MemoryMode::TextToImage
            || context.has_reference
            || context.geometry.reference_count != 0
            || context.overlay.as_deref() != Some(DEV_CONTROL_OVERLAY)
        {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_CONTROL_ID}: memory-safety context must describe reference-free text-to-image with the control overlay"
            )));
        }
        if context.load_shape != contract.load_shape {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_CONTROL_ID}: memory context load shape does not match the loaded contract"
            )));
        }
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_CONTROL_ID}: PiD decode is not implemented for the control route"
            )));
        }
        if context.has_phases {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_CONTROL_ID}: multi-phase denoise is not implemented for the control route"
            )));
        }
        if expected_tier.quant == Some(Quant::Nvfp4) {
            return Err(CoreError::Unsupported(format!(
                "{FLUX2_DEV_CONTROL_ID}: NVFP4 is not implemented by the MLX provider"
            )));
        }
        Ok(())
    };
    standard_memory_strategy_safety_check(contract, context, Some(expected_tier), Some(&route_gate))
}

pub fn registered_dev_contract(
    _spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract())
}

pub fn registered_dev_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match crate::model::effective_dev_memory_numeric_tier(spec, FLUX2_DEV_EDIT_ID) {
        Ok(tier) => safety_check(contract, context, tier),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub fn registered_dev_t2i_contract(
    _spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_dev_t2i_contract())
}

pub fn registered_dev_t2i_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match crate::model::effective_dev_memory_numeric_tier(spec, FLUX2_DEV_ID) {
        Ok(tier) => dev_t2i_safety_check(contract, context, tier),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub fn registered_dev_control_contract(
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_dev_control_contract(spec))
}

pub fn registered_dev_control_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    if let Err(error) = mlx_gen::require_control(
        spec,
        FLUX2_DEV_CONTROL_ID,
        "FLUX.2-dev-Fun-Controlnet-Union",
    ) {
        return MemorySafetyDecision::Reject {
            reason: error.to_string(),
        };
    }
    match crate::model::effective_dev_memory_numeric_tier(spec, FLUX2_DEV_CONTROL_ID) {
        Ok(tier) => dev_control_safety_check(contract, context, tier),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

// ---- FLUX.2 Klein shared image-memory ladder (SC-15518) -------------------------------

pub const KLEIN_STATIC_BEHAVIOR_FINGERPRINT: &str = "flux2-klein-static-registry-behavior-v2";
pub const KLEIN_MEMORY_CALIBRATION_FINGERPRINT: &str =
    "flux2-klein-9b-bf16-mlx-shared-ladder-t2i-v1";
pub const KLEIN_CALIBRATED_REVISION: &str = "1d36c68041725a14c76566cdf6cea4270b264b03";
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512];
pub const DECODE_OVERLAP: u32 = 128;
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;

fn klein_decode_routes(provider_id: &str) -> mlx_gen::gen_core::Result<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(
        provider_id,
        DECODE_TILE_EDGES.iter().copied(),
        DECODE_OVERLAP,
    )
}

fn klein_overlay(spec: &LoadSpec) -> Option<String> {
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
    if !spec.components.is_empty() {
        axes.push("components");
    }
    (!axes.is_empty()).then(|| axes.join("-"))
}

pub(crate) fn klein_streamable(spec: &LoadSpec) -> bool {
    spec.offload_policy == OffloadPolicy::Sequential
        && spec.load_shape == LoadShape::DeferredMaterialization
        && spec.quantize.is_none()
        && klein_overlay(spec).is_none()
        && matches!(spec.weights, WeightsSource::Dir(_))
}

fn klein_provider_static(provider_id: &str) -> mlx_gen::gen_core::Result<&'static str> {
    match provider_id {
        crate::FLUX2_KLEIN_9B_ID => Ok(crate::FLUX2_KLEIN_9B_ID),
        crate::FLUX2_KLEIN_9B_EDIT_ID => Ok(crate::FLUX2_KLEIN_9B_EDIT_ID),
        crate::FLUX2_KLEIN_9B_KV_EDIT_ID => Ok(crate::FLUX2_KLEIN_9B_KV_EDIT_ID),
        _ => Err(CoreError::Unsupported(format!(
            "unknown FLUX.2 Klein memory provider {provider_id}"
        ))),
    }
}

fn klein_calibration_fingerprint(
    provider_id: &str,
    artifact_tag: &str,
) -> mlx_gen::gen_core::Result<String> {
    let route = match provider_id {
        crate::FLUX2_KLEIN_9B_ID if artifact_tag == "base" => {
            return Ok(KLEIN_MEMORY_CALIBRATION_FINGERPRINT.to_owned());
        }
        crate::FLUX2_KLEIN_9B_ID => "t2i",
        crate::FLUX2_KLEIN_9B_EDIT_ID => "edit",
        crate::FLUX2_KLEIN_9B_KV_EDIT_ID => "kv-edit",
        _ => {
            return Err(CoreError::Unsupported(format!(
                "unknown FLUX.2 Klein memory provider {provider_id}"
            )))
        }
    };
    Ok(format!(
        "flux2-klein-9b-bf16-mlx-shared-ladder-{artifact_tag}-{route}-v1"
    ))
}

fn build_klein_contract(
    provider_id: &str,
    spec: &LoadSpec,
    footprint: mlx_gen::PerComponentBytes,
    streamable: bool,
    calibration: Option<MemoryCalibrationIdentity>,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    let staged = spec.offload_policy == OffloadPolicy::Sequential;
    let clean = klein_overlay(spec).is_none();
    klein_provider_static(provider_id)?;
    let routes = klein_decode_routes(provider_id)?;
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
        decode_tiling: clean,
        attention_chunking: clean,
        transformer_window_materialization: streamable,
    };
    if spec.pid.is_some() {
        contract.pid_decode_routes = Some(mlx_gen::gen_core::MemoryPidDecodeRoutes {
            native: mlx_gen::gen_core::MemoryDecodeRouteDomain {
                tile_edges: routes.native_edges().to_vec(),
                tile_overlap: DECODE_OVERLAP,
            },
            pid: mlx_gen::gen_core::MemoryDecodeRouteDomain {
                tile_edges: mlx_gen_pid::DecodeRoutes::pid_edges(),
                tile_overlap: mlx_gen_pid::DecodeRoutes::pid_overlap(),
            },
        });
    }
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency if staged => MemoryStrategySupport::Implemented,
            MemoryStrategy::BoundedDecode if clean => {
                if contract.pid_decode_routes.is_some() {
                    capability.parameters.decode_tile_edges = routes.published_edges();
                    capability.parameters.decode_overlaps = routes.published_overlaps();
                } else {
                    capability.parameters.decode_tile_edges = routes.native_edges().to_vec();
                    capability.parameters.decode_overlaps = vec![DECODE_OVERLAP];
                }
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedAttention if clean => {
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

pub fn klein_contract_for(
    provider_id: &str,
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    crate::model::validate_klein_load_axes(spec, provider_id)
        .map_err(|error| CoreError::Unsupported(error.to_string()))?;
    let inventory =
        crate::artifact_inventory::KleinArtifactInventory::verify_for_provider(provider_id, spec)?;
    let streamable = inventory.is_some() && klein_streamable(spec);
    let calibration = if streamable {
        let fingerprint = match inventory
            .as_ref()
            .expect("streamable requires an admitted inventory")
            .calibration_tag()
        {
            Some(tag) => klein_calibration_fingerprint(provider_id, tag)?,
            None => format!(
                "{KLEIN_STATIC_BEHAVIOR_FINGERPRINT}-{}",
                provider_id.replace('_', "-")
            ),
        };
        Some(MemoryCalibrationIdentity::new(fingerprint, spec.load_shape))
    } else {
        None
    };
    build_klein_contract(
        provider_id,
        spec,
        match provider_id {
            crate::FLUX2_KLEIN_9B_ID => crate::model::component_footprint(spec)?,
            crate::FLUX2_KLEIN_9B_EDIT_ID => crate::model::klein_edit_component_footprint(spec)?,
            crate::FLUX2_KLEIN_9B_KV_EDIT_ID => {
                crate::model::klein_kv_edit_component_footprint(spec)?
            }
            _ => unreachable!("validate_klein_load_axes rejects unknown providers"),
        },
        streamable,
        calibration,
    )
}

pub fn klein_contract(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    klein_contract_for(crate::FLUX2_KLEIN_9B_ID, spec)
}

pub(crate) fn weights_free_klein_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    build_klein_contract(
        provider_id,
        spec,
        Default::default(),
        klein_streamable(spec),
        Some(MemoryCalibrationIdentity::new(
            format!(
                "{KLEIN_STATIC_BEHAVIOR_FINGERPRINT}-{}",
                provider_id.replace('_', "-")
            ),
            spec.load_shape,
        )),
    )
}

fn surface_selector_matches_spec(
    surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec,
) -> mlx_gen::gen_core::Result<()> {
    use mlx_gen::gen_core::MemoryContractSurfaceTier;

    let tier_matches = match surface.resolved_artifact_tier() {
        MemoryContractSurfaceTier::Bf16 => {
            surface.spec.precision == mlx_gen::Precision::Bf16 && surface.spec.quantize.is_none()
        }
        MemoryContractSurfaceTier::Q4 => surface.spec.quantize == Some(Quant::Q4),
        MemoryContractSurfaceTier::Q8 => surface.spec.quantize == Some(Quant::Q8),
        MemoryContractSurfaceTier::Nvfp4 => false,
    };
    let plain = surface.spec.adapters.is_empty()
        && surface.spec.control.is_none()
        && surface.spec.extra_controls.is_empty()
        && surface.spec.ip_adapter.is_none()
        && surface.spec.identity.is_none()
        && surface.spec.text_encoder.is_none()
        && surface.spec.components.is_empty()
        && surface.spec.pid.is_none();
    if tier_matches
        && surface.spec.precision == mlx_gen::Precision::Bf16
        && plain
        && matches!(surface.spec.weights, WeightsSource::Dir(_))
        && surface.selector.offload_policy == surface.spec.offload_policy
        && surface.selector.load_shape == surface.spec.load_shape
    {
        Ok(())
    } else {
        Err(CoreError::Unsupported(format!(
            "FLUX.2 Klein memory surface selector '{}' does not match its plain registry LoadSpec",
            surface.selector.id()
        )))
    }
}

/// Resolve the finite catalog surface from the already-selected artifact tier.
///
/// Packed Klein turnkeys reach production with `LoadSpec::quantize == None` because only the DiT is
/// packed and the Qwen3 tower deliberately stays dense. The generic witness keeps Q4/Q8 in the
/// synthetic `LoadSpec` solely to make the selector self-checking; this resolver consumes the typed
/// selector and never reinterprets it as load-time quantization.
pub(crate) fn weights_free_klein_surface_contract(
    provider_id: &str,
    surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    use mlx_gen::gen_core::MemoryContractSurfaceTier;

    klein_provider_static(provider_id)?;
    surface_selector_matches_spec(surface)?;
    let supported_tier = matches!(
        surface.resolved_artifact_tier(),
        MemoryContractSurfaceTier::Bf16
            | MemoryContractSurfaceTier::Q4
            | MemoryContractSurfaceTier::Q8
    );
    let streamable = supported_tier
        && surface.spec.offload_policy == OffloadPolicy::Sequential
        && surface.spec.load_shape == LoadShape::DeferredMaterialization;
    build_klein_contract(
        provider_id,
        &surface.spec,
        Default::default(),
        streamable,
        Some(MemoryCalibrationIdentity::new(
            format!(
                "{KLEIN_STATIC_BEHAVIOR_FINGERPRINT}-{}",
                provider_id.replace('_', "-")
            ),
            surface.spec.load_shape,
        )),
    )
}

fn klein_expected_tier(
    provider_id: &str,
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryNumericTier> {
    let quant = match spec.quantize {
        Some(quant) => Some(quant),
        None => {
            if let Some(inventory) =
                crate::artifact_inventory::KleinArtifactInventory::verify_for_provider(
                    provider_id,
                    spec,
                )?
            {
                inventory.resolved_quant()
            } else {
                let WeightsSource::Dir(root) = &spec.weights else {
                    return Err(CoreError::Unsupported(format!(
                        "{provider_id}: FLUX.2 Klein memory routes require a snapshot directory"
                    )));
                };
                match crate::loader::read_component_quant(&root.join("transformer"))
                    .map_err(|error| CoreError::Unsupported(error.to_string()))?
                {
                    Some(quant) if quant.bits == 4 && quant.group_size == 64 => Some(Quant::Q4),
                    Some(quant) if quant.bits == 8 && quant.group_size == 64 => Some(Quant::Q8),
                    Some(quant) => {
                        return Err(CoreError::Unsupported(format!(
                            "{provider_id}: unsupported packed transformer quantization bits={} group_size={}",
                            quant.bits, quant.group_size
                        )))
                    }
                    None => None,
                }
            }
        }
    };
    Ok(MemoryNumericTier {
        precision: spec.precision,
        quant,
        component_precision_floors: &[],
    })
}

pub(crate) fn klein_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    expected_tier: MemoryNumericTier,
) -> MemorySafetyDecision {
    let route_gate = || {
        klein_provider_static(&contract.provider_id)?;
        let reference_count = context.geometry.reference_count;
        let route_matches = match contract.provider_id.as_str() {
            crate::FLUX2_KLEIN_9B_ID => matches!(
                (&context.mode, reference_count),
                (MemoryMode::TextToImage, 0) | (MemoryMode::ImageToImage, 1)
            ),
            crate::FLUX2_KLEIN_9B_EDIT_ID | crate::FLUX2_KLEIN_9B_KV_EDIT_ID => {
                context.mode == MemoryMode::Edit && (1..=8).contains(&reference_count)
            }
            _ => false,
        } && context.has_reference == (reference_count > 0);
        if !route_matches {
            return Err(CoreError::Unsupported(format!(
                "{}: memory route does not match the loaded provider mode/reference domain",
                contract.provider_id
            )));
        }
        if context.has_phases {
            return Err(CoreError::Unsupported(format!(
                "{}: FLUX.2 Klein memory routes are single-phase",
                contract.provider_id
            )));
        }
        if context.use_pid && spec.pid.is_none() {
            return Err(CoreError::Unsupported(format!(
                "{}: PiD route requested without loaded PiD weights",
                contract.provider_id
            )));
        }
        if context.overlay != klein_overlay(spec) {
            return Err(CoreError::Unsupported(
                "flux2 Klein memory overlay does not match the loaded route".to_owned(),
            ));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            klein_decode_routes(&contract.provider_id)?
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
        ) && !contract.lifecycle.transformer_window_materialization
        {
            return Err(CoreError::Unsupported(
                "flux2 Klein transformer streaming requires Sequential + DeferredMaterialization"
                    .to_owned(),
            ));
        }
        Ok(())
    };
    standard_memory_strategy_safety_check(contract, context, Some(expected_tier), Some(&route_gate))
}

pub(crate) fn registered_klein_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match klein_expected_tier(&contract.provider_id, spec) {
        Ok(tier) => klein_safety_check(spec, contract, context, tier),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub(crate) fn registered_klein_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> mlx_gen::gen_core::Result<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let tier = klein_expected_tier(&contract.provider_id, spec)?;
    let routes = match contract.provider_id.as_str() {
        crate::FLUX2_KLEIN_9B_ID => {
            vec![(MemoryMode::TextToImage, 0), (MemoryMode::ImageToImage, 1)]
        }
        crate::FLUX2_KLEIN_9B_EDIT_ID | crate::FLUX2_KLEIN_9B_KV_EDIT_ID => (1..=8)
            .map(|references| (MemoryMode::Edit, references))
            .collect(),
        provider_id => {
            return Err(CoreError::Unsupported(format!(
                "unknown FLUX.2 Klein memory provider {provider_id}"
            )))
        }
    };
    let permits_pid = spec.pid.is_some()
        && contract.pid_decode_routes.is_some()
        && contract.engages(strategy, MemoryStrategy::BoundedDecode);
    let mut fixtures = Vec::new();
    for (mode, reference_count) in routes {
        for use_pid in [false, true]
            .into_iter()
            .filter(|use_pid| !*use_pid || permits_pid)
        {
            let route = mlx_gen::gen_core::MemoryBehaviorRoute {
                mode: mode.clone(),
                reference_count,
                use_pid,
                has_phases: false,
                overlay: klein_overlay(spec),
            };
            let context = mlx_gen::gen_core::standard_memory_behavior_context(
                contract, strategy, tier, route,
            )?;
            fixtures.push(executable_klein_fixture(context));
        }
    }
    Ok(fixtures)
}

fn executable_klein_fixture(context: MemoryRunContext) -> mlx_gen::gen_core::MemoryBehaviorFixture {
    let mut fixture = mlx_gen::gen_core::MemoryBehaviorFixture::new(context);
    fixture.request.prompt = "weights-free FLUX.2 Klein memory behavior".to_owned();
    let reference = || mlx_gen::media::Image {
        width: 1,
        height: 1,
        pixels: vec![0, 0, 0],
    };
    fixture.request.conditioning = match (
        &fixture.context.mode,
        fixture.context.geometry.reference_count,
    ) {
        (MemoryMode::TextToImage, 0) => Vec::new(),
        (MemoryMode::ImageToImage, 1) => vec![mlx_gen::Conditioning::Reference {
            image: reference(),
            strength: Some(1.0),
        }],
        (MemoryMode::Edit, 1) => vec![mlx_gen::Conditioning::Reference {
            image: reference(),
            strength: None,
        }],
        (MemoryMode::Edit, count @ 2..=8) => vec![mlx_gen::Conditioning::MultiReference {
            images: (0..count).map(|_| reference()).collect(),
        }],
        _ => unreachable!("provider-owned route validation constructs only executable fixtures"),
    };
    fixture.request.phases = None;
    fixture
}

pub(crate) fn registered_klein_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_klein_request_with_cleanup(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub(crate) fn begin_klein_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_klein_request_with_cleanup(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

fn begin_klein_request_with_cleanup(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let expected_tier = klein_expected_tier(&contract.provider_id, spec)?;
    if let MemorySafetyDecision::Reject { reason } =
        klein_safety_check(spec, contract, context, expected_tier)
    {
        return Err(CoreError::Unsupported(reason));
    }
    let routes = klein_decode_routes(&contract.provider_id)?;
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        klein_provider_static(&contract.provider_id)?,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        32,
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

pub(crate) fn attention_plan(req: &GenerationRequest) -> AttentionPlan<'_> {
    match req.memory {
        Some(memory) if memory.chunk_attention => {
            AttentionPlan::budgeted(AttentionBudget::CONSTRAINED).with_cancel(&req.cancel)
        }
        _ => AttentionPlan::UNBOUNDED,
    }
}

pub(crate) fn transformer_window(req: &GenerationRequest) -> mlx_gen::Result<Option<usize>> {
    let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
        return Ok(None);
    };
    if memory
        .transformer_window_component
        .unwrap_or(TransformerComponent::Dit)
        != TransformerComponent::Dit
    {
        return Err(mlx_gen::Error::Unsupported(
            "flux2 Klein supports only the DiT transformer window component".to_owned(),
        ));
    }
    Ok(Some(
        memory
            .transformer_window_size
            .unwrap_or(TRANSFORMER_WINDOW_SIZE) as usize,
    ))
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryGeometry, MemoryNumericTier, MemorySelection,
    };
    use mlx_gen::{Precision, Quant};

    fn context(total_gb: f64) -> MemoryRunContext {
        let contract = build_contract();
        let calibration = contract.calibration.expect("calibration");
        let bytes = |gb: f64| (gb * 1024.0 * 1024.0 * 1024.0).round() as u64;
        MemoryRunContext {
            optimization_authority: mlx_gen::gen_core::MemoryOptimizationAuthority::Calibrated,
            selection: MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: Default::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
            },
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint,
            load_shape: calibration.load_shape,
            mode: MemoryMode::Edit,
            has_reference: true,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 2,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: bytes(total_gb),
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: bytes(81.0),
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "worker-owned-exact-evidence".to_owned(),
        }
    }

    fn dev_t2i_context(total_gb: f64) -> MemoryRunContext {
        let contract = build_dev_t2i_contract();
        let calibration = contract.calibration.expect("calibration");
        let mut context = context(total_gb);
        context.calibration_abi = calibration.abi;
        context.calibration_fingerprint = calibration.fingerprint;
        context.load_shape = calibration.load_shape;
        context.mode = MemoryMode::TextToImage;
        context.has_reference = false;
        context.geometry.reference_count = 0;
        context.predicted_peak_bytes = context.budget.total_bytes;
        context
    }

    #[test]
    fn dev_t2i_contract_is_distinct_conforming_and_resident_only() {
        let t2i = build_dev_t2i_contract();
        let edit = build_contract();

        assert_eq!(t2i.provider_id, FLUX2_DEV_ID);
        assert_eq!(edit.provider_id, FLUX2_DEV_EDIT_ID);
        assert_eq!(
            t2i.calibration.as_ref().unwrap().fingerprint,
            DEV_T2I_CALIBRATION_FINGERPRINT
        );
        assert_ne!(t2i.calibration, edit.calibration);
        assert!(t2i.conformance_errors().is_empty());
        assert!(t2i.strategies.iter().all(|capability| {
            capability.strategy == MemoryStrategy::Resident
                && capability.support == MemoryStrategySupport::Implemented
                || capability.strategy != MemoryStrategy::Resident
                    && capability.support == MemoryStrategySupport::Missing
        }));

        let mut non_resident = dev_t2i_context(96.0);
        non_resident.selection.strategy = MemoryStrategy::StagedResidency;
        let MemorySafetyDecision::Reject { reason } =
            dev_t2i_safety_check(&t2i, &non_resident, non_resident.selection.tier)
        else {
            panic!("a missing non-resident rung must reject");
        };
        assert!(
            reason.contains("cannot execute StagedResidency"),
            "{reason}"
        );
    }

    #[test]
    fn dev_t2i_route_gate_accepts_only_reference_free_text_to_image() {
        let contract = build_dev_t2i_contract();
        let exact = dev_t2i_context(96.0);
        assert_eq!(
            dev_t2i_safety_check(&contract, &exact, exact.selection.tier),
            MemorySafetyDecision::Accept
        );

        let mut edit_shaped = exact.clone();
        edit_shaped.mode = MemoryMode::Edit;
        edit_shaped.has_reference = true;
        edit_shaped.geometry.reference_count = 2;
        let MemorySafetyDecision::Reject { reason } =
            dev_t2i_safety_check(&contract, &edit_shaped, edit_shaped.selection.tier)
        else {
            panic!("the edit route must not inherit the T2I contract");
        };
        assert!(reason.contains("reference-free text-to-image"), "{reason}");

        let mut referenced_t2i = exact.clone();
        referenced_t2i.has_reference = true;
        referenced_t2i.geometry.reference_count = 1;
        let MemorySafetyDecision::Reject { reason } =
            dev_t2i_safety_check(&contract, &referenced_t2i, referenced_t2i.selection.tier)
        else {
            panic!("a referenced request must not enter the base T2I evidence lane");
        };
        assert!(reason.contains("reference-free text-to-image"), "{reason}");
    }

    #[test]
    fn dev_t2i_handshake_numeric_tier_and_quantization_fail_closed() {
        let contract = build_dev_t2i_contract();
        let exact = dev_t2i_context(96.0);

        let mut stale = exact.clone();
        stale.calibration_fingerprint = "stale-flux2-dev-t2i".to_owned();
        let MemorySafetyDecision::Reject { reason } =
            dev_t2i_safety_check(&contract, &stale, stale.selection.tier)
        else {
            panic!("stale T2I evidence must reject");
        };
        assert!(
            reason.contains("calibration handshake mismatch"),
            "{reason}"
        );

        let q8 = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q8),
            component_precision_floors: &[],
        };
        let MemorySafetyDecision::Reject { reason } = dev_t2i_safety_check(&contract, &exact, q8)
        else {
            panic!("the selected tier must match the loaded T2I artifact");
        };
        assert!(reason.contains("does not match loaded tier"), "{reason}");

        let mut nvfp4 = exact;
        nvfp4.selection.tier.quant = Some(Quant::Nvfp4);
        let MemorySafetyDecision::Reject { reason } =
            dev_t2i_safety_check(&contract, &nvfp4, nvfp4.selection.tier)
        else {
            panic!("the unwired NVFP4 tier must reject");
        };
        assert!(reason.contains("NVFP4 is not implemented"), "{reason}");
    }

    #[test]
    fn provider_safety_uses_the_caller_owned_peak_without_recomputing_it() {
        let contract = build_contract();
        let mut exact = context(81.0);
        assert_eq!(
            safety_check(&contract, &exact, exact.selection.tier),
            MemorySafetyDecision::Accept
        );
        exact.budget.total_bytes -= 1;
        assert!(matches!(
            safety_check(&contract, &exact, exact.selection.tier),
            MemorySafetyDecision::Reject { .. }
        ));

        let mut four = context(81.0);
        four.geometry.reference_count = 4;
        assert_eq!(
            safety_check(&contract, &four, four.selection.tier),
            MemorySafetyDecision::Accept
        );
    }

    #[test]
    fn provider_safety_rejects_a_stale_calibration_identity() {
        let contract = build_contract();
        let mut stale = context(96.0);
        stale.calibration_fingerprint = "stale".to_owned();
        assert!(matches!(
            safety_check(&contract, &stale, stale.selection.tier),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn provider_contract_quarantines_structured_overlays_and_false_reference_summaries() {
        let contract = build_contract();
        assert!(contract.strategies.iter().all(|capability| {
            capability.strategy == MemoryStrategy::Resident
                || capability.support == MemoryStrategySupport::Missing
        }));

        let mut structured_overlay = context(128.0);
        structured_overlay.overlay = Some("references=2".to_owned());
        let MemorySafetyDecision::Reject { reason } = safety_check(
            &contract,
            &structured_overlay,
            structured_overlay.selection.tier,
        ) else {
            panic!("structured overlay data must reject");
        };
        assert!(reason.contains("overlay is an identity axis"), "{reason}");

        let mut inconsistent = context(128.0);
        inconsistent.has_reference = false;
        let MemorySafetyDecision::Reject { reason } =
            safety_check(&contract, &inconsistent, inconsistent.selection.tier)
        else {
            panic!("inconsistent compatibility summary must reject");
        };
        assert!(
            reason.contains("inconsistent with reference_count=2"),
            "{reason}"
        );
    }

    #[test]
    fn provider_safety_owns_tier_identity_but_not_tier_peak_estimation() {
        let contract = build_contract();
        let q4_tier = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        };
        for quant in [Some(Quant::Q8), None] {
            let mut larger_tier = context(128.0);
            larger_tier.selection.tier.quant = quant;
            let expected_tier = larger_tier.selection.tier;
            assert_eq!(
                safety_check(&contract, &larger_tier, expected_tier),
                MemorySafetyDecision::Accept
            );
            assert!(matches!(
                safety_check(&contract, &larger_tier, q4_tier),
                MemorySafetyDecision::Reject { .. }
            ));
        }
    }

    #[test]
    fn shared_rejections_keep_their_reason_before_provider_policy_and_budget_advice() {
        let contract = build_contract();

        let mut stale_and_wrong_route = context(1.0);
        stale_and_wrong_route.calibration_fingerprint = "stale".to_owned();
        stale_and_wrong_route.mode = MemoryMode::TextToImage;
        let MemorySafetyDecision::Reject { reason } = safety_check(
            &contract,
            &stale_and_wrong_route,
            stale_and_wrong_route.selection.tier,
        ) else {
            panic!("stale handshake must reject");
        };
        assert!(
            reason.contains("calibration handshake mismatch"),
            "{reason}"
        );
        assert!(!reason.contains("Lower the output resolution"), "{reason}");

        let mut wrong_tier = context(1.0);
        wrong_tier.selection.tier.quant = Some(Quant::Q8);
        let q4 = MemoryNumericTier {
            precision: Precision::Bf16,
            quant: Some(Quant::Q4),
            component_precision_floors: &[],
        };
        let MemorySafetyDecision::Reject { reason } = safety_check(&contract, &wrong_tier, q4)
        else {
            panic!("wrong tier must reject");
        };
        assert!(reason.contains("does not match loaded tier"), "{reason}");
        assert!(!reason.contains("Lower the output resolution"), "{reason}");

        let mut invalid_selection = context(1.0);
        invalid_selection.selection.parameters.decode_tile_edge = Some(512);
        let MemorySafetyDecision::Reject { reason } = safety_check(
            &contract,
            &invalid_selection,
            invalid_selection.selection.tier,
        ) else {
            panic!("invalid selection must reject");
        };
        assert!(reason.contains("decode_tile_edge"), "{reason}");
        assert!(!reason.contains("Lower the output resolution"), "{reason}");

        let mut wrong_route = context(1.0);
        wrong_route.mode = MemoryMode::TextToImage;
        let MemorySafetyDecision::Reject { reason } =
            safety_check(&contract, &wrong_route, wrong_route.selection.tier)
        else {
            panic!("provider route policy must reject");
        };
        assert!(reason.contains("referenced edit"), "{reason}");
        assert!(!reason.contains("Lower the output resolution"), "{reason}");

        let admitted_route = context(1.0);
        let MemorySafetyDecision::Reject { reason } =
            safety_check(&contract, &admitted_route, admitted_route.selection.tier)
        else {
            panic!("under-budget request must reject");
        };
        assert!(reason.contains("Lower the output resolution"), "{reason}");
    }

    #[test]
    fn klein_contract_declares_every_shared_rung_for_the_exact_structural_route() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        let contract = build_klein_contract(
            crate::FLUX2_KLEIN_9B_ID,
            &spec,
            Default::default(),
            true,
            Some(MemoryCalibrationIdentity::new(
                "fixture-v1",
                spec.load_shape,
            )),
        )
        .unwrap();
        assert!(
            contract.conformance_errors().is_empty(),
            "{:?}",
            contract.conformance_errors()
        );
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "{strategy:?}"
            );
        }
        let transformer = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(transformer.parameters.transformer_window_sizes, [1]);
        assert_eq!(
            transformer.parameters.transformer_window_components,
            [TransformerComponent::Dit]
        );
    }

    #[test]
    fn klein_rung_four_fails_closed_for_eager_quantized_and_overlay_loads() {
        let base = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        let lora = base.clone().with_adapters(vec![mlx_gen::AdapterSpec::new(
            "/nonexistent/lora.safetensors".into(),
            1.0,
            mlx_gen::AdapterKind::Lora,
        )]);
        let lokr = base.clone().with_adapters(vec![mlx_gen::AdapterSpec::new(
            "/nonexistent/lokr.safetensors".into(),
            1.0,
            mlx_gen::AdapterKind::Lokr,
        )]);
        let external_te = base
            .clone()
            .with_text_encoder(WeightsSource::Dir("/nonexistent/external-te".into()));
        for spec in [
            base.clone()
                .with_load_shape(LoadShape::EagerMaterialization),
            base.with_quant(Quant::Q4),
            lora,
            lokr,
            external_te,
        ] {
            let contract = build_klein_contract(
                crate::FLUX2_KLEIN_9B_ID,
                &spec,
                Default::default(),
                klein_streamable(&spec),
                None,
            )
            .unwrap();
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
    fn klein_selector_surfaces_are_exact_and_fail_closed_on_axis_mutation() {
        let providers = [
            crate::FLUX2_KLEIN_9B_ID,
            crate::FLUX2_KLEIN_9B_EDIT_ID,
            crate::FLUX2_KLEIN_9B_KV_EDIT_ID,
        ];
        for provider_id in providers {
            let mut implemented = 0;
            for surface in mlx_gen::gen_core::mlx_memory_contract_surface_specs() {
                let contract = weights_free_klein_surface_contract(provider_id, &surface).unwrap();
                let expected = surface.spec.offload_policy == OffloadPolicy::Sequential
                    && surface.spec.load_shape == LoadShape::DeferredMaterialization;
                assert_eq!(
                    contract
                        .capability(MemoryStrategy::BoundedTransformerResidency)
                        .unwrap()
                        .support,
                    if expected {
                        MemoryStrategySupport::Implemented
                    } else {
                        MemoryStrategySupport::Missing
                    },
                    "{provider_id}: {}",
                    surface.selector.id()
                );
                implemented += usize::from(expected);
                assert_eq!(contract.asset_facts, Default::default());
            }
            assert_eq!(implemented, 3, "{provider_id}");
        }

        let q4 = mlx_gen::gen_core::mlx_memory_contract_surface_specs()
            .into_iter()
            .find(|surface| {
                surface.resolved_artifact_tier() == mlx_gen::gen_core::MemoryContractSurfaceTier::Q4
                    && surface.spec.offload_policy == OffloadPolicy::Sequential
                    && surface.spec.load_shape == LoadShape::DeferredMaterialization
            })
            .unwrap();
        let mut mutations = Vec::new();
        let copy = || mlx_gen::gen_core::MemoryContractSurfaceSpec {
            selector: q4.selector,
            spec: q4.spec.clone(),
        };
        let mut tier = copy();
        tier.spec.quantize = Some(Quant::Q8);
        mutations.push(tier);
        let mut source = copy();
        source.spec.weights = WeightsSource::File("/klein.safetensors".into());
        mutations.push(source);
        let mut precision = copy();
        precision.spec.precision = Precision::Fp32;
        mutations.push(precision);
        let mut control = copy();
        control.spec.control = Some(WeightsSource::File("/control.safetensors".into()));
        mutations.push(control);
        let mut component = copy();
        component.spec.components.insert(
            "unknown".to_owned(),
            WeightsSource::File("/component.safetensors".into()),
        );
        mutations.push(component);
        let mut pid = copy();
        pid.spec = pid.spec.with_pid(
            WeightsSource::File("/pid.safetensors".into()),
            WeightsSource::Dir("/gemma".into()),
        );
        mutations.push(pid);
        for provider_id in providers {
            for mutation in &mutations {
                assert!(
                    weights_free_klein_surface_contract(provider_id, mutation).is_err(),
                    "{provider_id} admitted mutated selector {}",
                    mutation.selector.id()
                );
            }
        }
        assert!(weights_free_klein_surface_contract("flux2_klein_alias", &q4).is_err());
    }

    #[test]
    fn klein_behavior_routes_are_executable_reachable_and_fail_closed() {
        let q4 = mlx_gen::gen_core::mlx_memory_contract_surface_specs()
            .into_iter()
            .find(|surface| {
                surface.resolved_artifact_tier() == mlx_gen::gen_core::MemoryContractSurfaceTier::Q4
                    && surface.spec.offload_policy == OffloadPolicy::Sequential
                    && surface.spec.load_shape == LoadShape::DeferredMaterialization
            })
            .unwrap();
        for provider_id in [
            crate::FLUX2_KLEIN_9B_ID,
            crate::FLUX2_KLEIN_9B_EDIT_ID,
            crate::FLUX2_KLEIN_9B_KV_EDIT_ID,
        ] {
            let spec = q4.spec.clone().with_pid(
                WeightsSource::File("/pid.safetensors".into()),
                WeightsSource::Dir("/gemma".into()),
            );
            let contract = build_klein_contract(
                provider_id,
                &spec,
                Default::default(),
                true,
                Some(MemoryCalibrationIdentity::new(
                    format!(
                        "{KLEIN_STATIC_BEHAVIOR_FINGERPRINT}-{}",
                        provider_id.replace('_', "-")
                    ),
                    spec.load_shape,
                )),
            )
            .unwrap();
            let fixtures = registered_klein_fixture(
                &spec,
                &contract,
                MemoryStrategy::BoundedTransformerResidency,
            )
            .unwrap();
            let expected_routes = if provider_id == crate::FLUX2_KLEIN_9B_ID {
                4
            } else {
                16
            };
            assert_eq!(fixtures.len(), expected_routes, "{provider_id}");
            assert!(fixtures.iter().any(|fixture| fixture.context.use_pid));
            assert!(fixtures.iter().any(|fixture| !fixture.context.use_pid));

            let (is_edit, is_kv) = (
                provider_id != crate::FLUX2_KLEIN_9B_ID,
                provider_id == crate::FLUX2_KLEIN_9B_KV_EDIT_ID,
            );
            let descriptor = match provider_id {
                crate::FLUX2_KLEIN_9B_ID => crate::model::descriptor_klein_9b(),
                crate::FLUX2_KLEIN_9B_EDIT_ID => crate::model::descriptor_klein_9b_edit(),
                crate::FLUX2_KLEIN_9B_KV_EDIT_ID => crate::model::descriptor_klein_9b_kv_edit(),
                _ => unreachable!(),
            };
            for fixture in &fixtures {
                assert_eq!(
                    registered_klein_safety_check(&spec, &contract, &fixture.context),
                    MemorySafetyDecision::Accept,
                    "{provider_id}: {:?}",
                    fixture.context
                );
                let mut scope = registered_klein_begin_request(&spec, &contract, &fixture.context)
                    .unwrap()
                    .expect("optimized Klein fixture creates a request scope");
                let mut configured = fixture.request.clone();
                scope.configure_request(&mut configured).unwrap();
                let memory = configured.memory.expect("request memory controls");
                assert!(memory.stage_residency);
                assert!(memory.tile_vae_decode);
                assert!(memory.chunk_attention);
                assert!(memory.stream_transformer_blocks);
                assert_eq!(
                    memory.transformer_window_size,
                    Some(TRANSFORMER_WINDOW_SIZE)
                );
                assert_eq!(memory.attention_chunk_size, Some(ATTENTION_CHUNK_SIZE));
                if fixture.context.use_pid {
                    assert_eq!(memory.decode_tile_edge, Some(2048));
                    assert_eq!(memory.decode_overlap, Some(256));
                } else {
                    assert!(memory
                        .decode_tile_edge
                        .is_some_and(|edge| DECODE_TILE_EDGES.contains(&edge)));
                    assert_eq!(memory.decode_overlap, Some(DECODE_OVERLAP));
                }
                crate::model::validate_request(&descriptor, is_edit, is_kv, &fixture.request)
                    .unwrap_or_else(|error| {
                        panic!("{provider_id} fixture is not executable: {error}")
                    });
                assert_eq!(fixture.request.use_pid, fixture.context.use_pid);
                assert!(fixture.request.phases.is_none());
                assert!(!fixture.context.has_phases);
            }

            let exact = &fixtures[0].context;
            let mut mutations = Vec::new();
            let mut mode = exact.clone();
            mode.mode = if is_edit {
                MemoryMode::ImageToImage
            } else {
                MemoryMode::Edit
            };
            mutations.push(mode);
            let mut references = exact.clone();
            references.geometry.reference_count = if is_edit { 9 } else { 2 };
            references.has_reference = true;
            mutations.push(references);
            let mut phases = exact.clone();
            phases.has_phases = true;
            mutations.push(phases);
            let mut overlay = exact.clone();
            overlay.overlay = Some("references:1".to_owned());
            mutations.push(overlay);
            let mut tier = exact.clone();
            tier.selection.tier.quant = Some(Quant::Q8);
            mutations.push(tier);
            for mutation in mutations {
                assert!(matches!(
                    registered_klein_safety_check(&spec, &contract, &mutation),
                    MemorySafetyDecision::Reject { .. }
                ));
                assert!(registered_klein_begin_request(&spec, &contract, &mutation).is_err());
            }
        }
    }

    #[test]
    fn klein_admission_binds_provider_route_and_loaded_pid() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        let contract = weights_free_klein_contract(crate::FLUX2_KLEIN_9B_ID, &spec).unwrap();
        let tier = MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        };
        let route = mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        };
        let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::Resident,
            tier,
            route,
        )
        .unwrap();

        context.mode = MemoryMode::Edit;
        context.has_reference = true;
        context.geometry.reference_count = 1;
        let MemorySafetyDecision::Reject { reason } =
            klein_safety_check(&spec, &contract, &context, tier)
        else {
            panic!("base contract admitted an edit route");
        };
        assert!(
            reason.contains("provider mode/reference domain"),
            "{reason}"
        );

        context.mode = MemoryMode::TextToImage;
        context.has_reference = false;
        context.geometry.reference_count = 0;
        context.use_pid = true;
        let MemorySafetyDecision::Reject { reason } =
            klein_safety_check(&spec, &contract, &context, tier)
        else {
            panic!("base contract admitted PiD without loaded weights");
        };
        assert!(reason.contains("without loaded PiD weights"), "{reason}");
    }

    #[test]
    fn klein_contract_advertises_pid_decode_only_when_pid_weights_are_loaded() {
        let base = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        let native = weights_free_klein_contract(crate::FLUX2_KLEIN_9B_ID, &base).unwrap();
        assert!(native.pid_decode_routes.is_none());
        assert_eq!(
            native
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .parameters
                .decode_tile_edges,
            DECODE_TILE_EDGES
        );

        let pid_spec = base.with_pid(
            WeightsSource::File("/nonexistent/pid.safetensors".into()),
            WeightsSource::Dir("/nonexistent/gemma".into()),
        );
        let pid = weights_free_klein_contract(crate::FLUX2_KLEIN_9B_ID, &pid_spec).unwrap();
        assert!(pid.pid_decode_routes.is_some());
        let fixtures =
            registered_klein_fixture(&pid_spec, &pid, MemoryStrategy::BoundedDecode).unwrap();
        assert!(fixtures.iter().any(|fixture| !fixture.context.use_pid));
        assert!(fixtures.iter().any(|fixture| fixture.context.use_pid));
    }
}
