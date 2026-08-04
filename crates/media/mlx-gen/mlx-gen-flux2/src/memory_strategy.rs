//! FLUX.2-dev edit's provider-side memory-safety contract.
//!
//! The provider already bounds long multi-reference sequences internally with `MemoryConfig::LONG_SEQ`.
//! SceneWorks supplies request geometry, numeric tier, incremental live demand derived from the
//! evidence-owned absolute peak, and the live unified-memory budget. This module validates the
//! provider route and tier, then delegates the canonical budget comparison to `gen-core`;
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

use crate::config::{Flux2Variant, FLUX2_DEV_EDIT_ID};

pub const CALIBRATION_FINGERPRINT: &str = "sc-16593-flux2-dev-edit-evidence-v2";

pub fn contract_for_variant(
    variant: Flux2Variant,
    spec: &LoadSpec,
) -> mlx_gen::Result<Option<MemoryProviderContract>> {
    if variant == Flux2Variant::DevEdit {
        return Ok(Some(build_contract()));
    }
    if matches!(variant, Flux2Variant::Klein9b | Flux2Variant::Klein9bEdit) {
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
    safety_check(
        contract,
        context,
        MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
    )
}

// ---- FLUX.2 Klein shared image-memory ladder (SC-15518) -------------------------------

const KLEIN_STATIC_CALIBRATION: &str = "flux2-klein-static-registry-behavior-v1";
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
    (!axes.is_empty()).then(|| axes.join("-"))
}

fn klein_streamable(spec: &LoadSpec) -> bool {
    spec.offload_policy == OffloadPolicy::Sequential
        && spec.load_shape == LoadShape::DeferredMaterialization
        && spec.quantize.is_none()
        && klein_overlay(spec).is_none()
        && matches!(spec.weights, WeightsSource::Dir(_))
}

fn klein_route(provider_id: &str) -> mlx_gen::gen_core::Result<(MemoryMode, u32)> {
    match provider_id {
        crate::FLUX2_KLEIN_9B_ID => Ok((MemoryMode::TextToImage, 0)),
        crate::FLUX2_KLEIN_9B_EDIT_ID => Ok((MemoryMode::Edit, 1)),
        _ => Err(CoreError::Unsupported(format!(
            "unknown FLUX.2 Klein memory provider {provider_id}"
        ))),
    }
}

fn klein_provider_static(provider_id: &str) -> mlx_gen::gen_core::Result<&'static str> {
    match provider_id {
        crate::FLUX2_KLEIN_9B_ID => Ok(crate::FLUX2_KLEIN_9B_ID),
        crate::FLUX2_KLEIN_9B_EDIT_ID => Ok(crate::FLUX2_KLEIN_9B_EDIT_ID),
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
    klein_route(provider_id)?;
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
    if provider_id == crate::FLUX2_KLEIN_9B_ID {
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
                if provider_id == crate::FLUX2_KLEIN_9B_ID {
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
    let inventory = crate::artifact_inventory::KleinArtifactInventory::verify(spec)?;
    let streamable = inventory.is_some() && klein_streamable(spec);
    let calibration = streamable
        .then(|| {
            klein_calibration_fingerprint(
                provider_id,
                inventory
                    .as_ref()
                    .expect("streamable inventory")
                    .calibration_tag(),
            )
            .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape))
        })
        .transpose()?;
    build_klein_contract(
        provider_id,
        spec,
        crate::model::component_footprint(spec)?,
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
                "{KLEIN_STATIC_CALIBRATION}-{}",
                provider_id.replace('_', "-")
            ),
            spec.load_shape,
        )),
    )
}

pub(crate) fn klein_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        let (expected_mode, minimum_references) = klein_route(&contract.provider_id)?;
        let route_matches = context.mode == expected_mode
            && if minimum_references == 0 {
                context.geometry.reference_count == 0
            } else {
                context.geometry.reference_count >= minimum_references
            };
        if !route_matches {
            return Err(CoreError::Unsupported(format!(
                "{}: memory route does not match the loaded provider mode/reference domain",
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
        ) && (!contract.lifecycle.transformer_window_materialization || !context.has_phases)
        {
            return Err(CoreError::Unsupported(
                "flux2 Klein transformer streaming requires Sequential + DeferredMaterialization"
                    .to_owned(),
            ));
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

pub(crate) fn registered_klein_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    klein_safety_check(spec, contract, context)
}

pub(crate) fn registered_klein_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> mlx_gen::gen_core::Result<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let (mode, reference_count) = klein_route(&contract.provider_id)?;
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
    let route = |use_pid| mlx_gen::gen_core::MemoryBehaviorRoute {
        mode: mode.clone(),
        reference_count,
        use_pid,
        has_phases: spec.offload_policy == OffloadPolicy::Sequential,
        overlay: klein_overlay(spec),
    };
    let mut fixtures = vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(
        mlx_gen::gen_core::standard_memory_behavior_context(
            contract,
            strategy,
            tier,
            route(false),
        )?,
    )];
    if spec.pid.is_some()
        && contract.pid_decode_routes.is_some()
        && contract.engages(strategy, MemoryStrategy::BoundedDecode)
    {
        fixtures.push(mlx_gen::gen_core::MemoryBehaviorFixture::new(
            mlx_gen::gen_core::standard_memory_behavior_context(
                contract,
                strategy,
                tier,
                route(true),
            )?,
        ));
    }
    Ok(fixtures)
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
    if let MemorySafetyDecision::Reject { reason } = klein_safety_check(spec, contract, context) {
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
        let mut overlay = base.clone();
        overlay.adapters.push(mlx_gen::AdapterSpec::new(
            "/nonexistent/adapter.safetensors".into(),
            1.0,
            mlx_gen::AdapterKind::Lora,
        ));
        for spec in [
            base.clone()
                .with_load_shape(LoadShape::EagerMaterialization),
            base.with_quant(Quant::Q4),
            overlay,
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
            has_phases: true,
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
            klein_safety_check(&spec, &contract, &context)
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
            klein_safety_check(&spec, &contract, &context)
        else {
            panic!("base contract admitted PiD without loaded weights");
        };
        assert!(reason.contains("without loaded PiD weights"), "{reason}");
    }
}
