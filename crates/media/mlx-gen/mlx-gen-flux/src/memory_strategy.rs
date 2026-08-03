//! FLUX.1 MLX shared memory-provider contract foundation (SC-15514).
//!
//! This slice exposes the existing two-phase `Residency` lifecycle through the shared provider
//! contract. The clean schnell/dev routes use the shared head-once/tail-tiled native VAE decode and
//! thread the shared bounded-attention kernel through every double- and single-stream block. Control
//! and every loaded overlay remain `Missing` until their additional paths have independent coverage.
//! Production contracts deliberately carry no calibration identity; weights-free registry
//! conformance receives an isolated synthetic identity.

use mlx_gen::attention::{AttentionBudget, AttentionPlan};
use mlx_gen::gen_core::{
    Error as CoreError, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyPrerequisite, MemoryStrategySupport,
    Result as CoreResult, TransformerComponent,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{GenerationRequest, LoadSpec, OffloadPolicy};

const STATIC_CALIBRATION: &str = "flux-one-static-registry-behavior-v1";
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
    let inventory = crate::artifact_inventory::verified_stream_inventory(provider_id, spec);
    memory_strategy_contract_with_inventory(provider_id, spec, inventory.as_ref())
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
pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
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
            format!("{STATIC_CALIBRATION}-{route}"),
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
        if contract
            .calibration
            .as_ref()
            .is_some_and(|identity| identity.fingerprint == MEMORY_CALIBRATION_FINGERPRINT)
            && (context.mode != MemoryMode::TextToImage
                || context.geometry.reference_count != 0
                || context.geometry.width != 1024
                || context.geometry.height != 1024
                || context.geometry.batch != 1
                || context.geometry.frames != 1
                || context.use_pid
                || context.overlay.is_some())
        {
            return Err(CoreError::Unsupported(format!(
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
                .map_err(CoreError::Unsupported)?;
        }
        if contract.engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        ) && (!contract.lifecycle.transformer_window_materialization || !context.has_phases)
        {
            return Err(CoreError::Unsupported(format!(
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
            use_pid: false,
            has_phases: matches!(spec.offload_policy, OffloadPolicy::Sequential),
            overlay: route_overlay(&contract.provider_id, spec),
        },
    )?;
    Ok(vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(context)])
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

    fn write_exact_snapshot(root: &std::path::Path, quant: Option<mlx_gen::Quant>) {
        crate::artifact_inventory::write_test_snapshot(root, quant);
    }

    #[test]
    fn static_contract_declares_native_decode_and_attention_only_for_clean_base_routes() {
        for provider in [
            crate::FLUX1_SCHNELL_ID,
            crate::FLUX1_DEV_ID,
            crate::FLUX1_DEV_CONTROL_ID,
        ] {
            let spec = sequential_spec();
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
    fn production_rung4_needs_verified_exact_inventory_and_unknown_fixture_stays_uncalibrated() {
        let root = std::env::temp_dir().join(format!("flux-rung4-contract-{}", std::process::id()));
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
        std::fs::remove_dir_all(root).ok();
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
    fn every_overlay_keeps_bounded_decode_and_attention_missing() {
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
                spec.control = Some(WeightsSource::File("/control.safetensors".into()));
                spec
            },
            {
                let mut spec = sequential_spec();
                spec.extra_controls
                    .push(WeightsSource::File("/control-2.safetensors".into()));
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
            {
                let mut spec = sequential_spec();
                spec.identity = Some(Default::default());
                spec
            },
            {
                let mut spec = sequential_spec();
                spec.text_encoder = Some(WeightsSource::Dir("/external-text".into()));
                spec
            },
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
        for mutation in 0..4 {
            let mut changed = context.clone();
            match mutation {
                0 => changed.overlay = Some("different".to_owned()),
                1 => changed.geometry.reference_count = 0,
                2 => changed.selection.tier.precision = mlx_gen::Precision::Fp32,
                _ => {
                    changed.load_shape = mlx_gen::LoadShape::DeferredMaterialization;
                }
            }
            assert!(matches!(
                registered_safety_check(&spec, &contract, &changed),
                MemorySafetyDecision::Reject { .. }
            ));
        }
        context.calibration_fingerprint.push_str("-stale");
        assert!(matches!(
            registered_safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn production_unknown_artifact_has_no_calibration_and_rejects_static_context() {
        let root = std::env::temp_dir().join(format!("flux-memory-{}", std::process::id()));
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
        std::fs::remove_dir_all(root).ok();
    }
}
