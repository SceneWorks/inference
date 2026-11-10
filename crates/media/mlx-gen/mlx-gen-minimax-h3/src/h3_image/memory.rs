//! MiniMax-H3-Image's shared-ladder memory contract on MLX/Metal.
//!
//! **What is implemented, and nothing more.**
//!
//! * [`MemoryStrategy::StagedResidency`] — structural: `generate` builds the text encoder, encodes,
//!   forces the context and releases it before the DiT is mapped; denoises and releases the DiT
//!   before the VAE is mapped. There is no co-resident mode, so the descriptor sets
//!   `unconditionally_engages_staged_residency` and every selection stages.
//! * [`MemoryStrategy::BoundedDecode`] — the image VAE's shipped 256 px / 64 px tiled decode
//!   (SW-7089's [`crate::image_vae::bounded_decode_parameters`]). The domain is that singleton:
//!   upstream releases the blended-tile images, so the tile edge is an output-correctness input,
//!   not a lever. Any other geometry is refused at admission and again at decode.
//! * [`MemoryStrategy::BoundedAttention`] and [`MemoryStrategy::BoundedTransformerResidency`] are
//!   `Missing`. H3 video's measured verdicts are not copied: this family's sequence and residency
//!   differ, and the ladder for it is a later item.
//!
//! No calibration is claimed for a production load (nothing has been measured); the weights-free
//! registry surface carries its own static identity so behavior conformance can run.

use mlx_gen::gen_core::{
    safetensors_path_bytes, LoadShape, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryNumericTier, MemoryPhase, MemoryProviderContract,
    MemoryRequestScope, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategySupport, WeightsSource,
};

use crate::h3_image::dit::IMAGE_DIT_COMPONENT;
use crate::h3_image::model::IMAGE_MODEL_ID;
use crate::image_vae::{
    bounded_decode_parameters, IMAGE_DECODE_OVERLAP, IMAGE_DECODE_TILE_EDGE, IMAGE_VAE_COMPONENT,
};
use crate::model::TEXT_ENCODER_COMPONENT;

/// Calibration identity of the weights-free registry conformance walk. Never published by a
/// production load, which reads real bytes and has no measured anchor.
pub const STATIC_CALIBRATION_FINGERPRINT: &str = "minimax-h3-image-mlx-registry-behavior-v1";

/// DiT blocks at the shipped geometry, for the request scope.
const DIT_BLOCKS: usize = 50;

/// Refuse every decode geometry except the shipped one.
pub fn validate_decode_geometry(edge: u32, overlap: u32) -> mlx_gen::gen_core::Result<()> {
    if edge == IMAGE_DECODE_TILE_EDGE && overlap == IMAGE_DECODE_OVERLAP {
        return Ok(());
    }
    Err(mlx_gen::gen_core::Error::Unsupported(format!(
        "{IMAGE_MODEL_ID}: bounded decode admits only the shipped geometry {IMAGE_DECODE_TILE_EDGE}px \
         tile / {IMAGE_DECODE_OVERLAP}px overlap, got {edge}/{overlap}; the released images are \
         the blended-tile ones, so another tile edge changes the picture"
    )))
}

fn phases() -> Vec<MemoryPhase> {
    vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ]
}

/// The bytes each phase maps, from the files on disk (0 for a weights-free spec).
fn asset_facts(spec: &LoadSpec) -> MemoryAssetFacts {
    let Some(root) = mlx_gen::gen_core::resolved_snapshot_root(spec) else {
        return MemoryAssetFacts::default();
    };
    let te_dir = match spec.components.get(TEXT_ENCODER_COMPONENT) {
        Some(WeightsSource::Dir(dir)) => dir.clone(),
        _ => root.join(TEXT_ENCODER_COMPONENT),
    };
    // t2i maps the language-model shards only (`crate::text_encoder::TE_SHARDS`).
    let conditioning: u64 = crate::text_encoder::TE_SHARDS
        .map(|i| safetensors_path_bytes(te_dir.join(format!("model-{i:05}-of-00014.safetensors"))))
        .sum();
    let transformer = safetensors_path_bytes(root.join(IMAGE_DIT_COMPONENT));
    let decoder = safetensors_path_bytes(root.join(IMAGE_VAE_COMPONENT));
    MemoryAssetFacts {
        base_bytes: conditioning
            .saturating_add(transformer)
            .saturating_add(decoder),
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: decoder,
        overlay_bytes: 0,
    }
}

fn architecture_facts(spec: &LoadSpec) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    let dit = crate::dit::config::MiniMaxH3ImageDitConfig::default().base;
    let vae = crate::config::MiniMaxH3VaeConfig::image();
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(dit.num_attention_heads),
        head_dim: mlx_gen::architecture_facts::axis(dit.attention_head_dim),
        transformer_blocks: mlx_gen::architecture_facts::axis(dit.num_layers),
        patch_size: mlx_gen::architecture_facts::axis(dit.patch_size[1]),
        latent_channels: mlx_gen::architecture_facts::axis(vae.latent_channels),
        vae_spatial_scale: mlx_gen::architecture_facts::axis(vae.patch_size),
        // An image is one latent frame; the temporal axis does not apply.
        vae_temporal_scale: None,
        activation_dtype_width: Some(if spec.precision == mlx_gen::Precision::Fp32 {
            mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH
        } else {
            mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH
        }),
    }
}

fn build_contract(
    spec: &LoadSpec,
    asset_facts: MemoryAssetFacts,
    calibration: Option<MemoryCalibrationIdentity>,
) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        IMAGE_MODEL_ID,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident
            | MemoryStrategy::StagedResidency
            | MemoryStrategy::BoundedDecode => MemoryStrategySupport::Implemented,
            MemoryStrategy::BoundedAttention | MemoryStrategy::BoundedTransformerResidency => {
                MemoryStrategySupport::Missing
            }
        };
        if capability.strategy == MemoryStrategy::BoundedDecode {
            capability.parameters = bounded_decode_parameters();
        }
    }
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: phases(),
        synchronized_phase_release: true,
        decode_tiling: true,
        attention_chunking: false,
        transformer_window_materialization: false,
    };
    contract.phase_facts = Some(mlx_gen::gen_core::MemoryPhaseFacts::staged(
        mlx_gen::gen_core::StagedWeightSchedule::ThreeStage,
    ));
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: phases(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::DecodeTileArea,
        ],
    };
    contract.load_shape = LoadShape::EagerMaterialization;
    contract.architecture_facts = architecture_facts(spec);
    contract.asset_facts = asset_facts;
    contract.calibration = calibration;
    contract
}

/// The production contract: real file bytes, no calibration identity (unmeasured).
pub fn contract_for(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract(spec, asset_facts(spec), None))
}

/// The weights-free contract catalog conformance resolves instead of [`contract_for`].
pub fn weights_free_contract(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract(
        spec,
        MemoryAssetFacts::default(),
        Some(MemoryCalibrationIdentity::new(
            STATIC_CALIBRATION_FINGERPRINT,
            LoadShape::EagerMaterialization,
        )),
    ))
}

/// The family's admission predicates the shared pipeline cannot know: the decode geometry, no PiD
/// route, one image per sequence (a batch renders sequentially), one latent frame, a stride-aligned
/// canvas.
pub fn route_gate(context: &MemoryRunContext) -> mlx_gen::gen_core::Result<()> {
    if context.use_pid {
        return Err(mlx_gen::gen_core::Error::Unsupported(format!(
            "{IMAGE_MODEL_ID} has no PiD decode route"
        )));
    }
    if context
        .selection
        .strategy
        .engages(MemoryStrategy::BoundedDecode)
    {
        if let (Some(edge), Some(overlap)) = (
            context.selection.parameters.decode_tile_edge,
            context.selection.parameters.decode_overlap,
        ) {
            validate_decode_geometry(edge, overlap)?;
        }
    }
    let g = context.geometry;
    if g.frames > 1 {
        return Err(mlx_gen::gen_core::Error::Unsupported(format!(
            "{IMAGE_MODEL_ID} renders still images, got {} frames",
            g.frames
        )));
    }
    let stride = mlx_gen::gen_core::h3_image::SIZE_STRIDE;
    if g.width == 0
        || g.height == 0
        || !g.width.is_multiple_of(stride)
        || !g.height.is_multiple_of(stride)
    {
        return Err(mlx_gen::gen_core::Error::Unsupported(format!(
            "{IMAGE_MODEL_ID}: admission geometry {}x{} must be the {stride}px-aligned generation canvas",
            g.width, g.height
        )));
    }
    Ok(())
}

fn profile_route_gate(
    _contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    _tier: MemoryNumericTier,
) -> mlx_gen::gen_core::Result<()> {
    route_gate(context)
}

/// The shared-skeleton profile: requested tier, the route gate above, text-to-image fixtures.
pub(crate) const MEMORY_PROFILE: mlx_gen::gen_core::ProviderMemoryProfile =
    mlx_gen::gen_core::ProviderMemoryProfile {
        gates: mlx_gen::gen_core::MemoryProfileGates::within(profile_route_gate),
        ..mlx_gen::gen_core::ProviderMemoryProfile::STANDARD
    };

/// The loaded generator's safety check.
pub fn safety_check_at_tier(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    mlx_gen::gen_core::standard_safety_check(&MEMORY_PROFILE, contract, context, tier)
}

fn begin_request_with_cleanup(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    mlx_gen::request_scope::begin_scope(
        safety_check_at_tier(contract, tier, context),
        cleanup,
        move || {
            mlx_gen::request_scope::MlxRequestScopeConfig::new(
                IMAGE_MODEL_ID,
                context.geometry,
                contract.generation_memory(&context.selection),
                context.use_pid,
                DIT_BLOCKS,
                |_use_pid, edge, overlap| validate_decode_geometry(edge, overlap),
            )
        },
    )
}

/// The loaded generator's request scope (device cleanup on exit).
pub fn begin_request(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        contract,
        tier,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

fn registered_scope(
    _provider_id: &'static str,
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    tier: MemoryNumericTier,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        contract,
        tier,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

/// The memory-strategy registration paired with the `minimax_h3_image` generator.
pub const IMAGE_MEMORY_REGISTRATION: mlx_gen::gen_core::MemoryRegistration =
    mlx_gen::gen_core::profile_memory_registration!(MEMORY_PROFILE, IMAGE_MODEL_ID, contract_for);

/// The weights-free contract fixture.
pub const IMAGE_MEMORY_CONTRACT_FIXTURE: mlx_gen::gen_core::MemoryContractFixtureRegistration =
    mlx_gen::gen_core::MemoryContractFixtureRegistration {
        provider_id: IMAGE_MODEL_ID,
        contract: weights_free_contract,
        surface_specs: memory_contract_surface_specs,
    };

/// The executable behavior seam every implemented optimized rung must have.
pub const IMAGE_MEMORY_BEHAVIOR: mlx_gen::gen_core::MemoryBehaviorRegistration = mlx_gen::gen_core::profile_memory_behavior_registration!(
    MEMORY_PROFILE,
    IMAGE_MODEL_ID,
    registered_scope
);

/// The surfaces this route publishes: the dense bf16 / fp32 eager loads (no packed tier and no
/// deferred loader exist for this family yet).
fn memory_contract_surface_specs() -> Vec<mlx_gen::gen_core::MemoryContractSurfaceSpec> {
    mlx_gen::gen_core::mlx_memory_contract_surface_specs()
        .into_iter()
        .filter(|s| {
            s.spec.quantize.is_none()
                && s.spec.load_shape == LoadShape::EagerMaterialization
                && s.spec.offload_policy == mlx_gen::gen_core::OffloadPolicy::Resident
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights_free() -> MemoryProviderContract {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()));
        weights_free_contract(&spec).unwrap()
    }

    /// Exactly what is implemented: staged residency and the singleton tiled decode. Rungs 3 and
    /// 4 are `Missing`, not copied from H3 video's measured verdicts.
    #[test]
    fn declares_staged_residency_and_bounded_decode_only() {
        let c = weights_free();
        let support = |s| &c.capability(s).unwrap().support;
        assert_eq!(
            support(MemoryStrategy::StagedResidency),
            &MemoryStrategySupport::Implemented
        );
        assert_eq!(
            support(MemoryStrategy::BoundedDecode),
            &MemoryStrategySupport::Implemented
        );
        assert_eq!(
            support(MemoryStrategy::BoundedAttention),
            &MemoryStrategySupport::Missing
        );
        assert_eq!(
            support(MemoryStrategy::BoundedTransformerResidency),
            &MemoryStrategySupport::Missing
        );
        assert_eq!(
            c.capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .parameters,
            bounded_decode_parameters()
        );
        assert!(c.lifecycle.decode_tiling);
        assert!(!c.lifecycle.transformer_window_materialization);
        assert_eq!(c.provider_id, IMAGE_MODEL_ID);
    }

    /// A production contract claims no calibration: nothing has been measured.
    #[test]
    fn production_contract_is_uncalibrated() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()));
        assert!(contract_for(&spec).unwrap().calibration.is_none());
        assert!(weights_free().calibration.is_some());
    }

    #[test]
    fn only_the_shipped_decode_geometry_is_admitted() {
        validate_decode_geometry(256, 64).unwrap();
        assert!(validate_decode_geometry(512, 64).is_err());
        assert!(validate_decode_geometry(256, 0).is_err());
    }
}
