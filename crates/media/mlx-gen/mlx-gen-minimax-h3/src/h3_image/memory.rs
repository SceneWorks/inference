//! MiniMax-H3-Image's shared-ladder memory contract on MLX/Metal.
//!
//! | rung | strategy | status |
//! |---|---|---|
//! | 0 | [`MemoryStrategy::Resident`] | implemented — the baseline; the generator always stages, so a resident admission over-provisions |
//! | 1 | [`MemoryStrategy::StagedResidency`] | implemented, structural — the text encoder is built, encodes, is forced and released before the DiT maps; the DiT denoises and is released before the VAE maps. `unconditionally_engages_staged_residency` |
//! | 2 | [`MemoryStrategy::BoundedDecode`] | implemented — SW-7089's 256 px / 64 px tiled decode ([`crate::image_vae::bounded_decode_parameters`]), a singleton domain because the tile edge changes the picture |
//! | 3 | [`MemoryStrategy::BoundedAttention`] | **structurally not applicable** — see below |
//! | 4 | [`MemoryStrategy::BoundedTransformerResidency`] | implemented on a [`LoadShape::DeferredMaterialization`] load for **both** transformers, window 1 (SW-7094) |
//!
//! # Rung 4 (SW-7094)
//!
//! The DiT streams through H3's [`crate::block_stream::DitBlockStream`]
//! ([`crate::h3_image::MiniMaxH3ImageDit::load_dir_deferred`],
//! [`crate::h3_image::ImageDenoiser::new_windowed`]): each step rebuilds the 50 block bodies one
//! window at a time, and the AdaLN projections are read once per request, also one window at a
//! time, by H3's `precompute_adaln_windowed`. The I/O projections, the six per-step output heads
//! (`pdd_proj_out.{0..6}`, 12.4 MB of float32) and the token refiner stay resident. The text
//! encoder is H3's own deferred Qwen3-VL encoder, windowed over its 50 tapped layers. LoRA factors
//! stay resident as the stream's overlay and are re-applied to each rebuilt block (SW-7096), so
//! the overlay is charged in full at every rung. Rung 4 is `Missing` on a resident load: its
//! shared prerequisite is the load shape.
//!
//! The window domain is H3's measured singleton ([`IMAGE_TRANSFORMER_WINDOW_SIZE`] = 1): the same
//! block stack and the same encoder, whose residency floor H3 measured at window 1 on every tier
//! (`crate::memory_strategy::TRANSFORMER_WINDOW_SIZE`). No image-family measurement is recorded
//! here; the family's anchors are measured once at the end of the feature.
//!
//! # Rung 3 is not applicable on MLX, and that verdict is re-taken at image geometry
//!
//! MLX's fused `scaled_dot_product_attention` never materializes the `[B, H, S, S]` score tensor
//! rung 3 bounds — a property of the kernel, not of the sequence length. H3 measured it at
//! `S = 104_030` (sc-18661, [`crate::memory_strategy::RUNG3_MEASURED_PEAK_DELTAS`]): every chunked
//! arm costs denoise peak. This family's sequences are shorter (16,384 image rows at 16 MP plus at
//! most 4,096 prompt tokens, plus up to 16 references' rows), so they sit inside the range H3
//! measured. The verdict is held to an image-geometry measurement by
//! `tests/h3_image_bounded_attention_real.rs`, which runs the real q4 image DiT at 2048² and 4096²
//! on both chunk axes and fails if any chunked arm saves more than 0.5 % of peak — the condition
//! under which this
//! declaration must become an implementation.
//!
//! # Admission
//!
//! [`crate::h3_image::admission`] prices a request per phase, host and GPU separately, at each
//! rung, and walks the ladder to the first rung that fits — so a 16 MP request that does not fit
//! resident is admitted streamed. The contract publishes the same weights as
//! [`mlx_gen::gen_core::MemoryPhaseFacts::transformer_stream`] (a resident trunk plus one stack of
//! 50 per-block byte counts), so a consumer's generic phase estimate sees the streamed DiT too.
//! Every figure is derived from tensor headers and architecture; none is measured.

use mlx_gen::gen_core::{
    LoadShape, LoadSpec, MemoryAssetFacts, MemoryBackendRealization, MemoryCalibrationIdentity,
    MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategySupport, StreamedWeightFacts,
    TransformerComponent,
};

use crate::h3_image::admission::ImageMemoryFacts;
use crate::h3_image::model::IMAGE_MODEL_ID;
use crate::image_vae::{bounded_decode_parameters, IMAGE_DECODE_OVERLAP, IMAGE_DECODE_TILE_EDGE};

/// Calibration identity of the weights-free registry conformance walk. Never published by a
/// production load, which reads real bytes and has no measured anchor.
pub const STATIC_CALIBRATION_FINGERPRINT: &str = "minimax-h3-image-mlx-registry-behavior-v1";

/// DiT blocks at the shipped geometry, for the request scope.
const DIT_BLOCKS: usize = 50;

/// Rung 4's window domain: one block (one encoder layer) at a time — H3's measured residency
/// floor for this same block stack and encoder (`crate::memory_strategy::TRANSFORMER_WINDOW_SIZE`).
pub const IMAGE_TRANSFORMER_WINDOW_SIZE: u32 = 1;

/// Contract-stable id of the evictable AdaLN sub-stack (`transformer_blocks.*.adaln_proj`).
const ADALN_COMPONENT: &str = "dit_adaln_proj_stack";

/// Bytes of the modulation table the AdaLN precompute keeps: `MODULATION_PARAMS · rows · hidden ·
/// blocks` bf16 elements, `rows = 3 · (6 step timesteps + the edit references' 0.999)`. DERIVED
/// from `ImageDenoiser`'s schedule, the worst case being an edit.
const ADALN_TABLE_MAX_BYTES: u64 = 6 * (3 * 7) * 5376 * 50 * 2;

/// The bf16 AdaLN stack — the architecture fact the weights-free contract declares
/// (`50 · (96768 · 2688 + 96768)` bf16 elements, the same projections as H3's).
const ADALN_BF16_BYTES: u64 = 50 * (96_768 * 2_688 + 96_768) * 2;

/// The text rung 3's `StructurallyNotApplicable` carries.
fn rung3_inapplicability_reason() -> String {
    let cells = crate::memory_strategy::RUNG3_MEASURED_PEAK_DELTAS
        .iter()
        .map(|(tier, frames, seq, heads, rows, one_head)| {
            format!(
                "{tier} {frames}f/seq {seq}: {:+.2}% heads, {:+.2}% query rows, {:+.2}% one-head",
                heads * 100.0,
                rows * 100.0,
                one_head * 100.0
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    format!(
        "MLX's fused SDPA streams the attention scores, so the [B,H,S,S] tensor rung 3 bounds is \
         never built at any sequence length. The H3 block stack this family runs was measured \
         in-forward on both chunk axes at S=104030, above every image sequence (16384 image rows \
         at 16 MP + up to 4096 prompt tokens + reference rows), and every chunked arm COSTS denoise \
         peak ({cells}). tests/h3_image_bounded_attention_real.rs re-takes the verdict on the real \
         image DiT at 2048 and 4096 px and fails if any arm saves more than 0.5% of peak. MLX-scoped: the candle \
         backend materializes its scores."
    )
}

/// Rung 4 exists exactly on a deferred load — the rung's shared prerequisite.
pub fn streamable(spec: &LoadSpec) -> bool {
    spec.load_shape == LoadShape::DeferredMaterialization
}

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

/// The contract's byte facts, from [`ImageMemoryFacts`] (header-exact; zero for a weights-free
/// spec). The conditioning phase carries the vision tower too — `edit` maps it with the text
/// encoder — so the route-independent bound covers both tasks.
fn asset_facts(facts: &ImageMemoryFacts) -> MemoryAssetFacts {
    let conditioning = facts.te_bytes().saturating_add(facts.vision_bytes);
    let transformer = facts.dit_bytes();
    let decoder = facts.decoder_bytes;
    MemoryAssetFacts {
        base_bytes: conditioning
            .saturating_add(transformer)
            .saturating_add(decoder),
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: decoder,
        // The configured LoRA / LoKr stack and nothing else, resident for the whole render at
        // every rung (H3's installer accounting; see `ImageMemoryFacts::adapters`).
        overlay_bytes: facts.adapters.resident_bytes,
    }
}

/// The streamable DiT as the shared phase facts spell it: the resident trunk (I/O projections,
/// per-step heads, refiner) plus one stack of 50 per-block byte counts (body + `adaln_proj`), whose
/// unwindowed sum is exactly `transformer_bytes`.
fn transformer_stream(facts: &ImageMemoryFacts) -> Option<StreamedWeightFacts> {
    if facts.dit_body_bytes.iter().all(|&b| b == 0) {
        return None;
    }
    Some(StreamedWeightFacts {
        resident_bytes: facts.dit_trunk_bytes,
        stacks: vec![facts
            .dit_body_bytes
            .iter()
            .zip(&facts.dit_adaln_bytes)
            .map(|(body, adaln)| body.saturating_add(*adaln))
            .collect()],
    })
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
    facts: &ImageMemoryFacts,
    calibration: Option<MemoryCalibrationIdentity>,
) -> MemoryProviderContract {
    let streamable = streamable(spec);
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
            MemoryStrategy::BoundedAttention => MemoryStrategySupport::StructurallyNotApplicable {
                reason: rung3_inapplicability_reason(),
            },
            MemoryStrategy::BoundedTransformerResidency if streamable => {
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
        };
        capability.parameters = match capability.strategy {
            MemoryStrategy::BoundedDecode => bounded_decode_parameters(),
            MemoryStrategy::BoundedTransformerResidency if streamable => MemoryParameterRanges {
                transformer_window_sizes: vec![IMAGE_TRANSFORMER_WINDOW_SIZE],
                transformer_window_components: vec![TransformerComponent::Both],
                ..Default::default()
            },
            _ => MemoryParameterRanges::default(),
        };
    }
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: phases(),
        synchronized_phase_release: true,
        decode_tiling: true,
        attention_chunking: false,
        transformer_window_materialization: streamable,
    };
    let mut phase_facts = mlx_gen::gen_core::MemoryPhaseFacts::staged(
        mlx_gen::gen_core::StagedWeightSchedule::ThreeStage,
    );
    phase_facts.transformer_stream = transformer_stream(facts);
    contract.phase_facts = Some(phase_facts);
    let asset_facts = asset_facts(facts);
    let adaln_resident = match facts.dit_adaln_bytes.iter().sum::<u64>() {
        0 => ADALN_BF16_BYTES,
        resolved => resolved,
    };
    let mut resident_components = vec![MemoryResidentComponent {
        id: ADALN_COMPONENT.to_owned(),
        kind: MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
        resident_bytes: adaln_resident,
        bounded_by: None,
        residency: MemoryComponentResidency::PrecomputedThenEvicted {
            precomputed_in: MemoryPhase::Denoise,
            retained_bytes: ADALN_TABLE_MAX_BYTES,
            evidence: "ImageDenoiser::new runs H3's AdaLnCache::precompute_and_evict (driven by \
                       tests/adaln_evict_real_weights.rs and tests/adaln_evict_memory.rs on the \
                       same block stack); ImageDenoiser::new_windowed never holds the stack \
                       (precompute_adaln_windowed)"
                .to_owned(),
        },
    }];
    let mut variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::DecodeTileArea,
    ];
    if asset_facts.overlay_bytes > 0 {
        variables.push(MemoryFormulaVariable::OverlayBytes);
        resident_components.push(MemoryResidentComponent {
            id: "adapter_stack".to_owned(),
            kind: MemoryComponentKind::AdapterStack,
            resident_bytes: asset_facts.overlay_bytes,
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        });
    }
    contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
        phases: phases(),
        variables,
        resident_components,
    };
    contract.load_shape = spec.load_shape;
    contract.architecture_facts = architecture_facts(spec);
    contract.asset_facts = asset_facts;
    contract.calibration = calibration;
    contract
}

/// The production contract: header-exact bytes, no calibration identity (unmeasured).
pub fn contract_for(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract(
        spec,
        &ImageMemoryFacts::resolve(spec)?,
        None,
    ))
}

/// The weights-free contract catalog conformance resolves instead of [`contract_for`].
pub fn weights_free_contract(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract(
        spec,
        &ImageMemoryFacts::weights_free(),
        Some(MemoryCalibrationIdentity::new(
            STATIC_CALIBRATION_FINGERPRINT,
            spec.load_shape,
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
            let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
                IMAGE_MODEL_ID,
                context.geometry,
                contract.generation_memory(&context.selection),
                context.use_pid,
                DIT_BLOCKS,
                |_use_pid, edge, overlap| validate_decode_geometry(edge, overlap),
            )?;
            // Declaration is not enforcement: without this the scope's
            // `materialize_transformer_window` refuses the window the contract advertises. Guarded
            // by contract-aware `engages`, so a resident load (rung 4 `Missing`) admits none.
            config.transformer_window = contract
                .engages(
                    context.selection.strategy,
                    MemoryStrategy::BoundedTransformerResidency,
                )
                .then(|| {
                    context
                        .selection
                        .parameters
                        .transformer_window_size
                        .unwrap_or(IMAGE_TRANSFORMER_WINDOW_SIZE)
                });
            Ok(config)
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

/// The surfaces this route publishes: the dense bf16 / fp32 loads at both materialization shapes —
/// eager, and deferred (rung 4, SW-7094). The quantized tiers add no strategy, only smaller bytes
/// read from the staged directory.
fn memory_contract_surface_specs() -> Vec<mlx_gen::gen_core::MemoryContractSurfaceSpec> {
    mlx_gen::gen_core::mlx_memory_contract_surface_specs()
        .into_iter()
        .filter(|s| {
            s.spec.quantize.is_none()
                && s.spec.offload_policy == mlx_gen::gen_core::OffloadPolicy::Resident
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::WeightsSource;

    fn weights_free() -> MemoryProviderContract {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()));
        weights_free_contract(&spec).unwrap()
    }

    fn deferred() -> MemoryProviderContract {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()));
        spec.load_shape = LoadShape::DeferredMaterialization;
        weights_free_contract(&spec).unwrap()
    }

    /// Rungs 0-2 are implemented on every load; rung 3 is not applicable on MLX with a reason that
    /// names the measurement it rests on; rung 4 is `Missing` on a resident load.
    #[test]
    fn a_resident_load_declares_rungs_0_to_2_and_rung_3_not_applicable() {
        let c = weights_free();
        let support = |s| c.capability(s).unwrap().support.clone();
        for rung in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
        ] {
            assert_eq!(
                support(rung),
                MemoryStrategySupport::Implemented,
                "{rung:?}"
            );
        }
        let MemoryStrategySupport::StructurallyNotApplicable { reason } =
            support(MemoryStrategy::BoundedAttention)
        else {
            panic!("rung 3 must be StructurallyNotApplicable on MLX");
        };
        assert!(
            reason.contains("h3_image_bounded_attention_real"),
            "{reason}"
        );
        assert!(reason.contains("104030"), "{reason}");
        assert_eq!(
            support(MemoryStrategy::BoundedTransformerResidency),
            MemoryStrategySupport::Missing
        );
        assert_eq!(
            c.capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .parameters,
            bounded_decode_parameters()
        );
        assert!(c.lifecycle.decode_tiling);
        assert!(!c.lifecycle.attention_chunking);
        assert!(!c.lifecycle.transformer_window_materialization);
        assert_eq!(c.load_shape, LoadShape::EagerMaterialization);
        assert_eq!(c.provider_id, IMAGE_MODEL_ID);
        assert!(
            c.conformance_errors().is_empty(),
            "{:?}",
            c.conformance_errors()
        );
    }

    /// AC1: a deferred load declares rung 4 implemented for BOTH transformers at the singleton
    /// window, with the window hook and the load shape to match.
    #[test]
    fn a_deferred_load_declares_rung_4_for_both_transformers() {
        let c = deferred();
        let rung4 = c
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(rung4.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            rung4.parameters.transformer_window_sizes,
            vec![IMAGE_TRANSFORMER_WINDOW_SIZE]
        );
        assert_eq!(
            rung4.parameters.transformer_window_components,
            vec![TransformerComponent::Both]
        );
        assert!(c.lifecycle.transformer_window_materialization);
        assert_eq!(c.load_shape, LoadShape::DeferredMaterialization);
        assert_eq!(
            c.calibration.as_ref().unwrap().load_shape,
            LoadShape::DeferredMaterialization
        );
        assert!(
            c.conformance_errors().is_empty(),
            "{:?}",
            c.conformance_errors()
        );
    }

    /// The AdaLN eviction carries over as a declared sub-stack that keeps the (small) table, and a
    /// weights-free contract publishes no stream facts it could not have read.
    #[test]
    fn the_adaln_eviction_is_declared_net_of_its_table() {
        let c = weights_free();
        let MemoryFormulaKind::ComponentPhaseEnvelope {
            resident_components,
            ..
        } = &c.formula
        else {
            panic!("expected a component phase envelope");
        };
        let adaln = &resident_components[0];
        assert_eq!(adaln.resident_bytes, ADALN_BF16_BYTES);
        assert_eq!(
            ADALN_BF16_BYTES,
            crate::memory_strategy::ADALN_EVICTED_BYTES
        );
        let MemoryComponentResidency::PrecomputedThenEvicted { retained_bytes, .. } =
            adaln.residency
        else {
            panic!("expected PrecomputedThenEvicted");
        };
        assert_eq!(retained_bytes, 67_737_600);
        assert!(c.phase_facts.as_ref().unwrap().transformer_stream.is_none());
    }

    /// The request scope admits the declared window on a deferred load and none on a resident one.
    #[test]
    fn the_request_scope_carries_the_declared_window() {
        use mlx_gen::gen_core::{MemorySelection, MemoryStrategyParameters};
        let c = deferred();
        let selection = MemorySelection {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(256),
                decode_overlap: Some(64),
                transformer_window_size: Some(1),
                transformer_window_component: Some(TransformerComponent::Both),
                ..Default::default()
            },
            tier: MemoryNumericTier {
                precision: mlx_gen::Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
        };
        c.validate_selection(&selection).unwrap();
        let memory = c.generation_memory(&selection).unwrap();
        assert!(memory.stream_transformer_blocks);
        assert_eq!(memory.transformer_window_size, Some(1));
        assert!(weights_free().validate_selection(&selection).is_err());
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
