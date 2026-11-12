//! MiniMax-H3-Image's shared-ladder memory contract on MLX/Metal.
//!
//! | rung | strategy | status |
//! |---|---|---|
//! | 0 | [`MemoryStrategy::Resident`] | implemented — the baseline; the generator always stages, so a resident admission over-provisions |
//! | 1 | [`MemoryStrategy::StagedResidency`] | implemented, structural — the text encoder is built, encodes, is forced and released before the DiT maps; the DiT denoises and is released before the VAE maps. `unconditionally_engages_staged_residency` |
//! | 2 | [`MemoryStrategy::BoundedDecode`] | implemented — SW-7089's 256 px / 64 px tiled decode ([`crate::image_vae::bounded_decode_parameters`]), a singleton domain because the tile edge changes the picture |
//! | 3 | [`MemoryStrategy::BoundedAttention`] | implemented (SW-7094) — every DiT block's attention runs as whole-head chunks at the shared 64 Mi score budget ([`IMAGE_ATTENTION_CHUNK_SIZE`]), floored at one whole head ([`image_attention_budget`]); see below |
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
//! # Rung 3 (SW-7094)
//!
//! MLX's fused `scaled_dot_product_attention` never materializes the `[B, H, S, S]` score tensor,
//! so the rung does not bound scores here: it bounds the head-major `q`/`k`/`v` operands and the
//! attention output, which a chunk holds for its own heads only. H3 video measured every chunked
//! arm costing denoise peak at `S = 104_030` (sc-18661), and this family declared the rung not
//! applicable on that ground; the image-geometry measurement then found the head axis at 64 Mi
//! **saving** 1.05 % of the q4 denoise peak at 2048² (run 43042251357), with a bit-identical
//! velocity, so the rung is implemented. Each block's attention runs through
//! [`crate::h3_image::ImageDenoiser::set_bounded_attention`] on [`AttentionChunkAxis::Heads`],
//! resident and windowed alike, so rung 4 (which engages rung 3 by cost order) chunks too.
//!
//! The budget is the shared [`IMAGE_ATTENTION_CHUNK_SIZE`] (64 Mi score elements), floored at one
//! whole head of the request's sequence ([`image_attention_budget`]). Without the floor a sequence
//! above 8,192 rows (any 16 MP request) falls back to query-row chunks inside a head, the axis H3
//! measured costing peak and the one whose reconstruction is only tolerance-equivalent. With it,
//! every chunk is whole heads and the output is the unbounded one bit for bit.
//! `tests/h3_image_bounded_attention_real.rs` asserts on the real q4 image DiT at 2048² and 4096²
//! that the rung lowers the denoise peak and leaves the velocity bit-identical.
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

use mlx_gen::attention::{AttentionBudget, AttentionChunkAxis};

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

/// Rung 3's one declared parameter: the shared constrained score budget (64 Mi elements per
/// attention call) — the arm `tests/h3_image_bounded_attention_real.rs` measured saving peak.
pub const IMAGE_ATTENTION_CHUNK_SIZE: u32 =
    mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;

/// The attention budget a rung-3 request runs at over a packed sequence of `seq_len` rows: the
/// declared `chunk_size`, floored at one whole head (`seq_len²` scores, batch 1). The floor keeps
/// the planner on whole heads at every admitted sequence, so a long sequence never falls back to
/// query-row chunks (see the module docs). MLX never builds the scores either way.
pub fn image_attention_budget(chunk_size: u32, seq_len: u64) -> AttentionBudget {
    AttentionBudget::from_score_elements(
        u64::from(chunk_size).max(seq_len.saturating_mul(seq_len)),
        true,
    )
}

/// Whole heads per attention chunk at `budget` over `seq_len` rows of `heads` heads — the
/// planner's own answer ([`AttentionBudget::head_chunks`]), for admission's pricing.
pub fn image_attention_heads_per_chunk(budget: AttentionBudget, heads: u64, seq_len: u64) -> u64 {
    budget
        .head_chunks(1, heads, seq_len, seq_len)
        .heads_per_chunk()
}

/// The rung-3 plan one request runs at: `None` when its memory block does not chunk attention,
/// else the budget over `seq_len` rows on the head axis. A chunk size outside the declared domain
/// is refused, not clamped.
pub fn requested_bounded_attention(
    memory: Option<mlx_gen::gen_core::GenerationMemory>,
    seq_len: u64,
) -> mlx_gen::gen_core::Result<Option<(AttentionBudget, AttentionChunkAxis)>> {
    let Some(memory) = memory.filter(|memory| memory.chunk_attention) else {
        return Ok(None);
    };
    let chunk = memory
        .attention_chunk_size
        .unwrap_or(IMAGE_ATTENTION_CHUNK_SIZE);
    if chunk != IMAGE_ATTENTION_CHUNK_SIZE {
        return Err(mlx_gen::gen_core::Error::Unsupported(format!(
            "{IMAGE_MODEL_ID}: attention chunk size {chunk} is outside the published domain \
             [{IMAGE_ATTENTION_CHUNK_SIZE}]"
        )));
    }
    Ok(Some((
        image_attention_budget(chunk, seq_len),
        AttentionChunkAxis::Heads,
    )))
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
            | MemoryStrategy::BoundedDecode
            | MemoryStrategy::BoundedAttention => MemoryStrategySupport::Implemented,
            MemoryStrategy::BoundedTransformerResidency if streamable => {
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
        };
        capability.parameters = match capability.strategy {
            MemoryStrategy::BoundedDecode => bounded_decode_parameters(),
            MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                attention_chunk_sizes: vec![IMAGE_ATTENTION_CHUNK_SIZE],
                ..Default::default()
            },
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
        attention_chunking: true,
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
        MemoryFormulaVariable::AttentionChunkSize,
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
/// route, one image per sequence (a batch renders sequentially), one latent frame, a positive
/// requested size whose generation canvas has a latent grid.
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
    admission_canvas(g.width, g.height)?;
    Ok(())
}

/// The canvas a `width x height` request renders on, which is what its resources are sized from.
///
/// Admission takes the **requested** size, exactly as the generator does: any positive size
/// ([`mlx_gen::gen_core::h3_image::validate_size`]). The generator renders on
/// [`mlx_gen::gen_core::h3_image::generation_canvas`] at the default minimum area and resizes back,
/// so the latent grid, the packed sequence and the decode all follow the canvas, not the request.
/// The admitted geometry stays the requested size, so the request scope's equality check binds
/// the request the caller actually sends.
pub fn admission_canvas(width: u32, height: u32) -> mlx_gen::gen_core::Result<(u32, u32)> {
    use mlx_gen::gen_core::h3_image::{
        generation_canvas, latent_grid, validate_size, MIN_GENERATION_AREA_DEFAULT,
    };
    validate_size(width, height).map_err(mlx_gen::gen_core::Error::from)?;
    let canvas = generation_canvas(width, height, MIN_GENERATION_AREA_DEFAULT);
    latent_grid(canvas.0, canvas.1)?;
    Ok(canvas)
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
            // The same guard for rung 3: the scope admits the declared chunk size exactly when
            // the selection engages the rung (selected directly, or by rung 4's cost order).
            config.attention_chunk_size = contract
                .engages(context.selection.strategy, MemoryStrategy::BoundedAttention)
                .then(|| {
                    context
                        .selection
                        .parameters
                        .attention_chunk_size
                        .unwrap_or(IMAGE_ATTENTION_CHUNK_SIZE)
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
    use crate::h3_image::IMAGE_DIT_COMPONENT;
    use crate::image_vae::IMAGE_VAE_COMPONENT;
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

    /// Rungs 0-3 are implemented on every load, rung 3 at the one shared 64 Mi budget; rung 4 is
    /// `Missing` on a resident load.
    #[test]
    fn a_resident_load_declares_rungs_0_to_3() {
        let c = weights_free();
        let support = |s| c.capability(s).unwrap().support.clone();
        for rung in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
        ] {
            assert_eq!(
                support(rung),
                MemoryStrategySupport::Implemented,
                "{rung:?}"
            );
        }
        assert_eq!(
            c.capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .parameters
                .attention_chunk_sizes,
            vec![IMAGE_ATTENTION_CHUNK_SIZE]
        );
        assert_eq!(IMAGE_ATTENTION_CHUNK_SIZE, 64 << 20);
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
        assert!(c.lifecycle.attention_chunking);
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
                // Rung 4 engages rung 3 by cost order, so its chunk size is required too.
                attention_chunk_size: Some(IMAGE_ATTENTION_CHUNK_SIZE),
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

    /// Rung 3's runtime plan: a request whose memory block chunks attention runs whole heads at
    /// the declared budget — 3 heads per chunk at the 4 MP canvas's 4,608 rows — floored at one
    /// whole head on a sequence one head already overflows (a 16 MP request's 20,480 rows), so the
    /// planner never falls back to query rows. Unselected is `None`; an undeclared size is refused.
    #[test]
    fn rung_3_runs_whole_heads_floored_at_one_head() {
        use mlx_gen::gen_core::GenerationMemory;
        let heads = 56u64;
        let chunk = |memory, seq| requested_bounded_attention(memory, seq).unwrap();
        let on = GenerationMemory {
            chunk_attention: true,
            attention_chunk_size: Some(IMAGE_ATTENTION_CHUNK_SIZE),
            ..Default::default()
        };
        assert!(chunk(None, 4608).is_none());
        assert!(chunk(Some(GenerationMemory::default()), 4608).is_none());

        let (budget, axis) = chunk(Some(on), 4608).unwrap();
        assert_eq!(axis, AttentionChunkAxis::Heads);
        assert_eq!(budget.max_score_elements(), 64 << 20);
        assert!(budget.eval_per_chunk());
        let plan = budget.head_chunks(1, heads, 4608, 4608);
        assert!(plan.chunks_heads());
        assert_eq!(plan.heads_per_chunk(), 3);
        // Each 3-head chunk fits, so the query axis is not split inside it.
        assert_eq!(budget.query_block(1, 3, 4608, 4608), 4608);

        let (budget, _) = chunk(Some(on), 20_480).unwrap();
        assert_eq!(budget.max_score_elements(), 20_480 * 20_480);
        assert_eq!(image_attention_heads_per_chunk(budget, heads, 20_480), 1);
        assert_eq!(budget.query_block(1, 1, 20_480, 20_480), 20_480);

        let named_default = GenerationMemory {
            attention_chunk_size: None,
            ..on
        };
        assert!(chunk(Some(named_default), 4608).is_some());
        let other = GenerationMemory {
            attention_chunk_size: Some(IMAGE_ATTENTION_CHUNK_SIZE / 2),
            ..on
        };
        assert!(requested_bounded_attention(Some(other), 4608).is_err());
    }

    /// The contract's own rung-3 selection derives a generation memory that chunks attention at
    /// the declared size, and rung 4 engages rung 3 by cost order on a deferred load; the request
    /// scope admits exactly that chunk size.
    #[test]
    fn rung_3_selection_reaches_the_request_scope() {
        use mlx_gen::gen_core::{MemorySelection, MemoryStrategyParameters};
        let tier = MemoryNumericTier {
            precision: mlx_gen::Precision::Bf16,
            quant: None,
            component_precision_floors: &[],
        };
        let decode = MemoryStrategyParameters {
            decode_tile_edge: Some(256),
            decode_overlap: Some(64),
            attention_chunk_size: Some(IMAGE_ATTENTION_CHUNK_SIZE),
            ..Default::default()
        };
        for (contract, strategy, parameters) in [
            (weights_free(), MemoryStrategy::BoundedAttention, decode),
            (
                deferred(),
                MemoryStrategy::BoundedTransformerResidency,
                MemoryStrategyParameters {
                    transformer_window_size: Some(1),
                    transformer_window_component: Some(TransformerComponent::Both),
                    ..decode
                },
            ),
        ] {
            let selection = MemorySelection {
                strategy,
                parameters,
                tier,
            };
            contract.validate_selection(&selection).unwrap();
            let memory = contract.generation_memory(&selection).unwrap();
            assert!(memory.chunk_attention, "{strategy:?}");
            assert_eq!(
                memory.attention_chunk_size,
                Some(IMAGE_ATTENTION_CHUNK_SIZE)
            );
            assert!(requested_bounded_attention(Some(memory), 4608)
                .unwrap()
                .is_some());
        }
        // Rung 2 alone does not chunk attention.
        let rung2 = MemorySelection {
            strategy: MemoryStrategy::BoundedDecode,
            parameters: MemoryStrategyParameters {
                attention_chunk_size: None,
                ..decode
            },
            tier,
        };
        let c = weights_free();
        assert!(!c.generation_memory(&rung2).unwrap().chunk_attention);

        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()));
        let tier = mlx_gen::request_scope::loaded_tier(&spec);
        let scope = |strategy| {
            let context = mlx_gen::gen_core::standard_memory_behavior_context(
                &c,
                strategy,
                tier,
                mlx_gen::gen_core::MemoryBehaviorRoute {
                    mode: mlx_gen::gen_core::MemoryMode::TextToImage,
                    reference_count: 0,
                    use_pid: false,
                    has_phases: true,
                    overlay: None,
                },
            )
            .unwrap();
            registered_scope(IMAGE_MODEL_ID, &spec, &c, &context, tier)
                .unwrap()
                .unwrap()
        };
        let mut rung3 = scope(MemoryStrategy::BoundedAttention);
        rung3
            .configure_attention(IMAGE_ATTENTION_CHUNK_SIZE)
            .unwrap();
        assert!(rung3
            .configure_attention(IMAGE_ATTENTION_CHUNK_SIZE / 2)
            .is_err());
        let mut rung2 = scope(MemoryStrategy::BoundedDecode);
        assert!(rung2
            .configure_attention(IMAGE_ATTENTION_CHUNK_SIZE)
            .is_err());
    }

    /// A production contract claims no calibration: nothing has been measured.
    #[test]
    fn production_contract_is_uncalibrated() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()));
        assert!(contract_for(&spec).unwrap().calibration.is_none());
        assert!(weights_free().calibration.is_some());
    }

    /// Write a header-plus-zeros safetensors file of `(name, dtype, shape)` tensors.
    fn write_safetensors(path: &std::path::Path, tensors: &[(&str, &str, &[usize])]) {
        let width = |dtype: &str| if dtype == "F32" { 4 } else { 2 };
        let mut header = serde_json::Map::new();
        let mut offset = 0_usize;
        for (name, dtype, shape) in tensors {
            let len = shape.iter().product::<usize>() * width(dtype);
            header.insert(
                (*name).into(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len]}),
            );
            offset += len;
        }
        let json = serde_json::Value::Object(header).to_string().into_bytes();
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend(json);
        bytes.resize(bytes.len() + offset, 0);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    /// A snapshot whose DiT mixes a cast bf16 block tensor (64 elements) with a stored-f32 PDD
    /// head (16) and a stored-bf16 `context_embedder` (8), and whose VAE carries a decode tensor
    /// (32, bf16), a stored-f32 decode tensor (4) and an encoder tensor the loader never reads.
    fn mixed_snapshot() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        write_safetensors(
            &root
                .path()
                .join(IMAGE_DIT_COMPONENT)
                .join("model.safetensors"),
            &[
                ("transformer_blocks.0.attn.to_q.weight", "BF16", &[8, 8]),
                ("pdd_proj_out.0.weight", "F32", &[4, 4]),
                ("context_embedder.weight", "BF16", &[2, 4]),
            ],
        );
        write_safetensors(
            &root
                .path()
                .join(IMAGE_VAE_COMPONENT)
                .join("model.safetensors"),
            &[
                ("decoder.proj_in.weight", "BF16", &[4, 8]),
                ("post_quant_conv.bias", "F32", &[4]),
                ("encoder.proj_in.weight", "BF16", &[1000]),
            ],
        );
        root
    }

    /// Fp32 casts the block stack and the VAE decode half to float32 at load, so they are priced
    /// at 4 bytes per element; the top-level DiT tensors keep their stored widths, and the
    /// encoder half of the VAE is not priced at all. Bf16 prices the same tensors at 2 bytes,
    /// never below a stored-f32 tensor's own width.
    #[test]
    fn asset_facts_price_the_dtype_the_precision_loads() {
        let root = mixed_snapshot();
        let facts = |precision| {
            let mut spec = LoadSpec::new(WeightsSource::Dir(root.path().to_path_buf()));
            spec.precision = precision;
            contract_for(&spec).unwrap().asset_facts
        };
        let fp32 = facts(mlx_gen::Precision::Fp32);
        assert_eq!(fp32.transformer_bytes, 64 * 4 + 16 * 4 + 8 * 2);
        assert_eq!(fp32.decoder_bytes, 32 * 4 + 4 * 4);
        let bf16 = facts(mlx_gen::Precision::Bf16);
        assert_eq!(bf16.transformer_bytes, 64 * 2 + 16 * 4 + 8 * 2);
        assert_eq!(bf16.decoder_bytes, 32 * 2 + 4 * 4);
        assert_eq!(fp32.conditioning_bytes, 0);
        assert_eq!(fp32.base_bytes, fp32.transformer_bytes + fp32.decoder_bytes);
    }

    /// A non-32-aligned request the generator accepts is admitted at its requested size, and its
    /// resources follow the canvas the generator renders on. The request scope still binds the
    /// requested size: the canvas itself is not the admitted request.
    #[test]
    fn a_non_aligned_request_is_admitted_and_sized_at_its_canvas() {
        use mlx_gen::gen_core::h3_image::{generation_canvas, MIN_GENERATION_AREA_DEFAULT};
        use mlx_gen::gen_core::{
            GenerationRequest, MemoryBehaviorRoute, MemoryGeometry, MemoryMode,
        };

        let canvas = admission_canvas(1000, 1001).unwrap();
        assert_eq!(
            canvas,
            generation_canvas(1000, 1001, MIN_GENERATION_AREA_DEFAULT)
        );
        assert_ne!(canvas, (1000, 1001));
        assert!(admission_canvas(0, 1001).is_err());

        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()));
        let contract = weights_free_contract(&spec).unwrap();
        let tier = mlx_gen::request_scope::loaded_tier(&spec);
        let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::StagedResidency,
            tier,
            MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: true,
                overlay: None,
            },
        )
        .unwrap();
        context.geometry = MemoryGeometry {
            width: 1000,
            height: 1001,
            batch: 1,
            frames: 1,
            reference_count: 0,
        };
        assert_eq!(
            safety_check_at_tier(&contract, tier, &context),
            MemorySafetyDecision::Accept
        );
        let request = |width, height| GenerationRequest {
            prompt: "a lighthouse at dusk".into(),
            width,
            height,
            ..Default::default()
        };
        let mut scope = registered_scope(IMAGE_MODEL_ID, &spec, &contract, &context, tier)
            .unwrap()
            .unwrap();
        scope.configure_request(&mut request(1000, 1001)).unwrap();
        let mut scope = registered_scope(IMAGE_MODEL_ID, &spec, &contract, &context, tier)
            .unwrap()
            .unwrap();
        assert!(scope
            .configure_request(&mut request(canvas.0, canvas.1))
            .is_err());
    }

    #[test]
    fn only_the_shipped_decode_geometry_is_admitted() {
        validate_decode_geometry(256, 64).unwrap();
        assert!(validate_decode_geometry(512, 64).is_err());
        assert!(validate_decode_geometry(256, 0).is_err());
    }
}
