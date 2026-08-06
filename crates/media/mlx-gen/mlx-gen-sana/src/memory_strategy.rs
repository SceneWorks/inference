//! Shared memory-strategy contract for base SANA and SANA-Sprint (SC-16783, extended by SC-15523).
//!
//! The provider implements Resident, load-time staged residency, the measured DC-AE tiled decode
//! ladder, and — since SC-15523 — bounded attention over `attn2` and bounded transformer residency
//! over the 20-block Linear-DiT stack.
//!
//! ## Rung 3's scope is `attn2` only, and that is architecture, not convenience
//!
//! SANA has two attentions per block and only one of them has anything for a score budget to bound:
//!
//! | site | kernel | score domain |
//! |---|---|---|
//! | `attn1` (self) | ReLU-**linear**, `num = (V·Kᵀ)·Q` | none — the key axis is contracted into a `[B,H,hd,hd]` gram matrix |
//! | `attn2` (caption cross) | hand-rolled softmax SDPA, f32 | `[B, H, N, 300]`, explicitly materialized |
//!
//! So the rung is genuinely applicable (it is **not** `StructurallyNotApplicable`) and its reach is
//! genuinely one of the two sites. The key axis is fixed at 300 caption slots, so the domain grows
//! only with the token count `N = (edge/32)²` — 6.14 Mi f32 scores at 1024², which is why the
//! sibling families' 64 Mi operating point is inert here and SANA publishes its own budgets.
//!
//! **The exactness claim has a measured boundary.** Query-row chunking leaves each row's complete
//! k/v and both reductions untouched, but it changes the query GEMM's `M` dimension, and at `M = 1`
//! MLX dispatches a different (gemv) kernel whose accumulation order differs — measured at ~1e-6
//! relative on this provider. So rung 3 is bit-exact **over its published domain**, not
//! unconditionally: the narrowest chunk any published budget produces anywhere in the advertised
//! 256..=1024 range is **174 query rows**, and
//! `a_single_query_row_chunk_is_not_bit_exact_and_the_domain_cannot_reach_one` pins both halves of
//! that — the degenerate case is not exact, and the domain cannot reach it.
//!
//! ## Rung 4's availability is a per-LOAD fact
//!
//! A window rebuilds trunk blocks from the snapshot, so it needs a **re-openable source**
//! ([`WeightsSource::Dir`] — every registry load; SANA already refuses a single-file load) *and*
//! [`LoadShape::DeferredMaterialization`], which says the request wants those re-openable blocks
//! rather than a bulk-committed stack. `OffloadPolicy` is deliberately absent from that pair: phase
//! release and intra-phase block materialization are separate axes. The rung-1 edge is declared
//! separately as a per-provider `additional_prerequisites` entry, because
//! [`MemoryStrategy::engages`] does **not** make rung 4 engage rung 1 universally.

use mlx_gen::asset_facts::ResidentProjection;
use mlx_gen::gen_core::{
    standard_memory_strategy_safety_check, Error as CoreError, GenerationMemory,
    MemoryBackendRealization, MemoryBehaviorFixture, MemoryBehaviorRoute,
    MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyPrerequisite, MemoryStrategySupport,
    ResidentRequestMemory, Result as CoreResult, TransformerComponent,
};
use mlx_gen::{LoadShape, LoadSpec, OffloadPolicy, WeightsSource};

use crate::pipeline::{DECODE_OVERLAP, DECODE_TILE_EDGE};

pub const DECODE_TILE_EDGES: &[u32] = &[512, 384, 256, 192];
pub const DECODE_TILE_EDGES_REJECTED: &[u32] = &[128, 96, 64];

/// Rung 3's published score budgets, in `[B, H, N, 300]` f32 elements per `attn2` call.
///
/// SANA's whole 1024² cross-attention domain is `1 · 20 · 1024 · 300 = 6.14 Mi` scores, so the
/// Z-Image family operating point ([`mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET`], 64 Mi)
/// never chunks here — the shared planner is budget-agnostic precisely so a family can publish its
/// own. These three bracket the domain: at 1024² they chunk into 2, 3 and 6 calls respectively, and
/// the smallest still chunks at 512² (2 calls). None of them chunks the 256² floor, whose whole
/// domain is 384 Ki scores — deliberately, because 1.5 MiB of f32 scratch is not what this rung
/// exists to bound. `the_shared_64mi_budget_never_chunks_sana_and_the_published_ones_do` pins all of
/// that arithmetic.
pub const ATTENTION_CHUNK_SIZES: &[u32] = &[4_194_304, 2_097_152, 1_048_576];
/// The default budget — the tightest published bound, which is what the rung exists for.
pub const ATTENTION_CHUNK_SIZE: u32 = 1_048_576;
/// Budgets swept and REJECTED, kept with their rejection so the domain is not silently narrowed.
/// 64 Mi is the sibling families' constant and is inert at every advertised SANA geometry.
pub const ATTENTION_CHUNK_SIZES_REJECTED: &[u32] = &[67_108_864];

/// Rung 4's published block cadences over the 20-block `transformer_blocks` stack.
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 4, 5, 10];
/// The default cadence — the tightest weight bound.
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;
/// The only component scope SANA windows. The Gemma-2 caption encoder is a uniform decoder stack and
/// is structurally windowable too (SC-15969's survey says so), but it lives in `mlx-gen-pid` and is
/// shared with PiD and LTX; nothing here is measured for it, so it stays unpublished and the
/// contract REFUSES it rather than letting an unmeasured scope be selected.
pub const TRANSFORMER_WINDOW_COMPONENT: TransformerComponent = TransformerComponent::Dit;

pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "sana-mlx-full-ladder-2026-08-06-v1-sequential";
pub const RESIDENT_MEMORY_CALIBRATION_FINGERPRINT: &str =
    "sana-mlx-full-ladder-2026-08-06-v1-resident";

pub const fn calibration_fingerprint(policy: OffloadPolicy) -> &'static str {
    match policy {
        OffloadPolicy::Sequential => MEMORY_CALIBRATION_FINGERPRINT,
        OffloadPolicy::Resident => RESIDENT_MEMORY_CALIBRATION_FINGERPRINT,
    }
}

/// Whether THIS load can execute a rung-4 window. See the module doc — two independent load-time
/// facts, and `OffloadPolicy` is not one of them.
pub(crate) fn is_streamable(spec: &LoadSpec) -> bool {
    matches!(spec.weights, WeightsSource::Dir(_))
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && spec.adapters.is_empty()
}

fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(
        provider_id,
        DECODE_TILE_EDGES.iter().copied(),
        DECODE_OVERLAP as u32,
    )
}

/// Price the DC-AE decoder at its **resident** width rather than its stored one (the sc-15839
/// class), falling back to the stored sum when the component is not on disk.
///
/// [`crate::dc_ae::DcAeDecoder::from_weights`] and its encoder twin `as_dtype(Float32)` every conv
/// weight, bias and norm they read, on every load, unconditionally — the same shape as
/// `mlx_gen_sdxl::load_vae`'s `cast_all(Float32)`, which sc-15839 measured as a **2x underprice at
/// every tier**.
///
/// Every SANA tier shipped so far already stores the DC-AE in f32, so this projection is IDENTITY
/// today. That is exactly why it is written down instead of left implicit: a narrower VAE in some
/// future tier would be underpriced by 2x from the moment it shipped, and an underpriced decoder
/// reads as a memory regression in the engine rather than as a pricing bug in the contract. The
/// trunk and the Gemma-2 caption encoder stay [`ResidentProjection::Stored`] — both keep their
/// on-disk width (packed tiers load packed; the dtype casts in those two components are all on
/// activations, never on a retained weight).
fn resident_decoder_bytes(spec: &LoadSpec, stored: u64) -> u64 {
    let WeightsSource::Dir(root) = &spec.weights else {
        return stored;
    };
    let vae = root.join("vae");
    if !vae.is_dir() {
        return stored;
    }
    mlx_gen::asset_facts::projected_safetensors_bytes(&vae, |_| ResidentProjection::Float32)
        .unwrap_or(stored)
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let components = crate::model::component_footprint(spec)?;
    contract_with_asset_facts(
        provider_id,
        spec,
        components.text_encoder,
        components.dit,
        resident_decoder_bytes(spec, components.vae),
    )
}

pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    contract_with_asset_facts(provider_id, spec, 0, 0, 0)
}

fn contract_with_asset_facts(
    provider_id: &str,
    spec: &LoadSpec,
    conditioning_bytes: u64,
    transformer_bytes: u64,
    decoder_bytes: u64,
) -> CoreResult<MemoryProviderContract> {
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
    let staged = matches!(spec.offload_policy, OffloadPolicy::Sequential);
    // Rung 4 needs BOTH load-time facts AND rung 1, whose own availability IS the `Sequential`
    // policy — declaring it on a Resident load would advertise a composition its own declared
    // prerequisite could never satisfy. ONE binding, so the capability, the lifecycle hook and the
    // formula variable cannot disagree about whether a window can run on this load.
    let windowed = is_streamable(spec) && staged;
    let mut variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::DecodeTileArea,
        MemoryFormulaVariable::AttentionChunkSize,
    ];
    if windowed {
        variables.push(MemoryFormulaVariable::TransformerWindowSize);
    }
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        variables,
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        calibration_fingerprint(spec.offload_policy),
        spec.load_shape,
    ));
    contract.asset_facts.conditioning_bytes = conditioning_bytes;
    contract.asset_facts.transformer_bytes = transformer_bytes;
    contract.asset_facts.decoder_bytes = decoder_bytes;
    contract.asset_facts.base_bytes = conditioning_bytes
        .saturating_add(transformer_bytes)
        .saturating_add(decoder_bytes);
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: matches!(spec.offload_policy, OffloadPolicy::Sequential),
        decode_tiling: true,
        attention_chunking: true,
        transformer_window_materialization: windowed,
    };
    // Sequential loads ship the measured bounded-decode default. An explicit shared-contract
    // Resident selection must therefore write an all-disabled request block to override it.
    contract.resident_request_memory = ResidentRequestMemory::ExplicitResident;

    for strategy in [
        MemoryStrategy::StagedResidency,
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency,
    ] {
        let capability = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == strategy)
            .expect("compatibility contract declares every rung");
        capability.support = match strategy {
            MemoryStrategy::StagedResidency if staged => MemoryStrategySupport::Implemented,
            MemoryStrategy::BoundedDecode | MemoryStrategy::BoundedAttention => {
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency if windowed => {
                MemoryStrategySupport::Implemented
            }
            _ => MemoryStrategySupport::Missing,
        };
        capability.parameters = match strategy {
            MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                decode_tile_edges: routes.native_edges().to_vec(),
                decode_overlaps: vec![DECODE_OVERLAP as u32],
                ..Default::default()
            },
            MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                attention_chunk_sizes: ATTENTION_CHUNK_SIZES.to_vec(),
                ..Default::default()
            },
            MemoryStrategy::BoundedTransformerResidency
                if capability.support == MemoryStrategySupport::Implemented =>
            {
                MemoryParameterRanges {
                    transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                    transformer_window_components: vec![TRANSFORMER_WINDOW_COMPONENT],
                    ..Default::default()
                }
            }
            _ => MemoryParameterRanges::default(),
        };
    }
    if windowed {
        // Rung 4 holds ~95 MiB of trunk weights instead of ~1.85 GiB, but that only moves the
        // REQUEST peak if the phases it does not bound have already been shed. `engages` does not
        // supply this edge (rung 4 does not engage rung 1 universally), so the provider declares it.
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
    }
    // The compatibility constructor supplies the required empty engagement-exclusion surface.
    debug_assert!(contract.default_engagement_exclusions.is_empty());
    Ok(contract)
}

/// Refuse a request-scoped memory selection whose parameters are outside the published domain —
/// **on the production `generate` path**, at the same layer for every rung.
///
/// A caller can set [`GenerationMemory`] on a request without going through
/// `begin_memory_strategy_request`, so admission-time validation alone leaves a hole through which
/// an unmeasured tile edge, score budget or block cadence reaches the engine and is silently
/// executed. Every published parameter is checked here, including rung 2's, so the three rungs are
/// enforced in one place rather than one of them being enforced twice and the others not at all.
///
/// The decode route goes through the same checked [`mlx_gen_pid::DecodeRoutes`] the admission gate
/// uses, so the native ladder and the PiD refusal cannot drift apart.
pub(crate) fn validate_request_memory(
    provider_id: &str,
    spec: &LoadSpec,
    memory: &GenerationMemory,
) -> CoreResult<()> {
    if memory.tile_vae_decode {
        decode_routes(provider_id)?
            .validate(
                false,
                Some(memory.decode_tile_edge.unwrap_or(DECODE_TILE_EDGE as u32)),
                Some(memory.decode_overlap.unwrap_or(DECODE_OVERLAP as u32)),
            )
            .map_err(CoreError::Unsupported)?;
    }
    if memory.chunk_attention {
        let size = memory.attention_chunk_size.unwrap_or(ATTENTION_CHUNK_SIZE);
        if !ATTENTION_CHUNK_SIZES.contains(&size) {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: attention_chunk_size {size} is outside the published domain \
                 {ATTENTION_CHUNK_SIZES:?}"
            )));
        }
    }
    if memory.stream_transformer_blocks {
        if !is_streamable(spec) {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: bounded transformer residency needs a re-openable snapshot \
                 directory loaded with DeferredMaterialization and no adapters"
            )));
        }
        // Rung 4's declared `EngagedInSameRequest` prerequisite, enforced rather than documented.
        // SANA's rung 1 is a LOAD-time mechanism (the `Sequential` policy drives the shared
        // `Residency` seam; sc-16783 measured and published it that way), so `is_streamable` above
        // plus the contract's `Sequential`-only rung-4 declaration already guarantee the mechanism
        // is running. This check is the other half: the SELECTION must say so too, or a caller
        // could compose a request the selector can never produce and get a peak nobody predicted.
        if !memory.stage_residency {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: bounded transformer residency requires staged residency engaged in \
                 the same request"
            )));
        }
        let window = memory
            .transformer_window_size
            .unwrap_or(TRANSFORMER_WINDOW_SIZE);
        if !TRANSFORMER_WINDOW_SIZES.contains(&window) {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: transformer_window_size {window} is outside the published domain \
                 {TRANSFORMER_WINDOW_SIZES:?}"
            )));
        }
        let component = memory
            .transformer_window_component
            .unwrap_or(TransformerComponent::Dit);
        if component != TRANSFORMER_WINDOW_COMPONENT {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: transformer_window_component {component:?} is outside the published \
                 domain [{TRANSFORMER_WINDOW_COMPONENT:?}]"
            )));
        }
    }
    Ok(())
}

pub(crate) fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            if context.use_pid {
                return Err(CoreError::Unsupported(format!(
                    "{}: SANA has no PiD decoder; mlx-gen-pid is used only for Gemma-2 conditioning",
                    contract.provider_id
                )));
            }
            decode_routes(&contract.provider_id)?
                .validate(
                    false,
                    context.selection.parameters.decode_tile_edge,
                    context.selection.parameters.decode_overlap,
                )
                .map_err(CoreError::Unsupported)?;
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

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<MemoryBehaviorFixture>> {
    if !matches!(
        contract
            .capability(strategy)
            .map(|capability| &capability.support),
        Some(MemoryStrategySupport::Implemented)
    ) || !strategy.is_optimized()
    {
        return Ok(Vec::new());
    }
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
        MemoryBehaviorRoute {
            mode: mlx_gen::gen_core::MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: matches!(spec.offload_policy, OffloadPolicy::Sequential),
            overlay: None,
        },
    )?;
    Ok(vec![MemoryBehaviorFixture::new(context)])
}

pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_with_cleanup(
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
    begin_with_cleanup(
        provider_id,
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

fn begin_with_cleanup(
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
        crate::config::SanaTransformerConfig::sana_1600m().num_layers as usize,
        move |use_pid, edge, overlap| {
            if use_pid {
                return Err(CoreError::Unsupported(format!(
                    "{provider_id}: SANA has no PiD decoder"
                )));
            }
            routes
                .validate(false, Some(edge), Some(overlap))
                .map_err(CoreError::Unsupported)
        },
    )?;
    config.attention_chunk_size = contract
        .engages(context.selection.strategy, MemoryStrategy::BoundedAttention)
        .then(|| {
            context
                .selection
                .parameters
                .attention_chunk_size
                .unwrap_or(ATTENTION_CHUNK_SIZE)
        });
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
                .unwrap_or(TRANSFORMER_WINDOW_SIZE)
        });
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    )))
}

pub fn declared_parameters() -> mlx_gen::gen_core::MemoryStrategyParameters {
    mlx_gen::gen_core::MemoryStrategyParameters {
        decode_tile_edge: Some(DECODE_TILE_EDGE as u32),
        decode_overlap: Some(DECODE_OVERLAP as u32),
        attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
        transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
        transformer_window_component: Some(TRANSFORMER_WINDOW_COMPONENT),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::MemoryStrategySupport;
    use mlx_gen::WeightsSource;

    fn spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent/sana-contract".into()))
            .with_offload_policy(policy)
            .with_load_shape(LoadShape::EagerMaterialization)
    }

    fn streamable_spec(policy: OffloadPolicy) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent/sana-contract".into()))
            .with_offload_policy(policy)
            .with_load_shape(LoadShape::DeferredMaterialization)
    }

    #[test]
    fn both_ids_publish_the_same_five_rung_ladder() {
        let spec = streamable_spec(OffloadPolicy::Sequential);
        let base = weights_free_memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        let sprint = weights_free_memory_strategy_contract(crate::SPRINT_MODEL_ID, &spec).unwrap();
        assert!(
            base.conformance_errors().is_empty(),
            "{:?}",
            base.conformance_errors()
        );
        for strategy in MemoryStrategy::ALL {
            assert_eq!(base.capability(strategy), sprint.capability(strategy));
        }
        for strategy in MemoryStrategy::ALL {
            assert_eq!(
                base.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "{strategy:?} must be implemented on the full-ladder route"
            );
        }
        assert_eq!(
            base.resident_request_memory,
            ResidentRequestMemory::ExplicitResident
        );
        assert!(base.lifecycle.attention_chunking);
        assert!(base.lifecycle.transformer_window_materialization);
        // Rung 4's rung-1 edge is a PROVIDER declaration: the shared `engages` order deliberately
        // does not drag rung 1 in, so a missing `additional_prerequisites` entry would leave the
        // composition selectable without the phase release that makes it move the request peak.
        assert!(
            base.requires(MemoryStrategy::BoundedTransformerResidency)
                .any(|prerequisite| prerequisite
                    == MemoryStrategyPrerequisite::Rung {
                        rung: MemoryStrategy::StagedResidency,
                        scope: MemoryPrerequisiteScope::EngagedInSameRequest,
                    }),
            "rung 4 must declare its rung-1 engagement prerequisite"
        );
        assert!(
            !MemoryStrategy::BoundedTransformerResidency.engages(MemoryStrategy::StagedResidency),
            "the shared cost order must NOT be the source of that edge"
        );
    }

    #[test]
    fn published_rung_three_and_four_domains_are_pinned() {
        assert_eq!(ATTENTION_CHUNK_SIZES, &[4_194_304, 2_097_152, 1_048_576]);
        assert_eq!(ATTENTION_CHUNK_SIZE, 1_048_576);
        assert_eq!(ATTENTION_CHUNK_SIZES_REJECTED, &[67_108_864]);
        assert_eq!(TRANSFORMER_WINDOW_SIZES, &[1, 2, 4, 5, 10]);
        assert_eq!(TRANSFORMER_WINDOW_SIZE, 1);
        assert_eq!(TRANSFORMER_WINDOW_COMPONENT, TransformerComponent::Dit);
        let contract = weights_free_memory_strategy_contract(
            crate::MODEL_ID,
            &streamable_spec(OffloadPolicy::Sequential),
        )
        .unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .parameters
                .attention_chunk_sizes,
            ATTENTION_CHUNK_SIZES.to_vec()
        );
        let window = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(
            window.parameters.transformer_window_sizes,
            TRANSFORMER_WINDOW_SIZES.to_vec()
        );
        assert_eq!(
            window.parameters.transformer_window_components,
            vec![TransformerComponent::Dit]
        );
        // The rejected sibling constant must stay out of the published domain in both directions.
        for rejected in ATTENTION_CHUNK_SIZES_REJECTED {
            assert!(!ATTENTION_CHUNK_SIZES.contains(rejected));
        }
    }

    /// **Rung 4's availability is a per-LOAD fact, and the two facts are independent.**
    ///
    /// An eager load and a single-file load can BOTH satisfy every other declaration, so a rung-4
    /// declaration keyed off anything else (the offload policy, the provider id, the tier) would
    /// pass a shape it cannot execute to the selector.
    #[test]
    fn rung_four_availability_reads_source_load_shape_and_the_staged_prerequisite() {
        let deferred_dir = streamable_spec(OffloadPolicy::Sequential);
        assert!(is_streamable(&deferred_dir));

        let mut eager = deferred_dir.clone();
        eager.load_shape = LoadShape::EagerMaterialization;
        assert!(!is_streamable(&eager));

        let single_file =
            LoadSpec::new(WeightsSource::File("/nonexistent/sana.safetensors".into()))
                .with_offload_policy(OffloadPolicy::Sequential)
                .with_load_shape(LoadShape::DeferredMaterialization);
        assert!(!is_streamable(&single_file));

        let mut adapted = deferred_dir.clone();
        adapted.adapters.push(mlx_gen::AdapterSpec::new(
            "/nonexistent/lora.safetensors".into(),
            1.0,
            mlx_gen::AdapterKind::Lora,
        ));
        assert!(!is_streamable(&adapted));

        for unavailable in [eager, single_file, adapted] {
            let contract =
                weights_free_memory_strategy_contract(crate::MODEL_ID, &unavailable).unwrap();
            assert!(contract.conformance_errors().is_empty());
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing,
                "a load that cannot stream must not advertise rung 4"
            );
            assert!(!contract.lifecycle.transformer_window_materialization);
            let MemoryFormulaKind::PhaseEnvelope { variables, .. } = &contract.formula else {
                panic!("SANA declares a phase envelope")
            };
            assert!(!variables.contains(&MemoryFormulaVariable::TransformerWindowSize));
            assert!(
                !contract
                    .requires(MemoryStrategy::BoundedTransformerResidency)
                    .any(|prerequisite| matches!(
                        prerequisite,
                        MemoryStrategyPrerequisite::Rung { .. }
                    )),
                "an unavailable rung must not carry a prerequisite edge"
            );
            // Rung 3 is load-shape independent: it bounds scratch, not residency.
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedAttention)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Implemented
            );
        }

        // A Resident load can stream in principle, but rung 4's declared prerequisite is rung 1,
        // whose availability IS the Sequential policy — so declaring it would advertise a
        // composition the contract itself can never satisfy.
        let resident = weights_free_memory_strategy_contract(
            crate::MODEL_ID,
            &streamable_spec(OffloadPolicy::Resident),
        )
        .unwrap();
        assert!(resident.conformance_errors().is_empty());
        assert_eq!(
            resident
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        // The capability, the lifecycle hook and the formula variable are ONE binding: a contract
        // that withheld the rung but kept advertising the hook or the variable would tell a caller
        // its peak depends on a cadence it cannot select.
        assert!(!resident.lifecycle.transformer_window_materialization);
        let MemoryFormulaKind::PhaseEnvelope { variables, .. } = &resident.formula else {
            panic!("SANA declares a phase envelope")
        };
        assert!(!variables.contains(&MemoryFormulaVariable::TransformerWindowSize));
        assert!(variables.contains(&MemoryFormulaVariable::AttentionChunkSize));
    }

    /// **Every published parameter is refused outside its domain, on the production layer.**
    #[test]
    fn request_scoped_parameters_are_refused_outside_the_published_domain() {
        let spec = streamable_spec(OffloadPolicy::Sequential);
        let full = GenerationMemory {
            stage_residency: true,
            tile_vae_decode: true,
            chunk_attention: true,
            stream_transformer_blocks: true,
            decode_tile_edge: Some(DECODE_TILE_EDGE as u32),
            decode_overlap: Some(DECODE_OVERLAP as u32),
            attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
            transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
            transformer_window_component: Some(TransformerComponent::Dit),
            ..Default::default()
        };
        validate_request_memory(crate::MODEL_ID, &spec, &full)
            .expect("the fully published selection must be accepted");
        // Every published value in every domain is reachable.
        for edge in DECODE_TILE_EDGES {
            let mut memory = full;
            memory.decode_tile_edge = Some(*edge);
            validate_request_memory(crate::MODEL_ID, &spec, &memory).unwrap();
        }
        for size in ATTENTION_CHUNK_SIZES {
            let mut memory = full;
            memory.attention_chunk_size = Some(*size);
            validate_request_memory(crate::MODEL_ID, &spec, &memory).unwrap();
        }
        for window in TRANSFORMER_WINDOW_SIZES {
            let mut memory = full;
            memory.transformer_window_size = Some(*window);
            validate_request_memory(crate::MODEL_ID, &spec, &memory).unwrap();
        }

        let mut refusals = Vec::new();
        let mut check = |label: &str, memory: GenerationMemory| {
            if validate_request_memory(crate::MODEL_ID, &spec, &memory).is_ok() {
                refusals.push(label.to_owned());
            }
        };
        for rejected in DECODE_TILE_EDGES_REJECTED {
            check(
                "decode edge",
                GenerationMemory {
                    decode_tile_edge: Some(*rejected),
                    ..full
                },
            );
        }
        for rejected in ATTENTION_CHUNK_SIZES_REJECTED.iter().chain(&[0, 7, 999]) {
            check(
                "attention budget",
                GenerationMemory {
                    attention_chunk_size: Some(*rejected),
                    ..full
                },
            );
        }
        for rejected in [0_u32, 3, 6, 7, 11, 20, 21] {
            check(
                "window cadence",
                GenerationMemory {
                    transformer_window_size: Some(rejected),
                    ..full
                },
            );
        }
        for component in [
            TransformerComponent::TextEncoder,
            TransformerComponent::Both,
        ] {
            check(
                "window component",
                GenerationMemory {
                    transformer_window_component: Some(component),
                    ..full
                },
            );
        }
        check(
            "rung 4 without rung 1",
            GenerationMemory {
                stage_residency: false,
                ..full
            },
        );
        assert!(
            refusals.is_empty(),
            "these out-of-domain selections were silently admitted: {refusals:?}"
        );

        // …and rung 4 is refused outright on a load that cannot stream it.
        let mut eager = spec.clone();
        eager.load_shape = LoadShape::EagerMaterialization;
        assert!(validate_request_memory(crate::MODEL_ID, &eager, &full).is_err());
        // …while a request that selects no rung is untouched by any of this.
        validate_request_memory(crate::MODEL_ID, &eager, &GenerationMemory::default()).unwrap();
    }

    #[test]
    fn decode_domain_and_rejection_set_are_pinned() {
        assert_eq!(DECODE_TILE_EDGES, &[512, 384, 256, 192]);
        assert_eq!(DECODE_TILE_EDGES_REJECTED, &[128, 96, 64]);
        assert_eq!(DECODE_TILE_EDGE, 192);
        assert_eq!(DECODE_OVERLAP, 48);
        let routes = decode_routes(crate::MODEL_ID).unwrap();
        routes.validate(false, Some(192), Some(48)).unwrap();
        assert!(routes.validate(false, Some(128), Some(48)).is_err());
        assert!(routes.validate(true, Some(192), Some(48)).is_err());
    }

    #[test]
    fn calibration_identity_is_split_by_offload_policy() {
        let resident =
            weights_free_memory_strategy_contract(crate::MODEL_ID, &spec(OffloadPolicy::Resident))
                .unwrap();
        let sequential = weights_free_memory_strategy_contract(
            crate::MODEL_ID,
            &spec(OffloadPolicy::Sequential),
        )
        .unwrap();
        assert_ne!(resident.calibration, sequential.calibration);
        assert_eq!(
            resident
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert_eq!(
            sequential
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        assert!(resident.pid_decode_routes.is_none());
        assert!(sequential.pid_decode_routes.is_none());
    }
}
