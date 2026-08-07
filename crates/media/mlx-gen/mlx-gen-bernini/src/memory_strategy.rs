//! Bernini MLX adoption of the shared memory-strategy contract (sc-15528) — the **complete** ladder
//! over a dual-expert Wan2.2-A14B trunk.
//!
//! Both registered Bernini providers adopt it. `bernini` is the entry the SceneWorks catalog's
//! `bernini_image` maps onto (the worker sets `frames: Some(1)`); `bernini_renderer` is the
//! renderer-only sibling. They share one architecture, one staged phase order, one tiled VAE decode,
//! one bounded-attention primitive and one block stream, and differ only in whether the Qwen2.5-VL
//! planner runs — so they share one contract builder. Sharing code is explicitly NOT what makes a
//! catalog entry Verified: each entry still owes its own per-tier evidence.
//!
//! ## Declared rungs
//!
//! | Rung | Support | Executable seam |
//! |---|---|---|
//! | 0 Resident | Implemented | Both experts co-resident for the whole denoise — today's shipped behaviour |
//! | 1 Staged residency | Implemented (**unconditional**) | Every completed phase dropped + `clear_cache`d before the next: conditioning → source VAE → experts → decode |
//! | 2 Bounded decode | Implemented | [`TilingConfig::spatial_only`] over the [`DECODE_TILE_EDGES`] ladder, replacing `TilingConfig::auto` |
//! | 3 Bounded attention | Implemented (trunk) | [`mlx_gen::attention::sdpa_budgeted_bhsd`] through both trunk SDPA seams, on both experts |
//! | 4 Bounded transformer residency | Implemented (deferred-materialization loads) | [`mlx_gen::block_residency::run_windowed`] over the **80-block** trunk |
//!
//! ## Why the trunk is 80 blocks and not 40 (sc-16354)
//!
//! `bernini.rs` refuses a single-expert snapshot outright (`if !config.dual_model { return Err(…) }`)
//! and every `dual_model` Wan config is `num_layers: 40`, so the trunk is `2 x 40 = 80` blocks,
//! code-enforced, and this is the only shape the family has. sc-16354 raised the hypothesis that a
//! naive per-expert window would bound the active expert while the **idle** expert's 40 blocks stayed
//! fully resident — buying at most half of what the arithmetic suggests, and possibly nothing at the
//! request level.
//!
//! This implementation does not inherit that conclusion; it removes its precondition. Rung 4 loads
//! **both** experts deferred ([`WanTransformer::from_weights_deferred`](mlx_gen_wan::WanTransformer::from_weights_deferred)),
//! so neither expert holds any blocks and there is no idle half to pay for. The window is therefore
//! declared and validated over the whole 80-block trunk in one global index space — high-noise blocks
//! are `0..40`, low-noise blocks are `40..80` — rather than as two independent 40-block plans.
//!
//! That global indexing is load-bearing and it constrains the published cadence domain: the shared
//! [`MlxRequestScopeCore`](mlx_gen::request_scope::MlxRequestScopeCore) requires a window start
//! aligned to the window size, and the low expert's first block is index 40, so **every published
//! window size must divide 40**. [`TRANSFORMER_WINDOW_SIZES`] is exactly the set of proper divisors
//! of 40, and [`the_published_window_sizes_all_divide_one_expert`](self) pins it.
//!
//! The cheap half of the same problem — releasing the *high* expert at the monotone boundary switch
//! so only one expert is resident during the denoise — is NOT landed here and is not claimed: see
//! `_RUNG_ONE_IS_UNCONDITIONAL`. Rung 4 achieves strictly more (zero blocks of both experts), so
//! nothing in this ladder depends on it.
//!
//! ## The cost side of rung 4, priced honestly
//!
//! A window plan is re-walked once per forward, and the guided-velocity modes run several full
//! forwards per step — `VitMode::VaeTxtVitWapg`, the `bernini_image` default, runs four. At the
//! variant's real 40 denoise steps that is `40 blocks x 40 steps x 4 passes = 6400` window
//! materializations against one expert per render. That is a latency consequence to price into the
//! chosen cadence, not a correctness problem, and it is why [`TRANSFORMER_WINDOW_SIZE`] defaults to a
//! wide cadence rather than to 1. The full domain down to 1 is still published, because narrowing a
//! mechanism's advertised surface on a latency judgement the caller is better placed to make is not
//! this contract's job.
//!
//! ## Disclosure: an optimized request may EVICT the warm cross-request cache
//!
//! Rung 4 is request-scoped and never materializes a block at all, so a warm generator that served a
//! windowed request has no block residency to hand the next one — every subsequent request re-reads
//! the stack it needs. Stated here rather than discovered as a latency regression.
//!
//! ## What is NOT declared, and why
//!
//! * **PiD decode.** Bernini decodes through the Wan z16 `AutoencoderKLWan`; the PiD students are the
//!   FLUX and Qwen-Image ones. The crate has no `mlx-gen-pid` dependency, so `use_pid` is rejected at
//!   admission rather than silently degraded to the native decode.
//! * **Adapters.** The descriptor advertises `supports_lora: false`, and Wan MERGES adapter deltas
//!   into the weight map at load, so a streamed block re-read from the snapshot would silently carry
//!   none of them. [`WanBlockStream::new`](mlx_gen_wan::WanBlockStream) refuses an adapted load.
//! * **`TransformerComponent::TextEncoder` / `Both`.** See [`TRANSFORMER_WINDOW_COMPONENTS`].

use mlx_gen::gen_core::{
    standard_memory_strategy_safety_check, Error as CoreError, MemoryBackendRealization,
    MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategySupport, ResidentRequestMemory, Result as CoreResult,
    TransformerComponent,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{GenerationRequest, LoadShape, LoadSpec};
use mlx_gen_wan::config::WanModelConfig;

/// The full Bernini pipeline — the provider the SceneWorks `bernini_image` catalog entry resolves to.
pub const FULL_ID: &str = crate::bernini::MODEL_ID;
/// The renderer-only sibling.
pub const RENDERER_ID: &str = crate::pipeline::MODEL_ID;

/// Both adopting providers, in registration order.
pub const PROVIDER_IDS: [&str; 2] = [RENDERER_ID, FULL_ID];

/// Calibration identity for the weights-free registry conformance walk.
///
/// Deliberately distinct from [`MEMORY_CALIBRATION_FINGERPRINT`]: this one describes *declaration*
/// behaviour over a synthetic spec, and must never be mistaken for a measured production cell.
const STATIC_CALIBRATION: &str = "bernini-mlx-registry-behavior-v1";

/// The production calibration identity, minted once a cell has real-weight evidence behind it.
///
/// Not yet returned by `production_calibration_fingerprint` for any load: no `MEMORY_EVIDENCE_V1`
/// record exists for this family. Until one does, `contract_for` carries `calibration: None`, which
/// is what makes `MemoryEvidence::optimized_eligibility` refuse every optimized fit — the resident
/// path still runs, and no selector can claim a verified saving this repository cannot show.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "bernini-image-q4-mlx-dual-expert-ladder-v1";

/// Rung 2 — production decode tile edges, in **output pixels**.
///
/// `TilingConfig` converts to latent by dividing by the VAE's spatial scale (8 for the z16
/// `AutoencoderKLWan`), so every published edge is a multiple of 8 and lands on a whole latent cell.
/// The floor is geometric rather than measured: a tile must exceed twice the overlap by at least one
/// latent cell or successive tiles do not advance, which puts the smallest admissible edge at
/// `2 * DECODE_OVERLAP + 8 = 136`. 256 is the first published multiple comfortably above it.
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512, 384, 320, 256];

/// The default edge when a request enables rung 2 without naming one.
pub const DECODE_TILE_EDGE: u32 = 512;

/// The single published overlap. One overlap per route keeps admission able to reject a geometry
/// assembled for a different one.
pub const DECODE_OVERLAP: u32 = 64;

/// Rung 3 — the shared constrained score budget. Bernini does not invent its own.
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;

/// Rung 4 — the published window cadences, in blocks.
///
/// **Every value divides 40**, which is not a style choice: the trunk is indexed globally across both
/// experts (`0..40` high, `40..80` low) and the shared request scope requires a window start aligned
/// to the window size, so a cadence that does not divide one expert's depth would leave the low
/// expert's first window mis-aligned. `40` itself is excluded because it degenerates to fully
/// resident (`BlockPlan::is_bounded` is false there), which is rung 0 wearing rung 4's name.
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 4, 5, 8, 10, 20];

/// The default cadence when a request enables rung 4 without naming one.
///
/// 10 rather than 1. At 1 the trunk is re-walked `40 x 40 x 4 = 6400` times per render for the
/// tightest possible bound; 10 costs a tenth of that for a bound still an order of magnitude below
/// resident. A caller that wants 1 can still ask for it — the domain publishes it.
pub const TRANSFORMER_WINDOW_SIZE: u32 = 10;

/// The rung-4 component scope this family implements.
pub const TRANSFORMER_WINDOW_COMPONENT: TransformerComponent = TransformerComponent::Dit;

/// The component scopes this family implements — the DiT trunk only, and the reason is structural
/// rather than inherited (sc-15794 says a scope decision must not be copied from a sibling).
///
/// Bernini's conditioning networks are the largest in the catalog: a Qwen2.5-VL-7B planner backbone, a
/// NEO vision embedder, a clip-diff head and a UMT5-XXL text encoder. They are also **completely
/// released before either expert is loaded** — `generate_impl` drops each one and calls `clear_cache`
/// at the phase boundary, unconditionally, on both providers. A `TextEncoder`-scoped window would
/// therefore bound a phase that has already returned its bytes by the time the request reaches its
/// high-water mark, which is the expert phase.
///
/// That is an argument about *where the peak is*, and it is backed by measurement already on this
/// branch's parent: `sequential_residency_real_weights.rs` bounds the observed peak to the expert
/// phase. It is **not** a claim that the conditioning phase is small — it is a claim that it is not
/// concurrent with the peak. `TextEncoder` and `Both` are consequently declared unimplemented and
/// rejected with a typed error rather than silently narrowed to `Dit`, and a future story that wants
/// them must first show a route where the conditioning phase IS the peak.
pub const TRANSFORMER_WINDOW_COMPONENTS: &[TransformerComponent] = &[TRANSFORMER_WINDOW_COMPONENT];

/// One Wan expert's block depth, read from the model config rather than written as `40`.
fn expert_blocks() -> usize {
    WanModelConfig::wan22_t2v_14b().num_layers
}

/// The whole trunk's block depth — both experts, in the one global index space rung 4 windows over.
pub fn trunk_blocks() -> usize {
    2 * expert_blocks()
}

/// Whether THIS load can execute rung 4.
///
/// Two independent facts decide it, and only one of them is a [`LoadShape`]:
///
/// 1. The window rebuilds blocks from the snapshot, so it needs a **re-openable source**. Both
///    providers reject anything but a `WeightsSource::Dir`, so a load that got this far is
///    re-openable by construction.
/// 2. The load must not have bulk-committed the stack. [`LoadShape::DeferredMaterialization`] is the
///    shared contract's declared prerequisite for rung 4, and it is checked per LOAD, not per
///    provider: a window over an already-materialized trunk bounds nothing, it *adds* a copy on top.
///
/// Adapters would be a third fact, but the descriptor advertises `supports_lora: false`, so an
/// adapted load cannot reach here. The refusal still lives in
/// [`WanBlockStream::new`](mlx_gen_wan::WanBlockStream) where the mechanism is, so a future provider
/// that flips the capability bit cannot silently ship un-adapted streamed blocks.
pub fn structurally_streamable(spec: &LoadSpec) -> bool {
    matches!(spec.weights, mlx_gen::WeightsSource::Dir(_))
        && spec.load_shape == LoadShape::DeferredMaterialization
        && spec.adapters.is_empty()
}

fn known_provider(provider_id: &str) -> CoreResult<()> {
    PROVIDER_IDS.contains(&provider_id).then_some(()).ok_or_else(|| {
        CoreError::Msg(format!(
            "bernini memory strategy: unknown provider `{provider_id}`; expected one of {PROVIDER_IDS:?}"
        ))
    })
}

/// The measured production key, or `None`.
///
/// `None` for every load today: no `MEMORY_EVIDENCE_V1` record exists for this family, so there is no
/// key to hand a selector. Returning a fingerprint here without a record behind it is precisely the
/// "unknown, stale, or fingerprint-mismatched evidence selects a claimed fit" failure the epic
/// forbids, so this function stays honest and the ladder ships selectable-but-uncalibrated: the
/// mechanisms are reachable through an explicit request, and no automatic optimized fit is claimed.
///
/// When the first cell is measured, this returns [`MEMORY_CALIBRATION_FINGERPRINT`] for exactly the
/// measured axes — provider, precision, packed tier and route — and nothing else, the same way
/// Chroma's does.
fn production_calibration_fingerprint(
    _provider_id: &str,
    _spec: &LoadSpec,
) -> Option<&'static str> {
    None
}

/// The public contract for a loaded provider.
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let facts = crate::pipeline::component_footprint(spec)?;
    build_contract(
        provider_id,
        spec,
        conditioning_bytes(provider_id, spec, facts.text_encoder),
        facts.dit,
        facts.vae,
        production_calibration_fingerprint(provider_id, spec)
            .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape)),
    )
}

/// Declaration-equivalent contract used by weights-free registry conformance. Structure, parameter
/// domains and prerequisites are identical; only the measured asset facts are absent, and the
/// calibration identity is the static declaration key rather than a production one.
pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    build_contract(
        provider_id,
        spec,
        0,
        0,
        0,
        Some(MemoryCalibrationIdentity::new(
            STATIC_CALIBRATION,
            spec.load_shape,
        )),
    )
}

/// The conditioning bytes this provider actually loads.
///
/// [`component_footprint`](crate::pipeline::component_footprint) counts only `t5_encoder` because the
/// worker's fit gate consumes that split. The **full** pipeline additionally loads the Qwen2.5-VL-7B
/// planner backbone, the MLP connector, the ViT decoder and the MAR mask tokens before the experts —
/// on the shipped q4 tier that is another ~9 GB, and charging zero for it would under-price the
/// conditioning phase by more than the whole text encoder.
fn conditioning_bytes(provider_id: &str, spec: &LoadSpec, text_encoder_bytes: u64) -> u64 {
    if provider_id != FULL_ID {
        return text_encoder_bytes;
    }
    let mlx_gen::WeightsSource::Dir(root) = &spec.weights else {
        return text_encoder_bytes;
    };
    let planner = mlx_gen::gen_core::PerComponentBytes::from_root_subdirs(
        root,
        &[
            "qwen2_5_vl.safetensors",
            "connector.safetensors",
            "vit_decoder.safetensors",
            "mask_tokens.safetensors",
        ],
        &[],
        &[],
    );
    text_encoder_bytes.saturating_add(planner.text_encoder)
}

fn build_contract(
    provider_id: &str,
    spec: &LoadSpec,
    conditioning_bytes: u64,
    transformer_bytes: u64,
    decoder_bytes: u64,
    calibration: Option<MemoryCalibrationIdentity>,
) -> CoreResult<MemoryProviderContract> {
    known_provider(provider_id)?;
    let streamable = structurally_streamable(spec);
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            // Unified memory: the wired-residency budget is what the staged phases release, weights
            // are mmap-backed and lazy per tensor, and MLX's lazy graph needs an explicit `eval`
            // before a phase drop (or a window drop) frees anything.
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.calibration = calibration;
    // Bernini's shipped default is BOTH experts co-resident, which is exactly what rung 0 means here,
    // so the resident selection preserves the load defaults rather than writing an all-disabled
    // block. The cross-phase staging that runs unconditionally is not a lever and is not claimed as
    // one: rung 1's lever is the expert sequencing.
    contract.resident_request_memory = ResidentRequestMemory::PreserveLoadDefaults;

    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::DecodeTileArea,
        MemoryFormulaVariable::AttentionChunkSize,
        MemoryFormulaVariable::TransformerWindowSize,
    ];
    contract.formula = MemoryFormulaKind::PhaseEnvelope { phases, variables };

    // No auxiliary resident network: no control branch, no IP-adapter, no identity encoder, and
    // `supports_lora: false`. `overlay_bytes` is therefore 0 as a positive statement, not an omission.
    contract.asset_facts.overlay_bytes = 0;
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
        synchronized_phase_release: true,
        decode_tiling: true,
        attention_chunking: true,
        transformer_window_materialization: streamable,
    };

    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident | MemoryStrategy::StagedResidency => {
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedDecode => {
                capability.parameters.decode_tile_edges = DECODE_TILE_EDGES.to_vec();
                capability.parameters.decode_overlaps = vec![DECODE_OVERLAP];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedAttention => {
                capability.parameters.attention_chunk_sizes = vec![ATTENTION_CHUNK_SIZE];
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency if streamable => {
                capability.parameters.transformer_window_sizes = TRANSFORMER_WINDOW_SIZES.to_vec();
                capability.parameters.transformer_window_components =
                    TRANSFORMER_WINDOW_COMPONENTS.to_vec();
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
        };
    }

    // Deliberately NO rung-4 -> rung-1 `additional_prerequisites` edge. Chroma and Anima declare one
    // because their window bounds the DiT's *denoise-phase* residency only, so without the phase
    // release the conditioner and VAE sit alongside it and the request peak does not move. Bernini's
    // conditioning phase is already released unconditionally before either expert loads, so the
    // denoise phase IS the peak with or without rung 1 — and rung 4's deferred load holds zero blocks
    // of BOTH experts, which is strictly more than rung 1's sequencing achieves. Adding the edge here
    // would force a caller to pay rung 1's warm-cache eviction for a saving rung 4 already has, and
    // `MemoryStrategy::engages` is explicit that rung 4 does not universally engage rung 1.
    Ok(contract)
}

/// Reject a decode geometry outside the published domain.
fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> CoreResult<()> {
    let edge = edge.ok_or_else(|| {
        CoreError::Unsupported("bernini: bounded decode requires a tile edge".to_owned())
    })?;
    let overlap = overlap.ok_or_else(|| {
        CoreError::Unsupported("bernini: bounded decode requires a tile overlap".to_owned())
    })?;
    if !DECODE_TILE_EDGES.contains(&edge) {
        return Err(CoreError::Unsupported(format!(
            "bernini: decode tile edge {edge} is outside the published domain {DECODE_TILE_EDGES:?}"
        )));
    }
    if overlap != DECODE_OVERLAP {
        return Err(CoreError::Unsupported(format!(
            "bernini: decode overlap {overlap} is not the published {DECODE_OVERLAP}"
        )));
    }
    Ok(())
}

/// Reject a window cadence outside the published domain.
fn validate_window(size: u32) -> CoreResult<()> {
    if !TRANSFORMER_WINDOW_SIZES.contains(&size) {
        return Err(CoreError::Unsupported(format!(
            "bernini: transformer window {size} is outside the published domain \
             {TRANSFORMER_WINDOW_SIZES:?}; every published cadence divides one expert's \
             {}-block depth so the low expert's first window stays aligned in the {}-block trunk",
            expert_blocks(),
            trunk_blocks()
        )));
    }
    Ok(())
}

/// Reject an attention chunk outside the published domain.
fn validate_attention(size: u32) -> CoreResult<()> {
    if size != ATTENTION_CHUNK_SIZE {
        return Err(CoreError::Unsupported(format!(
            "bernini: attention chunk size {size} is not the published {ATTENTION_CHUNK_SIZE}"
        )));
    }
    Ok(())
}

pub(crate) fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        // sc-15839 review defect, Resident+PiD admission. Bernini decodes through the Wan z16 VAE and
        // the crate carries no `mlx-gen-pid` dependency at all, so a PiD selection would silently
        // execute the ordinary native decode — a different strategy than the selector chose. Reject
        // at admission rather than degrade.
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{}: bernini decodes through the Wan z16 AutoencoderKLWan and has no PiD student; \
                 the overlay is not implementable on this provider",
                contract.provider_id
            )));
        }
        // sc-15839 review defect, unconstrained batch geometry. `max_count: 1` on both descriptors,
        // so a batched admission would record evidence for a route `validate` rejects anyway.
        if context.geometry.batch != 1 {
            return Err(CoreError::Unsupported(format!(
                "{}: bernini renders one image per request (max_count = 1); got batch {}",
                contract.provider_id, context.geometry.batch
            )));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            validate_decode(
                context.selection.parameters.decode_tile_edge,
                context.selection.parameters.decode_overlap,
            )?;
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedAttention) {
            if let Some(size) = context.selection.parameters.attention_chunk_size {
                validate_attention(size)?;
            }
        }
        if contract.engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        ) {
            if !contract.lifecycle.transformer_window_materialization {
                return Err(CoreError::Unsupported(format!(
                    "{}: bounded transformer residency requires a DeferredMaterialization load",
                    contract.provider_id
                )));
            }
            // The scope AND the cadence are checked here as well as by the shared parameter
            // validator, so a request that reached the provider by another path still cannot ask for
            // a scope this family does not implement or a cadence that would mis-align the low
            // expert's first window.
            let component = context.selection.parameters.window_component();
            if !TRANSFORMER_WINDOW_COMPONENTS.contains(&component) {
                return Err(CoreError::Unsupported(format!(
                    "{}: transformer window component {component:?} is not implemented; bernini \
                     releases every conditioning network before either expert loads, so a window \
                     over them cannot move the request peak",
                    contract.provider_id
                )));
            }
            if let Some(size) = context.selection.parameters.transformer_window_size {
                validate_window(size)?;
            }
        }
        Ok(())
    };
    standard_memory_strategy_safety_check(
        contract,
        context,
        Some(loaded_tier(spec)),
        Some(&route_gate),
    )
}

/// The numeric tier this generator actually runs.
///
/// Every shipped Bernini tier is **pre-packed** — `config.json` carries a `quantization` block and
/// the loader packed-detects off the on-disk `.scales` — so `LoadSpec::quantize` records the resolved
/// tier rather than a load-time transform request.
fn loaded_tier(spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    }
}

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract.capability(strategy).map(|c| &c.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        loaded_tier(spec),
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: true,
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

pub fn begin_request(
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
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        // The WHOLE trunk, both experts, in one global index space — see the module docs. Derived
        // from the model config rather than written as 80.
        trunk_blocks(),
        move |use_pid, edge, overlap| {
            if use_pid {
                return Err(CoreError::Unsupported(format!(
                    "{provider_id}: bernini has no PiD decoder"
                )));
            }
            validate_decode(Some(edge), Some(overlap))
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

/// The registry rows for one adopting provider.
///
/// Both providers get the full set — contract, safety check, weights-free contract fixture and
/// behaviour — because a half-registered family is exactly the "declaration is not reachability"
/// hazard: a contract nothing resolves through cannot be walked by registry conformance, and a
/// behaviour registration without a contract fixture cannot be exercised weights-free.
macro_rules! memory_registration {
    ($registration:ident, $behavior:ident, $provider_id:expr) => {
        pub(crate) const $registration: mlx_gen::gen_core::MemoryRegistration =
            mlx_gen::gen_core::MemoryRegistration {
                provider_id: $provider_id,
                contract: |spec| {
                    crate::memory_strategy::memory_strategy_contract($provider_id, spec)
                },
                safety_check: crate::memory_strategy::safety_check,
            };
        pub(crate) const $behavior: mlx_gen::gen_core::MemoryBehaviorRegistration =
            mlx_gen::gen_core::MemoryBehaviorRegistration {
                provider_id: $provider_id,
                valid_fixtures: crate::memory_strategy::registered_valid_fixture,
                begin_request: |spec, contract, context| {
                    crate::memory_strategy::registered_begin_request(
                        $provider_id,
                        spec,
                        contract,
                        context,
                    )
                },
            };
    };
}

memory_registration!(
    RENDERER_MEMORY_REGISTRATION,
    RENDERER_MEMORY_BEHAVIOR,
    RENDERER_ID
);
memory_registration!(FULL_MEMORY_REGISTRATION, FULL_MEMORY_BEHAVIOR, FULL_ID);

// ── Request-side resolution: the shared `GenerationMemory` signal → this provider's levers ────────

/// Rung 3: the budget applied to every trunk SDPA seam.
///
/// A budget rather than an `AttentionPlan` because a `WanBlockStream` outlives the borrow a plan's
/// cancel flag would need. The denoise loop already checks cancellation once per step and a rung-3
/// chunk is far shorter than a step, so nothing is lost.
///
/// **Scope, stated precisely.** This covers both per-block SDPA seams of the denoise trunk — the
/// self-attention over the packed `[sources…, target]` sequence and the cross-attention over the
/// prompt streams — on both experts, resident or windowed. It does **not** cover the planner's
/// hand-rolled softmax (`qwen2_5_vl.rs`, `vision.rs`): those run in the conditioning phase, which is
/// fully released before either expert loads, so bounding them cannot move the request peak the way
/// bounding the trunk can. Tracked separately; see the story linked on sc-15528.
pub(crate) fn attention_budget(req: &GenerationRequest) -> mlx_gen::attention::AttentionBudget {
    match req.memory {
        Some(memory) if memory.chunk_attention => mlx_gen::attention::AttentionBudget::CONSTRAINED,
        _ => mlx_gen::attention::AttentionBudget::UNBOUNDED,
    }
}

/// Rung 4: the requested cadence, or `None` for the resident stack.
///
/// A scope this family does not implement — or a cadence outside the published
/// [`TRANSFORMER_WINDOW_SIZES`] — is a typed rejection rather than a silently narrowed (or silently
/// *widened*) execution.
pub(crate) fn transformer_window_size(req: &GenerationRequest) -> mlx_gen::Result<Option<usize>> {
    let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
        return Ok(None);
    };
    let component = memory
        .transformer_window_component
        .unwrap_or(TRANSFORMER_WINDOW_COMPONENT);
    if !TRANSFORMER_WINDOW_COMPONENTS.contains(&component) {
        return Err(mlx_gen::Error::Unsupported(format!(
            "bernini implements only the {TRANSFORMER_WINDOW_COMPONENT:?} transformer window \
             component, got {component:?}"
        )));
    }
    let size = memory
        .transformer_window_size
        .unwrap_or(TRANSFORMER_WINDOW_SIZE);
    validate_window(size).map_err(|error| {
        mlx_gen::Error::Unsupported(format!("bernini transformer window rejected: {error}"))
    })?;
    Ok(Some(size as usize))
}

/// Rung 2: the decode tiling for this request.
///
/// `None` means "this request did not select rung 2", and the caller then keeps
/// [`TilingConfig::auto`] — Bernini's shipped behaviour, which already tiles a large decode. Rung 2
/// is therefore *explicit geometry*, not "tiling versus no tiling": the A/B arms differ in exactly
/// one thing, the tile edge, and the harness must compare against the composition this extends
/// rather than against an untiled decode that never shipped.
pub(crate) fn decode_tiling(req: &GenerationRequest) -> mlx_gen::Result<Option<TilingConfig>> {
    let Some(memory) = req.memory.filter(|memory| memory.tile_vae_decode) else {
        return Ok(None);
    };
    let edge = memory.decode_tile_edge.unwrap_or(DECODE_TILE_EDGE);
    let overlap = memory.decode_overlap.unwrap_or(DECODE_OVERLAP);
    validate_decode(Some(edge), Some(overlap)).map_err(|error| {
        mlx_gen::Error::Unsupported(format!("bernini decode tiling rejected: {error}"))
    })?;
    Ok(Some(TilingConfig::spatial_only(
        edge as i32,
        overlap as i32,
    )))
}

/// Rung 1 has **no request-side resolver, deliberately**, and this note is the declaration.
///
/// Both `generate_impl`s release every completed phase unconditionally: the UMT5 encoder is dropped
/// and `clear_cache`d before the source-VAE encode, the source-VAE encoder before the experts, and
/// the experts before the decode. `GenerationMemory::stage_residency` therefore selects nothing —
/// the provider is *structurally* always-staged, which is why both descriptors already advertise
/// `supports_sequential_offload` and why `advertises_sequential_offload` pins it on each.
///
/// A resolver that branched on the flag would be a lever over behaviour that does not vary, i.e. a
/// declaration with no enforcement behind it. The rung is `Implemented` because the synchronized
/// phase release is real and executed, not because a flag is read.
///
/// **What this does NOT include**, stated so it is not read as covered: releasing the *high* expert
/// at the boundary switch so only one expert is resident during the denoise. The switch is monotone,
/// so that is sound, and sc-16354 identifies it as the cheap half of the dual-expert problem — but it
/// needs `BVitExpert`/`BExpert` to own their transformer rather than borrow it, which is a larger
/// refactor than this story lands. Rung 4 already achieves strictly more (zero blocks of BOTH
/// experts), so nothing here depends on it. Tracked separately; see the story linked on sc-15528.
const _RUNG_ONE_IS_UNCONDITIONAL: () = ();

/// The strategy parameters this provider accepts, for a caller that wants the whole domain in one
/// value (the conformance tests and the SceneWorks evidence writer both key off this).
pub fn declared_parameters() -> mlx_gen::gen_core::MemoryStrategyParameters {
    mlx_gen::gen_core::MemoryStrategyParameters {
        decode_tile_edge: Some(DECODE_TILE_EDGE),
        decode_overlap: Some(DECODE_OVERLAP),
        attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
        transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
        transformer_window_component: Some(TRANSFORMER_WINDOW_COMPONENT),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::GenerationMemory;
    use mlx_gen::{OffloadPolicy, WeightsSource};

    fn spec(shape: LoadShape) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent/bernini-contract".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(shape)
    }

    fn contract(provider_id: &str, shape: LoadShape) -> MemoryProviderContract {
        weights_free_memory_strategy_contract(provider_id, &spec(shape)).expect("contract")
    }

    /// **The dual-expert invariant (sc-16354).** The trunk rung 4 windows over is BOTH experts in one
    /// global index space, so the block count the shared request scope validates against is 80.
    ///
    /// A contract that declared 40 would accept a window at block 39 and reject one at block 40 — the
    /// low expert's first block — which is exactly the "windowing one expert while the other stays
    /// resident" shape the survey warned about, expressed as an off-by-one-expert admission bug.
    #[test]
    fn the_windowed_trunk_is_both_experts() {
        assert_eq!(expert_blocks(), 40, "a Wan A14B expert is 40 blocks");
        assert_eq!(
            trunk_blocks(),
            80,
            "rung 4 windows the whole dual-expert trunk, not one expert"
        );
    }

    /// Every published cadence divides one expert's depth, so the low expert's first window (global
    /// block 40) is aligned under the shared scope's `first_block % window == 0` rule. A cadence that
    /// did not divide 40 would be admitted by the contract and then rejected mid-denoise.
    #[test]
    fn the_published_window_sizes_all_divide_one_expert() {
        let expert = expert_blocks() as u32;
        for &size in TRANSFORMER_WINDOW_SIZES {
            assert_eq!(
                expert % size,
                0,
                "window {size} does not divide the {expert}-block expert, so the low expert's \
                 first window would be mis-aligned in the trunk index space"
            );
            assert!(size < expert, "window {size} degenerates to fully resident");
        }
        assert!(
            TRANSFORMER_WINDOW_SIZES.contains(&TRANSFORMER_WINDOW_SIZE),
            "the default cadence must be inside the published domain"
        );
    }

    /// Rung 4 is declared per LOAD. An eager load publishes it `Missing`; only a
    /// `DeferredMaterialization` load publishes it `Implemented`, and every other rung is unchanged
    /// between the two — a load shape must not silently move rungs 1-3.
    #[test]
    fn rung_four_is_declared_per_load_and_moves_nothing_else() {
        for provider_id in PROVIDER_IDS {
            let deferred = contract(provider_id, LoadShape::DeferredMaterialization);
            let eager = contract(provider_id, LoadShape::EagerMaterialization);
            for rung in [
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
            ] {
                assert_eq!(
                    deferred.capability(rung).map(|c| &c.support),
                    Some(&MemoryStrategySupport::Implemented),
                    "{provider_id}: {rung:?} must be implemented on a deferred load"
                );
                assert_eq!(
                    eager.capability(rung).map(|c| &c.support),
                    Some(&MemoryStrategySupport::Implemented),
                    "{provider_id}: {rung:?} must not depend on the load shape"
                );
            }
            assert_eq!(
                deferred
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .map(|c| &c.support),
                Some(&MemoryStrategySupport::Implemented)
            );
            assert_eq!(
                eager
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .map(|c| &c.support),
                Some(&MemoryStrategySupport::Missing),
                "{provider_id}: a window over an already-materialized trunk bounds nothing"
            );
        }
    }

    /// Rung 4 does NOT declare a rung-1 prerequisite on this family, and that is a deliberate
    /// departure from Chroma/Anima justified in the module docs. Pinning it keeps a later
    /// copy-paste from quietly importing an edge that would charge a caller for a saving rung 4
    /// already has.
    #[test]
    fn rung_four_does_not_require_rung_one_here() {
        let contract = contract(FULL_ID, LoadShape::DeferredMaterialization);
        assert!(
            contract.additional_prerequisites.is_empty(),
            "bernini declares no provider-specific prerequisite edges, got {:?}",
            contract.additional_prerequisites
        );
        assert!(
            !contract.engages(
                MemoryStrategy::BoundedTransformerResidency,
                MemoryStrategy::StagedResidency
            ),
            "rung 4 must not drag rung 1 in by cost order"
        );
    }

    /// Every published parameter is accepted and everything outside the domain is refused — on the
    /// production path, by the same validators admission uses.
    #[test]
    fn the_published_domains_are_the_accepted_domains() {
        for &edge in DECODE_TILE_EDGES {
            validate_decode(Some(edge), Some(DECODE_OVERLAP))
                .unwrap_or_else(|error| panic!("published edge {edge} refused: {error}"));
        }
        // Off-domain edges: one below the published floor, one that is not a published multiple, and
        // one from a PiD-shaped domain this provider does not have.
        for edge in [128_u32, 700, 1024] {
            assert!(
                validate_decode(Some(edge), Some(DECODE_OVERLAP)).is_err(),
                "edge {edge} is outside the published domain and must be refused"
            );
        }
        assert!(validate_decode(Some(DECODE_TILE_EDGE), Some(32)).is_err());
        assert!(validate_decode(None, Some(DECODE_OVERLAP)).is_err());
        assert!(validate_decode(Some(DECODE_TILE_EDGE), None).is_err());

        for &size in TRANSFORMER_WINDOW_SIZES {
            validate_window(size).unwrap_or_else(|error| panic!("window {size} refused: {error}"));
        }
        // 3, 6, 7 and 9 are all plausible cadences that do NOT divide 40.
        for size in [0_u32, 3, 6, 7, 9, 40, 41] {
            assert!(
                validate_window(size).is_err(),
                "window {size} must be refused: it does not divide one expert's depth"
            );
        }

        validate_attention(ATTENTION_CHUNK_SIZE).expect("the published chunk size");
        assert!(validate_attention(ATTENTION_CHUNK_SIZE + 1).is_err());
    }

    /// The request-side resolvers refuse the same values the contract refuses, so a request that
    /// bypassed admission still cannot execute an unpublished geometry.
    #[test]
    fn the_request_resolvers_refuse_what_the_contract_refuses() {
        let base = GenerationRequest {
            prompt: "x".into(),
            width: 1024,
            height: 1024,
            count: 1,
            ..Default::default()
        };

        // Absent memory block: every lever off, nothing rejected.
        assert!(decode_tiling(&base).expect("no memory").is_none());
        assert!(transformer_window_size(&base).expect("no memory").is_none());
        assert!(attention_budget(&base).is_unbounded());

        let with = |memory: GenerationMemory| GenerationRequest {
            memory: Some(memory),
            ..base.clone()
        };

        // Rung 2 accepts the published domain and refuses everything else.
        let ok = with(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(384),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        });
        assert!(decode_tiling(&ok).expect("published edge").is_some());
        let bad = with(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(700),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        });
        assert!(decode_tiling(&bad).is_err(), "700 is not published");

        // Rung 4 accepts a published cadence, refuses an unpublished one, and refuses a scope this
        // family does not implement.
        let ok = with(GenerationMemory {
            stream_transformer_blocks: true,
            transformer_window_size: Some(20),
            ..Default::default()
        });
        assert_eq!(transformer_window_size(&ok).expect("published"), Some(20));
        let bad = with(GenerationMemory {
            stream_transformer_blocks: true,
            transformer_window_size: Some(3),
            ..Default::default()
        });
        assert!(
            transformer_window_size(&bad).is_err(),
            "3 does not divide 40"
        );
        let scope = with(GenerationMemory {
            stream_transformer_blocks: true,
            transformer_window_component: Some(TransformerComponent::TextEncoder),
            ..Default::default()
        });
        let error = match transformer_window_size(&scope) {
            Ok(_) => panic!("the TextEncoder scope is not implemented and must be refused"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("TextEncoder"), "{error}");

        // Rung 4 defaulting: an enabled block with no cadence takes the published default, not 1.
        let defaulted = with(GenerationMemory {
            stream_transformer_blocks: true,
            ..Default::default()
        });
        assert_eq!(
            transformer_window_size(&defaulted).expect("default"),
            Some(TRANSFORMER_WINDOW_SIZE as usize)
        );

        // Rung 3 is a plain switch. Rung 1 has no resolver by design — see
        // `_RUNG_ONE_IS_UNCONDITIONAL`.
        let chunked = with(GenerationMemory {
            chunk_attention: true,
            ..Default::default()
        });
        assert!(!attention_budget(&chunked).is_unbounded());
    }

    /// No PiD route, no overlay, one image per request — each stated positively rather than by
    /// omission, because the sc-15839 review found all three shipped as silent gaps elsewhere.
    #[test]
    fn the_absent_capabilities_are_declared_absent() {
        for provider_id in PROVIDER_IDS {
            let contract = contract(provider_id, LoadShape::DeferredMaterialization);
            assert!(
                contract.pid_decode_routes.is_none(),
                "{provider_id}: bernini has no PiD route"
            );
            assert_eq!(
                contract.asset_facts.overlay_bytes, 0,
                "{provider_id}: bernini loads no auxiliary resident network"
            );
            assert_eq!(
                contract.resident_request_memory,
                ResidentRequestMemory::PreserveLoadDefaults
            );
        }
    }

    /// An unknown provider id is refused rather than silently handed the family contract.
    #[test]
    fn an_unknown_provider_is_refused() {
        assert!(weights_free_memory_strategy_contract(
            "bernini_imaginary",
            &spec(LoadShape::DeferredMaterialization)
        )
        .is_err());
    }

    /// No production calibration is minted until a `MEMORY_EVIDENCE_V1` record exists, so no selector
    /// can reach an optimized fit on evidence this repository cannot show. This test is what will
    /// redden — deliberately — on the commit that mints the first cell.
    #[test]
    fn no_production_calibration_is_claimed_without_evidence() {
        for provider_id in PROVIDER_IDS {
            let contract =
                memory_strategy_contract(provider_id, &spec(LoadShape::DeferredMaterialization))
                    .expect("contract");
            assert!(
                contract.calibration.is_none(),
                "{provider_id}: a production calibration identity without a measured record would \
                 let a stale fit be selected"
            );
        }
        // The weights-free walk still gets a declaration key, so registry conformance can run.
        assert!(contract(FULL_ID, LoadShape::DeferredMaterialization)
            .calibration
            .is_some());
    }
}
