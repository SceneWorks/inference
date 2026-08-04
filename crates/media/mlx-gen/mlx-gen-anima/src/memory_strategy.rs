//! Anima MLX adoption of the shared memory-strategy contract (SC-15524) — the **complete** ladder:
//! rung 1 (request-scoped staged residency), rung 2 (bounded Qwen-Image VAE decode), rung 3 (bounded
//! attention) and rung 4 (bounded transformer residency over the 28 Cosmos DiT blocks).
//!
//! All three registered Anima variants — `anima_base`, `anima_aesthetic`, `anima_turbo` — are real
//! providers (no alias, no bespoke route): they share one architecture, one staged
//! [`Residency`](mlx_gen::Residency) seam, one tiled VAE decode, one bounded-attention primitive and
//! one block-residency stream, and differ only in the DiT checkpoint file and their default
//! steps/CFG. So they share one contract builder, and each entry still owes its own per-tier
//! evidence — sharing code is explicitly NOT what makes a catalog entry Verified.
//!
//! ## Declared rungs
//!
//! | Rung | Support | Executable seam |
//! |---|---|---|
//! | 0 Resident | Implemented | Warm [`Residency`](mlx_gen::Residency) pair — Qwen3 TE + (DiT + conditioner + VAE) held across requests |
//! | 1 Staged residency | Implemented (request-scoped) | [`GenerationMemory::stage_residency`](mlx_gen::gen_core::GenerationMemory::stage_residency) drives encode → **drop the Qwen3 TE** → conditioner + denoise → **drop the DiT + conditioner** → decode |
//! | 2 Bounded decode | Implemented | [`QwenVae::decode_tiled`](mlx_gen_qwen_image::QwenVae::decode_tiled) over the [`DECODE_TILE_EDGES`] ladder |
//! | 3 Bounded attention | Implemented | [`mlx_gen::attention::sdpa_budgeted_bhsd`] threaded through BOTH DiT attentions |
//! | 4 Bounded transformer residency | Implemented (deferred-materialization loads) | [`mlx_gen::block_residency::run_windowed`] over the 28 `blocks` |
//!
//! **Rung 4 is declared per LOAD, not per provider.** It needs a re-openable checkpoint AND a load
//! that did not bulk-commit the stack, so it is `Missing` unless
//! [`LoadShape::DeferredMaterialization`](mlx_gen::LoadShape) was selected — see [`streamable`]. An
//! eager load rejects the window rather than silently adding a second block copy on top of the
//! materialized trunk.
//!
//! **Rung 4 additionally requires rung 1 ENGAGED in the same request.** The window bounds the DiT's
//! *denoise-phase* residency; without the phase release the conditioner and VAE stay resident
//! alongside it and the request peak does not move. The prerequisite is declared on the contract, so
//! the shared selector enforces it rather than each caller re-deriving it.
//!
//! ## Disclosure: an optimized request may EVICT the warm cross-request cache
//!
//! Rungs 1 and 4 are request-scoped, so one cached generator serves warm → staged → warm without
//! reconstruction — but that is not free in both directions, and the cost is worth stating plainly
//! rather than discovering as a latency regression:
//!
//! * **Rung 1** evicts the warm pair for the duration of the staged request and rebuilds it lazily on
//!   the next warm one. That is inherent: a phase schedule that releases completed phases cannot also
//!   hold them.
//! * **Rung 4** additionally needs the heavy bundle in its *streamable* component form, which is a
//!   different shape from the warm one. `Residency::ensure_warm_locked` therefore evicts and rebuilds
//!   a warm pair whose shape does not match, so alternating windowed and non-windowed requests on one
//!   generator pays a heavy reload each time it flips.
//!
//! Both are already carried in the contract's declared identity rather than left implicit: the
//! default [`MemoryRuntimeSemantics`](mlx_gen::gen_core::MemoryRuntimeSemantics)
//! `cache` axis keys a cached provider on its **engaged composition**, and the component shape is the
//! typed `load_shape` axis on [`MemoryCalibrationIdentity`] and on the contract itself. A selector
//! that treats a rung-4 admission as free of reload cost is reading neither.
//!
//! This is also why `generate` arms the streamable form **only when the request will actually window**
//! (`window_size.is_some()`), instead of whenever the load permits it: arming it on a load-shape flag
//! alone would evict warm pairs for requests that never use the stream.
//!
//! **This is the MLX column, and it is not the Candle column.** `candle-gen-anima` is a separate
//! delivery and inherits nothing from here — not a rung's presence, not its magnitude, not its
//! mechanism, and above all not a candidate set. Every number in this file was measured on
//! Apple/Metal against the exact untiled/unwindowed reference in
//! `tests/shared_memory_ladder_real_weights.rs`.
//!
//! ## Route coverage
//!
//! Anima is text-to-image only (`Capabilities::conditioning` is empty — there is no reference or
//! control route to cover) and has **no PiD overlay**: the heavy loader ignores `use_pid`, and there
//! is no super-resolving student to plan a second decode domain for. `safety_check` therefore rejects
//! `use_pid` outright rather than admitting a selection this provider would silently execute
//! differently, and the contract publishes no `pid_decode_routes`.
//!
//! Rungs 3 and 4 reach every advertised denoise route: the Turbo CFG-free single forward, and BOTH
//! the cond and uncond forwards of the Base/Aesthetic CFG pair.
//!
//! ## Fail closed — the complete list
//!
//! Every one of these is asserted on real weights by
//! `tests/shared_memory_ladder_real_weights.rs::rung_four_preconditions_fail_closed_on_real_weights`,
//! and each rejection is a typed error rather than a silently narrowed execution:
//!
//! * a PiD selection — this family has no PiD student at all (`safety_check`);
//! * any mode other than text-to-image, or a non-zero reference count (`safety_check`);
//! * rung 4 on an eager load, and rung 4 without rung 1 engaged in the same request;
//! * a transformer window **component** this family does not implement (`TextEncoder` / `Both`) —
//!   never narrowed to `Dit`;
//! * a transformer window **size** outside [`TRANSFORMER_WINDOW_SIZES`] (`validate_window`);
//! * a decode tile edge or overlap outside [`DECODE_TILE_EDGES`] / [`DECODE_OVERLAP`]
//!   (`validate_decode`) — never re-planned onto the default.
//!
//! The last two are the same shape and are enforced on the same two layers on purpose: the
//! admission gate (`safety_check`) *and* the request-side resolver
//! (`transformer_window_size` / `decode_tiling`). A selector that reached the generator by
//! another path — which is exactly what every calibration harness does when it hand-builds a
//! [`GenerationMemory`](mlx_gen::gen_core::GenerationMemory) and calls `generate` — still cannot
//! execute an unmeasured geometry.
//!
//! ## Ownership
//!
//! This file declares *structure and parameter domains only*. Measured coefficients, envelopes and
//! per-tier peaks live in SceneWorks generated evidence keyed by [`MEMORY_CALIBRATION_FINGERPRINT`];
//! the worker owns live-budget accounting and least-cost selection. The scope below is defense in
//! depth: it can reject a selection, never substitute one.

use mlx_gen::gen_core::{
    standard_memory_strategy_safety_check, Error as CoreError, MemoryBackendRealization,
    MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier,
    MemoryParameterRanges, MemoryPhase, MemoryPrerequisiteScope, MemoryProviderContract,
    MemoryRequestScope, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyPrerequisite, MemoryStrategySupport, ResidentRequestMemory, Result as CoreResult,
    TransformerComponent,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{GenerationRequest, LoadShape, LoadSpec};

use crate::config::{DitConfig, Variant};

/// The **default** decode tile edge in output pixels — what a rung-2 request that names no geometry
/// decodes at.
///
/// 640 rather than the tightest candidate, and that is measured: it is the smallest edge that costs
/// **no measurable quality** (max Δ 1/255, mean Δ 0.0107) while still having the best request peak of
/// the drift-free tiles — 4.153 GiB against 768's 4.649. See [`DECODE_TILE_EDGES`] for the sweep. A
/// caller that needs the tighter end of the domain names it; the default does not gamble quality on
/// its behalf.
pub const DECODE_TILE_EDGE: u32 = 640;

/// The decode overlap paired with every advertised edge, in output pixels.
///
/// One overlap, deliberately: the tile ladder trades peak against seam risk on the EDGE, and moving
/// both axes at once would make a calibration row un-attributable to either.
pub const DECODE_OVERLAP: u32 = 64;

/// Anima's production decode tile-edge ladder (SC-15524), output pixels, at [`DECODE_OVERLAP`].
///
/// **The candidate set was measured here, and it is deliberately NOT Z-Image's.** Both families
/// drive `mlx_gen::vae_tiling`, and Anima additionally decodes through the very same
/// `mlx_gen_qwen_image::QwenVae` that Qwen-Image does — so adopting a published ladder would have
/// looked defensible. It is not: what transfers is the tiling *machinery*, never the candidate set.
/// Z-Image publishes `[768, 640, 512]` and rejects everything below 512 because its GroupNorm VAE's
/// per-tile norm statistics drift (max Δ 64→83/255). This decoder does not have that failure: the
/// Qwen VAE's `decode_tiled` runs the whole head — denormalize, `post_quant_conv`, `conv_in` and the
/// mid-block **global** attention — ONCE on the full latent and tiles only the upsample tail, so the
/// statistics Z-Image drifts on are computed globally here. Anima's ladder is consequently more than
/// twice as deep, and its worst admitted drift is max Δ 6/255 rather than 48.
///
/// Swept against the **exact untiled decode of the same latent** on real `anima_base` bf16 weights at
/// 1024², plus the same candidates re-run inside the production staged+windowed envelope
/// (`tests/shared_memory_ladder_real_weights.rs::decode_tile_sweep_bounds_the_admitted_ladder`):
///
/// | tile | isolated decode | max Δ | mean Δ | request peak (run 1 / run 2) |
/// |---:|---:|---:|---:|---:|
/// | untiled | 12.927 GiB | — | — | 7.921 |
/// | 768 | 9.657 | 1 | 0.0106 | 4.649 / 4.649 |
/// | **640** (default) | **9.159** | **1** | **0.0107** | **4.153 / 4.153** |
/// | 512 | 9.265 | 4 | 0.0208 | 4.261 / 4.261 |
/// | 384 | 8.159 | 3 | 0.0242 | 3.176 / 3.069 |
/// | 320 | 8.179 | 3 | 0.0319 | 2.891 / 2.840 |
/// | 256 | 8.083 | 4 | 0.0366 | 3.036 / 2.677 |
/// | 192 | 7.943 | 6 | 0.0589 | 2.680 / 2.680 |
/// | *160* | 7.855 | 7 | 0.0706 | 2.815 |
/// | *128* | 7.737 | 4 | 0.1000 | 2.666 |
/// | *96* | 7.727 | 8 | 0.1685 | 2.493 |
///
/// **Read the two columns differently, because they have different error bars.** The drift columns
/// are deterministic — the sweep decodes ONE materialized latent, so re-running reproduces them to
/// the sixth decimal. The request-peak column is an MLX allocator high-water mark and is not: the
/// second full run of the identical suite moved the 256 row by 0.36 GiB (3.036 → 2.677) while every
/// row at 512 and above reproduced exactly. The spread grows as the tile shrinks and the tile count
/// climbs.
///
/// The ladder therefore stops at 192, for the two-pronged reason the epic asks for:
///
/// 1. **Drift keeps rising below it** — mean Δ 0.0589 → 0.0706 → 0.1000 → 0.1685, a 2.9× increase by
///    96, monotone throughout and reproducible.
/// 2. **And there is no measurable admission win left to pay it with.** Below ~384 the request peaks
///    stop separating: their run-to-run spread (0.36 GiB) is larger than the entire 192 → 96
///    improvement (0.19 GiB in the one run where it was measured), and 160 landed *above* 192 while
///    drifting more. Paying reproducible image drift for a saving that does not survive a re-run is a
///    strictly bad trade.
///
/// This is also why the ladder's own admitted rows are not read as a ranking below 384: 256 beat 320
/// in run 2 and lost to it in run 1. They are a domain the selector may choose from, not an order.
///
/// The ladder is a **domain**, not a recommendation: a larger edge costs more decode memory and fewer
/// forwards, and the selector — not this file — owns the peak-vs-latency choice.
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512, 384, 320, 256, 192];

/// Tile edges swept and **rejected** by measurement (see [`DECODE_TILE_EDGES`]), kept so the sweep can
/// re-assert the exclusion with real numbers rather than leaving it as prose that drifts.
///
/// A future VAE change that made these worth taking would show up as this list failing its rejection
/// check — which is the point. Silently dropping them would leave nothing to notice that with.
pub const DECODE_TILE_EDGES_REJECTED: &[u32] = &[160, 128, 96];

/// The one bounded-attention parameter this provider accepts: the shared
/// [`mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET`] (64 Mi score elements per attention call).
/// Reusing the exact knob every other adopting family measured is what makes a cross-family or
/// cross-backend comparison of rung 3 meaningful. The request scope re-validates it.
///
/// **What it is worth here, measured against this family's own rung-2 top** (`anima_base` bf16,
/// 1024², staged + tiled at 640, `full_shared_ladder_reduces_peak_and_preserves_output`): the
/// denoise phase goes **5.378 → 5.229 GiB**, and the output is unchanged (max Δ 0). Reproducible to
/// the milli-GiB across runs. That is a modest 2.8%, and it is stated rather than left implicit
/// because it is small enough to be mistaken for noise if nobody wrote it down — the ladder arm
/// asserts on exactly this inequality, so a rung that stopped engaging reddens instead of quietly
/// costing the selector a rung it thinks it has.
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;

/// The production transformer-window domain for rung 4: how many of the 28 Cosmos blocks are held
/// materialized at once.
///
/// **It is one value, and that is a measurement rather than a placeholder.**
///
/// Swept on real `anima_base` bf16 weights at 1024², staged, over the rung-3 top
/// (`tests/shared_memory_ladder_real_weights.rs::transformer_window_sweep_shows_the_published_domain_is_exact`):
///
/// | window | denoise peak | production path |
/// |---:|---:|---|
/// | resident (no rung 4) | 5.229 GiB | admitted — request peak **5.229** |
/// | **1** (published) | **1.110** | admitted — request peak **4.151** |
/// | 2 | 1.497 | **refused** (outside the published domain) |
/// | 4 | 1.497 | refused |
/// | 14 | 1.497 | refused |
/// | 28 — one all-covering window, bounds nothing | 1.497 | refused |
///
/// **Read the two columns as two different measurements, because they are.** The denoise column is
/// the *mechanism* — one staged schedule by hand, a fresh heavy bundle per row, denoising through
/// `AnimaHeavy::calibration_denoise_one`, which takes the window raw. That is the only way to
/// measure a cadence this provider refuses to execute, and it is the same split rung 2 already has
/// (the decode sweep measures every candidate through `QwenVae::decode_tiled` and separately proves
/// the rejected edges are refused). The production column is the full request envelope, and it
/// exists only for the cadences a request may actually name; in it the published window 1 measures
/// **1.304 GiB** of denoise inside a **4.151 GiB** request, the small offset from 1.110 being the
/// residency machinery and the count-loop latent the mechanism harness does not hold.
///
/// The request peak is 4.151 at *every* admitted configuration and does not move with the window at
/// all: rung 4 already pushed the binding phase off the denoise onto the decode, so above rung 3 the
/// tile edge is the lever and this parameter is not.
///
/// (Reproducible: a second full run returned all six denoise rows and both request rows **bit for
/// bit** — 1.110 / 1.497 / 1.497 / 1.497 / 1.497 / 5.229, and 1.304 / 4.151.)
///
/// The last row is the load-bearing control: a window of 28 walks the identical driver and bounds
/// *nothing*, and it lands on windows 2, 4 and 14 exactly. So above 1 the window is not what sets
/// the peak — the per-block materialize/drop is, because the loop materializes one block, runs it,
/// and releases it before advancing, and the shared driver's `materialize` callback forces the
/// carried activation first (without which the drop would free nothing, silently).
///
/// Window 1 is the one row that differs, and it differs by a real 26% (1.110 vs 1.497). That is the
/// whole reason the domain has exactly one element rather than either four indistinguishable rows or
/// none: 2/4/14/28 are not four candidates, they are one number repeated, and dropping 1 would give
/// up the only measured saving the parameter has.
///
/// Declaring a wider window while executing at 1 would *over*-predict peak and make the selector
/// refuse fits it could have taken; executing at a wider cadence while declaring 1 would
/// *under*-predict, which is the direction that actually breaks a render. That second direction is
/// closed by `validate_window`, on the same two layers the decode tile edge is closed on — the
/// sweep above reaches the unpublished cadences through the mechanism seam
/// ([`AnimaHeavy::calibration_denoise_one`](crate::pipeline::AnimaHeavy::calibration_denoise_one)),
/// never through the production request path.
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1];

/// The transformer window rung 4 executes at — the single published candidate.
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;

/// The rung-4 **component scopes** this provider implements.
///
/// `Dit` only, and that is a structural fact rather than an omission. The other candidate scope is
/// the text encoder, and Anima's conditioning stack cannot supply it as a *request* saving:
///
/// * the Qwen3-0.6B tower is already **shed by rung 1** before the DiT loads, so windowing it would
///   bound a phase the prerequisite rung already releases; and
/// * the 6-block `AnimaTextConditioner` is not a separate component at all — it is
///   `{prefix}.llm_adapter.*` inside the DiT checkpoint, rides the HEAVY bundle, and is ~1% of it.
///
/// SC-15969's survey noted the encoder is the catalog's smallest TE-window port because it reuses
/// Z-Image's `EncoderLayer`. That remains true and remains beside the point: SC-15998 established
/// that a phase win which does not move the REQUEST peak is not a saving, and on this architecture
/// the conditioning phase is not what binds a staged Anima request. Publishing a scope whose only
/// honest measured value is 0.0% would put a meaningless choice in front of the selector.
pub const TRANSFORMER_WINDOW_COMPONENTS: &[TransformerComponent] = &[TransformerComponent::Dit];

/// The scope this provider declares as its production selection.
pub const TRANSFORMER_WINDOW_COMPONENT: TransformerComponent = TransformerComponent::Dit;

/// Calibration content fingerprint. It must change whenever quantization floors, tensor layout, or
/// execution structure change in a way that invalidates measurements taken against this provider.
///
/// Load shape is a typed evidence-key axis carried separately on [`MemoryCalibrationIdentity`]; this
/// content fingerprint stays shape-independent.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "anima-mlx-cosmos-shared-ladder-v1";

/// Whether THIS load can execute rung 4.
///
/// Two independent facts decide it, and only one of them is a `LoadShape`:
///
/// 1. The window rebuilds blocks from the checkpoint, so it needs a **re-openable source**. Anima's
///    loader resolves every `WeightsSource` — `Dir(split_files)` or `File(dit)` — to the variant's
///    exact `diffusion_models/anima-{variant}-v1.0.safetensors` on disk, so both forms are
///    re-openable. Unlike Z-Image this family has no in-memory / ComfyUI-normalized build, so there
///    is no source form to exclude. That is asserted, not assumed:
///    `every_load_source_form_resolves_to_a_reopenable_checkpoint`.
/// 2. [`LoadShape::DeferredMaterialization`] says the request wants those re-openable blocks instead
///    of bulk-committing the stack. `OffloadPolicy` is deliberately absent here — phase release is a
///    separate axis, and rung 1's prerequisite is enforced on the SELECTION, not on the load.
pub fn streamable(spec: &LoadSpec) -> bool {
    matches!(spec.load_shape, LoadShape::DeferredMaterialization)
}

fn variant_for(provider_id: &str) -> CoreResult<Variant> {
    Variant::from_id(provider_id).ok_or_else(|| {
        CoreError::Unsupported(format!("unknown Anima memory provider {provider_id}"))
    })
}

/// Build the Anima MLX provider contract for `provider_id` at this [`LoadSpec`].
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let footprint = crate::loader::component_footprint(spec)?;
    contract_with_asset_facts(
        provider_id,
        spec,
        footprint.text_encoder,
        footprint.dit,
        footprint.vae,
    )
}

/// Declaration-equivalent contract used by weights-free registry conformance. Structure, parameter
/// domains and prerequisites are identical; only the measured asset facts are absent.
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
    variant_for(provider_id)?;
    let streamable = streamable(spec);
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
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        MEMORY_CALIBRATION_FINGERPRINT,
        spec.load_shape,
    ));
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
            // Rung 4 makes transformer weight residency a VARIABLE of the peak rather than a
            // constant folded into `AssetBytes`: at window 1 the trunk contributes one block of 28.
            MemoryFormulaVariable::TransformerWindowSize,
        ],
    };
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
        // Request-scoped: the same cached generator serves warm → staged → warm without
        // reconstruction, so the hook is available regardless of the load-time `OffloadPolicy`.
        synchronized_phase_release: true,
        decode_tiling: true,
        attention_chunking: true,
        transformer_window_materialization: streamable,
    };
    // A Resident selection preserves the load-time defaults: Anima's historical warm render tiles
    // nothing and windows nothing, and there is no Sequential-only shipped tiling default to
    // override (unlike SANA).
    contract.resident_request_memory = ResidentRequestMemory::PreserveLoadDefaults;

    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::BoundedTransformerResidency if !streamable => {
                MemoryStrategySupport::Missing
            }
            _ => MemoryStrategySupport::Implemented,
        };
        capability.parameters = match capability.strategy {
            MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                decode_tile_edges: DECODE_TILE_EDGES.to_vec(),
                decode_overlaps: vec![DECODE_OVERLAP],
                ..Default::default()
            },
            MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                attention_chunk_sizes: vec![ATTENTION_CHUNK_SIZE],
                ..Default::default()
            },
            MemoryStrategy::BoundedTransformerResidency if streamable => MemoryParameterRanges {
                transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                transformer_window_components: TRANSFORMER_WINDOW_COMPONENTS.to_vec(),
                ..Default::default()
            },
            _ => MemoryParameterRanges::default(),
        };
    }
    if streamable {
        // Rung 4 bounds the DiT's denoise residency. Without rung 1 engaged, the conditioner and VAE
        // remain resident through the denoise and the REQUEST peak does not move — a phase win that
        // is not a saving. Declaring the edge lets the shared selector refuse that composition
        // instead of every caller re-deriving it from the cost order.
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

/// Validate a decode geometry against the published domain. Narrower than the static contract only
/// in the direction the contract allows — it can reject, never broaden.
fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> CoreResult<()> {
    let edge = edge.ok_or_else(|| {
        CoreError::Unsupported("anima bounded decode requires an explicit tile edge".to_owned())
    })?;
    let overlap = overlap.ok_or_else(|| {
        CoreError::Unsupported("anima bounded decode requires an explicit overlap".to_owned())
    })?;
    if !DECODE_TILE_EDGES.contains(&edge) || overlap != DECODE_OVERLAP {
        return Err(CoreError::Unsupported(format!(
            "anima decode geometry {edge}/{overlap} is outside the calibrated domain \
             {DECODE_TILE_EDGES:?} at overlap {DECODE_OVERLAP}"
        )));
    }
    Ok(())
}

/// Validate a rung-4 transformer window against the published domain — the window-size twin of
/// [`validate_decode`], on the same layers, for the same reason.
///
/// The shared `validate_selection` already refuses an out-of-domain window at admission, so this is
/// defense in depth. It is not redundant: the direction it closes is the one
/// [`TRANSFORMER_WINDOW_SIZES`]' own doc names as the dangerous one — *executing* at a wider cadence
/// than the declared 1 **under**-predicts peak, and a selector that sized a fit off the published
/// domain would then be handed a render that does not fit. Every calibration and real-weight harness
/// reaches `generate` with a hand-built [`GenerationMemory`](mlx_gen::gen_core::GenerationMemory)
/// rather than through a validated `MemorySelection`, which is precisely the path admission does not
/// see.
fn validate_window(size: u32) -> CoreResult<()> {
    if !TRANSFORMER_WINDOW_SIZES.contains(&size) {
        return Err(CoreError::Unsupported(format!(
            "anima transformer window {size} is outside the calibrated domain \
             {TRANSFORMER_WINDOW_SIZES:?}"
        )));
    }
    Ok(())
}

/// The provider safety check every Anima variant shares: the calibration handshake and tier match,
/// then the shared contract's own selection validation, then this provider's route gate, then the
/// budget. Defense in depth only — it can reject, never swap in a different strategy or tier.
pub(crate) fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        // Anima advertises no reference/control conditioning, so text-to-image with zero references
        // is the only route that exists. Admitting anything else would record evidence for a route
        // the generator would reject at `validate` anyway.
        if !matches!(context.mode, MemoryMode::TextToImage) || context.geometry.reference_count != 0
        {
            return Err(CoreError::Unsupported(format!(
                "{}: anima is text-to-image only; got {:?} with {} references",
                contract.provider_id, context.mode, context.geometry.reference_count
            )));
        }
        // SC-15839 review defect: a Resident+PiD admission. Anima has no PiD student at all — the
        // heavy loader ignores `use_pid` — so a PiD selection would silently execute the ordinary
        // native decode, i.e. a different strategy than the selector chose. Reject at admission
        // rather than degrade.
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{}: anima has no PiD decoder; the overlay is not implementable on this provider",
                contract.provider_id
            )));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            validate_decode(
                context.selection.parameters.decode_tile_edge,
                context.selection.parameters.decode_overlap,
            )?;
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
            // The scope AND the size are checked here as well as by the shared parameter validator,
            // so a request that reached the provider by another path still cannot ask for a scope
            // this family does not implement or a cadence it never measured.
            let component = context.selection.parameters.window_component();
            if !TRANSFORMER_WINDOW_COMPONENTS.contains(&component) {
                return Err(CoreError::Unsupported(format!(
                    "{}: transformer window component {component:?} is not implemented",
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
/// Anima is convert-at-install: the SceneWorks worker packs the DiT on-device and the loader
/// packed-detects the tier off the on-disk `{base}.scales`, so `LoadSpec::quantize` is the recorded
/// tier for the resolved directory rather than a load-time transform request. `precision` is
/// pinned to bf16 by `load_variant`.
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
            // Rung 1 is request-scoped, and every optimized selection engages it by cost order —
            // which is also rung 4's declared prerequisite.
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
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        DitConfig::anima().num_layers,
        move |use_pid, edge, overlap| {
            if use_pid {
                return Err(CoreError::Unsupported(format!(
                    "{provider_id}: anima has no PiD decoder"
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

// ── Request-side resolution: the shared `GenerationMemory` signal → this provider's levers ────────

/// Rung 3: the bounded-attention plan for this request. Cancellation is threaded so a per-chunk
/// `eval` observes a cancel promptly rather than at the next step boundary.
pub(crate) fn attention_plan(req: &GenerationRequest) -> mlx_gen::attention::AttentionPlan<'_> {
    match req.memory {
        Some(memory) if memory.chunk_attention => mlx_gen::attention::AttentionPlan::budgeted(
            mlx_gen::attention::AttentionBudget::CONSTRAINED,
        )
        .with_cancel(&req.cancel),
        _ => mlx_gen::attention::AttentionPlan::UNBOUNDED,
    }
}

/// Rung 4: the requested window size, or `None` for the resident stack. A scope this family does not
/// implement — or a cadence outside the measured [`TRANSFORMER_WINDOW_SIZES`] — is a typed rejection
/// rather than a silently narrowed (or silently *widened*) execution.
pub(crate) fn transformer_window_size(req: &GenerationRequest) -> mlx_gen::Result<Option<usize>> {
    let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
        return Ok(None);
    };
    let component = memory
        .transformer_window_component
        .unwrap_or(TRANSFORMER_WINDOW_COMPONENT);
    if !TRANSFORMER_WINDOW_COMPONENTS.contains(&component) {
        return Err(mlx_gen::Error::Unsupported(format!(
            "anima implements only the {TRANSFORMER_WINDOW_COMPONENT:?} transformer window \
             component, got {component:?}"
        )));
    }
    let size = memory
        .transformer_window_size
        .unwrap_or(TRANSFORMER_WINDOW_SIZE);
    validate_window(size).map_err(|error| {
        mlx_gen::Error::Unsupported(format!("anima transformer window rejected: {error}"))
    })?;
    Ok(Some(size as usize))
}

/// Rung 2: the decode tiling for this request, or `None` for the single-pass decode.
pub(crate) fn decode_tiling(req: &GenerationRequest) -> mlx_gen::Result<Option<TilingConfig>> {
    let Some(memory) = req.memory.filter(|memory| memory.tile_vae_decode) else {
        return Ok(None);
    };
    let edge = memory.decode_tile_edge.unwrap_or(DECODE_TILE_EDGE);
    let overlap = memory.decode_overlap.unwrap_or(DECODE_OVERLAP);
    validate_decode(Some(edge), Some(overlap)).map_err(|error| {
        mlx_gen::Error::Unsupported(format!("anima decode tiling rejected: {error}"))
    })?;
    Ok(Some(TilingConfig::spatial_only(
        edge as i32,
        overlap as i32,
    )))
}

/// Rung 1: whether this request stages its component residency.
pub(crate) fn stage_residency(req: &GenerationRequest, default_staged: bool) -> bool {
    req.memory
        .map_or(default_staged, |memory| memory.stage_residency)
}

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

    const IDS: [&str; 3] = ["anima_base", "anima_aesthetic", "anima_turbo"];

    fn spec(shape: LoadShape) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent/anima-contract".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(shape)
    }

    #[test]
    fn every_catalog_entry_publishes_the_same_full_ladder_on_a_deferred_load() {
        let spec = spec(LoadShape::DeferredMaterialization);
        let contracts: Vec<_> = IDS
            .iter()
            .map(|id| weights_free_memory_strategy_contract(id, &spec).unwrap())
            .collect();
        for (id, contract) in IDS.iter().zip(&contracts) {
            assert!(contract.conformance_errors().is_empty(), "{id}");
            for strategy in MemoryStrategy::ALL {
                assert_eq!(
                    contract.capability(strategy).unwrap().support,
                    MemoryStrategySupport::Implemented,
                    "{id} must implement {strategy:?}"
                );
            }
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .parameters
                    .transformer_window_components,
                vec![TransformerComponent::Dit]
            );
            assert_eq!(
                contract.calibration.as_ref().unwrap().fingerprint,
                MEMORY_CALIBRATION_FINGERPRINT
            );
            // No PiD route is published at all: the overlay is not implementable here.
            assert!(contract.pid_decode_routes.is_none(), "{id}");
        }
        // Base/Aesthetic/Turbo share the implementation, so their DECLARATIONS must be identical.
        // What they must NOT share is evidence — that is each catalog story's own obligation, and
        // it lives in the generated matrix, not in this contract.
        for strategy in MemoryStrategy::ALL {
            let first = contracts[0].capability(strategy);
            for contract in &contracts[1..] {
                assert_eq!(first, contract.capability(strategy));
            }
        }
    }

    /// Rung 4 is a per-LOAD declaration, never a per-provider one.
    #[test]
    fn an_eager_load_declares_rung_four_missing_and_keeps_the_rest() {
        let spec = spec(LoadShape::EagerMaterialization);
        for id in IDS {
            let contract = weights_free_memory_strategy_contract(id, &spec).unwrap();
            assert!(contract.conformance_errors().is_empty(), "{id}");
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing,
                "{id}: an eager load cannot window"
            );
            assert!(!contract.lifecycle.transformer_window_materialization);
            for strategy in [
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
            ] {
                assert_eq!(
                    contract.capability(strategy).unwrap().support,
                    MemoryStrategySupport::Implemented,
                    "{id}: {strategy:?} does not depend on the load shape"
                );
            }
            // The provider's OWN rung-1 edge is not declared for a rung it does not implement —
            // only the shared load-shape prerequisite every contract carries remains.
            let edges: Vec<_> = contract
                .requires(MemoryStrategy::BoundedTransformerResidency)
                .collect();
            assert_eq!(
                edges,
                vec![MemoryStrategyPrerequisite::LoadShape(
                    LoadShape::DeferredMaterialization
                )],
                "{id}"
            );
        }
    }

    /// Rung 4 requires rung 1 ENGAGED in the same request — availability is not engagement.
    #[test]
    fn the_transformer_window_declares_staged_residency_as_a_same_request_prerequisite() {
        let contract = weights_free_memory_strategy_contract(
            "anima_base",
            &spec(LoadShape::DeferredMaterialization),
        )
        .unwrap();
        let edges: Vec<_> = contract
            .requires(MemoryStrategy::BoundedTransformerResidency)
            .collect();
        assert!(edges.iter().any(|prerequisite| matches!(
            prerequisite,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            }
        )));
    }

    /// The decode domain and its rejection set are pinned, and the rejection is asserted rather than
    /// described. A widened ladder that quietly swallowed a rejected edge would fail here.
    #[test]
    fn the_decode_domain_and_its_rejection_set_are_pinned() {
        assert_eq!(DECODE_TILE_EDGES, &[768, 640, 512, 384, 320, 256, 192]);
        assert_eq!(DECODE_TILE_EDGES_REJECTED, &[160, 128, 96]);
        assert_eq!(DECODE_TILE_EDGE, 640);
        assert_eq!(DECODE_OVERLAP, 64);
        // The two sets partition the swept candidates: a rejected edge that quietly reappeared in
        // the admitted ladder (or vice versa) would make the rejection evidence unattributable.
        for edge in DECODE_TILE_EDGES_REJECTED {
            assert!(!DECODE_TILE_EDGES.contains(edge));
        }
        // Every admitted edge is strictly larger than every rejected one — the sweep found a single
        // knee, not an interleaved set, and the contract must not publish one that says otherwise.
        assert!(
            DECODE_TILE_EDGES.iter().min() > DECODE_TILE_EDGES_REJECTED.iter().max(),
            "the admitted ladder must sit entirely above the rejected knee"
        );
        for edge in DECODE_TILE_EDGES {
            validate_decode(Some(*edge), Some(DECODE_OVERLAP)).unwrap();
        }
        for edge in DECODE_TILE_EDGES_REJECTED {
            assert!(
                validate_decode(Some(*edge), Some(DECODE_OVERLAP)).is_err(),
                "measured-and-rejected edge {edge} must stay outside the admitted domain"
            );
            assert!(!DECODE_TILE_EDGES.contains(edge));
        }
        // The overlap is a single point: moving it off the calibrated value is rejected too.
        assert!(validate_decode(Some(DECODE_TILE_EDGE), Some(32)).is_err());
    }

    /// A request that names an out-of-domain geometry must be a typed rejection, not a silent
    /// re-plan onto the default — the selector chose a strategy and must get that strategy.
    #[test]
    fn an_out_of_domain_request_geometry_is_rejected_rather_than_silently_replanned() {
        let request = |memory| GenerationRequest {
            prompt: "x".into(),
            memory: Some(memory),
            ..Default::default()
        };
        let rejected = request(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(DECODE_TILE_EDGES_REJECTED[0]),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        });
        assert!(decode_tiling(&rejected).is_err());

        let admitted = request(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(DECODE_TILE_EDGE),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        });
        assert!(decode_tiling(&admitted).unwrap().is_some());

        // An untouched request is byte-for-byte unaffected: no tiling, no chunking, no window.
        let untouched = GenerationRequest {
            prompt: "x".into(),
            ..Default::default()
        };
        assert!(decode_tiling(&untouched).unwrap().is_none());
        assert!(transformer_window_size(&untouched).unwrap().is_none());
        assert!(!stage_residency(&untouched, false));
    }

    /// An unimplemented window scope is rejected, not narrowed to `Dit`. Narrowing would execute a
    /// different strategy than the selector chose while reporting success.
    #[test]
    fn an_unimplemented_window_component_is_rejected() {
        for component in [
            TransformerComponent::TextEncoder,
            TransformerComponent::Both,
        ] {
            let request = GenerationRequest {
                prompt: "x".into(),
                memory: Some(GenerationMemory {
                    stream_transformer_blocks: true,
                    transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
                    transformer_window_component: Some(component),
                    ..Default::default()
                }),
                ..Default::default()
            };
            assert!(
                transformer_window_size(&request).is_err(),
                "{component:?} must be rejected"
            );
        }
        let dit = GenerationRequest {
            prompt: "x".into(),
            memory: Some(GenerationMemory {
                stream_transformer_blocks: true,
                transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
                transformer_window_component: Some(TransformerComponent::Dit),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            transformer_window_size(&dit).unwrap(),
            Some(TRANSFORMER_WINDOW_SIZE as usize)
        );
    }

    /// **Rung 3's only unit-level gate.** Without it the rung is unfalsifiable in this crate:
    /// replacing this function's body with [`AttentionPlan::UNBOUNDED`] silently turns bounded
    /// attention off, and every other assertion in the crate — including the real-weight ladder's
    /// image-equality checks and its `full.denoise < chunked.denoise` — passes *more easily* with the
    /// rung disabled. So assert the two halves that actually distinguish them: the selected request
    /// gets the CONSTRAINED budget **and** a cancel (the per-chunk `eval` is what makes a cancel
    /// observable inside a step), and the unselected one gets *exactly* `UNBOUNDED` with no cancel
    /// attached at all.
    ///
    /// Request-local is the other half: rung 3 is a per-request choice on a cached generator, so a
    /// plain follow-up request must come back unbounded rather than inherit the previous plan.
    /// (Mirrors `mlx_gen_flux`'s `attention_plan_is_request_local_and_unselected_is_exactly_unbounded`.)
    #[test]
    fn attention_plan_is_request_local_and_unselected_is_exactly_unbounded() {
        use mlx_gen::attention::AttentionBudget;

        let plain = GenerationRequest {
            prompt: "x".into(),
            ..Default::default()
        };
        let plan = attention_plan(&plain);
        assert_eq!(
            plan.budget,
            AttentionBudget::UNBOUNDED,
            "a request that did not select rung 3 must run the byte-identical unbounded forward"
        );
        assert!(
            plan.cancel.is_none(),
            "the unbounded fast path has no chunk boundary, so it must not carry a cancel"
        );

        let selected = GenerationRequest {
            prompt: "x".into(),
            memory: Some(GenerationMemory {
                chunk_attention: true,
                attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                ..Default::default()
            }),
            ..Default::default()
        };
        let plan = attention_plan(&selected);
        assert_eq!(
            plan.budget,
            AttentionBudget::CONSTRAINED,
            "rung 3 must plan the single published {ATTENTION_CHUNK_SIZE}-element budget"
        );
        assert!(
            plan.cancel.is_some(),
            "the chunked plan must carry the request's cancel so a per-chunk eval observes it"
        );

        // The published domain and the planned budget are the SAME value — a contract that
        // advertised one budget while the pipeline ran another would make rung-3 evidence
        // unattributable.
        assert_eq!(
            ATTENTION_CHUNK_SIZE,
            mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32
        );

        // Request-local: the next plain request on the same generator is unbounded again.
        let follow_up = GenerationRequest {
            prompt: "x".into(),
            ..Default::default()
        };
        assert_eq!(
            attention_plan(&follow_up).budget,
            AttentionBudget::UNBOUNDED
        );
        assert!(attention_plan(&follow_up).cancel.is_none());

        // Every other rung selected, rung 3 NOT selected, is still exactly unbounded: the flag is
        // the signal, not "this request carries a memory block".
        let other_rungs = GenerationRequest {
            prompt: "x".into(),
            memory: Some(GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                stream_transformer_blocks: true,
                decode_tile_edge: Some(DECODE_TILE_EDGE),
                decode_overlap: Some(DECODE_OVERLAP),
                transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            attention_plan(&other_rungs).budget,
            AttentionBudget::UNBOUNDED
        );
        assert!(attention_plan(&other_rungs).cancel.is_none());
    }

    /// The window domain is enforced, not merely published. Without this the sweep's own finding —
    /// that 2/4/14/28 bound nothing the per-block materialize/drop does not already bound — would be
    /// a comment beside a parameter the production path executes anyway, and executing at a wider
    /// cadence while declaring 1 UNDER-predicts peak.
    #[test]
    fn an_out_of_domain_transformer_window_is_rejected() {
        let request = |size: Option<u32>| GenerationRequest {
            prompt: "x".into(),
            memory: Some(GenerationMemory {
                stream_transformer_blocks: true,
                transformer_window_size: size,
                ..Default::default()
            }),
            ..Default::default()
        };
        // Every published size is admitted...
        for size in TRANSFORMER_WINDOW_SIZES {
            assert_eq!(
                transformer_window_size(&request(Some(*size))).unwrap(),
                Some(*size as usize),
                "the published window {size} must be admitted"
            );
        }
        // ...and the sweep's own unpublished cadences are refused, including the all-covering
        // control that bounds nothing and the 0 that is not a window at all.
        for size in [
            0_u32,
            2,
            4,
            14,
            28,
            DitConfig::anima().num_layers as u32 + 1,
        ] {
            let error = transformer_window_size(&request(Some(size)))
                .expect_err("an unpublished window must be refused");
            assert!(
                error.to_string().contains("calibrated domain"),
                "expected the window-domain rejection for {size}, got: {error}"
            );
        }
        // Omitting the size still resolves to the published default rather than rejecting: `None`
        // means "the provider's own constant", which is in-domain by construction.
        assert_eq!(
            transformer_window_size(&request(None)).unwrap(),
            Some(TRANSFORMER_WINDOW_SIZE as usize)
        );
        assert!(TRANSFORMER_WINDOW_SIZES.contains(&TRANSFORMER_WINDOW_SIZE));
        // Every published window must bound something: a window covering the whole stack walks the
        // identical driver and bounds nothing, so publishing one would record a zero saving.
        for size in TRANSFORMER_WINDOW_SIZES {
            assert!(
                *size >= 1 && (*size as usize) < DitConfig::anima().num_layers,
                "window {size} must bound something"
            );
        }
    }

    #[test]
    fn an_unknown_provider_id_is_a_typed_error() {
        assert!(weights_free_memory_strategy_contract(
            "anima_nonexistent",
            &spec(LoadShape::EagerMaterialization)
        )
        .is_err());
    }
}
