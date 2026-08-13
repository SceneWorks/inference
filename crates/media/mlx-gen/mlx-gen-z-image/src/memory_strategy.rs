//! Z-Image MLX adoption of the shared memory-strategy contract (SC-15449) — the **complete** ladder:
//! rung 3 (bounded attention, SC-15615), rung 4 (bounded transformer residency, SC-15754), and the
//! widened rung-2 decode domain plus the PiD reconciliation (SC-15510).
//!
//! All four registered Z-Image variants — `z_image_turbo`, `z_image`, `z_image_turbo_control`,
//! `z_image_control` — share one DiT, one staged [`Residency`](mlx_gen::Residency) seam, one tiled VAE
//! decode, one bounded-attention primitive and one block-residency stream, so they share one contract
//! builder. Only the provider id and the advertised request surface differ.
//!
//! **`z_image_edit` is a catalog id, not a provider.** It is served by `z_image_turbo` (edit mode over
//! Turbo weights), so it inherits this contract rather than declaring one — there is no fifth provider
//! here and no alias mechanism in this crate. The resolution lives on the SceneWorks side, in two
//! places, and both are pinned there rather than asserted here:
//! `jobs_store::routing::mlx` maps `z_image_turbo | z_image_edit` to the same eligibility and engine,
//! and the matrix generator's `backendScopes` inherits the entry's advertised backends from
//! `z_image_turbo` (`z_image_edit inherits its backend scopes from the z_image_turbo provider`).
//! Consequently `MemoryMode::Edit` must be admissible on this contract — see
//! `the_edit_surface_z_image_edit_resolves_to_is_admissible`.
//!
//! ## Declared rungs
//!
//! | Rung | Support | Executable seam |
//! |---|---|---|
//! | 0 Resident | Implemented | Request-scoped `Residency` warm pair — encoder + DiT + VAE held across requests |
//! | 1 Staged residency | Implemented (request-scoped) | `GenerationMemory::stage_residency` drives encode → drop encoder → denoise → **drop DiT** → decode |
//! | 2 Bounded decode | Implemented | `Vae::decode_tiled` over the [`DECODE_TILE_EDGES`] ladder, or the PiD student over [`pid_decode_tile_edges`] (`pipeline::decode_tiling`) |
//! | 3 Bounded attention | Implemented | [`mlx_gen::attention::sdpa_budgeted_bhsd`] threaded through every DiT attention (SC-15615) |
//! | 4 Bounded transformer residency | Implemented (snapshot loads) | [`mlx_gen::block_residency::run_windowed`] over the 30 unified blocks, and the 15 ControlNet blocks (SC-15754) |
//!
//! Rung 4 is available only for a snapshot load that explicitly requests
//! [`LoadShape::DeferredMaterialization`](mlx_gen::LoadShape). The two facts are independent:
//! `WeightsSource::Dir` supplies a re-openable source, while the load shape says not to bulk-commit
//! the resident transformer stacks. Phase-level release remains a separate request-scoped axis.
//!
//! **This is the MLX column, and it is not the Candle column.** As of SC-15754 the MLX lane carries
//! all five rungs; `candle-gen-z-image` carries 0-3 and has no rung 4. Neither the presence nor the
//! *magnitude* of a rung transfers between the two backends — rung 3 exists on both and is worth far
//! more on Candle (see below), and rung 4 exists only here. Every per-backend saving in this file was
//! measured on this backend.
//!
//! **Compare rung-3 magnitudes like-for-like.** An earlier revision of this note said rung 3 is worth
//! "~6×" more on Candle. That was a scope error corrected in SC-15793: Candle's −32% is a **denoise-
//! phase** delta while MLX's −5.0% is a **whole-request** one. On the denoise phase MLX measures
//! **−1.7%**, so like-for-like the gap is −32% vs −1.7% — roughly an order of magnitude, not ~6×
//! (and the two figures are published in different units, GB vs GiB). This file is calibration-facing,
//! so the scope of every quoted number matters: see `gen_core::attention_budget` for the full table.
//!
//! **Rung 1 is request-scoped (SC-15806).** `z_image_generation_memory` sets
//! [`mlx_gen::gen_core::GenerationMemory::stage_residency`] only when the contract says rung 1 is
//! engaged. The same
//! cached generator can therefore serve warm → staged → warm without reconstruction. Z-Image ignores
//! the legacy load-time [`OffloadPolicy`](mlx_gen::OffloadPolicy); the shared contract selection is
//! the authority.
//!
//! **Rung 4 does not depend on rung 1.** A window needs a deferred-materialization load, which can be
//! composed with either `Resident` or `Sequential` phase residency. An eager load rejects the window
//! because it would add a second block copy on top of the materialized trunk.
//!
//! ## What each rung is worth here (measured, Apple M5 Max, real `z_image_turbo`, 1024², 4 steps,
//! staged residency, count 1 — `tests/block_residency_real_weights.rs`)
//!
//! Staged **denoise** peak, every hosted tier:
//!
//! | tier | rungs 1-3 (the SC-15615 top) | + rung 4 | cut | request peak | bound by |
//! |---|---:|---:|---:|---:|---|
//! | q4 | 4.653 | **1.795** | **−61%** | 4.363 GiB | decode |
//! | q8 | 7.486 | **2.149** | **−71%** | 5.087 GiB | conditioning |
//! | bf16 | 13.362 | **3.201** | **−76%** | 8.489 GiB | decode |
//!
//! The saving scales with the tier's weight size — which is what a *weights*-bounding rung should do,
//! and exactly what distinguishes it from rung 3, whose saving is activation-side and
//! tier-independent (0.245 GiB on q4, an order of magnitude smaller).
//!
//! **On a memory-constrained host** ([`mlx_gen::memory::apply_memory_cap_env`]) the bound holds and
//! re-materialization does **not** get materially more expensive, which is the question SC-15754 asked
//! before choosing a production window: q4 under a 4 GiB MLX cap costs 2.918 s/step resident against
//! 3.191 windowed (+9%, against +4-8% unconstrained), and q8 under a 6 GiB cap lands at a 4.768 GiB
//! request peak.
//!
//! **Both q4 and q8 fit an 8 GiB budget** on both host classes (4.363 / 5.087 GiB unconstrained,
//! 4.363 / 4.768 constrained, against 6.0 GiB usable after the generic gate's 2 GiB reserve). q8 was
//! SC-15615's open question — it concluded only rung 4 could move it, and predicted the binding phase
//! would become the Qwen text encoder. Both hold. **Disclosure:** measured on a 128 GB machine with
//! MLX's ACTIVE-bytes counter, not on a physical 8 GB Mac whose unified pool is shared with
//! macOS/WindowServer/SceneWorks — what decides that render is SC-15611's admission arithmetic and
//! SC-15614's reserve.
//!
//! **Rung 4 cuts the denoise phase by 61%** — 2.86 GiB, an order of magnitude more than rung 3's
//! measured 0.245 GiB on this lane — and in doing so it **moves the binding phase off the denoise**.
//! The q4 request peak is now the *decode* (4.363 GiB), which is why SC-15510's other half, the
//! widened [`DECODE_TILE_EDGES`] ladder, is the lever that matters next. Rung 3's small saving is not
//! a defect: MLX's fused SDPA never materializes the `[B,H,Sq,Sk]` score tensor that Candle's
//! `attention_basic` does, so the same knob buys a lazy-graph cut here and a bounded score matrix
//! there (SC-15615 pins the mechanism with an inert never-chunks control).
//!
//! **The window size is inert here, and that is a measurement.** Windows 15, 8, 4 and 2 all land on
//! 1.832 GiB, and so does a window of **30** — one all-covering window that bounds nothing. The
//! family's block loop materializes one block, runs it and drops it before advancing, and that drop is
//! a real release because rung 3's per-chunk `eval` already evaluated the carried activation inside
//! the block's own attention. So residency is per-BLOCK by construction, and
//! [`TRANSFORMER_WINDOW_SIZES`] publishes the single exact value instead of four rows that differ only
//! by noise.
//!
//! ## Route coverage
//!
//! Rungs 3 and 4 reach **every** advertised denoise route — plain t2i, base CFG (both the cond and the
//! uncond forward), turbo control and base control (including the ControlNet branch's own 15 blocks),
//! and the PiD route, whose denoise is the ordinary DiT.
//!
//! **The PiD route is now a first-class rung-2+ route** (SC-15510). SC-15615 had to refuse rungs 2 and
//! above with the overlay, because the super-resolving student planned its own tile edge/overlap from
//! `mlx_gen_pid::budget` and never read the contract's parameters — admitting a selection would have
//! executed a different strategy than the selector chose, and the cumulative ladder dragged rung 3 down
//! with it. `mint_planned_decoder_with_tiling` now honours an explicit plan and validates it against
//! the planner's own invariants, so the overlay has a real, published candidate domain.
//!
//! That domain is **not the native one**. Both are "decode tile edge in output pixels", but the native
//! VAE tiles a 1024²-class output at 512-768 px while the student tiles a `scale×` super-resolved
//! output at its own, much larger edge (2048 px — see [`pid_decode_tile_edges`]), so the two sets are
//! disjoint. The static contract publishes their union —
//! it is the provider's whole production domain, and `validate_selection` is not route-aware — while
//! `safety_check` and the request scope, which both see `use_pid`, enforce the route's own subset. A
//! selection built for one route is rejected on the other rather than silently re-planned.
//!
//! Both the split and its disjointness are the shared [`mlx_gen_pid::DecodeRoutes`]' (SC-15775), not
//! this file's: this module's `decode_routes()` declares only the native ladder, the PiD half comes
//! from the student, and
//! [`DecodeRoutes::new`](mlx_gen_pid::DecodeRoutes::new) **refuses to construct** a native ladder that
//! reaches into the PiD domain, so this file cannot publish one. The
//! obligation used to be a doc comment on `mlx_gen_pid::engine::selected_decode_tiling` telling the
//! next adopter to copy this file's shape; it is now an API that the next adopter cannot get wrong
//! quietly.
//!
//! ## Ownership
//!
//! This file declares *structure and parameter domains only*. Measured coefficients, envelopes, and
//! per-tier peaks live in SceneWorks generated evidence keyed by
//! [`MEMORY_CALIBRATION_FINGERPRINT`]; the worker owns live-budget accounting and least-cost
//! selection. The scope below is defense in depth: it can reject a selection, never substitute one.

use mlx_gen::asset_facts::{
    projected_safetensors_bytes, projected_safetensors_tensors, projected_tensor_headers_bytes,
    ResidentProjection,
};
use mlx_gen::gen_core::{
    Error as CoreError, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryComponentKind, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent, MemoryRunContext,
    MemoryRuntimeSemantics, MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability,
    MemoryStrategyParameters, MemoryStrategySupport, Result as CoreResult, TransformerComponent,
};
#[cfg(test)]
use mlx_gen::gen_core::{GenerationMemory, MemoryGeometry, MemoryRunOutcome};
#[cfg(test)]
use mlx_gen::GenerationRequest;
use mlx_gen::{Quant, WeightsSource};

/// The **default** decode tile edge for the native VAE — the 512 px parity sweet spot for this
/// GroupNorm VAE (sc-13571). A request that names no geometry decodes here, which is what keeps every
/// pre-SC-15510 render byte-identical.
pub const DECODE_TILE_EDGE: u32 = 512;
/// The decode overlap paired with [`DECODE_TILE_EDGE`], and the only native overlap advertised: the
/// tile ladder trades peak against seam risk on the edge, and moving both axes at once would make a
/// calibration row un-attributable.
pub const DECODE_OVERLAP: u32 = 64;

/// The native VAE's production tile-edge ladder (SC-15510), output pixels, at [`DECODE_OVERLAP`].
///
/// Before this the contract advertised the single hardcoded 512, which is accurate but collides with
/// SC-15508's *"a single-point pass cannot mark untested production parameters Verified"*.
///
/// **The candidate set was measured, not inherited.** SC-15510's own note proposed adopting the
/// current-pin-verified `768/640/512/448/384/320/256` ladder on the grounds that both families drive
/// the same `mlx_gen::vae_tiling` machinery. That ladder is SceneWorks'
/// `memory-mlx-adapter` probe sweep, and it was measured on the **Qwen** VAE — a different
/// decoder. Sharing the tiling machinery does not transfer a candidate set any more than sharing a
/// rung's name transfers its magnitude between backends; what transfers is the mechanism, and the
/// numbers have to be re-measured on the decoder that will run them. Swept against the **exact
/// untiled decode** on real q4 weights at 1024² (`tests/block_residency_real_weights.rs`):
///
/// | tile | decode peak | max Δ vs exact | mean Δ |
/// |---:|---:|---:|---:|
/// | untiled | 19.422 GiB | — | — |
/// | 768 | 8.089 | 48 | 2.82 |
/// | 640 | 5.795 | 41 | 2.42 |
/// | **512** (default) | **4.363** | **46** | **2.09** |
/// | 448 | 4.645 | 64 | 2.90 |
/// | 384 | 3.896 | 72 | 3.01 |
/// | 320 | 4.051 | 74 | 3.05 |
/// | 256 | 3.157 | 83 | 3.34 |
///
/// Two things fall out, and both are why the ladder stops at 512:
///
/// 1. **Below 512 the image measurably degrades** — max Δ jumps 46 → 64 → 83 and mean Δ rises
///    monotonically. That is the sc-13571 comment ("smaller tiles would drift the per-tile norm
///    statistics") confirmed by measurement on this GroupNorm VAE. `GenerationMemory`'s levers are
///    documented as *quality-preserving*; a tile that visibly seams is a quality trade, not a rung.
/// 2. **And it buys nothing at the request level.** Every sub-512 tile leaves the request peak at
///    4.898 GiB, because the denoise phase binds there — and the decode peak is not even monotone
///    (4.645 at 448 is *worse* than 4.363 at 512). Paying image quality for no admission win is a
///    strictly bad trade.
///
/// The separation is clean rather than marginal: the admitted set tops out at 48/255 and the rejected
/// set starts at 64/255, a 33% gap, which is what the sweep's bound is set from.
///
/// 640 and 768 cost *more* decode memory than the default and are published anyway, because the
/// ladder is a **domain**, not a recommendation: at a larger output a 512 px tile is many more
/// forwards, and the selector — not this file — owns the peak-vs-latency choice. Selection is the
/// worker's; this is the set it may choose from.
///
/// Why widening mattered at all: decode is **tier-independent** — measured 4.363 GiB (q4) and
/// 4.364 GiB (q8) at 1024², because after `shed_dit` only the ~150 MB VAE remains and the rest is the
/// tiled-decode transient — and after rung 4 lands it becomes the *binding* phase on q4.
pub const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512];

/// Tile edges swept and **rejected** by measurement (see [`DECODE_TILE_EDGES`]), kept so the sweep can
/// re-assert the exclusion rather than leaving it as a comment that drifts.
///
/// A future VAE change that made these seam-free would show up as this list failing its rejection
/// check — which is the point. Silently dropping them would leave nothing to notice that with.
pub const DECODE_TILE_EDGES_REJECTED: &[u32] = &[448, 384, 320, 256];

/// The **PiD overlay route's** decode tile-edge ladder (SC-15510), output pixels.
///
/// Same parameter, same unit, a different decoder: `use_pid` replaces the native VAE with a
/// super-resolving student that decodes in high-resolution *pixel* space at `scale×` the request, so
/// its legal tiles live in an entirely different range.
///
/// SC-15615 had to *refuse* rung 2 (and cumulatively rung 3) on the PiD route because the student
/// planned its own tiling and never read the contract's parameters — admitting a selection would have
/// executed a different strategy than the selector chose. SC-15510 reconciles them instead: the
/// student now accepts an explicit plan, and this is the value it accepts.
///
/// **The value is not this provider's to choose** (SC-15775). The student is shared across every
/// PiD-eligible provider and tied to a *latent space* rather than to a model, so the domain — and the
/// seam evidence behind its single candidate — lives with the student in
/// [`DecodeRoutes::pid_edges`](mlx_gen_pid::DecodeRoutes::pid_edges). This function only re-exports it
/// under the name the rest of this module reads. A provider that could *supply* this ladder could
/// supply its native one by mistake, which is the whole defect the shared type exists to make
/// unrepresentable.
pub fn pid_decode_tile_edges() -> Vec<u32> {
    mlx_gen_pid::DecodeRoutes::pid_edges()
}

/// The PiD route's feather overlap — the sc-10087 A/B's 256 px, which is `mlx_gen_pid`'s own default.
/// Like the ladder above, owned by the student rather than by this provider.
pub const PID_DECODE_OVERLAP: u32 = mlx_gen_pid::budget::DEFAULT_TILE_OVERLAP as u32;

/// The production transformer-window domain for ladder rung 4 (SC-15754): how many of the 30 unified
/// blocks are held materialized at once.
///
/// **It is one value, and that is a measurement, not a placeholder.**
///
/// The obvious declaration is a `1/2/4/8/15` ladder — that is the shape SC-15750's own sweep has, and
/// it is what this contract first published. An attribution control killed it. Swept on real q4
/// weights at 1024² (`tests/block_residency_real_weights.rs`), the staged denoise peak is:
///
/// | window | denoise peak | s/step |
/// |---:|---:|---:|
/// | resident (no rung 4) | 4.653 GiB | 3.074 |
/// | 15 | 1.832 | 3.122 |
/// | 8 | 1.832 | 3.067 |
/// | 4 | 1.832 | 3.082 |
/// | 2 | 1.832 | 3.189 |
/// | 1 | 1.795 | ~3.3 |
/// | **30 — one all-covering window, bounds nothing** | **1.832** | 3.613 |
///
/// The control is the load-bearing row: a window of 30 walks the identical driver and bounds
/// *nothing*, and it lands on the same 1.832 GiB. So the window size does not move the peak at any
/// setting — which means the peak is not set by the window at all.
///
/// The reason is structural, and it is in this family's block loop rather than in the primitive:
/// [`ZImageTransformer::run_windowed_layers_with_hints`](crate::ZImageTransformer) materializes ONE
/// block, runs it, and drops it before advancing, so at most one block is ever held. The drop is a
/// real release because rung 3's per-chunk `eval` has already forced the carried activation to
/// evaluate inside that block's own attention, cutting the lazy graph that would otherwise keep the
/// block's weights alive. Rung 4 is cumulative over rung 3, so that is true on every request that can
/// select it.
///
/// A window is an **upper bound** on residency, so holding fewer blocks conforms — but the number the
/// contract publishes is what a caller uses to *predict peak*, and declaring 8 while holding 1 would
/// over-predict and make the selector reject fits it could have taken. Declaring `[1]` is the exact,
/// measured truth.
///
/// Advertising the four indistinguishable values would also have been the epic's own failure mode one
/// level down: a calibration sweep would have recorded four rows differing only by noise, and a
/// selector would have "chosen" between them meaninglessly.
///
/// The sweep did find a real ~5% *latency* effect — batching several blocks per snapshot re-open is
/// cheaper than re-opening per block (2.967 s/step at 4, 3.111 at 1). **It is deliberately not taken.**
/// Executing at a wider cadence while declaring 1 would raise the true peak to 1.832 GiB against a
/// declared 1.795, i.e. the contract would **under-predict by 37 MiB** — the one direction that
/// matters, because the caller uses the declared window to decide whether a request fits. Five percent
/// of the denoise phase on the rung reached only under memory pressure is the right thing to spend to
/// keep the declaration exact.
///
/// **Attribution (same sweep, rung 3 disabled — not a selectable configuration, the ladder is
/// cumulative):** resident 4.898 GiB, window 1 → 2.072, never-bounds control → 2.247. So the per-block
/// materialize/drop is the mechanism in both cases; rung 3's per-chunk `eval` tightens it further
/// (1.795 vs 2.072), and the window itself is worth ≤175 MiB of a ~2.8 GiB saving even where it is not
/// fully masked. One candidate is the honest domain either way.
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1];

/// The transformer window rung 4 executes at — the single published candidate. See
/// [`TRANSFORMER_WINDOW_SIZES`] for why the domain has one element.
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;

/// The rung-4 **component scopes** this provider implements (SC-15794).
///
/// All three are real: the DiT stream (SC-15754), the text-encoder stream, and both together. Each is
/// measured on real weights — `text_encoder_window_real_weights` owns the encoder numbers and
/// `block_residency_real_weights` the DiT ones.
pub const TRANSFORMER_WINDOW_COMPONENTS: &[TransformerComponent] = &[
    TransformerComponent::Dit,
    TransformerComponent::TextEncoder,
    TransformerComponent::Both,
];

/// The scope this provider declares as its **production selection**.
///
/// `Dit`, not `Both`, and that is a measured decision rather than caution.
///
/// SC-15998 retired the old `0.0%` request-saving conclusion: that measurement ran inside a
/// Sequential load whose decode phase bound the composed request. With phase residency held
/// SC-15998 measured the scopes against the missing like-for-like control. At q4 512²/1 step,
/// Resident+Deferred without rung 4 and each of `Dit`, `TextEncoder`, and `Both` all reached the same
/// 4.847 GiB request peak: 0.0% incremental request saving. The windows did move their targeted
/// phase peaks, but decode bound this envelope, so no scope may be recorded as a request saving.
///
/// `Dit` remains the production default only as the narrowest historical scope, not because this
/// acceptance probe showed a request-peak advantage. Selection still requires production-envelope
/// evidence keyed to the exact load shape.
///
/// [`TRANSFORMER_WINDOW_COMPONENTS`] still publishes all three, because the capability is real and a
/// caller may select it; this is only the default. The families where it should actually pay are the
/// TE-dominant ones (Mage-Flow, lens, Kolors, flux2-klein, Sana), each of which owes its own
/// measurement before flipping this — see `crate::text_encoder::stream`.
///
/// The historical coupled figures remain useful only as evidence for that exact staged composition;
/// the v3 calibration fingerprint prevents them from being read as resident+deferred evidence.
pub const TRANSFORMER_WINDOW_COMPONENT: TransformerComponent = TransformerComponent::Dit;

/// Resolve the numeric tier the transformer actually loads. A packed snapshot marker is
/// authoritative even when `LoadSpec::quantize` is absent; a requested tier is the fallback only for
/// a dense snapshot, where the loader performs the quantization in memory. The existing shared guard
/// rejects a requested tier that disagrees with a packed turnkey before admission can mislabel it.
pub(crate) fn loaded_tier(spec: &LoadSpec, provider_id: &str) -> CoreResult<MemoryNumericTier> {
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(_) => {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: actual numeric tier requires a snapshot directory"
            )))
        }
    };
    if let Some(requested) = spec.quantize {
        mlx_gen::quant::needs_load_time_quant(root, "transformer", requested.bits(), provider_id)
            .map_err(CoreError::backend)?;
    }
    let packed_bits =
        mlx_gen::quant::packed_quant_bits(root, "transformer").map_err(CoreError::backend)?;
    let quant = match packed_bits {
        Some(4) => Some(Quant::Q4),
        Some(8) => Some(Quant::Q8),
        Some(bits) => {
            return Err(CoreError::Unsupported(format!(
                "{provider_id}: transformer declares unsupported packed quantization width {bits}"
            )))
        }
        None => spec.quantize,
    };
    Ok(MemoryNumericTier {
        precision: spec.precision,
        quant,
        component_precision_floors: &[],
    })
}

/// Stable identity for the Fun-Controlnet-Union network held beside the base Z-Image model.
pub const CONTROL_STACK_COMPONENT_ID: &str = "fun_controlnet_union.control_layers";
pub const CONTROL_PERSISTENT_COMPONENT_ID: &str = "fun_controlnet_union.persistent";
const CONTROL_STACK_PREFIX: &str = "control_layers.";

/// The one bounded-attention parameter this provider accepts: the shared
/// [`mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET`] (64 Mi score elements per attention call),
/// the exact knob the Candle/CUDA Z-Image rung 3 measured in SC-15256 — reused verbatim so a
/// cross-backend comparison of the same rung is meaningful. It is the only candidate advertised in
/// `attention_chunk_sizes`, and the request scope re-validates it.
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;

/// Calibration content fingerprint. It must change whenever quantization floors, tensor layout, or
/// execution structure change in a way that invalidates measurements taken against this provider.
///
/// Load shape is a typed evidence-key axis; this content fingerprint remains shape-independent.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "z-image-mlx-independent-materialization-v3";

/// This provider's two bounded-decode routes, reconciled by the shared
/// [`DecodeRoutes`](mlx_gen_pid::DecodeRoutes) (SC-15775).
///
/// `use_pid` swaps the native VAE for a super-resolving student whose legal tiles live in a different
/// range of the same unit (see [`pid_decode_tile_edges`]), so the domain is per-route. The contract's
/// static `decode_tile_edges` publishes the **union** — it is the provider's whole production domain,
/// and the static validator is not route-aware — while `safety_check` and the request scope, which
/// both see `use_pid`, enforce the route's own subset. Defense in depth in the direction the contract
/// allows: narrower at admission, never broader.
///
/// The split used to be hand-rolled here. It is now the shared type's, because the PiD half of it is
/// not this provider's to state: `DecodeRoutes::new` takes only the native ladder, derives the PiD one
/// from the student, and **refuses to construct** when the native ladder reaches into the PiD domain —
/// the cross-provider hazard SC-15775 exists to close.
///
/// It is therefore fallible, and deliberately propagated rather than `expect`-ed: [`DECODE_TILE_EDGES`]
/// is a `const` this provider owns, so the `Err` arm is unreachable for any shipping load — and
/// `no_native_decode_candidate_is_a_legal_pid_tile_edge` proves it — but a future widening of the
/// ladder into the student's range must fail typed at load rather than panic in a release build.
fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(
        provider_id,
        DECODE_TILE_EDGES.iter().copied(),
        DECODE_OVERLAP,
    )
}

/// Build the Z-Image MLX provider contract for `provider_id`.
///
/// `spec` supplies the load-exact asset facts: the component `.safetensors` sums under the resolved
/// snapshot root, which is what the MLX loader actually materializes (the tier subdirectory is already
/// the spec root for a pre-quantized turnkey). A single-file (ComfyUI) source has no component tree,
/// so its asset facts stay zero rather than reporting a fabricated split.
///
/// Fallible when route geometry is invalid or required component assets cannot be inspected.
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    // SC-15754 / SC-15998: rung 4 is available only when this SPECIFIC load can execute it, and two
    // independent load-time
    // facts decide that. Declaring it here rather than failing at generate time is what keeps the
    // selector from choosing a strategy the loaded generator cannot run — the contract is built per
    // `LoadSpec`, so it can say so.
    //
    // 1. It rebuilds trunk blocks from the snapshot per window, so it needs a **re-openable source**.
    //    A `WeightsSource::Dir` (every registry load) has one; a single-file / in-place ComfyUI load
    //    does not.
    // 2. `LoadShape::DeferredMaterialization` says the request wants those re-openable blocks instead
    //    of bulk-committing the stack. `OffloadPolicy` is deliberately absent: phase release and
    //    intra-phase block materialization are separate axes.
    let streamable = matches!(spec.weights, mlx_gen::WeightsSource::Dir(_))
        && matches!(spec.load_shape, mlx_gen::LoadShape::DeferredMaterialization);
    let (asset_facts, resident_components) = asset_facts(provider_id, spec, streamable)?;
    memory_strategy_contract_with_asset_facts(
        provider_id,
        spec,
        streamable,
        asset_facts,
        resident_components,
    )
}

/// Declaration-equivalent contract used only by weights-free registry conformance.
pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let streamable = matches!(spec.weights, mlx_gen::WeightsSource::Dir(_))
        && matches!(spec.load_shape, mlx_gen::LoadShape::DeferredMaterialization);
    memory_strategy_contract_with_asset_facts(
        provider_id,
        spec,
        streamable,
        MemoryAssetFacts::default(),
        Vec::new(),
    )
}

fn memory_strategy_contract_with_asset_facts(
    provider_id: &str,
    spec: &LoadSpec,
    streamable: bool,
    asset_facts: MemoryAssetFacts,
    resident_components: Vec<MemoryResidentComponent>,
) -> CoreResult<MemoryProviderContract> {
    // Bound once: the declaration is checked at construction, so building it twice inside the
    // capability map would re-run the check and re-allocate for no gain.
    let routes = decode_routes(provider_id)?;
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
        // SC-16065: a control-variant load holds the Fun-Controlnet-Union network beside the
        // base model. This term is zero for the base provider and load-bearing when `spec`
        // carries the auxiliary checkpoint.
        MemoryFormulaVariable::OverlayBytes,
        MemoryFormulaVariable::DecodeTileArea,
        MemoryFormulaVariable::AttentionChunkSize,
        // SC-15754: transformer weight residency is now a *variable* of the peak, not a
        // constant folded into `AssetBytes` — at window 1 the trunk contributes one block
        // instead of thirty.
        MemoryFormulaVariable::TransformerWindowSize,
    ];
    let formula = if resident_components.is_empty() {
        MemoryFormulaKind::PhaseEnvelope { phases, variables }
    } else {
        MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables,
            resident_components,
        }
    };
    Ok(MemoryProviderContract {
        provider_id: provider_id.to_owned(),
        backend: MemoryBackendRealization::MlxMetal {
            // Unified memory: the wired-residency budget is what the staged phases release, weights
            // are mmap-backed, and MLX's lazy graph needs explicit `eval` before a phase drop frees
            // anything (`Residency::run_staged` owns that discipline). No host↔device transfer.
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
        strategies: MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: match strategy {
                    MemoryStrategy::BoundedTransformerResidency if !streamable => {
                        MemoryStrategySupport::Missing
                    }
                    _ => MemoryStrategySupport::Implemented,
                },
                parameters: match strategy {
                    MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                        decode_tile_edges: routes.published_edges(),
                        decode_overlaps: routes.published_overlaps(),
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                        attention_chunk_sizes: vec![ATTENTION_CHUNK_SIZE],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedTransformerResidency if streamable => {
                        MemoryParameterRanges {
                            transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                            transformer_window_components: TRANSFORMER_WINDOW_COMPONENTS.to_vec(),
                            ..Default::default()
                        }
                    }
                    _ => MemoryParameterRanges::default(),
                },
            })
            .collect(),
        pid_decode_routes: Some(mlx_gen::gen_core::MemoryPidDecodeRoutes {
            native: mlx_gen::gen_core::MemoryDecodeRouteDomain {
                tile_edges: routes.native_edges().to_vec(),
                tile_overlap: DECODE_OVERLAP,
            },
            pid: mlx_gen::gen_core::MemoryDecodeRouteDomain {
                tile_edges: mlx_gen_pid::DecodeRoutes::pid_edges(),
                tile_overlap: mlx_gen_pid::DecodeRoutes::pid_overlap(),
            },
        }),
        load_shape: spec.load_shape,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: mlx_gen::gen_core::ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: vec![
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: streamable,
        },
        formula,
        calibration: Some(MemoryCalibrationIdentity::new(
            MEMORY_CALIBRATION_FINGERPRINT,
            spec.load_shape,
        )),
        asset_facts,
        runtime: MemoryRuntimeSemantics::default(),
    })
}

/// Component `.safetensors` sums for the spec's snapshot root. A [`WeightsSource::File`] base source
/// has no component tree, so its base-model fields stay `0` (the truthful "unknown", not a guess);
/// a separately addressed control checkpoint remains independently countable.
fn asset_facts(
    provider_id: &str,
    spec: &LoadSpec,
    streamable: bool,
) -> CoreResult<(MemoryAssetFacts, Vec<MemoryResidentComponent>)> {
    let (mut components, effective_quant) = match &spec.weights {
        WeightsSource::Dir(root) => {
            // This is the same resolver every concrete loader captures for its safety gate and
            // selected-encoder policy. Besides naming a packed turnkey when no override was
            // requested, it fails closed on requested-vs-packed transformer mismatches before facts
            // can describe a tier the runtime would refuse.
            let effective_quant = loaded_tier(spec, provider_id)?.quant;
            // Heavy components follow the actual load call: only `spec.quantize` invokes their
            // in-memory `quantize` methods. A packed transformer with no request does make that tier
            // the model policy for a dense selected text encoder, but it does *not* implicitly pack a
            // separately dense VAE or ControlNet. Existing affine triples are preserved by the shared
            // header projector in either case.
            let project = |path: &std::path::Path| {
                projected_safetensors_bytes(path, |_| match spec.quantize {
                    Some(quant) => ResidentProjection::GroupQuantized {
                        bits: quant.bits(),
                        group_size: crate::quant::GROUP_SIZE as usize,
                    },
                    None => ResidentProjection::Stored,
                })
            };
            (
                mlx_gen::PerComponentBytes {
                    text_encoder: 0,
                    dit: project(&root.join("transformer"))?,
                    vae: project(&root.join("vae"))?,
                },
                effective_quant,
            )
        }
        // A combined single-file checkpoint cannot be split into the three contract components.
        // Zero remains the truthful unknown; the independently addressed control is still exact.
        WeightsSource::File(_) => (mlx_gen::PerComponentBytes::default(), None),
    };
    if let WeightsSource::Dir(root) = &spec.weights {
        let selected = crate::ENCODER_CONTRACT.source_for_load(spec, root)?;
        // Runtime derives this conversion from the effective transformer tier, not directly from
        // `spec.quantize`: a packed Q4/Q8 base with no explicit request still quantizes a dense
        // selected encoder to the loaded tier. Matching already-packed selections return `None` and
        // keep their exact stored triples; packed mismatches reject here exactly as at load.
        let load_time_quant_bits =
            selected.load_time_quant_bits(effective_quant.map(Quant::bits), provider_id)?;
        components.text_encoder = projected_tensor_headers_bytes(
            &selected.materialized_language_tensor_headers(&crate::ENCODER_CONTRACT)?,
            |_| match load_time_quant_bits {
                Some(bits) => ResidentProjection::GroupQuantized {
                    bits,
                    group_size: crate::quant::GROUP_SIZE as usize,
                },
                None => ResidentProjection::Stored,
            },
        )?;
    }
    let resident_components = match &spec.control {
        Some(source) => control_resident_components(source, spec.quantize, streamable)?,
        None => Vec::new(),
    };
    let control_bytes = resident_components.iter().fold(0_u64, |total, component| {
        total.saturating_add(component.resident_bytes)
    });
    Ok((
        MemoryAssetFacts {
            base_bytes: components
                .text_encoder
                .saturating_add(components.dit)
                .saturating_add(components.vae),
            conditioning_bytes: components.text_encoder,
            transformer_bytes: components.dit,
            decoder_bytes: components.vae,
            // The load-time control checkpoint is resident for every request made by a control-provider
            // instance. A request-selected PiD decoder remains excluded: availability on `LoadSpec` does
            // not mean the request selected it, so pricing it here would overstate the ordinary path.
            overlay_bytes: control_bytes,
        },
        resident_components,
    ))
}

/// Exact post-load tensor bytes for the ControlNet at this [`LoadSpec`]'s numeric tier. Dense
/// tensors retain their safetensors payload size. A Q4/Q8 request packs each group-aligned Linear
/// weight to u32 codes plus bf16 scale/bias tables, exactly matching `AdaptableLinear::quantize`;
/// non-packable leaves such as the 33-channel patch embedder remain dense. Existing packed triples
/// are counted as stored rather than projected a second time.
fn control_resident_components(
    source: &mlx_gen::WeightsSource,
    quant: Option<mlx_gen::gen_core::Quant>,
    streamable: bool,
) -> CoreResult<Vec<MemoryResidentComponent>> {
    let path = match source {
        mlx_gen::WeightsSource::Dir(path) | mlx_gen::WeightsSource::File(path) => path,
    };
    let bits = quant.map(|quant| quant.bits());
    let mut stack_bytes = 0_u64;
    let mut persistent_bytes = 0_u64;
    let tensors = projected_safetensors_tensors(path, |_| match bits {
        Some(bits) => ResidentProjection::GroupQuantized {
            bits,
            group_size: crate::quant::GROUP_SIZE as usize,
        },
        None => ResidentProjection::Stored,
    })?;
    for tensor in &tensors {
        let projected_bytes = tensor.resident_bytes;
        if tensor.name.starts_with(CONTROL_STACK_PREFIX) {
            stack_bytes = stack_bytes.saturating_add(projected_bytes);
        } else {
            persistent_bytes = persistent_bytes.saturating_add(projected_bytes);
        }
    }
    let mut components = Vec::with_capacity(2);
    if stack_bytes > 0 {
        components.push(MemoryResidentComponent {
            id: CONTROL_STACK_COMPONENT_ID.to_owned(),
            kind: MemoryComponentKind::ControlBranch,
            resident_bytes: stack_bytes,
            bounded_by: streamable.then_some(MemoryStrategy::BoundedTransformerResidency),
        });
    }
    if persistent_bytes > 0 {
        components.push(MemoryResidentComponent {
            id: CONTROL_PERSISTENT_COMPONENT_ID.to_owned(),
            kind: MemoryComponentKind::ControlBranch,
            resident_bytes: persistent_bytes,
            bounded_by: None,
        });
    }
    Ok(components)
}

/// The shared ladder → this provider's existing per-request execution controls.
///
/// The ladder is cumulative *by default*, so a rung-3 selection tiles the decode as well —
/// but SC-15805: that default is **defeasible and contract-owned**, so ask
/// [`MemoryProviderContract::engages`] which rungs this selection actually engages rather than
/// re-deriving it from a `match` over the ladder's numeric order. A hardcoded `..decode` arm is the
/// same hazard as a `>=` comparison wearing different syntax: it switches a rung's lever on
/// underneath a provider that does not declare that rung `Implemented`. For this provider's
/// current contract (rungs 1-3 always `Implemented`) the two agree exactly, which is why this is a
/// consistency fix rather than a behavior change.
///
/// `Resident` returns `None`, which is the historical fast path (`GenerationRequest::memory`
/// untouched).
/// The provider safety check every Z-Image variant shares: the calibration handshake, then the shared
/// contract's own selection validation, then the budget. Defense in depth only — it can reject, it can
/// never swap in a different strategy or numeric tier.
pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        // Route-aware decode-parameter validation (SC-15510).
        //
        // SC-15615 had to REFUSE rungs 2+ on the PiD route outright, because the super-resolving student
        // planned its own tile edge/overlap from `mlx_gen_pid::budget` and never read this contract's
        // parameters — admitting a selection would have executed a different strategy than the selector
        // chose. That is now reconciled rather than refused: `mint_planned_decoder_with_tiling` honours an
        // explicit plan and validates it against the planner's own invariants, so the PiD route is a
        // first-class rung-2 route with its own candidate domain.
        //
        // The domains do not overlap — the native VAE tiles the output at 512-768 px, the PiD student
        // tiles a `scale×` super-resolved output at 2048 px — so a selection built for one route is
        // rejected on the other rather than silently re-planned. The static `validate_selection` above
        // sees only the published union, which is why this check exists.
        // SC-15805: ask the contract whether this selection ENGAGES rung 2 rather than re-deriving it
        // from the enum's numeric order. Same answer for every shipping z-image load; the difference is
        // that the cost-order default now lives in exactly one documented place.
        // SC-15775: the check itself is the shared `DecodeRoutes` gate rather than a per-provider match,
        // so the next PiD-eligible adopter inherits it instead of re-deriving (or forgetting) it.
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            let routes = decode_routes(contract.provider_id.as_str())?;
            routes
                .validate(
                    context.use_pid,
                    context.selection.parameters.decode_tile_edge,
                    context.selection.parameters.decode_overlap,
                )
                .map_err(CoreError::Unsupported)?;
        }
        Ok(())
    };
    mlx_gen::gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(loaded_tier),
        Some(&route_gate),
    )
}

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match loaded_tier(spec, &contract.provider_id) {
        Ok(tier) => safety_check(contract, tier, context),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let tier = loaded_tier(spec, &contract.provider_id)?;
    let is_control = contract.provider_id.ends_with("_control");
    let route = |use_pid| mlx_gen::gen_core::MemoryBehaviorRoute {
        mode: if is_control {
            mlx_gen::gen_core::MemoryMode::ImageToImage
        } else {
            mlx_gen::gen_core::MemoryMode::TextToImage
        },
        reference_count: u32::from(is_control),
        use_pid,
        has_phases: false,
        overlay: None,
    };
    let mut fixtures = vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(
        mlx_gen::gen_core::standard_memory_behavior_context(
            contract,
            strategy,
            tier,
            route(false),
        )?,
    )];
    if contract.pid_decode_routes.is_some()
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

pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        provider_id,
        contract,
        loaded_tier(spec, provider_id)?,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

/// Open a request scope after `safety_check` accepted `context`.
pub(crate) fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    begin_request_with_cleanup(
        provider_id,
        contract,
        loaded_tier,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

fn begin_request_with_cleanup(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    loaded_tier: MemoryNumericTier,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(contract, loaded_tier, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let routes = decode_routes(provider_id)?;
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        crate::transformer::ZImageTransformerConfig::turbo().n_layers,
        move |use_pid, tile_edge, overlap| {
            routes
                .validate(use_pid, Some(tile_edge), Some(overlap))
                .map_err(CoreError::Unsupported)
        },
    )?;
    config.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
    config.transformer_window = (context.selection.strategy
        == MemoryStrategy::BoundedTransformerResidency)
        .then_some(())
        .and(context.selection.parameters.transformer_window_size);
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    )))
}

#[cfg(test)]
fn z_image_generation_memory(
    contract: &MemoryProviderContract,
    selection: &mlx_gen::gen_core::MemorySelection,
) -> Option<GenerationMemory> {
    contract.generation_memory(selection)
}

/// The strategy parameters this provider accepts, for a caller that wants the whole domain in one
/// value (the conformance tests and the SceneWorks evidence writer both key off this).
pub fn declared_parameters() -> MemoryStrategyParameters {
    MemoryStrategyParameters {
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
    use mlx_gen::attention::AttentionBudget;
    use mlx_gen::gen_core::WeightsSource;
    use mlx_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryMode, MemoryNumericTier, MemorySelection, Precision,
        Quant, MEMORY_CALIBRATION_ABI,
    };

    fn write_minimal_safetensors(path: &std::path::Path) {
        let mut header = br#"{"probe":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 2]);
        std::fs::write(path, bytes).unwrap();
    }

    fn write_typed_sparse_safetensors(
        path: &std::path::Path,
        tensors: &[(String, &'static str, Vec<usize>)],
    ) -> u64 {
        use std::io::Write as _;

        let mut offset = 0_u64;
        let mut header = serde_json::Map::new();
        for (name, dtype, shape) in tensors {
            let width = match *dtype {
                "F16" | "BF16" => 2_u64,
                "F32" | "U32" => 4_u64,
                dtype => panic!("unsupported test dtype {dtype}"),
            };
            let bytes = shape
                .iter()
                .try_fold(width, |total, dimension| {
                    total.checked_mul(*dimension as u64)
                })
                .unwrap();
            let end = offset.checked_add(bytes).unwrap();
            header.insert(
                name.clone(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, end],
                }),
            );
            offset = end;
        }
        let encoded = serde_json::to_vec(&header).unwrap();
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&(encoded.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&encoded).unwrap();
        file.set_len(8 + encoded.len() as u64 + offset).unwrap();
        offset
    }

    fn quantizable_probe_bytes(bits: i32) -> u64 {
        match bits {
            4 => 72,
            8 => 136,
            _ => panic!("test supports Q4/Q8 only"),
        }
    }

    fn write_dense_quantizable_probe(path: &std::path::Path, name: &str) -> u64 {
        write_typed_sparse_safetensors(path, &[(name.to_owned(), "BF16", vec![2, 64])])
    }

    fn write_packed_quantizable_probe(path: &std::path::Path, name: &str, bits: i32) -> u64 {
        write_typed_sparse_safetensors(
            path,
            &[
                (
                    format!("{name}.weight"),
                    "U32",
                    vec![2, 64 * bits as usize / 32],
                ),
                (format!("{name}.scales"), "BF16", vec![2, 1]),
                (format!("{name}.biases"), "BF16", vec![2, 1]),
            ],
        )
    }

    fn install_component_tier_fixture(root: &std::path::Path, packed_bits: Option<i32>) {
        let transformer = root.join("transformer");
        let vae = root.join("vae");
        match packed_bits {
            Some(bits) => {
                std::fs::write(
                    transformer.join("config.json"),
                    format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
                )
                .unwrap();
                assert_eq!(
                    write_packed_quantizable_probe(
                        &transformer.join("model.safetensors"),
                        "blocks.0.attn.to_q",
                        bits,
                    ),
                    quantizable_probe_bytes(bits)
                );
            }
            None => {
                let config = transformer.join("config.json");
                if config.exists() {
                    std::fs::remove_file(config).unwrap();
                }
                assert_eq!(
                    write_dense_quantizable_probe(
                        &transformer.join("model.safetensors"),
                        "blocks.0.attn.to_q.weight",
                    ),
                    256
                );
            }
        }
        assert_eq!(
            write_dense_quantizable_probe(
                &vae.join("model.safetensors"),
                "decoder.attn.to_q.weight",
            ),
            256
        );
    }

    fn write_dense_control_fixture(path: &std::path::Path) -> u64 {
        write_typed_sparse_safetensors(
            path,
            &[
                (
                    "control_layers.0.attn.to_q.weight".to_owned(),
                    "BF16",
                    vec![2, 64],
                ),
                (
                    "control_all_x_embedder.2-2.weight".to_owned(),
                    "BF16",
                    vec![2, 33],
                ),
                (
                    "control_all_x_embedder.2-2.bias".to_owned(),
                    "BF16",
                    vec![2],
                ),
            ],
        )
    }

    fn expected_conditioning_bytes(bits: Option<i32>) -> u64 {
        let contract = crate::ENCODER_CONTRACT;
        let attention_width = contract.num_attention_heads * contract.head_dim;
        let kv_width = contract.num_key_value_heads * contract.head_dim;
        let matrix_elements = contract.vocab_size * contract.hidden_size
            + contract.loaded_hidden_layers
                * (2 * attention_width * contract.hidden_size
                    + 2 * kv_width * contract.hidden_size
                    + 3 * contract.intermediate_size * contract.hidden_size);
        let matrix_elements = u64::try_from(matrix_elements).unwrap();
        let matrix_bytes = match bits {
            Some(bits) => {
                matrix_elements * u64::try_from(bits).unwrap() / 8 + matrix_elements / 64 * 4
            }
            None => matrix_elements * 2,
        };
        let vector_bytes = u64::try_from(
            contract.loaded_hidden_layers * (2 * contract.hidden_size + 2 * contract.head_dim) * 2,
        )
        .unwrap();
        matrix_bytes + vector_bytes
    }

    fn write_snapshot(root: &std::path::Path) {
        for component in ["transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            write_minimal_safetensors(&dir.join("model.safetensors"));
        }
        gen_core_testkit::write_encoder_contract_fixture(
            &root.join("text_encoder"),
            crate::ENCODER_CONTRACT,
        )
        .unwrap();
    }

    fn append_sparse_f16_tensor(path: &std::path::Path, name: &str, shape: &[usize]) {
        use std::io::{Read as _, Seek as _, SeekFrom, Write as _};

        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        let mut encoded_len = [0_u8; 8];
        file.read_exact(&mut encoded_len).unwrap();
        let mut encoded = vec![0_u8; u64::from_le_bytes(encoded_len) as usize];
        file.read_exact(&mut encoded).unwrap();
        let mut header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&encoded).unwrap();
        let start = header
            .values()
            .filter_map(|entry| entry["data_offsets"][1].as_u64())
            .max()
            .unwrap_or(0);
        let bytes = shape
            .iter()
            .try_fold(2_u64, |total, dimension| {
                total.checked_mul(*dimension as u64)
            })
            .unwrap();
        let end = start.checked_add(bytes).unwrap();
        assert!(header
            .insert(
                name.to_owned(),
                serde_json::json!({
                    "dtype": "F16",
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            )
            .is_none());
        let encoded = serde_json::to_vec(&header).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&(encoded.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&encoded).unwrap();
        file.set_len(8 + encoded.len() as u64 + end).unwrap();
    }

    #[derive(Clone, Copy, Debug)]
    enum EncoderSelection {
        Builtin,
        OverrideDir,
        OverrideFile,
        CompleteSnapshot,
    }

    fn snapshot_spec_with_encoder(
        tmp: &tempfile::TempDir,
        quant: Option<Quant>,
        selection: EncoderSelection,
    ) -> (std::path::PathBuf, LoadSpec, std::path::PathBuf) {
        let root = tmp.path().join("snapshot");
        write_snapshot(&root);
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        spec.quantize = quant;
        let selected = match selection {
            EncoderSelection::Builtin => root.join("text_encoder"),
            EncoderSelection::OverrideDir | EncoderSelection::OverrideFile => {
                let selected = tmp.path().join("selected-text-encoder");
                gen_core_testkit::write_encoder_contract_fixture(
                    &selected,
                    crate::ENCODER_CONTRACT,
                )
                .unwrap();
                spec.text_encoder = Some(match selection {
                    EncoderSelection::OverrideDir => WeightsSource::Dir(selected.clone()),
                    EncoderSelection::OverrideFile => {
                        WeightsSource::File(selected.join("model.safetensors"))
                    }
                    _ => unreachable!(),
                });
                selected
            }
            EncoderSelection::CompleteSnapshot => {
                let selected = tmp.path().join("selected-snapshot");
                gen_core_testkit::write_encoder_contract_fixture(
                    &selected.join("text_encoder"),
                    crate::ENCODER_CONTRACT,
                )
                .unwrap();
                spec.text_encoder = Some(WeightsSource::Dir(selected.clone()));
                selected.join("text_encoder")
            }
        };
        (root, spec, selected.join("model.safetensors"))
    }

    /// The load rung 4 is available on: a re-openable snapshot dir with deferred materialization.
    fn spec(tmp: &tempfile::TempDir) -> LoadSpec {
        let root = tmp.path().join("z-image-memory-spec");
        write_snapshot(&root);
        LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization)
    }

    fn contract(tmp: &tempfile::TempDir) -> MemoryProviderContract {
        memory_strategy_contract(crate::model::MODEL_ID, &spec(tmp)).unwrap()
    }

    #[test]
    fn required_component_directory_cannot_disappear_into_zero_facts() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        std::fs::remove_dir_all(root.join("text_encoder")).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        assert!(memory_strategy_contract(crate::model::MODEL_ID, &spec).is_err());
        assert!(weights_free_memory_strategy_contract(crate::model::MODEL_ID, &spec).is_ok());
    }

    #[test]
    fn conditioning_prices_only_the_materialized_36_layer_language_surface() {
        for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
            for selection in [
                EncoderSelection::Builtin,
                EncoderSelection::OverrideDir,
                EncoderSelection::OverrideFile,
                EncoderSelection::CompleteSnapshot,
            ] {
                let tmp = tempfile::tempdir().unwrap();
                let (root, spec, selected_path) =
                    snapshot_spec_with_encoder(&tmp, quant, selection);
                let conditioning = || {
                    memory_strategy_contract(crate::model::MODEL_ID, &spec)
                        .unwrap()
                        .asset_facts
                        .conditioning_bytes
                };
                let baseline = conditioning();
                let selected = crate::ENCODER_CONTRACT
                    .source_for_load(&spec, &root)
                    .unwrap();
                let names = selected
                    .materialized_language_tensor_headers(&crate::ENCODER_CONTRACT)
                    .unwrap()
                    .into_iter()
                    .map(|header| header.name)
                    .collect::<std::collections::BTreeSet<_>>();
                assert!(names.contains("model.layers.35.self_attn.q_proj.weight"));
                assert!(!names
                    .iter()
                    .any(|name| name.starts_with("model.layers.36.")));
                assert!(!names.contains("model.norm.weight"));

                for (name, shape) in [
                    (
                        "model.norm.weight",
                        vec![crate::ENCODER_CONTRACT.hidden_size],
                    ),
                    ("model.unrelated_projection.weight", vec![17]),
                ] {
                    append_sparse_f16_tensor(&selected_path, name, &shape);
                    assert_eq!(
                        conditioning(),
                        baseline,
                        "{quant:?} {selection:?} charged an unmaterialized tensor {name}"
                    );
                }
            }
        }
    }

    fn tiered_asset_spec(
        tmp: &tempfile::TempDir,
        base_packed_bits: Option<i32>,
        requested: Option<Quant>,
        selection: EncoderSelection,
        selected_packed_bits: Option<i32>,
    ) -> (LoadSpec, std::path::PathBuf) {
        let (root, spec, selected_path) = snapshot_spec_with_encoder(tmp, requested, selection);
        install_component_tier_fixture(&root, base_packed_bits);
        if let Some(bits) = selected_packed_bits {
            gen_core_testkit::write_encoder_contract_fixture_with_quant(
                selected_path.parent().unwrap(),
                crate::ENCODER_CONTRACT,
                Some(bits),
            )
            .unwrap();
        }
        let control = tmp.path().join("control.safetensors");
        assert_eq!(write_dense_control_fixture(&control), 392);
        (spec, control)
    }

    fn is_control_registration(registration: &mlx_gen::gen_core::MemoryRegistration) -> bool {
        matches!(
            registration.provider_id,
            crate::model_control::MODEL_ID | crate::model_base_control::MODEL_ID
        )
    }

    fn assert_all_route_asset_facts(
        spec: &LoadSpec,
        control: &std::path::Path,
        expected_conditioning: u64,
        expected_transformer: u64,
        expected_decoder: u64,
        expected_control: u64,
        label: &str,
    ) {
        for registration in memory_registrations() {
            let control_route = is_control_registration(&registration);
            let mut route_spec = spec.clone();
            if control_route {
                route_spec.control = Some(WeightsSource::File(control.to_path_buf()));
            }
            let contract = (registration.contract)(&route_spec).unwrap_or_else(|error| {
                panic!(
                    "{} {label} memory contract failed: {error}",
                    registration.provider_id
                )
            });
            let facts = contract.asset_facts;
            assert_eq!(
                facts.conditioning_bytes, expected_conditioning,
                "{} {label}: conditioning",
                registration.provider_id
            );
            assert_eq!(
                facts.transformer_bytes, expected_transformer,
                "{} {label}: transformer",
                registration.provider_id
            );
            assert_eq!(
                facts.decoder_bytes, expected_decoder,
                "{} {label}: decoder",
                registration.provider_id
            );
            assert_eq!(
                facts.base_bytes,
                expected_conditioning + expected_transformer + expected_decoder,
                "{} {label}: base",
                registration.provider_id
            );
            assert_eq!(
                facts.overlay_bytes,
                if control_route { expected_control } else { 0 },
                "{} {label}: control",
                registration.provider_id
            );
            assert_eq!(
                contract
                    .resident_components()
                    .iter()
                    .map(|component| component.resident_bytes)
                    .sum::<u64>(),
                facts.overlay_bytes,
                "{} {label}: decomposed control facts",
                registration.provider_id
            );
        }
    }

    #[test]
    fn prepacked_base_quantizes_every_dense_selected_encoder_surface_to_the_effective_tier() {
        for bits in [4, 8] {
            for selection in [
                EncoderSelection::Builtin,
                EncoderSelection::OverrideDir,
                EncoderSelection::OverrideFile,
            ] {
                let tmp = tempfile::tempdir().unwrap();
                let (spec, control) = tiered_asset_spec(&tmp, Some(bits), None, selection, None);
                assert_all_route_asset_facts(
                    &spec,
                    &control,
                    expected_conditioning_bytes(Some(bits)),
                    quantizable_probe_bytes(bits),
                    256,
                    392,
                    &format!("prepacked Q{bits} + dense {selection:?}"),
                );
            }
        }
    }

    #[test]
    fn matching_prepacked_selected_encoders_stay_exact_and_mismatches_reject_every_route() {
        for base_bits in [4, 8] {
            for selection in [
                EncoderSelection::Builtin,
                EncoderSelection::OverrideDir,
                EncoderSelection::OverrideFile,
            ] {
                let matching = tempfile::tempdir().unwrap();
                let (spec, control) =
                    tiered_asset_spec(&matching, Some(base_bits), None, selection, Some(base_bits));
                assert_all_route_asset_facts(
                    &spec,
                    &control,
                    expected_conditioning_bytes(Some(base_bits)),
                    quantizable_probe_bytes(base_bits),
                    256,
                    392,
                    &format!("matching prepacked Q{base_bits} {selection:?}"),
                );

                let mismatched = tempfile::tempdir().unwrap();
                let selected_bits = if base_bits == 4 { 8 } else { 4 };
                let (spec, control) = tiered_asset_spec(
                    &mismatched,
                    Some(base_bits),
                    None,
                    selection,
                    Some(selected_bits),
                );
                for registration in memory_registrations() {
                    let mut route_spec = spec.clone();
                    if is_control_registration(&registration) {
                        route_spec.control = Some(WeightsSource::File(control.clone()));
                    }
                    let error = (registration.contract)(&route_spec)
                        .expect_err("a packed selected-encoder mismatch must reject")
                        .to_string();
                    assert!(
                        error.contains(&format!("pre-quantized Q{selected_bits}"))
                            && error.contains(&format!("model policy is Q{base_bits}")),
                        "{} Q{base_bits}/Q{selected_bits} {selection:?}: {error}",
                        registration.provider_id
                    );
                }
            }
        }
    }

    #[test]
    fn dense_base_requested_quant_still_projects_every_runtime_quantized_component() {
        for (bits, requested) in [(4, Quant::Q4), (8, Quant::Q8)] {
            for selection in [
                EncoderSelection::Builtin,
                EncoderSelection::OverrideDir,
                EncoderSelection::OverrideFile,
            ] {
                let tmp = tempfile::tempdir().unwrap();
                let (spec, control) =
                    tiered_asset_spec(&tmp, None, Some(requested), selection, None);
                assert_all_route_asset_facts(
                    &spec,
                    &control,
                    expected_conditioning_bytes(Some(bits)),
                    quantizable_probe_bytes(bits),
                    quantizable_probe_bytes(bits),
                    quantizable_probe_bytes(bits) + 136,
                    &format!("dense + requested Q{bits} {selection:?}"),
                );
            }
        }
    }

    #[test]
    fn matching_request_over_a_prepacked_base_still_quantizes_dense_attachments() {
        for (bits, requested) in [(4, Quant::Q4), (8, Quant::Q8)] {
            let tmp = tempfile::tempdir().unwrap();
            let (spec, control) = tiered_asset_spec(
                &tmp,
                Some(bits),
                Some(requested),
                EncoderSelection::Builtin,
                None,
            );
            assert_all_route_asset_facts(
                &spec,
                &control,
                expected_conditioning_bytes(Some(bits)),
                quantizable_probe_bytes(bits),
                quantizable_probe_bytes(bits),
                quantizable_probe_bytes(bits) + 136,
                &format!("matching Q{bits} request over prepacked base"),
            );
        }
    }

    /// A selection carrying exactly the parameters the rungs up to and including `strategy` own —
    /// no more, no less, which is what the shared validator requires. `use_pid` picks the route's
    /// decode domain, since the two do not overlap.
    fn selection_for(strategy: MemoryStrategy, use_pid: bool) -> MemorySelection {
        let (edges, overlap) = decode_routes(crate::model::MODEL_ID)
            .unwrap()
            .domain(use_pid);
        let edge = if use_pid {
            // The largest PiD candidate, so the value is unambiguously from the PiD ladder.
            edges[0]
        } else {
            DECODE_TILE_EDGE
        };
        let decode = MemoryStrategyParameters {
            decode_tile_edge: Some(edge),
            decode_overlap: Some(overlap),
            ..Default::default()
        };
        MemorySelection {
            strategy,
            parameters: match strategy {
                MemoryStrategy::Resident | MemoryStrategy::StagedResidency => {
                    MemoryStrategyParameters::default()
                }
                MemoryStrategy::BoundedDecode => decode,
                MemoryStrategy::BoundedAttention => MemoryStrategyParameters {
                    attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                    ..decode
                },
                MemoryStrategy::BoundedTransformerResidency => MemoryStrategyParameters {
                    attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                    transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
                    transformer_window_component: Some(TRANSFORMER_WINDOW_COMPONENT),
                    ..decode
                },
            },
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
                component_precision_floors: &[],
            },
        }
    }

    fn selection(strategy: MemoryStrategy) -> MemorySelection {
        selection_for(strategy, false)
    }

    fn context_for(strategy: MemoryStrategy, use_pid: bool) -> MemoryRunContext {
        MemoryRunContext {
            selection: selection_for(strategy, use_pid),
            calibration_abi: MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: MEMORY_CALIBRATION_FINGERPRINT.to_owned(),
            load_shape: mlx_gen::LoadShape::DeferredMaterialization,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid,
            has_phases: true,
            geometry: MemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 8 * 1000 * 1000 * 1000,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 2 * 1000 * 1000 * 1000,
            },
            predicted_peak_bytes: 4 * 1000 * 1000 * 1000,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "test".to_owned(),
        }
    }

    fn tier_spec(
        tmp: &tempfile::TempDir,
        tag: &str,
        packed_bits: Option<i32>,
        requested: Option<Quant>,
    ) -> (std::path::PathBuf, LoadSpec) {
        let root = tmp.path().join(format!(
            "z-image-memory-tier-{tag}-{:?}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
        ));
        write_snapshot(&root);
        if let Some(bits) = packed_bits {
            std::fs::write(
                root.join("transformer/config.json"),
                format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            )
            .unwrap();
        }
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization);
        spec.quantize = requested;
        (root, spec)
    }

    fn resident_tier_context(
        contract: &MemoryProviderContract,
        quant: Option<Quant>,
    ) -> MemoryRunContext {
        let mut context = context_for(MemoryStrategy::Resident, false);
        let calibration = contract.calibration.as_ref().unwrap();
        context.calibration_abi = calibration.abi;
        context.calibration_fingerprint = calibration.fingerprint.clone();
        context.load_shape = calibration.load_shape;
        context.selection.tier.quant = quant;
        context
    }

    fn memory_registrations() -> [mlx_gen::gen_core::MemoryRegistration; 4] {
        [
            crate::model::MEMORY_REGISTRATION,
            crate::model_base::MEMORY_REGISTRATION,
            crate::model_control::MEMORY_REGISTRATION,
            crate::model_base_control::MEMORY_REGISTRATION,
        ]
    }

    #[test]
    fn all_four_registrations_bind_prepacked_q4_and_q8_without_an_override() {
        let tmp = tempfile::tempdir().unwrap();
        for (bits, actual, wrong) in [
            (4, Quant::Q4, Some(Quant::Q8)),
            (8, Quant::Q8, Some(Quant::Q4)),
        ] {
            let (root, spec) = tier_spec(&tmp, &format!("packed-{bits}"), Some(bits), None);
            for registration in memory_registrations() {
                let contract = (registration.contract)(&spec).unwrap();
                assert_eq!(
                    loaded_tier(&spec, registration.provider_id).unwrap().quant,
                    Some(actual),
                    "{} loaded-generator capture must use the packed marker",
                    registration.provider_id
                );
                assert_eq!(
                    (registration.safety_check)(
                        &spec,
                        &contract,
                        &resident_tier_context(&contract, Some(actual)),
                    ),
                    MemorySafetyDecision::Accept,
                    "{} must admit its actual packed tier",
                    registration.provider_id
                );
                for selected in [None, wrong] {
                    assert!(matches!(
                        (registration.safety_check)(
                            &spec,
                            &contract,
                            &resident_tier_context(&contract, selected),
                        ),
                        MemorySafetyDecision::Reject { reason }
                            if reason.contains("does not match loaded tier")
                    ));
                }
            }
            std::fs::remove_dir_all(root).ok();
        }
    }

    #[test]
    fn all_four_registrations_use_requested_quant_only_for_dense_snapshots() {
        let tmp = tempfile::tempdir().unwrap();
        for requested in [Quant::Q4, Quant::Q8] {
            let (root, spec) = tier_spec(&tmp, "dense-requested", None, Some(requested));
            for registration in memory_registrations() {
                let contract = (registration.contract)(&spec).unwrap();
                assert_eq!(
                    loaded_tier(&spec, registration.provider_id).unwrap().quant,
                    Some(requested)
                );
                assert_eq!(
                    (registration.safety_check)(
                        &spec,
                        &contract,
                        &resident_tier_context(&contract, Some(requested)),
                    ),
                    MemorySafetyDecision::Accept
                );
            }
            std::fs::remove_dir_all(root).ok();
        }
    }

    #[test]
    fn packed_request_mismatch_rejects_all_registrations_and_load_entrypoints() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = tier_spec(&tmp, "packed-mismatch", Some(8), Some(Quant::Q4));
        for registration in memory_registrations() {
            let error = (registration.contract)(&spec)
                .expect_err("asset-fact resolution must reject the impossible requested tier")
                .to_string();
            assert!(
                error.contains("pre-quantized Q8") && error.contains("Q4"),
                "{}: {error}",
                registration.provider_id
            );
            // The independently callable admission seam retains the same defense in depth even
            // when fed a weights-free declaration that intentionally skips asset inspection.
            let contract =
                weights_free_memory_strategy_contract(registration.provider_id, &spec).unwrap();
            let decision = (registration.safety_check)(
                &spec,
                &contract,
                &resident_tier_context(&contract, Some(Quant::Q4)),
            );
            assert!(matches!(
                decision,
                MemorySafetyDecision::Reject { reason }
                    if reason.contains("pre-quantized Q8") && reason.contains("Q4")
            ));
        }

        type Loader = fn(&LoadSpec) -> mlx_gen::Result<Box<dyn mlx_gen::Generator>>;
        let mut control_spec = spec.clone();
        control_spec.control = Some(WeightsSource::File("/control.safetensors".into()));
        for (provider_id, load, load_spec) in [
            (crate::model::MODEL_ID, crate::model::load as Loader, &spec),
            (
                crate::model_base::MODEL_ID,
                crate::model_base::load as Loader,
                &spec,
            ),
            (
                crate::model_control::MODEL_ID,
                crate::model_control::load as Loader,
                &control_spec,
            ),
            (
                crate::model_base_control::MODEL_ID,
                crate::model_base_control::load as Loader,
                &control_spec,
            ),
        ] {
            let error = load(load_spec)
                .err()
                .expect("tier mismatch must reject")
                .to_string();
            assert!(error.contains(provider_id), "{provider_id}: {error}");
            assert!(error.contains("pre-quantized Q8"), "{provider_id}: {error}");
        }
        std::fs::remove_dir_all(root).ok();
    }

    fn context(strategy: MemoryStrategy) -> MemoryRunContext {
        context_for(strategy, false)
    }

    #[test]
    fn contract_is_internally_conformant() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract(&tmp);
        assert_eq!(contract.conformance_errors(), Vec::<String>::new());
        gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
        // SC-15775: the PiD-eligible half of the same conformance obligation, run on the DECLARATION
        // INPUTS so the failure lands here with a named message rather than as a load-time `Err` from
        // `decode_routes`. The shared check refuses a native ladder that reaches into the PiD student's
        // tile domain, which is what keeps a `use_pid` + rung-2 request from carrying a geometry the
        // shared seam would have to reject at generate time.
        let routes = mlx_gen_pid::assert_decode_routes(
            crate::model::MODEL_ID,
            DECODE_TILE_EDGES.iter().copied(),
            DECODE_OVERLAP,
        );
        // The contract really publishes that declaration's union, not a separately-maintained list.
        let published = contract
            .strategies
            .iter()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedDecode)
            .expect("rung 2 is declared");
        assert_eq!(
            published.parameters.decode_tile_edges,
            routes.published_edges()
        );
        assert_eq!(
            published.parameters.decode_overlaps,
            routes.published_overlaps()
        );
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            MEMORY_CALIBRATION_FINGERPRINT
        );
    }

    /// SC-16065: the control branch is a second resident network, not the base DiT. This one test is
    /// deliberately mutation-sensitive across all three required declaration legs: its byte count,
    /// its typed component identity, and the formula term that makes those bytes affect admission.
    #[test]
    fn control_branch_is_a_decomposed_load_bearing_peak_component() {
        use safetensors::{serialize, tensor::TensorView, Dtype};
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let control = root.join("control.safetensors");
        let stack_weight = vec![0_u8; 2 * 64 * 2];
        let patch_weight = vec![0_u8; 2 * 33 * 2];
        let patch_bias = vec![0_u8; 2 * 2];
        let stack = TensorView::new(Dtype::BF16, vec![2, 64], &stack_weight).unwrap();
        let patch = TensorView::new(Dtype::BF16, vec![2, 33], &patch_weight).unwrap();
        let bias = TensorView::new(Dtype::BF16, vec![2], &patch_bias).unwrap();
        let bytes = serialize(
            [
                ("control_layers.0.attn.to_q.weight", stack),
                ("control_all_x_embedder.2-2.weight", patch),
                ("control_all_x_embedder.2-2.bias", bias),
            ],
            &None,
        )
        .unwrap();
        std::fs::write(&control, bytes).unwrap();
        write_snapshot(&root);
        let base_weight = vec![0_u8; 2 * 64 * 2];
        let base = TensorView::new(Dtype::BF16, vec![2, 64], &base_weight).unwrap();
        let base_bytes = serialize([("blocks.0.attn.to_q.weight", base)], &None).unwrap();
        std::fs::write(root.join("transformer/model.safetensors"), base_bytes).unwrap();

        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization)
            .with_quant(Quant::Q4)
            .with_control(WeightsSource::File(control));
        let contract = memory_strategy_contract(crate::model_control::MODEL_ID, &spec).unwrap();
        assert_eq!(crate::control_transformer::CONTROL_LAYERS_PLACES.len(), 15);

        // The 2x64 stack Linear is packed to 64 bytes of Q4 codes plus two 2x1 bf16 tables
        // (8 bytes). The non-divisible 2x33 patch weight and its bias stay dense (136 bytes).
        const STACK_RESIDENT_BYTES: u64 = 72;
        const PERSISTENT_RESIDENT_BYTES: u64 = 136;
        const CONTROL_RESIDENT_BYTES: u64 = STACK_RESIDENT_BYTES + PERSISTENT_RESIDENT_BYTES;
        let selected = crate::ENCODER_CONTRACT
            .source_for_load(&spec, &root)
            .unwrap();
        let expected_conditioning_bytes = projected_tensor_headers_bytes(
            &selected
                .materialized_language_tensor_headers(&crate::ENCODER_CONTRACT)
                .unwrap(),
            |_| ResidentProjection::GroupQuantized {
                bits: Quant::Q4.bits(),
                group_size: crate::quant::GROUP_SIZE as usize,
            },
        )
        .unwrap();
        assert_eq!(
            contract.asset_facts.conditioning_bytes,
            expected_conditioning_bytes
        );
        assert_eq!(contract.asset_facts.transformer_bytes, 72);
        assert_eq!(contract.asset_facts.decoder_bytes, 2);
        assert_eq!(
            contract.asset_facts.base_bytes,
            expected_conditioning_bytes + 74
        );
        assert_eq!(contract.asset_facts.overlay_bytes, CONTROL_RESIDENT_BYTES);
        let component = contract
            .resident_components()
            .iter()
            .find(|component| component.id == CONTROL_STACK_COMPONENT_ID)
            .expect("Z-Image must declare the 15-block ControlNet stack as its own component");
        assert_eq!(component.kind, MemoryComponentKind::ControlBranch);
        assert_ne!(
            component.kind,
            MemoryComponentKind::Transformer(TransformerComponent::Dit),
            "the ControlNet must not be reported as the base denoising transformer"
        );
        assert_eq!(component.resident_bytes, STACK_RESIDENT_BYTES);
        assert_eq!(
            component.bounded_by,
            Some(MemoryStrategy::BoundedTransformerResidency),
            "the 15-block control stack is streamed by rung 4"
        );
        let persistent = contract
            .resident_components()
            .iter()
            .find(|component| component.id == CONTROL_PERSISTENT_COMPONENT_ID)
            .expect("the patch embedder and two-block refiner remain separately attributable");
        assert_eq!(persistent.kind, MemoryComponentKind::ControlBranch);
        assert_eq!(persistent.resident_bytes, PERSISTENT_RESIDENT_BYTES);
        assert_eq!(persistent.bounded_by, None);
        assert!(
            contract.formula.uses(MemoryFormulaVariable::OverlayBytes),
            "declared auxiliary bytes are inert unless the provider formula includes them"
        );

        const BASE_PEAK: u64 = 1_000;
        let prediction = contract.predicted_peak_from_base(BASE_PEAK);
        assert_eq!(
            prediction.predicted_peak_bytes(),
            BASE_PEAK + CONTROL_RESIDENT_BYTES,
            "zeroing the auxiliary bytes, removing its component, or dropping OverlayBytes from the \
             formula must change this result"
        );
        assert_eq!(prediction.unattributed_bytes, BASE_PEAK);
        assert_eq!(prediction.components, contract.resident_components());
    }

    /// SC-15775: the route domains are disjoint **by construction**, not by the current numbers
    /// happening not to collide.
    ///
    /// The native ladder is a real-weight measurement that could legitimately move; what may never
    /// move is that a native edge is never also a legal PiD tile, because the student decodes a
    /// `scale×` super-resolved output and its floor sits above the whole native output. This asserts
    /// the property against `mlx_gen_pid`'s own predicate rather than against the literals, so a future
    /// widening of either side is caught here.
    #[test]
    fn no_native_decode_candidate_is_a_legal_pid_tile_edge() {
        let routes = decode_routes(crate::model::MODEL_ID).unwrap();
        // Not a vacuous pass: the same constructor REFUSES this provider's ladder widened by one
        // PiD-legal edge, which is exactly the mutation (M1) this check exists to catch. So the `Ok`
        // above is a property of these numbers, not of the check being toothless.
        let widened: Vec<u32> = DECODE_TILE_EDGES
            .iter()
            .copied()
            .chain([mlx_gen_pid::budget::MIN_TILE_EDGE as u32])
            .collect();
        let errors =
            mlx_gen_pid::DecodeRoutes::new(crate::model::MODEL_ID, widened, DECODE_OVERLAP)
                .unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.contains("ALSO a legal PiD tile edge")),
            "{errors:?}"
        );
        for &edge in routes.native_edges() {
            assert!(
                !mlx_gen_pid::budget::is_tile_edge_candidate(edge as i32),
                "native candidate {edge} is also a legal PiD tile edge; the shared seam could not \
                 tell the routes apart"
            );
        }
        // The rejected native edges (which a future sweep might readmit) are on the same side.
        for &edge in DECODE_TILE_EDGES_REJECTED {
            assert!(!mlx_gen_pid::budget::is_tile_edge_candidate(edge as i32));
        }
        // And the PiD ladder this provider publishes is one the seam will actually honour.
        for edge in pid_decode_tile_edges() {
            assert!(mlx_gen_pid::budget::is_tile_edge_candidate(edge as i32));
            assert!(!DECODE_TILE_EDGES.contains(&edge));
        }
    }

    /// The fingerprint must move when the execution structure does. SC-15998 removes staged
    /// residency from rung 4's shared composition, so evidence from the coupled v2 execution must
    /// not be readable as covering this one.
    #[test]
    fn the_fingerprint_retired_the_coupled_staged_window_generation() {
        let tmp = tempfile::tempdir().unwrap();
        assert_ne!(
            MEMORY_CALIBRATION_FINGERPRINT,
            "z-image-mlx-staged-tiled-decode-bounded-attention-block-window-v2"
        );
        let contract = contract(&tmp);
        let mut ctx = context(MemoryStrategy::BoundedAttention);
        ctx.calibration_fingerprint =
            "z-image-mlx-staged-tiled-decode-bounded-attention-block-window-v2".to_owned();
        assert!(matches!(
            safety_check(&contract, ctx.selection.tier, &ctx),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn eager_and_deferred_evidence_identities_cannot_cross_authorize() {
        let tmp = tempfile::tempdir().unwrap();
        let deferred = contract(&tmp);
        let mut eager_spec = spec(&tmp);
        eager_spec.load_shape = mlx_gen::LoadShape::EagerMaterialization;
        let eager = memory_strategy_contract(crate::model::MODEL_ID, &eager_spec).unwrap();
        assert_eq!(
            deferred.calibration.as_ref().unwrap().fingerprint,
            MEMORY_CALIBRATION_FINGERPRINT
        );
        assert_eq!(
            eager.calibration.as_ref().unwrap().fingerprint,
            MEMORY_CALIBRATION_FINGERPRINT
        );
        assert_ne!(
            deferred.calibration.as_ref().unwrap().load_shape,
            eager.calibration.as_ref().unwrap().load_shape
        );

        // Use a lower rung implemented by both contracts so rejection proves evidence identity,
        // not Deferred-only rung-4 availability.
        let mut deferred_context = context(MemoryStrategy::BoundedAttention);
        deferred_context.calibration_fingerprint = MEMORY_CALIBRATION_FINGERPRINT.to_owned();
        assert!(matches!(
            safety_check(&eager, deferred_context.selection.tier, &deferred_context),
            MemorySafetyDecision::Reject { reason }
                if reason.contains("calibration handshake mismatch")
        ));

        let mut eager_context = context(MemoryStrategy::BoundedAttention);
        eager_context.calibration_fingerprint = MEMORY_CALIBRATION_FINGERPRINT.to_owned();
        eager_context.load_shape = mlx_gen::LoadShape::EagerMaterialization;
        assert!(matches!(
            safety_check(&deferred, eager_context.selection.tier, &eager_context),
            MemorySafetyDecision::Reject { reason }
                if reason.contains("calibration handshake mismatch")
        ));
    }

    #[test]
    fn every_rung_is_implemented_and_selectable_on_a_snapshot_load() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract(&tmp);
        for strategy in MemoryStrategy::ALL {
            assert!(
                matches!(
                    contract.capability(strategy).map(|c| &c.support),
                    Some(MemoryStrategySupport::Implemented)
                ),
                "{strategy:?} must be Implemented"
            );
            contract.validate_selection(&selection(strategy)).unwrap();
        }
    }

    /// SC-15615: rung 3's exact chunk parameter is recorded — the same 64 Mi score budget the Candle
    /// twin measured, so the two backends' rung 3 is the same knob. Nothing else is accepted, at
    /// either the contract or the scope layer.
    #[test]
    fn bounded_attention_records_exactly_one_chunk_parameter() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract(&tmp);
        let capability = contract
            .capability(MemoryStrategy::BoundedAttention)
            .unwrap();
        assert_eq!(
            capability.parameters.attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        assert_eq!(ATTENTION_CHUNK_SIZE, 64 * 1024 * 1024);
        assert!(contract.lifecycle.attention_chunking);

        let mut sel = selection(MemoryStrategy::BoundedAttention);
        sel.parameters.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE / 2);
        let err = contract.validate_selection(&sel).unwrap_err().to_string();
        assert!(err.contains("attention"), "{err}");

        let mut sel = selection(MemoryStrategy::BoundedAttention);
        sel.parameters.attention_chunk_size = None;
        assert!(contract.validate_selection(&sel).is_err());
    }

    /// SC-15510: rung 2 publishes a **ladder**, not a point, which is what SC-15508's "a single-point
    /// pass cannot mark untested production parameters Verified" requires. The default is still the
    /// sc-13571 512/64, so nothing about an unparameterized render changed.
    #[test]
    fn bounded_decode_publishes_a_candidate_ladder_with_the_historical_default_in_it() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract(&tmp);
        let capability = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
        assert!(
            capability.parameters.decode_tile_edges.len() > 1,
            "a one-element domain cannot be swept"
        );
        assert!(capability
            .parameters
            .decode_tile_edges
            .contains(&DECODE_TILE_EDGE));
        // The set is measured, not inherited (see `DECODE_TILE_EDGES`): the sub-512 edges Qwen ships
        // were swept on real weights and rejected for seaming without buying a request-level saving.
        assert_eq!(DECODE_TILE_EDGES, &[768, 640, 512]);
        assert_eq!(DECODE_TILE_EDGES_REJECTED, &[448, 384, 320, 256]);
        // The two sets are disjoint, and a rejected edge is refused end to end — not merely absent
        // from a doc comment.
        for &rejected in DECODE_TILE_EDGES_REJECTED {
            assert!(!DECODE_TILE_EDGES.contains(&rejected));
            let mut ctx = context(MemoryStrategy::BoundedDecode);
            ctx.selection.parameters.decode_tile_edge = Some(rejected);
            assert!(
                contract.validate_selection(&ctx.selection).is_err(),
                "rejected edge {rejected} must not validate"
            );
            assert!(matches!(
                safety_check(&contract, ctx.selection.tier, &ctx),
                MemorySafetyDecision::Reject { .. }
            ));
        }
        // Every native candidate is selectable end to end, not just advertised.
        for &edge in DECODE_TILE_EDGES {
            let mut ctx = context(MemoryStrategy::BoundedDecode);
            ctx.selection.parameters.decode_tile_edge = Some(edge);
            assert!(
                matches!(
                    safety_check(&contract, ctx.selection.tier, &ctx),
                    MemorySafetyDecision::Accept
                ),
                "native candidate {edge} must be admissible"
            );
            let mut scope =
                begin_request(crate::model::MODEL_ID, &contract, ctx.selection.tier, &ctx)
                    .unwrap()
                    .unwrap();
            scope
                .configure_decode(edge, DECODE_OVERLAP, ctx.geometry)
                .unwrap();
            scope.finish(MemoryRunOutcome::Complete).unwrap();
        }
        // An edge outside the ladder is refused at both layers.
        let mut ctx = context(MemoryStrategy::BoundedDecode);
        ctx.selection.parameters.decode_tile_edge = Some(500);
        assert!(contract.validate_selection(&ctx.selection).is_err());
        assert!(matches!(
            safety_check(&contract, ctx.selection.tier, &ctx),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    /// SC-15754: rung 4 is Implemented for a snapshot load, with its window ladder recorded, and 30
    /// (the whole stack) is deliberately absent — one all-covering window bounds nothing, so
    /// advertising it would let a "rung 4" selection run the resident stack and record a zero saving.
    #[test]
    fn bounded_transformer_residency_publishes_only_bounding_windows() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract(&tmp);
        let capability = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert!(matches!(
            capability.support,
            MemoryStrategySupport::Implemented
        ));
        assert!(contract.lifecycle.transformer_window_materialization);
        assert_eq!(
            capability.parameters.transformer_window_sizes,
            TRANSFORMER_WINDOW_SIZES.to_vec()
        );
        let n_layers = crate::transformer::ZImageTransformerConfig::turbo().n_layers as u32;
        for &w in TRANSFORMER_WINDOW_SIZES {
            assert!(w >= 1 && w < n_layers, "window {w} must bound something");
        }
        assert!(TRANSFORMER_WINDOW_SIZES.contains(&TRANSFORMER_WINDOW_SIZE));

        // A window outside the ladder is refused by the static validator.
        let mut sel = selection(MemoryStrategy::BoundedTransformerResidency);
        sel.parameters.transformer_window_size = Some(7);
        assert!(contract.validate_selection(&sel).is_err());
        // As is one that omits the parameter entirely.
        let mut sel = selection(MemoryStrategy::BoundedTransformerResidency);
        sel.parameters.transformer_window_size = None;
        assert!(contract.validate_selection(&sel).is_err());
    }

    /// Rung 4 is declared per load, and both load-time preconditions are enforced at the contract
    /// layer rather than at generate time.
    #[test]
    fn rung_four_availability_uses_source_and_load_shape_not_offload_policy() {
        let tmp = tempfile::tempdir().unwrap();
        let mut eager = spec(&tmp);
        eager.load_shape = mlx_gen::LoadShape::EagerMaterialization;
        eager.offload_policy = mlx_gen::OffloadPolicy::Resident;
        let contract = memory_strategy_contract(crate::model::MODEL_ID, &eager).unwrap();
        assert_eq!(contract.conformance_errors(), Vec::<String>::new());
        assert!(matches!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .map(|c| &c.support),
            Some(MemoryStrategySupport::Missing)
        ));
        assert!(!contract.lifecycle.transformer_window_materialization);
        assert!(contract
            .validate_selection(&selection(MemoryStrategy::BoundedTransformerResidency))
            .is_err());
        // Every rung below it stays available — this narrows one cell, not the ladder.
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
        ] {
            contract.validate_selection(&selection(strategy)).unwrap();
        }
        // Resident+Deferred advertises rung 4: phase release is not the discriminator.
        let deferred =
            super::memory_strategy_contract(crate::model::MODEL_ID, &spec(&tmp)).unwrap();
        assert!(matches!(
            deferred
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .map(|c| &c.support),
            Some(MemoryStrategySupport::Implemented)
        ));
        // Sequential+Eager remains unavailable: staged residency does not imply deferred blocks.
        let mut staged_eager = spec(&tmp);
        staged_eager.load_shape = mlx_gen::LoadShape::EagerMaterialization;
        staged_eager.offload_policy = mlx_gen::OffloadPolicy::Sequential;
        let staged_eager =
            super::memory_strategy_contract(crate::model::MODEL_ID, &staged_eager).unwrap();
        assert!(matches!(
            staged_eager
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .map(|c| &c.support),
            Some(MemoryStrategySupport::Missing)
        ));
    }

    /// A load with no re-openable source (a ComfyUI single-file build) cannot stream blocks, so the
    /// rung is declared `Missing` for it rather than advertised and then failed at run time.
    #[test]
    fn a_single_file_load_declares_rung_four_missing() {
        let spec = LoadSpec::new(WeightsSource::File(
            "/nonexistent/z-image.safetensors".into(),
        ))
        .with_load_shape(mlx_gen::LoadShape::DeferredMaterialization);
        let contract = memory_strategy_contract(crate::model::MODEL_ID, &spec).unwrap();
        assert_eq!(contract.conformance_errors(), Vec::<String>::new());
        assert!(matches!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .map(|c| &c.support),
            Some(MemoryStrategySupport::Missing)
        ));
        assert!(!contract.lifecycle.transformer_window_materialization);
        assert!(contract
            .validate_selection(&selection(MemoryStrategy::BoundedTransformerResidency))
            .is_err());
        // ...and its asset facts stay the truthful zero (no component tree to sum).
        assert_eq!(contract.asset_facts, MemoryAssetFacts::default());
    }

    /// SC-15794: the rung-4 **component scope** must survive the selection -> `GenerationMemory`
    /// mapping. This is the seam where it is easiest to lose: everything still compiles, the window
    /// size still arrives, the run still succeeds — and the encoder silently never streams while the
    /// evidence records that it did.
    #[test]
    fn the_rung_four_component_scope_reaches_the_request() {
        let tmp = tempfile::tempdir().unwrap();
        for component in [
            TransformerComponent::Dit,
            TransformerComponent::TextEncoder,
            TransformerComponent::Both,
        ] {
            let mut selection = selection_for(MemoryStrategy::BoundedTransformerResidency, false);
            selection.parameters.transformer_window_component = Some(component);
            let memory = z_image_generation_memory(&contract(&tmp), &selection)
                .expect("rung 4 maps to a request-scoped control set");
            assert_eq!(
                memory.transformer_window_component,
                Some(component),
                "the {component:?} scope was dropped between the selection and the request"
            );
        }

        // A selection written before the component existed must keep its exact previous meaning: the
        // DiT-only default, not "unset".
        let selection = selection_for(MemoryStrategy::BoundedTransformerResidency, false);
        assert_eq!(
            selection.parameters.transformer_window_component,
            Some(TRANSFORMER_WINDOW_COMPONENT)
        );
        let memory =
            z_image_generation_memory(&contract(&tmp), &selection).expect("rung 4 controls");
        assert_eq!(
            memory.transformer_window_component,
            Some(TransformerComponent::Dit),
            "the provider's declared production scope is Dit; the request must say so explicitly \
             rather than relying on a None that a later reader could interpret differently"
        );

        // Below rung 4 the scope is meaningless and must not be smuggled in.
        for lower in [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
        ] {
            let memory = z_image_generation_memory(&contract(&tmp), &selection_for(lower, false))
                .expect("controls");
            assert!(
                memory.transformer_window_component.is_none(),
                "{lower:?} carried a rung-4 component scope"
            );
            assert!(!memory.stream_transformer_blocks);
        }
    }

    #[test]
    fn the_ladder_maps_to_cumulative_request_controls() {
        let tmp = tempfile::tempdir().unwrap();
        let decode = GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(DECODE_TILE_EDGE),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        };
        assert_eq!(
            z_image_generation_memory(&contract(&tmp), &selection(MemoryStrategy::Resident)),
            None
        );
        assert_eq!(
            z_image_generation_memory(&contract(&tmp), &selection(MemoryStrategy::StagedResidency)),
            Some(GenerationMemory {
                stage_residency: true,
                ..Default::default()
            })
        );
        assert_eq!(
            z_image_generation_memory(&contract(&tmp), &selection(MemoryStrategy::BoundedDecode)),
            Some(decode)
        );
        assert_eq!(
            z_image_generation_memory(
                &contract(&tmp),
                &selection(MemoryStrategy::BoundedAttention)
            ),
            Some(GenerationMemory {
                chunk_attention: true,
                attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                ..decode
            })
        );
        assert_eq!(
            z_image_generation_memory(
                &contract(&tmp),
                &selection(MemoryStrategy::BoundedTransformerResidency)
            ),
            Some(GenerationMemory {
                chunk_attention: true,
                attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                stream_transformer_blocks: true,
                transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
                // SC-15794: rung 4 now carries its component scope through to the request.
                transformer_window_component: Some(TRANSFORMER_WINDOW_COMPONENT),
                ..decode
            })
        );
    }

    /// SC-15805: "cumulative" is a DEFEASIBLE default owned by the contract, not a fact derived from
    /// the ladder's numeric order. Pin that at this provider's selection -> controls mapping with a
    /// contract that declares a cheaper rung unavailable: a deeper selection must leave that rung's
    /// lever off, and must not ship the parameters that lever would have been driven with.
    ///
    /// Every other test here uses the production contract, where rungs 1-3 are always `Implemented`
    /// and the contract-driven and cost-order forms agree exactly — so without this test, reverting
    /// `z_image_generation_memory` to a `match` over the ladder is invisible.
    #[test]
    fn a_rung_the_provider_does_not_implement_is_not_engaged_by_a_deeper_selection() {
        let tmp = tempfile::tempdir().unwrap();
        let mut contract = contract(&tmp);
        for capability in &mut contract.strategies {
            if capability.strategy == MemoryStrategy::BoundedDecode {
                capability.support = MemoryStrategySupport::Missing;
                capability.parameters = MemoryParameterRanges::default();
            }
        }

        let memory = z_image_generation_memory(
            &contract,
            &selection(MemoryStrategy::BoundedTransformerResidency),
        )
        .expect("an optimized rung maps to a request-scoped control set");

        assert!(
            !memory.tile_vae_decode,
            "rung 2 is declared Missing, so a rung-4 selection must not tile the decode: the cost \
             ordering is not a dependency"
        );
        assert_eq!(
            (memory.decode_tile_edge, memory.decode_overlap),
            (None, None),
            "a lever that is off must not ship the parameters it would have executed"
        );
        // ...while the rungs the provider DOES declare stay on, so this is not a vacuous all-false.
        assert!(memory.chunk_attention);
        assert!(memory.stream_transformer_blocks);
        assert_eq!(
            memory.transformer_window_size,
            Some(TRANSFORMER_WINDOW_SIZE)
        );
    }

    /// The executable half of the ladder: each request knob is read by exactly the pipeline seam that
    /// owns it, and a lower rung does not turn a higher one on.
    #[test]
    fn the_request_level_knobs_are_read_by_their_own_seams() {
        let tmp = tempfile::tempdir().unwrap();
        let plain = GenerationRequest {
            prompt: "a fox".to_owned(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        assert_eq!(
            crate::pipeline::attention_budget(&plain),
            AttentionBudget::UNBOUNDED
        );
        assert_eq!(crate::pipeline::block_window_size(&plain), None);
        assert_eq!(
            crate::pipeline::decode_tile_geometry(&plain),
            (DECODE_TILE_EDGE, DECODE_OVERLAP)
        );

        let full = GenerationRequest {
            memory: z_image_generation_memory(
                &contract(&tmp),
                &selection(MemoryStrategy::BoundedTransformerResidency),
            ),
            ..plain.clone()
        };
        assert_eq!(
            crate::pipeline::attention_budget(&full),
            AttentionBudget::CONSTRAINED
        );
        assert_eq!(
            crate::pipeline::block_window_size(&full),
            Some(TRANSFORMER_WINDOW_SIZE as usize)
        );

        // A rung-2 selection at a non-default edge executes THAT edge.
        let mut sel = selection(MemoryStrategy::BoundedDecode);
        // A published non-default candidate — 640, not one of the measured-and-rejected sub-512 edges.
        sel.parameters.decode_tile_edge = Some(640);
        let tiled = GenerationRequest {
            memory: z_image_generation_memory(&contract(&tmp), &sel),
            ..plain.clone()
        };
        assert_eq!(
            crate::pipeline::decode_tile_geometry(&tiled),
            (640, DECODE_OVERLAP)
        );
        // ...and rung 2 does not turn rung 3 or 4 on.
        assert_eq!(
            crate::pipeline::attention_budget(&tiled),
            AttentionBudget::UNBOUNDED
        );
        assert_eq!(crate::pipeline::block_window_size(&tiled), None);

        // Staged-only selects nothing below it.
        let staged = GenerationRequest {
            memory: Some(GenerationMemory::default()),
            ..plain
        };
        assert_eq!(
            crate::pipeline::attention_budget(&staged),
            AttentionBudget::UNBOUNDED
        );
        assert_eq!(crate::pipeline::block_window_size(&staged), None);
    }

    /// Rung 4 has no request-scoped lever on an eager generator. That is an error rather than a
    /// silent downgrade because a window over an already-materialized trunk would poison evidence.
    #[test]
    fn an_eager_load_refuses_rung_four_instead_of_degrading() {
        let tmp = tempfile::tempdir().unwrap();
        let req = GenerationRequest {
            prompt: "a fox".to_owned(),
            memory: z_image_generation_memory(
                &contract(&tmp),
                &selection(MemoryStrategy::BoundedTransformerResidency),
            ),
            ..Default::default()
        };
        let err = crate::pipeline::resolve_block_window(&req, false, "z_image_turbo")
            .unwrap_err()
            .to_string();
        assert!(err.contains("DeferredMaterialization"), "{err}");
        assert_eq!(
            crate::pipeline::resolve_block_window(&req, true, "z_image_turbo").unwrap(),
            Some(TRANSFORMER_WINDOW_SIZE as usize)
        );
        let full_width = crate::transformer::ZImageTransformerConfig::turbo().n_layers;
        let deferred_plain = GenerationRequest {
            prompt: "a fox".to_owned(),
            ..Default::default()
        };
        assert_eq!(
            crate::pipeline::resolve_block_window(&deferred_plain, true, "z_image_turbo").unwrap(),
            Some(full_width),
            "a deferred load must not materialize the lazy resident stack on an unscoped request"
        );
        let deferred_text_encoder = GenerationRequest {
            prompt: "a fox".to_owned(),
            memory: Some(GenerationMemory {
                stream_transformer_blocks: true,
                transformer_window_size: Some(1),
                transformer_window_component: Some(TransformerComponent::TextEncoder),
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            crate::pipeline::resolve_block_window(&deferred_text_encoder, true, "z_image_turbo")
                .unwrap(),
            Some(full_width),
            "excluding the DiT from rung 4 must preserve its deferred materialization shape"
        );
        // Every lower rung is unaffected by the residency policy.
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
        ] {
            let req = GenerationRequest {
                prompt: "a fox".to_owned(),
                memory: z_image_generation_memory(&contract(&tmp), &selection(strategy)),
                ..Default::default()
            };
            assert_eq!(
                crate::pipeline::resolve_block_window(&req, false, "z_image_turbo").unwrap(),
                None
            );
        }
    }

    /// SC-15510's PiD reconciliation. SC-15615 had to refuse rungs 2+ outright on the PiD route; now
    /// they are admissible, with the PiD student's OWN candidate domain — and a native-VAE geometry is
    /// rejected there rather than silently re-planned, which is the property that makes the rung
    /// honest on both routes.
    #[test]
    fn the_pid_route_admits_bounded_decode_on_its_own_candidate_domain() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract(&tmp);
        let pid_edges = pid_decode_tile_edges();
        assert!(!pid_edges.is_empty());
        assert!(
            pid_edges.iter().all(|e| !DECODE_TILE_EDGES.contains(e)),
            "the two routes' domains must not overlap, or a selection would be ambiguous"
        );

        for strategy in [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            let ctx = context_for(strategy, true);
            assert!(
                matches!(
                    safety_check(&contract, ctx.selection.tier, &ctx),
                    MemorySafetyDecision::Accept
                ),
                "{strategy:?} must now be admissible on the PiD route"
            );
            let mut scope =
                begin_request(crate::model::MODEL_ID, &contract, ctx.selection.tier, &ctx)
                    .unwrap()
                    .unwrap();
            scope
                .configure_decode(pid_edges[0], PID_DECODE_OVERLAP, ctx.geometry)
                .unwrap();
            // A native-VAE geometry is refused on the PiD route.
            assert!(scope
                .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, ctx.geometry)
                .is_err());
            scope.finish(MemoryRunOutcome::Complete).unwrap();
        }

        // The catalog-level registry conformance walk owns the two cross-route safety-check probes
        // for every registered provider. Keeping a second direct copy here would make the walk's
        // mutation proof non-discriminating. This provider test owns the complementary execution
        // seam above: a matching route is admitted, and the request scope still refuses the other
        // route's geometry instead of re-planning it.

        // Rungs 0-1 stay available on both routes (they own no decode parameters).
        for strategy in [MemoryStrategy::Resident, MemoryStrategy::StagedResidency] {
            assert!(matches!(
                {
                    let ctx = context_for(strategy, true);
                    safety_check(&contract, ctx.selection.tier, &ctx)
                },
                MemorySafetyDecision::Accept
            ));
        }
    }

    /// `z_image_edit` resolves to this provider in edit mode (see the module header), so the contract
    /// must admit `MemoryMode::Edit` across the whole ladder. Nothing else in this crate mentions
    /// that id — the alias itself is SceneWorks-side and pinned there — but if the edit *surface* were
    /// inadmissible here, the alias would resolve to a provider that refuses the mode it was aliased
    /// for, and the catalog entry would be unservable with no local test to say so.
    #[test]
    fn the_edit_surface_z_image_edit_resolves_to_is_admissible() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract(&tmp);
        for strategy in MemoryStrategy::ALL {
            let mut ctx = context(strategy);
            ctx.mode = MemoryMode::Edit;
            ctx.has_reference = true;
            ctx.geometry.reference_count = 1;
            assert!(
                matches!(
                    safety_check(&contract, ctx.selection.tier, &ctx),
                    MemorySafetyDecision::Accept
                ),
                "{strategy:?} must be admissible in edit mode"
            );
            let mut scope =
                begin_request(crate::model::MODEL_ID, &contract, ctx.selection.tier, &ctx)
                    .unwrap()
                    .expect("edit mode opens a scope");
            scope.finish(MemoryRunOutcome::Complete).unwrap();
        }
    }

    /// The scope's decode parameters are route-specific, so applying a PiD-admitted scope to a
    /// non-PiD request (or vice versa) must be refused the same way geometry drift is. Without this,
    /// a PiD edge (2048 px of super-resolved output) lands in a NATIVE decode, where at 1024² it
    /// collapses to a single tile — an effectively untiled decode while the evidence records a
    /// bounded one.
    #[test]
    fn the_scope_rejects_route_drift_the_way_it_rejects_geometry_drift() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract(&tmp);
        for admitted_pid in [false, true] {
            let ctx = context_for(MemoryStrategy::BoundedDecode, admitted_pid);
            let mut scope =
                begin_request(crate::model::MODEL_ID, &contract, ctx.selection.tier, &ctx)
                    .unwrap()
                    .unwrap();
            let mut request = GenerationRequest {
                prompt: "a fox".to_owned(),
                width: 1024,
                height: 1024,
                count: 1,
                use_pid: admitted_pid,
                ..Default::default()
            };
            // The admitted route configures fine...
            scope.configure_request(&mut request).unwrap();
            // ...and the other one is refused.
            request.use_pid = !admitted_pid;
            let err = scope
                .configure_request(&mut request)
                .unwrap_err()
                .to_string();
            assert!(err.contains("use_pid"), "{err}");
            scope.finish(MemoryRunOutcome::Complete).unwrap();
        }
    }

    #[test]
    fn a_stale_calibration_fingerprint_never_admits_an_optimized_fit() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract(&tmp);
        let mut ctx = context(MemoryStrategy::BoundedDecode);
        ctx.calibration_fingerprint = "z-image-mlx-something-older".to_owned();
        match safety_check(&contract, ctx.selection.tier, &ctx) {
            MemorySafetyDecision::Reject { reason } => {
                assert!(
                    reason.contains("calibration handshake mismatch"),
                    "{reason}"
                )
            }
            other => panic!("stale fingerprint must be rejected, got {other:?}"),
        }
        assert!(
            begin_request(crate::model::MODEL_ID, &contract, ctx.selection.tier, &ctx).is_err()
        );

        let mut ctx = context(MemoryStrategy::BoundedDecode);
        ctx.calibration_abi = MEMORY_CALIBRATION_ABI + 1;
        assert!(matches!(
            safety_check(&contract, ctx.selection.tier, &ctx),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn a_selection_for_a_different_numeric_tier_is_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract(&tmp);
        let mut ctx = context(MemoryStrategy::BoundedDecode);
        let loaded_tier = ctx.selection.tier;
        ctx.selection.tier.quant = Some(Quant::Q8);
        assert!(matches!(
            safety_check(&contract, loaded_tier, &ctx),
            MemorySafetyDecision::Reject { reason }
                if reason.contains("does not match loaded tier")
        ));
    }

    #[test]
    fn an_over_budget_prediction_is_rejected_before_any_work() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract(&tmp);
        let mut ctx = context(MemoryStrategy::BoundedDecode);
        ctx.predicted_peak_bytes = ctx.budget.effective_bytes() + 1;
        match safety_check(&contract, ctx.selection.tier, &ctx) {
            MemorySafetyDecision::Reject { reason } => {
                assert!(reason.contains("exceeds effective budget"), "{reason}")
            }
            other => panic!("over-budget must be rejected, got {other:?}"),
        }
        let mut ctx = context(MemoryStrategy::BoundedDecode);
        ctx.predicted_peak_bytes = ctx.budget.effective_bytes();
        assert!(matches!(
            safety_check(&contract, ctx.selection.tier, &ctx),
            MemorySafetyDecision::Accept
        ));
    }

    #[test]
    fn the_scope_overwrites_warm_request_state_and_finishes_once() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract(&tmp);
        let ctx = context(MemoryStrategy::BoundedDecode);
        let mut scope = begin_request(crate::model::MODEL_ID, &contract, ctx.selection.tier, &ctx)
            .unwrap()
            .expect("an accepted context opens a scope");

        // A warm request carrying a DEEPER prior rung must be overwritten, not merged.
        let mut request = GenerationRequest {
            prompt: "a fox".to_owned(),
            width: 1024,
            height: 1024,
            count: 1,
            memory: Some(GenerationMemory {
                chunk_attention: true,
                stream_transformer_blocks: true,
                transformer_window_size: Some(1),
                decode_tile_edge: Some(256),
                ..Default::default()
            }),
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        assert_eq!(
            request.memory,
            Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(DECODE_TILE_EDGE),
                decode_overlap: Some(DECODE_OVERLAP),
                ..Default::default()
            })
        );
        assert_eq!(
            crate::pipeline::attention_budget(&request),
            AttentionBudget::UNBOUNDED,
            "a warm request's stale chunk_attention must not survive re-selection"
        );
        assert_eq!(
            crate::pipeline::block_window_size(&request),
            None,
            "a warm request's stale transformer window must not survive re-selection"
        );
        assert_eq!(
            crate::pipeline::decode_tile_geometry(&request),
            (DECODE_TILE_EDGE, DECODE_OVERLAP),
            "a warm request's stale tile edge must not survive re-selection"
        );

        // Geometry drift from the admitted envelope is rejected.
        request.width = 1280;
        assert!(scope.configure_request(&mut request).is_err());
        request.width = 1024;
        request.count = 2;
        assert!(scope.configure_request(&mut request).is_err());
        request.count = 0;
        assert!(scope.configure_request(&mut request).is_err());
        request.count = 1;

        scope.finish(MemoryRunOutcome::Complete).unwrap();
        assert!(scope.finish(MemoryRunOutcome::Complete).is_err());
        assert!(scope.configure_request(&mut request).is_err());
    }

    #[test]
    fn the_scope_accepts_only_its_declared_parameters() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract(&tmp);
        let ctx = context(MemoryStrategy::BoundedTransformerResidency);
        let mut scope = begin_request(crate::model::MODEL_ID, &contract, ctx.selection.tier, &ctx)
            .unwrap()
            .unwrap();
        scope
            .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, ctx.geometry)
            .unwrap();
        for geometry in [
            MemoryGeometry {
                width: ctx.geometry.width / 2,
                ..ctx.geometry
            },
            MemoryGeometry {
                height: ctx.geometry.height / 2,
                ..ctx.geometry
            },
            MemoryGeometry {
                frames: ctx.geometry.frames + 1,
                ..ctx.geometry
            },
            MemoryGeometry {
                reference_count: 1,
                ..ctx.geometry
            },
            MemoryGeometry {
                batch: 0,
                ..ctx.geometry
            },
            MemoryGeometry {
                batch: ctx.geometry.batch + 1,
                ..ctx.geometry
            },
        ] {
            assert!(scope
                .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, geometry)
                .is_err());
        }
        // Off-ladder edges and the wrong overlap are refused.
        assert!(scope
            .configure_decode(500, DECODE_OVERLAP, ctx.geometry)
            .is_err());
        assert!(scope
            .configure_decode(DECODE_TILE_EDGE, 128, ctx.geometry)
            .is_err());
        scope.configure_attention(ATTENTION_CHUNK_SIZE).unwrap();
        assert!(scope.configure_attention(ATTENTION_CHUNK_SIZE / 2).is_err());

        // The transformer-window hook validates against the ADMITTED window, not any pair: full
        // windows at every stride, and the ragged tail at the end of the 30-block stack.
        let w = TRANSFORMER_WINDOW_SIZE;
        let n = crate::transformer::ZImageTransformerConfig::turbo().n_layers as u32;
        let mut first = 0;
        while first < n {
            let expected = w.min(n - first);
            scope
                .materialize_transformer_window(first, expected)
                .unwrap();
            assert!(
                scope
                    .materialize_transformer_window(first, expected + 1)
                    .is_err(),
                "a wider window than admitted must be refused at block {first}"
            );
            first += w;
        }
        assert!(scope.materialize_transformer_window(n, w).is_err());
        assert!(scope.materialize_transformer_window(0, 0).is_err());
        scope.finish(MemoryRunOutcome::Complete).unwrap();

        // Below rung 4 there is no window at all, so the hook refuses everything.
        let ctx = context(MemoryStrategy::BoundedAttention);
        let mut scope = begin_request(crate::model::MODEL_ID, &contract, ctx.selection.tier, &ctx)
            .unwrap()
            .unwrap();
        assert!(scope.materialize_transformer_window(0, 1).is_err());
        scope.finish(MemoryRunOutcome::Complete).unwrap();
    }

    #[test]
    fn registered_behavior_enforces_decode_geometry_batch_prefixes_and_terminal_request_state() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = spec(&tmp).with_quant(Quant::Q4);
        let contract = memory_strategy_contract(crate::model::MODEL_ID, &spec).unwrap();
        let mut ctx = context(MemoryStrategy::BoundedDecode);
        ctx.geometry.batch = 3;
        let mut scope =
            (crate::model::MEMORY_BEHAVIOR_REGISTRATION.begin_request)(&spec, &contract, &ctx)
                .unwrap()
                .expect("registered behavior must adopt the shared MLX scope");
        scope
            .configure_decode(
                DECODE_TILE_EDGE,
                DECODE_OVERLAP,
                MemoryGeometry {
                    batch: 1,
                    ..ctx.geometry
                },
            )
            .unwrap();
        assert!(scope
            .configure_decode(
                DECODE_TILE_EDGE,
                DECODE_OVERLAP,
                MemoryGeometry {
                    width: ctx.geometry.width / 2,
                    ..ctx.geometry
                },
            )
            .is_err());
        assert!(scope
            .configure_decode(
                DECODE_TILE_EDGE,
                DECODE_OVERLAP,
                MemoryGeometry {
                    batch: 4,
                    ..ctx.geometry
                },
            )
            .is_err());
        scope.finish(MemoryRunOutcome::Complete).unwrap();
        let mut request = GenerationRequest {
            prompt: "after-finish".to_owned(),
            width: ctx.geometry.width,
            height: ctx.geometry.height,
            count: 1,
            ..Default::default()
        };
        assert!(scope.configure_request(&mut request).is_err());
    }

    #[test]
    fn a_canceled_or_errored_run_still_releases() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract(&tmp);
        for outcome in [
            MemoryRunOutcome::Canceled,
            MemoryRunOutcome::Error {
                message: "boom".to_owned(),
            },
        ] {
            let ctx = context(MemoryStrategy::BoundedDecode);
            let mut scope =
                begin_request(crate::model::MODEL_ID, &contract, ctx.selection.tier, &ctx)
                    .unwrap()
                    .unwrap();
            scope.finish(outcome).unwrap();
        }
        let ctx = context(MemoryStrategy::BoundedDecode);
        drop(begin_request(crate::model::MODEL_ID, &contract, ctx.selection.tier, &ctx).unwrap());
    }

    #[test]
    fn every_registered_variant_declares_the_same_conformant_contract() {
        let tmp = tempfile::tempdir().unwrap();
        for id in [
            crate::model::MODEL_ID,
            crate::model_base::MODEL_ID,
            crate::model_control::MODEL_ID,
            crate::model_base_control::MODEL_ID,
        ] {
            let contract = memory_strategy_contract(id, &spec(&tmp)).unwrap();
            assert_eq!(contract.provider_id, id);
            assert_eq!(contract.conformance_errors(), Vec::<String>::new(), "{id}");
            gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
        }
    }

    #[test]
    fn declared_parameters_are_the_defaults_and_all_live_in_the_published_ladders() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = contract(&tmp);
        let params = declared_parameters();
        let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
        assert!(decode
            .parameters
            .decode_tile_edges
            .contains(&params.decode_tile_edge.unwrap()));
        assert!(decode
            .parameters
            .decode_overlaps
            .contains(&params.decode_overlap.unwrap()));
        assert_eq!(params.decode_tile_edge, Some(DECODE_TILE_EDGE));
        assert_eq!(params.decode_overlap, Some(DECODE_OVERLAP));
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .parameters
                .attention_chunk_sizes,
            vec![params.attention_chunk_size.unwrap()]
        );
        assert!(contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap()
            .parameters
            .transformer_window_sizes
            .contains(&params.transformer_window_size.unwrap()));
    }
}
