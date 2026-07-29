//! Z-Image MLX adoption of the shared image-memory contract (SC-15449) — the **complete** ladder:
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
//! Consequently `ImageMemoryMode::Edit` must be admissible on this contract — see
//! `the_edit_surface_z_image_edit_resolves_to_is_admissible`.
//!
//! ## Declared rungs
//!
//! | Rung | Support | Executable seam |
//! |---|---|---|
//! | 0 Resident | Implemented | `Residency::resident` — encoder + DiT + VAE held warm |
//! | 1 Staged residency | Implemented (load-time, see below) | `Residency::run_staged` (sc-10839 / sc-13571): encode → drop encoder → denoise → **drop DiT** → decode |
//! | 2 Bounded decode | Implemented | `Vae::decode_tiled` over the [`DECODE_TILE_EDGES`] ladder, or the PiD student over [`pid_decode_tile_edges`] (`pipeline::decode_tiling`) |
//! | 3 Bounded attention | Implemented | [`mlx_gen::attention::sdpa_budgeted_bhsd`] threaded through every DiT attention (SC-15615) |
//! | 4 Bounded transformer residency | Implemented (snapshot loads) | [`mlx_gen::block_residency::run_windowed`] over the 30 unified blocks, and the 15 ControlNet blocks (SC-15754) |
//!
//! Rung 4 is `Missing` for a **single-file / ComfyUI** load and only that: streaming rebuilds blocks
//! from the snapshot per window, and an in-memory `Weights` has no re-openable source. The contract is
//! built per `LoadSpec`, so that is declared per load rather than averaged into a half-truth.
//!
//! **This is the MLX column, and it is not the Candle column.** As of SC-15754 the MLX lane carries
//! all five rungs; `candle-gen-z-image` carries 0-3 and has no rung 4. Neither the presence nor the
//! *magnitude* of a rung transfers between the two backends — rung 3 exists on both and is worth ~6×
//! more on Candle (see below), and rung 4 exists only here. Every per-backend saving in this file was
//! measured on this backend.
//!
//! **Rung 1 has no request-scoped lever.** Staging is gated on [`mlx_gen::Residency::is_sequential`],
//! which comes from the *load-time* [`OffloadPolicy`](mlx_gen::OffloadPolicy), so selecting
//! `StagedResidency` on a generator that was loaded `Resident` yields resident behaviour — the
//! selection is honoured only if the consumer also loaded the provider `Sequential`.
//! `z_image_generation_memory` maps rung 1 to an all-false [`GenerationMemory`] for exactly that
//! reason: there is nothing per-request to turn on. Krea's CUDA adoption has the same shape, so this
//! is a shared-contract gap (a load-time-vs-request-time seam) rather than a Z-Image one; it is
//! recorded here so no calibration reads a rung-1 cell as request-selectable.
//!
//! **Rung 4 has the same load-time dependency and resolves it the opposite way — it REJECTS.** A
//! window only bounds anything if the resident trunk is not also materialized, which is true under
//! `Sequential` and false on a warm `Resident` generator. Yielding resident behaviour there (rung 1's
//! answer) would write a rung-4 row into the calibration evidence for a run that saved nothing, so
//! `pipeline::resolve_block_window` returns a typed error instead. Same seam, different blast radius.
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
//! ## Ownership
//!
//! This file declares *structure and parameter domains only*. Measured coefficients, envelopes, and
//! per-tier peaks live in SceneWorks generated evidence keyed by
//! [`IMAGE_MEMORY_CALIBRATION_FINGERPRINT`]; the worker owns live-budget accounting and least-cost
//! selection. The scope below is defense in depth: it can reject a selection, never substitute one.

use mlx_gen::gen_core::{
    Error as CoreError, GenerationMemory, GenerationRequest, ImageMemoryAssetFacts,
    ImageMemoryBackendRealization, ImageMemoryCalibrationIdentity, ImageMemoryFormulaKind,
    ImageMemoryFormulaVariable, ImageMemoryGeometry, ImageMemoryLifecycleCapabilities,
    ImageMemoryParameterRanges, ImageMemoryPhase, ImageMemoryProviderContract,
    ImageMemoryRequestScope, ImageMemoryRunContext, ImageMemoryRunOutcome,
    ImageMemoryRuntimeSemantics, ImageMemorySafetyDecision, ImageMemorySelection,
    ImageMemoryStrategy, ImageMemoryStrategyCapability, ImageMemoryStrategyParameters,
    ImageMemoryStrategySupport, LoadSpec, PerComponentBytes, Result as CoreResult,
};

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
/// `image-memory-mlx-adapter` probe sweep, and it was measured on the **Qwen** VAE — a different
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
/// its legal tiles live in an entirely different range — [`mlx_gen_pid::budget`]'s `TILE_ALIGN`
/// multiples between `MIN_TILE_EDGE` and the Metal watchdog-safe `WATCHDOG_SAFE_EDGE`. Derived from
/// those constants rather than re-spelled, so the two cannot drift.
///
/// SC-15615 had to *refuse* rung 2 (and cumulatively rung 3) on the PiD route because the student
/// planned its own tiling and never read the contract's parameters — admitting a selection would have
/// executed a different strategy than the selector chose. SC-15510 reconciles them instead: the
/// student now accepts an explicit plan, and this is the value it accepts.
///
/// **One candidate, for the same reason the native ladder has three.** `mlx_gen_pid::budget` will
/// *validate* seven aligned edges between `MIN_TILE_EDGE` and `WATCHDOG_SAFE_EDGE`, and publishing all
/// seven was the first draft. But the native domain earned its three by a real-weight sweep that
/// rejected four for seaming — and publishing an unmeasured seven beside it would be exactly the
/// double standard this story spent its evidence budget arguing against. Only
/// [`PREFERRED_TILE_EDGE`](mlx_gen_pid::budget::PREFERRED_TILE_EDGE) has seam evidence: sc-10087's A/B
/// measured 2048 px tiles at 256 px overlap as showing no measurable seam on this student, which is
/// why the planner itself prefers it.
///
/// Widening this needs a per-student sweep against the untiled super-resolved decode, the way
/// `every_declared_decode_candidate_runs_and_preserves_the_image` does for the native VAE. Until then
/// one measured point is the honest domain — and a one-element domain is fully covered by a
/// single-point pass, which is what SC-15508 actually asks.
pub fn pid_decode_tile_edges() -> Vec<u32> {
    vec![mlx_gen_pid::budget::PREFERRED_TILE_EDGE as u32]
}

/// The PiD route's feather overlap — the sc-10087 A/B's 256 px, which is `mlx_gen_pid`'s own default.
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

/// The one bounded-attention parameter this provider accepts: the shared
/// [`mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET`] (64 Mi score elements per attention call),
/// the exact knob the Candle/CUDA Z-Image rung 3 measured in SC-15256 — reused verbatim so a
/// cross-backend comparison of the same rung is meaningful. It is the only candidate advertised in
/// `attention_chunk_sizes`, and the request scope re-validates it.
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;

/// Calibration content fingerprint. It must change whenever quantization floors, tensor layout, or
/// execution structure change in a way that invalidates measurements taken against this provider.
///
/// `-v2` retires the SC-15615 rung-3 declaration: rung 4 changes *when the trunk weights exist*
/// during a denoise step, and the rung-2 decode domain is no longer a single point — both are
/// execution-structure changes, so evidence taken against `-v1` must not be readable as covering this.
pub const IMAGE_MEMORY_CALIBRATION_FINGERPRINT: &str =
    "z-image-mlx-staged-tiled-decode-bounded-attention-block-window-v2";

/// The decode candidates a route will accept, and the default it falls back to.
///
/// `use_pid` swaps the native VAE for a super-resolving student whose legal tiles live in a different
/// range of the same unit (see [`pid_decode_tile_edges`]), so the domain is per-route. The contract's
/// static `decode_tile_edges` publishes the **union** — it is the provider's whole production domain,
/// and the static validator is not route-aware — while `safety_check` and the request scope, which
/// both see `use_pid`, enforce the route's own subset. Defense in depth in the direction the contract
/// allows: narrower at admission, never broader.
fn decode_domain(use_pid: bool) -> (Vec<u32>, u32) {
    if use_pid {
        (pid_decode_tile_edges(), PID_DECODE_OVERLAP)
    } else {
        (DECODE_TILE_EDGES.to_vec(), DECODE_OVERLAP)
    }
}

/// The union of both routes' tile edges, descending. Sorted and deduped because the shared
/// conformance check rejects duplicate candidates.
fn all_decode_tile_edges() -> Vec<u32> {
    let mut edges: Vec<u32> = DECODE_TILE_EDGES
        .iter()
        .copied()
        .chain(pid_decode_tile_edges())
        .collect();
    edges.sort_unstable_by(|a, b| b.cmp(a));
    edges.dedup();
    edges
}

/// The union of both routes' overlaps, descending.
fn all_decode_overlaps() -> Vec<u32> {
    let mut overlaps = vec![PID_DECODE_OVERLAP, DECODE_OVERLAP];
    overlaps.sort_unstable_by(|a, b| b.cmp(a));
    overlaps.dedup();
    overlaps
}

/// Build the Z-Image MLX provider contract for `provider_id`.
///
/// `spec` supplies the load-exact asset facts: the component `.safetensors` sums under the resolved
/// snapshot root, which is what the MLX loader actually materializes (the tier subdirectory is already
/// the spec root for a pre-quantized turnkey). A single-file (ComfyUI) source has no component tree,
/// so its asset facts stay zero rather than reporting a fabricated split.
pub fn image_memory_contract(provider_id: &str, spec: &LoadSpec) -> ImageMemoryProviderContract {
    // SC-15754: rung 4 is available only when this SPECIFIC load can execute it, and two load-time
    // facts decide that. Declaring it here rather than failing at generate time is what keeps the
    // selector from choosing a strategy the loaded generator cannot run — the contract is built per
    // `LoadSpec`, so it can say so.
    //
    // 1. It rebuilds trunk blocks from the snapshot per window, so it needs a **re-openable source**.
    //    A `WeightsSource::Dir` (every registry load) has one; a single-file / in-place ComfyUI load
    //    does not.
    // 2. It only bounds anything under `OffloadPolicy::Sequential`. Under `Resident` the trunk is
    //    already materialized and a window would *add* a copy on top of it — strictly more memory and
    //    strictly slower. `pipeline::resolve_block_window` refuses that at generate time as defense in
    //    depth, but a run-time error is a poor failure mode for a memory optimization: the selector
    //    would have picked a working lower rung had it known. So the contract says `Missing` and the
    //    selector never reaches the refusal.
    //
    // This is the same load-time-vs-request-time seam rung 1 documents, resolved at the layer that can
    // actually express it.
    let streamable = matches!(spec.weights, mlx_gen::WeightsSource::Dir(_))
        && matches!(spec.offload_policy, mlx_gen::OffloadPolicy::Sequential);
    ImageMemoryProviderContract {
        provider_id: provider_id.to_owned(),
        backend: ImageMemoryBackendRealization::MlxMetal {
            // Unified memory: the wired-residency budget is what the staged phases release, weights
            // are mmap-backed, and MLX's lazy graph needs explicit `eval` before a phase drop frees
            // anything (`Residency::run_staged` owns that discipline). No host↔device transfer.
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
        strategies: ImageMemoryStrategy::ALL
            .into_iter()
            .map(|strategy| ImageMemoryStrategyCapability {
                strategy,
                support: match strategy {
                    ImageMemoryStrategy::BoundedTransformerResidency if !streamable => {
                        ImageMemoryStrategySupport::Missing
                    }
                    _ => ImageMemoryStrategySupport::Implemented,
                },
                parameters: match strategy {
                    ImageMemoryStrategy::BoundedDecode => ImageMemoryParameterRanges {
                        decode_tile_edges: all_decode_tile_edges(),
                        decode_overlaps: all_decode_overlaps(),
                        ..Default::default()
                    },
                    ImageMemoryStrategy::BoundedAttention => ImageMemoryParameterRanges {
                        attention_chunk_sizes: vec![ATTENTION_CHUNK_SIZE],
                        ..Default::default()
                    },
                    ImageMemoryStrategy::BoundedTransformerResidency if streamable => {
                        ImageMemoryParameterRanges {
                            transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                            ..Default::default()
                        }
                    }
                    _ => ImageMemoryParameterRanges::default(),
                },
            })
            .collect(),
        lifecycle: ImageMemoryLifecycleCapabilities {
            phases: vec![
                ImageMemoryPhase::Conditioning,
                ImageMemoryPhase::Denoise,
                ImageMemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: streamable,
        },
        formula: ImageMemoryFormulaKind::PhaseEnvelope {
            phases: vec![
                ImageMemoryPhase::Conditioning,
                ImageMemoryPhase::Denoise,
                ImageMemoryPhase::Decode,
            ],
            variables: vec![
                ImageMemoryFormulaVariable::AssetBytes,
                ImageMemoryFormulaVariable::PixelCount,
                ImageMemoryFormulaVariable::BatchCount,
                ImageMemoryFormulaVariable::ConditioningTokenCount,
                ImageMemoryFormulaVariable::DecodeTileArea,
                ImageMemoryFormulaVariable::AttentionChunkSize,
                // SC-15754: transformer weight residency is now a *variable* of the peak, not a
                // constant folded into `AssetBytes` — at window 1 the trunk contributes one block
                // instead of thirty.
                ImageMemoryFormulaVariable::TransformerWindowSize,
            ],
        },
        calibration: Some(ImageMemoryCalibrationIdentity::new(
            IMAGE_MEMORY_CALIBRATION_FINGERPRINT,
        )),
        asset_facts: asset_facts(spec),
        runtime: ImageMemoryRuntimeSemantics::default(),
    }
}

/// Component `.safetensors` sums for the spec's snapshot root. A [`WeightsSource::File`] source has
/// no component tree, so every field stays `0` (the truthful "unknown", not a guess).
fn asset_facts(spec: &LoadSpec) -> ImageMemoryAssetFacts {
    let Ok(components) =
        PerComponentBytes::from_spec_subdirs(spec, &["text_encoder"], &["transformer"], &["vae"])
    else {
        return ImageMemoryAssetFacts::default();
    };
    ImageMemoryAssetFacts {
        base_bytes: components
            .text_encoder
            .saturating_add(components.dit)
            .saturating_add(components.vae),
        conditioning_bytes: components.text_encoder,
        transformer_bytes: components.dit,
        decoder_bytes: components.vae,
        // A PiD overlay is a separate student checkpoint outside the snapshot component tree, and it
        // is only resident when a request sets `use_pid`. Reporting the base snapshot's bytes here
        // would be wrong, and guessing the overlay's would be worse, so it stays 0 until per-model
        // calibration measures it.
        overlay_bytes: 0,
    }
}

/// The shared ladder → this provider's existing per-request execution controls.
///
/// The ladder is **cumulative**: every rung above `StagedResidency` also carries the levers below it,
/// so a rung-3 selection tiles the decode as well. `Resident` returns `None`, which is the historical
/// fast path (`GenerationRequest::memory` untouched).
pub(crate) fn z_image_generation_memory(
    selection: &ImageMemorySelection,
) -> Option<GenerationMemory> {
    // SC-15510: the selected *parameters* travel with the levers. `validate_selection` has already
    // established that a rung carries exactly the parameters it owns and no more, so copying them
    // wholesale cannot smuggle a decode edge into a staged-only selection: the fields below are only
    // read by a pipeline whose matching boolean is set.
    let parameters = selection.parameters;
    let decode = GenerationMemory {
        tile_vae_decode: true,
        decode_tile_edge: parameters.decode_tile_edge,
        decode_overlap: parameters.decode_overlap,
        ..Default::default()
    };
    match selection.strategy {
        ImageMemoryStrategy::Resident => None,
        ImageMemoryStrategy::StagedResidency => Some(GenerationMemory::default()),
        ImageMemoryStrategy::BoundedDecode => Some(decode),
        ImageMemoryStrategy::BoundedAttention => Some(GenerationMemory {
            chunk_attention: true,
            ..decode
        }),
        ImageMemoryStrategy::BoundedTransformerResidency => Some(GenerationMemory {
            chunk_attention: true,
            stream_transformer_blocks: true,
            transformer_window_size: parameters.transformer_window_size,
            ..decode
        }),
    }
}

/// Request-scoped lifecycle state for one admitted Z-Image generation.
///
/// Holds no MLX arrays: its whole job is to translate the shared selection into
/// [`GenerationRequest::memory`], reject parameters this provider does not implement, and guarantee
/// the terminal synchronize-and-release on success, cancellation, **and** error.
pub(crate) struct ZImageImageMemoryScope {
    pub(crate) provider_id: &'static str,
    pub(crate) geometry: ImageMemoryGeometry,
    pub(crate) memory: Option<GenerationMemory>,
    /// Which decode this request runs, so `configure_decode` validates the route's own candidate
    /// subset rather than the published union (see [`decode_domain`]).
    pub(crate) use_pid: bool,
    /// The window a rung-4 selection admitted, so `materialize_transformer_window` can check the
    /// hook's `(first_block, block_count)` against the plan actually running instead of accepting any
    /// pair. `None` below rung 4.
    pub(crate) transformer_window: Option<u32>,
    pub(crate) finished: bool,
}

impl ZImageImageMemoryScope {
    fn ensure_active(&self) -> CoreResult<()> {
        if self.finished {
            Err(CoreError::Msg(format!(
                "{}: image-memory request scope is already finished",
                self.provider_id
            )))
        } else {
            Ok(())
        }
    }

    /// Terminal barrier + cache eviction, idempotent.
    ///
    /// MLX is lazy and its allocator retains freed buffers in a cache, so "the request is over" is
    /// only true after (a) a synchronous barrier on the default stream — which is what
    /// [`mlx_rs::Array::eval`] is — and (b) an explicit [`clear_cache`](mlx_rs::memory::clear_cache).
    /// Without both, a canceled or errored request can leave partially-resident buffers that poison
    /// the next request's budget. This runs on every exit path, including [`Drop`].
    fn synchronize_and_release(&mut self) -> CoreResult<()> {
        let barrier = mlx_rs::Array::from(0.0_f32);
        barrier.eval().map_err(mlx_gen::Error::from)?;
        drop(barrier);
        mlx_rs::memory::clear_cache();
        self.finished = true;
        Ok(())
    }
}

impl ImageMemoryRequestScope for ZImageImageMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> CoreResult<()> {
        self.ensure_active()?;
        // Route drift is as fatal as geometry drift, and for the same reason. The scope's decode
        // parameters are route-specific — a PiD-admitted scope carries a 1024-4096 px super-resolved
        // tile — so applying it to a non-PiD request would write a PiD edge into a NATIVE decode. At
        // 1024² a 4096 px tile collapses to a single tile, i.e. an effectively untiled decode while the
        // evidence records a bounded one. That is the exact "executed a different strategy than the
        // selector chose" failure the contract exists to prevent, so it is checked here beside the
        // geometry it is a sibling of.
        if request.use_pid != self.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{}: request use_pid={} does not match the admitted route (use_pid={}); the decode \
                 parameters are route-specific and cannot be carried across",
                self.provider_id, request.use_pid, self.use_pid
            )));
        }
        if request.width != self.geometry.width
            || request.height != self.geometry.height
            || request.count == 0
            || request.count > self.geometry.batch
        {
            return Err(CoreError::Unsupported(format!(
                "{}: request geometry {}x{} count {} does not match admitted {}x{} count {}",
                self.provider_id,
                request.width,
                request.height,
                request.count,
                self.geometry.width,
                self.geometry.height,
                self.geometry.batch
            )));
        }
        // The shared selection is authoritative and request-scoped: overwrite (never merge) whatever
        // a reused warm request carried, so a deeper prior rung cannot leak into this run.
        request.memory = self.memory;
        Ok(())
    }

    fn enter_phase(&mut self, _phase: ImageMemoryPhase) -> CoreResult<()> {
        // The phase boundaries themselves are owned by `Residency::run_staged`, which already
        // evaluates and drops between phases; the scope only has to stay live across them.
        self.ensure_active()
    }

    fn leave_phase(&mut self, _phase: ImageMemoryPhase) -> CoreResult<()> {
        self.ensure_active()
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        _geometry: ImageMemoryGeometry,
    ) -> CoreResult<()> {
        self.ensure_active()?;
        let (edges, expected_overlap) = decode_domain(self.use_pid);
        if !edges.contains(&tile_edge) {
            return Err(CoreError::Unsupported(format!(
                "{}: decode tile edge {tile_edge} is outside the {} route's candidates {edges:?}",
                self.provider_id,
                if self.use_pid { "PiD" } else { "native VAE" },
            )));
        }
        if overlap != expected_overlap {
            return Err(CoreError::Unsupported(format!(
                "{}: the {} decode overlap is {expected_overlap}, got {overlap}",
                self.provider_id,
                if self.use_pid { "PiD" } else { "native VAE" },
            )));
        }
        Ok(())
    }

    fn configure_attention(&mut self, chunk_size: u32) -> CoreResult<()> {
        self.ensure_active()?;
        if chunk_size == ATTENTION_CHUNK_SIZE {
            Ok(())
        } else {
            Err(CoreError::Unsupported(format!(
                "{}: attention chunk size is fixed at {ATTENTION_CHUNK_SIZE} score elements, got \
                 {chunk_size}",
                self.provider_id
            )))
        }
    }

    /// SC-15754. The window schedule itself is driven by
    /// [`mlx_gen::block_residency::run_windowed`] inside the DiT forward — this hook is the shared
    /// contract's *validation* seam, not a second driver: it answers "would the window you are about
    /// to run be the one this request was admitted for?".
    ///
    /// Accepting an arbitrary `(first_block, block_count)` here would let a harness record a sweep
    /// point that the provider never executed, which is the same class of false green as declaring a
    /// rung Implemented because its Candle twin exists.
    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> CoreResult<()> {
        self.ensure_active()?;
        let Some(window) = self.transformer_window else {
            return Err(CoreError::Unsupported(format!(
                "{}: this request did not select bounded transformer residency, so no transformer \
                 window is active",
                self.provider_id
            )));
        };
        let n_blocks = crate::transformer::ZImageTransformerConfig::turbo().n_layers as u32;
        if first_block >= n_blocks {
            return Err(CoreError::Unsupported(format!(
                "{}: transformer window starts at block {first_block}, past the {n_blocks}-block \
                 stack",
                self.provider_id
            )));
        }
        // Every window is `window` blocks except a ragged tail at the end of the stack.
        let expected = window.min(n_blocks - first_block);
        if block_count != expected {
            return Err(CoreError::Unsupported(format!(
                "{}: transformer window at block {first_block} is {block_count} blocks, but the \
                 admitted window size {window} makes it {expected}",
                self.provider_id
            )));
        }
        Ok(())
    }

    fn finish(&mut self, _outcome: ImageMemoryRunOutcome) -> CoreResult<()> {
        // Deliberately outcome-independent: cancellation and error need the barrier + eviction at
        // least as much as success does.
        self.ensure_active()?;
        self.synchronize_and_release()
    }
}

impl Drop for ZImageImageMemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.synchronize_and_release();
        }
    }
}

/// The provider safety check every Z-Image variant shares: the calibration handshake, then the shared
/// contract's own selection validation, then the budget. Defense in depth only — it can reject, it can
/// never swap in a different strategy or numeric tier.
pub(crate) fn safety_check(
    contract: &ImageMemoryProviderContract,
    context: &ImageMemoryRunContext,
) -> ImageMemorySafetyDecision {
    let Some(calibration) = contract.calibration.as_ref() else {
        return ImageMemorySafetyDecision::Reject {
            reason: format!("{}: no calibration identity declared", contract.provider_id),
        };
    };
    if context.calibration_abi != calibration.abi
        || context.calibration_fingerprint != calibration.fingerprint
    {
        return ImageMemorySafetyDecision::Reject {
            reason: format!(
                "{}: calibration handshake mismatch (admitted abi {} fingerprint {:?}, provider abi \
                 {} fingerprint {:?})",
                contract.provider_id,
                context.calibration_abi,
                context.calibration_fingerprint,
                calibration.abi,
                calibration.fingerprint
            ),
        };
    }
    if let Err(error) = contract.validate_selection(&context.selection) {
        return ImageMemorySafetyDecision::Reject {
            reason: error.to_string(),
        };
    }
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
    if context.selection.strategy >= ImageMemoryStrategy::BoundedDecode {
        let (edges, overlap) = decode_domain(context.use_pid);
        let route = if context.use_pid {
            "PiD overlay"
        } else {
            "native VAE"
        };
        match context.selection.parameters.decode_tile_edge {
            Some(edge) if edges.contains(&edge) => {}
            other => {
                return ImageMemorySafetyDecision::Reject {
                    reason: format!(
                        "{}: decode tile edge {other:?} is not a {route} candidate {edges:?}",
                        contract.provider_id
                    ),
                };
            }
        }
        if context.selection.parameters.decode_overlap != Some(overlap) {
            return ImageMemorySafetyDecision::Reject {
                reason: format!(
                    "{}: the {route} decode overlap is {overlap}, got {:?}",
                    contract.provider_id, context.selection.parameters.decode_overlap
                ),
            };
        }
    }
    if !context.budget.fits(context.predicted_peak_bytes) {
        return ImageMemorySafetyDecision::Reject {
            reason: format!(
                "{}: predicted peak {} exceeds effective budget {}",
                contract.provider_id,
                context.predicted_peak_bytes,
                context.budget.effective_bytes()
            ),
        };
    }
    ImageMemorySafetyDecision::Accept
}

/// Open a request scope after `safety_check` accepted `context`.
pub(crate) fn begin_request(
    provider_id: &'static str,
    contract: &ImageMemoryProviderContract,
    context: &ImageMemoryRunContext,
) -> CoreResult<Option<Box<dyn ImageMemoryRequestScope + 'static>>> {
    if let ImageMemorySafetyDecision::Reject { reason } = safety_check(contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    Ok(Some(Box::new(ZImageImageMemoryScope {
        provider_id,
        geometry: context.geometry,
        memory: z_image_generation_memory(&context.selection),
        use_pid: context.use_pid,
        transformer_window: (context.selection.strategy
            == ImageMemoryStrategy::BoundedTransformerResidency)
            .then_some(())
            .and(context.selection.parameters.transformer_window_size),
        finished: false,
    })))
}

/// The strategy parameters this provider accepts, for a caller that wants the whole domain in one
/// value (the conformance tests and the SceneWorks evidence writer both key off this).
pub fn declared_parameters() -> ImageMemoryStrategyParameters {
    ImageMemoryStrategyParameters {
        decode_tile_edge: Some(DECODE_TILE_EDGE),
        decode_overlap: Some(DECODE_OVERLAP),
        attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
        transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::attention::AttentionBudget;
    use mlx_gen::gen_core::WeightsSource;
    use mlx_gen::gen_core::{
        ImageMemoryBudget, ImageMemoryCacheState, ImageMemoryMode, ImageMemoryNumericTier,
        Precision, Quant, IMAGE_MEMORY_CALIBRATION_ABI,
    };

    /// The load rung 4 is available on: a re-openable snapshot dir, loaded `Sequential`.
    fn spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent-z-image-snapshot".into()))
            .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
    }

    fn contract() -> ImageMemoryProviderContract {
        image_memory_contract(crate::model::MODEL_ID, &spec())
    }

    /// A selection carrying exactly the parameters the rungs up to and including `strategy` own —
    /// no more, no less, which is what the shared validator requires. `use_pid` picks the route's
    /// decode domain, since the two do not overlap.
    fn selection_for(strategy: ImageMemoryStrategy, use_pid: bool) -> ImageMemorySelection {
        let (edges, overlap) = decode_domain(use_pid);
        let edge = if use_pid {
            // The largest PiD candidate, so the value is unambiguously from the PiD ladder.
            edges[0]
        } else {
            DECODE_TILE_EDGE
        };
        let decode = ImageMemoryStrategyParameters {
            decode_tile_edge: Some(edge),
            decode_overlap: Some(overlap),
            ..Default::default()
        };
        ImageMemorySelection {
            strategy,
            parameters: match strategy {
                ImageMemoryStrategy::Resident | ImageMemoryStrategy::StagedResidency => {
                    ImageMemoryStrategyParameters::default()
                }
                ImageMemoryStrategy::BoundedDecode => decode,
                ImageMemoryStrategy::BoundedAttention => ImageMemoryStrategyParameters {
                    attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                    ..decode
                },
                ImageMemoryStrategy::BoundedTransformerResidency => ImageMemoryStrategyParameters {
                    attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                    transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
                    ..decode
                },
            },
            tier: ImageMemoryNumericTier {
                precision: Precision::Bf16,
                quant: Some(Quant::Q4),
            },
        }
    }

    fn selection(strategy: ImageMemoryStrategy) -> ImageMemorySelection {
        selection_for(strategy, false)
    }

    fn context_for(strategy: ImageMemoryStrategy, use_pid: bool) -> ImageMemoryRunContext {
        ImageMemoryRunContext {
            selection: selection_for(strategy, use_pid),
            calibration_abi: IMAGE_MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: IMAGE_MEMORY_CALIBRATION_FINGERPRINT.to_owned(),
            mode: ImageMemoryMode::TextToImage,
            has_reference: false,
            use_pid,
            has_phases: true,
            geometry: ImageMemoryGeometry {
                width: 1024,
                height: 1024,
                batch: 1,
                frames: 1,
            },
            overlay: None,
            budget: ImageMemoryBudget {
                total_bytes: 8 * 1000 * 1000 * 1000,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 2 * 1000 * 1000 * 1000,
            },
            predicted_peak_bytes: 4 * 1000 * 1000 * 1000,
            cache_state: ImageMemoryCacheState::Cold,
            evidence_revision: "test".to_owned(),
        }
    }

    fn context(strategy: ImageMemoryStrategy) -> ImageMemoryRunContext {
        context_for(strategy, false)
    }

    #[test]
    fn contract_is_internally_conformant() {
        let contract = contract();
        assert_eq!(contract.conformance_errors(), Vec::<String>::new());
        gen_core_testkit::check_image_memory_contract(&contract).unwrap();
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            "z-image-mlx-staged-tiled-decode-bounded-attention-block-window-v2"
        );
    }

    /// The fingerprint must move when the execution structure does. SC-15754 changes *when the trunk
    /// weights exist* during a denoise step, which is precisely the "execution structure" the shared
    /// contract says invalidates prior measurements — so evidence taken against the rung-3 declaration
    /// must not be readable as covering this one.
    #[test]
    fn the_fingerprint_retired_the_rung_three_generation() {
        assert_ne!(
            IMAGE_MEMORY_CALIBRATION_FINGERPRINT,
            "z-image-mlx-staged-tiled-decode-bounded-attention-v1"
        );
        let contract = contract();
        let mut ctx = context(ImageMemoryStrategy::BoundedAttention);
        ctx.calibration_fingerprint =
            "z-image-mlx-staged-tiled-decode-bounded-attention-v1".to_owned();
        assert!(matches!(
            safety_check(&contract, &ctx),
            ImageMemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn every_rung_is_implemented_and_selectable_on_a_snapshot_load() {
        let contract = contract();
        for strategy in ImageMemoryStrategy::ALL {
            assert!(
                matches!(
                    contract.capability(strategy).map(|c| &c.support),
                    Some(ImageMemoryStrategySupport::Implemented)
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
        let contract = contract();
        let capability = contract
            .capability(ImageMemoryStrategy::BoundedAttention)
            .unwrap();
        assert_eq!(
            capability.parameters.attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        assert_eq!(ATTENTION_CHUNK_SIZE, 64 * 1024 * 1024);
        assert!(contract.lifecycle.attention_chunking);

        let mut sel = selection(ImageMemoryStrategy::BoundedAttention);
        sel.parameters.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE / 2);
        let err = contract.validate_selection(&sel).unwrap_err().to_string();
        assert!(err.contains("attention"), "{err}");

        let mut sel = selection(ImageMemoryStrategy::BoundedAttention);
        sel.parameters.attention_chunk_size = None;
        assert!(contract.validate_selection(&sel).is_err());
    }

    /// SC-15510: rung 2 publishes a **ladder**, not a point, which is what SC-15508's "a single-point
    /// pass cannot mark untested production parameters Verified" requires. The default is still the
    /// sc-13571 512/64, so nothing about an unparameterized render changed.
    #[test]
    fn bounded_decode_publishes_a_candidate_ladder_with_the_historical_default_in_it() {
        let contract = contract();
        let capability = contract
            .capability(ImageMemoryStrategy::BoundedDecode)
            .unwrap();
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
            let mut ctx = context(ImageMemoryStrategy::BoundedDecode);
            ctx.selection.parameters.decode_tile_edge = Some(rejected);
            assert!(
                contract.validate_selection(&ctx.selection).is_err(),
                "rejected edge {rejected} must not validate"
            );
            assert!(matches!(
                safety_check(&contract, &ctx),
                ImageMemorySafetyDecision::Reject { .. }
            ));
        }
        // Every native candidate is selectable end to end, not just advertised.
        for &edge in DECODE_TILE_EDGES {
            let mut ctx = context(ImageMemoryStrategy::BoundedDecode);
            ctx.selection.parameters.decode_tile_edge = Some(edge);
            assert!(
                matches!(
                    safety_check(&contract, &ctx),
                    ImageMemorySafetyDecision::Accept
                ),
                "native candidate {edge} must be admissible"
            );
            let mut scope = begin_request(crate::model::MODEL_ID, &contract, &ctx)
                .unwrap()
                .unwrap();
            scope
                .configure_decode(edge, DECODE_OVERLAP, ctx.geometry)
                .unwrap();
            scope.finish(ImageMemoryRunOutcome::Complete).unwrap();
        }
        // An edge outside the ladder is refused at both layers.
        let mut ctx = context(ImageMemoryStrategy::BoundedDecode);
        ctx.selection.parameters.decode_tile_edge = Some(500);
        assert!(contract.validate_selection(&ctx.selection).is_err());
        assert!(matches!(
            safety_check(&contract, &ctx),
            ImageMemorySafetyDecision::Reject { .. }
        ));
    }

    /// SC-15754: rung 4 is Implemented for a snapshot load, with its window ladder recorded, and 30
    /// (the whole stack) is deliberately absent — one all-covering window bounds nothing, so
    /// advertising it would let a "rung 4" selection run the resident stack and record a zero saving.
    #[test]
    fn bounded_transformer_residency_publishes_only_bounding_windows() {
        let contract = contract();
        let capability = contract
            .capability(ImageMemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert!(matches!(
            capability.support,
            ImageMemoryStrategySupport::Implemented
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
        let mut sel = selection(ImageMemoryStrategy::BoundedTransformerResidency);
        sel.parameters.transformer_window_size = Some(7);
        assert!(contract.validate_selection(&sel).is_err());
        // As is one that omits the parameter entirely.
        let mut sel = selection(ImageMemoryStrategy::BoundedTransformerResidency);
        sel.parameters.transformer_window_size = None;
        assert!(contract.validate_selection(&sel).is_err());
    }

    /// Rung 4 is declared per LOAD, and both load-time preconditions are enforced at the contract
    /// layer rather than at generate time.
    ///
    /// The `Resident` half is the one that matters in production: the worker selects from evidence, so
    /// a contract that advertised rung 4 on a `Resident`-loaded generator would let it pick a strategy
    /// that then hard-errors in `resolve_block_window` — a run-time failure for a request that had a
    /// perfectly good lower rung available. Saying `Missing` up front means the selector simply picks
    /// the next rung down.
    #[test]
    fn a_resident_load_declares_rung_four_missing_so_the_selector_never_picks_it() {
        let resident = LoadSpec::new(WeightsSource::Dir("/nonexistent-z-image-snapshot".into()))
            .with_offload_policy(mlx_gen::OffloadPolicy::Resident);
        let contract = image_memory_contract(crate::model::MODEL_ID, &resident);
        assert_eq!(contract.conformance_errors(), Vec::<String>::new());
        assert!(matches!(
            contract
                .capability(ImageMemoryStrategy::BoundedTransformerResidency)
                .map(|c| &c.support),
            Some(ImageMemoryStrategySupport::Missing)
        ));
        assert!(!contract.lifecycle.transformer_window_materialization);
        assert!(contract
            .validate_selection(&selection(ImageMemoryStrategy::BoundedTransformerResidency))
            .is_err());
        // Every rung below it stays available — this narrows one cell, not the ladder.
        for strategy in [
            ImageMemoryStrategy::Resident,
            ImageMemoryStrategy::StagedResidency,
            ImageMemoryStrategy::BoundedDecode,
            ImageMemoryStrategy::BoundedAttention,
        ] {
            contract.validate_selection(&selection(strategy)).unwrap();
        }
        // ...and the Sequential load of the same snapshot does advertise it, so this is the policy
        // doing the work rather than the path.
        let sequential = super::image_memory_contract(crate::model::MODEL_ID, &spec());
        assert!(matches!(
            sequential
                .capability(ImageMemoryStrategy::BoundedTransformerResidency)
                .map(|c| &c.support),
            Some(ImageMemoryStrategySupport::Implemented)
        ));
    }

    /// A load with no re-openable source (a ComfyUI single-file build) cannot stream blocks, so the
    /// rung is declared `Missing` for it rather than advertised and then failed at run time.
    #[test]
    fn a_single_file_load_declares_rung_four_missing() {
        let spec = LoadSpec::new(WeightsSource::File(
            "/nonexistent/z-image.safetensors".into(),
        ))
        .with_offload_policy(mlx_gen::OffloadPolicy::Sequential);
        let contract = image_memory_contract(crate::model::MODEL_ID, &spec);
        assert_eq!(contract.conformance_errors(), Vec::<String>::new());
        assert!(matches!(
            contract
                .capability(ImageMemoryStrategy::BoundedTransformerResidency)
                .map(|c| &c.support),
            Some(ImageMemoryStrategySupport::Missing)
        ));
        assert!(!contract.lifecycle.transformer_window_materialization);
        assert!(contract
            .validate_selection(&selection(ImageMemoryStrategy::BoundedTransformerResidency))
            .is_err());
        // ...and its asset facts stay the truthful zero (no component tree to sum).
        assert_eq!(contract.asset_facts, ImageMemoryAssetFacts::default());
    }

    #[test]
    fn the_ladder_maps_to_cumulative_request_controls() {
        let decode = GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(DECODE_TILE_EDGE),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        };
        assert_eq!(
            z_image_generation_memory(&selection(ImageMemoryStrategy::Resident)),
            None
        );
        assert_eq!(
            z_image_generation_memory(&selection(ImageMemoryStrategy::StagedResidency)),
            Some(GenerationMemory::default())
        );
        assert_eq!(
            z_image_generation_memory(&selection(ImageMemoryStrategy::BoundedDecode)),
            Some(decode)
        );
        assert_eq!(
            z_image_generation_memory(&selection(ImageMemoryStrategy::BoundedAttention)),
            Some(GenerationMemory {
                chunk_attention: true,
                ..decode
            })
        );
        assert_eq!(
            z_image_generation_memory(&selection(ImageMemoryStrategy::BoundedTransformerResidency)),
            Some(GenerationMemory {
                chunk_attention: true,
                stream_transformer_blocks: true,
                transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
                ..decode
            })
        );
    }

    /// The executable half of the ladder: each request knob is read by exactly the pipeline seam that
    /// owns it, and a lower rung does not turn a higher one on.
    #[test]
    fn the_request_level_knobs_are_read_by_their_own_seams() {
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
            memory: z_image_generation_memory(&selection(
                ImageMemoryStrategy::BoundedTransformerResidency,
            )),
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
        let mut sel = selection(ImageMemoryStrategy::BoundedDecode);
        // A published non-default candidate — 640, not one of the measured-and-rejected sub-512 edges.
        sel.parameters.decode_tile_edge = Some(640);
        let tiled = GenerationRequest {
            memory: z_image_generation_memory(&sel),
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

    /// Rung 4 has no request-scoped lever on a `Resident`-loaded generator, and unlike rung 1 that is
    /// an ERROR rather than a silent downgrade — a windowed forward there would add memory, and a
    /// silently-resident "rung 4" run would poison the calibration evidence.
    #[test]
    fn a_resident_load_refuses_rung_four_instead_of_degrading() {
        let req = GenerationRequest {
            prompt: "a fox".to_owned(),
            memory: z_image_generation_memory(&selection(
                ImageMemoryStrategy::BoundedTransformerResidency,
            )),
            ..Default::default()
        };
        let err = crate::pipeline::resolve_block_window(&req, false, "z_image_turbo")
            .unwrap_err()
            .to_string();
        assert!(err.contains("Sequential"), "{err}");
        assert_eq!(
            crate::pipeline::resolve_block_window(&req, true, "z_image_turbo").unwrap(),
            Some(TRANSFORMER_WINDOW_SIZE as usize)
        );
        // Every lower rung is unaffected by the residency policy.
        for strategy in [
            ImageMemoryStrategy::StagedResidency,
            ImageMemoryStrategy::BoundedDecode,
            ImageMemoryStrategy::BoundedAttention,
        ] {
            let req = GenerationRequest {
                prompt: "a fox".to_owned(),
                memory: z_image_generation_memory(&selection(strategy)),
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
        let contract = contract();
        let pid_edges = pid_decode_tile_edges();
        assert!(!pid_edges.is_empty());
        assert!(
            pid_edges.iter().all(|e| !DECODE_TILE_EDGES.contains(e)),
            "the two routes' domains must not overlap, or a selection would be ambiguous"
        );

        for strategy in [
            ImageMemoryStrategy::BoundedDecode,
            ImageMemoryStrategy::BoundedAttention,
            ImageMemoryStrategy::BoundedTransformerResidency,
        ] {
            let ctx = context_for(strategy, true);
            assert!(
                matches!(
                    safety_check(&contract, &ctx),
                    ImageMemorySafetyDecision::Accept
                ),
                "{strategy:?} must now be admissible on the PiD route"
            );
            let mut scope = begin_request(crate::model::MODEL_ID, &contract, &ctx)
                .unwrap()
                .unwrap();
            scope
                .configure_decode(pid_edges[0], PID_DECODE_OVERLAP, ctx.geometry)
                .unwrap();
            // A native-VAE geometry is refused on the PiD route.
            assert!(scope
                .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, ctx.geometry)
                .is_err());
            scope.finish(ImageMemoryRunOutcome::Complete).unwrap();
        }

        // A selection built for the native route is rejected when the request uses PiD...
        let mut ctx = context_for(ImageMemoryStrategy::BoundedDecode, true);
        ctx.selection.parameters.decode_tile_edge = Some(DECODE_TILE_EDGE);
        ctx.selection.parameters.decode_overlap = Some(DECODE_OVERLAP);
        match safety_check(&contract, &ctx) {
            ImageMemorySafetyDecision::Reject { reason } => {
                assert!(reason.contains("PiD overlay"), "{reason}")
            }
            other => panic!("a native geometry under PiD must be rejected, got {other:?}"),
        }
        // ...and symmetrically, a PiD geometry without the overlay.
        let mut ctx = context(ImageMemoryStrategy::BoundedDecode);
        ctx.selection.parameters.decode_tile_edge = Some(pid_edges[0]);
        ctx.selection.parameters.decode_overlap = Some(PID_DECODE_OVERLAP);
        match safety_check(&contract, &ctx) {
            ImageMemorySafetyDecision::Reject { reason } => {
                assert!(reason.contains("native VAE"), "{reason}")
            }
            other => panic!("a PiD geometry without PiD must be rejected, got {other:?}"),
        }

        // Rungs 0-1 stay available on both routes (they own no decode parameters).
        for strategy in [
            ImageMemoryStrategy::Resident,
            ImageMemoryStrategy::StagedResidency,
        ] {
            assert!(matches!(
                safety_check(&contract, &context_for(strategy, true)),
                ImageMemorySafetyDecision::Accept
            ));
        }
    }

    /// `z_image_edit` resolves to this provider in edit mode (see the module header), so the contract
    /// must admit `ImageMemoryMode::Edit` across the whole ladder. Nothing else in this crate mentions
    /// that id — the alias itself is SceneWorks-side and pinned there — but if the edit *surface* were
    /// inadmissible here, the alias would resolve to a provider that refuses the mode it was aliased
    /// for, and the catalog entry would be unservable with no local test to say so.
    #[test]
    fn the_edit_surface_z_image_edit_resolves_to_is_admissible() {
        let contract = contract();
        for strategy in ImageMemoryStrategy::ALL {
            let mut ctx = context(strategy);
            ctx.mode = ImageMemoryMode::Edit;
            ctx.has_reference = true;
            assert!(
                matches!(
                    safety_check(&contract, &ctx),
                    ImageMemorySafetyDecision::Accept
                ),
                "{strategy:?} must be admissible in edit mode"
            );
            let mut scope = begin_request(crate::model::MODEL_ID, &contract, &ctx)
                .unwrap()
                .expect("edit mode opens a scope");
            scope.finish(ImageMemoryRunOutcome::Complete).unwrap();
        }
    }

    /// The scope's decode parameters are route-specific, so applying a PiD-admitted scope to a
    /// non-PiD request (or vice versa) must be refused the same way geometry drift is. Without this,
    /// a PiD edge (2048 px of super-resolved output) lands in a NATIVE decode, where at 1024² it
    /// collapses to a single tile — an effectively untiled decode while the evidence records a
    /// bounded one.
    #[test]
    fn the_scope_rejects_route_drift_the_way_it_rejects_geometry_drift() {
        let contract = contract();
        for admitted_pid in [false, true] {
            let ctx = context_for(ImageMemoryStrategy::BoundedDecode, admitted_pid);
            let mut scope = begin_request(crate::model::MODEL_ID, &contract, &ctx)
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
            scope.finish(ImageMemoryRunOutcome::Complete).unwrap();
        }
    }

    #[test]
    fn a_stale_calibration_fingerprint_never_admits_an_optimized_fit() {
        let contract = contract();
        let mut ctx = context(ImageMemoryStrategy::BoundedDecode);
        ctx.calibration_fingerprint = "z-image-mlx-something-older".to_owned();
        match safety_check(&contract, &ctx) {
            ImageMemorySafetyDecision::Reject { reason } => {
                assert!(
                    reason.contains("calibration handshake mismatch"),
                    "{reason}"
                )
            }
            other => panic!("stale fingerprint must be rejected, got {other:?}"),
        }
        assert!(begin_request(crate::model::MODEL_ID, &contract, &ctx).is_err());

        let mut ctx = context(ImageMemoryStrategy::BoundedDecode);
        ctx.calibration_abi = IMAGE_MEMORY_CALIBRATION_ABI + 1;
        assert!(matches!(
            safety_check(&contract, &ctx),
            ImageMemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn an_over_budget_prediction_is_rejected_before_any_work() {
        let contract = contract();
        let mut ctx = context(ImageMemoryStrategy::BoundedDecode);
        ctx.predicted_peak_bytes = ctx.budget.effective_bytes() + 1;
        match safety_check(&contract, &ctx) {
            ImageMemorySafetyDecision::Reject { reason } => {
                assert!(reason.contains("exceeds effective budget"), "{reason}")
            }
            other => panic!("over-budget must be rejected, got {other:?}"),
        }
        let mut ctx = context(ImageMemoryStrategy::BoundedDecode);
        ctx.predicted_peak_bytes = ctx.budget.effective_bytes();
        assert!(matches!(
            safety_check(&contract, &ctx),
            ImageMemorySafetyDecision::Accept
        ));
    }

    #[test]
    fn the_scope_overwrites_warm_request_state_and_finishes_once() {
        let contract = contract();
        let ctx = context(ImageMemoryStrategy::BoundedDecode);
        let mut scope = begin_request(crate::model::MODEL_ID, &contract, &ctx)
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

        scope.finish(ImageMemoryRunOutcome::Complete).unwrap();
        assert!(scope.finish(ImageMemoryRunOutcome::Complete).is_err());
        assert!(scope.configure_request(&mut request).is_err());
    }

    #[test]
    fn the_scope_accepts_only_its_declared_parameters() {
        let contract = contract();
        let ctx = context(ImageMemoryStrategy::BoundedTransformerResidency);
        let mut scope = begin_request(crate::model::MODEL_ID, &contract, &ctx)
            .unwrap()
            .unwrap();
        scope
            .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, ctx.geometry)
            .unwrap();
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
        scope.finish(ImageMemoryRunOutcome::Complete).unwrap();

        // Below rung 4 there is no window at all, so the hook refuses everything.
        let ctx = context(ImageMemoryStrategy::BoundedAttention);
        let mut scope = begin_request(crate::model::MODEL_ID, &contract, &ctx)
            .unwrap()
            .unwrap();
        assert!(scope.materialize_transformer_window(0, 1).is_err());
        scope.finish(ImageMemoryRunOutcome::Complete).unwrap();
    }

    #[test]
    fn a_canceled_or_errored_run_still_releases() {
        let contract = contract();
        for outcome in [
            ImageMemoryRunOutcome::Canceled,
            ImageMemoryRunOutcome::Error {
                message: "boom".to_owned(),
            },
        ] {
            let ctx = context(ImageMemoryStrategy::BoundedDecode);
            let mut scope = begin_request(crate::model::MODEL_ID, &contract, &ctx)
                .unwrap()
                .unwrap();
            scope.finish(outcome).unwrap();
        }
        let ctx = context(ImageMemoryStrategy::BoundedDecode);
        drop(begin_request(crate::model::MODEL_ID, &contract, &ctx).unwrap());
    }

    #[test]
    fn every_registered_variant_declares_the_same_conformant_contract() {
        for id in [
            crate::model::MODEL_ID,
            crate::model_base::MODEL_ID,
            crate::model_control::MODEL_ID,
            crate::model_base_control::MODEL_ID,
        ] {
            let contract = image_memory_contract(id, &spec());
            assert_eq!(contract.provider_id, id);
            assert_eq!(contract.conformance_errors(), Vec::<String>::new(), "{id}");
            gen_core_testkit::check_image_memory_contract(&contract).unwrap();
        }
    }

    #[test]
    fn declared_parameters_are_the_defaults_and_all_live_in_the_published_ladders() {
        let contract = contract();
        let params = declared_parameters();
        let decode = contract
            .capability(ImageMemoryStrategy::BoundedDecode)
            .unwrap();
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
                .capability(ImageMemoryStrategy::BoundedAttention)
                .unwrap()
                .parameters
                .attention_chunk_sizes,
            vec![params.attention_chunk_size.unwrap()]
        );
        assert!(contract
            .capability(ImageMemoryStrategy::BoundedTransformerResidency)
            .unwrap()
            .parameters
            .transformer_window_sizes
            .contains(&params.transformer_window_size.unwrap()));
    }
}
