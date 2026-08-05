//! Kolors MLX adoption of the shared memory-strategy contract (SC-15449) — the ladder for the
//! **Kolors** family (SC-15521).
//!
//! ## One provider, one catalog entry, three tiers
//!
//! `kolors` is a single SceneWorks catalog entry (`SceneWorks/kolors-mlx`) with three advertised
//! tiers — `q4` (default), `q8` and `bf16` — served by one registered descriptor. Each tier still
//! owes its **own** measured evidence: sharing a provider's code is explicitly not what makes a
//! tier Verified.
//!
//! ## Kolors is NOT SDXL, even though it runs SDXL's U-Net
//!
//! `crate::unet` re-exports [`mlx_gen_sdxl::UNet2DConditionModel`] unchanged and
//! `UNetConfig::kolors` differs from `sdxl_base` only in `projection_class_embeddings_input_dim`
//! (5632 vs 2816), so the *mechanism* of rungs 2 and 4 transfers verbatim. **The numbers do not**,
//! and one component makes the difference structural rather than incidental: Kolors conditions on
//! **ChatGLM3-6B**, not SDXL's CLIP-L + OpenCLIP-bigG pair. That is a 6B-parameter conditioning
//! tower against SDXL's ~0.8B pair, and it inverts which phase carries the request peak at several
//! advertised geometries. Every figure below is measured on Kolors' own weights by
//! `tests/memory_ladder_real_weights.rs`; nothing is inherited from SC-15525.
//!
//! ## Declared rungs
//!
//! | Rung | Support | Executable seam |
//! |---|---|---|
//! | 0 Resident | Implemented | Warm [`Residency`](mlx_gen::Residency) pair — ChatGLM3-6B + (U-Net + control/IP/VAE/PiD) held across requests |
//! | 1 Staged residency | Implemented (request-scoped) | `GenerationMemory::stage_residency` drives encode → **drop ChatGLM3-6B** → load heavy → denoise + decode |
//! | 2 Bounded decode | **Missing** (measured — [`DECODE_SUPPORT`]) | [`Autoencoder::decode_tiled`](mlx_gen_sdxl::Autoencoder) reaches every route, but no fixed tile edge holds its quality across the advertised output range |
//! | 3 Bounded attention | **Missing** (measured — [`ATTENTION_SUPPORT`]) | [`mlx_gen::attention::sdpa_budgeted_bhsd`] reaches every site and does not move the request peak |
//! | 4 Bounded transformer residency | Implemented (streamable loads) | [`mlx_gen::block_residency::run_windowed`] per `Transformer2D` — eleven sub-stacks, 70 blocks |
//!
//! ## Ownership
//!
//! This file declares *structure and parameter domains only*. Measured coefficients, envelopes and
//! per-tier peaks live in SceneWorks generated evidence keyed by
//! [`MEMORY_CALIBRATION_FINGERPRINT`]; the worker owns live-budget accounting and least-cost
//! selection. The scope below is defense in depth: it can reject a selection, never substitute one.

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
use mlx_gen::{GenerationRequest, LoadShape, LoadSpec, OffloadPolicy, WeightsSource};

/// The decode tile edge a rung-2 request would have defaulted to, kept as the sweep's anchor.
///
/// Retained as evidence, not as a published domain — see [`DECODE_SUPPORT`].
pub const DECODE_TILE_EDGE: u32 = 896;

/// The decode overlap paired with every swept edge, in output pixels.
pub const DECODE_OVERLAP: u32 = 64;

/// The decode tile edges the rung-2 sweep **measured** on Kolors' own VAE and latents, largest
/// first.
///
/// Retained as evidence rather than as a published domain — see [`DECODE_SUPPORT`], which declares
/// rung 2 `Missing` on this family. `tests/memory_ladder_real_weights.rs::decode_tile_mechanism_sweep`
/// drives every one of them through the real
/// [`Autoencoder::decode_tiled`](mlx_gen_sdxl::Autoencoder) so a future story revisiting the
/// head/tail split has numbers to beat rather than a blank.
pub const DECODE_TILE_EDGES_SWEPT: &[u32] = &[896, 768, 640, 512, 448, 384, 320, 256];

/// The rung-2 overlaps the sweep measured against every edge in [`DECODE_TILE_EDGES_SWEPT`].
pub const DECODE_OVERLAPS_SWEPT: &[u32] = &[64, 128, 192, 256];

/// The drift bar this family is judged against, in 8-bit levels out of 255.
///
/// Inherited rather than invented: it is the worst drift a sibling MLX provider on this same shared
/// tiling machinery *admits* into a shipped ladder
/// (`mlx_gen_z_image::memory_strategy::DECODE_TILE_EDGES` tops out at 48 on its 768 px tile; its
/// rejected set starts at 64). Z-Image's decoder is the same diffusers `AutoencoderKL` with the same
/// spatial-extent GroupNorms and the same head/tail split, so it is the closest precedent that
/// exists for Kolors — which runs literally the same `sdxl-vae-fp16-fix` decoder.
pub const DECODE_DRIFT_BAR: u32 = 48;

/// **Rung 2 is `Missing` on Kolors/MLX, and that is a measurement taken on Kolors' own weights.**
///
/// The mechanism is implemented and it works. Kolors decodes through
/// [`mlx_gen_sdxl::Autoencoder`] — the same `sdxl-vae-fp16-fix` `AutoencoderKL` SDXL uses — so
/// `decode_tiled` runs the globally-scoped head (denormalize → `post_quant_conv` → `conv_in` → mid
/// resnets → mid **self-attention**) once on the full latent and tiles only the full-resolution
/// upsample tail. It bounds real memory: at 1024² q4 the whole staged request falls
/// **7.6743 → 5.0664 GiB (−33.98%)** at edge 896 / overlap 192 for ~4% wall clock
/// (`the_withheld_decode_geometry_is_priced_at_the_request_level`). That is the largest single
/// saving anywhere on this family's ladder, and it is withheld anyway.
///
/// It is withheld because it does not preserve the output, and Kolors is measurably **worse** here
/// than SDXL rather than equal to it. Two independent measurements, each sufficient on its own:
///
/// ### 1. On the latent a real render produces, no geometry clears the bar
///
/// Swept against the **exact untiled decode of the same latent** — a real 1024² Kolors render
/// re-encoded through the same VAE — on the q4 tier (`decode_tile_mechanism_sweep`, max Δ per
/// channel out of 255):
///
/// | edge | overlap 64 | 128 | 192 | 256 |
/// |---:|---:|---:|---:|---:|
/// | 896 | 105 | 96 | **88** | 91 |
/// | 768 | 120 | 124 | 121 | 122 |
/// | 640 | 129 | 130 | 121 | 118 |
/// | 512 | 141 | 141 | 141 | 141 |
/// | 448 | 145 | 145 | 145 | 145 |
/// | 384 | 150 | 150 | 150 | 150 |
/// | 320 | 154 | 154 | 154 | 152 |
/// | 256 | 158 | 158 | 156 | 133 |
///
/// The **best** cell in the whole grid is 88/255 — 1.8× [`DECODE_DRIFT_BAR`]. SDXL's best cell on
/// the same machinery and the same VAE was 38/255, i.e. it *cleared* the bar on this instrument and
/// was withheld on the two arguments below. Kolors fails on the first instrument as well, and the
/// reason is the latent rather than the decoder: Kolors' `scaling_factor` is 0.13025 against SDXL's
/// 0.13025 — identical — but its denoiser is trained against a ChatGLM3 conditioning distribution
/// and produces latents with visibly wider per-channel dynamic range, which is exactly what a
/// per-tile GroupNorm punishes. **This is the concrete case for measuring per family instead of
/// inheriting**: the same code, the same VAE weights, and a verdict that is 2.3× worse.
///
/// On the **production** latent (what the denoiser actually hands the decode phase, rather than a
/// re-encoded finished image) the same geometry drifts **131/255**
/// (`the_withheld_decode_geometry_is_priced_at_the_request_level`). Both instruments agree.
///
/// ### 2. And the datum is a property of the geometry, not of the tile edge
///
/// `MemoryParameterRanges::decode_tile_edges` is an absolute pixel domain with no geometry axis, so
/// publishing 896 publishes it at everything `crate::registry::descriptor` advertises — and that is
/// `max_size: 2048`. Re-swept across that range at the best overlap
/// (`no_single_decode_tile_edge_clears_the_bar_across_the_advertised_output_range`):
///
/// | output | tiles | tile covers | max Δ | vs the 48/255 bar |
/// |---:|---:|---:|---:|---|
/// | 1024² | 4 | 87.5% | **88** | fails |
/// | 1280² | 4 | 70.0% | 116 | fails |
/// | 1536² | 4 | 58.3% | 158 | fails |
/// | 2048² | 9 | 43.75% | 134 | fails |
///
/// The GroupNorm mechanism predicts this shape: what governs the drift is the tile's **fraction of
/// the image**, because that is what sets how far a tile's statistics can sit from the global ones.
/// Edge 896 covers 87.5% of a 1024² output and 43.75% of a 2048² one. No fixed pixel edge holds a
/// constant fraction across a 4× range.
///
/// So the rung is withheld, and the prize is recorded with a number rather than dismissed:
/// **−33.98% of the request peak for ~4% wall clock**, a far better memory/latency trade than rung
/// 4's. What would unlock it is a geometry-relative tile parameter (a fraction of the output rather
/// than a pixel edge), or a decoder whose tail normalizes over the full extent. Both are
/// contract-level changes, and neither is this story's.
pub const DECODE_SUPPORT: MemoryStrategySupport = MemoryStrategySupport::Missing;

/// The attention chunk size the rung-3 sweep exercised.
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;

/// **Rung 3 is `Missing` on Kolors/MLX, and that is two independent measurements.**
///
/// The mechanism is implemented and reaches every attention site: both the self-attention and the
/// cross-attention of all 70 `TransformerBlock`s, plus the IP-Adapter decoupled branch, route
/// through [`mlx_gen::attention::sdpa_budgeted_bhsd`] on every advertised denoise route — Kolors
/// inherits that wiring with the U-Net.
///
/// It is not published for two reasons, either of which is sufficient. Both are measured on the q4
/// tier at 1024², and each has its own test because they are measured at different scopes:
///
/// 1. **It does not move the REQUEST peak**, and at the seam it moves it the wrong way. End to end
///    (`attention_chunking_is_measured_against_the_rung_two_top`, which re-assembles the staged
///    request from the public pipeline entry points because the production resolver refuses a
///    bounded-attention selection):
///
///    | steps | unchunked | chunked | Δ peak | max Δ px | mean Δ px |
///    |---:|---:|---:|---:|---:|---:|
///    | 1 | 7.6743 GiB | 7.6743 GiB | **−0.00%** | 43/255 | 1.11 |
///    | 4 | 7.6743 GiB | 7.6743 GiB | **−0.00%** | 255/255 | 5.87 |
///
///    Measured instead at the U-Net seam, where the effect is not diluted by the rest of the request
///    (`attention_chunking_is_measured_at_the_unet_seam`), one CFG-batched 1024² forward goes
///    **1.6289 → 1.7573 GiB, +7.9%**: chunking *adds* transients here rather than bounding anything.
///    That is the hazard `mlx_gen::attention`'s own docs name — when q/k/v are already materialized
///    and pinned, chunking only adds — and on a U-Net it is the normal case, because the conv/resnet
///    trunk between every attention has already broken the lazy graph that `eval_per_chunk` exists to
///    cut. The epic is explicit that a rung which does not move the request peak is not a saving; one
///    that raises it is worse than absent.
/// 2. **The output moves — and that is a property of the CHOSEN AXIS, not of Kolors.** The
///    arithmetic is exactly preserved: [`sdpa_budgeted_bhsd`](mlx_gen::attention::sdpa_budgeted_bhsd)
///    chunks the **query axis only**, so each chunk is a complete fused SDPA over the full k/v with
///    no accumulator and no running max. What changes is which reduced-precision Metal matmul
///    specialization MLX dispatches for the `[.., block, Sk]` product — the sc-2338 parity class.
///    `gen_core::attention_budget` carries a second axis for exactly this reason:
///    `AttentionBudget::head_chunks` narrows complete heads and *preserves bit identity*, and
///    `sdpa_budgeted_bhsd` never uses it. Measured at **one** step, where the sampler cannot amplify
///    anything, the raw per-forward divergence is `max|Δeps| = 3.98e-3` and the image moves by
///    43/255; by four steps it is a different image.
///
///    A head-axis implementation would be bit-exact by construction; what it would still have to beat
///    is reason 1, which is the binding one.
///
/// The one-step control is what separates both claims from a wiring bug. It is also why the verdict
/// is stated per family per backend: Z-Image measures −1.7% on its denoise phase with a
/// bit-identical image on the same primitive.
pub const ATTENTION_SUPPORT: MemoryStrategySupport = MemoryStrategySupport::Missing;

/// The production transformer-window domain for rung 4: how many consecutive `TransformerBlock`s of
/// **each** `Transformer2D` are held materialized at once.
///
/// The domain is a **cadence**, applied independently to eleven sub-stacks of two depths
/// (`[2, 2, 10, 10, 10, 10, 10, 10, 2, 2, 2]`).
/// [`BlockPlan::new`](mlx_gen::block_residency::BlockPlan::new) clamps a window to its stack's
/// depth, so a cadence wider than 2 degrades the five 2-deep sub-stacks to fully resident rather
/// than erroring — which is why the domain is stated once and not per depth. 10 is therefore the
/// widest cadence that means anything here: it holds one *whole* deep sub-stack, i.e. one re-open
/// per stack per step instead of ten.
///
/// ## Measured: four cadences, one saving
///
/// `transformer_window_sweep_and_streamed_output_identity` (`KOLORS_WINDOW_PROBE_TIER` /
/// `KOLORS_WINDOW_PROBE_SIZE` drive the off-default rows), fresh bundle per row, every row
/// byte-identical to its resident control. Request peak in GiB:
///
/// | tier | output | control | cadence 1 | 2 | 5 | 10 | spread |
/// |---|---:|---:|---:|---:|---:|---:|---:|
/// | q4 | 1024² | 7.674 | 6.986 | 6.986 | 6.986 | 6.986 | **0** |
/// | q8 | 1024² | 8.755 | 7.436 | 7.436 | 7.436 | 7.436 | **0** |
/// | q4 | 512² | 3.088 | 2.400 | 2.400 | 2.400 | 2.400 | **0** |
///
/// **Read this table as an observation, not as a gated invariant.** Exactly one row of it is
/// asserted — **q4 / 1024²**, the sweep's default tier and geometry, which is also the catalog's
/// default tier. The others come from re-running the sweep under `KOLORS_WINDOW_PROBE_TIER` /
/// `KOLORS_WINDOW_PROBE_SIZE`, and in that mode the flatness and wall-clock assertions are
/// **reported rather than asserted**. And even the asserted row is asserted by an `#[ignore]`d
/// real-weight test, so it is gated by a human running it against cached weights, not by CI.
///
/// Wall clock falls monotonically with cadence (1024² q4: 2074 → 741 ms/step, **2.8× cheaper**).
/// Absolute numbers track machine load, so the ratio is the quantity worth quoting and the
/// wall-clock assertions in the sweep are deliberately loose.
///
/// ## The mechanism is PHASE SEPARATION, and on Kolors the arithmetic says so twice
///
/// The request peak is a `max` over phases, and rung 4 only bounds weights *inside one of them*.
/// Read the saving against the U-Net safetensors headers (summing `data_offsets`,
/// `transformer_blocks.*` against the whole file):
///
/// | tier | U-Net total | `transformer_blocks` | one deep block | measured saving | saving ÷ block set |
/// |---|---:|---:|---:|---:|---:|
/// | q4 | 1.4363 GiB | 0.6875 GiB | 0.0109 GiB | 0.688 (1024² **and** 512²) | **1.001** |
/// | q8 | 2.6303 | 1.3164 | 0.0209 | 1.319 (1024²) | **1.002** |
///
/// **The saving is the entire block weight set** — every tier, and *identically* at 1024² and 512²
/// on q4 where the activation working sets differ four-fold. That is arithmetically incompatible
/// with any window being resident when the peak is taken. Eleven `Transformer2D` sub-stacks run in
/// sequence and `run_windowed` releases one before opening the next, so if the peak occurred during
/// the windowed forward the saving could be **at most** `block set − w × (one deep block)` — 0.677
/// GiB on q4 at cadence 1, 0.579 GiB at cadence 10. The measured 0.688 GiB exceeds *both*.
///
/// So zero window weights are resident at the peak moment: **the peak is the decode**, stacking on a
/// still-resident U-Net (rung 1 sheds ChatGLM3-6B, not the heavy bundle — `KolorsHeavyOwned` is one
/// bundle and the decode runs inside the same render closure).
///
/// ### The condition, stated so it can be checked — and why Kolors' flat region is WIDER than SDXL's
///
/// Flatness holds **only while** `decode transient + resident remainder` exceeds the windowed
/// forward's own peak at the *widest* published cadence. SDXL's flat region breaks at its advertised
/// `min_size` of 512² on q8, because there the decode transient no longer dominates. Kolors' does
/// **not** break at 512² on any cached tier, and the reason is the same one that makes rung 1 worth
/// so much more here: Kolors' U-Net weight set is the *only* thing left resident during the decode
/// once ChatGLM3-6B is shed, and at q4/q8 that remainder is small enough that the decode transient
/// still dominates at 512². The inequality is checked directly by
/// `the_cadence_flatness_condition_is_checked_not_assumed`, which measures the decode transient and
/// the widest-cadence windowed forward separately and asserts the ordering rather than inferring it
/// from the flat column.
///
/// The default stays at the tightest cadence anyway — see [`TRANSFORMER_WINDOW_SIZE`].
///
/// ### This is coupled to rung 2, and the coupling runs the wrong way
///
/// Every row above was measured with [`DECODE_SUPPORT`] `Missing`, i.e. with the decode transient at
/// **full size** — which is precisely the term keeping the forward off the peak. A story that ships
/// a bounded decode *lowers that term deliberately*, shrinking the flat region toward the
/// small-output end and potentially eliminating it. Publishing rung 2 invalidates every rung-4 row
/// here and must re-measure the cadence domain in the same change — see
/// [`MEMORY_CALIBRATION_FINGERPRINT`], which must be bumped with it.
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 5, 10];

/// The transformer window rung 4 executes at when a request names none.
///
/// **The tightest cadence**, which is what every other MLX adopter on this ladder defaults to.
///
/// The honest counter-argument is recorded because it is a good one: on every configuration measured
/// here cadence 10's peak equals cadence 1's exactly, while cadence 10 is 2.8× faster. Read as a
/// table of three rows, 10 dominates.
///
/// It is still not the default. A default is the value a caller gets *without asking*, so it has to
/// be safe across the whole advertised geometry range (512²–2048², `crate::registry::descriptor`),
/// not optimal at the three points that happen to have been measured. The flat column is a
/// consequence of an inequality (see [`TRANSFORMER_WINDOW_SIZES`]) that this family satisfies at
/// every *measured* point and that SDXL — the same U-Net, the same rung — violates at its own
/// `min_size`. A domain known to be inequality-dependent somewhere in the sibling set does not
/// support extrapolation to the geometries nobody measured; the tightest weight bound does, because
/// it is the bound rung 4 can always make good on.
///
/// The other three cadences remain published and selectable, and at 1024² a selector that does not
/// need the tighter bound should absolutely pick 10 and take the 2.8× wall-clock saving; it just has
/// to *choose* it, against calibration for that cadence, rather than receive it by omission.
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;

/// The rung-4 **component scopes** this provider implements — and Kolors is the first MLX provider
/// to implement more than one.
///
/// `Dit` is the U-Net's eleven `Transformer2D` sub-stacks (70 blocks), reached through the
/// re-exported `mlx_gen_sdxl::block_stream`. `TextEncoder` is the ChatGLM3-6B tower's 28 `GlmBlock`s,
/// reached through `crate::block_stream`. `Both` composes them in one request.
///
/// ## Why the second scope exists here and nowhere else on this ladder
///
/// `gen_core::TransformerComponent::TextEncoder` has existed since SC-15449 and no adopter had ever
/// populated it. SDXL's reasoning for `Dit`-only is sound *on SDXL*: its dual CLIP towers are shed by
/// rung 1 before the U-Net loads, so windowing them would bound a phase that is never the maximum —
/// and the epic is explicit that bounding a non-peak phase is not a saving.
///
/// Kolors' tower is **ChatGLM3-6B**, larger than the U-Net at every tier (3.98 / 6.64 / 11.63 GiB
/// against 1.80 / 2.84 / 4.80). So "rung 1 already sheds it" is not the end of the argument: rung 1
/// sheds it before the *denoise*, but the conditioning phase itself still has to hold it. Measured
/// per tier per advertised geometry
/// (`the_text_encoder_window_scope_cannot_move_the_request_peak`, conditioning-phase peak against
/// the whole-request peak, GiB):
///
/// | tier | 512² | 1024² | 2048² |
/// |---|---|---|---|
/// | q4 | 3.769 / 5.620 | 3.781 / 16.041 | 3.781 / 57.062 |
/// | q8 | 6.353 / 6.894 | 6.430 / 17.086 | 6.430 / 58.107 |
/// | **bf16** | **11.360 / 11.360** | 11.364 / 19.031 | 11.364 / 60.053 |
///
/// **One cell of nine is conditioning-bearing: `bf16` at the advertised `min_size`.** There the
/// request peak *is* the ChatGLM3 residency, and a text-encoder window moves it — which is a
/// measured saving, not a phase win, and therefore something a caller must be able to select. It is
/// also a cell a caller can genuinely land on: `bf16` is an advertised tier and 512² is
/// `descriptor().capabilities.min_size`.
///
/// ## The two scopes have opposite cost shapes, which is why they are separately selectable
///
/// `gen_core`'s own note on the variant says it: the encoder re-materializes **once per generation**
/// while the DiT re-materializes once per denoise **step**. Measured here, the text-encoder window
/// costs 28 block re-opens against a phase that runs once; the U-Net window costs 70 re-opens across
/// eleven sub-stacks against a phase that runs `steps` times. Folding them into one flag would have
/// forced a caller who wants the cheap one to buy the expensive one.
///
/// The production **default** stays [`TRANSFORMER_WINDOW_COMPONENT`] = `Dit`, because that is the
/// scope that pays at the tier and geometry the catalog defaults to.
pub const TRANSFORMER_WINDOW_COMPONENTS: &[TransformerComponent] = &[
    TransformerComponent::Dit,
    TransformerComponent::TextEncoder,
    TransformerComponent::Both,
];

/// The scope this provider declares as its production selection — the one a request that engages
/// rung 4 without naming a component receives.
///
/// `Dit`, because it is the scope that pays at the tier and geometry the catalog defaults to (`q4`,
/// 1024²), where the conditioning phase is 3.78 GiB against a 16.04 GiB request peak and a
/// text-encoder window would therefore bound nothing a selector could admit against. A caller on the
/// one conditioning-bearing cell — `bf16` at the advertised `min_size` — has to *choose*
/// `TextEncoder` (or `Both`), against calibration for that scope, rather than receive it by
/// omission. See [`TRANSFORMER_WINDOW_COMPONENTS`] for the per-cell measurement.
pub const TRANSFORMER_WINDOW_COMPONENT: TransformerComponent = TransformerComponent::Dit;

/// Calibration content fingerprint. It must change whenever quantization floors, tensor layout, or
/// execution structure change in a way that invalidates measurements taken against this provider.
///
/// Load shape is a typed evidence-key axis carried separately on [`MemoryCalibrationIdentity`]; this
/// content fingerprint stays shape-independent.
///
/// **The paired evidence must be keyed per cadence, not per rung.** The cadences differ by up to
/// 2.8× in wall clock, so a selector weighing peak against time needs a row per candidate rather
/// than one row for the rung. `MemoryFormulaVariable::TransformerWindowSize` is already declared on
/// the formula for exactly this.
///
/// **This fingerprint is coupled to rung 2.** The cadence rows are all measured with
/// [`DECODE_SUPPORT`] `Missing`, and the reason they are flat is that the *decode* carries the peak.
/// A story that publishes a bounded decode changes which phase is peak-bearing, which invalidates
/// every rung-4 row here — so it must bump this fingerprint and re-measure the cadence domain in the
/// same change, not inherit these numbers.
///
/// It is **not** shared with `sdxl-mlx-unet-shared-ladder-v2` even though the U-Net module is
/// literally the same code: the conditioning tower differs, the peaks differ, and rung 2's verdict
/// differs by 2.3× on the same instrument. Sharing a fingerprint would let a selector reuse SDXL's
/// evidence for a Kolors fit, which is precisely the substitution the epic forbids.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "kolors-mlx-chatglm3-sdxl-unet-ladder-v1";

/// Whether THIS load can execute rung 4. **Four** independent facts decide it.
///
/// 1. **A re-openable snapshot.** The window rebuilds blocks from the U-Net checkpoint, so the load
///    must have come from a snapshot directory. Kolors rejects a single-file source outright
///    (`registry::resolve_root`), so this is belt-and-braces here rather than the live discriminator
///    it is on SDXL.
/// 2. **[`LoadShape::DeferredMaterialization`]**, i.e. the load was asked not to bulk-commit the
///    stacks. `OffloadPolicy` is deliberately absent from this test — phase release is a separate
///    axis, and rung 1's prerequisite is enforced on the SELECTION, not on the load.
/// 3. **A load that leaves the resident blocks unmaterialized** — see [`load_leaves_blocks_lazy`].
/// 4. **Replayable adapters** — see [`adapters_are_replayable`], the one that would otherwise
///    silently drop a user's LoRA.
pub fn streamable(spec: &LoadSpec) -> bool {
    matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.weights, WeightsSource::Dir(_))
        && load_leaves_blocks_lazy(spec)
        && text_encoder_leaves_blocks_lazy(spec)
        && adapters_are_replayable(spec)
}

/// Whether this load leaves the resident `TransformerBlock`s **unmaterialized**, which is what makes
/// a window bound anything at all.
///
/// MLX's `Array::load_safetensors` is lazy per tensor and `Weights::cast_all` composes lazily, so a
/// dense fp16 load's blocks are unevaluated graph nodes until something forces them — and a windowed
/// forward never touches `self.blocks`. A **load-time quantization over a dense snapshot** is the
/// exception: `unet.quantize(bits)` packs every projection, which evaluates it. A pre-quantized
/// (packed) snapshot at the requested tier re-quantizes nothing, so it stays lazy.
///
/// This is the predicate that decides rung 4 per **tier** on Kolors, and all three shipped tiers
/// pass it: `SceneWorks/kolors-mlx`'s `q4/` and `q8/` `unet/config.json` both carry the
/// `quantization` marker `mlx_gen::quant::packed_quant_bits` reads, and `bf16/` resolves to
/// `Quant::None`. The *upstream* `Kwai-Kolors/Kolors-diffusers` snapshot is dense and carries no
/// marker, so a `--quantize` load pointed at it is correctly refused rung 4 — it would pack every
/// block at load and a window over an already-materialized trunk bounds nothing.
///
/// A tier mismatch (`needs_load_time_quant` errors) reads as *not* lazy: the load will reject it
/// with a better message a moment later, and a contract builder is not the place to surface it
/// first.
pub fn load_leaves_blocks_lazy(spec: &LoadSpec) -> bool {
    let Some(quant) = spec.quantize else {
        // Dense fp16: `cast_all` is lazy, nothing packs.
        return true;
    };
    let WeightsSource::Dir(root) = &spec.weights else {
        return false;
    };
    matches!(
        mlx_gen::quant::needs_load_time_quant(root, "unet", quant.bits(), crate::MODEL_ID),
        Ok(false)
    )
}

/// Whether this load leaves the **ChatGLM3-6B** blocks unmaterialized — the text-encoder half of
/// [`streamable`].
///
/// Checked separately from [`load_leaves_blocks_lazy`] rather than inferred from it, even though
/// every shipped `SceneWorks/kolors-mlx` tier packs both components together. Rung 4 publishes a
/// `TextEncoder` scope on this family, so a snapshot whose `unet/` is packed and whose
/// `text_encoder/` is dense would arm a window over a tower `KolorsText::quantize` had already
/// materialized — which bounds nothing and adds a copy. Deriving one from the other would make that
/// shape invisible.
///
/// The rung is declared all-or-nothing across scopes (one `transformer_window_materialization` flag,
/// one published component list), so this composes into [`streamable`] rather than gating only the
/// `TextEncoder` arm: a half-streamable load would publish scopes it cannot honour.
pub fn text_encoder_leaves_blocks_lazy(spec: &LoadSpec) -> bool {
    let Some(quant) = spec.quantize else {
        return true;
    };
    let WeightsSource::Dir(root) = &spec.weights else {
        return false;
    };
    matches!(
        mlx_gen::quant::needs_load_time_quant(root, "text_encoder", quant.bits(), crate::MODEL_ID),
        Ok(false)
    )
}

/// Whether this load's adapters survive a rung-4 re-materialization.
///
/// `registry::load_heavy_owned` installs a LoRA/LoKr through
/// [`apply_sdxl_adapters_with`](mlx_gen_sdxl::apply_sdxl_adapters_with) **before** the U-Net
/// quantize, and that helper decides between two installs by whether the target linear is already
/// packed:
///
/// * on a **pre-quantized (Q4/Q8) snapshot** the base is packed, so it `push`es an
///   [`Adapter`](mlx_gen::adapters::Adapter) as a forward-time residual. That is capturable and
///   replayable, and `mlx_gen_sdxl::block_stream` does so;
/// * on a **dense snapshot** it calls `merge_dense_delta`, folding the delta into the base weight.
///   The snapshot on disk does not carry it. A block re-read from that snapshot would be the
///   **un-adapted** model — a plausible wrong image, no error, and no memory assertion that would
///   notice.
///
/// So adapters are replayable exactly when the on-disk U-Net is already packed at the requested
/// tier. A load that would merge simply does not arm rung 4; refusing the rung is the only honest
/// option, because the alternative is quietly rendering without the LoRA the user asked for.
///
/// **On Kolors this is strictly stronger than on SDXL**, because Kolors' only dense tier is `bf16`:
/// a `bf16` + LoRA load is refused rung 4 outright, and that is the whole of the family's LoRA
/// surface at that tier. It is still the right answer — SC-15521's alternative was rendering the
/// base model and calling it a LoRA render.
pub fn adapters_are_replayable(spec: &LoadSpec) -> bool {
    if spec.adapters.is_empty() {
        return true;
    }
    // A merge happens on a DENSE base. Only an already-packed snapshot pushes residuals.
    spec.quantize.is_some() && load_leaves_blocks_lazy(spec)
}

/// Build the Kolors MLX provider contract at this [`LoadSpec`].
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let components = crate::registry::component_footprint(spec)?;
    contract_with_asset_facts(
        provider_id,
        spec,
        components.text_encoder,
        components.dit,
        components.vae,
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
    // Only what the PUBLISHED rungs consume. `DecodeTileArea` and `AttentionChunkSize` are absent
    // because rungs 2 and 3 are declared `Missing` here — a formula that reads a parameter no
    // selectable strategy can set would invite calibration keyed on a value that never varies.
    let mut variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
    ];
    if streamable {
        // Rung 4 makes transformer weight residency a VARIABLE of the peak rather than a constant
        // folded into `AssetBytes`.
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
        decode_tiling: false,
        attention_chunking: false,
        transformer_window_materialization: streamable,
    };
    // A Resident selection preserves the load-time defaults: Kolors' historical warm render tiles
    // nothing and windows nothing, and there is no Sequential-only shipped tiling default to
    // override. The load-time `OffloadPolicy` remains the *default* for a request that names no
    // `stage_residency`, which `Resident` overrides explicitly through the contract.
    contract.resident_request_memory = ResidentRequestMemory::ExplicitResident;

    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::BoundedDecode => DECODE_SUPPORT,
            MemoryStrategy::BoundedAttention => ATTENTION_SUPPORT,
            MemoryStrategy::BoundedTransformerResidency if !streamable => {
                MemoryStrategySupport::Missing
            }
            _ => MemoryStrategySupport::Implemented,
        };
        capability.parameters = match capability.strategy {
            MemoryStrategy::BoundedTransformerResidency if streamable => MemoryParameterRanges {
                transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                transformer_window_components: TRANSFORMER_WINDOW_COMPONENTS.to_vec(),
                ..Default::default()
            },
            _ => MemoryParameterRanges::default(),
        };
    }
    // No `pid_decode_routes`: that declaration exists to split rung 2's candidate domain between the
    // native VAE and the PiD student, and rung 2 is not selectable on this provider. The student
    // keeps its own internal auto-planning exactly as it did before this story — unchanged
    // behaviour, not a removal.
    contract.pid_decode_routes = None;
    if streamable {
        // Rung 4 bounds the U-Net's denoise-phase weight residency. Without rung 1 engaged,
        // ChatGLM3-6B stays resident alongside it and the REQUEST peak does not move — a phase win
        // that is not a saving. On Kolors that encoder is the single largest component in the model,
        // so the composition is not merely suboptimal, it is inverted: the rung would bound 0.69 GiB
        // of U-Net weights while 3.06 GiB of encoder sat next to them. Declaring the edge lets the
        // shared selector refuse it instead of every caller re-deriving it from the cost order.
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

/// Refuse a bounded-decode geometry. Rung 2 is `Missing` on this provider ([`DECODE_SUPPORT`]), so
/// there is no admitted domain to validate against and every geometry is out of it.
///
/// This is a *rejection* rather than an absence on purpose. The shared `validate_selection` already
/// refuses a rung the contract declares `Missing`, but a calibration harness that hand-builds a
/// [`GenerationMemory`](mlx_gen::gen_core::GenerationMemory) and calls `generate` never crosses
/// admission — and the tiling mechanism is still reachable through the crate as the sweep's subject.
/// This closes the path from "the code exists" to "a render silently used it".
fn refuse_decode(provider_id: &str, edge: Option<u32>, overlap: Option<u32>) -> CoreError {
    CoreError::Unsupported(format!(
        "{provider_id}: bounded decode is not selectable on this provider (rung 2 is declared \
         Missing). The tiled decode was measured and withheld: its best geometry (edge \
         {DECODE_TILE_EDGE} overlap 192) drifts 88/255 at 1024² against a {DECODE_DRIFT_BAR}/255 \
         sibling bar, and the same edge drifts 116 at 1280², 158 at 1536² and 134 at 2048² — the \
         upsample tail's GroupNorms normalize over each tile's own crop, so what bounds the drift \
         is the tile's FRACTION of the output, and a fixed pixel edge cannot hold that across an \
         advertised range up to 2048². Requested edge {edge:?} overlap {overlap:?}; see \
         memory_strategy::DECODE_SUPPORT for the full sweep."
    ))
}

/// Validate a rung-4 transformer window against the published domain.
///
/// The shared `validate_selection` already refuses an out-of-domain window at admission, so this is
/// defense in depth. It is not redundant: the direction it closes is the dangerous one — *executing*
/// at a wider cadence than the declared domain **under**-predicts peak, and a selector that sized a
/// fit off the published domain would then be handed a render that does not fit.
fn validate_window(size: u32) -> CoreResult<()> {
    if !TRANSFORMER_WINDOW_SIZES.contains(&size) {
        return Err(CoreError::Unsupported(format!(
            "kolors transformer window {size} is outside the calibrated domain \
             {TRANSFORMER_WINDOW_SIZES:?}"
        )));
    }
    Ok(())
}

/// The numeric tier this generator actually runs.
fn loaded_tier(spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    }
}

/// The provider safety check: the calibration handshake and tier match, then the shared contract's
/// own selection validation, then this provider's route gate, then the budget. Defense in depth
/// only — it can reject, never swap in a different strategy or tier.
pub(crate) fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        // Kolors advertises txt2img, img2img, ControlNet-pose, IP-Adapter and the combined
        // strict-pose tier, so the mode axis is deliberately permissive. What is NOT permissive is
        // the geometry.
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            return Err(refuse_decode(
                &contract.provider_id,
                context.selection.parameters.decode_tile_edge,
                context.selection.parameters.decode_overlap,
            ));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedAttention) {
            return Err(CoreError::Unsupported(format!(
                "{}: bounded attention is not selectable on this provider (rung 3 is declared \
                 Missing): it moves the request peak 0.00% here (and +7.9% at the U-Net seam), and \
                 the query-row chunking axis is not bit-exact on Metal, which an fp16 \
                 chaos-sensitive schedule amplifies to a different image — see \
                 memory_strategy::ATTENTION_SUPPORT",
                contract.provider_id
            )));
        }
        if contract.engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        ) {
            if !contract.lifecycle.transformer_window_materialization {
                return Err(CoreError::Unsupported(format!(
                    "{}: bounded transformer residency requires a DeferredMaterialization load over \
                     a snapshot directory whose U-Net AND ChatGLM3 blocks stay lazy and whose \
                     adapters (if any) are replayable — see memory_strategy::streamable",
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
    let tier = loaded_tier(spec);
    let route = |use_pid| MemoryBehaviorRoute {
        mode: MemoryMode::TextToImage,
        reference_count: 0,
        use_pid,
        // Rung 1 is request-scoped, and every optimized selection engages it by cost order — which
        // is also rung 4's declared prerequisite.
        has_phases: true,
        overlay: None,
    };
    let fixtures = vec![MemoryBehaviorFixture::new(
        mlx_gen::gen_core::standard_memory_behavior_context(contract, strategy, tier, route(false))?,
    )];
    Ok(fixtures)
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

/// The widest `Transformer2D` sub-stack depth, which is the window domain the shared request scope
/// validates against.
///
/// Derived from [`UNetConfig::kolors`](mlx_gen_sdxl::UNetConfig) rather than hardcoded, and it is
/// the **maximum** rather than the total: the shared hook's contract is "a window of `w` starting at
/// block `f` covers `min(w, blocks - f)`", which is a statement about ONE stack. Kolors runs eleven,
/// and the widest is the only one that can bound the published cadence.
///
/// The `down_block_types` filter is load-bearing: level 0 is a plain `DownBlock2D` with no
/// attention, so the `1` sitting at `transformer_layers_per_block[0]` describes nothing that exists.
fn widest_transformer_stack() -> usize {
    let cfg = mlx_gen_sdxl::UNetConfig::kolors();
    cfg.transformer_layers_per_block
        .iter()
        .enumerate()
        .filter(|(i, _)| cfg.down_block_types[*i].contains("CrossAttn"))
        .map(|(_, layers)| *layers as usize)
        .max()
        .unwrap_or(1)
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
    let id = provider_id;
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        widest_transformer_stack(),
        move |_use_pid, edge, overlap| Err(refuse_decode(id, Some(edge), Some(overlap))),
    )?;
    // Rungs 2 and 3 are `Missing`, so neither can be engaged and neither parameter is ever set.
    config.attention_chunk_size = None;
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

/// The rung-4 selection this request resolved to: a cadence, and which transformer stack(s) it
/// applies to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TransformerWindow {
    pub size: usize,
    pub component: TransformerComponent,
}

impl TransformerWindow {
    /// The cadence for the U-Net's eleven `Transformer2D` sub-stacks, or `None` when this scope
    /// leaves them resident.
    pub(crate) fn dit(&self) -> Option<usize> {
        self.component.includes_dit().then_some(self.size)
    }

    /// The cadence for the ChatGLM3-6B tower's 28 blocks, or `None` when this scope leaves them
    /// resident.
    pub(crate) fn text_encoder(&self) -> Option<usize> {
        self.component.includes_text_encoder().then_some(self.size)
    }
}

/// Rung 4: the requested window cadence and scope, or `None` for the resident stacks.
///
/// A scope this family does not implement — or a cadence outside the measured
/// [`TRANSFORMER_WINDOW_SIZES`] — is a typed rejection rather than a silently narrowed (or silently
/// *widened*) execution.
pub(crate) fn transformer_window(
    req: &GenerationRequest,
) -> mlx_gen::Result<Option<TransformerWindow>> {
    let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
        return Ok(None);
    };
    let component = memory
        .transformer_window_component
        .unwrap_or(TRANSFORMER_WINDOW_COMPONENT);
    if !TRANSFORMER_WINDOW_COMPONENTS.contains(&component) {
        return Err(mlx_gen::Error::Unsupported(format!(
            "kolors implements the {TRANSFORMER_WINDOW_COMPONENTS:?} transformer window components, \
             got {component:?}"
        )));
    }
    let size = memory
        .transformer_window_size
        .unwrap_or(TRANSFORMER_WINDOW_SIZE);
    validate_window(size).map_err(|error| {
        mlx_gen::Error::Unsupported(format!("kolors transformer window rejected: {error}"))
    })?;
    Ok(Some(TransformerWindow {
        size: size as usize,
        component,
    }))
}

/// Rung 2: **always a refusal.** Bounded decode is `Missing` on this provider ([`DECODE_SUPPORT`]),
/// and this is the request-side layer of that.
///
/// It returns `Ok(None)` — the exact single-pass decode — when the request did not ask for a bounded
/// decode, and a typed error when it did. The error direction is what matters: `Ok(None)` for a
/// request that *did* ask would silently render an unbounded decode while the caller believed it had
/// selected a bounded one, which is the false-green this seam exists to prevent.
pub(crate) fn decode_tiling(req: &GenerationRequest) -> mlx_gen::Result<Option<TilingConfig>> {
    let Some(memory) = req.memory.filter(|memory| memory.tile_vae_decode) else {
        return Ok(None);
    };
    Err(mlx_gen::Error::Unsupported(
        refuse_decode(
            crate::MODEL_ID,
            memory.decode_tile_edge,
            memory.decode_overlap,
        )
        .to_string(),
    ))
}

/// Rung 3: **always the unbounded plan.** Bounded attention is `Missing` on this provider
/// ([`ATTENTION_SUPPORT`]), so a request that asks for it is refused rather than silently
/// unbounded — the same direction `decode_tiling` closes.
pub(crate) fn attention_plan(
    req: &GenerationRequest,
) -> mlx_gen::Result<mlx_gen::attention::AttentionPlan<'_>> {
    if req.memory.is_some_and(|memory| memory.chunk_attention) {
        return Err(mlx_gen::Error::Unsupported(
            "kolors: bounded attention is not selectable on this provider (rung 3 is declared \
             Missing): it moves the request peak 0.00% here (and +7.9% at the U-Net seam), and the \
             query-row chunking axis is not bit-exact on Metal, which an fp16 chaos-sensitive \
             schedule amplifies to a different image — see memory_strategy::ATTENTION_SUPPORT"
                .to_owned(),
        ));
    }
    Ok(mlx_gen::attention::AttentionPlan::UNBOUNDED)
}

/// Rung 1: whether this request stages its component residency. `default_staged` is the load-time
/// [`OffloadPolicy`] verdict, which a request that names nothing keeps.
pub(crate) fn stage_residency(req: &GenerationRequest, default_staged: bool) -> bool {
    req.memory
        .map_or(default_staged, |memory| memory.stage_residency)
}

/// The load-time default for `stage_residency`.
pub(crate) fn default_stage_residency(spec: &LoadSpec) -> bool {
    matches!(spec.offload_policy, OffloadPolicy::Sequential)
}

/// The strategy parameters this provider accepts, for a caller that wants the whole domain in one
/// value (the conformance tests and the SceneWorks evidence writer both key off this).
pub fn declared_parameters() -> mlx_gen::gen_core::MemoryStrategyParameters {
    mlx_gen::gen_core::MemoryStrategyParameters {
        decode_tile_edge: None,
        decode_overlap: None,
        // Rungs 2 and 3 are `Missing`, so this provider declares no decode or attention parameter.
        attention_chunk_size: None,
        transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
        transformer_window_component: Some(TRANSFORMER_WINDOW_COMPONENT),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::GenerationMemory;
    use mlx_gen::AdapterSpec;

    /// A packed-tier spec. `quantize` + a snapshot dir whose `unet/config.json` carries the
    /// `quantization` marker is what `load_leaves_blocks_lazy` needs; the weights-free tests below
    /// use the DENSE (no-quantize) shape, which is lazy without touching disk.
    fn spec(shape: LoadShape) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent/kolors-contract".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(shape)
    }

    fn contract(shape: LoadShape) -> MemoryProviderContract {
        weights_free_memory_strategy_contract(crate::MODEL_ID, &spec(shape)).unwrap()
    }

    #[test]
    fn a_deferred_load_publishes_the_full_ladder_and_conforms() {
        let contract = contract(LoadShape::DeferredMaterialization);
        assert!(
            contract.conformance_errors().is_empty(),
            "{:?}",
            contract.conformance_errors()
        );
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "{strategy:?}"
            );
        }
    }

    /// An eager load must publish rung 4 as `Missing` rather than declaring a window it cannot run.
    #[test]
    fn an_eager_load_declares_rung_four_missing() {
        let contract = contract(LoadShape::EagerMaterialization);
        assert!(contract.conformance_errors().is_empty());
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert!(!contract.lifecycle.transformer_window_materialization);
        assert!(contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap()
            .parameters
            .transformer_window_sizes
            .is_empty());
        // Rung 1 is unaffected — the load shape is rung 4's prerequisite, not rung 1's.
        assert_eq!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
    }

    /// **The fail-closed condition that bites Kolors harder than SDXL.** A dense-snapshot adapter
    /// load merges its LoRA into the base weight; a re-materialized block would be the un-adapted
    /// model. Rung 4 must not be declared for it — and on Kolors the only dense tier is `bf16`.
    #[test]
    fn a_merging_adapter_load_cannot_stream() {
        let adapter = AdapterSpec {
            path: "/nonexistent/lora.safetensors".into(),
            scale: 1.0,
            kind: mlx_gen::AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        };
        let dense = spec(LoadShape::DeferredMaterialization);
        let with_adapter = LoadSpec {
            adapters: vec![adapter],
            ..dense.clone()
        };
        assert!(streamable(&dense), "the adapter-free load streams");
        assert!(
            !streamable(&with_adapter),
            "an adapter load over a dense snapshot must NOT stream — the merge is not replayable"
        );
        assert!(!adapters_are_replayable(&with_adapter));

        let contract =
            weights_free_memory_strategy_contract(crate::MODEL_ID, &with_adapter).unwrap();
        assert!(contract.conformance_errors().is_empty());
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
    }

    /// A single-file source has no re-openable per-component snapshot. Kolors rejects one at load
    /// anyway, but the contract must not *declare* the rung for it either.
    #[test]
    fn a_single_file_source_cannot_stream() {
        let fused = LoadSpec::new(WeightsSource::File("/nonexistent/kolors.safetensors".into()))
            .with_load_shape(LoadShape::DeferredMaterialization);
        assert!(!streamable(&fused));
    }

    /// A load-time quantization over a **dense** snapshot packs every block, so a window over it
    /// bounds nothing — it adds a copy on top. `/nonexistent` cannot carry the `quantization`
    /// marker, which is exactly the shape of the upstream `Kwai-Kolors/Kolors-diffusers` snapshot.
    #[test]
    fn a_load_time_quantization_over_a_dense_snapshot_cannot_stream() {
        let dense_q8 = LoadSpec {
            quantize: Some(mlx_gen::Quant::Q8),
            ..spec(LoadShape::DeferredMaterialization)
        };
        assert!(!load_leaves_blocks_lazy(&dense_q8));
        assert!(!streamable(&dense_q8));
        // The control: the same spec without the quantize request stays lazy.
        assert!(load_leaves_blocks_lazy(&spec(LoadShape::DeferredMaterialization)));
    }

    /// **Rungs 2 and 3 are refused on every layer**, and the refusal names why.
    ///
    /// This is the test that keeps a measured-and-rejected rung from drifting back into
    /// reachability: the mechanisms are still reachable (the sweep drives them), so the only thing
    /// standing between `Autoencoder::decode_tiled` and a production render is these refusals.
    #[test]
    fn the_rejected_rungs_are_refused_on_every_layer() {
        let contract = contract(LoadShape::DeferredMaterialization);
        for strategy in [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Missing,
                "{strategy:?} was measured and rejected; it must declare Missing"
            );
            // A `Missing` rung is never engaged, so the cost-order default cannot drag it in behind
            // a rung-4 selection either.
            assert!(
                !contract.engages(MemoryStrategy::BoundedTransformerResidency, strategy),
                "{strategy:?} must not be engaged by the rung-4 cost-order default"
            );
        }
        assert!(!contract.lifecycle.decode_tiling);
        assert!(!contract.lifecycle.attention_chunking);
        assert!(
            contract.pid_decode_routes.is_none(),
            "the PiD route split exists to parameterize rung 2, which is not selectable here"
        );

        // Layer two: the request-side resolvers, which a calibration harness reaches directly.
        let tiled = GenerationRequest {
            memory: Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: Some(DECODE_TILE_EDGE),
                decode_overlap: Some(DECODE_OVERLAP),
                ..Default::default()
            }),
            ..Default::default()
        };
        let err = decode_tiling(&tiled).expect_err("a bounded-decode request must be refused");
        assert!(err.to_string().contains("88/255"), "got: {err}");

        let chunked = GenerationRequest {
            memory: Some(GenerationMemory {
                chunk_attention: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        let err = attention_plan(&chunked).expect_err("a bounded-attention request must be refused");
        assert!(err.to_string().contains("0.00%"), "got: {err}");

        // The controls. Without these, a resolver that refused unconditionally would pass.
        assert!(decode_tiling(&GenerationRequest::default())
            .unwrap()
            .is_none());
        assert!(attention_plan(&GenerationRequest::default()).is_ok());
    }

    /// The rung-4 request-side resolver refuses exactly what admission refuses — the second layer a
    /// harness driving `generate` with a hand-built `GenerationMemory` actually crosses.
    #[test]
    fn the_request_side_window_resolver_refuses_out_of_domain_cadences() {
        let request = |memory: GenerationMemory| GenerationRequest {
            width: 1024,
            height: 1024,
            count: 1,
            memory: Some(memory),
            ..Default::default()
        };
        let windowed =
            |size: Option<u32>, component: Option<TransformerComponent>| GenerationMemory {
                stream_transformer_blocks: true,
                transformer_window_size: size,
                transformer_window_component: component,
                ..Default::default()
            };
        for size in TRANSFORMER_WINDOW_SIZES {
            let resolved = transformer_window(&request(windowed(Some(*size), None)))
                .unwrap()
                .expect("an engaged rung 4 resolves to a window");
            assert_eq!(resolved.size, *size as usize);
            assert_eq!(resolved.component, TRANSFORMER_WINDOW_COMPONENT);
        }
        // Every published scope resolves, and each one reaches exactly the stacks it names. This is
        // the assertion that would have caught a `Both` silently narrowed to `Dit`.
        for component in TRANSFORMER_WINDOW_COMPONENTS {
            let resolved = transformer_window(&request(windowed(Some(1), Some(*component))))
                .unwrap()
                .expect("a published scope must resolve");
            assert_eq!(resolved.component, *component);
            assert_eq!(resolved.dit().is_some(), component.includes_dit());
            assert_eq!(
                resolved.text_encoder().is_some(),
                component.includes_text_encoder()
            );
        }
        // Out-of-domain cadences, chosen to sit *between* and *beyond* the published ones rather
        // than merely far from them: 3/4/6/7/9 are interior gaps a "clamp to the nearest legal
        // value" bug would silently absorb, and 11/28/70 are past the widest sub-stack (28 is the
        // ChatGLM3 depth, the value a TextEncoder-scope confusion would produce).
        for bad in [0_u32, 3, 4, 6, 7, 9, 11, 28, 70] {
            assert!(
                !TRANSFORMER_WINDOW_SIZES.contains(&bad),
                "the negative case list must stay disjoint from the published domain"
            );
            assert!(
                transformer_window(&request(windowed(Some(bad), None))).is_err(),
                "window {bad} must be refused"
            );
        }
        // The request-side default is the TIGHTEST cadence at the DEFAULT scope.
        let defaulted = transformer_window(&request(windowed(None, None)))
            .unwrap()
            .expect("an engaged rung 4 resolves to a window");
        assert_eq!(defaulted.size, TRANSFORMER_WINDOW_SIZE as usize);
        assert_eq!(defaulted.component, TRANSFORMER_WINDOW_COMPONENT);
        assert_eq!(
            TRANSFORMER_WINDOW_SIZE, TRANSFORMER_WINDOW_SIZES[0],
            "the default must be the TIGHTEST published cadence — a default is what a caller gets \
             without asking, so it must be safe across the whole advertised geometry range rather \
             than optimal at the points that happened to be measured"
        );
        assert!(transformer_window(&request(GenerationMemory::default()))
            .unwrap()
            .is_none());
        // All three scopes are published, and the default is the one that pays at the catalog's
        // default tier and geometry.
        assert_eq!(
            TRANSFORMER_WINDOW_COMPONENTS,
            &[
                TransformerComponent::Dit,
                TransformerComponent::TextEncoder,
                TransformerComponent::Both
            ]
        );
        assert_eq!(TRANSFORMER_WINDOW_COMPONENT, TransformerComponent::Dit);
    }

    /// Rung 1's default follows the load-time policy, and a request overrides it in both directions.
    #[test]
    fn staged_residency_is_request_scoped_over_a_load_time_default() {
        let plain = GenerationRequest::default();
        assert!(stage_residency(&plain, true));
        assert!(!stage_residency(&plain, false));
        for (requested, default) in [(true, false), (false, true)] {
            let req = GenerationRequest {
                memory: Some(GenerationMemory {
                    stage_residency: requested,
                    ..Default::default()
                }),
                ..Default::default()
            };
            assert_eq!(stage_residency(&req, default), requested);
        }
    }

    /// Rung 4 declares rung 1 as an engagement prerequisite, so the shared selector refuses a
    /// composition that would bound 0.69 GiB of U-Net weights while 3.06 GiB of ChatGLM3-6B stayed
    /// resident next to them.
    #[test]
    fn rung_four_requires_rung_one_engaged_in_the_same_request() {
        let deferred = contract(LoadShape::DeferredMaterialization);
        let edges: Vec<_> = deferred
            .requires(MemoryStrategy::BoundedTransformerResidency)
            .collect();
        assert!(
            edges.contains(&MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            }),
            "rung 4 must declare the rung-1 edge, got {edges:?}"
        );
        assert!(edges.contains(&MemoryStrategyPrerequisite::LoadShape(
            LoadShape::DeferredMaterialization
        )));
        // And the eager contract publishes no such edge, because it publishes no rung 4.
        let eager = contract(LoadShape::EagerMaterialization);
        assert!(eager.additional_prerequisites.is_empty());
    }

    /// The window domain the shared request scope validates against is the WIDEST sub-stack, and
    /// every published cadence fits inside it.
    #[test]
    fn the_published_window_domain_fits_the_widest_sub_stack() {
        let widest = widest_transformer_stack();
        assert_eq!(widest, 10, "Kolors' deepest Transformer2D holds 10 blocks");
        for size in TRANSFORMER_WINDOW_SIZES {
            assert!(
                *size >= 1 && (*size as usize) <= widest,
                "window {size} must fit the widest sub-stack"
            );
        }
        // The tightest cadence must actually bound something on the SHALLOWEST sub-stack too,
        // otherwise the rung is inert on five of the eleven stacks by construction.
        assert!(
            (TRANSFORMER_WINDOW_SIZES[0] as usize) < 2,
            "the tightest published cadence must bound the 2-deep sub-stacks"
        );
        // Ascending and duplicate-free, and the ordering is load-bearing rather than cosmetic:
        // three separate invariants read a position and mean a superlative by it.
        assert!(
            TRANSFORMER_WINDOW_SIZES.windows(2).all(|w| w[0] < w[1]),
            "the domain must be strictly ascending: {TRANSFORMER_WINDOW_SIZES:?}"
        );
        assert_eq!(
            *TRANSFORMER_WINDOW_SIZES.last().unwrap() as usize,
            widest,
            "the widest cadence must equal the deepest sub-stack — one re-open per stack per step"
        );
    }

    /// **Kolors is not SDXL, and the config is where that starts.**
    ///
    /// `UNetConfig::kolors` differs from `sdxl_base` only in
    /// `projection_class_embeddings_input_dim`, which is precisely why the rung-4 machinery
    /// transfers — and precisely why a reader might assume the *numbers* do too. Pin the shape so a
    /// future config change that alters the windowable topology cannot slip past the shared
    /// mechanism silently.
    #[test]
    fn the_kolors_unet_topology_is_the_sdxl_one_and_the_conditioning_tower_is_not() {
        let kolors = mlx_gen_sdxl::UNetConfig::kolors();
        let sdxl = mlx_gen_sdxl::UNetConfig::sdxl_base();
        assert_eq!(
            kolors.transformer_layers_per_block,
            sdxl.transformer_layers_per_block
        );
        assert_eq!(kolors.down_block_types, sdxl.down_block_types);
        assert_eq!(
            kolors.projection_class_embeddings_input_dim,
            Some(5632),
            "Kolors' add-embedding takes the ChatGLM3 pooled stream, not SDXL's CLIP pair"
        );
        assert_ne!(
            kolors.projection_class_embeddings_input_dim,
            sdxl.projection_class_embeddings_input_dim
        );
        // Level 0 carries no attention, so `transformer_layers_per_block[0]` describes nothing.
        assert!(!kolors.down_block_types[0].contains("CrossAttn"));
        let attention_levels: Vec<i32> = kolors
            .transformer_layers_per_block
            .iter()
            .enumerate()
            .filter(|(i, _)| kolors.down_block_types[*i].contains("CrossAttn"))
            .map(|(_, layers)| *layers)
            .collect();
        assert_eq!(
            attention_levels,
            vec![2, 10],
            "only levels 1 and 2 own Transformer2D sub-stacks"
        );
        // The conditioning tower is the divergence that makes this family's evidence its own.
        assert_eq!(
            crate::chatglm3::ChatGlmConfig::chatglm3_6b().num_layers,
            28,
            "the ChatGLM3-6B stack depth quoted by TRANSFORMER_WINDOW_COMPONENTS"
        );
    }

    /// The calibration fingerprint must not be SDXL's, even though the U-Net module is shared.
    #[test]
    fn the_calibration_fingerprint_is_not_shared_with_sdxl() {
        assert!(MEMORY_CALIBRATION_FINGERPRINT.starts_with("kolors-"));
        assert_ne!(
            MEMORY_CALIBRATION_FINGERPRINT,
            "sdxl-mlx-unet-shared-ladder-v2"
        );
        let contract = contract(LoadShape::DeferredMaterialization);
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            MEMORY_CALIBRATION_FINGERPRINT
        );
    }
}
