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
//! | 2 Bounded decode | **Missing** (measured — [`DECODE_SUPPORT`]) | [`Autoencoder::decode_tiled`](mlx_gen_sdxl::Autoencoder) bounds the request 16.041 → 9.998 GiB (−37.67%) and clears the drift bar at 1024²/1280², but fails it at 1536²/2048² and `decode_tile_edges` has no geometry axis |
//! | 3 Bounded attention | **Missing** (measured — [`ATTENTION_SUPPORT`]) | [`mlx_gen::attention::sdpa_budgeted_bhsd`] reaches every site, moves the peak **0.00% at BOTH scopes** — the request and the U-Net seam — and is not bit-exact on its query-row axis |
//! | 4 Bounded transformer residency | Implemented (streamable loads), **two scopes** | [`mlx_gen::block_residency::run_windowed`] over the U-Net's eleven `Transformer2D` sub-stacks (70 blocks) **and** over ChatGLM3-6B's 28 `GlmBlock`s |

//!
//! ## What this family actually buys, measured (q4 unless stated, Apple/Metal, 1024², 4 steps)
//!
//! | composition | request peak | vs baseline | ms/step | output |
//! |---|---:|---:|---:|---|
//! | resident | 19.5296 GiB | — | 849 | — |
//! | + rung 1 | **16.0411** | **−17.86%** | 883 | byte-identical |
//! | + rung 4 `Dit`, cadence 1 (default) | **14.8840** | **−7.21%** vs staged | 3745 | byte-identical |
//! | + rung 4 `Dit`, cadence 10 (selectable) | **14.8840** | **−7.21%** vs staged | 1507 | byte-identical |
//! | bf16 @ 512², + rung 4 `Both` | **4.5436** | **−60.02%** vs staged | 2043 | byte-identical |
//!
//! Per tier at 1024², rung 4 `Dit` against that tier's own staged control:
//! q4 **−7.21%**, q8 **−12.72%**, bf16 **−21.37%** — the saving is the whole `transformer_blocks`
//! weight set, so what changes across tiers is the numerator.
//!
//! ## Ownership
//!
//! This file declares *structure and parameter domains only*. Measured coefficients, envelopes and
//! per-tier peaks live in SceneWorks generated evidence keyed by
//! [`MEMORY_CALIBRATION_FINGERPRINT`]; the worker owns live-budget accounting and least-cost
//! selection. The scope below is defense in depth: it can reject a selection, never substitute one.

use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};
use mlx_gen::gen_core::{
    standard_memory_strategy_safety_check, Error as CoreError, MemoryBackendRealization,
    MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryCalibrationIdentity, MemoryComponentKind,
    MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent,
    MemoryRunContext, MemorySafetyDecision, MemoryStrategy, MemoryStrategyPrerequisite,
    MemoryStrategySupport, ResidentRequestMemory, Result as CoreResult, TransformerComponent,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{GenerationRequest, LoadShape, LoadSpec, OffloadPolicy, WeightsSource};

/// The decode tile edge a rung-2 request would have defaulted to, kept as the sweep's anchor.
///
/// Retained as evidence, not as a published domain — see [`DECODE_SUPPORT`].
pub const DECODE_TILE_EDGE: u32 = 768;

/// The decode overlap paired with every swept edge, in output pixels.
pub const DECODE_OVERLAP: u32 = 256;

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
/// The mechanism is implemented and it works. Kolors decodes through [`mlx_gen_sdxl::Autoencoder`] —
/// the same `sdxl-vae-fp16-fix` `AutoencoderKL` SDXL uses — so `decode_tiled` runs the globally-scoped
/// head (denormalize → `post_quant_conv` → `conv_in` → mid resnets → mid **self-attention**) once on
/// the full latent and tiles only the full-resolution upsample tail.
///
/// **It is also, by a wide margin, the biggest lever on this family's ladder.** At 1024² q4 it takes
/// the whole staged request from **16.0412 → 9.9981 GiB (−37.67%)** for **+10% wall clock**
/// (925 → 1018 ms/step) — against rung 4's −7.21% for +4.2× at the same tier and geometry
/// (`the_withheld_decode_geometry_is_priced_at_the_request_level`). It is withheld anyway, and the
/// size of the prize is exactly why the withholding argument had to be measured rather than
/// asserted.
///
/// ## The candidate is real, and at 1024² it CLEARS the bar
///
/// Swept against the **exact untiled decode of the same latent**, on the *production* latent — what
/// the denoiser actually hands the decode phase, not a re-encoded finished image — at q4 1024²
/// (`decode_tile_mechanism_sweep`, max Δ per channel out of 255):
///
/// | edge | overlap 64 | 128 | 192 | 256 |
/// |---:|---:|---:|---:|---:|
/// | 896 | 80 | 69 | 62 | 49 |
/// | 768 | 59 | 59 | 44 | **38** |
/// | 640 | 55 | 45 | 46 | 44 |
/// | 512 | 61 | 68 | 62 | 60 |
/// | 448 | 74 | 80 | 78 | 68 |
/// | 384 | 101 | 85 | 70 | 69 |
/// | 320 | 92 | 108 | 86 | 83 |
/// | 256 | 120 | 98 | 100 | 103 |
///
/// (mean Δ ranges 0.43 → 2.78.) The best cell — edge [`DECODE_TILE_EDGE`] at overlap
/// [`DECODE_OVERLAP`] — drifts **38/255**, comfortably inside [`DECODE_DRIFT_BAR`]. **On the 1024²
/// evidence alone this candidate should ship.** Saying so plainly matters: a reader who assumed the
/// sweep simply failed would look for the wrong repair.
///
/// **The instrument is the harsher of the two available, deliberately.** An earlier revision of this
/// sweep decoded a latent obtained by re-encoding a finished image, whose statistics have already
/// been through the VAE round trip; that instrument reads **29/255** at edge 768 / overlap 192 where
/// the production latent reads 44 at the same geometry. A user gets the production latent, so the
/// absolute bar is judged on it.
///
/// ## What withholds it: the datum is a property of the GEOMETRY, not of the tile edge
///
/// `MemoryParameterRanges::decode_tile_edges` is an absolute pixel domain with no geometry axis, so
/// publishing 768 publishes it at everything `crate::registry::descriptor` advertises — and that is
/// `max_size: 2048`. Re-swept across that range at the same overlap, on a **rendered production
/// latent at each output size**
/// (`no_single_decode_tile_edge_clears_the_bar_across_the_advertised_output_range`):
///
/// | output | tile covers | max Δ | vs the 48/255 bar |
/// |---:|---:|---:|---|
/// | 1024² | 75.0% | **38** | clears |
/// | 1280² | 60.0% | **26** | clears |
/// | 1536² | 50.0% | 69 | **fails** |
/// | 2048² | 37.5% | 53 | **fails** |
///
/// The GroupNorm mechanism predicts this shape: every `ResnetBlock2D` in `up_blocks` and the final
/// `conv_norm_out` computes statistics over the spatial extent it is handed, so what governs the
/// drift is the tile's **fraction of the image** — that is what sets how far a tile's statistics can
/// sit from the global ones. Edge 768 covers 75% of a 1024² output and 37.5% of a 2048² one. No
/// fixed pixel edge holds a constant fraction across a 2× range, so no fixed pixel edge is
/// admissible across this provider's advertised range. A selector would size a fit off `[768]`,
/// choose it at 1536², and get a visibly seamed image.
///
/// The range sweep is run on **rendered** latents rather than synthetic ones for the same reason the
/// mechanism sweep is: a unit-normal latent is a materially friendlier instrument (35/255 at 1024²
/// against the rendered 38), and a rung admitted on the friendly instrument would be admitted on
/// evidence no user ever sees.
///
/// ## How this differs from SDXL, which shares the decoder
///
/// SDXL withholds rung 2 too, and it is worth being precise about where the two families agree,
/// because the shared `AutoencoderKL` makes it tempting to inherit:
///
/// * **SDXL's 1024² cell fails on the production latent (84/255); Kolors' passes (38/255).** Same
///   decoder weights, same head/tail split, different denoiser and different conditioning
///   distribution. Kolors' rung 2 is the *better* candidate of the two.
/// * **Both fail the range argument, at different edges and different magnitudes** — SDXL at 896
///   with 64/120/77 across 1280²/1536²/2048², Kolors at 768 with 26/69/53. Kolors even clears one
///   extra output size.
///
/// So the verdict transfers and not one number does. Publishing SDXL's figures here would have
/// misstated the best geometry (896 vs 768), the best overlap (192 vs 256), the prize (−16.51% vs
/// −37.67%) and which output sizes clear.
///
/// What would unlock it is a geometry-relative tile parameter (a fraction of the output rather than
/// a pixel edge), or a decoder whose tail normalizes over the full extent. Both are contract-level
/// changes, and neither is this story's — tracked as sc-17678.
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
/// 1. **It does not move the peak, at either scope.** End to end
///    (`attention_chunking_is_measured_against_the_rung_two_top`, which re-assembles the staged
///    request from the public entry points because the production resolver refuses a
///    bounded-attention selection):
///
///    | steps | unchunked | chunked | Δ peak | max Δ px | mean Δ px |
///    |---:|---:|---:|---:|---:|---:|
///    | 1 | 16.0412 GiB | 16.0410 GiB | **−0.00%** | 16/255 | 0.28 |
///    | 4 | 16.0411 GiB | 16.0410 GiB | **−0.00%** | 212/255 | 0.70 |
///
///    Measured instead at the U-Net seam, where the effect is not diluted by the rest of the request
///    (`attention_chunking_is_measured_at_the_unet_seam`), one CFG-batched 1024² forward goes
///    **5.3652 → 5.3652 GiB, +0.00%** — the two plans peak identically to the millibyte, reproduced
///    across three runs. So the budget is not merely diluted at the request level: there is **no
///    scope on this family at which the query-row budget changes the high-water mark**, which is a
///    cleaner statement of the withholding reason than a diluted one would have been.
///
///    That is consistent with the hazard `mlx_gen::attention`'s own docs name — when q/k/v are
///    already materialized and pinned, a query-row budget has nothing left to bound — and on a U-Net
///    that is the normal case, because the conv/resnet trunk between every attention has already
///    broken the lazy graph `eval_per_chunk` exists to cut. The epic is explicit that a rung which
///    does not move the peak is not a saving.
///
///    **Kolors was the more promising candidate of the two U-Nets and still failed.** Its
///    cross-attention memory is the ChatGLM3 context — 256 tokens against SDXL's 77 — so the
///    cross-attention score tensors are 3.3× larger here, which is exactly where a query-row budget
///    would be most likely to pay. It does not: the seam is flat at the 3.3×-larger key axis too.
///
///    **Correction (sc-15521 review).** An earlier revision of this file published the seam as
///    **4.7672 → 5.3651 GiB, +12.54%** and argued from it that chunking *adds* transients here and
///    that the regression was "worse than SDXL's +5.1%". Both statements were wrong and the number
///    was an artifact: `attention_chunking_is_measured_at_the_unet_seam` was the only
///    peak-publishing test in the harness with no discarded warm-up row, and MLX's first `run` in a
///    process reads ~4.6–4.8 GiB *whichever plan it executes*. With the warm-up in place the pair
///    reads 5.3652 → 5.3652. Nothing about the `Missing` verdict changes — the end-to-end row was
///    always warmed and the bit-exactness reason (2) is independent — but the "chunking adds
///    transients" narrative it supported is withdrawn.
/// 2. **The output moves — and that is a property of the CHOSEN AXIS, not of Kolors.** The
///    arithmetic is exactly preserved: [`sdpa_budgeted_bhsd`](mlx_gen::attention::sdpa_budgeted_bhsd)
///    chunks the **query axis only**, so each chunk is a complete fused SDPA over the full k/v with
///    no accumulator and no running max, and the class of bug that would make chunked softmax
///    genuinely wrong is structurally unwritable. What changes is which reduced-precision Metal
///    matmul specialization MLX dispatches for the `[.., block, Sk]` product — the sc-2338 parity
///    class.
///
///    `gen_core::attention_budget` carries a **second** axis for exactly this reason:
///    `AttentionBudget::head_chunks` narrows complete heads and *preserves bit identity*, and its
///    docs say outright that the two axes "share a score budget, but not a numerical contract".
///    `sdpa_budgeted_bhsd` never uses it. So the honest statement is: **the query-row axis is not
///    bit-exact on Metal, and Kolors is exposed to that** — it runs fp16 through a leading-Euler
///    schedule across 140 attention sites. Measured at **one** step, where the sampler cannot
///    amplify anything, the raw per-forward divergence is `max|Δeps| = 3.929e-3` and the image moves
///    by 16/255; by four steps it is 212/255.
///
///    A head-axis implementation would be bit-exact by construction; what it would still have to
///    beat is reason 1, which is the binding one.
///
/// The one-step control is what separates both claims from a wiring bug. It is also why the verdict
/// is stated per family per backend: Z-Image measures −1.7% on its denoise phase with a
/// bit-identical image on the same primitive.
pub const ATTENTION_SUPPORT: MemoryStrategySupport = MemoryStrategySupport::Missing;

/// The production transformer-window domain for rung 4: how many consecutive blocks of **each**
/// windowed stack are held materialized at once.
///
/// The domain is a **cadence**, shared by both published scopes. On `Dit` it applies independently
/// to eleven `Transformer2D` sub-stacks of two depths (`[2, 2, 10, 10, 10, 10, 10, 10, 2, 2, 2]`, 70
/// blocks); on `TextEncoder` it applies to the ChatGLM3-6B tower's single 28-block stack.
/// [`BlockPlan::new`](mlx_gen::block_residency::BlockPlan::new) clamps a window to its stack's depth,
/// so a cadence wider than 2 degrades the five 2-deep sub-stacks to fully resident rather than
/// erroring — which is why the domain is stated once and not per depth. 10 is the widest cadence
/// that means anything on `Dit`: it holds one *whole* deep sub-stack, i.e. one re-open per stack per
/// step instead of ten. On `TextEncoder` a cadence of 10 means three re-opens per request instead of
/// 28.
///
/// ## Measured: four cadences, one saving
///
/// `transformer_window_sweep_and_streamed_output_identity` (`KOLORS_WINDOW_PROBE_TIER` /
/// `KOLORS_WINDOW_PROBE_SIZE` drive the off-default rows), `Dit` scope, fresh bundle per row, every
/// row byte-identical to its resident control. Request peak in GiB:
///
/// | tier | output | staged control | cadence 1 | 2 | 5 | 10 | spread |
/// |---|---:|---:|---:|---:|---:|---:|---:|
/// | **q4** | **1024²** | 16.0412 | 14.8840 | 14.8840 | 14.8840 | 14.8840 | **0.00%** |
/// | q4 | 512² | 5.6193 | 4.6924 | 4.6924 | 4.6924 | 4.6924 | **0.00%** |
///
/// **Read this as an observation, not as a gated invariant.** Exactly one row is asserted — q4 /
/// 1024², the sweep's default tier and geometry, which is also the catalog's default tier. The 512²
/// row comes from re-running under `KOLORS_WINDOW_PROBE_SIZE`, and in that mode the flatness and
/// wall-clock assertions are reported rather than asserted. And even the asserted row is asserted by
/// an `#[ignore]`d real-weight test, so it is gated by a human running it against cached weights,
/// not by CI.
///
/// Wall clock falls monotonically with cadence (1024² q4: 3745 → 1507 ms/step, **2.5× cheaper**;
/// 512² q4: 5347 → 990, **5.4×**). Absolute numbers track machine load — the same rows measured
/// 4268 → 1621 on a busier pass with **the peak column unchanged to the millibyte** — so the ratio
/// is the quantity worth quoting and the wall-clock assertions in the sweep are deliberately loose.
///
/// ### Rung 4's *relative* latency cost grows as the output shrinks
///
/// Against each geometry's own unwindowed control, at the shipped cadence of 1: **4.2× at 1024²,
/// 10.5× at 512².** That is arithmetic, not a regression appearing at small outputs — a re-open is
/// weight I/O, so its cost is fixed and area-independent, while the control's per-step forward
/// scales with area. It is also the strongest argument a selector has for *choosing* a wide cadence:
/// at 512² q4, cadence 10 lands on the identical 4.6924 GiB peak and is 5.4× faster.
///
/// ## The mechanism is PHASE SEPARATION, and the arithmetic says so at every tier
///
/// The request peak is a `max` over phases, and rung 4 only bounds weights *inside one of them*.
/// Read the saving against the U-Net safetensors headers (summing `data_offsets`,
/// `transformer_blocks.*` against the whole file) at 1024²
/// (`the_rung_four_saving_is_the_whole_transformer_block_weight_set`,
/// `every_advertised_tier_loads_and_publishes_the_ladder`):
///
/// | tier | U-Net total | `transformer_blocks` | one deep block | measured saving | saving ÷ block set |
/// |---|---:|---:|---:|---:|---:|
/// | q4 | 1.7995 GiB | 1.1468 GiB | 0.0182 GiB | 1.1572 | **1.009** |
/// | q8 | 2.8448 | 2.1638 | 0.0344 | 2.1737 | **1.005** |
/// | bf16 | 4.8046 | 4.0706 | 0.0647 | 4.0668 | **0.999** |
///
/// **The saving is the entire block weight set** — every tier, to within a percent, and *identically*
/// at 1024² and 512² on q4 where the activation working sets differ four-fold. That is
/// arithmetically incompatible with any window being resident when the peak is taken. Eleven
/// `Transformer2D` sub-stacks run in sequence and `run_windowed` releases one before opening the
/// next, so at most one cadence-worth of blocks is materialized at any instant: if the peak occurred
/// during the windowed forward, the saving could be **at most** `block set − w × (one deep block)` —
/// 1.1285 GiB on q4 at cadence 1. The measured 1.1572 GiB exceeds it.
///
/// So **zero window weights are resident at the peak moment: the peak is not in the windowed forward
/// at all.** It is the **decode**. Rung 1's staged schedule releases the ChatGLM3-6B tower but *not*
/// the U-Net (`KolorsHeavyOwned` is one bundle and the decode runs inside the same render closure),
/// so the decode transient stacks on the full resident remainder. Rung 4 lowers the *floor* that
/// transient stacks on, by the whole block set, and never touches the transient itself. Cadence is
/// then invisible to the peak for the same reason the choice of denoise step count is.
///
/// Two independent corroborations, both in this file. [`DECODE_SUPPORT`] measures that tiling **the
/// decode alone** moves the request peak −37.67% — no change confined to the forward could do that
/// if the forward were peak-bearing. And [`TRANSFORMER_WINDOW_COMPONENTS`] measures the phase split
/// directly.
///
/// ### The condition, checked rather than inferred
///
/// Flatness holds **only while** `decode transient + resident remainder` exceeds the windowed
/// forward's own peak at the *widest* published cadence. The decode transient scales with output
/// area; the windowed forward's peak scales with weights and is geometry-insensitive. So the
/// inequality has to reverse as the output shrinks, and the first advertised geometry where it does
/// is the boundary of the flat region. **On SDXL that boundary is inside the advertised range** —
/// its 512² q8 row opens to a 5.7% spread and goes non-monotonic. **On Kolors it is not**: at the
/// advertised `min_size` the spread is still 0.00% and rung 4 still bounds the peak by 16.49%
/// (`the_cadence_flatness_condition_is_checked_not_assumed`). Kolors' resident remainder after rung
/// 1 is smaller relative to its decode transient, which keeps the decode ahead further down.
///
/// One measurement hazard is worth recording because it produced a convincing false positive during
/// this story: the **first** measured row in a process reads a peak biased by MLX's cold allocator.
/// A draft of the flatness test measured a windowed row first and read 4.4632 GiB where the sweep
/// reads 4.6924 — a 4.9% phantom spread that looked exactly like the flat region breaking at
/// `min_size`. Every peak in this file now has a discarded warm-up row ahead of it.
///
/// ### This is coupled to rung 2, and the coupling runs the wrong way
///
/// Every row above was measured with [`DECODE_SUPPORT`] `Missing`, i.e. with the decode transient at
/// **full size** — precisely the term keeping the forward off the peak. A story that ships a bounded
/// decode *lowers that term deliberately*, shrinking the flat region toward the small-output end and
/// potentially eliminating it. **"Widening the cadence is free" is not a property of this family; it
/// is a property of this family with rung 2 withheld.** Publishing rung 2 invalidates every rung-4
/// row here and must re-measure the cadence domain in the same change — see
/// [`MEMORY_CALIBRATION_FINGERPRINT`], which must be bumped with it.
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 5, 10];

/// The transformer window rung 4 executes at when a request names none.
///
/// **The tightest cadence**, which is what every other MLX adopter on this ladder defaults to.
///
/// The honest counter-argument is recorded because it is a good one: on every configuration measured
/// here cadence 10's peak equals cadence 1's **to the millibyte**, while cadence 10 is 2.5–5.4×
/// faster. Read as a table of two rows, 10 dominates.
///
/// It is still not the default, and the reason is what the flat column *depends on* rather than what
/// its endpoints happen to tie at. Flatness is the consequence of an inequality
/// ([`TRANSFORMER_WINDOW_SIZES`]) that this family satisfies at both measured geometries and that
/// **SDXL — the same U-Net, the same rung, the same shared primitive — violates at its own
/// `min_size`**, where the spread opens to 5.7% and goes non-monotonic. A property that fails on the
/// nearest sibling is not one to extrapolate from two samples of a three-tier × 512²-to-2048² space.
/// A default is the value a caller gets *without asking*, so it has to be the bound rung 4 can
/// always make good on.
///
/// The other three cadences remain published and selectable, and at 1024² a selector that does not
/// need the tighter bound should absolutely pick 10 and take the 2.5× wall-clock saving; it just has
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
/// Reproduction: on a re-run, seven of the nine cells return to four decimals; the two 512² cells at
/// `q4` and `q8` moved by ~4% (5.620 → 5.850 and 6.894 → 6.665 on the request side). Those two are
/// the first end-to-end row of their kind per tier — the same cold-allocator sensitivity the
/// harness's warm-up exists for, one level down — and they carry the smallest peaks in the table, so
/// the absolute movement is small. **No cell's verdict moves**: the two phases are 1.5–5× apart
/// everywhere except the one conditioning-bearing cell.
///
/// **Instrument scope, stated plainly: eight of these nine cells are HARNESS-ONLY.** Every cell in
/// this table is produced by `measure_end_to_end_phased`, a hand-rolled re-assembly of the staged
/// request from the crate's public entry points — necessary, because `generate` exposes no
/// per-phase peak and so cannot supply a conditioning/request split at all.
/// `the_end_to_end_reassembly_reproduces_the_real_generate_peak` binds that re-assembly to
/// production, but it binds it at **one cell (q4, 1024²) and only on the whole-request peak**. The
/// conditioning-phase split — the left-hand number in every cell — is bound to production
/// **nowhere**.
///
/// What that does and does not undermine: the *decisive* cell is corroborated by production rows, so
/// the headline below stands on more than the harness — `the_text_encoder_window_bounds_the_
/// conditioning_bearing_cell` drives the real `generate` at bf16 512² and measures the `TextEncoder`
/// scope actually moving the request peak, which is only possible if the conditioning phase really
/// does carry it there. The other eight cells have no such corroboration and should be read as
/// harness evidence for the *shape* of the finding, not as production measurements.
///
/// **One cell of nine is conditioning-bearing: `bf16` at the advertised `min_size`.** There the
/// request peak *is* the ChatGLM3 residency, and a text-encoder window moves it. Measured at that
/// cell against its staged control of 11.3644 GiB, at the shipped cadence
/// (`the_text_encoder_window_bounds_the_conditioning_bearing_cell`, every row byte-identical):
///
/// | scope | request peak | vs control | ms/step |
/// |---|---:|---:|---:|
/// | `Dit` | 11.3644 GiB | **+0.00%** | 1860 |
/// | `TextEncoder` | 8.8396 | **−22.22%** | 542 |
/// | `Both` | 4.5436 | **−60.02%** | 2043 |
///
/// **The `Dit` scope buys exactly nothing there** — a correct, complete bound on a phase that is not
/// the maximum, which is the epic's definition of a phase win rather than a saving. A provider that
/// declared `Dit` only, as SDXL does and as every prior adopter does, would have shipped a rung that
/// is measurably inert at an advertised cell while a −60.02% composition sat unreachable behind an
/// unimplemented enum variant. That is the concrete cost of inheriting a sibling's scope decision,
/// and it is why this story implemented `crate::block_stream` rather than copying SDXL's
/// `Dit`-only declaration.
///
/// It is also a cell a caller can genuinely land on: `bf16` is an advertised tier and 512² is
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
/// differs by 2.2× on the same instrument (SDXL's production latent drifts 84/255 at its best
/// geometry, Kolors' 38/255 at its own). Sharing a fingerprint would let a selector reuse SDXL's
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
        resident_overlay_components(spec)?,
    )
}

/// The **auxiliary networks this load keeps resident alongside the base three**, priced load-exact
/// (sc-15839).
///
/// Kolors is not a bare txt2img provider: it advertises a ControlNet-pose route, an IP-Adapter
/// route, a **combined strict-pose tier that loads two at once**, and a PiD decode overlay. Every
/// one of those is loaded by `registry::load_heavy_owned` and held for the life of the generator,
/// and until this function existed all of them were declared at **zero**. The shared validator only
/// cross-checks the overlay legs `if overlay_bytes > 0`, so a zero passed silently — and a rung-1 or
/// rung-4 selection admitted for a ControlNet/IP/PiD load under-predicted peak by the whole overlay
/// set.
///
/// Each projection mirrors what the loader actually does, not what the file stores:
///
/// * **ControlNet** — `mlx_gen_sdxl::load_controlnet` casts to the compute dtype (fp16) unless the
///   checkpoint is already packed, so a dense branch is priced at 2 bytes/element. The shared
///   primitive independently forces `Stored` for a packed triple, so a pre-quantized branch is
///   counted at its code size rather than projected a second time.
/// * **IP-Adapter** — two files under the snapshot (`image_encoder/model.safetensors` and
///   `ip_adapter_plus_general.safetensors`), both cast the same way. Both are resident: the encoder
///   and resampler live on the generator, and the K/V pairs are installed into the U-Net. A
///   `WeightsSource::File` IP-Adapter is rejected by `load` before it reaches here and loads nothing,
///   so it is priced at nothing.
/// * **PiD** — the student checkpoint and the Gemma caption encoder, both loaded verbatim
///   (`Weights::from_file` / `from_dir`, no cast), so `Stored` is exact.
///
/// **The PiD carve-out, and why it is narrower here than on z-image.** The sibling providers exclude
/// a PiD overlay outright, on the reasoning that its presence on a `LoadSpec` does not mean the
/// request selected it. That reasoning is sound *for a Sequential load* — `registry::build_residency`
/// passes `req.use_pid` per request there, so a non-PiD generate never materializes the student. It
/// is **false for a Resident load**: that arm passes `use_pid = true` unconditionally, so
/// `Residency::ensure_warm_locked` loads the PiD superset once and every subsequent request runs
/// with it resident whether or not it asked for it. Pricing it at zero under `Resident` would
/// under-declare the ordinary path rather than avoid overstating it, so the carve-out is applied to
/// exactly the policy where the overlay really is request-selected.
fn resident_overlay_components(spec: &LoadSpec) -> CoreResult<Vec<MemoryResidentComponent>> {
    // Matches `load_controlnet` / `load_kolors_ip_adapter`: dense payloads are cast to the fp16
    // compute dtype (2 bytes/element); an already-packed triple is left alone, which the shared
    // primitive handles on its own.
    let compute_dtype = |_: &_| ResidentProjection::Bfloat16;
    let mut components = Vec::new();

    if let Some(source) = &spec.control {
        let path = match source {
            WeightsSource::Dir(path) | WeightsSource::File(path) => path,
        };
        push_overlay(
            &mut components,
            CONTROL_COMPONENT_ID,
            MemoryComponentKind::ControlBranch,
            projected_safetensors_bytes(path, compute_dtype)?,
        );
    }

    // Only a `Dir` IP-Adapter loads; `load` rejects a `File` one before any weights are opened.
    if let Some(WeightsSource::Dir(root)) = &spec.ip_adapter {
        let encoder = projected_safetensors_bytes(
            root.join("image_encoder/model.safetensors"),
            compute_dtype,
        )?;
        let adapter = projected_safetensors_bytes(
            root.join("ip_adapter_plus_general.safetensors"),
            compute_dtype,
        )?;
        push_overlay(
            &mut components,
            IP_ADAPTER_COMPONENT_ID,
            MemoryComponentKind::IpAdapter,
            encoder.saturating_add(adapter),
        );
    }

    // See the carve-out above: priced only where the load-time policy makes it unconditional.
    if spec.offload_policy == OffloadPolicy::Resident {
        if let Some(pid) = &spec.pid {
            let (WeightsSource::Dir(checkpoint) | WeightsSource::File(checkpoint)) =
                &pid.checkpoint;
            let (WeightsSource::Dir(gemma) | WeightsSource::File(gemma)) = &pid.gemma;
            // `PidEngine::load` prefers Gemma's merged single file and falls back to the shard dir.
            let merged = gemma.join(mlx_gen_pid::engine::GEMMA_MERGED_FILE);
            let gemma_source = if merged.is_file() {
                merged
            } else {
                gemma.clone()
            };
            let student = projected_safetensors_bytes(checkpoint, |_| ResidentProjection::Stored)?;
            let caption =
                projected_safetensors_bytes(&gemma_source, |_| ResidentProjection::Stored)?;
            push_overlay(
                &mut components,
                PID_COMPONENT_ID,
                // `AdapterStack` is the closest existing kind for an auxiliary network installed
                // beside the base model's own transformers. What the contract arithmetic actually
                // consumes is `MemoryComponentKind::is_auxiliary()`, which is true for it; a
                // dedicated `DecoderOverlay` variant would be a shared-contract change for a naming
                // nicety, so it is deliberately not made here.
                MemoryComponentKind::AdapterStack,
                student.saturating_add(caption),
            );
        }
    }

    Ok(components)
}

/// Record one overlay, skipping a zero. The shared validator refuses a declared component with zero
/// bytes, and a component that measured zero is not evidence of residency anyway.
fn push_overlay(
    into: &mut Vec<MemoryResidentComponent>,
    id: &str,
    kind: MemoryComponentKind,
    resident_bytes: u64,
) {
    if resident_bytes == 0 {
        return;
    }
    into.push(MemoryResidentComponent {
        id: id.to_owned(),
        kind,
        resident_bytes,
        // No published rung bounds an overlay on this family: rung 4's window covers the U-Net's
        // `Transformer2D` sub-stacks and ChatGLM3's `GlmBlock`s, and nothing else.
        bounded_by: None,
        residency: MemoryComponentResidency::WholeRender,
    });
}

/// Stable provider-local ids for the overlays [`resident_overlay_components`] declares.
const CONTROL_COMPONENT_ID: &str = "kolors.controlnet.pose";
const IP_ADAPTER_COMPONENT_ID: &str = "kolors.ip_adapter.plus_general";
const PID_COMPONENT_ID: &str = "kolors.pid.student_and_caption_encoder";

/// Declaration-equivalent contract used by weights-free registry conformance. Structure, parameter
/// domains and prerequisites are identical; only the measured asset facts are absent.
pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    // No overlays either: sizing one means opening its checkpoint, and this path exists precisely to
    // produce the declaration without touching a weight file.
    contract_with_asset_facts(provider_id, spec, 0, 0, 0, Vec::new())
}

fn contract_with_asset_facts(
    provider_id: &str,
    spec: &LoadSpec,
    conditioning_bytes: u64,
    transformer_bytes: u64,
    decoder_bytes: u64,
    overlays: Vec<MemoryResidentComponent>,
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
    let overlay_bytes = overlays.iter().fold(0_u64, |total, component| {
        total.saturating_add(component.resident_bytes)
    });
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    contract.formula = if overlays.is_empty() {
        MemoryFormulaKind::PhaseEnvelope { phases, variables }
    } else {
        // The overlay set is only a variable of the peak when there IS one, so a plain txt2img load
        // keeps the exact envelope it declared before sc-15839 rather than gaining a variable that
        // never varies.
        variables.push(MemoryFormulaVariable::OverlayBytes);
        MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables,
            resident_components: overlays,
        }
    };
    contract.asset_facts.conditioning_bytes = conditioning_bytes;
    contract.asset_facts.transformer_bytes = transformer_bytes;
    contract.asset_facts.decoder_bytes = decoder_bytes;
    // `base_bytes` is the three BASE components and must exclude the overlay, which the shared
    // validator checks; `total_resident_bytes()` is what adds the two together for an adopter.
    contract.asset_facts.base_bytes = conditioning_bytes
        .saturating_add(transformer_bytes)
        .saturating_add(decoder_bytes);
    contract.asset_facts.overlay_bytes = overlay_bytes;
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
        // so the composition is not merely suboptimal, it is inverted: at q4 the `Dit` scope would
        // bound 1.147 GiB of U-Net blocks while 3.984 GiB of encoder sat next to them, and at bf16
        // 4.071 against 11.630. Declaring the edge lets the
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
         Missing). The tiled decode was measured and withheld — NOT because no geometry works: its \
         best (edge {DECODE_TILE_EDGE} overlap {DECODE_OVERLAP}) clears the \
         {DECODE_DRIFT_BAR}/255 sibling bar at 1024² with 38/255 and at 1280² with 26/255, and \
         would have bought -37.67% of the request peak for +10% wall clock. It fails at 1536² \
         (69/255) and 2048² (53/255), because the upsample tail's GroupNorms normalize over each \
         tile's own crop — so what bounds the drift is the tile's FRACTION of the output, and a \
         fixed pixel edge cannot hold that across an advertised range up to 2048². Requested edge \
         {edge:?} overlap {overlap:?}; see memory_strategy::DECODE_SUPPORT for the full sweep."
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
                 Missing): it moves the peak 0.00% at both measured scopes (the request and the \
                 U-Net seam), and the query-row chunking axis is not bit-exact on Metal, which an \
                 fp16 chaos-sensitive schedule amplifies to a different image — see \
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
        mlx_gen::gen_core::standard_memory_behavior_context(
            contract,
            strategy,
            tier,
            route(false),
        )?,
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

/// The deepest stack **any published scope can window**, which is the domain the shared request
/// scope validates a window's start block against.
///
/// This is the max over BOTH armed stacks, and that is not tidiness. Kolors publishes two rung-4
/// scopes ([`TRANSFORMER_WINDOW_COMPONENTS`]): `Dit` windows the U-Net's `Transformer2D` sub-stacks,
/// which are **10** deep at their widest, and `TextEncoder` windows ChatGLM3-6B's **28** `GlmBlock`s.
/// Passing only the U-Net's 10 would make the shared hook reject a legitimate `TextEncoder` window
/// starting at block ≥ 10 — dormant today, because nothing drives a start block, and wrong the
/// moment anything does. Every sibling provider passes its actual depth.
///
/// Both halves are read off the model configs rather than written down here, so a topology change
/// moves this number with it.
fn widest_windowable_stack() -> usize {
    widest_transformer_stack().max(crate::chatglm3::ChatGlmConfig::chatglm3_6b().num_layers)
}

/// The widest `Transformer2D` sub-stack depth — the `Dit` scope's half of
/// [`widest_windowable_stack`].
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
        widest_windowable_stack(),
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
             Missing): it moves the peak 0.00% at both measured scopes (the request and the U-Net \
             seam), and the query-row chunking axis is not bit-exact on Metal, which an fp16 \
             chaos-sensitive schedule amplifies to a different image — see \
             memory_strategy::ATTENTION_SUPPORT"
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

    // ── Asset-fact projections on real safetensors headers (sc-15839) ────────────────────────────
    //
    // Synthesized snapshots, not a cached tier: the arithmetic under test is header-only, so a
    // handful of tensors proves it exactly and a 6 GB tier would only make it slow and gated.

    /// A throwaway snapshot root under the temp dir, removed by [`Snapshot`]'s `Drop`.
    struct Snapshot(std::path::PathBuf);

    impl Snapshot {
        fn new(tmp: &tempfile::TempDir, tag: &str) -> Self {
            let root = tmp.path().join(format!(
                "kolors-assetfacts-{tag}-{}-{:?}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
            ));
            std::fs::create_dir_all(&root).unwrap();
            Self(root)
        }

        /// Write one `dtype` tensor of `shape` at `rel`, creating parents.
        fn write(&self, rel: &str, name: &str, dtype: safetensors::Dtype, shape: &[usize]) {
            let width = match dtype {
                safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
                safetensors::Dtype::F32 => 4,
                other => panic!("unhandled test dtype {other:?}"),
            };
            let bytes = vec![0_u8; shape.iter().product::<usize>() * width];
            let view = safetensors::tensor::TensorView::new(dtype, shape.to_vec(), &bytes).unwrap();
            let path = self.0.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, safetensors::serialize([(name, view)], &None).unwrap()).unwrap();
        }

        /// The three base components, each one small dense tensor. The VAE deliberately ships ONLY
        /// the fp16 variant, which is the layout every cached `SceneWorks/kolors-mlx` tier has.
        fn with_base(self) -> Self {
            self.write(
                "text_encoder/model.safetensors",
                "embedding.weight",
                safetensors::Dtype::F16,
                &[4, 8],
            );
            self.write(
                "unet/diffusion_pytorch_model.safetensors",
                "conv_in.weight",
                safetensors::Dtype::F16,
                &[4, 8],
            );
            self.write(
                "vae/diffusion_pytorch_model.fp16.safetensors",
                "decoder.conv_in.weight",
                safetensors::Dtype::F16,
                &[4, 8],
            );
            self
        }

        fn path(&self, rel: &str) -> std::path::PathBuf {
            self.0.join(rel)
        }
    }

    impl Drop for Snapshot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// 4x8 f16 = 64 stored **payload** bytes, which is what the header-only projections sum.
    ///
    /// Note this is NOT what `PerComponentBytes::from_spec_subdirs` reports for the two components
    /// it still prices from disk: that helper sums whole-FILE bytes, so a base component also
    /// carries its safetensors header. The tests below read the real file lengths for those rather
    /// than assuming, so they assert the projection under test and nothing else.
    const STORED_TENSOR_BYTES: u64 = 4 * 8 * 2;

    fn file_bytes(path: &std::path::Path) -> u64 {
        std::fs::metadata(path).unwrap().len()
    }

    /// **The VAE is priced at its RESIDENT f32 width, not its stored fp16 one** (sc-15839).
    ///
    /// `mlx_gen_sdxl::load_vae` casts unconditionally, so a stored-bytes footprint under-declared the
    /// decoder by exactly 2x at every tier — and the worker sizes budgets off these facts. The
    /// assertion is the factor, not a literal: it is the thing that was wrong.
    #[test]
    fn the_decoder_is_priced_at_its_resident_f32_width_not_its_stored_fp16_one() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = Snapshot::new(&tmp, "vae").with_base();
        let spec = LoadSpec::new(WeightsSource::Dir(snapshot.0.clone()));
        let components = crate::registry::component_footprint(&spec).unwrap();

        // The two components that really do load at their stored precision are untouched: still the
        // on-disk file sum `from_spec_subdirs` has always reported.
        assert_eq!(
            components.text_encoder,
            file_bytes(&snapshot.path("text_encoder/model.safetensors"))
        );
        assert_eq!(
            components.dit,
            file_bytes(&snapshot.path("unet/diffusion_pytorch_model.safetensors"))
        );
        // The decoder is the one that changed, and the factor is the whole point.
        let stored = snapshot.path("vae/diffusion_pytorch_model.fp16.safetensors");
        assert_eq!(
            components.vae,
            STORED_TENSOR_BYTES * 2,
            "the decoder must be projected to f32: `load_vae` does `cast_all(Float32)` on every \
             load and every cached tier ships only the fp16 variant, so stored bytes are half the \
             truth"
        );
        // Payload against payload — the invariant is the FACTOR, and it must not be read off a file
        // length: at this synthetic scale the safetensors header dwarfs the tensor, while on the
        // real 159.56 MiB VAE it is noise.
        let stored_payload =
            projected_safetensors_bytes(&stored, |_| ResidentProjection::Stored).unwrap();
        assert_eq!(stored_payload, STORED_TENSOR_BYTES);
        assert_eq!(
            components.vae,
            stored_payload * 2,
            "exactly 2x the stored fp16 payload, at every tier"
        );

        // And the contract carries the projected value, not just the footprint helper.
        let contract = memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        assert_eq!(contract.asset_facts.decoder_bytes, STORED_TENSOR_BYTES * 2);
        assert!(contract.conformance_errors().is_empty());
    }

    /// An f32 master alongside the fp16 variant must not double-count: the footprint sizes the ONE
    /// file `load_vae` opens, and it prefers the f32 master.
    #[test]
    fn the_decoder_footprint_sizes_the_file_the_loader_opens_not_the_whole_vae_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = Snapshot::new(&tmp, "vae-both").with_base();
        snapshot.write(
            "vae/diffusion_pytorch_model.safetensors",
            "decoder.conv_in.weight",
            safetensors::Dtype::F32,
            &[4, 8],
        );
        let spec = LoadSpec::new(WeightsSource::Dir(snapshot.0.clone()));
        let components = crate::registry::component_footprint(&spec).unwrap();
        assert_eq!(
            components.vae,
            STORED_TENSOR_BYTES * 2,
            "the f32 master is already 2x the fp16 variant's stored size and is loaded verbatim; \
             summing the dir would report 3x"
        );
    }

    /// **The three resident overlays are declared and priced** (sc-15839).
    ///
    /// Kolors loads a ControlNet, an IP-Adapter image encoder + K/V pairs, and a PiD student — and
    /// `overlay_bytes` was zero for all of them, which the shared validator waves through because it
    /// only cross-checks the overlay legs `if overlay_bytes > 0`. A rung-1 or rung-4 selection
    /// admitted for such a load under-predicted the peak by the whole overlay set.
    #[test]
    fn the_resident_overlays_are_declared_priced_and_attributable() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = Snapshot::new(&tmp, "overlays").with_base();
        snapshot.write(
            "control/diffusion_pytorch_model.safetensors",
            "controlnet_cond_embedding.conv_in.weight",
            safetensors::Dtype::F16,
            &[4, 8],
        );
        snapshot.write(
            "ip/image_encoder/model.safetensors",
            "vision_model.embeddings.patch_embedding.weight",
            safetensors::Dtype::F16,
            &[4, 8],
        );
        snapshot.write(
            "ip/ip_adapter_plus_general.safetensors",
            "ip_adapter.0.to_k_ip.weight",
            safetensors::Dtype::F16,
            &[4, 8],
        );

        let spec = LoadSpec::new(WeightsSource::Dir(snapshot.0.clone()))
            .with_load_shape(LoadShape::DeferredMaterialization)
            .with_control(WeightsSource::Dir(snapshot.path("control")))
            .with_ip_adapter(WeightsSource::Dir(snapshot.path("ip")));
        let contract = memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        assert!(
            contract.conformance_errors().is_empty(),
            "{:?}",
            contract.conformance_errors()
        );

        // Every tensor here is dense f16, which the fp16 compute cast leaves at 2 bytes/element.
        const CONTROL_BYTES: u64 = STORED_TENSOR_BYTES;
        const IP_BYTES: u64 = STORED_TENSOR_BYTES * 2;
        assert_eq!(
            contract.asset_facts.overlay_bytes,
            CONTROL_BYTES + IP_BYTES,
            "both overlays this load holds resident must be priced"
        );
        // `base_bytes` stays the three BASE components and must not absorb the overlay.
        assert_eq!(
            contract.asset_facts.base_bytes,
            contract.asset_facts.conditioning_bytes
                + contract.asset_facts.transformer_bytes
                + contract.asset_facts.decoder_bytes,
            "base_bytes is the three base components and excludes the overlay"
        );
        assert_eq!(
            contract.total_resident_bytes(),
            contract.asset_facts.base_bytes + CONTROL_BYTES + IP_BYTES
        );

        let by_id = |id: &str| {
            contract
                .resident_components()
                .iter()
                .find(|component| component.id == id)
                .unwrap_or_else(|| panic!("{id} must be declared"))
                .clone()
        };
        let control = by_id(CONTROL_COMPONENT_ID);
        assert_eq!(control.kind, MemoryComponentKind::ControlBranch);
        assert_eq!(control.resident_bytes, CONTROL_BYTES);
        let ip = by_id(IP_ADAPTER_COMPONENT_ID);
        assert_eq!(ip.kind, MemoryComponentKind::IpAdapter);
        assert_eq!(
            ip.resident_bytes, IP_BYTES,
            "BOTH IP files are resident: the encoder/resampler on the generator, the K/V pairs \
             installed into the U-Net"
        );
        assert!(
            contract.resident_components().iter().all(|component| {
                component.kind != MemoryComponentKind::Transformer(TransformerComponent::Dit)
            }),
            "an overlay must never be reported as the base denoising transformer"
        );
        // No published rung bounds an overlay here — rung 4 windows the U-Net and ChatGLM3 only.
        assert!(contract
            .resident_components()
            .iter()
            .all(|component| component.bounded_by.is_none()));
        assert!(
            contract.formula.uses(MemoryFormulaVariable::OverlayBytes),
            "declared auxiliary bytes are inert unless the formula consumes them"
        );

        const BASE_PEAK: u64 = 1_000;
        let prediction = contract.predicted_peak_from_base(BASE_PEAK);
        assert_eq!(
            prediction.predicted_peak_bytes(),
            BASE_PEAK + CONTROL_BYTES + IP_BYTES,
            "zeroing the overlay, dropping a component, or dropping OverlayBytes from the formula \
             must change this"
        );
        assert_eq!(prediction.unattributed_bytes, BASE_PEAK);
    }

    /// A plain txt2img load declares **no** overlay and keeps its pre-sc-15839 envelope exactly.
    ///
    /// The point is that the component axis is adopted only where there is something to attribute: a
    /// provider that declared an always-on `OverlayBytes` variable would invite calibration keyed on
    /// a value that never varies.
    #[test]
    fn a_plain_load_declares_no_overlay_and_keeps_the_phase_envelope() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = Snapshot::new(&tmp, "plain").with_base();
        let spec = LoadSpec::new(WeightsSource::Dir(snapshot.0.clone()))
            .with_load_shape(LoadShape::DeferredMaterialization);
        let contract = memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        assert!(contract.conformance_errors().is_empty());
        assert_eq!(contract.asset_facts.overlay_bytes, 0);
        assert!(contract.resident_components().is_empty());
        assert!(!contract.formula.uses(MemoryFormulaVariable::OverlayBytes));
        assert!(matches!(
            contract.formula,
            MemoryFormulaKind::PhaseEnvelope { .. }
        ));
    }

    /// **The PiD carve-out is policy-scoped, not blanket.**
    ///
    /// `registry::build_residency` passes `use_pid = true` unconditionally on the `Resident` arm, so
    /// the student is loaded once and every later request runs with it resident whether or not it
    /// asked. On `Sequential` it passes `req.use_pid`, so the overlay really is request-selected and
    /// pricing it at load time would overstate the ordinary path. Both directions are asserted,
    /// because a blanket carve-out (what the sibling providers do) is wrong in one of them.
    #[test]
    fn the_pid_overlay_is_priced_where_the_policy_makes_it_unconditional() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = Snapshot::new(&tmp, "pid").with_base();
        snapshot.write(
            "pid/student.safetensors",
            "net.blocks.0.weight",
            safetensors::Dtype::F16,
            &[4, 8],
        );
        snapshot.write(
            "pid/gemma/gemma-2-2b-it.safetensors",
            "model.embed_tokens.weight",
            safetensors::Dtype::F16,
            &[4, 8],
        );
        let with_pid = |policy: OffloadPolicy| {
            let spec = LoadSpec::new(WeightsSource::Dir(snapshot.0.clone()))
                .with_offload_policy(policy)
                .with_pid(
                    WeightsSource::File(snapshot.path("pid/student.safetensors")),
                    WeightsSource::Dir(snapshot.path("pid/gemma")),
                );
            memory_strategy_contract(crate::MODEL_ID, &spec).unwrap()
        };

        let resident = with_pid(OffloadPolicy::Resident);
        assert!(resident.conformance_errors().is_empty());
        assert_eq!(
            resident.asset_facts.overlay_bytes,
            STORED_TENSOR_BYTES * 2,
            "under Resident the student + caption encoder are loaded unconditionally and are \
             resident for every request, so declaring them at zero under-prices the ordinary path"
        );
        let pid = resident
            .resident_components()
            .iter()
            .find(|component| component.id == PID_COMPONENT_ID)
            .expect("the PiD overlay must be attributable under Resident");
        assert!(
            pid.kind.is_auxiliary(),
            "the contract arithmetic keys on is_auxiliary(), which is the property that matters"
        );

        let sequential = with_pid(OffloadPolicy::Sequential);
        assert!(sequential.conformance_errors().is_empty());
        assert_eq!(
            sequential.asset_facts.overlay_bytes, 0,
            "under Sequential the student is loaded per request from `req.use_pid`, so its presence \
             on the LoadSpec is availability rather than residency"
        );
        assert!(sequential.resident_components().is_empty());
    }

    /// **A mixed snapshot re-quantizes the 6B encoder every generate, and the advisory must say so.**
    ///
    /// `registry::load`'s `warn_sequential_requantize` gate read `load_leaves_blocks_lazy` alone,
    /// which inspects `unet/`. A Sequential load re-runs the whole staged schedule per request —
    /// encoder included — so on a snapshot whose `unet/` is packed and whose `text_encoder/` is
    /// dense the advisory was suppressed while the ChatGLM3-6B tower genuinely re-packed on every
    /// request. That is the larger of the two costs on this family, not the smaller.
    ///
    /// Asserted on the predicates rather than by capturing a log line, so it pins the condition the
    /// gate is built from.
    #[test]
    fn a_mixed_snapshot_still_repacks_the_encoder_each_generate() {
        let tmp = tempfile::tempdir().unwrap();
        let snapshot = Snapshot::new(&tmp, "mixed").with_base();
        // Packed `unet/`, dense `text_encoder/` — the shape that slipped through.
        std::fs::write(
            snapshot.path("unet/config.json"),
            br#"{"quantization": {"bits": 8, "group_size": 64}}"#,
        )
        .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(snapshot.0.clone()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization)
            .with_quant(mlx_gen::Quant::Q8);

        assert!(
            load_leaves_blocks_lazy(&spec),
            "the U-Net is already packed, so the U-Net-only predicate reports nothing to do — this \
             is exactly what suppressed the advisory"
        );
        assert!(
            !text_encoder_leaves_blocks_lazy(&spec),
            "the dense text_encoder/ DOES re-pack on every staged generate"
        );
        assert!(
            !load_leaves_blocks_lazy(&spec) || !text_encoder_leaves_blocks_lazy(&spec),
            "the gate registry::load uses must fire on this snapshot"
        );

        // The control: a fully packed snapshot re-packs nothing and must stay quiet, which is the
        // narrowing sc-15521 made and which this must not undo.
        std::fs::write(
            snapshot.path("text_encoder/config.json"),
            br#"{"quantization": {"bits": 8, "group_size": 64}}"#,
        )
        .unwrap();
        assert!(load_leaves_blocks_lazy(&spec));
        assert!(text_encoder_leaves_blocks_lazy(&spec));
    }

    /// The `TextEncoder` scope's stack is 28 blocks deep, so the window domain the shared request
    /// scope validates against must be 28 — not the U-Net's 10.
    #[test]
    fn the_windowable_depth_covers_the_deeper_of_the_two_armed_stacks() {
        assert_eq!(widest_transformer_stack(), 10, "the U-Net's deepest stack");
        assert_eq!(
            widest_windowable_stack(),
            crate::chatglm3::ChatGlmConfig::chatglm3_6b().num_layers,
            "ChatGLM3-6B's 28 GlmBlocks are windowable by the published TextEncoder scope, and a \
             domain of 10 would reject a legitimate window starting at block >= 10"
        );
        assert!(
            widest_windowable_stack() > widest_transformer_stack(),
            "a regression to the U-Net-only depth must redden here"
        );
    }

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
        let fused = LoadSpec::new(WeightsSource::File(
            "/nonexistent/kolors.safetensors".into(),
        ))
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
        assert!(load_leaves_blocks_lazy(&spec(
            LoadShape::DeferredMaterialization
        )));
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
        assert!(err.to_string().contains("38/255"), "got: {err}");

        let chunked = GenerationRequest {
            memory: Some(GenerationMemory {
                chunk_attention: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        let err =
            attention_plan(&chunked).expect_err("a bounded-attention request must be refused");
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
    /// composition that would bound 1.147 GiB of U-Net blocks (q4) while 3.984 GiB of ChatGLM3-6B
    /// stayed resident next to them.
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
