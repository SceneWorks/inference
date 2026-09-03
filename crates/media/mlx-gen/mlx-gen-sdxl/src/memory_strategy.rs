//! SDXL MLX adoption of the shared memory-strategy contract (SC-15449) — the ladder for the
//! **SDXL and derivatives** family (SC-15525), including its rung-4 half (SC-16355).
//!
//! ## One provider, five catalog entries
//!
//! `sdxl`, `realvisxl`, `realvisxl_lightning`, `illustrious_xl_v1` and `illustrious_xl_v2` are five
//! SceneWorks catalog entries with five separate `downloads` blocks and five weight repositories.
//! They are **not** five providers: `sceneworks-worker`'s `MODEL_TABLE` maps all five to
//! `engine_id: "sdxl"`, and this crate registers exactly one descriptor. So they share one contract
//! builder, one architecture, one `Residency` seam, one tiled decode, one bounded-attention
//! primitive and one block-residency stream — and each entry still owes its **own** per-tier
//! evidence. Sharing code is explicitly not what makes a catalog entry Verified.
//!
//! ## Declared rungs
//!
//! | Rung | Support | Executable seam |
//! |---|---|---|
//! | 0 Resident | Implemented | Warm [`Residency`](mlx_gen::Residency) pair — dual CLIP + (U-Net + control/IP/VAE/PiD) held across requests |
//! | 1 Staged residency | Implemented (request-scoped) | `GenerationMemory::stage_residency` drives encode → **drop both CLIP encoders** → load heavy → denoise + decode |
//! | 2 Bounded decode | route-blind **Missing**; exact measured rows may be Implemented ([`DECODE_SUPPORT`]) | [`Autoencoder::decode_tiled`](crate::vae::Autoencoder) uses SC-19753 full-image GroupNorm statistics and sealed route/tier/mode/overlay/geometry admission; historical whole-tail memory was 14.360 → 11.237 GiB and must be recaptured for the new arithmetic |
//! | 3 Bounded attention | **Missing** (measured — [`ATTENTION_SUPPORT`]) | [`mlx_gen::attention::sdpa_budgeted_bhsd`] reaches every site, moves the peak **0.00% at BOTH scopes** (the request and the U-Net seam), and its query-row axis is not bit-exact on Metal |
//! | 4 Bounded transformer residency | Implemented (streamable loads) | [`mlx_gen::block_residency::run_windowed`] per `Transformer2D` — eleven sub-stacks, 70 blocks |
//!
//! The rung-2 row is the route-blind legacy default. A caller may now replace it for one exact
//! catalog route, tier, mode, overlay and output geometry by packaging a sealed production-latent
//! quality policy. Coordinates absent from that table remain refused; an empty table preserves the
//! default and the existing estimated-fit fallback.
//! The older range failures documented below predate SC-19753's layer-wise normalization and are
//! retained as history, not as the current decoder's expected quality result.
//!
//! ## What this family actually buys, measured (`realvisxl`, Apple/Metal, 1024², 6 steps)
//!
//! | composition | tier | request peak | vs baseline | output |
//! |---|---|---:|---:|---|
//! | resident | bf16 | 20.513 GiB | — | — |
//! | + rung 1 | bf16 | **19.003** | **−7.4%** | byte-identical |
//! | staged (control) | q8 | 17.693 | — | — |
//! | + rung 4, any published cadence | q8 | **15.516** | **−12.31%** | byte-identical |
//!
//! ## Rung 4's product evidence is allocation and correctness
//!
//! The full sweep — peak and byte-identity per cadence, over three tiers and three output sizes —
//! is on [`TRANSFORMER_WINDOW_SIZES`]. The resolved peak column is **flat** because phase separation
//! puts the request high-water in decode, not in the windowed forward. The two cells that once looked
//! non-flat (512² q8 and q4) were allocator noise indexed by execution position (SC-17679).
//!
//! Wall-clock timing is retained as diagnostic output only: it changes with host state even when
//! peaks and pixels are identical. Rung 4 therefore publishes no time/memory frontier between
//! cadences; the default remains the widest complete-sub-stack cadence for its deterministic block
//! plan, while callers may select any admitted cadence.
//!
//! ## Per-entry coverage is NOT uniform — measured, per entry, per tier
//!
//! Both columns are per entry per tier, because SC-16355's AC is a **request peak measured per
//! tier** and a capability published per tier that is measured on one entry is the same substitution
//! this table exists to refuse. Every `Implemented` row below carries its own rung-4 row, taken at
//! the shipped default cadence against that same entry's staged control
//! (`every_catalog_entry_loads_and_publishes_the_ladder`, 1024², 4 steps):
//!
//! | entry | tier | staged peak | rung-4 peak (cadence 1) | rung 4 buys | rung 4 |
//! |---|---|---:|---:|---:|---|
//! | `sdxl` | bf16 | 19.005 GiB | 14.938 | **−21.40%** | Implemented |
//! | `realvisxl` | bf16 | 19.003 | 14.936 | **−21.40%** | Implemented |
//! | `realvisxl` | q4 | 16.654 | 15.493 | **−6.97%** | Implemented |
//! | `realvisxl` | q8 | 17.693 | 15.516 | **−12.31%** | Implemented |
//! | `realvisxl_lightning` | q4 | 16.654 | 15.493 | **−6.97%** | Implemented |
//! | `illustrious_xl_v1` | q8 | 17.693 | — | — | **Missing** |
//! | `illustrious_xl_v2` | q8 | 17.693 | — | — | **Missing** |
//!
//! **What rung 4 is worth is a function of the tier, and it is not small variation: −21.4% at bf16
//! against −7.0% at q4.** That is the phase-separation mechanism read off a second axis — the saving
//! is the whole `transformer_blocks` weight set (4.071 / 2.166 / 1.149 GiB by tier), so what changes
//! across tiers is the numerator while the resident remainder and the decode transient barely move.
//! A selector that sized rung 4's value off the q8 row would under-buy at bf16 by a factor of three.
//! See [`TRANSFORMER_WINDOW_SIZES`] for the arithmetic.
//!
//! `realvisxl`/bf16 reads 19.003 GiB at **four** steps and also at six, because under rung 1 the
//! request peak is the phase envelope `max(CLIP-L + bigG, U-Net + VAE)` — a residency bound, not an
//! accumulation — so step count does not move it. Quoting the same figure for both is correct; what
//! would be wrong is quoting it without saying which row it came from.
//!
//! **Rung 4 is `Missing` on `illustrious_xl_v1`/q8 and `illustrious_xl_v2`/q8** — which is, today,
//! those two entries' only advertised tier, so it is `Missing` for those entries outright. The first
//! revision of this ladder's evidence did not say so, because its per-entry test recorded a peak and
//! never checked which ladder each entry published. It does now
//! (`every_catalog_entry_loads_and_publishes_the_ladder`, and the `EXPECTED_RUNG_FOUR` table beside
//! it).
//!
//! The cause is the **snapshot**, not the architecture. Both entries' `unet/` weights are genuinely
//! packed — u32 codes with `.scales`/`.biases`, 3.46 GiB, byte-for-byte the same shape as
//! `realvisxl`'s q8, and their measured 17.693 GiB peak is `realvisxl` q8's to the millibyte — but
//! their `unet/config.json` ships **without the `quantization` marker**. `mlx_gen::quant::packed_quant_bits`
//! reads that marker and only that marker, so [`load_leaves_blocks_lazy`] cannot distinguish this
//! load from one that would quantize at load time and materialize the trunk, and [`streamable`]
//! refuses. (The same missing marker also makes `warn_sequential_requantize` fire on every load of
//! these two — a visible symptom of the same root cause.)
//!
//! **The rung fails closed, which is correct, and the repair is a republished snapshot rather than a
//! looser predicate here** (tracked as sc-17522). Sniffing the tensors instead would put this
//! predicate and the F-144 requested-vs-packed tier guard on two different definitions of "packed",
//! and rung 4 replays the *recorded tier* into every re-materialized block — so the two must not be
//! allowed to disagree about what tier that is.
//!
//! **Two of the five rungs are `Missing`, and both verdicts are measurements rather than gaps.**
//! Their mechanisms are implemented, reach every advertised route, and are exercised by
//! `tests/memory_ladder_real_weights.rs`; what they failed is the bar, not the wiring. Recording
//! that with numbers is the point — a rung quietly omitted and a rung measured-and-rejected are very
//! different facts for a selector, and only one of them tells the next story where to look.
//!
//! ## Rung 1 is request-scoped, and what it does NOT release
//!
//! Before this story SDXL's residency was a load-time `OffloadPolicy` only. It is now request-scoped
//! through `stage_residency`: the same cached generator serves warm → staged → warm without
//! reconstruction, and the load-time policy survives only as the **default** for a request that
//! names nothing.
//!
//! Be precise about what the staged schedule buys, because it is two phases and not three. The seam
//! releases the **text encoders** before the heavy bundle loads, so the peak is bounded by
//! `max(CLIP-L + bigG, U-Net + VAE + overlays)` rather than the sum. It does **not** release the
//! U-Net before the decode: `SdxlHeavyOwned` is one bundle and the decode runs inside the same
//! render closure the denoise did. That is why rung 2 matters here more than on a family that can
//! shed its transformer first — SDXL's decode peak stacks on top of a resident U-Net.
//!
//! ## Rung 4: a plan per `Transformer2D`, not one per model
//!
//! SDXL is the first **U-Net** on this ladder; every prior MLX adopter is a DiT with one flat block
//! list. The windowable blocks live in eleven independent `Transformer2D` sub-stacks — six of depth
//! 10 (`down_blocks.2` ×2, `mid_block` ×1, `up_blocks.0` ×3) and five of depth 2 (`down_blocks.1`
//! ×2, `up_blocks.1` ×3), 70 blocks in total — so one U-Net forward drives `run_windowed` **eleven**
//! times, each against its own depth. See `crate::block_stream`.
//!
//! The conv/resnet trunk, the up/down samplers, each sub-stack's `norm`/`proj_in`/`proj_out`, and
//! the down→up skip stack are the **resident remainder**. The skip stack is the reason this rung's
//! *value* needed measuring rather than assuming: it holds down-path **activations** alive across
//! the whole forward, and rung 4 bounds **weights**, so the term it competes against does not
//! shrink. `docs/sc-16195/resolution-sweep.json` puts illustrious q8 at 4.74 GiB resident against a
//! 14.04 GiB transient — an activation-dominated profile, which is the opposite of the
//! weight-dominated DiTs where this rung has been worth 60-76%.
//!
//! ## Fail closed — the complete list
//!
//! Each of these is a typed rejection rather than a silently narrowed execution:
//!
//! * a bounded-decode selection without an exact admitted route/geometry policy (`refuse_decode`,
//!   and again in `decode_tiling`) and **any** bounded-attention selection (`attention_plan`) — the
//!   route-blind rung 2 fallback and rung 3 are `Missing`, so the path from "the code exists" to "a
//!   render silently used it" remains closed while sealed SC-19753 rows can engage rung 2;
//! * a transformer window **component** this family does not implement (`TextEncoder` / `Both`) —
//!   never narrowed to `Dit`;
//! * a transformer window **size** outside [`TRANSFORMER_WINDOW_SIZES`] (`validate_window`);
//! * rung 4 on a load that cannot stream — a single-file/LDM source, an eager load shape, a
//!   load-time quantization that materializes the trunk, or an **adapter load that merged**
//!   (see [`streamable`]);
//! * rung 4 without rung 1 engaged in the same request (declared as a prerequisite edge, so the
//!   shared selector enforces it).
//!
//! The decode and window checks are enforced on **two** layers on purpose: the admission gate
//! (`safety_check`) *and* the request-side resolvers. Every calibration and real-weight harness
//! reaches `generate` with a hand-built
//! [`GenerationMemory`](mlx_gen::gen_core::GenerationMemory) rather than through a validated
//! `MemorySelection`, which is precisely the path admission does not see.
//!
//! ## Ownership
//!
//! This file declares *structure and parameter domains only*. Measured coefficients, envelopes and
//! per-tier peaks live in SceneWorks generated evidence keyed by
//! [`MEMORY_CALIBRATION_FINGERPRINT`]; the worker owns live-budget accounting and least-cost
//! selection. The scope below is defense in depth: it can reject a selection, never substitute one.

use mlx_gen::gen_core::{
    adapter_stack_resident_bytes, standard_memory_strategy_safety_check, AdapterResidencyMode,
    Error as CoreError, MemoryBackendRealization, MemoryBehaviorFixture, MemoryBehaviorRoute,
    MemoryCalibrationIdentity, MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier,
    MemoryParameterRanges, MemoryPhase, MemoryPrerequisiteScope, MemoryProviderContract,
    MemoryRequestScope, MemoryResidentComponent, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategyPrerequisite, MemoryStrategySupport, ResidentRequestMemory,
    Result as CoreResult, TransformerComponent,
};
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{GenerationRequest, LoadShape, LoadSpec, OffloadPolicy, WeightsSource};

/// The decode tile edge a rung-2 request would have defaulted to, kept as the sweep's anchor.
pub const DECODE_TILE_EDGE: u32 = 896;

/// The decode overlap paired with every swept edge, in output pixels.
pub const DECODE_OVERLAP: u32 = 64;

/// The decode tile edges the rung-2 sweep **measured**, largest first.
///
/// Retained as evidence, not as a published domain — see [`DECODE_SUPPORT`], which declares rung 2
/// `Missing` on this family. `tests/memory_ladder_real_weights.rs::decode_tile_mechanism_sweep`
/// drives every one of them through the real [`Autoencoder::decode_tiled`](crate::vae::Autoencoder)
/// so a future story revisiting the head/tail split has numbers to beat rather than a blank.
pub const DECODE_TILE_EDGES_SWEPT: &[u32] = &[896, 768, 640, 512, 448, 384, 320, 256];

/// The rung-2 overlaps the sweep measured against every edge in [`DECODE_TILE_EDGES_SWEPT`].
pub const DECODE_OVERLAPS_SWEPT: &[u32] = &[64, 128, 192, 256];

/// **The route-blind rung-2 fallback is `Missing` on SDXL/MLX.**
///
/// The measurements below are the historical pre-SC-19753 evidence for refusing one unconditional
/// tile domain across every route and output size. SC-19753 changed the decoder to preserve global
/// GroupNorm statistics and packages new production-latent results as exact sealed route/geometry
/// policies. Those rows can adopt rung 2 without weakening this fail-closed fallback; coordinates
/// absent from the table still see [`DECODE_SUPPORT`] `Missing`.
///
/// The original mechanism was implemented and bounded memory: [`Autoencoder::decode_tiled`](crate::vae::Autoencoder)
/// runs the globally-scoped decoder head (denormalize → `post_quant_conv` → `conv_in` → mid resnets →
/// mid **self-attention**) once on the full latent and tiles only the full-resolution upsample tail,
/// which is the same head/tail split `mlx_gen_qwen_image` uses. It bounds real memory: at 1024² bf16
/// the isolated decode peak falls **14.360 → 11.237 GiB (−21.7%)** at edge 896, and the whole request
/// falls by the margin `layerwise_decode_is_priced_at_the_request_level` measures.
///
/// Both scopes are quoted because they answer different questions and an earlier revision of this
/// file quoted a request-level figure no test in this crate produced. The mechanism sweep measures
/// the decode in isolation — the right scope for the drift; the request row measures what a caller
/// would actually have paid — the right scope for the saving.
///
/// The rest of this section is historical pre-SC-19753 evidence for the retired whole-tail decoder;
/// it is not a current quality claim. That decoder was not published because it did not preserve the output. Swept against the **exact untiled
/// decode of the same latent** — a real 1024² render re-encoded through the same VAE — on
/// `realvisxl` bf16 (`decode_tile_mechanism_sweep`):
///
/// | edge | overlap 64 | 128 | 192 | 256 |
/// |---:|---:|---:|---:|---:|
/// | 896 | 65 | 64 | **38** | 40 |
/// | 768 | 82 | 89 | 87 | 89 |
/// | 640 | 92 | 93 | 81 | 77 |
/// | 512 | 109 | 109 | 109 | 109 |
/// | 448 | 114 | 114 | 114 | 114 |
/// | 384 | 121 | 121 | 121 | 121 |
/// | 320 | 126 | 126 | 126 | 124 |
/// | 256 | 132 | 132 | 129 | 103 |
///
/// (max Δ per channel, out of 255. Mean Δ ranges 0.89 → 4.12.) The **best** geometry in the whole
/// grid moves a channel by 38/255, and every other one at this output size is worse.
///
/// **Two controls establish that the drift is the architecture and not the implementation.**
///
/// 1. *It is not a ragged-split artifact.* Re-run at 1536², where 768 / 512 / 384 all divide the
///    output evenly, the drift gets **worse**, not better: 160, 170, 183 at overlap 64.
/// 2. *It is not the blend loop.* `mlx_gen::vae_tiling` reconstructs a tile-consistent decode
///    exactly (`image_spatial_tiles_reconstruct`), and raising the overlap — which widens each
///    tile's context — moves 896 from 65 to 38, the direction a normalization-extent effect predicts
///    and a blending bug does not.
///
/// The mechanism is the **GroupNorms in the tail**: every `ResnetBlock2D` in `up_blocks` and the
/// final `conv_norm_out` computes statistics over the spatial extent it is handed, so a tile
/// normalizes against its own crop rather than the image.
///
/// ## Why 38/255 is nevertheless not publishable — the bar, and the range
///
/// **The product bar remains 48/255.** Historically it came from Z's old
/// 48-admitted/64-rejected split; Z now admits all measured edges with a much lower worst result, so
/// that split is provenance rather than current domain evidence. By that bar the old 1024² sweep
/// datum **passed**: 38 < 48, buying a measured
/// **19.0032 → 15.8660 GiB (−16.51%)** request peak at only ~7% wall clock (876 → 936 ms/step). On
/// the 1024² sweep alone this candidate should ship, and an earlier revision of this file was wrong
/// to claim otherwise — it asserted SDXL was "past that wall at every tile size", which its own
/// table above contradicts.
///
/// **Two independent measurements withhold it anyway**, and both were missing from that revision.
///
/// ### 1. On the latent a real render produces, 1024² does not clear the bar either
///
/// The 38/255 above is measured against a latent obtained by **re-encoding a finished image** — its
/// statistics have already been through the VAE round trip. Decode what the denoiser actually hands
/// the decode phase and the same geometry drifts **84/255**
/// (the pre-SC-19753 form of `layerwise_decode_is_priced_at_the_request_level`). The production latent is the one
/// a user would get, so the one output size where the candidate looked admissible does not survive
/// contact with a production request. Nothing about the sweep was wrong — it is the right instrument
/// for comparing geometries against each other, and the wrong one for deciding an absolute bar.
///
/// ### 2. And the datum is a property of the geometry, not of the tile edge
///
/// `MemoryParameterRanges::decode_tile_edges` is an absolute pixel domain with no geometry axis, so
/// publishing 896 publishes it at everything `crate::model::descriptor` advertises — and that is
/// `max_size: 2048`. Re-swept across that range at the best overlap
/// (the pre-SC-19753 form of `layerwise_decode_clears_the_bar_across_the_advertised_output_range`):
///
/// | output | tiles | tile covers | max Δ | vs the 48/255 bar |
/// |---:|---:|---:|---:|---|
/// | 1024² | 4 | 87.5% | **38** | clears (but see 1 — 84 on the production latent) |
/// | 1280² | 4 | 70.0% | 64 | fails |
/// | 1536² | 4 | 58.3% | 120 | fails |
/// | 2048² | 9 | 43.75% | 77 | fails |
///
/// The GroupNorm mechanism predicts exactly this shape: what governs the drift is the tile's
/// **fraction of the image**, because that is what sets how far a tile's statistics can sit from the
/// global ones. Edge 896 covers 87.5% of a 1024² output and 43.75% of a 2048² one. No fixed pixel
/// edge holds a constant fraction across a 4× range, so no fixed pixel edge is admissible across
/// this provider's advertised range — and a domain admissible at exactly one output size is not
/// something `decode_tile_edges` can express. A selector would size a fit off `[896]`, choose it at
/// 1536², and get either a hard admission rejection or a visibly seamed image.
///
/// So the rung is withheld — but the reason is now stated correctly and each half is asserted by a
/// test, and the prize is recorded with a number rather than dismissed: **−16.51% of the request
/// peak for ~7% wall clock**, which is a better memory/latency trade than rung 4's. SC-19753 supplies
/// the previously missing full-extent normalization and exact geometry policy; the constant remains
/// `Missing` only because a route-blind value cannot express that sealed identity.
///
/// Publishing it as-is would be substituting quality for memory without saying so — the same refusal
/// the catalog already makes for precision (tier integrity) and geometry (SC-15807).
pub const DECODE_SUPPORT: MemoryStrategySupport = MemoryStrategySupport::Missing;

/// The attention chunk size the rung-3 sweep exercised.
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;

/// **Rung 3 is `Missing` on SDXL/MLX, and that is two independent measurements.**
///
/// The mechanism is implemented and reaches every attention site: both the self-attention and the
/// cross-attention of all 70 `TransformerBlock`s, plus the IP-Adapter decoupled branch, route
/// through [`mlx_gen::attention::sdpa_budgeted_bhsd`] on every advertised denoise route.
///
/// It is not published for two reasons, either of which is sufficient. Both are measured on
/// `realvisxl` bf16 at 1024², and each has its own test because they are measured at different
/// scopes and the scopes are not interchangeable:
///
/// 1. **It does not move the REQUEST peak at all**, and at the seam it moves it the wrong way.
///    End to end (`attention_chunking_is_measured_against_the_rung_two_top`, which re-assembles the
///    staged request from the public pipeline entry points because the production resolver refuses a
///    bounded-attention selection):
///
///    | steps | unchunked | chunked | Δ peak | max Δ px | mean Δ px |
///    |---:|---:|---:|---:|---:|---:|
///    | 1 | 19.0032 GiB | 19.0032 GiB | **−0.00%** | 57/255 | 1.53 |
///    | 6 | 19.0032 GiB | 19.0032 GiB | **−0.00%** | 255/255 | 4.55 |
///
///    Measured instead at the U-Net seam, where the effect is not diluted by the rest of the request
///    (`attention_chunking_is_measured_at_the_unet_seam`), one CFG-batched 1024² forward goes
///    **6.5868 → 6.5868 GiB, +0.00%** — the two plans peak identically to the millibyte, reproduced
///    across runs. So there is **no scope on this family at which the query-row budget changes the
///    high-water mark**, which is a cleaner statement of the withholding reason than a diluted one.
///
///    That is consistent with the hazard `mlx_gen::attention`'s own docs name — when q/k/v are
///    already materialized and pinned, a query-row budget has nothing left to bound — and on a U-Net
///    that is the normal case, because the conv/resnet trunk between every attention has already
///    broken the lazy graph `eval_per_chunk` exists to cut. The epic is explicit that a rung which
///    does not move the peak is not a saving.
///
///    **Correction (SC-17679).** An earlier revision published this seam as
///    **6.2587-6.2685 → 6.5868 GiB, +5.1%** and argued from it that chunking *adds* transients here.
///    Both were wrong: `attention_chunking_is_measured_at_the_unet_seam` had no discarded warm-up
///    row, so its first (unbounded) `run` was measured against a cold allocator — that is exactly
///    why the published figure had a *range* on one side and a single value on the other. With a
///    warm-up the unbounded row reads the same 6.5868 the bounded one always did. The sibling Kolors
///    harness had the identical defect and published +12.54% for the same reason; corrected, both
///    U-Nets read +0.00%.
/// 2. **The output moves — and that is a property of the CHOSEN AXIS, not of SDXL.** Be precise
///    here, because an earlier revision of this file was not: it said chunking "is not
///    output-preserving on this family", which overstates what was measured and points the next
///    reader at the wrong suspect.
///
///    The arithmetic is exactly preserved. [`sdpa_budgeted_bhsd`](mlx_gen::attention::sdpa_budgeted_bhsd)
///    chunks the **query axis only**: each chunk is a complete fused SDPA over the full k/v, so there
///    is no accumulator and no running max, and the class of bug that would make chunked softmax
///    genuinely wrong is structurally unwritable. What changes is *which reduced-precision Metal
///    matmul specialization MLX dispatches* for the `[.., block, Sk]` product — the sc-2338 parity
///    class — which `mlx_gen::attention` documents as exactly `0` at some block sizes and ~1e-3
///    peak-relative at others, with no monotonic relationship to block size.
///
///    `gen_core::attention_budget` carries a **second** axis for exactly this reason:
///    `AttentionBudget::head_chunks` narrows complete heads and *preserves bit identity*, and its
///    docs say outright that the two axes "share a score budget, but not a numerical contract" and
///    must not be treated as interchangeable. `sdpa_budgeted_bhsd` never uses it — but since
///    sc-18661 `mlx_gen::attention::sdpa_head_budgeted_bhsd` does, and the prediction below is now
///    measured rather than reasoned: at a budget admitting one whole head the head axis reconstructs
///    the unbounded forward at **exactly** `0` relative max-abs, on four shapes including a
///    production 56-head bf16 DiT. Adopting it here is still gated on reason 1. So the honest
///    statement is: **the query-row axis is not bit-exact on Metal, and SDXL is unusually exposed to
///    that** — it runs fp16 through a chaos-sensitive Euler-Ancestral schedule across 140 attention
///    sites. Measured at **one** step, where the sampler cannot amplify anything, the raw per-forward
///    divergence is `max|Δeps| = 5.25e-3` and the image moves by 57/255; by six steps it is a
///    different image.
///
///    This is a real reason to withhold *this* rung as implemented, and it is **not** a reason to
///    conclude the family cannot support bounded attention. A head-axis implementation would be
///    bit-exact by construction; what it would still have to beat is reason 1, which is the binding
///    one.
///
/// The one-step control is what separates both claims from a wiring bug. It is also why the verdict
/// is stated per family per backend: Z-Image measures −1.7% on its denoise phase with a bit-identical
/// image on the same primitive. A rung's magnitude and its output-preservation are both per family
/// per backend, and neither transferred here.
pub const ATTENTION_SUPPORT: MemoryStrategySupport = MemoryStrategySupport::Missing;

/// The production transformer-window domain for rung 4: how many consecutive `TransformerBlock`s of
/// **each** `Transformer2D` are held materialized at once.
///
/// The domain is a **cadence**, applied independently to eleven sub-stacks of two depths
/// (`[2,2,10,10,10,10,10,10,2,2,2]`). [`BlockPlan::new`](mlx_gen::block_residency::BlockPlan::new)
/// clamps a window to its stack's depth, so a cadence wider than 2 degrades the five 2-deep
/// sub-stacks to fully resident rather than erroring — which is why the domain is stated once and
/// not per depth. 10 is therefore the widest cadence that means anything here: it holds one *whole*
/// deep sub-stack, i.e. one re-open per stack per step instead of ten.
///
/// ## Measured: four cadences, one saving — except at the smallest advertised output
///
/// `transformer_window_sweep_and_streamed_output_identity` (`SDXL_WINDOW_PROBE_TIER` /
/// `SDXL_WINDOW_PROBE_SIZE` drive the off-default rows), `realvisxl`, fresh bundle per row, every row
/// byte-identical to its resident control. Request peak in GiB:
///
/// | tier | output | control | cadence 1 | 2 | 5 | 10 | spread |
/// |---|---:|---:|---:|---:|---:|---:|---:|
/// | bf16 | 1024² | 19.003 | 14.936 | 14.936 | 14.936 | 14.936 | **0** |
/// | q8 | 1024² | 17.693 | 15.516 | 15.516 | 15.516 | 15.516 | **0** |
/// | q4 | 1024² | 16.654 | 15.493 | 15.493 | 15.493 | 15.493 | **0** |
/// | q8 | 768² | 11.546 | 9.368 | 9.368 | 9.368 | 9.368 | **0** |
/// | bf16 | 512² | 8.809 | 4.742 | 4.742 | 4.742 | 4.742 | **0** |
/// | q8 | 512² | 7.499 | *withdrawn — instrument resolution 5.67%, no cadence claim possible* |||||
/// | q4 | 512² | 6.460 | *withdrawn — same artifact, and never in the published table* |||||
///
/// Five of six configurations are flat to the millibyte. **The sixth is not, and it is the advertised
/// `min_size`** — which is exactly where the phase-separation argument below says the flatness should
/// fail, so it is a confirmed prediction rather than an anomaly.
///
/// **Read this table as an observation, not as a gated invariant.** Exactly one row of it is
/// asserted — **q8 / 1024²**, the sweep's default tier and geometry. The others come from re-running
/// the sweep under `SDXL_WINDOW_PROBE_TIER` / `SDXL_WINDOW_PROBE_SIZE`, and in that mode the
/// flatness is reported rather than asserted. Timing is diagnostic output only. (And even the
/// asserted row is asserted by an `#[ignore]`d real-weight test, so it is gated by a human running
/// it against cached weights, not by CI.)
///
/// ### SC-17679: the 512² "non-flat" row was the instrument, not the family
///
/// Every row above was originally taken by a harness with **no warm-up of any kind**, which made the
/// cadence-1 row the first *windowed* row in its process. Re-measuring with a discarded row ahead of
/// every peak left five of the six rows byte-identical to what was published — and moved the 512² q8
/// row's values onto **different cadences** while keeping the same spread. That prompted three
/// controls, and together they settle it:
///
/// 1. **Execution-order permutation** (`SDXL_WINDOW_PROBE_ORDER`). Running the same four cadences
///    as `10,5,2,1` and as `2,10,1,5` produces the *same sequence of peaks in the same positions*:
///    `5.095, 5.384, 5.321, 5.384`, whichever cadence occupies each slot. The low value follows
///    **whichever cadence runs first after the warm-up** — cadence 10 when 10 runs first, cadence 2
///    when 2 runs first.
/// 2. **Identical-request repetition**
///    (`identical_requests_reproduce_once_the_allocator_has_settled`). Eight *byte-identical*
///    requests at 512² q8 read `5.3213 5.3214 5.0949 5.3839 5.3213 5.3838 5.3214 5.3838` — a
///    **5.67% spread with the input held constant**, wandering among the same three values and never
///    converging. At q8 1024² the same eight reads are `15.5155` **eight times, 0.00%**.
/// 3. **The negative control.** In the same runs, *wall clock* tracks the cadence and not the
///    position: cadence 1 costs ~2.9-3.0 s/step and cadence 10 ~0.67-0.69 s/step in **every**
///    execution order. So the instrument is perfectly capable of detecting a real cadence effect —
///    it detects one in latency, and none in peak.
///
/// **Conclusion: cadence has no measurable effect on the request peak anywhere in the advertised
/// range.** At every cell where the instrument resolves better than the differences being claimed
/// (1024² bf16/q8/q4, 768² q8, 512² bf16) all four cadences read the *same peak to the millibyte*,
/// order-independently. At the two cells where it does not (512² q8 and 512² q4, both ~5.7%
/// resolution) **no cadence claim is possible in either direction**, and the previously published
/// "5.7% non-monotonic spread" was allocator noise on unchanged input.
///
/// The 512² q8 row is therefore **withdrawn as evidence for anything**. It was the sole measured
/// objection to a wide default, and it was never evidence.
///
/// ### The condition, restated — and still unobserved
///
/// Flatness holds **only while** `decode transient + resident remainder` exceeds the windowed
/// forward's own peak at the *widest* published cadence. The decode transient scales with output
/// area; the windowed forward's peak scales with weights and is geometry-insensitive. So the
/// inequality must reverse as the output shrinks, and the first advertised geometry where it does
/// would be the boundary of the flat region.
///
/// **That boundary has not been observed anywhere in the advertised range.** The mechanism says the
/// hardest cell is the one with the largest weights at the smallest output — `bf16` at the
/// advertised `min_size`, where the forward's peak is biggest relative to the decode transient. That
/// cell is measured **flat at 0.00% resolution**: all four cadences read 4.7422 GiB. 512² q8 was
/// believed to be the boundary; it is an instrument artifact, and 512² q4 shows the same artifact
/// and was never in the published table at all.
///
/// The condition remains the right thing to check and remains checkable — it is simply not
/// satisfied-and-broken anywhere a user can currently reach. Publishing rung 2 is the change most
/// likely to break it (see below).
///
/// ### This is coupled to rung 2, and the coupling runs the wrong way
///
/// Every row above was measured with [`DECODE_SUPPORT`] `Missing`, i.e. with the decode transient at
/// **full size** — which is precisely the term keeping the forward off the peak. A story that ships a
/// bounded decode *lowers that term deliberately*, shrinking the flat region toward the small-output
/// end and potentially eliminating it. **"Widening the cadence is free" is not a property of this
/// family; it is a property of this family with rung 2 withheld.** Publishing rung 2 invalidates
/// every rung-4 row here and must re-measure the cadence domain in the same change — see
/// [`MEMORY_CALIBRATION_FINGERPRINT`], which must be bumped with it.
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 5, 10];

/// The transformer window rung 4 executes at when a request names none.
///
/// **The widest cadence — changed from the tightest by SC-17679, on measurement.**
///
/// The previous default of 1 rested on one measured claim: that "the peak is *not*
/// cadence-independent at 512² q8, where the spread is 5.7% and non-monotonic", so a default had to
/// be safe across a range where cadence demonstrably mattered. **That claim was an instrument
/// artifact and is withdrawn** — see [`TRANSFORMER_WINDOW_SIZES`] for the three controls that
/// settled it (execution-order permutation, identical-request repetition, and the wall-clock
/// negative control). Cadence has no measurable effect on the request peak anywhere in the
/// advertised range.
///
/// What is left is a deterministic allocation and topology choice:
///
/// * **Peak: identical.** At every cell where the instrument resolves better than the difference
///   being claimed — 1024² bf16/q8/q4, 768² q8, 512² bf16 — all four cadences read the same peak to
///   the millibyte, and re-running the sweep in a permuted execution order does not move it. At the
///   two cells that do not resolve (512² q8/q4, ~5.7% on unchanged input) nothing can be claimed in
///   either direction, so they support neither default.
/// * **Topology: widest complete sub-stack.** Cadence 10 holds a whole deep sub-stack, minimizing
///   repeated materialization within the defined block plan without asserting a host-clock margin.
/// * **The mechanism's worst cell is measured flat.** Flatness should break first where the weights
///   are largest and the output smallest (`bf16` at the advertised `min_size`); that cell reads
///   4.7422 GiB at all four cadences, 0.00% resolution.
///
/// A default is what every caller who names nothing receives on every request. The widest complete
/// sub-stack keeps the same resolved allocation evidence without claiming host-dependent timing.
///
/// **The residual risk, stated so it is not rediscovered as a surprise.** The flat region is a
/// measured property of six configurations, not a proof. If some unmeasured cell *is*
/// cadence-dependent, this default under-delivers memory there, and that failure mode (a render that
/// does not fit) is harsher than the one it replaces (a slow render). Three things bound it: the
/// mechanism's hardest predicted cell is measured flat; the whole domain stays published and
/// selectable, so a caller that needs the tightest weight bound can ask for 1 explicitly against
/// calibration for that cadence; and this constant is coupled to rung 2 (below), whose publication
/// would invalidate the flat region and must re-measure the domain anyway.
///
/// This reverses the posture sc-17535 asked for in advance ("keep `TRANSFORMER_WINDOW_SIZE` at the
/// tightest"). That instruction was given against the belief that the peak varied with cadence
/// somewhere in the range. It does not, and the reasoning is recorded here rather than silently
/// overridden — the standing "do not extrapolate from `realvisxl`" half of it still holds, and is
/// why the residual risk above is stated rather than dismissed.
///
/// Reproduce before changing this constant:
/// `SDXL_WINDOW_PROBE_TIER=q8 SDXL_WINDOW_PROBE_SIZE=512 SDXL_WINDOW_PROBE_ORDER=10,5,2,1`, and
/// `identical_requests_reproduce_once_the_allocator_has_settled` for the cell's resolution.
pub const TRANSFORMER_WINDOW_SIZE: u32 = 10;

/// The rung-4 **component scopes** this provider implements.
///
/// `Dit` only — here meaning the U-Net's eleven `Transformer2D` sub-stacks, which is what this
/// architecture's denoising transformer *is*. The other candidate scope is the text encoder, and
/// SDXL's cannot supply a request saving: both CLIP towers are **shed by rung 1** before the U-Net
/// loads, so windowing them would bound a phase the prerequisite rung already releases entirely.
/// Publishing a scope whose only honest measured value is 0.0% would put a meaningless choice in
/// front of the selector.
pub const TRANSFORMER_WINDOW_COMPONENTS: &[TransformerComponent] = &[TransformerComponent::Dit];

/// The scope this provider declares as its production selection.
pub const TRANSFORMER_WINDOW_COMPONENT: TransformerComponent = TransformerComponent::Dit;

/// Calibration content fingerprint. It must change whenever quantization floors, tensor layout, or
/// execution structure change in a way that invalidates measurements taken against this provider.
///
/// Load shape is a typed evidence-key axis carried separately on [`MemoryCalibrationIdentity`]; this
/// content fingerprint stays shape-independent.
/// `v2` because the rung-4 parameter domain changed shape: [`TRANSFORMER_WINDOW_SIZES`] went from
/// one cadence to four. Evidence generated against the single-cadence draft describes a different
/// set of *selectable executions*, so it must not be reusable here — the domain change alone
/// justifies the bump. (That draft left [`TRANSFORMER_WINDOW_SIZE`] at **1**; v3 below moves it to
/// **10** on measurement, which is a second, independent reason no v2 evidence is reusable.)
///
/// `v3` (SC-17679) because **the published default changed**, from the tightest cadence to the
/// widest. [`TRANSFORMER_WINDOW_SIZE`] went 1 → 10, which changes the execution a caller who names
/// nothing receives — so every v2 row describing "the default cadence" describes a different
/// execution and must not be reused.
///
/// The measurement that forced it also invalidates v2's per-cadence rows at 512² q8 independently:
/// that row was taken by a harness with no warm-up, and its apparent per-cadence peaks
/// (1→5.321, 2→5.095, 5→5.384, 10→5.321) are **allocator noise indexed by execution position**, not
/// by cadence — eight byte-identical requests span the same 5.67%. So v2 evidence records a
/// cadence-2 peak of 5.095 GiB for a cadence with no distinguishable peak at all. Either fact alone
/// justifies the bump.
///
/// Peak evidence must not invent per-cadence differences at a cell whose instrument resolution
/// exceeds the difference being recorded — see
/// `identical_requests_reproduce_once_the_allocator_has_settled`, which measures that resolution
/// per cell. `MemoryFormulaVariable::TransformerWindowSize` remains declared because cadence is an
/// executable request parameter, not because the flat rows form a time/memory frontier.
///
/// **This fingerprint is coupled to rung 2.** The cadence rows above were all measured with
/// [`DECODE_SUPPORT`] `Missing`, and the reason most of them are flat is that the *decode* carries the
/// peak (see [`TRANSFORMER_WINDOW_SIZES`]). A story that publishes a bounded decode changes which
/// phase is peak-bearing, which invalidates every rung-4 row here — so it must bump this fingerprint
/// and re-measure the cadence domain in the same change, not inherit these numbers.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "sdxl-mlx-unet-shared-ladder-v3";

/// Whether THIS load can execute rung 4. **Four** independent facts decide it.
///
/// 1. **A re-openable snapshot.** The window rebuilds blocks from the U-Net checkpoint, so the load
///    must have come from a snapshot directory. A fused LDM/A1111 single file goes through
///    `ldm::split_ldm_checkpoint` into an in-memory `Weights` with no re-openable per-component
///    source, so `load_from_ldm_file` never arms a stream.
/// 2. **[`LoadShape::DeferredMaterialization`]**, i.e. the load was asked not to bulk-commit the
///    stacks. `OffloadPolicy` is deliberately absent from this test — phase release is a separate
///    axis, and rung 1's prerequisite is enforced on the SELECTION, not on the load.
/// 3. **A load that leaves the resident blocks unmaterialized** — see
///    [`load_leaves_blocks_lazy`]. This is the fact the shared contract states as arithmetic: *"a
///    block window over an already-materialized trunk bounds nothing — it adds a copy on top"*. On
///    MLX the trunk is lazy by default, but a **load-time quantization over a dense snapshot** packs
///    every block and forces it real. Rung 4 on that load would cost the windowing machinery for
///    zero residency saving.
/// 4. **Replayable adapters** — see `adapters_are_replayable`, the SDXL-specific one, and the one
///    that would otherwise silently drop a user's LoRA.
pub fn streamable(spec: &LoadSpec) -> bool {
    matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.weights, WeightsSource::Dir(_))
        && load_leaves_blocks_lazy(spec)
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

/// Whether this load's adapters survive a rung-4 re-materialization.
///
/// **This is the SDXL-specific fail-closed condition, and it has no analogue in any DiT adopter.**
/// `adapters::apply_sdxl_adapters_with` installs a LoRA/LoKr one of two ways, decided by whether the
/// target linear is already packed:
///
/// * on a **pre-quantized (Q4/Q8) snapshot** the base is packed, so it `push`es an
///   [`Adapter`](mlx_gen::adapters::Adapter) as a forward-time residual. That is capturable and
///   replayable, exactly like Z-Image's, and `crate::block_stream` does so;
/// * on a **dense snapshot** it calls `merge_dense_delta`, folding the delta into the base weight.
///   The snapshot on disk does not carry it. A block re-read from that snapshot would be the
///   **un-adapted** model — a plausible wrong image, no error, and no memory assertion that would
///   notice.
///
/// So adapters are replayable exactly when the on-disk U-Net is already packed at the requested
/// tier. A load that would merge simply does not arm rung 4; refusing the rung is the only honest
/// option, because the alternative is quietly rendering without the LoRA the user asked for.
pub fn adapters_are_replayable(spec: &LoadSpec) -> bool {
    if spec.adapters.is_empty() {
        return true;
    }
    // A merge happens on a DENSE base. Only an already-packed snapshot pushes residuals.
    spec.quantize.is_some() && load_leaves_blocks_lazy(spec)
}

/// Build the SDXL MLX provider contract at this [`LoadSpec`].
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
        components.vae,
        resident_overlay_components(spec)?,
    )
}

/// Provider-local identities of the auxiliary networks an SDXL load can keep resident.
const CONTROL_COMPONENT_ID: &str = "sdxl.control.branches";
const IP_ADAPTER_COMPONENT_ID: &str = "sdxl.ip_adapter.image_encoder_resampler_and_kv";
const PID_COMPONENT_ID: &str = "sdxl.pid.student_and_caption_encoder";
const ADAPTER_COMPONENT_ID: &str = "sdxl.adapters.forward_residuals";

/// The **auxiliary networks this load keeps resident alongside the base three**, priced load-exact
/// (SC-22667, E1).
///
/// The module doc has always said what rung 0 holds warm — "U-Net + control/IP/VAE/PiD" — and
/// `model::load_heavy` / `build_ldm_heavy` materialize exactly that set, yet every contract this
/// provider published carried `overlay_bytes == 0` with no component. The shared validator only
/// cross-checks the overlay legs when there is one, so the zero passed silently, and a rung-1 or
/// rung-4 selection admitted for a ControlNet / IP-Adapter / PiD load under-predicted its peak by
/// the whole overlay set. Kolors, which loads these through this crate's own loaders, has priced
/// them since sc-15839; this is the same derivation on the family's base provider.
///
/// Each projection mirrors what the loader actually does, not what the file stores:
///
/// * **ControlNet** — `loader::load_controlnet` opens `spec.control` and then every
///   `spec.extra_controls` entry (MultiControlNet, sc-3378) and casts each dense branch to the fp16
///   compute dtype, so a branch is priced at 2 bytes/element. The shared primitive independently
///   forces `Stored` for an already-packed triple. All branches are one `ControlBranch` component:
///   they are installed together and no published rung releases one without the others.
/// * **IP-Adapter** — the image encoder plus the first existing IP weight file in
///   `loader::IP_ADAPTER_WEIGHT_FILES` order, both cast to fp16. Both are resident: the encoder and
///   resampler live on the generator, the K/V pairs are installed into the U-Net. A
///   `WeightsSource::File` IP-Adapter is rejected by `load` before any weight is opened, so it prices
///   nothing; a directory holding neither candidate is priced as its encoder alone, and the loader
///   refuses it moments later with the actionable message.
/// * **PiD** — priced only under [`OffloadPolicy::Resident`], where `build_residency` passes
///   `load_pid = true` unconditionally and the student plus its Gemma caption encoder are held for
///   the life of the generator. The `Sequential` arm passes `req.use_pid` per request, so a
///   non-PiD generate never materializes them and a load-time contract must not charge them. Both
///   are loaded verbatim (`PidEngine::load`: `Weights::from_file` / `from_dir`, no cast), so
///   `Stored` is exact.
/// * **Adapters** — `adapters::apply_sdxl_adapters_with` installs a LoRA/LoKr one of two ways, and
///   `adapters_are_replayable` above spells out which: on a dense base it `merge_dense_delta`s the
///   factors into the U-Net weight, which adds no independent residency
///   ([`AdapterResidencyMode::Folded`], positive evidence of zero); on a pre-quantized snapshot at
///   the requested tier the merge is impossible and the factors ride as forward-time residuals
///   ([`AdapterResidencyMode::Additive`]). `None` from the shared helper means an additive stack was
///   requested and a source could not be sized; that fails closed rather than declaring a zero the
///   shared validator would wave through.
fn resident_overlay_components(spec: &LoadSpec) -> CoreResult<Vec<MemoryResidentComponent>> {
    use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};

    // Matches `load_controlnet` / `load_ip_adapter`: dense payloads are cast to the fp16 compute
    // dtype (2 bytes/element); an already-packed triple is left alone, which the shared primitive
    // handles on its own.
    let compute_dtype = |_: &_| ResidentProjection::Bfloat16;
    let mut components = Vec::new();

    let mut control_bytes = 0_u64;
    for source in spec.control.iter().chain(&spec.extra_controls) {
        let (WeightsSource::Dir(path) | WeightsSource::File(path)) = source;
        control_bytes =
            control_bytes.saturating_add(projected_safetensors_bytes(path, compute_dtype)?);
    }
    push_overlay(
        &mut components,
        CONTROL_COMPONENT_ID,
        MemoryComponentKind::ControlBranch,
        control_bytes,
    );

    if let Some(WeightsSource::Dir(root)) = &spec.ip_adapter {
        let encoder = projected_safetensors_bytes(
            root.join(crate::loader::IP_ADAPTER_IMAGE_ENCODER_FILE),
            compute_dtype,
        )?;
        let adapter = match crate::loader::IP_ADAPTER_WEIGHT_FILES
            .iter()
            .map(|file| root.join(file))
            .find(|path| path.exists())
        {
            Some(file) => projected_safetensors_bytes(file, compute_dtype)?,
            None => 0,
        };
        push_overlay(
            &mut components,
            IP_ADAPTER_COMPONENT_ID,
            MemoryComponentKind::IpAdapter,
            encoder.saturating_add(adapter),
        );
    }

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
                // beside the base model's own transformers. What the contract arithmetic consumes
                // is `MemoryComponentKind::is_auxiliary()`, which is true for it — the same
                // disposition Kolors and FLUX.1 take.
                MemoryComponentKind::AdapterStack,
                student.saturating_add(caption),
            );
        }
    }

    let adapter_mode = if spec.quantize.is_some() && load_leaves_blocks_lazy(spec) {
        AdapterResidencyMode::Additive
    } else {
        AdapterResidencyMode::Folded
    };
    let adapter_bytes =
        adapter_stack_resident_bytes(&spec.adapters, adapter_mode).ok_or_else(|| {
            CoreError::Unsupported(
                "sdxl: an adapter stack was requested on a packed base but at least one source \
                 could not be sized; refusing to declare a zero the shared validator would wave \
                 through"
                    .to_owned(),
            )
        })?;
    push_overlay(
        &mut components,
        ADAPTER_COMPONENT_ID,
        MemoryComponentKind::AdapterStack,
        adapter_bytes,
    );

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
        // `Transformer2D` sub-stacks and nothing else.
        bounded_by: None,
        residency: MemoryComponentResidency::WholeRender,
    });
}

/// Declaration-equivalent contract used by weights-free registry conformance. Structure, parameter
/// domains and prerequisites are identical; only the measured asset facts are absent.
pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    contract_with_asset_facts(provider_id, spec, 0, 0, 0, Vec::new())
}

/// Whether the registry surface this selector names leaves the U-Net blocks lazy.
///
/// [`load_leaves_blocks_lazy`] answers that by probing the packed-quant marker under the snapshot
/// root, which a weights-free witness has no directory to read. The selector already names the
/// *resolved* artifact tier, so a packed snapshot at that tier is exactly the shape it denotes and
/// re-quantizes nothing. A witness carries no adapters, so [`adapters_are_replayable`] is vacuous.
fn surface_streamable(surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec) -> bool {
    use mlx_gen::gen_core::MemoryContractSurfaceTier;

    matches!(
        surface.resolved_artifact_tier(),
        MemoryContractSurfaceTier::Bf16
            | MemoryContractSurfaceTier::Q4
            | MemoryContractSurfaceTier::Q8
    ) && matches!(surface.spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(surface.spec.weights, WeightsSource::Dir(_))
        && adapters_are_replayable(&surface.spec)
}

/// Declaration-equivalent contract resolved from the explicit surface selector.
pub(crate) fn weights_free_memory_surface_contract(
    provider_id: &str,
    surface: &mlx_gen::gen_core::MemoryContractSurfaceSpec,
) -> CoreResult<MemoryProviderContract> {
    contract_with_asset_facts_and_streamability(
        provider_id,
        &surface.spec,
        surface_streamable(surface),
        0,
        0,
        0,
        // No overlays either: sizing one means opening its checkpoint, and this path exists to
        // produce the declaration without touching a weight file.
        Vec::new(),
    )
}

fn contract_with_asset_facts(
    provider_id: &str,
    spec: &LoadSpec,
    conditioning_bytes: u64,
    transformer_bytes: u64,
    decoder_bytes: u64,
    overlays: Vec<MemoryResidentComponent>,
) -> CoreResult<MemoryProviderContract> {
    contract_with_asset_facts_and_streamability(
        provider_id,
        spec,
        streamable(spec),
        conditioning_bytes,
        transformer_bytes,
        decoder_bytes,
        overlays,
    )
}

fn contract_with_asset_facts_and_streamability(
    provider_id: &str,
    spec: &LoadSpec,
    streamable: bool,
    conditioning_bytes: u64,
    transformer_bytes: u64,
    decoder_bytes: u64,
    overlays: Vec<MemoryResidentComponent>,
) -> CoreResult<MemoryProviderContract> {
    let decode_policies = spec.decode_geometry_policies_for_loaded_contract(
        mlx_gen::gen_core::MemoryBackend::Mlx,
        loaded_tier(spec),
    )?;
    let geometry_decode = decode_policies.iter().any(|policy| {
        matches!(
            policy.disposition,
            mlx_gen::gen_core::MemoryDecodeQualityDisposition::Admitted
        )
    });
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
    // Epic SC-22657 (E2): the axes this crate's own U-Net and VAE constants declare. `model.rs`
    // pins `DTYPE = Dtype::Float16` for the U-Net and both CLIP towers, so fp16 is the real
    // activation width here rather than a memory-model literal.
    contract.architecture_facts = crate::config::architecture_facts(
        &crate::config::UNetConfig::sdxl_base(),
        &crate::config::VaeConfig::sdxl_base(),
        mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH,
    );
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
    if geometry_decode {
        variables.push(MemoryFormulaVariable::DecodeTileArea);
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
        // SC-22667: the overlay set is only a variable of the peak when there IS one, so a plain
        // txt2img load keeps the exact envelope it always declared rather than gaining a variable
        // that never varies — the same shape Kolors has published since sc-15839.
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
        decode_tiling: geometry_decode,
        attention_chunking: false,
        transformer_window_materialization: streamable,
    };
    // A Resident selection preserves the load-time defaults: SDXL's historical warm render tiles
    // nothing and windows nothing, and there is no Sequential-only shipped tiling default to
    // override (unlike SANA). The load-time `OffloadPolicy` remains the *default* for a request that
    // names no `stage_residency`, which `Resident` overrides explicitly through the contract.
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
    contract.decode_geometry_policy_authoritative = spec.decode_geometry_policy_authoritative;
    contract.adopt_decode_geometry_policies("sdxl", decode_policies)?;
    // No `pid_decode_routes`: that declaration exists to split rung 2's candidate domain between the
    // native VAE and the PiD student, and rung 2 is not selectable on this provider. The student
    // keeps its own internal auto-planning (`mlx_gen_pid::mint_planned_decoder_with_tiling` with
    // `selected = None`) exactly as it did before this story — unchanged behaviour, not a removal.
    contract.pid_decode_routes = None;
    if streamable {
        // Rung 4 bounds the U-Net's denoise-phase weight residency. Without rung 1 engaged, both
        // CLIP towers stay resident alongside it and the REQUEST peak does not move — a phase win
        // that is not a saving. Declaring the edge lets the shared selector refuse that composition
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

/// Refuse a bounded-decode geometry. Rung 2 is `Missing` on this provider ([`DECODE_SUPPORT`]), so
/// there is no admitted domain to validate against and every geometry is out of it.
///
/// This is a *rejection* rather than an absence on purpose. The shared `validate_selection` already
/// refuses a rung the contract declares `Missing`, but a calibration harness that hand-builds a
/// [`GenerationMemory`](mlx_gen::gen_core::GenerationMemory) and calls `generate` never crosses
/// admission — and the tiling mechanism is still present in the crate as the sweep's subject. This
/// closes the path from "the code exists" to "a render silently used it".
fn refuse_decode(provider_id: &str, edge: Option<u32>, overlap: Option<u32>) -> CoreError {
    CoreError::Unsupported(format!(
        "{provider_id}: bounded decode is not selectable under the route-blind fallback (rung 2 is \
         declared Missing). SC-19753 admits only exact sealed route/tier/mode/overlay/geometry \
         quality rows; no applicable admitted row reached this fallback. Requested edge {edge:?} \
         overlap {overlap:?}."
    ))
}

/// Validate a rung-4 transformer window against the published domain — the window-size twin of
/// `validate_decode`, on the same layers, for the same reason.
///
/// The shared `validate_selection` already refuses an out-of-domain window at admission, so this is
/// defense in depth. It is not redundant: the direction it closes is the dangerous one — *executing*
/// at a wider cadence than the declared domain **under**-predicts peak, and a selector that sized a
/// fit off the published domain would then be handed a render that does not fit.
fn validate_window(size: u32) -> CoreResult<()> {
    if !TRANSFORMER_WINDOW_SIZES.contains(&size) {
        return Err(CoreError::Unsupported(format!(
            "sdxl transformer window {size} is outside the calibrated domain \
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
        // SDXL advertises txt2img, img2img, inpaint, control and IP-Adapter reference routes, so the
        // mode axis is deliberately permissive — unlike a text-to-image-only family. What is NOT
        // permissive is the geometry: a bounded decode must name a geometry from the route it will
        // actually execute on.
        if contract.engages_selection(&context.selection, MemoryStrategy::BoundedDecode)
            && contract
                .capability(MemoryStrategy::BoundedDecode)
                .is_none_or(|capability| capability.parameters.decode_geometry_policies.is_empty())
        {
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
                 U-Net seam), and \
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
                     a snapshot directory whose adapters (if any) are replayable — see \
                     memory_strategy::streamable",
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

/// The widest `Transformer2D` sub-stack depth, which is the window domain the shared request scope
/// validates against.
///
/// Derived from the config rather than hardcoded, and it is the **maximum** rather than the total:
/// the shared hook's contract is "a window of `w` starting at block `f` covers
/// `min(w, blocks - f)`", which is a statement about ONE stack. SDXL runs eleven, and the widest is
/// the only one that can bound the published cadence.
fn widest_transformer_stack() -> usize {
    let cfg = crate::config::UNetConfig::sdxl_base();
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
    config.load_shape = context.load_shape;
    mlx_gen::request_scope::authorize_selected_geometry_decode(&mut config, contract, context)?;
    // Rung 2 may be engaged only by the exact policy authorization above. Rung 3 remains `Missing`,
    // so attention chunking is never set.
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

/// Rung 4: the requested window cadence, or `None` for the resident stacks. A scope this family does
/// not implement — or a cadence outside the measured [`TRANSFORMER_WINDOW_SIZES`] — is a typed
/// rejection rather than a silently narrowed (or silently *widened*) execution.
pub(crate) fn transformer_window_size(req: &GenerationRequest) -> mlx_gen::Result<Option<usize>> {
    let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
        return Ok(None);
    };
    let component = memory
        .transformer_window_component
        .unwrap_or(TRANSFORMER_WINDOW_COMPONENT);
    if !TRANSFORMER_WINDOW_COMPONENTS.contains(&component) {
        return Err(mlx_gen::Error::Unsupported(format!(
            "sdxl implements only the {TRANSFORMER_WINDOW_COMPONENT:?} transformer window \
             component, got {component:?}"
        )));
    }
    let size = memory
        .transformer_window_size
        .unwrap_or(TRANSFORMER_WINDOW_SIZE);
    validate_window(size).map_err(|error| {
        mlx_gen::Error::Unsupported(format!("sdxl transformer window rejected: {error}"))
    })?;
    Ok(Some(size as usize))
}

/// Resolve rung 2 for this request. The route-blind default remains `Missing`
/// ([`DECODE_SUPPORT`]); only a contract carrying the exact sealed geometry row may tile.
///
/// It returns `Ok(None)` — the exact single-pass decode — when the request did not ask for a bounded
/// decode, and a typed error when it asked without exact authorization. The error direction is what
/// matters: `Ok(None)` for a
/// request that *did* ask would silently render an unbounded decode while the caller believed it had
/// selected a bounded one, which is the false-green this seam exists to prevent.
#[cfg(test)]
pub(crate) fn decode_tiling(req: &GenerationRequest) -> mlx_gen::Result<Option<TilingConfig>> {
    decode_tiling_with_contract(req, None)
}

pub(crate) fn decode_tiling_for_contract(
    req: &GenerationRequest,
    contract: &MemoryProviderContract,
) -> mlx_gen::Result<Option<TilingConfig>> {
    decode_tiling_with_contract(req, Some(contract))
}

fn decode_tiling_with_contract(
    req: &GenerationRequest,
    contract: Option<&MemoryProviderContract>,
) -> mlx_gen::Result<Option<TilingConfig>> {
    if let Some(control) = mlx_gen::diagnostics::benchmark_decode_control() {
        mlx_gen::diagnostics::record_geometry_decode_decision(
            mlx_gen::diagnostics::DecodePolicyDisposition::Unchanged,
            mlx_gen::diagnostics::DecodePathDisposition::Tiled,
            None,
        );
        return Ok(Some(control.tiling_config()));
    }
    let Some(memory) = req.memory.filter(|memory| memory.tile_vae_decode) else {
        mlx_gen::diagnostics::record_geometry_decode_decision(
            mlx_gen::diagnostics::DecodePolicyDisposition::Unchanged,
            mlx_gen::diagnostics::DecodePathDisposition::Dense,
            None,
        );
        return Ok(None);
    };
    if contract.is_some_and(|contract| {
        contract
            .capability(MemoryStrategy::BoundedDecode)
            .is_some_and(|capability| !capability.parameters.decode_geometry_policies.is_empty())
    }) {
        let edge = memory.decode_tile_edge.ok_or_else(|| {
            mlx_gen::Error::Unsupported(
                "sdxl: geometry-aware bounded decode requires an exact tile edge".to_owned(),
            )
        })?;
        let overlap = memory.decode_overlap.ok_or_else(|| {
            mlx_gen::Error::Unsupported(
                "sdxl: geometry-aware bounded decode requires an exact overlap".to_owned(),
            )
        })?;
        let evidence = mlx_gen::request_scope::validate_geometry_decode_authorization(
            crate::MODEL_ID,
            req,
            edge,
            overlap,
        )?;
        mlx_gen::diagnostics::record_geometry_decode_decision(
            mlx_gen::diagnostics::DecodePolicyDisposition::GeometryTiled,
            mlx_gen::diagnostics::DecodePathDisposition::Tiled,
            Some(&evidence),
        );
        return Ok(Some(TilingConfig::spatial_only(
            edge as i32,
            overlap as i32,
        )));
    }
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
            "sdxl: bounded attention is not selectable on this provider (rung 3 is declared \
             Missing): it moves the peak 0.00% at both measured scopes (the request and the U-Net \
             seam), and the \
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
    use mlx_gen::{AdapterSpec, Quant};

    fn spec(shape: LoadShape) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent/sdxl-contract".into()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(shape)
    }

    fn contract(shape: LoadShape) -> MemoryProviderContract {
        weights_free_memory_strategy_contract(crate::MODEL_ID, &spec(shape)).unwrap()
    }

    fn write_fused_contract_fixture(path: &std::path::Path) {
        let entries = [
            (
                "model.diffusion_model.time_embed.0.weight",
                vec![64, 64],
                64 * 64 * 2,
            ),
            (
                "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
                vec![64, 64],
                64 * 64 * 2,
            ),
            (
                "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_weight",
                vec![192, 64],
                192 * 64 * 2,
            ),
            (
                "first_stage_model.decoder.up.1.block.2.nin_shortcut.weight",
                vec![1, 1, 1, 1],
                2,
            ),
        ];
        let mut offset = 0usize;
        let mut header = serde_json::Map::new();
        for (name, shape, bytes) in entries {
            header.insert(
                name.to_owned(),
                serde_json::json!({
                    "dtype": "BF16",
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let mut json = serde_json::to_vec(&header).unwrap();
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut file = (json.len() as u64).to_le_bytes().to_vec();
        file.extend(json);
        file.resize(file.len() + offset, 0);
        std::fs::write(path, file).unwrap();
    }

    /// AC (SC-22662): the published contract carries the crate's own U-Net/VAE axes, and passes the
    /// shared facts conformance check on both load shapes.
    #[test]
    fn the_contract_publishes_the_crate_architecture_axes() {
        for shape in [
            LoadShape::DeferredMaterialization,
            LoadShape::EagerMaterialization,
        ] {
            let contract = contract(shape);
            assert_eq!(
                contract.architecture_facts,
                crate::config::architecture_facts(
                    &crate::config::UNetConfig::sdxl_base(),
                    &crate::config::VaeConfig::sdxl_base(),
                    2,
                ),
                "{shape:?}"
            );
            assert_eq!(contract.architecture_facts.head_dim, Some(64));
            assert_eq!(contract.architecture_facts.latent_channels, Some(4));
            assert_eq!(contract.architecture_facts.vae_spatial_scale, Some(8));
            assert_eq!(contract.architecture_facts.activation_dtype_width, Some(2));
            assert!(contract.architecture_facts.has_declared_architecture_axis());
            gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        }
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

    /// **The SDXL-specific fail-closed condition.** A dense-snapshot adapter load merges its LoRA
    /// into the base weight; a re-materialized block would be the un-adapted model. Rung 4 must not
    /// be declared for it.
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
        let mut with_adapter = dense.clone();
        with_adapter.adapters = vec![adapter];
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

    /// A single-file (LDM/A1111) source has no re-openable per-component snapshot.
    #[test]
    fn a_single_file_source_cannot_stream() {
        let fused = LoadSpec::new(WeightsSource::File("/nonexistent/sdxl.safetensors".into()))
            .with_load_shape(LoadShape::DeferredMaterialization);
        assert!(!streamable(&fused));
    }

    #[test]
    fn fused_file_contract_projects_remapped_components_and_selected_quant() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("sdxl.safetensors");
        write_fused_contract_fixture(&file);
        let spec = LoadSpec::new(WeightsSource::File(file)).with_quant(Quant::Q4);

        let contract = memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();

        assert_eq!(contract.asset_facts.transformer_bytes, 2_304);
        assert_eq!(contract.asset_facts.conditioning_bytes, 9_216);
        assert_eq!(contract.asset_facts.decoder_bytes, 4);
        assert_eq!(contract.asset_facts.base_bytes, 11_524);
        assert!(contract.conformance_errors().is_empty());
    }

    /// **Rungs 2 and 3 are refused on every layer**, and the refusal names why.
    ///
    /// This is the test that keeps a measured-and-rejected rung from drifting back into
    /// reachability: the mechanisms are still in the crate (the sweep drives them), so the only
    /// thing standing between `Autoencoder::decode_tiled` and a production render is these
    /// refusals.
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
        assert!(
            err.to_string().contains("no applicable admitted row"),
            "got: {err}"
        );

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

    #[test]
    fn benchmark_fixed_tile_control_is_isolated_from_production_refusal() {
        let scope = mlx_gen::diagnostics::begin_benchmark_request(
            "sdxl-fixed-tile",
            "sdxl_unet",
            &[mlx_gen::diagnostics::GEOMETRY_AWARE_DECODE],
            Some(mlx_gen::diagnostics::BenchmarkDecodeControl {
                spatial_tile_px: 256,
                spatial_overlap_px: 64,
                temporal_tile_frames: None,
                temporal_overlap_frames: None,
            }),
            |_| {},
        )
        .unwrap();
        let tiling = decode_tiling(&GenerationRequest::default())
            .unwrap()
            .unwrap();
        let spatial = tiling.spatial.unwrap();
        assert_eq!((spatial.tile_px, spatial.overlap_px), (256, 64));
        assert!(tiling.temporal.is_none());
        let report = scope.finish();
        assert!(report.counters.iter().any(|counter| matches!(
            counter,
            mlx_gen::diagnostics::DiagnosticCounter::Toggle {
                toggle: mlx_gen::diagnostics::GEOMETRY_AWARE_DECODE,
                disposition: mlx_gen::diagnostics::ToggleDisposition::Applied,
                count: 1,
            }
        )));
        assert!(report.counters.iter().any(|counter| matches!(
            counter,
            mlx_gen::diagnostics::DiagnosticCounter::DecodePolicy {
                disposition: mlx_gen::diagnostics::DecodePolicyDisposition::Unchanged,
                decode_path: mlx_gen::diagnostics::DecodePathDisposition::Tiled,
                production_evidence_sha256: None,
                count: 1,
            }
        )));

        assert!(decode_tiling(&GenerationRequest::default())
            .unwrap()
            .is_none());
    }

    #[test]
    fn p9_unadmitted_route_receipts_the_unchanged_dense_path() {
        let scope = mlx_gen::diagnostics::begin_request_with_toggles(
            "sdxl-policy",
            "sdxl_unet",
            &[mlx_gen::diagnostics::GEOMETRY_AWARE_DECODE],
        )
        .unwrap();
        assert!(decode_tiling(&GenerationRequest::default())
            .unwrap()
            .is_none());
        let report = scope.finish();
        assert!(report.counters.iter().any(|counter| matches!(
            counter,
            mlx_gen::diagnostics::DiagnosticCounter::DecodePolicy {
                disposition: mlx_gen::diagnostics::DecodePolicyDisposition::Unchanged,
                decode_path: mlx_gen::diagnostics::DecodePathDisposition::Dense,
                production_evidence_sha256: None,
                count: 1,
            }
        )));
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
            assert_eq!(
                transformer_window_size(&request(windowed(Some(*size), None))).unwrap(),
                Some(*size as usize)
            );
        }
        // Out-of-domain cadences, chosen to sit *between* and *beyond* the published ones rather
        // than merely far from them: 3/4/6/7/9 are interior gaps a "clamp to the nearest legal
        // value" bug would silently absorb, and 11/70 are past the widest sub-stack.
        for bad in [0_u32, 3, 4, 6, 7, 9, 11, 70] {
            assert!(
                !TRANSFORMER_WINDOW_SIZES.contains(&bad),
                "the negative case list must stay disjoint from the published domain"
            );
            assert!(
                transformer_window_size(&request(windowed(Some(bad), None))).is_err(),
                "window {bad} must be refused"
            );
        }
        // The request-side default is the TIGHTEST cadence — a request that engages rung 4 without
        // naming a size must get the weight bound this provider can always make good on, not the
        // point that is cheapest at the geometries that happen to have been measured. This is the
        // resolver half of `TRANSFORMER_WINDOW_SIZE`'s claim; the assertion below is the other half,
        // and the two must not be allowed to drift apart.
        assert_eq!(
            transformer_window_size(&request(windowed(None, None))).unwrap(),
            Some(TRANSFORMER_WINDOW_SIZE as usize)
        );
        assert_eq!(
            TRANSFORMER_WINDOW_SIZE,
            *TRANSFORMER_WINDOW_SIZES.last().unwrap(),
            "the default must be the WIDEST published cadence (SC-17679). Cadence has no measurable \
             effect on the request peak anywhere in the advertised range — the 512² q8 'non-flat' \
             row that previously justified defaulting to the tightest is allocator noise on \
             unchanged input (5.67% across eight IDENTICAL requests) and its apparent cadence \
             ordering follows execution position, not cadence. The widest is the deterministic \
             complete-sub-stack choice, not a host-clock claim. If a future measurement finds a \
             cell where the peak really is cadence-dependent, change this back and say where — do \
             not change it back on the withdrawn 512² row"
        );
        for component in [
            TransformerComponent::TextEncoder,
            TransformerComponent::Both,
        ] {
            assert!(
                transformer_window_size(&request(windowed(Some(1), Some(component)))).is_err(),
                "component {component:?} must be refused, never narrowed to Dit"
            );
        }
        assert!(
            transformer_window_size(&request(GenerationMemory::default()))
                .unwrap()
                .is_none()
        );
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
    /// composition that would bound the U-Net's weights while both CLIP towers stayed resident.
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
        let _ = &deferred;
        // And the eager contract publishes no such edge, because it publishes no rung 4.
        let eager = contract(LoadShape::EagerMaterialization);
        assert!(eager.additional_prerequisites.is_empty());
    }

    /// The window domain the shared request scope validates against is the WIDEST sub-stack, and
    /// every published cadence fits inside it.
    #[test]
    fn the_published_window_domain_fits_the_widest_sub_stack() {
        let widest = widest_transformer_stack();
        assert_eq!(widest, 10, "SDXL's deepest Transformer2D holds 10 blocks");
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
        // THREE separate invariants read a position and mean a superlative by it. `[0]` is "the
        // tightest" both just above (it must bound the 2-deep sub-stacks) and in
        // `the_request_side_window_resolver_refuses_out_of_domain_cadences`, where
        // `TRANSFORMER_WINDOW_SIZE` must equal it.
        // `.last()` is "the widest", checked just below against `widest_transformer_stack()`.
        // An unsorted domain would silently redirect all three at the wrong element.
        assert!(
            TRANSFORMER_WINDOW_SIZES.windows(2).all(|w| w[0] < w[1]),
            "the domain must be strictly ascending: {TRANSFORMER_WINDOW_SIZES:?}"
        );
        // The widest published cadence must be the widest sub-stack. Anything larger is
        // indistinguishable from it (`BlockPlan::new` clamps), so publishing it would put a
        // choice in front of the selector that cannot differ from one it already has.
        assert_eq!(
            *TRANSFORMER_WINDOW_SIZES.last().unwrap() as usize,
            widest,
            "the widest cadence must equal the deepest sub-stack — one re-open per stack per step"
        );
    }

    /// **The `transformer_layers_per_block[0]`-is-never-read trap**, pinned — this is the defect
    /// SC-16355 was filed over, and it bit this PR's own documentation once already.
    ///
    /// SDXL's level-0 down block is a plain `DownBlock2D` with no attention, so the `1` sitting at
    /// `transformer_layers_per_block[0]` describes nothing that exists. Two things follow, and both
    /// were stated wrongly at some point:
    ///
    /// * `widest_transformer_stack` must filter on `down_block_types` rather than taking a bare
    ///   `max` — that is what this asserts by mutation-resistant construction below;
    /// * the shallowest **attention-bearing** level is level 1, one downsample in, so at 1024² the
    ///   self-attention key axis tops out at 64·64 = **4096** and not at the latent's 128·128 =
    ///   16384. `crate::unet::transformer` cites that number.
    #[test]
    fn the_level_zero_transformer_layer_count_describes_no_attention() {
        let cfg = crate::config::UNetConfig::sdxl_base();
        assert!(
            !cfg.down_block_types[0].contains("CrossAttn"),
            "level 0 is {} — if it ever gains attention, both the window domain and the 4096 key-axis \
             claim in crate::unet::transformer must be re-derived",
            cfg.down_block_types[0]
        );
        assert_eq!(
            cfg.transformer_layers_per_block[0], 1,
            "the inert level-0 entry is still 1; it is never read"
        );
        // The bare `max` a reader reaches for first would also be 10 here, so that alone proves
        // nothing. What separates the correct filter from the naive one is that the *sum* over
        // attention-bearing levels excludes level 0 entirely.
        let attention_levels: Vec<i32> = cfg
            .transformer_layers_per_block
            .iter()
            .enumerate()
            .filter(|(i, _)| cfg.down_block_types[*i].contains("CrossAttn"))
            .map(|(_, layers)| *layers)
            .collect();
        assert_eq!(
            attention_levels,
            vec![2, 10],
            "only levels 1 and 2 own Transformer2D sub-stacks"
        );
        // At 1024² the latent is 128², halved once before the first attention-bearing level.
        let shallowest_attention_level = cfg
            .down_block_types
            .iter()
            .position(|kind| kind.contains("CrossAttn"))
            .expect("SDXL has cross-attention down blocks");
        let latent_edge = 1024 / 8;
        let key_axis = (latent_edge >> shallowest_attention_level) as u32;
        assert_eq!(
            key_axis * key_axis,
            4096,
            "the largest self-attention key axis at 1024² is 4096, not the full-resolution 16384"
        );
    }

    /// One-tensor safetensors file; `dtype` must agree with `width` because the shared header
    /// reader refuses a payload length that is not `elements * dtype.size()`.
    fn write_tensor(path: &std::path::Path, dtype: &str, elements: usize, width: usize) {
        let data_bytes = elements * width;
        let mut header = format!(
            "{{\"weight\":{{\"dtype\":\"{dtype}\",\"shape\":[{elements}],\"data_offsets\":[0,{data_bytes}]}}}}"
        )
        .into_bytes();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.resize(bytes.len() + data_bytes, 0);
        std::fs::write(path, bytes).unwrap();
    }

    /// Feature-end review (SC-22667, E1): the module doc has always said what rung 0 holds warm —
    /// "U-Net + control/IP/VAE/PiD" — and `model::load_heavy` / `build_ldm_heavy` materialize
    /// exactly that, yet every contract published `overlay_bytes == 0` with no component. Each
    /// auxiliary network is now declared once in `overlay_bytes` and once as a typed component, at
    /// the width the loader materializes it in, and `base_bytes` stays exactly the sum of the three
    /// base-model fields.
    ///
    /// Mutation that fails this: passing `Vec::new()` at the production `memory_strategy_contract`
    /// call site instead of `resident_overlay_components(spec)?` — `overlay_bytes` drops to 0,
    /// `resident_components()` empties, and the formula falls back to `PhaseEnvelope`. Pricing the
    /// ControlNet at `Stored` instead of the fp16 compute width reds the f32-stored branch (40 B
    /// stored, 20 B resident).
    #[test]
    fn resident_auxiliary_networks_are_declared_as_overlay_components() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let fused = root.join("sdxl.safetensors");
        write_fused_contract_fixture(&fused);
        let clean =
            LoadSpec::new(WeightsSource::File(fused)).with_offload_policy(OffloadPolicy::Resident);
        let base = memory_strategy_contract(crate::MODEL_ID, &clean).unwrap();
        assert_eq!(base.asset_facts.overlay_bytes, 0);
        assert!(base.resident_components().is_empty());
        assert!(
            matches!(base.formula, MemoryFormulaKind::PhaseEnvelope { .. }),
            "a clean base load must stay on the componentless formula"
        );

        // Two ControlNet branches, one stored f16 and one stored f32. `load_controlnet` casts BOTH
        // to the fp16 compute dtype, so the f32 file is priced at half its stored size.
        let control = root.join("control.safetensors");
        write_tensor(&control, "F16", 40, 2);
        let extra = root.join("extra_control.safetensors");
        write_tensor(&extra, "F32", 10, 4);
        // An h94/IP-Adapter layout: the ViT-H encoder plus the `plus` weights. No plus-face file,
        // so the loader's fallback candidate is the one that must be priced.
        let ip = root.join("ip");
        std::fs::create_dir_all(ip.join("models/image_encoder")).unwrap();
        std::fs::create_dir_all(ip.join("sdxl_models")).unwrap();
        write_tensor(
            &ip.join(crate::loader::IP_ADAPTER_IMAGE_ENCODER_FILE),
            "F16",
            30,
            2,
        );
        write_tensor(
            &ip.join(crate::loader::IP_ADAPTER_WEIGHT_FILES[1]),
            "F16",
            20,
            2,
        );
        // A PiD pair, loaded verbatim: the student file and a Gemma shard directory that carries no
        // merged single file, so `PidEngine::load` falls back to the directory.
        let pid = root.join("pid.safetensors");
        write_tensor(&pid, "BF16", 10, 2);
        let gemma = root.join("gemma");
        std::fs::create_dir_all(&gemma).unwrap();
        write_tensor(
            &gemma.join("model-00001-of-00001.safetensors"),
            "BF16",
            5,
            2,
        );
        // A LoRA on a DENSE base is merged into the U-Net weight: folded, no independent residency.
        let lora = root.join("lora.safetensors");
        std::fs::write(&lora, vec![0_u8; 64]).unwrap();

        let mut spec = clean
            .clone()
            .with_control(WeightsSource::File(control))
            .with_pid(WeightsSource::File(pid), WeightsSource::Dir(gemma));
        spec.extra_controls.push(WeightsSource::File(extra));
        spec.ip_adapter = Some(WeightsSource::Dir(ip));
        spec.adapters
            .push(AdapterSpec::new(lora, 1.0, mlx_gen::AdapterKind::Lora));

        let contract = memory_strategy_contract(crate::MODEL_ID, &spec).unwrap();
        let control_bytes: u64 = 40 * 2 + 10 * 2;
        let ip_bytes: u64 = 30 * 2 + 20 * 2;
        let pid_bytes: u64 = 10 * 2 + 5 * 2;
        assert_eq!(
            contract.asset_facts.overlay_bytes,
            control_bytes + ip_bytes + pid_bytes
        );
        assert_eq!(
            contract.auxiliary_resident_bytes(),
            contract.asset_facts.overlay_bytes
        );
        assert_eq!(
            contract.asset_facts.base_bytes, base.asset_facts.base_bytes,
            "an auxiliary network must never move the base decomposition"
        );
        assert_eq!(
            contract.total_resident_bytes(),
            base.asset_facts.base_bytes + contract.asset_facts.overlay_bytes
        );
        let components = contract.resident_components();
        let ids: Vec<&str> = components
            .iter()
            .map(|component| component.id.as_str())
            .collect();
        assert_eq!(
            ids,
            vec![
                CONTROL_COMPONENT_ID,
                IP_ADAPTER_COMPONENT_ID,
                PID_COMPONENT_ID
            ],
            "a merged LoRA declares no component"
        );
        assert_eq!(components[0].kind, MemoryComponentKind::ControlBranch);
        assert_eq!(components[0].resident_bytes, control_bytes);
        assert_eq!(components[1].kind, MemoryComponentKind::IpAdapter);
        assert_eq!(components[1].resident_bytes, ip_bytes);
        assert_eq!(components[2].resident_bytes, pid_bytes);
        assert!(matches!(
            &contract.formula,
            MemoryFormulaKind::ComponentPhaseEnvelope { variables, .. }
                if variables.contains(&MemoryFormulaVariable::OverlayBytes)
        ));
        assert!(
            contract.conformance_errors().is_empty(),
            "{:?}",
            contract.conformance_errors()
        );
        gen_core_testkit::assert_memory_contract_facts_conform(&contract);

        // Under `Sequential` the PiD pair follows `req.use_pid` per request, so a load-time contract
        // must not charge it.
        let sequential = spec.clone().with_offload_policy(OffloadPolicy::Sequential);
        let sequential = memory_strategy_contract(crate::MODEL_ID, &sequential).unwrap();
        assert_eq!(
            sequential.asset_facts.overlay_bytes,
            control_bytes + ip_bytes
        );
        assert_eq!(sequential.resident_components().len(), 2);
    }

    /// Feature-end review (SC-22667, E1): `apply_sdxl_adapters_with` merges a LoRA into a dense
    /// U-Net weight (no independent residency) but pushes it as a forward-time residual on a
    /// pre-quantized snapshot at the requested tier, where the merge is impossible — the same
    /// distinction `adapters_are_replayable` draws for rung 4. Only the residual case is a resident
    /// overlay.
    ///
    /// Mutation that fails this: pinning the residency mode to `Folded` — the packed leg reads back
    /// `overlay_bytes == 0` while the loader holds 2048 B of factors — or to `Additive`, which
    /// charges the dense leg for factors that were merged away.
    #[test]
    fn a_lora_on_a_packed_snapshot_is_priced_as_a_resident_overlay() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("unet")).unwrap();
        // The packed-tier marker `load_leaves_blocks_lazy` reads: at the requested tier the load
        // re-quantizes nothing, and the LoRA cannot be merged into a packed weight.
        std::fs::write(
            root.join("unet/config.json"),
            br#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        write_tensor(
            &root.join("unet/diffusion_pytorch_model.safetensors"),
            "F16",
            100,
            2,
        );
        let lora = root.join("lora.safetensors");
        std::fs::write(&lora, vec![0_u8; 2048]).unwrap();
        let mut adapted = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()));
        adapted
            .adapters
            .push(AdapterSpec::new(lora, 1.0, mlx_gen::AdapterKind::Lora));

        // No tier requested: a dense load, so the factors are merged and add nothing.
        let folded = memory_strategy_contract(crate::MODEL_ID, &adapted).unwrap();
        assert_eq!(folded.asset_facts.overlay_bytes, 0);
        assert!(folded.resident_components().is_empty());

        // The pre-quantized base at its own tier: forward-time residuals beside the packed U-Net.
        let packed = adapted.clone().with_quant(Quant::Q4);
        let additive = memory_strategy_contract(crate::MODEL_ID, &packed).unwrap();
        assert_eq!(additive.asset_facts.overlay_bytes, 2048);
        assert_eq!(
            additive.asset_facts.base_bytes, folded.asset_facts.base_bytes,
            "an adapter is auxiliary: it must never move the base decomposition"
        );
        let components = additive.resident_components();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].id, ADAPTER_COMPONENT_ID);
        assert_eq!(components[0].kind, MemoryComponentKind::AdapterStack);
        assert_eq!(components[0].resident_bytes, 2048);
        assert!(
            additive.conformance_errors().is_empty(),
            "{:?}",
            additive.conformance_errors()
        );

        // An additive stack that cannot be sized fails closed rather than declaring a zero the
        // shared validator would wave through.
        let mut unsizable =
            LoadSpec::new(WeightsSource::Dir(root.to_path_buf())).with_quant(Quant::Q4);
        unsizable.adapters.push(AdapterSpec::new(
            root.join("absent.safetensors"),
            1.0,
            mlx_gen::AdapterKind::Lora,
        ));
        assert!(memory_strategy_contract(crate::MODEL_ID, &unsizable).is_err());
    }
}
