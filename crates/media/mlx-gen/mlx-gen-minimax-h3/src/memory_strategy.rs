//! MiniMax-H3's shared-ladder `MemoryProviderContract` on MLX/Metal (sc-18659).
//!
//! **This is the ladder's first video family.** Every one of the 23 providers that declared a
//! contract before it is an image family, and a family without a contract silently resolves to
//! [`MemoryProviderContract::compatibility_default`] — resident-only, no levers, no fitted
//! estimate. The contract vocabulary already expresses video without extension:
//! [`MemoryFormulaVariable::FrameCount`] exists, and the phase envelope is exactly the shape a
//! staged video pipeline needs.
//!
//! # Attribute every number to a stage, or the contract lies
//!
//! The epic produced two wrong handoffs by attributing a **process-wide** peak to a component.
//! Nothing in this crate measured per stage, so a high-water mark set by a tier-independent stage
//! read as a defect in the tiered one. The numbers below each name the stage they belong to:
//!
//! | Quantity | Stage | Value | Source |
//! | --- | --- | --- | --- |
//! | [`CONDITIONING_STAGE_PEAK_BYTES`] | conditioning | 53.07 GB | measured in isolation |
//! | [`DENOISE_RESIDENT_BF16_BYTES`] | denoise | 40.43 GB | measured, post-AdaLN-evict |
//! | [`DENOISE_RESIDENT_Q4_BYTES`] | denoise | 11.63 GB | measured, post-AdaLN-evict |
//! | [`ADALN_EVICTED_BYTES`] | denoise | 26,020,915,200 B | exact tensor bytes |
//!
//! The **~53 GB floor is the dense Qwen3-VL-32B text encoder**, not the DiT and not activation
//! pressure. It runs *before* the DiT is mapped and `reset_peak_memory()` fires before `generate`,
//! so its high-water masks every later stage. That is why the observed peak is flat across tier,
//! canvas and duration — the binding stage's cost is a function of prompt tokens only. **The
//! formula below must not bake that flatness in**: it is an artifact of which stage binds today,
//! not a property of the DiT, and sc-19120 (a packed TE tier) removes it.
//!
//! The DiT is genuinely tiered: 40.43 GB bf16 → 11.63 GB q4 denoise-resident, a 28.80 GB
//! reduction. Quantization is packed end to end — there is no `dequantize` on the DiT path.
//!
//! # What this provider implements today, and what it does not
//!
//! [`MemoryStrategy::StagedResidency`] is **structural** here rather than optional:
//! `MiniMaxH3::generate_impl` builds each phase's component, forces evaluation, drops it and drains
//! MLX's allocator cache (`model::release`) before the next is mapped, across all three of
//! conditioning → denoise → decode. Holding any two of the three heavy components at once does not
//! fit a sensible budget, so the pipeline has never had a co-resident mode. sc-17151 owns making
//! that residency *enforced* rather than merely observed; the mechanism itself is live in every
//! render today, which is why it is declared `Implemented` and not `Missing`.
//!
//! # Rung 2 is implemented with a domain of exactly one geometry (sc-18660)
//!
//! [`MemoryStrategy::BoundedDecode`] is `Implemented`, and its published domain is the single pair
//! ([`DECODE_TILE_EDGE`], [`DECODE_OVERLAP`]) = 256 px / 64 px. **That singleton is the finding, not
//! a placeholder.**
//!
//! The video VAE decode is bounded two ways, and *neither* geometry is a memory lever:
//!
//! * **temporally**, by the reference's own `decode_temporal` ([`crate::chunking`]) — derived from
//!   `clip_length` 17, `token_drop` 3 and `vae_ratio_t` 4, which are checkpoint facts;
//! * **spatially**, by the reference's own `_decode_clip` ([`crate::spatial_tiling`]) — 256 px tiles
//!   at a 64 px overlap with a linear cross-fade, on by default.
//!
//! sc-18786 measured what happens if the spatial geometry is treated as tunable: the released
//! frames *are* the blended-tile ones, and decoding the shipped canvas in one pass moved both
//! backends by a rel-max-abs of ~0.647. It then pinned a test asserting that a tile smaller than
//! the canvas **changes** the decode. So the usual rung-2 move — shrink the tile until the request
//! fits — would silently reintroduce that defect, and no memory assertion would catch it. The
//! contract says so in the only way a consumer honors: `validate_selection` admits 256/64 and
//! refuses every other value, and `route_gate` refuses it again at admission.
//!
//! What rung 2 *does* bound here is the decoder scratch that is genuinely free: the number of
//! decoded tiles held live during the stitch. [`crate::spatial_tiling::BoundedStitch`] streams the
//! grid row-major and retains `O(cols)` overlap strips instead of the `O(rows × cols)` whole grid
//! `stitch_tiles` holds — bit-identical output, asserted at `max|Δ| == 0.0`.
//!
//! **The audio VAE (0.61 GB) is explicitly out of scope**, and that is a declaration rather than an
//! omission: see [`AUDIO_VAE_IS_OUT_OF_SCOPE_FOR_TILING`]. It is a DAC-lineage 1-D stack whose
//! decode is a BigVGAN vocoder over a waveform, with no spatial grid to tile and a footprint 17×
//! below the video VAE's.
//!
//! # Rung 3 is measured-inapplicable on this backend (sc-18661)
//!
//! [`MemoryStrategy::BoundedAttention`] is [`MemoryStrategySupport::StructurallyNotApplicable`],
//! and the reason is a measurement rather than an argument. It was
//! [`MemoryStrategySupport::Missing`] until sc-18661 built the mechanism and ran it; the verdict is
//! what changed the declaration, not the other way round.
//!
//! **The rung optimizes a tensor this backend does not build.** MLX's fused
//! `scaled_dot_product_attention` streams the scores: measured peak tracks the `4·B·H·S·D` q/k/v/out
//! prediction exactly — 5.966 GB against 5.965 GB predicted at `S = 104_030`, where materializing
//! `[B, H, S, S]` would add ~1,212 GB (`tests/sequence_cost_real.rs`, sc-17152). That is
//! [`MemoryStrategySupport::StructurallyNotApplicable`]'s own definition: "the architecture lacks
//! the component … the rung would optimize".
//!
//! **And the remaining mechanism was measured in-forward, where it could still have won.** On MLX
//! the saving Z-Image gets from this rung is not score bounding at all — it is
//! `AttentionBudget::eval_per_chunk` cutting the lazy graph, worth −1.7 % on Z-Image's denoise phase.
//! That mechanism exists only inside a real forward, so `tests/bounded_attention_real.rs` runs it on
//! the real 50-block q4 DiT across the frame lattice, on **both** chunk axes. Every arm costs peak:
//! see [`RUNG3_MEASURED_PEAK_DELTAS`]. There is nothing left for the rung to recover here, because
//! the denoise peak is the resident DiT plus a transient that chunking *adds* to.
//!
//! Three scoping notes the next reader needs:
//!
//! * **MLX-scoped.** The contract carries a [`MemoryBackendRealization`], and
//!   `candle-gen-minimax-h3` has no fused streaming SDPA — it materializes its scores, where this
//!   rung is worth −32 % (`gen_core::attention_budget`). **Do not copy this verdict to candle.**
//! * **The mechanism is retained, not reverted.** [`crate::dit::model::MiniMaxH3Dit::forward_packed_bounded`]
//!   and its block-level seam stay wired at `BoundedAttention::UNBOUNDED`, which is byte-identical to
//!   the pre-rung forward, so no shipped render's numerics move. They are the instrument the verdict
//!   was measured with, and a verdict that is per-tier and per-budget must stay re-measurable —
//!   `tests/bounded_attention.rs` holds the seam reachable and the declaration honest.
//! * **`StructurallyNotApplicable` satisfies prerequisite edges vacuously**
//!   (`gen_core::memory_strategy`). Nothing in the shared graph names rung 3 as a prerequisite today,
//!   and rung 4's sole prerequisite is a [`LoadShape`], so this introduces no vacuous edge. sc-18662
//!   is what would first be affected, and it is declared against the load shape rather than against
//!   this rung.
//!
//! # Rung 4 is implemented on both transformers (sc-18662)
//!
//! [`MemoryStrategy::BoundedTransformerResidency`] is `Implemented` on a
//! [`LoadShape::DeferredMaterialization`] load and `Missing` on a resident one — the rung's own
//! shared prerequisite is that shape, so a resident load genuinely does not have it. Both loaders
//! now exist ([`crate::dit::MiniMaxH3Dit::load_dir_deferred`],
//! [`crate::text_encoder::MiniMaxH3TextEncoder::from_dir_deferred`]), so the contract honours the
//! spec's shape rather than pinning one; see `resolved_load_shape` (private to this module).
//!
//! **Measured on both arms, separately, because `TransformerComponent::Both` is two claims**
//! (`tests/block_window_real.rs`):
//!
//! | arm | tier | resident | window 1 | ratio |
//! | --- | --- | ---: | ---: | ---: |
//! | text encoder | bf16 (dense) | 53.07 GB | 4.50 GB | **11.79x** |
//! | text encoder | q8 | 26.98 GB | 1.43 GB | **18.88x** |
//! | text encoder | q4 | 14.43 GB | 0.80 GB | **18.13x** |
//! | DiT | bf16 | 39.58 GB | 1.90 GB | **20.82x** |
//! | DiT | q8 | 21.64 GB | 1.58 GB | **13.73x** |
//! | DiT | q4 | 12.00 GB | 1.36 GB | **8.85x** |
//!
//! Output is **bit-identical** (`0.000e0` relative max-abs) at every window on every arm and tier,
//! so nothing is traded. Both resident figures reproduce this module's own stage constants from a
//! different harness — [`CONDITIONING_STAGE_PEAK_BYTES`] and [`DENOISE_RESIDENT_BF16_BYTES`] —
//! which is what makes them attributable to a component rather than to a process-wide mark.
//!
//! **What "bit-identical" now rests on, and what it used to rest on (sc-17153).** The claim was
//! previously made from the dense arm alone, which is the one encoder no tiered render loads; the
//! packed tiers' windowed output was compared against nothing. Both packed arms are now measured
//! and both are bit-identical, so the claim is true of the encoders that ship — but it is a
//! statement about *measured runs*, not a property the path guarantees: see
//! [`TRANSFORMER_WINDOW_SIZE`]'s fault note for an observed all-zero windowed context that no
//! memory assertion in this rung can see.
//!
//! Three things that declaration decides deliberately:
//!
//! * **[`TransformerComponent::Both`], not `Dit`.** The conditioning stage is still the taller of
//!   the two at every tier even after sc-19120's packed encoder, and declaring the DiT alone would
//!   leave it with no lever at all. The TE arm is the larger absolute saving — 48.57 GB against the
//!   DiT's 37.68 GB at bf16.
//! * **A singleton window domain.** [`TRANSFORMER_WINDOW_SIZE`] is `1` because 1 is the *measured
//!   residency floor* on every tier, not because the parameter is inert above it (it is not — the
//!   packed tiers spread 2.17-3.12x) and not because 1 is a safe default. See that constant.
//! * **Rung 4 does not depend on rung 3.** Its sole shared prerequisite is the load shape
//!   (`BOUNDED_TRANSFORMER_RESIDENCY_REQUIRES`), not rung 1 and not rung 3 — so sc-18661's
//!   `StructurallyNotApplicable` verdict, which satisfies prerequisite edges vacuously, introduces
//!   no vacuous edge here.
//!
//! # The AdaLN eviction is declared, and it is declared NET (sc-18665)
//!
//! The formula is [`MemoryFormulaKind::ComponentPhaseEnvelope`], carrying one
//! [`MemoryComponentKind::TransformerSubStack`] component: the 50-block `adaln_proj` stack, at
//! [`ADALN_EVICTED_BYTES`] resident and [`ADALN_MODULATION_TABLE_MAX_BYTES`] retained, with
//! [`MemoryComponentResidency::PrecomputedThenEvicted`] naming [`MemoryPhase::Denoise`] as the
//! phase whose steady state runs without it.
//!
//! Three things that shape decides deliberately:
//!
//! * **`TransformerSubStack`, not `Transformer`.** The projections are *inside*
//!   `asset_facts.transformer_bytes`; declaring them as a whole transformer would charge 26 GB
//!   twice. They are not auxiliary either, so they contribute nothing to `overlay_bytes` — which
//!   carries the configured LoRA stack and only that (sc-18650), and is 0 on a render with no
//!   adapter.
//! * **`residency`, not `default_engagement_exclusions`.** That list excludes a *rung* from
//!   cost-order engagement; it cannot remove bytes from a formula, and there is no rung here to
//!   exclude. `bounded_by` is `None` for the same reason — the drop is unconditional on the shipped
//!   path.
//! * **Net, not gross.** The precompute keeps a modulation table in the projections' place, so the
//!   declared exclusion is `ADALN_EVICTED_BYTES − ADALN_MODULATION_TABLE_MAX_BYTES`. Declaring the
//!   flat 26.02 GB claims a saving the runtime does not deliver, and the declared-versus-measured
//!   guard goes red on it.
//!
//! # Source-pruned centered-curve AdaLN (sc-26684)
//!
//! `SourcePruneSizing` prices the complete supplied table/reduced projections/folded biases
//! from canonical headers, preserving source storage widths but charging their once-widened F32
//! compute set. The full admitted schedule/shift domain bounds F32 block/final outputs and
//! interpolation coordinates. Those retained buffers coexist with the reduced operands: unlike
//! the dense path, nothing is evicted. They are charged once inside the transformer high-water
//! total, with a `WholeRender` sub-stack and zero eviction credit. Each render constructs only
//! its selected task DiT; a mixed dense/pruned route compares both corrected task footprints and
//! conservatively retains the dense companion's full floor. Pure dense/stock sizing is unchanged.

use mlx_gen::gen_core::{
    safetensors_path_bytes, standard_memory_behavior_context, AdapterResidencyMode,
    Error as CoreError, LoadShape, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryCalibrationIdentity, MemoryComponentKind,
    MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable, MemoryGeometry,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability, MemoryStrategySupport,
    ResidentRequestMemory, TransformerComponent, WeightsSource,
};

mod adapters;
mod conditioning;

use crate::denoise::LEGAL_FRAME_COUNTS;
use crate::model::{DIT_COMPONENT, MODEL_ID, TEXT_ENCODER_COMPONENT};
use crate::pipeline::{CANVAS_MAX_PIXELS, SPATIAL_STRIDE};
use mlx_gen::gen_core::pruned_adaln::{PrunedAdaLn, SourcePruneSizing, TABLE_KEY};

/// Calibration identity for this provider's measurements.
///
/// Bump the `vN` token whenever tensor layout, the quantization floor, or the execution structure
/// changes — every one of those invalidates the measured stage peaks above.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "minimax-h3-mlx-staged-joint-av-eager-abi3-v1";

/// The tier [`MEMORY_CALIBRATION_FINGERPRINT`] was measured on, and whose key it therefore keeps.
pub const CALIBRATED_TIER: &str = "bf16";

/// Calibration identity of the weights-free registry conformance walk.
///
/// **sc-22737.** [`weights_free_contract`] used to publish [`MEMORY_CALIBRATION_FINGERPRINT`],
/// because it shares `build_contract` with the production path. That is a false green in the worst
/// direction: a contract that read no weights at all reported the identity of a MEASURED production
/// cell, so an evidence runner could satisfy a plan row naming that cell without ever loading the
/// artifact. The declaration now carries its own key, and
/// `the_weights_free_declaration_cannot_publish_a_production_string` holds the two apart.
pub const STATIC_CALIBRATION_FINGERPRINT: &str = "minimax-h3-mlx-registry-behavior-v1";

/// Artifact-tier label of a MiniMax-H3 load: `bf16` dense, `q4`/`q8` packed.
pub fn calibration_tier_label(quant: Option<mlx_gen::Quant>) -> Option<&'static str> {
    match quant {
        None => Some("bf16"),
        Some(mlx_gen::Quant::Q4) => Some("q4"),
        Some(mlx_gen::Quant::Q8) => Some("q8"),
        Some(_) => None,
    }
}

/// The DiT directory a load resolves — the staged override when one is configured, the snapshot's
/// own `transformer` otherwise. Shared with [`ComponentBytes::resolve`] so the tier is read off the
/// same directory whose bytes are counted.
fn dit_dir(spec: &LoadSpec) -> std::path::PathBuf {
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root.clone(),
        WeightsSource::File(path) => path.parent().unwrap_or(path).to_path_buf(),
    };
    match spec.components.get(DIT_COMPONENT) {
        Some(WeightsSource::Dir(staged)) => staged.clone(),
        _ => root.join(DIT_COMPONENT),
    }
}

/// The tier the ARTIFACT carries, read from the DiT's own packed marker.
///
/// `Err` for an unreadable marker — `packed_quant_bits_at` answers `Ok(None)` for a MISSING
/// `config.json`, which would make an absent snapshot look like a dense tier and publish the bf16
/// identity for weights nobody can see. Fail closed instead.
pub fn resolved_artifact_tier(
    spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<Option<mlx_gen::Quant>> {
    let dir = dit_dir(spec);
    if !dir.join("config.json").is_file() {
        return Err(CoreError::Unsupported(format!(
            "{MODEL_ID} artifact tier: {} has no readable transformer config.json marker",
            dir.display()
        )));
    }
    Ok(
        match mlx_gen::quant::packed_quant_bits_at(&dir)
            .map_err(|error| CoreError::Unsupported(format!("{MODEL_ID} artifact tier: {error}")))?
        {
            None => None,
            Some(4) => Some(mlx_gen::Quant::Q4),
            Some(8) => Some(mlx_gen::Quant::Q8),
            Some(bits) => {
                return Err(CoreError::Unsupported(format!(
                "{MODEL_ID} artifact tier: {} declares an unshipped packed width of {bits} bits",
                dir.display()
            )))
            }
        },
    )
}

/// The production calibration identity for one artifact-proven tier.
///
/// ## sc-22737 (epic sc-22723 E1/E4): the key is per TIER
///
/// It used to be one string for every load, so a q4 render and a bf16 render published the same
/// identity and an anchor measured on one tier would be matched against loads of the other two.
///
/// The tier is proven, not requested: the DiT's own packed marker decides, and a `spec.quantize`
/// that disagrees with it publishes `None` — a dense snapshot asked for as q4 is a load-time
/// requantization whose peak no anchor measured. Withheld, never a failed load; `contract_for`
/// still returns a contract, just an uncalibrated one.
pub fn production_calibration_fingerprint(spec: &LoadSpec) -> Option<String> {
    let resolved = resolved_artifact_tier(spec).ok()?;
    if spec.quantize.is_some() && spec.quantize != resolved {
        return None;
    }
    let tier = calibration_tier_label(resolved)?;
    Some(MEMORY_PROFILE.fingerprint?.render("", tier))
}

/// The DiT block count, mirrored from `MiniMaxH3DitConfig::default().num_layers`.
pub const DIT_BLOCKS: u32 = 50;

// --- measured asset facts -------------------------------------------------------------------
//
// Exact `.safetensors` bytes under each component directory of the upstream bf16 snapshot
// (`MiniMaxAI/MiniMax-H3` @ `939557dc`), which is the same accounting
// `gen_core::safetensors_path_bytes` performs at load. These are on-disk footprints, NOT resident
// peaks — see the module docs for the stage-attributed peaks.

/// Qwen3-VL-32B text encoder — 14 shards, the conditioning component. 66.71 GB.
pub const TEXT_ENCODER_BYTES: u64 = 66_714_912_872;

/// One 33 B DiT partition at bf16 — 14 shards. 66.28 GB.
///
/// `transformer` and `transformer_ref` are byte-identical in size and a render loads **exactly
/// one**, so this is charged once (`crate::model::MiniMaxH3Task` resolves which).
pub const DIT_BF16_BYTES: u64 = 66_280_504_216;

/// Video VAE — 3 shards; the decoder is a 36-layer transformer, not a conv stack. 10.42 GB.
pub const VIDEO_VAE_BYTES: u64 = 10_415_558_888;

/// Audio VAE — 1 shard, DAC-lineage encoder plus BigVGAN decoder. 0.61 GB.
pub const AUDIO_VAE_BYTES: u64 = 605_429_340;

// --- measured stage peaks -------------------------------------------------------------------
//
// MLX active bytes, decimal GB, measured to 0.01 GB. These are NOT contract fields: the contract
// has no place for a measured stage peak (that is calibration evidence, sc-17153). They live here
// so the attribution is greppable, testable and cannot silently drift.

/// **Conditioning stage.** The dense Qwen3-VL-32B text encoder measured in isolation: 53.07 GB.
/// This, not the DiT, is the ~53 GB floor.
pub const CONDITIONING_STAGE_PEAK_BYTES: u64 = 53_070_000_000;

/// **Conditioning stage, packed `q4` tier** — 14.43 GB (sc-19120), against the 53.07 GB dense
/// datum. A **38.64 GB** reduction, and the single largest memory change in this model.
///
/// Measured by `tests/te_tier_real_weights.rs` on the stage in its own process, with the encoder
/// forced through a forward first: MLX mmaps lazily, and a bare 66 GB load leaves the peak at
/// ~33 KB. The same run's `get_active_memory` is 14.19 GB against the encoder's own
/// `nbytes()` accounting of 14.15 GB, so essentially nothing dense survives the tiering.
///
/// End to end at `384x224 / 124 frames`, per stage (`tests/te_tier_generate_stages.rs`):
///
/// | stage | dense TE | q8 TE | q4 TE |
/// |---|---:|---:|---:|
/// | conditioning | **52.80** | **26.91** | **14.68** |
/// | DiT load + AdaLN precompute | 12.66 | 12.66 | 12.66 |
/// | denoise (q4 DiT) | 12.75 | 12.75 | 12.75 |
/// | decode | 5.77 | 5.77 | 5.77 |
/// | **process** | **52.80** | **26.91** | **14.68** |
///
/// Two things that table settles. The DiT column is finally *legible* — every later stage was
/// sitting under a 40 GB shadow before. And the conditioning stage is still the tallest one even
/// packed, but by 1.93 GB rather than 40.05 GB, so the model's floor is now the DiT's residency
/// rather than its conditioner's.
///
/// **This is not the 12 GB ComfyUI footprint and does not claim to be.** The whole DiT is still
/// held resident through the denoise; bounding *that* is sc-18662's block streaming, and the two
/// stories are a pair — neither reaches consumer hardware alone.
pub const CONDITIONING_STAGE_PEAK_Q4_BYTES: u64 = 14_430_000_000;

/// **Conditioning stage, packed `q8` tier** — 26.94 GB in isolation (26.91 GB end to end).
///
/// Sits between the two: half the dense stage, and still above the q4 DiT's 12.75 GB denoise, so
/// q8 leaves the conditioner as the binding stage by a wide margin where q4 does not.
pub const CONDITIONING_STAGE_PEAK_Q8_BYTES: u64 = 26_940_000_000;

/// **Denoise stage.** DiT weights resident through the denoise loop at bf16, after the AdaLN
/// precompute-and-evict: 40.43 GB — 12.64 GB *below* the conditioning stage's mark.
pub const DENOISE_RESIDENT_BF16_BYTES: u64 = 40_430_000_000;

/// **Denoise stage.** The same residency at q4: 11.63 GB. The 28.80 GB delta against bf16 is the
/// measured tiering win, and it is real — the peak looked flat only because the conditioning stage
/// set the high-water mark first.
pub const DENOISE_RESIDENT_Q4_BYTES: u64 = 11_630_000_000;

/// **Denoise stage.** Exact bytes the AdaLN precompute-and-evict drops (64.56 → 38.70 GB active):
/// the 50-block stack's `adaln_proj` at bf16. Asserted against the loader in
/// `crate::dit::adaln`. Both actives are measured in `tests/adaln_evict_real_weights.rs`, which
/// force-materializes the whole stack first; the render path does not, so neither is a render-time
/// resident.
pub const ADALN_EVICTED_BYTES: u64 = 26_020_915_200;

/// **The same 50 projections on the packed `q8` tier — 13.83 GB.**
///
/// [`ADALN_EVICTED_BYTES`] is a **bf16** figure and the lever shrinks with the tier exactly as the
/// DiT that contains it does (`crate::dit::block::AdaLnProjection::nbytes`). Derived rather than
/// measured: the private `adaln_stack_bytes` re-derives it from the shipped configuration and
/// `the_packed_tier_stack_sizes_are_the_loaders_own_accounting` holds the two together.
///
/// The 13.02 GB `crate::quant` quotes for this tier is the **code buffer only**; this figure also
/// carries the per-group bf16 scale and bias the packed triple cannot run without, which is what
/// `crate::quant::nbytes` sums and therefore what is actually resident.
pub const ADALN_EVICTED_Q8_BYTES: u64 = 13_828_147_200;

/// **The same 50 projections on the packed `q4` tier — 7.33 GB.**
///
/// The figure `tests/common/mod.rs`'s (g) section quotes as "≈7.3 GB (≈6.5 GB if the group metadata
/// is not counted)". See [`ADALN_EVICTED_Q8_BYTES`] for why the group metadata is counted here.
pub const ADALN_EVICTED_Q4_BYTES: u64 = 7_325_337_600;

/// **Denoise stage.** Bytes of the modulation table the precompute keeps in the projections' place,
/// at the **longest schedule this model admits** — `MODULATION_PARAMS · modulation_rows ·
/// hidden_size · DIT_BLOCKS` elements at bf16, with `modulation_rows` read off a real
/// [`crate::model::MAX_STEPS`]-evaluation schedule rather than derived by hand
/// (`the_retained_table_is_the_worst_case_over_the_admitted_schedule`).
///
/// **The evict is not free, and this is the price.** [`ADALN_EVICTED_BYTES`] is what the projections
/// cost; this is what replaces them. The contract declares the *net* difference, because declaring
/// the gross figure claims a saving the runtime does not deliver — the exact overstatement
/// `the_declared_saving_does_not_exceed_the_measured_residency_drop` refuses.
///
/// **Why the worst case rather than the default schedule.** The table is linear in the schedule's
/// distinct-timestep count and independent of resolution and duration (nothing in it has a token
/// axis — `AdaLnCache::bytes`). The contract is static and has no request to read a step count
/// from, so it must pick one point of that line. Under-declaring what the precompute keeps
/// over-declares the saving, and an over-declared saving turns a suppressed configuration into an
/// OOM; over-declaring it only leaves some of the win on the table. The asymmetry picks the end.
///
/// **This figure does NOT scale with the tier, and that asymmetry is the whole reason the two
/// quantities are declared separately.** The table is the projections' *output*, so its dtype is the
/// **compute** dtype — `crate::quant::compute_dtype`, which reads bf16 off a packed tier's own
/// scales — not the tier's bit width. `tests/common/mod.rs`'s (f) section states the same fact from
/// the measurement side ("tier-free … the retained table, whose dtype is the *compute* dtype …
/// rather than the tier's bit width"). So the resident side falls 26.02 → 13.83 → 7.33 GB across
/// bf16/q8/q4 while this stays at 3.87 GB, and applying one factor to both would be wrong at every
/// tier but bf16.
pub const ADALN_MODULATION_TABLE_MAX_BYTES: u64 = 3_870_720_000;

/// Contract-stable identity of the evictable AdaLN sub-stack, so a consumer can find the
/// declaration by name rather than by matching on bytes.
pub const ADALN_COMPONENT_ID: &str = "dit_adaln_proj_stack";

// --- rung 2: the bounded-decode domain ---------------------------------------------------------

/// The **only** decode tile edge this provider admits, in output pixels.
///
/// Derived from [`crate::spatial_tiling::TILE_SAMPLE_MIN_SIZE`] rather than re-typed, so a change
/// to the reference geometry cannot leave the contract publishing a domain the decode no longer
/// executes. It is a singleton because the tile edge is an **output-correctness** input copied from
/// the published model (sc-18786), not a tunable — see the module docs.
pub const DECODE_TILE_EDGE: u32 = crate::spatial_tiling::TILE_SAMPLE_MIN_SIZE as u32;

/// The **only** decode tile overlap this provider admits, in output pixels. Derived from
/// [`crate::spatial_tiling::TILE_SAMPLE_MIN_OVERLAP`] for the same reason as [`DECODE_TILE_EDGE`].
///
/// A zero overlap is the tile-starvation failure mode this rung must not be able to select: it
/// abuts the tiles with no cross-fade, and the corruption shows up **across frames** rather than
/// within one, so no memory number can see it. `validate_ranges` independently forbids a zero
/// entry in a published range, so the domain cannot be widened to include it by accident.
pub const DECODE_OVERLAP: u32 = crate::spatial_tiling::TILE_SAMPLE_MIN_OVERLAP as u32;

// --- rung 3: the measured inapplicability -------------------------------------------------------

/// **The rung-3 verdict's evidence, as data rather than prose** (sc-18661).
///
/// Denoise-phase MLX peak under each bounded attention configuration, as a fraction of the same
/// geometry's unbounded peak, measured on the real 50-block `q4` DiT by
/// `tests/bounded_attention_real.rs`. Positive means the rung **cost** peak.
///
/// Rows are `(tier, frames, packed_seq_len, head-axis @ 64 Mi, query-row @ 64 Mi, head-axis
/// one-head)` — **tier-keyed since the sweep was extended to all three** (the q4-only table it
/// started as made the bf16/q8 verdict an argument rather than a measurement).
/// Both ends of the `17n + 5` frame lattice are present because `frames` is an exact-match evidence
/// key (`mlx_fit_gate.rs:1381`) and fitted-curve extrapolation scales by area only — a single
/// duration cell may not stand in for another one.
///
/// The third column is the configuration whose budget is derived from the geometry (`B · Sq²`, one
/// head per chunk with the query axis whole). It is measured because the shipped 64 Mi operating
/// point is **Z-Image's**, and at 56 heads it stops being bit-exact above `Sq = 8192` — inside this
/// family's legal duration range. Without that column the verdict would be about a budget rather
/// than about the rung.
/// Measured 2026-08-15 at 576x320 on the `SceneWorks/minimax-h3-mlx` `q4` transformer
/// (`f22bc294`), two evaluations per arm, `tests/bounded_attention_real.rs`:
///
/// | frames | seq | unbounded | heads @ 64 Mi | query rows @ 64 Mi | heads @ one-head |
/// |---:|---:|---:|---:|---:|---:|
/// | 124 | 7 586 | 13.4894 GB | 13.5614 GB (+0.53 %) | 13.5640 GB (+0.55 %) | 13.5666 GB (+0.57 %) |
/// | 345 | 20 022 | 16.4951 GB | 16.7053 GB (+1.27 %) | 16.7123 GB (+1.32 %) | 16.7194 GB (+1.36 %) |
///
/// **The cost grows with the sequence**, which is the direction that settles the verdict: the axis
/// the rung exists to relieve is the one on which it gets worse. Wall-clock moves the same way at the
/// shipped budget — +1.6 %/+15.6 % at 124 frames, +44.3 %/+71.7 % at 345.
///
/// Output agreement was **exactly zero** relative max-abs on all six bounded arms, so nothing here is
/// a quality trade that memory could be bought with.
pub const RUNG3_MEASURED_PEAK_DELTAS: [(&str, i32, i32, f64, f64, f64); 4] = [
    ("q4", 124, 7_586, 0.005341, 0.005532, 0.005723),
    ("q4", 345, 20_022, 0.012740, 0.013168, 0.013596),
    ("q8", 124, 7_586, 0.003065, 0.003174, 0.003284),
    ("q8", 345, 20_022, 0.007931, 0.008197, 0.008464),
];

/// **`bf16` is measured but UNRESOLVED, and is deliberately absent from
/// [`RUNG3_MEASURED_PEAK_DELTAS`] rather than recorded with a number.**
///
/// The q4 and q8 sweeps agree with each other and with the verdict: every bounded arm costs peak,
/// the cost grows with sequence, and the magnitude shrinks as the tier's resident baseline grows
/// (+0.53%/+1.27% at q4 against +0.31%/+0.79% at q8). Extrapolating that trend to bf16 predicts a
/// small positive delta.
///
/// **It measured NEGATIVE.** At 124 frames the unbounded arm peaked at 42.3451 GB and all three
/// bounded arms at 42.145-42.151 GB — a **-0.47%** delta, i.e. chunking appearing to *save* peak.
/// That is the sign the verdict says cannot happen, so it is recorded here rather than declared
/// either way.
///
/// Two readings, and this constant exists because they have not been separated:
///
/// * **an ordering artifact.** The three bounded arms agree with each other to 0.01% while the
///   unbounded arm — which runs FIRST in the process — sits 0.47% above all of them. That is the
///   shape of a first-arm allocator transient, and it is a known hazard on this exact comparison:
///   `tests/sequence_cost_real.rs` documents the same harness reporting **+0.2% first-in-binary and
///   +50.3% under a filter** for one pair of arms, and adds a warm-up arm for it. This harness
///   reloads the 62 GB DiT per arm and clears the cache between them, which is stronger — but it has
///   no unmeasured warm-up arm, and at bf16 the load is 3.7x the q4 one it was validated at.
/// * **a real bf16 effect.** The dtype is the same on every tier's activations, but the *weights*
///   are not, and the graph a dense DiT builds differs from a packed one's.
///
/// Resolving it needs a warm-up arm plus an arm-order permutation on this tier; until then the
/// contract's inapplicability reason quotes q4 and q8 only, and `rung3_cells_for_tier("bf16")`
/// returns empty so the real-weight harness fails loudly rather than validating against a guess.
/// **The claim in PR #649 that q4 bounds the other tiers from above is contradicted at bf16/124f
/// and must not be relied on.**
pub const RUNG3_BF16_IS_UNRESOLVED: &str =
    "bf16 measured -0.47% at 124 frames (42.3451 GB unbounded against 42.145-42.151 GB bounded);      unseparated first-arm allocator transient vs a real dense-tier effect";

/// The rung-3 cells measured for one tier.
pub fn rung3_cells_for_tier(tier: &str) -> Vec<(i32, i32, f64, f64, f64)> {
    RUNG3_MEASURED_PEAK_DELTAS
        .iter()
        .filter(|(t, ..)| *t == tier)
        .map(|(_, frames, seq, a, b, c)| (*frames, *seq, *a, *b, *c))
        .collect()
}

/// The audio VAE is out of scope for rung 2, declared rather than implied.
///
/// It is 0.61 GB against the video VAE's 10.42 GB (17× smaller), and structurally has nothing to
/// tile: `crate::audio_vae` is a DAC-lineage 1-D stack whose decode is a BigVGAN vocoder over a
/// waveform. There is no spatial grid, so the `decode_tile_edge` / `decode_overlap` domain has no
/// meaning for it and the rung's mechanism does not run on that path. Named as a constant so the
/// boundary is greppable and testable rather than a sentence in a doc comment.
pub const AUDIO_VAE_IS_OUT_OF_SCOPE_FOR_TILING: bool = true;

/// The load shape a **resident** load has, kept as the provider's default and its calibration
/// identity's shape.
///
/// Until sc-18662 this was the only shape this provider had, because `MiniMaxH3Dit::load_dir` builds
/// the whole 50-block stack no matter what a caller asks for. It is no longer: `load_dir_deferred`
/// and `MiniMaxH3TextEncoder::from_dir_deferred` implement
/// [`LoadShape::DeferredMaterialization`] for both transformers, so the contract now **honours the
/// spec's shape** through `resolved_load_shape` instead of pinning one.
pub const LOAD_SHAPE: LoadShape = LoadShape::EagerMaterialization;

/// The load shape this contract is resolved at — the spec's, because both are now implemented.
///
/// The distinction from [`LOAD_SHAPE`] is the whole of sc-18662: a provider may only mirror a
/// requested shape once it actually has a loader for it, and answering
/// [`LoadShape::DeferredMaterialization`] without one is how a rung comes to be declared over
/// nothing. Both loaders exist and `tests/block_window_real.rs` measures each.
fn resolved_load_shape(spec: &LoadSpec) -> LoadShape {
    spec.load_shape
}

// --- rung 4: the bounded transformer-residency domain -------------------------------------------

/// The **only** transformer window size this provider admits: one block or layer at a time.
///
/// **The singleton is the finding, not a placeholder** — the same shape rung 2's
/// [`DECODE_TILE_EDGE`] has, for its own separate reason. What makes it the right point is that
/// **1 is the measured residency floor**, not that the parameter is inert.
///
/// # The inertness claim was wrong, and it was wrong because it was measured on one tier (sc-17153)
///
/// This constant used to say the window parameter is *measured inert* above 1, citing a **1.09x**
/// spread across `[1, 5, 10, 50]`. That spread was taken on the **dense** encoder, where a single
/// Qwen3 layer's f32 upcast dominates the peak at every window size and flattens the curve
/// (re-measured: 1.39x). It does not describe the encoder a shipped tiered render loads. Measured
/// on the packed tiers, per window, on real weights:
///
/// | tier | resident | window 1 | window 5 | window 10 | window 50 | spread |
/// | --- | ---: | ---: | ---: | ---: | ---: | ---: |
/// | `bf16` (dense) | 53.07 GB | **4.50 GB** | 6.24 GB | 6.24 GB | 6.24 GB | 1.39x |
/// | `q8` | 26.98 GB | **1.43 GB** | 2.65 GB | 2.78 GB | 3.10 GB | 2.17x |
/// | `q4` | 14.43 GB | **0.80 GB** | 1.78 GB | 1.95 GB | 1.87 GB | 2.45x |
///
/// The packed spread runs 2.17-3.12x across runs (sc-17153's terminal campaign independently
/// measured 3.12x on `q4`), so the parameter is a **real lever** on every encoder a tiered render
/// uses, and the retired `< 2.0` spread assertion in `tests/block_window_real.rs` was asserting
/// something false about them.
///
/// What holds on **every encoder tier measured**, with a wide margin, is that window 1 is the
/// minimum: 4.50 GB against 6.24 (dense), 1.43 against 2.65-3.10 (`q8`), 0.80 against 1.78-1.95
/// (`q4`). The curve above 1 is not strictly monotonic — `q4` runs 1.78 -> 1.95 -> 1.87 across
/// windows 5/10/50, so the ordering *among the larger windows* is counter noise — but the floor is
/// separated from all of them by 1.9x or more. That is what justifies publishing the singleton:
/// `[1]` is the cheapest point of a real curve, and the other values would advertise a lever that
/// trades memory *away*.
///
/// `tests/block_window_real.rs::measure_te_arm` asserts the floor (with the harness's 256 MiB / 5 %
/// margin, not as a strict ordering) and bounds the spread, on **both TE arms**. The DiT arm is
/// deliberately not covered by that claim: it measures resident against window 1 only, so whether
/// 1 is *its* floor is unmeasured — see [`RUNG4_DIT_MEASURED_PEAKS`].
///
/// # ⚠ The DENSE encoder has been observed returning an all-zero context — UNFIXED
///
/// Five sightings in ~32 real-weight forwards on this hardware (sc-17153): identical shapes, clean
/// exit, no Metal diagnostic, `max = 0.0` where the correct answer is `1.55e4`. What the sightings
/// have in common is now specific enough to be actionable:
///
/// * **Every sighting is the `bf16` dense encoder.** Zero sightings across ~14 `q4` / `q8` arms.
/// * **It is NOT windowing.** Three sightings were *resident* forwards, two windowed — including
///   two resident ones after this module's loader fix, on a path that never goes near
///   `from_dir_deferred`. Rung 4 neither causes it nor is exposed to it more than the resident
///   path is, and the loader fix neither introduced nor removed it.
/// * **It is NOT the window size.** Reversing the order of `[1, 5, 10, 50]` moved the zero from
///   window 1 to window 50.
/// * **It IS the second multi-gigabyte encoder load in one process, reading cold.** Four of the
///   five fit exactly that: two dense resident loads that followed the packed arm in the same test
///   binary (167.2 s and 166.9 s cold — both zero), against two that were the *first* load in
///   their process (156.0 s and 72.6 s cold — both correct), and twelve warm loads at any position
///   (all correct). The two windowed sightings are the same shape: a window that followed a 53 GB
///   resident load in the same process. The one outlier was a 2-layer probe taken while ~20
///   compilers were running.
///
/// The mechanism is below this crate and is **unattributed**; nothing here claims to have removed
/// it. Two things this crate does about it:
///
/// 1. `tests/block_window_real.rs` compares the context against a reference on every arm and
///    refuses a degenerate reference — that assertion caught sightings four and five live. The peak
///    counter reads *lower* when the work does not happen, so no memory assertion in this rung can
///    see it, and neither can a norm, a cosine or a checksum.
/// 2. That file now **requires** each arm to be its own process invocation, which is what its own
///    module text had always claimed and what `--test-threads=1` does not provide.
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;

/// **The TE arm** (sc-18662, AC3), measured on the real Qwen3-VL-32B tap by
/// `tests/block_window_real.rs`: resident bytes, and the window-1 floor.
///
/// 53.07 GB -> 4.50 GB, an **11.79x** reduction and the single largest memory change this rung
/// produces on this model. The resident figure independently reproduces
/// [`CONDITIONING_STAGE_PEAK_BYTES`]'s 53.07 GB from a different harness, which is what makes it
/// attributable to the encoder rather than to a process-wide mark.
///
/// Output was **bit-identical** (`0.000e0` relative max-abs) at every window size, so nothing is
/// traded for it — but read the fault note on [`TRANSFORMER_WINDOW_SIZE`] before treating
/// "bit-identical" as a property of the path rather than of the runs that were measured.
///
/// # The windowed figure moved with the mlx-rs pin (sc-17153): 5.81 GB -> 4.50 GB
///
/// `RUNG4_TE_WINDOWED_PEAK_BYTES` was declared at `5_814_845_376`. It does not reproduce: eight
/// runs on real weights land at `4_501_440_448` B ± ~0.5 MB, 1.31 GB (22.6 %) below the declared
/// value and far outside `common::assert_declared_peak_constant`'s bound — which is the
/// per-constant absolute bound sc-17153 introduced precisely so a stale member of a pair could not
/// hide inside a quotient. The resident member is unchanged and reproduces to 2.4 MB, so this was
/// exactly the *single-member* drift a ratio band cancels.
///
/// **The cause is the MLX pin, not this crate.** `git diff` from the sc-18662 measurement
/// (`19efeb265`) to the current tree over `src/text_encoder/`, `src/block_stream.rs` and the
/// harness is doc- and comment-only — nothing in the crate accounts for a 22.6 % drop. What moved
/// is `Cargo.toml`'s mlx-rs revision, `932beb4e6` -> **`7151a9b27`**, in `dfd76b5ae`
/// ("fix(mlx): preserve global VAE normalization in tiled decode", sc-19753). So:
///
/// * this figure is **pinned to mlx-rs `7151a9b27`**, and an MLX pin bump is the event that should
///   be expected to move it again — when it does, the assertion will red with a *stale constant*
///   message, and the pin is the first thing to check rather than this crate's loaders;
/// * it also independently rules out the alternative reading, that 4.50 GB is what a *degenerate*
///   (all-zero) run happens to peak at. It is not: the zero-output runs peak lower still, and
///   `tests/block_window_real.rs` compares output before it records a cell, so no run that
///   produced zeros can contribute a peak to this constant.
pub const RUNG4_TE_RESIDENT_PEAK_BYTES: u64 = 53_074_721_216;
/// The TE arm at [`TRANSFORMER_WINDOW_SIZE`]. See [`RUNG4_TE_RESIDENT_PEAK_BYTES`].
pub const RUNG4_TE_WINDOWED_PEAK_BYTES: u64 = 4_501_440_448;

/// **The DiT arm**, per tier: `(tier, resident_peak_bytes, window_1_peak_bytes)`.
///
/// Measured by `tests/block_window_real.rs` over the 50-block stack at `seq = 4096`, walking block
/// bodies with no modulation and no rotary — the thinnest call that touches every one of a block's
/// ten body tensors, so the ratio is weight residency rather than activation.
///
/// | tier | resident | window 1 | ratio |
/// |---|---:|---:|---:|
/// | `q4` | 12.00 GB | 1.36 GB | **8.85x** |
/// | `q8` | 21.64 GB | 1.58 GB | **13.73x** |
/// | `bf16` | 39.58 GB | 1.90 GB | **20.82x** |
///
/// The ratio scales with the tier — 8.85x / 13.73x / 20.82x across q4 / q8 / bf16 — because the
/// *window* is the same activation-plus-one-block at every tier while the resident baseline is not.
/// That is the rung behaving exactly as declared, and it is the **opposite** of rung 3's tier story
/// on this model, where the transient the rung acts on is bf16 regardless of tier so the lower tier
/// was the favourable one. The two rungs' tier arguments do not transfer to each other.
///
/// Both resident figures independently reproduce the provider's own stage constants from a different
/// harness — [`DENOISE_RESIDENT_Q4_BYTES`]'s 11.63 GB and [`DENOISE_RESIDENT_BF16_BYTES`]'s 40.43 GB
/// — which is what makes them attributable to the DiT rather than to a process-wide mark. Output was
/// bit-identical at every window on every tier.
pub const RUNG4_DIT_MEASURED_PEAKS: [(&str, u64, u64); 3] = [
    ("q4", 12_002_937_856, 1_355_656_192),
    ("q8", 21_636_046_848, 1_575_493_632),
    ("bf16", 39_578_684_416, 1_901_221_376),
];

/// **The rung-4 REQUEST peak** (sc-18662): the max over one whole streamed render's stage peaks,
/// measured on the shipped `generate` by `tests/streamed_generate_real.rs` — q4 DiT tier, q4
/// packed text encoder, 124 frames at 384x224, 4 steps, window 1, `TransformerComponent::Both`.
///
/// This is the number the per-arm cells above cannot stand in for: they measure each phase in its
/// own process window, while the SceneWorks survey's `requestPeak` axis asks about the composition
/// — and its own note says a static proxy will not do. The resident request's high-water is the
/// conditioning stage's mark ([`CONDITIONING_STAGE_PEAK_BYTES`], 53.07 GB dense); the streamed
/// request pins at the decode floor instead, because conditioning and denoise are both windowed
/// and the VAE decode is untouched by this rung (rung 2 owns it).
///
/// Measured per stage: conditioning 0.78 GB, dit-load + adaln-precompute 2.03 GB, denoise
/// 2.03 GB, decode **5.80 GB** — so the request is decode-bound, which is what the phase
/// envelope's `max(TE, DiT, decode)` arithmetic predicted (5.77 GB) before the request ran.
///
/// **Graded to ±290 MB, not to a ±25% band** (sc-17153). `tests/streamed_generate_real.rs` puts
/// this constant through `common::assert_declared_peak_constant`, whose tolerance is the larger of
/// 256 MiB and 5 % of the declared value — 5 % wins here, at 290_218_452 B. That grader replaced the
/// `|measured/declared − 1| < 0.25` ratio band, which was multiplicatively asymmetric and, on a
/// quotient of two constants, blind to pair-proportional drift.
///
/// The retired "~1 GB of run-to-run spread on this machine" figure was **retracted** with it: it was
/// not reproduced on either arm of the sc-17153 terminal campaign (the q4 DiT windowed peak came
/// back bit-identical across two runs, its resident peak moved 0.36 %). Do not size a new bound
/// from it. The grading is TIGHTER than the retired comment claimed, so a red here is a real drift
/// or a pin bump — `assert_declared_peak_constant`'s own docs carry the two limits of the bound.
///
/// Only graded when the run is q4 DiT **and** q4 packed TE, the conditions this figure was captured
/// under; the harness prints that it asserted nothing when either differs.
pub const RUNG4_REQUEST_PEAK_Q4_BYTES: u64 = 5_804_369_044;

/// The stage-ordered lifecycle every MiniMax-H3 render runs.
fn phases() -> Vec<MemoryPhase> {
    vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ]
}

/// On-disk bytes of the four components one render can touch.
///
/// Absent components contribute zero ([`safetensors_path_bytes`]), so a weights-free spec yields a
/// zero footprint rather than an error — the same behavior `PerComponentBytes` has. The DiT is
/// resolved the way `crate::model` resolves it, so a **tiered** install (a staged `transformer`
/// component) is charged at the staged tier's bytes rather than the flat snapshot's.
struct ComponentBytes {
    text_encoder: u64,
    dit: u64,
    /// The AdaLN sub-stack's residency **on the tier that was actually resolved** — see
    /// [`resolved_adaln_bytes`]. Carried alongside `dit` rather than recomputed in
    /// [`adaln_component`] so that the weights-free contract, which resolves nothing, states its
    /// architecture fact in one place instead of re-deriving it from a zero footprint.
    adaln: u64,
    /// Source-pruned operands/cache remain live; a mixed task route cannot claim dense eviction.
    pruning: Option<SourcePruneSizing>,
    video_vae: u64,
    audio_vae: u64,
    /// Load-exact bytes of the configured LoRA stack, held **resident for the whole render**
    /// (sc-18650).
    ///
    /// [`crate::adapters`]'s declared strategy is a *forward-time residual* over
    /// `mlx_gen::adapters::AdaptableLinear` — "never a merged weight", on **every** quant tier,
    /// because a packed leaf cannot absorb a bf16 delta without dequantizing and quant tier is a
    /// creative choice in this product. The factors are therefore still resident at the last denoise
    /// step, which makes [`AdapterResidencyMode::Additive`] the honest mode: `Folded` would declare
    /// zero and under-declare the floor, exactly the failure the text-encoder note below records.
    overlay: u64,
    native_quant: Option<mlx_gen::gen_core::h3_memory::NativeAllocationFacts>,
}

/// Bytes a `spec.precision`-following component occupies once loaded (SC-22667).
///
/// `LinearNoBias::from_weights` and the DiT block builder call `cast_weights(dtype)` on every
/// **dense** base, and `model.rs` selects that dtype as `Float32` under `Precision::Fp32` and
/// `Bfloat16` otherwise. A bf16-stored dense checkpoint loaded at Fp32 therefore materializes twice
/// its on-disk size, which the plain `safetensors_path_bytes` sum could not see. A packed q4/q8
/// tier is unaffected: `cast_weights` is deliberately not applied to a packed base, and the shared
/// projection detects an existing affine pack by its `.scales` companion and leaves it at stored
/// width.
fn materialized_component_bytes(path: &std::path::Path, precision: mlx_gen::Precision) -> u64 {
    if precision == mlx_gen::Precision::Fp32 {
        materialized_f32_bytes(path)
    } else {
        safetensors_path_bytes(path)
    }
}

/// Bytes a component that is materialized f32 regardless of `spec.precision` occupies.
///
/// Falls back to the on-disk sum when the source cannot be read as safetensors. This runs at
/// contract time, ahead of any component load, so an absent or not-yet-staged component must keep
/// its previous accounting rather than turn the contract into a refusal.
fn materialized_f32_bytes(path: &std::path::Path) -> u64 {
    let stored = safetensors_path_bytes(path);
    mlx_gen::asset_facts::projected_safetensors_bytes(path, |_| {
        mlx_gen::asset_facts::ResidentProjection::Float32
    })
    .unwrap_or(stored)
}

fn native_backend() -> mlx_gen::gen_core::h3_memory::NativeBackend {
    mlx_gen::gen_core::h3_memory::NativeBackend::Mlx {
        tile_columns: mlx_gen::w4a8_int8::TILE_COLS as u64,
    }
}

fn dit_allocation(
    path: &std::path::Path,
    precision: mlx_gen::Precision,
    pruned: bool,
) -> mlx_gen::gen_core::Result<mlx_gen::gen_core::h3_memory::NativeComponentAllocation> {
    let width = if precision == mlx_gen::Precision::Fp32 {
        4
    } else {
        2
    };
    let mut allocation =
        mlx_gen::gen_core::h3_memory::component_allocation(path, native_backend(), width, |h| {
            if pruned
                && (h.name == TABLE_KEY
                    || h.name.contains(".adaln_proj.linear.")
                    || h.name.starts_with("norm_out.linear."))
            {
                None
            } else if h.name.starts_with("transformer_blocks.")
                || h.name.starts_with("token_refiner.")
            {
                Some(width)
            } else {
                None
            }
        })?;
    // Unresolved inventory probes name no tensor the DiT consumes. Retain their established
    // file-size bound; real dense/native sources use the materialized header facts above.
    if allocation.projections.is_empty() && safetensors_path_bytes(path) > 0 {
        let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
        if !headers
            .iter()
            .any(|h| h.name.ends_with(".weight") || h.name == TABLE_KEY)
        {
            allocation.resident_bytes = materialized_component_bytes(path, precision);
        }
    }
    Ok(allocation)
}

/// Source headers and the loader's config, without mapping any tensor payload. An absent
/// component retains the existing unresolved declaration; unreadable materialized headers refuse.
fn source_prune_sizing(
    dir: &std::path::Path,
) -> mlx_gen::gen_core::Result<Option<SourcePruneSizing>> {
    if safetensors_path_bytes(dir) == 0 {
        return Ok(None);
    }
    let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(dir)?;
    if !headers.iter().any(|h| {
        h.name == TABLE_KEY
            || h.name.ends_with(".adaln_proj.linear.weight")
            || h.name == "norm_out.linear.weight"
    }) {
        return Ok(None);
    }
    let config_path = dir.join("config.json");
    let text = std::fs::read_to_string(&config_path)
        .map_err(|error| CoreError::Unsupported(format!("{}: {error}", config_path.display())))?;
    let config = crate::dit::MiniMaxH3DitConfig::from_diffusers_json(&text)
        .map_err(|error| CoreError::Unsupported(error.to_string()))?;
    let headers = headers
        .into_iter()
        .map(|h| (h.name, (h.dtype, h.shape)))
        .collect::<std::collections::BTreeMap<_, _>>();
    let Some(topology) = PrunedAdaLn::inspect(
        config.hidden_size as usize,
        config.num_layers as usize,
        config.time_embed_dim as usize,
        |name| {
            // Match native curve inspection: packed code columns are not reduced coordinates.
            if let Some(base) = name.strip_suffix(".weight") {
                if headers.contains_key(&format!("{base}.scales"))
                    || headers.contains_key(&format!("{base}.comfy_quant"))
                {
                    return None;
                }
            }
            headers
                .get(name)
                .filter(|(dtype, _)| {
                    matches!(
                        dtype,
                        mlx_gen::gen_core::weightsmeta::Dtype::F16
                            | mlx_gen::gen_core::weightsmeta::Dtype::BF16
                            | mlx_gen::gen_core::weightsmeta::Dtype::F32
                    )
                })
                .cloned()
        },
    )
    .map_err(CoreError::Unsupported)?
    else {
        return Ok(None);
    };
    let rows = SourcePruneSizing::conditioned_timestep_bound(crate::model::MAX_STEPS as usize)
        .map_err(CoreError::Unsupported)?;
    SourcePruneSizing::for_topology(&topology, rows)
        .map(Some)
        .map_err(CoreError::Unsupported)
}

impl ComponentBytes {
    fn resolve(spec: &LoadSpec) -> mlx_gen::gen_core::Result<Self> {
        let root = match &spec.weights {
            WeightsSource::Dir(root) => root.clone(),
            WeightsSource::File(path) => path.parent().unwrap_or(path).to_path_buf(),
        };
        let dit = match spec.components.get(DIT_COMPONENT) {
            Some(WeightsSource::Dir(staged)) => staged.clone(),
            _ => root.join(DIT_COMPONENT),
        };
        use mlx_gen::gen_core::h3_memory::{NativeAllocationFacts, NativeTaskAllocation};
        let reference = dit.with_file_name(crate::model::REFERENCE_DIT_PARTITION);
        let backend = native_backend();
        let mut tasks = Vec::new();
        let mut overlay = 0u64;
        let mut dit_bytes = 0;
        let mut pruning = None;
        let mut pruned_resident = 0;
        let mut max_adaln = 0;
        let mut resolved = false;
        // Each render loads ONE task. Correct source-pruned operands before taking the maximum.
        for directory in [&dit, &reference] {
            let sizing = source_prune_sizing(directory)?;
            let allocation = dit_allocation(directory, spec.precision, sizing.is_some())?;
            let bytes = if let Some(sizing) = sizing {
                let live = sizing.resident_bytes().map_err(CoreError::Unsupported)?;
                if live > pruned_resident {
                    pruning = Some(sizing);
                    pruned_resident = live;
                }
                // Preserve source-width pruning operands; S5 alone replaces this subset with
                // the actual F32 compute operands and retained schedule tables.
                sizing
                    .materialized_bytes(allocation.resident_bytes, false)
                    .map_err(CoreError::Unsupported)?
            } else {
                allocation.resident_bytes
            };
            max_adaln = max_adaln.max(allocation.adaln_bytes);
            dit_bytes = dit_bytes.max(bytes);
            resolved |= allocation.resolved;
            let patch_size = if !allocation.resolved {
                [1, 2, 2]
            } else {
                let text = std::fs::read_to_string(directory.join("config.json"))?;
                crate::dit::MiniMaxH3DitConfig::from_diffusers_json(&text)
                    .map_err(|e| CoreError::Unsupported(e.to_string()))?
                    .patch_size
                    .map(|n| n as u64)
            };
            let window = if allocation.resolved {
                Some(
                    mlx_gen::gen_core::h3_memory::window::WindowAllocation::from_resident_tensors(
                        &allocation.phase_resident_by_name,
                        &allocation.projections,
                        bytes.saturating_sub(allocation.resident_bytes),
                        sizing.is_some(),
                    )?
                    .with_source_casts(&allocation.source_cast_by_name)?,
                )
            } else {
                None
            };
            let adapter = if allocation.resolved && !spec.adapters.is_empty() {
                let profile = adapters::profile(
                    directory,
                    &spec.adapters,
                    &allocation.projections,
                    spec.precision,
                )?;
                overlay = overlay.max(profile.resident_bytes);
                Some(profile)
            } else {
                None
            };
            tasks.push(NativeTaskAllocation {
                adapter,
                resident_bytes: bytes,
                constructor_bytes: 0,
                patch_size,
                reference_task: directory == &reference,
                window,
                evicted_bytes: if sizing.is_some() {
                    0
                } else {
                    allocation.adaln_bytes
                },
                table_bytes_per_timestep: if sizing.is_some() {
                    0
                } else {
                    allocation.table_bytes_per_timestep
                },
                projections: allocation.projections,
            });
        }
        let has_native = resolved;
        let mut native_quant = has_native.then_some(NativeAllocationFacts {
            conditioning_stream: None,
            conditioning: None,
            conditioning_resident_bytes: 0,
            conditioning_load_bytes: 0,
            audio_decode_load_bytes: 0,
            video_decode_resident_bytes: 0,
            video_decode_load_bytes: 0,
            audio_decode_resident_bytes: 0,
            default_evaluations: crate::model::DEFAULT_STEPS as usize,
            backend,
            tasks,
        });
        if let Some(facts) = &mut native_quant {
            conditioning::complete(&root, spec.precision, facts)?;
        }
        // **The condition encoder honors its own staged override too** (sc-19120 / sc-20267). It did
        // NOT before, and the omission was `DIT_COMPONENT`'s own bug one component over: sc-19120
        // made the text encoder per-tier and staged **independently** of the DiT, so a split `q4`
        // install stages [`TEXT_ENCODER_COMPONENT`] outside the snapshot and may carry no
        // `root/text_encoder` at all. That measured 0 and published an 18.72 GB **under**-declaration
        // of the largest single component this family loads — and an under-declared floor admits a
        // render that then OOMs, which is the whole failure the ladder exists to prevent.
        //
        // `safetensors_path_bytes` returns 0 for an absent directory, so the flat-snapshot path is
        // byte-identical. Found while porting this contract to the candle lane (sc-20267), where the
        // same line had the same defect; fixed in both rather than recorded in one.
        let text_encoder = match spec.components.get(TEXT_ENCODER_COMPONENT) {
            Some(WeightsSource::Dir(staged)) => staged.clone(),
            _ => root.join(TEXT_ENCODER_COMPONENT),
        };
        let mut conditioning_bytes = safetensors_path_bytes(&text_encoder);
        if let Some(native) = &mut native_quant {
            native.conditioning_stream = conditioning::text_encoder_stream(&text_encoder)?;
            if let Some(stream) = &native.conditioning_stream {
                conditioning_bytes = stream.peak_bytes(None);
            }
        }
        // Fails closed rather than guessing: an additive stack whose file cannot be sized would
        // otherwise be charged 0, and a zero overlay on a render that holds 300 MB of factors is an
        // under-declaration in the OOM direction. The shared helper is the same one ltx uses.
        // Only unresolved inventory fixtures retain the compatibility file bound. Real task
        // partitions use source-selected installed originals/prepared caches above.
        if !resolved && !spec.adapters.is_empty() {
            overlay = mlx_gen::gen_core::adapter_stack_resident_bytes(
                &spec.adapters,
                AdapterResidencyMode::Additive,
            )
            .ok_or_else(|| CoreError::Unsupported("H3 adapter source is unavailable".into()))?;
        }
        Ok(Self {
            // The Qwen3 condition encoder is built at `Dtype::Bfloat16` on every path
            // (`text_encoder/encoder.rs`), so it does not follow `spec.precision`.
            text_encoder: conditioning_bytes,
            dit: dit_bytes,
            adaln: if pruning.is_some() {
                pruned_resident
            } else {
                if has_native {
                    max_adaln
                } else {
                    resolved_adaln_bytes(&dit, dit_bytes)
                }
            },
            pruning,
            video_vae: native_quant
                .as_ref()
                .filter(|n| n.conditioning.is_some())
                .map_or_else(
                    || materialized_component_bytes(&root.join("vae"), spec.precision),
                    |n| n.video_decode_resident_bytes,
                ),
            // The audio VAE is a BigVGAN stack built at `Dtype::Float32` unconditionally, on both
            // the encode and the decode side (`model.rs`) — never at `self.dtype`.
            audio_vae: native_quant
                .as_ref()
                .filter(|n| n.conditioning.is_some())
                .map_or_else(
                    || materialized_f32_bytes(&root.join("audio_vae")),
                    |n| n.audio_decode_resident_bytes,
                ),
            overlay,
            native_quant,
        })
    }

    /// The declaration-only footprint: no filesystem, no resolved tier, and therefore the
    /// architecture's own bf16 figure for the sub-stack.
    fn weights_free() -> Self {
        Self {
            text_encoder: 0,
            dit: 0,
            adaln: ADALN_EVICTED_BYTES,
            pruning: None,
            video_vae: 0,
            audio_vae: 0,
            overlay: 0,
            native_quant: None,
        }
    }

    /// The two decoders are one contract field; H3 is the first family with two of them.
    fn decoder(&self) -> u64 {
        self.video_vae.saturating_add(self.audio_vae)
    }

    fn base(&self) -> u64 {
        self.text_encoder
            .saturating_add(self.dit)
            .saturating_add(self.decoder())
    }
}

/// Device bytes the 50-block `adaln_proj` stack holds on a tier packed at `bits`.
///
/// The same accounting `crate::quant::nbytes` performs on a *loaded* projection, done from the
/// shipped configuration instead of from a device tensor, because a contract is resolved before
/// anything is loaded. Per block:
///
/// * a packed tier holds the triple — `out · in · bits / 8` code bytes, plus a bf16 `scales` **and**
///   `biases` entry per [`crate::quant::GROUP_SIZE`] input group, plus the dense `bias` row;
/// * `bits >= 16` is the unpacked bf16 `weight` + `bias`, i.e. [`ADALN_EVICTED_BYTES`].
///
/// `crate::convert` packs the projections at the same width and group size as the rest of the DiT's
/// linears, so this is a *tier* fact rather than a per-artifact measurement.
fn adaln_stack_bytes(bits: i32) -> u64 {
    let config = crate::dit::MiniMaxH3DitConfig::default();
    let out = config.adaln_out_features() as u64;
    let inp = config.time_embed_dim as u64;
    let blocks = DIT_BLOCKS as u64;
    if bits >= 16 {
        return blocks * (out * inp + out) * 2;
    }
    let groups = inp / crate::quant::GROUP_SIZE as u64;
    let codes = out * inp * bits.max(0) as u64 / 8;
    // `scales` and `biases` are one bf16 element per output row per input group.
    let group_metadata = out * groups * 2 * 2;
    let dense_bias = out * 2;
    blocks * (codes + group_metadata + dense_bias)
}

/// The AdaLN sub-stack's resident bytes **on the tier staged at `dit_dir`**, whose footprint is
/// `dit_bytes`.
///
/// [`ADALN_EVICTED_BYTES`] is a bf16 figure. `ComponentBytes::resolve` already honours a staged tier
/// for `transformer_bytes`, so declaring the flat 26.02 GB against an 18.78 GB q4 DiT declares a
/// sub-stack larger than the stack containing it — which `conformance_errors` refuses and
/// `Registry::memory_strategy_contract` turns into a hard error, i.e. a q4 render that cannot
/// resolve a contract at all. **A quant tier is a whole-pipeline contract; this is the segment that
/// was still reading bf16.**
///
/// Two independent legs, and the **smaller** wins:
///
/// * the **marker** leg reads the staged tier's own `config.json` `quantization.bits` — the exact
///   authority `crate::model::reconcile_tier` treats as decisive about which tier is staged — and
///   re-derives the stack at that width through [`adaln_stack_bytes`]. Exact at every shipped tier.
/// * the **footprint** leg scales the bf16 stack by the resolved DiT's share of the bf16 DiT. Never
///   exact — the f32 I/O heads, `context_embedder` and norms `crate::convert` leaves dense in every
///   tier do not shrink, so the whole-DiT ratio sits ~0.7 % above the projections' own at q4 — but
///   it reads nothing except a number `build_contract` already holds.
///
/// Taking the minimum is what makes this **safe** rather than merely accurate, and each leg closes
/// the other's failure:
///
/// * marker alone reports 26.02 GB for a packed tier whose marker is missing or unreadable, which is
///   BLOCKER 1 again;
/// * footprint alone over-declares the eviction, and an over-declared saving is the OOM direction —
///   the exact asymmetry [`ADALN_MODULATION_TABLE_MAX_BYTES`] is chosen on.
///
/// Containment holds unconditionally either way: [`ADALN_EVICTED_BYTES`] is 39.3 % of
/// [`DIT_BF16_BYTES`], so the footprint leg is always strictly below `dit_bytes` and the minimum is
/// therefore always below it too.
///
/// **The floor.** Below a resolved DiT of ~9.86 GB the scaled stack falls under the retained
/// [`ADALN_MODULATION_TABLE_MAX_BYTES`] table and the declared eviction would genuinely exclude
/// nothing, which conformance refuses. No shipped tier is close: `q4` is 18.78 GB, 1.9x above it —
/// `the_shipped_tiers_sit_clear_of_the_floor_where_the_eviction_stops_excluding_anything` pins that.
fn resolved_adaln_bytes(dit_dir: &std::path::Path, dit_bytes: u64) -> u64 {
    if dit_bytes == 0 {
        // Nothing was resolved, so there is no tier to scale to and the declaration falls back to
        // the architecture fact. `conformance_errors` skips sub-stack containment against zero asset
        // facts for exactly this case.
        return ADALN_EVICTED_BYTES;
    }
    let marked = mlx_gen::quant::packed_quant_bits_at(dit_dir)
        .ok()
        .flatten()
        .map_or(ADALN_EVICTED_BYTES, adaln_stack_bytes);
    // u128 because `ADALN_EVICTED_BYTES · DIT_BF16_BYTES` is ~1.7e21 and overflows u64.
    let scaled = (u128::from(ADALN_EVICTED_BYTES) * u128::from(dit_bytes)
        / u128::from(DIT_BF16_BYTES)) as u64;
    marked.min(scaled)
}

/// The five capability entries, with the parameter domain each implemented lever owns.
///
/// Every lever the contract marks `Implemented` must publish a [`MemoryParameterRanges`] for the
/// parameters that lever consumes, and must publish **none** for the ones it does not — both
/// directions are enforced by `validate_owned_parameter_domain`. Rungs 0 and 1 own no numeric
/// parameters, so their ranges are legitimately empty; rungs 2, 3 and 4 own tile/chunk/window
/// domains and cannot be flipped to `Implemented` without filling them in.
/// Rung 3's inapplicability reason, built from [`RUNG3_MEASURED_PEAK_DELTAS`] rather than typed out.
///
/// A reason string is the *only* justification a `StructurallyNotApplicable` declaration carries, and
/// `conformance_errors` refuses an empty one — but nothing stops a non-empty one from going stale
/// against the measurement it claims to report. Deriving it from the same constant the tests assert
/// against means a re-measurement that moves a number moves the declaration with it.
fn rung3_inapplicability_reason() -> String {
    let cells = RUNG3_MEASURED_PEAK_DELTAS
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
        "MLX's fused SDPA streams the scores at this family's shape, so there is no materialized \
         [B,H,S,S] tensor for rung 3 to bound: measured peak tracks the 4·B·H·S·D q/k/v/out \
         prediction exactly (5.966 GB against 5.965 GB predicted at S=104030, where materializing \
         the scores would add ~1212 GB). The remaining MLX mechanism — eval_per_chunk cutting the \
         lazy graph, worth -1.7% on Z-Image's denoise phase — was measured in-forward on the real \
         50-block q4 DiT on both chunk axes and COSTS denoise peak at every arm ({cells}). \
         MLX-scoped: candle-gen-minimax-h3 materializes its scores and must measure its own verdict."
    )
}

fn strategies(streamable: bool) -> Vec<MemoryStrategyCapability> {
    MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: match strategy {
                MemoryStrategy::Resident
                | MemoryStrategy::StagedResidency
                // sc-18660. The mechanism is `crate::spatial_tiling::BoundedStitch` behind
                // `MiniMaxH3VideoVae::decode_clip`; the domain is one geometry, because the tile
                // edge is an output-correctness input rather than a lever (module docs).
                | MemoryStrategy::BoundedDecode => MemoryStrategySupport::Implemented,
                // sc-18661. Measured, not argued — see the module docs and
                // `RUNG3_MEASURED_PEAK_DELTAS`. MLX-scoped: candle materializes its scores.
                MemoryStrategy::BoundedAttention => {
                    MemoryStrategySupport::StructurallyNotApplicable {
                        reason: rung3_inapplicability_reason(),
                    }
                }
                // sc-18662. Implemented on a `LoadShape::DeferredMaterialization` load and
                // `Missing` on a resident one — the rung's own shared prerequisite is that shape, so
                // a resident load genuinely does not have it. Not `StructurallyNotApplicable`: a
                // 50-block DiT and a 50-layer transformer encoder are exactly the components this
                // rung optimizes, and both are measured to bound ~9x.
                //
                // sc-18650: `streamable` is **not** the load shape alone — a configured adapter
                // makes it false too, because a windowed block is rebuilt from the staged tier and
                // would drop the factors. See [`streamable`] for why that gate cannot live in
                // `route_gate`.
                MemoryStrategy::BoundedTransformerResidency if streamable => {
                    MemoryStrategySupport::Implemented
                }
                MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                    decode_tile_edges: vec![DECODE_TILE_EDGE],
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                },
                // sc-18662. `Both`, and the second half is the point: declaring `Dit` alone would
                // leave the conditioning phase — the taller of the two stages at every tier — with
                // no lever at all, which is what AC3 refuses. Both arms are measured separately in
                // `tests/block_window_real.rs`.
                MemoryStrategy::BoundedTransformerResidency if streamable => {
                    MemoryParameterRanges {
                        transformer_window_sizes: vec![TRANSFORMER_WINDOW_SIZE],
                        transformer_window_components: vec![TransformerComponent::Both],
                        ..Default::default()
                    }
                }
                // `validate_owned_parameter_domain` enforces this in BOTH directions: an
                // implemented rung that does not own the decode parameters must publish none.
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect()
}

/// The AdaLN projection stack as a typed, evictable intra-transformer component (sc-18665).
///
/// Every field but `resident_bytes` is an architecture or checkpoint fact, so the shape of this
/// component is identical in the production and weights-free contracts — which is what lets catalog
/// conformance see the exclusion without a snapshot.
///
/// * `resident_bytes` is [`ComponentBytes::adaln`], i.e. the stack **on the tier that was actually
///   resolved** — see [`resolved_adaln_bytes`]. It is passed in rather than read from
///   [`ADALN_EVICTED_BYTES`] here because that constant is the **bf16** figure, and declaring it
///   against an 18.78 GB q4 DiT declares a sub-stack larger than the stack containing it, which
///   `conformance_errors` refuses and `Registry::memory_strategy_contract` turns into a hard error
///   — a q4 render that cannot resolve a contract at all. The weights-free contract resolves
///   nothing and so passes the architecture fact through [`ComponentBytes::weights_free`].
///
/// * `kind` is [`MemoryComponentKind::TransformerSubStack`], not `Transformer`: these bytes are
///   already inside `asset_facts.transformer_bytes`, and naming a whole transformer would charge
///   them twice.
/// * `bounded_by` is `None` because **no rung bounds this**. `crate::model::generate_impl` passes
///   [`crate::dit::adaln::AdaLnResidency::PrecomputeAndEvict`] as a *literal*, so nothing a request
///   can select reaches the other arm, and `AdaLnCache::precompute_and_evict` refuses `Resident`
///   outright. The type is not vacuous — `crate::dit::model` implements a live `Resident` arm for a
///   solver whose evaluation timesteps are not enumerable up front, and `tests/adaln_cache.rs`
///   drives it — but no shipped render selects it, which is what makes the exclusion declarable as
///   an unconditional property of this provider rather than as a lever.
fn adaln_component(resident_bytes: u64, pruned: bool) -> MemoryResidentComponent {
    if pruned {
        return MemoryResidentComponent {
            id: ADALN_COMPONENT_ID.to_owned(),
            kind: MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
            resident_bytes,
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        };
    }
    MemoryResidentComponent {
        id: ADALN_COMPONENT_ID.to_owned(),
        kind: MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
        resident_bytes,
        bounded_by: None,
        residency: MemoryComponentResidency::PrecomputedThenEvicted {
            // The projections are mapped with the rest of the DiT at the head of the denoise phase,
            // consumed once to project the whole schedule's modulation, and released before the
            // first denoise step. There is no earlier phase they survive into and no later one.
            precomputed_in: MemoryPhase::Denoise,
            retained_bytes: ADALN_MODULATION_TABLE_MAX_BYTES,
            evidence: format!(
                "tests/adaln_evict_real_weights.rs and tests/adaln_evict_memory.rs drive \
                 AdaLnCache::precompute_and_evict and pin the phase pair through \
                 common::assert_adaln_phase_envelope (sc-19449); the bf16 {ADALN_EVICTED_BYTES} B \
                 is asserted against the sum over the 50 real adaln_proj tensors in \
                 crate::dit::adaln, and resident_bytes above is that stack on the resolved tier"
            ),
        },
    }
}

/// Rung 4's prerequisites, both of them, resolved from the spec.
///
/// **The deferred load shape is necessary but not sufficient** (sc-18650). A configured adapter is
/// the second gate, and it belongs *here* rather than only in the render: `route_gate` cannot enforce
/// it, because [`MemoryRunContext`] carries no adapter axis at all — a selection is geometry, tier
/// and parameters, and the adapter list is a property of the *load*. So the only place the two can
/// be made to agree is the declaration.
///
/// Without this clause the contract published rung 4 `Implemented` for an adapter-carrying deferred
/// load, `validate_selection` admitted it, the request scope opened, and
/// `MemoryProviderContract::generation_memory` set `stream_transformer_blocks` from that engagement —
/// whereupon `crate::model::MiniMaxH3::requested_transformer_window` refused the very selection the
/// contract had just promised. Declaring `Missing` closes the loop at its head instead.
///
/// The refusal it mirrors is not a caprice: a window rebuilds each block from the staged tier per
/// step, so a forward-time residual adapter would be dropped from every windowed block. Both
/// precedents this provider mirrors gate on the same thing — `mlx-gen-wan`'s adapter residency mode
/// and `mlx-gen-ltx`'s overlay legs.
pub fn streamable(spec: &LoadSpec) -> bool {
    resolved_load_shape(spec) == LoadShape::DeferredMaterialization && spec.adapters.is_empty()
}

/// Architecture axes for the MiniMax-H3 route (epic SC-22657, E2).
///
/// [`MiniMaxH3DitConfig::default`](crate::dit::config::MiniMaxH3DitConfig) mirrors the reference
/// `transformer/config.json`, and `MiniMaxH3DitConfig::from_diffusers_json` parses that same file —
/// every key required, no defaults — at load, so the two cannot disagree silently.
///
/// `head_dim` is the config's declared `attention_head_dim`, not `hidden_size / num_heads`: this DiT
/// is deliberately non-uniform (`inner_dim` 7168 is wider than `hidden_size` 5376), so the quotient
/// would be neither integral nor the head width the attention actually uses.
///
/// The VAE is a **video** autoencoder: `patch_size` pixels and `patch_size_t` frames per latent
/// unit as `MiniMaxH3VaeConfig` states them (the products of the per-level downsample factors), so
/// `vae_temporal_scale` is a real value here.
///
/// When `spec` names a materialized snapshot directory, [`dit_config`] and [`vae_config`] re-run the
/// loaders' own `from_diffusers_json` parses over that snapshot's `transformer/config.json` and
/// `vae/config.json`, so every published axis is the snapshot's rather than the preset's
/// (SC-22667: the VAE axes were preset constants although `vae.rs` parses the snapshot config).
///
/// SC-22667 review: an install that HAS a DiT to load — a materialized snapshot, or a staged
/// [`DIT_COMPONENT`] directory — but whose `transformer/config.json` is missing, partial or
/// unparseable no longer degrades into the preset. `MiniMaxH3Dit::load_dir` errors on exactly that
/// file, so a preset published here would describe a model this load will never build. A provider
/// falls back to a preset only where the LOADER falls back to it; otherwise the trunk axes are
/// declared absent. The VAE axes are their own config read (see [`vae_config`]) and normally survive
/// that branch, so the contract still declares a real architecture axis and the MLX weights-free
/// gate stays satisfied.
fn architecture_facts(spec: &LoadSpec) -> mlx_gen::gen_core::MemoryArchitectureFacts {
    // SC-22667 review: a `vae/config.json` that is present but does not parse declines the VAE axes
    // rather than substituting the crate mirror — `vae.rs` reads that same file through the same
    // `from_diffusers_json` and propagates the failure, so no load of this snapshot has the mirror's
    // geometry. An ABSENT file keeps the mirror: that is the shape the weights-free surface and an
    // unmaterialized VAE both have, and it is the only branch where there is nothing to contradict.
    let vae = vae_config(spec);
    let latent_channels = vae
        .as_ref()
        .and_then(|vae| mlx_gen::architecture_facts::axis(vae.latent_channels));
    let vae_spatial_scale = vae
        .as_ref()
        .and_then(|vae| mlx_gen::architecture_facts::axis(vae.patch_size));
    let vae_temporal_scale = vae
        .as_ref()
        .and_then(|vae| mlx_gen::architecture_facts::axis(vae.patch_size_t));
    let Some(dit) = dit_config(spec) else {
        return mlx_gen::gen_core::MemoryArchitectureFacts {
            attention_heads: None,
            head_dim: None,
            transformer_blocks: None,
            patch_size: None,
            latent_channels,
            vae_spatial_scale,
            vae_temporal_scale,
            activation_dtype_width: Some(if spec.precision == mlx_gen::Precision::Fp32 {
                mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH
            } else {
                mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH
            }),
        };
    };
    // `patch_size` is `(t, h, w)`; only the square spatial patch has a single honest scalar, and
    // the temporal factor is already carried by `vae_temporal_scale`.
    let [_, patch_h, patch_w] = dit.patch_size;
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(dit.num_attention_heads),
        head_dim: mlx_gen::architecture_facts::axis(dit.attention_head_dim),
        transformer_blocks: mlx_gen::architecture_facts::axis(dit.num_layers),
        patch_size: (patch_h == patch_w)
            .then(|| mlx_gen::architecture_facts::axis(patch_h))
            .flatten(),
        latent_channels,
        vae_spatial_scale,
        vae_temporal_scale,
        // SC-22667: this hardcoded `HALF_ACTIVATION_WIDTH` while its own comment named the
        // exception. gen-core documents the axis as the DENOISE-phase width — explicitly not a
        // per-component byte fact — and `model.rs` selects `Dtype::Float32` for the DiT under
        // `Precision::Fp32` and `Dtype::Bfloat16` otherwise, so an Fp32 load published half the
        // width its own denoise loop computes at. The f32 audio VAE is deliberately NOT folded in
        // here: it belongs to the decode phase, which this axis does not describe.
        activation_dtype_width: Some(if spec.precision == mlx_gen::Precision::Fp32 {
            mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH
        } else {
            mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH
        }),
    }
}

/// The DiT geometry a contract should describe: the config the loader would parse, or `None` when
/// there IS a DiT to load and that config cannot be read.
///
/// `from_diffusers_json` requires every key, so a partial or variant file declines rather than
/// half-defaulting into this model's numbers.
///
/// SC-22667: the DiT is honored at its **resolved** location, not under the snapshot root. It is
/// the one tiered component, so a split `q4` install stages [`DIT_COMPONENT`] outside the snapshot
/// and has no `root/transformer` at all — `model.rs` says exactly that where it probes the same
/// file, and `ComponentBytes::resolve` already applies the staged override for both the DiT and the
/// text encoder. Reading only under the root made the probe miss on every split install, so the
/// preset was published although the loaded config was one `spec.components` lookup away.
///
/// SC-22667 review: the preset is returned only where the LOADER would use it — a spec that names
/// neither a materialized root nor a staged DiT directory, i.e. the weights-free registry surface
/// with no load to describe. Once either is present, `MiniMaxH3Dit::load_dir` reads
/// `<resolved>/config.json` and propagates any read or parse failure, so this returns `None` and the
/// caller declares the trunk axes absent instead of substituting numbers the load cannot produce.
fn dit_config(spec: &LoadSpec) -> Option<crate::dit::config::MiniMaxH3DitConfig> {
    let staged = matches!(
        spec.components.get(DIT_COMPONENT),
        Some(WeightsSource::Dir(_))
    );
    if !staged && mlx_gen::architecture_facts::materialized_root(spec).is_none() {
        return Some(crate::dit::config::MiniMaxH3DitConfig::default());
    }
    resolved_dit_config(spec)
}

/// The DiT config as the loader would resolve it, or `None` when that directory carries no
/// parseable one.
///
/// The directory precedence is `resolve_dit_dir`'s exactly: a staged [`DIT_COMPONENT`] directory
/// WINS OUTRIGHT and is never followed by a root fallback, because the loader does not fall back
/// either — a staged tier whose `config.json` is unreadable fails the load rather than silently
/// building the snapshot's base geometry.
fn resolved_dit_config(spec: &LoadSpec) -> Option<crate::dit::config::MiniMaxH3DitConfig> {
    let dir = match spec.components.get(DIT_COMPONENT) {
        Some(WeightsSource::Dir(staged)) => staged.clone(),
        _ => mlx_gen::architecture_facts::materialized_root(spec)?
            .join(crate::model::BASE_DIT_PARTITION),
    };
    let text = mlx_gen::gen_core::__model_input_reader!(
std::fs::read_to_string(dir.join("config.json")),
"crates/media/mlx-gen/mlx-gen-minimax-h3/src/memory_strategy.rs::resolved_dit_config::std::fs::read_to_string@76461",
ArtifactInput,
dir.join("config.json"),
|__model_input_path| std::fs::read_to_string(__model_input_path)
).ok()?;
    crate::dit::config::MiniMaxH3DitConfig::from_diffusers_json(&text).ok()
}

/// The video-VAE geometry a contract should describe: the materialized snapshot's own
/// `vae/config.json` — the file `vae.rs` parses through this same
/// `MiniMaxH3VaeConfig::from_diffusers_json` at load — when one exists and parses, this crate's
/// mirror of the shipped VAE when there is no such file to read, and `None` when the file is there
/// and does not parse.
///
/// That last arm is the SC-22667 review fix, and it mirrors [`dit_config`]: `from_diffusers_json`
/// requires every key, so a partial or variant file declines whole rather than half-defaulting —
/// but declining used to mean `unwrap_or_default()`, i.e. publishing the crate mirror's geometry
/// for a snapshot whose own file says something else. `MiniMaxH3Vae::load` reads that exact path
/// and propagates both the read and the parse failure, so a load of such a tree does not exist to
/// describe, and E2 declares the axes absent instead.
///
/// An ABSENT file keeps the mirror deliberately: the weights-free registry surface has no snapshot
/// at all, and a caller building a contract before the VAE component is materialized is in the same
/// position — nothing there contradicts the shipped geometry.
fn vae_config(spec: &LoadSpec) -> Option<crate::config::MiniMaxH3VaeConfig> {
    let Some(root) = mlx_gen::architecture_facts::materialized_root(spec) else {
        return Some(crate::config::MiniMaxH3VaeConfig::default());
    };
    let Ok(text) = mlx_gen::gen_core::__model_input_reader!(
std::fs::read_to_string(root.join("vae").join("config.json")),
"crates/media/mlx-gen/mlx-gen-minimax-h3/src/memory_strategy.rs::vae_config::std::fs::read_to_string@78050",
ArtifactInput,
root.join("vae").join("config.json"),
|__model_input_path| std::fs::read_to_string(__model_input_path)
) else {
        return Some(crate::config::MiniMaxH3VaeConfig::default());
    };
    crate::config::MiniMaxH3VaeConfig::from_diffusers_json(&text).ok()
}

fn build_contract(
    spec: &LoadSpec,
    components: &ComponentBytes,
    load_shape: LoadShape,
    streamable: bool,
    // sc-22737: passed in rather than hard-coded, so the weights-free surface cannot publish the
    // production identity and the production one is keyed on the artifact's own tier.
    calibration: Option<MemoryCalibrationIdentity>,
) -> MemoryProviderContract {
    MemoryProviderContract {
        phase_facts: components.native_quant.as_ref().map(|native| {
            let mut facts = mlx_gen::gen_core::MemoryPhaseFacts::staged(
                mlx_gen::gen_core::StagedWeightSchedule::ThreeStage,
            );
            facts.native_quant = Some(native.clone());
            facts
        }),
        architecture_facts: architecture_facts(spec),
        provider_id: MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::MlxMetal {
            // This flag rests on the AdaLN evict and nothing else. That evict drains the allocator
            // cache rather than migrating active → cache, because
            // `crate::dit::adaln::drain_allocator_cache` repeats `clear_cache()` while active is
            // still falling — and it is the one site sc-18665's exclusion is declared against, so
            // the declaration stands on a drain that is real where the bytes are claimed.
            //
            // It is deliberately NOT the claim it used to carry — that "the AdaLN evict AND EVERY
            // PHASE RELEASE" drain — and the retracted half is deleted rather than restated. The
            // phase releases are a separate question, owned by sc-17151. This comment therefore
            // states no `clear_cache()` call count for them: such a count is a fact about another
            // story's code, and it would go stale the moment sc-17151 lands — that story moves the
            // release path onto a shared retrying drain, which can only strengthen this flag.
            bounded_wired_residency: true,
            // MLX mmaps and materializes per tensor: a bare 66 GB `MiniMaxH3Dit::load` leaves peak
            // memory at 33 KB. This is the *intra-tensor* fact, and it is independent of
            // `LOAD_SHAPE`, which is about a block schedule.
            lazy_or_mmap_materialization: true,
            // `mlx_rs::transforms::eval` is forced before every release; under lazy evaluation a
            // drop without it frees nothing.
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
        strategies: strategies(streamable),
        // No decode-quality geometry table is declared for this route, and that is a statement of
        // fact rather than a declared-but-refused authority — identical to every other MLX video
        // provider after the sc-18325 sweep.
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape,
        // The shared graph is sufficient: rung 1 is selected explicitly and depends on nothing, and
        // no rung this provider implements adds a realization-specific edge. sc-18662 adds rung 4's.
        additional_prerequisites: Vec::new(),
        // Nothing to exclude: the cost-order default never drags in an unimplemented rung, because
        // `MemoryProviderContract::engages` already intersects with declared support.
        //
        // sc-18665 asked whether the AdaLN evict belongs here. It does not, and the two are not
        // alternatives: a `MemoryStrategyEngagementExclusion` removes a *rung* from a selection's
        // engaged composition, and the evict is not a rung — it removes *bytes* from a formula.
        // Conformance would refuse the attempt anyway, since both ends of an exclusion must be
        // implemented strategies. The evict is declared on the formula's resident component instead
        // (`adaln_component`), which is also where its unconditionality is recorded.
        default_engagement_exclusions: Vec::new(),
        // The staged phase order is hardcoded in `generate_impl`, not a load-time default a
        // `GenerationMemory` block could switch off, so an explicit all-disabled block would
        // misrepresent what a `Resident` selection gets. Leaving the block absent preserves exactly
        // the behavior the loader has.
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases(),
            synchronized_phase_release: true,
            // sc-18660. `conformance_errors` requires this hook whenever rung 2 is `Implemented`,
            // and its converse forbids declaring rung 2 `StructurallyNotApplicable` while it is
            // set — so the pair cannot drift apart.
            decode_tiling: true,
            attention_chunking: false,
            // sc-18662. `conformance_errors` requires this hook whenever rung 4 is `Implemented`
            // and forbids it while the rung is `StructurallyNotApplicable`, so the pair cannot
            // drift; here it tracks the resolved load shape for the same reason the rung does.
            transformer_window_materialization: streamable,
        },
        // A phase envelope is *max over phases*, which is the whole point for this family: the
        // floor is `max(TE, DiT, VAE)`, never the sum, because the three are never co-resident.
        //
        // `FrameCount` is the axis no image family needed. It is declared even though the peak
        // sc-18659 observed was flat across duration, because that flatness belonged to the
        // conditioning stage: the denoise and decode phase expressions are genuinely
        // frame-dependent. sc-19120's packed text-encoder tiers landed and removed the TE's mark
        // (see `CONDITIONING_STAGE_PEAK_Q4_BYTES`), so the later stages are legible and the
        // variable is load-bearing rather than anticipatory. `ConditioningTokenCount` is the
        // conditioning phase's only real input.
        //
        // sc-18665 makes this the *component* variant. The AdaLN projections are the one part of
        // the DiT that does not survive into the denoise steady state, and a plain `PhaseEnvelope`
        // has nowhere to say so — `asset_facts.transformer_bytes` is a single load-exact scalar, so
        // an estimate built from it charges 26 GB the runtime does not hold.
        formula: MemoryFormulaKind::ComponentPhaseEnvelope {
            phases: phases(),
            variables: {
                let mut variables = vec![
                    MemoryFormulaVariable::AssetBytes,
                    MemoryFormulaVariable::PixelCount,
                    MemoryFormulaVariable::FrameCount,
                    MemoryFormulaVariable::ConditioningTokenCount,
                    // sc-18660. The decode phase's scratch is a function of the tile area, not the
                    // canvas area — one 256 px tile is decoded at a time regardless of canvas. The
                    // variable is declared because the phase expression needs it; its coefficient is
                    // calibration evidence and is NOT measured yet (see the sc-18660 notes).
                    MemoryFormulaVariable::DecodeTileArea,
                ];
                // Declared **only when an adapter is configured** (sc-18650), so an ordinary load's
                // formula is byte-for-byte the one this provider has always published and no
                // existing calibration record is disturbed. `conformance_errors` requires this
                // variable exactly when a typed auxiliary component is present, and the two are
                // built from the same condition here so they cannot drift.
                if components.overlay > 0 {
                    variables.push(MemoryFormulaVariable::OverlayBytes);
                }
                variables
            },
            resident_components: {
                let mut resident = Vec::new();
                if components.adaln > 0 {
                    let mut component =
                        adaln_component(components.adaln, components.pruning.is_some());
                    if components.pruning.is_none() {
                        if let Some(native) = &components.native_quant {
                            if let MemoryComponentResidency::PrecomputedThenEvicted {
                                retained_bytes,
                                ..
                            } = &mut component.residency
                            {
                                let rows = SourcePruneSizing::conditioned_timestep_bound(
                                    crate::model::MAX_STEPS as usize,
                                )
                                .expect("nonempty native schedule");
                                *retained_bytes = native
                                    .tasks
                                    .iter()
                                    .map(|t| t.table_bytes_per_timestep.saturating_mul(rows as u64))
                                    .max()
                                    .unwrap_or(0);
                            }
                        }
                    }
                    // A supplied small projection stack can retain more cache than it frees.
                    // Native phases still charge that cache; no generic net eviction is claimed.
                    let no_saving = components.native_quant.is_some()
                        && matches!(component.residency,
                        MemoryComponentResidency::PrecomputedThenEvicted { retained_bytes, .. }
                        if retained_bytes >= component.resident_bytes);
                    if !no_saving {
                        resident.push(component);
                    }
                }
                // The LoRA stack as a typed auxiliary component. `MemoryComponentKind::AdapterStack`
                // is the ltx spelling for the same forward-time-residual install, and
                // `WholeRender` is literal here: `crate::model::load_task_dit` installs the factors
                // onto the task's DiT before the first step and nothing releases them until the DiT
                // itself goes at the end of denoise.
                if components.overlay > 0 {
                    resident.push(MemoryResidentComponent {
                        id: "adapter_stack".to_owned(),
                        kind: MemoryComponentKind::AdapterStack,
                        resident_bytes: components.overlay,
                        bounded_by: None,
                        residency: MemoryComponentResidency::WholeRender,
                    });
                }
                resident
            },
        },
        // The calibration identity carries the RESOLVED shape, not the resident default: a
        // deferred load's peaks are a different curve from a resident load's, and an evidence
        // record that named the wrong one would be matched against measurements of the other.
        calibration,
        asset_facts: MemoryAssetFacts {
            base_bytes: components.base(),
            conditioning_bytes: components.text_encoder,
            transformer_bytes: components.dit,
            decoder_bytes: components.decoder(),
            // **The configured LoRA stack, and nothing else** (sc-18650).
            //
            // This said "MiniMax-H3 accepts no adapters … `reject_unknown_components` allows only
            // `transformer`" until this story. Both halves were false and had been since sc-18724
            // landed adapter support on 2026-08-13: the descriptor declares `supports_lora: true`,
            // `crate::model::load_task_dit` installs the published `lightx2v/Minimax-h3-Turbo`
            // exports through `crate::adapters::apply_minimax_h3_adapters`, and
            // `reject_unknown_components` is called with **two** components (`transformer` and
            // `text_encoder`). The comment simply predated the feature, and while the contract went
            // unpublished nothing read it.
            //
            // The rest of the sentence is still true and is what this field means: there is no
            // ControlNet, no IP-adapter and no identity encoder on this family, so the LoRA stack is
            // the only auxiliary residency there is.
            overlay_bytes: components.overlay,
        },
        runtime: Default::default(),
    }
}

/// The production contract: asset facts read off the resolved snapshot.
pub fn contract_for(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    let load_shape = resolved_load_shape(spec);
    Ok(build_contract(
        spec,
        &ComponentBytes::resolve(spec)?,
        load_shape,
        streamable(spec),
        // The identity carries the RESOLVED shape, not the resident default: a deferred load's
        // peaks are a different curve from a resident load's.
        production_calibration_fingerprint(spec)
            .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, load_shape)),
    ))
}

/// The weights-free fixture contract: the identical route declaration with zero asset facts.
///
/// Catalog conformance uses this when the snapshot is unavailable. It must **not** require the
/// snapshot to exist — no byte count is read off disk here — and it must not diverge from
/// [`contract_for`] in anything but the byte counts. It does consult the spec's weights directory
/// for the DiT config when one is materialized (see the crate-private `dit_config`); that is not a
/// byte count, and on the registry's never-created sentinel path it reads nothing and publishes
/// the preset.
pub fn weights_free_contract(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    let load_shape = resolved_load_shape(spec);
    Ok(build_contract(
        spec,
        &ComponentBytes::weights_free(),
        load_shape,
        streamable(spec),
        Some(MemoryCalibrationIdentity::new(
            STATIC_CALIBRATION_FINGERPRINT,
            load_shape,
        )),
    ))
}

/// The canvas the weights-free behavior fixtures admit: the shipped default, exactly at
/// [`CANVAS_MAX_PIXELS`].
const FIXTURE_WIDTH: u32 = 1344;
const FIXTURE_HEIGHT: u32 = 768;

/// The shortest legal clip — 124 frames, 5.1667 s at 24 fps.
///
/// The shared [`standard_memory_behavior_context`] defaults to a single 1024x1024 frame, and
/// **both halves of that are illegal here**: `1024·1024` is 1.6 % over [`CANVAS_MAX_PIXELS`], and
/// `T = 1` is off the `17n + 5` lattice and does not render at all. A fixture that kept the default
/// would be admitted only by a route gate that checks nothing.
const FIXTURE_FRAMES: u32 = LEGAL_FRAME_COUNTS[0] as u32;

/// The three render routes, each with its own evidence-key mode spelling.
///
/// `t2va` and `fl2va` map onto the shared text-to-image / image-to-image spellings; `ref2va` has no
/// shared spelling and takes [`MemoryMode::Other`], which exists for exactly this.
fn routes() -> Vec<(MemoryMode, u32)> {
    vec![
        (MemoryMode::TextToImage, 0),
        (MemoryMode::ImageToImage, 0),
        (MemoryMode::Other("ref2va".to_owned()), 1),
    ]
}

/// Admit a bounded-decode geometry — **the** rung-2 predicate, and the only one.
///
/// Both admission seams call this: `route_gate` (the `safety_check` half) and the request scope's
/// `decode_validator` (the `configure_decode` half). Sharing one predicate is not tidiness — it is
/// what stops the pair drifting, which is the failure this family has already shipped twice, most
/// recently when `encode_clip` hardcoded its tile constants while `decode_clip` read `self.tiling`
/// so `disable_tiling()` disabled only half the VAE (sc-19008).
///
/// The domain is a singleton, so this refuses everything except the reference geometry. The message
/// names *why* rather than only *what*, because a caller reaching it has asked for the one thing
/// this rung cannot trade: output fidelity for memory.
pub fn validate_decode_geometry(edge: u32, overlap: u32) -> mlx_gen::gen_core::Result<()> {
    if edge == DECODE_TILE_EDGE && overlap == DECODE_OVERLAP {
        return Ok(());
    }
    Err(CoreError::Unsupported(format!(
        "{MODEL_ID}: bounded decode admits only the reference geometry {DECODE_TILE_EDGE}px tile / \
         {DECODE_OVERLAP}px overlap, got {edge}/{overlap}. The tile geometry is an output-\
         correctness input copied from the published VAE, not a memory lever: MiniMax-H3 was \
         released with tiling on and the released frames are the blended-tile ones, so a budgeted \
         tile would change the output (sc-18786). A zero overlap additionally starves the seams, \
         and that corruption is visible across frames rather than within one."
    )))
}

/// Reject a run context whose geometry the render itself would refuse.
///
/// This is the non-vacuous half of admission: the lattice and canvas gates are the same ones
/// `crate::pipeline::resolve_geometry` enforces, so a context that passes here is a context the
/// generator can actually run. `use_pid` is refused outright — there is no PiD decode route.
fn route_gate(context: &MemoryRunContext) -> mlx_gen::gen_core::Result<()> {
    if context.use_pid {
        return Err(CoreError::Unsupported(format!(
            "{MODEL_ID} has no PiD decode route"
        )));
    }
    // sc-18660. Guarded by contract-aware engagement rather than an ordinal compare, so a rung the
    // contract does not implement never demands its parameters.
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
    let geometry = context.geometry;
    if !LEGAL_FRAME_COUNTS.contains(&(geometry.frames as i32)) {
        return Err(CoreError::Unsupported(format!(
            "{MODEL_ID}: {} frames is off the 17n+5 lattice (124..=345); T=1 does not render",
            geometry.frames
        )));
    }
    if !geometry.width.is_multiple_of(SPATIAL_STRIDE)
        || !geometry.height.is_multiple_of(SPATIAL_STRIDE)
    {
        return Err(CoreError::Unsupported(format!(
            "{MODEL_ID}: {}x{} is not a multiple of the {SPATIAL_STRIDE}px stride",
            geometry.width, geometry.height
        )));
    }
    if geometry.width.saturating_mul(geometry.height) > CANVAS_MAX_PIXELS {
        return Err(CoreError::Unsupported(format!(
            "{MODEL_ID}: {}x{} exceeds the {CANVAS_MAX_PIXELS}px canvas budget",
            geometry.width, geometry.height
        )));
    }
    if geometry.batch != 1 {
        return Err(CoreError::Unsupported(format!(
            "{MODEL_ID} renders one clip per request, got batch {}",
            geometry.batch
        )));
    }
    Ok(())
}

/// The numeric tier a spec loads at — the one immutable axis of a selection this provider varies.
///
/// The floor list is empty because every component is packed at the DiT's own tier: the packed text
/// encoder tracks whatever `{base}.scales` it ships with rather than a declared per-component floor
/// (see [`crate::model::TEXT_ENCODER_COMPONENT`]), and the two VAEs are dense on every tier.
///
/// Resolved in one place so the loaded route (`crate::model::load`, which captures this on the
/// generator) and the registry's weights-free probe cannot end up grading against different tiers.
pub fn numeric_tier(spec: &LoadSpec) -> MemoryNumericTier {
    mlx_gen::request_scope::loaded_tier(spec)
}

/// The provider's real admission check for a **loaded** route, callable before any weight file is
/// opened.
///
/// Takes the already-resolved tier rather than the spec because the loaded generator is the caller
/// that matters: `Generator::memory_strategy_safety_check` holds a contract and a tier captured once
/// at load (sc-18650), not the `LoadSpec` that produced them.
pub fn safety_check_at_tier(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    mlx_gen::gen_core::standard_safety_check(&MEMORY_PROFILE, contract, context, tier)
}

/// The MiniMax-H3 MLX memory profile (sc-27307): requested tier, the joint-AV route gate inside the
/// shared pipeline, and the per-tier fingerprint spelling with the measured bf16 cell pinned. The
/// behavior fixture stays provider-owned (the shared default geometry is off this family's frame
/// lattice, so each route's context carries the `FIXTURE_*` geometry).
const MEMORY_PROFILE: mlx_gen::gen_core::ProviderMemoryProfile =
    mlx_gen::gen_core::ProviderMemoryProfile {
        gates: mlx_gen::gen_core::MemoryProfileGates::within(profile_route_gate),
        fingerprint: Some(mlx_gen::gen_core::MemoryFingerprintTable {
            template: "minimax-h3-{tier}-mlx-staged-joint-av-eager-abi3-v1",
            routes: &[],
            pins: &[mlx_gen::gen_core::MemoryFingerprintPin {
                route: "",
                tier: CALIBRATED_TIER,
                fingerprint: MEMORY_CALIBRATION_FINGERPRINT,
            }],
        }),
        ..mlx_gen::gen_core::ProviderMemoryProfile::STANDARD
    };

fn profile_route_gate(
    _contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    _tier: MemoryNumericTier,
) -> mlx_gen::gen_core::Result<()> {
    route_gate(context)
}

/// The spec-taking admission check — the [`MEMORY_REGISTRATION`] entry point, which probes
/// weights-free from a spec and so resolves the tier here rather than holding one.
///
/// **The bare name is load-bearing across backends.** `scripts/check-workspace.py` compares the two
/// backends' `pub const` declarations as canonicalized *source text*, so `MEMORY_REGISTRATION` must
/// spell its `safety_check` field identically here and in `candle-gen-minimax-h3` — which registers
/// the shorthand `safety_check,` against its own spec-taking function of that name. Renaming this to
/// the wan-style `registered_safety_check` reds that gate even though nothing about the registration
/// actually changed; the loaded variant is [`safety_check_at_tier`] for that reason.
pub fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    mlx_gen::gen_core::standard_registered_safety_check(&MEMORY_PROFILE, spec, contract, context)
}

/// Weights-free executable fixtures for one implemented optimized rung — one per render route.
pub fn registered_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> mlx_gen::gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || contract
            .capability(strategy)
            .is_none_or(|capability| capability.support != MemoryStrategySupport::Implemented)
    {
        return Ok(Vec::new());
    }
    let tier = numeric_tier(spec);
    routes()
        .into_iter()
        .map(|(mode, reference_count)| {
            let mut context = standard_memory_behavior_context(
                contract,
                strategy,
                tier,
                MemoryBehaviorRoute {
                    mode,
                    reference_count,
                    use_pid: false,
                    has_phases: true,
                    overlay: None,
                },
            )?;
            // The shared default geometry is illegal for this family — see `FIXTURE_FRAMES`.
            context.geometry = MemoryGeometry {
                width: FIXTURE_WIDTH,
                height: FIXTURE_HEIGHT,
                batch: 1,
                frames: FIXTURE_FRAMES,
                reference_count,
            };
            // `MemoryBehaviorFixture::new` carries `frames` across from `context.geometry` along
            // with width, height, batch and reference count (sc-19591), so the request this family
            // is graded against declares `FIXTURE_FRAMES` rather than the one-frame default that
            // would be refused as off-lattice. Asserted by
            // `the_fixture_request_declares_the_lattice_frame_count` below; no post-hoc override.
            Ok(MemoryBehaviorFixture::new(context))
        })
        .collect()
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
                MODEL_ID,
                context.geometry,
                contract.generation_memory(&context.selection),
                context.use_pid,
                DIT_BLOCKS as usize,
                // sc-18660. The same predicate the `safety_check` route gate uses, so `configure_decode`
                // and admission cannot disagree about what is legal.
                |_use_pid, edge, overlap| validate_decode_geometry(edge, overlap),
            )?;
            // sc-18662. **Declaration is not enforcement.** Publishing `transformer_window_sizes` without
            // this leaves `materialize_transformer_window` refusing every window the contract just
            // advertised — the registry conformance walk drives the declared parameter through the scope and
            // caught exactly that. Guarded by contract-aware `engages` rather than an ordinal compare, so a
            // resident load (where rung 4 is `Missing`) never admits a window.
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
            Ok(config)
        },
    )
}

/// Registry-facing entry point: no device cleanup, because the weights-free probe never allocated.
pub fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    mlx_gen::gen_core::standard_begin_request(
        &MEMORY_PROFILE,
        MODEL_ID,
        spec,
        contract,
        context,
        registered_scope,
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

/// Production entry point: terminal cleanup drains the MLX allocator cache.
///
/// This is what `Generator::begin_memory_strategy_request` calls on the loaded generator (sc-18650).
/// Until that override existed the function was reachable from nothing: the macro-emitted
/// `impl Generator` left the trait default in the vtable slot, so a shipped render opened no scope.
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

/// The memory-strategy registration paired with the `minimax_h3` generator.
pub const MEMORY_REGISTRATION: mlx_gen::gen_core::MemoryRegistration =
    mlx_gen::gen_core::MemoryRegistration {
        provider_id: MODEL_ID,
        contract: contract_for,
        safety_check,
    };

/// The weights-free contract fixture catalog conformance resolves instead of [`contract_for`].
pub const MEMORY_CONTRACT_FIXTURE: mlx_gen::gen_core::MemoryContractFixtureRegistration =
    mlx_gen::gen_core::MemoryContractFixtureRegistration {
        provider_id: MODEL_ID,
        contract: weights_free_contract,
        surface_specs: memory_contract_surface_specs,
    };

/// The shared MLX Bf16/Q4/Q8 witness set: this provider ships all three tiers and both eager and
/// deferred (sc-18662) materialization shapes, so no selector filter is needed. A local name (the
/// wan/ltx idiom) so the fixture registration spells identically across the two backends for the
/// cross-backend geometry gate.
fn memory_contract_surface_specs() -> Vec<mlx_gen::gen_core::MemoryContractSurfaceSpec> {
    mlx_gen::gen_core::mlx_memory_contract_surface_specs()
}

/// The executable behavior seam every optimized implemented rung must have.
pub const MEMORY_BEHAVIOR: mlx_gen::gen_core::MemoryBehaviorRegistration =
    mlx_gen::gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: registered_fixture,
        begin_request: registered_begin_request,
    };

/// Request-only sparse estimate; off leaves the dense memory contract byte-for-byte unchanged.
pub fn veda_workspace_bound(
    req: mlx_gen::gen_core::veda::workspace::Request<'_>,
) -> mlx_gen::gen_core::Result<Option<mlx_gen::gen_core::veda::workspace::WorkspaceBound>> {
    mlx_gen::gen_core::veda::workspace::for_video_request(
        req,
        mlx_gen::gen_core::veda::workspace::Backend::Mlx,
    )
}

/// Loaded-generator sparse bounds use the same selected source geometry as conditioning.
pub fn veda_workspace_bound_for_contract(
    req: mlx_gen::gen_core::veda::workspace::Request<'_>,
    contract: &MemoryProviderContract,
) -> mlx_gen::gen_core::Result<Option<mlx_gen::gen_core::veda::workspace::WorkspaceBound>> {
    if req.veda.is_none_or(|veda| !veda.enabled) {
        return Ok(None);
    }
    let patch = match contract
        .phase_facts
        .as_ref()
        .and_then(|p| p.native_quant.as_ref())
    {
        Some(native) => native.selected_patch(req)?,
        None => [1, 2, 2], // unresolved inventory only; no materialized source geometry exists
    };
    mlx_gen::gen_core::veda::workspace::for_video_request_with_patch(
        req,
        mlx_gen::gen_core::veda::workspace::Backend::Mlx,
        patch,
    )
}

#[cfg(test)]
mod tests;
