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

// sc-27342: the measured constants, component byte accounting, contract assembly and route
// admission live in sibling modules; the calibration identity and registrations stay here.
mod adapters;
mod components;
mod conditioning;
mod contract;
mod measurements;
mod routes;
use components::*;
pub use contract::*;
pub use measurements::*;
pub use routes::*;

use crate::denoise::LEGAL_FRAME_COUNTS;
use crate::model::{DIT_COMPONENT, MODEL_ID, TEXT_ENCODER_COMPONENT};
use crate::pipeline::{CANVAS_MAX_PIXELS, SPATIAL_STRIDE};
use gen_core_h3::h3_memory::NativeAllocationFactsExt;
use gen_core_h3::pruned_adaln::{PrunedAdaLn, SourcePruneSizing, TABLE_KEY};

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

/// The executable behavior seam every optimized implemented rung must have.
pub const MEMORY_BEHAVIOR: mlx_gen::gen_core::MemoryBehaviorRegistration =
    mlx_gen::gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: registered_fixture,
        begin_request: registered_begin_request,
    };

#[cfg(test)]
mod tests;
