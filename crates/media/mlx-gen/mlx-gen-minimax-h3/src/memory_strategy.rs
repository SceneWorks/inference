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
//! Rungs 3 and 4 are declared [`MemoryStrategySupport::Missing`] — honestly, because neither is
//! built:
//!
//! * **Rung 3 (`BoundedAttention`)** — sc-18661. Deliberately **not**
//!   [`MemoryStrategySupport::StructurallyNotApplicable`], even though MLX's fused SDPA already
//!   streams the scores: measured peak tracks `4·B·H·S·D` (5.966 GB measured against 5.965 GB
//!   predicted at `S = 104_030`, where materializing the score tensor would add ~1,212 GB). That
//!   is a strong argument that chunking bounds nothing *on this backend*, but sc-18661 owns
//!   measuring and recording that verdict, and `StructurallyNotApplicable` additionally satisfies
//!   prerequisite edges **vacuously** — a semantic that should not be introduced ahead of the
//!   measurement. **Do not copy this MLX reasoning to candle**, which has no fused streaming SDPA.
//! * **Rung 4 (`BoundedTransformerResidency`)** — sc-18662, and it is gated on
//!   [`LoadShape::DeferredMaterialization`], which this loader does not implement (see
//!   [`LOAD_SHAPE`]).
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
//!   twice. They are not auxiliary either, so `overlay_bytes` stays 0.
//! * **`residency`, not `default_engagement_exclusions`.** That list excludes a *rung* from
//!   cost-order engagement; it cannot remove bytes from a formula, and there is no rung here to
//!   exclude. `bounded_by` is `None` for the same reason — the drop is unconditional on the shipped
//!   path.
//! * **Net, not gross.** The precompute keeps a modulation table in the projections' place, so the
//!   declared exclusion is `ADALN_EVICTED_BYTES − ADALN_MODULATION_TABLE_MAX_BYTES`. Declaring the
//!   flat 26.02 GB claims a saving the runtime does not deliver, and the declared-versus-measured
//!   guard goes red on it.

use mlx_gen::gen_core::{
    safetensors_path_bytes, standard_memory_behavior_context,
    standard_memory_strategy_safety_check, Error as CoreError, LoadShape, LoadSpec,
    MemoryAssetFacts, MemoryBackendRealization, MemoryBehaviorFixture, MemoryBehaviorRoute,
    MemoryCalibrationIdentity, MemoryComponentKind, MemoryComponentResidency, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryGeometry, MemoryLifecycleCapabilities, MemoryMode,
    MemoryNumericTier, MemoryParameterRanges, MemoryPhase, MemoryProviderContract,
    MemoryRequestScope, MemoryResidentComponent, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategyCapability, MemoryStrategySupport, ResidentRequestMemory,
    TransformerComponent, WeightsSource,
};

use crate::denoise::LEGAL_FRAME_COUNTS;
use crate::model::{DIT_COMPONENT, MODEL_ID};
use crate::pipeline::{CANVAS_MAX_PIXELS, SPATIAL_STRIDE};

/// Calibration identity for this provider's measurements.
///
/// Bump the `vN` token whenever tensor layout, the quantization floor, or the execution structure
/// changes — every one of those invalidates the measured stage peaks above.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "minimax-h3-mlx-staged-joint-av-eager-abi3-v1";

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

/// The audio VAE is out of scope for rung 2, declared rather than implied.
///
/// It is 0.61 GB against the video VAE's 10.42 GB (17× smaller), and structurally has nothing to
/// tile: `crate::audio_vae` is a DAC-lineage 1-D stack whose decode is a BigVGAN vocoder over a
/// waveform. There is no spatial grid, so the `decode_tile_edge` / `decode_overlap` domain has no
/// meaning for it and the rung's mechanism does not run on that path. Named as a constant so the
/// boundary is greppable and testable rather than a sentence in a doc comment.
pub const AUDIO_VAE_IS_OUT_OF_SCOPE_FOR_TILING: bool = true;

/// The load shape this loader actually has today, pinned rather than mirrored from the spec.
///
/// [`LoadShape::DeferredMaterialization`] means *transformer blocks are materialized through a
/// block schedule*. `MiniMaxH3Dit::load_dir` builds the whole 50-block stack, so this provider is
/// [`LoadShape::EagerMaterialization`] no matter what a caller asks for. A `LoadShape` a provider
/// does not implement may be answered by advertising the corresponding rung unavailable rather than
/// by rejecting the load, which is what rung 4's `Missing` does. sc-18662 changes this.
pub const LOAD_SHAPE: LoadShape = LoadShape::EagerMaterialization;

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
    video_vae: u64,
    audio_vae: u64,
}

impl ComponentBytes {
    fn resolve(spec: &LoadSpec) -> Self {
        let root = match &spec.weights {
            WeightsSource::Dir(root) => root.clone(),
            WeightsSource::File(path) => path.parent().unwrap_or(path).to_path_buf(),
        };
        let dit = match spec.components.get(DIT_COMPONENT) {
            Some(WeightsSource::Dir(staged)) => staged.clone(),
            _ => root.join(DIT_COMPONENT),
        };
        Self {
            text_encoder: safetensors_path_bytes(root.join("text_encoder")),
            dit: safetensors_path_bytes(dit),
            video_vae: safetensors_path_bytes(root.join("vae")),
            audio_vae: safetensors_path_bytes(root.join("audio_vae")),
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

/// The five capability entries, with the parameter domain each implemented lever owns.
///
/// Every lever the contract marks `Implemented` must publish a [`MemoryParameterRanges`] for the
/// parameters that lever consumes, and must publish **none** for the ones it does not — both
/// directions are enforced by `validate_owned_parameter_domain`. Rungs 0 and 1 own no numeric
/// parameters, so their ranges are legitimately empty; rungs 2, 3 and 4 own tile/chunk/window
/// domains and cannot be flipped to `Implemented` without filling them in.
fn strategies() -> Vec<MemoryStrategyCapability> {
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
                // sc-18661 / sc-18662. See the module docs for why rung 3 is `Missing` and not
                // `StructurallyNotApplicable` despite the fused-SDPA measurement.
                MemoryStrategy::BoundedAttention
                | MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                    decode_tile_edges: vec![DECODE_TILE_EDGE],
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                },
                // `validate_owned_parameter_domain` enforces this in BOTH directions: an
                // implemented rung that does not own the decode parameters must publish none.
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect()
}

/// The AdaLN projection stack as a typed, evictable intra-transformer component (sc-18665).
///
/// Every field is an architecture or checkpoint fact rather than a resolved footprint, so this is
/// identical in the production and weights-free contracts — which is what lets catalog conformance
/// see the exclusion without a snapshot.
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
fn adaln_component() -> MemoryResidentComponent {
    MemoryResidentComponent {
        id: ADALN_COMPONENT_ID.to_owned(),
        kind: MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
        resident_bytes: ADALN_EVICTED_BYTES,
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
                 common::assert_adaln_phase_envelope (sc-19449); the {ADALN_EVICTED_BYTES} B is \
                 asserted against the sum over the 50 real adaln_proj tensors in crate::dit::adaln"
            ),
        },
    }
}

fn build_contract(components: &ComponentBytes) -> MemoryProviderContract {
    MemoryProviderContract {
        provider_id: MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::MlxMetal {
            // The AdaLN evict drains the allocator cache rather than migrating active → cache:
            // `crate::dit::adaln::drain_allocator_cache` repeats `clear_cache()` while active is
            // still falling. `model::release` and `mlx_gen::residency::clear_allocator_cache` call
            // `clear_cache()` exactly once each, so this flag is NOT the claim it used to carry
            // — that "the AdaLN evict AND EVERY PHASE RELEASE" drain — and the retracted half is
            // deleted rather than restated. sc-17151 owns making the other two sites match.
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
        strategies: strategies(),
        pid_decode_routes: None,
        load_shape: LOAD_SHAPE,
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
            transformer_window_materialization: false,
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
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::FrameCount,
                MemoryFormulaVariable::ConditioningTokenCount,
                // sc-18660. The decode phase's scratch is a function of the tile area, not the
                // canvas area — one 256 px tile is decoded at a time regardless of canvas. The
                // variable is declared because the phase expression needs it; its coefficient is
                // calibration evidence and is NOT measured yet (see the sc-18660 notes).
                MemoryFormulaVariable::DecodeTileArea,
            ],
            resident_components: vec![adaln_component()],
        },
        calibration: Some(MemoryCalibrationIdentity::new(
            MEMORY_CALIBRATION_FINGERPRINT,
            LOAD_SHAPE,
        )),
        asset_facts: MemoryAssetFacts {
            base_bytes: components.base(),
            conditioning_bytes: components.text_encoder,
            transformer_bytes: components.dit,
            decoder_bytes: components.decoder(),
            // No auxiliary networks: MiniMax-H3 accepts no adapters, no ControlNet, no IP-adapter
            // and no identity encoder (`reject_unknown_components` allows only `transformer`).
            overlay_bytes: 0,
        },
        runtime: Default::default(),
    }
}

/// The production contract: asset facts read off the resolved snapshot.
pub fn contract_for(spec: &LoadSpec) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract(&ComponentBytes::resolve(spec)))
}

/// The weights-free fixture contract: the identical route declaration with zero asset facts.
///
/// Catalog conformance uses this when the snapshot is unavailable. It must **not** touch the
/// filesystem, and it must not diverge from [`contract_for`] in anything but the byte counts.
pub fn weights_free_contract(
    _spec: &LoadSpec,
) -> mlx_gen::gen_core::Result<MemoryProviderContract> {
    Ok(build_contract(&ComponentBytes {
        text_encoder: 0,
        dit: 0,
        video_vae: 0,
        audio_vae: 0,
    }))
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

/// The provider's real admission check, callable before any weight file is opened.
pub fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        }),
        Some(&|| route_gate(context)),
    )
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
    let tier = MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    };
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
            let mut fixture = MemoryBehaviorFixture::new(context);
            // `MemoryBehaviorFixture::new` derives the request from `context.geometry` for width,
            // height, batch and reference count, but **not** for `frames`, which falls through to
            // `GenerationRequest::default()` and lands at `None` (== a single frame). That is
            // invisible to every provider whose legal geometry includes one frame; this family is
            // the first whose lattice excludes it, so the defaulted request gets re-graded against
            // the 124-frame geometry admitted just above and refused as off-lattice — a
            // self-inconsistent fixture failing its own provider. Restate the frame count so the
            // request and the geometry it is meant to represent agree.
            //
            // Any future provider whose legal frame counts exclude 1 must do the same until the
            // shared builder propagates `frames` itself. Deliberately not fixed there: that builder
            // is a main-owned contract read by every provider on the ladder and needs its own
            // reviewed PR, coordinated with the epics that own memory admission.
            fixture.request.frames = Some(FIXTURE_FRAMES);
            Ok(fixture)
        })
        .collect()
}

fn begin_request_with_cleanup(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        MODEL_ID,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        DIT_BLOCKS as usize,
        // sc-18660. The same predicate the `safety_check` route gate uses, so `configure_decode`
        // and admission cannot disagree about what is legal.
        |_use_pid, edge, overlap| validate_decode_geometry(edge, overlap),
    )?;
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    )))
}

/// Registry-facing entry point: no device cleanup, because the weights-free probe never allocated.
pub fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        spec,
        contract,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

/// Production entry point: terminal cleanup drains the MLX allocator cache.
pub fn begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> mlx_gen::gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        spec,
        contract,
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
    };

/// The executable behavior seam every optimized implemented rung must have.
pub const MEMORY_BEHAVIOR: mlx_gen::gen_core::MemoryBehaviorRegistration =
    mlx_gen::gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: registered_fixture,
        begin_request: registered_begin_request,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryProviderContract, MemoryRunOutcome, MemorySelection,
        MemoryStrategyParameters, Precision, ProviderRegistryBuilder,
    };
    use mlx_gen::GenerationRequest;
    use std::path::Path;

    /// One named, independently applied mutation of a known-good contract.
    type ContractMutation = (&'static str, Box<dyn Fn(&mut MemoryProviderContract)>);

    /// The catalog conformance spec: a directory that does not exist, and a load shape this
    /// provider deliberately does **not** implement.
    fn weightless_spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/nonexistent".into()))
            .with_load_shape(LoadShape::DeferredMaterialization)
    }

    fn declared() -> MemoryProviderContract {
        weights_free_contract(&weightless_spec()).expect("weights-free contract")
    }

    fn support(
        contract: &MemoryProviderContract,
        strategy: MemoryStrategy,
    ) -> MemoryStrategySupport {
        contract
            .capability(strategy)
            .unwrap_or_else(|| panic!("{strategy:?} must appear in the strategy table"))
            .support
            .clone()
    }

    /// Write a snapshot tree whose component directories hold **sparse** `.safetensors` files of
    /// the exact measured sizes. `safetensors_path_bytes` stats rather than parses, so this costs
    /// no disk and still exercises the real directory-name wiring.
    fn sparse_snapshot(root: &Path, sizes: &[(&str, u64)]) {
        for (component, bytes) in sizes {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).expect("component dir");
            let file = std::fs::File::create(dir.join("model.safetensors")).expect("shard");
            file.set_len(*bytes).expect("sparse shard");
        }
    }

    // --- AC1: the resolved contract is the DECLARED one, never the compatibility default ---------

    /// A Resident-only assertion is a false green here: `compatibility_default` also reports
    /// Resident. This distinguishes the two by the things a fallback contract **cannot** produce.
    #[test]
    fn resolved_contract_is_declared_and_not_the_compatibility_default() {
        let spec = weightless_spec();
        let registry = crate::provider_registry().expect("registry");
        let registration = registry
            .memory_strategy_registrations()
            .find(|registration| registration.provider_id == MODEL_ID)
            .expect("minimax_h3 must be registered on the memory ladder");
        let contract = (registration.contract)(&spec).expect("registered contract");

        let fallback = MemoryProviderContract::compatibility_default(
            MODEL_ID,
            MemoryBackendRealization::MlxMetal {
                bounded_wired_residency: true,
                lazy_or_mmap_materialization: true,
                explicit_evaluation_and_synchronization: true,
                cache_eviction: true,
            },
        );
        assert_ne!(
            contract, fallback,
            "the resolved contract must not be the resident-only fallback"
        );

        // Each of these is impossible for `compatibility_default`, so each independently proves the
        // declaration was resolved rather than the fallback.
        assert_eq!(
            support(&contract, MemoryStrategy::StagedResidency),
            MemoryStrategySupport::Implemented,
            "the fallback declares every optimized rung Missing"
        );
        assert!(
            contract.calibration.is_some(),
            "the fallback carries no calibration identity"
        );
        assert!(
            matches!(
                contract.formula,
                MemoryFormulaKind::ComponentPhaseEnvelope { .. }
            ),
            "the fallback formula is AssetBytesPlusHeadroom, got {:?}",
            contract.formula
        );
        assert_eq!(contract.lifecycle.phases, phases());
        assert!(contract.lifecycle.synchronized_phase_release);
    }

    /// The registration is wired end to end: removing any one of the three lines, or misspelling
    /// the provider id on any of them, fails `build()`.
    #[test]
    fn the_three_memory_registrations_are_present_and_id_matched() {
        let registry = crate::provider_registry().expect("registry");
        for (kind, found) in [
            (
                "memory strategy",
                registry
                    .memory_strategy_registrations()
                    .any(|r| r.provider_id == MODEL_ID),
            ),
            (
                "contract fixture",
                registry
                    .memory_contract_fixture_registrations()
                    .any(|r| r.provider_id == MODEL_ID),
            ),
            (
                "behavior",
                registry
                    .memory_behavior_registrations()
                    .any(|r| r.provider_id == MODEL_ID),
            ),
        ] {
            assert!(found, "{MODEL_ID} is missing its {kind} registration");
        }

        // A misspelled id has no matching generator and must be rejected at build time rather than
        // becoming a contract with no executable owner.
        let typo = ProviderRegistryBuilder::new()
            .register_generator(crate::model::REGISTRATION)
            .register_memory_strategy(mlx_gen::gen_core::MemoryRegistration {
                provider_id: "minimax_h4",
                contract: contract_for,
                safety_check,
            })
            .build();
        assert!(
            typo.is_err(),
            "a misspelled memory-strategy provider id must fail registry build"
        );
    }

    /// The contract a caller resolves reports the declared id, not a neighbour's.
    #[test]
    fn contract_reports_its_own_provider_id() {
        assert_eq!(declared().provider_id, MODEL_ID);
        assert_eq!(MEMORY_REGISTRATION.provider_id, MODEL_ID);
        assert_eq!(MEMORY_CONTRACT_FIXTURE.provider_id, MODEL_ID);
        assert_eq!(MEMORY_BEHAVIOR.provider_id, MODEL_ID);
    }

    // --- AC2 + AC3: weights-free conformance drives every implemented rung ----------------------

    /// The shipped registry conformance, run in this crate's own default lane rather than only in
    /// the catalog. It resolves the **fixture** contract, then for every optimized rung declared
    /// `Implemented` it drives `valid_fixtures` → `safety_check` → `begin_request` →
    /// `configure_request` → `enter_phase` → `leave_phase` → `finish`, and proves the safety check
    /// is blind to none of the calibration ABI, fingerprint, load shape, numeric tier or budget.
    #[test]
    fn registry_conformance_drives_every_implemented_rung_weights_free() {
        let registry = crate::provider_registry().expect("registry");
        gen_core_testkit::memory_strategy_registry_conformance(&registry, &weightless_spec());
    }

    /// The fixture contract is genuinely weights-free: zero asset facts, no filesystem traversal,
    /// and the identical route declaration.
    #[test]
    fn fixture_contract_is_weights_free_and_matches_the_route_declaration() {
        let spec = weightless_spec();
        let fixture = (MEMORY_CONTRACT_FIXTURE.contract)(&spec).expect("fixture contract");
        assert_eq!(fixture.asset_facts, MemoryAssetFacts::default());
        let production = (MEMORY_REGISTRATION.contract)(&spec).expect("production contract");
        assert_eq!(fixture.strategies, production.strategies);
        assert_eq!(fixture.lifecycle, production.lifecycle);
        assert_eq!(fixture.formula, production.formula);
        assert_eq!(fixture.calibration, production.calibration);
        assert_eq!(fixture.load_shape, production.load_shape);
        assert_eq!(fixture.backend, production.backend);
    }

    /// Rung 1's mechanism is reachable in the provider's own request surface: the scope writes the
    /// canonical memory block, and the resulting request survives the generator's real validator.
    ///
    /// This is the step registry conformance cannot take — it checks the block against the
    /// contract, not against the model. A `Resident` selection is included as the control arm, so
    /// the test proves the two selections are *distinguishable* rather than that some block exists.
    #[test]
    fn staged_residency_reaches_the_generators_own_validator() {
        let spec = weightless_spec();
        let contract = declared();
        let caps = crate::model::descriptor().capabilities;

        for (strategy, expect_staged) in [
            (MemoryStrategy::StagedResidency, true),
            (MemoryStrategy::Resident, false),
        ] {
            let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
                &contract,
                strategy,
                MemoryNumericTier {
                    precision: spec.precision,
                    quant: spec.quantize,
                    component_precision_floors: &[],
                },
                MemoryBehaviorRoute {
                    mode: MemoryMode::TextToImage,
                    reference_count: 0,
                    use_pid: false,
                    has_phases: true,
                    overlay: None,
                },
            )
            .expect("behavior context");
            context.geometry = MemoryGeometry {
                width: FIXTURE_WIDTH,
                height: FIXTURE_HEIGHT,
                batch: 1,
                frames: FIXTURE_FRAMES,
                reference_count: 0,
            };
            assert_eq!(
                safety_check(&spec, &contract, &context),
                MemorySafetyDecision::Accept,
                "{strategy:?} must admit its own legal geometry"
            );

            let mut scope = registered_begin_request(&spec, &contract, &context)
                .expect("begin_request")
                .expect("scope");
            let mut request = GenerationRequest {
                prompt: "a slow pan across a rainy street at night".into(),
                width: FIXTURE_WIDTH,
                height: FIXTURE_HEIGHT,
                frames: Some(FIXTURE_FRAMES),
                ..Default::default()
            };
            scope.configure_request(&mut request).expect("configure");
            assert_eq!(
                request.memory.is_some_and(|memory| memory.stage_residency),
                expect_staged,
                "{strategy:?} must map to stage_residency={expect_staged}"
            );
            crate::model::validate_request(&caps, &request).unwrap_or_else(|error| {
                panic!("{strategy:?} produced a request the generator rejects: {error}")
            });
            scope.finish(MemoryRunOutcome::Complete).expect("finish");
        }
    }

    // --- sc-18660 AC4: rung 2 is REACHABLE, not merely declared --------------------------------

    /// **The whole rung-2 chain, executed.** Contract → selection → `safety_check` → scope →
    /// `configure_decode` → `GenerationRequest.memory` → the pipeline's own reader.
    ///
    /// The chain ends at [`crate::pipeline::decode_tiling_for`], which is the function
    /// `MiniMaxH3::decode_video` calls. Nothing here asserts that a *default* arrived — the
    /// admitted edge equals the default, so such an assertion would pass with the plumbing deleted
    /// (`test_asserting_a_default_value_is_a_false_green`). What is asserted instead is that the
    /// selection's values travel intact **and** that the two negative directions are refused.
    #[test]
    fn bounded_decode_reaches_the_decode_call_through_the_request() {
        let spec = weightless_spec();
        let contract = declared();
        let tier = MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        };
        let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::BoundedDecode,
            tier,
            MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: true,
                overlay: None,
            },
        )
        .expect("behavior context");
        context.geometry = MemoryGeometry {
            width: FIXTURE_WIDTH,
            height: FIXTURE_HEIGHT,
            batch: 1,
            frames: FIXTURE_FRAMES,
            reference_count: 0,
        };

        // The representative selection carries the published domain, not a hardcoded pair.
        assert_eq!(
            (
                context.selection.parameters.decode_tile_edge,
                context.selection.parameters.decode_overlap
            ),
            (Some(DECODE_TILE_EDGE), Some(DECODE_OVERLAP)),
            "the representative rung-2 selection must carry the published domain"
        );
        assert_eq!(
            safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Accept,
            "rung 2 must admit its own published geometry"
        );

        let mut scope = registered_begin_request(&spec, &contract, &context)
            .expect("begin_request")
            .expect("scope");
        // The scope's own decode hook admits it — this is the seam registry conformance drives.
        scope
            .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, context.geometry)
            .expect("the published geometry must configure");
        // …and refuses a budgeted substitute, which is the direction that carries information.
        let refused = scope
            .configure_decode(128, 32, context.geometry)
            .expect_err("a smaller tile must be refused, not silently clamped");
        assert!(
            refused.to_string().contains("output-correctness"),
            "the refusal must name WHY the geometry is pinned, got: {refused}"
        );

        let mut request = GenerationRequest {
            prompt: "a slow pan across a rainy street at night".into(),
            width: FIXTURE_WIDTH,
            height: FIXTURE_HEIGHT,
            frames: Some(FIXTURE_FRAMES),
            ..Default::default()
        };
        scope.configure_request(&mut request).expect("configure");
        let memory = request
            .memory
            .expect("a rung-2 request carries a memory block");
        assert!(memory.tile_vae_decode, "rung 2 must set the decode signal");
        assert_eq!(
            (memory.decode_tile_edge, memory.decode_overlap),
            (Some(DECODE_TILE_EDGE), Some(DECODE_OVERLAP)),
            "the selected geometry must survive onto the request"
        );

        // The pipeline reader accepts the block the scope built, and produces the reference
        // geometry — the same one `decode_clip` executes.
        let tiling = crate::pipeline::decode_tiling_for(&request).expect("admitted block resolves");
        assert!(tiling.enabled);
        assert_eq!(
            (tiling.tile_height, tiling.tile_width),
            (DECODE_TILE_EDGE as i32, DECODE_TILE_EDGE as i32)
        );
        assert_eq!(
            (tiling.overlap_height, tiling.overlap_width),
            (DECODE_OVERLAP as i32, DECODE_OVERLAP as i32)
        );
        scope.finish(MemoryRunOutcome::Complete).expect("finish");
    }

    /// **The reader is not a constant folder.** A hand-built block naming an unadmitted geometry —
    /// including the starved zero overlap — must be refused by the same predicate admission uses.
    ///
    /// This is the half that goes red if `decode_tiling_for` is replaced by
    /// `Ok(SpatialTiling::default())`: every positive assertion above would still pass, because the
    /// admitted value *is* the default.
    #[test]
    fn the_pipeline_reader_refuses_every_unadmitted_decode_geometry() {
        use mlx_gen::gen_core::GenerationMemory;
        let request = |edge: Option<u32>, overlap: Option<u32>| GenerationRequest {
            prompt: "p".into(),
            width: FIXTURE_WIDTH,
            height: FIXTURE_HEIGHT,
            frames: Some(FIXTURE_FRAMES),
            memory: Some(GenerationMemory {
                tile_vae_decode: true,
                decode_tile_edge: edge,
                decode_overlap: overlap,
                ..Default::default()
            }),
            ..Default::default()
        };
        // A budgeted tile — the substitution that would reintroduce sc-18786's defect.
        for edge in [64u32, 128, 192, 320, 512, 2048] {
            assert!(
                crate::pipeline::decode_tiling_for(&request(Some(edge), Some(DECODE_OVERLAP)))
                    .is_err(),
                "a {edge}px tile must be refused: it changes the decode"
            );
        }
        // **A starved overlap**, the failure mode this rung must not be able to select.
        for overlap in [0u32, 16, 32, 128] {
            assert!(
                crate::pipeline::decode_tiling_for(&request(Some(DECODE_TILE_EDGE), Some(overlap)))
                    .is_err(),
                "a {overlap}px overlap must be refused"
            );
        }
        // The admitted pair resolves, so the guard rejects the geometry and not the request shape.
        assert!(crate::pipeline::decode_tiling_for(&request(
            Some(DECODE_TILE_EDGE),
            Some(DECODE_OVERLAP)
        ))
        .is_ok());
        // A request that engages no rung, or names no geometry, gets the shipped default — the
        // pre-sc-18660 behaviour, byte for byte.
        for plain in [
            GenerationRequest::default(),
            GenerationRequest {
                memory: Some(GenerationMemory::default()),
                ..Default::default()
            },
        ] {
            assert_eq!(
                crate::pipeline::decode_tiling_for(&plain).expect("plain request resolves"),
                crate::spatial_tiling::SpatialTiling::default()
            );
        }
    }

    /// A zero overlap cannot be published, independent of the reader: `validate_ranges` forbids a
    /// zero entry in any declared range. Two guards, one failure mode, neither relying on the other.
    #[test]
    fn a_starved_overlap_cannot_be_published_as_a_candidate() {
        let mut starved = declared();
        for entry in &mut starved.strategies {
            if entry.strategy == MemoryStrategy::BoundedDecode {
                entry.parameters.decode_overlaps = vec![0];
            }
        }
        let errors = starved.conformance_errors();
        assert!(
            !errors.is_empty(),
            "a zero decode overlap must fail conformance, not ship as a candidate"
        );
        // And the shipped domain is the geometry the decode actually runs — read back through
        // `SpatialTiling::default()`, which is what `decode_clip` consults, rather than off the
        // constants, since a constant compared with itself proves nothing.
        let tiling = crate::spatial_tiling::SpatialTiling::default();
        assert!(tiling.enabled);
        assert_eq!(
            (DECODE_TILE_EDGE as i32, DECODE_OVERLAP as i32),
            (tiling.tile_height, tiling.overlap_height),
            "the published domain must be the geometry decode_clip executes"
        );
        assert_eq!(
            (DECODE_TILE_EDGE as i32, DECODE_OVERLAP as i32),
            (tiling.tile_width, tiling.overlap_width),
            "the domain is square; a per-axis split would need two more candidates"
        );
        // The audio VAE's exclusion is a size-and-shape fact, not a preference: an order of
        // magnitude smaller, and a waveform has no spatial grid a tile edge could mean anything on.
        // Both operands are constants, so this is a compile-time guard — which is the stronger
        // form, since the premise cannot then drift into a runtime-only check nobody runs.
        const _: () = assert!(
            VIDEO_VAE_BYTES > AUDIO_VAE_BYTES * 10,
            "the audio-VAE exclusion assumes it is an order of magnitude smaller"
        );
    }

    /// Reachability has a negative half: a rung this provider does not implement must be refused by
    /// the same admission path, not silently accepted.
    #[test]
    fn unimplemented_rungs_are_refused_at_admission() {
        let spec = weightless_spec();
        let contract = declared();
        // sc-18660 landed rung 2, so it is no longer in this list. Rungs 3 and 4 remain unbuilt —
        // see the module docs for why rung 3 is `Missing` rather than `StructurallyNotApplicable`.
        for strategy in [
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                support(&contract, strategy),
                MemoryStrategySupport::Missing,
                "{strategy:?} is not built yet"
            );
            assert!(
                contract
                    .validate_selection(&MemorySelection {
                        strategy,
                        tier: MemoryNumericTier {
                            precision: spec.precision,
                            quant: spec.quantize,
                            component_precision_floors: &[],
                        },
                        parameters: MemoryStrategyParameters::default(),
                    })
                    .is_err(),
                "{strategy:?} must not be selectable"
            );
            assert!(
                !contract.engages(MemoryStrategy::StagedResidency, strategy),
                "{strategy:?} must not be engaged by a rung-1 selection"
            );
        }
    }

    /// The geometry gate is real: the shared fixture default this family had to override is
    /// refused, and so is every other illegal shape — one mutation at a time, so each clause is
    /// proven independently rather than as a set.
    #[test]
    fn each_geometry_clause_refuses_its_own_illegal_shape() {
        let spec = weightless_spec();
        let contract = declared();
        let legal = MemoryGeometry {
            width: FIXTURE_WIDTH,
            height: FIXTURE_HEIGHT,
            batch: 1,
            frames: FIXTURE_FRAMES,
            reference_count: 0,
        };
        let context = |geometry: MemoryGeometry, use_pid: bool| MemoryRunContext {
            selection: MemorySelection {
                strategy: MemoryStrategy::StagedResidency,
                tier: MemoryNumericTier {
                    precision: spec.precision,
                    quant: spec.quantize,
                    component_precision_floors: &[],
                },
                parameters: MemoryStrategyParameters::default(),
            },
            calibration_abi: mlx_gen::gen_core::MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: MEMORY_CALIBRATION_FINGERPRINT.to_owned(),
            load_shape: LOAD_SHAPE,
            mode: MemoryMode::TextToImage,
            has_reference: geometry.reference_count > 0,
            use_pid,
            has_phases: true,
            geometry,
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 128 * 1024 * 1024 * 1024,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: CONDITIONING_STAGE_PEAK_BYTES,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "unit-test".to_owned(),
        };

        assert_eq!(
            safety_check(&spec, &contract, &context(legal, false)),
            MemorySafetyDecision::Accept,
            "the control arm must be admitted, or every rejection below is vacuous"
        );

        // 1. the shared 1024x1024 single-frame default: over the canvas budget AND off-lattice.
        let shared_default = MemoryGeometry {
            width: 1024,
            height: 1024,
            frames: 1,
            ..legal
        };
        // 2. off-lattice frame count, canvas legal.
        let off_lattice = MemoryGeometry {
            frames: 125,
            ..legal
        };
        // 3. T=1, which does not render at all.
        let single_frame = MemoryGeometry { frames: 1, ..legal };
        // 4. stride violation, canvas and lattice legal.
        let off_stride = MemoryGeometry {
            width: 1330,
            ..legal
        };
        // 5. canvas budget, stride and lattice legal.
        let over_canvas = MemoryGeometry {
            width: 1344,
            height: 1024,
            ..legal
        };
        // 6. batch, everything else legal.
        let batched = MemoryGeometry { batch: 2, ..legal };

        for (name, geometry, use_pid) in [
            ("shared 1024x1024 T=1 default", shared_default, false),
            ("off-lattice frames", off_lattice, false),
            ("T=1", single_frame, false),
            ("off-stride width", off_stride, false),
            ("over-budget canvas", over_canvas, false),
            ("batch > 1", batched, false),
            ("PiD decode", legal, true),
        ] {
            assert!(
                matches!(
                    safety_check(&spec, &contract, &context(geometry, use_pid)),
                    MemorySafetyDecision::Reject { .. }
                ),
                "{name} must be refused"
            );
        }
    }

    // --- AC3: a malformed or incomplete contract fails, one mutation at a time -------------------

    /// Each mutation is applied **alone** to a known-good contract. Mutating them together would
    /// prove the set conforms, not that each guard independently detects its own breakage.
    #[test]
    fn each_contract_mutation_is_independently_detected() {
        assert!(
            gen_core_testkit::check_memory_strategy_contract(&declared()).is_ok(),
            "the shipped contract must conform, or every mutation below is vacuous"
        );

        let mutations: Vec<ContractMutation> = vec![
            (
                "a dropped strategy entry",
                Box::new(|c| {
                    c.strategies
                        .retain(|entry| entry.strategy != MemoryStrategy::BoundedAttention);
                }),
            ),
            (
                "a duplicated strategy entry",
                Box::new(|c| {
                    let first = c.strategies[0].clone();
                    c.strategies.push(first);
                }),
            ),
            (
                "an empty StructurallyNotApplicable reason",
                Box::new(|c| {
                    for entry in &mut c.strategies {
                        if entry.strategy == MemoryStrategy::BoundedDecode {
                            entry.support = MemoryStrategySupport::StructurallyNotApplicable {
                                reason: "   ".to_owned(),
                            };
                        }
                    }
                }),
            ),
            (
                "StagedResidency without synchronized phase release",
                Box::new(|c| c.lifecycle.synchronized_phase_release = false),
            ),
            (
                "StagedResidency with a missing lifecycle phase",
                Box::new(|c| c.lifecycle.phases.retain(|p| *p != MemoryPhase::Decode)),
            ),
            (
                "BoundedDecode implemented with no tile domain",
                Box::new(|c| implement_without_range(c, MemoryStrategy::BoundedDecode)),
            ),
            (
                "BoundedAttention implemented with no chunk domain",
                Box::new(|c| implement_without_range(c, MemoryStrategy::BoundedAttention)),
            ),
            (
                "BoundedTransformerResidency implemented with no window domain",
                Box::new(|c| {
                    implement_without_range(c, MemoryStrategy::BoundedTransformerResidency)
                }),
            ),
            (
                "base_bytes that does not equal its components",
                Box::new(|c| c.asset_facts.base_bytes += 1),
            ),
            (
                "a malformed calibration fingerprint",
                Box::new(|c| {
                    c.calibration = Some(MemoryCalibrationIdentity::new("No_Version", LOAD_SHAPE));
                }),
            ),
            (
                "a calibration load shape that disagrees with the contract",
                Box::new(|c| {
                    c.calibration = Some(MemoryCalibrationIdentity::new(
                        MEMORY_CALIBRATION_FINGERPRINT,
                        LoadShape::DeferredMaterialization,
                    ));
                }),
            ),
            (
                "an overlay charged with no typed component",
                Box::new(|c| {
                    c.asset_facts.overlay_bytes = 1;
                    c.asset_facts.base_bytes = c.asset_facts.base_bytes.saturating_add(1);
                }),
            ),
        ];

        for (name, mutate) in mutations {
            let mut contract = declared();
            mutate(&mut contract);
            assert!(
                gen_core_testkit::check_memory_strategy_contract(&contract).is_err(),
                "conformance must reject {name}"
            );
        }
    }

    /// Flip one rung to `Implemented` while leaving its parameter domain empty — the exact shape
    /// AC5 asks a test to catch.
    fn implement_without_range(contract: &mut MemoryProviderContract, strategy: MemoryStrategy) {
        for entry in &mut contract.strategies {
            if entry.strategy == strategy {
                entry.support = MemoryStrategySupport::Implemented;
                entry.parameters = MemoryParameterRanges::default();
            }
        }
        match strategy {
            MemoryStrategy::BoundedDecode => contract.lifecycle.decode_tiling = true,
            MemoryStrategy::BoundedAttention => contract.lifecycle.attention_chunking = true,
            MemoryStrategy::BoundedTransformerResidency => {
                contract.lifecycle.transformer_window_materialization = true
            }
            _ => {}
        }
    }

    // --- AC5: parameter ranges are declared exactly where they are owned ------------------------

    /// Both directions: an implemented lever publishes its domain, and a rung that does not own a
    /// parameter publishes none.
    ///
    /// Since sc-18660 the positive half is real rather than vacuous — rung 2 publishes exactly the
    /// reference geometry — while every other rung is still legitimately empty.
    #[test]
    fn parameter_ranges_are_owned_by_the_rung_that_consumes_them() {
        let contract = declared();
        assert!(contract.conformance_errors().is_empty());
        for capability in &contract.strategies {
            if capability.strategy == MemoryStrategy::BoundedDecode {
                assert_eq!(
                    capability.parameters,
                    MemoryParameterRanges {
                        decode_tile_edges: vec![DECODE_TILE_EDGE],
                        decode_overlaps: vec![DECODE_OVERLAP],
                        ..Default::default()
                    },
                    "rung 2 publishes the reference geometry and nothing else"
                );
                continue;
            }
            assert_eq!(
                capability.parameters,
                MemoryParameterRanges::default(),
                "{:?} owns no numeric parameters until its rung lands",
                capability.strategy
            );
        }
        // And an implemented lever with an empty domain is an error, per rung.
        for strategy in [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            let mut mutated = declared();
            implement_without_range(&mut mutated, strategy);
            assert!(
                !mutated.conformance_errors().is_empty(),
                "{strategy:?} implemented with no MemoryParameterRanges must fail"
            );
        }
    }

    // --- AC4: asset facts, and the stage the other measured numbers belong to -------------------

    /// Tolerance for the GB figures recorded on sc-18659, in bytes. The story quotes the text
    /// encoder as 66.73 GB; the measured on-disk total is 66,714,912,872 B = 66.715 GB, so the byte
    /// constants are authoritative and the GB figures are held only to this window.
    const GB_TOLERANCE_BYTES: u64 = 20_000_000;

    fn assert_within(measured: u64, story_gb: f64, what: &str) {
        let story_bytes = (story_gb * 1e9) as u64;
        let delta = measured.abs_diff(story_bytes);
        assert!(
            delta <= GB_TOLERANCE_BYTES,
            "{what}: measured {measured} B ({:.3} GB) is {delta} B from the recorded {story_gb} GB, \
             outside the {GB_TOLERANCE_BYTES} B tolerance",
            measured as f64 / 1e9
        );
    }

    /// The four measured on-disk footprints, held to the figures recorded on the story.
    #[test]
    fn measured_component_bytes_match_the_recorded_footprints() {
        assert_within(DIT_BF16_BYTES, 66.28, "33 B DiT partition at bf16");
        assert_within(TEXT_ENCODER_BYTES, 66.73, "Qwen3-VL-32B text encoder");
        assert_within(VIDEO_VAE_BYTES, 10.42, "video VAE");
        assert_within(AUDIO_VAE_BYTES, 0.61, "audio VAE");
    }

    /// The contract charges each measured component to the right field, resolved off a real
    /// directory tree rather than from a constant.
    #[test]
    fn asset_facts_charge_each_component_to_its_own_field() {
        let root = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(
            root.path(),
            &[
                ("text_encoder", TEXT_ENCODER_BYTES),
                ("transformer", DIT_BF16_BYTES),
                ("vae", VIDEO_VAE_BYTES),
                ("audio_vae", AUDIO_VAE_BYTES),
                // `transformer_ref` is byte-identical and a render loads exactly one partition, so
                // it must NOT be added on top.
                ("transformer_ref", DIT_BF16_BYTES),
            ],
        );
        let contract =
            contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
        let facts = contract.asset_facts;
        assert_eq!(facts.conditioning_bytes, TEXT_ENCODER_BYTES);
        assert_eq!(facts.transformer_bytes, DIT_BF16_BYTES);
        assert_eq!(
            facts.decoder_bytes,
            VIDEO_VAE_BYTES + AUDIO_VAE_BYTES,
            "both decoders are charged, and only the decoders"
        );
        assert_eq!(facts.overlay_bytes, 0);
        assert_eq!(
            facts.base_bytes,
            TEXT_ENCODER_BYTES + DIT_BF16_BYTES + VIDEO_VAE_BYTES + AUDIO_VAE_BYTES
        );
        assert!(contract.conformance_errors().is_empty());
    }

    /// A tiered install stages the DiT elsewhere; the contract must charge the staged tier's bytes,
    /// not the flat snapshot's. This is the field that would silently report a bf16 floor for a q4
    /// render.
    #[test]
    fn a_staged_dit_component_is_charged_at_its_own_size() {
        let root = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(root.path(), &[("transformer", DIT_BF16_BYTES)]);
        let staged = tempfile::tempdir().expect("tempdir");
        const Q4_BYTES: u64 = 18_779_970_678;
        sparse_snapshot(staged.path(), &[("transformer", Q4_BYTES)]);

        let contract = contract_for(
            &LoadSpec::new(WeightsSource::Dir(root.path().into())).with_component(
                DIT_COMPONENT,
                WeightsSource::Dir(staged.path().join(DIT_COMPONENT)),
            ),
        )
        .expect("contract");
        assert_eq!(contract.asset_facts.transformer_bytes, Q4_BYTES);
    }

    /// The stage-attributed peaks, and the relationships between them that the epic measured. The
    /// point of this test is the **attribution**: it fails if someone re-labels the ~53 GB floor as
    /// the DiT's, or drops the tiering win the flat process-wide peak once hid.
    ///
    /// Two layers, and both are needed. The **literals** pin each stage peak to the number that
    /// stage was actually measured at; the **identities** pin the relationships the epic's argument
    /// rests on. Identities alone are not sufficient: they are relative, so a consistent re-basing
    /// of the constants — or an edit that moves a constant and its own identity literal together —
    /// satisfies every one of them while silently re-writing a measurement. These are the three
    /// numbers most exposed to the mis-attribution class, so they get the same literal anchor the
    /// AdaLN figure already had.
    #[test]
    fn measured_peaks_stay_attributed_to_the_stage_that_produced_them() {
        // Each stage peak against the literal it was measured at, independent of the others.
        assert_eq!(
            CONDITIONING_STAGE_PEAK_BYTES, 53_070_000_000,
            "conditioning stage: the dense Qwen3-VL-32B text encoder measured in isolation at \
             53.07 GB"
        );
        assert_eq!(
            DENOISE_RESIDENT_BF16_BYTES, 40_430_000_000,
            "denoise stage: bf16 DiT residency after the AdaLN precompute-and-evict, 40.43 GB"
        );
        assert_eq!(
            DENOISE_RESIDENT_Q4_BYTES, 11_630_000_000,
            "denoise stage: the same residency at q4, 11.63 GB"
        );
        // The floor is the conditioning stage, not the denoise one.
        const {
            assert!(
                CONDITIONING_STAGE_PEAK_BYTES > DENOISE_RESIDENT_BF16_BYTES,
                "the ~53 GB floor is the text encoder; the DiT's post-evict residency sits below it"
            )
        };
        assert_eq!(
            CONDITIONING_STAGE_PEAK_BYTES - DENOISE_RESIDENT_BF16_BYTES,
            12_640_000_000,
            "measured gap between the conditioning mark and bf16 denoise residency"
        );
        // The DiT is genuinely tiered: the win is real, it was only masked.
        assert_eq!(
            DENOISE_RESIDENT_BF16_BYTES - DENOISE_RESIDENT_Q4_BYTES,
            28_800_000_000,
            "bf16 -> q4 denoise-resident reduction"
        );
        // And the AdaLN eviction is the loader's own figure, not a contract-local copy.
        assert_eq!(
            ADALN_EVICTED_BYTES, 26_020_915_200,
            "the exact bytes crate::dit::adaln releases"
        );

        // --- the packed text-encoder tiers (sc-19120) ---------------------------------------
        assert_eq!(
            CONDITIONING_STAGE_PEAK_Q4_BYTES, 14_430_000_000,
            "conditioning stage on the packed q4 text-encoder tier, measured in isolation"
        );
        assert_eq!(
            CONDITIONING_STAGE_PEAK_Q8_BYTES, 26_940_000_000,
            "conditioning stage on the packed q8 text-encoder tier, measured in isolation"
        );
        assert_eq!(
            CONDITIONING_STAGE_PEAK_BYTES - CONDITIONING_STAGE_PEAK_Q4_BYTES,
            38_640_000_000,
            "dense -> q4 conditioning-stage reduction: the largest single memory change in this \
             model, and larger than the DiT's own 28.80 GB"
        );
        // The tier ladder is monotone and the stage really is tiered — a packed TE that measured
        // at or above the dense datum would mean the pack never engaged.
        const {
            assert!(
                CONDITIONING_STAGE_PEAK_Q4_BYTES < CONDITIONING_STAGE_PEAK_Q8_BYTES
                    && CONDITIONING_STAGE_PEAK_Q8_BYTES < CONDITIONING_STAGE_PEAK_BYTES,
                "the conditioning stage must fall monotonically q4 < q8 < dense"
            )
        };
        // **What the pair still owes.** Packed q4 puts the conditioner (14.43 GB) within 2.8 GB of
        // the q4 DiT's own denoise residency (11.63 GB), so the floor is now the DiT's — which
        // sc-18662's bounded transformer residency is what bounds. Neither story reaches a
        // consumer-GPU footprint alone, and this assertion is the reason why, in executable form.
        const {
            assert!(
                CONDITIONING_STAGE_PEAK_Q4_BYTES > DENOISE_RESIDENT_Q4_BYTES,
                "if the packed conditioner ever fell below the q4 denoise residency, the DiT would \
                 be the sole floor and this comment would be stale"
            )
        };
        const {
            assert!(
                ADALN_EVICTED_BYTES < DIT_BF16_BYTES,
                "the evicted projections are part of the DiT partition"
            )
        };
    }

    // --- sc-18665: the AdaLN evict as a typed intra-transformer exclusion -----------------------

    /// The declared component, resolved off the shipped contract rather than off `adaln_component`,
    /// so every assertion below travels the same seam a consumer does.
    fn adaln_declaration() -> mlx_gen::gen_core::MemoryResidentComponent {
        let components = declared().resident_components().to_vec();
        assert_eq!(
            components.len(),
            1,
            "the contract declares exactly one resident component: the AdaLN sub-stack"
        );
        assert_eq!(components[0].id, ADALN_COMPONENT_ID);
        components[0].clone()
    }

    /// **AC1.** The 26.02 GB drop is in the declared formula, and the steady-state transformer
    /// charge is lower than the load-exact one by exactly the net exclusion.
    ///
    /// Two literals, because a ratio alone would survive a consistent re-basing: the gross figure is
    /// the loader's own [`ADALN_EVICTED_BYTES`] and the retained figure is
    /// [`ADALN_MODULATION_TABLE_MAX_BYTES`]. Deleting the component, or flipping the formula back to
    /// a plain `PhaseEnvelope`, takes the delta to zero and this red.
    #[test]
    fn the_declared_formula_carries_the_adaln_exclusion_as_a_typed_sub_stack() {
        let root = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(root.path(), &[("transformer", DIT_BF16_BYTES)]);
        let contract =
            contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
        assert!(contract.conformance_errors().is_empty());

        let component = adaln_declaration();
        assert_eq!(component.resident_bytes, ADALN_EVICTED_BYTES);
        assert_eq!(
            component.steady_state_bytes(),
            ADALN_MODULATION_TABLE_MAX_BYTES
        );
        assert_eq!(
            component.kind,
            MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
            "these bytes are INSIDE transformer_bytes; a whole-Transformer kind charges them twice"
        );

        // The number the ladder's arithmetic sees change.
        let net = ADALN_EVICTED_BYTES - ADALN_MODULATION_TABLE_MAX_BYTES;
        assert_eq!(contract.evicted_component_bytes(), net);
        assert_eq!(
            contract.asset_facts.transformer_bytes - contract.steady_state_transformer_bytes(),
            net,
            "the post-precompute charge must be the load-exact one minus the net exclusion"
        );
        assert_eq!(
            contract.steady_state_transformer_bytes(),
            DIT_BF16_BYTES - net
        );
        // The exclusion is not free, and the contract says by how much.
        assert!(
            net < ADALN_EVICTED_BYTES,
            "declaring the gross {ADALN_EVICTED_BYTES} B would claim a saving the precompute does \
             not deliver: it keeps a modulation table in the projections' place"
        );
        // A sub-stack must not leak into the auxiliary legs — those are for networks beside the
        // base model, and `overlay_bytes` would charge these bytes a second time.
        assert_eq!(contract.auxiliary_resident_bytes(), 0);
        assert_eq!(contract.asset_facts.overlay_bytes, 0);
        assert_eq!(
            contract.total_resident_bytes(),
            contract.asset_facts.base_bytes
        );
    }

    /// **AC1's negative half.** With the component removed the delta collapses to zero, so the
    /// assertions above are grading the declaration rather than an accessor's arithmetic.
    #[test]
    fn removing_the_component_takes_the_declared_exclusion_to_zero() {
        let mut stripped = declared();
        let MemoryFormulaKind::ComponentPhaseEnvelope {
            phases,
            variables,
            resident_components,
        } = stripped.formula.clone()
        else {
            panic!("the shipped formula is a component envelope");
        };
        assert!(!resident_components.is_empty());
        stripped.formula = MemoryFormulaKind::PhaseEnvelope { phases, variables };
        assert_eq!(stripped.evicted_component_bytes(), 0);
        assert_eq!(
            stripped.steady_state_transformer_bytes(),
            stripped.asset_facts.transformer_bytes,
            "a plain PhaseEnvelope has nowhere to record the drop — this is the pre-sc-18665 state"
        );
        assert_ne!(declared().evicted_component_bytes(), 0);
    }

    /// **AC2.** The saving the contract claims may not exceed the saving the runtime was measured
    /// to deliver, and may not fall short of it by more than the table the precompute retains.
    ///
    /// The tolerance is [`ADALN_MODULATION_TABLE_MAX_BYTES`] — a quantity the schedule domain fixes,
    /// pinned to the real schedule by `the_retained_table_is_the_worst_case_over_the_admitted_schedule`
    /// — not a window sized to fit the numbers. Both directions carry information:
    ///
    /// * **upper** — declaring the gross [`ADALN_EVICTED_BYTES`] as if the evict were free puts the
    ///   claim 170_410_984 B above the measured drop, and this red. That is the overstatement the
    ///   story's own measurement warned about.
    /// * **lower** — declaring no eviction, or retaining more than the table, puts the claim below
    ///   the measured drop by more than the table, and this red.
    #[test]
    fn the_declared_saving_does_not_exceed_the_measured_residency_drop() {
        let contract = declared();
        let claimed = contract.evicted_component_bytes();
        // What the measurement says the drop was worth: the load-exact DiT against the residency
        // measured through the denoise loop, after the precompute-and-evict.
        let measured_drop = DIT_BF16_BYTES - DENOISE_RESIDENT_BF16_BYTES;

        assert!(
            claimed <= measured_drop,
            "the contract claims a {claimed} B saving against a measured {measured_drop} B drop \
             ({} B over). A declaration may under-claim; over-claiming turns a suppressed \
             configuration into an OOM",
            claimed.saturating_sub(measured_drop)
        );
        assert!(
            claimed + ADALN_MODULATION_TABLE_MAX_BYTES >= measured_drop,
            "the contract claims only {claimed} B of a measured {measured_drop} B drop; the \
             shortfall {} B exceeds the {ADALN_MODULATION_TABLE_MAX_BYTES} B modulation table, \
             which is the only thing that is supposed to account for it",
            measured_drop.saturating_sub(claimed)
        );

        // The gross figure is bracketed the same way, which is the sharper statement: the whole
        // disagreement between the projections' exact bytes and the measured residency drop is the
        // retained table, and nothing else.
        assert!(
            ADALN_EVICTED_BYTES >= measured_drop,
            "the evict cannot have released fewer bytes ({ADALN_EVICTED_BYTES}) than the residency \
             it caused to drop ({measured_drop})"
        );
        assert!(
            ADALN_EVICTED_BYTES - measured_drop <= ADALN_MODULATION_TABLE_MAX_BYTES,
            "{} B of the eviction is unaccounted for once the retained table is allowed for",
            ADALN_EVICTED_BYTES - measured_drop
        );
    }

    /// **AC2's constants are bound to the code, not typed twice.**
    ///
    /// [`ADALN_EVICTED_BYTES`] is re-derived from the shipped DiT configuration the way
    /// `crate::dit::adaln` derives it — `num_layers · (out_features · time_embed_dim + out_features)`
    /// at 2 B — so a config change moves both together or this goes red. And
    /// [`ADALN_MODULATION_TABLE_MAX_BYTES`] is read off a **real** [`crate::model::MAX_STEPS`]
    /// schedule rather than from a row count guessed by hand.
    #[test]
    fn the_retained_table_is_the_worst_case_over_the_admitted_schedule() {
        let config = crate::dit::MiniMaxH3DitConfig::default();
        let out = config.adaln_out_features() as u64;
        let derived = DIT_BLOCKS as u64 * (out * config.time_embed_dim as u64 + out) * 2;
        assert_eq!(
            derived, ADALN_EVICTED_BYTES,
            "the declared eviction must be the shipped config's own projection bytes"
        );

        // `num_inference_steps` counts the terminal sigma = 0, at which the model is never
        // evaluated, so the longest admitted run is `MAX_STEPS + 1` inference steps.
        let longest = crate::denoise::JointSchedule::new(crate::model::MAX_STEPS as usize + 1)
            .expect("the longest admitted schedule");
        assert_eq!(longest.num_evals(), crate::model::MAX_STEPS as usize);
        let rows = crate::denoise::adaln_schedule(&longest)
            .expect("adaln schedule")
            .modulation_rows() as u64;
        let widest = crate::dit::config::MODULATION_PARAMS as u64
            * rows
            * config.hidden_size as u64
            * DIT_BLOCKS as u64
            * 2;
        assert_eq!(
            widest, ADALN_MODULATION_TABLE_MAX_BYTES,
            "the declared retained table must be the widest one the admitted schedule can produce"
        );

        // …and it really is the worst case: a shorter schedule keeps strictly less. Without this
        // the constant could be any figure at all and the equality above would still hold.
        let default = crate::denoise::JointSchedule::new(crate::model::DEFAULT_STEPS as usize + 1)
            .expect("the default schedule");
        let default_rows = crate::denoise::adaln_schedule(&default)
            .expect("adaln schedule")
            .modulation_rows() as u64;
        assert!(
            default_rows < rows,
            "the default schedule's {default_rows} rows must sit below the admitted maximum's \
             {rows}, or the declaration is not conservative"
        );
    }

    /// **AC5.** The evict is reachable on the real generate path, and the declaration is not merely
    /// a statement about a mechanism nobody calls.
    ///
    /// This constructs what it watches rather than describing it: `AdaLnResidency` is the enum
    /// `crate::model::generate_impl` passes, and `AdaLnCache::precompute_and_evict` is the function
    /// that releases the bytes. It refuses [`crate::dit::adaln::AdaLnResidency::Resident`]
    /// **before** touching the block stack, which is why an empty stack reaches the residency guard
    /// here and not the "empty stack" one — so the arm that keeps 26 GB resident cannot be selected
    /// by accident. The end-to-end drop itself is measured by `tests/adaln_evict_memory.rs` and
    /// `tests/adaln_evict_real_weights.rs` through `common::assert_adaln_phase_envelope`.
    #[test]
    fn the_declared_eviction_is_reachable_on_the_generate_path() {
        use crate::dit::adaln::{AdaLnCache, AdaLnResidency};

        let schedule =
            crate::denoise::adaln_schedule(&crate::denoise::JointSchedule::new(9).expect("joint"))
                .expect("adaln schedule");
        let refused =
            AdaLnCache::precompute_and_evict(&mut [], schedule, AdaLnResidency::Resident, |_| {
                unreachable!("the residency guard runs before the stack is touched")
            })
            .expect_err("Resident must not precompute");
        assert!(
            refused.to_string().contains("does not precompute"),
            "the Resident arm must refuse rather than silently evict: {refused}"
        );

        // The contract's evidence string names the tests that drive the positive direction, so a
        // reader can follow the declaration to an executable measurement.
        let MemoryComponentResidency::PrecomputedThenEvicted { evidence, .. } =
            adaln_declaration().residency
        else {
            panic!("the AdaLN component declares an eviction");
        };
        for named in ["adaln_evict_real_weights.rs", "assert_adaln_phase_envelope"] {
            assert!(
                evidence.contains(named),
                "the declared evidence must name {named}, got: {evidence}"
            );
        }
    }

    /// **AC6.** The evict is **unconditional**, not opt-in and not rung-selected, and the contract
    /// says so in the only two places that can carry it.
    ///
    /// `default_engagement_exclusions` is asserted empty on purpose: it is the seam the story
    /// offered as an alternative, and it is not applicable. An exclusion removes a *rung* from a
    /// selection's engaged composition — conformance requires both ends to be implemented
    /// strategies — where this removes *bytes* from a formula. The last arm proves that is not
    /// merely an opinion: attempting to express the eviction there is refused.
    #[test]
    fn adaln_eviction_is_unconditional_not_rung_selected() {
        let root = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(root.path(), &[("transformer", DIT_BF16_BYTES)]);
        let contract =
            contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
        assert!(
            contract.default_engagement_exclusions.is_empty(),
            "an engagement exclusion removes a rung, not bytes"
        );
        assert_eq!(
            adaln_declaration().bounded_by,
            None,
            "no rung bounds this: crate::model passes PrecomputeAndEvict as a literal, so there is \
             no selection that keeps the projections loaded"
        );
        // The charge is the same whatever a request selects, including the plain resident rung —
        // which is what "unconditional" means, and would be false if the drop were a lever. Read
        // through `engaged_composition` so the loop covers compositions a caller can actually
        // select rather than an enum this accessor never sees.
        for strategy in MemoryStrategy::ALL {
            assert!(
                !contract
                    .engaged_composition(strategy)
                    .iter()
                    .any(|rung| adaln_declaration().bounded_by == Some(*rung)),
                "{strategy:?} must not engage a rung that bounds the AdaLN stack — there is none"
            );
            assert_eq!(
                contract.steady_state_transformer_bytes(),
                DIT_BF16_BYTES - (ADALN_EVICTED_BYTES - ADALN_MODULATION_TABLE_MAX_BYTES),
                "{strategy:?} must not change the AdaLN charge"
            );
        }

        // And the alternative seam is genuinely closed rather than merely unused.
        let mut misdeclared = declared();
        misdeclared.default_engagement_exclusions.push(
            mlx_gen::gen_core::MemoryStrategyEngagementExclusion {
                selection: MemoryStrategy::StagedResidency,
                excluded_rung: MemoryStrategy::BoundedAttention,
                evidence: "the AdaLN evict".to_owned(),
            },
        );
        assert!(
            !misdeclared.conformance_errors().is_empty(),
            "an exclusion naming a rung this provider does not implement must fail conformance"
        );
    }

    // --- the declaration facts that are easy to get wrong ----------------------------------------

    /// The load shape is pinned to the loader, not mirrored from the request. The spec asks for
    /// `DeferredMaterialization` and this provider still reports `EagerMaterialization`, because
    /// `MiniMaxH3Dit::load_dir` builds the whole block stack (sc-18662 changes that).
    #[test]
    fn load_shape_is_pinned_to_the_loader_not_taken_from_the_spec() {
        let spec = weightless_spec();
        assert_eq!(spec.load_shape, LoadShape::DeferredMaterialization);
        let contract = contract_for(&spec).expect("contract");
        assert_eq!(contract.load_shape, LoadShape::EagerMaterialization);
        assert_eq!(
            contract.calibration.as_ref().unwrap().load_shape,
            LoadShape::EagerMaterialization,
            "the calibration identity must carry the shape it was measured at"
        );
        // ...and the spec IS read: the asset facts come off it.
        let root = tempfile::tempdir().expect("tempdir");
        sparse_snapshot(root.path(), &[("vae", VIDEO_VAE_BYTES)]);
        let resolved =
            contract_for(&LoadSpec::new(WeightsSource::Dir(root.path().into()))).expect("contract");
        assert_eq!(resolved.asset_facts.decoder_bytes, VIDEO_VAE_BYTES);
        assert_eq!(contract.asset_facts.decoder_bytes, 0);
    }

    /// The rung-4 precondition is unmet today, and that is why rung 4 is `Missing` rather than a
    /// declaration waiting on a flag.
    #[test]
    fn rung_four_is_missing_because_its_load_shape_precondition_is_unmet() {
        let contract = declared();
        assert_eq!(contract.load_shape, LoadShape::EagerMaterialization);
        assert!(!contract.lifecycle.transformer_window_materialization);
        assert_eq!(
            support(&contract, MemoryStrategy::BoundedTransformerResidency),
            MemoryStrategySupport::Missing
        );
    }

    /// `Precision::Fp32` is a different numeric tier and must not be admitted against a bf16 load.
    #[test]
    fn a_numeric_tier_mismatch_is_refused() {
        let spec = weightless_spec();
        let contract = declared();
        let mut context = mlx_gen::gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::StagedResidency,
            MemoryNumericTier {
                precision: spec.precision,
                quant: spec.quantize,
                component_precision_floors: &[],
            },
            MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: true,
                overlay: None,
            },
        )
        .expect("context");
        context.geometry = MemoryGeometry {
            width: FIXTURE_WIDTH,
            height: FIXTURE_HEIGHT,
            batch: 1,
            frames: FIXTURE_FRAMES,
            reference_count: 0,
        };
        assert_eq!(
            safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Accept
        );
        context.selection.tier.precision = Precision::Fp32;
        assert!(matches!(
            safety_check(&spec, &contract, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    /// The block count the request scope validates windows against is the DiT's real depth.
    #[test]
    fn declared_block_count_matches_the_dit_configuration() {
        assert_eq!(
            DIT_BLOCKS as i32,
            crate::dit::MiniMaxH3DitConfig::default().num_layers
        );
    }

    /// Every render route gets its own driven fixture — `t2va`, `fl2va` and `ref2va` are three
    /// admission shapes, and `ref2va` reads a different 66 GB checkpoint.
    #[test]
    fn every_render_route_has_a_driven_fixture() {
        let spec = weightless_spec();
        let contract = declared();
        let fixtures = registered_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
            .expect("fixtures");
        let modes: Vec<String> = fixtures
            .iter()
            .map(|fixture| fixture.context.mode.as_key().to_owned())
            .collect();
        assert_eq!(modes, vec!["text_to_image", "image_to_image", "ref2va"]);
        for fixture in &fixtures {
            assert_eq!(fixture.context.geometry.frames, FIXTURE_FRAMES);
            assert_eq!(fixture.context.geometry.width, FIXTURE_WIDTH);
        }
        // A rung that is not implemented yields no fixture at all, rather than an empty-but-present
        // one the conformance harness would flag as a missing context.
        assert!(
            registered_fixture(&spec, &contract, MemoryStrategy::BoundedAttention)
                .expect("fixtures")
                .is_empty()
        );
    }
}
