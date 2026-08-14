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
//! Rungs 2, 3 and 4 are declared [`MemoryStrategySupport::Missing`] — honestly, because none is
//! built:
//!
//! * **Rung 2 (`BoundedDecode`)** — sc-18660. The video VAE decode is chunked *temporally* today,
//!   but that is the reference's own `decode_temporal` algorithm ([`crate::chunking`]) and is
//!   unconditional; it is not a selectable bound and there is no spatial tiling at all. The audio
//!   VAE is 0.61 GB and is explicitly out of scope for tiling.
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
//! The 26.02 GB AdaLN eviction is a *typed resident-component exclusion* the ladder should be able
//! to see; that is sc-18665, and it lands as [`MemoryFormulaKind::ComponentPhaseEnvelope`] with a
//! `bounded_by` component. Until then the formula stays a plain [`MemoryFormulaKind::PhaseEnvelope`]
//! rather than declaring a component axis it cannot yet populate truthfully.

use mlx_gen::gen_core::{
    safetensors_path_bytes, standard_memory_behavior_context,
    standard_memory_strategy_safety_check, Error as CoreError, LoadShape, LoadSpec,
    MemoryAssetFacts, MemoryBackendRealization, MemoryBehaviorFixture, MemoryBehaviorRoute,
    MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable, MemoryGeometry,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategyCapability, MemoryStrategySupport, ResidentRequestMemory,
    WeightsSource,
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
/// This, not the DiT, is the ~53 GB floor. sc-19120 publishes a packed TE tier that removes it.
pub const CONDITIONING_STAGE_PEAK_BYTES: u64 = 53_070_000_000;

/// **Denoise stage.** DiT weights resident through the denoise loop at bf16, after the AdaLN
/// precompute-and-evict: 40.43 GB — 12.64 GB *below* the conditioning stage's mark.
pub const DENOISE_RESIDENT_BF16_BYTES: u64 = 40_430_000_000;

/// **Denoise stage.** The same residency at q4: 11.63 GB. The 28.80 GB delta against bf16 is the
/// measured tiering win, and it is real — the peak looked flat only because the conditioning stage
/// set the high-water mark first.
pub const DENOISE_RESIDENT_Q4_BYTES: u64 = 11_630_000_000;

/// **Denoise stage.** Exact bytes the AdaLN precompute-and-evict drops (64.56 → 38.70 GB active):
/// the 50-block stack's `adaln_proj` at bf16. Asserted against the loader in
/// `crate::dit::adaln`.
pub const ADALN_EVICTED_BYTES: u64 = 26_020_915_200;

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
                MemoryStrategy::Resident | MemoryStrategy::StagedResidency => {
                    MemoryStrategySupport::Implemented
                }
                // sc-18660 / sc-18661 / sc-18662. See the module docs for why rung 3 is `Missing`
                // and not `StructurallyNotApplicable` despite the fused-SDPA measurement.
                MemoryStrategy::BoundedDecode
                | MemoryStrategy::BoundedAttention
                | MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
            },
            parameters: MemoryParameterRanges::default(),
        })
        .collect()
}

fn build_contract(components: &ComponentBytes) -> MemoryProviderContract {
    MemoryProviderContract {
        provider_id: MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::MlxMetal {
            // The AdaLN evict and every phase release drain the allocator cache rather than
            // migrating active → cache, so residency here is genuinely bounded.
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
        default_engagement_exclusions: Vec::new(),
        // The staged phase order is hardcoded in `generate_impl`, not a load-time default a
        // `GenerationMemory` block could switch off, so an explicit all-disabled block would
        // misrepresent what a `Resident` selection gets. Leaving the block absent preserves exactly
        // the behavior the loader has.
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases(),
            synchronized_phase_release: true,
            decode_tiling: false,
            attention_chunking: false,
            transformer_window_materialization: false,
        },
        // `PhaseEnvelope` is *max over phases*, which is the whole point for this family: the floor
        // is `max(TE, DiT, VAE)`, never the sum, because the three are never co-resident.
        //
        // `FrameCount` is the axis no image family needed. It is declared even though today's
        // observed peak is flat across duration, because that flatness belongs to the conditioning
        // stage: the denoise and decode phase expressions are genuinely frame-dependent, and a
        // formula that omitted the variable would be unable to express them once sc-19120 removes
        // the TE's mark. `ConditioningTokenCount` is the conditioning phase's only real input.
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases: phases(),
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::FrameCount,
                MemoryFormulaVariable::ConditioningTokenCount,
            ],
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
            Ok(MemoryBehaviorFixture::new(context))
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
        // Rung 2 is not implemented, so every decode geometry is outside the admitted domain. This
        // is a refusal, not a placeholder: a caller that reaches it has selected a lever this
        // provider does not have.
        |_use_pid, edge, overlap| {
            Err(CoreError::Unsupported(format!(
                "{MODEL_ID} has no bounded decode route (sc-18660); refused tile {edge} / overlap {overlap}"
            )))
        },
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
            matches!(contract.formula, MemoryFormulaKind::PhaseEnvelope { .. }),
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

    /// Reachability has a negative half: a rung this provider does not implement must be refused by
    /// the same admission path, not silently accepted.
    #[test]
    fn unimplemented_rungs_are_refused_at_admission() {
        let spec = weightless_spec();
        let contract = declared();
        for strategy in [
            MemoryStrategy::BoundedDecode,
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
    /// parameter publishes none. The shipped contract implements no numeric lever, so the positive
    /// half is that every entry is legitimately empty and conformance still passes.
    #[test]
    fn parameter_ranges_are_owned_by_the_rung_that_consumes_them() {
        let contract = declared();
        assert!(contract.conformance_errors().is_empty());
        for capability in &contract.strategies {
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
        const {
            assert!(
                ADALN_EVICTED_BYTES < DIT_BF16_BYTES,
                "the evicted projections are part of the DiT partition"
            )
        };
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
