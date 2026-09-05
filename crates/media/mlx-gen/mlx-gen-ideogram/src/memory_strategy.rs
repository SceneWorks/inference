//! Ideogram 4 / Ideogram 4 Turbo MLX memory-strategy contract (sc-22732, epic sc-22723 E1/E4).
//!
//! Before this module the crate published **no** [`MemoryProviderContract`] at all: the generator
//! inherited `Generator::memory_strategy_contract`'s default `-> None`, so all six
//! `<route>:<tier>:mlx` cells the worker can load were unmeasurable — the SceneWorks memory-anchor
//! capture arm reads `contract.calibration` off the LOADED generator and refuses a contract without
//! one.
//!
//! # What this engine can actually execute
//!
//! Every rung below is decided from Ideogram's own code, never inherited from a sibling crate:
//!
//! * **[`MemoryStrategy::Resident`]** — the warm whole-model residency
//!   (`crate::model::build_residency` under [`OffloadPolicy::Resident`]). Always executable.
//! * **[`MemoryStrategy::StagedResidency`]** — executable exactly under
//!   [`OffloadPolicy::Sequential`]. See [`stages_residency`] for the code path and why the load-time
//!   policy, not the request, is the discriminator.
//! * **[`MemoryStrategy::BoundedDecode`]** — [`DECODE_SUPPORT`], `Missing`.
//! * **[`MemoryStrategy::BoundedAttention`]** — [`ATTENTION_SUPPORT`], `Missing`.
//! * **[`MemoryStrategy::BoundedTransformerResidency`]** — [`TRANSFORMER_WINDOW_SUPPORT`],
//!   `Missing`.
//!
//! # Production calibration identity
//!
//! [`production_calibration_fingerprint`] is the pure TABLE, keyed on (route, proven artifact tier);
//! `production_calibration_identity` is the BINDING, which proves the tier against the safetensors
//! headers on disk before any string is published. Six distinct cells: two routes x three tiers,
//! because the three tiers of one route are three different resident sets and one anchor cannot
//! price all three. Nothing is preserved here — this crate has never published a calibration
//! identity, so there is no measured cell to retire.
//!
//! Offload policy and load shape are deliberately NOT inputs to the string:
//! [`MemoryCalibrationIdentity::load_shape`] carries the materialization axis separately.
//!
//! The weights-free registry-conformance surface keeps its own [`STATIC_BEHAVIOR_FINGERPRINT`]
//! namespace, which can never equal a production string.

use mlx_gen::gen_core::{
    standard_memory_strategy_safety_check, Error as CoreError, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryNumericTier, MemoryPhase, MemoryProviderContract,
    MemoryRequestScope, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategySupport, Result as CoreResult,
};
use mlx_gen::{LoadSpec, OffloadPolicy, Precision, Quant, WeightsSource};

use crate::config::Ideogram4DitConfig;

/// Source-owned identity of the **weights-free registry-conformance** surface.
///
/// It exists so the declaration walk — which never opens a weight file and therefore never proves an
/// artifact tier — can build a run context (`standard_memory_behavior_context` requires *some*
/// identity) without any measured claim attaching to it. It is expanded per surface exactly the way
/// `mlx_gen_lens::memory_strategy::static_behavior_identity` expands its own, and it can never equal
/// a [`production_calibration_fingerprint`] string.
pub const STATIC_BEHAVIOR_FINGERPRINT: &str = "ideogram4-mlx-registry-behavior-v1";

/// **Rung 2 is `Missing` on Ideogram/MLX: the mechanism does not exist in this crate.**
///
/// `crate::pipeline::Ideogram4Heavy::decode` calls the FLUX.2 `Flux2Vae` decoder in one shot, and
/// nothing on the pipeline reads a tile edge, a tile overlap or a
/// [`mlx_gen::tiling::TilingConfig`] from the request — `crate::pipeline` names no tiling type at
/// all, and `crate::model::generate_impl` never inspects `req.memory`. A declared rung 2 would be a
/// parameter domain no code consumes.
pub const DECODE_SUPPORT: MemoryStrategySupport = MemoryStrategySupport::Missing;

/// **Rung 3 is `Missing` on Ideogram/MLX: the mechanism does not exist in this crate.**
///
/// `crate::transformer::block` computes attention through the unbudgeted shared SDPA path; neither
/// `crate::transformer` nor `crate::text_encoder::attention` constructs a
/// [`mlx_gen::attention::AttentionPlan`] or an `AttentionBudget`, and no request field reaches
/// either. There is no chunk size to declare a domain for.
pub const ATTENTION_SUPPORT: MemoryStrategySupport = MemoryStrategySupport::Missing;

/// **Rung 4 is `Missing` on Ideogram/MLX: the mechanism does not exist in this crate.**
///
/// The two DiTs are built whole by `crate::loader::load_transformer` /
/// `crate::loader::load_unconditional_transformer` and held by
/// `crate::pipeline::Ideogram4Heavy`; the crate names no `mlx_gen::block_residency::BlockPlan`, no
/// block stream, and no window size, so no consecutive-block window can be materialized. A load
/// shape of [`mlx_gen::LoadShape::DeferredMaterialization`] changes nothing here — it is not the
/// missing half.
pub const TRANSFORMER_WINDOW_SUPPORT: MemoryStrategySupport = MemoryStrategySupport::Missing;

/// Fail closed on an id this crate does not register: the rungs below are claims about the Ideogram
/// engine specifically, and nothing else is entitled to them.
fn validate_provider(provider_id: &str) -> CoreResult<()> {
    if matches!(provider_id, crate::MODEL_ID | crate::MODEL_ID_TURBO) {
        Ok(())
    } else {
        Err(CoreError::Unsupported(format!(
            "unknown Ideogram provider {provider_id}"
        )))
    }
}

/// The kebab route token for a registered provider id, or `None` for an id this crate does not own.
fn route_label(provider_id: &str) -> Option<&'static str> {
    match provider_id {
        crate::MODEL_ID => Some("ideogram-4"),
        crate::MODEL_ID_TURBO => Some("ideogram-4-turbo"),
        _ => None,
    }
}

/// The label a PROVEN artifact tier carries inside a production calibration string.
///
/// The input is the tier of the snapshot on disk — never [`LoadSpec::quantize`] — so a tier this
/// family does not ship is unnameable rather than silently collapsed onto a neighbour.
fn calibration_tier_label(artifact_tier: Option<Quant>) -> Option<&'static str> {
    match artifact_tier {
        None => Some("bf16"),
        Some(Quant::Q4) => Some("q4"),
        Some(Quant::Q8) => Some("q8"),
        Some(_) => None,
    }
}

/// Whether this load STAGES the render phases, i.e. whether rung 1 is executable.
///
/// `crate::model::build_residency` dispatches `spec.offload_policy` through
/// `Residency::from_policy`, whose `Sequential`
/// arm builds a `sequential` owner with `default_stage_residency = true` and whose `Resident` arm
/// warms the pair and leaves that flag `false`. `crate::model::Ideogram4::generate_impl` then
/// drives `Residency::run`, which forwards `self.default_stage_residency` as the staging authority
/// and encodes → materializes the embeds → **drops the text encoder** → loads the two DiTs + VAE,
/// bounding peak to `max(TE, DiTs+VAE)`.
///
/// So the load-time policy, not the request, is the discriminator: Ideogram has no
/// `run_request_scoped` call site and `generate_impl` never reads `req.memory`. Declaring rung 1 on
/// a `Resident` load would advertise a per-request lever this engine cannot pull; declaring it on a
/// `Sequential` load states exactly what that load already does.
pub fn stages_residency(spec: &LoadSpec) -> bool {
    matches!(spec.offload_policy, OffloadPolicy::Sequential)
}

/// Whether this load is a **clean base** load: the route's own components and nothing else.
///
/// `crate::model::load_heavy_owned` applies `spec.adapters` onto the conditional DiT and
/// `crate::model::load_pid` loads `spec.pid`, and both add resident bytes for the whole render, so
/// an overlay is a different resident set than any clean-base anchor priced.
///
/// The bundled TurboTime LoRA is deliberately NOT covered by this test. It is installed from
/// `root.join(TURBO_LORA_FILE)` inside `load_heavy_owned` — it never travels through
/// `spec.adapters` — so it is part of the turbo route's own identity rather than a user overlay, and
/// the turbo cells stay publishable.
fn clean_base(spec: &LoadSpec) -> bool {
    spec.adapters.is_empty()
        && spec.pid.is_none()
        && spec.control.is_none()
        && spec.extra_controls.is_empty()
        && spec.ip_adapter.is_none()
        && spec.identity.is_none()
        && spec.text_encoder.is_none()
        && spec.components.is_empty()
}

/// The component subdirectories whose packed width proves the tier.
///
/// `crate::convert::prequantize_turnkey` packs exactly these three and passes the VAE, tokenizer
/// and scheduler through dense, so they are the whole of the tier evidence a snapshot carries.
/// `unconditional_transformer/` is absent on a turbo snapshot and is skipped there.
const TIER_COMPONENTS: [&str; 3] = ["transformer", "unconditional_transformer", "text_encoder"];

/// The tier the snapshot on disk actually **is**, read from the seam the Ideogram loaders themselves
/// packed-detect on.
///
/// **There is no quantization marker to read.** `crate::quant` states that `lin`/`embedding`
/// auto-detect packing by the presence of `{base}.scales`, with "no `quantization` manifest to
/// read", and `crate::convert::prequantize_turnkey` only *copies* the dense source's
/// `config.json` through verbatim (`convert.rs`, the `for comp in [...]` loop) — it never calls
/// `mlx_gen::quant::write_quantized_config`. A probe on `config.json` would therefore read every
/// shipped packed tier as dense. The width is instead inferred from the u32 codes / `.scales` shape
/// ratio at `crate::quant::GROUP_SIZE`, exactly as [`mlx_gen::quant::packed_bits`] infers it at
/// load, which is the same rule `mlx-gen-sana` applies for the same reason.
///
/// The outer `Option` is "is there an artifact to name at all": `None` for a non-directory source,
/// which this engine refuses to load anyway (`crate::model::load` → `snapshot_dir`). The inner
/// `Option<Quant>` is the tier itself, `None` meaning dense/bf16.
///
/// `Err` fails closed on anything that is not a readable shipped tier: an unreadable or empty
/// component, a packed base with no `.weight` codes, a codes/scales ratio that is not an exact
/// shipped width, or two components that disagree.
fn resolved_artifact_tier(spec: &LoadSpec) -> mlx_gen::Result<Option<Option<Quant>>> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Ok(None);
    };
    let mut resolved: Option<(&str, Option<Quant>)> = None;
    for component in TIER_COMPONENTS {
        let dir = root.join(component);
        // A turbo snapshot ships no unconditional DiT; an absent optional component is not evidence.
        if component == "unconditional_transformer" && !dir.exists() {
            continue;
        }
        let width = component_packed_width(&dir, component)?;
        match resolved {
            None => resolved = Some((component, width)),
            Some((_, seen)) if seen == width => {}
            Some((earlier, seen)) => {
                return Err(mlx_gen::Error::Unsupported(format!(
                    "ideogram artifact tier: `{component}` is {width:?} but `{earlier}` is {seen:?}"
                )))
            }
        }
    }
    let (_, width) = resolved.ok_or_else(|| {
        mlx_gen::Error::Unsupported(format!(
            "ideogram artifact tier: {} carries no readable component to read a tier from",
            root.display()
        ))
    })?;
    Ok(Some(width))
}

/// The packed width of one component directory, or `None` when it is dense.
fn component_packed_width(
    dir: &std::path::Path,
    component: &str,
) -> mlx_gen::Result<Option<Quant>> {
    let fail = |detail: String| {
        mlx_gen::Error::Unsupported(format!(
            "ideogram artifact tier: {} — {detail}",
            dir.display()
        ))
    };
    let headers = mlx_gen::gen_core::weightsmeta::safetensors_path_tensor_headers(dir)
        .map_err(|error| fail(format!("no readable `{component}` weights ({error})")))?;
    if headers.is_empty() {
        return Err(fail(format!(
            "`{component}` has no weights to read a tier from"
        )));
    }
    let shapes: std::collections::HashMap<&str, &[usize]> = headers
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor.shape.as_slice()))
        .collect();
    let mut width = None;
    for tensor in &headers {
        let Some(base) = tensor.name.strip_suffix(".scales") else {
            continue;
        };
        let codes = shapes
            .get(format!("{base}.weight").as_str())
            .ok_or_else(|| fail(format!("`{base}.scales` has no `{base}.weight` codes")))?;
        let bits = packed_width_from_shapes(codes, &tensor.shape)
            .ok_or_else(|| fail(format!("`{base}` is not an exact 4- or 8-bit pack")))?;
        match width {
            None => width = Some(bits),
            Some(seen) if seen == bits => {}
            Some(seen) => {
                return Err(fail(format!(
                    "packed bases disagree on their width: `{base}` is {bits:?}, an earlier base \
                     is {seen:?}"
                )))
            }
        }
    }
    Ok(width)
}

/// [`mlx_gen::quant::packed_bits`] on header shapes alone: `scales` is `[out, in / GROUP_SIZE]` and
/// the u32 codes are `[out, in * bits / 32]`, so `bits = code_cols * 32 / (scale_cols * GROUP_SIZE)`
/// when that division is exact and lands on a shipped width.
fn packed_width_from_shapes(codes: &[usize], scales: &[usize]) -> Option<Quant> {
    let (&[out, code_cols], &[scale_rows, scale_cols]) = (codes, scales) else {
        return None;
    };
    if out != scale_rows {
        return None;
    }
    let group = usize::try_from(crate::quant::GROUP_SIZE).ok()?;
    let in_dim = scale_cols.checked_mul(group)?;
    let packed_width = code_cols.checked_mul(32)?;
    if in_dim == 0 || packed_width % in_dim != 0 {
        return None;
    }
    match packed_width / in_dim {
        4 => Some(Quant::Q4),
        8 => Some(Quant::Q8),
        _ => None,
    }
}

/// Production calibration identity TABLE of the two Ideogram routes, keyed on (route, **proven
/// artifact tier**) — sc-22732, epic sc-22723 E1/E4.
///
/// This is the table, not the binding: only `production_calibration_identity`, which proves the tier
/// against the snapshot on disk first, may turn one of these strings into a contract identity.
///
/// Six distinct cells. Nothing is preserved: this crate has never published a calibration identity,
/// so no measured cell is being retired here.
///
/// Offload policy and load shape are deliberately NOT inputs:
/// [`MemoryCalibrationIdentity::load_shape`] carries the materialization axis separately, and both
/// policies of one (route, tier) open the same resident set.
pub fn production_calibration_fingerprint(
    provider_id: &str,
    artifact_tier: Option<Quant>,
) -> Option<String> {
    let route = route_label(provider_id)?;
    let tier = calibration_tier_label(artifact_tier)?;
    Some(format!("{route}-{tier}-mlx-shared-ladder-v1"))
}

/// Per-surface static behavior identity for the weights-free declaration surface.
///
/// Keying it on the axes the declared contract shape depends on — provider, execution precision,
/// requested quant and offload policy, plus the load shape [`MemoryCalibrationIdentity`] already
/// carries — keeps the declaration walk fail-closed the same way a measured identity is: a context
/// assembled for one declared surface cannot be replayed against another.
///
/// [`MemoryProviderContract::conformance_errors`] requires lowercase kebab tokens and the provider
/// ids are snake_case, so `ideogram_4_turbo` is spelled `ideogram-4-turbo` here.
fn static_behavior_identity(provider_id: &str, spec: &LoadSpec) -> MemoryCalibrationIdentity {
    let precision = match spec.precision {
        Precision::Bf16 => "bf16",
        Precision::Fp32 => "fp32",
    };
    let quant = match spec.quantize {
        None => "dense",
        Some(Quant::Q4) => "q4",
        Some(Quant::Q8) => "q8",
        Some(Quant::Nvfp4) => "nvfp4",
    };
    let policy = match spec.offload_policy {
        OffloadPolicy::Resident => "resident",
        OffloadPolicy::Sequential => "sequential",
    };
    let route = provider_id.replace('_', "-");
    MemoryCalibrationIdentity::new(
        format!("{STATIC_BEHAVIOR_FINGERPRINT}-{route}-{precision}-{quant}-{policy}"),
        spec.load_shape,
    )
}

/// The identity a loaded Ideogram generator publishes, bound to the artifact it opens.
///
/// Five fail-closed gates, in order, with **no short-circuit ahead of the tier proof**:
///
/// 1. the id must be one this crate registers;
/// 2. the load must be a [`clean_base`] one — a user adapter stack or a PiD overlay is a resident
///    set no clean-base anchor measured. The bundled TurboTime LoRA is exempt by construction: it
///    never reaches `spec.adapters`;
/// 3. execution precision must be [`Precision::Bf16`], which is the only precision
///    `crate::model::load` / `crate::model::load_turbo` accept at all;
/// 4. the artifact tier must be readable — through `.ok().flatten()?`, so an unreadable or
///    self-inconsistent snapshot WITHHOLDS the identity instead of failing the LOAD
///    (`crate::model::load` propagates [`memory_strategy_contract`]'s error into the load); and
/// 5. `spec.quantize` must be the tier the artifact already **is**. This engine quantizes DENSE AT
///    LOAD in place — `build_residency`'s text closure calls `text.quantize(q.bits())` and
///    `load_heavy_owned` calls `heavy.quantize(q.bits())` after the dense load — so a dense snapshot
///    opened with `quantize = Some(Q4)` is a genuine runtime requantization whose peak no anchor
///    measured. `warn_if_sequential_requantize` records the sharper form of the same fact: under
///    `Sequential` + `quantize` the model is re-quantized on EVERY generate. The request knob can
///    therefore never stand in for the artifact.
fn production_calibration_identity(
    provider_id: &str,
    spec: &LoadSpec,
) -> Option<MemoryCalibrationIdentity> {
    if route_label(provider_id).is_none() || !clean_base(spec) || spec.precision != Precision::Bf16
    {
        return None;
    }
    let artifact_tier = resolved_artifact_tier(spec).ok().flatten()?;
    if artifact_tier != spec.quantize {
        return None;
    }
    production_calibration_fingerprint(provider_id, artifact_tier)
        .map(|fingerprint| MemoryCalibrationIdentity::new(fingerprint, spec.load_shape))
}

/// Architecture axes of the one DiT both Ideogram routes run (epic SC-22657, E2).
///
/// Mirrored from [`Ideogram4DitConfig::v4`] and the `crate::pipeline` stride constants, which are
/// the configuration the loaders actually build from — the Rust modules never parse a snapshot
/// `config.json`.
///
/// `latent_channels` is divided out of `in_channels` because the DiT config states the 2x2-packed
/// view: `in_channels = latent_channels * PATCH^2` (128 = 32 * 4) for the 32-channel FLUX.2
/// autoencoder Ideogram decodes through.
///
/// `vae_temporal_scale` stays `None`: that autoencoder is an image autoencoder with no temporal
/// axis, and a structurally absent axis is declared absent, never zero.
///
/// The activation width is the half width unconditionally, because
/// `crate::model::load`/`crate::model::load_turbo` reject every precision but
/// [`Precision::Bf16`], so no f32 route is served.
fn architecture_facts() -> mlx_gen::gen_core::MemoryArchitectureFacts {
    let dit = Ideogram4DitConfig::v4();
    let patch = crate::pipeline::PATCH as i32;
    mlx_gen::gen_core::MemoryArchitectureFacts {
        attention_heads: mlx_gen::architecture_facts::axis(dit.num_heads),
        head_dim: mlx_gen::architecture_facts::axis(dit.head_dim),
        transformer_blocks: mlx_gen::architecture_facts::axis(dit.num_layers),
        patch_size: mlx_gen::architecture_facts::axis(patch),
        latent_channels: mlx_gen::architecture_facts::axis(dit.in_channels / (patch * patch)),
        vae_spatial_scale: mlx_gen::architecture_facts::axis(crate::pipeline::AE_SCALE),
        vae_temporal_scale: None,
        activation_dtype_width: Some(mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH),
    }
}

fn contract_with_asset_facts(
    provider_id: &str,
    spec: &LoadSpec,
    footprint: mlx_gen::gen_core::PerComponentBytes,
    calibration: Option<MemoryCalibrationIdentity>,
) -> CoreResult<MemoryProviderContract> {
    validate_provider(provider_id)?;
    let staged = stages_residency(spec);
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            // Unified memory: the wired-residency budget is what the staged phases release, weights
            // are mmap-backed and lazy per tensor, and MLX's lazy graph needs an explicit `eval`
            // before a phase drop frees anything — which is exactly what the residency seam's
            // `materialize_before_text_drop` closure does here.
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.architecture_facts = architecture_facts();
    // Decided by the CALLER, never here: production binds the artifact-proven
    // `production_calibration_identity`, the weights-free surface binds `static_behavior_identity`,
    // and the two must never be spelled the same.
    contract.calibration = calibration;
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: phases.clone(),
        // Only what the PUBLISHED rungs consume. `DecodeTileArea`, `AttentionChunkSize` and
        // `TransformerWindowSize` are absent because rungs 2-4 are `Missing` here: a formula that
        // read a parameter no selectable strategy can set would invite calibration keyed on a value
        // that never varies.
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
        ],
    };
    contract.asset_facts.conditioning_bytes = footprint.text_encoder;
    contract.asset_facts.transformer_bytes = footprint.dit;
    contract.asset_facts.decoder_bytes = footprint.vae;
    contract.asset_facts.base_bytes = footprint
        .text_encoder
        .saturating_add(footprint.dit)
        .saturating_add(footprint.vae);
    contract.lifecycle = MemoryLifecycleCapabilities {
        // The three phases exist on both policies — `generate_impl` encodes, denoises and decodes in
        // that order either way. What the `Resident` policy lacks is the RELEASE between them.
        phases,
        synchronized_phase_release: staged,
        decode_tiling: false,
        attention_chunking: false,
        transformer_window_materialization: false,
    };
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency if staged => MemoryStrategySupport::Implemented,
            MemoryStrategy::StagedResidency => MemoryStrategySupport::Missing,
            MemoryStrategy::BoundedDecode => DECODE_SUPPORT,
            MemoryStrategy::BoundedAttention => ATTENTION_SUPPORT,
            MemoryStrategy::BoundedTransformerResidency => TRANSFORMER_WINDOW_SUPPORT,
        };
    }
    Ok(contract)
}

/// The production contract a LOADED Ideogram generator publishes.
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    validate_provider(provider_id)?;
    contract_with_asset_facts(
        provider_id,
        spec,
        crate::model::component_footprint(spec)?,
        production_calibration_identity(provider_id, spec),
    )
}

/// Declaration-equivalent contract for weights-free registry conformance: identical structure,
/// parameter domains and prerequisites, with no asset facts and no filesystem traversal.
///
/// It cannot prove an artifact tier, so it publishes the source-owned
/// [`STATIC_BEHAVIOR_FINGERPRINT`] surface identity rather than a production string.
pub fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    validate_provider(provider_id)?;
    contract_with_asset_facts(
        provider_id,
        spec,
        mlx_gen::gen_core::PerComponentBytes::default(),
        Some(static_behavior_identity(provider_id, spec)),
    )
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<Quant>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision,
            quant,
            component_precision_floors: &[],
        }),
        None,
    )
}

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    safety_check(contract, spec.precision, spec.quantize, context)
}

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || contract
            .capability(strategy)
            .is_none_or(|capability| capability.support != MemoryStrategySupport::Implemented)
    {
        return Ok(Vec::new());
    }
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: mlx_gen::gen_core::MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            // Read off the contract this fixture is built for: rung 1 IS the phase schedule here.
            has_phases: contract.engages(strategy, MemoryStrategy::StagedResidency),
            overlay: None,
        },
    )?;
    Ok(vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(context)])
}

/// The refusal a bounded-decode selection earns on this provider.
///
/// Rung 2 is declared `Missing` ([`DECODE_SUPPORT`]) because the Ideogram pipeline decodes in one
/// shot (`crate::pipeline` `self.vae.decode(&latent)`) and consumes no tiling config. The shared
/// `validate_selection` already refuses a rung the contract declares `Missing`; this closes the
/// remaining path, where a harness hand-builds a `GenerationMemory` and calls `generate` without
/// crossing admission.
fn refuse_decode(provider_id: &str, edge: u32, overlap: u32) -> CoreError {
    CoreError::Unsupported(format!(
        "{provider_id}: bounded decode is not implemented (rung 2 is declared Missing; requested \
         edge {edge} overlap {overlap})"
    ))
}

fn begin_request_with_cleanup(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<Quant>,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } =
        safety_check(contract, precision, quant, context)
    {
        return Err(CoreError::Unsupported(reason));
    }
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        Ideogram4DitConfig::v4().num_layers.max(0) as usize,
        // Rung 2 is `Missing` (see [`DECODE_SUPPORT`]), so no tile geometry is ever in domain.
        //
        // Written as a brace-free closure over a named constructor, the shape
        // `mlx-gen-kolors/src/memory_strategy.rs:1414` already ships. `scripts/check-workspace.py`'s
        // `check_pid_decode_route_adoption` exempts a rung-2-`Missing` provider only when every
        // decode validator it installs is one unconditional `Err(...)`, and its
        // `_is_single_err_expression` reader rejects both a `{ … }` body and a macro inside the
        // `Err` argument — a macro body is opaque to it, so `format!` there arms the gate.
        move |_use_pid, edge, overlap| Err(refuse_decode(provider_id, edge, overlap)),
    )?;
    config.load_shape = contract.load_shape;
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    )))
}

pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    begin_request_with_cleanup(
        provider_id,
        contract,
        spec.precision,
        spec.quantize,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub(crate) fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<Quant>,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    begin_request_with_cleanup(
        provider_id,
        contract,
        precision,
        quant,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::Device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{AdapterKind, AdapterSpec, LoadShape};

    const PROVIDERS: [&str; 2] = [crate::MODEL_ID, crate::MODEL_ID_TURBO];

    /// A structurally valid safetensors file with a zeroed body.
    ///
    /// `safetensors_path_tensor_headers` reads only the header, but it validates it in full: every
    /// tensor's `data_offsets` span must equal its shape product times its dtype width, the spans
    /// must be contiguous from the header end, and they must fit the file. A header-only stub is
    /// therefore not enough — the payload has to be allocated even though it is never read.
    fn write_safetensors(path: &std::path::Path, tensors: &[(&str, &str, &[usize])]) {
        let width = |dtype: &str| match dtype {
            "BF16" => 2_usize,
            "U32" | "F32" => 4,
            other => panic!("unhandled fixture dtype {other}"),
        };
        let mut offset = 0_usize;
        let mut header = serde_json::Map::new();
        for (name, dtype, shape) in tensors {
            let bytes = shape.iter().product::<usize>() * width(dtype);
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let encoded = serde_json::to_vec(&header).unwrap();
        let mut bytes = (encoded.len() as u64).to_le_bytes().to_vec();
        bytes.extend(encoded);
        bytes.resize(bytes.len() + offset, 0);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    /// One heavy component directory at `tier`, in the shape `crate::convert` writes: a packed
    /// tier carries the `{base}.weight` (U32 codes) / `.scales` / `.biases` triple and NO
    /// quantization marker, because the converter writes none.
    fn write_component(dir: &std::path::Path, tier: &str, base: &str) {
        let scale_cols = 1;
        let in_dim = scale_cols * crate::quant::GROUP_SIZE as usize;
        let tensors: Vec<(String, &str, Vec<usize>)> = match tier {
            "bf16" => vec![(format!("{base}.weight"), "BF16", vec![2, in_dim])],
            "q4" | "q8" => {
                let bits = if tier == "q4" { 4 } else { 8 };
                vec![
                    (format!("{base}.weight"), "U32", vec![2, in_dim * bits / 32]),
                    (format!("{base}.scales"), "BF16", vec![2, scale_cols]),
                    (format!("{base}.biases"), "BF16", vec![2, scale_cols]),
                ]
            }
            other => panic!("unknown tier {other}"),
        };
        let borrowed: Vec<(&str, &str, &[usize])> = tensors
            .iter()
            .map(|(name, dtype, shape)| (name.as_str(), *dtype, shape.as_slice()))
            .collect();
        write_safetensors(&dir.join("model.safetensors"), &borrowed);
    }

    /// A converted Ideogram snapshot for `provider` at `tier`: exactly the component dirs the
    /// loaders open. No `config.json` marker anywhere — matching the proof this crate implements.
    fn fixture(provider: &str, tier: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("snapshot");
        write_component(&root.join("transformer"), tier, "blocks.0.attn.qkv");
        write_component(
            &root.join("text_encoder"),
            tier,
            "layers.0.self_attn.q_proj",
        );
        if provider == crate::MODEL_ID {
            write_component(
                &root.join("unconditional_transformer"),
                tier,
                "blocks.0.attn.qkv",
            );
        }
        write_safetensors(
            &root.join("vae/model.safetensors"),
            &[("decoder.conv_out.weight", "BF16", &[2, 2])],
        );
        std::fs::create_dir_all(root.join("tokenizer")).unwrap();
        std::fs::write(root.join("tokenizer/tokenizer.json"), b"{}").unwrap();
        (temp, root)
    }

    fn spec_for(root: &std::path::Path, quant: Option<Quant>) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir(root.to_path_buf()));
        spec.quantize = quant;
        spec
    }

    /// sc-22732 (epic sc-22723, E1 measurable / E4 production loader): every (route, tier) cell the
    /// worker can load publishes its own production calibration identity, under the worker's own
    /// `Resident + Eager` load shape and under the staged `Sequential + Deferred` one, so a memory
    /// anchor has something to bind. All six cells published NOTHING before this story — the crate
    /// had no contract at all.
    ///
    /// *Mutations this kills:* dropping the `{tier}` token from
    /// [`production_calibration_fingerprint`] (the six strings collapse to two and `published`
    /// rejects the repeat); dropping the route token (they collapse to three); keying the string on
    /// the offload policy or load shape (the two specs per cell would disagree); publishing the
    /// weights-free registry string in production (`assert_ne!` against `weights_free`); and
    /// deleting the `artifact_tier != spec.quantize` refusal, which the wrong-quant loop catches —
    /// this engine quantizes dense at load, so a snapshot opened at a tier it is not is a runtime
    /// requantization no anchor measured.
    #[test]
    fn every_route_and_tier_publishes_its_own_production_identity() {
        // The six strings verbatim. They are published downstream as the anchor plan's cell keys,
        // so a rename is a breaking change and has to be made deliberately here.
        const CELLS: [(&str, &str, &str); 6] = [
            (
                crate::MODEL_ID,
                "bf16",
                "ideogram-4-bf16-mlx-shared-ladder-v1",
            ),
            (crate::MODEL_ID, "q4", "ideogram-4-q4-mlx-shared-ladder-v1"),
            (crate::MODEL_ID, "q8", "ideogram-4-q8-mlx-shared-ladder-v1"),
            (
                crate::MODEL_ID_TURBO,
                "bf16",
                "ideogram-4-turbo-bf16-mlx-shared-ladder-v1",
            ),
            (
                crate::MODEL_ID_TURBO,
                "q4",
                "ideogram-4-turbo-q4-mlx-shared-ladder-v1",
            ),
            (
                crate::MODEL_ID_TURBO,
                "q8",
                "ideogram-4-turbo-q8-mlx-shared-ladder-v1",
            ),
        ];

        let mut published = std::collections::BTreeSet::new();
        for provider in PROVIDERS {
            for (tier, quant) in [
                ("bf16", None),
                ("q4", Some(Quant::Q4)),
                ("q8", Some(Quant::Q8)),
            ] {
                let expected = production_calibration_fingerprint(provider, quant).unwrap();
                assert!(expected.contains(tier), "{provider} {tier}: {expected}");
                assert_eq!(
                    expected,
                    CELLS
                        .iter()
                        .find(|(id, label, _)| *id == provider && *label == tier)
                        .unwrap()
                        .2,
                    "{provider} {tier} renamed its published cell key"
                );
                let (_temp, root) = fixture(provider, tier);
                for (offload, load_shape) in [
                    (OffloadPolicy::Resident, LoadShape::EagerMaterialization),
                    (
                        OffloadPolicy::Sequential,
                        LoadShape::DeferredMaterialization,
                    ),
                ] {
                    let spec = spec_for(&root, quant)
                        .with_offload_policy(offload)
                        .with_load_shape(load_shape);
                    let weights_free = weights_free_memory_strategy_contract(provider, &spec)
                        .unwrap()
                        .calibration
                        .unwrap()
                        .fingerprint;
                    assert_ne!(expected, weights_free, "{provider} {tier}");
                    let contract = memory_strategy_contract(provider, &spec).unwrap();
                    let identity = contract.calibration.as_ref().unwrap_or_else(|| {
                        panic!("{provider} {tier} {load_shape:?} publishes no identity")
                    });
                    assert_eq!(identity.fingerprint, expected, "{provider} {tier}");
                    assert_eq!(identity.load_shape, load_shape);
                    assert!(
                        contract.conformance_errors().is_empty(),
                        "{provider} {tier}: {:?}",
                        contract.conformance_errors()
                    );
                }

                // The request knob never outranks the artifact: every tier the snapshot is NOT.
                for wrong in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                    if wrong == quant {
                        continue;
                    }
                    assert!(
                        memory_strategy_contract(provider, &spec_for(&root, wrong))
                            .unwrap()
                            .calibration
                            .is_none(),
                        "{provider} {tier} must publish nothing for a {wrong:?} load quant"
                    );
                }
                assert!(
                    published.insert(expected.clone()),
                    "{provider} {tier} repeats another cell's identity: {expected}"
                );
            }
        }
        // Six distinct strings: two routes x three tiers. A route-only or tier-only key collapses
        // this.
        assert_eq!(published.len(), PROVIDERS.len() * 3);
    }

    /// The weights-free identity lives in its own namespace and is distinct from EVERY production
    /// string, on every registry surface.
    ///
    /// *Mutation this kills:* handing `production_calibration_identity(..)` to
    /// [`weights_free_memory_strategy_contract`] — a registry-conformance context would then satisfy
    /// the calibration handshake of a real load.
    #[test]
    fn the_weights_free_identity_is_never_a_production_string() {
        let production: std::collections::BTreeSet<String> = PROVIDERS
            .into_iter()
            .flat_map(|provider| {
                [None, Some(Quant::Q4), Some(Quant::Q8)]
                    .into_iter()
                    .map(move |quant| production_calibration_fingerprint(provider, quant).unwrap())
            })
            .collect();
        assert_eq!(production.len(), 6);

        let mut seen = std::collections::BTreeSet::new();
        for surface in mlx_gen::gen_core::mlx_memory_contract_surface_specs() {
            for provider in PROVIDERS {
                let contract =
                    weights_free_memory_strategy_contract(provider, &surface.spec).unwrap();
                assert!(
                    contract.conformance_errors().is_empty(),
                    "{provider} {}: {:?}",
                    surface.selector.id(),
                    contract.conformance_errors()
                );
                let fingerprint = contract.calibration.as_ref().unwrap().fingerprint.clone();
                assert!(
                    fingerprint.starts_with(STATIC_BEHAVIOR_FINGERPRINT),
                    "{fingerprint}"
                );
                assert!(!production.contains(&fingerprint), "{fingerprint}");
                assert!(
                    seen.insert(format!("{fingerprint}:{:?}", surface.spec.load_shape)),
                    "static behavior identity {fingerprint} is reused across surfaces"
                );
            }
        }
        assert_eq!(seen.len(), 2 * 12);
    }

    /// An overlay is a different resident set than the clean-base cell an anchor prices, so it
    /// publishes nothing — while the turbo route's BUNDLED TurboTime LoRA, which never travels
    /// through `spec.adapters`, leaves the turbo cells intact.
    ///
    /// *Mutation this kills:* dropping `spec.adapters.is_empty()` (or `spec.pid.is_none()`) from
    /// [`clean_base`].
    #[test]
    fn an_overlay_withholds_the_identity_but_the_bundled_turbo_lora_does_not() {
        for provider in PROVIDERS {
            let (_temp, root) = fixture(provider, "q4");
            let clean = spec_for(&root, Some(Quant::Q4));
            // The turbo snapshot carries its bundled LoRA on disk, not on the spec: the turbo cell
            // publishes exactly like the base cell.
            assert_eq!(
                memory_strategy_contract(provider, &clean)
                    .unwrap()
                    .calibration
                    .unwrap()
                    .fingerprint,
                production_calibration_fingerprint(provider, Some(Quant::Q4)).unwrap()
            );

            let mut adapted = clean.clone();
            adapted.adapters.push(AdapterSpec::new(
                root.join("user_lora.safetensors"),
                1.0,
                AdapterKind::Lora,
            ));
            assert!(
                memory_strategy_contract(provider, &adapted)
                    .unwrap()
                    .calibration
                    .is_none(),
                "{provider}: a user adapter stack must publish no identity"
            );

            let mut pid = clean;
            pid.pid = Some(mlx_gen::PidWeights {
                checkpoint: WeightsSource::File(root.join("pid.safetensors")),
                gemma: WeightsSource::Dir(root.join("gemma")),
            });
            assert!(
                memory_strategy_contract(provider, &pid)
                    .unwrap()
                    .calibration
                    .is_none(),
                "{provider}: a PiD overlay must publish no identity"
            );
        }
    }

    /// A snapshot whose components disagree about their packed width WITHHOLDS the identity and
    /// does NOT fail the load — `crate::model::load` propagates this builder's error, so an
    /// escaping error would turn a loadable snapshot into a refused one.
    ///
    /// *Mutation this kills:* `resolved_artifact_tier(spec).ok().flatten()?` ->
    /// `.unwrap().flatten()?` (the error escapes and `memory_strategy_contract` returns `Err`).
    #[test]
    fn a_disagreeing_snapshot_withholds_the_identity_without_failing_the_load() {
        for provider in PROVIDERS {
            let (_temp, root) = fixture(provider, "q4");
            // The text encoder is repacked at q8 while the DiT stays q4: the two proofs disagree.
            write_component(
                &root.join("text_encoder"),
                "q8",
                "layers.0.self_attn.q_proj",
            );
            let spec = spec_for(&root, Some(Quant::Q4));
            assert!(
                resolved_artifact_tier(&spec).is_err(),
                "{provider}: the tier probe keeps its hard error"
            );
            let contract = memory_strategy_contract(provider, &spec)
                .unwrap_or_else(|error| panic!("{provider} must still load: {error}"));
            assert!(contract.calibration.is_none(), "{provider}");
            assert!(contract.conformance_errors().is_empty(), "{provider}");
        }
    }

    /// Rung 1 is declared exactly where [`stages_residency`] says the engine executes it, and rungs
    /// 2-4 are declared `Missing` everywhere because this crate implements none of them.
    #[test]
    fn the_published_ladder_is_the_ladder_the_engine_executes() {
        for provider in PROVIDERS {
            let (_temp, root) = fixture(provider, "bf16");
            for (policy, staged) in [
                (OffloadPolicy::Resident, false),
                (OffloadPolicy::Sequential, true),
            ] {
                let spec = spec_for(&root, None).with_offload_policy(policy);
                let contract = memory_strategy_contract(provider, &spec).unwrap();
                assert_eq!(
                    contract
                        .capability(MemoryStrategy::Resident)
                        .unwrap()
                        .support,
                    MemoryStrategySupport::Implemented
                );
                assert_eq!(
                    contract
                        .capability(MemoryStrategy::StagedResidency)
                        .unwrap()
                        .support
                        == MemoryStrategySupport::Implemented,
                    staged,
                    "{provider} {policy:?}"
                );
                assert_eq!(contract.lifecycle.synchronized_phase_release, staged);
                for missing in [
                    MemoryStrategy::BoundedDecode,
                    MemoryStrategy::BoundedAttention,
                    MemoryStrategy::BoundedTransformerResidency,
                ] {
                    assert_eq!(
                        contract.capability(missing).unwrap().support,
                        MemoryStrategySupport::Missing,
                        "{provider} {policy:?} {missing:?}"
                    );
                }
                assert!(!contract.lifecycle.decode_tiling);
                assert!(!contract.lifecycle.attention_chunking);
                assert!(!contract.lifecycle.transformer_window_materialization);
                assert!(contract.conformance_errors().is_empty());
            }
        }
    }

    #[test]
    fn an_unregistered_provider_is_never_handed_an_ideogram_ladder() {
        let (_temp, root) = fixture(crate::MODEL_ID, "bf16");
        let spec = spec_for(&root, None);
        for provider in PROVIDERS {
            assert!(memory_strategy_contract(provider, &spec).is_ok());
            assert!(weights_free_memory_strategy_contract(provider, &spec).is_ok());
        }
        for provider in ["ideogram_5", "flux2", "", "Ideogram_4"] {
            assert!(
                memory_strategy_contract(provider, &spec).is_err(),
                "{provider}"
            );
            assert!(
                weights_free_memory_strategy_contract(provider, &spec).is_err(),
                "{provider}"
            );
            assert!(production_calibration_fingerprint(provider, None).is_none());
        }
    }
}
