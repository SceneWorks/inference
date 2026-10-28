//! The distilled FLUX 3 MLX request-memory contract (the shared ladder, sc-26234).
//!
//! | rung | strategy | `flux3_klein` |
//! |---|---|---|
//! | 0 | [`MemoryStrategy::Resident`] | implemented |
//! | 1 | [`MemoryStrategy::StagedResidency`] | **implemented** — the Qwen3-VL tower is loaded, encodes, and is dropped before the DiT + autoencoder load |
//! | 2 | [`MemoryStrategy::BoundedDecode`] | missing — the autoencoder decodes whole (the tiled decode is sc-26241's) |
//! | 3 | [`MemoryStrategy::BoundedAttention`] | missing |
//! | 4 | [`MemoryStrategy::BoundedTransformerResidency`] | missing |
//!
//! # What each declaration is true of
//!
//! * **Rung 1.** [`crate::model::Flux3Klein`] holds a request-scoped [`mlx_gen::Residency`]: a
//!   request whose admitted `GenerationMemory::stage_residency` is set (written by the request
//!   scope this module opens), or any request of a generator loaded `OffloadPolicy::Sequential`,
//!   evicts the warm pair, loads the text encoder, encodes (the conditioning is evaluated inside
//!   the encode), drops the encoder, then loads the DiT and the autoencoder. The staged weight
//!   floor is therefore `max(tower, DiT + autoencoder)`, never the sum.
//! * **Rungs 2–4 are `Missing`, not "not applicable".** The route has an autoencoder, attention
//!   and a block stack, so all three describe real work; none is wired in S1. See
//!   [`DECODE_NOT_ENABLED`], [`ATTENTION_NOT_ENABLED`] and [`TRANSFORMER_WINDOW_NOT_ENABLED`].
//!
//! # Nothing here is a measurement
//!
//! The contract carries the route shape and, for a materialized package, the resident bytes of
//! the components the loaders build — priced from safetensors **headers** at the width each
//! loader materializes them (the DiT at the compute dtype, the Qwen3-VL language tower at bf16,
//! the autoencoder at f32). No activation number is published and this crate registers no
//! [`mlx_gen::gen_core::ActivationMemoryRegistration`]: that carrier is measurement-only, and the
//! epic's terminal campaign owns the anchors.

use std::collections::BTreeSet;
use std::path::Path;

use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};
use mlx_gen::gen_core::flux3::artifacts::NativeDitStorage;
use mlx_gen::gen_core::flux3::quantization::DitFormat;
use mlx_gen::gen_core::flux3::{Flux3Config, GenerationResources};
use mlx_gen::gen_core::weightsmeta::{safetensors_path_tensor_headers, SafetensorsTensorHeader};
use mlx_gen::gen_core::{
    self, Error as CoreError, LoadShape, MemoryArchitectureFacts, MemoryAssetFacts,
    MemoryBackendRealization, MemoryBehaviorFixture, MemoryBehaviorRoute,
    MemoryCalibrationIdentity, MemoryContractSurfaceSpec, MemoryContractSurfaceTier,
    MemoryFormulaKind, MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategySupport, Precision, Quant, ResidentRequestMemory,
    Result as CoreResult,
};
use mlx_gen::LoadSpec;

use crate::model::MODEL_ID;

/// Content fingerprint of the contract. Load shape is a separate typed axis on
/// [`MemoryCalibrationIdentity`], so it is deliberately absent from the string.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "flux3-klein-mlx-derived-2026-10-15-v1";

/// Why rung 2 is declared `Missing` (applicable, not enabled): the autoencoder's decoder runs
/// whole-image GroupNorm resnets; a bounded decode is the normalization-correct tiled tail the
/// FLUX.2 crate has (sc-19753), not yet ported to this autoencoder (sc-26241).
pub const DECODE_NOT_ENABLED: &str =
    "the FLUX 3 autoencoder decodes the whole image; the normalization-correct tiled decode tail \
     is not ported to it yet (sc-26241)";

/// Why rung 3 is declared `Missing`: the shared MLX kernel's saving is a per-family real-weight
/// measurement this story's lane cannot take, and nothing is wired through `Flux3Dit::forward`.
pub const ATTENTION_NOT_ENABLED: &str =
    "bounded attention is not wired through Flux3Dit::forward: the shared MLX kernel's saving is \
     a per-family real-weight measurement";

/// Why rung 4 is declared `Missing`: the loader materializes and evaluates the whole DiT at load;
/// there is no deferred-materialization load shape and no block-window inference loader.
pub const TRANSFORMER_WINDOW_NOT_ENABLED: &str =
    "the FLUX 3 loader materializes and evaluates the whole DiT at load; there is no \
     deferred-materialization load shape or block-window inference loader";

/// Refuse a deferred-materialization load shape: the loader has none and would otherwise accept
/// the spec and silently load eagerly.
pub(crate) fn refuse_deferred_load_shape(id: &str, spec: &LoadSpec) -> CoreResult<()> {
    if spec.load_shape != LoadShape::EagerMaterialization {
        return Err(CoreError::Unsupported(format!(
            "{id}: LoadSpec::load_shape {:?} is not supported; the FLUX 3 loader has no \
             deferred/block-window load path, use EagerMaterialization",
            spec.load_shape
        )));
    }
    Ok(())
}

/// Refuse the spec axes that would make a contract describe a load that cannot happen.
fn refuse_unloadable_spec(spec: &LoadSpec) -> CoreResult<()> {
    refuse_deferred_load_shape(MODEL_ID, spec)?;
    match spec.quantize {
        None | Some(Quant::Q4 | Quant::Q8) => Ok(()),
        Some(other) => Err(CoreError::Unsupported(format!(
            "{MODEL_ID}: {} has no MLX FLUX3 kernel; the native MLX tiers are affine q4/q8 \
             converted from the original dense package",
            other.key()
        ))),
    }
}

/// The calibration identity of one tier: a packed MLX affine tier never inherits the dense
/// identity. Weights-free surfaces know only the bit width; a loaded package adds its group.
pub fn tier_fingerprint(base: &str, quant: Option<Quant>, group: Option<u32>) -> String {
    match (quant, group) {
        (None, _) => base.to_owned(),
        (Some(q), None) => format!("{base}-mlx-affine-{}", q.key()),
        (Some(q), Some(g)) => format!("{base}-mlx-affine-{}-g{g}", q.key()),
    }
}

/// Bytes per element of the DiT's compute dtype.
const fn compute_width(precision: Precision) -> u32 {
    match precision {
        Precision::Bf16 => mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH,
        Precision::Fp32 => mlx_gen::architecture_facts::FLOAT32_ACTIVATION_WIDTH,
    }
}

/// Architecture axes of the DiT. With a parsed `config.json` these are the package's own values;
/// weights-free, the one axis the route fixes at compile time is the 16× token size (the request
/// grid every size is validated against). Heads, width and depth live only in the file and are
/// left undeclared rather than guessed.
fn architecture_facts(
    config: Option<&Flux3Config>,
    precision: Precision,
) -> MemoryArchitectureFacts {
    use mlx_gen::architecture_facts as af;
    let dit = config.map(|c| &c.dit);
    MemoryArchitectureFacts {
        attention_heads: dit.and_then(|d| af::axis(d.heads)),
        head_dim: dit.and_then(|d| af::head_dim(d.hidden, d.heads)),
        transformer_blocks: dit.and_then(|d| af::axis(d.depth + d.depth_single)),
        // The DiT's token is the 2×2 patch of the 8× VAE latent.
        patch_size: af::axis(2u32),
        latent_channels: dit.and_then(|d| af::axis(d.image_in_channels)),
        vae_spatial_scale: af::axis(8u32),
        vae_temporal_scale: None,
        activation_dtype_width: Some(compute_width(precision)),
    }
}

/// The declaration every surface shares: route shape, calibration identity and strategy
/// classification, with no asset facts.
fn base_contract(precision: Precision, config: Option<&Flux3Config>) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        MODEL_ID,
        MemoryBackendRealization::MlxMetal {
            // Unified memory: a released phase returns to the wired-residency budget.
            bounded_wired_residency: true,
            // Every loader casts each tensor and evaluates it before returning, so the resident
            // arrays are materialized copies, never the mapped file.
            lazy_or_mmap_materialization: false,
            // The conditioning is evaluated inside the encode and every Euler step ends in an
            // `eval`, so a phase drop frees what it names.
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = LoadShape::EagerMaterialization;
    contract.architecture_facts = architecture_facts(config, precision);
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        MEMORY_CALIBRATION_FINGERPRINT,
        LoadShape::EagerMaterialization,
    ));
    // A `Resident` selection leaves the request memory block absent, which keeps exactly the
    // loaded policy: warm for a `Resident` load, staged for a `Sequential` one.
    contract.resident_request_memory = ResidentRequestMemory::PreserveLoadDefaults;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: false,
        attention_chunking: false,
        transformer_window_materialization: false,
    };
    contract.phase_facts = Some(gen_core::MemoryPhaseFacts::staged(
        gen_core::StagedWeightSchedule::ThreeStage,
    ));
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        // The attention transient moves with the pixel count (image tokens) and with the text
        // window (padded to a multiple of 80 up to 4096, so it IS a request variable here);
        // `count` renders one image after another, so no batch axis.
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::ConditioningTokenCount,
        ],
    };
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident | MemoryStrategy::StagedResidency => {
                MemoryStrategySupport::Implemented
            }
            MemoryStrategy::BoundedDecode
            | MemoryStrategy::BoundedAttention
            | MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
        };
    }
    contract
}

/// Declaration-equivalent contract with no asset facts — the registry conformance surface.
pub fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    if provider_id != MODEL_ID && provider_id != gen_core::flux3::BASE_MODEL_ID {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: not the FLUX 3 MLX memory provider"
        )));
    }
    refuse_unloadable_spec(spec)?;
    if !spec.adapters.is_empty() {
        return Err(CoreError::Unsupported(
            "flux3: adapters require the real package config for identity/rank admission".into(),
        ));
    }

    let mut contract = base_contract(spec.precision, None);
    contract.provider_id = provider_id.into();
    if provider_id == gen_core::flux3::BASE_MODEL_ID {
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            "flux3-klein-base-mlx-cfg-source-v1",
            LoadShape::EagerMaterialization,
        ));
        // CFG's [unconditional, conditional] pack is a FIXED x2 coefficient of the base route
        // (always two rows, any guidance), carried by its separate calibration identity. It is
        // not a request variable: `count` still renders sequentially, so no BatchCount axis.
    }
    if let Some(calibration) = &mut contract.calibration {
        calibration.fingerprint = tier_fingerprint(&calibration.fingerprint, spec.quantize, None);
    }
    Ok(contract)
}

/// The executable contract for a real load: the declaration plus the resident bytes of what the
/// loaders build, read from the resources `spec` names.
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    if provider_id != MODEL_ID && provider_id != gen_core::flux3::BASE_MODEL_ID {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: not the FLUX 3 MLX memory provider"
        )));
    }
    refuse_unloadable_spec(spec)?;
    let package_dir = match &spec.weights {
        gen_core::WeightsSource::Dir(dir) => dir.clone(),
        gen_core::WeightsSource::File(file) => {
            return Err(CoreError::Msg(format!(
                "{provider_id}: the weights resource must be the package directory, not {}",
                file.display()
            )))
        }
    };
    // The same header authority as the loader, before any component is priced.
    let (config, _) =
        crate::model::resolve_package(&package_dir, provider_id, spec.quantize, spec.precision)?;
    let resources = GenerationResources::from_spec(spec, &config, provider_id)?;
    generation_contract(spec, &resources, &config)
}

/// The contract for already-resolved resources (what [`crate::model::load`] holds).
pub(crate) fn generation_contract(
    spec: &LoadSpec,
    resources: &GenerationResources,
    config: &Flux3Config,
) -> CoreResult<MemoryProviderContract> {
    refuse_unloadable_spec(spec)?;
    let precision = spec.precision;
    let mut contract = base_contract(precision, Some(config));
    if !config.dit.flow_map {
        contract.provider_id = gen_core::flux3::BASE_MODEL_ID.into();
        // Base is a distinct CFG graph (no target-time embedder), never a distilled calibration.
        contract.calibration = Some(MemoryCalibrationIdentity::new(
            "flux3-klein-base-mlx-cfg-source-v1",
            LoadShape::EagerMaterialization,
        ));
        // CFG's [unconditional, conditional] pack is a FIXED x2 coefficient of the base route
        // (always two rows, any guidance), carried by its separate calibration identity. It is
        // not a request variable: `count` still renders sequentially, so no BatchCount axis.
    }
    for adapter in &spec.adapters {
        gen_core::flux3::adapters::inspect(adapter, &config.dit)?;
    }

    let storage = NativeDitStorage::from_package(&resources.package_dir)?;
    storage.validate_selection(spec.quantize, precision)?;
    if let Some(calibration) = &mut contract.calibration {
        let group = match storage {
            NativeDitStorage::MlxAffine { group_size, .. } => Some(group_size),
            _ => None,
        };
        calibration.fingerprint = tier_fingerprint(&calibration.fingerprint, spec.quantize, group);
    }
    let transformer = native_dit_bytes(&resources.dit_weights(), precision, storage, &config.dit)?;
    let conditioning = text_encoder_bytes(&resources.text_encoder_dir)?;
    let decoder = autoencoder_bytes(&resources.autoencoder_file)?;
    let base_bytes = conditioning
        .checked_add(transformer)
        .and_then(|b| b.checked_add(decoder))
        .ok_or_else(|| CoreError::Msg(format!("{MODEL_ID}: base component byte sum overflow")))?;
    contract.asset_facts = MemoryAssetFacts {
        base_bytes,
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: decoder,
        overlay_bytes: 0,
    };
    if let Some(dir) = gen_core::flux3::pup::UpsamplePackage::component_dir(spec)? {
        gen_core::flux3::pup::UpsamplePackage::check(&dir)?;
        let llm_spec = gen_core::core_llm::LoadSpec::dense(dir.to_string_lossy().into_owned());
        let load_bound = mlx_llm::load_memory_required_bytes(&llm_spec).map_err(|e| {
            CoreError::Unsupported(format!(
                "{MODEL_ID}: cannot account for prompt_upsampler at {}: {e}",
                dir.display()
            ))
        })?;
        let (bytes, residency) = upsampler_charge(load_bound, spec.offload_policy)?;
        account_upsampler(&mut contract, bytes, residency)?;
    }
    let mut adapter_bytes = 0u64;
    for adapter in &spec.adapters {
        for header in safetensors_path_tensor_headers(&adapter.path)? {
            // Source F32 plus one compute-dtype preparation: a source-derived upper bound,
            // not a measured peak. Each admitted tensor is charged once per loaded adapter.
            adapter_bytes = adapter_bytes
                .checked_add(header.materialized_bytes(8)?)
                .ok_or_else(|| CoreError::Unsupported("flux3: adapter byte sum overflow".into()))?;
        }
    }
    if adapter_bytes > 0 {
        contract.asset_facts.overlay_bytes = contract
            .asset_facts
            .overlay_bytes
            .checked_add(adapter_bytes)
            .ok_or_else(|| CoreError::Unsupported("flux3: overlay byte sum overflow".into()))?;
        let component = gen_core::MemoryResidentComponent {
            id: "lora_adapters".into(),
            kind: gen_core::MemoryComponentKind::AdapterStack,
            resident_bytes: adapter_bytes,
            bounded_by: None,
            residency: gen_core::MemoryComponentResidency::WholeRender,
        };
        match &mut contract.formula {
            MemoryFormulaKind::ComponentPhaseEnvelope {
                resident_components,
                ..
            } => resident_components.push(component),
            _ => {
                contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
                    phases: contract.lifecycle.phases.clone(),
                    variables: vec![
                        MemoryFormulaVariable::AssetBytes,
                        MemoryFormulaVariable::PixelCount,
                        MemoryFormulaVariable::ConditioningTokenCount,
                        MemoryFormulaVariable::OverlayBytes,
                    ],
                    resident_components: vec![component],
                }
            }
        }
    }
    Ok(contract)
}

/// What the PUP slot actually holds per load policy. `Sequential` always takes the exclusive
/// path: `load_bounded` (prefix cache 0), fenced and dropped before the text encoder loads, so
/// the bare load bound is consumed in conditioning with nothing retained. `Resident` keeps the
/// default `load` warm, whose cross-request prefix cache reserves its default budget on top of
/// the weights; the header-only load bound does not include that budget, so it is added here.
pub(crate) fn upsampler_charge(
    load_bound: u64,
    policy: gen_core::OffloadPolicy,
) -> CoreResult<(u64, gen_core::MemoryComponentResidency)> {
    Ok(match policy {
        gen_core::OffloadPolicy::Resident => (
            load_bound
                .checked_add(gen_core::core_llm::defaults::MLX.prefix_cache_bytes)
                .ok_or_else(|| CoreError::Unsupported("flux3: prompt_upsampler byte overflow".into()))?,
            gen_core::MemoryComponentResidency::WholeRender,
        ),
        gen_core::OffloadPolicy::Sequential => (
            load_bound,
            gen_core::MemoryComponentResidency::PrecomputedThenEvicted {
                precomputed_in: gen_core::MemoryPhase::Conditioning,
                retained_bytes: 0,
                evidence: "mlx_gen_flux3::model::tests::actual_generator_lazy_pup_bypass_and_exclusive_ownership_are_honest; \
                           gen_core::flux3::phase_resource::tests::exclusive_work_fences_and_really_drops_before_the_next_phase"
                    .into(),
            },
        ),
    })
}

/// Charge the optional VLM exactly once, separate from the base encoder. It does NOT estimate
/// upsampling activations or its KV cache: the LLM admits those at request time.
fn account_upsampler(
    contract: &mut MemoryProviderContract,
    bytes: u64,
    residency: gen_core::MemoryComponentResidency,
) -> CoreResult<()> {
    if bytes == 0 {
        return Err(CoreError::Unsupported(
            "flux3: prompt_upsampler has no accountable weights; provision a complete package"
                .into(),
        ));
    }
    contract.asset_facts.overlay_bytes = bytes;
    contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::OverlayBytes,
        ],
        resident_components: vec![gen_core::MemoryResidentComponent {
            id: gen_core::flux3::pup::UPSAMPLER_COMPONENT.into(),
            kind: gen_core::MemoryComponentKind::PromptUpsampler,
            resident_bytes: bytes,
            bounded_by: None,
            residency,
        }],
    };
    Ok(())
}

/// Resident bytes of the DiT as [`crate::model::load_dit`] holds it, priced from the file's
/// headers by [`dense_dit_projection`].
fn dit_bytes(weights: &Path, precision: Precision) -> CoreResult<u64> {
    projected_safetensors_bytes(weights, dense_dit_projection(precision))
}

/// What [`crate::model::load_dit`] retains of one dense DiT tensor: every tensor of rank two or
/// more cast to the compute dtype; vectors (the q/k norm scales) at f32 — `Loader::f32`
/// materializes them at f32 whatever the file stores. The released packages ship them bf16, so
/// pricing them as stored would undercharge `2 × (3 × depth + depth_single) × head_dim × 2`
/// bytes (22,016 on both released checkpoints). Twin of the Candle pricing.
fn dense_dit_projection(
    precision: Precision,
) -> impl Fn(&SafetensorsTensorHeader) -> ResidentProjection {
    let cast = match precision {
        Precision::Bf16 => ResidentProjection::Bfloat16,
        Precision::Fp32 => ResidentProjection::Float32,
    };
    move |header| {
        if header.shape.len() >= 2 {
            cast
        } else {
            ResidentProjection::Float32
        }
    }
}

/// What `load_dit_with_storage` retains of one tensor of a packed MLX affine DiT: packed `u32`
/// codes at their exact stored bytes, everything else (their scales/biases, every other matrix,
/// the f32 norm scales) exactly as [`dense_dit_projection`].
fn native_dit_projection(
    precision: Precision,
    targets: &BTreeSet<String>,
) -> impl Fn(&SafetensorsTensorHeader) -> ResidentProjection + '_ {
    let dense = dense_dit_projection(precision);
    move |header| {
        if targets.contains(&header.name) {
            ResidentProjection::Stored
        } else {
            dense(header)
        }
    }
}

/// Resident DiT bytes for the declared storage, as `load_dit_with_storage` holds it, priced by
/// [`native_dit_projection`]. MLX affine is W4/8A16: `quantized_matmul` reads the compute input
/// directly and writes the same output a dense GEMM does, so it adds no packed transient.
fn native_dit_bytes(
    weights: &Path,
    precision: Precision,
    storage: NativeDitStorage,
    cfg: &gen_core::flux3::DitConfig,
) -> CoreResult<u64> {
    if storage == NativeDitStorage::Publisher(DitFormat::Dense) {
        return dit_bytes(weights, precision);
    }
    let targets = storage.packed_targets(cfg)?;
    projected_safetensors_bytes(weights, native_dit_projection(precision, &targets))
}

/// Resident bytes of the autoencoder as [`crate::model::load_autoencoder`] holds it: f32.
fn autoencoder_bytes(weights: &Path) -> CoreResult<u64> {
    projected_safetensors_bytes(weights, |_| ResidentProjection::Float32)
}

/// Key prefix of the Qwen3-VL language tower inside the VLM checkpoint.
const LANGUAGE_TOWER_PREFIX: &str = "model.language_model.";
/// The untied output head, at the checkpoint root.
const LANGUAGE_HEAD_KEY: &str = "lm_head.weight";

/// Resident bytes of the text encoder as [`crate::text_encoder::Flux3TextEncoder::load`] builds
/// it: the shared decoder reads the language tower (`model.language_model.*`) and, when the head
/// is untied, `lm_head.weight`, casting each to bf16 whatever the file stores. The vision tower
/// rides along on disk and is materialized by nothing on this route.
fn text_encoder_bytes(dir: &Path) -> CoreResult<u64> {
    let untied_head = !mlx_llm::ModelConfig::from_dir(dir)
        .map_err(|e| CoreError::Msg(format!("{MODEL_ID}: reading the text encoder config: {e}")))?
        .tie_word_embeddings;
    let mut total = 0_u64;
    for shard in crate::text_encoder::weight_shards(dir)? {
        let bytes = projected_safetensors_bytes(&shard, |header| {
            if header.name.starts_with(LANGUAGE_TOWER_PREFIX)
                || (untied_head && header.name == LANGUAGE_HEAD_KEY)
            {
                ResidentProjection::Bfloat16
            } else {
                ResidentProjection::Omit
            }
        })?;
        total = total
            .checked_add(bytes)
            .ok_or_else(|| CoreError::Msg(format!("{MODEL_ID}: text encoder byte sum overflow")))?;
    }
    Ok(total)
}

/// The provider safety check: the shared handshake (calibration identity, loaded tier,
/// selection, budget), then the route's own gate. It can reject, never swap in a different
/// strategy or tier.
pub fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        gen_core::flux3::references::validate_memory_route(
            &context.mode,
            context.geometry.reference_count,
        )?;
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{MODEL_ID}: there is no PiD decoder on this route"
            )));
        }
        if context.geometry.frames > 1 {
            return Err(CoreError::Unsupported(format!(
                "{MODEL_ID}: a still-image route cannot admit {} frames",
                context.geometry.frames
            )));
        }
        Ok(())
    };
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(mlx_gen::request_scope::loaded_tier(spec)),
        Some(&route_gate),
    )
}

/// One provider-owned valid context per implemented optimized rung (staged residency).
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
    let context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        mlx_gen::request_scope::loaded_tier(spec),
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: true,
            overlay: None,
        },
    )?;
    Ok(vec![MemoryBehaviorFixture::new(context)])
}

/// The executable admission seam: the shared MLX request scope, whose `configure_request`
/// writes the admitted `GenerationMemory` onto the request that [`crate::model::Flux3Klein`]
/// then honours.
pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    mlx_gen::request_scope::admit(safety_check(spec, contract, context))?;
    let config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        if contract.provider_id == gen_core::flux3::BASE_MODEL_ID {
            gen_core::flux3::BASE_MODEL_ID
        } else {
            MODEL_ID
        },
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        contract
            .architecture_facts
            .transformer_blocks
            .unwrap_or_default() as usize,
        |_use_pid, _edge, _overlap| {
            Err(CoreError::Unsupported(format!(
                "{MODEL_ID}: {DECODE_NOT_ENABLED}"
            )))
        },
    )?;
    Ok(Some(
        gen_core::flux3::references::guard_memory_scope_with_resources(
            Box::new(mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(
                config,
                mlx_gen::request_scope::MlxScopeCleanup::Device,
            )),
            context.geometry,
            context.reference_resources.clone(),
        ),
    ))
}

/// The registry-load surface the route witnesses: the dense tier and the MLX affine q4/q8
/// tiers, eagerly materialized, under both offload policies. NVFP4 and the deferred load
/// shape have no MLX FLUX3 loader, so neither is witnessed.
pub fn memory_contract_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    gen_core::mlx_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| {
            matches!(
                surface.selector.tier,
                MemoryContractSurfaceTier::Bf16
                    | MemoryContractSurfaceTier::Q4
                    | MemoryContractSurfaceTier::Q8
            ) && surface.selector.load_shape == LoadShape::EagerMaterialization
        })
        .collect()
}

/// PRE-FIT reference workspace of one request on the package `spec` names, through the shared
/// formula ([`gen_core::flux3::reference_workspace`]) at this backend's compute width.
fn reference_workspace_for(
    provider_id: &str,
    spec: &LoadSpec,
    facts: &gen_core::MemoryReferenceResourceFacts,
) -> CoreResult<u64> {
    refuse_unloadable_spec(spec)?;
    let inputs = gen_core::flux3::reference_workspace::ReferenceWorkspaceInputs::from_spec(
        spec,
        provider_id,
        u64::from(compute_width(spec.precision)),
    )?;
    gen_core::flux3::reference_workspace::reference_workspace_bytes(&inputs, Some(facts))
}

/// `flux3_klein`'s pre-fit reference pricer.
pub fn reference_workspace_bytes(
    spec: &LoadSpec,
    facts: &gen_core::MemoryReferenceResourceFacts,
) -> CoreResult<u64> {
    reference_workspace_for(MODEL_ID, spec, facts)
}

/// `flux3_klein_base`'s pre-fit reference pricer (references occupy both CFG rows).
pub fn base_reference_workspace_bytes(
    spec: &LoadSpec,
    facts: &gen_core::MemoryReferenceResourceFacts,
) -> CoreResult<u64> {
    reference_workspace_for(gen_core::flux3::BASE_MODEL_ID, spec, facts)
}

pub const REFERENCE_WORKSPACE_REGISTRATION: gen_core::ReferenceWorkspaceRegistration =
    gen_core::ReferenceWorkspaceRegistration {
        provider_id: MODEL_ID,
        reference_workspace_bytes,
    };

pub const BASE_REFERENCE_WORKSPACE_REGISTRATION: gen_core::ReferenceWorkspaceRegistration =
    gen_core::ReferenceWorkspaceRegistration {
        provider_id: gen_core::flux3::BASE_MODEL_ID,
        reference_workspace_bytes: base_reference_workspace_bytes,
    };

/// The pre-load contract and admission check of `flux3_klein`.
pub const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: MODEL_ID,
    contract: |spec| memory_strategy_contract(MODEL_ID, spec),
    safety_check,
};

/// The weights-free contract fixture.
pub const MEMORY_CONTRACT_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: MODEL_ID,
        contract: |spec| weights_free_memory_strategy_contract(MODEL_ID, spec),
        surface_specs: memory_contract_surface_specs,
    };

/// The provider-owned valid fixtures and the executable admission seam.
pub const MEMORY_BEHAVIOR_REGISTRATION: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: MODEL_ID,
        valid_fixtures: registered_valid_fixture,
        begin_request: registered_begin_request,
    };

pub const BASE_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: gen_core::flux3::BASE_MODEL_ID,
    contract: |spec| memory_strategy_contract(gen_core::flux3::BASE_MODEL_ID, spec),
    safety_check,
};

/// The weights-free contract fixture.
pub const BASE_MEMORY_CONTRACT_FIXTURE: gen_core::MemoryContractFixtureRegistration =
    gen_core::MemoryContractFixtureRegistration {
        provider_id: gen_core::flux3::BASE_MODEL_ID,
        contract: |spec| {
            weights_free_memory_strategy_contract(gen_core::flux3::BASE_MODEL_ID, spec)
        },
        surface_specs: memory_contract_surface_specs,
    };

/// The provider-owned valid fixtures and the executable admission seam.
pub const BASE_MEMORY_BEHAVIOR_REGISTRATION: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: gen_core::flux3::BASE_MODEL_ID,
        valid_fixtures: registered_valid_fixture,
        begin_request: registered_begin_request,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::{Quant, WeightsSource};

    /// A spec no loader could open: the weights-free surface.
    #[test]
    fn base_identity_and_adapter_overlay_are_distinct_and_counted_once() {
        let empty = sentinel();
        let distilled = weights_free_memory_strategy_contract(MODEL_ID, &empty).unwrap();
        let base =
            weights_free_memory_strategy_contract(gen_core::flux3::BASE_MODEL_ID, &empty).unwrap();
        assert_ne!(base.provider_id, distilled.provider_id);
        assert_ne!(base.calibration, distilled.calibration);
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        // The real base fixture, resolved from its headers — never the distilled package with
        // `flow_map` hand-flipped. It ships only config + DiT, so the distilled package's
        // encoder and autoencoder ride in as components.
        let mut spec = LoadSpec::new(WeightsSource::Dir(fixture.join("tiny-base-snapshot")));
        spec.components.insert(
            gen_core::flux3::TEXT_ENCODER_COMPONENT.into(),
            WeightsSource::Dir(fixture.join("tiny-snapshot/text_encoder")),
        );
        spec.components.insert(
            gen_core::flux3::AUTOENCODER_COMPONENT.into(),
            WeightsSource::File(fixture.join("tiny-snapshot/ae.safetensors")),
        );
        let config =
            gen_core::flux3::resolve_checkpoint(&fixture.join("tiny-base-snapshot")).unwrap();
        assert!(!config.dit.flow_map);
        let resources =
            GenerationResources::from_spec(&spec, &config, gen_core::flux3::BASE_MODEL_ID).unwrap();
        let plain = generation_contract(&spec, &resources, &config).unwrap();
        // The registered entry point resolves the same package to the same contract on the base
        // route, and refuses the distilled route for it.
        let registered = memory_strategy_contract(gen_core::flux3::BASE_MODEL_ID, &spec).unwrap();
        assert_eq!(registered.provider_id, gen_core::flux3::BASE_MODEL_ID);
        assert_eq!(registered.asset_facts, plain.asset_facts);
        // Exercise the EARLY resolver directly: a later from_spec route check must not
        // rescue this test if resolve_package stops validating the route before component work.
        // Mutation: removing resolve_package's validate_route makes this return Ok.
        match crate::model::resolve_package(
            &fixture.join("tiny-base-snapshot"),
            MODEL_ID,
            spec.quantize,
            spec.precision,
        ) {
            Err(CoreError::Unsupported(message)) => {
                assert!(message.contains("flow_map"), "{message}")
            }
            other => panic!("the early wrong-route resolver must refuse, got {other:?}"),
        }
        match memory_strategy_contract(MODEL_ID, &spec) {
            Err(CoreError::Unsupported(message)) => {
                assert!(message.contains("flow_map"), "{message}")
            }
            other => panic!("wrong-route contract must retain the typed refusal, got {other:?}"),
        }
        spec.adapters.push(gen_core::AdapterSpec::new(
            fixture.join("flux3_base_lora.safetensors"),
            0.4,
            gen_core::AdapterKind::Lora,
        ));
        let adapted = generation_contract(&spec, &resources, &config).unwrap();
        assert!(adapted.asset_facts.overlay_bytes > 0);
        assert_eq!(adapted.asset_facts.base_bytes, plain.asset_facts.base_bytes);
        for contract in [&plain, &adapted] {
            let variables = match &contract.formula {
                MemoryFormulaKind::PhaseEnvelope { variables, .. }
                | MemoryFormulaKind::ComponentPhaseEnvelope { variables, .. } => variables.clone(),
                other => panic!("unexpected formula {other:?}"),
            };
            assert!(
                !variables.contains(&MemoryFormulaVariable::BatchCount),
                "CFG x2 is fixed, not a batch axis"
            );
        }
        spec.components.insert(
            gen_core::flux3::pup::UPSAMPLER_COMPONENT.into(),
            WeightsSource::Dir(fixture.join("tiny-pup-snapshot")),
        );
        let both = generation_contract(&spec, &resources, &config).unwrap();
        spec.adapters.clear();
        let pup = generation_contract(&spec, &resources, &config).unwrap();
        assert_eq!(
            both.asset_facts.overlay_bytes,
            adapted.asset_facts.overlay_bytes + pup.asset_facts.overlay_bytes
        );
        let MemoryFormulaKind::ComponentPhaseEnvelope {
            resident_components,
            ..
        } = both.formula
        else {
            panic!("component envelope required")
        };
        assert_eq!(resident_components.len(), 2);
        assert_eq!(
            resident_components
                .iter()
                .filter(|c| c.kind == gen_core::MemoryComponentKind::AdapterStack)
                .count(),
            1
        );
        assert_eq!(
            resident_components
                .iter()
                .filter(|c| c.kind == gen_core::MemoryComponentKind::PromptUpsampler)
                .count(),
            1
        );
    }

    fn sentinel() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/flux3-not-on-disk".into()))
    }

    /// The released packages' own `config.json` bodies (pinned snapshots of
    /// `flux-3-image-klein-base` and `flux-3-image-klein`): the SAME `model` label, and the base
    /// config carries no `dit.flow_map` at all.
    const REAL_BASE_CONFIG: &str = r#"{"format_version":1,"model":"flux-3-image-klein","quantization":null,"torch_dtype":"bfloat16","dit":{"hidden":3072,"heads":24,"depth":5,"depth_single":28,"mlp_ratio":3.0,"axes_dim":[32,32,32,32],"theta":10000,"in_channels":{"image":128,"image_cond":128,"text":20480},"out_modes":["image"],"vec_in_dim":null,"image_rope_scale":0.5},"text_encoder":{"path":"text_encoder","tokenizer":"text_encoder","layers":[4,8,12,16,20,24,28,32],"pad_multiple":80,"max_tokens":8192,"dtype":"bfloat16"},"vae":{"path":"ae.safetensors","dtype":"float32"},"sampler":{"steps":50,"guidance":3.5,"alpha":6.93,"size":[1024,1024]}}"#;
    const REAL_DISTILLED_CONFIG: &str = r#"{"format_version":1,"model":"flux-3-image-klein","quantization":null,"torch_dtype":"bfloat16","dit":{"hidden":3072,"heads":24,"depth":5,"depth_single":28,"mlp_ratio":3.0,"axes_dim":[32,32,32,32],"theta":10000,"in_channels":{"image":128,"image_cond":128,"text":20480},"out_modes":["image"],"vec_in_dim":null,"image_rope_scale":0.5,"flow_map":true},"text_encoder":{"path":"text_encoder","tokenizer":"text_encoder","layers":[4,8,12,16,20,24,28,32],"pad_multiple":80,"max_tokens":8192,"dtype":"bfloat16"},"vae":{"path":"ae.safetensors","dtype":"float32"},"sampler":{"steps":16,"guidance":null,"alpha":6.93,"size":[1024,1024]}}"#;

    /// Header records of the inventory `storage` declares for `cfg`, at their stored bytes —
    /// for the dense publisher format exactly the released `model.safetensors` inventory
    /// (names, BF16 dtypes, shapes). No file and no tensor payload is read.
    fn declared_headers(
        storage: NativeDitStorage,
        cfg: &gen_core::flux3::DitConfig,
    ) -> Vec<SafetensorsTensorHeader> {
        storage
            .expected_tensors(cfg)
            .unwrap()
            .into_iter()
            .map(|(name, spec)| SafetensorsTensorHeader {
                data_bytes: (spec.shape.iter().product::<usize>() * spec.dtype.size()) as u64,
                name,
                dtype: spec.dtype,
                shape: spec.shape,
            })
            .collect()
    }

    /// Pure planning: the real config bodies bind to their catalog route by id, not label, and
    /// plan the same geometry; the base body (no `flow_map`) never binds the distilled route by
    /// default, and the explicit distilled flag refuses the base route.
    #[test]
    fn real_config_bodies_plan_by_catalog_route_not_label() {
        let base_meta = gen_core::flux3::Flux3MetadataConfig::parse(REAL_BASE_CONFIG).unwrap();
        let distilled_meta =
            gen_core::flux3::Flux3MetadataConfig::parse(REAL_DISTILLED_CONFIG).unwrap();
        assert_eq!(base_meta.model, distilled_meta.model);
        assert_eq!(base_meta.dit.flow_map, None, "absent stays unresolved");
        assert_eq!(distilled_meta.dit.flow_map, Some(true));
        let base = base_meta.for_model(gen_core::flux3::BASE_MODEL_ID).unwrap();
        let distilled = distilled_meta.for_model(MODEL_ID).unwrap();
        assert!(!base.dit.flow_map && distilled.dit.flow_map);
        assert_eq!(base.sampler.guidance, Some(3.5));
        assert_eq!(distilled.sampler.guidance, None);
        assert!(distilled_meta
            .for_model(gen_core::flux3::BASE_MODEL_ID)
            .is_err());
        base.validate_route(gen_core::flux3::BASE_MODEL_ID).unwrap();
        assert!(base.validate_route(MODEL_ID).is_err());
        assert!(distilled
            .validate_route(gen_core::flux3::BASE_MODEL_ID)
            .is_err());
        // Identical architecture facts: the two routes differ only by the target-time embedder.
        assert_eq!(
            architecture_facts(Some(&base), Precision::Bf16),
            architecture_facts(Some(&distilled), Precision::Bf16)
        );
    }

    /// The DiT is priced at what the loader HOLDS: matrices at the compute width, the 86 bf16
    /// `[128]` q/k norm scales of either released checkpoint at f32 (`Loader::f32`), never as
    /// stored — on the dense publisher inventory and on a packed MLX affine one alike.
    ///
    /// *Mutation that reds this:* pricing rank-1 tensors `ResidentProjection::Stored` in
    /// [`dense_dit_projection`] (the pre-fix undercharge of exactly 22,016 bytes).
    #[test]
    fn dit_pricing_charges_norm_scales_at_f32_on_both_real_inventories() {
        const NORM_SCALES: usize = 2 * (3 * 5 + 28);
        const UNDERCHARGE: u64 = (NORM_SCALES * 128 * 2) as u64;
        assert_eq!(UNDERCHARGE, 22_016);
        let as_stored_vectors = |precision: Precision| {
            let dense = dense_dit_projection(precision);
            move |header: &SafetensorsTensorHeader| {
                if header.shape.len() >= 2 {
                    dense(header)
                } else {
                    ResidentProjection::Stored
                }
            }
        };
        for (body, id, real_tensor_count, ref_time_bytes) in [
            (REAL_BASE_CONFIG, gen_core::flux3::BASE_MODEL_ID, 357, 0u64),
            // `ref_time_in.{in_layer [3072, 256], out_layer [3072, 3072]}`, the only difference.
            (
                REAL_DISTILLED_CONFIG,
                MODEL_ID,
                359,
                (3072 * 256 + 3072 * 3072) as u64,
            ),
        ] {
            let cfg = gen_core::flux3::Flux3MetadataConfig::parse(body)
                .unwrap()
                .for_model(id)
                .unwrap()
                .dit;
            let dense = declared_headers(NativeDitStorage::Publisher(DitFormat::Dense), &cfg);
            assert_eq!(
                dense.len(),
                real_tensor_count,
                "{id}: the released inventory"
            );
            let vectors: Vec<_> = dense.iter().filter(|h| h.shape.len() < 2).collect();
            assert_eq!(vectors.len(), NORM_SCALES, "{id}");
            assert!(vectors.iter().all(|h| h.shape == [128]
                && h.dtype == gen_core::weightsmeta::Dtype::BF16
                && (h.name.ends_with(".norm.query_norm.scale")
                    || h.name.ends_with(".norm.key_norm.scale"))));
            let matrix_elements: u64 = dense
                .iter()
                .filter(|h| h.shape.len() >= 2)
                .map(|h| h.element_count().unwrap())
                .sum();
            let scale_bytes = (NORM_SCALES * 128 * 4) as u64;
            for (precision, width) in [(Precision::Bf16, 2u64), (Precision::Fp32, 4)] {
                let priced = mlx_gen::asset_facts::projected_tensor_headers_bytes(
                    &dense,
                    dense_dit_projection(precision),
                )
                .unwrap();
                assert_eq!(
                    priced,
                    matrix_elements * width + scale_bytes,
                    "{id} {precision:?}"
                );
                let stored = mlx_gen::asset_facts::projected_tensor_headers_bytes(
                    &dense,
                    as_stored_vectors(precision),
                )
                .unwrap();
                assert_eq!(priced - stored, UNDERCHARGE, "{id} {precision:?}");
            }
            for storage in [
                NativeDitStorage::MlxAffine {
                    bits: 4,
                    group_size: 64,
                },
                NativeDitStorage::MlxAffine {
                    bits: 8,
                    group_size: 128,
                },
            ] {
                let packed = declared_headers(storage, &cfg);
                let targets = storage.packed_targets(&cfg).unwrap();
                assert!(!targets.is_empty());
                let priced = mlx_gen::asset_facts::projected_tensor_headers_bytes(
                    &packed,
                    native_dit_projection(Precision::Bf16, &targets),
                )
                .unwrap();
                let stored = mlx_gen::asset_facts::projected_tensor_headers_bytes(&packed, |h| {
                    if h.shape.len() < 2 {
                        ResidentProjection::Stored
                    } else {
                        native_dit_projection(Precision::Bf16, &targets)(h)
                    }
                })
                .unwrap();
                assert_eq!(priced - stored, UNDERCHARGE, "{id} {storage:?}");
                // Packed codes stay at their exact stored bytes.
                let codes: u64 = packed
                    .iter()
                    .filter(|h| targets.contains(&h.name))
                    .map(|h| h.data_bytes)
                    .sum();
                let rest = mlx_gen::asset_facts::projected_tensor_headers_bytes(
                    &packed
                        .iter()
                        .filter(|h| !targets.contains(&h.name))
                        .cloned()
                        .collect::<Vec<_>>(),
                    dense_dit_projection(Precision::Bf16),
                )
                .unwrap();
                assert_eq!(priced, codes + rest, "{id} {storage:?}");
            }
            let base_cfg = gen_core::flux3::Flux3MetadataConfig::parse(REAL_BASE_CONFIG)
                .unwrap()
                .for_model(gen_core::flux3::BASE_MODEL_ID)
                .unwrap()
                .dit;
            let base_dense =
                declared_headers(NativeDitStorage::Publisher(DitFormat::Dense), &base_cfg);
            let price = |headers: &[SafetensorsTensorHeader]| {
                mlx_gen::asset_facts::projected_tensor_headers_bytes(
                    headers,
                    dense_dit_projection(Precision::Bf16),
                )
                .unwrap()
            };
            assert_eq!(
                price(&dense) - price(&base_dense),
                ref_time_bytes * 2,
                "{id}"
            );
        }
    }

    /// The tiny fixtures through the real file pricer: every vector the loader holds is charged
    /// at f32 and every matrix at the compute width, from the file's own headers.
    #[test]
    fn file_pricer_matches_the_loader_widths_on_the_tiny_packages() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        for dir in ["tiny-snapshot", "tiny-base-snapshot"] {
            let weights = fixtures.join(dir).join("model.safetensors");
            let headers = safetensors_path_tensor_headers(&weights).unwrap();
            let elements = |matrices: bool| -> u64 {
                headers
                    .iter()
                    .filter(|h| (h.shape.len() >= 2) == matrices)
                    .map(|h| h.element_count().unwrap())
                    .sum()
            };
            let cfg = gen_core::flux3::resolve_checkpoint(&fixtures.join(dir))
                .unwrap()
                .dit;
            let vectors = headers.iter().filter(|h| h.shape.len() < 2).count();
            assert_eq!(vectors, 2 * (3 * cfg.depth + cfg.depth_single), "{dir}");
            for (precision, width) in [(Precision::Bf16, 2u64), (Precision::Fp32, 4)] {
                assert_eq!(
                    dit_bytes(&weights, precision).unwrap(),
                    elements(true) * width + elements(false) * 4,
                    "{dir} {precision:?}"
                );
                assert_eq!(
                    native_dit_bytes(
                        &weights,
                        precision,
                        NativeDitStorage::Publisher(DitFormat::Dense),
                        &cfg
                    )
                    .unwrap(),
                    dit_bytes(&weights, precision).unwrap()
                );
            }
        }
    }

    fn support(contract: &MemoryProviderContract, strategy: MemoryStrategy) -> &'static str {
        match &contract.capability(strategy).unwrap().support {
            MemoryStrategySupport::Implemented => "implemented",
            MemoryStrategySupport::StructurallyNotApplicable { .. } => "not-applicable",
            MemoryStrategySupport::Missing => "missing",
        }
    }

    /// The ladder the route publishes, on every surface it witnesses.
    ///
    /// *Mutations that red this:* declaring `StagedResidency` `Missing`; promoting a bounded
    /// rung to `Implemented` (no lifecycle hook backs it); dropping the eager filter in
    /// `memory_contract_surface_specs`.
    /// Hosted-only twin of the Candle test: the registered pre-fit reference pricer is zero
    /// without references and priced and monotone with them, on both routes.
    #[test]
    fn registered_reference_pricer_is_additive_and_monotone_on_both_routes() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let registry = crate::provider_registry().unwrap();
        let facts = |canvases: &[(u32, u32)]| gen_core::MemoryReferenceResourceFacts {
            conditioning_tokens: canvases
                .iter()
                .map(|&(w, h)| u64::from(w / 16) * u64::from(h / 16))
                .sum(),
            prepared_canvases: canvases.to_vec(),
        };
        for (id, dir) in [
            (MODEL_ID, "tiny-snapshot"),
            (gen_core::flux3::BASE_MODEL_ID, "tiny-base-snapshot"),
        ] {
            let mut spec = LoadSpec::new(gen_core::WeightsSource::Dir(fixtures.join(dir)));
            // The tiny base package ships only config + DiT; both routes take the distilled
            // package's encoder and autoencoder as components, as the generator tests do.
            spec.components.insert(
                gen_core::flux3::TEXT_ENCODER_COMPONENT.into(),
                gen_core::WeightsSource::Dir(fixtures.join("tiny-snapshot/text_encoder")),
            );
            spec.components.insert(
                gen_core::flux3::AUTOENCODER_COMPONENT.into(),
                gen_core::WeightsSource::File(fixtures.join("tiny-snapshot/ae.safetensors")),
            );
            assert_eq!(
                registry.reference_workspace_bytes(id, &spec, None).unwrap(),
                Some(0)
            );
            let one = registry
                .reference_workspace_bytes(id, &spec, Some(&facts(&[(64, 64)])))
                .unwrap()
                .unwrap();
            let two = registry
                .reference_workspace_bytes(id, &spec, Some(&facts(&[(64, 64), (64, 64)])))
                .unwrap()
                .unwrap();
            assert!(one > 0 && two > one, "{id}: {one} {two}");
        }
    }

    #[test]
    fn the_route_declares_exactly_the_rungs_its_engine_runs() {
        let surfaces = memory_contract_surface_specs();
        assert_eq!(
            surfaces.len(),
            6,
            "dense + MLX affine q4/q8 tiers, eager, x 2 offload policies"
        );
        let mut selectors: Vec<_> = surfaces.iter().map(|s| s.selector.id()).collect();
        selectors.sort();
        assert_eq!(
            selectors,
            [
                "bf16:resident:eager",
                "bf16:sequential:eager",
                "q4:resident:eager",
                "q4:sequential:eager",
                "q8:resident:eager",
                "q8:sequential:eager"
            ]
        );
        for surface in &surfaces {
            let contract = weights_free_memory_strategy_contract(MODEL_ID, &surface.spec).unwrap();
            gen_core_testkit::memory_strategy_conformance(&contract);
            assert_eq!(contract.provider_id, MODEL_ID);
            let calibration = contract.calibration.as_ref().unwrap();
            assert_eq!(
                calibration.fingerprint,
                tier_fingerprint(MEMORY_CALIBRATION_FINGERPRINT, surface.spec.quantize, None)
            );
            gen_core::validate_calibration_fingerprint(&calibration.fingerprint).unwrap();
            assert_eq!(contract.asset_facts, MemoryAssetFacts::default());
            let declared: Vec<&str> = MemoryStrategy::ALL
                .into_iter()
                .map(|strategy| support(&contract, strategy))
                .collect();
            assert_eq!(
                declared,
                [
                    "implemented",
                    "implemented",
                    "missing",
                    "missing",
                    "missing"
                ],
                "[{}]",
                surface.selector.id()
            );
            assert!(!contract.lifecycle.decode_tiling);
            assert!(!contract.lifecycle.attention_chunking);
            assert!(!contract.lifecycle.transformer_window_materialization);
            assert!(contract.lifecycle.synchronized_phase_release);
            assert!(contract.phase_facts.is_some());
        }
    }

    #[test]
    fn optional_upsampler_is_separate_and_charged_once() {
        let mut contract = weights_free_memory_strategy_contract(MODEL_ID, &sentinel()).unwrap();
        assert!(matches!(
            contract.formula,
            MemoryFormulaKind::PhaseEnvelope { .. }
        ));
        assert!(contract.resident_components().is_empty());
        contract.asset_facts = MemoryAssetFacts {
            base_bytes: 60,
            conditioning_bytes: 10,
            transformer_bytes: 40,
            decoder_bytes: 10,
            overlay_bytes: 0,
        };
        let original = contract.asset_facts;
        let before = contract.total_resident_bytes();
        account_upsampler(
            &mut contract,
            37,
            gen_core::MemoryComponentResidency::WholeRender,
        )
        .unwrap();
        assert_eq!(contract.asset_facts.base_bytes, original.base_bytes);
        assert_eq!(
            contract.asset_facts.conditioning_bytes,
            original.conditioning_bytes
        );
        assert_eq!(contract.asset_facts.overlay_bytes, 37);
        assert_eq!(contract.total_resident_bytes(), before + 37);
        assert_eq!(contract.auxiliary_resident_bytes(), 37);
        assert_eq!(contract.resident_components().len(), 1);
        let component = &contract.resident_components()[0];
        assert_eq!(component.id, "prompt_upsampler");
        assert_eq!(
            component.kind,
            gen_core::MemoryComponentKind::PromptUpsampler
        );
        assert_eq!(component.resident_bytes, 37);
        assert_eq!(
            component.residency,
            gen_core::MemoryComponentResidency::WholeRender
        );
        assert!(
            contract.conformance_errors().is_empty(),
            "{:?}",
            contract.conformance_errors()
        );
        let mut absent = base_contract(Precision::Bf16, None);
        assert!(account_upsampler(
            &mut absent,
            0,
            gen_core::MemoryComponentResidency::WholeRender
        )
        .is_err());
        assert_eq!(absent.asset_facts.overlay_bytes, 0);
    }

    #[test]
    fn upsampler_charge_matches_the_loader_each_policy_actually_runs() {
        let prefix = gen_core::core_llm::defaults::MLX.prefix_cache_bytes;
        assert!(prefix > 0);
        let (resident, whole) = upsampler_charge(37, gen_core::OffloadPolicy::Resident).unwrap();
        assert_eq!(
            resident,
            37 + prefix,
            "a warm default load reserves its prefix-cache budget"
        );
        assert_eq!(whole, gen_core::MemoryComponentResidency::WholeRender);
        let (bounded, evicted) = upsampler_charge(37, gen_core::OffloadPolicy::Sequential).unwrap();
        assert_eq!(bounded, 37, "load_bounded disables the prefix cache");
        let mut staged = base_contract(Precision::Bf16, None);
        account_upsampler(&mut staged, bounded, evicted).unwrap();
        let component = &staged.resident_components()[0];
        assert_eq!(component.steady_state_bytes(), 0);
        assert_eq!(component.evicted_bytes(), 37);
        assert!(
            staged.conformance_errors().is_empty(),
            "{:?}",
            staged.conformance_errors()
        );
        assert!(upsampler_charge(u64::MAX, gen_core::OffloadPolicy::Resident).is_err());
    }

    #[test]
    fn fingerprint_is_valid() {
        gen_core::validate_calibration_fingerprint(MEMORY_CALIBRATION_FINGERPRINT).unwrap();
    }

    /// *Mutation that reds this:* dropping either arm of `refuse_unloadable_spec`.
    #[test]
    fn a_spec_no_loader_opens_has_no_contract() {
        for spec in [
            sentinel().with_quant(Quant::Nvfp4),
            sentinel().with_quant(Quant::Fp8Rowwise),
            sentinel().with_load_shape(LoadShape::DeferredMaterialization),
        ] {
            match weights_free_memory_strategy_contract(MODEL_ID, &spec) {
                Err(CoreError::Unsupported(_)) => {}
                other => panic!("expected a typed refusal, got {other:?}"),
            }
        }
        assert!(matches!(
            weights_free_memory_strategy_contract("iris_3b", &sentinel()),
            Err(CoreError::Unsupported(_))
        ));
    }

    /// The route gate rejects a reference-carrying, PiD or multi-frame context by name.
    #[test]
    fn route_gate_rejects_foreign_requests() {
        let contract = weights_free_memory_strategy_contract(MODEL_ID, &sentinel()).unwrap();
        let mut context = gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::StagedResidency,
            mlx_gen::request_scope::loaded_tier(&sentinel()),
            MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: true,
                overlay: None,
            },
        )
        .unwrap();
        assert!(matches!(
            safety_check(&sentinel(), &contract, &context),
            MemorySafetyDecision::Accept
        ));
        context.geometry.reference_count = 1;
        match safety_check(&sentinel(), &contract, &context) {
            MemorySafetyDecision::Reject { reason } => {
                assert!(reason.contains("reference"), "{reason}")
            }
            other => panic!("expected a rejection, got {other:?}"),
        }
    }
}
