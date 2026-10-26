//! The distilled FLUX 3 Candle request-memory contract (the shared ladder, sc-26235) — the Candle
//! twin of `mlx_gen_flux3::memory_strategy`, published through the seams the other Candle families
//! use (`gen_core::MemoryRegistration`, its weights-free contract fixture, and the executable
//! behavior registration).
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
//! * **Rung 1.** [`crate::model::Flux3Klein`] holds a request-scoped [`candle_gen::Residency`]: a
//!   request whose admitted `GenerationMemory::stage_residency` is set (written by the request
//!   scope this module opens), or any request of a generator loaded `OffloadPolicy::Sequential`,
//!   evicts the warm pair, loads the text encoder, encodes, synchronizes, drops the encoder, then
//!   loads the DiT and the autoencoder. The staged weight floor is therefore
//!   `max(tower, DiT + autoencoder)`, never the sum.
//! * **Rungs 2–4 are `Missing`, not "not applicable".** The route has an autoencoder, attention
//!   and a block stack, so all three describe real work; none is wired in S2. See
//!   [`DECODE_NOT_ENABLED`], [`ATTENTION_NOT_ENABLED`] and [`TRANSFORMER_WINDOW_NOT_ENABLED`].
//!   (The shared i32-overflow guards the DiT attention and the autoencoder convolutions run
//!   through are correctness bounds, not a selectable memory rung.)
//!
//! # Nothing here is a measurement
//!
//! The contract carries the route shape and, for a materialized package, the resident bytes of
//! the components the loaders build — priced from safetensors **headers** at the width each
//! loader materializes them (the DiT at the compute dtype for every tensor of rank ≥ 2 and f32
//! for the q/k norm scales, the Qwen3-VL language tower at the dtype the production device runs
//! it in, the autoencoder at f32). No activation number is published and this crate registers no
//! `ActivationMemoryRegistration`: that carrier is measurement-only, and the epic's terminal
//! campaign owns the anchors. The fingerprint is distinct from the MLX twin's: the two backends
//! never share calibration evidence.

use std::path::Path;

use candle_gen::candle_core::DType;
use candle_gen::gen_core::flux3::artifacts::NativeDitStorage;
use candle_gen::gen_core::flux3::quantization::DitFormat;
use candle_gen::gen_core::flux3::{
    resolve_checkpoint, Flux3Config, GenerationResources, TextEncoderConfig, BASE_MODEL_ID,
};
use candle_gen::gen_core::weightsmeta::{safetensors_path_tensor_headers, SafetensorsTensorHeader};
use candle_gen::gen_core::{
    self, Error as CoreError, LoadShape, LoadSpec, MemoryArchitectureFacts, MemoryAssetFacts,
    MemoryBackendRealization, MemoryBehaviorFixture, MemoryBehaviorRoute,
    MemoryCalibrationIdentity, MemoryContractSurfaceSpec, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategySupport, MemoryWindowMaterialization, Precision, Quant,
    ResidentRequestMemory, Result as CoreResult,
};

use crate::model::MODEL_ID;

/// Content fingerprint of the contract. Load shape is a separate typed axis on
/// [`MemoryCalibrationIdentity`], so it is deliberately absent from the string.
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "flux3-klein-candle-derived-2026-10-15-v1";

/// Why rung 2 is declared `Missing` (applicable, not enabled): the autoencoder's decoder runs
/// whole-image GroupNorm resnets; a bounded decode is the normalization-correct tiled tail the
/// FLUX.2 crate has (sc-19753), not yet ported to this autoencoder (sc-26241).
pub const DECODE_NOT_ENABLED: &str =
    "the FLUX 3 autoencoder decodes the whole image; the normalization-correct tiled decode tail \
     is not ported to it yet (sc-26241)";

/// Why rung 3 is declared `Missing`: the shared bounded-attention plan's saving is a per-family
/// real-weight measurement this story's lane cannot take, and no request budget is wired through
/// `Flux3Dit::forward` (the i32-overflow guard it runs through is a correctness bound).
pub const ATTENTION_NOT_ENABLED: &str =
    "bounded attention is not wired through Flux3Dit::forward: the shared plan's saving is a \
     per-family real-weight measurement";

/// Why rung 4 is declared `Missing`: the loader materializes the whole DiT at load; there is no
/// deferred-materialization load shape and no block-window inference loader.
pub const TRANSFORMER_WINDOW_NOT_ENABLED: &str =
    "the FLUX 3 loader materializes the whole DiT at load; there is no deferred-materialization \
     load shape or block-window inference loader";

/// The numeric tier a spec loads: its precision and its DiT storage tier (`None` dense, or the
/// GGML Q4_1 / Q8_0 package `LoadSpec::quantize` selects). No component is resident above the
/// selected width; the encoder and autoencoder keep their own representations.
fn loaded_tier(spec: &LoadSpec) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: spec.precision,
        quant: spec.quantize,
        component_precision_floors: &[],
    }
}

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
fn refuse_unloadable_spec(id: &str, spec: &LoadSpec) -> CoreResult<()> {
    refuse_deferred_load_shape(id, spec)?;
    selected_storage(id, spec.quantize).map(|_| ())
}

/// The `'static` registry id of one of this crate's two routes (the request-scope config keeps
/// it). Anything else is not this provider.
pub(crate) fn route_id(provider_id: &str) -> CoreResult<&'static str> {
    match provider_id {
        MODEL_ID => Ok(MODEL_ID),
        BASE_MODEL_ID => Ok(BASE_MODEL_ID),
        other => Err(CoreError::Unsupported(format!(
            "{other}: not the FLUX 3 Candle memory provider"
        ))),
    }
}

/// The Candle native DiT storage a `LoadSpec::quantize` selects: dense, or the GGML Q4_1 / Q8_0
/// packages converted from the ORIGINAL dense checkpoint; on the CUDA build also the publisher
/// FP8R/NVFP4 artifacts as shipped. Every other format (and FP8R/NVFP4 off CUDA) is a typed
/// refusal naming `id`, never a dequantized or dense substitute.
pub fn selected_storage(id: &str, quant: Option<Quant>) -> CoreResult<NativeDitStorage> {
    Ok(match quant {
        None => NativeDitStorage::Publisher(DitFormat::Dense),
        Some(Quant::Q4) => NativeDitStorage::GgmlQ4_1,
        Some(Quant::Q8) => NativeDitStorage::GgmlQ8_0,
        // The frozen publisher artifacts, loaded as shipped onto their CUDA kernels. CUDA-only
        // formats are normal: other builds refuse them below rather than dequantizing.
        #[cfg(feature = "cuda")]
        Some(Quant::Fp8Rowwise) => NativeDitStorage::Publisher(DitFormat::Fp8Rowwise),
        #[cfg(feature = "cuda")]
        Some(Quant::Nvfp4) => NativeDitStorage::Publisher(DitFormat::Nvfp4),
        // Every format has a storage on the CUDA build; this refusal is the other builds'.
        #[cfg_attr(feature = "cuda", allow(unreachable_patterns))]
        Some(other) => {
            return Err(CoreError::Unsupported(format!(
                "{id}: LoadSpec::quantize {} has no Candle FLUX3 kernel on this build; the \
                 native Candle tiers are q4 (GGML Q4_1) and q8 (GGML Q8_0) converted from the \
                 original dense package",
                other.key()
            )))
        }
    })
}

/// The calibration identity of one numeric tier. A packed tier is a different stored footprint
/// and a different GEMM, so it never inherits the dense tier's identity.
pub fn tier_fingerprint(id: &str, base: &str, quant: Option<Quant>) -> CoreResult<String> {
    Ok(match selected_storage(id, quant)? {
        NativeDitStorage::Publisher(DitFormat::Dense) => base.to_owned(),
        storage => format!("{base}-{}", storage.key()?.replace('_', "-")),
    })
}

// ================================================================================================
// Pricing.
// ================================================================================================

/// Whether the build's production device ([`candle_gen::default_device`]) is the Candle CPU
/// backend. The contract prices the production device without opening it.
fn production_device_is_cpu() -> bool {
    !cfg!(any(feature = "cuda", feature = "metal"))
}

/// The dtype the Qwen3-VL tower is materialized in on the production device — the same rule as
/// [`crate::text_encoder::tower_dtype`]: the package's `text_encoder.dtype` on a GPU, f32 on the
/// Candle CPU backend.
pub fn production_tower_dtype(cfg: &TextEncoderConfig) -> CoreResult<DType> {
    let declared = crate::text_encoder::declared_tower_dtype(cfg)?;
    Ok(if production_device_is_cpu() {
        DType::F32
    } else {
        declared
    })
}

fn tensor_bytes(header: &SafetensorsTensorHeader, float_width: u64) -> CoreResult<u64> {
    if header.is_float() {
        header.materialized_bytes(float_width)
    } else {
        Ok(header.data_bytes)
    }
}

fn checked_sum(id: &str, parts: impl Iterator<Item = CoreResult<u64>>) -> CoreResult<u64> {
    let mut sum = 0_u64;
    for part in parts {
        sum = sum
            .checked_add(part?)
            .ok_or_else(|| CoreError::Msg(format!("{id}: component byte sum overflow")))?;
    }
    Ok(sum)
}

/// Resident bytes of the DiT as [`crate::model::load_dit`] holds it: every tensor of rank two or
/// more at the compute dtype; vectors (the q/k norm scales) in f32.
pub fn dit_bytes(id: &str, weights: &Path, precision: Precision) -> CoreResult<u64> {
    let compute_width = compute_width(precision) as u64;
    let headers = safetensors_path_tensor_headers(weights)?;
    checked_sum(
        id,
        headers.iter().map(|header| {
            tensor_bytes(
                header,
                if header.shape.len() >= 2 {
                    compute_width
                } else {
                    4
                },
            )
        }),
    )
}

/// Resident DiT bytes for an explicit native storage, as `Flux3Dit::load_with_storage` holds it:
///
/// * each packed target at its EXACT stored bytes (GGML block scales live inside the blocks;
///   publisher FP8 codes and packed E2M1 nibbles stage as stored), no dequantized copy anywhere;
/// * a publisher `<stem>.weight_scale` at its STAGED width whatever its stored rank: FP8R row
///   scales stage as f32 (4 B/element), NVFP4 block scales as their E4M3 bytes;
/// * every other matrix at the compute width and every vector at f32, exactly as the dense path;
/// * plus [`gen_core::flux3::packed_gemm::PACKED_GEMM_TRANSIENT_BUDGET`], charged ONCE as resident.
///   The row-chunk planner caps every packed launch (f32 activation copy, Q8/FP8/FP4 activation
///   codes, output and padding) at it, and it also bounds candle's grows-only CUDA quantized-matmul
///   workspace, which persists after the largest launch. An upper bound, not a measurement.
pub fn native_dit_bytes(
    id: &str,
    weights: &Path,
    precision: Precision,
    storage: NativeDitStorage,
    cfg: &gen_core::flux3::DitConfig,
) -> CoreResult<u64> {
    if storage == NativeDitStorage::Publisher(DitFormat::Dense) {
        return dit_bytes(id, weights, precision);
    }
    let headers = safetensors_path_tensor_headers(weights)?;
    storage.validate_headers(cfg, &headers)?;
    let targets = storage.packed_targets(cfg)?;
    let compute_width = compute_width(precision) as u64;
    let weights = checked_sum(
        id,
        headers.iter().map(|header| {
            if targets.contains(&header.name) {
                Ok(header.data_bytes)
            } else if header.name.ends_with(".weight_scale") {
                let elements = header
                    .shape
                    .iter()
                    .try_fold(1u64, |acc, &d| acc.checked_mul(d as u64))
                    .ok_or_else(|| CoreError::Unsupported(format!("{id}: scale size overflow")))?;
                let staged = if storage == NativeDitStorage::Publisher(DitFormat::Fp8Rowwise) {
                    4
                } else {
                    1
                };
                elements
                    .checked_mul(staged)
                    .ok_or_else(|| CoreError::Unsupported(format!("{id}: scale size overflow")))
            } else {
                tensor_bytes(
                    header,
                    if header.shape.len() >= 2 {
                        compute_width
                    } else {
                        4
                    },
                )
            }
        }),
    )?;
    weights
        .checked_add(gen_core::flux3::packed_gemm::PACKED_GEMM_TRANSIENT_BUDGET)
        .ok_or_else(|| CoreError::Unsupported(format!("{id}: DiT byte overflow")))
}

fn apply_tier_fingerprint(
    contract: &mut MemoryProviderContract,
    quant: Option<Quant>,
) -> CoreResult<()> {
    if let Some(calibration) = &mut contract.calibration {
        calibration.fingerprint =
            tier_fingerprint(&contract.provider_id, &calibration.fingerprint, quant)?;
    }
    Ok(())
}

/// Resident bytes of the autoencoder as [`crate::model::load_autoencoder`] holds it: f32.
pub fn autoencoder_bytes(id: &str, weights: &Path) -> CoreResult<u64> {
    let headers = safetensors_path_tensor_headers(weights)?;
    checked_sum(id, headers.iter().map(|header| tensor_bytes(header, 4)))
}

/// Whether [`crate::text_encoder::Flux3TextEncoder::load`] keeps the checkpoint tensor `name`
/// resident: the language tower under `model.language_model.*`, plus the root `lm_head.weight`
/// when the head is untied (the shared `CausalLm` holds it; a tied head is a handle to the
/// embedding). The vision tower (`model.visual.*`) never reaches the device.
pub fn text_encoder_keeps(name: &str, tie_word_embeddings: bool) -> bool {
    name.starts_with("model.language_model.") || (name == "lm_head.weight" && !tie_word_embeddings)
}

/// Resident bytes of the text encoder as the loader builds it, at a `tower_width`-byte dtype,
/// over the exact shard files the loader opens.
pub fn text_encoder_bytes(id: &str, dir: &Path, tower_width: u64) -> CoreResult<u64> {
    let tied = candle_llm::ModelConfig::from_dir(dir)
        .map_err(|e| CoreError::Msg(format!("{id}: text encoder config: {e}")))?
        .tie_word_embeddings;
    let mut total = 0_u64;
    for file in crate::text_encoder::shard_files(dir)? {
        let headers = safetensors_path_tensor_headers(&file)?;
        let part = checked_sum(
            id,
            headers
                .iter()
                .filter(|header| text_encoder_keeps(&header.name, tied))
                .map(|header| tensor_bytes(header, tower_width)),
        )?;
        total = total
            .checked_add(part)
            .ok_or_else(|| CoreError::Msg(format!("{id}: text encoder byte sum overflow")))?;
    }
    Ok(total)
}

/// Bytes per element of the DiT's compute dtype.
const fn compute_width(precision: Precision) -> u32 {
    match precision {
        Precision::Bf16 => 2,
        Precision::Fp32 => 4,
    }
}

/// Architecture axes of the DiT, read from a parsed package `config.json` (the package's own
/// values; the 16× token is the 2×2 patch of the 8× VAE latent). A weights-free surface names a
/// sentinel path with no config to read, so it declares nothing — the shared Candle rule
/// (`MemoryArchitectureFacts::default()` on every weights-free surface) rather than a guessed axis.
fn architecture_facts(
    config: Option<&Flux3Config>,
    precision: Precision,
) -> MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;
    let Some(config) = config else {
        return MemoryArchitectureFacts::default();
    };
    let dit = &config.dit;
    let heads = af::declared(dit.heads);
    MemoryArchitectureFacts {
        attention_heads: heads,
        head_dim: af::head_dim(af::declared(dit.hidden), heads),
        transformer_blocks: af::declared(dit.depth + dit.depth_single),
        patch_size: Some(2),
        latent_channels: af::declared(dit.image_in_channels),
        vae_spatial_scale: Some(8),
        vae_temporal_scale: None,
        activation_dtype_width: Some(compute_width(precision)),
    }
}

// ================================================================================================
// Contracts.
// ================================================================================================

/// The calibration identity of the base CFG route: a distinct graph (no target-time embedder),
/// never a distilled calibration. CFG's [unconditional, conditional] pack is a FIXED x2
/// coefficient of the base route (always two rows, any guidance), carried by this separate
/// identity. It is not a request variable: `count` still renders sequentially, so no BatchCount
/// axis.
const BASE_CALIBRATION_FINGERPRINT: &str = "flux3-klein-base-candle-cfg-source-v1";

/// The declaration every surface shares: route shape, calibration identity and strategy
/// classification, with no asset facts, for the route `id` names.
fn base_contract(
    id: &'static str,
    precision: Precision,
    config: Option<&Flux3Config>,
) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        id,
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            // Rung 4 is not implemented, so nothing materializes a window; the field has no default
            // and must still state what one would do. The loader is an mmap read of dense tensors.
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    contract.load_shape = LoadShape::EagerMaterialization;
    contract.architecture_facts = architecture_facts(config, precision);
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        if id == BASE_MODEL_ID {
            BASE_CALIBRATION_FINGERPRINT
        } else {
            MEMORY_CALIBRATION_FINGERPRINT
        },
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
    let id = route_id(provider_id)?;
    refuse_unloadable_spec(id, spec)?;
    if !spec.adapters.is_empty() {
        return Err(CoreError::Unsupported(
            "flux3: adapters require the real package config for identity/rank admission".into(),
        ));
    }

    let mut contract = base_contract(id, spec.precision, None);
    apply_tier_fingerprint(&mut contract, spec.quantize)?;
    Ok(contract)
}

/// The package root a spec names; a single file is a typed refusal naming the route.
pub(crate) fn package_dir(id: &str, spec: &LoadSpec) -> CoreResult<std::path::PathBuf> {
    match &spec.weights {
        gen_core::WeightsSource::Dir(dir) => Ok(dir.clone()),
        gen_core::WeightsSource::File(file) => Err(CoreError::Unsupported(format!(
            "{id}: the weights resource must be the package directory, not {}",
            file.display()
        ))),
    }
}

/// The header-authoritative checkpoint of the route `id` selects: the package resolved through
/// [`resolve_checkpoint`] (the mode is the DiT header's, never the metadata's) and refused unless
/// it IS that route. Runs before any component, adapter or pricing work.
pub(crate) fn resolve_route_checkpoint(id: &str, package_dir: &Path) -> CoreResult<Flux3Config> {
    let config = resolve_checkpoint(package_dir)?;
    config.validate_route(id)?;
    Ok(config)
}

/// The DiT storage `spec` selects on a resolved package, preflighted header-only BEFORE any
/// text-encoder, autoencoder, adapter, memory or device work, for EVERY storage (dense included):
/// the package marker is exactly the selection (no tier or model fallback), the selection is
/// legal at the spec's precision, the config's graph is one the native graph implements, and the
/// selected checkpoint's full physical inventory (names / dtypes / shapes and the `ref_time_in`
/// pair) is the one that storage and the resolved config imply.
pub(crate) fn preflight_checkpoint(
    id: &str,
    spec: &LoadSpec,
    package_dir: &Path,
    config: &Flux3Config,
) -> CoreResult<NativeDitStorage> {
    let storage = selected_storage(id, spec.quantize)?;
    let package_storage = NativeDitStorage::from_package(package_dir)?;
    if package_storage != storage {
        return Err(CoreError::Unsupported(format!(
            "{id}: selected {} but the quantization marker of {} is {}; install or convert the \
             exact requested native tier (no tier or model fallback)",
            storage.key()?,
            package_dir.display(),
            package_storage.key()?
        )));
    }
    storage.validate_selection(spec.quantize, spec.precision)?;
    if storage == NativeDitStorage::Publisher(DitFormat::Dense) {
        config.validate_supported()?;
    } else {
        config.validate_native_storage_graph()?;
    }
    let headers = gen_core::flux3::checkpoint_tensor_headers(package_dir)?;
    crate::dit::validate_checkpoint_headers(&config.dit, storage, &headers)?;
    Ok(storage)
}

/// The executable contract for a real load: the declaration plus the resident bytes of what the
/// loaders build, read from the resources `spec` names.
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let id = route_id(provider_id)?;
    refuse_unloadable_spec(id, spec)?;
    let package_dir = package_dir(id, spec)?;
    let config = resolve_route_checkpoint(id, &package_dir)?;
    preflight_checkpoint(id, spec, &package_dir, &config)?;
    let resources = GenerationResources::from_spec(spec, &config, id)?;
    generation_contract(id, spec, &resources, &config)
}

/// The contract for already-resolved resources (what [`crate::model::load`] holds), for the
/// route `provider_id` names (already checked against the resolved checkpoint by the caller).
pub(crate) fn generation_contract(
    provider_id: &str,
    spec: &LoadSpec,
    resources: &GenerationResources,
    config: &Flux3Config,
) -> CoreResult<MemoryProviderContract> {
    let id = route_id(provider_id)?;
    refuse_unloadable_spec(id, spec)?;
    let precision = spec.precision;
    let mut contract = base_contract(id, precision, Some(config));
    for adapter in &spec.adapters {
        gen_core::flux3::adapters::inspect(adapter, &config.dit)?;
    }

    let storage = selected_storage(id, spec.quantize)?;
    let package_storage = NativeDitStorage::from_package(&resources.package_dir)?;
    if package_storage != storage {
        return Err(CoreError::Unsupported(format!(
            "{id}: selected {} but the package quantization marker is {}; install the \
             exact requested native tier (no tier or model fallback)",
            storage.key()?,
            package_storage.key()?
        )));
    }
    storage.validate_selection(spec.quantize, precision)?;
    apply_tier_fingerprint(&mut contract, spec.quantize)?;
    let transformer = native_dit_bytes(
        id,
        &resources.dit_weights(),
        precision,
        storage,
        &config.dit,
    )?;
    let conditioning = text_encoder_bytes(
        id,
        &resources.text_encoder_dir,
        production_tower_dtype(&config.text_encoder)?.size_in_bytes() as u64,
    )?;
    let decoder = autoencoder_bytes(id, &resources.autoencoder_file)?;
    let base_bytes = conditioning
        .checked_add(transformer)
        .and_then(|b| b.checked_add(decoder))
        .ok_or_else(|| CoreError::Msg(format!("{id}: base component byte sum overflow")))?;
    contract.asset_facts = MemoryAssetFacts {
        base_bytes,
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: decoder,
        overlay_bytes: 0,
    };
    if let Some(dir) = gen_core::flux3::pup::UpsamplePackage::component_dir(spec)? {
        account_upsampler(
            id,
            &mut contract,
            crate::pup::load_memory_bytes(&dir)?,
            upsampler_residency(spec.offload_policy),
        )?;
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

/// A `Sequential` load always runs the PUP slot exclusively: the bounded VLM is fenced and its
/// actual owner dropped before the text encoder loads, so it is priced as consumed during
/// conditioning with nothing retained. A `Resident` load may keep it warm across requests and
/// stays conservatively whole-render (a request-admitted staging only lowers the real peak).
pub(crate) fn upsampler_residency(
    policy: gen_core::OffloadPolicy,
) -> gen_core::MemoryComponentResidency {
    match policy {
        gen_core::OffloadPolicy::Resident => gen_core::MemoryComponentResidency::WholeRender,
        gen_core::OffloadPolicy::Sequential => {
            gen_core::MemoryComponentResidency::PrecomputedThenEvicted {
                precomputed_in: gen_core::MemoryPhase::Conditioning,
                retained_bytes: 0,
                evidence: "candle_gen_flux3::model::tests::actual_generator_lazy_pup_bypass_and_exclusive_ownership_are_honest; \
                           gen_core::flux3::phase_resource::tests::exclusive_work_fences_and_really_drops_before_the_next_phase"
                    .into(),
            }
        }
    }
}

/// Keep the optional VLM separate from the image encoder. `bytes` is its native header-only
/// load bound (prefix cache and CUDA graphs disabled), NOT measured KV/activation peak memory;
/// the LLM admits its request-time KV separately.
fn account_upsampler(
    id: &str,
    contract: &mut MemoryProviderContract,
    bytes: u64,
    residency: gen_core::MemoryComponentResidency,
) -> CoreResult<()> {
    if bytes == 0 {
        return Err(CoreError::Unsupported(format!(
            "{id}: prompt_upsampler has no accountable weights; install the complete package"
        )));
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

/// The provider safety check: the shared handshake (calibration identity, loaded tier,
/// selection, budget), then the route's own gate. It can reject, never swap in a different
/// strategy or tier.
pub fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let id = contract.provider_id.as_str();
    let route_gate = || {
        gen_core::flux3::references::validate_memory_route(
            &context.mode,
            context.geometry.reference_count,
        )?;
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{id}: there is no PiD decoder on this route"
            )));
        }
        if context.geometry.frames > 1 {
            return Err(CoreError::Unsupported(format!(
                "{id}: a still-image route cannot admit {} frames",
                context.geometry.frames
            )));
        }
        Ok(())
    };
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(loaded_tier(spec)),
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
        loaded_tier(spec),
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

/// The executable admission seam: the shared Candle request scope, whose `configure_request`
/// writes the admitted `GenerationMemory` onto the request that [`crate::model::Flux3Klein`]
/// then honours.
pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(spec, contract, context) {
        return Err(CoreError::Unsupported(reason));
    }
    let id = route_id(&contract.provider_id)?;
    let config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        id,
        candle_gen::default_device()?,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        // Rung 4 is not selectable, so no window is ever checked against this count; a resolved
        // load states its block depth, a weights-free surface has none to state.
        contract
            .architecture_facts
            .transformer_blocks
            .unwrap_or_default() as usize,
        move |_use_pid, _edge, _overlap| {
            Err(CoreError::Unsupported(format!(
                "{id}: {DECODE_NOT_ENABLED}"
            )))
        },
    )?;
    Ok(Some(
        gen_core::flux3::references::guard_memory_scope_with_resources(
            Box::new(candle_gen::request_scope::CandleRequestScopeCore::new(
                config,
            )),
            context.geometry,
            context.reference_resources.clone(),
        ),
    ))
}

/// The registry-load surface the route witnesses: the dense tier and the two native GGML tiers
/// (q4 = Q4_1, q8 = Q8_0), eagerly materialized, under both offload policies. NVFP4 and the
/// deferred load shape have no Candle FLUX3 loader, so neither is witnessed.
pub fn memory_contract_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    // The CUDA build additionally witnesses the publisher FP8R/NVFP4 tiers it loads as shipped.
    #[cfg(feature = "cuda")]
    let surfaces = gen_core::candle_publisher_memory_contract_surface_specs();
    #[cfg(not(feature = "cuda"))]
    let surfaces = gen_core::candle_memory_contract_surface_specs();
    surfaces
        .into_iter()
        .filter(|surface| {
            selected_storage(MODEL_ID, surface.spec.quantize).is_ok()
                && surface.selector.load_shape == LoadShape::EagerMaterialization
        })
        .collect()
}

/// PRE-FIT reference workspace of one request on the package `spec` names, through the shared
/// formula ([`gen_core::flux3::reference_workspace`]). On the Candle CPU backend every DiT GEMM
/// runs over f32 temporaries whatever the compute dtype, so the per-token term is priced at f32.
fn reference_workspace_for(
    provider_id: &str,
    spec: &LoadSpec,
    facts: &gen_core::MemoryReferenceResourceFacts,
) -> CoreResult<u64> {
    refuse_unloadable_spec(route_id(provider_id)?, spec)?;
    let width = if production_device_is_cpu() {
        4
    } else {
        u64::from(compute_width(spec.precision))
    };
    let inputs = gen_core::flux3::reference_workspace::ReferenceWorkspaceInputs::from_spec(
        spec,
        provider_id,
        width,
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
    reference_workspace_for(BASE_MODEL_ID, spec, facts)
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
    use candle_gen::gen_core::{Quant, WeightsSource};

    /// A spec no loader could open: the weights-free surface.
    #[test]
    fn base_identity_and_adapter_overlay_are_distinct_and_counted_once() {
        let empty = sentinel();
        let distilled = weights_free_memory_strategy_contract(MODEL_ID, &empty).unwrap();
        let base =
            weights_free_memory_strategy_contract(gen_core::flux3::BASE_MODEL_ID, &empty).unwrap();
        assert_ne!(base.provider_id, distilled.provider_id);
        assert_ne!(base.calibration, distilled.calibration);
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mlx-gen/mlx-gen-flux3/tests/fixtures");
        // The real tiny CFG package (its DiT header carries no `ref_time_in`), with the distilled
        // package's encoder and autoencoder as components.
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
            resolve_route_checkpoint(BASE_MODEL_ID, &fixture.join("tiny-base-snapshot")).unwrap();
        let resources = GenerationResources::from_spec(&spec, &config, BASE_MODEL_ID).unwrap();
        let plain = generation_contract(BASE_MODEL_ID, &spec, &resources, &config).unwrap();
        assert_eq!(plain.provider_id, BASE_MODEL_ID);
        assert_eq!(plain.calibration, base.calibration);
        spec.adapters.push(gen_core::AdapterSpec::new(
            fixture.join("flux3_base_lora.safetensors"),
            0.4,
            gen_core::AdapterKind::Lora,
        ));
        let adapted = generation_contract(BASE_MODEL_ID, &spec, &resources, &config).unwrap();
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
        let both = generation_contract(BASE_MODEL_ID, &spec, &resources, &config).unwrap();
        spec.adapters.clear();
        let pup = generation_contract(BASE_MODEL_ID, &spec, &resources, &config).unwrap();
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
    #[test]
    fn optional_upsampler_is_a_separate_whole_render_component() {
        let mut contract = base_contract(MODEL_ID, Precision::Bf16, None);
        contract.asset_facts.base_bytes = 113;
        contract.asset_facts.conditioning_bytes = 17;
        account_upsampler(
            MODEL_ID,
            &mut contract,
            37,
            upsampler_residency(gen_core::OffloadPolicy::Resident),
        )
        .unwrap();
        assert_eq!(contract.asset_facts.base_bytes, 113);
        assert_eq!(contract.asset_facts.conditioning_bytes, 17);
        assert_eq!(contract.asset_facts.overlay_bytes, 37);
        let MemoryFormulaKind::ComponentPhaseEnvelope {
            resident_components,
            variables,
            ..
        } = &contract.formula
        else {
            panic!("optional model must be included in the component envelope");
        };
        assert!(variables.contains(&MemoryFormulaVariable::OverlayBytes));
        assert_eq!(resident_components.len(), 1);
        assert_eq!(resident_components[0].id, "prompt_upsampler");
        assert_eq!(resident_components[0].resident_bytes, 37);
        assert_eq!(
            resident_components[0].kind,
            gen_core::MemoryComponentKind::PromptUpsampler
        );
        assert_eq!(
            resident_components[0].residency,
            gen_core::MemoryComponentResidency::WholeRender
        );
        assert_eq!(contract.auxiliary_resident_bytes(), 37);
        assert!(account_upsampler(
            MODEL_ID,
            &mut contract,
            0,
            upsampler_residency(gen_core::OffloadPolicy::Resident)
        )
        .is_err());

        // Sequential: the exclusive slot drops PUP before conditioning work, so the bound is the
        // conditioning-phase head only and NO later phase is charged for it.
        let mut staged = base_contract(MODEL_ID, Precision::Bf16, None);
        staged.asset_facts.base_bytes = 113;
        staged.asset_facts.conditioning_bytes = 17;
        staged.asset_facts.transformer_bytes = 96;
        account_upsampler(
            MODEL_ID,
            &mut staged,
            37,
            upsampler_residency(gen_core::OffloadPolicy::Sequential),
        )
        .unwrap();
        let component = &staged.resident_components()[0];
        assert_eq!(component.resident_bytes, 37);
        assert_eq!(component.steady_state_bytes(), 0);
        assert_eq!(component.evicted_bytes(), 37);
        assert!(matches!(
            component.residency,
            gen_core::MemoryComponentResidency::PrecomputedThenEvicted {
                precomputed_in: gen_core::MemoryPhase::Conditioning,
                retained_bytes: 0,
                ..
            }
        ));
        assert!(
            staged.conformance_errors().is_empty(),
            "{:?}",
            staged.conformance_errors()
        );
    }

    /// The pre-fit reference pricer through the real catalog: zero without references, priced
    /// and monotone with them, and the base CFG route charges its reference tokens in both rows.
    ///
    /// *Mutations that red this:* not registering the pricer, pricing `None` as non-zero, or the
    /// base route pricing one CFG row.
    /// Publisher packages price codes as stored, `weight_scale` at its STAGED width (FP8R f32 row
    /// scales 4 B, NVFP4 E4M3 block scales 1 B), every other matrix at the compute width and
    /// vectors at f32, plus the one resident packed-launch cap.
    ///
    /// *Mutations that red this:* pricing FP8R scales at the compute width, or NVFP4 block scales
    /// as a float matrix.
    #[test]
    fn publisher_scales_price_at_their_staged_width() {
        use gen_core::flux3::safetensors_stream::{PlannedTensor, SafetensorsStreamWriter};
        use gen_core::weightsmeta::Dtype;
        let cfg = resolve_checkpoint(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../mlx-gen/mlx-gen-flux3/tests/fixtures/tiny-snapshot"),
        )
        .unwrap()
        .dit;
        let dir = tempfile::tempdir().unwrap();
        for format in [DitFormat::Fp8Rowwise, DitFormat::Nvfp4] {
            let storage = NativeDitStorage::Publisher(format);
            let expected = storage.expected_tensors(&cfg).unwrap();
            let path = dir.path().join(format!("{format:?}.safetensors"));
            let planned: Vec<PlannedTensor> = expected
                .iter()
                .map(|(name, spec)| PlannedTensor {
                    name: name.clone(),
                    dtype: format!("{:?}", spec.dtype),
                    shape: spec.shape.clone(),
                })
                .collect();
            let mut writer =
                SafetensorsStreamWriter::create(&path, planned, &Default::default()).unwrap();
            for (name, spec) in &expected {
                writer
                    .write(name, &vec![0u8; spec.bytes().unwrap() as usize])
                    .unwrap();
            }
            writer.finish().unwrap();
            let targets = storage.packed_targets(&cfg).unwrap();
            let want: u64 = expected
                .iter()
                .map(|(name, spec)| {
                    let elements: u64 = spec.shape.iter().map(|&d| d as u64).product();
                    if targets.contains(name) {
                        spec.bytes().unwrap()
                    } else if name.ends_with(".weight_scale") {
                        elements
                            * if format == DitFormat::Fp8Rowwise {
                                4
                            } else {
                                1
                            }
                    } else if spec.dtype == Dtype::BF16 && spec.shape.len() >= 2 {
                        elements * 2
                    } else if spec.dtype == Dtype::BF16 || spec.dtype == Dtype::F32 {
                        elements * 4
                    } else {
                        spec.bytes().unwrap()
                    }
                })
                .sum::<u64>()
                + gen_core::flux3::packed_gemm::PACKED_GEMM_TRANSIENT_BUDGET;
            assert_eq!(
                native_dit_bytes(MODEL_ID, &path, Precision::Bf16, storage, &cfg).unwrap(),
                want,
                "{format:?}"
            );
        }
    }

    #[test]
    fn registered_reference_pricer_is_additive_and_monotone_on_both_routes() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mlx-gen/mlx-gen-flux3/tests/fixtures");
        let registry = crate::provider_registry().unwrap();
        let facts = |canvases: &[(u32, u32)]| gen_core::MemoryReferenceResourceFacts {
            conditioning_tokens: canvases
                .iter()
                .map(|&(w, h)| u64::from(w / 16) * u64::from(h / 16))
                .sum(),
            prepared_canvases: canvases.to_vec(),
        };
        let mut per_route = Vec::new();
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
            let big = registry
                .reference_workspace_bytes(id, &spec, Some(&facts(&[(128, 128)])))
                .unwrap()
                .unwrap();
            assert!(one > 0 && two > one && big > one, "{id}: {one} {two} {big}");
            per_route.push(two - one);
        }
        // Same tiny DiT widths: the base route's per-reference DiT term is at least the
        // distilled one's (two CFG rows vs one).
        assert!(per_route[1] > per_route[0], "{per_route:?}");
    }

    #[test]
    fn the_route_declares_exactly_the_rungs_its_engine_runs() {
        let surfaces = memory_contract_surface_specs();
        // Dense + GGML q4/q8 everywhere; the CUDA build adds the publisher fp8r/nvfp4 tiers.
        let mut tiers = vec!["bf16", "q4", "q8"];
        if cfg!(feature = "cuda") {
            tiers.extend(["fp8r", "nvfp4"]);
        }
        let mut want: Vec<String> = tiers
            .iter()
            .flat_map(|t| {
                [
                    format!("{t}:resident:eager"),
                    format!("{t}:sequential:eager"),
                ]
            })
            .collect();
        want.sort();
        let mut selectors: Vec<String> = surfaces
            .iter()
            .map(|s| s.selector.id().to_string())
            .collect();
        selectors.sort();
        assert_eq!(selectors, want, "eager tiers x 2 offload policies");
        for surface in &surfaces {
            let contract = weights_free_memory_strategy_contract(MODEL_ID, &surface.spec).unwrap();
            gen_core_testkit::memory_strategy_conformance(&contract);
            assert_eq!(contract.provider_id, MODEL_ID);
            let calibration = contract.calibration.as_ref().unwrap();
            assert_eq!(
                calibration.fingerprint,
                tier_fingerprint(
                    MODEL_ID,
                    MEMORY_CALIBRATION_FINGERPRINT,
                    surface.spec.quantize
                )
                .unwrap()
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
            assert!(matches!(
                contract.backend,
                MemoryBackendRealization::CandleCuda { .. }
            ));
        }
    }

    #[test]
    fn fingerprint_is_valid_and_not_the_mlx_twins() {
        gen_core::validate_calibration_fingerprint(MEMORY_CALIBRATION_FINGERPRINT).unwrap();
        assert!(MEMORY_CALIBRATION_FINGERPRINT.contains("candle"));
    }

    /// *Mutation that reds this:* dropping any arm of `refuse_unloadable_spec`, or advertising
    /// the publisher FP8R/NVFP4 tiers off the CUDA build (or refusing them on it).
    #[test]
    fn a_spec_no_loader_opens_has_no_contract() {
        let mut adapters = sentinel();
        adapters.adapters.push(gen_core::AdapterSpec {
            path: "/lora-not-on-disk".into(),
            scale: 1.0,
            kind: gen_core::AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        });
        let mut refused = vec![
            sentinel().with_load_shape(LoadShape::DeferredMaterialization),
            adapters,
        ];
        let publisher = [
            sentinel().with_quant(Quant::Nvfp4),
            sentinel().with_quant(Quant::Fp8Rowwise),
        ];
        // The publisher tiers are a CUDA-build capability (loaded as shipped onto their kernels):
        // a weights-free contract on CUDA, a typed refusal everywhere else.
        if cfg!(feature = "cuda") {
            for id in [MODEL_ID, BASE_MODEL_ID] {
                for spec in &publisher {
                    let contract = weights_free_memory_strategy_contract(id, spec).unwrap();
                    assert_eq!(contract.provider_id, id);
                }
            }
        } else {
            refused.extend(publisher);
        }
        for id in [MODEL_ID, BASE_MODEL_ID] {
            for spec in &refused {
                match weights_free_memory_strategy_contract(id, spec) {
                    Err(CoreError::Unsupported(_)) => {}
                    other => panic!("{id}: expected a typed refusal, got {other:?}"),
                }
            }
        }
    }

    /// The executable contract resolves the checkpoint from its DiT HEADER and refuses the wrong
    /// route, and a dense package whose physical inventory is not the resolved graph's, BEFORE
    /// any text-encoder / autoencoder resource is resolved or priced (neither package below names
    /// one: reaching that step would be a missing-resource `Msg`, not a typed refusal).
    ///
    /// *Mutations that red this:* dropping `validate_route`, resolving the route from
    /// `config.json` (both real packages carry the same metadata), skipping the dense inventory
    /// preflight, or running it after `GenerationResources::from_spec`.
    #[test]
    fn route_and_dense_inventory_are_refused_before_component_work() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mlx-gen/mlx-gen-flux3/tests/fixtures");
        for (id, dir) in [
            (BASE_MODEL_ID, "tiny-snapshot"),
            (MODEL_ID, "tiny-base-snapshot"),
        ] {
            let spec = LoadSpec::new(WeightsSource::Dir(fixtures.join(dir)));
            match memory_strategy_contract(id, &spec) {
                Err(CoreError::Unsupported(m)) => assert!(m.contains("flow_map"), "{id}: {m}"),
                other => panic!("{id} on {dir}: expected a typed route refusal, got {other:?}"),
            }
        }
        // A base package whose DiT lacks one projection the resolved graph needs.
        let dir = tempfile::tempdir().unwrap();
        let base = fixtures.join("tiny-base-snapshot");
        std::fs::copy(base.join("config.json"), dir.path().join("config.json")).unwrap();
        let mut tensors = candle_gen::candle_core::safetensors::load(
            base.join("model.safetensors"),
            &candle_gen::candle_core::Device::Cpu,
        )
        .unwrap();
        assert!(tensors.remove("final_layer.image.linear.weight").is_some());
        candle_gen::candle_core::safetensors::save(&tensors, dir.path().join("model.safetensors"))
            .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(dir.path().to_path_buf()));
        match memory_strategy_contract(BASE_MODEL_ID, &spec) {
            Err(CoreError::Unsupported(m)) => {
                assert!(m.contains("final_layer.image.linear.weight"), "{m}")
            }
            other => panic!("expected a typed inventory refusal, got {other:?}"),
        }
    }

    /// The route gate rejects a reference-carrying, PiD or multi-frame context by name.
    #[test]
    fn route_gate_rejects_foreign_requests() {
        let contract = weights_free_memory_strategy_contract(MODEL_ID, &sentinel()).unwrap();
        let mut context = gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::StagedResidency,
            loaded_tier(&sentinel()),
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
        context.geometry.reference_count = 0;
        context.use_pid = true;
        match safety_check(&sentinel(), &contract, &context) {
            MemorySafetyDecision::Reject { reason } => assert!(reason.contains("PiD"), "{reason}"),
            other => panic!("expected a rejection, got {other:?}"),
        }
    }

    /// The architecture axes are read off the package's `config.json` (the miniature fixture's
    /// widths, not the release's), the weights-free surface declares none, and the executable
    /// contract prices every component the loaders build from the resources on disk.
    ///
    /// *Mutation that reds this:* returning `MemoryArchitectureFacts::default()` for a parsed
    /// config, or pricing the DiT at f32 under the bf16 tier.
    #[test]
    fn architecture_facts_are_read_from_the_package_config() {
        let package = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../mlx-gen/mlx-gen-flux3/tests/fixtures/tiny-snapshot");
        let spec = LoadSpec::new(WeightsSource::Dir(package.clone()));
        let weights_free = weights_free_memory_strategy_contract(MODEL_ID, &spec).unwrap();
        assert_eq!(
            weights_free.architecture_facts,
            MemoryArchitectureFacts::default()
        );
        let contract = memory_strategy_contract(MODEL_ID, &spec).unwrap();
        let facts = &contract.architecture_facts;
        assert_eq!(facts.attention_heads, Some(2));
        assert_eq!(facts.head_dim, Some(32));
        assert_eq!(facts.transformer_blocks, Some(5));
        assert_eq!(facts.patch_size, Some(2));
        assert_eq!(facts.latent_channels, Some(16));
        assert_eq!(facts.vae_spatial_scale, Some(8));
        assert_eq!(facts.vae_temporal_scale, None);
        assert_eq!(facts.activation_dtype_width, Some(2));
        let assets = &contract.asset_facts;
        assert!(assets.transformer_bytes > 0 && assets.decoder_bytes > 0);
        assert!(assets.conditioning_bytes > 0);
        assert_eq!(
            assets.base_bytes,
            assets.conditioning_bytes + assets.transformer_bytes + assets.decoder_bytes
        );
        assert_eq!(assets.overlay_bytes, 0);
        // The DiT is priced at the compute width: the f32 tier doubles its rank-2 tensors.
        let mut fp32 = spec.clone();
        fp32.precision = Precision::Fp32;
        let wide = memory_strategy_contract(MODEL_ID, &fp32).unwrap();
        assert_eq!(wide.architecture_facts.activation_dtype_width, Some(4));
        assert!(wide.asset_facts.transformer_bytes > assets.transformer_bytes);
        assert_eq!(wide.asset_facts.decoder_bytes, assets.decoder_bytes);
    }

    /// The language-tower filter keeps exactly what the loader uploads.
    #[test]
    fn text_encoder_pricing_keeps_the_language_tower_only() {
        assert!(text_encoder_keeps(
            "model.language_model.layers.0.mlp.up_proj.weight",
            true
        ));
        assert!(!text_encoder_keeps(
            "model.visual.blocks.0.attn.qkv.weight",
            true
        ));
        assert!(!text_encoder_keeps("lm_head.weight", true));
        assert!(text_encoder_keeps("lm_head.weight", false));
    }
}
