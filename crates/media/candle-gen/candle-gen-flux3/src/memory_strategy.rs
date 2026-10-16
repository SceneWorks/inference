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
use candle_gen::gen_core::flux3::{Flux3Config, GenerationResources, TextEncoderConfig};
use candle_gen::gen_core::weightsmeta::{safetensors_path_tensor_headers, SafetensorsTensorHeader};
use candle_gen::gen_core::{
    self, Error as CoreError, LoadShape, LoadSpec, MemoryArchitectureFacts, MemoryAssetFacts,
    MemoryBackendRealization, MemoryBehaviorFixture, MemoryBehaviorRoute,
    MemoryCalibrationIdentity, MemoryContractSurfaceSpec, MemoryContractSurfaceTier,
    MemoryFormulaKind, MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode,
    MemoryNumericTier, MemoryPhase, MemoryProviderContract, MemoryRequestScope, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategySupport, MemoryWindowMaterialization,
    Precision, ResidentRequestMemory, Result as CoreResult,
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

/// The numeric tier a spec loads: the precision it names and no quantization (the loader refuses
/// `LoadSpec::quantize`). No component is resident above the selected width.
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
fn refuse_unloadable_spec(spec: &LoadSpec) -> CoreResult<()> {
    refuse_deferred_load_shape(MODEL_ID, spec)?;
    if spec.quantize.is_some() {
        return Err(CoreError::Unsupported(format!(
            "{MODEL_ID}: the distilled route ships one dense tier; LoadSpec::quantize is refused \
             at load (sc-26241), so there is no quantized memory contract"
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(CoreError::Unsupported(format!(
            "{MODEL_ID}: LoadSpec::adapters is refused at load (sc-26239), so there is no adapter \
             memory contract"
        )));
    }
    Ok(())
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

fn checked_sum(parts: impl Iterator<Item = CoreResult<u64>>) -> CoreResult<u64> {
    let mut sum = 0_u64;
    for part in parts {
        sum = sum
            .checked_add(part?)
            .ok_or_else(|| CoreError::Msg(format!("{MODEL_ID}: component byte sum overflow")))?;
    }
    Ok(sum)
}

/// Resident bytes of the DiT as [`crate::model::load_dit`] holds it: every tensor of rank two or
/// more at the compute dtype; vectors (the q/k norm scales) in f32.
pub fn dit_bytes(weights: &Path, precision: Precision) -> CoreResult<u64> {
    let compute_width = compute_width(precision) as u64;
    let headers = safetensors_path_tensor_headers(weights)?;
    checked_sum(headers.iter().map(|header| {
        tensor_bytes(
            header,
            if header.shape.len() >= 2 {
                compute_width
            } else {
                4
            },
        )
    }))
}

/// Resident bytes of the autoencoder as [`crate::model::load_autoencoder`] holds it: f32.
pub fn autoencoder_bytes(weights: &Path) -> CoreResult<u64> {
    let headers = safetensors_path_tensor_headers(weights)?;
    checked_sum(headers.iter().map(|header| tensor_bytes(header, 4)))
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
pub fn text_encoder_bytes(dir: &Path, tower_width: u64) -> CoreResult<u64> {
    let tied = candle_llm::ModelConfig::from_dir(dir)
        .map_err(|e| CoreError::Msg(format!("{MODEL_ID}: text encoder config: {e}")))?
        .tie_word_embeddings;
    let mut total = 0_u64;
    for file in crate::text_encoder::shard_files(dir)? {
        let headers = safetensors_path_tensor_headers(&file)?;
        let part = checked_sum(
            headers
                .iter()
                .filter(|header| text_encoder_keeps(&header.name, tied))
                .map(|header| tensor_bytes(header, tower_width)),
        )?;
        total = total
            .checked_add(part)
            .ok_or_else(|| CoreError::Msg(format!("{MODEL_ID}: text encoder byte sum overflow")))?;
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

/// The declaration every surface shares: route shape, calibration identity and strategy
/// classification, with no asset facts.
fn base_contract(precision: Precision, config: Option<&Flux3Config>) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        MODEL_ID,
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
        MEMORY_CALIBRATION_FINGERPRINT,
        LoadShape::EagerMaterialization,
    ));
    // A `Resident` selection leaves the request memory block absent, which keeps exactly the
    // loaded policy: warm for a `Resident` load, staged for a `Sequential` one.
    contract.resident_request_memory = ResidentRequestMemory::PreserveLoadDefaults;
    // After the conditioning release the DiT + autoencoder are the one heavy bundle, through the
    // decode.
    contract.phase_facts = Some(gen_core::MemoryPhaseFacts::staged(
        gen_core::StagedWeightSchedule::TwoStage,
    ));
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
    if provider_id != MODEL_ID {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: not the FLUX 3 Candle memory provider"
        )));
    }
    refuse_unloadable_spec(spec)?;
    Ok(base_contract(spec.precision, None))
}

/// The executable contract for a real load: the declaration plus the resident bytes of what the
/// loaders build, read from the resources `spec` names.
pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    if provider_id != MODEL_ID {
        return Err(CoreError::Unsupported(format!(
            "{provider_id}: not the FLUX 3 Candle memory provider"
        )));
    }
    refuse_unloadable_spec(spec)?;
    let package_dir = match &spec.weights {
        gen_core::WeightsSource::Dir(dir) => dir.clone(),
        gen_core::WeightsSource::File(file) => {
            return Err(CoreError::Msg(format!(
                "{MODEL_ID}: the weights resource must be the package directory, not {}",
                file.display()
            )))
        }
    };
    let config = Flux3Config::from_dir(&package_dir)?;
    let resources = GenerationResources::from_spec(spec, &config, MODEL_ID)?;
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
    let transformer = dit_bytes(&resources.dit_weights(), precision)?;
    let conditioning = text_encoder_bytes(
        &resources.text_encoder_dir,
        production_tower_dtype(&config.text_encoder)?.size_in_bytes() as u64,
    )?;
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
        account_upsampler(&mut contract, crate::pup::load_memory_bytes(&dir)?)?;
    }
    Ok(contract)
}

/// Keep the optional VLM separate from the image encoder. This is its conservative native
/// source/load bound held throughout the render, NOT measured KV/activation peak memory.
fn account_upsampler(contract: &mut MemoryProviderContract, bytes: u64) -> CoreResult<()> {
    if bytes == 0 {
        return Err(CoreError::Unsupported(format!(
            "{MODEL_ID}: prompt_upsampler has no accountable weights; install the complete package"
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
            residency: gen_core::MemoryComponentResidency::WholeRender,
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
    let route_gate = || {
        if context.mode != MemoryMode::TextToImage {
            return Err(CoreError::Unsupported(format!(
                "{MODEL_ID}: this route's request surface is text_to_image; got {:?}",
                context.mode.as_key()
            )));
        }
        if context.geometry.reference_count > 0 {
            return Err(CoreError::Unsupported(format!(
                "{MODEL_ID}: the distilled text-to-image route takes no reference images \
                 (sc-26236)"
            )));
        }
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
    let config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        MODEL_ID,
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
        |_use_pid, _edge, _overlap| {
            Err(CoreError::Unsupported(format!(
                "{MODEL_ID}: {DECODE_NOT_ENABLED}"
            )))
        },
    )?;
    Ok(Some(Box::new(
        candle_gen::request_scope::CandleRequestScopeCore::new(config),
    )))
}

/// The registry-load surface the route witnesses: the one dense tier, eagerly materialized,
/// under both offload policies. No loader accepts a quantized tier or has a deferred load shape,
/// so neither is witnessed — a selector no load can take would be a witness-only surface.
pub fn memory_contract_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    gen_core::candle_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| {
            surface.selector.tier == MemoryContractSurfaceTier::Bf16
                && surface.selector.load_shape == LoadShape::EagerMaterialization
        })
        .collect()
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{Quant, WeightsSource};

    /// A spec no loader could open: the weights-free surface.
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
        let mut contract = base_contract(Precision::Bf16, None);
        contract.asset_facts.base_bytes = 113;
        contract.asset_facts.conditioning_bytes = 17;
        account_upsampler(&mut contract, 37).unwrap();
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
        assert!(account_upsampler(&mut contract, 0).is_err());
    }

    #[test]
    fn the_route_declares_exactly_the_rungs_its_engine_runs() {
        let surfaces = memory_contract_surface_specs();
        assert_eq!(
            surfaces.len(),
            2,
            "one dense tier, eager, x 2 offload policies"
        );
        let selectors: Vec<_> = surfaces.iter().map(|s| s.selector.id()).collect();
        assert_eq!(selectors, ["bf16:resident:eager", "bf16:sequential:eager"]);
        for surface in &surfaces {
            let contract = weights_free_memory_strategy_contract(MODEL_ID, &surface.spec).unwrap();
            gen_core_testkit::memory_strategy_conformance(&contract);
            assert_eq!(contract.provider_id, MODEL_ID);
            let calibration = contract.calibration.as_ref().unwrap();
            assert_eq!(calibration.fingerprint, MEMORY_CALIBRATION_FINGERPRINT);
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

    /// *Mutation that reds this:* dropping any arm of `refuse_unloadable_spec`.
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
        for spec in [
            sentinel().with_quant(Quant::Q4),
            sentinel().with_load_shape(LoadShape::DeferredMaterialization),
            adapters,
        ] {
            match weights_free_memory_strategy_contract(MODEL_ID, &spec) {
                Err(CoreError::Unsupported(_)) => {}
                other => panic!("expected a typed refusal, got {other:?}"),
            }
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
