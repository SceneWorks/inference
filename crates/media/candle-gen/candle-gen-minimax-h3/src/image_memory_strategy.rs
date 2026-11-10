//! The **`minimax_h3_image`** memory contract on candle/CUDA (SW-7091).
//!
//! | rung | strategy | status |
//! |---|---|---|
//! | 0 | [`MemoryStrategy::Resident`] | **implemented** as the baseline — the render never exceeds it: the generator always stages (as H3's does), so a resident admission over-provisions rather than under-provisions |
//! | 1 | [`MemoryStrategy::StagedResidency`] | **implemented** — `crate::image_model` encodes the prompt, drops the conditioner and synchronizes, maps the DiT, evicts `adaln_proj`, denoises, drops it, then maps the VAE |
//! | 2 | [`MemoryStrategy::BoundedDecode`] | **implemented** — the 2-D VAE's shipped 256 / 64 tiled decode ([`crate::image_vae::bounded_decode_parameters`]) |
//! | 3 | [`MemoryStrategy::BoundedAttention`] | **missing** — not wired for this family yet |
//! | 4 | [`MemoryStrategy::BoundedTransformerResidency`] | **missing** — not wired for this family yet |
//!
//! Nothing here declares a rung the generator does not run: the decode always tiles at the
//! shipped geometry (it is an output, not a lever — see [`crate::image_vae`]), so the bounded
//! decode's parameter domain is that single point. No calibration is claimed.

use std::path::Path;

use candle_gen::gen_core::{
    self, LoadSpec, MemoryAssetFacts, MemoryBackendRealization, MemoryCalibrationIdentity,
    MemoryFixturePhases, MemoryFixtureRoute, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProfileFixtures, MemoryProfileGates, MemoryProviderContract, MemoryRequestScope,
    MemoryRunContext, MemorySafetyDecision, MemoryStrategy, MemoryStrategySupport,
    MemoryWindowMaterialization, ProviderMemoryProfile, ResidentRequestMemory, WeightsSource,
};

use crate::image_model::IMAGE_MODEL_ID;
use crate::image_vae::{IMAGE_DECODE_OVERLAP, IMAGE_DECODE_TILE_EDGE, IMAGE_VAE_COMPONENT};

/// Content fingerprint of this contract.
pub const IMAGE_MEMORY_CALIBRATION_FINGERPRINT: &str =
    "minimax-h3-image-candle-staged-bounded-decode-2026-11-10-v1";

/// Width every float component is held at: bf16 for the conditioner and the DiT block store. The
/// DiT's float32 top-level heads and the float32 VAE are priced at their stored width.
const BF16_WIDTH: u64 = 2;

/// Bytes the resolved tensors of one component occupy once loaded. Float tensors price at
/// `float_width` (or their stored width when `None`); `keep` selects the tensors the loader opens.
fn component_bytes(
    path: &Path,
    float_width: Option<u64>,
    keep: &dyn Fn(&str) -> bool,
) -> gen_core::Result<u64> {
    gen_core::weightsmeta::safetensors_path_tensor_headers(path)?
        .iter()
        .filter(|header| keep(&header.name))
        .try_fold(0_u64, |sum, header| {
            let bytes = match float_width {
                Some(width) if header.is_float() => header.materialized_bytes(width)?,
                _ => header.data_bytes,
            };
            sum.checked_add(bytes).ok_or_else(|| {
                gen_core::Error::Msg(format!("{IMAGE_MODEL_ID}: component byte sum overflow"))
            })
        })
}

/// The load-exact inventory: the conditioner's layer-50 tap (the `lm_prefixes` the render maps)
/// at bf16, the DiT at its stored mixed bf16/f32 widths, and the VAE's decode half at its stored
/// f32.
fn asset_facts(dit_dir: &Path, root: &Path, te_dir: &Path) -> gen_core::Result<MemoryAssetFacts> {
    let te_cfg = crate::text_encoder::MiniMaxH3TeConfig::from_component_dir(te_dir)
        .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
    let prefixes = crate::text_encoder::lm_prefixes(crate::text_encoder::LM_PREFIX, &te_cfg);
    let conditioning = component_bytes(te_dir, Some(BF16_WIDTH), &|name: &str| {
        prefixes.iter().any(|p| name.starts_with(p.as_str()))
    })?;
    let transformer = component_bytes(dit_dir, None, &|_| true)?;
    let decoder = component_bytes(&root.join(IMAGE_VAE_COMPONENT), None, &|name: &str| {
        name.starts_with("decoder.") || name.starts_with("post_quant_conv.")
    })?;
    Ok(MemoryAssetFacts {
        base_bytes: conditioning
            .saturating_add(transformer)
            .saturating_add(decoder),
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: decoder,
        overlay_bytes: 0,
    })
}

/// Architecture axes read from the snapshot's own `transformer_diffusers/` and `vae_diffusers/`
/// configs. Gated on a materialized root, as every Candle provider's are.
fn architecture_facts_from_snapshot(
    _provider_id: &str,
    root: &Path,
    spec: &LoadSpec,
) -> gen_core::MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;

    // The staged tier's directory when one is staged, else the snapshot's own.
    let dit = af::config_file(&crate::image_model::resolve_dit_dir(root, spec).join("config.json"));
    let vae = af::component_config(root, IMAGE_VAE_COMPONENT);
    gen_core::MemoryArchitectureFacts {
        attention_heads: af::axis_of(dit.as_ref(), &["num_attention_heads"]),
        head_dim: af::axis_of(dit.as_ref(), &["attention_head_dim"]),
        transformer_blocks: af::axis_of(dit.as_ref(), &["num_layers"]),
        patch_size: af::axis_of(dit.as_ref(), &["patch_size"]),
        latent_channels: af::axis_of(dit.as_ref(), &["in_channels"]),
        vae_spatial_scale: vae.as_ref().map(|_| crate::config::VAE_RATIO as u32),
        // A still image: one latent frame, no temporal compression on this route.
        vae_temporal_scale: None,
        activation_dtype_width: u32::try_from(BF16_WIDTH).ok(),
    }
}

/// The weights-free contract — the registry conformance surface.
pub fn weights_free_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let mut contract = MemoryProviderContract::compatibility_default(
        IMAGE_MODEL_ID,
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: false,
            host_to_device_block_materialization: false,
            // Rung 4 is not implemented; this states what a window WOULD do — the DiT loader
            // reads device-format tensors.
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.architecture_facts =
        gen_core::standard_architecture_facts(&MEMORY_PROFILE, IMAGE_MODEL_ID, spec);
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        IMAGE_MEMORY_CALIBRATION_FINGERPRINT,
        spec.load_shape,
    ));
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: true,
        attention_chunking: false,
        transformer_window_materialization: false,
    };
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::DecodeTileArea,
        ],
    };
    contract.resident_request_memory = ResidentRequestMemory::PreserveLoadDefaults;
    for capability in &mut contract.strategies {
        capability.support = match capability.strategy {
            MemoryStrategy::Resident
            | MemoryStrategy::StagedResidency
            | MemoryStrategy::BoundedDecode => MemoryStrategySupport::Implemented,
            MemoryStrategy::BoundedAttention | MemoryStrategy::BoundedTransformerResidency => {
                MemoryStrategySupport::Missing
            }
        };
        capability.parameters = match capability.strategy {
            MemoryStrategy::BoundedDecode => crate::image_vae::bounded_decode_parameters(),
            _ => MemoryParameterRanges::default(),
        };
    }
    Ok(contract)
}

/// The executable contract for a real load: [`weights_free_contract`] plus the on-disk inventory.
pub fn contract_for(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    let mut contract = weights_free_contract(spec)?;
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(gen_core::Error::Msg(format!(
            "{IMAGE_MODEL_ID}: memory facts require a snapshot directory"
        )));
    };
    let te_dir = crate::image_model::resolve_text_encoder_dir(root, spec);
    let dit_dir = crate::image_model::resolve_dit_dir(root, spec);
    contract.asset_facts = asset_facts(&dit_dir, root, &te_dir)?;
    Ok(contract)
}

fn validate_decode(edge: Option<u32>, overlap: Option<u32>) -> gen_core::Result<()> {
    gen_core::validate_bounded_decode_params(
        IMAGE_MODEL_ID,
        &[IMAGE_DECODE_TILE_EDGE],
        IMAGE_DECODE_OVERLAP,
        edge,
        overlap,
    )
}

const MEMORY_PROFILE: ProviderMemoryProfile = ProviderMemoryProfile {
    gates: MemoryProfileGates::within(route_gate),
    fixtures: MemoryProfileFixtures::routes(&[
        MemoryFixtureRoute::TEXT_TO_IMAGE.with_phases(MemoryFixturePhases::Always)
    ])
    .implemented_only(),
    architecture: Some(gen_core::MemoryArchitectureSource::Snapshot(
        architecture_facts_from_snapshot,
    )),
    ..ProviderMemoryProfile::STANDARD
};

fn route_gate(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    _loaded_tier: MemoryNumericTier,
) -> gen_core::Result<()> {
    let id = contract.provider_id.as_str();
    if !matches!(context.mode, MemoryMode::TextToImage) {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: the candle route is text-to-image; got {:?}",
            context.mode
        )));
    }
    if context.geometry.reference_count != 0 || context.has_reference {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: reference-image editing is not wired on the candle route"
        )));
    }
    if context.use_pid {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: there is no PiD decoder on this route"
        )));
    }
    if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
        validate_decode(
            context.selection.parameters.decode_tile_edge,
            context.selection.parameters.decode_overlap,
        )?;
    }
    Ok(())
}

/// The provider safety check: the shared handshake, then this route's gate.
pub fn safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    gen_core::standard_registered_safety_check(&MEMORY_PROFILE, spec, contract, context)
}

/// The executable admission seam the loaded generator calls — the same
/// [`IMAGE_MEMORY_BEHAVIOR`] routing, through the profile's standard request handshake.
pub(crate) fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    gen_core::standard_begin_request(
        &MEMORY_PROFILE,
        IMAGE_MODEL_ID,
        spec,
        contract,
        context,
        registered_scope,
    )
}

fn registered_scope(
    provider_id: &'static str,
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    _loaded_tier: MemoryNumericTier,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        provider_id,
        candle_gen::default_device()?,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        crate::dit::MiniMaxH3DitConfig::image().num_layers,
        move |use_pid, edge, overlap| {
            if use_pid {
                return Err(gen_core::Error::Unsupported(format!(
                    "{provider_id}: there is no PiD decoder on this route"
                )));
            }
            validate_decode(Some(edge), Some(overlap))
        },
    )?;
    Ok(Some(Box::new(
        candle_gen::request_scope::CandleRequestScopeCore::new(config),
    )))
}

/// The memory-strategy registration paired with the `minimax_h3_image` generator.
pub const IMAGE_MEMORY_REGISTRATION: candle_gen::gen_core::MemoryRegistration = candle_gen::gen_core::profile_memory_registration!(
    MEMORY_PROFILE,
    IMAGE_MODEL_ID,
    contract_for
);

/// The weights-free contract fixture, dense tier only: the quantized tiers add no strategy or
/// formula, only smaller asset bytes read from the staged directories.
pub const IMAGE_MEMORY_CONTRACT_FIXTURE: candle_gen::gen_core::MemoryContractFixtureRegistration =
    candle_gen::gen_core::MemoryContractFixtureRegistration {
        provider_id: IMAGE_MODEL_ID,
        contract: weights_free_contract,
        surface_specs: memory_contract_surface_specs,
    };

/// The executable behavior seam every implemented optimized rung must have.
pub const IMAGE_MEMORY_BEHAVIOR: candle_gen::gen_core::MemoryBehaviorRegistration = candle_gen::gen_core::profile_memory_behavior_registration!(
    MEMORY_PROFILE,
    IMAGE_MODEL_ID,
    registered_scope
);

fn memory_contract_surface_specs() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::candle_memory_contract_surface_specs()
        .into_iter()
        .filter(|s| s.selector.tier == gen_core::MemoryContractSurfaceTier::Bf16)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("/minimax-h3-image".into()))
    }

    /// Only the rungs the generator runs are implemented (the resident baseline, staging and the
    /// shipped tiled decode); the bounded decode's domain is the 256 / 64 tile and nothing else.
    #[test]
    fn only_staged_residency_and_the_shipped_bounded_decode_are_declared() {
        let c = weights_free_contract(&spec()).unwrap();
        for (strategy, want) in [
            (MemoryStrategy::Resident, MemoryStrategySupport::Implemented),
            (
                MemoryStrategy::StagedResidency,
                MemoryStrategySupport::Implemented,
            ),
            (
                MemoryStrategy::BoundedDecode,
                MemoryStrategySupport::Implemented,
            ),
            (
                MemoryStrategy::BoundedAttention,
                MemoryStrategySupport::Missing,
            ),
            (
                MemoryStrategy::BoundedTransformerResidency,
                MemoryStrategySupport::Missing,
            ),
        ] {
            assert_eq!(
                c.capability(strategy).unwrap().support,
                want,
                "{strategy:?}"
            );
        }
        assert_eq!(
            c.capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .parameters,
            crate::image_vae::bounded_decode_parameters()
        );
        assert!(c.lifecycle.synchronized_phase_release && c.lifecycle.decode_tiling);
        assert!(validate_decode(Some(256), Some(64)).is_ok());
        assert!(validate_decode(Some(512), Some(64)).is_err());
        // The packed tiers use the same contract; only the asset bytes differ.
        for tier in [gen_core::Quant::Q4, gen_core::Quant::Q8] {
            let mut q = spec();
            q.quantize = Some(tier);
            assert!(weights_free_contract(&q).is_ok(), "{tier:?}");
        }
        assert!(!memory_contract_surface_specs().is_empty());
    }

    /// The weights-free surface publishes no axis it could only have inferred from the id.
    #[test]
    fn the_weights_free_surface_publishes_no_architecture_axis() {
        let surface = LoadSpec::new(WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ));
        assert!(weights_free_contract(&surface)
            .unwrap()
            .architecture_facts
            .is_empty());
    }
}
