//! Candle/CUDA Kolors memory contract.
//!
//! This is intentionally a narrow contract: the registered provider owns only base T2I and the
//! single-reference edit route.  IP-Adapter and pose ControlNet have independent model identities
//! and must never borrow this base receipt or memory evidence.

use std::path::Path;

use candle_gen::gen_core::{
    self, GenerationMemory, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategySupport, MemoryWindowMaterialization, Precision, Quant,
};

pub const REQUEST_EVIDENCE_REVISION: &str = "kolors-candle-request-contract-v1";
const CALIBRATION_FINGERPRINT: &str = "kolors-candle-staged-chatglm-unet-f32-vae-v1";

pub fn provider_contract() -> MemoryProviderContract {
    provider_contract_for(crate::MODEL_ID)
}

/// Bespoke IP-Adapter and ControlNet routes deliberately mint independent contracts; they may share
/// a physical Kolors base but never a provider/evidence identity.
pub fn provider_contract_for(provider_id: &str) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    contract.lifecycle = MemoryLifecycleCapabilities {
        // Edit has a VAE-init phase, but it is deliberately represented by the decoder/VAE phase:
        // no additional advertised optimization rung is claimed for it.
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
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::OverlayBytes,
        ],
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        CALIBRATION_FINGERPRINT,
        gen_core::LoadShape::EagerMaterialization,
    ));
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

/// The selected numeric tier comes from the physical component headers/configuration, never the
/// requested UI quantization label.  q4/q8 are only valid when both ChatGLM and UNet agree.
pub fn physical_tier(root: &Path) -> gen_core::Result<MemoryNumericTier> {
    let text = crate::pipeline::detect_packed_group(&root.join("text_encoder/config.json"))
        .map_err(gen_core::Error::backend)?;
    let unet = crate::pipeline::detect_packed_group(&root.join("unet/config.json"))
        .map_err(gen_core::Error::backend)?;
    let quant = match (text, unet) {
        (None, None) => None,
        (Some(text_group), Some(unet_group)) => {
            if text_group != unet_group {
                return Err(gen_core::Error::Unsupported(format!(
                    "kolors: packed ChatGLM group {text_group} does not match packed UNet group {unet_group}"
                )));
            }
            let packed_bits = |component: &str| -> gen_core::Result<u64> {
                let config =
                    std::fs::read(root.join(component).join("config.json")).map_err(|error| {
                        gen_core::Error::Msg(format!(
                            "kolors: read physical {component} tier: {error}"
                        ))
                    })?;
                let value: serde_json::Value =
                    serde_json::from_slice(&config).map_err(|error| {
                        gen_core::Error::Msg(format!(
                            "kolors: parse physical {component} tier: {error}"
                        ))
                    })?;
                value
                    .pointer("/quantization/bits")
                    .and_then(serde_json::Value::as_u64)
                    .ok_or_else(|| {
                        gen_core::Error::Unsupported(format!(
                            "kolors: packed {component} lacks quantization.bits"
                        ))
                    })
            };
            let bits = packed_bits("unet")?;
            if packed_bits("text_encoder")? != bits {
                return Err(gen_core::Error::Unsupported(
                    "kolors: packed ChatGLM and UNet bit widths differ".into(),
                ));
            }
            match bits {
                4 => Some(Quant::Q4),
                8 => Some(Quant::Q8),
                _ => {
                    return Err(gen_core::Error::Unsupported(format!(
                        "kolors: unsupported physical packed width {bits}"
                    )))
                }
            }
        }
        _ => {
            return Err(gen_core::Error::Unsupported(
                "kolors: mixed dense/packed ChatGLM and UNet artifacts are refused".into(),
            ))
        }
    };
    // The canonical bf16 snapshot executes Candle's exact F32 loader recipe and F32 VAE.  The
    // numeric identity stays BF16 (physical bytes), while execution precision is documented by the
    // provider's receipt/phase implementation rather than pretending the source artifact is F32.
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: &[],
    })
}

pub fn validate_context(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    tier: MemoryNumericTier,
) -> gen_core::Result<()> {
    if context.evidence_revision != REQUEST_EVIDENCE_REVISION {
        return Err(gen_core::Error::Unsupported(format!(
            "kolors: evidence revision {} does not match {}",
            context.evidence_revision, REQUEST_EVIDENCE_REVISION
        )));
    }
    let route = || match context.mode {
        MemoryMode::TextToImage
            if !context.has_reference && context.geometry.reference_count == 0 =>
        {
            Ok(())
        }
        MemoryMode::Edit | MemoryMode::ImageToImage
            if context.has_reference && context.geometry.reference_count == 1 =>
        {
            Ok(())
        }
        _ => Err(gen_core::Error::Unsupported(
            "kolors: memory evidence is bound only to base T2I or one-reference edit".into(),
        )),
    };
    match gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(tier),
        Some(&route),
    ) {
        MemorySafetyDecision::Accept => Ok(()),
        MemorySafetyDecision::Reject { reason } => Err(gen_core::Error::Unsupported(reason)),
    }
}

pub fn validate_bespoke_context(
    contract: &MemoryProviderContract,
    root: &Path,
    context: &MemoryRunContext,
    require_reference: bool,
    require_pid: bool,
) -> gen_core::Result<()> {
    if context.has_reference != require_reference || context.use_pid != require_pid {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: crossed bespoke reference/PiD receipt",
            contract.provider_id
        )));
    }
    match gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(physical_tier(root)?),
        None,
    ) {
        MemorySafetyDecision::Accept => Ok(()),
        MemorySafetyDecision::Reject { reason } => Err(gen_core::Error::Unsupported(reason)),
    }
}

pub fn request_memory(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> Option<GenerationMemory> {
    contract.generation_memory(&context.selection)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn packed(bits: u8, group: u32) -> String {
        format!(r#"{{"quantization":{{"bits":{bits},"group_size":{group}}}}}"#)
    }

    #[test]
    fn only_resident_and_staged_are_advertised() {
        let contract = provider_contract();
        for capability in contract.strategies {
            assert_eq!(
                capability.support,
                if matches!(
                    capability.strategy,
                    MemoryStrategy::Resident | MemoryStrategy::StagedResidency
                ) {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                }
            );
        }
        assert!(contract.lifecycle.synchronized_phase_release);
        assert!(!contract.lifecycle.decode_tiling);
        assert!(!contract.lifecycle.attention_chunking);
        assert!(!contract.lifecycle.transformer_window_materialization);
    }

    #[test]
    fn physical_tier_requires_a_matching_packed_pair() {
        let temp = tempfile::tempdir().unwrap();
        for component in ["text_encoder", "unet"] {
            let dir = temp.path().join(component);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("config.json"), packed(4, 64)).unwrap();
        }
        assert_eq!(physical_tier(temp.path()).unwrap().quant, Some(Quant::Q4));

        std::fs::write(temp.path().join("text_encoder/config.json"), packed(8, 64)).unwrap();
        assert!(physical_tier(temp.path()).is_err());
        std::fs::write(temp.path().join("text_encoder/config.json"), packed(4, 32)).unwrap();
        assert!(physical_tier(temp.path()).is_err());
    }

    #[test]
    fn dense_tier_is_not_relabelled_as_requested_quantization() {
        let temp = tempfile::tempdir().unwrap();
        for component in ["text_encoder", "unet"] {
            let dir = temp.path().join(component);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("config.json"), "{}").unwrap();
        }
        let tier = physical_tier(temp.path()).unwrap();
        assert_eq!(tier.precision, Precision::Bf16);
        assert_eq!(tier.quant, None);
    }
}
