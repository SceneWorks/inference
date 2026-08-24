//! Request-scoped memory contract for the bespoke InstantID Candle/CUDA route.
//!
//! This contract deliberately does not inherit SDXL or PuLID evidence. InstantID's IdentityNet,
//! face IP adapter, optional OpenPose branch, adapters, PiD decoder, and restoration pass form one
//! exact composition identity and must be admitted together.

use std::path::Path;

use candle_gen::gen_core::{
    self, AdapterResidencyMode, LoadShape, MemoryAssetFacts, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategySupport, MemoryWindowMaterialization, Precision, WeightsSource,
};

use crate::model::InstantIdPaths;

pub const PROVIDER_ID: &str = "instantid";
/// Revision of the shared InstantID request/evidence schema. Backend is an independent identity
/// axis, so encoding Candle here would split otherwise identical cross-backend evidence semantics.
pub const REQUEST_EVIDENCE_REVISION: &str = "instantid-request-contract-v1";
/// Identity of the *executable memory semantics* this contract calibrates — deliberately a separate
/// string from [`REQUEST_EVIDENCE_REVISION`], which versions the request/evidence schema. They are
/// two independent axes: sharing one literal made a bump to either silently restate the other.
const CALIBRATION_FINGERPRINT: &str = "instantid-candle-staged-conditioning-v1";

/// InstantID materializes fp16 weights ([`crate::model`]'s `DTYPE`), so each float element of every
/// component costs two bytes once loaded.
const FLOAT_WIDTH: u64 = 2;

fn source_path(source: &WeightsSource) -> &Path {
    match source {
        WeightsSource::Dir(path) | WeightsSource::File(path) => path,
    }
}

/// Bytes one resolved component occupies once loaded: float tensors at the compute width, integer
/// tensors at their stored width. Header-only — no tensor data is materialized.
fn component_bytes(path: &Path) -> gen_core::Result<u64> {
    gen_core::weightsmeta::safetensors_path_tensor_headers(path)?
        .iter()
        .try_fold(0_u64, |sum, header| {
            let bytes = if header.is_float() {
                header.materialized_bytes(FLOAT_WIDTH)?
            } else {
                header.data_bytes
            };
            sum.checked_add(bytes).ok_or_else(|| {
                gen_core::Error::Msg("instantid: component byte sum overflow".into())
            })
        })
}

/// Load-exact component bytes for the exact InstantID composition.
///
/// IdentityNet and the face IP-Adapter are auxiliary networks resident alongside the SDXL base, so
/// they are declared in `overlay_bytes` (the aggregate this contract's
/// [`MemoryFormulaVariable::OverlayBytes`] makes load-bearing) rather than folded into the three
/// base-model fields. User LoRA/LoKr adapters are folded onto the UNet at load and therefore add no
/// resident bytes of their own.
pub fn asset_facts(paths: &InstantIdPaths) -> gen_core::Result<MemoryAssetFacts> {
    let conditioning = component_bytes(&paths.sdxl_base.join("text_encoder"))?
        .saturating_add(component_bytes(&paths.sdxl_base.join("text_encoder_2"))?);
    let transformer = component_bytes(&paths.sdxl_base.join("unet"))?;
    let decoder = component_bytes(source_path(paths.sdxl.vae_fp16_fix()))?;
    let overlay = component_bytes(source_path(&paths.identitynet))?
        .saturating_add(component_bytes(&paths.ip_adapter)?)
        .saturating_add(
            gen_core::adapter_stack_resident_bytes(&paths.adapters, AdapterResidencyMode::Folded)
                .ok_or_else(|| {
                gen_core::Error::Unsupported(
                    "instantid: every resident adapter must have an exact non-zero size".into(),
                )
            })?,
        );
    Ok(MemoryAssetFacts {
        base_bytes: conditioning
            .saturating_add(transformer)
            .saturating_add(decoder),
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: decoder,
        overlay_bytes: overlay,
    })
}

/// The executable contract for a real InstantID composition: identical to [`provider_contract`]
/// except that its declared [`MemoryFormulaVariable::AssetBytes`] / [`MemoryFormulaVariable::OverlayBytes`]
/// inputs are the exact on-disk component inventory rather than zero placeholders.
pub fn provider_contract_for_paths(
    paths: &InstantIdPaths,
) -> gen_core::Result<MemoryProviderContract> {
    let mut contract = provider_contract();
    contract.asset_facts = asset_facts(paths)?;
    Ok(contract)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstantIdRoute {
    Identity,
    Angle,
    Pose,
}

impl InstantIdRoute {
    pub const fn as_key(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::Angle => "angle",
            Self::Pose => "pose-openpose",
        }
    }
}

/// Immutable identity of everything that can change InstantID residency or execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InstantIdMemoryIdentity {
    pub route: InstantIdRoute,
    pub adapter_count: usize,
    pub use_pid: bool,
    pub face_restore: bool,
    pub artifact_fingerprint: String,
}

impl InstantIdMemoryIdentity {
    pub fn overlay_key(&self) -> String {
        format!(
            "instantid-v1/{}/a{}/p{}/r{}/{}",
            self.route.as_key(),
            self.adapter_count,
            u8::from(self.use_pid),
            u8::from(self.face_restore),
            self.artifact_fingerprint
        )
    }
}

pub fn provider_contract() -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        PROVIDER_ID,
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
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
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        CALIBRATION_FINGERPRINT,
        LoadShape::EagerMaterialization,
    ));
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

pub const fn resolved_numeric_tier() -> MemoryNumericTier {
    MemoryNumericTier {
        // `Bf16` is gen-core's dense-default sentinel; InstantID materializes fp16 weights.
        precision: Precision::Bf16,
        quant: None,
        component_precision_floors: &[],
    }
}

pub fn safety_check(
    contract: &MemoryProviderContract,
    identity: &InstantIdMemoryIdentity,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route = || {
        if context.mode != MemoryMode::Other("character_image".into())
            || !context.has_reference
            || context.geometry.reference_count != 1
            || context.geometry.batch != 1
            || context.geometry.frames != 1
            || context.use_pid != identity.use_pid
            || context.has_phases != identity.face_restore
            || context.overlay.as_deref() != Some(identity.overlay_key().as_str())
            || context.evidence_revision != REQUEST_EVIDENCE_REVISION
        {
            return Err(gen_core::Error::Msg(format!(
                "{PROVIDER_ID}: context does not match the exact character_image composition"
            )));
        }
        Ok(())
    };
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(resolved_numeric_tier()),
        Some(&route),
    )
}

pub fn validate_context(
    contract: &MemoryProviderContract,
    identity: &InstantIdMemoryIdentity,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    match safety_check(contract, identity, context) {
        MemorySafetyDecision::Accept => Ok(()),
        MemorySafetyDecision::Reject { reason } => Err(gen_core::Error::Unsupported(reason)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use candle_gen::gen_core::{MemoryBehaviorRoute, MemoryStrategy};

    #[test]
    fn only_resident_and_staged_are_selectable() {
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
        assert!(!contract.lifecycle.decode_tiling);
        assert!(!contract.lifecycle.attention_chunking);
        assert!(!contract.lifecycle.transformer_window_materialization);
    }

    #[test]
    fn composition_identity_distinguishes_every_overlay_axis() {
        let base = InstantIdMemoryIdentity {
            route: InstantIdRoute::Identity,
            adapter_count: 0,
            use_pid: false,
            face_restore: false,
            artifact_fingerprint: "a".into(),
        };
        let mut variants = vec![];
        let mut value = base.clone();
        value.route = InstantIdRoute::Angle;
        variants.push(value);
        let mut value = base.clone();
        value.adapter_count = 1;
        variants.push(value);
        let mut value = base.clone();
        value.use_pid = true;
        variants.push(value);
        let mut value = base.clone();
        value.face_restore = true;
        variants.push(value);
        let mut value = base.clone();
        value.artifact_fingerprint = "b".into();
        variants.push(value);
        assert!(variants
            .iter()
            .all(|variant| variant.overlay_key() != base.overlay_key()));
    }

    #[test]
    fn exact_route_context_accepts_and_crossed_evidence_fails_closed() {
        let contract = provider_contract();
        let identity = InstantIdMemoryIdentity {
            route: InstantIdRoute::Pose,
            adapter_count: 2,
            use_pid: true,
            face_restore: true,
            artifact_fingerprint: "artifacts-a".into(),
        };
        let mut context = gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::StagedResidency,
            resolved_numeric_tier(),
            MemoryBehaviorRoute {
                mode: MemoryMode::Other("character_image".into()),
                reference_count: 1,
                use_pid: true,
                has_phases: true,
                overlay: Some(identity.overlay_key()),
            },
        )
        .unwrap();
        context.evidence_revision = REQUEST_EVIDENCE_REVISION.into();
        assert_eq!(
            safety_check(&contract, &identity, &context),
            MemorySafetyDecision::Accept
        );
        context.evidence_revision = "borrowed-sdxl-evidence".into();
        assert!(matches!(
            safety_check(&contract, &identity, &context),
            MemorySafetyDecision::Reject { .. }
        ));
        context.evidence_revision = REQUEST_EVIDENCE_REVISION.into();
        context.overlay = Some("instantid-v1/identity/a2/p1/r1/artifacts-a".into());
        assert!(matches!(
            safety_check(&contract, &identity, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    /// Executable memory semantics and the request/evidence schema are independent axes. Sharing one
    /// literal made a bump to either silently restate the other.
    #[test]
    fn calibration_identity_is_not_the_request_evidence_revision() {
        let fingerprint = provider_contract().calibration.unwrap().fingerprint;
        assert_ne!(fingerprint, REQUEST_EVIDENCE_REVISION);
        assert!(!fingerprint.is_empty());
    }

    /// Write a synthetic composition whose components have deliberately distinct element counts, so
    /// a swapped assignment cannot pass. Returns the paths plus the exact per-field byte totals
    /// derived from the tensors actually written.
    pub(crate) fn priced_paths(temp: &tempfile::TempDir) -> (InstantIdPaths, u64, u64, u64, u64) {
        use candle_gen::candle_core::{DType, Device, Tensor};
        use std::collections::HashMap;

        let root = temp.path().join("instantid_priced");
        let write = |relative: &str, rows: usize, columns: usize| -> u64 {
            let path = root.join(relative);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            let mut tensors = HashMap::new();
            tensors.insert(
                "x.weight".to_string(),
                Tensor::zeros((rows, columns), DType::F32, &Device::Cpu).unwrap(),
            );
            candle_gen::candle_core::safetensors::save(&tensors, &path).unwrap();
            // Written F32, materialized fp16: two bytes per element, derived from the shape written
            // here rather than pinned to a literal.
            (rows as u64) * (columns as u64) * FLOAT_WIDTH
        };
        let conditioning = write("sdxl/text_encoder/model.safetensors", 16, 8)
            + write("sdxl/text_encoder_2/model.safetensors", 12, 8);
        let transformer = write("sdxl/unet/diffusion_pytorch_model.safetensors", 64, 32);
        let decoder = write("vae/diffusion_pytorch_model.safetensors", 4, 2);
        let overlay = write("identitynet/diffusion_pytorch_model.safetensors", 24, 16)
            + write("ip-adapter.safetensors", 6, 4);
        let paths = InstantIdPaths {
            sdxl_base: root.join("sdxl"),
            identitynet: WeightsSource::Dir(root.join("identitynet")),
            ip_adapter: root.join("ip-adapter.safetensors"),
            adapters: Vec::new(),
            sdxl: crate::model::SdxlComponents::for_test(
                WeightsSource::Dir(root.join("sdxl")),
                WeightsSource::Dir(root.join("sdxl")),
                WeightsSource::File(root.join("vae/diffusion_pytorch_model.safetensors")),
            ),
        };
        (paths, conditioning, transformer, decoder, overlay)
    }

    #[test]
    fn the_contract_prices_its_declared_asset_and_overlay_bytes_from_disk() {
        let temp = tempfile::tempdir().unwrap();
        let (paths, conditioning, transformer, decoder, overlay) = priced_paths(&temp);
        let contract = provider_contract_for_paths(&paths).unwrap();

        assert!(contract.formula.uses(MemoryFormulaVariable::AssetBytes));
        assert!(contract.formula.uses(MemoryFormulaVariable::OverlayBytes));
        assert_eq!(contract.asset_facts.conditioning_bytes, conditioning);
        assert_eq!(contract.asset_facts.transformer_bytes, transformer);
        assert_eq!(contract.asset_facts.decoder_bytes, decoder);
        assert_eq!(contract.asset_facts.overlay_bytes, overlay);
        assert_eq!(
            contract.asset_facts.base_bytes,
            conditioning + transformer + decoder
        );
        assert_ne!(
            contract.asset_facts,
            MemoryAssetFacts::default(),
            "declared AssetBytes/OverlayBytes variables must not be pinned at zero"
        );
    }
}
