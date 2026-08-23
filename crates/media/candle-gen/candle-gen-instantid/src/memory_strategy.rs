//! Request-scoped memory contract for the bespoke InstantID Candle/CUDA route.
//!
//! This contract deliberately does not inherit SDXL or PuLID evidence. InstantID's IdentityNet,
//! face IP adapter, optional OpenPose branch, adapters, PiD decoder, and restoration pass form one
//! exact composition identity and must be admitted together.

use candle_gen::gen_core::{
    self, LoadShape, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategySupport, MemoryWindowMaterialization, Precision,
};

pub const PROVIDER_ID: &str = "instantid";
/// Revision of the shared InstantID request/evidence schema. Backend is an independent identity
/// axis, so encoding Candle here would split otherwise identical cross-backend evidence semantics.
pub const REQUEST_EVIDENCE_REVISION: &str = "instantid-request-contract-v1";

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
        REQUEST_EVIDENCE_REVISION,
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
mod tests {
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
}
