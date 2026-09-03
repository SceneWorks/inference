//! Request-scoped memory contract for the bespoke InstantID MLX/Metal route.
//!
//! This is an InstantID-owned contract. It does not reuse generic SDXL or PuLID evidence.

use mlx_gen::gen_core::{
    self, LoadShape, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryRunContext, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategySupport, Precision,
};

pub const PROVIDER_ID: &str = "instantid";
/// Revision of the shared InstantID request/evidence schema. Backend is an independent identity
/// axis, so encoding MLX here would split otherwise identical cross-backend evidence semantics.
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

pub fn provider_contract(tier: MemoryNumericTier) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        PROVIDER_ID,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
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
    // Epic SC-22657 (E2). InstantID owns no denoiser: it layers an IdentityNet ControlNet and face
    // IP tokens onto a stock SDXL base, loaded through `mlx_gen_sdxl::load_unet_dtype` with
    // `UNetConfig::sdxl_base()` and `mlx_gen_sdxl::load_vae`. The axes are therefore the shared SDXL
    // derivation's, at this crate's own `DTYPE = Dtype::Float16` activation width.
    contract.architecture_facts = mlx_gen_sdxl::config::architecture_facts(
        &mlx_gen_sdxl::UNetConfig::sdxl_base(),
        &mlx_gen_sdxl::VaeConfig::sdxl_base(),
        mlx_gen::architecture_facts::HALF_ACTIVATION_WIDTH,
    );
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
    // Tier is validated request-by-request; retaining it here would fabricate a calibration row.
    let _ = tier;
    contract
}

pub const fn dense_numeric_tier() -> MemoryNumericTier {
    MemoryNumericTier {
        precision: Precision::Bf16,
        quant: None,
        component_precision_floors: &[],
    }
}

pub fn safety_check(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
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
    gen_core::standard_memory_strategy_safety_check(contract, context, Some(tier), Some(&route))
}

pub fn validate_context(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    identity: &InstantIdMemoryIdentity,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    match safety_check(contract, tier, identity, context) {
        MemorySafetyDecision::Accept => Ok(()),
        MemorySafetyDecision::Reject { reason } => Err(gen_core::Error::Unsupported(reason)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{MemoryBehaviorRoute, MemoryStrategy};

    /// AC (SC-22662): InstantID publishes the axes of the SDXL base it layers onto — it owns no
    /// denoiser of its own — and its contract passes the shared facts conformance check.
    #[test]
    fn architecture_facts_are_the_shared_sdxl_base_axes() {
        let contract = provider_contract(dense_numeric_tier());
        assert_eq!(
            contract.architecture_facts,
            mlx_gen::gen_core::MemoryArchitectureFacts {
                // A conv U-Net has no single head count (5/10/20 across three resolutions) and no
                // uniform transformer trunk depth; the head WIDTH is uniform and is published.
                attention_heads: None,
                head_dim: Some(64),
                transformer_blocks: None,
                patch_size: None,
                latent_channels: Some(4),
                vae_spatial_scale: Some(8),
                vae_temporal_scale: None,
                activation_dtype_width: Some(2),
            }
        );
        assert!(contract.architecture_facts.has_snapshot_read_axis());
        gen_core_testkit::assert_memory_contract_facts_conform(&contract);
    }

    #[test]
    fn only_resident_and_staged_are_selectable() {
        let contract = provider_contract(dense_numeric_tier());
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
    fn composition_identity_distinguishes_openpose_and_second_pass() {
        let base = InstantIdMemoryIdentity {
            route: InstantIdRoute::Identity,
            adapter_count: 0,
            use_pid: false,
            face_restore: false,
            artifact_fingerprint: "same".into(),
        };
        let mut pose = base.clone();
        pose.route = InstantIdRoute::Pose;
        let mut restore = base.clone();
        restore.face_restore = true;
        assert_ne!(base.overlay_key(), pose.overlay_key());
        assert_ne!(base.overlay_key(), restore.overlay_key());
    }

    #[test]
    fn exact_route_context_accepts_and_crossed_tier_fails_closed() {
        let tier = dense_numeric_tier();
        let contract = provider_contract(tier);
        let identity = InstantIdMemoryIdentity {
            route: InstantIdRoute::Angle,
            adapter_count: 1,
            use_pid: false,
            face_restore: false,
            artifact_fingerprint: "artifacts-a".into(),
        };
        let mut context = gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::StagedResidency,
            tier,
            MemoryBehaviorRoute {
                mode: MemoryMode::Other("character_image".into()),
                reference_count: 1,
                use_pid: false,
                has_phases: false,
                overlay: Some(identity.overlay_key()),
            },
        )
        .unwrap();
        context.evidence_revision = REQUEST_EVIDENCE_REVISION.into();
        assert_eq!(
            safety_check(&contract, tier, &identity, &context),
            MemorySafetyDecision::Accept
        );
        let crossed = MemoryNumericTier {
            quant: Some(mlx_gen::Quant::Q4),
            ..tier
        };
        assert!(matches!(
            safety_check(&contract, crossed, &identity, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }
}
