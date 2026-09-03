//! Request-scoped memory contract for the bespoke InstantID MLX/Metal route.
//!
//! This is an InstantID-owned contract. It does not reuse generic SDXL or PuLID evidence.

use mlx_gen::gen_core::{
    self, LoadShape, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryComponentKind,
    MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryResidentComponent, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategySupport, Precision, WeightsSource,
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

/// Provider-local identities of the two auxiliary networks every InstantID load materializes.
pub const IDENTITYNET_COMPONENT_ID: &str = "instantid.identitynet.control_branch";
pub const FACE_IP_COMPONENT_ID: &str = "instantid.face_ip_adapter.resampler_and_kv";

/// The contract for a load whose checkpoints are known — the one every real InstantID load uses.
///
/// Epic SC-22657, E1. [`provider_contract`] takes only a numeric tier and is therefore structurally
/// incapable of pricing anything, so every contract this provider published carried
/// `MemoryAssetFacts::default()`: five zero fields for a route that materializes a full SDXL base
/// plus two auxiliary networks. All-zero facts pass the shared conformance walk vacuously
/// (`base == cond + trans + dec` holds at `0 == 0`, and the repeated-total rule is guarded on a
/// non-zero left side), which is why nothing failed; the declaration was simply absent.
///
/// The base three come from `mlx_gen_sdxl::snapshot_component_footprint`, this crate's own loader's
/// per-component resolution and dtype projection — not a second derivation of the rule — because
/// InstantID loads the stock SDXL base through `load_text_encoder_1_dtype` /
/// `load_text_encoder_2_dtype` / `load_unet_dtype` at `DTYPE` and `load_vae`, which upcasts to f32.
///
/// The two auxiliary networks are declared in `overlay_bytes` and as typed components:
/// the IdentityNet ControlNet (`load_controlnet(&paths.identitynet, DTYPE)`) and the converted face
/// IP-Adapter bundle (`Weights::from_file(&paths.ip_adapter)` + `cast_all(DTYPE)`, split into the
/// `image_proj` resampler and 70 decoupled cross-attention K/V pairs). Both are materialized at
/// `DTYPE`, so a 16-bit projection is exact.
///
/// Deliberately NOT priced here, with reasons:
///
/// * `paths.adapters` — merged onto the DENSE fp16 U-Net at load
///   (`apply_sdxl_adapters_with`, before any `quantize()`), so the factors have no independent
///   residency: `AdapterResidencyMode::Folded`, which is positive evidence of zero rather than an
///   omission.
/// * The optional OpenPose ControlNet, SCRFD/ArcFace face stack and PiD decoder — those are
///   attached after construction through `with_openpose` / `with_face` / the PiD spec and are not
///   reachable from [`InstantIdPaths`](crate::InstantIdPaths), so a paths-keyed contract cannot see
///   them. `InstantIdMemoryIdentity` already keys the admission on `face_restore` and `use_pid`, so
///   they remain distinguishable route axes.
pub fn provider_contract_for_paths(
    paths: &crate::InstantIdPaths,
    tier: MemoryNumericTier,
) -> MemoryProviderContract {
    use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};

    let mut contract = provider_contract(tier);
    let base = mlx_gen_sdxl::snapshot_component_footprint(&paths.sdxl_base);
    contract.asset_facts.conditioning_bytes = base.text_encoder;
    contract.asset_facts.transformer_bytes = base.dit;
    contract.asset_facts.decoder_bytes = base.vae;
    contract.asset_facts.base_bytes = base
        .text_encoder
        .saturating_add(base.dit)
        .saturating_add(base.vae);

    // Both auxiliaries are cast to `DTYPE` (fp16) on load, so 16 bits per element is exact; the
    // shared projection leaves an already-packed tensor at its stored width. A source that cannot
    // be read prices zero rather than turning the contract into a refusal — the loader raises the
    // actionable error against the same path moments later.
    let projected = |path: &std::path::Path, projection: ResidentProjection| {
        projected_safetensors_bytes(path, move |_| projection).unwrap_or(0)
    };
    let (WeightsSource::Dir(identitynet) | WeightsSource::File(identitynet)) = &paths.identitynet;
    let mut components = Vec::new();
    let identitynet_bytes = projected(identitynet, ResidentProjection::Bfloat16);
    if identitynet_bytes > 0 {
        components.push(MemoryResidentComponent {
            id: IDENTITYNET_COMPONENT_ID.to_owned(),
            kind: MemoryComponentKind::ControlBranch,
            resident_bytes: identitynet_bytes,
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        });
    }
    let face_ip_bytes = projected(&paths.ip_adapter, ResidentProjection::Bfloat16);
    if face_ip_bytes > 0 {
        components.push(MemoryResidentComponent {
            id: FACE_IP_COMPONENT_ID.to_owned(),
            kind: MemoryComponentKind::IpAdapter,
            resident_bytes: face_ip_bytes,
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        });
    }
    contract.asset_facts.overlay_bytes = identitynet_bytes.saturating_add(face_ip_bytes);
    if !components.is_empty() {
        // The formula already lists `OverlayBytes`, so a non-zero overlay MUST come with typed
        // components or the shared validator reports it.
        if let MemoryFormulaKind::PhaseEnvelope { phases, variables } = contract.formula {
            contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
                phases,
                variables,
                resident_components: components,
            };
        }
    }
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

    /// Feature-end review (SC-22667, E1): the LOADED contract must publish honest per-component
    /// asset bytes. `provider_contract` takes only a numeric tier and so published
    /// `MemoryAssetFacts::default()` — five zeros for a route that materializes a whole SDXL base
    /// plus an IdentityNet ControlNet and the face IP-Adapter bundle. All-zero facts pass the
    /// shared conformance walk vacuously, which is why nothing ever failed.
    ///
    /// Mutation that fails this: pointing `load_with_memory_context` back at
    /// `provider_contract(tier)` — or dropping the `overlay_bytes` assignment — leaves the loaded
    /// contract declaring zero for every field while the loader holds all of it.
    #[test]
    fn the_loaded_contract_prices_the_sdxl_base_and_both_identity_networks() {
        /// One-tensor safetensors file; `dtype` must agree with `width`, because the shared header
        /// reader refuses a payload length that is not `elements * dtype.size()`.
        fn write_tensor(path: &std::path::Path, dtype: &str, elements: usize, width: usize) {
            let data_bytes = elements * width;
            let mut header = format!(
                "{{\"weight\":{{\"dtype\":\"{dtype}\",\"shape\":[{elements}],\"data_offsets\":[0,{data_bytes}]}}}}"
            )
            .into_bytes();
            while !header.len().is_multiple_of(8) {
                header.push(b' ');
            }
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            bytes.resize(bytes.len() + data_bytes, 0);
            std::fs::write(path, bytes).unwrap();
        }

        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().join("sdxl");
        for component in ["unet", "vae", "text_encoder", "text_encoder_2"] {
            std::fs::create_dir_all(base.join(component)).unwrap();
        }
        write_tensor(
            &base.join("unet/diffusion_pytorch_model.fp16.safetensors"),
            "F16",
            100,
            2,
        );
        // The VAE ships fp16 and is upcast to f32 on every load.
        write_tensor(
            &base.join("vae/diffusion_pytorch_model.fp16.safetensors"),
            "F16",
            50,
            2,
        );
        write_tensor(
            &base.join("text_encoder/model.fp16.safetensors"),
            "F16",
            20,
            2,
        );
        write_tensor(
            &base.join("text_encoder_2/model.fp16.safetensors"),
            "F16",
            30,
            2,
        );
        let identitynet = tmp.path().join("identitynet.safetensors");
        write_tensor(&identitynet, "F16", 40, 2);
        let ip_adapter = tmp.path().join("ip-adapter.safetensors");
        write_tensor(&ip_adapter, "F16", 60, 2);

        let paths = crate::InstantIdPaths {
            sdxl_base: base,
            identitynet: WeightsSource::File(identitynet),
            ip_adapter,
            adapters: Vec::new(),
        };
        let contract = provider_contract_for_paths(&paths, dense_numeric_tier());

        assert_eq!(contract.asset_facts.conditioning_bytes, 20 * 2 + 30 * 2);
        assert_eq!(contract.asset_facts.transformer_bytes, 100 * 2);
        assert_eq!(
            contract.asset_facts.decoder_bytes,
            50 * 4,
            "the SDXL VAE is materialized f32 whatever it is stored as"
        );
        assert_eq!(
            contract.asset_facts.base_bytes,
            100 + 200 + 200,
            "base must be exactly its own three-way decomposition"
        );
        assert_eq!(
            contract.asset_facts.overlay_bytes,
            40 * 2 + 60 * 2,
            "the IdentityNet and the face IP bundle are auxiliary networks beside the base"
        );
        assert_eq!(
            contract.auxiliary_resident_bytes(),
            contract.asset_facts.overlay_bytes
        );
        let ids: Vec<&str> = contract
            .resident_components()
            .iter()
            .map(|component| component.id.as_str())
            .collect();
        assert_eq!(ids, vec![IDENTITYNET_COMPONENT_ID, FACE_IP_COMPONENT_ID]);
        assert_eq!(
            contract.resident_components()[0].kind,
            MemoryComponentKind::ControlBranch
        );
        assert_eq!(
            contract.resident_components()[1].kind,
            MemoryComponentKind::IpAdapter
        );
        assert!(
            contract.conformance_errors().is_empty(),
            "{:?}",
            contract.conformance_errors()
        );
        gen_core_testkit::assert_memory_contract_facts_conform(&contract);
        // The tier-only entry point remains the declaration-equivalent surface it always was.
        assert_eq!(
            provider_contract(dense_numeric_tier()).asset_facts,
            gen_core::MemoryAssetFacts::default()
        );
    }

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
        assert!(contract.architecture_facts.has_declared_architecture_axis());
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
