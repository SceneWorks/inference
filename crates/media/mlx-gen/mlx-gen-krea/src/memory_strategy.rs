//! Shared memory-strategy contract for the MLX Krea pose-control provider.
//!
//! This module declares only the quality-preserving mechanisms the provider actually executes.
//! Exact peak envelopes remain in SceneWorks' promoted calibration bundle.
//!
//! SC-15517's real q4 1024²/1-step pose-control A/B held staged residency plus the verified 512/64
//! decode in both arms. Adding 64 Mi-score attention to the resident seven-block pose branch and both
//! attention/windowing to the reopenable 28-block base DiT reduced request peak from 15.574 GiB to
//! 9.200 GiB (40.9%) with zero pixel delta. The overlay remains explicitly resident in
//! `resident_components`; only the base DiT advertises `TransformerComponent::Dit` windowing.

use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};
#[cfg(test)]
use mlx_gen::gen_core::MemoryGeometry;
use mlx_gen::gen_core::{
    Error as CoreError, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryCalibrationIdentity, MemoryComponentKind, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryLifecycleCapabilities, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent, MemoryRunContext,
    MemoryRuntimeSemantics, MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability,
    MemoryStrategySupport, Result as CoreResult, TransformerComponent,
};

pub const MEMORY_CALIBRATION_FINGERPRINT: &str =
    "krea-control-mlx-full-ladder-512-64-attn64m-window1-2026-08-03-v2";
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_OVERLAP: u32 = 64;
pub const ATTENTION_CHUNK_SIZE: u32 = mlx_gen::attention::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;
/// Exact tile-edge domain admitted by current real-weight evidence. The 384 px candidate is
/// deliberately excluded: the clean 1024² sc-16099 run exceeded the established diffusion-latent
/// maximum-error threshold, so it must not inherit the 512 px calibration.
pub const DECODE_TILE_EDGES: [u32; 1] = [DECODE_TILE_EDGE];

fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(
        provider_id,
        DECODE_TILE_EDGES.iter().copied(),
        DECODE_OVERLAP,
    )
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let (asset_facts, resident_components) = asset_facts(spec, provider_id)?;
    memory_strategy_contract_with_asset_facts(
        provider_id,
        spec,
        asset_facts,
        resident_components,
        streamable_base_transformer(spec, provider_id)?,
    )
}

/// Declaration-equivalent contract used only by weights-free registry conformance.
pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    memory_strategy_contract_with_asset_facts(
        provider_id,
        spec,
        MemoryAssetFacts::default(),
        Vec::new(),
        streamable_base_transformer(spec, provider_id)?,
    )
}

fn memory_strategy_contract_with_asset_facts(
    provider_id: &str,
    spec: &LoadSpec,
    asset_facts: MemoryAssetFacts,
    resident_components: Vec<MemoryResidentComponent>,
    streamable_transformer: bool,
) -> CoreResult<MemoryProviderContract> {
    let routes = decode_routes(provider_id)?;
    let staged_residency = matches!(spec.offload_policy, mlx_gen::OffloadPolicy::Sequential);
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::OverlayBytes,
        MemoryFormulaVariable::DecodeTileArea,
        MemoryFormulaVariable::AttentionChunkSize,
        MemoryFormulaVariable::TransformerWindowSize,
    ];
    Ok(MemoryProviderContract {
        provider_id: provider_id.to_owned(),
        backend: MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
        strategies: MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: match strategy {
                    MemoryStrategy::Resident
                    | MemoryStrategy::BoundedDecode
                    | MemoryStrategy::BoundedAttention => MemoryStrategySupport::Implemented,
                    MemoryStrategy::BoundedTransformerResidency if streamable_transformer => {
                        MemoryStrategySupport::Implemented
                    }
                    MemoryStrategy::StagedResidency if staged_residency => {
                        MemoryStrategySupport::Implemented
                    }
                    MemoryStrategy::StagedResidency
                    | MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
                },
                parameters: match strategy {
                    MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                        decode_tile_edges: routes.native_edges().to_vec(),
                        decode_overlaps: vec![DECODE_OVERLAP],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                        attention_chunk_sizes: vec![ATTENTION_CHUNK_SIZE],
                        ..Default::default()
                    },
                    MemoryStrategy::BoundedTransformerResidency if streamable_transformer => {
                        MemoryParameterRanges {
                            transformer_window_sizes: vec![TRANSFORMER_WINDOW_SIZE],
                            transformer_window_components: vec![TransformerComponent::Dit],
                            ..Default::default()
                        }
                    }
                    _ => MemoryParameterRanges::default(),
                },
            })
            .collect(),
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: streamable_transformer
            .then_some((
                MemoryStrategy::BoundedTransformerResidency,
                mlx_gen::gen_core::MemoryStrategyPrerequisite::Rung {
                    rung: MemoryStrategy::StagedResidency,
                    scope: mlx_gen::gen_core::MemoryPrerequisiteScope::EngagedInSameRequest,
                },
            ))
            .into_iter()
            .collect(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: mlx_gen::gen_core::ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: vec![
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ],
            synchronized_phase_release: staged_residency,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: streamable_transformer,
        },
        formula: if resident_components.is_empty() {
            MemoryFormulaKind::PhaseEnvelope { phases, variables }
        } else {
            MemoryFormulaKind::ComponentPhaseEnvelope {
                phases,
                variables,
                resident_components,
            }
        },
        calibration: Some(MemoryCalibrationIdentity::new(
            MEMORY_CALIBRATION_FINGERPRINT,
            spec.load_shape,
        )),
        asset_facts,
        runtime: MemoryRuntimeSemantics::default(),
    })
}

fn streamable_base_transformer(spec: &LoadSpec, provider_id: &str) -> CoreResult<bool> {
    let mlx_gen::WeightsSource::Dir(root) = &spec.weights else {
        return Ok(false);
    };
    let plan = crate::model::resolve_load_plan(spec, root, provider_id)?;
    Ok(
        matches!(spec.offload_policy, mlx_gen::OffloadPolicy::Sequential)
            && matches!(
                spec.load_shape,
                mlx_gen::gen_core::LoadShape::DeferredMaterialization
            )
            && !crate::model::adapters_have_diff_patch(&spec.adapters)
            && plan.load_time_quant_bits.is_none(),
    )
}

fn asset_facts(
    spec: &LoadSpec,
    provider_id: &str,
) -> CoreResult<(MemoryAssetFacts, Vec<MemoryResidentComponent>)> {
    let mlx_gen::WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Msg(
            "krea pose memory facts require a snapshot directory".to_owned(),
        ));
    };
    let project = |path: &std::path::Path, select: &dyn Fn(&str) -> bool| -> CoreResult<u64> {
        projected_safetensors_bytes(path, |tensor| {
            if let Some(quant) = spec.quantize.filter(|_| select(&tensor.name)) {
                ResidentProjection::GroupQuantized {
                    bits: quant.bits(),
                    group_size: crate::quant::GROUP_SIZE as usize,
                }
            } else {
                ResidentProjection::Stored
            }
        })
    };
    let conditioning_bytes = project(
        &root.join("text_encoder"),
        &crate::convert::is_text_encoder_quant_target,
    )?;
    let transformer_bytes = project(&root.join("transformer"), &|name| {
        crate::convert::is_transformer_quant_target(name)
    })?;
    let decoder_bytes = project(&root.join("vae"), &|_| false)?;
    let overlay_bytes = match &spec.control {
        Some(mlx_gen::WeightsSource::Dir(path)) | Some(mlx_gen::WeightsSource::File(path)) => {
            let base_bits = crate::model::effective_base_quant_bits(spec, root, provider_id)?;
            let branch_bits = crate::memory::control_branch_quant_bits(base_bits);
            projected_safetensors_bytes(path, |_| match branch_bits {
                Some(bits) => ResidentProjection::GroupQuantized {
                    bits,
                    group_size: crate::quant::GROUP_SIZE as usize,
                },
                None => ResidentProjection::Stored,
            })?
        }
        None => 0,
    };
    let resident_components = (overlay_bytes > 0)
        .then(|| MemoryResidentComponent {
            id: "pose_control_branch".to_owned(),
            kind: MemoryComponentKind::ControlBranch,
            resident_bytes: overlay_bytes,
            bounded_by: None,
        })
        .into_iter()
        .collect();
    Ok((
        MemoryAssetFacts {
            base_bytes: conditioning_bytes
                .saturating_add(transformer_bytes)
                .saturating_add(decoder_bytes),
            conditioning_bytes,
            transformer_bytes,
            decoder_bytes,
            overlay_bytes,
        },
        resident_components,
    ))
}

pub fn safety_check(
    contract: &MemoryProviderContract,
    precision: mlx_gen::Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        // The Krea pose-control composition deliberately has no PiD decoder (its heavy bundle is the
        // base VAE plus the pose branch). Reject the flag explicitly instead of letting the residency
        // seam ignore it and execute a native decode under a PiD-labelled request.
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{}: PiD decode is not implemented for pose control",
                contract.provider_id
            )));
        }
        if contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode) {
            let routes = decode_routes(&contract.provider_id)?;
            routes
                .validate(
                    false,
                    context.selection.parameters.decode_tile_edge,
                    context.selection.parameters.decode_overlap,
                )
                .map_err(CoreError::Unsupported)?;
        }
        Ok(())
    };
    mlx_gen::gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision,
            quant,
            component_precision_floors: &[],
        }),
        Some(&route_gate),
    )
}

pub fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match crate::model::effective_base_quant_tier(spec, &contract.provider_id) {
        Ok(quant) => safety_check(contract, spec.precision, quant, context),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let quant = crate::model::effective_base_quant_tier(spec, &contract.provider_id)?;
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: spec.precision,
            quant,
            component_precision_floors: &[],
        },
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: mlx_gen::gen_core::MemoryMode::ImageToImage,
            reference_count: 1,
            use_pid: false,
            has_phases: false,
            overlay: Some("pose-control".to_owned()),
        },
    )?;
    Ok(vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(context)])
}

pub fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> CoreResult<Option<Box<dyn MemoryRequestScope>>> {
    begin_request_with_cleanup(
        provider_id,
        contract,
        spec.precision,
        crate::model::effective_base_quant_tier(spec, provider_id)?,
        context,
        mlx_gen::request_scope::MlxScopeCleanup::None,
    )
}

pub fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: mlx_gen::Precision,
    quant: Option<mlx_gen::Quant>,
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

fn begin_request_with_cleanup(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: mlx_gen::Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } =
        safety_check(contract, precision, quant, context)
    {
        return Err(CoreError::Unsupported(reason));
    }
    let routes = decode_routes(provider_id)?;
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        crate::config::Krea2Config::turbo().num_layers,
        move |use_pid, edge, overlap| {
            routes
                .validate(use_pid, Some(edge), Some(overlap))
                .map_err(CoreError::Unsupported)
        },
    )?;
    config.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
    config.transformer_window = contract
        .engages(
            context.selection.strategy,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .then_some(context.selection.parameters.transformer_window_size)
        .flatten();
    Ok(Some(Box::new(
        mlx_gen::request_scope::MlxRequestScopeCore::with_cleanup(config, cleanup),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryMode, MemorySelection, MemoryStrategyParameters,
        Precision, Quant, WeightsSource,
    };

    fn write_control(path: &std::path::Path) {
        let mut header =
            br#"{"control.weight":{"dtype":"BF16","shape":[2,64],"data_offsets":[0,256]}}"#
                .to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 256]);
        std::fs::write(path, bytes).unwrap();
    }

    fn write_snapshot(root: &std::path::Path) {
        for component in ["text_encoder", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            write_control(&dir.join("model.safetensors"));
        }
    }

    #[test]
    fn staged_residency_and_synchronized_release_require_a_sequential_control_load() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);

        let resident_spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let resident = memory_strategy_contract("krea_2_turbo_control", &resident_spec).unwrap();
        assert_eq!(
            resident
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert!(!resident.lifecycle.synchronized_phase_release);
        assert_eq!(
            resident
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented,
            "bounded decode is independent of the component load policy"
        );

        let sequential_spec = resident_spec.with_offload_policy(mlx_gen::OffloadPolicy::Sequential);
        let sequential =
            memory_strategy_contract("krea_2_turbo_control", &sequential_spec).unwrap();
        assert_eq!(
            sequential
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        assert!(sequential.lifecycle.synchronized_phase_release);
        assert_eq!(
            sequential
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );

        let streamable_spec =
            sequential_spec.with_load_shape(mlx_gen::gen_core::LoadShape::DeferredMaterialization);
        let streamable =
            memory_strategy_contract("krea_2_turbo_control", &streamable_spec).unwrap();
        for strategy in [
            MemoryStrategy::Resident,
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                streamable.capability(strategy).unwrap().support,
                MemoryStrategySupport::Implemented,
                "{strategy:?} must be executable on the deferred control composition"
            );
        }
        assert_eq!(
            streamable
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .parameters
                .transformer_window_components,
            vec![TransformerComponent::Dit],
            "the seven-block pose overlay stays explicitly resident; the reopenable base DiT is windowed"
        );
        assert_eq!(
            streamable.engaged_composition(MemoryStrategy::BoundedTransformerResidency),
            vec![
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ]
        );
    }

    #[test]
    fn prepacked_q8_pose_without_an_override_accepts_only_the_actual_tier() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":8,"group_size":64}}"#,
        )
        .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let contract = memory_strategy_contract("krea_2_turbo_control", &spec).unwrap();
        let calibration = contract.calibration.as_ref().unwrap();
        let context_for = |quant| MemoryRunContext {
            selection: MemorySelection {
                strategy: MemoryStrategy::Resident,
                parameters: MemoryStrategyParameters::default(),
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant,
                    component_precision_floors: &[],
                },
            },
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint.clone(),
            load_shape: calibration.load_shape,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 512,
                height: 512,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 1024,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 512,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "test".to_owned(),
        };

        assert_eq!(
            registered_safety_check(&spec, &contract, &context_for(Some(Quant::Q8))),
            MemorySafetyDecision::Accept
        );
        for wrong in [None, Some(Quant::Q4)] {
            assert!(matches!(
                registered_safety_check(&spec, &contract, &context_for(wrong)),
                MemorySafetyDecision::Reject { reason }
                    if reason.contains("does not match loaded tier")
            ));
        }
    }

    #[test]
    fn q4_base_projects_pose_overlay_at_the_declared_q8_floor() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        std::fs::write(root.join("transformer/config.json"), "{}").unwrap();
        let control = root.join("control.safetensors");
        write_control(&control);
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_quant(Quant::Q4)
            .with_control(WeightsSource::File(control));
        let contract = memory_strategy_contract("krea_2_turbo_control", &spec).unwrap();
        // Q8: 128 code bytes + two 2x1 bf16 tables (8 bytes). A uniform Q4 projection would be 72.
        assert_eq!(contract.asset_facts.overlay_bytes, 136);
        assert_eq!(contract.asset_facts.conditioning_bytes, 256);
        assert_eq!(contract.asset_facts.transformer_bytes, 256);
        assert_eq!(contract.asset_facts.decoder_bytes, 256);
        assert_eq!(contract.asset_facts.base_bytes, 768);
        assert_eq!(contract.auxiliary_resident_bytes(), 136);
        assert!(contract.conformance_errors().is_empty());
    }

    #[test]
    fn empty_pose_base_component_cannot_be_reported_as_zero() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        std::fs::remove_file(root.join("vae/model.safetensors")).unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        assert!(memory_strategy_contract("krea_2_turbo_control", &spec).is_err());
        assert!(weights_free_memory_strategy_contract("krea_2_turbo_control", &spec).is_ok());
    }
}
