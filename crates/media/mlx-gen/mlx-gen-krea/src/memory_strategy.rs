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
    if matches!(spec.weights, mlx_gen::WeightsSource::File(_)) {
        crate::model_control::validate_control_spec(spec)
            .map_err(|error| CoreError::Msg(error.to_string()))?;
        let base = mlx_gen::require_base_snapshot(spec, provider_id)?;
        return native_memory_strategy_contract_from_spec(provider_id, spec, base, false);
    }
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
        decode_geometry_policy_authoritative: false,
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

/// Exact contract for the community single-file DiT **pose-control** composition (the control twin of
/// `crate::block_memory_strategy::native_memory_strategy_contract`): the imported DiT is read from
/// `dit_file` (its native I8 projections are dequantized to bf16 at load, scale/descriptor tensors
/// consumed and dropped), the text encoder and VAE come from the resident base tier, and the pose
/// overlay loads dense — the native DiT is dense in memory, and a dense base carries a dense branch
/// (`crate::memory::control_branch_quant_bits(None)`), so the overlay bytes are its stored bytes.
/// Same `provider_id` + calibration fingerprint as the snapshot control contract: implementation and
/// provider evidence stay on one identity. The evidence matrix has no load-source axis, though, so a
/// `Dir`-measured rung-4 cell must not be reported as a `File` measurement until this re-openable path
/// is measured directly. The loader's reopenable mechanism stays available for its source smoke, but
/// the registry contract must pass `streamable = false` until File evidence is promoted.
pub(crate) fn native_memory_strategy_contract_from_spec(
    provider_id: &str,
    spec: &LoadSpec,
    base_snapshot_dir: &std::path::Path,
    streamable: bool,
) -> CoreResult<MemoryProviderContract> {
    let dit_file = match &spec.weights {
        mlx_gen::WeightsSource::File(path) => path,
        mlx_gen::WeightsSource::Dir(path) => {
            return Err(CoreError::Msg(format!(
                "{provider_id}: native pose-control facts require a single-file DiT, not {}",
                path.display()
            )))
        }
    };
    let control = mlx_gen::require_control(spec, provider_id, "Krea 2 pose control overlay")?;
    let stored = |path: &std::path::Path, what: &str| -> CoreResult<u64> {
        projected_safetensors_bytes(path, |_| ResidentProjection::Stored).map_err(|error| {
            CoreError::Msg(format!(
                "{provider_id}: native {what} asset facts for '{}': {error}",
                path.display()
            ))
        })
    };
    let text_plan =
        crate::model::resolve_load_plan_for_component(spec, base_snapshot_dir, provider_id, false)?;
    let conditioning_bytes =
        projected_safetensors_bytes(base_snapshot_dir.join("text_encoder"), |tensor| {
            if let Some(bits) = text_plan
                .load_time_quant_bits
                .filter(|_| crate::convert::is_text_encoder_quant_target(&tensor.name))
            {
                ResidentProjection::GroupQuantized {
                    bits,
                    group_size: crate::quant::GROUP_SIZE as usize,
                }
            } else {
                ResidentProjection::Stored
            }
        })
        .map_err(|error| {
            CoreError::Msg(format!(
                "{provider_id}: native base text-encoder asset facts for '{}': {error}",
                base_snapshot_dir.join("text_encoder").display()
            ))
        })?;
    let decoder_bytes = stored(&base_snapshot_dir.join("vae"), "base VAE")?;
    let transformer_bytes = spec.read_file_unchanged_if_prepared(dit_file, |p| {
        crate::block_memory_strategy::native_dit_transformer_bytes(provider_id, p, spec.quantize)
    })?;
    let branch_bits =
        crate::memory::control_branch_quant_bits(spec.quantize.map(mlx_gen::Quant::bits));
    let projected_control = |path: &std::path::Path| {
        projected_safetensors_bytes(path, |tensor| {
            if let Some(bits) =
                branch_bits.filter(|_| crate::control::is_control_quant_target(&tensor.name))
            {
                ResidentProjection::GroupQuantized {
                    bits,
                    group_size: crate::quant::GROUP_SIZE as usize,
                }
            } else {
                ResidentProjection::Stored
            }
        })
        .map_err(|error| {
            CoreError::Msg(format!(
                "{provider_id}: native pose control overlay asset facts for '{}': {error}",
                path.display()
            ))
        })
    };
    let (control_path, overlay_bytes) = match control {
        mlx_gen::WeightsSource::Dir(path) => (path, projected_control(path)?),
        mlx_gen::WeightsSource::File(path) => (
            path,
            spec.read_file_unchanged_if_prepared(path, projected_control)?,
        ),
    };
    if overlay_bytes == 0 {
        return Err(CoreError::Msg(format!(
            "{provider_id}: pose control overlay '{}' contains no tensor bytes",
            control_path.display()
        )));
    }
    let resident_components = vec![MemoryResidentComponent {
        id: "pose_control_branch".to_owned(),
        kind: MemoryComponentKind::ControlBranch,
        resident_bytes: overlay_bytes,
        bounded_by: None,
    }];
    let asset_facts = MemoryAssetFacts {
        base_bytes: conditioning_bytes
            .saturating_add(transformer_bytes)
            .saturating_add(decoder_bytes),
        conditioning_bytes,
        transformer_bytes,
        decoder_bytes,
        overlay_bytes,
    };
    memory_strategy_contract_with_asset_facts(
        provider_id,
        spec,
        asset_facts,
        resident_components,
        streamable,
    )
}

/// Compatibility shim for the former bespoke imported-control entrypoint.
#[cfg(test)]
pub(crate) fn native_memory_strategy_contract(
    provider_id: &str,
    dit_file: &std::path::Path,
    base_snapshot_dir: &std::path::Path,
    control: &std::path::Path,
) -> CoreResult<MemoryProviderContract> {
    let spec = LoadSpec::new(mlx_gen::WeightsSource::File(dit_file.to_path_buf()))
        .with_component(
            mlx_gen::BASE_SNAPSHOT_COMPONENT,
            mlx_gen::WeightsSource::Dir(base_snapshot_dir.to_path_buf()),
        )
        .with_control(mlx_gen::WeightsSource::File(control.to_path_buf()));
    native_memory_strategy_contract_from_spec(provider_id, &spec, base_snapshot_dir, false)
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
            && !crate::model::adapters_have_diff_patch_for_spec(spec)?
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
        Some(mlx_gen::WeightsSource::Dir(path)) => {
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
        Some(mlx_gen::WeightsSource::File(path)) => {
            let base_bits = crate::model::effective_base_quant_bits(spec, root, provider_id)?;
            let branch_bits = crate::memory::control_branch_quant_bits(base_bits);
            spec.read_file_unchanged_if_prepared(path, |p| {
                projected_safetensors_bytes(p, |_| match branch_bits {
                    Some(bits) => ResidentProjection::GroupQuantized {
                        bits,
                        group_size: crate::quant::GROUP_SIZE as usize,
                    },
                    None => ResidentProjection::Stored,
                })
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
    fn native_control_contract_keeps_the_builtin_provider_identity_and_components() {
        // The imported single-file pose-control contract must resolve evidence through the SAME
        // provider identity as the snapshot lane: same provider_id, same calibration fingerprint,
        // same resident pose-branch component — never a bespoke "imported" identity that would
        // orphan promoted evidence. Resident-only: no staged residency, no windowing.
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        let dit = root.join("imported-dit.safetensors");
        write_control(&dit);
        let overlay = root.join("control.safetensors");
        write_control(&overlay);

        let contract =
            native_memory_strategy_contract("krea_2_turbo_control", &dit, &root, &overlay).unwrap();
        assert_eq!(contract.provider_id, "krea_2_turbo_control");
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            MEMORY_CALIBRATION_FINGERPRINT
        );
        let components = contract.resident_components();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].id, "pose_control_branch");
        assert_eq!(components[0].kind, MemoryComponentKind::ControlBranch);
        assert!(components[0].resident_bytes > 0);
        // The transformer term is the native FILE's projection, not the base tier's transformer dir;
        // the fixture stores one dense bf16 tensor, so projected == stored bytes.
        assert_eq!(contract.asset_facts.transformer_bytes, 256);
        assert_eq!(
            contract.asset_facts.base_bytes,
            contract.asset_facts.conditioning_bytes
                + contract.asset_facts.transformer_bytes
                + contract.asset_facts.decoder_bytes
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        // A missing / zero-byte overlay is refused loudly — the branch is a REQUIRED component.
        let missing = native_memory_strategy_contract(
            "krea_2_turbo_control",
            &dit,
            &root,
            &root.join("missing-control.safetensors"),
        )
        .expect_err("missing overlay must fail")
        .to_string();
        assert!(missing.contains("pose control overlay"), "got: {missing}");
    }

    #[test]
    fn registry_file_control_contract_uses_file_bytes_but_withholds_rung_four() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        let dit = root.join("imported-dit.safetensors");
        write_control(&dit);
        let overlay = root.join("control.safetensors");
        write_control(&overlay);
        let spec = LoadSpec::new(WeightsSource::File(dit))
            .with_component(mlx_gen::BASE_SNAPSHOT_COMPONENT, WeightsSource::Dir(root))
            .with_control(WeightsSource::File(overlay))
            .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::gen_core::LoadShape::DeferredMaterialization);

        let contract = memory_strategy_contract("krea_2_turbo_control", &spec).unwrap();
        assert_eq!(contract.provider_id, "krea_2_turbo_control");
        assert_eq!(contract.asset_facts.transformer_bytes, 256);
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing,
            "the pinned File mechanism needs source-specific evidence before authorization"
        );
        assert!(!contract.lifecycle.transformer_window_materialization);
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
            optimization_authority: mlx_gen::gen_core::MemoryOptimizationAuthority::Calibrated,
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

    #[test]
    fn imported_pose_overlay_inventory_rejects_missing_empty_and_corrupt_files() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        let dit = root.join("imported.safetensors");
        write_control(&dit);
        let overlay = root.join("control.safetensors");
        let spec = LoadSpec::new(WeightsSource::File(dit))
            .with_component(
                mlx_gen::BASE_SNAPSHOT_COMPONENT,
                WeightsSource::Dir(root.clone()),
            )
            .with_control(WeightsSource::File(overlay.clone()));

        for (case, replacement) in [
            ("empty", Some(Vec::new())),
            ("corrupt", Some(b"corrupt".to_vec())),
            ("missing", None),
        ] {
            match replacement {
                Some(bytes) => std::fs::write(&overlay, bytes).unwrap(),
                None => {
                    if overlay.exists() {
                        std::fs::remove_file(&overlay).unwrap();
                    }
                }
            }
            let error = memory_strategy_contract("krea_2_turbo_control", &spec)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("pose control") || error.contains("safetensors"),
                "{case}: {error}"
            );
            write_control(&overlay);
        }
    }

    #[test]
    fn imported_file_contract_matches_the_control_loader_for_every_typed_field() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        write_snapshot(&root);
        let dit = root.join("imported.safetensors");
        write_control(&dit);
        let overlay = root.join("control.safetensors");
        write_control(&overlay);
        let valid = LoadSpec::new(WeightsSource::File(dit))
            .with_component(
                mlx_gen::BASE_SNAPSHOT_COMPONENT,
                WeightsSource::Dir(root.clone()),
            )
            .with_control(WeightsSource::File(overlay));

        let mut precision = valid.clone();
        precision.precision = Precision::Fp32;
        let mut extra_control = valid.clone();
        extra_control
            .extra_controls
            .push(WeightsSource::File(root.join("extra-control.safetensors")));
        let mut ip_adapter = valid.clone();
        ip_adapter.ip_adapter = Some(WeightsSource::Dir(root.join("ip-adapter")));
        let mut identity = valid.clone();
        identity.identity = Some(mlx_gen::gen_core::IdentityWeights::default());
        let mut text_encoder = valid.clone();
        text_encoder.text_encoder = Some(WeightsSource::Dir(root.join("external-text")));
        let mut pid = valid.clone();
        pid.pid = Some(mlx_gen::gen_core::PidWeights {
            checkpoint: WeightsSource::File(root.join("pid.safetensors")),
            gemma: WeightsSource::Dir(root.join("gemma")),
        });
        let mut unknown_component = valid.clone();
        unknown_component.components.insert(
            "unknown".into(),
            WeightsSource::File(root.join("unknown.safetensors")),
        );
        let mut missing_base = valid.clone();
        missing_base.components.clear();
        let accepted_adapter = valid.clone().with_adapters(vec![mlx_gen::AdapterSpec::new(
            root.join("adapter.safetensors"),
            1.0,
            mlx_gen::AdapterKind::Lora,
        )]);
        let accepted_deferred = valid
            .clone()
            .with_offload_policy(mlx_gen::OffloadPolicy::Sequential)
            .with_load_shape(mlx_gen::gen_core::LoadShape::DeferredMaterialization);

        for (case, spec, expected) in [
            ("valid", valid.clone(), true),
            ("adapter", accepted_adapter, true),
            ("deferred", accepted_deferred, true),
            ("precision", precision, false),
            ("quantize", valid.clone().with_quant(Quant::Q4), true),
            ("extra_control", extra_control, false),
            ("ip_adapter", ip_adapter, false),
            ("pid", pid, false),
            ("identity", identity, false),
            ("text_encoder", text_encoder, false),
            ("unknown_component", unknown_component, false),
            ("missing_base", missing_base, false),
        ] {
            let loader = crate::model_control::validate_control_spec(&spec).is_ok();
            let contract = memory_strategy_contract("krea_2_turbo_control", &spec).is_ok();
            assert_eq!(loader, expected, "loader validation for {case}");
            assert_eq!(contract, loader, "contract/loader parity for {case}");
        }
    }
}
