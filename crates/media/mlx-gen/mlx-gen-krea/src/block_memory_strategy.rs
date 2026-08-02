//! SC-16352: request-scoped memory contract for the four base Krea 2 providers.
//!
//! Pose control has its own contract and seven-block control branch. This module covers the shared
//! 28-block base DiT used by Turbo, Raw, and their edit surfaces.
//!
//! The production domain is one block. Real `krea_2_turbo` weights at 512²/1 step measured the full
//! request (max of conditioning, denoise, decode) against an otherwise-identical Sequential + deferred
//! resident-stack attribution control:
//!
//! | tier / overlay | resident request | window 1 request | reduction |
//! |---|---:|---:|---:|
//! | q4 base | 11.928 GiB | 5.555 GiB | 53.4% |
//! | q8 base | 17.748 GiB | 5.715 GiB | 67.8% |
//! | bf16 base | 28.660 GiB | 8.383 GiB | 70.8% |
//! | q4 LoRA | 12.141 GiB | 5.768 GiB | 52.5% |
//! | q4 LoKr | 13.383 GiB | 7.010 GiB | 47.6% |
//!
//! Every windowed image was byte-identical to its resident control. Low-rank adapters are captured
//! and replayed per materialized block. A dense `.diff`/`.diff_b` patch is excluded at contract build
//! time because it mutates the resident base and cannot be reconstructed from the pristine snapshot.
//! The calibrated rung-4 key includes `engaged_composition=[Resident, StagedResidency,
//! BoundedTransformerResidency]`: this loader reopens components between phases, so block streaming
//! is valid only when staged residency is active in the same request.

use mlx_gen::asset_facts::{projected_safetensors_bytes, ResidentProjection};
#[cfg(test)]
use mlx_gen::gen_core::MemoryGeometry;
use mlx_gen::gen_core::{
    Error as CoreError, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryNumericTier, MemoryPhase,
    MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemorySafetyDecision,
    MemoryStrategy, MemoryStrategySupport, Result as CoreResult, TransformerComponent,
};
use mlx_gen::{LoadShape, LoadSpec, OffloadPolicy, Precision, WeightsSource};

pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;
pub const MEMORY_CALIBRATION_FINGERPRINT: &str =
    "krea-2-mlx-request-peak-block-residency-2026-08-01-v1";

pub(crate) fn is_streamable_spec(provider_id: &str, spec: &LoadSpec) -> CoreResult<bool> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Ok(false);
    };
    let mut plan = crate::model::resolve_load_plan(spec, root, provider_id)?;
    plan.streamable_transformer = matches!(spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && !crate::model::adapters_have_diff_patch(&spec.adapters)
        && plan.load_time_quant_bits.is_none();
    Ok(plan.streamable_transformer)
}

fn resolved_load_plan(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<crate::model::ResolvedLoadPlan> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Msg(
            "krea memory facts require a snapshot directory".to_owned(),
        ));
    };
    let mut plan = crate::model::resolve_load_plan(spec, root, provider_id)?;
    plan.streamable_transformer = matches!(spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && !crate::model::adapters_have_diff_patch(&spec.adapters)
        && plan.load_time_quant_bits.is_none();
    Ok(plan)
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    Ok(memory_strategy_contract_with_plan(provider_id, spec)?.0)
}

pub(crate) fn memory_strategy_contract_with_plan(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<(MemoryProviderContract, crate::model::ResolvedLoadPlan)> {
    let _ = crate::model::component_footprint(spec)?;
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(CoreError::Msg(
            "krea memory facts require a snapshot directory".to_owned(),
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
    let components = mlx_gen::PerComponentBytes {
        text_encoder: project(
            &root.join("text_encoder"),
            &crate::convert::is_text_encoder_quant_target,
        )?,
        dit: project(&root.join("transformer"), &|name| {
            crate::convert::is_transformer_quant_target(name)
        })?,
        vae: project(&root.join("vae"), &|_| false)?,
    };
    let plan = resolved_load_plan(provider_id, spec)?;
    Ok((
        memory_strategy_contract_with_components(
            provider_id,
            spec,
            components,
            plan.streamable_transformer,
        ),
        plan,
    ))
}

/// Declaration-equivalent contract used only by weights-free registry conformance.
pub(crate) fn weights_free_memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let plan = resolved_load_plan(provider_id, spec)?;
    Ok(memory_strategy_contract_with_components(
        provider_id,
        spec,
        Default::default(),
        plan.streamable_transformer,
    ))
}

fn memory_strategy_contract_with_components(
    provider_id: &str,
    spec: &LoadSpec,
    components: mlx_gen::PerComponentBytes,
    streamable: bool,
) -> MemoryProviderContract {
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::MlxMetal {
            bounded_wired_residency: true,
            lazy_or_mmap_materialization: true,
            explicit_evaluation_and_synchronization: true,
            cache_eviction: true,
        },
    );
    contract.load_shape = spec.load_shape;
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::TransformerWindowSize,
        ],
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        MEMORY_CALIBRATION_FINGERPRINT,
        spec.load_shape,
    ));
    contract.asset_facts.base_bytes = components
        .text_encoder
        .saturating_add(components.dit)
        .saturating_add(components.vae);
    contract.asset_facts.conditioning_bytes = components.text_encoder;
    contract.asset_facts.transformer_bytes = components.dit;
    contract.asset_facts.decoder_bytes = components.vae;
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: matches!(spec.offload_policy, OffloadPolicy::Sequential),
        transformer_window_materialization: streamable,
        ..Default::default()
    };
    if matches!(spec.offload_policy, OffloadPolicy::Sequential) {
        contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::StagedResidency)
            .expect("compatibility contract contains every strategy")
            .support = MemoryStrategySupport::Implemented;
    }
    if streamable {
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            mlx_gen::gen_core::MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: mlx_gen::gen_core::MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
        let capability = contract
            .strategies
            .iter_mut()
            .find(|capability| capability.strategy == MemoryStrategy::BoundedTransformerResidency)
            .expect("compatibility contract contains every strategy");
        capability.support = MemoryStrategySupport::Implemented;
        capability.parameters.transformer_window_sizes = vec![TRANSFORMER_WINDOW_SIZE];
        capability.parameters.transformer_window_components = vec![TransformerComponent::Dit];
    }
    contract
}

/// Exact contract for the supported community single-file DiT composition. The native I8 format is
/// dequantized projection-by-projection to bf16, while its scale/descriptor tensors are consumed and
/// dropped; the text encoder and VAE remain sourced from the resident base snapshot.
pub(crate) fn native_memory_strategy_contract(
    provider_id: &str,
    dit_file: &std::path::Path,
    base_snapshot_dir: &std::path::Path,
) -> CoreResult<MemoryProviderContract> {
    let base_spec = LoadSpec::new(WeightsSource::Dir(base_snapshot_dir.to_path_buf()));
    let mut contract = memory_strategy_contract(provider_id, &base_spec).map_err(|error| {
        CoreError::Msg(format!(
            "{provider_id}: native base snapshot asset facts for '{}': {error}",
            base_snapshot_dir.display()
        ))
    })?;
    let transformer_bytes = projected_safetensors_bytes(dit_file, |tensor| {
        if tensor.name.ends_with(".weight_scale") || tensor.name.ends_with(".comfy_quant") {
            ResidentProjection::Omit
        } else if tensor.dtype == mlx_gen::gen_core::weightsmeta::Dtype::I8 {
            ResidentProjection::Bfloat16
        } else {
            ResidentProjection::Stored
        }
    })
    .map_err(|error| {
        CoreError::Msg(format!(
            "{provider_id}: native DiT asset facts for '{}': {error}",
            dit_file.display()
        ))
    })?;
    contract.asset_facts.transformer_bytes = transformer_bytes;
    contract.asset_facts.base_bytes = contract
        .asset_facts
        .conditioning_bytes
        .checked_add(transformer_bytes)
        .and_then(|bytes| bytes.checked_add(contract.asset_facts.decoder_bytes))
        .ok_or_else(|| CoreError::Msg("krea native resident byte sum overflow".to_owned()))?;
    Ok(contract)
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{}: Krea rung 4 is calibrated for native VAE decode, not the PiD overlay",
                contract.provider_id
            )));
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

pub(crate) fn registered_safety_check(
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

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let quant = crate::model::effective_base_quant_tier(spec, &contract.provider_id)?;
    let is_edit = contract.provider_id.contains("edit");
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: spec.precision,
            quant,
            component_precision_floors: &[],
        },
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: if is_edit {
                mlx_gen::gen_core::MemoryMode::Edit
            } else {
                mlx_gen::gen_core::MemoryMode::TextToImage
            },
            reference_count: u32::from(is_edit),
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )?;
    Ok(vec![mlx_gen::gen_core::MemoryBehaviorFixture::new(context)])
}

pub(crate) fn registered_begin_request(
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

pub(crate) fn begin_request(
    provider_id: &'static str,
    contract: &MemoryProviderContract,
    precision: Precision,
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
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
    cleanup: mlx_gen::request_scope::MlxScopeCleanup,
) -> CoreResult<Option<Box<dyn MemoryRequestScope + 'static>>> {
    if let MemorySafetyDecision::Reject { reason } =
        safety_check(contract, precision, quant, context)
    {
        return Err(CoreError::Unsupported(reason));
    }
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        crate::config::Krea2Config::turbo().num_layers,
        |_use_pid, _edge, _overlap| {
            Err(CoreError::Unsupported(
                "krea: bounded decode is not implemented by the base provider".into(),
            ))
        },
    )?;
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
        MemoryBudget, MemoryCacheState, MemoryMode, MemorySelection, MemoryStrategy,
        MemoryStrategyParameters, MemoryStrategySupport,
    };
    use mlx_gen::{AdapterKind, AdapterSpec, Quant};
    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn write_minimal_safetensors(path: &std::path::Path) {
        let mut header = br#"{"probe":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 2]);
        std::fs::write(path, bytes).unwrap();
    }

    fn write_native_i8_safetensors(path: &std::path::Path) {
        let mut header = br#"{
            "model.diffusion_model.proj.weight":{"dtype":"I8","shape":[2,64],"data_offsets":[0,128]},
            "model.diffusion_model.proj.weight_scale":{"dtype":"F32","shape":[2],"data_offsets":[128,136]},
            "model.diffusion_model.proj.comfy_quant":{"dtype":"U8","shape":[2],"data_offsets":[136,138]}
        }"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend([0_u8; 138]);
        std::fs::write(path, bytes).unwrap();
    }

    fn fixture() -> (std::path::PathBuf, LoadSpec) {
        let root = std::env::temp_dir().join(format!(
            "mlx_gen_krea_sc16352_{}_{}",
            std::process::id(),
            FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
        ));
        for component in ["text_encoder", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            write_minimal_safetensors(&dir.join("model.safetensors"));
        }
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization)
            .with_quant(Quant::Q4);
        (root, spec)
    }

    fn write_diff_patch(path: &std::path::Path) {
        let header = br#"{"transformer_blocks.0.attn.to_q.diff":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::with_capacity(8 + header.len() + 4);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header);
        bytes.extend_from_slice(&0.0_f32.to_le_bytes());
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn identical_fingerprint_is_separated_by_typed_load_shape() {
        let (root, deferred_spec) = fixture();
        let deferred = memory_strategy_contract("krea_2_turbo", &deferred_spec).unwrap();
        let mut eager_spec = deferred_spec;
        eager_spec.load_shape = LoadShape::EagerMaterialization;
        let eager = memory_strategy_contract("krea_2_turbo", &eager_spec).unwrap();
        assert_eq!(
            deferred.calibration.as_ref().unwrap().fingerprint,
            eager.calibration.as_ref().unwrap().fingerprint
        );
        assert_ne!(
            deferred.calibration.as_ref().unwrap().load_shape,
            eager.calibration.as_ref().unwrap().load_shape
        );
        std::fs::remove_dir_all(root).ok();
    }

    fn resident_context(
        contract: &MemoryProviderContract,
        quant: Option<Quant>,
    ) -> MemoryRunContext {
        let calibration = contract.calibration.as_ref().unwrap();
        MemoryRunContext {
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
        }
    }

    #[test]
    fn prepacked_q4_without_an_override_binds_registration_to_the_actual_tier() {
        let (root, mut spec) = fixture();
        spec.quantize = None;
        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        assert_eq!(
            registered_safety_check(
                &spec,
                &contract,
                &resident_context(&contract, Some(Quant::Q4))
            ),
            MemorySafetyDecision::Accept
        );
        for wrong in [None, Some(Quant::Q8)] {
            assert!(matches!(
                registered_safety_check(&spec, &contract, &resident_context(&contract, wrong)),
                MemorySafetyDecision::Reject { reason }
                    if reason.contains("does not match loaded tier")
            ));
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn rung_four_declares_and_engages_staged_residency_in_the_same_request() {
        let (root, spec) = fixture();
        let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        assert_eq!(
            contract.engaged_composition(MemoryStrategy::BoundedTransformerResidency),
            vec![
                MemoryStrategy::Resident,
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedTransformerResidency,
            ]
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn sequential_deferred_snapshot_advertises_the_exact_dit_window() {
        let (root, spec) = fixture();
        let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        assert!(contract.conformance_errors().is_empty());
        let staged = contract
            .capability(MemoryStrategy::StagedResidency)
            .unwrap();
        assert_eq!(staged.support, MemoryStrategySupport::Implemented);
        let rung = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(rung.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            rung.parameters.transformer_window_sizes,
            [TRANSFORMER_WINDOW_SIZE]
        );
        assert_eq!(
            rung.parameters.transformer_window_components,
            [TransformerComponent::Dit]
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn missing_required_component_directory_fails_closed() {
        let (root, spec) = fixture();
        std::fs::remove_dir_all(root.join("text_encoder")).unwrap();
        assert!(memory_strategy_contract("krea_2_turbo", &spec).is_err());
        assert!(weights_free_memory_strategy_contract("krea_2_turbo", &spec).is_ok());
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn native_required_file_rejects_missing_empty_and_corrupt_sources() {
        let (root, _) = fixture();
        let native = root.join("native.safetensors");
        let missing = native_memory_strategy_contract("krea_2_turbo", &native, &root)
            .unwrap_err()
            .to_string();
        assert!(missing.contains("native DiT asset facts"), "{missing}");
        std::fs::write(&native, []).unwrap();
        let empty = native_memory_strategy_contract("krea_2_turbo", &native, &root)
            .unwrap_err()
            .to_string();
        assert!(empty.contains("native DiT asset facts"), "{empty}");
        std::fs::write(&native, b"corrupt").unwrap();
        let corrupt = native_memory_strategy_contract("krea_2_turbo", &native, &root)
            .unwrap_err()
            .to_string();
        assert!(corrupt.contains("native DiT asset facts"), "{corrupt}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn low_rank_overlay_is_admissible_but_dense_diff_patch_is_not() {
        let (root, mut spec) = fixture();
        let low_rank = root.join("low-rank.safetensors");
        std::fs::write(&low_rank, [0_u8; 8]).unwrap();
        spec.adapters = vec![AdapterSpec::new(low_rank, 1.0, AdapterKind::Lora)];
        assert!(is_streamable_spec("krea_2_turbo", &spec).unwrap());

        let diff = root.join("dense-diff.safetensors");
        write_diff_patch(&diff);
        spec.adapters = vec![AdapterSpec::new(diff, 1.0, AdapterKind::Lora)];
        assert!(!is_streamable_spec("krea_2_turbo", &spec).unwrap());
        let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn rung_four_resolves_load_time_quantization_instead_of_the_override_presence() {
        let (root, spec) = fixture();

        for (quant, bits) in [(Quant::Q4, 4), (Quant::Q8, 8)] {
            std::fs::write(
                root.join("transformer/config.json"),
                format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            )
            .unwrap();
            let mut packed = spec.clone();
            packed.quantize = Some(quant);
            assert!(
                is_streamable_spec("krea_2_turbo", &packed).unwrap(),
                "a matching prepacked Q{bits} override is a no-op and must remain streamable"
            );
            let (contract, plan) =
                memory_strategy_contract_with_plan("krea_2_turbo", &packed).unwrap();
            assert!(contract.lifecycle.transformer_window_materialization);
            assert_eq!(plan.effective_quant, Some(quant));
            assert_eq!(plan.load_time_quant_bits, None);
            assert!(plan.streamable_transformer);
        }

        std::fs::write(root.join("transformer/config.json"), "{}").unwrap();
        assert!(
            !is_streamable_spec("krea_2_turbo", &spec).unwrap(),
            "a dense snapshot requiring per-window Q4 packing must not be streamable"
        );
        let dense = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
        assert!(!dense.lifecycle.transformer_window_materialization);
        assert_eq!(
            dense
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );

        std::fs::write(
            root.join("transformer/config.json"),
            r#"{"quantization":{"bits":8,"group_size":64}}"#,
        )
        .unwrap();
        let mismatch = memory_strategy_contract("krea_2_turbo", &spec)
            .unwrap_err()
            .to_string();
        assert!(
            mismatch.contains("Q8") && mismatch.contains("Q4"),
            "{mismatch}"
        );

        let mut no_override = spec.clone();
        no_override.quantize = None;
        std::fs::write(root.join("transformer/config.json"), "{ malformed").unwrap();
        let eligibility_error = is_streamable_spec("krea_2_turbo", &no_override)
            .unwrap_err()
            .to_string();
        assert!(
            eligibility_error.contains("packed quant"),
            "{eligibility_error}"
        );
        let contract_error = memory_strategy_contract("krea_2_turbo", &no_override)
            .unwrap_err()
            .to_string();
        assert!(contract_error.contains("packed quant"), "{contract_error}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn non_reopenable_or_wrong_loader_shapes_do_not_advertise_rung_four() {
        let (root, base) = fixture();
        let mut resident = base.clone();
        resident.offload_policy = OffloadPolicy::Resident;
        let mut eager = base.clone();
        eager.load_shape = LoadShape::EagerMaterialization;
        let file = LoadSpec::new(WeightsSource::File(root.join("single.safetensors")))
            .with_offload_policy(OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization);
        for spec in [resident, eager] {
            let contract = memory_strategy_contract("krea_2_turbo", &spec).unwrap();
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing
            );
        }
        let error = memory_strategy_contract("krea_2_turbo", &file)
            .unwrap_err()
            .to_string();
        assert!(error.contains("snapshot directory"), "{error}");
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn native_i8_contract_counts_bf16_materialization_and_omits_source_companions() {
        let (root, _) = fixture();
        let native = root.join("native.safetensors");
        write_native_i8_safetensors(&native);
        let contract = native_memory_strategy_contract("krea_2_turbo", &native, &root).unwrap();
        assert_eq!(contract.asset_facts.conditioning_bytes, 2);
        assert_eq!(contract.asset_facts.decoder_bytes, 2);
        assert_eq!(contract.asset_facts.transformer_bytes, 2 * 64 * 2);
        assert_eq!(contract.asset_facts.base_bytes, 260);
        assert!(contract.conformance_errors().is_empty());
        std::fs::remove_dir_all(root).ok();
    }
}
