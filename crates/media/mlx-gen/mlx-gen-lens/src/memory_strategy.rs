//! Lens / Lens-Turbo MLX shared image-memory ladder.
//!
//! SC-15800's dense-bf16 text-encoder-only result remains an independent legacy measurement. The
//! full ladder uses a new identity and does not relabel that result as DiT, Both, Q4, decode, or
//! attention evidence.

use mlx_gen::gen_core::{
    Error as CoreError, MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryNumericTier, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, MemoryStrategyPrerequisite, MemoryStrategySupport,
    Result as CoreResult, TransformerComponent,
};
#[cfg(test)]
use mlx_gen::{gen_core::MemoryGeometry, GenerationRequest};
use mlx_gen::{LoadShape, LoadSpec, OffloadPolicy, Precision, WeightsSource};

pub const LEGACY_TEXT_ENCODER_FINGERPRINT: &str = "lens-text-encoder-window-2026-07-31-v1";
pub const MEMORY_CALIBRATION_FINGERPRINT: &str = "lens-mlx-shared-ladder-2026-08-03-v1";
pub const TEXT_ENCODER_WINDOW: u32 = 1;
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_OVERLAP: u32 = 128;
pub const ATTENTION_CHUNK_SIZE: u32 = 16_777_216;

fn decode_routes(provider_id: &str) -> CoreResult<mlx_gen_pid::DecodeRoutes> {
    mlx_gen_pid::DecodeRoutes::new_core(provider_id, [DECODE_TILE_EDGE], DECODE_OVERLAP)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CalibrationRoute {
    FullQ4Lens,
    LegacyDenseLensTurboTextEncoder,
    Unmeasured,
}

/// The exact load shape for which the measured production rung is executable and beneficial.
pub(crate) fn is_streamable_spec(spec: &LoadSpec) -> bool {
    matches!(spec.weights, WeightsSource::Dir(_))
        && matches!(spec.offload_policy, OffloadPolicy::Sequential)
        && matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.precision, Precision::Bf16)
        && spec.quantize.is_none()
        && spec.adapters.is_empty()
        && spec.pid.is_none()
}

fn base_streamable(spec: &LoadSpec) -> Option<&std::path::Path> {
    if spec.load_shape != LoadShape::DeferredMaterialization
        || spec.precision != Precision::Bf16
        || spec.pid.is_some()
    {
        return None;
    }
    match &spec.weights {
        WeightsSource::Dir(root) => Some(root),
        WeightsSource::File(_) => None,
    }
}

fn calibration_route(
    provider_id: &str,
    spec: &LoadSpec,
    text_streamable: bool,
    dit_streamable: bool,
) -> CalibrationRoute {
    if provider_id == "lens"
        && spec.quantize == Some(mlx_gen::Quant::Q4)
        && spec.adapters.is_empty()
        && text_streamable
        && dit_streamable
    {
        CalibrationRoute::FullQ4Lens
    } else if provider_id == "lens_turbo" && is_streamable_spec(spec) {
        CalibrationRoute::LegacyDenseLensTurboTextEncoder
    } else {
        CalibrationRoute::Unmeasured
    }
}

pub(crate) fn can_stream_text(spec: &LoadSpec) -> CoreResult<bool> {
    let Some(root) = base_streamable(spec) else {
        return Ok(false);
    };
    Ok(match spec.quantize {
        Some(quant) => {
            !mlx_gen::quant::needs_load_time_quant(root, "text_encoder", quant.bits(), "lens")?
        }
        None => true,
    })
}

pub(crate) fn can_stream_dit(spec: &LoadSpec) -> CoreResult<bool> {
    let Some(root) = base_streamable(spec) else {
        return Ok(false);
    };
    if !spec.adapters.is_empty() {
        return Ok(false);
    }
    Ok(match spec.quantize {
        Some(quant) => {
            !mlx_gen::quant::needs_load_time_quant(root, "transformer", quant.bits(), "lens")?
        }
        None => true,
    })
}

pub fn memory_strategy_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> CoreResult<MemoryProviderContract> {
    let text_streamable = can_stream_text(spec)?;
    let dit_streamable = can_stream_dit(spec)?;
    let calibration_route = calibration_route(provider_id, spec, text_streamable, dit_streamable);
    let footprint = crate::registry::component_footprint(spec)?;
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
    let mut formula_variables = vec![
        MemoryFormulaVariable::AssetBytes,
        MemoryFormulaVariable::PixelCount,
        MemoryFormulaVariable::BatchCount,
        MemoryFormulaVariable::ConditioningTokenCount,
        MemoryFormulaVariable::TransformerWindowSize,
    ];
    if calibration_route != CalibrationRoute::LegacyDenseLensTurboTextEncoder {
        formula_variables.extend([
            MemoryFormulaVariable::DecodeTileArea,
            MemoryFormulaVariable::AttentionChunkSize,
        ]);
    }
    contract.formula = MemoryFormulaKind::PhaseEnvelope {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        variables: formula_variables,
    };
    contract.calibration = match calibration_route {
        CalibrationRoute::FullQ4Lens => Some(MemoryCalibrationIdentity::new(
            MEMORY_CALIBRATION_FINGERPRINT,
            spec.load_shape,
        )),
        CalibrationRoute::LegacyDenseLensTurboTextEncoder => Some(MemoryCalibrationIdentity::new(
            LEGACY_TEXT_ENCODER_FINGERPRINT,
            spec.load_shape,
        )),
        CalibrationRoute::Unmeasured => None,
    };
    contract.asset_facts.base_bytes = footprint
        .text_encoder
        .saturating_add(footprint.dit)
        .saturating_add(footprint.vae);
    contract.asset_facts.conditioning_bytes = footprint.text_encoder;
    contract.asset_facts.transformer_bytes = footprint.dit;
    contract.asset_facts.decoder_bytes = footprint.vae;
    contract.lifecycle = match calibration_route {
        CalibrationRoute::LegacyDenseLensTurboTextEncoder => MemoryLifecycleCapabilities {
            phases: vec![
                MemoryPhase::Conditioning,
                MemoryPhase::Denoise,
                MemoryPhase::Decode,
            ],
            synchronized_phase_release: true,
            transformer_window_materialization: true,
            ..Default::default()
        },
        CalibrationRoute::FullQ4Lens | CalibrationRoute::Unmeasured => {
            MemoryLifecycleCapabilities {
                phases: vec![
                    MemoryPhase::Conditioning,
                    MemoryPhase::Denoise,
                    MemoryPhase::Decode,
                ],
                synchronized_phase_release: true,
                decode_tiling: true,
                attention_chunking: true,
                transformer_window_materialization: text_streamable || dit_streamable,
            }
        }
    };
    for capability in &mut contract.strategies {
        match capability.strategy {
            MemoryStrategy::Resident => {
                capability.support = MemoryStrategySupport::Implemented;
            }
            MemoryStrategy::StagedResidency
                if calibration_route == CalibrationRoute::FullQ4Lens =>
            {
                capability.support = MemoryStrategySupport::Implemented;
            }
            MemoryStrategy::BoundedDecode if calibration_route == CalibrationRoute::FullQ4Lens => {
                capability.support = MemoryStrategySupport::Implemented;
                capability.parameters.decode_tile_edges = vec![DECODE_TILE_EDGE];
                capability.parameters.decode_overlaps = vec![DECODE_OVERLAP];
            }
            MemoryStrategy::BoundedAttention
                if calibration_route == CalibrationRoute::FullQ4Lens =>
            {
                capability.support = MemoryStrategySupport::Implemented;
                capability.parameters.attention_chunk_sizes = vec![ATTENTION_CHUNK_SIZE];
            }
            MemoryStrategy::BoundedTransformerResidency
                if calibration_route == CalibrationRoute::FullQ4Lens =>
            {
                capability.support = MemoryStrategySupport::Implemented;
                capability.parameters.transformer_window_sizes = vec![TEXT_ENCODER_WINDOW];
                capability.parameters.transformer_window_components =
                    vec![TransformerComponent::Both];
            }
            MemoryStrategy::BoundedTransformerResidency
                if calibration_route == CalibrationRoute::LegacyDenseLensTurboTextEncoder =>
            {
                capability.support = MemoryStrategySupport::Implemented;
                capability.parameters.transformer_window_sizes = vec![TEXT_ENCODER_WINDOW];
                capability.parameters.transformer_window_components =
                    vec![TransformerComponent::TextEncoder];
            }
            _ => {}
        }
    }
    if calibration_route == CalibrationRoute::FullQ4Lens {
        contract.additional_prerequisites.push((
            MemoryStrategy::BoundedTransformerResidency,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        ));
    }
    Ok(contract)
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    precision: Precision,
    quant: Option<mlx_gen::Quant>,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let route_gate = || {
        if context.use_pid
            || contract.engages(context.selection.strategy, MemoryStrategy::BoundedDecode)
        {
            let routes = decode_routes(&contract.provider_id)?;
            routes
                .validate(
                    context.use_pid,
                    context.selection.parameters.decode_tile_edge,
                    context.selection.parameters.decode_overlap,
                )
                .map_err(|reason| {
                    let detail = if context.use_pid {
                        "the Lens rung-4 calibration covers native VAE decode only, not the PiD/Gemma overlay"
                    } else {
                        "Lens decode route validation failed"
                    };
                    CoreError::Unsupported(format!(
                        "{}: {detail}: {reason}",
                        contract.provider_id
                    ))
                })?;
        }
        if context.use_pid {
            return Err(CoreError::Unsupported(format!(
                "{}: the Lens rung-4 calibration covers native VAE decode only, not the PiD/Gemma overlay",
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
    safety_check(contract, spec.precision, spec.quantize, context)
}

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> CoreResult<Vec<mlx_gen::gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let full_ladder = contract
        .calibration
        .as_ref()
        .is_some_and(|identity| identity.fingerprint == MEMORY_CALIBRATION_FINGERPRINT);
    let context = mlx_gen::gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: spec.precision,
            quant: spec.quantize,
            component_precision_floors: &[],
        },
        mlx_gen::gen_core::MemoryBehaviorRoute {
            mode: mlx_gen::gen_core::MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: full_ladder && contract.engages(strategy, MemoryStrategy::StagedResidency),
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
        spec.quantize,
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
    let full_ladder = contract
        .calibration
        .as_ref()
        .is_some_and(|identity| identity.fingerprint == MEMORY_CALIBRATION_FINGERPRINT);
    let component = context.selection.parameters.window_component();
    let routes = decode_routes(provider_id)?;
    let transformer_blocks = match component {
        TransformerComponent::TextEncoder => crate::config::GptOssConfig::lens().num_layers,
        TransformerComponent::Dit | TransformerComponent::Both => {
            crate::dit::LensDitConfig::lens().num_layers
        }
    };
    let mut config = mlx_gen::request_scope::MlxRequestScopeConfig::new(
        provider_id,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        transformer_blocks,
        move |use_pid, edge, overlap| {
            routes
                .validate(use_pid, Some(edge), Some(overlap))
                .map_err(CoreError::Unsupported)
        },
    )?;
    config.attention_chunk_size = full_ladder.then_some(ATTENTION_CHUNK_SIZE);
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
        MemoryBudget, MemoryCacheState, MemoryMode, MemoryNumericTier, MemorySelection,
        MemoryStrategyParameters, MemoryStrategySupport, TransformerComponent,
        MEMORY_CALIBRATION_ABI,
    };
    use mlx_gen::{AdapterKind, AdapterSpec, PidWeights, Quant};
    use std::sync::atomic::{AtomicU64, Ordering};

    static FIXTURE_ID: AtomicU64 = AtomicU64::new(0);

    fn fixture(tmp: &tempfile::TempDir, bits: Option<i32>) -> (std::path::PathBuf, LoadSpec) {
        let root = tmp.path().join(format!(
            "mlx_gen_lens_sc15800_{}",
            FIXTURE_ID.fetch_add(1, Ordering::Relaxed)
        ));
        for component in ["text_encoder", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.safetensors"), [0_u8; 8]).unwrap();
        }
        for component in ["text_encoder", "transformer"] {
            let config = match bits {
                Some(bits) => {
                    format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#)
                }
                None => r#"{"dtype":"bfloat16"}"#.to_owned(),
            };
            std::fs::write(root.join(component).join("config.json"), config).unwrap();
        }
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_load_shape(LoadShape::DeferredMaterialization);
        (root, spec)
    }

    fn dense_legacy_spec(tmp: &tempfile::TempDir) -> (std::path::PathBuf, LoadSpec) {
        let (root, spec) = fixture(tmp, None);
        (root, spec.with_offload_policy(OffloadPolicy::Sequential))
    }

    fn packed_spec(
        tmp: &tempfile::TempDir,
        bits: i32,
        quant: Quant,
    ) -> (std::path::PathBuf, LoadSpec) {
        let (root, mut spec) = fixture(tmp, Some(bits));
        spec.quantize = Some(quant);
        (root, spec)
    }

    fn assert_unmeasured(contract: &MemoryProviderContract) {
        assert!(contract.calibration.is_none());
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Missing,
                "{strategy:?} must not inherit unmeasured evidence"
            );
        }
    }

    #[test]
    fn exact_q4_lens_route_publishes_the_measured_full_ladder() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = packed_spec(&tmp, 4, Quant::Q4);
        let contract = memory_strategy_contract("lens", &spec).unwrap();
        assert!(contract.conformance_errors().is_empty());
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            MEMORY_CALIBRATION_FINGERPRINT
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::StagedResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
        let decode = contract.capability(MemoryStrategy::BoundedDecode).unwrap();
        assert_eq!(decode.support, MemoryStrategySupport::Implemented);
        assert_eq!(decode.parameters.decode_tile_edges, vec![DECODE_TILE_EDGE]);
        assert_eq!(decode.parameters.decode_overlaps, vec![DECODE_OVERLAP]);
        let attention = contract
            .capability(MemoryStrategy::BoundedAttention)
            .unwrap();
        assert_eq!(attention.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            attention.parameters.attention_chunk_sizes,
            vec![ATTENTION_CHUNK_SIZE]
        );
        let window = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(window.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            window.parameters.transformer_window_components,
            vec![TransformerComponent::Both]
        );
        let measured = rung_four_context().selection;
        contract
            .validate_selection(&measured)
            .expect("the measured Both scope must remain selectable");
        for unmeasured in [TransformerComponent::TextEncoder, TransformerComponent::Dit] {
            let mut selection = measured;
            selection.parameters.transformer_window_component = Some(unmeasured);
            let error = contract
                .validate_selection(&selection)
                .expect_err("an unmeasured component scope must remain unpublished");
            assert!(
                error.to_string().contains("transformer_window_component")
                    && error.to_string().contains("[Both]"),
                "the refusal must identify the unadvertised component: {error}"
            );
        }
        assert!(contract.lifecycle.decode_tiling);
        assert!(contract.lifecycle.attention_chunking);
        assert!(contract.lifecycle.transformer_window_materialization);
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn legacy_dense_lens_turbo_te_only_identity_remains_separate() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = dense_legacy_spec(&tmp);
        let contract = memory_strategy_contract("lens_turbo", &spec).unwrap();
        assert!(contract.conformance_errors().is_empty());
        assert_eq!(
            contract.calibration.as_ref().unwrap().fingerprint,
            LEGACY_TEXT_ENCODER_FINGERPRINT
        );
        let window = contract
            .capability(MemoryStrategy::BoundedTransformerResidency)
            .unwrap();
        assert_eq!(window.support, MemoryStrategySupport::Implemented);
        assert_eq!(
            window.parameters.transformer_window_components,
            vec![TransformerComponent::TextEncoder]
        );
        assert!(!contract.lifecycle.decode_tiling);
        assert!(!contract.lifecycle.attention_chunking);
        let MemoryFormulaKind::PhaseEnvelope { variables, .. } = &contract.formula else {
            panic!("legacy Lens-Turbo calibration must retain its phase envelope")
        };
        assert!(!variables.contains(&MemoryFormulaVariable::DecodeTileArea));
        assert!(!variables.contains(&MemoryFormulaVariable::AttentionChunkSize));
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
        ] {
            assert_eq!(
                contract.capability(strategy).unwrap().support,
                MemoryStrategySupport::Missing
            );
        }
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn provider_tier_and_load_shape_mutations_do_not_fan_out_evidence() {
        let tmp = tempfile::tempdir().unwrap();
        let (q4_root, q4) = packed_spec(&tmp, 4, Quant::Q4);
        assert_unmeasured(&memory_strategy_contract("lens_turbo", &q4).unwrap());

        let (q8_root, q8) = packed_spec(&tmp, 8, Quant::Q8);
        assert_unmeasured(&memory_strategy_contract("lens", &q8).unwrap());

        let (dense_root, dense) = dense_legacy_spec(&tmp);
        assert_unmeasured(&memory_strategy_contract("lens", &dense).unwrap());

        let mut eager = q4.clone();
        eager.load_shape = LoadShape::EagerMaterialization;
        assert_unmeasured(&memory_strategy_contract("lens", &eager).unwrap());

        let mut adapted = q4.clone();
        adapted.adapters.push(AdapterSpec::new(
            q4_root.join("adapter.safetensors"),
            1.0,
            AdapterKind::Lora,
        ));
        assert_unmeasured(&memory_strategy_contract("lens", &adapted).unwrap());

        let mut pid = q4.clone();
        pid.pid = Some(PidWeights {
            checkpoint: WeightsSource::File(q4_root.join("pid.safetensors")),
            gemma: WeightsSource::Dir(q4_root.join("gemma")),
        });
        assert_unmeasured(&memory_strategy_contract("lens", &pid).unwrap());

        for root in [q4_root, q8_root, dense_root] {
            std::fs::remove_dir_all(root).ok();
        }
    }

    fn rung_four_context() -> MemoryRunContext {
        MemoryRunContext {
            optimization_authority: mlx_gen::gen_core::MemoryOptimizationAuthority::Calibrated,
            selection: MemorySelection {
                strategy: MemoryStrategy::BoundedTransformerResidency,
                parameters: MemoryStrategyParameters {
                    decode_tile_edge: Some(DECODE_TILE_EDGE),
                    decode_overlap: Some(DECODE_OVERLAP),
                    attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                    transformer_window_size: Some(TEXT_ENCODER_WINDOW),
                    transformer_window_component: Some(TransformerComponent::Both),
                },
                tier: MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: Some(Quant::Q4),
                    component_precision_floors: &[],
                },
            },
            calibration_abi: MEMORY_CALIBRATION_ABI,
            calibration_fingerprint: MEMORY_CALIBRATION_FINGERPRINT.to_owned(),
            load_shape: LoadShape::DeferredMaterialization,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: true,
            geometry: MemoryGeometry {
                width: 256,
                height: 256,
                batch: 1,
                frames: 1,
                reference_count: 0,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: 1_000_000,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "test".to_owned(),
        }
    }

    #[test]
    fn selected_contract_scope_reaches_the_generation_request_and_pid_fails_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let (root, spec) = packed_spec(&tmp, 4, Quant::Q4);
        let contract = memory_strategy_contract("lens", &spec).unwrap();
        let context = rung_four_context();
        let mut scope = registered_begin_request("lens", &spec, &contract, &context)
            .unwrap()
            .unwrap();
        let mut request = GenerationRequest {
            width: 256,
            height: 256,
            count: 1,
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        let memory = request.memory.expect("rung 4 configures request memory");
        assert!(memory.stage_residency);
        assert!(memory.tile_vae_decode);
        assert!(memory.chunk_attention);
        assert!(memory.stream_transformer_blocks);
        assert_eq!(memory.transformer_window_size, Some(TEXT_ENCODER_WINDOW));
        assert_eq!(
            memory.transformer_window_component,
            Some(TransformerComponent::Both)
        );
        // Registry conformance uses the same tensor-free scope cleanup exercised here; the Device
        // cleanup path is covered by the real-Metal runner.
        drop(scope);

        let mut unmeasured_native = context.clone();
        unmeasured_native.selection.parameters.decode_tile_edge = Some(640);
        assert!(matches!(
            safety_check(
                &contract,
                Precision::Bf16,
                Some(Quant::Q4),
                &unmeasured_native
            ),
            MemorySafetyDecision::Reject { .. }
        ));

        let mut pid = context;
        pid.use_pid = true;
        assert!(matches!(
            safety_check(&contract, Precision::Bf16, Some(Quant::Q4), &pid),
            MemorySafetyDecision::Reject { reason } if reason.contains("native VAE decode only")
        ));
        std::fs::remove_dir_all(root).ok();
    }
}
