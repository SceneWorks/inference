//! Candle/CUDA Z-Image adoption of the shared five-rung image memory contract (sc-15815).
//!
//! `z_image_edit` is a SceneWorks catalog alias for the `z_image_turbo` provider and therefore
//! inherits the Turbo contract. The two bespoke control providers are registered separately so the
//! matrix records their current gaps rather than inferring capability from the plain provider.

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, GenerationMemory, GenerationRequest, LoadSpec, MemoryGeometry, MemoryNumericTier,
    MemoryPhase, MemoryProviderContract, MemoryRequestScope, MemoryRunContext, MemoryRunOutcome,
    MemorySafetyDecision, MemoryStrategy, Precision, Quant, WeightsSource,
};
#[cfg(any(feature = "cuda", test))]
use candle_gen::gen_core::{
    LoadShape, MemoryAssetFacts, MemoryBackendRealization, MemoryCalibrationIdentity,
    MemoryFormulaKind, MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryParameterRanges,
    MemoryPrerequisiteScope, MemoryStrategyCapability, MemoryStrategyPrerequisite,
    MemoryStrategySupport, MemoryWindowMaterialization, PerComponentBytes,
};

pub(crate) const DECODE_TILE_EDGE: u32 = 512;
pub(crate) const DECODE_OVERLAP: u32 = 128;
pub(crate) const ATTENTION_CHUNK_SIZE: u32 =
    gen_core::attention_budget::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
#[cfg(any(feature = "cuda", test))]
pub(crate) const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 4, 8, 15, 30];
pub(crate) const DEFAULT_TRANSFORMER_WINDOW: usize = 1;
#[cfg(any(feature = "cuda", test))]
pub(crate) const CALIBRATION_FINGERPRINT: &str =
    "z-image-cuda-staged-tiled-decode-bounded-attention-device-format-blocks-v2";

#[cfg(any(feature = "cuda", test))]
pub(crate) fn provider_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let streamable = matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.weights, gen_core::WeightsSource::Dir(_))
        && spec.adapters.is_empty();
    let components =
        PerComponentBytes::from_spec_subdirs(spec, &["text_encoder"], &["transformer"], &["vae"])
            .unwrap_or_default();
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let strategies = MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| MemoryStrategyCapability {
            strategy,
            support: if strategy == MemoryStrategy::BoundedTransformerResidency && !streamable {
                MemoryStrategySupport::Missing
            } else {
                MemoryStrategySupport::Implemented
            },
            parameters: match strategy {
                MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                    decode_tile_edges: vec![DECODE_TILE_EDGE],
                    decode_overlaps: vec![DECODE_OVERLAP],
                    ..Default::default()
                },
                MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                    attention_chunk_sizes: vec![ATTENTION_CHUNK_SIZE],
                    ..Default::default()
                },
                MemoryStrategy::BoundedTransformerResidency if streamable => {
                    MemoryParameterRanges {
                        transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                        ..Default::default()
                    }
                }
                _ => MemoryParameterRanges::default(),
            },
        })
        .collect();

    Ok(MemoryProviderContract {
        provider_id: provider_id.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: true,
            // Packed q4/q8 `layers.N` projections are prepared once as content-addressed GGML
            // sidecars; each window maps and transfers only those already-device-format bytes.
            // Dense snapshots already transfer their stored tensor format directly.
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies,
        // PiD replaces the native VAE with a separately planned decoder. Until that decoder accepts
        // this provider's explicit tile plan, the request safety gate rejects optimized PiD runs.
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: [
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ]
        .into_iter()
        .map(|strategy| {
            (
                strategy,
                MemoryStrategyPrerequisite::Rung {
                    rung: MemoryStrategy::StagedResidency,
                    scope: MemoryPrerequisiteScope::EngagedInSameRequest,
                },
            )
        })
        .collect(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: gen_core::ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: true,
            attention_chunking: true,
            transformer_window_materialization: streamable,
        },
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases,
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::ConditioningTokenCount,
                MemoryFormulaVariable::OverlayBytes,
                MemoryFormulaVariable::DecodeTileArea,
                MemoryFormulaVariable::AttentionChunkSize,
                MemoryFormulaVariable::TransformerWindowSize,
            ],
        },
        calibration: Some(MemoryCalibrationIdentity::new(
            CALIBRATION_FINGERPRINT,
            spec.load_shape,
        )),
        asset_facts: MemoryAssetFacts {
            base_bytes: components
                .text_encoder
                .saturating_add(components.dit)
                .saturating_add(components.vae),
            conditioning_bytes: components.text_encoder,
            transformer_bytes: components.dit,
            decoder_bytes: components.vae,
            overlay_bytes: 0,
        },
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
}

/// Explicit contract for bespoke control routes. Their eager dual-network implementation has not
/// adopted the request lifecycle, so every constrained rung is `Missing`; recording the provider ids
/// prevents the catalog from inheriting the plain provider's declaration.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn control_contract(
    provider_id: &str,
    _spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let mut contract = MemoryProviderContract::compatibility_default(
        provider_id,
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        format!("{provider_id}-cuda-control-v1"),
        LoadShape::EagerMaterialization,
    ));
    Ok(contract)
}

pub(crate) struct ZImageMemoryScope {
    provider_id: &'static str,
    device: Device,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
    transformer_window: Option<u32>,
    use_pid: bool,
    finished: bool,
}

impl ZImageMemoryScope {
    pub(crate) fn new(
        provider_id: &'static str,
        device: Device,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> Self {
        Self {
            provider_id,
            device,
            geometry: context.geometry,
            memory: contract.generation_memory(&context.selection),
            transformer_window: contract
                .engages(
                    context.selection.strategy,
                    MemoryStrategy::BoundedTransformerResidency,
                )
                .then_some(context.selection.parameters.transformer_window_size)
                .flatten(),
            use_pid: context.use_pid,
            finished: false,
        }
    }

    fn ensure_active(&self) -> gen_core::Result<()> {
        if self.finished {
            Err(gen_core::Error::Msg(format!(
                "{}: memory-strategy request scope is already finished",
                self.provider_id
            )))
        } else {
            Ok(())
        }
    }
}

impl MemoryRequestScope for ZImageMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        self.ensure_active()?;
        if request.use_pid != self.use_pid {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request PiD route changed after memory admission",
                self.provider_id
            )));
        }
        if request.width != self.geometry.width
            || request.height != self.geometry.height
            || request.count != self.geometry.batch
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request geometry {}x{} count {} does not match admitted {}x{} count {}",
                self.provider_id,
                request.width,
                request.height,
                request.count,
                self.geometry.width,
                self.geometry.height,
                self.geometry.batch
            )));
        }
        request.memory = self.memory;
        Ok(())
    }

    fn enter_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
        self.ensure_active()
    }

    fn leave_phase(&mut self, _phase: MemoryPhase) -> gen_core::Result<()> {
        self.ensure_active()
    }

    fn configure_decode(
        &mut self,
        tile_edge: u32,
        overlap: u32,
        _geometry: MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.ensure_active()?;
        if self.use_pid {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: PiD uses an alternate decoder whose explicit tile plan is not yet wired",
                self.provider_id
            )));
        }
        if tile_edge == DECODE_TILE_EDGE && overlap == DECODE_OVERLAP {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: native decode tiling is fixed at {DECODE_TILE_EDGE}/{DECODE_OVERLAP}, got \
                 {tile_edge}/{overlap}",
                self.provider_id
            )))
        }
    }

    fn configure_attention(&mut self, chunk_size: u32) -> gen_core::Result<()> {
        self.ensure_active()?;
        if chunk_size == ATTENTION_CHUNK_SIZE {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: attention chunk size is fixed at {ATTENTION_CHUNK_SIZE}, got {chunk_size}",
                self.provider_id
            )))
        }
    }

    fn materialize_transformer_window(
        &mut self,
        first_block: u32,
        block_count: u32,
    ) -> gen_core::Result<()> {
        self.ensure_active()?;
        let Some(window) = self.transformer_window else {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: bounded transformer residency was not selected",
                self.provider_id
            )));
        };
        const BLOCKS: u32 = 30;
        if first_block >= BLOCKS {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: transformer window starts past the {BLOCKS}-block stack",
                self.provider_id
            )));
        }
        let expected = window.min(BLOCKS - first_block);
        if block_count == expected {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: admitted window {window} requires {expected} blocks at {first_block}, got \
                 {block_count}",
                self.provider_id
            )))
        }
    }

    fn finish(&mut self, _outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        self.ensure_active()?;
        self.device
            .synchronize()
            .map_err(gen_core::Error::backend)?;
        self.finished = true;
        Ok(())
    }
}

impl Drop for ZImageMemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.device.synchronize();
            self.finished = true;
        }
    }
}

pub(crate) fn validate_context(
    provider_id: &str,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_quant: Option<Quant>,
) -> gen_core::Result<()> {
    if let MemorySafetyDecision::Reject { reason } =
        safety_check(provider_id, contract, context, loaded_quant)
    {
        return Err(gen_core::Error::Unsupported(reason));
    }
    if context.has_phases {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: optimized memory strategies do not cover multi-phase denoise"
        )));
    }
    if context.use_pid && context.selection.strategy.is_optimized() {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: PiD uses an alternate decode planner and cannot consume this native-VAE \
             memory selection"
        )));
    }
    Ok(())
}

/// The complete pre-request admission predicate shared by loaded generators and the weights-free
/// registry. Keep this as the decision-form adapter over [`validate_context`] so neither route can
/// admit a context that [`registered_begin_request`] (or a loaded generator's begin hook) rejects.
pub(crate) fn admission_safety_check(
    provider_id: &str,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_quant: Option<Quant>,
) -> MemorySafetyDecision {
    match validate_context(provider_id, contract, context, loaded_quant) {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub(crate) fn safety_check(
    _provider_id: &str,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_quant: Option<Quant>,
) -> MemorySafetyDecision {
    gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: Precision::Bf16,
            quant: loaded_quant,
            component_precision_floors: &[],
        }),
        None,
    )
}

pub(crate) fn snapshot_quant_tier(
    spec: &LoadSpec,
    provider_id: &str,
) -> gen_core::Result<Option<Quant>> {
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{provider_id}: actual numeric tier requires a snapshot directory"
            )))
        }
    };
    crate::pipeline::packed_config_at(root, "transformer")
        .map_err(gen_core::Error::backend)?
        .map(|packed| match packed.bits {
            4 => Ok(Quant::Q4),
            8 => Ok(Quant::Q8),
            bits => Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: transformer declares unsupported packed quantization width {bits}"
            ))),
        })
        .transpose()
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match snapshot_quant_tier(spec, &contract.provider_id) {
        Ok(quant) => admission_safety_check(&contract.provider_id, contract, context, quant),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: snapshot_quant_tier(spec, &contract.provider_id)?,
            component_precision_floors: &[],
        },
        gen_core::MemoryBehaviorRoute {
            mode: gen_core::MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )?;
    Ok(vec![gen_core::MemoryBehaviorFixture::new(context)])
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let quant = snapshot_quant_tier(spec, provider_id)?;
    validate_context(provider_id, contract, context, quant)?;
    Ok(Some(Box::new(ZImageMemoryScope::new(
        provider_id,
        Device::Cpu,
        contract,
        context,
    ))))
}

#[cfg(test)]
fn generation_memory(
    contract: &MemoryProviderContract,
    selection: gen_core::MemorySelection,
) -> Option<GenerationMemory> {
    contract.generation_memory(&selection)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{
        MemoryBudget, MemoryCacheState, MemoryMode, MemoryNumericTier, MemorySelection,
        MemoryStrategyParameters, Precision, WeightsSource,
    };

    fn spec() -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        spec.load_shape = LoadShape::DeferredMaterialization;
        spec
    }

    #[test]
    fn plain_contract_shape_controls_streaming_and_rung_four() {
        let deferred = provider_contract(crate::MODEL_ID, &spec()).unwrap();
        assert_eq!(deferred.load_shape, LoadShape::DeferredMaterialization);
        assert_eq!(
            deferred.calibration.as_ref().unwrap().load_shape,
            LoadShape::DeferredMaterialization
        );
        assert!(deferred.lifecycle.transformer_window_materialization);
        assert!(matches!(
            deferred
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        ));

        let mut eager_spec = spec();
        eager_spec.load_shape = LoadShape::EagerMaterialization;
        let eager = provider_contract(crate::MODEL_ID, &eager_spec).unwrap();
        assert_eq!(eager.load_shape, LoadShape::EagerMaterialization);
        assert_eq!(
            eager.calibration.as_ref().unwrap().load_shape,
            LoadShape::EagerMaterialization
        );
        assert!(!eager.lifecycle.transformer_window_materialization);
        assert!(matches!(
            eager
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        ));
    }

    #[test]
    fn weights_free_behavior_uses_the_cpu_scope_path() {
        let spec = spec();
        let contract = provider_contract(crate::MODEL_ID, &spec).unwrap();
        let mut fixture =
            registered_valid_fixture(&spec, &contract, MemoryStrategy::StagedResidency)
                .unwrap()
                .into_iter()
                .next()
                .unwrap();
        let mut scope =
            registered_begin_request(crate::MODEL_ID, &spec, &contract, &fixture.context)
                .unwrap()
                .unwrap();
        scope.configure_request(&mut fixture.request).unwrap();
        assert_eq!(
            fixture.request.memory,
            contract.generation_memory(&fixture.context.selection)
        );
        scope.finish(gen_core::MemoryRunOutcome::Complete).unwrap();
    }

    #[test]
    fn plain_contract_is_conformant_and_exposes_every_candidate_range() {
        for id in [crate::MODEL_ID, crate::base::MODEL_ID] {
            let contract = provider_contract(id, &spec()).unwrap();
            assert!(contract.conformance_errors().is_empty());
            gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedDecode)
                    .unwrap()
                    .parameters
                    .decode_tile_edges,
                vec![DECODE_TILE_EDGE]
            );
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedAttention)
                    .unwrap()
                    .parameters
                    .attention_chunk_sizes,
                vec![ATTENTION_CHUNK_SIZE]
            );
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .parameters
                    .transformer_window_sizes,
                TRANSFORMER_WINDOW_SIZES
            );
        }
    }

    fn rung_four_selection() -> MemorySelection {
        MemorySelection {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(DECODE_TILE_EDGE),
                decode_overlap: Some(DECODE_OVERLAP),
                attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                transformer_window_size: Some(4),
                ..Default::default()
            },
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
        }
    }

    fn context(contract: &MemoryProviderContract) -> MemoryRunContext {
        let calibration = contract.calibration.as_ref().unwrap();
        MemoryRunContext {
            selection: rung_four_selection(),
            calibration_abi: calibration.abi,
            calibration_fingerprint: calibration.fingerprint.clone(),
            load_shape: calibration.load_shape,
            mode: MemoryMode::TextToImage,
            has_reference: false,
            use_pid: false,
            has_phases: false,
            geometry: MemoryGeometry {
                width: 768,
                height: 768,
                batch: 1,
                frames: 1,
            },
            overlay: None,
            budget: MemoryBudget {
                total_bytes: u64::MAX,
                committed_bytes: 0,
                reclaimable_bytes: 0,
                reserved_headroom_bytes: 0,
            },
            predicted_peak_bytes: 1,
            cache_state: MemoryCacheState::Cold,
            evidence_revision: "sc-15815-test".to_owned(),
        }
    }

    #[test]
    fn rung_four_selection_maps_every_engaged_control_and_preserves_the_window() {
        let contract = provider_contract(crate::MODEL_ID, &spec()).unwrap();
        let selection = rung_four_selection();
        contract.validate_selection(&selection).unwrap();
        let memory = generation_memory(&contract, selection).unwrap();
        assert!(memory.stage_residency);
        assert!(memory.tile_vae_decode);
        assert!(memory.chunk_attention);
        assert!(memory.stream_transformer_blocks);
        assert_eq!(memory.transformer_window_size, Some(4));
    }

    #[test]
    fn request_scope_applies_exact_parameters_and_finishes_once() {
        let contract = provider_contract(crate::MODEL_ID, &spec()).unwrap();
        let context = context(&contract);
        validate_context(crate::MODEL_ID, &contract, &context, None).unwrap();
        let mut scope = ZImageMemoryScope::new(crate::MODEL_ID, Device::Cpu, &contract, &context);
        scope
            .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, context.geometry)
            .unwrap();
        scope.configure_attention(ATTENTION_CHUNK_SIZE).unwrap();
        scope.materialize_transformer_window(0, 4).unwrap();
        scope.materialize_transformer_window(28, 2).unwrap();
        let mut request = GenerationRequest {
            width: 768,
            height: 768,
            count: 1,
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        assert_eq!(
            request.memory,
            generation_memory(&contract, context.selection)
        );
        scope.finish(MemoryRunOutcome::Complete).unwrap();
        assert!(scope.enter_phase(MemoryPhase::Denoise).is_err());
        assert!(scope.finish(MemoryRunOutcome::Complete).is_err());
    }

    #[test]
    fn stale_fingerprint_and_pid_optimized_routes_are_rejected_before_execution() {
        let contract = provider_contract(crate::MODEL_ID, &spec()).unwrap();
        let mut stale = context(&contract);
        stale.calibration_fingerprint.push_str("-stale");
        assert!(validate_context(crate::MODEL_ID, &contract, &stale, None).is_err());

        let mut pid = context(&contract);
        pid.use_pid = true;
        let error = validate_context(crate::MODEL_ID, &contract, &pid, None).unwrap_err();
        assert!(error.to_string().contains("PiD"));
    }

    #[test]
    fn registered_admission_matches_loaded_begin_request_policy() {
        let spec = spec();
        let contract = provider_contract(crate::MODEL_ID, &spec).unwrap();

        let mut phases = context(&contract);
        phases.has_phases = true;
        let error = validate_context(crate::MODEL_ID, &contract, &phases, None).unwrap_err();
        assert!(error.to_string().contains("multi-phase"));
        assert_eq!(
            registered_safety_check(&spec, &contract, &phases),
            admission_safety_check(crate::MODEL_ID, &contract, &phases, None)
        );
        assert!(matches!(
            registered_safety_check(&spec, &contract, &phases),
            MemorySafetyDecision::Reject { reason } if reason == error.to_string()
        ));

        let mut pid = context(&contract);
        pid.use_pid = true;
        let error = validate_context(crate::MODEL_ID, &contract, &pid, None).unwrap_err();
        assert!(error.to_string().contains("PiD"));
        assert_eq!(
            registered_safety_check(&spec, &contract, &pid),
            admission_safety_check(crate::MODEL_ID, &contract, &pid, None)
        );
        assert!(matches!(
            registered_safety_check(&spec, &contract, &pid),
            MemorySafetyDecision::Reject { reason } if reason == error.to_string()
        ));

        let mut stale = context(&contract);
        stale.calibration_fingerprint.push_str("-stale");
        let error = validate_context(crate::MODEL_ID, &contract, &stale, None).unwrap_err();
        assert!(error.to_string().contains("calibration handshake mismatch"));
        assert_eq!(
            registered_safety_check(&spec, &contract, &stale),
            admission_safety_check(crate::MODEL_ID, &contract, &stale, None)
        );
        assert!(matches!(
            registered_safety_check(&spec, &contract, &stale),
            MemorySafetyDecision::Reject { reason } if reason == error.to_string()
        ));
    }

    #[test]
    fn packed_q4_and_q8_snapshots_bind_weights_free_admission_to_the_detected_tier() {
        for (bits, actual, wrong) in [
            (4, Quant::Q4, Some(Quant::Q8)),
            (8, Quant::Q8, Some(Quant::Q4)),
        ] {
            let root = std::env::temp_dir()
                .join(format!("candle-z-image-tier-{bits}-{}", std::process::id()));
            std::fs::create_dir_all(root.join("transformer")).unwrap();
            std::fs::write(
                root.join("transformer/config.json"),
                format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            )
            .unwrap();
            let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
            for (provider_id, control) in [
                (crate::MODEL_ID, false),
                (crate::base::MODEL_ID, false),
                ("z_image_turbo_control", true),
                ("z_image_control", true),
            ] {
                let contract = if control {
                    control_contract(provider_id, &spec).unwrap()
                } else {
                    provider_contract(provider_id, &spec).unwrap()
                };
                let mut actual_context = context(&contract);
                actual_context.selection.strategy = MemoryStrategy::Resident;
                actual_context.selection.parameters = Default::default();
                actual_context.selection.tier.quant = Some(actual);
                assert_eq!(
                    registered_safety_check(&spec, &contract, &actual_context),
                    MemorySafetyDecision::Accept
                );
                for selected in [None, wrong] {
                    let mut wrong_context = context(&contract);
                    wrong_context.selection.tier.quant = selected;
                    assert!(matches!(
                        registered_safety_check(&spec, &contract, &wrong_context),
                        MemorySafetyDecision::Reject { reason }
                            if reason.contains("does not match loaded tier")
                    ));
                    assert!(matches!(
                        safety_check(provider_id, &contract, &wrong_context, Some(actual)),
                        MemorySafetyDecision::Reject { reason }
                            if reason.contains("does not match loaded tier")
                    ));
                }
            }
            std::fs::remove_dir_all(root).ok();
        }
    }

    #[test]
    fn adapters_keep_lower_rungs_but_cannot_claim_streamed_blocks() {
        let mut spec = spec();
        spec.adapters.push(gen_core::AdapterSpec::new(
            "/adapter.safetensors".into(),
            1.0,
            gen_core::AdapterKind::Lora,
        ));
        let contract = provider_contract(crate::MODEL_ID, &spec).unwrap();
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        );
        assert_eq!(
            contract
                .capability(MemoryStrategy::BoundedAttention)
                .unwrap()
                .support,
            MemoryStrategySupport::Implemented
        );
    }

    #[test]
    fn block_materialization_reports_device_format_transfer_for_dense_and_packed_snapshots() {
        let dense = provider_contract(crate::MODEL_ID, &spec()).unwrap();
        assert!(matches!(
            dense.backend,
            MemoryBackendRealization::CandleCuda {
                block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
                ..
            }
        ));

        let root = std::env::temp_dir().join(format!(
            "z-image-sc15815-packed-contract-{}",
            std::process::id()
        ));
        let transformer = root.join("transformer");
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(
            transformer.join("config.json"),
            br#"{"quantization":{"group_size":64,"bits":4}}"#,
        )
        .unwrap();
        let packed_spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let packed = provider_contract(crate::MODEL_ID, &packed_spec).unwrap();
        std::fs::remove_dir_all(root).unwrap();
        assert!(matches!(
            packed.backend,
            MemoryBackendRealization::CandleCuda {
                block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
                ..
            }
        ));
    }

    #[test]
    fn control_routes_are_explicit_and_do_not_inherit_plain_capabilities() {
        for id in ["z_image_turbo_control", "z_image_control"] {
            let contract = control_contract(id, &spec()).unwrap();
            assert!(contract.conformance_errors().is_empty());
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing
            );
        }
    }
}
