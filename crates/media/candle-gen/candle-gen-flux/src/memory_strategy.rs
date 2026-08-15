//! Candle/CUDA FLUX.1 adoption of the shared image-memory ladder (SC-15823).
//!
//! Both `flux1_schnell` and `flux1_dev` execute this one contract. Rung 1 is the existing
//! CLIP/T5 -> denoise/decode residency split, rung 2 bounds the VAE decode, rung 3 uses the shared
//! attention planner, and rung 4 windows the 19 double plus 38 single FLUX transformer blocks.

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, GenerationMemory, GenerationRequest, LoadShape, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryComponentKind,
    MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable, MemoryGeometry,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent,
    MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyCapability, MemoryStrategyPrerequisite, MemoryStrategySupport,
    MemoryWindowMaterialization, PerComponentBytes, Precision, Quant, TransformerComponent,
    WeightsSource,
};

pub const DECODE_TILE_EDGE: u32 = 512;
// The tuple bounds row-wise latent transfer into a whole-frame CPU decode. It is deliberately not a
// family of independently decoded spatial VAE tiles: FLUX GroupNorm makes that numerically unsafe.
pub const DECODE_TILE_EDGES: &[u32] = &[DECODE_TILE_EDGE];
pub const DECODE_OVERLAP: u32 = 128;
pub const ATTENTION_CHUNK_SIZE: u32 =
    gen_core::attention_budget::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1];
pub const DEFAULT_TRANSFORMER_WINDOW: usize = 1;
pub const TRANSFORMER_BLOCKS: u32 = 57;
pub const CALIBRATION_FINGERPRINT: &str =
    "flux1-cuda-staged-tiled-decode-bounded-attention-device-format-blocks-v1";

fn streamable(spec: &LoadSpec) -> bool {
    matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.weights, WeightsSource::Dir(_))
        && spec.adapters.is_empty()
        && spec.extra_controls.is_empty()
        && spec.pid.is_none()
        && spec.identity.is_none()
        // The strict-control and XLabs providers each execute one resident overlay over the shared
        // streamed base. Their combined route is not wired by SC-15823 and therefore fails closed.
        && !(spec.control.is_some() && spec.ip_adapter.is_some())
}

fn overlay_components(spec: &LoadSpec) -> Vec<MemoryResidentComponent> {
    let mut components = [
        (
            "flux_control",
            MemoryComponentKind::ControlBranch,
            spec.control.as_ref(),
        ),
        (
            "flux_xlabs_ip",
            MemoryComponentKind::IpAdapter,
            spec.ip_adapter.as_ref(),
        ),
    ]
    .into_iter()
    .filter_map(|(id, kind, source)| {
        let source = source?;
        let path = match source {
            WeightsSource::Dir(path) | WeightsSource::File(path) => path,
        };
        let resident_bytes = gen_core::weightsmeta::safetensors_path_bytes(path);
        (resident_bytes > 0).then(|| MemoryResidentComponent {
            id: id.to_owned(),
            kind,
            resident_bytes,
            bounded_by: None,
            residency: MemoryComponentResidency::WholeRender,
        })
    })
    .collect::<Vec<_>>();
    if let Some(source) = spec.components.get("flux_ip_image_encoder") {
        let path = match source {
            WeightsSource::Dir(path) | WeightsSource::File(path) => path,
        };
        let resident_bytes = gen_core::weightsmeta::safetensors_path_bytes(path);
        if resident_bytes > 0 {
            components.push(MemoryResidentComponent {
                id: "flux_ip_image_encoder".to_owned(),
                kind: MemoryComponentKind::IpAdapter,
                resident_bytes,
                bounded_by: None,
                residency: MemoryComponentResidency::WholeRender,
            });
        }
    }
    components
}

pub(crate) fn provider_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    reference_backbone_contract(
        provider_id,
        spec,
        overlay_components(spec),
        CALIBRATION_FINGERPRINT,
    )
}

/// Build the shared FLUX.1 reference-backbone ladder for a bespoke provider that owns a resident
/// conditioning stack. The caller supplies the exact resident components and a provider-specific
/// calibration identity; this deliberately prevents a bespoke identity route from inheriting base
/// FLUX evidence merely because both routes execute the same 57-block trunk.
pub fn reference_backbone_contract(
    provider_id: &str,
    spec: &LoadSpec,
    resident_components: Vec<MemoryResidentComponent>,
    calibration_fingerprint: &str,
) -> gen_core::Result<MemoryProviderContract> {
    let streamable = streamable(spec);
    let components = PerComponentBytes::from_spec_subdirs(
        spec,
        &["text_encoder", "text_encoder_2"],
        &["transformer"],
        &["vae"],
    )
    .unwrap_or_default();
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    let overlay_bytes = resident_components
        .iter()
        .map(|component| component.resident_bytes)
        .sum();
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
                    decode_tile_edges: DECODE_TILE_EDGES.to_vec(),
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
                        transformer_window_components: vec![TransformerComponent::Dit],
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
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies,
        // PiD has its own decoder and candidate domain. It must fail closed until that route adopts
        // an explicit ladder rather than silently consuming the native FLUX VAE plan.
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
        formula: MemoryFormulaKind::ComponentPhaseEnvelope {
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
            resident_components,
        },
        calibration: Some(MemoryCalibrationIdentity::new(
            calibration_fingerprint,
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
            overlay_bytes,
        },
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    })
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
    let config = root.join("transformer/config.json");
    let packed = std::fs::read_to_string(&config)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|value| candle_gen::quant::PackedConfig::from_config(&value));
    packed
        .map(|packed| match packed.bits {
            4 => Ok(Quant::Q4),
            8 => Ok(Quant::Q8),
            bits => Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: transformer declares unsupported packed quantization width {bits}"
            ))),
        })
        .transpose()
}

pub fn resolved_numeric_tier(
    spec: &LoadSpec,
    provider_id: &str,
) -> gen_core::Result<MemoryNumericTier> {
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant: snapshot_quant_tier(spec, provider_id)?,
        component_precision_floors: &[],
    })
}

#[cfg(test)]
pub(crate) fn evidence_identity_and_tier(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<(MemoryCalibrationIdentity, MemoryNumericTier)> {
    let contract = provider_contract(provider_id, spec)?;
    let calibration = contract.calibration.ok_or_else(|| {
        gen_core::Error::Msg(format!(
            "{provider_id}: executable memory contract has no calibration identity"
        ))
    })?;
    Ok((calibration, resolved_numeric_tier(spec, provider_id)?))
}

pub(crate) fn validate_context(
    provider_id: &str,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    loaded_quant: Option<Quant>,
) -> gen_core::Result<()> {
    if let MemorySafetyDecision::Reject { reason } = safety_check(contract, context, loaded_quant) {
        return Err(gen_core::Error::Unsupported(reason));
    }
    if context.has_phases {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: optimized memory strategies do not cover multi-phase denoise"
        )));
    }
    if context.use_pid && context.selection.strategy.is_optimized() {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: PiD cannot consume the native FLUX VAE memory selection"
        )));
    }
    Ok(())
}

pub(crate) fn safety_check(
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

pub(crate) struct FluxMemoryScope {
    provider_id: &'static str,
    device: Device,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
    transformer_window: Option<u32>,
    use_pid: bool,
    finished: bool,
}

impl FluxMemoryScope {
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

    fn validate_geometry(&self, geometry: MemoryGeometry) -> gen_core::Result<()> {
        if geometry.width == self.geometry.width
            && geometry.height == self.geometry.height
            && geometry.frames == self.geometry.frames
            && geometry.reference_count == self.geometry.reference_count
            && geometry.batch > 0
            && geometry.batch <= self.geometry.batch
        {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: hook geometry does not fit admitted request geometry",
                self.provider_id
            )))
        }
    }
}

impl MemoryRequestScope for FluxMemoryScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        self.ensure_active()?;
        if request.use_pid != self.use_pid
            || request.width != self.geometry.width
            || request.height != self.geometry.height
            || request.count != self.geometry.batch
            || request.image_reference_count() != self.geometry.reference_count
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request route or geometry changed after memory admission",
                self.provider_id
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
        geometry: MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.ensure_active()?;
        self.validate_geometry(geometry)?;
        if self.use_pid {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: PiD has no admitted FLUX VAE tile plan",
                self.provider_id
            )));
        }
        if DECODE_TILE_EDGES.contains(&tile_edge) && overlap == DECODE_OVERLAP {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: native decode tiling does not publish {tile_edge}/{overlap}",
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
        if window == 0 || block_count == 0 || !first_block.is_multiple_of(window) {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: invalid transformer window {block_count} at {first_block}",
                self.provider_id
            )));
        }
        if first_block >= TRANSFORMER_BLOCKS {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: transformer window starts past the {TRANSFORMER_BLOCKS}-block stack",
                self.provider_id
            )));
        }
        let expected = window.min(TRANSFORMER_BLOCKS - first_block);
        if block_count == expected {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: admitted window {window} requires {expected} blocks at {first_block}, got {block_count}",
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

impl Drop for FluxMemoryScope {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.device.synchronize();
            self.finished = true;
        }
    }
}

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
        resolved_numeric_tier(spec, &contract.provider_id)?,
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )?;
    Ok(vec![gen_core::MemoryBehaviorFixture::new(context)])
}

pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let quant = snapshot_quant_tier(spec, provider_id)?;
    validate_context(provider_id, contract, context, quant)?;
    Ok(Some(Box::new(FluxMemoryScope::new(
        provider_id,
        Device::Cpu,
        contract,
        context,
    ))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gen_core::{MemorySelection, MemoryStrategyParameters};

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

    fn spec(tmp: &tempfile::TempDir) -> LoadSpec {
        let root = tmp.path().join("flux1-candle-memory-spec");
        for component in ["text_encoder", "text_encoder_2", "transformer", "vae"] {
            let dir = root.join(component);
            std::fs::create_dir_all(&dir).unwrap();
            write_control(&dir.join("model.safetensors"));
        }
        LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_offload_policy(gen_core::OffloadPolicy::Sequential)
            .with_load_shape(LoadShape::DeferredMaterialization)
    }

    fn selection(strategy: MemoryStrategy) -> MemorySelection {
        let mut parameters = MemoryStrategyParameters::default();
        if strategy as u8 >= MemoryStrategy::BoundedDecode as u8 {
            parameters.decode_tile_edge = Some(DECODE_TILE_EDGE);
            parameters.decode_overlap = Some(DECODE_OVERLAP);
        }
        if strategy as u8 >= MemoryStrategy::BoundedAttention as u8 {
            parameters.attention_chunk_size = Some(ATTENTION_CHUNK_SIZE);
        }
        if strategy == MemoryStrategy::BoundedTransformerResidency {
            parameters.transformer_window_size = Some(DEFAULT_TRANSFORMER_WINDOW as u32);
            parameters.transformer_window_component = Some(TransformerComponent::Dit);
        }
        MemorySelection {
            strategy,
            parameters,
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
        }
    }

    #[test]
    fn schnell_and_dev_publish_the_same_full_cuda_ladder() {
        let tmp = tempfile::tempdir().unwrap();
        for id in [crate::FLUX1_SCHNELL_ID, crate::FLUX1_DEV_ID] {
            let contract = provider_contract(id, &spec(&tmp)).unwrap();
            assert!(contract.conformance_errors().is_empty());
            gen_core_testkit::check_memory_strategy_contract(&contract).unwrap();
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedDecode)
                    .unwrap()
                    .parameters
                    .decode_tile_edges,
                DECODE_TILE_EDGES
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

    #[test]
    fn every_optimized_selection_is_staged_and_exactly_parameterized() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = spec(&tmp);
        let contract = provider_contract(crate::FLUX1_DEV_ID, &spec).unwrap();
        let (identity, tier) = evidence_identity_and_tier(crate::FLUX1_DEV_ID, &spec).unwrap();
        assert_eq!(identity.fingerprint, CALIBRATION_FINGERPRINT);
        assert_eq!(tier.quant, None);
        for strategy in [
            MemoryStrategy::StagedResidency,
            MemoryStrategy::BoundedDecode,
            MemoryStrategy::BoundedAttention,
            MemoryStrategy::BoundedTransformerResidency,
        ] {
            let selection = selection(strategy);
            contract.validate_selection(&selection).unwrap();
            let memory = contract.generation_memory(&selection).unwrap();
            assert!(memory.stage_residency);
        }
    }

    #[test]
    fn exact_control_and_ip_load_specs_keep_rung_four_and_price_the_resident_overlay() {
        let tmp = tempfile::tempdir().unwrap();
        let overlay_root_tmp = tempfile::tempdir().unwrap();
        let overlay_root = overlay_root_tmp.path().to_path_buf();
        let control_path = overlay_root.join("control.safetensors");
        let ip_path = overlay_root.join("ip_adapter.safetensors");
        write_control(&control_path);
        write_control(&ip_path);

        for (source, kind) in [
            (
                WeightsSource::File(control_path.clone()),
                MemoryComponentKind::ControlBranch,
            ),
            (
                WeightsSource::File(ip_path.clone()),
                MemoryComponentKind::IpAdapter,
            ),
        ] {
            let mut exact = spec(&tmp);
            match kind {
                MemoryComponentKind::ControlBranch => exact.control = Some(source),
                MemoryComponentKind::IpAdapter => exact.ip_adapter = Some(source),
                _ => unreachable!(),
            }
            let contract = provider_contract(crate::FLUX1_DEV_ID, &exact).unwrap();
            assert_eq!(
                contract
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Implemented
            );
            assert!(contract.conformance_errors().is_empty());
            assert!(contract.asset_facts.overlay_bytes > 0);
            assert_eq!(contract.resident_components().len(), 1);
            assert_eq!(contract.resident_components()[0].kind, kind);
            assert_eq!(contract.resident_components()[0].bounded_by, None);
        }

        let mut combined = spec(&tmp);
        combined.control = Some(WeightsSource::File(control_path));
        combined.ip_adapter = Some(WeightsSource::File(ip_path));
        assert_eq!(
            provider_contract(crate::FLUX1_DEV_ID, &combined)
                .unwrap()
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing,
            "combined XLabs+control is not an implemented SC-15823 route"
        );
    }

    #[test]
    fn packed_q4_q8_evidence_identity_uses_snapshot_tier_not_request_hint() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let transformer = root.join("transformer");
        std::fs::create_dir_all(&transformer).unwrap();

        for (bits, expected) in [(4, Quant::Q4), (8, Quant::Q8)] {
            std::fs::write(
                transformer.join("config.json"),
                format!(r#"{{"num_layers":19,"quantization":{{"bits":{bits},"group_size":64}}}}"#),
            )
            .unwrap();
            let packed = LoadSpec::new(WeightsSource::Dir(root.clone()))
                .with_offload_policy(gen_core::OffloadPolicy::Sequential)
                .with_load_shape(LoadShape::DeferredMaterialization);
            assert_eq!(packed.quantize, None, "packed tier is not a request hint");

            let (identity, tier) =
                evidence_identity_and_tier(crate::FLUX1_SCHNELL_ID, &packed).unwrap();
            assert_eq!(identity.fingerprint, CALIBRATION_FINGERPRINT);
            assert_eq!(identity.load_shape, LoadShape::DeferredMaterialization);
            assert_eq!(tier.precision, Precision::Bf16);
            assert_eq!(tier.quant, Some(expected));
        }
    }

    #[test]
    fn stale_identity_and_pid_are_rejected_before_execution() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = spec(&tmp);
        let contract = provider_contract(crate::FLUX1_SCHNELL_ID, &spec).unwrap();
        let mut fixture = registered_valid_fixture(
            &spec,
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .pop()
        .unwrap();
        fixture.context.calibration_fingerprint = "stale-flux1-calibration".into();
        assert!(matches!(
            registered_safety_check(&spec, &contract, &fixture.context),
            MemorySafetyDecision::Reject { .. }
        ));
        fixture.context.calibration_fingerprint = CALIBRATION_FINGERPRINT.into();
        fixture.context.use_pid = true;
        assert!(matches!(
            registered_safety_check(&spec, &contract, &fixture.context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn scope_covers_both_57_block_namespaces_and_cleanup() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = spec(&tmp);
        let contract = provider_contract(crate::FLUX1_DEV_ID, &spec).unwrap();
        let mut fixture = registered_valid_fixture(
            &spec,
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .pop()
        .unwrap();
        fixture.context.selection.parameters.transformer_window_size = Some(1);
        let mut scope =
            registered_begin_request(crate::FLUX1_DEV_ID, &spec, &contract, &fixture.context)
                .unwrap()
                .unwrap();
        scope.configure_request(&mut fixture.request).unwrap();
        scope.materialize_transformer_window(0, 1).unwrap();
        scope.materialize_transformer_window(56, 1).unwrap();
        scope.finish(MemoryRunOutcome::Complete).unwrap();
        assert!(scope.finish(MemoryRunOutcome::Complete).is_err());
    }
}
