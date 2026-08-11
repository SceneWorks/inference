//! Candle/CUDA Qwen-Image adoption of the shared image memory ladder (sc-15817).
//!
//! The base and edit routes share one provider contract. Rung 1 uses the existing request-scoped
//! conditioning/render residency split; rung 2 drives the head-once/tail-tiled Qwen VAE; rung 3
//! supplies the shared attention planner; and rung 4 materializes the uniform 60-block DiT trunk
//! through the shared Candle block-window driver. Catalog calibration remains entry-specific.

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, GenerationMemory, GenerationRequest, LoadShape, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryFormulaKind, MemoryFormulaVariable,
    MemoryGeometry, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier,
    MemoryParameterRanges, MemoryPhase, MemoryPrerequisiteScope, MemoryProviderContract,
    MemoryRequestScope, MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyCapability, MemoryStrategyPrerequisite, MemoryStrategySupport,
    MemoryWindowMaterialization, PerComponentBytes, Precision, Quant, TransformerComponent,
    WeightsSource,
};

pub(crate) const DECODE_TILE_EDGE: u32 = 512;
pub(crate) const DECODE_TILE_EDGES: &[u32] = &[768, 640, 512, 448, 384, 320, 256];
pub(crate) const DECODE_OVERLAP: u32 = 64;
#[cfg(test)]
pub(crate) const REJECTED_SUB_512_OVERLAP: u32 = 96;
pub(crate) const ATTENTION_CHUNK_SIZE: u32 =
    gen_core::attention_budget::CONSTRAINED_ATTN_SCORES_BUDGET as u32;
pub(crate) const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 4, 8, 15, 30];
#[cfg(test)]
pub(crate) const DEFAULT_TRANSFORMER_WINDOW: usize = 1;
pub(crate) const TRANSFORMER_BLOCKS: u32 = 60;
pub const CALIBRATION_FINGERPRINT: &str =
    "qwen-image-cuda-staged-tiled-decode-bounded-attention-device-format-blocks-v1";

fn streamable(spec: &LoadSpec) -> bool {
    // File and Dir share the provider identity, but the evidence matrix has no source axis. Keep the
    // imported path's rung 4 Missing until its pinned/re-openable implementation is measured directly;
    // a snapshot measurement must not be claimed for a File source.
    matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.weights, WeightsSource::Dir(_))
        && spec.adapters.is_empty()
        && spec.pid.is_none()
}

fn cast_component_bytes(
    path: &std::path::Path,
    float_width: u64,
    component: &str,
    validate_name: impl Fn(&str) -> gen_core::Result<()>,
) -> gen_core::Result<u64> {
    use gen_core::weightsmeta::Dtype;

    let tensors = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
    if tensors.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "qwen-image imported {component} '{}' contains no tensors",
            path.display()
        )));
    }
    tensors.into_iter().try_fold(0_u64, |total, tensor| {
        validate_name(&tensor.name)?;
        let resident = match tensor.dtype {
            Dtype::U8 | Dtype::U32 | Dtype::I16 | Dtype::I32 | Dtype::I64 => tensor.data_bytes,
            Dtype::U16 => tensor.materialized_bytes(4)?,
            Dtype::F8_E4M3
            | Dtype::F16
            | Dtype::BF16
            | Dtype::F32
            | Dtype::F64 => tensor.materialized_bytes(float_width)?,
            dtype => {
                return Err(gen_core::Error::Unsupported(format!(
                    "qwen-image imported {component} tensor {:?} uses unsupported Candle dtype {dtype:?}",
                    tensor.name
                )))
            }
        };
        total.checked_add(resident).ok_or_else(|| {
            gen_core::Error::Msg(format!(
                "qwen-image imported {component} resident byte sum overflow"
            ))
        })
    })
}

fn imported_dit_bytes(path: &std::path::Path) -> gen_core::Result<u64> {
    const PREFIX: &str = "model.diffusion_model.";
    cast_component_bytes(path, 2, "DiT", |name| {
        let Some(mapped) = name.strip_prefix(PREFIX) else {
            return Err(gen_core::Error::Msg(format!(
                "qwen-image ComfyUI DiT tensor {name:?} is outside the required {PREFIX:?} namespace"
            )));
        };
        if mapped.is_empty() {
            return Err(gen_core::Error::Msg(
                "qwen-image ComfyUI DiT tensor maps to an empty key".into(),
            ));
        }
        Ok(())
    })
}

fn f32_component_bytes(path: &std::path::Path, component: &str) -> gen_core::Result<u64> {
    cast_component_bytes(path, 4, component, |_| Ok(()))
}

pub(crate) fn provider_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    if provider_id == crate::MODEL_ID {
        crate::validate_load_spec(spec)?;
    }
    let streamable = streamable(spec);
    let components = match &spec.weights {
        WeightsSource::Dir(_) => PerComponentBytes::from_spec_subdirs(
            spec,
            &["text_encoder"],
            &["transformer"],
            &["vae"],
        )
        .unwrap_or_default(),
        WeightsSource::File(path) => {
            let base = gen_core::require_base_snapshot(spec, provider_id)?;
            let vae = match spec.components.get(gen_core::COMFYUI_VAE_COMPONENT) {
                Some(WeightsSource::Dir(path)) => f32_component_bytes(path, "VAE")?,
                Some(WeightsSource::File(path)) => {
                    spec.read_file_unchanged_if_prepared(path, |p| f32_component_bytes(p, "VAE"))?
                }
                None => f32_component_bytes(&base.join("vae"), "base VAE")?,
            };
            PerComponentBytes {
                text_encoder: f32_component_bytes(&base.join("text_encoder"), "base text encoder")?,
                dit: spec.read_file_unchanged_if_prepared(path, imported_dit_bytes)?,
                vae,
            }
        }
    };
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
        // PiD owns a distinct decoder and tile domain. Until it accepts this explicit native-VAE
        // plan, optimized selections are rejected instead of silently applying the wrong geometry.
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

pub(crate) fn snapshot_quant_tier(
    spec: &LoadSpec,
    provider_id: &str,
) -> gen_core::Result<Option<Quant>> {
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(_) => return Ok(None),
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

pub(crate) fn resolved_numeric_tier(
    spec: &LoadSpec,
    provider_id: &str,
) -> gen_core::Result<MemoryNumericTier> {
    Ok(MemoryNumericTier {
        precision: Precision::Bf16,
        quant: snapshot_quant_tier(spec, provider_id)?,
        component_precision_floors: &[],
    })
}

/// Resolve the exact executable contract identity and loaded numeric tier used by V1 evidence.
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
            "{provider_id}: PiD uses an alternate decode planner and cannot consume this native-VAE memory selection"
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

pub(crate) struct QwenMemoryScope {
    provider_id: &'static str,
    device: Device,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
    transformer_window: Option<u32>,
    use_pid: bool,
    finished: bool,
}

impl QwenMemoryScope {
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
            return Ok(());
        }
        Err(gen_core::Error::Unsupported(format!(
            "{}: hook geometry {}x{}x{} frames={} references={} does not fit admitted {}x{}x{} frames={} references={}",
            self.provider_id,
            geometry.width,
            geometry.height,
            geometry.batch,
            geometry.frames,
            geometry.reference_count,
            self.geometry.width,
            self.geometry.height,
            self.geometry.batch,
            self.geometry.frames,
            self.geometry.reference_count
        )))
    }
}

impl MemoryRequestScope for QwenMemoryScope {
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
            || request.image_reference_count() != self.geometry.reference_count
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request geometry {}x{} count {} references={} does not match admitted {}x{} count {} references={}",
                self.provider_id,
                request.width,
                request.height,
                request.count,
                request.image_reference_count(),
                self.geometry.width,
                self.geometry.height,
                self.geometry.batch,
                self.geometry.reference_count
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
                "{}: PiD uses an alternate decoder whose explicit tile plan is not wired",
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
                "{}: transformer window {window} requires a non-zero block count and aligned start, got {block_count} blocks at {first_block}",
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

impl Drop for QwenMemoryScope {
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
    let edit = contract.provider_id == "qwen_image_edit";
    let context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        resolved_numeric_tier(spec, &contract.provider_id)?,
        gen_core::MemoryBehaviorRoute {
            mode: if edit {
                MemoryMode::Edit
            } else {
                MemoryMode::TextToImage
            },
            reference_count: u32::from(edit),
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
    Ok(Some(Box::new(QwenMemoryScope::new(
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

    fn write_typed_safetensors(path: &std::path::Path, tensors: &[(&str, &str, &[usize], usize)]) {
        let mut offset = 0_usize;
        let mut header = serde_json::Map::new();
        for (name, dtype, shape, bytes) in tensors {
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.resize(bytes.len() + offset, 0);
        std::fs::write(path, bytes).unwrap();
    }

    fn file_spec(tmp: &tempfile::TempDir) -> LoadSpec {
        let root = tmp.path().join("base");
        for component in ["text_encoder", "vae"] {
            std::fs::create_dir_all(root.join(component)).unwrap();
        }
        write_typed_safetensors(
            &root.join("text_encoder/model.safetensors"),
            &[("model.layer.weight", "F16", &[2], 4)],
        );
        write_typed_safetensors(
            &root.join("vae/model.safetensors"),
            &[("decoder.weight", "BF16", &[3], 6)],
        );
        let dit = tmp.path().join("dit.safetensors");
        write_typed_safetensors(
            &dit,
            &[("model.diffusion_model.img_in.weight", "F8_E4M3", &[2, 4], 8)],
        );
        LoadSpec::new(WeightsSource::File(dit))
            .with_component(gen_core::BASE_SNAPSHOT_COMPONENT, WeightsSource::Dir(root))
    }

    #[test]
    fn imported_file_asset_facts_follow_dit_bf16_and_te_vae_f32_load_dtypes() {
        let tmp = tempfile::tempdir().unwrap();
        let mut spec = file_spec(&tmp);
        let vae = tmp.path().join("imported-vae.safetensors");
        write_typed_safetensors(&vae, &[("decoder.weight", "F16", &[5], 10)]);
        spec.components.insert(
            gen_core::COMFYUI_VAE_COMPONENT.into(),
            WeightsSource::File(vae),
        );
        let contract = provider_contract("qwen_image", &spec).unwrap();
        assert_eq!(contract.asset_facts.conditioning_bytes, 2 * 4);
        assert_eq!(contract.asset_facts.transformer_bytes, 2 * 4 * 2);
        assert_eq!(contract.asset_facts.decoder_bytes, 5 * 4);
        assert_eq!(contract.asset_facts.base_bytes, 44);
    }

    #[test]
    fn imported_file_contract_and_loader_share_the_full_typed_field_matrix() {
        let tmp = tempfile::tempdir().unwrap();
        let valid = file_spec(&tmp);
        let mut cases = vec![("valid", valid.clone())];

        let mut precision = valid.clone();
        precision.precision = Precision::Fp32;
        cases.push(("precision-is-accepted", precision));
        let mut pid = valid.clone();
        pid.pid = Some(gen_core::PidWeights {
            checkpoint: WeightsSource::File(tmp.path().join("pid.safetensors")),
            gemma: WeightsSource::Dir(tmp.path().join("gemma")),
        });
        cases.push(("pid-is-accepted", pid));

        let mut adapter = valid.clone();
        adapter.adapters.push(gen_core::AdapterSpec::new(
            tmp.path().join("adapter.safetensors"),
            1.0,
            gen_core::AdapterKind::Lora,
        ));
        cases.push(("adapter", adapter));
        let mut quant = valid.clone();
        quant.quantize = Some(Quant::Q4);
        cases.push(("quant", quant));
        let mut control = valid.clone();
        control.control = Some(WeightsSource::File(tmp.path().join("control.safetensors")));
        cases.push(("control", control));
        let mut extra = valid.clone();
        extra
            .extra_controls
            .push(WeightsSource::File(tmp.path().join("extra.safetensors")));
        cases.push(("extra-control", extra));
        let mut ip = valid.clone();
        ip.ip_adapter = Some(WeightsSource::File(tmp.path().join("ip.safetensors")));
        cases.push(("ip-adapter", ip));
        let mut identity = valid.clone();
        identity.identity = Some(gen_core::IdentityWeights::default());
        cases.push(("identity", identity));
        let mut external_te = valid.clone();
        external_te.text_encoder = Some(WeightsSource::Dir(tmp.path().join("external-te")));
        cases.push(("external-text-encoder", external_te));
        let mut unknown = valid.clone();
        unknown.components.insert(
            "unknown".into(),
            WeightsSource::File(tmp.path().join("unknown.safetensors")),
        );
        cases.push(("unknown-component", unknown));
        let mut vae_dir = valid.clone();
        vae_dir.components.insert(
            gen_core::COMFYUI_VAE_COMPONENT.into(),
            WeightsSource::Dir(tmp.path().join("vae-dir")),
        );
        cases.push(("vae-dir", vae_dir));

        for (name, spec) in cases {
            assert_eq!(
                crate::validate_load_spec(&spec).is_ok(),
                provider_contract("qwen_image", &spec).is_ok(),
                "File loader/contract validation drift for {name}"
            );
        }
    }

    fn spec(tmp: &tempfile::TempDir) -> LoadSpec {
        let root = tmp.path().join("qwen-candle-memory-spec");
        for component in ["text_encoder", "transformer", "vae"] {
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
        if matches!(
            strategy,
            MemoryStrategy::BoundedDecode
                | MemoryStrategy::BoundedAttention
                | MemoryStrategy::BoundedTransformerResidency
        ) {
            parameters.decode_tile_edge = Some(DECODE_TILE_EDGE);
            parameters.decode_overlap = Some(DECODE_OVERLAP);
        }
        if matches!(
            strategy,
            MemoryStrategy::BoundedAttention | MemoryStrategy::BoundedTransformerResidency
        ) {
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
    fn qwen_base_and_edit_publish_the_full_candle_ladder() {
        let tmp = tempfile::tempdir().unwrap();
        for id in ["qwen_image", "qwen_image_edit"] {
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
                    .capability(MemoryStrategy::BoundedAttention)
                    .unwrap()
                    .parameters
                    .attention_chunk_sizes,
                [ATTENTION_CHUNK_SIZE]
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
    fn evidence_identity_and_tier_match_the_executable_contract_and_packed_snapshot() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let transformer = root.join("transformer");
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(
            transformer.join("config.json"),
            br#"{"quantization":{"group_size":64,"bits":4}}"#,
        )
        .unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir(root.clone()));

        for provider_id in ["qwen_image", "qwen_image_edit"] {
            let contract = provider_contract(provider_id, &spec).unwrap();
            let (identity, tier) = evidence_identity_and_tier(provider_id, &spec).unwrap();
            assert_eq!(identity, contract.calibration.unwrap());
            assert_eq!(identity.fingerprint, CALIBRATION_FINGERPRINT);
            assert_eq!(tier.precision, Precision::Bf16);
            assert_eq!(tier.quant, Some(Quant::Q4));
        }
    }

    #[test]
    fn weights_free_behavior_configures_and_finishes_the_exact_request_scope() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = spec(&tmp);
        let contract = provider_contract("qwen_image_edit", &spec).unwrap();
        let mut fixture = registered_valid_fixture(
            &spec,
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
        )
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
        fixture.context.selection.parameters.transformer_window_size = Some(4);
        let mut scope =
            registered_begin_request("qwen_image_edit", &spec, &contract, &fixture.context)
                .unwrap()
                .unwrap();
        let admitted_request = fixture.request.clone();
        scope.configure_request(&mut fixture.request).unwrap();
        assert_eq!(
            fixture.request.memory,
            contract.generation_memory(&fixture.context.selection)
        );
        let mut missing_reference = admitted_request.clone();
        missing_reference.conditioning.clear();
        assert!(scope.configure_request(&mut missing_reference).is_err());
        let mut extra_reference = admitted_request;
        let duplicate_reference = extra_reference.conditioning[0].clone();
        extra_reference.conditioning.push(duplicate_reference);
        assert!(scope.configure_request(&mut extra_reference).is_err());
        let mut wrong_decode_geometry = fixture.context.geometry;
        wrong_decode_geometry.reference_count = 0;
        assert!(scope
            .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, wrong_decode_geometry)
            .is_err());
        assert!(scope
            .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, fixture.context.geometry)
            .is_ok());
        assert!(scope.materialize_transformer_window(1, 4).is_err());
        assert!(scope.materialize_transformer_window(0, 4).is_ok());
        scope.finish(MemoryRunOutcome::Complete).unwrap();
        assert!(scope.finish(MemoryRunOutcome::Complete).is_err());
    }

    #[test]
    fn stale_calibration_fingerprint_is_rejected_before_execution() {
        let tmp = tempfile::tempdir().unwrap();
        let spec = spec(&tmp);
        let contract = provider_contract("qwen_image", &spec).unwrap();
        let mut fixture =
            registered_valid_fixture(&spec, &contract, MemoryStrategy::BoundedAttention)
                .unwrap()
                .pop()
                .unwrap();
        fixture.context.calibration_fingerprint = "stale-qwen-calibration".into();
        assert!(matches!(
            registered_safety_check(&spec, &contract, &fixture.context),
            MemorySafetyDecision::Reject { .. }
        ));
    }

    #[test]
    fn every_optimized_selection_is_staged_and_exactly_parameterized() {
        let tmp = tempfile::tempdir().unwrap();
        let contract = provider_contract("qwen_image", &spec(&tmp)).unwrap();
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
            assert_eq!(
                memory.tile_vae_decode,
                matches!(
                    strategy,
                    MemoryStrategy::BoundedDecode
                        | MemoryStrategy::BoundedAttention
                        | MemoryStrategy::BoundedTransformerResidency
                )
            );
            assert_eq!(
                memory.chunk_attention,
                matches!(
                    strategy,
                    MemoryStrategy::BoundedAttention | MemoryStrategy::BoundedTransformerResidency
                )
            );
        }

        let mut rejected = selection(MemoryStrategy::BoundedDecode);
        rejected.parameters.decode_overlap = Some(REJECTED_SUB_512_OVERLAP);
        assert!(contract.validate_selection(&rejected).is_err());
    }

    #[test]
    fn adapters_and_eager_loads_do_not_overstate_block_streaming() {
        let tmp = tempfile::tempdir().unwrap();
        let mut adapted = spec(&tmp);
        adapted.adapters.push(gen_core::AdapterSpec::new(
            "lightning.safetensors".into(),
            1.0,
            gen_core::AdapterKind::Lora,
        ));
        let mut eager = spec(&tmp);
        eager.load_shape = LoadShape::EagerMaterialization;
        for candidate in [adapted, eager] {
            let contract = provider_contract("qwen_image_edit", &candidate).unwrap();
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
