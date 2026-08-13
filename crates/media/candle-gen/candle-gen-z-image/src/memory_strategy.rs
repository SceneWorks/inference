//! Candle/CUDA Z-Image adoption of the shared five-rung image memory contract (sc-15815).
//!
//! `z_image_edit` is a SceneWorks catalog alias for the `z_image_turbo` provider and therefore
//! inherits the Turbo contract. The two bespoke control providers are registered separately so the
//! matrix records their current gaps rather than inferring capability from the plain provider.

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, LoadSpec, MemoryNumericTier, MemoryProviderContract, MemoryRunContext,
    MemorySafetyDecision, MemoryStrategy, Precision, Quant, WeightsSource,
};
#[cfg(any(feature = "cuda", test))]
use candle_gen::gen_core::{
    LoadShape, MemoryAssetFacts, MemoryBackendRealization, MemoryCalibrationIdentity,
    MemoryFormulaKind, MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryParameterRanges,
    MemoryPrerequisiteScope, MemoryStrategyCapability, MemoryStrategyPrerequisite,
    MemoryStrategySupport, MemoryWindowMaterialization, PerComponentBytes,
};
#[cfg(test)]
use gen_core::{GenerationMemory, GenerationRequest, MemoryGeometry, MemoryRunOutcome};
#[cfg(any(feature = "cuda", test))]
use gen_core::{MemoryPhase, MemoryRequestScope};

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
pub(crate) const CONTROL_CALIBRATION_FINGERPRINT: &str =
    "z-image-cuda-base-control-host-decode-streamed-device-format-blocks-v2";

#[cfg(any(feature = "cuda", test))]
fn imported_tensor_bytes(
    tensor: &gen_core::weightsmeta::SafetensorsTensorHeader,
    loaded_name: &str,
    component: &str,
) -> gen_core::Result<u64> {
    use gen_core::weightsmeta::Dtype;

    // `candle_core::safetensors::load` first materializes every source tensor. U16 is promoted to
    // Candle's U32 storage; the remaining accepted integer widths stay native. Every float then
    // passes through `normalize_fp8_map(..., BF16)`, including dense f32/f64 and plain/scaled fp8.
    let loaded = match tensor.dtype {
        Dtype::U8 | Dtype::U32 | Dtype::I16 | Dtype::I32 | Dtype::I64 => tensor.data_bytes,
        Dtype::U16 => tensor.materialized_bytes(4)?,
        Dtype::F8_E4M3 | Dtype::F16 | Dtype::BF16 | Dtype::F32 | Dtype::F64 => {
            tensor.materialized_bytes(2)?
        }
        dtype => {
            return Err(gen_core::Error::Unsupported(format!(
                "z-image imported {component} tensor {:?} uses unsupported Candle dtype {dtype:?}",
                tensor.name
            )))
        }
    };
    // Test the key *after* any combined-checkpoint component prefix is stripped. That is the key
    // `normalize_fp8_map` sees, so a combined `model.diffusion_model.scaled_fp8` marker is omitted
    // exactly like a standalone transformer's unprefixed `scaled_fp8` marker.
    if loaded_name == "scaled_fp8"
        || loaded_name.ends_with(".scale_weight")
        || loaded_name.ends_with(".weight_scale")
        || loaded_name.ends_with(".scale_input")
        || loaded_name.ends_with(".input_scale")
    {
        Ok(0)
    } else {
        Ok(loaded)
    }
}

#[cfg(any(feature = "cuda", test))]
fn single_file_tensor_bytes(path: &std::path::Path, component: &str) -> gen_core::Result<u64> {
    let bytes = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?
        .iter()
        .try_fold(0_u64, |total, tensor| {
            total
                .checked_add(imported_tensor_bytes(tensor, &tensor.name, component)?)
                .ok_or_else(|| {
                    gen_core::Error::Msg(format!(
                        "z-image imported {component} resident byte sum overflow"
                    ))
                })
        })?;
    if bytes == 0 {
        return Err(gen_core::Error::Msg(format!(
            "z-image imported {component} '{}' contains no tensor bytes",
            path.display()
        )));
    }
    Ok(bytes)
}

#[cfg(any(feature = "cuda", test))]
fn combined_file_components(path: &std::path::Path) -> gen_core::Result<PerComponentBytes> {
    let mut components = PerComponentBytes::default();
    let mut mapped_keys = [
        std::collections::HashSet::new(),
        std::collections::HashSet::new(),
        std::collections::HashSet::new(),
    ];
    for tensor in gen_core::weightsmeta::safetensors_path_tensor_headers(path)? {
        let Some((component, mapped)) = crate::comfyui::combined_component_key(&tensor.name) else {
            return Err(gen_core::Error::Msg(format!(
                "z-image combined checkpoint tensor {:?} has no component mapping",
                tensor.name
            )));
        };
        if mapped.is_empty() {
            return Err(gen_core::Error::Msg(format!(
                "z-image combined checkpoint tensor {:?} maps to an empty component key",
                tensor.name
            )));
        }
        let (component_index, component_name) = match component {
            crate::comfyui::CombinedComponent::Transformer => (0, "transformer"),
            crate::comfyui::CombinedComponent::TextEncoder => (1, "text encoder"),
            crate::comfyui::CombinedComponent::Vae => (2, "VAE"),
        };
        if !mapped_keys[component_index].insert(mapped.to_owned()) {
            return Err(gen_core::Error::Msg(format!(
                "z-image combined checkpoint tensor {:?} collides at {component_name} key {mapped:?}",
                tensor.name
            )));
        }
        let resident_bytes = imported_tensor_bytes(&tensor, mapped, component_name)?;
        match component {
            crate::comfyui::CombinedComponent::Transformer => {
                components.dit = components.dit.checked_add(resident_bytes).ok_or_else(|| {
                    gen_core::Error::Msg(
                        "z-image combined transformer resident byte sum overflow".into(),
                    )
                })?
            }
            crate::comfyui::CombinedComponent::TextEncoder => {
                components.text_encoder = components
                    .text_encoder
                    .checked_add(resident_bytes)
                    .ok_or_else(|| {
                        gen_core::Error::Msg(
                            "z-image combined text-encoder resident byte sum overflow".into(),
                        )
                    })?
            }
            crate::comfyui::CombinedComponent::Vae => {
                components.vae = components.vae.checked_add(resident_bytes).ok_or_else(|| {
                    gen_core::Error::Msg("z-image combined VAE resident byte sum overflow".into())
                })?
            }
        }
    }
    for (component, bytes) in [
        ("transformer", components.dit),
        ("text encoder", components.text_encoder),
        ("VAE", components.vae),
    ] {
        if bytes == 0 {
            return Err(gen_core::Error::Msg(format!(
                "z-image combined checkpoint '{}' is missing the {component} component",
                path.display()
            )));
        }
    }
    Ok(components)
}

#[cfg(any(feature = "cuda", test))]
fn imported_file_components(
    spec: &LoadSpec,
    primary: &std::path::Path,
    provider_id: &str,
) -> gen_core::Result<PerComponentBytes> {
    let _ = gen_core::require_base_snapshot(spec, provider_id)?;
    let text_encoder = spec
        .components
        .get(gen_core::COMFYUI_TEXT_ENCODER_COMPONENT);
    let vae = spec.components.get(gen_core::COMFYUI_VAE_COMPONENT);
    match (text_encoder, vae) {
        (None, None) => {
            spec.read_file_unchanged_if_prepared(primary, combined_file_components)
        }
        (Some(WeightsSource::File(text_encoder)), Some(WeightsSource::File(vae))) => {
            Ok(PerComponentBytes {
                text_encoder: spec.read_file_unchanged_if_prepared(text_encoder, |p| {
                    single_file_tensor_bytes(p, "text encoder")
                })?,
                dit: spec.read_file_unchanged_if_prepared(primary, |p| {
                    single_file_tensor_bytes(p, "transformer")
                })?,
                vae: spec.read_file_unchanged_if_prepared(vae, |p| {
                    single_file_tensor_bytes(p, "VAE")
                })?,
            })
        }
        (Some(WeightsSource::Dir(path)), _) => Err(gen_core::Error::Msg(format!(
            "{provider_id}: component '{}' must be a file, not {}",
            gen_core::COMFYUI_TEXT_ENCODER_COMPONENT,
            path.display()
        ))),
        (_, Some(WeightsSource::Dir(path))) => Err(gen_core::Error::Msg(format!(
            "{provider_id}: component '{}' must be a file, not {}",
            gen_core::COMFYUI_VAE_COMPONENT,
            path.display()
        ))),
        _ => Err(gen_core::Error::Msg(format!(
            "{provider_id}: separate ComfyUI import requires both '{}' and '{}', or neither for a combined checkpoint",
            gen_core::COMFYUI_TEXT_ENCODER_COMPONENT,
            gen_core::COMFYUI_VAE_COMPONENT
        ))),
    }
}

#[cfg(any(feature = "cuda", test))]
pub(crate) fn provider_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    if provider_id == crate::base::MODEL_ID
        && matches!(spec.weights, gen_core::WeightsSource::File(_))
    {
        return Err(gen_core::Error::Msg(
            "z_image expects a snapshot directory (tokenizer/ text_encoder/ transformer/ vae/), \
             not a single .safetensors file"
                .into(),
        ));
    }
    if provider_id == crate::MODEL_ID {
        crate::validate_load_spec(spec)?;
    }
    // File and Dir intentionally retain one provider/calibration identity: the executable provider,
    // phase graph, and output semantics are the same. The promoted matrix has no load-source axis,
    // however, so only Dir advertises rung 4 until the pinned/re-openable File path has its own real
    // measurement. A Dir rung-4 cell must never be relabeled as File evidence.
    let streamable = matches!(spec.load_shape, LoadShape::DeferredMaterialization)
        && matches!(spec.weights, gen_core::WeightsSource::Dir(_));
    let components = match &spec.weights {
        gen_core::WeightsSource::Dir(_) => PerComponentBytes::from_spec_subdirs(
            spec,
            &["text_encoder"],
            &["transformer"],
            &["vae"],
        )
        .unwrap_or_default(),
        gen_core::WeightsSource::File(path) => imported_file_components(spec, path, provider_id)?,
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
        decode_geometry_policy_authoritative: false,
        // PiD replaces the native VAE with a separately planned decoder. Until that decoder accepts
        // this provider's bounded host-decode route, the request safety gate rejects optimized PiD
        // runs.
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

/// Explicit contract for the bespoke dual-network control routes. The control encoder, text encoder,
/// denoiser, and decoder are phase-loaded; both the base and control main stacks honor the selected
/// transformer window.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn control_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    let mut contract = provider_contract(provider_id, spec)?;
    let overlay_bytes = match spec.control.as_ref() {
        Some(gen_core::WeightsSource::Dir(path)) => gen_core::safetensors_path_bytes(path),
        Some(gen_core::WeightsSource::File(path)) => spec
            .read_file_unchanged_if_prepared(path, |p| -> gen_core::Result<u64> {
                Ok(gen_core::safetensors_path_bytes(p))
            })?,
        None => 0,
    };
    contract.asset_facts.base_bytes = contract
        .asset_facts
        .base_bytes
        .saturating_add(overlay_bytes);
    contract.asset_facts.transformer_bytes = contract
        .asset_facts
        .transformer_bytes
        .saturating_add(overlay_bytes);
    contract.asset_facts.conditioning_bytes = contract
        .asset_facts
        .conditioning_bytes
        .max(contract.asset_facts.decoder_bytes);
    contract.asset_facts.overlay_bytes = overlay_bytes;
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        CONTROL_CALIBRATION_FINGERPRINT,
        spec.load_shape,
    ));
    Ok(contract)
}

pub(crate) fn request_scope(
    provider_id: &'static str,
    device: Device,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<candle_gen::request_scope::CandleRequestScopeCore> {
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        provider_id,
        device,
        context.geometry,
        contract.generation_memory(&context.selection),
        context.use_pid,
        candle_transformers::models::z_image::transformer::Config::z_image_turbo().n_layers,
        move |use_pid, tile_edge, overlap| {
            if use_pid {
                return Err(gen_core::Error::Unsupported(format!(
                    "{provider_id}: PiD uses an alternate decoder whose explicit tile plan is not yet wired"
                )));
            }
            if tile_edge == DECODE_TILE_EDGE && overlap == DECODE_OVERLAP {
                Ok(())
            } else {
                Err(gen_core::Error::Unsupported(format!(
                    "{provider_id}: native decode tiling is fixed at {DECODE_TILE_EDGE}/{DECODE_OVERLAP}, got {tile_edge}/{overlap}"
                )))
            }
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
    Ok(candle_gen::request_scope::CandleRequestScopeCore::new(
        config,
    ))
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
        WeightsSource::File(_) => return Ok(None),
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
    let is_control = contract.provider_id.ends_with("_control");
    let context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        MemoryNumericTier {
            precision: Precision::Bf16,
            quant: snapshot_quant_tier(spec, &contract.provider_id)?,
            component_precision_floors: &[],
        },
        gen_core::MemoryBehaviorRoute {
            mode: if is_control {
                gen_core::MemoryMode::ImageToImage
            } else {
                gen_core::MemoryMode::TextToImage
            },
            reference_count: u32::from(is_control),
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
    Ok(Some(Box::new(request_scope(
        provider_id,
        Device::Cpu,
        contract,
        context,
    )?)))
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

    fn write_safetensors(path: &std::path::Path, tensors: &[(&str, usize)]) {
        let mut offset = 0_usize;
        let mut header = serde_json::Map::new();
        for (name, bytes) in tensors {
            let bytes = *bytes;
            header.insert(
                (*name).to_owned(),
                serde_json::json!({
                    "dtype": "U8",
                    "shape": [bytes],
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(vec![0_u8; offset]);
        std::fs::write(path, file).unwrap();
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
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.resize(file.len() + offset, 0);
        std::fs::write(path, file).unwrap();
    }

    fn separate_file_spec(tmp: &tempfile::TempDir) -> LoadSpec {
        let base = tmp.path().join("base");
        std::fs::create_dir_all(&base).unwrap();
        let dit = tmp.path().join("dit.safetensors");
        let text = tmp.path().join("text.safetensors");
        let vae = tmp.path().join("vae.safetensors");
        write_safetensors(&dit, &[("block.weight", 32)]);
        write_safetensors(&text, &[("layer.weight", 16)]);
        write_safetensors(&vae, &[("decoder.weight", 8)]);
        LoadSpec::new(WeightsSource::File(dit))
            .with_component(gen_core::BASE_SNAPSHOT_COMPONENT, WeightsSource::Dir(base))
            .with_component(
                gen_core::COMFYUI_TEXT_ENCODER_COMPONENT,
                WeightsSource::File(text),
            )
            .with_component(gen_core::COMFYUI_VAE_COMPONENT, WeightsSource::File(vae))
    }

    #[test]
    fn file_asset_facts_follow_bf16_materialization_and_omit_fp8_companions() {
        let tmp = tempfile::tempdir().unwrap();
        let base = tmp.path().join("base");
        std::fs::create_dir_all(&base).unwrap();
        let dit = tmp.path().join("dit.safetensors");
        let text = tmp.path().join("text.safetensors");
        let vae = tmp.path().join("vae.safetensors");
        write_typed_safetensors(
            &dit,
            &[
                ("block.weight", "F8_E4M3", &[2, 4], 8),
                ("block.weight_scale", "F32", &[], 4),
                ("dense.weight", "F32", &[3], 12),
            ],
        );
        write_typed_safetensors(&text, &[("layer.weight", "F32", &[3], 12)]);
        write_typed_safetensors(&vae, &[("decoder.weight", "BF16", &[4], 8)]);
        let spec = LoadSpec::new(WeightsSource::File(dit))
            .with_component(gen_core::BASE_SNAPSHOT_COMPONENT, WeightsSource::Dir(base))
            .with_component(
                gen_core::COMFYUI_TEXT_ENCODER_COMPONENT,
                WeightsSource::File(text),
            )
            .with_component(gen_core::COMFYUI_VAE_COMPONENT, WeightsSource::File(vae));
        let contract = provider_contract(crate::MODEL_ID, &spec).unwrap();
        assert_eq!(contract.asset_facts.transformer_bytes, 16 + 6);
        assert_eq!(contract.asset_facts.conditioning_bytes, 6);
        assert_eq!(contract.asset_facts.decoder_bytes, 8);
        assert_eq!(contract.asset_facts.base_bytes, 36);
    }

    #[test]
    fn file_contract_and_loader_share_the_full_typed_field_matrix() {
        let tmp = tempfile::tempdir().unwrap();
        let valid = separate_file_spec(&tmp);
        let mut cases = vec![("valid", valid.clone())];

        let mut precision = valid.clone();
        precision.precision = Precision::Fp32;
        cases.push(("precision-is-accepted", precision));
        let mut adapter = valid.clone();
        adapter.adapters.push(gen_core::AdapterSpec::new(
            tmp.path().join("adapter.safetensors"),
            1.0,
            gen_core::AdapterKind::Lora,
        ));
        cases.push(("adapter-is-accepted", adapter));

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
        let mut half_separate = valid.clone();
        half_separate
            .components
            .remove(gen_core::COMFYUI_VAE_COMPONENT);
        cases.push(("half-separate", half_separate));

        for (name, spec) in cases {
            assert_eq!(
                crate::validate_load_spec(&spec).is_ok(),
                provider_contract(crate::MODEL_ID, &spec).is_ok(),
                "File loader/contract validation drift for {name}"
            );
        }
    }

    #[test]
    fn combined_and_separate_imports_publish_the_same_exact_phase_inventory() {
        let tmp = tempfile::tempdir().unwrap();
        let tokenizer = tmp.path().join("tokenizer-base");
        std::fs::create_dir_all(&tokenizer).unwrap();
        let combined = tmp.path().join("combined.safetensors");
        write_safetensors(
            &combined,
            &[
                ("model.diffusion_model.block.weight", 11),
                // The combined loader strips the component prefix, then `normalize_fp8_map`
                // discards this exact marker. Asset facts must not count its source payload.
                ("model.diffusion_model.scaled_fp8", 13),
                ("conditioner.embedders.0.transformer.layer.weight", 7),
                ("first_stage_model.decoder.weight", 5),
            ],
        );
        let combined_spec = LoadSpec::new(WeightsSource::File(combined)).with_component(
            gen_core::BASE_SNAPSHOT_COMPONENT,
            WeightsSource::Dir(tokenizer.clone()),
        );
        let combined_contract = provider_contract(crate::MODEL_ID, &combined_spec).unwrap();

        let dit = tmp.path().join("dit.safetensors");
        let text_encoder = tmp.path().join("text-encoder.safetensors");
        let vae = tmp.path().join("vae.safetensors");
        write_safetensors(&dit, &[("block.weight", 11)]);
        write_safetensors(&text_encoder, &[("layer.weight", 7)]);
        write_safetensors(&vae, &[("decoder.weight", 5)]);
        let separate_spec = LoadSpec::new(WeightsSource::File(dit))
            .with_component(
                gen_core::BASE_SNAPSHOT_COMPONENT,
                WeightsSource::Dir(tokenizer),
            )
            .with_component(
                gen_core::COMFYUI_TEXT_ENCODER_COMPONENT,
                WeightsSource::File(text_encoder),
            )
            .with_component(gen_core::COMFYUI_VAE_COMPONENT, WeightsSource::File(vae));
        let separate_contract = provider_contract(crate::MODEL_ID, &separate_spec).unwrap();

        for contract in [&combined_contract, &separate_contract] {
            assert_eq!(contract.asset_facts.transformer_bytes, 11);
            assert_eq!(contract.asset_facts.conditioning_bytes, 7);
            assert_eq!(contract.asset_facts.decoder_bytes, 5);
            assert_eq!(contract.asset_facts.base_bytes, 23);
        }
    }

    #[test]
    fn combined_inventory_fails_closed_on_missing_or_unmapped_components() {
        let tmp = tempfile::tempdir().unwrap();
        let tokenizer = tmp.path().join("tokenizer-base");
        std::fs::create_dir_all(&tokenizer).unwrap();
        for (name, tensors, expected) in [
            (
                "missing-vae.safetensors",
                vec![
                    ("model.diffusion_model.block.weight", 11),
                    ("text_encoder.layer.weight", 7),
                ],
                "missing the VAE",
            ),
            (
                "unknown.safetensors",
                vec![
                    ("model.diffusion_model.block.weight", 11),
                    ("text_encoder.layer.weight", 7),
                    ("first_stage_model.decoder.weight", 5),
                    ("mystery.weight", 3),
                ],
                "no component mapping",
            ),
            (
                "collision.safetensors",
                vec![
                    ("model.diffusion_model.block.weight", 11),
                    ("transformer.block.weight", 11),
                    ("text_encoder.layer.weight", 7),
                    ("first_stage_model.decoder.weight", 5),
                ],
                "collides",
            ),
        ] {
            let checkpoint = tmp.path().join(name);
            write_safetensors(&checkpoint, &tensors);
            let spec = LoadSpec::new(WeightsSource::File(checkpoint)).with_component(
                gen_core::BASE_SNAPSHOT_COMPONENT,
                WeightsSource::Dir(tokenizer.clone()),
            );
            let error = provider_contract(crate::MODEL_ID, &spec)
                .unwrap_err()
                .to_string();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn z_image_base_load_and_memory_contract_reject_the_same_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/z-image.safetensors".into()));
        let load_error = crate::base::load(&spec)
            .err()
            .expect("base generator must reject File")
            .to_string();
        let contract_error = provider_contract(crate::base::MODEL_ID, &spec)
            .unwrap_err()
            .to_string();
        assert_eq!(contract_error, load_error);
        assert!(contract_error.contains("snapshot directory"));
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
            optimization_authority: gen_core::MemoryOptimizationAuthority::Calibrated,
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
                reference_count: 0,
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
        let mut scope = request_scope(crate::MODEL_ID, Device::Cpu, &contract, &context).unwrap();
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
    fn registered_behavior_treats_admitted_batch_as_a_maximum() {
        let spec = spec();
        let contract = provider_contract(crate::MODEL_ID, &spec).unwrap();
        let mut context = context(&contract);
        context.geometry.batch = 3;
        let mut scope = (crate::TURBO_MEMORY_BEHAVIOR.begin_request)(&spec, &contract, &context)
            .unwrap()
            .expect("registered behavior must construct the Candle scope core");
        let mut prefix = GenerationRequest {
            width: 768,
            height: 768,
            count: 1,
            ..Default::default()
        };
        scope.configure_request(&mut prefix).unwrap();
        scope
            .configure_decode(
                DECODE_TILE_EDGE,
                DECODE_OVERLAP,
                MemoryGeometry {
                    batch: 1,
                    ..context.geometry
                },
            )
            .unwrap();
        for geometry in [
            MemoryGeometry {
                width: context.geometry.width / 2,
                ..context.geometry
            },
            MemoryGeometry {
                height: context.geometry.height / 2,
                ..context.geometry
            },
            MemoryGeometry {
                frames: context.geometry.frames + 1,
                ..context.geometry
            },
            MemoryGeometry {
                reference_count: 1,
                ..context.geometry
            },
            MemoryGeometry {
                batch: 0,
                ..context.geometry
            },
            MemoryGeometry {
                batch: 4,
                ..context.geometry
            },
        ] {
            assert!(scope
                .configure_decode(DECODE_TILE_EDGE, DECODE_OVERLAP, geometry)
                .is_err());
        }
        scope.materialize_transformer_window(0, 4).unwrap();
        assert!(scope.materialize_transformer_window(0, 0).is_err());
        assert!(scope.materialize_transformer_window(1, 4).is_err());
        let mut overflow = GenerationRequest {
            width: 768,
            height: 768,
            count: 4,
            ..Default::default()
        };
        assert!(scope.configure_request(&mut overflow).is_err());
        scope.finish(MemoryRunOutcome::Complete).unwrap();
        assert!(scope.configure_request(&mut prefix).is_err());
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
            let root_tmp = tempfile::tempdir().unwrap();
            let root = root_tmp.path().to_path_buf();
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
        }
    }

    #[test]
    fn adapters_preserve_the_full_streamed_transformer_ladder() {
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
            MemoryStrategySupport::Implemented
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

        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        let transformer = root.join("transformer");
        std::fs::create_dir_all(&transformer).unwrap();
        std::fs::write(
            transformer.join("config.json"),
            br#"{"quantization":{"group_size":64,"bits":4}}"#,
        )
        .unwrap();
        let packed_spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let packed = provider_contract(crate::MODEL_ID, &packed_spec).unwrap();
        assert!(matches!(
            packed.backend,
            MemoryBackendRealization::CandleCuda {
                block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
                ..
            }
        ));
    }

    #[test]
    fn control_routes_publish_the_full_executable_ladder() {
        for id in ["z_image_turbo_control", "z_image_control"] {
            let contract = control_contract(id, &spec()).unwrap();
            assert!(contract.conformance_errors().is_empty());
            assert_eq!(
                contract.calibration.as_ref().unwrap().fingerprint,
                CONTROL_CALIBRATION_FINGERPRINT
            );
            assert_ne!(
                contract.calibration.as_ref().unwrap().fingerprint,
                CALIBRATION_FINGERPRINT
            );
            assert_eq!(contract.load_shape, LoadShape::DeferredMaterialization);
            assert!(contract
                .strategies
                .iter()
                .all(|capability| { capability.support == MemoryStrategySupport::Implemented }));
            assert!(contract.lifecycle.synchronized_phase_release);
            assert!(contract.lifecycle.decode_tiling);
            assert!(contract.lifecycle.attention_chunking);
            assert!(contract.lifecycle.transformer_window_materialization);
        }
    }
}
