//! Request-scoped Candle/CUDA memory ladder for SenseNova-U1.
//!
//! SenseNova is one fused dual-path Qwen3 checkpoint.  The understanding and generation paths are
//! interleaved in every decoder layer and the FM head emits RGB patches directly, so component
//! staging and bounded decode are structural N/A.  The two executable levers are bounded attention
//! on every public mode and generation-path-only block residency on re-openable deferred snapshots.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, GenerationRequest, LoadShape, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier,
    MemoryParameterRanges, MemoryPhase, MemoryProviderContract, MemoryRequestScope,
    MemoryRunContext, MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability,
    MemoryStrategySupport, MemoryWindowMaterialization, Precision, Quant, TransformerComponent,
    WeightsSource,
};

pub const ATTENTION_CHUNK_SIZE: u32 = 16_777_216;
pub const TRANSFORMER_WINDOW_SIZE: u32 = 1;
/// Weights-free registry behavior identity. Production Candle contracts remain uncalibrated until
/// the deferred Windows/CUDA terminal campaign records an artifact-bound measurement.
pub const CALIBRATION_FINGERPRINT: &str = "sensenova-u1-candle-request-memory-ladder-static-v1";
const QUALITY_ROUTES: &[&str] = &[
    "sensenova_u1_8b",
    "sensenova_u1_8b_infographic_v2",
    "sensenova_u1_8b_infographic_v3",
];
const FAST_ROUTES: &[&str] = &[
    "sensenova_u1_8b_fast",
    "sensenova_u1_8b_infographic_v2_fast",
    "sensenova_u1_8b_infographic_v3_fast",
];

fn expected_repository(route: &str) -> Option<String> {
    QUALITY_ROUTES
        .iter()
        .chain(FAST_ROUTES)
        .find(|candidate| **candidate == route)
        .map(|route| format!("{}-mlx", route.replace('_', "-")))
}

/// Bind the caller's public route to the repository-bearing resolved path. The retained shard pins
/// then make that exact path immutable for the request lifetime. This deliberately accepts the
/// three repository spellings used by HF snapshots, the app-owned cache, and local tier mirrors.
pub(crate) fn validate_resolved_artifact_binding(spec: &LoadSpec) -> gen_core::Result<()> {
    let (Some(route), WeightsSource::Dir(root)) = (spec.resolved_route.as_deref(), &spec.weights)
    else {
        return Ok(());
    };
    let expected = expected_repository(route).ok_or_else(|| {
        gen_core::Error::Unsupported(format!("unknown SenseNova resolved route {route}"))
    })?;
    let expected_hf = format!("models--SceneWorks--{expected}");
    let expected_app = format!("SceneWorks__{expected}");
    let bound = root.components().any(|component| {
        let component = component.as_os_str().to_string_lossy();
        component == expected || component == expected_hf || component == expected_app
    });
    if !bound {
        return Err(gen_core::Error::Unsupported(format!(
            "sensenova: resolved route {route} requires repository identity SceneWorks/{expected}, but weights path {} carries no matching repository component",
            root.display()
        )));
    }
    Ok(())
}

/// Exact immutable identities for the checkpoint shards a deferred generation window re-opens.
///
/// `PinnedWeightsFile` is cross-platform and checks the lexical parent chain, snapshot entry,
/// resolved target, and target metadata.  The sorted inventory additionally rejects a shard being
/// added, removed, or renamed after load.
#[derive(Clone, Debug)]
pub(crate) struct CheckpointInventory {
    root: PathBuf,
    files: Vec<(PathBuf, gen_core::PinnedWeightsFile)>,
    config: gen_core::PinnedWeightsFile,
}

impl CheckpointInventory {
    pub(crate) fn capture(root: &Path) -> gen_core::Result<Self> {
        let files = crate::backbone_files(root).map_err(gen_core::Error::backend)?;
        let files = files
            .into_iter()
            .map(|path| {
                let pin = gen_core::PinnedWeightsFile::pin(&path)?;
                Ok((path, pin))
            })
            .collect::<gen_core::Result<Vec<_>>>()?;
        let config_path = root.join("config.json");
        let config = gen_core::PinnedWeightsFile::pin(&config_path).map_err(|error| {
            gen_core::Error::Unsupported(format!(
                "sensenova: cannot pin required tier/config identity {}: {error}",
                config_path.display()
            ))
        })?;
        let inventory = Self {
            root: root.to_path_buf(),
            files,
            config,
        };
        inventory.ensure_unchanged()?;
        Ok(inventory)
    }

    pub(crate) fn ensure_unchanged(&self) -> gen_core::Result<()> {
        let current = crate::backbone_files(&self.root).map_err(gen_core::Error::backend)?;
        let expected = self
            .files
            .iter()
            .map(|(path, _)| path.clone())
            .collect::<Vec<_>>();
        if current != expected {
            return Err(gen_core::Error::Unsupported(format!(
                "sensenova: checkpoint shard inventory changed after load (expected {}, found {})",
                expected.len(),
                current.len()
            )));
        }
        for (_, pin) in &self.files {
            pin.ensure_unchanged()?;
        }
        self.config.ensure_unchanged()?;
        Ok(())
    }

    fn bytes(&self) -> u64 {
        self.files
            .iter()
            .filter_map(|(path, _)| std::fs::metadata(path).ok())
            .fold(0_u64, |total, metadata| {
                total.saturating_add(metadata.len())
            })
    }

    pub(crate) fn validate_numeric_tier(&self, spec: &LoadSpec) -> gen_core::Result<()> {
        let actual = detect_checkpoint_quantization(&self.files)?;
        let declared = spec.quantize.map(|quant| quant.bits() as u8);
        if declared != actual {
            return Err(gen_core::Error::Unsupported(format!(
                "sensenova: declared numeric tier {:?} does not match checkpoint tensor packing {:?}",
                declared, actual
            )));
        }

        // Converter metadata is secondary provenance only. If present, it must agree with the
        // tensor keys/shapes that `quant::detect_linear` actually consumes; it can never authorize a
        // packed tier by itself.
        let path = self.root.join("config.json");
        let body = std::fs::read_to_string(&path).map_err(|error| {
            gen_core::Error::Unsupported(format!(
                "sensenova: cannot read pinned tier provenance {}: {error}",
                path.display()
            ))
        })?;
        let config: serde_json::Value = serde_json::from_str(&body).map_err(|error| {
            gen_core::Error::Unsupported(format!(
                "sensenova: malformed tier provenance {}: {error}",
                path.display()
            ))
        })?;
        if let Some(quantization) = config
            .get("quantization")
            .and_then(|value| value.as_object())
        {
            let recorded = (
                quantization.get("bits").and_then(|value| value.as_u64()),
                quantization
                    .get("group_size")
                    .and_then(|value| value.as_u64()),
            );
            if recorded != (actual.map(u64::from), Some(64)) {
                return Err(gen_core::Error::Unsupported(format!(
                    "sensenova: config quantization provenance {recorded:?} crosses checkpoint tensor packing {:?}/group64",
                    actual
                )));
            }
        }
        self.ensure_unchanged()
    }
}

fn is_backbone_linear(base: &str) -> bool {
    let Some(rest) = base.strip_prefix("language_model.model.layers.") else {
        return false;
    };
    let Some((_layer, tail)) = rest.split_once('.') else {
        return false;
    };
    if let Some(projection) = tail.strip_prefix("self_attn.") {
        let projection = projection.strip_suffix("_mot_gen").unwrap_or(projection);
        return matches!(projection, "q_proj" | "k_proj" | "v_proj" | "o_proj");
    }
    ["mlp.", "mlp_mot_gen."].iter().any(|prefix| {
        tail.strip_prefix(prefix)
            .is_some_and(|projection| matches!(projection, "gate_proj" | "up_proj" | "down_proj"))
    })
}

/// Header-only mirror of `quant::detect_linear`: no tensor data is materialized.
fn detect_checkpoint_quantization(
    files: &[(PathBuf, gen_core::PinnedWeightsFile)],
) -> gen_core::Result<Option<u8>> {
    let paths = files.iter().map(|(path, _)| path).collect::<Vec<_>>();
    // SAFETY: read-only mappings held only for this header scan; the retained pins reject mutation.
    let tensors = unsafe { candle_gen::candle_core::safetensors::MmapedSafetensors::multi(&paths) }
        .map_err(gen_core::Error::backend)?;
    let views = tensors.tensors();
    let names = views
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let mut detected = None;
    let mut projections = 0_usize;
    for (name, weight) in &views {
        let Some(base) = name.strip_suffix(".weight") else {
            continue;
        };
        if !is_backbone_linear(base) {
            continue;
        }
        projections += 1;
        let scales_name = format!("{base}.scales");
        let biases_name = format!("{base}.biases");
        let packed = names.contains(scales_name.as_str());
        let tier = if packed {
            if !names.contains(biases_name.as_str()) {
                return Err(gen_core::Error::Unsupported(format!(
                    "sensenova: packed projection {base} has scales but no affine biases"
                )));
            }
            let scales = views
                .iter()
                .find(|(candidate, _)| candidate == &scales_name)
                .map(|(_, view)| view)
                .expect("name set came from views");
            let biases = views
                .iter()
                .find(|(candidate, _)| candidate == &biases_name)
                .map(|(_, view)| view)
                .expect("bias name was checked above");
            if weight.dtype() != safetensors_candle::Dtype::U32
                || scales.dtype() != safetensors_candle::Dtype::BF16
                || biases.dtype() != safetensors_candle::Dtype::BF16
            {
                return Err(gen_core::Error::Unsupported(format!(
                    "sensenova: packed projection {base} must use U32 codes with BF16 scales/biases, got {:?}/{:?}/{:?}",
                    weight.dtype(),
                    scales.dtype(),
                    biases.dtype()
                )));
            }
            let [out_features, lanes] = weight.shape() else {
                return Err(gen_core::Error::Unsupported(format!(
                    "sensenova: packed projection {base}.weight is not rank two"
                )));
            };
            let [scale_rows, groups] = scales.shape() else {
                return Err(gen_core::Error::Unsupported(format!(
                    "sensenova: packed projection {base}.scales is not rank two"
                )));
            };
            if out_features != scale_rows || *groups == 0 {
                return Err(gen_core::Error::Unsupported(format!(
                    "sensenova: packed projection {base} carries incompatible weight/scales shapes"
                )));
            }
            if biases.shape() != scales.shape() {
                return Err(gen_core::Error::Unsupported(format!(
                    "sensenova: packed projection {base} carries incompatible scales/biases shapes"
                )));
            }
            let input_features = groups.saturating_mul(crate::quant::PACKED_GROUP_SIZE);
            let numerator = lanes.saturating_mul(32);
            if input_features == 0 || !numerator.is_multiple_of(input_features) {
                return Err(gen_core::Error::Unsupported(format!(
                    "sensenova: packed projection {base} cannot resolve an exact bit width"
                )));
            }
            let bits = u8::try_from(numerator / input_features).map_err(|_| {
                gen_core::Error::Unsupported(format!(
                    "sensenova: packed projection {base} bit width overflows"
                ))
            })?;
            if !matches!(bits, 4 | 8) {
                return Err(gen_core::Error::Unsupported(format!(
                    "sensenova: packed projection {base} resolves unsupported {bits}-bit codes"
                )));
            }
            Some(bits)
        } else {
            if names.contains(biases_name.as_str()) {
                return Err(gen_core::Error::Unsupported(format!(
                    "sensenova: dense projection {base} has affine biases without scales"
                )));
            }
            if weight.dtype() != safetensors_candle::Dtype::BF16 {
                return Err(gen_core::Error::Unsupported(format!(
                    "sensenova: dense bf16 projection {base}.weight must use BF16, got {:?}",
                    weight.dtype()
                )));
            }
            None
        };
        if projections > 1 && detected != tier {
            return Err(gen_core::Error::Unsupported(
                "sensenova: checkpoint mixes dense/q4/q8 decoder projections".into(),
            ));
        }
        detected = tier;
    }
    if projections == 0 {
        return Err(gen_core::Error::Unsupported(
            "sensenova: checkpoint contains no recognizable decoder projection weights".into(),
        ));
    }
    Ok(detected)
}

pub(crate) fn validate_load_spec(provider_id: &str, spec: &LoadSpec) -> gen_core::Result<()> {
    if !matches!(provider_id, crate::MODEL_ID | crate::MODEL_ID_FAST) {
        return Err(gen_core::Error::Unsupported(format!(
            "unknown SenseNova provider {provider_id}"
        )));
    }
    if !matches!(spec.weights, WeightsSource::Dir(_)) {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: SenseNova requires a snapshot directory"
        )));
    }
    if spec.precision != Precision::Bf16 {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: only the dense-default bf16 source is supported"
        )));
    }
    if !matches!(spec.quantize, None | Some(Quant::Q4 | Quant::Q8)) {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: only bf16, q4, and q8 turnkey tiers are supported"
        )));
    }
    if !spec.adapters.is_empty()
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: control, adapter, PiD, identity, and external text-encoder composition is unsupported"
        )));
    }
    let known: &[&str] = if provider_id == crate::MODEL_ID_FAST {
        &["distill_lora"]
    } else {
        &[]
    };
    gen_core::reject_unknown_components(spec, known, provider_id)?;
    if let Some(route) = spec.resolved_route.as_deref() {
        let allowed = if provider_id == crate::MODEL_ID {
            QUALITY_ROUTES
        } else {
            FAST_ROUTES
        };
        if !allowed.contains(&route) {
            return Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: resolved route {route:?} is not one of the provider's exact public checkpoint identities"
            )));
        }
    }
    Ok(())
}

pub(crate) fn streamable_spec(provider_id: &str, spec: &LoadSpec) -> bool {
    if validate_load_spec(provider_id, spec).is_err()
        || spec.load_shape != LoadShape::DeferredMaterialization
    {
        return false;
    }
    let WeightsSource::Dir(root) = &spec.weights else {
        return false;
    };
    provider_id == crate::MODEL_ID
        || (provider_id == crate::MODEL_ID_FAST
            && root.join(crate::DISTILL_MERGED_MARKER).is_file())
}

/// Bind the declared numeric tier to the turnkey's converter-written provenance before any model
/// tensor reaches CUDA. Candle does not quantize at load time: q4/q8 must already be packed, while a
/// bf16 declaration must not point at a packed directory.
pub(crate) fn validate_artifact_tier(spec: &LoadSpec) -> gen_core::Result<()> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Ok(());
    };
    CheckpointInventory::capture(root)?.validate_numeric_tier(spec)
}

pub(crate) fn provider_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    validate_load_spec(provider_id, spec)?;
    validate_resolved_artifact_binding(spec)?;
    let inventory = match &spec.weights {
        WeightsSource::Dir(root) if root.is_dir() => Some(CheckpointInventory::capture(root)?),
        _ => None,
    };
    if let Some(inventory) = &inventory {
        inventory.validate_numeric_tier(spec)?;
    }
    Ok(build_contract(
        provider_id,
        spec,
        inventory.as_ref().map_or(0, CheckpointInventory::bytes),
        None,
    ))
}

pub(crate) fn weights_free_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    validate_load_spec(provider_id, spec)?;
    Ok(build_contract(
        provider_id,
        spec,
        0,
        Some(MemoryCalibrationIdentity::new(
            CALIBRATION_FINGERPRINT,
            spec.load_shape,
        )),
    ))
}

/// Production eager/lazy contract before weights materialize. Unlike the registry-only fixture
/// seam, it never grants a synthetic calibration identity.
pub(crate) fn uncalibrated_contract(
    provider_id: &str,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    validate_load_spec(provider_id, spec)?;
    validate_resolved_artifact_binding(spec)?;
    if matches!(&spec.weights, WeightsSource::Dir(root) if root.is_dir()) {
        validate_artifact_tier(spec)?;
    }
    Ok(build_contract(provider_id, spec, 0, None))
}

fn build_contract(
    provider_id: &str,
    spec: &LoadSpec,
    base_bytes: u64,
    calibration: Option<MemoryCalibrationIdentity>,
) -> MemoryProviderContract {
    let streamable = streamable_spec(provider_id, spec);
    let strategies = MemoryStrategy::ALL
        .into_iter()
        .map(|strategy| {
            let mut parameters = MemoryParameterRanges::default();
            let support = match strategy {
                MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
                MemoryStrategy::StagedResidency => {
                    MemoryStrategySupport::StructurallyNotApplicable {
                        reason: "SenseNova is one fused dual-path transformer with no separable conditioning component".into(),
                    }
                }
                MemoryStrategy::BoundedDecode => {
                    MemoryStrategySupport::StructurallyNotApplicable {
                        reason: "SenseNova has no VAE decode phase; its FM head emits RGB patches".into(),
                    }
                }
                MemoryStrategy::BoundedAttention => {
                    parameters.attention_chunk_sizes = vec![ATTENTION_CHUNK_SIZE];
                    MemoryStrategySupport::Implemented
                }
                MemoryStrategy::BoundedTransformerResidency if streamable => {
                    parameters.transformer_window_sizes = vec![TRANSFORMER_WINDOW_SIZE];
                    parameters.transformer_window_components = vec![TransformerComponent::Dit];
                    MemoryStrategySupport::Implemented
                }
                MemoryStrategy::BoundedTransformerResidency => MemoryStrategySupport::Missing,
            };
            MemoryStrategyCapability {
                strategy,
                support,
                parameters,
            }
        })
        .collect();
    // SenseNova has no independently releasable phase boundary: the same fused transformer owns
    // conditioning/understanding and generation. Keep lifecycle phases empty so the contract does
    // not falsely advertise the staged-residency hook. The formula may still describe the denoise
    // envelope used for admission accounting.
    let formula_phases = vec![MemoryPhase::Conditioning, MemoryPhase::Denoise];
    MemoryProviderContract {
        provider_id: provider_id.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: true,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies,
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: gen_core::ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: Vec::new(),
            synchronized_phase_release: false,
            decode_tiling: false,
            attention_chunking: true,
            transformer_window_materialization: streamable,
        },
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases: formula_phases,
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::BatchCount,
                MemoryFormulaVariable::ConditioningTokenCount,
                MemoryFormulaVariable::AttentionChunkSize,
                MemoryFormulaVariable::TransformerWindowSize,
            ],
        },
        calibration,
        asset_facts: MemoryAssetFacts {
            base_bytes,
            conditioning_bytes: base_bytes,
            transformer_bytes: base_bytes,
            decoder_bytes: 0,
            overlay_bytes: 0,
        },
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    }
}

fn validate_route(provider_id: &str, context: &MemoryRunContext) -> gen_core::Result<()> {
    if context.use_pid || context.overlay.is_some() || context.has_phases {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: SenseNova memory routes require plain, single-phase, non-PiD requests"
        )));
    }
    let interleave = matches!(&context.mode, MemoryMode::Other(mode) if mode == "interleave");
    let valid_batch = if interleave {
        (1..=10).contains(&context.geometry.batch)
    } else {
        context.geometry.batch == 1
    };
    if context.geometry.width == 0
        || context.geometry.height == 0
        || !valid_batch
        || context.geometry.frames != 1
        || context.has_reference != (context.geometry.reference_count > 0)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: invalid SenseNova memory geometry {}x{} batch={} frames={} references={} has_reference={}",
            context.geometry.width,
            context.geometry.height,
            context.geometry.batch,
            context.geometry.frames,
            context.geometry.reference_count,
            context.has_reference
        )));
    }
    let refs = context.geometry.reference_count;
    let route_ok = match &context.mode {
        MemoryMode::TextToImage => refs == 0,
        MemoryMode::ImageToImage | MemoryMode::Edit => (1..=5).contains(&refs),
        MemoryMode::Other(mode) if mode == "character_image" || mode == "edit_image" => {
            (1..=5).contains(&refs)
        }
        MemoryMode::Other(mode) if mode == "vqa" => {
            provider_id == crate::MODEL_ID
                && refs == 1
                && context.selection.strategy != MemoryStrategy::BoundedTransformerResidency
        }
        MemoryMode::Other(mode) if mode == "interleave" => {
            provider_id == crate::MODEL_ID && refs <= 10
        }
        _ => false,
    };
    if !route_ok {
        return Err(gen_core::Error::Unsupported(format!(
            "{provider_id}: memory mode {:?} with {refs} references is not an executable SenseNova route",
            context.mode
        )));
    }
    Ok(())
}

pub(crate) fn validate_context(
    provider_id: &str,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    quant: Option<Quant>,
) -> gen_core::Result<()> {
    if let MemorySafetyDecision::Reject { reason } = gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(MemoryNumericTier {
            precision: Precision::Bf16,
            quant,
            component_precision_floors: &[],
        }),
        None,
    ) {
        return Err(gen_core::Error::Unsupported(reason));
    }
    validate_route(provider_id, context)
}

pub(crate) fn safety_check(
    provider_id: &str,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    quant: Option<Quant>,
) -> MemorySafetyDecision {
    match validate_context(provider_id, contract, context, quant) {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub(crate) fn request_scope(
    provider_id: &'static str,
    device: Device,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
    transformer_blocks: usize,
) -> gen_core::Result<candle_gen::request_scope::CandleRequestScopeCore> {
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        provider_id,
        device,
        context.geometry,
        contract.generation_memory(&context.selection),
        false,
        transformer_blocks,
        move |_use_pid, _tile_edge, _overlap| {
            Err(gen_core::Error::Unsupported(format!(
                "{provider_id}: bounded decode is structurally not applicable"
            )))
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

pub(crate) fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    safety_check(&contract.provider_id, contract, context, spec.quantize)
}

pub(crate) fn validate_direct_operation_identity(
    provider_id: &str,
    context: &MemoryRunContext,
    actual_mode: &MemoryMode,
    actual_geometry: gen_core::MemoryGeometry,
) -> gen_core::Result<()> {
    if &context.mode == actual_mode && context.geometry == actual_geometry {
        return Ok(());
    }
    Err(gen_core::Error::Unsupported(format!(
        "{provider_id}: direct operation {}/{} references at {}x{} does not match admitted {}/{} references at {}x{}",
        actual_mode.as_key(),
        actual_geometry.reference_count,
        actual_geometry.width,
        actual_geometry.height,
        context.mode.as_key(),
        context.geometry.reference_count,
        context.geometry.width,
        context.geometry.height
    )))
}

pub(crate) fn registered_begin_request(
    provider_id: &'static str,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    validate_context(provider_id, contract, context, spec.quantize)?;
    Ok(Some(Box::new(request_scope(
        provider_id,
        Device::Cpu,
        contract,
        context,
        42,
    )?)))
}

pub(crate) fn registered_valid_fixture(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let routes = if strategy == MemoryStrategy::BoundedTransformerResidency {
        vec![
            (MemoryMode::TextToImage, 0),
            (MemoryMode::Edit, 1),
            (MemoryMode::Other("interleave".into()), 0),
        ]
    } else {
        vec![
            (MemoryMode::TextToImage, 0),
            (MemoryMode::Edit, 1),
            (MemoryMode::Other("vqa".into()), 1),
            (MemoryMode::Other("interleave".into()), 0),
        ]
    };
    routes
        .into_iter()
        .filter(|(mode, _)| {
            contract.provider_id == crate::MODEL_ID
                || !matches!(mode, MemoryMode::Other(mode) if mode == "vqa" || mode == "interleave")
        })
        .map(|(mode, reference_count)| {
            let context = gen_core::standard_memory_behavior_context(
                contract,
                strategy,
                MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: spec.quantize,
                    component_precision_floors: &[],
                },
                MemoryBehaviorRoute {
                    mode,
                    reference_count,
                    use_pid: false,
                    has_phases: false,
                    overlay: None,
                },
            )?;
            let mut fixture = MemoryBehaviorFixture::new(context);
            fixture.request.prompt = "weights-free SenseNova memory behavior".into();
            Ok(fixture)
        })
        .collect()
}

/// Convert the request's selected generation-memory block into the exact attention score plan.
pub(crate) fn request_attention_budget(request: &GenerationRequest) -> u64 {
    request
        .memory
        .filter(|memory| memory.chunk_attention)
        .and_then(|memory| memory.attention_chunk_size)
        .map(u64::from)
        .unwrap_or(candle_gen::ATTN_SCORES_BUDGET as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::{DType, Tensor};
    use gen_core::{MemorySelection, MemoryStrategyParameters};
    use std::collections::HashMap;

    fn spec(shape: LoadShape) -> LoadSpec {
        LoadSpec::new(WeightsSource::Dir("C:\\models\\sensenova\\bf16".into()))
            .with_load_shape(shape)
    }

    fn write_tier_fixture(root: &Path, bits: Option<u8>) {
        let weight_dtype = if bits.is_some() {
            DType::U32
        } else {
            DType::BF16
        };
        write_tier_fixture_with_dtypes(root, bits, weight_dtype, DType::BF16);
    }

    fn write_tier_fixture_with_dtypes(
        root: &Path,
        bits: Option<u8>,
        weight_dtype: DType,
        affine_dtype: DType,
    ) {
        std::fs::create_dir_all(root).unwrap();
        let base = "language_model.model.layers.0.self_attn.k_proj";
        let device = Device::Cpu;
        let mut tensors = HashMap::new();
        match bits {
            None => {
                tensors.insert(
                    format!("{base}.weight"),
                    Tensor::zeros((2, 64), weight_dtype, &device).unwrap(),
                );
            }
            Some(bits @ (4 | 8)) => {
                let lanes = 64 * bits as usize / 32;
                tensors.insert(
                    format!("{base}.weight"),
                    Tensor::zeros((2, lanes), weight_dtype, &device).unwrap(),
                );
                tensors.insert(
                    format!("{base}.scales"),
                    Tensor::ones((2, 1), affine_dtype, &device).unwrap(),
                );
                tensors.insert(
                    format!("{base}.biases"),
                    Tensor::zeros((2, 1), affine_dtype, &device).unwrap(),
                );
            }
            Some(bits) => panic!("unsupported fixture bits {bits}"),
        }
        candle_gen::candle_core::safetensors::save(&tensors, root.join("model.safetensors"))
            .unwrap();
        let config = bits.map_or_else(
            || "{}".to_owned(),
            |bits| format!(r#"{{"quantization":{{"bits":{bits},"group_size":64}}}}"#),
        );
        std::fs::write(root.join("config.json"), config).unwrap();
    }

    #[test]
    fn contract_declares_only_structurally_real_rungs() {
        let eager =
            weights_free_contract(crate::MODEL_ID, &spec(LoadShape::EagerMaterialization)).unwrap();
        gen_core_testkit::check_memory_strategy_contract(&eager).unwrap();
        let support = |strategy| {
            &eager
                .strategies
                .iter()
                .find(|capability| capability.strategy == strategy)
                .unwrap()
                .support
        };
        assert!(matches!(
            support(MemoryStrategy::StagedResidency),
            MemoryStrategySupport::StructurallyNotApplicable { .. }
        ));
        assert!(matches!(
            support(MemoryStrategy::BoundedDecode),
            MemoryStrategySupport::StructurallyNotApplicable { .. }
        ));
        assert_eq!(
            support(MemoryStrategy::BoundedAttention),
            &MemoryStrategySupport::Implemented
        );
        assert_eq!(
            support(MemoryStrategy::BoundedTransformerResidency),
            &MemoryStrategySupport::Missing
        );
    }

    #[test]
    fn request_budget_is_selected_only_when_attention_is_engaged() {
        let mut request = GenerationRequest {
            prompt: "x".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        assert_eq!(
            request_attention_budget(&request),
            candle_gen::ATTN_SCORES_BUDGET as u64
        );
        request.memory = Some(gen_core::GenerationMemory {
            chunk_attention: true,
            attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
            ..Default::default()
        });
        assert_eq!(
            request_attention_budget(&request),
            ATTENTION_CHUNK_SIZE as u64
        );
    }

    #[test]
    fn synthetic_registry_identity_never_becomes_production_cuda_evidence() {
        let tmp = tempfile::tempdir().unwrap();
        write_tier_fixture(tmp.path(), None);
        let spec = LoadSpec::new(WeightsSource::Dir(tmp.path().to_path_buf()));
        assert!(weights_free_contract(crate::MODEL_ID, &spec)
            .unwrap()
            .calibration
            .is_some());
        assert!(provider_contract(crate::MODEL_ID, &spec)
            .unwrap()
            .calibration
            .is_none());
        assert!(uncalibrated_contract(crate::MODEL_ID, &spec)
            .unwrap()
            .calibration
            .is_none());
    }

    #[test]
    fn pinned_inventory_rejects_replacement_and_shard_set_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let model = tmp.path().join("model.safetensors");
        std::fs::write(&model, [1_u8; 8]).unwrap();
        std::fs::write(tmp.path().join("config.json"), "{}").unwrap();
        let inventory = CheckpointInventory::capture(tmp.path()).unwrap();
        let replacement = tmp.path().join("replacement.safetensors.tmp");
        std::fs::write(&replacement, [2_u8; 8]).unwrap();
        std::fs::rename(&replacement, &model).unwrap();
        assert!(inventory.ensure_unchanged().is_err());

        let inventory = CheckpointInventory::capture(tmp.path()).unwrap();
        let config_replacement = tmp.path().join("config.json.tmp");
        std::fs::write(&config_replacement, r#"{"crossed":true}"#).unwrap();
        std::fs::rename(&config_replacement, tmp.path().join("config.json")).unwrap();
        assert!(inventory.ensure_unchanged().is_err());

        let inventory = CheckpointInventory::capture(tmp.path()).unwrap();
        std::fs::write(
            tmp.path().join("model-00002-of-00002.safetensors"),
            [3_u8; 8],
        )
        .unwrap();
        assert!(inventory.ensure_unchanged().is_err());
    }

    #[test]
    fn declared_numeric_tier_must_match_turnkey_provenance() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("tier");
        let base = LoadSpec::new(WeightsSource::Dir(root.clone()));

        write_tier_fixture(&root, None);
        assert!(validate_artifact_tier(&base).is_ok());
        assert!(validate_artifact_tier(&base.clone().with_quant(Quant::Q4)).is_err());

        write_tier_fixture(&root, Some(4));
        assert!(validate_artifact_tier(&base.clone().with_quant(Quant::Q4)).is_ok());
        assert!(validate_artifact_tier(&base.clone().with_quant(Quant::Q8)).is_err());
        assert!(validate_artifact_tier(&base).is_err());

        write_tier_fixture(&root, Some(8));
        assert!(validate_artifact_tier(&base.clone().with_quant(Quant::Q8)).is_ok());

        // Metadata alone can never authorize a packed tier over dense tensors.
        write_tier_fixture(&root, None);
        std::fs::write(
            root.join("config.json"),
            r#"{"quantization":{"bits":4,"group_size":64}}"#,
        )
        .unwrap();
        assert!(validate_artifact_tier(&base.clone().with_quant(Quant::Q4)).is_err());
    }

    #[test]
    fn numeric_tiers_reject_crossed_tensor_dtypes_before_loading() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("tier");
        let dense = LoadSpec::new(WeightsSource::Dir(root.clone()));
        let q4 = dense.clone().with_quant(Quant::Q4);

        for dtype in [DType::F16, DType::F32] {
            write_tier_fixture_with_dtypes(&root, None, dtype, DType::BF16);
            assert!(validate_artifact_tier(&dense).is_err(), "dense {dtype:?}");
        }

        write_tier_fixture_with_dtypes(&root, Some(4), DType::BF16, DType::BF16);
        assert!(
            validate_artifact_tier(&q4).is_err(),
            "packed-shape BF16 weights must not masquerade as U32 codes"
        );

        write_tier_fixture_with_dtypes(&root, Some(4), DType::U32, DType::F32);
        assert!(
            validate_artifact_tier(&q4).is_err(),
            "packed affine metadata must retain exact BF16 dtype"
        );
    }

    #[test]
    fn exact_six_routes_accept_bf16_q4_q8_and_reject_cross_provider_aliases() {
        for (provider, routes) in [
            (crate::MODEL_ID, QUALITY_ROUTES),
            (crate::MODEL_ID_FAST, FAST_ROUTES),
        ] {
            for route in routes {
                for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
                    let mut exact =
                        spec(LoadShape::EagerMaterialization).with_resolved_route(*route);
                    exact.quantize = quant;
                    let contract = weights_free_contract(provider, &exact)
                        .unwrap_or_else(|error| panic!("{provider}/{route}/{quant:?}: {error}"));
                    assert_eq!(contract.provider_id, provider);
                    assert_eq!(exact.quantize, quant);
                }
            }
        }

        let quality_as_fast =
            spec(LoadShape::EagerMaterialization).with_resolved_route(QUALITY_ROUTES[0]);
        assert!(weights_free_contract(crate::MODEL_ID_FAST, &quality_as_fast).is_err());
        let fast_as_quality =
            spec(LoadShape::EagerMaterialization).with_resolved_route(FAST_ROUTES[0]);
        assert!(weights_free_contract(crate::MODEL_ID, &fast_as_quality).is_err());
    }

    #[test]
    fn each_public_route_is_bound_to_its_repository_artifact() {
        let tmp = tempfile::tempdir().unwrap();
        for route in QUALITY_ROUTES.iter().chain(FAST_ROUTES) {
            let repository = expected_repository(route).unwrap();
            let exact = LoadSpec::new(WeightsSource::Dir(
                tmp.path().join(repository).join("snapshots/revision/q8"),
            ))
            .with_resolved_route(*route);
            assert!(
                validate_resolved_artifact_binding(&exact).is_ok(),
                "{route}"
            );

            let crossed_route = QUALITY_ROUTES
                .iter()
                .chain(FAST_ROUTES)
                .find(|candidate| *candidate != route)
                .unwrap();
            assert!(validate_resolved_artifact_binding(
                &exact.clone().with_resolved_route(*crossed_route)
            )
            .is_err());
        }
    }

    #[test]
    fn direct_execution_requires_the_exact_admitted_mode_and_geometry() {
        let contract =
            weights_free_contract(crate::MODEL_ID, &spec(LoadShape::EagerMaterialization)).unwrap();
        let context = gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::BoundedAttention,
            MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
            MemoryBehaviorRoute {
                mode: MemoryMode::Other("vqa".into()),
                reference_count: 1,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .unwrap();
        assert!(validate_direct_operation_identity(
            crate::MODEL_ID,
            &context,
            &context.mode,
            context.geometry,
        )
        .is_ok());
        assert!(validate_direct_operation_identity(
            crate::MODEL_ID,
            &context,
            &MemoryMode::Other("interleave".into()),
            context.geometry,
        )
        .is_err());
        let mut crossed_geometry = context.geometry;
        crossed_geometry.reference_count = 2;
        assert!(validate_direct_operation_identity(
            crate::MODEL_ID,
            &context,
            &context.mode,
            crossed_geometry,
        )
        .is_err());
    }

    #[test]
    fn advertised_mode_census_keeps_direct_modes_quality_only() {
        for provider_id in [crate::MODEL_ID, crate::MODEL_ID_FAST] {
            let contract =
                weights_free_contract(provider_id, &spec(LoadShape::EagerMaterialization)).unwrap();
            for (mode, reference_count, supported) in [
                (MemoryMode::TextToImage, 0, true),
                (MemoryMode::Edit, 1, true),
                (MemoryMode::Other("character_image".into()), 5, true),
                (
                    MemoryMode::Other("vqa".into()),
                    1,
                    provider_id == crate::MODEL_ID,
                ),
                (
                    MemoryMode::Other("interleave".into()),
                    0,
                    provider_id == crate::MODEL_ID,
                ),
            ] {
                let interleave = matches!(&mode, MemoryMode::Other(name) if name == "interleave");
                let mut context = gen_core::standard_memory_behavior_context(
                    &contract,
                    MemoryStrategy::BoundedAttention,
                    MemoryNumericTier {
                        precision: Precision::Bf16,
                        quant: None,
                        component_precision_floors: &[],
                    },
                    MemoryBehaviorRoute {
                        mode,
                        reference_count,
                        use_pid: false,
                        has_phases: false,
                        overlay: None,
                    },
                )
                .unwrap();
                if interleave {
                    context.geometry.batch = 10;
                }
                assert_eq!(
                    validate_context(provider_id, &contract, &context, None).is_ok(),
                    supported,
                    "{provider_id} {}/{}",
                    context.mode.as_key(),
                    reference_count
                );
            }
        }
    }

    #[test]
    fn interleave_admission_binds_one_through_ten_generated_images() {
        let contract =
            weights_free_contract(crate::MODEL_ID, &spec(LoadShape::EagerMaterialization)).unwrap();
        for count in 1..=10 {
            let mut context = gen_core::standard_memory_behavior_context(
                &contract,
                MemoryStrategy::BoundedAttention,
                MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[],
                },
                MemoryBehaviorRoute {
                    mode: MemoryMode::Other("interleave".into()),
                    reference_count: 0,
                    use_pid: false,
                    has_phases: false,
                    overlay: None,
                },
            )
            .unwrap();
            context.geometry.batch = count;
            assert!(validate_context(crate::MODEL_ID, &contract, &context, None).is_ok());
        }
        for count in [0, 11] {
            let mut context = gen_core::standard_memory_behavior_context(
                &contract,
                MemoryStrategy::BoundedAttention,
                MemoryNumericTier {
                    precision: Precision::Bf16,
                    quant: None,
                    component_precision_floors: &[],
                },
                MemoryBehaviorRoute {
                    mode: MemoryMode::Other("interleave".into()),
                    reference_count: 0,
                    use_pid: false,
                    has_phases: false,
                    overlay: None,
                },
            )
            .unwrap();
            context.geometry.batch = count;
            assert!(validate_context(crate::MODEL_ID, &contract, &context, None).is_err());
        }
    }

    #[test]
    fn an_existing_but_incomplete_snapshot_fails_contract_construction() {
        let root = tempfile::tempdir().unwrap();
        let exact = LoadSpec::new(WeightsSource::Dir(root.path().to_path_buf()))
            .with_resolved_route(QUALITY_ROUTES[0]);
        assert!(provider_contract(crate::MODEL_ID, &exact).is_err());
    }

    #[test]
    fn vqa_refuses_generation_only_transformer_window() {
        let contract =
            weights_free_contract(crate::MODEL_ID, &spec(LoadShape::DeferredMaterialization))
                .unwrap();
        let selection = MemorySelection {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            parameters: MemoryStrategyParameters {
                attention_chunk_size: Some(ATTENTION_CHUNK_SIZE),
                transformer_window_size: Some(TRANSFORMER_WINDOW_SIZE),
                transformer_window_component: Some(TransformerComponent::Dit),
                ..Default::default()
            },
            tier: MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
        };
        let mut context = gen_core::standard_memory_behavior_context(
            &contract,
            MemoryStrategy::BoundedTransformerResidency,
            MemoryNumericTier {
                precision: Precision::Bf16,
                quant: None,
                component_precision_floors: &[],
            },
            MemoryBehaviorRoute {
                mode: MemoryMode::Other("vqa".into()),
                reference_count: 1,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .unwrap();
        context.selection = selection;
        assert!(matches!(
            safety_check(crate::MODEL_ID, &contract, &context, None),
            MemorySafetyDecision::Reject { .. }
        ));
    }
}
