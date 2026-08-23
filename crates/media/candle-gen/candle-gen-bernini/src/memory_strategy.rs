//! Request-scoped Candle/CUDA memory contract for the public Bernini still route.
//!
//! Bernini keeps its planner, both A14B experts, and z16 VAE resident. The only optimized rung the
//! still path can truthfully execute is a caller-selected maximum z16 decode tile. Component
//! phases, global attention knobs, and eager cached experts are deliberately not relabeled as
//! staged/attention/transformer-residency rungs.

use std::io::Read;
use std::path::{Path, PathBuf};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, AdapterKind, GenerationRequest, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryBehaviorFixture, MemoryBehaviorRoute, MemoryCalibrationIdentity, MemoryFormulaKind,
    MemoryFormulaVariable, MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier,
    MemoryParameterRanges, MemoryPhase, MemoryProviderContract, MemoryRequestScope,
    MemoryRunContext, MemorySafetyDecision, MemoryStrategy, MemoryStrategyCapability,
    MemoryStrategySupport, MemoryWindowMaterialization, Precision, Quant, ResidentRequestMemory,
    WeightsSource,
};
use sha2::{Digest, Sha256};

pub const PROVIDER_ID: &str = "bernini";
pub const CANONICAL_REPOSITORY: &str = "SceneWorks/bernini";
pub const CANONICAL_REVISION: &str = "f9f95e0ff3d7940664ab8163ebf71bf0c8018b27";
pub const CALIBRATION_FINGERPRINT: &str =
    "bernini-candle-still-resident-z16-bounded-decode-static-v1";

/// CUDA z16 decode tile caps in output pixels, intersected with the live free-VRAM plan.
///
/// This domain is deliberately backend-specific. Candle's conv2d im2col path has a hard 512px
/// safety ceiling (`WAN_Z16_VAE_IM2COL_SAFE_PX` in `candle-gen-wan`), so the Candle ladder cannot
/// publish the MLX provider's 768/640 Metal candidates. Candle also retains 448 and 192 because
/// they are valid members of the CUDA z16 production tiler; the shared 64px overlap still makes
/// every edge advance by at least one whole VAE latent cell.
pub const DECODE_TILE_EDGES: &[u32] = &[512, 448, 384, 320, 256, 192];
pub const DECODE_OVERLAP: u32 = 64;
const PUBLIC_GEOMETRIES: &[(u32, u32)] = &[
    (512, 512),
    (768, 768),
    (1024, 1024),
    (1280, 720),
    (720, 1280),
];

#[derive(Clone, Debug)]
pub(crate) struct AdapterReceipt {
    pub(crate) ordinal: usize,
    pin: gen_core::PinnedWeightsFile,
    pub(crate) canonical_path: PathBuf,
    pub(crate) sha256: String,
    pub(crate) kind: AdapterKind,
    pub(crate) scale_bits: u32,
    pub(crate) realized_bytes_per_expert: u64,
}

impl AdapterReceipt {
    fn capture(
        spec: &LoadSpec,
        ordinal: usize,
        adapter: &gen_core::AdapterSpec,
    ) -> gen_core::Result<Self> {
        if adapter.pass_scales.is_some() || adapter.moe_expert.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: adapter {ordinal} must be an untargeted, full-run LoRA/LoKr"
            )));
        }
        if !adapter.scale.is_finite() {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: adapter {ordinal} scale must be finite"
            )));
        }
        let adapter_path = std::path::absolute(&adapter.path)?;
        let pin = if spec.prepared_file_pins().is_prepared() {
            spec.prepared_file_pins()
                .get(&adapter_path)
                .cloned()
                .ok_or_else(|| {
                    gen_core::Error::Unsupported(format!(
                        "{PROVIDER_ID}: sealed receipt is missing adapter {ordinal} at {}",
                        adapter_path.display()
                    ))
                })?
        } else {
            gen_core::PinnedWeightsFile::pin(&adapter_path)?
        };
        pin.ensure_unchanged()?;
        let canonical_path = pin.canonical_target_path().to_path_buf();
        let (sha256, realized_bytes_per_expert) = pin.read_unchanged(|path| {
            let mut file = std::fs::File::open(path)?;
            let mut hasher = Sha256::new();
            let mut buffer = [0_u8; 1024 * 1024];
            loop {
                let read = file.read(&mut buffer)?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
            }
            let bytes = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?
                .into_iter()
                .try_fold(0_u64, |total, tensor| {
                    total.checked_add(tensor.data_bytes).ok_or_else(|| {
                        gen_core::Error::Msg(format!(
                            "{PROVIDER_ID}: adapter {ordinal} resident-byte sum overflow"
                        ))
                    })
                })?;
            Ok::<_, gen_core::Error>((format!("{:x}", hasher.finalize()), bytes))
        })?;
        if realized_bytes_per_expert == 0 {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: adapter {ordinal} contains no realized tensor bytes"
            )));
        }
        Ok(Self {
            ordinal,
            pin,
            canonical_path,
            sha256,
            kind: adapter.kind,
            scale_bits: adapter.scale.to_bits(),
            realized_bytes_per_expert,
        })
    }

    fn ensure_unchanged(&self) -> gen_core::Result<()> {
        self.pin.ensure_unchanged()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ArtifactReceipt {
    root: PathBuf,
    pins: Vec<gen_core::PinnedWeightsFile>,
    pub(crate) canonical: bool,
    pub(crate) tier: Option<Quant>,
    pub(crate) facts: MemoryAssetFacts,
}

impl ArtifactReceipt {
    fn capture(spec: &LoadSpec, adapters: &[AdapterReceipt]) -> gen_core::Result<Self> {
        let WeightsSource::Dir(root) = &spec.weights else {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: requires a tier snapshot directory"
            )));
        };
        let lexical_root = std::path::absolute(root)?;
        let root = std::fs::canonicalize(root)?;
        let tier = detect_expert_tier(&root.join("transformer"))?;
        let second = detect_expert_tier(&root.join("transformer_2"))?;
        if tier != second {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: high/low experts cross numeric tiers ({tier:?} versus {second:?})"
            )));
        }
        if tier != spec.quantize {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: requested tier {:?} does not match packed tensor tier {tier:?}",
                spec.quantize
            )));
        }
        let canonical = canonical_artifact_path(&root, tier)
            && spec.resolved_route.as_deref() == Some(PROVIDER_ID);

        let conditioning_bytes = checked_sum(
            "conditioning/planner",
            [
                projected_component_bytes(&root.join("text_encoder"), 4)?,
                projected_component_bytes(&root.join("mllm"), 2)?,
                projected_component_bytes(&root.join("connector"), 2)?,
                projected_component_bytes(&root.join("vit_decoder"), 2)?,
                projected_file_bytes(&root.join("mask_tokens.safetensors"), 2)?,
            ],
        )?;
        let transformer_bytes = checked_sum(
            "dual experts",
            [
                projected_component_bytes(&root.join("transformer"), 2)?,
                projected_component_bytes(&root.join("transformer_2"), 2)?,
            ],
        )?;
        let decoder_bytes = projected_component_bytes(&root.join("vae"), 4)?;
        let overlay_bytes = if tier.is_some() {
            adapters.iter().try_fold(0_u64, |total, receipt| {
                total
                    .checked_add(receipt.realized_bytes_per_expert.saturating_mul(2))
                    .ok_or_else(|| {
                        gen_core::Error::Msg(format!(
                            "{PROVIDER_ID}: dual-expert adapter byte sum overflow"
                        ))
                    })
            })?
        } else {
            0
        };
        let base_bytes = checked_sum(
            "base model",
            [conditioning_bytes, transformer_bytes, decoder_bytes],
        )?;
        let files = recursive_files(&root)?;
        let pins = if spec.prepared_file_pins().is_prepared() {
            let mut sealed = spec
                .prepared_file_pins()
                .iter()
                .filter(|(_, pin)| pin.loader_path().starts_with(&lexical_root))
                .map(|(_, pin)| pin.clone())
                .collect::<Vec<_>>();
            sealed.sort_by(|left, right| {
                left.canonical_target_path()
                    .cmp(right.canonical_target_path())
            });
            sealed.dedup_by(|left, right| {
                left.canonical_target_path() == right.canonical_target_path()
            });
            let sealed_files = sealed
                .iter()
                .map(|pin| pin.canonical_target_path().to_path_buf())
                .collect::<Vec<_>>();
            if sealed_files != files {
                return Err(gen_core::Error::Unsupported(format!(
                    "{PROVIDER_ID}: sealed base receipt does not match the complete artifact inventory"
                )));
            }
            sealed
        } else {
            files
                .iter()
                .map(gen_core::PinnedWeightsFile::pin)
                .collect::<gen_core::Result<Vec<_>>>()?
        };
        let receipt = Self {
            root,
            pins,
            canonical,
            tier,
            facts: MemoryAssetFacts {
                base_bytes,
                conditioning_bytes,
                transformer_bytes,
                decoder_bytes,
                overlay_bytes,
            },
        };
        receipt.ensure_unchanged()?;
        Ok(receipt)
    }

    pub(crate) fn ensure_unchanged(&self) -> gen_core::Result<()> {
        let current = recursive_files(&self.root)?;
        let expected = self
            .pins
            .iter()
            .map(|pin| pin.canonical_target_path().to_path_buf())
            .collect::<Vec<_>>();
        if current != expected {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: artifact inventory changed after load"
            )));
        }
        for pin in &self.pins {
            pin.ensure_unchanged()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedMemory {
    pub(crate) contract: MemoryProviderContract,
    pub(crate) tier: MemoryNumericTier,
    pub(crate) artifact: ArtifactReceipt,
    pub(crate) adapters: Vec<AdapterReceipt>,
}

impl PreparedMemory {
    pub(crate) fn prepare(spec: &LoadSpec) -> gen_core::Result<Self> {
        validate_load_spec(spec)?;
        spec.validate_prepared_file_pins()?;
        let adapters = spec
            .adapters
            .iter()
            .enumerate()
            .map(|(index, adapter)| AdapterReceipt::capture(spec, index, adapter))
            .collect::<gen_core::Result<Vec<_>>>()?;
        let artifact = ArtifactReceipt::capture(spec, &adapters)?;
        let tier = numeric_tier(artifact.tier);
        let contract = build_contract(spec, artifact.canonical, artifact.facts);
        Ok(Self {
            contract,
            tier,
            artifact,
            adapters,
        })
    }

    pub(crate) fn ensure_unchanged(
        &self,
        adapters: &[gen_core::AdapterSpec],
    ) -> gen_core::Result<()> {
        self.artifact.ensure_unchanged()?;
        if adapters.len() != self.adapters.len() {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: adapter stack changed after its load receipt was sealed"
            )));
        }
        for (index, receipt) in self.adapters.iter().enumerate() {
            let adapter = &adapters[index];
            if receipt.ordinal != index
                || std::fs::canonicalize(&adapter.path)? != receipt.canonical_path
                || adapter.kind != receipt.kind
                || adapter.scale.to_bits() != receipt.scale_bits
                || adapter.pass_scales.is_some()
                || adapter.moe_expert.is_some()
                || receipt.sha256.len() != 64
                || receipt.realized_bytes_per_expert == 0
            {
                return Err(gen_core::Error::Unsupported(format!(
                    "{PROVIDER_ID}: adapter {index} crossed its immutable load receipt"
                )));
            }
            receipt.ensure_unchanged()?;
        }
        Ok(())
    }
}

fn validate_load_spec(spec: &LoadSpec) -> gen_core::Result<()> {
    if !matches!(spec.weights, WeightsSource::Dir(_)) {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: requires directory weights"
        )));
    }
    if spec.precision != Precision::Bf16
        || !matches!(spec.quantize, None | Some(Quant::Q4 | Quant::Q8))
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: only bf16, q8, and q4 turnkey tiers are supported"
        )));
    }
    if spec.adapters.len() > 5
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: supports only an ordered stack of 0..=5 untargeted LoRA/LoKr adapters"
        )));
    }
    gen_core::reject_unknown_components(spec, &[], PROVIDER_ID)
}

fn canonical_artifact_path(root: &Path, tier: Option<Quant>) -> bool {
    let expected_tier = match tier {
        None => "bf16",
        Some(Quant::Q4) => "q4",
        Some(Quant::Q8) => "q8",
        Some(_) => return false,
    };
    root.file_name().and_then(|name| name.to_str()) == Some(expected_tier)
        && root.components().any(|part| {
            matches!(
                part.as_os_str().to_str(),
                Some("models--SceneWorks--bernini") | Some("SceneWorks__bernini")
            )
        })
        && root
            .components()
            .any(|part| part.as_os_str().to_str() == Some(CANONICAL_REVISION))
}

fn recursive_files(root: &Path) -> gen_core::Result<Vec<PathBuf>> {
    fn visit(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
        let mut entries = std::fs::read_dir(dir)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                visit(&path, out)?;
            } else if entry.file_type()?.is_file() || entry.file_type()?.is_symlink() {
                out.push(std::fs::canonicalize(path)?);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    visit(root, &mut files)?;
    files.sort();
    files.dedup();
    if files.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: empty artifact directory {}",
            root.display()
        )));
    }
    Ok(files)
}

fn projected_component_bytes(path: &Path, float_width: u64) -> gen_core::Result<u64> {
    let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
    projected_headers_bytes(&headers, float_width, path)
}

fn projected_file_bytes(path: &Path, float_width: u64) -> gen_core::Result<u64> {
    let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
    projected_headers_bytes(&headers, float_width, path)
}

fn projected_headers_bytes(
    headers: &[gen_core::weightsmeta::SafetensorsTensorHeader],
    float_width: u64,
    path: &Path,
) -> gen_core::Result<u64> {
    use gen_core::weightsmeta::Dtype;
    if headers.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: {} contains no tensors",
            path.display()
        )));
    }
    headers.iter().try_fold(0_u64, |total, tensor| {
        let bytes = match tensor.dtype {
            Dtype::U8 | Dtype::U32 | Dtype::I16 | Dtype::I32 | Dtype::I64 => tensor.data_bytes,
            Dtype::U16 => tensor.materialized_bytes(4)?,
            Dtype::F8_E4M3 | Dtype::F16 | Dtype::BF16 | Dtype::F32 | Dtype::F64 => {
                tensor.materialized_bytes(float_width)?
            }
            dtype => {
                return Err(gen_core::Error::Unsupported(format!(
                    "{PROVIDER_ID}: tensor {:?} has unsupported resident dtype {dtype:?}",
                    tensor.name
                )))
            }
        };
        total.checked_add(bytes).ok_or_else(|| {
            gen_core::Error::Msg(format!("{PROVIDER_ID}: component byte projection overflow"))
        })
    })
}

fn detect_expert_tier(path: &Path) -> gen_core::Result<Option<Quant>> {
    use gen_core::weightsmeta::Dtype;
    let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
    let names = headers
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    let weight = headers
        .iter()
        .find(|tensor| tensor.name.ends_with("proj_out.weight"))
        .or_else(|| {
            headers.iter().find(|tensor| {
                tensor.name.ends_with(".weight")
                    && names.contains(
                        tensor
                            .name
                            .strip_suffix(".weight")
                            .map(|base| format!("{base}.scales"))
                            .as_deref()
                            .unwrap_or_default(),
                    )
            })
        })
        .ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: {} has no recognizable expert projection",
                path.display()
            ))
        })?;
    let base = weight.name.strip_suffix(".weight").expect("matched suffix");
    let scales = headers
        .iter()
        .find(|tensor| tensor.name == format!("{base}.scales"));
    let biases = headers
        .iter()
        .find(|tensor| tensor.name == format!("{base}.biases"));
    match (scales, biases) {
        (None, None) if weight.dtype == Dtype::BF16 => Ok(None),
        (Some(scales), Some(biases))
            if weight.dtype == Dtype::U32
                && scales.dtype == Dtype::BF16
                && biases.dtype == Dtype::BF16 =>
        {
            let [out, lanes] = weight.shape.as_slice() else {
                return Err(gen_core::Error::Unsupported(format!(
                    "{PROVIDER_ID}: packed expert projection is not rank two"
                )));
            };
            let [scale_out, groups] = scales.shape.as_slice() else {
                return Err(gen_core::Error::Unsupported(format!(
                    "{PROVIDER_ID}: packed expert scales are not rank two"
                )));
            };
            if out != scale_out || biases.shape != scales.shape || *groups == 0 {
                return Err(gen_core::Error::Unsupported(format!(
                    "{PROVIDER_ID}: packed expert projection shapes disagree"
                )));
            }
            let numerator = lanes.saturating_mul(32);
            let denominator = groups.saturating_mul(64);
            if denominator == 0 || !numerator.is_multiple_of(denominator) {
                return Err(gen_core::Error::Unsupported(format!(
                    "{PROVIDER_ID}: packed expert projection has no exact group64 bit width"
                )));
            }
            match numerator / denominator {
                4 => Ok(Some(Quant::Q4)),
                8 => Ok(Some(Quant::Q8)),
                bits => Err(gen_core::Error::Unsupported(format!(
                    "{PROVIDER_ID}: packed expert projection resolves unsupported {bits}-bit codes"
                ))),
            }
        }
        _ => Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: expert projection mixes dense and affine-packed tensor state"
        ))),
    }
}

fn checked_sum(label: &str, values: impl IntoIterator<Item = u64>) -> gen_core::Result<u64> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total.checked_add(value).ok_or_else(|| {
            gen_core::Error::Msg(format!("{PROVIDER_ID}: {label} byte sum overflow"))
        })
    })
}

fn numeric_tier(quant: Option<Quant>) -> MemoryNumericTier {
    MemoryNumericTier {
        precision: Precision::Bf16,
        quant,
        component_precision_floors: &[],
    }
}

fn build_contract(
    spec: &LoadSpec,
    canonical: bool,
    facts: MemoryAssetFacts,
) -> MemoryProviderContract {
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    MemoryProviderContract {
        provider_id: PROVIDER_ID.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: false,
            host_to_device_block_materialization: false,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies: MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: match strategy {
                    MemoryStrategy::Resident => MemoryStrategySupport::Implemented,
                    MemoryStrategy::BoundedDecode if canonical => {
                        MemoryStrategySupport::Implemented
                    }
                    _ => MemoryStrategySupport::Missing,
                },
                parameters: if strategy == MemoryStrategy::BoundedDecode && canonical {
                    MemoryParameterRanges {
                        decode_tile_edges: DECODE_TILE_EDGES.to_vec(),
                        decode_overlaps: vec![DECODE_OVERLAP],
                        ..Default::default()
                    }
                } else {
                    MemoryParameterRanges::default()
                },
            })
            .collect(),
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: ResidentRequestMemory::PreserveLoadDefaults,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: false,
            decode_tiling: canonical,
            attention_chunking: false,
            transformer_window_materialization: false,
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
            ],
        },
        calibration: canonical
            .then(|| MemoryCalibrationIdentity::new(CALIBRATION_FINGERPRINT, spec.load_shape)),
        asset_facts: facts,
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    }
}

pub fn provider_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    PreparedMemory::prepare(spec).map(|prepared| prepared.contract)
}

pub fn weights_free_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    validate_load_spec(spec)?;
    Ok(build_contract(spec, true, MemoryAssetFacts::default()))
}

fn validate_route(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    let mode_refs = match &context.mode {
        MemoryMode::TextToImage => 0,
        MemoryMode::Edit => 1,
        mode => {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: memory mode {mode:?} is not public T2I/edit"
            )))
        }
    };
    let geometry = context.geometry;
    if geometry.batch != 1
        || geometry.frames != 1
        || geometry.reference_count != mode_refs
        || context.has_reference != (mode_refs == 1)
        || !PUBLIC_GEOMETRIES.contains(&(geometry.width, geometry.height))
        || context.use_pid
        || context.has_phases
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: memory route requires one public 1-frame T2I/edit still at an advertised geometry"
        )));
    }
    if context
        .overlay
        .as_deref()
        .is_some_and(|overlay| overlay != "lora")
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: unsupported memory overlay {:?}",
            context.overlay
        )));
    }
    if context.selection.strategy == MemoryStrategy::BoundedDecode
        && (!DECODE_TILE_EDGES.contains(
            &context
                .selection
                .parameters
                .decode_tile_edge
                .unwrap_or_default(),
        ) || context.selection.parameters.decode_overlap != Some(DECODE_OVERLAP))
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: bounded decode requires a published z16 tile edge and overlap {DECODE_OVERLAP}"
        )));
    }
    match gen_core::standard_memory_strategy_safety_check(contract, context, Some(tier), None) {
        MemorySafetyDecision::Accept => Ok(()),
        MemorySafetyDecision::Reject { reason } => Err(gen_core::Error::Unsupported(reason)),
    }
}

pub(crate) fn safety_check(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match validate_route(contract, tier, context) {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

pub(crate) fn validate_generation_request(
    request: &GenerationRequest,
    has_adapters: bool,
) -> gen_core::Result<()> {
    let refs = request.image_reference_count();
    let expected = match request.video_mode.as_deref() {
        Some("t2i") if refs == 0 => 0,
        Some("i2i") if refs == 1 => 1,
        mode => {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: still memory route crossed task/reference identity {mode:?}/{refs}"
            )))
        }
    };
    if request.frames != Some(1)
        || request.count != 1
        || request.phases.is_some()
        || request.use_pid
        || !PUBLIC_GEOMETRIES.contains(&(request.width, request.height))
        || refs != expected
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: request left the admitted single-still T2I/edit envelope"
        )));
    }
    if let Some(memory) = request.memory {
        if memory.stage_residency
            || memory.chunk_attention
            || memory.stream_transformer_blocks
            || (memory.tile_vae_decode
                && (!DECODE_TILE_EDGES.contains(&memory.decode_tile_edge.unwrap_or_default())
                    || memory.decode_overlap != Some(DECODE_OVERLAP)))
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: request carries an unsupported memory mechanism"
            )));
        }
    }
    let _ = has_adapters;
    Ok(())
}

pub(crate) fn selected_decode_cap(request: &GenerationRequest) -> gen_core::Result<Option<u32>> {
    let Some(memory) = request.memory else {
        return Ok(None);
    };
    if !memory.tile_vae_decode {
        if memory.decode_tile_edge.is_some() || memory.decode_overlap.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: decode parameters supplied without bounded decode"
            )));
        }
        return Ok(None);
    }
    let edge = memory.decode_tile_edge.ok_or_else(|| {
        gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: bounded decode omitted its tile edge"
        ))
    })?;
    if !DECODE_TILE_EDGES.contains(&edge) || memory.decode_overlap != Some(DECODE_OVERLAP) {
        return Err(gen_core::Error::Unsupported(format!(
            "{PROVIDER_ID}: invalid bounded z16 decode parameters"
        )));
    }
    Ok(Some(edge))
}

fn begin_with_device(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    device: Device,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    validate_route(contract, tier, context)?;
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        PROVIDER_ID,
        device,
        context.geometry,
        contract.generation_memory(&context.selection),
        false,
        0,
        |_pid, edge, overlap| {
            if DECODE_TILE_EDGES.contains(&edge) && overlap == DECODE_OVERLAP {
                Ok(())
            } else {
                Err(gen_core::Error::Unsupported(format!(
                    "{PROVIDER_ID}: decode hook crossed the selected z16 tile domain"
                )))
            }
        },
    )?;
    config.default_frames = 1;
    Ok(Some(Box::new(
        candle_gen::request_scope::CandleRequestScopeCore::new(config),
    )))
}

pub(crate) fn begin_request(
    contract: &MemoryProviderContract,
    tier: MemoryNumericTier,
    device: Device,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    begin_with_device(contract, tier, device, context)
}

fn registered_safety_check(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match registered_numeric_tier(spec, contract) {
        Ok(tier) => safety_check(contract, tier, context),
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

/// Resolve the numeric tier for registry behavior without weakening production artifact checks.
///
/// A real contract always carries non-zero measured artifact facts and must reconstruct its exact
/// immutable receipt. The registry-only weights-free contract carries zero facts, so its positive
/// control instead presents the canonical lexical route witness built below; no model files are read.
fn registered_numeric_tier(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
) -> gen_core::Result<MemoryNumericTier> {
    if contract.asset_facts == MemoryAssetFacts::default() {
        validate_load_spec(spec)?;
        let WeightsSource::Dir(root) = &spec.weights else {
            unreachable!("validated directory source")
        };
        if spec.resolved_route.as_deref() != Some(PROVIDER_ID)
            || !canonical_artifact_path(root, spec.quantize)
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: weights-free behavior requires the canonical repository/revision/tier witness"
            )));
        }
        return Ok(numeric_tier(spec.quantize));
    }
    PreparedMemory::prepare(spec).map(|prepared| prepared.tier)
}

fn weights_free_behavior_spec(spec: &LoadSpec) -> gen_core::Result<LoadSpec> {
    validate_load_spec(spec)?;
    let tier = match spec.quantize {
        None => "bf16",
        Some(Quant::Q4) => "q4",
        Some(Quant::Q8) => "q8",
        Some(other) => {
            return Err(gen_core::Error::Unsupported(format!(
                "{PROVIDER_ID}: unsupported weights-free behavior tier {other:?}"
            )))
        }
    };
    let mut exact = spec.clone();
    exact.weights = WeightsSource::Dir(
        PathBuf::from("models--SceneWorks--bernini")
            .join("snapshots")
            .join(CANONICAL_REVISION)
            .join(tier),
    );
    exact.resolved_route = Some(PROVIDER_ID.to_owned());
    Ok(exact)
}

fn registered_valid_fixtures(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<MemoryBehaviorFixture>> {
    if strategy != MemoryStrategy::BoundedDecode {
        return Ok(Vec::new());
    }
    let context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        numeric_tier(spec.quantize),
        MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )?;
    Ok(vec![
        MemoryBehaviorFixture::new(context).with_load_spec(weights_free_behavior_spec(spec)?)
    ])
}

fn registered_begin_request(
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let tier = registered_numeric_tier(spec, contract)?;
    begin_with_device(contract, tier, Device::Cpu, context)
}

pub const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: PROVIDER_ID,
    contract: provider_contract,
    safety_check: registered_safety_check,
};

pub const MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: PROVIDER_ID,
        valid_fixtures: registered_valid_fixtures,
        begin_request: registered_begin_request,
    };

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_gen::candle_core::{
        safetensors as candle_safetensors, DType as CandleDType, Tensor,
    };
    use candle_gen::gen_core::{
        AdapterSpec, Conditioning, GenerationMemory, Image, MemoryBehaviorRoute,
        MemoryStrategyParameters,
    };

    use super::*;

    fn write_shard(path: &Path, tensors: impl IntoIterator<Item = (&'static str, Tensor)>) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let tensors = tensors
            .into_iter()
            .map(|(name, tensor)| (name.to_owned(), tensor))
            .collect::<HashMap<_, _>>();
        candle_safetensors::save(&tensors, path).unwrap();
    }

    fn component(root: &Path, name: &str, dtype: CandleDType) {
        write_shard(
            &root.join(name).join("model.safetensors"),
            [(
                "component.weight",
                Tensor::zeros((2, 3), dtype, &Device::Cpu).unwrap(),
            )],
        );
    }

    fn expert(root: &Path, name: &str, quant: Option<Quant>) {
        let path = root.join(name).join("model.safetensors");
        match quant {
            None => write_shard(
                &path,
                [(
                    "blocks.0.proj_out.weight",
                    Tensor::zeros((2, 64), CandleDType::BF16, &Device::Cpu).unwrap(),
                )],
            ),
            Some(quant @ (Quant::Q4 | Quant::Q8)) => {
                let lanes = if quant == Quant::Q4 { 8 } else { 16 };
                write_shard(
                    &path,
                    [
                        (
                            "blocks.0.proj_out.weight",
                            Tensor::zeros((2, lanes), CandleDType::U32, &Device::Cpu).unwrap(),
                        ),
                        (
                            "blocks.0.proj_out.scales",
                            Tensor::zeros((2, 1), CandleDType::BF16, &Device::Cpu).unwrap(),
                        ),
                        (
                            "blocks.0.proj_out.biases",
                            Tensor::zeros((2, 1), CandleDType::BF16, &Device::Cpu).unwrap(),
                        ),
                    ],
                );
            }
            Some(other) => panic!("unsupported test tier {other:?}"),
        }
    }

    struct Fixture {
        _temp: tempfile::TempDir,
        root: PathBuf,
    }

    fn fixture(quant: Option<Quant>, canonical: bool) -> Fixture {
        let temp = tempfile::tempdir().unwrap();
        let root = if canonical {
            temp.path()
                .join("models--SceneWorks--bernini")
                .join("snapshots")
                .join(CANONICAL_REVISION)
                .join(match quant {
                    None => "bf16",
                    Some(Quant::Q4) => "q4",
                    Some(Quant::Q8) => "q8",
                    Some(other) => panic!("unsupported test tier {other:?}"),
                })
        } else {
            temp.path().join("legacy")
        };
        component(&root, "text_encoder", CandleDType::F16);
        component(&root, "mllm", CandleDType::BF16);
        component(&root, "connector", CandleDType::BF16);
        component(&root, "vit_decoder", CandleDType::BF16);
        write_shard(
            &root.join("mask_tokens.safetensors"),
            [(
                "mask_tokens",
                Tensor::zeros((2, 3), CandleDType::BF16, &Device::Cpu).unwrap(),
            )],
        );
        expert(&root, "transformer", quant);
        expert(&root, "transformer_2", quant);
        component(&root, "vae", CandleDType::F16);
        Fixture { _temp: temp, root }
    }

    fn spec(fixture: &Fixture, quant: Option<Quant>) -> LoadSpec {
        let mut spec = LoadSpec::new(WeightsSource::Dir(fixture.root.clone()))
            .with_resolved_route(PROVIDER_ID);
        spec.quantize = quant;
        spec
    }

    fn adapter(path: &Path, value: f32, kind: AdapterKind) -> AdapterSpec {
        write_shard(
            path,
            [(
                "adapter.weight",
                Tensor::zeros((2, 3), CandleDType::BF16, &Device::Cpu).unwrap(),
            )],
        );
        AdapterSpec::new(path.to_path_buf(), value, kind)
    }

    fn still(mode: &str, width: u32, height: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "fixture".into(),
            width,
            height,
            count: 1,
            frames: Some(1),
            video_mode: Some(mode.into()),
            ..Default::default()
        }
    }

    #[test]
    fn weights_free_behavior_uses_canonical_witness_without_skipping_production_receipts() {
        for quant in [None, Some(Quant::Q4), Some(Quant::Q8)] {
            let mut common = LoadSpec::new(WeightsSource::Dir(PathBuf::from("weights-free")));
            common.quantize = quant;
            let contract = weights_free_contract(&common).unwrap();
            let exact = weights_free_behavior_spec(&common).unwrap();
            let WeightsSource::Dir(root) = &exact.weights else {
                panic!("weights-free fixture must use a directory witness")
            };
            assert_eq!(exact.resolved_route.as_deref(), Some(PROVIDER_ID));
            assert!(canonical_artifact_path(root, quant));
            assert_eq!(
                registered_numeric_tier(&exact, &contract).unwrap(),
                numeric_tier(quant)
            );
            assert!(registered_numeric_tier(&common, &contract).is_err());

            let mut crossed_route = exact;
            crossed_route.resolved_route = Some("crossed-provider".to_owned());
            assert!(registered_numeric_tier(&crossed_route, &contract).is_err());
        }

        let fixture = fixture(Some(Quant::Q4), true);
        let load = spec(&fixture, Some(Quant::Q4));
        let production_contract = provider_contract(&load).unwrap();
        let lexical_only = weights_free_behavior_spec(&load).unwrap();
        let WeightsSource::Dir(root) = &lexical_only.weights else {
            panic!("weights-free fixture must use a directory witness")
        };
        assert!(!root.exists());
        assert!(registered_numeric_tier(&lexical_only, &production_contract).is_err());
    }

    #[test]
    fn actual_tensor_packing_selects_tier_and_exact_projected_bytes() {
        for (quant, transformer_bytes) in
            [(None, 512), (Some(Quant::Q8), 272), (Some(Quant::Q4), 144)]
        {
            let fixture = fixture(quant, true);
            let prepared = PreparedMemory::prepare(&spec(&fixture, quant)).unwrap();
            assert_eq!(prepared.artifact.tier, quant);
            assert_eq!(prepared.artifact.facts.conditioning_bytes, 72);
            assert_eq!(prepared.artifact.facts.transformer_bytes, transformer_bytes);
            assert_eq!(prepared.artifact.facts.decoder_bytes, 24);
            assert_eq!(prepared.artifact.facts.base_bytes, 96 + transformer_bytes);
            assert!(matches!(
                prepared
                    .contract
                    .capability(MemoryStrategy::Resident)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Implemented
            ));
            assert!(matches!(
                prepared
                    .contract
                    .capability(MemoryStrategy::BoundedDecode)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Implemented
            ));
            for missing in [
                MemoryStrategy::StagedResidency,
                MemoryStrategy::BoundedAttention,
                MemoryStrategy::BoundedTransformerResidency,
            ] {
                assert!(matches!(
                    prepared.contract.capability(missing).unwrap().support,
                    MemoryStrategySupport::Missing
                ));
            }
        }
    }

    #[test]
    fn aliases_remain_resident_only_and_requested_tier_cannot_cross_loaded_tier() {
        let legacy = fixture(Some(Quant::Q4), false);
        let prepared = PreparedMemory::prepare(&spec(&legacy, Some(Quant::Q4))).unwrap();
        assert!(matches!(
            prepared
                .contract
                .capability(MemoryStrategy::BoundedDecode)
                .unwrap()
                .support,
            MemoryStrategySupport::Missing
        ));
        assert!(PreparedMemory::prepare(&spec(&legacy, Some(Quant::Q8))).is_err());
    }

    #[test]
    fn ordered_duplicate_adapter_receipts_bind_kind_scale_bytes_and_mutation() {
        let fixture = fixture(Some(Quant::Q4), true);
        let shared = fixture.root.join("shared.safetensors");
        let first = adapter(&shared, 0.25, AdapterKind::Lora);
        let mut second = first.clone();
        second.scale = 0.75;
        second.kind = AdapterKind::Lokr;
        let mut load = spec(&fixture, Some(Quant::Q4));
        load.adapters = vec![first, second];
        let prepared = PreparedMemory::prepare(&load).unwrap();
        assert_eq!(prepared.adapters.len(), 2);
        assert_eq!(prepared.adapters[0].ordinal, 0);
        assert_eq!(prepared.adapters[1].ordinal, 1);
        assert_eq!(prepared.artifact.facts.overlay_bytes, 48);
        prepared.ensure_unchanged(&load.adapters).unwrap();

        load.adapters.swap(0, 1);
        assert!(prepared.ensure_unchanged(&load.adapters).is_err());
        load.adapters.swap(0, 1);
        write_shard(
            &shared,
            [(
                "adapter.weight",
                Tensor::ones((2, 3), CandleDType::BF16, &Device::Cpu).unwrap(),
            )],
        );
        assert!(prepared.ensure_unchanged(&load.adapters).is_err());
    }

    #[test]
    fn caller_sealed_base_and_adapter_receipts_refuse_preload_drift() {
        fn sealed(fixture: &Fixture, adapter_path: &Path) -> (LoadSpec, PathBuf) {
            fn lexical_files(dir: &Path, files: &mut Vec<PathBuf>) {
                let mut entries = std::fs::read_dir(dir)
                    .unwrap()
                    .collect::<Result<Vec<_>, _>>()
                    .unwrap();
                entries.sort_by_key(|entry| entry.file_name());
                for entry in entries {
                    let path = entry.path();
                    if entry.file_type().unwrap().is_dir() {
                        lexical_files(&path, files);
                    } else {
                        files.push(path);
                    }
                }
            }
            let adapter = adapter(adapter_path, 0.5, AdapterKind::Lora);
            let mut load = spec(fixture, Some(Quant::Q4));
            load.adapters = vec![adapter];
            let mut files = Vec::new();
            lexical_files(&fixture.root, &mut files);
            let mut pins = files
                .into_iter()
                .map(gen_core::PinnedWeightsFile::pin)
                .collect::<gen_core::Result<Vec<_>>>()
                .unwrap();
            pins.push(gen_core::PinnedWeightsFile::pin(adapter_path).unwrap());
            load.prepare_with_file_pins(pins).unwrap();
            load.validate_prepared_file_pins().unwrap();
            (load, fixture.root.join("transformer/model.safetensors"))
        }

        let base_fixture = fixture(Some(Quant::Q4), true);
        let base_adapter = base_fixture._temp.path().join("base-adapter.safetensors");
        let (base_load, base_shard) = sealed(&base_fixture, &base_adapter);
        write_shard(
            &base_shard,
            [(
                "blocks.0.proj_out.weight",
                Tensor::ones((2, 8), CandleDType::U32, &Device::Cpu).unwrap(),
            )],
        );
        assert!(PreparedMemory::prepare(&base_load).is_err());

        let adapter_fixture = fixture(Some(Quant::Q4), true);
        let adapter_path = adapter_fixture._temp.path().join("adapter.safetensors");
        let (adapter_load, _) = sealed(&adapter_fixture, &adapter_path);
        write_shard(
            &adapter_path,
            [(
                "adapter.weight",
                Tensor::ones((2, 3), CandleDType::BF16, &Device::Cpu).unwrap(),
            )],
        );
        assert!(PreparedMemory::prepare(&adapter_load).is_err());
    }

    #[test]
    fn still_gate_covers_all_public_geometries_and_rejects_crossed_routes() {
        for &(width, height) in PUBLIC_GEOMETRIES {
            assert!(validate_generation_request(&still("t2i", width, height), false).is_ok());
            let mut edit = still("i2i", width, height);
            edit.conditioning.push(Conditioning::Reference {
                image: Image {
                    width: 2,
                    height: 2,
                    pixels: vec![0; 12],
                },
                strength: None,
            });
            assert!(validate_generation_request(&edit, true).is_ok());
        }
        let mut unknown = still("unknown", 512, 512);
        assert!(validate_generation_request(&unknown, false).is_err());
        unknown.video_mode = Some("t2i".into());
        unknown.count = 2;
        assert!(validate_generation_request(&unknown, false).is_err());
        let mut multi = still("i2i", 512, 512);
        multi.conditioning.push(Conditioning::MultiReference {
            images: vec![Image::default(), Image::default()],
        });
        assert!(validate_generation_request(&multi, false).is_err());
        let mut pid = still("t2i", 512, 512);
        pid.use_pid = true;
        assert!(validate_generation_request(&pid, false).is_err());
    }

    #[test]
    fn request_selected_decode_cap_is_exact_and_cannot_smuggle_other_rungs() {
        let mut request = still("t2i", 1280, 720);
        request.memory = Some(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(320),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        });
        assert_eq!(selected_decode_cap(&request).unwrap(), Some(320));
        validate_generation_request(&request, false).unwrap();
        request.memory.as_mut().unwrap().chunk_attention = true;
        assert!(validate_generation_request(&request, false).is_err());
        request.memory = Some(GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(128),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        });
        assert!(selected_decode_cap(&request).is_err());
    }

    #[test]
    fn safety_gate_binds_mode_reference_tier_geometry_and_selection() {
        let fixture = fixture(Some(Quant::Q8), true);
        let prepared = PreparedMemory::prepare(&spec(&fixture, Some(Quant::Q8))).unwrap();
        let mut context = gen_core::standard_memory_behavior_context(
            &prepared.contract,
            MemoryStrategy::BoundedDecode,
            prepared.tier,
            MemoryBehaviorRoute {
                mode: MemoryMode::TextToImage,
                reference_count: 0,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .unwrap();
        context.geometry.width = 512;
        context.geometry.height = 512;
        context.geometry.frames = 1;
        context.geometry.batch = 1;
        context.geometry.reference_count = 0;
        context.selection.parameters = MemoryStrategyParameters {
            decode_tile_edge: Some(320),
            decode_overlap: Some(DECODE_OVERLAP),
            ..Default::default()
        };
        assert!(matches!(
            safety_check(&prepared.contract, prepared.tier, &context),
            MemorySafetyDecision::Accept
        ));
        context.geometry.reference_count = 1;
        assert!(matches!(
            safety_check(&prepared.contract, prepared.tier, &context),
            MemorySafetyDecision::Reject { .. }
        ));
    }
}
