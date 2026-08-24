//! Request-authoritative Candle/CUDA memory contract for dense SANA Base and Sprint.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, GenerationMemory, GenerationRequest, LoadSpec, MemoryAssetFacts,
    MemoryBackendRealization, MemoryCalibrationIdentity, MemoryComponentKind,
    MemoryComponentResidency, MemoryFormulaKind, MemoryFormulaVariable, MemoryGeometry,
    MemoryLifecycleCapabilities, MemoryMode, MemoryNumericTier, MemoryParameterRanges, MemoryPhase,
    MemoryPrerequisiteScope, MemoryProviderContract, MemoryRequestScope, MemoryResidentComponent,
    MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyPrerequisite, MemoryStrategySupport, MemoryWindowMaterialization, Precision,
    TransformerComponent, WeightsSource,
};
use sha2::{Digest, Sha256};

pub const BASE_REPOSITORY: &str = "Efficient-Large-Model/Sana_1600M_1024px_diffusers";
pub const BASE_REVISION: &str = "ac0da2ff55fbe434795be0dce883042e4d49e2fc";
pub const SPRINT_REPOSITORY: &str = "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers";
pub const SPRINT_REVISION: &str = "19683c58b7ea290e55cedd8950ae1d86ada7ef96";
pub const REQUEST_EVIDENCE_REVISION: &str = "sana-candle-dense-request-contract-v1";
// The released DC-AE bounded-decode domain is 192 px with one 48 px blend cell.
// Candle executes this exact singleton through `DcAeDecoder::decode_with`; unlike MLX it does
// not expose the larger measured menu, so publishing any additional edge here would advertise an
// unsupported request shape.
pub const DECODE_TILE_EDGE: u32 = 192;
pub const DECODE_OVERLAP: u32 = 48;
pub const ATTENTION_CHUNK_SIZES: &[u32] = &[4_194_304, 2_097_152, 1_048_576];
pub const TRANSFORMER_WINDOW_SIZES: &[u32] = &[1, 2, 4, 5, 10];
pub const TRANSFORMER_BLOCKS: u32 = 20;
pub const PHYSICAL_RECEIPT_PREFIX: &str = "sana.candle.dense.physical.sha256:";

const BASE_FILES: &[&str] = &[
    ".gitattributes",
    "LICENSE",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.fp16-00001-of-00002.safetensors",
    "text_encoder/model.fp16-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.fp16.json",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.fp16.safetensors",
    "transformer/diffusion_pytorch_model.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
];

const SPRINT_FILES: &[&str] = &[
    ".gitattributes",
    "LICENSE",
    "README.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "text_encoder/model.safetensors.index.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SanaVariant {
    Base,
    Sprint,
}

impl SanaVariant {
    pub const fn provider_id(self) -> &'static str {
        match self {
            Self::Base => crate::MODEL_ID,
            Self::Sprint => crate::SPRINT_MODEL_ID,
        }
    }
    pub const fn repository(self) -> &'static str {
        match self {
            Self::Base => BASE_REPOSITORY,
            Self::Sprint => SPRINT_REPOSITORY,
        }
    }
    pub const fn revision(self) -> &'static str {
        match self {
            Self::Base => BASE_REVISION,
            Self::Sprint => SPRINT_REVISION,
        }
    }
    const fn inventory(self) -> &'static [&'static str] {
        match self {
            Self::Base => BASE_FILES,
            Self::Sprint => SPRINT_FILES,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SanaLoadSeal {
    variant: SanaVariant,
    root: PathBuf,
    paths: Vec<PathBuf>,
    files: Vec<(gen_core::PinnedWeightsFile, [u8; 32])>,
    contract: MemoryProviderContract,
}

impl SanaLoadSeal {
    pub fn capture(variant: SanaVariant, spec: &LoadSpec) -> gen_core::Result<Self> {
        validate_load_spec(variant, spec)?;
        let WeightsSource::Dir(root) = &spec.weights else {
            unreachable!()
        };
        validate_immutable_root(variant, root)?;
        let paths = exact_inventory(variant, root)?;
        validate_loader_tensor_formats(variant, root)?;
        let files = paths
            .iter()
            .map(|path| {
                let pin = gen_core::PinnedWeightsFile::pin(path)?;
                let digest = pin.read_unchanged(sha256_file)?;
                Ok((pin, digest))
            })
            .collect::<gen_core::Result<Vec<_>>>()?;
        let contract = build_contract(variant, spec, root, &files)?;
        let seal = Self {
            variant,
            root: std::path::absolute(root)?,
            paths,
            files,
            contract,
        };
        seal.ensure_unchanged()?;
        Ok(seal)
    }

    pub fn contract(&self) -> &MemoryProviderContract {
        &self.contract
    }

    pub fn ensure_unchanged(&self) -> gen_core::Result<()> {
        let current = exact_inventory(self.variant, &self.root)?;
        if current != self.paths {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: immutable snapshot inventory changed after admission",
                self.variant.provider_id()
            )));
        }
        for (pin, digest) in &self.files {
            pin.ensure_unchanged()?;
            if pin.read_unchanged(sha256_file)? != *digest {
                return Err(gen_core::Error::Unsupported(format!(
                    "{}: immutable snapshot content changed after admission: {}",
                    self.variant.provider_id(),
                    pin.loader_path().display()
                )));
            }
        }
        validate_loader_tensor_formats(self.variant, &self.root)
    }
}

fn validate_load_spec(variant: SanaVariant, spec: &LoadSpec) -> gen_core::Result<()> {
    if !matches!(spec.weights, WeightsSource::Dir(_)) {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: requires an immutable snapshot directory",
            variant.provider_id()
        )));
    }
    if spec.precision != Precision::Bf16 || spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: Candle supports only the dense physical tier (precision=Bf16 sentinel, quant=None)",
            variant.provider_id()
        )));
    }
    if !spec.adapters.is_empty()
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
        || !spec.components.is_empty()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: does not accept external components or adapters",
            variant.provider_id()
        )));
    }
    Ok(())
}

fn validate_immutable_root(variant: SanaVariant, root: &Path) -> gen_core::Result<()> {
    let parts = std::path::absolute(root)?
        .components()
        .map(|part| part.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    let marker = format!("models--{}", variant.repository().replace('/', "--"));
    let app = variant.repository().replace('/', "__");
    let name = variant.repository().rsplit('/').next().unwrap_or_default();
    let revision = variant.revision();
    let valid = parts
        .windows(3)
        .any(|w| w == [marker.as_str(), "snapshots", revision])
        || parts.windows(2).any(|w| w == [app.as_str(), revision])
        || parts.windows(2).any(|w| w == [name, revision]);
    if valid {
        Ok(())
    } else {
        Err(gen_core::Error::Unsupported(format!(
            "{}: source must be exact immutable {}@{}",
            variant.provider_id(),
            variant.repository(),
            revision
        )))
    }
}

fn collect_files(root: &Path, out: &mut Vec<PathBuf>) -> gen_core::Result<()> {
    let metadata = std::fs::symlink_metadata(root)?;
    if metadata.file_type().is_symlink() || metadata.is_file() {
        if !std::fs::metadata(root)?.is_file() {
            return Err(gen_core::Error::Unsupported(format!(
                "non-file snapshot entry {}",
                root.display()
            )));
        }
        out.push(std::path::absolute(root)?);
        return Ok(());
    }
    let mut entries = std::fs::read_dir(root)?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    entries.sort();
    for entry in entries {
        collect_files(&entry, out)?;
    }
    Ok(())
}

fn exact_inventory(variant: SanaVariant, root: &Path) -> gen_core::Result<Vec<PathBuf>> {
    let root = std::path::absolute(root)?;
    let mut paths = Vec::new();
    collect_files(&root, &mut paths)?;
    paths.sort();
    let actual = paths
        .iter()
        .map(|path| {
            path.strip_prefix(&root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect::<BTreeSet<_>>();
    let expected = variant
        .inventory()
        .iter()
        .map(|value| (*value).to_owned())
        .collect::<BTreeSet<_>>();
    if actual != expected {
        let missing = expected.difference(&actual).cloned().collect::<Vec<_>>();
        let extra = actual.difference(&expected).cloned().collect::<Vec<_>>();
        return Err(gen_core::Error::Unsupported(format!(
            "{}: snapshot inventory differs from {}@{}; missing={missing:?} extra={extra:?}",
            variant.provider_id(),
            variant.repository(),
            variant.revision()
        )));
    }
    Ok(paths)
}

fn validate_component_dtype(
    path: &Path,
    label: &str,
    allowed: &[safetensors::Dtype],
) -> gen_core::Result<u64> {
    let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
    if headers.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "SANA {label} tensor inventory is empty"
        )));
    }
    let mut runtime = 0_u64;
    for header in headers {
        if !allowed.contains(&header.dtype) {
            return Err(gen_core::Error::Unsupported(format!(
                "SANA {label} tensor {} has forbidden physical dtype {:?}",
                header.name, header.dtype
            )));
        }
        let elements = header
            .shape
            .iter()
            .try_fold(1_u64, |n, dim| n.checked_mul(*dim as u64))
            .ok_or_else(|| {
                gen_core::Error::Unsupported(format!("SANA {label} tensor shape overflows"))
            })?;
        runtime = runtime.saturating_add(elements.saturating_mul(4));
    }
    Ok(runtime)
}

fn selected_component_bytes(
    variant: SanaVariant,
    root: &Path,
) -> gen_core::Result<(u64, u64, u64)> {
    let te = crate::pipeline::resolve_component_files(&root.join("text_encoder"))?;
    let transformer = crate::pipeline::resolve_component_files(&root.join("transformer"))?;
    let vae = crate::pipeline::resolve_component_files(&root.join("vae"))?;
    let te_bytes = te.iter().try_fold(0_u64, |sum, file| {
        validate_component_dtype(file, "text encoder", &[safetensors::Dtype::BF16])
            .map(|n| sum.saturating_add(n))
    })?;
    let trunk_dtype = match variant {
        SanaVariant::Base => &[safetensors::Dtype::F32][..],
        SanaVariant::Sprint => &[safetensors::Dtype::BF16][..],
    };
    let transformer_bytes = transformer.iter().try_fold(0_u64, |sum, file| {
        validate_component_dtype(file, "transformer", trunk_dtype).map(|n| sum.saturating_add(n))
    })?;
    let vae_bytes = vae.iter().try_fold(0_u64, |sum, file| {
        validate_component_dtype(file, "VAE", &[safetensors::Dtype::F32])
            .map(|n| sum.saturating_add(n))
    })?;
    Ok((te_bytes, transformer_bytes, vae_bytes))
}

fn validate_loader_tensor_formats(variant: SanaVariant, root: &Path) -> gen_core::Result<()> {
    selected_component_bytes(variant, root).map(|_| ())
}

pub(crate) fn sha256_file(path: &Path) -> gen_core::Result<[u8; 32]> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
    }
    Ok(digest.finalize().into())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Whether this load can actually execute [`MemoryStrategy::BoundedTransformerResidency`].
///
/// Block windowing runs through [`crate::transformer::SanaTransformer::from_files_windowed`], which
/// pins every transformer shard with `PinnedWeightsFile::pin` and **re-opens** it once per denoise
/// forward to materialize the selected block window. An eager load has already bulk-materialized the
/// stack and holds no re-openable pinned files, so the rung is not executable there and must not be
/// advertised as `Implemented`. Mirrors `candle-gen-qwen-image` and `candle-gen-sensenova`.
fn streamable(spec: &LoadSpec) -> bool {
    matches!(
        spec.load_shape,
        gen_core::LoadShape::DeferredMaterialization
    ) && matches!(spec.weights, WeightsSource::Dir(_))
}

fn build_contract(
    variant: SanaVariant,
    spec: &LoadSpec,
    root: &Path,
    files: &[(gen_core::PinnedWeightsFile, [u8; 32])],
) -> gen_core::Result<MemoryProviderContract> {
    let (conditioning, transformer, decoder) = selected_component_bytes(variant, root)?;
    let mut assembly = Sha256::new();
    assembly.update(variant.repository().as_bytes());
    assembly.update(variant.revision().as_bytes());
    assembly.update(match variant {
        SanaVariant::Base => b"true-cfg-negative-prompt" as &[u8],
        SanaVariant::Sprint => b"cfg-free-embedded-guidance" as &[u8],
    });
    for (pin, digest) in files {
        assembly.update(pin.loader_path().to_string_lossy().as_bytes());
        assembly.update(digest);
    }
    Ok(assemble_contract(
        variant,
        spec,
        format!("{PHYSICAL_RECEIPT_PREFIX}{}", hex(&assembly.finalize())),
        MemoryAssetFacts {
            base_bytes: conditioning
                .saturating_add(transformer)
                .saturating_add(decoder),
            conditioning_bytes: conditioning,
            transformer_bytes: transformer,
            decoder_bytes: decoder,
            overlay_bytes: 0,
        },
    ))
}

/// The registry-only, weights-free contract: the exact route declaration the sealed contract
/// publishes, with zero asset facts injected and no filesystem traversal.
///
/// The physical receipt is deliberately absent rather than synthesized — it is a digest OF the
/// pinned assets, and a stand-in would be a machine-independent-looking value standing for facts
/// nobody measured. The route stays distinguishable through `contract.calibration`, which is keyed
/// on the variant.
pub fn weights_free_contract(
    variant: SanaVariant,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    validate_load_spec(variant, spec)?;
    Ok(assemble_contract(
        variant,
        spec,
        String::new(),
        MemoryAssetFacts::default(),
    ))
}

fn assemble_contract(
    variant: SanaVariant,
    spec: &LoadSpec,
    receipt: String,
    asset_facts: MemoryAssetFacts,
) -> MemoryProviderContract {
    let transformer = asset_facts.transformer_bytes;
    let streamable = streamable(spec);
    let mut contract = MemoryProviderContract::compatibility_default(
        variant.provider_id(),
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: streamable,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    contract.load_shape = spec.load_shape;
    for capability in &mut contract.strategies {
        capability.support =
            if capability.strategy == MemoryStrategy::BoundedTransformerResidency && !streamable {
                MemoryStrategySupport::Missing
            } else {
                MemoryStrategySupport::Implemented
            };
        capability.parameters = match capability.strategy {
            MemoryStrategy::BoundedDecode => MemoryParameterRanges {
                decode_tile_edges: vec![DECODE_TILE_EDGE],
                decode_overlaps: vec![DECODE_OVERLAP],
                ..Default::default()
            },
            MemoryStrategy::BoundedAttention => MemoryParameterRanges {
                attention_chunk_sizes: ATTENTION_CHUNK_SIZES.to_vec(),
                ..Default::default()
            },
            MemoryStrategy::BoundedTransformerResidency if streamable => MemoryParameterRanges {
                transformer_window_sizes: TRANSFORMER_WINDOW_SIZES.to_vec(),
                transformer_window_components: vec![TransformerComponent::Dit],
                ..Default::default()
            },
            _ => MemoryParameterRanges::default(),
        };
    }
    contract.additional_prerequisites = [
        MemoryStrategy::BoundedDecode,
        MemoryStrategy::BoundedAttention,
        MemoryStrategy::BoundedTransformerResidency,
    ]
    .into_iter()
    .filter(|strategy| streamable || *strategy != MemoryStrategy::BoundedTransformerResidency)
    .map(|strategy| {
        (
            strategy,
            MemoryStrategyPrerequisite::Rung {
                rung: MemoryStrategy::StagedResidency,
                scope: MemoryPrerequisiteScope::EngagedInSameRequest,
            },
        )
    })
    .collect();
    contract.lifecycle = MemoryLifecycleCapabilities {
        phases: vec![
            MemoryPhase::Conditioning,
            MemoryPhase::Denoise,
            MemoryPhase::Decode,
        ],
        synchronized_phase_release: true,
        decode_tiling: true,
        attention_chunking: true,
        transformer_window_materialization: streamable,
    };
    contract.formula = MemoryFormulaKind::ComponentPhaseEnvelope {
        phases: contract.lifecycle.phases.clone(),
        variables: vec![
            MemoryFormulaVariable::AssetBytes,
            MemoryFormulaVariable::PixelCount,
            MemoryFormulaVariable::BatchCount,
            MemoryFormulaVariable::ConditioningTokenCount,
            MemoryFormulaVariable::DecodeTileArea,
            MemoryFormulaVariable::AttentionChunkSize,
            MemoryFormulaVariable::TransformerWindowSize,
        ],
        // A resident component must declare non-zero bytes, so the weights-free fixture — which
        // injects zero asset facts by construction — publishes none. The physical receipt it would
        // have carried is an asset digest and has no meaning without the assets.
        resident_components: if transformer == 0 {
            Vec::new()
        } else {
            vec![MemoryResidentComponent {
                id: receipt,
                kind: MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
                resident_bytes: transformer,
                bounded_by: Some(MemoryStrategy::StagedResidency),
                residency: MemoryComponentResidency::WholeRender,
            }]
        },
    };
    contract.calibration = Some(MemoryCalibrationIdentity::new(
        format!(
            "sana-candle-dense-{}-full-ladder-v1",
            match variant {
                SanaVariant::Base => "base",
                SanaVariant::Sprint => "sprint",
            }
        ),
        spec.load_shape,
    ));
    contract.asset_facts = asset_facts;
    contract
}

pub fn resolved_numeric_tier() -> MemoryNumericTier {
    MemoryNumericTier {
        precision: Precision::Bf16,
        quant: None,
        component_precision_floors: &[],
    }
}

fn supported_route(context: &MemoryRunContext) -> bool {
    !context.use_pid
        && context.overlay.is_none()
        && context.geometry.batch >= 1
        && context.geometry.frames == 1
        && matches!(
            (
                context.mode.clone(),
                context.geometry.reference_count,
                context.has_reference
            ),
            (MemoryMode::TextToImage, 0, false) | (MemoryMode::ImageToImage, 1, true)
        )
}

pub fn validate_context(seal: &SanaLoadSeal, context: &MemoryRunContext) -> gen_core::Result<()> {
    seal.ensure_unchanged()?;
    validate_context_axes(seal.contract(), context)
}

/// Every route/tier/evidence axis check the sealed admission runs, minus the on-disk snapshot
/// re-verification. Split out so the weights-free registry seam runs the *same* admission logic
/// against a fixture contract instead of a parallel, weaker reimplementation.
fn validate_context_axes(
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    if let MemorySafetyDecision::Reject { reason } = gen_core::standard_memory_strategy_safety_check(
        contract,
        context,
        Some(resolved_numeric_tier()),
        None,
    ) {
        return Err(gen_core::Error::Unsupported(reason));
    }
    if !supported_route(context) {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: only exact T2I/ref0 and I2I/ref1 routes are admitted",
            contract.provider_id
        )));
    }
    if context.evidence_revision != REQUEST_EVIDENCE_REVISION {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: request evidence {} does not match {}",
            contract.provider_id, context.evidence_revision, REQUEST_EVIDENCE_REVISION
        )));
    }
    if context.has_phases && context.mode != MemoryMode::ImageToImage {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: a Hires final-pass context must be I2I/ref1",
            contract.provider_id
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Binding {
    address: usize,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
    use_pid: bool,
    public_identity: [u8; 32],
}
impl Binding {
    fn new(req: &GenerationRequest) -> Self {
        let mut digest = Sha256::new();
        for bytes in [
            req.prompt.as_bytes(),
            req.negative_prompt
                .as_deref()
                .unwrap_or_default()
                .as_bytes(),
            req.sampler.as_deref().unwrap_or_default().as_bytes(),
            req.scheduler.as_deref().unwrap_or_default().as_bytes(),
            req.guidance_method
                .as_deref()
                .unwrap_or_default()
                .as_bytes(),
        ] {
            digest.update((bytes.len() as u64).to_le_bytes());
            digest.update(bytes);
        }
        for value in [
            req.width as u64,
            req.height as u64,
            req.count as u64,
            req.frames.unwrap_or(1) as u64,
            req.seed.unwrap_or_default(),
            req.steps.unwrap_or_default() as u64,
            req.guidance.map(f32::to_bits).unwrap_or_default() as u64,
            req.true_cfg.map(f32::to_bits).unwrap_or_default() as u64,
            req.scheduler_shift.map(f32::to_bits).unwrap_or_default() as u64,
            req.strength.map(f32::to_bits).unwrap_or_default() as u64,
        ] {
            digest.update(value.to_le_bytes());
        }
        for conditioning in &req.conditioning {
            if let gen_core::Conditioning::Reference { image, strength } = conditioning {
                digest.update(b"reference");
                digest.update(image.width.to_le_bytes());
                digest.update(image.height.to_le_bytes());
                digest.update((image.pixels.len() as u64).to_le_bytes());
                digest.update(&image.pixels);
                digest.update(strength.map(f32::to_bits).unwrap_or_default().to_le_bytes());
            } else {
                digest.update(format!("{conditioning:?}").as_bytes());
            }
        }
        Self {
            address: std::ptr::from_ref(req).addr(),
            geometry: MemoryGeometry {
                width: req.width,
                height: req.height,
                batch: req.count,
                frames: req.frames.unwrap_or(1),
                reference_count: req.image_reference_count(),
            },
            memory: req.memory,
            use_pid: req.use_pid,
            public_identity: digest.finalize().into(),
        }
    }
}
struct Active {
    token: u64,
    context: MemoryRunContext,
    expected: Option<GenerationMemory>,
    binding: Option<Binding>,
    consumed: bool,
}
#[derive(Default)]
struct AdmissionState {
    next: u64,
    approved: Option<MemoryRunContext>,
    active: Option<Active>,
}
#[derive(Clone)]
pub struct AdmissionRegistry {
    provider_id: &'static str,
    state: Arc<Mutex<AdmissionState>>,
}
impl AdmissionRegistry {
    pub fn new(provider_id: &'static str) -> Self {
        Self {
            provider_id,
            state: Arc::new(Mutex::new(AdmissionState::default())),
        }
    }
    pub fn approve(&self, context: &MemoryRunContext) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.state);
        if state.active.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: another request is active",
                self.provider_id
            )));
        }
        state.approved = Some(context.clone());
        Ok(())
    }
    pub fn clear(&self) {
        candle_gen::lock_recover(&self.state).approved = None;
    }
    fn begin(
        &self,
        contract: &MemoryProviderContract,
        context: &MemoryRunContext,
    ) -> gen_core::Result<u64> {
        let mut state = candle_gen::lock_recover(&self.state);
        if state.active.is_some() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: another request scope is active",
                self.provider_id
            )));
        }
        let approved = state.approved.take().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{}: request skipped safety approval",
                self.provider_id
            ))
        })?;
        if approved != *context || contract.provider_id != self.provider_id {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: crossed or changed safety context",
                self.provider_id
            )));
        }
        state.next = state.next.wrapping_add(1).max(1);
        let token = state.next;
        state.active = Some(Active {
            token,
            context: context.clone(),
            expected: contract.generation_memory(&context.selection),
            binding: None,
            consumed: false,
        });
        Ok(token)
    }
    fn configure(&self, token: u64, request: &GenerationRequest) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.state);
        let active = state.active.as_mut().ok_or_else(|| {
            gen_core::Error::Unsupported(format!("{}: inactive request scope", self.provider_id))
        })?;
        let binding = Binding::new(request);
        if active.token != token
            || active.binding.is_some()
            || binding.geometry != active.context.geometry
            || binding.memory != active.expected
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: stale or changed request",
                self.provider_id
            )));
        }
        active.binding = Some(binding);
        Ok(())
    }
    pub fn consume(&self, request: &GenerationRequest) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.state);
        let constrained = request
            .memory
            .is_some_and(|memory| memory != GenerationMemory::default());
        let Some(active) = state.active.as_mut() else {
            return if constrained {
                Err(gen_core::Error::Unsupported(format!(
                    "{}: constrained request lacks admission",
                    self.provider_id
                )))
            } else {
                Ok(())
            };
        };
        if active.binding.as_ref() != Some(&Binding::new(request)) || active.consumed {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request changed or was already consumed",
                self.provider_id
            )));
        }
        active.consumed = true;
        Ok(())
    }
    fn finish(&self, token: u64) -> gen_core::Result<()> {
        let mut state = candle_gen::lock_recover(&self.state);
        if state
            .active
            .as_ref()
            .is_some_and(|active| active.token == token)
        {
            state.active = None;
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: stale request token",
                self.provider_id
            )))
        }
    }
    fn abandon(&self, token: u64) {
        let mut state = candle_gen::lock_recover(&self.state);
        if state
            .active
            .as_ref()
            .is_some_and(|active| active.token == token)
        {
            state.active = None;
        }
    }
}

struct Scope {
    device: Device,
    admission: AdmissionRegistry,
    token: u64,
    geometry: MemoryGeometry,
    memory: Option<GenerationMemory>,
    finished: bool,
}
impl Drop for Scope {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.device.synchronize();
            self.admission.abandon(self.token);
        }
    }
}
impl MemoryRequestScope for Scope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        if self.finished {
            return Err(gen_core::Error::Unsupported("SANA scope finished".into()));
        }
        request.memory = self.memory;
        self.admission.configure(self.token, request)
    }
    fn enter_phase(&mut self, _: MemoryPhase) -> gen_core::Result<()> {
        if self.finished {
            Err(gen_core::Error::Unsupported("SANA scope finished".into()))
        } else {
            Ok(())
        }
    }
    fn leave_phase(&mut self, _: MemoryPhase) -> gen_core::Result<()> {
        if self.finished {
            return Err(gen_core::Error::Unsupported("SANA scope finished".into()));
        }
        self.device.synchronize().map_err(gen_core::Error::backend)
    }
    fn configure_decode(
        &mut self,
        edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> gen_core::Result<()> {
        if self.finished {
            return Err(gen_core::Error::Unsupported("SANA scope finished".into()));
        }
        if geometry != self.geometry || edge != DECODE_TILE_EDGE || overlap != DECODE_OVERLAP {
            Err(gen_core::Error::Unsupported(
                "SANA decode parameters or geometry were not admitted".into(),
            ))
        } else {
            Ok(())
        }
    }
    fn configure_attention(&mut self, size: u32) -> gen_core::Result<()> {
        if self.finished {
            return Err(gen_core::Error::Unsupported("SANA scope finished".into()));
        }
        if ATTENTION_CHUNK_SIZES.contains(&size) {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(
                "SANA attention budget was not admitted".into(),
            ))
        }
    }
    fn materialize_transformer_window(&mut self, first: u32, count: u32) -> gen_core::Result<()> {
        if self.finished {
            return Err(gen_core::Error::Unsupported("SANA scope finished".into()));
        }
        let window = self
            .memory
            .and_then(|memory| memory.transformer_window_size)
            .unwrap_or(1);
        if first < TRANSFORMER_BLOCKS
            && count == window.min(TRANSFORMER_BLOCKS - first)
            && first.is_multiple_of(window)
        {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(
                "SANA transformer window was not admitted".into(),
            ))
        }
    }
    fn finish(&mut self, _: MemoryRunOutcome) -> gen_core::Result<()> {
        if self.finished {
            return Err(gen_core::Error::Unsupported(
                "SANA scope was already finished".into(),
            ));
        }
        self.device
            .synchronize()
            .map_err(gen_core::Error::backend)?;
        self.admission.finish(self.token)?;
        self.finished = true;
        Ok(())
    }
}

pub fn begin_request(
    seal: &SanaLoadSeal,
    admission: AdmissionRegistry,
    device: Device,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    validate_context(seal, context)?;
    let token = admission.begin(seal.contract(), context)?;
    Ok(Some(Box::new(Scope {
        device,
        admission,
        token,
        geometry: context.geometry,
        memory: seal.contract().generation_memory(&context.selection),
        finished: false,
    })))
}

pub fn safety_check(
    seal: &SanaLoadSeal,
    admission: &AdmissionRegistry,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match validate_context(seal, context).and_then(|()| admission.approve(context)) {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => {
            admission.clear();
            MemorySafetyDecision::Reject {
                reason: error.to_string(),
            }
        }
    }
}

pub fn registered_valid_fixture(
    seal: &SanaLoadSeal,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized() {
        return Ok(Vec::new());
    }
    let mut context = gen_core::standard_memory_behavior_context(
        seal.contract(),
        strategy,
        resolved_numeric_tier(),
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )?;
    context.evidence_revision = REQUEST_EVIDENCE_REVISION.to_owned();
    Ok(vec![gen_core::MemoryBehaviorFixture::new(context)])
}

// -------------------------------------------------------------------------------------------------
// Pre-load registry seams (sc-19753 feature review, BLOCKER 4)
//
// Without these a selector cannot price SANA before weights land: the crate registered two
// generators and nothing else, so `ProviderRegistry` memory resolution had no SANA route at all.
// Construction is deliberately CUDA-free — everything below runs on a host with no GPU, so
// `register_memory_contract_surfaces` is called unconditionally from `register_providers`.
// -------------------------------------------------------------------------------------------------

/// Production, pre-load contract for one variant: seals the immutable snapshot exactly as `load`
/// does, so the registered contract is the one the loaded generator will publish.
pub fn provider_contract(
    variant: SanaVariant,
    spec: &LoadSpec,
) -> gen_core::Result<MemoryProviderContract> {
    Ok(SanaLoadSeal::capture(variant, spec)?.contract().clone())
}

/// The loaded generator's real admission check, reachable before weights are opened.
///
/// Weights-free fixture contracts (zero [`MemoryAssetFacts`]) take the axis-only path; a real
/// contract re-seals the snapshot, so this is the same decision [`safety_check`] returns.
pub fn registered_safety_check(
    variant: SanaVariant,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    let result = if contract.asset_facts == MemoryAssetFacts::default() {
        validate_context_axes(contract, context)
    } else {
        SanaLoadSeal::capture(variant, spec).and_then(|seal| validate_context(&seal, context))
    };
    match result {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

/// Executable conformance entry point. Only rungs this contract actually declares `Implemented`
/// produce a fixture, so a non-streamable spec yields none for
/// [`MemoryStrategy::BoundedTransformerResidency`].
pub fn registered_valid_fixtures(
    _spec: &LoadSpec,
    contract: &MemoryProviderContract,
    strategy: MemoryStrategy,
) -> gen_core::Result<Vec<gen_core::MemoryBehaviorFixture>> {
    if !strategy.is_optimized()
        || !matches!(
            contract
                .capability(strategy)
                .map(|capability| &capability.support),
            Some(MemoryStrategySupport::Implemented)
        )
    {
        return Ok(Vec::new());
    }
    let mut context = gen_core::standard_memory_behavior_context(
        contract,
        strategy,
        resolved_numeric_tier(),
        gen_core::MemoryBehaviorRoute {
            mode: MemoryMode::TextToImage,
            reference_count: 0,
            use_pid: false,
            has_phases: false,
            overlay: None,
        },
    )?;
    context.evidence_revision = REQUEST_EVIDENCE_REVISION.to_owned();
    Ok(vec![gen_core::MemoryBehaviorFixture::new(context)])
}

/// Open a conformance request scope through the same admission state machine production uses:
/// validate, `approve`, then `begin`. `Device::Cpu` keeps construction GPU-free — the scope's only
/// device interaction is a `synchronize()` on `finish`, a no-op on CPU.
pub fn registered_begin_request(
    variant: SanaVariant,
    spec: &LoadSpec,
    contract: &MemoryProviderContract,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope>>> {
    let admission = AdmissionRegistry::new(variant.provider_id());
    if contract.asset_facts == MemoryAssetFacts::default() {
        validate_context_axes(contract, context)?;
        admission.approve(context)?;
        let token = admission.begin(contract, context)?;
        return Ok(Some(Box::new(Scope {
            device: Device::Cpu,
            admission,
            token,
            geometry: context.geometry,
            memory: contract.generation_memory(&context.selection),
            finished: false,
        })));
    }
    let seal = SanaLoadSeal::capture(variant, spec)?;
    validate_context(&seal, context)?;
    admission.approve(context)?;
    begin_request(&seal, admission, Device::Cpu, context)
}

/// Complete weights-free registry-load surface for a SANA route.
///
/// The common Candle surface publishes bf16/q4/q8, but Candle SANA is **dense-only**:
/// this crate's private `validate_load_spec` rejects any `quantize`, so a q4/q8 witness would name
/// a route this crate cannot load and the registry would fail the whole surface. Filter the shared helper rather than
/// hand-rolling the offload/load-shape cross product, so a future axis added to gen-core still
/// reaches this provider.
pub fn surface_specs() -> Vec<gen_core::MemoryContractSurfaceSpec> {
    gen_core::candle_memory_contract_surface_specs()
        .into_iter()
        .filter(|surface| {
            surface.resolved_artifact_tier() == gen_core::MemoryContractSurfaceTier::Bf16
        })
        .collect()
}

pub const BASE_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: crate::MODEL_ID,
    contract: |spec| provider_contract(SanaVariant::Base, spec),
    safety_check: |spec, contract, context| {
        registered_safety_check(SanaVariant::Base, spec, contract, context)
    },
};

pub const SPRINT_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: crate::SPRINT_MODEL_ID,
    contract: |spec| provider_contract(SanaVariant::Sprint, spec),
    safety_check: |spec, contract, context| {
        registered_safety_check(SanaVariant::Sprint, spec, contract, context)
    },
};

pub const BASE_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: crate::MODEL_ID,
        valid_fixtures: registered_valid_fixtures,
        begin_request: |spec, contract, context| {
            registered_begin_request(SanaVariant::Base, spec, contract, context)
        },
    };

pub const SPRINT_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: crate::SPRINT_MODEL_ID,
        valid_fixtures: registered_valid_fixtures,
        begin_request: |spec, contract, context| {
            registered_begin_request(SanaVariant::Sprint, spec, contract, context)
        },
    };

#[cfg(test)]
pub(crate) fn fixture_snapshot(variant: SanaVariant) -> (tempfile::TempDir, PathBuf) {
    let temp = tempfile::tempdir().unwrap();
    let marker = format!("models--{}", variant.repository().replace('/', "--"));
    let root = temp
        .path()
        .join(marker)
        .join("snapshots")
        .join(variant.revision());
    for relative in variant.inventory() {
        let path = root.join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let selected = match variant {
            SanaVariant::Base => matches!(
                *relative,
                "text_encoder/model-00001-of-00002.safetensors"
                    | "text_encoder/model-00002-of-00002.safetensors"
                    | "transformer/diffusion_pytorch_model-00001-of-00002.safetensors"
                    | "transformer/diffusion_pytorch_model-00002-of-00002.safetensors"
                    | "vae/diffusion_pytorch_model.safetensors"
            ),
            SanaVariant::Sprint => matches!(
                *relative,
                "text_encoder/model-00001-of-00002.safetensors"
                    | "text_encoder/model-00002-of-00002.safetensors"
                    | "transformer/diffusion_pytorch_model.safetensors"
                    | "vae/diffusion_pytorch_model.safetensors"
            ),
        };
        if selected {
            let dtype = if relative.starts_with("text_encoder/")
                || (variant == SanaVariant::Sprint && relative.starts_with("transformer/"))
            {
                safetensors::Dtype::BF16
            } else {
                safetensors::Dtype::F32
            };
            let bytes = vec![
                0_u8;
                if dtype == safetensors::Dtype::F32 {
                    4
                } else {
                    2
                }
            ];
            let view = safetensors::tensor::TensorView::new(dtype, vec![1], &bytes).unwrap();
            safetensors::serialize_to_file(
                vec![(format!("fixture.{relative}"), view)],
                &None,
                &path,
            )
            .unwrap();
        } else {
            std::fs::write(path, b"fixture").unwrap();
        }
    }
    (temp, root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{LoadShape, Quant};

    fn sealed(variant: SanaVariant) -> (tempfile::TempDir, Arc<SanaLoadSeal>) {
        let (temp, root) = fixture_snapshot(variant);
        let mut spec = LoadSpec::new(WeightsSource::Dir(root));
        spec.load_shape = LoadShape::DeferredMaterialization;
        (
            temp,
            Arc::new(SanaLoadSeal::capture(variant, &spec).unwrap()),
        )
    }

    fn context(
        seal: &SanaLoadSeal,
        strategy: MemoryStrategy,
        mode: MemoryMode,
        refs: u32,
    ) -> MemoryRunContext {
        let mut context = gen_core::standard_memory_behavior_context(
            seal.contract(),
            strategy,
            resolved_numeric_tier(),
            gen_core::MemoryBehaviorRoute {
                mode,
                reference_count: refs,
                use_pid: false,
                has_phases: false,
                overlay: None,
            },
        )
        .unwrap();
        context.evidence_revision = REQUEST_EVIDENCE_REVISION.to_owned();
        context
    }

    /// sc-19753 feature review, MAJOR 10 — rung 4 is only executable when the load can re-open its
    /// transformer shards.
    ///
    /// `SanaTransformer::from_files_windowed` pins every transformer file and re-opens it per
    /// denoise forward to materialize the selected block window. An eager load has already
    /// bulk-materialized the stack, so the rung is not executable there and declaring it
    /// `Implemented` would let a selector price and then select a window SANA cannot honor. Both
    /// arms use the identical snapshot; only `load_shape` differs, so the rung declaration is
    /// attributable to streamability alone.
    #[test]
    fn block_windowing_is_declared_only_for_a_streamable_load() {
        for variant in [SanaVariant::Base, SanaVariant::Sprint] {
            let (_temp, root) = fixture_snapshot(variant);
            let capability = |shape| {
                let mut spec = LoadSpec::new(WeightsSource::Dir(root.clone()));
                spec.load_shape = shape;
                let contract = SanaLoadSeal::capture(variant, &spec)
                    .unwrap()
                    .contract()
                    .clone();
                assert!(
                    contract.conformance_errors().is_empty(),
                    "{:?}",
                    contract.conformance_errors()
                );
                contract
            };

            let eager = capability(LoadShape::EagerMaterialization);
            let rung = eager
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap();
            assert_eq!(
                rung.support,
                MemoryStrategySupport::Missing,
                "{}: an eager load holds no re-openable pinned shards, so block windowing is not \
                 executable and must not be advertised",
                variant.provider_id()
            );
            assert!(
                rung.parameters.transformer_window_sizes.is_empty()
                    && rung.parameters.transformer_window_components.is_empty(),
                "{}: a Missing rung must not publish a selectable window menu",
                variant.provider_id()
            );
            assert!(
                !eager.lifecycle.transformer_window_materialization,
                "{}: the lifecycle hook must agree with the rung declaration",
                variant.provider_id()
            );
            assert!(
                !eager
                    .additional_prerequisites
                    .iter()
                    .any(|(strategy, _)| *strategy == MemoryStrategy::BoundedTransformerResidency),
                "{}: an undeclared rung must not carry a prerequisite",
                variant.provider_id()
            );
            // Every other optimized rung is unaffected by load shape.
            for strategy in [
                MemoryStrategy::BoundedDecode,
                MemoryStrategy::BoundedAttention,
            ] {
                assert_eq!(
                    eager.capability(strategy).unwrap().support,
                    MemoryStrategySupport::Implemented,
                    "{}: {strategy:?} does not depend on streamability",
                    variant.provider_id()
                );
            }

            let deferred = capability(LoadShape::DeferredMaterialization);
            let rung = deferred
                .capability(MemoryStrategy::BoundedTransformerResidency)
                .unwrap();
            assert_eq!(
                rung.support,
                MemoryStrategySupport::Implemented,
                "{}: a deferred load CAN re-open its shards, so the rung must stay available",
                variant.provider_id()
            );
            assert_eq!(
                rung.parameters.transformer_window_sizes,
                TRANSFORMER_WINDOW_SIZES.to_vec()
            );
            assert!(deferred.lifecycle.transformer_window_materialization);
        }
    }

    /// sc-19753 feature review, BLOCKER 4 — the fixture contract must be usable with no weights and
    /// no filesystem, and must still publish the same route declaration as the sealed contract.
    #[test]
    fn the_weights_free_contract_needs_no_snapshot_and_keeps_the_route_declaration() {
        for variant in [SanaVariant::Base, SanaVariant::Sprint] {
            let missing = tempfile::tempdir().unwrap().path().join("never-created");
            assert!(!missing.exists());
            let mut spec = LoadSpec::new(WeightsSource::Dir(missing));
            spec.load_shape = LoadShape::DeferredMaterialization;

            let fixture = weights_free_contract(variant, &spec).unwrap();
            assert_eq!(fixture.provider_id, variant.provider_id());
            assert_eq!(
                fixture.asset_facts,
                MemoryAssetFacts::default(),
                "a fixture must inject zero asset facts"
            );
            assert!(
                fixture.conformance_errors().is_empty(),
                "{:?}",
                fixture.conformance_errors()
            );
            for strategy in MemoryStrategy::ALL {
                assert_eq!(
                    fixture.capability(strategy).unwrap().support,
                    MemoryStrategySupport::Implemented,
                    "{}: {strategy:?}",
                    variant.provider_id()
                );
            }
            assert_eq!(
                fixture
                    .capability(MemoryStrategy::BoundedDecode)
                    .unwrap()
                    .parameters
                    .decode_tile_edges,
                vec![DECODE_TILE_EDGE]
            );

            // An eager fixture drops rung 4 exactly as the sealed contract does.
            let mut eager = spec.clone();
            eager.load_shape = LoadShape::EagerMaterialization;
            assert_eq!(
                weights_free_contract(variant, &eager)
                    .unwrap()
                    .capability(MemoryStrategy::BoundedTransformerResidency)
                    .unwrap()
                    .support,
                MemoryStrategySupport::Missing
            );
        }

        // The two routes must stay distinguishable without assets, and must not silently claim a
        // physical receipt they never measured.
        let mut spec = LoadSpec::new(WeightsSource::Dir("unused".into()));
        spec.load_shape = LoadShape::DeferredMaterialization;
        let base = weights_free_contract(SanaVariant::Base, &spec).unwrap();
        let sprint = weights_free_contract(SanaVariant::Sprint, &spec).unwrap();
        assert_ne!(base.provider_id, sprint.provider_id);
        assert_ne!(base.calibration, sprint.calibration);
        for contract in [&base, &sprint] {
            let MemoryFormulaKind::ComponentPhaseEnvelope {
                resident_components,
                ..
            } = &contract.formula
            else {
                panic!("SANA publishes a component-phase envelope");
            };
            assert!(
                resident_components.is_empty(),
                "{}: a weights-free contract has no measured resident bytes to attest",
                contract.provider_id
            );
        }
    }

    #[test]
    fn base_and_sprint_are_distinct_conformant_dense_full_ladders() {
        let (_base_temp, base) = sealed(SanaVariant::Base);
        let (_sprint_temp, sprint) = sealed(SanaVariant::Sprint);
        for contract in [base.contract(), sprint.contract()] {
            assert!(
                contract.conformance_errors().is_empty(),
                "{:?}",
                contract.conformance_errors()
            );
            assert_eq!(contract.asset_facts.overlay_bytes, 0);
            assert_eq!(
                contract.lifecycle.phases,
                vec![
                    MemoryPhase::Conditioning,
                    MemoryPhase::Denoise,
                    MemoryPhase::Decode
                ]
            );
            assert!(contract.lifecycle.synchronized_phase_release);
            assert!(contract.lifecycle.decode_tiling);
            assert!(contract.lifecycle.attention_chunking);
            assert!(contract.lifecycle.transformer_window_materialization);
            for strategy in MemoryStrategy::ALL {
                assert_eq!(
                    contract.capability(strategy).unwrap().support,
                    MemoryStrategySupport::Implemented
                );
            }
        }
        assert_ne!(base.contract().provider_id, sprint.contract().provider_id);
        assert_ne!(base.contract().calibration, sprint.contract().calibration);
        assert_ne!(
            base.contract().formula,
            sprint.contract().formula,
            "physical receipts must not cross"
        );
        assert_eq!(BASE_REVISION.len(), 40);
        assert_eq!(SPRINT_REVISION.len(), 40);
    }

    #[test]
    fn packed_tiers_mutable_sources_and_forged_tensor_headers_fail_closed() {
        let (_temp, root) = fixture_snapshot(SanaVariant::Base);
        let q4 = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q4);
        assert!(SanaLoadSeal::capture(SanaVariant::Base, &q4).is_err());
        let mutable = tempfile::tempdir().unwrap();
        assert!(SanaLoadSeal::capture(
            SanaVariant::Base,
            &LoadSpec::new(WeightsSource::Dir(mutable.path().into()))
        )
        .is_err());

        let (_forged_temp, forged) = fixture_snapshot(SanaVariant::Base);
        let path = forged.join("transformer/diffusion_pytorch_model-00001-of-00002.safetensors");
        let bytes = [0_u8; 2];
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![1], &bytes)
            .unwrap();
        safetensors::serialize_to_file(vec![("forged", view)], &None, &path).unwrap();
        assert!(SanaLoadSeal::capture(
            SanaVariant::Base,
            &LoadSpec::new(WeightsSource::Dir(forged))
        )
        .is_err());
    }

    #[test]
    fn mutation_after_admission_is_rejected_before_lazy_load() {
        let (_temp, seal) = sealed(SanaVariant::Sprint);
        std::fs::write(seal.root.join("README.md"), b"mutated").unwrap();
        assert!(seal.ensure_unchanged().is_err());
    }

    #[test]
    fn exact_routes_and_evidence_do_not_cross() {
        let (_temp, seal) = sealed(SanaVariant::Base);
        assert!(validate_context(
            &seal,
            &context(&seal, MemoryStrategy::Resident, MemoryMode::TextToImage, 0)
        )
        .is_ok());
        assert!(validate_context(
            &seal,
            &context(
                &seal,
                MemoryStrategy::BoundedDecode,
                MemoryMode::ImageToImage,
                1
            )
        )
        .is_ok());
        let mut crossed = context(&seal, MemoryStrategy::Resident, MemoryMode::Edit, 1);
        assert!(validate_context(&seal, &crossed).is_err());
        crossed = context(&seal, MemoryStrategy::Resident, MemoryMode::TextToImage, 0);
        crossed.evidence_revision = "legacy-cannot-grant".into();
        assert!(validate_context(&seal, &crossed).is_err());
    }

    #[test]
    fn request_identity_concurrency_and_cleanup_are_authoritative() {
        let (_temp, seal) = sealed(SanaVariant::Base);
        let admission = AdmissionRegistry::new(crate::MODEL_ID);
        let context = context(
            &seal,
            MemoryStrategy::BoundedAttention,
            MemoryMode::TextToImage,
            0,
        );
        assert_eq!(
            safety_check(&seal, &admission, &context),
            MemorySafetyDecision::Accept
        );
        let mut scope = begin_request(&seal, admission.clone(), Device::Cpu, &context)
            .unwrap()
            .unwrap();
        assert!(begin_request(&seal, admission.clone(), Device::Cpu, &context).is_err());
        let mut request = GenerationRequest {
            prompt: "sealed prompt".into(),
            width: 1024,
            height: 1024,
            count: 1,
            ..Default::default()
        };
        scope.configure_request(&mut request).unwrap();
        request.prompt.push_str(" crossed");
        assert!(admission.consume(&request).is_err());
        scope
            .finish(MemoryRunOutcome::Error {
                message: "fixture".into(),
            })
            .unwrap();
        for (index, outcome) in [
            MemoryRunOutcome::Complete,
            MemoryRunOutcome::Canceled,
            MemoryRunOutcome::Error {
                message: "fixture".into(),
            },
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                safety_check(&seal, &admission, &context),
                MemorySafetyDecision::Accept
            );
            let mut scope = begin_request(&seal, admission.clone(), Device::Cpu, &context)
                .unwrap()
                .unwrap();
            let mut request = GenerationRequest {
                prompt: format!("warm request {index}"),
                width: 1024,
                height: 1024,
                count: 1,
                ..Default::default()
            };
            scope.configure_request(&mut request).unwrap();
            admission.consume(&request).unwrap();
            scope.finish(outcome).unwrap();
        }
        assert_eq!(
            safety_check(&seal, &admission, &context),
            MemorySafetyDecision::Accept
        );
        let abandoned = begin_request(&seal, admission.clone(), Device::Cpu, &context).unwrap();
        drop(abandoned);
        assert_eq!(
            safety_check(&seal, &admission, &context),
            MemorySafetyDecision::Accept
        );
    }
}
