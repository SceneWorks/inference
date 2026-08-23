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
pub const DECODE_TILE_EDGE: u32 = 512;
pub const DECODE_OVERLAP: u32 = 128;
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
    let receipt = format!("{PHYSICAL_RECEIPT_PREFIX}{}", hex(&assembly.finalize()));
    let mut contract = MemoryProviderContract::compatibility_default(
        variant.provider_id(),
        MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: true,
            block_materialization: MemoryWindowMaterialization::DeviceFormatTransfer,
        },
    );
    contract.load_shape = spec.load_shape;
    for capability in &mut contract.strategies {
        capability.support = MemoryStrategySupport::Implemented;
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
            MemoryStrategy::BoundedTransformerResidency => MemoryParameterRanges {
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
        transformer_window_materialization: true,
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
        resident_components: vec![MemoryResidentComponent {
            id: receipt,
            kind: MemoryComponentKind::TransformerSubStack(TransformerComponent::Dit),
            resident_bytes: transformer,
            bounded_by: Some(MemoryStrategy::StagedResidency),
            residency: MemoryComponentResidency::WholeRender,
        }],
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
    contract.asset_facts = MemoryAssetFacts {
        base_bytes: conditioning
            .saturating_add(transformer)
            .saturating_add(decoder),
        conditioning_bytes: conditioning,
        transformer_bytes: transformer,
        decoder_bytes: decoder,
        overlay_bytes: 0,
    };
    Ok(contract)
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
    let contract = seal.contract();
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
        self.device.synchronize().map_err(gen_core::Error::backend)
    }
    fn configure_decode(
        &mut self,
        edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> gen_core::Result<()> {
        if geometry != self.geometry || edge != DECODE_TILE_EDGE || overlap != DECODE_OVERLAP {
            Err(gen_core::Error::Unsupported(
                "SANA decode parameters or geometry were not admitted".into(),
            ))
        } else {
            Ok(())
        }
    }
    fn configure_attention(&mut self, size: u32) -> gen_core::Result<()> {
        if ATTENTION_CHUNK_SIZES.contains(&size) {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(
                "SANA attention budget was not admitted".into(),
            ))
        }
    }
    fn materialize_transformer_window(&mut self, first: u32, count: u32) -> gen_core::Result<()> {
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
