//! Exact, estimate-backed Resident authority for SceneWorks' public Candle SVD-XT route.

use std::cell::RefCell;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use candle_gen::gen_core::{
    self, GenerationRequest, LoadShape, LoadSpec, MemoryAssetFacts, MemoryBackendRealization,
    MemoryContractSurfaceSelector, MemoryContractSurfaceSpec, MemoryContractSurfaceTier,
    MemoryFormulaKind, MemoryFormulaVariable, MemoryGeometry, MemoryLifecycleCapabilities,
    MemoryNumericTier, MemoryParameterRanges, MemoryPhase, MemoryProviderContract,
    MemoryRequestScope, MemoryRunContext, MemoryRunOutcome, MemorySafetyDecision, MemoryStrategy,
    MemoryStrategyCapability, MemoryStrategySupport, Precision,
    ResidentOnlyMemoryContractRegistration, ResidentRequestMemory, WeightsSource,
};
use sha2::{Digest, Sha256};

pub const CANONICAL_REPOSITORY: &str = "stabilityai/stable-video-diffusion-img2vid-xt";
pub const PUBLIC_GEOMETRIES: &[(u32, u32)] = &[(1024, 576), (576, 1024)];
pub const PUBLIC_FPS: &[u32] = &[6, 7, 8, 10, 12, 25];
pub const NATIVE_SCHEDULE: &str = "svd-edm-karras-v-prediction";
const RECEIPT_VERSION: &str = "svd-candle-resident-v1";
const EXPECTED_TENSOR_INVENTORY: &[&str] = &[
    "image_encoder/model.fp16.safetensors",
    "image_encoder/model.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
];
const SELECTED_FILES: &[(&str, Component)] = &[
    ("image_encoder/model.safetensors", Component::Conditioning),
    (
        "unet/diffusion_pytorch_model.safetensors",
        Component::Transformer,
    ),
    (
        "vae/diffusion_pytorch_model.safetensors",
        Component::Decoder,
    ),
];

#[derive(Clone, Copy)]
enum Component {
    Conditioning,
    Transformer,
    Decoder,
}

#[derive(Clone, Debug)]
struct SealedFile {
    relative: &'static str,
    pin: gen_core::PinnedWeightsFile,
    digest: String,
    resident_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct PreparedSvdMemory {
    pub contract: MemoryProviderContract,
    pub artifact_identity: String,
    pub revision: String,
    tier: MemoryNumericTier,
    root: PathBuf,
    files: Vec<SealedFile>,
    /// Whether the full content digest of every sealed file has already been verified for this
    /// loaded artifact. `components()` re-runs `ensure_unchanged` on every generate, and a full
    /// re-hash of the multi-GB snapshot per request is not what that guard is for: the per-file
    /// stat pin (size + mtime + ctime + inode + symlink target) is the cheap identity check, and it
    /// is re-run every time. The expensive byte-for-byte digest runs once, at admission.
    digest_verified: Arc<AtomicBool>,
}

fn sha256_file(path: &Path) -> gen_core::Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn tensor_inventory(root: &Path) -> gen_core::Result<Vec<String>> {
    let mut inventory = Vec::new();
    for sub in ["image_encoder", "unet", "vae"] {
        let dir = root.join(sub);
        if !dir.is_dir() {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: snapshot is missing {sub}/",
                crate::MODEL_ID
            )));
        }
        for entry in std::fs::read_dir(&dir)? {
            let path = entry?.path();
            if path.extension().and_then(|value| value.to_str()) == Some("safetensors") {
                inventory.push(format!(
                    "{sub}/{}",
                    path.file_name()
                        .and_then(|value| value.to_str())
                        .ok_or_else(|| gen_core::Error::Unsupported(format!(
                            "{}: non-UTF8 tensor inventory",
                            crate::MODEL_ID
                        )))?
                ));
            }
        }
    }
    inventory.sort();
    Ok(inventory)
}

fn repository_revision(root: &Path) -> gen_core::Result<String> {
    const REPO_DIR: &str = "models--stabilityai--stable-video-diffusion-img2vid-xt";
    let canonical = std::fs::canonicalize(root)?;
    let components = canonical
        .components()
        .map(|component| component.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    if let Some(repo) = components.iter().position(|part| part == REPO_DIR) {
        if components.get(repo + 1).map(String::as_str) == Some("snapshots")
            && repo + 3 == components.len()
        {
            let revision = components.get(repo + 2).cloned().unwrap_or_default();
            if revision.len() == 40
                && revision
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Ok(revision);
            }
        }
    }

    let mut identity = Sha256::new();
    identity.update(b"svd-local-snapshot-path-v1");
    identity.update(canonical.as_os_str().as_encoded_bytes());
    Ok(format!("local:{:x}", identity.finalize()))
}

fn validate_load_spec(spec: &LoadSpec) -> gen_core::Result<&Path> {
    let WeightsSource::Dir(root) = &spec.weights else {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: exact SVD memory authority requires directory weights",
            crate::MODEL_ID
        )));
    };
    if std::env::var_os("SVD_FORCE_F16").is_some()
        || std::env::var_os("SVD_FORCE_BF16").is_some()
        || spec.precision != Precision::Bf16
        || spec.quantize.is_some()
        || !spec.adapters.is_empty()
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
        || spec.offload_policy != gen_core::OffloadPolicy::Resident
        || spec.resolved_route.as_deref() != Some(crate::MODEL_ID)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: load is outside the unquantized Resident SVD-XT surface",
            crate::MODEL_ID
        )));
    }
    gen_core::reject_unknown_components(spec, &[], crate::MODEL_ID)?;
    Ok(root)
}

pub fn prepare_load_spec(spec: &mut LoadSpec) -> gen_core::Result<()> {
    let root = validate_load_spec(spec)?.to_path_buf();
    let actual = tensor_inventory(&root)?;
    let expected = EXPECTED_TENSOR_INVENTORY
        .iter()
        .map(|value| (*value).to_owned())
        .collect::<Vec<_>>();
    if actual != expected {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: incomplete or extra SVD tensor inventory (expected {expected:?}, got {actual:?})",
            crate::MODEL_ID
        )));
    }
    let pins = EXPECTED_TENSOR_INVENTORY
        .iter()
        .map(|relative| gen_core::PinnedWeightsFile::pin(root.join(relative)))
        .collect::<gen_core::Result<Vec<_>>>()?;
    spec.prepare_with_file_pins(pins)
}

impl PreparedSvdMemory {
    pub fn prepare(spec: &LoadSpec) -> gen_core::Result<Self> {
        let root = validate_load_spec(spec)?;
        spec.validate_prepared_file_pins()?;
        let revision = repository_revision(root)?;
        let actual = tensor_inventory(root)?;
        let expected = EXPECTED_TENSOR_INVENTORY
            .iter()
            .map(|value| (*value).to_owned())
            .collect::<Vec<_>>();
        if actual != expected {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: incomplete or extra SVD tensor inventory",
                crate::MODEL_ID
            )));
        }

        let mut facts = MemoryAssetFacts::default();
        let mut files = Vec::new();
        let mut identity = Sha256::new();
        identity.update(RECEIPT_VERSION.as_bytes());
        identity.update(CANONICAL_REPOSITORY.as_bytes());
        identity.update(revision.as_bytes());
        for &(relative, component) in SELECTED_FILES {
            let path = root.join(relative);
            let absolute = std::path::absolute(&path)?;
            let pin = spec
                .prepared_file_pins()
                .get(&absolute)
                .cloned()
                .ok_or_else(|| {
                    gen_core::Error::Unsupported(format!(
                        "{}: prepared receipt is missing {relative}",
                        crate::MODEL_ID
                    ))
                })?;
            pin.ensure_unchanged()?;
            let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(&path)?;
            if headers.is_empty() || headers.iter().any(|header| !header.is_float()) {
                return Err(gen_core::Error::Unsupported(format!(
                    "{}: {relative} is empty or contains a non-float tensor",
                    crate::MODEL_ID
                )));
            }
            let resident_bytes = headers.iter().try_fold(0_u64, |sum, header| {
                sum.checked_add(header.materialized_bytes(4)?)
                    .ok_or_else(|| {
                        gen_core::Error::Msg(format!("{}: resident byte overflow", crate::MODEL_ID))
                    })
            })?;
            if resident_bytes == 0 {
                return Err(gen_core::Error::Unsupported(format!(
                    "{}: {relative} has zero resident bytes",
                    crate::MODEL_ID
                )));
            }
            let digest = sha256_file(&path)?;
            identity.update(relative.as_bytes());
            identity.update(digest.as_bytes());
            identity.update(resident_bytes.to_le_bytes());
            match component {
                Component::Conditioning => facts.conditioning_bytes = resident_bytes,
                Component::Transformer => facts.transformer_bytes = resident_bytes,
                Component::Decoder => facts.decoder_bytes = resident_bytes,
            }
            files.push(SealedFile {
                relative,
                pin,
                digest,
                resident_bytes,
            });
        }
        facts.base_bytes = facts
            .conditioning_bytes
            .checked_add(facts.transformer_bytes)
            .and_then(|bytes| bytes.checked_add(facts.decoder_bytes))
            .ok_or_else(|| {
                gen_core::Error::Msg(format!("{}: base byte overflow", crate::MODEL_ID))
            })?;
        let artifact_identity = format!("{:x}", identity.finalize());
        let tier = MemoryNumericTier {
            // The LoadSpec uses Bf16 as the dense product sentinel, but Candle SVD deliberately
            // materializes the image encoder, UNet, and VAE as F32.
            precision: Precision::Fp32,
            quant: None,
            component_precision_floors: &[],
        };
        let prepared = Self {
            contract: build_contract(spec, facts),
            artifact_identity,
            revision,
            tier,
            root: root.to_path_buf(),
            files,
            digest_verified: Arc::new(AtomicBool::new(false)),
        };
        prepared.ensure_unchanged()?;
        Ok(prepared)
    }

    pub fn ensure_unchanged(&self) -> gen_core::Result<()> {
        if std::env::var_os("SVD_FORCE_F16").is_some()
            || std::env::var_os("SVD_FORCE_BF16").is_some()
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: Candle dtype override appeared after admission",
                crate::MODEL_ID
            )));
        }
        if tensor_inventory(&self.root)?
            != EXPECTED_TENSOR_INVENTORY
                .iter()
                .map(|value| (*value).to_owned())
                .collect::<Vec<_>>()
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: SVD tensor inventory changed after admission",
                crate::MODEL_ID
            )));
        }
        // Verify content digests only until the artifact has been proven once; the stat pin below
        // is re-checked on every call and is what detects a swapped or rewritten file thereafter.
        let verify_digests = !self.digest_verified.load(Ordering::Acquire);
        for file in &self.files {
            file.pin.ensure_unchanged()?;
            let path = self.root.join(file.relative);
            if verify_digests && sha256_file(&path)? != file.digest {
                return Err(gen_core::Error::Unsupported(format!(
                    "{}: {} changed after admission",
                    crate::MODEL_ID,
                    file.relative
                )));
            }
            let headers = gen_core::weightsmeta::safetensors_path_tensor_headers(path)?;
            let resident = headers.iter().try_fold(0_u64, |sum, header| {
                sum.checked_add(header.materialized_bytes(4)?)
                    .ok_or_else(|| {
                        gen_core::Error::Msg(format!("{}: resident byte overflow", crate::MODEL_ID))
                    })
            })?;
            if resident != file.resident_bytes {
                return Err(gen_core::Error::Unsupported(format!(
                    "{}: {} tensor geometry changed after admission",
                    crate::MODEL_ID,
                    file.relative
                )));
            }
        }
        if verify_digests {
            self.digest_verified.store(true, Ordering::Release);
        }
        Ok(())
    }

    pub fn numeric_tier(&self) -> MemoryNumericTier {
        self.tier
    }
}

/// Snapshot-gated architecture facts for the SVD-XT route (epic SC-22657, E2).
///
/// SVD's denoiser is a **UNet**, not a transformer stack, and this crate's loader does not parse
/// `unet/config.json`: it builds [`crate::config::UnetConfig`] and [`crate::config::VaeConfig`] from
/// their `Default` impls (a transcription of the published `stabilityai/...-img2vid-xt` JSON), so
/// those are the structures read here. Three of the eight axes are structurally absent as a result,
/// and each is declared absent rather than guessed:
///
/// * `attention_heads` — the UNet's head count is **per stage** (`[5, 10, 20, 20]`). There is no
///   single uniform head count to publish, and naming any one of them would describe one block.
/// * `transformer_blocks` — a UNet is not a uniform stack of transformer blocks; it is a
///   down/mid/up ladder of res-blocks with attention interleaved.
/// * `patch_size` — the UNet consumes the latent directly; there is no patchification stage.
///
/// `head_dim` survives because it is uniform even though the head *count* is not: every stage's
/// `block_out_channels[i] / num_attention_heads[i]` is 64. The quotient is computed across all
/// stages and published only if they agree, so a config that broke that property would decline the
/// axis rather than pick a stage.
///
/// `vae_temporal_scale` is `Some(1)`, and that is a real value rather than a placeholder: SVD ships
/// `AutoencoderKLTemporalDecoder`, a 2-D encoder with a temporal decoder, so pixel frames map 1:1
/// onto latent frames (`SvdVae::VAE_TILING { temporal_scale: 1 }`).
///
/// `activation_dtype_width` comes from [`crate::dense_dtype`] — the same function the loader calls
/// to pick the dtype it materializes — so the declared width tracks the `SVD_FORCE_F16` /
/// `SVD_FORCE_BF16` escapes instead of restating the f32 default.
///
/// A weights-free contract — no materialized snapshot directory — publishes
/// `MemoryArchitectureFacts::default()`.
fn architecture_facts(spec: &LoadSpec) -> gen_core::MemoryArchitectureFacts {
    use candle_gen::architecture_facts as af;

    if af::snapshot_root(spec).is_none() {
        return gen_core::MemoryArchitectureFacts::default();
    }
    let unet = crate::config::UnetConfig::default();
    let vae = crate::config::VaeConfig::default();
    let mut per_stage = unet
        .block_out_channels
        .iter()
        .zip(&unet.num_attention_heads)
        .map(|(channels, heads)| af::head_dim(af::declared(*channels), af::declared(*heads)));
    // Uniform across every stage or nothing: `first` seeds the comparison so an empty ladder is
    // `None` rather than a vacuous agreement.
    let head_dim = match per_stage.next() {
        Some(first) if first.is_some() && per_stage.all(|stage| stage == first) => first,
        _ => None,
    };
    gen_core::MemoryArchitectureFacts {
        // Structurally absent: the UNet's head count differs per stage (5/10/20/20).
        attention_heads: None,
        head_dim,
        // Structurally absent: a UNet is not a uniform stack of transformer blocks.
        transformer_blocks: None,
        // Structurally absent: the UNet consumes the latent directly; there is no patchification.
        patch_size: None,
        latent_channels: af::declared(vae.latent_channels),
        vae_spatial_scale: u32::try_from(crate::vae::SvdVae::VAE_TILING.spatial_scale)
            .ok()
            .filter(|scale| *scale != 0),
        // 1 is a real value here: the temporal decoder emits one frame per latent frame.
        vae_temporal_scale: u32::try_from(crate::vae::SvdVae::VAE_TILING.temporal_scale)
            .ok()
            .filter(|scale| *scale != 0),
        activation_dtype_width: af::dtype_width(crate::dense_dtype()),
    }
}

fn build_contract(spec: &LoadSpec, facts: MemoryAssetFacts) -> MemoryProviderContract {
    let phases = vec![
        MemoryPhase::Conditioning,
        MemoryPhase::Denoise,
        MemoryPhase::Decode,
    ];
    MemoryProviderContract {
        architecture_facts: architecture_facts(spec),
        provider_id: crate::MODEL_ID.to_owned(),
        backend: MemoryBackendRealization::CandleCuda {
            device_residency: true,
            host_backed_weights: true,
            host_to_device_block_materialization: false,
            block_materialization: gen_core::MemoryWindowMaterialization::DeviceFormatTransfer,
        },
        strategies: MemoryStrategy::ALL
            .into_iter()
            .map(|strategy| MemoryStrategyCapability {
                strategy,
                support: if strategy == MemoryStrategy::Resident {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                },
                parameters: MemoryParameterRanges::default(),
            })
            .collect(),
        decode_geometry_policy_authoritative: false,
        pid_decode_routes: None,
        load_shape: spec.load_shape,
        additional_prerequisites: Vec::new(),
        default_engagement_exclusions: Vec::new(),
        resident_request_memory: ResidentRequestMemory::ExplicitResident,
        lifecycle: MemoryLifecycleCapabilities {
            phases: phases.clone(),
            synchronized_phase_release: true,
            decode_tiling: false,
            attention_chunking: false,
            transformer_window_materialization: false,
        },
        formula: MemoryFormulaKind::PhaseEnvelope {
            phases,
            variables: vec![
                MemoryFormulaVariable::AssetBytes,
                MemoryFormulaVariable::PixelCount,
                MemoryFormulaVariable::FrameCount,
                MemoryFormulaVariable::BatchCount,
            ],
        },
        calibration: None,
        asset_facts: facts,
        runtime: gen_core::MemoryRuntimeSemantics::default(),
    }
}

pub fn memory_strategy_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    PreparedSvdMemory::prepare(spec).map(|prepared| prepared.contract)
}

/// Weights-free registry proof for the deliberately dense, resident-only SVD-XT surface.
/// Production admission still uses [`memory_strategy_contract`] and its sealed physical facts.
fn weights_free_resident_contract(spec: &LoadSpec) -> gen_core::Result<MemoryProviderContract> {
    if spec.precision != Precision::Bf16
        || spec.quantize.is_some()
        || spec.offload_policy != gen_core::OffloadPolicy::Resident
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: resident witness only covers the dense bf16 catalog surface",
            crate::MODEL_ID
        )));
    }
    Ok(build_contract(spec, MemoryAssetFacts::default()))
}

fn resident_only_surface_specs() -> Vec<MemoryContractSurfaceSpec> {
    [
        LoadShape::EagerMaterialization,
        LoadShape::DeferredMaterialization,
    ]
    .into_iter()
    .map(|load_shape| MemoryContractSurfaceSpec {
        selector: MemoryContractSurfaceSelector {
            tier: MemoryContractSurfaceTier::Bf16,
            offload_policy: gen_core::OffloadPolicy::Resident,
            load_shape,
        },
        spec: LoadSpec::new(WeightsSource::Dir(
            "/__sceneworks_memory_contract_surface__".into(),
        ))
        .with_offload_policy(gen_core::OffloadPolicy::Resident)
        .with_load_shape(load_shape),
    })
    .collect()
}

fn one_reference(request: &GenerationRequest) -> gen_core::Result<&gen_core::Image> {
    if request.conditioning.len() != 1 {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: public SVD memory route requires exactly one conditioning carrier",
            crate::MODEL_ID
        )));
    }
    match &request.conditioning[0] {
        gen_core::Conditioning::Reference {
            image,
            strength: None,
        } => Ok(image),
        _ => Err(gen_core::Error::Unsupported(format!(
            "{}: public SVD memory route requires one strength-free Reference",
            crate::MODEL_ID
        ))),
    }
}

pub fn validate_memory_request(request: &GenerationRequest) -> gen_core::Result<()> {
    let reference = one_reference(request)?;
    let expected_pixels = gen_core::imageops::checked_image_buffer_len(
        reference.width as usize,
        reference.height as usize,
        3,
    );
    let unsupported_sampler = request
        .sampler
        .as_deref()
        .is_some_and(|sampler| !candle_gen::curated_sampler_names().contains(&sampler));
    let frames = request.frames.unwrap_or_default();
    let steps = request.steps.unwrap_or_default();
    let conditioning_fps = request.conditioning_fps.unwrap_or_default();
    let motion = request.motion_bucket_id.unwrap_or(f32::NAN);
    let noise = request.noise_aug_strength.unwrap_or(f32::NAN);
    let chunk = request.decode_chunk_size.unwrap_or_default();
    if request.video_mode.as_deref() != Some("image_to_video")
        || !request.prompt.is_empty()
        || request.negative_prompt.is_some()
        || request.guidance.is_some()
        || request.true_cfg.is_some()
        || request.audio.is_some()
        || request.phases.is_some()
        || request.count != 1
        || !PUBLIC_GEOMETRIES.contains(&(request.width, request.height))
        || !(1..=25).contains(&frames)
        || !PUBLIC_FPS.contains(&request.fps.unwrap_or_default())
        || !(1..=30).contains(&conditioning_fps)
        || !motion.is_finite()
        || motion.fract() != 0.0
        || !(1.0..=255.0).contains(&motion)
        || !noise.is_finite()
        || noise < 0.0
        || !(1..=64).contains(&chunk)
        || !(1..=80).contains(&steps)
        || request.seed.is_none()
        || request.scheduler.is_some()
        || request.scheduler_shift.is_some()
        || unsupported_sampler
        || request.use_pid
        || request.enhance_prompt
        || request.use_uncensored_enhancer
        || request
            .memory
            .is_some_and(|memory| memory != gen_core::GenerationMemory::default())
        || reference.width != request.width
        || reference.height != request.height
        || reference.width > crate::MAX_REFERENCE_DIM
        || reference.height > crate::MAX_REFERENCE_DIM
        || expected_pixels != Some(reference.pixels.len())
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: request left the exact public pre-fit one-still SVD envelope",
            crate::MODEL_ID
        )));
    }
    Ok(())
}

pub fn request_evidence_revision(
    prepared: &PreparedSvdMemory,
    request: &GenerationRequest,
) -> gen_core::Result<String> {
    validate_memory_request(request)?;
    let reference = one_reference(request)?;
    let mut hash = Sha256::new();
    hash.update(RECEIPT_VERSION.as_bytes());
    hash.update(prepared.artifact_identity.as_bytes());
    hash.update(prepared.revision.as_bytes());
    hash.update(b"candle:f32-image-encoder-unet-vae");
    hash.update(request.width.to_le_bytes());
    hash.update(request.height.to_le_bytes());
    hash.update(request.frames.unwrap().to_le_bytes());
    hash.update(request.fps.unwrap().to_le_bytes());
    hash.update(request.conditioning_fps.unwrap().to_le_bytes());
    hash.update(request.motion_bucket_id.unwrap().to_bits().to_le_bytes());
    hash.update(request.noise_aug_strength.unwrap().to_bits().to_le_bytes());
    hash.update(request.decode_chunk_size.unwrap().to_le_bytes());
    hash.update(request.steps.unwrap().to_le_bytes());
    hash.update(request.seed.unwrap().to_le_bytes());
    hash.update(
        request
            .sampler
            .as_deref()
            .unwrap_or("native_euler")
            .as_bytes(),
    );
    hash.update(NATIVE_SCHEDULE.as_bytes());
    hash.update(reference.width.to_le_bytes());
    hash.update(reference.height.to_le_bytes());
    hash.update(Sha256::digest(&reference.pixels));
    Ok(format!(
        "{RECEIPT_VERSION}:{}:{:x}",
        prepared.artifact_identity,
        hash.finalize()
    ))
}

fn validate_context(
    prepared: &PreparedSvdMemory,
    context: &MemoryRunContext,
) -> gen_core::Result<()> {
    prepared.ensure_unchanged()?;
    let geometry = context.geometry;
    let artifact_prefix = format!("{RECEIPT_VERSION}:{}:", prepared.artifact_identity);
    if context.selection.strategy != MemoryStrategy::Resident
        || context.mode.as_key() != "image_to_video"
        || geometry.batch != 1
        || geometry.reference_count != 1
        || !context.has_reference
        || !PUBLIC_GEOMETRIES.contains(&(geometry.width, geometry.height))
        || !(1..=25).contains(&geometry.frames)
        || context.overlay.is_some()
        || context.use_pid
        || context.has_phases
        || !context.evidence_revision.starts_with(&artifact_prefix)
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{}: crossed SVD Resident context",
            crate::MODEL_ID
        )));
    }
    match gen_core::standard_memory_strategy_safety_check(
        &prepared.contract,
        context,
        Some(prepared.tier),
        None,
    ) {
        MemorySafetyDecision::Accept => Ok(()),
        MemorySafetyDecision::Reject { reason } => Err(gen_core::Error::Unsupported(reason)),
    }
}

pub fn safety_check(
    prepared: &PreparedSvdMemory,
    context: &MemoryRunContext,
) -> MemorySafetyDecision {
    match validate_context(prepared, context) {
        Ok(()) => MemorySafetyDecision::Accept,
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    }
}

thread_local! {
    static ACTIVE_EVIDENCE: RefCell<Option<String>> = const { RefCell::new(None) };
}

pub fn validate_active_request(
    prepared: &PreparedSvdMemory,
    request: &GenerationRequest,
) -> gen_core::Result<()> {
    if request.memory.is_none() {
        return Ok(());
    }
    let expected = request_evidence_revision(prepared, request)?;
    ACTIVE_EVIDENCE.with(|active| {
        if active.borrow().as_deref() == Some(expected.as_str()) {
            Ok(())
        } else {
            Err(gen_core::Error::Unsupported(format!(
                "{}: generation request does not match its admitted physical/request identity",
                crate::MODEL_ID
            )))
        }
    })
}

struct SvdRequestScope {
    core: candle_gen::request_scope::CandleRequestScopeCore,
    prepared: PreparedSvdMemory,
    evidence_revision: String,
    armed: bool,
}

impl Drop for SvdRequestScope {
    fn drop(&mut self) {
        if self.armed {
            ACTIVE_EVIDENCE.with(|active| *active.borrow_mut() = None);
        }
    }
}

impl MemoryRequestScope for SvdRequestScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> gen_core::Result<()> {
        let actual = request_evidence_revision(&self.prepared, request)?;
        if actual != self.evidence_revision {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: request axes crossed after admission",
                crate::MODEL_ID
            )));
        }
        self.core.configure_request(request)?;
        ACTIVE_EVIDENCE.with(|active| {
            if active.borrow().is_some() {
                return Err(gen_core::Error::Unsupported(format!(
                    "{}: another SVD request is active on this thread",
                    crate::MODEL_ID
                )));
            }
            *active.borrow_mut() = Some(actual);
            Ok(())
        })?;
        self.armed = true;
        Ok(())
    }

    fn enter_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.core.enter_phase(phase)
    }
    fn leave_phase(&mut self, phase: MemoryPhase) -> gen_core::Result<()> {
        self.core.leave_phase(phase)
    }
    fn configure_decode(
        &mut self,
        edge: u32,
        overlap: u32,
        geometry: MemoryGeometry,
    ) -> gen_core::Result<()> {
        self.core.configure_decode(edge, overlap, geometry)
    }
    fn configure_attention(&mut self, size: u32) -> gen_core::Result<()> {
        self.core.configure_attention(size)
    }
    fn materialize_transformer_window(&mut self, first: u32, count: u32) -> gen_core::Result<()> {
        self.core.materialize_transformer_window(first, count)
    }
    fn finish(&mut self, outcome: MemoryRunOutcome) -> gen_core::Result<()> {
        let result = self.core.finish(outcome);
        if self.armed {
            ACTIVE_EVIDENCE.with(|active| *active.borrow_mut() = None);
            self.armed = false;
        }
        result
    }
}

pub fn begin_request<'a>(
    prepared: &'a PreparedSvdMemory,
    device: candle_gen::candle_core::Device,
    context: &MemoryRunContext,
) -> gen_core::Result<Option<Box<dyn MemoryRequestScope + 'a>>> {
    validate_context(prepared, context)?;
    let mut config = candle_gen::request_scope::CandleRequestScopeConfig::new(
        crate::MODEL_ID,
        device,
        context.geometry,
        prepared.contract.generation_memory(&context.selection),
        false,
        0,
        |_pid, _edge, _overlap| {
            Err(gen_core::Error::Unsupported(format!(
                "{}: bounded decode is Missing",
                crate::MODEL_ID
            )))
        },
    )?;
    config.default_frames = context.geometry.frames;
    Ok(Some(Box::new(SvdRequestScope {
        core: candle_gen::request_scope::CandleRequestScopeCore::new(config),
        prepared: prepared.clone(),
        evidence_revision: context.evidence_revision.clone(),
        armed: false,
    })))
}

pub const MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: crate::MODEL_ID,
    contract: memory_strategy_contract,
    safety_check: |spec, contract, context| match PreparedSvdMemory::prepare(spec) {
        Ok(prepared) if prepared.contract == *contract => safety_check(&prepared, context),
        Ok(_) => MemorySafetyDecision::Reject {
            reason: format!(
                "{}: provider contract crossed its sealed artifact",
                crate::MODEL_ID
            ),
        },
        Err(error) => MemorySafetyDecision::Reject {
            reason: error.to_string(),
        },
    },
};

pub const RESIDENT_ONLY_WITNESS: ResidentOnlyMemoryContractRegistration =
    ResidentOnlyMemoryContractRegistration {
        provider_id: crate::MODEL_ID,
        contract: weights_free_resident_contract,
        surface_specs: resident_only_surface_specs,
    };

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{Conditioning, Image, MemoryStrategySupport, OffloadPolicy, Quant};

    fn write_safetensors(path: &Path, shape: &[usize], fp16: bool) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let width = if fp16 { 2 } else { 4 };
        let bytes = shape.iter().product::<usize>() * width;
        let mut json = serde_json::to_vec(&serde_json::json!({
            "weight": {"dtype": if fp16 { "F16" } else { "F32" }, "shape": shape, "data_offsets": [0, bytes]}
        })).unwrap();
        while !json.len().is_multiple_of(8) {
            json.push(b' ');
        }
        let mut output = (json.len() as u64).to_le_bytes().to_vec();
        output.extend(json);
        output.resize(output.len() + bytes, 7);
        std::fs::write(path, output).unwrap();
    }

    fn fixture() -> (tempfile::TempDir, LoadSpec) {
        fixture_under("models--stabilityai--stable-video-diffusion-img2vid-xt")
    }

    /// The same complete, readable SVD snapshot, published under an arbitrary repository dir. Used
    /// to build a *real* decoy — a crossed-repo assertion against a path that does not exist proves
    /// only that `canonicalize` fails.
    fn fixture_under(repo_dir: &str) -> (tempfile::TempDir, LoadSpec) {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp
            .path()
            .join(repo_dir)
            .join("snapshots")
            .join("0123456789abcdef0123456789abcdef01234567");
        for relative in EXPECTED_TENSOR_INVENTORY {
            let shape: &[usize] = if relative.starts_with("image_encoder") {
                &[2, 3]
            } else if relative.starts_with("unet") {
                &[3, 5]
            } else {
                &[5, 7]
            };
            write_safetensors(&root.join(relative), shape, relative.contains("fp16"));
        }
        let mut spec = LoadSpec::new(WeightsSource::Dir(root))
            .with_resolved_route(crate::MODEL_ID)
            .with_offload_policy(OffloadPolicy::Resident);
        prepare_load_spec(&mut spec).unwrap();
        (tmp, spec)
    }

    #[test]
    fn physical_facts_are_all_f32_and_resident_only() {
        let (_tmp, spec) = fixture();
        let prepared = PreparedSvdMemory::prepare(&spec).unwrap();
        assert_eq!(prepared.contract.asset_facts.conditioning_bytes, 24);
        assert_eq!(prepared.contract.asset_facts.transformer_bytes, 60);
        assert_eq!(prepared.contract.asset_facts.decoder_bytes, 140);
        assert_eq!(prepared.contract.asset_facts.base_bytes, 224);
        assert_eq!(prepared.numeric_tier().precision, Precision::Fp32);
        for strategy in MemoryStrategy::ALL {
            assert_eq!(
                prepared.contract.capability(strategy).unwrap().support,
                if strategy == MemoryStrategy::Resident {
                    MemoryStrategySupport::Implemented
                } else {
                    MemoryStrategySupport::Missing
                }
            );
        }
    }

    #[test]
    fn weights_free_witness_covers_only_dense_resident_load_shapes() {
        let surfaces = resident_only_surface_specs();
        assert_eq!(surfaces.len(), 2);
        for surface in surfaces {
            let contract = weights_free_resident_contract(&surface.spec).unwrap();
            assert_eq!(contract.provider_id, crate::MODEL_ID);
            assert_eq!(contract.asset_facts, MemoryAssetFacts::default());
            assert!(contract.conformance_errors().is_empty());
            assert!(contract.strategies.iter().all(|capability| {
                capability.support
                    == if capability.strategy == MemoryStrategy::Resident {
                        MemoryStrategySupport::Implemented
                    } else {
                        MemoryStrategySupport::Missing
                    }
            }));
        }
        let witness_spec = || {
            LoadSpec::new(WeightsSource::Dir(
                "/__sceneworks_memory_contract_surface__".into(),
            ))
        };
        assert!(weights_free_resident_contract(&witness_spec().with_quant(Quant::Q4)).is_err());
        assert!(weights_free_resident_contract(
            &witness_spec().with_offload_policy(OffloadPolicy::Sequential)
        )
        .is_err());
    }

    /// AC (epic SC-22657, E2): the SVD contract publishes the architecture axes of the config the
    /// loader actually instantiates, declares the three that a UNet structurally lacks, and the
    /// weights-free surface publishes none of them.
    #[test]
    fn architecture_facts_match_the_loader_config_and_pass_conformance() {
        let (_tmp, spec) = fixture();
        let contract = build_contract(&spec, MemoryAssetFacts::default());
        assert_eq!(
            contract.architecture_facts,
            gen_core::MemoryArchitectureFacts {
                // Structurally absent: `UnetConfig::num_attention_heads` is per stage
                // (`[5, 10, 20, 20]`) — there is no single uniform head count to declare.
                attention_heads: None,
                // Uniform even though the head count is not: every stage's
                // `block_out_channels[i] / num_attention_heads[i]` is 64.
                head_dim: Some(64),
                // Structurally absent: a UNet is not a uniform stack of transformer blocks.
                transformer_blocks: None,
                // Structurally absent: the UNet consumes the latent directly; no patchification.
                patch_size: None,
                // `VaeConfig::latent_channels`.
                latent_channels: Some(4),
                // `SvdVae::VAE_TILING.spatial_scale` (four `block_out_channels` stages).
                vae_spatial_scale: Some(8),
                // A real value, not a placeholder: `AutoencoderKLTemporalDecoder` is a 2-D encoder
                // with a temporal decoder, so pixel frames map 1:1 onto latent frames.
                vae_temporal_scale: Some(1),
                // `crate::dense_dtype()` — f32 by default; the only dtype that renders correctly.
                activation_dtype_width: Some(4),
            }
        );
        assert!(contract.architecture_facts.has_declared_architecture_axis());
        gen_core_testkit::assert_memory_contract_facts_conform(&contract);

        // The registry surface names a sentinel that is not on disk.
        for surface in resident_only_surface_specs() {
            let weights_free = weights_free_resident_contract(&surface.spec).unwrap();
            assert!(weights_free.architecture_facts.is_empty());
        }
    }

    #[test]
    fn mutation_and_inventory_drift_fail_closed() {
        let (_tmp, spec) = fixture();
        let prepared = PreparedSvdMemory::prepare(&spec).unwrap();
        let root = match &spec.weights {
            WeightsSource::Dir(root) => root,
            _ => unreachable!(),
        };
        let path = root.join(SELECTED_FILES[0].0);
        let mut bytes = std::fs::read(&path).unwrap();
        *bytes.last_mut().unwrap() ^= 1;
        std::fs::write(&path, bytes).unwrap();
        assert!(prepared.ensure_unchanged().is_err());

        let (_tmp, spec) = fixture();
        let root = match &spec.weights {
            WeightsSource::Dir(root) => root,
            _ => unreachable!(),
        };
        write_safetensors(&root.join("vae/extra.safetensors"), &[1], false);
        assert!(PreparedSvdMemory::prepare(&spec).is_err());
    }

    /// `components()` runs `ensure_unchanged` on every generate. The stat pin re-checks every
    /// call; the full byte-for-byte digest of a multi-GB snapshot must not.
    #[test]
    fn the_snapshot_content_digest_is_hashed_once_per_loaded_artifact() {
        let (_tmp, spec) = fixture();
        let prepared = PreparedSvdMemory::prepare(&spec).unwrap();
        let root = match &spec.weights {
            WeightsSource::Dir(root) => root,
            _ => unreachable!(),
        };
        let path = root.join(SELECTED_FILES[0].0);

        // Rewrite the file with the SAME bytes: the content digest is unchanged, so only the stat
        // pin can notice. It does — the guard stays closed without re-hashing.
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        assert!(prepared.ensure_unchanged().is_err());

        // A clone shares the verification state (the request scope owns a clone of the receipt).
        let (_tmp, spec) = fixture();
        let prepared = PreparedSvdMemory::prepare(&spec).unwrap();
        let scoped = prepared.clone();
        prepared.ensure_unchanged().unwrap();
        scoped.ensure_unchanged().unwrap();
        assert!(prepared.digest_verified.load(Ordering::Acquire));
        assert!(scoped.digest_verified.load(Ordering::Acquire));
    }

    #[test]
    fn repository_tier_overlay_and_request_identity_are_exact() {
        let (_tmp, spec) = fixture();
        let prepared = PreparedSvdMemory::prepare(&spec).unwrap();
        let mut request = GenerationRequest {
            prompt: String::new(),
            width: 1024,
            height: 576,
            count: 1,
            seed: Some(17),
            steps: Some(25),
            sampler: Some("heun".to_owned()),
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 1024,
                    height: 576,
                    pixels: vec![7; 1024 * 576 * 3],
                },
                strength: None,
            }],
            frames: Some(25),
            fps: Some(7),
            video_mode: Some("image_to_video".to_owned()),
            conditioning_fps: Some(7),
            motion_bucket_id: Some(127.0),
            noise_aug_strength: Some(0.02),
            decode_chunk_size: Some(8),
            ..Default::default()
        };
        let baseline = request_evidence_revision(&prepared, &request).unwrap();
        request.seed = Some(18);
        assert_ne!(
            baseline,
            request_evidence_revision(&prepared, &request).unwrap()
        );
        request.conditioning.push(request.conditioning[0].clone());
        assert!(validate_memory_request(&request).is_err());

        let mut invalid = request.clone();
        invalid.conditioning.pop();
        invalid.conditioning.push(Conditioning::Reference {
            image: Image {
                width: 1024,
                height: 576,
                pixels: vec![7; 1024 * 576 * 3 - 1],
            },
            strength: None,
        });
        assert!(validate_memory_request(&invalid).is_err());
        let mut invalid = request.clone();
        invalid.sampler = Some("not-a-sampler".into());
        assert!(validate_memory_request(&invalid).is_err());

        let mut quantized = spec.clone();
        quantized.quantize = Some(Quant::Q4);
        assert!(PreparedSvdMemory::prepare(&quantized).is_err());
        let mut overlay = spec.clone();
        overlay.ip_adapter = Some(WeightsSource::Dir(PathBuf::from("overlay")));
        assert!(PreparedSvdMemory::prepare(&overlay).is_err());
        // A crossed repository, as a REAL, readable, byte-identical snapshot. The old assertion
        // pointed at a nonexistent path, so it passed on `canonicalize` failing and said nothing
        // about repository identity.
        let (_decoy_tmp, decoy_spec) = fixture_under("models--other--repo");
        let decoy_root = match &decoy_spec.weights {
            WeightsSource::Dir(root) => root.clone(),
            _ => unreachable!(),
        };
        assert!(
            decoy_root.is_dir(),
            "the decoy snapshot must actually exist"
        );
        let canonical_root = match &spec.weights {
            WeightsSource::Dir(root) => root.clone(),
            _ => unreachable!(),
        };
        // `repository_revision` recognizes the canonical repo dir and falls back to a
        // path-derived `local:` identity for anything else — that is the discriminator.
        let canonical_revision = repository_revision(&canonical_root).unwrap();
        let decoy_revision = repository_revision(&decoy_root).unwrap();
        assert_eq!(
            canonical_revision,
            "0123456789abcdef0123456789abcdef01234567"
        );
        assert!(decoy_revision.starts_with("local:"), "{decoy_revision}");
        assert_ne!(canonical_revision, decoy_revision);

        // The decoy prepares (it is a complete snapshot) but seals a DIFFERENT artifact identity,
        // so a receipt minted against the canonical repo can never match it.
        let canonical = PreparedSvdMemory::prepare(&spec).unwrap();
        let decoy = PreparedSvdMemory::prepare(&decoy_spec).unwrap();
        assert_ne!(canonical.artifact_identity, decoy.artifact_identity);
        assert_ne!(canonical.revision, decoy.revision);

        // ...and the canonical spec's sealed pins do not cover the decoy's files.
        let mut crossed_repo = spec.clone();
        crossed_repo.weights = WeightsSource::Dir(decoy_root);
        assert!(PreparedSvdMemory::prepare(&crossed_repo).is_err());
    }
}
