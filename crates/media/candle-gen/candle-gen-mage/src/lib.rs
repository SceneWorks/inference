//! Candle/CUDA Mage-Flow RL generation provider.
//!
//! Mage is not implemented by parameterizing Z-Image: its GELU dual-stream NR-MMDiT, adjacent-pair
//! image-only MSRoPE, 128-channel one-step VAE, final-normalized Qwen3-VL conditioning, and
//! velocity convention are independent parity surfaces.

pub mod config;
pub mod edit_provider;
pub mod latent;
pub mod memory_strategy;
pub mod pipeline;
pub mod quant;
pub mod rope;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

pub use config::{MageConfig, MODEL_ID};
pub use edit_provider::{MageEdit, MageEditVariant};
pub use pipeline::MagePipeline;
pub use text_encoder::MageTextEncoder;
pub use transformer::MageTransformer;
pub use vae::MageVae;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::{io::Read, io::Seek, io::SeekFrom};

use candle_gen::gen_core::{
    self, Capabilities, Conditioning, ConditioningKind, GenerationOutput, GenerationRequest,
    Generator, LoadSpec, Modality, ModelDescriptor, Progress, WeightsSource,
};
use sha2::{Digest, Sha256};

fn cancel_aware<T>(
    result: candle_core::Result<T>,
    cancel: &gen_core::CancelFlag,
) -> candle_gen::Result<T> {
    if cancel.is_cancelled() {
        Err(candle_gen::CandleError::Canceled)
    } else {
        result.map_err(candle_gen::CandleError::from)
    }
}

pub(crate) fn begin_decode(
    cancel: &gen_core::CancelFlag,
    label: &str,
    on_progress: &mut dyn FnMut(Progress),
) -> candle_core::Result<()> {
    if cancel.is_cancelled() {
        candle_core::bail!("{label} canceled");
    }
    on_progress(Progress::Decoding);
    if cancel.is_cancelled() {
        candle_core::bail!("{label} canceled");
    }
    Ok(())
}

/// Caller-provisioned shared component ids. These match the MLX provider and the SceneWorks
/// manifest: each per-variant tier contains only the transformer, while the bit-identical text
/// encoder and VAE are staged once from the shared components mirror.
pub const COMPONENT_TEXT_ENCODER: &str = "text_encoder";
pub const COMPONENT_VAE: &str = "vae";
pub const REQUIRED_COMPONENTS: &[&str] = &[COMPONENT_TEXT_ENCODER, COMPONENT_VAE];

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MageComponentDirs {
    pub(crate) transformer: PathBuf,
    pub(crate) text_encoder: PathBuf,
    pub(crate) vae: PathBuf,
}

impl MageComponentDirs {
    pub(crate) fn flat(root: &Path) -> Self {
        Self {
            transformer: root.join("transformer"),
            text_encoder: root.join("text_encoder"),
            vae: root.join("vae"),
        }
    }
}

/// Resolve SceneWorks' split layout while retaining the upstream flat-snapshot fallback. Unknown
/// component ids and file-valued directory components fail at load time instead of surfacing as a
/// misleading missing-weight error during the first render.
fn resolve_component_dirs(root: &Path, spec: &LoadSpec) -> gen_core::Result<MageComponentDirs> {
    gen_core::reject_unknown_components(spec, REQUIRED_COMPONENTS, config::FAMILY)?;
    let staged = |id: &str, fallback: &str| -> gen_core::Result<PathBuf> {
        match spec.components.get(id) {
            Some(WeightsSource::Dir(dir)) => Ok(dir.clone()),
            Some(WeightsSource::File(file)) => Err(gen_core::Error::Msg(format!(
                "mage_flow: the '{id}' component must be staged as a directory, got the file {}",
                file.display()
            ))),
            None => Ok(root.join(fallback)),
        }
    };
    Ok(MageComponentDirs {
        transformer: root.join("transformer"),
        text_encoder: staged(COMPONENT_TEXT_ENCODER, "text_encoder")?,
        vae: staged(COMPONENT_VAE, "vae")?,
    })
}

/// On-disk footprint of the exact component directories resolved by the production loader.
///
/// SceneWorks stages the shared text encoder and VAE through `LoadSpec::components`, outside the
/// route-local transformer snapshot. Computing from `spec.weights` alone therefore under-counts
/// split installs. Keep this resolver coupled to [`resolve_component_dirs`] so admission and load
/// cannot silently disagree about which assets participate in the request.
pub(crate) fn component_footprint(
    spec: &LoadSpec,
) -> gen_core::Result<gen_core::PerComponentBytes> {
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "mage-flow component footprint requires a snapshot directory".to_owned(),
            ))
        }
    };
    let dirs = resolve_component_dirs(root, spec)?;
    Ok(gen_core::PerComponentBytes {
        text_encoder: gen_core::safetensors_path_bytes(dirs.text_encoder),
        dit: gen_core::safetensors_path_bytes(dirs.transformer),
        vae: gen_core::safetensors_path_bytes(dirs.vae),
    })
}

fn generation_descriptor(
    id: &'static str,
    supports_guidance: bool,
    supports_negative_prompt: bool,
) -> ModelDescriptor {
    ModelDescriptor {
        control_kinds: None,
        id,
        family: config::FAMILY,
        backend: "candle",
        modality: Modality::Image,
        required_components: REQUIRED_COMPONENTS,
        capabilities: Capabilities {
            supports_negative_prompt,
            supports_guidance,
            min_size: config::MIN_SIZE,
            max_size: config::MAX_SIZE,
            max_count: 8,
            mac_only: false,
            supported_quants: &[
                candle_gen::gen_core::Quant::Q4,
                candle_gen::gen_core::Quant::Q8,
            ],
            component_precision_floors: quant::COMPONENT_PRECISION_FLOORS,
            ..Default::default()
        },
    }
}

pub fn descriptor() -> ModelDescriptor {
    generation_descriptor(MODEL_ID, true, true)
}

pub fn descriptor_base() -> ModelDescriptor {
    generation_descriptor(config::BASE_MODEL_ID, true, true)
}

pub fn descriptor_turbo() -> ModelDescriptor {
    generation_descriptor(config::TURBO_MODEL_ID, false, false)
}

pub struct MageGenerator {
    descriptor: ModelDescriptor,
    component_dirs: MageComponentDirs,
    device: candle_core::Device,
    quant: Option<candle_gen::gen_core::Quant>,
    default_steps: u32,
    default_guidance: f32,
    components: Mutex<Option<Arc<MagePipeline>>>,
    lifecycle: Mutex<()>,
    loaded_quant: Option<candle_gen::gen_core::Quant>,
    memory_strategy: Option<gen_core::MemoryProviderContract>,
    memory_admission: memory_strategy::AdmissionRegistry,
}

pub struct MageEditGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    component_dirs: MageComponentDirs,
    variant: MageEditVariant,
    device: candle_core::Device,
    quant: Option<candle_gen::gen_core::Quant>,
    components: Mutex<Option<Arc<MageEdit>>>,
    lifecycle: Mutex<()>,
    loaded_quant: Option<candle_gen::gen_core::Quant>,
    memory_strategy: Option<gen_core::MemoryProviderContract>,
    memory_admission: memory_strategy::AdmissionRegistry,
}

impl MageEditGenerator {
    fn components(&self) -> gen_core::Result<Arc<MageEdit>> {
        candle_gen::cached(&self.components, || {
            verify_edit_checkpoint(&self.root, self.variant)?;
            MageEdit::load_components(&self.component_dirs, self.quant, &self.device)
                .map(Arc::new)
                .map_err(candle_gen::CandleError::from)
        })
        .map_err(Into::into)
    }
}

fn verify_edit_checkpoint(
    root: &std::path::Path,
    variant: MageEditVariant,
) -> candle_core::Result<()> {
    let (revision, expected_sha256) = match variant {
        MageEditVariant::Edit => (
            "b01d524f86498b7dabcc4b3572c6d264d786a16e",
            "bd24b2009764136298499d60750ded8ebdfa7950981d116e9937588471b2ecab",
        ),
        MageEditVariant::EditBase => (
            "8654a7bc0283ab2946385230b5b2eb944e0b76ea",
            "bb53a04c20e5df443bb093c3f24027f9391f6d65e3edd60ed96546b050db717b",
        ),
        MageEditVariant::EditTurbo => (
            "14427bd7627d3a25436497a5939e1096f6a0d523",
            "d387be05845ea0e0fc6b2bec5c05bccb3808c25a0123d9e2b3459e2e7f9705df",
        ),
    };
    let tensor_name = "transformer_blocks.0.attn.add_k_proj.bias";
    let path = root
        .join("transformer")
        .join("diffusion_pytorch_model.safetensors");
    let mut file = std::fs::File::open(&path)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > 1_048_576 {
        return Err(candle_core::Error::Msg(format!(
            "{}: invalid safetensors header length {header_len}",
            variant.id()
        )));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)?;
    let metadata: serde_json::Value = serde_json::from_slice(&header)
        .map_err(|error| candle_core::Error::Msg(format!("invalid safetensors header: {error}")))?;
    let tensor = metadata.get(tensor_name).ok_or_else(|| {
        candle_core::Error::Msg(format!("{}: missing {tensor_name}", variant.id()))
    })?;
    if tensor.get("dtype").and_then(serde_json::Value::as_str) != Some("BF16")
        || tensor.get("shape").and_then(serde_json::Value::as_array)
            != Some(&vec![serde_json::json!(3072)])
    {
        return Err(candle_core::Error::Msg(format!(
            "{}: {tensor_name} has the wrong dtype or shape",
            variant.id()
        )));
    }
    let offsets = tensor["data_offsets"]
        .as_array()
        .ok_or_else(|| candle_core::Error::Msg("invalid tensor data offsets".into()))?;
    let start = offsets.first().and_then(serde_json::Value::as_u64);
    let end = offsets.get(1).and_then(serde_json::Value::as_u64);
    let (Some(start), Some(end)) = (start, end) else {
        return Err(candle_core::Error::Msg(format!(
            "{}: {tensor_name} has invalid data offsets",
            variant.id()
        )));
    };
    if end.saturating_sub(start) < 4096 {
        return Err(candle_core::Error::Msg(format!(
            "{}: {tensor_name} is too short for identity verification",
            variant.id()
        )));
    }
    file.seek(SeekFrom::Start(8 + header_len + start))?;
    let mut bytes = vec![0u8; 4096];
    file.read_exact(&mut bytes)?;
    let got = format!("{:x}", Sha256::digest(bytes));
    if got != expected_sha256 {
        return Err(candle_core::Error::Msg(format!(
            "{}: checkpoint fingerprint mismatch for {tensor_name}; expected revision {revision}, \
             got sha256 {got}",
            variant.id()
        )));
    }
    Ok(())
}

fn verify_staged_edit_checkpoint(
    root: &Path,
    variant: MageEditVariant,
    stage_residency: bool,
) -> candle_core::Result<()> {
    if stage_residency {
        verify_edit_checkpoint(root, variant)
    } else {
        Ok(())
    }
}

impl Generator for MageEditGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.memory_strategy.as_ref()
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        let Some(contract) = self.memory_strategy.as_ref() else {
            return gen_core::MemorySafetyDecision::Accept;
        };
        match memory_strategy::validate_context(contract, context, self.loaded_quant) {
            Ok(()) => match self.memory_admission.approve(context) {
                Ok(()) => gen_core::MemorySafetyDecision::Accept,
                Err(error) => gen_core::MemorySafetyDecision::Reject {
                    reason: error.to_string(),
                },
            },
            Err(error) => {
                self.memory_admission.clear_approval();
                gen_core::MemorySafetyDecision::Reject {
                    reason: error.to_string(),
                }
            }
        }
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        let Some(contract) = self.memory_strategy.as_ref() else {
            return Ok(None);
        };
        memory_strategy::validate_context(contract, context, self.loaded_quant)?;
        Ok(Some(Box::new(memory_strategy::MageMemoryScope::new_bound(
            self.device.clone(),
            contract,
            context,
            self.memory_admission.clone(),
        )?)))
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        if req.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg(
                "mage edit: instruction must not be empty".into(),
            ));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg("mage edit: steps must be >= 1".into()));
        }
        if req.strength.is_some() {
            return Err(gen_core::Error::Unsupported(
                "mage edit uses full reference-token conditioning and does not support request \
                 strength"
                    .into(),
            ));
        }
        if !req.width.is_multiple_of(config::SIZE_MULTIPLE)
            || !req.height.is_multiple_of(config::SIZE_MULTIPLE)
        {
            return Err(gen_core::Error::Msg(format!(
                "mage edit: dimensions must be multiples of {}",
                config::SIZE_MULTIPLE
            )));
        }
        resolve_edit_references(req)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let _lifecycle = candle_gen::lock_recover(&self.lifecycle);
        self.memory_admission.consume_for_generate(req)?;
        let stage_residency = req.memory.is_some_and(|memory| memory.stage_residency);
        let stream_transformer_blocks = req
            .memory
            .is_some_and(|memory| memory.stream_transformer_blocks);
        if req.memory.is_some_and(|memory| {
            memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks
        }) && !stage_residency
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: constrained strategies require staged residency",
                self.descriptor.id
            )));
        }
        let references = resolve_edit_references(req)?
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        let (default_steps, default_guidance) = self.variant.defaults();
        let base_seed = req.seed.unwrap_or(0);
        let steps = req.steps.map_or(default_steps, |steps| steps as usize);
        let guidance = req.guidance.unwrap_or(default_guidance);
        let mut render_resident = || -> gen_core::Result<Vec<gen_core::Image>> {
            let components = self.components()?;
            let mut images = Vec::with_capacity(req.count as usize);
            for index in 0..req.count {
                let result = components.edit_with_memory(
                    &req.prompt,
                    req.negative_prompt.as_deref().unwrap_or(" "),
                    &references,
                    req.width,
                    req.height,
                    steps,
                    guidance,
                    base_seed.wrapping_add(index as u64),
                    req.memory,
                    &req.cancel,
                    on_progress,
                );
                if req.cancel.is_cancelled() {
                    return Err(gen_core::Error::Canceled);
                }
                images.push(result.map_err(candle_gen::CandleError::from)?);
            }
            Ok(images)
        };
        let images = if !stage_residency {
            render_resident()?
        } else {
            // The staged loader bypasses `self.components()`, so retain the exact same sibling-
            // checkpoint fingerprint gate before opening any conditioning/heavy component.
            verify_staged_edit_checkpoint(&self.root, self.variant, stage_residency)
                .map_err(candle_gen::CandleError::from)?;
            let resident = candle_gen::lock_recover(&self.components).take();
            drop(resident);
            self.device
                .synchronize()
                .map_err(gen_core::Error::backend)?;
            let dirs = self.component_dirs.clone();
            let quant = self.quant;
            let device = self.device.clone();
            candle_gen::run_sequential(
                &req.cancel,
                &device,
                on_progress,
                || {
                    cancel_aware(
                        MageEdit::load_conditioning(&dirs, quant, &device),
                        &req.cancel,
                    )
                },
                |conditioning| {
                    cancel_aware(
                        MageEdit::encode_conditioning(
                            conditioning,
                            &req.prompt,
                            req.negative_prompt.as_deref().unwrap_or(" "),
                            &references,
                            req.width,
                            req.height,
                            guidance,
                            base_seed,
                        ),
                        &req.cancel,
                    )
                },
                || {
                    cancel_aware(
                        MageEdit::load_heavy(
                            &dirs,
                            quant,
                            &device,
                            stream_transformer_blocks,
                            &req.cancel,
                        ),
                        &req.cancel,
                    )
                },
                |heavy, encoded, on_progress| {
                    // Optimized contexts are single-image; the shared safety gate rejects larger
                    // batches before this point so reference-posterior seeding stays exact.
                    cancel_aware(
                        MageEdit::sample_heavy(
                            heavy,
                            encoded,
                            req.width,
                            req.height,
                            steps,
                            guidance,
                            base_seed,
                            req.memory,
                            &req.cancel,
                            on_progress,
                        ),
                        &req.cancel,
                    )
                    .map(|image| vec![image])
                },
            )
            .map_err(gen_core::Error::from)?
        };
        Ok(GenerationOutput::Images(images))
    }
}

fn resolve_edit_references(
    req: &GenerationRequest,
) -> gen_core::Result<Vec<&candle_gen::gen_core::Image>> {
    let mut references = Vec::new();
    for conditioning in &req.conditioning {
        match conditioning {
            Conditioning::Reference { image, strength } => {
                if strength.is_some() {
                    return Err(gen_core::Error::Unsupported(
                        "mage edit uses full reference-token conditioning and does not support \
                         per-reference strength"
                            .into(),
                    ));
                }
                references.push(image);
            }
            Conditioning::MultiReference { images } => references.extend(images),
            _ => {}
        }
    }
    if references.is_empty() {
        return Err(gen_core::Error::Msg(
            "mage edit requires Reference or MultiReference conditioning".into(),
        ));
    }
    for (index, image) in references.iter().enumerate() {
        let expected = image.width as usize * image.height as usize * 3;
        if image.width == 0 || image.height == 0 || image.pixels.len() != expected {
            return Err(gen_core::Error::Msg(format!(
                "mage edit reference {index} is not valid RGB8"
            )));
        }
    }
    Ok(references)
}

impl MageGenerator {
    fn components(&self) -> gen_core::Result<Arc<MagePipeline>> {
        candle_gen::cached(&self.components, || {
            MagePipeline::load_components(&self.component_dirs, self.quant, &self.device)
                .map(Arc::new)
                .map_err(candle_gen::CandleError::from)
        })
        .map_err(Into::into)
    }
}

impl Generator for MageGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.memory_strategy.as_ref()
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        let Some(contract) = self.memory_strategy.as_ref() else {
            return gen_core::MemorySafetyDecision::Accept;
        };
        match memory_strategy::validate_context(contract, context, self.loaded_quant) {
            Ok(()) => match self.memory_admission.approve(context) {
                Ok(()) => gen_core::MemorySafetyDecision::Accept,
                Err(error) => gen_core::MemorySafetyDecision::Reject {
                    reason: error.to_string(),
                },
            },
            Err(error) => {
                self.memory_admission.clear_approval();
                gen_core::MemorySafetyDecision::Reject {
                    reason: error.to_string(),
                }
            }
        }
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        let Some(contract) = self.memory_strategy.as_ref() else {
            return Ok(None);
        };
        memory_strategy::validate_context(contract, context, self.loaded_quant)?;
        Ok(Some(Box::new(memory_strategy::MageMemoryScope::new_bound(
            self.device.clone(),
            contract,
            context,
            self.memory_admission.clone(),
        )?)))
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        if req.prompt.trim().is_empty() {
            return Err(gen_core::Error::Msg(
                "mage_flow: prompt must not be empty".into(),
            ));
        }
        if req.steps == Some(0) {
            return Err(gen_core::Error::Msg("mage_flow: steps must be >= 1".into()));
        }
        if !req.width.is_multiple_of(config::SIZE_MULTIPLE)
            || !req.height.is_multiple_of(config::SIZE_MULTIPLE)
        {
            return Err(gen_core::Error::Msg(format!(
                "mage_flow: dimensions must be multiples of {}",
                config::SIZE_MULTIPLE
            )));
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let _lifecycle = candle_gen::lock_recover(&self.lifecycle);
        self.memory_admission.consume_for_generate(req)?;
        let stage_residency = req.memory.is_some_and(|memory| memory.stage_residency);
        let stream_transformer_blocks = req
            .memory
            .is_some_and(|memory| memory.stream_transformer_blocks);
        if req.memory.is_some_and(|memory| {
            memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks
        }) && !stage_residency
        {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: constrained strategies require staged residency",
                self.descriptor.id
            )));
        }
        let base_seed = req.seed.unwrap_or(0);
        let steps = req.steps.unwrap_or(self.default_steps) as usize;
        let guidance = req.guidance.unwrap_or(self.default_guidance);
        let images = if !stage_residency {
            let components = self.components()?;
            let mut images = Vec::with_capacity(req.count as usize);
            for index in 0..req.count {
                if req.cancel.is_cancelled() {
                    return Err(gen_core::Error::Canceled);
                }
                let result = components.generate_with_memory(
                    &req.prompt,
                    req.negative_prompt.as_deref().unwrap_or(" "),
                    req.width,
                    req.height,
                    steps,
                    guidance,
                    base_seed.wrapping_add(index as u64),
                    req.memory,
                    &req.cancel,
                    on_progress,
                );
                if req.cancel.is_cancelled() {
                    return Err(gen_core::Error::Canceled);
                }
                images.push(result.map_err(candle_gen::CandleError::from)?);
            }
            images
        } else {
            let resident = candle_gen::lock_recover(&self.components).take();
            drop(resident);
            self.device
                .synchronize()
                .map_err(gen_core::Error::backend)?;
            let dirs = self.component_dirs.clone();
            let quant = self.quant;
            let device = self.device.clone();
            candle_gen::run_sequential(
                &req.cancel,
                &device,
                on_progress,
                || cancel_aware(MagePipeline::load_text(&dirs, quant, &device), &req.cancel),
                |text| {
                    cancel_aware(
                        MagePipeline::encode_prompt(
                            text,
                            &req.prompt,
                            req.negative_prompt.as_deref().unwrap_or(" "),
                            guidance,
                        ),
                        &req.cancel,
                    )
                },
                || {
                    cancel_aware(
                        MagePipeline::load_heavy(
                            &dirs,
                            quant,
                            &device,
                            stream_transformer_blocks,
                            &req.cancel,
                        ),
                        &req.cancel,
                    )
                },
                |heavy, encoded, on_progress| {
                    cancel_aware(
                        MagePipeline::sample(
                            &heavy.transformer,
                            &heavy.vae,
                            encoded,
                            req.width,
                            req.height,
                            steps,
                            guidance,
                            base_seed,
                            req.memory,
                            &req.cancel,
                            on_progress,
                        ),
                        &req.cancel,
                    )
                    .map(|image| vec![image])
                },
            )
            .map_err(gen_core::Error::from)?
        };
        Ok(GenerationOutput::Images(images))
    }
}

fn load_generation_variant(
    spec: &LoadSpec,
    descriptor: ModelDescriptor,
    default_steps: u32,
    default_guidance: f32,
) -> gen_core::Result<Box<dyn Generator>> {
    if matches!(spec.quantize, Some(candle_gen::gen_core::Quant::Nvfp4)) {
        return Err(gen_core::Error::Unsupported(
            "mage_flow does not support NVFP4".into(),
        ));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(path) => path.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "mage_flow expects a diffusers snapshot directory".into(),
            ))
        }
    };
    let component_dirs = resolve_component_dirs(&root, spec)?;
    if !spec.adapters.is_empty()
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
    {
        return Err(gen_core::Error::Unsupported(
            "mage_flow RL generation does not accept adapters or control overlays".into(),
        ));
    }
    let device = candle_gen::default_device()?;
    #[cfg(any(feature = "cuda", test))]
    let memory_strategy = Some(memory_strategy::provider_contract_for(descriptor.id, spec)?);
    #[cfg(not(any(feature = "cuda", test)))]
    let memory_strategy = None;
    Ok(Box::new(MageGenerator {
        memory_admission: memory_strategy::AdmissionRegistry::new(descriptor.id),
        descriptor,
        component_dirs,
        device,
        quant: spec.quantize,
        default_steps,
        default_guidance,
        components: Mutex::new(None),
        lifecycle: Mutex::new(()),
        loaded_quant: spec.quantize,
        memory_strategy,
    }))
}

pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_generation_variant(spec, descriptor(), 20, 5.0)
}

pub fn load_base(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_generation_variant(spec, descriptor_base(), 30, 5.0)
}

pub fn load_turbo(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_generation_variant(spec, descriptor_turbo(), 4, 1.0)
}

pub fn edit_descriptor(variant: MageEditVariant) -> ModelDescriptor {
    ModelDescriptor {
        control_kinds: None,
        id: variant.id(),
        family: config::FAMILY,
        backend: "candle",
        modality: Modality::Image,
        required_components: REQUIRED_COMPONENTS,
        capabilities: Capabilities {
            supports_negative_prompt: !matches!(variant, MageEditVariant::EditTurbo),
            supports_guidance: !matches!(variant, MageEditVariant::EditTurbo),
            // Mage exposes its two-forward CFG scale through `guidance`; `true_cfg` is a distinct
            // request field and is not consumed by this provider.
            supports_true_cfg: false,
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
            ],
            min_size: config::MIN_SIZE,
            max_size: config::MAX_SIZE,
            max_count: 8,
            mac_only: false,
            supported_quants: &[
                candle_gen::gen_core::Quant::Q4,
                candle_gen::gen_core::Quant::Q8,
            ],
            component_precision_floors: quant::COMPONENT_PRECISION_FLOORS,
            ..Default::default()
        },
    }
}

pub fn descriptor_edit() -> ModelDescriptor {
    edit_descriptor(MageEditVariant::Edit)
}

pub fn descriptor_edit_base() -> ModelDescriptor {
    edit_descriptor(MageEditVariant::EditBase)
}

pub fn descriptor_edit_turbo() -> ModelDescriptor {
    edit_descriptor(MageEditVariant::EditTurbo)
}

fn load_edit_variant(
    spec: &LoadSpec,
    variant: MageEditVariant,
) -> gen_core::Result<Box<dyn Generator>> {
    if matches!(spec.quantize, Some(candle_gen::gen_core::Quant::Nvfp4)) {
        return Err(gen_core::Error::Unsupported(format!(
            "{} does not support NVFP4",
            variant.id()
        )));
    }
    if !spec.adapters.is_empty()
        || spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{} does not accept adapters or control overlays",
            variant.id()
        )));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(path) => path.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{} expects a diffusers snapshot directory",
                variant.id()
            )))
        }
    };
    let component_dirs = resolve_component_dirs(&root, spec)?;
    let device = candle_gen::default_device()?;
    let descriptor = edit_descriptor(variant);
    #[cfg(any(feature = "cuda", test))]
    let memory_strategy = Some(memory_strategy::provider_contract_for(descriptor.id, spec)?);
    #[cfg(not(any(feature = "cuda", test)))]
    let memory_strategy = None;
    Ok(Box::new(MageEditGenerator {
        memory_admission: memory_strategy::AdmissionRegistry::new(descriptor.id),
        descriptor,
        root,
        component_dirs,
        variant,
        device,
        quant: spec.quantize,
        components: Mutex::new(None),
        lifecycle: Mutex::new(()),
        loaded_quant: spec.quantize,
        memory_strategy,
    }))
}

pub fn load_edit(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_edit_variant(spec, MageEditVariant::Edit)
}

pub fn load_edit_base(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_edit_variant(spec, MageEditVariant::EditBase)
}

pub fn load_edit_turbo(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_edit_variant(spec, MageEditVariant::EditTurbo)
}

candle_gen::register_generators! {
    pub const REGISTRATION = descriptor => load
}
candle_gen::register_generators! {
    pub const BASE_REGISTRATION = descriptor_base => load_base
}
candle_gen::register_generators! {
    pub const TURBO_REGISTRATION = descriptor_turbo => load_turbo
}
candle_gen::register_generators! {
    pub const EDIT_REGISTRATION = descriptor_edit => load_edit
}
candle_gen::register_generators! {
    pub const EDIT_BASE_REGISTRATION = descriptor_edit_base => load_edit_base
}
candle_gen::register_generators! {
    pub const EDIT_TURBO_REGISTRATION = descriptor_edit_turbo => load_edit_turbo
}

pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    let registry = registry
        .register_generator(REGISTRATION)
        .register_generator(BASE_REGISTRATION)
        .register_generator(TURBO_REGISTRATION)
        .register_generator(EDIT_REGISTRATION)
        .register_generator(EDIT_BASE_REGISTRATION)
        .register_generator(EDIT_TURBO_REGISTRATION);
    #[cfg(feature = "cuda")]
    let registry = registry
        .register_memory_strategy(RL_MEMORY_REGISTRATION)
        .register_memory_behavior(RL_MEMORY_BEHAVIOR)
        .register_memory_strategy(BASE_MEMORY_REGISTRATION)
        .register_memory_behavior(BASE_MEMORY_BEHAVIOR)
        .register_memory_strategy(TURBO_MEMORY_REGISTRATION)
        .register_memory_behavior(TURBO_MEMORY_BEHAVIOR)
        .register_memory_strategy(EDIT_MEMORY_REGISTRATION)
        .register_memory_behavior(EDIT_MEMORY_BEHAVIOR)
        .register_memory_strategy(EDIT_BASE_MEMORY_REGISTRATION)
        .register_memory_behavior(EDIT_BASE_MEMORY_BEHAVIOR)
        .register_memory_strategy(EDIT_TURBO_MEMORY_REGISTRATION)
        .register_memory_behavior(EDIT_TURBO_MEMORY_BEHAVIOR);
    registry
}

#[cfg(feature = "cuda")]
const RL_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: config::MODEL_ID,
    contract: memory_strategy::contract_rl,
    safety_check: memory_strategy::registered_safety_check,
};
#[cfg(feature = "cuda")]
const BASE_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: config::BASE_MODEL_ID,
    contract: memory_strategy::contract_base,
    safety_check: memory_strategy::registered_safety_check,
};
#[cfg(feature = "cuda")]
const TURBO_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: config::TURBO_MODEL_ID,
    contract: memory_strategy::contract_turbo,
    safety_check: memory_strategy::registered_safety_check,
};
#[cfg(feature = "cuda")]
const EDIT_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: config::EDIT_MODEL_ID,
    contract: memory_strategy::contract_edit,
    safety_check: memory_strategy::registered_safety_check,
};
#[cfg(feature = "cuda")]
const EDIT_BASE_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: config::EDIT_BASE_MODEL_ID,
    contract: memory_strategy::contract_edit_base,
    safety_check: memory_strategy::registered_safety_check,
};
#[cfg(feature = "cuda")]
const EDIT_TURBO_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: config::EDIT_TURBO_MODEL_ID,
    contract: memory_strategy::contract_edit_turbo,
    safety_check: memory_strategy::registered_safety_check,
};

macro_rules! memory_behavior {
    ($name:ident, $id:expr) => {
        #[cfg(feature = "cuda")]
        const $name: gen_core::MemoryBehaviorRegistration = gen_core::MemoryBehaviorRegistration {
            provider_id: $id,
            valid_fixtures: memory_strategy::registered_valid_fixture,
            begin_request: memory_strategy::registered_begin_request,
        };
    };
}

memory_behavior!(RL_MEMORY_BEHAVIOR, config::MODEL_ID);
memory_behavior!(BASE_MEMORY_BEHAVIOR, config::BASE_MODEL_ID);
memory_behavior!(TURBO_MEMORY_BEHAVIOR, config::TURBO_MODEL_ID);
memory_behavior!(EDIT_MEMORY_BEHAVIOR, config::EDIT_MODEL_ID);
memory_behavior!(EDIT_BASE_MEMORY_BEHAVIOR, config::EDIT_BASE_MODEL_ID);
memory_behavior!(EDIT_TURBO_MEMORY_BEHAVIOR, config::EDIT_TURBO_MODEL_ID);

pub fn provider_registry() -> gen_core::Result<gen_core::ProviderRegistry> {
    register_providers(gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod registry_tests {
    use super::*;

    #[test]
    fn rl_registry_surface_and_geometry_are_exact() {
        let registry = provider_registry().unwrap();
        let ids: Vec<_> = registry.generators().map(|r| (r.descriptor)().id).collect();
        assert_eq!(
            ids,
            [
                "mage_flow",
                "mage_flow_base",
                "mage_flow_turbo",
                "mage_flow_edit",
                "mage_flow_edit_base",
                "mage_flow_edit_turbo"
            ]
        );
        let g = registry
            .load(
                MODEL_ID,
                &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            )
            .unwrap();
        assert!(g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1024,
                height: 1024,
                ..Default::default()
            })
            .is_ok());
        assert!(g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1000,
                height: 1024,
                ..Default::default()
            })
            .is_err());
    }

    #[test]
    fn edit_variants_require_references_and_pin_defaults() {
        let registry = provider_registry().unwrap();
        for (id, steps, guidance) in [
            ("mage_flow_edit", 30, 5.0),
            ("mage_flow_edit_base", 30, 5.0),
            ("mage_flow_edit_turbo", 4, 1.0),
        ] {
            let generator = registry
                .load(
                    id,
                    &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
                )
                .unwrap();
            assert!(generator
                .validate(&GenerationRequest {
                    prompt: "make it night".into(),
                    width: 1024,
                    height: 1024,
                    ..Default::default()
                })
                .is_err());
            let variant = match id {
                "mage_flow_edit" => MageEditVariant::Edit,
                "mage_flow_edit_base" => MageEditVariant::EditBase,
                _ => MageEditVariant::EditTurbo,
            };
            assert_eq!(variant.defaults(), (steps, guidance));
            assert!(
                !generator.descriptor().capabilities.supports_true_cfg,
                "the provider must not advertise the unused true_cfg request field"
            );
            assert!(generator
                .validate(&GenerationRequest {
                    prompt: "make it night".into(),
                    width: 1024,
                    height: 1024,
                    conditioning: vec![Conditioning::Reference {
                        image: candle_gen::gen_core::Image {
                            width: 1,
                            height: 1,
                            pixels: vec![0; 3],
                        },
                        strength: Some(0.5),
                    }],
                    ..Default::default()
                })
                .is_err());
            assert!(generator
                .validate(&GenerationRequest {
                    prompt: "make it night".into(),
                    width: 1024,
                    height: 1024,
                    strength: Some(0.5),
                    conditioning: vec![Conditioning::Reference {
                        image: candle_gen::gen_core::Image {
                            width: 1,
                            height: 1,
                            pixels: vec![0; 3],
                        },
                        strength: None,
                    }],
                    ..Default::default()
                })
                .is_err());
        }
    }

    #[test]
    fn quantized_loading_is_advertised_and_reaches_the_lazy_generator() {
        use candle_gen::gen_core::Quant;

        assert_eq!(
            descriptor().capabilities.supported_quants,
            &[Quant::Q4, Quant::Q8]
        );
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        spec.quantize = Some(Quant::Q4);
        let generator = load(&spec).expect("q4 must reach the production lazy generator");
        assert_eq!(
            generator.descriptor().capabilities.supported_quants,
            &[Quant::Q4, Quant::Q8]
        );
        spec.quantize = Some(Quant::Nvfp4);
        assert!(load(&spec).is_err(), "unsupported NVFP4 must fail loudly");
    }

    #[test]
    fn all_six_descriptors_publish_the_same_components_and_precision_floors() {
        for descriptor in [
            descriptor(),
            descriptor_base(),
            descriptor_turbo(),
            descriptor_edit(),
            descriptor_edit_base(),
            descriptor_edit_turbo(),
        ] {
            assert_eq!(descriptor.required_components, REQUIRED_COMPONENTS);
            assert_eq!(
                descriptor.capabilities.component_precision_floors,
                quant::COMPONENT_PRECISION_FLOORS,
                "{} hid a load-time precision raise from the worker",
                descriptor.id
            );
        }
    }

    #[test]
    fn split_component_layout_and_flat_fallback_resolve_identically_for_every_variant() {
        let root = PathBuf::from("/variant/q4");
        let flat = LoadSpec::new(WeightsSource::Dir(root.clone()));
        assert_eq!(
            resolve_component_dirs(&root, &flat).unwrap(),
            MageComponentDirs::flat(&root)
        );

        let split = LoadSpec::new(WeightsSource::Dir(root.clone()))
            .with_component(
                COMPONENT_TEXT_ENCODER,
                WeightsSource::Dir("/shared/q8/text_encoder".into()),
            )
            .with_component(COMPONENT_VAE, WeightsSource::Dir("/shared/bf16/vae".into()));
        assert_eq!(
            resolve_component_dirs(&root, &split).unwrap(),
            MageComponentDirs {
                transformer: root.join("transformer"),
                text_encoder: "/shared/q8/text_encoder".into(),
                vae: "/shared/bf16/vae".into(),
            }
        );

        let invalid = LoadSpec::new(WeightsSource::Dir(root.clone())).with_component(
            COMPONENT_TEXT_ENCODER,
            WeightsSource::File("/shared/text_encoder.safetensors".into()),
        );
        assert!(resolve_component_dirs(&root, &invalid)
            .unwrap_err()
            .to_string()
            .contains("must be staged as a directory"));
    }

    #[test]
    fn split_component_footprint_follows_the_production_loader_paths() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path();
        let transformer = root.join("tier/transformer");
        let text = root.join("shared/text_encoder");
        let vae = root.join("shared/vae");
        for directory in [&transformer, &text, &vae] {
            std::fs::create_dir_all(directory).unwrap();
        }
        std::fs::write(transformer.join("model.safetensors"), vec![0u8; 11]).unwrap();
        std::fs::write(text.join("model.safetensors"), vec![0u8; 13]).unwrap();
        std::fs::write(vae.join("model.safetensors"), vec![0u8; 17]).unwrap();

        let spec = LoadSpec::new(WeightsSource::Dir(root.join("tier")))
            .with_component(COMPONENT_TEXT_ENCODER, WeightsSource::Dir(text))
            .with_component(COMPONENT_VAE, WeightsSource::Dir(vae));
        assert_eq!(
            component_footprint(&spec).unwrap(),
            gen_core::PerComponentBytes {
                text_encoder: 13,
                dit: 11,
                vae: 17,
            }
        );
        let generation = memory_strategy::contract_rl(&spec).unwrap();
        let edit = memory_strategy::contract_edit(&spec).unwrap();
        assert_eq!(generation.asset_facts.conditioning_bytes, 13);
        assert_eq!(edit.asset_facts.conditioning_bytes, 30);
        assert_eq!(generation.asset_facts.base_bytes, 41);
        assert_eq!(edit.asset_facts.base_bytes, 41);
    }

    #[test]
    fn staged_edit_keeps_the_route_checkpoint_fingerprint_gate() {
        let root_tmp = tempfile::tempdir().unwrap();
        let root = root_tmp.path().to_path_buf();
        std::fs::create_dir_all(root.join("transformer")).unwrap();
        let mut invalid_checkpoint = 2u64.to_le_bytes().to_vec();
        invalid_checkpoint.extend_from_slice(b"{}");
        std::fs::write(
            root.join("transformer/diffusion_pytorch_model.safetensors"),
            invalid_checkpoint,
        )
        .unwrap();

        assert!(verify_staged_edit_checkpoint(&root, MageEditVariant::Edit, false).is_ok());
        let error = verify_staged_edit_checkpoint(&root, MageEditVariant::Edit, true)
            .expect_err("staged execution must verify the exact edit checkpoint");
        assert!(error.to_string().contains("missing transformer_blocks.0"));
    }

    #[test]
    fn decode_callback_cancellation_is_preserved_as_typed_canceled() {
        let cancel = gen_core::CancelFlag::default();
        let callback_flag = cancel.clone();
        let mut progress = move |event| {
            if event == Progress::Decoding {
                callback_flag.cancel();
            }
        };
        let decoded = begin_decode(&cancel, "mage-test", &mut progress);
        assert!(decoded.is_err());
        assert!(matches!(
            cancel_aware(decoded.map(|_| ()), &cancel),
            Err(candle_gen::CandleError::Canceled)
        ));
    }
}
