//! Candle/CUDA Mage-Flow RL generation provider.
//!
//! Mage is not implemented by parameterizing Z-Image: its GELU dual-stream NR-MMDiT, adjacent-pair
//! image-only MSRoPE, 128-channel one-step VAE, final-normalized Qwen3-VL conditioning, and
//! velocity convention are independent parity surfaces.

pub mod config;
pub mod edit_provider;
pub mod latent;
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

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::{io::Read, io::Seek, io::SeekFrom};

use candle_gen::gen_core::{
    self, Capabilities, Conditioning, ConditioningKind, GenerationOutput, GenerationRequest,
    Generator, LoadSpec, Modality, ModelDescriptor, Progress, WeightsSource,
};
use sha2::{Digest, Sha256};

fn generation_descriptor(
    id: &'static str,
    supports_guidance: bool,
    supports_negative_prompt: bool,
) -> ModelDescriptor {
    ModelDescriptor {
        id,
        family: config::FAMILY,
        backend: "candle",
        modality: Modality::Image,
        required_components: &[],
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
    root: PathBuf,
    device: candle_core::Device,
    quant: Option<candle_gen::gen_core::Quant>,
    default_steps: u32,
    default_guidance: f32,
    components: Mutex<Option<Arc<MagePipeline>>>,
}

pub struct MageEditGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    variant: MageEditVariant,
    device: candle_core::Device,
    quant: Option<candle_gen::gen_core::Quant>,
    components: Mutex<Option<Arc<MageEdit>>>,
}

impl MageEditGenerator {
    fn components(&self) -> gen_core::Result<Arc<MageEdit>> {
        candle_gen::cached(&self.components, || {
            verify_edit_checkpoint(&self.root, self.variant)?;
            MageEdit::load_with_quant(&self.root, self.quant, &self.device)
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

impl Generator for MageEditGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
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
        let references = resolve_edit_references(req)?
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        let components = self.components()?;
        let (default_steps, default_guidance) = self.variant.defaults();
        let base_seed = req.seed.unwrap_or(0);
        let mut images = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            images.push(
                components
                    .edit(
                        &req.prompt,
                        req.negative_prompt.as_deref().unwrap_or(" "),
                        &references,
                        req.width,
                        req.height,
                        req.steps.map_or(default_steps, |steps| steps as usize),
                        req.guidance.unwrap_or(default_guidance),
                        base_seed.wrapping_add(index as u64),
                        &req.cancel,
                        on_progress,
                    )
                    .map_err(candle_gen::CandleError::from)?,
            );
        }
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
            MagePipeline::load_with_quant(&self.root, self.quant, &self.device)
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
        let components = self.components()?;
        let base_seed = req.seed.unwrap_or(0);
        let mut images = Vec::with_capacity(req.count as usize);
        for index in 0..req.count {
            if req.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            images.push(
                components
                    .generate(
                        &req.prompt,
                        req.negative_prompt.as_deref().unwrap_or(" "),
                        req.width,
                        req.height,
                        req.steps.unwrap_or(self.default_steps) as usize,
                        req.guidance.unwrap_or(self.default_guidance),
                        base_seed.wrapping_add(index as u64),
                        on_progress,
                    )
                    .map_err(candle_gen::CandleError::from)?,
            );
        }
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
    Ok(Box::new(MageGenerator {
        descriptor,
        root,
        device,
        quant: spec.quantize,
        default_steps,
        default_guidance,
        components: Mutex::new(None),
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
        id: variant.id(),
        family: config::FAMILY,
        backend: "candle",
        modality: Modality::Image,
        required_components: &[],
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
    let device = candle_gen::default_device()?;
    Ok(Box::new(MageEditGenerator {
        descriptor: edit_descriptor(variant),
        root,
        variant,
        device,
        quant: spec.quantize,
        components: Mutex::new(None),
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
    registry
        .register_generator(REGISTRATION)
        .register_generator(BASE_REGISTRATION)
        .register_generator(TURBO_REGISTRATION)
        .register_generator(EDIT_REGISTRATION)
        .register_generator(EDIT_BASE_REGISTRATION)
        .register_generator(EDIT_TURBO_REGISTRATION)
}

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
}
