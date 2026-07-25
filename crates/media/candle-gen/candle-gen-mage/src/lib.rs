//! Candle/CUDA Mage-Flow RL generation provider.
//!
//! Mage is not implemented by parameterizing Z-Image: its GELU dual-stream NR-MMDiT, adjacent-pair
//! image-only MSRoPE, 128-channel one-step VAE, final-normalized Qwen3-VL conditioning, and
//! velocity convention are independent parity surfaces.

pub mod config;
pub mod latent;
pub mod pipeline;
pub mod rope;
pub mod scheduler;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

pub use config::{MageConfig, MODEL_ID};
pub use pipeline::MagePipeline;
pub use text_encoder::MageTextEncoder;
pub use transformer::MageTransformer;
pub use vae::MageVae;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, WeightsSource,
};

pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: config::FAMILY,
        backend: "candle",
        modality: Modality::Image,
        required_components: &[],
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            min_size: config::MIN_SIZE,
            max_size: config::MAX_SIZE,
            max_count: 8,
            mac_only: false,
            // sc-14051 is the dense RL generation lane. Quantized loading is a separate story.
            supported_quants: &[],
            ..Default::default()
        },
    }
}

pub struct MageGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: candle_core::Device,
    components: Mutex<Option<Arc<MagePipeline>>>,
}

impl MageGenerator {
    fn components(&self) -> gen_core::Result<Arc<MagePipeline>> {
        candle_gen::cached(&self.components, || {
            MagePipeline::load(&self.root, &self.device)
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
                        req.steps.unwrap_or(20) as usize,
                        req.guidance.unwrap_or(5.0),
                        base_seed.wrapping_add(index as u64),
                        on_progress,
                    )
                    .map_err(candle_gen::CandleError::from)?,
            );
        }
        Ok(GenerationOutput::Images(images))
    }
}

pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "mage_flow RL generation currently supports dense weights only".into(),
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
        descriptor: descriptor(),
        root,
        device,
        components: Mutex::new(None),
    }))
}

candle_gen::register_generators! {
    pub const REGISTRATION = descriptor => load
}

pub fn register_providers(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry.register_generator(REGISTRATION)
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
        assert_eq!(ids, ["mage_flow"]);
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
    fn quantized_loading_is_not_advertised_or_silently_ignored() {
        use candle_gen::gen_core::Quant;

        assert!(descriptor().capabilities.supported_quants.is_empty());
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        spec.quantize = Some(Quant::Q4);
        let err = load(&spec)
            .err()
            .expect("quantized loading must be rejected");
        assert!(err.to_string().contains("dense weights only"), "{err}");
    }
}
