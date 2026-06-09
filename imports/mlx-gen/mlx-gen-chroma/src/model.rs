//! Chroma provider registration + (forthcoming) txt2img generation path.
//!
//! **Skeleton (sc-3835):** the three variants register and load (tokenizer + T5 + VAE + transformer
//! weights), and `validate` enforces the advertised capability surface. The full generate path —
//! T5 masked encode (sc-3838), the Chroma DiT forward (sc-3836/sc-3837), and the true-CFG flow-match
//! denoise + VAE decode (sc-3839) — lands in its own slices.

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    Error, GenerationOutput, GenerationRequest, Generator, LoadSpec, ModelDescriptor,
    ModelRegistration, Precision, Progress, Result, WeightsSource,
};
use mlx_gen_flux::T5TextEncoder;
use mlx_gen_z_image::vae::Vae;

use crate::config::{ChromaTransformerConfig, ChromaVariant};
use crate::loader;
use crate::transformer::ChromaTransformer;

pub fn descriptor_hd() -> ModelDescriptor {
    ChromaVariant::Hd.descriptor()
}

pub fn descriptor_base() -> ModelDescriptor {
    ChromaVariant::Base.descriptor()
}

pub fn descriptor_flash() -> ModelDescriptor {
    ChromaVariant::Flash.descriptor()
}

pub fn load_hd(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(load_chroma(ChromaVariant::Hd, spec)?))
}

pub fn load_base(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(load_chroma(ChromaVariant::Base, spec)?))
}

pub fn load_flash(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(load_chroma(ChromaVariant::Flash, spec)?))
}

pub fn load_chroma(variant: ChromaVariant, spec: &LoadSpec) -> Result<Chroma> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{}: only dense bf16 is wired for the Chroma port (quant = sc-3841)",
            variant.id()
        )));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{} expects a Chroma diffusers snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ vae/), not a single .safetensors file",
                variant.id()
            )))
        }
    };

    let cfg = ChromaTransformerConfig::default();
    let tokenizer = loader::load_tokenizer()?;
    let t5 = loader::load_t5_encoder(root)?;
    let transformer = loader::load_transformer(root, cfg)?;
    let vae = loader::load_vae(root)?;

    Ok(Chroma {
        descriptor: variant.descriptor(),
        variant,
        tokenizer: Some(tokenizer),
        t5: Some(t5),
        transformer: Some(transformer),
        vae: Some(vae),
    })
}

pub struct Chroma {
    descriptor: ModelDescriptor,
    #[allow(dead_code)]
    variant: ChromaVariant,
    #[allow(dead_code)]
    tokenizer: Option<TextTokenizer>,
    #[allow(dead_code)]
    t5: Option<T5TextEncoder>,
    #[allow(dead_code)]
    transformer: Option<ChromaTransformer>,
    #[allow(dead_code)]
    vae: Option<Vae>,
}

impl Generator for Chroma {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        if req.prompt.trim().is_empty() {
            return Err(Error::Msg(format!(
                "{}: prompt must not be empty",
                self.descriptor.id
            )));
        }
        Ok(())
    }

    fn generate(
        &self,
        _req: &GenerationRequest,
        _on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        Err(Error::Msg(format!(
            "{}: generate is not yet wired — Chroma DiT forward (sc-3836/sc-3837) + true-CFG \
             flow-match denoise (sc-3839) are pending",
            self.descriptor.id
        )))
    }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_hd, load: load_hd }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_base, load: load_base }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_flash, load: load_flash }
}
