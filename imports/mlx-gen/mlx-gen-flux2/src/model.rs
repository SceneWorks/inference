//! FLUX.2-klein provider registration.
//!
//! S0 registers the two variant ids (`flux2_klein_9b`, `flux2_klein_9b_edit`) so the registry
//! resolves them and consumers can introspect their descriptors **without loading weights**.
//! Actual loading + generation is **guarded**: the Qwen3 text encoder (S1), the 32-ch VAE (S2),
//! and the MMDiT transformer (S3) don't exist yet, so `load()` returns a clear,
//! slice-referencing error. `validate()` is real (and tested) — it gates request shape/size now
//! and will gate the wired model later.

use mlx_gen::{
    Error, GenerationOutput, GenerationRequest, Generator, LoadSpec, ModelDescriptor,
    ModelRegistration, Progress, Result,
};

use crate::config::Flux2Variant;

pub fn descriptor_klein_9b() -> ModelDescriptor {
    Flux2Variant::Klein9b.descriptor()
}

pub fn descriptor_klein_9b_edit() -> ModelDescriptor {
    Flux2Variant::Klein9bEdit.descriptor()
}

pub fn load_klein_9b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9b, spec)
}

pub fn load_klein_9b_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9bEdit, spec)
}

fn load_variant(variant: Flux2Variant, _spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    // S0 guard: the model modules (Qwen3 TE / FLUX.2 VAE / MMDiT) are not ported yet.
    Err(Error::Msg(format!(
        "{}: model loading lands across S1 (Qwen3 text encoder), S2 (FLUX.2 VAE), and S3 \
         (MMDiT transformer); S0 ships the scaffold, config, flow-match schedule, 2×2 \
         pack/unpack, the 4-axis RoPE table, and the latent/text id builders only",
        variant.id()
    )))
}

/// The FLUX.2-klein generator. S0 is a registration/validation shell — the model fields land in
/// S1–S3 and `generate()` is wired in S4 (txt2img) / S5 (edit).
pub struct Flux2 {
    descriptor: ModelDescriptor,
    #[allow(dead_code)]
    variant: Flux2Variant,
}

impl Flux2 {
    /// Construct a weightless instance for validation tests.
    pub fn new_for_tests(variant: Flux2Variant) -> Self {
        Self {
            descriptor: variant.descriptor(),
            variant,
        }
    }
}

impl Generator for Flux2 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        validate_request(&self.descriptor, req)
    }

    fn generate(
        &self,
        _req: &GenerationRequest,
        _on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        Err(Error::Msg(format!(
            "{}: generation is wired in S4 (txt2img) / S5 (edit); S0 is scaffold only",
            self.descriptor.id
        )))
    }
}

fn validate_request(desc: &ModelDescriptor, req: &GenerationRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(Error::Msg(format!("{}: prompt is required", desc.id)));
    }
    if !req.width.is_multiple_of(16) || !req.height.is_multiple_of(16) {
        return Err(Error::Msg(format!(
            "{}: width and height must be multiples of 16, got {}x{}",
            desc.id, req.width, req.height
        )));
    }
    let caps = &desc.capabilities;
    if req.width < caps.min_size
        || req.height < caps.min_size
        || req.width > caps.max_size
        || req.height > caps.max_size
    {
        return Err(Error::Msg(format!(
            "{}: size {}x{} outside supported range {}..={}",
            desc.id, req.width, req.height, caps.min_size, caps.max_size
        )));
    }
    if req.count == 0 || req.count > caps.max_count {
        return Err(Error::Msg(format!(
            "{}: count must be 1..={}",
            desc.id, caps.max_count
        )));
    }
    if req.negative_prompt.is_some() && !caps.supports_negative_prompt {
        return Err(Error::Msg(format!(
            "{}: negative prompts are not supported by FLUX.2",
            desc.id
        )));
    }
    if req.true_cfg.is_some() && !caps.supports_true_cfg {
        return Err(Error::Msg(format!(
            "{}: true_cfg is not supported",
            desc.id
        )));
    }
    for c in &req.conditioning {
        let kind = conditioning_kind(c);
        if !caps.accepts(kind) {
            return Err(Error::Msg(format!(
                "{}: conditioning {kind:?} is not supported by this variant",
                desc.id
            )));
        }
    }
    Ok(())
}

fn conditioning_kind(c: &mlx_gen::Conditioning) -> mlx_gen::ConditioningKind {
    use mlx_gen::{Conditioning as C, ConditioningKind as K};
    match c {
        C::Reference { .. } => K::Reference,
        C::MultiReference { .. } => K::MultiReference,
        C::ReduxRefs { .. } => K::ReduxRefs,
        C::Control { .. } => K::Control,
        C::Depth { .. } => K::Depth,
        C::Mask { .. } => K::Mask,
    }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_klein_9b, load: load_klein_9b }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_klein_9b_edit, load: load_klein_9b_edit }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FLUX2_KLEIN_9B_EDIT_ID, FLUX2_KLEIN_9B_ID};
    use mlx_gen::media::Image;
    use mlx_gen::Conditioning;

    #[test]
    fn validates_basic_txt2img_request() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "a hummingbird".into(),
            ..Default::default()
        };
        model.validate(&req).unwrap();
    }

    #[test]
    fn rejects_empty_prompt() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest::default();
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("prompt is required"));
    }

    #[test]
    fn rejects_non_multiple_of_16() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            width: 1023,
            ..Default::default()
        };
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("multiples of 16"));
    }

    #[test]
    fn txt2img_rejects_reference_conditioning() {
        // img2img (Reference) is sc-2644, not this story's txt2img variant.
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: None,
            }],
            ..Default::default()
        };
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("conditioning"));
    }

    #[test]
    fn edit_accepts_single_reference() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9bEdit);
        let req = GenerationRequest {
            prompt: "make it night".into(),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: None,
            }],
            ..Default::default()
        };
        model.validate(&req).unwrap();
    }

    #[test]
    fn generate_is_guarded_in_s0() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            ..Default::default()
        };
        let mut progress = |_p: Progress| {};
        let err = model.generate(&req, &mut progress).unwrap_err().to_string();
        assert!(err.contains("S4"));
    }

    #[test]
    fn ids_match_expected() {
        assert_eq!(descriptor_klein_9b().id, FLUX2_KLEIN_9B_ID);
        assert_eq!(descriptor_klein_9b_edit().id, FLUX2_KLEIN_9B_EDIT_ID);
    }
}
