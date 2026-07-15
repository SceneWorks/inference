//! # candle-gen-mochi
//!
//! The **Mochi 1** (`genmo/mochi-1-preview`, Apache-2.0) text-to-video provider for
//! [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling of `mlx-gen-mochi`. Mochi is a
//! T5-XXL-conditioned dual-stream MMDiT (the **AsymmDiT**) with an asymmetric 3-D causal-conv VAE
//! (6× temporal, 8× spatial). It has **no** `candle-transformers` reference: the masked T5 encode
//! ([`text_encoder`], run through the reused `candle_gen_flux::packed_te::PackedT5Encoder` + the
//! tokenizer key-padding mask Mochi's `_get_t5_prompt_embeds` applies), the learned 3-D RoPE ([`rope`]),
//! the linear-quadratic flow-match scheduler ([`scheduler`]), the dual-stream AsymmDiT denoiser
//! ([`transformer`]), and the AsymmVAE decoder ([`vae`], on a from-scratch conv2d-tap [`conv3d`]) are
//! all ported here, preserving the exact `mlx-gen-mochi` math.
//!
//! **txt2video, true CFG:** Mochi is **not** distilled, so it exposes negative-prompt + `guidance`
//! true classifier-free guidance over the `[neg, pos]` batch. The denoise runs the inverted
//! linear-quadratic Euler loop with the CFG recombine `uncond + g·(cond − uncond)` inside the predict
//! step ([`pipeline`]).
//!
//! **Dtypes:** the AsymmDiT + T5 store weights at **bf16** (the checkpoint's native dtype; the 10B DiT
//! does not fit f32 on a single consumer GPU), the AsymmVAE runs **f32**. The AsymmDiT **computes in
//! f32** (the MLX parity regime — the `nn::linear_*` helpers upcast each bf16 weight to f32 at the
//! matmul). `backend = "candle"`, `mac_only = false`. Quant tiers ship as pre-quantized per-tier
//! checkpoints (epic 1788 / A6), *not* on-the-fly requant, so `supported_quants` is empty.

pub mod config;
pub mod conv3d;
pub mod nn;
pub mod pipeline;
pub mod rope;
pub mod scheduler;
pub mod text_encoder;
pub mod tokenizer;
pub mod transformer;
pub mod vae;

use std::path::PathBuf;
use std::sync::Mutex;

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, Quant, WeightsSource,
};

pub use config::{MochiConfig, MochiVaeConfig};
pub use pipeline::{denoise, frames_to_images, to_uint8_frames, Components, Pipeline};
pub use rope::{get_positions, MochiRope};
pub use scheduler::{cfg_combine, linear_quadratic_schedule, MochiScheduler};
pub use text_encoder::{encode_prompt, load_indexed_var_builder, MochiT5, MochiTextConditioning};
pub use tokenizer::{load_tokenizer, MAX_SEQUENCE_LENGTH, PAD_TOKEN_ID};
pub use transformer::{
    load_transformer_var_builder, MochiAttention, MochiDitConfig, MochiTransformer3DModel,
    MochiTransformerBlock,
};
pub use vae::MochiVaeDecoder;

/// Public provider id: `"mochi_1"`.
pub const MODEL_ID: &str = "mochi_1";

/// The AsymmDiT + T5 weight-storage dtype (the checkpoint's native bf16; the DiT then computes f32).
pub const DIT_DTYPE: DType = DType::BF16;
/// The AsymmVAE compute dtype (the decoder is numerically f32-only — bf16 intermediates reach O(100)).
pub const VAE_DTYPE: DType = DType::F32;

/// Width/height must be divisible by 16 (VAE 8× spatial × DiT patch 2).
const SIZE_MULTIPLE: u32 = 16;
/// The AsymmVAE temporal ratio: a valid clip length is `1 + 6·k`.
const TEMPORAL_RATIO: u32 = 6;

// Production defaults when the request leaves a knob unset.
/// Diffusers `MochiPipeline` default `num_inference_steps` for the preview.
pub(crate) const DEFAULT_STEPS: u32 = 64;
/// Diffusers `MochiPipeline` default `guidance_scale`.
pub(crate) const DEFAULT_GUIDANCE: f32 = 4.5;
/// A safe default frame count on the `6·k + 1` lattice (`6·3 + 1`).
pub(crate) const DEFAULT_FRAMES: u32 = 19;
/// Mochi renders ~30 fps.
pub(crate) const DEFAULT_FPS: u32 = 30;

/// Stable identity + advertised capabilities for Mochi 1 (text-to-video, true CFG, no audio).
///
/// Mochi is **text-to-video only** in the base preview (no audio, no I2V). It is **not** distilled, so
/// it exposes **true CFG** (negative prompt + `guidance` scale). Quant tiers ship as pre-quantized
/// per-tier checkpoints (epic 1788 / A6), *not* on-the-fly requant, so [`Capabilities::supported_quants`]
/// is empty and `load` rejects a stray `spec.quantize`.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "mochi",
        backend: "candle",
        modality: Modality::Video,
        capabilities: Capabilities {
            // Not distilled → true classifier-free guidance over a [neg, pos] batch.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // Text-to-video only in the base preview (I2V = a follow-on).
            conditioning: Vec::new(),
            supports_lora: false,
            supports_lokr: false,
            // A single fixed flow-match Euler integrator is wired; no selectable sampler/scheduler axis.
            samplers: Vec::new(),
            schedulers: Vec::new(),
            supported_guidance_methods: Vec::new(),
            // Width/height must be divisible by 16 (VAE 8× spatial × DiT patch 2). 480p target = 848×480.
            min_size: SIZE_MULTIPLE,
            max_size: 1280,
            max_count: 1,
            mac_only: false,
            // Quant tiers are pre-quantized per-tier checkpoints (epic 1788 / A6) — NOT on-the-fly requant.
            supported_quants: &[] as &[Quant],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
        },
    }
}

/// The lazy candle Mochi 1 generator. `spec.weights` is a `genmo/mochi-1-preview` diffusers snapshot
/// dir (`tokenizer/` vendored, `text_encoder/`, `transformer/`, `vae/`).
pub struct MochiGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    components: Mutex<Option<Components>>,
}

impl MochiGenerator {
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        Ok(candle_gen::cached(&self.components, || {
            pipe.load_components()
        })?)
    }
}

impl Generator for MochiGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    /// Reject a request Mochi cannot serve: the shared capability floor plus the model-specific
    /// constraints — non-empty prompt, 16-divisible width/height, and `num_frames = 1 + 6·k`.
    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        if req.prompt.is_empty() {
            return Err(gen_core::Error::Msg(
                "mochi_1: prompt must not be empty".into(),
            ));
        }
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
            return Err(gen_core::Error::Msg(format!(
                "mochi_1: width/height must be divisible by {SIZE_MULTIPLE} (got {}x{})",
                req.width, req.height
            )));
        }
        if let Some(frames) = req.frames {
            // The AsymmVAE has a 6× temporal ratio, so a valid clip length is `1 + 6·k` latent-aligned.
            if frames % TEMPORAL_RATIO != 1 {
                return Err(gen_core::Error::Msg(format!(
                    "mochi_1: num_frames must be 1 + 6·k (got {frames})"
                )));
            }
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let pipe = Pipeline::new(&self.root, &self.device);
        let components = self.components(&pipe)?;
        let (frames, fps) = pipe.render(req, &components, on_progress)?;
        Ok(GenerationOutput::Video {
            frames,
            fps,
            audio: None,
        })
    }
}

/// Construct a lazy candle Mochi 1 generator. `spec.weights` is a `genmo/mochi-1-preview` snapshot dir
/// (a split-weight diffusers layout). Adapters / on-the-fly quantization / conditioning are rejected
/// (not wired — Mochi ships pre-quantized per-tier checkpoints).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(gen_core::Error::Msg(
                "mochi_1: expected a model directory (split-weight snapshot), not a single file"
                    .into(),
            )),
        };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle mochi does not support LoRA/LoKr".into(),
        ));
    }
    // On-the-fly requant is not the Mochi tier mechanism (epic 1788: self-contained pre-quantized
    // q4/q8/bf16 tier dirs, A6). Reject a stray `spec.quantize` rather than silently ignore it.
    if let Some(q) = spec.quantize {
        return Err(gen_core::Error::Unsupported(format!(
            "mochi_1: spec.quantize={q:?} unsupported — Mochi ships pre-quantized per-tier \
             checkpoints; point WeightsSource at the q4/q8/bf16 tier dir (no on-the-fly requant)"
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle mochi does not support image / I2V conditioning yet (txt2video only)".into(),
        ));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(MochiGenerator {
        descriptor: descriptor(),
        root,
        device,
        components: Mutex::new(None),
    }))
}

candle_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}

/// Add the Candle Mochi 1 generator to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry.register_generator(REGISTRATION)
}

/// Build the complete explicit Candle Mochi 1 provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    use super::*;

    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = provider_registry().unwrap();
        let generators: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();
        assert_eq!(generators, ["mochi_1"]);
    }

    #[test]
    fn registered_descriptor_conforms() {
        let registry = provider_registry().unwrap();
        assert!(
            registry.descriptor_conformance_errors().is_empty(),
            "mochi descriptor conformance violations: {:?}",
            registry.descriptor_conformance_errors()
        );
    }

    #[test]
    fn descriptor_identity_and_capabilities() {
        let d = descriptor();
        assert_eq!(d.id, "mochi_1");
        assert_eq!(d.family, "mochi");
        assert_eq!(d.backend, "candle");
        assert_eq!(d.modality, Modality::Video);
        let c = &d.capabilities;
        assert!(c.supports_negative_prompt);
        assert!(c.supports_guidance);
        assert!(c.supports_true_cfg);
        assert!(!c.mac_only);
        assert_eq!(c.max_count, 1);
        assert!(c.conditioning.is_empty(), "t2v-only: no conditioning kinds");
        assert!(c.supported_quants.is_empty(), "no on-the-fly requant");
    }
}
