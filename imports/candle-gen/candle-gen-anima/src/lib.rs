//! # candle-gen-anima
//!
//! The **Anima** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA) sibling of
//! `mlx-gen-anima` (epic 10512, sc-10525). Three variants share **one architecture** and differ only
//! in the DiT weights file:
//! - **`anima_base`** — the base model (30 steps, CFG 4.5),
//! - **`anima_aesthetic`** — the aesthetic fine-tune (30 steps, CFG 4.5),
//! - **`anima_turbo`** — the merged CFG-free few-step student (10 steps, CFG 1.0).
//!
//! ## Architecture (transcribed from the MLX port; no candle-transformers reference)
//! - **DiT** — the **Cosmos-Predict2** `CosmosTransformer3DModel` (28 layers, hidden 2048 = 16×128,
//!   patch `(1,2,2)`, adaLN-LoRA 256, 3-axis NTK RoPE `rope_scale (1,4,4)`, `concat_padding_mask` ⇒
//!   **17-channel** patch-embed input) — [`transformer::CosmosDiT`].
//! - **Text conditioner** — the **`AnimaTextConditioner`** (bundled under `{prefix}.llm_adapter.*`):
//!   `nn.Embedding(32128, 1024)` over T5 ids → 6 × [self-attn → cross-attn into Qwen3 states → GELU
//!   MLP] → out_proj + RMSNorm, right-padded to **512** — [`conditioner::AnimaTextConditioner`].
//! - **Text encoder** — **Qwen3-0.6B base** (`last_hidden_state`, GQA 16/8 handled with `repeat_kv`) —
//!   [`text_encoder::AnimaQwen3`].
//! - **VAE** — the **Qwen-Image** `AutoencoderKLQwenImage`, reusing [`vae::QwenVae`] (from
//!   `candle-gen-qwen-image`) via the original→diffusers key rename [`vae::convert_vae_key`].
//! - **Scheduler** — `FlowMatchEulerDiscreteScheduler` static `shift=3.0`, `sigmas = linspace(1, 1/N, N)`
//!   ([`pipeline::anima_sigmas`]); default solver the recommended `er_sde` (sc-10519), carried by the
//!   `441ecec` gen-core pin ([`pipeline::DEFAULT_SAMPLER`]).
//!
//! **`backend = "candle"`, `mac_only = false`** — this crate is what lets the manifest drop the
//! `macOnly: true` gate the MLX-only port carried (sc-10523 wires it worker-side).
//!
//! **Surface:** txt2img at the single-file dense checkpoint, with **LoRA/LoKr injection** (448 DiT + 60
//! conditioner targets folded at load, stacked + mixed, strict routing — [`adapters`]). Q4/Q8 candle
//! quant tiers are the counterpart of MLX sc-10517 (see the `quant` gap note in [`loader`]).

pub mod adapters;
pub mod conditioner;
pub mod config;
pub mod loader;
pub mod nn;
pub mod pipeline;
pub mod rope;
pub mod text_encoder;
pub mod tokenizer;
pub mod transformer;
pub mod vae;

pub use conditioner::AnimaTextConditioner;
pub use config::{ConditionerConfig, DitConfig, Qwen3Config, Variant};
pub use loader::{detect_dit_prefix, AnimaComponents};
pub use pipeline::{anima_sigmas, AnimaPipeline, GenOptions, DEFAULT_SAMPLER};
pub use text_encoder::AnimaQwen3;
pub use transformer::CosmosDiT;
pub use vae::{load_vae, QwenVae};

use std::sync::Mutex;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, LoadSpec, Modality,
    ModelDescriptor, Progress, Quant, WeightsSource,
};

use crate::config::RES_MULTIPLE;

/// The candle quant tiers Anima advertises — Q4 + Q8 (the counterpart of MLX sc-10517). The DiT loads
/// packed (dequant-dense per step, CPU-capable); the conditioner / Qwen3 TE / VAE stay dense bf16.
const ANIMA_QUANTS: &[Quant] = &[Quant::Q4, Quant::Q8];

const MAX_COUNT: u32 = 8;
const RES_MIN: u32 = 512;
/// Above ~1920 px/side the Cosmos RoPE would index out of its trained range; `rope.rs` **rejects**
/// such a request rather than clamping. The shipped ceiling is 1536² (post-patch 96 < the 120-position
/// max_size), so the guard is unreachable via the normal path. See [`crate::rope`].
const RES_MAX: u32 = 1536;

/// Build the descriptor for a variant. Turbo is the merged CFG-free student (no guidance / negative
/// prompt); Base/Aesthetic run true classifier-free guidance.
fn descriptor_for(variant: Variant) -> ModelDescriptor {
    let cfg_capable = variant.uses_cfg();
    ModelDescriptor {
        id: variant.id(),
        family: "anima",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: cfg_capable,
            supports_guidance: cfg_capable,
            supports_true_cfg: false,
            conditioning: vec![],
            // LoRA/LoKr injection is wired (the candle counterpart of MLX sc-10521): every trained
            // adapter's 448 DiT + 60 conditioner targets fold at load, stacked + mixed, strict routing
            // (`adapters::apply_anima_adapters`). Weight-level fold, validated bit-exact on CPU.
            supports_lora: true,
            supports_lokr: true,
            // Rectified-flow over the unified curated-sampler framework (epic 7114). The native default
            // (req.sampler == None) is the recommended er_sde solver; the full curated menu is advertised.
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            supported_guidance_methods: vec![],
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            // The whole point of the candle port: Anima is no longer Mac-only.
            mac_only: false,
            // Q4 + Q8 (the candle counterpart of MLX sc-10517): the DiT packed-detects and runs the
            // dequant-dense forward (CPU-capable — NOT the CUDA-only int8 fast GEMM); conditioner /
            // Qwen3 TE / VAE stay dense bf16. A pre-packed tier is a real, loadable snapshot.
            supported_quants: ANIMA_QUANTS,
            supports_kv_cache: false,
            requires_sigma_shift: true,
        },
    }
}

pub fn descriptor_base() -> ModelDescriptor {
    descriptor_for(Variant::Base)
}
pub fn descriptor_aesthetic() -> ModelDescriptor {
    descriptor_for(Variant::Aesthetic)
}
pub fn descriptor_turbo() -> ModelDescriptor {
    descriptor_for(Variant::Turbo)
}

/// A loaded Anima generator: the cached descriptor + variant + lazily-built pipeline (mirrors the
/// candle-gen-qwen-image lazy component cache).
pub struct Anima {
    descriptor: ModelDescriptor,
    variant: Variant,
    root: WeightsSource,
    device: Device,
    /// LoRA/LoKr adapters to bake onto the DiT + conditioner at pipeline build (empty for the plain
    /// model). Captured at load; folded lazily when the pipeline is first assembled.
    adapters: Vec<gen_core::AdapterSpec>,
    pipeline: Mutex<Option<AnimaPipeline>>,
}

pub fn load_base(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Base)
}
pub fn load_aesthetic(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Aesthetic)
}
pub fn load_turbo(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Turbo)
}

fn load_variant(spec: &LoadSpec, variant: Variant) -> gen_core::Result<Box<dyn Generator>> {
    let id = variant.id();
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: candle Anima is txt2img only (no control / IP-adapter)"
        )));
    }
    // A Q4/Q8 tier is a **pre-packed** snapshot (the worker points `spec.weights` at it; the DiT
    // packed-detects it at load). Folding a LoRA/LoKr into u32-packed codes is a separate dequant-fold
    // job — reject the packed+adapter COMBO rather than silently mis-merge into the codes.
    if spec.quantize.is_some() && !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: LoRA/LoKr on a quantized (Q4/Q8) tier is not wired — apply the adapter to the dense \
             checkpoint, or use the dense tier"
        )));
    }
    // LoRA/LoKr adapters (`spec.adapters`) are accepted — folded onto the DiT + conditioner when the
    // pipeline is assembled (`adapters::apply_anima_adapters`).
    let device = candle_gen::default_device().map_err(gen_core::Error::from)?;
    Ok(Box::new(Anima {
        descriptor: descriptor_for(variant),
        variant,
        root: spec.weights.clone(),
        device,
        adapters: spec.adapters.clone(),
        pipeline: Mutex::new(None),
    }))
}

impl Anima {
    /// Lazily assemble (and cache) the pipeline on first `generate`.
    fn pipeline(&self) -> gen_core::Result<()> {
        let mut guard = self
            .pipeline
            .lock()
            .expect("anima pipeline cache mutex poisoned");
        if guard.is_none() {
            *guard = Some(AnimaPipeline::from_source(
                &self.root,
                self.variant,
                &self.device,
                &self.adapters,
            )?);
        }
        Ok(())
    }
}

impl Generator for Anima {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        validate_request(&self.descriptor, req)?;
        self.pipeline()?;
        let guard = self
            .pipeline
            .lock()
            .expect("anima pipeline cache mutex poisoned");
        let pipeline = guard.as_ref().expect("pipeline built above");

        let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let steps = req.steps.unwrap_or(self.variant.default_steps()) as usize;
        let guidance = if self.variant.uses_cfg() {
            req.guidance.unwrap_or(self.variant.default_guidance())
        } else {
            1.0
        };
        let negative = req.negative_prompt.clone().unwrap_or_default();

        let mut images = Vec::with_capacity(req.count as usize);
        for n in 0..req.count {
            let opts = GenOptions {
                width: req.width,
                height: req.height,
                steps,
                guidance,
                seed: candle_gen::image_seed(base_seed, n),
                sampler: req.sampler.clone(),
            };
            let img = pipeline.generate(
                &req.prompt,
                &negative,
                self.variant,
                &opts,
                &req.cancel,
                on_progress,
            )?;
            images.push(img);
        }
        Ok(GenerationOutput::Images(images))
    }
}

/// Capability-driven request validation (testable without loaded weights): non-empty prompt, size a
/// multiple of 16, on top of the shared [`Capabilities::validate_request`] floor.
pub(crate) fn validate_request(
    desc: &ModelDescriptor,
    req: &GenerationRequest,
) -> gen_core::Result<()> {
    let id = desc.id;
    if req.prompt.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{id}: prompt must not be empty"
        )));
    }
    desc.capabilities.validate_request(id, req)?;
    if !req.width.is_multiple_of(RES_MULTIPLE) || !req.height.is_multiple_of(RES_MULTIPLE) {
        return Err(gen_core::Error::Msg(format!(
            "{id}: {}x{} must be a multiple of {RES_MULTIPLE}",
            req.width, req.height
        )));
    }
    Ok(())
}

// Link-time registration of all three variants.
candle_gen::register_generators! {
    descriptor_base => load_base,
    descriptor_aesthetic => load_aesthetic,
    descriptor_turbo => load_turbo,
}

/// Force-link hook (keeps the `inventory::submit!` registrations from being dead-stripped).
pub fn force_link() {}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::registry;

    fn req(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "an anime girl with silver hair, detailed".into(),
            width: w,
            height: h,
            ..Default::default()
        }
    }

    #[test]
    fn three_variants_registered_as_candle() {
        for id in ["anima_base", "anima_aesthetic", "anima_turbo"] {
            let g = registry::load(
                id,
                &LoadSpec::new(WeightsSource::Dir("/nonexistent".into())),
            )
            .unwrap_or_else(|_| panic!("id {id} not registered"));
            assert_eq!(g.descriptor().id, id);
            assert_eq!(g.descriptor().family, "anima");
            assert_eq!(g.descriptor().backend, "candle");
        }
    }

    #[test]
    fn descriptors_surface() {
        let b = descriptor_base();
        assert_eq!(b.id, "anima_base");
        assert_eq!(b.backend, "candle");
        assert_eq!(b.modality, Modality::Image);
        assert!(b.capabilities.supports_guidance);
        assert!(b.capabilities.supports_negative_prompt);
        assert!(b.capabilities.requires_sigma_shift);
        // The candle port removes the Mac-only gate.
        assert!(!b.capabilities.mac_only);
        // LoRA/LoKr injection is wired; Q4/Q8 tiers are advertised (packed-detect, dequant-dense).
        assert!(b.capabilities.supports_lora && b.capabilities.supports_lokr);
        assert_eq!(b.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert_eq!(b.capabilities.min_size, 512);
        assert_eq!(b.capabilities.max_size, 1536);
        // Turbo is the CFG-free merged student.
        let t = descriptor_turbo();
        assert!(!t.capabilities.supports_guidance);
        assert!(!t.capabilities.supports_negative_prompt);
        // The default flow solver (er_sde on the 441ecec gen-core pin) is a real curated sampler.
        assert_eq!(pipeline::DEFAULT_SAMPLER, "er_sde");
        assert!(
            b.capabilities.samplers.contains(&pipeline::DEFAULT_SAMPLER),
            "er_sde advertised in the curated menu (441ecec gen-core pin carries the ErSde solver)"
        );
    }

    #[test]
    fn validate_rejects_bad_requests() {
        assert!(validate_request(&descriptor_base(), &GenerationRequest::default()).is_err()); // empty prompt
        assert!(validate_request(&descriptor_base(), &req(1000, 1024)).is_err()); // not mult of 16
        assert!(validate_request(&descriptor_base(), &req(256, 256)).is_err()); // below min
        assert!(validate_request(&descriptor_base(), &req(2048, 2048)).is_err()); // above max
        assert!(validate_request(&descriptor_base(), &req(1024, 1024)).is_ok());
        assert!(validate_request(&descriptor_base(), &req(1536, 1536)).is_ok());
    }

    #[test]
    fn load_accepts_adapters_and_quant_but_rejects_the_combo_and_control() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        let base = WeightsSource::Dir("/snap".into());
        let lora_spec = || {
            vec![AdapterSpec::new(
                "/lora.safetensors".into(),
                1.0,
                AdapterKind::Lora,
            )]
        };

        // A plain LoRA load and a plain Q4 load both SUCCEED (lazily built; the fold / packed-detect
        // happen at first generate). Advertising the capability and then rejecting at load would be a lie.
        assert!(load_base(&LoadSpec::new(base.clone()).with_adapters(lora_spec())).is_ok());
        assert!(load_base(&LoadSpec::new(base.clone()).with_quant(Quant::Q4)).is_ok());

        // The packed+LoRA COMBO is rejected (folding into u32 codes is a dequant-fold follow-up).
        let combo = LoadSpec::new(base.clone())
            .with_quant(Quant::Q8)
            .with_adapters(lora_spec());
        assert!(matches!(
            load_base(&combo).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
    }
}
