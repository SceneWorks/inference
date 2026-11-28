//! # candle-gen-sd3
//!
//! The **Stable Diffusion 3.5** provider crate for [`candle-gen`](candle_gen) — the candle
//! (Windows/CUDA) sibling of `mlx-gen-sd3` (epic 7841). It ports the SD3.5 architecture to candle:
//! the **MMDiT** (joint / double-stream) transformer ([`transformer`]), the **triple text-encoder
//! aggregator** ([`conditioning`]) that combines CLIP-L + OpenCLIP bigG + T5-XXL into the pooled
//! (2048) + context (333×4096) conditioning, and the 16-channel **VAE** wiring ([`vae`]).
//!
//! **C1 scope (sc-7876):** this story is the *architecture foundation* — the FULL SD3.5 Large
//! forward at real shapes, the aggregator, the VAE config, and crate registration, with
//! shape/structural parity tests. The actual T2I **pipeline** (text encoder forward + flow-match
//! sampler + VAE decode) and the CUDA real-weight smoke land in **C2** (sc-7877); the registered
//! descriptor here therefore advertises the txt2img surface but [`generate`](Generator::generate)
//! returns a typed `Unsupported` until C2 wires the pipeline. This keeps the worker honest (it falls
//! back rather than calling an un-wired path) while the registration + architecture are in place.
//!
//! **Parity approach (re-validated):** the macOS `mlx-gen-sd3` reference is NOT present on this
//! Windows machine, so correctness is established the way epic 7841 did — line-by-line against the
//! PUBLIC diffusers `SD3Transformer2DModel` / `StableDiffusion3Pipeline` and the SD3 paper, plus
//! structural/shape unit tests (the `#[cfg(test)]` modules in each sub-module). Any cross-engine
//! numeric A/B harness is left `#[ignore]`d (reference unavailable here); coherent real-weight
//! rendering is validated in C2/C6.
//!
//! **Key SD3.5 vs FLUX parity note:** SD3.5's MMDiT uses a **learned 2D positional embedding** added
//! at patchify and runs attention with **NO RoPE** — do not copy FLUX's rotary embedding. The AdaLN
//! modulation apply order (`x·(1+scale)+shift`, with the correct chunk split) is the documented bug
//! magnet; see [`transformer`].

// Private opt-in model-proof plumbing; feature-OFF observers expand to the original expression.
#[macro_use]
#[path = "../../../../llm/test-support/model_proof.rs"]
mod model_proof;
const _: fn() = || {
    let _: std::result::Result<candle_core::Tensor, ()> =
        __model_proof_output!(std::result::Result::<candle_core::Tensor, ()>::Err(()));
};

pub mod adapters;
pub mod clip_tokenizer;
pub mod conditioning;
pub mod config;
pub mod memory;
pub mod memory_strategy;
pub mod pipeline;
pub mod preview;
pub mod quant;
pub mod training;
pub mod transformer;
pub mod vae;

// SD3.5 img2img/`Reference` real-weight GPU validation (sc-11784) — env-driven, `#[ignore]`d
// integration tests driving the REGISTERED Large/Medium (true-CFG) + Turbo (distilled) generators
// through a `Conditioning::Reference` (strength ablation + the strength-1.0 source round-trip + prompt
// divergence). The candle/CUDA parity of the mlx-gen-sd3 `denoise_img2img_cfg` lane (sc-10189).
#[cfg(test)]
mod img2img_validate;

pub use adapters::{install_additive, merge_adapters, AdditiveReport, MergeReport};
pub use config::Sd3Config;
pub use memory::{min_memory_gb, MemoryProfile, ParamCounts, Precision};
pub use pipeline::Variant;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::{
    self, AdapterSpec, ConditioningKind, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    Modality, ModelDescriptor, Progress, Quant, WeightsSource,
};

use pipeline::{Components, Pipeline};
use vae::VaeEncoder;

/// Registry id for SD3.5 **Large** — matches the SceneWorks worker's `payload.model` and the macOS
/// `mlx-gen-sd3` SD3.5-Large descriptor.
pub const MODEL_ID: &str = "sd3_5_large";

/// Registry id for SD3.5 **Large Turbo** — the guidance-distilled 4-step sibling.
pub const MODEL_ID_TURBO: &str = "sd3_5_large_turbo";

/// Registry id for SD3.5 **Medium** — the MMDiT-X (dual-attention) model. Matches the SceneWorks
/// worker's `payload.model` and the macOS `mlx-gen-sd3` SD3.5-Medium descriptor.
pub const MODEL_ID_MEDIUM: &str = "sd3_5_medium";

/// SD3.5 works in latent space at /8 and the MMDiT patchifies that at /2, so both image dims must be
/// multiples of **16** for a clean patchify. Exposed as the pinned-engine stride SceneWorks ties each
/// advertised SD3.5 image bucket to (sc-12612), mirroring `wan::config::SIZE_MULTIPLE_14B`. `validate`
/// enforces exactly this value, so the const cannot drift from the check.
pub const SIZE_MULTIPLE: u32 = 16;

/// A loaded candle SD3.5 generator. Loading is **lazy**: `load` does no file I/O, and the heavy
/// components (triple text encoders + MMDiT + VAE) are built on the first
/// [`generate`](Generator::generate) call and then **cached** so back-to-back requests skip the disk
/// re-read. The [`Variant`] (Large vs Large Turbo) is fixed at load from the registered id.
pub struct Sd3Generator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    dtype: DType,
    variant: Variant,
    /// Optional DiT quantization (`Q4`/`Q8`) applied at first component load (sc-7879). `None` ⇒
    /// dense bf16. The text encoders + VAE stay dense regardless.
    quant: Option<Quant>,
    /// LoRA/LoKr adapters merged into the MMDiT weights at component-load (sc-7881). Empty ⇒ the stock
    /// mmap build (zero regression). Fixed at load from `spec.adapters`.
    adapters: Vec<AdapterSpec>,
    /// Cached components. `Mutex` because `Generator` is shared and `generate` takes `&self`; the lock
    /// is held only to read/populate the cache, never across the denoise.
    components: Mutex<Option<Components>>,
    /// Lazily-built, cached f32 VAE **encoder** for the img2img / `Reference` path (sc-11784). Built on
    /// the first request that carries a reference AND yields a structure-preserving denoise
    /// (`start_step > 0`), so a txt2img-only workload never builds it. Cached here (not in `Components`)
    /// because it loads in f32 for the deterministic mean-encode while `Components` is bf16.
    vae_encoder: Mutex<Option<Arc<VaeEncoder>>>,
    /// Serializes cache transition + generation so a staged request cannot race a resident warm hit.
    lifecycle: Mutex<()>,
    loaded_spec: LoadSpec,
    memory_contract: gen_core::MemoryProviderContract,
    load_receipt: Option<memory_strategy::Sd35LoadReceipt>,
    memory_admission: memory_strategy::AdmissionRegistry,
}

impl Sd3Generator {
    /// The registry id for this generator's variant (used in validation error prefixes).
    fn model_id(&self) -> &'static str {
        match self.variant {
            Variant::Large => MODEL_ID,
            Variant::LargeTurbo => MODEL_ID_TURBO,
            Variant::Medium => MODEL_ID_MEDIUM,
        }
    }

    /// Get the cached components, loading (and caching) them on a miss.
    fn components(&self, pipe: &Pipeline) -> gen_core::Result<Components> {
        if let Some(receipt) = &self.load_receipt {
            receipt.ensure_unchanged()?;
        }
        // `?` bridges the candle-side `load_components` error into `gen_core::Error`.
        Ok(candle_gen::cached(&self.components, || {
            pipe.load_components()
        })?)
    }

    /// Get the cached f32 VAE encoder for the img2img / `Reference` path (sc-11784), building it on a
    /// miss. Only reached when a request carries a reference that yields a structure-preserving denoise
    /// (`start_step > 0`), so a txt2img-only workload never builds it.
    fn vae_encoder(&self, pipe: &Pipeline) -> gen_core::Result<Arc<VaeEncoder>> {
        if let Some(receipt) = &self.load_receipt {
            receipt.ensure_unchanged()?;
        }
        // The inner `?` bridges the candle-side `load_vae_encoder` error into `gen_core::Error`.
        candle_gen::cached(&self.vae_encoder, || Ok(Arc::new(pipe.load_vae_encoder()?)))
    }
}

fn evict_warm_for_staged<T>(slot: &Mutex<Option<T>>) -> Option<T> {
    candle_gen::lock_recover(slot).take()
}

fn release_warm_after_synchronize<T>(device: &Device, component: T) -> gen_core::Result<()> {
    match device.synchronize() {
        Ok(()) => {
            drop(component);
            Ok(())
        }
        Err(error) => {
            std::mem::forget(component);
            Err(gen_core::Error::backend(error))
        }
    }
}

impl Generator for Sd3Generator {
    fn generate_with_execution_report(
        &self,
        req: &candle_gen::gen_core::GenerationRequest,
        on_progress: &mut dyn FnMut(candle_gen::gen_core::Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        let plan = compile_trajectory_for(self.variant, req)?.bind_load(&self.loaded_spec);
        self.generate_planned_with_execution_report(req, &plan, None, on_progress)
    }

    fn generate_planned_with_execution_report(
        &self,
        req: &candle_gen::gen_core::GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(candle_gen::gen_core::Progress),
    ) -> candle_gen::gen_core::Result<candle_gen::gen_core::generator::CompletedGeneration> {
        candle_gen::execution::capture_planned_with(self, req, plan, scope, on_progress, || {
            candle_gen::execution::unobserved_adapters(
                self.adapters.len(),
                "candle-gen-sd3/src/lib.rs",
            );
            candle_gen::execution::policy("loaded/dtype", format!("{:?}", self.dtype).into());
            candle_gen::execution::policy("loaded/quant", format!("{:?}", self.quant).into());
        })
    }

    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(&self.memory_contract)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        let Some(receipt) = &self.load_receipt else {
            self.memory_admission.clear();
            return gen_core::MemorySafetyDecision::Reject {
                reason: format!("{}: immutable load receipt is missing", self.model_id()),
            };
        };
        if let Err(error) = receipt.ensure_unchanged().and_then(|()| {
            memory_strategy::validate_context(
                receipt.route,
                &self.memory_contract,
                context,
                receipt.tier,
            )
        }) {
            self.memory_admission.clear();
            return gen_core::MemorySafetyDecision::rejected(error);
        }
        gen_core::MemorySafetyDecision::from_validation(self.memory_admission.approve(context))
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        let receipt = self.load_receipt.as_ref().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{}: immutable load receipt is missing",
                self.model_id()
            ))
        })?;
        memory_strategy::begin_request(
            receipt,
            &self.memory_contract,
            self.memory_admission.clone(),
            self.device.clone(),
            context,
        )
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request_for_variant(self.variant, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_with_report(req, on_progress)?
            .output
            .ok_or_else(|| {
                gen_core::Error::Msg(format!(
                    "{}: generation completed without output",
                    self.model_id()
                ))
            })
    }

    fn generate_with_report(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        let plan = compile_trajectory_for(self.variant, req)?.bind_load(&self.loaded_spec);
        self.generate_planned(req, &plan, None, on_progress)
    }

    fn generate_planned(
        &self,
        req: &GenerationRequest,
        plan: &gen_core::RenderPlan,
        scope: Option<&mut dyn gen_core::MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<gen_core::GenerationReport> {
        self.validate(req)?;
        plan.check_request(&self.descriptor, req)?;
        plan.check_load(&gen_core::RenderLoadBinding::from_spec(&self.loaded_spec))?;
        let _lifecycle = candle_gen::lock_recover(&self.lifecycle);
        self.memory_admission.consume_for_generate(req)?;
        if let Some(receipt) = &self.load_receipt {
            receipt.ensure_unchanged()?;
        }
        let memory = req.memory.unwrap_or_default();
        if memory.tile_vae_decode || memory.chunk_attention || memory.stream_transformer_blocks {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: bounded decode, attention chunking, and transformer windowing are Missing",
                self.model_id()
            )));
        }
        let pipe = Pipeline::load(
            &self.root,
            &self.device,
            self.dtype,
            self.variant,
            self.quant,
            &self.adapters,
        );
        let (_, start_step) = resolved_schedule_for_request(self.variant, req)?;
        let run = gen_core::RenderRun::new(plan, &req.cancel, scope);
        let images = if memory.stage_residency {
            let warm = (
                evict_warm_for_staged(&self.components),
                evict_warm_for_staged(&self.vae_encoder),
            );
            release_warm_after_synchronize(&self.device, warm)?;
            let verify = || {
                self.load_receipt
                    .as_ref()
                    .ok_or_else(|| candle_gen::CandleError::Msg("SD3.5 receipt missing".into()))?
                    .ensure_unchanged()
                    .map_err(|error| candle_gen::CandleError::Msg(error.to_string()))
            };
            pipe.render_staged_planned(req, &run, start_step, on_progress, &verify)?
        } else {
            let components = self.components(&pipe)?;
            let encoder = if start_step > 0 {
                Some(self.vae_encoder(&pipe)?)
            } else {
                None
            };
            pipe.render_planned(req, &run, &components, encoder.as_deref(), on_progress)?
        };
        let mut report = gen_core::GenerationReport::from_output(GenerationOutput::Images(images));
        report.low_res_start = plan.low_res_start_execution_report("bf16", 2, "base_mmdit")?;
        Ok(report)
    }
}

fn validate_request_for_variant(variant: Variant, req: &GenerationRequest) -> gen_core::Result<()> {
    let descriptor = descriptor_for(variant);
    let id = descriptor.id;
    // The shared capability floor (count/size/negative/guidance/conditioning/sampler).
    descriptor.capabilities.validate_request(id, req)?;
    let guidance = req.guidance.unwrap_or_else(|| variant.default_cfg());
    if req.negative_prompt.is_some() && guidance == 1.0 {
        return Err(gen_core::Error::Msg(format!(
            "{id}: a negative prompt requires guidance != 1.0; change guidance or drop the negative prompt"
        )));
    }
    if req.prompt.is_empty() {
        return Err(gen_core::Error::Msg(format!(
            "{id}: prompt must not be empty"
        )));
    }
    if req.steps == Some(0) {
        return Err(gen_core::Error::Msg(format!("{id}: steps must be >= 1")));
    }
    if !req.width.is_multiple_of(SIZE_MULTIPLE) || !req.height.is_multiple_of(SIZE_MULTIPLE) {
        return Err(gen_core::Error::Msg(format!(
            "{id}: width/height must be multiples of {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    if req.frames.is_some_and(|frames| frames != 1) || req.image_reference_count() > 1 {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: image generation requires one frame and either T2I/ref0 or I2I/ref1"
        )));
    }
    let valid_strength = |strength: Option<f32>| {
        strength.is_none_or(|value| value.is_finite() && (0.0..=1.0).contains(&value))
    };
    if !valid_strength(req.strength)
        || req
            .conditioning
            .iter()
            .any(|conditioning| match conditioning {
                gen_core::Conditioning::Reference { strength, .. } => !valid_strength(*strength),
                _ => false,
            })
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: I2I strength must be finite and in [0, 1]"
        )));
    }
    pipeline::resolve_reference(req)?;
    if let Some(low_res) = &req.features.low_res_start {
        let (sigmas, start_step) = resolved_schedule_for_request(variant, req)?;
        let executed_steps = sigmas[start_step..].len().saturating_sub(1);
        low_res.resolve_for_descriptor(&descriptor, req, executed_steps, SIZE_MULTIPLE)?;
    }
    Ok(())
}

fn resolved_schedule_for_request(
    variant: Variant,
    req: &GenerationRequest,
) -> gen_core::Result<(Vec<f32>, usize)> {
    let steps = req
        .steps
        .map(|s| s as usize)
        .unwrap_or_else(|| variant.default_steps());
    let shift = req.scheduler_shift.unwrap_or_else(|| variant.shift());
    let sigmas = pipeline::resolve_sd3_sigmas(req.scheduler.as_deref(), steps, shift);
    let reference = pipeline::resolve_reference(req)?;
    let start_step = match &reference {
        Some((_, strength)) => {
            pipeline::init_time_step(steps, Some(pipeline::resolve_img2img_strength(*strength)))
                .min(sigmas.len().saturating_sub(1))
        }
        None => 0,
    };
    Ok((sigmas, start_step))
}

/// Compile the weights-free Candle SD3.5 trajectory for `variant` and `req` (`SW-6946`).
pub fn compile_trajectory_for(
    variant: Variant,
    req: &GenerationRequest,
) -> gen_core::Result<gen_core::RenderPlan> {
    validate_request_for_variant(variant, req)?;
    let descriptor = descriptor_for(variant);
    let (sigmas, start_step) = resolved_schedule_for_request(variant, req)?;
    let sliced = sigmas[start_step..].to_vec();
    let executed_steps = sliced.len().saturating_sub(1);
    let capture_sigma = *sliced.get(executed_steps).unwrap_or(&0.0);
    let guidance = req.guidance.unwrap_or_else(|| variant.default_cfg());
    let sampler = req.sampler.as_deref().unwrap_or("euler");
    gen_core::RenderPlan::packed_image_trajectory(
        &descriptor,
        req,
        gen_core::render::PackedImagePass {
            pixel_stride: SIZE_MULTIPLE,
            token_width: 64,
            sampler,
            guidance,
            sigmas: sliced,
            executed_steps,
            capture_sigma,
        },
    )
}

fn compile_variant_plan(
    variant: Variant,
    spec: &LoadSpec,
    req: &GenerationRequest,
) -> gen_core::Result<Option<gen_core::RenderPlan>> {
    let id = match variant {
        Variant::Large => MODEL_ID,
        Variant::LargeTurbo => MODEL_ID_TURBO,
        Variant::Medium => MODEL_ID_MEDIUM,
    };
    if matches!(spec.weights, WeightsSource::File(_)) {
        return Err(gen_core::Error::Msg(format!(
            "{id} expects a snapshot directory (text_encoder/ text_encoder_2/ text_encoder_3/ \
             transformer/ vae/), not a single .safetensors file"
        )));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support control / IP-adapter overlays"
        )));
    }
    compile_trajectory_for(variant, req)
        .map(|plan| plan.bind_load(spec))
        .map(Some)
}

/// SD3.5 **Large**'s identity + the wired surface: txt2img with CFG + negative prompt (SD3.5 Large is
/// NOT guidance-distilled — it uses classifier-free guidance, unlike Turbo), img2img reference-guided
/// latent-init (sc-11784), Q4/Q8 quant (sc-7879), and LoRA/LoKr adapter merge (sc-7881). Control /
/// IP-adapter overlays stay off until their own story.
pub fn descriptor() -> ModelDescriptor {
    descriptor_for(Variant::Large)
}

/// SD3.5 **Large Turbo**'s identity + wired surface: guidance-distilled txt2img (no CFG, no negative
/// prompt) — the 4-step sibling. Same architecture/encoders as Large; only the schedule + CFG-off
/// differ.
pub fn descriptor_turbo() -> ModelDescriptor {
    descriptor_for(Variant::LargeTurbo)
}

/// SD3.5 **Medium**'s identity + wired surface: the MMDiT-X (dual-attention) txt2img model with CFG +
/// negative prompt (like Large — NOT distilled). Same triple-TE encoders / 16-ch VAE / flow-match
/// pipeline; only the transformer (24×1536 MMDiT-X with 13 dual blocks) differs.
pub fn descriptor_medium() -> ModelDescriptor {
    descriptor_for(Variant::Medium)
}

/// Build the descriptor for `variant`. Large advertises CFG + negative prompt; Turbo (distilled)
/// advertises neither — so the shared `validate_request` rejects guidance/negative on Turbo (the
/// distilled-model honesty the Z-Image provider uses), keeping the worker from promising a path
/// `generate` ignores.
/// Primary typed route declaration for a Candle SD3.5 variant (`REG-1`, `AP5`).
pub(crate) fn route_for(variant: Variant) -> gen_core::RouteRecord {
    let (id, cfg, dit_id, bindings) = match variant {
        Variant::Large => (
            MODEL_ID,
            true,
            "sd3_5_large_mmdit",
            gen_core::RouteBindings::generator(descriptor, load)
                .with_render_plan(|spec, req| compile_variant_plan(Variant::Large, spec, req))
                .with_memory_registration(LARGE_MEMORY_REGISTRATION)
                .with_memory_fixture(
                    gen_core::candle_memory_contract_surface_specs,
                    large_weights_free_contract,
                )
                .with_memory_surface_resolver(large_surface_contract)
                .with_memory_behavior_registration(LARGE_MEMORY_BEHAVIOR),
        ),
        Variant::LargeTurbo => (
            MODEL_ID_TURBO,
            true,
            "sd3_5_large_turbo_mmdit",
            gen_core::RouteBindings::generator(descriptor_turbo, load_turbo)
                .with_render_plan(|spec, req| compile_variant_plan(Variant::LargeTurbo, spec, req))
                .with_memory_registration(TURBO_MEMORY_REGISTRATION)
                .with_memory_fixture(
                    gen_core::candle_memory_contract_surface_specs,
                    turbo_weights_free_contract,
                )
                .with_memory_surface_resolver(turbo_surface_contract)
                .with_memory_behavior_registration(TURBO_MEMORY_BEHAVIOR),
        ),
        Variant::Medium => (
            MODEL_ID_MEDIUM,
            true,
            "sd3_5_medium_mmdit_x",
            gen_core::RouteBindings::generator(descriptor_medium, load_medium)
                .with_render_plan(|spec, req| compile_variant_plan(Variant::Medium, spec, req))
                .with_memory_registration(MEDIUM_MEMORY_REGISTRATION)
                .with_memory_fixture(
                    gen_core::candle_memory_contract_surface_specs,
                    medium_weights_free_contract,
                )
                .with_memory_surface_resolver(medium_surface_contract)
                .with_memory_behavior_registration(MEDIUM_MEMORY_BEHAVIOR),
        ),
    };
    gen_core::RouteRecord {
        route_id: id,
        family: "stable-diffusion-3",
        backend: "candle",
        modality: Modality::Image,
        mac_only: false,
        structure: gen_core::RouteStructure {
            text_encoders: gen_core::TextEncoderSurface {
                encoders: vec![
                    gen_core::TextEncoderDeclaration::new("clip_l", Some(77)),
                    gen_core::TextEncoderDeclaration::new("clip_bigg", Some(77)),
                    gen_core::TextEncoderDeclaration::new("t5_xxl", Some(256)),
                ],
                ..Default::default()
            },
            conditioning: gen_core::ConditioningSurface {
                kinds: vec![ConditioningKind::Reference],
                pass0_requirement: gen_core::Pass0ConditioningRequirement::NoiseStartSupported,
                supports_negative_prompt: cfg,
                ..Default::default()
            },
            sampling: gen_core::SamplingSurface {
                samplers: candle_gen::curated_sampler_names(),
                schedulers: candle_gen::curated_scheduler_names(),
                supports_guidance: cfg,
                requires_sigma_shift: false,
                ..Default::default()
            },
            denoiser: gen_core::ModelBinding::family_model(
                gen_core::FamilyModelSpec::new(dit_id).with_low_res_start(),
            ),
            latent_codec: gen_core::LatentCodecDeclaration {
                codec_id: Some("sd3_vae"),
                output_latent_space: Some(&candle_gen::gen_core::SD3_LATENT_SPACE),
                pixel_stride: Some(SIZE_MULTIPLE),
                token_width: Some(64),
                ..Default::default()
            },
            adapters: gen_core::AdapterSurface::lora_and_lokr(
                gen_core::AdapterInstallPolicy::FusedMerge,
            ),
            residency: gen_core::ResidencySurface {
                lifetime_graph: vec![
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::PromptEncode,
                        ["clip_l", "clip_bigg", "t5_xxl"],
                        false,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Denoise,
                        [dit_id],
                        false,
                    ),
                    gen_core::ResidencyPhaseNode::new(
                        gen_core::ResidencyPhaseKind::Decode,
                        ["sd3_vae"],
                        false,
                    ),
                ],
                ..Default::default()
            },
            weight_plan: gen_core::WeightPlanDeclaration {
                source_kind: gen_core::WeightSourceKind::SnapshotDirectory,
                supported_quants: &[Quant::Q4, Quant::Q8],
                ..Default::default()
            },
            output: gen_core::OutputSurface {
                preview: gen_core::PreviewSupport::LatentProjection,
                supports_hdr: false,
                supports_alpha_output: false,
            },
            ..Default::default()
        },
        limits: gen_core::ProductLimits {
            min_size: 256,
            max_size: 2048,
            size_floor: gen_core::SizeFloor::RangeChecked,
            supported_steps: gen_core::StepSupport::Unconstrained,
            max_audio_duration_secs: None,
        },
        seed_and_batch: gen_core::SeedAndBatchSemantics {
            seed_scheme: gen_core::SeedScheme::DeterministicOffset,
            max_count: 8,
            supports_prompt_batch: false,
        },
        defaults: gen_core::RouteDefaults {
            steps: Some(variant.default_steps() as u32),
            guidance: Some(variant.default_cfg()),
            sampler: Some("euler"),
            shift: Some(variant.shift()),
            dimensions: Some((1024, 1024)),
            ..Default::default()
        },
        bindings,
    }
}

/// SD3.5 descriptor for `variant` — computed from `route_for`.
pub fn descriptor_for(variant: Variant) -> ModelDescriptor {
    route_for(variant).compute_descriptor()
}

/// Construct the candle SD3.5 **Large** generator from a [`LoadSpec`]. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a `stabilityai/stable-diffusion-3.5-large`-layout diffusers
/// snapshot (`text_encoder/`, `text_encoder_2/`, `text_encoder_3/`, `transformer/`, `vae/`, plus the
/// tokenizers). `spec.quantize` (Q4/Q8) folds the MMDiT at load (sc-7879); `spec.adapters` (LoRA/LoKr)
/// merge into the MMDiT at load (sc-7881). Control / IP-adapter overlays are still refused.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Large)
}

/// Construct the candle SD3.5 **Large Turbo** generator (the guidance-distilled 4-step sibling) from a
/// `stabilityai/stable-diffusion-3.5-large-turbo`-layout snapshot.
pub fn load_turbo(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::LargeTurbo)
}

/// Construct the candle SD3.5 **Medium** generator (the MMDiT-X dual-attention model) from a
/// `stabilityai/stable-diffusion-3.5-medium`-layout diffusers snapshot.
pub fn load_medium(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_variant(spec, Variant::Medium)
}

/// Shared loader for both variants. Lazy — no file I/O until the first `generate`.
fn load_variant(spec: &LoadSpec, variant: Variant) -> gen_core::Result<Box<dyn Generator>> {
    let id = match variant {
        Variant::Large => MODEL_ID,
        Variant::LargeTurbo => MODEL_ID_TURBO,
        Variant::Medium => MODEL_ID_MEDIUM,
    };
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{id} expects a snapshot directory (text_encoder/ text_encoder_2/ text_encoder_3/ \
                 transformer/ vae/), not a single .safetensors file"
            )));
        }
    };
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {id} does not support control / IP-adapter overlays"
        )));
    }
    // sc-7881: LoRA/LoKr adapters are merged into the MMDiT at load (kohya `lora_sd3` fused-QKV +
    // diffusers-named); they are validated/folded lazily on the first `generate`. No refusal here.
    // SD3.5 is a bf16 model; the device is the backend selected at compile time.
    let device = candle_gen::default_device()?;
    let route = match variant {
        Variant::Large => memory_strategy::Sd35Route::Large,
        Variant::LargeTurbo => memory_strategy::Sd35Route::LargeTurbo,
        Variant::Medium => memory_strategy::Sd35Route::Medium,
    };
    let (load_receipt, memory_contract, quant) =
        match memory_strategy::Sd35LoadReceipt::capture(route, spec) {
            Ok(receipt) => {
                let quant = receipt.tier;
                let contract = memory_strategy::contract_from_receipt(spec, &receipt);
                (Some(receipt), contract, quant)
            }
            #[cfg(test)]
            Err(_) if !root.exists() => {
                let mut fixture_spec = spec.clone();
                fixture_spec.resolved_route = Some(id.to_owned());
                let contract = memory_strategy::weights_free_contract(id, &fixture_spec)?;
                (None, contract, spec.quantize)
            }
            Err(error) => return Err(error),
        };
    Ok(Box::new(Sd3Generator {
        descriptor: descriptor_for(variant),
        root,
        device,
        dtype: DType::BF16,
        variant,
        quant,
        adapters: spec.adapters.clone(),
        components: Mutex::new(None),
        vae_encoder: Mutex::new(None),
        lifecycle: Mutex::new(()),
        loaded_spec: spec.clone(),
        memory_contract,
        load_receipt,
        memory_admission: memory_strategy::AdmissionRegistry::new(id),
    }))
}

/// Add all Candle SD3 providers to an explicit media registry builder via one-call
/// [`ProviderRegistryBuilder::register_route`](candle_gen::gen_core::ProviderRegistryBuilder::register_route)
/// (`SW-6979`, `REG-1`).
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    registry
        .register_route(route_for(Variant::Large))
        .register_route(route_for(Variant::LargeTurbo))
        .register_route(route_for(Variant::Medium))
        .register_trainer(training::LARGE_TRAINER_REGISTRATION)
        .register_trainer(training::MEDIUM_TRAINER_REGISTRATION)
}

macro_rules! memory_surface {
    ($contract_fn:ident, $fixture_fn:ident, $surface_fn:ident, $registration:ident, $id:expr) => {
        fn $contract_fn(spec: &LoadSpec) -> gen_core::Result<gen_core::MemoryProviderContract> {
            memory_strategy::provider_contract($id, spec)
        }
        fn $fixture_fn(spec: &LoadSpec) -> gen_core::Result<gen_core::MemoryProviderContract> {
            memory_strategy::weights_free_contract($id, spec)
        }
        fn $surface_fn(
            surface: &gen_core::MemoryContractSurfaceSpec,
        ) -> gen_core::Result<gen_core::MemoryProviderContract> {
            memory_strategy::weights_free_surface_contract($id, surface)
        }
        const $registration: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
            provider_id: $id,
            contract: $contract_fn,
            safety_check: memory_strategy::registered_safety_check,
        };
    };
}

memory_surface!(
    large_memory_contract,
    large_weights_free_contract,
    large_surface_contract,
    LARGE_MEMORY_REGISTRATION,
    MODEL_ID
);
memory_surface!(
    turbo_memory_contract,
    turbo_weights_free_contract,
    turbo_surface_contract,
    TURBO_MEMORY_REGISTRATION,
    MODEL_ID_TURBO
);
memory_surface!(
    medium_memory_contract,
    medium_weights_free_contract,
    medium_surface_contract,
    MEDIUM_MEMORY_REGISTRATION,
    MODEL_ID_MEDIUM
);

macro_rules! memory_behavior {
    ($name:ident, $id:expr) => {
        const $name: gen_core::MemoryBehaviorRegistration = gen_core::MemoryBehaviorRegistration {
            provider_id: $id,
            valid_fixtures: memory_strategy::registered_valid_fixture,
            begin_request: |spec, contract, context| {
                memory_strategy::registered_begin_request($id, spec, contract, context)
            },
        };
    };
}

memory_behavior!(LARGE_MEMORY_BEHAVIOR, MODEL_ID);
memory_behavior!(TURBO_MEMORY_BEHAVIOR, MODEL_ID_TURBO);
memory_behavior!(MEDIUM_MEMORY_BEHAVIOR, MODEL_ID_MEDIUM);

/// Register exhaustive weights-free q4/q8/bf16 contract surfaces on every platform.
pub fn register_memory_contract_surfaces(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(LARGE_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            provider_id: MODEL_ID,
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            contract: large_weights_free_contract,
        })
        .register_memory_contract_surface_resolver(
            gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: MODEL_ID,
                contract: large_surface_contract,
            },
        )
        .register_memory_strategy(TURBO_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            provider_id: MODEL_ID_TURBO,
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            contract: turbo_weights_free_contract,
        })
        .register_memory_contract_surface_resolver(
            gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: MODEL_ID_TURBO,
                contract: turbo_surface_contract,
            },
        )
        .register_memory_strategy(MEDIUM_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            provider_id: MODEL_ID_MEDIUM,
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            contract: medium_weights_free_contract,
        })
        .register_memory_contract_surface_resolver(
            gen_core::MemoryContractSurfaceResolverRegistration {
                provider_id: MODEL_ID_MEDIUM,
                contract: medium_surface_contract,
            },
        )
}

/// Build the complete explicit Candle SD3 provider catalog.
pub fn provider_registry() -> candle_gen::gen_core::Result<candle_gen::gen_core::ProviderRegistry> {
    register_providers(candle_gen::gen_core::ProviderRegistryBuilder::new()).build()
}

#[cfg(test)]
mod explicit_registry_tests {
    #[test]
    fn explicit_catalog_has_stable_surface() {
        let registry = super::provider_registry().unwrap();
        let explicit: Vec<String> = registry
            .generators()
            .map(|registration| (registration.descriptor)().id.to_string())
            .collect();

        assert_eq!(
            explicit,
            ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"]
        );
    }

    /// The registry-level memory lifecycle seams must be reachable on a build with no CUDA
    /// feature: building the provider catalog is contract-only (no device, no weights), so
    /// `register_providers` publishes the memory-strategy, weights-free contract-fixture and
    /// memory-behavior rows on every platform. Gating these behind `cuda` left registry
    /// lifecycle conformance running on no CPU CI configuration at all.
    #[test]
    fn register_providers_publishes_memory_lifecycle_seams_without_cuda() {
        let registry = super::provider_registry().unwrap();

        let strategies: Vec<&str> = registry
            .memory_strategy_registrations()
            .map(|registration| registration.provider_id)
            .collect();
        let fixtures: Vec<&str> = registry
            .memory_contract_fixture_registrations()
            .map(|registration| registration.provider_id)
            .collect();
        let behaviors: Vec<&str> = registry
            .memory_behavior_registrations()
            .map(|registration| registration.provider_id)
            .collect();

        assert_eq!(
            strategies,
            ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"]
        );
        assert_eq!(
            fixtures,
            ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"]
        );
        assert_eq!(
            behaviors,
            ["sd3_5_large", "sd3_5_large_turbo", "sd3_5_medium"]
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{Conditioning, ConditioningKind, Image, LoadSpec, WeightsSource};

    #[test]
    fn sd3_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .expect("candle sd3 is registered");
        assert_eq!(g.descriptor().id, MODEL_ID);
        assert_eq!(g.descriptor().family, "stable-diffusion-3");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn descriptor_advertises_cfg_txt2img_surface() {
        let d = descriptor();
        // SD3.5 Large uses classifier-free guidance with a negative prompt (NOT distilled).
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.mac_only);
        // sc-11784: img2img reference-guided latent-init is now advertised (single-ref, NOT MultiReference).
        assert_eq!(
            d.capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        // sc-7881: LoRA/LoKr are now wired (merged into the MMDiT at load).
        assert!(d.capabilities.supports_lora);
        assert!(d.capabilities.supports_lokr);
        // sc-7879: the MMDiT folds to Q4/Q8 at load (TE+VAE stay dense).
        assert_eq!(
            d.capabilities.supported_quants,
            &[
                candle_gen::gen_core::Quant::Q4,
                candle_gen::gen_core::Quant::Q8
            ]
        );
        assert_eq!(d.capabilities.min_size, 256);
        assert_eq!(d.capabilities.max_size, 2048);
        assert_eq!(d.capabilities.samplers, candle_gen::curated_sampler_names());
        // sc-11784: the img2img `Reference` conditioning is now accepted.
        assert!(descriptor()
            .capabilities
            .accepts(ConditioningKind::Reference));
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_unsupported() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();

        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());

        // sc-11784: an img2img `Reference` request now validates (the capability floor accepts it).
        let img2img = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: Some(0.6),
            }],
            ..Default::default()
        };
        assert!(g.validate(&img2img).is_ok());

        for bad in [
            GenerationRequest::default(), // empty prompt
            GenerationRequest {
                prompt: "x".into(),
                width: 1000, // not multiple of 16
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                steps: Some(0),
                ..Default::default()
            },
        ] {
            assert!(g.validate(&bad).is_err(), "should reject: {bad:?}");
        }

        // sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised SD3.5 bucket
        // to. Pin the value and mutation-check that a size which is a multiple of 8 (the VAE scale) but
        // not SIZE_MULTIPLE (16) is still rejected with the stride error, and an on-stride size passes.
        assert_eq!(SIZE_MULTIPLE, 16);
        let off_stride = g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1000, // 125×8 — a multiple of 8 but not SIZE_MULTIPLE
                ..Default::default()
            })
            .unwrap_err()
            .to_string();
        assert!(
            off_stride.contains("multiples of 16"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(g
            .validate(&GenerationRequest {
                prompt: "x".into(),
                width: 1024, // 64×16 — on-stride
                ..Default::default()
            })
            .is_ok());
    }

    /// C2 wires `generate`, so it now passes validation and proceeds to component load. Against a
    /// nonexistent snapshot it fails at load with a *non-`Unsupported`* error (the dir is missing) —
    /// i.e. the pipeline path is reached, not the old typed-`Unsupported` stub.
    #[test]
    fn generate_is_wired_and_loads_components() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .unwrap();
        let req = GenerationRequest {
            prompt: "x".into(),
            ..Default::default()
        };
        let mut progress = |_p: Progress| {};
        let err = g
            .generate(&req, &mut progress)
            .expect_err("missing snapshot dir must error at load");
        // The pipeline IS wired now — the error is the missing component dir, not the C1 stub.
        assert!(
            !matches!(err, gen_core::Error::Unsupported(_)),
            "generate should reach the pipeline (load error), not the un-wired stub: {err:?}"
        );
    }

    /// Both the Large and Turbo ids register and resolve as candle generators with distinct surfaces.
    #[test]
    fn turbo_registers_and_advertises_distilled_surface() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID_TURBO, &spec)
            .expect("candle sd3 turbo is registered");
        assert_eq!(g.descriptor().id, MODEL_ID_TURBO);
        assert_eq!(g.descriptor().family, "stable-diffusion-3");
        assert_eq!(g.descriptor().backend, "candle");

        let d = descriptor_turbo();
        // Turbo defaults to CFG off but supports experimental negative conditioning.
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.mac_only);

        // Explicit guidance enables negative conditioning without changing the native default.
        for bad in [
            GenerationRequest {
                prompt: "x".into(),
                guidance: Some(5.0),
                ..Default::default()
            },
            GenerationRequest {
                prompt: "x".into(),
                negative_prompt: Some("blurry".into()),
                guidance: Some(1.5),
                ..Default::default()
            },
        ] {
            assert!(
                g.validate(&bad).is_ok(),
                "turbo supports CFG overrides: {bad:?}"
            );
        }
        for guidance in [None, Some(1.0)] {
            assert!(g
                .validate(&GenerationRequest {
                    prompt: "x".into(),
                    negative_prompt: Some("blurry".into()),
                    guidance,
                    ..Default::default()
                })
                .is_err());
        }
    }

    /// SD3.5 **Medium** registers + resolves as a candle generator and advertises the CFG txt2img
    /// surface (like Large, NOT distilled).
    #[test]
    fn medium_registers_and_advertises_cfg_surface() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID_MEDIUM, &spec)
            .expect("candle sd3 medium is registered");
        assert_eq!(g.descriptor().id, MODEL_ID_MEDIUM);
        assert_eq!(g.descriptor().family, "stable-diffusion-3");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);

        let d = descriptor_medium();
        // Medium uses classifier-free guidance + a negative prompt (it is NOT guidance-distilled).
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.mac_only);

        // A txt2img request validates; sc-11784 also accepts an img2img `Reference` (single init image).
        let ok = GenerationRequest {
            prompt: "a rusty robot holding a lit candle".into(),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
    }

    #[test]
    fn load_accepts_quant_and_adapter_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec, Quant};
        // `/snap` exists on Ubuntu hosts. This test exercises lazy load-shape admission, not a
        // physical checkpoint receipt, so use a process-unique guaranteed-missing root.
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("missing-sd3-snapshot");
        // sc-7879: Q4/Q8 quant is ACCEPTED (the MMDiT folds at load); loading is lazy so this builds
        // the generator without touching the (nonexistent) snapshot.
        let quant = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q8);
        assert!(
            load(&quant).is_ok(),
            "Q8 quant load must be accepted (sc-7879)"
        );
        let quant4 = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q4);
        assert!(
            load(&quant4).is_ok(),
            "Q4 quant load must be accepted (sc-7879)"
        );

        // sc-7881: LoRA/LoKr adapters are now ACCEPTED (merged into the MMDiT at load, lazily). The
        // spec is stashed; the actual merge happens on the first `generate` against a real snapshot.
        for kind in [AdapterKind::Lora, AdapterKind::Lokr] {
            let adapter = LoadSpec::new(WeightsSource::Dir(root.clone())).with_adapters(vec![
                AdapterSpec::new("/lora.safetensors".into(), 1.0, kind),
            ]);
            assert!(
                load(&adapter).is_ok(),
                "{kind:?} adapter load must be accepted (sc-7881)"
            );
        }
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/sd3.safetensors".into()));
        let Err(e) = load(&spec) else {
            panic!("single-file source must be refused")
        };
        assert!(e.to_string().contains("snapshot directory"), "got: {e}");
    }

    #[test]
    fn sw6946_candle_sd3_m3_non_square_plain_img2img_low_res_start_fixture() {
        use candle_gen::gen_core::{
            latent::SpatialCompression,
            sampling::{init_time_step, sd3_sigmas, FlowModelSampling, TimestepConvention},
            transition_clean_estimate_f32, BoundaryState, EstimateSource, LatentPatchLayout,
            LowResStartSpec, SD3_LATENT_SPACE,
        };

        let registry = crate::provider_registry().unwrap();
        let all_features = registry.all_route_feature_applicability();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));

        assert_eq!(SD3_LATENT_SPACE.channels, 16);
        assert_eq!(
            SD3_LATENT_SPACE.spatial_compression,
            SpatialCompression {
                height: 8,
                width: 8
            }
        );
        assert_eq!(SD3_LATENT_SPACE.patch_layout, LatentPatchLayout::Unpacked);

        let source_1200x700 = Image {
            width: 1200,
            height: 700,
            pixels: vec![112; 1200 * 700 * 3],
        };
        let steps = 10_usize;
        let strength = 0.4_f32;
        assert_eq!(init_time_step(steps, Some(strength)), 4);
        let full_sigmas = sd3_sigmas(steps, 3.0);
        assert_eq!(full_sigmas.len(), 11);

        for provider_id in [MODEL_ID, MODEL_ID_TURBO, MODEL_ID_MEDIUM] {
            let app = &all_features[provider_id]["low_res_start"];
            assert!(
                app.applicable,
                "candle {provider_id} must support low_res_start"
            );

            for via_source_image in [false, true] {
                let mut m3_req = GenerationRequest {
                    prompt: "a vintage locomotive crossing a stone viaduct".into(),
                    width: 1024,
                    height: 768,
                    steps: Some(steps as u32),
                    seed: Some(1234),
                    features: candle_gen::gen_core::FeatureParams {
                        low_res_start: Some(LowResStartSpec::new(3, Some(0.5))),
                        ..Default::default()
                    },
                    ..Default::default()
                };
                if via_source_image {
                    m3_req.source_image = Some(source_1200x700.clone());
                    m3_req.strength = Some(strength);
                } else {
                    m3_req.conditioning = vec![Conditioning::Reference {
                        image: source_1200x700.clone(),
                        strength: Some(strength),
                    }];
                }

                let plan = registry
                    .compile_render_plan(provider_id, &spec, &m3_req)
                    .unwrap_or_else(|e| panic!("{provider_id} (source={via_source_image}): {e}"))
                    .expect("compiled Candle SD3.5 RenderPlan");

                assert_eq!(plan.executed_steps(), 6, "{provider_id}");
                assert_eq!(plan.denoise_spans(), vec![0..3, 3..6], "{provider_id}");
                assert_eq!(plan.phase_output_size(0), (512, 384), "{provider_id}");
                assert_eq!(plan.phase_output_size(1), (1024, 768), "{provider_id}");
                assert_eq!(plan.phase_packed_tokens(0), 32 * 24, "{provider_id}");
                assert_eq!(plan.phase_packed_tokens(1), 64 * 48, "{provider_id}");

                let denoise_sigmas = plan.denoise_sigmas();
                assert_eq!(denoise_sigmas.len(), 7, "{provider_id}");
                assert_eq!(denoise_sigmas, &full_sigmas[4..=10], "{provider_id}");
                assert_eq!(denoise_sigmas[0], full_sigmas[4], "{provider_id}");
                assert_eq!(denoise_sigmas[3], full_sigmas[7], "{provider_id}");

                let low_res_plan = plan.low_res_start().expect("low_res_start active");
                let phase0_clean = vec![0.5_f32; 16 * 48 * 64];
                let transitioned = transition_clean_estimate_f32(
                    BoundaryState::CleanEstimate {
                        latent: &phase0_clean,
                        source: EstimateSource::PreviousEvaluation,
                    },
                    1,
                    16,
                    8,
                    low_res_plan,
                    full_sigmas[7],
                    &FlowModelSampling::new(TimestepConvention::Sigma),
                    1234,
                    0,
                    0,
                    0,
                )
                .expect("Candle SD3.5 phase-0 -> phase-1 clean estimate transition at node 7");
                assert_eq!(transitioned.len(), 16 * 96 * 128);
            }
        }
    }

    #[test]
    fn strength_one_img2img_compiles_zero_step_endpoint_plan_without_weights() {
        let registry = crate::provider_registry().unwrap();
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let ref_img = Image {
            width: 256,
            height: 256,
            pixels: vec![128; 256 * 256 * 3],
        };
        for (variant, provider_id) in [
            (Variant::Large, MODEL_ID),
            (Variant::LargeTurbo, MODEL_ID_TURBO),
            (Variant::Medium, MODEL_ID_MEDIUM),
        ] {
            for steps in [1u32, 4u32] {
                let req = GenerationRequest {
                    prompt: "sd3 strength 1.0 zero-step endpoint".into(),
                    width: 256,
                    height: 256,
                    steps: Some(steps),
                    seed: Some(42),
                    conditioning: vec![Conditioning::Reference {
                        image: ref_img.clone(),
                        strength: Some(1.0),
                    }],
                    ..Default::default()
                };
                let (sigmas, start_step) = resolved_schedule_for_request(variant, &req).unwrap();
                assert_eq!(start_step, steps as usize, "{provider_id} steps={steps}");
                assert_eq!(&sigmas[start_step..], &[0.0], "{provider_id} steps={steps}");

                let plan = registry
                    .compile_render_plan(provider_id, &spec, &req)
                    .unwrap()
                    .expect("compiled plan for strength 1.0 img2img");
                assert_eq!(plan.executed_steps(), 0, "{provider_id} steps={steps}");
                assert_eq!(plan.denoise_sigmas(), &[0.0], "{provider_id} steps={steps}");
                assert_eq!(
                    plan.denoise_spans(),
                    vec![0..0],
                    "{provider_id} steps={steps}"
                );
            }
        }
    }
}
