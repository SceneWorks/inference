//! # candle-gen-boogu
//!
//! The **Boogu-Image-0.1** provider crate for [`candle-gen`](candle_gen) — the candle (Windows/CUDA)
//! sibling of `mlx-gen-boogu`. Registers three engine ids:
//!
//! * **`boogu_image`** — the Base variant: a 10.3B Lumina-Image-2.0 / OmniGen2-lineage hybrid MMDiT
//!   (8 double + 32 single + 2 refiner layers, GQA, 3-axis interleaved RoPE) with true-CFG, driven by
//!   a Qwen3-VL-8B condition encoder and a FLUX.1 16-channel VAE. 50-step rectified-flow Euler over a
//!   static-shift (`mu = 1.15`) schedule, routed through the unified curated-sampler framework.
//! * **`boogu_image_turbo`** — the same Base weights-arch + a DMD-distilled few-step (4) sampler loop,
//!   CFG-free (guidance inert). The default fast surface.
//! * **`boogu_image_edit`** — text+image-to-image with one or more reference images (sc-7523 single,
//!   sc-7645 multi up to 5): each source ([`ConditioningKind::Reference`] /
//!   [`ConditioningKind::MultiReference`]) is VAE-encoded into the DiT's spatial reference sequence
//!   (`forward_edit`) **and** read by the Qwen3-VL **vision tower** so the MLLM "sees" it
//!   (image-conditioned instruction features). Same true-CFG / static-shift schedule as Base.
//!
//! **Reuse:** the FLUX.1 VAE is `candle-transformers`' `z_image::vae::AutoEncoderKL` (the exact 16-ch
//! AutoencoderKL Z-Image ships, scaling 0.3611 / shift 0.1159) — reused verbatim, as `mlx-gen-boogu`
//! reuses `mlx-gen-z-image`'s `Vae`. The Qwen3-VL-8B condition encoder, its vision tower, and the
//! hybrid DiT are ported here.
//!
//! `backend = "candle"`, `mac_only = false`. Apache-2.0, ungated.

pub mod config;
pub mod loader;
pub mod memory_strategy;
pub mod pipeline;
pub mod preview;
pub mod quant;
pub mod text_encoder;
pub mod tokenizer;
pub mod transformer;
pub mod vision;

// Boogu Base + Turbo img2img / `Reference` real-weight GPU validation (sc-11786) — env-driven,
// `#[ignore]`d integration tests driving the REGISTERED Base/Turbo generators through a
// `Conditioning::Reference` (strength ablation + monotone reference fidelity + prompt divergence),
// the candle mirror of `mlx-gen-boogu`'s sc-10191 validation.
#[cfg(test)]
mod img2img_validate;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::{
    self, Capabilities, Conditioning, ConditioningKind, GenerationOutput, GenerationRequest,
    Generator, Image, LoadSpec, Modality, ModelDescriptor, PidWeights, Progress, Quant,
    WeightsSource,
};
use candle_transformers::models::z_image::vae::Encoder;

use pipeline::{Components, EditComponents};

/// Registry id for the Base text-to-image variant (true-CFG).
pub const BOOGU_IMAGE_ID: &str = "boogu_image";
/// Registry id for the Turbo variant (DMD few-step, CFG-free).
pub const BOOGU_IMAGE_TURBO_ID: &str = "boogu_image_turbo";
/// Registry id for the instruction image-edit variant (single- or multi-reference TI2I).
pub const BOOGU_IMAGE_EDIT_ID: &str = "boogu_image_edit";

/// Patch(2)·ae_scale(8) = 16 — `patchify` requires latent dims divisible by this. Exposed as the
/// pinned-engine stride SceneWorks ties each advertised Boogu image bucket to (sc-12612), mirroring
/// `wan::config::SIZE_MULTIPLE_14B`. `validate` enforces exactly this value, so the const cannot
/// drift from the check.
pub const SIZE_MULTIPLE: u32 = 16;

/// Maximum reference images the Edit lane accepts — the DiT's `image_index_embedding` row count (the
/// OmniGen2-lineage `[5, hidden]` parameter supports up to 5 distinct reference index slots).
const MAX_EDIT_REFERENCES: usize = 5;

/// The curated samplers the Turbo DMD student stays coherent under (the stochastic / re-noising
/// solvers — `lcm` most of all; real-weight survey sc-7491). The student was distilled against a
/// stochastic (re-noised) trajectory, so the curated stochastic solvers match its training regime;
/// the deterministic ODE solvers feed the few-step student out-of-regime latents, so they stay off
/// the menu. A selected name routes `render_turbo` through the unified `run_flow_sampler` over the
/// DMD σ grid (sc-9009); unset stays the byte-exact native DMD loop. Mirrors `mlx-gen-boogu`'s
/// `TURBO_SAMPLERS`.
const TURBO_SAMPLERS: &[&str] = &["lcm", "euler_ancestral", "dpmpp_sde"];

/// Which Boogu sampler path a generator drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Variant {
    /// Base — true-CFG text-to-image.
    Base,
    /// Turbo — CFG-free DMD few-step text-to-image.
    Turbo,
    /// Edit — TI2I (true-CFG) with one or more reference images VAE-encoded + vision-conditioned.
    Edit,
}

/// A lazily-loaded Boogu generator. `Variant` selects the sampler path. The shared T2I components
/// load on the first `generate`; the Edit-only components (vision tower + VAE encoder) load lazily on
/// the first edit, so the T2I paths keep their footprint.
pub struct BooguGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    variant: Variant,
    device: Device,
    /// The `LoadSpec::pid` component captured at load (epic 7840 / sc-7853), threaded into the lazy
    /// component build so the PiD engine loads once alongside the base model. `None` when not opted in.
    pid_spec: Option<PidWeights>,
    components: Mutex<Option<Arc<Components>>>,
    edit_components: Mutex<Option<Arc<EditComponents>>>,
    /// Lazily-built, cached f32 VAE **encoder** for the Base/Turbo img2img latent-init path (sc-11786).
    /// Built on the **first img2img request only** — a pure txt2img (or Edit) workload never populates
    /// it. Distinct from [`EditComponents`]'s encoder so a plain img2img never loads the Edit vision
    /// tower. Accel-independent (no attention-dispatch toggle), so one cached instance serves every
    /// request. Mirrors Z-Image's `vae_encoder` cache.
    img2img_encoder: Mutex<Option<Arc<Encoder>>>,
    memory_strategy: Option<memory_strategy::PreparedMemory>,
    memory_admission: memory_strategy::AdmissionRegistry,
    /// Serializes warm and staged ownership. A staged request evicts all warm provider caches before
    /// loading its text phase and cannot race a resident request that would repopulate them.
    request_lock: Mutex<()>,
}

impl BooguGenerator {
    fn staged_boundary(
        &self,
        req: &GenerationRequest,
        phase: gen_core::MemoryPhase,
    ) -> gen_core::Result<()> {
        candle_gen::check_cancel(&req.cancel)?;
        if req
            .memory
            .is_some_and(|memory| memory.calibration_error_phase == Some(phase))
        {
            return Err(gen_core::Error::Msg(format!(
                "{}: injected memory-strategy calibration error at {phase:?}",
                self.descriptor.id
            )));
        }
        Ok(())
    }

    fn components(&self) -> gen_core::Result<Arc<Components>> {
        candle_gen::cached(&self.components, || {
            Ok(Arc::new(pipeline::load_components(
                &self.root,
                &self.device,
                self.pid_spec.as_ref(),
            )?))
        })
    }

    fn edit_components(&self) -> gen_core::Result<Arc<EditComponents>> {
        candle_gen::cached(&self.edit_components, || {
            Ok(Arc::new(pipeline::load_edit_components(
                &self.root,
                &self.device,
            )?))
        })
    }

    /// The cached f32 VAE encoder for the img2img latent-init path (sc-11786), built on a miss. Only
    /// ever called when a Base/Turbo request carries a `Reference` at a strength yielding a non-empty
    /// denoise (`start_step > 0`), so a txt2img-only workload never builds it.
    fn img2img_encoder(&self) -> gen_core::Result<Arc<Encoder>> {
        candle_gen::cached(&self.img2img_encoder, || {
            Ok(Arc::new(pipeline::load_vae_encoder(
                &self.root,
                &self.device,
            )?))
        })
    }

    /// Resolve the Base/Turbo img2img latent-init (sc-11786): the single [`Conditioning::Reference`] +
    /// its strength-derived `start_step` (via [`pipeline::init_time_step`] over `default_steps`, which
    /// differs Base vs Turbo), VAE-encoding the reference to the clean init latent only when the strength
    /// yields a non-empty structure-preserving denoise (`start_step > 0`). Returns `(None, 0)` for a
    /// pure txt2img request — the render paths then stay byte-identical to the pre-sc-11786 path.
    fn img2img_init(
        &self,
        req: &GenerationRequest,
        default_steps: usize,
    ) -> gen_core::Result<(Option<Tensor>, usize)> {
        let reference = pipeline::resolve_reference(req, self.descriptor.id)?;
        let steps = req.steps.map(|s| s as usize).unwrap_or(default_steps);
        let start_step = reference
            .map(|(_, strength)| pipeline::init_time_step(steps, strength))
            .unwrap_or(0);
        let clean = if start_step > 0 {
            let (image, _) = reference.expect("start_step > 0 implies a reference");
            let encoder = self.img2img_encoder()?;
            Some(pipeline::encode_reference(
                &encoder,
                image,
                req.width,
                req.height,
                &self.device,
            )?)
        } else {
            None
        };
        Ok((clean, start_step))
    }

    fn generate_staged(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<Vec<Image>> {
        // Evict every reloadable warm cache before phase A. The request lock held by `generate`
        // makes this eviction and the subsequent staged body exclusive, so no concurrent resident
        // request can repopulate a cache between phases.
        *self
            .components
            .lock()
            .map_err(|_| gen_core::Error::Msg("boogu: poisoned warm component cache".into()))? =
            None;
        *self
            .edit_components
            .lock()
            .map_err(|_| gen_core::Error::Msg("boogu: poisoned warm edit cache".into()))? = None;
        *self
            .img2img_encoder
            .lock()
            .map_err(|_| gen_core::Error::Msg("boogu: poisoned warm reference cache".into()))? =
            None;

        let result = (|| {
            if req.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            on_progress(Progress::Loading(gen_core::LoadPhase::TextEncoder));
            self.staged_boundary(req, gen_core::MemoryPhase::Conditioning)?;
            let text = pipeline::load_staged_text(&self.root, &self.device)?;
            let encoded = match self.variant {
                Variant::Base | Variant::Turbo => {
                    let default_steps = if self.variant == Variant::Base {
                        pipeline::DEFAULT_STEPS
                    } else {
                        pipeline::DEFAULT_TURBO_STEPS
                    };
                    let active = pipeline::resolve_reference(req, self.descriptor.id)?.is_some_and(
                        |(_, strength)| {
                            pipeline::init_time_step(
                                req.steps.map(|s| s as usize).unwrap_or(default_steps),
                                strength,
                            ) > 0
                        },
                    );
                    let encoder = active
                        .then(|| pipeline::load_vae_encoder(&self.root, &self.device))
                        .transpose()?;
                    if self.variant == Variant::Base {
                        pipeline::stage_encode_base(
                            &text,
                            encoder.as_ref(),
                            req,
                            pipeline::DEFAULT_STEPS,
                            &self.device,
                        )?
                    } else {
                        pipeline::stage_encode_turbo(&text, encoder.as_ref(), req, &self.device)?
                    }
                }
                Variant::Edit => {
                    let references = resolve_edit_references(req)?;
                    let edit = pipeline::load_edit_components(&self.root, &self.device)?;
                    pipeline::stage_encode_edit(&text, &edit, req, &references, &self.device)?
                }
            };
            self.device
                .synchronize()
                .map_err(gen_core::Error::backend)?;
            drop(text);
            if req.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }

            on_progress(Progress::Loading(gen_core::LoadPhase::Renderer));
            self.staged_boundary(req, gen_core::MemoryPhase::Denoise)?;
            let denoise = pipeline::load_staged_denoise(&self.root, &self.device)?;
            let latents = match self.variant {
                Variant::Base => {
                    pipeline::stage_denoise_base(&denoise, req, encoded, &self.device, on_progress)?
                }
                Variant::Turbo => pipeline::stage_denoise_turbo(
                    &denoise,
                    req,
                    encoded,
                    &self.device,
                    on_progress,
                )?,
                Variant::Edit => {
                    pipeline::stage_denoise_edit(&denoise, req, encoded, &self.device, on_progress)?
                }
            };
            self.device
                .synchronize()
                .map_err(gen_core::Error::backend)?;
            drop(denoise);
            if req.cancel.is_cancelled() {
                return Err(gen_core::Error::Canceled);
            }
            self.staged_boundary(req, gen_core::MemoryPhase::Decode)?;
            let decode = pipeline::load_staged_decode(&self.root, &self.device, None)?;
            pipeline::stage_decode(&decode, req, self.descriptor.id, latents, on_progress)
                .map_err(gen_core::Error::backend)
        })();
        let cleanup = self.device.synchronize().map_err(gen_core::Error::backend);
        match (result, cleanup) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(images), Ok(())) => Ok(images),
        }
    }
}

impl Generator for BooguGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        self.memory_strategy.as_ref().map(|memory| &memory.contract)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        let Some(memory) = &self.memory_strategy else {
            return gen_core::MemorySafetyDecision::Reject {
                reason: format!("{} has no exact Boogu artifact receipt", self.descriptor.id),
            };
        };
        if let Err(error) = memory.receipt.ensure_unchanged() {
            self.memory_admission.clear_approval();
            return gen_core::MemorySafetyDecision::Reject {
                reason: error.to_string(),
            };
        }
        match memory_strategy::safety_check(
            self.descriptor.id,
            &memory.contract,
            gen_core::MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: memory.receipt.tier,
                component_precision_floors: &[],
            },
            context,
        ) {
            gen_core::MemorySafetyDecision::Accept => {
                match self.memory_admission.approve(context) {
                    Ok(()) => gen_core::MemorySafetyDecision::Accept,
                    Err(error) => gen_core::MemorySafetyDecision::Reject {
                        reason: error.to_string(),
                    },
                }
            }
            reject => {
                self.memory_admission.clear_approval();
                reject
            }
        }
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        let memory = self.memory_strategy.as_ref().ok_or_else(|| {
            gen_core::Error::Unsupported(format!(
                "{} has no exact Boogu artifact receipt",
                self.descriptor.id
            ))
        })?;
        memory.receipt.ensure_unchanged()?;
        memory_strategy::begin_request(
            self.descriptor.id,
            &memory.contract,
            gen_core::MemoryNumericTier {
                precision: gen_core::Precision::Bf16,
                quant: memory.receipt.tier,
                component_precision_floors: &[],
            },
            self.device.clone(),
            context,
            self.memory_admission.clone(),
        )
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        let id = self.descriptor.id;
        self.descriptor.capabilities.validate_request(id, req)?;
        if req.prompt.trim().is_empty() {
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
        // The Edit variant needs 1..=5 source references; Base/Turbo take at most one (img2img
        // latent-init, sc-11786) — `resolve_reference` fails fast on a second reference. The capability
        // floor already rejects a `MultiReference` on Base/Turbo (only `Reference` is advertised there).
        match self.variant {
            Variant::Edit => {
                resolve_edit_references(req)?;
            }
            Variant::Base | Variant::Turbo => {
                pipeline::resolve_reference(req, self.descriptor.id)?;
            }
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        let _request = self.request_lock.lock().map_err(|_| {
            gen_core::Error::Msg(format!(
                "{}: poisoned request ownership lock",
                self.descriptor.id
            ))
        })?;
        self.validate(req)?;
        self.memory_admission.consume_for_generate(req)?;
        if let Some(memory) = &self.memory_strategy {
            memory.receipt.ensure_unchanged()?;
            if req.memory.is_some() {
                memory_strategy::validate_generation_request(self.descriptor.id, req)?;
            }
            if req.memory.is_some_and(|request| request.stage_residency)
                && !matches!(
                    memory
                        .contract
                        .capability(gen_core::MemoryStrategy::StagedResidency)
                        .map(|capability| &capability.support),
                    Some(gen_core::MemoryStrategySupport::Implemented)
                )
            {
                return Err(gen_core::Error::Unsupported(format!(
                    "{}: staged residency is outside this exact load receipt",
                    self.descriptor.id
                )));
            }
        } else if req.memory.is_some_and(|memory| {
            memory.stage_residency
                || memory.tile_vae_decode
                || memory.chunk_attention
                || memory.stream_transformer_blocks
        }) {
            return Err(gen_core::Error::Unsupported(format!(
                "{}: optimized memory execution requires an exact artifact receipt",
                self.descriptor.id
            )));
        }
        if req.memory.is_some_and(|memory| memory.stage_residency) {
            return Ok(GenerationOutput::Images(
                self.generate_staged(req, on_progress)?,
            ));
        }
        let comps = self.components()?;
        let images = match self.variant {
            Variant::Turbo => {
                // img2img latent-init (sc-11786): a single `Reference` seeds the few-step DMD denoise
                // from the VAE-encoded reference; no reference (or strength→start 0) stays pure txt2img.
                let (clean, start_step) = self.img2img_init(req, pipeline::DEFAULT_TURBO_STEPS)?;
                pipeline::render_turbo(
                    &comps,
                    req,
                    clean.as_ref(),
                    start_step,
                    &self.device,
                    on_progress,
                )?
            }
            Variant::Base => {
                // img2img latent-init (sc-11786): a single `Reference` seeds the true-CFG denoise from
                // the VAE-encoded reference; no reference (or strength→start 0) stays pure txt2img.
                let (clean, start_step) = self.img2img_init(req, pipeline::DEFAULT_STEPS)?;
                pipeline::render_base(
                    &comps,
                    req,
                    clean.as_ref(),
                    start_step,
                    &self.device,
                    on_progress,
                )?
            }
            Variant::Edit => {
                let references = resolve_edit_references(req)?;
                let edit = self.edit_components()?;
                pipeline::render_edit(&comps, &edit, req, &references, &self.device, on_progress)?
            }
        };
        Ok(GenerationOutput::Images(images))
    }
}

/// The img2img/instruction-edit source images, in order — collected from both
/// [`Conditioning::Reference`] (single) and [`Conditioning::MultiReference`] (multi). At least one and
/// at most [`MAX_EDIT_REFERENCES`] (the DiT's `image_index_embedding` row count) is required; zero or
/// more than the cap is an error.
fn resolve_edit_references(req: &GenerationRequest) -> gen_core::Result<Vec<&Image>> {
    let mut refs: Vec<&Image> = Vec::new();
    for c in &req.conditioning {
        match c {
            Conditioning::Reference { image, .. } => refs.push(image),
            Conditioning::MultiReference { images } => refs.extend(images.iter()),
            _ => {} // the capability floor already rejects other conditioning kinds.
        }
    }
    if refs.is_empty() {
        return Err(gen_core::Error::Msg(
            "boogu_image_edit: an instruction edit requires at least one source reference image"
                .into(),
        ));
    }
    if refs.len() > MAX_EDIT_REFERENCES {
        return Err(gen_core::Error::Msg(format!(
            "boogu_image_edit: at most {MAX_EDIT_REFERENCES} reference images are supported (got {})",
            refs.len()
        )));
    }
    Ok(refs)
}

/// Boogu Base descriptor — true-CFG text-to-image; no user negative prompt (the CFG-negative is the
/// model's own fixed empty/drop instruction). A single [`ConditioningKind::Reference`] opts into
/// img2img latent-init (sc-11786): VAE-encode the reference + noise-blend at a strength-derived start
/// step. The instruction-edit `MultiReference` (Qwen3-VL semantic edit) is the Edit checkpoint's alone
/// (`descriptor_edit`); Turbo inherits this img2img surface via `descriptor()`.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&candle_gen::gen_core::FLUX1_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: BOOGU_IMAGE_ID,
        family: "boogu",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_guidance: true,
            // All three registered routes emit one latent preview per outer denoise step. Base and
            // Edit use the shared flow driver; Turbo additionally covers its default native DMD loop.
            supports_preview: true,
            // Base/Turbo are text-to-image, and a single `Reference` opts them into img2img latent-init
            // (sc-11786): VAE-encode the reference + noise-blend at a strength-derived start step. The
            // multi-image instruction-edit path is the Edit checkpoint's (`descriptor_edit`).
            conditioning: vec![ConditioningKind::Reference],
            // Base is rectified-flow Euler over the static-shift schedule, routed through the unified
            // curated-sampler framework (epic 7114).
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            // sc-9607: advertise the packed tiers so the worker's A-B quant toggle engages off-Mac.
            // The resolved `base/`-`-q4/`-`-bf16/` turnkey subdir self-describes its tier
            // (`loader::linear_detect`, sc-9410, group-size-aware); `build` no-ops the requested quant.
            // Turbo + edit inherit this via `descriptor()`.
            supported_quants: &[Quant::Q4, Quant::Q8],
            ..Default::default()
        },
    }
}

/// Boogu Turbo descriptor — same base, CFG-free DMD few-step; guidance is inert. The advertised
/// sampler menu is the DMD-compatible stochastic subset (`TURBO_SAMPLERS`); a selected sampler or
/// scheduler routes the few-step denoise through the unified curated framework over the DMD σ grid
/// (sc-9009), while unset keeps the byte-exact native DMD student loop.
pub fn descriptor_turbo() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = BOOGU_IMAGE_TURBO_ID;
    d.capabilities.supports_guidance = false;
    d.capabilities.samplers = TURBO_SAMPLERS.to_vec();
    d
}

/// Boogu Edit descriptor — same true-CFG surface as the Base path plus one or more img2img/instruction
/// -edit source images ([`ConditioningKind::Reference`] for a single source, or
/// [`ConditioningKind::MultiReference`] for up to `MAX_EDIT_REFERENCES`): each source is read by the
/// Qwen3-VL vision tower (semantic edit) and VAE-encoded into the DiT's spatial reference sequence.
pub fn descriptor_edit() -> ModelDescriptor {
    let mut d = descriptor();
    d.id = BOOGU_IMAGE_EDIT_ID;
    d.capabilities.conditioning = vec![
        ConditioningKind::Reference,
        ConditioningKind::MultiReference,
    ];
    d
}

fn build(
    spec: &LoadSpec,
    descriptor: ModelDescriptor,
    variant: Variant,
) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(format!(
                "{} expects a snapshot directory (mllm/ transformer/ vae/), not a single \
                 .safetensors file",
                descriptor.id
            )));
        }
    };
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {} does not accept user LoRA/LoKr adapters",
            descriptor.id
        )));
    }
    // sc-9607: `spec.quantize` (Q4/Q8) is ACCEPTED and no-ops — the resolved per-tier turnkey is
    // already MLX-packed and `loader::linear_detect` builds each `QLinear::Quantized` straight from the
    // packed parts (sc-9410, group-size-aware), so no on-the-fly quant pass runs. Advertising
    // `supported_quants` lets the worker's A-B tier toggle engage; the requested quant is recipe-only.
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(format!(
            "candle {} does not support ControlNet / IP-Adapter overlays",
            descriptor.id
        )));
    }
    let memory_strategy =
        if root.exists() && memory_strategy::canonical_load_identity(descriptor.id, spec) {
            Some(memory_strategy::PreparedMemory::prepare(
                descriptor.id,
                spec,
            )?)
        } else {
            // Imported/custom snapshots retain the historical resident loader without an eager digest
            // pass. They have no optimized contract, and a direct optimized request fails closed.
            None
        };
    let device = candle_gen::default_device()?;
    let memory_admission = memory_strategy::AdmissionRegistry::new(descriptor.id);
    Ok(Box::new(BooguGenerator {
        descriptor,
        root,
        variant,
        device,
        // PiD is an optional aux decoder (epic 7840 / sc-7853): capture the load-spec component (if
        // any) so the lazy component build loads the engine once. Unlike adapters/control above, it is
        // not rejected — `None` simply keeps the byte-exact native-VAE path.
        pid_spec: spec.pid.clone(),
        components: Mutex::new(None),
        edit_components: Mutex::new(None),
        img2img_encoder: Mutex::new(None),
        memory_strategy,
        memory_admission,
        request_lock: Mutex::new(()),
    }))
}

/// Construct a lazy candle Boogu **Base** generator. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a candle-readable (bf16) Boogu snapshot (`mllm/ transformer/ vae/`).
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, descriptor(), Variant::Base)
}

/// Construct a lazy candle Boogu **Turbo** generator (DMD few-step, CFG-free).
pub fn load_turbo(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, descriptor_turbo(), Variant::Turbo)
}

/// Construct a lazy candle Boogu **Edit** generator (single-reference TI2I, true-CFG).
pub fn load_edit(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    build(spec, descriptor_edit(), Variant::Edit)
}

candle_gen::register_generators! {
    pub(crate) const BASE_REGISTRATION = descriptor => load
}
candle_gen::register_generators! {
    pub(crate) const TURBO_REGISTRATION = descriptor_turbo => load_turbo
}
candle_gen::register_generators! {
    pub(crate) const EDIT_REGISTRATION = descriptor_edit => load_edit
}

/// Add all Candle Boogu providers to an explicit media registry builder.
pub fn register_providers(
    registry: candle_gen::gen_core::ProviderRegistryBuilder,
) -> candle_gen::gen_core::ProviderRegistryBuilder {
    let registry = registry
        .register_generator(BASE_REGISTRATION)
        .register_generator(TURBO_REGISTRATION)
        .register_generator(EDIT_REGISTRATION);
    register_memory_contract_surfaces(registry)
        .register_memory_behavior(BASE_MEMORY_BEHAVIOR)
        .register_memory_behavior(TURBO_MEMORY_BEHAVIOR)
        .register_memory_behavior(EDIT_MEMORY_BEHAVIOR)
}

pub fn register_memory_contract_surfaces(
    registry: gen_core::ProviderRegistryBuilder,
) -> gen_core::ProviderRegistryBuilder {
    registry
        .register_memory_strategy(BASE_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            provider_id: BOOGU_IMAGE_ID,
            contract: memory_strategy::weights_free_base,
        })
        .register_memory_strategy(TURBO_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            provider_id: BOOGU_IMAGE_TURBO_ID,
            contract: memory_strategy::weights_free_turbo,
        })
        .register_memory_strategy(EDIT_MEMORY_REGISTRATION)
        .register_memory_contract_fixture(gen_core::MemoryContractFixtureRegistration {
            surface_specs: gen_core::candle_memory_contract_surface_specs,
            provider_id: BOOGU_IMAGE_EDIT_ID,
            contract: memory_strategy::weights_free_edit,
        })
}

const BASE_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: BOOGU_IMAGE_ID,
    contract: memory_strategy::registered_base,
    safety_check: memory_strategy::registered_safety_check,
};
const TURBO_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: BOOGU_IMAGE_TURBO_ID,
    contract: memory_strategy::registered_turbo,
    safety_check: memory_strategy::registered_safety_check,
};
const EDIT_MEMORY_REGISTRATION: gen_core::MemoryRegistration = gen_core::MemoryRegistration {
    provider_id: BOOGU_IMAGE_EDIT_ID,
    contract: memory_strategy::registered_edit,
    safety_check: memory_strategy::registered_safety_check,
};

const BASE_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: BOOGU_IMAGE_ID,
        valid_fixtures: memory_strategy::valid_fixtures,
        begin_request: memory_strategy::registered_begin,
    };
const TURBO_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: BOOGU_IMAGE_TURBO_ID,
        valid_fixtures: memory_strategy::valid_fixtures,
        begin_request: memory_strategy::registered_begin,
    };
const EDIT_MEMORY_BEHAVIOR: gen_core::MemoryBehaviorRegistration =
    gen_core::MemoryBehaviorRegistration {
        provider_id: BOOGU_IMAGE_EDIT_ID,
        valid_fixtures: memory_strategy::valid_fixtures,
        begin_request: memory_strategy::registered_begin,
    };

/// Build the complete explicit Candle Boogu provider catalog.
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
            ["boogu_image", "boogu_image_turbo", "boogu_image_edit"]
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
            ["boogu_image", "boogu_image_turbo", "boogu_image_edit"]
        );
        assert_eq!(
            fixtures,
            ["boogu_image", "boogu_image_turbo", "boogu_image_edit"]
        );
        assert_eq!(
            behaviors,
            ["boogu_image", "boogu_image_turbo", "boogu_image_edit"]
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights_free_generator() -> BooguGenerator {
        BooguGenerator {
            descriptor: descriptor(),
            root: PathBuf::from("/nonexistent"),
            variant: Variant::Base,
            device: Device::Cpu,
            pid_spec: None,
            components: Mutex::new(None),
            edit_components: Mutex::new(None),
            img2img_encoder: Mutex::new(None),
            memory_strategy: None,
            memory_admission: memory_strategy::AdmissionRegistry::new(BOOGU_IMAGE_ID),
            request_lock: Mutex::new(()),
        }
    }

    #[test]
    fn staged_boundaries_honor_every_fault_and_cancellation_before_loading() {
        let generator = weights_free_generator();
        for phase in [
            gen_core::MemoryPhase::Conditioning,
            gen_core::MemoryPhase::Denoise,
            gen_core::MemoryPhase::Decode,
        ] {
            let mut memory = gen_core::GenerationMemory {
                stage_residency: true,
                ..Default::default()
            };
            memory.authorize_calibration_fault(phase);
            let request = GenerationRequest {
                memory: Some(memory),
                ..Default::default()
            };
            let error = generator.staged_boundary(&request, phase).unwrap_err();
            assert!(error.to_string().contains(&format!("{phase:?}")));

            let clean = GenerationRequest {
                memory: Some(gen_core::GenerationMemory {
                    stage_residency: true,
                    ..Default::default()
                }),
                ..Default::default()
            };
            generator.staged_boundary(&clean, phase).unwrap();
        }

        let canceled = GenerationRequest::default();
        canceled.cancel.cancel();
        assert!(matches!(
            generator.staged_boundary(&canceled, gen_core::MemoryPhase::Conditioning),
            Err(gen_core::Error::Canceled)
        ));
    }

    #[test]
    fn registers_all_three_ids_as_candle() {
        for id in [BOOGU_IMAGE_ID, BOOGU_IMAGE_TURBO_ID, BOOGU_IMAGE_EDIT_ID] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
            let g = crate::provider_registry()
                .unwrap()
                .load(id, &spec)
                .unwrap_or_else(|_| panic!("{id} is registered"));
            assert_eq!(g.descriptor().id, id);
            assert_eq!(g.descriptor().family, "boogu");
            assert_eq!(g.descriptor().backend, "candle");
            assert!(!g.descriptor().capabilities.mac_only);
        }
    }

    #[test]
    fn descriptor_surfaces() {
        let b = descriptor();
        assert!(b.capabilities.supports_guidance);
        assert!(!b.capabilities.supports_negative_prompt);
        assert!(b.capabilities.supports_preview);
        // sc-11786: Base advertises a single-`Reference` img2img surface (no MultiReference).
        assert_eq!(
            b.capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        // sc-9607: packed tiers advertised so the worker A-B toggle engages; turbo + edit inherit it.
        assert_eq!(b.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        let t = descriptor_turbo();
        assert_eq!(t.id, BOOGU_IMAGE_TURBO_ID);
        assert!(!t.capabilities.supports_guidance);
        assert_eq!(t.capabilities.samplers, TURBO_SAMPLERS.to_vec());
        assert_eq!(t.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
        assert!(t.capabilities.supports_preview);
        assert_eq!(
            descriptor_edit().capabilities.supported_quants,
            &[Quant::Q4, Quant::Q8]
        );
        assert!(descriptor_edit().capabilities.supports_preview);
    }

    #[test]
    fn turbo_advertised_menu_is_honored_curated_names() {
        // sc-9009: every advertised Turbo sampler must be a real curated solver name — the routing in
        // `render_turbo` hands `req.sampler` to `run_flow_sampler`, whose N3 fallback silently
        // substitutes Euler for an unknown name, which would resurrect the silent-ignore trap.
        let curated = candle_gen::curated_sampler_names();
        for s in TURBO_SAMPLERS {
            assert!(
                curated.contains(s),
                "advertised turbo sampler {s:?} is not a curated solver: {curated:?}"
            );
        }
        // The scheduler axis is advertised (inherited from Base) and honored by the same routing.
        let t = descriptor_turbo();
        assert_eq!(
            t.capabilities.schedulers,
            candle_gen::curated_scheduler_names()
        );
    }

    #[test]
    fn descriptor_edit_adds_reference() {
        let d = descriptor_edit();
        assert_eq!(d.id, BOOGU_IMAGE_EDIT_ID);
        assert!(d.capabilities.supports_guidance);
        // Edit advertises both single- and multi-reference conditioning.
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::Reference));
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::MultiReference));
        // Base/Turbo advertise the single-`Reference` img2img surface (sc-11786) but NOT the
        // multi-image edit path — that stays the Edit checkpoint's alone.
        for t2i in [descriptor(), descriptor_turbo()] {
            assert_eq!(
                t2i.capabilities.conditioning,
                vec![ConditioningKind::Reference]
            );
            assert!(!t2i
                .capabilities
                .conditioning
                .contains(&ConditioningKind::MultiReference));
        }
    }

    #[test]
    fn edit_validate_reference_count() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(BOOGU_IMAGE_EDIT_ID, &spec)
            .unwrap();
        let img = |w: u32, h: u32| Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        };
        let one_ref = || Conditioning::Reference {
            image: img(512, 512),
            strength: None,
        };
        let base = GenerationRequest {
            prompt: "make it autumn".into(),
            width: 512,
            height: 512,
            ..Default::default()
        };
        // No reference → error.
        assert!(g.validate(&base).is_err());
        // A single `Reference` → ok.
        let one = GenerationRequest {
            conditioning: vec![one_ref()],
            ..base.clone()
        };
        assert!(g.validate(&one).is_ok());
        // Two references (now supported, up to 5) → ok.
        let two = GenerationRequest {
            conditioning: vec![one_ref(), one_ref()],
            ..base.clone()
        };
        assert!(g.validate(&two).is_ok());
        // A `MultiReference` with the max 5 images → ok.
        let five = GenerationRequest {
            conditioning: vec![Conditioning::MultiReference {
                images: (0..5).map(|_| img(512, 512)).collect(),
            }],
            ..base.clone()
        };
        assert!(g.validate(&five).is_ok());
        // Six references → error (past the `image_index_embedding` cap).
        let six = GenerationRequest {
            conditioning: vec![Conditioning::MultiReference {
                images: (0..6).map(|_| img(512, 512)).collect(),
            }],
            ..base
        };
        assert!(g.validate(&six).is_err());
    }

    #[test]
    fn base_and_turbo_accept_a_single_img2img_reference() {
        // sc-11786: Base/Turbo advertise a single-`Reference` img2img surface, so the capability floor
        // accepts one reference (with or without a strength) but a second reference on the t2i path is
        // rejected (single img2img init only; multi-image is the Edit checkpoint's path).
        let img = |w: u32, h: u32| Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        };
        let one_ref = || Conditioning::Reference {
            image: img(512, 512),
            strength: Some(0.6),
        };
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        for id in [BOOGU_IMAGE_ID, BOOGU_IMAGE_TURBO_ID] {
            let g = crate::provider_registry().unwrap().load(id, &spec).unwrap();
            let one = GenerationRequest {
                prompt: "make it autumn".into(),
                width: 512,
                height: 512,
                conditioning: vec![one_ref()],
                ..Default::default()
            };
            assert!(
                g.validate(&one).is_ok(),
                "{id}: single Reference is img2img"
            );
            // Two references on the t2i path → error (single img2img init only).
            let two = GenerationRequest {
                conditioning: vec![one_ref(), one_ref()],
                ..one
            };
            assert!(g.validate(&two).is_err(), "{id}: two references rejected");
            // A `MultiReference` is not advertised on Base/Turbo → floor rejects it.
            let multi = GenerationRequest {
                prompt: "x".into(),
                width: 512,
                height: 512,
                conditioning: vec![Conditioning::MultiReference {
                    images: vec![img(512, 512), img(512, 512)],
                }],
                ..Default::default()
            };
            assert!(g.validate(&multi).is_err(), "{id}: MultiReference rejected");
        }
    }

    #[test]
    fn resolve_reference_strength_falls_back_to_request() {
        // sc-11786: a per-reference `strength` overrides `req.strength`; an unset one falls back to it.
        let img = Image {
            width: 512,
            height: 512,
            pixels: vec![0u8; 512 * 512 * 3],
        };
        let fallback = GenerationRequest {
            prompt: "x".into(),
            strength: Some(0.4),
            conditioning: vec![Conditioning::Reference {
                image: img.clone(),
                strength: None,
            }],
            ..Default::default()
        };
        assert_eq!(
            pipeline::resolve_reference(&fallback, BOOGU_IMAGE_ID)
                .unwrap()
                .unwrap()
                .1,
            Some(0.4)
        );
        // No reference → None (pure txt2img).
        assert!(
            pipeline::resolve_reference(&GenerationRequest::default(), BOOGU_IMAGE_ID)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn validate_accepts_txt2img_and_rejects_bad() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(BOOGU_IMAGE_ID, &spec)
            .unwrap();
        let ok = GenerationRequest {
            prompt: "a red apple on a wooden table".into(),
            guidance: Some(4.0),
            ..Default::default()
        };
        assert!(g.validate(&ok).is_ok());
        for bad in [
            GenerationRequest::default(),
            GenerationRequest {
                prompt: "x".into(),
                width: 1000,
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

        // sc-12612: `SIZE_MULTIPLE` is the pinned stride SceneWorks ties every advertised Boogu bucket
        // to. Pin the value and mutation-check that an in-range size which is a multiple of 8 (the VAE
        // scale) but not SIZE_MULTIPLE (16) is still rejected with the stride error, and on-stride passes.
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

    /// F-154 (sc-11210): the empty-prompt guard rejects a whitespace-only prompt (`trim().is_empty()`),
    /// matching the chroma/krea-control siblings — a whitespace prompt otherwise reaches the TE as an
    /// effectively-empty sequence.
    #[test]
    fn validate_rejects_whitespace_only_prompt() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = crate::provider_registry()
            .unwrap()
            .load(BOOGU_IMAGE_ID, &spec)
            .unwrap();
        for ws in ["   ", "\t", "\n", " \t\n "] {
            let req = GenerationRequest {
                prompt: ws.into(),
                guidance: Some(4.0),
                ..Default::default()
            };
            assert!(
                g.validate(&req).is_err(),
                "whitespace-only prompt {ws:?} must be rejected"
            );
        }
    }

    #[test]
    fn load_rejects_single_file_and_unwired_surfaces() {
        use candle_gen::gen_core::{AdapterKind, AdapterSpec};
        let file = LoadSpec::new(WeightsSource::File("/tmp/q.safetensors".into()));
        assert!(load(&file).is_err());
        let lora = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_adapters(vec![
            AdapterSpec::new("/lora.safetensors".into(), 1.0, AdapterKind::Lora),
        ]);
        assert!(matches!(
            load(&lora).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
    }
}
