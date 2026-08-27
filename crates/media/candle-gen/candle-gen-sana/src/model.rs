//! `SanaGenerator` — the [`gen_core::Generator`] implementation for **SANA-1.6B 1024px** on the candle
//! (Windows/CUDA + Linux) backend, plus its [`descriptor`]/[`load`] entry points and the
//! explicit registration constant that wires it into a provider catalog under the id `"sana_1600m"`
//! (epic 11776, story sc-11780 — the candle-gen half; the mlx sibling is `mlx-gen-sana::model`).
//!
//! The family and platform catalogs compose `REGISTRATION`, so a registry load for
//! `"sana_1600m"` returns this Candle generator.
//!
//! ## Snapshot layout
//!
//! [`load`] assembles the pipeline from an `Efficient-Large-Model/Sana_1600M_1024px_diffusers`-shaped
//! snapshot directory (the whole-repo HF snapshot):
//!
//! ```text
//!   transformer/…safetensors   → SanaTransformer   (the Linear-DiT trunk)
//!   vae/…safetensors           → DcAeEncoder/Decoder (DC-AE f32c32 autoencoder)
//!   text_encoder/…safetensors  → gemma-2-2b-it     (CHI caption encoder weights)
//!   tokenizer/tokenizer.json   ↗ gemma tokenizer
//! ```
//!
//! [`crate::pipeline::resolve_component_files`] tolerates the diffusers tree's fp16/fp32 and
//! single/sharded duplication, so no curated allow-list is needed — the whole repo snapshot loads.
//!
//! ## Sampling recipe
//!
//! SANA-1.6B is a **true-CFG** flow-match model: default **20 steps / guidance 4.5** (diffusers
//! `SanaPipeline.__call__`), negative prompt supported, flow-match Euler over a static shift 3.0
//! schedule routed through the unified epic-7114 sampler. When `guidance <= 1.0` the uncond forward is
//! skipped (CFG off). A single reference image drives latent-init img2img with strength-based schedule
//! truncation; control/IP-adapter overlays, LoRA, and load-time quantization remain unwired and are
//! rejected rather than silently dropped.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, Capabilities, Conditioning, ConditioningKind, GenerationOutput, GenerationRequest,
    Generator, Image, LoadSpec, Modality, ModelDescriptor, Progress, WeightsSource,
};

use crate::memory_strategy::{AdmissionRegistry, SanaLoadSeal, SanaVariant};
use crate::pipeline::{SanaGenerateRequest, SanaPipeline, SanaSprintPipeline};

/// Registry id for SANA-1.6B 1024px (must match the SceneWorks worker's routing / `payload.model`).
pub const MODEL_ID: &str = "sana_1600m";

/// Registry id for **SANA-Sprint** 1.6B 1024px — the CFG-free, SCM/TrigFlow few-step variant
/// (sc-11781). The SceneWorks worker catalog (5b) routes to this EXACT id.
pub const SPRINT_MODEL_ID: &str = "sana_sprint_1600m";

/// SANA-1.6B's native generation resolution. The model is bucket-trained at 1024² and the only
/// real-weight e2e that exists validates 1024², so 1024 is the validated engine envelope; the DC-AE
/// decoder runs the full f32 decode monolithically (no tiling), so we advertise only what we can honor.
const RES_MIN: u32 = 256;
const RES_MAX: u32 = 1024;
/// DC-AE 32× spatial compression — requested dims must be a multiple of this so the latent edge
/// (`image / 32`) is integral. Exposed as the pinned-engine stride SceneWorks ties each advertised
/// SANA image bucket to (sc-12612), mirroring `wan::config::SIZE_MULTIPLE_14B`. `validate_request`
/// enforces exactly this value, so the const cannot drift from the check.
pub const RES_MULTIPLE: u32 = crate::pipeline::SPATIAL_SCALE;
/// Max images per request (the image-model standard, shared with the other candle families).
const MAX_COUNT: u32 = 8;

/// A loaded candle SANA generator. Loading is **lazy** (no file I/O in [`load`]); the heavy components
/// (gemma-2-2b-it TE + Linear-DiT trunk + DC-AE encoder/decoder) are built on the first
/// [`generate`](Generator::generate) call and cached (mirrors the sibling candle providers).
pub struct SanaGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// Cached composed pipeline. `Mutex` because `Generator` is shared and `generate` takes `&self`.
    pipeline: Mutex<Option<std::sync::Arc<SanaPipeline>>>,
    load_seal: Arc<SanaLoadSeal>,
    memory_admission: AdmissionRegistry,
    lifecycle: Mutex<()>,
}

trait BaseBatchPipeline {
    type Conditioning;
    type PreparedReference;

    fn encode_batch(
        &self,
        req: &SanaGenerateRequest<'_>,
        guidance: f32,
    ) -> candle_gen::Result<Self::Conditioning>;
    fn prepare_reference(
        &self,
        req: &SanaGenerateRequest<'_>,
        device: &Device,
        cancel: &gen_core::CancelFlag,
    ) -> candle_gen::Result<Option<Self::PreparedReference>>;
    #[allow(clippy::too_many_arguments)]
    fn render_seed(
        &self,
        req: &SanaGenerateRequest<'_>,
        conditioning: &Self::Conditioning,
        prepared_reference: Option<&Self::PreparedReference>,
        device: &Device,
        cancel: &gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &candle_gen::preview::PreviewHook<'_>,
        memory: Option<gen_core::GenerationMemory>,
    ) -> candle_gen::Result<Image>;
}

impl BaseBatchPipeline for SanaPipeline {
    type Conditioning = crate::pipeline::SanaConditioning;
    type PreparedReference = candle_gen::candle_core::Tensor;

    fn encode_batch(
        &self,
        req: &SanaGenerateRequest<'_>,
        guidance: f32,
    ) -> candle_gen::Result<Self::Conditioning> {
        self.encode_conditioning(req, guidance)
    }

    fn prepare_reference(
        &self,
        req: &SanaGenerateRequest<'_>,
        device: &Device,
        cancel: &gen_core::CancelFlag,
    ) -> candle_gen::Result<Option<Self::PreparedReference>> {
        self.prepare_reference(req, device, cancel)
    }

    fn render_seed(
        &self,
        req: &SanaGenerateRequest<'_>,
        conditioning: &Self::Conditioning,
        prepared_reference: Option<&Self::PreparedReference>,
        device: &Device,
        cancel: &gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &candle_gen::preview::PreviewHook<'_>,
        memory: Option<gen_core::GenerationMemory>,
    ) -> candle_gen::Result<Image> {
        self.generate_with_conditioning_and_reference_memory(
            req,
            conditioning,
            prepared_reference,
            device,
            cancel,
            on_progress,
            preview,
            memory,
        )
    }
}

impl SanaGenerator {
    /// Get the cached pipeline, building (and caching) it from the snapshot on the first call.
    fn pipeline(&self) -> gen_core::Result<std::sync::Arc<SanaPipeline>> {
        self.load_seal.ensure_unchanged()?;
        let mut guard = candle_gen::lock_recover(&self.pipeline);
        if let Some(p) = guard.as_ref() {
            return Ok(p.clone());
        }
        // The inner `?` bridges the candle-side load error into `gen_core::Error`.
        let built = std::sync::Arc::new(SanaPipeline::from_diffusers_snapshot(
            &self.root,
            &self.device,
        )?);
        *guard = Some(built.clone());
        Ok(built)
    }
}

/// SANA-1.6B's identity + capabilities — constructible without loading weights for registry
/// introspection and capability advertisement. True-CFG text-to-image and singular-reference img2img:
/// negative prompt + guidance scale, flow-match Euler over the unified curated sampler/scheduler menu
/// (epic 7114).
/// Control/IP-adapter overlays and LoRA are not wired on the candle base path. The production route
/// is sealed to the immutable dense upstream snapshot; packed Q4/Q8 and NVFP4 are not advertised.
/// Backend `"candle"`, `mac_only = false`.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&candle_gen::gen_core::SANA_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: MODEL_ID,
        family: "sana",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // A singular reference is latent-init img2img; control/IP-adapter overlays remain unwired.
            conditioning: vec![ConditioningKind::Reference],
            // Flow-match Euler over the unified curated sampler/scheduler framework (epic 7114); the
            // native loop (`req.sampler == None`) stays the byte-exact default.
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            // A packed quant selector cannot relabel or convert the sealed dense snapshot at load.
            supported_quants: &[],
            // Static flow-match shift 3.0, resolution-independent (handled by the unified sampler).
            requires_sigma_shift: false,
            // sc-16959: the base flow lane emits per-step latent previews through
            // `crate::preview::base_hook` over the epic-16624 BASE DC-AE fit — not Sprint's.
            supports_preview: true,
            supports_sequential_offload: true,
            ..Default::default()
        },
    }
}

/// **SANA-Sprint** identity + capabilities (sc-11781) — same `sana` family / `candle` backend / image
/// modality as the base, but the distilled variant is **CFG-free** (the guidance scale is an *embedded
/// scalar* fed to the trunk, not classifier-free guidance) and **few-step** (1–4, default 2): so
/// `supports_true_cfg = false`, `supports_negative_prompt = false`, and NO
/// `supported_guidance_methods` (the epic-7434 cfg/cfg_rescale/apg/cfg_pp combine operators do not
/// apply — there is no cond/uncond pair). `supports_guidance` stays `true` because the guidance scale
/// is still an honored request knob (it modulates the embedded scalar). The SCM/TrigFlow sampler is a
/// dedicated few-step loop, so the curated epic-7114 sampler/scheduler menu is NOT advertised — only
/// the `"default"` engine sentinel.
pub fn sprint_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: Some(&candle_gen::gen_core::SANA_LATENT_SPACE),
        control_kinds: None,
        required_components: &[],
        id: SPRINT_MODEL_ID,
        family: "sana",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Embedded guidance scalar — honored knob, but NOT classifier-free (no uncond forward).
            supports_negative_prompt: false,
            supports_guidance: true,
            conditioning: vec![ConditioningKind::Reference],
            // The SCM/TrigFlow consistency loop is a dedicated few-step sampler, not a curated
            // epic-7114 `Solver`; only the engine-default sentinel is advertised.
            samplers: vec!["default"],
            schedulers: vec!["default"],
            // CFG-free: no cfg/cfg_rescale/apg/cfg_pp combine (the guidance axis embedded case).
            supported_guidance_methods: vec![],
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            supported_quants: &[],
            // sc-16959: the SCM lane emits per-step latent previews through
            // `crate::preview::sprint_hook` over the epic-16624 SPRINT fit, with the `1/σ_data`
            // correction the SCM driver's pre-scaled running latent needs.
            supports_preview: true,
            ..Default::default()
        },
    }
}

/// Capability-driven request validation, factored out so it can be unit-tested without loaded weights.
/// Delegates the shared size/count/guidance/negative/conditioning checks to the descriptor
/// (`Capabilities::validate_request`) and adds SANA's `RES_MULTIPLE` (32×, DC-AE) divisor rule.
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
    if req.strength.is_some()
        && !req
            .conditioning
            .iter()
            .any(|conditioning| matches!(conditioning, Conditioning::Reference { .. }))
    {
        return Err(gen_core::Error::Unsupported(format!(
            "{id}: img2img strength requires Reference conditioning"
        )));
    }
    if req.steps == Some(0) {
        return Err(gen_core::Error::Msg(format!("{id}: steps must be >= 1")));
    }
    if !req.width.is_multiple_of(RES_MULTIPLE) || !req.height.is_multiple_of(RES_MULTIPLE) {
        return Err(gen_core::Error::Msg(format!(
            "{id}: {}x{} must be a multiple of {RES_MULTIPLE} (DC-AE 32× spatial compression)",
            req.width, req.height
        )));
    }
    Ok(())
}

/// Resolve the single non-edit img2img reference and the common effective strength.
fn resolve_reference<'a>(
    req: &'a GenerationRequest,
    id: &str,
) -> gen_core::Result<Option<(&'a Image, f32)>> {
    let mut reference = None;
    for conditioning in &req.conditioning {
        if let Conditioning::Reference { image, strength } = conditioning {
            if reference.is_some() {
                return Err(gen_core::Error::Msg(format!(
                    "{id}: multiple reference images are not supported (single img2img init only)"
                )));
            }
            reference = Some((
                image,
                crate::pipeline::resolve_strength(*strength, req.strength),
            ));
        }
    }
    Ok(reference)
}

/// Construct the (lazy) candle SANA-1.6B generator from a [`LoadSpec`]. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a `Sana_1600M_1024px_diffusers`-layout snapshot. LoRA/LoKr
/// adapters, control/IP-adapter overlays, and packed tier selectors are rejected.
pub fn load(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "sana_1600m expects a snapshot directory (transformer/ vae/ text_encoder/ \
                 tokenizer/), not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if let Some(quant) = spec.quantize {
        return Err(gen_core::Error::Unsupported(
            format!("candle sana_1600m has no packed {quant:?} route; only the immutable dense checkpoint is executable"),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle sana_1600m does not support LoRA/LoKr adapters yet".into(),
        ));
    }
    if spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
        || !spec.components.is_empty()
    {
        return Err(gen_core::Error::Unsupported(
            "candle sana_1600m supports plain txt2img and singular-reference img2img, not control / \
             IP-adapter overlays"
                .into(),
        ));
    }
    let load_seal = Arc::new(SanaLoadSeal::capture(SanaVariant::Base, spec)?);
    let device = candle_gen::default_device()?;
    Ok(Box::new(SanaGenerator {
        descriptor: descriptor(),
        root,
        device,
        pipeline: Mutex::new(None),
        load_seal,
        memory_admission: AdmissionRegistry::new(MODEL_ID),
        lifecycle: Mutex::new(()),
    }))
}

/// Render `req.count` base-SANA images, previewing every step of every one.
///
/// `preview` is built HERE, over the request's own [`gen_core::PreviewSink`], and threaded down to
/// [`crate::pipeline::denoise_cfg`]'s [`candle_gen::run_flow_sampler`] call as a non-`Option`
/// reference — see `crate::preview` for why the whole path is typed that way rather than only the
/// driver argument. It is the **base** hook: `sana_1600m` and `sana_sprint_1600m` are fits over two
/// different DC-AE autoencoders and must never share one.
fn generate_base_images(
    pipeline: &impl BaseBatchPipeline,
    req: &GenerationRequest,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
    preview: &candle_gen::preview::PreviewHook<'_>,
) -> gen_core::Result<Vec<Image>> {
    let reference = resolve_reference(req, MODEL_ID)?;
    let (init_image, strength) = reference
        .map(|(image, strength)| (Some(image), Some(strength)))
        .unwrap_or((None, None));
    let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
    let steps = req.steps.map(|s| s as usize);
    let guidance_scale = req.true_cfg.or(req.guidance);
    let guidance = guidance_scale.unwrap_or(crate::pipeline::DEFAULT_GUIDANCE);
    let conditioning = pipeline
        .encode_batch(
            &SanaGenerateRequest {
                prompt: &req.prompt,
                negative_prompt: req.negative_prompt.as_deref(),
                height: req.height,
                width: req.width,
                steps,
                guidance_scale,
                seed: None,
                sampler: req.sampler.as_deref(),
                scheduler: req.scheduler.as_deref(),
                init_image,
                strength,
            },
            guidance,
        )
        .map_err(gen_core::Error::from)?;
    let prepared_reference = pipeline
        .prepare_reference(
            &SanaGenerateRequest {
                prompt: &req.prompt,
                negative_prompt: req.negative_prompt.as_deref(),
                height: req.height,
                width: req.width,
                steps,
                guidance_scale,
                seed: None,
                sampler: req.sampler.as_deref(),
                scheduler: req.scheduler.as_deref(),
                init_image,
                strength,
            },
            device,
            &req.cancel,
        )
        .map_err(gen_core::Error::from)?;

    candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
        pipeline
            .render_seed(
                &SanaGenerateRequest {
                    prompt: &req.prompt,
                    negative_prompt: req.negative_prompt.as_deref(),
                    height: req.height,
                    width: req.width,
                    steps,
                    guidance_scale,
                    seed: Some(seed),
                    sampler: req.sampler.as_deref(),
                    scheduler: req.scheduler.as_deref(),
                    init_image,
                    strength,
                },
                &conditioning,
                prepared_reference.as_ref(),
                device,
                &req.cancel,
                on_progress,
                preview,
                req.memory,
            )
            .map_err(gen_core::Error::from)
    })
}

impl Generator for SanaGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor, req)
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(self.load_seal.contract())
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        crate::memory_strategy::safety_check(&self.load_seal, &self.memory_admission, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        crate::memory_strategy::begin_request(
            &self.load_seal,
            self.memory_admission.clone(),
            self.device.clone(),
            context,
        )
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let _lifecycle = candle_gen::lock_recover(&self.lifecycle);
        self.memory_admission.consume(req)?;
        self.load_seal.ensure_unchanged()?;
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let preview = crate::preview::base_hook(&req.preview);
        let images = if req.memory.is_some_and(|memory| memory.stage_residency) {
            let resident = candle_gen::lock_recover(&self.pipeline).take();
            drop(resident);
            self.device
                .synchronize()
                .map_err(gen_core::Error::backend)?;
            let reference = resolve_reference(req, MODEL_ID)?;
            let (init_image, strength) = reference
                .map(|(image, strength)| (Some(image), Some(strength)))
                .unwrap_or((None, None));
            let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
            let seeds = (0..req.count)
                .map(|index| base_seed.wrapping_add(index as u64))
                .collect::<Vec<_>>();
            let staged = SanaGenerateRequest {
                prompt: &req.prompt,
                negative_prompt: req.negative_prompt.as_deref(),
                height: req.height,
                width: req.width,
                steps: req.steps.map(|steps| steps as usize),
                guidance_scale: req.true_cfg.or(req.guidance),
                seed: None,
                sampler: req.sampler.as_deref(),
                scheduler: req.scheduler.as_deref(),
                init_image,
                strength,
            };
            crate::pipeline::generate_base_staged(
                &self.root,
                &staged,
                &seeds,
                req.memory.expect("staged branch has memory"),
                &self.device,
                &req.cancel,
                on_progress,
                &preview,
                || self.load_seal.ensure_unchanged(),
            )
            .map_err(gen_core::Error::from)?
        } else {
            let pipeline = self.pipeline()?;
            generate_base_images(pipeline.as_ref(), req, &self.device, on_progress, &preview)?
        };
        Ok(GenerationOutput::Images(images))
    }
}

/// A loaded candle **SANA-Sprint** generator (sc-11781). Same lazy-load discipline as
/// [`SanaGenerator`] (no file I/O in [`load_sprint`]; the components are built + cached on the first
/// `generate`), but it composes the CFG-free SCM few-step [`SanaSprintPipeline`].
pub struct SanaSprintGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    pipeline: Mutex<Option<std::sync::Arc<SanaSprintPipeline>>>,
    load_seal: Arc<SanaLoadSeal>,
    memory_admission: AdmissionRegistry,
    lifecycle: Mutex<()>,
}

impl SanaSprintGenerator {
    fn pipeline(&self) -> gen_core::Result<std::sync::Arc<SanaSprintPipeline>> {
        self.load_seal.ensure_unchanged()?;
        let mut guard = candle_gen::lock_recover(&self.pipeline);
        if let Some(p) = guard.as_ref() {
            return Ok(p.clone());
        }
        let built = std::sync::Arc::new(SanaSprintPipeline::from_diffusers_snapshot(
            &self.root,
            &self.device,
        )?);
        *guard = Some(built.clone());
        Ok(built)
    }
}

trait SprintBatchPipeline {
    type Conditioning;
    type PreparedReference;

    fn encode_batch(&self, prompt: &str) -> candle_gen::Result<Self::Conditioning>;
    fn prepare_reference(
        &self,
        req: &SanaGenerateRequest<'_>,
        device: &Device,
        cancel: &gen_core::CancelFlag,
    ) -> candle_gen::Result<Option<Self::PreparedReference>>;
    #[allow(clippy::too_many_arguments)]
    fn render_seed(
        &self,
        req: &SanaGenerateRequest<'_>,
        conditioning: &Self::Conditioning,
        prepared_reference: Option<&Self::PreparedReference>,
        device: &Device,
        cancel: &gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &candle_gen::preview::PreviewHook<'_>,
        memory: Option<gen_core::GenerationMemory>,
    ) -> candle_gen::Result<Image>;
}

impl SprintBatchPipeline for SanaSprintPipeline {
    type Conditioning = candle_gen::candle_core::Tensor;
    type PreparedReference = candle_gen::candle_core::Tensor;

    fn encode_batch(&self, prompt: &str) -> candle_gen::Result<Self::Conditioning> {
        self.encode_conditioning(prompt)
    }

    fn prepare_reference(
        &self,
        req: &SanaGenerateRequest<'_>,
        device: &Device,
        cancel: &gen_core::CancelFlag,
    ) -> candle_gen::Result<Option<Self::PreparedReference>> {
        self.prepare_reference(req, device, cancel)
    }

    fn render_seed(
        &self,
        req: &SanaGenerateRequest<'_>,
        conditioning: &Self::Conditioning,
        prepared_reference: Option<&Self::PreparedReference>,
        device: &Device,
        cancel: &gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
        preview: &candle_gen::preview::PreviewHook<'_>,
        memory: Option<gen_core::GenerationMemory>,
    ) -> candle_gen::Result<Image> {
        self.generate_with_conditioning_and_reference_memory(
            req,
            conditioning,
            prepared_reference,
            device,
            cancel,
            on_progress,
            preview,
            memory,
        )
    }
}

/// Render `req.count` SANA-Sprint images, previewing every SCM step of every one.
///
/// The Sprint twin of [`generate_base_images`], and deliberately its own function rather than a
/// generic over both: the hook it threads carries the **Sprint** fit and the `1/σ_data` correction the
/// SCM driver's running latent needs, neither of which the base lane wants.
fn generate_sprint_images(
    pipeline: &impl SprintBatchPipeline,
    req: &GenerationRequest,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
    preview: &candle_gen::preview::PreviewHook<'_>,
) -> gen_core::Result<Vec<Image>> {
    let reference = resolve_reference(req, SPRINT_MODEL_ID)?;
    let (init_image, strength) = reference
        .map(|(image, strength)| (Some(image), Some(strength)))
        .unwrap_or((None, None));
    let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
    let steps = req.steps.map(|s| s as usize);
    let conditioning = pipeline
        .encode_batch(&req.prompt)
        .map_err(gen_core::Error::from)?;
    let prepared_reference = pipeline
        .prepare_reference(
            &SanaGenerateRequest {
                prompt: &req.prompt,
                negative_prompt: None,
                height: req.height,
                width: req.width,
                steps,
                guidance_scale: req.guidance,
                seed: None,
                sampler: None,
                scheduler: None,
                init_image,
                strength,
            },
            device,
            &req.cancel,
        )
        .map_err(gen_core::Error::from)?;

    candle_gen::for_each_image_seed(base_seed, req.count, |seed| {
        pipeline
            .render_seed(
                &SanaGenerateRequest {
                    prompt: &req.prompt,
                    negative_prompt: None,
                    height: req.height,
                    width: req.width,
                    steps,
                    guidance_scale: req.guidance,
                    seed: Some(seed),
                    sampler: None,
                    scheduler: None,
                    init_image,
                    strength,
                },
                &conditioning,
                prepared_reference.as_ref(),
                device,
                &req.cancel,
                on_progress,
                preview,
                req.memory,
            )
            .map_err(gen_core::Error::from)
    })
}

/// Construct the (lazy) candle **SANA-Sprint** generator (sc-11781) from a [`LoadSpec`]. Identical
/// snapshot contract to [`load`] (`transformer/ vae/ text_encoder/ tokenizer/`), but the transformer
/// loads the Sprint config (guidance embedder + qk-norm) and the CFG-free SCM few-step pipeline drives
/// it. LoRA/LoKr adapters, control/IP-adapter overlays, and packed tier selectors are rejected.
pub fn load_sprint(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(gen_core::Error::Msg(
                "sana_sprint_1600m expects a snapshot directory (transformer/ vae/ text_encoder/ \
                 tokenizer/), not a single .safetensors file"
                    .into(),
            ));
        }
    };
    if let Some(quant) = spec.quantize {
        return Err(gen_core::Error::Unsupported(
            format!("candle sana_sprint_1600m has no packed {quant:?} route; only the immutable dense checkpoint is executable"),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle sana_sprint_1600m does not support LoRA/LoKr adapters yet".into(),
        ));
    }
    if spec.control.is_some()
        || !spec.extra_controls.is_empty()
        || spec.ip_adapter.is_some()
        || spec.pid.is_some()
        || spec.identity.is_some()
        || spec.text_encoder.is_some()
        || !spec.components.is_empty()
    {
        return Err(gen_core::Error::Unsupported(
            "candle sana_sprint_1600m supports plain txt2img and singular-reference img2img, not \
             control / IP-adapter overlays"
                .into(),
        ));
    }
    let load_seal = Arc::new(SanaLoadSeal::capture(SanaVariant::Sprint, spec)?);
    let device = candle_gen::default_device()?;
    Ok(Box::new(SanaSprintGenerator {
        descriptor: sprint_descriptor(),
        root,
        device,
        pipeline: Mutex::new(None),
        load_seal,
        memory_admission: AdmissionRegistry::new(SPRINT_MODEL_ID),
        lifecycle: Mutex::new(()),
    }))
}

impl Generator for SanaSprintGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor, req)
    }

    fn memory_strategy_contract(&self) -> Option<&gen_core::MemoryProviderContract> {
        Some(self.load_seal.contract())
    }

    fn memory_strategy_safety_check(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::MemorySafetyDecision {
        crate::memory_strategy::safety_check(&self.load_seal, &self.memory_admission, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &gen_core::MemoryRunContext,
    ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
        crate::memory_strategy::begin_request(
            &self.load_seal,
            self.memory_admission.clone(),
            self.device.clone(),
            context,
        )
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let _lifecycle = candle_gen::lock_recover(&self.lifecycle);
        self.memory_admission.consume(req)?;
        self.load_seal.ensure_unchanged()?;
        if req.cancel.is_cancelled() {
            return Err(gen_core::Error::Canceled);
        }
        let preview = crate::preview::sprint_hook(&req.preview);
        let images = if req.memory.is_some_and(|memory| memory.stage_residency) {
            let resident = candle_gen::lock_recover(&self.pipeline).take();
            drop(resident);
            self.device
                .synchronize()
                .map_err(gen_core::Error::backend)?;
            let reference = resolve_reference(req, SPRINT_MODEL_ID)?;
            let (init_image, strength) = reference
                .map(|(image, strength)| (Some(image), Some(strength)))
                .unwrap_or((None, None));
            let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
            let seeds = (0..req.count)
                .map(|index| base_seed.wrapping_add(index as u64))
                .collect::<Vec<_>>();
            let staged = SanaGenerateRequest {
                prompt: &req.prompt,
                negative_prompt: None,
                height: req.height,
                width: req.width,
                steps: req.steps.map(|steps| steps as usize),
                guidance_scale: req.guidance,
                seed: None,
                sampler: None,
                scheduler: None,
                init_image,
                strength,
            };
            crate::pipeline::generate_sprint_staged(
                &self.root,
                &staged,
                &seeds,
                req.memory.expect("staged branch has memory"),
                &self.device,
                &req.cancel,
                on_progress,
                &preview,
                || self.load_seal.ensure_unchanged(),
            )
            .map_err(gen_core::Error::from)?
        } else {
            let pipeline = self.pipeline()?;
            generate_sprint_images(pipeline.as_ref(), req, &self.device, on_progress, &preview)?
        };
        Ok(GenerationOutput::Images(images))
    }
}

// Named registrations composed by the explicit SANA family and Candle platform catalogs.
candle_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}
candle_gen::register_generators! {
    pub(crate) const SPRINT_REGISTRATION = sprint_descriptor => load_sprint
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::{PreviewSink, Quant};
    use candle_gen::preview::PreviewHook;

    use std::cell::{Cell, RefCell};

    /// An inert preview hook for the adapter rows below, which measure conditioning reuse and the
    /// per-seed fan-out rather than previews. The seam's own coverage is `crate::preview`'s tests and
    /// `tests/preview_wiring.rs`; an inert sink here keeps these rows byte-identical to pre-sc-16959.
    fn inert_hook(sink: &PreviewSink) -> PreviewHook<'_> {
        crate::preview::base_hook(sink)
    }

    /// Return one Rust item beginning at `marker`, including its balanced outer braces. The guarded
    /// production adapters contain no brace-bearing string literals, so this deliberately small
    /// parser is stricter and easier to audit than a whole-file substring count.
    fn braced_item<'a>(source: &'a str, marker: &str) -> &'a str {
        let start = source
            .find(marker)
            .unwrap_or_else(|| panic!("missing production item {marker}"));
        let open = source[start..]
            .find('{')
            .map(|offset| start + offset)
            .unwrap_or_else(|| panic!("production item {marker} has no body"));
        let mut depth = 0usize;
        for (offset, byte) in source[open..].bytes().enumerate() {
            match byte {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &source[start..open + offset + 1];
                    }
                }
                _ => {}
            }
        }
        panic!("production item {marker} has unbalanced braces")
    }

    fn replace_in_item(source: &str, marker: &str, from: &str, to: &str) -> String {
        let start = source.find(marker).unwrap();
        let item = braced_item(source, marker);
        let replaced = item.replacen(from, to, 1);
        assert_ne!(item, replaced, "mutation target must exist in {marker}");
        format!(
            "{}{}{}",
            &source[..start],
            replaced,
            &source[start + item.len()..]
        )
    }

    fn check_registered_reference_adapters(source: &str) -> Result<(), String> {
        for marker in [
            "impl BaseBatchPipeline for SanaPipeline {",
            "impl SprintBatchPipeline for SanaSprintPipeline {",
        ] {
            let adapter = braced_item(source, marker);
            if adapter
                .matches("self.generate_with_conditioning_and_reference_memory(")
                .count()
                != 1
                || adapter.contains("self.generate_with_conditioning_memory(")
            {
                return Err(format!(
                    "{marker} must select only the typed prepared-reference render tail"
                ));
            }
        }

        for marker in ["fn generate_base_images(", "fn generate_sprint_images("] {
            let batch = braced_item(source, marker);
            let fanout = batch
                .find("candle_gen::for_each_image_seed(")
                .ok_or_else(|| format!("{marker} lost the production seed fanout"))?;
            let (request_preamble, seed_tail) = batch.split_at(fanout);
            if request_preamble.matches(".prepare_reference(").count() != 1
                || seed_tail.contains(".prepare_reference(")
                || seed_tail.matches("prepared_reference.as_ref(),").count() != 1
            {
                return Err(format!(
                    "{marker} must prepare once before fanout and borrow that value into every tail"
                ));
            }
        }
        Ok(())
    }

    #[test]
    fn registered_adapters_are_bound_to_the_prepared_reference_tail() {
        let shipped = include_str!("model.rs");
        check_registered_reference_adapters(shipped).unwrap();

        for marker in [
            "impl BaseBatchPipeline for SanaPipeline {",
            "impl SprintBatchPipeline for SanaSprintPipeline {",
        ] {
            let reverted = replace_in_item(
                shipped,
                marker,
                "self.generate_with_conditioning_and_reference_memory(",
                "self.generate_with_conditioning_memory(",
            );
            assert!(
                check_registered_reference_adapters(&reverted).is_err(),
                "{marker}: reverting the real adapter to per-seed preparation must fail"
            );
        }

        for marker in ["fn generate_base_images(", "fn generate_sprint_images("] {
            let dropped = replace_in_item(shipped, marker, "prepared_reference.as_ref(),", "None,");
            assert!(
                check_registered_reference_adapters(&dropped).is_err(),
                "{marker}: dropping the request-prepared value must fail"
            );
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct RenderedInputs {
        has_reference: bool,
        prepared_reference: Option<Vec<u8>>,
        strength: Option<f32>,
        guidance_scale: Option<f32>,
    }

    struct BaseFixturePipeline {
        encoder_calls: Cell<usize>,
        reference_encoder_calls: Cell<usize>,
        rendered_seeds: RefCell<Vec<u64>>,
        rendered_inputs: RefCell<Vec<RenderedInputs>>,
    }

    impl BaseBatchPipeline for BaseFixturePipeline {
        type Conditioning = Vec<u8>;
        type PreparedReference = Vec<u8>;

        fn encode_batch(
            &self,
            req: &SanaGenerateRequest<'_>,
            guidance: f32,
        ) -> candle_gen::Result<Self::Conditioning> {
            self.encoder_calls
                .set(self.encoder_calls.get() + if guidance > 1.0 { 2 } else { 1 });
            let mut bytes = req.prompt.as_bytes().to_vec();
            if guidance > 1.0 {
                bytes.extend_from_slice(req.negative_prompt.unwrap_or("").as_bytes());
            }
            Ok(bytes)
        }

        fn prepare_reference(
            &self,
            req: &SanaGenerateRequest<'_>,
            _device: &Device,
            _cancel: &gen_core::CancelFlag,
        ) -> candle_gen::Result<Option<Self::PreparedReference>> {
            if req.init_image.is_some() && req.strength.unwrap_or(0.0) > 0.0 {
                let call = self.reference_encoder_calls.get() + 1;
                self.reference_encoder_calls.set(call);
                Ok(Some(vec![call as u8]))
            } else {
                Ok(None)
            }
        }

        fn render_seed(
            &self,
            req: &SanaGenerateRequest<'_>,
            conditioning: &Self::Conditioning,
            prepared_reference: Option<&Self::PreparedReference>,
            _device: &Device,
            _cancel: &gen_core::CancelFlag,
            _on_progress: &mut dyn FnMut(Progress),
            _preview: &candle_gen::preview::PreviewHook<'_>,
            _memory: Option<gen_core::GenerationMemory>,
        ) -> candle_gen::Result<Image> {
            let seed = req.seed.expect("the adapter supplies every per-image seed");
            self.rendered_seeds.borrow_mut().push(seed);
            self.rendered_inputs.borrow_mut().push(RenderedInputs {
                has_reference: req.init_image.is_some(),
                prepared_reference: prepared_reference.cloned(),
                strength: req.strength,
                guidance_scale: req.guidance_scale,
            });
            Ok(fixture_image(conditioning, seed))
        }
    }

    struct SprintFixturePipeline {
        encoder_calls: Cell<usize>,
        reference_encoder_calls: Cell<usize>,
        rendered_seeds: RefCell<Vec<u64>>,
        rendered_references: RefCell<Vec<Option<Vec<u8>>>>,
    }

    impl SprintBatchPipeline for SprintFixturePipeline {
        type Conditioning = Vec<u8>;
        type PreparedReference = Vec<u8>;

        fn encode_batch(&self, prompt: &str) -> candle_gen::Result<Self::Conditioning> {
            self.encoder_calls.set(self.encoder_calls.get() + 1);
            Ok(prompt.as_bytes().to_vec())
        }

        fn prepare_reference(
            &self,
            req: &SanaGenerateRequest<'_>,
            _device: &Device,
            _cancel: &gen_core::CancelFlag,
        ) -> candle_gen::Result<Option<Self::PreparedReference>> {
            if req.init_image.is_some() && req.strength.unwrap_or(0.0) > 0.0 {
                let call = self.reference_encoder_calls.get() + 1;
                self.reference_encoder_calls.set(call);
                Ok(Some(vec![call as u8]))
            } else {
                Ok(None)
            }
        }

        fn render_seed(
            &self,
            req: &SanaGenerateRequest<'_>,
            conditioning: &Self::Conditioning,
            prepared_reference: Option<&Self::PreparedReference>,
            _device: &Device,
            _cancel: &gen_core::CancelFlag,
            _on_progress: &mut dyn FnMut(Progress),
            _preview: &candle_gen::preview::PreviewHook<'_>,
            _memory: Option<gen_core::GenerationMemory>,
        ) -> candle_gen::Result<Image> {
            let seed = req.seed.expect("the adapter supplies every per-image seed");
            self.rendered_seeds.borrow_mut().push(seed);
            self.rendered_references
                .borrow_mut()
                .push(prepared_reference.cloned());
            Ok(fixture_image(conditioning, seed))
        }
    }

    #[test]
    fn base_adapter_encodes_cfg_once_and_preserves_per_seed_tail() {
        let pipeline = BaseFixturePipeline {
            encoder_calls: Cell::new(0),
            reference_encoder_calls: Cell::new(0),
            rendered_seeds: RefCell::new(Vec::new()),
            rendered_inputs: RefCell::new(Vec::new()),
        };
        let request = GenerationRequest {
            prompt: "cond".into(),
            negative_prompt: Some("uncond".into()),
            guidance: Some(1.0),
            true_cfg: Some(4.5),
            strength: Some(0.6),
            conditioning: vec![Conditioning::Reference {
                image: reference_image(),
                strength: None,
            }],
            seed: Some(u64::MAX - 1),
            count: 4,
            ..req(256, 256)
        };
        let expected_conditioning = b"conduncond";
        let expected = [u64::MAX - 1, u64::MAX, 0, 1]
            .map(|seed| fixture_image(expected_conditioning, seed))
            .to_vec();

        let inert = PreviewSink::default();
        let actual = generate_base_images(
            &pipeline,
            &request,
            &Device::Cpu,
            &mut |_| {},
            &inert_hook(&inert),
        )
        .unwrap();

        assert_eq!(pipeline.encoder_calls.get(), 2);
        assert_eq!(pipeline.reference_encoder_calls.get(), 1);
        assert_eq!(
            *pipeline.rendered_seeds.borrow(),
            vec![u64::MAX - 1, u64::MAX, 0, 1]
        );
        assert_eq!(
            *pipeline.rendered_inputs.borrow(),
            vec![
                RenderedInputs {
                    has_reference: true,
                    prepared_reference: Some(vec![1]),
                    strength: Some(0.6),
                    guidance_scale: Some(4.5),
                };
                4
            ],
            "the base adapter must carry the Reference strength and true_cfg precedence into each render"
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn base_adapter_without_cfg_encodes_only_cond_once() {
        let pipeline = BaseFixturePipeline {
            encoder_calls: Cell::new(0),
            reference_encoder_calls: Cell::new(0),
            rendered_seeds: RefCell::new(Vec::new()),
            rendered_inputs: RefCell::new(Vec::new()),
        };
        let request = GenerationRequest {
            guidance: Some(1.0),
            seed: Some(7),
            count: 3,
            ..req(256, 256)
        };

        let inert = PreviewSink::default();
        let images = generate_base_images(
            &pipeline,
            &request,
            &Device::Cpu,
            &mut |_| {},
            &inert_hook(&inert),
        )
        .unwrap();

        assert_eq!(pipeline.encoder_calls.get(), 1);
        assert_eq!(pipeline.reference_encoder_calls.get(), 0);
        assert_eq!(*pipeline.rendered_seeds.borrow(), vec![7, 8, 9]);
        assert_eq!(
            *pipeline.rendered_inputs.borrow(),
            vec![
                RenderedInputs {
                    has_reference: false,
                    prepared_reference: None,
                    strength: None,
                    guidance_scale: Some(1.0),
                };
                3
            ],
            "the reference-free request must retain the existing text-to-image inputs"
        );
        assert_eq!(images.len(), 3);
    }

    #[test]
    fn sprint_adapter_encodes_once_and_preserves_per_seed_tail() {
        let pipeline = SprintFixturePipeline {
            encoder_calls: Cell::new(0),
            reference_encoder_calls: Cell::new(0),
            rendered_seeds: RefCell::new(Vec::new()),
            rendered_references: RefCell::new(Vec::new()),
        };
        let request = GenerationRequest {
            prompt: "sprint cond".into(),
            seed: Some(11),
            count: 3,
            conditioning: vec![Conditioning::Reference {
                image: reference_image(),
                strength: Some(0.6),
            }],
            ..req(256, 256)
        };
        let expected = [11, 12, 13]
            .map(|seed| fixture_image(b"sprint cond", seed))
            .to_vec();

        let inert = PreviewSink::default();
        let actual = generate_sprint_images(
            &pipeline,
            &request,
            &Device::Cpu,
            &mut |_| {},
            &inert_hook(&inert),
        )
        .unwrap();

        assert_eq!(pipeline.encoder_calls.get(), 1);
        assert_eq!(pipeline.reference_encoder_calls.get(), 1);
        assert_eq!(*pipeline.rendered_seeds.borrow(), vec![11, 12, 13]);
        assert_eq!(
            *pipeline.rendered_references.borrow(),
            vec![Some(vec![1]); 3]
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn both_registered_batch_seams_are_request_local_and_skip_txt2img_encode() {
        let base = BaseFixturePipeline {
            encoder_calls: Cell::new(0),
            reference_encoder_calls: Cell::new(0),
            rendered_seeds: RefCell::new(Vec::new()),
            rendered_inputs: RefCell::new(Vec::new()),
        };
        let sprint = SprintFixturePipeline {
            encoder_calls: Cell::new(0),
            reference_encoder_calls: Cell::new(0),
            rendered_seeds: RefCell::new(Vec::new()),
            rendered_references: RefCell::new(Vec::new()),
        };
        let inert = PreviewSink::default();

        let reference_request = |seed, count| GenerationRequest {
            seed: Some(seed),
            count,
            conditioning: vec![Conditioning::Reference {
                image: reference_image(),
                strength: Some(0.6),
            }],
            ..req(256, 256)
        };
        let text_request = |seed, count| GenerationRequest {
            seed: Some(seed),
            count,
            ..req(256, 256)
        };

        generate_base_images(
            &base,
            &reference_request(20, 3),
            &Device::Cpu,
            &mut |_| {},
            &inert_hook(&inert),
        )
        .unwrap();
        generate_base_images(
            &base,
            &text_request(30, 2),
            &Device::Cpu,
            &mut |_| {},
            &inert_hook(&inert),
        )
        .unwrap();
        generate_base_images(
            &base,
            &reference_request(40, 2),
            &Device::Cpu,
            &mut |_| {},
            &inert_hook(&inert),
        )
        .unwrap();
        assert_eq!(base.reference_encoder_calls.get(), 2);
        assert_eq!(
            base.rendered_inputs
                .borrow()
                .iter()
                .map(|input| input.prepared_reference.clone())
                .collect::<Vec<_>>(),
            [
                vec![Some(vec![1]); 3],
                vec![None; 2],
                vec![Some(vec![2]); 2],
            ]
            .concat(),
            "base must share one latent per img2img request, retain none across requests, and encode none for txt2img"
        );

        generate_sprint_images(
            &sprint,
            &reference_request(50, 3),
            &Device::Cpu,
            &mut |_| {},
            &crate::preview::sprint_hook(&inert),
        )
        .unwrap();
        generate_sprint_images(
            &sprint,
            &text_request(60, 2),
            &Device::Cpu,
            &mut |_| {},
            &crate::preview::sprint_hook(&inert),
        )
        .unwrap();
        generate_sprint_images(
            &sprint,
            &reference_request(70, 2),
            &Device::Cpu,
            &mut |_| {},
            &crate::preview::sprint_hook(&inert),
        )
        .unwrap();
        assert_eq!(sprint.reference_encoder_calls.get(), 2);
        assert_eq!(
            *sprint.rendered_references.borrow(),
            [
                vec![Some(vec![1]); 3],
                vec![None; 2],
                vec![Some(vec![2]); 2],
            ]
            .concat(),
            "Sprint must share one latent per img2img request, retain none across requests, and encode none for txt2img"
        );
    }

    fn fixture_image(conditioning: &[u8], seed: u64) -> Image {
        let mut pixels = conditioning.to_vec();
        pixels.extend_from_slice(&seed.to_le_bytes());
        Image {
            width: pixels.len() as u32,
            height: 1,
            pixels,
        }
    }

    fn req(w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a red panda on a mossy log in a misty forest".into(),
            width: w,
            height: h,
            ..Default::default()
        }
    }

    fn reference_image() -> Image {
        Image {
            width: 32,
            height: 32,
            pixels: vec![128; 32 * 32 * 3],
        }
    }

    #[test]
    fn img2img_strength_precedence_default_zero_and_multiple_rejection() {
        let effective = |reference, request| {
            let mut r = req(256, 256);
            r.strength = request;
            r.conditioning = vec![Conditioning::Reference {
                image: reference_image(),
                strength: reference,
            }];
            resolve_reference(&r, MODEL_ID).unwrap().unwrap().1
        };
        assert_eq!(effective(Some(0.7), Some(0.3)), 0.7);
        assert_eq!(effective(None, Some(0.3)), 0.3);
        assert_eq!(
            effective(None, None),
            crate::pipeline::DEFAULT_IMG2IMG_STRENGTH
        );
        assert_eq!(effective(Some(0.0), Some(0.3)), 0.0);

        let mut r = req(256, 256);
        r.conditioning = vec![
            Conditioning::Reference {
                image: reference_image(),
                strength: None,
            },
            Conditioning::Reference {
                image: reference_image(),
                strength: None,
            },
        ];
        assert!(resolve_reference(&r, SPRINT_MODEL_ID).is_err());
        assert!(validate_request(&descriptor(), &r).is_ok());
        assert!(validate_request(&sprint_descriptor(), &r).is_ok());
    }

    #[test]
    fn refree_strength_is_a_typed_unsupported_knob() {
        let mut r = req(256, 256);
        r.strength = Some(0.6);
        let error = validate_request(&descriptor(), &r).unwrap_err();
        assert!(matches!(error, gen_core::Error::Unsupported(_)));
        assert!(error.to_string().contains("requires Reference"));

        // Omitted reference and omitted strength remain the existing text-to-image request.
        r.strength = None;
        assert!(validate_request(&descriptor(), &r).is_ok());
    }

    /// The seam under test: this provider's explicit family registry resolves our Candle generator.
    /// `load` is lazy, so a nonexistent weights dir still resolves (no file I/O until `generate`).
    #[test]
    fn registers_and_resolves_as_candle() {
        let (_temp, root) = crate::memory_strategy::fixture_snapshot(SanaVariant::Base);
        let spec = LoadSpec::new(WeightsSource::Dir(root));
        let g = crate::provider_registry()
            .unwrap()
            .load(MODEL_ID, &spec)
            .expect("candle sana_1600m is registered");
        assert_eq!(g.descriptor().id, "sana_1600m");
        assert_eq!(g.descriptor().family, "sana");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn descriptor_advertises_cfg_surface() {
        let d = descriptor();
        assert!(d.capabilities.supports_true_cfg);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert_eq!(
            d.capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        assert!(d.capabilities.supported_quants.is_empty());
        assert!(!d.capabilities.mac_only, "candle is Windows/CUDA, not Mac");
        assert_eq!(d.capabilities.samplers, candle_gen::curated_sampler_names());
        assert_eq!(
            d.capabilities.schedulers,
            candle_gen::curated_scheduler_names()
        );
    }

    #[test]
    fn defaults_match_diffusers() {
        // The worker reads steps/guidance defaults from the catalog, but the engine's own
        // diffusers-parity defaults are the source of truth they mirror.
        assert_eq!(crate::pipeline::DEFAULT_STEPS, 20);
        assert!((crate::pipeline::DEFAULT_GUIDANCE - 4.5).abs() < 1e-6);
    }

    #[test]
    fn validate_accepts_1024_square_and_rejects_off_envelope() {
        let d = descriptor();
        assert!(validate_request(&d, &req(1024, 1024)).is_ok());
        // Above the validated DC-AE envelope.
        assert!(validate_request(&d, &req(2048, 2048)).is_err());
        // Not a multiple of 32.
        assert!(validate_request(&d, &req(1000, 1024)).is_err());
        // Empty prompt.
        let mut r = req(1024, 1024);
        r.prompt.clear();
        assert!(validate_request(&d, &r).is_err());
        // Explicit zero steps.
        let mut r = req(1024, 1024);
        r.steps = Some(0);
        assert!(validate_request(&d, &r).is_err());

        // sc-12612: `RES_MULTIPLE` is the pinned stride SceneWorks ties every advertised SANA bucket
        // to. Pin the value and mutation-check that an in-range size which is a multiple of 16 but
        // not RES_MULTIPLE (32) is still rejected with the stride error, and an on-stride size passes.
        assert_eq!(RES_MULTIPLE, 32);
        let off_stride = validate_request(&d, &req(1008, 1024)) // 63×16 — a multiple of 16, not 32
            .unwrap_err()
            .to_string();
        assert!(
            off_stride.contains("multiple of 32"),
            "expected the stride error, got: {off_stride}"
        );
        assert!(validate_request(&d, &req(1024, 1024)).is_ok());
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        let e = load(&spec).err().expect("error").to_string();
        assert!(e.contains("snapshot directory"), "got: {e}");
    }

    #[test]
    fn load_rejects_every_false_packed_tier_and_external_component() {
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(load(&quant).is_err(), "dense Candle must reject Q8");
        let q4 = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q4);
        assert!(load(&q4).is_err(), "dense Candle must reject Q4");
        let nvfp4 = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Nvfp4);
        let error = load(&nvfp4)
            .err()
            .expect("NVFP4 must not be relabeled as affine Q4");
        assert!(error.to_string().contains("no packed"));
        let control = LoadSpec::new(WeightsSource::Dir("/snap".into()))
            .with_control(WeightsSource::Dir("/ctrl".into()));
        assert!(matches!(
            load(&control).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
    }

    #[test]
    fn canceled_requests_fail_before_lazy_weight_loading_for_both_variants() {
        let request = req(256, 256);
        request.cancel.cancel();
        let (_base_temp, base_root) = crate::memory_strategy::fixture_snapshot(SanaVariant::Base);
        let (_sprint_temp, sprint_root) =
            crate::memory_strategy::fixture_snapshot(SanaVariant::Sprint);
        for generator in [
            load(&LoadSpec::new(WeightsSource::Dir(base_root))).unwrap(),
            load_sprint(&LoadSpec::new(WeightsSource::Dir(sprint_root))).unwrap(),
        ] {
            let error = generator.generate(&request, &mut |_| {}).unwrap_err();
            assert!(matches!(error, gen_core::Error::Canceled));
        }
    }

    // =============================================================================================
    // SANA-Sprint (sc-11781) — the CFG-free SCM/TrigFlow few-step adapter.
    // =============================================================================================

    /// The Sprint seam under test: the second `register_generators!` submission resolves the EXACT id
    /// `"sana_sprint_1600m"` (the id the worker catalog 5b routes to) to OUR candle Sprint generator.
    /// `load_sprint` is lazy, so a nonexistent weights dir still resolves.
    #[test]
    fn sprint_registers_and_resolves_as_candle() {
        let (_temp, root) = crate::memory_strategy::fixture_snapshot(SanaVariant::Sprint);
        let spec = LoadSpec::new(WeightsSource::Dir(root));
        let g = crate::provider_registry()
            .unwrap()
            .load(SPRINT_MODEL_ID, &spec)
            .expect("candle sana_sprint_1600m registered");
        assert_eq!(g.descriptor().id, "sana_sprint_1600m");
        assert_eq!(g.descriptor().family, "sana");
        assert_eq!(g.descriptor().backend, "candle");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    /// The Sprint descriptor advertises the CFG-free few-step surface: NO true-CFG, NO negative prompt,
    /// guidance still an honored (embedded) knob, NO curated sampler/scheduler menu, NO guidance
    /// combine methods.
    #[test]
    fn sprint_descriptor_is_cfg_free_few_step() {
        let d = sprint_descriptor();
        assert_eq!(d.id, "sana_sprint_1600m");
        assert!(!d.capabilities.supports_true_cfg, "Sprint is CFG-free");
        assert!(!d.capabilities.supports_negative_prompt);
        assert!(
            d.capabilities.supports_guidance,
            "guidance stays an honored embedded knob"
        );
        assert!(d.capabilities.supported_guidance_methods.is_empty());
        assert_eq!(
            d.capabilities.conditioning,
            vec![ConditioningKind::Reference]
        );
        assert_eq!(d.capabilities.samplers, vec!["default"]);
        assert_eq!(d.capabilities.schedulers, vec!["default"]);
        assert!(d.capabilities.supported_quants.is_empty());
        assert!(!d.capabilities.mac_only, "candle is Windows/CUDA");
    }

    #[test]
    fn sprint_defaults_match_diffusers() {
        assert_eq!(crate::pipeline::SPRINT_DEFAULT_STEPS, 2);
        assert!((crate::pipeline::SPRINT_DEFAULT_GUIDANCE - 4.5).abs() < 1e-6);
    }

    #[test]
    fn sprint_load_rejects_single_file_and_packed_tier_selector() {
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        let e = load_sprint(&file).err().expect("error").to_string();
        assert!(e.contains("snapshot directory"), "got: {e}");
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(load_sprint(&quant).is_err());
    }

    /// CRITICAL base-unchanged regression: adding the Sprint adapter must NOT perturb the base
    /// `sana_1600m` descriptor — it stays true-CFG, negative-prompt, with the curated sampler/scheduler
    /// menu. The base and Sprint descriptors are DISTINCT ids that coexist in the registry.
    #[test]
    fn base_sana_1600m_descriptor_unchanged_by_sprint() {
        let base = descriptor();
        let sprint = sprint_descriptor();
        assert_eq!(base.id, "sana_1600m");
        assert_ne!(base.id, sprint.id, "distinct registry ids");
        // Base is unchanged: true-CFG + negative prompt + the full curated menu.
        assert!(base.capabilities.supports_true_cfg);
        assert!(base.capabilities.supports_negative_prompt);
        assert_eq!(
            base.capabilities.samplers,
            candle_gen::curated_sampler_names()
        );
        assert_eq!(
            base.capabilities.schedulers,
            candle_gen::curated_scheduler_names()
        );
        // Both ids resolve independently through the registry.
        let (_base_temp, base_root) = crate::memory_strategy::fixture_snapshot(SanaVariant::Base);
        let (_sprint_temp, sprint_root) =
            crate::memory_strategy::fixture_snapshot(SanaVariant::Sprint);
        let base_spec = LoadSpec::new(WeightsSource::Dir(base_root));
        let sprint_spec = LoadSpec::new(WeightsSource::Dir(sprint_root));
        assert_eq!(
            crate::provider_registry()
                .unwrap()
                .load(MODEL_ID, &base_spec)
                .unwrap()
                .descriptor()
                .id,
            "sana_1600m"
        );
        assert_eq!(
            crate::provider_registry()
                .unwrap()
                .load(SPRINT_MODEL_ID, &sprint_spec)
                .unwrap()
                .descriptor()
                .id,
            "sana_sprint_1600m"
        );
    }
}
