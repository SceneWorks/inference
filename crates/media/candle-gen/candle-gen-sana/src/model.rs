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
//!   vae/…safetensors           → DcAeDecoder       (DC-AE f32c32 decoder)
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
//! skipped (CFG off). No img2img/control conditioning, LoRA, or load-time quantization is wired on the
//! candle base path — those are rejected rather than silently dropped (the worker routes them to the
//! Python fallback).

use std::path::PathBuf;
use std::sync::Mutex;

use candle_gen::candle_core::Device;
use candle_gen::gen_core::{
    self, Capabilities, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality,
    ModelDescriptor, Progress, WeightsSource,
};

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
/// (gemma-2-2b-it TE + Linear-DiT trunk + DC-AE decoder) are built on the first
/// [`generate`](Generator::generate) call and cached (mirrors the sibling candle providers).
pub struct SanaGenerator {
    descriptor: ModelDescriptor,
    root: PathBuf,
    device: Device,
    /// Cached composed pipeline. `Mutex` because `Generator` is shared and `generate` takes `&self`.
    pipeline: Mutex<Option<std::sync::Arc<SanaPipeline>>>,
}

trait BaseBatchPipeline {
    type Conditioning;

    fn encode_batch(
        &self,
        req: &SanaGenerateRequest<'_>,
        guidance: f32,
    ) -> candle_gen::Result<Self::Conditioning>;
    fn render_seed(
        &self,
        req: &SanaGenerateRequest<'_>,
        conditioning: &Self::Conditioning,
        device: &Device,
        cancel: &gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::Result<Image>;
}

impl BaseBatchPipeline for SanaPipeline {
    type Conditioning = crate::pipeline::SanaConditioning;

    fn encode_batch(
        &self,
        req: &SanaGenerateRequest<'_>,
        guidance: f32,
    ) -> candle_gen::Result<Self::Conditioning> {
        self.encode_conditioning(req, guidance)
    }

    fn render_seed(
        &self,
        req: &SanaGenerateRequest<'_>,
        conditioning: &Self::Conditioning,
        device: &Device,
        cancel: &gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::Result<Image> {
        self.generate_with_conditioning(req, conditioning, device, cancel, on_progress)
    }
}

impl SanaGenerator {
    /// Get the cached pipeline, building (and caching) it from the snapshot on the first call.
    fn pipeline(&self) -> gen_core::Result<std::sync::Arc<SanaPipeline>> {
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

/// SANA-1.6B's identity + capabilities — constructible without loading weights (registry introspection
/// / capability advertisement). True-CFG text-to-image: negative prompt + guidance scale, flow-match
/// Euler over the unified curated sampler/scheduler menu (epic 7114). No img2img / control conditioning,
/// LoRA, or quantization is wired on the candle base path. Backend `"candle"`, `mac_only = false`.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "sana",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // Plain txt2img — no img2img/control conditioning on the base SANA checkpoint.
            conditioning: Vec::new(),
            supports_lora: false,
            supports_lokr: false,
            // Flow-match Euler over the unified curated sampler/scheduler framework (epic 7114); the
            // native loop (`req.sampler == None`) stays the byte-exact default.
            samplers: candle_gen::curated_sampler_names(),
            schedulers: candle_gen::curated_scheduler_names(),
            supported_guidance_methods: vec![],
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            // candle is the Windows/CUDA backend — NOT Mac-only (the MLX provider sets this true).
            mac_only: false,
            // SANA is the f32/bf16 weight path; no load-time quantization is wired yet.
            supported_quants: &[],
            supports_kv_cache: false,
            // Static flow-match shift 3.0, resolution-independent (handled by the unified sampler).
            requires_sigma_shift: false,
            // No candle `render_sequential` residency seam wired (sc-11126).
            supports_sequential_offload: false,
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
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
        id: SPRINT_MODEL_ID,
        family: "sana",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Embedded guidance scalar — honored knob, but NOT classifier-free (no uncond forward).
            supports_negative_prompt: false,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: Vec::new(),
            supports_lora: false,
            supports_lokr: false,
            // The SCM/TrigFlow consistency loop is a dedicated few-step sampler, not a curated
            // epic-7114 `Solver`; only the engine-default sentinel is advertised.
            samplers: vec!["default"],
            schedulers: vec!["default"],
            // CFG-free: no cfg/cfg_rescale/apg/cfg_pp combine (the guidance axis embedded case).
            supported_guidance_methods: vec![],
            min_size: RES_MIN,
            max_size: RES_MAX,
            max_count: MAX_COUNT,
            mac_only: false,
            supported_quants: &[],
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supports_sequential_offload: false,
            // No audio surface (sc-12834): pure image/video model.
            audio_sample_rates: vec![],
            max_audio_duration_secs: None,
            audio_voices: vec![],
            audio_languages: vec![],
            audio_edit_modes: vec![],
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

/// Construct the (lazy) candle SANA-1.6B generator from a [`LoadSpec`]. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a `Sana_1600M_1024px_diffusers`-layout snapshot. LoRA/LoKr
/// adapters, on-the-fly quantization, and control/IP-adapter overlays are rejected (not wired —
/// refusing is more honest than silently dropping; the worker falls back to Python).
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
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle sana_1600m does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle sana_1600m does not support LoRA/LoKr adapters yet".into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle sana_1600m does not support control / IP-adapter overlays yet (txt2img only)"
                .into(),
        ));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(SanaGenerator {
        descriptor: descriptor(),
        root,
        device,
        pipeline: Mutex::new(None),
    }))
}

fn generate_base_images(
    pipeline: &impl BaseBatchPipeline,
    req: &GenerationRequest,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> gen_core::Result<Vec<Image>> {
    let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
    let steps = req.steps.map(|s| s as usize);
    let guidance = req.guidance.unwrap_or(crate::pipeline::DEFAULT_GUIDANCE);
    let conditioning = pipeline
        .encode_batch(
            &SanaGenerateRequest {
                prompt: &req.prompt,
                negative_prompt: req.negative_prompt.as_deref(),
                height: req.height,
                width: req.width,
                steps,
                guidance_scale: req.guidance,
                seed: None,
                sampler: req.sampler.as_deref(),
                scheduler: req.scheduler.as_deref(),
            },
            guidance,
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
                    guidance_scale: req.guidance,
                    seed: Some(seed),
                    sampler: req.sampler.as_deref(),
                    scheduler: req.scheduler.as_deref(),
                },
                &conditioning,
                device,
                &req.cancel,
                on_progress,
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

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.validate(req)?;
        let pipeline = self.pipeline()?;

        let images = generate_base_images(pipeline.as_ref(), req, &self.device, on_progress)?;
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
}

impl SanaSprintGenerator {
    fn pipeline(&self) -> gen_core::Result<std::sync::Arc<SanaSprintPipeline>> {
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

    fn encode_batch(&self, prompt: &str) -> candle_gen::Result<Self::Conditioning>;
    fn render_seed(
        &self,
        req: &SanaGenerateRequest<'_>,
        conditioning: &Self::Conditioning,
        device: &Device,
        cancel: &gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::Result<Image>;
}

impl SprintBatchPipeline for SanaSprintPipeline {
    type Conditioning = candle_gen::candle_core::Tensor;

    fn encode_batch(&self, prompt: &str) -> candle_gen::Result<Self::Conditioning> {
        self.encode_conditioning(prompt)
    }

    fn render_seed(
        &self,
        req: &SanaGenerateRequest<'_>,
        conditioning: &Self::Conditioning,
        device: &Device,
        cancel: &gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::Result<Image> {
        self.generate_with_conditioning(req, conditioning, device, cancel, on_progress)
    }
}

fn generate_sprint_images(
    pipeline: &impl SprintBatchPipeline,
    req: &GenerationRequest,
    device: &Device,
    on_progress: &mut dyn FnMut(Progress),
) -> gen_core::Result<Vec<Image>> {
    let base_seed = req.seed.unwrap_or_else(gen_core::default_seed);
    let steps = req.steps.map(|s| s as usize);
    let conditioning = pipeline
        .encode_batch(&req.prompt)
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
                },
                &conditioning,
                device,
                &req.cancel,
                on_progress,
            )
            .map_err(gen_core::Error::from)
    })
}

/// Construct the (lazy) candle **SANA-Sprint** generator (sc-11781) from a [`LoadSpec`]. Identical
/// snapshot contract to [`load`] (`transformer/ vae/ text_encoder/ tokenizer/`), but the transformer
/// loads the Sprint config (guidance embedder + qk-norm) and the CFG-free SCM few-step pipeline drives
/// it. LoRA/LoKr adapters, on-the-fly quantization, and control/IP-adapter overlays are rejected.
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
    if spec.quantize.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle sana_sprint_1600m does not support on-the-fly Q4/Q8 quantization yet".into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(gen_core::Error::Unsupported(
            "candle sana_sprint_1600m does not support LoRA/LoKr adapters yet".into(),
        ));
    }
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(gen_core::Error::Unsupported(
            "candle sana_sprint_1600m does not support control / IP-adapter overlays yet (txt2img \
             only)"
                .into(),
        ));
    }
    let device = candle_gen::default_device()?;
    Ok(Box::new(SanaSprintGenerator {
        descriptor: sprint_descriptor(),
        root,
        device,
        pipeline: Mutex::new(None),
    }))
}

impl Generator for SanaSprintGenerator {
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
        self.validate(req)?;
        let pipeline = self.pipeline()?;

        let images = generate_sprint_images(pipeline.as_ref(), req, &self.device, on_progress)?;
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
    use candle_gen::gen_core::Quant;

    use std::cell::{Cell, RefCell};

    struct BaseFixturePipeline {
        encoder_calls: Cell<usize>,
        rendered_seeds: RefCell<Vec<u64>>,
    }

    impl BaseBatchPipeline for BaseFixturePipeline {
        type Conditioning = Vec<u8>;

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

        fn render_seed(
            &self,
            req: &SanaGenerateRequest<'_>,
            conditioning: &Self::Conditioning,
            _device: &Device,
            _cancel: &gen_core::CancelFlag,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> candle_gen::Result<Image> {
            let seed = req.seed.expect("the adapter supplies every per-image seed");
            self.rendered_seeds.borrow_mut().push(seed);
            Ok(fixture_image(conditioning, seed))
        }
    }

    struct SprintFixturePipeline {
        encoder_calls: Cell<usize>,
        rendered_seeds: RefCell<Vec<u64>>,
    }

    impl SprintBatchPipeline for SprintFixturePipeline {
        type Conditioning = Vec<u8>;

        fn encode_batch(&self, prompt: &str) -> candle_gen::Result<Self::Conditioning> {
            self.encoder_calls.set(self.encoder_calls.get() + 1);
            Ok(prompt.as_bytes().to_vec())
        }

        fn render_seed(
            &self,
            req: &SanaGenerateRequest<'_>,
            conditioning: &Self::Conditioning,
            _device: &Device,
            _cancel: &gen_core::CancelFlag,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> candle_gen::Result<Image> {
            let seed = req.seed.expect("the adapter supplies every per-image seed");
            self.rendered_seeds.borrow_mut().push(seed);
            Ok(fixture_image(conditioning, seed))
        }
    }

    #[test]
    fn base_adapter_encodes_cfg_once_and_preserves_per_seed_tail() {
        let pipeline = BaseFixturePipeline {
            encoder_calls: Cell::new(0),
            rendered_seeds: RefCell::new(Vec::new()),
        };
        let request = GenerationRequest {
            prompt: "cond".into(),
            negative_prompt: Some("uncond".into()),
            guidance: Some(4.5),
            seed: Some(u64::MAX - 1),
            count: 4,
            ..req(256, 256)
        };
        let expected_conditioning = b"conduncond";
        let expected = [u64::MAX - 1, u64::MAX, 0, 1]
            .map(|seed| fixture_image(expected_conditioning, seed))
            .to_vec();

        let actual = generate_base_images(&pipeline, &request, &Device::Cpu, &mut |_| {}).unwrap();

        assert_eq!(pipeline.encoder_calls.get(), 2);
        assert_eq!(
            *pipeline.rendered_seeds.borrow(),
            vec![u64::MAX - 1, u64::MAX, 0, 1]
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn base_adapter_without_cfg_encodes_only_cond_once() {
        let pipeline = BaseFixturePipeline {
            encoder_calls: Cell::new(0),
            rendered_seeds: RefCell::new(Vec::new()),
        };
        let request = GenerationRequest {
            guidance: Some(1.0),
            seed: Some(7),
            count: 3,
            ..req(256, 256)
        };

        let images = generate_base_images(&pipeline, &request, &Device::Cpu, &mut |_| {}).unwrap();

        assert_eq!(pipeline.encoder_calls.get(), 1);
        assert_eq!(*pipeline.rendered_seeds.borrow(), vec![7, 8, 9]);
        assert_eq!(images.len(), 3);
    }

    #[test]
    fn sprint_adapter_encodes_once_and_preserves_per_seed_tail() {
        let pipeline = SprintFixturePipeline {
            encoder_calls: Cell::new(0),
            rendered_seeds: RefCell::new(Vec::new()),
        };
        let request = GenerationRequest {
            prompt: "sprint cond".into(),
            seed: Some(11),
            count: 3,
            ..req(256, 256)
        };
        let expected = [11, 12, 13]
            .map(|seed| fixture_image(b"sprint cond", seed))
            .to_vec();

        let actual =
            generate_sprint_images(&pipeline, &request, &Device::Cpu, &mut |_| {}).unwrap();

        assert_eq!(pipeline.encoder_calls.get(), 1);
        assert_eq!(*pipeline.rendered_seeds.borrow(), vec![11, 12, 13]);
        assert_eq!(actual, expected);
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

    /// The seam under test: this provider's explicit family registry resolves our Candle generator.
    /// `load` is lazy, so a nonexistent weights dir still resolves (no file I/O until `generate`).
    #[test]
    fn registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
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
        assert!(d.capabilities.conditioning.is_empty());
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
    fn load_rejects_unwired_surfaces() {
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(matches!(
            load(&quant).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
        let control = LoadSpec::new(WeightsSource::Dir("/snap".into()))
            .with_control(WeightsSource::Dir("/ctrl".into()));
        assert!(matches!(
            load(&control).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
    }

    // =============================================================================================
    // SANA-Sprint (sc-11781) — the CFG-free SCM/TrigFlow few-step adapter.
    // =============================================================================================

    /// The Sprint seam under test: the second `register_generators!` submission resolves the EXACT id
    /// `"sana_sprint_1600m"` (the id the worker catalog 5b routes to) to OUR candle Sprint generator.
    /// `load_sprint` is lazy, so a nonexistent weights dir still resolves.
    #[test]
    fn sprint_registers_and_resolves_as_candle() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
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
        assert!(d.capabilities.conditioning.is_empty());
        assert_eq!(d.capabilities.samplers, vec!["default"]);
        assert_eq!(d.capabilities.schedulers, vec!["default"]);
        assert!(!d.capabilities.mac_only, "candle is Windows/CUDA");
    }

    #[test]
    fn sprint_defaults_match_diffusers() {
        assert_eq!(crate::pipeline::SPRINT_DEFAULT_STEPS, 2);
        assert!((crate::pipeline::SPRINT_DEFAULT_GUIDANCE - 4.5).abs() < 1e-6);
    }

    #[test]
    fn sprint_load_rejects_single_file_and_unwired_surfaces() {
        let file = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        let e = load_sprint(&file).err().expect("error").to_string();
        assert!(e.contains("snapshot directory"), "got: {e}");
        let quant = LoadSpec::new(WeightsSource::Dir("/snap".into())).with_quant(Quant::Q8);
        assert!(matches!(
            load_sprint(&quant).err().expect("err"),
            gen_core::Error::Unsupported(_)
        ));
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
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert_eq!(
            crate::provider_registry()
                .unwrap()
                .load(MODEL_ID, &spec)
                .unwrap()
                .descriptor()
                .id,
            "sana_1600m"
        );
        assert_eq!(
            crate::provider_registry()
                .unwrap()
                .load(SPRINT_MODEL_ID, &spec)
                .unwrap()
                .descriptor()
                .id,
            "sana_sprint_1600m"
        );
    }
}
