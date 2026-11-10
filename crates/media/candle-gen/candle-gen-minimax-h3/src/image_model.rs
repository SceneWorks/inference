//! The **`minimax_h3_image`** generator on candle/CUDA (SW-7091): MiniMax-H3-Image
//! text-to-image at bf16, prompt → one image.
//!
//! Its own engine id, [`IMAGE_MODEL_ID`], with an **image** descriptor — none of H3 video's
//! audio/video, frame, keyframe or reference surface, and none of its canvas caps
//! (`CANVAS_MAX_PIXELS`, `MAX_CANVAS_EDGE`): the request contract is `gen_core::h3_image`.
//!
//! # Phases
//!
//! The render is staged exactly as H3's is, for the same reason — the layer-50 Qwen3-VL-32B
//! conditioner (~58 GB at the tap) and the 66 GB DiT cannot be resident together on any card:
//!
//! 1. **conditioning** — tokenize the prompt verbatim (no chat template, no special tokens,
//!    truncated to 4096 tokens), run H3's text-only conditioner pass over the byte-identical
//!    `text_encoder/` (this snapshot's, or MiniMax-H3's staged as the `text_encoder` component) (`crate::model::run_conditioner`), drop it and synchronize;
//! 2. **denoise** — map `transformer_diffusers/` (no audio tensors), evict `adaln_proj` after
//!    projecting the six fixed timesteps' modulation ([`crate::dit::ImageDit`]), run the six-step
//!    velocity loop ([`crate::image_pipeline::denoise_image`]) with step `i` through head `i`,
//!    drop it and synchronize;
//! 3. **decode** — map `vae_diffusers/`'s decode half ([`MiniMaxH3ImageVae`]), decode with its
//!    shipped 256 / 64 tiling, and resize back from the generation canvas to the requested size.
//!
//! # Size
//!
//! Any positive size: the image is generated on `gen_core::h3_image::generation_canvas` (a
//! 32-multiple canvas at least `min_generation_area` = 4 MP by default) and resized back. A
//! `0 x 0` request is the upstream 2048 x 2048 default.

use std::path::{Path, PathBuf};

use candle_gen::candle_core::{DType, Device};
use candle_gen::gen_core::h3_image as contract;
use candle_gen::gen_core::{
    Capabilities, GenerationOutput, GenerationRequest, Generator, LoadShape, LoadSpec,
    MemoryProviderContract, Modality, ModelDescriptor, Precision, Progress, SizeFloor, StepSupport,
    WeightsSource,
};
use candle_gen::{CandleError, Result};
use gen_core_h3::h3_schedule::SigmaSchedule;

use crate::dit::{ImageDit, MiniMaxH3Dit};
use crate::image_pipeline::{
    decoded_to_image, denoise_image, initial_image_rows, latent_grid, unpack_image_rows,
    ImageLayout,
};
use crate::image_vae::{MiniMaxH3ImageVae, IMAGE_VAE_COMPONENT};
use crate::model::{require_component_shards, resolve_seed, run_conditioner, PhaseLog};
use crate::text_encoder::MiniMaxH3Tokenizer;

/// The engine id. A separate family from H3 video (`minimax_h3`): a different checkpoint, a
/// different modality and a different request contract.
pub const IMAGE_MODEL_ID: &str = "minimax_h3_image";

/// The DiT partition: the diffusers-converted `MiniMaxH3ImageTransformer2DModel`. The native
/// `transformer/` copy is never read.
pub const IMAGE_DIT_PARTITION: &str = "transformer_diffusers";

/// The conditioner partition — the same Qwen3-VL-32B shards MiniMax-H3 ships — and the
/// `LoadSpec::components` key that stages it from elsewhere (an H3 snapshot's `text_encoder/`),
/// the MLX twin's `TEXT_ENCODER_COMPONENT`.
pub const IMAGE_TEXT_ENCODER: &str = "text_encoder";

/// The conditioner directory a load reads: the staged [`IMAGE_TEXT_ENCODER`] component when it is
/// a directory, else `root/text_encoder`.
pub fn resolve_text_encoder_dir(root: &Path, spec: &LoadSpec) -> PathBuf {
    match spec.components.get(IMAGE_TEXT_ENCODER) {
        Some(WeightsSource::Dir(dir)) => dir.clone(),
        _ => root.join(IMAGE_TEXT_ENCODER),
    }
}

/// The upstream default output size when a request names none (`height = width = 2048`).
pub const DEFAULT_SIZE: u32 = 2048;

/// The largest edge the descriptor advertises. Upstream has **no** maximum; gen-core needs a
/// finite range, and the memory admission is the real bound, so this is a generous ceiling
/// rather than a published limit.
pub const MAX_EDGE: u32 = 16_384;

/// The identity, modality and capability surface.
pub fn image_descriptor() -> ModelDescriptor {
    ModelDescriptor {
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: IMAGE_MODEL_ID,
        family: "minimax_h3_image",
        backend: "candle",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Guidance-distilled: no negative prompt, no CFG (`gen_core::h3_image` refuses both).
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // Text-to-image only on this route; reference editing is a later item.
            conditioning: Vec::new(),
            // SW-7096: H3's adapter install over the image DiT's block/refiner surface —
            // diffusers/PEFT, kohya `lora_unet_`, ComfyUI and LoKr, each at its own scale.
            supports_lora: true,
            supports_lokr: true,
            supported_quants: &[],
            // The generator always stages its three components (see the module docs).
            supports_sequential_offload: true,
            max_count: 1,
            // PDD-distilled: one output head per step, so six is the only legal count.
            supported_steps: StepSupport::Exact(vec![contract::FIXED_STEPS]),
            component_precision_floors: &[],
            samplers: Vec::new(),
            schedulers: Vec::new(),
            min_size: 1,
            max_size: MAX_EDGE,
            // Any positive size: the canvas rounding is internal and resized back.
            size_floor: SizeFloor::RangeChecked,
            ..Default::default()
        },
    }
}

/// The loaded `minimax_h3_image` provider — **paths only**; every phase maps and releases its
/// own component.
pub struct MiniMaxH3Image {
    descriptor: ModelDescriptor,
    root: PathBuf,
    text_encoder_dir: PathBuf,
    device: Device,
    contract: MemoryProviderContract,
    spec: LoadSpec,
}

impl MiniMaxH3Image {
    /// Validate the snapshot and record the paths a render reads: `transformer_diffusers/`,
    /// `text_encoder/` and `vae_diffusers/` with shards, plus `tokenizer/`.
    pub fn load(spec: &LoadSpec) -> Result<Self> {
        let root = match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(p) => p
                .parent()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| p.clone()),
        };
        for component in [IMAGE_DIT_PARTITION, IMAGE_VAE_COMPONENT] {
            require_component_shards(&root.join(component), component)?;
        }
        let text_encoder_dir = resolve_text_encoder_dir(&root, spec);
        require_component_shards(&text_encoder_dir, IMAGE_TEXT_ENCODER)?;
        if !root.join("tokenizer").join("tokenizer.json").is_file() {
            return Err(CandleError::Msg(format!(
                "{IMAGE_MODEL_ID}: the snapshot has no tokenizer/tokenizer.json at {}",
                root.display()
            )));
        }
        // The partition must be the IMAGE transformer: H3 video's `transformer/` config would
        // load and run with an audio stream this pipeline never feeds.
        let cfg = image_dit_config(&root)?;
        if cfg.num_pdd_heads() != Some(contract::FIXED_STEPS as usize) {
            return Err(CandleError::Msg(format!(
                "{IMAGE_MODEL_ID}: `{IMAGE_DIT_PARTITION}` carries {:?} per-step heads; the fixed \
                 sampler runs {} steps",
                cfg.num_pdd_heads(),
                contract::FIXED_STEPS
            )));
        }
        Ok(Self {
            descriptor: image_descriptor(),
            root,
            text_encoder_dir,
            device: candle_gen::default_device()?,
            contract: crate::image_memory_strategy::contract_for(spec)
                .map_err(|e| CandleError::Msg(e.to_string()))?,
            spec: spec.clone(),
        })
    }

    /// The snapshot root a render reads from.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The requested output size: the upstream 2048 x 2048 default for `0 x 0`.
    fn output_size(req: &GenerationRequest) -> (u32, u32) {
        if req.width == 0 && req.height == 0 {
            (DEFAULT_SIZE, DEFAULT_SIZE)
        } else {
            (req.width, req.height)
        }
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        Generator::validate(self, req)?;
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let (width, height) = Self::output_size(req);
        let (canvas_w, canvas_h) =
            contract::generation_canvas(width, height, contract::MIN_GENERATION_AREA_DEFAULT);
        let (latent_h, latent_w) = latent_grid(canvas_w, canvas_h)?;
        let schedule =
            SigmaSchedule::h3_image_distilled().map_err(|e| CandleError::Msg(e.to_string()))?;
        let seed = resolve_seed(req);
        tracing::info!(
            width,
            height,
            canvas_w,
            canvas_h,
            seed,
            "minimax-h3-image: text-to-image request"
        );

        // --- 1. conditioning ---------------------------------------------------------------
        let phase = PhaseLog::start("text_encoder");
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let (ids, mask) =
            tok.encode_prompt_truncated(&req.prompt, contract::MAX_PROMPT_TOKENS, &self.device)?;
        let context = run_conditioner(
            &self.text_encoder_dir,
            &ids,
            &mask,
            &self.device,
            on_progress,
        )?;
        drop((ids, mask));
        phase.done();
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }

        // --- 2. denoise --------------------------------------------------------------------
        let phase = PhaseLog::start("dit_load");
        let mut dit = MiniMaxH3Dit::load_from_dir(
            self.root.join(IMAGE_DIT_PARTITION),
            &self.device,
            DType::BF16,
        )?;
        if !self.spec.adapters.is_empty() {
            // Strict, like H3 video: an unmatched target or a file that matched nothing fails the
            // render here rather than producing an image the LoRA barely touched.
            crate::adapters::apply_minimax_h3_adapters(&mut dit, &self.spec.adapters)?;
        }
        phase.done();
        let patch = dit.config().patch_size;
        let layout = ImageLayout::text_to_image(context.dims()[1], latent_h, latent_w, patch)?;
        tracing::info!(
            seq_len = layout.seq_len(),
            text = layout.num_text_tokens(),
            image = layout.num_image_rows(),
            "minimax-h3-image: packed DiT sequence"
        );
        let mut model = ImageDit::new(dit, &layout, &context, schedule.timesteps())?;
        model.set_cancel(&req.cancel);
        drop(context);
        let initial = initial_image_rows(latent_h, latent_w, patch, seed, &self.device)?;
        let total = schedule.num_evals() as u32;
        let phase = PhaseLog::start("denoise");
        let rows = denoise_image(
            &mut model,
            &schedule,
            &initial,
            &self.device,
            &req.cancel,
            &mut |completed| {
                on_progress(Progress::Step {
                    current: completed as u32,
                    total,
                })
            },
        )?;
        phase.done();
        drop(model);
        drop(initial);
        crate::dit::release_device_memory(&self.device)?;

        // --- 3. decode ---------------------------------------------------------------------
        on_progress(Progress::Decoding);
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        let phase = PhaseLog::start("image_vae_decode");
        let latents = unpack_image_rows(&rows, &layout)?;
        drop(rows);
        let vae = MiniMaxH3ImageVae::load(&self.root, &self.device, DType::F32)?;
        let decoded = vae.decode(&latents)?;
        drop(vae);
        drop(latents);
        crate::dit::release_device_memory(&self.device)?;
        let image = decoded_to_image(&decoded, width, height)?;
        phase.done();
        Ok(GenerationOutput::Images(vec![image]))
    }
}

/// Parse `transformer_diffusers/config.json` and require it to be the image transformer.
pub fn image_dit_config(root: &Path) -> Result<crate::dit::MiniMaxH3DitConfig> {
    let path = root.join(IMAGE_DIT_PARTITION).join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| {
        CandleError::Msg(format!("{IMAGE_MODEL_ID}: reading {}: {e}", path.display()))
    })?;
    crate::dit::MiniMaxH3DitConfig::from_image_diffusers_json(&text)
}

impl Generator for MiniMaxH3Image {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> candle_gen::gen_core::Result<()> {
        validate_image_request(&self.descriptor, req)
    }

    fn memory_strategy_contract(&self) -> Option<&MemoryProviderContract> {
        Some(&self.contract)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &candle_gen::gen_core::MemoryRunContext,
    ) -> candle_gen::gen_core::MemorySafetyDecision {
        crate::image_memory_strategy::safety_check(&self.spec, &self.contract, context)
    }

    fn begin_memory_strategy_request(
        &self,
        context: &candle_gen::gen_core::MemoryRunContext,
    ) -> candle_gen::gen_core::Result<Option<Box<dyn candle_gen::gen_core::MemoryRequestScope>>>
    {
        crate::image_memory_strategy::registered_begin_request(&self.spec, &self.contract, context)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> candle_gen::gen_core::Result<GenerationOutput> {
        Ok(self.generate_impl(req, on_progress)?)
    }
}

/// The request floor plus the H3-Image contract: a non-empty prompt, any positive size (or the
/// `0 x 0` default), exactly six steps, no guidance other than 1, no negative prompt, no
/// references.
pub fn validate_image_request(
    descriptor: &ModelDescriptor,
    req: &GenerationRequest,
) -> candle_gen::gen_core::Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(candle_gen::gen_core::Error::Msg(format!(
            "{IMAGE_MODEL_ID}: prompt must not be empty"
        )));
    }
    let (width, height) = MiniMaxH3Image::output_size(req);
    contract::validate_request(
        width,
        height,
        None,
        req.steps.unwrap_or(contract::FIXED_STEPS),
        req.guidance,
        req.negative_prompt
            .as_deref()
            .is_some_and(|n| !n.trim().is_empty()),
        req.conditioning.len(),
    )?;
    let mut sized = req.clone();
    sized.width = width;
    sized.height = height;
    // `guidance = 1.0` is the contract's "no CFG" value and already passed it; the shared floor
    // refuses ANY guidance on a model without CFG, so the admitted 1.0 is not handed to it.
    sized.guidance = None;
    descriptor
        .capabilities
        .validate_request(IMAGE_MODEL_ID, &sized)
}

/// Registry entry point: refuse every `LoadSpec` slot this provider does not read, then load.
pub fn load_image(spec: &LoadSpec) -> candle_gen::gen_core::Result<Box<dyn Generator>> {
    if let Some(q) = spec.quantize {
        return Err(candle_gen::gen_core::Error::Unsupported(format!(
            "{IMAGE_MODEL_ID}: spec.quantize={q:?} — no quantized tier is published for this \
             family yet; the render runs the checkpoint's bf16 block store"
        )));
    }
    if spec.load_shape == LoadShape::DeferredMaterialization {
        return Err(candle_gen::gen_core::Error::Unsupported(format!(
            "{IMAGE_MODEL_ID}: block-streamed (deferred) materialization is not wired for this \
             family; use the resident load"
        )));
    }
    candle_gen::gen_core::reject_unknown_components(spec, &[IMAGE_TEXT_ENCODER], IMAGE_MODEL_ID)?;
    crate::model::reject_unread_slots_for(spec, IMAGE_MODEL_ID)?;
    debug_assert_eq!(spec.precision, Precision::Bf16);
    Ok(Box::new(MiniMaxH3Image::load(spec)?))
}

candle_gen::register_generators! {
    pub const IMAGE_REGISTRATION = image_descriptor => load_image
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::gen_core::Conditioning;

    fn request(width: u32, height: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a lighthouse at dusk".into(),
            width,
            height,
            ..Default::default()
        }
    }

    /// The descriptor is its own image family — not H3 video's id, modality or flags.
    #[test]
    fn the_descriptor_is_an_image_family_of_its_own() {
        let d = image_descriptor();
        assert_eq!(d.id, IMAGE_MODEL_ID);
        assert_ne!(d.id, crate::MODEL_ID);
        assert_eq!(d.modality, Modality::Image);
        assert_eq!(d.backend, "candle");
        let c = &d.capabilities;
        assert!(!c.supports_negative_prompt && !c.supports_guidance && !c.supports_true_cfg);
        assert!(c.conditioning.is_empty());
        assert_eq!(c.supported_steps, StepSupport::Exact(vec![6]));
        assert_eq!(c.max_count, 1);
        // Not H3 video's 1.03 MP / 2016 caps.
        assert!(c.max_size > crate::MAX_CANVAS_EDGE);
        assert_eq!(c.min_size, 1);
        assert!(candle_gen::gen_core::registry::model_descriptor_errors(&d).is_empty());
    }

    /// The H3-Image contract at the engine boundary: 2048² and odd sizes pass, steps other than
    /// six, guidance other than 1, a negative prompt and an empty prompt are refused.
    #[test]
    fn validate_enforces_the_h3_image_contract() {
        let d = image_descriptor();
        validate_image_request(&d, &request(2048, 2048)).unwrap();
        validate_image_request(&d, &request(0, 0)).unwrap();
        validate_image_request(&d, &request(1000, 1001)).unwrap();
        let mut r = request(2048, 2048);
        r.steps = Some(6);
        r.guidance = Some(1.0);
        validate_image_request(&d, &r).unwrap();
        r.steps = Some(8);
        assert!(validate_image_request(&d, &r).is_err());
        let mut r = request(2048, 2048);
        r.guidance = Some(4.0);
        assert!(validate_image_request(&d, &r).is_err());
        let mut r = request(2048, 2048);
        r.negative_prompt = Some("blurry".into());
        assert!(validate_image_request(&d, &r).is_err());
        let mut r = request(2048, 2048);
        r.prompt = "  ".into();
        assert!(validate_image_request(&d, &r).is_err());
        assert!(validate_image_request(&d, &request(0, 2048)).is_err());
        let mut r = request(2048, 2048);
        r.conditioning.push(Conditioning::Reference {
            image: candle_gen::gen_core::Image {
                width: 1,
                height: 1,
                pixels: vec![0; 3],
            },
            strength: None,
        });
        assert!(validate_image_request(&d, &r).is_err());
    }

    /// A 2048² request generates on a 2048² canvas (128² latent); a 1024² request generates at
    /// 4 MP and is resized back.
    #[test]
    fn the_canvas_comes_from_the_h3_image_contract() {
        let (w, h) = MiniMaxH3Image::output_size(&request(0, 0));
        assert_eq!((w, h), (2048, 2048));
        let canvas = contract::generation_canvas(w, h, contract::MIN_GENERATION_AREA_DEFAULT);
        assert_eq!(latent_grid(canvas.0, canvas.1).unwrap(), (128, 128));
        let canvas = contract::generation_canvas(1024, 1024, contract::MIN_GENERATION_AREA_DEFAULT);
        assert_eq!(canvas, (2048, 2048));
    }

    /// Every slot the provider does not read is refused before any path is touched.
    #[test]
    fn unread_load_slots_are_refused() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-h3-image".into()));
        let mut q = spec.clone();
        q.quantize = Some(candle_gen::gen_core::Quant::Q4);
        assert!(matches!(
            load_image(&q),
            Err(candle_gen::gen_core::Error::Unsupported(_))
        ));
        // SW-7096: adapters are a read slot now — an adapter spec over a missing snapshot fails
        // at the component check, not as an unsupported slot.
        let mut a = spec.clone();
        a.adapters.push(candle_gen::gen_core::AdapterSpec::new(
            "/x.safetensors".into(),
            1.0,
            candle_gen::gen_core::AdapterKind::Lora,
        ));
        let e = load_image(&a).err().expect("missing snapshot");
        assert!(e.to_string().contains(IMAGE_DIT_PARTITION), "{e}");
        let mut d = spec.clone();
        d.load_shape = LoadShape::DeferredMaterialization;
        assert!(load_image(&d).is_err());
        // A plain spec over a missing snapshot fails at the component check, naming the path.
        let e = load_image(&spec).err().expect("missing snapshot");
        assert!(e.to_string().contains(IMAGE_DIT_PARTITION), "{e}");
    }
}
