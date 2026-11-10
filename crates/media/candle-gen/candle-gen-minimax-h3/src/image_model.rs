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

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::h3_image as contract;
use candle_gen::gen_core::{
    Capabilities, ConditioningKind, GenerationOutput, GenerationRequest, Generator, LoadShape,
    LoadSpec, MemoryProviderContract, Modality, ModelDescriptor, Precision, Progress, Quant,
    SizeFloor, StepSupport, WeightsSource,
};
use candle_gen::{CandleError, Result};
use gen_core_h3::h3_schedule::SigmaSchedule;

use crate::dit::{ImageDit, MiniMaxH3Dit};
use crate::image_pipeline::{
    decoded_to_image, denoise_image, initial_image_rows, latent_grid, reference_condition_rows,
    reference_noise, reference_vae_pixels, resize_reference, unpack_image_rows, ImageLayout,
};
use crate::image_vae::{MiniMaxH3ImageVae, MiniMaxH3ImageVaeEncoder, IMAGE_VAE_COMPONENT};
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

/// The DiT directory a load reads: the staged [`IMAGE_DIT_PARTITION`] component (a packed tier's
/// `transformer_diffusers/`, SW-7093) when it is a directory, else `root/transformer_diffusers`.
pub fn resolve_dit_dir(root: &Path, spec: &LoadSpec) -> PathBuf {
    match spec.components.get(IMAGE_DIT_PARTITION) {
        Some(WeightsSource::Dir(dir)) => dir.clone(),
        _ => root.join(IMAGE_DIT_PARTITION),
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
            // `edit` (SW-7092): 1..=16 reference images, as single references and/or a
            // multi-reference list, in request order. No reference is `text2image`.
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
            ],
            // SW-7096: H3's adapter install over the image DiT's block/refiner surface --
            // diffusers/PEFT, kohya `lora_unet_`, ComfyUI and LoKr, each at its own scale.
            supports_lora: true,
            supports_lokr: true,
            // Pre-quantized MLX-affine tiers (SW-7093), the artifacts the MLX lane reads: staged as
            // the DiT component; `spec.quantize` asserts which tier was staged.
            supported_quants: &[Quant::Q4, Quant::Q8],
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
    dit_dir: PathBuf,
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
        let dit_dir = resolve_dit_dir(&root, spec);
        require_component_shards(&dit_dir, IMAGE_DIT_PARTITION)?;
        require_component_shards(&root.join(IMAGE_VAE_COMPONENT), IMAGE_VAE_COMPONENT)?;
        let text_encoder_dir = resolve_text_encoder_dir(&root, spec);
        require_component_shards(&text_encoder_dir, IMAGE_TEXT_ENCODER)?;
        // The tier is an assertion about what is staged (the marker is authoritative); `None`
        // asserts nothing and the loaders auto-detect from `.scales`. The group size is checked
        // either way, and for the text encoder -- MiniMax-H3's own tier artifact -- by H3's check.
        let tier =
            crate::model::requested_tier(spec).map_err(|e| CandleError::Msg(e.to_string()))?;
        crate::tier::require_partition(&dit_dir, tier, IMAGE_DIT_PARTITION)?;
        crate::tier::MiniMaxH3TierPaths::resolve(&root, None, Some(&text_encoder_dir))
            .require_text_encoder()?;
        if !root.join("tokenizer").join("tokenizer.json").is_file() {
            return Err(CandleError::Msg(format!(
                "{IMAGE_MODEL_ID}: the snapshot has no tokenizer/tokenizer.json at {}",
                root.display()
            )));
        }
        // The partition must be the IMAGE transformer: H3 video's `transformer/` config would
        // load and run with an audio stream this pipeline never feeds.
        let cfg = image_dit_config_at(&dit_dir)?;
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
            dit_dir,
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

    /// The `edit` conditioning (SW-7092): H3's Qwen3-VL vision tower over every vision-resized
    /// reference (a 1024 short edge), spliced into the `Picture i: ` + vision-block presentation
    /// and run through the layer-50 conditioner; then every VAE-resized reference (a
    /// `[1024, 2048]` short edge) encoded on its own latent grid, its posterior sampled with
    /// [`reference_noise`] stream 0. Each component is mapped, used, dropped and released in turn.
    /// Returns the context, its per-row tags and the reference latents `[1, 24, h, w]`.
    fn encode_edit(
        &self,
        prompt: &str,
        references: &[&candle_gen::gen_core::Image],
        seed: u64,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<(Tensor, Vec<u32>, Vec<Tensor>)> {
        let vision_images = references
            .iter()
            .map(|r| resize_reference(r, contract::reference_vision_size(r.width, r.height)))
            .collect::<Result<Vec<_>>>()?;
        let phase = PhaseLog::start("vision_tower");
        let vision = crate::text_encoder::load_vision_tower(&self.text_encoder_dir, &self.device)?;
        let refs: Vec<&candle_gen::gen_core::Image> = vision_images.iter().collect();
        let grounded = crate::text_encoder::run_vision(&vision, &refs, &self.device)?;
        drop(vision);
        drop(vision_images);
        crate::dit::release_device_memory(&self.device)?;
        phase.done();

        let phase = PhaseLog::start("text_encoder");
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let (ids, mask, tags) = tok.encode_image_edit(prompt, &grounded.counts, &self.device)?;
        let context = run_conditioner(
            &self.text_encoder_dir,
            &ids,
            &mask,
            Some(&grounded),
            &self.device,
            on_progress,
        )?;
        drop((ids, mask, grounded));
        phase.done();
        if context.dims()[1] != tags.len() {
            return Err(CandleError::Msg(format!(
                "{IMAGE_MODEL_ID}: the edit context has {} rows but {} modality tags",
                context.dims()[1],
                tags.len()
            )));
        }

        let phase = PhaseLog::start("image_vae_encode");
        let patch = image_dit_config(&self.root)?.patch_size;
        let encoder = MiniMaxH3ImageVaeEncoder::load(&self.root, &self.device)?;
        let mut latents = Vec::with_capacity(references.len());
        for (i, r) in references.iter().enumerate() {
            let image = resize_reference(r, contract::reference_vae_size(r.width, r.height))?;
            let posterior = encoder.encode(&reference_vae_pixels(&image, &self.device)?)?;
            let noise = reference_noise(seed, 0, i, posterior.mean().dims(), &self.device)?;
            latents.push(encoder.reference_latent(&posterior, &noise, (patch[1], patch[2]))?);
        }
        drop(encoder);
        crate::dit::release_device_memory(&self.device)?;
        phase.done();
        Ok((context, tags, latents))
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

        let references = contract::edit_reference_images(&req.conditioning)
            .map_err(|r| CandleError::Msg(r.to_string()))?;

        // --- 1. conditioning ---------------------------------------------------------------
        let (context, text_tags, reference_latents) = if references.is_empty() {
            let phase = PhaseLog::start("text_encoder");
            let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
            let (ids, mask) = tok.encode_prompt_truncated(
                &req.prompt,
                contract::MAX_PROMPT_TOKENS,
                &self.device,
            )?;
            let context = run_conditioner(
                &self.text_encoder_dir,
                &ids,
                &mask,
                None,
                &self.device,
                on_progress,
            )?;
            drop((ids, mask));
            phase.done();
            let n = context.dims()[1];
            (context, vec![contract::TEXT_TAG as u32; n], Vec::new())
        } else {
            self.encode_edit(&req.prompt, &references, seed, on_progress)?
        };
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }

        // --- 2. denoise --------------------------------------------------------------------
        let phase = PhaseLog::start("dit_load");
        let mut dit = MiniMaxH3Dit::load_from_dir(&self.dit_dir, &self.device, DType::BF16)?;
        if !self.spec.adapters.is_empty() {
            // Strict, like H3 video: an unmatched target or a file that matched nothing fails the
            // render here rather than producing an image the LoRA barely touched.
            crate::adapters::apply_minimax_h3_adapters(&mut dit, &self.spec.adapters)?;
        }
        phase.done();
        let patch = dit.config().patch_size;
        let layout = if reference_latents.is_empty() {
            ImageLayout::text_to_image(context.dims()[1], latent_h, latent_w, patch)?
        } else {
            let grids = reference_latents
                .iter()
                .map(|l| {
                    let d = l.dims();
                    (d[2], d[3])
                })
                .collect::<Vec<_>>();
            ImageLayout::edit(&text_tags, &grids, latent_h, latent_w, patch)?
        };
        tracing::info!(
            seq_len = layout.seq_len(),
            text = layout.num_text_tokens(),
            image = layout.num_image_rows(),
            "minimax-h3-image: packed DiT sequence"
        );
        let mut model = ImageDit::new(dit, &layout, &context, schedule.timesteps())?;
        model.set_cancel(&req.cancel);
        drop(context);
        let noise = initial_image_rows(latent_h, latent_w, patch, seed, &self.device)?;
        // An edit's reference rows lead the packed image stream, noised to 0.999 with one draw per
        // reference after the generated image's own.
        let initial = if reference_latents.is_empty() {
            noise
        } else {
            let aug = reference_latents
                .iter()
                .enumerate()
                .map(|(i, l)| reference_noise(seed, 1, i, l.dims(), &self.device))
                .collect::<Result<Vec<_>>>()?;
            let condition = reference_condition_rows(&reference_latents, &aug, patch)?;
            Tensor::cat(&[&condition, &noise], 1)?
        };
        drop(reference_latents);
        let total = schedule.num_evals() as u32;
        let phase = PhaseLog::start("denoise");
        let rows = denoise_image(
            &mut model,
            &schedule,
            &initial,
            layout.num_reference_rows(),
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

/// Parse `transformer_diffusers/config.json` of a snapshot root and require it to be the image
/// transformer.
pub fn image_dit_config(root: &Path) -> Result<crate::dit::MiniMaxH3DitConfig> {
    image_dit_config_at(&root.join(IMAGE_DIT_PARTITION))
}

/// [`image_dit_config`] against an already-resolved (possibly staged tier) DiT directory.
pub fn image_dit_config_at(dit_dir: &Path) -> Result<crate::dit::MiniMaxH3DitConfig> {
    let path = dit_dir.join("config.json");
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
/// `0 x 0` default), exactly six steps, no guidance other than 1, no negative prompt, and 0..=16
/// references of at least 32 px a side with no per-reference strength.
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
        contract::edit_reference_images(&req.conditioning)?.len(),
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
    // Q4 / Q8 are the published MLX-affine tiers; NVFP4 and FP8 are different formats and are
    // refused by name rather than reconciled against a q4/q8 marker.
    crate::model::requested_tier(spec)?;
    if spec.load_shape == LoadShape::DeferredMaterialization {
        return Err(candle_gen::gen_core::Error::Unsupported(format!(
            "{IMAGE_MODEL_ID}: block-streamed (deferred) materialization is not wired for this \
             family; use the resident load"
        )));
    }
    candle_gen::gen_core::reject_unknown_components(
        spec,
        &[IMAGE_TEXT_ENCODER, IMAGE_DIT_PARTITION],
        IMAGE_MODEL_ID,
    )?;
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
        assert_eq!(
            c.conditioning,
            vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference
            ]
        );
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
        // NVFP4 is a different format from the published MLX-affine tiers.
        let mut q = spec.clone();
        q.quantize = Some(candle_gen::gen_core::Quant::Nvfp4);
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

    fn with_refs(n: usize, w: u32, h: u32) -> GenerationRequest {
        let mut r = request(1024, 1024);
        r.conditioning.push(Conditioning::MultiReference {
            images: (0..n)
                .map(|_| candle_gen::gen_core::Image {
                    width: w,
                    height: h,
                    pixels: vec![0; (w * h * 3) as usize],
                })
                .collect(),
        });
        r
    }

    /// `edit` admits 1..=16 references of at least 32 px a side and refuses 17 or a 31 px one.
    #[test]
    fn edit_admits_one_to_sixteen_references_of_at_least_32_px() {
        let d = image_descriptor();
        validate_image_request(&d, &with_refs(1, 32, 32)).unwrap();
        validate_image_request(&d, &with_refs(16, 64, 40)).unwrap();
        let e = validate_image_request(&d, &with_refs(17, 64, 64))
            .unwrap_err()
            .to_string();
        assert!(e.contains("17"), "{e}");
        let e = validate_image_request(&d, &with_refs(2, 31, 64))
            .unwrap_err()
            .to_string();
        assert!(e.contains("32"), "{e}");
    }

    /// SW-7093: Q4 and Q8 are advertised, and the staged `transformer_diffusers` component -- not the
    /// snapshot root -- is the DiT a tier load reads.
    #[test]
    fn the_packed_tiers_are_advertised_and_staged_as_the_dit_component() {
        assert_eq!(
            image_descriptor().capabilities.supported_quants,
            &[Quant::Q4, Quant::Q8]
        );
        let mut spec = LoadSpec::new(WeightsSource::Dir("/snap".into()));
        assert_eq!(
            resolve_dit_dir(Path::new("/snap"), &spec),
            Path::new("/snap/transformer_diffusers")
        );
        spec.components.insert(
            IMAGE_DIT_PARTITION.into(),
            WeightsSource::Dir("/tiers/q4/transformer_diffusers".into()),
        );
        assert_eq!(
            resolve_dit_dir(Path::new("/snap"), &spec),
            Path::new("/tiers/q4/transformer_diffusers")
        );
        // The staged key is accepted; any other component is still refused.
        let mut other = spec.clone();
        other
            .components
            .insert("transformer".into(), WeightsSource::Dir("/x".into()));
        assert!(load_image(&other).is_err());
    }

    /// The on-disk `quantization` marker is authoritative: a q4 DiT staged under a Q8 request is
    /// refused before any tensor is read, and the text encoder resolves to the staged H3 artifact.
    #[test]
    fn a_staged_tier_must_agree_with_the_request() {
        let tmp = tempfile::tempdir().unwrap();
        let comp = |dir: &Path, config: &str| {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(dir.join("config.json"), config).unwrap();
            std::fs::write(dir.join("m.safetensors"), b"").unwrap();
        };
        let root = tmp.path().join("root");
        comp(&root.join(IMAGE_VAE_COMPONENT), "{}");
        let dit = tmp.path().join("q4").join(IMAGE_DIT_PARTITION);
        comp(&dit, r#"{"quantization": {"bits": 4, "group_size": 64}}"#);
        let te = tmp.path().join("q4").join(IMAGE_TEXT_ENCODER);
        comp(&te, r#"{"quantization": {"bits": 4, "group_size": 64}}"#);
        let mut spec = LoadSpec::new(WeightsSource::Dir(root));
        spec.components
            .insert(IMAGE_DIT_PARTITION.into(), WeightsSource::Dir(dit));
        spec.components
            .insert(IMAGE_TEXT_ENCODER.into(), WeightsSource::Dir(te.clone()));
        assert_eq!(resolve_text_encoder_dir(Path::new("/unused"), &spec), te);
        spec.quantize = Some(Quant::Q8);
        let e = load_image(&spec).err().expect("q4 staged, q8 asked");
        assert!(e.to_string().contains("bits 4"), "{e}");
    }
}
