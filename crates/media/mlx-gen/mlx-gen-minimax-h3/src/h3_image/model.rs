//! The **`minimax_h3_image` generator**: prompt (and up to 16 reference images) → image,
//! `MiniMaxAI/MiniMax-H3-Image` on MLX.
//!
//! # Tasks
//!
//! * **`text2image`** — no reference: the phases below.
//! * **`edit`** — 1 to 16 `Conditioning::Reference` / `Conditioning::MultiReference` images,
//!   in request order (`<Picture 1>` is the first). Each reference takes two paths, as upstream:
//!   resized to a 1024 short edge, it goes through H3's Qwen3-VL vision tower and is spliced into the
//!   `Picture i: ` + vision-block presentation ([`MiniMaxH3Tokenizer::image_edit_ids`]);
//!   resized into a `[1024, 2048]` short edge, it is VAE-encoded on its own latent grid
//!   ([`MiniMaxH3ImageVaeEncoder`]), noised to `t = 0.999` and packed in front of the generated
//!   rows ([`ImageLayout::edit`]), held there for every step. The output size is the request's.
//!
//! # Phases (staged; nothing is co-resident)
//!
//! 1. **Conditioning.** The prompt is tokenized verbatim — no chat template, no special tokens —
//!    and truncated to 4096 tokens ([`mlx_gen::gen_core::h3_image::t2i_prompt_ids`]); H3's Qwen3-VL-32B text
//!    encoder runs over it and the layer-50 hidden state is the context. The text encoder is
//!    byte-identical to `MiniMaxAI/MiniMax-H3`'s, so it is read from the staged
//!    [`TEXT_ENCODER_COMPONENT`] (an H3 snapshot's `text_encoder/`) when one is given, else from the
//!    snapshot's own copy. Forced, then released.
//! 2. **Denoise.** `transformer_diffusers/` ([`MiniMaxH3ImageDit`]) over the `[text | image]`
//!    sequence, six fixed steps on [`SigmaSchedule::h3_image_distilled`], step `i` through head `i`,
//!    velocity Euler. Released.
//!
//! Under rung 4 (a [`mlx_gen::gen_core::LoadShape::DeferredMaterialization`] load and a request
//! whose memory block streams, SW-7094) phases 1 and 2 run H3's deferred text encoder and
//! [`MiniMaxH3ImageDit::load_dir_deferred`] one layer / block at a time
//! ([`MiniMaxH3Image::requested_transformer_window`]).
//!
//! 3. **Decode.** `vae_diffusers/` ([`MiniMaxH3ImageVae`]), tiled at the shipped 256 / 64; pixel
//!    normalization reverted; 8-bit by truncation (decoders.py); resized back to the requested size
//!    with Lanczos when the canvas differs.
//!
//! # Size
//!
//! Any positive size. The generation canvas is
//! [`mlx_gen::gen_core::h3_image::generation_canvas`] at upstream's default 4 MP
//! minimum area: a smaller request is generated larger at the same aspect and resized back. None
//! of H3 video's caps (1.03 MP canvas, 2016 px edge, 17n+5 frames) apply.

use std::path::{Path, PathBuf};

use mlx_rs::ops::{add, clip, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::gen_core::h3_image::{
    edit_reference_images, latent_grid, reference_vae_size, reference_vision_size, request_sampler,
    t2i_prompt_ids, validate_request as validate_h3_image, FIXED_GUIDANCE, FIXED_STEPS,
    REFERENCE_NOISE_AUG, VAE_SPATIAL_RATIO,
};
use mlx_gen::gen_core::{
    reject_unknown_components, Capabilities, ConditioningKind, GenerationOutput, GenerationRequest,
    Generator, LoadSpec, Modality, ModelDescriptor, Precision, Progress, Quant, StepSupport,
    WeightsSource,
};
use mlx_gen::{default_seed, Error, Result};

use crate::config::LATENT_CHANNELS;
use crate::denoise::schedule::SigmaSchedule;
use crate::h3_image::dit::{ImageDenoiser, MiniMaxH3ImageDit, IMAGE_DIT_COMPONENT};
use crate::h3_image::layout::ImageLayout;
use crate::image_vae::{MiniMaxH3ImageVae, MiniMaxH3ImageVaeEncoder, IMAGE_VAE_COMPONENT};
use crate::model::TEXT_ENCODER_COMPONENT;
use crate::pipeline::{patchify_video_latents, unpatchify_video_rows, PIXEL_MEAN, PIXEL_STD};
use crate::spatial_tiling::SpatialTiling;
use crate::text_encoder::{
    MiniMaxH3TeConfig, MiniMaxH3TextEncoder, MiniMaxH3Tokenizer, LM_PREFIX, VISION_PREFIX,
};

/// The published provider id.
pub const IMAGE_MODEL_ID: &str = "minimax_h3_image";

/// The descriptor family (the latent space and decoder are this family's own).
pub const FAMILY: &str = "minimax_h3_image";

/// Images one request may ask for. Each is its own packed sequence (upstream packs one request
/// into one sequence), so a count renders sequentially over one encoded prompt and one loaded DiT.
pub const MAX_COUNT: u32 = 8;

/// The loaded generator — paths and precision, no tensors (every phase maps and releases its own).
pub struct MiniMaxH3Image {
    descriptor: ModelDescriptor,
    root: PathBuf,
    dit_dir: PathBuf,
    text_encoder_dir: PathBuf,
    dtype: Dtype,
    memory_strategy: mlx_gen::gen_core::MemoryProviderContract,
    memory_tier: mlx_gen::gen_core::MemoryNumericTier,
    /// LoRA / LoKr adapters (SW-7096), installed on the DiT each time it is mapped for a render
    /// and released with it.
    adapters: Vec<mlx_gen::gen_core::AdapterSpec>,
    /// The load shape this generator was loaded at — the one field rung 4's declaration and its
    /// runtime admission ([`MiniMaxH3Image::requested_transformer_window`]) both read, so the two
    /// cannot disagree about whether the rung exists (SW-7094).
    load_shape: mlx_gen::gen_core::LoadShape,
}

/// The identity, modality and capability surface.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        // The text encoder is not advertised as substitutable, and the denoiser's 24-channel
        // latent is decoded only by this family's own 2-D VAE: `None` fails every external
        // decoder swap closed.
        encoder_contract: None,
        denoiser_output_latent_space: None,
        control_kinds: None,
        required_components: &[],
        id: IMAGE_MODEL_ID,
        family: FAMILY,
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Guidance-distilled: no unconditional branch. Guidance 1.0 is accepted as the
            // documented no-op (`validate` clears it before the shared floor).
            supports_negative_prompt: false,
            supports_guidance: false,
            // `edit`: 1..=16 reference images, as single references and/or one multi-reference
            // list, in request order. No reference is `text2image`.
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
            ],
            // SW-7096: H3's adapter install over the image DiT's block/refiner surface --
            // diffusers/PEFT, kohya `lora_unet_`, ComfyUI and LoKr, each at its own scale.
            supports_lora: true,
            supports_lokr: true,
            // Pre-quantized tiers only (SW-7093): packed offline by
            // `crate::convert::quantize_minimax_h3_image_dit`, staged as the DiT component.
            supported_quants: &[Quant::Q4, Quant::Q8],
            // Upstream requires only a positive size: no maximum area or edge.
            min_size: 1,
            max_size: u32::MAX,
            max_count: MAX_COUNT,
            supported_steps: StepSupport::Exact(vec![FIXED_STEPS]),
            mac_only: false,
            // TE -> DiT -> VAE are built, used and released in order in `generate_impl`, so every
            // selection stages whatever residency it names.
            unconditionally_engages_staged_residency: true,
            max_prompt_tokens: Some(mlx_gen::gen_core::h3_image::MAX_PROMPT_TOKENS as u32),
            // SW-7106: ComfyUI's stochastic LCM (the default, 1.1 → 0.9) or the released Euler on
            // the same six-step grid, named by `sampler` and parameterized by `h3_image_sampler`.
            samplers: mlx_gen::gen_core::h3_image::SAMPLER_NAMES.to_vec(),
            supports_h3_image_sampler: true,
            ..Default::default()
        },
    }
}

/// Resolve the text-encoder directory: the staged [`TEXT_ENCODER_COMPONENT`] (an H3 snapshot's
/// `text_encoder/`, byte-identical to this repository's) or `root/text_encoder`.
fn resolve_text_encoder_dir(root: &Path, spec: &LoadSpec) -> PathBuf {
    match spec.components.get(TEXT_ENCODER_COMPONENT) {
        Some(WeightsSource::Dir(p)) => p.clone(),
        _ => root.join(TEXT_ENCODER_COMPONENT),
    }
}

/// Load the generator from a `MiniMaxAI/MiniMax-H3-Image` snapshot root. Probes every file the
/// render reads and reads no tensor.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(load_image(spec)?))
}

/// [`load`] without the box — the trainer (SW-7097) encodes its dataset and renders its previews
/// through the generator's own phases.
pub(crate) fn load_image(spec: &LoadSpec) -> Result<MiniMaxH3Image> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{IMAGE_MODEL_ID}: expected a MiniMax-H3-Image snapshot directory (the dir holding \
                 `{IMAGE_DIT_COMPONENT}/`), not a single file"
            )))
        }
    };
    reject_unknown_components(
        spec,
        &[TEXT_ENCODER_COMPONENT, IMAGE_DIT_COMPONENT],
        IMAGE_MODEL_ID,
    )?;
    let text_encoder_dir = resolve_text_encoder_dir(&root, spec);
    let dit_dir = crate::h3_image::tier::resolve_dit_dir(&root, spec);
    let mut probes: Vec<PathBuf> = vec![
        dit_dir.join("config.json"),
        root.join(IMAGE_VAE_COMPONENT).join("config.json"),
        root.join("tokenizer").join("tokenizer.json"),
        root.join("tokenizer").join("tokenizer_config.json"),
    ];
    probes.extend(
        crate::text_encoder::TE_SHARDS
            .map(|i| text_encoder_dir.join(format!("model-{i:05}-of-00014.safetensors"))),
    );
    for path in probes {
        if !path.is_file() {
            return Err(Error::Msg(format!(
                "{IMAGE_MODEL_ID}: missing {} — point at a MiniMax-H3-Image snapshot root holding \
                 `{IMAGE_DIT_COMPONENT}/`, `{IMAGE_VAE_COMPONENT}/` and `tokenizer/`, and stage \
                 MiniMax-H3's `text_encoder/` as the '{TEXT_ENCODER_COMPONENT}' component when \
                 this snapshot does not carry it",
                path.display()
            )));
        }
    }
    crate::h3_image::tier::reconcile(&dit_dir, &text_encoder_dir, spec)?;
    let dtype = match spec.precision {
        Precision::Fp32 => Dtype::Float32,
        _ => Dtype::Bfloat16,
    };
    let memory_strategy = crate::h3_image::memory::contract_for(spec).map_err(Error::from)?;
    Ok(MiniMaxH3Image {
        descriptor: descriptor(),
        root,
        dit_dir,
        text_encoder_dir,
        dtype,
        memory_strategy,
        memory_tier: mlx_gen::request_scope::loaded_tier(spec),
        adapters: spec.adapters.clone(),
        load_shape: spec.load_shape,
    })
}

/// Reject a request this model cannot serve, before any weight is read: the family contract
/// (gen-core's `h3_image` refusals) first, then the shared floor.
pub fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(Error::Msg(format!(
            "{IMAGE_MODEL_ID}: prompt must not be empty"
        )));
    }
    let references = edit_reference_images(&req.conditioning)
        .map_err(|r| Error::from(mlx_gen::gen_core::Error::from(r)))?;
    validate_h3_image(
        req.width,
        req.height,
        None,
        req.steps.unwrap_or(FIXED_STEPS),
        req.guidance,
        req.negative_prompt
            .as_deref()
            .is_some_and(|n| !n.trim().is_empty()),
        references.len(),
    )
    .map_err(|r| Error::from(mlx_gen::gen_core::Error::from(r)))?;
    request_sampler(req).map_err(|r| Error::from(mlx_gen::gen_core::Error::from(r)))?;
    // Guidance 1.0 is the family's documented no-op, admitted above; the shared floor refuses any
    // guidance on a model without CFG, so it sees the request without it.
    if req.guidance == Some(FIXED_GUIDANCE) {
        let mut floor = req.clone();
        floor.guidance = None;
        caps.validate_request(IMAGE_MODEL_ID, &floor)?;
    } else {
        caps.validate_request(IMAGE_MODEL_ID, req)?;
    }
    Ok(())
}

/// The tiling one request decodes at: always the shipped 256 / 64. A request naming that pair
/// gets it; one naming anything else is refused, not clamped.
pub fn decode_tiling_for(req: &GenerationRequest) -> Result<SpatialTiling> {
    let Some(memory) = req.memory else {
        return Ok(SpatialTiling::default());
    };
    if !memory.tile_vae_decode {
        return Ok(SpatialTiling::default());
    }
    let edge = memory
        .decode_tile_edge
        .unwrap_or(crate::image_vae::IMAGE_DECODE_TILE_EDGE);
    let overlap = memory
        .decode_overlap
        .unwrap_or(crate::image_vae::IMAGE_DECODE_OVERLAP);
    crate::h3_image::memory::validate_decode_geometry(edge, overlap).map_err(Error::from)?;
    Ok(SpatialTiling::square(edge as i32, overlap as i32))
}

/// Seeded noise for one image, drawn as `[1, C, 1, H, W]` and patchified (before_denoise.py draws
/// the latent tensor, then packs it).
pub fn initial_rows(layout: &ImageLayout, seed: u64) -> Result<Array> {
    let key = mlx_rs::random::key(seed)?;
    let noise = mlx_rs::random::normal::<f32>(
        &[
            1,
            LATENT_CHANNELS as i32,
            1,
            layout.latent_height(),
            layout.latent_width(),
        ],
        None,
        None,
        Some(&key),
    )?;
    patchify_video_latents(&noise, layout.patch())
}

/// `[1, 3, H, W]` ImageNet-normalized RGB → one 8-bit RGB image, reverting the normalization,
/// clamping, and quantizing by **truncation** as the released pipeline does (decoders.py:156-161).
pub fn decoded_to_image(decoded: &Array) -> Result<mlx_gen::media::Image> {
    let s = decoded.shape();
    if s.len() != 4 || s[0] != 1 || s[1] != 3 {
        return Err(Error::Msg(format!(
            "{IMAGE_MODEL_ID} decode: expected [1, 3, H, W], got {s:?}"
        )));
    }
    let x = decoded.as_dtype(Dtype::Float32)?;
    let mean = Array::from_slice(&PIXEL_MEAN, &[1, 3, 1, 1]);
    let std = Array::from_slice(&PIXEL_STD, &[1, 3, 1, 1]);
    let px = clip(&add(&multiply(&x, &std)?, &mean)?, (0.0, 1.0))?;
    let px = mlx_rs::ops::floor(&clip(
        &multiply(&px, Array::from_f32(255.0))?,
        (0.0, 255.0),
    )?)?
    .as_dtype(Dtype::Uint8)?
    .transpose_axes(&[0, 2, 3, 1])?;
    // The shared splitter forces logical C-order before reading the host buffer.
    mlx_gen::image::u8_frames_to_images(&px)?
        .pop()
        .ok_or_else(|| Error::Msg(format!("{IMAGE_MODEL_ID} decode: no image decoded")))
}

/// Resize an 8-bit RGB image to `width x height` with PIL-equivalent Lanczos, as the released
/// pipeline's `VaeImageProcessor.resize` does; a same-size image is returned unchanged.
pub fn resize_to_request(
    image: mlx_gen::media::Image,
    width: u32,
    height: u32,
) -> Result<mlx_gen::media::Image> {
    if (image.width, image.height) == (width, height) {
        return Ok(image);
    }
    let resized = mlx_gen::gen_core::imageops::resize_lanczos_u8(
        &image.pixels,
        image.height as usize,
        image.width as usize,
        height as usize,
        width as usize,
    )?;
    Ok(mlx_gen::media::Image {
        width,
        height,
        pixels: resized
            .into_iter()
            .map(|v| v.round().clamp(0.0, 255.0) as u8)
            .collect(),
    })
}

/// One reference resized for one of its consumers, PIL-Lanczos as `VaeImageProcessor.resize`
/// does; a reference already at `(width, height)` keeps its pixels.
pub fn resize_reference(
    image: &mlx_gen::media::Image,
    (width, height): (u32, u32),
) -> Result<mlx_gen::media::Image> {
    resize_to_request(image.clone(), width, height)
}

/// The VAE copy's pixels: cropped at the top left to whole [`VAE_SPATIAL_RATIO`] cells, then
/// ImageNet-normalized to `[1, 3, H, W]` (`MiniMaxH3ImageEditVaeEncoderStep`).
pub fn reference_vae_pixels(image: &mlx_gen::media::Image) -> Result<Array> {
    let r = VAE_SPATIAL_RATIO;
    let (w, h) = (image.width / r * r, image.height / r * r);
    if w == 0 || h == 0 {
        return Err(Error::Msg(format!(
            "{IMAGE_MODEL_ID}: a {}x{} reference holds no whole {r}px VAE cell",
            image.width, image.height
        )));
    }
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h as usize {
        let row = y * image.width as usize * 3;
        pixels.extend_from_slice(&image.pixels[row..row + w as usize * 3]);
    }
    let cropped = mlx_gen::media::Image {
        width: w,
        height: h,
        pixels,
    };
    Ok(crate::keyframe::keyframe_to_vae_pixels(&cropped)?.reshape(&[1, 3, h as i32, w as i32])?)
}

/// Seeded `[shape]` standard-normal noise for an edit's per-reference draws: the posterior
/// sample (`stream` 0) and the conditioning augmentation (`stream` 1) of reference `index`, each
/// its own key so no two draws share a stream and none disturbs the generated image's noise.
pub fn reference_noise(seed: u64, stream: u64, index: usize, shape: &[i32]) -> Result<Array> {
    let key = mlx_rs::random::key(
        seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(stream.wrapping_mul(1 << 32))
            .wrapping_add(index as u64 + 1),
    )?;
    Ok(mlx_rs::random::normal::<f32>(
        shape,
        None,
        None,
        Some(&key),
    )?)
}

/// An edit's conditioning rows: every reference latent `[1, 24, h, w]` noised to
/// [`REFERENCE_NOISE_AUG`] with its `noise` (`scale_noise`, `t · x + (1 - t) · n` in float32),
/// patchified as one latent frame, and concatenated in packed order —
/// `MiniMaxH3ImageEditPrepareLatentsStep`. Returns `[1, num_reference_rows, 96]`.
pub fn reference_condition_rows(
    latents: &[Array],
    noise: &[Array],
    patch: [i32; 3],
) -> Result<Array> {
    if latents.len() != noise.len() || latents.is_empty() {
        return Err(Error::Msg(format!(
            "{IMAGE_MODEL_ID}: {} reference latents for {} noise tensors",
            latents.len(),
            noise.len()
        )));
    }
    let mut rows = Vec::with_capacity(latents.len());
    for (latent, n) in latents.iter().zip(noise) {
        let latent = latent.as_dtype(Dtype::Float32)?;
        let noised = crate::conditioning::scale_noise(&latent, REFERENCE_NOISE_AUG, n)?;
        rows.push(patchify_video_latents(&noised.expand_dims(2)?, patch)?);
    }
    Ok(mlx_rs::ops::concatenate_axis(&rows, 1)?)
}

/// Drop a component and drain MLX's allocator cache (H3's retried drain).
fn release<T>(component: T) {
    drop(component);
    mlx_gen::residency::drain_allocator_cache();
}

impl MiniMaxH3Image {
    /// The snapshot root this generator reads.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The resolved DiT directory (the staged tier's `transformer_diffusers/`).
    pub fn dit_dir(&self) -> &Path {
        &self.dit_dir
    }

    /// The resolved text-encoder directory.
    pub fn text_encoder_dir(&self) -> &Path {
        &self.text_encoder_dir
    }

    /// Build the request's text encoder at the residency [`Self::requested_transformer_window`]
    /// admitted: resident over the mapped shards, or H3's deferred encoder with a per-window
    /// layer rebuild (rung 4, SW-7094). One constructor for both presentations, so `text2image`
    /// cannot stream while `edit` silently stays resident.
    fn build_te(
        &self,
        w: Option<&mlx_gen::weights::Weights>,
        cfg: &MiniMaxH3TeConfig,
        window: Option<u32>,
        cancel: &mlx_gen::CancelFlag,
    ) -> Result<MiniMaxH3TextEncoder> {
        match (window, w) {
            (None, Some(w)) => MiniMaxH3TextEncoder::from_weights(w, LM_PREFIX, cfg),
            (Some(window), None) => {
                let mut te = MiniMaxH3TextEncoder::from_dir_deferred(
                    &self.text_encoder_dir,
                    LM_PREFIX,
                    cfg,
                )?;
                te.set_block_window(window as usize, cancel.clone())?;
                Ok(te)
            }
            _ => Err(Error::Msg(format!(
                "{IMAGE_MODEL_ID}: a resident text encoder builds from mapped shards and a \
                 windowed one from none"
            ))),
        }
    }

    /// Encode the prompt (layer-50 hidden state) and release the text encoder. `window` is the
    /// rung-4 layer window, `None` for the resident encoder.
    pub fn encode_prompt(
        &self,
        prompt: &str,
        window: Option<u32>,
        cancel: &mlx_gen::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let ids = t2i_prompt_ids(tok.ids(prompt)?)?;
        let len = ids.len() as i32;
        let input_ids = Array::from_slice(&ids, &[1, len]);
        let mask = Array::from_slice(&vec![1i32; ids.len()], &[1, len]);
        // Under a window the shard map is skipped: `from_dir_deferred` reads only `embed_tokens`,
        // and a retained map would keep every layer tensor reachable across the windowed walk.
        let w = match window {
            None => Some(crate::text_encoder::map_shards(
                &self.text_encoder_dir,
                false,
            )?),
            Some(_) => None,
        };
        let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
        let te = self.build_te(w.as_ref(), &cfg, window, cancel)?;
        let context =
            mlx_gen::gen_core::with_prompt_encoding(on_progress, || te.forward(&input_ids, &mask))?;
        // Force before the encoder is dropped: a lazy context still references every weight.
        mlx_rs::transforms::eval([&context])?;
        release((te, w));
        Ok(context)
    }

    /// The `edit` presentation (`MiniMaxH3ImageEditTextEncoderStep`): H3's Qwen3-VL vision tower
    /// over every vision-resized reference, spliced into the `Picture i: ` + vision-block
    /// presentation, then the layer-50 context. Returns the context and its per-row tags.
    pub fn encode_edit_prompt(
        &self,
        prompt: &str,
        vision_images: &[mlx_gen::media::Image],
        window: Option<u32>,
        cancel: &mlx_gen::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<(Array, Vec<i32>)> {
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let shard = self.text_encoder_dir.join(format!(
            "model-{:05}-of-00014.safetensors",
            crate::text_encoder::VISION_SHARD
        ));
        if !shard.is_file() {
            return Err(Error::Msg(format!(
                "{IMAGE_MODEL_ID}: an edit reads the vision tower from {} (MiniMax-H3's \
                 text_encoder/ shard {}), which is missing",
                shard.display(),
                crate::text_encoder::VISION_SHARD
            )));
        }
        let mut w = crate::text_encoder::map_shards(&self.text_encoder_dir, true)?;
        let vision = mlx_gen_boogu::vision::VisionTower::from_weights(
            &w,
            crate::text_encoder::minimax_h3_vision_config(),
            VISION_PREFIX,
            crate::model::VISION_GROUP_SIZE,
        )?;
        w.materialize_accessed()?;
        let refs: Vec<&mlx_gen::media::Image> = vision_images.iter().collect();
        let grounded = crate::text_encoder::run_vision(&vision, &refs)?;
        let mut forced: Vec<&Array> = grounded.embeds.iter().collect();
        forced.extend(grounded.deepstack.iter().flatten());
        mlx_rs::transforms::eval(forced)?;
        release(vision);
        w.remove_prefix(VISION_PREFIX);

        let (ids, tags) = tok.image_edit_ids(prompt, &grounded.counts)?;
        let len = ids.len() as i32;
        let input_ids = Array::from_slice(&ids, &[1, len]);
        let mask = Array::from_slice(&vec![1i32; ids.len()], &[1, len]);
        let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
        // Under a window the mapped shards are released before the encoder is built: the deferred
        // loader reopens the staged directory per window (H3's `encode_prompt_grounded`).
        let w = match window {
            None => Some(w),
            Some(_) => {
                release(w);
                None
            }
        };
        let te = self.build_te(w.as_ref(), &cfg, window, cancel)?;
        let context = mlx_gen::gen_core::with_prompt_encoding(on_progress, || {
            te.forward_with_images(
                &input_ids,
                &mask,
                &grounded.embeds,
                &grounded.deepstack,
                &grounded.grids,
            )
        })?;
        mlx_rs::transforms::eval([&context])?;
        release((te, w, grounded));
        if context.shape()[1] != tags.len() as i32 {
            return Err(Error::Msg(format!(
                "{IMAGE_MODEL_ID}: the edit context has {} rows but {} modality tags",
                context.shape()[1],
                tags.len()
            )));
        }
        Ok((context, tags))
    }

    /// Every VAE-resized reference's conditioning latent `[1, 24, h, w]` on its own grid
    /// (`MiniMaxH3ImageEditVaeEncoderStep`), the posterior sampled with [`reference_noise`]
    /// stream 0. The encoder is loaded, used and released here.
    pub fn encode_reference_latents(
        &self,
        vae_images: &[mlx_gen::media::Image],
        seed: u64,
        patch: [i32; 3],
    ) -> Result<Vec<Array>> {
        let encoder = MiniMaxH3ImageVaeEncoder::load(&self.root)?;
        let mut latents = Vec::with_capacity(vae_images.len());
        for (i, image) in vae_images.iter().enumerate() {
            let posterior = encoder.encode(&reference_vae_pixels(image)?)?;
            let noise = reference_noise(seed, 0, i, posterior.mean().shape())?;
            let latent = encoder.reference_latent(&posterior, &noise, (patch[1], patch[2]))?;
            mlx_rs::transforms::eval([&latent])?;
            latents.push(latent);
        }
        release(encoder);
        Ok(latents)
    }

    /// The rung-4 admission for one request (SW-7094): `Some(window)` routes the render through
    /// the deferred loaders ([`MiniMaxH3ImageDit::load_dir_deferred`],
    /// [`MiniMaxH3TextEncoder::from_dir_deferred`]), `None` is the resident staged path.
    ///
    /// Each guard mirrors a declaration clause of [`crate::h3_image::memory`], because a request
    /// the contract refuses must be refused here even without a request scope, and one it admits
    /// must not be silently downgraded:
    ///
    /// * the load shape must be [`mlx_gen::gen_core::LoadShape::DeferredMaterialization`] — rung
    ///   4's prerequisite, and what `memory::streamable` resolves `Implemented` on;
    /// * the component must be [`mlx_gen::gen_core::TransformerComponent::Both`] — the only
    ///   declared arm (the conditioning phase is the taller one at every tier);
    /// * the window must be the published singleton
    ///   [`crate::h3_image::memory::IMAGE_TRANSFORMER_WINDOW_SIZE`].
    ///
    /// Adapters are admitted: the block stream re-applies them to every rebuilt block (SW-7096).
    pub fn requested_transformer_window(&self, req: &GenerationRequest) -> Result<Option<u32>> {
        use mlx_gen::gen_core::{LoadShape, TransformerComponent};
        let Some(memory) = req.memory.filter(|memory| memory.stream_transformer_blocks) else {
            return Ok(None);
        };
        if self.load_shape != LoadShape::DeferredMaterialization {
            return Err(Error::Unsupported(format!(
                "{IMAGE_MODEL_ID}: transformer streaming (rung 4) requires a \
                 LoadShape::DeferredMaterialization load; this generator was loaded resident, \
                 where the rung is declared Missing"
            )));
        }
        let component = memory
            .transformer_window_component
            .unwrap_or(TransformerComponent::Dit);
        if component != TransformerComponent::Both {
            return Err(Error::Unsupported(format!(
                "{IMAGE_MODEL_ID}: transformer streaming is declared for TransformerComponent::Both \
                 only — got {component:?}"
            )));
        }
        let size = memory
            .transformer_window_size
            .unwrap_or(crate::h3_image::memory::IMAGE_TRANSFORMER_WINDOW_SIZE);
        if size != crate::h3_image::memory::IMAGE_TRANSFORMER_WINDOW_SIZE {
            return Err(Error::Unsupported(format!(
                "{IMAGE_MODEL_ID}: transformer window {size} is outside the published domain [{}]",
                crate::h3_image::memory::IMAGE_TRANSFORMER_WINDOW_SIZE
            )));
        }
        Ok(Some(size))
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        // Resolved before any weight maps, so an unadmitted geometry or window is a request error.
        let tiling = decode_tiling_for(req)?;
        let window = self.requested_transformer_window(req)?;
        // Rung 3: an undeclared chunk size is refused here; the budget itself depends on the
        // packed sequence and is resolved once the layout is known.
        crate::h3_image::memory::requested_bounded_attention(req.memory, 0).map_err(Error::from)?;
        // The same canvas admission sized this request's resources from.
        let (canvas_w, canvas_h) = crate::h3_image::memory::admission_canvas(req.width, req.height)
            .map_err(Error::from)?;
        let (latent_h, latent_w) = latent_grid(canvas_w, canvas_h)?;
        let seed = req.seed.unwrap_or_else(default_seed);
        let count = req.count.max(1);
        let sampler =
            request_sampler(req).map_err(|r| Error::from(mlx_gen::gen_core::Error::from(r)))?;

        let references = edit_reference_images(&req.conditioning)
            .map_err(|r| Error::from(mlx_gen::gen_core::Error::from(r)))?;
        let patch = crate::dit::config::MiniMaxH3ImageDitConfig::default()
            .base
            .patch_size;

        // --- 1. conditioning ---------------------------------------------------------------
        // An edit's references are resized once per consumer; neither copy is resized to the
        // generated image's size.
        on_progress(Progress::Loading(mlx_gen::gen_core::LoadPhase::TextEncoder));
        let (context, text_tags, reference_latents) = if references.is_empty() {
            let context = self.encode_prompt(&req.prompt, window, &req.cancel, on_progress)?;
            let n = context.shape()[1];
            (
                context,
                vec![mlx_gen::gen_core::h3_image::TEXT_TAG; n as usize],
                Vec::new(),
            )
        } else {
            let vision_images = references
                .iter()
                .map(|r| resize_reference(r, reference_vision_size(r.width, r.height)))
                .collect::<Result<Vec<_>>>()?;
            let (context, tags) = self.encode_edit_prompt(
                &req.prompt,
                &vision_images,
                window,
                &req.cancel,
                on_progress,
            )?;
            drop(vision_images);
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let vae_images = references
                .iter()
                .map(|r| resize_reference(r, reference_vae_size(r.width, r.height)))
                .collect::<Result<Vec<_>>>()?;
            let latents = self.encode_reference_latents(&vae_images, seed, patch)?;
            (context, tags, latents)
        };
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // --- 2. denoise --------------------------------------------------------------------
        on_progress(Progress::Loading(mlx_gen::gen_core::LoadPhase::Renderer));
        let mut dit = match window {
            None => MiniMaxH3ImageDit::load_dir(&self.dit_dir, self.dtype)?,
            Some(_) => MiniMaxH3ImageDit::load_dir_deferred(&self.dit_dir, self.dtype)?,
        };
        if !self.adapters.is_empty() {
            // Strict, like H3 video: an unmatched target or a file that matched nothing fails the
            // render here rather than producing an image the LoRA barely touched. Under rung 4 the
            // block residuals land in the stream's overlay and are forced once here.
            crate::adapters::apply_minimax_h3_adapters(&mut dit, &self.adapters)?;
            dit.materialize_streamed_adapters()?;
        }
        debug_assert_eq!(patch, dit.config().base.patch_size);
        let layout = if reference_latents.is_empty() {
            ImageLayout::t2i(
                context.shape()[1],
                latent_h as i32,
                latent_w as i32,
                dit.config().base.patch_size,
            )?
        } else {
            let grids: Vec<(i32, i32)> = reference_latents
                .iter()
                .map(|l| (l.shape()[2], l.shape()[3]))
                .collect();
            ImageLayout::edit(
                &text_tags,
                &grids,
                latent_h as i32,
                latent_w as i32,
                dit.config().base.patch_size,
            )?
        };
        let mut denoiser = match window {
            None => ImageDenoiser::new(
                dit,
                layout.clone(),
                &context,
                SigmaSchedule::h3_image_distilled()?,
            )?,
            Some(window) => ImageDenoiser::new_windowed(
                dit,
                layout.clone(),
                &context,
                SigmaSchedule::h3_image_distilled()?,
                window as usize,
                req.cancel.clone(),
            )?,
        };
        // Rung 3 (SW-7094): every block's attention on whole heads, resident or windowed.
        if let Some((budget, axis)) = crate::h3_image::memory::requested_bounded_attention(
            req.memory,
            layout.seq_len() as u64,
        )
        .map_err(Error::from)?
        {
            denoiser.set_bounded_attention(budget, axis);
        }
        release(context);
        let steps = denoiser.num_steps() as u32;
        let total = steps * count;
        let mut latents = Vec::with_capacity(count as usize);
        for n in 0..count {
            let image_seed = seed.wrapping_add(u64::from(n));
            let noise = initial_rows(&layout, image_seed)?;
            // An edit's reference rows lead the packed image stream, noised to 0.999 with one draw
            // per reference after the generated image's own.
            let packed = if reference_latents.is_empty() {
                noise
            } else {
                let aug = reference_latents
                    .iter()
                    .enumerate()
                    .map(|(i, l)| reference_noise(image_seed, 1, i, l.shape()))
                    .collect::<Result<Vec<_>>>()?;
                let condition = reference_condition_rows(&reference_latents, &aug, patch)?;
                mlx_rs::ops::concatenate_axis(&[condition, noise], 1)?
            };
            let done = n * steps;
            // Image `n`'s LCM noise is seeded from its own seed, like its initial latent.
            let rows =
                denoiser.denoise_with(&sampler, image_seed, &packed, &req.cancel, &mut |s| {
                    on_progress(Progress::Step {
                        current: done + s as u32,
                        total,
                    })
                })?;
            let rows = denoiser.generated_rows(&rows)?;
            let latent = unpatchify_video_rows(
                &rows,
                LATENT_CHANNELS as i32,
                1,
                layout.latent_height(),
                layout.latent_width(),
                layout.patch(),
            )?
            .reshape(&[
                1,
                LATENT_CHANNELS as i32,
                layout.latent_height(),
                layout.latent_width(),
            ])?;
            mlx_rs::transforms::eval([&latent])?;
            latents.push(latent);
        }
        release((denoiser, reference_latents));

        // --- 3. decode ---------------------------------------------------------------------
        on_progress(Progress::Decoding);
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let vae = MiniMaxH3ImageVae::load(&self.root, self.dtype)?.with_tiling(tiling);
        let mut images = Vec::with_capacity(latents.len());
        for latent in &latents {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let decoded = vae.decode(latent)?;
            images.push(resize_to_request(
                decoded_to_image(&decoded)?,
                req.width,
                req.height,
            )?);
        }
        release((vae, latents));
        Ok(GenerationOutput::Images(images))
    }
}

impl Generator for MiniMaxH3Image {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> mlx_gen::gen_core::Result<()> {
        validate_request(&self.descriptor.capabilities, req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> mlx_gen::gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }

    fn memory_strategy_contract(&self) -> Option<&mlx_gen::gen_core::MemoryProviderContract> {
        Some(&self.memory_strategy)
    }

    fn memory_strategy_safety_check(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::MemorySafetyDecision {
        crate::h3_image::memory::safety_check_at_tier(
            &self.memory_strategy,
            self.memory_tier,
            context,
        )
    }

    fn begin_memory_strategy_request(
        &self,
        context: &mlx_gen::gen_core::MemoryRunContext,
    ) -> mlx_gen::gen_core::Result<Option<Box<dyn mlx_gen::gen_core::MemoryRequestScope + '_>>>
    {
        crate::h3_image::memory::begin_request(&self.memory_strategy, self.memory_tier, context)
    }
}

mlx_gen::register_generators! {
    pub(crate) const REGISTRATION = descriptor => load
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(width: u32, height: u32) -> GenerationRequest {
        GenerationRequest {
            prompt: "a lighthouse at dusk".into(),
            width,
            height,
            count: 1,
            ..Default::default()
        }
    }

    /// Its own id, an IMAGE descriptor, and none of H3 video's flags (Video modality, mac_only,
    /// max_count 1, H3's canvas caps).
    #[test]
    fn the_descriptor_is_its_own_image_model() {
        let d = descriptor();
        assert_eq!(d.id, "minimax_h3_image");
        assert_ne!(d.id, crate::MODEL_ID);
        assert_eq!(d.modality, Modality::Image);
        assert!(!d.capabilities.mac_only);
        assert!(d.capabilities.max_count > 1);
        assert_eq!(d.capabilities.supported_steps, StepSupport::Exact(vec![6]));
        assert!(d.capabilities.max_size > crate::pipeline::MAX_CANVAS_EDGE);
        assert!(d.capabilities.unconditionally_engages_staged_residency);
        let h3 = crate::model::descriptor();
        assert_eq!(
            h3.modality,
            Modality::Video,
            "H3 video's descriptor is unchanged"
        );
        assert!(h3.capabilities.mac_only);
        assert_eq!(h3.capabilities.max_count, 1);
    }

    #[test]
    fn the_fixed_sampler_and_cfg_refusals_hold() {
        let caps = descriptor().capabilities;
        validate_request(&caps, &req(2048, 2048)).unwrap();
        let ok = GenerationRequest {
            steps: Some(6),
            guidance: Some(1.0),
            ..req(1000, 1001)
        };
        validate_request(&caps, &ok).unwrap();
        for bad in [
            GenerationRequest {
                steps: Some(8),
                ..req(2048, 2048)
            },
            GenerationRequest {
                guidance: Some(4.0),
                ..req(2048, 2048)
            },
            GenerationRequest {
                negative_prompt: Some("blurry".into()),
                ..req(2048, 2048)
            },
            GenerationRequest {
                prompt: "  ".into(),
                ..req(2048, 2048)
            },
            req(0, 2048),
            GenerationRequest {
                sampler: Some("dpmpp_2m".into()),
                ..req(2048, 2048)
            },
            GenerationRequest {
                h3_image_sampler: Some(mlx_gen::gen_core::h3_image::H3ImageSampler::lcm(0.0)),
                ..req(2048, 2048)
            },
            GenerationRequest {
                sampler: Some("euler".into()),
                h3_image_sampler: Some(mlx_gen::gen_core::h3_image::H3ImageSampler::lcm_default()),
                ..req(2048, 2048)
            },
        ] {
            assert!(validate_request(&caps, &bad).is_err(), "{bad:?}");
        }
        // SW-7106: both samplers are admitted, by name or by block.
        for good in [
            GenerationRequest {
                sampler: Some("lcm".into()),
                ..req(2048, 2048)
            },
            GenerationRequest {
                sampler: Some("euler".into()),
                ..req(2048, 2048)
            },
            GenerationRequest {
                h3_image_sampler: Some(mlx_gen::gen_core::h3_image::H3ImageSampler::Lcm {
                    s_noise: 1.1,
                    s_noise_end: 0.9,
                    noise_clip_std: 2.0,
                }),
                ..req(2048, 2048)
            },
        ] {
            validate_request(&caps, &good).unwrap();
        }
        assert_eq!(caps.samplers, vec!["euler", "lcm"]);
        // Above H3 video's 1.03 MP canvas and 2016 px edge: admitted for this family.
        validate_request(&caps, &req(4736, 3552)).unwrap();
    }

    /// The decode geometry is the shipped one or a refusal.
    #[test]
    fn decode_tiling_admits_only_the_shipped_geometry() {
        let mut r = req(2048, 2048);
        assert_eq!(decode_tiling_for(&r).unwrap(), SpatialTiling::default());
        r.memory = Some(mlx_gen::gen_core::GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(256),
            decode_overlap: Some(64),
            ..Default::default()
        });
        assert_eq!(
            decode_tiling_for(&r).unwrap(),
            SpatialTiling::square(256, 64)
        );
        r.memory = Some(mlx_gen::gen_core::GenerationMemory {
            tile_vae_decode: true,
            decode_tile_edge: Some(512),
            decode_overlap: Some(64),
            ..Default::default()
        });
        assert!(decode_tiling_for(&r).is_err());
    }

    fn generator(shape: mlx_gen::gen_core::LoadShape) -> MiniMaxH3Image {
        let mut spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()));
        spec.load_shape = shape;
        MiniMaxH3Image {
            descriptor: descriptor(),
            root: "/nonexistent-minimax-h3-image".into(),
            dit_dir: "/nonexistent-minimax-h3-image/transformer_diffusers".into(),
            text_encoder_dir: "/nonexistent-minimax-h3-image/text_encoder".into(),
            dtype: Dtype::Bfloat16,
            memory_strategy: crate::h3_image::memory::weights_free_contract(&spec).unwrap(),
            memory_tier: mlx_gen::request_scope::loaded_tier(&spec),
            adapters: Vec::new(),
            load_shape: shape,
        }
    }

    /// SW-7094: rung 4's runtime admission mirrors the declaration — the generation memory the
    /// contract derives for its own rung-4 selection is admitted with window 1; a resident load,
    /// a DiT-only scope or another window is refused rather than silently run resident.
    #[test]
    fn transformer_streaming_is_admitted_exactly_as_declared() {
        use mlx_gen::gen_core::{
            GenerationMemory, LoadShape, MemorySelection, MemoryStrategy, MemoryStrategyParameters,
            TransformerComponent,
        };
        let deferred = generator(LoadShape::DeferredMaterialization);
        let selection = MemorySelection {
            strategy: MemoryStrategy::BoundedTransformerResidency,
            parameters: MemoryStrategyParameters {
                decode_tile_edge: Some(256),
                decode_overlap: Some(64),
                transformer_window_size: Some(1),
                transformer_window_component: Some(TransformerComponent::Both),
                attention_chunk_size: Some(crate::h3_image::memory::IMAGE_ATTENTION_CHUNK_SIZE),
                ..Default::default()
            },
            tier: deferred.memory_tier,
        };
        let declared = deferred.memory_strategy.generation_memory(&selection);
        let streamed = GenerationRequest {
            memory: declared,
            ..req(4096, 4096)
        };
        assert_eq!(
            deferred.requested_transformer_window(&streamed).unwrap(),
            Some(1)
        );
        assert_eq!(
            deferred
                .requested_transformer_window(&req(4096, 4096))
                .unwrap(),
            None
        );
        let with = |memory: GenerationMemory| GenerationRequest {
            memory: Some(memory),
            ..req(4096, 4096)
        };
        let both = GenerationMemory {
            stream_transformer_blocks: true,
            transformer_window_component: Some(TransformerComponent::Both),
            ..Default::default()
        };
        assert_eq!(
            deferred.requested_transformer_window(&with(both)).unwrap(),
            Some(1)
        );
        for refused in [
            GenerationMemory {
                transformer_window_component: Some(TransformerComponent::Dit),
                ..both
            },
            GenerationMemory {
                transformer_window_component: None,
                ..both
            },
            GenerationMemory {
                transformer_window_size: Some(2),
                ..both
            },
        ] {
            assert!(deferred
                .requested_transformer_window(&with(refused))
                .is_err());
        }
        let resident = generator(LoadShape::EagerMaterialization);
        let e = resident
            .requested_transformer_window(&with(both))
            .unwrap_err()
            .to_string();
        assert!(e.contains("DeferredMaterialization"), "{e}");
        // Declaration and admission agree: the resident contract does not admit the selection.
        assert!(resident
            .memory_strategy
            .validate_selection(&selection)
            .is_err());
    }

    /// A missing component is a load-time error naming the path, and only the text-encoder
    /// component key is accepted.
    #[test]
    fn load_probes_the_snapshot_and_rejects_unknown_components() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent-minimax-h3-image".into()));
        let e = load(&spec).err().unwrap().to_string();
        assert!(e.contains(IMAGE_DIT_COMPONENT), "{e}");
        let mut spec = spec;
        spec.components.insert(
            "transformer".into(),
            WeightsSource::Dir("/nonexistent".into()),
        );
        let e = load(&spec).err().unwrap().to_string();
        assert!(e.contains("transformer"), "{e}");
    }

    #[test]
    fn resize_is_identity_at_the_canvas_and_lanczos_otherwise() {
        let px = vec![10u8; 4 * 4 * 3];
        let image = |pixels| mlx_gen::media::Image {
            width: 4,
            height: 4,
            pixels,
        };
        let same = resize_to_request(image(px.clone()), 4, 4).unwrap();
        assert_eq!(same.pixels, px);
        let down = resize_to_request(image(px), 2, 3).unwrap();
        assert_eq!((down.width, down.height), (2, 3));
        assert_eq!(down.pixels, vec![10u8; 2 * 3 * 3]);
    }

    fn image(w: u32, h: u32) -> mlx_gen::media::Image {
        mlx_gen::media::Image {
            width: w,
            height: h,
            pixels: (0..w * h * 3).map(|v| (v % 251) as u8).collect(),
        }
    }

    fn with_refs(n: usize, w: u32, h: u32) -> GenerationRequest {
        GenerationRequest {
            conditioning: vec![mlx_gen::gen_core::Conditioning::MultiReference {
                images: (0..n).map(|_| image(w, h)).collect(),
            }],
            ..req(1024, 1024)
        }
    }

    /// `edit` is advertised as reference images, and the validator admits 1..=16 references of
    /// at least 32 px a side and refuses 17 or a 31 px one.
    #[test]
    fn edit_admits_one_to_sixteen_references_of_at_least_32_px() {
        let caps = descriptor().capabilities;
        assert!(caps.conditioning.contains(&ConditioningKind::Reference));
        assert!(caps
            .conditioning
            .contains(&ConditioningKind::MultiReference));
        validate_request(&caps, &with_refs(1, 32, 32)).unwrap();
        validate_request(&caps, &with_refs(16, 64, 40)).unwrap();
        let e = validate_request(&caps, &with_refs(17, 64, 64))
            .unwrap_err()
            .to_string();
        assert!(e.contains("17"), "{e}");
        let e = validate_request(&caps, &with_refs(2, 31, 64))
            .unwrap_err()
            .to_string();
        assert!(e.contains("32"), "{e}");
    }

    /// The VAE copy is cropped at the top left to whole 16 px cells before normalization.
    #[test]
    fn reference_vae_pixels_crop_to_whole_cells() {
        let img = image(40, 35);
        let px = reference_vae_pixels(&img).unwrap();
        assert_eq!(px.shape(), &[1, 3, 32, 32]);
        // Pixel (y=1, x=2) red channel survives the crop at the same place.
        let want = (f32::from(img.pixels[(40 + 2) * 3]) / 255.0 - PIXEL_MEAN[0]) / PIXEL_STD[0];
        let got: Vec<f32> = px
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert!((got[32 + 2] - want).abs() < 1e-6);
        assert!(reference_vae_pixels(&image(15, 64)).is_err());
    }

    /// Reference rows: `0.999 · x + 0.001 · n`, one latent frame each, concatenated in order.
    #[test]
    fn reference_condition_rows_noise_then_pack_in_order() {
        let a = Array::from_slice(&vec![1.0f32; 24 * 2 * 4], &[1, 24, 2, 4]);
        let b = Array::from_slice(&vec![2.0f32; 24 * 4 * 2], &[1, 24, 4, 2]);
        let za = Array::from_slice(&vec![0.0f32; 24 * 8], &[1, 24, 2, 4]);
        let zb = Array::from_slice(&vec![1000.0f32; 24 * 8], &[1, 24, 4, 2]);
        let rows = reference_condition_rows(&[a, b], &[za, zb], [1, 2, 2]).unwrap();
        assert_eq!(rows.shape(), &[1, 2 + 2, 96]);
        let v: Vec<f32> = rows.as_slice::<f32>().to_vec();
        assert!((v[0] - 0.999).abs() < 1e-6);
        assert!((v[2 * 96] - (2.0 * 0.999 + 1000.0 * (1.0 - 0.999f32))).abs() < 1e-3);
        assert!(reference_condition_rows(&[], &[], [1, 2, 2]).is_err());
    }

    /// Distinct references and streams draw distinct noise; the same key draws the same noise.
    #[test]
    fn reference_noise_streams_are_distinct_and_deterministic() {
        let draw = |seed, stream, i| -> Vec<f32> {
            reference_noise(seed, stream, i, &[1, 4])
                .unwrap()
                .as_slice::<f32>()
                .to_vec()
        };
        assert_eq!(draw(7, 0, 0), draw(7, 0, 0));
        assert_ne!(draw(7, 0, 0), draw(7, 1, 0));
        assert_ne!(draw(7, 0, 0), draw(7, 0, 1));
        assert_ne!(draw(7, 0, 0), draw(8, 0, 0));
    }
}
