//! The **`minimax_h3_image` generator**: prompt → image, `MiniMaxAI/MiniMax-H3-Image` on MLX.
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
//! 3. **Decode.** `vae_diffusers/` ([`MiniMaxH3ImageVae`]), tiled at the shipped 256 / 64; pixel
//!    normalization reverted; 8-bit by truncation (decoders.py); resized back to the requested size
//!    with Lanczos when the canvas differs.
//!
//! # Size
//!
//! Any positive size. The generation canvas is [`generation_canvas`] at upstream's default 4 MP
//! minimum area: a smaller request is generated larger at the same aspect and resized back. None
//! of H3 video's caps (1.03 MP canvas, 2016 px edge, 17n+5 frames) apply.

use std::path::{Path, PathBuf};

use mlx_rs::ops::{add, clip, multiply};
use mlx_rs::{Array, Dtype};

use mlx_gen::gen_core::h3_image::{
    generation_canvas, latent_grid, t2i_prompt_ids, validate_request as validate_h3_image,
    FIXED_GUIDANCE, FIXED_STEPS, MIN_GENERATION_AREA_DEFAULT,
};
use mlx_gen::gen_core::{
    reject_unknown_components, Capabilities, GenerationOutput, GenerationRequest, Generator,
    LoadSpec, Modality, ModelDescriptor, Precision, Progress, Quant, StepSupport, WeightsSource,
};
use mlx_gen::{default_seed, Error, Result};

use crate::config::LATENT_CHANNELS;
use crate::denoise::schedule::SigmaSchedule;
use crate::h3_image::dit::{ImageDenoiser, MiniMaxH3ImageDit, IMAGE_DIT_COMPONENT};
use crate::h3_image::layout::ImageLayout;
use crate::image_vae::{MiniMaxH3ImageVae, IMAGE_VAE_COMPONENT};
use crate::model::TEXT_ENCODER_COMPONENT;
use crate::pipeline::{patchify_video_latents, unpatchify_video_rows, PIXEL_MEAN, PIXEL_STD};
use crate::spatial_tiling::SpatialTiling;
use crate::text_encoder::{MiniMaxH3TeConfig, MiniMaxH3TextEncoder, MiniMaxH3Tokenizer, LM_PREFIX};

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
            // text2image only in this item; edit (reference images) is a separate route.
            conditioning: vec![],
            supports_lora: false,
            supports_lokr: false,
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
    if !spec.adapters.is_empty() {
        return Err(Error::Unsupported(format!(
            "{IMAGE_MODEL_ID}: adapters are not supported"
        )));
    }
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
    Ok(Box::new(MiniMaxH3Image {
        descriptor: descriptor(),
        root,
        dit_dir,
        text_encoder_dir,
        dtype,
        memory_strategy,
        memory_tier: mlx_gen::request_scope::loaded_tier(spec),
    }))
}

/// Reject a request this model cannot serve, before any weight is read: the family contract
/// (gen-core's `h3_image` refusals) first, then the shared floor.
pub fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(Error::Msg(format!(
            "{IMAGE_MODEL_ID}: prompt must not be empty"
        )));
    }
    validate_h3_image(
        req.width,
        req.height,
        None,
        req.steps.unwrap_or(FIXED_STEPS),
        req.guidance,
        req.negative_prompt
            .as_deref()
            .is_some_and(|n| !n.trim().is_empty()),
        req.conditioning.len(),
    )
    .map_err(|r| Error::from(mlx_gen::gen_core::Error::from(r)))?;
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

    /// The resolved text-encoder directory.
    pub fn text_encoder_dir(&self) -> &Path {
        &self.text_encoder_dir
    }

    /// Encode the prompt (layer-50 hidden state) and release the text encoder.
    pub fn encode_prompt(
        &self,
        prompt: &str,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let ids = t2i_prompt_ids(tok.ids(prompt)?)?;
        let len = ids.len() as i32;
        let input_ids = Array::from_slice(&ids, &[1, len]);
        let mask = Array::from_slice(&vec![1i32; ids.len()], &[1, len]);
        let w = crate::text_encoder::map_shards(&self.text_encoder_dir, false)?;
        let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
        let te = MiniMaxH3TextEncoder::from_weights(&w, LM_PREFIX, &cfg)?;
        let context =
            mlx_gen::gen_core::with_prompt_encoding(on_progress, || te.forward(&input_ids, &mask))?;
        // Force before the encoder is dropped: a lazy context still references every weight.
        mlx_rs::transforms::eval([&context])?;
        release((te, w));
        Ok(context)
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
        // Resolved before any weight maps, so an unadmitted geometry is a request error.
        let tiling = decode_tiling_for(req)?;
        let (canvas_w, canvas_h) =
            generation_canvas(req.width, req.height, MIN_GENERATION_AREA_DEFAULT);
        let (latent_h, latent_w) = latent_grid(canvas_w, canvas_h)?;
        let seed = req.seed.unwrap_or_else(default_seed);
        let count = req.count.max(1);

        // --- 1. conditioning ---------------------------------------------------------------
        on_progress(Progress::Loading(mlx_gen::gen_core::LoadPhase::TextEncoder));
        let context = self.encode_prompt(&req.prompt, on_progress)?;
        if req.cancel.is_cancelled() {
            return Err(Error::Canceled);
        }

        // --- 2. denoise --------------------------------------------------------------------
        on_progress(Progress::Loading(mlx_gen::gen_core::LoadPhase::Renderer));
        let dit = MiniMaxH3ImageDit::load_dir(&self.dit_dir, self.dtype)?;
        let layout = ImageLayout::t2i(
            context.shape()[1],
            latent_h as i32,
            latent_w as i32,
            dit.config().base.patch_size,
        )?;
        let denoiser = ImageDenoiser::new(
            dit,
            layout.clone(),
            &context,
            SigmaSchedule::h3_image_distilled()?,
        )?;
        release(context);
        let steps = denoiser.num_steps() as u32;
        let total = steps * count;
        let mut latents = Vec::with_capacity(count as usize);
        for n in 0..count {
            let noise = initial_rows(&layout, seed.wrapping_add(u64::from(n)))?;
            let done = n * steps;
            let rows = denoiser.denoise(&noise, &req.cancel, &mut |s| {
                on_progress(Progress::Step {
                    current: done + s as u32,
                    total,
                })
            })?;
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
        release(denoiser);

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
        ] {
            assert!(validate_request(&caps, &bad).is_err(), "{bad:?}");
        }
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
}
