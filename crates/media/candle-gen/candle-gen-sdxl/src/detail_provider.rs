//! SDXL tile-ControlNet img2img provider for the image-detail utility.
//!
//! The registered SDXL generator deliberately remains txt2img-only. Image detail is a bespoke
//! worker route: it VAE-encodes one source tile, uses that same-sized tile (or another caller-supplied
//! image) as a diffusers SDXL ControlNet condition, denoises the strength-truncated ancestral
//! schedule, and decodes one refined image. Keeping this surface explicit prevents the generic
//! descriptor from advertising arbitrary ControlNet requests that this provider does not implement.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::imageops::resize_lanczos_u8;
use candle_gen::gen_core::runtime::CancelFlag;
use candle_gen::gen_core::{AdapterSpec, Image, PreviewSink, Progress, WeightsSource};
use candle_gen::{CandleError, Result, STEP_RNG_SALT};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::conditioning::SdxlConditioner;
use crate::denoise::{
    decode_image, denoise_ip_control, preprocess_control_image, text_time_ids, ControlContext,
    Denoiser,
};
use crate::loaders::{
    load_instantid_unet_with_adapters, load_sdxl_controlnet, load_sdxl_vae, load_sdxl_vae_encoder,
};
use crate::sampler::EulerAncestralSampler;
use crate::unet::{ControlNet, UNet2DConditionModel, VaeMomentsEncoder};
use crate::{SdxlVaeDecoder, SIZE_MULTIPLE};

const DTYPE: DType = DType::F16;

/// Caller-staged components for one SDXL-family tile detail model.
pub struct SdxlDetailPaths {
    /// SDXL, RealVisXL, or Illustrious diffusers snapshot root.
    pub sdxl_base: PathBuf,
    pub tokenizer_clip_l: WeightsSource,
    pub tokenizer_clip_bigg: WeightsSource,
    pub vae_fp16_fix: WeightsSource,
    /// The diffusers SDXL tile ControlNet checkpoint.
    pub tile_controlnet: WeightsSource,
    /// Optional user SDXL-family adapters, applied to the UNet with per-file apply-or-reject.
    pub adapters: Vec<AdapterSpec>,
}

/// One source-tile refinement request.
#[derive(Clone)]
pub struct SdxlDetailRequest {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub guidance: f32,
    /// Fraction of the ancestral schedule used for the img2img denoise.
    pub strength: f32,
    pub control_scale: f32,
    pub seed: u64,
    pub cancel: CancelFlag,
    pub preview: PreviewSink,
}

impl Default for SdxlDetailRequest {
    fn default() -> Self {
        Self {
            prompt: "ultra detailed, sharp focus, fine texture, high quality".to_owned(),
            negative: "blurry, soft, lowres, smooth, plastic".to_owned(),
            width: 1024,
            height: 1024,
            steps: 24,
            guidance: 5.0,
            strength: 0.55,
            control_scale: 0.7,
            seed: 7,
            cancel: CancelFlag::default(),
            preview: PreviewSink::default(),
        }
    }
}

/// Loaded SDXL img2img + tile-ControlNet stack.
pub struct SdxlDetail {
    conditioner: SdxlConditioner,
    unet: UNet2DConditionModel,
    vae: SdxlVaeDecoder,
    vae_encoder: VaeMomentsEncoder,
    controlnet: ControlNet,
    sampler: EulerAncestralSampler,
    device: Device,
}

impl SdxlDetail {
    pub fn load(paths: &SdxlDetailPaths) -> Result<Self> {
        let device = candle_gen::default_device()?;
        let root = paths.sdxl_base.as_path();
        let conditioner = SdxlConditioner::load(
            root,
            &device,
            DTYPE,
            &paths.tokenizer_clip_l,
            &paths.tokenizer_clip_bigg,
        )?;
        let unet = load_instantid_unet_with_adapters(root, &device, DTYPE, &paths.adapters)?;
        let vae = load_sdxl_vae(&paths.vae_fp16_fix, &device, DTYPE)?;
        let vae_encoder = load_sdxl_vae_encoder(&paths.vae_fp16_fix, &device, DTYPE)?;
        let controlnet = load_sdxl_controlnet(&paths.tile_controlnet, &device, DTYPE)?;
        Ok(Self {
            conditioner,
            unet,
            vae,
            vae_encoder,
            controlnet,
            sampler: EulerAncestralSampler::sdxl(),
            device,
        })
    }

    /// Refine `source` while using `control` as the SDXL tile-ControlNet condition.
    pub fn generate(
        &self,
        req: &SdxlDetailRequest,
        source: &Image,
        control: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_detail_request(req, source, control)?;
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }

        let cfg_on = req.guidance > 1.0;
        let (conditioning, pooled) = self
            .conditioner
            .encode(&req.prompt, &req.negative, cfg_on)?;
        let batch = conditioning.dim(0)?;
        let time_ids = text_time_ids(batch, &self.device, DTYPE)?;

        let x0 = self.encode_source(source, req.width, req.height)?;
        let (_, channels, latent_h, latent_w) = x0.dims4()?;
        let strength = req.strength as f64;
        let start_time = self.sampler.max_time() * strength;
        let mut init_rng = StdRng::seed_from_u64(req.seed);
        let init_noise = candle_gen::seeded_noise_nchw(
            &mut init_rng,
            channels,
            latent_h,
            latent_w,
            &self.device,
        )?;
        let x_t = self.sampler.add_noise(&x0, &init_noise, start_time)?;

        let control = preprocess_control_image(control, req.width, req.height, &self.device)?
            .to_dtype(DTYPE)?;
        let control = if cfg_on {
            Tensor::cat(&[&control, &control], 0)?
        } else {
            control
        };
        let control_ctx = ControlContext {
            controlnet: &self.controlnet,
            cond_embed: self.controlnet.embed_cond(&control)?,
            scale: req.control_scale as f64,
        };

        let effective_steps = ancestral_strength_steps(req.steps, req.strength);
        let steps = self.sampler.timesteps(effective_steps, start_time);
        let denoiser = Denoiser {
            unet: &self.unet,
            sampler: &self.sampler,
        };
        let mut step_rng = StdRng::seed_from_u64(req.seed.wrapping_add(STEP_RNG_SALT));
        let latents = denoise_ip_control(
            &denoiser,
            x_t,
            &conditioning,
            &pooled,
            &time_ids,
            req.guidance as f64,
            &steps,
            &mut step_rng,
            &req.cancel,
            on_progress,
            &req.preview,
            &control_ctx,
            &conditioning,
        )?;
        on_progress(Progress::Decoding);
        decode_image(&self.vae, &latents, None, Some(&req.cancel))
    }

    fn encode_source(&self, source: &Image, width: u32, height: u32) -> Result<Tensor> {
        let (source_w, source_h) = (source.width as usize, source.height as usize);
        let expected =
            candle_gen::gen_core::imageops::checked_image_buffer_len(source_w, source_h, 3)
                .unwrap_or(usize::MAX);
        if source.pixels.len() != expected {
            return Err(CandleError::Msg(format!(
                "sdxl detail: source pixel buffer {} != {source_w}x{source_h}x3",
                source.pixels.len()
            )));
        }
        let (target_w, target_h) = (width as usize, height as usize);
        let resized = resize_lanczos_u8(&source.pixels, source_h, source_w, target_h, target_w)?;
        let data: Vec<f32> = resized.iter().map(|&value| value / 127.5 - 1.0).collect();
        let hwc = Tensor::from_vec(data, (target_h, target_w, 3), &self.device)?;
        let nchw = hwc
            .permute((2, 0, 1))?
            .unsqueeze(0)?
            .contiguous()?
            .to_dtype(DTYPE)?;
        Ok(self.vae_encoder.encode_mean(&nchw)?)
    }
}

/// Match the authoritative MLX img2img schedule: strength selects a truncated prefix length.
/// Rust's float-to-integer cast floors finite non-negative values, which is deliberate here.
fn ancestral_strength_steps(steps: usize, strength: f32) -> usize {
    (steps as f32 * strength) as usize
}

fn validate_detail_request(req: &SdxlDetailRequest, source: &Image, control: &Image) -> Result<()> {
    if req.steps == 0 {
        return Err(CandleError::Msg(
            "sdxl detail: steps must be at least 1".to_owned(),
        ));
    }
    if req.width == 0
        || req.height == 0
        || !req.width.is_multiple_of(SIZE_MULTIPLE)
        || !req.height.is_multiple_of(SIZE_MULTIPLE)
    {
        return Err(CandleError::Msg(format!(
            "sdxl detail: width/height must be nonzero multiples of {SIZE_MULTIPLE} (got {}x{})",
            req.width, req.height
        )));
    }
    if !req.guidance.is_finite() || req.guidance < 1.0 {
        return Err(CandleError::Msg(
            "sdxl detail: guidance must be finite and at least 1".to_owned(),
        ));
    }
    if !req.strength.is_finite() || !(0.0..=1.0).contains(&req.strength) {
        return Err(CandleError::Msg(
            "sdxl detail: strength must be finite and in [0, 1]".to_owned(),
        ));
    }
    if !req.control_scale.is_finite() || req.control_scale < 0.0 {
        return Err(CandleError::Msg(
            "sdxl detail: control_scale must be finite and non-negative".to_owned(),
        ));
    }
    for (name, image) in [("source", source), ("control", control)] {
        let expected = candle_gen::gen_core::imageops::checked_image_buffer_len(
            image.width as usize,
            image.height as usize,
            3,
        )
        .unwrap_or(usize::MAX);
        if image.pixels.len() != expected {
            return Err(CandleError::Msg(format!(
                "sdxl detail: {name} pixel buffer {} != {}x{}x3",
                image.pixels.len(),
                image.width,
                image.height
            )));
        }
    }
    if (control.width, control.height) != (req.width, req.height) {
        return Err(CandleError::Msg(format!(
            "sdxl detail: control image is {}x{} but request is {}x{}",
            control.width, control.height, req.width, req.height
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: u32, height: u32) -> Image {
        Image {
            width,
            height,
            pixels: vec![127; (width * height * 3) as usize],
        }
    }

    #[test]
    fn detail_contract_accepts_exact_tile_and_rejects_silent_drops() {
        let source = image(64, 64);
        let control = image(64, 64);
        let mut req = SdxlDetailRequest {
            width: 64,
            height: 64,
            ..Default::default()
        };
        validate_detail_request(&req, &source, &control).unwrap();

        req.steps = 0;
        assert!(validate_detail_request(&req, &source, &control)
            .unwrap_err()
            .to_string()
            .contains("steps"));
        req.steps = 1;
        req.strength = f32::NAN;
        assert!(validate_detail_request(&req, &source, &control).is_err());
        req.strength = 0.5;
        req.control_scale = f32::INFINITY;
        assert!(validate_detail_request(&req, &source, &control).is_err());
        req.control_scale = 0.7;
        assert!(validate_detail_request(&req, &source, &image(56, 64))
            .unwrap_err()
            .to_string()
            .contains("control image"));
    }

    #[test]
    fn detail_contract_allows_resized_source_but_requires_valid_rgb() {
        let req = SdxlDetailRequest {
            width: 64,
            height: 64,
            ..Default::default()
        };
        validate_detail_request(&req, &image(32, 48), &image(64, 64)).unwrap();
        let bad = Image {
            width: 32,
            height: 48,
            pixels: vec![0; 3],
        };
        assert!(validate_detail_request(&req, &bad, &image(64, 64))
            .unwrap_err()
            .to_string()
            .contains("source pixel buffer"));
    }

    #[test]
    fn detail_strength_schedule_truncates_like_mlx() {
        assert_eq!(ancestral_strength_steps(30, 0.55), 16);
        assert_eq!(ancestral_strength_steps(30, 0.0), 0);
        assert_eq!(ancestral_strength_steps(30, 1.0), 30);
        assert_eq!(ancestral_strength_steps(1, 0.999), 0);
    }
}
