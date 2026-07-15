//! Mochi 1 text-to-video pipeline — component load + the flow-match **true-CFG** denoise loop +
//! AsymmVAE decode.
//!
//! [`denoise`] runs `MochiPipeline`'s sampling loop: at each step the seeded latent is doubled into a
//! `[neg, pos]` CFG batch, the AsymmDiT predicts the velocity for both branches, true CFG recombines
//! them (`uncond + g·(cond − uncond)`, inside the predict closure), and a 1st-order Euler step advances
//! the latent. `req.cancel` is honored at the top of every step and each Euler step forces a
//! materialization (candle is eager), so cancellation lands promptly and the streamed [`Progress::Step`]
//! reflects real work.
//!
//! The 3-D RoPE / positions are built **inside** [`MochiTransformer3DModel::forward`] from the geometry,
//! so the pipeline only supplies the latent + conditioning.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::{self, CancelFlag, GenerationRequest, Image, Progress};
use candle_gen::{CandleError, Result as CResult};
use rand::rngs::StdRng;
use rand::SeedableRng;
use tokenizers::Tokenizer;

use crate::config::{MochiConfig, MochiVaeConfig};
use crate::scheduler::{cfg_combine, MochiScheduler};
use crate::text_encoder::{encode_prompt, MochiT5};
use crate::transformer::{load_transformer_var_builder, MochiDitConfig, MochiTransformer3DModel};
use crate::vae::MochiVaeDecoder;
use crate::{DEFAULT_FPS, DEFAULT_FRAMES, DEFAULT_GUIDANCE, DEFAULT_STEPS, DIT_DTYPE};

/// AsymmVAE latent channels fed to the DiT / seeded as init noise.
const LATENT_CHANNELS: usize = 12;
/// Resolution `shift` for the flow-match schedule — Mochi config `shift = 1.0` (no shift).
const MOCHI_SHIFT: f32 = 1.0;

/// The loaded Mochi 1 model, held resident for the whole generation. `Arc`-wrapped so the lazy
/// component cache clones cheaply.
#[derive(Clone)]
pub struct Components {
    tokenizer: Arc<Tokenizer>,
    t5: Arc<MochiT5>,
    transformer: Arc<MochiTransformer3DModel>,
    vae: Arc<MochiVaeDecoder>,
    vae_cfg: MochiVaeConfig,
}

/// The lazy pipeline handle (root + device); [`load_components`](Pipeline::load_components) assembles
/// the heavy components on first use.
pub struct Pipeline {
    root: PathBuf,
    device: Device,
}

impl Pipeline {
    pub fn new(root: &Path, device: &Device) -> Self {
        Self {
            root: root.to_path_buf(),
            device: device.clone(),
        }
    }

    /// Assemble the full Mochi model from a snapshot directory: the reused T5-XXL encoder + tokenizer
    /// (bf16), the AsymmDiT transformer (bf16-stored, f32-activation), and the AsymmVAE decoder (f32).
    pub fn load_components(&self) -> CResult<Components> {
        // MochiConfig is currently config-as-code (the T5 geometry is fixed by the reused encoder); the
        // call keeps the snapshot-driven seam for symmetry with the VAE config.
        let _cfg = MochiConfig::from_model_dir(&self.root)?;
        let vae_cfg = MochiVaeConfig::from_model_dir(&self.root)?;
        let dit_cfg = MochiDitConfig::default();

        let tokenizer = crate::tokenizer::load_tokenizer()?;
        let t5 = MochiT5::load(&self.root.join("text_encoder"), DIT_DTYPE, &self.device)?;
        let dit_vb = load_transformer_var_builder(&self.root, DIT_DTYPE, &self.device)?;
        let transformer = MochiTransformer3DModel::new(dit_vb, &dit_cfg, &self.device)?;
        let vae = MochiVaeDecoder::load(&self.root, &self.device)?;

        Ok(Components {
            tokenizer: Arc::new(tokenizer),
            t5: Arc::new(t5),
            transformer: Arc::new(transformer),
            vae: Arc::new(vae),
            vae_cfg,
        })
    }

    /// Render: T5-XXL masked encode (positive + CFG unconditional) → seeded latents → the flow-match
    /// true-CFG denoise loop → VAE decode → per-frame `Image`s. Returns `(frames, fps)`.
    pub fn render(
        &self,
        req: &GenerationRequest,
        comps: &Components,
        on_progress: &mut dyn FnMut(Progress),
    ) -> CResult<(Vec<Image>, u32)> {
        if req.cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }

        // T5-XXL masked encode (`_get_t5_prompt_embeds`), positive + the CFG-unconditional branch.
        let pos = encode_prompt(&comps.tokenizer, &comps.t5, &req.prompt, &self.device)?;
        let neg_prompt = req.negative_prompt.as_deref().unwrap_or("");
        let neg = encode_prompt(&comps.tokenizer, &comps.t5, neg_prompt, &self.device)?;
        // CFG batch order [neg, pos] — matches `scheduler::cfg_combine` (uncond = half 0, cond = half 1).
        let enc = Tensor::cat(&[&neg.prompt_embeds, &pos.prompt_embeds], 0)?;
        let enc_mask = Tensor::cat(&[&neg.prompt_attention_mask, &pos.prompt_attention_mask], 0)?;

        // Geometry: AsymmVAE 6× temporal / 8× spatial; the DiT sees the `[1, 12, F_lat, H/8, W/8]`
        // latent (frames already gated to `1 + 6·k`, size to multiple-of-16 by `validate`).
        let t_ratio = comps.vae_cfg.temporal_compression_ratio() as u32;
        let s_ratio = comps.vae_cfg.spatial_compression_ratio() as u32;
        let frames = req.frames.unwrap_or(DEFAULT_FRAMES);
        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let lf = 1 + (frames - 1) / t_ratio;
        let lh = req.height / s_ratio;
        let lw = req.width / s_ratio;

        // Seeded init noise `[1, 12, F_lat, H_lat, W_lat]` (FlowMatchEuler `init_noise_sigma = 1`, used
        // unscaled). The draw is CPU-`StdRng` (launch-portable); the reference's `torch.Generator` RNG is
        // not portable, so the e2e parity gate teacher-forces the init latents instead.
        let seed = req.seed.unwrap_or_else(gen_core::default_seed);
        let init = seeded_latents(seed, lf as usize, lh as usize, lw as usize, &self.device)?;

        let latents = denoise(
            &comps.transformer,
            &init,
            &enc,
            &enc_mask,
            steps,
            guidance,
            MOCHI_SHIFT,
            &req.cancel,
            on_progress,
        )?;

        on_progress(Progress::Decoding);
        let video = comps.vae.decode(&latents)?; // [1, 3, F, H, W], ~[-1, 1]
        let frames_arr = to_uint8_frames(&video)?; // [F, H, W, 3] u8
        let images = frames_to_images(&frames_arr)?;
        Ok((images, req.fps.unwrap_or(DEFAULT_FPS)))
    }
}

/// Deterministic N(0,1) init latent `[1, 12, F_lat, H_lat, W_lat]` (f32) — CPU `StdRng`, launch-portable.
fn seeded_latents(seed: u64, lf: usize, lh: usize, lw: usize, device: &Device) -> CResult<Tensor> {
    let n = LATENT_CHANNELS * lf * lh * lw;
    let mut rng = StdRng::seed_from_u64(seed);
    let data = candle_gen::seeded_normal_vec(&mut rng, n);
    Ok(Tensor::from_vec(
        data,
        (1, LATENT_CHANNELS, lf, lh, lw),
        device,
    )?)
}

/// N-step 1st-order Euler **true-CFG** denoise (Mochi is not distilled).
///
/// `init_latents [1, C, F, H, W]` are the seeded init noise. `enc [2, L, 4096]` / `enc_mask [2, L]`
/// carry the raw T5 conditioning for the **[neg, pos]** CFG batch. `guidance` is the CFG scale; `shift`
/// the flow-match resolution shift (Mochi = 1.0). Returns the post-loop latents `[1, C, F, H, W]`.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    transformer: &MochiTransformer3DModel,
    init_latents: &Tensor,
    enc: &Tensor,
    enc_mask: &Tensor,
    steps: usize,
    guidance: f32,
    shift: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> CResult<Tensor> {
    let mut sched = MochiScheduler::new();
    sched.set_timesteps(steps, shift);
    let timesteps: Vec<f32> = sched.timesteps().to_vec();
    let device = init_latents.device();

    let mut latents = init_latents.clone();
    for (i, &t) in timesteps.iter().enumerate() {
        // Cooperative cancel at the top of every step.
        if cancel.is_cancelled() {
            return Err(CandleError::Canceled);
        }
        // latent_model_input = cat([latents, latents]) → [2, C, F, H, W] (the two CFG branches).
        let lmi = Tensor::cat(&[&latents, &latents], 0)?;
        // Same model timestep for both branches.
        let timestep = Tensor::from_vec(vec![t, t], 2, device)?;
        // AsymmDiT velocity for [neg, pos]; true-CFG recombine inside the predict step.
        let noise_pred = transformer.forward(&lmi, enc, &timestep, enc_mask)?; // [2, C, F, H, W] f32
        let velocity = cfg_combine(&noise_pred, guidance)?; // [1, C, F, H, W]
        latents = sched.step(&velocity, &latents)?;
        on_progress(Progress::Step {
            current: (i + 1) as u32,
            total: steps as u32,
        });
    }
    Ok(latents)
}

/// `(1, 3, F, H, W)` video in ~[-1, 1] → `(F, H, W, 3)` uint8. Mirrors the reference
/// `((x + 1) / 2).clamp(0, 1)·255` with a truncating cast.
pub fn to_uint8_frames(video: &Tensor) -> CResult<Tensor> {
    let (b, c, f, h, w) = video.dims5()?;
    if b != 1 {
        return Err(CandleError::Msg(format!(
            "mochi to_uint8_frames: batch size must be 1, got {b}"
        )));
    }
    // (1, 3, F, H, W) → (3, F, H, W) → (F, H, W, 3).
    let chw = video
        .reshape((c, f, h, w))?
        .permute((1, 2, 3, 0))?
        .contiguous()?;
    let half = chw.to_dtype(DType::F32)?.affine(0.5, 0.5)?; // (x + 1) / 2
    let clipped = half.clamp(0.0f32, 1.0f32)?;
    let scaled = clipped.affine(255.0, 0.0)?;
    Ok(scaled.to_dtype(DType::U8)?)
}

/// `(F, H, W, 3)` uint8 → one [`Image`] per frame.
pub fn frames_to_images(frames: &Tensor) -> CResult<Vec<Image>> {
    let dims = frames.dims();
    if dims.len() != 4 || dims[3] != 3 {
        return Err(CandleError::Msg(format!(
            "mochi frames_to_images: expected (F, H, W, 3) uint8, got {dims:?}"
        )));
    }
    let (fr, h, w) = (dims[0], dims[1], dims[2]);
    let data = frames.to_dtype(DType::U8)?.flatten_all()?.to_vec1::<u8>()?;
    let per = h * w * 3;
    Ok((0..fr)
        .map(|i| Image {
            width: w as u32,
            height: h as u32,
            pixels: data[i * per..(i + 1) * per].to_vec(),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `to_uint8_frames` clips to [0, 255] and lays frames out row-major `(F, H, W, 3)`.
    #[test]
    fn to_uint8_frames_clips_and_scales() {
        let dev = Device::Cpu;
        // 1 frame, 1×2 pixels, 3 channels: pixel0=(-2,-1,0)→((x+1)/2)=(-0.5,0,0.5)→clip→(0,0,127);
        // pixel1=(1,2,3)→(1,1.5,2)→clip→(255,255,255).
        let v = Tensor::from_vec(
            vec![-2.0f32, 1.0, -1.0, 2.0, 0.0, 3.0],
            (1, 3, 1, 1, 2),
            &dev,
        )
        .unwrap();
        let out = to_uint8_frames(&v).unwrap();
        assert_eq!(out.dims(), &[1, 1, 2, 3]);
        let px = out.flatten_all().unwrap().to_vec1::<u8>().unwrap();
        assert_eq!(px, vec![0, 0, 127, 255, 255, 255]);
    }

    /// `frames_to_images` splits `(F, H, W, 3)` into per-frame RGB8 `Image`s.
    #[test]
    fn frames_to_images_splits_per_frame() {
        let dev = Device::Cpu;
        let f = Tensor::from_vec(vec![10u8, 20, 30, 40, 50, 60], (2, 1, 1, 3), &dev).unwrap();
        let imgs = frames_to_images(&f).unwrap();
        assert_eq!(imgs.len(), 2);
        assert_eq!(imgs[0].width, 1);
        assert_eq!(imgs[0].height, 1);
        assert_eq!(imgs[0].pixels, vec![10, 20, 30]);
        assert_eq!(imgs[1].pixels, vec![40, 50, 60]);
    }
}
