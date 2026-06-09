//! Kolors T2I pipeline (sc-3094) — composes the ChatGLM3 conditioning, the leading-Euler scheduler,
//! the SDXL U-Net (with the ChatGLM context projection), real CFG, and the SDXL VAE decode.
//!
//! Mirrors diffusers `KolorsPipeline`: tokenize → ChatGLM3 `encode_prompt` (context = `hidden[-2]`,
//! pooled = `hidden[-1]` last token, with the left-padded `position_ids`) for the positive AND
//! negative prompt → CFG-batched U-Net denoise over `EulerDiscreteScheduler(leading)` → VAE decode
//! (latents / 0.13025). `time_ids` = `(H, W, 0, 0, H, W)` (the SDXL `_get_add_time_ids`).
//!
//! The whole pipeline is dtype-parametric; the parity gate (`tests/t2i_parity.rs`) runs f32.

use mlx_rs::{random, Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, DiffusionSampler, Image, Result};

use mlx_gen_sdxl::{
    decode_image, denoise, load_unet_kolors_dtype, load_vae, Autoencoder, Denoiser,
    UNet2DConditionModel,
};

use crate::chatglm3::{ChatGlmConfig, ChatGlmModel};
use crate::sampler::KolorsEulerSampler;
use crate::tokenizer::KolorsTokenizer;

/// VAE spatial downscale (latent is image/8 per side).
pub const SPATIAL_SCALE: i32 = 8;

/// A loaded Kolors model: ChatGLM3 text encoder + tokenizer + SDXL-family U-Net (with the ChatGLM
/// context projection) + SDXL VAE.
pub struct Kolors {
    chatglm: ChatGlmModel,
    tokenizer: KolorsTokenizer,
    unet: UNet2DConditionModel,
    vae: Autoencoder,
    dtype: Dtype,
}

/// The SDXL-style micro-conditioning `time_ids` = `(H, W, 0, 0, H, W)` per row (the diffusers
/// `_get_add_time_ids` for `original_size == target_size`, no crop).
fn kolors_time_ids(batch: i32, height: i32, width: i32) -> Array {
    let (h, w) = (height as f32, width as f32);
    let row = [h, w, 0.0, 0.0, h, w];
    let mut v = Vec::with_capacity(batch as usize * 6);
    for _ in 0..batch {
        v.extend_from_slice(&row);
    }
    Array::from_slice(&v, &[batch, 6])
}

impl Kolors {
    /// Load every Kolors component from the `Kwai-Kolors/Kolors-diffusers` snapshot at `dtype`.
    /// `tokenizer/tokenizer.json` must already be materialized (`tools/build_kolors_tokenizer.py`).
    pub fn load(snapshot: &std::path::Path, dtype: Dtype) -> Result<Self> {
        let te_w = Weights::from_dir(snapshot.join("text_encoder"))?;
        let chatglm = ChatGlmModel::from_weights(&te_w, ChatGlmConfig::chatglm3_6b(), None, dtype)?;
        let tokenizer = KolorsTokenizer::from_dir(snapshot.join("tokenizer"))?;
        let unet = load_unet_kolors_dtype(snapshot, dtype)?;
        let vae = load_vae(snapshot)?; // SDXL VAE (sdxl-vae-fp16-fix), f32
        Ok(Self {
            chatglm,
            tokenizer,
            unet,
            vae,
            dtype,
        })
    }

    /// Encode one prompt → `(context [1, 256, 4096], pooled [1, 4096])`, threading the tokenizer's
    /// left-padded `position_ids` into the ChatGLM3 RoPE (as `KolorsPipeline.encode_prompt` does).
    pub fn encode(&self, prompt: &str) -> Result<(Array, Array)> {
        // Kolors tokenizes the raw prompt (no chat template).
        let t = self.tokenizer.encode(prompt)?;
        self.chatglm
            .encode_prompt(&t.input_ids, &t.attention_mask, Some(&t.position_ids))
    }

    /// Decode latents `[1, h, w, 4]` → an RGB [`Image`] (`vae.decode(latents / 0.13025)`).
    pub fn decode(&self, latents: &Array) -> Result<Image> {
        decode_image(&self.vae, latents)
    }

    /// Run the CFG denoise loop from a (raw, unit-normal) initial-noise tensor `init_noise`
    /// `[1, h, w, 4]` — split out so the parity gate can feed diffusers' exact noise. `pos`/`neg` are
    /// the `(context, pooled)` from [`encode`](Self::encode). Returns the final latents `[1, h, w, 4]`.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_latents(
        &self,
        init_noise: &Array,
        pos: &(Array, Array),
        neg: &(Array, Array),
        num_steps: usize,
        cfg: f32,
        height: i32,
        width: i32,
    ) -> Result<Array> {
        use mlx_rs::ops::concatenate_axis;
        let sampler = KolorsEulerSampler::kolors(num_steps, self.dtype)?;
        // CFG batch order is [positive, negative] — `mlx_gen_sdxl::denoise` reads row 0 as the text
        // (cond) and row 1 as the uncond.
        let conditioning = concatenate_axis(&[&pos.0, &neg.0], 0)?;
        let pooled = concatenate_axis(&[&pos.1, &neg.1], 0)?;
        let time_ids = kolors_time_ids(2, height, width);
        let latents = sampler.scale_initial_noise(init_noise)?;

        let d = Denoiser {
            unet: &self.unet,
            sampler: &sampler,
        };
        let cancel = CancelFlag::new();
        denoise(
            &d,
            latents,
            &conditioning,
            &pooled,
            &time_ids,
            cfg,
            &cancel,
            &mut |_p| {},
        )
    }

    /// Full T2I: seed the RNG, draw the initial noise, encode the prompt + negative prompt, denoise,
    /// and VAE-decode. `height`/`width` are pixels (multiples of 8). `cfg` ≤ 1 disables guidance.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &str,
        negative: &str,
        num_steps: usize,
        cfg: f32,
        seed: u64,
        height: i32,
        width: i32,
    ) -> Result<Image> {
        random::seed(seed)?;
        let (lh, lw) = (height / SPATIAL_SCALE, width / SPATIAL_SCALE);
        let init_noise = random::normal::<f32>(&[1, lh, lw, 4], None, None, None)?;
        let pos = self.encode(prompt)?;
        let neg = self.encode(negative)?;
        let latents =
            self.denoise_latents(&init_noise, &pos, &neg, num_steps, cfg, height, width)?;
        self.decode(&latents)
    }
}
