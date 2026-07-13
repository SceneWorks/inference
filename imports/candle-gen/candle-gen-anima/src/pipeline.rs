//! The end-to-end Anima txt2img pipeline — the candle transcription of `mlx-gen-anima`'s
//! `pipeline.rs`: prompt → (Qwen3 encode → mask-multiply → conditioner) → DiT denoise (flow-match) →
//! VAE decode → image.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::gen_core::sampling::TimestepConvention;
use candle_gen::gen_core::{runtime::CancelFlag, Image, Progress, WeightsSource};
use candle_gen::{CandleError, Result};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::{Variant, SIGMA_SHIFT, VAE_CHANNELS, VAE_COMPRESSION};
use crate::loader::AnimaComponents;

/// Anima's default flow solver: the recommended **ER-SDE-3** (`er_sde`, sc-10519), matching the MLX
/// lane. The workspace `gen-core` is pinned to `441ecec` (mlx-gen PR #673's CI-green head), which
/// carries the `ErSde` solver in the curated menu — so this is a real curated sampler, not a silent
/// fallback. A request `sampler` overrides it; any curated flow solver (euler, dpmpp_2m, …) is valid.
pub const DEFAULT_SAMPLER: &str = "er_sde";

/// The Anima sigma schedule: `linspace(1.0, 1/N, N)` (**NOT** the diffusers default) time-shifted by
/// the static `shift=3.0` (`3σ / (1 + 2σ)`), with the trailing terminal `0.0`. Length `N + 1`, descending.
pub fn anima_sigmas(steps: usize) -> Vec<f32> {
    let n = steps.max(1);
    let shift = SIGMA_SHIFT as f64;
    let mut sigmas: Vec<f32> = (0..n)
        .map(|i| {
            // linspace(1.0, 1/n, n)
            let s = if n == 1 {
                1.0
            } else {
                1.0 + (i as f64) * (1.0 / n as f64 - 1.0) / ((n - 1) as f64)
            };
            // static time-shift: shift·s / (1 + (shift−1)·s)
            (shift * s / (1.0 + (shift - 1.0) * s)) as f32
        })
        .collect();
    sigmas.push(0.0);
    sigmas
}

/// Seeded initial latent noise `[1, 16, 1, H/8, W/8]` (f32 standard normal, CPU-drawn for launch
/// portability — sc-3673), the 5-D Cosmos latent, moved to `device`.
fn create_noise(seed: u64, width: u32, height: u32, device: &Device) -> Result<Tensor> {
    let lat_h = (height / VAE_COMPRESSION) as usize;
    let lat_w = (width / VAE_COMPRESSION) as usize;
    let n = VAE_CHANNELS * lat_h * lat_w;
    let mut rng = StdRng::seed_from_u64(seed);
    let data = candle_gen::seeded_normal_vec(&mut rng, n);
    Ok(
        Tensor::from_vec(data, (1, VAE_CHANNELS, 1, lat_h, lat_w), &Device::Cpu)?
            .to_device(device)?,
    )
}

/// Per-generation options.
pub struct GenOptions {
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    pub guidance: f32,
    pub seed: u64,
    /// Curated sampler name; `None` ⇒ [`DEFAULT_SAMPLER`].
    pub sampler: Option<String>,
}

/// The assembled Anima pipeline.
pub struct AnimaPipeline {
    components: AnimaComponents,
    device: Device,
}

impl AnimaPipeline {
    pub fn from_source(
        source: &WeightsSource,
        variant: Variant,
        device: &Device,
        adapters: &[candle_gen::gen_core::AdapterSpec],
    ) -> Result<Self> {
        Ok(Self {
            components: AnimaComponents::load(source, variant, device, adapters)?,
            device: device.clone(),
        })
    }

    pub fn components(&self) -> &AnimaComponents {
        &self.components
    }

    /// Encode a prompt to the DiT's `encoder_hidden_states` `[1, 512, 1024]`: Qwen3 `last_hidden_state`
    /// → **mask-multiply** (VERIFIED trap) → `AnimaTextConditioner`.
    pub fn encode_prompt(&self, prompt: &str) -> Result<Tensor> {
        let c = &self.components;
        let dtype = c.dtype;
        let (qwen_ids, qwen_mask) = c.tokenizers.encode_qwen(prompt)?;
        let s = qwen_ids.len();
        let ids_u32: Vec<u32> = qwen_ids.iter().map(|&i| i as u32).collect();
        let input_ids = Tensor::from_vec(ids_u32, (1, s), &self.device)?;
        let source = c.text_encoder.forward(&input_ids, dtype)?; // [1, S, 1024]

        // Multiply the Qwen states by the attention mask BEFORE the conditioner (zeros padded/uncond
        // tokens) — the flagged trap. Batch-1 real prompts have an all-ones mask (no-op); the empty
        // uncond prompt's single token (mask 0) is zeroed so the conditioner cross-attn contributes 0.
        let mask_f: Vec<f32> = qwen_mask.iter().map(|&m| m as f32).collect();
        let mask = Tensor::from_vec(mask_f, (1, s, 1), &self.device)?.to_dtype(dtype)?;
        let source = source.broadcast_mul(&mask)?;

        let t5_ids = c.tokenizers.encode_t5(prompt)?;
        let st = t5_ids.len();
        let t5_u32: Vec<u32> = t5_ids.iter().map(|&i| i as u32).collect();
        let target_ids = Tensor::from_vec(t5_u32, (1, st), &self.device)?;
        c.conditioner.forward(&source, &target_ids, dtype)
    }

    /// Generate one image. `negative` is used only when `variant.uses_cfg()`.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        prompt: &str,
        negative: &str,
        variant: Variant,
        opts: &GenOptions,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let dtype = self.components.dtype;
        let cond = self.encode_prompt(prompt)?;
        let uncond = if variant.uses_cfg() {
            Some(self.encode_prompt(negative)?)
        } else {
            None
        };

        let sigmas = anima_sigmas(opts.steps);
        // Keep the initial latent in **f32** — the sampler integrates in f32 (`predict` returns the
        // velocity as f32; the VAE is loaded + decodes in f32), and the DiT casts its input to the
        // compute dtype internally (`CosmosDiT::forward`). Casting the noise to `dtype` here is a no-op
        // on the CPU/f32 parity lane but yields a **bf16** latent on the GPU lanes, so the sampler's
        // `x + (σ_next − σ)·v` add mixes a bf16 `x` with an f32 `v` → "dtype mismatch in add" on the
        // very first step. This never fired on the CPU-only goldens; it only bites on real CUDA/Metal
        // (sc-10625). Latents stay f32 end-to-end; only the DiT forward runs in `dtype`.
        let noise = create_noise(opts.seed, opts.width, opts.height, &self.device)?;
        let guidance = opts.guidance as f64;
        let sampler = opts.sampler.as_deref().or(Some(DEFAULT_SAMPLER));

        let dit = &self.components.dit;
        // The DiT is a **standard flow denoiser**: it predicts the flow velocity `v ≈ ε − x0` and
        // embeds the **raw σ** as its timestep. So `run_flow_sampler` (`TimestepConvention::Sigma`,
        // integrating `x + (σ_next − σ)·v`) consumes the DiT output directly — no negation, no `1 − σ`.
        let predict = |x: &Tensor, sigma: f32| -> Result<Tensor> {
            let s = Tensor::from_vec(vec![sigma], (1,), &self.device)?;
            let v_cond = dit.forward(x, &s, &cond, dtype)?;
            let v = match &uncond {
                // CFG: v = v_uncond + guidance·(v_cond − v_uncond).
                Some(u) => {
                    let v_u = dit.forward(x, &s, u, dtype)?;
                    (&v_u + ((v_cond - &v_u)? * guidance)?)?
                }
                None => v_cond,
            };
            // Integrate in f32 (the reference keeps latents f32).
            Ok(v.to_dtype(DType::F32)?)
        };

        let latent = candle_gen::run_flow_sampler(
            sampler,
            TimestepConvention::Sigma,
            &sigmas,
            noise,
            opts.seed,
            cancel,
            on_progress,
            predict,
        )?;

        on_progress(Progress::Decoding);
        // The Cosmos latent is 5-D `[1,16,1,H/8,W/8]`; the QwenVae decode is NCHW — drop the length-1
        // temporal axis. VAE applies the baked latents_mean/std de-norm → `[1,3,H,W]` f32 in `[-1,1]`.
        let latent_nchw = latent.squeeze(2)?;
        let decoded = self.components.vae.decode(&latent_nchw)?;
        to_image(&decoded)
    }
}

/// `[1, 3, H, W]` in `[-1, 1]` → an 8-bit RGB [`Image`].
fn to_image(decoded: &candle_gen::candle_core::Tensor) -> Result<Image> {
    use candle_gen::candle_core::IndexOp;
    let img = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(DType::U8)?;
    let img = img.i(0)?.to_device(&Device::Cpu)?;
    let (c, h, w) = img.dims3()?;
    if c != 3 {
        return Err(CandleError::Msg(format!(
            "anima: expected 3 channels, got {c}"
        )));
    }
    let pixels = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_schedule_linspace_shift3() {
        // N=10 ⇒ linspace(1.0, 0.1, 10) time-shifted by 3.0, trailing 0.0. Length 11.
        let s = anima_sigmas(10);
        assert_eq!(s.len(), 11);
        assert!((s[0] - 1.0).abs() < 1e-6, "s0={}", s[0]);
        assert!((s[1] - (2.7 / 2.8)).abs() < 1e-5, "s1={}", s[1]);
        assert!((s[9] - 0.25).abs() < 1e-5, "s9={}", s[9]);
        assert_eq!(s[10], 0.0);
        for w in s.windows(2) {
            assert!(w[0] > w[1], "not descending: {} !> {}", w[0], w[1]);
        }
    }

    #[test]
    fn sigma_schedule_turbo_10_and_base_30_lengths() {
        assert_eq!(anima_sigmas(10).len(), 11);
        assert_eq!(anima_sigmas(30).len(), 31);
        assert_eq!(anima_sigmas(1), vec![1.0, 0.0]); // shift(1.0)=1.0
    }
}
