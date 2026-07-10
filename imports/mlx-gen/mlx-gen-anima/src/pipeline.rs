//! The end-to-end Anima txt2img pipeline: prompt → (Qwen3 encode → mask-multiply → conditioner) →
//! DiT denoise (flow-match) → VAE decode → image. Transcribed from the diffusers Anima modular
//! pipeline (`encoders.py` / `before_denoise.py` / `denoise.py` / `decoders.py`).

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use mlx_gen::image::decoded_to_image;
use mlx_gen::media::Image;
use mlx_gen::runtime::CancelFlag;
use mlx_gen::{run_flow_sampler, Progress, Result, TimestepConvention, WeightsSource};

use crate::config::{Variant, SIGMA_SHIFT, VAE_CHANNELS, VAE_COMPRESSION};
use crate::loader::AnimaComponents;

/// Anima's recommended default sampler (the ER-SDE-3 solver added for this epic, sc-10519). A request
/// `sampler` overrides it; any curated flow solver (euler, dpmpp_2m, …) is valid.
pub const DEFAULT_SAMPLER: &str = "er_sde";

/// The Anima sigma schedule: `linspace(1.0, 1/N, N)` (**NOT** the diffusers default) time-shifted by
/// the static `shift=3.0` (`3σ / (1 + 2σ)`), with the trailing terminal `0.0` the flow sampler
/// integrates to. Length `N + 1`, descending.
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

/// Seeded initial latent noise `[1, 16, 1, H/8, W/8]` (f32 standard normal), the 5-D Cosmos latent.
fn create_noise(seed: u64, width: u32, height: u32) -> Result<Array> {
    let key = random::key(seed)?;
    let shape = [
        1,
        VAE_CHANNELS as i32,
        1,
        (height / VAE_COMPRESSION) as i32,
        (width / VAE_COMPRESSION) as i32,
    ];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?)
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
}

impl AnimaPipeline {
    pub fn from_source(source: &WeightsSource, variant: Variant) -> Result<Self> {
        Ok(Self {
            components: AnimaComponents::load(source, variant)?,
        })
    }

    pub fn components(&self) -> &AnimaComponents {
        &self.components
    }

    /// Encode a prompt to the DiT's `encoder_hidden_states` `[1, 512, 1024]` (bf16): Qwen3
    /// `last_hidden_state` → **mask-multiply** (VERIFIED trap) → `AnimaTextConditioner`.
    ///
    /// ComfyUI-style `(text:weight)` emphasis (sc-10566) applies to the **T5 query-token path only**:
    /// the Qwen tower is tokenized on the de-weighted text (its token weights are forced to `1.0`),
    /// while the parsed per-token weights scale the conditioner output. See [`crate::prompt_weight`].
    pub fn encode_prompt(&self, prompt: &str) -> Result<Array> {
        let c = &self.components;
        // Qwen is weight-blind: strip the emphasis syntax to plain text before tokenizing (a no-op for
        // an unweighted prompt). This mirrors ComfyUI forcing the Qwen token weights to 1.0.
        let qwen_text = crate::prompt_weight::strip_prompt_weights(prompt);
        let (qwen_ids, qwen_mask) = c.tokenizers.encode_qwen(&qwen_text)?;
        let source = c.text_encoder.forward(&qwen_ids, &qwen_mask)?; // [1, S, 1024] bf16
                                                                     // Multiply the Qwen states by the attention mask BEFORE the conditioner (zeros padded/uncond
                                                                     // tokens) — the flagged trap. Batch-1 real prompts have an all-ones mask (no-op); the empty
                                                                     // uncond prompt's single token (mask 0) is zeroed so the conditioner cross-attn contributes 0.
        let mask = qwen_mask.as_dtype(source.dtype())?.expand_dims(2)?; // [1, S, 1]
        let source = multiply(&source, &mask)?;
        // T5 carries the per-token weights (all 1.0 ⇒ strict no-op equal to the unweighted path).
        let (t5_ids, t5_weights) = c.tokenizers.encode_t5_weighted(prompt)?; // [1, St], len St
        c.conditioner
            .forward_weighted(&source, &t5_ids, Some(&t5_weights), source.dtype())
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
        let noise = create_noise(opts.seed, opts.width, opts.height)?;
        let sampler = opts.sampler.as_deref().unwrap_or(DEFAULT_SAMPLER);
        // VAE decode (applies the baked latents_mean/std de-norm) → [1, 3, 1, H, W] f32 in [-1, 1].
        let latent = self.denoise(
            &noise,
            prompt,
            negative,
            variant,
            opts.steps,
            opts.guidance,
            sampler,
            opts.seed,
            Dtype::Bfloat16,
            cancel,
            on_progress,
        )?;
        let decoded = self.components.vae.decode(&latent)?;
        decoded_to_image(&decoded)
    }

    /// The flow denoise loop shared by [`generate`](Self::generate) and the stage-7 parity hook. Encodes
    /// the prompt (+ negative for CFG variants), then runs `sampler` over [`anima_sigmas`] from the given
    /// `init` latent, evaluating the DiT in `dtype`. Returns the final latent `[1, 16, 1, H/8, W/8]`
    /// (f32, pre-VAE). The DiT is a **standard flow denoiser**: it predicts `v ≈ ε − x0` and embeds the
    /// **raw σ** as its timestep, so the sampler (`TimestepConvention::Sigma`, `x + (σ_next − σ)·v`)
    /// consumes it directly — no negation, no `1 − σ` timestep.
    #[allow(clippy::too_many_arguments)]
    fn denoise(
        &self,
        init: &Array,
        prompt: &str,
        negative: &str,
        variant: Variant,
        steps: usize,
        guidance: f32,
        sampler: &str,
        seed: u64,
        dtype: Dtype,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let cond = self.encode_prompt(prompt)?;
        let uncond = if variant.uses_cfg() {
            Some(self.encode_prompt(negative)?)
        } else {
            None
        };
        let sigmas = anima_sigmas(steps);
        let guidance = Array::from_slice(&[guidance], &[1]);
        let dit = &self.components.dit;
        let predict = |x: &Array, sigma: f32| -> Result<Array> {
            let s = Array::from_slice(&[sigma], &[1]);
            let v_cond = dit.forward(x, &s, &cond, dtype)?;
            let v = match &uncond {
                // CFG: v = v_uncond + guidance·(v_cond − v_uncond).
                Some(u) => {
                    let v_u = dit.forward(x, &s, u, dtype)?;
                    add(&v_u, &multiply(&subtract(&v_cond, &v_u)?, &guidance)?)?
                }
                None => v_cond,
            };
            // Integrate in f32 (the reference keeps latents f32).
            Ok(v.as_dtype(Dtype::Float32)?)
        };
        run_flow_sampler(
            Some(sampler),
            TimestepConvention::Sigma,
            &sigmas,
            init.clone(),
            seed,
            cancel,
            on_progress,
            predict,
        )
    }

    /// Test hook (sc-10524 stage-7 golden): run the flow denoise from an **injected** initial latent
    /// (instead of sampling noise) with an explicit `sampler`/`dtype`, returning the final latent
    /// `[1, 16, 1, H/8, W/8]` (pre-VAE). Lets an MLX-vs-diffusers end-to-end comparison feed BOTH sides
    /// the identical starting point + a deterministic solver, so residual drift is float error, not the
    /// chaos of two independently-sampled noises.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_from_latent(
        &self,
        init: &Array,
        prompt: &str,
        negative: &str,
        variant: Variant,
        steps: usize,
        guidance: f32,
        sampler: &str,
        dtype: Dtype,
    ) -> Result<Array> {
        let cancel = CancelFlag::default();
        let mut prog = |_p: Progress| {};
        self.denoise(
            init, prompt, negative, variant, steps, guidance, sampler, 0, dtype, &cancel, &mut prog,
        )
    }

    /// Test hook (sc-10524 stage-7 golden, intermediate-latent capture): run the same deterministic
    /// denoise as [`denoise_from_latent`], additionally snapshotting the latent AFTER each step count in
    /// `capture_after` (`x_k` = the state after `k` Euler steps). Returns `(final_latent, [(k, x_k)])`.
    ///
    /// `x_k` is exactly the input the `(k+1)`-th DiT call sees (the sampler calls `predict(x_k, σ_k)` at
    /// step `k`), so it matches the Python generator's `caps[k]` bit-for-bit in definition. Comparing the
    /// per-step deltas lets the parity test distinguish a systematic BIAS (the MLX bf16-conditioning lock —
    /// present from step 1) from diffuse float ACCUMULATION (grows with step count).
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_from_latent_capture(
        &self,
        init: &Array,
        prompt: &str,
        negative: &str,
        variant: Variant,
        steps: usize,
        guidance: f32,
        sampler: &str,
        dtype: Dtype,
        capture_after: &[usize],
    ) -> Result<(Array, Vec<(usize, Array)>)> {
        let cancel = CancelFlag::default();
        let mut prog = |_p: Progress| {};
        let cond = self.encode_prompt(prompt)?;
        let uncond = if variant.uses_cfg() {
            Some(self.encode_prompt(negative)?)
        } else {
            None
        };
        let sigmas = anima_sigmas(steps);
        let guidance = Array::from_slice(&[guidance], &[1]);
        let dit = &self.components.dit;
        let want: std::collections::HashSet<usize> = capture_after.iter().copied().collect();
        let call = std::cell::Cell::new(0usize);
        let captures: std::cell::RefCell<Vec<(usize, Array)>> = std::cell::RefCell::new(Vec::new());
        let predict = |x: &Array, sigma: f32| -> Result<Array> {
            // The sampler calls predict(x_k, σ_k) at step k, so `x` here is the state after k steps.
            let k = call.get();
            if want.contains(&k) {
                let snap = x.as_dtype(Dtype::Float32)?;
                mlx_rs::transforms::eval([&snap])?;
                captures.borrow_mut().push((k, snap));
            }
            call.set(k + 1);
            let s = Array::from_slice(&[sigma], &[1]);
            let v_cond = dit.forward(x, &s, &cond, dtype)?;
            let v = match &uncond {
                Some(u) => {
                    let v_u = dit.forward(x, &s, u, dtype)?;
                    add(&v_u, &multiply(&subtract(&v_cond, &v_u)?, &guidance)?)?
                }
                None => v_cond,
            };
            Ok(v.as_dtype(Dtype::Float32)?)
        };
        let final_latent = run_flow_sampler(
            Some(sampler),
            TimestepConvention::Sigma,
            &sigmas,
            init.clone(),
            0,
            &cancel,
            &mut prog,
            predict,
        )?;
        Ok((final_latent, captures.into_inner()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_schedule_linspace_shift3() {
        // N=10 ⇒ linspace(1.0, 0.1, 10) time-shifted by 3.0, trailing 0.0. Length 11.
        let s = anima_sigmas(10);
        assert_eq!(s.len(), 11);
        // shift(σ) = 3σ/(1+2σ): shift(1.0)=1.0, shift(0.9)=2.7/2.8, shift(0.1)=0.3/1.2=0.25.
        assert!((s[0] - 1.0).abs() < 1e-6, "s0={}", s[0]);
        assert!(
            (s[1] - (2.7 / 2.8)) < 1e-5 && (s[1] - (2.7 / 2.8)).abs() < 1e-5,
            "s1={}",
            s[1]
        );
        assert!((s[9] - 0.25).abs() < 1e-5, "s9={}", s[9]);
        assert_eq!(s[10], 0.0);
        // strictly descending (a valid flow schedule).
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
