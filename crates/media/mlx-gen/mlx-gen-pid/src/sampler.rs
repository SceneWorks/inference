//! The distilled few-step pixel-diffusion sampler — faithful port of
//! `pid_distill_model.py::_student_sample_loop` (+ `_velocity_to_x0`). The released students run the
//! **SDE / velocity-prediction** schedule (`student_t_list=[0.999,0.866,0.634,0.342,0.0]`,
//! `fm_timescale=1000`, cfg 1 — no classifier-free guidance). PiD denoises directly in high-res
//! **pixel** space: `noise`/`x` are `[B, 3, H, W]` at the *output* resolution, conditioned on the LQ
//! latent + caption + degrade σ.
//!
//! Per step `(t_cur, t_next)`: `v = net(x, t_cur·timescale, …)`, `x0 = x − t_cur·v`; then for an SDE
//! interior step `x = (1−t_next)·x0 + t_next·ε` (fresh noise), and the final `t_next=0` step takes
//! `x = x0`. Output is clamped to `[-1, 1]`.
//!
//! The step math is RNG-free and deterministic — [`Sampler::run`] takes the initial noise and the
//! per-step ε injected, so it parity-tests bit-for-bit against the torch loop. [`Sampler::sample`]
//! draws them from MLX's PRNG for production (cross-backend RNG does not match torch — a same-backend
//! decode, per the repo's full-trajectory chaos note).

use mlx_rs::ops::{add, clip, multiply, subtract};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array};

use mlx_gen::array::scalar;
use mlx_gen::{CancelFlag, Error, Result};

use crate::config::{SampleType, SamplerConfig};
use crate::lq::PidNet;
use crate::tiling::{forward_tiled, FeatherCache};

/// The distilled few-step sampler.
pub struct Sampler {
    t_list: Vec<f32>,
    timescale: f32,
    sde: bool,
    /// Per-decode cache of the tiled path's feather-weight tables (F-153) — deterministic across the
    /// 4 steps, so each tile shape's feather is built once. Lives here (one `Sampler` per decode) so it
    /// persists across the per-step `forward_tiled` calls without leaking across generations.
    feather_cache: FeatherCache,
}

impl Sampler {
    pub fn new(cfg: &SamplerConfig) -> Self {
        Self {
            t_list: cfg.student_t_list.clone(),
            timescale: cfg.fm_timescale,
            sde: cfg.sample_type == SampleType::Sde,
            feather_cache: FeatherCache::default(),
        }
    }

    /// Host metadata from this actual sampler; no model execution is needed to test the receipt.
    fn observe_execution(&self, seed: u64) {
        mlx_gen::image_execution::scalar("pid_timescale", self.timescale);
        mlx_gen::image_execution::policy_debug("pid_epsilon_draws", &self.num_eps());
        mlx_gen::image_execution::policy_debug("pid_output_clamp", &"[-1,1]");
        mlx_gen::image_execution::policy_debug("pid_dtypes", &"network=bf16;noise=f32");
        mlx_gen::image_execution::sampled_decode(
            if self.sde { "pid_sde" } else { "pid_ode" }.into(),
            &self.t_list, seed,
            "mlx-split-key-normal-v1;noise=split(key(seed),2).0;epsilon=repeated-split(rest,2).0;full-image-noise-independent-of-tiling",
        );
    }

    /// Number of denoising steps (`len(t_list) − 1`).
    pub fn steps(&self) -> usize {
        self.t_list.len().saturating_sub(1)
    }

    /// Number of fresh-noise draws the SDE loop consumes (one per interior step with `t_next>0`).
    pub fn num_eps(&self) -> usize {
        if !self.sde {
            return 0;
        }
        (1..self.t_list.len())
            .filter(|&i| self.t_list[i] > 0.0)
            .count()
    }

    /// velocity-prediction `x0 = x − t·v`.
    fn velocity_to_x0(x: &Array, v: &Array, t: f32) -> Result<Array> {
        Ok(subtract(x, &multiply(v, scalar(t))?)?)
    }

    /// Deterministic loop with the initial `noise` and the per-step `eps` injected (one ε per SDE
    /// interior step, in order). `caption`/`lq_latent`/`sigma` condition the net every step.
    ///
    /// `cancel` is the cooperative cancellation handle (F-006): checked at each of the ~4 step
    /// boundaries and forced-eval'd there so a cancel actually interrupts this multi-second decode
    /// (MLX is lazy — without the per-step `eval` the whole graph would schedule at once and a
    /// mid-loop check could never observe the trip). Returns [`Error::Canceled`] on trip.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        net: &PidNet,
        noise: &Array,
        eps: &[Array],
        caption: &Array,
        lq_latent: &Array,
        sigma: &Array,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        self.run_inner(noise, eps, cancel, |x, t_scaled| {
            net.forward(x, t_scaled, caption, lq_latent, sigma)
        })
    }

    /// Like [`Self::run`] but the per-step **velocity** forward is spatially tiled (sc-10087):
    /// [`crate::tiling::forward_tiled`] runs the net on overlapping `tile`-px pixel windows and
    /// feather-blends them, so the whole-image `PidNet::forward` peak (and its single long Metal command
    /// buffer) never materializes. The 4-step SDE loop stays whole-image — `noise`/`eps` are the same
    /// full-res seeded draws, so the sampler math + RNG sequence are unchanged and only the forward is
    /// approximated. `overlap` is the feather width (px).
    #[allow(clippy::too_many_arguments)]
    pub fn run_tiled(
        &self,
        net: &PidNet,
        noise: &Array,
        eps: &[Array],
        caption: &Array,
        lq_latent: &Array,
        sigma: &Array,
        tile: i32,
        overlap: i32,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        self.run_inner(noise, eps, cancel, |x, t_scaled| {
            forward_tiled(
                net,
                x,
                t_scaled,
                caption,
                lq_latent,
                sigma,
                tile,
                overlap,
                &self.feather_cache,
            )
        })
    }

    /// Shared SDE step loop. `forward(x, t_scaled) -> v` is the per-step velocity predictor — either the
    /// whole-image `PidNet::forward` ([`Self::run`]) or the tiled forward ([`Self::run_tiled`]). The step
    /// math, ε injection, cancel handling, and per-step `eval` are identical between the two paths.
    fn run_inner(
        &self,
        noise: &Array,
        eps: &[Array],
        cancel: Option<&CancelFlag>,
        forward: impl Fn(&Array, &Array) -> Result<Array>,
    ) -> Result<Array> {
        // F-100: the SDE loop consumes one `eps[ei]` per interior step; a caller that supplies fewer
        // than `num_eps()` draws would panic OOB mid-loop. Validate the contract up front.
        if eps.len() < self.num_eps() {
            return Err(Error::Msg(format!(
                "pid sampler: need {} eps draws for this schedule, got {}",
                self.num_eps(),
                eps.len()
            )));
        }
        let b = noise.shape()[0];
        let mut x = noise.clone();
        let mut ei = 0usize;
        for i in 0..self.steps() {
            if cancel.is_some_and(CancelFlag::is_cancelled) {
                return Err(Error::Canceled);
            }
            let t_cur = self.t_list[i];
            let t_next = self.t_list[i + 1];
            let t_scaled = Array::from_slice(&vec![t_cur * self.timescale; b as usize], &[b]);
            let v = forward(&x, &t_scaled)?;
            if t_next > 0.0 {
                if self.sde {
                    let x0 = Self::velocity_to_x0(&x, &v, t_cur)?;
                    let e = &eps[ei];
                    ei += 1;
                    x = add(
                        &multiply(&x0, scalar(1.0 - t_next))?,
                        &multiply(e, scalar(t_next))?,
                    )?;
                } else {
                    // ODE: x = x + (t_next − t_cur)·v (velocity prediction).
                    x = add(&x, &multiply(&v, scalar(t_next - t_cur))?)?;
                }
            } else {
                x = Self::velocity_to_x0(&x, &v, t_cur)?;
            }
            // Materialize this step so a cancel between steps is actually observed (MLX is lazy),
            // and to bound transient peak memory (F-013): the 4-step graph no longer schedules at once.
            eval([&x])?;
        }
        Ok(clip(&x, (-1.0, 1.0))?)
    }

    /// Draw the seeded initial noise + per-step ε from MLX's PRNG — the single source of truth for the
    /// RNG stream both [`Self::sample`] and [`Self::sample_tiled`] consume (F-155). The two production
    /// entries MUST see byte-for-byte the same draw (the tiled-vs-whole A/B invariant, sc-10087), so the
    /// key-split order lives here once rather than duplicated verbatim in each. Full-resolution
    /// `[b, 3, h, w]` regardless of tiling (only the *forward* is tiled), so the sequence is unchanged.
    fn draw_noise_eps(&self, b: i32, h: i32, w: i32, seed: u64) -> Result<(Array, Vec<Array>)> {
        let (k_noise, mut k_rest) = random::split(&random::key(seed)?, 2)?;
        let noise = random::normal::<f32>(&[b, 3, h, w], None, None, Some(&k_noise))?;
        let mut eps = Vec::with_capacity(self.num_eps());
        for _ in 0..self.num_eps() {
            let (k_e, k_n) = random::split(&k_rest, 2)?;
            eps.push(random::normal::<f32>(
                &[b, 3, h, w],
                None,
                None,
                Some(&k_e),
            )?);
            k_rest = k_n;
        }
        Ok((noise, eps))
    }

    /// Production entry: draw the initial noise + per-step ε from MLX's PRNG (seeded), then run the
    /// loop. Returns clamped pixels `[B, 3, H, W]`. `cancel` is threaded into [`Self::run`] (F-006).
    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        net: &PidNet,
        caption: &Array,
        lq_latent: &Array,
        sigma: &Array,
        b: i32,
        h: i32,
        w: i32,
        seed: u64,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        let (noise, eps) = self.draw_noise_eps(b, h, w, seed)?;
        let out = self.run(net, &noise, &eps, caption, lq_latent, sigma, cancel)?;
        mlx_gen::image_execution::policy_debug("pid_forward_tiled", &false);
        self.observe_execution(seed);
        Ok(out)
    }

    /// Tiled production entry (sc-10087): identical seeded noise/ε draw as [`Self::sample`] (full-res, so
    /// the RNG sequence is byte-for-byte the same), then [`Self::run_tiled`] with the spatial `tile` /
    /// `overlap`. Use for large outputs where the whole-image forward overflows the memory /
    /// command-buffer envelope. `tile`/`overlap` are output-pixel units (rounded to the pixel→latent
    /// factor internally).
    #[allow(clippy::too_many_arguments)]
    pub fn sample_tiled(
        &self,
        net: &PidNet,
        caption: &Array,
        lq_latent: &Array,
        sigma: &Array,
        b: i32,
        h: i32,
        w: i32,
        seed: u64,
        tile: i32,
        overlap: i32,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        let (noise, eps) = self.draw_noise_eps(b, h, w, seed)?;
        let out = self.run_tiled(
            net, &noise, &eps, caption, lq_latent, sigma, tile, overlap, cancel,
        )?;
        self.observe_execution(seed);
        Ok(out)
    }
}

#[cfg(test)]
mod execution_tests {
    use super::*;
    use mlx_gen::gen_core::generator::{ExecutionFact, ExecutionRandomness, NumericScheduleDigest};
    use mlx_gen::{Capabilities, GenerationOutput, Image, Modality, ModelDescriptor};

    #[test]
    fn completed_pid_observation_uses_sampler_schedule_and_shared_decoder_seed() {
        let descriptor = ModelDescriptor {
            id: "pid_probe",
            family: "probe",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities::default(),
            encoder_contract: None,
            denoiser_output_latent_space: None,
            control_kinds: None,
            required_components: &[],
        };
        for sample_type in [SampleType::Sde, SampleType::Ode] {
            let mut cfg = SamplerConfig::distill_4step();
            cfg.sample_type = sample_type;
            cfg.student_t_list = vec![0.9, 0.4, 0.0];
            cfg.fm_timescale = 321.0;
            let sampler = Sampler::new(&cfg);
            let report = mlx_gen::image_execution::complete_image_attempt(
                &descriptor,
                &CancelFlag::new(),
                || {
                    for seed in [9, 10] {
                        mlx_gen::image_execution::sampled_native(
                            "euler",
                            "Sigma",
                            mlx_rs::Dtype::Float32,
                            &[1.0, 0.0],
                            seed,
                        );
                    }
                    // Production shares one PiD decoder (and therefore its seed) across the count loop.
                    sampler.observe_execution(9);
                    sampler.observe_execution(9);
                    Ok(GenerationOutput::Images(vec![
                        Image {
                            width: 1,
                            height: 1,
                            pixels: vec![0; 3]
                        };
                        2
                    ]))
                },
            )
            .unwrap();
            let resolved = report.execution.resolved().unwrap();
            let ExecutionRandomness::Draws { draws, scheme } = &resolved.randomness else {
                panic!()
            };
            assert_eq!(
                draws
                    .iter()
                    .map(|draw| (draw.output_index, draw.seed))
                    .collect::<Vec<_>>(),
                vec![(0, Some(9)), (1, Some(10)), (0, Some(9)), (1, Some(9))]
            );
            assert!(scheme.contains("full-image-noise-independent-of-tiling"));
            assert_eq!(
                resolved.provider_choices["pid_timescale"],
                ExecutionFact::f32_bits(321.0).unwrap()
            );
            let ExecutionFact::Resolved { value: stages } =
                &resolved.provider_choices["auxiliary_decode"]
            else {
                panic!()
            };
            assert_eq!(
                stages[0]["schedule"]["sha256"],
                NumericScheduleDigest::f32_values(&cfg.student_t_list)
                    .unwrap()
                    .sha256
            );
            assert_eq!(
                stages[0]["sampling"],
                if sample_type == SampleType::Sde {
                    "pid_sde"
                } else {
                    "pid_ode"
                }
            );
        }
    }
}
