//! SVD image-to-video pipeline (sc-3375) — the `StableVideoDiffusionPipeline` orchestration over the
//! S0–S3 components: image → CLIP `image_embeds` + noise-augmented VAE `image_latents`; a frame-wise
//! CFG denoise loop (EDM v-prediction Euler, image-latent channel-concat) with `guidance_scale =
//! linspace(min, max, num_frames)`; chunked temporal VAE decode → frames. Port of diffusers
//! `pipeline_stable_video_diffusion.py` `__call__` + `_encode_image`/`_encode_vae_image`/
//! `_get_add_time_ids`/`decode_latents`.

use mlx_rs::ops::{add, concatenate_axis, multiply, subtract, zeros_like};
use mlx_rs::Array;

use mlx_gen::array::scalar;
use mlx_gen::gen_core::{
    render::video::{run_video_steps, VideoDenoisePhase},
    MemoryRequestScope,
};
use mlx_gen::{CancelFlag, EdmModelSampling, Error, Progress, Result};

use crate::config::SchedulerConfig;
use crate::scheduler::{euler_step, scale_model_input, v_pred_denoised, EdmSchedule};
use crate::{SvdImageEncoder, SvdUnet, SvdVae};

/// Image-to-video generation parameters (the `StableVideoDiffusionPipeline.__call__` knobs).
#[derive(Clone, Debug)]
pub struct SvdParams {
    pub num_frames: i32,
    pub num_inference_steps: usize,
    pub min_guidance_scale: f32,
    pub max_guidance_scale: f32,
    pub fps: i32,
    pub motion_bucket_id: f32,
    pub noise_aug_strength: f32,
    /// Frames decoded per temporal VAE pass (diffusers default = `num_frames`).
    pub decode_chunk_size: i32,
}

impl Default for SvdParams {
    fn default() -> Self {
        Self {
            num_frames: 25,
            num_inference_steps: 25,
            min_guidance_scale: 1.0,
            max_guidance_scale: 3.0,
            fps: 7,
            motion_bucket_id: 127.0,
            noise_aug_strength: 0.02,
            decode_chunk_size: 25,
        }
    }
}

/// The assembled SVD pipeline (the image encoder, VAE, and UNet + scheduler config).
pub struct SvdPipeline {
    pub image_encoder: SvdImageEncoder,
    pub vae: SvdVae,
    pub unet: SvdUnet,
    pub scheduler: SchedulerConfig,
}

impl SvdPipeline {
    pub fn new(
        image_encoder: SvdImageEncoder,
        vae: SvdVae,
        unet: SvdUnet,
        scheduler: SchedulerConfig,
    ) -> Self {
        Self {
            image_encoder,
            vae,
            unet,
            scheduler,
        }
    }

    /// The `added_time_ids` micro-conditioning row `[1, 3]` = `[fps − 1, motion_bucket_id,
    /// noise_aug_strength]` (the SVD pipeline reduces fps by 1 — the model was trained on fps−1).
    pub fn added_time_ids(params: &SvdParams) -> Array {
        Array::from_slice(
            &[
                (params.fps - 1) as f32,
                params.motion_bucket_id,
                params.noise_aug_strength,
            ],
            &[1, 3],
        )
    }

    /// The frame-wise CFG schedule `linspace(min, max, F)` shaped `[1, F, 1, 1, 1]` to broadcast over
    /// the `[1, F, H, W, 4]` latents.
    fn guidance_schedule(num_frames: i32, min_g: f32, max_g: f32) -> Array {
        let f = num_frames.max(1);
        let vals: Vec<f32> = (0..f)
            .map(|i| {
                if f == 1 {
                    min_g
                } else {
                    min_g + (max_g - min_g) * (i as f32) / ((f - 1) as f32)
                }
            })
            .collect();
        Array::from_slice(&vals, &[1, f, 1, 1, 1])
    }

    /// The frame-wise CFG v-prediction Euler denoise loop. Inputs are the **conditional** rows
    /// (`[1, …]`); the uncond CFG branch is the diffusers zeros (`image_embeds`/`image_latents` →
    /// `cat([zeros, cond])`). Returns the final `[1, F, H/8, W/8, 4]` latents.
    /// - `latents`: the seeded init noise already scaled by `init_noise_sigma` (`[1, F, H/8, W/8, 4]`).
    /// - `image_embeds`: CLIP conditioning `[1, ctx, 1024]`.
    /// - `image_latents`: per-frame-repeated VAE conditioning latent `[1, F, H/8, W/8, 4]`.
    /// - `added_time_ids`: `[1, 3]`.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        latents: &Array,
        image_embeds: &Array,
        image_latents: &Array,
        added_time_ids: &Array,
        num_frames: i32,
        steps: usize,
        min_g: f32,
        max_g: f32,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize),
    ) -> Result<Array> {
        self.denoise_scoped(
            EdmSchedule::karras(steps, &self.scheduler),
            None,
            0,
            latents,
            image_embeds,
            image_latents,
            added_time_ids,
            num_frames,
            min_g,
            max_g,
            cancel,
            None,
            &mut |event| {
                if let Progress::Step { current, .. } = event {
                    on_step(current as usize);
                }
            },
        )
    }

    /// The curated unified-sampler denoise (epic 7114, sc-7122) — the additive alternative to the native
    /// EDM Euler [`Self::denoise`], routing any curated [`mlx_gen::Solver`] through
    /// [`mlx_gen::run_curated_sampler`] over an `EdmModelSampling` (v-prediction). SVD is **sampler-only**
    /// (no scheduler axis, matching ComfyUI): it keeps its native Karras EDM σ schedule and only swaps
    /// the integrator. An explicit Euler uses the curated driver's scalar arithmetic and progress cadence.
    ///
    /// The per-frame CFG (`uncond + linspace(min,max,F)·(cond − uncond)`) lives in the predict closure,
    /// so a multi-eval solver re-applies it at each intermediate σ. `x_in` is the `c_in`-scaled latent
    /// (`gc_denoise` applied `input_scale = 1/√(σ²+1)`); `t` is the EDM `c_noise`.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_curated(
        &self,
        sampler_name: &str,
        latents: &Array,
        image_embeds: &Array,
        image_latents: &Array,
        added_time_ids: &Array,
        num_frames: i32,
        steps: usize,
        min_g: f32,
        max_g: f32,
        seed: u64,
        cancel: &CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        self.denoise_scoped(
            EdmSchedule::karras(steps, &self.scheduler),
            Some(sampler_name),
            seed,
            latents,
            image_embeds,
            image_latents,
            added_time_ids,
            num_frames,
            min_g,
            max_g,
            cancel,
            None,
            on_progress,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn denoise_scoped(
        &self,
        schedule: EdmSchedule,
        sampler: Option<&str>,
        seed: u64,
        latents: &Array,
        image_embeds: &Array,
        image_latents: &Array,
        added_time_ids: &Array,
        num_frames: i32,
        min_g: f32,
        max_g: f32,
        cancel: &CancelFlag,
        scope: Option<&mut dyn MemoryRequestScope>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Array> {
        let &[batch, frames, height, width, _] = latents.shape() else {
            return Err(Error::Msg("svd expects B,F,H,W,C latents".into()));
        };
        let phase = gen_core_family::svd::phase(
            [batch as u32, frames as u32, height as u32, width as u32],
            sampler,
            schedule.sigmas,
            seed,
        )?;
        let active = phase.enter(cancel, scope)?;
        let guide = serde_json::json!({
            "min_guidance_bits": min_g.to_bits(),
            "max_guidance_bits": max_g.to_bits(),
        });
        if let Some(name) = phase.solver {
            mlx_gen::video_execution::curated_sampling(
                "svd_curated",
                name,
                "edm_v_prediction",
                guide,
            );
            mlx_gen::video_execution::schedule("svd_curated", &phase.sigmas);
        } else {
            mlx_gen::video_execution::sampling(
                "svd_native",
                "edm_euler",
                "edm_v_prediction",
                guide,
            );
            mlx_gen::video_execution::schedule("svd_native", &phase.sigmas);
        }

        // Both solvers use one batch-2 prediction, with the unconditional row first.
        let embeds2 = concatenate_axis(&[&zeros_like(image_embeds)?, image_embeds], 0)?;
        let img_lat2 = concatenate_axis(&[&zeros_like(image_latents)?, image_latents], 0)?;
        let atid2 = concatenate_axis(&[added_time_ids, added_time_ids], 0)?;
        let guidance = Self::guidance_schedule(num_frames, min_g, max_g);
        let output = sample_edm(
            &phase,
            &schedule.timesteps,
            latents,
            cancel,
            on_progress,
            |x_in, t| {
                let lat2 = concatenate_axis(&[x_in, x_in], 0)?;
                let inp = concatenate_axis(&[&lat2, &img_lat2], -1)?;
                let pred = self.unet.forward(&inp, t, &embeds2, &atid2, num_frames)?;
                let uncond = pred.take_axis(Array::from_int(0), 0)?.expand_dims(0)?;
                let cond = pred.take_axis(Array::from_int(1), 0)?.expand_dims(0)?;
                Ok(add(
                    &uncond,
                    &multiply(&guidance, &subtract(&cond, &uncond)?)?,
                )?)
            },
        )?;
        active.finish()?;
        Ok(output)
    }

    /// Chunked temporal VAE decode (diffusers `decode_latents`): divide by `scaling_factor`, decode in
    /// `chunk`-frame windows, concat. `latents` `[1, F, H/8, W/8, 4]` → frames `[1, F, H, W, 3]`
    /// (roughly `[-1, 1]`; the caller maps to `[0, 1]` for display).
    pub fn decode(
        &self,
        latents: &Array,
        num_frames: i32,
        chunk: i32,
        cancel: &CancelFlag,
        on_chunk: &mut dyn FnMut(),
    ) -> Result<Array> {
        let sh = latents.shape();
        // The reshape below collapses `[B, F, h, w, 4] → [num_frames, h, w, 4]`, which only preserves
        // frame identity when `B == 1`; a `B > 1` caller would silently interleave `B·F` frames as
        // `F`. The generator caps `max_count = 1`, but `decode` is public — reject `B > 1` (F-029).
        if sh[0] != 1 {
            return Err(Error::Msg(format!(
                "svd decode: batch size must be 1 (got {})",
                sh[0]
            )));
        }
        let (h, w_) = (sh[2], sh[3]);
        let flat = latents.reshape(&[num_frames, h, w_, sh[4]])?; // [F,h,w,4]
        let z = multiply(&flat, scalar(1.0 / self.vae.scaling_factor()))?;
        let chunk = chunk.max(1);

        let mut start = 0;
        let mut chunks: Vec<Array> = Vec::new();
        while start < num_frames {
            // Honor the cancellation contract during the (silent, multi-second) chunked temporal
            // decode too (F-001 / L-H): a cancel arriving after denoise must still stop the job.
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            let n = chunk.min(num_frames - start);
            let idx = Array::from_slice(&(start..start + n).collect::<Vec<i32>>(), &[n]);
            let zc = z.take_axis(&idx, 0)?; // [n,h,w,4]
            let frames_chunk = self.vae.decode(&zc, n)?; // [n,H,W,3]
                                                         // Bound peak memory and surface liveness: materialize each decoded chunk and emit a
                                                         // per-chunk progress heartbeat (the decode no longer runs silently after one `Decoding`).
            mlx_rs::transforms::eval([&frames_chunk])?;
            chunks.push(frames_chunk);
            start += n;
            on_chunk();
        }
        let refs: Vec<&Array> = chunks.iter().collect();
        let frames = concatenate_axis(&refs, 0)?; // [F,H,W,3]
        let fs = frames.shape();
        Ok(frames.reshape(&[1, num_frames, fs[1], fs[2], fs[3]])?)
    }
}

/// An omitted sampler keeps the native f64-derived coefficients and timetable. Every explicit
/// name, including Euler, uses the curated EDM conversion and its per-evaluation progress.
fn sample_edm(
    phase: &VideoDenoisePhase<(), Option<&str>>,
    timesteps: &[f32],
    latents: &Array,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
    mut predict: impl FnMut(&Array, f32) -> Result<Array>,
) -> Result<Array> {
    if let Some(name) = phase.solver {
        return mlx_gen::run_curated_sampler(
            mlx_gen::SamplerRun {
                sampler_name: Some(name),
                sigmas: &phase.sigmas,
                seed: phase.seed,
                cancel,
                on_progress,
            },
            &EdmModelSampling::svd(),
            latents.clone(),
            predict,
        );
    }
    let mut latents = latents.clone();
    run_video_steps(&phase.sigmas, cancel, |step| -> Result<()> {
        let scaled = scale_model_input(&latents, step.sigma)?;
        let prediction = predict(&scaled, timesteps[step.index])?;
        let denoised = v_pred_denoised(&prediction, &latents, step.sigma)?;
        latents = euler_step(&latents, &denoised, step.sigma, step.next_sigma)?;
        mlx_rs::transforms::eval([&latents])?;
        on_progress(Progress::Step {
            current: (step.index + 1) as u32,
            total: timesteps.len() as u32,
        });
        Ok(())
    })?;
    Ok(latents)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::MemoryGeometry;
    use std::cell::RefCell;

    #[derive(Debug, PartialEq)]
    enum Event {
        Predict(u32),
        Step(u32),
    }

    fn trace_schedule() -> EdmSchedule {
        // At this positive f32 node, Darwin logf and f64 log rounded to f32 differ by one ULP.
        EdmSchedule::karras(
            3,
            &SchedulerConfig {
                sigma_max: f32::from_bits(0x3f60_3bc6),
                ..Default::default()
            },
        )
    }

    fn edm_trace(sampler: Option<&str>, cancel_on_progress: bool) -> (Result<Array>, Vec<Event>) {
        let schedule = trace_schedule();
        let geometry = MemoryGeometry {
            width: 8,
            height: 8,
            batch: 1,
            frames: 1,
            reference_count: 1,
        };
        let phase = VideoDenoisePhase::fixed(geometry, 1, sampler, schedule.sigmas, 29);
        let cancel = CancelFlag::new();
        let events = RefCell::new(Vec::new());
        let result = sample_edm(
            &phase,
            &schedule.timesteps,
            &Array::from_slice(&[1f32], &[1, 1, 1, 1, 1]),
            &cancel,
            &mut |event| {
                let Progress::Step { current, total: 3 } = event else {
                    panic!("unexpected {event:?}")
                };
                events.borrow_mut().push(Event::Step(current));
                if cancel_on_progress {
                    cancel.cancel();
                }
            },
            |scaled, time| {
                events.borrow_mut().push(Event::Predict(time.to_bits()));
                Ok(scaled.clone())
            },
        );
        (result, events.into_inner())
    }

    #[test]
    fn native_and_explicit_euler_preserve_timestep_and_progress_order() {
        mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || {
            let schedule = trace_schedule();
            for sampler in [None, Some("euler")] {
                let (result, events) = edm_trace(sampler, false);
                let output = result.unwrap().item::<f32>();
                assert!(output.is_finite());
                if sampler.is_none() {
                    // Same mock predictor as edm_trace: its output is the scaled input.
                    let mut reference = Array::from_slice(&[1f32], &[1, 1, 1, 1, 1]);
                    for nodes in schedule.sigmas.windows(2) {
                        let prediction = scale_model_input(&reference, nodes[0]).unwrap();
                        let denoised = v_pred_denoised(&prediction, &reference, nodes[0]).unwrap();
                        reference = euler_step(&reference, &denoised, nodes[0], nodes[1]).unwrap();
                        mlx_rs::transforms::eval([&reference]).unwrap();
                    }
                    assert_eq!(output.to_bits(), reference.item::<f32>().to_bits());
                }
                let mut expected = Vec::new();
                for (index, &sigma) in schedule.sigmas[..3].iter().enumerate() {
                    let step = Event::Step(index as u32 + 1);
                    if sampler.is_none() {
                        expected.push(Event::Predict(
                            (0.25 * (sigma as f64).ln() as f32).to_bits(),
                        ));
                        expected.push(step);
                    } else {
                        expected.push(step);
                        expected.push(Event::Predict((0.25 * sigma.ln()).to_bits()));
                    }
                }
                assert_eq!(events, expected);
            }
        });
    }

    #[test]
    fn progress_cancellation_stops_before_the_next_prediction() {
        mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || {
            let (native, events) = edm_trace(None, true);
            assert!(matches!(native, Err(Error::Canceled)));
            assert!(matches!(&events[..], [Event::Predict(_), Event::Step(1)]));
            let (explicit, events) = edm_trace(Some("euler"), true);
            assert!(matches!(explicit, Err(Error::Canceled)));
            assert_eq!(events, [Event::Step(1)]);
        });
    }

    #[test]
    fn guidance_ramp_follows_frame_order_including_single_frames() {
        mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || {
            let ramp = SvdPipeline::guidance_schedule(3, 1., 3.);
            assert_eq!(ramp.shape(), [1, 3, 1, 1, 1]);
            assert_eq!(ramp.as_slice::<f32>(), [1., 2., 3.]);
            assert_eq!(SvdPipeline::guidance_schedule(1, 1., 3.).item::<f32>(), 1.);
            assert_eq!(
                SvdPipeline::guidance_schedule(3, 1., 0.).as_slice::<f32>(),
                [1., 0.5, 0.]
            );
        });
    }
}
