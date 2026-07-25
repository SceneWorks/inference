//! SVD image-to-video pipeline — the `StableVideoDiffusionPipeline` orchestration over the
//! components: a frame-wise CFG denoise loop (EDM v-prediction Euler, image-latent channel-concat)
//! with `guidance_scale = linspace(min, max, num_frames)`; chunked temporal VAE decode → frames.
//! candle port of `mlx-gen-svd`'s `pipeline.rs`. Latents are `[1, F, 4, h, w]`; guided denoise runs
//! the unconditioned and conditioned branches as sequential B=1 forwards to cap the activation peak.
//! Deterministic CPU-seeded noise (sc-3673 convention, matching candle-gen-wan).

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::gen_core::sampling::EdmModelSampling;
use candle_gen::gen_core::{CancelFlag, Image, Progress};
use candle_gen::{check_cancel, CandleError, Result as CResult};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::{SchedulerConfig, DEFAULT_CONDITIONING_FPS, DEFAULT_FRAMES, DEFAULT_STEPS};
use crate::scheduler::EdmSchedule;
use crate::unet::SvdUnet;
use crate::vae::SvdVae;

/// Image-to-video generation parameters (the `StableVideoDiffusionPipeline.__call__` knobs).
#[derive(Clone, Debug)]
pub struct SvdParams {
    pub num_frames: usize,
    pub num_inference_steps: usize,
    pub min_guidance_scale: f32,
    pub max_guidance_scale: f32,
    /// Motion-conditioning cadence (the `fps_id` SVD was trained on) — distinct from output fps.
    pub fps: u32,
    pub motion_bucket_id: f32,
    pub noise_aug_strength: f32,
    /// Frames decoded per temporal VAE pass (diffusers default = `num_frames`).
    pub decode_chunk_size: usize,
}

impl Default for SvdParams {
    fn default() -> Self {
        Self {
            num_frames: DEFAULT_FRAMES as usize,
            num_inference_steps: DEFAULT_STEPS as usize,
            min_guidance_scale: 1.0,
            max_guidance_scale: 3.0,
            fps: DEFAULT_CONDITIONING_FPS,
            motion_bucket_id: 127.0,
            noise_aug_strength: 0.02,
            // diffusers default = `num_frames`.
            decode_chunk_size: DEFAULT_FRAMES as usize,
        }
    }
}

/// Deterministic N(0,1) latent noise `[1, F, 4, h, w]` (f32) — CPU `StdRng` (ChaCha), launch-portable
/// per seed (matches the candle-gen-wan convention).
pub fn create_noise(
    seed: u64,
    num_frames: usize,
    h: usize,
    w: usize,
    device: &Device,
) -> CResult<Tensor> {
    let n = num_frames * 4 * h * w;
    let mut rng = StdRng::seed_from_u64(seed);
    let data = candle_gen::seeded_normal_vec(&mut rng, n);
    Ok(Tensor::from_vec(data, (1, num_frames, 4, h, w), device)?)
}

/// Deterministic N(0,1) noise of an arbitrary shape (the image-latent noise augmentation).
pub fn seeded_normal(
    seed: u64,
    shape: (usize, usize, usize, usize),
    device: &Device,
) -> CResult<Tensor> {
    let (a, b, c, d) = shape;
    let mut rng = StdRng::seed_from_u64(seed);
    let data = candle_gen::seeded_normal_vec(&mut rng, a * b * c * d);
    Ok(Tensor::from_vec(data, shape, device)?)
}

/// The `added_time_ids` micro-conditioning row `[1, 3]` = `[fps − 1, motion_bucket_id,
/// noise_aug_strength]` (the SVD pipeline reduces fps by 1 — the model was trained on fps−1).
pub fn added_time_ids(params: &SvdParams, device: &Device) -> CResult<Tensor> {
    let v = vec![
        (params.fps as f32) - 1.0,
        params.motion_bucket_id,
        params.noise_aug_strength,
    ];
    Ok(Tensor::from_vec(v, (1, 3), device)?)
}

/// The frame-wise CFG schedule `linspace(min, max, F)` shaped `[1, F, 1, 1, 1]` to broadcast over the
/// `[1, F, 4, h, w]` latents.
fn guidance_schedule(
    num_frames: usize,
    min_g: f32,
    max_g: f32,
    device: &Device,
) -> CResult<Tensor> {
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
    Ok(Tensor::from_vec(vals, (1, f, 1, 1, 1), device)?)
}

/// Run one guided prediction as two cancel-responsive batch-1 forwards, then apply the standard
/// frame-wise CFG blend. Keeping this as a tested seam prevents a future refactor from quietly
/// rebuilding the batch-2 activation graph that OOMed the canonical 32 GB workload.
#[allow(clippy::too_many_arguments)]
fn sequential_cfg_prediction(
    x_in: &Tensor,
    image_latents: &Tensor,
    image_embeds: &Tensor,
    zeros_l: &Tensor,
    zeros_e: &Tensor,
    guidance: &Tensor,
    cancel: &CancelFlag,
    mut forward: impl FnMut(&Tensor, &Tensor) -> CResult<Tensor>,
) -> CResult<Tensor> {
    check_cancel(cancel)?;
    let uncond_inp = Tensor::cat(&[x_in, zeros_l], 2)?;
    let uncond = forward(&uncond_inp, zeros_e)?;
    drop(uncond_inp);

    check_cancel(cancel)?;
    let cond_inp = Tensor::cat(&[x_in, image_latents], 2)?;
    let cond = forward(&cond_inp, image_embeds)?;
    drop(cond_inp);

    // noise_pred = uncond + guidance · (cond − uncond), frame-wise.
    Ok(uncond.add(&guidance.broadcast_mul(&(cond - &uncond)?)?)?)
}

/// The frame-wise CFG v-prediction denoise — routed through the unified curated sampler framework
/// (epic 7114 P4, sc-7125). SVD is EDM **v-prediction** over a native Karras σ schedule, so this drives
/// any curated solver (default `euler` = the byte-faithful N1 native path — `euler` over the EDM
/// contract IS exactly the legacy `v_pred_denoised` → `euler_step` loop) via
/// [`candle_gen::run_curated_sampler`] over [`EdmModelSampling::svd`]. Per **decision 3b** SVD exposes
/// the sampler axis but NO scheduler axis: the native Karras EDM schedule is kept verbatim.
///
/// The [`EdmModelSampling`] supplies the `1/√(σ²+1)` input scaling + the v→x0 recombine, so the
/// `predict` closure only does what's model-specific: sequential uncond/cond batch-1 forwards, the
/// image-latent **channel concat**, and the **per-frame** guidance ramp (re-applied each eval, so
/// multi-eval solvers stay correct). The uncond CFG branch zeros `image_embeds`/`image_latents` (the
/// diffusers SVD uncond). This preserves the batch-2 CFG algebra at a lower activation peak in
/// exchange for serial branch latency. Latents are the init noise scaled by `init_noise_sigma`.
/// Returns the final `[1, F, 4, h, w]` latents.
///
/// When guidance is disabled (`max_g <= 1.0`, the diffusers `do_classifier_free_guidance` gate) the
/// uncond half is neither built nor forwarded: the loop runs a single batch-1 cond-only UNet forward,
/// since the per-frame blend collapses to the conditional (sc-8993 F-013).
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    unet: &SvdUnet,
    scheduler: &SchedulerConfig,
    latents: &Tensor,       // [1, F, 4, h, w] (init noise · init_noise_sigma)
    image_embeds: &Tensor,  // [1, ctx, 1024]
    image_latents: &Tensor, // [1, F, 4, h, w]
    added_time_ids: &Tensor,
    num_frames: usize,
    steps: usize,
    min_g: f32,
    max_g: f32,
    sampler: Option<&str>,
    seed: u64,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> CResult<Tensor> {
    let device = latents.device().clone();
    let sched = EdmSchedule::karras(steps, scheduler);
    let ms = EdmModelSampling::svd();

    // Whether classifier-free guidance is active. SVD ramps the guidance per frame via
    // `linspace(min, max, F)`, so the diffusers `do_classifier_free_guidance` gate is `max > 1.0`:
    // when the max is 1.0 every per-frame scale is ≤ 1.0 and the blend `uncond + g·(cond − uncond)`
    // collapses to the conditional, making the uncond half + the doubled UNet forward pure waste
    // (sc-8993 F-013; the parent story scoped only lens/sd3/wan14b). Defaults (max 3.0) keep CFG on.
    let cfg_active = max_g > 1.0;

    // CFG's constant uncond inputs and guidance ramp. Keep uncond and cond as separate batch-1
    // forwards: retaining one batch-2 activation graph OOMs the canonical 25-frame workload on a
    // physical 32 GB card even after the component-residency boundaries are made sequential.
    let cfg_inputs = if cfg_active {
        let zeros_e = image_embeds.zeros_like()?;
        let zeros_l = image_latents.zeros_like()?;
        let guidance = guidance_schedule(num_frames, min_g, max_g, &device)?;
        Some((zeros_e, zeros_l, guidance))
    } else {
        None
    };

    candle_gen::run_curated_sampler(
        sampler,
        &ms,
        &sched.sigmas,
        latents.clone(),
        seed,
        cancel,
        on_progress,
        |x_in, t| -> CResult<Tensor> {
            // `x_in` is already the `1/√(σ²+1)`-scaled latent (`scale_model_input`) the driver applied
            // via `EdmModelSampling::input_scale`; `t` is the continuous EDM timestep `0.25·ln σ`.
            match &cfg_inputs {
                Some((zeros_e, zeros_l, guidance)) => sequential_cfg_prediction(
                    x_in,
                    image_latents,
                    image_embeds,
                    zeros_l,
                    zeros_e,
                    guidance,
                    cancel,
                    |inp, embeds| Ok(unet.forward(inp, t, embeds, added_time_ids, num_frames)?),
                ),
                // Guidance disabled (max ≤ 1.0): single-batch cond-only forward. The CFG blend at
                // scale 1.0 is exactly `cond`, so this returns the same velocity the 2-batch path
                // would — at half the UNet compute.
                None => {
                    let inp = Tensor::cat(&[x_in, image_latents], 2)?; // [1, F, 8, h, w]
                    Ok(unet.forward(&inp, t, image_embeds, added_time_ids, num_frames)?)
                    // [1,F,4,h,w]
                }
            }
        },
    )
}

/// Chunked temporal VAE decode (diffusers `decode_latents`): divide by `scaling_factor`, decode in
/// `chunk`-frame windows, concat. `latents` `[1, F, 4, h, w]` → frames `[1, F, 3, H, W]` (roughly
/// `[-1, 1]`; the caller maps to `[0, 1]` for display).
pub fn decode(vae: &SvdVae, latents: &Tensor, num_frames: usize, chunk: usize) -> CResult<Tensor> {
    let (b, f, c, h, w) = latents.dims5()?;
    if b != 1 {
        return Err(CandleError::Msg(format!(
            "svd decode: batch size must be 1 (got {b})"
        )));
    }
    // [1, F, 4, h, w] → [F, 4, h, w], divide by scaling_factor.
    let z = latents
        .reshape((f, c, h, w))?
        .affine(1.0 / vae.scaling_factor() as f64, 0.0)?;
    let chunk = chunk.max(1);

    let mut start = 0usize;
    let mut chunks: Vec<Tensor> = Vec::new();
    while start < num_frames {
        let n = chunk.min(num_frames - start);
        let zc = z.narrow(0, start, n)?; // [n, 4, h, w]
        chunks.push(vae.decode(&zc, n)?); // [n, 3, H, W]
        start += n;
    }
    let refs: Vec<&Tensor> = chunks.iter().collect();
    let frames = Tensor::cat(&refs, 0)?; // [F, 3, H, W]
    let (_, oc, oh, ow) = frames.dims4()?;
    Ok(frames.reshape((1, num_frames, oc, oh, ow))?)
}

/// Run a temporal chunk pipeline in strict decode → materialize → release order. `materialize`
/// consumes the decoded chunk, so its accelerator tensor drops before the next `decode` call begins.
fn decode_chunks_incrementally<Chunk, Output, E>(
    num_frames: usize,
    chunk_size: usize,
    mut decode_chunk: impl FnMut(usize, usize) -> std::result::Result<Chunk, E>,
    mut materialize: impl FnMut(Chunk) -> std::result::Result<Vec<Output>, E>,
) -> std::result::Result<Vec<Output>, E> {
    let mut output = Vec::with_capacity(num_frames);
    let mut start = 0usize;
    let chunk_size = chunk_size.max(1);
    while start < num_frames {
        let count = chunk_size.min(num_frames - start);
        let decoded = decode_chunk(start, count)?;
        output.extend(materialize(decoded)?);
        start += count;
    }
    Ok(output)
}

/// Decode temporal chunks one at a time, spatially tiling each chunk against live free VRAM, and
/// materialize its RGB frames on CPU before the next GPU chunk begins. This deliberately has no final
/// GPU concat: the returned value is already the provider's host-side `Vec<Image>`.
pub fn decode_to_images_incremental(
    vae: &SvdVae,
    latents: &Tensor,
    num_frames: usize,
    chunk_size: usize,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> CResult<Vec<Image>> {
    let (b, f, c, h, w) = latents.dims5()?;
    if b != 1 || f != num_frames {
        return Err(CandleError::Msg(format!(
            "svd decode: expected [1,{num_frames},C,H,W], got [{b},{f},{c},{h},{w}]"
        )));
    }
    let z = latents
        .reshape((f, c, h, w))?
        .affine(1.0 / vae.scaling_factor() as f64, 0.0)?;

    decode_chunks_incrementally(
        num_frames,
        chunk_size,
        |start, count| {
            candle_gen::check_cancel(cancel)?;
            let chunk = z.narrow(0, start, count)?;
            vae.decode_budgeted_with_progress(&chunk, count, cancel, on_progress)
        },
        |decoded| {
            let images = decoded_chunk_to_images(&decoded)?;
            // `decoded` is consumed by this closure and drops here, after the explicit CPU transfer
            // inside `decoded_chunk_to_images`, before the next decode closure is entered.
            drop(decoded);
            candle_gen::check_cancel(cancel)?;
            Ok(images)
        },
    )
}

/// One decoded GPU chunk `[F,3,H,W]` → CPU RGB images. The device transfer is intentionally inside
/// the per-chunk materializer so no decoded GPU chunk survives into the next VAE pass.
fn decoded_chunk_to_images(decoded: &Tensor) -> CResult<Vec<Image>> {
    let scaled = ((decoded.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?;
    let u8s = candle_gen::round_rgb8(&scaled)?.to_device(&Device::Cpu)?;
    let (f, c, h, w) = u8s.dims4()?;
    debug_assert_eq!(c, 3);
    let mut out = Vec::with_capacity(f);
    for fi in 0..f {
        let frame = u8s.narrow(0, fi, 1)?.squeeze(0)?;
        let pixels = frame.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
        out.push(Image {
            width: w as u32,
            height: h as u32,
            pixels,
        });
    }
    Ok(out)
}

/// Decoded frames `[1, F, 3, H, W]` (roughly `[-1, 1]`) → `Vec<Image>` (`clip(x·0.5+0.5)·255`).
pub fn frames_to_images(decoded: &Tensor) -> CResult<Vec<Image>> {
    let frames = decoded.squeeze(0)?;
    decoded_chunk_to_images(&frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// sc-8993 (F-013): when guidance is disabled every per-frame scale is `1.0`, and the CFG blend
    /// `uncond + g·(cond − uncond)` with `g == 1.0` is exactly `cond` for ANY uncond. This is the
    /// algebraic justification for [`denoise`] skipping the uncond half + doubled UNet forward when
    /// `max_guidance_scale <= 1.0`: the single-batch cond-only forward returns bit-identical output to
    /// what the 2-batch CFG path would have produced. Mirrors sd3's `cfg_scale_one_equals_cond_only_path`
    /// and lens's `cfg_rescale_at_guidance_one_is_cond_for_any_uncond`.
    #[test]
    fn guidance_one_blend_equals_cond_for_any_uncond() {
        let dev = Device::Cpu;
        let num_frames = 2;
        // A guidance schedule with min == max == 1.0 → every per-frame scale is exactly 1.0, shaped
        // `[1, F, 1, 1, 1]` to broadcast over the per-frame `[1, F, 4, h, w]` predictions.
        let guidance = guidance_schedule(num_frames, 1.0, 1.0, &dev).unwrap();
        // Per-frame prediction shape `[1, F, 4, 1, 1]` (a 1×1 spatial grid keeps the tensors tiny).
        let cond = Tensor::from_vec(
            vec![3.0f32, 4.0, 0.0, -2.0, 1.0, 2.0, 2.0, 7.0],
            (1, num_frames, 4, 1, 1),
            &dev,
        )
        .unwrap();
        // A deliberately unrelated uncond — the result must ignore it entirely at guidance 1.0.
        let uncond = Tensor::from_vec(
            vec![9.5f32, -0.5, 1.0, 3.0, -1.0, 8.0, 0.5, -4.5],
            (1, num_frames, 4, 1, 1),
            &dev,
        )
        .unwrap();
        // The exact blend the `denoise` CFG path computes each step.
        let blended = uncond
            .add(&guidance.broadcast_mul(&(&cond - &uncond).unwrap()).unwrap())
            .unwrap();
        let diff = (&blended - &cond)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            diff < 1e-5,
            "guidance-1.0 CFG blend must equal cond; max |diff| = {diff}"
        );
    }

    /// The memory-saving CFG seam must retain the prior batch-2 algebra, issue two batch-1/channel-8
    /// forwards in uncond → cond order, and honor cancellation before starting the second expensive
    /// branch. This is deliberately injectable so the ordering regression runs hermetically on CPU.
    #[test]
    fn sequential_cfg_matches_batched_reference_and_cancels_between_branches() {
        let dev = Device::Cpu;
        let num_frames = 2;
        let x_values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let x_in = Tensor::from_vec(x_values.clone(), (1, num_frames, 4, 1, 1), &dev).unwrap();
        let image_values = vec![11.0f32, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
        let image_latents =
            Tensor::from_vec(image_values.clone(), (1, num_frames, 4, 1, 1), &dev).unwrap();
        let zeros_l = image_latents.zeros_like().unwrap();
        let image_embeds =
            Tensor::ones((1, 1, 2), candle_gen::candle_core::DType::F32, &dev).unwrap();
        let zeros_e = image_embeds.zeros_like().unwrap();
        let guidance = guidance_schedule(num_frames, 1.0, 3.0, &dev).unwrap();
        let uncond_pred = Tensor::from_vec(
            vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            (1, num_frames, 4, 1, 1),
            &dev,
        )
        .unwrap();
        let cond_pred = Tensor::from_vec(
            vec![8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            (1, num_frames, 4, 1, 1),
            &dev,
        )
        .unwrap();

        let calls = Cell::new(0usize);
        let sequential = sequential_cfg_prediction(
            &x_in,
            &image_latents,
            &image_embeds,
            &zeros_l,
            &zeros_e,
            &guidance,
            &CancelFlag::new(),
            |inp, embeds| {
                assert_eq!(inp.dims5().unwrap(), (1, num_frames, 8, 1, 1));
                assert_eq!(embeds.dims3().unwrap(), (1, 1, 2));
                let call = calls.get();
                calls.set(call + 1);
                assert_eq!(
                    inp.narrow(2, 0, 4)?.flatten_all()?.to_vec1::<f32>()?,
                    x_values,
                    "the first four channels must remain the scaled noise latent"
                );
                let routed_image = inp.narrow(2, 4, 4)?.flatten_all()?.to_vec1::<f32>()?;
                match call {
                    0 => {
                        assert_eq!(
                            routed_image,
                            vec![0.0; image_values.len()],
                            "uncond must receive zero image latents in channels 4..8"
                        );
                        assert_eq!(embeds.sum_all()?.to_scalar::<f32>()?, 0.0);
                        Ok(uncond_pred.clone())
                    }
                    1 => {
                        assert_eq!(
                            routed_image, image_values,
                            "cond must receive source-image latents in channels 4..8"
                        );
                        assert_eq!(embeds.sum_all()?.to_scalar::<f32>()?, 2.0);
                        Ok(cond_pred.clone())
                    }
                    _ => panic!("CFG must issue exactly two forwards"),
                }
            },
        )
        .unwrap();
        assert_eq!(calls.get(), 2);

        // Reconstruct the old batch-2 result and prove the sequential seam returns the same blend.
        let batched = Tensor::cat(&[&uncond_pred, &cond_pred], 0).unwrap();
        let uncond_ref = batched.narrow(0, 0, 1).unwrap();
        let cond_ref = batched.narrow(0, 1, 1).unwrap();
        let expected = uncond_ref
            .add(
                &guidance
                    .broadcast_mul(&(&cond_ref - &uncond_ref).unwrap())
                    .unwrap(),
            )
            .unwrap();
        let diff = (&sequential - &expected)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(diff, 0.0, "sequential CFG must match the batch-2 algebra");

        let cancel = CancelFlag::new();
        let canceled_calls = Cell::new(0usize);
        let canceled = sequential_cfg_prediction(
            &x_in,
            &image_latents,
            &image_embeds,
            &zeros_l,
            &zeros_e,
            &guidance,
            &cancel,
            |_inp, _embeds| {
                canceled_calls.set(canceled_calls.get() + 1);
                cancel.cancel();
                Ok(uncond_pred.clone())
            },
        );
        assert!(matches!(canceled, Err(CandleError::Canceled)));
        assert_eq!(
            canceled_calls.get(),
            1,
            "cancellation after uncond must prevent the cond forward"
        );
    }

    /// Mutation witness for the production ordering: materialization consumes and releases each
    /// decoded GPU chunk before the next decode begins. Removing the materialize step either leaves
    /// the liveness witness resident or loses the output, and this test fails.
    #[test]
    fn incremental_materialization_releases_each_chunk_before_next_decode() {
        struct GpuChunk<'a> {
            id: usize,
            live: &'a Cell<usize>,
        }
        impl Drop for GpuChunk<'_> {
            fn drop(&mut self) {
                self.live.set(self.live.get() - 1);
            }
        }

        let live = Cell::new(0usize);
        let out = decode_chunks_incrementally(
            5,
            2,
            |start, count| {
                assert_eq!(
                    live.get(),
                    0,
                    "the previous GPU chunk must be released before decoding the next"
                );
                live.set(live.get() + 1);
                Ok::<_, ()>(GpuChunk {
                    id: start + count,
                    live: &live,
                })
            },
            |chunk| {
                let id = chunk.id;
                drop(chunk);
                assert_eq!(
                    live.get(),
                    0,
                    "CPU materialization must consume the GPU chunk"
                );
                Ok(vec![id])
            },
        )
        .unwrap();

        assert_eq!(out, vec![2, 4, 5]);
        assert_eq!(live.get(), 0);
    }
}
