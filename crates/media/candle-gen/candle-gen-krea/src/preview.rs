//! Krea 2's per-step latent preview seam (epic 16948, sc-16950) — the candle twin of `mlx-gen-krea`'s
//! `KreaPreview`.
//!
//! Krea keeps its denoise state in the **spatial** QwenVae layout `[1, 16, H/8, W/8]` at every step of
//! every route, so wiring a preview here is two things and nothing else:
//!
//! 1. a projector closure over [`candle_gen_qwen_image::preview::project_spatial_latents`] — the
//!    epic-16624 fit, **reused** rather than refitted (see that module for the tensor-byte grounding);
//! 2. handing that closure to [`candle_gen::run_flow_sampler`] as a [`PreviewHook`], which already
//!    owns per-step numbering, multi-eval dedup, and the swallow-on-failure contract (sc-16949).
//!
//! No route restructures its loop, and no route unpacks anything: `unpack_latents` is a Qwen-Image /
//! FLUX.2 concern, not a Krea one.
//!
//! ## What the hook does and does not see
//!
//! The sampler drivers hand the hook the **running latent** `x`, which for Krea is always the single
//! `[1, 16, H/8, W/8]` target trajectory:
//!
//! * **CFG is invisible here.** Krea's true-CFG routes (Raw t2i/img2img, Raw edit, CFG multi-phase)
//!   run the conditional and unconditional forwards *inside* the predict closure and combine them with
//!   [`crate::pipeline::krea_cfg_combine`] before returning a single velocity. There is no fused
//!   `[2, …]` batch anywhere in the sampler, so the preview cannot project an unconditional half even
//!   by accident.
//! * **Edit references are invisible here.** `forward_edit_with_memory` concatenates the VAE-encoded
//!   reference latents into the DiT sequence internally; `x` stays the target alone. A preview of a
//!   Reference / MultiReference edit is therefore a preview of the image being generated, not of a
//!   reference.
//!
//! Both properties are structural rather than defensive, and
//! [`tests::sampler_hook_only_ever_sees_the_single_target_latent`] pins them against a predict closure
//! that fuses a CFG batch and concatenates references the way the real routes do.
//!
//! ## Numbering across a sliced schedule
//!
//! Every route but one drives a single [`candle_gen::run_flow_sampler`] call per image and takes the
//! driver's numbering. Multi-phase Raw resolves ONE global schedule and drives a call per phase over
//! `sigmas[start..=end]`, so it builds a global [`PreviewCounter`] with [`multiphase_hook`] and the
//! frames stay `1..=total` across phase boundaries — matching `mlx-gen-krea`.

use candle_gen::gen_core::{Image, PreviewSink};
use candle_gen::preview::{PreviewCounter, PreviewHook};
use candle_gen::{candle_core::Tensor, Result};

/// Project one Krea denoise latent. The single place the Qwen fit is applied on the candle side —
/// every route's hook routes through here, so a family-level projection change cannot reach some
/// routes and miss others.
fn project(latents: &Tensor) -> Result<Image> {
    candle_gen_qwen_image::preview::project_spatial_latents(latents)
}

/// The ordinary per-route preview hook: hand this to [`candle_gen::run_flow_sampler`] and the driver
/// numbers frames `1..=steps` against the schedule it is integrating.
///
/// Build it **per image**. A batched request (`count > 1`) runs one driver call per seed, and each
/// call must start a fresh trajectory at frame 1.
pub(crate) fn hook(sink: &PreviewSink) -> PreviewHook<'_> {
    PreviewHook::new(sink, project)
}

/// A global [`PreviewCounter`] over a multi-phase render's whole schedule. One per image.
pub(crate) fn multiphase_counter(sigmas: &[f32]) -> PreviewCounter {
    PreviewCounter::new(sigmas)
}

/// The multi-phase preview hook: numbers frames against the **global** schedule rather than the
/// per-phase slice each driver call receives, so one trajectory split across phases reads as one
/// continuous `1..=total` run instead of restarting at each boundary.
pub(crate) fn multiphase_hook<'a>(
    sink: &'a PreviewSink,
    counter: &'a PreviewCounter,
    sigmas: &'a [f32],
) -> PreviewHook<'a> {
    PreviewHook::over_schedule(sink, counter, sigmas, project)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use candle_gen::candle_core::{DType, Device, IndexOp};
    use candle_gen::gen_core::sampling::TimestepConvention;
    use candle_gen::gen_core::{CancelFlag, PreviewFrame, Progress};

    use super::*;

    /// A collecting sink plus the frames it captured.
    fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame));
        (sink, frames)
    }

    fn frames_of(captured: &Arc<Mutex<Vec<PreviewFrame>>>) -> Vec<(u32, u32)> {
        candle_gen::lock_recover(captured)
            .iter()
            .map(|f| (f.current, f.total))
            .collect()
    }

    /// A Krea-shaped latent: the spatial `[1, 16, h, w]` every route denoises in.
    fn latent(h: usize, w: usize) -> Tensor {
        Tensor::zeros((1, 16, h, w), DType::F32, &Device::Cpu).unwrap()
    }

    /// A velocity of exactly zero — the flow-Euler step then leaves the latent untouched, so the
    /// sampler's output is a pure function of its input and any byte difference is the wiring's.
    fn zero_velocity(x: &Tensor, _t: f32) -> Result<Tensor> {
        Ok(x.zeros_like()?)
    }

    /// Drive the real flow sampler over `sigmas`, optionally with a preview hook.
    fn run(
        sampler: Option<&str>,
        sigmas: &[f32],
        start: Tensor,
        preview: Option<&PreviewHook<'_>>,
        predict: impl FnMut(&Tensor, f32) -> Result<Tensor>,
    ) -> Result<Tensor> {
        candle_gen::run_flow_sampler(
            sampler,
            TimestepConvention::Sigma,
            sigmas,
            start,
            7,
            &CancelFlag::new(),
            &mut |_: Progress| {},
            preview,
            predict,
        )
    }

    /// An 8-step Turbo-shaped schedule.
    fn sigmas(steps: usize) -> Vec<f32> {
        crate::schedule::turbo_sigmas(steps)
    }

    // --- One frame per outer solver step ----------------------------------------------------------

    /// Euler evaluates once per step: an N-step render emits exactly N frames, numbered 1..=N, each
    /// carrying `total == N`.
    #[test]
    fn euler_emits_exactly_one_numbered_frame_per_step() {
        for steps in [1usize, 4, 8] {
            let (sink, captured) = collecting_sink();
            let hook = hook(&sink);
            let s = sigmas(steps);
            run(None, &s, latent(4, 4), Some(&hook), zero_velocity).unwrap();
            assert_eq!(
                frames_of(&captured),
                (1..=steps as u32)
                    .map(|n| (n, steps as u32))
                    .collect::<Vec<_>>(),
                "{steps}-step Euler render"
            );
        }
    }

    /// The candle-specific hazard the shared counter exists for: heun and dpmpp_sde evaluate the
    /// predict closure **twice** per outer step, so an undeduped path would emit 2N frames. Both must
    /// land on exactly N.
    #[test]
    fn multi_eval_solvers_still_emit_exactly_one_frame_per_outer_step() {
        for name in ["heun", "dpmpp_sde"] {
            let steps = 6usize;
            let s = sigmas(steps);

            // Count the actual evaluations to prove the solver really is multi-eval here — otherwise
            // this test would pass vacuously against a solver that fell back to Euler.
            let evaluations = std::cell::Cell::new(0usize);
            let (sink, captured) = collecting_sink();
            let hook = hook(&sink);
            run(Some(name), &s, latent(4, 4), Some(&hook), |x, t| {
                evaluations.set(evaluations.get() + 1);
                zero_velocity(x, t)
            })
            .unwrap();

            assert!(
                evaluations.get() > steps,
                "{name} must evaluate more than once per step for this test to mean anything \
                 (got {} evaluations for {steps} steps)",
                evaluations.get()
            );
            assert_eq!(
                frames_of(&captured),
                (1..=steps as u32)
                    .map(|n| (n, steps as u32))
                    .collect::<Vec<_>>(),
                "{name} must still emit exactly one frame per outer step"
            );
        }
    }

    /// Multi-phase drives one trajectory as several slices of one global schedule. Frames must run
    /// 1..=total across the boundaries, with the GLOBAL total, not restart per phase.
    #[test]
    fn multiphase_slices_share_one_continuous_frame_numbering() {
        let (sink, captured) = collecting_sink();
        let global = sigmas(9);
        let counter = multiphase_counter(&global);
        let hook = multiphase_hook(&sink, &counter, &global);

        let mut x = latent(4, 4);
        for slice in [0usize..3, 3..6, 6..9] {
            let sub = &global[slice.start..=slice.end];
            x = run(None, sub, x, Some(&hook), zero_velocity).unwrap();
        }

        assert_eq!(
            frames_of(&captured),
            (1..=9).map(|n| (n, 9)).collect::<Vec<_>>(),
            "three phases of one 9-step schedule must read as one 9-frame run"
        );
    }

    /// A fresh counter per image is what keeps a batched render numbering each trajectory from 1 —
    /// the trap `multiphase_counter`'s "one per image" contract exists to avoid.
    #[test]
    fn a_reused_multiphase_counter_would_starve_the_second_image() {
        let (sink, captured) = collecting_sink();
        let global = sigmas(3);

        // Correct: a counter per image.
        for _ in 0..2 {
            let counter = multiphase_counter(&global);
            let hook = multiphase_hook(&sink, &counter, &global);
            run(None, &global, latent(4, 4), Some(&hook), zero_velocity).unwrap();
        }
        assert_eq!(
            frames_of(&captured),
            [(1, 3), (2, 3), (3, 3), (1, 3), (2, 3), (3, 3)]
        );

        // Incorrect, pinned so the contract is visible: one counter across both images emits nothing
        // for the second.
        let (sink, captured) = collecting_sink();
        let counter = multiphase_counter(&global);
        for _ in 0..2 {
            let hook = multiphase_hook(&sink, &counter, &global);
            run(None, &global, latent(4, 4), Some(&hook), zero_velocity).unwrap();
        }
        assert_eq!(frames_of(&captured), [(1, 3), (2, 3), (3, 3)]);
    }

    // --- The latent the hook is allowed to see -----------------------------------------------------

    /// CFG and edit references never reach the preview: the sampler's running latent is the single
    /// `[1, 16, h, w]` target, and both the fused CFG batch and the reference concatenation live
    /// inside the predict closure. The closure here does both, exactly as the Raw-edit route does.
    #[test]
    fn sampler_hook_only_ever_sees_the_single_target_latent() {
        let (sink, _captured) = collecting_sink();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&seen);
        let hook = PreviewHook::new(&sink, move |x: &Tensor| {
            candle_gen::lock_recover(&recorded).push(x.dims().to_vec());
            project(x)
        });

        let s = sigmas(4);
        let reference = latent(4, 4);
        run(None, &s, latent(4, 4), Some(&hook), |x, _t| {
            // A Raw-edit-shaped forward: concatenate the reference latents onto the sequence, then
            // fuse a two-leg CFG batch. Neither is visible to the sampler, which only ever receives
            // the combined velocity back.
            let with_reference = Tensor::cat(&[x, &reference], 3)?;
            let fused = Tensor::cat(&[&with_reference, &with_reference], 0)?;
            let cond = fused.i(0)?.unsqueeze(0)?.i((.., .., .., ..4))?;
            Ok(cond.zeros_like()?)
        })
        .unwrap();

        let seen = candle_gen::lock_recover(&seen);
        assert_eq!(seen.len(), 4);
        assert!(
            seen.iter().all(|dims| dims == &[1, 16, 4, 4]),
            "the hook must only ever see the single unfused target latent, got {seen:?}"
        );
    }

    // --- Decorative by contract --------------------------------------------------------------------

    /// An inert sink must be byte-identical to no hook at all: same final latent, no projection work.
    #[test]
    fn an_inert_sink_is_byte_identical_to_an_unhooked_render() {
        let s = sigmas(6);
        let start = Tensor::rand(-1f32, 1f32, (1, 16, 8, 8), &Device::Cpu).unwrap();
        let velocity = |x: &Tensor, t: f32| Ok((x * (t as f64 + 0.25))?);

        let bare = run(None, &s, start.clone(), None, velocity).unwrap();

        let inert = PreviewSink::default();
        let inert_hook = hook(&inert);
        assert!(!inert_hook.is_active());
        let hooked = run(None, &s, start.clone(), Some(&inert_hook), velocity).unwrap();

        assert_eq!(
            bare.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            hooked.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            "an inert preview sink must not perturb a single latent byte"
        );

        // And an ACTIVE sink must not perturb them either — the preview reads the latent, never writes.
        let (sink, captured) = collecting_sink();
        let active_hook = hook(&sink);
        let active = run(None, &s, start, Some(&active_hook), velocity).unwrap();
        assert_eq!(
            bare.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            active.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        assert_eq!(candle_gen::lock_recover(&captured).len(), 6);
    }

    /// A projection failure loses its frame and never fails the render. A latent that is not in the
    /// fitted 16-channel space is the realistic shape of that failure.
    #[test]
    fn a_projection_failure_loses_the_frame_and_never_fails_the_render() {
        let (sink, captured) = collecting_sink();
        let hook = hook(&sink);
        let s = sigmas(5);
        // 4-channel latents are not QwenVae latents: every projection errors.
        let start = Tensor::zeros((1, 4, 4, 4), DType::F32, &Device::Cpu).unwrap();

        let out = run(None, &s, start, Some(&hook), zero_velocity)
            .expect("a failing projection must not fail the render");

        assert_eq!(out.dims(), [1, 4, 4, 4]);
        assert!(
            candle_gen::lock_recover(&captured).is_empty(),
            "no frame may be emitted when every projection fails"
        );
    }

    // --- Route inventory ---------------------------------------------------------------------------

    /// Split a source file at each `run_flow_sampler(` and return, per call site, the argument text up
    /// to the predict closure — the window the `preview` argument sits in.
    fn sampler_call_sites(source: &str) -> Vec<&str> {
        source
            .split("run_flow_sampler(")
            .skip(1)
            .map(|rest| {
                let end = rest.find("|x,").unwrap_or(rest.len());
                &rest[..end]
            })
            .collect()
    }

    /// Every shipped Krea **render** route emits previews. A route left unwired is a route that
    /// silently shows nothing, and no weights-free test can otherwise reach the routes that need a
    /// 12B DiT — so this pins the wiring at the source level, the way `mlx-gen-krea` does.
    ///
    /// The seven `pipeline.rs` sites are: Turbo t2i three-stage (`render_three_stage`), Turbo t2i
    /// (`render_from_context`), Turbo img2img (`render_img2img_from_context`), Raw t2i
    /// (`render_base_from_contexts`), multi-phase Raw (`render_multiphase`), Raw img2img
    /// (`render_base_img2img_from_contexts`), and edit — Turbo **and** Raw, which share
    /// `render_edit_from_context`. `control_provider.rs` adds the pose-control route.
    #[test]
    fn every_krea_render_route_passes_a_preview_hook() {
        for (file, source, expected) in [
            ("pipeline.rs", include_str!("pipeline.rs"), 7usize),
            (
                "control_provider.rs",
                include_str!("control_provider.rs"),
                1,
            ),
        ] {
            let sites = sampler_call_sites(source);
            assert_eq!(
                sites.len(),
                expected,
                "{file}: expected {expected} sampler call sites, found {}. A new render route must \
                 pass a preview hook and be named in this inventory.",
                sites.len()
            );
            for (index, args) in sites.iter().enumerate() {
                assert!(
                    args.contains("Some(&preview)"),
                    "{file} sampler site #{index} does not pass a preview hook: {args:?}"
                );
            }
        }
    }

    /// The trainer's periodic sample render is deliberately NOT a preview route: it renders a whole
    /// image for the training UI from a synthetic request that carries no sink. Pinned so the
    /// omission stays a decision rather than an oversight.
    #[test]
    fn the_trainer_sample_render_is_deliberately_unhooked() {
        let sites = sampler_call_sites(include_str!("training.rs"));
        assert_eq!(sites.len(), 1);
        assert!(
            sites[0].contains("None,"),
            "the trainer sample render has no PreviewSink to emit into"
        );
    }

    /// The frame is a latent-resolution image of the running trajectory: `[1, 16, H/8, W/8]` in,
    /// `W/8 × H/8` RGB8 out.
    #[test]
    fn emitted_frames_are_latent_resolution_rgb8() {
        let (sink, captured) = collecting_sink();
        let hook = hook(&sink);
        let s = sigmas(2);
        run(None, &s, latent(16, 12), Some(&hook), zero_velocity).unwrap();

        let frames = candle_gen::lock_recover(&captured);
        assert_eq!(frames.len(), 2);
        for frame in frames.iter() {
            assert_eq!((frame.image.width, frame.image.height), (12, 16));
            assert_eq!(frame.image.pixels.len(), 12 * 16 * 3);
        }
    }
}
