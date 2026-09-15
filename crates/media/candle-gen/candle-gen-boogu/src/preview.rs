//! Boogu's per-step latent preview seam (epic 16948, sc-17218).
//!
//! Boogu's transformer state is already the native `[1, 16, h, w]` VAE latent. It does not use
//! FLUX.1's packed `[1, S, 64]` token layout, so this module reuses only
//! [`candle_gen_flux::preview::project_raw_latents`]. The coefficient table remains owned by
//! `candle-gen-flux`; there is deliberately no second fit here.
//!
//! That reuse is grounded in weights, not a family-name guess. The Boogu VAE has the same 244 learned
//! tensors and config as FLUX.1's 16-channel `AutoencoderKL`; its f32 tensors round exactly to the
//! donor's bf16 bits. The per-snapshot and packed-tier record lives in
//! `docs/migration/evidence/sc-17218-boogu-candle-preview.md`.
//!
//! Base, Edit, and Turbo's curated lane pass [`hook`] to [`candle_gen::run_flow_sampler`]. Turbo's
//! default lane is different: it owns a DMD loop, so it calls `PreviewHook::emit_step` at the top of
//! each iteration. Both paths therefore project the running latent **entering** an outer step. In the
//! native DMD lane that means the initial noise on step zero and the prior step's re-noised state on
//! later steps—not the transient clean estimate between prediction and re-noise. This matches the
//! shared driver's observation point and the state the next model evaluation actually consumes.
//!
//! The flow cohort has unit input scaling, so no sigma correction is needed. Projection is decorative:
//! an inactive sink performs no tensor work and a projection failure loses only that frame.

use candle_gen::gen_core::PreviewSink;
use candle_gen::preview::{PreviewCounter, PreviewHook};

pub use candle_gen_flux::preview::{project_raw_latents, PREVIEW_LATENT_CHANNELS};

/// Build the shared/native preview hook over Boogu's already-spatial 16-channel running latent.
pub fn hook(sink: &PreviewSink) -> PreviewHook<'_> {
    PreviewHook::new(sink, project_raw_latents)
}

/// Frame numbering for Turbo's sigma-array-free native DMD loop.
pub(crate) fn native_counter(steps: usize) -> PreviewCounter {
    PreviewCounter::with_steps(steps)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::sync::{Arc, Mutex};

    use candle_gen::candle_core::{Device, Tensor};
    use candle_gen::gen_core::{CancelFlag, PreviewFrame, PreviewSink, Progress};
    use candle_gen::preview::PreviewHook;

    use super::*;

    fn latent() -> Tensor {
        let values = (0..(PREVIEW_LATENT_CHANNELS * 2 * 3))
            .map(|i| (i as f32 - 48.0) / 24.0)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (1, PREVIEW_LATENT_CHANNELS, 2, 3), &Device::Cpu).unwrap()
    }

    fn run(
        sampler: Option<&str>,
        sigmas: &[f32],
        preview: Option<&PreviewHook<'_>>,
        evaluations: &Cell<usize>,
    ) -> Vec<f32> {
        candle_gen::run_flow_sampler(
            sampler,
            candle_gen::gen_core::sampling::TimestepConvention::OneMinusSigma,
            sigmas,
            latent(),
            17218,
            &CancelFlag::new(),
            &mut |_: Progress| {},
            preview,
            |x: &Tensor, timestep: f32| {
                evaluations.set(evaluations.get() + 1);
                Ok((x * (timestep as f64 + 0.25))?)
            },
        )
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
    }

    fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| candle_gen::lock_recover(&captured).push(frame));
        (sink, frames)
    }

    #[test]
    fn boogu_reuses_the_native_sixteen_channel_flux1_fit() {
        assert_eq!(PREVIEW_LATENT_CHANNELS, 16);
        assert_eq!(PREVIEW_LATENT_CHANNELS, crate::pipeline::LATENT_CHANNELS);
        let frame = project_raw_latents(&latent()).unwrap();
        assert_eq!((frame.width, frame.height), (3, 2));
    }

    /// Boogu uses the flow driver's native state directly. Prove both sides of that decision: the
    /// sampling convention has unit input scaling at every sigma, and projecting the unit-normal
    /// state seen by the first emission produces a readable field rather than rail-clipped pixels.
    #[test]
    fn boogu_flow_state_needs_no_sigma_correction() {
        use candle_gen::gen_core::sampling::{
            FlowModelSampling, ModelSampling, TimestepConvention,
        };

        let sampling = FlowModelSampling::new(TimestepConvention::OneMinusSigma);
        for sigma in [0.0f32, 0.05, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(
                sampling.input_scale(sigma),
                1.0,
                "Boogu's running flow state would need PreviewHook::with_sigma at sigma {sigma}"
            );
        }

        let (height, width) = (32usize, 32usize);
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(17_218);
        let noise =
            candle_gen::seeded_normal_vec(&mut rng, PREVIEW_LATENT_CHANNELS * height * width);
        let state = Tensor::from_vec(
            noise,
            (1, PREVIEW_LATENT_CHANNELS, height, width),
            &Device::Cpu,
        )
        .unwrap();
        let frame = project_raw_latents(&state).unwrap();
        let rails = frame
            .pixels
            .iter()
            .filter(|&&value| value == 0 || value == 255)
            .count() as f64
            / frame.pixels.len() as f64;
        assert!(
            rails < 0.05,
            "the uncorrected first flow preview must be readable, not rail-clipped ({rails:.4})"
        );
    }

    /// Heun evaluates the model more than once per outer step. The shared counter must still emit
    /// exactly one frame per schedule position, and merely observing the latent cannot move it.
    #[test]
    fn shared_multi_eval_lane_deduplicates_and_is_byte_identical() {
        let sigmas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0];
        let bare_evals = Cell::new(0);
        let bare = run(Some("heun"), &sigmas, None, &bare_evals);

        let (sink, frames) = collecting_sink();
        let preview = hook(&sink);
        let live_evals = Cell::new(0);
        let live = run(Some("heun"), &sigmas, Some(&preview), &live_evals);

        let steps = sigmas.len() - 1;
        assert!(
            live_evals.get() > steps,
            "Heun must exercise a multi-eval path"
        );
        assert_eq!(live_evals.get(), bare_evals.get());
        assert_eq!(live, bare, "a live preview sink must not move the latent");
        assert_eq!(
            candle_gen::lock_recover(&frames)
                .iter()
                .map(|frame| (frame.current, frame.total))
                .collect::<Vec<_>>(),
            (1..=steps as u32)
                .map(|current| (current, steps as u32))
                .collect::<Vec<_>>()
        );

        let inert = PreviewSink::default();
        let quiet = hook(&inert);
        let inert_evals = Cell::new(0);
        assert!(!quiet.is_active());
        assert_eq!(run(Some("heun"), &sigmas, Some(&quiet), &inert_evals), bare);
        assert_eq!(inert_evals.get(), bare_evals.get());
    }

    #[test]
    fn native_dmd_counter_emits_once_per_outer_step() {
        let steps = 4usize;
        let (sink, frames) = collecting_sink();
        let preview_hook = hook(&sink);
        let counter = native_counter(steps);
        let state = latent();

        for step in 0..steps {
            preview_hook.emit_step(&counter, step, &state);
            // A repeated model evaluation at the same outer position must not duplicate a frame.
            preview_hook.emit_step(&counter, step, &state);
        }

        assert_eq!(
            candle_gen::lock_recover(&frames)
                .iter()
                .map(|frame| (frame.current, frame.total))
                .collect::<Vec<_>>(),
            (1..=steps as u32)
                .map(|current| (current, steps as u32))
                .collect::<Vec<_>>()
        );
    }
}
