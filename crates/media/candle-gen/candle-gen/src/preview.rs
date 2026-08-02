//! Shared per-step latent preview machinery — the candle twin of `mlx_gen::preview` (epic 16948,
//! sc-16949; the MLX original is epic 16624).
//!
//! Providers own their latent layout and linear RGB fit. This module only numbers schedule
//! positions, validates an already-unpacked `[1, C, h, w]` latent, projects it with a caller-owned
//! `C x 3` fit, and emits the resulting RGB8 frame. The fits themselves are **backend-neutral** —
//! they are least-squares constants over a VAE *latent space*, so candle reuses the constants epic
//! 16624 committed rather than refitting them, once a family has proven it loads the same VAE bytes.
//!
//! ## Why the counter dedups (the candle-specific hazard)
//!
//! [`crate::sampler::run_curated_sampler`] recomputes the step counter on **every** model evaluation
//! and calls `on_progress` each time. Heun and DPM++ SDE evaluate twice per outer step, so that
//! value deliberately repeats. Repetition is correct for `Progress` (monotone and bounded is all it
//! promises) and **wrong** for previews: an undeduped path would burn a projection and push a
//! duplicate frame. [`PreviewCounter::next`] therefore returns `None` for an already-emitted
//! schedule position, exactly as `mlx_gen::preview::PreviewCounter::next` does.
//!
//! ## Decorative by contract
//!
//! A projection failure is swallowed — the frame is lost, the counter keeps its advanced position,
//! and the caller's render is unaffected. A stale or absent fit degrades preview colour only; the
//! denoise path never reads these constants.

use std::cell::Cell;

use candle_core::{DType, Tensor};
use gen_core::{Image, PreviewFrame, PreviewSink};

use crate::{CandleError, Result};

/// The projection a family hands the sampler drivers: unpack the running latent into the
/// `[1, C, h, w]` contract and project it with the family's committed fit.
type Projector<'a> = Box<dyn Fn(&Tensor) -> Result<Image> + 'a>;

/// Numbers preview frames by schedule position.
///
/// The sampler drivers invoke the hook once per model evaluation. Keying the resulting frames to
/// schedule positions — and refusing a position that already produced a frame — keeps numbering
/// monotonic, bounded, and one-per-outer-step even for multi-eval solvers and truncated schedules.
pub struct PreviewCounter {
    emitted: Cell<u32>,
    total: u32,
}

impl PreviewCounter {
    /// Build a counter for a sampler schedule (`n + 1` sigma entries for `n` steps).
    pub fn new(sigmas: &[f32]) -> Self {
        Self {
            emitted: Cell::new(0),
            total: sigmas.len().saturating_sub(1).max(1) as u32,
        }
    }

    /// Build a counter for a driver that has **no** sigma schedule to key on — the SCM / TrigFlow
    /// loop ([`crate::sampler::run_scm_sampler`], SANA-Sprint) walks `ScmScheduler` *angle* timesteps
    /// rather than a descending σ array, so its frames are keyed on the step index instead.
    pub fn with_steps(steps: usize) -> Self {
        Self {
            emitted: Cell::new(0),
            total: steps.max(1) as u32,
        }
    }

    /// Total denoise steps represented by this schedule.
    pub fn total(&self) -> u32 {
        self.total
    }

    /// Return the 1-based frame number for `sigma`, or `None` if that position was emitted.
    ///
    /// An off-schedule solver evaluation (DPM++ SDE's midpoint) advances one position beyond the
    /// last emission. Frame numbering remains monotonic and bounded in both cases.
    pub fn next(&self, sigmas: &[f32], sigma: f32) -> Option<u32> {
        let candidate = match sigmas.iter().position(|s| *s == sigma) {
            Some(index) => index as u32 + 1,
            None => self.emitted.get().saturating_add(1),
        }
        .min(self.total);
        self.advance(candidate)
    }

    /// Return the 1-based frame number for a 0-based **step index**, or `None` if that position was
    /// emitted. The index-keyed counterpart of [`next`](Self::next), for the SCM driver.
    pub fn next_step(&self, step: usize) -> Option<u32> {
        let candidate = (step as u64 + 1).min(u32::MAX as u64) as u32;
        self.advance(candidate.min(self.total))
    }

    /// Commit `candidate` if it is genuinely ahead of the last emission.
    fn advance(&self, candidate: u32) -> Option<u32> {
        if candidate <= self.emitted.get() {
            return None;
        }
        self.emitted.set(candidate);
        Some(candidate)
    }
}

/// Run a provider-owned projection and emit it for this schedule position.
///
/// Projection failures are deliberately swallowed: previews are decorative and losing a frame must
/// never fail the caller's render. The projection closure runs only after the counter advances,
/// preserving schedule-position consumption when family-specific unpacking fails, and is never
/// invoked for an inert sink or an already-emitted position.
pub fn emit_preview<F>(
    sink: &PreviewSink,
    counter: &PreviewCounter,
    sigmas: &[f32],
    sigma: f32,
    project: F,
) where
    F: FnOnce() -> Result<Image>,
{
    if !sink.is_active() {
        return;
    }
    let Some(current) = counter.next(sigmas, sigma) else {
        return;
    };
    deliver(sink, counter, current, project);
}

/// [`emit_preview`] keyed on a 0-based **step index** instead of a schedule sigma — for a driver
/// with no σ array (the SCM / TrigFlow loop). Same inert-sink, dedup, and swallow semantics.
pub fn emit_preview_at<F>(sink: &PreviewSink, counter: &PreviewCounter, step: usize, project: F)
where
    F: FnOnce() -> Result<Image>,
{
    if !sink.is_active() {
        return;
    }
    let Some(current) = counter.next_step(step) else {
        return;
    };
    deliver(sink, counter, current, project);
}

/// Project and push one frame, swallowing a projection failure.
fn deliver<F>(sink: &PreviewSink, counter: &PreviewCounter, current: u32, project: F)
where
    F: FnOnce() -> Result<Image>,
{
    if let Ok(image) = project() {
        sink.emit(PreviewFrame {
            current,
            total: counter.total(),
            image,
        });
    }
}

/// A family's opt-in preview seam for the shared sampler drivers
/// ([`crate::sampler::run_curated_sampler`], [`crate::sampler::run_flow_sampler`],
/// [`crate::sampler::run_scm_sampler`]).
///
/// On candle nearly every family funnels its denoise through those drivers, so a family opts into
/// previews by handing the driver one of these rather than by restructuring its loop. The hook
/// carries only the sink and the family's projection closure — **the driver owns the
/// [`PreviewCounter`]**, built from the very schedule it is integrating, so a family cannot
/// accidentally number frames against a different schedule than the one being run.
///
/// The closure receives the driver's **running latent** in the family's own native layout; unpacking
/// to the `[1, C, h, w]` [`project_latents`] contract is the family's job.
pub struct PreviewHook<'a> {
    sink: &'a PreviewSink,
    project: Projector<'a>,
}

impl<'a> PreviewHook<'a> {
    /// Build a hook from a request's sink and the family's projection closure.
    pub fn new<F>(sink: &'a PreviewSink, project: F) -> Self
    where
        F: Fn(&Tensor) -> Result<Image> + 'a,
    {
        Self {
            sink,
            project: Box::new(project),
        }
    }

    /// Whether anyone is listening. The drivers gate on this so an inert sink costs one branch per
    /// evaluation and nothing else.
    pub fn is_active(&self) -> bool {
        self.sink.is_active()
    }

    /// σ-keyed emission — the curated / flow drivers.
    pub fn emit(&self, counter: &PreviewCounter, sigmas: &[f32], sigma: f32, latents: &Tensor) {
        emit_preview(self.sink, counter, sigmas, sigma, || {
            (self.project)(latents)
        });
    }

    /// Step-index-keyed emission — the SCM driver, which has no σ schedule.
    pub fn emit_step(&self, counter: &PreviewCounter, step: usize, latents: &Tensor) {
        emit_preview_at(self.sink, counter, step, || (self.project)(latents));
    }
}

impl std::fmt::Debug for PreviewHook<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreviewHook")
            .field("active", &self.sink.is_active())
            .finish_non_exhaustive()
    }
}

/// Project an unpacked `[1, C, h, w]` latent to a latent-resolution RGB8 image.
///
/// `factors` must contain exactly one RGB row per latent channel. The fit remains provider-owned:
/// sharing this operation does not imply that latent spaces share coefficients.
///
/// Two candle-specific details, both load-bearing:
///
/// * **The computation runs in f32 regardless of the incoming dtype.** The candle lanes denoise in
///   bf16 on GPU and a bf16 matmul against f32 constants panics, so the latent is cast up front.
/// * **Ties round to even**, via [`crate::round_rgb8`]. Candle's native `round` breaks `.5` ties away
///   from zero while MLX/PyTorch round to even; using the shared helper is what makes this
///   byte-identical to the MLX projection for the same latent and constants.
pub fn project_latents(latents: &Tensor, factors: &[[f32; 3]], bias: [f32; 3]) -> Result<Image> {
    let dims = latents.dims();
    if dims.len() != 4 || dims[0] != 1 {
        return Err(CandleError::Msg(format!(
            "preview latent must have shape [1, C, h, w], got {dims:?}"
        )));
    }
    if dims[1] == 0 || dims[2] == 0 || dims[3] == 0 {
        return Err(CandleError::Msg(format!(
            "preview latent channel and spatial dimensions must be non-zero, got {dims:?}"
        )));
    }
    let (channels, h, w) = (dims[1], dims[2], dims[3]);
    if factors.len() != channels {
        return Err(CandleError::Msg(format!(
            "preview factor row count {} does not match latent channel count {channels}",
            factors.len()
        )));
    }

    let device = latents.device();
    // f32 for the whole projection: the bf16-GPU / f32-constant matmul is a hard panic on candle.
    let x = latents
        .to_dtype(DType::F32)?
        .reshape((channels, h * w))?
        .t()?
        .contiguous()?;

    let factor_values: Vec<f32> = factors.iter().flatten().copied().collect();
    let factors = Tensor::from_vec(factor_values, (channels, 3), device)?;
    let bias = Tensor::from_vec(bias.to_vec(), 3, device)?;
    let rgb = x
        .matmul(&factors)?
        .broadcast_add(&bias)?
        .clamp(0f32, 1f32)?;
    let pixels = crate::round_rgb8(&rgb.affine(255.0, 0.0)?)?
        .flatten_all()?
        .to_vec1::<u8>()?;

    Ok(Image {
        width: w as u32,
        height: h as u32,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use candle_core::Device;

    use super::*;

    /// An 8-step schedule has 9 sigmas. Euler evaluates once per step, at sigmas[0..8].
    const SIGMAS: [f32; 9] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.0];

    fn zeros(shape: (usize, usize, usize, usize)) -> Tensor {
        Tensor::zeros(shape, DType::F32, &Device::Cpu).unwrap()
    }

    fn collecting_sink() -> (PreviewSink, Arc<Mutex<Vec<PreviewFrame>>>) {
        let frames = Arc::new(Mutex::new(Vec::new()));
        let captured = Arc::clone(&frames);
        let sink = PreviewSink::new(move |frame| crate::lock_recover(&captured).push(frame));
        (sink, frames)
    }

    // --- Counter: schedule-position numbering + the candle dedup contract -------------------------

    #[test]
    fn euler_cadence_numbers_every_step_once() {
        let counter = PreviewCounter::new(&SIGMAS);
        assert_eq!(counter.total(), 8);
        let frames: Vec<_> = SIGMAS[..8]
            .iter()
            .filter_map(|&s| counter.next(&SIGMAS, s))
            .collect();
        assert_eq!(frames, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn duplicate_schedule_positions_are_not_emitted_twice() {
        let counter = PreviewCounter::new(&SIGMAS);
        let mut frames = Vec::new();
        for &sigma in &SIGMAS[..8] {
            frames.extend(counter.next(&SIGMAS, sigma));
            frames.extend(counter.next(&SIGMAS, sigma));
        }
        assert_eq!(frames, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(frames.iter().all(|&f| f <= counter.total()));
    }

    #[test]
    fn off_schedule_sigma_falls_back_to_monotonic_numbering() {
        let counter = PreviewCounter::new(&SIGMAS);
        assert_eq!(counter.next(&SIGMAS, 0.95), Some(1));
        assert_eq!(counter.next(&SIGMAS, 0.85), Some(2));
        assert_eq!(counter.next(&SIGMAS, 0.9), None);
        assert_eq!(counter.next(&SIGMAS, 0.8), Some(3));
    }

    #[test]
    fn single_sigma_schedule_has_a_total_of_one() {
        let sigmas = [1.0_f32];
        let counter = PreviewCounter::new(&sigmas);
        assert_eq!(counter.total(), 1);
        assert_eq!(counter.next(&sigmas, 1.0), Some(1));
        assert_eq!(counter.next(&sigmas, 1.0), None);
    }

    /// The SCM counter shape: keyed on step index, never exceeding `total`, and deduping a repeat.
    #[test]
    fn step_keyed_counter_numbers_each_index_once_and_stays_bounded() {
        let counter = PreviewCounter::with_steps(4);
        assert_eq!(counter.total(), 4);
        let frames: Vec<_> = (0..4).filter_map(|i| counter.next_step(i)).collect();
        assert_eq!(frames, vec![1, 2, 3, 4]);
        // A repeat of an emitted index yields nothing, and an over-run index clamps and dedups.
        assert_eq!(counter.next_step(3), None);
        assert_eq!(counter.next_step(9), None);
    }

    #[test]
    fn step_keyed_counter_floors_a_zero_step_schedule_at_one() {
        let counter = PreviewCounter::with_steps(0);
        assert_eq!(counter.total(), 1);
        assert_eq!(counter.next_step(0), Some(1));
        assert_eq!(counter.next_step(1), None);
    }

    // --- Emission: inert sink, dedup, swallowed failures ------------------------------------------

    #[test]
    fn emit_preview_swallows_projection_errors_and_continues_the_schedule() {
        let (sink, frames) = collecting_sink();
        let counter = PreviewCounter::new(&SIGMAS);
        let latents = zeros((1, 4, 2, 3));

        emit_preview(&sink, &counter, &SIGMAS, SIGMAS[0], || {
            Err(CandleError::Msg("synthetic projection failure".into()))
        });
        emit_preview(&sink, &counter, &SIGMAS, SIGMAS[1], || {
            project_latents(&latents, &[[0.0; 3]; 4], [0.0; 3])
        });

        let frames = crate::lock_recover(&frames);
        assert_eq!(frames.len(), 1, "the failed projection must not emit");
        // The counter kept its advanced position: the surviving frame is 2, not 1.
        assert_eq!(frames[0].current, 2);
        assert_eq!(frames[0].total, 8);
    }

    #[test]
    fn inert_sink_does_not_advance_the_counter_or_project() {
        let counter = PreviewCounter::new(&SIGMAS);
        emit_preview(
            &PreviewSink::default(),
            &counter,
            &SIGMAS,
            SIGMAS[0],
            || panic!("an inert preview sink must not invoke projection"),
        );
        assert_eq!(counter.next(&SIGMAS, SIGMAS[0]), Some(1));
    }

    #[test]
    fn inert_sink_does_not_advance_the_step_counter_or_project() {
        let counter = PreviewCounter::with_steps(4);
        emit_preview_at(&PreviewSink::default(), &counter, 0, || {
            panic!("an inert preview sink must not invoke projection")
        });
        assert_eq!(counter.next_step(0), Some(1));
    }

    #[test]
    fn emit_preview_at_numbers_and_swallows_like_the_sigma_keyed_path() {
        let (sink, frames) = collecting_sink();
        let counter = PreviewCounter::with_steps(3);
        let latents = zeros((1, 4, 2, 3));

        emit_preview_at(&sink, &counter, 0, || {
            Err(CandleError::Msg("synthetic projection failure".into()))
        });
        emit_preview_at(&sink, &counter, 1, || {
            project_latents(&latents, &[[0.0; 3]; 4], [0.0; 3])
        });
        // A repeat of an emitted index is dropped before the (panicking) projection runs.
        emit_preview_at(&sink, &counter, 1, || {
            panic!("duplicate index must not project")
        });

        let frames = crate::lock_recover(&frames);
        assert_eq!(frames.len(), 1);
        assert_eq!((frames[0].current, frames[0].total), (2, 3));
    }

    // --- The hook wrapper -------------------------------------------------------------------------

    #[test]
    fn hook_reports_activity_and_forwards_the_running_latent() {
        let (sink, frames) = collecting_sink();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&seen);
        let hook = PreviewHook::new(&sink, move |x: &Tensor| {
            crate::lock_recover(&recorded).push(x.dims().to_vec());
            project_latents(x, &[[0.0; 3]; 4], [0.0; 3])
        });
        assert!(hook.is_active());
        assert!(!PreviewHook::new(&PreviewSink::default(), |_| unreachable!()).is_active());

        let counter = PreviewCounter::new(&SIGMAS);
        let latents = zeros((1, 4, 2, 3));
        hook.emit(&counter, &SIGMAS, SIGMAS[0], &latents);
        hook.emit(&counter, &SIGMAS, SIGMAS[0], &latents); // deduped

        assert_eq!(*crate::lock_recover(&seen), vec![vec![1, 4, 2, 3]]);
        assert_eq!(crate::lock_recover(&frames).len(), 1);
    }

    // --- Projection --------------------------------------------------------------------------------

    #[test]
    fn projection_accepts_every_shipped_latent_channel_count() {
        for channels in [4, 12, 16, 32, 128] {
            let latents = zeros((1, channels, 2, 3));
            let factors = vec![[0.0; 3]; channels];
            let image = project_latents(&latents, &factors, [0.0, 0.5, 1.0]).unwrap();
            assert_eq!((image.width, image.height), (3, 2));
            assert_eq!(image.pixels, [0, 128, 255].repeat(6));
        }
    }

    #[test]
    fn projection_rejects_invalid_layout_and_factor_rows() {
        let rank_three = Tensor::zeros((4, 2, 3), DType::F32, &Device::Cpu).unwrap();
        let error = project_latents(&rank_three, &[[0.0; 3]; 4], [0.0; 3]).unwrap_err();
        assert!(error.to_string().contains("[1, C, h, w]"));

        let batched = zeros((2, 4, 2, 3));
        let error = project_latents(&batched, &[[0.0; 3]; 4], [0.0; 3]).unwrap_err();
        assert!(error.to_string().contains("[1, C, h, w]"));

        let four_channels = zeros((1, 4, 2, 3));
        let error = project_latents(&four_channels, &[[0.0; 3]; 3], [0.0; 3]).unwrap_err();
        assert!(error
            .to_string()
            .contains("3 does not match latent channel count 4"));
    }

    #[test]
    fn projection_rejects_zero_channel_or_spatial_dimensions() {
        for shape in [(1, 0, 2, 3), (1, 4, 0, 3), (1, 4, 2, 0)] {
            let latents = zeros(shape);
            let factors = vec![[0.0; 3]; shape.1];
            let error = project_latents(&latents, &factors, [0.0; 3]).unwrap_err();
            assert!(error.to_string().contains("must be non-zero"));
        }
    }

    /// Hand-computed reference over dyadic values, so the result is exact under any matmul
    /// accumulation order: pixel0 = (0.5, 0.25) and pixel1 = (0.25, 0.5) over two channels.
    #[test]
    fn projection_matches_a_hand_computed_reference() {
        // [1, C=2, h=1, w=2]: channel 0 = [0.5, 0.25], channel 1 = [0.25, 0.5].
        let latents =
            Tensor::from_vec(vec![0.5f32, 0.25, 0.25, 0.5], (1, 2, 1, 2), &Device::Cpu).unwrap();
        let factors = [[1.0f32, 0.0, 0.5], [0.0, 1.0, 0.5]];
        let image = project_latents(&latents, &factors, [0.0; 3]).unwrap();
        assert_eq!((image.width, image.height), (2, 1));
        // pixel0 = (0.5, 0.25, 0.375) → (127.5, 63.75, 95.625) → (128, 64, 96)
        // pixel1 = (0.25, 0.5, 0.375) → (63.75, 127.5, 95.625) → (64, 128, 96)
        assert_eq!(image.pixels, vec![128, 64, 96, 64, 128, 96]);
    }

    /// The dtype trap: the candle lanes denoise in bf16 on GPU and a bf16 matmul against f32
    /// constants panics. The projection must cast up front rather than inherit the latent's dtype.
    #[test]
    fn projection_runs_in_f32_for_a_low_precision_latent() {
        for dtype in [DType::BF16, DType::F16, DType::F32, DType::F64] {
            let latents = zeros((1, 4, 2, 3)).to_dtype(dtype).unwrap();
            let image = project_latents(&latents, &[[0.0; 3]; 4], [0.0, 0.5, 1.0])
                .unwrap_or_else(|e| panic!("{dtype:?} latent failed to project: {e}"));
            assert_eq!(image.pixels, [0, 128, 255].repeat(6));
        }
    }

    /// Ties must round to even (MLX/PyTorch), not away from zero (candle's native `round`), or the
    /// two backends' previews differ by one code value on any tie.
    #[test]
    fn projection_rounds_ties_to_even_like_mlx() {
        // A factor whose ×255 product lands exactly on a `.5` tie above an EVEN floor.
        let (factor, even_target) = (0u32..127)
            .map(|n| n * 2)
            .find_map(|target| {
                let scaled = target as f32 + 0.5;
                let factor = scaled / 255.0;
                (factor * 255.0 == scaled).then_some((factor, target as u8))
            })
            .expect("some even .5 tie is reachable through the ×255 scale");

        let latents = Tensor::from_vec(vec![1.0f32], (1, 1, 1, 1), &Device::Cpu).unwrap();
        let image = project_latents(&latents, &[[factor; 3]], [0.0; 3]).unwrap();
        assert_eq!(
            image.pixels,
            vec![even_target; 3],
            "a .5 tie above an even floor must round DOWN to even, not away from zero"
        );
    }
}
