//! The **`t2va` render core** (sc-17147), driven by a synthetic velocity model — no weights.
//!
//! # Why cancel *latency* is not the thing to measure
//!
//! sc-17146 established this and it applies unchanged one layer up. MLX is lazily evaluated, so a
//! loop that computes **nothing** returns `Canceled` *faster* than one that computes everything: a
//! latency assertion is satisfied best by the most broken implementation. The property that
//! actually matters is **what fraction of the render's compute happens inside the cancel-checked
//! region**, and that is measurable.
//!
//! [`the_render_core_keeps_its_compute_inside_the_cancel_checked_region`] measures it end to end
//! over [`render_latents`] — denoise **and** the unpatchify tail — by timing the call against the
//! time the caller's first host readback still has to pay. Its mutation arm removes the terminal
//! `eval` and shows the fraction collapsing, so the assertion is demonstrated to be discriminating
//! rather than assumed to be.

mod common;

use std::time::Instant;

use mlx_rs::ops::matmul;
use mlx_rs::Array;

use mlx_gen::{CancelFlag, Result};
use mlx_gen_minimax_h3::denoise::{JointStep, JointVelocity, PackedLayout};
use mlx_gen_minimax_h3::{
    initial_latents, render_latents, resolve_geometry, JointSchedule, RequestGeometry,
    LEGAL_FRAME_COUNTS, PATCH_SIZE, SMALLEST_LEGAL_FRAMES,
};

/// A canvas small enough for a synthetic model to drive at real row counts: 32x32 pixels is a 2x2
/// latent, so the shortest legal clip is 37 latent frames of one patched row each.
const WIDTH: u32 = 32;
const HEIGHT: u32 = 32;

fn geometry() -> RequestGeometry {
    resolve_geometry(WIDTH, HEIGHT, SMALLEST_LEGAL_FRAMES).expect("the smallest legal render")
}

fn layout(g: &RequestGeometry) -> PackedLayout {
    mlx_gen_minimax_h3::t2va_layout(g, 6, PATCH_SIZE).expect("t2va layout")
}

/// A velocity model whose per-step cost is real GPU work, so the compute split is measurable.
///
/// Returns a scaled multiple of its input so the loop's arithmetic stays well-conditioned, but pays
/// `matmuls` square matmuls first — the synthetic stand-in for 50 transformer blocks.
struct Costly {
    weight: Array,
    matmuls: usize,
    calls: usize,
}

impl Costly {
    fn new(dim: i32, matmuls: usize) -> Self {
        let n = (dim * dim) as usize;
        let v: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) / 512.0).collect();
        Self {
            weight: Array::from_slice(&v, &[dim, dim]),
            matmuls,
            calls: 0,
        }
    }

    fn burn(&self) -> Result<Array> {
        let mut acc = self.weight.clone();
        for _ in 0..self.matmuls {
            acc = matmul(&acc, &self.weight)?;
        }
        Ok(acc)
    }
}

impl JointVelocity for Costly {
    fn forward(&mut self, step: &JointStep<'_>) -> Result<(Array, Array)> {
        self.calls += 1;
        // Real work whose result feeds the returned velocity, so it cannot be optimized away —
        // scaled to nothing so the schedule stays well-conditioned, with a fixed 0.5 on top so the
        // Euler update actually moves the latents.
        let burned = self.burn()?;
        let k = mlx_rs::ops::add(
            &mlx_rs::ops::multiply(&burned.sum(None)?, Array::from_f32(1e-30))?,
            Array::from_f32(0.5),
        )?;
        Ok((
            mlx_rs::ops::multiply(step.video_rows, &k)?,
            mlx_rs::ops::multiply(step.audio_rows, &k)?,
        ))
    }
}

/// **The cancellation measurement.**
///
/// A render that deferred its compute to the caller's readback would still return `Canceled`
/// promptly on a mid-render trip — and would have stopped nothing, because the caller then pays for
/// the whole schedule outside every check. So this measures the split.
#[test]
fn the_render_core_keeps_its_compute_inside_the_cancel_checked_region() {
    const EVALS: usize = 6;
    const DIM: i32 = 1536;
    const MATMULS: usize = 8;

    let g = geometry();
    let l = layout(&g);
    let schedule = JointSchedule::new(EVALS + 1).expect("schedule");
    assert_eq!(schedule.num_evals(), EVALS);
    let (video, audio) = initial_latents(&g, PATCH_SIZE, 17).expect("latents");

    // Warm the kernels so the measured run is not timing Metal compilation.
    let mut warm = Costly::new(DIM, MATMULS);
    let out = render_latents(
        &mut warm,
        &l,
        &schedule,
        &video,
        &audio,
        PATCH_SIZE,
        &CancelFlag::default(),
        &mut |_| {},
    )
    .expect("warm render");
    mlx_rs::transforms::eval([&out.video, &out.audio]).unwrap();

    let mut model = Costly::new(DIM, MATMULS);
    let start = Instant::now();
    let rendered = render_latents(
        &mut model,
        &l,
        &schedule,
        &video,
        &audio,
        PATCH_SIZE,
        &CancelFlag::default(),
        &mut |_| {},
    )
    .expect("render");
    let in_region = start.elapsed();
    // Everything the caller's first host readback would still have to pay for.
    let _ = rendered.video.sum(None).unwrap().item::<f32>();
    let _ = rendered.audio.sum(None).unwrap().item::<f32>();
    let deferred = start.elapsed() - in_region;
    let total = in_region + deferred;
    let fraction = 100.0 * in_region.as_secs_f64() / total.as_secs_f64();

    println!(
        "  {EVALS} evaluations: {in_region:?} inside the cancel-checked region, {deferred:?} \
         deferred to the caller's readback -> {fraction:.1}% cancellable"
    );
    assert_eq!(
        model.calls, EVALS,
        "one forward per evaluation, no CFG pair"
    );
    assert!(
        total.as_millis() >= 8,
        "the synthetic model is too cheap to measure on this machine ({total:?}); raise \
         DIM/MATMULS"
    );
    assert!(
        fraction > 90.0,
        "only {fraction:.1}% of the render happened inside the cancel-checked region; the rest was \
         deferred to the caller's first readback, where no cancel check exists. The per-step eval \
         in `denoise_av` and the terminal eval in `render_latents` are what keep it there"
    );

    // --- the mutation arm ----------------------------------------------------------------------
    // The same measurement over an eval-free reimplementation of the render: the identical model,
    // the identical schedule, the identical row bookkeeping — with **no `eval` anywhere**. That is
    // exactly the shape a port arrives at by writing the obvious loop, and it is the shape a cancel
    // *latency* assertion prefers, because it returns fastest of all.
    //
    // Without this arm the 98%+ above would be a number with nothing to compare against; with it,
    // the measurement is shown to separate the two implementations.
    let mut mutant = Costly::new(DIM, MATMULS);
    let start = Instant::now();
    let (mv, ma) = denoise_without_any_eval(&mut mutant, &l, &schedule, &video, &audio);
    let unpacked = mlx_gen_minimax_h3::unpatchify_video_rows(
        &mv,
        24,
        g.joint.num_latent_frames,
        g.joint.latent_height,
        g.joint.latent_width,
        PATCH_SIZE,
    )
    .unwrap();
    let audio_unpacked =
        mlx_gen_minimax_h3::unpack_audio_rows(&ma, g.joint.num_audio_latents, 2, 32).unwrap();
    let mutant_in_region = start.elapsed();
    let _ = unpacked.sum(None).unwrap().item::<f32>();
    let _ = audio_unpacked.sum(None).unwrap().item::<f32>();
    let mutant_deferred = start.elapsed() - mutant_in_region;
    let mutant_fraction =
        100.0 * mutant_in_region.as_secs_f64() / (mutant_in_region + mutant_deferred).as_secs_f64();
    println!(
        "  MUTATION (no eval at all): {mutant_in_region:?} in-region, {mutant_deferred:?} \
         deferred -> {mutant_fraction:.1}% cancellable"
    );
    assert_eq!(mutant.calls, EVALS, "the mutant ran the same schedule");
    assert!(
        mutant_fraction < 50.0 && mutant_fraction < fraction - 20.0,
        "the eval-free mutant must defer most of its compute past every cancel check \
         ({mutant_fraction:.1}% against the real path's {fraction:.1}%); if it does not, this \
         measurement cannot tell a cancellable render from an uncancellable one"
    );
}

/// The render loop **minus every `eval`** — the mutation arm's implementation.
///
/// Deliberately a reimplementation rather than a flag on the real one: a production knob that
/// disables the thing correctness depends on is a worse artifact than a test-local copy.
fn denoise_without_any_eval(
    model: &mut dyn JointVelocity,
    layout: &PackedLayout,
    schedule: &JointSchedule,
    video: &Array,
    audio: &Array,
) -> (Array, Array) {
    let adaln = mlx_gen_minimax_h3::adaln_schedule(schedule).unwrap();
    let mut vlat = video.clone();
    let mut alat = audio.clone();
    for i in 0..schedule.num_evals() {
        let adaln_indices = adaln
            .adaln_indices(i, layout.row_classes(), layout.token_tags())
            .unwrap();
        let row_timesteps = PackedLayout::row_timesteps(
            schedule.video().timesteps()[i],
            schedule.audio().timesteps()[i],
        );
        let (vvel, avel) = model
            .forward(&JointStep {
                index: i,
                layout,
                video_rows: &vlat,
                audio_rows: &alat,
                adaln_indices: &adaln_indices,
                row_timesteps,
            })
            .unwrap();
        vlat = schedule.video().step(i, &vlat, &vvel).unwrap();
        alat = schedule.audio().step(i, &alat, &avel).unwrap();
        // ...and no `eval` here, which is the whole mutation.
    }
    (vlat, alat)
}

/// A cancel tripped from the progress tick stops the loop at a step boundary and returns the typed
/// error; the model is not called again.
#[test]
fn a_cancel_stops_the_render_and_returns_the_typed_error() {
    let g = geometry();
    let l = layout(&g);
    let schedule = JointSchedule::new(7).unwrap();
    let (video, audio) = initial_latents(&g, PATCH_SIZE, 3).unwrap();

    let cancel = CancelFlag::default();
    let trip = cancel.clone();
    let mut model = Costly::new(128, 1);
    let err = render_latents(
        &mut model,
        &l,
        &schedule,
        &video,
        &audio,
        PATCH_SIZE,
        &cancel,
        &mut |completed| {
            if completed == 2 {
                trip.cancel();
            }
        },
    )
    .expect_err("a tripped cancel must surface");
    assert!(matches!(err, mlx_gen::Error::Canceled), "{err}");
    assert_eq!(model.calls, 2, "the model must not run after the cancel");

    // ...and a cancel tripped after the last step still stops the tail rather than decoding.
    let cancel = CancelFlag::default();
    let trip = cancel.clone();
    let mut model = Costly::new(128, 1);
    let err = render_latents(
        &mut model,
        &l,
        &schedule,
        &video,
        &audio,
        PATCH_SIZE,
        &cancel,
        &mut |completed| {
            if completed == schedule.num_evals() {
                trip.cancel();
            }
        },
    )
    .expect_err("the tail must observe a late cancel");
    assert!(matches!(err, mlx_gen::Error::Canceled), "{err}");
}

/// The render core returns the two shapes the VAEs consume, and the denoise really moved the
/// latents.
#[test]
fn the_render_core_returns_decodable_latents() {
    let g = geometry();
    let l = layout(&g);
    let schedule = JointSchedule::new(4).unwrap();
    let (video, audio) = initial_latents(&g, PATCH_SIZE, 99).unwrap();

    let mut model = Costly::new(64, 1);
    let out = render_latents(
        &mut model,
        &l,
        &schedule,
        &video,
        &audio,
        PATCH_SIZE,
        &CancelFlag::default(),
        &mut |_| {},
    )
    .unwrap();

    assert_eq!(
        out.video.shape(),
        &[
            1,
            24,
            g.joint.num_latent_frames,
            g.joint.latent_height,
            g.joint.latent_width
        ]
    );
    assert_eq!(out.audio.shape(), &[1, 2, 32, g.joint.num_audio_latents]);
    // The video latents are not the noise they started as.
    let start_rows = mlx_gen_minimax_h3::patchify_video_latents(&out.video, PATCH_SIZE).unwrap();
    let (peak, _) = common::rel(&start_rows, &video);
    assert!(
        peak > 1e-3,
        "the denoise must have moved the latents, got rel-max-abs {peak:.3e}"
    );
    assert!(
        out.video.sum(None).unwrap().item::<f32>().is_finite(),
        "the rendered latents carry NaN/Inf"
    );
}

/// Every legal duration builds a layout whose row count and audio unpack agree with the geometry —
/// the 14 lattice points, not just the one the render uses.
#[test]
fn every_legal_duration_builds_a_consistent_layout() {
    for &frames in &LEGAL_FRAME_COUNTS {
        let g = resolve_geometry(WIDTH, HEIGHT, frames).unwrap();
        let l = layout(&g);
        let rows_per_frame = l.rows_per_frame();
        assert_eq!(
            l.seq_len(),
            6 + g.joint.num_audio_latents * 2 + g.joint.num_latent_frames * rows_per_frame,
            "{frames} frames: text + stereo audio + video"
        );
        assert_eq!(l.num_condition_video_rows(), 0);
        // The unpack contract the decode depends on.
        let rows = Array::from_slice(
            &vec![0.25f32; (g.joint.num_audio_latents * 2 * 32) as usize],
            &[1, g.joint.num_audio_latents * 2, 32],
        );
        let out =
            mlx_gen_minimax_h3::unpack_audio_rows(&rows, g.joint.num_audio_latents, 2, 32).unwrap();
        assert_eq!(out.shape(), &[1, 2, 32, g.joint.num_audio_latents]);
    }
}
