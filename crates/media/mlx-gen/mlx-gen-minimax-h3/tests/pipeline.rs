//! The **`t2va` render core** (sc-17147), driven by a synthetic velocity model — no weights.
//!
//! # Why neither cancel *latency* nor a wall-clock *fraction* is the thing to measure
//!
//! sc-17146 established the first half and it applies unchanged one layer up. MLX is lazily
//! evaluated, so a loop that computes **nothing** returns `Canceled` *faster* than one that
//! computes everything: a latency assertion is satisfied best by the most broken implementation.
//! The property that actually matters is whether the render's compute happens **inside the
//! cancel-checked region** or is deferred past every check to the caller's first host readback.
//!
//! The obvious repair — time the call, time the readback, gate the ratio — reaches for the right
//! property through the wrong instrument. A wall-clock ratio moves with *host load*, not with the
//! code, and this test carried two of them. Reproducing the reported flake under deliberate load
//! (sc-19452) pinned down which one actually breaks, and it is not the one the reports named: the
//! **mutation arm's** ratio, which gates an eval-free render at `< 50%` and reads ~25% on a quiet
//! machine, came back at **72.9 / 75.0%**. Its in-region half is pure CPU graph-building, which
//! starves under CPU contention, while its deferred half is GPU compute, which does not starve in
//! step — so a busy host makes the *deliberately uncancellable* implementation look cancellable.
//! That also fits the three reported readings (65.9 / 74.1 / 76.8) far better than the real path's
//! ratio does: that one held at 94.8-99.1% across every loaded run measured here, well clear of
//! its `> 90.0` floor. Three false reds, each charged to whatever unrelated PR was in flight.
//!
//! The false-green side was **measured rather than inferred**: with the terminal `eval` in
//! [`render_latents`] deleted — a real regression in exactly the region under test — the old
//! assertions *passed*, at 97.8%, because the surviving per-step evals still dominate the clock. A
//! margin that swings 20-plus points for reasons unrelated to the code cannot also resolve the
//! code.
//!
//! So [`the_render_core_keeps_its_compute_inside_the_cancel_checked_region`] times **nothing**.
//! Lazy evaluation — the thing that makes timing treacherous here — is also what makes the
//! property *directly* observable: an MLX array carries an unscheduled graph until something
//! forces it, and [`is_materialized`] reads that bit straight off the array. Compute that happened
//! inside the cancel-checked region leaves its outputs materialized when the region ends; compute
//! deferred to the caller's readback leaves them unmaterialized. The test reads that bit at three
//! places — every step's inputs, [`mlx_gen_minimax_h3::denoise_av`]'s outputs, and
//! [`render_latents`]'s outputs — and its mutation arm shows all three readings inverted for an
//! eval-free render, so the instrument is demonstrated to discriminate rather than assumed to. No
//! clock is involved, so no amount of host load can move any of it.

mod common;

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

/// **The instrument.** Whether MLX has already *materialized* this array — whether the compute that
/// produces it has actually happened, rather than sitting in an unscheduled graph waiting for
/// somebody to force it.
///
/// `_mlx_array_is_available` is mlx-c's accessor for `mlx::core::array::is_available()`, which is
/// `false` for every array an op returned until an `eval` (or a host readback) schedules it *and*
/// its completion event signals. That makes it a direct, binary reading of the exact property the
/// cancellation contract depends on, with no clock and therefore no load sensitivity: an `eval`
/// inside the cancel-checked region flips it to `true` before the region ends, and a render that
/// defers its compute past every check leaves it `false`.
///
/// mlx-rs exposes no safe wrapper, so the call goes through `mlx-sys` — the same binding crate
/// `mlx_rs::Array` is built on, so [`Array::as_ptr`] hands back exactly the `mlx_array` this
/// signature wants. The `unsafe` obligation is only that the handle outlive the call, which the
/// `&Array` borrow guarantees; the C side merely reads a status word.
fn is_materialized(array: &Array) -> bool {
    let mut available = false;
    let status = unsafe { mlx_sys::_mlx_array_is_available(&mut available, array.as_ptr()) };
    assert_eq!(
        status, 0,
        "_mlx_array_is_available failed on a live array handle"
    );
    available
}

/// A velocity model that pays real GPU work per step **and records what it was handed**.
///
/// Returns a scaled multiple of its input so the loop's arithmetic stays well-conditioned, but pays
/// `matmuls` square matmuls first — the synthetic stand-in for 50 transformer blocks. Every forward
/// also stamps [`is_materialized`] over the two latent blocks it received, *before* touching them,
/// which is what turns "the loop forces each step inside itself" from a timing inference into an
/// observation: at step `i` those latents are step `i - 1`'s output, so a `false` says step
/// `i - 1`'s compute was still an unscheduled graph when the cancel check at the top of step `i`
/// ran — a check that would then have stopped nothing.
struct Costly {
    weight: Array,
    matmuls: usize,
    calls: usize,
    /// `(video_rows, audio_rows)` materialization, one entry per forward, in call order.
    inputs_materialized: Vec<(bool, bool)>,
}

impl Costly {
    fn new(dim: i32, matmuls: usize) -> Self {
        let n = (dim * dim) as usize;
        let v: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) / 512.0).collect();
        Self {
            weight: Array::from_slice(&v, &[dim, dim]),
            matmuls,
            calls: 0,
            inputs_materialized: Vec::new(),
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
        // Read the incoming latents' state FIRST: everything below builds new graph on top of them,
        // and the readings must describe what the loop handed over, not what this forward did.
        self.inputs_materialized.push((
            is_materialized(step.video_rows),
            is_materialized(step.audio_rows),
        ));
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

/// **The cancellation observation.**
///
/// A render that deferred its compute to the caller's readback would still return `Canceled`
/// promptly on a mid-render trip — and would have stopped nothing, because the caller then pays for
/// the whole schedule outside every check. So this observes where the compute actually landed, by
/// reading [`is_materialized`] rather than by timing anything (see the module docs for why the
/// wall-clock version of this test had to go).
///
/// Three readings, each pinned to the `eval` that produces it:
///
/// 1. every step's **inputs** — the per-step `eval` at the bottom of `denoise_av`'s loop;
/// 2. `denoise_av`'s **outputs** — the same `eval`, on the final step, which no later reading would
///    otherwise distinguish from the tail's;
/// 3. `render_latents`'s **outputs** — the terminal `eval` covering the unpatchify tail.
///
/// They are separate assertions on purpose. The single fraction they replace could not say *which*
/// `eval` had gone missing — deleting the terminal one alone left it reading 97.8%, a clean pass
/// (sc-19452) — so each is now gated on its own.
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

    // Materialize the inputs before anything under test runs. Two reasons: step 0 would otherwise
    // legitimately observe an unscheduled graph, saying nothing about the loop; and this doubles as
    // a self-check of the instrument, so a `false` here fails loudly instead of quietly making
    // every reading below meaningless.
    mlx_rs::transforms::eval([&video, &audio]).expect("materialize the initial latents");
    assert!(
        is_materialized(&video) && is_materialized(&audio),
        "the instrument is broken: `eval` did not mark its own outputs available, so nothing this \
         test reads below means anything"
    );

    // --- 1 + 2: the loop -------------------------------------------------------------------------
    // Driven through `denoise_av` directly rather than through `render_latents`, because the
    // terminal eval in the latter would materialize the last step's output regardless of whether
    // the loop had already done so — which is exactly the confusion the single fraction suffered.
    let adaln = mlx_gen_minimax_h3::adaln_schedule(&schedule).expect("adaln schedule");
    let mut looped = Costly::new(DIM, MATMULS);
    let (vout, aout) = mlx_gen_minimax_h3::denoise_av(
        &mut looped,
        &l,
        &schedule,
        &adaln,
        &video,
        &audio,
        &CancelFlag::default(),
        &mut |_| {},
    )
    .expect("denoise");
    println!(
        "  denoise_av step inputs materialized: {:?}",
        looped.inputs_materialized
    );
    assert_eq!(
        looped.calls, EVALS,
        "one forward per evaluation, no CFG pair"
    );
    assert_eq!(
        looped.inputs_materialized,
        vec![(true, true); EVALS],
        "every step must be handed latents whose compute has already landed. A `false` at index i \
         means step i-1's compute was still an unscheduled graph when the cancel check at the top \
         of step i ran, so a trip there would have stopped nothing — the per-step `eval` at the \
         bottom of `denoise_av`'s loop is what makes that check a real seam"
    );
    assert!(
        is_materialized(&vout) && is_materialized(&aout),
        "the LAST step's compute must land inside the loop too, before `denoise_av` returns; it is \
         the one step no later reading can attribute to the loop rather than to the tail"
    );

    // --- 3: the tail -----------------------------------------------------------------------------
    let mut model = Costly::new(DIM, MATMULS);
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
    assert_eq!(
        model.calls, EVALS,
        "one forward per evaluation, no CFG pair"
    );
    assert_eq!(
        model.inputs_materialized,
        vec![(true, true); EVALS],
        "the same per-step property, end to end through `render_latents`"
    );
    assert!(
        is_materialized(&rendered.video) && is_materialized(&rendered.audio),
        "`render_latents` returned latents whose compute had not happened. The unpatchify tail — \
         and, if the caller chains it, the decode — would then be paid by the caller's first host \
         readback, which sits outside every cancel check; the terminal `eval` in `render_latents` \
         is what keeps it inside"
    );

    // --- the mutation arm ------------------------------------------------------------------------
    // The same readings over an eval-free reimplementation of the render: the identical model, the
    // identical schedule, the identical row bookkeeping — with **no `eval` anywhere**. That is
    // exactly the shape a port arrives at by writing the obvious loop, and it is the shape a cancel
    // *latency* assertion prefers, because it returns fastest of all.
    //
    // Without this arm the `true`s above would be readings with nothing to compare against; with
    // it, the instrument is shown to separate the two implementations rather than to report `true`
    // unconditionally.
    let mut mutant = Costly::new(DIM, MATMULS);
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
    println!(
        "  MUTATION (no eval at all) step inputs materialized: {:?}",
        mutant.inputs_materialized
    );
    assert_eq!(mutant.calls, EVALS, "the mutant ran the same schedule");
    assert_eq!(
        mutant.inputs_materialized[0],
        (true, true),
        "the mutant's first step still receives the caller's already-materialized latents; if this \
         flipped, the readings below would be `false` for a reason other than the missing evals"
    );
    assert!(
        mutant.inputs_materialized[1..]
            .iter()
            .all(|&(v, a)| !v && !a),
        "an eval-free loop must hand every step after the first an unscheduled graph; it does not, \
         so this reading cannot tell a cancellable loop from an uncancellable one. Observed: {:?}",
        mutant.inputs_materialized
    );
    assert!(
        !is_materialized(&unpacked) && !is_materialized(&audio_unpacked),
        "an eval-free render must leave its whole schedule for the caller's first readback; it \
         does not, so this reading cannot tell a cancellable tail from an uncancellable one"
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
