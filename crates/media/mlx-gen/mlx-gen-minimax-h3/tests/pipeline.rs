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
//! code, and this test carried two of them. Reproducing the flake deliberately (sc-19452) showed
//! **both** break, and that the more fragile one is not the one the reports named. Sixteen loaded
//! runs of the pre-fix test produced five failures: four in the **mutation arm** — the control that
//! gates an eval-free render at `< 50%` and reads ~25% on a quiet machine — at 52.1 / 72.9 / 75.0%,
//! and one in the real path's `> 90.0`, at 86.3%. The mutant arm is the weaker of the two because
//! its in-region half is pure CPU graph-building, which starves under contention, while its
//! deferred half is GPU compute, which does not starve in step: a busy host makes the
//! *deliberately uncancellable* implementation look cancellable, so the guard's own control is what
//! load breaks first. The three readings the story reports (65.9 / 74.1 / 76.8) sit in that band
//! rather than the real path's. Either way, three false reds charged to unrelated PRs in flight.
//!
//! Neither ratio moves under CPU load alone — five runs at load average 18 stayed green, because
//! CPU contention scales both halves together. It took *cross-process* Metal contention on top.
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
//! clock is involved, so no amount of host load can move any of it — see [`is_materialized`] for
//! the one production property that guarantee rests on.

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
/// `_mlx_array_is_available` is mlx-c's accessor for `mlx::core::array::is_available()`: the status
/// is `available`, or it is `evaluated` with the completion event signalled or absent. A
/// **synchronous** `mlx_rs::transforms::eval` — the only kind this crate uses — attaches no event
/// and marks its outputs `evaluated` before returning, so the reading is exact and clock-free. That
/// is where the load-insensitivity comes from, and it is conditional: if a production `eval` here
/// were ever switched to `async_eval`, this would become a race against the event signal and would
/// have to `wait()` first. Anyone making that change has to revisit this test.
///
/// Two precision notes, because the obvious paraphrases are both wrong:
///
/// * it is **not** "`false` for anything an op returned". MLX short-circuits identity ops — a
///   `reshape` to the same shape, a full-range `slice`, an `astype` to the same dtype — by
///   returning the *input*, which keeps the input's status. A view defers no compute, so the
///   reading stays semantically right, but the mechanism is not "every op yields unscheduled";
/// * it is **not** a pure read. `is_available()` detaches the event and promotes the status on the
///   shared descriptor, non-atomically. Harmless under this repo's forced `RUST_TEST_THREADS = 1`,
///   but do not treat it as a race-free probe of an `Array` shared across threads.
///
/// mlx-rs exposes no safe wrapper, so the call goes through `mlx-sys` — the same binding crate
/// `mlx_rs::Array` is built on, so [`Array::as_ptr`] hands back exactly the `mlx_array` this
/// signature wants. Two obligations, not one: the handle must outlive the call, which the `&Array`
/// borrow guarantees; and some mlx-rs op must already have run on this thread, because mlx-rs
/// installs its error handler lazily inside its own wrappers and mlx-c's default handler is
/// `exit(-1)` rather than a status return. Every call site here runs after `initial_latents`.
fn is_materialized(array: &Array) -> bool {
    let mut available = false;
    let status = unsafe { mlx_sys::_mlx_array_is_available(&mut available, array.as_ptr()) };
    assert_eq!(
        status, 0,
        "_mlx_array_is_available failed on a live array handle"
    );
    available
}

/// A velocity model that does real GPU work per step **and records what it was handed**.
///
/// The work needs to be *real* — a genuine multi-primitive graph the loop has to force — but no
/// longer needs to be *large*: every surviving assertion reads a status bit, which is independent
/// of how long the graph takes. `DIM`/`MATMULS` were sized for the wall-clock version, whose
/// companion guard (`total.as_millis() >= 8`, "too cheap to measure on this machine; raise
/// DIM/MATMULS") went out with the clock, so they are now sized for a cheap test instead.
///
/// Returns a scaled multiple of its input so the loop's arithmetic stays well-conditioned, but pays
/// `matmuls` square matmuls first. Every forward
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
    const DIM: i32 = 256;
    const MATMULS: usize = 2;

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
        "the mutant's first step is still handed the caller's already-materialized latents — the \
         baseline the readings below are a departure from, and the thing that would break first if \
         this reimplementation stopped starting from the same place the real loop does"
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
        !is_materialized(&mv) && !is_materialized(&ma),
        "an eval-free loop must leave even its final rows unscheduled — asserted on the loop's own \
         outputs, not just on the unpacked ones, so the reading does not depend on whether the \
         unpatchify happens to be a view over them"
    );
    assert!(
        !is_materialized(&unpacked) && !is_materialized(&audio_unpacked),
        "an eval-free render must leave its whole schedule for the caller's first readback; it \
         does not, so this reading cannot tell a cancellable tail from an uncancellable one"
    );

    // The other direction of the control, and the coverage the timing version got for free by
    // forcing the mutant with `.sum().item()`. Read it back **after** every assertion above, so
    // the forcing cannot contaminate them, and assert two things: the eval-free reimplementation is
    // computable end to end rather than merely constructible, and the instrument flips `false` ->
    // `true` when compute is genuinely forced. Without this it could be stuck at `false` and every
    // mutant reading above would still pass.
    let video_sum = unpacked.sum(None).unwrap().item::<f32>();
    let audio_sum = audio_unpacked.sum(None).unwrap().item::<f32>();
    assert!(
        video_sum.is_finite() && audio_sum.is_finite(),
        "the eval-free mutant must still be a runnable stand-in for the real loop, not just a \
         graph that builds: got {video_sum} / {audio_sum}"
    );
    assert!(
        is_materialized(&unpacked) && is_materialized(&audio_unpacked),
        "the readback forced the mutant's whole schedule, so the instrument must now read `true` \
         for exactly the arrays it read `false` for a moment ago; if it does not, it is stuck low \
         and every `false` above was worthless"
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
