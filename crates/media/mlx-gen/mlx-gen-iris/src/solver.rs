//! FlowDPM-Solver++ (`iris3b/flow/solver.py`) applied to MLX tensors. The schedule and every
//! scalar coefficient come from the backend-neutral plan ([`gen_core::iris::dpm_solver_plan`]); this
//! module only applies them. The integration state (`x`, the `x0` history) is **f32** throughout,
//! as upstream keeps it (`z.to(float32)`, `out.x.float()`).

use gen_core::iris::{DpmStep, DpmUpdate, Prediction};
use mlx_gen::gen_core;
use mlx_gen::{CancelFlag, Error, Result};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

/// Label of one solver update's iOS execution quantum.
pub(crate) const STEP_QUANTUM: &str = "Iris solver step";
/// Host-owned canonical plan and preview sigma payload, checked before allocation.
pub(crate) const PLAN_QUANTUM: &str = "Iris solver plan";
/// The final pixel-space clamp, evaluated before the solve returns and releases its text batches.
pub(crate) const FINALIZE_QUANTUM: &str = "Iris solver final clamp";

/// `clip(x, -1, 1)` is a maximum followed by a minimum: two f32 arrays and two f32 bounds.
pub(crate) fn final_clamp_scratch(elements: u64) -> u64 {
    2 * crate::retention::allocated(mlx_gen::ios_quantum::tensor_bytes(4, &[elements])) + 8
}

/// Integrate `z` along `plan`. `model_out(x, step)` returns the raw (CFG-combined) network output
/// at `step.s` in f32, read as `prediction` (`_pred_x0`: velocity `x0 = x − s·out`, or the clean
/// image `x0 = out`); `on_step(i, x0)` fires after step `i` (1-based) completes with that step's
/// predicted clean image. Cancellation is checked before every network evaluation and surfaces as
/// [`Error::Canceled`] — the partial state is dropped, never returned.
pub fn sample(
    z: &Array,
    plan: &[DpmStep],
    prediction: Prediction,
    cancel: &CancelFlag,
    mut model_out: impl FnMut(&Array, &DpmStep) -> Result<Array>,
    mut on_step: impl FnMut(usize, &Array),
) -> Result<Array> {
    let mut x = z.as_dtype(Dtype::Float32)?;
    let mut prev_x0: Option<Array> = None;
    let mut completed_steps = 0u64;
    for (i, step) in plan.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        mlx_gen::ios_quantum::diagnostic_metadata(
            crate::model::MODEL_ID,
            "solver",
            &[
                ("step", i as u64),
                ("steps", plan.len() as u64),
                ("first", u64::from(i == 0)),
                ("terminal", u64::from(i + 1 == plan.len())),
                (
                    "update_order",
                    match step.update {
                        DpmUpdate::First { .. } => 1,
                        DpmUpdate::Second { .. } => 2,
                    },
                ),
                ("history_present", u64::from(prev_x0.is_some())),
                (
                    "history_rows",
                    prev_x0
                        .as_ref()
                        .map_or(0, |previous| previous.shape()[0] as u64),
                ),
                ("state_rows", x.shape()[0] as u64),
                ("state_f32", u64::from(x.dtype() == Dtype::Float32)),
                // This is completed-loop evidence, not a native availability assertion on lazy state.
                ("prior_completed_steps", completed_steps),
            ],
        );
        #[cfg(all(test, target_os = "macos"))]
        crate::anchor_tests::observe_solver_epoch(i, step, &x, prev_x0.as_ref());
        let out = model_out(&x, step)?.as_dtype(Dtype::Float32)?;
        mlx_gen::ios_quantum::before(STEP_QUANTUM, step_scratch(x.size() as u64, &step.update))?;
        let (next, x0) = step_update(&x, &out, step, prediction, prev_x0.as_ref())?;
        x = next;
        eval([&x, &x0])?;
        mlx_gen::ios_quantum::finish(STEP_QUANTUM, &[&x, &x0])?;
        completed_steps += 1;
        on_step(i + 1, &x0);
        prev_x0 = Some(x0);
    }
    Ok(x)
}

/// iOS floor of one solver update over `state_elements` f32 elements, counted under MLX's
/// retention ([`crate::retention`]) from [`step_update`] together with the lazy CFG combine it
/// materializes, each `[B, C, H, W]` f32 buffer rounded as allocated: the combine's difference (its
/// scaled copy and the sum donate), `x0` (`s·out` and the difference donate), the update's `cx·x`
/// and `c0·x0` (the difference donates) — three states for a first-order step — and for a
/// second-order step the difference to the previous `x0` (its divided and scaled copies donate).
/// Plus the update's scalar coefficients (four, six; 4 B each).
pub(crate) fn step_scratch(state_elements: u64, update: &DpmUpdate) -> u64 {
    let (states, scalars) = match update {
        DpmUpdate::First { .. } => (3, 4),
        DpmUpdate::Second { .. } => (4, 6),
    };
    states * crate::retention::allocated(4 * state_elements) + 4 * scalars
}

/// One FlowDPM-Solver++ update of `x` from the network output `out` (both f32): returns the new
/// state and this step's predicted clean image `x0` (the body of one [`STEP_QUANTUM`]).
pub(crate) fn step_update(
    x: &Array,
    out: &Array,
    step: &DpmStep,
    prediction: Prediction,
    prev_x0: Option<&Array>,
) -> Result<(Array, Array)> {
    let x0 = match prediction {
        // v-prediction: x0 = x − s·v
        Prediction::Velocity => x.subtract(&out.multiply(Array::from_f32(step.s_f32()))?)?,
        Prediction::Clean => out.clone(),
    };
    let next = match step.update {
        DpmUpdate::First { cx, c0 } => x
            .multiply(Array::from_f32(cx))?
            .subtract(&x0.multiply(Array::from_f32(c0))?)?,
        DpmUpdate::Second { cx, c0, c1, r0 } => {
            let prev = prev_x0.ok_or_else(|| {
                Error::Msg("iris solver: a second-order step needs the previous x0".into())
            })?;
            let d = x0.subtract(prev)?.divide(Array::from_f32(r0))?;
            x.multiply(Array::from_f32(cx))?
                .subtract(&x0.multiply(Array::from_f32(c0))?)?
                .subtract(&d.multiply(Array::from_f32(c1))?)?
        }
    };
    Ok((next, x0))
}

/// `out_uncond + scale · (out_cond − out_uncond)` in f32.
pub fn cfg_combine(out_uncond: &Array, out_cond: &Array, scale: f32) -> Result<Array> {
    let u = out_uncond.as_dtype(Dtype::Float32)?;
    let c = out_cond.as_dtype(Dtype::Float32)?;
    Ok(u.add(&c.subtract(&u)?.multiply(Array::from_f32(scale))?)?)
}
