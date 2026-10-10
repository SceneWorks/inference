//! FlowDPM-Solver++ (`iris3b/flow/solver.py`) applied to MLX tensors. The schedule and every
//! scalar coefficient come from the backend-neutral plan ([`gen_core::iris::dpm_solver_plan`]); this
//! module only applies them. The integration state (`x`, the `x0` history) is **f32** throughout,
//! as upstream keeps it (`z.to(float32)`, `out.x.float()`).

use gen_core::iris::{DpmStep, DpmUpdate, Prediction};
use mlx_gen::gen_core;
use mlx_gen::{CancelFlag, Error, Result};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

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
    for (i, step) in plan.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let out = model_out(&x, step)?.as_dtype(Dtype::Float32)?;
        let x0 = match prediction {
            // v-prediction: x0 = x − s·v
            Prediction::Velocity => x.subtract(&out.multiply(Array::from_f32(step.s_f32()))?)?,
            Prediction::Clean => out,
        };
        x = match step.update {
            DpmUpdate::First { cx, c0 } => x
                .multiply(Array::from_f32(cx))?
                .subtract(&x0.multiply(Array::from_f32(c0))?)?,
            DpmUpdate::Second { cx, c0, c1, r0 } => {
                let prev = prev_x0.as_ref().ok_or_else(|| {
                    Error::Msg("iris solver: a second-order step needs the previous x0".into())
                })?;
                let d = x0.subtract(prev)?.divide(Array::from_f32(r0))?;
                x.multiply(Array::from_f32(cx))?
                    .subtract(&x0.multiply(Array::from_f32(c0))?)?
                    .subtract(&d.multiply(Array::from_f32(c1))?)?
            }
        };
        eval([&x, &x0])?;
        on_step(i + 1, &x0);
        prev_x0 = Some(x0);
    }
    Ok(x)
}

/// `out_uncond + scale · (out_cond − out_uncond)` in f32.
pub fn cfg_combine(out_uncond: &Array, out_cond: &Array, scale: f32) -> Result<Array> {
    let u = out_uncond.as_dtype(Dtype::Float32)?;
    let c = out_cond.as_dtype(Dtype::Float32)?;
    Ok(u.add(&c.subtract(&u)?.multiply(Array::from_f32(scale))?)?)
}
