//! FlowDPM-Solver++ (`iris3b/flow/solver.py`) applied to Candle tensors. The schedule and every
//! scalar coefficient come from the backend-neutral plan ([`gen_core::iris::dpm_solver_plan`]); this
//! module only applies them, exactly as `mlx_gen_iris::solver` does. The integration state (`x`, the
//! `x0` history) is **f32** throughout, as upstream keeps it (`z.to(float32)`, `out.x.float()`).
//!
//! [`gen_core::iris::dpm_solver_plan`]: candle_gen::gen_core::iris::dpm_solver_plan

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::gen_core::iris::{DpmStep, DpmUpdate};
use candle_gen::gen_core::CancelFlag;
use candle_gen::{CandleError as Error, Result};

/// `x · c` with `c` an f32 scalar tensor — an elementwise f32 multiply, the same rounding as the
/// tensor-by-scalar products upstream and MLX perform.
fn scale(x: &Tensor, c: f32) -> Result<Tensor> {
    let c = Tensor::new(c, x.device())?;
    Ok(x.broadcast_mul(&c)?)
}

/// Integrate `z` along `plan`. `model_out(x, step)` returns the raw (CFG-combined) network output
/// at `step.s`; `on_step(i)` fires after step `i` (1-based) completes. Cancellation is checked before
/// every network evaluation and surfaces as [`Error::Canceled`] — the partial state is dropped,
/// never returned.
pub fn sample(
    z: &Tensor,
    plan: &[DpmStep],
    cancel: &CancelFlag,
    mut model_out: impl FnMut(&Tensor, &DpmStep) -> Result<Tensor>,
    mut on_step: impl FnMut(usize),
) -> Result<Tensor> {
    let mut x = z.to_dtype(DType::F32)?;
    let mut prev_x0: Option<Tensor> = None;
    for (i, step) in plan.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let out = model_out(&x, step)?.to_dtype(DType::F32)?;
        // v-prediction: x0 = x − s·v
        let x0 = x.sub(&scale(&out, step.s_f32())?)?;
        x = match step.update {
            DpmUpdate::First { cx, c0 } => scale(&x, cx)?.sub(&scale(&x0, c0)?)?,
            DpmUpdate::Second { cx, c0, c1, r0 } => {
                let prev = prev_x0.as_ref().ok_or_else(|| {
                    Error::Msg("iris solver: a second-order step needs the previous x0".into())
                })?;
                let d = x0
                    .sub(prev)?
                    .broadcast_div(&Tensor::new(r0, x0.device())?)?;
                scale(&x, cx)?.sub(&scale(&x0, c0)?)?.sub(&scale(&d, c1)?)?
            }
        };
        prev_x0 = Some(x0);
        on_step(i + 1);
    }
    Ok(x)
}

/// `out_uncond + scale · (out_cond − out_uncond)` in f32.
pub fn cfg_combine(out_uncond: &Tensor, out_cond: &Tensor, cfg_scale: f32) -> Result<Tensor> {
    let u = out_uncond.to_dtype(DType::F32)?;
    let c = out_cond.to_dtype(DType::F32)?;
    Ok(u.add(&scale(&c.sub(&u)?, cfg_scale)?)?)
}
