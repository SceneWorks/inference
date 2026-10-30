//! Upstream's sampling loop (`inference.py::generate`) on Candle — the same free functions as the MLX
//! twin's `sampling` module (`broadcast`, `velocity`, `denoise` over a [`ContextBatch`]), so the
//! weights-free tiny fixture exercises exactly the code the provider runs on both backends: uniform
//! Euler integration of the flow ODE from noise (`t = 0`) to data, with batched classifier-free
//! guidance when an unconditional context is supplied.

use candle_gen::candle_core::Tensor;
use candle_gen::gen_core::{CancelFlag, Progress};
use candle_gen::{CandleError as Error, Result};
use gen_core_checkpoint::supra2::{euler_dt, euler_timesteps};

use crate::dit::SupraDit;

/// A batch of text conditioning: context `[B, M, ctx_dim]` and its additive mask `[B, 1, 1, M]`.
#[derive(Clone, Copy)]
pub struct ContextBatch<'a> {
    pub context: &'a Tensor,
    pub mask: &'a Tensor,
}

/// Repeat one prompt's `[1, M, D]` context and 0/1 mask `n` times (upstream's
/// `[prompt] * n` / `.expand(n, …)`), returning the context and its additive mask (on the
/// context's device).
pub fn broadcast(context: &Tensor, mask01: &[f32], n: usize) -> Result<(Tensor, Tensor)> {
    let ctx = context.repeat((n, 1, 1))?;
    let mask = SupraDit::additive_mask(&mask01.repeat(n), n, mask01.len(), context.device())?;
    Ok((ctx, mask))
}

/// One velocity evaluation, upstream's loop body: with `uncond`, the conditional and unconditional
/// halves run as one `2B` batch and mix as `v_u + g·(v_c − v_u)`; without, a plain conditional pass.
pub fn velocity(
    dit: &SupraDit,
    z: &Tensor,
    t: f32,
    cond: ContextBatch<'_>,
    uncond: Option<ContextBatch<'_>>,
    guidance: f32,
) -> Result<Tensor> {
    let n = z.dim(0)?;
    match uncond {
        Some(u) => {
            let z2 = Tensor::cat(&[z, z], 0)?;
            let tf = dit.timestep_features(&vec![t; 2 * n])?;
            let ctx = Tensor::cat(&[cond.context, u.context], 0)?;
            let mask = Tensor::cat(&[cond.mask, u.mask], 0)?;
            let both = dit.forward(&z2, &tf, &ctx, &mask)?;
            let vc = both.narrow(0, 0, n)?;
            let vu = both.narrow(0, n, n)?;
            Ok((&vu + ((vc - &vu)? * guidance as f64)?)?)
        }
        None => {
            let tf = dit.timestep_features(&vec![t; n])?;
            dit.forward(z, &tf, cond.context, cond.mask)
        }
    }
}

/// Integrate `z0` over `steps` uniform Euler steps (`z += dt · v`). CFG runs iff `uncond` is
/// supplied — callers pass it exactly when
/// [`cfg_enabled`](gen_core_checkpoint::supra2::cfg_enabled) holds for `guidance`. Checks
/// `cancel` before every step and reports 1-based step progress.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    dit: &SupraDit,
    z0: Tensor,
    steps: u32,
    guidance: f32,
    cond: ContextBatch<'_>,
    uncond: Option<ContextBatch<'_>>,
    cancel: &CancelFlag,
    progress: &mut dyn FnMut(Progress),
) -> Result<Tensor> {
    if steps == 0 {
        return Err(Error::Unsupported(
            "supra2: steps must be at least 1".into(),
        ));
    }
    let dt = euler_dt(steps) as f64;
    let mut z = z0;
    for (i, t) in euler_timesteps(steps).into_iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let v = velocity(dit, &z, t, cond, uncond, guidance)?;
        z = (&z + (v * dt)?)?;
        progress(Progress::Step {
            current: i as u32 + 1,
            total: steps,
        });
    }
    Ok(z)
}
