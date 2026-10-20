//! Upstream's sampling loop (`inference.py::generate`), as free functions so the weights-free tiny
//! DiT fixture exercises exactly the code the provider runs: uniform Euler integration of the flow
//! ODE from noise (`t = 0`) to data, with batched classifier-free guidance when an unconditional
//! context is supplied.

use mlx_gen::gen_core::supra2::{euler_dt, euler_timesteps};
use mlx_gen::{CancelFlag, Error, Progress, Result};
use mlx_rs::ops::{add, concatenate_axis, multiply, split, subtract};
use mlx_rs::transforms::eval;
use mlx_rs::Array;

use crate::dit::SupraDit;

/// A batch of text conditioning: context `[B, M, ctx_dim]` and its additive mask `[B, 1, 1, M]`.
#[derive(Clone, Copy)]
pub struct ContextBatch<'a> {
    pub context: &'a Array,
    pub mask: &'a Array,
}

/// Repeat one prompt's `[1, M, D]` context and 0/1 mask `n` times (upstream's
/// `[prompt] * n` / `.expand(n, …)`), returning the context and its additive mask.
pub fn broadcast(context: &Array, mask01: &[f32], n: usize) -> Result<(Array, Array)> {
    let ctx = concatenate_axis(&vec![context.clone(); n], 0)?;
    let mask = SupraDit::additive_mask(&mask01.repeat(n), n, mask01.len())?;
    Ok((ctx, mask))
}

/// One velocity evaluation, upstream's loop body: with `uncond`, the conditional and unconditional
/// halves run as one `2B` batch and mix as `v_u + g·(v_c − v_u)`; without, a plain conditional pass.
pub fn velocity(
    dit: &SupraDit,
    z: &Array,
    t: f32,
    cond: ContextBatch<'_>,
    uncond: Option<ContextBatch<'_>>,
    guidance: f32,
) -> Result<Array> {
    let n = z.shape()[0] as usize;
    match uncond {
        Some(u) => {
            let z2 = concatenate_axis(&[z, z], 0)?;
            let tf = dit.timestep_features(&vec![t; 2 * n]);
            let ctx = concatenate_axis(&[cond.context, u.context], 0)?;
            let mask = concatenate_axis(&[cond.mask, u.mask], 0)?;
            let both = dit.forward(&z2, &tf, &ctx, &mask)?;
            let halves = split(&both, 2, 0)?;
            let (vc, vu) = (&halves[0], &halves[1]);
            let g = Array::from_f32(guidance);
            Ok(add(vu, &multiply(&g, &subtract(vc, vu)?)?)?)
        }
        None => {
            let tf = dit.timestep_features(&vec![t; n]);
            dit.forward(z, &tf, cond.context, cond.mask)
        }
    }
}

/// Integrate `z0` over `steps` uniform Euler steps (`z += dt · v`). CFG runs iff `uncond` is
/// supplied — callers pass it exactly when [`cfg_enabled`](mlx_gen::gen_core::supra2::cfg_enabled)
/// holds for `guidance`. Checks `cancel` before every step and reports 1-based step progress.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    dit: &SupraDit,
    z0: Array,
    steps: u32,
    guidance: f32,
    cond: ContextBatch<'_>,
    uncond: Option<ContextBatch<'_>>,
    cancel: &CancelFlag,
    progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    if steps == 0 {
        return Err(Error::Unsupported(
            "supra2: steps must be at least 1".into(),
        ));
    }
    let dt = Array::from_f32(euler_dt(steps));
    let mut z = z0;
    for (i, t) in euler_timesteps(steps).into_iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(Error::Canceled);
        }
        let v = velocity(dit, &z, t, cond, uncond, guidance)?;
        z = add(&z, &multiply(&dt, &v)?)?;
        eval([&z])?;
        progress(Progress::Step {
            current: i as u32 + 1,
            total: steps,
        });
    }
    Ok(z)
}
