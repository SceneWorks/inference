//! The Iris training optimizers, native: `torch.optim.AdamW` and the hybrid Dion Muon of
//! `iris3b/train/optim.py` (`microsoft/dion@58d38adb` `Muon`, `use_triton=True,
//! use_polar_express=False` ⇒ the quintic Newton–Schulz of `newton_schulz_triton`, identical math to
//! `zeropower_via_newtonschulz5`), with the backend-neutral routing of
//! [`gen_core::iris::train::full_param_route`] / [`adapter_param_route`].
//!
//! * **AdamW** (both the plain optimizer and Muon's AdamW groups — Dion's fused AdamW is the same
//!   update): `p ← p·(1 − lr·wd)`; `m ← β₁m + (1−β₁)g`; `v ← β₂v + (1−β₂)g²`;
//!   `p ← p − (lr / (1 − β₁ᵗ)) · m / (√v / √(1 − β₂ᵗ) + ε)`, all f32.
//! * **Muon**: `M ← μM + G` (f32); `U = μM + G` under Nesterov, else `M`; `U → bf16`; orthogonalize
//!   (per row block for a fused matrix, each block rescaled by its own adjusted-lr ratio over the
//!   whole matrix's); `p ← p·(1 − lr·wd)`; `p ← p − (lr · adjust(shape)) · U` (the product rounded
//!   to bf16 as torch's `_foreach_mul(U_bf16, lr)` is, then subtracted from the f32 parameter).
//!
//! [`adapter_param_route`]: gen_core::iris::train::adapter_param_route

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use gen_core::iris::train::{
    muon_split_scales, MuonAdjustLr, OptimizerKind, OptimizerPlan, ParamRoute, ADAMW_EPSILON,
    MUON_EPSILON, NEWTON_SCHULZ_COEFFS,
};
use mlx_gen::gen_core;
use mlx_gen::{Error, Result};
use mlx_rs::ops::{concatenate_axis, matmul, sqrt, sum_axes};
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

/// A keyed tensor map (parameters, gradients, EMA, optimizer slots).
pub type Params = HashMap<Rc<str>, Array>;

fn s(v: f64) -> Array {
    Array::from_f32(v as f32)
}

/// Dion's Newton–Schulz orthogonalization of one matrix `g` (any float dtype): bf16 throughout,
/// wide orientation, Frobenius pre-normalisation by `‖X‖ + eps`, five quintic steps. Returns bf16 in
/// `g`'s orientation.
pub fn newton_schulz(g: &Array, eps: f64) -> Result<Array> {
    let sh = g.shape();
    let (rows, cols) = (sh[sh.len() - 2], sh[sh.len() - 1]);
    let mut x = g.as_dtype(Dtype::Bfloat16)?;
    let tall = rows > cols;
    if tall {
        x = x.t();
    }
    // torch `X.norm(dim=(-2,-1))` on bf16 accumulates in f32 and rounds the result to bf16; the
    // f32 `epsilon` tensor is a 0-d operand, so the sum stays bf16.
    let xf = x.as_dtype(Dtype::Float32)?;
    let norm = sqrt(&sum_axes(&xf.square()?, &[-2, -1], true)?)?.as_dtype(Dtype::Bfloat16)?;
    let denom = scale_bf16(&norm, 1.0, eps)?;
    x = x
        .as_dtype(Dtype::Float32)?
        .divide(&denom.as_dtype(Dtype::Float32)?)?
        .as_dtype(Dtype::Bfloat16)?;
    for (a, b, c) in NEWTON_SCHULZ_COEFFS {
        let a_mat = mm_bf16(&x, &x.t())?;
        let bb = add_bf16(
            &scale_bf16(&a_mat, b as f64, 0.0)?,
            &scale_bf16(&mm_bf16(&a_mat, &a_mat)?, c as f64, 0.0)?,
        )?;
        x = add_bf16(&scale_bf16(&x, a as f64, 0.0)?, &mm_bf16(&bb, &x)?)?;
    }
    if tall {
        x = x.t();
    }
    Ok(x)
}

/// torch's bf16-tensor-times-python-scalar (plus an optional scalar offset): computed in f32
/// ("opmath") from the exact bf16 values, rounded once to bf16.
fn scale_bf16(x: &Array, mul: f64, add: f64) -> Result<Array> {
    let y = x.as_dtype(Dtype::Float32)?.multiply(s(mul))?;
    let y = if add != 0.0 { y.add(s(add))? } else { y };
    Ok(y.as_dtype(Dtype::Bfloat16)?)
}

/// A bf16 GEMM with f32 accumulation and one rounding of the result (torch's bf16 `@` on CPU and
/// CUDA). The products of two bf16 values are exact in f32, so promoting the operands and rounding
/// the f32 product is that GEMM up to summation order — and it never touches MLX's CPU bf16 GEMM
/// path, which faults on some transposed narrow shapes.
fn mm_bf16(a: &Array, b: &Array) -> Result<Array> {
    Ok(
        matmul(&a.as_dtype(Dtype::Float32)?, &b.as_dtype(Dtype::Float32)?)?
            .as_dtype(Dtype::Bfloat16)?,
    )
}

/// bf16 + bf16, rounded once.
fn add_bf16(a: &Array, b: &Array) -> Result<Array> {
    Ok(a.as_dtype(Dtype::Float32)?
        .add(&b.as_dtype(Dtype::Float32)?)?
        .as_dtype(Dtype::Bfloat16)?)
}

/// Orthogonalize `u` (`[rows, cols]`), independently per equal row block when `split` is set,
/// each block scaled by its split scale (`_newton_schulz_row_blocks`).
fn orthogonalize(u: &Array, split: Option<usize>, adjust: MuonAdjustLr) -> Result<Array> {
    let Some(n) = split else {
        return newton_schulz(u, MUON_EPSILON);
    };
    let sh = u.shape();
    let (rows, cols) = (sh[0] as usize, sh[1] as usize);
    if !rows.is_multiple_of(n) {
        return Err(Error::Msg(format!(
            "iris muon: a {rows}-row matrix does not split into {n} equal blocks"
        )));
    }
    let block = rows / n;
    let scales = muon_split_scales(adjust, &vec![block; n], cols);
    let parts = u.split(n as i32, 0)?;
    let mut out = Vec::with_capacity(n);
    for (part, scale) in parts.iter().zip(scales) {
        let o = newton_schulz(part, MUON_EPSILON)?;
        // `block * split_scales[i]`: a bf16 tensor times a python float stays bf16.
        out.push(scale_bf16(&o, scale, 0.0)?);
    }
    Ok(concatenate_axis(&out, 0)?)
}

/// One tensor's optimizer slots.
enum Slot {
    Adam {
        m: Array,
        v: Array,
    },
    Muon {
        momentum: Array,
        split: Option<usize>,
    },
    Frozen,
}

/// The run's optimizer over a keyed parameter map.
pub struct IrisOptimizer {
    plan: OptimizerPlan,
    slots: HashMap<Rc<str>, Slot>,
    /// Optimizer steps taken (the AdamW bias-correction step `t` after the next increment).
    pub t: u64,
}

impl IrisOptimizer {
    /// Build zero state for `params` with each key's route.
    pub fn new(
        plan: &OptimizerPlan,
        params: &Params,
        route: impl Fn(&str, usize) -> ParamRoute,
    ) -> Result<Self> {
        let mut slots = HashMap::with_capacity(params.len());
        for (k, p) in params {
            let slot = match route(k, p.ndim()) {
                ParamRoute::Frozen => Slot::Frozen,
                ParamRoute::AdamW => Slot::Adam {
                    m: Array::zeros::<f32>(p.shape())?,
                    v: Array::zeros::<f32>(p.shape())?,
                },
                ParamRoute::Muon { split } => {
                    if p.ndim() != 2 {
                        return Err(Error::Msg(format!(
                            "iris muon: {k} is not a matrix ({:?})",
                            p.shape()
                        )));
                    }
                    Slot::Muon {
                        momentum: Array::zeros::<f32>(p.shape())?,
                        split,
                    }
                }
            };
            slots.insert(k.clone(), slot);
        }
        Ok(Self {
            plan: plan.clone(),
            slots,
            t: 0,
        })
    }

    pub fn kind(&self) -> OptimizerKind {
        self.plan.kind
    }

    /// Whether `key` is updated (not frozen).
    pub fn is_trained(&self, key: &str) -> bool {
        !matches!(self.slots.get(key), Some(Slot::Frozen) | None)
    }

    /// One optimizer step at learning rate `lr` over (clipped) `grads`, in place.
    pub fn step(&mut self, params: &mut Params, grads: &Params, lr: f64) -> Result<()> {
        self.t += 1;
        let (b1, b2) = self.plan.betas;
        let wd = self.plan.weight_decay;
        let eps = match self.plan.kind {
            OptimizerKind::AdamW => ADAMW_EPSILON,
            OptimizerKind::Muon => MUON_EPSILON,
        };
        let bc1 = 1.0 - b1.powi(self.t as i32);
        let bc2 = 1.0 - b2.powi(self.t as i32);
        let mu = self.plan.muon_momentum;
        let adjust = self.plan.muon_adjust_lr;
        let mut keys: Vec<Rc<str>> = params.keys().cloned().collect();
        keys.sort();
        let mut touched: Vec<Rc<str>> = Vec::with_capacity(keys.len());
        for k in keys {
            let Some(slot) = self.slots.get_mut(&k) else {
                return Err(Error::Msg(format!("iris optimizer: no state for {k}")));
            };
            if matches!(slot, Slot::Frozen) {
                continue;
            }
            let g = grads
                .get(&k)
                .ok_or_else(|| Error::Msg(format!("iris optimizer: no gradient for {k}")))?
                .as_dtype(Dtype::Float32)?;
            let p = params[&k].clone();
            let new_p = match slot {
                Slot::Frozen => unreachable!(),
                Slot::Adam { m, v } => {
                    let p = if wd != 0.0 {
                        p.multiply(s(1.0 - lr * wd))?
                    } else {
                        p
                    };
                    *m = m.multiply(s(b1))?.add(&g.multiply(s(1.0 - b1))?)?;
                    *v = v
                        .multiply(s(b2))?
                        .add(&g.square()?.multiply(s(1.0 - b2))?)?;
                    let denom = sqrt(&*v)?.divide(s(bc2.sqrt()))?.add(s(eps))?;
                    p.subtract(&m.divide(&denom)?.multiply(s(lr / bc1))?)?
                }
                Slot::Muon { momentum, split } => {
                    *momentum = momentum.multiply(s(mu))?.add(&g)?;
                    let u = if self.plan.muon_nesterov {
                        momentum.multiply(s(mu))?.add(&g)?
                    } else {
                        momentum.clone()
                    };
                    let u = orthogonalize(&u.as_dtype(Dtype::Bfloat16)?, *split, adjust)?;
                    let sh = p.shape();
                    let adjusted = lr * adjust.ratio(sh[0] as usize, sh[1] as usize);
                    let p = p.multiply(s(1.0 - lr * wd))?;
                    let step = scale_bf16(&u, adjusted, 0.0)?;
                    p.subtract(&step.as_dtype(Dtype::Float32)?)?
                }
            };
            params.insert(k.clone(), new_p);
            touched.push(k);
        }
        // Materialize parameters + state so the lazy graph never spans steps.
        let mut refs: Vec<&Array> = Vec::with_capacity(touched.len() * 3);
        for k in &touched {
            refs.push(&params[k]);
            match &self.slots[k] {
                Slot::Adam { m, v } => {
                    refs.push(m);
                    refs.push(v);
                }
                Slot::Muon { momentum, .. } => refs.push(momentum),
                Slot::Frozen => {}
            }
        }
        for chunk in refs.chunks(256) {
            eval(chunk.iter().copied())?;
        }
        Ok(())
    }

    /// Every state tensor under its checkpoint key (`‹key›::m`, `‹key›::v`, `‹key›::momentum`) plus
    /// the step counter `__t`.
    pub fn state_tensors(&self) -> Vec<(String, Array)> {
        let mut out = vec![("__t".to_string(), Array::from_slice(&[self.t as f32], &[1]))];
        for (k, slot) in &self.slots {
            match slot {
                Slot::Adam { m, v } => {
                    out.push((format!("{k}::m"), m.clone()));
                    out.push((format!("{k}::v"), v.clone()));
                }
                Slot::Muon { momentum, .. } => {
                    out.push((format!("{k}::momentum"), momentum.clone()))
                }
                Slot::Frozen => {}
            }
        }
        out
    }

    /// Restore state written by [`state_tensors`](Self::state_tensors) into this (identically
    /// routed) optimizer.
    pub fn load_state(&mut self, tensors: &HashMap<String, Array>) -> Result<()> {
        let t = tensors
            .get("__t")
            .ok_or_else(|| Error::Msg("iris optimizer state: missing __t".into()))?;
        self.t = t.as_dtype(Dtype::Float32)?.item::<f32>() as u64;
        for (k, slot) in self.slots.iter_mut() {
            let need = |suffix: &str| -> Result<Array> {
                tensors
                    .get(&format!("{k}::{suffix}"))
                    .cloned()
                    .ok_or_else(|| {
                        Error::Msg(format!(
                        "iris optimizer state: missing {k}::{suffix} (the checkpoint was written \
                         by a differently routed optimizer)"
                    ))
                    })
            };
            match slot {
                Slot::Adam { m, v } => {
                    *m = need("m")?;
                    *v = need("v")?;
                }
                Slot::Muon { momentum, .. } => *momentum = need("momentum")?,
                Slot::Frozen => {}
            }
        }
        Ok(())
    }

    /// Save the state to a safetensors file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let tensors = self.state_tensors();
        Array::save_safetensors(
            tensors.iter().map(|(k, v)| (k.as_str(), v)),
            None::<&HashMap<String, String>>,
            path,
        )?;
        Ok(())
    }
}

/// `clip_grad_norm_(params, max_norm)`: the global L2 norm over every gradient (f32) and the
/// clipped map. Returns `(total_norm, clipped)`.
pub fn clip_grads(grads: Params, max_norm: f64) -> Result<(f64, Params)> {
    let mut total = Array::from_f32(0.0);
    let mut keys: Vec<Rc<str>> = grads.keys().cloned().collect();
    keys.sort();
    for k in &keys {
        let g = grads[k].as_dtype(Dtype::Float32)?;
        let axes: Vec<i32> = (0..g.ndim() as i32).collect();
        total = total.add(&sum_axes(&g.square()?, &axes[..], false)?)?;
    }
    let total = sqrt(&total)?.item::<f32>() as f64;
    let coef = gen_core::iris::train::clip_coefficient(max_norm, total);
    let c = s(coef);
    let mut out = HashMap::with_capacity(grads.len());
    for (k, g) in grads {
        out.insert(k, g.as_dtype(Dtype::Float32)?.multiply(&c)?);
    }
    Ok((total, out))
}

/// `EMA.update`: `e ← decay·e + (1 − decay)·p` for every key of `ema`.
pub fn ema_update(ema: &mut Params, params: &Params, decay: f64) -> Result<()> {
    let keys: Vec<Rc<str>> = ema.keys().cloned().collect();
    for k in &keys {
        let p = params
            .get(k)
            .ok_or_else(|| Error::Msg(format!("iris ema: no parameter {k}")))?;
        let e = ema[k]
            .multiply(s(decay))?
            .add(&p.multiply(s(1.0 - decay))?)?;
        ema.insert(k.clone(), e);
    }
    let refs: Vec<&Array> = ema.values().collect();
    for chunk in refs.chunks(256) {
        eval(chunk.iter().copied())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every factor shape the miniature exercises, on both streams: near-orthogonal output (the
    /// quintic leaves singular values in roughly [0.7, 1.2]) and CPU/GPU agreement to bf16
    /// rounding. (MLX's CPU bf16 GEMM faults on some transposed narrow shapes such as 17x2 —
    /// `mm_bf16` keeps the orthogonalization off that path; this test is the regression.)
    #[test]
    fn newton_schulz_orthogonalizes_every_factor_shape() {
        let key = mlx_rs::random::key(3).unwrap();
        for (r, c) in [
            (17, 2),
            (2, 8),
            (10, 8),
            (4, 8),
            (8, 8),
            (2, 17),
            (10, 17),
            (17, 10),
            (1, 4),
        ] {
            let g = mlx_rs::random::normal::<f32>(&[r, c][..], None, None, Some(&key)).unwrap();
            let gpu = newton_schulz(&g, 1e-8)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap();
            let cpu = mlx_rs::with_new_default_stream(mlx_rs::Stream::cpu(), || {
                let o = newton_schulz(&g, 1e-8)
                    .unwrap()
                    .as_dtype(Dtype::Float32)
                    .unwrap();
                eval([&o]).unwrap();
                o
            });
            let wide = if r > c { gpu.t() } else { gpu.clone() };
            let gram = matmul(&wide, wide.t()).unwrap();
            let k = r.min(c);
            let diag: Vec<f32> = gram
                .reshape(&[-1])
                .unwrap()
                .as_slice::<f32>()
                .iter()
                .step_by(k as usize + 1)
                .copied()
                .collect();
            assert!(
                diag.iter().all(|d| (0.5..1.4).contains(d)),
                "{r}x{c}: gram diagonal {diag:?}"
            );
            let diff = gpu
                .subtract(&cpu)
                .unwrap()
                .abs()
                .unwrap()
                .max(None)
                .unwrap()
                .item::<f32>();
            assert!(diff <= 3e-2, "{r}x{c}: cpu vs gpu {diff}");
        }
    }
}
