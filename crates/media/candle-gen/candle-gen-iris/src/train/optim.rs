//! The Iris training optimizers on Candle — the twin of `mlx_gen_iris::train::optim`, op for op:
//! `torch.optim.AdamW` and the hybrid Dion Muon of `iris3b/train/optim.py`
//! (`microsoft/dion@58d38adb` `Muon`, `use_triton=True, use_polar_express=False` ⇒ the quintic
//! Newton–Schulz of `newton_schulz_triton`), with the backend-neutral routing of
//! [`full_param_route`] / [`adapter_param_route`].
//!
//! * **AdamW**: `p ← p·(1 − lr·wd)`; `m ← β₁m + (1−β₁)g`; `v ← β₂v + (1−β₂)g²`;
//!   `p ← p − (lr / (1 − β₁ᵗ)) · m / (√v / √(1 − β₂ᵗ) + ε)`, all f32.
//! * **Muon**: `M ← μM + G` (f32); `U = μM + G` under Nesterov, else `M`; `U → bf16`; orthogonalize
//!   in bf16 like upstream (per row block for a fused matrix, each block rescaled by its own
//!   adjusted-lr ratio over the whole matrix's); `p ← p·(1 − lr·wd)`; `p ← p − (lr·adjust)·U` (the
//!   product rounded to bf16 as torch's `_foreach_mul(U_bf16, lr)` is).
//!
//! Every scalar product runs against an f32 scalar tensor (`x · s`, `x / s` — torch's opmath and
//! MLX's f32 scalar), never candle's `affine` (which multiplies by a reciprocal for a division).
//! The bf16 Newton–Schulz GEMMs run as native bf16 GEMMs (f32 accumulation, one rounding — torch's
//! bf16 `@`) on a GPU; Candle's CPU backend has no half GEMM, so there they run as exactly that
//! contract spelled out (bf16 operands promoted, f32 product, one rounding).
//!
//! [`full_param_route`]: candle_gen::gen_core::iris::train::full_param_route
//! [`adapter_param_route`]: candle_gen::gen_core::iris::train::adapter_param_route

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_gen::candle_core::{DType, Tensor, Var};
use candle_gen::gen_core::iris::train::{
    clip_coefficient, muon_split_scales, MuonAdjustLr, OptimizerKind, OptimizerPlan, ParamRoute,
    ADAMW_EPSILON, MUON_EPSILON, NEWTON_SCHULZ_COEFFS,
};
use candle_gen::{CandleError as Error, Result};

/// The trainable tensors of a run (f32 `Var` masters), keyed and ordered by name.
pub type Params = BTreeMap<String, Var>;
/// A keyed tensor map (gradients, EMA, snapshots).
pub type Tensors = BTreeMap<String, Tensor>;

/// `x · v` against an f32 scalar.
pub fn mul_s(x: &Tensor, v: f64) -> Result<Tensor> {
    let s = Tensor::new(v as f32, x.device())?.to_dtype(x.dtype())?;
    Ok(x.broadcast_mul(&s)?)
}

/// `x / v` against an f32 scalar (a true division).
pub fn div_s(x: &Tensor, v: f64) -> Result<Tensor> {
    let s = Tensor::new(v as f32, x.device())?.to_dtype(x.dtype())?;
    Ok(x.broadcast_div(&s)?)
}

/// `x + v` against an f32 scalar.
pub fn add_s(x: &Tensor, v: f64) -> Result<Tensor> {
    let s = Tensor::new(v as f32, x.device())?.to_dtype(x.dtype())?;
    Ok(x.broadcast_add(&s)?)
}

/// The tensors behind `params` (shared storage, same autograd identity).
pub fn snapshot(params: &Params) -> Tensors {
    params
        .iter()
        .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
        .collect()
}

/// Deep copies of `params`' current values (an EMA seed or a saved snapshot — never aliased to a
/// `Var` the optimizer later overwrites in place).
pub fn detached_copy(params: &Params) -> Result<Tensors> {
    params
        .iter()
        .map(|(k, v)| Ok((k.clone(), v.as_tensor().detach().copy()?)))
        .collect()
}

/// torch's bf16-tensor-times-python-scalar (plus an optional scalar offset): computed in f32
/// ("opmath") from the exact bf16 values, rounded once to bf16.
fn scale_bf16(x: &Tensor, mul: f64, add: f64) -> Result<Tensor> {
    let y = mul_s(&x.to_dtype(DType::F32)?, mul)?;
    let y = if add != 0.0 { add_s(&y, add)? } else { y };
    Ok(y.to_dtype(DType::BF16)?)
}

/// A bf16 GEMM with f32 accumulation and one rounding of the result (torch's bf16 `@`).
fn mm_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.device().is_cpu() {
        // No CPU half GEMM: the products of two bf16 values are exact in f32, so promoting the
        // operands and rounding the f32 product is that GEMM up to summation order.
        Ok(a.to_dtype(DType::F32)?
            .matmul(&b.to_dtype(DType::F32)?)?
            .to_dtype(DType::BF16)?)
    } else {
        Ok(a.to_dtype(DType::BF16)?
            .contiguous()?
            .matmul(&b.to_dtype(DType::BF16)?.contiguous()?)?)
    }
}

/// bf16 + bf16, rounded once.
fn add_bf16(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    Ok(a.to_dtype(DType::F32)?
        .add(&b.to_dtype(DType::F32)?)?
        .to_dtype(DType::BF16)?)
}

/// Dion's Newton–Schulz orthogonalization of one matrix `g` (any float dtype): bf16 throughout,
/// wide orientation, Frobenius pre-normalisation by `‖X‖ + eps`, five quintic steps. Returns bf16 in
/// `g`'s orientation.
pub fn newton_schulz(g: &Tensor, eps: f64) -> Result<Tensor> {
    let (rows, cols) = g.dims2()?;
    let mut x = g.to_dtype(DType::BF16)?;
    let tall = rows > cols;
    if tall {
        x = x.t()?.contiguous()?;
    }
    // torch `X.norm(dim=(-2,-1))` on bf16 accumulates in f32 and rounds the result to bf16; the
    // f32 `epsilon` tensor is a 0-d operand, so the sum stays bf16.
    let norm = x
        .to_dtype(DType::F32)?
        .sqr()?
        .sum_all()?
        .sqrt()?
        .to_dtype(DType::BF16)?;
    let denom = scale_bf16(&norm, 1.0, eps)?;
    x = x
        .to_dtype(DType::F32)?
        .broadcast_div(&denom.to_dtype(DType::F32)?)?
        .to_dtype(DType::BF16)?;
    for (a, b, c) in NEWTON_SCHULZ_COEFFS {
        let a_mat = mm_bf16(&x, &x.t()?)?;
        let bb = add_bf16(
            &scale_bf16(&a_mat, b as f64, 0.0)?,
            &scale_bf16(&mm_bf16(&a_mat, &a_mat)?, c as f64, 0.0)?,
        )?;
        x = add_bf16(&scale_bf16(&x, a as f64, 0.0)?, &mm_bf16(&bb, &x)?)?;
    }
    if tall {
        x = x.t()?.contiguous()?;
    }
    Ok(x)
}

/// Orthogonalize `u` (`[rows, cols]`), independently per equal row block when `split` is set,
/// each block scaled by its split scale (`_newton_schulz_row_blocks`).
fn orthogonalize(u: &Tensor, split: Option<usize>, adjust: MuonAdjustLr) -> Result<Tensor> {
    let Some(n) = split else {
        return newton_schulz(u, MUON_EPSILON);
    };
    let (rows, cols) = u.dims2()?;
    if !rows.is_multiple_of(n) {
        return Err(Error::Msg(format!(
            "iris muon: a {rows}-row matrix does not split into {n} equal blocks"
        )));
    }
    let block = rows / n;
    let scales = muon_split_scales(adjust, &vec![block; n], cols);
    let mut out = Vec::with_capacity(n);
    for (i, scale) in scales.into_iter().enumerate() {
        let o = newton_schulz(&u.narrow(0, i * block, block)?, MUON_EPSILON)?;
        // `block * split_scales[i]`: a bf16 tensor times a python float stays bf16.
        out.push(scale_bf16(&o, scale, 0.0)?);
    }
    Ok(Tensor::cat(&out, 0)?)
}

/// One tensor's optimizer slots.
enum Slot {
    Adam {
        m: Tensor,
        v: Tensor,
    },
    Muon {
        momentum: Tensor,
        split: Option<usize>,
    },
    Frozen,
}

/// The run's optimizer over a keyed parameter map.
pub struct IrisOptimizer {
    plan: OptimizerPlan,
    slots: BTreeMap<String, Slot>,
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
        let mut slots = BTreeMap::new();
        for (k, p) in params {
            let p = p.as_tensor();
            let zeros = || Tensor::zeros(p.dims(), DType::F32, p.device());
            let slot = match route(k, p.rank()) {
                ParamRoute::Frozen => Slot::Frozen,
                ParamRoute::AdamW => Slot::Adam {
                    m: zeros()?,
                    v: zeros()?,
                },
                ParamRoute::Muon { split } => {
                    if p.rank() != 2 {
                        return Err(Error::Msg(format!(
                            "iris muon: {k} is not a matrix ({:?})",
                            p.dims()
                        )));
                    }
                    Slot::Muon {
                        momentum: zeros()?,
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
    pub fn step(&mut self, params: &Params, grads: &Tensors, lr: f64) -> Result<()> {
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
        let nesterov = self.plan.muon_nesterov;
        for (k, var) in params {
            let Some(slot) = self.slots.get_mut(k) else {
                return Err(Error::Msg(format!("iris optimizer: no state for {k}")));
            };
            if matches!(slot, Slot::Frozen) {
                continue;
            }
            let g = grads
                .get(k)
                .ok_or_else(|| Error::Msg(format!("iris optimizer: no gradient for {k}")))?
                .to_dtype(DType::F32)?;
            let p = var.as_tensor().detach();
            let new_p = match slot {
                Slot::Frozen => unreachable!(),
                Slot::Adam { m, v } => {
                    let p = if wd != 0.0 {
                        mul_s(&p, 1.0 - lr * wd)?
                    } else {
                        p
                    };
                    *m = mul_s(m, b1)?.add(&mul_s(&g, 1.0 - b1)?)?;
                    *v = mul_s(v, b2)?.add(&mul_s(&g.sqr()?, 1.0 - b2)?)?;
                    let denom = add_s(&div_s(&v.sqrt()?, bc2.sqrt())?, eps)?;
                    p.sub(&mul_s(&m.div(&denom)?, lr / bc1)?)?
                }
                Slot::Muon { momentum, split } => {
                    *momentum = mul_s(momentum, mu)?.add(&g)?;
                    let u = if nesterov {
                        mul_s(momentum, mu)?.add(&g)?
                    } else {
                        momentum.clone()
                    };
                    let u = orthogonalize(&u.to_dtype(DType::BF16)?, *split, adjust)?;
                    let (rows, cols) = p.dims2()?;
                    let adjusted = lr * adjust.ratio(rows, cols);
                    let p = mul_s(&p, 1.0 - lr * wd)?;
                    let step = scale_bf16(&u, adjusted, 0.0)?;
                    p.sub(&step.to_dtype(DType::F32)?)?
                }
            };
            var.set(&new_p)?;
        }
        Ok(())
    }

    /// Every state tensor under its checkpoint key (`‹key›::m`, `‹key›::v`, `‹key›::momentum`) plus
    /// the step counter `__t` — the MLX twin's layout, key for key.
    pub fn state_tensors(&self) -> Result<Vec<(String, Tensor)>> {
        let device = self
            .slots
            .values()
            .find_map(|s| match s {
                Slot::Adam { m, .. } => Some(m.device().clone()),
                Slot::Muon { momentum, .. } => Some(momentum.device().clone()),
                Slot::Frozen => None,
            })
            .unwrap_or(candle_gen::candle_core::Device::Cpu);
        let mut out = vec![(
            "__t".to_string(),
            Tensor::from_vec(vec![self.t as f32], (1,), &device)?,
        )];
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
        Ok(out)
    }

    /// Restore state written by [`state_tensors`](Self::state_tensors) (by either backend) into
    /// this (identically routed) optimizer.
    pub fn load_state(&mut self, tensors: &HashMap<String, Tensor>) -> Result<()> {
        let t = tensors
            .get("__t")
            .ok_or_else(|| Error::Msg("iris optimizer state: missing __t".into()))?;
        self.t = t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0] as u64;
        for (k, slot) in self.slots.iter_mut() {
            let need = |suffix: &str, like: &Tensor| -> Result<Tensor> {
                let t = tensors.get(&format!("{k}::{suffix}")).ok_or_else(|| {
                    Error::Msg(format!(
                        "iris optimizer state: missing {k}::{suffix} (the checkpoint was written \
                         by a differently routed optimizer)"
                    ))
                })?;
                if t.dims() != like.dims() {
                    return Err(Error::Msg(format!(
                        "iris optimizer state: {k}::{suffix} is {:?}, the run's is {:?}",
                        t.dims(),
                        like.dims()
                    )));
                }
                Ok(t.to_dtype(DType::F32)?.to_device(like.device())?)
            };
            match slot {
                Slot::Adam { m, v } => {
                    *m = need("m", m)?;
                    *v = need("v", v)?;
                }
                Slot::Muon { momentum, .. } => *momentum = need("momentum", momentum)?,
                Slot::Frozen => {}
            }
        }
        Ok(())
    }
}

/// `clip_grad_norm_(params, max_norm)`: the global L2 norm over every gradient (f32, accumulated
/// in key order as the MLX twin does) and the clipped map. Returns `(total_norm, clipped)`.
pub fn clip_grads(grads: Tensors, max_norm: f64) -> Result<(f64, Tensors)> {
    let mut total = 0f32;
    for g in grads.values() {
        let sq = g.to_dtype(DType::F32)?.sqr()?.sum_all()?.to_vec0::<f32>()?;
        total += sq;
    }
    let total = total.sqrt() as f64;
    let coef = clip_coefficient(max_norm, total);
    let mut out = BTreeMap::new();
    for (k, g) in grads {
        out.insert(k, mul_s(&g.to_dtype(DType::F32)?, coef)?);
    }
    Ok((total, out))
}

/// `EMA.update`: `e ← decay·e + (1 − decay)·p` for every key of `ema`.
pub fn ema_update(ema: &mut Tensors, params: &Params, decay: f64) -> Result<()> {
    for (k, e) in ema.iter_mut() {
        let p = params
            .get(k)
            .ok_or_else(|| Error::Msg(format!("iris ema: no parameter {k}")))?;
        // Detached: the EMA is plain data, never a node of the next step's autograd graph.
        *e = mul_s(e, decay)?.add(&mul_s(&p.as_tensor().detach(), 1.0 - decay)?)?;
    }
    Ok(())
}

/// Save keyed tensors to a safetensors file (sorted keys, no metadata) — a checkpoint member,
/// published by [`gen_core::iris::train::publish_checkpoint`]'s fsync + rename.
///
/// [`gen_core::iris::train::publish_checkpoint`]: candle_gen::gen_core::iris::train::publish_checkpoint
pub fn save_tensors(path: &Path, tensors: &[(String, Tensor)]) -> Result<()> {
    let mut list: Vec<(&str, &Tensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    list.sort_by(|a, b| a.0.cmp(b.0));
    safetensors::serialize_to_file(list, None, path)
        .map_err(|e| Error::Msg(format!("iris checkpoint: write {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    /// The quintic leaves singular values in roughly [0.7, 1.2]: every factor shape the miniature
    /// exercises comes out near-orthogonal, in bf16, in its own orientation.
    #[test]
    fn newton_schulz_orthogonalizes_every_factor_shape() {
        let dev = Device::Cpu;
        for (r, c) in [(17, 2), (2, 8), (10, 8), (8, 8), (2, 17), (17, 10), (1, 4)] {
            let g = Tensor::randn(0f32, 1.0, (r, c), &dev).unwrap();
            let o = newton_schulz(&g, 1e-8).unwrap();
            assert_eq!(o.dtype(), DType::BF16);
            assert_eq!(o.dims(), [r, c]);
            let o = o.to_dtype(DType::F32).unwrap();
            let wide = if r > c { o.t().unwrap() } else { o.clone() };
            let gram = wide.matmul(&wide.t().unwrap()).unwrap();
            let k = r.min(c);
            let diag: Vec<f32> = (0..k)
                .map(|i| {
                    gram.get(i)
                        .unwrap()
                        .get(i)
                        .unwrap()
                        .to_vec0::<f32>()
                        .unwrap()
                })
                .collect();
            assert!(
                diag.iter().all(|d| (0.5..1.4).contains(d)),
                "{r}x{c}: gram diagonal {diag:?}"
            );
        }
    }

    /// Clipping scales every gradient by `max_norm / (‖g‖ + 1e-6)` when over, and leaves it when
    /// under.
    #[test]
    fn clip_grads_uses_the_global_norm() {
        let dev = Device::Cpu;
        let mut g = Tensors::new();
        g.insert("a".into(), Tensor::new(&[3f32, 0.0], &dev).unwrap());
        g.insert("b".into(), Tensor::new(&[4f32], &dev).unwrap());
        let (norm, clipped) = clip_grads(g.clone(), 0.5).unwrap();
        assert!((norm - 5.0).abs() < 1e-6);
        let a = clipped["a"].to_vec1::<f32>().unwrap();
        assert!((a[0] - 0.3).abs() < 1e-5, "{a:?}");
        let (_, kept) = clip_grads(g, 10.0).unwrap();
        assert_eq!(kept["b"].to_vec1::<f32>().unwrap(), [4.0]);
    }
}
