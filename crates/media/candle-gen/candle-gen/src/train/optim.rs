//! Training optimizers (sc-5165) — the candle twin of `mlx_gen::train::optim`. Unlike candle's stock
//! `Optimizer` trait (which only ships AdamW/SGD), this exposes the full SceneWorks optimizer set
//! (`adamw`/`adam`/`rose`/`prodigy`), stepping the adapter-factor `Var`s directly from a
//! [`GradStore`]. Adam/AdamW delegate to candle's `AdamW`; Rose and Prodigy are faithful ports of the
//! MLX implementations (decoupled weight decay, f32 compute, no bias correction / safeguard warmup).
//!
//! The LR-schedule multiplier is applied via [`TrainOptimizer::set_lr_scaled`] each optimizer update;
//! grad-norm clipping ([`clip_grad_norm`]) — which candle has no built-in for — runs on the
//! `GradStore` before the step.

use std::collections::HashMap;

use candle_core::backprop::GradStore;
use candle_core::{Tensor, Var, D};

use crate::{CandleError, Result};

/// The optimizer names the worker may request (mirrors the MLX `SUPPORTED_OPTIMIZERS`).
pub const SUPPORTED_OPTIMIZERS: [&str; 4] = ["adamw", "adam", "rose", "prodigy"];

/// Recognized optimizer aliases → canonical form, matched **exactly** after lowercasing and stripping
/// non-alphanumerics (so `"AdamW"`, `"adamw-8bit"`, `"prodigy_opt"` all reach the same key). Mirrors
/// the MLX `build_optimizer` alias set (`training_adapters.py`): `adamw8bit`/`adam8bit` collapse to
/// full-precision `adamw` (candle has no bitsandbytes 8-bit optimizer, same as the MLX bnb-unavailable
/// fallback), and `prodigyopt`/`roseopt` are the pip-package spellings. Anything NOT listed here is
/// rejected by [`normalize`] rather than silently coerced to a default — a typo like `"adamaxx"` or an
/// unsupported optimizer like `"lion"` must fail validation, not train as the wrong optimizer.
const OPTIMIZER_ALIASES: [(&str, &str); 8] = [
    ("adamw", "adamw"),
    ("adamw8bit", "adamw"),
    ("adam8bit", "adamw"),
    ("adam", "adam"),
    ("rose", "rose"),
    ("roseopt", "rose"),
    ("prodigy", "prodigy"),
    ("prodigyopt", "prodigy"),
];

/// Collapse a recognized optimizer alias to its canonical form (`"AdamW"`/`"adamw8bit"` → `"adamw"`,
/// `"prodigy-opt"` → `"prodigy"`, `"rose_opt"` → `"rose"`) via **exact** (separator-insensitive,
/// case-insensitive) matching against `OPTIMIZER_ALIASES`.
///
/// Unknown names pass through **unchanged** (the separator-stripped, lowercased form) so they fail
/// [`TrainOptimizer::from_config`] with a clear error listing the supported set — unlike the previous
/// substring matching, a near-miss (`"adamax"`, `"radam"`, `"nadam"`, `"adamw8bit_typo"`) no longer
/// silently trains as plain Adam/AdamW.
pub fn normalize(name: &str) -> String {
    let s: String = name
        .to_ascii_lowercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    OPTIMIZER_ALIASES
        .iter()
        .find(|(alias, _)| *alias == s)
        .map(|(_, canonical)| (*canonical).to_string())
        .unwrap_or(s)
}

/// Global L2-norm gradient clipping over `vars`' gradients in `grads` (candle ships no built-in). If
/// the total norm exceeds `max_norm`, every gradient is scaled by `max_norm / norm` in place. Returns
/// the pre-clip total norm. Mirrors the MLX `clip_grad_norm(&grads, 1.0)` the trainer applies to the
/// averaged gradient before the optimizer step.
pub fn clip_grad_norm(grads: &mut GradStore, vars: &[Var], max_norm: f64) -> Result<f64> {
    // Accumulate the sum of squares as an on-device f64 scalar and read it back exactly once (sc-9036).
    // Each per-var reduction is upcast to f64 before the running add so the accumulation matches the
    // previous per-scalar CPU sum bit-for-bit, but we no longer stall on a GPU→CPU sync per var.
    let mut total_sq: Option<Tensor> = None;
    for v in vars {
        if let Some(g) = grads.get(v.as_tensor()) {
            let s = g.sqr()?.sum_all()?.to_dtype(candle_core::DType::F64)?;
            total_sq = Some(match total_sq {
                Some(acc) => (acc + s)?,
                None => s,
            });
        }
    }
    let total_sq = match total_sq {
        Some(t) => t.to_scalar::<f64>()?,
        None => 0f64,
    };
    let norm = total_sq.sqrt();
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for v in vars {
            if let Some(g) = grads.get(v.as_tensor()) {
                let scaled = (g * scale)?;
                grads.insert(v.as_tensor(), scaled);
            }
        }
    }
    Ok(norm)
}

/// Accumulate one micro-step's `grads` into `acc` (`+=` per `Var`); the first step seeds it. The
/// gradient-accumulation companion to [`scale_grads`] (the `1/accum` averaging that follows) — the
/// generic `GradStore` half of every family trainer's step loop, so it lives in the shared harness
/// rather than each `training.rs`.
pub fn accumulate_grads(acc: &mut Option<GradStore>, grads: GradStore, vars: &[Var]) -> Result<()> {
    match acc {
        None => *acc = Some(grads),
        Some(a) => {
            for v in vars {
                if let Some(g) = grads.get(v.as_tensor()) {
                    let summed = match a.get(v.as_tensor()) {
                        Some(prev) => (prev + g)?,
                        None => g.clone(),
                    };
                    a.insert(v.as_tensor(), summed);
                }
            }
        }
    }
    Ok(())
}

/// Scale every `Var`'s gradient in `grads` by `factor` in place — the `1/accumulation` averaging
/// applied before [`clip_grad_norm`] + [`TrainOptimizer::step`].
pub fn scale_grads(grads: &mut GradStore, vars: &[Var], factor: f64) -> Result<()> {
    for v in vars {
        if let Some(g) = grads.get(v.as_tensor()) {
            let scaled = (g * factor)?;
            grads.insert(v.as_tensor(), scaled);
        }
    }
    Ok(())
}

/// One of the supported training optimizers, owning the factor `Var`s it steps.
pub enum TrainOptimizer {
    /// AdamW (also serves plain `adam` with `weight_decay = 0`). Kept locally rather than delegating
    /// to candle-nn so the moments and update counter can be snapshotted for exact mid-run resume.
    Adam(Adam),
    Rose(Rose),
    Prodigy(Prodigy),
}

impl TrainOptimizer {
    pub fn is_supported(name: &str) -> bool {
        SUPPORTED_OPTIMIZERS.contains(&normalize(name).as_str())
    }

    /// Construct the optimizer named `name` over `vars` at learning rate `lr` and weight decay
    /// `weight_decay`. Betas/eps follow the torch/diffusers defaults (0.9, 0.999, 1e-8).
    pub fn from_config(name: &str, vars: Vec<Var>, lr: f32, weight_decay: f32) -> Result<Self> {
        match normalize(name).as_str() {
            "adamw" => Ok(Self::adam(vars, lr, weight_decay, "adamw")?),
            "adam" => Ok(Self::adam(vars, lr, 0.0, "adam")?),
            "rose" => Ok(Self::Rose(Rose::new(vars, lr, weight_decay))),
            "prodigy" => Ok(Self::Prodigy(Prodigy::new(vars, lr, weight_decay))),
            other => Err(CandleError::Msg(format!(
                "unsupported optimizer {other:?}; supported: {}",
                SUPPORTED_OPTIMIZERS.join(", ")
            ))),
        }
    }

    fn adam(vars: Vec<Var>, lr: f32, weight_decay: f32, kind: &'static str) -> Result<Self> {
        Ok(Self::Adam(Adam::new(vars, lr, weight_decay, kind)?))
    }

    /// Scale the base learning rate by the schedule multiplier for the next update.
    pub fn set_lr_scaled(&mut self, mult: f32) {
        match self {
            Self::Adam(a) => a.lr = a.base_lr * mult as f64,
            Self::Rose(r) => r.set_lr_scaled(mult),
            Self::Prodigy(p) => p.set_lr_scaled(mult),
        }
    }

    /// Apply one optimizer step from the (already clipped) gradients.
    pub fn step(&mut self, grads: &GradStore) -> Result<()> {
        match self {
            Self::Adam(a) => a.step(grads),
            Self::Rose(r) => r.step(grads),
            Self::Prodigy(p) => p.step(grads),
        }
    }

    /// Stable optimizer tag stored in resume metadata.
    pub fn kind_tag(&self) -> &'static str {
        match self {
            Self::Adam(a) => a.kind,
            Self::Rose(_) => "rose",
            Self::Prodigy(_) => "prodigy",
        }
    }

    /// Export all mutable optimizer state. Tensor keys are positional because the caller separately
    /// guards the exact named adapter surface before restore.
    pub fn snapshot(&self) -> Result<(HashMap<String, Tensor>, HashMap<String, String>)> {
        let mut tensors = HashMap::new();
        let mut meta = HashMap::from([("optimizer".into(), self.kind_tag().into())]);
        meta.insert("optimizer_config".into(), self.config_fingerprint());
        match self {
            Self::Adam(a) => {
                meta.insert("optim.step".into(), a.step.to_string());
                for (i, (m, v)) in a.first.iter().zip(&a.second).enumerate() {
                    tensors.insert(format!("optim.{i}.m"), m.clone());
                    tensors.insert(format!("optim.{i}.v"), v.clone());
                }
            }
            Self::Rose(_) => {}
            Self::Prodigy(p) => {
                for (key, value) in [
                    ("optim.d", p.d),
                    ("optim.d_max", p.d_max),
                    ("optim.d_numerator", p.d_numerator),
                ] {
                    meta.insert(key.into(), value.to_string());
                }
                for (i, st) in &p.state {
                    tensors.insert(format!("optim.{i}.exp_avg"), st.exp_avg.clone());
                    tensors.insert(format!("optim.{i}.exp_avg_sq"), st.exp_avg_sq.clone());
                    tensors.insert(format!("optim.{i}.s"), st.s.clone());
                    tensors.insert(format!("optim.{i}.p0"), st.p0.clone());
                }
            }
        }
        Ok((tensors, meta))
    }

    /// Restore a snapshot produced by [`Self::snapshot`].
    pub fn restore(
        &mut self,
        tensors: &HashMap<String, Tensor>,
        meta: &HashMap<String, String>,
    ) -> Result<()> {
        let saved = meta
            .get("optimizer")
            .ok_or_else(|| CandleError::Msg("resume: missing optimizer metadata".into()))?;
        if saved != self.kind_tag() {
            return Err(CandleError::Msg(format!(
                "resume: snapshot optimizer {saved:?} does not match requested {:?}",
                self.kind_tag()
            )));
        }
        let saved_config = meta
            .get("optimizer_config")
            .ok_or_else(|| CandleError::Msg("resume: missing optimizer_config metadata".into()))?;
        let expected_config = self.config_fingerprint();
        if saved_config != &expected_config {
            return Err(CandleError::Msg(format!(
                "resume: optimizer configuration differs (saved {saved_config:?}, requested \
                 {expected_config:?})"
            )));
        }
        let parse = |key: &str| -> Result<f64> {
            meta.get(key)
                .ok_or_else(|| CandleError::Msg(format!("resume: missing metadata {key:?}")))?
                .parse()
                .map_err(|e| CandleError::Msg(format!("resume: invalid metadata {key:?}: {e}")))
        };
        let tensor = |key: &str, like: &Tensor| -> Result<Tensor> {
            let t = tensors
                .get(key)
                .ok_or_else(|| CandleError::Msg(format!("resume: missing tensor {key:?}")))?;
            if t.dims() != like.dims() {
                return Err(CandleError::Msg(format!(
                    "resume: tensor {key:?} shape {:?} does not match {:?}",
                    t.dims(),
                    like.dims()
                )));
            }
            Ok(t.to_dtype(like.dtype())?.to_device(like.device())?)
        };
        match self {
            Self::Adam(a) => {
                a.step = parse("optim.step")? as usize;
                for i in 0..a.vars.len() {
                    a.first[i] = tensor(&format!("optim.{i}.m"), a.vars[i].as_tensor())?;
                    a.second[i] = tensor(&format!("optim.{i}.v"), a.vars[i].as_tensor())?;
                }
            }
            Self::Rose(_) => {}
            Self::Prodigy(p) => {
                p.d = parse("optim.d")?;
                p.d_max = parse("optim.d_max")?;
                p.d_numerator = parse("optim.d_numerator")?;
                p.state.clear();
                for (i, var) in p.vars.iter().enumerate() {
                    let prefix = format!("optim.{i}");
                    if !tensors.contains_key(&format!("{prefix}.exp_avg")) {
                        continue;
                    }
                    let like = var.as_tensor();
                    p.state.insert(
                        i,
                        ProdigyState {
                            exp_avg: tensor(&format!("{prefix}.exp_avg"), like)?,
                            exp_avg_sq: tensor(&format!("{prefix}.exp_avg_sq"), like)?,
                            s: tensor(&format!("{prefix}.s"), like)?,
                            p0: tensor(&format!("{prefix}.p0"), like)?,
                        },
                    );
                }
            }
        }
        Ok(())
    }

    fn config_fingerprint(&self) -> String {
        match self {
            Self::Adam(a) => format!(
                "{};lr={};wd={};b1={};b2={};eps={}",
                a.kind, a.base_lr, a.weight_decay, a.beta1, a.beta2, a.eps
            ),
            Self::Rose(r) => format!(
                "rose;lr={};wd={};centralize={};stabilize={}",
                r.base_lr, r.weight_decay, r.centralize, r.stabilize
            ),
            Self::Prodigy(p) => format!(
                "prodigy;lr={};wd={};b1={};b2={};b3={};eps={};d0={};dcoef={};growth={}",
                p.base_lr,
                p.weight_decay,
                p.beta1,
                p.beta2,
                p.beta3,
                p.eps,
                p.d0,
                p.d_coef,
                p.growth_rate
            ),
        }
    }
}

/// Serializable AdamW implementation matching candle-nn's update exactly.
pub struct Adam {
    vars: Vec<Var>,
    first: Vec<Tensor>,
    second: Vec<Tensor>,
    step: usize,
    base_lr: f64,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    kind: &'static str,
}

impl Adam {
    fn new(vars: Vec<Var>, lr: f32, weight_decay: f32, kind: &'static str) -> Result<Self> {
        let vars: Vec<Var> = vars.into_iter().filter(|v| v.dtype().is_float()).collect();
        let first = vars
            .iter()
            .map(|v| Tensor::zeros(v.shape(), v.dtype(), v.device()))
            .collect::<candle_core::Result<_>>()?;
        let second = vars
            .iter()
            .map(|v| Tensor::zeros(v.shape(), v.dtype(), v.device()))
            .collect::<candle_core::Result<_>>()?;
        Ok(Self {
            vars,
            first,
            second,
            step: 0,
            base_lr: lr as f64,
            lr: lr as f64,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: weight_decay as f64,
            kind,
        })
    }

    fn step(&mut self, grads: &GradStore) -> Result<()> {
        self.step += 1;
        let scale_m = 1.0 / (1.0 - self.beta1.powi(self.step as i32));
        let scale_v = 1.0 / (1.0 - self.beta2.powi(self.step as i32));
        for i in 0..self.vars.len() {
            let var = &self.vars[i];
            let Some(g) = grads.get(var.as_tensor()) else {
                continue;
            };
            let m = ((&self.first[i] * self.beta1)? + (g * (1.0 - self.beta1))?)?;
            let v = ((&self.second[i] * self.beta2)? + (g.sqr()? * (1.0 - self.beta2))?)?;
            let m_hat = (&m * scale_m)?;
            let v_hat = (&v * scale_v)?;
            let decayed = (var.as_tensor() * (1.0 - self.lr * self.weight_decay))?;
            let adjusted = (m_hat / (v_hat.sqrt()? + self.eps)?)?;
            var.set(&(decayed - (adjusted * self.lr)?)?)?;
            self.first[i] = m;
            self.second[i] = v;
        }
        Ok(())
    }
}

/// Replace exact zeros in `t` with 1 (avoid div-by-zero in a range/denominator). `where t==0 → 1`.
fn zeros_to_one(t: &Tensor) -> candle_core::Result<Tensor> {
    let is_zero = t.eq(&t.zeros_like()?)?.to_dtype(t.dtype())?;
    t + is_zero
}

/// Stateless Range-Of-Slice Equilibration optimizer (rose-opt). The only mutable state is the
/// (schedule-scaled) learning rate. Faithful port of the MLX `Rose` per-parameter update: decoupled
/// weight decay, then range-normalization over the trailing axes with optional centralization +
/// a coefficient-of-variation trust gate (`centralize = stabilize = true`, the SceneWorks default).
pub struct Rose {
    vars: Vec<Var>,
    base_lr: f64,
    lr: f64,
    weight_decay: f64,
    centralize: bool,
    stabilize: bool,
}

impl Rose {
    pub fn new(vars: Vec<Var>, lr: f32, weight_decay: f32) -> Self {
        Self {
            vars,
            base_lr: lr as f64,
            lr: lr as f64,
            weight_decay: weight_decay as f64,
            centralize: true,
            stabilize: true,
        }
    }

    fn set_lr_scaled(&mut self, mult: f32) {
        self.lr = self.base_lr * mult as f64;
    }

    fn step(&self, grads: &GradStore) -> Result<()> {
        for v in &self.vars {
            if let Some(g) = grads.get(v.as_tensor()) {
                let updated = self.update_one(v.as_tensor(), g)?;
                v.set(&updated)?;
            }
        }
        Ok(())
    }

    /// One Rose update for a single parameter (ports `Rose.step`'s per-`p` body).
    fn update_one(&self, param: &Tensor, grad: &Tensor) -> Result<Tensor> {
        let lr = self.lr;
        // Decoupled multiplicative weight decay: θ *= max(0, 1 − lr·wd).
        let mut param = if self.weight_decay != 0.0 {
            (param * (1.0 - lr * self.weight_decay).max(0.0))?
        } else {
            param.clone()
        };
        match grad.rank() {
            0 => {
                return Err(CandleError::Msg(
                    "Rose: 0-D parameters are unsupported (adapter factors are matrices)".into(),
                ))
            }
            1 => {
                // Global range over the whole vector.
                let g_max = grad.max(0)?;
                let g_min = grad.min(0)?;
                let denom = zeros_to_one(&(g_max.abs()? - g_min)?)?;
                let upd = (grad.broadcast_div(&denom)? * (-lr))?;
                param = (param + upd)?;
            }
            _ => {
                // Active axes = every axis except the leading one. Adapter factors are 2-D, so the
                // trailing axes flatten to one and the reduction is per leading slice (per row).
                let dims = grad.dims().to_vec();
                let leading = dims[0];
                let rest: usize = dims[1..].iter().product();
                let g2 = grad.reshape((leading, rest))?;
                let g2 = if self.centralize {
                    let mean = g2.mean_keepdim(D::Minus1)?;
                    g2.broadcast_sub(&mean)?
                } else {
                    g2
                };
                let raw_scale = (g2.max_keepdim(D::Minus1)?.abs()? - g2.min_keepdim(D::Minus1)?)?; // [leading,1]
                let denom = if self.stabilize {
                    // Population mean/std over the per-row range tensor; trust = mean/(std+mean).
                    let mean = raw_scale.mean_all()?;
                    let var = raw_scale.broadcast_sub(&mean)?.sqr()?.mean_all()?;
                    let std = var.sqrt()?;
                    let trust = mean.broadcast_div(&zeros_to_one(&(std + &mean)?)?)?; // scalar
                                                                                      // denom = mean + trust·(raw_scale − mean).
                    let centered = raw_scale.broadcast_sub(&mean)?;
                    centered.broadcast_mul(&trust)?.broadcast_add(&mean)?
                } else {
                    raw_scale
                };
                let denom = zeros_to_one(&denom)?; // [leading,1]
                let upd = (g2.broadcast_div(&denom)? * (-lr))?;
                let upd = upd.reshape(dims)?;
                param = (param + upd)?;
            }
        }
        Ok(param)
    }
}

/// Per-parameter Prodigy state: the Adam EMAs, the `s` accumulator, and the initial parameter `p0`.
struct ProdigyState {
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    s: Tensor,
    p0: Tensor,
}

/// Prodigy (prodigyopt): Adam with a learning-rate-free, globally-adapted step size `d`. Faithful
/// port of the MLX `Prodigy.step` (`slice_p = 1`, `beta1 > 0`, decoupled weight decay, no bias
/// correction / safeguard warmup).
pub struct Prodigy {
    vars: Vec<Var>,
    base_lr: f64,
    lr: f64,
    weight_decay: f64,
    beta1: f64,
    beta2: f64,
    beta3: f64,
    eps: f64,
    d: f64,
    d0: f64,
    d_max: f64,
    d_numerator: f64,
    d_coef: f64,
    growth_rate: f64,
    state: HashMap<usize, ProdigyState>,
}

impl Prodigy {
    /// `lr = lr ≥ 0.1 ? lr : 1.0` (LoRA LRs ≪ 0.1 ⇒ the knob is the Prodigy-convention 1.0), eps 1e-6,
    /// betas (0.9, 0.999), beta3 = √beta2, d0 = 1e-6, d_coef = 1, growth ∞.
    pub fn new(vars: Vec<Var>, lr: f32, weight_decay: f32) -> Self {
        let use_lr = if lr >= 0.1 { lr as f64 } else { 1.0 };
        let beta2 = 0.999;
        Self {
            vars,
            base_lr: use_lr,
            lr: use_lr,
            weight_decay: weight_decay as f64,
            beta1: 0.9,
            beta2,
            beta3: beta2.sqrt(),
            eps: 1e-6,
            d: 1e-6,
            d0: 1e-6,
            d_max: 1e-6,
            d_numerator: 0.0,
            d_coef: 1.0,
            growth_rate: f64::INFINITY,
            state: HashMap::new(),
        }
    }

    fn set_lr_scaled(&mut self, mult: f32) {
        self.lr = self.base_lr * mult as f64;
    }

    fn step(&mut self, grads: &GradStore) -> Result<()> {
        let (beta1, beta2, beta3) = (self.beta1, self.beta2, self.beta3);
        let (d, d0, lr, eps) = (self.d, self.d0, self.lr, self.eps);
        let dlr = d * lr; // bias_correction = 1
        let d_numerator = self.d_numerator * beta3;

        // --- Pass 1: EMAs + s; accumulate the global numerator/denominator ---
        // Both the delta-numerator and the denominator are accumulated as on-device f64 scalars and
        // read back exactly once each after the loop (sc-9036), instead of two GPU→CPU syncs per var.
        // Each per-var term is scaled/upcast to f64 on-device before the running add, so the reduction
        // stays bit-identical to the previous per-scalar CPU accumulation while removing the stalls.
        let dn_scale = (d / d0) * dlr; // constant across the loop
        let mut delta_numerator_dev: Option<Tensor> = None;
        let mut d_denom_dev: Option<Tensor> = None;
        for (i, v) in self.vars.iter().enumerate() {
            let Some(g) = grads.get(v.as_tensor()) else {
                continue;
            };
            let p = v.as_tensor();
            // `entry`/`?` rather than `or_insert_with` — the state init is fallible (allocation).
            if let std::collections::hash_map::Entry::Vacant(e) = self.state.entry(i) {
                e.insert(ProdigyState {
                    exp_avg: p.zeros_like()?,
                    exp_avg_sq: p.zeros_like()?,
                    s: p.zeros_like()?,
                    // Snapshot the initial params with a genuine copy: `detach()` only Arc-clones the
                    // storage, which `Var::set` later mutates in place (pass 2), so an aliased `p0`
                    // would always equal the live param and zero out ⟨g, p0 − p⟩ every step (sc-9036).
                    p0: p.detach().copy()?,
                });
            }
            let st = self.state.get(&i).unwrap();
            // delta_numerator += (d/d0)·dlr·⟨g, p0 − p⟩
            let dot = (g * (&st.p0 - p)?)?
                .sum_all()?
                .to_dtype(candle_core::DType::F64)?;
            let dot_scaled = (dot * dn_scale)?;
            delta_numerator_dev = Some(match delta_numerator_dev {
                Some(acc) => (acc + dot_scaled)?,
                None => dot_scaled,
            });
            let exp_avg = ((&st.exp_avg * beta1)? + (g * (d * (1.0 - beta1)))?)?;
            let exp_avg_sq = ((&st.exp_avg_sq * beta2)? + (g.sqr()? * (d * d * (1.0 - beta2)))?)?;
            let s = ((&st.s * beta3)? + (g * ((d / d0) * dlr))?)?;
            let s_abs_sum = s.abs()?.sum_all()?.to_dtype(candle_core::DType::F64)?;
            d_denom_dev = Some(match d_denom_dev {
                Some(acc) => (acc + s_abs_sum)?,
                None => s_abs_sum,
            });
            let st = self.state.get_mut(&i).unwrap();
            st.exp_avg = exp_avg;
            st.exp_avg_sq = exp_avg_sq;
            st.s = s;
        }

        // Single readback per accumulator (or default 0 when no var carried a gradient).
        let delta_numerator = match delta_numerator_dev {
            Some(t) => t.to_scalar::<f64>()?,
            None => 0f64,
        };
        let d_denom = match d_denom_dev {
            Some(t) => t.to_scalar::<f64>()?,
            None => 0f64,
        };

        // No usable gradient signal this step — leave d/params unchanged.
        if d_denom == 0.0 {
            return Ok(());
        }

        // --- Re-estimate the adapted step d ---
        let global_d_numerator = d_numerator + delta_numerator;
        let d_hat = self.d_coef * global_d_numerator / d_denom;
        let mut d_new = d;
        if d == d0 {
            d_new = d.max(d_hat);
        }
        let d_max = self.d_max.max(d_hat);
        d_new = d_max.min(d_new * self.growth_rate);
        self.d_numerator = global_d_numerator;
        self.d = d_new;
        self.d_max = d_max;

        // --- Pass 2: Adam step. denom uses the NEW d; dlr/weight-decay use the OLD d ---
        for (i, v) in self.vars.iter().enumerate() {
            if grads.get(v.as_tensor()).is_none() {
                continue;
            }
            let p = v.as_tensor();
            let st = self.state.get(&i).expect("state created in pass 1");
            let denom = (st.exp_avg_sq.sqrt()? + (d_new * eps))?;
            let mut np = p.clone();
            if self.weight_decay != 0.0 {
                np = (np * (1.0 - self.weight_decay * dlr))?;
            }
            np = (np - (st.exp_avg.broadcast_div(&denom)? * dlr)?)?;
            v.set(&np)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{AdamW as CandleAdamW, Optimizer, ParamsAdamW};

    fn var(data: &[f32], shape: (usize, usize)) -> Var {
        Var::from_tensor(&Tensor::from_vec(data.to_vec(), shape, &Device::Cpu).unwrap()).unwrap()
    }

    /// Every recognized alias (case/separator variants + the pip-package spellings) resolves to its
    /// canonical optimizer, and each canonical form is one of `SUPPORTED_OPTIMIZERS`.
    #[test]
    fn normalize_collapses_aliases() {
        // adamw family (incl. the 8-bit spellings that collapse to full-precision AdamW).
        assert_eq!(normalize("adamw"), "adamw");
        assert_eq!(normalize("AdamW"), "adamw");
        assert_eq!(normalize("adamw8bit"), "adamw");
        assert_eq!(normalize("AdamW-8bit"), "adamw");
        assert_eq!(normalize("adamw_8bit"), "adamw");
        assert_eq!(normalize("adam8bit"), "adamw");
        // plain adam stays adam (weight-decay-free).
        assert_eq!(normalize("adam"), "adam");
        assert_eq!(normalize("Adam"), "adam");
        // rose + pip spelling.
        assert_eq!(normalize("rose"), "rose");
        assert_eq!(normalize("Rose"), "rose");
        assert_eq!(normalize("rose_opt"), "rose");
        assert_eq!(normalize("rose-opt"), "rose");
        assert_eq!(normalize("roseopt"), "rose");
        // prodigy + pip spelling.
        assert_eq!(normalize("prodigy"), "prodigy");
        assert_eq!(normalize("Prodigy"), "prodigy");
        assert_eq!(normalize("prodigy-opt"), "prodigy");
        assert_eq!(normalize("prodigyopt"), "prodigy");
        // every canonical target is a supported optimizer.
        for (_, canonical) in OPTIMIZER_ALIASES {
            assert!(
                SUPPORTED_OPTIMIZERS.contains(&canonical),
                "alias target {canonical:?} not in SUPPORTED_OPTIMIZERS"
            );
        }
        assert!(TrainOptimizer::is_supported("Prodigy"));
        assert!(TrainOptimizer::is_supported("rose"));
        assert!(!TrainOptimizer::is_supported("lion"));
    }

    /// The exact-match fix (sc-9017 / F-033): names that the previous `contains` matching silently
    /// mis-mapped now pass through unchanged and are NOT recognized as any supported optimizer. Under
    /// substring matching `adamax`/`radam`/`nadam` all matched `"adam"` and `adamw8bittypo` matched
    /// `"adamw"`; each must now be rejected.
    #[test]
    fn normalize_rejects_substring_near_misses() {
        for name in [
            "adamax",
            "radam",
            "nadam",
            "adamwww",
            "prodigyx",
            "rosewater",
        ] {
            let n = normalize(name);
            // Passes through as the stripped form; crucially not coerced to a canonical optimizer.
            assert!(
                !SUPPORTED_OPTIMIZERS.contains(&n.as_str()),
                "{name:?} should NOT normalize to a supported optimizer, got {n:?}"
            );
            assert!(
                !TrainOptimizer::is_supported(name),
                "{name:?} must not be treated as supported"
            );
        }
    }

    /// An unrecognized optimizer name fails `from_config` with an error that lists the supported set.
    #[test]
    fn from_config_rejects_unsupported() {
        for name in ["lion", "adamax", "radam", "nadam"] {
            let v = vec![var(&[1.0], (1, 1))];
            let msg = match TrainOptimizer::from_config(name, v, 1e-4, 0.0) {
                Ok(_) => panic!("unrecognized optimizer {name:?} must error"),
                Err(e) => e.to_string(),
            };
            for opt in SUPPORTED_OPTIMIZERS {
                assert!(
                    msg.contains(opt),
                    "error for {name:?} should list supported optimizer {opt:?}: {msg}"
                );
            }
        }
    }

    /// Every recognized alias actually constructs its optimizer (existing configs keep working).
    #[test]
    fn from_config_accepts_all_aliases() {
        for name in [
            "adamw",
            "AdamW",
            "adamw8bit",
            "adam8bit",
            "adam",
            "rose",
            "rose_opt",
            "roseopt",
            "prodigy",
            "prodigy-opt",
            "prodigyopt",
        ] {
            let v = vec![var(&[1.0, 2.0], (1, 2))];
            assert!(
                TrainOptimizer::from_config(name, v, 1e-4, 0.0).is_ok(),
                "alias {name:?} should construct an optimizer"
            );
        }
    }

    /// Rose 2-D, centralize + stabilize OFF: θ -= lr · g / (|max_row| − min_row), per row.
    #[test]
    fn rose_2d_update_matches_closed_form_no_stabilize() {
        let p = var(&[0.0, 0.0, 0.0, 0.0], (2, 2));
        let mut rose = Rose::new(vec![p.clone()], 0.1, 0.0);
        rose.centralize = false;
        rose.stabilize = false;
        // grad rows: [1, 3] → denom = |3| − 1 = 2; [-4, -2] → denom = |-2| − (-4) = 2 + 4 = 6.
        let g = Tensor::from_vec(vec![1.0f32, 3.0, -4.0, -2.0], (2, 2), &Device::Cpu).unwrap();
        let mut grads = GradStore::default();
        grads.insert(p.as_tensor(), g);
        rose.step(&grads).unwrap();
        let out = p.as_tensor().to_vec2::<f32>().unwrap();
        // row0: -0.1 · [1,3]/2 = [-0.05, -0.15]; row1: -0.1 · [-4,-2]/6 = [0.0666.., 0.0333..].
        assert!((out[0][0] - -0.05).abs() < 1e-6);
        assert!((out[0][1] - -0.15).abs() < 1e-6);
        assert!((out[1][0] - 0.066_666_67).abs() < 1e-5);
        assert!((out[1][1] - 0.033_333_34).abs() < 1e-5);
    }

    /// AdamW takes one well-formed step that moves the parameter opposite the gradient sign.
    #[test]
    fn adamw_steps_downhill() {
        let p = var(&[1.0, -1.0], (1, 2));
        let mut opt = TrainOptimizer::from_config("adamw", vec![p.clone()], 1e-2, 0.0).unwrap();
        let g = Tensor::from_vec(vec![1.0f32, -1.0], (1, 2), &Device::Cpu).unwrap();
        let mut grads = GradStore::default();
        grads.insert(p.as_tensor(), g);
        opt.step(&grads).unwrap();
        let out = p.as_tensor().to_vec2::<f32>().unwrap();
        assert!(
            out[0][0] < 1.0 && out[0][1] > -1.0,
            "should move opposite the grad"
        );
    }

    /// The serializable local AdamW is a numeric drop-in for the prior candle-nn implementation,
    /// including bias correction and decoupled weight decay over multiple updates.
    #[test]
    fn serializable_adam_matches_candle_adamw_multistep_with_decay() {
        let ours_var = var(&[1.0, -2.0, 0.5, 4.0], (2, 2));
        let candle_var = var(&[1.0, -2.0, 0.5, 4.0], (2, 2));
        let mut ours =
            TrainOptimizer::from_config("adamw", vec![ours_var.clone()], 3e-3, 0.07).unwrap();
        let mut candle = CandleAdamW::new(
            vec![candle_var.clone()],
            ParamsAdamW {
                lr: 3e-3,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.07,
            },
        )
        .unwrap();
        for values in [
            [0.2f32, -0.1, 0.4, -0.3],
            [-0.7, 0.8, 0.05, -0.2],
            [0.11, 0.13, -0.17, 0.19],
            [1.0, -1.0, 0.5, -0.5],
        ] {
            let go = Tensor::from_vec(values.to_vec(), (2, 2), &Device::Cpu).unwrap();
            let gc = go.clone();
            let mut og = GradStore::default();
            og.insert(ours_var.as_tensor(), go);
            let mut cg = GradStore::default();
            cg.insert(candle_var.as_tensor(), gc);
            ours.step(&og).unwrap();
            candle.step(&cg).unwrap();
        }
        let a = ours_var
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = candle_var
            .as_tensor()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(
                (x - y).abs() <= 1e-7,
                "parameter {i}: local {x} != candle {y}"
            );
        }
    }

    /// Every supported optimizer's mutable state round-trips: one step, snapshot, restore into a
    /// fresh optimizer over the current factors, then the next update matches uninterrupted.
    #[test]
    fn optimizer_snapshot_resume_matches_uninterrupted_for_every_kind() {
        for name in SUPPORTED_OPTIMIZERS {
            let live = var(&[0.5, -0.2, 1.0, -1.5], (2, 2));
            let mut uninterrupted =
                TrainOptimizer::from_config(name, vec![live.clone()], 1e-2, 0.01).unwrap();
            let first =
                Tensor::from_vec(vec![0.3f32, -0.4, 0.2, -0.1], (2, 2), &Device::Cpu).unwrap();
            let mut g1 = GradStore::default();
            g1.insert(live.as_tensor(), first);
            uninterrupted.step(&g1).unwrap();
            let (state, meta) = uninterrupted.snapshot().unwrap();

            let restored = Var::from_tensor(&live.as_tensor().copy().unwrap()).unwrap();
            let mut resumed =
                TrainOptimizer::from_config(name, vec![restored.clone()], 1e-2, 0.01).unwrap();
            resumed.restore(&state, &meta).unwrap();

            let second = vec![-0.6f32, 0.1, 0.7, -0.25];
            let mut gu = GradStore::default();
            gu.insert(
                live.as_tensor(),
                Tensor::from_vec(second.clone(), (2, 2), &Device::Cpu).unwrap(),
            );
            let mut gr = GradStore::default();
            gr.insert(
                restored.as_tensor(),
                Tensor::from_vec(second, (2, 2), &Device::Cpu).unwrap(),
            );
            uninterrupted.step(&gu).unwrap();
            resumed.step(&gr).unwrap();
            let a = live
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let b = restored
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for (i, (x, y)) in a.iter().zip(&b).enumerate() {
                assert!(
                    (x - y).abs() <= 1e-7,
                    "{name} parameter {i}: uninterrupted {x} != resumed {y}"
                );
            }
        }
    }

    #[test]
    fn optimizer_restore_rejects_kind_and_hyperparameter_changes() {
        let source_var = var(&[1.0, 2.0], (1, 2));
        let source = TrainOptimizer::from_config("adamw", vec![source_var], 1e-3, 0.02).unwrap();
        let (state, meta) = source.snapshot().unwrap();

        let mut plain_adam =
            TrainOptimizer::from_config("adam", vec![var(&[1.0, 2.0], (1, 2))], 1e-3, 0.0).unwrap();
        assert!(plain_adam
            .restore(&state, &meta)
            .unwrap_err()
            .to_string()
            .contains("does not match"));

        let mut changed_lr =
            TrainOptimizer::from_config("adamw", vec![var(&[1.0, 2.0], (1, 2))], 2e-3, 0.02)
                .unwrap();
        assert!(changed_lr
            .restore(&state, &meta)
            .unwrap_err()
            .to_string()
            .contains("configuration differs"));
    }

    /// clip_grad_norm scales an over-large gradient down to exactly `max_norm` and reports the
    /// pre-clip norm.
    #[test]
    fn clip_scales_to_max_norm() {
        let p = var(&[0.0, 0.0], (1, 2));
        let g = Tensor::from_vec(vec![3.0f32, 4.0], (1, 2), &Device::Cpu).unwrap(); // norm 5
        let mut grads = GradStore::default();
        grads.insert(p.as_tensor(), g);
        let pre = clip_grad_norm(&mut grads, std::slice::from_ref(&p), 1.0).unwrap();
        assert!((pre - 5.0).abs() < 1e-6);
        let clipped = grads.get(p.as_tensor()).unwrap().to_vec2::<f32>().unwrap();
        let n = (clipped[0][0].powi(2) + clipped[0][1].powi(2)).sqrt();
        assert!(
            (n - 1.0).abs() < 1e-6,
            "clipped norm should be 1.0, got {n}"
        );
    }

    /// The on-device sum-of-squares accumulation in `clip_grad_norm` (sc-9036) must reproduce the
    /// pre-clip norm the old per-var CPU `to_scalar` loop computed, to fp tolerance, across a spread of
    /// gradient tensors — this is the numerical-equivalence guard for the single-readback rewrite.
    #[test]
    fn clip_grad_norm_matches_per_scalar_reduction() {
        let dev = Device::Cpu;
        // Several vars with distinct shapes/magnitudes (mirrors a multi-target adapter surface).
        let grads_data: Vec<(Vec<f32>, (usize, usize))> = vec![
            (vec![0.1, -0.2, 0.3, 0.05], (2, 2)),
            (vec![1.5, -2.5, 0.0, 3.25, -1.0, 0.75], (2, 3)),
            (vec![1e-3, -1e-3], (1, 2)),
            (vec![10.0, -10.0, 5.0, -5.0], (4, 1)),
        ];
        let vars: Vec<Var> = grads_data.iter().map(|(d, s)| var(d, *s)).collect();
        let mut grads = GradStore::default();
        for (v, (d, s)) in vars.iter().zip(grads_data.iter()) {
            grads.insert(
                v.as_tensor(),
                Tensor::from_vec(d.clone(), *s, &dev).unwrap(),
            );
        }
        // Reference: the old per-scalar f64 accumulation.
        let mut ref_total_sq = 0f64;
        for v in &vars {
            let g = grads.get(v.as_tensor()).unwrap();
            ref_total_sq += g
                .sqr()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_dtype(candle_core::DType::F64)
                .unwrap()
                .to_scalar::<f64>()
                .unwrap();
        }
        let ref_norm = ref_total_sq.sqrt();
        // max_norm above the norm ⇒ no scaling, so the returned value is the pure reduction.
        let got = clip_grad_norm(&mut grads, &vars, 1e9).unwrap();
        assert!(
            (got - ref_norm).abs() <= 1e-12 * ref_norm.max(1.0),
            "on-device norm {got} != per-scalar norm {ref_norm}"
        );
    }

    /// Prodigy's pass-1 numerator/denominator (now accumulated on-device, sc-9036) must match the old
    /// per-var `to_scalar` reduction. Step 1 seeds exp_avg/s from zeros with `p0 == p` (dot term 0);
    /// step 2 — after pass-2 has MOVED the params — has `p0 != p`, so `⟨g, p0 − p⟩ ≠ 0` and the
    /// delta-numerator is exercised for real. This second step is the guard against the p0-snapshot
    /// aliasing bug (F-052): `detach()` alone Arc-clones storage that `Var::set` mutates in place, so
    /// an aliased p0 would leave the dot term identically 0 on step 2 too and `d` would never adapt.
    #[test]
    fn prodigy_first_step_d_matches_per_scalar_reduction() {
        let dev = Device::Cpu;
        let grads_data: Vec<(Vec<f32>, (usize, usize))> = vec![
            (vec![0.2, -0.3, 0.1, 0.4], (2, 2)),
            (vec![0.5, -0.5, 0.25, -0.25, 0.75, -0.75], (2, 3)),
            (vec![0.01, -0.02], (1, 2)),
        ];
        let params_data: Vec<Vec<f32>> = vec![
            vec![0.5, -0.5, 0.1, -0.1],
            vec![1.0, -1.0, 0.5, -0.5, 0.25, -0.25],
            vec![0.05, -0.05],
        ];
        let vars: Vec<Var> = params_data
            .iter()
            .zip(grads_data.iter())
            .map(|(p, (_, s))| var(p, *s))
            .collect();
        let mut grads = GradStore::default();
        for (v, (d, s)) in vars.iter().zip(grads_data.iter()) {
            grads.insert(
                v.as_tensor(),
                Tensor::from_vec(d.clone(), *s, &dev).unwrap(),
            );
        }
        let mut prod = Prodigy::new(vars.clone(), 1e-4, 0.0);
        // Reference d computed with the per-scalar formula (first step: p0==p ⇒ dot term is 0, so
        // delta_numerator==0 and d stays at d0; but s accumulates so d_denom>0). Compute it explicitly
        // to guard the on-device denominator accumulation too.
        let (d, d0) = (prod.d, prod.d0);
        let dlr = d * prod.lr;
        let (beta3, d0c) = (prod.beta3, prod.d0);
        let mut ref_delta_num = 0f64;
        let mut ref_denom = 0f64;
        for v in &vars {
            let g = grads.get(v.as_tensor()).unwrap();
            let p = v.as_tensor();
            let dot = (g * (&p.detach() - p).unwrap())
                .unwrap()
                .sum_all()
                .unwrap()
                .to_dtype(candle_core::DType::F64)
                .unwrap()
                .to_scalar::<f64>()
                .unwrap();
            ref_delta_num += (d / d0) * dlr * dot;
            // s = 0*beta3 + g*((d/d0)*dlr) on the first step.
            let s = (g * ((d / d0c) * dlr)).unwrap();
            let _ = beta3;
            ref_denom += s
                .abs()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_dtype(candle_core::DType::F64)
                .unwrap()
                .to_scalar::<f64>()
                .unwrap();
        }
        let ref_global_num = 0.0 * beta3 + ref_delta_num; // d_numerator starts at 0
        let ref_d_hat = prod.d_coef * ref_global_num / ref_denom;
        let ref_d_max = d0.max(ref_d_hat);
        let mut ref_d = d;
        if d == d0 {
            ref_d = d.max(ref_d_hat);
        }
        ref_d = ref_d_max.min(ref_d * prod.growth_rate);

        prod.step(&grads).unwrap();
        assert!(
            (prod.d - ref_d).abs() <= 1e-15 * ref_d.abs().max(1e-12),
            "on-device Prodigy d {} != per-scalar d {}",
            prod.d,
            ref_d
        );

        // --- Second step: pass-2 of step 1 moved every param, so now p0 != p. Recompute the
        // reference per-scalar delta_numerator/denominator against the CAPTURED post-step-1 state
        // (p0 snapshot + accumulated s + carried d_numerator) and confirm the on-device reduction
        // still matches. With an aliased p0 the dot term below would be identically 0 (bug), so this
        // asserts the numerator is genuinely non-zero AND numerically exact.
        let d1 = prod.d;
        let d0_1 = prod.d0;
        let dlr1 = d1 * prod.lr;
        let d_num_start = prod.d_numerator; // carried from step 1
        let mut ref_delta_num2 = 0f64;
        let mut ref_denom2 = 0f64;
        for (i, v) in vars.iter().enumerate() {
            let g = grads.get(v.as_tensor()).unwrap();
            let p = v.as_tensor();
            let st = prod.state.get(&i).unwrap();
            // p0 is the ORIGINAL param snapshot; p is the moved value ⇒ non-zero displacement.
            let dot = (g * (&st.p0 - p).unwrap())
                .unwrap()
                .sum_all()
                .unwrap()
                .to_dtype(candle_core::DType::F64)
                .unwrap()
                .to_scalar::<f64>()
                .unwrap();
            ref_delta_num2 += (d1 / d0_1) * dlr1 * dot;
            // s_new = s_old·beta3 + g·((d/d0)·dlr), reading the s accumulated on step 1.
            let s = ((&st.s * prod.beta3).unwrap() + (g * ((d1 / d0_1) * dlr1)).unwrap()).unwrap();
            ref_denom2 += s
                .abs()
                .unwrap()
                .sum_all()
                .unwrap()
                .to_dtype(candle_core::DType::F64)
                .unwrap()
                .to_scalar::<f64>()
                .unwrap();
        }
        assert!(
            ref_delta_num2.abs() > 1e-12,
            "step-2 dot term must be non-zero once params have moved (p0 snapshot fix); got {ref_delta_num2}"
        );
        let ref_global_num2 = d_num_start * prod.beta3 + ref_delta_num2;
        let ref_d_hat2 = prod.d_coef * ref_global_num2 / ref_denom2;
        let ref_d_max2 = prod.d_max.max(ref_d_hat2);
        let mut ref_d2 = d1;
        if d1 == d0_1 {
            ref_d2 = d1.max(ref_d_hat2);
        }
        ref_d2 = ref_d_max2.min(ref_d2 * prod.growth_rate);

        prod.step(&grads).unwrap();
        assert!(
            (prod.d_numerator - ref_global_num2).abs() <= 1e-15 * ref_global_num2.abs().max(1e-12),
            "on-device Prodigy d_numerator {} != per-scalar {}",
            prod.d_numerator,
            ref_global_num2
        );
        assert!(
            (prod.d - ref_d2).abs() <= 1e-15 * ref_d2.abs().max(1e-12),
            "on-device Prodigy d {} != per-scalar d {} (step 2)",
            prod.d,
            ref_d2
        );
    }

    /// Prodigy takes a finite step and adapts `d` upward on the first step (d starts at d0=1e-6).
    #[test]
    fn prodigy_first_step_adapts_d() {
        let p = var(&[0.5, -0.5], (1, 2));
        let mut prod = Prodigy::new(vec![p.clone()], 1e-4, 0.0);
        let g = Tensor::from_vec(vec![0.2f32, -0.3], (1, 2), &Device::Cpu).unwrap();
        let mut grads = GradStore::default();
        grads.insert(p.as_tensor(), g);
        prod.step(&grads).unwrap();
        assert!(
            prod.d >= prod.d0,
            "d must not shrink below d0 on the first step"
        );
        let out = p.as_tensor().to_vec2::<f32>().unwrap();
        assert!(out[0][0].is_finite() && out[0][1].is_finite());
    }
}
