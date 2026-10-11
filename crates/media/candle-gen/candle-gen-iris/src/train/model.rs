//! The trainable Iris backbone on Candle — the twin of `mlx_gen_iris::train::model`: which tensors
//! are trained (every backbone tensor, or LoRA/LoKr factors over frozen Linears), how a step's
//! backbone is built from them, the rectified-flow loss and its gradient, random init, and the
//! artifact writers.
//!
//! The forward is **the inference forward**: every step builds an [`IrisDiT`] through
//! [`IrisDiT::from_checkpoint`] — from the f32 `Var` masters of a full run (an in-memory
//! [`Checkpoint`], cast to the compute dtype exactly as the provider casts a loaded checkpoint, the
//! casts on the autograd tape), or from the frozen base with each target's adapter installed as the
//! provider installs an exported adapter file: a forward-time [`Residual`] built by the same
//! [`lora_residual`] / [`lokr_residual`] the provider's loader calls. Training and inference
//! therefore cannot disagree about the graph, and a preview renders what the exported artifact
//! renders through `load_backbone_with_adapters`.
//!
//! Under autograd the backbone's fused RMSNorm / softmax kernels (no backward in candle) are swapped
//! for the same math in composable ops ([`crate::nn`]); inference never takes that branch.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor, TensorId, Var, D};
use candle_gen::gen_core::iris::train::ac::AcPolicy;
use candle_gen::gen_core::iris::train::repa::{
    bilinear_matrix, RepaPlan, RepaVariant, PROJECTOR_PREFIX,
};
use candle_gen::gen_core::iris::train::{
    backbone_tensor_shapes, init_kind, mix_seed, FlowObjective, HostRng, InitKind, Prediction,
};
use candle_gen::gen_core::iris::ModelConfig;
use candle_gen::train::lora::factorization;
use candle_gen::{CandleError as Error, Result};

use super::optim::{mul_s, Params, Tensors};
use crate::adapters::{lokr_residual, lora_residual};
use crate::dit::{ForwardHooks, IrisDiT, NoHooks, PixelCtx, TextBatch, TrunkCtx};
use crate::nn::{with_composable, Checkpoint, Residual};

/// The adapter parameterization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AdapterKind {
    Lora {
        rank: usize,
        alpha: f32,
    },
    Lokr {
        rank: usize,
        alpha: f32,
        factor: i32,
    },
}

impl AdapterKind {
    pub fn rank(self) -> usize {
        match self {
            AdapterKind::Lora { rank, .. } | AdapterKind::Lokr { rank, .. } => rank,
        }
    }
    pub fn alpha(self) -> f32 {
        match self {
            AdapterKind::Lora { alpha, .. } | AdapterKind::Lokr { alpha, .. } => alpha,
        }
    }
}

/// One adapted Linear.
#[derive(Clone, Debug)]
pub struct AdapterTarget {
    /// Backbone module path (`blocks.3.attn.q_proj_x`).
    pub path: String,
    pub out_f: usize,
    pub in_f: usize,
}

/// LyCORIS' low-rank-`w2` rule: `w2` is factored (`w2_a · w2_b`) when the rank is below half the
/// larger second-leg dimension.
fn lokr_w2_factored(rank: usize, out_b: usize, in_b: usize) -> bool {
    (rank as f32) < (out_b.max(in_b) as f32) / 2.0
}

impl AdapterTarget {
    /// The factor keys this target trains, in a fixed order (the MLX twin's layout).
    pub fn factor_keys(&self, kind: AdapterKind) -> Vec<String> {
        let p = &self.path;
        match kind {
            AdapterKind::Lora { .. } => {
                vec![format!("{p}.lora_A.weight"), format!("{p}.lora_B.weight")]
            }
            AdapterKind::Lokr { rank, factor, .. } => {
                let (_, out_b) = factorization(self.out_f, factor);
                let (_, in_b) = factorization(self.in_f, factor);
                if lokr_w2_factored(rank, out_b, in_b) {
                    vec![
                        format!("{p}.lokr_w1"),
                        format!("{p}.lokr_w2_a"),
                        format!("{p}.lokr_w2_b"),
                    ]
                } else {
                    vec![format!("{p}.lokr_w1"), format!("{p}.lokr_w2")]
                }
            }
        }
    }
}

/// What a run trains.
pub enum Trainable {
    /// Every backbone tensor (the trainable map holds them all, f32).
    Full,
    /// Factors over frozen Linears; `base` holds every backbone tensor at the provider's load
    /// dtypes (matrices in the compute dtype, vectors and the text position table f32).
    Adapter {
        kind: AdapterKind,
        targets: Vec<AdapterTarget>,
        base: HashMap<String, Tensor>,
    },
}

/// The trainable backbone.
pub struct TrainModel {
    pub cfg: ModelConfig,
    pub compute: DType,
    pub device: Device,
    pub trainable: Trainable,
}

/// Cast a backbone tensor the way the provider's loader does ([`crate::nn::Loader`]): matrices
/// (except the text position table) to the compute dtype, vectors kept f32.
pub fn provider_dtype(key: &str, a: &Tensor, compute: DType) -> Result<Tensor> {
    let target = if a.rank() >= 2 && key != "y_pos_embedding" {
        compute
    } else {
        DType::F32
    };
    Ok(if a.dtype() == target {
        a.clone()
    } else {
        a.to_dtype(target)?
    })
}

/// One target's residual exactly as the provider's loader installs the exported file: a LoRA via
/// [`lora_residual`] (`Aᵀ`, `Bᵀ · alpha/rank`, f32), a LoKr via [`lokr_residual`] (the structured
/// Kronecker factors, `w2` scaled by `alpha/rank`, f32 — never an `[out, in]` delta). Every op is on
/// the autograd tape, so gradients reach the f32 factors.
pub fn provider_residual(kind: AdapterKind, t: &AdapterTarget, p: &Tensors) -> Result<Residual> {
    let get = |name: &str| p.get(&format!("{}.{name}", t.path));
    let need = |name: &str| -> Result<&Tensor> {
        get(name)
            .ok_or_else(|| Error::Msg(format!("iris adapter: factor {}.{name} missing", t.path)))
    };
    match kind {
        AdapterKind::Lora { rank, alpha } => lora_residual(
            need("lora_A.weight")?,
            need("lora_B.weight")?,
            alpha,
            rank as f32,
            1.0,
        ),
        AdapterKind::Lokr { rank, alpha, .. } => lokr_residual(
            get("lokr_w1"),
            None,
            None,
            get("lokr_w2"),
            get("lokr_w2_a"),
            get("lokr_w2_b"),
            alpha,
            rank as f32,
            1.0,
            (t.out_f, t.in_f),
        ),
    }
}

impl TrainModel {
    /// The backbone of one step at the compute dtype, from `trainable` (the run's tensors — `Var`
    /// tensors under autograd, or plain snapshots / the EMA for a preview).
    ///
    /// A full run builds it from the trainable tensors as the provider builds a loaded checkpoint.
    /// An adapter run builds the frozen base and installs each target's adapter as a forward-time
    /// residual (`y = base(x) + r(x)`) — never merged into the (bf16) base weight, so a delta below
    /// the base's bf16 ulp still reaches the forward, the gradient and the previews.
    pub fn dit(&self, trainable: &Tensors) -> Result<IrisDiT> {
        match &self.trainable {
            Trainable::Full => {
                let map: HashMap<String, Tensor> = trainable
                    .iter()
                    .filter(|(k, _)| !k.starts_with(PROJECTOR_PREFIX))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                let ckpt = Checkpoint::from_tensors(map, &self.device);
                IrisDiT::from_checkpoint(&ckpt, &self.cfg, self.compute, &self.device)
            }
            Trainable::Adapter {
                kind,
                targets,
                base,
            } => {
                let mut residuals = BTreeMap::new();
                for t in targets {
                    residuals.insert(
                        t.path.clone(),
                        vec![provider_residual(*kind, t, trainable)?],
                    );
                }
                let ckpt = Checkpoint::from_tensors(base.clone(), &self.device)
                    .with_residuals(residuals)?;
                IrisDiT::from_checkpoint(&ckpt, &self.cfg, self.compute, &self.device)
            }
        }
    }
}

/// One micro-batch, assembled host-side from the data walk and the positional draws.
pub struct StepBatch {
    /// `[B, C, H, W]` f32 in `[−1, 1]`.
    pub x0: Tensor,
    /// `[B, C, H, W]` f32.
    pub noise: Tensor,
    /// `[B, 1, 1, 1]` f32 — `sigmas[idx]`.
    pub sigma: Tensor,
    /// `[B]` f32 — the integer-truncated model time.
    pub t: Tensor,
    /// `[B, T, L, Dt]` f32 text states (dropped rows already replaced by the null).
    pub states: Tensor,
    pub masks: Vec<Vec<i32>>,
}

/// `RectifiedFlow.training_loss` on an already-built backbone: `x_t = (1 − σ)x0 + σε`, the
/// network at the integer model time, target `ε − x0`, per-sample MSE over `(C, H, W)` then the
/// batch mean (f32). Returns a scalar tensor.
pub fn flow_loss(dit: &IrisDiT, b: &StepBatch, obj: &FlowObjective) -> Result<Tensor> {
    let one_minus = b.sigma.affine(-1.0, 1.0)?;
    let x_t = one_minus
        .broadcast_mul(&b.x0)?
        .add(&b.sigma.broadcast_mul(&b.noise)?)?;
    let out = dit.forward(
        &x_t,
        &b.t,
        &TextBatch {
            states: &b.states,
            mask: &b.masks,
        },
    )?;
    let target = b.noise.sub(&b.x0)?;
    let pred = match obj.prediction {
        Prediction::Velocity => out,
        Prediction::Clean => {
            let floor = b.sigma.clamp(obj.x_pred_sigma_min as f32, f32::INFINITY)?;
            x_t.sub(&out)?.broadcast_div(&floor)?
        }
    };
    let err = pred.sub(&target)?.sqr()?;
    let (n, c, h, w) = err.dims4()?;
    let per_sample = err.reshape((n, c * h * w))?.mean(1)?;
    Ok(per_sample.mean(0)?)
}

/// The loss of one micro-batch and its gradient over every trainable tensor (a zero gradient for a
/// tensor the graph never reads, as the MLX twin's `keyed_value_and_grad` returns). The gradient is
/// of `loss · grad_scale` (accelerate divides the loss by the accumulation length before backward);
/// the returned loss is unscaled.
pub fn loss_and_grads(
    model: &TrainModel,
    params: &Params,
    batch: &StepBatch,
    obj: &FlowObjective,
    grad_scale: f32,
) -> Result<(f32, Tensors)> {
    let dit = model.dit(&super::optim::snapshot(params))?;
    let loss = flow_loss(&dit, batch, obj)?;
    let scaled = mul_s(&loss, grad_scale as f64)?;
    let store = scaled.backward()?;
    drop(dit);
    let mut grads = Tensors::new();
    for (k, v) in params {
        let g = match store.get(v.as_tensor()) {
            Some(g) => g.detach(),
            None => v.as_tensor().zeros_like()?.detach(),
        };
        grads.insert(k.clone(), g);
    }
    Ok((loss.to_dtype(DType::F32)?.to_vec0::<f32>()?, grads))
}

/// FNV-1a of a tensor key — the per-tensor RNG key component (shared with the MLX twin).
pub fn fnv(key: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in key.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// `n` draws of `U(−bound, bound)` from the host RNG.
fn uniform(rng: &mut HostRng, n: usize, bound: f64) -> Vec<f32> {
    (0..n)
        .map(|_| ((rng.uniform() * 2.0 - 1.0) * bound) as f32)
        .collect()
}

fn to_var(data: Vec<f32>, shape: &[usize], device: &Device) -> Result<Var> {
    Ok(Var::from_tensor(&Tensor::from_vec(data, shape, device)?)?)
}

/// Upstream `initialize_weights` for every backbone tensor (f32 `Var`s), drawn from the shared
/// host RNG keyed on `mix_seed(seed, fnv(key))` — device- and launch-portable (the torch stream is
/// not reproduced; the distribution law is).
pub fn random_init(
    cfg: &ModelConfig,
    seed: u64,
    adaln_zero_init: bool,
    device: &Device,
) -> Result<Params> {
    let shapes = backbone_tensor_shapes(cfg);
    let mut out = Params::new();
    for (key, shape) in &shapes {
        let fan_in = key
            .strip_suffix(".bias")
            .and_then(|m| shapes.iter().find(|(w, _)| *w == format!("{m}.weight")))
            .map(|(_, s)| s[1]);
        let n: usize = shape.iter().product();
        let mut rng = HostRng::new(mix_seed(&[seed, fnv(key)]));
        let data = match init_kind(key, shape, fan_in, adaln_zero_init) {
            InitKind::Zeros => vec![0f32; n],
            InitKind::Ones => vec![1f32; n],
            InitKind::Normal { std } => rng
                .normals_f32(n)
                .into_iter()
                .map(|v| v * std as f32)
                .collect(),
            InitKind::Uniform { bound } => uniform(&mut rng, n, bound),
        };
        out.insert(key.clone(), to_var(data, shape, device)?);
    }
    Ok(out)
}

/// Fresh adapter factors (PEFT init): LoRA `A ~ kaiming_uniform(a=√5) = U(±1/√in)`, `B = 0`; LoKr
/// `w1 = 0`, `w2` (or `w2_a`/`w2_b`) kaiming-uniform on its own fan-in. Every delta starts at 0.
pub fn init_adapter(
    kind: AdapterKind,
    targets: &[AdapterTarget],
    seed: u64,
    device: &Device,
) -> Result<Params> {
    let mut out = Params::new();
    let kaiming = |shape: [usize; 2], key: &str| -> Result<Var> {
        let mut rng = HostRng::new(mix_seed(&[seed, fnv(key)]));
        let bound = 1.0 / (shape[1] as f64).sqrt();
        to_var(
            uniform(&mut rng, shape[0] * shape[1], bound),
            &shape,
            device,
        )
    };
    let zeros = |shape: [usize; 2]| to_var(vec![0f32; shape[0] * shape[1]], &shape, device);
    for t in targets {
        let p = &t.path;
        match kind {
            AdapterKind::Lora { rank, .. } => {
                let a_key = format!("{p}.lora_A.weight");
                out.insert(a_key.clone(), kaiming([rank, t.in_f], &a_key)?);
                out.insert(format!("{p}.lora_B.weight"), zeros([t.out_f, rank])?);
            }
            AdapterKind::Lokr { rank, factor, .. } => {
                let (out_a, out_b) = factorization(t.out_f, factor);
                let (in_a, in_b) = factorization(t.in_f, factor);
                out.insert(format!("{p}.lokr_w1"), zeros([out_a, in_a])?);
                if lokr_w2_factored(rank, out_b, in_b) {
                    let ka = format!("{p}.lokr_w2_a");
                    let kb = format!("{p}.lokr_w2_b");
                    out.insert(ka.clone(), kaiming([out_b, rank], &ka)?);
                    out.insert(kb.clone(), kaiming([rank, in_b], &kb)?);
                } else {
                    let k2 = format!("{p}.lokr_w2");
                    out.insert(k2.clone(), kaiming([out_b, in_b], &k2)?);
                }
            }
        }
    }
    Ok(out)
}

/// Write `tensors` (with `metadata`) to `path` atomically and durably: a sibling temp file, fsync,
/// rename, then (best effort) the directory.
pub fn save_safetensors_atomic(
    path: &Path,
    tensors: &[(String, Tensor)],
    metadata: &BTreeMap<String, String>,
) -> Result<()> {
    let tmp = path.with_file_name(format!(
        ".{}.tmp{}.safetensors",
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("artifact"),
        std::process::id()
    ));
    let meta: HashMap<String, String> = metadata.clone().into_iter().collect();
    let list: Vec<(&str, &Tensor)> = tensors.iter().map(|(k, v)| (k.as_str(), v)).collect();
    safetensors::serialize_to_file(list, Some(meta), &tmp)
        .map_err(|e| Error::Msg(format!("iris: write {}: {e}", tmp.display())))?;
    // Write access: Windows' `FlushFileBuffers` refuses a read-only handle.
    std::fs::OpenOptions::new()
        .write(true)
        .open(&tmp)
        .and_then(|f| f.sync_all())
        .map_err(|e| Error::Msg(format!("iris: sync {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, path)
        .map_err(|e| Error::Msg(format!("iris: publish {}: {e}", path.display())))?;
    if let Some(dir) = path.parent() {
        if let Ok(d) = std::fs::File::open(dir) {
            let _ = d.sync_all();
        }
    }
    Ok(())
}

/// The adapter artifact's tensors: the f32 factors plus a `‹path›.alpha` scalar per target.
pub fn adapter_tensors(
    kind: AdapterKind,
    targets: &[AdapterTarget],
    factors: &Tensors,
) -> Result<Vec<(String, Tensor)>> {
    let mut out = Vec::new();
    for t in targets {
        for k in t.factor_keys(kind) {
            let a = factors
                .get(&k)
                .ok_or_else(|| Error::Msg(format!("iris adapter: factor {k} missing")))?;
            out.push((k, a.to_dtype(DType::F32)?));
        }
        out.push((
            format!("{}.alpha", t.path),
            Tensor::from_vec(vec![kind.alpha()], (1,), &Device::Cpu)?,
        ));
    }
    Ok(out)
}

// =============================================================================================
// sc-25689: activation checkpointing (segmented recompute), REPA, per-sample (holdout) loss
// =============================================================================================

/// What a training forward adds to the inference graph: the REPA objective and the activation
/// checkpointing policy.
#[derive(Clone, Debug)]
pub struct TrainExtras {
    pub repa: Option<RepaPlan>,
    pub ac: AcPolicy,
}

impl Default for TrainExtras {
    fn default() -> Self {
        Self {
            repa: None,
            ac: AcPolicy::None,
        }
    }
}

/// One micro-batch's REPA targets: the frozen teacher's tokens (spatially normalized under iREPA)
/// and the student / teacher patch grids.
pub struct RepaTargets {
    /// `[B, Nt, Dt]` f32.
    pub teacher: Tensor,
    pub student_grid: (usize, usize),
    pub teacher_grid: (usize, usize),
}

/// One activation-checkpointing region of a forward: its original inputs (on the tape, or not)
/// and the boundary `Var`s its tape-free outputs became.
enum Region {
    Trunk {
        i: usize,
        ctx: TrunkCtx,
        inputs: Vec<Tensor>,
        has_txt: bool,
        outs: (Var, Var),
    },
    Pixel {
        i: usize,
        ctx: PixelCtx,
        inputs: Vec<Tensor>,
        out: Var,
    },
}

/// The training forward's hooks. A checkpointed region runs **off the tape** on the detached
/// backbone (`tape_free`, composable kernels forced so it computes exactly what its recompute
/// computes) and hands its outputs on as fresh boundary `Var`s; the segmented backward
/// ([`backward_segmented`]) replays it on the tape. Candle has no checkpoint primitive; this is
/// the chain rule over the region boundary — the gradient of the plain graph.
struct TrainHooks<'a> {
    tape_free: &'a IrisDiT,
    ac: AcPolicy,
    capture: Option<usize>,
    captured: Option<Tensor>,
    regions: Vec<Region>,
}

impl ForwardHooks for TrainHooks<'_> {
    fn trunk_block(
        &mut self,
        dit: &IrisDiT,
        i: usize,
        ctx: &TrunkCtx,
        s: &Tensor,
        y: &Tensor,
        core_img: &Tensor,
        core_txt: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (so, yo) = if self.ac.should_checkpoint(i) {
            let (so, yo) = with_composable(|| {
                self.tape_free.run_trunk_block(
                    i,
                    ctx,
                    &s.detach(),
                    &y.detach(),
                    &core_img.detach(),
                    core_txt.map(Tensor::detach).as_ref(),
                )
            })?;
            let sv = Var::from_tensor(&so)?;
            let yv = Var::from_tensor(&yo)?;
            let mut inputs = vec![s.clone(), y.clone(), core_img.clone()];
            if let Some(t) = core_txt {
                inputs.push(t.clone());
            }
            self.regions.push(Region::Trunk {
                i,
                ctx: ctx.clone(),
                inputs,
                has_txt: core_txt.is_some(),
                outs: (sv.clone(), yv.clone()),
            });
            (sv.as_tensor().clone(), yv.as_tensor().clone())
        } else {
            dit.run_trunk_block(i, ctx, s, y, core_img, core_txt)?
        };
        if self.capture == Some(i + 1) {
            self.captured = Some(so.clone());
        }
        Ok((so, yo))
    }

    fn pixel_block(
        &mut self,
        dit: &IrisDiT,
        i: usize,
        ctx: &PixelCtx,
        pixels: &Tensor,
        s_cond: &Tensor,
    ) -> Result<Tensor> {
        if !self.ac.should_checkpoint(i) {
            return dit.run_pixel_block(i, ctx, pixels, s_cond);
        }
        let out = with_composable(|| {
            self.tape_free
                .run_pixel_block(i, ctx, &pixels.detach(), &s_cond.detach())
        })?;
        let v = Var::from_tensor(&out)?;
        self.regions.push(Region::Pixel {
            i,
            ctx: ctx.clone(),
            inputs: vec![pixels.clone(), s_cond.clone()],
            out: v.clone(),
        });
        Ok(v.as_tensor().clone())
    }
}

/// `Σ sum(a · b)` over the pairs with a cotangent, or `None` when there is none.
fn surrogate(pairs: &[(Tensor, Option<Tensor>)]) -> Result<Option<Tensor>> {
    let mut acc: Option<Tensor> = None;
    for (a, g) in pairs {
        let Some(g) = g else { continue };
        let term = a
            .to_dtype(DType::F32)?
            .mul(&g.to_dtype(DType::F32)?)?
            .sum_all()?;
        acc = Some(match acc {
            None => term,
            Some(x) => x.add(&term)?,
        });
    }
    Ok(acc)
}

/// Add every parameter gradient of `store` into `acc`.
fn merge_params(
    acc: &mut Tensors,
    store: &candle_gen::candle_core::backprop::GradStore,
    params: &Params,
) -> Result<()> {
    for (k, v) in params {
        if let Some(g) = store.get(v.as_tensor()) {
            let g = g.detach();
            let s = match acc.remove(k) {
                Some(prev) => prev.add(&g)?,
                None => g,
            };
            acc.insert(k.clone(), s);
        }
    }
    Ok(())
}

/// Add the gradients `store` holds for the earlier regions' boundary `Var`s into `cot`.
fn merge_boundaries(
    cot: &mut HashMap<TensorId, Tensor>,
    store: &candle_gen::candle_core::backprop::GradStore,
    regions: &[Region],
) -> Result<()> {
    for r in regions {
        let outs: Vec<&Var> = match r {
            Region::Trunk { outs, .. } => vec![&outs.0, &outs.1],
            Region::Pixel { out, .. } => vec![out],
        };
        for v in outs {
            if let Some(g) = store.get(v.as_tensor()) {
                let id = v.as_tensor().id();
                let s = match cot.remove(&id) {
                    Some(prev) => prev.add(&g.detach())?,
                    None => g.detach(),
                };
                cot.insert(id, s);
            }
        }
    }
    Ok(())
}

/// The gradient of `loss` over `params` through a forward with checkpointed `regions`: the plain
/// backward to the last boundaries, then, region by region in reverse, its recompute on the tape
/// (the region's own parameter gradients and its input cotangents) and the propagation of those
/// cotangents into whatever produced its inputs.
fn backward_segmented(
    loss: &Tensor,
    regions: Vec<Region>,
    tracked: &IrisDiT,
    params: &Params,
) -> Result<Tensors> {
    let mut acc = Tensors::new();
    let mut cot: HashMap<TensorId, Tensor> = HashMap::new();
    let store = loss.backward()?;
    merge_params(&mut acc, &store, params)?;
    merge_boundaries(&mut cot, &store, &regions)?;
    drop(store);
    for idx in (0..regions.len()).rev() {
        let (earlier, rest) = regions.split_at(idx);
        let region = &rest[0];
        let take = |cot: &mut HashMap<TensorId, Tensor>, v: &Var| cot.remove(&v.as_tensor().id());
        let (outs, in_vars): (Vec<(Tensor, Option<Tensor>)>, Vec<Var>) = match region {
            Region::Trunk {
                i,
                ctx,
                inputs,
                has_txt,
                outs,
            } => {
                let vars: Vec<Var> = inputs
                    .iter()
                    .map(|t| Var::from_tensor(&t.detach()))
                    .collect::<candle_gen::candle_core::Result<_>>()?;
                let (so, yo) = tracked.run_trunk_block(
                    *i,
                    ctx,
                    vars[0].as_tensor(),
                    vars[1].as_tensor(),
                    vars[2].as_tensor(),
                    has_txt.then(|| vars[3].as_tensor()),
                )?;
                (
                    vec![(so, take(&mut cot, &outs.0)), (yo, take(&mut cot, &outs.1))],
                    vars,
                )
            }
            Region::Pixel {
                i,
                ctx,
                inputs,
                out,
            } => {
                let vars: Vec<Var> = inputs
                    .iter()
                    .map(|t| Var::from_tensor(&t.detach()))
                    .collect::<candle_gen::candle_core::Result<_>>()?;
                let o =
                    tracked.run_pixel_block(*i, ctx, vars[0].as_tensor(), vars[1].as_tensor())?;
                (vec![(o, take(&mut cot, out))], vars)
            }
        };
        let Some(sur) = surrogate(&outs)? else {
            continue;
        };
        let st = sur.backward()?;
        merge_params(&mut acc, &st, params)?;
        let inputs = match region {
            Region::Trunk { inputs, .. } | Region::Pixel { inputs, .. } => inputs,
        };
        let pairs: Vec<(Tensor, Option<Tensor>)> = inputs
            .iter()
            .zip(&in_vars)
            .filter(|(orig, _)| orig.track_op())
            .map(|(orig, v)| (orig.clone(), st.get(v.as_tensor()).map(Tensor::detach)))
            .collect();
        drop(st);
        if let Some(sur2) = surrogate(&pairs)? {
            let st2 = sur2.backward()?;
            merge_params(&mut acc, &st2, params)?;
            merge_boundaries(&mut cot, &st2, earlier)?;
        }
    }
    Ok(acc)
}

/// `F.normalize(x, dim=-1)`: `x / max(‖x‖₂, 1e−12)`.
fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let n = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?.maximum(1e-12)?;
    Ok(x.broadcast_div(&n)?)
}

/// `_resize_token_grid`: bilinearly resample `[B, h·w, C]` tokens onto `dst`.
pub fn resize_token_grid(
    tokens: &Tensor,
    src: (usize, usize),
    dst: (usize, usize),
) -> Result<Tensor> {
    let (b, _, c) = tokens.dims3()?;
    let dev = tokens.device();
    let mh = Tensor::from_vec(bilinear_matrix(src.0, dst.0), (dst.0, src.0), dev)?;
    let mw = Tensor::from_vec(bilinear_matrix(src.1, dst.1), (dst.1, src.1), dev)?;
    let x = tokens
        .transpose(1, 2)?
        .contiguous()?
        .reshape((b, c, src.0, src.1))?;
    let x = mh.broadcast_matmul(&x)?.broadcast_matmul(&mw.t()?)?;
    Ok(x.reshape((b, c, dst.0 * dst.1))?
        .transpose(1, 2)?
        .contiguous()?)
}

fn project(plan: &RepaPlan, p: &Tensors, x: &Tensor, grid: (usize, usize)) -> Result<Tensor> {
    let get = |leaf: &str| -> Result<&Tensor> {
        let k = format!("{PROJECTOR_PREFIX}{leaf}");
        p.get(&k)
            .ok_or_else(|| Error::Msg(format!("iris repa: projector tensor {k} missing")))
    };
    let lin = |x: &Tensor, i: usize| -> Result<Tensor> {
        Ok(x.broadcast_matmul(&get(&format!("{i}.weight"))?.t()?)?
            .broadcast_add(get(&format!("{i}.bias"))?)?)
    };
    match plan.variant {
        RepaVariant::Repa => {
            let h = lin(x, 0)?.silu()?;
            let h = lin(&h, 2)?.silu()?;
            lin(&h, 4)
        }
        RepaVariant::Irepa => {
            let (b, n, d) = x.dims3()?;
            let xs = x
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b, d, grid.0, grid.1))?;
            let y = xs.conv2d(get("weight")?, 1, 1, 1, 1)?;
            let t = y.dim(1)?;
            let y = y.broadcast_add(&get("bias")?.reshape((1, t, 1, 1))?)?;
            Ok(y.reshape((b, t, n))?.transpose(1, 2)?.contiguous()?)
        }
    }
}

/// `REPALoss.forward` given the teacher's tokens: `−mean(cos(student, teacher))`.
pub fn repa_loss(
    plan: &RepaPlan,
    p: &Tensors,
    student_tokens: &Tensor,
    t: &RepaTargets,
) -> Result<Tensor> {
    let x = student_tokens.to_dtype(DType::F32)?;
    let mut student = l2_normalize(&project(plan, p, &x, t.student_grid)?)?;
    let mut teacher = t.teacher.clone();
    if t.teacher_grid != t.student_grid {
        let nt = t.teacher_grid.0 * t.teacher_grid.1;
        let ns = t.student_grid.0 * t.student_grid.1;
        if nt > ns {
            teacher = resize_token_grid(&teacher, t.teacher_grid, t.student_grid)?;
        } else {
            student = resize_token_grid(&student, t.student_grid, t.teacher_grid)?;
        }
    }
    let teacher = l2_normalize(&teacher)?;
    let student = l2_normalize(&student)?;
    let cos = student.mul(&teacher)?.sum(D::Minus1)?;
    Ok(cos.mean_all()?.neg()?)
}

fn per_sample_loss(
    dit: &IrisDiT,
    b: &StepBatch,
    obj: &FlowObjective,
    hooks: &mut dyn ForwardHooks,
) -> Result<Tensor> {
    let one_minus = b.sigma.affine(-1.0, 1.0)?;
    let x_t = one_minus
        .broadcast_mul(&b.x0)?
        .add(&b.sigma.broadcast_mul(&b.noise)?)?;
    let out = dit.forward_hooked(
        &x_t,
        &b.t,
        &TextBatch {
            states: &b.states,
            mask: &b.masks,
        },
        hooks,
    )?;
    let target = b.noise.sub(&b.x0)?;
    let pred = match obj.prediction {
        Prediction::Velocity => out,
        Prediction::Clean => {
            let floor = b.sigma.clamp(obj.x_pred_sigma_min as f32, f32::INFINITY)?;
            x_t.sub(&out)?.broadcast_div(&floor)?
        }
    };
    let err = pred.sub(&target)?.sqr()?;
    let (n, c, h, w) = err.dims4()?;
    Ok(err.reshape((n, c * h * w))?.mean(1)?)
}

/// The per-sample rectified-flow loss `[B]` (`reduction="none"`, the holdout grid's form).
pub fn flow_loss_per_sample(dit: &IrisDiT, b: &StepBatch, obj: &FlowObjective) -> Result<Tensor> {
    per_sample_loss(dit, b, obj, &mut NoHooks)
}

/// The losses of one micro-batch and the gradient of `(flow + weight · repa) · grad_scale` over
/// every trainable tensor (backbone or adapter factors, plus the REPA projector; a zero gradient
/// for a tensor the graph never reads). Returns the unscaled flow and REPA losses.
#[allow(clippy::too_many_arguments)]
pub fn loss_and_grads_ext(
    model: &TrainModel,
    params: &Params,
    batch: &StepBatch,
    obj: &FlowObjective,
    extras: &TrainExtras,
    repa: Option<&RepaTargets>,
    grad_scale: f32,
) -> Result<(f32, Option<f32>, Tensors)> {
    let snap = super::optim::snapshot(params);
    let tracked = model.dit(&snap)?;
    let detached: Tensors = snap.iter().map(|(k, v)| (k.clone(), v.detach())).collect();
    let tape_free = if extras.ac.active() {
        Some(model.dit(&detached)?)
    } else {
        None
    };
    let dummy;
    let mut hooks = TrainHooks {
        tape_free: match &tape_free {
            Some(d) => d,
            None => {
                dummy = &tracked;
                dummy
            }
        },
        ac: extras.ac,
        capture: extras.repa.as_ref().map(|r| r.layer),
        captured: None,
        regions: Vec::new(),
    };
    let flow = per_sample_loss(&tracked, batch, obj, &mut hooks)?.mean(0)?;
    let repa_l = match (&extras.repa, repa) {
        (Some(plan), Some(t)) => {
            let feats = hooks.captured.take().ok_or_else(|| {
                Error::Msg(format!("iris repa: block {} was never reached", plan.layer))
            })?;
            Some(repa_loss(plan, &snap, &feats, t)?)
        }
        (Some(_), None) => {
            return Err(Error::Msg(
                "iris repa: the micro-batch carries no teacher targets".into(),
            ))
        }
        _ => None,
    };
    let total = match (&repa_l, &extras.repa) {
        (Some(r), Some(plan)) => flow.add(&mul_s(r, plan.weight)?)?,
        _ => flow.clone(),
    };
    let scaled = mul_s(&total, grad_scale as f64)?;
    let regions = std::mem::take(&mut hooks.regions);
    let mut grads = backward_segmented(&scaled, regions, &tracked, params)?;
    drop(tracked);
    for (k, v) in params {
        if !grads.contains_key(k) {
            grads.insert(k.clone(), v.as_tensor().zeros_like()?.detach());
        }
    }
    let scalar = |t: &Tensor| -> Result<f32> { Ok(t.to_dtype(DType::F32)?.to_vec0::<f32>()?) };
    Ok((
        scalar(&flow)?,
        repa_l.as_ref().map(scalar).transpose()?,
        grads,
    ))
}

/// PyTorch-default init of the REPA projector (`U(±1/√fan_in)`), from the shared host RNG keyed
/// on `mix_seed(seed, fnv(key))`.
pub fn init_projector(plan: &RepaPlan, seed: u64, device: &Device) -> Result<Params> {
    let mut out = Params::new();
    for (key, shape) in plan.projector_shapes() {
        let bound = plan.projector_init_bound(&key);
        let n: usize = shape.iter().product();
        let mut rng = HostRng::new(mix_seed(&[seed, fnv(&key)]));
        out.insert(
            key.clone(),
            to_var(uniform(&mut rng, n, bound), &shape, device)?,
        );
    }
    Ok(out)
}
