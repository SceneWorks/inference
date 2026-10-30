//! The trainable Iris backbone: which tensors are trained (every backbone tensor, or LoRA/LoKr
//! factors over frozen Linears), how a step's weights are assembled from them, the rectified-flow
//! loss and its gradient, random init, and the artifact writers.
//!
//! The forward is **the inference forward**: every step rebuilds an [`IrisDiT`] inside the autograd
//! trace exactly as the provider loads a checkpoint ([`IrisDiT::from_weights`] from the traced
//! tensors of a full run) or installs an exported adapter file (forward-time residuals over the
//! frozen base, [`IrisDiT::from_weights_adapted`]), so training and inference can never disagree
//! about the graph. Mixed precision is the provider's autocast policy ([`crate::nn`]): f32 master tensors,
//! matmuls/attention in the compute dtype (bf16 under `mixed_precision: bf16`), norms/residuals/loss
//! in f32; the casts are traced, so gradients reach the f32 masters.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::rc::Rc;

use gen_core_iris::train::{
    backbone_tensor_shapes, init_kind, mix_seed, FlowObjective, InitKind, Prediction,
};
use gen_core_iris::ModelConfig;
use mlx_gen::adapters::{reconstruct_lokr_delta, AdaptableLinear, Adapter};
use mlx_gen::train::lora::factorization;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::error::Exception;
use mlx_rs::ops::{maximum, mean_axes};
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use super::optim::Params;
use crate::dit::{ForwardHooks, IrisDiT, PixelCtx, TextBatch, TrunkCtx};
use crate::nn::AdaptedLinears;
use gen_core_iris::train::ac::AcPolicy;
use gen_core_iris::train::repa::{bilinear_matrix, RepaPlan, RepaVariant, PROJECTOR_PREFIX};
use mlx_rs::nn::silu;
use mlx_rs::ops::{matmul, sqrt, sum_axes};
use mlx_rs::transforms::checkpoint;

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
    pub out_f: i32,
    pub in_f: i32,
}

impl AdapterTarget {
    /// The factor keys this target trains, in a fixed order.
    pub fn factor_keys(&self, kind: AdapterKind) -> Vec<String> {
        let p = &self.path;
        match kind {
            AdapterKind::Lora { .. } => {
                vec![format!("{p}.lora_A.weight"), format!("{p}.lora_B.weight")]
            }
            AdapterKind::Lokr { rank, factor, .. } => {
                let (_, out_b) = factorization(self.out_f, factor);
                let (_, in_b) = factorization(self.in_f, factor);
                if (rank as f32) < (out_b.max(in_b) as f32) / 2.0 {
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
    /// dtypes (matrices in the compute dtype, vectors f32).
    Adapter {
        kind: AdapterKind,
        targets: Vec<AdapterTarget>,
        base: HashMap<String, Array>,
    },
}

/// The trainable backbone.
pub struct TrainModel {
    pub cfg: ModelConfig,
    pub compute: Dtype,
    pub trainable: Trainable,
}

fn to_exc(e: impl std::fmt::Display) -> Exception {
    Exception::custom(e.to_string())
}

pub(crate) fn fnv(key: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in key.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// Cast a backbone tensor the way [`crate::model::load_backbone`] does: matrices (except the text
/// position table) to the compute dtype, vectors kept.
pub fn provider_dtype(key: &str, a: &Array, compute: Dtype) -> Result<Array> {
    Ok(if a.ndim() >= 2 && key != "y_pos_embedding" {
        a.as_dtype(compute)?
    } else {
        a.clone()
    })
}

/// One target's adapter exactly as the provider's strict loader installs the exported file
/// (`mlx_gen::adapters::loader`): a LoRA as `Adapter::Lora { a: Aᵀ, b: Bᵀ · alpha/rank, scale: 1 }`
/// (factors at their f32 master dtype, the file's dtype); a LoKr as `Adapter::Lokr` over the
/// `[out, in]` delta reconstructed at `compute` and `scale = 1`. Under bf16 compute that is exactly
/// the provider (`apply_lokr` reconstructs in bf16, the residual runs in the bf16 activation dtype);
/// under f32 compute the delta stays f32 (the provider's bf16 reconstruction would round the
/// gradient path of an f32 run — a sub-RGB8-level difference in the render). Every op is traced, so
/// gradients reach the f32 factors.
pub fn provider_adapter(
    compute: Dtype,
    kind: AdapterKind,
    t: &AdapterTarget,
    p: &HashMap<Rc<str>, Array>,
) -> Result<Adapter> {
    let get = |k: &str| -> Result<&Array> {
        p.get(k)
            .ok_or_else(|| Error::Msg(format!("iris adapter: factor {k} missing")))
    };
    match kind {
        AdapterKind::Lora { rank, alpha } => {
            let a = get(&format!("{}.lora_A.weight", t.path))?.t();
            let b = get(&format!("{}.lora_B.weight", t.path))?
                .t()
                .multiply(Array::from_slice(&[alpha / rank as f32], &[1]))?;
            Ok(Adapter::Lora { a, b, scale: 1.0 })
        }
        AdapterKind::Lokr { rank, alpha, .. } => {
            let f = |name: &str| p.get(format!("{}.{name}", t.path).as_str());
            let delta = reconstruct_lokr_delta(
                alpha,
                rank as f32,
                &[t.out_f, t.in_f],
                f("lokr_w1"),
                None,
                None,
                f("lokr_w2"),
                f("lokr_w2_a"),
                f("lokr_w2_b"),
                compute,
            )?;
            Ok(Adapter::Lokr { delta, scale: 1.0 })
        }
    }
}

impl TrainModel {
    /// The backbone of one step at the compute dtype.
    ///
    /// A full run builds it from the (traced) trainable tensors. An adapter run builds the frozen
    /// base exactly as the provider does and installs each target's adapter as a **forward-time
    /// residual** — `y = base(x) + adapter(x)`, narrowed to the projection's output dtype — through
    /// the provider's own [`AdaptedLinears`] path, never merged into the (bf16) base weight, so a
    /// delta below the base's bf16 ulp still reaches the forward, the gradient and the previews,
    /// and a preview renders what the exported file renders in the provider.
    pub fn dit(&self, trainable: &HashMap<Rc<str>, Array>) -> Result<IrisDiT> {
        match &self.trainable {
            Trainable::Full => {
                // The REPA projector (sc-25689) trains beside the backbone; it is not a backbone
                // tensor.
                let map: HashMap<String, Array> = trainable
                    .iter()
                    .filter(|(k, _)| !k.starts_with(PROJECTOR_PREFIX))
                    .map(|(k, v)| (k.to_string(), v.clone()))
                    .collect();
                IrisDiT::from_weights(&Weights::from_map(map), &self.cfg, self.compute)
            }
            Trainable::Adapter {
                kind,
                targets,
                base,
            } => {
                let mut linears = BTreeMap::new();
                let mut adapted = BTreeSet::new();
                for t in targets {
                    let w = base.get(&format!("{}.weight", t.path)).ok_or_else(|| {
                        Error::Msg(format!("iris adapter: base has no {}.weight", t.path))
                    })?;
                    let bias = base
                        .get(&format!("{}.bias", t.path))
                        .map(|b| b.as_dtype(self.compute))
                        .transpose()?;
                    let mut lin = AdaptableLinear::dense(w.as_dtype(self.compute)?, bias);
                    lin.set_adapters(vec![provider_adapter(self.compute, *kind, t, trainable)?]);
                    linears.insert(t.path.clone(), lin);
                    adapted.insert(t.path.clone());
                }
                let adapted = AdaptedLinears::new(linears, adapted);
                IrisDiT::from_weights_adapted(
                    &Weights::from_map(base.clone()),
                    &self.cfg,
                    self.compute,
                    Some(&adapted),
                )
            }
        }
    }
}

/// One micro-batch, assembled host-side from the data walk and the positional draws.
pub struct StepBatch {
    /// `[B, C, H, W]` f32 in `[−1, 1]`.
    pub x0: Array,
    /// `[B, C, H, W]` f32.
    pub noise: Array,
    /// `[B, 1, 1, 1]` f32 — `sigmas[idx]`.
    pub sigma: Array,
    /// `[B]` f32 — the integer-truncated model time.
    pub t: Array,
    /// `[B, T, L, Dt]` f32 text states (dropped rows already replaced by the null).
    pub states: Array,
    pub masks: Vec<Vec<i32>>,
}

/// `RectifiedFlow.training_loss` on an already-built backbone: `x_t = (1 − σ)x0 + σε`, the
/// network at the integer model time, target `ε − x0`, per-sample MSE over `(C, H, W)` then the
/// batch mean (f32).
pub fn flow_loss(dit: &IrisDiT, b: &StepBatch, obj: &FlowObjective) -> Result<Array> {
    let one = Array::from_f32(1.0);
    let x_t = one
        .subtract(&b.sigma)?
        .multiply(&b.x0)?
        .add(&b.sigma.multiply(&b.noise)?)?;
    let out = dit.forward(
        &x_t,
        &b.t,
        &TextBatch {
            states: &b.states,
            mask: &b.masks,
        },
    )?;
    let target = b.noise.subtract(&b.x0)?;
    let pred = match obj.prediction {
        Prediction::Velocity => out,
        Prediction::Clean => {
            let floor = Array::from_f32(obj.x_pred_sigma_min as f32);
            x_t.subtract(&out)?.divide(&maximum(&b.sigma, &floor)?)?
        }
    };
    let err = pred.subtract(&target)?.square()?;
    let per_sample = mean_axes(&err, &[1, 2, 3], false)?;
    Ok(per_sample.mean(None)?)
}

/// The loss of one micro-batch and its gradient over every trainable tensor. The gradient is of
/// `loss · grad_scale` (accelerate divides the loss by the accumulation length before backward);
/// the returned loss is unscaled.
pub fn loss_and_grads(
    model: &TrainModel,
    trainable: &Params,
    batch: &StepBatch,
    obj: &FlowObjective,
    grad_scale: f32,
) -> Result<(f32, Params)> {
    let loss_fn = |p: HashMap<Rc<str>, Array>, _: i32| -> mlx_rs::error::Result<Vec<Array>> {
        // Never poll cancellation in here: the trace is one atomic unit.
        let dit = model.dit(&p).map_err(to_exc)?;
        let loss = flow_loss(&dit, batch, obj).map_err(to_exc)?;
        let scaled = loss.multiply(Array::from_f32(grad_scale))?;
        Ok(vec![scaled, loss])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (vals, grads) =
        vg(trainable.clone(), 0).map_err(|e| Error::Msg(format!("iris train step: {e}")))?;
    let loss = vals[1].clone();
    let mut refs: Vec<&Array> = vec![&loss];
    refs.extend(grads.values());
    for chunk in refs.chunks(256) {
        eval(chunk.iter().copied())?;
    }
    Ok((super::scalar_f32(&loss)?, grads))
}

/// Upstream `initialize_weights` for every backbone tensor (f32), drawn from MLX's RNG keyed on
/// `mix_seed(seed, fnv(key))` (the torch stream is not reproduced; the distribution law is).
pub fn random_init(cfg: &ModelConfig, seed: u64, adaln_zero_init: bool) -> Result<Params> {
    let shapes = backbone_tensor_shapes(cfg);
    let mut out = Params::with_capacity(shapes.len());
    for (key, shape) in &shapes {
        let fan_in = key
            .strip_suffix(".bias")
            .and_then(|m| shapes.iter().find(|(w, _)| *w == format!("{m}.weight")))
            .map(|(_, s)| s[1]);
        let dims: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let rng = random::key(mix_seed(&[seed, fnv(key)]))?;
        let a = match init_kind(key, shape, fan_in, adaln_zero_init) {
            InitKind::Zeros => Array::zeros::<f32>(&dims)?,
            InitKind::Ones => Array::ones::<f32>(&dims)?,
            InitKind::Normal { std } => {
                random::normal::<f32>(&dims[..], None, Some(std as f32), Some(&rng))?
            }
            InitKind::Uniform { bound } => {
                random::uniform::<_, f32>(-(bound as f32), bound as f32, &dims[..], Some(&rng))?
            }
        };
        eval([&a])?;
        out.insert(Rc::from(key.as_str()), a);
    }
    Ok(out)
}

/// Fresh adapter factors (PEFT init): LoRA `A ~ kaiming_uniform(a=√5) = U(±1/√in)`, `B = 0`; LoKr
/// `w1 = 0`, `w2` (or `w2_a`/`w2_b`) kaiming-uniform on its own fan-in. Every delta starts at 0.
pub fn init_adapter(kind: AdapterKind, targets: &[AdapterTarget], seed: u64) -> Result<Params> {
    let mut out = Params::new();
    let kaiming = |shape: [i32; 2], key: &str| -> Result<Array> {
        let bound = 1.0f32 / (shape[1] as f32).sqrt();
        let k = random::key(mix_seed(&[seed, fnv(key)]))?;
        Ok(random::uniform::<_, f32>(
            -bound,
            bound,
            &shape[..],
            Some(&k),
        )?)
    };
    for t in targets {
        let p = &t.path;
        match kind {
            AdapterKind::Lora { rank, .. } => {
                let r = rank as i32;
                let a_key = format!("{p}.lora_A.weight");
                out.insert(Rc::from(a_key.as_str()), kaiming([r, t.in_f], &a_key)?);
                out.insert(
                    Rc::from(format!("{p}.lora_B.weight").as_str()),
                    Array::zeros::<f32>(&[t.out_f, r])?,
                );
            }
            AdapterKind::Lokr { rank, factor, .. } => {
                let (out_a, out_b) = factorization(t.out_f, factor);
                let (in_a, in_b) = factorization(t.in_f, factor);
                out.insert(
                    Rc::from(format!("{p}.lokr_w1").as_str()),
                    Array::zeros::<f32>(&[out_a, in_a])?,
                );
                let r = rank as i32;
                if (rank as f32) < (out_b.max(in_b) as f32) / 2.0 {
                    let ka = format!("{p}.lokr_w2_a");
                    let kb = format!("{p}.lokr_w2_b");
                    out.insert(Rc::from(ka.as_str()), kaiming([out_b, r], &ka)?);
                    out.insert(Rc::from(kb.as_str()), kaiming([r, in_b], &kb)?);
                } else {
                    let k2 = format!("{p}.lokr_w2");
                    out.insert(Rc::from(k2.as_str()), kaiming([out_b, in_b], &k2)?);
                }
            }
        }
    }
    let refs: Vec<&Array> = out.values().collect();
    eval(refs)?;
    Ok(out)
}

/// Write `tensors` to `path` atomically (a sibling temp `.safetensors`, then rename).
pub fn save_safetensors_atomic(
    path: &Path,
    tensors: &[(String, Array)],
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
    Array::save_safetensors(
        tensors.iter().map(|(k, v)| (k.as_str(), v)),
        Some(&meta),
        &tmp,
    )?;
    std::fs::rename(&tmp, path)
        .map_err(|e| Error::Msg(format!("iris: publish {}: {e}", path.display())))?;
    Ok(())
}

/// The adapter artifact's tensors: the factors plus a `‹path›.alpha` scalar per target.
pub fn adapter_tensors(
    kind: AdapterKind,
    targets: &[AdapterTarget],
    factors: &Params,
) -> Result<Vec<(String, Array)>> {
    let mut out = Vec::new();
    for t in targets {
        for k in t.factor_keys(kind) {
            let a = factors
                .get(k.as_str())
                .ok_or_else(|| Error::Msg(format!("iris adapter: factor {k} missing")))?;
            out.push((k, a.as_dtype(Dtype::Float32)?));
        }
        out.push((
            format!("{}.alpha", t.path),
            Array::from_slice(&[kind.alpha()], &[1]),
        ));
    }
    Ok(out)
}

// =============================================================================================
// sc-25689: activation checkpointing, REPA, per-sample (holdout) loss
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
    pub teacher: Array,
    pub student_grid: (usize, usize),
    pub teacher_grid: (usize, usize),
}

/// The trainable keys under `prefix`, sorted (one region's inputs).
fn region_keys(p: &HashMap<Rc<str>, Array>, prefix: &str) -> Vec<Rc<str>> {
    let mut keys: Vec<Rc<str>> = p
        .keys()
        .filter(|k| k.starts_with(prefix))
        .cloned()
        .collect();
    keys.sort();
    keys
}

impl TrainModel {
    /// The weights + adapter install of the block under `prefix`, built from `arrays` (the
    /// region's trainable inputs under `keys`) exactly as [`TrainModel::dit`] builds the whole
    /// backbone.
    fn region_source(
        &self,
        prefix: &str,
        keys: &[Rc<str>],
        arrays: &[Array],
    ) -> Result<(Weights, Option<AdaptedLinears>)> {
        match &self.trainable {
            Trainable::Full => {
                let map: HashMap<String, Array> = keys
                    .iter()
                    .zip(arrays)
                    .map(|(k, a)| (k.to_string(), a.clone()))
                    .collect();
                Ok((Weights::from_map(map), None))
            }
            Trainable::Adapter {
                kind,
                targets,
                base,
            } => {
                let factors: HashMap<Rc<str>, Array> = keys
                    .iter()
                    .zip(arrays)
                    .map(|(k, a)| (k.clone(), a.clone()))
                    .collect();
                let base_map: HashMap<String, Array> = base
                    .iter()
                    .filter(|(k, _)| k.starts_with(prefix))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                let mut linears = BTreeMap::new();
                let mut adapted = BTreeSet::new();
                for t in targets.iter().filter(|t| t.path.starts_with(prefix)) {
                    let w = base.get(&format!("{}.weight", t.path)).ok_or_else(|| {
                        Error::Msg(format!("iris adapter: base has no {}.weight", t.path))
                    })?;
                    let bias = base
                        .get(&format!("{}.bias", t.path))
                        .map(|b| b.as_dtype(self.compute))
                        .transpose()?;
                    let mut lin = AdaptableLinear::dense(w.as_dtype(self.compute)?, bias);
                    lin.set_adapters(vec![provider_adapter(self.compute, *kind, t, &factors)?]);
                    linears.insert(t.path.clone(), lin);
                    adapted.insert(t.path.clone());
                }
                Ok((
                    Weights::from_map(base_map),
                    Some(AdaptedLinears::new(linears, adapted)),
                ))
            }
        }
    }
}

thread_local! {
    static CHECKPOINTED_REGIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// How many activation-checkpointed regions this thread's training forwards entered — a test
/// probe: a policy's region set is otherwise invisible (the gradient is the plain graph's either
/// way).
#[doc(hidden)]
pub fn checkpointed_regions() -> usize {
    CHECKPOINTED_REGIONS.with(|c| c.get())
}

fn note_checkpointed_region() {
    CHECKPOINTED_REGIONS.with(|c| c.set(c.get() + 1));
}

/// The training forward's hooks: activation-checkpointing regions (MLX's own checkpoint
/// transform — the region's trainable tensors and activations are its inputs, so the recompute
/// rebuilds the block from them) and the REPA capture of block `capture` (1-based).
struct TrainHooks<'a> {
    model: &'a TrainModel,
    p: &'a HashMap<Rc<str>, Array>,
    ac: AcPolicy,
    capture: Option<usize>,
    captured: Option<Array>,
}

/// The activation-checkpointing hooks of a **task** trainer's forward (sc-25690): the generation
/// trainer's regions — one per trunk block and per PiT block that `ac` selects, each MLX's own
/// checkpoint transform over the block's trainable tensors and input activations — around a
/// backbone the task builds itself (depth's widened one, restoration's generator). `model` names
/// what a region rebuilds its block from and `p` holds the traced tensors of the step; whatever the
/// task computes downstream of the backbone (the depth reducer and Eq. 5, restoration's L2 / LPIPS
/// / critic terms) stays outside every region.
pub(crate) fn region_hooks<'a>(
    model: &'a TrainModel,
    p: &'a HashMap<Rc<str>, Array>,
    ac: AcPolicy,
) -> impl ForwardHooks + 'a {
    TrainHooks {
        model,
        p,
        ac,
        capture: None,
        captured: None,
    }
}

impl<'a> ForwardHooks for TrainHooks<'a> {
    fn trunk_block(
        &mut self,
        dit: &IrisDiT,
        i: usize,
        ctx: &TrunkCtx,
        s: &Array,
        y: &Array,
        core_img: &Array,
        core_txt: Option<&Array>,
    ) -> Result<(Array, Array)> {
        let (so, yo) = if self.ac.should_checkpoint(i) {
            note_checkpointed_region();
            let prefix = format!("blocks.{i}.");
            let keys = region_keys(self.p, &prefix);
            let has_txt = core_txt.is_some();
            let mut inputs = vec![
                s.clone(),
                y.clone(),
                core_img.clone(),
                core_txt.cloned().unwrap_or_else(|| core_img.clone()),
            ];
            inputs.extend(keys.iter().map(|k| self.p[k].clone()));
            // The recompute runs during the backward, after this forward returned: it owns its
            // context and borrows only the model (alive for the whole value-and-grad call).
            let model: &'a TrainModel = self.model;
            let ctx = ctx.clone();
            let mut region = checkpoint(move |xs: &[Array]| -> mlx_rs::error::Result<Vec<Array>> {
                let (w, adapted) = model
                    .region_source(&prefix, &keys, &xs[4..])
                    .map_err(to_exc)?;
                let blk =
                    IrisDiT::load_trunk_block(&w, &model.cfg, model.compute, i, adapted.as_ref())
                        .map_err(to_exc)?;
                let (so, yo) = blk
                    .run(&ctx, &xs[0], &xs[1], &xs[2], has_txt.then(|| &xs[3]))
                    .map_err(to_exc)?;
                Ok(vec![so, yo])
            });
            let mut out = region(&inputs)?;
            let yo = out.pop().expect("two outputs");
            (out.pop().expect("two outputs"), yo)
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
        pixels: &Array,
        s_cond: &Array,
    ) -> Result<Array> {
        if !self.ac.should_checkpoint(i) {
            return dit.run_pixel_block(i, ctx, pixels, s_cond);
        }
        note_checkpointed_region();
        let prefix = format!("pixel_blocks.{i}.");
        let keys = region_keys(self.p, &prefix);
        let mut inputs = vec![pixels.clone(), s_cond.clone()];
        inputs.extend(keys.iter().map(|k| self.p[k].clone()));
        let model: &'a TrainModel = self.model;
        let ctx = ctx.clone();
        let mut region = checkpoint(move |xs: &[Array]| -> mlx_rs::error::Result<Vec<Array>> {
            let (w, adapted) = model
                .region_source(&prefix, &keys, &xs[2..])
                .map_err(to_exc)?;
            let blk = IrisDiT::load_pixel_block(&w, &model.cfg, model.compute, i, adapted.as_ref())
                .map_err(to_exc)?;
            Ok(vec![blk.run(&ctx, &xs[0], &xs[1]).map_err(to_exc)?])
        });
        Ok(region(&inputs)?.pop().expect("one output"))
    }
}

/// `F.normalize(x, dim=-1)`: `x / max(‖x‖₂, 1e−12)`.
fn l2_normalize(x: &Array) -> Result<Array> {
    let n = sqrt(&sum_axes(&x.square()?, &[-1], true)?)?;
    Ok(x.divide(&maximum(&n, Array::from_f32(1e-12))?)?)
}

/// `_resize_token_grid`: bilinearly resample `[B, h·w, C]` tokens onto `dst`.
pub fn resize_token_grid(
    tokens: &Array,
    src: (usize, usize),
    dst: (usize, usize),
) -> Result<Array> {
    let sh = tokens.shape();
    let (b, c) = (sh[0], sh[2]);
    let mh = Array::from_slice(
        &bilinear_matrix(src.0, dst.0),
        &[dst.0 as i32, src.0 as i32],
    );
    let mw = Array::from_slice(
        &bilinear_matrix(src.1, dst.1),
        &[dst.1 as i32, src.1 as i32],
    );
    let x = tokens
        .transpose_axes(&[0, 2, 1])?
        .reshape(&[b, c, src.0 as i32, src.1 as i32])?;
    let x = matmul(&matmul(&mh, &x)?, mw.t())?;
    Ok(x.reshape(&[b, c, (dst.0 * dst.1) as i32])?
        .transpose_axes(&[0, 2, 1])?)
}

/// The projector (`REPALoss._project`) in fp32: the 3-layer SiLU MLP, or iREPA's 3×3 conv on the
/// student grid.
fn project(
    plan: &RepaPlan,
    p: &HashMap<Rc<str>, Array>,
    x: &Array,
    grid: (usize, usize),
) -> Result<Array> {
    let get = |leaf: &str| -> Result<&Array> {
        let k = format!("{PROJECTOR_PREFIX}{leaf}");
        p.get(k.as_str())
            .ok_or_else(|| Error::Msg(format!("iris repa: projector tensor {k} missing")))
    };
    let lin = |x: &Array, i: usize| -> Result<Array> {
        Ok(matmul(x, get(&format!("{i}.weight"))?.t())?.add(get(&format!("{i}.bias"))?)?)
    };
    match plan.variant {
        RepaVariant::Repa => {
            let h = silu(lin(x, 0)?)?;
            let h = silu(lin(&h, 2)?)?;
            lin(&h, 4)
        }
        RepaVariant::Irepa => {
            let sh = x.shape();
            let (b, d) = (sh[0], sh[2]);
            let (gh, gw) = (grid.0 as i32, grid.1 as i32);
            // NHWC input, OHWI weight.
            let xs = x.reshape(&[b, gh, gw, d])?;
            let w = get("weight")?.transpose_axes(&[0, 2, 3, 1])?;
            let y = mlx_rs::ops::conv2d(&xs, &w, (1, 1), (1, 1), (1, 1), 1)?.add(get("bias")?)?;
            Ok(y.reshape(&[b, gh * gw, -1])?)
        }
    }
}

/// `REPALoss.forward` given the teacher's tokens: `−mean(cos(student, teacher))`.
pub fn repa_loss(
    plan: &RepaPlan,
    p: &HashMap<Rc<str>, Array>,
    student_tokens: &Array,
    t: &RepaTargets,
) -> Result<Array> {
    let x = student_tokens.as_dtype(Dtype::Float32)?;
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
    let cos = sum_axes(&student.multiply(&teacher)?, &[-1], false)?;
    Ok(cos.mean(None)?.negative()?)
}

/// The per-sample rectified-flow loss `[B]` (`reduction="none"`, the holdout grid's form).
pub fn flow_loss_per_sample(dit: &IrisDiT, b: &StepBatch, obj: &FlowObjective) -> Result<Array> {
    per_sample_loss(dit, b, obj, &mut crate::dit::NoHooks)
}

fn per_sample_loss(
    dit: &IrisDiT,
    b: &StepBatch,
    obj: &FlowObjective,
    hooks: &mut dyn ForwardHooks,
) -> Result<Array> {
    let one = Array::from_f32(1.0);
    let x_t = one
        .subtract(&b.sigma)?
        .multiply(&b.x0)?
        .add(&b.sigma.multiply(&b.noise)?)?;
    let out = dit.forward_hooked(
        &x_t,
        &b.t,
        &TextBatch {
            states: &b.states,
            mask: &b.masks,
        },
        hooks,
    )?;
    let target = b.noise.subtract(&b.x0)?;
    let pred = match obj.prediction {
        Prediction::Velocity => out,
        Prediction::Clean => {
            let floor = Array::from_f32(obj.x_pred_sigma_min as f32);
            x_t.subtract(&out)?.divide(&maximum(&b.sigma, &floor)?)?
        }
    };
    let err = pred.subtract(&target)?.square()?;
    Ok(mean_axes(&err, &[1, 2, 3], false)?)
}

/// The training forward of one micro-batch: the flow loss (batch mean) and, under REPA, the
/// alignment loss — with the activation-checkpointing regions of `extras.ac`.
pub fn forward_losses(
    model: &TrainModel,
    p: &HashMap<Rc<str>, Array>,
    batch: &StepBatch,
    obj: &FlowObjective,
    extras: &TrainExtras,
    repa: Option<&RepaTargets>,
) -> Result<(Array, Option<Array>)> {
    let dit = model.dit(p)?;
    let mut hooks = TrainHooks {
        model,
        p,
        ac: extras.ac,
        capture: extras.repa.as_ref().map(|r| r.layer),
        captured: None,
    };
    let flow = per_sample_loss(&dit, batch, obj, &mut hooks)?.mean(None)?;
    let repa_l = match (&extras.repa, repa) {
        (Some(plan), Some(t)) => {
            let feats = hooks.captured.take().ok_or_else(|| {
                Error::Msg(format!("iris repa: block {} was never reached", plan.layer))
            })?;
            Some(repa_loss(plan, p, &feats, t)?)
        }
        (Some(_), None) => {
            return Err(Error::Msg(
                "iris repa: the micro-batch carries no teacher targets".into(),
            ))
        }
        _ => None,
    };
    Ok((flow, repa_l))
}

/// The losses of one micro-batch and the gradient of `(flow + weight · repa) · grad_scale` over
/// every trainable tensor (backbone or adapter factors, plus the REPA projector). Returns the
/// unscaled flow and REPA losses.
pub fn loss_and_grads_ext(
    model: &TrainModel,
    trainable: &Params,
    batch: &StepBatch,
    obj: &FlowObjective,
    extras: &TrainExtras,
    repa: Option<&RepaTargets>,
    grad_scale: f32,
) -> Result<(f32, Option<f32>, Params)> {
    let loss_fn = |p: HashMap<Rc<str>, Array>, _: i32| -> mlx_rs::error::Result<Vec<Array>> {
        let (flow, rl) = forward_losses(model, &p, batch, obj, extras, repa).map_err(to_exc)?;
        let total = match (&rl, &extras.repa) {
            (Some(r), Some(plan)) => flow.add(&r.multiply(Array::from_f32(plan.weight as f32))?)?,
            _ => flow.clone(),
        };
        let scaled = total.multiply(Array::from_f32(grad_scale))?;
        let mut out = vec![scaled, flow];
        if let Some(r) = rl {
            out.push(r);
        }
        Ok(out)
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (vals, grads) =
        vg(trainable.clone(), 0).map_err(|e| Error::Msg(format!("iris train step: {e}")))?;
    let flow = vals[1].clone();
    let rl = vals.get(2).cloned();
    let mut refs: Vec<&Array> = vec![&flow];
    refs.extend(rl.iter());
    refs.extend(grads.values());
    for chunk in refs.chunks(256) {
        eval(chunk.iter().copied())?;
    }
    let rl = rl.as_ref().map(super::scalar_f32).transpose()?;
    Ok((super::scalar_f32(&flow)?, rl, grads))
}

/// PyTorch-default init of the REPA projector (`U(±1/√fan_in)` weights and biases), from MLX's RNG
/// keyed on `mix_seed(seed, fnv(key))`.
pub fn init_projector(plan: &RepaPlan, seed: u64) -> Result<Params> {
    let mut out = Params::new();
    for (key, shape) in plan.projector_shapes() {
        let bound = plan.projector_init_bound(&key) as f32;
        let dims: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let k = random::key(mix_seed(&[seed, fnv(&key)]))?;
        let a = random::uniform::<_, f32>(-bound, bound, &dims[..], Some(&k))?;
        eval([&a])?;
        out.insert(Rc::from(key.as_str()), a);
    }
    Ok(out)
}
