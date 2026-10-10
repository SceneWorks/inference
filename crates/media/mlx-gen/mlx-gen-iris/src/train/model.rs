//! The trainable Iris backbone: which tensors are trained (every backbone tensor, or LoRA/LoKr
//! factors over frozen Linears), how a step's weights are assembled from them, the rectified-flow
//! loss and its gradient, random init, and the artifact writers.
//!
//! The forward is **the inference forward**: every step rebuilds an [`IrisDiT`] with
//! [`IrisDiT::from_weights`] from the (traced) trainable tensors inside the autograd trace, exactly
//! as the provider loads a checkpoint, so training and inference can never disagree about the
//! graph. Mixed precision is the provider's autocast policy ([`crate::nn`]): f32 master tensors,
//! matmuls/attention in the compute dtype (bf16 under `mixed_precision: bf16`), norms/residuals/loss
//! in f32; the casts are traced, so gradients reach the f32 masters.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::rc::Rc;

use gen_core::iris::train::{
    backbone_tensor_shapes, init_kind, mix_seed, FlowObjective, InitKind, Prediction,
};
use gen_core::iris::ModelConfig;
use mlx_gen::adapters::reconstruct_lokr_delta;
use mlx_gen::gen_core;
use mlx_gen::train::lora::factorization;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::error::Exception;
use mlx_rs::ops::{matmul, maximum, mean_axes};
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use super::optim::Params;
use crate::dit::{IrisDiT, TextBatch};

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

fn fnv(key: &str) -> u64 {
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

/// The LoRA delta `(alpha / rank) · B · A` (f32).
pub fn lora_delta(a: &Array, b: &Array, alpha: f32, rank: usize) -> Result<Array> {
    Ok(matmul(b, a)?.multiply(Array::from_f32(alpha / rank as f32))?)
}

/// The LoKr delta `(alpha / rank) · kron(w1, w2)` at `[out, in]` (f32).
pub fn lokr_delta(
    factors: &HashMap<&str, &Array>,
    out_f: i32,
    in_f: i32,
    alpha: f32,
    rank: usize,
) -> Result<Array> {
    reconstruct_lokr_delta(
        alpha,
        rank as f32,
        &[out_f, in_f],
        factors.get("lokr_w1").copied(),
        None,
        None,
        factors.get("lokr_w2").copied(),
        factors.get("lokr_w2_a").copied(),
        factors.get("lokr_w2_b").copied(),
        Dtype::Float32,
    )
}

impl TrainModel {
    /// The complete backbone tensor map of one step (for [`IrisDiT::from_weights`]).
    pub fn weights_map(
        &self,
        trainable: &HashMap<Rc<str>, Array>,
    ) -> Result<HashMap<String, Array>> {
        match &self.trainable {
            Trainable::Full => Ok(trainable
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect()),
            Trainable::Adapter {
                kind,
                targets,
                base,
            } => {
                let mut map = base.clone();
                for t in targets {
                    let w = map.get(&format!("{}.weight", t.path)).ok_or_else(|| {
                        Error::Msg(format!("iris adapter: base has no {}.weight", t.path))
                    })?;
                    let delta = self.target_delta(*kind, t, trainable)?;
                    let merged = w.add(&delta.as_dtype(w.dtype())?)?;
                    map.insert(format!("{}.weight", t.path), merged);
                }
                Ok(map)
            }
        }
    }

    fn target_delta(
        &self,
        kind: AdapterKind,
        t: &AdapterTarget,
        p: &HashMap<Rc<str>, Array>,
    ) -> Result<Array> {
        let get = |k: &str| -> Result<&Array> {
            p.get(k)
                .ok_or_else(|| Error::Msg(format!("iris adapter: factor {k} missing")))
        };
        match kind {
            AdapterKind::Lora { rank, alpha } => lora_delta(
                get(&format!("{}.lora_A.weight", t.path))?,
                get(&format!("{}.lora_B.weight", t.path))?,
                alpha,
                rank,
            ),
            AdapterKind::Lokr { rank, alpha, .. } => {
                let mut f: HashMap<&str, &Array> = HashMap::new();
                for name in ["lokr_w1", "lokr_w2", "lokr_w2_a", "lokr_w2_b"] {
                    if let Some(a) = p.get(format!("{}.{name}", t.path).as_str()) {
                        f.insert(name, a);
                    }
                }
                lokr_delta(&f, t.out_f, t.in_f, alpha, rank)
            }
        }
    }

    /// The backbone of one step at the compute dtype.
    pub fn dit(&self, trainable: &HashMap<Rc<str>, Array>) -> Result<IrisDiT> {
        let w = Weights::from_map(self.weights_map(trainable)?);
        IrisDiT::from_weights(&w, &self.cfg, self.compute)
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
        Prediction::V => out,
        Prediction::X => {
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
    Ok((loss.item::<f32>(), grads))
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

/// Merge an Iris adapter artifact into a backbone tensor map in place at `strength`
/// (`W ← W + strength · ΔW`, ΔW per [`gen_core::iris::train::AdapterMetadata`]'s schema). The
/// merged matrix keeps the base tensor's dtype. Returns the adapted module paths.
pub fn merge_adapter_file(
    base: &mut HashMap<String, Array>,
    path: &Path,
    strength: f32,
) -> Result<Vec<String>> {
    let (tensors, meta) = Array::load_safetensors_with_metadata(path)
        .map_err(|e| Error::Msg(format!("iris adapter: read {}: {e}", path.display())))?;
    let meta: BTreeMap<String, String> = meta.into_iter().collect();
    let m = gen_core::iris::train::AdapterMetadata::from_map(&meta)?;
    for target in &m.targets {
        let wkey = format!("{target}.weight");
        let w = base
            .get(&wkey)
            .ok_or_else(|| Error::Msg(format!("iris adapter: backbone has no {wkey}")))?;
        let (out_f, in_f) = (w.shape()[0], w.shape()[1]);
        let delta = match m.network_type.as_str() {
            "lora" => {
                let a = tensors.get(&format!("{target}.lora_A.weight"));
                let b = tensors.get(&format!("{target}.lora_B.weight"));
                match (a, b) {
                    (Some(a), Some(b)) => lora_delta(a, b, m.alpha, m.rank)?,
                    _ => {
                        return Err(Error::Msg(format!(
                            "iris adapter: {target} lacks its lora_A/lora_B factors"
                        )))
                    }
                }
            }
            _ => {
                let mut f: HashMap<&str, &Array> = HashMap::new();
                for name in ["lokr_w1", "lokr_w2", "lokr_w2_a", "lokr_w2_b"] {
                    if let Some(a) = tensors.get(&format!("{target}.{name}")) {
                        f.insert(name, a);
                    }
                }
                lokr_delta(&f, out_f, in_f, m.alpha, m.rank)?
            }
        };
        let merged = w
            .as_dtype(Dtype::Float32)?
            .add(&delta.multiply(Array::from_f32(strength))?)?
            .as_dtype(w.dtype())?;
        eval([&merged])?;
        base.insert(wkey, merged);
    }
    Ok(m.targets)
}
