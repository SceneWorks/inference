//! sc-3042 SPIKE — LoRA *training* on the Z-Image DiT, in pure Rust on mlx-rs.
//!
//! This is the gate for epic 3039 (LoRA/LoKr training in Rust). It proves the mechanism the whole
//! epic rests on, on the REAL 30-block Z-Image transformer:
//!
//!   * **Trainable LoRA injection** — the model crates do NOT use mlx-rs's `Module`/`ModuleParameters`
//!     system (hand-rolled `&self` forwards over raw `Array`s, `src/adapters.rs:6`), so training uses
//!     the *functional* autograd: the trainable factors live OUTSIDE the model in a
//!     `HashMap<Rc<str>, Array>`, and each step they are re-injected into the target [`AdaptableLinear`]s
//!     as a single `Adapter::Lora` via [`AdaptableLinear::set_adapters`]. The injection mirrors the
//!     inference reload (`adapters::loader::install_lora_groups`) op-for-op — transpose the
//!     `[r,in]`/`[out,r]` factors, fold `alpha/rank` into `b`, `scale = 1` — so the trained adapter
//!     round-trips through the normal inference path bit-for-bit.
//!   * **Autograd + optimizer** — `keyed_value_and_grad` over the factor map + `AdamW::update_single`
//!     per parameter + `clip_grad_norm` (proven in `tests/lora_train_probe.rs`).
//!   * **Flow-match velocity target** — the Z-Image `forward()` already negates its raw output
//!     (`transformer.rs:246`) and the denoise loop integrates it as `latents += dσ·v` with
//!     `timestep = 1-σ`, so the regression target for `forward()` is `noise - latents` (the *raw*
//!     diffusers output trains toward `latents - noise`; the negation flips the sign — see
//!     `SceneWorks training_adapters.py:485` `flow_matching_velocity_target`).
//!   * **safetensors out** — PEFT keys `{path}.lora_A.weight` `[r,in]`, `{path}.lora_B.weight`
//!     `[out,r]`, `{path}.alpha`, reloadable by `apply_z_image_adapters`.
//!
//! sc-3043 generalizes this into a reusable `Trainer` surface (dataset/VAE-cache/bucket, checkpoint,
//! LR schedule, the `lora_train` job); sc-3044 hardens it for Z-Image + adds LoKr.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::Result;
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::{multiply, subtract};
use mlx_rs::optimizers::{clip_grad_norm, AdamW, Optimizer};
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array};

use crate::transformer::ZImageTransformer;

/// One LoRA-trained Linear: its dotted module path (e.g. `layers.0.attention.to_q`) plus the
/// pre-built parameter-map keys and the `[out, in]` dims read off the base weight.
pub struct LoraTarget {
    pub path: String,
    a_key: Rc<str>,
    b_key: Rc<str>,
    pub in_f: i32,
    pub out_f: i32,
}

/// The default Z-Image attention LoRA targets across the main `layers` stack — the suffixes
/// `to_q`/`to_k`/`to_v`/`to_out.0` the SceneWorks torch trainer uses
/// (`DEFAULT_LORA_TARGET_MODULES`, `training_adapters.py:72`).
pub fn attention_targets(n_layers: usize) -> Vec<String> {
    let mut out = Vec::with_capacity(n_layers * 4);
    for i in 0..n_layers {
        for proj in [
            "attention.to_q",
            "attention.to_k",
            "attention.to_v",
            "attention.to_out.0",
        ] {
            out.push(format!("layers.{i}.{proj}"));
        }
    }
    out
}

/// A minimal Z-Image LoRA trainer: a frozen base transformer + an external trainable factor map,
/// stepped with `keyed_value_and_grad` + AdamW. Spike-scoped (single in-memory sample at a time);
/// the dataset/scheduling glue is sc-3043.
pub struct ZImageLoraTrainer {
    transformer: ZImageTransformer,
    targets: Vec<LoraTarget>,
    params: HashMap<Rc<str>, Array>,
    alpha: f32,
    opt: AdamW,
}

impl ZImageLoraTrainer {
    /// Build a trainer over `target_paths` (dotted module paths into the DiT). LoRA factors are
    /// initialised the Python `_MlxLoRALinear` way — `A ~ N(0, 0.02)` `[rank, in]`, `B = 0`
    /// `[out, rank]` — so the adapter starts as an exact no-op and only learns from the gradient.
    pub fn new(
        transformer: ZImageTransformer,
        target_paths: &[String],
        rank: i32,
        alpha: f32,
        lr: f32,
        seed: u64,
    ) -> Result<Self> {
        let mut transformer = transformer;
        let mut targets = Vec::with_capacity(target_paths.len());
        let mut params: HashMap<Rc<str>, Array> = HashMap::new();
        for (i, path) in target_paths.iter().enumerate() {
            let segs: Vec<&str> = path.split('.').collect();
            let lin = AdaptableHost::adaptable_mut(&mut transformer, &segs).ok_or_else(
                || -> mlx_gen::Error {
                    format!("LoRA target does not resolve on the Z-Image DiT: {path}").into()
                },
            )?;
            let shape = lin.base_shape(); // [out, in]
            let (out_f, in_f) = (shape[0], shape[1]);

            let a_key: Rc<str> = Rc::from(format!("{path}.lora_a"));
            let b_key: Rc<str> = Rc::from(format!("{path}.lora_b"));
            // Distinct subkeys per target so the RNG init differs per layer.
            let ka = random::key(seed.wrapping_add(2 * i as u64 + 1))?;
            let a = multiply(
                &random::normal::<f32>(&[rank, in_f], None, None, Some(&ka))?,
                Array::from_slice(&[0.02f32], &[1]),
            )?;
            let b = Array::zeros::<f32>(&[out_f, rank])?;
            eval([&a, &b])?;
            params.insert(a_key.clone(), a);
            params.insert(b_key.clone(), b);
            targets.push(LoraTarget {
                path: path.clone(),
                a_key,
                b_key,
                in_f,
                out_f,
            });
        }
        Ok(Self {
            transformer,
            targets,
            params,
            alpha,
            opt: AdamW::new(lr),
        })
    }

    pub fn num_targets(&self) -> usize {
        self.targets.len()
    }

    /// Overwrite the optimizer learning rate (LR schedules mutate this between steps — mlx-rs has no
    /// built-in scheduler).
    pub fn set_lr(&mut self, lr: f32) {
        self.opt.lr = Array::from_slice(&[lr], &[1]);
    }

    /// One optimizer step on a single `(clean_latent, cap_feats)` sample at flow-match `sigma`.
    /// `x_t = (1-σ)·x0 + σ·noise`, target `= noise - x0`, `timestep = 1-σ`. Returns the scalar loss.
    pub fn train_step(
        &mut self,
        x0: &Array,
        cap_feats: &Array,
        sigma: f32,
        noise: &Array,
    ) -> Result<f32> {
        let (x_t, target, timestep) = build_batch(x0, noise, sigma)?;
        let params_now = self.params.clone();

        // Disjoint field borrows: the loss closure needs `&mut transformer` AND `&targets` at once.
        let transformer = &mut self.transformer;
        let targets: &[LoraTarget] = &self.targets;
        let alpha = self.alpha;
        let capf = cap_feats.clone();

        let (grads, loss) = {
            let loss_fn = move |p: HashMap<Rc<str>, Array>, _: i32| -> MlxResult<Vec<Array>> {
                install_training_lora(transformer, &p, targets, alpha)?;
                let v = transformer
                    .forward(&x_t, timestep, &capf)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                // MSE — `mean(None)` reduces to a 0-d scalar (grad requires a scalar cotangent).
                Ok(vec![subtract(&v, &target)?.square()?.mean(None)?])
            };
            let mut vg = keyed_value_and_grad(loss_fn);
            let (val, grads) = vg(params_now, 0)?;
            (grads, val[0].item::<f32>())
        };

        // Global-norm clip then AdamW per parameter.
        let (clipped, _norm) = clip_grad_norm(&grads, 1.0)?;
        for (k, g) in clipped.iter() {
            let mut param = self.params[k].clone();
            self.opt.update_single(k, g.as_ref(), &mut param)?;
            self.params.insert(k.clone(), param);
        }
        eval(self.params.values())?;
        Ok(loss)
    }

    /// The flow-match loss at `sigma` for the CURRENT adapter state, with no gradient — the
    /// verification probe (base-vs-trained, and round-trip). `with_adapter=false` evaluates the bare
    /// frozen base (adapters cleared) to measure the LoRA's effect.
    pub fn eval_loss(
        &mut self,
        x0: &Array,
        cap_feats: &Array,
        sigma: f32,
        noise: &Array,
        with_adapter: bool,
    ) -> Result<f32> {
        let (x_t, target, timestep) = build_batch(x0, noise, sigma)?;
        if with_adapter {
            let params = self.params.clone();
            install_training_lora(&mut self.transformer, &params, &self.targets, self.alpha)?;
        } else {
            clear_lora(&mut self.transformer, &self.targets);
        }
        let v = self.transformer.forward(&x_t, timestep, cap_feats)?;
        let loss = subtract(&v, &target)?.square()?.mean(None)?;
        Ok(loss.item::<f32>())
    }

    /// Round-trip proof: clear the trainable injection, reload `adapter_path` through the REAL
    /// inference path ([`crate::apply_z_image_adapters`]) onto this same frozen base, and re-measure
    /// the flow-match loss at `sigma`. Should reproduce [`eval_loss`](Self::eval_loss)`(…, true)`
    /// bit-for-bit — the trainer injects the SAME `(transpose, alpha/rank fold, scale=1)` the loader
    /// applies. Restores the cleared state afterwards (a bare base). Uses one transformer (no second
    /// multi-GB load).
    pub fn roundtrip_eval(
        &mut self,
        adapter_path: impl AsRef<Path>,
        x0: &Array,
        cap_feats: &Array,
        sigma: f32,
        noise: &Array,
    ) -> Result<f32> {
        clear_lora(&mut self.transformer, &self.targets);
        let spec = mlx_gen::AdapterSpec {
            path: adapter_path.as_ref().to_path_buf(),
            scale: 1.0,
            kind: mlx_gen::AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        };
        crate::apply_z_image_adapters(&mut self.transformer, std::slice::from_ref(&spec))?;
        let (x_t, target, timestep) = build_batch(x0, noise, sigma)?;
        let v = self.transformer.forward(&x_t, timestep, cap_feats)?;
        let loss = subtract(&v, &target)?.square()?.mean(None)?;
        clear_lora(&mut self.transformer, &self.targets);
        Ok(loss.item::<f32>())
    }

    /// Write the trained adapter as PEFT-format safetensors — `{path}.lora_A.weight` `[r,in]`,
    /// `{path}.lora_B.weight` `[out,r]`, scalar `{path}.alpha` — reloadable by
    /// [`crate::apply_z_image_adapters`]. Metadata records the network type/rank/alpha (the epic-2193
    /// reload contract; LoKr will also need `decomposeFactor`).
    pub fn save_peft(&self, path: impl AsRef<Path>, rank: i32) -> Result<()> {
        let mut entries: Vec<(String, &Array)> = Vec::with_capacity(self.targets.len() * 3);
        let mut alphas: Vec<(String, Array)> = Vec::with_capacity(self.targets.len());
        for t in &self.targets {
            entries.push((format!("{}.lora_A.weight", t.path), &self.params[&t.a_key]));
            entries.push((format!("{}.lora_B.weight", t.path), &self.params[&t.b_key]));
            alphas.push((
                format!("{}.alpha", t.path),
                Array::from_slice(&[self.alpha], &[1]),
            ));
        }
        let mut all: Vec<(String, &Array)> = entries;
        for (k, v) in &alphas {
            all.push((k.clone(), v));
        }
        let mut meta: HashMap<String, String> = HashMap::new();
        meta.insert("networkType".to_string(), "lora".to_string());
        meta.insert("rank".to_string(), rank.to_string());
        meta.insert("alpha".to_string(), self.alpha.to_string());
        Array::save_safetensors(all, Some(&meta), path)?;
        Ok(())
    }
}

/// `(x_t, target, timestep)` for a single sample at flow-match `sigma`:
/// `x_t = (1-σ)·x0 + σ·noise`, `target = noise - x0`, `timestep = 1-σ`.
fn build_batch(x0: &Array, noise: &Array, sigma: f32) -> Result<(Array, Array, f32)> {
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    let x_t = mlx_rs::ops::add(&multiply(x0, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, x0)?; // velocity for the already-negated forward output
    Ok((x_t, target, 1.0 - sigma))
}

/// Inject the current trainable factors as one `Adapter::Lora` per target — EXACTLY as the inference
/// reload (`install_lora_groups`): transpose `[r,in]`→`[in,r]` and `[out,r]`→`[r,out]`, fold
/// `alpha/rank` into `b`, `scale = 1`. Differentiable (the transposes/fold are traced).
fn install_training_lora(
    transformer: &mut ZImageTransformer,
    params: &HashMap<Rc<str>, Array>,
    targets: &[LoraTarget],
    alpha: f32,
) -> MlxResult<()> {
    for t in targets {
        let a = params[&t.a_key].t(); // [r,in] -> [in,r]
        let b_t = params[&t.b_key].t(); // [out,r] -> [r,out]
        let rank = a.shape()[1] as f32;
        let b = b_t.multiply(Array::from_slice(&[alpha / rank], &[1]))?;
        let segs: Vec<&str> = t.path.split('.').collect();
        let lin = AdaptableHost::adaptable_mut(transformer, &segs)
            .ok_or_else(|| Exception::custom(format!("LoRA target not found: {}", t.path)))?;
        lin.set_adapters(vec![Adapter::Lora { a, b, scale: 1.0 }]);
    }
    Ok(())
}

/// Clear every target's adapter stack (back to the bare frozen base).
fn clear_lora(transformer: &mut ZImageTransformer, targets: &[LoraTarget]) {
    for t in targets {
        let segs: Vec<&str> = t.path.split('.').collect();
        if let Some(lin) = AdaptableHost::adaptable_mut(transformer, &segs) {
            lin.set_adapters(Vec::new());
        }
    }
}
