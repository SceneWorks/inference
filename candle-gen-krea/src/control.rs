//! Krea 2 **pose-ControlNet control branch** — the sc-8460 spike (epic 8459).
//!
//! A ControlNet-style trainable side branch over the Krea 2 single-stream DiT: the first `N`
//! (default 7 of 28) single-stream blocks are **copied** into a fully trainable branch
//! ([`ControlBranch`]), driven by the same frozen pre-main front-end the base uses. The branch input
//! is the base's joint `[ctx; img]` sequence with the **control latent embedding** (the VAE-encoded
//! pose skeleton, patch-embedded through the *frozen* base `img_in` —
//! [`KreaTrainDit::embed_latent`]) added onto the image-token slice. After each branch block `i` a
//! **zero-initialized** output projection produces a per-block residual that is added (scaled by
//! `control_scale`) onto the frozen main stream's **image tokens** entering main block `i` — so at
//! step 0 the branch is an exact identity over the base model (the ControlNet zero-init property).
//!
//! Everything in the branch trains (attention, gate, SwiGLU, norms, modulation table, and the output
//! projections); the entire base (DiT + text encoder + VAE) stays frozen. The branch reuses the
//! composable (differentiable) ops of [`crate::train_dit`] — the fused `softmax_last_dim` /
//! `ops::rms_norm` kernels have no backward.
//!
//! **Dtype.** Branch matmul weights live in a configurable dtype (bf16 default — full f32 master
//! weights for a ~3B-param branch triple the optimizer footprint; see the trainer CLI's
//! `--branch-dtype`); the `+1`-folded RMSNorm scales stay f32 (tiny, and [`rms_scale_diff`] reduces
//! in f32 anyway). Weights are cast to the activation dtype inside the forward when they differ —
//! the cast is differentiable, so f32 master-weight training works through the same code path.
//!
//! **Checkpoint format (spike).** A flat `.safetensors` of the branch tensors keyed
//! `blocks.{i}.<leaf>` (norm scales stored **pre-folded** `scale + 1` under `*.weight_p1` — this
//! checkpoint is only read back by [`ControlBranch::from_checkpoint`], never by a diffusers loader),
//! plus `blocks.{i}.proj_out.weight` for the zero-init projections. `N` is inferred from the keys.

use std::collections::HashMap;
use std::path::Path;

use candle_gen::candle_core::{DType, Device, Tensor, Var};
use candle_gen::candle_nn::ops::sigmoid;
use candle_gen::{CandleError, Result};

use crate::config::Krea2Config;
use crate::loader::{rms_scale_weight, Weights};
use crate::train_dit::{repeat_kv, rms_scale_diff, sdpa_diff, KreaTrainDit, MainCtx};
use crate::transformer::rope::apply_interleaved_rope;

/// A trainable no-bias linear over a [`Var`] weight `[out, in]`; the weight is cast to the activation
/// dtype in the forward (differentiable — f32 master weights train through it).
struct VLin {
    w: Var,
}

impl VLin {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.w.as_tensor().to_dtype(x.dtype())?;
        Ok(x.broadcast_matmul(&w.t()?)?)
    }
}

/// Registers every created [`Var`] under its checkpoint key, in a deterministic order.
#[derive(Default)]
struct VarReg {
    named: Vec<(String, Var)>,
}

impl VarReg {
    fn var(&mut self, name: String, t: &Tensor) -> Result<Var> {
        let v = Var::from_tensor(t)?;
        self.named.push((name, v.clone()));
        Ok(v)
    }

    fn lin(&mut self, name: String, t: &Tensor) -> Result<VLin> {
        Ok(VLin {
            w: self.var(name, t)?,
        })
    }
}

/// The tensor source a branch is built from: base-block weights (fresh training) or a saved
/// checkpoint (resume / inference). Returns tensors already in their final storage dtype.
type Getter<'a> = dyn Fn(&str) -> Result<Tensor> + 'a;

/// Trainable twin of the base block's sigmoid-gated GQA attention — every projection is a [`Var`].
struct ControlAttention {
    q: VLin,
    k: VLin,
    v: VLin,
    gate: VLin,
    o: VLin,
    norm_q: Var, // f32, scale + 1
    norm_k: Var, // f32, scale + 1
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f64,
    scale: f64,
}

impl ControlAttention {
    fn build(reg: &mut VarReg, get: &Getter, prefix: &str, cfg: &Krea2Config) -> Result<Self> {
        let key = |leaf: &str| format!("{prefix}.{leaf}");
        Ok(Self {
            q: reg.lin(key("to_q.weight"), &get(&key("to_q.weight"))?)?,
            k: reg.lin(key("to_k.weight"), &get(&key("to_k.weight"))?)?,
            v: reg.lin(key("to_v.weight"), &get(&key("to_v.weight"))?)?,
            gate: reg.lin(key("to_gate.weight"), &get(&key("to_gate.weight"))?)?,
            o: reg.lin(key("to_out.0.weight"), &get(&key("to_out.0.weight"))?)?,
            norm_q: reg.var(key("norm_q.weight_p1"), &get(&key("norm_q.weight_p1"))?)?,
            norm_k: reg.var(key("norm_k.weight_p1"), &get(&key("norm_k.weight_p1"))?)?,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_kv_heads,
            head_dim: cfg.attention_head_dim,
            eps: cfg.norm_eps,
            scale: (cfg.attention_head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, x: &Tensor, rope: (&Tensor, &Tensor)) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let (nh, nkv, hd) = (self.heads, self.kv_heads, self.head_dim);

        let q = self.q.forward(x)?.reshape((b, s, nh, hd))?;
        let k = self.k.forward(x)?.reshape((b, s, nkv, hd))?;
        let v = self.v.forward(x)?.reshape((b, s, nkv, hd))?;
        let gate = self.gate.forward(x)?;

        let q = rms_scale_diff(&q, self.norm_q.as_tensor(), self.eps)?;
        let k = rms_scale_diff(&k, self.norm_k.as_tensor(), self.eps)?;
        let (cos, sin) = rope;
        let q = apply_interleaved_rope(&q, cos, sin)?;
        let k = apply_interleaved_rope(&k, cos, sin)?;

        let groups = nh / nkv;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let o = sdpa_diff(&q, &k, &v, self.scale)?;
        let o = o.transpose(1, 2)?.contiguous()?.reshape((b, s, nh * hd))?;

        let gated = (o * sigmoid(&gate)?)?;
        self.o.forward(&gated)
    }
}

/// One fully trainable branch block (a `Var`-backed copy of a base single-stream block) plus its
/// zero-initialized per-block output projection.
struct ControlBlock {
    sst: Var,      // [1, 1, 6·hidden], branch dtype
    prenorm: Var,  // f32, scale + 1
    postnorm: Var, // f32, scale + 1
    attn: ControlAttention,
    ff_gate: VLin,
    ff_up: VLin,
    ff_down: VLin,
    proj_out: VLin, // zero-init [hidden, hidden] — the ControlNet identity seam
    eps: f64,
}

impl ControlBlock {
    fn build(reg: &mut VarReg, get: &Getter, i: usize, cfg: &Krea2Config) -> Result<Self> {
        let key = |leaf: &str| format!("blocks.{i}.{leaf}");
        Ok(Self {
            sst: reg.var(key("scale_shift_table"), &get(&key("scale_shift_table"))?)?,
            prenorm: reg.var(key("norm1.weight_p1"), &get(&key("norm1.weight_p1"))?)?,
            postnorm: reg.var(key("norm2.weight_p1"), &get(&key("norm2.weight_p1"))?)?,
            attn: ControlAttention::build(reg, get, &key("attn"), cfg)?,
            ff_gate: reg.lin(key("ff.gate.weight"), &get(&key("ff.gate.weight"))?)?,
            ff_up: reg.lin(key("ff.up.weight"), &get(&key("ff.up.weight"))?)?,
            ff_down: reg.lin(key("ff.down.weight"), &get(&key("ff.down.weight"))?)?,
            proj_out: reg.lin(key("proj_out.weight"), &get(&key("proj_out.weight"))?)?,
            eps: cfg.norm_eps,
        })
    }

    /// Mirror of [`crate::train_dit`]'s `TrainBlock::forward` over the `Var`-backed weights.
    fn forward(&self, x: &Tensor, tvec: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        use candle_gen::candle_core::D;
        let sst = self.sst.as_tensor().to_dtype(tvec.dtype())?;
        let m = tvec.broadcast_add(&sst)?; // [b, 1, 6·hidden]
        let chunks = m.chunk(6, D::Minus1)?;
        let (prescale, preshift, pregate) = (&chunks[0], &chunks[1], &chunks[2]);
        let (postscale, postshift, postgate) = (&chunks[3], &chunks[4], &chunks[5]);

        let pre = rms_scale_diff(x, self.prenorm.as_tensor(), self.eps)?
            .broadcast_mul(&(prescale + 1.0)?)?
            .broadcast_add(preshift)?;
        let attn = self.attn.forward(&pre, (cos, sin))?;
        let x = (x + attn.broadcast_mul(pregate)?)?;

        let post = rms_scale_diff(&x, self.postnorm.as_tensor(), self.eps)?
            .broadcast_mul(&(postscale + 1.0)?)?
            .broadcast_add(postshift)?;
        let gated = (self.ff_gate.forward(&post)?.silu()? * self.ff_up.forward(&post)?)?;
        let mlp = self.ff_down.forward(&gated)?;
        Ok((&x + mlp.broadcast_mul(postgate)?)?)
    }
}

/// The trainable Krea 2 control branch: `N` `Var`-backed copies of the first `N` base single-stream
/// blocks + one zero-init output projection each. Built either fresh from the base weights
/// ([`from_base`](Self::from_base)) or from a saved spike checkpoint
/// ([`from_checkpoint`](Self::from_checkpoint)).
pub struct ControlBranch {
    blocks: Vec<ControlBlock>,
    named: Vec<(String, Var)>,
}

impl ControlBranch {
    /// Copy the first `n` base single-stream blocks (from the same mmap'd `transformer/` [`Weights`]
    /// the frozen DiT loads) into a trainable branch at `dtype`, with zero-init output projections.
    pub fn from_base(w: &Weights, cfg: &Krea2Config, n: usize, dtype: DType) -> Result<Self> {
        if n == 0 || n > cfg.num_layers {
            return Err(CandleError::Msg(format!(
                "control branch: n_blocks must be in 1..={} (got {n})",
                cfg.num_layers
            )));
        }
        let hidden = cfg.hidden_size;
        let device = w.device().clone();
        let get = move |key: &str| -> Result<Tensor> {
            // "blocks.{i}.<leaf>" -> the base "transformer_blocks.{i}.<leaf>" (or synthesized).
            let (i, leaf) = split_block_key(key)?;
            let base = format!("transformer_blocks.{i}");
            match leaf {
                "scale_shift_table" => Ok(w
                    .get(&format!("{base}.scale_shift_table"))?
                    .reshape((1, 1, 6 * hidden))?
                    .to_dtype(dtype)?),
                "norm1.weight_p1" => Ok(rms_scale_weight(w, &format!("{base}.norm1.weight"))?),
                "norm2.weight_p1" => Ok(rms_scale_weight(w, &format!("{base}.norm2.weight"))?),
                "attn.norm_q.weight_p1" => {
                    Ok(rms_scale_weight(w, &format!("{base}.attn.norm_q.weight"))?)
                }
                "attn.norm_k.weight_p1" => {
                    Ok(rms_scale_weight(w, &format!("{base}.attn.norm_k.weight"))?)
                }
                "proj_out.weight" => Ok(Tensor::zeros((hidden, hidden), dtype, &device)?),
                other => Ok(w.get(&format!("{base}.{other}"))?.to_dtype(dtype)?),
            }
        };
        Self::build(&get, cfg, n)
    }

    /// Load a branch back from a spike checkpoint (`.safetensors` written by [`save`](Self::save));
    /// `N` and dtypes are read from the file.
    pub fn from_checkpoint(path: &Path, cfg: &Krea2Config, device: &Device) -> Result<Self> {
        let tensors = candle_gen::candle_core::safetensors::load(path, device)?;
        let n = 1 + tensors
            .keys()
            .filter_map(|k| split_block_key(k).ok().map(|(i, _)| i))
            .max()
            .ok_or_else(|| {
                CandleError::Msg(format!(
                    "control branch: no blocks.{{i}}.* keys in {}",
                    path.display()
                ))
            })?;
        let get = |key: &str| -> Result<Tensor> {
            tensors.get(key).cloned().ok_or_else(|| {
                CandleError::Msg(format!(
                    "control branch: missing key {key} in {}",
                    path.display()
                ))
            })
        };
        Self::build(&get, cfg, n)
    }

    fn build(get: &Getter, cfg: &Krea2Config, n: usize) -> Result<Self> {
        let mut reg = VarReg::default();
        let blocks = (0..n)
            .map(|i| ControlBlock::build(&mut reg, get, i, cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            blocks,
            named: reg.named,
        })
    }

    /// Number of branch blocks (`N`).
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Every trainable `Var` (blocks + projections), in deterministic construction order — the
    /// optimizer parameter list.
    pub fn vars(&self) -> Vec<Var> {
        self.named.iter().map(|(_, v)| v.clone()).collect()
    }

    /// Trainable parameter count.
    pub fn num_params(&self) -> usize {
        self.named
            .iter()
            .map(|(_, v)| v.as_tensor().elem_count())
            .sum()
    }

    /// Write the branch to a flat `.safetensors` (the spike checkpoint [`from_checkpoint`] reads).
    pub fn save(&self, path: &Path) -> Result<()> {
        let map: HashMap<String, Tensor> = self
            .named
            .iter()
            .map(|(k, v)| (k.clone(), v.as_tensor().clone()))
            .collect();
        Ok(candle_gen::candle_core::safetensors::save(&map, path)?)
    }

    /// Run the branch over the base's joint sequence and return the per-block residuals
    /// `[b, img_len, hidden]` (one per branch block, for main blocks `0..N`). The control tokens are
    /// added onto the image-token slice of the branch **input**; each residual is the zero-init
    /// projection of the branch hidden state's image tokens after that branch block.
    fn residuals(
        &self,
        combined: &Tensor,
        ctrl_tokens: &Tensor,
        ctx: &MainCtx,
    ) -> Result<Vec<Tensor>> {
        let txt = combined.narrow(1, 0, ctx.cap_len)?;
        let img = (combined.narrow(1, ctx.cap_len, ctx.img_len)? + ctrl_tokens)?;
        let mut h = Tensor::cat(&[&txt, &img], 1)?;
        let mut out = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            h = blk.forward(&h, &ctx.tvec, &ctx.rcos, &ctx.rsin)?;
            let h_img = h.narrow(1, ctx.cap_len, ctx.img_len)?;
            out.push(blk.proj_out.forward(&h_img)?);
        }
        Ok(out)
    }
}

/// `"blocks.{i}.<leaf>"` → `(i, leaf)`.
fn split_block_key(key: &str) -> Result<(usize, &str)> {
    let rest = key
        .strip_prefix("blocks.")
        .ok_or_else(|| CandleError::Msg(format!("control branch: unexpected key {key}")))?;
    let (idx, leaf) = rest
        .split_once('.')
        .ok_or_else(|| CandleError::Msg(format!("control branch: unexpected key {key}")))?;
    let i: usize = idx
        .parse()
        .map_err(|_| CandleError::Msg(format!("control branch: unexpected key {key}")))?;
    Ok((i, leaf))
}

/// Velocity prediction **with** the control branch: the frozen base forward with the branch's
/// per-block residuals (scaled by `control_scale`) added onto the main stream's image tokens
/// entering main blocks `0..N`.
///
/// `control_scale == 0.0` short-circuits to the plain base forward — **byte-identical** to an
/// un-branched generation (the branch is never run), which is the spike's identity contract.
///
/// `ctrl_latent` is the VAE-encoded pose skeleton `[b, 16, H/8, W/8]` (the same normalized latent
/// space as the noisy latent); it is patch-embedded through the frozen base `img_in`.
pub fn forward_with_control(
    dit: &KreaTrainDit,
    branch: &ControlBranch,
    latent: &Tensor,
    timestep: &Tensor,
    context: &Tensor,
    ctrl_latent: &Tensor,
    control_scale: f64,
) -> Result<Tensor> {
    if control_scale == 0.0 {
        return Ok(dit.forward(latent, timestep, context)?);
    }
    let (combined, ctx) = dit.forward_pre_main(latent, timestep, context)?;
    let ctrl_tokens = dit.embed_latent(ctrl_latent)?;
    let residuals = branch.residuals(&combined, &ctrl_tokens, &ctx)?;

    let mut x = combined;
    for (i, blk) in dit.blocks().iter().enumerate() {
        if let Some(r) = residuals.get(i) {
            let scaled = (r * control_scale)?.to_dtype(x.dtype())?;
            let txt = x.narrow(1, 0, ctx.cap_len)?;
            let img = (x.narrow(1, ctx.cap_len, ctx.img_len)? + scaled)?;
            x = Tensor::cat(&[&txt, &img], 1)?;
        }
        x = blk.forward(&x, &ctx.tvec, &ctx.rcos, &ctx.rsin)?;
    }
    Ok(dit.velocity_out(&x, &ctx)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testfix::{tiny_batch, tiny_dit};
    use candle_gen::candle_core::Device;
    use candle_gen::train::flow_match::velocity_loss;
    use candle_gen::train::optim::{clip_grad_norm, TrainOptimizer};

    fn nudge_vars(branch: &ControlBranch, dev: &Device) {
        for v in branch.vars() {
            v.set(&Tensor::randn(0f32, 0.02f32, v.as_tensor().dims(), dev).unwrap())
                .unwrap();
        }
    }

    /// The keystone zero-init gate: with untouched (zero) output projections the branched forward is
    /// **exactly** the base forward at `control_scale = 1.0` — the ControlNet step-0 identity.
    #[test]
    fn zero_init_branch_is_identity() {
        let dev = Device::Cpu;
        let (dit, c, path) = tiny_dit();
        let w = Weights::from_file(&path, &dev, DType::F32).unwrap();
        let branch = ControlBranch::from_base(&w, &c, 1, DType::F32).unwrap();

        let (x0, cap, _) = tiny_batch(&c);
        let ctrl = Tensor::randn(0f32, 1f32, x0.dims(), &dev).unwrap();
        let ctxt = cap.unsqueeze(0).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();

        let base = dit.forward(&x0, &t, &ctxt).unwrap();
        let with = forward_with_control(&dit, &branch, &x0, &t, &ctxt, &ctrl, 1.0).unwrap();
        let a = base.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = with.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b, "zero-init projections must be a step-0 identity");
        let _ = std::fs::remove_file(path);
    }

    /// `control_scale = 0` short-circuits to the plain base forward even with a trained (nonzero)
    /// branch — the byte-identity contract of the inference hook.
    #[test]
    fn scale_zero_is_base_forward() {
        let dev = Device::Cpu;
        let (dit, c, path) = tiny_dit();
        let w = Weights::from_file(&path, &dev, DType::F32).unwrap();
        let branch = ControlBranch::from_base(&w, &c, 1, DType::F32).unwrap();
        nudge_vars(&branch, &dev);

        let (x0, cap, _) = tiny_batch(&c);
        let ctrl = Tensor::randn(0f32, 1f32, x0.dims(), &dev).unwrap();
        let ctxt = cap.unsqueeze(0).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();

        let base = dit.forward(&x0, &t, &ctxt).unwrap();
        let with = forward_with_control(&dit, &branch, &x0, &t, &ctxt, &ctrl, 0.0).unwrap();
        let a = base.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = with.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a, b, "scale 0 must be the un-branched forward");
        let _ = std::fs::remove_file(path);
    }

    /// Backprop reaches **every** branch `Var` (through the frozen main stack into the injected
    /// residuals) with finite gradients, and a few AdamW steps lower the loss on a fixed batch.
    #[test]
    fn backward_reaches_branch_and_descends() {
        let dev = Device::Cpu;
        let (dit, c, path) = tiny_dit();
        let w = Weights::from_file(&path, &dev, DType::F32).unwrap();
        let branch = ControlBranch::from_base(&w, &c, 1, DType::F32).unwrap();
        nudge_vars(&branch, &dev);
        let vars = branch.vars();

        let (x0, cap, noise) = tiny_batch(&c);
        let ctrl = Tensor::randn(0f32, 1f32, x0.dims(), &dev).unwrap();
        let ctxt = cap.unsqueeze(0).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();
        let x_t = ((&x0 * 0.5).unwrap() + (&noise * 0.5).unwrap()).unwrap();
        let target = (&noise - &x0).unwrap();

        let loss_of = |()| -> (f32, candle_gen::candle_core::backprop::GradStore) {
            let v = forward_with_control(&dit, &branch, &x_t, &t, &ctxt, &ctrl, 1.0).unwrap();
            let loss = velocity_loss(&v, &target, false).unwrap();
            let lv = loss.to_vec0::<f32>().unwrap();
            (lv, loss.backward().unwrap())
        };

        let (loss0, grads) = loss_of(());
        assert!(loss0.is_finite());
        for (i, v) in vars.iter().enumerate() {
            let g = grads
                .get(v.as_tensor())
                .unwrap_or_else(|| panic!("branch var {i} has no gradient"));
            assert!(
                g.flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap()
                    .iter()
                    .all(|x| x.is_finite()),
                "branch var {i} gradient has non-finite entries"
            );
        }

        let mut opt = TrainOptimizer::from_config("adamw", vars.clone(), 1e-2, 0.0).unwrap();
        let mut grads = grads;
        for _ in 0..6 {
            clip_grad_norm(&mut grads, &vars, 1.0).unwrap();
            opt.step(&grads).unwrap();
            let (_l, g) = loss_of(());
            grads = g;
        }
        let (loss1, _) = loss_of(());
        assert!(
            loss1 < loss0,
            "AdamW steps on a fixed batch should lower the loss: {loss0} -> {loss1}"
        );
        let _ = std::fs::remove_file(path);
    }

    /// Save → load roundtrip: the reloaded branch (N inferred from the file) produces the same
    /// branched forward as the original.
    #[test]
    fn checkpoint_roundtrip() {
        let dev = Device::Cpu;
        let (dit, c, path) = tiny_dit();
        let w = Weights::from_file(&path, &dev, DType::F32).unwrap();
        let branch = ControlBranch::from_base(&w, &c, 1, DType::F32).unwrap();
        nudge_vars(&branch, &dev);

        let ckpt =
            std::env::temp_dir().join(format!("krea_ctrl_ckpt_{}.safetensors", std::process::id()));
        branch.save(&ckpt).unwrap();
        let loaded = ControlBranch::from_checkpoint(&ckpt, &c, &dev).unwrap();
        assert_eq!(loaded.num_blocks(), 1);
        assert_eq!(loaded.num_params(), branch.num_params());

        let (x0, cap, _) = tiny_batch(&c);
        let ctrl = Tensor::randn(0f32, 1f32, x0.dims(), &dev).unwrap();
        let ctxt = cap.unsqueeze(0).unwrap();
        let t = Tensor::from_vec(vec![0.5f32], (1,), &dev).unwrap();
        let a = forward_with_control(&dit, &branch, &x0, &t, &ctxt, &ctrl, 1.0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let b = forward_with_control(&dit, &loaded, &x0, &t, &ctxt, &ctrl, 1.0)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(a, b, "checkpoint roundtrip must reproduce the forward");
        let _ = std::fs::remove_file(ckpt);
        let _ = std::fs::remove_file(path);
    }
}
