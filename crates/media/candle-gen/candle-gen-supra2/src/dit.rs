//! `SupraDiT` on Candle — the same faithful port of the frozen upstream `inference.py` model classes
//! as `mlx-gen-supra2::dit`, op for op.
//!
//! Patchify → `x_embed` + learned `pos_embed`; sinusoidal timestep features (host-side, shared with
//! MLX through [`gen_core::supra2::timestep_features`]) → `t_embed` MLP; `ctx_proj` of the Flan-T5
//! context; `depth` × [AdaLN-Zero self-attention, **unmodulated** cross-attention masked by the text
//! attention mask, AdaLN-Zero MLP]; AdaLN final layer; unpatchify. Runs in f32 (the checkpoint's
//! dtype and upstream's CPU/MPS execution dtype). Weight keys are upstream's `state_dict` names
//! verbatim.
//!
//! [`gen_core::supra2::timestep_features`]: candle_gen::gen_core::supra2::timestep_features

use std::collections::HashMap;

use candle_gen::candle_core::{DType, Device, Tensor, D};
use candle_gen::candle_nn::{ops::softmax_last_dim, Linear, Module};
use candle_gen::gen_core::supra2::{timestep_features, Supra2Config};
use candle_gen::{CandleError as Error, Result};

const LN_EPS: f64 = 1e-6;

fn lin(w: &HashMap<String, Tensor>, prefix: &str, shape: [usize; 2]) -> Result<Linear> {
    let get = |name: &str| -> Result<Tensor> {
        w.get(name)
            .cloned()
            .ok_or_else(|| Error::Msg(format!("supra2: missing denoiser tensor `{name}`")))
    };
    let weight = get(&format!("{prefix}.weight"))?;
    let bias = get(&format!("{prefix}.bias"))?;
    if weight.dims() != shape || bias.dims() != [shape[0]] {
        return Err(Error::Msg(format!(
            "supra2: `{prefix}` has shape {:?}/{:?}, expected {shape:?}",
            weight.dims(),
            bias.dims()
        )));
    }
    Ok(Linear::new(weight, Some(bias)))
}

/// `x * (1 + scale[:, None]) + shift[:, None]`.
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let one = (scale.unsqueeze(1)? + 1.0)?;
    Ok(x.broadcast_mul(&one)?.broadcast_add(&shift.unsqueeze(1)?)?)
}

/// `F.layer_norm(x, (D,), eps=1e-6)` without affine parameters (`elementwise_affine=False`).
fn norm(x: &Tensor) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(centered.broadcast_div(&(var + LN_EPS)?.sqrt()?)?)
}

/// `[B, N, H·D]` → `[B, H, N, D]` (contiguous).
fn heads(x: &Tensor, h: usize) -> Result<Tensor> {
    let (b, n, c) = x.dims3()?;
    Ok(x.reshape((b, n, h, c / h))?.transpose(1, 2)?.contiguous()?)
}

/// `[B, H, N, D]` → `[B, N, H·D]`.
fn merge(x: &Tensor) -> Result<Tensor> {
    let (b, h, n, d) = x.dims4()?;
    Ok(x.transpose(1, 2)?.reshape((b, n, h * d))?)
}

/// `F.scaled_dot_product_attention` in f32: `softmax(q kᵀ / √d + mask) v`. `mask` is additive
/// `[B, 1, 1, M]` (`0` = attend, `-inf` = masked), broadcast over heads and queries.
fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let d = q.dim(D::Minus1)?;
    let scale = 1.0 / (d as f64).sqrt();
    let scores = (q.matmul(&k.t()?.contiguous()?)? * scale)?;
    let scores = match mask {
        Some(m) => scores.broadcast_add(m)?,
        None => scores,
    };
    Ok(softmax_last_dim(&scores)?.matmul(v)?)
}

struct Block {
    qkv: Linear,
    self_proj: Linear,
    q: Linear,
    kv: Linear,
    cross_proj: Linear,
    mlp0: Linear,
    mlp2: Linear,
    adaln: Linear,
    heads: usize,
}

impl Block {
    fn from_weights(w: &HashMap<String, Tensor>, i: usize, c: &Supra2Config) -> Result<Self> {
        let d = c.d_model;
        let p = |s: &str| format!("blocks.{i}.{s}");
        Ok(Self {
            qkv: lin(w, &p("self_attn.qkv"), [3 * d, d])?,
            self_proj: lin(w, &p("self_attn.proj"), [d, d])?,
            q: lin(w, &p("cross_attn.q"), [d, d])?,
            kv: lin(w, &p("cross_attn.kv"), [2 * d, d])?,
            cross_proj: lin(w, &p("cross_attn.proj"), [d, d])?,
            mlp0: lin(w, &p("mlp.0"), [c.mlp_hidden, d])?,
            mlp2: lin(w, &p("mlp.2"), [d, c.mlp_hidden])?,
            adaln: lin(w, &p("adaln.1"), [6 * d, d])?,
            heads: c.num_heads,
        })
    }

    /// `x`: `[B, N, D]`; `c`: `[B, D]` conditioning; `ctx`: `[B, M, D]` projected context;
    /// `mask`: additive `[B, 1, 1, M]`.
    fn forward(&self, x: &Tensor, c: &Tensor, ctx: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let m = self.adaln.forward(&c.silu()?)?.chunk(6, 1)?;
        let (shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp) =
            (&m[0], &m[1], &m[2], &m[3], &m[4], &m[5]);
        let h = self.heads;

        // Self-attention (no mask), AdaLN-Zero gated.
        let y = modulate(&norm(x)?, shift_sa, scale_sa)?;
        let qkv = self.qkv.forward(&y)?;
        let (b, n, c3) = qkv.dims3()?;
        // qkv.view(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        let qkv = qkv
            .reshape((b, n, 3, h, c3 / (3 * h)))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let attn = merge(&sdpa(&qkv.get(0)?, &qkv.get(1)?, &qkv.get(2)?, None)?)?;
        let x = (x + gate_sa
            .unsqueeze(1)?
            .broadcast_mul(&self.self_proj.forward(&attn)?)?)?;

        // Cross-attention on the text context: plain LayerNorm, no modulation, no gate.
        let q = heads(&self.q.forward(&norm(&x)?)?, h)?;
        let kv = self.kv.forward(ctx)?;
        let (kb, km, kc) = kv.dims3()?;
        let kv = kv
            .reshape((kb, km, 2, h, kc / (2 * h)))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let cross = merge(&sdpa(&q, &kv.get(0)?, &kv.get(1)?, Some(mask))?)?;
        let x = (x + self.cross_proj.forward(&cross)?)?;

        // MLP (GELU tanh approximation), AdaLN-Zero gated.
        let y = modulate(&norm(&x)?, shift_mlp, scale_mlp)?;
        let y = self.mlp2.forward(&self.mlp0.forward(&y)?.gelu()?)?;
        Ok((&x + gate_mlp.unsqueeze(1)?.broadcast_mul(&y)?)?)
    }
}

/// The Supra2 denoiser.
pub struct SupraDit {
    config: Supra2Config,
    device: Device,
    x_embed: Linear,
    pos_embed: Tensor,
    t_mlp0: Linear,
    t_mlp2: Linear,
    ctx_proj: Linear,
    blocks: Vec<Block>,
    final_linear: Linear,
    final_adaln: Linear,
}

impl SupraDit {
    /// Build from upstream-keyed f32 tensors (already on the target device); every key and shape of
    /// [`Supra2Config::expected_tensors`] is required and no other key may be present.
    pub fn from_tensors(w: &HashMap<String, Tensor>, config: &Supra2Config) -> Result<Self> {
        let expected = config.expected_tensors();
        let mut unexpected: Vec<&str> = w
            .keys()
            .map(String::as_str)
            .filter(|k| !expected.iter().any(|(n, _)| n == k))
            .collect();
        if !unexpected.is_empty() {
            unexpected.sort_unstable();
            return Err(Error::Msg(format!(
                "supra2: unexpected denoiser tensors {unexpected:?}"
            )));
        }
        if let Some((name, t)) = w.iter().find(|(_, t)| t.dtype() != DType::F32) {
            return Err(Error::Msg(format!(
                "supra2: denoiser tensor `{name}` is {:?}, expected F32",
                t.dtype()
            )));
        }
        let d = config.d_model;
        let pos_embed = w
            .get("pos_embed")
            .cloned()
            .ok_or_else(|| Error::Msg("supra2: missing denoiser tensor `pos_embed`".into()))?;
        if pos_embed.dims() != [1, config.num_tokens(), d] {
            return Err(Error::Msg(format!(
                "supra2: pos_embed shape {:?} does not match the geometry",
                pos_embed.dims()
            )));
        }
        let device = pos_embed.device().clone();
        Ok(Self {
            config: *config,
            device,
            x_embed: lin(w, "x_embed", [d, config.patch_dim()])?,
            pos_embed,
            t_mlp0: lin(w, "t_embed.mlp.0", [d, config.freq_dim])?,
            t_mlp2: lin(w, "t_embed.mlp.2", [d, d])?,
            ctx_proj: lin(w, "ctx_proj", [d, config.ctx_dim])?,
            blocks: (0..config.depth)
                .map(|i| Block::from_weights(w, i, config))
                .collect::<Result<_>>()?,
            final_linear: lin(w, "final.linear", [config.patch_dim(), d])?,
            final_adaln: lin(w, "final.adaln.1", [2 * d, d])?,
        })
    }

    /// Load the converted bundle's `denoiser.safetensors` onto `device`, verifying the inventory.
    pub fn load(path: &std::path::Path, config: &Supra2Config, device: &Device) -> Result<Self> {
        let w = candle_gen::candle_core::safetensors::load(path, device)?;
        Self::from_tensors(&w, config)
    }

    pub fn config(&self) -> &Supra2Config {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Host sinusoidal features for a batch of timesteps → `[B, freq_dim]` on the model's device.
    pub fn timestep_features(&self, t: &[f32]) -> Result<Tensor> {
        let f = self.config.freq_dim;
        let host: Vec<f32> = t.iter().flat_map(|&t| timestep_features(t, f)).collect();
        Ok(Tensor::from_vec(host, (t.len(), f), &self.device)?)
    }

    /// Additive cross-attention mask `[B, 1, 1, M]` from a `0/1` text mask (`1` = attend, as
    /// upstream's boolean `scaled_dot_product_attention` mask).
    pub fn additive_mask(
        mask01: &[f32],
        batch: usize,
        len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        if mask01.len() != batch * len {
            return Err(Error::Msg(format!(
                "supra2: mask has {} values, expected {batch}×{len}",
                mask01.len()
            )));
        }
        let data: Vec<f32> = mask01
            .iter()
            .map(|&m| if m != 0.0 { 0.0 } else { f32::NEG_INFINITY })
            .collect();
        Ok(Tensor::from_vec(data, (batch, 1, 1, len), device)?)
    }

    /// Velocity `v(z, t | ctx)`: `z` NCHW `[B, C, S, S]` f32, `t_features` `[B, freq_dim]`,
    /// `ctx` `[B, M, ctx_dim]`, `mask` additive `[B, 1, 1, M]` → `[B, C, S, S]`.
    pub fn forward(
        &self,
        z: &Tensor,
        t_features: &Tensor,
        ctx: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let c = &self.config;
        let (b, ch, hh, ww) = z.dims4()?;
        let p = c.patch;
        let (h, w) = (hh / p, ww / p);
        // z.view(B, C, h, P, w, P).permute(0, 2, 4, 1, 3, 5).reshape(B, h*w, C*P*P)
        let x = z
            .reshape((b, ch, h, p, w, p))?
            .permute((0, 2, 4, 1, 3, 5))?
            .contiguous()?
            .reshape((b, h * w, ch * p * p))?;
        let x = self.x_embed.forward(&x)?.broadcast_add(&self.pos_embed)?;
        let cond = self
            .t_mlp2
            .forward(&self.t_mlp0.forward(t_features)?.silu()?)?;
        let ctx = self.ctx_proj.forward(ctx)?;
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x, &cond, &ctx, mask)?;
        }
        let m = self.final_adaln.forward(&cond.silu()?)?.chunk(2, 1)?;
        let x = self
            .final_linear
            .forward(&modulate(&norm(&x)?, &m[0], &m[1])?)?;
        // x.view(B, h, w, C, P, P).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        Ok(x.reshape((b, h, w, ch, p, p))?
            .permute((0, 3, 1, 4, 2, 5))?
            .contiguous()?
            .reshape((b, ch, hh, ww))?)
    }
}
