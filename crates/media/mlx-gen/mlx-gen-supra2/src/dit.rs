//! `SupraDiT` — faithful port of the frozen upstream `inference.py` model classes.
//!
//! Patchify → `x_embed` + learned `pos_embed`; sinusoidal timestep features (host-side, shared with
//! Candle through [`gen_core_checkpoint::supra2::timestep_features`]) → `t_embed` MLP; `ctx_proj` of the
//! Flan-T5 context; `depth` × [AdaLN-Zero self-attention, **unmodulated** cross-attention masked by
//! the text attention mask, AdaLN-Zero MLP]; AdaLN final layer; unpatchify. Runs in f32 (the
//! checkpoint's dtype and upstream's CPU/MPS execution dtype). Weight keys are upstream's
//! `state_dict` names verbatim.

use gen_core_checkpoint::supra2::{timestep_features, Supra2Config};
use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::nn::{gelu_tanh, silu};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::fast::{layer_norm, scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::{add, multiply, split};
use mlx_rs::Array;

const LN_EPS: f32 = 1e-6;

fn lin(w: &Weights, prefix: &str, shape: [usize; 2]) -> Result<AdaptableLinear> {
    let weight = w.require(&format!("{prefix}.weight"))?;
    let bias = w.require(&format!("{prefix}.bias"))?;
    let want = [shape[0] as i32, shape[1] as i32];
    if weight.shape() != want || bias.shape() != [want[0]] {
        return Err(Error::Msg(format!(
            "supra2: `{prefix}` has shape {:?}/{:?}, expected {want:?}",
            weight.shape(),
            bias.shape()
        )));
    }
    Ok(AdaptableLinear::dense(weight.clone(), Some(bias.clone())))
}

/// `x * (1 + scale[:, None]) + shift[:, None]`.
fn modulate(x: &Array, shift: &Array, scale: &Array) -> Result<Array> {
    let one = add(&scale.expand_dims(1)?, scalar(1.0))?;
    Ok(add(&multiply(x, &one)?, &shift.expand_dims(1)?)?)
}

fn norm(x: &Array) -> Result<Array> {
    Ok(layer_norm(x, None, None, LN_EPS)?)
}

/// `[B, N, H·D]` → `[B, H, N, D]`.
fn heads(x: &Array, h: i32) -> Result<Array> {
    let sh = x.shape();
    Ok(x.reshape(&[sh[0], sh[1], h, sh[2] / h])?
        .transpose_axes(&[0, 2, 1, 3])?)
}

fn merge(x: &Array) -> Result<Array> {
    let sh = x.shape();
    Ok(x.transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[sh[0], sh[2], sh[1] * sh[3]])?)
}

fn sdpa(q: &Array, k: &Array, v: &Array, mask: Option<&Array>) -> Result<Array> {
    let d = q.shape()[3];
    let scale = 1.0 / (d as f32).sqrt();
    Ok(match mask {
        Some(m) => scaled_dot_product_attention(
            q,
            k,
            v,
            scale,
            ScaledDotProductAttentionMask::Array(m),
            None,
        )?,
        None => scaled_dot_product_attention(q, k, v, scale, None, None)?,
    })
}

struct Block {
    qkv: AdaptableLinear,
    self_proj: AdaptableLinear,
    q: AdaptableLinear,
    kv: AdaptableLinear,
    cross_proj: AdaptableLinear,
    mlp0: AdaptableLinear,
    mlp2: AdaptableLinear,
    adaln: AdaptableLinear,
    heads: i32,
}

impl Block {
    fn from_weights(w: &Weights, i: usize, c: &Supra2Config) -> Result<Self> {
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
            heads: c.num_heads as i32,
        })
    }

    /// `x`: `[B, N, D]`; `c`: `[B, D]` conditioning; `ctx`: `[B, M, D]` projected context;
    /// `mask`: additive `[B, 1, 1, M]`.
    fn forward(&self, x: &Array, c: &Array, ctx: &Array, mask: &Array) -> Result<Array> {
        let m = split(&self.adaln.forward(&silu(c)?)?, 6, 1)?;
        let (shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp) =
            (&m[0], &m[1], &m[2], &m[3], &m[4], &m[5]);
        let h = self.heads;

        // Self-attention (no mask), AdaLN-Zero gated.
        let y = modulate(&norm(x)?, shift_sa, scale_sa)?;
        let qkv = self.qkv.forward(&y)?;
        let sh = qkv.shape();
        let qkv = qkv.reshape(&[sh[0], sh[1], 3, h, sh[2] / (3 * h)])?;
        let take = |i: i32| -> Result<Array> {
            Ok(qkv
                .take_axis(Array::from_int(i), 2)?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let attn = merge(&sdpa(&take(0)?, &take(1)?, &take(2)?, None)?)?;
        let x = add(
            x,
            &multiply(&gate_sa.expand_dims(1)?, &self.self_proj.forward(&attn)?)?,
        )?;

        // Cross-attention on the text context: plain LayerNorm, no modulation, no gate.
        let q = heads(&self.q.forward(&norm(&x)?)?, h)?;
        let kv = self.kv.forward(ctx)?;
        let ks = kv.shape();
        let kv = kv.reshape(&[ks[0], ks[1], 2, h, ks[2] / (2 * h)])?;
        let k = kv
            .take_axis(Array::from_int(0), 2)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = kv
            .take_axis(Array::from_int(1), 2)?
            .transpose_axes(&[0, 2, 1, 3])?;
        let cross = merge(&sdpa(&q, &k, &v, Some(mask))?)?;
        let x = add(&x, &self.cross_proj.forward(&cross)?)?;

        // MLP, AdaLN-Zero gated.
        let y = modulate(&norm(&x)?, shift_mlp, scale_mlp)?;
        let y = self.mlp2.forward(&gelu_tanh(&self.mlp0.forward(&y)?)?)?;
        Ok(add(&x, &multiply(&gate_mlp.expand_dims(1)?, &y)?)?)
    }
}

/// The Supra2 denoiser.
pub struct SupraDit {
    config: Supra2Config,
    x_embed: AdaptableLinear,
    pos_embed: Array,
    t_mlp0: AdaptableLinear,
    t_mlp2: AdaptableLinear,
    ctx_proj: AdaptableLinear,
    blocks: Vec<Block>,
    final_linear: AdaptableLinear,
    final_adaln: AdaptableLinear,
}

impl SupraDit {
    /// Build from upstream-keyed f32 weights; every key and shape of
    /// [`Supra2Config::expected_tensors`] is required and no other key may be present.
    pub fn from_weights(w: &Weights, config: &Supra2Config) -> Result<Self> {
        let expected = config.expected_tensors();
        let unexpected: Vec<&str> = w
            .keys()
            .filter(|k| !expected.iter().any(|(n, _)| n == k))
            .collect();
        if !unexpected.is_empty() {
            return Err(Error::Msg(format!(
                "supra2: unexpected denoiser tensors {unexpected:?}"
            )));
        }
        let d = config.d_model;
        let pos_embed = w.require("pos_embed")?.clone();
        if pos_embed.shape() != [1, config.num_tokens() as i32, d as i32] {
            return Err(Error::Msg(format!(
                "supra2: pos_embed shape {:?} does not match the geometry",
                pos_embed.shape()
            )));
        }
        Ok(Self {
            config: *config,
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

    pub fn config(&self) -> &Supra2Config {
        &self.config
    }

    /// Host sinusoidal features for a batch of timesteps → `[B, freq_dim]`.
    pub fn timestep_features(&self, t: &[f32]) -> Array {
        let f = self.config.freq_dim;
        let host: Vec<f32> = t.iter().flat_map(|&t| timestep_features(t, f)).collect();
        Array::from_slice(&host, &[t.len() as i32, f as i32])
    }

    /// Additive cross-attention mask `[B, 1, 1, M]` from a `0/1` text mask (`True` = attend, as
    /// upstream's boolean `scaled_dot_product_attention` mask).
    pub fn additive_mask(mask01: &[f32], batch: usize, len: usize) -> Result<Array> {
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
        Ok(Array::from_slice(&data, &[batch as i32, 1, 1, len as i32]))
    }

    /// Velocity `v(z, t | ctx)`: `z` NCHW `[B, C, S, S]` f32, `t_features` `[B, freq_dim]`,
    /// `ctx` `[B, M, ctx_dim]`, `mask` additive `[B, 1, 1, M]` → `[B, C, S, S]`.
    pub fn forward(
        &self,
        z: &Array,
        t_features: &Array,
        ctx: &Array,
        mask: &Array,
    ) -> Result<Array> {
        let c = &self.config;
        let sh = z.shape();
        let (b, ch, hh, ww) = (sh[0], sh[1], sh[2], sh[3]);
        let p = c.patch as i32;
        let (h, w) = (hh / p, ww / p);
        // z.view(B, C, h, P, w, P).permute(0, 2, 4, 1, 3, 5).reshape(B, h*w, C*P*P)
        let x = z
            .reshape(&[b, ch, h, p, w, p])?
            .transpose_axes(&[0, 2, 4, 1, 3, 5])?
            .reshape(&[b, h * w, ch * p * p])?;
        let x = add(&self.x_embed.forward(&x)?, &self.pos_embed)?;
        let cond = self
            .t_mlp2
            .forward(&silu(&self.t_mlp0.forward(t_features)?)?)?;
        let ctx = self.ctx_proj.forward(ctx)?;
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x, &cond, &ctx, mask)?;
        }
        let m = split(&self.final_adaln.forward(&silu(&cond)?)?, 2, 1)?;
        let x = self
            .final_linear
            .forward(&modulate(&norm(&x)?, &m[0], &m[1])?)?;
        // x.view(B, h, w, C, P, P).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)
        Ok(x.reshape(&[b, h, w, ch, p, p])?
            .transpose_axes(&[0, 3, 1, 4, 2, 5])?
            .reshape(&[b, ch, hh, ww])?)
    }
}
