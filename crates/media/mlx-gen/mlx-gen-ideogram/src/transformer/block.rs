//! Ideogram 4 DiT block: attention + SwiGLU MLP with AdaLN "sandwich" norms (a pre-norm scaled by
//! `1+scale`, a post-norm gated by `tanh(gate)`), full segment-masked attention, per-head q/k
//! RMSNorm, and interleaved 3D MRoPE. Port of `Ideogram4Attention` / `Ideogram4MLP` /
//! `Ideogram4TransformerBlock`.

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, multiply, split, tanh};
use mlx_rs::Array;

use mlx_gen::adapters::{prefixed_paths, AdaptableHost, AdaptableLinear};
use mlx_gen::nn::silu;
use mlx_gen::qkv::{
    self, AttnPrepSpec, QkNormSpec, QkvSource, RopeSpec, RopeStyle, RopeTables, RotationAxes,
};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, lin};

/// `1.0 + a`, broadcasting the scalar.
fn plus1(a: &Array) -> Result<Array> {
    Ok(add(a, Array::from_f32(1.0))?)
}

// ── Attention ────────────────────────────────────────────────────────────────────────────
pub struct Ideogram4Attention {
    qkv: AdaptableLinear,
    o: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
}

impl Ideogram4Attention {
    pub fn from_weights(w: &Weights, prefix: &str, num_heads: i32, head_dim: i32) -> Result<Self> {
        Ok(Self {
            qkv: lin(w, &join(prefix, "qkv"), false)?,
            o: lin(w, &join(prefix, "o"), false)?,
            norm_q: w.require(&join(prefix, "norm_q.weight"))?.clone(),
            norm_k: w.require(&join(prefix, "norm_k.weight"))?.clone(),
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            eps: 1e-5,
        })
    }

    /// `x`: `[B, L, hidden]`; `cos`/`sin`: `[B, L, head_dim]`; `mask`: optional additive `[B, 1, L, L]`.
    /// `None` = no attention mask (the packed Ideogram sequence is a single segment, so the additive
    /// segment mask is identically zero — `logit + 0 == logit` — and is skipped rather than built and
    /// added; F-029).
    pub fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
    ) -> Result<Array> {
        // SC-18319 — the shared prologue. Ideogram's knob selection: fused packed QKV (knob 9),
        // per-head q/k RMSNorm after the head split (knob 1), **half-split** `rotate_half` rotation
        // over FULL-width `[B, L, head_dim]` tables (knob 2 — deliberately NOT the adjacent-pair
        // convention Lens/Mage use), applied head-major (after the SDPA transpose), and a single
        // stream (no join, so knob 11 does not apply).
        let spec = AttnPrepSpec::new(self.num_heads, self.head_dim)
            .with_qk_norm(QkNormSpec::per_head(&self.norm_q, &self.norm_k, self.eps))
            .with_rope(RopeSpec {
                style: RopeStyle::RotateHalf,
                q: Some(RopeTables::new(cos, sin)),
                k: Some(RopeTables::new(cos, sin)),
                ..RopeSpec::default()
            })
            .with_rotation_axes(RotationAxes::HeadMajor);
        let heads = qkv::prepare(QkvSource::Packed(&self.qkv.forward(x)?), &spec)?;
        let (q, k, v) = (heads.q, heads.k, heads.v);

        let mask = mask.map(|m| m.as_dtype(q.dtype())).transpose()?;
        let sdpa_mask = mask
            .as_ref()
            .map(mlx_rs::fast::ScaledDotProductAttentionMask::from);
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, sdpa_mask, None)?;
        self.o.forward(&qkv::merge_heads(&o)?)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.qkv.quantize(bits, None)?;
        self.o.quantize(bits, None)?;
        Ok(())
    }
}

impl AdaptableHost for Ideogram4Attention {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["qkv"] => Some(&mut self.qkv),
            ["o"] => Some(&mut self.o),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["qkv", "o"].into_iter().map(String::from).collect()
    }
}

// The HF half-split RoPE that used to live here is now
// `mlx_gen::qkv::apply_rope(.., RopeStyle::RotateHalf, RotationAxes::HeadMajor, ..)` (SC-18319) —
// the identical `x·cos + rotate_half(x)·sin` expression over a full-width table.

// ── SwiGLU MLP ───────────────────────────────────────────────────────────────────────────
pub struct Ideogram4Mlp {
    w1: AdaptableLinear,
    w2: AdaptableLinear,
    w3: AdaptableLinear,
}

impl Ideogram4Mlp {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: lin(w, &join(prefix, "w1"), false)?,
            w2: lin(w, &join(prefix, "w2"), false)?,
            w3: lin(w, &join(prefix, "w3"), false)?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let gated = multiply(&silu(&self.w1.forward(x)?)?, &self.w3.forward(x)?)?;
        self.w2.forward(&gated)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.w1.quantize(bits, None)?;
        self.w2.quantize(bits, None)?;
        self.w3.quantize(bits, None)?;
        Ok(())
    }
}

impl AdaptableHost for Ideogram4Mlp {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["w1"] => Some(&mut self.w1),
            ["w2"] => Some(&mut self.w2),
            ["w3"] => Some(&mut self.w3),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["w1", "w2", "w3"].into_iter().map(String::from).collect()
    }
}

// ── Block ────────────────────────────────────────────────────────────────────────────────
pub struct Ideogram4Block {
    attention: Ideogram4Attention,
    feed_forward: Ideogram4Mlp,
    attention_norm1: Array,
    attention_norm2: Array,
    ffn_norm1: Array,
    ffn_norm2: Array,
    adaln_modulation: AdaptableLinear,
    eps: f32,
}

impl Ideogram4Block {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
        norm_eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            attention: Ideogram4Attention::from_weights(
                w,
                &join(prefix, "attention"),
                num_heads,
                head_dim,
            )?,
            feed_forward: Ideogram4Mlp::from_weights(w, &join(prefix, "feed_forward"))?,
            attention_norm1: w.require(&join(prefix, "attention_norm1.weight"))?.clone(),
            attention_norm2: w.require(&join(prefix, "attention_norm2.weight"))?.clone(),
            ffn_norm1: w.require(&join(prefix, "ffn_norm1.weight"))?.clone(),
            ffn_norm2: w.require(&join(prefix, "ffn_norm2.weight"))?.clone(),
            adaln_modulation: lin(w, &join(prefix, "adaln_modulation"), true)?,
            eps: norm_eps,
        })
    }

    /// `x`: `[B, L, hidden]`; `adaln_input`: `[B, 1, adaln_dim]`; `cos`/`sin`: `[B, L, head_dim]`;
    /// `mask`: optional additive `[B, 1, L, L]` (`None` = unmasked; see [`Ideogram4Attention::forward`]).
    pub fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        mask: Option<&Array>,
        adaln_input: &Array,
    ) -> Result<Array> {
        let mod_ = self.adaln_modulation.forward(adaln_input)?; // [B,1,4*hidden]
        let chunks = split(&mod_, 4, 2)?;
        let (scale_msa, gate_msa, scale_mlp, gate_mlp) =
            (&chunks[0], &chunks[1], &chunks[2], &chunks[3]);
        let gate_msa = tanh(gate_msa)?;
        let gate_mlp = tanh(gate_mlp)?;
        let scale_msa = plus1(scale_msa)?;
        let scale_mlp = plus1(scale_mlp)?;

        let normed = multiply(&rms_norm(x, &self.attention_norm1, self.eps)?, &scale_msa)?;
        let attn_out = self.attention.forward(&normed, cos, sin, mask)?;
        let x = add(
            x,
            &multiply(
                &gate_msa,
                &rms_norm(&attn_out, &self.attention_norm2, self.eps)?,
            )?,
        )?;

        let normed2 = multiply(&rms_norm(&x, &self.ffn_norm1, self.eps)?, &scale_mlp)?;
        let ff = self.feed_forward.forward(&normed2)?;
        let x = add(
            &x,
            &multiply(&gate_mlp, &rms_norm(&ff, &self.ffn_norm2, self.eps)?)?,
        )?;
        Ok(x)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attention.quantize(bits)?;
        self.feed_forward.quantize(bits)?;
        self.adaln_modulation.quantize(bits, None)?;
        Ok(())
    }
}

impl AdaptableHost for Ideogram4Block {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["attention", rest @ ..] => self.attention.adaptable_mut(rest),
            ["feed_forward", rest @ ..] => self.feed_forward.adaptable_mut(rest),
            ["adaln_modulation"] => Some(&mut self.adaln_modulation),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = prefixed_paths("attention", &self.attention);
        out.extend(prefixed_paths("feed_forward", &self.feed_forward));
        out.push("adaln_modulation".to_string());
        out
    }
}
