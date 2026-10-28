//! The **Qwen3-style text-encoder decoder block** shared by the decoder-LM text encoders
//! (sc-27326): pre-norm residual layer, GQA self-attention with optional per-head q/k RMSNorm and
//! HF half-split RoPE, and the bias-free SwiGLU [`SwiGluMlp`].
//!
//! Krea, Boogu, Ideogram, Mage, MiniMax-H3 and FLUX.2 each carried a private copy of these three
//! modules with the same op sequence; what differed is data, carried here as
//! [`Qwen3TextConfig`] plus a loader closure:
//!
//! * how a projection is loaded (packed-detect vs dense, the family's group size) — the
//!   provider's `linear` closure, handed each projection's weight key
//!   (`{prefix}.self_attn.q_proj.weight`);
//! * whether projections run through [`AdaptableLinear::forward_upcast`] (bf16-resident weights at
//!   the activation dtype) or plain [`AdaptableLinear::forward`] — [`Qwen3TextConfig::upcast`];
//! * whether the per-head q/k RMSNorm exists (Qwen3) or not (the FLUX.2-dev Mistral tower) —
//!   [`Qwen3TextConfig::qk_norm`].
//!
//! Z-Image and Qwen-Image keep their own text-encoder attention and layer (Qwen-Image's Qwen2.5
//! tower has biased q/k/v over raw arrays and no q/k norm; Z-Image's layer carries its streamed /
//! compiled forward variants), so only their SwiGLU MLP is shared.

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::add;
use mlx_rs::Array;

use crate::adapters::AdaptableLinear;
use crate::nn::{apply_text_rope, repeat_kv, SwiGluMlp};
use crate::weights::{join, Weights};
use crate::Result;

/// The per-family knobs of [`Qwen3TextDecoderLayer`] / [`Qwen3TextAttention`].
#[derive(Clone, Copy, Debug)]
pub struct Qwen3TextConfig {
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    /// RMSNorm epsilon, shared by the layer norms and the per-head q/k norms.
    pub eps: f32,
    /// Load and apply the per-head q/k RMSNorm (`self_attn.{q,k}_norm.weight`).
    pub qk_norm: bool,
    /// Run every projection through [`AdaptableLinear::forward_upcast`] instead of
    /// [`AdaptableLinear::forward`].
    pub upcast: bool,
}

fn project(linear: &AdaptableLinear, x: &Array, upcast: bool) -> Result<Array> {
    if upcast {
        linear.forward_upcast(x)
    } else {
        linear.forward(x)
    }
}

/// GQA self-attention: bias-less q/k/v/o, optional per-head q/k RMSNorm (head dim, before RoPE),
/// HF half-split RoPE, `repeat_kv`, masked SDPA.
pub struct Qwen3TextAttention {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
    q_norm: Option<Array>,
    k_norm: Option<Array>,
    cfg: Qwen3TextConfig,
    scale: f32,
}

impl Qwen3TextAttention {
    /// Load `{prefix}.{q,k,v,o}_proj` through `linear` (handed each `….weight` key) and, when
    /// [`Qwen3TextConfig::qk_norm`], `{prefix}.{q,k}_norm.weight`.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: Qwen3TextConfig,
        mut linear: impl FnMut(&str) -> Result<AdaptableLinear>,
    ) -> Result<Self> {
        let (q_norm, k_norm) = if cfg.qk_norm {
            (
                Some(w.require(&join(prefix, "q_norm.weight"))?.clone()),
                Some(w.require(&join(prefix, "k_norm.weight"))?.clone()),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            q: linear(&join(prefix, "q_proj.weight"))?,
            k: linear(&join(prefix, "k_proj.weight"))?,
            v: linear(&join(prefix, "v_proj.weight"))?,
            o: linear(&join(prefix, "o_proj.weight"))?,
            q_norm,
            k_norm,
            cfg,
            scale: (cfg.head_dim as f32).powf(-0.5),
        })
    }

    pub fn num_heads(&self) -> i32 {
        self.cfg.num_heads
    }

    pub fn num_kv_heads(&self) -> i32 {
        self.cfg.num_kv_heads
    }

    pub fn head_dim(&self) -> i32 {
        self.cfg.head_dim
    }

    /// The SDPA scale `head_dim^-0.5`.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// The q/k/v/o projections, in that order.
    pub fn projections(&self) -> [&AdaptableLinear; 4] {
        [&self.q, &self.k, &self.v, &self.o]
    }

    /// The per-head q/k RMSNorm weights (empty without [`Qwen3TextConfig::qk_norm`]).
    pub fn norm_weights(&self) -> impl Iterator<Item = &Array> {
        self.q_norm.iter().chain(self.k_norm.iter())
    }

    /// Quantize the q/k/v/o projections (`None` = the default group of 64). The q/k norms stay
    /// full precision.
    pub fn quantize(&mut self, bits: i32, group_size: Option<i32>) -> Result<()> {
        for linear in [&mut self.q, &mut self.k, &mut self.v, &mut self.o] {
            linear.quantize(bits, group_size)?;
        }
        Ok(())
    }

    /// Materialize the projections' and q/k norms' (lazy) weights.
    pub fn materialize_weights(&self) -> Result<()> {
        for linear in self.projections() {
            linear.materialize_weights()?;
        }
        mlx_rs::transforms::eval(self.norm_weights())?;
        Ok(())
    }

    /// The attention prologue up to (not including) RoPE: project `x` `[b, s, hidden]` to
    /// `q` `[b, s, heads, head_dim]` and `k`/`v` `[b, s, kv_heads, head_dim]`, then apply the
    /// per-head q/k RMSNorm when present.
    pub fn project_qkv(&self, x: &Array) -> Result<(Array, Array, Array)> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let c = &self.cfg;
        let q = project(&self.q, x, c.upcast)?.reshape(&[b, s, c.num_heads, c.head_dim])?;
        let k = project(&self.k, x, c.upcast)?.reshape(&[b, s, c.num_kv_heads, c.head_dim])?;
        let v = project(&self.v, x, c.upcast)?.reshape(&[b, s, c.num_kv_heads, c.head_dim])?;
        let q = match &self.q_norm {
            Some(g) => rms_norm(&q, g, c.eps)?,
            None => q,
        };
        let k = match &self.k_norm {
            Some(g) => rms_norm(&k, g, c.eps)?,
            None => k,
        };
        Ok((q, k, v))
    }

    /// The output projection over the merged-head attention output `[b, s, heads·head_dim]`.
    pub fn project_out(&self, o: &Array) -> Result<Array> {
        project(&self.o, o, self.cfg.upcast)
    }

    /// `x`: `[b, s, hidden]`; `cos`/`sin`: `[1, s, head_dim]`; `mask`: additive `[b, 1, s, s]`.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let (q, k, v) = self.project_qkv(x)?;

        let q = apply_text_rope(&q, cos, sin)?;
        let k = apply_text_rope(&k, cos, sin)?;

        // GQA: repeat each kv head `groups` times to match the query heads.
        let groups = self.cfg.num_heads / self.cfg.num_kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        // [b,s,h,hd] → [b,h,s,hd]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let mask = mask.as_dtype(q.dtype())?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, &mask, None)?;
        let o = o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            b,
            s,
            self.cfg.num_heads * self.cfg.head_dim,
        ])?;
        self.project_out(&o)
    }
}

/// Pre-norm residual decoder block: `h = x + attn(input_ln(x))`, then `h + mlp(post_ln(h))`.
pub struct Qwen3TextDecoderLayer {
    input_ln: Array,
    post_ln: Array,
    attn: Qwen3TextAttention,
    mlp: SwiGluMlp,
    eps: f32,
    upcast: bool,
}

impl Qwen3TextDecoderLayer {
    /// Load `{prefix}.{input,post_attention}_layernorm.weight`, `{prefix}.self_attn.*` and
    /// `{prefix}.mlp.{gate,up,down}_proj`, every projection through `linear` (handed its weight
    /// key, e.g. `{prefix}.mlp.gate_proj.weight`).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: Qwen3TextConfig,
        mut linear: impl FnMut(&str) -> Result<AdaptableLinear>,
    ) -> Result<Self> {
        let input_ln = w.require(&join(prefix, "input_layernorm.weight"))?.clone();
        let post_ln = w
            .require(&join(prefix, "post_attention_layernorm.weight"))?
            .clone();
        let attn =
            Qwen3TextAttention::from_weights(w, &join(prefix, "self_attn"), cfg, &mut linear)?;
        let mlp_prefix = join(prefix, "mlp");
        let mlp = SwiGluMlp::new(
            linear(&join(&mlp_prefix, "gate_proj.weight"))?,
            linear(&join(&mlp_prefix, "up_proj.weight"))?,
            linear(&join(&mlp_prefix, "down_proj.weight"))?,
        );
        Ok(Self {
            input_ln,
            post_ln,
            attn,
            mlp,
            eps: cfg.eps,
            upcast: cfg.upcast,
        })
    }

    pub fn attention(&self) -> &Qwen3TextAttention {
        &self.attn
    }

    pub fn mlp(&self) -> &SwiGluMlp {
        &self.mlp
    }

    /// The seven projections: attention q/k/v/o, then MLP gate/up/down.
    pub fn projections(&self) -> [&AdaptableLinear; 7] {
        let [q, k, v, o] = self.attn.projections();
        let [gate, up, down] = self.mlp.projections();
        [q, k, v, o, gate, up, down]
    }

    /// The RMSNorm weights: input/post layer norms, then the attention's q/k norms (if any).
    pub fn norm_weights(&self) -> impl Iterator<Item = &Array> {
        [&self.input_ln, &self.post_ln]
            .into_iter()
            .chain(self.attn.norm_weights())
    }

    /// Quantize every projection (`None` = the default group of 64).
    pub fn quantize(&mut self, bits: i32, group_size: Option<i32>) -> Result<()> {
        self.attn.quantize(bits, group_size)?;
        self.mlp.quantize(bits, group_size)
    }

    /// Materialize every (lazy) weight of the block.
    pub fn materialize_weights(&self) -> Result<()> {
        mlx_rs::transforms::eval([&self.input_ln, &self.post_ln])?;
        self.attn.materialize_weights()?;
        self.mlp.materialize_weights()
    }

    /// `x`: `[b, s, hidden]`; `cos`/`sin`: `[1, s, head_dim]`; `mask`: additive `[b, 1, s, s]`.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        self.forward_with(x, |attn, normed| attn.forward(normed, cos, sin, mask))
    }

    /// The block with a caller-supplied attention call (`attend(attn, input_ln(x))`), e.g. a
    /// KV-cached decode step; the norms, residuals and MLP are this block's.
    pub fn forward_with(
        &self,
        x: &Array,
        attend: impl FnOnce(&Qwen3TextAttention, &Array) -> Result<Array>,
    ) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let h = add(x, &attend(&self.attn, &normed)?)?;
        let normed2 = rms_norm(&h, &self.post_ln, self.eps)?;
        let mlp = if self.upcast {
            self.mlp.forward_upcast(&normed2)?
        } else {
            self.mlp.forward(&normed2)?
        };
        Ok(add(&h, &mlp)?)
    }
}
