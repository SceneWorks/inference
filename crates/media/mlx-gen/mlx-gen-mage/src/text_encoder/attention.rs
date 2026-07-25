//! Qwen3-VL LM self-attention: GQA (32 query / 8 kv heads), **bias-less** q/k/v/o, per-head
//! `q_norm`/`k_norm` RMSNorm applied on the head dim *before* RoPE, interleaved M-RoPE, causal SDPA.
//!
//! Port of `transformers`' `Qwen3VLTextAttention` as re-bound by the reference's
//! `qwen3_patch_forward` (`_vendor/mage_flow/models/modules/text_encoder.py:298-360`, `:369`).
//!
//! **`eps` is `rms_norm_eps` (1e-6), not mlx's 1e-5 default.** `Qwen3VLTextAttention.__init__`
//! constructs both head norms as `Qwen3VLTextRMSNorm(head_dim, eps=config.rms_norm_eps)`. The
//! `mlx-gen-z-image` sibling deliberately uses mlx's default there because its fork constructs the
//! norms without an explicit eps — inheriting that here would be wrong, which is why
//! [`TE_RMS_NORM_EPS`](crate::config::TE_RMS_NORM_EPS) is threaded in explicitly rather than
//! defaulted.
//!
//! `head_dim` is **decoupled** from `hidden_size / num_heads`: 32 × 128 = 4096 ≠ 2560, so `q_proj`
//! is `Linear(2560 → 4096)` and `o_proj` is `Linear(4096 → 2560)`. Reshapes must use the configured
//! `head_dim`, never a derived one.

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::{apply_text_rope, repeat_kv};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, lin};

/// One decoder layer's grouped-query self-attention.
pub struct Qwen3VlAttention {
    q_proj: AdaptableLinear,
    k_proj: AdaptableLinear,
    v_proj: AdaptableLinear,
    o_proj: AdaptableLinear,
    q_norm: Array,
    k_norm: Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
}

impl Qwen3VlAttention {
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.q_proj.quantize(bits, None)?;
        self.k_proj.quantize(bits, None)?;
        self.v_proj.quantize(bits, None)?;
        self.o_proj.quantize(bits, None)
    }

    pub(crate) fn quantized_linear_count(&self) -> usize {
        [&self.q_proj, &self.k_proj, &self.v_proj, &self.o_proj]
            .into_iter()
            .filter(|linear| linear.is_quantized())
            .count()
    }

    /// Load from `{prefix}.{q,k,v,o}_proj.weight` + `{prefix}.{q,k}_norm.weight`.
    ///
    /// `attention_bias` is `false` for this checkpoint
    /// ([`QwenVlTextConfig::attention_bias`](crate::config::QwenVlTextConfig)), so no `.bias`
    /// tensor is looked up — a checkpoint that shipped one would be a different model and is
    /// caught by the missing-key error rather than silently ignored.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: lin(w, &join(prefix, "q_proj"))?,
            k_proj: lin(w, &join(prefix, "k_proj"))?,
            v_proj: lin(w, &join(prefix, "v_proj"))?,
            o_proj: lin(w, &join(prefix, "o_proj"))?,
            q_norm: w.require(&join(prefix, "q_norm.weight"))?.clone(),
            k_norm: w.require(&join(prefix, "k_norm.weight"))?.clone(),
            num_heads,
            num_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            eps,
        })
    }

    /// `x`: `[b, s, hidden]`; `cos`/`sin`: `[1, s, head_dim]`; `mask`: additive `[b, 1, s, s]`.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);

        let q = self
            .q_proj
            .forward_upcast(x)?
            .reshape(&[b, s, self.num_heads, self.head_dim])?;
        let k =
            self.k_proj
                .forward_upcast(x)?
                .reshape(&[b, s, self.num_kv_heads, self.head_dim])?;
        let v =
            self.v_proj
                .forward_upcast(x)?
                .reshape(&[b, s, self.num_kv_heads, self.head_dim])?;

        // Per-head RMSNorm over the head dim, BEFORE RoPE (`text_encoder.py:310-311`).
        let q = rms_norm(&q, &self.q_norm, self.eps)?;
        let k = rms_norm(&k, &self.k_norm, self.eps)?;

        let q = apply_text_rope(&q, cos, sin)?;
        let k = apply_text_rope(&k, cos, sin)?;

        let groups = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, groups)?;
        let v = repeat_kv(&v, groups)?;

        // [b, s, h, hd] → [b, h, s, hd]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let mask = mask.as_dtype(q.dtype())?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, &mask, None)?;
        let o =
            o.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, s, self.num_heads * self.head_dim])?;
        self.o_proj.forward_upcast(&o)
    }
}
