//! Qwen3-VL LM decoder block — pre-norm residual:
//! `h += attn(input_layernorm(h))`, then `h += mlp(post_attention_layernorm(h))`.
//!
//! Port of `Qwen3VLTextDecoderLayer`. Both norms are RMSNorm at
//! [`TE_RMS_NORM_EPS`](crate::config::TE_RMS_NORM_EPS).
//!
//! `forward` is public so the vision/edit path (sc-14048) can drive the stack itself when it needs
//! to inject Qwen3-VL **deepstack** features between layers — see the seam note on
//! [`Qwen3VlTextEncoder`](super::Qwen3VlTextEncoder).

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, Qwen3VlAttention, Qwen3VlMlp};

/// One of the 36 decoder blocks.
pub struct Qwen3VlDecoderLayer {
    input_ln: Array,
    post_ln: Array,
    attn: Qwen3VlAttention,
    mlp: Qwen3VlMlp,
    eps: f32,
}

impl Qwen3VlDecoderLayer {
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.mlp.quantize(bits)
    }

    pub(crate) fn quantized_linear_count(&self) -> usize {
        self.attn.quantized_linear_count() + self.mlp.quantized_linear_count()
    }

    /// Load from `{prefix}.input_layernorm.weight`, `{prefix}.post_attention_layernorm.weight`,
    /// `{prefix}.self_attn.*` and `{prefix}.mlp.*`.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            input_ln: w.require(&join(prefix, "input_layernorm.weight"))?.clone(),
            post_ln: w
                .require(&join(prefix, "post_attention_layernorm.weight"))?
                .clone(),
            attn: Qwen3VlAttention::from_weights(
                w,
                &join(prefix, "self_attn"),
                num_heads,
                num_kv_heads,
                head_dim,
                eps,
            )?,
            mlp: Qwen3VlMlp::from_weights(w, &join(prefix, "mlp"))?,
            eps,
        })
    }

    /// `x`: `[b, s, hidden]`; `cos`/`sin`: `[1, s, head_dim]`; `mask`: additive `[b, 1, s, s]`.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let h = add(x, &self.attn.forward(&normed, cos, sin, mask)?)?;
        let normed2 = rms_norm(&h, &self.post_ln, self.eps)?;
        Ok(add(&h, &self.mlp.forward(&normed2)?)?)
    }
}
