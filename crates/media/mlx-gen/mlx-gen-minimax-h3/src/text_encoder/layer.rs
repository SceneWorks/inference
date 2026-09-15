//! Qwen3 decoder block (pre-norm residual): `h += attn(input_ln(h))`, then `h += mlp(post_ln(h))`.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join_key, Qwen3Attention, Qwen3Mlp};

/// One Qwen3 decoder layer.
pub struct Qwen3DecoderLayer {
    input_ln: Array,
    post_ln: Array,
    attn: Qwen3Attention,
    mlp: Qwen3Mlp,
    eps: f32,
}

impl Qwen3DecoderLayer {
    /// Load `{prefix}.input_layernorm.weight`, `{prefix}.post_attention_layernorm.weight`, plus the
    /// `self_attn` and `mlp` submodules.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            input_ln: w
                .require(&join_key(prefix, "input_layernorm.weight"))?
                .clone(),
            post_ln: w
                .require(&join_key(prefix, "post_attention_layernorm.weight"))?
                .clone(),
            attn: Qwen3Attention::from_weights(
                w,
                &join_key(prefix, "self_attn"),
                num_heads,
                num_kv_heads,
                head_dim,
                eps,
            )?,
            mlp: Qwen3Mlp::from_weights(w, &join_key(prefix, "mlp"))?,
            eps,
        })
    }

    /// `x`: `[b, s, hidden]` → `[b, s, hidden]`.
    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let h = add(x, &self.attn.forward(&normed, cos, sin, mask)?)?;
        let normed2 = rms_norm(&h, &self.post_ln, self.eps)?;
        Ok(add(&h, &self.mlp.forward(&normed2)?)?)
    }

    /// Quantize both submodules in place.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.mlp.quantize(bits)?;
        Ok(())
    }

    /// The packed width of this layer's projections, or `None` if any of them loaded dense.
    ///
    /// `None` rather than a per-projection report: a layer's seven projections are packed by one
    /// converter pass at one width, so a partially packed layer is a mis-built artifact and the
    /// caller only needs to know it is not uniformly packed.
    pub fn packed_bits(&self) -> Option<i32> {
        let a = self.attn.packed_bits()?;
        let m = self.mlp.packed_bits()?;
        (a == m).then_some(a)
    }

    /// Device bytes this layer holds, including the two dense-by-policy norms — the layer's real
    /// residency, not just its packable part.
    pub fn nbytes(&self) -> usize {
        self.input_ln.nbytes() + self.post_ln.nbytes() + self.attn.nbytes() + self.mlp.nbytes()
    }
}
