//! Text-encoder decoder block (pre-norm residual): `h += attn(input_ln(h))`, then
//! `h += mlp(post_ln(h))`. Port of the fork's `EncoderLayer`.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, TextAttention, TextMlp};

pub struct EncoderLayer {
    input_ln: Array,
    post_ln: Array,
    attn: TextAttention,
    mlp: TextMlp,
    eps: f32,
}

impl EncoderLayer {
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
            attn: TextAttention::from_weights(
                w,
                &join(prefix, "self_attn"),
                num_heads,
                num_kv_heads,
                head_dim,
            )?,
            mlp: TextMlp::from_weights(w, &join(prefix, "mlp"))?,
            eps,
        })
    }

    /// Quantize the block's Linears (attention QKV/out + MLP) to Q4/Q8. The two layer-norm scales
    /// stay dense (not Linears).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.mlp.quantize(bits)?;
        Ok(())
    }

    pub fn forward(&self, x: &Array, cos: &Array, sin: &Array, mask: &Array) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let h = add(x, &self.attn.forward(&normed, cos, sin, mask)?)?;
        let normed2 = rms_norm(&h, &self.post_ln, self.eps)?;
        Ok(add(&h, &self.mlp.forward(&normed2)?)?)
    }
}
