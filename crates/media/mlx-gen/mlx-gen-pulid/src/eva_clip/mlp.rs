//! EVA `SwiGLU` FFN with sub-LN: `w3( ffn_ln( silu(w1·x) * (w2·x) ) )`. Port of
//! `eva_vit_model.py SwiGLU` (naiveswiglu + subln). All three linears are biased; `ffn_ln` is a
//! LayerNorm over the hidden dim (2730).

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::multiply;
use mlx_rs::Array;

use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::eva_clip::{join, EPS};

pub struct SwiGlu {
    w1_w: mlx_gen::adapters::AdaptableLinear,

    w2_w: mlx_gen::adapters::AdaptableLinear,

    ffn_ln_w: Array,
    ffn_ln_b: Array,
    w3_w: mlx_gen::adapters::AdaptableLinear,
}

impl SwiGlu {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1_w: mlx_gen::quant::lin_with_bias(
                w,
                &join(prefix, "w1"),
                Some(w.require(&join(prefix, "w1.bias"))?.clone()),
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
            )?,
            w2_w: mlx_gen::quant::lin_with_bias(
                w,
                &join(prefix, "w2"),
                Some(w.require(&join(prefix, "w2.bias"))?.clone()),
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
            )?,
            ffn_ln_w: w.require(&join(prefix, "ffn_ln.weight"))?.clone(),
            ffn_ln_b: w.require(&join(prefix, "ffn_ln.bias"))?.clone(),
            w3_w: mlx_gen::quant::lin_with_bias(
                w,
                &join(prefix, "w3"),
                Some(w.require(&join(prefix, "w3.bias"))?.clone()),
                mlx_gen::quant::DEFAULT_GROUP_SIZE,
            )?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x1 = self.w1_w.forward(x)?;
        let x2 = self.w2_w.forward(x)?;
        let hidden = multiply(&silu(&x1)?, &x2)?;
        let hidden = layer_norm(&hidden, Some(&self.ffn_ln_w), Some(&self.ffn_ln_b), EPS)?;
        self.w3_w.forward(&hidden)
    }
}
