//! VAE `UNetMidBlock`: resnet → spatial attention → resnet. NCHW I/O.

use mlx_rs::Array;

use super::{ResnetBlock2D, VaeAttention};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

pub struct UNetMidBlock {
    resnet0: ResnetBlock2D,
    attention: VaeAttention,
    resnet1: ResnetBlock2D,
}

impl UNetMidBlock {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            resnet0: ResnetBlock2D::from_weights(w, &format!("{prefix}.resnets.0"))?,
            attention: VaeAttention::from_weights(w, &format!("{prefix}.attentions.0"))?,
            resnet1: ResnetBlock2D::from_weights(w, &format!("{prefix}.resnets.1"))?,
        })
    }

    /// Quantize the block's only quantizable Linears — the spatial-attention QKV/out (the resnets
    /// are all conv). Q4/Q8, group_size 64.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attention.quantize(bits)
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.resnet0.forward(x)?;
        let h = self.attention.forward(&h)?;
        self.resnet1.forward(&h)
    }
}
