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
    /// A decoder-side mid block (the decoder's quantum labels).
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Self::from_weights_labeled(w, prefix, &crate::quantum::DECODE_LABELS)
    }

    /// A mid block whose iOS quanta carry `labels` (the encoder's or the decoder's).
    pub(crate) fn from_weights_labeled(
        w: &Weights,
        prefix: &str,
        labels: &'static crate::quantum::VaeLabels,
    ) -> Result<Self> {
        Ok(Self {
            resnet0: ResnetBlock2D::from_weights_labeled(
                w,
                &format!("{prefix}.resnets.0"),
                labels,
            )?,
            attention: VaeAttention::from_weights_labeled(
                w,
                &format!("{prefix}.attentions.0"),
                labels,
            )?,
            resnet1: ResnetBlock2D::from_weights_labeled(
                w,
                &format!("{prefix}.resnets.1"),
                labels,
            )?,
        })
    }

    /// Quantize the block's only quantizable Linears — the spatial-attention QKV/out (the resnets
    /// are all conv). Q4/Q8, group_size 64.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attention.quantize(bits)
    }

    /// The block's two resnets and its attention (the iOS pricing prices them).
    pub(crate) fn parts(&self) -> (&ResnetBlock2D, &VaeAttention, &ResnetBlock2D) {
        (&self.resnet0, &self.attention, &self.resnet1)
    }

    /// iOS: each resnet and the attention is one bounded quantum (pass-throughs off device).
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.resnet0.forward_quantum(x)?;
        let h = self.attention.forward_quantum(&h)?;
        self.resnet1.forward_quantum(&h)
    }
}
