//! VAE encoder assembly — the image side of img2img. Port of the fork's `Encoder.__call__`:
//! conv_in → down-blocks → mid-block → GroupNorm-out → SiLU → conv_out, producing the `2·C`
//! latent-distribution channels (mean + logvar). NCHW throughout.
//!
//! Reuses the same `ResnetBlock2D` / `UNetMidBlock` / `ConvLayer` / `ConvNormOut` modules as the
//! decoder; only [`crate::vae::DownSampler`] / [`crate::vae::DownEncoderBlock`] are encoder-specific.

use mlx_rs::Array;

use super::conv_layers::{ConvLayer, ConvNormOut};
use super::down_encoder_block::DownEncoderBlock;
use super::mid_block::UNetMidBlock;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// Per-down-block `(num_resnet_layers, add_downsample)`.
#[derive(Debug, Clone)]
pub struct VaeEncoderConfig {
    pub down_blocks: Vec<(usize, bool)>,
}

impl VaeEncoderConfig {
    /// The production Z-Image VAE encoder: 4 down-blocks of 2 resnets, downsampling on the first 3.
    pub fn default_z_image() -> Self {
        Self {
            down_blocks: vec![(2, true), (2, true), (2, true), (2, false)],
        }
    }
}

pub struct Encoder {
    conv_in: ConvLayer,
    down_blocks: Vec<DownEncoderBlock>,
    mid_block: UNetMidBlock,
    conv_norm_out: ConvNormOut,
    conv_out: ConvLayer,
}

impl Encoder {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &VaeEncoderConfig) -> Result<Self> {
        let p = |s: &str| {
            if prefix.is_empty() {
                s.to_string()
            } else {
                format!("{prefix}.{s}")
            }
        };
        let down_blocks = cfg
            .down_blocks
            .iter()
            .enumerate()
            .map(|(i, &(layers, down))| {
                DownEncoderBlock::from_weights(w, &p(&format!("down_blocks.{i}")), layers, down)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in: ConvLayer::from_weights(w, &p("conv_in"))?,
            down_blocks,
            mid_block: UNetMidBlock::from_weights(w, &p("mid_block"))?,
            conv_norm_out: ConvNormOut::from_weights(w, &p("conv_norm_out"))?,
            conv_out: ConvLayer::from_weights(w, &p("conv_out"))?,
        })
    }

    /// Quantize the encoder's only quantizable Linears — the mid-block spatial attention.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.mid_block.quantize(bits)
    }

    /// `image` NCHW (3 channels) → `2·C` latent-dist channels NCHW (spatial ÷8).
    ///
    /// iOS: `conv_in`, every resnet / downsample / mid-block stage and the output head are bounded
    /// quanta (pass-throughs off device).
    pub fn forward(&self, image: &Array) -> Result<Array> {
        self.forward_then(image, 0, Ok)
    }

    /// [`Self::forward`] with `then` applied to the encoder output inside the output head's quantum
    /// (priced with `then_bytes` on top) — `Vae::encode`'s latent-mean selection and affine.
    pub(crate) fn forward_then(
        &self,
        image: &Array,
        then_bytes: u64,
        then: impl FnOnce(Array) -> Result<Array>,
    ) -> Result<Array> {
        let mut h = self.conv_in_quantum(image)?;
        for down in &self.down_blocks {
            h = down.forward(&h)?;
        }
        h = self.mid_block.forward(&h)?;
        self.head(&h, then_bytes, then)
    }

    /// The `conv_in` (`first`) or `conv_out` NHWC weight (for the iOS accounting proofs).
    #[cfg(test)]
    pub(crate) fn conv_weight(&self, first: bool) -> &Array {
        if first {
            self.conv_in.weight()
        } else {
            self.conv_out.weight()
        }
    }

    /// `conv_in` as one bounded iOS quantum over the (materialized) input image.
    pub(crate) fn conv_in_quantum(&self, image: &Array) -> Result<Array> {
        crate::quantum::conv_quantum(
            crate::quantum::VAE_ENCODE_IN,
            |backend| {
                let call = mlx_gen::ios_quantum::ConvCall::new(
                    &crate::quantum::nhwc(image),
                    image.item_size() as u64,
                    self.conv_in.weight(),
                    1,
                    1,
                );
                mlx_gen::ios_quantum::single_conv_bytes(&call, 0, 0, backend)
            },
            || self.conv_in.forward(image),
        )
    }

    /// The output GroupNorm + SiLU + `conv_out` (`then` appended inside the same quantum), as one
    /// iOS quantum priced by [`mlx_gen::ios_quantum::norm_conv_bytes`] plus `then_bytes`.
    pub(crate) fn head(
        &self,
        h: &Array,
        then_bytes: u64,
        then: impl FnOnce(Array) -> Result<Array>,
    ) -> Result<Array> {
        crate::quantum::conv_quantum(
            crate::quantum::VAE_ENCODE_OUT,
            |backend| {
                let call = mlx_gen::ios_quantum::ConvCall::new(
                    &crate::quantum::nhwc(h),
                    h.item_size() as u64,
                    self.conv_out.weight(),
                    1,
                    1,
                );
                mlx_gen::ios_quantum::norm_conv_bytes(&call, backend).saturating_add(then_bytes)
            },
            || {
                let h = self.conv_norm_out.forward(h)?;
                let h = silu(&h)?;
                then(self.conv_out.forward(&h)?)
            },
        )
    }
}
