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
    pub fn forward(&self, image: &Array) -> Result<Array> {
        let mut h = self.conv_in.forward(image)?;
        for down in &self.down_blocks {
            h = down.forward(&h)?;
        }
        h = self.mid_block.forward(&h)?;
        h = self.conv_norm_out.forward(&h)?;
        h = silu(&h)?;
        self.conv_out.forward(&h)
    }
}
