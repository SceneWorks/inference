//! Thin VAE decoder conv layers: `ConvIn`/`ConvOut` (3×3, pad 1) and `ConvNormOut`
//! (pytorch-compatible GroupNorm). NCHW I/O.

use mlx_rs::Array;

use mlx_gen::nn::{conv2d, group_norm, silu};
use mlx_gen::vae_tiling::{tiled_conv2d_3x3_nhwc, GlobalGroupNorm};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Result};

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-6;

/// A 3×3 stride-1 pad-1 conv (used for both `conv_in` and `conv_out`).
pub struct ConvLayer {
    w: Array,
    b: Array,
}

impl ConvLayer {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{prefix}.conv.weight"))?.clone(),
            b: w.require(&format!("{prefix}.conv.bias"))?.clone(),
        })
    }

    pub fn forward(&self, x_nchw: &Array) -> Result<Array> {
        let x = x_nchw.transpose_axes(&[0, 2, 3, 1])?; // NHWC
        let h = conv2d(&x, &self.w, Some(&self.b), 1, 1)?;
        Ok(h.transpose_axes(&[0, 3, 1, 2])?) // NCHW
    }

    /// Final VAE `GroupNorm → SiLU → conv_out`, preserving full-image normalization while bounding
    /// the 3×3 convolution (sc-19753).
    pub fn forward_tiled_after_norm(
        &self,
        x_nchw: &Array,
        norm: &ConvNormOut,
        tile_edge: i32,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        let x = x_nchw.transpose_axes(&[0, 2, 3, 1])?;
        let global = GlobalGroupNorm::new(&x, &norm.weight, &norm.bias, GN_GROUPS, GN_EPS)?;
        let h = tiled_conv2d_3x3_nhwc(&x, &self.w, Some(&self.b), tile_edge, cancel, |tile| {
            silu(&global.apply(tile)?)
        })?;
        Ok(h.transpose_axes(&[0, 3, 1, 2])?)
    }
}

/// Final GroupNorm before the output conv. NCHW I/O.
pub struct ConvNormOut {
    weight: Array,
    bias: Array,
}

impl ConvNormOut {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{prefix}.norm.weight"))?.clone(),
            bias: w.require(&format!("{prefix}.norm.bias"))?.clone(),
        })
    }

    pub fn forward(&self, x_nchw: &Array) -> Result<Array> {
        let x = x_nchw.transpose_axes(&[0, 2, 3, 1])?; // NHWC
        let h = group_norm(&x, &self.weight, &self.bias, GN_GROUPS, GN_EPS)?;
        Ok(h.transpose_axes(&[0, 3, 1, 2])?) // NCHW
    }
}
