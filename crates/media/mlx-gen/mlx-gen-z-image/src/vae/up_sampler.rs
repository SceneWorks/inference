//! VAE UpSampler: nearest-2× upsample then a 3×3 conv. NCHW I/O.

use mlx_rs::Array;

use mlx_gen::nn::{conv2d, upsample_nearest};
use mlx_gen::vae_tiling::tiled_conv2d_3x3_nhwc;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Result};

pub struct UpSampler {
    conv_w: Array,
    conv_b: Array,
}

impl UpSampler {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv_w: w.require(&format!("{prefix}.conv.weight"))?.clone(),
            conv_b: w.require(&format!("{prefix}.conv.bias"))?.clone(),
        })
    }

    pub fn forward(&self, x_nchw: &Array) -> Result<Array> {
        let x = x_nchw.transpose_axes(&[0, 2, 3, 1])?; // NHWC
        let up = upsample_nearest(&x, 2)?;
        let h = conv2d(&up, &self.conv_w, Some(&self.conv_b), 1, 1)?;
        Ok(h.transpose_axes(&[0, 3, 1, 2])?) // NCHW
    }

    pub fn forward_tiled(
        &self,
        x_nchw: &Array,
        tile_edge: i32,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        let x = x_nchw.transpose_axes(&[0, 2, 3, 1])?;
        let up = upsample_nearest(&x, 2)?;
        let h = tiled_conv2d_3x3_nhwc(
            &up,
            &self.conv_w,
            Some(&self.conv_b),
            tile_edge,
            cancel,
            |tile| Ok(tile.clone()),
        )?;
        Ok(h.transpose_axes(&[0, 3, 1, 2])?)
    }
}
