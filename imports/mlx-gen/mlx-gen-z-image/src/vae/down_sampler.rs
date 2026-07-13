//! VAE encoder `DownSampler`: an asymmetric (bottom/right) 1-pixel pad then a 3×3 stride-2
//! conv with no padding — halving each spatial dim. Port of the fork's `DownSampler`
//! (`mx.pad(((0,0),(0,0),(0,1),(0,1)))` → conv stride 2 pad 0). NCHW I/O.

use mlx_rs::ops::pad;
use mlx_rs::Array;

use mlx_gen::nn::conv2d;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

pub struct DownSampler {
    w: Array,
    b: Array,
}

impl DownSampler {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{prefix}.conv.weight"))?.clone(),
            b: w.require(&format!("{prefix}.conv.bias"))?.clone(),
        })
    }

    pub fn forward(&self, x_nchw: &Array) -> Result<Array> {
        let x = x_nchw.transpose_axes(&[0, 2, 3, 1])?; // NHWC
                                                       // Pad H (axis 1) and W (axis 2) by one at the end (bottom/right). Equivalent to the
                                                       // fork's NCHW `pad(((0,0),(0,0),(0,1),(0,1)))` followed by the NHWC transpose.
        let x = pad(&x, &[(0, 0), (0, 1), (0, 1), (0, 0)], None, None)?;
        let h = conv2d(&x, &self.w, Some(&self.b), 2, 0)?; // stride 2, padding 0
        Ok(h.transpose_axes(&[0, 3, 1, 2])?) // NCHW
    }
}
