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

    /// The upsample as one iOS quantum: the nearest-2x copy (input width at the doubled extent),
    /// the convolution's output and kernel transient ([`mlx_gen::ios_quantum::single_conv_bytes`]).
    pub(crate) fn forward_quantum(&self, x_nchw: &Array) -> Result<Array> {
        crate::quantum::conv_quantum(
            crate::quantum::VAE_UPSAMPLE,
            |backend| {
                let [b, h, w, c] = crate::quantum::nhwc(x_nchw);
                let call = mlx_gen::ios_quantum::ConvCall::new(
                    &[b, 2 * h, 2 * w, c],
                    x_nchw.item_size() as u64,
                    &self.conv_w,
                    1,
                    1,
                );
                mlx_gen::ios_quantum::single_conv_bytes(&call, 1, 0, backend)
            },
            || self.forward(x_nchw),
        )
    }

    /// The `[out, 3, 3, in]` NHWC conv weight (for the iOS convolution accounting proofs).
    #[cfg(test)]
    pub(crate) fn weight(&self) -> &Array {
        &self.conv_w
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
        // iOS: one tiled stage at the doubled extent (it also holds the nearest-2x copy); each conv
        // tile is checked on its own.
        let s = x.shape();
        crate::quantum::before_tiled_stage(
            &[s[0], 2 * s[1], 2 * s[2], s[3]],
            x.item_size() as u64,
            &self.conv_w,
            false,
        )?;
        let up = upsample_nearest(&x, 2)?;
        let h = tiled_conv2d_3x3_nhwc(
            &up,
            &self.conv_w,
            Some(&self.conv_b),
            tile_edge,
            cancel,
            |tile| {
                crate::quantum::before_tile(tile, &self.conv_w)?;
                Ok(tile.clone())
            },
        )?;
        if mlx_gen::ios_quantum::enabled() {
            mlx_gen::ios_quantum::finish(crate::quantum::VAE_TILED_STAGE, &[&h])?;
        }
        Ok(h.transpose_axes(&[0, 3, 1, 2])?)
    }
}
