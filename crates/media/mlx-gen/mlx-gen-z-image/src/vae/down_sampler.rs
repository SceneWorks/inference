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

    /// The downsample as one iOS quantum: the bottom/right padded copy and the stride-2
    /// convolution's output and kernel transient ([`mlx_gen::ios_quantum::single_conv_bytes`]).
    pub(crate) fn forward_quantum(&self, x_nchw: &Array) -> Result<Array> {
        crate::quantum::conv_quantum(
            crate::quantum::VAE_DOWNSAMPLE,
            |backend| {
                let [b, h, w, c] = crate::quantum::nhwc(x_nchw);
                let call = mlx_gen::ios_quantum::ConvCall::new(
                    &[b, h + 1, w + 1, c],
                    x_nchw.item_size() as u64,
                    &self.w,
                    2,
                    0,
                );
                mlx_gen::ios_quantum::single_conv_bytes(&call, 1, 0, backend)
            },
            || self.forward(x_nchw),
        )
    }

    /// The downsample over output cores of at most `tile` pixels a side with exact halos (the
    /// bounded encode, sc-26433): the shared [`mlx_gen::vae_tiling::downsample_tiled_nhwc`] — one
    /// tiled iOS stage, every tile checked before its graph — under Z-Image's labels. NCHW I/O.
    pub(crate) fn forward_tiled(
        &self,
        x_nchw: &Array,
        tile: i32,
        cancel: Option<&mlx_gen::CancelFlag>,
    ) -> Result<Array> {
        let x = x_nchw.transpose_axes(&[0, 2, 3, 1])?;
        let h = mlx_gen::vae_tiling::downsample_tiled_nhwc(
            &x,
            &self.w,
            &self.b,
            tile,
            cancel,
            crate::quantum::VAE_TILED_STAGE,
            crate::quantum::VAE_TILE,
        )?;
        Ok(h.transpose_axes(&[0, 3, 1, 2])?)
    }

    /// The `[out, 3, 3, in]` NHWC conv weight (for the iOS convolution accounting proofs).
    #[cfg(test)]
    pub(crate) fn weight(&self) -> &Array {
        &self.w
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
