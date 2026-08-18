//! VAE ResnetBlock2D: GroupNorm→SiLU→Conv3×3 ×2 with a residual (1×1 conv shortcut when the
//! channel count changes). NCHW I/O.

use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::nn::{conv2d, group_norm, silu};
use mlx_gen::vae_tiling::{tiled_conv2d_3x3_nhwc, GlobalGroupNorm};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Result};

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-6;

pub struct ResnetBlock2D {
    norm1_w: Array,
    norm1_b: Array,
    conv1_w: Array,
    conv1_b: Array,
    norm2_w: Array,
    norm2_b: Array,
    conv2_w: Array,
    conv2_b: Array,
    shortcut: Option<(Array, Array)>,
}

impl ResnetBlock2D {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let g = |s: &str| w.require(&format!("{prefix}.{s}")).cloned();
        let shortcut = match (
            w.get(&format!("{prefix}.conv_shortcut.weight")),
            w.get(&format!("{prefix}.conv_shortcut.bias")),
        ) {
            (Some(sw), Some(sb)) => Some((sw.clone(), sb.clone())),
            _ => None,
        };
        Ok(Self {
            norm1_w: g("norm1.weight")?,
            norm1_b: g("norm1.bias")?,
            conv1_w: g("conv1.weight")?,
            conv1_b: g("conv1.bias")?,
            norm2_w: g("norm2.weight")?,
            norm2_b: g("norm2.bias")?,
            conv2_w: g("conv2.weight")?,
            conv2_b: g("conv2.bias")?,
            shortcut,
        })
    }

    pub fn forward(&self, x_nchw: &Array) -> Result<Array> {
        let x = x_nchw.transpose_axes(&[0, 2, 3, 1])?; // NHWC

        let h = group_norm(&x, &self.norm1_w, &self.norm1_b, GN_GROUPS, GN_EPS)?;
        let h = conv2d(&silu(&h)?, &self.conv1_w, Some(&self.conv1_b), 1, 1)?;
        let h = group_norm(&h, &self.norm2_w, &self.norm2_b, GN_GROUPS, GN_EPS)?;
        let h = conv2d(&silu(&h)?, &self.conv2_w, Some(&self.conv2_b), 1, 1)?;

        let residual = match &self.shortcut {
            Some((sw, sb)) => conv2d(&x, sw, Some(sb), 1, 0)?, // 1x1
            None => x,
        };
        Ok(add(&residual, &h)?.transpose_axes(&[0, 3, 1, 2])?) // NCHW
    }

    /// Normalization-correct bounded forward for a VAE decode tail (sc-19753).
    ///
    /// GroupNorm statistics are captured from each full layer activation, while the expensive 3×3
    /// convolutions run on halo-expanded tiles. This matches dense normalization semantics instead
    /// of normalizing each whole-decoder crop independently.
    pub fn forward_tiled(
        &self,
        x_nchw: &Array,
        tile_edge: i32,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        let x = x_nchw.transpose_axes(&[0, 2, 3, 1])?;

        let norm1 = GlobalGroupNorm::new(&x, &self.norm1_w, &self.norm1_b, GN_GROUPS, GN_EPS)?;
        let h = tiled_conv2d_3x3_nhwc(
            &x,
            &self.conv1_w,
            Some(&self.conv1_b),
            tile_edge,
            cancel,
            |tile| silu(&norm1.apply(tile)?),
        )?;
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        let norm2 = GlobalGroupNorm::new(&h, &self.norm2_w, &self.norm2_b, GN_GROUPS, GN_EPS)?;
        let h = tiled_conv2d_3x3_nhwc(
            &h,
            &self.conv2_w,
            Some(&self.conv2_b),
            tile_edge,
            cancel,
            |tile| silu(&norm2.apply(tile)?),
        )?;

        let residual = match &self.shortcut {
            Some((sw, sb)) => conv2d(&x, sw, Some(sb), 1, 0)?,
            None => x,
        };
        let out = add(&residual, &h)?;
        out.eval()?;
        Ok(out.transpose_axes(&[0, 3, 1, 2])?)
    }
}
