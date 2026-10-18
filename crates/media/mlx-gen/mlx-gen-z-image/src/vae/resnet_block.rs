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
    /// The quantum labels of the side (encoder / decoder) this resnet runs on.
    labels: &'static crate::quantum::VaeLabels,
}

impl ResnetBlock2D {
    /// A decoder-side resnet (the decoder's quantum labels).
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Self::from_weights_labeled(w, prefix, &crate::quantum::DECODE_LABELS)
    }

    /// A resnet whose iOS quanta carry `labels` (the encoder's or the decoder's).
    pub(crate) fn from_weights_labeled(
        w: &Weights,
        prefix: &str,
        labels: &'static crate::quantum::VaeLabels,
    ) -> Result<Self> {
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
            labels,
        })
    }

    /// `(conv1, conv2)` NHWC weights (the iOS pricing prices tiled stages with them).
    pub(crate) fn conv_weights(&self) -> (&Array, &Array) {
        (&self.conv1_w, &self.conv2_w)
    }

    /// The resnet as one iOS quantum, priced by [`Self::quantum_scratch`]; a pass-through off
    /// device.
    pub(crate) fn forward_quantum(&self, x_nchw: &Array) -> Result<Array> {
        crate::quantum::conv_quantum(
            self.labels.resnet,
            |backend| self.quantum_scratch(x_nchw, backend),
            || self.forward(x_nchw),
        )
    }

    /// The resnet's counted price over `x_nchw` ([`mlx_gen::ios_quantum::resnet_bytes`]: GroupNorm,
    /// SiLU, both convolutions and their kernel transients, the 1x1 shortcut's output, at the input
    /// width) plus the shortcut convolution's own kernel transient.
    pub(crate) fn quantum_scratch(
        &self,
        x_nchw: &Array,
        backend: mlx_gen::ios_quantum::Backend,
    ) -> u64 {
        let act = x_nchw.item_size() as u64;
        let x = crate::quantum::nhwc(x_nchw);
        // The 1x1 shortcut convolution's own kernel transient (its output is counted).
        let shortcut = self.shortcut.as_ref().map_or(0, |(w, _)| {
            mlx_gen::ios_quantum::ConvCall::new(&x, act, w, 1, 0).transient(backend)
        });
        mlx_gen::ios_quantum::resnet_bytes(&x, act, act, &self.conv1_w, &self.conv2_w, backend)
            .saturating_add(shortcut)
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
        // iOS: the tiled layer is one stage quantum (full-resolution input, both conv outputs,
        // shortcut and sum), inside which every conv tile is checked on its own.
        crate::quantum::before_tiled_stage(
            self.labels.stage,
            x.shape(),
            x.item_size() as u64,
            &self.conv1_w,
            true,
        )?;

        let source = mlx_gen::ios_quantum::TileSource {
            full_width: x.shape()[2],
            normalized: true,
        };
        let norm1 = GlobalGroupNorm::new(&x, &self.norm1_w, &self.norm1_b, GN_GROUPS, GN_EPS)?;
        let h = tiled_conv2d_3x3_nhwc(
            &x,
            &self.conv1_w,
            Some(&self.conv1_b),
            tile_edge,
            cancel,
            |tile| {
                crate::quantum::before_tile(self.labels.tile, tile, source, &self.conv1_w)?;
                silu(&norm1.apply(tile)?)
            },
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
            |tile| {
                crate::quantum::before_tile(self.labels.tile, tile, source, &self.conv2_w)?;
                silu(&norm2.apply(tile)?)
            },
        )?;

        let residual = match &self.shortcut {
            Some((sw, sb)) => conv2d(&x, sw, Some(sb), 1, 0)?,
            None => x,
        };
        let out = add(&residual, &h)?;
        out.eval()?;
        if mlx_gen::ios_quantum::enabled() {
            mlx_gen::ios_quantum::finish(self.labels.stage, &[&out])?;
        }
        Ok(out.transpose_axes(&[0, 3, 1, 2])?)
    }
}
