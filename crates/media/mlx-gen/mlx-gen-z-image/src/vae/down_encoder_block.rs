//! VAE `DownEncoderBlock`: a run of resnet blocks then an optional stride-2 downsampler. The
//! encoder mirror of `crate::vae::UpDecoderBlock`. NCHW I/O.

use mlx_rs::Array;

use super::{DownSampler, ResnetBlock2D};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

pub struct DownEncoderBlock {
    resnets: Vec<ResnetBlock2D>,
    downsampler: Option<DownSampler>,
}

impl DownEncoderBlock {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_layers: usize,
        add_downsample: bool,
    ) -> Result<Self> {
        let resnets = (0..num_layers)
            .map(|i| {
                ResnetBlock2D::from_weights_labeled(
                    w,
                    &format!("{prefix}.resnets.{i}"),
                    &crate::quantum::ENCODE_LABELS,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let downsampler = if add_downsample {
            Some(DownSampler::from_weights(
                w,
                &format!("{prefix}.downsamplers.0"),
            )?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsampler,
        })
    }

    /// Whether the block ends in a stride-2 downsample.
    pub(crate) fn downsamples(&self) -> bool {
        self.downsampler.is_some()
    }

    /// The block's resnets and downsampler (the iOS pricing prices them).
    pub(crate) fn parts(&self) -> (&[ResnetBlock2D], Option<&DownSampler>) {
        (&self.resnets, self.downsampler.as_ref())
    }

    /// The bounded encode's block (sc-26433): each resnet layer-wise tiled at `edge` (whole-image
    /// GroupNorm statistics, halo-expanded convolution tiles) and the downsample over cores of
    /// `down_edge` output pixels; one tiled iOS stage per layer, every tile checked.
    pub(crate) fn forward_tiled(
        &self,
        x: &Array,
        edge: i32,
        down_edge: i32,
        cancel: Option<&mlx_gen::CancelFlag>,
    ) -> Result<Array> {
        let mut h = x.clone();
        for resnet in &self.resnets {
            h = resnet.forward_tiled(&h, edge, cancel)?;
        }
        if let Some(ds) = &self.downsampler {
            h = ds.forward_tiled(&h, down_edge, cancel)?;
        }
        Ok(h)
    }

    /// iOS: each resnet and the downsample is one bounded quantum (pass-throughs off device).
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = x.clone();
        for resnet in &self.resnets {
            h = resnet.forward_quantum(&h)?;
        }
        if let Some(ds) = &self.downsampler {
            h = ds.forward_quantum(&h)?;
        }
        Ok(h)
    }
}
