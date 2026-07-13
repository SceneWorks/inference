//! VAE `DownEncoderBlock`: a run of resnet blocks then an optional stride-2 downsampler. The
//! encoder mirror of [`crate::vae::UpDecoderBlock`]. NCHW I/O.

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
            .map(|i| ResnetBlock2D::from_weights(w, &format!("{prefix}.resnets.{i}")))
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

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = x.clone();
        for resnet in &self.resnets {
            h = resnet.forward(&h)?;
        }
        if let Some(ds) = &self.downsampler {
            h = ds.forward(&h)?;
        }
        Ok(h)
    }
}
