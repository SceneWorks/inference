//! VAE `UpDecoderBlock`: a run of resnet blocks then an optional nearest-2× upsampler.
//! NCHW I/O.

use mlx_rs::Array;

use super::{ResnetBlock2D, UpSampler};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Result};

pub struct UpDecoderBlock {
    resnets: Vec<ResnetBlock2D>,
    upsampler: Option<UpSampler>,
}

impl UpDecoderBlock {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_layers: usize,
        add_upsample: bool,
    ) -> Result<Self> {
        let resnets = (0..num_layers)
            .map(|i| ResnetBlock2D::from_weights(w, &format!("{prefix}.resnets.{i}")))
            .collect::<Result<Vec<_>>>()?;
        let upsampler = if add_upsample {
            Some(UpSampler::from_weights(
                w,
                &format!("{prefix}.upsamplers.0"),
            )?)
        } else {
            None
        };
        Ok(Self { resnets, upsampler })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = x.clone();
        for resnet in &self.resnets {
            h = resnet.forward(&h)?;
        }
        if let Some(up) = &self.upsampler {
            h = up.forward(&h)?;
        }
        Ok(h)
    }

    pub fn forward_tiled(
        &self,
        x: &Array,
        tile_edge: i32,
        upsampled_tile_edge: i32,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        let mut h = x.clone();
        for resnet in &self.resnets {
            h = resnet.forward_tiled(&h, tile_edge, cancel)?;
        }
        if let Some(up) = &self.upsampler {
            h = up.forward_tiled(&h, upsampled_tile_edge, cancel)?;
        }
        Ok(h)
    }

    pub fn upsamples(&self) -> bool {
        self.upsampler.is_some()
    }
}
