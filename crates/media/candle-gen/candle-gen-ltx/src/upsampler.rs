//! Learned LTX 2x latent upsampler.  This is deliberately a real checkpoint
//! component, not an interpolation fallback: stage two is valid only after this
//! network has transformed the half-resolution stage-one latent.

use candle_gen::candle_core::{DType, Result, Tensor};
use candle_gen::candle_nn::VarBuilder;

use crate::conv3d::{chunked_conv2d, ZeroPaddedConv3d};
use crate::quant::guard_no_scales;

const GROUPS: usize = 32;
const EPS: f64 = 1e-5;

struct Conv2d {
    weight: Tensor,
    bias: Tensor,
}

impl Conv2d {
    fn load(vb: VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: guard_no_scales(&vb, prefix, vb.dtype())?.contiguous()?,
            bias: vb.get_unchecked(&format!("{prefix}.bias"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = chunked_conv2d(x, &self.weight, 1)?;
        y.broadcast_add(&self.bias.reshape((1, self.bias.elem_count(), 1, 1))?)
    }
}

/// Checkpoint GroupNorm(32), evaluated in f32 across channel-group, time and
/// spatial axes before returning the input dtype.
struct GroupNorm32 {
    weight: Tensor,
    bias: Tensor,
}

impl GroupNorm32 {
    fn load(vb: VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: vb.get_unchecked(&format!("{prefix}.weight"))?,
            bias: vb.get_unchecked(&format!("{prefix}.bias"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let (b, c, t, h, w) = x.dims5()?;
        if c % GROUPS != 0 {
            return Err(candle_gen::candle_core::Error::Msg(format!(
                "ltx upsampler: GroupNorm32 requires channels divisible by {GROUPS}, got {c}"
            )));
        }
        let groups = x
            .to_dtype(DType::F32)?
            .reshape((b, GROUPS, c / GROUPS, t, h, w))?;
        let mean = groups.mean_keepdim((2, 3, 4, 5))?;
        let centered = groups.broadcast_sub(&mean)?;
        let variance = centered.sqr()?.mean_keepdim((2, 3, 4, 5))?;
        let normalized = (centered / (variance + EPS)?.sqrt()?)?.reshape((b, c, t, h, w))?;
        let weight = self.weight.to_dtype(DType::F32)?.reshape((1, c, 1, 1, 1))?;
        let bias = self.bias.to_dtype(DType::F32)?.reshape((1, c, 1, 1, 1))?;
        ((normalized.broadcast_mul(&weight)?).broadcast_add(&bias)?).to_dtype(dtype)
    }
}

struct ResBlock {
    conv1: ZeroPaddedConv3d,
    norm1: GroupNorm32,
    conv2: ZeroPaddedConv3d,
    norm2: GroupNorm32,
}

impl ResBlock {
    fn load(vb: VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv1: ZeroPaddedConv3d::load(vb.clone(), &format!("{prefix}.conv1"))?,
            norm1: GroupNorm32::load(vb.clone(), &format!("{prefix}.norm1"))?,
            conv2: ZeroPaddedConv3d::load(vb.clone(), &format!("{prefix}.conv2"))?,
            norm2: GroupNorm32::load(vb, &format!("{prefix}.norm2"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let y = self.norm1.forward(&self.conv1.forward(x)?)?;
        let y = candle_gen::candle_nn::ops::silu(&y)?;
        let y = self.conv2.forward(&y)?;
        let y = self.norm2.forward(&y)?;
        candle_gen::candle_nn::ops::silu(&(y + residual)?)
    }
}

/// NCHW PixelShuffle with the PyTorch channel order used by the upsampler.
pub(crate) fn pixel_shuffle2d(x: &Tensor, upscale: usize) -> Result<Tensor> {
    let (n, channels, h, w) = x.dims4()?;
    let divisor = upscale * upscale;
    if channels % divisor != 0 {
        return Err(candle_gen::candle_core::Error::Msg(format!(
            "ltx upsampler: PixelShuffle({upscale}) cannot divide {channels} channels"
        )));
    }
    let out_channels = channels / divisor;
    x.reshape((n, out_channels, upscale, upscale, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .reshape((n, out_channels, h * upscale, w * upscale))?
        .contiguous()
}

struct SpatialResampler {
    conv: Conv2d,
}

impl SpatialResampler {
    fn load(vb: VarBuilder, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv: Conv2d::load(vb, prefix)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, t, h, w) = x.dims5()?;
        let frames = x
            .permute((0, 2, 1, 3, 4))?
            .reshape((b * t, c, h, w))?
            .contiguous()?;
        let up = pixel_shuffle2d(&self.conv.forward(&frames)?, 2)?;
        up.reshape((b, t, c, h * 2, w * 2))?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()
    }
}

/// LTX's learned `LatentUpsampler`: 128→1024 Conv3d, four residual blocks,
/// framewise 1024→4096 Conv2d + PixelShuffle2d, four residual blocks, 1024→128
/// Conv3d. Names match the published `upsampler.safetensors` layout exactly.
pub(crate) struct LatentUpsampler {
    initial_conv: ZeroPaddedConv3d,
    initial_norm: GroupNorm32,
    res_blocks: Vec<ResBlock>,
    spatial: SpatialResampler,
    post_res_blocks: Vec<ResBlock>,
    final_conv: ZeroPaddedConv3d,
}

impl LatentUpsampler {
    pub(crate) fn load(vb: VarBuilder) -> Result<Self> {
        let res_blocks = (0..4)
            .map(|i| ResBlock::load(vb.clone(), &format!("res_blocks.{i}")))
            .collect::<Result<Vec<_>>>()?;
        let post_res_blocks = (0..4)
            .map(|i| ResBlock::load(vb.clone(), &format!("post_upsample_res_blocks.{i}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            initial_conv: ZeroPaddedConv3d::load(vb.clone(), "initial_conv")?,
            initial_norm: GroupNorm32::load(vb.clone(), "initial_norm")?,
            res_blocks,
            spatial: SpatialResampler::load(vb.clone(), "upsampler.0")?,
            post_res_blocks,
            final_conv: ZeroPaddedConv3d::load(vb, "final_conv")?,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = candle_gen::candle_nn::ops::silu(
            &self.initial_norm.forward(&self.initial_conv.forward(x)?)?,
        )?;
        for block in &self.res_blocks {
            y = block.forward(&y)?;
        }
        y = self.spatial.forward(&y)?;
        for block in &self.post_res_blocks {
            y = block.forward(&y)?;
        }
        self.final_conv.forward(&y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    #[test]
    fn pixel_shuffle_preserves_pytorch_channel_mapping() -> Result<()> {
        let x = Tensor::arange(0f32, 16f32, &Device::Cpu)?.reshape((1, 4, 2, 2))?;
        let y = pixel_shuffle2d(&x, 2)?;
        assert_eq!(y.dims(), &[1, 1, 4, 4]);
        assert_eq!(
            y.flatten_all()?.to_vec1::<f32>()?,
            vec![0., 4., 1., 5., 8., 12., 9., 13., 2., 6., 3., 7., 10., 14., 11., 15.]
        );
        Ok(())
    }
}
