//! ResNet Building Blocks
//!
//! Some Residual Network blocks used in UNet models.
//!
//! Denoising Diffusion Implicit Models, K. He and al, 2015.
//! - [Paper](https://arxiv.org/abs/1512.03385)
//!
use super::conv::{conv2d, Conv2d};
use candle_core::{Result, Tensor, D};
use candle_gen::train::lora::{lora_linear_detect, LoraLinear};
use candle_nn as nn;
use candle_nn::Module;

/// Configuration for a ResNet block.
#[derive(Debug, Clone, Copy)]
pub struct ResnetBlock2DConfig {
    /// The number of output channels, defaults to the number of input channels.
    pub out_channels: Option<usize>,
    pub temb_channels: Option<usize>,
    /// The number of groups to use in group normalization.
    pub groups: usize,
    pub groups_out: Option<usize>,
    /// The epsilon to be used in the group normalization operations.
    pub eps: f64,
    /// Whether to use a 2D convolution in the skip connection. When using None,
    /// such a convolution is used if the number of input channels is different from
    /// the number of output channels.
    pub use_in_shortcut: Option<bool>,
    // non_linearity: silu
    /// The final output is scaled by dividing by this value.
    pub output_scale_factor: f64,
}

impl Default for ResnetBlock2DConfig {
    fn default() -> Self {
        Self {
            out_channels: None,
            temb_channels: Some(512),
            groups: 32,
            groups_out: None,
            eps: 1e-6,
            use_in_shortcut: None,
            output_scale_factor: 1.,
        }
    }
}

#[derive(Debug)]
pub struct ResnetBlock2D {
    norm1: nn::GroupNorm,
    conv1: Conv2d,
    norm2: nn::GroupNorm,
    conv2: Conv2d,
    // sc-9416: the `time_emb_proj` Linear packed-detects (the MLX SDXL tiers pack
    // `resnets.*.time_emb_proj`); the surrounding convs/norms stay dense. Dense checkpoints have no
    // `.scales` sibling, so `lora_linear_detect` takes the plain dense path unchanged.
    // sc-11679: a `LoraLinear` (frozen packed/dense base + optional forward-time additive residual) so a
    // distill LoRA targeting `time_emb_proj` rides it additively on a packed tier — byte-identical with
    // no residual (the packed q4/q8 base is never dequantized).
    time_emb_proj: Option<LoraLinear>,
    conv_shortcut: Option<Conv2d>,
    span: tracing::Span,
    config: ResnetBlock2DConfig,
}

impl ResnetBlock2D {
    pub fn new(
        vs: nn::VarBuilder,
        in_channels: usize,
        config: ResnetBlock2DConfig,
    ) -> Result<Self> {
        Self::new_gs(vs, in_channels, config, candle_gen::quant::MLX_GROUP_SIZE)
    }

    /// As [`new`](Self::new), but at an explicit MLX packed `group_size` (sc-9416) for the
    /// packed-detecting `time_emb_proj`.
    pub fn new_gs(
        vs: nn::VarBuilder,
        in_channels: usize,
        config: ResnetBlock2DConfig,
        group_size: usize,
    ) -> Result<Self> {
        let out_channels = config.out_channels.unwrap_or(in_channels);
        let conv_cfg = nn::Conv2dConfig {
            stride: 1,
            padding: 1,
            groups: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        };
        let norm1 = nn::group_norm(config.groups, in_channels, config.eps, vs.pp("norm1"))?;
        let conv1 = conv2d(in_channels, out_channels, 3, conv_cfg, vs.pp("conv1"))?;
        let groups_out = config.groups_out.unwrap_or(config.groups);
        let norm2 = nn::group_norm(groups_out, out_channels, config.eps, vs.pp("norm2"))?;
        let conv2 = conv2d(out_channels, out_channels, 3, conv_cfg, vs.pp("conv2"))?;
        let use_in_shortcut = config
            .use_in_shortcut
            .unwrap_or(in_channels != out_channels);
        let conv_shortcut = if use_in_shortcut {
            let conv_cfg = nn::Conv2dConfig {
                stride: 1,
                padding: 0,
                groups: 1,
                dilation: 1,
                cudnn_fwd_algo: None,
            };
            Some(conv2d(
                in_channels,
                out_channels,
                1,
                conv_cfg,
                vs.pp("conv_shortcut"),
            )?)
        } else {
            None
        };
        let time_emb_proj = match config.temb_channels {
            None => None,
            Some(temb_channels) => Some(lora_linear_detect(
                temb_channels,
                out_channels,
                &vs,
                "time_emb_proj",
                group_size,
            )?),
        };
        let span = tracing::span!(tracing::Level::TRACE, "resnet2d");
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            time_emb_proj,
            span,
            config,
            conv_shortcut,
        })
    }

    pub fn forward(&self, xs: &Tensor, temb: Option<&Tensor>) -> Result<Tensor> {
        let _enter = self.span.enter();
        let shortcut_xs = match &self.conv_shortcut {
            Some(conv_shortcut) => conv_shortcut.forward(xs)?,
            None => xs.clone(),
        };
        let xs = self.norm1.forward(xs)?;
        let xs = self.conv1.forward(&nn::ops::silu(&xs)?)?;
        let xs = match (temb, &self.time_emb_proj) {
            (Some(temb), Some(time_emb_proj)) => time_emb_proj
                .forward(&nn::ops::silu(temb)?)?
                .unsqueeze(D::Minus1)?
                .unsqueeze(D::Minus1)?
                .broadcast_add(&xs)?,
            _ => xs,
        };
        let xs = self
            .conv2
            .forward(&nn::ops::silu(&self.norm2.forward(&xs)?)?)?;
        (shortcut_xs + xs)? / self.config.output_scale_factor
    }

    /// Visit this resnet's adaptable **Linear** (`time_emb_proj`, when present) so a LoRA targeting the
    /// resnet timestep projection can install a forward-time additive residual on it (sc-11679). The
    /// convolutions are the conv-LoRA path's job ([`visit_conv_lora_mut`](Self::visit_conv_lora_mut)).
    pub fn visit_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut LoraLinear) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        if let Some(time_emb_proj) = &mut self.time_emb_proj {
            f(time_emb_proj)?;
        }
        Ok(())
    }

    /// Visit this resnet's convolutions (`conv1`, `conv2`, and the optional `conv_shortcut`) so a
    /// conv-layer LoRA can install additive residuals on them (sc-11682 — keeps the dense base a
    /// pristine mmap instead of folding). The `time_emb_proj` Linear is NOT a conv target.
    pub fn visit_conv_lora_mut(
        &mut self,
        f: &mut dyn FnMut(&mut Conv2d) -> candle_gen::Result<()>,
    ) -> candle_gen::Result<()> {
        f(&mut self.conv1)?;
        f(&mut self.conv2)?;
        if let Some(conv_shortcut) = &mut self.conv_shortcut {
            f(conv_shortcut)?;
        }
        Ok(())
    }
}
