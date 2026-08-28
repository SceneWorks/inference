//! The SDXL VAE **decoder**, ported natively so its tiled route can be normalization-correct
//! (sc-19753).
//!
//! ## Why this exists rather than reusing `candle_transformers`' `AutoEncoderKL`
//!
//! The stock `AutoEncoderKL` exposes exactly one decode entry point and keeps its `Decoder`,
//! `UpDecoderBlock2D` resnets and `GroupNorm`s private. sc-19753's fix requires reaching *inside*
//! the decoder — running the globally-scoped head once and bounding each 3×3 convolution
//! individually while every `GroupNorm` still reduces the whole image. That is unreachable through
//! the upstream surface, and the backend revision is pinned, so the decoder is ported here.
//!
//! **This is a port, not a redesign.** Layer order, key layout, group counts, epsilons, padding and
//! the `output_scale_factor` divide are the upstream ones. The mid block uses the local faithful
//! copy so its spatial attention can take the shared i32-safe budget seam.
//! `tests/vae_decoder_parity.rs` builds this decoder and the stock `AutoEncoderKL` from the same
//! synthetic checkpoint and asserts their dense decodes agree at both f32 and f16 — so a key-naming,
//! ordering, epsilon or dtype divergence fails there rather than silently in a render.
//!
//! Only the decode half is built: nothing in this crate ever called `AutoEncoderKL::encode` (the
//! edit path uses the vendored deterministic `VaeMomentsEncoder` instead, because the stock
//! `sample` is device-RNG and non-portable — sc-3673/sc-6037), so the encoder tower and
//! `quant_conv` are not loaded at all.
//!
//! ## What the bounded route costs
//!
//! [`SdxlVaeDecoder::decode_tiled`] bounds **convolution work**, not every activation: each layer's
//! full-resolution feature map is materialized, and the tiling applies within a layer. That is the
//! same tradeoff the MLX side took in sc-19753, and it is the only shape that keeps GroupNorm
//! statistics global. The previous whole-decode tiling bounded the activation too, but decoded a
//! different image — every tile normalized against its own crop and the mid block attended only to
//! its own tokens.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::{self as nn, Module, VarBuilder};
use candle_gen::gen_core::CancelFlag;
use candle_gen::vae_tiling::{tiled_conv2d_3x3_nchw, GlobalGroupNorm};
use candle_gen::{CandleError, Result};
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKLConfig;

use crate::unet::{UNetMidBlock2D, UNetMidBlock2DConfig};

/// Every `GroupNorm` on the SDXL VAE decode path uses this epsilon — upstream's
/// `UpDecoderBlock2DConfig::resnet_eps`, `UNetMidBlock2DConfig::resnet_eps` and the literal passed
/// to `conv_norm_out`. Note it is **1e-6**, not the U-Net's 1e-5.
const NORM_EPS: f64 = 1e-6;
/// Upstream's `output_scale_factor` for every VAE resnet.
const OUTPUT_SCALE_FACTOR: f64 = 1.0;

fn conv3x3(in_c: usize, out_c: usize, vs: VarBuilder) -> Result<nn::Conv2d> {
    Ok(nn::conv2d(
        in_c,
        out_c,
        3,
        nn::Conv2dConfig {
            padding: 1,
            ..Default::default()
        },
        vs,
    )?)
}

fn conv1x1(in_c: usize, out_c: usize, vs: VarBuilder) -> Result<nn::Conv2d> {
    Ok(nn::conv2d(in_c, out_c, 1, Default::default(), vs)?)
}

/// A `GroupNorm` kept alongside its raw affine parameters, so the same layer can be evaluated
/// densely (upstream's module) or through [`GlobalGroupNorm`] on halo-expanded crops.
struct Norm {
    dense: nn::GroupNorm,
    weight: Tensor,
    bias: Tensor,
    groups: usize,
}

impl Norm {
    fn new(groups: usize, channels: usize, vs: VarBuilder) -> Result<Self> {
        let dense = nn::group_norm(groups, channels, NORM_EPS, vs.clone())?;
        Ok(Self {
            dense,
            weight: vs.get(channels, "weight")?,
            bias: vs.get(channels, "bias")?,
            groups,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(self.dense.forward(xs)?)
    }

    fn global(&self, xs: &Tensor) -> Result<GlobalGroupNorm> {
        GlobalGroupNorm::new(xs, &self.weight, &self.bias, self.groups, NORM_EPS)
    }
}

/// The temb-free VAE flavour of upstream's `ResnetBlock2D`:
/// `norm1 → silu → conv1 → norm2 → silu → conv2`, plus the shortcut, divided by
/// [`OUTPUT_SCALE_FACTOR`].
struct VaeResnetBlock {
    norm1: Norm,
    conv1: nn::Conv2d,
    norm2: Norm,
    conv2: nn::Conv2d,
    shortcut: Option<nn::Conv2d>,
}

impl VaeResnetBlock {
    fn new(vs: VarBuilder, in_c: usize, out_c: usize, groups: usize) -> Result<Self> {
        Ok(Self {
            norm1: Norm::new(groups, in_c, vs.pp("norm1"))?,
            conv1: conv3x3(in_c, out_c, vs.pp("conv1"))?,
            norm2: Norm::new(groups, out_c, vs.pp("norm2"))?,
            conv2: conv3x3(out_c, out_c, vs.pp("conv2"))?,
            // Upstream's `use_in_shortcut` default for a VAE resnet.
            shortcut: if in_c != out_c {
                Some(conv1x1(in_c, out_c, vs.pp("conv_shortcut"))?)
            } else {
                None
            },
        })
    }

    fn shortcut(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.shortcut {
            Some(conv) => Ok(conv.forward(xs)?),
            None => Ok(xs.clone()),
        }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = self.shortcut(xs)?;
        let h = self
            .conv1
            .forward(&nn::ops::silu(&self.norm1.forward(xs)?)?)?;
        let h = self
            .conv2
            .forward(&nn::ops::silu(&self.norm2.forward(&h)?)?)?;
        Ok(((residual + h)? / OUTPUT_SCALE_FACTOR)?)
    }

    /// Normalization-correct bounded forward (sc-19753): both `GroupNorm`s reduce the whole layer
    /// activation and only the two 3×3 convolutions are evaluated on halo-expanded crops. The 1×1
    /// shortcut has no spatial extent, so it runs whole.
    fn forward_tiled(
        &self,
        xs: &Tensor,
        tile_edge: usize,
        cancel: Option<&CancelFlag>,
    ) -> Result<Tensor> {
        let residual = self.shortcut(xs)?;
        let norm1 = self.norm1.global(xs)?;
        let h = tiled_conv2d_3x3_nchw(
            xs,
            tile_edge,
            cancel,
            |tile| Ok(nn::ops::silu(&norm1.apply(tile)?)?),
            |tile| Ok(self.conv1.forward(tile)?),
        )?;
        let norm2 = self.norm2.global(&h)?;
        let h = tiled_conv2d_3x3_nchw(
            &h,
            tile_edge,
            cancel,
            |tile| Ok(nn::ops::silu(&norm2.apply(tile)?)?),
            |tile| Ok(self.conv2.forward(tile)?),
        )?;
        Ok(((residual + h)? / OUTPUT_SCALE_FACTOR)?)
    }
}

/// Upstream's `UpDecoderBlock2D`: a run of resnets and an optional nearest-2× + 3×3 upsampler.
struct UpBlock {
    resnets: Vec<VaeResnetBlock>,
    upsampler: Option<nn::Conv2d>,
}

impl UpBlock {
    fn new(
        vs: VarBuilder,
        in_c: usize,
        out_c: usize,
        num_layers: usize,
        groups: usize,
        add_upsample: bool,
    ) -> Result<Self> {
        let resnets_vs = vs.pp("resnets");
        let resnets = (0..num_layers)
            .map(|i| {
                let in_c = if i == 0 { in_c } else { out_c };
                VaeResnetBlock::new(resnets_vs.pp(i.to_string()), in_c, out_c, groups)
            })
            .collect::<Result<Vec<_>>>()?;
        let upsampler = if add_upsample {
            Some(conv3x3(
                out_c,
                out_c,
                vs.pp("upsamplers").pp("0").pp("conv"),
            )?)
        } else {
            None
        };
        Ok(Self { resnets, upsampler })
    }

    fn upsample(xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        Ok(xs.upsample_nearest2d(2 * h, 2 * w)?)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for resnet in &self.resnets {
            xs = resnet.forward(&xs)?;
        }
        match &self.upsampler {
            Some(conv) => Ok(conv.forward(&Self::upsample(&xs)?)?),
            None => Ok(xs),
        }
    }

    /// `tile_edge` bounds the resnet convolutions at this block's input resolution;
    /// `upsampled_tile_edge` bounds the post-upsample convolution, which runs at twice that.
    fn forward_tiled(
        &self,
        xs: &Tensor,
        tile_edge: usize,
        upsampled_tile_edge: usize,
        cancel: Option<&CancelFlag>,
    ) -> Result<Tensor> {
        let mut xs = xs.clone();
        for resnet in &self.resnets {
            xs = resnet.forward_tiled(&xs, tile_edge, cancel)?;
        }
        match &self.upsampler {
            Some(conv) => {
                let up = Self::upsample(&xs)?;
                tiled_conv2d_3x3_nchw(
                    &up,
                    upsampled_tile_edge,
                    cancel,
                    |tile| Ok(tile.clone()),
                    |tile| Ok(conv.forward(tile)?),
                )
            }
            None => Ok(xs),
        }
    }

    fn upsamples(&self) -> bool {
        self.upsampler.is_some()
    }
}

/// The SDXL VAE decode path: `post_quant_conv` → `conv_in` → mid block → `up_blocks` →
/// `conv_norm_out` → silu → `conv_out`.
pub struct SdxlVaeDecoder {
    post_quant_conv: Option<nn::Conv2d>,
    conv_in: nn::Conv2d,
    mid_block: UNetMidBlock2D,
    up_blocks: Vec<UpBlock>,
    norm_out: Norm,
    conv_out: nn::Conv2d,
}

impl SdxlVaeDecoder {
    /// Build from a `VarBuilder` rooted at the VAE checkpoint — the same root, and the same keys,
    /// upstream's `AutoEncoderKL::new` reads.
    pub fn new(vs: VarBuilder, out_channels: usize, config: &AutoEncoderKLConfig) -> Result<Self> {
        let post_quant_conv = if config.use_post_quant_conv {
            Some(conv1x1(
                config.latent_channels,
                config.latent_channels,
                vs.pp("post_quant_conv"),
            )?)
        } else {
            None
        };
        let vs = vs.pp("decoder");
        let groups = config.norm_num_groups;
        let channels = &config.block_out_channels;
        let last = *channels
            .last()
            .ok_or_else(|| CandleError::Msg("sdxl vae: empty block_out_channels".into()))?;
        let conv_in = conv3x3(config.latent_channels, last, vs.pp("conv_in"))?;
        let mid_block = UNetMidBlock2D::new(
            vs.pp("mid_block"),
            last,
            None,
            UNetMidBlock2DConfig {
                resnet_eps: NORM_EPS,
                output_scale_factor: OUTPUT_SCALE_FACTOR,
                attn_num_head_channels: None,
                resnet_groups: Some(groups),
                ..Default::default()
            },
        )?;
        let reversed: Vec<usize> = channels.iter().copied().rev().collect();
        let up_vs = vs.pp("up_blocks");
        let up_blocks = (0..reversed.len())
            .map(|index| {
                let out_c = reversed[index];
                let in_c = if index > 0 {
                    reversed[index - 1]
                } else {
                    reversed[0]
                };
                UpBlock::new(
                    up_vs.pp(index.to_string()),
                    in_c,
                    out_c,
                    config.layers_per_block + 1,
                    groups,
                    index + 1 != reversed.len(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            post_quant_conv,
            conv_in,
            mid_block,
            up_blocks,
            norm_out: Norm::new(groups, channels[0], vs.pp("conv_norm_out"))?,
            conv_out: conv3x3(channels[0], out_channels, vs.pp("conv_out"))?,
        })
    }

    /// Build from a `.safetensors` VAE checkpoint at `dtype`.
    pub fn from_file(
        path: &std::path::Path,
        device: &Device,
        dtype: DType,
        config: &AutoEncoderKLConfig,
    ) -> Result<Self> {
        let files = [path.to_path_buf()];
        Self::new(
            candle_gen::mmap_var_builder(&files, dtype, device)?,
            3,
            config,
        )
    }

    /// The **globally-scoped** half: `post_quant_conv` → `conv_in` → mid block, all at latent
    /// resolution. The mid block's `AttentionBlock` attends over every `H·W` latent token, so its
    /// result depends on the whole grid; splitting across it would silently change the arithmetic.
    /// It is also cheap to run whole — at 1024² it is a `[1, 512, 128, 128]` activation.
    fn head(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = match &self.post_quant_conv {
            None => xs.clone(),
            Some(conv) => conv.forward(xs)?,
        };
        Ok(self.mid_block.forward(&self.conv_in.forward(&xs)?, None)?)
    }

    /// Decode an already-unscaled latent `[B, latent_channels, h, w]` → `[B, 3, h·8, w·8]`.
    ///
    /// Layer-for-layer identical to `AutoEncoderKL::decode`; `tests/vae_decoder_parity.rs` pins that.
    pub fn decode(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.head(xs)?;
        for up in &self.up_blocks {
            xs = up.forward(&xs)?;
        }
        Ok(self
            .conv_out
            .forward(&nn::ops::silu(&self.norm_out.forward(&xs)?)?)?)
    }

    /// Bounded decode with dense-image GroupNorm semantics (sc-19753).
    ///
    /// The head runs once on the whole latent; in the tail every `GroupNorm` reduces the full layer
    /// activation and only halo-expanded 3×3 convolution work is tiled. `output_tile_edge` is
    /// expressed in **output** pixels — each stage's convolution edge is that bound divided by the
    /// upsampling still ahead of it, so the physical crop stays comparable at every resolution.
    pub fn decode_tiled(
        &self,
        xs: &Tensor,
        output_tile_edge: usize,
        cancel: Option<&CancelFlag>,
    ) -> Result<Tensor> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(CandleError::Canceled);
        }
        let tile_edge_at = |remaining: usize| output_tile_edge.div_ceil(remaining.max(1)).max(3);
        let mut remaining = self
            .up_blocks
            .iter()
            .filter(|block| block.upsamples())
            .fold(1usize, |scale, _| scale * 2);
        let mut xs = self.head(xs)?;
        for up in &self.up_blocks {
            let current = tile_edge_at(remaining);
            let next = if up.upsamples() {
                (remaining / 2).max(1)
            } else {
                remaining
            };
            xs = up.forward_tiled(&xs, current, tile_edge_at(next), cancel)?;
            remaining = next;
        }
        if remaining != 1 {
            return Err(CandleError::Msg(format!(
                "sdxl tiled VAE tail ended at remaining spatial scale {remaining}, expected 1"
            )));
        }
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(CandleError::Canceled);
        }
        let norm = self.norm_out.global(&xs)?;
        tiled_conv2d_3x3_nchw(
            &xs,
            output_tile_edge.max(3),
            cancel,
            |tile| Ok(nn::ops::silu(&norm.apply(tile)?)?),
            |tile| Ok(self.conv_out.forward(tile)?),
        )
    }
}
