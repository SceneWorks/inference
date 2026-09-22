//! The Qwen-Image 2.1 **RGBA** autoencoder — a port of diffusers' `AutoencoderKLQwenImage21`
//! (`autoencoder_kl_qwenimage21.py` @ [`crate::UPSTREAM_DIFFUSERS_REVISION`]) for the single-frame
//! image case it is used in.
//!
//! Architecture (the Wan 2.2 residual VAE family, image-specialised): 4-channel RGBA in/out,
//! `z_dim = 64`, five stages `dim_mult [1, 2, 4, 8, 8]` → **16×** spatial compression, encoder
//! width 96 / decoder width 144, two residual blocks per stage, one single-head attention block in
//! each mid block, and **parameter-free** stage shortcuts (`AvgDown3D` channel-group means on the
//! way down, `DupUp3D` channel duplication on the way up). Every convolution is 2-D: upstream's
//! `QwenImage21CausalConv3d` folds the single frame away and refuses a temporal cache, and the
//! temporal `time_conv`s of the 3-D resample stages never run on the first (only) frame — their
//! weights exist in the checkpoint and are deliberately not loaded.
//!
//! Layout is channels-last `[B, H, W, C]` internally (mlx convolutions are NHWC); the public
//! entry points take and return the NCHW `[1, C, H, W]` tensors the rest of the pipeline uses.
//! Compute runs in the weight dtype (bf16 released, f32 fixtures); the channel-L2 `RMS_norm`
//! normalises in f32 as upstream does for half-precision inputs (`eps = 1e-12`).
//!
//! Latent normalisation (`latents_mean` / `latents_std`) is **not** applied here — the pipeline
//! owns it, exactly as upstream's `QwenImage21Pipeline` does around `vae.encode` / `vae.decode`.

use mlx_gen::gen_core::LatentSpace;
use mlx_gen::nn::{conv2d, silu, upsample_nearest};
use mlx_gen::tiling::{TilingConfig, VaeTiling};
use mlx_gen::vae_tiling::tiled_decode;
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, LatentDecoder, Result};
use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{
    add, clip, concatenate_axis, divide, maximum, mean_axis, multiply, pad, split, sum_axis,
    zeros_like,
};
use mlx_rs::{Array, Dtype};

use crate::config::VaeConfig;

/// `F.normalize` floor.
const NORM_EPS: f32 = 1e-12;

fn key(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// A 2-D convolution loaded from a torch `[out, in, kh, kw]` weight (+ bias).
struct Conv {
    weight: Array,
    bias: Option<Array>,
    stride: i32,
    padding: i32,
}

impl Conv {
    fn from_weights(w: &Weights, base: &str, stride: i32, padding: i32) -> Result<Self> {
        let weight = w.require(&format!("{base}.weight"))?;
        let rank = weight.shape().len();
        // The 3-D `time_conv` weights are `[out, in, kt, kh, kw]`; a 2-D conv must be rank 4.
        if rank != 4 {
            return Err(Error::Msg(format!(
                "qwen_image_2_1 vae: {base}.weight has rank {rank}, expected a 2-D conv"
            )));
        }
        Ok(Self {
            weight: weight.transpose_axes(&[0, 2, 3, 1])?,
            bias: w.get(&format!("{base}.bias")).cloned(),
            stride,
            padding,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        conv2d(
            x,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
        )
    }
}

/// `QwenImage21RMS_norm`: channel-L2 normalisation `x / max(‖x‖₂, 1e-12) · √C · γ`, f32.
struct ChannelNorm {
    gamma: Array,
    scale: f32,
}

impl ChannelNorm {
    fn from_weights(w: &Weights, base: &str) -> Result<Self> {
        let gamma = w.require(&format!("{base}.gamma"))?;
        let channels = gamma.shape()[0];
        Ok(Self {
            gamma: gamma.reshape(&[channels])?,
            scale: (channels as f32).sqrt(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let dtype = x.dtype();
        let x32 = x.as_dtype(Dtype::Float32)?;
        let norm = sum_axis(&multiply(&x32, &x32)?, -1, true)?.sqrt()?;
        let normalized =
            divide(&x32, &maximum(&norm, Array::from_f32(NORM_EPS))?)?.as_dtype(dtype)?;
        let gamma = multiply(
            &self.gamma.as_dtype(dtype)?,
            Array::from_f32(self.scale).as_dtype(dtype)?,
        )?;
        Ok(multiply(&normalized, &gamma)?)
    }
}

struct ResidualBlock {
    norm1: ChannelNorm,
    conv1: Conv,
    norm2: ChannelNorm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

impl ResidualBlock {
    fn from_weights(w: &Weights, base: &str) -> Result<Self> {
        let shortcut = if w.get(&format!("{base}.conv_shortcut.weight")).is_some() {
            Some(Conv::from_weights(
                w,
                &format!("{base}.conv_shortcut"),
                1,
                0,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: ChannelNorm::from_weights(w, &format!("{base}.norm1"))?,
            conv1: Conv::from_weights(w, &format!("{base}.conv1"), 1, 1)?,
            norm2: ChannelNorm::from_weights(w, &format!("{base}.norm2"))?,
            conv2: Conv::from_weights(w, &format!("{base}.conv2"), 1, 1)?,
            shortcut,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = match &self.shortcut {
            Some(conv) => conv.forward(x)?,
            None => x.clone(),
        };
        let y = self.conv1.forward(&silu(&self.norm1.forward(x)?)?)?;
        let y = self.conv2.forward(&silu(&self.norm2.forward(&y)?)?)?;
        Ok(add(&y, &h)?)
    }
}

/// `QwenImage21AttentionBlock`: single-head self-attention over the spatial positions.
struct AttentionBlock {
    norm: ChannelNorm,
    to_qkv: Conv,
    proj: Conv,
}

impl AttentionBlock {
    fn from_weights(w: &Weights, base: &str) -> Result<Self> {
        Ok(Self {
            norm: ChannelNorm::from_weights(w, &format!("{base}.norm"))?,
            to_qkv: Conv::from_weights(w, &format!("{base}.to_qkv"), 1, 0)?,
            proj: Conv::from_weights(w, &format!("{base}.proj"), 1, 0)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape().to_vec();
        let (b, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
        let qkv = self
            .to_qkv
            .forward(&self.norm.forward(x)?)?
            .reshape(&[b, 1, h * w, 3 * c])?;
        let parts = split(&qkv, 3, 3)?;
        let scale = (c as f32).powf(-0.5);
        let attended =
            scaled_dot_product_attention(&parts[0], &parts[1], &parts[2], scale, None, None)?;
        let attended = self.proj.forward(&attended.reshape(&[b, h, w, c])?)?;
        Ok(add(&attended, x)?)
    }
}

struct MidBlock {
    first: ResidualBlock,
    attention: AttentionBlock,
    second: ResidualBlock,
}

impl MidBlock {
    fn from_weights(w: &Weights, base: &str) -> Result<Self> {
        Ok(Self {
            first: ResidualBlock::from_weights(w, &format!("{base}.resnets.0"))?,
            attention: AttentionBlock::from_weights(w, &format!("{base}.attentions.0"))?,
            second: ResidualBlock::from_weights(w, &format!("{base}.resnets.1"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.first.forward(x)?;
        let x = self.attention.forward(&x)?;
        self.second.forward(&x)
    }
}

/// The learned half of a resample stage: nearest ×2 then a 3×3 conv (up), or a right/bottom zero
/// pad then a stride-2 3×3 conv (down). The temporal `time_conv` of the 3-D variants never runs on
/// a single frame.
struct Resample {
    up: bool,
    conv: Conv,
}

impl Resample {
    fn from_weights(w: &Weights, base: &str, up: bool) -> Result<Self> {
        let conv = if up {
            Conv::from_weights(w, &format!("{base}.resample.1"), 1, 1)?
        } else {
            Conv::from_weights(w, &format!("{base}.resample.1"), 2, 0)?
        };
        Ok(Self { up, conv })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        if self.up {
            self.conv.forward(&upsample_nearest(x, 2)?)
        } else {
            let padded = pad(x, &[(0, 0), (0, 1), (0, 1), (0, 0)][..], None, None)?;
            self.conv.forward(&padded)
        }
    }
}

/// `QwenImage21AvgDown3D` on a single frame: group-mean shortcut across the folded
/// `(channel, [zero-padded frame], row-offset, col-offset)` axis.
struct AvgDown {
    out_channels: i32,
    temporal: bool,
    spatial: bool,
}

impl AvgDown {
    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape().to_vec();
        let (b, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
        let fs = if self.spatial { 2 } else { 1 };
        let (h2, w2) = (h / fs, w / fs);
        // [B, H', fs, W', fs, C] → [B, H', W', C, fs, fs] → flat (c, i, j).
        let folded = x
            .reshape(&[b, h2, fs, w2, fs, c])?
            .transpose_axes(&[0, 1, 3, 5, 2, 4])?
            .reshape(&[b, h2, w2, c, fs * fs])?;
        let folded = if self.temporal {
            // `pad_t` prepends one all-zero frame: flat (c, a, i, j) with a = 0 the zero frame.
            let real = folded.reshape(&[b, h2, w2, c, 1, fs * fs])?;
            let zero = zeros_like(&real)?;
            concatenate_axis(&[&zero, &real], 4)?.reshape(&[b, h2, w2, c * 2 * fs * fs])?
        } else {
            folded.reshape(&[b, h2, w2, c * fs * fs])?
        };
        let total = folded.shape()[3];
        let group = total / self.out_channels;
        let grouped = folded.reshape(&[b, h2, w2, self.out_channels, group])?;
        Ok(mean_axis(&grouped, 4, false)?)
    }
}

/// `QwenImage21DupUp3D` on the first (only) frame: channel duplication into a 2×2 spatial expansion.
struct DupUp {
    in_channels: i32,
    out_channels: i32,
    temporal: bool,
}

impl DupUp {
    fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape().to_vec();
        let (b, h, w) = (shape[0], shape[1], shape[2]);
        let (ft, fs) = (if self.temporal { 2 } else { 1 }, 2);
        let repeats = self.out_channels * ft * fs * fs / self.in_channels;
        // Output (o, row-offset i, col-offset j) reads the duplicated channel index
        // (((o·ft + (ft−1))·fs + i)·fs + j) / repeats — `first_chunk` keeps the last temporal slot.
        let mut index = Vec::with_capacity((self.out_channels * fs * fs) as usize);
        for o in 0..self.out_channels {
            for i in 0..fs {
                for j in 0..fs {
                    let flat = ((o * ft + (ft - 1)) * fs + i) * fs + j;
                    index.push(flat / repeats);
                }
            }
        }
        let index = Array::from_slice(&index, &[index.len() as i32]);
        let gathered = x
            .take_axis(&index, 3)?
            .reshape(&[b, h, w, self.out_channels, fs, fs])?
            .transpose_axes(&[0, 1, 4, 2, 5, 3])?
            .reshape(&[b, h * fs, w * fs, self.out_channels])?;
        Ok(gathered)
    }
}

struct DownBlock {
    resnets: Vec<ResidualBlock>,
    downsampler: Option<Resample>,
    shortcut: AvgDown,
}

impl DownBlock {
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut y = x.clone();
        for block in &self.resnets {
            y = block.forward(&y)?;
        }
        if let Some(down) = &self.downsampler {
            y = down.forward(&y)?;
        }
        Ok(add(&y, &self.shortcut.forward(x)?)?)
    }
}

struct UpBlock {
    resnets: Vec<ResidualBlock>,
    upsampler: Option<Resample>,
    shortcut: Option<DupUp>,
}

impl UpBlock {
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut y = x.clone();
        for block in &self.resnets {
            y = block.forward(&y)?;
        }
        if let Some(up) = &self.upsampler {
            y = up.forward(&y)?;
        }
        match &self.shortcut {
            Some(dup) => Ok(add(&y, &dup.forward(x)?)?),
            None => Ok(y),
        }
    }
}

struct Encoder {
    conv_in: Conv,
    down_blocks: Vec<DownBlock>,
    mid: MidBlock,
    norm_out: ChannelNorm,
    conv_out: Conv,
}

impl Encoder {
    fn from_weights(w: &Weights, base: &str, cfg: &VaeConfig) -> Result<Self> {
        let dims: Vec<usize> = std::iter::once(1)
            .chain(cfg.dim_mult.iter().copied())
            .map(|m| cfg.base_dim * m)
            .collect();
        let last = cfg.dim_mult.len() - 1;
        let mut down_blocks = Vec::with_capacity(cfg.dim_mult.len());
        for i in 0..cfg.dim_mult.len() {
            let prefix = key(base, &format!("down_blocks.{i}"));
            let down = i != last;
            let temporal = down && cfg.temperal_downsample[i];
            let resnets = (0..cfg.num_res_blocks)
                .map(|j| ResidualBlock::from_weights(w, &format!("{prefix}.resnets.{j}")))
                .collect::<Result<Vec<_>>>()?;
            let downsampler = if down {
                Some(Resample::from_weights(
                    w,
                    &format!("{prefix}.downsampler"),
                    false,
                )?)
            } else {
                None
            };
            let (in_dim, out_dim) = (dims[i] as i32, dims[i + 1] as i32);
            let factor = if temporal { 2 } else { 1 } * if down { 4 } else { 1 };
            if (in_dim * factor) % out_dim != 0 {
                return Err(Error::Msg(format!(
                    "qwen_image_2_1 vae: encoder stage {i}: {in_dim}·{factor} is not divisible by {out_dim}"
                )));
            }
            down_blocks.push(DownBlock {
                resnets,
                downsampler,
                shortcut: AvgDown {
                    out_channels: out_dim,
                    temporal,
                    spatial: down,
                },
            });
        }
        Ok(Self {
            conv_in: Conv::from_weights(w, &key(base, "conv_in"), 1, 1)?,
            down_blocks,
            mid: MidBlock::from_weights(w, &key(base, "mid_block"))?,
            norm_out: ChannelNorm::from_weights(w, &key(base, "norm_out"))?,
            conv_out: Conv::from_weights(w, &key(base, "conv_out"), 1, 1)?,
        })
    }

    fn forward(&self, x: &Array, trace: &mut Trace<'_>) -> Result<Array> {
        let mut x = self.conv_in.forward(x)?;
        trace.push("encoder/conv_in", &x)?;
        for (i, block) in self.down_blocks.iter().enumerate() {
            x = block.forward(&x)?;
            trace.push(format!("encoder/down_block_{i}"), &x)?;
        }
        let x = self.mid.forward(&x)?;
        trace.push("encoder/mid_block", &x)?;
        let out = self.conv_out.forward(&silu(&self.norm_out.forward(&x)?)?)?;
        trace.push("encoder/conv_out", &out)?;
        Ok(out)
    }
}

struct Decoder {
    conv_in: Conv,
    mid: MidBlock,
    up_blocks: Vec<UpBlock>,
    norm_out: ChannelNorm,
    conv_out: Conv,
}

impl Decoder {
    fn from_weights(w: &Weights, base: &str, cfg: &VaeConfig) -> Result<Self> {
        let dims: Vec<usize> = std::iter::once(*cfg.dim_mult.last().unwrap_or(&1))
            .chain(cfg.dim_mult.iter().rev().copied())
            .map(|m| cfg.decoder_base_dim * m)
            .collect();
        let temporal_upsample: Vec<bool> = cfg.temperal_downsample.iter().rev().copied().collect();
        let last = cfg.dim_mult.len() - 1;
        let mut up_blocks = Vec::with_capacity(cfg.dim_mult.len());
        for i in 0..cfg.dim_mult.len() {
            let prefix = key(base, &format!("up_blocks.{i}"));
            let up = i != last;
            let temporal = up && temporal_upsample[i];
            let resnets = (0..=cfg.num_res_blocks)
                .map(|j| ResidualBlock::from_weights(w, &format!("{prefix}.resnets.{j}")))
                .collect::<Result<Vec<_>>>()?;
            let (in_dim, out_dim) = (dims[i] as i32, dims[i + 1] as i32);
            let (upsampler, shortcut) = if up {
                let factor = if temporal { 2 } else { 1 } * 4;
                if (out_dim * factor) % in_dim != 0 {
                    return Err(Error::Msg(format!(
                        "qwen_image_2_1 vae: decoder stage {i}: {out_dim}·{factor} is not divisible by {in_dim}"
                    )));
                }
                (
                    Some(Resample::from_weights(
                        w,
                        &format!("{prefix}.upsampler"),
                        true,
                    )?),
                    Some(DupUp {
                        in_channels: in_dim,
                        out_channels: out_dim,
                        temporal,
                    }),
                )
            } else {
                (None, None)
            };
            up_blocks.push(UpBlock {
                resnets,
                upsampler,
                shortcut,
            });
        }
        Ok(Self {
            conv_in: Conv::from_weights(w, &key(base, "conv_in"), 1, 1)?,
            mid: MidBlock::from_weights(w, &key(base, "mid_block"))?,
            up_blocks,
            norm_out: ChannelNorm::from_weights(w, &key(base, "norm_out"))?,
            conv_out: Conv::from_weights(w, &key(base, "conv_out"), 1, 1)?,
        })
    }

    fn forward(&self, x: &Array, trace: &mut Trace<'_>) -> Result<Array> {
        let head = self.forward_head(x, trace)?;
        self.forward_tail(&head, trace)
    }

    /// The **global** head: `conv_in` → mid block (whose single-head attention spans the whole
    /// latent). Runs once on the full latent; cheap at latent resolution.
    fn forward_head(&self, x: &Array, trace: &mut Trace<'_>) -> Result<Array> {
        let x = self.conv_in.forward(x)?;
        trace.push("decoder/conv_in", &x)?;
        let x = self.mid.forward(&x)?;
        trace.push("decoder/mid_block", &x)?;
        Ok(x)
    }

    /// The **spatially local** upsample tail: `up_blocks` → `norm_out` → SiLU → `conv_out`. Every
    /// op is a per-pixel norm, a 3×3 conv, a nearest ×2 or a channel shuffle, so it tiles.
    fn forward_tail(&self, head: &Array, trace: &mut Trace<'_>) -> Result<Array> {
        let mut x = head.clone();
        for (i, block) in self.up_blocks.iter().enumerate() {
            x = block.forward(&x)?;
            trace.push(format!("decoder/up_block_{i}"), &x)?;
        }
        let out = self.conv_out.forward(&silu(&self.norm_out.forward(&x)?)?)?;
        trace.push("decoder/conv_out", &out)?;
        Ok(out)
    }
}

/// Optional per-stage capture for the parity tests: every named intermediate as NCHW f32.
pub struct Trace<'a>(Option<&'a mut Vec<(String, Array)>>);

impl Trace<'_> {
    fn push(&mut self, name: impl Into<String>, nhwc: &Array) -> Result<()> {
        if let Some(sink) = self.0.as_deref_mut() {
            sink.push((name.into(), QwenImage21Vae::to_nchw(nhwc)?));
        }
        Ok(())
    }
}

/// The Qwen-Image 2.1 RGBA autoencoder.
pub struct QwenImage21Vae {
    cfg: VaeConfig,
    encoder: Encoder,
    quant_conv: Conv,
    post_quant_conv: Conv,
    decoder: Decoder,
}

impl QwenImage21Vae {
    /// Build from diffusers-keyed weights (`encoder.…`, `decoder.…`, `quant_conv`, `post_quant_conv`).
    pub fn from_weights(w: &Weights, cfg: &VaeConfig) -> Result<Self> {
        Ok(Self {
            cfg: cfg.clone(),
            encoder: Encoder::from_weights(w, "encoder", cfg)?,
            quant_conv: Conv::from_weights(w, "quant_conv", 1, 0)?,
            post_quant_conv: Conv::from_weights(w, "post_quant_conv", 1, 0)?,
            decoder: Decoder::from_weights(w, "decoder", cfg)?,
        })
    }

    pub fn config(&self) -> &VaeConfig {
        &self.cfg
    }

    /// The dtype the model computes in — its weight dtype.
    pub fn compute_dtype(&self) -> Dtype {
        self.quant_conv.weight.dtype()
    }

    fn to_nhwc(x: &Array, dtype: Dtype) -> Result<Array> {
        if x.shape().len() != 4 {
            return Err(Error::Msg(format!(
                "qwen_image_2_1 vae: expected an NCHW tensor, got shape {:?}",
                x.shape()
            )));
        }
        Ok(x.transpose_axes(&[0, 2, 3, 1])?.as_dtype(dtype)?)
    }

    fn to_nchw(x: &Array) -> Result<Array> {
        Ok(x.transpose_axes(&[0, 3, 1, 2])?.as_dtype(Dtype::Float32)?)
    }

    /// Encoder moments for an NCHW RGBA image in `[-1, 1]`: `[1, 2·z_dim, H/16, W/16]` (mean |
    /// logvar), f32 — `_encode` (encoder + `quant_conv`).
    pub fn encode_moments(&self, image: &Array) -> Result<Array> {
        self.encode_moments_inner(image, Trace(None))
    }

    /// [`Self::encode_moments`] plus every named encoder stage (`encoder/conv_in`,
    /// `encoder/down_block_{i}`, `encoder/mid_block`, `encoder/conv_out`, `quant_conv`) as NCHW
    /// f32 — the localisation seam the parity tests read.
    pub fn encode_moments_traced(&self, image: &Array) -> Result<(Array, Vec<(String, Array)>)> {
        let mut trace = Vec::new();
        let moments = self.encode_moments_inner(image, Trace(Some(&mut trace)))?;
        Ok((moments, trace))
    }

    fn encode_moments_inner(&self, image: &Array, mut trace: Trace<'_>) -> Result<Array> {
        let x = Self::to_nhwc(image, self.compute_dtype())?;
        let moments = self
            .quant_conv
            .forward(&self.encoder.forward(&x, &mut trace)?)?;
        trace.push("quant_conv", &moments)?;
        Self::to_nchw(&moments)
    }

    /// The posterior mode (the mean half of the moments): `[1, z_dim, H/16, W/16]`, f32, in the
    /// VAE's own latent space (not yet normalised by `latents_mean`/`latents_std`).
    pub fn encode_mode(&self, image: &Array) -> Result<Array> {
        let moments = self.encode_moments(image)?;
        let z = self.cfg.z_dim as i32;
        Ok(moments.index((.., 0..z, .., ..)))
    }

    /// Decode a VAE-space latent `[1, z_dim, h, w]` (already denormalised) to **RGBA** NCHW
    /// `[1, 4, 16h, 16w]` in `[-1, 1]`, f32. This is the native four-channel output; the pipeline
    /// composites it to RGB for the current gen-core image surface.
    pub fn decode_rgba(&self, latents: &Array) -> Result<Array> {
        self.decode_rgba_inner(latents, Trace(None))
    }

    /// [`Self::decode_rgba`] plus every named decoder stage (`post_quant_conv`, `decoder/conv_in`,
    /// `decoder/mid_block`, `decoder/up_block_{i}`, `decoder/conv_out`) as NCHW f32.
    pub fn decode_rgba_traced(&self, latents: &Array) -> Result<(Array, Vec<(String, Array)>)> {
        let mut trace = Vec::new();
        let rgba = self.decode_rgba_inner(latents, Trace(Some(&mut trace)))?;
        Ok((rgba, trace))
    }

    fn decode_rgba_inner(&self, latents: &Array, mut trace: Trace<'_>) -> Result<Array> {
        let z = Self::to_nhwc(latents, self.compute_dtype())?;
        let z = self.post_quant_conv.forward(&z)?;
        trace.push("post_quant_conv", &z)?;
        let x = self.decoder.forward(&z, &mut trace)?;
        let x = clip(&x, (-1.0f32, 1.0f32))?;
        Self::to_nchw(&x)
    }

    /// The decode **head** run once on the full latent: `post_quant_conv` → `conv_in` → the mid
    /// block with its global attention. NCHW in (VAE-space latent), NCHW out at latent resolution.
    pub fn decode_head(&self, latents: &Array) -> Result<Array> {
        let z = Self::to_nhwc(latents, self.compute_dtype())?;
        let z = self.post_quant_conv.forward(&z)?;
        let head = self.decoder.forward_head(&z, &mut Trace(None))?;
        Ok(head.transpose_axes(&[0, 3, 1, 2])?)
    }

    /// The decode **tail** for one (tile of the) head output: the spatially local up-blocks →
    /// `norm_out` → SiLU → `conv_out` → clamp. NCHW in (head dtype) → RGBA NCHW `[B, 4, 16h, 16w]`
    /// f32. `decode_rgba(z) == decode_tail(decode_head(z))` exactly.
    pub fn decode_tail(&self, head: &Array) -> Result<Array> {
        let x = head.transpose_axes(&[0, 2, 3, 1])?;
        let x = self.decoder.forward_tail(&x, &mut Trace(None))?;
        let x = clip(&x, (-1.0f32, 1.0f32))?;
        Self::to_nchw(&x)
    }

    /// **Bounded** RGBA decode for large outputs (the 2752² presets): the global head runs once,
    /// then the up-sampling tail — where the decode memory spike lives (144 channels at full
    /// output resolution) — runs per overlapping spatial tile and the tiles are trapezoidally
    /// blended by the shared [`mlx_gen::vae_tiling::tiled_decode`] machinery, with a cancel check
    /// and a per-tile `eval` between tiles. Falls back to the single pass when `cfg` does not fire
    /// for these dimensions. The only divergence from [`Self::decode_rgba`] is the conv-halo seam
    /// term the overlap attenuates (see `tests/vae_parity.rs` for the measured bound); the head's
    /// global attention is never tiled, so there is no per-tile normalisation/attention term.
    pub fn decode_rgba_tiled(
        &self,
        latents: &Array,
        cfg: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        let shape = latents.shape().to_vec();
        if shape.len() != 4 {
            return Err(Error::Msg(format!(
                "qwen_image_2_1 vae: decode_rgba_tiled expects [B, C, h, w], got {shape:?}"
            )));
        }
        let (h, w) = (shape[2], shape[3]);
        if !cfg.needs_tiling(VaeTiling::QWEN_IMAGE_2_1, 1, h, w) {
            return self.decode_rgba(latents);
        }
        let head = self.decode_head(latents)?;
        // The tiler works on NCTHW with a singleton frame axis; tile axes are [T, H, W] = [2, 3, 4].
        let hs = head.shape().to_vec();
        let head5 = head.reshape(&[hs[0], hs[1], 1, hs[2], hs[3]])?;
        let plan = cfg.plan(VaeTiling::QWEN_IMAGE_2_1, 1, h, w);
        let out5 = tiled_decode(&head5, &plan, [2, 3, 4], cancel, |tile| {
            let ts = tile.shape().to_vec();
            let tile4 = tile.reshape(&[ts[0], ts[1], ts[3], ts[4]])?;
            let dec = self.decode_tail(&tile4)?;
            let ds = dec.shape().to_vec();
            Ok(dec.reshape(&[ds[0], ds[1], 1, ds[2], ds[3]])?)
        })?;
        let os = out5.shape().to_vec();
        Ok(out5.reshape(&[os[0], os[1], os[3], os[4]])?)
    }

    /// The denoiser-space → VAE-space affine: `z · std + mean` (per channel, NCHW).
    pub fn denormalize(&self, latents: &Array) -> Result<Array> {
        let z = self.cfg.z_dim as i32;
        let mean = Array::from_slice(&self.cfg.latents_mean, &[1, z, 1, 1]);
        let std = Array::from_slice(&self.cfg.latents_std, &[1, z, 1, 1]);
        Ok(add(
            &multiply(&latents.as_dtype(Dtype::Float32)?, &std)?,
            &mean,
        )?)
    }

    /// The VAE-space → denoiser-space affine: `(z − mean) / std` (per channel, NCHW).
    pub fn normalize(&self, latents: &Array) -> Result<Array> {
        let z = self.cfg.z_dim as i32;
        let mean = Array::from_slice(&self.cfg.latents_mean, &[1, z, 1, 1]);
        let std = Array::from_slice(&self.cfg.latents_std, &[1, z, 1, 1]);
        Ok(divide(
            &mlx_rs::ops::subtract(&latents.as_dtype(Dtype::Float32)?, &mean)?,
            &std,
        )?)
    }
}

impl LatentDecoder for QwenImage21Vae {
    fn input_latent_space(&self) -> Option<&LatentSpace> {
        let production = &mlx_gen::gen_core::QWEN_IMAGE_2_1_Z64_LATENT_SPACE;
        (self.cfg.z_dim == production.channels as usize
            && self.cfg.scale_factor_spatial == production.spatial_compression.height as usize)
            .then_some(production)
    }

    /// Denoiser-space latent `[1, z_dim, h, w]` → RGB NCHW `[1, 3, 16h, 16w]` in `[-1, 1]`
    /// (alpha composited over white; use [`Self::decode_rgba`] for the four-channel output).
    fn decode(&self, latents: &Array) -> Result<Array> {
        let rgba = self.decode_rgba(&self.denormalize(latents)?)?;
        crate::pipeline::rgba_to_rgb_over_white(&rgba)
    }

    /// Bounded decode: [`Self::decode_rgba_tiled`] over the denormalised latent, composited to RGB.
    fn decode_tiled(
        &self,
        latents: &Array,
        tiling: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        let rgba = self.decode_rgba_tiled(&self.denormalize(latents)?, tiling, cancel)?;
        crate::pipeline::rgba_to_rgb_over_white(&rgba)
    }
}
