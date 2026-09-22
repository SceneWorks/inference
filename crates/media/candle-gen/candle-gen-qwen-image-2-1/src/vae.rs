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
//! weights exist in the checkpoint and are deliberately not loaded (a rank-5 weight reaching
//! the 2-D conv loader is an error, never a silent reinterpretation).
//!
//! Unlike the MLX twin — which works channels-last because mlx convolutions are NHWC — candle is
//! NCHW natively and the torch weights already ship as `[out, in, kh, kw]`, so this port stays in
//! `[B, C, H, W]` end to end and the layout shuffles disappear. The two places that is *not* a
//! transcription are the parameter-free shortcuts: `AvgDown3D` folds `(channel, [zero-padded
//! frame], row-offset, col-offset)` into the channel axis (dim 1) rather than the trailing one, and
//! `DupUp3D` gathers along dim 1 and expands into dims 2/3.
//!
//! Compute runs in the [`VarBuilder`]'s dtype (f32 on CPU — the parity lane — bf16 on CUDA); the
//! channel-L2 `RMS_norm` normalises in f32 as upstream does for half-precision inputs
//! (`eps = 1e-12`), and every public entry point returns f32.
//!
//! Latent normalisation (`latents_mean` / `latents_std`) is **not** applied here — the pipeline
//! owns it, exactly as upstream's `QwenImage21Pipeline` does around `vae.encode` / `vae.decode`.

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};
use candle_gen::gen_core::tiling::{TilingConfig, VaeTiling};
use candle_gen::gen_core::{CancelFlag, LatentSpace};
use candle_gen::{CandleError as Error, LatentDecoder, Result};

use crate::config::VaeConfig;

/// `F.normalize` floor.
const NORM_EPS: f64 = 1e-12;

/// A 2-D convolution loaded from a torch `[out, in, kh, kw]` weight (+ optional bias).
struct Conv {
    inner: Conv2d,
}

impl Conv {
    fn new(vb: &VarBuilder, base: &str, stride: usize, padding: usize) -> Result<Self> {
        let vb = vb.pp(base);
        let weight = vb.get_unchecked_dtype("weight", vb.dtype())?;
        let rank = weight.dims().len();
        // The 3-D `time_conv` weights are `[out, in, kt, kh, kw]`; a 2-D conv must be rank 4.
        if rank != 4 {
            return Err(Error::Msg(format!(
                "qwen_image_2_1 vae: {base}.weight has rank {rank}, expected a 2-D conv"
            )));
        }
        let bias = if vb.contains_tensor("bias") {
            Some(vb.get_unchecked_dtype("bias", vb.dtype())?)
        } else {
            None
        };
        Ok(Self {
            inner: Conv2d::new(
                weight.contiguous()?,
                bias,
                Conv2dConfig {
                    padding,
                    stride,
                    ..Default::default()
                },
            ),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.inner.forward(&x.contiguous()?)?)
    }
}

/// `QwenImage21RMS_norm`: channel-L2 normalisation `x / max(‖x‖₂, 1e-12) · √C · γ`, f32.
///
/// The norm is taken over the channel axis (dim 1 in NCHW) — not GroupNorm, and not a
/// trailing-feature RMSNorm.
struct ChannelNorm {
    /// `γ · √C`, broadcast-shaped `[1, C, 1, 1]`, f32.
    gamma: Tensor,
}

impl ChannelNorm {
    fn new(vb: &VarBuilder, base: &str) -> Result<Self> {
        // `gamma` ships as [C, 1, 1, 1] (resnets / norm_out) or [C, 1, 1] (attention) — flatten.
        let gamma = vb
            .pp(base)
            .get_unchecked_dtype("gamma", DType::F32)?
            .flatten_all()?;
        let channels = gamma.elem_count();
        let gamma = (gamma.reshape((1, channels, 1, 1))? * (channels as f64).sqrt())?;
        Ok(Self { gamma })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x32 = x.to_dtype(DType::F32)?;
        let norm = x32.sqr()?.sum_keepdim(1)?.sqrt()?.maximum(NORM_EPS)?;
        let normalized = x32.broadcast_div(&norm)?;
        Ok(normalized.broadcast_mul(&self.gamma)?.to_dtype(dtype)?)
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
    fn new(vb: &VarBuilder, base: &str) -> Result<Self> {
        let shortcut = if vb.pp(base).pp("conv_shortcut").contains_tensor("weight") {
            Some(Conv::new(vb, &format!("{base}.conv_shortcut"), 1, 0)?)
        } else {
            None
        };
        Ok(Self {
            norm1: ChannelNorm::new(vb, &format!("{base}.norm1"))?,
            conv1: Conv::new(vb, &format!("{base}.conv1"), 1, 1)?,
            norm2: ChannelNorm::new(vb, &format!("{base}.norm2"))?,
            conv2: Conv::new(vb, &format!("{base}.conv2"), 1, 1)?,
            shortcut,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = match &self.shortcut {
            Some(conv) => conv.forward(x)?,
            None => x.clone(),
        };
        let y = self.conv1.forward(&self.norm1.forward(x)?.silu()?)?;
        let y = self.conv2.forward(&self.norm2.forward(&y)?.silu()?)?;
        Ok((y + h)?)
    }
}

/// `QwenImage21AttentionBlock`: single-head self-attention over the spatial positions, scale
/// `C^-0.5`.
struct AttentionBlock {
    norm: ChannelNorm,
    to_qkv: Conv,
    proj: Conv,
}

impl AttentionBlock {
    fn new(vb: &VarBuilder, base: &str) -> Result<Self> {
        Ok(Self {
            norm: ChannelNorm::new(vb, &format!("{base}.norm"))?,
            to_qkv: Conv::new(vb, &format!("{base}.to_qkv"), 1, 0)?,
            proj: Conv::new(vb, &format!("{base}.proj"), 1, 0)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        // `[B, 3C, H, W]` → the three `[B, H·W, C]` single-head projections: the conv's output
        // channels are the concatenated (q, k, v) feature vectors of each spatial position.
        let qkv = self
            .to_qkv
            .forward(&self.norm.forward(x)?)?
            .reshape((b, 3, c, h * w))?;
        let q = qkv.i((.., 0))?.transpose(1, 2)?.contiguous()?;
        let k = qkv.i((.., 1))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.i((.., 2))?.transpose(1, 2)?.contiguous()?;
        let scale = (c as f64).powf(-0.5);
        // Shared chunked SDPA: the single-head spatial score matrix is `[B, H·W, H·W]`, which
        // overflows candle's i32 element indexing at a large decode (sc-9116).
        let attended = candle_gen::sdpa_budgeted_flat(
            &q,
            &k,
            &v,
            scale,
            softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )?;
        let attended = attended.transpose(1, 2)?.reshape((b, c, h, w))?;
        Ok((self.proj.forward(&attended)? + x)?)
    }
}

struct MidBlock {
    first: ResidualBlock,
    attention: AttentionBlock,
    second: ResidualBlock,
}

impl MidBlock {
    fn new(vb: &VarBuilder, base: &str) -> Result<Self> {
        Ok(Self {
            first: ResidualBlock::new(vb, &format!("{base}.resnets.0"))?,
            attention: AttentionBlock::new(vb, &format!("{base}.attentions.0"))?,
            second: ResidualBlock::new(vb, &format!("{base}.resnets.1"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
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
    fn new(vb: &VarBuilder, base: &str, up: bool) -> Result<Self> {
        let conv = if up {
            Conv::new(vb, &format!("{base}.resample.1"), 1, 1)?
        } else {
            Conv::new(vb, &format!("{base}.resample.1"), 2, 0)?
        };
        Ok(Self { up, conv })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.up {
            let (_, _, h, w) = x.dims4()?;
            self.conv.forward(&x.upsample_nearest2d(h * 2, w * 2)?)
        } else {
            // `F.pad(x, (0, 1, 0, 1))`: one zero column on the right, one zero row at the bottom.
            let padded = x.pad_with_zeros(2, 0, 1)?.pad_with_zeros(3, 0, 1)?;
            self.conv.forward(&padded)
        }
    }
}

/// `QwenImage21AvgDown3D` on a single frame: group-mean shortcut across the folded
/// `(channel, [zero-padded frame], row-offset, col-offset)` axis.
///
/// NCHW re-derivation: the fold that the NHWC twin performs into the trailing axis happens here
/// into the channel axis, so the flat group index runs `(c, a, i, j)` with `c` slowest — the same
/// ordering upstream's `rearrange` produces, and the ordering the group mean depends on.
struct AvgDown {
    out_channels: usize,
    temporal: bool,
    spatial: bool,
}

impl AvgDown {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let fs = if self.spatial { 2 } else { 1 };
        let (h2, w2) = (h / fs, w / fs);
        // [B, C, H', fs, W', fs] → [B, C, fs, fs, H', W'] → flat (c, i, j) on the channel axis.
        let folded = x
            .reshape((b, c, h2, fs, w2, fs))?
            .permute((0, 1, 3, 5, 2, 4))?
            .contiguous()?;
        let folded = if self.temporal {
            // `pad_t` prepends one all-zero frame: flat (c, a, i, j) with a = 0 the zero frame.
            let real = folded.reshape((b, c, 1, fs * fs, h2, w2))?;
            let zero = real.zeros_like()?;
            Tensor::cat(&[&zero, &real], 2)?
                .contiguous()?
                .reshape((b, c * 2 * fs * fs, h2, w2))?
        } else {
            folded.reshape((b, c * fs * fs, h2, w2))?
        };
        let total = folded.dim(1)?;
        let group = total / self.out_channels;
        Ok(folded
            .reshape((b, self.out_channels, group, h2, w2))?
            .mean(2)?)
    }
}

/// `QwenImage21DupUp3D` on the first (only) frame: channel duplication into a 2×2 spatial
/// expansion.
///
/// NCHW re-derivation: the duplicated channels are gathered along dim 1 and then scattered into the
/// two spatial axes, so the `(out, row-offset, col-offset)` triple the NHWC twin builds in its
/// trailing axis is built here in dims 1..4 before the interleave.
struct DupUp {
    in_channels: usize,
    out_channels: usize,
    temporal: bool,
}

impl DupUp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, cin, h, w) = x.dims4()?;
        if cin != self.in_channels {
            return Err(Error::Msg(format!(
                "qwen_image_2_1 vae: DupUp expects {} channels, got {cin}",
                self.in_channels
            )));
        }
        let (ft, fs) = (if self.temporal { 2 } else { 1 }, 2usize);
        let repeats = self.out_channels * ft * fs * fs / self.in_channels;
        // Output (o, row-offset i, col-offset j) reads the duplicated channel index
        // (((o·ft + (ft−1))·fs + i)·fs + j) / repeats — `first_chunk` keeps the last temporal slot.
        let mut index = Vec::with_capacity(self.out_channels * fs * fs);
        for o in 0..self.out_channels {
            for i in 0..fs {
                for j in 0..fs {
                    let flat = ((o * ft + (ft - 1)) * fs + i) * fs + j;
                    index.push((flat / repeats) as u32);
                }
            }
        }
        let index = Tensor::from_vec(index, (self.out_channels * fs * fs,), x.device())?;
        Ok(x.index_select(&index, 1)?
            .reshape((b, self.out_channels, fs, fs, h, w))?
            .permute((0, 1, 4, 2, 5, 3))?
            .contiguous()?
            .reshape((b, self.out_channels, h * fs, w * fs))?)
    }
}

struct DownBlock {
    resnets: Vec<ResidualBlock>,
    downsampler: Option<Resample>,
    shortcut: AvgDown,
}

impl DownBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = x.clone();
        for block in &self.resnets {
            y = block.forward(&y)?;
        }
        if let Some(down) = &self.downsampler {
            y = down.forward(&y)?;
        }
        Ok((y + self.shortcut.forward(x)?)?)
    }
}

struct UpBlock {
    resnets: Vec<ResidualBlock>,
    upsampler: Option<Resample>,
    shortcut: Option<DupUp>,
}

impl UpBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = x.clone();
        for block in &self.resnets {
            y = block.forward(&y)?;
        }
        if let Some(up) = &self.upsampler {
            y = up.forward(&y)?;
        }
        match &self.shortcut {
            Some(dup) => Ok((y + dup.forward(x)?)?),
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
    fn new(vb: &VarBuilder, base: &str, cfg: &VaeConfig) -> Result<Self> {
        let dims: Vec<usize> = std::iter::once(1)
            .chain(cfg.dim_mult.iter().copied())
            .map(|m| cfg.base_dim * m)
            .collect();
        let last = stage_count(cfg)? - 1;
        let mut down_blocks = Vec::with_capacity(last + 1);
        for i in 0..=last {
            let prefix = format!("{base}.down_blocks.{i}");
            let down = i != last;
            let temporal = down && cfg.temperal_downsample[i];
            let resnets = (0..cfg.num_res_blocks)
                .map(|j| ResidualBlock::new(vb, &format!("{prefix}.resnets.{j}")))
                .collect::<Result<Vec<_>>>()?;
            let downsampler = if down {
                Some(Resample::new(vb, &format!("{prefix}.downsampler"), false)?)
            } else {
                None
            };
            let (in_dim, out_dim) = (dims[i], dims[i + 1]);
            let factor = if temporal { 2 } else { 1 } * if down { 4 } else { 1 };
            if out_dim == 0 || (in_dim * factor) % out_dim != 0 {
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
            conv_in: Conv::new(vb, &format!("{base}.conv_in"), 1, 1)?,
            down_blocks,
            mid: MidBlock::new(vb, &format!("{base}.mid_block"))?,
            norm_out: ChannelNorm::new(vb, &format!("{base}.norm_out"))?,
            conv_out: Conv::new(vb, &format!("{base}.conv_out"), 1, 1)?,
        })
    }

    fn forward(&self, x: &Tensor, trace: &mut Trace<'_>) -> Result<Tensor> {
        let mut x = self.conv_in.forward(x)?;
        trace.push("encoder/conv_in", &x)?;
        for (i, block) in self.down_blocks.iter().enumerate() {
            x = block.forward(&x)?;
            trace.push(format!("encoder/down_block_{i}"), &x)?;
        }
        let x = self.mid.forward(&x)?;
        trace.push("encoder/mid_block", &x)?;
        let out = self.conv_out.forward(&self.norm_out.forward(&x)?.silu()?)?;
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
    fn new(vb: &VarBuilder, base: &str, cfg: &VaeConfig) -> Result<Self> {
        let dims: Vec<usize> = std::iter::once(*cfg.dim_mult.last().unwrap_or(&1))
            .chain(cfg.dim_mult.iter().rev().copied())
            .map(|m| cfg.decoder_base_dim * m)
            .collect();
        let temporal_upsample: Vec<bool> = cfg.temperal_downsample.iter().rev().copied().collect();
        let last = stage_count(cfg)? - 1;
        let mut up_blocks = Vec::with_capacity(last + 1);
        for i in 0..=last {
            let prefix = format!("{base}.up_blocks.{i}");
            let up = i != last;
            let temporal = up && temporal_upsample[i];
            let resnets = (0..=cfg.num_res_blocks)
                .map(|j| ResidualBlock::new(vb, &format!("{prefix}.resnets.{j}")))
                .collect::<Result<Vec<_>>>()?;
            let (in_dim, out_dim) = (dims[i], dims[i + 1]);
            let (upsampler, shortcut) = if up {
                let factor = if temporal { 2 } else { 1 } * 4;
                if in_dim == 0 || (out_dim * factor) % in_dim != 0 {
                    return Err(Error::Msg(format!(
                        "qwen_image_2_1 vae: decoder stage {i}: {out_dim}·{factor} is not divisible by {in_dim}"
                    )));
                }
                (
                    Some(Resample::new(vb, &format!("{prefix}.upsampler"), true)?),
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
            conv_in: Conv::new(vb, &format!("{base}.conv_in"), 1, 1)?,
            mid: MidBlock::new(vb, &format!("{base}.mid_block"))?,
            up_blocks,
            norm_out: ChannelNorm::new(vb, &format!("{base}.norm_out"))?,
            conv_out: Conv::new(vb, &format!("{base}.conv_out"), 1, 1)?,
        })
    }

    fn forward(&self, x: &Tensor, trace: &mut Trace<'_>) -> Result<Tensor> {
        let head = self.forward_head(x, trace)?;
        self.forward_tail(&head, trace)
    }

    /// The **global** head: `conv_in` → mid block (whose single-head attention spans the whole
    /// latent). Runs once on the full latent; cheap at latent resolution.
    fn forward_head(&self, x: &Tensor, trace: &mut Trace<'_>) -> Result<Tensor> {
        let x = self.conv_in.forward(x)?;
        trace.push("decoder/conv_in", &x)?;
        let x = self.mid.forward(&x)?;
        trace.push("decoder/mid_block", &x)?;
        Ok(x)
    }

    /// The **spatially local** upsample tail: `up_blocks` → `norm_out` → SiLU → `conv_out`. Every
    /// op is a per-pixel norm, a 3×3 conv, a nearest ×2 or a channel shuffle, so it tiles.
    fn forward_tail(&self, head: &Tensor, trace: &mut Trace<'_>) -> Result<Tensor> {
        let mut x = head.clone();
        for (i, block) in self.up_blocks.iter().enumerate() {
            x = block.forward(&x)?;
            trace.push(format!("decoder/up_block_{i}"), &x)?;
        }
        let out = self.conv_out.forward(&self.norm_out.forward(&x)?.silu()?)?;
        trace.push("decoder/conv_out", &out)?;
        Ok(out)
    }
}

/// The number of resample stages, refusing a config with no stage at all (the `len() - 1`
/// arithmetic below would otherwise underflow).
fn stage_count(cfg: &VaeConfig) -> Result<usize> {
    if cfg.dim_mult.is_empty() {
        return Err(Error::Msg(
            "qwen_image_2_1 vae: `dim_mult` must name at least one stage".into(),
        ));
    }
    Ok(cfg.dim_mult.len())
}

/// Optional per-stage capture for the parity tests: every named intermediate as NCHW f32.
pub struct Trace<'a>(Option<&'a mut Vec<(String, Tensor)>>);

impl Trace<'_> {
    fn push(&mut self, name: impl Into<String>, x: &Tensor) -> Result<()> {
        if let Some(sink) = self.0.as_deref_mut() {
            sink.push((name.into(), x.to_dtype(DType::F32)?));
        }
        Ok(())
    }
}

/// The Qwen-Image 2.1 RGBA autoencoder.
pub struct QwenImage21Vae {
    cfg: VaeConfig,
    device: Device,
    dtype: DType,
    encoder: Encoder,
    quant_conv: Conv,
    post_quant_conv: Conv,
    decoder: Decoder,
}

impl QwenImage21Vae {
    /// Build from diffusers-keyed weights (`encoder.…`, `decoder.…`, `quant_conv`,
    /// `post_quant_conv`).
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            cfg: cfg.clone(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
            encoder: Encoder::new(&vb, "encoder", cfg)?,
            quant_conv: Conv::new(&vb, "quant_conv", 1, 0)?,
            post_quant_conv: Conv::new(&vb, "post_quant_conv", 1, 0)?,
            decoder: Decoder::new(&vb, "decoder", cfg)?,
        })
    }

    pub fn config(&self) -> &VaeConfig {
        &self.cfg
    }

    /// The device the weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The dtype the model computes in — its weight dtype.
    pub fn compute_dtype(&self) -> DType {
        self.dtype
    }

    /// An NCHW input cast into the compute dtype, refusing any other rank.
    fn to_compute(&self, x: &Tensor) -> Result<Tensor> {
        if x.dims().len() != 4 {
            return Err(Error::Msg(format!(
                "qwen_image_2_1 vae: expected an NCHW tensor, got shape {:?}",
                x.dims()
            )));
        }
        Ok(x.to_dtype(self.dtype)?.contiguous()?)
    }

    /// Encoder moments for an NCHW RGBA image in `[-1, 1]`: `[1, 2·z_dim, H/16, W/16]` (mean |
    /// logvar), f32 — `_encode` (encoder + `quant_conv`).
    pub fn encode_moments(&self, image: &Tensor) -> Result<Tensor> {
        self.encode_moments_inner(image, Trace(None))
    }

    /// [`Self::encode_moments`] plus every named encoder stage (`encoder/conv_in`,
    /// `encoder/down_block_{i}`, `encoder/mid_block`, `encoder/conv_out`, `quant_conv`) as NCHW
    /// f32 — the localisation seam the parity tests read.
    pub fn encode_moments_traced(&self, image: &Tensor) -> Result<(Tensor, Vec<(String, Tensor)>)> {
        let mut trace = Vec::new();
        let moments = self.encode_moments_inner(image, Trace(Some(&mut trace)))?;
        Ok((moments, trace))
    }

    fn encode_moments_inner(&self, image: &Tensor, mut trace: Trace<'_>) -> Result<Tensor> {
        let x = self.to_compute(image)?;
        let moments = self
            .quant_conv
            .forward(&self.encoder.forward(&x, &mut trace)?)?;
        trace.push("quant_conv", &moments)?;
        Ok(moments.to_dtype(DType::F32)?)
    }

    /// The posterior mode (the mean half of the moments): `[1, z_dim, H/16, W/16]`, f32, in the
    /// VAE's own latent space (not yet normalised by `latents_mean`/`latents_std`).
    pub fn encode_mode(&self, image: &Tensor) -> Result<Tensor> {
        let moments = self.encode_moments(image)?;
        Ok(moments.narrow(1, 0, self.cfg.z_dim)?.contiguous()?)
    }

    /// Decode a VAE-space latent `[1, z_dim, h, w]` (already denormalised) to **RGBA** NCHW
    /// `[1, 4, 16h, 16w]` in `[-1, 1]`, f32. This is the native four-channel output; the pipeline
    /// composites it to RGB for the current gen-core image surface.
    pub fn decode_rgba(&self, latents: &Tensor) -> Result<Tensor> {
        self.decode_rgba_inner(latents, Trace(None))
    }

    /// [`Self::decode_rgba`] plus every named decoder stage (`post_quant_conv`, `decoder/conv_in`,
    /// `decoder/mid_block`, `decoder/up_block_{i}`, `decoder/conv_out`) as NCHW f32.
    pub fn decode_rgba_traced(&self, latents: &Tensor) -> Result<(Tensor, Vec<(String, Tensor)>)> {
        let mut trace = Vec::new();
        let rgba = self.decode_rgba_inner(latents, Trace(Some(&mut trace)))?;
        Ok((rgba, trace))
    }

    fn decode_rgba_inner(&self, latents: &Tensor, mut trace: Trace<'_>) -> Result<Tensor> {
        let z = self.post_quant_conv.forward(&self.to_compute(latents)?)?;
        trace.push("post_quant_conv", &z)?;
        let x = self.decoder.forward(&z, &mut trace)?;
        Ok(x.clamp(-1f32, 1f32)?.to_dtype(DType::F32)?)
    }

    /// The decode **head** run once on the full latent: `post_quant_conv` → `conv_in` → the mid
    /// block with its global attention. NCHW in (VAE-space latent), NCHW out at latent resolution.
    pub fn decode_head(&self, latents: &Tensor) -> Result<Tensor> {
        let z = self.post_quant_conv.forward(&self.to_compute(latents)?)?;
        self.decoder.forward_head(&z, &mut Trace(None))
    }

    /// The decode **tail** for one (tile of the) head output: the spatially local up-blocks →
    /// `norm_out` → SiLU → `conv_out` → clamp. NCHW in (head dtype) → RGBA NCHW `[B, 4, 16h, 16w]`
    /// f32. `decode_rgba(z) == decode_tail(decode_head(z))` exactly.
    pub fn decode_tail(&self, head: &Tensor) -> Result<Tensor> {
        let x = self.decoder.forward_tail(head, &mut Trace(None))?;
        Ok(x.clamp(-1f32, 1f32)?.to_dtype(DType::F32)?)
    }

    /// **Bounded** RGBA decode for large outputs (the 2752² presets): the global head runs once,
    /// then the up-sampling tail — where the decode memory spike lives (144 channels at full
    /// output resolution) — runs per overlapping spatial tile and the tiles are trapezoidally
    /// blended by the shared [`candle_gen::vae_tiling::decode_tiled`] machinery, with a cancel
    /// check between tiles. Falls back to the single pass when `cfg` does not fire for these
    /// dimensions. The only divergence from [`Self::decode_rgba`] is the conv-halo seam term the
    /// overlap attenuates (see `tests/vae_parity.rs` for the measured bound); the head's global
    /// attention is never tiled, so there is no per-tile normalisation/attention term.
    ///
    /// The candle twin of `mlx-gen-qwen-image-2-1`'s `decode_rgba_tiled`, over the SAME
    /// [`VaeTiling::QWEN_IMAGE_2_1`] geometry in gen-core — one declaration, two engines.
    pub fn decode_rgba_tiled(
        &self,
        latents: &Tensor,
        cfg: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Tensor> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        let (b, c, h, w) = latents.dims4()?;
        if !cfg.needs_tiling(VaeTiling::QWEN_IMAGE_2_1, 1, h as i32, w as i32) {
            return self.decode_rgba(latents);
        }
        let _ = (b, c);
        let head = self.decode_head(latents)?;
        // The shared tiler works on NCTHW with a singleton frame axis.
        let (hb, hc, hh, hw) = head.dims4()?;
        let head5 = head.reshape((hb, hc, 1, hh, hw))?;
        let out5 = candle_gen::vae_tiling::decode_tiled::<_, Error>(
            VaeTiling::QWEN_IMAGE_2_1,
            "qwen_image_2_1 rgba vae",
            &head5,
            cfg,
            |tile| {
                if cancel.is_some_and(CancelFlag::is_cancelled) {
                    return Err(Error::Canceled);
                }
                let (tb, tc, _, th, tw) = tile.dims5()?;
                let dec = self.decode_tail(&tile.reshape((tb, tc, th, tw))?.contiguous()?)?;
                let (db, dc, dh, dw) = dec.dims4()?;
                Ok(dec.reshape((db, dc, 1, dh, dw))?)
            },
        )?;
        let (ob, oc, _, oh, ow) = out5.dims5()?;
        Ok(out5.reshape((ob, oc, oh, ow))?)
    }

    /// `[1, z_dim, 1, 1]` broadcast constants for the latent affine.
    fn latent_affine(&self) -> Result<(Tensor, Tensor)> {
        let z = self.cfg.z_dim;
        let mean = Tensor::from_vec(self.cfg.latents_mean.clone(), (1, z, 1, 1), &self.device)?;
        let std = Tensor::from_vec(self.cfg.latents_std.clone(), (1, z, 1, 1), &self.device)?;
        Ok((mean, std))
    }

    /// The denoiser-space → VAE-space affine: `z · std + mean` (per channel, NCHW).
    pub fn denormalize(&self, latents: &Tensor) -> Result<Tensor> {
        let (mean, std) = self.latent_affine()?;
        Ok(latents
            .to_dtype(DType::F32)?
            .broadcast_mul(&std)?
            .broadcast_add(&mean)?)
    }

    /// The VAE-space → denoiser-space affine: `(z − mean) / std` (per channel, NCHW).
    pub fn normalize(&self, latents: &Tensor) -> Result<Tensor> {
        let (mean, std) = self.latent_affine()?;
        Ok(latents
            .to_dtype(DType::F32)?
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?)
    }
}

impl LatentDecoder for QwenImage21Vae {
    fn input_latent_space(&self) -> Option<&LatentSpace> {
        let production = &candle_gen::gen_core::QWEN_IMAGE_2_1_Z64_LATENT_SPACE;
        (self.cfg.z_dim == production.channels as usize
            && self.cfg.scale_factor_spatial == production.spatial_compression.height as usize)
            .then_some(production)
    }

    /// Denoiser-space latent `[1, z_dim, h, w]` → RGB NCHW `[1, 3, 16h, 16w]` in `[-1, 1]`
    /// (alpha composited over white; use [`Self::decode_rgba`] for the four-channel output).
    fn decode(&self, latents: &Tensor) -> Result<Tensor> {
        let rgba = self.decode_rgba(&self.denormalize(latents)?)?;
        crate::pipeline::rgba_to_rgb_over_white(&rgba)
    }

    /// Bounded decode: [`Self::decode_rgba_tiled`] over the denormalised latent, composited to RGB.
    fn decode_tiled(
        &self,
        latents: &Tensor,
        tiling: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Tensor> {
        let rgba = self.decode_rgba_tiled(&self.denormalize(latents)?, tiling, cancel)?;
        crate::pipeline::rgba_to_rgb_over_white(&rgba)
    }
}
