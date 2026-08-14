//! The MiniMax-H3 video VAE **encode** half — the 3-D causal CNN
//! (`MiniMaxH3VideoEncoder3d` + `quant_conv`), 118 published tensors.
//!
//! sc-17154 ported `decode` only. `fl2va` (sc-19008) needs this half: a keyframe is conditioned
//! through **both** the text encoder's vision tower *and* this encoder, so there is no keyframe
//! conditioning without it.
//!
//! The two halves share nothing structurally. The decoder is a 36-layer ViT
//! ([`crate::decoder::ViT3dDecoder`]); the encoder is a plain conv stack with no attention and no
//! mid-block. What it does have are four conventions that are each individually easy to get wrong
//! and impossible to see afterwards:
//!
//! # 1. Temporal padding is FRONT-ONLY ZEROS, and it is carried, not derived
//!
//! [`CausalConv3d`] pads `temporal_padding` zero frames before the clip and none after. Every
//! 3×3×3 conv in the encoder uses `temporal_padding = 2` against `kernel_t = 3` — the same number
//! as the causal-minimal `kernel_t - 1` here, but carried explicitly because the reference carries
//! it explicitly and a derived version would silently disagree the moment a kernel changed.
//!
//! The consequence that matters: **encoding a single frame is not the same as encoding one frame
//! out of a longer clip.** A `T = 1` input sees only zeros before it. That is exactly the
//! conditioning MiniMax-H3 was trained with for keyframes, and
//! [`crate::vae::MiniMaxH3VideoVae::encode`] takes the reference's `num_frames == 1` short circuit
//! rather than padding up to `clip_length`.
//!
//! # 2. Spatial padding is `reflect`, and the downsampler's is asymmetric
//!
//! Ordinary convs reflect-pad symmetrically by 1. The stride-2 downsampler instead pads
//! **bottom/right by 1 only** and then convolves with no spatial padding, so the output is exactly
//! `ceil(size / 2)`. Substituting a symmetric pad shifts every downsampled feature by half a pixel
//! and is invisible in any norm.
//!
//! # 3. GroupNorm statistics are computed PER LATENT FRAME, never across time
//!
//! [`FrameIsolatedGroupNorm`] folds `T` into the batch axis before normalizing. A standard
//! `GroupNorm` over `(C, T, H, W)` mixes statistics across time; it produces a plausible tensor of
//! the same shape whose values are wrong in a way that grows with clip length and is *zero* at
//! `T = 1` — so a single-frame test cannot distinguish the two. `groupnorm_is_frame_isolated`
//! gates it at `T > 1` for that reason.
//!
//! # 4. Tiling is ON by default, and the released latents are the blended-tile ones
//!
//! `AutoencoderKLMiniMaxH3.__init__` sets `use_tiling = True` (256 px tiles, 64 px minimum
//! overlap) — unusual for diffusers, where tiling is normally opt-in via `enable_tiling()`. So a
//! port that encodes the whole canvas in one pass does **not** reproduce the reference at any
//! canvas wider than 256 px. [`TilePlan`] is the reference's `_split_tiles` and
//! [`stitch_tiles`] its `_blend` / `_stitch_tiles`.
//!
//! # candle vs MLX: NCTHW, and a conv3d built from conv2d taps
//!
//! The MLX sibling runs this stack in NDHWC because that is mlx's native convolution layout. This
//! port runs it in **NCTHW** throughout, which is candle's — so the two lanes are not one
//! implementation typed twice, and their cross-backend agreement is evidence rather than tautology.
//!
//! candle ships **no `conv3d`**, so [`conv3d_ncthw`] decomposes each `kt×kh×kw` kernel into `kt`
//! `conv2d` taps summed over the temporal axis — the established pattern in this repo
//! (`candle-gen-ltx`, `-seedvr2`, `-wan`, `-mochi`). Unlike those, this one carries a **temporal
//! stride**, because two of the shipped encoder's four downsamplers stride time as well as space.
//!
//! # What this module does NOT do
//!
//! It stops at the posterior parameters. Sampling the posterior, the fp16 round-trip and the
//! per-channel latent normalization are conditioning policy, not VAE structure.

use candle_gen::candle_core::{DType, Tensor};
use candle_gen::{CandleError, Result, Weights};

use crate::blocks::blend;
use crate::config::MiniMaxH3VaeConfig;
use crate::nn::silu;

/// Temporal padding every 3×3×3 encoder conv applies, front-only. The reference's literal.
pub const ENCODER_TEMPORAL_PADDING: usize = 2;

/// Default tile edge, in pixels, for the spatially-tiled encode (`tile_sample_min_height/width`).
pub const TILE_SAMPLE_MIN_SIZE: usize = 256;

/// Default minimum tile overlap, in pixels (`tile_sample_min_overlap_height/width`).
pub const TILE_SAMPLE_MIN_OVERLAP: usize = 64;

/// Whether the shipped VAE encodes with spatial tiling enabled.
///
/// Pinned as a constant because it is a **default**, and defaults are the class of fact that gets
/// assumed rather than read. `AutoencoderKLMiniMaxH3` turns tiling on in `__init__`; almost every
/// other diffusers autoencoder leaves it off until `enable_tiling()` is called. Encoding a
/// 768×1344 keyframe untiled is a 28-tile difference in the result.
pub const ENCODER_TILING_IS_ON_BY_DEFAULT: bool = true;

/// The `logvar` clamp diffusers' `DiagonalGaussianDistribution` applies.
pub const LOGVAR_CLAMP: (f64, f64) = (-30.0, 20.0);

/// Keep every candle CUDA im2col launch well below `u32::MAX`. Larger launches truncate the element
/// count to `u32`, leaving an uninitialized output tail — the same guard `candle-gen-ltx` and
/// `-seedvr2` carry, and the encoder needs it because a production 256 px tile at 128 channels is
/// already within an order of the ceiling.
const IM2COL_BUDGET: usize = 128 * 1024 * 1024;

// -------------------------------------------------------------------------------------------
// Padding
// -------------------------------------------------------------------------------------------

/// Reflect-pad `x` along `axis` by `before` / `after`, matching `torch.nn.functional.pad(mode =
/// "reflect")`.
///
/// The reflection does **not** repeat the edge sample: padded position `j` before the start maps to
/// source `before - j`, and position `k` past the end maps to source `n - 2 - k`. Torch requires
/// the pad to be strictly smaller than the axis, and so does this — a larger pad would need to
/// reflect off an already-reflected sample, which torch rejects rather than defines.
pub fn reflect_pad_axis(x: &Tensor, axis: usize, before: usize, after: usize) -> Result<Tensor> {
    if before == 0 && after == 0 {
        return Ok(x.clone());
    }
    let rank = x.dims().len();
    if axis >= rank {
        return Err(CandleError::Msg(format!(
            "minimax-h3 encoder: reflect pad axis {axis} is outside a rank-{rank} tensor"
        )));
    }
    let n = x.dims()[axis];
    if before >= n || after >= n {
        return Err(CandleError::Msg(format!(
            "minimax-h3 encoder: cannot reflect-pad {before}/{after} off an axis of length {n}; \
             torch requires the pad to be smaller than the axis"
        )));
    }
    let mut idx: Vec<u32> = Vec::with_capacity(before + n + after);
    for j in (1..=before).rev() {
        idx.push(j as u32);
    }
    idx.extend((0..n).map(|i| i as u32));
    for k in 0..after {
        idx.push((n - 2 - k) as u32);
    }
    let index = Tensor::from_vec(idx, (before + n + after,), x.device())?;
    Ok(x.index_select(&index, axis)?.contiguous()?)
}

/// Prepend `frames` zero frames along the temporal axis of an NCTHW tensor — the causal pad.
pub fn zero_pad_front_time(x: &Tensor, frames: usize) -> Result<Tensor> {
    if frames == 0 {
        return Ok(x.clone());
    }
    let s = x.dims();
    if s.len() != 5 {
        return Err(CandleError::Msg(format!(
            "minimax-h3 encoder: expected an NCTHW tensor to causal-pad, got {s:?}"
        )));
    }
    let pad = Tensor::zeros((s[0], s[1], frames, s[3], s[4]), x.dtype(), x.device())?;
    Ok(Tensor::cat(&[&pad, x], 2)?)
}

// -------------------------------------------------------------------------------------------
// conv3d, built from conv2d taps
// -------------------------------------------------------------------------------------------

fn checked_product(values: &[usize], what: &str) -> Result<usize> {
    values.iter().try_fold(1usize, |acc, &v| {
        acc.checked_mul(v).ok_or_else(|| {
            CandleError::Msg(format!("minimax-h3 encoder: {what} size overflows usize"))
        })
    })
}

/// `x.conv2d(w, padding = 0, stride)`, split across the batch axis so no im2col buffer can enter
/// candle's u32-overflow band.
fn chunked_conv2d(x: &Tensor, w: &Tensor, stride: usize, budget: usize) -> Result<Tensor> {
    let (n, c, h, wd) = x.dims4()?;
    let (_o, _i, kh, kw) = w.dims4()?;
    let h_out = h.saturating_sub(kh) / stride + 1;
    let w_out = wd.saturating_sub(kw) / stride + 1;
    let cols_per_frame = checked_product(&[h_out, w_out, c, kh, kw], "im2col frame")?.max(1);
    let max_batch = (budget / cols_per_frame).clamp(1, n.max(1));
    if n <= max_batch {
        return Ok(x.conv2d(w, 0, stride, 1, 1)?);
    }
    let mut parts = Vec::new();
    let mut start = 0;
    while start < n {
        let len = (n - start).min(max_batch);
        parts.push(x.narrow(0, start, len)?.conv2d(w, 0, stride, 1, 1)?);
        start += len;
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&refs, 0)?)
}

/// A **strided** 3-D cross-correlation over NCTHW `x` with a torch-layout `[O, I, kT, kH, kW]`
/// weight, applying **no padding of its own** — every pad this encoder needs is asymmetric or
/// non-zero-valued, so the callers apply it explicitly and this stays a pure convolution.
///
/// candle ships no `conv3d`. Output frame `o` is the sum over taps `kd` of
/// `conv2d(x[:, :, o·stride_t + kd], W[:, :, kd])`, which is the definition of a 3-D convolution
/// with temporal stride `stride_t` written as `kT` two-dimensional ones.
///
/// The two spatial strides must be equal — every downsampler in this family strides height and
/// width by the same `spatial_downsample_factors[level]`, and candle's `conv2d` takes one stride.
pub fn conv3d_ncthw(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    stride: (usize, usize, usize),
) -> Result<Tensor> {
    let (b, c, t, h, wd) = x.dims5()?;
    let (o, i, kt, kh, kw) = w.dims5()?;
    let (st, sh, sw) = stride;
    if sh != sw {
        return Err(CandleError::Msg(format!(
            "minimax-h3 encoder: conv3d needs equal spatial strides, got {sh}/{sw}"
        )));
    }
    if st == 0 || sh == 0 {
        return Err(CandleError::Msg(
            "minimax-h3 encoder: conv3d strides must be positive".into(),
        ));
    }
    if c != i {
        return Err(CandleError::Msg(format!(
            "minimax-h3 encoder: conv3d input has {c} channels, the weight expects {i}"
        )));
    }
    if t < kt || h < kh || wd < kw {
        return Err(CandleError::Msg(format!(
            "minimax-h3 encoder: a {kt}x{kh}x{kw} kernel does not fit a {t}x{h}x{wd} volume; the \
             caller must pad before convolving"
        )));
    }
    let t_out = (t - kt) / st + 1;

    let mut acc: Option<Tensor> = None;
    for kd in 0..kt {
        // The `t_out` frames this tap convolves: x[:, :, kd + o·st] for o in 0..t_out.
        let frames = if st == 1 {
            x.narrow(2, kd, t_out)?
        } else {
            let idx: Vec<u32> = (0..t_out).map(|o| (kd + o * st) as u32).collect();
            let index = Tensor::from_vec(idx, (t_out,), x.device())?;
            x.index_select(&index, 2)?
        };
        // [B, C, T', H, W] -> [B·T', C, H, W] for conv2d.
        let merged =
            frames
                .permute((0, 2, 1, 3, 4))?
                .contiguous()?
                .reshape((b * t_out, c, h, wd))?;
        let wk = w.narrow(2, kd, 1)?.squeeze(2)?.contiguous()?;
        let y = chunked_conv2d(&merged, &wk, sh, IM2COL_BUDGET)?;
        acc = Some(match acc {
            Some(a) => (a + y)?,
            None => y,
        });
    }
    let y = acc.ok_or_else(|| {
        CandleError::Msg("minimax-h3 encoder: a conv3d kernel must have at least one tap".into())
    })?;
    let (_, _, h_out, w_out) = y.dims4()?;
    let y = y
        .reshape((b, t_out, o, h_out, w_out))?
        .permute((0, 2, 1, 3, 4))?
        .contiguous()?;
    match bias {
        Some(bias) => Ok(y.broadcast_add(&bias.reshape((1, o, 1, 1, 1))?)?),
        None => Ok(y),
    }
}

/// A 3-D convolution that is **causal in time** (front-only zero padding) and **reflecting in
/// space**.
#[derive(Debug, Clone)]
pub struct CausalConv3d {
    weight: Tensor,
    bias: Tensor,
    stride: (usize, usize, usize),
    spatial_padding: usize,
    temporal_padding: usize,
}

impl CausalConv3d {
    /// Load `{prefix}.weight` / `{prefix}.bias`.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        dtype: DType,
        stride: (usize, usize, usize),
        spatial_padding: usize,
        temporal_padding: usize,
    ) -> Result<Self> {
        let weight = w.require(&format!("{prefix}.weight"))?.to_dtype(dtype)?;
        let s = weight.dims();
        if s.len() != 5 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 encoder: {prefix}.weight must be a rank-5 Conv3d weight, got {s:?}"
            )));
        }
        Ok(Self {
            weight: weight.contiguous()?,
            bias: w.require(&format!("{prefix}.bias"))?.to_dtype(dtype)?,
            stride,
            spatial_padding,
            temporal_padding,
        })
    }

    /// The two tensor names this conv consumes.
    pub fn names(prefix: &str) -> Vec<String> {
        vec![format!("{prefix}.weight"), format!("{prefix}.bias")]
    }

    /// NCTHW in, NCTHW out.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        if self.spatial_padding > 0 {
            h = reflect_pad_axis(&h, 3, self.spatial_padding, self.spatial_padding)?;
            h = reflect_pad_axis(&h, 4, self.spatial_padding, self.spatial_padding)?;
        }
        h = zero_pad_front_time(&h, self.temporal_padding)?;
        conv3d_ncthw(&h, &self.weight, Some(&self.bias), self.stride)
    }
}

/// `GroupNorm` whose statistics are computed **per latent frame in isolation**.
///
/// The reference folds `T` into the batch axis (`view(B·T, C, 1, H, W)`) before normalizing, so no
/// statistic ever mixes across time. See the module docs for why substituting a plain 3-D
/// `GroupNorm` is invisible at `T = 1`.
///
/// Computed in f32 and cast back, mirroring the rest of [`crate::nn`]; the epsilon is **inside**
/// the square root.
#[derive(Debug, Clone)]
pub struct FrameIsolatedGroupNorm {
    weight: Tensor,
    bias: Tensor,
    groups: usize,
    eps: f64,
}

impl FrameIsolatedGroupNorm {
    /// Load `{prefix}.weight` / `{prefix}.bias`.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        dtype: DType,
        groups: usize,
        eps: f64,
    ) -> Result<Self> {
        if groups == 0 {
            return Err(CandleError::Msg(
                "minimax-h3 encoder: a GroupNorm needs at least one group".into(),
            ));
        }
        Ok(Self {
            weight: w.require(&format!("{prefix}.weight"))?.to_dtype(dtype)?,
            bias: w.require(&format!("{prefix}.bias"))?.to_dtype(dtype)?,
            groups,
            eps,
        })
    }

    /// The two tensor names this norm consumes.
    pub fn names(prefix: &str) -> Vec<String> {
        vec![format!("{prefix}.weight"), format!("{prefix}.bias")]
    }

    /// NCTHW in, NCTHW out.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let s = x.dims().to_vec();
        if s.len() != 5 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 encoder: frame-isolated GroupNorm expects NCTHW, got {s:?}"
            )));
        }
        let (b, c, t, h, w) = (s[0], s[1], s[2], s[3], s[4]);
        if !c.is_multiple_of(self.groups) {
            return Err(CandleError::Msg(format!(
                "minimax-h3 encoder: {c} channels is not divisible by {} GroupNorm groups",
                self.groups
            )));
        }
        let dtype = x.dtype();
        // Fold T into the batch axis so every reduction is per-frame, then split C into groups.
        let folded = x
            .to_dtype(DType::F32)?
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .reshape((b * t * self.groups, (c / self.groups) * h * w))?;
        let mean = folded.mean_keepdim(1)?;
        let centred = folded.broadcast_sub(&mean)?;
        let var = centred.sqr()?.mean_keepdim(1)?;
        let normed = centred
            .broadcast_div(&(var + self.eps)?.sqrt()?)?
            .reshape((b, t, c, h, w))?;
        let affine = normed
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?.reshape((1, 1, c, 1, 1))?)?
            .broadcast_add(&self.bias.to_dtype(DType::F32)?.reshape((1, 1, c, 1, 1))?)?;
        Ok(affine
            .permute((0, 2, 1, 3, 4))?
            .contiguous()?
            .to_dtype(dtype)?)
    }
}

/// Pre-norm residual block: `norm1 → silu → conv1 → norm2 → silu → conv2`, plus an optional
/// 1×1×1 projection on the residual when the block changes width. No timestep embedding.
#[derive(Debug, Clone)]
pub struct ResnetBlock3d {
    norm1: FrameIsolatedGroupNorm,
    conv1: CausalConv3d,
    norm2: FrameIsolatedGroupNorm,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResnetBlock3d {
    /// Load one `resnets.{j}` block. `shortcut` is present exactly when `in_channels !=
    /// out_channels`.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &MiniMaxH3VaeConfig,
        dtype: DType,
        in_channels: usize,
        out_channels: usize,
    ) -> Result<Self> {
        let groups = cfg.norm_num_groups;
        let eps = cfg.encoder_norm_eps;
        let conv = |name: &str| {
            CausalConv3d::from_weights(
                w,
                &format!("{prefix}.{name}"),
                dtype,
                (1, 1, 1),
                1,
                ENCODER_TEMPORAL_PADDING,
            )
        };
        Ok(Self {
            norm1: FrameIsolatedGroupNorm::from_weights(
                w,
                &format!("{prefix}.norm1"),
                dtype,
                groups,
                eps,
            )?,
            conv1: conv("conv1")?,
            norm2: FrameIsolatedGroupNorm::from_weights(
                w,
                &format!("{prefix}.norm2"),
                dtype,
                groups,
                eps,
            )?,
            conv2: conv("conv2")?,
            shortcut: if in_channels == out_channels {
                None
            } else {
                // A 1×1×1 kernel: no spatial padding and no temporal padding at all.
                Some(CausalConv3d::from_weights(
                    w,
                    &format!("{prefix}.conv_shortcut"),
                    dtype,
                    (1, 1, 1),
                    0,
                    0,
                )?)
            },
        })
    }

    /// The tensor names one block consumes.
    pub fn names(prefix: &str, in_channels: usize, out_channels: usize) -> Vec<String> {
        let mut v = FrameIsolatedGroupNorm::names(&format!("{prefix}.norm1"));
        v.extend(CausalConv3d::names(&format!("{prefix}.conv1")));
        v.extend(FrameIsolatedGroupNorm::names(&format!("{prefix}.norm2")));
        v.extend(CausalConv3d::names(&format!("{prefix}.conv2")));
        if in_channels != out_channels {
            v.extend(CausalConv3d::names(&format!("{prefix}.conv_shortcut")));
        }
        v
    }

    /// NCTHW in, NCTHW out.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv1.forward(&silu(&self.norm1.forward(x)?)?)?;
        let h = self.conv2.forward(&silu(&self.norm2.forward(&h)?)?)?;
        let residual = match &self.shortcut {
            Some(c) => c.forward(x)?,
            None => x.clone(),
        };
        Ok(residual.add(&h)?)
    }
}

/// One encoder level: `layers_per_block` residual blocks, then an optional strided downsampler.
#[derive(Debug, Clone)]
pub struct DownBlock3d {
    resnets: Vec<ResnetBlock3d>,
    downsampler: Option<CausalConv3d>,
    /// Whether the downsampler is preceded by the asymmetric bottom/right pad.
    pads_bottom_right: bool,
}

impl DownBlock3d {
    /// Load `down_blocks.{level}`.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &MiniMaxH3VaeConfig,
        dtype: DType,
        level: usize,
    ) -> Result<Self> {
        let in_channels = cfg.encoder_level_in_channels(level);
        let out_channels = cfg.block_out_channels[level];
        let resnets = (0..cfg.layers_per_block)
            .map(|j| {
                ResnetBlock3d::from_weights(
                    w,
                    &format!("{prefix}.resnets.{j}"),
                    cfg,
                    dtype,
                    if j == 0 { in_channels } else { out_channels },
                    out_channels,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let spatial = cfg.spatial_downsample_factors[level];
        let temporal = cfg.temporal_downsample_factors[level];
        let downsampler = if cfg.encoder_level_has_downsampler(level) {
            Some(CausalConv3d::from_weights(
                w,
                &format!("{prefix}.downsamplers.0.conv"),
                dtype,
                (temporal, spatial, spatial),
                0,
                ENCODER_TEMPORAL_PADDING,
            )?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsampler,
            pads_bottom_right: spatial == 2,
        })
    }

    /// The tensor names one level consumes.
    pub fn names(prefix: &str, cfg: &MiniMaxH3VaeConfig, level: usize) -> Vec<String> {
        let in_channels = cfg.encoder_level_in_channels(level);
        let out_channels = cfg.block_out_channels[level];
        let mut v = Vec::new();
        for j in 0..cfg.layers_per_block {
            v.extend(ResnetBlock3d::names(
                &format!("{prefix}.resnets.{j}"),
                if j == 0 { in_channels } else { out_channels },
                out_channels,
            ));
        }
        if cfg.encoder_level_has_downsampler(level) {
            v.extend(CausalConv3d::names(&format!(
                "{prefix}.downsamplers.0.conv"
            )));
        }
        v
    }

    /// NCTHW in, NCTHW out.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for r in &self.resnets {
            h = r.forward(&h)?;
        }
        if let Some(d) = &self.downsampler {
            if self.pads_bottom_right {
                // Asymmetric: bottom and right only, so the output is exactly `ceil(size / 2)`.
                h = reflect_pad_axis(&h, 3, 0, 1)?;
                h = reflect_pad_axis(&h, 4, 0, 1)?;
            }
            h = d.forward(&h)?;
        }
        Ok(h)
    }
}

/// The full CNN encoder: `conv_in → down_blocks → norm_out → silu → conv_out`.
#[derive(Debug, Clone)]
pub struct VideoEncoder3d {
    conv_in: CausalConv3d,
    down_blocks: Vec<DownBlock3d>,
    norm_out: FrameIsolatedGroupNorm,
    conv_out: CausalConv3d,
}

impl VideoEncoder3d {
    /// Load the encoder under `prefix` (`"encoder"` in the published checkpoint).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &MiniMaxH3VaeConfig,
        dtype: DType,
    ) -> Result<Self> {
        cfg.validate_encoder()?;
        Ok(Self {
            conv_in: CausalConv3d::from_weights(
                w,
                &format!("{prefix}.conv_in"),
                dtype,
                (1, 1, 1),
                1,
                ENCODER_TEMPORAL_PADDING,
            )?,
            down_blocks: (0..cfg.num_encoder_levels())
                .map(|i| {
                    DownBlock3d::from_weights(
                        w,
                        &format!("{prefix}.down_blocks.{i}"),
                        cfg,
                        dtype,
                        i,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            norm_out: FrameIsolatedGroupNorm::from_weights(
                w,
                &format!("{prefix}.norm_out"),
                dtype,
                cfg.norm_num_groups,
                cfg.encoder_norm_eps,
            )?,
            conv_out: CausalConv3d::from_weights(
                w,
                &format!("{prefix}.conv_out"),
                dtype,
                (1, 1, 1),
                1,
                ENCODER_TEMPORAL_PADDING,
            )?,
        })
    }

    /// Every tensor name the encoder consumes — the exhaustive mapping.
    pub fn names(prefix: &str, cfg: &MiniMaxH3VaeConfig) -> Vec<String> {
        let mut v = CausalConv3d::names(&format!("{prefix}.conv_in"));
        for i in 0..cfg.num_encoder_levels() {
            v.extend(DownBlock3d::names(
                &format!("{prefix}.down_blocks.{i}"),
                cfg,
                i,
            ));
        }
        v.extend(FrameIsolatedGroupNorm::names(&format!("{prefix}.norm_out")));
        v.extend(CausalConv3d::names(&format!("{prefix}.conv_out")));
        v
    }

    /// NCTHW `[B, 3, T, H, W]` → NCTHW `[B, 2·latent_channels, T', H/16, W/16]` — the posterior
    /// **parameters**, before `quant_conv`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(x)?;
        for b in &self.down_blocks {
            h = b.forward(&h)?;
        }
        self.conv_out.forward(&silu(&self.norm_out.forward(&h)?)?)
    }
}

// -------------------------------------------------------------------------------------------
// Spatial tiling
// -------------------------------------------------------------------------------------------

/// Where one axis's tiles start, how long each is, and the overlap between consecutive tiles.
///
/// This is the reference's `_split_tiles`. The tile count is the smallest whose union covers the
/// axis at the minimum overlap; the slack is then distributed round-robin over the overlaps in
/// whole `spatial_compression_ratio` steps, so every tile boundary stays latent-aligned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePlan {
    /// Pixel index each tile starts at.
    pub starts: Vec<usize>,
    /// Pixel length of each tile.
    pub lengths: Vec<usize>,
    /// Pixel overlap between tile `i` and tile `i + 1`; `starts.len() - 1` entries.
    pub overlaps: Vec<usize>,
}

impl TilePlan {
    /// Lay `tile_size`-wide tiles over `length` pixels at at least `min_overlap` overlap.
    pub fn split(
        length: usize,
        tile_size: usize,
        min_overlap: usize,
        spatial_compression_ratio: usize,
    ) -> Result<Self> {
        if length == 0 || tile_size == 0 || spatial_compression_ratio == 0 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 encoder: tile plan needs positive length/tile/ratio, got \
                 {length}/{tile_size}/{spatial_compression_ratio}"
            )));
        }
        if min_overlap >= tile_size {
            return Err(CandleError::Msg(format!(
                "minimax-h3 encoder: tile overlap {min_overlap} must be within [0, {tile_size})"
            )));
        }
        if tile_size >= length {
            return Ok(Self {
                starts: vec![0],
                lengths: vec![length],
                overlaps: Vec::new(),
            });
        }
        let mut num_tiles = length.div_ceil(tile_size);
        // `tile_size > min_overlap` above, so each extra tile adds `tile_size - min_overlap > 0`
        // coverage and this terminates.
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) < length {
            num_tiles += 1;
        }
        let mut overlaps = vec![min_overlap; num_tiles - 1];
        let remaining = tile_size * num_tiles - overlaps.iter().sum::<usize>() - length;
        for i in 0..(remaining / spatial_compression_ratio) {
            let slot = i % overlaps.len();
            overlaps[slot] += spatial_compression_ratio;
        }
        let mut starts = vec![0usize];
        for i in 0..num_tiles - 1 {
            starts.push(starts[i] + tile_size - overlaps[i]);
        }
        Ok(Self {
            starts,
            lengths: vec![tile_size; num_tiles],
            overlaps,
        })
    }

    /// Tiles on this axis.
    pub fn len(&self) -> usize {
        self.starts.len()
    }

    /// Whether the axis is a single untiled span.
    pub fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }
}

/// Blend and concatenate a grid of latent tiles back into one tensor — the reference's
/// `_stitch_tiles`. Overlaps are in **latent** units. Axes are `-2` (height) and `-1` (width).
pub fn stitch_tiles(
    tiles: &[Vec<Tensor>],
    height_overlaps: &[usize],
    width_overlaps: &[usize],
) -> Result<Tensor> {
    if tiles.is_empty() || tiles[0].is_empty() {
        return Err(CandleError::Msg(
            "minimax-h3 encoder: cannot stitch an empty tile grid".into(),
        ));
    }
    let rank = tiles[0][0].dims().len();
    if rank < 2 {
        return Err(CandleError::Msg(format!(
            "minimax-h3 encoder: a latent tile needs a height and a width axis, got rank {rank}"
        )));
    }
    let (h_axis, w_axis) = (rank - 2, rank - 1);
    let mut result_rows = Vec::with_capacity(tiles.len());
    for (i, row) in tiles.iter().enumerate() {
        let mut result_row: Vec<Tensor> = Vec::with_capacity(row.len());
        for (j, tile) in row.iter().enumerate() {
            let mut t = tile.clone();
            if i > 0 {
                t = blend(&tiles[i - 1][j], &t, height_overlaps[i - 1] as i32, h_axis)?;
            }
            if j > 0 {
                t = blend(&row[j - 1], &t, width_overlaps[j - 1] as i32, w_axis)?;
            }
            if i < tiles.len() - 1 {
                let n = t.dims()[h_axis];
                t = t.narrow(h_axis, 0, n - height_overlaps[i])?;
            }
            if j < row.len() - 1 {
                let n = t.dims()[w_axis];
                t = t.narrow(w_axis, 0, n - width_overlaps[j])?;
            }
            result_row.push(t.contiguous()?);
        }
        let refs: Vec<&Tensor> = result_row.iter().collect();
        result_rows.push(Tensor::cat(&refs, w_axis)?);
    }
    let refs: Vec<&Tensor> = result_rows.iter().collect();
    Ok(Tensor::cat(&refs, h_axis)?)
}

// -------------------------------------------------------------------------------------------
// The posterior
// -------------------------------------------------------------------------------------------

/// The posterior a `DiagonalGaussianDistribution` is parameterized by.
///
/// `mean` and `logvar` are the two channel halves of the `quant_conv` output; `logvar` is clamped
/// to `[-30, 20]` exactly as diffusers does, and the clamp is applied at construction so no caller
/// can forget it.
#[derive(Debug, Clone)]
pub struct DiagonalGaussian {
    mean: Tensor,
    std: Tensor,
}

impl DiagonalGaussian {
    /// Split `[B, 2C, T, H, W]` posterior parameters into mean and standard deviation.
    pub fn from_parameters(params: &Tensor) -> Result<Self> {
        let s = params.dims();
        if s.len() != 5 || !s[1].is_multiple_of(2) {
            return Err(CandleError::Msg(format!(
                "minimax-h3 encoder: posterior parameters must be [B, 2C, T, H, W], got {s:?}"
            )));
        }
        let c = s[1] / 2;
        let mean = params.narrow(1, 0, c)?.contiguous()?;
        let logvar = params
            .narrow(1, c, c)?
            .contiguous()?
            .clamp(LOGVAR_CLAMP.0, LOGVAR_CLAMP.1)?;
        Ok(Self {
            mean,
            std: (logvar * 0.5)?.exp()?,
        })
    }

    /// The distribution's mean — the sample at zero noise.
    pub fn mean(&self) -> &Tensor {
        &self.mean
    }

    /// The distribution's per-element standard deviation.
    pub fn std(&self) -> &Tensor {
        &self.std
    }

    /// `mean + std · noise`. `noise` must match the mean's shape.
    pub fn sample_with(&self, noise: &Tensor) -> Result<Tensor> {
        if noise.dims() != self.mean.dims() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 encoder: posterior noise {:?} does not match the mean {:?}",
                noise.dims(),
                self.mean.dims()
            )));
        }
        Ok(self
            .mean
            .add(&self.std.mul(&noise.to_dtype(self.std.dtype())?)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_gen::candle_core::Device;

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// Torch's reflect pad does **not** repeat the edge sample. Written against a ramp so an
    /// edge-replicating or symmetric ("reflect-101 off by one") implementation cannot pass.
    #[test]
    fn reflect_pad_matches_torch() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), &Device::Cpu).unwrap();
        // torch.nn.functional.pad(x, (2, 2), mode="reflect") == [3, 2, 1, 2, 3, 4, 3, 2]
        let p = reflect_pad_axis(&x, 1, 2, 2).unwrap();
        assert_eq!(flat(&p), vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]);
        // Asymmetric bottom/right-only pad — the downsampler's.
        let d = reflect_pad_axis(&x, 1, 0, 1).unwrap();
        assert_eq!(flat(&d), vec![1.0, 2.0, 3.0, 4.0, 3.0]);
        // A pad that would reflect off an already-reflected sample is rejected, as torch does.
        assert!(reflect_pad_axis(&x, 1, 4, 0).is_err());
        assert!(reflect_pad_axis(&x, 1, 0, 4).is_err());
        assert!(reflect_pad_axis(&x, 7, 1, 1).is_err(), "axis out of range");
        // Zero pad is the identity.
        assert_eq!(flat(&reflect_pad_axis(&x, 1, 0, 0).unwrap()), flat(&x));
    }

    /// The causal pad is zeros, at the FRONT, and it changes the temporal extent.
    #[test]
    fn causal_pad_is_front_only_zeros() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 1, 2, 1, 1), &Device::Cpu).unwrap();
        let p = zero_pad_front_time(&x, 2).unwrap();
        assert_eq!(p.dims(), &[1, 1, 4, 1, 1]);
        assert_eq!(flat(&p), vec![0.0, 0.0, 1.0, 2.0]);
        // A non-NCTHW tensor is a typed error, not a silent reshape.
        let flatx = Tensor::from_vec(vec![1.0f32], 1, &Device::Cpu).unwrap();
        assert!(zero_pad_front_time(&flatx, 1).is_err());
        // Zero frames is the identity.
        assert_eq!(flat(&zero_pad_front_time(&x, 0).unwrap()), flat(&x));
    }

    /// The conv2d-tap decomposition **is** a 3-D convolution, checked against a brute-force
    /// cross-correlation over the same volume — including a temporal stride, which none of the
    /// sibling crates' conv3d twins carry.
    #[test]
    fn conv3d_taps_match_a_brute_force_cross_correlation() {
        let dev = Device::Cpu;
        let (b, c, t, h, w) = (2usize, 3usize, 7usize, 5usize, 6usize);
        let (o, kt, kh, kw) = (4usize, 3usize, 3usize, 3usize);
        let xs: Vec<f32> = (0..b * c * t * h * w)
            .map(|i| ((i as f32) * 0.37).sin())
            .collect();
        let ws: Vec<f32> = (0..o * c * kt * kh * kw)
            .map(|i| ((i as f32) * 0.11).cos() * 0.4)
            .collect();
        let bs: Vec<f32> = (0..o).map(|i| i as f32 * 0.05).collect();
        let x = Tensor::from_vec(xs.clone(), (b, c, t, h, w), &dev).unwrap();
        let weight = Tensor::from_vec(ws.clone(), (o, c, kt, kh, kw), &dev).unwrap();
        let bias = Tensor::from_vec(bs.clone(), o, &dev).unwrap();

        for stride in [(1usize, 1usize, 1usize), (2, 2, 2), (2, 1, 1), (1, 2, 2)] {
            let (st, sh, _) = stride;
            let got = conv3d_ncthw(&x, &weight, Some(&bias), stride).unwrap();
            let (t_out, h_out, w_out) = ((t - kt) / st + 1, (h - kh) / sh + 1, (w - kw) / sh + 1);
            assert_eq!(got.dims(), &[b, o, t_out, h_out, w_out], "{stride:?}");

            let mut want = vec![0f32; b * o * t_out * h_out * w_out];
            for bi in 0..b {
                for oi in 0..o {
                    for ti in 0..t_out {
                        for hi in 0..h_out {
                            for wi in 0..w_out {
                                let mut acc = bs[oi];
                                for ci in 0..c {
                                    for a in 0..kt {
                                        for p in 0..kh {
                                            for q in 0..kw {
                                                let xi = ((bi * c + ci) * t + ti * st + a) * h;
                                                let xi = (xi + hi * sh + p) * w + wi * sh + q;
                                                let wi_ =
                                                    (((oi * c + ci) * kt + a) * kh + p) * kw + q;
                                                acc += xs[xi] * ws[wi_];
                                            }
                                        }
                                    }
                                }
                                want[((bi * o + oi) * t_out + ti) * h_out * w_out
                                    + hi * w_out
                                    + wi] = acc;
                            }
                        }
                    }
                }
            }
            let max_err = flat(&got)
                .iter()
                .zip(&want)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                max_err < 1e-4,
                "stride {stride:?}: max abs err {max_err:.3e}"
            );
        }
    }

    /// The im2col batch chunking must not change the answer — it exists only to keep candle's CUDA
    /// launches under `u32::MAX`.
    #[test]
    fn im2col_chunking_matches_an_unchunked_conv2d() {
        let dev = Device::Cpu;
        let x = Tensor::randn(0f32, 1f32, (6, 4, 9, 11), &dev).unwrap();
        let w = Tensor::randn(0f32, 1f32, (5, 4, 3, 3), &dev).unwrap();
        let want = x.conv2d(&w, 0, 1, 1, 1).unwrap();
        // A budget of 1 forces one frame per launch.
        let got = chunked_conv2d(&x, &w, 1, 1).unwrap();
        assert_eq!(got.dims(), want.dims());
        let max_err = flat(&got)
            .iter()
            .zip(flat(&want).iter())
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(max_err < 1e-5, "chunked conv diverged: {max_err:.3e}");
        assert!(checked_product(&[usize::MAX, 2], "probe").is_err());
    }

    /// A conv3d cannot invent samples: a kernel that does not fit the (already padded) volume is a
    /// typed error, and mismatched channels or spatial strides are too.
    #[test]
    fn conv3d_rejects_geometry_it_cannot_express() {
        let dev = Device::Cpu;
        let x = Tensor::zeros((1, 3, 2, 4, 4), DType::F32, &dev).unwrap();
        let w = Tensor::zeros((2, 3, 3, 3, 3), DType::F32, &dev).unwrap();
        assert!(
            conv3d_ncthw(&x, &w, None, (1, 1, 1)).is_err(),
            "kt=3 does not fit T=2"
        );
        let ok = Tensor::zeros((1, 3, 3, 4, 4), DType::F32, &dev).unwrap();
        assert!(conv3d_ncthw(&ok, &w, None, (1, 1, 1)).is_ok());
        assert!(
            conv3d_ncthw(&ok, &w, None, (1, 2, 1)).is_err(),
            "unequal spatial strides"
        );
        assert!(
            conv3d_ncthw(&ok, &w, None, (0, 1, 1)).is_err(),
            "zero stride"
        );
        let wrong_c = Tensor::zeros((2, 5, 3, 3, 3), DType::F32, &dev).unwrap();
        assert!(
            conv3d_ncthw(&ok, &wrong_c, None, (1, 1, 1)).is_err(),
            "channel mismatch"
        );
    }

    /// `_split_tiles`, against the reference's own measured output at the shipped canvas sizes.
    ///
    /// These rows were read off the running reference (`_split_tiles(768, 256, 64)` etc.), so they
    /// pin the round-robin slack distribution and not just the tile count.
    #[test]
    fn tile_plan_matches_the_reference() {
        let plan =
            |n| TilePlan::split(n, TILE_SAMPLE_MIN_SIZE, TILE_SAMPLE_MIN_OVERLAP, 16).unwrap();

        let p = plan(768);
        assert_eq!(p.starts, vec![0, 160, 336, 512]);
        assert_eq!(p.lengths, vec![256; 4]);
        assert_eq!(p.overlaps, vec![96, 80, 80]);

        let p = plan(1344);
        assert_eq!(p.starts, vec![0, 176, 352, 528, 704, 896, 1088]);
        assert_eq!(p.overlaps, vec![80, 80, 80, 80, 64, 64]);

        // A canvas no wider than one tile is a single untiled span with no overlaps.
        let p = plan(256);
        assert_eq!(p.starts, vec![0]);
        assert_eq!(p.lengths, vec![256]);
        assert!(p.overlaps.is_empty());

        let p = plan(2048);
        assert_eq!(p.len(), 11);
        assert_eq!(p.overlaps, vec![80, 80, 80, 80, 80, 80, 80, 80, 64, 64]);

        // Every tile boundary stays latent-aligned: the slack is distributed in whole ratio steps.
        for n in [768, 1344, 2048, 512, 1024, 320] {
            let p = plan(n);
            assert!(
                p.starts.iter().all(|s| s.is_multiple_of(16)),
                "{n}: tile starts must be latent-aligned, got {:?}",
                p.starts
            );
            let last = p.starts.last().unwrap() + p.lengths.last().unwrap();
            assert_eq!(last, n, "{n}: the tiles must exactly cover the axis");
        }
        assert!(TilePlan::split(0, 256, 64, 16).is_err());
        assert!(
            TilePlan::split(768, 256, 256, 16).is_err(),
            "overlap >= tile"
        );
        assert!(TilePlan::split(768, 0, 64, 16).is_err());
        assert!(TilePlan::split(768, 256, 64, 0).is_err());
    }

    /// Stitching a grid that was split with zero overlap is exact concatenation, and a
    /// single-tile grid is the identity.
    #[test]
    fn stitching_is_exact_without_overlap() {
        let dev = Device::Cpu;
        let t = |v: Vec<f32>| Tensor::from_vec(v, (1, 1, 1, 2, 2), &dev).unwrap();
        let a = t(vec![1.0, 2.0, 3.0, 4.0]);
        let single = stitch_tiles(&[vec![a.clone()]], &[], &[]).unwrap();
        assert_eq!(flat(&single), flat(&a));

        let b = t(vec![5.0, 6.0, 7.0, 8.0]);
        let row = stitch_tiles(&[vec![a.clone(), b.clone()]], &[], &[0]).unwrap();
        assert_eq!(row.dims(), &[1, 1, 1, 2, 4]);
        assert_eq!(flat(&row), vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);

        let col = stitch_tiles(&[vec![a.clone()], vec![b.clone()]], &[0], &[]).unwrap();
        assert_eq!(col.dims(), &[1, 1, 1, 4, 2]);
        assert_eq!(flat(&col), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        assert!(stitch_tiles(&[], &[], &[]).is_err());
        let rank1 = Tensor::zeros(3, DType::F32, &dev).unwrap();
        assert!(stitch_tiles(&[vec![rank1]], &[], &[]).is_err());
    }

    /// The posterior clamps `logvar` and `sample_with(0)` is the mean.
    #[test]
    fn posterior_clamps_logvar_and_samples_around_the_mean() {
        let dev = Device::Cpu;
        // [B=1, 2C=2, T=1, H=1, W=1]: mean 3.0, logvar 100.0 (clamped to 20).
        let p = Tensor::from_vec(vec![3.0f32, 100.0], (1, 2, 1, 1, 1), &dev).unwrap();
        let d = DiagonalGaussian::from_parameters(&p).unwrap();
        assert_eq!(flat(d.mean()), vec![3.0]);
        let std = flat(d.std())[0];
        assert!(
            (std - (0.5f32 * LOGVAR_CLAMP.1 as f32).exp()).abs() < 1e-2,
            "logvar must clamp to {}, got std {std}",
            LOGVAR_CLAMP.1
        );
        // The lower clamp too — an unclamped -100 would underflow to exactly 0.
        let low = Tensor::from_vec(vec![0.0f32, -100.0], (1, 2, 1, 1, 1), &dev).unwrap();
        let dl = DiagonalGaussian::from_parameters(&low).unwrap();
        assert!(
            (flat(dl.std())[0] - (0.5f32 * LOGVAR_CLAMP.0 as f32).exp()).abs() < 1e-6,
            "logvar must clamp at {}",
            LOGVAR_CLAMP.0
        );

        let zero = Tensor::zeros((1, 1, 1, 1, 1), DType::F32, &dev).unwrap();
        assert_eq!(flat(&d.sample_with(&zero).unwrap()), flat(d.mean()));
        // A shape mismatch is a typed error, not a broadcast.
        let wrong = Tensor::zeros((1, 1, 1, 2, 1), DType::F32, &dev).unwrap();
        assert!(d.sample_with(&wrong).is_err());
        let rank1 = Tensor::zeros(1, DType::F32, &dev).unwrap();
        assert!(
            DiagonalGaussian::from_parameters(&rank1).is_err(),
            "rank is checked"
        );
        let odd = Tensor::zeros((1, 3, 1, 1, 1), DType::F32, &dev).unwrap();
        assert!(
            DiagonalGaussian::from_parameters(&odd).is_err(),
            "an odd channel count cannot split into mean and logvar"
        );
    }

    /// Tiling being **on** is a default, and defaults get assumed rather than read.
    #[test]
    fn tiling_is_on_by_default() {
        const { assert!(ENCODER_TILING_IS_ON_BY_DEFAULT) };
        assert_eq!(TILE_SAMPLE_MIN_SIZE, 256);
        assert_eq!(TILE_SAMPLE_MIN_OVERLAP, 64);
        assert_eq!(ENCODER_TEMPORAL_PADDING, 2);
    }
}
