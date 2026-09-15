//! The MiniMax-H3 video VAE **encode** half — the 3-D causal CNN
//! (`MiniMaxH3VideoEncoder3d` + `quant_conv`), 118 published tensors.
//!
//! sc-17140 ported `decode` only and recorded the encode half as deliberately unported. `fl2va`
//! (sc-17148) needs it: a keyframe is conditioned through **both** the text encoder's vision tower
//! *and* this encoder, so there is no keyframe conditioning without it.
//!
//! The two halves share nothing structurally. The decoder is a 36-layer ViT
//! ([`crate::decoder::ViT3dDecoder`]); the encoder is a plain conv stack with no attention and no
//! mid-block. What it does have are four conventions that are each individually easy to get wrong
//! and impossible to see afterwards:
//!
//! # 1. Temporal padding is FRONT-ONLY ZEROS, and it is one frame more than causal-minimal
//!
//! [`CausalConv3d`] pads `temporal_padding` zero frames before the clip and none after. Every
//! 3×3×3 conv in the encoder uses `temporal_padding = 2` against `kernel_t = 3` — one *more* than
//! the `kernel_t - 1 = 2`… which is the same number here, but the value is carried explicitly
//! rather than derived, because the reference carries it explicitly and a derived version would
//! silently disagree the moment a kernel changed.
//!
//! The consequence that matters: **encoding a single frame is not the same as encoding one frame
//! out of a longer clip.** A `T = 1` input sees only zeros before it. That is exactly the
//! conditioning MiniMax-H3 was trained with for keyframes, and [`crate::vae::MiniMaxH3VideoVae::encode`] takes
//! the reference's `num_frames == 1` short circuit rather than padding up to `clip_length`.
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
//! [`FrameIsolatedGroupNorm`] reshapes `T` into the batch axis before normalizing. A standard
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
//! Both now live in [`crate::spatial_tiling`] and are re-exported here. sc-17148 ported them for
//! encode only; **decode tiles from the same flag and the same geometry** (sc-18786), so keeping
//! two copies would let the halves drift. The one asymmetry between the halves is the units the
//! stitch runs in — latent here, pixel there — and it is documented at [`stitch_tiles`].
//!
//! # What this module does NOT do
//!
//! It stops at the posterior parameters. Sampling the posterior, the fp16 round-trip and the
//! per-channel latent normalization are conditioning policy, not VAE structure, and live in
//! [`crate::conditioning`].

use mlx_rs::ops::{concatenate_axis, exp, maximum, minimum};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{conv3d, group_norm, silu};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::MiniMaxH3VaeConfig;
// `_split_tiles` / `_stitch_tiles` and the shipped geometry live in [`crate::spatial_tiling`]
// because **decode tiles too** (sc-18786) and the two halves must not drift apart. Re-exported
// here so the sc-17148 call sites and fixtures keep their paths.
pub use crate::spatial_tiling::{
    stitch_tiles, TilePlan, TILE_SAMPLE_MIN_OVERLAP, TILE_SAMPLE_MIN_SIZE, TILING_IS_ON_BY_DEFAULT,
};

/// Temporal padding every 3×3×3 encoder conv applies, front-only. The reference's literal.
pub const ENCODER_TEMPORAL_PADDING: i32 = 2;

/// Whether the shipped VAE encodes with spatial tiling enabled.
///
/// Encoding a 768×1344 keyframe untiled is a 28-tile difference in the result. Encode and decode
/// share one `use_tiling` flag upstream, so this is an alias for
/// [`crate::spatial_tiling::TILING_IS_ON_BY_DEFAULT`] kept for the sc-17148 call sites.
pub const ENCODER_TILING_IS_ON_BY_DEFAULT: bool = TILING_IS_ON_BY_DEFAULT;

/// Reflect-pad `x` along `axis` by `before` / `after`, matching `torch.nn.functional.pad(mode =
/// "reflect")`.
///
/// The reflection does **not** repeat the edge sample: padded position `j` before the start maps to
/// source `before - j`, and position `k` past the end maps to source `n - 2 - k`. Torch requires
/// the pad to be strictly smaller than the axis, and so does this — a larger pad would need to
/// reflect off an already-reflected sample, which torch rejects rather than defines.
pub fn reflect_pad_axis(x: &Array, axis: i32, before: i32, after: i32) -> Result<Array> {
    if before == 0 && after == 0 {
        return Ok(x.clone());
    }
    if before < 0 || after < 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 encoder: reflect pad must be non-negative, got {before}/{after}"
        )));
    }
    let rank = x.shape().len() as i32;
    if axis < 0 || axis >= rank {
        return Err(Error::Msg(format!(
            "minimax-h3 encoder: reflect pad axis {axis} is outside a rank-{rank} tensor"
        )));
    }
    let n = x.shape()[axis as usize];
    if before >= n || after >= n {
        return Err(Error::Msg(format!(
            "minimax-h3 encoder: cannot reflect-pad {before}/{after} off an axis of length {n}; \
             torch requires the pad to be smaller than the axis"
        )));
    }
    let mut idx: Vec<i32> = Vec::with_capacity((before + n + after) as usize);
    for j in (1..=before).rev() {
        idx.push(j);
    }
    idx.extend(0..n);
    for k in 0..after {
        idx.push(n - 2 - k);
    }
    let index = Array::from_slice(&idx, &[idx.len() as i32]);
    Ok(x.take_axis(&index, axis)?)
}

/// Prepend `frames` zero frames along the temporal axis of an NDHWC tensor — the causal pad.
pub fn zero_pad_front_time(x: &Array, frames: i32) -> Result<Array> {
    if frames == 0 {
        return Ok(x.clone());
    }
    if frames < 0 {
        return Err(Error::Msg(format!(
            "minimax-h3 encoder: temporal pad must be non-negative, got {frames}"
        )));
    }
    let s = x.shape();
    if s.len() != 5 {
        return Err(Error::Msg(format!(
            "minimax-h3 encoder: expected an NDHWC tensor to causal-pad, got {s:?}"
        )));
    }
    let pad = mlx_rs::ops::zeros_dtype(&[s[0], frames, s[2], s[3], s[4]], x.dtype())?;
    Ok(concatenate_axis(&[pad, x.clone()], 1)?)
}

/// NCTHW `[B, C, T, H, W]` → NDHWC `[B, T, H, W, C]`.
fn to_ndhwc(x: &Array) -> Result<Array> {
    Ok(x.transpose_axes(&[0, 2, 3, 4, 1])?)
}

/// NDHWC `[B, T, H, W, C]` → NCTHW `[B, C, T, H, W]`.
fn to_ncthw(x: &Array) -> Result<Array> {
    Ok(x.transpose_axes(&[0, 4, 1, 2, 3])?)
}

/// A stored torch `Conv3d` weight `[out, in, kT, kH, kW]` → mlx NDHWC `[out, kT, kH, kW, in]`.
fn conv3d_weight(w: &Array) -> Result<Array> {
    let s = w.shape();
    if s.len() != 5 {
        return Err(Error::Msg(format!(
            "minimax-h3 encoder: a Conv3d weight must be rank 5, got {s:?}"
        )));
    }
    Ok(w.transpose_axes(&[0, 2, 3, 4, 1])?)
}

/// A 3-D convolution that is **causal in time** (front-only zero padding) and **reflecting in
/// space**.
#[derive(Debug, Clone)]
pub struct CausalConv3d {
    weight: Array,
    bias: Array,
    stride: (i32, i32, i32),
    spatial_padding: i32,
    temporal_padding: i32,
}

impl CausalConv3d {
    /// Load `{prefix}.weight` / `{prefix}.bias`.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        dtype: Dtype,
        stride: (i32, i32, i32),
        spatial_padding: i32,
        temporal_padding: i32,
    ) -> Result<Self> {
        Ok(Self {
            weight: conv3d_weight(&w.require(&format!("{prefix}.weight"))?.as_dtype(dtype)?)?,
            bias: w.require(&format!("{prefix}.bias"))?.as_dtype(dtype)?,
            stride,
            spatial_padding,
            temporal_padding,
        })
    }

    /// The two tensor names this conv consumes.
    pub fn names(prefix: &str) -> Vec<String> {
        vec![format!("{prefix}.weight"), format!("{prefix}.bias")]
    }

    /// NDHWC in, NDHWC out.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = x.clone();
        if self.spatial_padding > 0 {
            h = reflect_pad_axis(&h, 2, self.spatial_padding, self.spatial_padding)?;
            h = reflect_pad_axis(&h, 3, self.spatial_padding, self.spatial_padding)?;
        }
        h = zero_pad_front_time(&h, self.temporal_padding)?;
        conv3d(&h, &self.weight, Some(&self.bias), self.stride, (0, 0, 0))
    }
}

/// `GroupNorm` whose statistics are computed **per latent frame in isolation**.
///
/// The reference folds `T` into the batch axis (`view(B·T, C, 1, H, W)`) before normalizing, so no
/// statistic ever mixes across time. See the module docs for why substituting a plain 3-D
/// `GroupNorm` is invisible at `T = 1`.
#[derive(Debug, Clone)]
pub struct FrameIsolatedGroupNorm {
    weight: Array,
    bias: Array,
    groups: i32,
    eps: f32,
}

impl FrameIsolatedGroupNorm {
    /// Load `{prefix}.weight` / `{prefix}.bias`.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        dtype: Dtype,
        groups: i32,
        eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{prefix}.weight"))?.as_dtype(dtype)?,
            bias: w.require(&format!("{prefix}.bias"))?.as_dtype(dtype)?,
            groups,
            eps,
        })
    }

    /// The two tensor names this norm consumes.
    pub fn names(prefix: &str) -> Vec<String> {
        vec![format!("{prefix}.weight"), format!("{prefix}.bias")]
    }

    /// NDHWC in, NDHWC out.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let s = x.shape();
        if s.len() != 5 {
            return Err(Error::Msg(format!(
                "minimax-h3 encoder: frame-isolated GroupNorm expects NDHWC, got {s:?}"
            )));
        }
        // Fold T into the batch axis so `group_norm`'s per-sample statistics are per-frame.
        let folded = x.reshape(&[s[0] * s[1], s[2], s[3], s[4]])?;
        let normed = group_norm(&folded, &self.weight, &self.bias, self.groups, self.eps)?;
        Ok(normed.reshape(&[s[0], s[1], s[2], s[3], s[4]])?)
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
        dtype: Dtype,
        in_channels: i32,
        out_channels: i32,
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
    pub fn names(prefix: &str, in_channels: i32, out_channels: i32) -> Vec<String> {
        let mut v = FrameIsolatedGroupNorm::names(&format!("{prefix}.norm1"));
        v.extend(CausalConv3d::names(&format!("{prefix}.conv1")));
        v.extend(FrameIsolatedGroupNorm::names(&format!("{prefix}.norm2")));
        v.extend(CausalConv3d::names(&format!("{prefix}.conv2")));
        if in_channels != out_channels {
            v.extend(CausalConv3d::names(&format!("{prefix}.conv_shortcut")));
        }
        v
    }

    /// NDHWC in, NDHWC out.
    pub fn forward(&self, x: &Array) -> Result<Array> {
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
        dtype: Dtype,
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

    /// NDHWC in, NDHWC out.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = x.clone();
        for r in &self.resnets {
            h = r.forward(&h)?;
        }
        if let Some(d) = &self.downsampler {
            if self.pads_bottom_right {
                // Asymmetric: bottom and right only, so the output is exactly `ceil(size / 2)`.
                h = reflect_pad_axis(&h, 2, 0, 1)?;
                h = reflect_pad_axis(&h, 3, 0, 1)?;
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
        dtype: Dtype,
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
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = self.conv_in.forward(&to_ndhwc(x)?)?;
        for b in &self.down_blocks {
            h = b.forward(&h)?;
        }
        h = self.conv_out.forward(&silu(&self.norm_out.forward(&h)?)?)?;
        to_ncthw(&h)
    }
}

/// The posterior a `DiagonalGaussianDistribution` is parameterized by.
///
/// `mean` and `logvar` are the two channel halves of the `quant_conv` output; `logvar` is clamped
/// to `[-30, 20]` exactly as diffusers does, and the clamp is applied at construction so no caller
/// can forget it.
#[derive(Debug, Clone)]
pub struct DiagonalGaussian {
    mean: Array,
    std: Array,
}

/// The `logvar` clamp diffusers' `DiagonalGaussianDistribution` applies.
pub const LOGVAR_CLAMP: (f32, f32) = (-30.0, 20.0);

impl DiagonalGaussian {
    /// Split `[B, 2C, T, H, W]` posterior parameters into mean and standard deviation.
    pub fn from_parameters(params: &Array) -> Result<Self> {
        let s = params.shape();
        if s.len() != 5 || s[1] % 2 != 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 encoder: posterior parameters must be [B, 2C, T, H, W], got {s:?}"
            )));
        }
        let c = s[1] / 2;
        let mean = crate::tensor::slice_axis(params, 1, 0, c)?;
        let raw_logvar = crate::tensor::slice_axis(params, 1, c, s[1])?;
        let logvar = minimum(
            &maximum(&raw_logvar, Array::from_f32(LOGVAR_CLAMP.0))?,
            Array::from_f32(LOGVAR_CLAMP.1),
        )?;
        Ok(Self {
            mean,
            std: exp(&logvar.multiply(Array::from_f32(0.5))?)?,
        })
    }

    /// The distribution's mean — the sample at zero noise.
    pub fn mean(&self) -> &Array {
        &self.mean
    }

    /// The distribution's per-element standard deviation.
    pub fn std(&self) -> &Array {
        &self.std
    }

    /// `mean + std · noise`. `noise` must match the mean's shape.
    pub fn sample_with(&self, noise: &Array) -> Result<Array> {
        if noise.shape() != self.mean.shape() {
            return Err(Error::Msg(format!(
                "minimax-h3 encoder: posterior noise {:?} does not match the mean {:?}",
                noise.shape(),
                self.mean.shape()
            )));
        }
        Ok(self.mean.add(&self.std.multiply(noise)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // `_blend` now lives with the rest of the temporal/spatial cross-fade in `blocks`; this module
    // no longer carries its own copy.
    use crate::blocks::blend;

    /// Torch's reflect pad does **not** repeat the edge sample. Written against a ramp so an
    /// edge-replicating or symmetric ("reflect-101 off by one") implementation cannot pass.
    #[test]
    fn reflect_pad_matches_torch() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4]);
        // torch.nn.functional.pad(x, (2, 2), mode="reflect") == [3, 2, 1, 2, 3, 4, 3, 2]
        let p = reflect_pad_axis(&x, 1, 2, 2).unwrap();
        assert_eq!(
            p.as_slice::<f32>(),
            &[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        );
        // Asymmetric bottom/right-only pad — the downsampler's.
        let d = reflect_pad_axis(&x, 1, 0, 1).unwrap();
        assert_eq!(d.as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0, 3.0]);
        // A pad that would reflect off an already-reflected sample is rejected, as torch does.
        assert!(reflect_pad_axis(&x, 1, 4, 0).is_err());
        assert!(reflect_pad_axis(&x, 1, 0, 4).is_err());
        assert!(reflect_pad_axis(&x, 7, 1, 1).is_err(), "axis out of range");
        // Zero pad is the identity.
        assert_eq!(
            reflect_pad_axis(&x, 1, 0, 0).unwrap().as_slice::<f32>(),
            x.as_slice::<f32>()
        );
    }

    /// The causal pad is zeros, at the FRONT, and it changes the temporal extent.
    #[test]
    fn causal_pad_is_front_only_zeros() {
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2, 1, 1, 1]);
        let p = zero_pad_front_time(&x, 2).unwrap();
        assert_eq!(p.shape(), &[1, 4, 1, 1, 1]);
        assert_eq!(p.as_slice::<f32>(), &[0.0, 0.0, 1.0, 2.0]);
        // Asserted by MESSAGE (sc-19488): a negative pad reaches `zeros_dtype` as a negative
        // dimension with the guard deleted, which mlx rejects on its own.
        let msg = zero_pad_front_time(&x, -1)
            .expect_err("a negative temporal pad must be refused")
            .to_string();
        assert!(
            msg.contains("minimax-h3 encoder: temporal pad must be non-negative"),
            "the sign guard must be what rejects this, not a negative-dimension fault: {msg}"
        );
        // A non-NDHWC tensor is a typed error, not a silent reshape.
        assert!(zero_pad_front_time(&Array::from_slice(&[1.0f32], &[1]), 1).is_err());
    }

    /// `_split_tiles`, against the reference's own measured output at the shipped canvas sizes.
    ///
    /// These four rows were read off the running reference (`_split_tiles(768, 256, 64)` etc.), so
    /// they pin the round-robin slack distribution and not just the tile count.
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
                p.starts.iter().all(|s| s % 16 == 0),
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
    }

    /// Stitching a grid that was split with zero overlap is exact concatenation, and a
    /// single-tile grid is the identity.
    #[test]
    fn stitching_is_exact_without_overlap() {
        let t = |v: &[f32]| Array::from_slice(v, &[1, 1, 1, 2, 2]);
        let a = t(&[1.0, 2.0, 3.0, 4.0]);
        let single = stitch_tiles(&[vec![a.clone()]], &[], &[]).unwrap();
        assert_eq!(single.as_slice::<f32>(), a.as_slice::<f32>());

        let b = t(&[5.0, 6.0, 7.0, 8.0]);
        let row = stitch_tiles(&[vec![a.clone(), b.clone()]], &[], &[0]).unwrap();
        assert_eq!(row.shape(), &[1, 1, 1, 2, 4]);
        assert_eq!(
            row.as_slice::<f32>(),
            &[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]
        );
        assert!(stitch_tiles(&[], &[], &[]).is_err());
    }

    /// The cross-fade is linear and weights the two tiles as `1 - i/n` / `i/n`.
    #[test]
    fn blend_is_a_linear_cross_fade() {
        let a = Array::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[1, 4]);
        let b = Array::from_slice(&[10.0f32, 10.0, 10.0, 10.0], &[1, 4]);
        let out = blend(&a, &b, 4, 1).unwrap();
        // i/4 · 10 for i in 0..4 — starts at a and ramps to (but never reaches) b.
        assert_eq!(out.as_slice::<f32>(), &[0.0, 2.5, 5.0, 7.5]);
        // Zero extent leaves b untouched.
        assert_eq!(
            blend(&a, &b, 0, 1).unwrap().as_slice::<f32>(),
            b.as_slice::<f32>()
        );
    }

    /// The posterior clamps `logvar` and `sample_with(0)` is the mean.
    #[test]
    fn posterior_clamps_logvar_and_samples_around_the_mean() {
        // [B=1, 2C=2, T=1, H=1, W=1]: mean 3.0, logvar 100.0 (clamped to 20).
        let p = Array::from_slice(&[3.0f32, 100.0], &[1, 2, 1, 1, 1]);
        let d = DiagonalGaussian::from_parameters(&p).unwrap();
        assert_eq!(d.mean().as_slice::<f32>(), &[3.0]);
        let std = d.std().as_slice::<f32>()[0];
        assert!(
            (std - (0.5f32 * LOGVAR_CLAMP.1).exp()).abs() < 1e-2,
            "logvar must clamp to {} , got std {std}",
            LOGVAR_CLAMP.1
        );
        let zero = mlx_rs::ops::zeros_dtype(&[1, 1, 1, 1, 1], p.dtype()).unwrap();
        assert_eq!(
            d.sample_with(&zero).unwrap().as_slice::<f32>(),
            d.mean().as_slice::<f32>()
        );
        // A shape mismatch is a typed error, not a broadcast.
        let wrong = mlx_rs::ops::zeros_dtype(&[1, 1, 1, 2, 1], p.dtype()).unwrap();
        assert!(d.sample_with(&wrong).is_err());
        assert!(
            DiagonalGaussian::from_parameters(&Array::from_slice(&[1.0f32], &[1])).is_err(),
            "rank is checked"
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
