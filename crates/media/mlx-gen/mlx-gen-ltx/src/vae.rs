//! S2 — the LTX-2.3 **video VAE** (causal 3-D conv autoencoder, latent 128-ch, patch 4, 8× temporal
//! / 32× spatial). Port of the `mlx_video` reference `models/ltx/video_vae/*`, gated against it
//! (`tests/vae_parity.rs`, real `vae_{decoder,encoder}.safetensors`).
//!
//! Two distinct sub-nets, both **structure-from-config / channels-from-weights**:
//!  - **Decoder** = the reference `LTX2VideoDecoder` (`decoder.py`): `conv_in 128→1024` → 9
//!    `up_blocks` (`ResBlockGroup` + `DepthToSpaceUpsample`) → pixel-norm → SiLU → `conv_out 128→48`
//!    → unpatchify(×4). The shipped 2.3 checkpoint sets `timestep_conditioning=false`, so the
//!    decode-time noise / per-block scale-shift modulation are **inert** (no such weights) — kept
//!    config-gated, off, not dropped.
//!  - **Encoder** = the reference `VideoEncoder` (`video_vae.py`): patchify(×4) → `conv_in 48→128`
//!    → 9 `down_blocks` (`UNetMidBlock3D` + `SpaceToDepthDownsample`) → PixelNorm → SiLU →
//!    `conv_out 1024→129` → `normalize(out[:, :128])`. Wired here for the I2V sibling.
//!
//! Reference quirks carried over verbatim:
//!  - Tensors stay **NCTHW** (channels-first) throughout, transposing to channels-last only inside
//!    the conv op (mlx convs are channels-last). Conv weights are the on-disk MLX layout
//!    `[O, kt, kh, kw, I]` — no transpose.
//!  - `CausalConv3d` pads time by **frame replication**: causal → first frame ×(kt−1) at the start;
//!    non-causal → first frame ×(kt−1)/2 at the start *and* last frame ×(kt−1)/2 at the end. Spatial
//!    pad is symmetric `(k−1)/2` **zeros** (2.3 `spatial_padding_mode="zeros"`). The **encoder runs
//!    causal**, the **decoder non-causal** (`generate_av.py` calls `vae_decoder(latents)` →
//!    `causal=False`).
//!  - `pixel_norm` = `x / sqrt(mean(x² over C) + eps)` over axis 1 (no √C, no γ). Decoder eps
//!    **1e-8**, encoder PixelNorm eps **1e-6**.
//!
//! Everything runs **f32** (quality target; the reference VAE is gated in f32). The parity gate
//! honors "divergence is not rounding" — any >1% gap gets root-caused, not written off.

use std::path::PathBuf;
use std::sync::OnceLock;

use mlx_rs::ops::{add, concatenate_axis, divide, mean_axes, multiply, pad, subtract, sum_axes};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{conv3d, silu};
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::{CancelFlag, Error, Result};

use crate::config::{LatentLogVar, LtxVaeConfig, VaeBlock};
use crate::contiguous;
use mlx_gen::tiling::{TilingConfig, VaeTiling};

/// Decoder inline `pixel_norm` epsilon (`decoder.py` `ResnetBlock3DSimple.pixel_norm`).
const DEC_NORM_EPS: f32 = 1e-8;
/// Encoder `PixelNorm` epsilon (`get_norm_layer(..., eps=1e-6)`).
const ENC_NORM_EPS: f32 = 1e-6;

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Load a conv weight+bias (`{prefix}.weight`, `{prefix}.bias`) cast to f32. The on-disk tensors are
/// already MLX conv layout `[O, kt, kh, kw, I]`, so no transpose is needed.
fn f32(w: &Weights, key: &str) -> Result<Array> {
    to_dtype(w.require(key)?, Dtype::Float32)
}

/// Slice `x` along `axis` to `[start, end)`.
fn slice_axis(x: &Array, axis: i32, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[end - start]), axis)?)
}

/// Temporal slice `x[:, :, start:end]` (axis 2).
fn slice_t(x: &Array, start: i32, end: i32) -> Result<Array> {
    slice_axis(x, 2, start, end)
}

/// `x / sqrt(mean(x² over C, axis 1, keepdims) + eps)` — LTX PixelNorm (no √C scale, no γ).
fn pixel_norm(x: &Array, eps: f32) -> Result<Array> {
    // A zero-channel tensor (malformed checkpoint) would divide by `c = 0` and silently yield
    // inf/nan; a rank<2 tensor would panic on `shape()[1]`. Reject both (F-049).
    if x.ndim() < 2 || x.shape()[1] == 0 {
        return Err(Error::Msg(format!(
            "ltx vae pixel_norm: expected a non-empty channel axis (dim 1), got shape {:?}",
            x.shape()
        )));
    }
    let sumsq = sum_axes(&multiply(x, x)?, &[1], true)?;
    let c = x.shape()[1] as f32;
    let mean = divide(&sumsq, scalar(c))?;
    let denom = add(&mean, scalar(eps))?.sqrt()?;
    Ok(divide(x, &denom)?)
}

/// 3-D conv with frame-replication temporal padding + symmetric spatial zero-pad. NCTHW I/O; weight
/// is the on-disk MLX `[O, kt, kh, kw, I]`. `causal` toggles left-only vs symmetric temporal pad.
struct CausalConv3d {
    w: Array,
    b: Array,
    kt: i32,
    ph: i32,
    pw: i32,
}

impl CausalConv3d {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let weight = f32(w, &format!("{prefix}.weight"))?;
        let sh = weight.shape(); // [O, kt, kh, kw, I]
        let (kt, kh, kw) = (sh[1], sh[2], sh[3]);
        Ok(Self {
            w: weight,
            b: f32(w, &format!("{prefix}.bias"))?,
            kt,
            ph: (kh - 1) / 2,
            pw: (kw - 1) / 2,
        })
    }

    /// Replicate `frame` (a `[B, C, 1, H, W]` slice) `n` times along the temporal axis.
    fn repeat_frame(frame: &Array, n: i32) -> Result<Array> {
        let parts: Vec<&Array> = (0..n).map(|_| frame).collect();
        Ok(concatenate_axis(&parts, 2)?)
    }

    fn forward(&self, x_ncthw: &Array, causal: bool) -> Result<Array> {
        let mut x = x_ncthw.clone();
        if self.kt > 1 {
            let t = x.shape()[2];
            if causal {
                let first = slice_t(&x, 0, 1)?;
                let pad_front = Self::repeat_frame(&first, self.kt - 1)?;
                x = concatenate_axis(&[&pad_front, &x], 2)?;
            } else {
                let ps = (self.kt - 1) / 2;
                if ps > 0 {
                    let first = slice_t(&x, 0, 1)?;
                    let last = slice_t(&x, t - 1, t)?;
                    let pad_front = Self::repeat_frame(&first, ps)?;
                    let pad_back = Self::repeat_frame(&last, ps)?;
                    x = concatenate_axis(&[&pad_front, &x, &pad_back], 2)?;
                }
            }
        }
        if self.ph > 0 || self.pw > 0 {
            x = pad(
                &x,
                &[
                    (0, 0),
                    (0, 0),
                    (0, 0),
                    (self.ph, self.ph),
                    (self.pw, self.pw),
                ][..],
                None,
                None,
            )?;
        }
        let x = x.transpose_axes(&[0, 2, 3, 4, 1])?; // NDHWC
        let y = conv3d(&x, &self.w, Some(&self.b), (1, 1, 1), (0, 0, 0))?;
        Ok(y.transpose_axes(&[0, 4, 1, 2, 3])?) // NCTHW
    }
}

// ---------------------------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------------------------

/// Decoder residual block (`ResnetBlock3DSimple`, timestep off): pixel-norm → SiLU → conv → repeat
/// → residual add. Channels constant (no shortcut). Inline pixel-norm eps 1e-8.
struct DecResBlock {
    conv1: CausalConv3d,
    conv2: CausalConv3d,
}

impl DecResBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv1: CausalConv3d::from_weights(w, &format!("{prefix}.conv1.conv"))?,
            conv2: CausalConv3d::from_weights(w, &format!("{prefix}.conv2.conv"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = silu(&pixel_norm(x, DEC_NORM_EPS)?)?;
        let h = self.conv1.forward(&h, false)?;
        let h = silu(&pixel_norm(&h, DEC_NORM_EPS)?)?;
        let h = self.conv2.forward(&h, false)?;
        Ok(add(&h, x)?)
    }
}

/// `DepthToSpaceUpsample` with `residual=false`: conv → depth-to-space → (st>1) drop first temporal
/// frame. `stride = (st, sh, sw)`; out-channels = conv_out / prod(stride).
struct DepthToSpace {
    conv: CausalConv3d,
    st: i32,
    sh: i32,
    sw: i32,
}

impl DepthToSpace {
    fn from_weights(w: &Weights, prefix: &str, stride: (i32, i32, i32)) -> Result<Self> {
        Ok(Self {
            conv: CausalConv3d::from_weights(w, &format!("{prefix}.conv.conv"))?,
            st: stride.0,
            sh: stride.1,
            sw: stride.2,
        })
    }

    /// `(B, C·st·sh·sw, D, H, W) -> (B, C, D·st, H·sh, W·sw)`.
    fn depth_to_space(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, c_packed, d, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let (st, shp, swp) = (self.st, self.sh, self.sw);
        let c = c_packed / (st * shp * swp);
        let x = x.reshape(&[b, c, st, shp, swp, d, h, w])?;
        let x = x.transpose_axes(&[0, 1, 5, 2, 6, 3, 7, 4])?;
        Ok(x.reshape(&[b, c, d * st, h * shp, w * swp])?)
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.conv.forward(x, false)?;
        let x = self.depth_to_space(&x)?;
        if self.st > 1 {
            let t = x.shape()[2];
            slice_t(&x, 1, t)
        } else {
            Ok(x)
        }
    }
}

enum UpLayer {
    Res(Vec<DecResBlock>),
    Up(DepthToSpace),
}

/// The LTX-2.3 video decoder (`LTX2VideoDecoder`, timestep conditioning off).
struct VideoDecoder {
    conv_in: CausalConv3d,
    up_blocks: Vec<UpLayer>,
    conv_out: CausalConv3d,
    patch_size: i32,
    mean: Array, // [1, C, 1, 1, 1]
    std: Array,  // [1, C, 1, 1, 1]
}

impl VideoDecoder {
    fn from_weights(w: &Weights, cfg: &LtxVaeConfig) -> Result<Self> {
        let mut up_blocks = Vec::new();
        // `decoder_blocks` is listed in encoder order; the decoder execution path reverses it.
        for (idx, block) in cfg.decoder_blocks.iter().rev().enumerate() {
            let prefix = format!("up_blocks.{idx}");
            up_blocks.push(build_up_layer(w, &prefix, block)?);
        }
        let c = cfg.latent_channels;
        let mean = f32(w, "per_channel_statistics.mean")?.reshape(&[1, c, 1, 1, 1])?;
        let std = f32(w, "per_channel_statistics.std")?.reshape(&[1, c, 1, 1, 1])?;
        Ok(Self {
            conv_in: CausalConv3d::from_weights(w, "conv_in.conv")?,
            up_blocks,
            conv_out: CausalConv3d::from_weights(w, "conv_out.conv")?,
            patch_size: cfg.patch_size,
            mean,
            std,
        })
    }

    /// `(B, C, F', H', W')` normalized latent → `(B, 3, F, H, W)` video. Non-causal, deterministic.
    fn decode(&self, latent: &Array) -> Result<Array> {
        // Denormalize: x · std + mean.
        let x = add(&multiply(latent, &self.std)?, &self.mean)?;
        let mut x = self.conv_in.forward(&x, false)?;
        for layer in &self.up_blocks {
            x = match layer {
                UpLayer::Res(blocks) => {
                    let mut h = x;
                    for b in blocks {
                        h = b.forward(&h)?;
                    }
                    h
                }
                UpLayer::Up(u) => u.forward(&x)?,
            };
        }
        let x = pixel_norm(&x, DEC_NORM_EPS)?;
        let x = silu(&x)?;
        let x = self.conv_out.forward(&x, false)?;
        unpatchify(&x, self.patch_size)
    }
}

fn build_up_layer(w: &Weights, prefix: &str, block: &VaeBlock) -> Result<UpLayer> {
    if block.is_compress() {
        Ok(UpLayer::Up(DepthToSpace::from_weights(
            w,
            prefix,
            block.stride(),
        )?))
    } else {
        let mut blocks = Vec::new();
        for j in 0..block.num_layers {
            blocks.push(DecResBlock::from_weights(
                w,
                &format!("{prefix}.res_blocks.{j}"),
            )?);
        }
        Ok(UpLayer::Res(blocks))
    }
}

// ---------------------------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------------------------

/// Encoder residual block (`ResnetBlock3D`, PixelNorm eps 1e-6): norm → SiLU → conv → norm → SiLU →
/// conv → residual. Channels constant within a `UNetMidBlock3D` (no shortcut). Causal.
struct EncResBlock {
    conv1: CausalConv3d,
    conv2: CausalConv3d,
}

impl EncResBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            conv1: CausalConv3d::from_weights(w, &format!("{prefix}.conv1.conv"))?,
            conv2: CausalConv3d::from_weights(w, &format!("{prefix}.conv2.conv"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = silu(&pixel_norm(x, ENC_NORM_EPS)?)?;
        let h = self.conv1.forward(&h, true)?;
        let h = silu(&pixel_norm(&h, ENC_NORM_EPS)?)?;
        let h = self.conv2.forward(&h, true)?;
        Ok(add(&h, x)?)
    }
}

/// `SpaceToDepthDownsample`: conv → space-to-depth (conv branch) + group-mean(space-to-depth(input))
/// skip. `out_channels` and `group_size` are derived from the conv weight + stride.
struct SpaceToDepth {
    conv: CausalConv3d,
    st: i32,
    sh: i32,
    sw: i32,
    out_channels: i32,
    group_size: i32,
}

impl SpaceToDepth {
    fn from_weights(w: &Weights, prefix: &str, stride: (i32, i32, i32)) -> Result<Self> {
        let conv = CausalConv3d::from_weights(w, &format!("{prefix}.conv.conv"))?;
        let csh = conv.w.shape(); // [O, kt, kh, kw, I]
        let (conv_out, in_channels) = (csh[0], csh[4]);
        let mult = stride.0 * stride.1 * stride.2;
        let out_channels = conv_out * mult; // after space-to-depth on the conv branch
        let group_size = in_channels * mult / out_channels;
        Ok(Self {
            conv,
            st: stride.0,
            sh: stride.1,
            sw: stride.2,
            out_channels,
            group_size,
        })
    }

    /// `(B, C, D, H, W) -> (B, C·st·sh·sw, D/st, H/sh, W/sw)`.
    fn space_to_depth(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, c, d, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let (st, shp, swp) = (self.st, self.sh, self.sw);
        let x = x.reshape(&[b, c, d / st, st, h / shp, shp, w / swp, swp])?;
        let x = x.transpose_axes(&[0, 1, 3, 5, 7, 2, 4, 6])?;
        Ok(x.reshape(&[b, c * st * shp * swp, d / st, h / shp, w / swp])?)
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        // Causal temporal pad: duplicate the first frame when downsampling time.
        if self.st == 2 {
            let first = slice_t(&x, 0, 1)?;
            x = concatenate_axis(&[&first, &x], 2)?;
        }
        // Pad the tail so D/H/W are divisible by stride (zeros, end-only).
        let sh = x.shape();
        let (d, h, w) = (sh[2], sh[3], sh[4]);
        let pad_d = (self.st - d % self.st) % self.st;
        let pad_h = (self.sh - h % self.sh) % self.sh;
        let pad_w = (self.sw - w % self.sw) % self.sw;
        if pad_d > 0 || pad_h > 0 || pad_w > 0 {
            x = pad(
                &x,
                &[(0, 0), (0, 0), (0, pad_d), (0, pad_h), (0, pad_w)][..],
                None,
                None,
            )?;
        }

        // Skip branch: space-to-depth on the input, then mean over the group axis.
        let x_in = self.space_to_depth(&x)?;
        let si = x_in.shape();
        let (b2, d2, h2, w2) = (si[0], si[2], si[3], si[4]);
        let x_in = x_in.reshape(&[b2, self.out_channels, self.group_size, d2, h2, w2])?;
        let x_in = mean_axes(&x_in, &[2], false)?; // (B, out_channels, D', H', W')

        // Conv branch: conv → space-to-depth.
        let x_conv = self.conv.forward(&x, true)?;
        let x_conv = self.space_to_depth(&x_conv)?;

        if x_conv.shape() == x_in.shape() {
            Ok(add(&x_conv, &x_in)?)
        } else {
            Ok(x_conv)
        }
    }
}

enum DownLayer {
    Res(Vec<EncResBlock>),
    Down(SpaceToDepth),
}

/// The LTX video encoder (`VideoEncoder`). Causal throughout. The log-variance head is
/// config-declared ([`LatentLogVar`]) and only ever affects how wide `conv_out` is — the encoder
/// output is `normalize(conv_out[:, :latent_channels])` in every mode the reference builds a
/// `(means, logvar)` pair from.
struct VideoEncoder {
    conv_in: CausalConv3d,
    down_blocks: Vec<DownLayer>,
    conv_out: CausalConv3d,
    patch_size: i32,
    latent_channels: i32,
    mean: Array, // [1, C, 1, 1, 1]
    std: Array,  // [1, C, 1, 1, 1]
}

impl VideoEncoder {
    fn from_weights(w: &Weights, cfg: &LtxVaeConfig) -> Result<Self> {
        let mut down_blocks = Vec::new();
        for (idx, block) in cfg.encoder_blocks.iter().enumerate() {
            let prefix = format!("down_blocks.{idx}");
            down_blocks.push(if block.is_compress() {
                DownLayer::Down(SpaceToDepth::from_weights(w, &prefix, block.stride())?)
            } else {
                let mut blocks = Vec::new();
                for j in 0..block.num_layers {
                    blocks.push(EncResBlock::from_weights(
                        w,
                        &format!("{prefix}.res_blocks.{j}"),
                    )?);
                }
                DownLayer::Res(blocks)
            });
        }
        let c = cfg.latent_channels;
        let mean = f32(w, "per_channel_statistics._mean_of_means")?.reshape(&[1, c, 1, 1, 1])?;
        let std = f32(w, "per_channel_statistics._std_of_means")?.reshape(&[1, c, 1, 1, 1])?;
        let conv_out = CausalConv3d::from_weights(w, "conv_out.conv")?;

        // sc-18765 — the declared log-variance mode fixes `conv_out`'s width. Check the weights
        // against it: a mismatch means the config and the checkpoint disagree about the shape of the
        // encoder head, and taking the first `latent_channels` channels anyway would hand back a
        // plausible-looking latent built from the wrong slice.
        let expected = cfg.latent_log_var.conv_out_channels(c);
        let actual = conv_out.w.shape()[0];
        if actual != expected {
            return Err(Error::Msg(format!(
                "ltx vae encoder: conv_out emits {actual} channels but latent_log_var={:?} with \
                 latent_channels={c} requires {expected}",
                cfg.latent_log_var
            )));
        }
        if cfg.latent_log_var == LatentLogVar::None {
            // `none` leaves no log-variance tail, so the reference's own `torch.chunk(sample, 2)`
            // would split the MEANS in half and return a `latent_channels / 2`-wide latent. No
            // shipped LTX checkpoint declares it, and guessing which of the two readings is intended
            // is not this port's call to make.
            return Err(Error::Msg(
                "ltx vae encoder: latent_log_var=\"none\" is not supported (the reference's own \
                 means/logvar split is ill-defined without a log-variance head)"
                    .into(),
            ));
        }

        Ok(Self {
            conv_in: CausalConv3d::from_weights(w, "conv_in.conv")?,
            down_blocks,
            conv_out,
            patch_size: cfg.patch_size,
            latent_channels: c,
            mean,
            std,
        })
    }

    /// `(B, 3, F, H, W)` video (F = 1 + 8·k, values in [-1, 1]) → `(B, 128, F', H/32, W/32)`
    /// normalized latent means. Causal, deterministic: the output is `normalize(means)` where the
    /// means are the leading `latent_channels` of `conv_out`, for every supported
    /// [`LatentLogVar`] mode. The log-variance tail is sliced off and never computed.
    fn encode(&self, video: &Array) -> Result<Array> {
        let mut x = patchify(video, self.patch_size)?;
        x = self.conv_in.forward(&x, true)?;
        for layer in &self.down_blocks {
            x = match layer {
                DownLayer::Res(blocks) => {
                    let mut h = x;
                    for b in blocks {
                        h = b.forward(&h)?;
                    }
                    h
                }
                DownLayer::Down(d) => d.forward(&x)?,
            };
        }
        let x = pixel_norm(&x, ENC_NORM_EPS)?;
        let x = silu(&x)?;
        // conv_out width is `LatentLogVar::conv_out_channels` (checked at load). The output is
        // normalize(means) where means = the first `latent_channels`; the log-variance tail —
        // whatever its width and meaning under the declared mode — is discarded.
        let x = self.conv_out.forward(&x, true)?;
        let means = slice_c(&x, 0, self.latent_channels)?;
        let normed = divide(&subtract(&means, &self.mean)?, &self.std)?;
        Ok(normed)
    }
}

/// Channel slice `x[:, start:end]` (axis 1).
fn slice_c(x: &Array, start: i32, end: i32) -> Result<Array> {
    slice_axis(x, 1, start, end)
}

// ---------------------------------------------------------------------------------------------
// patchify / unpatchify (spatial-only, patch_size_t = 1)
// ---------------------------------------------------------------------------------------------

/// `(B, C, F, H, W) -> (B, C·p², F, H/p, W/p)` (reference `ops.patchify`, `patch_size_t=1`).
pub(crate) fn patchify(x: &Array, p: i32) -> Result<Array> {
    let sh = x.shape();
    let (b, c, f, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
    let (nh, nw) = (h / p, w / p);
    // (B, C, F, 1, H/p, p, W/p, p) -> transpose (0,1,3,7,5,2,4,6) -> (B, C·p·p, F, H/p, W/p).
    let x = x.reshape(&[b, c, f, 1, nh, p, nw, p])?;
    let x = x.transpose_axes(&[0, 1, 3, 7, 5, 2, 4, 6])?;
    Ok(x.reshape(&[b, c * p * p, f, nh, nw])?)
}

/// `(B, C·p², F, H, W) -> (B, C, F, H·p, W·p)` (reference `ops.unpatchify`, `patch_size_t=1`).
pub(crate) fn unpatchify(x: &Array, p: i32) -> Result<Array> {
    let sh = x.shape();
    let (b, c_packed, f, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
    let c = c_packed / (p * p);
    // (B, C, 1, p, p, F, H, W) -> transpose (0,1,5,2,6,4,7,3) -> (B, C, F, H·p, W·p).
    let x = x.reshape(&[b, c, 1, p, p, f, h, w])?;
    let x = x.transpose_axes(&[0, 1, 5, 2, 6, 4, 7, 3])?;
    Ok(x.reshape(&[b, c, f, h * p, w * p])?)
}

// ---------------------------------------------------------------------------------------------
// Public VAE
// ---------------------------------------------------------------------------------------------

/// The LTX video VAE: a decoder and an encoder, each present only when its weights are (pure T2V
/// uses only `decode`; the I2V sibling adds the encoder; an LTX-2.5 `CausalDiffusionVAE` bundle
/// supplies only this conv encoder, its decoder being a different architecture entirely).
pub struct LtxVideoVae {
    /// The conv decoder. `None` for an encoder-only VAE ([`Self::encoder_only`]) — `decode` then
    /// errors rather than there being a decoder-shaped hole to fall into.
    decoder: Option<VideoDecoder>,
    /// The encoder — populated eagerly by [`from_weights`], or lazily on first [`encode`] from
    /// [`lazy`](Self::lazy). `OnceLock` (not `OnceCell`) so the VAE keeps its prior `Send`/`Sync`.
    encoder: OnceLock<VideoEncoder>,
    /// Deferred encoder source (`vae_encoder.safetensors` path + cfg). `Some` ⇒ build on first
    /// `encode`; `None` ⇒ the encoder was supplied eagerly (or is absent). F-048.
    lazy: Option<(PathBuf, LtxVaeConfig)>,
}

impl LtxVideoVae {
    /// Geometry owned by the concrete decoder and consumed by its planner and tiled driver.
    pub const VAE_TILING: VaeTiling = VaeTiling::LTX;

    /// Build the decoder from `decoder_w` and (optionally) the encoder from `encoder_w` — both eager.
    pub fn from_weights(
        decoder_w: &Weights,
        encoder_w: Option<&Weights>,
        cfg: &LtxVaeConfig,
    ) -> Result<Self> {
        let decoder = VideoDecoder::from_weights(decoder_w, cfg)?;
        let encoder = OnceLock::new();
        if let Some(w) = encoder_w {
            // A fresh cell is empty, so `set` always succeeds.
            let _ = encoder.set(VideoEncoder::from_weights(w, cfg)?);
        }
        Ok(Self {
            decoder: Some(decoder),
            encoder,
            lazy: None,
        })
    }

    /// Build an **encoder-only** VAE (sc-18765). LTX-2.5's `CausalDiffusionVAE` checkpoint carries
    /// the same conv encoder as the conv VAE — differing only in the declared
    /// [`LatentLogVar`] — but pairs it with an `NADiffusionDecoder`, so there is no conv decoder to
    /// load alongside it. [`Self::decode`] and [`Self::decode_tiled`] return an error on the result.
    pub fn encoder_only(encoder_w: &Weights, cfg: &LtxVaeConfig) -> Result<Self> {
        let encoder = OnceLock::new();
        let _ = encoder.set(VideoEncoder::from_weights(encoder_w, cfg)?);
        Ok(Self {
            decoder: None,
            encoder,
            lazy: None,
        })
    }

    /// Like [`Self::from_weights`] but **defers** loading the encoder until the first [`Self::encode`]:
    /// `vae_encoder.safetensors` (hundreds of MB) is never read for pure-T2V runs, which never encode
    /// (F-048). The decoder is built eagerly (every mode decodes). A missing encoder file surfaces as
    /// a clear error on the first `encode`, not at load.
    pub fn from_weights_lazy_encoder(
        decoder_w: &Weights,
        encoder_path: PathBuf,
        cfg: &LtxVaeConfig,
    ) -> Result<Self> {
        let decoder = VideoDecoder::from_weights(decoder_w, cfg)?;
        Ok(Self {
            decoder: Some(decoder),
            encoder: OnceLock::new(),
            lazy: Some((encoder_path, cfg.clone())),
        })
    }

    /// The encoder, building it from the deferred source on first use (F-048). Errors when neither an
    /// eager encoder nor a lazy source is present.
    fn encoder(&self) -> Result<&VideoEncoder> {
        if let Some(e) = self.encoder.get() {
            return Ok(e);
        }
        let (path, cfg) = self
            .lazy
            .as_ref()
            .ok_or_else(|| Error::Msg("LtxVideoVae: encode requires encoder weights".into()))?;
        let w = Weights::from_file(path)?;
        let enc = VideoEncoder::from_weights(&w, cfg)?;
        // Single producer here; `set` on the (still-empty) cell succeeds, and even under a race the
        // winner's value is returned by the `get` below.
        let _ = self.encoder.set(enc);
        Ok(self.encoder.get().expect("encoder just set"))
    }

    /// The conv decoder, or a message-bearing error for an [encoder-only](Self::encoder_only) VAE.
    fn decoder(&self) -> Result<&VideoDecoder> {
        self.decoder.as_ref().ok_or_else(|| {
            Error::Msg(
                "LtxVideoVae: decode requires conv decoder weights (this VAE was built \
                 encoder-only — an LTX-2.5 CausalDiffusionVAE checkpoint decodes through its \
                 NADiffusionDecoder, not through this port)"
                    .into(),
            )
        })
    }

    /// Decode a normalized latent `(B, 128, F', H', W')` → video `(B, 3, F, 32·H', 32·W')` in
    /// roughly [-1, 1] (the caller clips + scales to uint8). Non-causal single pass.
    pub fn decode(&self, latent: &Array) -> Result<Array> {
        contiguous(&self.decoder()?.decode(latent)?)
    }

    /// Whether the VAE can decode — i.e. it was built with conv decoder weights.
    pub fn has_decoder(&self) -> bool {
        self.decoder.is_some()
    }

    /// Encode a video `(B, 3, F, H, W)` (F = 1 + 8·k, [-1, 1]) → normalized latent
    /// `(B, 128, F', H/32, W/32)`. Causal. Requires encoder weights.
    pub fn encode(&self, video: &Array) -> Result<Array> {
        contiguous(&self.encoder()?.encode(video)?)
    }

    /// Decode with **tiling** for memory-bounded large/long-video decode (`cfg`). Splits the latent
    /// into overlapping spatial/temporal tiles, decodes each, and trapezoidally blends them. Falls
    /// back to the single-pass [`decode`](Self::decode) when `cfg` does not fire for these dims.
    ///
    /// `cancel` (F-051): checked once per tile. The per-tile `eval` already materializes the
    /// accumulators, so the check observes real progress (no lazy-eval false green — the check runs
    /// after the tile's compute is forced), and a cancel during the minutes-long full-envelope decode
    /// is honored within one tile rather than the whole video. Returns [`Error::Canceled`] on trip.
    ///
    /// The assembly itself is the shared
    /// [`tiled_decode_with_hooks`](mlx_gen::vae_tiling::tiled_decode_with_hooks) (sc-18320): LTX
    /// carried a private copy of the same slice/blend/place/accumulate loop, so the two drifted
    /// independently — the sc-19655 per-tile cache release lived only here, and the sc-18320
    /// per-axis fold would have had to be written twice. The hooks are what LTX actually needs from
    /// its own copy: force every live accumulator, then hand MLX's dead buffers back to the OS with
    /// only those accumulators live.
    pub fn decode_tiled(
        &self,
        latent: &Array,
        cfg: &TilingConfig,
        cancel: &CancelFlag,
    ) -> Result<Array> {
        let sh = latent.shape();
        let (f, h, w) = (sh[2], sh[3], sh[4]);
        if !cfg.needs_tiling(Self::VAE_TILING, f, h, w) {
            return self.decode(latent);
        }
        let plan = cfg.plan(Self::VAE_TILING, f, h, w);
        let decoder = self.decoder()?;

        mlx_gen::vae_tiling::tiled_decode_with_hooks(
            latent,
            &plan,
            [2, 3, 4],
            Some(cancel),
            |tile| decoder.decode(tile),
            |accumulators| {
                for accumulator in accumulators {
                    accumulator.eval()?;
                }
                Ok(())
            },
            mlx_rs::memory::clear_cache,
        )
    }

    /// Whether the VAE can encode — either the encoder is already built or a lazy source is set.
    pub fn has_encoder(&self) -> bool {
        self.encoder.get().is_some() || self.lazy.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_spatial_decode(tile: &Array) -> Result<Array> {
        let up_h = Array::repeat_axis::<f32>(tile.clone(), 32, 3)?;
        Ok(Array::repeat_axis::<f32>(up_h, 32, 4)?)
    }

    fn synthetic_tiled_case() -> (Array, mlx_gen::tiling::TilePlan) {
        let shape = [1, 3, 1, 5, 5];
        let count: i32 = shape.iter().product();
        let values: Vec<f32> = (0..count).map(|value| value as f32 * 0.125).collect();
        let latent = Array::from_slice(&values, &shape);
        let config = TilingConfig {
            spatial: Some(mlx_gen::tiling::SpatialTiling {
                tile_px: 96,
                overlap_px: 32,
            }),
            temporal: None,
        };
        assert!(config.needs_tiling(LtxVideoVae::VAE_TILING, 1, 5, 5));
        let plan = config.plan(LtxVideoVae::VAE_TILING, 1, 5, 5);
        (latent, plan)
    }

    #[test]
    fn tiled_decode_releases_dead_buffers_once_per_tile_and_preserves_parity() {
        let (latent, plan) = synthetic_tiled_case();
        let expected = synthetic_spatial_decode(&latent).unwrap();
        let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let decode_events = std::rc::Rc::clone(&events);
        let materialize_events = std::rc::Rc::clone(&events);
        let release_events = std::rc::Rc::clone(&events);
        let got = mlx_gen::vae_tiling::tiled_decode_with_hooks(
            &latent,
            &plan,
            [2, 3, 4],
            None,
            move |tile| {
                decode_events.borrow_mut().push("decode");
                synthetic_spatial_decode(tile)
            },
            move |accumulators: &[&Array]| {
                materialize_events.borrow_mut().push("materialize");
                for accumulator in accumulators {
                    accumulator.eval()?;
                }
                Ok(())
            },
            move || release_events.borrow_mut().push("release"),
        )
        .unwrap();
        got.eval().unwrap();
        expected.eval().unwrap();

        let tile_count = plan.t.len() * plan.h.len() * plan.w.len();
        assert!(tile_count > 1, "fixture must exercise multiple releases");
        assert_eq!(
            events.borrow().as_slice(),
            ["decode", "materialize", "release"].repeat(tile_count),
            "every tile must materialize before release, and release before the next decode"
        );
        assert_eq!(got.shape(), expected.shape());
        let max_delta = got
            .as_slice::<f32>()
            .iter()
            .zip(expected.as_slice::<f32>())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            max_delta < 1e-4,
            "tiled blend parity drifted by {max_delta}"
        );
    }

    #[test]
    fn tiled_decode_cancel_and_error_keep_release_boundaries_exact() {
        let (latent, plan) = synthetic_tiled_case();
        let cancel = CancelFlag::new();
        cancel.cancel();
        let decodes = std::cell::Cell::new(0usize);
        let releases = std::cell::Cell::new(0usize);
        let canceled = mlx_gen::vae_tiling::tiled_decode_with_hooks(
            &latent,
            &plan,
            [2, 3, 4],
            Some(&cancel),
            |tile| {
                decodes.set(decodes.get() + 1);
                synthetic_spatial_decode(tile)
            },
            |accumulators: &[&Array]| {
                for accumulator in accumulators {
                    accumulator.eval()?;
                }
                Ok(())
            },
            || releases.set(releases.get() + 1),
        );
        assert!(matches!(canceled, Err(Error::Canceled)));
        assert_eq!(decodes.get(), 0, "pre-tripped cancel runs no tile");
        assert_eq!(releases.get(), 0, "no tile means no release callback");

        let releases = std::cell::Cell::new(0usize);
        let failed = mlx_gen::vae_tiling::tiled_decode_with_hooks(
            &latent,
            &plan,
            [2, 3, 4],
            None,
            |_tile| Err(Error::Msg("synthetic decoder failure".into())),
            |_: &[&Array]| unreachable!("failed decode cannot produce accumulators"),
            || releases.set(releases.get() + 1),
        );
        assert!(
            matches!(failed, Err(Error::Msg(message)) if message == "synthetic decoder failure")
        );
        assert_eq!(
            releases.get(),
            1,
            "the failed tile still releases buffers after its locals drop"
        );

        let cancel = CancelFlag::new();
        let cancel_after_release = cancel.clone();
        let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let decode_events = std::rc::Rc::clone(&events);
        let materialize_events = std::rc::Rc::clone(&events);
        let release_events = std::rc::Rc::clone(&events);
        let canceled = mlx_gen::vae_tiling::tiled_decode_with_hooks(
            &latent,
            &plan,
            [2, 3, 4],
            Some(&cancel),
            |tile| {
                decode_events.borrow_mut().push("decode");
                synthetic_spatial_decode(tile)
            },
            |accumulators: &[&Array]| {
                materialize_events.borrow_mut().push("materialize");
                for accumulator in accumulators {
                    accumulator.eval()?;
                }
                Ok(())
            },
            || {
                let mut events = release_events.borrow_mut();
                events.push("release");
                if events.iter().filter(|event| **event == "release").count() == 1 {
                    cancel_after_release.cancel();
                }
            },
        );
        assert!(matches!(canceled, Err(Error::Canceled)));
        assert_eq!(
            events.borrow().as_slice(),
            ["decode", "materialize", "release", "release"],
            "one tile completes and releases before cancellation drops and releases its live accumulators"
        );

        let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        let decode_events = std::rc::Rc::clone(&events);
        let materialize_events = std::rc::Rc::clone(&events);
        let release_events = std::rc::Rc::clone(&events);
        let decodes = std::cell::Cell::new(0usize);
        let failed = mlx_gen::vae_tiling::tiled_decode_with_hooks(
            &latent,
            &plan,
            [2, 3, 4],
            None,
            |tile| {
                decodes.set(decodes.get() + 1);
                decode_events.borrow_mut().push("decode");
                if decodes.get() == 2 {
                    Err(Error::Msg("second synthetic decoder failure".into()))
                } else {
                    synthetic_spatial_decode(tile)
                }
            },
            |accumulators: &[&Array]| {
                materialize_events.borrow_mut().push("materialize");
                for accumulator in accumulators {
                    accumulator.eval()?;
                }
                Ok(())
            },
            || release_events.borrow_mut().push("release"),
        );
        assert!(
            matches!(failed, Err(Error::Msg(message)) if message == "second synthetic decoder failure")
        );
        assert_eq!(
            events.borrow().as_slice(),
            ["decode", "materialize", "release", "decode", "release"],
            "the failed tile skips materialization and gets only the one fail-closed cleanup release"
        );
    }

    // ------------------------------------------------------------------------------------------
    // sc-19753 — the LTX decoder's tiling is normalization-correct BY CONSTRUCTION; proof below.
    //
    // The sc-19753 rule is that globally-scoped ops (GroupNorm, spatial-softmax attention,
    // squeeze-and-excite / global pools) must run once on the whole latent and only spatially-local
    // work may be tiled. `LtxVideoVae::decode_tiled` tiles the WHOLE decoder — which is safe only
    // because this decoder has no globally-scoped op at all: it contains no attention (grep the
    // decoder half of this file), and its only normalization is [`pixel_norm`], whose reduction is
    // `sum_axes(x·x, &[1], true)` — the CHANNEL axis alone, per (b, t, h, w) position. Everything
    // else is convs, depth-to-space, unpatchify, SiLU and residual adds, all spatially local.
    //
    // These tests prove that rather than asserting it, each with its executed mutation control.
    // ------------------------------------------------------------------------------------------

    /// `a[:, :, :, :, 0..width]` — a left-hand spatial crop on the W axis.
    fn crop_w(a: &Array, width: i32) -> Array {
        contiguous(&slice_axis(a, 4, 0, width).unwrap()).unwrap()
    }

    fn max_abs_delta(left: &Array, right: &Array) -> f32 {
        left.as_slice::<f32>()
            .iter()
            .zip(right.as_slice::<f32>())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max)
    }

    /// A position-dependent NCTHW tensor whose *amplitude* grows across W, so a spatial reduction
    /// taken over a left-hand crop is visibly different from one taken over the whole tensor (a
    /// merely additive ramp barely moves a mean-square, and would make the controls below toothless).
    fn ramped(shape: [i32; 5]) -> Array {
        let (h, w) = (shape[3], shape[4]);
        let count: i32 = shape.iter().product();
        let data = (0..count)
            .map(|i| {
                let y = (i / w % h) as f32;
                let x = (i % w) as f32;
                (((i as f32) * 0.071).sin() + y * 0.31) * (1.0 + 0.6 * x)
            })
            .collect::<Vec<_>>();
        Array::from_slice(&data, &shape)
    }

    /// The **mutation control** used throughout: [`pixel_norm`] with its channel-axis reduction
    /// replaced by a GroupNorm-shaped reduction over (C, H, W). Everything else is identical.
    fn spatially_normalized(x: &Array, eps: f32) -> Result<Array> {
        let sumsq = mean_axes(&multiply(x, x)?, &[1, 3, 4], true)?;
        let denom = add(&sumsq, scalar(eps))?.sqrt()?;
        Ok(divide(x, &denom)?)
    }

    /// **`pixel_norm` commutes with spatial cropping; a spatial reduction does not.** This is the
    /// whole sc-19753 argument for LTX in one assertion: the decoder's only normalization is a
    /// per-position channel reduction, so a tile normalizes exactly as the dense decode would.
    ///
    /// Measured on `[1, 6, 3, 7, 11]` cropped to `W = 0..5`: the real `pixel_norm` commutes to
    /// **0.0** (bit-exact), while the executed control — the identical op with a GroupNorm-shaped
    /// (C, H, W) reduction — breaks commutation by **1.485e0**. Without the control the 0.0 would be
    /// satisfied by any implementation; with it, the assertion has teeth.
    #[test]
    fn pixel_norm_commutes_with_cropping_but_a_spatial_reduction_does_not() {
        let x = ramped([1, 6, 3, 7, 11]);

        let cropped_dense = crop_w(&pixel_norm(&x, DEC_NORM_EPS).unwrap(), 5);
        let norm_of_crop = pixel_norm(&crop_w(&x, 5), DEC_NORM_EPS).unwrap();
        cropped_dense.eval().unwrap();
        norm_of_crop.eval().unwrap();
        assert_eq!(cropped_dense.shape(), norm_of_crop.shape());
        assert_eq!(
            max_abs_delta(&cropped_dense, &norm_of_crop),
            0.0,
            "pixel_norm reduces the channel axis only, so it must commute with spatial cropping"
        );

        let cropped_mutant = crop_w(&spatially_normalized(&x, DEC_NORM_EPS).unwrap(), 5);
        let mutant_of_crop = spatially_normalized(&crop_w(&x, 5), DEC_NORM_EPS).unwrap();
        cropped_mutant.eval().unwrap();
        mutant_of_crop.eval().unwrap();
        let broken = max_abs_delta(&cropped_mutant, &mutant_of_crop);
        println!("pixel_norm commutation = 0.0; spatial-reduction control = {broken:.3e}");
        assert!(
            broken > 1e-1,
            "the spatial-reduction control must break commutation, got max|Δ|={broken:.3e} — the \
             assertion above would then be proving nothing"
        );
    }

    /// **The real `DecResBlock` is spatially local**: decoding a crop reproduces the dense result
    /// exactly outside the crop's conv halo (two `k=3` convs ⇒ radius 2 columns), so a tile's
    /// interior is bit-identical to the dense decode's. Driven on the shipped block, not a stand-in.
    ///
    /// The **executed mutation control** rebuilds the same block with `pixel_norm` swapped for the
    /// GroupNorm-shaped reduction and nothing else changed: its halo interior then moves by
    /// **2.791e0** against the real block's **0.0**. That is what distinguishes "the port's
    /// normalization is per-position" from "the port merely happens to agree here".
    #[test]
    fn dec_res_block_is_spatially_local_but_a_spatially_normalized_variant_is_not() {
        use std::collections::HashMap;

        const CH: i32 = 4;
        const CROP: i32 = 7;
        /// Two `k=3` convs ⇒ each output column depends on ±2 input columns.
        const HALO: i32 = 2;

        let mut tensors = HashMap::new();
        for (index, name) in ["conv1", "conv2"].iter().enumerate() {
            let count = CH * 3 * 3 * 3 * CH;
            let w = (0..count)
                .map(|i| ((i + index as i32 * 37) as f32 * 0.053).sin() * 0.11)
                .collect::<Vec<_>>();
            tensors.insert(
                format!("b.{name}.conv.weight"),
                Array::from_slice(&w, &[CH, 3, 3, 3, CH]),
            );
            tensors.insert(
                format!("b.{name}.conv.bias"),
                Array::from_slice(
                    &(0..CH)
                        .map(|i| (i as f32 * 0.7 + index as f32).cos() * 0.02)
                        .collect::<Vec<_>>(),
                    &[CH],
                ),
            );
        }
        let block = DecResBlock::from_weights(&Weights::from_map(tensors), "b").unwrap();
        let x = ramped([1, CH, 3, 9, 13]);

        // The real block: the crop's halo interior is bit-identical to the dense result.
        let dense = block.forward(&x).unwrap();
        let of_crop = block.forward(&crop_w(&x, CROP)).unwrap();
        let dense_window = crop_w(&dense, CROP - HALO);
        let crop_window = crop_w(&of_crop, CROP - HALO);
        dense_window.eval().unwrap();
        crop_window.eval().unwrap();
        assert_eq!(crop_window.shape(), dense_window.shape());
        let local = max_abs_delta(&dense_window, &crop_window);

        // The control: the identical block with a spatial reduction is no longer crop-local.
        let mutated = |input: &Array| -> Result<Array> {
            let h = silu(&spatially_normalized(input, DEC_NORM_EPS)?)?;
            let h = block.conv1.forward(&h, false)?;
            let h = silu(&spatially_normalized(&h, DEC_NORM_EPS)?)?;
            let h = block.conv2.forward(&h, false)?;
            Ok(add(&h, input)?)
        };
        let mutant_dense = crop_w(&mutated(&x).unwrap(), CROP - HALO);
        let mutant_crop = crop_w(&mutated(&crop_w(&x, CROP)).unwrap(), CROP - HALO);
        mutant_dense.eval().unwrap();
        mutant_crop.eval().unwrap();
        let mutated_delta = max_abs_delta(&mutant_dense, &mutant_crop);

        println!(
            "DecResBlock crop-locality = {local:.3e}; spatial-norm control = {mutated_delta:.3e}"
        );
        assert_eq!(
            local, 0.0,
            "the shipped DecResBlock must be exactly crop-local outside its conv halo"
        );
        assert!(
            mutated_delta > 1e-1,
            "the spatial-reduction control must destroy crop-locality, got \
             max|Δ|={mutated_delta:.3e}"
        );
    }

    /// **The real tiling driver reassembles a per-position-normalizing decoder exactly.** Runs
    /// [`mlx_gen::vae_tiling::tiled_decode_with_hooks`] — the same slice → decode →
    /// trapezoidal-blend → fold loop `decode_tiled` uses — over a tile decoder that applies the shipped
    /// [`pixel_norm`] and then the ×32 spatial expansion, and compares against the dense equivalent.
    ///
    /// Measured: the `pixel_norm` decoder reassembles to max|Δ| = **2.384e-7** across a genuinely
    /// split plan (f32 blend round-off; the blend is a convex combination of identical per-position
    /// values), while the executed control (the same driver, same plan, same blend, with the
    /// GroupNorm-shaped reduction substituted) diverges by **2.021e-1** — six orders of magnitude
    /// larger. The plan is asserted to split, so neither number is measured against an untiled
    /// fall-through.
    ///
    /// **Executed mutation** (the real [`pixel_norm`] widened to a (C, H, W) reduction): the exact
    /// route degrades 2.384e-7 → **2.021e-1** and this test goes RED.
    #[test]
    fn tiled_assembly_is_exact_for_a_pixel_norm_decoder_and_wrong_for_a_spatial_one() {
        let (latent, plan) = synthetic_tiled_case();
        let tiles = plan.t.len() * plan.h.len() * plan.w.len();
        assert!(
            tiles > 1,
            "the plan must genuinely split, got {tiles} tile(s)"
        );

        let run = |normalize: &dyn Fn(&Array) -> Result<Array>| -> (Array, Array) {
            let dense = synthetic_spatial_decode(&normalize(&latent).unwrap()).unwrap();
            let tiled = mlx_gen::vae_tiling::tiled_decode_with_hooks(
                &latent,
                &plan,
                [2, 3, 4],
                None,
                |tile| synthetic_spatial_decode(&normalize(tile).unwrap()),
                |accumulators: &[&Array]| {
                    for accumulator in accumulators {
                        accumulator.eval()?;
                    }
                    Ok(())
                },
                || {},
            )
            .unwrap();
            dense.eval().unwrap();
            tiled.eval().unwrap();
            (dense, tiled)
        };

        let (dense, tiled) = run(&|x| pixel_norm(x, DEC_NORM_EPS));
        assert_eq!(tiled.shape(), dense.shape());
        let exact = max_abs_delta(&dense, &tiled);

        let (mutant_dense, mutant_tiled) = run(&|x| spatially_normalized(x, DEC_NORM_EPS));
        let broken = max_abs_delta(&mutant_dense, &mutant_tiled);

        println!("tiled pixel_norm assembly = {exact:.3e}; spatial-norm control = {broken:.3e}");
        assert!(
            exact < 1e-5,
            "a per-position normalization must reassemble exactly under tiling, got \
             max|Δ|={exact:.3e}"
        );
        assert!(
            broken > 1e-1,
            "the spatial-reduction control must break tiled reassembly, got max|Δ|={broken:.3e} — \
             the bound above would then be proving nothing"
        );
    }

    #[test]
    fn pixel_norm_matches_closed_form() {
        // 2 channels, single cell: mean(x²) = (1+4)/2 = 2.5 → denom = √(2.5 + eps).
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2, 1, 1, 1]);
        let got = pixel_norm(&x, 0.0).unwrap();
        let got = got.as_slice::<f32>();
        let denom = (2.5f32).sqrt();
        assert!((got[0] - 1.0 / denom).abs() < 1e-6);
        assert!((got[1] - 2.0 / denom).abs() < 1e-6);
    }

    #[test]
    fn patchify_unpatchify_round_trip() {
        // (1, 1, 1, 4, 4) ascending → patchify(p=2) → unpatchify(p=2) restores the original.
        let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let x = Array::from_slice(&data, &[1, 1, 1, 4, 4]);
        let p = patchify(&x, 2).unwrap();
        assert_eq!(p.shape(), &[1, 4, 1, 2, 2]);
        let u = unpatchify(&p, 2).unwrap();
        assert_eq!(u.shape(), &[1, 1, 1, 4, 4]);
        let u = contiguous(&u).unwrap();
        for (a, b) in u.as_slice::<f32>().iter().zip(data.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn vae_block_stride_from_kind() {
        assert_eq!(
            VaeBlock {
                kind: "compress_space_res".into(),
                num_layers: 0,
                multiplier: 2
            }
            .stride(),
            (1, 2, 2)
        );
        assert_eq!(
            VaeBlock {
                kind: "compress_time".into(),
                num_layers: 0,
                multiplier: 2
            }
            .stride(),
            (2, 1, 1)
        );
        assert_eq!(
            VaeBlock {
                kind: "compress_all".into(),
                num_layers: 0,
                multiplier: 1
            }
            .stride(),
            (2, 2, 2)
        );
    }
}
