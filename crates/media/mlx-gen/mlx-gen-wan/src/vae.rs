//! S2 — the Wan **2.1 `WanVAE`** (z16, stride 4×8×8): the 3-D causal-conv video VAE used by the
//! dense Wan2.1 / Wan2.2-14B path (the 5B uses the distinct z48 `vae22` — sc-2680). Port of the
//! `mlx_video` reference `models/wan/vae.py`, gated bit-for-bit against it (`tests/s2_parity.rs`).
//!
//! Tensors stay **NCTHW** (channels-first) throughout — mirroring the reference — and transpose to
//! channels-last only inside the conv ops (mlx convs are channels-last). Everything runs **f32**
//! (the reference upcasts the VAE to f32; f32 also sidesteps the bf16 NAX kernel history).
//!
//! Three reference quirks carried over verbatim:
//!  - The VAE "RMS_norm" is a **channel-L2 normalization** — `x / max(‖x‖₂ over C, 1e-12) · √C · γ`
//!    over axis 1 (not feature-RMS over the last axis). See `rms_norm_channels`.
//!  - `CausalConv3d` pads time on the **left only** by `kt − st` (causal), with an optional
//!    `cache_x` left-context for the chunked encode. Spatial padding is symmetric `(kh−1)/2`.
//!  - Encode is **chunked** (frame 0 alone, then 4-frame chunks) with a persistent per-conv
//!    `feat_cache` of the last `CACHE_T` frames — reproducing the full-sequence causal result while
//!    bounding memory. Decode is a single non-causal pass (T latent → 4·T frames).

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{
    add, concatenate_axis, divide, maximum, minimum, multiply, pad, split, subtract, sum_axes,
};
use mlx_rs::Array;

use mlx_gen::gen_core::{QWEN_WAN_Z16_MEAN as VAE_MEAN, QWEN_WAN_Z16_STD as VAE_STD};
use mlx_gen::nn::{conv2d, conv3d, silu, upsample_nearest};
use mlx_gen::tiling::{TilingConfig, VaeTiling};
use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, LatentDecoder, PinnedWeightsFile, Result};

use crate::vae_common::{
    contiguous, last_t_axis, scalar, slice_axis, tile_decode_accumulate, validate_decoder_tiling,
    DeadStageBuffers, FeatCache,
};

/// Last-`CACHE_T` frames are carried across chunks as causal left-context during encode.
const CACHE_T: i32 = 2;
/// Channel-L2 norm floor (reference `mx.clip(..., a_min=1e-12)`).
const NORM_EPS: f32 = 1e-12;

/// Wan2.1 VAE fixed structure (z16, dim_mult [1,2,4,4], 2 res-blocks/stage).
const DIM_MULT: [i32; 4] = [1, 2, 4, 4];
const NUM_RES_BLOCKS: usize = 2;
/// Decoder temporal-upsample per stage (`upsample3d` vs `upsample2d`); encoder is its mirror.
const TEMPORAL_UPSAMPLE: [bool; 3] = [true, true, false];
const TEMPORAL_DOWNSAMPLE: [bool; 3] = [false, true, true];

/// `x / max(‖x‖₂ over C, 1e-12) · √C · γ` — channel-L2 norm over axis 1. `x` is any rank with the
/// channel axis at index 1 (NCTHW or NCHW); `gamma` carries `C` elements in any shape.
fn rms_norm_channels(x: &Array, gamma: &Array) -> Result<Array> {
    let shape = x.shape();
    let nd = shape.len();
    let c = shape[1];
    let sum_sq = sum_axes(&multiply(x, x)?, &[1], true)?;
    let denom = maximum(&sum_sq, scalar(NORM_EPS))?.sqrt()?;
    let normed = divide(x, &denom)?;
    let scaled = multiply(&normed, scalar((c as f32).sqrt()))?;
    let mut wshape = vec![1i32; nd];
    wshape[1] = c;
    Ok(multiply(&scaled, &gamma.reshape(&wshape)?)?)
}

/// Last `n` frames along the temporal axis (axis 2, NCTHW): the reference `x[:, :, -n:]`.
fn last_t(x: &Array, n: i32) -> Result<Array> {
    last_t_axis(x, n, 2)
}

/// Temporal slice `x[:, :, start:end]` (axis 2).
fn slice_t(x: &Array, start: i32, end: i32) -> Result<Array> {
    slice_axis(x, 2, start, end)
}

/// 3-D conv with causal temporal left-pad (`kt − st`) + symmetric spatial pad `(kh−1)/2`. NCTHW
/// I/O; weight is the reference's already-MLX `[out, kt, kh, kw, in]`.
struct CausalConv3d {
    w: Array,
    b: Array,
    kt: i32,
    st: i32,
    ph: i32,
    pw: i32,
}

impl CausalConv3d {
    /// `st` is 1 everywhere except the encoder's temporal `downsample3d` `time_conv` (stride 2).
    fn from_weights(w: &Weights, prefix: &str, st: i32) -> Result<Self> {
        let weight = w.require(&format!("{prefix}.weight"))?.clone();
        let sh = weight.shape(); // [O, kt, kh, kw, I]
        let (kt, kh, kw) = (sh[1], sh[2], sh[3]);
        Ok(Self {
            w: weight,
            b: w.require(&format!("{prefix}.bias"))?.clone(),
            kt,
            st,
            ph: (kh - 1) / 2,
            pw: (kw - 1) / 2,
        })
    }

    fn forward(&self, x_ncthw: &Array, cache_x: Option<&Array>) -> Result<Array> {
        let mut x = x_ncthw.clone();
        let mut causal = self.kt - self.st;
        if let Some(cx) = cache_x {
            if causal > 0 {
                x = concatenate_axis(&[cx, &x], 2)?;
                causal = (causal - cx.shape()[2]).max(0);
            }
        }
        if causal > 0 || self.ph > 0 || self.pw > 0 {
            x = pad(
                &x,
                &[
                    (0, 0),
                    (0, 0),
                    (causal, 0),
                    (self.ph, self.ph),
                    (self.pw, self.pw),
                ][..],
                None,
                None,
            )?;
        }
        let x = x.transpose_axes(&[0, 2, 3, 4, 1])?; // NDHWC
        let y = conv3d(&x, &self.w, Some(&self.b), (self.st, 1, 1), (0, 0, 0))?;
        Ok(y.transpose_axes(&[0, 4, 1, 2, 3])?) // NCTHW
    }
}

/// Run a cached conv: feed the *previous* slot as left-context, then store this chunk's last frames.
/// Mirrors the reference's `cache_x = x[:, :, -CACHE_T:]` (+ 1-frame prepend when short) dance.
fn cached_conv(conv: &CausalConv3d, x: &Array, cache: &mut FeatCache) -> Result<Array> {
    let idx = cache.idx;
    let t = x.shape()[2];
    let mut cache_x = last_t(x, t.min(CACHE_T))?;
    if cache_x.shape()[2] < CACHE_T {
        if let Some(prev) = &cache.slots[idx] {
            cache_x = concatenate_axis(&[&last_t(prev, 1)?, &cache_x], 2)?;
        }
    }
    let y = conv.forward(x, cache.slots[idx].as_ref())?;
    cache.slots[idx] = Some(cache_x);
    cache.idx += 1;
    Ok(y)
}

/// `norm → SiLU → conv(3³) → norm → SiLU → conv(3³)` + residual (1³ skip when channels differ).
/// Reference list indices: `residual.{0,2,3,6}` (the SiLU/Dropout gaps carry no params).
struct ResidualBlock {
    norm1: Array,
    conv1: CausalConv3d,
    norm2: Array,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
}

impl ResidualBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let shortcut = if w.get(&format!("{prefix}.shortcut.weight")).is_some() {
            Some(CausalConv3d::from_weights(
                w,
                &format!("{prefix}.shortcut"),
                1,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: w.require(&format!("{prefix}.residual.0.gamma"))?.clone(),
            conv1: CausalConv3d::from_weights(w, &format!("{prefix}.residual.2"), 1)?,
            norm2: w.require(&format!("{prefix}.residual.3.gamma"))?.clone(),
            conv2: CausalConv3d::from_weights(w, &format!("{prefix}.residual.6"), 1)?,
            shortcut,
        })
    }

    fn shortcut(&self, x: &Array) -> Result<Array> {
        match &self.shortcut {
            Some(s) => s.forward(x, None),
            None => Ok(x.clone()),
        }
    }

    /// Decode path (no cache).
    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.shortcut(x)?;
        let y = self
            .conv1
            .forward(&silu(&rms_norm_channels(x, &self.norm1)?)?, None)?;
        let y = self
            .conv2
            .forward(&silu(&rms_norm_channels(&y, &self.norm2)?)?, None)?;
        Ok(add(&y, &h)?)
    }

    /// Encode path (chunked, with `feat_cache`).
    fn forward_cached(&self, x: &Array, cache: &mut FeatCache) -> Result<Array> {
        let h = self.shortcut(x)?;
        let y = silu(&rms_norm_channels(x, &self.norm1)?)?;
        let y = cached_conv(&self.conv1, &y, cache)?;
        let y = silu(&rms_norm_channels(&y, &self.norm2)?)?;
        let y = cached_conv(&self.conv2, &y, cache)?;
        Ok(add(&y, &h)?)
    }
}

/// Per-frame single-head spatial self-attention (head_dim = C). NCTHW I/O.
struct AttentionBlock {
    norm: Array,
    qkv_w: Array,
    qkv_b: Array,
    proj_w: Array,
    proj_b: Array,
}

impl AttentionBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            norm: w.require(&format!("{prefix}.norm.gamma"))?.clone(),
            qkv_w: w.require(&format!("{prefix}.to_qkv.weight"))?.clone(),
            qkv_b: w.require(&format!("{prefix}.to_qkv.bias"))?.clone(),
            proj_w: w.require(&format!("{prefix}.proj.weight"))?.clone(),
            proj_b: w.require(&format!("{prefix}.proj.bias"))?.clone(),
        })
    }

    fn forward(&self, x_ncthw: &Array) -> Result<Array> {
        let sh = x_ncthw.shape();
        let (b, c, t, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let bt = b * t;
        // NCTHW -> (B·T, C, H, W), channel-L2 norm over C, then NHWC for the 1×1 convs.
        let x = x_ncthw
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[bt, c, h, w])?;
        let normed = rms_norm_channels(&x, &self.norm)?.transpose_axes(&[0, 2, 3, 1])?;
        let qkv = conv2d(&normed, &self.qkv_w, Some(&self.qkv_b), 1, 0)?; // (BT,H,W,3C)
        let qkv = qkv.reshape(&[bt, h * w, 3 * c])?;
        let parts = split(&qkv, 3, 2)?; // q,k,v each (BT, H·W, C)
        let q = parts[0].expand_dims(1)?; // (BT, 1, H·W, C)
        let k = parts[1].expand_dims(1)?;
        let v = parts[2].expand_dims(1)?;
        let scale = (c as f32).powf(-0.5);
        let o = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let o = o.reshape(&[bt, h, w, c])?;
        let o = conv2d(&o, &self.proj_w, Some(&self.proj_b), 1, 0)?; // (BT,H,W,C)
        let o = o
            .transpose_axes(&[0, 3, 1, 2])?
            .reshape(&[b, t, c, h, w])?
            .transpose_axes(&[0, 2, 1, 3, 4])?; // NCTHW
        Ok(add(&o, x_ncthw)?)
    }
}

/// Decoder spatial 2× upsample (`resample.1` = Conv2d C→C/2). `upsample3d` first doubles T via a
/// learned `time_conv` (C→2C, interleaved); `upsample2d` is spatial-only.
struct UpsampleBlock {
    conv_w: Array,
    conv_b: Array,
    time_conv: Option<CausalConv3d>,
}

impl UpsampleBlock {
    fn from_weights(w: &Weights, prefix: &str, temporal: bool) -> Result<Self> {
        let time_conv = if temporal {
            Some(CausalConv3d::from_weights(
                w,
                &format!("{prefix}.time_conv"),
                1,
            )?)
        } else {
            None
        };
        Ok(Self {
            conv_w: w.require(&format!("{prefix}.resample.1.weight"))?.clone(),
            conv_b: w.require(&format!("{prefix}.resample.1.bias"))?.clone(),
            time_conv,
        })
    }

    fn forward(&self, x_ncthw: &Array) -> Result<Array> {
        let sh = x_ncthw.shape();
        let (b, c) = (sh[0], sh[1]);
        let (mut x, mut t, h, w) = (x_ncthw.clone(), sh[2], sh[3], sh[4]);
        if let Some(tc) = &self.time_conv {
            // C→2C, then interleave the two halves into 2·T frames.
            let xt = tc.forward(&x, None)?.reshape(&[b, 2, c, t, h, w])?;
            x = xt
                .transpose_axes(&[0, 2, 3, 1, 4, 5])?
                .reshape(&[b, c, t * 2, h, w])?;
            t *= 2;
        }
        // Per-frame nearest-2× spatial upsample + 3×3 conv (C→C/2).
        let xs = x
            .transpose_axes(&[0, 2, 3, 4, 1])?
            .reshape(&[b * t, h, w, c])?;
        let up = upsample_nearest(&xs, 2)?;
        let y = conv2d(&up, &self.conv_w, Some(&self.conv_b), 1, 1)?;
        let c_out = y.shape()[3];
        Ok(y.reshape(&[b, t, h * 2, w * 2, c_out])?
            .transpose_axes(&[0, 4, 1, 2, 3])?)
    }
}

/// Encoder spatial 2× downsample (ZeroPad-(0,1,0,1) + stride-2 3×3 conv C→C). `downsample3d` adds a
/// temporal stride-2 `time_conv` with chunk-cache (first chunk passes through, later chunks fold the
/// previous chunk's last frame as left-context).
struct DownsampleBlock {
    conv_w: Array,
    conv_b: Array,
    time_conv: Option<CausalConv3d>,
}

impl DownsampleBlock {
    fn from_weights(w: &Weights, prefix: &str, temporal: bool) -> Result<Self> {
        let time_conv = if temporal {
            Some(CausalConv3d::from_weights(
                w,
                &format!("{prefix}.time_conv"),
                2,
            )?)
        } else {
            None
        };
        Ok(Self {
            conv_w: w.require(&format!("{prefix}.resample.1.weight"))?.clone(),
            conv_b: w.require(&format!("{prefix}.resample.1.bias"))?.clone(),
            time_conv,
        })
    }

    fn forward(&self, x_ncthw: &Array, cache: &mut FeatCache) -> Result<Array> {
        let sh = x_ncthw.shape();
        let (b, c, t, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let bt = b * t;
        // Per-frame ZeroPad(0,1,0,1) + valid stride-2 conv.
        let xs = x_ncthw
            .transpose_axes(&[0, 2, 3, 4, 1])?
            .reshape(&[bt, h, w, c])?;
        let xp = pad(&xs, &[(0, 0), (0, 1), (0, 1), (0, 0)][..], None, None)?;
        let y = conv2d(&xp, &self.conv_w, Some(&self.conv_b), 2, 0)?;
        let (h2, w2, c2) = (y.shape()[1], y.shape()[2], y.shape()[3]);
        let mut x = y
            .reshape(&[b, t, h2, w2, c2])?
            .transpose_axes(&[0, 4, 1, 2, 3])?; // NCTHW

        if let Some(tc) = &self.time_conv {
            let idx = cache.idx;
            if cache.slots[idx].is_none() {
                // First chunk: stash x, skip the temporal conv (no downsample this chunk).
                cache.slots[idx] = Some(x.clone());
            } else {
                let new_cache = last_t(&x, 1)?;
                let prev_last = last_t(cache.slots[idx].as_ref().unwrap(), 1)?;
                x = tc.forward(&x, Some(&prev_last))?;
                cache.slots[idx] = Some(new_cache);
            }
            cache.idx += 1;
        }
        Ok(x)
    }
}

/// One decoder up-stage entry: a residual block or a spatial/temporal upsample.
enum UpLayer {
    Res(ResidualBlock),
    Up(UpsampleBlock),
}

/// One encoder down-stage entry: a residual block or a spatial/temporal downsample.
enum DownLayer {
    Res(ResidualBlock),
    Down(DownsampleBlock),
}

/// `conv1 → [Res, Attn, Res] → upsamples → RMS+SiLU+conv` (z_dim → 3). Non-causal (single pass).
struct Decoder3d {
    conv1: CausalConv3d,
    middle: (ResidualBlock, AttentionBlock, ResidualBlock),
    upsamples: Vec<UpLayer>,
    head_norm: Array,
    head_conv: CausalConv3d,
}

impl Decoder3d {
    fn from_weights(w: &Weights) -> Result<Self> {
        // Structure (block counts, resample positions, temporal flags) is fixed by the 2.1 config;
        // channel sizes ride on the weights, so the flat `upsamples` indices are all that matter.
        let p = "decoder";
        let mut upsamples = Vec::new();
        let mut next = 0usize;
        for i in 0..DIM_MULT.len() {
            for _ in 0..(NUM_RES_BLOCKS + 1) {
                upsamples.push(UpLayer::Res(ResidualBlock::from_weights(
                    w,
                    &format!("{p}.upsamples.{next}"),
                )?));
                next += 1;
            }
            if let Some(&temporal) = TEMPORAL_UPSAMPLE.get(i) {
                upsamples.push(UpLayer::Up(UpsampleBlock::from_weights(
                    w,
                    &format!("{p}.upsamples.{next}"),
                    temporal,
                )?));
                next += 1;
            }
        }

        Ok(Self {
            conv1: CausalConv3d::from_weights(w, &format!("{p}.conv1"), 1)?,
            middle: (
                ResidualBlock::from_weights(w, &format!("{p}.middle.0"))?,
                AttentionBlock::from_weights(w, &format!("{p}.middle.1"))?,
                ResidualBlock::from_weights(w, &format!("{p}.middle.2"))?,
            ),
            upsamples,
            head_norm: w.require(&format!("{p}.head.0.gamma"))?.clone(),
            head_conv: CausalConv3d::from_weights(w, &format!("{p}.head.2"), 1)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_upsample_tail(&self.forward_middle(x)?, DeadStageBuffers::Release)
    }

    /// The **globally-scoped** half: `conv1` → the three middle blocks, all at latent resolution
    /// (sc-19753).
    ///
    /// `middle.1` is an [`AttentionBlock`]: a single-head softmax self-attention over every `H·W`
    /// spatial token of a frame ([`AttentionBlock::forward`]). Its result therefore depends on the
    /// whole spatial grid, so a spatial tile that runs it attends only to its own crop's token set —
    /// a *wrong decode*, not a blend artifact. The channel-L2 [`rms_norm_channels`] used throughout
    /// this VAE really is per-position and tiling-invariant; the attention is the one op that is
    /// not, and it is why this half must run whole.
    ///
    /// Cheap to run dense: it is entirely at latent resolution, orders of magnitude under the
    /// `[B, 3, 4·T, 8·H, 8·W]` output the tiling exists to bound.
    fn forward_middle(&self, x: &Array) -> Result<Array> {
        let mut x = self.conv1.forward(x, None)?;
        x = self.middle.0.forward(&x)?;
        x = self.middle.1.forward(&x)?;
        self.middle.2.forward(&x)
    }

    /// The **spatially-local** half: the upsample stack (×8 spatial, ×4 temporal) and the
    /// `RMS → SiLU → conv` head. Every op here is a convolution, a nearest upsample, or the
    /// per-position channel-L2 norm, so evaluating it on a crop is exact up to the convolution
    /// padding at the crop boundary — which is what the trapezoidal overlap blend absorbs.
    ///
    /// With [`DeadStageBuffers::Release`] every residual block and resample is materialized and its
    /// freed buffers released; with [`DeadStageBuffers::Keep`] the tail stays one lazy graph.
    fn forward_upsample_tail(&self, middle: &Array, dead: DeadStageBuffers) -> Result<Array> {
        let mut x = middle.clone();
        for layer in &self.upsamples {
            x = match layer {
                UpLayer::Res(r) => r.forward(&x)?,
                UpLayer::Up(u) => u.forward(&x)?,
            };
            if dead.releases() {
                dead.materialize(&x)?;
            }
        }
        let x = silu(&rms_norm_channels(&x, &self.head_norm)?)?;
        self.head_conv.forward(&x, None)
    }
}

/// `conv1 → downsamples → [Res, Attn, Res] → RMS+SiLU+conv` (3 → z_dim·2). Chunked + cached.
struct Encoder3d {
    conv1: CausalConv3d,
    downsamples: Vec<DownLayer>,
    middle: (ResidualBlock, AttentionBlock, ResidualBlock),
    head_norm: Array,
    head_conv: CausalConv3d,
    cache_slots: usize,
}

impl Encoder3d {
    fn from_weights(w: &Weights) -> Result<Self> {
        let p = "encoder";
        let mut downsamples = Vec::new();
        let mut next = 0usize;
        let mut cache_slots = 1usize; // conv1
        for i in 0..DIM_MULT.len() {
            for _ in 0..NUM_RES_BLOCKS {
                downsamples.push(DownLayer::Res(ResidualBlock::from_weights(
                    w,
                    &format!("{p}.downsamples.{next}"),
                )?));
                next += 1;
                cache_slots += 2; // two cached convs per residual block
            }
            if let Some(&temporal) = TEMPORAL_DOWNSAMPLE.get(i) {
                downsamples.push(DownLayer::Down(DownsampleBlock::from_weights(
                    w,
                    &format!("{p}.downsamples.{next}"),
                    temporal,
                )?));
                next += 1;
                if temporal {
                    cache_slots += 1; // downsample3d time_conv
                }
            }
        }
        cache_slots += 4; // middle: 2 residual blocks × 2 convs
        cache_slots += 1; // head conv

        Ok(Self {
            conv1: CausalConv3d::from_weights(w, &format!("{p}.conv1"), 1)?,
            downsamples,
            middle: (
                ResidualBlock::from_weights(w, &format!("{p}.middle.0"))?,
                AttentionBlock::from_weights(w, &format!("{p}.middle.1"))?,
                ResidualBlock::from_weights(w, &format!("{p}.middle.2"))?,
            ),
            head_norm: w.require(&format!("{p}.head.0.gamma"))?.clone(),
            head_conv: CausalConv3d::from_weights(w, &format!("{p}.head.2"), 1)?,
            cache_slots,
        })
    }

    fn forward(&self, x: &Array, cache: &mut FeatCache) -> Result<Array> {
        let mut x = cached_conv(&self.conv1, x, cache)?;
        for layer in &self.downsamples {
            x = match layer {
                DownLayer::Res(r) => r.forward_cached(&x, cache)?,
                DownLayer::Down(d) => d.forward(&x, cache)?,
            };
        }
        x = self.middle.0.forward_cached(&x, cache)?;
        x = self.middle.1.forward(&x)?;
        x = self.middle.2.forward_cached(&x, cache)?;
        let x = silu(&rms_norm_channels(&x, &self.head_norm)?)?;
        cached_conv(&self.head_conv, &x, cache)
    }
}

/// The Wan 2.1 VAE: a decoder (always) + optional encoder (I2V), with per-channel latent
/// normalization. Decode latent → video; encode video → normalized latent.
pub struct WanVae {
    conv2: CausalConv3d,
    decoder: Decoder3d,
    encoder: Option<(CausalConv3d, Encoder3d)>, // (post-encoder conv1, encoder)
    mean: Array,                                // [1, z, 1, 1, 1]
    inv_std: Array,                             // [1, z, 1, 1, 1]
}

/// Trait adapter for the ordinary Wan z16 video layout. It keeps the VAE's native NCTHW input/output
/// while publishing the video denoiser's temporal latent identity to generic decode callers.
pub struct WanVideoDecoder<'a> {
    vae: &'a WanVae,
}

impl<'a> WanVideoDecoder<'a> {
    pub fn new(vae: &'a WanVae) -> Self {
        Self { vae }
    }
}

impl LatentDecoder for WanVideoDecoder<'_> {
    fn input_latent_space(&self) -> Option<&mlx_gen::gen_core::LatentSpace> {
        Some(&mlx_gen::gen_core::WAN_Z16_VIDEO_LATENT_SPACE)
    }

    fn decode(&self, latents: &Array) -> Result<Array> {
        if latents.shape().len() != 5 {
            return Err(Error::Msg(format!(
                "Wan z16 video decoder expects [B,C,T,H,W], got {:?}",
                latents.shape()
            )));
        }
        self.vae.decode(latents)
    }

    fn decode_tiled(
        &self,
        latents: &Array,
        tiling: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        if latents.shape().len() != 5 {
            return Err(Error::Msg(format!(
                "Wan z16 video decoder expects [B,C,T,H,W], got {:?}",
                latents.shape()
            )));
        }
        validate_decoder_tiling(tiling, VaeTiling::WAN, latents.shape()[2])?;
        self.vae.decode_tiled(latents, tiling, cancel)
    }
}

/// Single-image z16 adapter used by image pipelines that intentionally substitute the Wan VAE. The
/// shared z16 normalization is accepted as rank-4 NCHW, lifted to a one-latent-frame Wan decode, and
/// reduced back to a singleton-frame NCTHW image by selecting the leading output frame. Wan z16 is
/// non-causal and expands one latent frame to four decoded frames; image lanes consume the first.
pub struct WanSingleFrameDecoder<'a> {
    vae: &'a WanVae,
}

/// Owned, mutation-pinned form of [`WanSingleFrameDecoder`] for cross-model decoder substitution.
///
/// SceneWorks stages only the standalone `vae.safetensors`; retaining the pin makes replacement of
/// either a Hugging Face snapshot symlink or its blob a hard error before every decode.
pub struct OwnedWanSingleFrameDecoder {
    vae: WanVae,
    source: PinnedWeightsFile,
}

impl OwnedWanSingleFrameDecoder {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_pinned(PinnedWeightsFile::pin(path)?)
    }

    /// Load through the exact token prepared by the caller's [`mlx_gen::LoadSpec`].
    ///
    /// The pre/post guard spans safetensors metadata and tensor materialization. Retaining the same
    /// token then protects every later decode from pathname, symlink-target, or file replacement.
    pub fn from_pinned(source: PinnedWeightsFile) -> Result<Self> {
        let vae = source.read_unchanged(|path| {
            let weights = Weights::from_file(path)?;
            WanVae::from_weights(&weights)
        })?;
        Ok(Self { vae, source })
    }

    pub fn source(&self) -> &PinnedWeightsFile {
        &self.source
    }
}

impl<'a> WanSingleFrameDecoder<'a> {
    pub fn new(vae: &'a WanVae) -> Self {
        Self { vae }
    }

    fn input_5d(latents: &Array) -> Result<Array> {
        let shape = latents.shape();
        if shape.len() != 4 {
            return Err(Error::Msg(format!(
                "Wan z16 single-frame decoder expects [B,C,H,W], got {shape:?}"
            )));
        }
        Ok(latents.reshape(&[shape[0], shape[1], 1, shape[2], shape[3]])?)
    }

    fn first_frame(decoded: &Array) -> Result<Array> {
        let shape = decoded.shape();
        if shape.len() != 5 || shape[2] < 1 {
            return Err(Error::Msg(format!(
                "Wan z16 single-frame decoder produced invalid [B,3,T,H,W] output {shape:?}"
            )));
        }
        slice_axis(decoded, 2, 0, 1)
    }
}

impl LatentDecoder for WanSingleFrameDecoder<'_> {
    fn input_latent_space(&self) -> Option<&mlx_gen::gen_core::LatentSpace> {
        Some(&mlx_gen::gen_core::WAN_Z16_LATENT_SPACE)
    }

    fn decode(&self, latents: &Array) -> Result<Array> {
        Self::first_frame(&self.vae.decode(&Self::input_5d(latents)?)?)
    }

    fn decode_tiled(
        &self,
        latents: &Array,
        tiling: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        let latents = Self::input_5d(latents)?;
        validate_decoder_tiling(tiling, VaeTiling::WAN, 1)?;
        Self::first_frame(&self.vae.decode_tiled(&latents, tiling, cancel)?)
    }
}

impl LatentDecoder for OwnedWanSingleFrameDecoder {
    fn input_latent_space(&self) -> Option<&mlx_gen::gen_core::LatentSpace> {
        Some(&mlx_gen::gen_core::WAN_Z16_LATENT_SPACE)
    }

    fn decode(&self, latents: &Array) -> Result<Array> {
        self.source.ensure_unchanged()?;
        WanSingleFrameDecoder::new(&self.vae).decode(latents)
    }

    fn decode_tiled(
        &self,
        latents: &Array,
        tiling: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        self.source.ensure_unchanged()?;
        WanSingleFrameDecoder::new(&self.vae).decode_tiled(latents, tiling, cancel)
    }
}

impl WanVae {
    /// Geometry owned by the concrete non-causal z16 decoder.
    pub const VAE_TILING: VaeTiling = VaeTiling::WAN;

    /// Build from a weight map. Structure is fixed by the 2.1 config and channel sizes ride on the
    /// weights, so the same builder serves any `dim` (96 in production; tiny in the parity fixture).
    /// The encoder is loaded only if its weights are present.
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let z = VAE_MEAN.len() as i32;
        let mean = Array::from_slice(&VAE_MEAN, &[1, z, 1, 1, 1]);
        let std = Array::from_slice(&VAE_STD, &[1, z, 1, 1, 1]);
        let inv_std = divide(scalar(1.0), &std)?;

        let encoder = if w.get("encoder.conv1.weight").is_some() {
            Some((
                CausalConv3d::from_weights(w, "conv1", 1)?,
                Encoder3d::from_weights(w)?,
            ))
        } else {
            None
        };

        let vae = Self {
            conv2: CausalConv3d::from_weights(w, "conv2", 1)?,
            decoder: Decoder3d::from_weights(w)?,
            encoder,
            mean,
            inv_std,
        };
        // Materialize at load: left lazy, the first encode/decode's command buffers wait on the
        // safetensors reads — past the GPU watchdog on a cold page cache (sc-24245; see
        // `mlx_gen_qwen_image::loader::load_transformer_with`).
        w.materialize_accessed()?;
        Ok(vae)
    }

    /// Decode a normalized latent `[B, z, T, H, W]` → video `[B, 3, 4·T, 8·H, 8·W]` in `[-1, 1]`.
    pub fn decode(&self, z: &Array) -> Result<Array> {
        // The intermediates are scoped to this block so they are dropped before the release below
        // rather than freed into the cache after it (sc-20686).
        let out = {
            let denorm = add(&divide(z, &self.inv_std)?, &self.mean)?;
            let x = self.conv2.forward(&denorm, None)?;
            let out = self.decoder.forward(&x)?;
            contiguous(&minimum(&maximum(&out, scalar(-1.0))?, scalar(1.0))?)?
        };
        // Single pass: the head's freed buffers leave the cache with the output (sc-20686).
        DeadStageBuffers::Release.materialize(&out)?;
        Ok(out)
    }

    /// Decode with **tiling** for memory-bounded large/long-video decode (`cfg`): split the latent
    /// into overlapping spatial/temporal tiles, decode each (conv2 + decoder + clamp), and
    /// trapezoidally blend them into the full video. Falls back to the single-pass [`Self::decode`] when
    /// `cfg` doesn't fire for these dims. The Wan z16 VAE is **non-causal** in time (`T → 4·T`) and
    /// upsamples 8× spatially — [`VaeTiling::WAN`].
    ///
    /// **Normalization semantics (sc-19753).** Denormalize, `conv2` and the decoder's middle blocks
    /// — including its spatial self-attention — run **once** on the full latent
    /// (`Decoder3d::forward_middle`); only the spatially-local upsample tail is tiled. The
    /// reference `WanVAE.decode_tiled` (`models/wan/tiling.py`) hoists just the denormalize and runs
    /// the *whole* decoder per tile, so every spatial tile's `middle.1` softmax attended only to its
    /// own crop's tokens. This port deliberately diverges from the reference there: the earlier
    /// clearance of this family was based on its channel-L2 norms being per-position, which is true,
    /// but the middle attention is a spatial global reduction that the norms audit missed.
    ///
    /// The middle blocks are shape-preserving at latent resolution, so the tile plan is unchanged —
    /// the same [`TilePlan`](mlx_gen::tiling::TilePlan) now partitions the middle feature map instead
    /// of the latent. The
    /// full-size `output`/`weights` accumulators are filled tile-by-tile (pad-and-add) so peak
    /// memory stays bounded by one tile's tail.
    ///
    /// **Memory cost of the dense head.** The middle feature map is now materialized whole rather
    /// than per tile: `dim·4` channels at *latent* resolution, so it scales with `T_lat·H/8·W/8`,
    /// not with the output. It sits alongside the full-size output accumulator the tiling already
    /// required and is a fraction of it. This is the same tradeoff sc-19753 took on every image VAE
    /// — bounding convolution work rather than every activation is the price of keeping global
    /// statistics global. Shared tiling geometry: [`mlx_gen::tiling`].
    pub fn decode_tiled(
        &self,
        z: &Array,
        cfg: &TilingConfig,
        cancel: Option<&CancelFlag>,
    ) -> Result<Array> {
        if cancel.is_some_and(CancelFlag::is_cancelled) {
            return Err(Error::Canceled);
        }
        let sh = z.shape();
        let (f, h, w) = (sh[2], sh[3], sh[4]);
        if !cfg.needs_tiling(Self::VAE_TILING, f, h, w) {
            return self.decode(z);
        }
        // Denormalize + conv2 + the attention-bearing middle blocks, once on the full latent.
        let denorm = add(&divide(z, &self.inv_std)?, &self.mean)?;
        let middle = self
            .decoder
            .forward_middle(&self.conv2.forward(&denorm, None)?)?;
        let plan = cfg.plan(Self::VAE_TILING, f, h, w);

        // NCTHW: channel axis at 1, tiled axes [2, 3, 4]. Per-tile work = upsample tail + clamp.
        tile_decode_accumulate(&middle, &plan, [2, 3, 4], cancel, |tile| {
            let dec = self
                .decoder
                .forward_upsample_tail(tile, DeadStageBuffers::Keep)?;
            Ok(minimum(&maximum(&dec, scalar(-1.0))?, scalar(1.0))?)
        })
    }

    /// Run the chunked causal encoder + the post-encoder conv → the raw Gaussian moments
    /// `(mean, logvar)`, each `[B, z, T_lat, H/8, W/8]` (the `post_conv1` output split on the channel
    /// axis), **before** latent normalization — the reference `DiagonalGaussianDistribution.parameters`.
    fn encode_moments(&self, video: &Array) -> Result<(Array, Array)> {
        let (post_conv1, encoder) = self
            .encoder
            .as_ref()
            .ok_or_else(|| Error::Msg("WanVae: encode requires encoder weights".into()))?;

        let t = video.shape()[2];
        let num_chunks = 1 + (t - 1) / 4;
        let mut cache = FeatCache::new(encoder.cache_slots);
        let mut out: Option<Array> = None;
        for i in 0..num_chunks {
            cache.idx = 0;
            let chunk = if i == 0 {
                slice_t(video, 0, 1)
            } else {
                slice_t(video, 1 + 4 * (i - 1), 1 + 4 * i)
            }?;
            let chunk_out = encoder.forward(&chunk, &mut cache)?;
            out = Some(match out {
                None => chunk_out,
                Some(o) => concatenate_axis(&[&o, &chunk_out], 2)?,
            });
        }
        let out = out.ok_or_else(|| Error::Msg("wan vae: encode produced no chunks".into()))?;
        let parts = split(&post_conv1.forward(&out, None)?, 2, 1)?; // [mean, logvar]
        Ok((parts[0].clone(), parts[1].clone()))
    }

    /// Latent normalization by the z16 stats: `(x − mean)·inv_std`.
    fn normalize_latent(&self, x: &Array) -> Result<Array> {
        contiguous(&multiply(&subtract(x, &self.mean)?, &self.inv_std)?)
    }

    /// Encode a video `[B, 3, T, H, W]` (T = 1 + 4·k, values in `[-1, 1]`) → normalized latent
    /// `[B, z, T_lat, H/8, W/8]` via chunked causal encoding (`DiagonalGaussianDistribution.mode()` =
    /// the Gaussian mean). Requires encoder weights.
    pub fn encode(&self, video: &Array) -> Result<Array> {
        let (mean, _logvar) = self.encode_moments(video)?;
        self.normalize_latent(&mean)
    }

    /// Encode + **sample** the Gaussian (`DiagonalGaussianDistribution.sample()`): `mean +
    /// exp(0.5·clamp(logvar, −30, 20))·eps`, then latent-normalize. `eps` is standard-normal noise of
    /// the latent shape `[B, z, T_lat, H/8, W/8]`, taken as an argument so the result is deterministic
    /// (the reference draws it from the request seed). Bernini's `get_vae_features` uses this for
    /// **video** source conditioning; images use [`Self::encode`] (`.mode()`).
    pub fn encode_sample(&self, video: &Array, eps: &Array) -> Result<Array> {
        let (mean, logvar) = self.encode_moments(video)?;
        self.normalize_latent(&reparameterize(&mean, &logvar, eps)?)
    }
}

/// `DiagonalGaussianDistribution.sample()`: `mean + exp(0.5·clamp(logvar, −30, 20))·eps`. The
/// `[−30, 20]` log-variance clamp is the diffusers default.
fn reparameterize(mean: &Array, logvar: &Array, eps: &Array) -> Result<Array> {
    let logvar = minimum(&maximum(logvar, scalar(-30.0))?, scalar(20.0))?;
    let std = multiply(&logvar, scalar(0.5))?.exp()?;
    Ok(add(mean, &multiply(&std, eps)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_channels_matches_closed_form() {
        // 2 channels, single spatial cell: ‖x‖₂ = √5, √C = √2 → out = x/√5·√2·γ.
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2, 1, 1, 1]);
        let gamma = Array::from_slice(&[1.0f32, 1.0], &[2, 1, 1, 1]);
        let got = rms_norm_channels(&x, &gamma).unwrap();
        let got = got.as_slice::<f32>();
        let s = (2.0f32).sqrt() / (5.0f32).sqrt();
        assert!((got[0] - 1.0 * s).abs() < 1e-6);
        assert!((got[1] - 2.0 * s).abs() < 1e-6);
    }

    /// `reparameterize` = `mean + exp(0.5·clamp(logvar, −30, 20))·eps`. Checks the formula and the
    /// log-variance clamp at both ends (50 → 20, −50 → −30) against hand-computed values.
    #[test]
    fn reparameterize_matches_closed_form() {
        let mean = Array::from_slice(&[1.0f32, -2.0, 0.5, 3.0], &[4]);
        let logvar = Array::from_slice(&[0.0f32, 4.0, 50.0, -50.0], &[4]);
        let eps = Array::from_slice(&[2.0f32, 1.0, -1.0, 4.0], &[4]);
        let got = reparameterize(&mean, &logvar, &eps).unwrap();
        let got = got.as_slice::<f32>();
        // std = exp(0.5·clamp(logvar)): exp(0)=1, exp(2), exp(10) (clamped 20), exp(-15) (clamped -30).
        let want = [
            1.0 + 1.0 * 2.0,
            -2.0 + (2.0f32).exp() * 1.0,
            0.5 - (10.0f32).exp(),
            3.0 + (-15.0f32).exp() * 4.0,
        ];
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-3 * w.abs().max(1.0), "got {g} want {w}");
        }
    }

    /// A z16 decoder with the production topology at base width `dim` (96 in production), seeded
    /// random f32 weights (production z16 decodes f32), each materialized as it is made.
    fn synthetic_decoder(dim: i32) -> WanVae {
        let key = mlx_rs::random::key(20686).unwrap();
        let mut tensors = std::collections::HashMap::new();
        let mut put = |name: String, shape: &[i32]| {
            let value = mlx_rs::random::normal::<f32>(shape, None, Some(0.2), Some(&key)).unwrap();
            mlx_rs::transforms::eval([&value]).unwrap();
            tensors.insert(name, value);
        };
        fn conv(put: &mut dyn FnMut(String, &[i32]), p: &str, o: i32, i: i32, kt: i32, k: i32) {
            put(format!("{p}.weight"), &[o, kt, k, k, i]);
            put(format!("{p}.bias"), &[o]);
        }
        fn res(put: &mut dyn FnMut(String, &[i32]), p: &str, i: i32, o: i32) {
            put(format!("{p}.residual.0.gamma"), &[i]);
            conv(put, &format!("{p}.residual.2"), o, i, 3, 3);
            put(format!("{p}.residual.3.gamma"), &[o]);
            conv(put, &format!("{p}.residual.6"), o, o, 3, 3);
            if i != o {
                conv(put, &format!("{p}.shortcut"), o, i, 1, 1);
            }
        }
        let top = dim * DIM_MULT[3];
        conv(&mut put, "conv2", 16, 16, 1, 1);
        conv(&mut put, "decoder.conv1", top, 16, 3, 3);
        res(&mut put, "decoder.middle.0", top, top);
        res(&mut put, "decoder.middle.2", top, top);
        put("decoder.middle.1.norm.gamma".into(), &[top]);
        put(
            "decoder.middle.1.to_qkv.weight".into(),
            &[3 * top, 1, 1, top],
        );
        put("decoder.middle.1.to_qkv.bias".into(), &[3 * top]);
        put("decoder.middle.1.proj.weight".into(), &[top, 1, 1, top]);
        put("decoder.middle.1.proj.bias".into(), &[top]);
        let mut input = top;
        let mut index = 0;
        for stage in 0..DIM_MULT.len() {
            let output = dim * DIM_MULT[DIM_MULT.len() - 1 - stage];
            for block in 0..=NUM_RES_BLOCKS {
                let block_input = if block == 0 { input } else { output };
                res(
                    &mut put,
                    &format!("decoder.upsamples.{index}"),
                    block_input,
                    output,
                );
                index += 1;
            }
            input = output;
            if let Some(&temporal) = TEMPORAL_UPSAMPLE.get(stage) {
                let p = format!("decoder.upsamples.{index}");
                if temporal {
                    conv(
                        &mut put,
                        &format!("{p}.time_conv"),
                        2 * output,
                        output,
                        3,
                        1,
                    );
                }
                put(
                    format!("{p}.resample.1.weight"),
                    &[output / 2, 3, 3, output],
                );
                put(format!("{p}.resample.1.bias"), &[output / 2]);
                input = output / 2;
                index += 1;
            }
        }
        put("decoder.head.0.gamma".into(), &[dim]);
        conv(&mut put, "decoder.head.2", 3, dim, 3, 3);
        WanVae::from_weights(&Weights::from_map(tensors)).unwrap()
    }

    fn block_releases() -> usize {
        crate::vae_common::BLOCK_RELEASES.with(std::cell::Cell::get)
    }

    fn eval(x: &Array) {
        mlx_rs::transforms::eval([x]).unwrap();
    }

    /// The same decode with the tail run in `dead` mode (denormalize → conv2 → middle → tail →
    /// clamp), evaluated.
    fn decode_with(vae: &WanVae, latent: &Array, dead: DeadStageBuffers) -> Array {
        let denorm = add(divide(latent, &vae.inv_std).unwrap(), &vae.mean).unwrap();
        let middle = vae
            .decoder
            .forward_middle(&vae.conv2.forward(&denorm, None).unwrap())
            .unwrap();
        let out = vae.decoder.forward_upsample_tail(&middle, dead).unwrap();
        let out = contiguous(&minimum(maximum(&out, scalar(-1.0)).unwrap(), scalar(1.0)).unwrap())
            .unwrap();
        eval(&out);
        out
    }

    /// sc-20686: a single-pass decode materializes and releases after every residual block and
    /// resample plus once with the output, without changing a single output value; a tiled decode
    /// stays one lazy graph per tile and releases nothing. Sized: dim 8, latent [1,16,2,8,8] → a
    /// few MiB.
    #[test]
    fn single_pass_decode_releases_every_block_and_tiles_keep_them() {
        let vae = synthetic_decoder(8);
        let latent = mlx_rs::random::normal::<f32>(&[1, 16, 2, 8, 8], None, None, None).unwrap();

        let before = block_releases();
        let released = vae.decode(&latent).unwrap();
        assert_eq!(block_releases() - before, vae.decoder.upsamples.len() + 1);

        let kept = decode_with(&vae, &latent, DeadStageBuffers::Keep);
        assert_eq!(released.shape(), kept.shape());
        assert_eq!(released.as_slice::<f32>(), kept.as_slice::<f32>());

        let before = block_releases();
        let tiled = vae
            .decode_tiled(&latent, &TilingConfig::spatial_only(32, 16), None)
            .unwrap();
        eval(&tiled);
        assert_eq!(tiled.shape(), released.shape());
        assert_eq!(block_releases(), before, "tiled decode must keep its pool");
    }

    const CACHE_CHILD: &str = "WAN_Z16_DECODE_CACHE_CHILD";

    /// sc-20686: right after a single-pass decode MLX's allocator cache is empty, both of what the
    /// decode freed and of what was cached before it, while the same decode with the pool kept
    /// leaves its garbage cached. MLX's counters are process-wide and this binary runs tests on
    /// parallel threads, so the measurement runs alone in a child process of this test binary.
    /// Sized: dim 16 f32 weights (< 8 MiB), latent [1,16,2,8,8], a 16 MiB seeded scratch; total
    /// well under 128 MiB.
    #[test]
    fn single_pass_decode_leaves_mlx_cache_empty() {
        if std::env::var_os(CACHE_CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "vae::tests::single_pass_decode_leaves_mlx_cache_empty",
                    "--test-threads=1",
                    "--nocapture",
                ])
                .env(CACHE_CHILD, "1")
                .output()
                .expect("spawn the isolated decode-cache child");
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success() && stdout.contains("1 passed"),
                "isolated decode-cache child failed:\n{stdout}\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        use mlx_rs::memory::{clear_cache, get_cache_memory};
        const MIB: usize = 1 << 20;
        let vae = synthetic_decoder(16);
        let latent = mlx_rs::random::normal::<f32>(&[1, 16, 2, 8, 8], None, None, None).unwrap();
        eval(&latent);

        clear_cache();
        drop(decode_with(&vae, &latent, DeadStageBuffers::Keep));
        let kept = get_cache_memory();
        assert!(
            kept >= 4 * MIB,
            "the control decode must leave garbage: {kept} B"
        );

        {
            let scratch = mlx_rs::ops::zeros::<f32>(&[(4 * MIB) as i32]).unwrap();
            eval(&scratch);
        }
        assert!(get_cache_memory() >= kept + 16 * MIB);
        let out = vae.decode(&latent).unwrap();
        eval(&out);
        let released = get_cache_memory();
        assert!(
            released < MIB && released * 16 < kept,
            "a single-pass decode must leave the cache empty: {released} B (kept pool {kept} B)"
        );
    }

    /// Measurement harness (sc-20686), not a gate: the production-width z16 decoder (dim 96, f32,
    /// synthetic weights) single-pass decode's live and live + cache peaks per output voxel. Sized:
    /// ~0.3 GB of weights + the decode, ~0.6 GB at the default latent `2,8,8` (32,768 voxels);
    /// total < 1 GB. Override the latent with `WAN_DECODE_HARNESS_LATENT=t,h,w` (sizes scale with
    /// the voxel count). Run alone: `cargo test -p mlx-gen-wan --lib
    /// vae::tests::decode_footprint_harness -- --ignored --exact --test-threads=1 --nocapture`.
    #[test]
    #[ignore = "measurement harness; production-width weights; run alone on request"]
    fn decode_footprint_harness() {
        let [t, h, w] = std::env::var("WAN_DECODE_HARNESS_LATENT").map_or([2, 8, 8], |raw| {
            let dims: Vec<i32> = raw.split(',').map(|v| v.trim().parse().unwrap()).collect();
            [dims[0], dims[1], dims[2]]
        });
        let vae = synthetic_decoder(96);
        let latent = mlx_rs::random::normal::<f32>(&[1, 16, t, h, w], None, None, None).unwrap();
        eval(&latent);
        mlx_rs::memory::clear_cache();
        let base = mlx_rs::memory::get_active_memory() as u64;
        mlx_rs::memory::reset_peak_memory();
        let probe =
            mlx_gen::memory_probe::AllocatorProbe::start(std::time::Duration::from_millis(1));
        let out = vae.decode(&latent).unwrap();
        eval(&out);
        let report = probe.finish();
        let live = (mlx_rs::memory::get_peak_memory() as u64).saturating_sub(base);
        let footprint = report
            .sampled_footprint_peak_bytes
            .max(live + base)
            .saturating_sub(base);
        let shape = out.shape();
        let voxels = f64::from(shape[2] * shape[3] * shape[4]);
        eprintln!(
            "z16 f32 single-pass latent {t},{h},{w}: {voxels} voxels; live {:.0} B/voxel, \
             live+cache {:.0} B/voxel",
            live as f64 / voxels,
            footprint as f64 / voxels
        );
    }
}
