//! The MiniMax-H3 audio VAE **encode** path (sc-17149, `ref2va`).
//!
//! [`crate::audio_vae`] ports the decode half; this is its counterpart, and it is what turns a
//! reference soundtrack into the conditioning latents `ref2va` denoises against.
//!
//! ```text
//! encode(waveform)     [B, 1, S]      mono 32 kHz, one BATCH item per stereo channel
//!   right-pad                          zeros, up to a multiple of hop = ∏ encoder_rates = 800
//!   DAC encoder                        block.0     Conv1d 1 -> 64, k7
//!     x5 EncoderBlock                  3x ResidualUnit(dil 1/3/9) -> Snake -> strided Conv1d
//!                                      channels DOUBLE per stage: 64 -> 128 … -> 2048
//!     Snake1d + Conv1d k3              -> [B, 2048, S/800]
//!   pre_block (AttnProjection)         causal attention + GeGLU, 2048 -> 32
//!   mean_proj / logs_proj              1x1 convs, [32, 32, 1] each
//!                    ->  DiagonalGaussian over [B, 32, S/800]
//! ```
//!
//! # Three things this module owns that a plausible port gets wrong
//!
//! **1. `pre_block` is not optional, and it is where the risk is.** The encode half is 173 of the
//! checkpoint's 1087 tensors: 147 under `encoder.`, **22 under `pre_block.`**, and the four
//! posterior-head tensors. `mean_proj.weight` is `[32, 32, 1]` while the encoder trunk emits
//! **2048** channels, so the projection between them is load-bearing, not decorative. Inside it,
//! [`CausalAttention`] does two unusual things: the heads are **mean-pooled away** rather than
//! concatenated, and the head width that remains is **adaptively average-pooled** down to
//! `latent_channels`. On the shipped model that pool is an exact 256 -> 32, which a
//! `reshape(.., 32, 8).mean(-1)` also reproduces — so the committed fixture deliberately runs a
//! geometry whose windows overlap, and pins the general formula ([`adaptive_avg_pool_last_axis`])
//! against torch directly.
//!
//! **2. The encoder's activation is `Snake1d`, not the decoder's alias-free `SnakeBeta`.** No
//! Kaiser-sinc resampling anywhere in the encode path: `x + (α + 1e-9)⁻¹·sin²(αx)` applied
//! directly, one `α` per channel, **no log scale**. That is exactly
//! [`crate::alias_free::SnakeBeta`] with `beta = alpha` and `logscale = false`, which is how this
//! module reuses it rather than growing a second periodic activation. (The reference writes the
//! reciprocal as a multiply and this crate writes it as a divide; the two differ by at most one
//! ulp and the parity fixture bounds the consequence.)
//!
//! **3. The fused QKV is contiguous THIRDS, and the bias is assembled from three tensors.**
//! `qkv.weight` is `[3·in_dim, in_dim]` and the reference reads it as
//! `reshape(B, N, 3, heads, head_dim)` — thirds first, heads second. [`crate::layout`]'s Rule 2
//! per-head interleaving does **not** apply here (that transform belongs to the video VAE and the
//! DiT; the audio VAE is carried through the conversion unchanged — see
//! [`crate::layout::AUDIO_VAE_IS_UNCONVERTED`]), and applying it anyway is shape-identical and
//! silently wrong. The `nn.Linear` itself is bias-less; the effective bias is
//! `cat(q_bias, zero_k_bias, v_bias)`, where `zero_k_bias` is a **zero buffer** the checkpoint
//! actually ships. It is read and concatenated rather than assumed, so a checkpoint whose key
//! bias is not zero is honoured instead of silently ignored.
//!
//! # Reference
//!
//! `diffusers.AutoencoderKLMiniMaxH3Audio.encode`. The snapshot's own `FL2VA/audio_vae` bundle —
//! which [`crate::audio_vae`] is ported from — is an inference-only package and **has no `encode`
//! method at all** (`DacAudioVAE` defines `preprocess` and `decode` only), so diffusers is the
//! only executable reference for this half. Its `Encoder` / `EncoderBlock` / `ResidualUnit` /
//! `Snake1d` and its `AttnProjection` / `CausalAttention` / `GeGluMlp` were cross-read against
//! `dac_audio_vae.py` and `dac_attn_proj.py` op for op before this port was written; the one
//! difference is unreachable here (the original picks `attn_proj_dim` as the next power of two
//! when `latent_dim % vae_latent_channels != 0`, a branch diffusers rejects outright and the
//! published config never takes — see [`MiniMaxH3AudioVaeEncoder::from_weights_with_heads`]).
//!
//! # Precision
//!
//! diffusers pins this component with `_keep_in_fp32_modules = ["encoder", …, "mean_proj",
//! "logs_proj"]`: the weight-normed convolutions and Snake activations "degrade audibly under
//! bfloat16 (roughly 20 dB quieter decodes)". Callers should pass [`Dtype::Float32`] unless they
//! have measured otherwise.
//!
//! # Deliberate duplication
//!
//! `fuse_weight_norm` and the weight-normed `Conv1d` are re-implemented here rather than shared
//! with [`crate::audio_vae`], whose versions are private and stride-less. The encoder needs a
//! **strided** convolution (`stride ∈ encoder_rates`), so the two are not interchangeable; the
//! fusion arithmetic (`w = g · v / ‖v‖`, norm over every axis but the first) is identical and is
//! asserted against the same rule in both modules' unit tests.

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention, ScaledDotProductAttentionMask};
use mlx_rs::ops::{add, concatenate_axis, conv1d, divide, exp, multiply, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{gelu_tanh, linear};
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::{Error, Result};

use crate::alias_free::SnakeBeta;
use crate::audio_config::MiniMaxH3AudioVaeConfig;
use crate::tensor::slice_axis;

/// Heads in the `pre_block` causal attention.
///
/// The original `DacAudioVAE` **hardcodes** `AttnProjection(latent_dim, attn_proj_dim,
/// num_heads=8)` — it is not in `metadata.json`, `config.yaml` or the FL2VA `config.json`, so
/// [`MiniMaxH3AudioVaeConfig`] has no field for it. The diffusers repackaging promoted it to a
/// config key (`audio_vae/config.json`: `"num_attention_heads": 8`), and the two agree.
///
/// Pinned as a constant for the same reason [`crate::layout::PUBLISHED_GATED_FFN_LAYOUT`] is:
/// changing the head count changes `head_dim`, which changes the attention scale *and* the
/// adaptive pool's window layout, while every tensor shape stays identical.
pub const ATTN_PROJ_HEADS: i32 = 8;

/// `eps` for every `nn.LayerNorm` in `pre_block` — torch's default, declared in no config file.
pub const LAYER_NORM_EPS: f32 = 1e-5;

/// `mlp_ratio` of the `pre_block` GeGLU MLP — `AttnProjection`'s default, which the audio VAE
/// never overrides, so the GeGLU hidden width is `latent_channels · 2` (64 on the shipped model).
///
/// **Enforced, not merely declared:** `GeGluMlp::from_weights` checks `w0`/`w1`/`w2` against it
/// and rejects a checkpoint at a different ratio. A constant nothing validates is a comment with
/// a type, and this crate has been bitten by exactly that (`crate::layout`).
pub const MLP_RATIO: i32 = 2;

/// Re-fuse a `weight_norm` pair into a dense weight: `g · v / ‖v‖`, the norm reduced over every
/// axis but axis 0 (torch's `dim=0` default).
///
/// Duplicated from [`crate::audio_vae`]'s private helper — see the module docs.
fn fuse_weight_norm(g: &Array, v: &Array) -> Result<Array> {
    let shape = v.shape();
    if shape.len() != 3 {
        return Err(Error::Msg(format!(
            "minimax-h3 audio encoder: weight_v must be rank 3, got {shape:?}"
        )));
    }
    if g.shape() != [shape[0], 1, 1] {
        return Err(Error::Msg(format!(
            "minimax-h3 audio encoder: weight_g {:?} does not match weight_v {shape:?}",
            g.shape()
        )));
    }
    let norm = v.square()?.sum_axes(&[1, 2], true)?.sqrt()?;
    Ok(multiply(g, &divide(v, &norm)?)?)
}

/// A **strided** weight-normed `Conv1d` in MLX's NLC layout.
///
/// Public so `tests/audio_vae_encode_parity.rs` can hold ONE convolution against the reference:
/// MLX's f32 matmul is reduced-precision on Metal, and the only way to show that the trunk's
/// end-to-end residual is that floor accumulating — rather than a defect — is to measure a single
/// convolution, then a residual unit, then a block, then the whole stack.
#[derive(Debug, Clone)]
pub struct WnConv1d {
    /// `[out, kernel, in]` — torch stores `[out, in, kernel]`.
    weight: Array,
    bias: Array,
    stride: i32,
    padding: i32,
    dilation: i32,
}

impl WnConv1d {
    /// Load a `weight_norm`-parametrized conv (`weight_g` / `weight_v` / `bias`).
    ///
    /// Torch stores `Conv1d` weights as `[out, in, kernel]`; MLX wants `[out, kernel, in]`.
    pub fn from_weights(
        w: &mut Weights,
        prefix: &str,
        stride: i32,
        padding: i32,
        dilation: i32,
        dtype: Dtype,
    ) -> Result<Self> {
        let g = to_dtype(w.require(&format!("{prefix}.weight_g"))?, Dtype::Float32)?;
        let v = to_dtype(w.require(&format!("{prefix}.weight_v"))?, Dtype::Float32)?;
        let fused = fuse_weight_norm(&g, &v)?.transpose_axes(&[0, 2, 1])?;
        Ok(Self {
            weight: fused.as_dtype(dtype)?,
            bias: to_dtype(w.require(&format!("{prefix}.bias"))?, dtype)?,
            stride,
            padding,
            dilation,
        })
    }

    fn names(prefix: &str) -> Vec<String> {
        vec![
            format!("{prefix}.weight_g"),
            format!("{prefix}.weight_v"),
            format!("{prefix}.bias"),
        ]
    }

    /// `[B, T, C_in]` → `[B, T', C_out]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let y = conv1d(x, &self.weight, self.stride, self.padding, self.dilation, 1)?;
        Ok(add(&y, &self.bias)?)
    }
}

/// Load a `Snake1d` (`x + (α + 1e-9)⁻¹·sin²(αx)`) as a [`SnakeBeta`] with `beta = alpha`.
///
/// The checkpoint stores `alpha` as `[1, C, 1]`; [`SnakeBeta`] reads it by size and reshapes, so
/// the rank does not have to be flattened first. `logscale = false` — the encoder's Snake is
/// **not** the decoder's log-scaled SnakeBeta, and taking `exp(α)` here would be a plausible,
/// silently wrong port.
fn load_snake1d(w: &mut Weights, prefix: &str, dtype: Dtype) -> Result<SnakeBeta> {
    let alpha = to_dtype(w.require(&format!("{prefix}.alpha"))?, dtype)?;
    SnakeBeta::new(alpha.clone(), alpha, false)
}

/// `MiniMaxH3AudioResidualUnit` — `Snake → dilated Conv1d(k7) → Snake → Conv1d(k1)`, residual.
///
/// Public so the parity suite can hold one unit in isolation — see [`WnConv1d`].
#[derive(Debug, Clone)]
pub struct ResidualUnit {
    act1: SnakeBeta,
    conv1: WnConv1d,
    act2: SnakeBeta,
    conv2: WnConv1d,
}

impl ResidualUnit {
    /// Load one residual unit from a checkpoint under `prefix`.
    pub fn from_weights(
        w: &mut Weights,
        prefix: &str,
        dilation: i32,
        dtype: Dtype,
    ) -> Result<Self> {
        // `padding = ((7 - 1) * dilation) // 2` — exactly 'same', so the k7 convolution never
        // shortens the time axis. The reference still center-crops the shortcut when it would;
        // `forward` keeps that arm (see there) rather than assuming the padding stays 'same'.
        let pad = (7 - 1) * dilation / 2;
        Ok(Self {
            act1: load_snake1d(w, &format!("{prefix}.block.0"), dtype)?,
            conv1: WnConv1d::from_weights(
                w,
                &format!("{prefix}.block.1"),
                1,
                pad,
                dilation,
                dtype,
            )?,
            act2: load_snake1d(w, &format!("{prefix}.block.2"), dtype)?,
            conv2: WnConv1d::from_weights(w, &format!("{prefix}.block.3"), 1, 0, 1, dtype)?,
        })
    }

    fn names(prefix: &str) -> Vec<String> {
        let mut out = vec![format!("{prefix}.block.0.alpha")];
        out.extend(WnConv1d::names(&format!("{prefix}.block.1")));
        out.push(format!("{prefix}.block.2.alpha"));
        out.extend(WnConv1d::names(&format!("{prefix}.block.3")));
        out
    }

    /// `[B, T, C]` → `[B, T, C]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let y = self.act1.forward(x)?;
        let y = self.conv1.forward(&y)?;
        let y = self.act2.forward(&y)?;
        let y = self.conv2.forward(&y)?;
        // `pad = (x.len - y.len) // 2; if pad > 0: x = x[..., pad:-pad]`.
        let (xl, yl) = (x.shape()[1], y.shape()[1]);
        let pad = (xl - yl) / 2;
        let x = if pad > 0 {
            slice_axis(x, 1, pad, xl - pad)?
        } else {
            x.clone()
        };
        if x.shape()[1] != yl {
            return Err(Error::Msg(format!(
                "minimax-h3 audio encoder: residual unit shortcut is {} long, its block emitted \
                 {yl}",
                x.shape()[1]
            )));
        }
        Ok(add(&x, &y)?)
    }
}

/// `MiniMaxH3AudioEncoderBlock` — three residual units at dilations 1/3/9, a Snake, then the
/// strided channel-doubling convolution.
///
/// Public so the parity suite can hold one stage in isolation — see [`WnConv1d`].
#[derive(Debug, Clone)]
pub struct EncoderBlock {
    units: Vec<ResidualUnit>,
    act: SnakeBeta,
    down: WnConv1d,
}

/// The dilations every `EncoderBlock` applies, in order. Hardcoded by the reference.
const RESIDUAL_DILATIONS: [i32; 3] = [1, 3, 9];

impl EncoderBlock {
    /// Load one downsampling stage from a checkpoint under `prefix`.
    pub fn from_weights(w: &mut Weights, prefix: &str, stride: i32, dtype: Dtype) -> Result<Self> {
        let mut units = Vec::with_capacity(RESIDUAL_DILATIONS.len());
        for (i, &dilation) in RESIDUAL_DILATIONS.iter().enumerate() {
            units.push(ResidualUnit::from_weights(
                w,
                &format!("{prefix}.block.{i}"),
                dilation,
                dtype,
            )?);
        }
        let idx = RESIDUAL_DILATIONS.len();
        Ok(Self {
            units,
            act: load_snake1d(w, &format!("{prefix}.block.{idx}"), dtype)?,
            down: WnConv1d::from_weights(
                w,
                &format!("{prefix}.block.{}", idx + 1),
                stride,
                // `padding = ceil(stride / 2)`, NOT `stride / 2`: the shipped chain has odd
                // strides (5), where the two differ and only the ceiling makes the stack land on
                // exactly `samples / hop`.
                stride.div_euclid(2) + i32::from(stride.rem_euclid(2) != 0),
                1,
                dtype,
            )?,
        })
    }

    fn names(prefix: &str) -> Vec<String> {
        let mut out = Vec::new();
        for i in 0..RESIDUAL_DILATIONS.len() {
            out.extend(ResidualUnit::names(&format!("{prefix}.block.{i}")));
        }
        let idx = RESIDUAL_DILATIONS.len();
        out.push(format!("{prefix}.block.{idx}.alpha"));
        out.extend(WnConv1d::names(&format!("{prefix}.block.{}", idx + 1)));
        out
    }

    /// `[B, T, C/2]` → `[B, T/stride, C]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for unit in &self.units {
            x = unit.forward(&x)?;
        }
        let x = self.act.forward(&x)?;
        self.down.forward(&x)
    }
}

/// `MiniMaxH3AudioEncoder` — the DAC convolutional trunk.
///
/// Public so `tests/audio_vae_encode_parity.rs` can hold it against the reference without
/// `pre_block`: an end-to-end golden alone cannot say which of the two halves is wrong.
#[derive(Debug, Clone)]
pub struct DacEncoder {
    conv_in: WnConv1d,
    blocks: Vec<EncoderBlock>,
    act_out: SnakeBeta,
    conv_out: WnConv1d,
}

impl DacEncoder {
    /// Load the trunk from a checkpoint under `prefix` (`"encoder"` in the published naming).
    pub fn from_weights(
        w: &mut Weights,
        prefix: &str,
        cfg: &MiniMaxH3AudioVaeConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        let conv_in = WnConv1d::from_weights(w, &format!("{prefix}.block.0"), 1, 3, 1, dtype)?;
        let mut blocks = Vec::with_capacity(cfg.encoder_rates.len());
        for (i, &stride) in cfg.encoder_rates.iter().enumerate() {
            if stride < 1 {
                return Err(Error::Msg(format!(
                    "minimax-h3 audio encoder: encoder_rates[{i}] = {stride} must be positive"
                )));
            }
            blocks.push(EncoderBlock::from_weights(
                w,
                &format!("{prefix}.block.{}", i + 1),
                stride,
                dtype,
            )?);
        }
        let tail = cfg.encoder_rates.len() + 1;
        Ok(Self {
            conv_in,
            blocks,
            act_out: load_snake1d(w, &format!("{prefix}.block.{tail}"), dtype)?,
            conv_out: WnConv1d::from_weights(
                w,
                &format!("{prefix}.block.{}", tail + 1),
                1,
                1,
                1,
                dtype,
            )?,
        })
    }

    fn names(prefix: &str, cfg: &MiniMaxH3AudioVaeConfig) -> Vec<String> {
        let mut out = WnConv1d::names(&format!("{prefix}.block.0"));
        for i in 0..cfg.encoder_rates.len() {
            out.extend(EncoderBlock::names(&format!("{prefix}.block.{}", i + 1)));
        }
        let tail = cfg.encoder_rates.len() + 1;
        out.push(format!("{prefix}.block.{tail}.alpha"));
        out.extend(WnConv1d::names(&format!("{prefix}.block.{}", tail + 1)));
        out
    }

    /// `[B, S, 1]` → `[B, S/hop, latent_dim]`, both NLC.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = self.conv_in.forward(x)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = self.act_out.forward(&x)?;
        self.conv_out.forward(&x)
    }
}

/// PyTorch's `F.adaptive_avg_pool1d` over the LAST axis.
///
/// Output element `i` averages the half-open window
/// `[⌊i·L / out⌋, ⌈(i+1)·L / out⌉)`. When `out` divides `L` this is the obvious disjoint tiling,
/// which is the only case the shipped model reaches (256 → 32) — and precisely why the committed
/// fixture runs a geometry where the windows **overlap**, so a `reshape(.., out, L/out).mean(-1)`
/// port cannot pass.
///
/// Bounds are derived from the window rule and re-checked by [`slice_axis`], so an out-of-range
/// window is an error rather than one of MLX's silent out-of-bounds gathers.
pub fn adaptive_avg_pool_last_axis(x: &Array, out: i32) -> Result<Array> {
    let rank = x.shape().len();
    if rank == 0 {
        return Err(Error::Msg(
            "minimax-h3 adaptive pool: cannot pool a zero-rank tensor".into(),
        ));
    }
    let axis = rank as i32 - 1;
    let len = x.shape()[rank - 1];
    if out < 1 || len < 1 {
        return Err(Error::Msg(format!(
            "minimax-h3 adaptive pool: {len} -> {out} is not a valid pooling"
        )));
    }
    // i64 so `(i + 1) * len` cannot overflow for a large width.
    let (len64, out64) = (i64::from(len), i64::from(out));
    let mut parts = Vec::with_capacity(out as usize);
    for i in 0..out64 {
        let start = i * len64 / out64;
        let end = ((i + 1) * len64).div_euclid(out64)
            + i64::from(((i + 1) * len64).rem_euclid(out64) != 0);
        if start >= end || end > len64 {
            return Err(Error::Msg(format!(
                "minimax-h3 adaptive pool: window {i} of {out} is [{start}, {end}) over {len}"
            )));
        }
        let window = slice_axis(x, axis, start as i32, end as i32)?;
        parts.push(window.mean_axes(&[axis], true)?);
    }
    let refs: Vec<&Array> = parts.iter().collect();
    Ok(concatenate_axis(&refs, axis)?)
}

/// `MiniMaxH3AudioCausalAttention` — causal self-attention that NARROWS `in_dim` to `out_dim`.
///
/// Two departures from an ordinary attention block, both of which a port reproduces by accident
/// only if it read the reference:
///
/// * the per-head outputs are **averaged**, not concatenated (`torch.mean(x, dim=heads)`), so the
///   width after attention is `head_dim`, not `heads · head_dim`;
/// * that width is then **adaptively average-pooled** to `out_dim`
///   ([`adaptive_avg_pool_last_axis`]) — 256 → 32 on the shipped model.
///
/// Public so the parity suite can hold the attention branch on its own; the residual sum in
/// [`AttnProjection`] would otherwise hide a wrong branch inside a right-looking total.
#[derive(Debug, Clone)]
pub struct CausalAttention {
    /// `[3·in_dim, in_dim]`, read as three CONTIGUOUS thirds — see the module docs.
    qkv_w: Array,
    /// `cat(q_bias, zero_k_bias, v_bias)`, assembled at load.
    qkv_b: Array,
    proj_w: Array,
    proj_b: Array,
    heads: i32,
    head_dim: i32,
    out_dim: i32,
    scale: f32,
}

impl CausalAttention {
    /// Load from a checkpoint under `prefix` (`"pre_block.attn"` in the published naming).
    pub fn from_weights(
        w: &mut Weights,
        prefix: &str,
        in_dim: i32,
        out_dim: i32,
        heads: i32,
        dtype: Dtype,
    ) -> Result<Self> {
        if heads < 1 || in_dim % heads != 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 audio encoder: {in_dim} attention channels do not divide into {heads} \
                 heads"
            )));
        }
        let head_dim = in_dim / heads;
        let qkv_w = to_dtype(w.require(&format!("{prefix}.qkv.weight"))?, dtype)?;
        if qkv_w.shape() != [3 * in_dim, in_dim] {
            return Err(Error::Msg(format!(
                "minimax-h3 audio encoder: {prefix}.qkv.weight is {:?}, expected [{}, {in_dim}]",
                qkv_w.shape(),
                3 * in_dim
            )));
        }
        // The `nn.Linear` is bias-less; the reference builds the effective bias from three stored
        // tensors, of which the middle one is a ZERO BUFFER the checkpoint actually ships. Read
        // and concatenated rather than assumed, so the tensor is genuinely consumed.
        let q_b = to_dtype(w.require(&format!("{prefix}.q_bias"))?, dtype)?;
        let k_b = to_dtype(w.require(&format!("{prefix}.zero_k_bias"))?, dtype)?;
        let v_b = to_dtype(w.require(&format!("{prefix}.v_bias"))?, dtype)?;
        let qkv_b = concatenate_axis(&[&q_b, &k_b, &v_b], 0)?;
        if qkv_b.shape() != [3 * in_dim] {
            return Err(Error::Msg(format!(
                "minimax-h3 audio encoder: {prefix} q/k/v biases assemble to {:?}, expected [{}]",
                qkv_b.shape(),
                3 * in_dim
            )));
        }
        Ok(Self {
            qkv_w,
            qkv_b,
            proj_w: to_dtype(w.require(&format!("{prefix}.proj.weight"))?, dtype)?,
            proj_b: to_dtype(w.require(&format!("{prefix}.proj.bias"))?, dtype)?,
            heads,
            head_dim,
            out_dim,
            // `self.scale = head_dim ** -0.5`, matching torch SDPA's default.
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn names(prefix: &str) -> Vec<String> {
        vec![
            format!("{prefix}.qkv.weight"),
            format!("{prefix}.q_bias"),
            format!("{prefix}.zero_k_bias"),
            format!("{prefix}.v_bias"),
            format!("{prefix}.proj.weight"),
            format!("{prefix}.proj.bias"),
        ]
    }

    /// `[B, N, in_dim]` → `[B, N, out_dim]`, NLC.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let s = x.shape();
        if s.len() != 3 || s[2] != self.heads * self.head_dim {
            return Err(Error::Msg(format!(
                "minimax-h3 audio encoder attention: expected [B, N, {}], got {s:?}",
                self.heads * self.head_dim
            )));
        }
        let (b, n) = (s[0], s[1]);
        let qkv = linear(x, &self.qkv_w, &self.qkv_b)?;
        // `[B, N, 3·D] → [B, N, 3, H, Dh]`: the THIRDS are the outer split and the heads the
        // inner one. Reading it per-head-interleaved instead is shape-identical and wrong.
        let qkv = qkv.reshape(&[b, n, 3, self.heads, self.head_dim])?;
        let mut parts = Vec::with_capacity(3);
        for i in 0..3 {
            parts.push(
                slice_axis(&qkv, 2, i, i + 1)?
                    .reshape(&[b, n, self.heads, self.head_dim])?
                    // [B, N, H, Dh] → [B, H, N, Dh]
                    .transpose_axes(&[0, 2, 1, 3])?,
            );
        }
        let attended = scaled_dot_product_attention(
            &parts[0],
            &parts[1],
            &parts[2],
            self.scale,
            // `is_causal=True` — every query attends to its own position and earlier ones only.
            ScaledDotProductAttentionMask::Causal,
            None,
        )?;
        // Mean-pool the HEADS away: `[B, H, N, Dh]` → `[B, N, Dh]`.
        let pooled = attended.mean_axes(&[1], false)?;
        let pooled = adaptive_avg_pool_last_axis(&pooled, self.out_dim)?;
        linear(&pooled, &self.proj_w, &self.proj_b)
    }
}

/// `MiniMaxH3AudioGeGluMlp` — `w2( gelu_tanh(w0(norm(x))) · w1(norm(x)) )`.
///
/// `w0` is the **gate** and `w1` the **value**. They are two separate, shape-identical tensors,
/// so [`crate::layout::split_gate_value`] does not apply (there is no fused projection to halve)
/// — but swapping them is exactly the sc-18740 signature, and the parity suite pins it.
#[derive(Debug, Clone)]
struct GeGluMlp {
    norm_w: Array,
    norm_b: Array,
    w0: (Array, Array),
    w1: (Array, Array),
    w2: (Array, Array),
}

fn load_linear(w: &mut Weights, prefix: &str, dtype: Dtype) -> Result<(Array, Array)> {
    Ok((
        to_dtype(w.require(&format!("{prefix}.weight"))?, dtype)?,
        to_dtype(w.require(&format!("{prefix}.bias"))?, dtype)?,
    ))
}

fn linear_names(prefix: &str) -> [String; 2] {
    [format!("{prefix}.weight"), format!("{prefix}.bias")]
}

impl GeGluMlp {
    fn from_weights(w: &mut Weights, prefix: &str, dim: i32, dtype: Dtype) -> Result<Self> {
        let (norm_w, norm_b) = load_linear(w, &format!("{prefix}.norm"), dtype)?;
        let hidden = dim.checked_mul(MLP_RATIO).ok_or_else(|| {
            Error::Msg(format!(
                "minimax-h3 audio encoder: {dim} overflows the GeGLU width"
            ))
        })?;
        let w0 = load_linear(w, &format!("{prefix}.w0"), dtype)?;
        let w1 = load_linear(w, &format!("{prefix}.w1"), dtype)?;
        let w2 = load_linear(w, &format!("{prefix}.w2"), dtype)?;
        // `w0` and `w1` are the gate and the value: shape-identical, and therefore silently
        // interchangeable. The shapes cannot tell them apart — only the parity fixture can — but
        // they CAN tell a checkpoint at a different `mlp_ratio` apart, which is what this checks.
        for (name, weight, want) in [
            ("w0", &w0.0, [hidden, dim]),
            ("w1", &w1.0, [hidden, dim]),
            ("w2", &w2.0, [dim, hidden]),
        ] {
            if weight.shape() != want {
                return Err(Error::Msg(format!(
                    "minimax-h3 audio encoder: {prefix}.{name}.weight is {:?}, expected {want:?}                      at mlp_ratio {MLP_RATIO}",
                    weight.shape()
                )));
            }
        }
        Ok(Self {
            norm_w,
            norm_b,
            w0,
            w1,
            w2,
        })
    }

    fn names(prefix: &str) -> Vec<String> {
        let mut out = Vec::new();
        for leaf in ["norm", "w0", "w1", "w2"] {
            out.extend(linear_names(&format!("{prefix}.{leaf}")));
        }
        out
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = layer_norm(x, Some(&self.norm_w), Some(&self.norm_b), LAYER_NORM_EPS)?;
        let gate = gelu_tanh(&linear(&h, &self.w0.0, &self.w0.1)?)?;
        let value = linear(&h, &self.w1.0, &self.w1.1)?;
        let h = multiply(&gate, &value)?;
        linear(&h, &self.w2.0, &self.w2.1)
    }
}

/// `pre_block` — `MiniMaxH3AudioAttnProjection`, the residual causal-attention + GeGLU block that
/// rewires the `latent_dim`-wide encoder trunk to the `latent_channels`-wide latent.
///
/// ```text
/// h = proj(norm3(x)) + attn(norm1(x))
/// y = h + mlp(norm2(h))
/// ```
///
/// Note `norm1` and `norm3` are two DIFFERENT LayerNorms over the same input, and `norm2` sees
/// the intermediate `h`, not `x`. Public for the same reason [`CausalAttention`] is.
#[derive(Debug, Clone)]
pub struct AttnProjection {
    norm1: (Array, Array),
    norm2: (Array, Array),
    norm3: (Array, Array),
    attn: CausalAttention,
    proj: (Array, Array),
    mlp: GeGluMlp,
}

impl AttnProjection {
    /// Load from a checkpoint under `prefix` (`"pre_block"` in the published naming).
    pub fn from_weights(
        w: &mut Weights,
        prefix: &str,
        in_dim: i32,
        out_dim: i32,
        heads: i32,
        dtype: Dtype,
    ) -> Result<Self> {
        Ok(Self {
            norm1: load_linear(w, &format!("{prefix}.norm1"), dtype)?,
            norm2: load_linear(w, &format!("{prefix}.norm2"), dtype)?,
            norm3: load_linear(w, &format!("{prefix}.norm3"), dtype)?,
            attn: CausalAttention::from_weights(
                w,
                &format!("{prefix}.attn"),
                in_dim,
                out_dim,
                heads,
                dtype,
            )?,
            proj: load_linear(w, &format!("{prefix}.proj"), dtype)?,
            mlp: GeGluMlp::from_weights(w, &format!("{prefix}.mlp"), out_dim, dtype)?,
        })
    }

    fn names(prefix: &str) -> Vec<String> {
        let mut out = Vec::new();
        for leaf in ["norm1", "norm2", "norm3", "proj"] {
            out.extend(linear_names(&format!("{prefix}.{leaf}")));
        }
        out.extend(CausalAttention::names(&format!("{prefix}.attn")));
        out.extend(GeGluMlp::names(&format!("{prefix}.mlp")));
        out
    }

    /// `[B, N, in_dim]` → `[B, N, out_dim]`, NLC.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let skip = linear(
            &layer_norm(x, Some(&self.norm3.0), Some(&self.norm3.1), LAYER_NORM_EPS)?,
            &self.proj.0,
            &self.proj.1,
        )?;
        let attended = self.attn.forward(&layer_norm(
            x,
            Some(&self.norm1.0),
            Some(&self.norm1.1),
            LAYER_NORM_EPS,
        )?)?;
        let h = add(&skip, &attended)?;
        let mlp = self.mlp.forward(&layer_norm(
            &h,
            Some(&self.norm2.0),
            Some(&self.norm2.1),
            LAYER_NORM_EPS,
        )?)?;
        Ok(add(&h, &mlp)?)
    }
}

/// The posterior of the MiniMax-H3 **audio** VAE: `(mean, log σ)`.
///
/// # This is NOT [`crate::vae_encoder::DiagonalGaussian`]
///
/// The video encoder's posterior splits one fused `[B, 2C, T, H, W]` moments tensor whose second
/// half is a **log variance**, and clamps it to `[-30, 20]` the way diffusers does. The audio
/// encoder is different on all three counts, and mixing them up is silent:
///
/// | | video (`vae_encoder`) | audio (here) |
/// |---|---|---|
/// | source | one fused `quant_conv` output, split in halves | two SEPARATE heads, `mean_proj` / `logs_proj` |
/// | second parameter | log **variance** ⇒ `σ = exp(½·logvar)` | log **standard deviation** ⇒ `σ = exp(logs)` |
/// | clamp | `[-30, 20]` | **none** |
///
/// Reusing the video type here would apply a factor of ½ to the exponent and a clamp the
/// reference does not have — a wrong `σ` that leaves `mode()` (the only thing the MiniMax-H3
/// pipeline consumes) completely unchanged, so nothing downstream of the mean would ever notice.
/// `tests/audio_vae_encode_parity.rs::log_std_is_not_log_variance` pins the distinction.
#[derive(Debug, Clone)]
pub struct AudioDiagonalGaussian {
    mean: Array,
    logs: Array,
    std: Array,
}

impl AudioDiagonalGaussian {
    /// Build from the two head outputs. `logs` is the log **standard deviation**.
    pub fn new(mean: Array, logs: Array) -> Result<Self> {
        if mean.shape() != logs.shape() {
            return Err(Error::Msg(format!(
                "minimax-h3 audio encoder: posterior mean {:?} and logs {:?} differ in shape",
                mean.shape(),
                logs.shape()
            )));
        }
        let std = exp(&logs)?;
        Ok(Self { mean, logs, std })
    }

    /// The posterior mean — and `mode()`, which is the ONLY thing the MiniMax-H3 pipeline uses.
    pub fn mean(&self) -> &Array {
        &self.mean
    }

    /// The raw `logs_proj` output: log **standard deviation**.
    pub fn logs(&self) -> &Array {
        &self.logs
    }

    /// `exp(logs)` — the per-element standard deviation.
    pub fn std(&self) -> &Array {
        &self.std
    }

    /// `mode()`: bit-for-bit `mean_proj`'s output, as the reference documents.
    pub fn mode(&self) -> &Array {
        &self.mean
    }

    /// `mean + std · noise`. `noise` must match the mean's shape.
    pub fn sample_with(&self, noise: &Array) -> Result<Array> {
        if noise.shape() != self.mean.shape() {
            return Err(Error::Msg(format!(
                "minimax-h3 audio encoder: posterior noise {:?} does not match the mean {:?}",
                noise.shape(),
                self.mean.shape()
            )));
        }
        Ok(add(&self.mean, &multiply(&self.std, noise)?)?)
    }
}

/// The MiniMax-H3 audio VAE's **encode** half.
#[derive(Debug, Clone)]
pub struct MiniMaxH3AudioVaeEncoder {
    encoder: DacEncoder,
    pre_block: AttnProjection,
    mean_w: Array,
    mean_b: Array,
    logs_w: Array,
    logs_b: Array,
    cfg: MiniMaxH3AudioVaeConfig,
    latents_mean: Array,
    latents_std: Array,
    hop: i32,
}

impl MiniMaxH3AudioVaeEncoder {
    /// Every tensor name the encode path consumes, in the published checkpoint's naming.
    ///
    /// 173 for the shipped geometry — the exact complement of
    /// [`crate::audio_vae::MiniMaxH3AudioVae::tensor_names`]'s 914 within the checkpoint's 1087.
    /// The real-weight test asserts that partition rather than assuming it.
    pub fn tensor_names(cfg: &MiniMaxH3AudioVaeConfig) -> Vec<String> {
        let mut out = DacEncoder::names("encoder", cfg);
        out.extend(AttnProjection::names("pre_block"));
        out.extend(linear_names("mean_proj"));
        out.extend(linear_names("logs_proj"));
        out
    }

    /// Load the encode half from a checkpoint in the published naming, at the reference's
    /// hardcoded [`ATTN_PROJ_HEADS`].
    pub fn from_weights(
        w: &mut Weights,
        cfg: &MiniMaxH3AudioVaeConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        Self::from_weights_with_heads(w, cfg, ATTN_PROJ_HEADS, dtype)
    }

    /// [`Self::from_weights`] with an explicit `pre_block` head count.
    ///
    /// The head count is genuinely a model parameter — diffusers exposes it as
    /// `num_attention_heads` and the published `audio_vae/config.json` declares `8` — but
    /// [`MiniMaxH3AudioVaeConfig`] is built from the FL2VA source triple, where it does not
    /// appear at all (the original hardcodes it). Rather than infer it from a shape (it is not
    /// recoverable from one: `qkv.weight` is `[3·in_dim, in_dim]` for every head count), it is
    /// passed here, so the parity fixture can pin a geometry whose adaptive pool is ragged
    /// instead of the shipped exact-8:1.
    ///
    /// # Errors
    ///
    /// - `attn_proj = false` — the published checkpoint sets it `true`, and without `pre_block`
    ///   the `latent_dim`-wide trunk cannot feed the `latent_channels`-wide posterior heads.
    /// - `latent_dim % latent_channels != 0` — the original then widens `attn_proj_dim` to the
    ///   next power of two, which changes `mean_proj`'s input width. diffusers rejects that
    ///   configuration outright and no published checkpoint takes it, so it is refused rather
    ///   than half-implemented.
    pub fn from_weights_with_heads(
        w: &mut Weights,
        cfg: &MiniMaxH3AudioVaeConfig,
        heads: i32,
        dtype: Dtype,
    ) -> Result<Self> {
        if !cfg.attn_proj {
            return Err(Error::Msg(
                "minimax-h3 audio encoder: attn_proj = false has no pre_block, so the encoder \
                 trunk cannot reach the posterior heads (the published checkpoint sets it true)"
                    .into(),
            ));
        }
        let latent_dim = cfg.bigvgan.num_mels;
        let channels = cfg.latent_channels;
        if channels < 1 || latent_dim < 1 || latent_dim % channels != 0 {
            return Err(Error::Msg(format!(
                "minimax-h3 audio encoder: latent_dim {latent_dim} is not a multiple of \
                 latent_channels {channels}; the reference would widen attn_proj_dim to the next \
                 power of two, which diffusers rejects"
            )));
        }
        let hop = hop_length(&cfg.encoder_rates)?;

        let encoder = DacEncoder::from_weights(w, "encoder", cfg, dtype)?;
        let pre_block =
            AttnProjection::from_weights(w, "pre_block", latent_dim, channels, heads, dtype)?;

        let head = |name: &str| -> Result<(Array, Array)> {
            let weight = to_dtype(w.require(&format!("{name}.weight"))?, dtype)?;
            if weight.shape() != [channels, channels, 1] {
                return Err(Error::Msg(format!(
                    "minimax-h3 audio encoder: {name}.weight is {:?}, expected \
                     [{channels}, {channels}, 1]",
                    weight.shape()
                )));
            }
            // A 1x1 conv is a pointwise linear: squeeze the kernel axis rather than convolve it.
            Ok((
                weight.reshape(&[channels, channels])?,
                to_dtype(w.require(&format!("{name}.bias"))?, dtype)?,
            ))
        };
        let (mean_w, mean_b) = head("mean_proj")?;
        let (logs_w, logs_b) = head("logs_proj")?;

        let n = channels as usize;
        if cfg.latents_mean.len() != n || cfg.latents_std.len() != n {
            return Err(Error::Msg(format!(
                "minimax-h3 audio encoder: latents_mean/std carry {}/{} entries for {channels} \
                 latent channels",
                cfg.latents_mean.len(),
                cfg.latents_std.len()
            )));
        }
        let latents_mean =
            Array::from_slice(&cfg.latents_mean, &[1, 1, channels]).as_dtype(dtype)?;
        let latents_std = Array::from_slice(&cfg.latents_std, &[1, 1, channels]).as_dtype(dtype)?;

        Ok(Self {
            encoder,
            pre_block,
            mean_w,
            mean_b,
            logs_w,
            logs_b,
            cfg: cfg.clone(),
            latents_mean,
            latents_std,
            hop,
        })
    }

    /// The configuration this instance was built from.
    pub fn config(&self) -> &MiniMaxH3AudioVaeConfig {
        &self.cfg
    }

    /// `∏ encoder_rates` — 800, i.e. 40 latents/s at 32 kHz.
    pub fn hop_length(&self) -> i32 {
        self.hop
    }

    /// `AutoencoderKLMiniMaxH3Audio.encode`: `[B, 1, samples]` → posterior over
    /// `[B, latent_channels, ⌈samples / hop⌉]`.
    ///
    /// **The model is mono.** MiniMax-H3 carries a stereo reference clip as `B = 2` — the two
    /// channels are two BATCH items through the same weights, exactly as the decode half emits
    /// them ([`crate::audio_vae::MiniMaxH3AudioVae::decode_stereo`] folds the channel axis into
    /// the batch for the same reason). A `[1, 2, samples]` clip is therefore rejected here rather
    /// than encoded as a two-channel waveform the model has never seen.
    ///
    /// Short inputs are **zero-padded on the right** to a whole number of hops, as the reference
    /// does; nothing is trimmed off the result.
    ///
    /// Reference-exact — **no** normalization (see [`Self::normalize`]).
    pub fn encode(&self, waveform: &Array) -> Result<AudioDiagonalGaussian> {
        let s = waveform.shape();
        if s.len() != 3 || s[1] != 1 {
            return Err(Error::Msg(format!(
                "minimax-h3 audio encode: expected [B, 1, samples] (stereo is B = 2), got {s:?}"
            )));
        }
        let samples = s[2];
        if samples < 1 {
            return Err(Error::Msg(
                "minimax-h3 audio encode: an empty waveform has no latents".into(),
            ));
        }
        // `right_pad = ceil(S / hop) * hop - S`, zeros.
        let frames = samples.div_euclid(self.hop) + i32::from(samples.rem_euclid(self.hop) != 0);
        let padded_len = frames * self.hop;
        // NCL -> NLC once, at the boundary.
        let mut x = waveform.transpose_axes(&[0, 2, 1])?;
        if padded_len > samples {
            let pad = Array::zeros::<f32>(&[s[0], padded_len - samples, 1])?.as_dtype(x.dtype())?;
            x = concatenate_axis(&[&x, &pad], 1)?;
        }

        let hidden = self.encoder.forward(&x)?;
        let got = hidden.shape()[1];
        if got != frames {
            return Err(Error::Msg(format!(
                "minimax-h3 audio encode: {padded_len} samples produced {got} frames, expected \
                 {frames} at hop {}",
                self.hop
            )));
        }
        let hidden = self.pre_block.forward(&hidden)?;
        let mean = linear(&hidden, &self.mean_w, &self.mean_b)?;
        let logs = linear(&hidden, &self.logs_w, &self.logs_b)?;
        AudioDiagonalGaussian::new(
            mean.transpose_axes(&[0, 2, 1])?,
            logs.transpose_axes(&[0, 2, 1])?,
        )
    }

    /// Per-channel latent normalization, `(z − latents_mean) / latents_std`.
    ///
    /// The exact inverse of [`crate::audio_vae::MiniMaxH3AudioVae::denormalize`], and applied
    /// over the same axis: the **channel** axis is the second-to-last, so this works unchanged
    /// for the mono `[B, C, T]` this encoder emits and for the stereo `[B, 2, C, T]` packing the
    /// decode half takes. `encode` deliberately does not apply it, so it stays a byte-for-byte
    /// analogue of the reference's `encode`; the pipeline normalizes afterwards.
    pub fn normalize(&self, z: &Array) -> Result<Array> {
        let rank = z.shape().len();
        if rank < 2 {
            return Err(Error::Msg(format!(
                "minimax-h3 audio encoder: cannot normalize a rank-{rank} latent"
            )));
        }
        let channels = z.shape()[rank - 2];
        if channels != self.cfg.latent_channels {
            return Err(Error::Msg(format!(
                "minimax-h3 audio encoder: latent has {channels} channels, config declares {}",
                self.cfg.latent_channels
            )));
        }
        let mut shape = vec![1; rank];
        shape[rank - 2] = channels;
        let mean = self.latents_mean.reshape(&shape)?;
        let std = self.latents_std.reshape(&shape)?;
        Ok(divide(&subtract(z, &mean)?, &std)?)
    }
}

/// `∏ encoder_rates`, checked for overflow and for a degenerate zero rate.
fn hop_length(rates: &[i32]) -> Result<i32> {
    if rates.is_empty() {
        return Err(Error::Msg(
            "minimax-h3 audio encoder: encoder_rates is empty, so the hop length is undefined"
                .into(),
        ));
    }
    let mut hop: i32 = 1;
    for &rate in rates {
        if rate < 1 {
            return Err(Error::Msg(format!(
                "minimax-h3 audio encoder: encoder rate {rate} must be positive"
            )));
        }
        hop = hop.checked_mul(rate).ok_or_else(|| {
            Error::Msg(format!(
                "minimax-h3 audio encoder: encoder_rates {rates:?} overflow the hop length"
            ))
        })?;
    }
    Ok(hop)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic host-side data — MLX's RNG needs a GPU stream that not every test worker has.
    fn spread(shape: &[i32]) -> Array {
        let n: i32 = shape.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin() * 1.4).collect();
        Array::from_slice(&vals, shape)
    }

    /// `w = g · v / ‖v‖` with the norm over axes 1.., same rule as the decoder's fusion.
    #[test]
    fn weight_norm_fusion_normalizes_per_output_channel() {
        let v = Array::from_slice(&[3.0f32, 4.0, 0.0, 1.0], &[2, 1, 2]);
        let g = Array::from_slice(&[10.0f32, 2.0], &[2, 1, 1]);
        let fused = fuse_weight_norm(&g, &v).unwrap();
        assert_eq!(fused.shape(), &[2, 1, 2]);
        for (a, b) in fused.as_slice::<f32>().iter().zip([6.0f32, 8.0, 0.0, 2.0]) {
            assert!((a - b).abs() < 1e-6);
        }
        // Shape mismatches are typed errors, not broadcasts.
        assert!(fuse_weight_norm(&Array::from_slice(&[1.0f32, 1.0], &[2]), &v).is_err());
        assert!(
            fuse_weight_norm(&g, &Array::from_slice(&[1.0f32, 2.0], &[2, 1])).is_err(),
            "weight_v must be rank 3"
        );
    }

    /// The strided convolution's padding is `ceil(stride / 2)`, and that ceiling is what makes the
    /// shipped `(2, 4, 4, 5, 5)` chain land on exactly `samples / 800`.
    ///
    /// Written as the length arithmetic rather than as a constant, so an implementation that used
    /// `stride / 2` fails here instead of only on a real 5-strided checkpoint.
    #[test]
    fn strided_padding_makes_every_stage_divide_exactly() {
        let out_len = |len: i32, stride: i32, padding: i32| {
            let kernel = 2 * stride;
            (len + 2 * padding - kernel).div_euclid(stride) + 1
        };
        let ceil_half = |stride: i32| stride.div_euclid(2) + i32::from(stride.rem_euclid(2) != 0);
        assert_eq!(ceil_half(2), 1);
        assert_eq!(ceil_half(4), 2);
        assert_eq!(ceil_half(5), 3, "ceil(5/2) = 3, NOT 2");

        // The shipped chain, from 800 samples down to one latent frame.
        let mut len = 800;
        for stride in [2, 4, 4, 5, 5] {
            len = out_len(len, stride, ceil_half(stride));
        }
        assert_eq!(
            len, 1,
            "800 samples must encode to exactly one latent frame"
        );

        // Flooring instead of ceiling loses a frame at every odd stride.
        let mut wrong = 800;
        for stride in [2, 4, 4, 5, 5] {
            wrong = out_len(wrong, stride, stride / 2);
        }
        assert_ne!(wrong, 1, "stride/2 padding must NOT reproduce the hop");
    }

    /// The general adaptive-pool window rule, against hand-computed cases.
    ///
    /// The shipped 256 → 32 is an exact tiling; 48 → 32 OVERLAPS (window 0 is `[0, 2)` and
    /// window 1 is `[1, 3)`), which no reshape-and-mean can express; 44 → 32 additionally has
    /// windows of two different widths.
    #[test]
    fn adaptive_pool_matches_the_window_rule() {
        // 4 → 2 is the exact-tiling case: means of [0,1] and [2,3].
        let x = Array::from_slice(&[1.0f32, 3.0, 10.0, 20.0], &[1, 4]);
        let y = adaptive_avg_pool_last_axis(&x, 2).unwrap();
        assert_eq!(y.shape(), &[1, 2]);
        assert_eq!(y.as_slice::<f32>(), &[2.0, 15.0]);

        // 3 → 2 overlaps: windows [0, 2) and [1, 3).
        let x = Array::from_slice(&[1.0f32, 3.0, 11.0], &[1, 3]);
        let y = adaptive_avg_pool_last_axis(&x, 2).unwrap();
        assert_eq!(y.as_slice::<f32>(), &[2.0, 7.0]);

        // out == len is the identity.
        let x = spread(&[2, 3, 5]);
        let same = adaptive_avg_pool_last_axis(&x, 5).unwrap();
        assert_eq!(same.shape(), x.shape());
        for (a, b) in same.as_slice::<f32>().iter().zip(x.as_slice::<f32>()) {
            assert!((a - b).abs() < 1e-6);
        }

        // Shapes for the three regimes the fixture pins.
        for (len, out) in [(256, 32), (48, 32), (44, 32)] {
            let pooled = adaptive_avg_pool_last_axis(&spread(&[2, 3, len]), out).unwrap();
            assert_eq!(pooled.shape(), &[2, 3, out]);
        }
        assert!(adaptive_avg_pool_last_axis(&spread(&[2, 4]), 0).is_err());
    }

    /// 48 → 32 really does overlap and 44 → 32 really does vary in width, so the fixture cases
    /// are the discriminating ones the parity suite claims they are.
    #[test]
    fn the_fixtures_pooling_regimes_are_distinguishable() {
        let windows = |len: i64, out: i64| -> Vec<(i64, i64)> {
            (0..out)
                .map(|i| {
                    let start = i * len / out;
                    let end = ((i + 1) * len).div_euclid(out)
                        + i64::from(((i + 1) * len).rem_euclid(out) != 0);
                    (start, end)
                })
                .collect()
        };
        let shipped = windows(256, 32);
        assert!(
            shipped.windows(2).all(|w| w[0].1 == w[1].0),
            "256 -> 32 must be a disjoint tiling"
        );
        let ragged = windows(48, 32);
        assert!(
            ragged.windows(2).any(|w| w[0].1 > w[1].0),
            "48 -> 32 must overlap, or a reshape-and-mean port would pass the fixture"
        );
        let varsize = windows(44, 32);
        let widths: std::collections::BTreeSet<i64> = varsize.iter().map(|(a, b)| b - a).collect();
        assert!(widths.len() > 1, "44 -> 32 must have windows of two widths");
    }

    /// `Snake1d` is `SnakeBeta` with `beta = alpha` and NO log scale — the identity this module
    /// relies on to reuse the decoder's activation instead of growing a second one.
    #[test]
    fn snake1d_is_snakebeta_with_beta_equal_to_alpha() {
        let x = spread(&[1, 4, 3]);
        let alpha = Array::from_slice(&[0.7f32, 1.1, 0.4], &[1, 3, 1]);
        let got = SnakeBeta::new(alpha.clone(), alpha.clone(), false)
            .unwrap()
            .forward(&x)
            .unwrap();
        // `x + (alpha + 1e-9)^-1 * sin(alpha*x)^2`, computed here on the host.
        let a = [0.7f32, 1.1, 0.4];
        let xs = x.as_slice::<f32>();
        for (i, (&xi, gi)) in xs.iter().zip(got.as_slice::<f32>()).enumerate() {
            let ai = a[i % 3];
            let want = xi + (ai * xi).sin().powi(2) / (ai + 1e-9);
            assert!((gi - want).abs() < 1e-5, "index {i}: {gi} vs {want}");
        }
        // The log scale is a DIFFERENT function; taking it here would be a plausible wrong port.
        let logged = SnakeBeta::new(alpha.clone(), alpha, true)
            .unwrap()
            .forward(&x)
            .unwrap();
        let gap = mlx_rs::ops::subtract(&logged, &got)
            .unwrap()
            .abs()
            .unwrap()
            .max(None)
            .unwrap()
            .item::<f32>();
        assert!(gap > 1e-2, "log-scaled Snake agreed with the linear one");
    }

    /// The hop is the product of the encoder rates, and degenerate rate lists are typed errors.
    #[test]
    fn hop_is_the_product_of_the_encoder_rates() {
        assert_eq!(hop_length(&[2, 4, 4, 5, 5]).unwrap(), 800);
        assert_eq!(hop_length(&[2]).unwrap(), 2);
        assert!(hop_length(&[]).is_err());
        assert!(hop_length(&[2, 0, 4]).is_err());
        assert!(hop_length(&[i32::MAX, 2]).is_err());
    }

    /// The declared name list must be unique and exactly the encode half's cardinality — 173 for
    /// the shipped geometry, the complement of the decoder's 914 within the published 1087.
    #[test]
    fn declared_tensor_names_are_unique_and_complete() {
        let cfg = MiniMaxH3AudioVaeConfig::default();
        let names = MiniMaxH3AudioVaeEncoder::tensor_names(&cfg);
        let unique: std::collections::BTreeSet<&String> = names.iter().collect();
        assert_eq!(unique.len(), names.len(), "duplicate declared tensor name");

        // 3 conv_in + 5 * (3 * 8 residual + 1 alpha + 3 downsample) + 1 alpha + 3 conv_out
        //   + 22 pre_block + 2 mean_proj + 2 logs_proj
        assert_eq!(names.len(), 3 + 5 * (24 + 4) + 4 + 22 + 4);
        assert_eq!(names.len(), 173);
        assert_eq!(
            names.len() + crate::audio_vae::MiniMaxH3AudioVae::tensor_names(&cfg).len(),
            1087,
            "the two halves must partition the published checkpoint exactly"
        );

        // Structural spot checks a renumbering would break.
        for expected in [
            "encoder.block.0.weight_v",
            "encoder.block.5.block.2.block.3.bias",
            "encoder.block.5.block.3.alpha",
            "encoder.block.5.block.4.weight_g",
            "encoder.block.6.alpha",
            "encoder.block.7.weight_v",
            "pre_block.attn.zero_k_bias",
            "pre_block.mlp.w1.weight",
            "logs_proj.bias",
        ] {
            assert!(names.iter().any(|n| n == expected), "missing {expected}");
        }
        // `block.6` is the trailing Snake, so there is no `block.6.block.*` sub-tree.
        assert!(!names.iter().any(|n| n.starts_with("encoder.block.6.block")));
        // The decode half must not leak in.
        assert!(!names.iter().any(|n| n.starts_with("decoder.")));
        assert!(!names.iter().any(|n| n.starts_with("dec_in_proj.")));
    }

    /// The declared names track `encoder_rates`, so a checkpoint at a different depth is a
    /// different key set rather than a silently truncated read.
    #[test]
    fn declared_names_follow_the_encoder_depth() {
        let cfg = MiniMaxH3AudioVaeConfig {
            encoder_rates: vec![2, 5],
            ..MiniMaxH3AudioVaeConfig::default()
        };
        let names = MiniMaxH3AudioVaeEncoder::tensor_names(&cfg);
        assert_eq!(names.len(), 3 + 2 * 28 + 4 + 26);
        assert!(names.iter().any(|n| n == "encoder.block.3.alpha"));
        assert!(names.iter().any(|n| n == "encoder.block.4.weight_v"));
        assert!(!names.iter().any(|n| n.starts_with("encoder.block.5")));
    }

    /// The posterior takes `logs` as a log STANDARD DEVIATION, with no clamp.
    #[test]
    fn posterior_exponentiates_the_log_standard_deviation() {
        let mean = Array::from_slice(&[1.0f32, -2.0], &[1, 2, 1]);
        let logs = Array::from_slice(&[0.0f32, 2.0], &[1, 2, 1]);
        let p = AudioDiagonalGaussian::new(mean, logs).unwrap();
        assert_eq!(p.mode().as_slice::<f32>(), &[1.0, -2.0]);
        let std = p.std().as_slice::<f32>();
        assert!((std[0] - 1.0).abs() < 1e-6);
        // exp(2), NOT exp(0.5 * 2) — the video encoder's log-variance convention would give e.
        assert!(
            (std[1] - std::f32::consts::E.powi(2)).abs() < 1e-4,
            "std = exp(logs), got {}",
            std[1]
        );

        let zero = mlx_rs::ops::zeros_dtype(&[1, 2, 1], Dtype::Float32).unwrap();
        assert_eq!(
            p.sample_with(&zero).unwrap().as_slice::<f32>(),
            p.mean().as_slice::<f32>()
        );
        let wrong = mlx_rs::ops::zeros_dtype(&[1, 3, 1], Dtype::Float32).unwrap();
        assert!(p.sample_with(&wrong).is_err());
        assert!(
            AudioDiagonalGaussian::new(
                Array::from_slice(&[1.0f32], &[1]),
                Array::from_slice(&[1.0f32, 2.0], &[2])
            )
            .is_err(),
            "mismatched head shapes are rejected"
        );
    }

    /// The `pre_block` head count is a real knob, not a spelling: it changes `head_dim`, which
    /// changes both the attention scale and the adaptive pool's window layout.
    #[test]
    fn head_count_changes_the_attention_geometry() {
        for (heads, head_dim) in [(8, 256), (2, 1024)] {
            assert_eq!(2048 / heads, head_dim);
            let scale = (head_dim as f32).powf(-0.5);
            assert!(scale > 0.0);
        }
        assert_eq!(ATTN_PROJ_HEADS, 8, "the reference hardcodes 8 heads");
        // 2048 / 8 = 256 pools to 32 as an exact 8:1 — the case a naive reshape reproduces, and
        // the reason the fixture runs a ragged one.
        assert_eq!((2048 / ATTN_PROJ_HEADS) % 32, 0);
    }
}
