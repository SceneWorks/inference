//! The MiniMax-H3 audio VAE **encode** path (sc-17157, `ref2va`), candle side.
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
//! Everything convolutional runs in **NCL**, candle's native conv layout *and* the reference's, so
//! this port needs none of the MLX twin's boundary transposes — the one transpose is into
//! `pre_block`, which is genuinely a sequence block over `[B, N, C]`.
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
//! directly, one `α` per channel, **no log scale**. That is exactly [`crate::alias_free::SnakeBeta`]
//! with `beta = alpha` and `logscale = false`, which is how this module reuses it rather than
//! growing a second periodic activation.
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
//! only executable reference for this half. `tests/audio_vae_encode_parity.rs` asserts against the
//! committed golden that generator produced, which is **byte-identical to the MLX lane's copy** —
//! the shared-fixture half of cross-backend agreement.
//!
//! # Precision
//!
//! diffusers pins this component with `_keep_in_fp32_modules = ["encoder", …, "mean_proj",
//! "logs_proj"]`: the weight-normed convolutions and Snake activations "degrade audibly under
//! bfloat16 (roughly 20 dB quieter decodes)". Callers should pass [`DType::F32`] unless they have
//! measured otherwise, and `crate::model` does.
//!
//! # Deliberate duplication
//!
//! `fuse_weight_norm` and the weight-normed `Conv1d` are re-implemented here rather than shared
//! with [`crate::audio_vae`], whose versions are private and stride-less. The encoder needs a
//! **strided** convolution (`stride ∈ encoder_rates`), so the two are not interchangeable; the
//! fusion arithmetic (`w = g · v / ‖v‖`, norm over every axis but the first) is identical and is
//! asserted against the same rule in both modules' unit tests.

use candle_gen::candle_core::{DType, Device, Tensor};
use candle_gen::candle_nn::ops::softmax_last_dim;
use candle_gen::grounding::causal_mask;
use candle_gen::{CandleError, Result, Weights};

use crate::alias_free::SnakeBeta;
use crate::audio_config::MiniMaxH3AudioVaeConfig;
use crate::nn::{layer_norm, linear};

/// Heads in the `pre_block` causal attention.
///
/// The original `DacAudioVAE` **hardcodes** `AttnProjection(latent_dim, attn_proj_dim,
/// num_heads=8)` — it is not in `metadata.json`, `config.yaml` or the FL2VA `config.json`, so
/// [`MiniMaxH3AudioVaeConfig`] has no field for it. The diffusers repackaging promoted it to a
/// config key (`audio_vae/config.json`: `"num_attention_heads": 8`), and the two agree.
///
/// Pinned as a constant for the same reason [`crate::layout::PUBLISHED_GATED_FFN_LAYOUT`] is:
/// changing the head count changes `head_dim`, which changes the attention scale *and* the
/// adaptive pool's window layout, while every tensor shape stays identical — and, unlike a shape
/// mismatch, produces a runnable model with a wrong soundtrack conditioning.
///
/// **Bound to a published document, not to itself.**
/// [`MiniMaxH3AudioVaeConfig::cross_check_diffusers_json`] reads `audio_vae/config.json`'s
/// `num_attention_heads`, compares it to this constant, and refuses a checkpoint that disagrees —
/// including one that OMITS the key, which is refused rather than defaulted. That check runs
/// weights-free over the published config text in `crate::audio_config`'s
/// `diffusers_root_config_cross_checks`, which asserts the refusal in both directions (4 and 16),
/// so a comparison hardcoded to 8 on either side fails one of them; and against the real
/// snapshot's own `audio_vae/config.json` in `tests/real_weights.rs`.
///
/// Before that it was pinned only by `assert_eq!(ATTN_PROJ_HEADS, 8)` — a literal restating itself
/// — while every executed test built the encoder at the fixture's 2 heads. An earlier revision of
/// THIS doc claimed the cross-check already compared the key. It did not: `cross_check_diffusers_json`
/// never read `num_attention_heads`, and `audio_config.rs` was not in that commit. The comparison
/// exists now; this paragraph stays as the reason not to describe a binding before writing it.
pub const ATTN_PROJ_HEADS: usize = 8;

/// **The dtype the encode half is always built at**, whatever the provider's own store dtype is.
///
/// diffusers pins this whole component with `_keep_in_fp32_modules = ["encoder", …, "mean_proj",
/// "logs_proj"]` — the weight-normed convolutions and Snake activations degrade audibly under
/// bf16 — while `MiniMaxH3`'s own `dtype` is `BF16`, matching the DiT's block store. The plausible
/// regression is a tidy-up that replaces this with `self.dtype` at the one call site
/// (`crate::model::MiniMaxH3::load_audio_encoder`), which would encode every reference soundtrack
/// at the wrong precision with no diagnostic. That call site is source-scanned for exactly that.
pub const ENCODER_DTYPE: DType = DType::F32;

/// `eps` for every `nn.LayerNorm` in `pre_block` — torch's default, declared in no config file.
pub const LAYER_NORM_EPS: f64 = 1e-5;

/// `mlp_ratio` of the `pre_block` GeGLU MLP — `AttnProjection`'s default, which the audio VAE
/// never overrides, so the GeGLU hidden width is `latent_channels · 2` (64 on the shipped model).
///
/// **Enforced, not merely declared:** `GeGluMlp::from_weights` checks `w0`/`w1`/`w2` against it
/// and rejects a checkpoint at a different ratio. A constant nothing validates is a comment with
/// a type, and this crate has been bitten by exactly that (`crate::layout`).
pub const MLP_RATIO: usize = 2;

/// Re-fuse a `weight_norm` pair into a dense weight: `g · v / ‖v‖`, the norm reduced over every
/// axis but axis 0 (torch's `dim=0` default).
///
/// Duplicated from [`crate::audio_vae`]'s private helper — see the module docs.
fn fuse_weight_norm(g: &Tensor, v: &Tensor) -> Result<Tensor> {
    let shape = v.dims();
    if shape.len() != 3 {
        return Err(CandleError::Msg(format!(
            "minimax-h3 audio encoder: weight_v must be rank 3, got {shape:?}"
        )));
    }
    if g.dims() != [shape[0], 1, 1] {
        return Err(CandleError::Msg(format!(
            "minimax-h3 audio encoder: weight_g {:?} does not match weight_v {shape:?}",
            g.dims()
        )));
    }
    let norm = v.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
    Ok(v.broadcast_div(&norm)?.broadcast_mul(g)?)
}

/// A **strided** weight-normed `Conv1d` in candle's NCL layout.
///
/// Public so `tests/audio_vae_encode_parity.rs` can hold ONE convolution against the reference:
/// an end-to-end residual alone cannot say whether it is round-off accumulating or a defect.
#[derive(Debug, Clone)]
pub struct WnConv1d {
    /// `[out, in, kernel]` — torch's own layout, which is also candle's.
    weight: Tensor,
    bias: Tensor,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl WnConv1d {
    /// Load a `weight_norm`-parametrized conv (`weight_g` / `weight_v` / `bias`).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        stride: usize,
        padding: usize,
        dilation: usize,
        dtype: DType,
    ) -> Result<Self> {
        let g = w
            .require(&format!("{prefix}.weight_g"))?
            .to_dtype(DType::F32)?;
        let v = w
            .require(&format!("{prefix}.weight_v"))?
            .to_dtype(DType::F32)?;
        let fused = fuse_weight_norm(&g, &v)?;
        Ok(Self {
            weight: fused.to_dtype(dtype)?.contiguous()?,
            bias: w.require(&format!("{prefix}.bias"))?.to_dtype(dtype)?,
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

    /// `[B, C_in, T]` → `[B, C_out, T']`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y =
            x.contiguous()?
                .conv1d(&self.weight, self.padding, self.stride, self.dilation, 1)?;
        let c = y.dims()[1];
        Ok(y.broadcast_add(&self.bias.reshape((1, c, 1))?)?)
    }
}

/// Load a `Snake1d` (`x + (α + 1e-9)⁻¹·sin²(αx)`) as a [`SnakeBeta`] with `beta = alpha`.
///
/// The checkpoint stores `alpha` as `[1, C, 1]`; [`SnakeBeta::new`] flattens it, so the rank does
/// not have to be squeezed first. `logscale = false` — the encoder's Snake is **not** the decoder's
/// log-scaled SnakeBeta, and taking `exp(α)` here would be a plausible, silently wrong port.
fn load_snake1d(w: &Weights, prefix: &str, dtype: DType) -> Result<SnakeBeta> {
    let alpha = w.require(&format!("{prefix}.alpha"))?.to_dtype(dtype)?;
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
    pub fn from_weights(w: &Weights, prefix: &str, dilation: usize, dtype: DType) -> Result<Self> {
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

    /// `[B, C, T]` → `[B, C, T]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.act1.forward(x)?;
        let y = self.conv1.forward(&y)?;
        let y = self.act2.forward(&y)?;
        let y = self.conv2.forward(&y)?;
        // `pad = (x.len - y.len) // 2; if pad > 0: x = x[..., pad:-pad]`.
        let (xl, yl) = (x.dims()[2], y.dims()[2]);
        let pad = xl.saturating_sub(yl) / 2;
        let x = if pad > 0 {
            x.narrow(2, pad, xl - 2 * pad)?
        } else {
            x.clone()
        };
        if x.dims()[2] != yl {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio encoder: residual unit shortcut is {} long, its block emitted \
                 {yl}",
                x.dims()[2]
            )));
        }
        Ok((x + y)?)
    }
}

/// The dilations every `EncoderBlock` applies, in order. Hardcoded by the reference.
const RESIDUAL_DILATIONS: [usize; 3] = [1, 3, 9];

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

impl EncoderBlock {
    /// Load one downsampling stage from a checkpoint under `prefix`.
    pub fn from_weights(w: &Weights, prefix: &str, stride: usize, dtype: DType) -> Result<Self> {
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
                stride.div_ceil(2),
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

    /// `[B, C/2, T]` → `[B, C, T/stride]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
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
        w: &Weights,
        prefix: &str,
        cfg: &MiniMaxH3AudioVaeConfig,
        dtype: DType,
    ) -> Result<Self> {
        let conv_in = WnConv1d::from_weights(w, &format!("{prefix}.block.0"), 1, 3, 1, dtype)?;
        let mut blocks = Vec::with_capacity(cfg.encoder_rates.len());
        for (i, &stride) in cfg.encoder_rates.iter().enumerate() {
            if stride == 0 {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 audio encoder: encoder_rates[{i}] = 0 must be positive"
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

    /// `[B, 1, S]` → `[B, latent_dim, S/hop]`, both NCL.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
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
pub fn adaptive_avg_pool_last_axis(x: &Tensor, out: usize) -> Result<Tensor> {
    let rank = x.dims().len();
    if rank == 0 {
        return Err(CandleError::Msg(
            "minimax-h3 adaptive pool: cannot pool a zero-rank tensor".into(),
        ));
    }
    let axis = rank - 1;
    let len = x.dims()[axis];
    if out == 0 || len == 0 {
        return Err(CandleError::Msg(format!(
            "minimax-h3 adaptive pool: {len} -> {out} is not a valid pooling"
        )));
    }
    let (len64, out64) = (len as u64, out as u64);
    let mut parts = Vec::with_capacity(out);
    for i in 0..out64 {
        let start = i * len64 / out64;
        let end = ((i + 1) * len64).div_ceil(out64);
        if start >= end || end > len64 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 adaptive pool: window {i} of {out} is [{start}, {end}) over {len}"
            )));
        }
        let window = x.narrow(axis, start as usize, (end - start) as usize)?;
        parts.push(window.mean_keepdim(axis)?);
    }
    Ok(Tensor::cat(&parts, axis)?.contiguous()?)
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
    qkv_w: Tensor,
    /// `cat(q_bias, zero_k_bias, v_bias)`, assembled at load.
    qkv_b: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
    heads: usize,
    head_dim: usize,
    out_dim: usize,
    scale: f64,
}

impl CausalAttention {
    /// Load from a checkpoint under `prefix` (`"pre_block.attn"` in the published naming).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        heads: usize,
        dtype: DType,
    ) -> Result<Self> {
        if heads == 0 || !in_dim.is_multiple_of(heads) {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio encoder: {in_dim} attention channels do not divide into {heads} \
                 heads"
            )));
        }
        let head_dim = in_dim / heads;
        let qkv_w = w
            .require(&format!("{prefix}.qkv.weight"))?
            .to_dtype(dtype)?;
        if qkv_w.dims() != [3 * in_dim, in_dim] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio encoder: {prefix}.qkv.weight is {:?}, expected [{}, {in_dim}]",
                qkv_w.dims(),
                3 * in_dim
            )));
        }
        // The `nn.Linear` is bias-less; the reference builds the effective bias from three stored
        // tensors, of which the middle one is a ZERO BUFFER the checkpoint actually ships. Read
        // and concatenated rather than assumed, so the tensor is genuinely consumed.
        let q_b = w.require(&format!("{prefix}.q_bias"))?.to_dtype(dtype)?;
        let k_b = w
            .require(&format!("{prefix}.zero_k_bias"))?
            .to_dtype(dtype)?;
        let v_b = w.require(&format!("{prefix}.v_bias"))?.to_dtype(dtype)?;
        let qkv_b = Tensor::cat(&[&q_b, &k_b, &v_b], 0)?.contiguous()?;
        if qkv_b.dims() != [3 * in_dim] {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio encoder: {prefix} q/k/v biases assemble to {:?}, expected [{}]",
                qkv_b.dims(),
                3 * in_dim
            )));
        }
        Ok(Self {
            qkv_w,
            qkv_b,
            proj_w: w
                .require(&format!("{prefix}.proj.weight"))?
                .to_dtype(dtype)?,
            proj_b: w.require(&format!("{prefix}.proj.bias"))?.to_dtype(dtype)?,
            heads,
            head_dim,
            out_dim,
            // `self.scale = head_dim ** -0.5`, matching torch SDPA's default.
            scale: (head_dim as f64).powf(-0.5),
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
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let s = x.dims();
        if s.len() != 3 || s[2] != self.heads * self.head_dim {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio encoder attention: expected [B, N, {}], got {s:?}",
                self.heads * self.head_dim
            )));
        }
        let (b, n) = (s[0], s[1]);
        let qkv = linear(x, &self.qkv_w, &self.qkv_b)?;
        // `[B, N, 3·D] → [B, N, 3, H, Dh]`: the THIRDS are the outer split and the heads the
        // inner one. Reading it per-head-interleaved instead is shape-identical and wrong.
        let qkv = qkv.reshape((b, n, 3, self.heads, self.head_dim))?;
        let mut parts = Vec::with_capacity(3);
        for i in 0..3 {
            parts.push(
                qkv.narrow(2, i, 1)?
                    .reshape((b, n, self.heads, self.head_dim))?
                    // [B, N, H, Dh] → [B, H, N, Dh]
                    .permute((0, 2, 1, 3))?
                    .contiguous()?,
            );
        }
        // `is_causal=True` — every query attends to its own position and earlier ones only.
        let mask = causal_mask(b, n, x.device())?.to_dtype(parts[0].dtype())?;
        let attended = candle_gen::sdpa_budgeted_bhsd(
            &parts[0],
            &parts[1],
            &parts[2],
            self.scale,
            Some(&mask),
            softmax_last_dim,
            candle_gen::ATTN_SCORES_BUDGET,
        )?;
        // Mean-pool the HEADS away: `[B, H, N, Dh]` → `[B, N, Dh]`.
        let pooled = attended.mean(1)?;
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
    norm_w: Tensor,
    norm_b: Tensor,
    w0: (Tensor, Tensor),
    w1: (Tensor, Tensor),
    w2: (Tensor, Tensor),
}

fn load_linear(w: &Weights, prefix: &str, dtype: DType) -> Result<(Tensor, Tensor)> {
    Ok((
        w.require(&format!("{prefix}.weight"))?.to_dtype(dtype)?,
        w.require(&format!("{prefix}.bias"))?.to_dtype(dtype)?,
    ))
}

fn linear_names(prefix: &str) -> [String; 2] {
    [format!("{prefix}.weight"), format!("{prefix}.bias")]
}

impl GeGluMlp {
    fn from_weights(w: &Weights, prefix: &str, dim: usize, dtype: DType) -> Result<Self> {
        let (norm_w, norm_b) = load_linear(w, &format!("{prefix}.norm"), dtype)?;
        let hidden = dim.checked_mul(MLP_RATIO).ok_or_else(|| {
            CandleError::Msg(format!(
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
            if weight.dims() != want {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 audio encoder: {prefix}.{name}.weight is {:?}, expected {want:?} \
                     at mlp_ratio {MLP_RATIO}",
                    weight.dims()
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

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = layer_norm(x, &self.norm_w, &self.norm_b, LAYER_NORM_EPS)?;
        // candle's `gelu` IS the tanh approximation (`gelu_erf` is the exact one), matching the
        // reference's `nn.GELU(approximate="tanh")`.
        let gate = linear(&h, &self.w0.0, &self.w0.1)?.gelu()?;
        let value = linear(&h, &self.w1.0, &self.w1.1)?;
        let h = (gate * value)?;
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
    norm1: (Tensor, Tensor),
    norm2: (Tensor, Tensor),
    norm3: (Tensor, Tensor),
    attn: CausalAttention,
    proj: (Tensor, Tensor),
    mlp: GeGluMlp,
}

impl AttnProjection {
    /// Load from a checkpoint under `prefix` (`"pre_block"` in the published naming).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        in_dim: usize,
        out_dim: usize,
        heads: usize,
        dtype: DType,
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

    /// The attention branch alone — `attn(norm1(x))`, without the residual that would hide it.
    ///
    /// Exposed for the parity suite: the sum in [`Self::forward`] is dominated by the `proj`
    /// skip, so a wrong attention branch lands inside a right-looking total.
    pub fn attention_branch(&self, x: &Tensor) -> Result<Tensor> {
        self.attn.forward(&layer_norm(
            x,
            &self.norm1.0,
            &self.norm1.1,
            LAYER_NORM_EPS,
        )?)
    }

    /// `[B, N, in_dim]` → `[B, N, out_dim]`, NLC.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let skip = linear(
            &layer_norm(x, &self.norm3.0, &self.norm3.1, LAYER_NORM_EPS)?,
            &self.proj.0,
            &self.proj.1,
        )?;
        let attended = self.attention_branch(x)?;
        let h = (skip + attended)?;
        let mlp = self.mlp.forward(&layer_norm(
            &h,
            &self.norm2.0,
            &self.norm2.1,
            LAYER_NORM_EPS,
        )?)?;
        Ok((h + mlp)?)
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
    mean: Tensor,
    logs: Tensor,
    std: Tensor,
}

impl AudioDiagonalGaussian {
    /// Build from the two head outputs. `logs` is the log **standard deviation**.
    pub fn new(mean: Tensor, logs: Tensor) -> Result<Self> {
        if mean.dims() != logs.dims() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio encoder: posterior mean {:?} and logs {:?} differ in shape",
                mean.dims(),
                logs.dims()
            )));
        }
        let std = logs.exp()?;
        Ok(Self { mean, logs, std })
    }

    /// The posterior mean — and `mode()`, which is the ONLY thing the MiniMax-H3 pipeline uses.
    pub fn mean(&self) -> &Tensor {
        &self.mean
    }

    /// The raw `logs_proj` output: log **standard deviation**.
    pub fn logs(&self) -> &Tensor {
        &self.logs
    }

    /// `exp(logs)` — the per-element standard deviation.
    pub fn std(&self) -> &Tensor {
        &self.std
    }

    /// `mode()`: bit-for-bit `mean_proj`'s output, as the reference documents.
    pub fn mode(&self) -> &Tensor {
        &self.mean
    }

    /// `mean + std · noise`. `noise` must match the mean's shape.
    pub fn sample_with(&self, noise: &Tensor) -> Result<Tensor> {
        if noise.dims() != self.mean.dims() {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio encoder: posterior noise {:?} does not match the mean {:?}",
                noise.dims(),
                self.mean.dims()
            )));
        }
        Ok((&self.mean + (&self.std * noise.to_dtype(self.std.dtype())?)?)?)
    }
}

/// The MiniMax-H3 audio VAE's **encode** half.
#[derive(Debug, Clone)]
pub struct MiniMaxH3AudioVaeEncoder {
    encoder: DacEncoder,
    pre_block: AttnProjection,
    mean_w: Tensor,
    mean_b: Tensor,
    logs_w: Tensor,
    logs_b: Tensor,
    cfg: MiniMaxH3AudioVaeConfig,
    latents_mean: Tensor,
    latents_std: Tensor,
    hop: usize,
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
        w: &Weights,
        cfg: &MiniMaxH3AudioVaeConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        Self::from_weights_with_heads(w, cfg, ATTN_PROJ_HEADS, device, dtype)
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
        w: &Weights,
        cfg: &MiniMaxH3AudioVaeConfig,
        heads: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        if !cfg.attn_proj {
            return Err(CandleError::Msg(
                "minimax-h3 audio encoder: attn_proj = false has no pre_block, so the encoder \
                 trunk cannot reach the posterior heads (the published checkpoint sets it true)"
                    .into(),
            ));
        }
        let latent_dim = cfg.bigvgan.num_mels;
        let channels = cfg.latent_channels;
        if channels == 0 || latent_dim == 0 || !latent_dim.is_multiple_of(channels) {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio encoder: latent_dim {latent_dim} is not a multiple of \
                 latent_channels {channels}; the reference would widen attn_proj_dim to the next \
                 power of two, which diffusers rejects"
            )));
        }
        let hop = hop_length(&cfg.encoder_rates)?;

        let encoder = DacEncoder::from_weights(w, "encoder", cfg, dtype)?;
        let pre_block =
            AttnProjection::from_weights(w, "pre_block", latent_dim, channels, heads, dtype)?;

        let head = |name: &str| -> Result<(Tensor, Tensor)> {
            let weight = w.require(&format!("{name}.weight"))?.to_dtype(dtype)?;
            if weight.dims() != [channels, channels, 1] {
                return Err(CandleError::Msg(format!(
                    "minimax-h3 audio encoder: {name}.weight is {:?}, expected \
                     [{channels}, {channels}, 1]",
                    weight.dims()
                )));
            }
            // A 1x1 conv is a pointwise linear: squeeze the kernel axis rather than convolve it.
            Ok((
                weight.reshape((channels, channels))?,
                w.require(&format!("{name}.bias"))?.to_dtype(dtype)?,
            ))
        };
        let (mean_w, mean_b) = head("mean_proj")?;
        let (logs_w, logs_b) = head("logs_proj")?;

        if cfg.latents_mean.len() != channels || cfg.latents_std.len() != channels {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio encoder: latents_mean/std carry {}/{} entries for {channels} \
                 latent channels",
                cfg.latents_mean.len(),
                cfg.latents_std.len()
            )));
        }
        let latents_mean = Tensor::from_vec(cfg.latents_mean.clone(), (1, 1, channels), device)?
            .to_dtype(dtype)?;
        let latents_std =
            Tensor::from_vec(cfg.latents_std.clone(), (1, 1, channels), device)?.to_dtype(dtype)?;

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

    /// The `pre_block` projection, for the parity suite's isolated comparison.
    pub fn pre_block(&self) -> &AttnProjection {
        &self.pre_block
    }

    /// The convolutional trunk, for the parity suite's isolated comparison.
    pub fn trunk(&self) -> &DacEncoder {
        &self.encoder
    }

    /// `∏ encoder_rates` — 800, i.e. 40 latents/s at 32 kHz.
    pub fn hop_length(&self) -> usize {
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
    pub fn encode(&self, waveform: &Tensor) -> Result<AudioDiagonalGaussian> {
        let s = waveform.dims();
        if s.len() != 3 || s[1] != 1 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio encode: expected [B, 1, samples] (stereo is B = 2), got {s:?}"
            )));
        }
        let samples = s[2];
        if samples == 0 {
            return Err(CandleError::Msg(
                "minimax-h3 audio encode: an empty waveform has no latents".into(),
            ));
        }
        // `right_pad = ceil(S / hop) * hop - S`, zeros.
        let frames = samples.div_ceil(self.hop);
        let padded_len = frames * self.hop;
        let mut x = waveform.clone();
        if padded_len > samples {
            let pad = Tensor::zeros(
                (s[0], 1, padded_len - samples),
                x.dtype(),
                waveform.device(),
            )?;
            x = Tensor::cat(&[&x, &pad], 2)?.contiguous()?;
        }

        let hidden = self.encoder.forward(&x)?;
        let got = hidden.dims()[2];
        if got != frames {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio encode: {padded_len} samples produced {got} frames, expected \
                 {frames} at hop {}",
                self.hop
            )));
        }
        // NCL -> NLC for the sequence block, then back for the posterior.
        let hidden = self
            .pre_block
            .forward(&hidden.permute((0, 2, 1))?.contiguous()?)?;
        let mean = linear(&hidden, &self.mean_w, &self.mean_b)?;
        let logs = linear(&hidden, &self.logs_w, &self.logs_b)?;
        AudioDiagonalGaussian::new(
            mean.permute((0, 2, 1))?.contiguous()?,
            logs.permute((0, 2, 1))?.contiguous()?,
        )
    }

    /// Per-channel latent normalization, `(z − latents_mean) / latents_std`.
    ///
    /// The exact inverse of [`crate::audio_vae::MiniMaxH3AudioVae::denormalize`], and applied
    /// over the same axis: the **channel** axis is the second-to-last, so this works unchanged
    /// for the mono `[B, C, T]` this encoder emits and for the stereo `[B, 2, C, T]` packing the
    /// decode half takes. `encode` deliberately does not apply it, so it stays a byte-for-byte
    /// analogue of the reference's `encode`; the pipeline normalizes afterwards.
    pub fn normalize(&self, z: &Tensor) -> Result<Tensor> {
        let rank = z.dims().len();
        if rank < 2 {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio encoder: cannot normalize a rank-{rank} latent"
            )));
        }
        let channels = z.dims()[rank - 2];
        if channels != self.cfg.latent_channels {
            return Err(CandleError::Msg(format!(
                "minimax-h3 audio encoder: latent has {channels} channels, config declares {}",
                self.cfg.latent_channels
            )));
        }
        let mut shape = vec![1usize; rank];
        shape[rank - 2] = channels;
        let mean = self
            .latents_mean
            .reshape(shape.clone())?
            .to_dtype(z.dtype())?;
        let std = self.latents_std.reshape(shape)?.to_dtype(z.dtype())?;
        Ok(z.broadcast_sub(&mean)?.broadcast_div(&std)?)
    }
}

/// `∏ encoder_rates`, checked for overflow and for a degenerate zero rate.
fn hop_length(rates: &[usize]) -> Result<usize> {
    if rates.is_empty() {
        return Err(CandleError::Msg(
            "minimax-h3 audio encoder: encoder_rates is empty, so the hop length is undefined"
                .into(),
        ));
    }
    let mut hop: usize = 1;
    for &rate in rates {
        if rate == 0 {
            return Err(CandleError::Msg(
                "minimax-h3 audio encoder: an encoder rate of 0 is not a downsampling".into(),
            ));
        }
        hop = hop.checked_mul(rate).ok_or_else(|| {
            CandleError::Msg("minimax-h3 audio encoder: encoder_rates overflow".to_owned())
        })?;
    }
    Ok(hop)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// The weight-norm fusion is `g · v / ‖v‖` with the norm over every axis **but the first** —
    /// the same rule [`crate::audio_vae`] applies, restated where the strided conv needs it.
    ///
    /// Written against a `v` whose two output rows have different norms, so a reduction over the
    /// wrong axes cannot pass.
    #[test]
    fn weight_norm_reduces_over_every_axis_but_the_output_channel() {
        let dev = Device::Cpu;
        // row 0 = [3, 4] (‖·‖ = 5), row 1 = [0, 1] (‖·‖ = 1).
        let v = Tensor::from_vec(vec![3.0f32, 4.0, 0.0, 1.0], (2, 1, 2), &dev).unwrap();
        let g = Tensor::from_vec(vec![10.0f32, 2.0], (2, 1, 1), &dev).unwrap();
        let fused = fuse_weight_norm(&g, &v).unwrap();
        assert_eq!(flat(&fused), vec![6.0, 8.0, 0.0, 2.0]);

        // A rank-2 `v` or a mismatched `g` is a typed error, not a broadcast that happens to run.
        let flat_v = Tensor::from_vec(vec![1.0f32, 2.0], (2, 1), &dev).unwrap();
        assert!(fuse_weight_norm(&g, &flat_v).is_err());
        let wrong_g = Tensor::from_vec(vec![1.0f32], (1, 1, 1), &dev).unwrap();
        assert!(fuse_weight_norm(&wrong_g, &v).is_err());
    }

    /// `adaptive_avg_pool1d`'s **overlapping** regime, which a `reshape().mean()` cannot reproduce.
    ///
    /// 5 → 3 gives windows `[0,2)`, `[1,4)`, `[3,5)`: the second one overlaps both neighbours.
    /// Values are `1..5`, so the three means are 1.5, 3.0 and 4.5 — all different from the
    /// disjoint-tiling answer a reshape would give.
    #[test]
    fn the_adaptive_pool_reproduces_torchs_overlapping_windows() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], (1, 1, 5), &dev).unwrap();
        let y = adaptive_avg_pool_last_axis(&x, 3).unwrap();
        assert_eq!(y.dims(), [1, 1, 3]);
        assert_eq!(flat(&y), vec![1.5, 3.0, 4.5]);

        // The exact-divisor regime (the only one the shipped 256 -> 32 reaches) still tiles.
        let y = adaptive_avg_pool_last_axis(&x, 5).unwrap();
        assert_eq!(flat(&y), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let x4 = Tensor::from_vec(vec![1.0f32, 3.0, 5.0, 9.0], (1, 1, 4), &dev).unwrap();
        assert_eq!(
            flat(&adaptive_avg_pool_last_axis(&x4, 2).unwrap()),
            vec![2.0, 7.0]
        );

        assert!(adaptive_avg_pool_last_axis(&x, 0).is_err());
    }

    /// The hop is the product of the rates, and a zero rate is refused rather than collapsing the
    /// hop to zero (which would divide by zero when the pad length is computed).
    #[test]
    fn the_hop_is_the_product_of_the_encoder_rates() {
        assert_eq!(hop_length(&[2, 4, 4, 5, 5]).unwrap(), 800);
        assert_eq!(hop_length(&[2, 5]).unwrap(), 10);
        assert!(hop_length(&[]).is_err());
        assert!(hop_length(&[2, 0, 5]).is_err());
    }

    /// The posterior's `σ` is `exp(logs)` — a log **standard deviation**, not the video encoder's
    /// log **variance**, and there is no clamp.
    ///
    /// The two are indistinguishable through `mode()`, which is all the render path reads, so this
    /// asserts on `std()` directly.
    #[test]
    fn log_std_is_not_log_variance() {
        let dev = Device::Cpu;
        let mean = Tensor::from_vec(vec![0.0f32, 0.0], (1, 2, 1), &dev).unwrap();
        let logs = Tensor::from_vec(vec![1.0f32, -40.0], (1, 2, 1), &dev).unwrap();
        let p = AudioDiagonalGaussian::new(mean, logs).unwrap();
        let std = flat(p.std());
        // exp(1) = 2.71828…; the log-VARIANCE reading would be exp(0.5) = 1.6487.
        assert!((std[0] - std::f32::consts::E).abs() < 1e-5, "{std:?}");
        // …and no `[-30, 20]` clamp: exp(-40) underflows towards zero rather than being pinned to
        // exp(-30) = 9.36e-14.
        assert!(
            std[1] < 1e-16,
            "the audio posterior must NOT clamp: {std:?}"
        );

        // `mode()` is the mean verbatim, which is why nothing downstream can see the difference.
        assert_eq!(flat(p.mode()), flat(p.mean()));

        let bad = Tensor::from_vec(vec![0.0f32], (1, 1, 1), &dev).unwrap();
        assert!(
            AudioDiagonalGaussian::new(bad.clone(), p.logs().clone()).is_err(),
            "a mean and logs of different shapes must not assemble"
        );
        assert!(p.sample_with(&bad).is_err());
    }

    /// The declared constants, pinned. None of them is in the FL2VA source triple, so nothing else
    /// in the crate can catch a drift.
    ///
    /// These are literals restating themselves and they cannot be anything else: a constant that
    /// appears in no document has nothing to be compared TO, and every executed test builds the
    /// encoder at the fixture's 2 heads / small dims rather than at these. That is exactly why
    /// [`ATTN_PROJ_HEADS`] is no longer in this list — the diffusers repackaging publishes it as
    /// `audio_vae/config.json`'s `num_attention_heads`, so it is bound to that document by
    /// [`MiniMaxH3AudioVaeConfig::cross_check_diffusers_json`] and asserted against a real snapshot
    /// in `tests/real_weights.rs`, which is a strictly stronger pin than restating `8` here.
    #[test]
    fn the_declared_constants_are_the_references() {
        assert_eq!(MLP_RATIO, 2);
        assert_eq!(LAYER_NORM_EPS, 1e-5);
        assert_eq!(RESIDUAL_DILATIONS, [1, 3, 9]);
    }

    /// The tensor-name inventory is exhaustive and disjoint from the decode half's.
    ///
    /// Both halves are generated from the same config, so this is the check that a renamed or
    /// dropped leaf shows up as a name the other half also does not claim — the partition that
    /// `tests/audio_vae_encode_parity.rs` then asserts against the real 1087-tensor checkpoint.
    #[test]
    fn the_encode_inventory_is_disjoint_from_the_decode_inventory() {
        use std::collections::BTreeSet;
        let cfg = MiniMaxH3AudioVaeConfig::default();
        let encode: BTreeSet<String> = MiniMaxH3AudioVaeEncoder::tensor_names(&cfg)
            .into_iter()
            .collect();
        let decode: BTreeSet<String> = crate::audio_vae::MiniMaxH3AudioVae::tensor_names(&cfg)
            .into_iter()
            .collect();
        assert!(!encode.is_empty() && !decode.is_empty());
        assert!(
            encode.is_disjoint(&decode),
            "the two halves must claim disjoint tensors, shared: {:?}",
            encode.intersection(&decode).collect::<Vec<_>>()
        );
        // Every encode name is under one of the four encode roots.
        for name in &encode {
            assert!(
                name.starts_with("encoder.")
                    || name.starts_with("pre_block.")
                    || name.starts_with("mean_proj.")
                    || name.starts_with("logs_proj."),
                "unexpected encode tensor {name}"
            );
        }
        // The `zero_k_bias` buffer is genuinely consumed rather than assumed zero.
        assert!(encode.contains("pre_block.attn.zero_k_bias"));
    }
}
