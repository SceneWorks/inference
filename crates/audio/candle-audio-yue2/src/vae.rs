//! Native FP32 port of the YuE2 Oobleck VAE (sc-22993) — `yue2/modeling_vae.py` at the pinned
//! commit, for **both** published decoders (`m-a-p/YuE2-Vae`, the standard listening decoder, and
//! `m-a-p/YuE2-Vae-legacy`, the benchmark decoder). The two releases share this architecture and
//! config and differ in `release_variant` and in their **decoder** weights only: all 218 encoder
//! tensors are identical between the two pinned files (per the committed conversion manifests),
//! so latents are decoder-agnostic and either decoder can decode any cached latent.
//!
//! ```text
//! decoder  z [B, 64, T]
//!   layers.0  WNConv1d(64 → ch·32, k7, p3)
//!   layers.1–6 DecoderBlock(stride 6, 5, 4, 4, 2, 2):
//!                SnakeBeta → WNConvTranspose1d(k 2s, s, p ⌈s/2⌉, no output padding)
//!                → ResidualUnit(dil 1) → ResidualUnit(dil 3) → ResidualUnit(dil 9)
//!   layers.7  SnakeBeta → layers.8 WNConv1d(ch → 2, k7, p3, no bias)    (no final tanh)
//!   → waveform [B, 2, 1920·T − 64] at 48 kHz, unclipped
//! encoder  audio [B, 2, S]
//!   layers.0  WNConv1d(2 → ch, k7, p3)
//!   layers.1–6 EncoderBlock(stride 2, 2, 4, 4, 5, 6):
//!                3 × ResidualUnit(dil 1/3/9) → SnakeBeta → WNConv1d(k 2s, s, p ⌈s/2⌉)
//!   layers.7  SnakeBeta → layers.8 WNConv1d(ch·32 → 128, k3, p1)
//!   → (mean, scale) = split(128 → 64 + 64); stdev = softplus(scale) + 1e-4
//! ResidualUnit(d) = x + WNConv1d(k1)(SnakeBeta(WNConv1d(k7, dil d, p 3d)(SnakeBeta(x))))
//! SnakeBeta(x)    = x + (exp(β) + 1e-9)⁻¹ · sin²(exp(α) · x)          (log-scale α, β)
//! ```
//!
//! Precision (epic E8): the released VAE is FP32-only — upstream refuses any other dtype and any
//! non-F32 tensor — and so does this port: weights are loaded, weight-norm-folded and run in FP32,
//! and a non-F32 tensor is a load error. There is no reduced-precision VAE path.
//!
//! Chunked decoding ([`Yue2Vae::decode_tiled`]) is upstream's exact-boundary halo/crop scheme:
//! every tile carries at least [`Yue2Vae::required_halo`] latent frames of context on each side
//! (derived from the decoder's actual layers by upstream's audited dependency-interval rule), and
//! only each tile's core is kept — no crossfade, no smoothing, no zero padding of the song. The
//! full decode ([`Yue2Vae::decode_full`]) is retained as the reference FP32 path for fidelity
//! validation; the two agree to FP32 rounding.
//!
//! Loading goes through [`Yue2Vae::load`], which only accepts a [`VerifiedComponent`] for one of
//! the two VAE components and loads only its verified `config.json` / `model.safetensors`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use candle_audio::candle_core::{self, DType, Device, Tensor, D};
use candle_audio::neural_codec::fold_weight_norm;
use serde_json::{json, Value};

use crate::inventory::{ComponentId, VaeVariant};
use crate::latent::{sha256_f32, AcousticLatents, LatentError, LatentSource, LATENT_CHANNELS};
use crate::snapshot::{AssetError, VerifiedComponent};

/// Output sample rate of both released decoders.
pub const SAMPLE_RATE: u32 = 48_000;
/// Output channels (stereo; channel 0 is left, channel 1 is right).
pub const AUDIO_CHANNELS: usize = 2;
/// Waveform samples per latent frame (the product of the decoder strides).
pub const DOWNSAMPLING_RATIO: usize = 1920;
/// Upstream's decode core size (`decode_core_frames` in both configs; the pipeline uses 512 under
/// a ≤12 GiB memory budget).
pub const DEFAULT_CORE_FRAMES: usize = 1024;
/// Upstream's halo (`decode_halo_frames` in both configs, and the pipeline's fixed value).
pub const DEFAULT_HALO_FRAMES: usize = 16;

/// `SnakeBeta`'s division guard (`no_div_by_zero`).
const SNAKE_EPS: f64 = 1e-9;
/// The posterior standard deviation floor (`softplus(scale) + 1e-4`).
const STDEV_FLOOR: f64 = 1e-4;
/// torch `softplus` switches to the identity above this input (`threshold=20`).
const SOFTPLUS_THRESHOLD: f64 = 20.0;

/// Every way loading or running the VAE fails.
#[derive(Debug, thiserror::Error)]
pub enum VaeError {
    /// Asset resolution / verification.
    #[error(transparent)]
    Asset(#[from] AssetError),
    /// The config is not one this port supports (the released decoders' options).
    #[error("YuE2 VAE config: {0}")]
    Config(String),
    /// The weights file does not hold exactly the expected FP32 tensors.
    #[error("YuE2 VAE weights: {0}")]
    Weights(String),
    /// A decode/encode input is invalid (shape, emptiness, non-finite values, halo too small).
    #[error("YuE2 VAE input: {0}")]
    Input(String),
    /// The decoder produced a non-finite sample (upstream: "VAE produced non-finite audio").
    #[error("YuE2 VAE produced non-finite audio at channel {channel}, sample {sample}")]
    NonFiniteAudio {
        /// The channel.
        channel: usize,
        /// The sample index.
        sample: usize,
    },
    /// The latents failed identity verification.
    #[error(transparent)]
    Latent(#[from] LatentError),
    /// The caller cancelled between tiles.
    #[error("YuE2 VAE decode cancelled after {completed} of {total} tiles")]
    Cancelled {
        /// Tiles already decoded.
        completed: usize,
        /// Total tiles.
        total: usize,
    },
    /// A tensor operation failed.
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
}

type Result<T> = std::result::Result<T, VaeError>;

/// One Oobleck stack's shape (`encoder_config` / `decoder_config`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OobleckConfig {
    /// Audio channels on the waveform side (`in_channels` / `out_channels`).
    pub audio_channels: usize,
    /// Base width.
    pub channels: usize,
    /// Channels on the latent side (encoder: 2 × latent width, mean + scale).
    pub latent_dim: usize,
    /// Width multipliers per stage.
    pub c_mults: Vec<usize>,
    /// Stride per stage (encoder order; the decoder walks them in reverse).
    pub strides: Vec<usize>,
}

/// The parsed, validated `config.json` of a YuE2 VAE snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VaeConfig {
    /// `release_variant`: which published decoder this is.
    pub release_variant: VaeVariant,
    /// `sample_rate` (48000).
    pub sample_rate: u32,
    /// `latent_dim` (64).
    pub latent_dim: usize,
    /// `downsampling_ratio` (1920).
    pub downsampling_ratio: usize,
    /// `audio_channels` (2).
    pub audio_channels: usize,
    /// `decode_core_frames`.
    pub decode_core_frames: usize,
    /// `decode_halo_frames`.
    pub decode_halo_frames: usize,
    /// The encoder stack.
    pub encoder: OobleckConfig,
    /// The decoder stack.
    pub decoder: OobleckConfig,
}

fn field<'a>(v: &'a Value, key: &str, ctx: &str) -> Result<&'a Value> {
    v.get(key)
        .ok_or_else(|| VaeError::Config(format!("{ctx}: missing `{key}`")))
}

fn as_usize(v: &Value, key: &str, ctx: &str) -> Result<usize> {
    field(v, key, ctx)?
        .as_u64()
        .map(|n| n as usize)
        .ok_or_else(|| VaeError::Config(format!("{ctx}: `{key}` is not an unsigned integer")))
}

fn usize_list(v: &Value, key: &str, ctx: &str) -> Result<Vec<usize>> {
    field(v, key, ctx)?
        .as_array()
        .and_then(|a| {
            a.iter()
                .map(|x| x.as_u64().map(|n| n as usize))
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| VaeError::Config(format!("{ctx}: `{key}` is not a list of integers")))
}

/// A boolean with upstream's constructor default when absent.
fn flag(v: &Value, key: &str, default: bool, ctx: &str) -> Result<bool> {
    match v.get(key) {
        None => Ok(default),
        Some(b) => b
            .as_bool()
            .ok_or_else(|| VaeError::Config(format!("{ctx}: `{key}` is not a boolean"))),
    }
}

fn require(ok: bool, what: impl FnOnce() -> String) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(VaeError::Config(what()))
    }
}

impl VaeConfig {
    /// Parse a `config.json` body. Options the released decoders do not use (ELU activations,
    /// anti-aliasing, nearest upsampling, filters, a final tanh, non-vanilla Snake) are refused
    /// with the same message class as upstream rather than silently mis-decoded; absent optional
    /// keys take upstream's constructor defaults (e.g. `final_tanh` defaults to `true`, so a config
    /// that omits it is refused).
    pub fn parse(text: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(text)
            .map_err(|e| VaeError::Config(format!("config.json is not JSON: {e}")))?;
        if v.get("model_type").and_then(Value::as_str) != Some("yue2_vae") {
            return Err(VaeError::Config(format!(
                "model_type {:?}, expected \"yue2_vae\"",
                v.get("model_type")
            )));
        }
        let release_variant = match v.get("release_variant").and_then(Value::as_str) {
            Some("standard") => VaeVariant::Standard,
            Some("legacy") => VaeVariant::Legacy,
            other => {
                return Err(VaeError::Config(format!(
                    "release_variant {other:?}, expected \"standard\" or \"legacy\""
                )))
            }
        };
        let ctx = "config";
        let cfg = Self {
            release_variant,
            sample_rate: as_usize(&v, "sample_rate", ctx)? as u32,
            latent_dim: as_usize(&v, "latent_dim", ctx)?,
            downsampling_ratio: as_usize(&v, "downsampling_ratio", ctx)?,
            audio_channels: as_usize(&v, "audio_channels", ctx)?,
            decode_core_frames: as_usize(&v, "decode_core_frames", ctx)?,
            decode_halo_frames: as_usize(&v, "decode_halo_frames", ctx)?,
            encoder: Self::stack(field(&v, "encoder_config", ctx)?, "encoder_config", true)?,
            decoder: Self::stack(field(&v, "decoder_config", ctx)?, "decoder_config", false)?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn stack(v: &Value, ctx: &str, encoder: bool) -> Result<OobleckConfig> {
        require(flag(v, "use_snake", false, ctx)?, || {
            format!("{ctx}: use_snake=false (ELU) is not a released YuE2 VAE option")
        })?;
        require(!flag(v, "antialias_activation", false, ctx)?, || {
            format!("{ctx}: antialias_activation is not a released YuE2 VAE option")
        })?;
        if !encoder {
            for key in ["use_nearest_upsample", "use_filter"] {
                require(!flag(v, key, false, ctx)?, || {
                    format!("{ctx}: {key} is not a released YuE2 VAE option")
                })?;
            }
            require(!flag(v, "final_tanh", true, ctx)?, || {
                format!("{ctx}: final_tanh (default true) is not a released YuE2 VAE option")
            })?;
            let snake = v
                .get("snake_type")
                .map(|s| s.as_str().unwrap_or("<non-string>"))
                .unwrap_or("vanilla");
            require(snake == "vanilla", || {
                format!("{ctx}: snake_type {snake:?}; the released decoder uses vanilla SnakeBeta")
            })?;
        }
        let io_key = if encoder {
            "in_channels"
        } else {
            "out_channels"
        };
        let stack = OobleckConfig {
            audio_channels: as_usize(v, io_key, ctx)?,
            channels: as_usize(v, "channels", ctx)?,
            latent_dim: as_usize(v, "latent_dim", ctx)?,
            c_mults: usize_list(v, "c_mults", ctx)?,
            strides: usize_list(v, "strides", ctx)?,
        };
        require(
            !stack.c_mults.is_empty()
                && stack.c_mults.len() == stack.strides.len()
                && stack.c_mults.iter().chain(&stack.strides).all(|&n| n > 0)
                && stack.channels > 0,
            || format!("{ctx}: c_mults/strides must be equal-length positive lists"),
        )?;
        Ok(stack)
    }

    fn validate(&self) -> Result<()> {
        require(self.sample_rate == SAMPLE_RATE, || {
            format!("sample_rate {}, expected {SAMPLE_RATE}", self.sample_rate)
        })?;
        require(self.audio_channels == AUDIO_CHANNELS, || {
            format!(
                "audio_channels {}, expected {AUDIO_CHANNELS}",
                self.audio_channels
            )
        })?;
        require(self.latent_dim == LATENT_CHANNELS, || {
            format!("latent_dim {}, expected {LATENT_CHANNELS}", self.latent_dim)
        })?;
        require(self.downsampling_ratio == DOWNSAMPLING_RATIO, || {
            format!(
                "downsampling_ratio {}, expected {DOWNSAMPLING_RATIO}",
                self.downsampling_ratio
            )
        })?;
        require(
            self.decoder.strides.iter().product::<usize>() == self.downsampling_ratio,
            || "decoder strides do not match downsampling_ratio".into(),
        )?;
        require(self.decoder.latent_dim == self.latent_dim, || {
            "decoder input channels do not match latent_dim".into()
        })?;
        require(self.decoder.audio_channels == self.audio_channels, || {
            "decoder out_channels do not match audio_channels".into()
        })?;
        require(self.encoder.audio_channels == self.audio_channels, || {
            "encoder in_channels do not match audio_channels".into()
        })?;
        require(self.encoder.latent_dim == 2 * self.latent_dim, || {
            "encoder latent_dim must be 2 × latent_dim (posterior mean + scale)".into()
        })?;
        require(self.decode_core_frames >= 1, || {
            "Invalid VAE core/halo configuration".into()
        })
    }
}

/// The identity of a loaded decoder, carried into every decode's metadata.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecoderIdentity {
    /// The component key (`yue2_vae` / `yue2_vae_legacy`).
    pub component_key: String,
    /// The pinned repository id.
    pub repo: String,
    /// The pinned repository revision.
    pub revision: String,
    /// `release_variant` from the verified config.
    pub release_variant: VaeVariant,
    /// SHA-256 of the verified `config.json`.
    pub config_sha256: String,
    /// SHA-256 of the verified `model.safetensors`.
    pub weights_sha256: String,
}

impl DecoderIdentity {
    /// `"standard"` / `"legacy"` — upstream's `decoder_release`.
    pub fn release(&self) -> &'static str {
        variant_name(self.release_variant)
    }

    /// The identity as JSON (artifact metadata).
    pub fn to_json(&self) -> Value {
        json!({
            "component": self.component_key,
            "repo": self.repo,
            "revision": self.revision,
            "decoder_release": self.release(),
            "config_sha256": self.config_sha256,
            "weights_sha256": self.weights_sha256,
        })
    }
}

/// `"standard"` / `"legacy"`.
pub fn variant_name(v: VaeVariant) -> &'static str {
    match v {
        VaeVariant::Standard => "standard",
        VaeVariant::Legacy => "legacy",
    }
}

/// Which halves of the VAE to load (upstream `decoder_only`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VaeParts {
    /// Decoder tensors only — what generation and decoding need (upstream's pipeline default).
    DecoderOnly,
    /// Encoder and decoder.
    Full,
}

// ---------------------------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------------------------

/// SnakeBeta with log-scale parameters, pre-exponentiated at load:
/// `x + inv_beta · sin²(alpha · x)` where `alpha = exp(α)`, `inv_beta = 1 / (exp(β) + 1e-9)`.
#[derive(Clone, Debug)]
struct SnakeBeta {
    alpha: Tensor,
    inv_beta: Tensor,
}

impl SnakeBeta {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let s = x.broadcast_mul(&self.alpha)?.sin()?.sqr()?;
        x + s.broadcast_mul(&self.inv_beta)?
    }
}

/// A (weight-norm-folded) 1-D convolution.
#[derive(Clone, Debug)]
struct Conv {
    weight: Tensor,
    bias: Option<Tensor>,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = x.conv1d(&self.weight, self.padding, self.stride, self.dilation, 1)?;
        match &self.bias {
            Some(b) => y.broadcast_add(&b.reshape((1, (), 1))?),
            None => Ok(y),
        }
    }

    fn output_length(&self, n: i64) -> i64 {
        let (k, s, p, d) = (
            self.kernel as i64,
            self.stride as i64,
            self.padding as i64,
            self.dilation as i64,
        );
        (n + 2 * p - d * (k - 1) - 1).div_euclid(s) + 1
    }
}

/// A (weight-norm-folded) transposed 1-D convolution with torch's `padding` and no output padding.
#[derive(Clone, Debug)]
struct ConvT {
    weight: Tensor,
    bias: Tensor,
    kernel: usize,
    stride: usize,
    padding: usize,
}

impl ConvT {
    /// Runs the unpadded transposed conv and crops `padding` samples from each end — exactly torch's
    /// `padding` semantics for a transposed conv, and it keeps Candle on its col2im fast path
    /// (which requires `padding == 0`).
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let y = x.conv_transpose1d(&self.weight, 0, 0, self.stride, 1, 1)?;
        let len = y.dim(D::Minus1)? - 2 * self.padding;
        y.narrow(D::Minus1, self.padding, len)?
            .broadcast_add(&self.bias.reshape((1, (), 1))?)
    }

    fn output_length(&self, n: i64) -> i64 {
        (n - 1) * self.stride as i64 - 2 * self.padding as i64 + (self.kernel as i64 - 1) + 1
    }
}

/// One node of an Oobleck stack. `Residual` is `x + inner(x)`.
#[derive(Clone, Debug)]
enum Layer {
    Snake(SnakeBeta),
    Conv(Conv),
    ConvT(ConvT),
    Residual(Vec<Layer>),
}

fn forward_all(layers: &[Layer], x: &Tensor) -> candle_core::Result<Tensor> {
    let mut x = x.clone();
    for layer in layers {
        x = match layer {
            Layer::Snake(s) => s.forward(&x)?,
            Layer::Conv(c) => c.forward(&x)?,
            Layer::ConvT(c) => c.forward(&x)?,
            Layer::Residual(inner) => (forward_all(inner, &x)? + &x)?,
        };
    }
    Ok(x)
}

/// Upstream `_output_length`.
fn output_length(layers: &[Layer], mut n: i64) -> i64 {
    for layer in layers {
        n = match layer {
            Layer::Conv(c) => c.output_length(n),
            Layer::ConvT(c) => c.output_length(n),
            Layer::Snake(_) | Layer::Residual(_) => n,
        };
    }
    n
}

/// Upstream `_dependency_interval`: the inclusive input support of the output interval
/// `[low, high]`, walking the layers in reverse. Python's `//` is floor division (`div_euclid` for
/// a positive divisor).
fn dependency_interval(layers: &[Layer], mut low: i64, mut high: i64) -> (i64, i64) {
    for layer in layers.iter().rev() {
        (low, high) = match layer {
            Layer::Snake(_) => (low, high),
            Layer::Conv(c) => {
                let (s, p, d, k) = (
                    c.stride as i64,
                    c.padding as i64,
                    c.dilation as i64,
                    c.kernel as i64,
                );
                (low * s - p, high * s - p + d * (k - 1))
            }
            Layer::ConvT(c) => {
                let (s, p, k) = (c.stride as i64, c.padding as i64, c.kernel as i64);
                // Python `-(-a // s)` is ceil(a / s); `b // s` is floor(b / s).
                let a = low + p - (k - 1);
                let ceil = -((-a).div_euclid(s));
                (ceil, (high + p).div_euclid(s))
            }
            Layer::Residual(inner) => {
                let (a, b) = dependency_interval(inner, low, high);
                (a.min(low), b.max(high))
            }
        };
    }
    (low, high)
}

/// Name-addressed FP32 tensors of one prefix, recording which names were consumed so leftovers
/// (unexpected tensors) and absences (missing tensors) are both errors.
struct Weights {
    map: BTreeMap<String, Tensor>,
    used: BTreeSet<String>,
}

impl Weights {
    fn take(&mut self, name: &str) -> Result<Tensor> {
        let t = self
            .map
            .get(name)
            .cloned()
            .ok_or_else(|| VaeError::Weights(format!("missing tensor `{name}`")))?;
        self.used.insert(name.to_string());
        Ok(t)
    }

    fn snake(&mut self, base: &str, channels: usize) -> Result<SnakeBeta> {
        let alpha = self.take(&format!("{base}.alpha"))?;
        let beta = self.take(&format!("{base}.beta"))?;
        for (n, t) in [("alpha", &alpha), ("beta", &beta)] {
            if t.dims() != [channels] {
                return Err(VaeError::Weights(format!(
                    "`{base}.{n}` has shape {:?}, expected [{channels}]",
                    t.dims()
                )));
            }
        }
        let alpha = alpha.exp()?.reshape((1, channels, 1))?;
        let inv_beta = (beta.exp()? + SNAKE_EPS)?
            .recip()?
            .reshape((1, channels, 1))?;
        Ok(SnakeBeta { alpha, inv_beta })
    }

    /// A weight-norm pair folded to a plain weight; the expected `[dim0, dim1, k]` is checked.
    fn wn(&mut self, base: &str, shape: [usize; 3]) -> Result<Tensor> {
        let v = self.take(&format!("{base}.weight_v"))?;
        let g = self.take(&format!("{base}.weight_g"))?;
        if v.dims() != shape || g.dims() != [shape[0], 1, 1] {
            return Err(VaeError::Weights(format!(
                "`{base}` weight_v {:?} / weight_g {:?}, expected {shape:?} / [{}, 1, 1]",
                v.dims(),
                g.dims(),
                shape[0]
            )));
        }
        Ok(fold_weight_norm(&v, &g)?.contiguous()?)
    }

    fn bias(&mut self, base: &str, n: usize) -> Result<Tensor> {
        let b = self.take(&format!("{base}.bias"))?;
        if b.dims() != [n] {
            return Err(VaeError::Weights(format!(
                "`{base}.bias` has shape {:?}, expected [{n}]",
                b.dims()
            )));
        }
        Ok(b)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        base: &str,
        cin: usize,
        cout: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        bias: bool,
    ) -> Result<Conv> {
        Ok(Conv {
            weight: self.wn(base, [cout, cin, kernel])?,
            bias: if bias {
                Some(self.bias(base, cout)?)
            } else {
                None
            },
            kernel,
            stride,
            padding,
            dilation,
        })
    }

    fn residual(&mut self, base: &str, ch: usize, dilation: usize) -> Result<Layer> {
        Ok(Layer::Residual(vec![
            Layer::Snake(self.snake(&format!("{base}.layers.0"), ch)?),
            Layer::Conv(self.conv(
                &format!("{base}.layers.1"),
                ch,
                ch,
                7,
                1,
                dilation * 3,
                dilation,
                true,
            )?),
            Layer::Snake(self.snake(&format!("{base}.layers.2"), ch)?),
            Layer::Conv(self.conv(&format!("{base}.layers.3"), ch, ch, 1, 1, 0, 1, true)?),
        ]))
    }

    fn finish(&self, prefix: &str) -> Result<()> {
        let unexpected: Vec<_> = self
            .map
            .keys()
            .filter(|k| !self.used.contains(*k))
            .take(5)
            .collect();
        if unexpected.is_empty() {
            Ok(())
        } else {
            Err(VaeError::Weights(format!(
                "unexpected `{prefix}` tensors: {unexpected:?}"
            )))
        }
    }
}

fn widths(cfg: &OobleckConfig) -> Vec<usize> {
    std::iter::once(1)
        .chain(cfg.c_mults.iter().copied())
        .map(|m| m * cfg.channels)
        .collect()
}

/// `OobleckDecoder`.
fn build_decoder(cfg: &OobleckConfig, w: &mut Weights) -> Result<Vec<Layer>> {
    let widths = widths(cfg);
    let depth = widths.len();
    let p = "decoder.layers";
    let mut layers = vec![Layer::Conv(w.conv(
        &format!("{p}.0"),
        cfg.latent_dim,
        widths[depth - 1],
        7,
        1,
        3,
        1,
        true,
    )?)];
    for (block, i) in (1..depth).rev().enumerate() {
        let (cin, cout, s) = (widths[i], widths[i - 1], cfg.strides[i - 1]);
        let base = format!("{p}.{}", block + 1);
        let up = format!("{base}.layers.1");
        layers.push(Layer::Snake(w.snake(&format!("{base}.layers.0"), cin)?));
        // ConvTranspose1d weights are [in, out, k] and weight_norm(dim=0) normalizes per in-channel.
        layers.push(Layer::ConvT(ConvT {
            weight: w.wn(&up, [cin, cout, 2 * s])?,
            bias: w.bias(&up, cout)?,
            kernel: 2 * s,
            stride: s,
            padding: s.div_ceil(2),
        }));
        for (j, d) in [1, 3, 9].into_iter().enumerate() {
            layers.push(w.residual(&format!("{base}.layers.{}", j + 2), cout, d)?);
        }
    }
    layers.push(Layer::Snake(w.snake(&format!("{p}.{depth}"), widths[0])?));
    layers.push(Layer::Conv(w.conv(
        &format!("{p}.{}", depth + 1),
        widths[0],
        cfg.audio_channels,
        7,
        1,
        3,
        1,
        false,
    )?));
    Ok(layers)
}

/// `OobleckEncoder`.
fn build_encoder(cfg: &OobleckConfig, w: &mut Weights) -> Result<Vec<Layer>> {
    let widths = widths(cfg);
    let depth = widths.len();
    let p = "encoder.layers";
    let mut layers = vec![Layer::Conv(w.conv(
        &format!("{p}.0"),
        cfg.audio_channels,
        widths[0],
        7,
        1,
        3,
        1,
        true,
    )?)];
    for i in 0..depth - 1 {
        let (cin, cout, s) = (widths[i], widths[i + 1], cfg.strides[i]);
        let base = format!("{p}.{}", i + 1);
        for (j, d) in [1, 3, 9].into_iter().enumerate() {
            layers.push(w.residual(&format!("{base}.layers.{j}"), cin, d)?);
        }
        layers.push(Layer::Snake(w.snake(&format!("{base}.layers.3"), cin)?));
        layers.push(Layer::Conv(w.conv(
            &format!("{base}.layers.4"),
            cin,
            cout,
            2 * s,
            s,
            s.div_ceil(2),
            1,
            true,
        )?));
    }
    layers.push(Layer::Snake(
        w.snake(&format!("{p}.{depth}"), widths[depth - 1])?,
    ));
    layers.push(Layer::Conv(w.conv(
        &format!("{p}.{}", depth + 1),
        widths[depth - 1],
        cfg.latent_dim,
        3,
        1,
        1,
        1,
        true,
    )?));
    Ok(layers)
}

/// The encoder's posterior for one audio batch.
#[derive(Clone, Debug)]
pub struct Posterior {
    /// Posterior mean `[B, 64, T]` — the default (deterministic) latent.
    pub mean: Tensor,
    /// Raw scale `[B, 64, T]`.
    pub scale: Tensor,
    /// `softplus(scale) + 1e-4`.
    pub stdev: Tensor,
}

impl Posterior {
    /// `noise · stdev + mean` with caller-injected standard-normal `noise` (upstream draws it from
    /// a `torch.Generator`; injecting it keeps sampling reproducible across runtimes, epic E9).
    pub fn sample(&self, noise: &Tensor) -> Result<Tensor> {
        if noise.dims() != self.mean.dims() {
            return Err(VaeError::Input(format!(
                "noise shape {:?}, expected {:?}",
                noise.dims(),
                self.mean.dims()
            )));
        }
        Ok(noise
            .to_dtype(DType::F32)?
            .to_device(self.mean.device())?
            .mul(&self.stdev)?
            .add(&self.mean)?)
    }
}

/// torch `softplus(x)` (β = 1, threshold 20), computed as `max(x, 0) + log1p(exp(-|x|))` with an
/// accurate `log1p` (`log(w)·u/(w−1)`, `w = 1 + u`; Goldberg), so small `exp(-|x|)` keep their
/// precision instead of rounding `1 + u` to 1.
fn softplus(x: &Tensor) -> candle_core::Result<Tensor> {
    let u = x.abs()?.neg()?.exp()?;
    let w = (&u + 1.0)?;
    let wm1 = (&w - 1.0)?;
    let zero = wm1.eq(0.0)?;
    let safe = zero.where_cond(&w.ones_like()?, &wm1)?;
    let log1p = zero.where_cond(&u, &w.log()?.mul(&u)?.div(&safe)?)?;
    let smooth = (x.relu()? + log1p)?;
    x.gt(SOFTPLUS_THRESHOLD)?.where_cond(x, &smooth)
}

/// A loaded YuE2 VAE (standard or legacy), FP32 on one device.
#[derive(Clone, Debug)]
pub struct Yue2Vae {
    config: VaeConfig,
    identity: DecoderIdentity,
    decoder: Vec<Layer>,
    encoder: Option<Vec<Layer>>,
    device: Device,
}

impl Yue2Vae {
    /// Load a verified VAE component (resolve it with [`crate::snapshot::resolve_closure`] /
    /// [`crate::snapshot::resolve_component`] immediately before calling this). Only the verified
    /// `config.json` and weights paths are read. The config's `release_variant` must agree with
    /// the component (a standard snapshot cannot load as legacy or vice versa).
    pub fn load(verified: &VerifiedComponent, parts: VaeParts, device: &Device) -> Result<Self> {
        let component = verified.component();
        let expected = match component.id {
            ComponentId::VaeStandard => VaeVariant::Standard,
            ComponentId::VaeLegacy => VaeVariant::Legacy,
            other => {
                return Err(VaeError::Config(format!(
                    "{other:?} is not a YuE2 VAE component"
                )))
            }
        };
        let missing = |file: &str| {
            VaeError::Asset(AssetError::MissingFile {
                component: component.key.to_string(),
                file: file.to_string(),
                path: verified.dir().join(file),
            })
        };
        let config_path = verified
            .path("config.json")
            .ok_or_else(|| missing("config.json"))?;
        let weights = component
            .weights()
            .ok_or_else(|| missing("model.safetensors"))?;
        let weights_path = verified
            .weights_path()
            .ok_or_else(|| missing(weights.path))?;
        let pinned = |file: &str| {
            component
                .file(file)
                .map(|f| f.sha256.to_string())
                .ok_or_else(|| missing(file))
        };
        let identity = DecoderIdentity {
            component_key: component.key.to_string(),
            repo: component.repo.id.to_string(),
            revision: component.repo.revision.to_string(),
            release_variant: expected,
            config_sha256: pinned("config.json")?,
            weights_sha256: pinned(weights.path)?,
        };
        Self::load_files(config_path, weights_path, identity, parts, device)
    }

    /// Load from a config + weights file pair under an explicit identity. Crate-internal: the
    /// public entry point is [`Self::load`], which only accepts verified snapshot paths. Used by the
    /// synthetic-weight tests (whose identity is the hashes of their fixture files).
    pub(crate) fn load_files(
        config_path: &Path,
        weights_path: &Path,
        identity: DecoderIdentity,
        parts: VaeParts,
        device: &Device,
    ) -> Result<Self> {
        let text = fs::read_to_string(config_path)
            .map_err(|e| VaeError::Config(format!("reading {}: {e}", config_path.display())))?;
        let config = VaeConfig::parse(&text)?;
        if config.release_variant != identity.release_variant {
            return Err(VaeError::Config(format!(
                "{} is a `{}` decoder config but `{}` was selected",
                config_path.display(),
                variant_name(config.release_variant),
                identity.release()
            )));
        }
        // SAFETY: memory-mapping is sound as long as the file is not mutated while mapped; the
        // production path passes a just-verified snapshot file that is only read.
        let st = unsafe { candle_core::safetensors::MmapedSafetensors::new(weights_path) }
            .map_err(|e| VaeError::Weights(format!("{}: {e}", weights_path.display())))?;
        let load_prefix = |prefix: &str| -> Result<Weights> {
            let mut map = BTreeMap::new();
            for (name, _) in st.tensors() {
                if name.starts_with(prefix) {
                    let t = st.load(&name, device)?;
                    if t.dtype() != DType::F32 {
                        return Err(VaeError::Weights(format!(
                            "`{name}` is {:?}; the validated VAE requires FP32",
                            t.dtype()
                        )));
                    }
                    map.insert(name, t);
                }
            }
            Ok(Weights {
                map,
                used: BTreeSet::new(),
            })
        };
        let mut dw = load_prefix("decoder.")?;
        let decoder = build_decoder(&config.decoder, &mut dw)?;
        dw.finish("decoder.")?;
        let encoder = match parts {
            VaeParts::DecoderOnly => None,
            VaeParts::Full => {
                let mut ew = load_prefix("encoder.")?;
                let enc = build_encoder(&config.encoder, &mut ew)?;
                ew.finish("encoder.")?;
                Some(enc)
            }
        };
        let foreign: Vec<_> = st
            .tensors()
            .into_iter()
            .map(|(n, _)| n)
            .filter(|n| !n.starts_with("decoder.") && !n.starts_with("encoder."))
            .take(5)
            .collect();
        if !foreign.is_empty() {
            return Err(VaeError::Weights(format!(
                "tensors outside encoder./decoder.: {foreign:?}"
            )));
        }
        Ok(Self {
            config,
            identity,
            decoder,
            encoder,
            device: device.clone(),
        })
    }

    /// The validated config.
    pub fn config(&self) -> &VaeConfig {
        &self.config
    }

    /// The decoder identity.
    pub fn identity(&self) -> &DecoderIdentity {
        &self.identity
    }

    /// Which published decoder this is.
    pub fn variant(&self) -> VaeVariant {
        self.identity.release_variant
    }

    /// The device the weights live on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Whether the encoder was loaded ([`VaeParts::Full`]).
    pub fn has_encoder(&self) -> bool {
        self.encoder.is_some()
    }

    /// The decoder's natural waveform length for `frames` latent frames (`1920·frames − 64` for the
    /// released geometry), computed from the loaded layers by upstream's length rule.
    pub fn natural_output_length(&self, frames: usize) -> Result<usize> {
        if frames == 0 {
            return Err(VaeError::Input("frames must be positive".into()));
        }
        Ok(output_length(&self.decoder, frames as i64) as usize)
    }

    /// The minimum halo (latent frames per side) for exact tiles of `core_frames`: upstream's
    /// `required_halo`, from the dependency interval of the loaded decoder layers.
    pub fn required_halo(&self, core_frames: usize) -> usize {
        let ratio = self.config.downsampling_ratio as i64;
        let core = core_frames as i64;
        let (low, high) = dependency_interval(&self.decoder, 0, core * ratio - 1);
        0.max(-low).max(high - core + 1) as usize
    }

    fn check_latent(&self, z: &Tensor) -> Result<()> {
        let dims = z.dims();
        if dims.len() != 3 || dims[1] != self.config.latent_dim || dims[0] < 1 || dims[2] < 1 {
            return Err(VaeError::Input(format!(
                "latents {dims:?}; expected nonempty [B, {}, T]",
                self.config.latent_dim
            )));
        }
        if z.dtype() != DType::F32 {
            return Err(VaeError::Input(format!(
                "latents are {:?}; the VAE decodes FP32 only",
                z.dtype()
            )));
        }
        let finite = z
            .flatten_all()?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_finite());
        if !finite {
            return Err(VaeError::Input(
                "VAE latents contain non-finite values".into(),
            ));
        }
        Ok(())
    }

    /// The reference FP32 full decode: `[B, 64, T]` → unclamped `[B, 2, 1920·T − 64]`
    /// (upstream `YuE2VAE.decode`).
    pub fn decode_full(&self, z: &Tensor) -> Result<Tensor> {
        self.check_latent(z)?;
        Ok(forward_all(&self.decoder, &z.to_device(&self.device)?)?)
    }

    /// Halo/crop tiled decode (upstream `YuE2VAE.decode_tiled`): tiles of `core_frames` latent
    /// frames, each decoded with `halo_frames` of context per side (at least
    /// [`Self::required_halo`]), cropped to its core and concatenated on the CPU. Returns the same
    /// unclamped `[B, 2, 1920·T − 64]` waveform as [`Self::decode_full`] up to FP32 rounding.
    /// `cancel` is polled before every tile; `on_progress(completed, total)` runs after each.
    pub fn decode_tiled(
        &self,
        z: &Tensor,
        core_frames: usize,
        halo_frames: usize,
        cancel: &dyn Fn() -> bool,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<Tensor> {
        self.check_latent(z)?;
        if core_frames < 1 {
            return Err(VaeError::Input(
                "core_frames must be a positive integer".into(),
            ));
        }
        let required = self.required_halo(core_frames);
        if halo_frames < required {
            return Err(VaeError::Input(format!(
                "halo_frames must be at least {required} for this decoder (got {halo_frames})"
            )));
        }
        let frames = z.dim(2)?;
        let ratio = self.config.downsampling_ratio;
        let total = self.natural_output_length(frames)?;
        let tiles = frames.div_ceil(core_frames);
        let mut pieces = Vec::with_capacity(tiles);
        for (index, start) in (0..frames).step_by(core_frames).enumerate() {
            if cancel() {
                return Err(VaeError::Cancelled {
                    completed: index,
                    total: tiles,
                });
            }
            let end = frames.min(start + core_frames);
            let left = start.saturating_sub(halo_frames);
            let right = frames.min(end + halo_frames);
            let tile = forward_all(
                &self.decoder,
                &z.narrow(2, left, right - left)?.to_device(&self.device)?,
            )?;
            let (out_start, out_end) = (start * ratio, (end * ratio).min(total));
            let crop_start = (start - left) * ratio;
            let len = out_end - out_start;
            if crop_start + len > tile.dim(2)? {
                return Err(VaeError::Input(
                    "VAE tile did not cover its requested output core".into(),
                ));
            }
            pieces.push(tile.narrow(2, crop_start, len)?.to_device(&Device::Cpu)?);
            on_progress(index + 1, tiles);
        }
        Ok(Tensor::cat(&pieces, 2)?)
    }

    /// Encode FP32 stereo audio `[B, 2, S]` (`S ≥ 1920`) to the posterior (upstream
    /// `YuE2VAE.encode(return_info=True)`). Requires [`VaeParts::Full`].
    pub fn encode(&self, audio: &Tensor) -> Result<Posterior> {
        let encoder = self.encoder.as_ref().ok_or_else(|| {
            VaeError::Input("Encoder not loaded; reload with VaeParts::Full".into())
        })?;
        let dims = audio.dims();
        if dims.len() != 3
            || dims[1] != self.config.audio_channels
            || dims[2] < self.config.downsampling_ratio
        {
            return Err(VaeError::Input(format!(
                "audio {dims:?}; expected [B, 2, S] with at least one latent frame"
            )));
        }
        let audio = audio.to_dtype(DType::F32)?;
        if !audio
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_finite())
        {
            return Err(VaeError::Input("Audio contains non-finite values".into()));
        }
        let pre = forward_all(encoder, &audio.to_device(&self.device)?)?;
        let half = self.config.latent_dim;
        let mean = pre.narrow(1, 0, half)?;
        let scale = pre.narrow(1, half, half)?;
        let stdev = (softplus(&scale)? + STDEV_FLOOR)?;
        Ok(Posterior { mean, scale, stdev })
    }

    /// Encode one `[2, S]` clip (channel 0 = left) to cached latents: the posterior mean, or a
    /// sample with injected `noise` (`[1, 64, T]`), attributed to this encoder and the audio hash.
    pub fn encode_latents(
        &self,
        audio: &Tensor,
        noise: Option<&Tensor>,
    ) -> Result<AcousticLatents> {
        let audio = audio.to_dtype(DType::F32)?;
        let audio_sha256 = sha256_f32(&audio.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?);
        let posterior = self.encode(&audio.unsqueeze(0)?)?;
        let z = match noise {
            Some(n) => posterior.sample(n)?,
            None => posterior.mean,
        };
        Ok(AcousticLatents::from_tensor(
            &z,
            LatentSource::Encoded {
                vae: self.identity.component_key.clone(),
                audio_sha256,
                sampled: noise.is_some(),
            },
        )?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::path::PathBuf;

    pub(crate) fn sha256_file(path: &Path) -> Result<String> {
        use sha2::{Digest, Sha256};
        let bytes = fs::read(path)
            .map_err(|e| VaeError::Weights(format!("reading {}: {e}", path.display())))?;
        Ok(Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect())
    }

    pub(crate) fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
    }

    pub(crate) fn fixture_meta() -> Value {
        serde_json::from_str(include_str!("../tests/fixtures/vae_tiny_reference.json")).unwrap()
    }

    pub(crate) fn reference() -> std::collections::HashMap<String, Tensor> {
        candle_core::safetensors::load(
            fixture_dir().join("vae_tiny_reference.safetensors"),
            &Device::Cpu,
        )
        .unwrap()
    }

    /// The synthetic-weight tiny VAE of `variant`, loaded through the same loader as production
    /// with an identity made of its fixture files' hashes.
    pub(crate) fn tiny(variant: VaeVariant, parts: VaeParts) -> Yue2Vae {
        let dir = fixture_dir().join("vae_tiny").join(variant_name(variant));
        let identity = DecoderIdentity {
            component_key: format!("tiny_{}", variant_name(variant)),
            repo: "fixture/vae_tiny".into(),
            revision: "fixture".into(),
            release_variant: variant,
            config_sha256: sha256_file(&dir.join("config.json")).unwrap(),
            weights_sha256: sha256_file(&dir.join("model.safetensors")).unwrap(),
        };
        Yue2Vae::load_files(
            &dir.join("config.json"),
            &dir.join("model.safetensors"),
            identity,
            parts,
            &Device::Cpu,
        )
        .unwrap()
    }

    pub(crate) fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        assert_eq!(a.dims(), b.dims());
        a.sub(b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    fn latent_bct() -> Tensor {
        reference()["latent"].t().unwrap().unsqueeze(0).unwrap()
    }

    /// FP32 agreement bound between the native and PyTorch CPU decoders on the tiny models.
    /// Measured max |Δ| is ~1.3e-6 (decode) / ~1.2e-7 (encode) on outputs of magnitude ≤ 2.7 (see the test's printout); 1e-5
    /// leaves ~10× headroom for BLAS/summation-order differences across machines while staying
    /// ~5 orders of magnitude below any structural error (a swapped channel, a missing Snake
    /// exponent or an off-by-one crop moves samples by O(0.1–1)).
    const TINY_TOL: f32 = 1e-5;

    /// Every released-topology invariant: the natural length rule (`1920·T − 64`) and the
    /// required halo (12 ≤ 16) agree with upstream's values recorded in the fixture
    /// (mutation: give ConvT an output padding of `s mod 2` → length red; drop the Residual branch
    /// from the dependency interval → halo red).
    #[test]
    fn geometry_matches_upstream() {
        let meta = fixture_meta();
        for v in [VaeVariant::Standard, VaeVariant::Legacy] {
            let vae = tiny(v, VaeParts::DecoderOnly);
            let m = &meta["variants"][variant_name(v)];
            let frames = meta["frames"].as_u64().unwrap() as usize;
            let core = meta["core_frames"].as_u64().unwrap() as usize;
            assert_eq!(
                vae.natural_output_length(frames).unwrap() as u64,
                m["natural_output_length"].as_u64().unwrap()
            );
            assert_eq!(
                vae.required_halo(core) as u64,
                m["required_halo"].as_u64().unwrap()
            );
            for t in [1usize, 2, 16, 257, 9000] {
                assert_eq!(vae.natural_output_length(t).unwrap(), 1920 * t - 64);
            }
            assert!(vae.required_halo(1024) <= DEFAULT_HALO_FRAMES);
        }
    }

    /// The native full decode reproduces upstream `YuE2VAE.decode` for both decoders
    /// (mutation: drop the `exp` of Snake α → red; swap output channels → red).
    #[test]
    fn full_decode_matches_upstream_for_both_decoders() {
        let r = reference();
        for v in [VaeVariant::Standard, VaeVariant::Legacy] {
            let vae = tiny(v, VaeParts::DecoderOnly);
            let got = vae.decode_full(&latent_bct()).unwrap();
            let want = &r[&format!("{}.full_raw", variant_name(v))];
            let err = max_abs_diff(&got, want);
            println!("{v:?}: native vs upstream full decode max|Δ| = {err:e}");
            assert!(err < TINY_TOL, "{v:?}: {err}");
        }
    }

    /// Tiled decode equals the full decode for many (frames, core) shapes, including a final short
    /// tile and tiles narrower than the halo; seams agree too. With an insufficient halo the
    /// decode is refused, and a deliberately under-haloed tiling (bypassing the guard) visibly
    /// differs — so the equality is not vacuous.
    #[test]
    fn tiled_decode_matches_full_decode_across_seams() {
        let vae = tiny(VaeVariant::Standard, VaeParts::DecoderOnly);
        let base = latent_bct();
        let long = Tensor::cat(&[&base, &base.affine(-0.7, 0.1).unwrap(), &base], 2).unwrap();
        let mut worst = 0f32;
        for (frames, core) in [
            (1usize, 1usize),
            (7, 3),
            (15, 7),
            (17, 8),
            (21, 4),
            (21, 20),
        ] {
            let z = long.narrow(2, 0, frames).unwrap();
            let full = vae.decode_full(&z).unwrap();
            let mut events = Vec::new();
            let tiled = vae
                .decode_tiled(&z, core, DEFAULT_HALO_FRAMES, &|| false, &mut |c, t| {
                    events.push((c, t))
                })
                .unwrap();
            assert_eq!(tiled.dims(), &[1, 2, 1920 * frames - 64]);
            let tiles = frames.div_ceil(core);
            assert_eq!(events, (1..=tiles).map(|c| (c, tiles)).collect::<Vec<_>>());
            let err = max_abs_diff(&tiled, &full);
            worst = worst.max(err);
            assert!(err < TINY_TOL, "frames {frames} core {core}: {err}");
        }
        println!("tiny tiled vs full worst max|Δ| = {worst:e}");

        let z = long.narrow(2, 0, 21).unwrap();
        let err = vae
            .decode_tiled(&z, 4, 11, &|| false, &mut |_, _| {})
            .unwrap_err();
        assert!(err.to_string().contains("at least 12"), "{err}");
        // Bypass the guard: crop tiles decoded with zero halo — seams must be visibly wrong.
        let full = vae.decode_full(&z).unwrap();
        let mut pieces = Vec::new();
        for start in (0..21).step_by(4) {
            let end = 21.min(start + 4);
            let tile = vae
                .decode_full(&z.narrow(2, start, end - start).unwrap())
                .unwrap();
            let len = (end * 1920).min(1920 * 21 - 64) - start * 1920;
            pieces.push(tile.narrow(2, 0, len.min(tile.dim(2).unwrap())).unwrap());
        }
        let naive = Tensor::cat(&pieces, 2).unwrap();
        let n = naive.dim(2).unwrap().min(full.dim(2).unwrap());
        let naive_err = max_abs_diff(
            &naive.narrow(2, 0, n).unwrap(),
            &full.narrow(2, 0, n).unwrap(),
        );
        assert!(
            naive_err > 100.0 * TINY_TOL,
            "zero-halo tiling unexpectedly exact: {naive_err}"
        );
    }

    /// Cancellation stops before the next tile and reports progress honestly.
    #[test]
    fn tiled_decode_honours_cancellation() {
        let vae = tiny(VaeVariant::Legacy, VaeParts::DecoderOnly);
        let z = latent_bct();
        let calls = std::cell::Cell::new(0);
        let err = vae
            .decode_tiled(
                &z,
                3,
                16,
                &|| {
                    calls.set(calls.get() + 1);
                    calls.get() > 2
                },
                &mut |_, _| {},
            )
            .unwrap_err();
        assert!(
            matches!(
                err,
                VaeError::Cancelled {
                    completed: 2,
                    total: 3
                }
            ),
            "{err}"
        );
    }

    /// Invalid latents are refused like upstream `_latent` (wrong width, empty, non-finite, BF16).
    #[test]
    fn invalid_latents_are_refused() {
        let vae = tiny(VaeVariant::Standard, VaeParts::DecoderOnly);
        let dev = Device::Cpu;
        for z in [
            Tensor::zeros((1, 63, 4), DType::F32, &dev).unwrap(),
            Tensor::zeros((1, 64, 0), DType::F32, &dev).unwrap(),
            Tensor::full(f32::NAN, (1, 64, 2), &dev).unwrap(),
            Tensor::zeros((1, 64, 2), DType::BF16, &dev).unwrap(),
        ] {
            assert!(
                matches!(vae.decode_full(&z), Err(VaeError::Input(_))),
                "{z:?}"
            );
        }
    }

    /// The encoder reproduces upstream's posterior (mean, scale, stdev) and the injected-noise
    /// sample for both VAEs (mutation: drop the +1e-4 floor or use plain `log(1+exp)` without the
    /// threshold → red at this tolerance for large |scale|; swap mean/scale halves → red).
    #[test]
    fn encoder_matches_upstream_posterior_and_sample() {
        let r = reference();
        for v in [VaeVariant::Standard, VaeVariant::Legacy] {
            let vae = tiny(v, VaeParts::Full);
            let name = variant_name(v);
            let post = vae.encode(&r["audio"]).unwrap();
            for (label, got) in [
                ("mean", &post.mean),
                ("scale", &post.scale),
                ("stdev", &post.stdev),
            ] {
                let err = max_abs_diff(got, &r[&format!("{name}.encode_{label}")]);
                println!("{v:?} encode {label}: max|Δ| = {err:e}");
                assert!(err < TINY_TOL, "{v:?} {label}: {err}");
            }
            let sampled = post.sample(&r[&format!("{name}.encode_noise")]).unwrap();
            assert!(max_abs_diff(&sampled, &r[&format!("{name}.encode_sampled")]) < TINY_TOL);
            let frames = meta_frames(name);
            assert_eq!(post.mean.dims(), &[1, 64, frames]);
        }
        let dec_only = tiny(VaeVariant::Standard, VaeParts::DecoderOnly);
        assert!(dec_only.encode(&r["audio"]).is_err());
        assert!(dec_only
            .encode(&Tensor::zeros((1, 2, 1919), DType::F32, &Device::Cpu).unwrap())
            .is_err());
    }

    fn meta_frames(name: &str) -> usize {
        fixture_meta()["variants"][name]["encode_frames"]
            .as_u64()
            .unwrap() as usize
    }

    /// `softplus` equals torch's definition across the threshold and deep in the negative tail.
    #[test]
    fn softplus_matches_torch_definition() {
        let xs: Vec<f32> = vec![
            -90.0, -30.0, -17.0, -5.0, -1e-3, 0.0, 1e-3, 3.0, 19.9, 20.0, 20.1, 60.0,
        ];
        let t = Tensor::new(xs.as_slice(), &Device::Cpu).unwrap();
        let got = softplus(&t).unwrap().to_vec1::<f32>().unwrap();
        for (x, g) in xs.iter().zip(got) {
            let x = *x as f64;
            let want = if x > 20.0 { x } else { x.exp().ln_1p() };
            let rel = ((g as f64 - want) / want.max(1e-38)).abs();
            assert!(rel < 1e-6, "softplus({x}) = {g}, want {want}");
        }
    }

    /// The loader refuses a config for the other variant, a non-FP32 tensor, a missing tensor and
    /// an unexpected tensor; each names the problem.
    #[test]
    fn loader_refuses_mismatched_variant_dtype_and_tensor_set() {
        let dir = fixture_dir().join("vae_tiny/standard");
        let ident = |v| DecoderIdentity {
            component_key: "x".into(),
            repo: "x".into(),
            revision: "x".into(),
            release_variant: v,
            config_sha256: String::new(),
            weights_sha256: String::new(),
        };
        let cfg = dir.join("config.json");
        let wts = dir.join("model.safetensors");
        let err = Yue2Vae::load_files(
            &cfg,
            &wts,
            ident(VaeVariant::Legacy),
            VaeParts::Full,
            &Device::Cpu,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("`standard` decoder config but `legacy`"),
            "{err}"
        );

        let tmp = tempfile::tempdir().unwrap();
        let base = candle_core::safetensors::load(&wts, &Device::Cpu).unwrap();
        let write = |map: &std::collections::HashMap<String, Tensor>| {
            let p = tmp.path().join("m.safetensors");
            candle_core::safetensors::save(map, &p).unwrap();
            p
        };
        let mut bf16 = base.clone();
        let key = "decoder.layers.0.bias".to_string();
        bf16.insert(key.clone(), base[&key].to_dtype(DType::BF16).unwrap());
        let err = Yue2Vae::load_files(
            &cfg,
            &write(&bf16),
            ident(VaeVariant::Standard),
            VaeParts::DecoderOnly,
            &Device::Cpu,
        )
        .unwrap_err();
        assert!(err.to_string().contains("requires FP32"), "{err}");

        let mut missing = base.clone();
        missing.remove("decoder.layers.3.layers.2.layers.1.weight_v");
        let err = Yue2Vae::load_files(
            &cfg,
            &write(&missing),
            ident(VaeVariant::Standard),
            VaeParts::DecoderOnly,
            &Device::Cpu,
        )
        .unwrap_err();
        assert!(err.to_string().contains("missing tensor"), "{err}");

        let mut extra = base.clone();
        extra.insert("decoder.layers.99.bias".into(), base[&key].clone());
        let err = Yue2Vae::load_files(
            &cfg,
            &write(&extra),
            ident(VaeVariant::Standard),
            VaeParts::DecoderOnly,
            &Device::Cpu,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("unexpected `decoder.` tensors"),
            "{err}"
        );

        // Decoder-only loading ignores the encoder half entirely (a broken encoder still loads).
        let mut enc_broken = base.clone();
        enc_broken.remove("encoder.layers.0.bias");
        Yue2Vae::load_files(
            &cfg,
            &write(&enc_broken),
            ident(VaeVariant::Standard),
            VaeParts::DecoderOnly,
            &Device::Cpu,
        )
        .unwrap();
        assert!(Yue2Vae::load_files(
            &cfg,
            &write(&enc_broken),
            ident(VaeVariant::Standard),
            VaeParts::Full,
            &Device::Cpu
        )
        .is_err());
    }

    /// Both pinned configs parse as their variant with the released geometry; unsupported options
    /// are refused (tanh by default, ELU, non-vanilla Snake, wrong rate/width).
    #[test]
    fn config_parsing_accepts_released_and_refuses_unsupported() {
        for (text, v) in [
            (
                include_str!("../tests/fixtures/vae_tiny/standard/config.json"),
                VaeVariant::Standard,
            ),
            (
                include_str!("../tests/fixtures/vae_tiny/legacy/config.json"),
                VaeVariant::Legacy,
            ),
        ] {
            let cfg = VaeConfig::parse(text).unwrap();
            assert_eq!(cfg.release_variant, v);
            assert_eq!(cfg.decoder.strides, vec![2, 2, 4, 4, 5, 6]);
        }
        let base: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/vae_tiny/standard/config.json"
        ))
        .unwrap();
        let mutate = |f: &dyn Fn(&mut Value)| {
            let mut v = base.clone();
            f(&mut v);
            VaeConfig::parse(&v.to_string())
        };
        assert!(mutate(&|v| {
            v["decoder_config"]
                .as_object_mut()
                .unwrap()
                .remove("final_tanh");
        })
        .is_err());
        assert!(mutate(&|v| v["decoder_config"]["use_snake"] = json!(false)).is_err());
        assert!(mutate(&|v| v["decoder_config"]["snake_type"] = json!("bigvgan")).is_err());
        assert!(mutate(&|v| v["sample_rate"] = json!(44100)).is_err());
        assert!(mutate(&|v| v["latent_dim"] = json!(32)).is_err());
        assert!(mutate(&|v| v["release_variant"] = json!("fast")).is_err());
        assert!(mutate(&|v| v["decoder_config"]["strides"] = json!([2, 2, 4, 4, 5, 5])).is_err());
    }
}
