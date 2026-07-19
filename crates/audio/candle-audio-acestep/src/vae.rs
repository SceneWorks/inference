//! The ACE-Step 1.5 **`AutoencoderOobleck` decoder** (sc-12842) — the stereo music VAE decode
//! path, ported from the diffusers `OobleckDecoder`:
//!
//! ```text
//!   z [B, 64, T] → conv1 WNConv1d(64 → channels·mult[-1], k7, p3)
//!     5 × OobleckDecoderBlock(in → out, stride s):
//!         Snake1d → WNConvTranspose1d(k=2s, s, p=⌈s/2⌉) → 3 × ResidualUnit(dil 1/3/9)
//!     Snake1d → WNConv1d(channels → 2, k7, p3, bias=False) → waveform [B, 2, T·1920]
//! ```
//!
//! `channel_multiples = [1,2,4,8,16]` and `downsampling_ratios = [2,4,4,6,10]` are read from the
//! VAE config; the decoder walks them in reverse (widest channels + coarsest upsample first).
//!
//! ## Weight-norm resolution
//!
//! Oobleck's convolutions are `weight_norm`-parametrized. Depending on the export, the checkpoint
//! stores either the folded effective `.weight`, the classic `.weight_g`/`.weight_v` pair, or the
//! `torch.nn.utils.parametrizations.weight_norm` `.parametrizations.weight.original0/original1`
//! pair. `resolve_weight_norm` handles all three (`w = g · v / ‖v‖`, norm over all dims except 0).
//!
//! Latents arrive scaled by the VAE's `scaling_factor`/`std`; ACE-Step folds that into the
//! transformer output, so this decoder consumes the latents as handed to it (no extra rescale).

use std::collections::HashMap;
use std::path::Path;

use candle_audio::candle_core::{DType, Device, Module, Result as CandleResult, Tensor, D};
use candle_audio::{AudioError, Result};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};

use crate::config::VaeConfig;

/// The VAE safetensors file inside `vae/`.
pub const VAE_FILE: &str = "diffusion_pytorch_model.safetensors";

/// SnakeBeta activation: `x + (β + 1e-9)⁻¹ · sin²(αx)`, `α`/`β` per channel `[1, C, 1]` (the
/// Oobleck checkpoint's Snake carries both `alpha` and `beta`).
struct Snake {
    alpha: Tensor,
    beta: Tensor,
}

impl Snake {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let ax = x.broadcast_mul(&self.alpha)?;
        let s = ax.sin()?;
        let s2 = (&s * &s)?;
        x + s2.broadcast_div(&(&self.beta + 1e-9)?)
    }
}

struct ResidualUnit {
    snake1: Snake,
    conv1: Conv1d,
    snake2: Snake,
    conv2: Conv1d,
}

impl ResidualUnit {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let y = self.conv2.forward(
            &self
                .snake2
                .forward(&self.conv1.forward(&self.snake1.forward(x)?)?)?,
        )?;
        let pad = (x.dim(D::Minus1)? - y.dim(D::Minus1)?) / 2;
        if pad > 0 {
            y.broadcast_add(&x.narrow(D::Minus1, pad, y.dim(D::Minus1)?)?)
        } else {
            y + x
        }
    }
}

struct DecoderBlock {
    snake: Snake,
    up: ConvTranspose1d,
    res: [ResidualUnit; 3],
}

impl DecoderBlock {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mut x = self.up.forward(&self.snake.forward(x)?)?;
        for r in &self.res {
            x = r.forward(&x)?;
        }
        Ok(x)
    }
}

/// The loaded decode path.
pub struct OobleckDecoder {
    conv1: Conv1d,
    blocks: Vec<DecoderBlock>,
    snake_out: Snake,
    conv_out: Conv1d,
    hop_length: usize,
    audio_channels: usize,
}

/// A name→tensor view over the resolved checkpoint map, with weight-norm folding.
struct W<'a> {
    map: &'a HashMap<String, Tensor>,
}

impl W<'_> {
    fn get(&self, name: &str) -> Result<Tensor> {
        self.map
            .get(name)
            .cloned()
            .ok_or_else(|| AudioError::Msg(format!("acestep VAE: missing tensor {name:?}")))
    }

    fn snake(&self, name: &str) -> Result<Snake> {
        Ok(Snake {
            alpha: self.get(&format!("{name}.alpha"))?,
            beta: self.get(&format!("{name}.beta"))?,
        })
    }

    fn conv(&self, name: &str, cfg: Conv1dConfig) -> Result<Conv1d> {
        let weight = self.get(&format!("{name}.weight"))?;
        let bias = self.map.get(&format!("{name}.bias")).cloned();
        Ok(Conv1d::new(weight, bias, cfg))
    }

    fn conv_transpose(&self, name: &str, cfg: ConvTranspose1dConfig) -> Result<ConvTranspose1d> {
        let weight = self.get(&format!("{name}.weight"))?;
        let bias = self.map.get(&format!("{name}.bias")).cloned();
        Ok(ConvTranspose1d::new(weight, bias, cfg))
    }

    fn residual_unit(&self, base: &str, dilation: usize) -> Result<ResidualUnit> {
        let pad = (7 - 1) * dilation / 2;
        Ok(ResidualUnit {
            snake1: self.snake(&format!("{base}.snake1"))?,
            conv1: self.conv(
                &format!("{base}.conv1"),
                Conv1dConfig {
                    padding: pad,
                    dilation,
                    ..Default::default()
                },
            )?,
            snake2: self.snake(&format!("{base}.snake2"))?,
            conv2: self.conv(&format!("{base}.conv2"), Conv1dConfig::default())?,
        })
    }
}

/// Fold every `weight_norm` parametrization in `raw` to a plain `.weight`, tolerant of the three
/// common storage shapes (folded, `weight_g`/`weight_v`, `parametrizations.*.original0/1`).
fn resolve_weight_norm(raw: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut out: HashMap<String, Tensor> = HashMap::with_capacity(raw.len());
    for (name, tensor) in raw {
        if let Some(base) = name.strip_suffix(".weight_g") {
            let v = raw
                .get(&format!("{base}.weight_v"))
                .ok_or_else(|| AudioError::Msg(format!("{base}: weight_g without weight_v")))?;
            out.insert(format!("{base}.weight"), fold(v, tensor)?);
        } else if let Some(base) = name.strip_suffix(".parametrizations.weight.original0") {
            // original0 = g, original1 = v.
            let v = raw
                .get(&format!("{base}.parametrizations.weight.original1"))
                .ok_or_else(|| AudioError::Msg(format!("{base}: original0 without original1")))?;
            out.insert(format!("{base}.weight"), fold(v, tensor)?);
        } else if name.ends_with(".weight_v")
            || name.ends_with(".parametrizations.weight.original1")
        {
            // Consumed alongside its `g` partner above.
        } else {
            out.insert(name.clone(), tensor.clone());
        }
    }
    Ok(out)
}

/// `w = g · v / ‖v‖`, the norm taken over every dim except 0 (torch `weight_norm(dim=0)`).
fn fold(v: &Tensor, g: &Tensor) -> Result<Tensor> {
    let dims: Vec<usize> = (1..v.rank()).collect();
    let norm = v.sqr()?.sum_keepdim(dims)?.sqrt()?;
    Ok(v.broadcast_mul(g)?.broadcast_div(&norm)?)
}

impl OobleckDecoder {
    /// Load the decode path from `vae/diffusion_pytorch_model.safetensors`. Only the
    /// `decoder.*` tensors are materialized; the encoder is skipped.
    pub fn load(path: &Path, cfg: &VaeConfig, device: &Device) -> Result<Self> {
        // Pull the decoder tensor set as a name→tensor map so weight-norm folding can see both
        // halves of each pair regardless of storage shape.
        let raw = read_decoder_tensors(path, device)?;
        let map = resolve_weight_norm(&raw)?;
        let w = W { map: &map };

        // The per-stage channel widths are carried by the loaded weight shapes; the decoder walks
        // the temporal ratios in reverse (widest channels + coarsest upsample first).
        let strides: Vec<usize> = cfg.downsampling_ratios.iter().rev().copied().collect();

        let conv1 = w.conv(
            "decoder.conv1",
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        )?;
        let mut blocks = Vec::with_capacity(strides.len());
        for (i, &stride) in strides.iter().enumerate() {
            let base = format!("decoder.block.{i}");
            let up_cfg = ConvTranspose1dConfig {
                stride,
                padding: stride.div_ceil(2),
                output_padding: (2 * stride).saturating_sub(stride + 2 * stride.div_ceil(2)),
                ..Default::default()
            };
            blocks.push(DecoderBlock {
                snake: w.snake(&format!("{base}.snake1"))?,
                up: w.conv_transpose(&format!("{base}.conv_t1"), up_cfg)?,
                res: [
                    w.residual_unit(&format!("{base}.res_unit1"), 1)?,
                    w.residual_unit(&format!("{base}.res_unit2"), 3)?,
                    w.residual_unit(&format!("{base}.res_unit3"), 9)?,
                ],
            });
        }
        let snake_out = w.snake("decoder.snake1")?;
        let conv_out = w.conv(
            "decoder.conv2",
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        )?;
        Ok(Self {
            conv1,
            blocks,
            snake_out,
            conv_out,
            hop_length: cfg.hop_length(),
            audio_channels: cfg.audio_channels,
        })
    }

    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    pub fn audio_channels(&self) -> usize {
        self.audio_channels
    }

    /// Decode latents `[B, 64, T]` → waveform `[B, audio_channels, T·hop]`. `cancel` is polled
    /// between decoder stages (the upsampling blocks dominate the cost).
    pub fn decode(&self, z: &Tensor, cancel: &dyn Fn() -> bool) -> CandleResult<Option<Tensor>> {
        let mut x = self.conv1.forward(z)?;
        for block in &self.blocks {
            if cancel() {
                return Ok(None);
            }
            x = block.forward(&x)?;
        }
        if cancel() {
            return Ok(None);
        }
        let x = self.conv_out.forward(&self.snake_out.forward(&x)?)?;
        Ok(Some(x))
    }
}

/// Read the `decoder.*` tensors of a safetensors file into an f32 name→tensor map.
fn read_decoder_tensors(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    read_prefixed_tensors(path, device, "decoder.")
}

/// Read every tensor whose name starts with `prefix` into an f32 name→tensor map (the VAE ships
/// both `encoder.*` and `decoder.*` in one file; each path materializes only its half).
fn read_prefixed_tensors(
    path: &Path,
    device: &Device,
    prefix: &str,
) -> Result<HashMap<String, Tensor>> {
    let tensors = candle_audio::candle_core::safetensors::load(path, device)
        .map_err(|e| AudioError::Msg(format!("load {}: {e}", path.display())))?;
    let mut out = HashMap::new();
    for (name, t) in tensors {
        if name.starts_with(prefix) {
            out.insert(name, t.to_dtype(DType::F32)?);
        }
    }
    if out.is_empty() {
        return Err(AudioError::Msg(format!(
            "{}: no {prefix}* tensors — not an ACE-Step Oobleck VAE checkpoint",
            path.display()
        )));
    }
    Ok(out)
}

/// The ACE-Step 1.5 **`AutoencoderOobleck` encoder** (sc-12847) — the stereo music VAE *encode*
/// path used to latent-encode a source clip for prompted editing, ported from the diffusers
/// `OobleckEncoder`:
///
/// ```text
///   waveform [B, 2, T] → conv1 WNConv1d(2 → channels, k7, p3)
///     5 × OobleckEncoderBlock(in → out, downsample-stride s):
///         3 × ResidualUnit(dil 1/3/9) → Snake1d → WNConv1d(in → out, k=2s, s, p=⌈s/2⌉)
///     Snake1d → WNConv1d(channels·mult[-1] → 2·latent, k3, p1) → parameters [B, 2·64, T/1920]
///   mean, scale = parameters.chunk(2, dim=1)     # OobleckDiagonalGaussianDistribution
///   src_latents = mean.transpose(1, 2)           # [B, T/1920, 64] — the DiT src-latent space
/// ```
///
/// `downsampling_ratios = [2,4,4,6,10]` are walked in **forward** order (the mirror of the decoder,
/// which walks them reversed). The deterministic **mode** (the Gaussian mean) is used, not a
/// stochastic `.sample()`, so encoding a clip is reproducible (the gen-core seed law) — the mean is
/// the conditioning signal the reference feeds as `src_latents`, and the pinned `silence_latent`
/// buffer is itself a raw encoder mean, so no `scaling_factor` rescale is applied (the same space
/// the decoder consumes and the DiT integrates in).
pub struct OobleckEncoder {
    conv1: Conv1d,
    blocks: Vec<EncoderBlock>,
    snake_out: Snake,
    conv_out: Conv1d,
    hop_length: usize,
    latent_channels: usize,
}

/// One `OobleckEncoderBlock`: three residual units, a Snake, then the strided downsample conv.
struct EncoderBlock {
    res: [ResidualUnit; 3],
    snake: Snake,
    down: Conv1d,
}

impl EncoderBlock {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mut x = x.clone();
        for r in &self.res {
            x = r.forward(&x)?;
        }
        self.down.forward(&self.snake.forward(&x)?)
    }
}

impl OobleckEncoder {
    /// Load the encode path from `vae/diffusion_pytorch_model.safetensors`. Only the `encoder.*`
    /// tensors are materialized; the decoder is skipped.
    pub fn load(path: &Path, cfg: &VaeConfig, device: &Device) -> Result<Self> {
        let raw = read_prefixed_tensors(path, device, "encoder.")?;
        let map = resolve_weight_norm(&raw)?;
        let w = W { map: &map };

        // Forward-order temporal ratios (the mirror of the decoder's reversed walk).
        let strides: Vec<usize> = cfg.downsampling_ratios.to_vec();

        let conv1 = w.conv(
            "encoder.conv1",
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        )?;
        let mut blocks = Vec::with_capacity(strides.len());
        for (i, &stride) in strides.iter().enumerate() {
            let base = format!("encoder.block.{i}");
            blocks.push(EncoderBlock {
                res: [
                    w.residual_unit(&format!("{base}.res_unit1"), 1)?,
                    w.residual_unit(&format!("{base}.res_unit2"), 3)?,
                    w.residual_unit(&format!("{base}.res_unit3"), 9)?,
                ],
                snake: w.snake(&format!("{base}.snake1"))?,
                down: w.conv(
                    &format!("{base}.conv1"),
                    Conv1dConfig {
                        stride,
                        padding: stride.div_ceil(2),
                        ..Default::default()
                    },
                )?,
            });
        }
        let snake_out = w.snake("encoder.snake1")?;
        let conv_out = w.conv(
            "encoder.conv2",
            Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
        )?;
        Ok(Self {
            conv1,
            blocks,
            snake_out,
            conv_out,
            hop_length: cfg.hop_length(),
            latent_channels: cfg.decoder_input_channels,
        })
    }

    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// Encode a waveform `[1, audio_channels, T]` → deterministic latents `[1, T/hop, latent]`
    /// (the Gaussian **mean**, in the DiT src-latent space). `T` should be a multiple of
    /// [`hop_length`](Self::hop_length) (the caller pads); the strided convs otherwise floor the
    /// last partial frame. `cancel` is polled between downsampling stages.
    pub fn encode(
        &self,
        waveform: &Tensor,
        cancel: &dyn Fn() -> bool,
    ) -> CandleResult<Option<Tensor>> {
        let mut x = self.conv1.forward(waveform)?;
        for block in &self.blocks {
            if cancel() {
                return Ok(None);
            }
            x = block.forward(&x)?;
        }
        if cancel() {
            return Ok(None);
        }
        let params = self.conv_out.forward(&self.snake_out.forward(&x)?)?; // [1, 2·latent, T']
                                                                           // OobleckDiagonalGaussianDistribution: mean = first `latent` channels; deterministic mode.
        let mean = params.narrow(1, 0, self.latent_channels)?;
        Ok(Some(mean.transpose(1, 2)?.contiguous()?)) // [1, T', latent]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_audio::candle_core::Device;

    #[test]
    fn upsample_product_matches_hop_length() {
        let cfg: VaeConfig = serde_json::from_str(
            r#"{"audio_channels": 2, "channel_multiples": [1,2,4,8,16], "decoder_channels": 128,
                "decoder_input_channels": 64, "downsampling_ratios": [2,4,4,6,10],
                "encoder_hidden_size": 128, "sampling_rate": 48000}"#,
        )
        .unwrap();
        assert_eq!(cfg.hop_length(), 1920);
        // Per-stage output-padding rule makes each ConvTranspose1d exactly ×stride.
        for &s in &cfg.downsampling_ratios {
            let op = (2 * s).saturating_sub(s + 2 * s.div_ceil(2));
            for l in [1usize, 4, 25] {
                let out = (l - 1) * s + 2 * s + op - 2 * s.div_ceil(2);
                assert_eq!(out, s * l, "stride {s} len {l}");
            }
        }
    }

    #[test]
    fn encoder_downsample_product_matches_hop_length() {
        // The encoder walks the ratios in FORWARD order; each strided conv (k=2s, stride s,
        // padding ⌈s/2⌉) is exactly ×(1/s), so a hop-aligned input yields hop_length⁻¹ frames.
        let cfg: VaeConfig = serde_json::from_str(
            r#"{"audio_channels": 2, "channel_multiples": [1,2,4,8,16], "decoder_channels": 128,
                "decoder_input_channels": 64, "downsampling_ratios": [2,4,4,6,10],
                "encoder_hidden_size": 128, "sampling_rate": 48000}"#,
        )
        .unwrap();
        assert_eq!(cfg.hop_length(), 1920);
        for &s in &cfg.downsampling_ratios {
            let pad = s.div_ceil(2);
            let k = 2 * s;
            for frames_out in [1usize, 4, 25] {
                let t_in = frames_out * s;
                // Conv1d output length: floor((T + 2p − k)/s) + 1.
                let out = (t_in + 2 * pad - k) / s + 1;
                assert_eq!(out, frames_out, "stride {s}: {t_in} in → {frames_out} out");
            }
        }
    }

    #[test]
    fn snake_identity_at_zero() {
        let dev = Device::Cpu;
        let snake = Snake {
            alpha: Tensor::ones((1, 2, 1), DType::F32, &dev).unwrap(),
            beta: Tensor::ones((1, 2, 1), DType::F32, &dev).unwrap(),
        };
        let x = Tensor::zeros((1, 2, 4), DType::F32, &dev).unwrap();
        assert_eq!(
            snake
                .forward(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            vec![0.0; 8]
        );
    }

    #[test]
    fn weight_norm_folds_g_v_pair() {
        let dev = Device::Cpu;
        let mut raw = HashMap::new();
        raw.insert(
            "c.weight_g".into(),
            Tensor::ones((2, 1, 1), DType::F32, &dev).unwrap(),
        );
        let v = Tensor::from_vec(vec![3.0f32, 4.0, 0.0, 5.0], (2, 1, 2), &dev).unwrap();
        raw.insert("c.weight_v".into(), v);
        let out = resolve_weight_norm(&raw).unwrap();
        let w = out
            .get("c.weight")
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // Row 0 norm 5 → [0.6, 0.8]; row 1 norm 5 → [0, 1].
        assert!((w[0] - 0.6).abs() < 1e-6 && (w[1] - 0.8).abs() < 1e-6);
        assert!((w[2] - 0.0).abs() < 1e-6 && (w[3] - 1.0).abs() < 1e-6);
    }
}
