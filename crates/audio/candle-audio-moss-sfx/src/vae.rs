//! The MOSS **continuous DAC VAE decoder** (sc-12841) — the `vae/vae_128d_48k.pth` checkpoint's
//! decode path, ported from the reference `dac_vae.py` `DAC(continuous=True)`:
//!
//! ```text
//!   z [B, 128, T] → post_quant_conv (1×1) → decoder:
//!     WNConv1d(128 → 2048, k7)
//!     5 × DecoderBlock(C → C/2, stride r) for r in decoder_rates    (upsample ×960 total)
//!         = Snake → WNConvTranspose1d(k=2r, s=r, p=⌈r/2⌉, op=r%2) → 3 × ResidualUnit(dil 1/3/9)
//!     Snake → WNConv1d(→ 1, k7) → tanh
//! ```
//!
//! The checkpoint is an `audiotools.ml.BaseModel` torch zip: `{"metadata": {"kwargs": …},
//! "state_dict": …}` with OLD-style weight-norm pairs (`weight_g`/`weight_v`). The pinned candle
//! pickle reader loads the `state_dict` section; weight-norm resolves at load
//! (`w = g · v / ‖v‖`, norm over all dims except 0 — the torch `weight_norm(dim=0)` default,
//! valid for Conv1d **and** ConvTranspose1d since `g` is `[dim0, 1, 1]` in both). Only the
//! decode-path tensors are materialized — the encoder (unused by text-to-audio) is skipped.
//!
//! The reference decodes under an fp32 autocast; this port computes in f32 throughout.
//!
//! Candle's upstream `dac.rs` decoder was deliberately **not** reused: it hardcodes
//! `output_padding = 0` (wrong for the odd strides 5 and 3 in this checkpoint's
//! `decoder_rates = [8, 5, 4, 3, 2]`) and omits the final `tanh`.

use std::collections::HashMap;
use std::path::Path;

use candle_audio::candle_core::pickle::PthTensors;
use candle_audio::candle_core::{DType, Device, Module, Result as CandleResult, Tensor, D};
use candle_audio::{AudioError, Result};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};

/// The checkpoint file inside `vae/`.
pub const VAE_FILE: &str = "vae_128d_48k.pth";

/// Decoder hyperparameters (from the checkpoint's `metadata.kwargs`; fixed for the pinned
/// snapshot and cross-checked against tensor shapes at load).
pub const LATENT_DIM: usize = 128;
pub const DECODER_DIM: usize = 2048;
pub const DECODER_RATES: [usize; 5] = [8, 5, 4, 3, 2];

/// Samples per latent frame (`∏ rates` — 960 at 48 kHz ⇒ 50 latent frames per second).
pub const HOP_LENGTH: usize = 960;

/// Snake activation: `x + (α + 1e-9)⁻¹ · sin²(αx)`, `α` per channel `[1, C, 1]`.
struct Snake {
    alpha: Tensor,
}

impl Snake {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let ax = x.broadcast_mul(&self.alpha)?;
        let s = ax.sin()?;
        let s2 = (&s * &s)?;
        x + s2.broadcast_div(&(&self.alpha + 1e-9)?)
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
        // Same-padding convs preserve length here; the reference crops symmetrically if not.
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
pub struct DacDecoder {
    post_quant_conv: Conv1d,
    conv_in: Conv1d,
    blocks: Vec<DecoderBlock>,
    snake_out: Snake,
    conv_out: Conv1d,
}

/// A name→tensor view over the resolved checkpoint map.
struct W<'a> {
    map: &'a HashMap<String, Tensor>,
}

impl W<'_> {
    fn get(&self, name: &str) -> Result<Tensor> {
        self.map
            .get(name)
            .cloned()
            .ok_or_else(|| AudioError::Msg(format!("moss-sfx VAE: missing tensor {name:?}")))
    }

    fn snake(&self, name: &str) -> Result<Snake> {
        Ok(Snake {
            alpha: self.get(&format!("{name}.alpha"))?,
        })
    }

    fn conv(&self, name: &str, cfg: Conv1dConfig) -> Result<Conv1d> {
        Ok(Conv1d::new(
            self.get(&format!("{name}.weight"))?,
            Some(self.get(&format!("{name}.bias"))?),
            cfg,
        ))
    }

    fn conv_transpose(&self, name: &str, cfg: ConvTranspose1dConfig) -> Result<ConvTranspose1d> {
        Ok(ConvTranspose1d::new(
            self.get(&format!("{name}.weight"))?,
            Some(self.get(&format!("{name}.bias"))?),
            cfg,
        ))
    }

    fn residual_unit(&self, base: &str, dilation: usize) -> Result<ResidualUnit> {
        let pad = (7 - 1) * dilation / 2;
        Ok(ResidualUnit {
            snake1: self.snake(&format!("{base}.block.0"))?,
            conv1: self.conv(
                &format!("{base}.block.1"),
                Conv1dConfig {
                    padding: pad,
                    dilation,
                    ..Default::default()
                },
            )?,
            snake2: self.snake(&format!("{base}.block.2"))?,
            conv2: self.conv(&format!("{base}.block.3"), Conv1dConfig::default())?,
        })
    }
}

/// Load the `state_dict` section of the checkpoint, keeping only the decode-path tensors,
/// with weight-norm pairs resolved to plain `weight`s in f32.
fn load_decode_tensors(pth: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let tensors = PthTensors::new(pth, Some("state_dict"))
        .map_err(|e| AudioError::Msg(format!("open {}: {e}", pth.display())))?;
    let names: Vec<String> = tensors
        .tensor_infos()
        .keys()
        .filter(|n| n.starts_with("decoder.") || n.starts_with("post_quant_conv."))
        .cloned()
        .collect();
    if names.is_empty() {
        return Err(AudioError::Msg(format!(
            "{}: no decoder tensors in state_dict — not a MOSS DAC VAE checkpoint",
            pth.display()
        )));
    }
    let mut raw: HashMap<String, Tensor> = HashMap::with_capacity(names.len());
    for name in names {
        let t = tensors
            .get(&name)
            .map_err(|e| AudioError::Msg(format!("read {name}: {e}")))?
            .ok_or_else(|| AudioError::Msg(format!("tensor {name} vanished")))?
            .to_dtype(DType::F32)?
            .to_device(device)?;
        raw.insert(name, t);
    }
    // Resolve old-style weight norm: `X.weight = X.weight_g · X.weight_v / ‖X.weight_v‖`
    // (norm over all dims except 0).
    let mut out = HashMap::with_capacity(raw.len());
    for (name, tensor) in &raw {
        if let Some(base) = name.strip_suffix(".weight_g") {
            let v = raw
                .get(&format!("{base}.weight_v"))
                .ok_or_else(|| AudioError::Msg(format!("{base}: weight_g without weight_v")))?;
            let norm = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
            let w = v.broadcast_mul(tensor)?.broadcast_div(&norm)?;
            out.insert(format!("{base}.weight"), w);
        } else if !name.ends_with(".weight_v") {
            out.insert(name.clone(), tensor.clone());
        }
    }
    Ok(out)
}

impl DacDecoder {
    /// Load the decode path from `vae/vae_128d_48k.pth`.
    pub fn load(pth: &Path, device: &Device) -> Result<Self> {
        let map = load_decode_tensors(pth, device)?;
        let w = W { map: &map };

        let post_quant_conv = w.conv("post_quant_conv", Conv1dConfig::default())?;
        if post_quant_conv.weight().dims() != [LATENT_DIM, LATENT_DIM, 1] {
            return Err(AudioError::Msg(format!(
                "moss-sfx VAE: post_quant_conv shape {:?} != the pinned [{LATENT_DIM}, \
                 {LATENT_DIM}, 1] layout",
                post_quant_conv.weight().dims()
            )));
        }
        let conv_in = w.conv(
            "decoder.model.0",
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        )?;
        let mut blocks = Vec::with_capacity(DECODER_RATES.len());
        for (i, &stride) in DECODER_RATES.iter().enumerate() {
            let base = format!("decoder.model.{}", i + 1);
            let up_cfg = ConvTranspose1dConfig {
                stride,
                padding: stride.div_ceil(2),
                output_padding: stride % 2,
                ..Default::default()
            };
            blocks.push(DecoderBlock {
                snake: w.snake(&format!("{base}.block.0"))?,
                up: w.conv_transpose(&format!("{base}.block.1"), up_cfg)?,
                res: [
                    w.residual_unit(&format!("{base}.block.2"), 1)?,
                    w.residual_unit(&format!("{base}.block.3"), 3)?,
                    w.residual_unit(&format!("{base}.block.4"), 9)?,
                ],
            });
        }
        let n = DECODER_RATES.len();
        let snake_out = w.snake(&format!("decoder.model.{}", n + 1))?;
        let conv_out = w.conv(
            &format!("decoder.model.{}", n + 2),
            Conv1dConfig {
                padding: 3,
                ..Default::default()
            },
        )?;
        Ok(Self {
            post_quant_conv,
            conv_in,
            blocks,
            snake_out,
            conv_out,
        })
    }

    /// Decode latents `[B, 128, T]` → waveform `[B, 1, T·960]` in `[-1, 1]`. `cancel` is
    /// polled between decoder stages (the upsampling blocks dominate the cost) so a cancel
    /// lands mid-decode, per the audio-lane cancellation contract.
    pub fn decode(&self, z: &Tensor, cancel: &dyn Fn() -> bool) -> CandleResult<Option<Tensor>> {
        let mut x = self.conv_in.forward(&self.post_quant_conv.forward(z)?)?;
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
        Ok(Some(x.tanh()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hop_length_is_the_rate_product() {
        assert_eq!(DECODER_RATES.iter().product::<usize>(), HOP_LENGTH);
        // 48 kHz / 960 = 50 latent frames per second — whole-frame durations at 0.02 s
        // granularity, so 30 s ⇒ exactly 1500 frames.
        assert_eq!(48_000 % HOP_LENGTH, 0);
        assert_eq!(30 * 48_000 / HOP_LENGTH, 1500);
    }

    #[test]
    fn upsample_geometry_is_exact_per_stage() {
        // (L−1)·s − 2·⌈s/2⌉ + 2s + (s mod 2) = s·L for every rate — the output-padding rule
        // that makes each stage exactly ×stride (and that upstream dac.rs gets wrong for odd
        // strides).
        for &s in &DECODER_RATES {
            for l in [1usize, 7, 50] {
                let out = (l - 1) * s + 2 * s + (s % 2) - 2 * s.div_ceil(2);
                assert_eq!(out, s * l, "stride {s} at length {l}");
            }
        }
    }

    #[test]
    fn snake_is_identity_at_zero_and_bounded_growth() {
        let dev = Device::Cpu;
        let alpha = Tensor::ones((1, 2, 1), DType::F32, &dev).unwrap();
        let snake = Snake { alpha };
        let x = Tensor::zeros((1, 2, 4), DType::F32, &dev).unwrap();
        let y = snake.forward(&x).unwrap();
        assert_eq!(
            y.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![0.0; 8]
        );
        // snake(π/2) with α=1: x + sin²(x) = π/2 + 1.
        let x = Tensor::full(std::f32::consts::FRAC_PI_2, (1, 2, 1), &dev).unwrap();
        let y = snake.forward(&x).unwrap();
        let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for got in v {
            assert!((got - (std::f32::consts::FRAC_PI_2 + 1.0)).abs() < 2e-6);
        }
    }
}
