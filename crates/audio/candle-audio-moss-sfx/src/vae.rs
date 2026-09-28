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
use candle_audio::candle_core::{DType, Device, Module, Result as CandleResult, Tensor};
use candle_audio::neural_codec::{resolve_weight_norm, DacDecoder as SharedDacDecoder};
use candle_audio::{AudioError, Result};
use candle_nn::{Conv1d, Conv1dConfig};

/// The checkpoint file inside `vae/`.
pub const VAE_FILE: &str = "vae_128d_48k.pth";

/// Decoder hyperparameters (from the checkpoint's `metadata.kwargs`; fixed for the pinned
/// snapshot and cross-checked against tensor shapes at load).
pub const LATENT_DIM: usize = 128;
pub const DECODER_DIM: usize = 2048;
pub const DECODER_RATES: [usize; 5] = [8, 5, 4, 3, 2];

/// Samples per latent frame (`∏ rates` — 960 at 48 kHz ⇒ 50 latent frames per second).
pub const HOP_LENGTH: usize = 960;

/// The loaded decode path: a 1×1 `post_quant_conv` feeding the shared DAC decoder
/// ([`candle_audio::neural_codec::DacDecoder`]), then `tanh`.
pub struct DacDecoder {
    post_quant_conv: Conv1d,
    decoder: SharedDacDecoder,
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
    Ok(resolve_weight_norm(raw)?)
}

impl DacDecoder {
    /// Load the decode path from `vae/vae_128d_48k.pth`.
    pub fn load(pth: &Path, device: &Device) -> Result<Self> {
        let map = load_decode_tensors(pth, device)?;
        let get = |name: &str| {
            map.get(name)
                .cloned()
                .ok_or_else(|| AudioError::Msg(format!("moss-sfx VAE: missing tensor {name:?}")))
        };
        let post_quant_conv = Conv1d::new(
            get("post_quant_conv.weight")?,
            Some(get("post_quant_conv.bias")?),
            Conv1dConfig::default(),
        );
        if post_quant_conv.weight().dims() != [LATENT_DIM, LATENT_DIM, 1] {
            return Err(AudioError::Msg(format!(
                "moss-sfx VAE: post_quant_conv shape {:?} != the pinned [{LATENT_DIM}, \
                 {LATENT_DIM}, 1] layout",
                post_quant_conv.weight().dims()
            )));
        }
        let decoder = SharedDacDecoder::load(&map, "decoder", &DECODER_RATES)?;
        Ok(Self {
            post_quant_conv,
            decoder,
        })
    }

    /// Decode latents `[B, 128, T]` → waveform `[B, 1, T·960]` in `[-1, 1]`. `cancel` is
    /// polled between decoder stages (the upsampling blocks dominate the cost) so a cancel
    /// lands mid-decode, per the audio-lane cancellation contract.
    pub fn decode(&self, z: &Tensor, cancel: &dyn Fn() -> bool) -> CandleResult<Option<Tensor>> {
        let x = self.post_quant_conv.forward(z)?;
        Ok(match self.decoder.decode(&x, cancel)? {
            Some(x) => Some(x.tanh()?),
            None => None,
        })
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
}
