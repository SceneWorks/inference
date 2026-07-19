//! The OpenVoice V2 **tone-color (speaker) reference encoder** — a faithful candle port of
//! `models.ReferenceEncoder` (`ref_enc`), which maps a reference clip's linear spectrogram to the
//! `gin_channels`-wide tone-color embedding `g` that conditions the flow.
//!
//! Structure (reference `models.py`): six weight-normed `(3,3)` stride-`(2,2)` padding-`(1,1)`
//! Conv2d layers `[1→32→32→64→64→128→128]` each followed by ReLU, over a `[1, 1, T, spec_channels]`
//! image whose frequency axis is first `LayerNorm`-ed; the `[N, 128, T', F']` result is
//! `transpose(1,2)` → `[N, T', 128, F']` and flattened to `[N, T', 128·F']`, fed to a
//! single-layer GRU (`hidden = 128`), whose **final hidden state** projects (`Linear 128→256`) to
//! `g`. `spectrogram_torch` produces the spectrogram in `[spec_channels, T]`; OpenVoice feeds
//! `y.transpose(1,2)` (i.e. `[1, T, spec_channels]`), matched here.

use candle_audio::candle_core::{Device, Tensor};
use candle_audio::Result;
use candle_nn::{
    conv2d, gru, layer_norm, linear, Conv2d, Conv2dConfig, GRUConfig, LayerNorm, LayerNormConfig,
    Linear, Module, VarBuilder, GRU, RNN,
};

use crate::config;
use crate::spectrogram::LinearSpectrogram;

/// The loaded tone-color reference encoder.
pub struct ReferenceEncoder {
    layernorm: LayerNorm,
    convs: Vec<Conv2d>,
    gru: GRU,
    proj: Linear,
    device: Device,
}

impl ReferenceEncoder {
    /// Build from a `ref_enc`-rooted [`VarBuilder`] (`ref_enc.layernorm.*`, `ref_enc.convs.{i}.*`,
    /// `ref_enc.gru.*`, `ref_enc.proj.*`).
    pub fn new(vb: VarBuilder, device: Device) -> Result<Self> {
        let layernorm = layer_norm(
            config::SPEC_CHANNELS,
            LayerNormConfig::default(),
            vb.pp("layernorm"),
        )?;
        let conv_cfg = Conv2dConfig {
            padding: 1,
            stride: 2,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let mut convs = Vec::with_capacity(config::REF_ENC_FILTERS.len());
        let mut in_ch = 1usize;
        for (i, &out_ch) in config::REF_ENC_FILTERS.iter().enumerate() {
            convs.push(conv2d(
                in_ch,
                out_ch,
                3,
                conv_cfg,
                vb.pp(format!("convs.{i}")),
            )?);
            in_ch = out_ch;
        }
        // The GRU input width is `last_filter · F'`, where `F'` is the spectral axis after six
        // stride-2 convs (`(L - 3 + 2)/2 + 1`, iterated), matching the checkpoint's `weight_ih_l0`.
        let f_out = collapse_freq(config::SPEC_CHANNELS, config::REF_ENC_FILTERS.len());
        let gru_in = config::REF_ENC_FILTERS[config::REF_ENC_FILTERS.len() - 1] * f_out;
        let gru = gru(
            gru_in,
            config::REF_ENC_GRU_HIDDEN,
            GRUConfig::default(),
            vb.pp("gru"),
        )?;
        let proj = linear(
            config::REF_ENC_GRU_HIDDEN,
            config::GIN_CHANNELS,
            vb.pp("proj"),
        )?;
        Ok(Self {
            layernorm,
            convs,
            gru,
            proj,
            device,
        })
    }

    /// Extract the tone-color embedding `g` `[1, GIN_CHANNELS, 1]` from a linear spectrogram.
    pub fn tone_color(&self, spec: &LinearSpectrogram) -> Result<Tensor> {
        // Bin-major mag `[n_bins, T]` → `[T, n_bins]` row-major, then `[1, 1, T, n_bins]`.
        let (n_bins, t) = (spec.n_bins, spec.n_frames);
        let mut transposed = vec![0.0f32; t * n_bins];
        for bin in 0..n_bins {
            for frame in 0..t {
                transposed[frame * n_bins + bin] = spec.mag[bin * t + frame];
            }
        }
        let x = Tensor::from_vec(transposed, (1, 1, t, n_bins), &self.device)?;
        // LayerNorm over the frequency axis (last dim).
        let mut out = self.layernorm.forward(&x)?;
        for conv in &self.convs {
            out = conv.forward(&out)?.relu()?;
        }
        // `[1, C, T', F'] → [1, T', C, F'] → [1, T', C·F']`.
        let out = out.transpose(1, 2)?.contiguous()?;
        let (_, tp, c, fp) = out.dims4()?;
        let out = out.reshape((1, tp, c * fp))?;
        // GRU over the sequence; take the final hidden state (reference `memory, out = gru(...)`).
        let states = self.gru.seq(&out)?;
        let last = states.last().expect("gru produced at least one state");
        let g = self.proj.forward(&last.h)?; // [1, GIN_CHANNELS]
        Ok(g.unsqueeze(2)?) // [1, GIN_CHANNELS, 1]
    }
}

/// `calculate_channels(L, 3, 2, 1, n)` from `models.py`: the spectral length after `n` stride-2
/// `(3,3)` padding-1 convs.
pub fn collapse_freq(mut l: usize, n_convs: usize) -> usize {
    for _ in 0..n_convs {
        l = (l - 3 + 2) / 2 + 1;
    }
    l
}

/// Split-off accessor for tests: the GRU input width the six-conv stack produces at
/// [`config::SPEC_CHANNELS`].
pub fn gru_input_width() -> usize {
    config::REF_ENC_FILTERS[config::REF_ENC_FILTERS.len() - 1]
        * collapse_freq(config::SPEC_CHANNELS, config::REF_ENC_FILTERS.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collapse_freq_matches_reference_calculation() {
        // 513 → 257 → 129 → 65 → 33 → 17 → 9 (six stride-2 convs).
        assert_eq!(collapse_freq(513, 6), 9);
        assert_eq!(gru_input_width(), 128 * 9); // matches checkpoint weight_ih_l0 [384, 1152]
    }
}
