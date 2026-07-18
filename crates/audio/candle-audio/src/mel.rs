//! Mel filterbank construction and application (sc-12835) — the mel-spectrogram
//! front-end the audio providers need for reference-audio conditioning and
//! preprocessing parity with their reference implementations.
//!
//! The filterbank uses the **HTK** mel scale (`m = 2595 log10(1 + f/700)`) with
//! unnormalized triangular filters — the `torchaudio.transforms.MelSpectrogram`
//! defaults (`mel_scale="htk"`, `norm=None`) that the StyleTTS2/Kokoro preprocessing
//! stack (sc-12836) is built on. Power/log compression and any Slaney-style variants
//! stay provider-owned composition over these primitives — no speculative knobs here.

use crate::{AudioError, Result};

/// Hz → HTK mel.
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// HTK mel → Hz.
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// An `n_mels × n_bins` triangular mel filterbank, stored mel-major
/// (`index = mel * n_bins + bin`, `n_bins = n_fft / 2 + 1`).
#[derive(Clone, Debug)]
pub struct MelFilterbank {
    pub n_mels: usize,
    pub n_bins: usize,
    pub weights: Vec<f32>,
}

impl MelFilterbank {
    /// Build the filterbank for an analysis of `n_fft` points at `sample_rate`, spanning
    /// `fmin..=fmax` Hz with `n_mels` triangles (HTK scale, unnormalized).
    pub fn new(
        sample_rate: u32,
        n_fft: usize,
        n_mels: usize,
        fmin: f32,
        fmax: f32,
    ) -> Result<Self> {
        if sample_rate == 0 || n_fft < 2 || n_mels == 0 {
            return Err(AudioError::Msg(format!(
                "mel filterbank needs sample_rate >= 1, n_fft >= 2, n_mels >= 1 \
                 (got {sample_rate}, {n_fft}, {n_mels})"
            )));
        }
        let nyquist = sample_rate as f32 / 2.0;
        if !(0.0..nyquist).contains(&fmin) || fmax <= fmin || fmax > nyquist {
            return Err(AudioError::Msg(format!(
                "mel filterbank needs 0 <= fmin < fmax <= nyquist ({nyquist} Hz), \
                 got fmin={fmin} fmax={fmax}"
            )));
        }
        let n_bins = n_fft / 2 + 1;
        // n_mels triangles need n_mels + 2 mel-spaced edge points.
        let (mel_lo, mel_hi) = (hz_to_mel(fmin), hz_to_mel(fmax));
        let edges_hz: Vec<f32> = (0..n_mels + 2)
            .map(|i| mel_to_hz(mel_lo + (mel_hi - mel_lo) * i as f32 / (n_mels + 1) as f32))
            .collect();
        let bin_hz = |bin: usize| bin as f32 * sample_rate as f32 / n_fft as f32;

        let mut weights = vec![0.0f32; n_mels * n_bins];
        for m in 0..n_mels {
            let (left, center, right) = (edges_hz[m], edges_hz[m + 1], edges_hz[m + 2]);
            for bin in 0..n_bins {
                let f = bin_hz(bin);
                let w = if f <= left || f >= right {
                    0.0
                } else if f <= center {
                    (f - left) / (center - left)
                } else {
                    (right - f) / (right - center)
                };
                weights[m * n_bins + bin] = w;
            }
        }
        Ok(Self {
            n_mels,
            n_bins,
            weights,
        })
    }

    /// Apply the filterbank to a bin-major `[n_bins, n_frames]` spectrogram (magnitude or
    /// power, per the provider's reference convention), yielding a mel-major
    /// `[n_mels, n_frames]` mel spectrogram.
    pub fn apply(&self, spectrum: &[f32], n_frames: usize) -> Result<Vec<f32>> {
        if spectrum.len() != self.n_bins * n_frames {
            return Err(AudioError::Msg(format!(
                "mel apply expects bin-major [{}, {n_frames}] input ({} values), got {}",
                self.n_bins,
                self.n_bins * n_frames,
                spectrum.len()
            )));
        }
        let mut out = vec![0.0f32; self.n_mels * n_frames];
        for m in 0..self.n_mels {
            let filter = &self.weights[m * self.n_bins..(m + 1) * self.n_bins];
            for (bin, &w) in filter.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                let row = &spectrum[bin * n_frames..(bin + 1) * n_frames];
                for (t, &v) in row.iter().enumerate() {
                    out[m * n_frames + t] += w * v;
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_scale_round_trips() {
        for hz in [0.0f32, 100.0, 440.0, 8000.0, 11025.0] {
            assert!((mel_to_hz(hz_to_mel(hz)) - hz).abs() < 0.5, "{hz}");
        }
    }

    #[test]
    fn filterbank_shape_and_triangles() {
        let fb = MelFilterbank::new(24_000, 2048, 80, 0.0, 12_000.0).unwrap();
        assert_eq!(fb.n_mels, 80);
        assert_eq!(fb.n_bins, 1025);
        assert_eq!(fb.weights.len(), 80 * 1025);
        // Every filter is non-negative, bounded by 1, and carries some energy.
        for m in 0..fb.n_mels {
            let row = &fb.weights[m * fb.n_bins..(m + 1) * fb.n_bins];
            assert!(row.iter().all(|&w| (0.0..=1.0).contains(&w)));
            assert!(row.iter().any(|&w| w > 0.0), "filter {m} is empty");
        }
        // Filter peaks move monotonically up the spectrum.
        let peak_bin = |m: usize| {
            let row = &fb.weights[m * fb.n_bins..(m + 1) * fb.n_bins];
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap()
        };
        for m in 1..fb.n_mels {
            assert!(peak_bin(m) >= peak_bin(m - 1));
        }
    }

    #[test]
    fn apply_projects_energy_into_the_matching_band() {
        let fb = MelFilterbank::new(16_000, 512, 40, 0.0, 8000.0).unwrap();
        // One frame with all energy in a single mid-spectrum FFT bin.
        let hot_bin = 100usize;
        let mut spectrum = vec![0.0f32; fb.n_bins];
        spectrum[hot_bin] = 1.0;
        let mel = fb.apply(&spectrum, 1).unwrap();
        assert_eq!(mel.len(), 40);
        let total: f32 = mel.iter().sum();
        assert!(total > 0.0, "energy must land in some mel band");
        // The hot mel band's filter must actually cover the hot bin.
        let hottest = mel
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        assert!(fb.weights[hottest * fb.n_bins + hot_bin] > 0.0);
    }

    #[test]
    fn rejects_malformed_configs() {
        assert!(MelFilterbank::new(0, 512, 40, 0.0, 8000.0).is_err());
        assert!(MelFilterbank::new(16_000, 512, 0, 0.0, 8000.0).is_err());
        assert!(MelFilterbank::new(16_000, 512, 40, 4000.0, 2000.0).is_err()); // fmax <= fmin
        assert!(MelFilterbank::new(16_000, 512, 40, 0.0, 9000.0).is_err()); // above nyquist
        let fb = MelFilterbank::new(16_000, 512, 40, 0.0, 8000.0).unwrap();
        assert!(fb.apply(&[0.0; 10], 1).is_err()); // wrong input size
    }
}
