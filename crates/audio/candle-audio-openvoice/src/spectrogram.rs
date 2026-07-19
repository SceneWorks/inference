//! Host-side (`f32`) front-end for the OpenVoice V2 converter (sc-13223): linear resampling and the
//! exact `spectrogram_torch` linear spectrogram both the reference encoder and the posterior
//! encoder consume.
//!
//! `spectrogram_torch` (OpenVoice `mel_processing.py`) is **not** the `center=True` librosa default
//! the shared [`candle_audio::dsp::stft`] implements — it reflect-pads the waveform by
//! `(n_fft - hop) / 2` on each side and then runs `torch.stft(..., center=False)`, taking
//! `sqrt(re² + im² + 1e-6)` (note the in-sqrt epsilon). Matching this framing exactly is what makes
//! the real weights produce a faithful conversion, so the spectrogram is reproduced here as
//! self-contained DSP with a small radix-2 FFT (`n_fft = 1024` is a power of two), unit-testable
//! without weights and numerically identical across CPU/Metal/CUDA.

use crate::config;

/// Resample `samples` from `src_rate` to [`config::SAMPLE_RATE`] with linear interpolation. Timbre
/// and content are spectral-envelope / pitch properties that survive linear resampling well enough
/// for the converter; exact soxr/librosa-kaiser parity is not required (mirrors the sibling
/// providers' `resample_to_16k`).
pub fn resample_to_native(samples: &[f32], src_rate: u32) -> Vec<f32> {
    if src_rate == config::SAMPLE_RATE || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = config::SAMPLE_RATE as f64 / src_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    let last = samples.len() - 1;
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let left = src_pos.floor() as usize;
        let frac = (src_pos - left as f64) as f32;
        let a = samples[left.min(last)];
        let b = samples[(left + 1).min(last)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// A periodic Hann window of length `n` (`torch.hann_window(n)`): `0.5 - 0.5·cos(2πk/n)`.
fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * k as f32 / n as f32).cos())
        .collect()
}

/// In-place iterative radix-2 Cooley–Tukey FFT over interleaved `(re, im)` pairs (forward, no
/// scaling — the `torch.stft(normalized=False)` convention). `data.len()` must be a power of two.
fn fft_forward(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        // f64 twiddles keep error flat over the 1024-point window.
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let wr = (ang * k as f64).cos() as f32;
                let wi = (ang * k as f64).sin() as f32;
                let a = start + k;
                let b = start + k + len / 2;
                let tr = re[b] * wr - im[b] * wi;
                let ti = re[b] * wi + im[b] * wr;
                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
        len <<= 1;
    }
}

/// A linear spectrogram in the `[spec_channels, n_frames]` bin-major layout OpenVoice's tensor
/// carries (`index = bin · n_frames + frame`).
#[derive(Clone, Debug)]
pub struct LinearSpectrogram {
    pub n_bins: usize,
    pub n_frames: usize,
    /// Magnitudes `sqrt(re² + im² + 1e-6)`, bin-major.
    pub mag: Vec<f32>,
}

/// Compute OpenVoice's `spectrogram_torch` linear spectrogram of a mono waveform at
/// [`config::SAMPLE_RATE`]: reflect-pad by `(n_fft - hop)/2`, frame with `center=False`, Hann
/// window, one-sided magnitude with the in-sqrt `1e-6`. Returns `None` for a clip too short to
/// yield a single frame.
pub fn spectrogram(samples: &[f32]) -> Option<LinearSpectrogram> {
    let n_fft = config::FILTER_LENGTH;
    let hop = config::HOP_LENGTH;
    let pad = (n_fft - hop) / 2; // 384
    if samples.len() <= 1 {
        return None;
    }
    // Reflect-pad both ends (torch F.pad mode="reflect"): mirror about the edge sample, excluding
    // the edge itself.
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    let reflect = |idx: i64, len: usize| -> f32 {
        // Mirror an out-of-range index back into [0, len) the way torch reflect padding does.
        let n = len as i64;
        let mut p = idx;
        if n == 1 {
            return samples[0];
        }
        let period = 2 * (n - 1);
        p = ((p % period) + period) % period;
        if p >= n {
            p = period - p;
        }
        samples[p as usize]
    };
    for i in 0..pad {
        padded.push(reflect(-(pad as i64) + i as i64, samples.len()));
    }
    padded.extend_from_slice(samples);
    for i in 0..pad {
        padded.push(reflect(samples.len() as i64 + i as i64, samples.len()));
    }

    if padded.len() < n_fft {
        return None;
    }
    let n_bins = n_fft / 2 + 1;
    let n_frames = 1 + (padded.len() - n_fft) / hop;
    let window = hann_window(config::WIN_LENGTH);
    let mut mag = vec![0.0f32; n_bins * n_frames];
    let mut re = vec![0.0f32; n_fft];
    let mut im = vec![0.0f32; n_fft];
    for t in 0..n_frames {
        let start = t * hop;
        for k in 0..n_fft {
            re[k] = padded[start + k] * window[k];
            im[k] = 0.0;
        }
        fft_forward(&mut re, &mut im);
        for bin in 0..n_bins {
            let r = re[bin];
            let i = im[bin];
            mag[bin * n_frames + t] = (r * r + i * i + 1e-6).sqrt();
        }
    }
    Some(LinearSpectrogram {
        n_bins,
        n_frames,
        mag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_is_identity_at_native_rate() {
        let s = vec![0.1f32, 0.2, 0.3];
        assert_eq!(resample_to_native(&s, config::SAMPLE_RATE), s);
    }

    #[test]
    fn resample_changes_length_by_ratio() {
        let s = vec![0.0f32; 24_000];
        let out = resample_to_native(&s, 24_000);
        // 24 kHz → 22.05 kHz shortens by the rate ratio.
        let expected = (24_000.0f64 * (22_050.0 / 24_000.0)).round() as usize;
        assert_eq!(out.len(), expected);
    }

    #[test]
    fn fft_matches_known_single_tone() {
        // A pure cosine at bin 1 of a 16-point frame: energy lands in bin 1 with re = n/2.
        let n = 16;
        let mut re: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
            .collect();
        let mut im = vec![0.0f32; n];
        fft_forward(&mut re, &mut im);
        assert!((re[1] - 8.0).abs() < 1e-3, "re[1] = n/2, got {}", re[1]);
        for k in 2..n / 2 {
            assert!(
                re[k].abs() < 1e-3 && im[k].abs() < 1e-3,
                "bin {k} not empty"
            );
        }
    }

    #[test]
    fn spectrogram_shape_and_duration_preservation() {
        // A deterministic tone; n_frames·hop should land within one hop of the input length
        // (center=False with the (n_fft-hop)/2 reflect pad ≈ duration-preserving).
        let len = 22_050; // 1 s
        let sig: Vec<f32> = (0..len)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 22_050.0).sin() * 0.5)
            .collect();
        let spec = spectrogram(&sig).expect("spectrogram");
        assert_eq!(spec.n_bins, config::SPEC_CHANNELS);
        let out_samples = spec.n_frames * config::HOP_LENGTH;
        assert!(
            (out_samples as i64 - len as i64).abs() <= config::HOP_LENGTH as i64,
            "n_frames·hop {out_samples} should be within one hop of {len}"
        );
        assert!(spec.mag.iter().all(|m| m.is_finite() && *m >= 0.0));
        // The 220 Hz tone must put real energy somewhere below the mid spectrum.
        assert!(spec.mag.iter().any(|&m| m > 1.0));
    }

    #[test]
    fn too_short_clip_yields_no_frames() {
        assert!(spectrogram(&[0.0]).is_none());
        assert!(spectrogram(&[]).is_none());
    }
}
