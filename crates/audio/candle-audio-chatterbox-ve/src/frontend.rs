//! Reference-audio → mel-frame front-end for the Chatterbox voice encoder (sc-12844).
//!
//! A self-contained, host-side (`f32`) reproduction of Resemblyzer's `wav_to_mel_spectrogram`
//! (the preprocessing Chatterbox's `VoiceEncoder` was trained with):
//!
//! 1. resample the reference clip to [`config::SAMPLE_RATE`] (16 kHz),
//! 2. loudness-normalize to [`config::AUDIO_NORM_TARGET_DBFS`] (increase-only),
//! 3. STFT (`n_fft = 400`, `hop = 160`, Hann window, `center=True` reflect padding),
//! 4. project the **power** spectrum through a librosa **Slaney** mel filterbank
//!    (`htk=False`, `norm="slaney"`), **without** log compression.
//!
//! The shared `candle_audio::mel` filterbank is HTK/unnormalized (torchaudio defaults, what the
//! Kokoro stack needs); the Chatterbox encoder needs the librosa Slaney bank, so it is built here
//! rather than reusing the HTK one — matching the trained front-end is what makes the real `ve`
//! weights produce meaningful speaker vectors. Likewise `n_fft = 400` is not a power of two, so
//! the power-of-two `candle_audio::dsp::stft` cannot serve it; a small direct real-DFT is used.

use std::f32::consts::PI;

use crate::config;

/// Resample `samples` from `src_rate` to [`config::SAMPLE_RATE`] with linear interpolation.
/// Speaker identity is a spectral-envelope property that survives linear resampling well enough
/// for the encoder; exact soxr/resampy parity is not required for identity extraction.
pub fn resample_to_16k(samples: &[f32], src_rate: u32) -> Vec<f32> {
    if src_rate == config::SAMPLE_RATE || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = config::SAMPLE_RATE as f64 / src_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 / ratio;
        let left = src_pos.floor() as usize;
        let frac = (src_pos - left as f64) as f32;
        let a = samples[left.min(samples.len() - 1)];
        let b = samples[(left + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Loudness-normalize to `AUDIO_NORM_TARGET_DBFS` (power dBFS), increase-only — Resemblyzer's
/// `normalize_volume(..., increase_only=True)`.
pub fn normalize_volume(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }
    let mean_sq: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
    if mean_sq <= 0.0 {
        return;
    }
    let wave_dbfs = 10.0 * mean_sq.log10();
    let dbfs_change = config::AUDIO_NORM_TARGET_DBFS - wave_dbfs;
    if dbfs_change <= 0.0 {
        return; // increase-only
    }
    let gain = 10f32.powf(dbfs_change / 20.0);
    for s in samples.iter_mut() {
        *s *= gain;
    }
}

/// A Hann window of length `n` (`0.5 - 0.5 cos`), matching librosa's symmetric-`False` Hann.
fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

/// Reflect-pad `samples` by `pad` on both ends (numpy `mode="reflect"`, edge sample excluded) —
/// the `center=True` framing librosa uses.
fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let n = samples.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    out.extend((1..=pad).rev().map(|i| samples[i.min(n - 1)]));
    out.extend_from_slice(samples);
    out.extend((0..pad).map(|i| samples[n.saturating_sub(2 + i)]));
    out
}

/// Power STFT of `samples`: `|X|^2` per bin, returned frame-major as `[n_frames][n_bins]`
/// (`n_bins = N_FFT/2 + 1`). Direct real-DFT — `N_FFT = 400` is not a power of two.
fn power_stft(samples: &[f32]) -> Vec<Vec<f32>> {
    let n_fft = config::N_FFT;
    let hop = config::HOP;
    let n_bins = n_fft / 2 + 1;
    let window = hann(n_fft);
    let pad = n_fft / 2;
    let padded = reflect_pad(samples, pad);
    if padded.len() < n_fft {
        return Vec::new();
    }
    let n_frames = 1 + (padded.len() - n_fft) / hop;
    // Precompute DFT twiddle factors: cos/sin for each (bin, sample).
    let mut cos_tab = vec![0.0f32; n_bins * n_fft];
    let mut sin_tab = vec![0.0f32; n_bins * n_fft];
    for k in 0..n_bins {
        for t in 0..n_fft {
            let ang = -2.0 * PI * k as f32 * t as f32 / n_fft as f32;
            cos_tab[k * n_fft + t] = ang.cos();
            sin_tab[k * n_fft + t] = ang.sin();
        }
    }
    let mut frames = Vec::with_capacity(n_frames);
    let mut windowed = vec![0.0f32; n_fft];
    for f in 0..n_frames {
        let start = f * hop;
        for (i, w) in windowed.iter_mut().enumerate() {
            *w = padded[start + i] * window[i];
        }
        let mut row = vec![0.0f32; n_bins];
        for (k, slot) in row.iter_mut().enumerate() {
            let (cos_k, sin_k) = (&cos_tab[k * n_fft..], &sin_tab[k * n_fft..]);
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            for (t, &x) in windowed.iter().enumerate() {
                re += x * cos_k[t];
                im += x * sin_k[t];
            }
            *slot = re * re + im * im;
        }
        frames.push(row);
    }
    frames
}

/// librosa Slaney Hz→mel.
fn hz_to_mel_slaney(hz: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if hz < min_log_hz {
        hz / f_sp
    } else {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    }
}

/// librosa Slaney mel→Hz.
fn mel_to_hz_slaney(mel: f32) -> f32 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = (6.4f32).ln() / 27.0;
    if mel < min_log_mel {
        mel * f_sp
    } else {
        min_log_hz * ((mel - min_log_mel) * logstep).exp()
    }
}

/// A librosa Slaney mel filterbank (`htk=False`, `norm="slaney"`), stored mel-major
/// (`[n_mels][n_bins]`). `fmin = 0`, `fmax = SAMPLE_RATE/2`.
fn slaney_mel_filterbank() -> Vec<Vec<f32>> {
    let n_fft = config::N_FFT;
    let n_mels = config::N_MELS;
    let sr = config::SAMPLE_RATE as f32;
    let n_bins = n_fft / 2 + 1;
    let fmin = 0.0f32;
    let fmax = sr / 2.0;

    let mel_lo = hz_to_mel_slaney(fmin);
    let mel_hi = hz_to_mel_slaney(fmax);
    // n_mels + 2 mel-spaced edge points → their Hz positions.
    let mel_edges: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_lo + (mel_hi - mel_lo) * i as f32 / (n_mels + 1) as f32)
        .map(mel_to_hz_slaney)
        .collect();
    let fft_freqs: Vec<f32> = (0..n_bins).map(|k| k as f32 * sr / n_fft as f32).collect();

    let mut fb = vec![vec![0.0f32; n_bins]; n_mels];
    for m in 0..n_mels {
        let (lower, center, upper) = (mel_edges[m], mel_edges[m + 1], mel_edges[m + 2]);
        // Slaney area normalization for this filter.
        let enorm = 2.0 / (mel_edges[m + 2] - mel_edges[m]);
        for (b, &f) in fft_freqs.iter().enumerate() {
            let down = (f - lower) / (center - lower);
            let up = (upper - f) / (upper - center);
            let w = down.min(up).max(0.0);
            fb[m][b] = w * enorm;
        }
    }
    fb
}

/// Full reference-audio → mel frames pipeline: resample → loudness-normalize → power STFT →
/// Slaney mel (raw power, no log). Returns frame-major mel `[n_frames][N_MELS]` — the exact shape
/// the encoder consumes (`[T, 40]`).
pub fn wav_to_mel_frames(samples: &[f32], src_rate: u32) -> Vec<Vec<f32>> {
    let mut wav = resample_to_16k(samples, src_rate);
    normalize_volume(&mut wav);
    let power = power_stft(&wav);
    if power.is_empty() {
        return Vec::new();
    }
    let fb = slaney_mel_filterbank();
    power
        .iter()
        .map(|spec| {
            fb.iter()
                .map(|filter| filter.iter().zip(spec).map(|(&w, &p)| w * p).sum::<f32>())
                .collect::<Vec<f32>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_is_identity_at_native_rate() {
        let s = vec![0.1, -0.2, 0.3, -0.4];
        assert_eq!(resample_to_16k(&s, config::SAMPLE_RATE), s);
    }

    #[test]
    fn resample_changes_length_by_ratio() {
        let s = vec![0.0f32; 24_000];
        let out = resample_to_16k(&s, 24_000);
        assert_eq!(out.len(), 16_000);
    }

    #[test]
    fn normalize_only_increases_quiet_signals() {
        // A very quiet signal is boosted...
        let mut quiet = vec![0.001f32; 1000];
        normalize_volume(&mut quiet);
        assert!(quiet[0] > 0.001, "quiet signal must be boosted");
        // ...a loud one (already above target) is left unchanged (increase-only).
        let mut loud = vec![0.9f32; 1000];
        let before = loud[0];
        normalize_volume(&mut loud);
        assert_eq!(loud[0], before);
    }

    #[test]
    fn slaney_mel_round_trips() {
        for hz in [0.0f32, 100.0, 700.0, 1000.0, 4000.0, 8000.0] {
            let back = mel_to_hz_slaney(hz_to_mel_slaney(hz));
            assert!((back - hz).abs() < 1.0, "hz {hz} -> {back}");
        }
    }

    #[test]
    fn filterbank_shape_and_nonneg() {
        let fb = slaney_mel_filterbank();
        assert_eq!(fb.len(), config::N_MELS);
        assert_eq!(fb[0].len(), config::N_FFT / 2 + 1);
        for row in &fb {
            assert!(row.iter().all(|&w| w >= 0.0));
            assert!(row.iter().any(|&w| w > 0.0));
        }
    }

    #[test]
    fn mel_frames_have_the_encoder_input_shape() {
        // 0.5 s of a 220 Hz tone at 16 kHz → some frames, each 40-wide, all finite & non-negative.
        let sr = 16_000u32;
        let tone: Vec<f32> = (0..sr / 2)
            .map(|i| (2.0 * PI * 220.0 * i as f32 / sr as f32).sin() * 0.3)
            .collect();
        let frames = wav_to_mel_frames(&tone, sr);
        assert!(!frames.is_empty());
        assert!(frames.iter().all(|f| f.len() == config::N_MELS));
        assert!(frames
            .iter()
            .all(|f| f.iter().all(|&v| v.is_finite() && v >= 0.0)));
        // A tone concentrates energy: the total mel energy must be clearly positive.
        let total: f32 = frames.iter().flatten().sum();
        assert!(total > 0.0);
    }
}
