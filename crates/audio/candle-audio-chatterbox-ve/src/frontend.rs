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
//! the power-of-two `candle_audio::dsp::stft` cannot serve it; the shared direct real-DFT
//! ([`candle_audio::dsp::DirectRdft`]) is used.

use candle_audio::dsp;

use crate::config;

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

/// Power STFT of `samples`: `|X|^2` per bin, returned frame-major as `[n_frames][n_bins]`
/// (`n_bins = N_FFT/2 + 1`): `center=True` reflect padding, periodic Hann (`0.5 - 0.5 cos`), and
/// the shared direct real-DFT (`N_FFT = 400` is not a power of two).
fn power_stft(samples: &[f32]) -> Vec<Vec<f32>> {
    let n_fft = config::N_FFT;
    let window = dsp::hann_window_torch_f32(n_fft);
    let padded = dsp::reflect_pad(samples, n_fft / 2);
    dsp::DirectRdft::new(n_fft).power_frames(&padded, config::HOP, &window)
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
pub fn wav_to_mel_frames(samples: &[f32], src_rate: u32) -> candle_audio::Result<Vec<Vec<f32>>> {
    let mut wav = candle_audio::dsp::resample(samples, src_rate, config::SAMPLE_RATE, 1)?;
    normalize_volume(&mut wav);
    let power = power_stft(&wav);
    if power.is_empty() {
        return Ok(Vec::new());
    }
    let fb = slaney_mel_filterbank();
    Ok(power
        .iter()
        .map(|spec| {
            fb.iter()
                .map(|filter| filter.iter().zip(spec).map(|(&w, &p)| w * p).sum::<f32>())
                .collect::<Vec<f32>>()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

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
        let frames = wav_to_mel_frames(&tone, sr).unwrap();
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
