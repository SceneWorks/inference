//! CLAP mel front-end (sc-12851): the host-side, tensor-free reproduction of
//! `ClapFeatureExtractor` with `truncation="rand_trunc"` — a **slaney** mel filterbank + power
//! spectrogram + dB log, feeding the HTSAT tower.
//!
//! Faithful to `transformers` `ClapFeatureExtractor` / `audio_utils.mel_filter_bank(norm="slaney",
//! mel_scale="slaney")` + `spectrogram(power=2.0, log_mel="dB")`, reusing
//! [`candle_audio::dsp`]'s periodic-Hann, reflect-padded (`center=True`) STFT — which matches
//! librosa/torch frame conventions exactly. The one deliberate divergence from HF: instead of
//! padding to 10 s (1001 frames) then bicubically interpolating to 1024, we pad/truncate the
//! waveform so the STFT yields exactly [`config::TARGET_FRAMES`] frames, hitting the Swin tower's
//! native `spec_width` with **no** interpolation.

use crate::config;
use candle_audio::dsp::{hann_window, stft};
use candle_audio::Result;

/// Slaney hertz→mel.
fn hertz_to_mel(freq: f32) -> f32 {
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    let logstep = 27.0 / (6.4f32).ln();
    if freq >= MIN_LOG_HZ {
        MIN_LOG_MEL + (freq / MIN_LOG_HZ).ln() * logstep
    } else {
        3.0 * freq / 200.0
    }
}

/// Slaney mel→hertz.
fn mel_to_hertz(mel: f32) -> f32 {
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = 15.0;
    let logstep = (6.4f32).ln() / 27.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    } else {
        200.0 * mel / 3.0
    }
}

/// The slaney-normalized triangular mel filterbank, row-major `[n_mels][n_freq_bins]`
/// (`filt[m * N_FREQ_BINS + b]`). Matches `transformers.audio_utils.mel_filter_bank(513, 64, 50,
/// 14000, 48000, norm="slaney", mel_scale="slaney")`.
pub fn slaney_filterbank() -> Vec<f32> {
    let n_mels = config::AUDIO_NUM_MEL_BINS;
    let n_bins = config::N_FREQ_BINS;
    let nyquist = config::SAMPLE_RATE as f32 / 2.0;

    // FFT bin center frequencies: linspace(0, sr/2, n_bins).
    let fft_freqs: Vec<f32> = (0..n_bins)
        .map(|i| i as f32 * nyquist / (n_bins - 1) as f32)
        .collect();

    // Filter edge frequencies: linspace over the mel scale, back to hertz — n_mels + 2 points.
    let mel_min = hertz_to_mel(config::MEL_FMIN);
    let mel_max = hertz_to_mel(config::MEL_FMAX);
    let filter_freqs: Vec<f32> = (0..n_mels + 2)
        .map(|j| {
            let mel = mel_min + (mel_max - mel_min) * j as f32 / (n_mels + 1) as f32;
            mel_to_hertz(mel)
        })
        .collect();

    let mut filt = vec![0.0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let f_left = filter_freqs[m];
        let f_center = filter_freqs[m + 1];
        let f_right = filter_freqs[m + 2];
        // Slaney area normalization: 2 / (f_right - f_left).
        let enorm = 2.0 / (f_right - f_left);
        for (b, &f) in fft_freqs.iter().enumerate() {
            let down = (f - f_left) / (f_center - f_left);
            let up = (f_right - f) / (f_right - f_center);
            let w = down.min(up).max(0.0);
            filt[m * n_bins + b] = w * enorm;
        }
    }
    filt
}

/// Average interleaved channels to mono (no-op for mono input).
pub fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    let ch = channels.max(1) as usize;
    if ch == 1 {
        return samples.to_vec();
    }
    samples
        .chunks(ch)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect()
}

/// Linear-interpolation resample to the target rate (no-op when already at `dst`). The audio lane's
/// standard host-side resample idiom.
pub fn resample(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round().max(1.0) as usize;
    let last = samples.len() - 1;
    (0..out_len)
        .map(|i| {
            let src_pos = i as f64 / ratio;
            let left = src_pos.floor() as usize;
            let frac = (src_pos - left as f64) as f32;
            let a = samples[left.min(last)];
            let b = samples[(left + 1).min(last)];
            a + (b - a) * frac
        })
        .collect()
}

/// Fit a mono waveform to exactly `target` samples: **repeat-pad** if short (CLAP's `padding=
/// "repeatpad"` — tile the clip to fill, matching how the feature extractor handles sub-10 s audio),
/// **center-crop** if long (the deterministic analogue of `rand_trunc`, so the same clip always
/// yields the same embedding).
pub fn fit_to_length(samples: &[f32], target: usize) -> Vec<f32> {
    if samples.is_empty() {
        return vec![0.0; target];
    }
    if samples.len() == target {
        return samples.to_vec();
    }
    if samples.len() < target {
        let mut out = Vec::with_capacity(target);
        while out.len() < target {
            let take = (target - out.len()).min(samples.len());
            out.extend_from_slice(&samples[..take]);
        }
        out
    } else {
        let start = (samples.len() - target) / 2;
        samples[start..start + target].to_vec()
    }
}

/// Full front-end: an [`AudioTrack`](candle_audio::gen_core::media::AudioTrack)'s raw PCM
/// (interleaved `samples`, `sample_rate`, `channels`) → a flat `[TARGET_FRAMES][n_mels]`
/// (`out[frame * n_mels + mel]`) log-mel spectrogram ready to reshape into the Swin input tensor.
pub fn log_mel(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    filterbank: &[f32],
) -> Result<Vec<f32>> {
    let mono = to_mono(samples, channels);
    let resampled = resample(&mono, sample_rate, config::SAMPLE_RATE);
    let fitted = fit_to_length(&resampled, config::TARGET_SAMPLES);

    let window = hann_window(config::N_FFT);
    let spec = stft(&fitted, config::N_FFT, config::HOP, &window)?;
    let n_frames = spec.n_frames;
    let n_bins = config::N_FREQ_BINS;
    let n_mels = config::AUDIO_NUM_MEL_BINS;

    // Power spectrogram (|X|^2), bin-major.
    let power: Vec<f32> = spec
        .re
        .iter()
        .zip(&spec.im)
        .map(|(r, i)| r * r + i * i)
        .collect();

    // mel[m, frame] = sum_b filt[m, b] * power[b, frame]; then dB log.
    let mut out = vec![0.0f32; n_frames * n_mels];
    for frame in 0..n_frames {
        for m in 0..n_mels {
            let mut acc = 0.0f32;
            let filt_row = &filterbank[m * n_bins..(m + 1) * n_bins];
            for (b, &fw) in filt_row.iter().enumerate() {
                acc += fw * power[b * n_frames + frame];
            }
            let acc = acc.max(1e-10);
            out[frame * n_mels + m] = 10.0 * acc.log10();
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filterbank_shape_and_partition_of_unity_ish() {
        let fb = slaney_filterbank();
        assert_eq!(fb.len(), config::AUDIO_NUM_MEL_BINS * config::N_FREQ_BINS);
        // Every filter has some positive weight (no dead mel channel).
        for m in 0..config::AUDIO_NUM_MEL_BINS {
            let row = &fb[m * config::N_FREQ_BINS..(m + 1) * config::N_FREQ_BINS];
            assert!(row.iter().any(|&w| w > 0.0), "mel {m} is all-zero");
            assert!(
                row.iter().all(|&w| w >= 0.0),
                "mel {m} has a negative weight"
            );
        }
    }

    #[test]
    fn mel_scale_round_trips() {
        for &hz in &[0.0f32, 100.0, 999.0, 1000.0, 5000.0, 14000.0] {
            let back = mel_to_hertz(hertz_to_mel(hz));
            assert!((back - hz).abs() < 1e-1, "{hz} -> {back}");
        }
    }

    #[test]
    fn log_mel_has_target_shape() {
        // A 2 s 440 Hz tone at 48 kHz; repeat-padded to the target window.
        let sr = 48_000u32;
        let samples: Vec<f32> = (0..sr * 2)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
            .collect();
        let fb = slaney_filterbank();
        let mel = log_mel(&samples, sr, 1, &fb).unwrap();
        assert_eq!(
            mel.len(),
            config::TARGET_FRAMES * config::AUDIO_NUM_MEL_BINS
        );
        assert!(mel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fit_to_length_repeats_and_crops() {
        assert_eq!(fit_to_length(&[1.0, 2.0], 5), vec![1.0, 2.0, 1.0, 2.0, 1.0]);
        assert_eq!(fit_to_length(&[1.0, 2.0, 3.0, 4.0], 2), vec![2.0, 3.0]);
        assert_eq!(fit_to_length(&[], 3), vec![0.0, 0.0, 0.0]);
    }
}
