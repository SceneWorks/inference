//! The S3Gen **24 kHz prompt-mel front-end** (sc-13237). A faithful native port of Chatterbox's
//! `models/s3gen/utils/mel.py` `mel_spectrogram` in the pinned CosyVoice configuration
//! (`n_fft = 1920`, `hop = 480`, `win = 1920`, `num_mels = 80`, `sr = 24000`, `fmin = 0`,
//! `fmax = 8000`, `center = False`) — a **50 Hz** 80-bin log-mel, exactly `token_mel_ratio = 2` ×
//! the 25 Hz speech-token rate.
//!
//! It supplies the flow's conditioning **prompt mel** (`prompt_feat`) of the reference voice. The
//! non-power-of-two `n_fft = 1920` STFT is served by a direct real-DFT (the shared radix-2
//! `candle_audio::dsp` cannot), the same idiom the s3tokenizer / chatterbox_ve mel front-ends use.
//! The librosa **Slaney** mel bank (`htk = False`, area-normalized) is reconstructed here exactly
//! by construction, so no librosa dependency is needed.
//!
//! It is implemented cleanly and reusably because the HiFTNet vocoder (sc-13238) consumes the
//! **same** 24 kHz mel extractor.
//!
//! ## Faithfulness (verified against `librosa` + the reference `mel_spectrogram`)
//!
//! - reflect-pad the waveform by `(n_fft − hop) / 2 = 720` on both ends (the reference pads before a
//!   `center = False` STFT — equivalent to a centered STFT), periodic Hann window;
//! - power-spectrum magnitude `sqrt(re² + im² + 1e-9)` per one-sided bin (`n_fft/2 + 1 = 961`);
//! - project through the librosa Slaney 80-mel bank, then `log(clamp(·, 1e-5))`
//!   (`spectral_normalize` / `dynamic_range_compression`).

use candle_audio::candle_core::{Device, Result as CandleResult, Tensor};

use crate::config::{S3GenConfig, S3GEN_SR};

/// Slaney `hz → mel` (librosa `htk = False`): linear below 1 kHz, log above.
fn hz_to_mel(hz: f64) -> f64 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp; // 15.0
    let logstep = (6.4f64).ln() / 27.0;
    if hz >= min_log_hz {
        min_log_mel + (hz / min_log_hz).ln() / logstep
    } else {
        (hz - f_min) / f_sp
    }
}

/// Slaney `mel → hz` — the inverse of [`hz_to_mel`].
fn mel_to_hz(mel: f64) -> f64 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4f64).ln() / 27.0;
    if mel >= min_log_mel {
        min_log_hz * ((mel - min_log_mel) * logstep).exp()
    } else {
        f_min + f_sp * mel
    }
}

/// librosa `mel(sr, n_fft, n_mels, fmin, fmax, htk=False, norm="slaney")`, mel-major
/// `[n_mels][n_bins]` (`n_bins = n_fft/2 + 1`). Triangular filters with Slaney area normalization.
fn librosa_mel_bank(cfg: &S3GenConfig) -> Vec<f32> {
    let sr = S3GEN_SR as f64;
    let n_fft = cfg.mel_n_fft;
    let n_mels = cfg.mel_num_mels;
    let n_bins = n_fft / 2 + 1;
    let fmin = cfg.mel_fmin as f64;
    let fmax = cfg.mel_fmax as f64;

    // FFT bin center frequencies: linspace(0, sr/2, n_bins).
    let fft_freqs: Vec<f64> = (0..n_bins)
        .map(|k| (sr / 2.0) * k as f64 / (n_bins as f64 - 1.0))
        .collect();

    // n_mels + 2 mel points, evenly spaced in mel, mapped back to Hz.
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_f: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels as f64 + 1.0)))
        .collect();

    let mut bank = vec![0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let left = mel_f[m];
        let center = mel_f[m + 1];
        let right = mel_f[m + 2];
        let fdiff_lo = center - left;
        let fdiff_hi = right - center;
        // Slaney area normalization: 2 / (right − left).
        let enorm = 2.0 / (right - left);
        for (k, &f) in fft_freqs.iter().enumerate() {
            let lower = (f - left) / fdiff_lo;
            let upper = (right - f) / fdiff_hi;
            let w = lower.min(upper).max(0.0) * enorm;
            bank[m * n_bins + k] = w as f32;
        }
    }
    bank
}

/// Periodic Hann window of length `n` (`torch.hann_window(n)`, `periodic = True`).
fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
        .collect()
}

/// Reflect-pad by `pad` on both ends (numpy/torch `mode="reflect"`, edge sample excluded).
fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let n = samples.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n + 2 * pad);
    out.extend((1..=pad).rev().map(|i| samples[i.min(n - 1)]));
    out.extend_from_slice(samples);
    out.extend((0..pad).map(|i| samples[n.saturating_sub(2 + i)]));
    out
}

/// Linear-interpolation resample to `dst` Hz (adequate for the mel-conditioning path; exact soxr
/// parity is not required for the flow's prompt mel).
pub fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = dst_rate as f64 / src_rate as f64;
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

/// The S3Gen 24 kHz log-mel extractor: precomputed Slaney mel bank, periodic Hann window, and a
/// direct real-DFT (`n_fft = 1920` is not a power of two).
pub struct Mel24Extractor {
    cfg: S3GenConfig,
    mel_bank: Vec<f32>, // mel-major [n_mels][n_bins]
    window: Vec<f32>,   // periodic Hann, length win == n_fft
    cos_tab: Vec<f32>,  // [n_bins][n_fft]
    sin_tab: Vec<f32>,
}

impl Default for Mel24Extractor {
    fn default() -> Self {
        Self::new(&S3GenConfig::DEFAULT)
    }
}

impl Mel24Extractor {
    /// Build the extractor for a given [`S3GenConfig`] (uses its `mel_*` fields).
    pub fn new(cfg: &S3GenConfig) -> Self {
        let n_fft = cfg.mel_n_fft;
        let n_bins = n_fft / 2 + 1;
        let mut cos_tab = vec![0f32; n_bins * n_fft];
        let mut sin_tab = vec![0f32; n_bins * n_fft];
        for k in 0..n_bins {
            for t in 0..n_fft {
                let ang = -2.0 * std::f32::consts::PI * k as f32 * t as f32 / n_fft as f32;
                cos_tab[k * n_fft + t] = ang.cos();
                sin_tab[k * n_fft + t] = ang.sin();
            }
        }
        Self {
            cfg: *cfg,
            mel_bank: librosa_mel_bank(cfg),
            window: hann_periodic(cfg.mel_win),
            cos_tab,
            sin_tab,
        }
    }

    /// Number of mel frames a `len`-sample (post-resample) 24 kHz clip yields.
    pub fn num_frames(&self, len: usize) -> usize {
        let pad = (self.cfg.mel_n_fft - self.cfg.mel_hop) / 2;
        let padded = len + 2 * pad;
        if padded < self.cfg.mel_n_fft {
            return 0;
        }
        1 + (padded - self.cfg.mel_n_fft) / self.cfg.mel_hop
    }

    /// Extract the log-mel of a waveform, resampling to 24 kHz if needed → `[n_frames, 80]`
    /// (matching the reference `mel_spectrogram(...).transpose(1, 2)` layout the flow's
    /// `prompt_feat` expects).
    pub fn mel(&self, samples: &[f32], sample_rate: u32, device: &Device) -> CandleResult<Tensor> {
        let wav = resample_linear(samples, sample_rate, S3GEN_SR);
        let (frames, n_frames) = self.log_mel_frame_major(&wav);
        Tensor::from_vec(frames, (n_frames, self.cfg.mel_num_mels), device)
    }

    /// The log-mel as frame-major `[n_frames * n_mels]` plus the frame count. Split out so the
    /// numeric path is unit-testable without a tensor.
    fn log_mel_frame_major(&self, wav: &[f32]) -> (Vec<f32>, usize) {
        let cfg = &self.cfg;
        let (n_fft, hop, n_mels) = (cfg.mel_n_fft, cfg.mel_hop, cfg.mel_num_mels);
        let n_bins = n_fft / 2 + 1;
        let pad = (n_fft - hop) / 2;
        let padded = reflect_pad(wav, pad);
        if padded.len() < n_fft {
            return (Vec::new(), 0);
        }
        let n_frames = 1 + (padded.len() - n_fft) / hop;
        let mut out = vec![0f32; n_frames * n_mels]; // frame-major [f * n_mels + m]
        let mut windowed = vec![0f32; n_fft];
        let mut mag = vec![0f32; n_bins];
        for f in 0..n_frames {
            let start = f * hop;
            for (i, w) in windowed.iter_mut().enumerate() {
                *w = padded[start + i] * self.window[i];
            }
            for (k, slot) in mag.iter_mut().enumerate() {
                let (cos_k, sin_k) = (&self.cos_tab[k * n_fft..], &self.sin_tab[k * n_fft..]);
                let mut re = 0f32;
                let mut im = 0f32;
                for (t, &x) in windowed.iter().enumerate() {
                    re += x * cos_k[t];
                    im += x * sin_k[t];
                }
                // sqrt(|.|^2 + 1e-9) — the reference's magnitude (not power).
                *slot = (re * re + im * im + 1e-9).sqrt();
            }
            for m in 0..n_mels {
                let filt = &self.mel_bank[m * n_bins..(m + 1) * n_bins];
                let mut acc = 0f32;
                for (k, &wgt) in filt.iter().enumerate() {
                    acc += wgt * mag[k];
                }
                out[f * n_mels + m] = acc.max(1e-5).ln();
            }
        }
        (out, n_frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> S3GenConfig {
        S3GenConfig::DEFAULT
    }

    #[test]
    fn slaney_mel_scale_roundtrips_and_is_monotone() {
        for hz in [0.0, 100.0, 700.0, 1000.0, 4000.0, 8000.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!((back - hz).abs() < 1e-6, "hz→mel→hz for {hz}");
        }
        // Anchored at the 1 kHz linear/log break: mel(1000) = 15.0 exactly (Slaney).
        assert!((hz_to_mel(1000.0) - 15.0).abs() < 1e-9);
        assert!(hz_to_mel(8000.0) > hz_to_mel(1000.0));
    }

    #[test]
    fn mel_bank_shape_nonneg_and_covers_the_band() {
        let c = cfg();
        let bank = librosa_mel_bank(&c);
        let n_bins = c.mel_n_fft / 2 + 1;
        assert_eq!(bank.len(), c.mel_num_mels * n_bins);
        assert_eq!(n_bins, 961);
        for m in 0..c.mel_num_mels {
            let row = &bank[m * n_bins..(m + 1) * n_bins];
            assert!(row.iter().all(|&w| w >= 0.0));
            assert!(row.iter().any(|&w| w > 0.0), "mel bin {m} is all-zero");
        }
    }

    #[test]
    fn hann_is_periodic_and_bounded() {
        let w = hann_periodic(1920);
        assert_eq!(w.len(), 1920);
        assert!(w[0].abs() < 1e-6);
        assert!((w[960] - 1.0).abs() < 1e-6, "peak at N/2");
        assert!(w.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn frame_count_is_50hz_and_finite_mel() {
        let ext = Mel24Extractor::default();
        // 1 s of a 220 Hz tone at 24 kHz → ~50 mel frames (50 Hz frame rate).
        let sr = S3GEN_SR as usize;
        let tone: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sr as f32).sin() * 0.3)
            .collect();
        let (frames, n) = ext.log_mel_frame_major(&tone);
        assert_eq!(n, ext.num_frames(sr));
        assert!((n as i64 - 50).abs() <= 1, "expected ≈50 frames, got {n}");
        assert_eq!(frames.len(), n * 80);
        assert!(frames.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn resample_changes_length_by_ratio_and_is_identity_at_rate() {
        let s = vec![0.1f32, -0.2, 0.3, -0.4];
        assert_eq!(resample_linear(&s, S3GEN_SR, S3GEN_SR), s);
        let out = resample_linear(&vec![0.0f32; 16_000], 16_000, 24_000);
        assert_eq!(out.len(), 24_000);
    }
}
