//! Short-time Fourier analysis/synthesis primitives for the candle audio providers
//! (sc-12835): Hann windowing, forward STFT, and the inverse-STFT overlap-add
//! reconstruction an iSTFT-Net-style vocoder head needs (Kokoro / StyleTTS2, sc-12836).
//!
//! Everything here is plain `f32` DSP with no tensor dependency: a provider's model
//! produces magnitude/phase (or mel) tensors on whatever candle device the bundle
//! selected, converts them to host `f32`, and reconstructs audio here — keeping the
//! numerics identical across CPU/Metal/CUDA and unit-testable without weights.
//!
//! Conventions (the librosa/torch `center=True` defaults the reference audio models
//! assume): frames are `n_fft`-long, spaced `hop` apart, over an input reflect-padded by
//! `n_fft / 2` on both ends; spectra carry the `n_fft / 2 + 1` one-sided bins; the
//! inverse normalizes by the summed squared window and trims the centering pad.

use crate::{AudioError, Result};

/// A periodic Hann window of length `len` — `0.5 * (1 - cos(2π n / N))`, the analysis and
/// synthesis window the reference STFT stacks (torch.hann_window / librosa) default to.
pub fn hann_window(len: usize) -> Vec<f32> {
    let n = len as f32;
    (0..len)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n).cos()))
        .collect()
}

/// A one-sided complex spectrogram: `n_bins = n_fft / 2 + 1` rows by `n_frames` columns,
/// stored bin-major (`index = bin * n_frames + frame`) — the layout of a `[n_bins,
/// n_frames]` model tensor's host copy, so provider code moves data without transposes.
#[derive(Clone, Debug)]
pub struct Spectrogram {
    pub n_bins: usize,
    pub n_frames: usize,
    /// Real parts, bin-major.
    pub re: Vec<f32>,
    /// Imaginary parts, bin-major.
    pub im: Vec<f32>,
}

impl Spectrogram {
    /// Per-bin magnitudes `sqrt(re² + im²)`, bin-major.
    pub fn magnitude(&self) -> Vec<f32> {
        self.re
            .iter()
            .zip(&self.im)
            .map(|(r, i)| (r * r + i * i).sqrt())
            .collect()
    }

    /// Per-bin phases `atan2(im, re)`, bin-major.
    pub fn phase(&self) -> Vec<f32> {
        self.re
            .iter()
            .zip(&self.im)
            .map(|(r, i)| i.atan2(*r))
            .collect()
    }
}

/// In-place iterative radix-2 Cooley–Tukey FFT over `(re, im)` pairs. `invert` runs the
/// inverse transform (without the `1/n` scale — callers apply it). `data.len()` must be a
/// power of two (checked by the public entry points).
fn fft_in_place(data: &mut [(f32, f32)], invert: bool) {
    let n = data.len();
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
            data.swap(i, j);
        }
    }
    let sign = if invert { 1.0f64 } else { -1.0f64 };
    let mut len = 2;
    while len <= n {
        // f64 twiddles: the recurrence-free per-butterfly angle keeps error flat over
        // long windows (n_fft up to 2048 for the mel front-ends).
        let ang = sign * 2.0 * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (wr, wi) = ((ang * k as f64).cos() as f32, (ang * k as f64).sin() as f32);
                let (ar, ai) = data[start + k];
                let (br, bi) = data[start + k + len / 2];
                let (tr, ti) = (br * wr - bi * wi, br * wi + bi * wr);
                data[start + k] = (ar + tr, ai + ti);
                data[start + k + len / 2] = (ar - tr, ai - ti);
            }
        }
        len <<= 1;
    }
}

fn require_power_of_two(n_fft: usize) -> Result<()> {
    if n_fft < 2 || !n_fft.is_power_of_two() {
        return Err(AudioError::Msg(format!(
            "n_fft {n_fft} must be a power of two >= 2 (radix-2 STFT)"
        )));
    }
    Ok(())
}

fn require_window(window: &[f32], n_fft: usize) -> Result<()> {
    if window.len() != n_fft {
        return Err(AudioError::Msg(format!(
            "window length {} does not match n_fft {n_fft}",
            window.len()
        )));
    }
    Ok(())
}

/// Real FFT of one `n_fft`-long frame → the `n_fft / 2 + 1` one-sided bins.
fn rfft(frame: &[f32]) -> Vec<(f32, f32)> {
    let n = frame.len();
    let mut data: Vec<(f32, f32)> = frame.iter().map(|&x| (x, 0.0)).collect();
    fft_in_place(&mut data, false);
    data.truncate(n / 2 + 1);
    data
}

/// Inverse real FFT of `n / 2 + 1` one-sided bins → an `n`-long real frame (with the
/// `1/n` scale applied), reconstructing the negative frequencies by conjugate symmetry.
fn irfft(bins: &[(f32, f32)], n: usize) -> Vec<f32> {
    let mut data: Vec<(f32, f32)> = Vec::with_capacity(n);
    data.extend_from_slice(bins);
    for k in (1..n / 2).rev() {
        let (r, i) = bins[k];
        data.push((r, -i));
    }
    fft_in_place(&mut data, true);
    let scale = 1.0 / n as f32;
    data.into_iter().map(|(r, _)| r * scale).collect()
}

/// Forward STFT with `center=True` reflect padding — the analysis half of the pair. The
/// output layout matches a `[n_bins, n_frames]` model tensor (see [`Spectrogram`]).
pub fn stft(samples: &[f32], n_fft: usize, hop: usize, window: &[f32]) -> Result<Spectrogram> {
    require_power_of_two(n_fft)?;
    require_window(window, n_fft)?;
    if hop == 0 {
        return Err(AudioError::Msg("hop must be >= 1".into()));
    }
    if samples.len() < 2 {
        return Err(AudioError::Msg(format!(
            "stft needs at least 2 samples to reflect-pad, got {}",
            samples.len()
        )));
    }
    let pad = n_fft / 2;
    if pad >= samples.len() {
        return Err(AudioError::Msg(format!(
            "stft reflect pad {pad} needs input longer than n_fft/2, got {} samples",
            samples.len()
        )));
    }
    // Reflect-pad by n_fft/2 on both ends (librosa `pad_mode="reflect"`).
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    padded.extend((1..=pad).rev().map(|i| samples[i]));
    padded.extend_from_slice(samples);
    padded.extend((0..pad).map(|i| samples[samples.len() - 2 - i]));

    let n_bins = n_fft / 2 + 1;
    let n_frames = 1 + (padded.len() - n_fft) / hop;
    let mut re = vec![0.0f32; n_bins * n_frames];
    let mut im = vec![0.0f32; n_bins * n_frames];
    let mut frame = vec![0.0f32; n_fft];
    for t in 0..n_frames {
        let start = t * hop;
        for (dst, (x, w)) in frame
            .iter_mut()
            .zip(padded[start..start + n_fft].iter().zip(window))
        {
            *dst = x * w;
        }
        for (bin, (r, i)) in rfft(&frame).into_iter().enumerate() {
            re[bin * n_frames + t] = r;
            im[bin * n_frames + t] = i;
        }
    }
    Ok(Spectrogram {
        n_bins,
        n_frames,
        re,
        im,
    })
}

/// Inverse STFT from bin-major magnitude + phase arrays (`[n_bins, n_frames]`, the host
/// copy of an iSTFT-Net head's output tensors) → time-domain samples. Windowed
/// overlap-add with summed-squared-window normalization, trimming the `center=True` pad —
/// the synthesis half of [`stft`] and the vocoder tail Kokoro's decoder needs (sc-12836).
pub fn istft(
    magnitude: &[f32],
    phase: &[f32],
    n_frames: usize,
    n_fft: usize,
    hop: usize,
    window: &[f32],
) -> Result<Vec<f32>> {
    require_power_of_two(n_fft)?;
    require_window(window, n_fft)?;
    if hop == 0 {
        return Err(AudioError::Msg("hop must be >= 1".into()));
    }
    let n_bins = n_fft / 2 + 1;
    if magnitude.len() != n_bins * n_frames || phase.len() != n_bins * n_frames {
        return Err(AudioError::Msg(format!(
            "istft expects bin-major [{n_bins}, {n_frames}] magnitude and phase \
             ({} values), got {} and {}",
            n_bins * n_frames,
            magnitude.len(),
            phase.len()
        )));
    }
    if n_frames == 0 {
        return Ok(Vec::new());
    }
    let out_len = n_fft + (n_frames - 1) * hop;
    let mut out = vec![0.0f32; out_len];
    let mut wsum = vec![0.0f32; out_len];
    let mut bins = vec![(0.0f32, 0.0f32); n_bins];
    for t in 0..n_frames {
        for (bin, dst) in bins.iter_mut().enumerate() {
            let m = magnitude[bin * n_frames + t];
            let p = phase[bin * n_frames + t];
            *dst = (m * p.cos(), m * p.sin());
        }
        let frame = irfft(&bins, n_fft);
        let start = t * hop;
        for (i, (x, w)) in frame.iter().zip(window).enumerate() {
            out[start + i] += x * w;
            wsum[start + i] += w * w;
        }
    }
    for (x, w) in out.iter_mut().zip(&wsum) {
        // Skip the (near-)zero-coverage edges rather than dividing by ~0.
        if *w > 1e-8 {
            *x /= *w;
        }
    }
    // Trim the center=True analysis pad so a stft→istft round trip aligns with the input.
    let pad = n_fft / 2;
    let end = out_len.saturating_sub(pad);
    Ok(out[pad.min(end)..end].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_shape_and_symmetry() {
        let w = hann_window(8);
        assert_eq!(w.len(), 8);
        assert!(w[0].abs() < 1e-7, "periodic Hann starts at 0");
        // Periodic symmetry: w[k] == w[N-k].
        for k in 1..8 {
            assert!((w[k] - w[8 - k]).abs() < 1e-6);
        }
        assert!((w[4] - 1.0).abs() < 1e-6, "peak at N/2");
    }

    #[test]
    fn rfft_matches_known_spectrum() {
        // A pure cosine at bin 1 of an 8-point frame: energy lands entirely in bin 1.
        let n = 8;
        let frame: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos())
            .collect();
        let bins = rfft(&frame);
        assert_eq!(bins.len(), 5);
        assert!((bins[1].0 - 4.0).abs() < 1e-4, "re[1] = n/2");
        for (k, (r, i)) in bins.iter().enumerate() {
            if k != 1 {
                assert!(r.abs() < 1e-4 && i.abs() < 1e-4, "bin {k} must be empty");
            }
        }
    }

    #[test]
    fn irfft_round_trips_rfft() {
        let frame: Vec<f32> = (0..64).map(|i| ((i * 7 % 13) as f32 - 6.0) / 6.0).collect();
        let back = irfft(&rfft(&frame), 64);
        for (a, b) in frame.iter().zip(&back) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn stft_istft_round_trip_reconstructs_signal() {
        // A deterministic multi-tone signal; hop = n_fft/4 gives full window coverage.
        let (n_fft, hop) = (256, 64);
        let signal: Vec<f32> = (0..4096)
            .map(|i| {
                let t = i as f32 / 4096.0;
                (2.0 * std::f32::consts::PI * 40.0 * t).sin()
                    + 0.5 * (2.0 * std::f32::consts::PI * 97.0 * t).cos()
            })
            .collect();
        let window = hann_window(n_fft);
        let spec = stft(&signal, n_fft, hop, &window).unwrap();
        assert_eq!(spec.n_bins, n_fft / 2 + 1);
        let out = istft(
            &spec.magnitude(),
            &spec.phase(),
            spec.n_frames,
            n_fft,
            hop,
            &window,
        )
        .unwrap();
        assert!(
            out.len() >= signal.len(),
            "{} < {}",
            out.len(),
            signal.len()
        );
        // Interior reconstruction error (the outermost hop on each side has partial
        // window coverage by construction).
        let mut worst = 0.0f32;
        for i in n_fft..signal.len() - n_fft {
            worst = worst.max((signal[i] - out[i]).abs());
        }
        assert!(worst < 1e-3, "worst interior error {worst}");
    }

    #[test]
    fn rejects_malformed_configs() {
        let w = hann_window(16);
        assert!(stft(&[0.0; 64], 12, 4, &hann_window(12)).is_err()); // not a power of two
        assert!(stft(&[0.0; 64], 16, 0, &w).is_err()); // zero hop
        assert!(stft(&[0.0; 64], 16, 4, &hann_window(8)).is_err()); // window mismatch
        assert!(stft(&[0.0; 4], 16, 4, &w).is_err()); // too short to reflect-pad
        assert!(istft(&[0.0; 8], &[0.0; 8], 1, 16, 4, &w).is_err()); // 8 != n_fft/2+1 bins
    }
}
