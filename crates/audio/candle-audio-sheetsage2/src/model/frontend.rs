//! The MERT2 mel front end (`MERT2MelFrontend`): torchaudio `Spectrogram(power=2)` with the
//! checkpoint's own Hann window (`center=True`, reflect padding), `MelScale` with the checkpoint's
//! filterbank, `AmplitudeToDB(top_db=None)`, the last frame dropped, then per-bin normalization with
//! the checkpoint's `mel_mean` / `mel_std`. Always float32, as upstream forces it.
//!
//! The STFT runs on the host with precomputed twiddles, split across threads; the filterbank
//! product and everything after it run on the model's device.

use candle_audio::candle_core::{DType, Device, Tensor, D};

use crate::Error;

/// A radix-2 complex FFT plan with precomputed twiddles.
struct FftPlan {
    n: usize,
    twiddles: Vec<(f32, f32)>,
    bitrev: Vec<usize>,
}

impl FftPlan {
    fn new(n: usize) -> Self {
        let bits = n.trailing_zeros();
        let bitrev = (0..n)
            .map(|i| i.reverse_bits() >> (usize::BITS - bits))
            .collect();
        let twiddles = (0..n / 2)
            .map(|k| {
                let angle = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                (angle.cos() as f32, angle.sin() as f32)
            })
            .collect();
        Self {
            n,
            twiddles,
            bitrev,
        }
    }

    fn forward(&self, data: &mut [(f32, f32)]) {
        let n = self.n;
        for i in 0..n {
            let j = self.bitrev[i];
            if i < j {
                data.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let stride = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let (wr, wi) = self.twiddles[k * stride];
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
}

/// `torch.stft(center=True, pad_mode="reflect", onesided=True).abs().pow(2)` → power spectrogram,
/// frame-major `[n_frames][n_fft/2 + 1]`.
pub fn power_spectrogram(
    samples: &[f32],
    n_fft: usize,
    hop: usize,
    window: &[f32],
) -> Result<(Vec<f32>, usize), Error> {
    if !n_fft.is_power_of_two() || n_fft < 2 {
        return Err(Error::Config(format!(
            "n_fft {n_fft} must be a power of two"
        )));
    }
    if window.len() > n_fft || window.is_empty() || hop == 0 {
        return Err(Error::Config(
            "window must be 1..=n_fft samples and hop > 0".into(),
        ));
    }
    let pad = n_fft / 2;
    if samples.len() <= pad {
        return Err(Error::Request(format!(
            "reflect padding of {pad} needs more than {pad} samples, got {}",
            samples.len()
        )));
    }
    // torch pads a shorter window with zeros on both sides to n_fft.
    let mut full_window = vec![0.0f32; n_fft];
    let left = (n_fft - window.len()) / 2;
    full_window[left..left + window.len()].copy_from_slice(window);
    let mut padded = Vec::with_capacity(samples.len() + 2 * pad);
    padded.extend((1..=pad).rev().map(|i| samples[i]));
    padded.extend_from_slice(samples);
    padded.extend((0..pad).map(|i| samples[samples.len() - 2 - i]));
    let n_frames = 1 + (padded.len() - n_fft) / hop;
    let n_bins = n_fft / 2 + 1;
    let plan = FftPlan::new(n_fft);
    let mut out = vec![0.0f32; n_frames * n_bins];
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .clamp(1, 16);
    let per_thread = n_frames.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in out.chunks_mut(per_thread * n_bins).enumerate() {
            let (plan, padded, window) = (&plan, &padded, &full_window);
            scope.spawn(move || {
                let mut buffer = vec![(0.0f32, 0.0f32); n_fft];
                for (local, row) in chunk.chunks_mut(n_bins).enumerate() {
                    let start = (chunk_index * per_thread + local) * hop;
                    for (slot, (x, w)) in buffer
                        .iter_mut()
                        .zip(padded[start..start + n_fft].iter().zip(window))
                    {
                        *slot = (x * w, 0.0);
                    }
                    plan.forward(&mut buffer);
                    for (value, &(re, im)) in row.iter_mut().zip(&buffer[..n_bins]) {
                        let magnitude = re.hypot(im);
                        *value = magnitude * magnitude;
                    }
                }
            });
        }
    });
    Ok((out, n_frames))
}

/// The front end's checkpoint buffers.
#[derive(Clone, Debug)]
pub struct MelFrontend {
    window: Vec<f32>,
    /// `[n_bins, n_mels]`.
    filterbank: Tensor,
    mean: Tensor,
    std: Tensor,
    n_fft: usize,
    hop: usize,
}

impl MelFrontend {
    /// From the checkpoint's `feature_extractor.*` buffers (all float32).
    pub fn new(
        window: &Tensor,
        filterbank: &Tensor,
        mean: &Tensor,
        std: &Tensor,
        n_fft: usize,
        hop: usize,
        device: &Device,
    ) -> Result<Self, Error> {
        let window = window.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let (bins, _) = filterbank.dims2()?;
        if bins != n_fft / 2 + 1 {
            return Err(Error::Config(format!(
                "mel filterbank has {bins} rows for n_fft {n_fft}"
            )));
        }
        Ok(Self {
            window,
            filterbank: filterbank.to_dtype(DType::F32)?.to_device(device)?,
            mean: mean.to_dtype(DType::F32)?.to_device(device)?,
            // std.clamp_min(1e-5)
            std: std
                .to_dtype(DType::F32)?
                .clamp(1e-5f32, f32::MAX)?
                .to_device(device)?,
            n_fft,
            hop,
        })
    }

    /// Normalized log-mel frames `[1, frames - 1, n_mels]` of one mono waveform.
    pub fn forward(&self, samples: &[f32]) -> Result<Tensor, Error> {
        let (power, frames) = power_spectrogram(samples, self.n_fft, self.hop, &self.window)?;
        let device = self.filterbank.device();
        let bins = self.n_fft / 2 + 1;
        let power = Tensor::from_vec(power, (frames, bins), device)?;
        let mel = power.matmul(&self.filterbank)?;
        // AmplitudeToDB(power, top_db=None): 10 * log10(max(x, 1e-10)).
        let db = (mel.clamp(1e-10f32, f32::MAX)?.log()? * (10.0 / std::f64::consts::LN_10))?;
        let db = db.narrow(0, 0, frames - 1)?;
        let normalized = db.broadcast_sub(&self.mean)?.broadcast_div(&self.std)?;
        Ok(normalized.unsqueeze(0)?)
    }

    /// Mel bins.
    pub fn n_mels(&self) -> Result<usize, Error> {
        Ok(self.filterbank.dim(D::Minus1)?)
    }
}
