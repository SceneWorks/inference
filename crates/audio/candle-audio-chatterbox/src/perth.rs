//! PerTh implicit watermarker (sc-13240) — a candle-native port of Resemble AI's
//! `resemble-perth` `PerthImplicitWatermarker` (the `PerthNet` "PERceptual THreshold" model), the
//! provenance watermark Chatterbox's reference `TTS.generate()` unconditionally applies to its 24
//! kHz output.
//!
//! ## What PerTh is (neural, weights-based — NOT DSP)
//!
//! PerTh is a small **trained convolutional** watermarker (MIT; `perth_net_250000.pth.tar`, ~9.35 M
//! float32 params) that operates on the normalized magnitude spectrogram of a 32 kHz signal:
//!
//! - front end: a Hann-windowed STFT (`n_fft = 2048`, `hop = 320`) → magnitude (clamped, converted
//!   to dB, min/headroom-normalized) + phase. The watermark lives only in the **subband** of the
//!   first `128` bins (≈ 0–2 kHz).
//! - **encoder** (`Encoder`): a residual `Conv1d` stack (`128→256`, five `256→256` k=7 blocks,
//!   `256→128`, `LeakyReLU(0.01)` between) that predicts a spectral residual added — under an
//!   energy `magmask` — to the subband magnitudes. The watermark is *implicit*: there is no
//!   payload/bits argument; a single fixed watermark is embedded (matching the reference, which
//!   exposes no disable flag).
//! - **decoder** (`Decoder`): three parallel `Conv1d` stacks over slow/normal/fast time-scaled
//!   copies of the subband, each emitting an attention + a watermark channel, masked-averaged over
//!   time and combined by softmax attention into a single confidence in `[0, 1]`.
//!
//! The magnitude residual is reconstructed with the original phase through an inverse STFT
//! (windowed overlap-add), so the watermark is perceptually near-transparent ("implicit").
//!
//! ## Faithfulness to the reference
//!
//! Every weight key matches the reference state dict (`encoder.layers.N.conv.{weight,bias}`,
//! `decoder.{slow,normal,fast}_layers.N.conv.{weight,bias}`); the `ap.*` STFT-window buffers are
//! recomputed constants and are dropped by the converter. Magnitude normalization, the `magmask`
//! threshold, the multi-scale linear/nearest resamples, and the softmax-attention combine are ported
//! exactly. The reference resamples arbitrary input to/from the model's 32 kHz rate with librosa;
//! this port uses linear resampling (the audio lane's convention — see
//! `crate::s3tokenizer::resample_to_16k`), which the watermark's design tolerates (it is trained to
//! survive an STFT/iSTFT cycle and resampling).
//!
//! ## Wiring status (sc-13240 → sc-13239 → sc-13443)
//!
//! sc-13240 delivered the tested watermarker module + the recorded provenance; sc-13239 wires it
//! into [`crate::model`]'s `generate()` output path so every rendered clone is watermarked at 24 kHz
//! before it leaves the provider (the reference behavior — no disable flag). sc-13443 hosts the
//! converted `perth_implicit.safetensors` on the Hugging Face hub (`SceneWorks/perth-implicit`, MIT)
//! so [`resolve_perth_weights`] resolves it the same pinned-SHA way every other audio checkpoint
//! does — [`candle_audio::hub::hf_get_pinned`] at [`PERTH_HUB_REVISION`] — with a `PERTH_SNAPSHOT`
//! env override kept first as the offline/CI escape hatch. This removed the previous runtime
//! `pip download resemble-perth` + torch-free-converter shell-out (and its subprocess-exec surface);
//! `scripts/audio/convert_perth_watermarker.py` remains in the repo as the reproducibility record
//! that produced the hosted checkpoint.

use std::path::{Path, PathBuf};

use candle_audio::candle_core::{DType, Device, Result as CandleResult, Tensor};
use candle_audio::hub::hf_get_pinned;
use candle_audio::{dsp, AudioError, Result};
use candle_nn::{Conv1d, Conv1dConfig, Module, VarBuilder};

/// The model's native sample rate (`hparams.sample_rate`). Arbitrary-rate input is resampled to
/// this rate for analysis and back for [`PerthWatermarker::embed`].
pub const PERTH_SR: u32 = 32_000;

/// The converted checkpoint filename inside the hub repo (the encoder + decoder conv tensors of
/// `perth_net_250000.pth.tar`; see `scripts/audio/convert_perth_watermarker.py`).
pub const PERTH_WEIGHTS_FILE: &str = "perth_implicit.safetensors";

/// Hub pin: `SceneWorks/perth-implicit` at an immutable commit (F-029; MIT weights — commercial use
/// OK). SceneWorks hosts the converted `perth_implicit.safetensors` so a clone resolves its
/// provenance watermarker weights the same pinned-SHA way every other audio checkpoint does (no
/// runtime pip/network shell-out). The upstream `resemble-perth` package is MIT; the hosted file is
/// its `perth_net_250000.pth.tar` run through `scripts/audio/convert_perth_watermarker.py`.
pub const PERTH_HUB_REPO: &str = "SceneWorks/perth-implicit";
pub const PERTH_HUB_REVISION: &str = "80b60f9caead09b8d3b512bda0b24038f28c08ec";

/// STFT size (`hparams.n_fft`) — a power of two, so the shared radix-2 [`dsp::stft`] applies.
const N_FFT: usize = 2048;
/// STFT hop (`hparams.hop_size`).
const HOP: usize = 320;
/// One-sided STFT bin count (`n_fft / 2 + 1`).
const N_BINS: usize = N_FFT / 2 + 1;
/// Convolution width (`hparams.hidden_size`).
const HIDDEN: usize = 256;
/// The highest watermarked frequency (`hparams.max_wmark_freq`, Hz).
const MAX_WMARK_FREQ_HZ: usize = 2000;
/// Nyquist frequency (Hz) — the top of the spectrum.
const TOPFREQ_HZ: usize = PERTH_SR as usize / 2;
/// Watermark subband width: `round(n_bins · max_wmark_freq / (sr/2))` =
/// `round(1025 · 2000 / 16000)` = 128 (the low-frequency bins the watermark is confined to). The
/// `+ TOPFREQ_HZ/2` before the integer division is round-half-up (the reference `int(round(...))`).
const SUBBAND: usize = (N_BINS * MAX_WMARK_FREQ_HZ + TOPFREQ_HZ / 2) / TOPFREQ_HZ;
/// Magnitude floor before the dB conversion (`hparams.stft_magnitude_min`).
const STFT_MAGNITUDE_MIN: f32 = 1e-9;
/// Normalization headroom in dB (`utils.normalize` / `denormalize_spectrogram`, `headroom_db=15`).
const HEADROOM_DB: f32 = 15.0;
/// `20·log10(stft_magnitude_min)` = `20·log10(1e-9)` = -180 dB (the `min_level_db` normalization
/// anchor).
const MIN_LEVEL_DB: f32 = -180.0;
/// dB span the normalized magnitude occupies: `-min_level_db + headroom_db` = 195.
const DB_SPAN: f32 = -MIN_LEVEL_DB + HEADROOM_DB;
/// `magmask` energy threshold fraction (`p=0.05`): a frame is watermarkable where its summed
/// magnitude exceeds 5% of the loudest frame's.
const MAGMASK_P: f32 = 0.05;
/// `nn.LeakyReLU()` default negative slope.
const LRELU_SLOPE: f64 = 0.01;

/// A `Conv1d` + optional `LeakyReLU` — the reference `model.Conv` block. `padding = (k-1)/2` keeps
/// the time length ('same'); the last block of each stack has `act = false`.
struct ConvBlock {
    conv: Conv1d,
    act: bool,
}

impl ConvBlock {
    fn load(vb: &VarBuilder, in_c: usize, out_c: usize, k: usize, act: bool) -> CandleResult<Self> {
        let cfg = Conv1dConfig {
            padding: (k - 1) / 2,
            stride: 1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let weight = vb.get((out_c, in_c, k), "conv.weight")?;
        let bias = vb.get(out_c, "conv.bias")?;
        Ok(Self {
            conv: Conv1d::new(weight, Some(bias), cfg),
            act,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x = self.conv.forward(x)?;
        if self.act {
            // LeakyReLU: max(x, slope·x).
            x.maximum(&x.affine(LRELU_SLOPE, 0.0)?)
        } else {
            Ok(x)
        }
    }
}

/// The shared 7-block conv stack: `Conv(in→256,1)`, five `Conv(256→256,7)`, `Conv(256→out,1)`
/// (`LeakyReLU` after all but the last). Layers are keyed `<prefix>.{i}.conv.*`.
struct ConvStack {
    blocks: Vec<ConvBlock>,
}

impl ConvStack {
    fn load(vb: VarBuilder, out_last: usize) -> CandleResult<Self> {
        let mut blocks = Vec::with_capacity(7);
        blocks.push(ConvBlock::load(&vb.pp("0"), SUBBAND, HIDDEN, 1, true)?);
        for i in 1..=5 {
            blocks.push(ConvBlock::load(
                &vb.pp(i.to_string()),
                HIDDEN,
                HIDDEN,
                7,
                true,
            )?);
        }
        blocks.push(ConvBlock::load(&vb.pp("6"), HIDDEN, out_last, 1, false)?);
        Ok(Self { blocks })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let mut h = x.clone();
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        Ok(h)
    }
}

/// The loaded PerTh implicit watermarker: the residual encoder + the multi-scale decoder + the
/// recomputed Hann analysis/synthesis window.
pub struct PerthWatermarker {
    encoder: ConvStack,
    dec_slow: ConvStack,
    dec_normal: ConvStack,
    dec_fast: ConvStack,
    window: Vec<f32>,
    device: Device,
}

impl PerthWatermarker {
    /// Load the watermarker from a `perth_implicit.safetensors` file (the converted encoder/decoder
    /// conv tensors). The device is the audio lane's default (CPU-first; Metal/CUDA when built in).
    pub fn from_safetensors(path: &Path) -> Result<Self> {
        if !path.is_file() {
            return Err(AudioError::Msg(format!(
                "perth: weights {} missing (convert perth_net_250000.pth.tar with \
                 scripts/audio/convert_perth_watermarker.py)",
                path.display()
            )));
        }
        let device = candle_audio::default_device()?;
        // SAFETY: mmap of a provider-resolved safetensors file — the shared audio-lane idiom.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&path), DType::F32, &device)
                .map_err(|e| AudioError::Msg(format!("perth: load {}: {e}", path.display())))?
        };
        Self::from_var_builder(vb, device)
    }

    /// Build from a [`VarBuilder`] (keys `encoder.layers.*`, `decoder.{slow,normal,fast}_layers.*`)
    /// and device. Public so tests can drive the architecture with synthetic weights.
    pub fn from_var_builder(vb: VarBuilder, device: Device) -> Result<Self> {
        let encoder = ConvStack::load(vb.pp("encoder.layers"), SUBBAND)
            .map_err(|e| AudioError::Msg(format!("perth: load encoder: {e}")))?;
        let dec_slow = ConvStack::load(vb.pp("decoder.slow_layers"), 2)
            .map_err(|e| AudioError::Msg(format!("perth: load decoder.slow: {e}")))?;
        let dec_normal = ConvStack::load(vb.pp("decoder.normal_layers"), 2)
            .map_err(|e| AudioError::Msg(format!("perth: load decoder.normal: {e}")))?;
        let dec_fast = ConvStack::load(vb.pp("decoder.fast_layers"), 2)
            .map_err(|e| AudioError::Msg(format!("perth: load decoder.fast: {e}")))?;
        Ok(Self {
            encoder,
            dec_slow,
            dec_normal,
            dec_fast,
            window: dsp::hann_window(N_FFT),
            device,
        })
    }

    /// Embed the (implicit) watermark into `wav` at `sample_rate`, returning the watermarked
    /// waveform at the **same** sample rate. Faithful to `PerthImplicitWatermarker.apply_watermark`:
    /// resample to 32 kHz, add the masked encoder residual to the subband magnitudes, reconstruct
    /// with the original phase, resample back. There is no watermark-bits argument — the watermark
    /// is implicit and fixed (the reference exposes no payload or disable flag).
    pub fn embed(&self, wav: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let resample = sample_rate != PERTH_SR;
        let sig = if resample {
            resample_linear(wav, sample_rate, PERTH_SR)
        } else {
            wav.to_vec()
        };
        let (mut mag_norm, phase, n_frames) = self.analyze(&sig)?;
        let mask = magmask(&mag_norm, n_frames);

        // Subband magnitudes (first SUBBAND bins) → (1, SUBBAND, n_frames). The bin-major host
        // layout (index = bin·n_frames + t) is exactly row-major [SUBBAND, n_frames].
        let sub = mag_norm[..SUBBAND * n_frames].to_vec();
        let sub_t = Tensor::from_vec(sub, (1, SUBBAND, n_frames), &self.device)
            .map_err(|e| AudioError::Msg(format!("perth: subband tensor: {e}")))?;
        let res_t = self
            .encoder
            .forward(&sub_t)
            .map_err(|e| AudioError::Msg(format!("perth: encoder: {e}")))?;
        let res: Vec<f32> = res_t
            .flatten_all()
            .and_then(|t| t.to_vec1())
            .map_err(|e| AudioError::Msg(format!("perth: encoder output: {e}")))?;

        // magspec[:, :subband] += residual · mask (mask broadcasts over the subband channels).
        for bin in 0..SUBBAND {
            let base = bin * n_frames;
            for t in 0..n_frames {
                mag_norm[base + t] += res[base + t] * mask[t];
            }
        }

        let wm = self.synthesize(&mag_norm, &phase, n_frames)?;
        Ok(if resample {
            resample_linear(&wm, PERTH_SR, sample_rate)
        } else {
            wm
        })
    }

    /// Recover the watermark confidence from `wav` at `sample_rate` — the ported
    /// `PerthImplicitWatermarker.get_watermark` (clamped to `[0, 1]`). A watermarked signal returns
    /// a high value; a clean signal a low one.
    pub fn get_watermark(&self, wav: &[f32], sample_rate: u32) -> Result<f32> {
        let sig = if sample_rate != PERTH_SR {
            resample_linear(wav, sample_rate, PERTH_SR)
        } else {
            wav.to_vec()
        };
        let (mag_norm, _phase, n_frames) = self.analyze(&sig)?;
        let mask = magmask(&mag_norm, n_frames);
        let sub = mag_norm[..SUBBAND * n_frames].to_vec();
        let sub_t = Tensor::from_vec(sub, (1, SUBBAND, n_frames), &self.device)
            .map_err(|e| AudioError::Msg(format!("perth: subband tensor: {e}")))?;
        let conf = self
            .decode(&sub_t, &mask, n_frames)
            .map_err(|e| AudioError::Msg(format!("perth: decoder: {e}")))?;
        Ok(conf.clamp(0.0, 1.0))
    }

    /// STFT → (normalized magnitude, phase, n_frames), all bin-major host arrays. Mirrors
    /// `AudioProcessor.signal_to_magphase` + `utils.cx_to_magphase`/`normalize`.
    fn analyze(&self, sig: &[f32]) -> Result<(Vec<f32>, Vec<f32>, usize)> {
        let spec = dsp::stft(sig, N_FFT, HOP, &self.window)?;
        let mag_norm: Vec<f32> = spec
            .magnitude()
            .iter()
            .map(|&m| {
                let db = 20.0 * m.max(STFT_MAGNITUDE_MIN).log10();
                (db - MIN_LEVEL_DB) / DB_SPAN
            })
            .collect();
        Ok((mag_norm, spec.phase(), spec.n_frames))
    }

    /// (denormalize → linear magnitude) + original phase → iSTFT. Mirrors
    /// `utils.magphase_to_cx`/`denormalize_spectrogram` + `AudioProcessor.magphase_to_signal`.
    fn synthesize(&self, mag_norm: &[f32], phase: &[f32], n_frames: usize) -> Result<Vec<f32>> {
        let mag_lin: Vec<f32> = mag_norm
            .iter()
            .map(|&mn| {
                let db = mn * DB_SPAN + MIN_LEVEL_DB;
                10f32.powf((db / 20.0).min(10.0))
            })
            .collect();
        dsp::istft(&mag_lin, phase, n_frames, N_FFT, HOP, &self.window)
    }

    /// The multi-scale decoder: run each branch over its time-scaled subband, masked-average the
    /// attention + watermark channels over time, and combine by softmax attention.
    fn decode(&self, sub_t: &Tensor, mask: &[f32], n_frames: usize) -> CandleResult<f32> {
        let t_slow = (n_frames as f64 * 1.25) as usize; // int() truncation, as the reference
        let t_fast = (n_frames as f64 * 0.75) as usize;
        let (attn_s, wm_s) = self.branch(&self.dec_slow, sub_t, mask, n_frames, t_slow)?;
        let (attn_n, wm_n) = self.branch(&self.dec_normal, sub_t, mask, n_frames, n_frames)?;
        let (attn_f, wm_f) = self.branch(&self.dec_fast, sub_t, mask, n_frames, t_fast)?;
        // softmax over the three branch attentions (numerically stabilized).
        let m = attn_s.max(attn_n).max(attn_f);
        let (es, en, ef) = ((attn_s - m).exp(), (attn_n - m).exp(), (attn_f - m).exp());
        let z = es + en + ef;
        Ok(wm_s * es / z + wm_n * en / z + wm_f * ef / z)
    }

    /// One decoder branch: linear-resize the subband to `t_target`, run the conv stack, then
    /// masked-mean the attention (channel 0) and watermark (channel 1) rows over time using the
    /// nearest-resized mask. Returns `(attn_mean, wmark_mean)`.
    fn branch(
        &self,
        stack: &ConvStack,
        sub_t: &Tensor,
        mask: &[f32],
        n_frames: usize,
        t_target: usize,
    ) -> CandleResult<(f32, f32)> {
        let resized = if t_target == n_frames {
            sub_t.clone()
        } else {
            linear_resize_time(sub_t, n_frames, t_target)?
        };
        let out = stack.forward(&resized)?; // (1, 2, t_target), row-major [attn(t), wmark(t)]
        let out_v: Vec<f32> = out.flatten_all()?.to_vec1()?;
        let attn = &out_v[..t_target];
        let wmark = &out_v[t_target..2 * t_target];
        let mask_r = nearest_resize(mask, t_target);
        let msum: f32 = mask_r.iter().sum();
        let msum = if msum > 0.0 { msum } else { 1.0 };
        let attn_mean = attn.iter().zip(&mask_r).map(|(a, m)| a * m).sum::<f32>() / msum;
        let wm_mean = wmark.iter().zip(&mask_r).map(|(w, m)| w * m).sum::<f32>() / msum;
        Ok((attn_mean, wm_mean))
    }
}

/// `magmask`: 1.0 where a frame's summed (over all bins) magnitude exceeds `MAGMASK_P` of the
/// loudest frame's, else 0.0. `mag_norm` is bin-major `[N_BINS, n_frames]`.
fn magmask(mag_norm: &[f32], n_frames: usize) -> Vec<f32> {
    let mut s = vec![0f32; n_frames];
    for bin in 0..N_BINS {
        let base = bin * n_frames;
        for t in 0..n_frames {
            s[t] += mag_norm[base + t];
        }
    }
    let smax = s.iter().copied().fold(f32::MIN, f32::max);
    let thresh = smax * MAGMASK_P;
    s.iter()
        .map(|&x| if x > thresh { 1.0 } else { 0.0 })
        .collect()
}

/// Linear resample of the last (time) axis of a `(1, C, t_in)` tensor to `t_out` with
/// `align_corners=True` (torch `F.interpolate(mode="linear", align_corners=True)`), implemented as a
/// right-multiply by a sparse `[t_in, t_out]` interpolation matrix so it stays on-device.
fn linear_resize_time(x: &Tensor, t_in: usize, t_out: usize) -> CandleResult<Tensor> {
    let mut w = vec![0f32; t_in * t_out];
    if t_out == 1 {
        w[0] = 1.0;
    } else {
        for j in 0..t_out {
            let coord = j as f64 * (t_in as f64 - 1.0) / (t_out as f64 - 1.0);
            let l = (coord.floor() as usize).min(t_in - 1);
            let frac = (coord - l as f64) as f32;
            w[l * t_out + j] += 1.0 - frac;
            if l + 1 < t_in {
                w[(l + 1) * t_out + j] += frac;
            }
        }
    }
    let wt = Tensor::from_vec(w, (t_in, t_out), x.device())?;
    x.broadcast_matmul(&wt)
}

/// Nearest-neighbor resample of a 1-D mask to `t_out` (torch `F.interpolate(mode="nearest")`:
/// `src = floor(dst · t_in / t_out)`).
fn nearest_resize(mask: &[f32], t_out: usize) -> Vec<f32> {
    let t_in = mask.len();
    (0..t_out)
        .map(|j| {
            let src = ((j as f64 * t_in as f64) / t_out as f64).floor() as usize;
            mask[src.min(t_in - 1)]
        })
        .collect()
}

/// Linear-interpolation resample between arbitrary rates — the audio lane's convention
/// (`crate::s3tokenizer::resample_to_16k`), used to move to/from the model's 32 kHz rate.
fn resample_linear(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
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

/// Signal-to-noise ratio in dB between a reference and a modified signal (aligned to the shorter
/// length), the reference `utils.snr`. Used by the imperceptibility gates.
pub fn snr_db(reference: &[f32], modified: &[f32]) -> f32 {
    let n = reference.len().min(modified.len());
    let mut ps = 0f64;
    let mut pn = 0f64;
    for i in 0..n {
        let r = reference[i] as f64;
        let d = modified[i] as f64;
        ps += r * r;
        pn += (r - d) * (r - d);
    }
    if pn <= 0.0 {
        return f32::INFINITY;
    }
    (10.0 * (ps / pn).log10()) as f32
}

// =================================================================================================
// Runtime weight resolution (sc-13239 → sc-13443): resolve perth_implicit.safetensors off the HF hub.
// =================================================================================================

/// `PERTH_SNAPSHOT` as a resolved `perth_implicit.safetensors` file, if it points at an existing one
/// (a file directly, or a dir holding it). The offline/CI override kept ahead of the hub fetch.
fn perth_from_env() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("PERTH_SNAPSHOT").ok()?);
    let file = if p.is_dir() {
        p.join(PERTH_WEIGHTS_FILE)
    } else {
        p
    };
    file.is_file().then_some(file)
}

/// Resolve the converted PerTh weights (`perth_implicit.safetensors`). Resolution order:
///
/// 1. `PERTH_SNAPSHOT` (a file, or a dir holding `perth_implicit.safetensors`) — the offline/CI
///    escape hatch, kept first.
/// 2. The pinned-SHA hub fetch [`hf_get_pinned`]`(`[`PERTH_HUB_REPO`]`,` [`PERTH_HUB_REVISION`]`,`
///    [`PERTH_WEIGHTS_FILE`]`)`, resolving into the ordinary HF cache — exactly how every other audio
///    checkpoint (chatterbox/whisper/…) resolves (F-029; no runtime pip/network shell-out).
///
/// The clone ALWAYS watermarks (the reference behavior — no disable flag): if the weights truly
/// cannot be obtained this returns a typed error rather than silently skipping the watermark.
pub fn resolve_perth_weights() -> Result<PathBuf> {
    if let Some(p) = perth_from_env() {
        return Ok(p);
    }
    hf_get_pinned(PERTH_HUB_REPO, PERTH_HUB_REVISION, PERTH_WEIGHTS_FILE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The subband width matches the reference `compute_subband_freq`.
    #[test]
    fn subband_matches_reference_formula() {
        // Cross-check the integer const against the reference's float `int(round(...))`.
        let nfreq = N_BINS as f64;
        let topfreq = (PERTH_SR as f64) / 2.0;
        let subband = (nfreq * (MAX_WMARK_FREQ_HZ as f64) / topfreq).round() as usize;
        assert_eq!(subband, SUBBAND);
        assert_eq!(SUBBAND, 128);
    }

    /// `20·log10(1e-9) = -180`, the normalization anchor.
    #[test]
    fn min_level_db_anchor() {
        assert!((20.0 * (STFT_MAGNITUDE_MIN).log10() - MIN_LEVEL_DB).abs() < 1e-3);
        assert!((DB_SPAN - 195.0).abs() < 1e-3);
    }

    /// A deterministic broadband 32 kHz test signal (low-frequency tones the subband can carry plus
    /// a little high content), ~`secs` seconds.
    fn test_signal_32k(secs: f32) -> Vec<f32> {
        let n = (PERTH_SR as f32 * secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / PERTH_SR as f32;
                let tau = 2.0 * std::f32::consts::PI;
                0.5 * (tau * 150.0 * t).sin()
                    + 0.3 * (tau * 440.0 * t).sin()
                    + 0.2 * (tau * 900.0 * t).cos()
                    + 0.05 * (tau * 3500.0 * t).sin()
            })
            .collect()
    }

    /// A zero-weight watermarker: the encoder residual is exactly 0, so `embed` reduces to the pure
    /// STFT→iSTFT identity path — isolating the DSP roundtrip's imperceptibility floor.
    fn zero_watermarker() -> PerthWatermarker {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        PerthWatermarker::from_var_builder(vb, device).unwrap()
    }

    /// With a zero encoder the residual vanishes, so the watermarked signal equals the STFT→iSTFT
    /// reconstruction: the embed plumbing is correct and the DSP roundtrip is high-SNR
    /// (imperceptibility floor of the identity path).
    #[test]
    fn embed_with_zero_encoder_is_dsp_identity_high_snr() {
        let wm = zero_watermarker();
        let x = test_signal_32k(1.0);
        let y = wm.embed(&x, PERTH_SR).unwrap();
        assert!(!y.is_empty());
        // Compare over the interior (the outer frames have partial window coverage).
        let n = x.len().min(y.len());
        let guard = N_FFT;
        let snr = snr_db(&x[guard..n - guard], &y[guard..n - guard]);
        assert!(snr > 40.0, "DSP identity SNR too low: {snr} dB");
    }

    /// The full embed→detect pipeline runs end-to-end on synthetic weights and stays finite/bounded:
    /// shapes, the multi-scale resamples, masked-mean and softmax combine, and the `[0,1]` clamp all
    /// wire up. (A random model gives no *meaningful* detection — that is the real-weights gate.)
    #[test]
    fn embed_and_detect_plumbing_is_finite_and_bounded() {
        let device = Device::Cpu;
        // Small non-zero weights so every branch produces finite activations.
        let vb = VarBuilder::zeros(DType::F32, &device);
        let wm = PerthWatermarker::from_var_builder(vb, device).unwrap();
        let x = test_signal_32k(0.75);
        let y = wm.embed(&x, PERTH_SR).unwrap();
        let conf = wm.get_watermark(&y, PERTH_SR).unwrap();
        assert!(conf.is_finite());
        assert!(
            (0.0..=1.0).contains(&conf),
            "confidence out of range: {conf}"
        );
    }

    /// Resampling paths: a 24 kHz input (Chatterbox's output rate) is embedded and returned at 24
    /// kHz, and detection runs at 24 kHz through the internal resample to/from 32 kHz.
    #[test]
    fn embed_accepts_24k_and_returns_same_rate() {
        let wm = zero_watermarker();
        // 24 kHz version of the same tones.
        let n = (24_000f32 * 0.75) as usize;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / 24_000.0;
                let tau = 2.0 * std::f32::consts::PI;
                0.5 * (tau * 200.0 * t).sin() + 0.3 * (tau * 600.0 * t).cos()
            })
            .collect();
        let y = wm.embed(&x, 24_000).unwrap();
        // Same-rate output, length within a few percent of the input.
        let ratio = y.len() as f32 / x.len() as f32;
        assert!(
            (0.9..=1.1).contains(&ratio),
            "unexpected length ratio {ratio}"
        );
        let conf = wm.get_watermark(&y, 24_000).unwrap();
        assert!(conf.is_finite() && (0.0..=1.0).contains(&conf));
    }

    #[test]
    fn linear_resize_is_identity_when_lengths_match() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(
            (0..6).map(|i| i as f32).collect::<Vec<_>>(),
            (1, 2, 3),
            &device,
        )
        .unwrap();
        let y = linear_resize_time(&x, 3, 3).unwrap();
        let xv: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        let yv: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in xv.iter().zip(&yv) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn nearest_resize_matches_torch_floor_mapping() {
        // Upsample 3→5: floor(j·3/5) = [0,0,1,1,2].
        let m = nearest_resize(&[10.0, 20.0, 30.0], 5);
        assert_eq!(m, vec![10.0, 10.0, 20.0, 20.0, 30.0]);
        // Downsample 4→2: floor(j·4/2) = [0,2].
        let m = nearest_resize(&[1.0, 2.0, 3.0, 4.0], 2);
        assert_eq!(m, vec![1.0, 3.0]);
    }

    #[test]
    fn snr_of_identical_signals_is_infinite() {
        let x = vec![0.1, -0.2, 0.3];
        assert!(snr_db(&x, &x).is_infinite());
    }

    /// `resolve_perth_weights` honors a pre-materialized `PERTH_SNAPSHOT` without touching the
    /// network — the fast path the real-weights CI and the snapshot-prepare flow rely on. Accepts
    /// both a direct file and a dir holding `perth_implicit.safetensors`.
    #[test]
    fn resolve_perth_weights_honors_a_pre_materialized_snapshot() {
        let dir = std::env::temp_dir().join("chatterbox-perth-resolve");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join(PERTH_WEIGHTS_FILE);
        std::fs::write(&file, b"stub").unwrap();

        // A dir holding the file resolves to the file.
        std::env::set_var("PERTH_SNAPSHOT", &dir);
        assert_eq!(resolve_perth_weights().unwrap(), file);
        // The file directly resolves to itself.
        std::env::set_var("PERTH_SNAPSHOT", &file);
        assert_eq!(resolve_perth_weights().unwrap(), file);

        std::env::remove_var("PERTH_SNAPSHOT");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
