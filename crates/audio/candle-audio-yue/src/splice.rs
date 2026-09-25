//! **Seam: post-process** — the reference's `save_audio` limiter plus its low-band splice, which
//! replaces the 44.1 kHz Vocos mix's band below [`CUTOFF_HZ`] with the energy-matched low band of
//! the 16 kHz codec mix.
//!
//! A plain function seam (no weights), ported from the YuE-v1 inference path
//! (`yue/inference/infer.py` after stage 2, and `xcodec_mini_infer/post_process_audio.py`
//! `replace_low_freq_with_energy_matched`; pinned revisions in
//! `tests/fixtures/vocos_splice_reference.json`). Upstream runs it through files; [`post_process`]
//! runs the same arithmetic in memory:
//!
//! ```text
//!   codec stems (16 kHz) ─ clamp ±0.99 ─ Σ ─ recons mix ─ resample → 44.1 kHz ─ lowpass ─ × rms_b/rms_a ─┐
//!   Vocos stems (44.1 kHz) ─ Σ ─ limit ─ vocoder mix ─┬─ lowpass ─ rms_b                                  ├─ Σ ─ mix
//!                                                      └─ highpass ─────────────────────────────────────────┘
//!   Vocos stems ─ limit ─ vocals, instrumental
//! ```
//!
//! * `limit` is upstream `save_audio`: [`Limiter::Clamp`] (`clamp(±0.99)`, the default) or
//!   [`Limiter::Rescale`] (`× min(0.99 / peak, 1)`, upstream's `--rescale`). The 16 kHz stems are
//!   always clamped — upstream never passes `rescale` to that `save_audio` call.
//! * The resampler is torchaudio's `Resample(16000, 44100)` (windowed sinc, Hann window, width 6,
//!   rolloff 0.99; [`resample_sinc_hann`]); the filters are torchaudio's `lowpass_biquad` /
//!   `highpass_biquad` (Q 0.707) through `lfilter`, whose output is clamped to `±1` ([`biquad`]).
//!   The RMS is global over the whole clip.
//! * The final mix is not limited again (upstream writes it unlimited).
//!
//! The one deliberate deviation: upstream's intermediate files are lossy mp3s, which this in-memory
//! port does not reproduce (an encoder artifact, not part of the model).
//!
//! [`splice_stub`] is the end-to-end seam test's double.

use candle_audio::gen_core;

use crate::codec::SAMPLE_RATE as CODEC_RATE;
use crate::vocoder::SAMPLE_RATE as VOCODER_RATE;

/// The splice crossover.
pub const CUTOFF_HZ: f32 = 5_500.0;
/// `save_audio`'s output limit.
pub const LIMIT: f32 = 0.99;
/// The biquads' Q (torchaudio's default).
pub const BIQUAD_Q: f32 = 0.707;
/// `replace_low_freq_with_energy_matched`'s RMS epsilon.
const RMS_EPS: f64 = 1e-10;

/// A render's two stems at one sample rate.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrackPair {
    /// The vocal stem.
    pub vocals: Vec<f32>,
    /// The instrumental stem.
    pub instrumental: Vec<f32>,
}

/// The post-processed render at the vocoder rate.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SplicedMix {
    /// The final mix (after the low-band splice).
    pub mix: Vec<f32>,
    /// The limited vocal stem.
    pub vocals: Vec<f32>,
    /// The limited instrumental stem.
    pub instrumental: Vec<f32>,
}

/// The post-process seam: `(stems at the codec rate, stems at the vocoder rate) → spliced mix +
/// limited stems at the vocoder rate`. The rates are [`crate::codec::SAMPLE_RATE`] and
/// [`crate::vocoder::SAMPLE_RATE`].
pub type SpliceFn = fn(&TrackPair, &TrackPair) -> gen_core::Result<SplicedMix>;

/// Upstream `save_audio`'s limiter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limiter {
    /// `clamp(±0.99)` — upstream's default.
    Clamp,
    /// `× min(0.99 / peak, 1)` — upstream's `--rescale`.
    Rescale,
}

/// Production post-process: [`post_process`] with upstream's default [`Limiter::Clamp`].
pub fn splice(codec_rate: &TrackPair, vocoder_rate: &TrackPair) -> gen_core::Result<SplicedMix> {
    Ok(post_process(codec_rate, vocoder_rate, Limiter::Clamp))
}

/// The upstream post-process (module docs): limits the stems, builds both mixes and splices the
/// codec mix's energy-matched low band into the vocoder mix.
pub fn post_process(
    codec_rate: &TrackPair,
    vocoder_rate: &TrackPair,
    limiter: Limiter,
) -> SplicedMix {
    let recons_mix = sum_tracks(
        &limit(&codec_rate.vocals, Limiter::Clamp),
        &limit(&codec_rate.instrumental, Limiter::Clamp),
    );
    let vocoder_mix = limit(
        &sum_tracks(&vocoder_rate.instrumental, &vocoder_rate.vocals),
        limiter,
    );
    SplicedMix {
        mix: replace_low_freq_with_energy_matched(
            &recons_mix,
            CODEC_RATE,
            &vocoder_mix,
            VOCODER_RATE,
            CUTOFF_HZ,
        ),
        vocals: limit(&vocoder_rate.vocals, limiter),
        instrumental: limit(&vocoder_rate.instrumental, limiter),
    }
}

/// Upstream `save_audio`'s limiter arithmetic.
pub fn limit(wave: &[f32], limiter: Limiter) -> Vec<f32> {
    match limiter {
        Limiter::Clamp => wave.iter().map(|v| v.clamp(-LIMIT, LIMIT)).collect(),
        Limiter::Rescale => {
            let peak = wave.iter().fold(0f32, |m, v| m.max(v.abs()));
            // `min(0.99 / peak, 1)`: a silent clip (peak 0 → ∞) keeps unit gain.
            let gain = if peak > 0.0 {
                (LIMIT / peak).min(1.0)
            } else {
                1.0
            };
            wave.iter().map(|v| v * gain).collect()
        }
    }
}

/// `replace_low_freq_with_energy_matched`: resample `a` to `b`'s rate, low-pass both at `cutoff`,
/// scale `a`'s low band to `b`'s low-band RMS, and add `b`'s high band (high-passed at `cutoff`),
/// over the shorter length.
pub fn replace_low_freq_with_energy_matched(
    a: &[f32],
    rate_a: u32,
    b: &[f32],
    rate_b: u32,
    cutoff: f32,
) -> Vec<f32> {
    let a = resample_sinc_hann(a, rate_a, rate_b);
    let a_low = biquad(&a, BiquadKind::Lowpass, rate_b, cutoff);
    let b_low = biquad(b, BiquadKind::Lowpass, rate_b, cutoff);
    let scale = ((rms(&b_low) + RMS_EPS) / (rms(&a_low) + RMS_EPS)) as f32;
    let b_high = biquad(b, BiquadKind::Highpass, rate_b, cutoff);
    a_low
        .iter()
        .zip(&b_high)
        .map(|(lo, hi)| lo * scale + hi)
        .collect()
}

fn rms(x: &[f32]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / x.len() as f64).sqrt()
}

/// Sample-wise sum of two mono tracks, over the shorter length.
fn sum_tracks(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// torchaudio's `transforms.Resample(orig, new)` with its defaults (`sinc_interp_hann`,
/// `lowpass_filter_width = 6`, `rolloff = 0.99`): the polyphase windowed-sinc kernel built in f64
/// and cached as f32 (`_get_sinc_resample_kernel`), applied as a strided convolution over the
/// zero-padded input and truncated to `⌈new · len / orig⌉` samples (`_apply_sinc_resample_kernel`).
pub fn resample_sinc_hann(x: &[f32], orig: u32, new: u32) -> Vec<f32> {
    const WIDTH_ZEROS: f64 = 6.0;
    const ROLLOFF: f64 = 0.99;
    if orig == new || x.is_empty() {
        return x.to_vec();
    }
    let g = gcd(orig, new);
    let (o, n) = ((orig / g) as usize, (new / g) as usize);
    let base = o.min(n) as f64 * ROLLOFF;
    let width = (WIDTH_ZEROS * o as f64 / base).ceil() as usize;
    let taps = 2 * width + o;
    let scale = base / o as f64;
    // kernel[j * taps + k]: output phase j, input tap k.
    let kernel: Vec<f32> = (0..n)
        .flat_map(|j| {
            // Upstream's `arange(0, -new, -1) / new` is an integer tensor's true division, so the
            // phase offset is rounded to the default f32 before it meets the f64 tap grid.
            let tj = (-(j as f32) / n as f32) as f64;
            (0..taps).map(move |k| {
                let idx = (k as f64 - width as f64) / o as f64;
                let t = ((tj + idx) * base).clamp(-WIDTH_ZEROS, WIDTH_ZEROS);
                let window = (t * std::f64::consts::PI / WIDTH_ZEROS / 2.0).cos().powi(2);
                let t = t * std::f64::consts::PI;
                let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
                (sinc * window * scale) as f32
            })
        })
        .collect();
    let len = x.len();
    let mut padded = vec![0f32; width + len + width + o];
    padded[width..width + len].copy_from_slice(x);
    let frames = (padded.len() - taps) / o + 1;
    let target = (n * len).div_ceil(o);
    let mut out = Vec::with_capacity(frames * n);
    for f in 0..frames {
        let window = &padded[f * o..f * o + taps];
        for row in kernel.chunks_exact(taps) {
            out.push(row.iter().zip(window).map(|(k, v)| k * v).sum::<f32>());
        }
        if out.len() >= target {
            break;
        }
    }
    out.truncate(target);
    out
}

/// Which torchaudio biquad.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiquadKind {
    /// `lowpass_biquad`.
    Lowpass,
    /// `highpass_biquad`.
    Highpass,
}

/// torchaudio's `lowpass_biquad` / `highpass_biquad` (Q [`BIQUAD_Q`]) through `lfilter`: RBJ
/// coefficients computed in the waveform's f32, normalized by `a0`, the FIR part then the
/// direct-form recursion, the output clamped to `±1` (`lfilter`'s default `clamp=True`).
pub fn biquad(x: &[f32], kind: BiquadKind, rate: u32, cutoff: f32) -> Vec<f32> {
    let w0 = (std::f64::consts::TAU as f32) * cutoff / rate as f32;
    let (sin, cos) = (w0.sin(), w0.cos());
    let alpha = sin / 2.0 / BIQUAD_Q;
    let (b0, b1) = match kind {
        BiquadKind::Lowpass => ((1.0 - cos) / 2.0, 1.0 - cos),
        BiquadKind::Highpass => ((1.0 + cos) / 2.0, -1.0 - cos),
    };
    let (a0, a1, a2) = (1.0 + alpha, -2.0 * cos, 1.0 - alpha);
    let (b0, b1, b2) = (b0 / a0, b1 / a0, b0 / a0);
    let (a1, a2) = (a1 / a0, a2 / a0);
    let (mut x1, mut x2, mut y1, mut y2) = (0f32, 0f32, 0f32, 0f32);
    x.iter()
        .map(|&x0| {
            let fir = b2 * x2 + b1 * x1 + b0 * x0;
            let y0 = fir - a2 * y2 - a1 * y1;
            (x2, x1, y2, y1) = (x1, x0, y1, y0);
            y0.clamp(-1.0, 1.0)
        })
        .collect()
}

/// **Stub post-process** — the end-to-end seam test's double: the vocoder-rate stems summed into
/// the mix, the stems unchanged, no limiting or splice.
pub fn splice_stub(
    _codec_rate: &TrackPair,
    vocoder_rate: &TrackPair,
) -> gen_core::Result<SplicedMix> {
    Ok(SplicedMix {
        mix: sum_tracks(&vocoder_rate.vocals, &vocoder_rate.instrumental),
        vocals: vocoder_rate.vocals.clone(),
        instrumental: vocoder_rate.instrumental.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limiter_clamps_or_rescales_like_save_audio() {
        let w = [0.5, -1.98, 1.2, 0.0];
        assert_eq!(limit(&w, Limiter::Clamp), [0.5, -0.99, 0.99, 0.0]);
        let r = limit(&w, Limiter::Rescale);
        assert!((r[1] + 0.99).abs() < 1e-7 && (r[0] - 0.25).abs() < 1e-7);
        // Under the limit, rescale is unit gain; silence stays silence.
        assert_eq!(limit(&[0.1, -0.2], Limiter::Rescale), [0.1, -0.2]);
        assert_eq!(limit(&[0.0; 3], Limiter::Rescale), [0.0; 3]);
    }

    #[test]
    fn resampling_16k_to_44k1_emits_882_samples_per_codec_frame_and_keeps_a_tone() {
        let n = 3200;
        let tone: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin() * 0.5)
            .collect();
        let up = resample_sinc_hann(&tone, 16_000, 44_100);
        assert_eq!(up.len(), n / 320 * 882);
        // Mid-clip, the upsampled tone is the same sine sampled at 44.1 kHz.
        for (i, &got) in up.iter().enumerate().take(2100).skip(2000) {
            let want = (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44_100.0).sin() * 0.5;
            assert!(
                (got as f64 - want).abs() < 1e-2,
                "sample {i}: {got} vs {want}"
            );
        }
        assert_eq!(resample_sinc_hann(&tone, 16_000, 16_000), tone);
    }

    #[test]
    fn biquads_split_the_band_and_lfilter_clamps() {
        let sr = 44_100;
        let tone = |f: f32| -> Vec<f32> {
            (0..8820)
                .map(|i| (2.0 * std::f32::consts::PI * f * i as f32 / sr as f32).sin() * 0.8)
                .collect()
        };
        let peak = |v: &[f32]| v[4410..].iter().fold(0f32, |m, x| m.max(x.abs()));
        let low = tone(200.0);
        let high = tone(15_000.0);
        assert!(peak(&biquad(&low, BiquadKind::Lowpass, sr, CUTOFF_HZ)) > 0.75);
        assert!(peak(&biquad(&low, BiquadKind::Highpass, sr, CUTOFF_HZ)) < 0.01);
        assert!(peak(&biquad(&high, BiquadKind::Lowpass, sr, CUTOFF_HZ)) < 0.1);
        assert!(peak(&biquad(&high, BiquadKind::Highpass, sr, CUTOFF_HZ)) > 0.75);
        let loud: Vec<f32> = low.iter().map(|v| v * 3.0).collect();
        let out = biquad(&loud, BiquadKind::Lowpass, sr, CUTOFF_HZ);
        assert!(out.iter().all(|v| v.abs() <= 1.0) && peak(&out) == 1.0);
    }

    #[test]
    fn production_splice_produces_a_44k1_mix_and_limited_stems_instead_of_refusing() {
        let frames = 4;
        let codec = TrackPair {
            vocals: vec![0.5; frames * 320],
            instrumental: vec![0.7; frames * 320],
        };
        let vocoder = TrackPair {
            vocals: vec![0.6; frames * 882],
            instrumental: vec![1.3; frames * 882],
        };
        let out = splice(&codec, &vocoder).expect("production splice runs");
        assert_eq!(out.mix.len(), frames * 882);
        assert_eq!(out.vocals.len(), frames * 882);
        assert!(out.instrumental.iter().all(|&v| v == LIMIT));
        assert!(out.mix.iter().all(|v| v.is_finite()));
    }
}
