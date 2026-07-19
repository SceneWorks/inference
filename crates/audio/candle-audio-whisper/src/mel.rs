//! Whisper log-mel front-end (sc-12850): the 16 kHz mono PCM → 80-bin log-mel spectrogram step.
//!
//! The mel projection itself is [`candle_transformers::models::whisper::audio::pcm_to_mel`]
//! (reused wholesale — the epic DoD forbids re-porting what candle ships). This module owns only
//! the two host-side wrappers around it: loading the Slaney-normalized mel **filterbank** Whisper
//! was trained with, and resampling an arbitrary-rate mono clip down to Whisper's native 16 kHz.
//!
//! ## The filterbank
//!
//! Whisper does NOT use the HTK unnormalized triangular filterbank in [`candle_audio::mel`]; it
//! uses librosa's Slaney-normalized 80-bin filterbank baked into the reference implementation.
//! Those exact float weights ship as a raw little-endian `f32` blob (`melfilters.bytes`, vendored
//! from the candle whisper example, which extracted them from OpenAI's assets — Apache-2.0/MIT).
//! Reconstructing them from scratch would risk a subtle numeric drift the encoder was never trained
//! on, so the blob is embedded verbatim via [`include_bytes!`]. base/small are 80-mel checkpoints;
//! the 128-mel large-v3 filterbank is a deliberate non-goal for this first provider.

use candle_transformers::models::whisper::{self as whisper, audio, Config};

/// Whisper's native input sample rate (16 kHz mono) — [`whisper::SAMPLE_RATE`].
pub const SAMPLE_RATE: u32 = whisper::SAMPLE_RATE as u32;

/// The 80-bin Slaney-normalized mel filterbank Whisper base/small were trained with, embedded as a
/// raw little-endian `f32` blob (80 mels × 201 rfft bins × 4 bytes = 64 320 bytes).
const MEL_FILTERS_80: &[u8] = include_bytes!("melfilters.bytes");

/// Decode the embedded mel filterbank for a checkpoint with `num_mel_bins` mel bands. Only the
/// 80-bin filterbank is bundled (base/small); any other width is a typed error rather than a silent
/// mismatch.
pub fn mel_filters(num_mel_bins: usize) -> candle_audio::Result<Vec<f32>> {
    let bytes = match num_mel_bins {
        80 => MEL_FILTERS_80,
        other => {
            return Err(candle_audio::AudioError::Msg(format!(
                "whisper: no bundled mel filterbank for num_mel_bins={other} (only 80-bin \
                 base/small are supported by this provider)"
            )))
        }
    };
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Linear-interpolation resample of a mono `f32` clip to Whisper's 16 kHz. A no-op when already at
/// 16 kHz. Mirrors the audio family's `resample_to_native` idiom (OpenVoice `spectrogram.rs`) — a
/// polyphase resampler is unnecessary for the walking-skeleton ASR path (the model's own front-end
/// tolerates the mild linear-interp artifacts on speech-band content).
pub fn resample_to_16k(samples: &[f32], src_rate: u32) -> Vec<f32> {
    if src_rate == SAMPLE_RATE || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = SAMPLE_RATE as f64 / src_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let last = samples.len() - 1;
    let mut out = Vec::with_capacity(out_len);
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

/// Down-mix interleaved `channels`-channel PCM to mono by averaging (a no-op for mono).
pub fn to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    samples
        .chunks(ch)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect()
}

/// Prepare a raw [`gen_core::AudioTrack`](candle_audio::gen_core::AudioTrack)'s interleaved PCM for
/// the encoder: downmix to mono, resample to 16 kHz, and project to a flat log-mel vector of length
/// `num_mel_bins * n_frames`. Returns `(mel, n_frames)`.
pub fn track_to_mel(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    config: &Config,
) -> candle_audio::Result<(Vec<f32>, usize)> {
    let mono = to_mono(samples, channels);
    let pcm = resample_to_16k(&mono, sample_rate);
    let filters = mel_filters(config.num_mel_bins)?;
    let mel = audio::pcm_to_mel(config, &pcm, &filters);
    let n_frames = mel.len() / config.num_mel_bins;
    Ok((mel, n_frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filterbank_has_the_expected_shape() {
        let f = mel_filters(80).unwrap();
        assert_eq!(f.len(), 80 * 201, "80 mels × 201 rfft bins");
        assert!(f.iter().all(|w| w.is_finite()));
        assert!(f.iter().any(|&w| w > 0.0), "filterbank carries energy");
        // A non-80 width is refused rather than silently mismatched.
        assert!(mel_filters(128).is_err());
    }

    #[test]
    fn resample_is_identity_at_native_rate() {
        let s = vec![0.1, -0.2, 0.3, -0.4];
        assert_eq!(resample_to_16k(&s, SAMPLE_RATE), s);
    }

    #[test]
    fn resample_changes_length_by_the_rate_ratio() {
        // 24 kHz → 16 kHz shrinks by 2/3.
        let s = vec![0.0f32; 2400];
        let out = resample_to_16k(&s, 24_000);
        assert!(
            (out.len() as i64 - 1600).abs() <= 1,
            "expected ~1600 samples, got {}",
            out.len()
        );
    }

    #[test]
    fn to_mono_averages_channels() {
        assert_eq!(to_mono(&[1.0, 3.0, 2.0, 4.0], 2), vec![2.0, 3.0]);
        assert_eq!(to_mono(&[1.0, 2.0], 1), vec![1.0, 2.0]);
    }
}
